"""Observer and explicit diagnostic control called by exact-source insertions.

Device clones add work; host copies after forward/selection add synchronization
and timing overhead. No inference tensor is written and
no sampler is re-executed. Observer errors are recorded without reaching
inference; diagnostic-control errors stop the server. Every observer error
invalidates capture, even if HTTP succeeds.

The optional KV-recompute control is intentionally mutating and fail-closed. It
preempts one Adaptive request at the reviewed boundary, recomputes all but the
last conditioning token, then restores the scheduler budget so the captured
target row retains the original one-token ordinary-decode shape.
"""
from __future__ import annotations

import array
import math
import os
from pathlib import Path
import sys
import threading
import uuid

from .common import WINDOW, OBSERVATION_TARGET, encoded, event, load, need, save, sha, write_new

_local = threading.local()

DISPATCH_ARGUMENT_FIELDS = (
    'max_seqlen_q', 'seq_threshold_3D', 'num_par_softmax_segments', 'num_seqs',
    'softmax_segm_output_present', 'softmax_segm_max_present', 'softmax_segm_expsum_present',
    'is_batch_invariant', 'use_3d', 'grid', 'tile_size', 'BLOCK_M', 'BLOCK_Q',
    'block_size', 'num_segments')
DISPATCH_FIELDS = DISPATCH_ARGUMENT_FIELDS + ('natural_use_3d', 'force_3d', 'q3_correctness_guards')
PRODUCTION_DISPATCH_FIELDS = DISPATCH_ARGUMENT_FIELDS + ('short_query_3d', 'guards_active', 'force_3d')


def force_attention_3d(max_seqlen_q, num_seqs, natural_use_3d):
    """Return the boundary-only diagnostic override without touching CUDA state."""
    capture = getattr(_local, 'attention_capture', None)
    if capture is None:
        return False
    need(type(max_seqlen_q) is int and type(num_seqs) is int
         and type(natural_use_3d) is bool, 'invalid natural attention dispatch')
    force = capture['force_3d']
    if force:
        need(max_seqlen_q == 3 and num_seqs == 1 and natural_use_3d is False,
             'forced 3D reached unexpected attention shape/path')
    controls = capture['controls']
    controls.append(dict(ordinal=len(controls), natural_use_3d=natural_use_3d,
                         force_3d=force))
    return force


def attention_dispatch(*values):
    """Scalar-only fast path: no configuration I/O, tensors, or CUDA operations."""
    capture = getattr(_local, 'attention_capture', None)
    if capture is None:
        return
    need(len(values) == len(DISPATCH_ARGUMENT_FIELDS) + 1, 'attention dispatch field count')
    record = dict(zip(DISPATCH_ARGUMENT_FIELDS, values[:-1]))
    need(type(values[-1]) is bool, 'non-boolean attention correctness guards')
    record['q3_correctness_guards'] = values[-1]
    for name, value in record.items():
        if name == 'grid':
            need(type(value) is tuple and all(type(x) is int for x in value),
                 'non-scalar attention grid')
            record[name] = list(value)
        else:
            need(type(value) in (int, bool) or value is None,
                 'non-scalar attention metadata: ' + name)
    calls = capture['calls']
    controls = capture['controls']
    need(len(controls) == len(calls) + 1, 'attention control/launch mismatch')
    control = controls[len(calls)]
    need(control['ordinal'] == len(calls), 'attention control order mismatch')
    record.update(natural_use_3d=control['natural_use_3d'], force_3d=control['force_3d'])
    calls.append(dict(record, ordinal=len(calls), key=capture['key']))


def production_attention_dispatch(*values):
    """Observe already-decided production routing; never call diagnostic control."""
    capture = getattr(_local, 'production_attention_capture', None)
    if capture is None:
        return
    need(capture['force_3d'] is False, 'production attention intervention forbidden')
    need(len(values) == len(DISPATCH_ARGUMENT_FIELDS) + 2, 'production dispatch field count')
    record = dict(zip(PRODUCTION_DISPATCH_FIELDS, (*values, False)))
    for name, value in record.items():
        if name == 'grid':
            need(type(value) is tuple and all(type(x) is int for x in value), 'production grid metadata')
            record[name] = list(value)
        else:
            need(type(value) in (int, bool) or value is None, 'production dispatch non-scalar')
    need(type(record['short_query_3d']) is bool and type(record['guards_active']) is bool,
         'production guard metadata not boolean')
    calls = capture['calls']
    calls.append(dict(record, ordinal=len(calls), key=capture['key']))


def ints(tensor):
    return tensor.detach().cpu().tolist()


def device_snapshot(tensor):
    """Own device storage on the inference stream; never read it on the host."""
    return tensor.detach().clone()


def materialize_preparation(prep):
    """Called only from end(), after the forward and sampler have returned."""
    result = dict(prep)
    for name in ('gpu_num_computed_tokens_before', 'gpu_num_computed_tokens_after',
                 'valid_sampled_token_count_gpu', 'prev_positions', 'prev_num_draft_tokens',
                 'query_start_loc', 'seq_lens'):
        value = prep[name]
        result[name] = value if value is None or isinstance(value, list) else ints(value)
    result['active_query'] = list(map(list, zip(ints(result.pop('_positions')),
                                               ints(result.pop('_input_ids')))))
    result['groups'] = [dict(group, block_table=ints(group['block_table']),
                             slot_mapping=ints(group['slot_mapping'])) for group in prep['groups']]
    return result


def observation_target(config):
    # Missing field denotes a preserved legacy index-60 capture, never a new run.
    return config.get("observation_target", 60)


def correction_before(config, runner, scheduled):
    """Clone device counts before correction, without host synchronization."""
    _local.preparation = None
    _local.attention_capture = None
    _local.production_attention_capture = None
    ids = list(runner.input_batch.req_ids[:int(runner.input_batch.num_reqs)])
    if not ids or all(x.startswith(("_warmup_", "_v2_mixed_warmup")) for x in ids):
        return
    need(len(ids) == 1, "transition requires one active request")
    request_id = ids[0]
    identity = (config['directory'], request_id)
    if getattr(_local, 'transition_identity', None) != identity:
        _local.transition_identity = identity
        _local.transition_sequence = 0
        _local.transition_previous = None
        _local.transition_published = False
    _local.transition_sequence += 1
    cpu = runner.input_batch.num_computed_tokens_cpu[:1].tolist()
    position = len(runner.requests[request_id].prompt_token_ids) + OBSERVATION_TARGET - 1
    # CPU values can be optimistic by up to four drafts. Include a predecessor
    # independently of the GPU position/row mapping that this probe audits.
    if not position - 16 <= cpu[0] <= position + 8:
        return
    dispatch = getattr(scheduled, '_m203_kv_recompute_provenance', None)
    need(isinstance(dispatch, dict), "missing transition dispatch linkage")
    mapping = dict(runner.input_batch.prev_req_id_to_index or {})
    valid = runner.valid_sampled_token_count_gpu
    key = uuid.uuid4().hex
    _local.preparation = dict(
        key=key, request_id=request_id, scheduler_step=dispatch['scheduler_step'],
        forward_sequence=_local.transition_sequence,
        previous_key=(None if _local.transition_previous is None
                      else _local.transition_previous['key']),
        scheduled_tokens=scheduled.num_scheduled_tokens[request_id],
        scheduled_draft_count=len(scheduled.scheduled_spec_decode_tokens.get(request_id, [])),
        prev_req_id_to_index=mapping, prev_positions=runner.prev_positions.np[:1].tolist(),
        prev_num_draft_tokens=runner.prev_num_draft_tokens.np[:max(1, len(mapping))].tolist(),
        valid_sampled_token_count_gpu=None if valid is None else device_snapshot(valid),
        correction_enabled=bool(runner.use_async_spec_decode and valid is not None and mapping),
        cpu_num_computed_tokens=cpu,
        gpu_num_computed_tokens_before=device_snapshot(runner.num_computed_tokens[:max(1, len(mapping))]))


def prepared(config, runner, scheduled, input_ids, positions, slot_mappings):
    prep = getattr(_local, 'preparation', None)
    if prep is None:
        return
    request_id = prep['request_id']
    need(list(runner.input_batch.req_ids[:1]) == [request_id], "prepared request changed")
    count = prep['scheduled_tokens']
    need(scheduled.total_num_scheduled_tokens == count, "prepared batch changed")
    prep.update(gpu_num_computed_tokens_after=device_snapshot(runner.num_computed_tokens[:1]),
                query_start_loc=device_snapshot(runner.query_start_loc.gpu[:2]),
                seq_lens=device_snapshot(runner.seq_lens[:1]),
                _positions=device_snapshot(positions[:count]),
                _input_ids=device_snapshot(input_ids[:count]))
    if prep['correction_enabled']:
        # These are the actual GPU gather inputs copied by _prepare_inputs,
        # not merely the CPU arrays from which the copies were requested.
        prep['prev_positions'] = device_snapshot(runner.prev_positions.gpu[:1])
        prep['prev_num_draft_tokens'] = device_snapshot(runner.prev_num_draft_tokens.gpu[:max(1, len(prep['prev_req_id_to_index']))])
    groups = []
    for gid, table in enumerate(runner.input_batch.block_table.block_tables):
        width = int(table.num_blocks_per_row[0])
        groups.append(dict(request_id=request_id, row_index=0,
                           block_table=device_snapshot(table.get_device_tensor(1)[0, :width]),
                           slot_mapping=device_snapshot(slot_mappings[gid][:count]),
                           block_size=table.block_size, kv_cache_block_size=table.kv_cache_block_size,
                           blocks_per_kv_block=table.blocks_per_kv_block,
                           dcp_world_size=table.dcp_world_size, pcp_world_size=table.pcp_world_size))
    prep['groups'] = groups
    if config.get('production_attention_observation') is True:
        from .sources import profile_options
        profile_options(config)
        need(config['lane'] == 'adaptive', 'production dispatch requires Adaptive target')
        expected = runner.model_config.hf_config.num_hidden_layers
        need(type(expected) is int and expected > 0, 'missing production attention layer count')
        prep['production_attention_dispatch'] = dict(
            key=prep['key'], expected_calls=expected, force_3d=False, calls=[])
        _local.production_attention_capture = prep['production_attention_dispatch']
    if config.get('r15_attention_dispatch') is True:
        need(config['lane'] == 'adaptive', 'R15 requires Adaptive target')
        expected = runner.model_config.hf_config.num_hidden_layers
        need(type(expected) is int and expected > 0, 'missing target attention layer count')
        force = config.get('r16_force_attention_3d', False)
        need(type(force) is bool, 'invalid R16 force gate')
        if force:
            request = runner.requests[request_id]
            boundary_position = len(request.prompt_token_ids) + OBSERVATION_TARGET - 1
            previous = getattr(_local, 'transition_previous', None)
            # Async CPU token lists can contain -1 placeholders. Use the
            # completed observer record of the preceding ordinary selection.
            force = (boundary_position == 556 and previous is not None
                     and previous['key'] == prep['previous_key']
                     and previous['request_id'] == request_id
                     and previous['selected'] == [47742]
                     and previous['draft_count'] == 0
                     and len(previous['rows']) == 1
                     and previous['rows'][0]['generated_index'] == OBSERVATION_TARGET - 1
                     and previous['rows'][0]['position'] == 555
                     and prep['scheduled_tokens'] == 3
                     and prep['scheduled_draft_count'] == 2
                     and prep['cpu_num_computed_tokens'] == [boundary_position])
        prep['attention_dispatch'] = dict(key=prep['key'], expected_calls=expected,
                                          force_3d=force, controls=[], calls=[])
        _local.attention_capture = prep['attention_dispatch']


def finish_transition(config, record):
    previous = getattr(_local, 'transition_previous', None)
    boundary = any(row['generated_index'] == OBSERVATION_TARGET and row['row'] < len(record['selected'])
                   for row in record['rows'])
    if boundary:
        need(not _local.transition_published, "duplicate emitted boundary forward")
        need(previous is not None, "missing immediately preceding forward")
        if config.get('production_attention_observation') is True:
            dispatch = record['preparation']['production_attention_dispatch']
            need(dispatch['force_3d'] is False
                 and len(dispatch['calls']) == dispatch['expected_calls'] > 0,
                 'incomplete production attention observation')
            event(config['directory'], 'production_attention_dispatch',
                  execution_sha=config['execution_sha'], lane=config['lane'],
                  request_id=record['request_id'], key=record['key'],
                  scheduler_step=record['scheduler_step'], forward_sequence=record['forward_sequence'],
                  generated_index=OBSERVATION_TARGET,
                  **{name: dispatch[name] for name in ('expected_calls', 'calls')})
        if config.get('r15_attention_dispatch') is True:
            dispatch = record['preparation']['attention_dispatch']
            need(len(dispatch['calls']) == dispatch['expected_calls'] > 0,
                 'incomplete target attention dispatch collection')
            event(config['directory'], 'attention_dispatch',
                  execution_sha=config['execution_sha'], lane=config['lane'],
                  request_id=record['request_id'], key=record['key'],
                  scheduler_step=record['scheduler_step'],
                  forward_sequence=record['forward_sequence'], generated_index=OBSERVATION_TARGET,
                  **{name: dispatch[name] for name in ('expected_calls', 'calls')})
        for name, raw in _local.pending_vectors.items():
            write_new(Path(config['directory']) / name, raw)
        for item in (previous, record):
            save(Path(config['directory']) / (item['key'] + '.forward.json'), item)
        _local.transition_published = True
    # The predecessor needs preparation/selection metadata, not rejected rows'
    # logits. Keep no tensor views or vectors in the rolling predecessor.
    record['scores'] = {}
    if not boundary:
        record['preparation'].pop('attention_dispatch', None)
        record['preparation'].pop('production_attention_dispatch', None)
    _local.pending_vectors = {}
    _local.transition_previous = record


def map_rows(prompt_len, positions, indices, draft_count, target_rows, bonus_rows):
    """Map actual flattened query positions, not placeholder/output lengths."""
    need(type(prompt_len) is int and prompt_len > 0, "invalid prompt length")
    need(type(draft_count) is int and 0 <= draft_count <= 4 and len(indices) == draft_count + 1, "draft/row count mismatch")
    need(all(type(p) is int and p >= 0 for p in positions), 'invalid actual model positions')
    need(len(set(indices)) == len(indices) and all(type(i) is int and 0 <= i < len(positions) for i in indices),
         "invalid flattened logits indices")
    need(target_rows == list(range(draft_count)) and bonus_rows == [draft_count], "unexpected row-role mapping")
    generated = [positions[i] - prompt_len + 1 for i in indices]
    need(all(b == a + 1 for a, b in zip(generated, generated[1:])), "noncontiguous output positions")
    return [{"row": j, "flat_index": i, "position": positions[i], "generated_index": generated[j],
             "role": "bonus" if j == draft_count else "draft_verification"}
            for j, i in enumerate(indices)]


def valid_selected(ids):
    result, padded = [], False
    for token in ids:
        if token == -1:
            padded = True
        else:
            need(type(token) is int and token >= 0 and not padded, "invalid sampler padding")
            result.append(token)
    return result


def score_stats(raw):
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    need(len(values) >= 2 and not any(math.isnan(x) or x == math.inf for x in values), "invalid score vector")
    finite = [(x, i) for i, x in enumerate(values) if math.isfinite(x)]
    need(bool(finite), "all scores masked")
    best = sorted(finite, key=lambda x: (-x[0], x[1]))[:2]
    return {"top1_id": best[0][1], "top1": best[0][0],
            "top2_id": best[1][1] if len(best) == 2 else None,
            "top2": best[1][0] if len(best) == 2 else None,
            "margin": best[0][0] - best[1][0] if len(best) == 2 else None,
            "vocabulary_size": len(values), "masked_count": len(values) - len(finite)}


def snapshot(tensor, root, name, pending=None):
    dtype = str(tensor.dtype)
    # tobytes owns its bytes even when .float()/.cpu() returned a view.
    raw = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=True).tobytes()
    stats = score_stats(raw)
    if pending is None:
        write_new(Path(root) / name, raw)
    else:
        pending[name] = raw
    return {"file": name, "sha256": sha(raw), "storage_dtype": "float32-le",
            "source_dtype": dtype, **stats}


def kv_recompute_target(config):
    """Return the reviewed lane-local KV control boundary, if enabled."""
    lane = config.get("lane")
    need(lane in ("vanilla", "adaptive"), "unknown diagnostic lane")
    return config.get("diagnostic_control", {}).get(
        f"{lane}_kv_recompute_output_count")


def r14_before_schedule(scheduler):
    """Suppress current drafts only at a CPU-proven, single-token boundary.

    v0.30.0 schedules W + P - C tokens, W = L + O + D. Decode maintains
    C = L + O + P - 1. For P == I == 0 or 1, pending output is absent or
    exactly one non-speculative token; hence the next generated index O + P
    is exact. Clearing D then yields one query token, without changing C/P/I
    or the previous GPU speculative correction inputs. Larger pending spans
    can still reject drafts, so their optimistic O + P cannot identify 368.
    """
    need(os.environ.get('M203_R14_SINGLE_TOKEN_BOUNDARY') == str(OBSERVATION_TARGET),
         'R14 gate must explicitly name boundary 368')
    config_path = os.environ.get('M203_TARGETED_CONFIG')
    need(bool(config_path), 'R14 requires the existing targeted capture config')
    config = load(config_path)
    need(config.get('lane') == 'adaptive' and type(config.get('r14_single_token_boundary')) is int
         and config['r14_single_token_boundary'] == OBSERVATION_TARGET
         and observation_target(config) == OBSERVATION_TARGET, 'R14 config/gate mismatch')
    running = [r for r in scheduler.running
               if not r.request_id.startswith(('_warmup_', '_v2_mixed_warmup'))]
    if not running:
        return
    need(len(running) == 1, 'R14 requires exactly one target request')
    request = running[0]
    need(scheduler.requests.get(request.request_id) is request, 'R14 request mapping mismatch')
    state = getattr(scheduler, '_m203_r14', None)
    if state is None:
        state = dict(request_id=request.request_id, fired=False)
        event(config['directory'], 'r14_control_enabled', control_enabled=True,
              request_id=request.request_id, target_generated_index=OBSERVATION_TARGET,
              execution_sha=config['execution_sha'])
        scheduler._m203_r14 = state
    need(state['request_id'] == request.request_id, 'R14 target request changed')
    if state['fired']:
        return
    counters = {name: getattr(request, name, None) for name in (
        'num_prompt_tokens', 'num_output_tokens', 'num_output_placeholders',
        'num_computed_tokens', 'num_in_flight_tokens', 'num_stale_output_tokens')}
    need(all(type(v) is int and v >= 0 for v in counters.values()), 'R14 malformed accounting')
    L, O, P, C, I = (counters[k] for k in ('num_prompt_tokens', 'num_output_tokens',
                                        'num_output_placeholders', 'num_computed_tokens', 'num_in_flight_tokens'))
    need(type(request.is_prefill_chunk) is bool, 'R14 unknown prefill state')
    if request.is_prefill_chunk or C < L:
        need(O + P < OBSERVATION_TARGET, 'R14 boundary cannot be proven during refill/prefill')
        return  # Includes the unchanged reconstruction/refill at output 60.
    drafts = request.spec_token_ids
    need(isinstance(drafts, list) and len(drafts) <= 4
         and all(type(t) is int and t >= -1 for t in drafts), 'R14 malformed drafts')
    need(request.num_tokens == L + O and request.num_tokens_with_spec == L + O + len(drafts)
         and len(request.output_token_ids) == O and C == L + O + P - 1,
         'R14 decode accounting invariant not proven')
    if O + P < OBSERVATION_TARGET:
        return
    need(O + P == OBSERVATION_TARGET and P == I and P <= 1
         and counters['num_stale_output_tokens'] == 0 and str(request.status) == 'RUNNING'
         and scheduler.num_sampled_tokens_per_step == 1,
         'R14 boundary ambiguous/missed or pending speculative accounting')
    need(bool(drafts), 'R14 boundary already ordinary; intervention not reproduced')
    step = scheduler.current_step
    need(type(step) is int and step >= 0, 'R14 invalid scheduler step')
    # Publish before mutation; I/O failure must stop without consuming drafts.
    event(config['directory'], 'r14_current_drafts_suppressed', control_enabled=True,
          request_id=request.request_id, target_generated_index=OBSERVATION_TARGET,
          pre_control_draft_count=len(drafts), post_control_draft_count=0,
          num_tokens=request.num_tokens, num_tokens_with_spec=request.num_tokens_with_spec,
          **counters, scheduler_step=step + 1, fire_count=1,
          execution_sha=config['execution_sha'])
    request.spec_token_ids = []  # Never mutate the shared placeholder list.
    state['fired'] = True


def skip_controlled_kv_refill(config, request_id, scheduled, query, rows, draft_count):
    """Skip only the proven all-but-last KV-recompute refill forward."""
    need(kv_recompute_target(config) == 60, "discarded sampler output")
    event_paths = sorted((Path(config["directory"]) / "events").glob("*.json"))
    events = [(p, load(p)) for p in event_paths]
    started = [(p, value) for p, value in events
               if value.get("kind") == "kv_recompute_started"]
    skipped = [value for _, value in events
               if value.get("kind") == "kv_recompute_refill_skipped"]
    need(len(started) == 1 and not skipped,
         "discarded refill provenance missing/ambiguous")
    started_path, proof = started[0]
    recomputed = proof.get("recomputed_tokens")
    need(proof.get("request_id") == request_id
         and proof.get("output_count") == 60
         and type(proof.get("total_tokens")) is int
         and recomputed == proof["total_tokens"] - 1,
         "discarded refill control proof invalid")
    scheduled_tokens = scheduled.num_scheduled_tokens.get(request_id)
    need(scheduled.total_num_scheduled_tokens == scheduled_tokens == recomputed
         and draft_count == 0 and len(rows) == 1
         and rows[0]["role"] == "ordinary"
         and rows[0]["generated_index"] == 59
         and len(query) == recomputed
         and [p for p, _ in query] == list(range(recomputed)),
         "discarded refill shape invalid")
    event(config["directory"], "kv_recompute_refill_skipped",
          request_id=request_id, output_count=60,
          total_tokens=proof["total_tokens"], recomputed_tokens=recomputed,
          scheduled_tokens=scheduled_tokens,
          active_query_token_count=len(query), query_start_position=query[0][0],
          query_end_position=query[-1][0], row_generated_index=59,
          decoding_path="ordinary",
          started_event_sha256=sha(started_path.read_bytes()))


def begin(config, runner, scheduled, logits, spec, grammar):
    _local.attention_capture = None  # Target forward ended; exclude sampling/drafter calls.
    _local.production_attention_capture = None
    _local.ctx = None
    _local.pending_vectors = {}
    batch = runner.input_batch
    ids = list(batch.req_ids[:int(batch.num_reqs)])
    if ids and all(x.startswith(("_warmup_", "_v2_mixed_warmup")) for x in ids):
        return
    need(len(ids) == 1, "expected one real request")
    need(grammar is None and not getattr(batch, "has_structured_output_reqs", False), "grammar unsupported")
    need(not runner.uses_mrope and not runner.enable_prompt_embeds, "nonstandard positions unsupported")
    need(not runner.cache_config.enable_prefix_caching, "prefix caching enabled")
    pc = runner.vllm_config.parallel_config
    need(all(int(getattr(pc, p, 1)) == 1 for p in
             ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size")),
         "parallel configuration unsupported")
    req = runner.requests[ids[0]]
    prompt = list(req.prompt_token_ids)
    need(prompt and all(type(i) is int and i >= 0 for i in prompt), "invalid runtime prompt IDs")
    lo, hi = map(int, batch_bounds(runner))
    need(lo == 0 and hi == scheduled.total_num_scheduled_tokens, "unexpected query bounds")
    positions = ints(runner.positions[lo:hi])
    need(logits is not None and logits.ndim == 2, "missing raw target logits")
    if spec is None:
        n, indices, targets, bonus = 0, [hi - 1], [], [0]
    else:
        need(len(spec.num_draft_tokens) == 1, "ambiguous draft metadata")
        n = int(spec.num_draft_tokens[0])
        indices, targets, bonus = ints(spec.logits_indices), ints(spec.target_logits_indices), ints(spec.bonus_logits_indices)
        need(ints(spec.cu_num_sampled_tokens) == [n + 1], "sampled cumulative row mismatch")
    rows = map_rows(len(prompt), positions, indices, n, targets, bonus)
    if spec is None:
        rows[0]["role"] = "ordinary"
    need(logits.shape[0] == len(rows), "logits/position row count mismatch")
    transition = observation_target(config) == OBSERVATION_TARGET
    prep = getattr(_local, 'preparation', None) if transition else None
    if transition and prep is None:
        # Reconstruction remains at 60. Its discarded refill still needs the
        # existing durable control proof, without retaining an index-60 forward.
        if (rows[0]['generated_index'] == 59
                and bool(getattr(getattr(runner, 'discard_request_mask', None), 'np', [False])[0])):
            query = list(zip(positions, ints(runner.input_ids.gpu[lo:hi])))
            skip_controlled_kv_refill(config, ids[0], scheduled, query, rows, n)
        return
    if not transition and not any(r["generated_index"] in WINDOW for r in rows):
        return
    query = list(zip(positions, ints(runner.input_ids.gpu[lo:hi])))
    need(len({p for p, _ in query}) == len(query) and all(p >= 0 and t >= 0 for p, t in query), "invalid active query")
    if bool(getattr(getattr(runner, "discard_request_mask", None), "np", [False])[0]):
        skip_controlled_kv_refill(config, ids[0], scheduled, query, rows, n)
        return
    dispatch = getattr(scheduled, "_m203_kv_recompute_provenance", None)
    key = prep['key'] if transition else uuid.uuid4().hex
    capture = {"schema": "m203.targeted.forward.v1", "key": key, "lane": config["lane"],
               "execution_sha": config["execution_sha"], "request_id": ids[0], "prompt_token_ids": prompt,
               "active_query": query, "rows": rows, "draft_count": n,
               "decoding_path": 'ordinary' if spec is None else 'speculative',
               "spec_target_rows": targets, "spec_bonus_rows": bonus,
               "async_scheduling": bool(runner.use_async_scheduling), "prefix_cache_enabled": False,
               "same_internal_state_proven": False, "runtime_probe_changes_timing": True,
               "kv_recompute_scheduler_provenance": dispatch,
               "scores": {}, "selected": None}
    if transition:
        capture.update(schema="m203.targeted.transition.v1", preparation=prep,
                       forward_sequence=prep['forward_sequence'], previous_key=prep['previous_key'],
                       scheduler_step=prep['scheduler_step'])
    for r in rows:
        if (r['generated_index'] == OBSERVATION_TARGET if transition else r["generated_index"] in WINDOW):
            j = r["row"]
            capture["scores"][str(j)] = {"raw": snapshot(
                logits[j], config["directory"], f"{key}-r{j}-raw.f32",
                _local.pending_vectors if transition else None)}
            capture['scores'][str(j)]['raw'].update(capture_point='runner_pre_grammar', source_row=j)
    _local.ctx = (config, capture)


def batch_bounds(runner):
    return runner.query_start_loc.np[:2]


def processed(stage, logits, metadata, spec=None):
    ctx = getattr(_local, "ctx", None)
    if ctx is None:
        return
    config, record = ctx
    need(metadata.all_greedy is True, "non-greedy sampling unsupported")
    n = record["draft_count"]
    indices = list(range(n)) if stage == "verify_scores" else [n]
    need(logits.ndim == 2 and logits.shape[0] == len(indices), "processed score mapping mismatch")
    if spec is not None:
        need(ints(spec.target_logits_indices) == indices, "processed target roles mismatch")
    for k, j in enumerate(indices):
        if str(j) in record["scores"]:
            need("processed" not in record["scores"][str(j)], "duplicate processed score capture")
            record["scores"][str(j)]["processed"] = snapshot(logits[k], config["directory"],
                                                             f'{record["key"]}-r{j}-processed.f32',
                                                             _local.pending_vectors if observation_target(config) == OBSERVATION_TARGET else None)
            record['scores'][str(j)]['processed'].update(
                capture_point='rejection_pre_acceptance' if stage == 'verify_scores' else 'sampler_pre_greedy',
                source_row=k)


def end(config, runner, output):
    ctx = getattr(_local, "ctx", None)
    _local.ctx = None
    if ctx is None:
        return
    _, record = ctx
    need(output.sampled_token_ids.ndim == 2 and output.sampled_token_ids.shape[0] == 1, "ambiguous sampler output")
    record["selected"] = valid_selected(ints(output.sampled_token_ids[0]))
    need(0 < len(record["selected"]) <= len(record["rows"]), "selected row count mismatch")
    need(all(set(s) == {"raw", "processed"} for s in record["scores"].values()), "incomplete raw/processed capture")
    record["accepted_drafts"] = len(record["selected"]) - 1
    record["rejected_drafts"] = record["draft_count"] - record["accepted_drafts"]
    need(record["rejected_drafts"] >= 0, "invalid draft accounting")
    if observation_target(config) == OBSERVATION_TARGET:
        record['preparation'] = materialize_preparation(record['preparation'])
        _local.preparation = None
        need(record['preparation']['active_query'] == list(map(list, record['active_query'])),
             'prepared query changed before logits')
        finish_transition(config, record)
    else:
        save(Path(config["directory"]) / (record["key"] + ".forward.json"), record)


def accounting(config, stage, request, output, scheduled, stale):
    before = len(request._output_token_ids)
    transition = observation_target(config) == OBSERVATION_TARGET
    if not ((OBSERVATION_TARGET - 12 <= before <= OBSERVATION_TARGET + 8) if transition else (52 <= before <= 70)):
        return
    need(type(stale) is bool, "invalid stale flag")
    linkage = {}
    if transition:
        dispatch = getattr(scheduled, '_m203_kv_recompute_provenance', None)
        need(isinstance(dispatch, dict), 'accounting dispatch linkage missing')
        linkage['scheduler_step'] = dispatch['scheduler_step']
    event(config["directory"], stage, **linkage, request_id=request.request_id,
          output_count=before, output_token_ids=list(output), stale=stale,
          num_computed_tokens=request.num_computed_tokens,
          num_in_flight_tokens=request.num_in_flight_tokens,
          num_output_placeholders=request.num_output_placeholders,
          invalid_spec_tokens=(getattr(scheduled, 'num_invalid_spec_tokens', None) or {}).get(request.request_id, 0),
          scheduled_tokens=scheduled.num_scheduled_tokens.get(request.request_id),
          scheduled_spec_token_ids=list(scheduled.scheduled_spec_decode_tokens.get(request.request_id, [])))


def kv_recompute_before_schedule(config, scheduler):
    target = kv_recompute_target(config)
    if target is None:
        return
    need(target == 60, "unreviewed KV recompute boundary")

    state = getattr(scheduler, "_m203_kv_recompute", None)
    if state is not None:
        need(state["phase"] in ("refill", "ready", "complete"),
             "KV recompute lifecycle invalid")
        if state["phase"] == "complete":
            return
        if state["request_id"] not in scheduler.requests:
            need(state["phase"] == "ready",
                 "KV recompute request disappeared")
            state["phase"] = "complete"
            return
        request = scheduler.requests[state["request_id"]]
        if state["phase"] == "refill":
            need(request.num_computed_tokens == state["recomputed_tokens"],
                 "KV recompute did not stop before boundary token")
            scheduler.max_num_scheduled_tokens = state["original_budget"]
            ready = {
                "kind": "kv_recompute_ready",
                "request_id": request.request_id,
                "output_count": len(request._output_token_ids),
                "total_tokens": state["total_tokens"],
                "recomputed_tokens": state["recomputed_tokens"],
                "num_computed_tokens": request.num_computed_tokens,
                "budget_restored": (scheduler.max_num_scheduled_tokens
                                    == state["original_budget"]),
            }
            raw = encoded(ready)
            write_new(Path(config["directory"]) / "events" /
                      (uuid.uuid4().hex + ".json"), raw)
            state["ready"] = {
                "kv_recompute_ready_event_sha256": sha(raw),
                **{key: value for key, value in ready.items()
                   if key != "kind"},
            }
            state["phase"] = "ready"
        return

    matches = [request for request in scheduler.running
               if len(request._output_token_ids) == target]
    if not matches:
        return
    need(len(scheduler.running) == len(matches) == 1,
         "KV recompute requires one boundary request")
    request = matches[0]
    total = request.num_prompt_tokens + len(request._output_token_ids)
    recomputed = total - 1
    original_budget = scheduler.max_num_scheduled_tokens
    need(recomputed > 0 and original_budget >= recomputed
         and scheduler.scheduler_config.max_num_batched_tokens >= recomputed,
         "KV recompute chunk cannot preserve ordinary boundary shape")
    scheduler.max_num_scheduled_tokens = recomputed
    event(config["directory"], "kv_recompute_started",
          request_id=request.request_id, output_count=target,
          total_tokens=total, recomputed_tokens=recomputed,
          original_budget=original_budget)
    need(scheduler.reset_prefix_cache(reset_running_requests=True) is True,
         "KV recompute reset failed")
    need(request.num_computed_tokens == 0 and not request.spec_token_ids,
         "KV recompute did not clear target state")
    scheduler._m203_kv_recompute = {
        "phase": "refill", "request_id": request.request_id,
        "control_epoch": 1,
        "total_tokens": total, "recomputed_tokens": recomputed,
        "original_budget": original_budget,
    }


def kv_recompute_after_schedule(config, scheduler, scheduled):
    """Stamp the batch at dispatch; later callback timing cannot change it."""
    target = kv_recompute_target(config)
    if target is None:
        return
    need(target == 60, "unreviewed KV recompute boundary")
    state = getattr(scheduler, "_m203_kv_recompute", None)
    provenance = {
        "state": "PRE_READY",
        "scheduler_step": scheduler.current_step,
        "control_epoch": None if state is None else state["control_epoch"],
    }
    if state is not None and state["phase"] in ("ready", "complete"):
        ready = state.get("ready")
        need(isinstance(ready, dict), "ready scheduler provenance missing")
        provenance = {
            "state": "POST_READY",
            "scheduler_step": scheduler.current_step,
            "control_epoch": state["control_epoch"],
            **ready,
        }
    scheduled._m203_kv_recompute_provenance = provenance


def control(stage, *args):
    if stage == 'force_attention_3d':
        return force_attention_3d(*args)
    config_path = os.environ.get("M203_TARGETED_CONFIG")
    if not config_path:
        return
    config = load(config_path)
    if kv_recompute_target(config) is None:
        return
    need(stage in ("before_schedule", "after_schedule"),
         "unknown diagnostic control stage")
    if stage == "before_schedule":
        kv_recompute_before_schedule(config, *args)
    else:
        kv_recompute_after_schedule(config, *args)


def observe(stage, *args):
    if stage == 'production_attention_dispatch' and getattr(_local, 'production_attention_capture', None) is None:
        return
    if stage == 'attention_dispatch' and getattr(_local, 'attention_capture', None) is None:
        return
    config_path = os.environ.get("M203_TARGETED_CONFIG")
    if not config_path:
        return
    try:
        if stage == 'production_attention_dispatch':
            production_attention_dispatch(*args)
            return
        if stage == 'attention_dispatch':
            attention_dispatch(*args)
            return
        config = load(config_path)
        if stage in ('correction_before', 'prepared'):
            if observation_target(config) == OBSERVATION_TARGET:
                (correction_before if stage == 'correction_before' else prepared)(config, *args)
        elif stage == "begin":
            begin(config, *args)
        elif stage == "end":
            end(config, *args)
        elif stage in ("sample_scores", "verify_scores"):
            processed(stage, *args)
        elif stage in ("account_before", "account_after"):
            accounting(config, stage, *args)
    except Exception as exc:
        _local.ctx = None
        _local.attention_capture = None
        _local.production_attention_capture = None
        _local.preparation = None
        try:
            config = load(config_path)
            event(config["directory"], "observer_error", stage=stage,
                  error=f"{type(exc).__name__}: {exc}")
        except Exception:
            # Missing error file must still fail capture completeness checks.
            print("M203_TARGETED_OBSERVER_ERROR_UNSAVED", file=sys.stderr, flush=True)
