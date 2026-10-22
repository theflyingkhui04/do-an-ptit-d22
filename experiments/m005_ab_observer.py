"""Read-only observer for M005/index162 only. No control or attention replay."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

from experiments.protocol_b import fingerprint, require, write_new, load_json
from experiments.protocol_b_boundary_observer import map_rows, selected_tokens, ints, tensor_identity

SAMPLE = 'm0_04_005'
INDEX, PROMPT, POSITION = 162, 1522, 1683
LAYER = 'model.layers.0.self_attn.attn'
_local = threading.local()
_state = {}


def bitwise(raw, shape, dtype):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), hex=raw.hex(),
                byte_count=len(raw), shape=list(shape), dtype=str(dtype), byteorder=sys.byteorder)


def tensor_bytes(tensor, *, include_bytes=True):
    import torch
    require(not torch.cuda.is_current_stream_capturing() if tensor.device.type == 'cuda' else True,
            'Tensor observation during CUDA capture')
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    result = bitwise(raw, tensor.shape, tensor.dtype)
    result['strides'] = list(tensor.stride())
    if not include_bytes:
        result.pop('hex')
    return result


def logical_kv(key, value, table, length):
    """Hash original bits in logical token order, excluding allocation padding."""
    require(key.ndim == value.ndim == 4 and key.shape == value.shape
            and list(key.shape[2:]) == [4, 128] and str(key.dtype) == str(value.dtype) == 'torch.float16',
            'Unsupported logical cache view')
    block_size = key.shape[1]
    require(0 < length <= 4096 and len(table) >= (length + block_size - 1) // block_size,
            'Incomplete block table')
    blocks = table[:(length + block_size - 1) // block_size]
    require(all(type(b) is int and 0 <= b < key.shape[0] for b in blocks)
            and len(blocks) == len(set(blocks)), 'Invalid/aliased logical blocks')
    # Bounded one-component-at-a-time host read; no CUDA allocation or KV write.
    def component(cache):
        parts = [cache[b, :min(block_size, length - i * block_size)].detach().cpu().contiguous()
                 for i, b in enumerate(blocks)]
        import torch
        logical = torch.cat(parts, dim=0)
        return tensor_bytes(logical, include_bytes=False)
    return dict(K=component(key), V=component(value), logical_length=length,
                kernel_block_size=block_size, key_layout=list(key.stride()), value_layout=list(value.stride()))


def controller_state(controller):
    # No select_budget/update calls, no stateful diagnostic queries.
    fields = ('window_size', 'ema_beta', 'alpha_ema', 'initial_alpha', 'fallback_threshold',
              'high_threshold', 'candidate_budgets', 'min_free_vram_mb', 'step_count',
              'probe_interval', 'probe_k', 'consecutive_fallback_steps', 'kv_cache_high_pct',
              'kv_cache_recover_pct', 'memory_cooldown_steps', 'enable_vram_guard',
              '_memory_pressure_active', '_memory_pressure_steps')
    values = {k: getattr(controller, k) for k in fields}
    values['window'] = list(controller._window)
    values['planned_history'] = [v['k'] for v in controller._history]
    values['cost_profile'] = [[list(k), v] for k, v in controller.cost_profile.matrix.items()]
    # Hex float values retain signed zero/exact controller pre-state.
    def freeze(v):
        if isinstance(v, float): return {'float_hex': v.hex()}
        if isinstance(v, dict): return {k: freeze(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)): return [freeze(w) for w in v]
        return v
    return freeze(values)


def emit(config, kind, **fields):
    state = _state[str(Path(config['directory']).resolve())]
    state['event_seq'] += 1
    record = dict(kind=kind, event_seq=state['event_seq'], pid=os.getpid(), config_sha256=fingerprint(config),
                  profile=config['profile'], **fields)
    write_new(Path(config['directory']) / 'observer' / ('%06d.json' % state['event_seq']), record)


def selected_request(config, request):
    return list(request.prompt_token_ids or []) == config['prompt_token_ids']


def process_chain():
    """EngineCore is a descendant, not necessarily the HTTP port-owner PID."""
    chain, pid = [], os.getpid()
    for _ in range(32):
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        chain.append(dict(pid=pid, start_ticks=stat[19],
                          cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
        parent = int(stat[1])
        if parent == 0: return chain
        require(parent != pid and parent not in [p['pid'] for p in chain], 'Process lineage loop')
        pid = parent
    raise ValueError('Process ancestry exceeded bound')


def initialize(config):
    from experiments.m005_ab_sources import authenticate, PROFILES
    require(config['sample_id'] == SAMPLE and config['index'] == INDEX
            and len(config['prompt_token_ids']) == PROMPT
            and config['profile'] == PROFILES[config['condition']], 'Wrong bounded observer scope')
    require(os.environ.get('PROTOCOL_B_BOUNDARY_CONFIG') is None
            and os.environ.get('M203_P177_PRODUCER_K0') != '1'
            and os.environ.get('M203_P177_ATTENTION_SUBSTITUTION') != '1', 'Historical control enabled')
    manifest = authenticate(config['source_directory'])
    require(manifest['profile'] == config['profile'], 'Observer/source profile mismatch')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == config['boot_id']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == config['gpu_uuid'], 'Observer physical session drift')
    require(Path(__file__).resolve() == Path(manifest['project']) / 'experiments/m005_ab_observer.py',
            'Observer imported from another checkout')
    import vllm
    require(Path(vllm.__file__).resolve().parent == Path(manifest['upstream']), 'Wrong runtime vLLM package')
    return dict(event_seq=0, steps={}, producers={}, origin=None, failed=False, ledger=[], closed=False)


def schedule(config, state, scheduler, scheduled):
    matches = [r for rid in scheduled.num_scheduled_tokens
               if (r := scheduler.requests.get(rid)) is not None and selected_request(config, r)]
    if not matches or state['closed']: return
    require(len(matches) == len(scheduled.num_scheduled_tokens) == 1, 'Ambiguous selected batch')
    request = matches[0]
    rid = request.request_id
    controller = scheduler._planner.adapter._active_sessions.get(rid)
    # Adapter sessions store controllers; never create one from the observer.
    entry = dict(schedule_seq=scheduler.current_step, request_id=rid,
                 planned_k=scheduler._plan.per_request_k.get(rid), batch_k=scheduler._plan.batch_k,
                 future_k=scheduled.num_spec_tokens_to_schedule,
                 controller=None if controller is None else controller_state(controller), producer_schedule_seq=state['origin'],
                 scheduled_tokens=scheduled.num_scheduled_tokens[rid],
                 slots=list(scheduled.scheduled_spec_decode_tokens.get(rid, [])),
                 invalid=(scheduled.num_invalid_spec_tokens or {}).get(rid, 0),
                 computed=request.num_computed_tokens, in_flight=request.num_in_flight_tokens,
                 placeholders=request.num_output_placeholders,
                 committed_count=len(request._output_token_ids))
    require(id(scheduled) not in state['steps'], 'Duplicate scheduler object')
    state['steps'][id(scheduled)] = entry
    # Metadata only; compact evidence of past decisions, no per-step tensor capture.
    state['ledger'].append((entry['planned_k'], entry['batch_k'], entry['future_k']))
    entry['past_budget_sha256'] = fingerprint(state['ledger'])
    state['origin'] = entry['schedule_seq']
    require(len(state['steps']) <= 32, 'Unresolved scheduler lineage exceeded bound')


def prepare(config, state, runner, scheduled, input_ids, positions, metadata, slots, mode):
    _local.attention = None
    ids = list(runner.input_batch.req_ids[:runner.input_batch.num_reqs])
    if not ids or not any(selected_request(config, runner.requests[rid]) for rid in ids): return
    require(len(ids) == 1 and len(config['prompt_token_ids']) == PROMPT, 'Wrong selected runner batch')
    entry = state['steps'].get(id(scheduled))
    require(entry is not None and entry['request_id'] == ids[0], 'Lost schedule/runner binding')
    entry['runner'] = id(runner)
    count = scheduled.total_num_scheduled_tokens
    # No prefill/whole-trajectory tensor reads. The bounded metadata window
    # covers two maximum-width speculative attempts around the output boundary.
    if count > 5 or not INDEX - 10 <= entry['committed_count'] <= INDEX + 5: return
    pos = ints(positions[:count])
    require(pos == list(range(pos[0], pos[-1] + 1)), 'Noncontiguous positions')
    entry['positions'] = pos
    if POSITION not in pos: return
    require(1 <= count <= 5 and 'PIECEWISE' in str(mode)
            and not runner.uses_mrope and not runner.enable_prompt_embeds, 'Unsupported boundary runner/mode')
    parallel = runner.vllm_config.parallel_config
    require(all(getattr(parallel, k, 1) == 1 for k in
                ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size', 'decode_context_parallel_size')),
            'Unsupported parallel runner')
    require(list(runner.query_start_loc.np[:2]) == [0, count], 'Wrong query bounds')
    entry.update(input_ids=ints(input_ids[:count]), M=count, graph_mode=str(mode), observed=True)
    context = runner.vllm_config.compilation_config.static_forward_context
    layer = context[LAYER]
    pending = dict(entry=entry, runner=runner, scheduled=scheduled, layer=layer,
                   metadata=metadata[LAYER], slots=slots[LAYER], sample_id=SAMPLE)
    _local.attention = pending
    entry['pre_count'] = entry['post_count'] = entry['dispatch_count'] = 0


def attention_pre(config, state, impl, layer, query, key, value, metadata, output):
    pending = getattr(_local, 'attention', None)
    if pending is None or layer is not pending['layer']: return
    require(impl is layer.impl and metadata is pending['metadata'], 'Layer0 metadata ownership differs')
    entry = pending['entry']
    require(entry['pre_count'] == 0 and list(query.shape) == [entry['M'], 28, 128], 'Duplicate/wrong layer0 query')
    starts, lengths = ints(metadata.query_start_loc), ints(metadata.seq_lens)
    require(starts == [0, entry['M']] and len(lengths) == 1
            and lengths[0] - entry['M'] == entry['positions'][0], 'Position/attention metadata mismatch')
    table = ints(metadata.block_table[0])
    slots = ints(pending['slots'])
    require(slots == [table[p // key.shape[1]] * key.shape[1] + p % key.shape[1] for p in entry['positions']],
            'Write-slot/block-table mismatch')
    expected_key, expected_value = layer.kv_cache.transpose(1, 2).split(128, dim=-1)
    require(tensor_identity(key) == tensor_identity(expected_key)
            and tensor_identity(value) == tensor_identity(expected_value), 'Wrong live cache views')
    entry['pre_count'] += 1
    pending['query_identity'] = tensor_identity(query)
    entry['pre'] = dict(Q=tensor_bytes(query), KV=logical_kv(key, value, table, lengths[0]),
                        query_starts=starts, seq_lens=lengths,
                        Lr=[lengths[0] - entry['M'] + r + 1 for r in range(entry['M'])])
    entry['cache_binding'] = dict(key=tensor_identity(key), value=tensor_identity(value),
                                  block_table=table, write_slots=slots)


def dispatch(config, state, values):
    pending = getattr(_local, 'attention', None)
    if pending is None or pending['entry']['pre_count'] != 1 or pending['entry']['post_count']: return
    entry = pending['entry']
    require(tensor_identity(values['q']) == pending['query_identity'], 'Dispatch query ownership differs')
    require(entry['dispatch_count'] == 0, 'Duplicate layer0 dispatch')
    entry['dispatch_count'] += 1
    # Admission operands are host configuration/shapes, not a reimplementation.
    names = ('rowwise_decode_max_q', 'max_seqlen_q', 'max_seqlen_k', 'num_seqs',
             'num_kv_heads', 'num_par_softmax_segments', 'seq_threshold_3D', 'causal',
             'window_size', 'softcap', 'use_td', 'use_alibi_sqrt', 'chunk_lookback', 'mm_prefix_clamp_sliding_window')
    operands = {n: values.get(n) for n in names}
    operands.update(q_shape=list(values['q'].shape), q_dtype=str(values['q'].dtype),
                    kv_quant_mode=str(values['kv_quant_mode']))
    operands['tensors'] = {n: dict(shape=list(values[n].shape), dtype=str(values[n].dtype),
        device=str(values[n].device), strides=list(values[n].stride())) for n in
        ('q', 'k', 'v', 'out', 'cu_seqlens_q', 'seqused_k', 'block_table') if n in values}
    module = sys.modules.get('vllm.v1.attention.ops.triton_unified_attention')
    operands['is_batch_invariant'] = getattr(module, 'is_batch_invariant', None)
    operands['absent_features'] = {n: values.get(n) is None for n in
        ('q_descale', 'output_scale', 'alibi_slopes', 'qq_bias', 'sinks', 'mm_prefix_range',
         'rswa_prefix_lens', 'rswa_window')}
    operands['scratch'] = {n: None if values.get(n) is None else dict(
        shape=list(values[n].shape), dtype=str(values[n].dtype), strides=list(values[n].stride()),
        contiguous=values[n].is_contiguous()) for n in
        ('softmax_segm_output', 'softmax_segm_max', 'softmax_segm_expsum')}
    entry['dispatch'] = dict(rowwise=bool(values.get('rowwise_3d', False)),
                             use_3d=bool(values['use_3d']), operands=operands,
                             grid=list(values['grid']), tile=values['tile_size'],
                             segments=values['num_par_softmax_segments'] if values['use_3d'] else 1)


def attention_post(config, state, impl, layer, output):
    pending = getattr(_local, 'attention', None)
    if pending is None or layer is not pending['layer']: return
    entry = pending['entry']
    require(entry['pre_count'] == entry['dispatch_count'] == 1 and entry['post_count'] == 0,
            'Missing/duplicate attention observation')
    row = entry['positions'].index(POSITION)
    entry['post'] = tensor_bytes(output[row:row + 1])
    entry['post_count'] = 1


def score_identity(vector):
    import torch
    # Inspect only the copied selected row; diagnostics add no GPU reductions.
    vector = vector.detach().cpu().contiguous()
    require(vector.ndim == 1 and vector.numel() > 5212 and not torch.isnan(vector).any().item(),
            'Malformed selected score vector')
    return dict(vector=tensor_bytes(vector, include_bytes=False), argmax=int(vector.argmax().item()),
                candidate_scores={str(t): float(vector[t].item()).hex() for t in (341, 5212)})


def scores(config, state, runner, scheduled, logits, spec, grammar):
    _local.scores = None
    entry = state['steps'].get(id(scheduled))
    if entry is None or not entry.get('observed'): return
    require(grammar is None and entry['post_count'] == 1, 'Missing attention/unsupported grammar')
    n = 0 if spec is None else int(spec.num_draft_tokens[0])
    indices = [entry['M'] - 1] if spec is None else ints(spec.logits_indices)
    targets = [] if spec is None else ints(spec.target_logits_indices)
    bonus = [0] if spec is None else ints(spec.bonus_logits_indices)
    rows = map_rows(PROMPT, entry['positions'], indices, n, targets, bonus)
    selected = [r for r in rows if r['generated_index'] == INDEX]
    require(len(selected) == 1 and logits.shape[0] == len(rows), 'Ambiguous boundary logits row')
    producer = state['producers'].get(entry['request_id'])
    if n:
        require(producer is not None and producer['tensor'] is runner._draft_token_ids
                and producer['schedule_seq'] == entry['producer_schedule_seq']
                and runner.input_batch.prev_req_id_to_index.get(entry['request_id']) == producer['row'],
                'Lost consumed-draft producer lineage')
    entry.update(rows=rows, target_row=selected[0], draft_count=n,
                 draft_ids=[] if spec is None else ints(spec.draft_token_ids),
                 effective_drafted=n - entry['invalid'],
                 producer=None if not n else {k: v for k, v in producer.items() if k != 'tensor'},
                 raw=score_identity(logits[selected[0]['row']]))
    require(0 <= entry['effective_drafted'] <= 4, 'Invalid consumed drafts')
    _local.scores = entry


def processed(config, state, stage, logits, metadata, spec=None):
    entry = getattr(_local, 'scores', None)
    if entry is None: return
    require(metadata.all_greedy is True, 'Non-greedy boundary')
    roles = list(range(entry['draft_count'])) if stage == 'verify_scores' else [entry['draft_count']]
    require(logits.shape[0] == len(roles), 'Processed row layout mismatch')
    row = entry['target_row']['row']
    if row not in roles: return
    require('processed' not in entry, 'Duplicate processed score')
    entry['processed'] = score_identity(logits[roles.index(row)])
    entry['processed_stage'] = stage


def sampled(config, state, runner, output):
    entry = getattr(_local, 'scores', None)
    if entry is None: return
    require(output.sampled_token_ids.shape[0] == 1 and 'processed' in entry, 'Missing processed/ambiguous sample')
    ids = selected_tokens(ints(output.sampled_token_ids[0]))
    require(len(ids) <= len(entry['rows']), 'Invalid sampler selection')
    entry['selected'] = ids


def produced(config, state, runner, scheduled):
    entry = state['steps'].get(id(scheduled))
    if entry is None: return
    tensor = runner._draft_token_ids
    require(tensor is not None and tensor.ndim == 2 and tensor.shape[0] == 1
            and 0 <= tensor.shape[1] <= 4, 'Malformed draft production')
    state['producers'][entry['request_id']] = dict(tensor=tensor, row=0,
        schedule_seq=entry['schedule_seq'], planned_k=entry['planned_k'], batch_k=entry['batch_k'],
        future_k=entry['future_k'], produced_count=tensor.shape[1], controller=entry['controller'])


def account_before(config, state, request, output, scheduled, stale, selected):
    if not selected_request(config, request): return
    entry = state['steps'].get(id(scheduled))
    require(entry is not None, 'Unbound pre-accounting')
    entry['before_count'] = len(request._output_token_ids)
    if not output:
        state['steps'].pop(id(scheduled))
        if entry.get('observed'):
            require('selected' in entry, 'Rejected attempt missing sampling')
            entry.update(stale=bool(stale), committed=list(request._output_token_ids), committing=False,
                         accepted_drafts=max(len(entry['selected'])-1, 0),
                         rejected_drafts=entry['effective_drafted']-max(len(entry['selected'])-1, 0))
            emit(config, 'attempt', **entry)


def account(config, state, request, output, scheduled, stale, selected):
    if not selected_request(config, request): return
    entry = state['steps'].pop(id(scheduled), None)
    require(entry is not None, 'Unbound accounting')
    if not entry.get('observed'): return
    require('selected' in entry and list(selected) == entry['selected'], 'Sampler/accounting mismatch')
    committed = list(request._output_token_ids)
    count = len(committed)
    row = entry['target_row']['row']
    # This hook follows _update_request_with_output, including stale handling.
    used = not stale and row < len(entry['selected']) and entry['before_count'] <= INDEX < count
    if used:
        require(entry['before_count'] + row == INDEX
                and committed[INDEX] == entry['selected'][row], 'Committed token not selected row')
    entry.update(stale=bool(stale), committed=committed, committing=used,
                 accepted_drafts=max(len(entry['selected']) - 1, 0),
                 rejected_drafts=entry['effective_drafted'] - max(len(entry['selected']) - 1, 0))
    emit(config, 'attempt', **entry)
    if used: state['closed'] = True


def observe(stage, *args):
    path = os.environ.get('M203_M005_CONFIG')
    if not path: return
    config = None
    try:
        config = load_json(path)
        key = str(Path(config['directory']).resolve())
        if key not in _state:
            _state[key] = initialize(config)
            emit(config, 'bound', process_chain=process_chain(), source_directory=config['source_directory'])
        state = _state[key]
        if state['failed'] or state['closed']: return
        if stage == 'cache_lookup':
            request, local, external = args
            if selected_request(config, request):
                emit(config, stage, local=int(local), external=int(external))
        elif stage in ('sample_scores', 'verify_scores'):
            processed(config, state, stage, *args)
        else:
            handlers = dict(schedule=schedule, prepare=prepare, attention_pre=attention_pre,
                            attention_post=attention_post, dispatch=dispatch, scores=scores,
                            sampled=sampled, produced=produced, account_before=account_before, account=account)
            require(stage in handlers, 'Unknown M005 hook')
            handlers[stage](config, state, *args)
    except Exception as exc:
        # Observation never changes inference. Sticky errors invalidate the run.
        if config is not None and str(Path(config.get('directory', '.')).resolve()) in _state:
            _state[str(Path(config['directory']).resolve())]['failed'] = True
            try: emit(config, 'observer_error', stage=stage, error=type(exc).__name__ + ': ' + str(exc))
            except Exception: pass
        print('M005_OBSERVER_ERROR ' + stage + ' ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
