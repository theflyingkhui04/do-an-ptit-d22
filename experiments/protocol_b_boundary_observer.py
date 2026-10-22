"""Read-only vLLM 0.30.0 observer, narrowly ported from archive a7b52e7.

Reuses m2_03_targeted/observer.py row/score/accounting hooks and A3/V8's
actual-position mapping. Separately gated diagnostics change either the P176
future budget or exactly one P177 layer0 attention-output row. Read-only hooks
never mutate inputs; the attention intervention has its own fail-closed entry.
"""
import json
import os
import threading
import sys
import uuid
from pathlib import Path

from experiments.protocol_b import require, utc_now, write_new, fingerprint

_local = threading.local()
LAYER1 = "model.layers.1.self_attn.attn"
LOCALIZATION_SCHEMA = "m203.boundary.layer1.positions.v1"
EARLY_SCHEMA = "m203.boundary.early.v1"
POSTWRITE_SCHEMA = "m203.boundary.postwrite.v1"
CONTROL_PROFILE = "m203.p177-producer-k0.v1"
ATTENTION_PROFILE = "m203.p177-layer0-attention-substitution.v1"
LAYER0 = "model.layers.0.self_attn.attn"
_controls = {}
_early = {}
_early_lock = threading.RLock()


def attention_settings(config):
    settings = config.get("p177_attention_substitution")
    if settings is None:
        return None
    require(os.environ.get("M203_P177_ATTENTION_SUBSTITUTION") == "1"
            and os.environ.get("M203_P177_PRODUCER_K0") != "1"
            and config.get("p177_k0_control") is None
            and config.get("non_canonical") is True and config.get("lane") == "adaptive"
            and settings["profile"] == ATTENTION_PROFILE
            and config.get("early_timeline") is True and config.get("postwrite_position177") is True,
            "Unauthenticated attention substitution profile/env")
    require(settings["request_sha256"] == fingerprint(settings["request"]), "Attention request hash differs")
    return settings


def attention_arm(config, pending):
    settings = attention_settings(config)
    if settings is None:
        return
    state = early_state(config)
    entry = early_step(state, pending["scheduled"])
    producers = [e for e in state["steps"].values()
                 if e["schedule_seq"] == entry["placeholder_origin_schedule_seq"]]
    require(len(producers) == 1, "Attention missing producer lineage")
    producer = producers[0]
    rid = pending["request_id"]
    owner = state["producers"].get((id(pending["runner"]), rid))
    require(producer.get("verified") is True and owner is not None
            and owner["tensor"] is pending["runner"]._draft_token_ids
            and owner["row"] == 0 and owner["schedule_seq"] == producer["schedule_seq"]
            and list(owner["tensor"].shape) == [1,3], "Attention draft producer tensor ownership differs")
    require(entry["request_id"] == producer["request_id"] == state["request_id"] == rid
            and pending["positions"] == [177, 178, 179, 180]
            and entry["active_query"][:1] == [[177, settings["token_prefix"][1]]]
            and producer["positions"] == [176]
            and producer["active_query"] == [[176, settings["token_prefix"][0]]]
            and producer["planned_k"] == producer["batch_k"] == producer["future_k"] == 3
            and len(pending["scheduled"].scheduled_spec_decode_tokens.get(rid, [])) == 3,
            "Attention wrong normal producer/query/drafts")
    require(all(177 not in e.get("positions", []) for e in state["steps"].values() if e is not entry),
            "Attention P177 already covered")
    require(pending["peers"][LAYER0] == pending["binding"]["layers"][LAYER0]["cache"],
            "Attention target cache ownership differs")
    left, right = cache_view_span(pending["peers"][LAYER0])
    for name, identity in pending["peers"].items():
        start, end = cache_view_span(identity)
        require(identity["device"] == pending["peers"][LAYER0]["device"]
                and (name == LAYER0 or end <= left or start >= right), "Attention target/draft cache overlap")
    pending["attention_state"] = dict(shadow_count=0, substitution_count=0)
    _local.attention_pending = pending


def attention_bytes(tensor):
    """Original FP16 bits, including signed zero; never cast to float32."""
    import hashlib
    import torch
    require(tensor.dtype == torch.float16 and sys.byteorder == "little", "Attention byte contract differs")
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), hex=raw.hex(), byte_count=len(raw),
                shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device))


def attention_shadow_arguments(impl, layer, query, key, value, metadata, output):
    """CPU-testable argument construction, not a simulation of Triton numerics."""
    import torch
    require(query.dtype == output.dtype == key.dtype == value.dtype == torch.float16
            and query.device == output.device == key.device == value.device
            and list(query.shape) == list(output.shape) == [4, 28, 128]
            and impl.num_heads == 28 and impl.num_kv_heads == 4 and impl.head_size == 128,
            "Attention tensor shape/dtype/device differs")
    require(metadata.num_actual_tokens == metadata.max_query_len == 4 and metadata.max_seq_len == 181
            and metadata.query_start_loc.device == metadata.seq_lens.device == metadata.block_table.device == query.device
            and ints(metadata.query_start_loc) == [0, 4] and ints(metadata.seq_lens) == [181]
            and metadata.causal is True and not metadata.use_cascade
            and metadata.mm_prefix_range_tensor is None and metadata.rswa_prefix_lens is None
            and metadata.rswa_window is None and impl._kv_quant_mode.name == "NONE"
            and not impl._is_per_token_head_quant and impl.alibi_slopes is None and impl.sinks is None
            and not impl.use_alibi_sqrt and not impl.use_td and impl.chunk_lookback == -1
            and tuple(impl.sliding_window) == (-1, -1) and impl.logits_soft_cap == 0,
            "Unsupported attention semantics")
    require(metadata.seq_threshold_3D >= 1 and metadata.num_par_softmax_segments == 16
            and all(getattr(metadata, name) is not None for name in
                    ("softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum")),
            "Missing native segmented-3D support")
    require(all(ints(getattr(layer, name).reshape(-1)) == [1.0]
                for name in ("_q_scale", "_k_scale", "_v_scale")), "Attention nonunit scales")
    # New metadata and scratch only: never alter live seq_lens/query_start_loc,
    # cache, slots, or native scratch. The existing compute-only kernel sees
    # exactly the prefix through P177, excluding cached speculative future rows.
    device = query.device
    return dict(q=query[:1], k=key, v=value, out=torch.empty_like(output[:1]),
        cu_seqlens_q=torch.tensor([0, 1], dtype=metadata.query_start_loc.dtype, device=device),
        max_seqlen_q=1, seqused_k=torch.tensor([178], dtype=metadata.seq_lens.dtype, device=device),
        max_seqlen_k=178, softmax_scale=impl.scale, causal=True, alibi_slopes=None,
        use_alibi_sqrt=False, window_size=impl.sliding_window, block_table=metadata.block_table[:1],
        softcap=impl.logits_soft_cap, q_descale=None, k_descale=layer._k_scale.expand(1, 4),
        v_descale=layer._v_scale.expand(1, 4), seq_threshold_3D=metadata.seq_threshold_3D,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty((1, 28, 16, 128), dtype=torch.float32, device=device),
        softmax_segm_max=torch.empty((1, 28, 16), dtype=torch.float32, device=device),
        softmax_segm_expsum=torch.empty((1, 28, 16), dtype=torch.float32, device=device),
        sinks=None, output_scale=None, mm_prefix_range=None, rswa_prefix_lens=None, rswa_window=None,
        kv_quant_mode=impl._kv_quant_mode, k_scale_cache=None, v_scale_cache=None,
        chunk_lookback=-1, use_td=False,
        mm_prefix_clamp_sliding_window=getattr(layer, "mm_prefix_clamp_sliding_window", False))


def attention_apply(config, pending, impl, layer, query, key, value, metadata, output, compute):
    """Only mutation: output[0:1].copy_(shadow). All other work is observation."""
    runner = pending["runner"]
    context = runner.vllm_config.compilation_config.static_forward_context
    require(layer is context[LAYER0] and impl is layer.impl and metadata is pending["metadata"][LAYER0]
            and pending["config_key"] == fingerprint(config)
            and pending["positions"] == [177, 178, 179, 180], "Attention target/request binding differs")
    require(tensor_identity(layer.kv_cache) == pending["binding"]["layers"][LAYER0]["cache"],
            "Attention cache identity changed")
    require(ints(pending["inputs"]["positions"]) == pending["positions"]
            and ints(pending["inputs"]["input_ids"])[0] == config["p177_attention_substitution"]["token_prefix"][1],
            "Attention live target inputs changed")
    expected_key, expected_value = layer.kv_cache.transpose(1, 2).split(128, dim=-1)
    require(tensor_identity(key) == tensor_identity(expected_key)
            and tensor_identity(value) == tensor_identity(expected_value), "Attention not the live cache views")
    slots = ints(pending["slots"][LAYER0])
    table = ints(metadata.block_table[0])
    position177_slot(table, slots, pending["positions"], int(layer.kv_cache.shape[2]), int(layer.kv_cache.shape[0]))
    require(slots == [table[pos // 16]*16 + pos % 16 for pos in pending["positions"]],
            "Attention actual row slots differ")
    state = pending["attention_state"]
    require(state == dict(shadow_count=0, substitution_count=0), "Duplicate attention shadow/substitution")
    args = attention_shadow_arguments(impl, layer, query, key, value, metadata, output)
    native, tail = attention_bytes(output[:1]), attention_bytes(output[1:])
    q_identity, q_bytes = tensor_identity(query[:1]), attention_bytes(query[:1])
    state["shadow_count"] = 1  # A failing compute can never be retried/accepted.
    compute(**args)  # compute-only unified_attention; no Attention.forward/cache writer.
    shadow = attention_bytes(args["out"])
    require(tensor_identity(query[:1]) == q_identity and attention_bytes(query[:1]) == q_bytes
            and attention_bytes(output[:1]) == native and attention_bytes(output[1:]) == tail,
            "Shadow changed native inputs/output")
    output[:1].copy_(args["out"])
    state["substitution_count"] = 1
    require(attention_bytes(output[:1]) == shadow and attention_bytes(output[1:]) == tail,
            "Attention substitution changed other rows")
    emit(config, "p177_attention_substitution", attention_profile=ATTENTION_PROFILE, non_canonical=True,
        pid=os.getpid(), request_id=pending["request_id"], schedule_seq=pending["schedule_seq"],
        scheduler_output_id=id(pending["scheduled"]), forward_id=pending["forward_id"],
        auth_id=kv_authenticate(config), actual_target_positions=pending["positions"], planned_k=3,
        scheduled_drafts=3, shadow_count=1, substitution_count=1, target_layer=LAYER0,
        native_attention_p177=native, shadow_3d_attention_p177=shadow, native_tail=tail,
        tail_after=attention_bytes(output[1:]), q_identity=q_identity, q_bytes=q_bytes,
        cache_identity=tensor_identity(layer.kv_cache), key_identity=tensor_identity(key),
        value_identity=tensor_identity(value), block_table=table, write_slots=slots,
        graph_mode="PIECEWISE", shadow_query_start_loc=[0, 1], shadow_seq_lens=[178],
        shadow_max_query_len=1, shadow_max_seq_len=178, shadow_segments=16,
        output_identity=tensor_identity(output), no_extra_kv_write=True,
        substitution_point="after_native_attention_before_return_to_output_projection",
        prefix_0_176=hash_control_prefix(context[LAYER1].kv_cache,
            ints(pending["metadata"][LAYER1].block_table[0]), ints(pending["slots"][LAYER1])))


def attention_substitute(impl, layer, query, key, value, metadata, output, output_scale, output_block_scale, compute):
    """Direct, fail-closed hook; never swallowed by the read-only observer wrapper."""
    if os.environ.get("M203_P177_ATTENTION_SUBSTITUTION") != "1":
        return
    config = json.loads(Path(os.environ["PROTOCOL_B_BOUNDARY_CONFIG"]).read_text())
    require(attention_settings(config) is not None, "Attention env requires explicit profile")
    if config.get("sample_id") != "w3_business_v2_004":
        return
    require(getattr(_local, "postwrite_v1_phase", "idle") != "failed", "Failed attention preparation")
    pending = getattr(_local, "attention_pending", None)
    if pending is None:
        return  # Warmup, other requests, and ordinary/other-position forwards.
    context = pending["runner"].vllm_config.compilation_config.static_forward_context
    if layer is not context[LAYER0]:
        require(any(layer is item for item in context.values()), "Unknown attention layer")
        return
    require(output_scale is None and output_block_scale is None and query.device.type == "cuda"
            and pending is getattr(_local, "postwrite_v1_pending", None)
            and pending["stream"] == postwrite_stream(), "Attention live forward/stream differs")
    module = sys.modules.get("vllm.v1.attention.ops.triton_unified_attention")
    require(module is not None and module.unified_attention is compute
            and module.is_batch_invariant is False, "Unbound/non-native shadow kernel")
    kv_authenticate(config)  # Authenticate installed source before the intervention.
    attention_apply(config, pending, impl, layer, query, key, value, metadata, output, compute)


def control_settings(config):
    settings = config.get("p177_k0_control")
    if settings is None:
        return None
    require(os.environ.get("M203_P177_PRODUCER_K0") == "1"
            and config.get("non_canonical") is True and config.get("lane") == "adaptive"
            and settings["profile"] == CONTROL_PROFILE and config.get("early_timeline") is True
            and config.get("postwrite_position177") is True, "Unauthenticated control profile/env")
    require(settings["request_sha256"] == fingerprint(settings["request"]), "Control request hash differs")
    return settings


def control_event(config, state, kind, **fields):
    emit(config, "p177_control_" + kind, control_profile=CONTROL_PROFILE, non_canonical=True,
         control_id=state["control_id"], request_id=state["request_id"], pid=os.getpid(), **fields)


def control_producer(scheduler, scheduled):
    """Sole intervention; outside observer/planner exception fallbacks, before native super."""
    if os.environ.get("M203_P177_PRODUCER_K0") != "1":
        return
    config = json.loads(Path(os.environ["PROTOCOL_B_BOUNDARY_CONFIG"]).read_text())
    settings = control_settings(config)
    require(settings is not None, "Control env requires explicit non-canonical profile")
    if config.get("sample_id") != "w3_business_v2_004" or len(config["prompt_token_ids"]) != 176:
        return
    early = early_state(config)
    candidates = [r for rid in scheduled.num_scheduled_tokens
                  if (r := scheduler.requests.get(rid)) is not None
                  and r.request_id == early["request_id"]
                  and list(r.prompt_token_ids or []) == config["prompt_token_ids"]
                  and not r.is_prefill_chunk and r.num_computed_tokens == 176]
    if not candidates:
        return
    require(len(candidates) == len(scheduled.num_scheduled_tokens) == 1, "Ambiguous control producer")
    request = candidates[0]
    rid = request.request_id
    require(scheduled.num_scheduled_tokens[rid] == scheduled.total_num_scheduled_tokens == 1
            and not scheduled.scheduled_spec_decode_tokens.get(rid), "Control producer must be ordinary [176]")
    original = scheduled.num_spec_tokens_to_schedule
    require(type(original) is int and original == settings["normal_producer_budget"] > 0
            and scheduler._plan.batch_k == original
            and scheduler._plan.per_request_k.get(rid) == original, "Control normal planner decision differs")
    key = fingerprint(config)
    with _early_lock:
        require(key not in _controls, "Second control override attempted")
        state = dict(control_id=uuid.uuid4().hex, request_id=rid, producer=scheduled,
                     producer_seq=scheduler.current_step, original_budget=original,
                     original_plan=dict(batch_k=scheduler._plan.batch_k,
                                        per_request_k=dict(scheduler._plan.per_request_k)))
        _controls[key] = state
        scheduled.num_spec_tokens_to_schedule = 0
    control_event(config, state, "override", producer_seq=state["producer_seq"],
        scheduler_output_id=id(scheduled), original_budget=original, effective_budget=0,
        original_plan=state["original_plan"], override_count=1,
        request_sha256=settings["request_sha256"], baseline_sha256=settings["baseline_sha256"])


def control_native(config, scheduler, scheduled):
    if control_settings(config) is None:
        return
    state = _controls.get(fingerprint(config))
    if state is None or state["producer"] is not scheduled:
        return
    rid = state["request_id"]
    require(not state.get("native") and scheduled.num_spec_tokens_to_schedule == 0
            and list(scheduler.requests[rid].spec_token_ids) == []
            and dict(batch_k=scheduler._plan.batch_k, per_request_k=dict(scheduler._plan.per_request_k))
                == state["original_plan"], "Native zero placeholders/planner proof failed")
    state["native"] = True
    control_event(config, state, "native", producer_seq=state["producer_seq"],
                  scheduler_output_id=id(scheduled), effective_budget=0, next_placeholders=[])


def control_query(config, runner, scheduled, positions):
    settings = control_settings(config)
    if settings is None or not ({176, 177} & set(positions)):
        return
    state = _controls.get(fingerprint(config))
    require(state is not None and state.get("native"), "Missing authenticated control producer")
    entry = early_step(early_state(config), scheduled)
    require(entry["request_id"] == state["request_id"], "Control query request changed")
    if scheduled is state["producer"]:
        require(not state.get("producer_verified") and positions == [176]
                and entry["active_query"] == [[176, settings["token_prefix"][0]]], "Wrong actual control producer")
        state["producer_verified"] = True
        control_event(config, state, "producer", producer_seq=entry["schedule_seq"],
                      scheduler_output_id=id(scheduled), actual_positions=positions, active_query=entry["active_query"])
    else:
        require(state.get("producer_verified") and state.get("drafts_verified")
                and not state.get("consumer") and positions == [177]
                and entry["placeholder_origin_schedule_seq"] == state["producer_seq"]
                and not scheduled.scheduled_spec_decode_tokens.get(state["request_id"])
                and entry["active_query"] == [[177, settings["token_prefix"][1]]], "Wrong first control consumer/rewrite")
        state["consumer"] = scheduled
        control_event(config, state, "consumer", producer_seq=state["producer_seq"],
            schedule_seq=entry["schedule_seq"], scheduler_output_id=id(scheduled),
            actual_positions=positions, active_query=entry["active_query"], scheduled_spec_tokens=[])


def control_drafts(config, runner, scheduled):
    if control_settings(config) is None:
        return
    state = _controls.get(fingerprint(config))
    if state is None or state["producer"] is not scheduled:
        return
    require(state.get("producer_verified") and not state.get("drafts_verified")
            and list(runner._draft_token_ids.shape) == [1, 0], "Control producer did not produce zero drafts")
    state["drafts_verified"] = True
    control_event(config, state, "drafts", producer_seq=state["producer_seq"],
                  scheduler_output_id=id(scheduled), produced_count=0)


def hash_control_prefix(cache, table, write_slots):
    """Exactly layer-1 logical positions 0..176; history_slots remains all-454."""
    import hashlib
    import torch
    require(cache.dtype == torch.float16 and cache.ndim == 4 and cache.shape[-1] % 2 == 0
            and sys.byteorder == "little", "Unsupported control prefix representation")
    block_size, heads, dim = int(cache.shape[2]), int(cache.shape[1]), int(cache.shape[3]) // 2
    require(len(table) > 176 // block_size, "Incomplete control prefix table")
    blocks = [table[p // block_size] for p in range(177)]
    require(all(type(b) is int and 0 <= b < cache.shape[0] for b in blocks), "Prefix block out of bounds")
    slots = [b * block_size + p % block_size for p, b in enumerate(blocks)]
    require(len(set(slots)) == 177 and not set(slots).intersection(s for s in write_slots if s >= 0),
            "Control prefix aliases history/current/padded writes")
    indices = torch.tensor(slots, dtype=torch.long, device=cache.device)
    result = dict(logical_positions=list(range(177)), dtype=str(cache.dtype), shape=[177, heads, dim],
                  bytes_per_position=heads * dim * 2, canonical_order=["kv_head", "head_dimension"])
    for component, start in (("K", 0), ("V", dim)):
        selected = cache[indices // block_size, :, indices % block_size, start:start + dim]
        raw = selected.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        size = result["bytes_per_position"]
        require(len(raw) == 177 * size, "Control prefix byte count differs")
        result[component] = [hashlib.sha256(raw[p*size:(p+1)*size]).hexdigest() for p in range(177)]
    return result


def early_state(config):
    if config.get("early_timeline") is not True:
        return None
    require(config["lane"] == "adaptive" and config["sample_id"] == "w3_business_v2_004"
            and len(config["prompt_token_ids"]) == 176, "Early scope differs")
    return _early.setdefault(fingerprint(config), dict(steps={}, producers={}, origins={}, sequence=0,
                                                     closed=False, request_id=None))


def early_emit(config, state, kind, **fields):
    with _early_lock:
        entries = [e for e in state["steps"].values() if e["schedule_seq"] == fields["schedule_seq"]]
        require(len(entries) == 1, "Early event has no unique live output")
        state["sequence"] += 1
        emit(config, "early_" + kind, early_schema=EARLY_SCHEMA, pid=os.getpid(),
             scheduler_output_id=id(entries[0]["output"]),
             event_seq=state["sequence"], runtime_probe_changes_timing=True, **fields)


def early_step(state, scheduled):
    entry = state["steps"].get(id(scheduled))
    require(entry is not None and entry["output"] is scheduled,
            "Early scheduler output identity lost (requires same-process executor)")
    return entry


def early_schedule(config, scheduler, scheduled):
    state = early_state(config)
    if state is None or state["closed"]:
        return
    matches = [r for rid in scheduled.num_scheduled_tokens
               if (r := scheduler.requests.get(rid)) is not None
               and list(r.prompt_token_ids or []) == config["prompt_token_ids"]]
    if not matches:
        return
    require(len(matches) == len(scheduled.num_scheduled_tokens) == 1, "Early ambiguous batch")
    request = matches[0]
    require(state["request_id"] in (None, request.request_id), "Early request collision")
    state["request_id"] = request.request_id
    # current_step is monotonic in pinned Scheduler.schedule(), but is not in
    # SchedulerOutput. Preserve its value against the exact live output object.
    seq = scheduler.current_step
    require(type(seq) is int and seq > 0 and id(scheduled) not in state["steps"]
            and all(seq > e["schedule_seq"] for e in state["steps"].values()),
            "Early nonmonotonic scheduling identity")
    origin = state["origins"].get(request.request_id)
    entry = dict(output=scheduled, schedule_seq=seq, request_id=request.request_id,
                 placeholder_origin_schedule_seq=origin)
    if attention_settings(config) is not None:
        entry.update(planned_k=scheduler._plan.per_request_k.get(request.request_id),
                     batch_k=scheduler._plan.batch_k, future_k=scheduled.num_spec_tokens_to_schedule)
    state["steps"][id(scheduled)] = entry
    state["origins"][request.request_id] = seq
    early_emit(config, state, "schedule", schedule_seq=seq, request_id=request.request_id,
               identity_source="Scheduler.current_step+same-output-object",
               placeholder_origin_schedule_seq=origin,
               planned_k=scheduler._plan.per_request_k.get(request.request_id),
               planner_request_id=request.request_id if request.request_id in scheduler._plan.per_request_k else None,
               batch_k=scheduler._plan.batch_k,
               num_spec_tokens_to_schedule=scheduled.num_spec_tokens_to_schedule,
               next_placeholders=list(request.spec_token_ids),
               scheduled_spec_decode_tokens=list(scheduled.scheduled_spec_decode_tokens.get(request.request_id, [])),
               scheduled_tokens=scheduled.num_scheduled_tokens[request.request_id],
               num_computed_tokens=request.num_computed_tokens,
               num_in_flight_tokens=request.num_in_flight_tokens,
               num_output_placeholders=request.num_output_placeholders)
    control_native(config, scheduler, scheduled)


def early_prepare(config, runner, scheduled, input_ids, positions):
    state = early_state(config)
    if state is None or state["closed"]:
        return
    entry = early_step(state, scheduled)
    require(list(runner.input_batch.req_ids[:runner.input_batch.num_reqs]) == [entry["request_id"]],
            "Early worker request ownership differs")
    require(positions and positions == list(range(positions[0], positions[-1] + 1)),
            "Early noncontiguous actual positions")
    if min(positions) > 180:
        state["closed"] = True
        early_emit(config, state, "complete", schedule_seq=entry["schedule_seq"],
                   request_id=entry["request_id"], first_outside_position=positions[0])
        return
    require(positions[-1] <= 175 or (positions[0] >= 176 and len(positions) <= 5),
            "Early unsupported prefill/decode crossing")
    # Prefill is a producer too. Only its last input row is needed; never copy
    # the entire prompt. Decode retains the whole intersecting attempt.
    start = len(positions) - 1 if positions[-1] < 176 else 0
    entry.update(positions=positions, active_query=list(map(list, zip(
        positions[start:], ints(input_ids[start:len(positions)])))), runner=id(runner))


def early_verify(config, runner, scheduled, rows, spec):
    state = early_state(config)
    if state is None:
        return
    entry = state["steps"].get(id(scheduled))
    if entry is None or "active_query" not in entry:
        return
    require(entry["output"] is scheduled and "verified" not in entry, "Early duplicate forward")
    entry["verified"] = True
    n = 0 if spec is None else int(spec.num_draft_tokens[0])
    drafts = [] if spec is None else ints(spec.draft_token_ids)
    producer = state["producers"].get((id(runner), entry["request_id"]))
    # Explicit tensor identity and prior-batch row mapping, not event adjacency.
    if n:
        require(producer is not None and producer["tensor"] is runner._draft_token_ids
                and producer["schedule_seq"] == entry["placeholder_origin_schedule_seq"]
                and runner.input_batch.prev_req_id_to_index.get(entry["request_id"]) == producer["row"],
                "Early draft producer ownership lost")
    invalid = (scheduled.num_invalid_spec_tokens or {}).get(entry["request_id"], 0)
    early_emit(config, state, "verify", schedule_seq=entry["schedule_seq"],
               request_id=entry["request_id"],
               producer_schedule_seq=None if producer is None else producer["schedule_seq"],
               producer_tensor_id=None if producer is None else id(runner._draft_token_ids),
               previous_request_row=runner.input_batch.prev_req_id_to_index.get(entry["request_id"]),
               actual_position_span=[entry["positions"][0], entry["positions"][-1] + 1],
               active_query=entry["active_query"], rows=rows,
               actual_draft_token_ids=drafts, verification_draft_slots=n,
               num_invalid=invalid, effective_num_drafted=n - invalid,
               decoding_path="speculative" if n else "ordinary")


def early_produced(config, runner, scheduled):
    state = early_state(config)
    if state is None:
        return
    entry = state["steps"].get(id(scheduled))
    if entry is None or "active_query" not in entry:
        return
    require(entry["output"] is scheduled and entry.get("verified") is True,
            "Early production without verification")
    ids = list(runner.input_batch.req_ids[:runner.input_batch.num_reqs])
    require(ids == [entry["request_id"]], "Early draft producer batch changed")
    tensor = runner._draft_token_ids
    require(tensor is not None and tensor.ndim == 2 and tensor.shape[0] == 1
            and 0 <= tensor.shape[1] <= 4, "Early unsupported produced drafts")
    # Metadata only. Actual consumed IDs are captured once from the existing
    # verification observer read; never force the drafter's optional D2H path.
    state["producers"][(id(runner), ids[0])] = dict(
        tensor=tensor, row=0, schedule_seq=entry["schedule_seq"])
    early_emit(config, state, "produced", request_id=ids[0], schedule_seq=entry["schedule_seq"],
               produced_count=int(tensor.shape[1]), producer_tensor_id=id(tensor), producer_row=0)
    control_drafts(config, runner, scheduled)


def early_account(config, stage, request, output, scheduled, stale, selected):
    state = early_state(config)
    if state is None:
        return
    entry = state["steps"].get(id(scheduled))
    if entry is None or "active_query" not in entry:
        return
    require(entry["output"] is scheduled and request.request_id == entry["request_id"],
            "Early accounting ownership lost")
    slots = len(scheduled.scheduled_spec_decode_tokens.get(request.request_id, []))
    invalid = (scheduled.num_invalid_spec_tokens or {}).get(request.request_id, 0)
    accepted = max(len(selected) - 1, 0)
    early_emit(config, state, stage, request_id=request.request_id,
               schedule_seq=entry["schedule_seq"], selected_token_ids=list(selected),
               output_token_ids=list(output), committed_ids=list(request._output_token_ids),
               output_count=len(request._output_token_ids), stale=stale,
               num_invalid=invalid, num_drafted=slots - invalid,
               num_accepted=accepted, num_rejected=slots - invalid - accepted,
               scheduler_rejected_slots=slots - accepted,
               scheduled_spec_decode_tokens=list(scheduled.scheduled_spec_decode_tokens.get(request.request_id, [])),
               num_computed_tokens=request.num_computed_tokens,
               num_in_flight_tokens=request.num_in_flight_tokens,
               num_output_placeholders=request.num_output_placeholders)


def postwrite_enabled(config):
    if config.get("postwrite_position177") is not True:
        return False
    require(config["sample_id"] == "w3_business_v2_004" and len(config["prompt_token_ids"]) == 176
            and config["lane"] in ("vanilla", "adaptive"), "Post-write scope differs")
    return True


def position177_slot(block_table, slots, positions, block_size, num_blocks):
    """One-position CPU addressing contract; does not relax history_slots()."""
    require(type(block_size) is int and block_size > 0 and type(num_blocks) is int and num_blocks > 0,
            "Post-write invalid cache dimensions")
    require(positions and all(type(p) is int and p >= 0 for p in positions)
            and positions.count(177) == 1 and len(set(positions)) == len(positions),
            "Post-write position 177 missing/ambiguous")
    require(len(slots) >= len(positions) and all(type(s) is int and s < num_blocks * block_size for s in slots),
            "Post-write slot extent/out of bounds")
    actual = slots[:len(positions)]
    require(all(s >= 0 for s in actual) and len(set(actual)) == len(actual), "Post-write actual-row alias")
    require(len(block_table) > 177 // block_size, "Post-write incomplete block table")
    physical = block_table[177 // block_size]
    require(type(physical) is int and 0 <= physical < num_blocks, "Post-write physical block out of bounds")
    offset = 177 % block_size
    slot = physical * block_size + offset
    row = positions.index(177)
    require(actual[row] == slot and slot not in slots[len(positions):], "Post-write wrong slot/padding alias")
    return dict(physical_block=physical, offset=offset, slot=slot, selected_row=row, kernel_block_size=block_size)


def cache_view_span(identity):
    """Bounding bytes reachable by a non-negative-stride FP16 cache view."""
    require(identity["dtype"] == "torch.float16" and len(identity["shape"]) == len(identity["strides"]) == 4,
            "Post-write unsupported cache representation")
    require(all(type(dim) is int and dim > 0 for dim in identity["shape"])
            and all(type(stride) is int and stride >= 0 for stride in identity["strides"]),
            "Post-write invalid cache shape/strides")
    storage_start, nbytes = identity["storage_ptr"], identity["storage_nbytes"]
    require(type(storage_start) is int and storage_start > 0 and type(nbytes) is int and nbytes > 0,
            "Post-write invalid backing storage bounds")
    offset = identity["storage_offset"]
    require(type(offset) is int and offset >= 0, "Post-write invalid storage offset")
    maximum = offset + sum((dim - 1) * stride for dim, stride in zip(identity["shape"], identity["strides"]))
    start, end = storage_start + offset * 2, storage_start + (maximum + 1) * 2
    require(storage_start <= start < end <= storage_start + nbytes,
            "Post-write reachable view outside backing storage")
    return start, end


def postwrite_ownership(binding, peers):
    require(set(binding["expected_layers"]) <= set(peers) and LAYER1 in binding["expected_layers"] and LAYER1 in peers
            and peers[LAYER1] == binding["layers"][LAYER1]["cache"], "Post-write wrong target layer")
    left, right = cache_view_span(peers[LAYER1])
    for name, identity in peers.items():
        start, end = cache_view_span(identity)
        require(identity["device"] == peers[LAYER1]["device"], "Post-write mixed cache devices")
        require(name == LAYER1 or end <= left or start >= right, "Post-write target/draft cache overlap")


def hash_position177(cache, physical_block, offset):
    """Exact original bits; CPU tests exercise bytes/indexing, never GPU behavior."""
    import hashlib
    import torch
    require(str(cache.dtype) == "torch.float16" and cache.ndim == 4 and cache.shape[-1] % 2 == 0
            and 0 <= physical_block < cache.shape[0] and 0 <= offset < cache.shape[2],
            "Post-write unsupported FP16 cache row")
    heads, dim = int(cache.shape[1]), int(cache.shape[3]) // 2
    result = dict(dtype=str(cache.dtype), shape=[heads, dim], heads=heads, head_dim=dim,
                  element_count=heads * dim, byte_count=heads * dim * 2,
                  total_value_bytes=heads * dim * 4, canonical_order=["kv_head", "head_dimension"])
    for component, start in (("K", 0), ("V", dim)):
        selected = cache[physical_block, :, offset, start:start + dim]
        raw = selected.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        require(len(raw) == result["byte_count"] and sys.byteorder == "little", "Post-write byte accounting differs")
        result[component] = dict(sha256=hashlib.sha256(raw).hexdigest())
    return result


def postwrite_stream():
    import torch
    require(not torch.cuda.is_current_stream_capturing(), "Post-write capture/warmup cannot qualify")
    stream = torch.cuda.current_stream()
    return dict(device=str(stream.device), cuda_stream=int(stream.cuda_stream))


def postwrite_reset(config, version):
    if not postwrite_enabled(config):
        return
    require(getattr(_local, "postwrite_" + version + "_pending", None) is None, "Stale post-write pending")
    setattr(_local, "postwrite_" + version + "_phase", "idle")


def postwrite_prepare(config, runner, inputs, metadata, slots, positions, rid, index,
                      mode, write_count, *, scheduled=None, manager=None, desc=None):
    version = "v2" if manager is not None else "v1"
    claims = getattr(_local, "postwrite_claims", {})
    key = fingerprint(config)
    if key in claims:
        require(claims[key] == rid, "Post-write request collision")
        require(version == "v1", "Duplicate Vanilla position-177 forward")
        return  # Ordered Adaptive rewrites are retained by the early timeline.
    control = control_settings(config)
    require(positions == ([177] if version == "v2" or control is not None else [177, 178, 179, 180]),
            "Post-write unexpected first position-177 query")
    require(mode.name == ("FULL" if version == "v2" else "PIECEWISE"), "Post-write graph mode differs")
    early = None
    if version == "v1":
        state = early_state(config)
        require(state is not None, "Post-write requires early timeline")
        early = early_step(state, scheduled)
        require(early["request_id"] == rid and early["positions"] == positions
                and len(scheduled.scheduled_spec_decode_tokens.get(rid, [])) == (0 if control is not None else 3),
                "Post-write wrong early scheduler/request")
        if control is not None:
            require(_controls[fingerprint(config)].get("consumer") is scheduled, "Unbound control consumer")
    else:
        require(manager._graphs_captured and manager._capture_mem_samples is None
                and desc.num_ubatches == 1, "Post-write unready/warmup graph manager")
    binding = kv_layer_bindings(runner.vllm_config, runner.kv_cache_config, inputs, metadata, slots,
                               v1_runner=runner if version == "v1" else None)
    context = runner.vllm_config.compilation_config.static_forward_context
    peers = {name: tensor_identity(context[name].kv_cache)
             for group in runner.kv_cache_config.kv_cache_groups for name in group.layer_names}
    postwrite_ownership(binding, peers)
    require(context[LAYER1].kv_cache.device.type == "cuda", "Post-write live cache must be CUDA")
    capture = None
    if version == "v2":
        capture = getattr(_local, "kv_captures", {}).get((id(manager), repr(desc)))
        require(capture is not None, "Post-write missing FULL capture binding")
        validate_full_binding(capture["bindings"], binding)
    stream = postwrite_stream()
    require(stream["device"] == peers[LAYER1]["device"], "Post-write execution/cache device differs")
    pending = dict(config_key=key, runner=runner, inputs=inputs, metadata=metadata, slots=slots,
                   positions=positions, request_id=rid, request_state_index=index, binding=binding,
                   peers=peers, stream=stream, write_count=write_count, scheduled=scheduled,
                   manager=manager, desc=desc, capture=capture,
                   schedule_seq=None if early is None else early["schedule_seq"], forward_id=uuid.uuid4().hex,
                   active_query=(early["active_query"] if early is not None else
                                 list(map(list, zip(positions, ints(inputs["input_ids"][:len(positions)]))))))
    # The writer launches over slot_mapping.shape[0], including padding rows.
    require(type(write_count) is int and len(positions) <= write_count == slots[LAYER1].numel(),
            "Post-write incomplete consumed slot extent")
    claims[key] = rid
    _local.postwrite_claims = claims
    setattr(_local, "postwrite_" + version + "_pending", pending)
    setattr(_local, "postwrite_" + version + "_phase", "armed")
    if version == "v1":
        attention_arm(config, pending)


def postwrite_finish(config, version, owner, execution):
    if not postwrite_enabled(config):
        return
    phase = getattr(_local, "postwrite_" + version + "_phase", "idle")
    require(phase != "consumed", "Duplicate post-write hook")
    pending = getattr(_local, "postwrite_" + version + "_pending", None)
    if pending is None:
        require(phase == "idle", "Missing post-write pending")
        return
    setattr(_local, "postwrite_" + version + "_pending", None)
    setattr(_local, "postwrite_" + version + "_phase", "consumed")
    require(pending["config_key"] == fingerprint(config) and pending["stream"] == postwrite_stream(),
            "Post-write stale config/execution stream")
    if version == "v2":
        require(pending["manager"] is owner and pending["desc"] == execution
                and owner._graphs_captured and owner._capture_mem_samples is None,
                "Post-write wrong manager/descriptor or warmup")
    else:
        require(pending["runner"] is owner and pending["scheduled"] is execution,
                "Post-write wrong V1 scheduler/runner identity")
        entry = early_step(early_state(config), execution)
        require(entry["schedule_seq"] == pending["schedule_seq"] and entry["request_id"] == pending["request_id"],
                "Post-write early identity drift")
    runner = pending["runner"]
    if attention_settings(config) is not None:
        require(version == "v1" and getattr(_local, "attention_pending", None) is pending
                and pending.get("attention_state") == dict(shadow_count=1, substitution_count=1),
                "Missing attention shadow/substitution")
        _local.attention_pending = None
    binding = kv_layer_bindings(runner.vllm_config, runner.kv_cache_config, pending["inputs"],
        pending["metadata"], pending["slots"], v1_runner=runner if version == "v1" else None)
    require(binding == pending["binding"], "Post-write prepared binding changed")
    context = runner.vllm_config.compilation_config.static_forward_context
    peers = {name: tensor_identity(context[name].kv_cache)
             for group in runner.kv_cache_config.kv_cache_groups for name in group.layer_names}
    require(peers == pending["peers"], "Post-write cache ownership changed")
    postwrite_ownership(binding, peers)
    cache, m = context[LAYER1].kv_cache, pending["metadata"][LAYER1]
    block_size = int(cache.shape[2])
    write_slots = ints(pending["slots"][LAYER1][:pending["write_count"]])
    table = ints(m.block_table[0, :177 // block_size + 1])
    location = position177_slot(table, write_slots, pending["positions"], block_size, int(cache.shape[0]))
    auth_id = kv_authenticate(config)
    content = hash_position177(cache, location["physical_block"], location["offset"])
    extra = {}
    if control_settings(config) is not None:
        state = _controls[fingerprint(config)]
        require(state.get("consumer") is execution, "Control post-write identity differs")
        extra = dict(control_profile=CONTROL_PROFILE, control_id=state["control_id"], non_canonical=True,
                     prefix_0_176=hash_control_prefix(cache, table, write_slots))
    capture = pending["capture"]
    emit(config, "postwrite_position177", postwrite_schema=POSTWRITE_SCHEMA, phase="post_target_forward",
         logical_position=177, target_layer=LAYER1, pid=os.getpid(), auth_id=auth_id,
         request_id=pending["request_id"], request_state_index=pending["request_state_index"], batch_row=0,
         actual_positions=pending["positions"], active_query=pending["active_query"],
         input_token_id=pending["active_query"][location["selected_row"]][1],
         graph_mode="FULL" if version == "v2" else "PIECEWISE", forward_id=pending["forward_id"],
         hook="postwrite_v2_full" if version == "v2" else "postwrite_v1",
         schedule_seq=pending["schedule_seq"], scheduler_output_id=None if version == "v2" else id(execution),
         descriptor=None if version == "v1" else repr(execution),
         capture_id=None if capture is None else capture["capture_id"],
         runtime_binding=binding, cache_peers=peers, cache_identity=tensor_identity(cache),
         write_slots=write_slots, block_table=table, execution_stream=pending["stream"],
         runtime_probe_changes_timing=True, blocking_device_to_cpu=True, device_wide_synchronize=False,
         capture_warmup=False,
         **location, **content, **extra)


def position_spans(raw, bytes_per_position):
    """CPU-only partition of the exact canonical component bytes; no value conversion."""
    require(isinstance(raw, bytes) and type(bytes_per_position) is int
            and bytes_per_position > 0 and len(raw) == 454 * bytes_per_position,
            "Layer-1 position byte accounting differs")
    view = memoryview(raw)
    spans = tuple(view[p * bytes_per_position:(p + 1) * bytes_per_position] for p in range(454))
    require(b"".join(spans) == raw, "Layer-1 position spans do not reconstruct prefix")
    return spans


def tensor_identity(tensor):
    """Metadata only: never copy tensor values or allocate CUDA storage at capture."""
    return dict(data_ptr=tensor.data_ptr(), storage_ptr=tensor.untyped_storage().data_ptr(),
                storage_nbytes=tensor.untyped_storage().nbytes(),
                storage_offset=tensor.storage_offset(), shape=list(tensor.shape),
                strides=list(tensor.stride()), dtype=str(tensor.dtype), device=str(tensor.device))


def history_slots(block_table, kernel_block_size, row=0, positions=range(454)):
    """CPU-testable addressing contract; gathered tables use batch rows, not state slots."""
    positions = list(positions)
    require(positions == list(range(454)), "KV incomplete historical positions")
    require(type(kernel_block_size) is int and kernel_block_size > 0, "KV invalid block size")
    table = ints(block_table[row]) if hasattr(block_table, "detach") else block_table[row]
    require(len(table) > 454 // kernel_block_size, "KV missing historical/current blocks")
    blocks = table[:454 // kernel_block_size + 1]
    require(all(type(b) is int and b >= 0 for b in blocks), "KV invalid physical block")
    slots = [table[p // kernel_block_size] * kernel_block_size + p % kernel_block_size
             for p in positions]
    require(len(set(slots)) == 454, "KV aliased historical slots")
    current = table[454 // kernel_block_size] * kernel_block_size + 454 % kernel_block_size
    require(current not in slots, "KV current write aliases history")
    return slots, current, blocks


def hash_history(cache, block_table, kernel_block_size, heads, head_dim, write_slot,
                 row=0, positions=range(454), *, localize_layer1=False):
    """Hash one layer/component at a time; CPU tensors test indexing, not CUDA behavior."""
    import torch
    import hashlib
    require(cache.dtype == torch.float16 and cache.ndim == 4
            and list(cache.shape[1:]) == [heads, kernel_block_size, 2 * head_dim],
            "KV unsupported dense FP16 cache representation")
    require(heads > 0 and head_dim > 0 and sys.byteorder == "little", "KV unsupported dimensions/endian")
    slots, current, blocks = history_slots(block_table, kernel_block_size, row, positions)
    require(type(write_slot) is int and write_slot == current, "KV current slot mapping differs")
    require(max(blocks) < cache.shape[0], "KV physical block outside layer cache")
    indices = torch.tensor(slots, device=cache.device, dtype=torch.long)
    physical, offsets = indices // kernel_block_size, indices % kernel_block_size
    digests = {}
    components = {}
    for component, start in (("K", 0), ("V", head_dim)):
        # Advanced indexing produces (logical_position, kv_head, head_dimension).
        selected = cache[physical, :, offsets, start:start + head_dim]
        raw = selected.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digests[component] = hashlib.sha256(raw).hexdigest()
        if localize_layer1:
            spans = position_spans(raw, heads * head_dim * cache.element_size())
            components[component] = dict(prefix_sha256=digests[component],
                position_sha256=[hashlib.sha256(span).hexdigest() for span in spans])
            del spans
        del raw, selected
    result = dict(coverage=[0, 454], order=["logical_position", "kv_head", "head_dimension"],
                heads=heads, head_dim=head_dim, dtype=str(cache.dtype), byte_order="little",
                elements_per_component=454 * heads * head_dim,
                bytes_per_component=454 * heads * head_dim * 2,
                kernel_block_size=kernel_block_size, physical_blocks=blocks,
                write_slot=write_slot, no_write_alias=True, digests=digests,
                cache_identity=tensor_identity(cache))
    if localize_layer1:
        result["localization"] = dict(schema=LOCALIZATION_SCHEMA, target_layer=LAYER1,
            coverage=[0, 454], position_count=454, logical_positions=list(range(454)),
            order=result["order"], heads=heads, head_dim=head_dim, dtype=result["dtype"],
            byte_order=result["byte_order"], bytes_per_position=heads * head_dim * cache.element_size(),
            **components)
    return result


def kv_layer_bindings(vllm_config, cache_config, inputs, metadata, slots, *, v1_runner=None):
    """Resolve only target decoder layers; capture invokes this without any value copies."""
    parallel = vllm_config.parallel_config
    require(all(int(getattr(parallel, k, 1)) == 1 for k in
                ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size")),
            "KV unsupported parallelism")
    require(isinstance(metadata, dict) and isinstance(slots, dict), "KV unsupported microbatch metadata")
    require(inputs.get("inputs_embeds") is None and inputs.get("intermediate_tensors") is None,
            "KV unsupported embedded/intermediate input")
    context = vllm_config.compilation_config.static_forward_context
    all_names = [name for group in cache_config.kv_cache_groups for name in group.layer_names]
    drafts = set()
    if v1_runner is not None:
        # Pinned V1 DraftModelProposer inherits this maintained set from
        # SpecDecodeBaseProposer.load_model(); never classify by name spelling.
        drafter = getattr(v1_runner, "drafter", None)
        require(v1_runner.speculative_config.uses_draft_model()
                and type(drafter).__name__ == "DraftModelProposer",
                "KV unsupported V1 draft ownership")
        drafts = getattr(drafter, "_draft_attn_layer_names", None)
        require(isinstance(drafts, set) and bool(drafts)
                and all(isinstance(name, str) for name in drafts)
                and drafts <= set(all_names), "KV missing/ambiguous draft ownership")
        grouped_drafts = [name for group in drafter.draft_attn_groups for name in group.layer_names]
        require(len(grouped_drafts) == len(set(grouped_drafts))
                and set(grouped_drafts) == drafts, "KV ambiguous draft attention groups")
        target_modules = {id(layer) for layer in v1_runner.get_model().modules()}
        draft_modules = {id(layer) for layer in v1_runner.get_draft_model().modules()}
        require(all(name in context and
                    ((id(context[name]) in draft_modules and id(context[name]) not in target_modules)
                     if name in drafts else
                     (id(context[name]) in target_modules and id(context[name]) not in draft_modules))
                    for name in all_names), "KV ambiguous target/draft model ownership")
    else:
        require(len(cache_config.kv_cache_groups) == 1, "KV unsupported target cache groups")
    names = sorted(set(all_names) - drafts)
    count = int(vllm_config.model_config.hf_text_config.num_hidden_layers)
    require(len(all_names) == len(set(all_names)) and len(names) == count
            and set(names) <= set(metadata) <= set(names) | drafts
            and set(names) <= set(slots) <= set(names) | drafts,
            "KV incomplete/ambiguous target layer set")
    require(sum(bool(set(group.layer_names) & set(names)) for group in cache_config.kv_cache_groups) == 1,
            "KV unsupported target cache groups")
    layers = {}
    for gid, group in enumerate(cache_config.kv_cache_groups):
        for name in group.layer_names:
            if name in drafts:
                continue
            layer, m = context[name], metadata[name]
            require(layer.attn_backend.get_name() == "TRITON_ATTN", "KV unsupported backend")
            impl = layer.impl
            require(impl._kv_quant_mode.name == "NONE"
                    and not impl._is_per_token_head_quant
                    and impl.kv_cache_dtype in ("auto", "float16")
                    and tuple(impl.sliding_window) == (-1, -1)
                    and impl.alibi_slopes is None and impl.sinks is None
                    and impl.chunk_lookback == -1 and not impl.use_alibi_sqrt,
                    "KV unsupported visibility/quantization")
            require(type(m.causal) is bool and m.causal and not m.use_cascade
                    and m.mm_prefix_range_tensor is None and m.rswa_prefix_lens is None,
                    "KV unsupported attention metadata")
            spec = group.kv_cache_spec
            if hasattr(spec, "kv_cache_specs"):
                spec = spec.kv_cache_specs[name]
            require(type(spec).__name__ == "FullAttentionSpec", "KV unsupported cache spec")
            cache = layer.kv_cache
            require(str(cache.dtype) == "torch.float16" and cache.ndim == 4
                    and list(cache.shape[1::2]) == [impl.num_kv_heads, 2 * impl.head_size]
                    and spec.num_kv_heads == impl.num_kv_heads
                    and spec.head_size == spec.head_size_v == impl.head_size
                    and spec.dtype == cache.dtype and cache.shape[2] % 16 == 0,
                    "KV unsupported layer cache layout")
            require(tensor_identity(m.slot_mapping) == tensor_identity(slots[name]),
                    "KV metadata/forward slot binding differs")
            layers[name] = dict(group=gid, cache=tensor_identity(cache),
                metadata={k: tensor_identity(getattr(m, k)) for k in
                          ("query_start_loc", "seq_lens", "block_table", "slot_mapping",
                           "softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum")},
                scales={k: tensor_identity(getattr(layer, k)) for k in ("_q_scale", "_k_scale", "_v_scale")},
                heads=impl.num_kv_heads, head_dim=impl.head_size, spec_block_size=spec.block_size,
                interpretation=dict(backend="TRITON_ATTN", quantization="NONE", causal=True,
                                    window=[-1, -1], use_cascade=False, scale=impl.scale,
                                    soft_cap=impl.logits_soft_cap),
                launch=dict(max_seq_len=m.max_seq_len, max_query_len=m.max_query_len,
                            num_actual_tokens=m.num_actual_tokens,
                            seq_threshold_3D=m.seq_threshold_3D))
    return dict(inputs={k: tensor_identity(inputs[k]) for k in ("input_ids", "positions")},
                layers=layers, expected_layers=names, model_layer_count=count)


def validate_full_binding(captured, runtime):
    """Compare storage consumed by FULL, allowing runtime-only Python launch scalars to differ."""
    require(captured["inputs"] == runtime["inputs"]
            and captured["expected_layers"] == runtime["expected_layers"]
            and captured["model_layer_count"] == runtime["model_layer_count"],
            "KV FULL input/layer binding differs")
    for name in captured["expected_layers"]:
        left, right = captured["layers"][name], runtime["layers"][name]
        require({k: v for k, v in left.items() if k != "launch"}
                == {k: v for k, v in right.items() if k != "launch"},
                "KV FULL capture/runtime identity mismatch: " + name)


def kv_authenticate(config):
    from experiments.protocol_b_boundary_sources import authenticate
    auth = authenticate(Path(config["directory"]).parent, installed=True)
    auth_id = fingerprint(dict(auth, pid=os.getpid()))
    if getattr(_local, "kv_auth_id", None) != auth_id:
        emit(config, "kv_sources", auth_id=auth_id, pid=os.getpid(), **auth)
        _local.kv_auth_id = auth_id
    return auth_id


def kv_capture(config, manager, desc, inputs, metadata, slots, cache_config, warmup):
    if warmup or desc.cg_mode.name != "FULL" or manager._capture_mem_samples is not None:
        return
    auth_id = kv_authenticate(config)
    bindings = kv_layer_bindings(manager.vllm_config, cache_config, inputs, metadata, slots)
    require(all(x["cache"]["device"].startswith("cuda") for x in bindings["layers"].values()),
            "KV capture requires CUDA layer storage")
    captures = getattr(_local, "kv_captures", {})
    key = (id(manager), repr(desc))
    require(key not in captures, "KV duplicate capture descriptor")
    record = dict(capture_id=uuid.uuid4().hex, descriptor=repr(desc), bindings=bindings,
                  auth_id=auth_id, pid=os.getpid())
    captures[key] = record
    _local.kv_captures = captures
    emit(config, "kv_capture", **record)


def kv_v2_prepare(config, runner, batch, desc, inputs, metadata, slots):
    _local.kv_pending = None  # Never reuse a preceding step's mutable views.
    postwrite_reset(config, "v2")
    ids = list(batch.req_ids or ())
    matching = [rid for rid in ids if rid in runner.req_states.req_id_to_index
                and int(runner.req_states.prompt_len.np[runner.req_states.req_id_to_index[rid]])
                == len(config["prompt_token_ids"])]
    if not matching:
        return
    # Position check is from the actual device input, not optimistic CPU counters.
    positions = ints(batch.positions[:batch.num_tokens])
    if postwrite_enabled(config) and 177 in positions:
        require(config["lane"] == "vanilla" and len(ids) == len(matching) == batch.num_reqs == 1,
                "Post-write ambiguous Vanilla request")
        index = runner.req_states.req_id_to_index[ids[0]]
        if ints(runner.req_states.all_token_ids.gpu[index, :176]) != config["prompt_token_ids"]:
            return
        require(batch.num_draft_tokens == 0 and not batch.is_prefilling_np[0]
                and list(batch.idx_mapping_np) == ints(batch.idx_mapping) == [index]
                and type(runner.model_state).__name__ == "DefaultModelState" and runner.pcp_manager is None,
                "Post-write V2 ownership/decode differs")
        postwrite_prepare(config, runner, inputs, metadata, slots, positions, ids[0], index,
                          desc.cg_mode, int(desc.num_tokens), manager=runner.cudagraph_manager, desc=desc)
    if 454 not in positions:
        return
    require(config["lane"] == "vanilla" and len(ids) == len(matching) == batch.num_reqs == 1,
            "KV ambiguous V2 request")
    index = runner.req_states.req_id_to_index[ids[0]]
    if ints(runner.req_states.all_token_ids.gpu[index, :len(config["prompt_token_ids"])]) != config["prompt_token_ids"]:
        return  # Another conditioning request can have the same prompt length.
    require(batch.num_draft_tokens == 0 and not batch.is_prefilling_np[0]
            and list(batch.idx_mapping_np) == ints(batch.idx_mapping) == [index],
            "KV V2 request-state mapping differs")
    require(positions == [454] and ints(batch.input_ids[:batch.num_tokens]) == [128367],
            "KV unexpected boundary query")
    require(desc.cg_mode.name in ("FULL", "PIECEWISE") and desc.num_ubatches == 1,
            "KV unsupported dispatched graph mode/microbatch")
    require(type(runner.model_state).__name__ == "DefaultModelState"
            and runner.pcp_manager is None, "KV unsupported V2 model state")
    _local.kv_pending = (runner, batch, desc, inputs, metadata, slots, ids[0], index)


def kv_v2_observe(config, manager, desc=None):
    pending = getattr(_local, "kv_pending", None)
    _local.kv_pending = None
    if pending is None:
        return
    runner, batch, prepared_desc, inputs, metadata, slots, rid, index = pending
    require(runner.cudagraph_manager is manager, "KV stale/wrong FULL manager")
    mode = prepared_desc.cg_mode.name
    require((mode == "FULL" and desc == prepared_desc) or (mode == "PIECEWISE" and desc is None),
            "KV dispatched hook differs")
    binding = kv_layer_bindings(runner.vllm_config, runner.kv_cache_config, inputs, metadata, slots)
    capture = None
    if mode == "FULL":
        capture = getattr(_local, "kv_captures", {}).get((id(manager), repr(desc)))
        require(capture is not None, "KV missing FULL capture binding")
        validate_full_binding(capture["bindings"], binding)
    kv_snapshot(config, runner, inputs, metadata, slots, rid, index, mode, binding, capture)


def kv_v1(config, runner, scheduled, input_ids, positions, metadata, slots, mode):
    postwrite_reset(config, "v1")
    ids = list(runner.input_batch.req_ids[:runner.input_batch.num_reqs])
    matching = [rid for rid in ids if list(runner.requests[rid].prompt_token_ids or [])
                == config["prompt_token_ids"]]
    if not matching:
        return
    n = scheduled.total_num_scheduled_tokens
    actual_positions = ints(positions[:n])
    early_prepare(config, runner, scheduled, input_ids, actual_positions)
    control_query(config, runner, scheduled, actual_positions)
    if postwrite_enabled(config) and 177 in actual_positions:
        require(config["lane"] == "adaptive" and len(ids) == len(matching) == 1, "Post-write ambiguous Adaptive batch")
        postwrite_prepare(config, runner, dict(input_ids=input_ids, positions=positions), metadata, slots,
                          actual_positions, ids[0], 0, mode, int(positions.shape[0]), scheduled=scheduled)
    if control_settings(config) is not None or attention_settings(config) is not None:
        return  # Control owns its P177/prefix contract, never the canonical boundary454 gate.
    if 454 not in actual_positions:
        return
    require(config["lane"] == "adaptive" and len(ids) == len(matching) == 1
            and n == 1 and actual_positions == [454] and ints(input_ids[:n]) == [128367]
            and not scheduled.scheduled_spec_decode_tokens, "KV unexpected V1 boundary query")
    # The approved Adaptive execution is PIECEWISE. FULL requires its own capture binding.
    require(mode.name == "PIECEWISE", "KV unsupported V1 dispatched mode")
    inputs = dict(input_ids=input_ids, positions=positions)
    binding = kv_layer_bindings(runner.vllm_config, runner.kv_cache_config, inputs, metadata, slots,
                                v1_runner=runner)
    kv_snapshot(config, runner, inputs, metadata, slots, ids[0], 0, mode.name, binding, None)


def kv_snapshot(config, runner, inputs, metadata, slots, rid, index, mode, binding, capture):
    auth_id = kv_authenticate(config)
    require(len(config["prompt_token_ids"]) == 176, "KV boundary prompt length changed")
    layers = {}
    localize = config.get("layer1_localization") is True
    require(not localize or LAYER1 in binding["expected_layers"], "Layer-1 target ownership missing")
    context = runner.vllm_config.compilation_config.static_forward_context
    for name in binding["expected_layers"]:
        layer, m = context[name], metadata[name]
        require(layer.kv_cache.device.type == "cuda", "KV runtime requires CUDA layer storage")
        require(ints(m.query_start_loc[:2]) == [0, 1] and ints(m.seq_lens[:1]) == [455],
                "KV effective query/sequence length differs")
        info = binding["layers"][name]
        scales = {k: ints(getattr(layer, k).reshape(-1)) for k in ("_q_scale", "_k_scale", "_v_scale")}
        require(all(v == [1.0] for v in scales.values()), "KV unsupported nonunit dense scales")
        write_slot, = ints(slots[name][:1])
        extracted = hash_history(layer.kv_cache, m.block_table, layer.kv_cache.shape[2],
                                 info["heads"], info["head_dim"], write_slot,
                                 localize_layer1=localize and name == LAYER1)
        layers[name] = dict(extracted, group=info["group"], spec_block_size=info["spec_block_size"],
                            interpretation=dict(info["interpretation"], scales=scales))
    # At FULL, include capture-bound launch scalars; runtime scalars are not claimed consumed.
    emit(config, "kv_history", history_schema="m203.boundary.kv.v1",
         snapshot_id=uuid.uuid4().hex, pid=os.getpid(), auth_id=auth_id,
         request_id=rid, request_state_index=int(index), batch_row=0,
         active_query=[[454, 128367]], position=454, generated_index=279,
         dispatched_graph_mode=mode, descriptor=None if capture is None else capture["descriptor"],
         capture_id=None if capture is None else capture["capture_id"],
         runtime_binding=binding, expected_layers=binding["expected_layers"],
         model_layer_count=binding["model_layer_count"], layers=layers,
         visibility=dict(query_start_loc=[0, 1], seq_len=455, historical_positions=[0, 454]),
         runtime_probe_changes_timing=True,
         copies=dict(device_to_cpu="metadata/scales; one historical layer/component at a time",
                     host_to_device="454 int64 slot indices per target layer",
                     device_gather="one historical K or V component into temporary storage",
                     cpu_localization="layer 1 only: span reconstruction and 454 SHA-256 per component"
                     if localize else None,
                     synchronization="blocking Tensor.cpu() on observation stream",
                     device_wide_synchronize=False), phase="pre_forward")


def map_rows(prompt_len, positions, indices, draft_count, target_rows, bonus_rows):
    """Archive mapping: generated index = actual model position - prompt + 1."""
    require(type(prompt_len) is int and prompt_len > 0, "Invalid prompt length")
    require(type(draft_count) is int and 0 <= draft_count <= 4
            and len(indices) == draft_count + 1, "Draft/row count mismatch")
    require(all(type(p) is int and p >= 0 for p in positions), "Invalid model positions")
    require(all(type(i) is int and 0 <= i < len(positions) for i in indices)
            and len(set(indices)) == len(indices), "Invalid logits indices")
    require(target_rows == list(range(draft_count)) and bonus_rows == [draft_count],
            "Ambiguous row roles")
    generated = [positions[i] - prompt_len + 1 for i in indices]
    require(all(b == a + 1 for a, b in zip(generated, generated[1:])),
            "Noncontiguous generated indices")
    return [dict(row=j, flat_index=i, position=positions[i], generated_index=generated[j],
                 role="bonus" if j == draft_count else "draft_verification")
            for j, i in enumerate(indices)]


def selected_tokens(ids):
    selected, padded = [], False
    for token in ids:
        require(type(token) is int, "Invalid sampled ID")
        if token == -1:
            padded = True
        else:
            require(token >= 0 and not padded, "Invalid sampler padding")
            selected.append(token)
    return selected


def ints(tensor):
    return tensor.detach().cpu().tolist()


def emit(config, kind, **fields):
    root = Path(config["directory"]) / "observer"
    write_new(root / (uuid.uuid4().hex + ".json"),
              dict(schema="m203.boundary.event.v1", kind=kind, timestamp=utc_now(),
                   lane=config["lane"], execution_sha=config["execution_sha"],
                   config_fingerprint=fingerprint(config), **fields))


def score(config, key, stage, vector):
    import numpy as np
    import hashlib
    raw = vector.detach().float().cpu().contiguous().numpy().copy()
    require(raw.ndim == 1 and len(raw) > 128273 and not np.isnan(raw).any()
            and not np.isposinf(raw).any() and np.isfinite(raw).any(), "Invalid scores")
    # -inf is valid for masked processed scores.
    name = key + "-" + stage + ".npy"
    with (Path(config["directory"]) / "observer" / name).open("xb") as handle:
        np.save(handle, raw, allow_pickle=False)
    top = np.argsort(-raw, kind="stable")[:10].tolist()
    candidates = list(dict.fromkeys(top + [79126, 128273]))
    return dict(file=name, sha256=hashlib.sha256(
        (Path(config["directory"]) / "observer" / name).read_bytes()).hexdigest(),
        argmax=int(np.argmax(raw)), shape=list(raw.shape), source_dtype=str(vector.dtype),
        candidates=[dict(token_id=i, score=float(raw[i]) if np.isfinite(raw[i]) else "-inf")
                    for i in candidates])


def begin(config, runner, scheduled, logits, spec, grammar):
    _local.context = None
    batch = runner.input_batch
    ids = list(batch.req_ids[:int(batch.num_reqs)])
    matching = [rid for rid in ids if list(runner.requests[rid].prompt_token_ids or [])
                == config["prompt_token_ids"]]
    if not matching:
        return
    require(len(ids) == len(matching) == 1, "Ambiguous target batch")
    require(grammar is None and not getattr(batch, "has_structured_output_reqs", False),
            "Target grammar unsupported")
    require(not runner.uses_mrope and not runner.enable_prompt_embeds, "Nonstandard positions")
    require(all(int(getattr(runner.vllm_config.parallel_config, k, 1)) == 1 for k in
                ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size")), "Parallel target unsupported")
    lo, hi = map(int, runner.query_start_loc.np[:2])
    require(lo == 0 and hi == scheduled.total_num_scheduled_tokens and hi > 0,
            "Ambiguous query bounds")
    state = early_state(config)
    early = None if state is None else state["steps"].get(id(scheduled))
    positions = early["positions"] if early is not None and "positions" in early else ints(runner.positions[:hi])
    if spec is None:
        n, indices, targets, bonus = 0, [hi - 1], [], [0]
    else:
        require(len(spec.num_draft_tokens) == 1, "Ambiguous drafts")
        n = int(spec.num_draft_tokens[0])
        indices, targets, bonus = map(ints, (spec.logits_indices, spec.target_logits_indices,
                                           spec.bonus_logits_indices))
        require(ints(spec.cu_num_sampled_tokens) == [n + 1], "Invalid cumulative rows")
    rows = map_rows(len(config["prompt_token_ids"]), positions, indices, n, targets, bonus)
    require(logits is not None and logits.ndim == 2 and logits.shape[0] == len(rows),
            "Missing/misaligned logits")
    early_verify(config, runner, scheduled, rows, spec)
    if control_settings(config) is not None or attention_settings(config) is not None:
        return
    target = [r for r in rows if r["generated_index"] == 279]
    if not target:
        return
    require(len(target) == 1, "Ambiguous target row")
    require(not bool(getattr(getattr(runner, "discard_request_mask", None), "np", [False])[0]),
            "Discarded prefill row")
    query = list(map(list, zip(positions, ints(runner.input_ids.gpu[:hi]))))
    require(len({p for p, _ in query}) == len(query)
            and all(p >= 0 and t >= 0 for p, t in query), "Invalid actual model inputs")
    row = target[0]
    key = uuid.uuid4().hex
    record = dict(key=key, pid=os.getpid(), request_id=ids[0], prompt_token_ids=config["prompt_token_ids"],
                  active_query=query, query_bounds=[lo, hi], logits_indices=indices, rows=rows,
                  target_row=row, draft_count=n, decoding_path="ordinary" if spec is None else "speculative",
                  draft_token_ids=[] if spec is None else ints(spec.draft_token_ids),
                  async_scheduling=bool(runner.use_async_scheduling),
                  prefix_cache_enabled=bool(runner.cache_config.enable_prefix_caching),
                  resolved_cudagraph_mode=str(runner.vllm_config.compilation_config.cudagraph_mode),
                  same_kv_cache_state_proven=False, runtime_probe_changes_timing=True,
                  raw=score(config, key, "raw", logits[row["row"]]))
    _local.context = (config, record)


def processed(stage, logits, metadata, spec=None):
    context = getattr(_local, "context", None)
    if context is None:
        return
    config, record = context
    require(metadata.all_greedy is True, "Only frozen greedy sampling supported")
    n = record["draft_count"]
    indices = list(range(n)) if stage == "verify_scores" else [n]
    require(logits.ndim == 2 and logits.shape[0] == len(indices), "Processed row mismatch")
    if spec is not None:
        require(ints(spec.target_logits_indices) == indices, "Processed role mismatch")
    j = record["target_row"]["row"]
    if j in indices:
        require("processed" not in record, "Duplicate processed scores")
        record["processed"] = score(config, record["key"], "processed", logits[indices.index(j)])
        record["processed_capture_point"] = stage


def end(config, runner, output):
    context = getattr(_local, "context", None)
    _local.context = None
    if context is None:
        return
    _, record = context
    require(output.sampled_token_ids.ndim == 2 and output.sampled_token_ids.shape[0] == 1,
            "Ambiguous sampled output")
    selected = selected_tokens(ints(output.sampled_token_ids[0]))
    require(0 < len(selected) <= len(record["rows"]), "Invalid sampled count")
    record.update(selected=selected, accepted_drafts=len(selected) - 1,
                  rejected_drafts=record["draft_count"] - len(selected) + 1)
    if record["target_row"]["row"] >= len(selected):
        emit(config, "rejected_candidate", **record)
        return
    require("processed" in record, "Missing processed logits")
    emit(config, "forward", **record)


def v2_begin(config, sampler, batch, logits):
    """V2 ordinary Vanilla only; snapshot actual inputs, never retain tensors."""
    _local.v2_context = None
    states = sampler.req_states
    # V2 profiles its sampler with a dummy batch before adding any requests.
    if not states.req_id_to_index:
        return
    ids = list(batch.req_ids or ())
    if not ids or not any(rid in states.req_id_to_index for rid in ids):
        return
    slots = [int(i) for i in batch.idx_mapping_np]
    require(len(ids) == len(slots) == batch.num_reqs
            and all(states.req_id_to_index.get(rid) == i for rid, i in zip(ids, slots)),
            "V2 request/idx mapping differs")
    matching = [rid for rid, i in zip(ids, slots)
                if int(states.prompt_len.np[i]) == len(config["prompt_token_ids"])
                and ints(states.all_token_ids.gpu[i, :len(config["prompt_token_ids"])])
                == config["prompt_token_ids"]]
    if not matching:
        return
    require(config["lane"] == "vanilla" and len(ids) == len(matching) == 1,
            "V2 requires single-request Vanilla batch")
    require(not batch.has_structured_output_reqs and batch.num_draft_tokens == 0,
            "V2 grammar/speculative path unsupported")
    require(ints(batch.idx_mapping) == slots and ints(batch.expanded_idx_mapping) == slots,
            "V2 GPU idx mapping differs")
    require(list(batch.cu_num_logits_np) == ints(batch.cu_num_logits) == [0, 1],
            "V2 ambiguous logits ownership")
    bounds = [int(i) for i in batch.query_start_loc_np[:2]]
    require(bounds == ints(batch.query_start_loc[:2]) and bounds == [0, batch.num_tokens]
            and batch.num_tokens > 0, "V2 ambiguous query bounds")
    positions = ints(batch.positions[:batch.num_tokens])
    indices = ints(batch.logits_indices)
    rows = map_rows(len(config["prompt_token_ids"]), positions, indices, 0, [], [0])
    require(logits.ndim == 2 and logits.shape[0] == len(rows), "V2 logits row mismatch")
    target = [r for r in rows if r["generated_index"] == 279]
    if not target:
        return
    require(len(target) == 1 and not bool(batch.is_prefilling_np[0]),
            "V2 missing/ambiguous decode row")
    require(float(sampler.sampling_states.temperature.np[slots[0]]) == 0.0,
            "V2 requires frozen greedy sampling")
    query = list(map(list, zip(positions, ints(batch.input_ids[:batch.num_tokens]))))
    require(len({p for p, _ in query}) == len(query)
            and all(p >= 0 and t >= 0 for p, t in query), "V2 invalid actual model inputs")
    key = uuid.uuid4().hex
    record = dict(key=key, pid=os.getpid(), request_id=ids[0], prompt_token_ids=list(config["prompt_token_ids"]),
                  active_query=query, query_bounds=bounds, logits_indices=indices, rows=rows,
                  target_row=target[0], draft_count=0, draft_token_ids=[], decoding_path="ordinary",
                  model_runner="v2", req_ids=ids, idx_mapping=slots,
                  expanded_idx_mapping=ints(batch.expanded_idx_mapping), cu_num_logits=[0, 1],
                  same_kv_cache_state_proven=False, runtime_probe_changes_timing=True,
                  raw=score(config, key, "raw", logits[target[0]["row"]]))
    _local.v2_context = (config, record)


def v2_processed(sampler, logits, expanded_idx_mapping, idx_mapping_np, pos):
    context = getattr(_local, "v2_context", None)
    if context is None:
        return
    config, record = context
    require([int(i) for i in idx_mapping_np] == record["idx_mapping"]
            and ints(expanded_idx_mapping) == record["expanded_idx_mapping"]
            and ints(pos) == [r["position"] for r in record["rows"]],
            "V2 processed row binding differs")
    require(all(float(sampler.sampling_states.temperature.np[i]) == 0.0
                for i in record["idx_mapping"]), "V2 sampling changed")
    require(logits.ndim == 2 and logits.shape[0] == len(record["rows"])
            and "processed" not in record, "V2 missing/duplicate processed row")
    record["processed"] = score(config, record["key"], "processed", logits[record["target_row"]["row"]])
    record["processed_capture_point"] = "v2_pre_gumbel_after_top_k_top_p"


def v2_end(output):
    context = getattr(_local, "v2_context", None)
    _local.v2_context = None
    if context is None:
        return
    config, record = context
    require("processed" in record, "V2 unsupported sampling path: missing pre-gumbel scores")
    sampled = ints(output.sampled_token_ids)
    num_sampled, num_rejected = ints(output.num_sampled), ints(output.num_rejected)
    require(output.sampled_token_ids.ndim == 2 and len(sampled) == 1
            and len(sampled[0]) == 1 and num_sampled == [1] and num_rejected == [0],
            "V2 invalid sampled output/counts")
    selected = selected_tokens(sampled[0])
    require(len(selected) == 1, "V2 missing selected token")
    record.update(selected=selected, sampled_token_ids=sampled, num_sampled=num_sampled,
                  num_rejected=num_rejected, accepted_drafts=0, rejected_drafts=0)
    emit(config, "forward", **record)


def accounting(config, stage, request, output, scheduled, stale, selected=None):
    if list(request.prompt_token_ids or []) != config["prompt_token_ids"]:
        return
    early_account(config, stage, request, output, scheduled, stale, selected)
    count = len(request._output_token_ids)
    if 267 <= count <= 291:
        emit(config, stage, request_id=request.request_id, output_count=count,
             committed_ids=list(request._output_token_ids), output_token_ids=list(output),
             stale=stale, scheduled_tokens=scheduled.num_scheduled_tokens.get(request.request_id),
             scheduled_spec_token_ids=list(scheduled.scheduled_spec_decode_tokens.get(request.request_id, [])),
             num_computed_tokens=request.num_computed_tokens,
             num_in_flight_tokens=request.num_in_flight_tokens,
             num_output_placeholders=request.num_output_placeholders)


def observe(stage, *args):
    path = os.environ.get("PROTOCOL_B_BOUNDARY_CONFIG")
    if not path:
        return
    config = None
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        require(isinstance(config, dict) and all(key in config for key in
                ("directory", "lane", "execution_sha", "prompt_token_ids")),
                "Incomplete observer config")
        if stage == "early_schedule":
            early_schedule(config, *args)
        elif stage == "early_produced":
            early_produced(config, *args)
        elif stage == "postwrite_v2_full":
            postwrite_finish(config, "v2", *args)
        elif stage == "postwrite_v1":
            postwrite_finish(config, "v1", *args)
        elif stage.startswith("kv_"):
            if config.get("kv_probe") is not True:
                return
            if stage == "kv_capture":
                kv_capture(config, *args)
            elif stage == "kv_v2_prepare":
                kv_v2_prepare(config, *args)
            elif stage == "kv_v2_full":
                kv_v2_observe(config, *args)
            elif stage == "kv_v2_piecewise":
                kv_v2_observe(config, *args)
            elif stage == "kv_v1":
                kv_v1(config, *args)
            else:
                require(False, "Unknown KV stage")
        elif stage == "begin":
            begin(config, *args)
        elif stage in ("sample_scores", "verify_scores"):
            processed(stage, *args)
        elif stage == "end":
            end(config, *args)
        elif stage == "v2_begin":
            v2_begin(config, *args)
        elif stage == "v2_processed":
            v2_processed(*args)
        elif stage == "v2_end":
            v2_end(*args)
        elif stage in ("account_before", "account_after"):
            accounting(config, stage, *args)
        elif stage == "cache_lookup":
            request, local, external = args
            if list(request.prompt_token_ids or []) == config["prompt_token_ids"]:
                emit(config, stage, request_id=request.request_id,
                     cached_prefix_tokens=int(local), external_cached_tokens=int(external),
                     prompt_length=len(request.prompt_token_ids),
                     num_computed_tokens=request.num_computed_tokens)
    except Exception as exc:
        # A failed observer never becomes usable evidence or changes a result.
        if config and config.get("postwrite_position177") is True:
            for version in ("v1", "v2"):
                setattr(_local, "postwrite_" + version + "_pending", None)
                setattr(_local, "postwrite_" + version + "_phase", "failed")
        if stage.startswith("kv_"):
            _local.kv_pending = None
        if stage.startswith("v2_"):
            _local.v2_context = None
        else:
            _local.context = None
        error = f"{type(exc).__name__}: {exc}"
        try:
            emit(config, "observer_error", stage=stage, error=error)
        except Exception:
            # Config identity may be unavailable. Retain an unqualified error
            # beside the configured lane; the analyzer cannot accept this event.
            evidence = dict(schema="m203.boundary.event.v1", kind="observer_error",
                            timestamp=utc_now(), stage=stage, error=error, config_path=path)
            try:
                write_new(Path(path).parent / "observer" / (uuid.uuid4().hex + ".json"), evidence)
            except Exception:
                # Missing/unwritable evidence is still visible in the server log.
                try:
                    print(json.dumps(evidence), file=sys.stderr, flush=True)
                except Exception:
                    pass  # A broken diagnostic log must not interrupt inference.
