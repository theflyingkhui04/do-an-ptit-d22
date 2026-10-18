"""Read-only vLLM 0.30.0 observer, narrowly ported from archive a7b52e7.

Reuses m2_03_targeted/observer.py row/score/accounting hooks and A3/V8's
actual-position mapping. No cache, scheduler, graph or sampling interventions.
Errors are retained as evidence; inference results are never replaced.
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
    ids = list(batch.req_ids or ())
    matching = [rid for rid in ids if rid in runner.req_states.req_id_to_index
                and int(runner.req_states.prompt_len.np[runner.req_states.req_id_to_index[rid]])
                == len(config["prompt_token_ids"])]
    if not matching:
        return
    # Position check is from the actual device input, not optimistic CPU counters.
    positions = ints(batch.positions[:batch.num_tokens])
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
    ids = list(runner.input_batch.req_ids[:runner.input_batch.num_reqs])
    matching = [rid for rid in ids if list(runner.requests[rid].prompt_token_ids or [])
                == config["prompt_token_ids"]]
    if not matching:
        return
    n = scheduled.total_num_scheduled_tokens
    if 454 not in ints(positions[:n]):
        return
    require(config["lane"] == "adaptive" and len(ids) == len(matching) == 1
            and n == 1 and ints(positions[:n]) == [454] and ints(input_ids[:n]) == [128367]
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
    positions = ints(runner.positions[:hi])
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


def accounting(config, stage, request, output, scheduled, stale):
    if list(request.prompt_token_ids or []) != config["prompt_token_ids"]:
        return
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
        if stage.startswith("kv_"):
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
