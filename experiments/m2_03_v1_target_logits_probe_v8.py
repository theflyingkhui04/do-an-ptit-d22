"""M2-03 A3 v8. Exact vLLM 0.30.0 V1 runner diagnostic with async scheduling.

Read-only observer. Captures a raw-logits candidate by ABSOLUTE MODEL POSITION,
then retains it only if rejection sampling actually selected that output row.
Never trusts optimistic CachedRequestState.output_token_ids (-1 placeholders).

Evidence is NOT established until the separate offline comparator checks the
full HTTP output-token prefix against the exact active model-forward inputs.

Probe exceptions are logged and suppressed so the diagnostic cannot kill an
otherwise functioning EngineCore. All such events count as missing evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from experiments.m2_03_target_logits_probe import (
    _load_runtime_config,
    _new_output_dir,
    token_ids_sha256,
)

SUPPORTED_VERSION = "0.30.0"


class V8ProbeSkip(RuntimeError):
    """Skip invalid evidence; never modify serving model behavior."""


@dataclass(frozen=True)
class CandidateV8:
    config: dict
    req_id: str
    prompt_ids: tuple[int, ...]
    generated_base: int
    offset: int
    row_count: int
    num_draft_tokens: int
    input_position: int
    input_flat_index: int
    query_inputs: tuple[tuple[int, int], ...]
    source_dtype: str
    raw_logits: np.ndarray
    prefix_cache_enabled: bool
    async_scheduling: bool
    req_output_tokens_observed: int
    req_has_optimistic_placeholders: bool


def _ints(tensor: torch.Tensor) -> list[int]:
    return [int(x) for x in tensor.detach().cpu().tolist()]


def _prepare(
    runner: object,
    scheduler_output: object,
    logits: torch.Tensor | None,
    spec_decode_metadata: object | None,
    grammar_output: object | None,
) -> CandidateV8 | None:
    config = _load_runtime_config()
    if config is None:
        return None
    if importlib.metadata.version("vllm") != SUPPORTED_VERSION:
        raise V8ProbeSkip("vLLM version differs from pinned 0.30.0")
    if config.get("lane") != "adaptive":
        raise V8ProbeSkip("V8 is only for adaptive V1; Vanilla V2 unchanged")

    batch = runner.input_batch
    num_reqs = int(batch.num_reqs)
    if num_reqs == 0:
        return None
    ids = list(batch.req_ids[:num_reqs])
    if ids and all(isinstance(x, str) and x.startswith(("_warmup_", "_v2_mixed_warmup")) for x in ids):
        return None
    if num_reqs != 1 or len(ids) != 1 or not isinstance(ids[0], str) or not ids[0]:
        raise V8ProbeSkip("Expected one real request, c1; refusing ambiguous batch")
    req_id = ids[0]
    if grammar_output is not None or getattr(batch, "has_structured_output_reqs", False):
        raise V8ProbeSkip("Grammar-constrained request unsupported")
    if runner.uses_mrope or runner.enable_prompt_embeds:
        raise V8ProbeSkip("Only text-input with standard positions supported")
    parallel = runner.vllm_config.parallel_config
    if any(int(getattr(parallel, x, 1)) != 1 for x in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "decode_context_parallel_size",
    )):
        raise V8ProbeSkip("Requires TP=PP=DP=DCP=1")
    req = runner.requests.get(req_id)
    if req is None or req.prompt_token_ids is None:
        raise V8ProbeSkip("V1 request state or prompt token IDs missing")
    prompt = tuple(int(t) for t in req.prompt_token_ids)
    if not prompt or any(t < 0 for t in prompt):
        raise V8ProbeSkip("Invalid prompt IDs")

    if logits is None or logits.ndim != 2:
        raise V8ProbeSkip("Logits missing/not 2D")
    if spec_decode_metadata is None:
        num_draft = 0
        flat_rows = [int(runner.query_start_loc.np[1]) - 1]
    else:
        drafts = list(map(int, spec_decode_metadata.num_draft_tokens))
        if len(drafts) != 1 or drafts[0] < 0:
            raise V8ProbeSkip("Invalid V1 draft metadata")
        num_draft = drafts[0]
        flat_rows = _ints(spec_decode_metadata.logits_indices)
        if _ints(spec_decode_metadata.cu_num_sampled_tokens) != [num_draft + 1]:
            raise V8ProbeSkip("Invalid sampled-token cumulative mapping")
    rows = num_draft + 1
    if len(flat_rows) != rows or logits.shape[0] != rows:
        raise V8ProbeSkip("Logits/draft/position row counts differ")

    lo, hi = int(runner.query_start_loc.np[0]), int(runner.query_start_loc.np[1])
    if lo != 0 or hi <= lo or hi != int(scheduler_output.total_num_scheduled_tokens):
        raise V8ProbeSkip("Unexpected active single-request query bounds")
    if any(not (lo <= j < hi) for j in flat_rows):
        raise V8ProbeSkip("Logits row index outside query bounds")

    # IMPORTANT: positions reflect what actually entered this model forward.
    # Do not use len(req.output_token_ids); async spec decode can append -1.
    row_positions = _ints(runner.positions[torch.tensor(flat_rows, device=runner.positions.device)])
    generated = [p - len(prompt) + 1 for p in row_positions]
    if any(b - a != 1 for a, b in zip(generated, generated[1:])):
        raise V8ProbeSkip(f"Noncontiguous generated indices in V1 rows: {generated}")
    target = int(config["target"])
    matches = [idx for idx, x in enumerate(generated) if x == target]
    if not matches:
        return None
    if len(matches) != 1 or target < 0:
        raise V8ProbeSkip("Ambiguous target row")
    j = matches[0]
    if generated[0] < 0:
        raise V8ProbeSkip("Target was mapped to prefill-only position")
    if getattr(runner, "discard_request_mask", None) is not None:
        if bool(runner.discard_request_mask.np[0]):
            raise V8ProbeSkip("Discarded chunked-prefill sampler output")
    # Current query input_ids can include GPU-copied previous sampled IDs and
    # speculative drafts. Copy the TRUE forward inputs, not stale CPU caches.
    p = _ints(runner.positions[lo:hi])
    tokens = _ints(runner.input_ids.gpu[lo:hi])
    if len(p) != hi - lo or len(tokens) != hi - lo:
        raise V8ProbeSkip("Model query tensors differ in length")
    active = tuple((position, tok) for position, tok in zip(p, tokens)
                   if position <= row_positions[j])
    if (row_positions[j], tokens[flat_rows[j] - lo]) not in active:
        raise V8ProbeSkip("Selected active conditioning token not present")
    if any(pos < 0 or tok < 0 for pos, tok in active):
        raise V8ProbeSkip("Invalid active model input position or token")
    if len({position for position, _ in active}) != len(active):
        raise V8ProbeSkip("Duplicate active model input positions")

    raw = logits[j].detach().float().cpu().contiguous().numpy().copy()
    if raw.ndim != 1 or raw.size < 2 or not np.isfinite(raw).all():
        raise V8ProbeSkip("Nonfinite/invalid raw logits")
    cached_output = list(req.output_token_ids)
    print(f"[M2-03 A3 V8 PRE] req={req_id} target={target} base={generated[0]} "
          f"offset={j} rowpos={row_positions[j]} async={bool(runner.use_async_scheduling)} "
          f"cached_output_len={len(cached_output)} placeholders={-1 in cached_output}",
          flush=True)
    return CandidateV8(
        config=config, req_id=req_id, prompt_ids=prompt,
        generated_base=generated[0], offset=j, row_count=rows,
        num_draft_tokens=num_draft, input_position=row_positions[j],
        input_flat_index=flat_rows[j], query_inputs=active,
        source_dtype=str(logits.dtype), raw_logits=raw,
        prefix_cache_enabled=bool(runner.cache_config.enable_prefix_caching),
        async_scheduling=bool(runner.use_async_scheduling),
        req_output_tokens_observed=len(cached_output),
        req_has_optimistic_placeholders=-1 in cached_output,
    )


def _finalize(runner: object, pending: CandidateV8 | None, sampler_output: object) -> dict | None:
    if pending is None:
        return None
    output = sampler_output.sampled_token_ids
    if output.ndim != 2 or output.shape[0] != 1:
        raise V8ProbeSkip("Invalid c1 sampler shape")
    sampled = _ints(output[0])
    valid: list[int] = []
    padding = False
    for token in sampled:
        if token == -1:
            padding = True
        elif padding or token < 0:
            raise V8ProbeSkip("Noncontiguous sampler padding")
        else:
            valid.append(token)
    if len(valid) > pending.row_count:
        raise V8ProbeSkip("Sampler produced more tokens than logits rows")
    j = pending.offset
    if j >= len(valid):
        print(f"[M2-03 A3 V8 REJECTED_EARLY] req={pending.req_id} "
              f"offset={j} accepted_in_step={len(valid)}", flush=True)
        return None

    # For any earlier accepted draft token i in this same speculative step,
    # forward input at position prompt_len + base + i MUST equal selected[i].
    # This uses true GPU model inputs, not scheduled/optimistic token arrays.
    query = dict(pending.query_inputs)
    for i, tok in enumerate(valid[:j]):
        position = len(pending.prompt_ids) + pending.generated_base + i
        if query.get(position) != tok:
            raise V8ProbeSkip(f"Accepted draft did not match GPU model input at {position}")
    if pending.generated_base + j != int(pending.config["target"]):
        raise V8ProbeSkip("Selected sampled row is not the generated target index")
    if pending.input_position != len(pending.prompt_ids) + int(pending.config["target"]) - 1:
        raise V8ProbeSkip("Absolute target model position mismatch")

    root = Path(pending.config["output_dir"]).expanduser().resolve()
    if any(x in root.parts for x in ("results", "m2-03-20261013-r03")):
        raise V8ProbeSkip("Refusing to write canonical M2-03 evidence")
    raw = pending.raw_logits
    top1 = int(np.argmax(raw))
    remaining = raw.copy()
    remaining[top1] = -np.inf
    top2 = int(np.argmax(remaining))
    key = hashlib.sha256(pending.req_id.encode("utf-8")).hexdigest()[:20]
    target = int(pending.config["target"])
    vector_name = f"adaptive_{key}_g{target}_p{pending.input_position}.npy"
    record = {
        "schema": "m2-03.a3.raw-target-logits.v1",
        "instrumentation_version": "v1-runner-a3-v8-async",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": pending.config["git_sha"],
        "vllm_version": SUPPORTED_VERSION,
        "config_source": pending.config["source"],
        "lane": "adaptive",
        "request_id": pending.req_id,
        "model_runner_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "prompt_token_count": len(pending.prompt_ids),
        "prompt_token_ids": list(pending.prompt_ids),
        "prompt_token_ids_sha256": token_ids_sha256(list(pending.prompt_ids)),
        "target_generated_index_zero_based": target,
        "model_position_zero_based": pending.input_position,
        "generated_base_index_from_forward_positions": pending.generated_base,
        "logits_row_index_in_forward": j,
        "selected_model_input_flat_index": pending.input_flat_index,
        "logits_row_range_for_request": [0, pending.row_count],
        "num_draft_tokens_in_batch": pending.num_draft_tokens,
        "row_role": "bonus" if j == pending.num_draft_tokens else "draft_verification",
        "logits_kind": "raw_target_before_grammar_and_sampling_processors",
        "original_logits_dtype": pending.source_dtype,
        "prefix_proof_status": "PENDING_API_FULL_PREFIX_VERIFICATION",
        "conditioning_token_prefix_sha256": None,
        "generated_prefix_token_ids_sha256": None,
        "active_query_input_ids_verified": False,
        "active_query_position_token_pairs": [list(x) for x in pending.query_inputs],
        "sampler_selected_step_prefix_ids": valid[:j],
        "sampled_token_id_at_target": int(valid[j]),
        "sampling_decision_captured": True,
        "api_response_confirmed": False,
        "async_scheduling_enabled": pending.async_scheduling,
        "optimistic_cached_output_len_at_capture": pending.req_output_tokens_observed,
        "optimistic_cached_output_has_negative_one": pending.req_has_optimistic_placeholders,
        "prefix_cache_enabled_config": pending.prefix_cache_enabled,
        "actual_prefix_cache_hit": "UNKNOWN_NOT_INSTRUMENTED",
        "target_top1_id": top1,
        "target_top1_logit": float(raw[top1]),
        "target_top2_id": top2,
        "target_top2_logit": float(raw[top2]),
        "top2_margin": float(raw[top1] - raw[top2]),
        "logits_vector_sha256_float32": hashlib.sha256(raw.tobytes()).hexdigest(),
        "logits_vector_file": vector_name,
        "same_kv_cache_state_proven": False,
        "runtime_probe_changes_timing": True,
    }
    _new_output_dir(root)
    with (root / vector_name).open("xb") as out:
        np.save(out, raw, allow_pickle=False)
    with (root / "records.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"[M2-03 A3 V8 RECORD] req={pending.req_id} target={target} "
          f"selected={valid[j]} row={j} file={vector_name} API_PENDING", flush=True)
    return record


def prepare_v1_capture(runner, scheduler_output, logits, spec_decode_metadata, grammar_output):
    """Called from unchanged v7 hook; never propagate observer errors."""
    try:
        return _prepare(runner, scheduler_output, logits, spec_decode_metadata, grammar_output)
    except Exception as exc:
        print(f"[M2-03 A3 V8 SKIP] stage=prepare kind={type(exc).__name__} "
              f"reason={exc}", flush=True)
        return None


def finalize_v1_capture(runner, pending, sampler_output):
    """Called from unchanged v7 hook; never propagate observer errors."""
    try:
        return _finalize(runner, pending, sampler_output)
    except Exception as exc:
        print(f"[M2-03 A3 V8 SKIP] stage=finalize kind={type(exc).__name__} "
              f"reason={exc}", flush=True)
        return None
