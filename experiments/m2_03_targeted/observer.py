"""Observer and explicit diagnostic control called by exact-source insertions.

Copies add synchronization/timing overhead. No inference tensor is written and
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

from .common import WINDOW, event, load, need, save, sha, write_new

_local = threading.local()


def ints(tensor):
    return tensor.detach().cpu().tolist()


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


def snapshot(tensor, root, name):
    dtype = str(tensor.dtype)
    # tobytes owns its bytes even when .float()/.cpu() returned a view.
    raw = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=True).tobytes()
    stats = score_stats(raw)
    write_new(Path(root) / name, raw)
    return {"file": name, "sha256": sha(raw), "storage_dtype": "float32-le",
            "source_dtype": dtype, **stats}


def begin(config, runner, scheduled, logits, spec, grammar):
    _local.ctx = None
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
    if not any(r["generated_index"] in WINDOW for r in rows):
        return
    need(not bool(getattr(getattr(runner, "discard_request_mask", None), "np", [False])[0]),
         "discarded sampler output")
    query = list(zip(positions, ints(runner.input_ids.gpu[lo:hi])))
    need(len({p for p, _ in query}) == len(query) and all(p >= 0 and t >= 0 for p, t in query), "invalid active query")
    key = uuid.uuid4().hex
    capture = {"schema": "m203.targeted.forward.v1", "key": key, "lane": config["lane"],
               "execution_sha": config["execution_sha"], "request_id": ids[0], "prompt_token_ids": prompt,
               "active_query": query, "rows": rows, "draft_count": n,
               "decoding_path": 'ordinary' if spec is None else 'speculative',
               "spec_target_rows": targets, "spec_bonus_rows": bonus,
               "async_scheduling": bool(runner.use_async_scheduling), "prefix_cache_enabled": False,
               "same_internal_state_proven": False, "runtime_probe_changes_timing": True,
               "scores": {}, "selected": None}
    for r in rows:
        if r["generated_index"] in WINDOW:
            j = r["row"]
            capture["scores"][str(j)] = {"raw": snapshot(logits[j], config["directory"], f"{key}-r{j}-raw.f32")}
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
                                                             f'{record["key"]}-r{j}-processed.f32')
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
    save(Path(config["directory"]) / (record["key"] + ".forward.json"), record)


def accounting(config, stage, request, output, scheduled, stale):
    before = len(request._output_token_ids)
    if not (52 <= before <= 70):
        return
    need(type(stale) is bool, "invalid stale flag")
    event(config["directory"], stage, request_id=request.request_id,
          output_count=before, output_token_ids=list(output), stale=stale,
          num_computed_tokens=request.num_computed_tokens,
          num_in_flight_tokens=request.num_in_flight_tokens,
          num_output_placeholders=request.num_output_placeholders,
          invalid_spec_tokens=(getattr(scheduled, 'num_invalid_spec_tokens', None) or {}).get(request.request_id, 0),
          scheduled_tokens=scheduled.num_scheduled_tokens.get(request.request_id),
          scheduled_spec_token_ids=list(scheduled.scheduled_spec_decode_tokens.get(request.request_id, [])))


def kv_recompute_before_schedule(config, scheduler):
    target = config.get("diagnostic_control", {}).get(
        "adaptive_kv_recompute_output_count")
    if target is None or config["lane"] != "adaptive":
        return
    need(target == 60, "unreviewed KV recompute boundary")

    state = getattr(scheduler, "_m203_kv_recompute", None)
    if state is not None:
        need(state["request_id"] in scheduler.requests,
             "KV recompute request disappeared")
        request = scheduler.requests[state["request_id"]]
        if state["phase"] == "refill":
            need(request.num_computed_tokens == state["recomputed_tokens"],
                 "KV recompute did not stop before boundary token")
            scheduler.max_num_scheduled_tokens = state["original_budget"]
            state["phase"] = "ready"
            event(config["directory"], "kv_recompute_ready",
                  request_id=request.request_id,
                  output_count=len(request._output_token_ids),
                  total_tokens=state["total_tokens"],
                  recomputed_tokens=state["recomputed_tokens"],
                  num_computed_tokens=request.num_computed_tokens,
                  budget_restored=(scheduler.max_num_scheduled_tokens
                                   == state["original_budget"]))
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
        "total_tokens": total, "recomputed_tokens": recomputed,
        "original_budget": original_budget,
    }


def control(stage, *args):
    config_path = os.environ.get("M203_TARGETED_CONFIG")
    if not config_path:
        return
    config = load(config_path)
    if not config.get("diagnostic_control", {}).get(
            "adaptive_kv_recompute_output_count"):
        return
    need(stage == "before_schedule", "unknown diagnostic control stage")
    kv_recompute_before_schedule(config, *args)


def observe(stage, *args):
    config_path = os.environ.get("M203_TARGETED_CONFIG")
    if not config_path:
        return
    try:
        config = load(config_path)
        if stage == "begin":
            begin(config, *args)
        elif stage == "end":
            end(config, *args)
        elif stage in ("sample_scores", "verify_scores"):
            processed(stage, *args)
        elif stage in ("account_before", "account_after"):
            accounting(config, stage, *args)
    except Exception as exc:
        _local.ctx = None
        try:
            config = load(config_path)
            event(config["directory"], "observer_error", stage=stage,
                  error=f"{type(exc).__name__}: {exc}")
        except Exception:
            # Missing error file must still fail capture completeness checks.
            print("M203_TARGETED_OBSERVER_ERROR_UNSAVED", file=sys.stderr, flush=True)
