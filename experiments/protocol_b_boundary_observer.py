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
    record = dict(key=key, request_id=ids[0], prompt_token_ids=config["prompt_token_ids"],
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
        if stage == "begin":
            begin(config, *args)
        elif stage in ("sample_scores", "verify_scores"):
            processed(stage, *args)
        elif stage == "end":
            end(config, *args)
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
