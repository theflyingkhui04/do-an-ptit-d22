"""M2-03 A3: diagnostic-only vLLM 0.30.0 raw target-logits observer.

Intended insertion point: GPUModelRunner.sample(), immediately after the
logits creation branch and BEFORE grammar bitmask/logits processors.

This is NOT a canonical gate. It only captures evidence for one diagnostic
request (c1), one target generated-token index, without altering logits.

Configuration priority:
1. M203_A3_OUTPUT_DIR / M203_A3_LANE environment variables, if present.
2. /tmp/m203_a3_probe_config.json as a process-independent fallback.

The config-file fallback exists because vLLM V1 runs EngineCore in a separate
background process and the Kaggle launch path used by this diagnostic did not
preserve the custom M203_A3_* environment variables into that process.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ProbeError(RuntimeError):
    pass


CONFIG_PATH = Path("/tmp/m203_a3_probe_config.json")
_OWNED_DIRS: set[str] = set()
_RUNTIME_CONFIG_CACHE: dict[str, Any] | None = None


def token_ids_sha256(tokens: list[int]) -> str:
    """Hash exact token IDs (not text) with a reproducible encoding."""
    data = json.dumps(tokens, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def _new_output_dir(root: Path) -> None:
    identity = str(root.resolve())
    if identity in _OWNED_DIRS:
        return
    # A pre-existing directory is a collision, even if it appears empty.
    root.mkdir(parents=True, exist_ok=False)
    _OWNED_DIRS.add(identity)


def _normalise_runtime_config(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ProbeError(f"A3 {source} config must be a JSON object")

    lane = raw.get("lane")
    if not isinstance(lane, str) or lane not in ("vanilla", "adaptive"):
        raise ProbeError(f"A3 {source} config lane must be vanilla or adaptive")

    output_dir = raw.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ProbeError(f"A3 {source} config output_dir must be a non-empty string")

    target = raw.get("target", 200)
    if type(target) is not int or target < 0:
        raise ProbeError(f"A3 {source} config target must be a nonnegative int")

    git_sha = raw.get("git_sha", "UNKNOWN")
    if not isinstance(git_sha, str) or not git_sha.strip():
        raise ProbeError(f"A3 {source} config git_sha must be a non-empty string")

    return {
        "source": source,
        "lane": lane,
        "target": target,
        "git_sha": git_sha.strip(),
        "output_dir": output_dir.strip(),
    }


def _load_runtime_config() -> dict[str, Any] | None:
    """Load diagnostic config once per process.

    Environment config is used only when OUTPUT_DIR or LANE is explicitly set.
    Otherwise CONFIG_PATH is used. A partial environment configuration fails
    closed instead of silently mixing environment fields with the file.
    """
    global _RUNTIME_CONFIG_CACHE
    if _RUNTIME_CONFIG_CACHE is not None:
        return _RUNTIME_CONFIG_CACHE

    env_out = os.environ.get("M203_A3_OUTPUT_DIR", "").strip()
    env_lane = os.environ.get("M203_A3_LANE", "").strip()

    if env_out or env_lane:
        if not env_out or not env_lane:
            raise ProbeError(
                "Partial M203_A3 environment: both M203_A3_OUTPUT_DIR and "
                "M203_A3_LANE are required"
            )
        raw_env: dict[str, Any] = {
            "output_dir": env_out,
            "lane": env_lane,
            "git_sha": os.environ.get("M203_A3_GIT_SHA", "UNKNOWN"),
        }
        raw_target = os.environ.get("M203_A3_TARGET", "200")
        try:
            raw_env["target"] = int(raw_target)
        except ValueError as exc:
            raise ProbeError("M203_A3_TARGET must be an integer") from exc
        # Environment configuration is read on each call. This preserves the
        # existing test/diagnostic behaviour where a single Python process can
        # switch lanes between isolated calls. The config-file fallback below
        # is cached because it is immutable for one EngineCore process.
        return _normalise_runtime_config(raw_env, source="environment")

    if not CONFIG_PATH.exists():
        return None

    try:
        raw_file = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"Cannot read valid A3 config file {CONFIG_PATH}: {exc}") from exc

    config = _normalise_runtime_config(raw_file, source="config_file")
    _RUNTIME_CONFIG_CACHE = config
    print(
        "[M2-03 A3 CONFIG]"
        f" source=config_file path={CONFIG_PATH} lane={config['lane']}"
        f" target={config['target']} output_dir={config['output_dir']}",
        flush=True,
    )
    return config


def _verify_actual_query_inputs(
    batch: object,
    *,
    selected_flat_index: int,
    expected_position: int,
    committed_tokens: list[int],
) -> dict:
    """Check the ACTIVE model-forward inputs against the committed token IDs.

    Scope: one request, no TP/PCP. Only positions up to the selected row's
    logical position are relevant. Other positions may be speculative drafts.
    This does NOT validate the KV tensors for tokens skipped via prefix cache.
    """
    if not hasattr(batch, "input_ids"):
        raise ProbeError("input_ids unavailable: cannot verify actual model inputs")
    if not hasattr(batch, "query_start_loc"):
        raise ProbeError("query_start_loc unavailable: cannot identify active query")

    query_limits = batch.query_start_loc[:2].detach().cpu().tolist()
    if len(query_limits) != 2 or int(query_limits[0]) != 0:
        raise ProbeError(f"Invalid single-request query_start_loc: {query_limits}")
    query_end = int(query_limits[1])
    if not (0 <= selected_flat_index < query_end):
        raise ProbeError("Selected logits source is outside active query")
    if query_end > len(batch.positions) or query_end > len(batch.input_ids):
        raise ProbeError("Actual model input tensors shorter than active query")

    active_positions = [
        int(v) for v in batch.positions[:query_end].detach().cpu().tolist()
    ]
    active_input_ids = [
        int(v) for v in batch.input_ids[:query_end].detach().cpu().tolist()
    ]
    if active_positions[selected_flat_index] != expected_position:
        raise ProbeError("Selected logits row has inconsistent model position")

    seen_positions: set[int] = set()
    verified = 0
    for position, model_input_id in zip(active_positions, active_input_ids):
        if position > expected_position:
            continue  # Future draft positions cannot condition the selected row.
        if position < 0 or position >= len(committed_tokens):
            raise ProbeError(f"Invalid active model position {position}")
        if position in seen_positions:
            raise ProbeError(f"Duplicate active model position {position}")
        seen_positions.add(position)
        if model_input_id != committed_tokens[position]:
            raise ProbeError(
                f"Actual model input token mismatch at model position {position}: "
                f"input_ids={model_input_id}, committed={committed_tokens[position]}"
            )
        verified += 1

    if expected_position not in seen_positions or verified == 0:
        raise ProbeError("Selected model position was not verified in active query")
    return {
        "active_query_input_ids_verified": True,
        "active_query_tokens_checked": verified,
        "selected_model_input_flat_index": selected_flat_index,
        "selected_model_input_token_id": active_input_ids[selected_flat_index],
    }


def capture_target_logits(
    runner: object,
    batch: object,
    logits: torch.Tensor,
) -> dict | None:
    """Capture a precisely identified row without returning/altering `logits`.

    Configuration is loaded from M203_A3_* environment variables when an
    environment configuration is explicitly present; otherwise it falls back
    to /tmp/m203_a3_probe_config.json.

    Returns None if disabled or if no row matches. Raises on unsafe ambiguity.
    """
    config = _load_runtime_config()
    if config is None:
        return None

    out = config["output_dir"]
    lane = config["lane"]
    target = config["target"]
    git_sha = config["git_sha"]

    version = importlib.metadata.version("vllm")
    if version != "0.30.0":
        raise ProbeError(f"vLLM version mismatch: expected 0.30.0, got {version}")
    if getattr(runner, "batch_sharder", None) is not None:
        raise ProbeError("A3 observer does not support batch sharding / TP")

    # vLLM 0.30.0 invokes sampling during synthetic kernel warm-up. Those
    # requests do not contain real user inputs and must not produce evidence.
    # Skip only batches where EVERY active request has a known warm-up prefix.
    num_reqs = int(batch.num_reqs)
    if num_reqs < 0:
        raise ProbeError("Negative active request count")
    if num_reqs == 0:
        return None
    active_req_ids = list(batch.req_ids[:num_reqs])
    if len(active_req_ids) != num_reqs or any(
        not isinstance(req_id, str) or not req_id for req_id in active_req_ids
    ):
        raise ProbeError("Invalid active request ID mapping")
    is_warmup = [
        req_id.startswith(("_warmup_", "_v2_mixed_warmup"))
        for req_id in active_req_ids
    ]
    if any(is_warmup):
        if all(is_warmup):
            return None
        raise ProbeError("Mixed warmup and real requests: refusing A3 capture")
    if num_reqs != 1:
        raise ProbeError("A3 observer requires one active request (diagnostic c1)")
    if getattr(batch, "has_structured_output_reqs", False):
        raise ProbeError("A3 observer excludes grammar-constrained requests")

    root = Path(out).expanduser().resolve()
    if "m2-03-20261013-r03" in root.parts or root.name == "results":
        raise ProbeError("Refusing to write in canonical evidence/results directory")

    # Device-side cu_num_logits is authoritative: CPU copy may be modified by
    # adaptive GPU-side draft reallocation in vLLM 0.30.0.
    cu = batch.cu_num_logits[:2].detach().cpu().tolist()
    lo, hi = map(int, cu)
    if not (0 <= lo <= hi <= logits.shape[0]):
        raise ProbeError(f"Invalid cumulative-logits range: {cu}")
    positions = batch.positions[batch.logits_indices].detach().cpu().tolist()
    if len(positions) != logits.shape[0]:
        raise ProbeError("positions/logits row-count mismatch")

    request_id = str(batch.req_ids[0])
    slot = int(batch.idx_mapping_np[0])
    prompt_len = int(runner.req_states.prompt_len.np[slot])
    if prompt_len <= 0:
        raise ProbeError("Prompt length must be positive")

    # Row computed on the last conditioning token predicts the NEXT token.
    expected_position = prompt_len + target - 1

    committed_len_now = int(runner.req_states.total_len.gpu[slot].item())
    committed_generated_now = committed_len_now - prompt_len

    matched_rows = [
        r for r in range(lo, hi)
        if int(positions[r]) == expected_position
    ]

    # Diagnostic mapping trace around the requested generated-token index.
    # This line is diagnostic only; it is NOT accepted logits evidence.
    if abs(committed_generated_now - target) <= 8:
        print(
            "[M2-03 A3 MAP]"
            f" request_id={request_id}"
            f" target={target}"
            f" prompt_len={prompt_len}"
            f" committed_len={committed_len_now}"
            f" committed_generated={committed_generated_now}"
            f" expected_position={expected_position}"
            f" cu=[{lo},{hi}]"
            f" positions={positions[lo:hi]}"
            f" logits_indices="
            f"{batch.logits_indices[lo:hi].detach().cpu().tolist()}"
            f" matched_rows={matched_rows}",
            flush=True,
        )

    if not matched_rows:
        return None

    if len(matched_rows) != 1:
        raise ProbeError(
            f"Ambiguous logits rows for generated index {target}: "
            f"{matched_rows}"
        )
    row_index = matched_rows[0]
    # logits_indices indexes the flattened model input, NOT generated output.
    selected_flat_index = int(batch.logits_indices[row_index].item())

    # A speculative row can rely on unaccepted draft tokens. A matching model
    # position alone does NOT establish the same conditioning prefix.
    required_prefix_len = prompt_len + target
    committed_len = int(runner.req_states.total_len.gpu[slot].item())
    if committed_len >= required_prefix_len:
        token_prefix = (
            runner.req_states.all_token_ids.gpu[slot, :required_prefix_len]
            .detach().cpu().tolist()
        )
        if len(token_prefix) != required_prefix_len:
            raise ProbeError("Committed prefix tensor is shorter than expected")
        committed_tokens = [int(x) for x in token_prefix]
        # A hash of all_token_ids alone is insufficient: vLLM combines the
        # sampled token and speculative drafts in the actual model input.
        # Fail before writing any evidence if those disagree.
        active_input_evidence = _verify_actual_query_inputs(
            batch,
            selected_flat_index=selected_flat_index,
            expected_position=expected_position,
            committed_tokens=committed_tokens,
        )
        prefix_sha256 = token_ids_sha256(committed_tokens)
        prefix_status = "COMMITTED_TOKEN_PREFIX_HASHED"
    else:
        prefix_sha256 = None
        prefix_status = "UNVERIFIED_SPECULATIVE_CONTEXT"
        active_input_evidence = {
            "active_query_input_ids_verified": False,
            "active_query_tokens_checked": 0,
            "selected_model_input_flat_index": selected_flat_index,
            "selected_model_input_token_id": None,
        }

    # float32 copy is for analysis; original GPU logits are not mutated.
    raw = logits[row_index].detach().float().cpu().contiguous().numpy().copy()
    if raw.ndim != 1 or raw.size < 2 or not np.isfinite(raw).all():
        raise ProbeError("Invalid/non-finite raw target-logits vector")

    # np.argmax is deterministic for ties: it returns the first maximal index.
    top1_id = int(np.argmax(raw))
    remaining = raw.copy()
    remaining[top1_id] = -np.inf
    top2_id = int(np.argmax(remaining))

    vec_sha = hashlib.sha256(raw.tobytes()).hexdigest()
    key = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:20]
    # Two logits rows may appear for the same position after a re-execution.
    # Never overwrite; use another run directory or investigate duplicates.
    vec_name = f"{lane}_{key}_g{target}_p{expected_position}.npy"
    record = {
        "schema": "m2-03.a3.raw-target-logits.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "config_source": config["source"],
        "vllm_version": version,
        "lane": lane,
        "request_id": request_id,
        "prompt_token_count": prompt_len,
        "target_generated_index_zero_based": target,
        "model_position_zero_based": expected_position,
        "logits_row_index_in_forward": row_index,
        "logits_row_range_for_request": [lo, hi],
        "num_draft_tokens_in_batch": int(batch.num_draft_tokens),
        "logits_kind": "raw_target_before_grammar_and_sampling_processors",
        "original_logits_dtype": str(logits.dtype),
        "prefix_proof_status": prefix_status,
        "committed_token_count_including_prompt": committed_len,
        "conditioning_token_prefix_sha256": prefix_sha256,
        **active_input_evidence,
        "prefix_cache_enabled_config": bool(runner.cache_config.enable_prefix_caching),
        "actual_prefix_cache_hit": "UNKNOWN_NOT_INSTRUMENTED",
        "target_top1_id": top1_id,
        "target_top1_logit": float(raw[top1_id]),
        "target_top2_id": top2_id,
        "target_top2_logit": float(raw[top2_id]),
        "top2_margin": float(raw[top1_id] - raw[top2_id]),
        "logits_vector_sha256_float32": vec_sha,
        "logits_vector_file": vec_name,
        "sampling_decision_captured": False,
        "verification_step": None,
        "runtime_probe_changes_timing": True,
    }
    _new_output_dir(root)
    with (root / vec_name).open("xb") as handle:
        np.save(handle, raw, allow_pickle=False)
    with (root / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return record
