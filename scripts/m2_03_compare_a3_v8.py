#!/usr/bin/env python3
"""Offline API-confirmed A3 v8 Adaptive-V1/Vanilla-V2 raw logits comparison.

NO effect on running vLLM. Fails closed on row/prefix/request ambiguity.
This is diagnostic evidence, never a substitute for canonical M2-03 gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


class InvalidEvidence(RuntimeError):
    pass


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _records(path: Path) -> list[dict]:
    f = path / "records.jsonl"
    if not f.is_file():
        raise InvalidEvidence(f"Missing records.jsonl in {path}")
    rows = [json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:
        raise InvalidEvidence(f"Empty records.jsonl in {path}")
    return rows


def _vector(directory: Path, rec: dict) -> np.ndarray:
    name = rec.get("logits_vector_file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npy"):
        raise InvalidEvidence("Unsafe logits vector file")
    vec = np.load(directory / name, allow_pickle=False)
    if vec.ndim != 1 or vec.dtype != np.float32 or not np.isfinite(vec).all():
        raise InvalidEvidence("Logits vector is not finite float32 1D")
    if hashlib.sha256(vec.tobytes()).hexdigest() != rec.get("logits_vector_sha256_float32"):
        raise InvalidEvidence("Logits vector SHA mismatch")
    return vec


def compare(vanilla_dir: Path, adaptive_dir: Path, response_path: Path) -> dict:
    resp = json.loads(response_path.read_text(encoding="utf-8"))
    req_id = resp.get("id")
    if not isinstance(req_id, str) or not req_id:
        raise InvalidEvidence("HTTP API response must include request id")
    choices = resp.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise InvalidEvidence("Expected one API response choice")
    api_ids = choices[0].get("token_ids")
    if not isinstance(api_ids, list) or any(type(x) is not int or x < 0 for x in api_ids):
        raise InvalidEvidence("HTTP API response lacks valid return_token_ids")

    va = _records(vanilla_dir)
    if len(va) != 1:
        raise InvalidEvidence(f"Expected exactly one Vanilla V2 record, got {len(va)}")
    v = va[0]
    target = v.get("target_generated_index_zero_based")
    if type(target) is not int or target < 0 or target >= len(api_ids):
        raise InvalidEvidence("API response ended before Vanilla target index")
    all_rows = _records(adaptive_dir)

    # API ID can be a prefix of the internal EngineCore request ID.
    # Match only an exact ID or a hyphen-delimited internal suffix.
    # The result MUST be unique; never silently select a candidate.
    rows = [
        r for r in all_rows
        if (
            r.get("request_id") == req_id
            or (
                isinstance(r.get("request_id"), str)
                and r["request_id"].startswith(req_id + "-")
            )
        )
        and r.get("target_generated_index_zero_based") == target
    ]
    if len(rows) != 1:
        raise InvalidEvidence(f"Expected one Adaptive record for request {req_id}, got {len(rows)}")
    a = rows[0]

    for rec, lane in ((v, "vanilla"), (a, "adaptive")):
        if rec.get("schema") != "m2-03.a3.raw-target-logits.v1" or rec.get("lane") != lane:
            raise InvalidEvidence(f"Wrong {lane} record schema or lane")
        if rec.get("vllm_version") != "0.30.0":
            raise InvalidEvidence("Incompatible vLLM versions")
        if rec.get("logits_kind") != "raw_target_before_grammar_and_sampling_processors":
            raise InvalidEvidence("Logits capture point differs")
    if v.get("prefix_proof_status") != "COMMITTED_TOKEN_PREFIX_HASHED" or v.get("active_query_input_ids_verified") is not True:
        raise InvalidEvidence("Vanilla conditioning prefix/active query unverified")
    if a.get("instrumentation_version") != "v1-runner-a3-v8-async":
        raise InvalidEvidence("Expected v8 Adaptive source instrumentation")
    if a.get("prefix_proof_status") != "PENDING_API_FULL_PREFIX_VERIFICATION":
        raise InvalidEvidence("Adaptive record is not in pending-API state")
    if a.get("sampling_decision_captured") is not True or a.get("api_response_confirmed") is not False:
        raise InvalidEvidence("Sampler evidence missing or raw evidence mutated")
    if a.get("sampled_token_id_at_target") != api_ids[target]:
        raise InvalidEvidence("Sampler selected token differs from HTTP token at target")

    prompt = a.get("prompt_token_ids")
    if not isinstance(prompt, list) or not prompt or any(type(t) is not int or t < 0 for t in prompt):
        raise InvalidEvidence("Missing Adaptive actual runtime prompt token IDs")
    if token_hash(prompt) != a.get("prompt_token_ids_sha256"):
        raise InvalidEvidence("Adaptive prompt-token hash mismatch")
    expected_position = len(prompt) + target - 1
    if (a.get("prompt_token_count") != v.get("prompt_token_count") or
            a.get("model_position_zero_based") != expected_position or
            v.get("model_position_zero_based") != expected_position):
        raise InvalidEvidence("Prompt length or absolute model position differs")

    base, offset = a.get("generated_base_index_from_forward_positions"), a.get("logits_row_index_in_forward")
    if type(base) is not int or type(offset) is not int or base < 0 or offset < 0 or base + offset != target:
        raise InvalidEvidence("Invalid V1 position-derived row accounting")
    if a.get("sampler_selected_step_prefix_ids") != api_ids[base:target]:
        raise InvalidEvidence("Earlier accepted sampled tokens differ from HTTP prefix")

    full_prefix = prompt + api_ids[:target]
    pairs = a.get("active_query_position_token_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise InvalidEvidence("Missing V1 active query input pairs")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise InvalidEvidence("Invalid V1 active query pair")
        pos, token = pair
        if (type(pos) is not int or type(token) is not int or pos < 0 or
                pos >= len(full_prefix) or pos in seen or full_prefix[pos] != token):
            raise InvalidEvidence(f"V1 active model input differs from real API prefix at {pos}")
        seen.add(pos)
    if expected_position not in seen:
        raise InvalidEvidence("Target-conditioning model input position not checked")
    prefix_hash = token_hash(full_prefix)
    if v.get("conditioning_token_prefix_sha256") != prefix_hash:
        raise InvalidEvidence("Vanilla/Adaptive complete conditioning token prefix differs")
    if a.get("prefix_cache_enabled_config") != v.get("prefix_cache_enabled_config"):
        raise InvalidEvidence("Prefix cache config differs")

    v_logits, a_logits = _vector(vanilla_dir, v), _vector(adaptive_dir, a)
    if v_logits.shape != a_logits.shape:
        raise InvalidEvidence("Target vocabulary/logits shape differs")
    diff = np.abs(v_logits.astype(np.float64) - a_logits.astype(np.float64))
    return {
        "result_kind": "DIAGNOSTIC_NOT_CANONICAL_GATE",
        "request_id": req_id,
        "target_generated_index_zero_based": target,
        "adaptive_sampler_token": api_ids[target],
        "conditioning_token_prefix_sha256": prefix_hash,
        "api_emitted_prefix_and_input_ids_verified": True,
        "sampler_selected_row_verified": True,
        "adaptive_async_scheduling_enabled": a["async_scheduling_enabled"],
        "adaptive_had_optimistic_placeholders": a["optimistic_cached_output_has_negative_one"],
        "adaptive_row_role": a["row_role"],
        "raw_argmax_vanilla": int(v_logits.argmax()),
        "raw_argmax_adaptive": int(a_logits.argmax()),
        "raw_argmax_equal": bool(v_logits.argmax() == a_logits.argmax()),
        "raw_fp32_vector_equal": bool(np.array_equal(v_logits, a_logits)),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "same_runner_implementation": False,
        "same_kv_cache_state_proven": False,
        "same_processed_logits_proven": False,
        "same_sampling_decision_proven": False,
        "runtime_probe_changes_timing": True,
        "note": "Prefix verified against actual model input pairs and HTTP token IDs. "
                "Vanilla V2 vs Adaptive V1 remains a confounder. Diagnostic only.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vanilla-dir", required=True, type=Path)
    ap.add_argument("--adaptive-dir", required=True, type=Path)
    ap.add_argument("--adaptive-response", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        result = compare(args.vanilla_dir, args.adaptive_dir, args.adaptive_response)
        text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            path = args.output.resolve()
            if "results" in path.parts or "m2-03-20261013-r03" in path.parts:
                raise InvalidEvidence("Refusing canonical evidence overwrite")
            with path.open("x", encoding="utf-8") as f:
                f.write(text)
        print(text, end="")
        return 0
    except (InvalidEvidence, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"M2_03_A3_V8_COMPARISON_INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
