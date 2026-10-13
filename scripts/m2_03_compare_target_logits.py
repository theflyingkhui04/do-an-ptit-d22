"""Offline comparison of two A3 raw target-logits records; never a PASS gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


class ComparisonError(RuntimeError):
    pass


def _read_record(directory: Path, request_id: str | None) -> dict:
    path = directory / "records.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if request_id is not None:
        records = [r for r in records if r["request_id"] == request_id]
    if len(records) != 1:
        raise ComparisonError(f"Expected exactly one record from {directory}, found {len(records)}")
    return records[0]


def _read_vector(directory: Path, record: dict) -> np.ndarray:
    name = record["logits_vector_file"]
    if Path(name).name != name or not name.endswith(".npy"):
        raise ComparisonError("Unsafe/invalid logits vector filename")
    vector = np.load(directory / name, allow_pickle=False)
    if vector.dtype != np.float32 or vector.ndim != 1:
        raise ComparisonError("Invalid float32 logits vector")
    actual = hashlib.sha256(vector.tobytes()).hexdigest()
    if actual != record["logits_vector_sha256_float32"]:
        raise ComparisonError("Logits vector SHA256 mismatch")
    if not np.isfinite(vector).all():
        raise ComparisonError("Non-finite logits")
    return vector


def compare(vanilla_dir: Path, adaptive_dir: Path, vanilla_request_id: str | None = None,
            adaptive_request_id: str | None = None) -> dict:
    va = _read_record(vanilla_dir, vanilla_request_id)
    ad = _read_record(adaptive_dir, adaptive_request_id)
    if (va["lane"], ad["lane"]) != ("vanilla", "adaptive"):
        raise ComparisonError("Expected vanilla vs adaptive records")
    for r in (va, ad):
        if r["schema"] != "m2-03.a3.raw-target-logits.v1" or r["vllm_version"] != "0.30.0":
            raise ComparisonError("Invalid schema or pinned runtime version")
        if r["prefix_proof_status"] != "COMMITTED_TOKEN_PREFIX_HASHED":
            raise ComparisonError("Candidate-only row: conditioning prefix is not verified")
        # Fail closed on legacy records lacking the new actual-input check.
        if r.get("active_query_input_ids_verified") is not True:
            raise ComparisonError("Actual active query input IDs have not been verified")
        if not isinstance(r.get("active_query_tokens_checked"), int) or r["active_query_tokens_checked"] <= 0:
            raise ComparisonError("Actual active query evidence has invalid token count")
        if r["logits_kind"] != "raw_target_before_grammar_and_sampling_processors":
            raise ComparisonError("Incompatible logits measurement point")
    for field in ("target_generated_index_zero_based", "model_position_zero_based", "prompt_token_count",
                  "conditioning_token_prefix_sha256"):
        if va[field] != ad[field]:
            raise ComparisonError(f"Cannot compare: {field} differs")
    a, b = _read_vector(vanilla_dir, va), _read_vector(adaptive_dir, ad)
    if a.shape != b.shape:
        raise ComparisonError("Vocabulary/logits length mismatch")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    report = {
        "result_kind": "DIAGNOSTIC_NOT_CANONICAL_GATE",
        "vanilla_request_id": va["request_id"],
        "adaptive_request_id": ad["request_id"],
        "target_generated_index_zero_based": va["target_generated_index_zero_based"],
        "conditioning_prefix_sha256": va["conditioning_token_prefix_sha256"],
        "raw_target_argmax_vanilla": int(a.argmax()),
        "raw_target_argmax_adaptive": int(b.argmax()),
        "raw_argmax_equal": bool(a.argmax() == b.argmax()),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "exact_fp32_vector_equal": bool(np.array_equal(a, b)),
        "same_kv_cache_state_proven": False,
        "same_sampling_decision_proven": False,
        "note": "A valid raw-logits comparison does not prove KV-cache equivalence or processed-logits/actual sampled-token equivalence.",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--vanilla-request-id")
    parser.add_argument("--adaptive-request-id")
    parser.add_argument("--output", type=Path, help="New file; never overwrite an existing report")
    args = parser.parse_args()
    try:
        result = compare(args.vanilla_dir, args.adaptive_dir, args.vanilla_request_id,
                         args.adaptive_request_id)
        text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text)
    except (OSError, ValueError, KeyError, ComparisonError) as exc:
        print(f"A3_COMPARISON_INVALID: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
