"""Strict complete-coverage token capture comparator; no model execution."""
from __future__ import annotations

from experiments.lossless_verifier import verify_token_sequence
from experiments.m4_treatments import TREATMENTS, TARGET, TARGET_REVISION


def validate_capture(record: dict) -> None:
    tokens = record.get("token_ids")
    if (not isinstance(tokens, list) or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)):
        raise ValueError("Empty or invalid emitted token-ID capture")
    if record.get("complete") is not True or record.get("finish_reason") not in ("stop", "length"):
        raise ValueError("Incomplete request capture")
    if record.get("target_model") != TARGET or record.get("target_revision") != TARGET_REVISION:
        raise ValueError("Target identity mismatch")
    if record.get("temperature") != 0 or record.get("seed") != 42:
        raise ValueError("Greedy/seed contract differs")


def compare_captures(captures: list[dict], expected: list[dict], source_commit: str) -> dict:
    """Two Vanilla captures qualify repeatability; every prompt/treatment required.

    expected entries retain workload, prompt ID/hash, cap and stopping semantics.
    All captures and mismatches are retained in the result, including provenance.
    """
    identities = [(e["workload"], e["prompt_id"]) for e in expected]
    if not expected or len(set(identities)) != len(identities):
        raise ValueError("Empty/duplicate expected prompt identity")
    index = {}
    for r in captures:
        validate_capture(r)
        key = (r["workload"], r["prompt_id"], r["treatment"], r["repeat"])
        if key in index or r.get("source_commit") != source_commit:
            raise ValueError("Duplicate capture or source provenance mismatch")
        index[key] = r
    expected_keys = {(w, p, t, repeat) for w, p in identities for t in TREATMENTS
                     for repeat in (range(2) if t == "vanilla" else range(1))}
    if set(index) != expected_keys:
        raise ValueError("Incomplete/unexpected lossless coverage")
    results, breakdown = [], {}
    for e in expected:
        w, p = e["workload"], e["prompt_id"]
        base = index[(w, p, "vanilla", 0)]
        for treatment, repeat in (("vanilla", 1), ("eagle_fixed", 0), ("eagle_adaptive", 0)):
            other = index[(w, p, treatment, repeat)]
            for r in (base, other):
                for field in ("prompt_hash", "max_tokens", "stopping", "tokenizer_revision"):
                    if r.get(field) != e.get(field) or e.get(field) is None:
                        raise ValueError(f"Prompt/request contract mismatch: {field}")
            comparison = verify_token_sequence(base["token_ids"], other["token_ids"], p, w).to_dict()
            comparison.update(treatment=treatment, baseline=base, compared=other)
            results.append(comparison)
            counts = breakdown.setdefault(w, dict(comparisons=0, mismatches=0))
            counts["comparisons"] += 1
            counts["mismatches"] += int(not comparison["is_match"])
    return dict(passed=all(r["is_match"] for r in results), source_commit=source_commit,
                prompt_count=len(expected), workload_breakdown=breakdown, comparisons=results,
                claim="Deterministic greedy equality only; no stochastic distribution proof")
