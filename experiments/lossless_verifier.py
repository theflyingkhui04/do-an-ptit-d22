"""Lossless Verification Utility for Speculative Decoding (M2-03).

Provides token-by-token and character-by-character absolute equivalence verification
between Vanilla and Adaptive decoding at temperature = 0.0 (greedy decoding).
Detects exact discrepancy locations, context windows, and aggregates batch summaries.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class TokenDiscrepancy:
    """Detailed record of a discrepancy between expected (Vanilla) and actual (Adaptive)."""

    index: int
    expected_token_id: Optional[int] = None
    actual_token_id: Optional[int] = None
    expected_text: Optional[str] = None
    actual_text: Optional[str] = None
    context_window: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LosslessVerificationResult:
    """Evaluation result for a single prompt."""

    prompt_id: str
    workload: str = "W1"
    is_match: bool = True
    match_rate: float = 1.0
    vanilla_token_count: int = 0
    adaptive_token_count: int = 0
    first_discrepancy: Optional[TokenDiscrepancy] = None
    all_discrepancies: List[TokenDiscrepancy] = field(default_factory=list)
    vanilla_text: Optional[str] = None
    adaptive_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.first_discrepancy:
            data["first_discrepancy"] = self.first_discrepancy.to_dict()
        data["all_discrepancies"] = [d.to_dict() for d in self.all_discrepancies]
        return data


@dataclass
class LosslessBatchSummary:
    """Summary of lossless verification over a batch of prompts (e.g. 50 prompts)."""

    total_prompts: int
    matched_prompts: int
    mismatched_prompts: int
    overall_match_rate: float
    passed: bool
    workload_breakdown: Dict[str, Dict[str, int]]
    discrepancy_details: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_to_json(self, filepath: Path | str) -> Path:
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return out_path


def verify_token_sequence(
    vanilla_tokens: Sequence[int],
    adaptive_tokens: Sequence[int],
    prompt_id: str = "",
    workload: str = "W1",
    tokenizer: Any = None,
    max_context_tokens: int = 5,
) -> LosslessVerificationResult:
    """Verifies token-by-token equivalence between Vanilla and Adaptive sequences."""
    len_v = len(vanilla_tokens)
    len_a = len(adaptive_tokens)
    max_len = max(len_v, len_a)

    if max_len == 0:
        return LosslessVerificationResult(
            prompt_id=prompt_id,
            workload=workload,
            is_match=True,
            match_rate=1.0,
            vanilla_token_count=0,
            adaptive_token_count=0,
        )

    discrepancies: List[TokenDiscrepancy] = []
    matched_count = 0

    for i in range(max_len):
        tok_v = vanilla_tokens[i] if i < len_v else None
        tok_a = adaptive_tokens[i] if i < len_a else None

        if tok_v == tok_a:
            matched_count += 1
        else:
            # Build context window string
            start_ctx = max(0, i - max_context_tokens)
            end_ctx = min(max_len, i + max_context_tokens + 1)
            ctx_v = list(vanilla_tokens[start_ctx:end_ctx])
            ctx_a = list(adaptive_tokens[start_ctx:end_ctx])

            context_str = f"Expected slice: {ctx_v} | Actual slice: {ctx_a}"
            exp_text = None
            act_text = None
            if tokenizer is not None:
                try:
                    exp_text = tokenizer.decode([tok_v]) if tok_v is not None else "<EOF>"
                    act_text = tokenizer.decode([tok_a]) if tok_a is not None else "<EOF>"
                except Exception:
                    pass

            disc = TokenDiscrepancy(
                index=i,
                expected_token_id=tok_v,
                actual_token_id=tok_a,
                expected_text=exp_text,
                actual_text=act_text,
                context_window=context_str,
            )
            discrepancies.append(disc)

    match_rate = matched_count / max_len
    is_match = len(discrepancies) == 0

    return LosslessVerificationResult(
        prompt_id=prompt_id,
        workload=workload,
        is_match=is_match,
        match_rate=match_rate,
        vanilla_token_count=len_v,
        adaptive_token_count=len_a,
        first_discrepancy=discrepancies[0] if discrepancies else None,
        all_discrepancies=discrepancies,
    )


def verify_text_output(
    vanilla_text: str,
    adaptive_text: str,
    prompt_id: str = "",
    workload: str = "W1",
    max_context_chars: int = 40,
) -> LosslessVerificationResult:
    """Verifies character-by-character equivalence between Vanilla and Adaptive strings."""
    len_v = len(vanilla_text)
    len_a = len(adaptive_text)

    if vanilla_text == adaptive_text:
        return LosslessVerificationResult(
            prompt_id=prompt_id,
            workload=workload,
            is_match=True,
            match_rate=1.0,
            vanilla_token_count=len_v,
            adaptive_token_count=len_a,
            vanilla_text=vanilla_text,
            adaptive_text=adaptive_text,
        )

    # Find first divergence index
    mismatch_idx = 0
    min_len = min(len_v, len_a)
    while mismatch_idx < min_len and vanilla_text[mismatch_idx] == adaptive_text[mismatch_idx]:
        mismatch_idx += 1

    start_ctx = max(0, mismatch_idx - max_context_chars)
    end_ctx_v = min(len_v, mismatch_idx + max_context_chars)
    end_ctx_a = min(len_a, mismatch_idx + max_context_chars)

    exp_slice = vanilla_text[start_ctx:end_ctx_v]
    act_slice = adaptive_text[start_ctx:end_ctx_a]

    disc = TokenDiscrepancy(
        index=mismatch_idx,
        expected_text=vanilla_text[mismatch_idx : mismatch_idx + 15] if mismatch_idx < len_v else "<EOF>",
        actual_text=adaptive_text[mismatch_idx : mismatch_idx + 15] if mismatch_idx < len_a else "<EOF>",
        context_window=f"...{exp_slice}... vs ...{act_slice}...",
    )

    max_len = max(len_v, len_a)
    # Estimate match rate based on common prefix
    match_rate = mismatch_idx / max_len if max_len > 0 else 0.0

    return LosslessVerificationResult(
        prompt_id=prompt_id,
        workload=workload,
        is_match=False,
        match_rate=match_rate,
        vanilla_token_count=len_v,
        adaptive_token_count=len_a,
        first_discrepancy=disc,
        all_discrepancies=[disc],
        vanilla_text=vanilla_text,
        adaptive_text=adaptive_text,
    )


def aggregate_lossless_results(
    results: Sequence[LosslessVerificationResult],
    min_required_prompts: int = 50,
) -> LosslessBatchSummary:
    """Aggregates a batch of prompt verification results into an overall summary."""
    total = len(results)
    matched = sum(1 for r in results if r.is_match)
    mismatched = total - matched

    workload_stats: Dict[str, Dict[str, int]] = {}
    discrepancies_list: List[Dict[str, Any]] = []

    for r in results:
        wl = r.workload or "unknown"
        if wl not in workload_stats:
            workload_stats[wl] = {"total": 0, "matched": 0, "mismatched": 0}
        workload_stats[wl]["total"] += 1
        if r.is_match:
            workload_stats[wl]["matched"] += 1
        else:
            workload_stats[wl]["mismatched"] += 1
            if r.first_discrepancy:
                discrepancies_list.append(
                    {
                        "prompt_id": r.prompt_id,
                        "workload": r.workload,
                        "match_rate": r.match_rate,
                        "first_discrepancy": r.first_discrepancy.to_dict(),
                    }
                )

    overall_rate = matched / total if total > 0 else 0.0
    passed = mismatched == 0 and total >= min_required_prompts

    return LosslessBatchSummary(
        total_prompts=total,
        matched_prompts=matched,
        mismatched_prompts=mismatched,
        overall_match_rate=overall_rate,
        passed=passed,
        workload_breakdown=workload_stats,
        discrepancy_details=discrepancies_list,
    )


def export_lossless_summary(
    summary: LosslessBatchSummary,
    output_path: Path | str = "experiments/results/m2_lossless_verification_summary.json",
) -> Path:
    """Saves the summary to a JSON file."""
    return summary.save_to_json(output_path)
