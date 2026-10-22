"""Tests for M2-03 Lossless Verifier: Token Equivalence and Discrepancy Detection."""
import json
import tempfile
from pathlib import Path
import pytest

from experiments.lossless_verifier import (
    TokenDiscrepancy,
    LosslessVerificationResult,
    LosslessBatchSummary,
    verify_token_sequence,
    verify_text_output,
    aggregate_lossless_results,
    export_lossless_summary,
)


def test_exact_token_match():
    tokens_v = [101, 2054, 2003, 1037, 3899, 102]
    tokens_a = [101, 2054, 2003, 1037, 3899, 102]

    result = verify_token_sequence(
        vanilla_tokens=tokens_v,
        adaptive_tokens=tokens_a,
        prompt_id="prompt_001",
        workload="W1",
    )

    assert result.is_match is True
    assert result.match_rate == 1.0
    assert result.vanilla_token_count == 6
    assert result.adaptive_token_count == 6
    assert result.first_discrepancy is None
    assert len(result.all_discrepancies) == 0


def test_token_discrepancy_detection():
    # Mismatch at index 3: 1037 vs 9999
    tokens_v = [101, 2054, 2003, 1037, 3899, 102]
    tokens_a = [101, 2054, 2003, 9999, 3899, 102]

    result = verify_token_sequence(
        vanilla_tokens=tokens_v,
        adaptive_tokens=tokens_a,
        prompt_id="prompt_002",
        workload="W3",
    )

    assert result.is_match is False
    assert result.match_rate == pytest.approx(5 / 6)
    assert result.first_discrepancy is not None
    assert result.first_discrepancy.index == 3
    assert result.first_discrepancy.expected_token_id == 1037
    assert result.first_discrepancy.actual_token_id == 9999
    assert len(result.all_discrepancies) == 1


def test_token_length_mismatch():
    # Adaptive output is truncated early (length 4 vs 6)
    tokens_v = [101, 2054, 2003, 1037, 3899, 102]
    tokens_a = [101, 2054, 2003, 1037]

    result = verify_token_sequence(
        vanilla_tokens=tokens_v,
        adaptive_tokens=tokens_a,
        prompt_id="prompt_003",
        workload="W2",
    )

    assert result.is_match is False
    assert result.vanilla_token_count == 6
    assert result.adaptive_token_count == 4
    assert result.match_rate == pytest.approx(4 / 6)
    assert result.first_discrepancy is not None
    assert result.first_discrepancy.index == 4
    assert result.first_discrepancy.expected_token_id == 3899
    assert result.first_discrepancy.actual_token_id is None


def test_empty_tokens_edge_case():
    # Both empty
    res_both_empty = verify_token_sequence([], [])
    assert res_both_empty.is_match is True
    assert res_both_empty.match_rate == 1.0

    # One empty
    res_one_empty = verify_token_sequence([1, 2, 3], [])
    assert res_one_empty.is_match is False
    assert res_one_empty.match_rate == 0.0
    assert res_one_empty.first_discrepancy.index == 0


def test_verify_text_output_exact_and_mismatch():
    text_v = "Công ty TNHH Giải pháp Phần mềm"
    text_a = "Công ty TNHH Giải pháp Phần mềm"

    res_match = verify_text_output(text_v, text_a, prompt_id="txt_01", workload="W1")
    assert res_match.is_match is True
    assert res_match.match_rate == 1.0
    assert res_match.first_discrepancy is None

    text_diff = "Công ty TNHH Giải pháp Công nghệ"
    res_diff = verify_text_output(text_v, text_diff, prompt_id="txt_02", workload="W3")
    assert res_diff.is_match is False
    assert res_diff.first_discrepancy is not None
    assert res_diff.first_discrepancy.index == 23  # index where "Phần mềm" vs "Công nghệ" starts
    assert "Phần mềm" in res_diff.first_discrepancy.expected_text
    assert "Công nghệ" in res_diff.first_discrepancy.actual_text


def test_batch_aggregation_and_summary_export():
    results = []
    # 50 prompts: 49 match, 1 mismatch
    for i in range(49):
        results.append(
            LosslessVerificationResult(
                prompt_id=f"p_{i:03d}",
                workload="W1" if i < 20 else ("W2" if i < 35 else "W3"),
                is_match=True,
                match_rate=1.0,
                vanilla_token_count=100,
                adaptive_token_count=100,
            )
        )
    # 1 mismatch prompt
    results.append(
        LosslessVerificationResult(
            prompt_id="p_049",
            workload="W3",
            is_match=False,
            match_rate=0.98,
            vanilla_token_count=100,
            adaptive_token_count=100,
            first_discrepancy=TokenDiscrepancy(index=50, expected_token_id=123, actual_token_id=456),
        )
    )

    summary = aggregate_lossless_results(results)
    assert summary.total_prompts == 50
    assert summary.matched_prompts == 49
    assert summary.mismatched_prompts == 1
    assert summary.passed is False  # Requires 100% match
    assert "W3" in summary.workload_breakdown
    assert summary.workload_breakdown["W3"]["mismatched"] == 1
    assert len(summary.discrepancy_details) == 1

    # Now test 50/50 matched
    results[49] = LosslessVerificationResult(
        prompt_id="p_049",
        workload="W3",
        is_match=True,
        match_rate=1.0,
        vanilla_token_count=100,
        adaptive_token_count=100,
    )
    summary_perfect = aggregate_lossless_results(results)
    assert summary_perfect.passed is True
    assert summary_perfect.matched_prompts == 50
    assert summary_perfect.mismatched_prompts == 0

    # Export to json
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "summary.json"
        saved_path = export_lossless_summary(summary_perfect, out_file)
        assert saved_path.exists()
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["passed"] is True
        assert data["total_prompts"] == 50
        assert data["matched_prompts"] == 50
