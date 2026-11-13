"""All token captures below are SYNTHETIC comparator fixtures, never model results."""
from copy import deepcopy

import pytest

from experiments.m4_lossless import compare_captures
from experiments.m4_treatments import TREATMENTS, TARGET, TARGET_REVISION

SHA = "a" * 40


def fixtures():
    expected = [dict(workload=w, prompt_id="p", prompt_hash=w, max_tokens=32,
                     stopping="EOS", tokenizer_revision=TARGET_REVISION) for w in ("W1", "W2", "W3")]
    captures = [dict(e, treatment=t, repeat=n, source_commit=SHA, target_model=TARGET,
                     target_revision=TARGET_REVISION, temperature=0, seed=42, token_ids=[1, 2, 3],
                     complete=True, finish_reason="stop") for e in expected for t in TREATMENTS
                for n in range(2 if t == "vanilla" else 1)]
    return expected, captures


def test_complete_coverage_and_provenance():
    expected, captures = fixtures()
    result = compare_captures(captures, expected, SHA)
    assert result["passed"] and result["prompt_count"] == 3
    assert set(result["workload_breakdown"]) == {"W1", "W2", "W3"}
    assert result["comparisons"][0]["baseline"]["source_commit"] == SHA


@pytest.mark.parametrize("tokens,index", [([9, 2, 3], 0), ([1, 9, 3], 1), ([1, 2], 2), ([1, 2, 3, 4], 3)])
def test_single_mismatch_fails(tokens, index):
    expected, captures = fixtures()
    captures[-1]["token_ids"] = tokens
    report = compare_captures(captures, expected, SHA)
    assert not report["passed"]
    assert report["comparisons"][-1]["first_discrepancy"]["index"] == index


@pytest.mark.parametrize("change", [{"token_ids": []}, {"token_ids": [True]}, {"token_ids": ["1"]},
    {"complete": False}, {"finish_reason": None}, {"prompt_hash": "wrong"}, {"source_commit": "b" * 40},
    {"target_revision": "wrong"}, {"temperature": 1}, {"max_tokens": 99}])
def test_invalid_capture_rejected(change):
    expected, captures = fixtures()
    captures[-1].update(change)
    with pytest.raises(ValueError):
        compare_captures(captures, expected, SHA)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "duplicate_expected", "empty"])
def test_coverage_rejected(kind):
    expected, captures = fixtures()
    if kind == "missing": captures.pop()
    if kind == "duplicate": captures.append(deepcopy(captures[0]))
    if kind == "duplicate_expected": expected.append(deepcopy(expected[0]))
    if kind == "empty": expected = []
    with pytest.raises(ValueError): compare_captures(captures, expected, SHA)
