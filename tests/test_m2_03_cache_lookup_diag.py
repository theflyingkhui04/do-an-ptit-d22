"""Offline contract tests for M2-03 cache lookup probe."""

import json
from types import SimpleNamespace

import pytest

from src.server import m2_03_cache_lookup_diag as diag


@pytest.fixture
def sample(tmp_path, monkeypatch):
    output = tmp_path / "cache_lookup.jsonl"
    monkeypatch.setenv("M203_CACHE_LOOKUP_JSONL", str(output))

    request = SimpleNamespace(
        request_id="test-request-001",
        prompt_token_ids=[10, 20, 30, 40],
        num_tokens=4,
        num_computed_tokens=0,
    )

    manager = SimpleNamespace(enable_caching=True)

    blocks = SimpleNamespace(
        blocks=([1, 2],)
    )

    result = (blocks, 3, 0)
    return output, request, manager, result


def test_lookup_preserves_original_result(sample, monkeypatch):
    output, request, manager, result = sample
    calls = []

    def original(obj, req):
        calls.append((obj, req))
        return result

    monkeypatch.setattr(diag, "_ORIGINAL", original)

    actual = diag._observed_lookup(manager, request)

    assert actual is result
    assert calls == [(manager, request)]
    assert output.exists()

    records = [
        json.loads(line)
        for line in output.read_text().splitlines()
    ]

    assert len(records) == 1
    row = records[0]

    assert row["request_id"] == "test-request-001"
    assert row["cached_prefix_tokens"] == 3
    assert row["shared_prefix_boundary"] == 0
    assert row["cached_blocks_per_group"] == [2]
    assert row["cache_enabled"] is True


def test_cache_miss_is_recorded(sample, monkeypatch):
    output, request, manager, result = sample
    blocks = SimpleNamespace(blocks=([],))
    expected = (blocks, 0, 0)

    monkeypatch.setattr(
        diag, "_ORIGINAL",
        lambda obj, req: expected
    )

    actual = diag._observed_lookup(manager, request)

    assert actual is expected

    row = json.loads(output.read_text().strip())
    assert row["cached_prefix_tokens"] == 0
    assert row["cached_blocks_per_group"] == [0]


def test_cache_disabled_record(sample, monkeypatch):
    output, request, manager, result = sample
    manager.enable_caching = False

    expected = (SimpleNamespace(blocks=([],)), 0, 0)

    monkeypatch.setattr(
        diag, "_ORIGINAL",
        lambda obj, req: expected
    )

    assert diag._observed_lookup(manager, request) is expected

    row = json.loads(output.read_text().strip())
    assert row["cache_enabled"] is False
    assert row["cached_prefix_tokens"] == 0


def test_logging_failure_does_not_change_lookup(
    sample, monkeypatch, capsys
):
    output, request, manager, result = sample

    monkeypatch.setattr(
        diag, "_ORIGINAL",
        lambda obj, req: result
    )

    def fail_write(record):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(diag, "_write_record", fail_write)

    actual = diag._observed_lookup(manager, request)

    assert actual is result
    assert "M203_CACHE_PROBE_ERROR" in capsys.readouterr().err


def test_original_exception_propagates(sample, monkeypatch):
    output, request, manager, result = sample

    def original_failure(obj, req):
        raise ValueError("original lookup failure")

    monkeypatch.setattr(
        diag, "_ORIGINAL",
        original_failure
    )

    with pytest.raises(
        ValueError,
        match="original lookup failure"
    ):
        diag._observed_lookup(manager, request)

    assert not output.exists()
