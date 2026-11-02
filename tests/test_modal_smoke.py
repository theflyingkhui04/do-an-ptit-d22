"""Smoke and client checks over the Modal lanes, using a fake lane client."""

from __future__ import annotations

import json

import pytest

from experiments import modal_smoke as smoke
from src.client import modal_client as mc


class FakeLane:
    def __init__(self, lane, *, margins_supported=True, fail=False):
        self.lane = lane
        self.model = mc.LANE_MODELS[lane]
        self.margins_supported = margins_supported
        self.fail = fail
        self.calls = []

    def stream(self, messages, **params):
        self.calls.append(("stream", messages, params))
        if self.fail:
            raise mc.ModalClientError(f"{self.lane}: HTTP 500")
        text = '{"seller": "x"}' if "json" in json.dumps(messages).lower() else "Xin chào"
        yield mc.StreamEvent("delta", text)
        metrics = None if self.lane == "vanilla" else {"speculative_decoding": {"num_spec_tokens": 3}}
        yield mc.StreamEvent("final", text, usage={"completion_tokens": 4}, metrics=metrics, ttft_s=0.2, total_s=0.9)

    def chat(self, messages, **params):
        self.calls.append(("chat", messages, params))
        return mc.ChatResult("ok", {"completion_tokens": 1}, None, {})

    def margins(self, messages, **params):
        self.calls.append(("margins", messages, params))
        if not self.margins_supported:
            raise mc.LogitMarginUnsupported(f"{self.lane}: response has no logprobs")
        return [mc.TokenMargin("a", -0.1, 2.5), mc.TokenMargin("b", -0.2, 0.0)]


def factory(**options):
    made = {}

    def make(lane):
        made[lane] = FakeLane(lane, **options.get(lane, {}))
        return made[lane]

    make.made = made
    return make


def test_smoke_samples_are_fixed_and_come_from_frozen_workloads():
    assert smoke.SMOKE_SAMPLES == (("W1", 0), ("W1", 1), ("W2", 0), ("W2", 1), ("W3", 0))
    rows = smoke.load_smoke_rows()

    assert [label for label, _ in rows] == ["W1", "W1", "W2", "W2", "W3"]
    assert rows[0][1]["sample_id"] == "m0_04_001"
    assert rows[4][1]["sample_id"] == "w3_business_v2_001"


def test_smoke_runs_5_prompts_per_lane_with_canonical_params():
    make = factory()

    records = smoke.run_smoke(["vanilla", "static", "adaptive"], make)

    assert len(records) == 15
    assert all(r["ok"] for r in records)
    first = records[0]
    for key in ("lane", "workload", "sample_id", "ok", "ttft_s", "total_s", "completion_tokens", "json_valid"):
        assert key in first
    _, messages, params = make.made["vanilla"].calls[0]
    assert messages  # same messages as the frozen request
    assert "model" not in params
    assert "stream" not in params
    assert "max_completion_tokens" in params


def test_smoke_runs_eagle_lane():
    make = factory()
    records = smoke.run_smoke(["eagle"], make)

    assert len(records) == 5
    assert all(r["ok"] for r in records)
    assert all(r["lane"] == "eagle" for r in records)


def test_smoke_records_failures_instead_of_crashing():
    records = smoke.run_smoke(["static"], factory(static={"fail": True}))

    assert len(records) == 5
    assert not any(r["ok"] for r in records)
    assert "HTTP 500" in records[0]["error"]


def test_json_valid_only_for_w1():
    records = smoke.run_smoke(["vanilla"], factory())

    assert all(r["json_valid"] in (True, False) for r in records if r["workload"] == "W1")
    assert all(r["json_valid"] is None for r in records if r["workload"] != "W1")


def test_client_check_reports_margins_or_unsupported():
    report = smoke.client_check(["vanilla", "static"], factory(static={"margins_supported": False}))

    by_lane = {r["lane"]: r for r in report}
    assert by_lane["vanilla"]["chat_ok"] is True
    assert by_lane["vanilla"]["stream_usage"] == {"completion_tokens": 4}
    assert by_lane["vanilla"]["margins"] == [{"token": "a", "logprob": -0.1, "margin": 2.5},
                                             {"token": "b", "logprob": -0.2, "margin": 0.0}]
    assert by_lane["static"]["stream_metrics"]["speculative_decoding"]["num_spec_tokens"] == 3
    assert by_lane["static"]["margins"] is None
    assert "no logprobs" in by_lane["static"]["margins_unsupported"]


@pytest.mark.parametrize("run_id", ["../x", "a/b", ""])
def test_evidence_path_rejects_unsafe_run_id(run_id):
    with pytest.raises(ValueError, match="run-id"):
        smoke.evidence_path(run_id, "smoke.json")


def test_main_writes_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(smoke, "make_client", factory())

    assert smoke.main(["smoke", "--run-id", "t1", "--lanes", "vanilla"]) == 0
    data = json.loads((tmp_path / "experiments/runs/m3_01/t1/smoke.json").read_text(encoding="utf-8"))
    assert data["summary"] == {"total": 5, "ok": 5}
    assert "ws-" not in json.dumps(data)


@pytest.mark.parametrize("text, valid", [
    ('{"a": 1}', True),
    ('```json\n{"a": 1}\n```', True),
    ('```\n{"a": 1}\n```', True),
    ("not json", False),
    ("```json\n{broken\n```", False),
])
def test_json_valid_handles_markdown_fences(text, valid):
    assert smoke._json_valid(text) is valid
