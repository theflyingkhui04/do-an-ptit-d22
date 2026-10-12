"""Glue tests for AdaptiveAsyncScheduler against a stubbed vLLM AsyncScheduler.

The stub mimics only what the glue touches (vLLM v0.30.0 names). Real vLLM
compatibility is verified by the Colab smoke test (scripts/smoke_lane3_adaptive.py).
"""
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest


class FakeAsyncScheduler:
    """Minimal stand-in for vllm.v1.core.sched.async_scheduler.AsyncScheduler."""

    def __init__(self, *args, num_spec_tokens=4, max_num_seqs=8, dynamic=True, **kwargs):
        self.num_spec_tokens = num_spec_tokens
        self.running = []
        self.requests = {}
        self.kv_usage = 0.10
        self.dynamic_sd_lookup = [0] + [num_spec_tokens] * max_num_seqs if dynamic else None
        self.freed = []
        self.stats_calls = []

    def get_kv_cache_usage(self):
        return self.kv_usage

    def schedule(self, throttle_prefills=False):
        # Like vLLM: batch K is read from the lookup by batch size.
        k = self.dynamic_sd_lookup[len(self.running)] if self.dynamic_sd_lookup else self.num_spec_tokens
        return SimpleNamespace(
            num_scheduled_tokens={r.request_id: 1 for r in self.running},
            num_spec_tokens_to_schedule=k,
        )

    def _update_after_schedule(self, scheduler_output):
        placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            self.requests[req_id].spec_token_ids = placeholders

    def make_spec_decoding_stats(self, stats, num_draft_tokens, num_accepted_tokens, invalid, request_id):
        self.stats_calls.append((request_id, num_draft_tokens, num_accepted_tokens))
        return stats

    def _free_request(self, request, delay_free_blocks=False):
        self.freed.append(request.request_id)
        return None, None


@pytest.fixture
def scheduler_module(monkeypatch):
    vllm = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    async_mod = types.ModuleType("vllm.v1.core.sched.async_scheduler")
    async_mod.AsyncScheduler = FakeAsyncScheduler
    for name, mod in {
        "vllm": vllm,
        "vllm.logger": logger_mod,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.async_scheduler": async_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "src.server.adaptive_scheduler", raising=False)
    return importlib.import_module("src.server.adaptive_scheduler")


def make_request(req_id, prefill=False):
    return SimpleNamespace(request_id=req_id, is_prefill_chunk=prefill, spec_token_ids=[])


def build(scheduler_module, monkeypatch, tmp_path, **env):
    monkeypatch.setenv("ADAPTIVE_CONTROLLER_KWARGS", json.dumps({"initial_alpha": 0.85}))
    monkeypatch.setenv("ADAPTIVE_TELEMETRY_PATH", str(tmp_path / "lane3.jsonl"))
    monkeypatch.setenv("ADAPTIVE_ENABLE_VRAM_GUARD", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return scheduler_module.AdaptiveAsyncScheduler()


def add_running(scheduler, *reqs):
    for req in reqs:
        scheduler.running.append(req)
        scheduler.requests[req.request_id] = req


def run_step(scheduler):
    out = scheduler.schedule()
    scheduler._update_after_schedule(out)
    return out


def test_schedule_writes_batch_k_into_lookup(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    add_running(sched, make_request("r1"), make_request("r2"))

    out = run_step(sched)

    assert out.num_spec_tokens_to_schedule == 3
    assert set(sched.dynamic_sd_lookup[1:]) == {3}


def test_placeholders_trimmed_per_request(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    good, bad = make_request("good"), make_request("bad")
    add_running(sched, good, bad)
    run_step(sched)
    sched.make_spec_decoding_stats(None, 3, 0, None, "bad")

    run_step(sched)

    assert len(good.spec_token_ids) == 3
    assert bad.spec_token_ids == []


def test_prefill_requests_are_not_planned(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    add_running(sched, make_request("p1", prefill=True))

    run_step(sched)

    assert sched._plan.per_request_k == {}


def test_kv_pressure_drops_batch_k_to_zero(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    add_running(sched, make_request("r1"))
    sched.kv_usage = 0.95

    out = run_step(sched)

    assert out.num_spec_tokens_to_schedule == 0


def test_feedback_reaches_controller_and_super(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    add_running(sched, make_request("r1"))
    run_step(sched)

    sched.make_spec_decoding_stats(None, 3, 2, {"r1": 1}, "r1")

    assert sched.stats_calls == [("r1", 3, 2)]
    controller = sched._planner.adapter.get_controller("r1")
    assert controller.step_count == 1


def test_free_request_writes_telemetry_line(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    req = make_request("r1")
    add_running(sched, req)
    run_step(sched)

    sched._free_request(req)

    lines = (tmp_path / "lane3.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["tree_budgets"] == [3]
    assert sched.freed == ["r1"]


def test_planner_error_does_not_crash_schedule(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    add_running(sched, make_request("r1"))

    def boom(*args, **kwargs):
        raise RuntimeError("planner bug")

    monkeypatch.setattr(sched._planner, "plan_step", boom)

    out = run_step(sched)

    assert out is not None


def test_max_k_capped_by_num_spec_tokens(scheduler_module, monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTIVE_MAX_K", "4")
    sched = build(scheduler_module, monkeypatch, tmp_path)

    assert sched._planner.max_k == 4
    sched2 = scheduler_module.AdaptiveAsyncScheduler(num_spec_tokens=3)
    assert sched2._planner.max_k == 3


def test_per_request_disabled_keeps_batch_placeholders(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_PER_REQUEST="0")
    good, bad = make_request("good"), make_request("bad")
    add_running(sched, good, bad)
    run_step(sched)
    sched.make_spec_decoding_stats(None, 3, 0, None, "bad")

    run_step(sched)

    assert len(bad.spec_token_ids) == 3


def test_missing_dynamic_lookup_still_schedules(scheduler_module, monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTIVE_ENABLE_VRAM_GUARD", "0")
    sched = scheduler_module.AdaptiveAsyncScheduler(dynamic=False)
    add_running(sched, make_request("r1"))

    out = run_step(sched)

    assert out.num_spec_tokens_to_schedule == 4
