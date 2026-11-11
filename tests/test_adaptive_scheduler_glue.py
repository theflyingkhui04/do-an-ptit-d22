"""Glue tests for AdaptiveAsyncScheduler against a stubbed vLLM AsyncScheduler.

L2 only: the stub mirrors the v0.30.0 source contract, not an installed
engine smoke or GPU verification. See the M4-02-04 worklog for source links.
"""
import importlib
import ast
import __future__
import json
import sys
import types
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path

import pytest


PINNED_SCHEDULER = (Path(__file__).resolve().parents[1] / "experiments/runs/m2_03"
    / "boundary279_20261018T132244Z_15f4dae9/vllm-originals/v1/core/sched/scheduler.py")


@lru_cache(maxsize=None)
def pinned_method(name):
    """L2 source fixture only; never imports or runs an installed vLLM engine."""
    module = ast.parse(PINNED_SCHEDULER.read_text(encoding="utf-8"))
    scheduler = next(node for node in module.body if isinstance(node, ast.ClassDef)
                     and node.name == "Scheduler")
    return next(node for node in scheduler.body if isinstance(node, ast.FunctionDef)
                and node.name == name)


@lru_cache(maxsize=1)
def pinned_lookup_code():
    # Execute the captured upstream lookup statements rather than duplicating
    # our expected indexing rule inside the fake scheduler.
    body = pinned_method("schedule").body
    index = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "num_spec_tokens_to_schedule"
                         for t in node.targets))
    assert isinstance(body[index + 1], ast.If)
    return compile(ast.Module(body=body[index:index + 2], type_ignores=[]),
                   str(PINNED_SCHEDULER), "exec")


def pinned_lookup(scheduler, scheduled):
    scope = {"self": scheduler, "num_scheduled_tokens": scheduled}
    exec(pinned_lookup_code(), scope)
    return scope["num_spec_tokens_to_schedule"]


def call_pinned_method(name, scheduler, *args):
    namespace = {}
    code = compile(ast.Module(body=[pinned_method(name)], type_ignores=[]),
                   str(PINNED_SCHEDULER), "exec", flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    return namespace[name](scheduler, *args)


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
        self.scheduled_ids = None

    def get_kv_cache_usage(self):
        return self.kv_usage

    def schedule(self, throttle_prefills=False):
        # v0.30.0 Scheduler.schedule reads by actual scheduled count, then
        # calls _update_after_schedule before returning. Existing placeholders
        # represent drafts verified NOW; K controls this output's proposer pass
        # for drafts that will be verified by a later output.
        scheduled = {r.request_id: 1 for r in self.running
                     if self.scheduled_ids is None or r.request_id in self.scheduled_ids}
        k = pinned_lookup(self, scheduled)
        out = SimpleNamespace(
            num_scheduled_tokens=scheduled,
            num_spec_tokens_to_schedule=k,
            scheduled_spec_decode_tokens={r.request_id: list(r.spec_token_ids)
                for r in self.running if r.request_id in scheduled and r.spec_token_ids},
        )
        self._update_after_schedule(out)
        return out

    def _update_after_schedule(self, scheduler_output):
        placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            # Scheduler's base hook recomputes this flag before AsyncScheduler
            # assigns future placeholders. Model just the completion boundary.
            if getattr(self.requests[req_id], "completes_prefill", False):
                self.requests[req_id].is_prefill_chunk = False
            if self.requests[req_id].is_prefill_chunk:
                continue
            self.requests[req_id].spec_token_ids = placeholders

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        self.stats_calls.append((request_id, num_draft_tokens, num_accepted_tokens))
        return spec_decoding_stats

    def _free_request(self, request, delay_free_blocks=False):
        self.freed.append(request.request_id)
        if not delay_free_blocks:
            self.requests.pop(request.request_id, None)
        self.running = [r for r in self.running if r.request_id != request.request_id]
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
    return scheduler.schedule()


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


def _fake_torch(initialized, free_bytes=2 * 1024 * 1024 * 1024):
    calls = []

    def mem_get_info():
        calls.append("mem_get_info")
        return free_bytes, 16 * 1024 * 1024 * 1024

    cuda = SimpleNamespace(is_initialized=lambda: initialized, mem_get_info=mem_get_info)
    return types.SimpleNamespace(cuda=cuda), calls


def test_free_vram_read_when_cuda_initialized(scheduler_module, monkeypatch, tmp_path):
    fake_torch, calls = _fake_torch(initialized=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_ENABLE_VRAM_GUARD="1")

    assert sched._free_vram_mb() == pytest.approx(2048.0)
    assert calls == ["mem_get_info"]


def test_free_vram_not_read_without_cuda_context(scheduler_module, monkeypatch, tmp_path):
    # Reading VRAM would create a new CUDA context (~300 MB) in this process.
    fake_torch, calls = _fake_torch(initialized=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_ENABLE_VRAM_GUARD="1")

    assert sched._free_vram_mb() is None
    assert calls == []


def test_missing_dynamic_lookup_still_schedules(scheduler_module, monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTIVE_ENABLE_VRAM_GUARD", "0")
    sched = scheduler_module.AdaptiveAsyncScheduler(dynamic=False)
    add_running(sched, make_request("r1"))

    out = run_step(sched)

    assert out.num_spec_tokens_to_schedule == 4


def test_current_lookup_read_uses_scheduled_subset(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    add_running(sched, make_request("r1"), make_request("r2"), make_request("p", prefill=True))
    sched.scheduled_ids = {"r2"}
    # Index one must be updated even though the planner saw three running requests.
    sched.dynamic_sd_lookup[1] = 4
    out = sched.schedule()
    assert out.num_spec_tokens_to_schedule == 3
    assert list(out.num_scheduled_tokens) == ["r2"]
    assert sched._plan.per_request_k == {"r1": 3, "r2": 3}
    assert sched.requests["p"].spec_token_ids == []


def test_proposer_k_and_current_verification_budget_are_distinct(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    req = make_request("r")
    req.spec_token_ids = [-1] * 4
    add_running(sched, req)
    out = sched.schedule()
    assert len(out.scheduled_spec_decode_tokens["r"]) == 4
    # This output selects proposer K=3 for drafts verified in a future output.
    assert out.num_spec_tokens_to_schedule == len(req.spec_token_ids) == 3


def test_scheduling_telemetry_reports_placeholders_not_execution(scheduler_module, monkeypatch, tmp_path):
    step_path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(step_path))
    req, prefill = make_request("r"), make_request("p", prefill=True)
    add_running(sched, req, prefill)
    sched.schedule()
    records = [json.loads(line) for line in step_path.read_text().splitlines()]
    scheduled = [r for r in records if r["record_type"] == "scheduled_budget"]
    assert len(scheduled) == 1
    assert scheduled[0]["request_id"] == "r"
    assert scheduled[0]["next_request_placeholder_budget"] == 3
    assert scheduled[0]["executed_draft_token_count"] is None
    assert scheduled[0]["executed_runtime_mode"] == "unverified"
    assert scheduled[0]["scheduled_runtime_mode"] == "linear_dynamic_k"
    assert scheduled[0]["scheduled_scalar_k"] == 3
    assert scheduled[0]["plan_sequence"] == records[0]["plan_sequence"]
    assert scheduled[0]["schedule_sequence"] == 1
    sched._free_request(req)
    final = json.loads((tmp_path / "lane3.jsonl").read_text())
    assert final["tree_execution_verified"] is False
    assert final["executed_runtime_mode"] == "unverified"
    assert final["scheduled_runtime_mode"] == "linear_dynamic_k"


def test_empty_schedule_keeps_upstream_default_but_plan_is_neutral(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    out = sched.schedule()
    assert sched._plan.batch_k == sched._plan.batch_node_count == 0
    assert out.num_spec_tokens_to_schedule == 4  # upstream empty-output default


def test_free_one_request_preserves_other_controller(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    a, b = make_request("a"), make_request("b")
    add_running(sched, a, b)
    sched.schedule()
    controller_b = sched._planner.adapter.get_controller("b")
    sched._free_request(a)
    assert "a" not in sched._planner._last_decisions
    assert "a" not in sched._plan.per_request_k
    assert "a" not in sched._plan.per_request_decisions
    assert "a" not in sched._plan.per_request_node_count
    assert sched._planner.adapter.active_request_count == 1
    assert sched._planner.adapter.get_controller("b") is controller_b


def test_delayed_feedback_does_not_inherit_latest_plan(scheduler_module, monkeypatch, tmp_path):
    path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    add_running(sched, make_request("r"))
    sched.schedule()
    sched.schedule()  # second plan exists before first verification arrives
    sched.make_spec_decoding_stats(None, 4, 1, {"r": 1}, "r")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    feedback = records[-1]
    assert feedback["record_type"] == "verification_feedback"
    assert feedback["executed_draft_token_count"] == 3
    assert feedback["accepted_draft_token_count"] == 1
    assert "selected_topology" not in feedback
    assert feedback["source_plan_sequence"] is None
    assert feedback["source_schedule_sequence"] is None
    assert feedback["executed_runtime_mode"] == "unverified"
    assert sched._planner.adapter.get_controller("r").alpha_ema == pytest.approx(0.8 * 0.85 + 0.2 / 3)


def test_pinned_lookup_and_hook_order_contract():
    body = pinned_method("schedule").body
    lookup_line = next(node.lineno for node in ast.walk(pinned_method("schedule"))
                       if isinstance(node, ast.Subscript)
                       and ast.unparse(node.value) == "self.dynamic_sd_lookup")
    hook_line = next(node.lineno for node in ast.walk(pinned_method("schedule"))
                     if isinstance(node, ast.Call)
                     and ast.unparse(node.func) == "self._update_after_schedule")
    assert lookup_line < hook_line < body[-1].lineno
    assert isinstance(body[-1], ast.Return)
    for name in ("make_spec_decoding_stats", "_free_request"):
        import inspect
        assert list(inspect.signature(getattr(FakeAsyncScheduler, name)).parameters) == [
            arg.arg for arg in pinned_method(name).args.args]


def test_prefill_completion_observation_does_not_invent_plan(scheduler_module, monkeypatch, tmp_path):
    path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    req = make_request("p", prefill=True)
    req.completes_prefill = True
    add_running(sched, req)
    sched.schedule()
    record = json.loads(path.read_text())
    assert record["record_type"] == "scheduled_budget"
    assert record["plan_sequence"] is None
    assert record["schedule_sequence"] == 1
    assert record["executed_runtime_mode"] == "unverified"


def test_trimmed_placeholders_reach_pinned_draft_update_and_feedback(scheduler_module, monkeypatch, tmp_path):
    path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    reqs = [make_request(name) for name in ("two", "four", "zero")]
    for req, alpha in zip(reqs, (0.7, 0.95, 0.4)):
        req.is_finished = lambda: False
        sched._planner.adapter.start_request(req.request_id, initial_alpha=alpha)
    add_running(sched, *reqs)
    first = sched.schedule()
    assert first.num_spec_tokens_to_schedule == 4
    assert sched._plan.per_request_k == {"two": 2, "four": 4, "zero": 0}
    assert {rid: decision.k for rid, decision in sched._plan.per_request_decisions.items()} == {
        "two": 2, "four": 4, "zero": 0}
    assert [len(r.spec_token_ids) for r in reqs] == [2, 4, 0]
    second = sched.schedule()
    sched.structured_output_manager = SimpleNamespace(validate_tokens=lambda request, tokens: tokens)
    drafts = SimpleNamespace(req_ids=[r.request_id for r in reqs], draft_token_ids=[[7] * 4 for r in reqs])
    # Run the captured upstream method on the next output's placeholders;
    # do not recreate its trim/pad implementation inside the stub.
    call_pinned_method("update_draft_token_ids_in_output", sched, drafts, second)
    assert {r: len(tokens) for r, tokens in second.scheduled_spec_decode_tokens.items()} == {"two": 2, "four": 4}
    adapter = sched._planner.adapter
    four = adapter.get_controller("four")
    before = (four.alpha_ema, four.get_window_samples(), four.get_stats())
    sched.make_spec_decoding_stats(None, 2, 1, second.num_invalid_spec_tokens, "two")
    assert adapter.get_controller("two").alpha_ema == pytest.approx(0.66)
    assert (four.alpha_ema, four.get_window_samples(), four.get_stats()) == before
    sched.make_spec_decoding_stats(None, 4, 4, second.num_invalid_spec_tokens, "four")
    assert adapter._accepted_history == {"two": [1], "four": [4], "zero": []}

    # K=0 still reaches completion without any draft feedback. Finish it while
    # the two speculative requests remain active, then exercise another step.
    sched._free_request(reqs[2])
    assert set(sched.requests) == {"two", "four"}
    assert set(adapter._active_sessions) == set(adapter._accepted_history) == {"two", "four"}
    assert set(sched._plan.per_request_k) == set(sched._plan.per_request_decisions) == {"two", "four"}
    assert set(sched._plan.per_request_node_count) == {"two", "four"}
    assert sched._planner._known_requests == {"two", "four"}
    assert set(sched._planner._last_decisions) == set(sched._planner._scheduled_runtime_modes) == {"two", "four"}
    assert adapter.get_controller("four") is four
    third = sched.schedule()
    assert third.num_spec_tokens_to_schedule == 4
    assert sched._plan.per_request_k == {"two": 0, "four": 4}
    for req in reqs[:2]:
        sched._free_request(req)
    assert sched.requests == {} and sched.running == []
    assert adapter._active_sessions == adapter._accepted_history == {}
    assert sched._planner._known_requests == set()
    assert sched._planner._last_decisions == sched._planner._scheduled_runtime_modes == {}
    assert sched._plan.per_request_k == sched._plan.per_request_node_count == sched._plan.per_request_decisions == {}

    records = [json.loads(line) for line in path.read_text().splitlines()]
    aggregates = {r["request_id"]: r for r in
                  map(json.loads, (tmp_path / "lane3.jsonl").read_text().splitlines())}
    assert set(aggregates) == {"two", "four", "zero"}
    assert aggregates["zero"]["step_count"] == 0
    assert aggregates["zero"]["tree_budgets"] == [0, 0]
    assert aggregates["two"]["tree_budgets"] == [2, 2, 0]
    assert aggregates["four"]["tree_budgets"] == [4, 4, 4]
    assert aggregates["two"]["mean_accept_len"] == 1
    assert aggregates["four"]["mean_accept_len"] == 4
    plans = {(r["request_id"], r["plan_sequence"]): r for r in records
             if r["record_type"] == "planned_decision"}
    for record in records:
        rid = record["request_id"]
        if record["record_type"] == "scheduled_budget":
            plan = plans[rid, record["plan_sequence"]]
            assert record["scheduled_scalar_k"] == plan["batch_scalar_budget"] == 4
            assert record["next_request_placeholder_budget"] == plan["scalar_runtime_budget"]
            assert record["schedule_sequence"] == record["plan_sequence"]
            assert record["executed_draft_token_count"] is None
        elif record["record_type"] == "verification_feedback":
            assert (record["executed_draft_token_count"], record["accepted_draft_token_count"]) == {
                "two": (2, 1), "four": (4, 4)}[rid]
            assert record["alpha_ema"] == pytest.approx({"two": 0.66, "four": 0.96}[rid])
            assert record["source_plan_sequence"] is record["source_schedule_sequence"] is None
        assert record["recorded_at"] <= aggregates[rid]["finished_at"]
    for rid, raw, effective in (("two", 0.66, 0.5), ("four", 0.96, 1.0)):
        last = aggregates[rid]["last_planned_decision"]
        assert last["acceptance_rate_ema"] == pytest.approx(raw)
        assert last["effective_alpha"] == pytest.approx(effective)
        assert aggregates[rid]["executed_draft_token_count"] is None
    for record in [*records, *aggregates.values()]:
        assert record["tree_execution_verified"] is False
        assert record["executed_runtime_mode"] == "unverified"
        assert record.get("selected_topology") is None


def test_feedback_drives_scheduler_k_transition(scheduler_module, monkeypatch, tmp_path):
    """L2: real scalar feedback changes future K, never the current draft count."""
    sched = build(scheduler_module, monkeypatch, tmp_path)
    req = make_request("r")
    req.is_finished = lambda: False
    # Three outstanding placeholders can precede a new K=2 decision in async
    # serving. Each next output verifies the preceding output's draft budget.
    req.spec_token_ids = [-1] * 3
    sched._planner.adapter.start_request("r", initial_alpha=0.7)
    add_running(sched, req)
    sched.structured_output_manager = SimpleNamespace(validate_tokens=lambda request, tokens: tokens)
    controller = sched._planner.adapter.get_controller("r")
    for current_drafts, accepted, future_k, effective in (
        (3, 2, 2, 2 / 3),
        (2, 2, 2, 23 / 27),
        (2, 2, 3, 167 / 183),
    ):
        out = sched.schedule()
        assert out.num_spec_tokens_to_schedule == future_k
        assert sched._plan.per_request_k == {"r": future_k}
        assert len(req.spec_token_ids) == future_k
        drafts = SimpleNamespace(req_ids=["r"], draft_token_ids=[[7] * current_drafts])
        call_pinned_method("update_draft_token_ids_in_output", sched, drafts, out)
        assert len(out.scheduled_spec_decode_tokens["r"]) == current_drafts
        sched.make_spec_decoding_stats(None, current_drafts, accepted, out.num_invalid_spec_tokens, "r")
        assert controller.get_effective_alpha() == pytest.approx(effective)
    final = sched.schedule()
    assert final.num_spec_tokens_to_schedule == len(req.spec_token_ids) == 4
    assert controller.get_stats()["tree_budgets"] == [2, 2, 3, 4]
    assert controller.get_window_samples() == [2 / 3, 1.0, 1.0]


@pytest.mark.parametrize("signal", ["kv", "vram"])
def test_runtime_memory_boundary_fallback_and_recovery(scheduler_module, monkeypatch, tmp_path, signal):
    """L2 signal producer -> planner units -> guard -> output/placeholders."""
    fake_torch, calls = _fake_torch(initialized=True, free_bytes=1500 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_ENABLE_VRAM_GUARD="1")
    req = make_request("r")
    add_running(sched, req)
    sched.kv_usage = 0.90
    assert sched.schedule().num_spec_tokens_to_schedule == 3  # equality is safe
    if signal == "kv":
        sched.kv_usage = 0.9001
    else:
        fake_torch.cuda.mem_get_info = lambda: (1499 * 1024 * 1024, 16000 * 1024 * 1024)
    assert sched.schedule().num_spec_tokens_to_schedule == 0
    assert req.spec_token_ids == []
    assert signal.upper() in sched._plan.per_request_decisions["r"].reason.upper()
    fake_torch.cuda.mem_get_info = lambda: (1500 * 1024 * 1024, 16000 * 1024 * 1024)
    sched.kv_usage = 0.80
    # Trigger counts as cooldown step one; equality at recovery remains held.
    assert [sched.schedule().num_spec_tokens_to_schedule for _ in range(3)] == [0, 0, 0]
    sched.kv_usage = 0.7999
    assert sched.schedule().num_spec_tokens_to_schedule == len(req.spec_token_ids) == 3
    assert calls


def test_runtime_cap_applies_to_scheduler_output(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    sched._planner.max_k = 2
    add_running(sched, make_request("r"))
    out = sched.schedule()
    assert sched._plan.per_request_node_count == {"r": 3}
    assert out.num_spec_tokens_to_schedule == len(sched.requests["r"].spec_token_ids) == 2


def test_aggregate_io_failure_still_cleans_only_finished_request(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    a, b = make_request("a"), make_request("b")
    add_running(sched, a, b)
    sched.schedule()
    sched._planner.telemetry_path = str(tmp_path)  # directory: aggregate open fails
    assert sched._free_request(a, delay_free_blocks=True) == (None, None)
    assert sched.freed == ["a"]
    assert "a" in sched.requests  # forwarded delay_free_blocks preserves base behavior
    assert sched._planner._known_requests == {"b"}
    assert set(sched._planner._last_decisions) == {"b"}
    assert set(sched._planner._scheduled_runtime_modes) == {"b"}
    assert sched._planner.adapter.active_request_count == 1
    assert set(sched._plan.per_request_decisions) == {"b"}
