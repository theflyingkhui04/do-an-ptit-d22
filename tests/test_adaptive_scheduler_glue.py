"""Glue tests for AdaptiveAsyncScheduler against a stubbed vLLM AsyncScheduler.

L2 only: the stub mirrors the v0.30.0 source contract, not an installed
engine smoke or GPU verification. See the M4-02-04 worklog for source links.
"""
import importlib
import os
import ast
import __future__
import json
import sys
import types
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path
from contextlib import nullcontext

import pytest
from src.server.lineage import _engine_startup


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
        # Simulated inline UniProc boundaries, not GPU execution. Separate
        # executor tests exercise the production observer on pinned methods.
        runner = getattr(self, "_test_runner", None)
        if runner is not None:
            previous = self._lineage.generation
            old_rows = tuple(runner.input_batch.req_ids)
            runner.input_batch.req_ids = list(scheduled)
            runner.prev_positions.np = [old_rows.index(rid) if rid in old_rows else -1
                                        for rid in scheduled]
            self._lineage.prepared(out, runner, previous)
            worker_k = out.num_spec_tokens_to_schedule
            runner._draft_token_ids = SimpleNamespace(shape=(len(scheduled), worker_k))
            runner.prev_num_spec_tokens = worker_k
            self._lineage.produced(out, runner)
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        for rid, drafted, accepted in model_runner_output.feedback:
            self.make_spec_decoding_stats(None, drafted, accepted,
                                         model_runner_output.invalid, rid)
        return "upstream-result"

    def shutdown(self):
        pass

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
    vllm.__version__ = "0.30.0"
    vllm.__file__ = str(installed_capture() / "__init__.py")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    async_mod = types.ModuleType("vllm.v1.core.sched.async_scheduler")
    async_mod.AsyncScheduler = FakeAsyncScheduler
    core_mod = types.ModuleType("vllm.v1.engine.core")
    core_mod.EngineCore = cpu_engine_core()
    plugins = types.ModuleType("vllm.plugins")
    plugins.load_general_plugins = lambda: None
    for name, mod in {
        "vllm": vllm,
        "vllm.logger": logger_mod,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.async_scheduler": async_mod,
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.core": core_mod,
        "vllm.plugins": plugins,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "src.server.adaptive_scheduler", raising=False)
    module = importlib.import_module("src.server.adaptive_scheduler")
    module._cpu_executor_owners = []
    module._cpu_engine_core = core_mod.EngineCore
    try:
        yield module
    finally:
        from src.server.lineage import release_ledger
        for owner in module._cpu_executor_owners:
            release_ledger(owner._lineage_registration)


def paired_cpu_config(module, config=None):
    """Explicit CPU observer stand-in; use the production registration contract.

    No runtime bypass is enabled. Tests simulate worker observations separately;
    test_executor_observation_and_terminal_cleanup uses the executor wrapper.
    """
    from src.server.lineage import register_lineage_executor
    class CpuExecutor:
        pass
    if config is None:
        config = SimpleNamespace(parallel_config=SimpleNamespace(
            distributed_executor_backend="tests.CpuLineageExecutor"))
    owner = CpuExecutor()
    owner.vllm_config = config
    owner._lineage_registration = register_lineage_executor(config, owner)
    owner._lineage = owner._lineage_registration.ledger
    module._cpu_executor_owners.append(owner)
    return config


def cpu_config():
    return SimpleNamespace(parallel_config=SimpleNamespace(
        distributed_executor_backend="tests.CpuLineageExecutor"))


def paired_cpu_scheduler(module, config=None, **kwargs):
    config = config if config is not None else cpu_config()
    with _engine_startup(config):
        paired_cpu_config(module, config)
        return module.AdaptiveAsyncScheduler(vllm_config=config, **kwargs)


def make_request(req_id, prefill=False):
    return SimpleNamespace(request_id=req_id, is_prefill_chunk=prefill, spec_token_ids=[])


def build(scheduler_module, monkeypatch, tmp_path, *, vllm_config=None, **env):
    monkeypatch.setenv("ADAPTIVE_CONTROLLER_KWARGS", json.dumps({"initial_alpha": 0.85}))
    monkeypatch.setenv("ADAPTIVE_TELEMETRY_PATH", str(tmp_path / "lane3.jsonl"))
    monkeypatch.setenv("ADAPTIVE_ENABLE_VRAM_GUARD", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = vllm_config if vllm_config is not None else cpu_config()
    with (_engine_startup(config) if vllm_config is None else nullcontext()):
        if vllm_config is None:
            paired_cpu_config(scheduler_module, config)
        sched = scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    sched._test_runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[], prev_sampled_token_ids=object()),
        prev_positions=SimpleNamespace(np=[]), prev_num_spec_tokens=0,
        _draft_token_ids=None)
    return sched


def add_running(scheduler, *reqs):
    for req in reqs:
        scheduler.running.append(req)
        scheduler.requests[req.request_id] = req


def run_step(scheduler):
    return scheduler.schedule()


def retire(scheduler, output, feedback, invalid=None):
    ids = list(output.num_scheduled_tokens)
    result = SimpleNamespace(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                             feedback=feedback, invalid=invalid)
    return scheduler.update_from_output(output, result)


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
    out = run_step(sched)
    retire(sched, out, [("bad", 3, 0)])

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

    out = run_step(sched)
    retire(sched, out, [("r1", 3, 2)], {"r1": 1})

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
    sched2 = paired_cpu_scheduler(scheduler_module, num_spec_tokens=3)
    assert sched2._planner.max_k == 3


def test_per_request_disabled_keeps_batch_placeholders(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_PER_REQUEST="0")
    good, bad = make_request("good"), make_request("bad")
    add_running(sched, good, bad)
    run_step(sched)
    out = run_step(sched)
    retire(sched, out, [("bad", 3, 0)])

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
    sched = paired_cpu_scheduler(scheduler_module, dynamic=False)
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
    out = sched.schedule()  # verifies first output's K=3, proposes newer K=3
    sched.schedule()  # third plan exists before first verification arrives
    retire(sched, out, [("r", 3, 1)])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    feedback = records[-1]
    assert feedback["record_type"] == "verification_feedback"
    assert feedback["executed_draft_token_count"] == 3
    assert feedback["accepted_draft_token_count"] == 1
    assert "selected_topology" not in feedback
    assert feedback["source_plan_sequence"] == 1
    assert feedback["source_schedule_sequence"] == 1
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
    retire(sched, second, [("two", 2, 1), ("four", 4, 4)], second.num_invalid_spec_tokens)
    assert adapter.get_controller("two").alpha_ema == pytest.approx(0.66)
    assert four.step_count == 1
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
            assert record["source_plan_sequence"] == record["source_schedule_sequence"] == 1
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
    sched._planner.adapter.start_request("r", initial_alpha=0.85)
    add_running(sched, req)
    # Establish an actual source K=3, then change the controller before verify.
    source = sched.schedule()
    assert source.num_spec_tokens_to_schedule == 3
    sched._planner.adapter.start_request("r", initial_alpha=0.7)
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
        retire(sched, out, [("r", current_drafts, accepted)], out.num_invalid_spec_tokens)
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


@pytest.mark.parametrize("reverse", [False, True], ids=["FIFO", "identity-stress-only"])
def test_multiple_inflight_changing_k_exact_producer(scheduler_module, monkeypatch, tmp_path, reverse):
    """T1-T4/T6: reverse is a ledger stress, NOT a pinned FIFO runtime claim."""
    path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    req = make_request("r")
    add_running(sched, req)
    sched._planner.adapter.start_request("r", initial_alpha=0.7)
    a = sched.schedule()
    assert a.num_spec_tokens_to_schedule == 2
    sched._planner.adapter.get_controller("r").initial_alpha = 0.95
    sched._planner.adapter.get_controller("r").alpha_ema = 0.95
    b = sched.schedule()
    c = sched.schedule()
    assert b.num_spec_tokens_to_schedule == c.num_spec_tokens_to_schedule == 4
    assert len(b.scheduled_spec_decode_tokens["r"]) == 2
    assert len(c.scheduled_spec_decode_tokens["r"]) == 4
    pairs = [(b, 2), (c, 4)]
    for output, count in reversed(pairs) if reverse else pairs:
        assert retire(sched, output, [("r", count, count - 1)]) == "upstream-result"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    feedback = [r for r in records if r["record_type"] == "verification_feedback"]
    assert [(r["source_plan_sequence"], r["source_schedule_sequence"],
             r["source_request_scalar_k"], r["executed_draft_token_count"]) for r in feedback] == (
                 [(2, 2, 4, 4), (1, 1, 2, 2)] if reverse else [(1, 1, 2, 2), (2, 2, 4, 4)])
    assert sched._planner.adapter.get_controller("r").step_count == 2
    retire(sched, a, [])  # no-feedback execution still terminates
    assert sched._lineage.executions == {}


def test_interleaved_requests_and_reordered_rows(scheduler_module, monkeypatch, tmp_path):
    """T5: current row zero can consume old row one; callback order is irrelevant."""
    path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    a, b = make_request("a"), make_request("b")
    add_running(sched, a, b)
    sched._planner.adapter.start_request("a", initial_alpha=0.7)
    sched._planner.adapter.start_request("b", initial_alpha=0.95)
    first = sched.schedule()
    sched.running.reverse()
    second = sched.schedule()
    third = sched.schedule()
    retire(sched, second, [("b", 4, 4), ("a", 2, 0)])
    retire(sched, third, [("a", 2, 1), ("b", 4, 3)])
    events = [r for r in map(json.loads, path.read_text().splitlines())
              if r["record_type"] == "verification_feedback"]
    assert [(r["request_id"], r["source_schedule_sequence"], r["source_request_scalar_k"])
            for r in events] == [("b", 1, 4), ("a", 1, 2), ("a", 2, 2), ("b", 2, 4)]
    assert sched._planner.adapter._accepted_history == {"a": [0, 1], "b": [4, 3]}
    retire(sched, first, [])
    sched._free_request(a)
    assert sched._lineage.active.keys() == {"b"}
    sched._free_request(b)
    assert sched._lineage.executions == sched._lineage.active == {}


def test_zero_k_retirement_has_no_feedback_or_orphan(scheduler_module, monkeypatch, tmp_path):
    """T7: zero K can verify OLD positive K, but creates no new draft birth."""
    sched = build(scheduler_module, monkeypatch, tmp_path)
    req = make_request("r")
    add_running(sched, req)
    a = sched.schedule()
    sched.kv_usage = 0.95
    b = sched.schedule()
    assert b.num_spec_tokens_to_schedule == 0
    assert sched._lineage.generation is None
    retire(sched, b, [("r", 3, 2)])
    for _ in range(30):
        out = sched.schedule()
        assert not out.scheduled_spec_decode_tokens
        retire(sched, out, [])
    retire(sched, a, [])
    assert sched._lineage.executions == {}
    assert sched._planner.adapter.get_controller("r").step_count == 1
    sched._free_request(req)
    assert sched._lineage.active == {}


def test_invalid_suffix_preserves_origin_and_accounting(scheduler_module, monkeypatch, tmp_path):
    """T8: real captured validation pads an invalid suffix, D=N-I, A unchanged."""
    path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    req = make_request("r")
    req.is_finished = lambda: False
    add_running(sched, req)
    sched.schedule()
    out = sched.schedule()
    sched.structured_output_manager = SimpleNamespace(validate_tokens=lambda request, tokens: tokens[:2])
    call_pinned_method("update_draft_token_ids_in_output", sched,
                       SimpleNamespace(req_ids=["r"], draft_token_ids=[[10, 11, 12, 13]]), out)
    assert out.scheduled_spec_decode_tokens == {"r": [10, 11, -1]}
    assert out.num_invalid_spec_tokens == {"r": 1}
    retire(sched, out, [("r", 3, 2)], out.num_invalid_spec_tokens)
    feedback = json.loads(path.read_text().splitlines()[-1])
    assert (feedback["executed_draft_token_count"], feedback["accepted_draft_token_count"]) == (2, 2)
    assert feedback["source_schedule_sequence"] == 1
    assert sched.stats_calls == [("r", 3, 2)]


def test_finished_reused_request_cannot_receive_late_feedback(scheduler_module, monkeypatch, tmp_path, caplog):
    """T9: removing one request keeps the other source and blocks resurrection."""
    sched = build(scheduler_module, monkeypatch, tmp_path)
    a, b = make_request("a"), make_request("b")
    add_running(sched, a, b)
    sched.schedule()
    out = sched.schedule()
    sched._free_request(a)
    fresh = make_request("a")
    add_running(sched, fresh)
    new_controller = sched._planner.adapter.start_request("a", initial_alpha=0.7)
    retire(sched, out, [("a", 3, 3), ("b", 3, 2)])
    assert new_controller.step_count == 0
    assert sched._planner.adapter.get_controller("b").step_count == 1
    assert "missing-lineage" in caplog.text
    sched._free_request(fresh)
    sched._free_request(b)
    sched.shutdown()
    assert sched._lineage.active == sched._lineage.executions == {}
    assert sched._lineage.generation is None


@pytest.mark.parametrize("damage", ["missing", "count", "incarnation", "request", "stale", "accepted"])
def test_invalid_lineage_fails_closed(scheduler_module, monkeypatch, tmp_path, caplog, damage):
    """T10: upstream statistics survive; adaptive state does not change."""
    from dataclasses import replace
    path = tmp_path / "steps.jsonl"
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    req = make_request("r")
    add_running(sched, req)
    sched.schedule()
    out = sched.schedule()
    entry = sched._lineage.get(out)
    origin, count = entry.consumed["r"]
    accepted = 1
    if damage == "missing":
        entry.consumed.clear()
    elif damage == "count":
        entry.consumed["r"] = (origin, count + 1)
    elif damage in {"incarnation", "request"}:
        entry.consumed["r"] = (replace(origin, **({"incarnation": -1} if damage == "incarnation"
                                                   else {"request_id": "other"})), count)
    elif damage == "stale":
        req.num_stale_output_tokens = 1
    else:
        accepted = 4
    controller = sched._planner.adapter.get_controller("r")
    before = (controller.alpha_ema, controller.get_window_samples(), controller.step_count)
    retire(sched, out, [("r", 3, accepted)])
    assert (controller.alpha_ema, controller.get_window_samples(), controller.step_count) == before
    assert sched.stats_calls == [("r", 3, accepted)]
    assert "feedback skipped" in caplog.text
    assert json.loads(path.read_text().splitlines()[-1])["record_type"] == "verification_feedback_skipped"
    assert sched._lineage.get(out) is None


def test_duplicate_feedback_and_retirement_exception_cleanup(scheduler_module, monkeypatch, tmp_path):
    sched = build(scheduler_module, monkeypatch, tmp_path)
    add_running(sched, make_request("r"))
    sched.schedule()
    out = sched.schedule()
    retire(sched, out, [("r", 3, 2), ("r", 3, 2)])
    retire(sched, out, [("r", 3, 2)])
    assert sched._planner.adapter.get_controller("r").step_count == 1
    assert len(sched.stats_calls) == 3
    out = sched.schedule()
    def fail(*args):
        raise RuntimeError("upstream failure")
    monkeypatch.setattr(FakeAsyncScheduler, "update_from_output", fail)
    with pytest.raises(RuntimeError, match="upstream failure"):
        retire(sched, out, [])
    assert sched._lineage.get(out) is None and sched._retiring is None


def installed_capture():
    external = os.environ.get('M4_VLLM_SOURCE_ROOT')
    if external:
        from experiments.m4_pinned_contract import verify
        root = Path(external)
        verify(root)
        return root / 'vllm'
    path = PINNED_SCHEDULER.parents[6] / "m4_02_04/20261110-async-lineage-design/installed/vllm"
    if not path.is_dir():
        pytest.skip("optional pinned installed-source cache unavailable; not a runtime test")
    return path


def cpu_engine_core():
    """Unchanged pinned constructor prefix through scheduler creation, CPU deps.

    Deliberately not a full runtime EngineCore: post-scheduler runtime/GPU setup
    is excluded. The production wrapper surrounds this same constructor entry.
    """
    source = installed_capture() / "v1/engine/core.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.AnnAssign)
               and ast.unparse(n.target) == "self.scheduler")
    fn.body = fn.body[:end + 1]
    namespace = {"envs": SimpleNamespace(VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=False),
                 "StructuredOutputManager": lambda c: SimpleNamespace(),
                 "resolve_kv_cache_block_sizes": lambda *a: (16, 16)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    class EngineCore:
        __init__ = namespace["__init__"]

        def _initialize_kv_caches(self, config):
            if getattr(config, "fail_kv", False):
                raise RuntimeError("KV startup failed")
            return SimpleNamespace(kv_cache_groups=[object()])
    return EngineCore


def captured_function(path, cls, method, namespace):
    module = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == method)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[method]


def test_actual_pinned_row_copy_binds_reorder_and_compaction(scheduler_module, monkeypatch, tmp_path):
    """Run unchanged pinned row preparation on CPU tensors, then observe it."""
    import numpy as np
    torch = pytest.importorskip("torch", reason="row-copy tensor test requires CPU torch")
    capture = installed_capture()
    source = capture / "v1/worker/gpu_model_runner.py"
    namespace = {"np": np, "torch": torch, "PIN_MEMORY": False}
    positions = captured_function(source, "GPUModelRunner", "_compute_prev_positions", namespace)
    prepare = captured_function(source, "GPUModelRunner", "_prepare_input_ids", namespace)
    sched = build(scheduler_module, monkeypatch, tmp_path)
    a, b, c = [make_request(rid) for rid in ("a", "b", "c")]
    add_running(sched, a, b, c)
    first = sched.schedule()
    buffer = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=torch.int32)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a", "b", "c"],
            prev_sampled_token_ids=torch.tensor([[10], [20], [30]], dtype=torch.int32),
            prev_req_id_to_index={"a": 0, "b": 1, "c": 2}),
        _draft_token_ids=buffer, prev_num_spec_tokens=3,
        prev_positions=SimpleNamespace(np=np.zeros(3, dtype=int)),
        _pp_recv_work=None, enable_prompt_embeds=False, device="cpu",
        input_ids=SimpleNamespace(gpu=torch.zeros(6, dtype=torch.int32), copy_to_gpu=lambda n: None))
    sched._lineage.produced(first, runner)
    generation = sched._lineage.generation
    # Remove a and compact/reorder c,b. Prefix of length two per request.
    sched.running = [c, b]
    c.spec_token_ids = b.spec_token_ids = [-1, -1]
    second = sched.schedule()
    runner.input_batch.req_ids = ["c", "b"]
    positions(runner, 2)
    assert list(runner.prev_positions.np[:2]) == [2, 1]
    prepare(runner, second, 2, 6, np.array([3, 6]))
    assert runner.input_ids.gpu.tolist() == [30, 31, 32, 20, 21, 22]
    sched._lineage.prepared(second, runner, generation)
    assert {rid: (origin.request_id, origin.schedule_sequence, count)
            for rid, (origin, count) in sched._lineage.get(second).consumed.items()} == {
                "c": ("c", 1, 2), "b": ("b", 1, 2)}


def test_executor_observation_and_terminal_cleanup(scheduler_module, monkeypatch, tmp_path):
    """Production executor wrapper, simulated worker; no installed engine."""
    from concurrent.futures import Future
    from src.server.lineage import release_ledger, require_lineage_executor
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=1, data_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(method="draft_model", disable_padded_drafter_batch=False))
    runner_type = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu_model_runner"})
    runner = runner_type()
    runner.use_async_scheduling = True
    runner.execute_model_state = None
    runner.input_batch = SimpleNamespace(req_ids=["r"], prev_sampled_token_ids=object())
    runner.prev_positions = SimpleNamespace(np=[0])
    runner.prev_num_spec_tokens = 3
    runner._draft_token_ids = SimpleNamespace(shape=(1, 3))
    runner._copy_draft_token_ids_to_cpu = lambda output, zeros_only=False: None
    class Base:
        def _init_executor(self):
            self.driver_worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
        def execute_model(self, output, non_block=False):
            runner.execute_model_state = (output,)
            future = Future()
            future.set_result(None)
            return future
        def sample_tokens(self, grammar, non_block=False):
            output = runner.execute_model_state[0]
            runner.execute_model_state = None
            runner._copy_draft_token_ids_to_cpu(output, zeros_only=grammar == "synthetic")
            return Future()  # caller materializes the result later
        def shutdown(self):
            pass
    mod = types.ModuleType("vllm.v1.executor.uniproc_executor")
    mod.UniProcExecutor = Base
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.uniproc_executor", mod)
    monkeypatch.delitem(sys.modules, "src.server.lineage_executor", raising=False)
    executor_module = importlib.import_module("src.server.lineage_executor")
    capture = installed_capture()
    from experiments.modal_lanes import build_vllm_args
    args = build_vllm_args("adaptive")
    config.parallel_config.distributed_executor_backend = args[args.index("--distributed-executor-backend") + 1]
    get_class = captured_function(capture / "v1/executor/abstract.py", "Executor", "get_class", {
        "Executor": Base, "resolve_obj_by_qualname": lambda name: getattr(
            importlib.import_module(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1])})
    assert get_class(config) is executor_module.LineageUniProcExecutor
    monkeypatch.setattr(executor_module.vllm, "__version__", "0.30.0", raising=False)
    monkeypatch.setattr(executor_module.vllm, "__file__", str(capture / "__init__.py"), raising=False)
    executor_module.verify_sources()
    monkeypatch.setattr(executor_module.vllm, "__version__", "different")
    with pytest.raises(RuntimeError, match="0.30.0"):
        executor_module.verify_sources()
    monkeypatch.setattr(executor_module, "verify_sources", lambda: None)
    executor = executor_module.LineageUniProcExecutor()
    executor.vllm_config = config
    config.parallel_config.world_size = 2
    with pytest.raises(RuntimeError, match="single-worker"):
        executor._init_executor()
    with pytest.raises(RuntimeError, match="lineage-capable executor"):
        require_lineage_executor(config)  # failed executor init never publishes readiness
    config.parallel_config.world_size = 1
    with _engine_startup(config):
        executor._init_executor()
        sched = build(scheduler_module, monkeypatch, tmp_path, vllm_config=config)
    assert sched._lineage is executor._lineage
    # Disable the fake schedule's simulated dispatch: exercise the wrapper.
    sched._test_runner = None
    req = make_request("r")
    add_running(sched, req)
    first = sched.schedule()
    executor.execute_model(first, non_block=True)
    f1 = executor.sample_tokens(None, non_block=True)
    assert sched._lineage.generation[3][0].schedule_sequence == 1
    second = sched.schedule()
    executor.execute_model(second, non_block=True)
    # Grammar sampling can be deferred while an older result is retired.
    retire(sched, first, [])
    f2 = executor.sample_tokens(None, non_block=True)
    assert sched._lineage.get(second).consumed["r"][0].schedule_sequence == 1
    # This stand-in reuses the SAME buffer object; the birth stamp must advance.
    assert sched._lineage.generation[3][0].schedule_sequence == 2
    third = sched.schedule()
    executor.execute_model(third, non_block=True)
    executor.sample_tokens("synthetic", non_block=True)
    assert sched._lineage.generation is None
    retire(sched, second, [("r", 3, 2)])
    assert sched._planner.adapter.get_controller("r").step_count == 1
    f1.set_result(None)
    f2.set_exception(RuntimeError("async copy failure"))
    assert sched._lineage.executions == {} and sched._lineage.generation is None
    executor.shutdown()
    with pytest.raises(RuntimeError, match="lineage-capable executor"):
        require_lineage_executor(config)
    release_ledger(executor._lineage_registration)


def test_source_gate_against_captured_installed_bytes(scheduler_module, monkeypatch):
    # Use the production verifier, without importing an installed engine.
    capture = installed_capture()
    import hashlib
    manifest = json.loads((PINNED_SCHEDULER.parents[7] / "scalar_vllm_source_manifest.json").read_text())
    for record in manifest["files"]:
        assert hashlib.sha256((capture.parent / record["path"]).read_bytes()).hexdigest() == record["sha256"]


def test_pinned_retirement_calls_feedback_on_its_own_output():
    method = pinned_method("update_from_output")
    source = ast.unparse(method)
    assert "scheduler_output.scheduled_spec_decode_tokens.get(req_id)" in source
    assert "num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens" in source
    assert source.index("self.make_spec_decoding_stats") < source.index("self._free_request")


def test_pinned_uniproc_nonblocking_rpc_runs_inline():
    """Unchanged pinned RPC/future code proves the host observation window."""
    from concurrent.futures import Future
    from typing import Any
    source = installed_capture() / "v1/executor/uniproc_executor.py"
    events = []
    class AsyncResult:
        def get_output(self):
            events.append("materialized")
            return "paired-result"
    namespace = {"Future": Future, "Any": Any, "AsyncModelRunnerOutput": AsyncResult,
                 "run_method": lambda *args: events.append("worker-ran") or AsyncResult()}
    module = ast.parse(source.read_text())
    future_cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
                      and n.name == "AsyncOutputFuture")
    exec(compile(ast.Module(body=[future_cls], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    rpc = captured_function(source, "UniProcExecutor", "collective_rpc", namespace)
    future = rpc(SimpleNamespace(driver_worker=object()), "execute_model", non_block=True, single_value=True)
    assert events == ["worker-ran"] and not future.done()
    assert future.result() == "paired-result"
    assert events == ["worker-ran", "materialized"]


@pytest.mark.parametrize("positional", [False, True])
def test_paired_scheduler_startup_uses_registered_executor(scheduler_module, positional):
    """P1: same config/registered capability succeeds before any scheduling."""
    config = cpu_config()
    with _engine_startup(config):
        paired_cpu_config(scheduler_module, config)
        sched = (scheduler_module.AdaptiveAsyncScheduler(config) if positional else
                 scheduler_module.AdaptiveAsyncScheduler(vllm_config=config))
    assert sched._lineage is scheduler_module._cpu_executor_owners[-1]._lineage
    assert sched._lineage_registration is scheduler_module._cpu_executor_owners[-1]._lineage_registration
    assert sched._lineage.executions == {}
    assert sched._planner.adapter.active_request_count == 0


@pytest.mark.parametrize("backend", ["uni", "unrelated.Executor",
    "src.server.lineage_executor.LineageUniProcExecutor",
    type("LineageUniProcExecutor", (), {"LINEAGE_CAPABLE": True})])
def test_unpaired_scheduler_rejected_before_base_init_or_feedback(scheduler_module, monkeypatch, backend):
    """P2/P3/P7: even the right configured NAME cannot fake live readiness."""
    config = SimpleNamespace(parallel_config=SimpleNamespace(distributed_executor_backend=backend))
    def unexpected(*args, **kwargs):
        pytest.fail("Unpaired startup reached scheduler init or feedback")
    monkeypatch.setattr(FakeAsyncScheduler, "__init__", unexpected)
    monkeypatch.setattr(scheduler_module.AdaptiveAsyncScheduler, "make_spec_decoding_stats", unexpected)
    with pytest.raises(RuntimeError, match="Adaptive async lineage requires a lineage-capable executor") as exc:
        scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    assert str(backend) in str(exc.value)


def test_missing_config_is_not_a_scheduler_test_bypass(scheduler_module):
    with pytest.raises(RuntimeError, match="missing VllmConfig"):
        scheduler_module.AdaptiveAsyncScheduler()


def test_executor_registration_is_engine_local_and_revoked(scheduler_module):
    from src.server.lineage import require_lineage_executor, release_ledger
    import gc
    first = cpu_config()
    with _engine_startup(first):
        paired_cpu_config(scheduler_module, first)
        second = SimpleNamespace(parallel_config=first.parallel_config)
        with pytest.raises(RuntimeError, match="exact VllmConfig"):
            require_lineage_executor(second)
        release_ledger(scheduler_module._cpu_executor_owners[-1]._lineage_registration)
        with pytest.raises(RuntimeError, match="lineage-capable executor"):
            require_lineage_executor(first)
    dead = cpu_config()
    with _engine_startup(dead):
        paired_cpu_config(scheduler_module, dead)
        scheduler_module._cpu_executor_owners.pop()
        gc.collect()
        with pytest.raises(RuntimeError, match="lineage-capable executor"):
            require_lineage_executor(dead)


def test_non_adaptive_scheduler_does_not_require_lineage_registration():
    """P4: validation is confined to our adaptive subclass, not global vLLM."""
    normal = FakeAsyncScheduler(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(
        distributed_executor_backend="uni")))
    assert normal.requests == {} and normal.stats_calls == []


def test_pinned_engine_constructs_executor_before_scheduler_with_same_config():
    """Source contract for the early handshake; no live engine construction."""
    source = installed_capture() / "v1/engine/core.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    engine = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    init = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
    executor = next(n for n in calls if ast.unparse(n.func) == "executor_class")
    scheduler = next(n for n in calls if ast.unparse(n.func) == "Scheduler")
    assert executor.lineno < scheduler.lineno
    assert ast.unparse(executor.args[0]) == "vllm_config"
    assert ast.unparse(next(k.value for k in scheduler.keywords if k.arg == "vllm_config")) == "vllm_config"


# R1/R2 ownership regressions. All identities below are actual objects, not
# readiness flags or configured class-name comparisons.
def test_ownership_distinct_configs_have_distinct_generations(scheduler_module):
    a, b = [paired_cpu_scheduler(scheduler_module) for _ in range(2)]
    assert a._lineage is not b._lineage
    assert a._lineage_registration.token is not b._lineage_registration.token
    assert a._lineage_registration.attempt is not b._lineage_registration.attempt
    for sched, owner in zip((a, b), scheduler_module._cpu_executor_owners):
        assert sched._lineage is owner._lineage
        assert sched._lineage_registration is owner._lineage_registration


def test_ownership_same_config_plain_second_engine_rejected(scheduler_module, monkeypatch):
    """R1/E2: previously B inherited A's exact ledger with backend='uni'."""
    config = cpu_config()
    a = paired_cpu_scheduler(scheduler_module, config)
    ledger_a = a._lineage
    config.parallel_config.distributed_executor_backend = "uni"
    def unexpected(*args, **kwargs):
        pytest.fail("Second engine reached base initialization/feedback")
    monkeypatch.setattr(FakeAsyncScheduler, "__init__", unexpected)
    monkeypatch.setattr(scheduler_module.AdaptiveAsyncScheduler, "make_spec_decoding_stats", unexpected)
    with pytest.raises(RuntimeError, match="already consumed"):
        scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    assert a._lineage is ledger_a
    assert scheduler_module._cpu_executor_owners[0]._lineage is ledger_a


def test_ownership_single_consumer_even_without_backend_mutation(scheduler_module):
    config = cpu_config()
    a = paired_cpu_scheduler(scheduler_module, config)
    with pytest.raises(RuntimeError, match="already consumed"):
        scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    # Scheduler collection is deliberately not permission to reuse a generation.
    token = a._lineage_registration.token
    del a
    with pytest.raises(RuntimeError, match="already consumed"):
        scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    assert scheduler_module._cpu_executor_owners[0]._lineage_registration.token is token


def test_ownership_restart_stale_release_and_old_gc(scheduler_module):
    """R2/E3-E6: GA release, GB publish, stale GA release/GC cannot affect GB."""
    import gc
    from weakref import ref
    from src.server.lineage import release_ledger
    config = cpu_config()
    a = paired_cpu_scheduler(scheduler_module, config)
    owner_a = scheduler_module._cpu_executor_owners.pop()
    ga = owner_a._lineage_registration
    release_ledger(ga)
    with _engine_startup(config):
        paired_cpu_config(scheduler_module, config)
        owner_b = scheduler_module._cpu_executor_owners[-1]
        gb = owner_b._lineage_registration
        assert ga is not gb and ga.token is not gb.token and ga.ledger is not gb.ledger
        assert ga.attempt is not gb.attempt
        marker = object()
        gb.ledger.active["B"] = marker
        release_ledger(ga)
        release_ledger(ga)
        # Exercise callbacks before B consumes, as well as subsequent teardown.
        ga.owner.__callback__(ga.owner)
        old = ref(owner_a)
        del owner_a
        gc.collect()
        assert old() is None
        b = scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    assert b._lineage_registration is gb and b._lineage is gb.ledger
    assert b._lineage.active["B"] is marker
    a.shutdown()
    a.shutdown()
    assert b._lineage.active["B"] is marker
    b.shutdown()


def test_ownership_live_same_config_replacement_rejected(scheduler_module):
    from src.server.lineage import register_lineage_executor
    config = cpu_config()
    with _engine_startup(config):
        paired_cpu_config(scheduler_module, config)
        a = scheduler_module._cpu_executor_owners[-1]
        class Owner:
            vllm_config = config
        with _engine_startup(config):
            with pytest.raises(RuntimeError, match="already registered"):
                register_lineage_executor(config, Owner())
        # Nested failure restored A's attempt; its exact ledger remains valid.
        sched = scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
        assert sched._lineage is a._lineage


def test_ownership_config_cannot_authorize_release(scheduler_module):
    from src.server.lineage import release_ledger
    config = cpu_config()
    with _engine_startup(config):
        paired_cpu_config(scheduler_module, config)
        with pytest.raises(TypeError, match="registration handle"):
            release_ledger(config)
        sched = scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    assert sched._lineage is scheduler_module._cpu_executor_owners[-1]._lineage


def test_ownership_pending_generation_cannot_cross_threads(scheduler_module):
    from concurrent.futures import ThreadPoolExecutor
    from src.server.lineage import require_lineage_executor
    config = cpu_config()
    with _engine_startup(config):
        paired_cpu_config(scheduler_module, config)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(require_lineage_executor, config)
            with pytest.raises(RuntimeError, match="another initialization thread"):
                future.result()
        # Wrong-thread attempt did not consume or change the rightful generation.
        sched = scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
    assert sched._lineage is scheduler_module._cpu_executor_owners[-1]._lineage


def test_ownership_concurrent_distinct_engine_initializations():
    """E8: independent publication/consumption on simultaneous CPU threads."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from src.server.lineage import register_lineage_executor, require_lineage_executor, release_ledger
    barrier = Barrier(4)
    class Owner:
        pass
    def engine(index):
        config = SimpleNamespace()
        owner = Owner()
        owner.vllm_config = config
        with _engine_startup(config):
            handle = register_lineage_executor(config, owner)
            owner._lineage = handle.ledger
            handle.ledger.active[index] = owner
            barrier.wait(timeout=10)
            try:
                assert require_lineage_executor(config) is handle
                assert handle.ledger.active == {index: owner}
                return handle
            finally:
                release_ledger(handle)
    with ThreadPoolExecutor(max_workers=4) as pool:
        handles = list(pool.map(engine, range(4)))
    assert len({id(h.token) for h in handles}) == 4
    assert len({id(h.ledger) for h in handles}) == 4
    assert len({id(h.attempt) for h in handles}) == 4


def test_ownership_concurrent_same_config_publication_is_exclusive():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from src.server.lineage import register_lineage_executor, release_ledger
    barrier = Barrier(4)
    config = SimpleNamespace()
    class Owner:
        vllm_config = config
    owners = [Owner() for _ in range(4)]
    published = Barrier(4)
    def publish(owner):
        with _engine_startup(config):
            barrier.wait(timeout=10)
            try:
                handle = register_lineage_executor(config, owner)
                owner._lineage = handle.ledger
            except RuntimeError as error:
                assert "already registered" in str(error)
                handle = None
            # Keep the winning attempt active until all publications finish.
            published.wait(timeout=10)
            return handle
    with ThreadPoolExecutor(max_workers=4) as pool:
        handles = [h for h in pool.map(publish, owners) if h is not None]
    assert len(handles) == 1
    release_ledger(handles[0])


@pytest.mark.parametrize("executor_first", [False, True])
def test_ownership_production_shutdown_orders_and_stale_restart(
        scheduler_module, monkeypatch, executor_first):
    """Both real project shutdown methods; upstream worker/scheduler are CPU stubs."""
    class Base:
        def _init_executor(self):
            runner_type = type("GPUModelRunner", (), {
                "__module__": "vllm.v1.worker.gpu_model_runner"})
            runner = runner_type()
            runner.use_async_scheduling = True
            runner._copy_draft_token_ids_to_cpu = lambda *a, **kw: None
            self.driver_worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
        def shutdown(self):
            pass
    mod = types.ModuleType("vllm.v1.executor.uniproc_executor")
    mod.UniProcExecutor = Base
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.delitem(sys.modules, "src.server.lineage_executor", raising=False)
    executor_module = importlib.import_module("src.server.lineage_executor")
    monkeypatch.setattr(executor_module, "verify_sources", lambda: None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1, data_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(method="draft_model", disable_padded_drafter_batch=False))
    def start():
        with _engine_startup(config):
            executor = executor_module.LineageUniProcExecutor()
            executor.vllm_config = config
            executor._init_executor()
            sched = scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
        assert sched._lineage is executor._lineage
        assert sched._lineage_registration is executor._lineage_registration
        return executor, sched
    ea, sa = start()
    first, second = (ea, sa) if executor_first else (sa, ea)
    first.shutdown()
    eb, sb = start()
    assert ea._lineage is not eb._lineage
    assert sa._lineage_registration.token is not sb._lineage_registration.token
    marker = object()
    eb._lineage.active["B"] = marker
    second.shutdown()
    first.shutdown()
    second.shutdown()
    assert sb._lineage.active["B"] is marker
    sb.shutdown()
    eb.shutdown()
    sb.shutdown()
    eb.shutdown()


def test_ownership_pinned_single_scheduler_and_shutdown_contract():
    """Hash verified exact lifecycle, not an assumption of config uniqueness."""
    import hashlib
    capture = installed_capture()
    manifest = json.loads((capture.parent.parent / "installed_source_manifest.json").read_text())
    paths = ["vllm/v1/engine/core.py", "vllm/v1/executor/abstract.py",
             "vllm/v1/executor/uniproc_executor.py", "vllm/v1/core/sched/scheduler.py"]
    for path in paths:
        record = next(r for r in manifest["files"] if r["path"] == path)
        assert hashlib.sha256((capture.parent / path).read_bytes()).hexdigest() == record["sha256"]
    tree = ast.parse((capture / "v1/engine/core.py").read_text())
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    assignments = [n for n in ast.walk(engine) if isinstance(n, (ast.Assign, ast.AnnAssign))
                   and any(isinstance(t, ast.Attribute) and ast.unparse(t) == "self.scheduler"
                           for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    assert len(assignments) == 1
    shutdown = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
    source = ast.unparse(shutdown)
    assert source.index("self.model_executor.shutdown()") < source.index("self.scheduler.shutdown()")


def test_ownership_gc_clears_current_ledger_without_retaining_config():
    import gc
    from weakref import ref
    from src.server.lineage import register_lineage_executor
    class Config:
        pass
    class Owner:
        pass
    config, owner = Config(), Owner()
    owner.vllm_config = config
    with _engine_startup(config):
        handle = register_lineage_executor(config, owner)
        owner._lineage_registration = handle
        owner._lineage = handle.ledger
        ledger = handle.ledger
        ledger.active["request"] = object()
        config_ref, owner_ref = ref(config), ref(owner)
        del owner, handle
        gc.collect()
        assert owner_ref() is None
        assert ledger.active == {} and ledger.executions == {} and ledger.generation is None
    del config
    gc.collect()
    assert config_ref() is None
