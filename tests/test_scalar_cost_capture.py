"""L2 captured pinned-method tests. No installed engine/GPU timing evidence.

Uses hash-verified historical captures or the exact installed distribution;
never vendors vLLM. Missing sources skip and cannot support verification.
"""
import __future__
import ast
import collections
from concurrent.futures import Future
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast

import pytest

from experiments.scalar_cost_capture import RetirementObserver, UnsupportedMeasurementMode, trajectory_signature
from experiments.scalar_pinned_sources import read_source

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "experiments/runs/m4_02_04/20261110-async-lineage-design"
RUNTIME = dict(vllm="0.30.0", executor="uni", runner="V1", tp=1, pp=1,
               scheduler_base="AdaptiveAsyncScheduler", async_scheduling=True)


def source(path):
    try:
        return ast.parse(read_source(path))
    except FileNotFoundError as error:
        pytest.skip(str(error))


def method(path, name, **namespace):
    tree = source(path)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


CORE = "vllm/v1/engine/core.py"
SCHED = "vllm/v1/core/sched/scheduler.py"


def schedule(name, k=2, ids=("a", "b"), deferred=False, slots=None):
    slots = k if slots is None else slots
    return NS(name=name, num_spec_tokens_to_schedule=k,
              num_scheduled_tokens={rid: slots + 1 for rid in ids},
              scheduled_spec_decode_tokens={rid: [-1] * slots for rid in ids if slots},
              total_num_scheduled_tokens=len(ids) * (slots + 1),
              pending_structured_output_tokens=deferred)


def output(s, order=None):
    ids = list(order or s.num_scheduled_tokens)
    return NS(name="O" + s.name, req_ids=ids,
              req_id_to_index={rid: i for i, rid in enumerate(ids)},
              sampled_token_ids=[[10 + i] * (i + 1) for i, _ in enumerate(ids)])


def shape(s, order=None, context_kind="MEASURED"):
    ids = list(order or s.num_scheduled_tokens)
    return dict(request_ids=ids, contexts=[40] * len(ids),
                queries=[s.num_scheduled_tokens[rid] for rid in ids],
                valid_drafts=[len(s.scheduled_spec_decode_tokens.get(rid, ())) for rid in ids],
                context_kind=context_kind, prefill=False, proposer_rows=len(ids),
                proposer_k=s.num_spec_tokens_to_schedule)


class Harness:
    def __init__(self, schedules, observer, *, sampling=True, contexts="MEASURED"):
        self.plans, self.observer = list(schedules), observer
        self.actual_pairs = []
        self.batch_queue = collections.deque()
        self.batch_queue_size = 2
        self.is_ec_consumer, self.is_pooling_model = True, False
        self.check_for_draft_tokens = False
        self.futures, self.exec_futures = {}, {}
        self.outputs = {s.name: output(s, list(reversed(s.num_scheduled_tokens))) for s in schedules}
        self.scheduler = NS(has_requests=lambda: bool(self.plans), schedule=self.schedule,
                            get_grammar_bitmask=lambda s: s.name, update_from_output=self.update)
        self.model_executor = NS(execute_model=self.execute, sample_tokens=self.sample)
        self.sampling, self.contexts = sampling, contexts
        self.last = None
        self.step_fn = method(CORE, "step", record_function_or_nullcontext=lambda _: nullcontext())
        self.queue_fn = method(CORE, "step_with_batch_queue", cast=cast, Future=Future,
                               ModelRunnerOutput=object,
                               record_function_or_nullcontext=lambda _: nullcontext())

    def _should_throttle_prefills(self): return False
    def log_error_detail(self, *args): return nullcontext()
    def capture_iteration_details(self, *args): return nullcontext()
    def _process_aborts_queue(self): pass
    def _attach_iteration_details(self, *args): pass

    def schedule(self, *_): return self.plans.pop(0)

    def execute(self, s, **kwargs):
        self.last = s
        self.observer.submit(s, planned_k=4,
                             planned_per_request={rid: 4 for rid in s.num_scheduled_tokens},
                             placeholder_counts={rid: s.num_spec_tokens_to_schedule for rid in s.num_scheduled_tokens},
                             incarnations={rid: rid + ":1" for rid in s.num_scheduled_tokens})
        self.observer.worker_snapshot(s, shape(s, list(reversed(s.num_scheduled_tokens)), self.contexts))
        future = Future()
        if not getattr(self, "pending_exec_failure", False):
            future.set_result(None if self.sampling else self.outputs[s.name])
        self.exec_futures[s.name] = future
        return future

    def sample(self, grammar, non_block=False):
        if not non_block: return self.outputs[grammar]
        future = self.futures.setdefault(grammar, Future())
        return future

    def consume(self, s, o):
        self.actual_pairs.append((s.name, o.name))
        assert o is self.outputs[s.name]
        # Final scheduler output can be shorter than raw sampled rows. Shape
        # and output row ordering must not be zipped with scheduler dict order.
        return {0: NS(outputs=[NS(request_id=rid, new_token_ids=[99], finish_reason=None)
                               for rid in s.num_scheduled_tokens])}

    def update(self, s, o): return self.observer.update(self.consume, s, o, unsafe_requests=())
    def tick(self, queue=True):
        return self.observer.step(lambda: (self.queue_fn if queue else self.step_fn)(self))


def test_two_inflight_out_of_order_completion_preserves_fifo_pairing():
    obs = RetirementObserver(RUNTIME, clock=iter([100, 200]).__next__, intervals=1)
    h = Harness([schedule("1"), schedule("2")], obs)
    h.tick()  # S1 queued, no retirement
    h.futures["2"] = Future()
    h.futures["2"].set_result(h.outputs["2"])  # O2 ready before O1
    h.futures["1"].set_result(h.outputs["1"])
    h.tick()  # submit S2, pop S1
    h.tick()  # pop S2
    assert h.actual_pairs == [("1", "O1"), ("2", "O2")]
    assert [r["verifier_id"] for r in obs.events] == ["v1", "v2"]
    assert obs.events[0]["worker"]["request_ids"] == ("b", "a")
    assert obs.events[0]["output_rows"] == ("b", "a")
    assert obs.events[0]["emitted_by_request"] == {"a": 1, "b": 1}
    assert obs.windows[0]["emitted"] == 2  # raw sampled rows contained 3


def test_structured_output_defers_sampling_without_repairing_pair():
    obs = RetirementObserver(RUNTIME)
    h = Harness([schedule("1"), schedule("2", deferred=True)], obs)
    validations = []
    h.check_for_draft_tokens = True
    h.model_executor.take_draft_token_ids = lambda: NS(req_ids=["a", "b"])
    h.scheduler.update_draft_token_ids_in_output = lambda drafts, s: validations.append(s.name)
    h.tick()
    h.futures["1"].set_result(h.outputs["1"])
    h.tick()  # S2 waits; S1 consumed before sample_tokens(S2)
    assert h.actual_pairs == [("1", "O1")]
    h.futures["2"].set_result(h.outputs["2"])
    h.tick()
    assert h.actual_pairs == [("1", "O1"), ("2", "O2")]
    assert validations == ["2"]


@pytest.mark.parametrize("sampling", [True, False])
def test_sync_direct_and_sampled_output_paths(sampling):
    obs = RetirementObserver(dict(RUNTIME, async_scheduling=False))
    h = Harness([schedule("1")], obs, sampling=sampling)
    h.tick(queue=False)
    assert h.actual_pairs == [("1", "O1")]
    assert obs.events[0]["status"] == "eligible"


def test_failed_future_is_retained_and_not_retired():
    obs = RetirementObserver(RUNTIME)
    h = Harness([schedule("1")], obs)
    h.tick()
    h.futures["1"].set_exception(RuntimeError("synthetic OOM"))
    with pytest.raises(RuntimeError, match="synthetic OOM"):
        h.tick()
    assert h.actual_pairs == [] and obs.windows == []
    assert obs.events[-1]["status"] == "failed"
    assert obs.events[-1]["affected_inflight"] == ("v1",)


def test_empty_schedule_and_missing_shape_never_count():
    obs = RetirementObserver(RUNTIME)
    h = Harness([schedule("1", ids=())], obs, sampling=False)
    h.tick(queue=False)
    assert obs.events[-1]["reason"] == "EMPTY_SCHEDULE"
    assert obs.windows == []


def test_cpu_optimistic_context_is_explicitly_unavailable():
    obs = RetirementObserver(RUNTIME)
    h = Harness([schedule("1")], obs, contexts="CPU_OPTIMISTIC")
    h.tick(queue=False)
    assert obs.events[-1]["reason"] == "ACTUAL_CONTEXT_UNAVAILABLE"
    assert obs.windows == []
    runner = source("vllm/v1/worker/gpu_model_runner.py")
    prepare = next(n for n in ast.walk(runner) if isinstance(n, ast.FunctionDef)
                   and n.name == "_prepare_inputs")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "update_num_computed_tokens_for_batch_change" for n in ast.walk(prepare))
    utils = source("vllm/v1/spec_decode/utils.py")
    correction = next(n for n in ast.walk(utils) if isinstance(n, ast.FunctionDef)
                      and n.name == "update_num_computed_tokens_for_batch_change")
    assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "corrected"
                                               for t in n.targets)
               and ast.unparse(n.value) == "prev_computed + valid_counts.int()"
               for n in ast.walk(correction))


def test_current_verification_slots_and_future_k_are_separate():
    obs = RetirementObserver(RUNTIME)
    s = schedule("1", k=2, slots=4)
    h = Harness([s], obs)
    h.tick(queue=False)
    event = obs.events[-1]
    assert event["planned_k"] == 4
    assert event["scheduled_k"] == event["worker"]["proposer_k"] == 2
    assert event["placeholder_counts"] == {"a": 2, "b": 2}
    assert event["verification_slots"] == {"a": 4, "b": 4}


@pytest.mark.parametrize("mutation", ["reorder", "remove", "reuse", "bad_map"])
def test_identity_and_incarnation_rules(mutation):
    obs = RetirementObserver(RUNTIME, intervals=1)
    h = Harness([schedule("1"), schedule("2")], obs)
    h.tick(queue=False)
    s = h.plans[0]
    h.execute(s)
    entry = obs.pending[id(s)]
    if mutation == "remove":
        h.outputs["2"].req_ids = ["a"]
        h.outputs["2"].req_id_to_index = {"a": 0}
    elif mutation == "reuse": entry["incarnations"]["a"] = "a:2"
    elif mutation == "bad_map": h.outputs["2"].req_id_to_index = {"a": 0, "b": 1}
    h.update(s, h.outputs["2"])
    if mutation == "reorder":
        assert obs.events[-1]["status"] == "eligible" and len(obs.windows) == 1
    else:
        assert obs.events[-1]["status"] == "unsupported" and obs.windows == []


@pytest.mark.parametrize("unsafe", [("a",), ("aborted",), ("stale",)])
def test_lifecycle_unsafe_output_is_not_eligible(unsafe):
    obs = RetirementObserver(RUNTIME)
    s = schedule("1")
    h = Harness([s], obs)
    h.execute(s)
    obs.update(h.consume, s, h.outputs["1"], unsafe_requests=unsafe)
    assert obs.events[-1]["reason"] == "LIFECYCLE_UNSAFE"


def test_window_arithmetic_counts_n_following_retirements_not_anchor():
    obs = RetirementObserver(RUNTIME, clock=iter([1_000_000, 4_000_000, 9_000_000]).__next__, intervals=2)
    h = Harness([schedule(str(i)) for i in range(3)], obs)
    for _ in range(3): h.tick(queue=False)
    assert obs.windows == [dict(start_ns=1_000_000, end_ns=9_000_000,
                               retirement_ids=["v2", "v3"], ms_per_batch=4, emitted=4)]


@pytest.mark.parametrize("field,value", [("vllm", "main"), ("executor", "mp"),
                                         ("tp", 2), ("runner", "V2"), ("async_scheduling", None)])
def test_capability_guards(field, value):
    with pytest.raises(UnsupportedMeasurementMode, match="UNSUPPORTED_MEASUREMENT_MODE"):
        RetirementObserver(dict(RUNTIME, **{field: value}))


def test_observer_never_reads_a_device_object():
    class Device:
        def item(self): pytest.fail("device synchronization")
        def __iter__(self): pytest.fail("device iteration")
    obs = RetirementObserver(RUNTIME)
    s = schedule("1")
    obs.submit(s)
    with pytest.raises(UnsupportedMeasurementMode): obs.worker_snapshot(s, {"contexts": Device()})


def test_primary_observer_has_no_sync_io_or_future_materialization_calls():
    tree = ast.parse((ROOT / "experiments/scalar_cost_capture.py").read_text())
    prohibited = {"synchronize", "item", "cpu", "numpy", "tolist", "result", "open", "write", "dumps", "Lock"}
    assert not any(isinstance(n, ast.Call) and
                   (getattr(n.func, "attr", None) in prohibited or getattr(n.func, "id", None) in prohibited)
                   for n in ast.walk(tree))


@pytest.mark.parametrize("tokens,cap,expected", [([1, 99, 2], 10, [1, 99]),
                                                ([1, 77, 2], 10, [1, 77]),
                                                ([1, 2, 3], 2, [1, 2]),
                                                ([1, 2, 3], 10, [1, 2, 3])])
def test_pinned_scheduler_clips_before_engine_output(tokens, cap, expected):
    status = NS(FINISHED_STOPPED="stop", FINISHED_LENGTH_CAPPED="length", FINISHED_REPETITION="repeat")
    stop = method("vllm/v1/core/sched/utils.py", "check_stop", RequestStatus=status)
    update = method(SCHED, "_update_request_with_output", check_stop=stop)

    class Request:
        pooling_params = None
        sampling_params = NS(eos_token_id=99, stop_token_ids=[77], min_tokens=0,
                             repetition_detection=None)
        max_tokens = cap
        def __init__(self): self.output_token_ids = []
        @property
        def num_output_tokens(self): return len(self.output_token_ids)
        @property
        def num_tokens(self): return 40 + len(self.output_token_ids)
        def append_output_token_ids(self, value): self.output_token_ids.append(value)
    new_tokens, _ = update(NS(max_model_len=4096), Request(), list(tokens))
    assert new_tokens == expected


@pytest.mark.parametrize("case", ["missing_worker", "prefill", "zero_output", "terminal", "query_mismatch", "unknown_state"])
def test_ineligible_observations_break_windows(case):
    obs = RetirementObserver(RUNTIME, intervals=1)
    s = schedule("1")
    h = Harness([s], obs)
    h.execute(s)
    if case == "missing_worker": obs.pending[id(s)]["worker"] = None
    elif case == "prefill": obs.pending[id(s)]["worker"]["prefill"] = True
    elif case == "query_mismatch": obs.pending[id(s)]["worker"]["queries"] = (1, 1)
    def consume(s, o):
        result = h.consume(s, o)
        if case == "zero_output": result[0].outputs = []
        elif case == "terminal": result[0].outputs[0].finish_reason = "stop"
        return result
    obs.update(consume, s, h.outputs["1"], unsafe_requests=None if case == "unknown_state" else ())
    assert obs.events[-1]["status"] == "unsupported" and obs.windows == []


def test_failed_execute_after_none_sample_result_is_not_a_retirement():
    obs = RetirementObserver(RUNTIME)
    h = Harness([schedule("1")], obs)
    h.pending_exec_failure = True
    h.tick()
    h.futures["1"].set_result(None)
    h.exec_futures["1"].set_exception(RuntimeError("execute failed"))
    with pytest.raises(RuntimeError, match="execute failed"): h.tick()
    assert h.actual_pairs == [] and obs.events[-1]["status"] == "failed"


def test_observation_error_preserves_delegate_result_and_call_count():
    obs = RetirementObserver(RUNTIME)
    calls = []
    result = object()  # unsupported schema, but must reach the engine unchanged
    value = obs.update(lambda s, o: calls.append(1) or result, schedule("1"), object())
    assert value is result and calls == [1]
    assert obs.events[-1]["reason"].startswith("OBSERVATION_ERROR")


def test_context_signature_uses_incarnations_and_exact_trajectory():
    obs = RetirementObserver(RUNTIME)
    h = Harness([schedule("1")], obs)
    h.tick(queue=False)
    event = obs.events[0]
    signature = trajectory_signature([event])
    event["worker"]["contexts"] = (40, 41)
    assert trajectory_signature([event]) != signature
    event["worker"]["contexts"] = (40, 40)
    event["incarnations"]["a"] = "a:2"
    assert trajectory_signature([event]) != signature


def test_bounded_buffer_overflow_blocks_export_as_complete_evidence():
    obs = RetirementObserver(RUNTIME, capacity=2)
    h = Harness([schedule(str(i)) for i in range(4)], obs)
    for _ in range(4): h.tick(queue=False)
    assert obs.export()["overflow"]
    assert len(obs.events) == 2


def test_pinned_async_future_materializes_only_its_own_output_once():
    path = "vllm/v1/executor/uniproc_executor.py"
    tree = source(path)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AsyncOutputFuture")
    namespace = dict(Future=Future)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    calls, expected = [], object()
    future = namespace["AsyncOutputFuture"](NS(get_output=lambda: calls.append(1) or expected), True)
    assert future.result() is future.result() is expected
    assert calls == [1]


@pytest.mark.parametrize("current,previous,expected", [
    (["b", "new", "a"], {"a": 0, "b": 1}, [1, -1, 0]),
    (["c", "b"], {"a": 0, "b": 1, "c": 2}, [2, 1]),
])
def test_pinned_worker_prev_positions_handles_reorder_compaction_and_slot_reuse(current, previous, expected):
    class CPUArray:
        def __init__(self): self.values = [99] * len(current)
        def __getitem__(self, key): return self if isinstance(key, slice) else self.values[key]
        def __setitem__(self, key, value): self.values[key] = value
        def fill(self, value): self.values[:] = [value] * len(self.values)
    positions = CPUArray()
    runner = NS(input_batch=NS(req_ids=current, prev_req_id_to_index=previous),
                prev_positions=NS(np=positions))
    compute = method("vllm/v1/worker/gpu_model_runner.py", "_compute_prev_positions")
    compute(runner, len(current))
    assert positions.values == expected
