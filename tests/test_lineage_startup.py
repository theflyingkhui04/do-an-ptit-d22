"""Failed-startup ownership against captured constructors and CPU worker stubs."""
import ast
from contextvars import copy_context
import gc
import hashlib
import importlib
import json
import sys
from threading import current_thread
from types import ModuleType, SimpleNamespace
from weakref import ref

import pytest

from test_adaptive_scheduler_glue import (
    FakeAsyncScheduler, captured_function, installed_capture, scheduler_module,
)
from src.server import lineage
from src.server.lineage_startup import install_enginecore_startup_binding


@pytest.fixture
def startup_runtime(scheduler_module, monkeypatch):
    """Real project wrappers; pinned Executor init and EngineCore prefix; no GPU."""
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1, data_parallel_size=1,
            data_parallel_rank_local=1, distributed_executor_backend="uni"),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=SimpleNamespace(method="draft_model", disable_padded_drafter_batch=False),
        fail_stage=None, fail_kv=False,
    )
    for field in ("model_config", "cache_config", "lora_config", "load_config",
                  "device_config", "observability_config"):
        setattr(config, field, SimpleNamespace())
    created, publications = [], []

    def fail(stage):
        if config.fail_stage == stage:
            raise RuntimeError("injected " + stage)

    def scheduler_class():
        fail("resolution")
        return scheduler_module.AdaptiveAsyncScheduler
    config.scheduler_config.get_scheduler_cls = scheduler_class

    def allocate_tags():
        # Actual pinned Executor.__init__ does this after _init_executor returns.
        fail("after_publication")
        return set()

    class UniProcExecutor:
        __init__ = captured_function(installed_capture() / "v1/executor/abstract.py",
            "Executor", "__init__", {"instrument": lambda **k: lambda f: f,
                                     "set": allocate_tags})

        def _init_executor(self):
            created.append(self)
            fail("before_publication")
            runner_type = type("GPUModelRunner", (), {
                "__module__": "vllm.v1.worker.gpu_model_runner"})
            runner = runner_type()
            runner.use_async_scheduling = True
            runner._copy_draft_token_ids_to_cpu = lambda *a, **k: None
            self.driver_worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))

        def shutdown(self):
            pass

    upstream = ModuleType("vllm.v1.executor.uniproc_executor")
    upstream.UniProcExecutor = UniProcExecutor
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    monkeypatch.delitem(sys.modules, "src.server.lineage_executor", raising=False)
    executor_module = importlib.import_module("src.server.lineage_executor")
    # The real version/hash gates operate against the captured source cache.
    executor_module.verify_sources()
    core = scheduler_module._cpu_engine_core
    assert getattr(core.__init__, "_lineage_startup_binding", False)

    original_register = executor_module.register_lineage_executor
    def publish(c, owner):
        handle = original_register(c, owner)
        publications.append((handle, lineage._registry[id(c)]))
        return handle
    monkeypatch.setattr(executor_module, "register_lineage_executor", publish)
    globals_ = core.__init__.__wrapped__.__globals__
    def structured(c):
        fail("structured")
        return SimpleNamespace()
    def blocks(*args):
        fail("blocks")
        return 16, 16
    monkeypatch.setitem(globals_, "StructuredOutputManager", structured)
    monkeypatch.setitem(globals_, "resolve_kv_cache_block_sizes", blocks)
    runtime = SimpleNamespace(config=config, core=core, created=created,
        publications=publications, plain=UniProcExecutor,
        executor=executor_module.LineageUniProcExecutor, scheduler=scheduler_module)
    try:
        yield runtime
    finally:
        for handle, _ in publications:
            lineage.release_ledger(handle)


def failed_a(runtime, stage="kv"):
    runtime.config.fail_kv = stage == "kv"
    runtime.config.fail_stage = stage
    try:
        runtime.core(runtime.config, runtime.executor, False)
    except RuntimeError as error:
        retained = error
    else:
        pytest.fail("injected startup did not fail")
    runtime.config.fail_kv = False
    runtime.config.fail_stage = None
    assert lineage._startup.get() is None
    return retained


def reject_plain_b(runtime, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("plain B reached base scheduler initialization/readiness")
    monkeypatch.setattr(FakeAsyncScheduler, "__init__", unexpected)
    with pytest.raises(RuntimeError, match="lineage-capable executor"):
        runtime.core(runtime.config, runtime.plain, False)
    assert lineage._startup.get() is None


def test_diagnostics_trace_exact_attempt_without_retaining_ownership(startup_runtime, monkeypatch, tmp_path):
    path = tmp_path / "startup.jsonl"
    monkeypatch.setenv("ADAPTIVE_STARTUP_TELEMETRY_PATH", str(path))
    r = startup_runtime
    engine = r.core(r.config, r.executor, False)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    events = {row["event"]: row for row in rows}
    names = ["attempt_created", "executor_published", "scheduler_consumed",
             "scheduler_initialized", "attempt_closed"]
    assert len({events[name]["attempt_id"] for name in names}) == 1
    assert len({events[name]["generation_id"] for name in names[1:]}) == 1
    assert len({events[name]["ledger_id"] for name in names[1:]}) == 1
    assert events["scheduler_consumed"]["same_active_attempt"] is True
    closed = events["attempt_closed"]
    assert closed["succeeded"] and closed["context_reset"]
    assert closed["registration_retained"] and closed["consumed"]
    assert rows[0]["event"] == "enginecore_wrapper_entered"
    registration = r.publications[-1][0]
    lineage.release_ledger(registration)
    assert id(r.config) not in lineage._registry
    assert engine.scheduler._lineage is registration.ledger


def test_diagnostics_io_failure_does_not_reject_startup(startup_runtime, monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTIVE_STARTUP_TELEMETRY_PATH", str(tmp_path / "absent" / "x.jsonl"))
    r = startup_runtime
    engine = r.core(r.config, r.executor, False)
    assert engine.scheduler._lineage is r.publications[-1][0].ledger


def test_startup_f1_failed_attempt_revokes_unconsumed_generation(startup_runtime):
    r = startup_runtime
    error = failed_a(r)
    ga, state = r.publications[-1]
    assert error.__traceback__ is not None
    assert state.consumed is False and ga.owner() is r.created[0]
    assert id(r.config) not in lineage._registry
    assert ga.ledger.active == ga.ledger.executions == {}


def test_startup_f2_plain_executor_cannot_consume_failed_a(startup_runtime, monkeypatch):
    r = startup_runtime
    error = failed_a(r)
    ga, state = r.publications[-1]
    reject_plain_b(r, monkeypatch)
    assert error.__traceback__ and ga.owner() is r.created[0]
    assert not state.consumed  # B never consumed or bound A's ledger.


def test_startup_f3_traceback_keeps_owner_alive_without_authorizing_b(startup_runtime, monkeypatch):
    r = startup_runtime
    error = failed_a(r)
    ga, _ = r.publications[-1]
    r.created.clear()
    # No GC: traceback itself must retain the failed EngineCore and executor.
    frames = []
    trace = error.__traceback__
    while trace:
        frames.append(trace.tb_frame)
        trace = trace.tb_next
    assert any(getattr(f.f_locals.get("self"), "model_executor", None) is ga.owner()
               for f in frames)
    assert ga.owner() is not None
    reject_plain_b(r, monkeypatch)


def test_startup_f4_valid_restart_has_fresh_attempt_generation_and_ledger(startup_runtime):
    r = startup_runtime
    error = failed_a(r)
    ga, _ = r.publications[-1]
    b = r.core(r.config, r.executor, False)
    gb = b.scheduler._lineage_registration
    assert ga.owner() is not None and error.__traceback__
    assert gb.attempt is not ga.attempt and gb.attempt.token is not ga.attempt.token
    assert gb.token is not ga.token and gb.ledger is not ga.ledger
    assert gb.owner() is b.model_executor
    assert b.scheduler._lineage is b.model_executor._lineage is gb.ledger
    assert lineage._registry[id(r.config)].registration is gb
    assert lineage._registry[id(r.config)].consumed


def test_startup_f5_exact_same_config_does_not_authorize_b(startup_runtime, monkeypatch):
    r = startup_runtime
    failed_a(r)
    ga, _ = r.publications[-1]
    assert ga.config is r.config
    reject_plain_b(r, monkeypatch)
    assert r.created[-1].vllm_config is ga.config


def test_startup_f6_exact_same_thread_does_not_authorize_b(startup_runtime, monkeypatch):
    r = startup_runtime
    thread = current_thread()
    failed_a(r)
    ga, _ = r.publications[-1]
    assert ga.publisher_thread is ga.attempt.publisher_thread is thread
    reject_plain_b(r, monkeypatch)
    assert current_thread() is thread


def test_startup_f7_orphan_cleanup_and_future_recovery_are_deterministic(startup_runtime):
    r = startup_runtime
    error = failed_a(r)
    ga, _ = r.publications[-1]
    assert id(r.config) not in lineage._registry and ga.owner() is not None
    b = r.core(r.config, r.executor, False)
    gb = b.scheduler._lineage_registration
    assert lineage._registry[id(r.config)].registration is gb
    assert lineage._startup.get() is None
    assert error.__traceback__


def test_startup_f8_old_teardown_and_gc_cannot_revoke_b(startup_runtime):
    r = startup_runtime
    error = failed_a(r)
    ga, _ = r.publications[-1]
    old = ref(ga.owner())
    b = r.core(r.config, r.executor, False)
    gb = b.scheduler._lineage_registration
    marker = object()
    gb.ledger.active["B"] = marker
    ga.owner().shutdown()
    lineage.release_ledger(ga)
    ga.owner.__callback__(ga.owner)
    r.created.pop(0)
    del error
    gc.collect()
    assert old() is None
    assert lineage._registry[id(r.config)].registration is gb
    assert gb.ledger.active["B"] is marker


@pytest.mark.parametrize("stage", ["before_publication", "after_publication", "kv",
                                    "structured", "resolution", "blocks"])
def test_startup_failure_boundaries_revoke_only_the_failed_attempt(startup_runtime, stage):
    r = startup_runtime
    error = failed_a(r, stage)
    assert error.__traceback__ and id(r.config) not in lineage._registry
    if stage == "before_publication":
        assert r.publications == []
    else:
        ga, state = r.publications[-1]
        assert ga.owner() is not None and not state.consumed
    b = r.core(r.config, r.executor, False)
    assert b.scheduler._lineage is b.model_executor._lineage


def test_startup_scheduler_failure_after_consume_is_fail_closed(startup_runtime, monkeypatch):
    r = startup_runtime
    def fail(*a, **k):
        raise RuntimeError("base scheduler failed")
    monkeypatch.setattr(FakeAsyncScheduler, "__init__", fail)
    with pytest.raises(RuntimeError, match="base scheduler failed"):
        r.core(r.config, r.executor, False)
    ga, state = r.publications[-1]
    assert state.consumed and id(r.config) not in lineage._registry
    with pytest.raises(RuntimeError, match="lineage-capable executor"):
        lineage.require_lineage_executor(r.config)
    lineage.release_ledger(ga)
    assert lineage._startup.get() is None


def test_startup_attempt_identity_rejects_live_pending_outer_attempt(scheduler_module):
    from test_adaptive_scheduler_glue import cpu_config, paired_cpu_config
    config = cpu_config()
    with lineage._engine_startup(config) as a:
        paired_cpu_config(scheduler_module, config)
        handle = lineage._registry[id(config)].registration
        with lineage._engine_startup(config) as b:
            assert a is not b
            with pytest.raises(RuntimeError, match="another EngineCore startup attempt"):
                lineage.require_lineage_executor(config)
        assert not lineage._registry[id(config)].consumed
        assert lineage.require_lineage_executor(config) is handle
    lineage.release_ledger(handle)


def test_startup_copied_context_cannot_reactivate_completed_attempt(scheduler_module):
    from test_adaptive_scheduler_glue import cpu_config, paired_cpu_config
    config = cpu_config()
    with lineage._engine_startup(config):
        paired_cpu_config(scheduler_module, config)
        copied = copy_context()
    assert id(config) not in lineage._registry
    owner = scheduler_module._cpu_executor_owners[-1]
    with pytest.raises(RuntimeError, match="active EngineCore startup attempt"):
        copied.run(lineage.register_lineage_executor, config, owner)
    assert lineage._startup.get() is None


@pytest.mark.parametrize("lineage_executor", [False, True])
def test_startup_wrapper_is_idempotent_and_non_adaptive_path_unaffected(startup_runtime, lineage_executor):
    r = startup_runtime
    wrapper = r.core.__init__
    install_enginecore_startup_binding()
    assert r.core.__init__ is wrapper
    r.config.scheduler_config.get_scheduler_cls = lambda: FakeAsyncScheduler
    engine = r.core(r.config, r.executor if lineage_executor else r.plain, False)
    assert isinstance(engine.scheduler, FakeAsyncScheduler)
    assert id(r.config) not in lineage._registry and lineage._startup.get() is None


def test_startup_nested_plain_engine_cannot_inherit_outer_readiness(startup_runtime, monkeypatch):
    r = startup_runtime
    original = r.core._initialize_kv_caches
    observed = []
    def initialize(engine, config):
        if not observed:
            ga, state = r.publications[-1]
            observed.append(ga)
            with pytest.raises(RuntimeError, match="another EngineCore startup attempt"):
                r.core(config, r.plain, False)
            assert lineage._registry[id(config)].registration is ga and not state.consumed
        return original(engine, config)
    monkeypatch.setattr(r.core, "_initialize_kv_caches", initialize)
    a = r.core(r.config, r.executor, False)
    assert a.scheduler._lineage_registration is observed[0]
    assert lineage._startup.get() is None


def test_startup_outer_failure_cleanup_cannot_delete_nested_replacement(startup_runtime, monkeypatch):
    r = startup_runtime
    original = r.core._initialize_kv_caches
    completed = []
    entered = []
    def initialize(engine, config):
        if not entered:
            ga = r.publications[-1][0]
            entered.append(ga)
            lineage.release_ledger(ga)
            completed.append(r.core(config, r.executor, False))
            raise RuntimeError("outer failed after nested replacement")
        return original(engine, config)
    monkeypatch.setattr(r.core, "_initialize_kv_caches", initialize)
    with pytest.raises(RuntimeError, match="outer failed"):
        r.core(r.config, r.executor, False)
    gb = completed[0].scheduler._lineage_registration
    assert lineage._registry[id(r.config)].registration is gb
    assert gb is not entered[0] and gb.attempt is not entered[0].attempt
    assert lineage._startup.get() is None


def test_startup_scheduler_retry_within_same_attempt_stays_consumed(scheduler_module, monkeypatch):
    from test_adaptive_scheduler_glue import cpu_config, paired_cpu_config
    config = cpu_config()
    original = FakeAsyncScheduler.__init__
    with lineage._engine_startup(config):
        paired_cpu_config(scheduler_module, config)
        handle = lineage._registry[id(config)].registration
        def fail(*a, **k):
            raise RuntimeError("base failed")
        monkeypatch.setattr(FakeAsyncScheduler, "__init__", fail)
        with pytest.raises(RuntimeError, match="base failed"):
            scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
        monkeypatch.setattr(FakeAsyncScheduler, "__init__", original)
        assert lineage._registry[id(config)].consumed
        with pytest.raises(RuntimeError, match="already consumed"):
            scheduler_module.AdaptiveAsyncScheduler(vllm_config=config)
        lineage.release_ledger(handle)
    assert id(config) not in lineage._registry


def test_startup_binding_rejects_unknown_constructor(startup_runtime, monkeypatch):
    r = startup_runtime
    monkeypatch.setattr(r.core, "__init__", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Unsupported EngineCore source"):
        install_enginecore_startup_binding()


def test_startup_baseexception_revokes_and_resets_context(startup_runtime, monkeypatch):
    r = startup_runtime
    def interrupt(*args):
        raise KeyboardInterrupt("startup interrupted")
    monkeypatch.setattr(r.core, "_initialize_kv_caches", interrupt)
    with pytest.raises(KeyboardInterrupt, match="startup interrupted"):
        r.core(r.config, r.executor, False)
    ga, state = r.publications[-1]
    assert not state.consumed and ga.owner() is not None
    assert id(r.config) not in lineage._registry
    assert lineage._startup.get() is None


def test_startup_pinned_order_and_preconstruction_executor_resolution():
    capture = installed_capture()
    manifest = json.loads((capture.parent.parent / "installed_source_manifest.json").read_text())
    paths = ["vllm/v1/engine/core.py", "vllm/v1/engine/core_client.py",
             "vllm/v1/engine/llm_engine.py", "vllm/v1/engine/async_llm.py",
             "vllm/v1/executor/abstract.py"]
    trees = {}
    for path in paths:
        record = next(r for r in manifest["files"] if r["path"] == path)
        data = (capture.parent / path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        trees[path] = ast.parse(data)
    engine = next(n for n in trees[paths[0]].body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    init = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    statements = [ast.unparse(n) for n in init.body]
    def index(text):
        return next(i for i, statement in enumerate(statements) if text in statement)
    assert index("executor_class(vllm_config)") < index("self._initialize_kv_caches(vllm_config)")
    assert index("self._initialize_kv_caches(vllm_config)") < index("StructuredOutputManager(vllm_config)")
    assert index("StructuredOutputManager(vllm_config)") < index("get_scheduler_cls()")
    assert index("get_scheduler_cls()") < index("resolve_kv_cache_block_sizes(")
    assert index("resolve_kv_cache_block_sizes(") < index("self.scheduler: SchedulerInterface = Scheduler(")
    # Both canonical entrypoints resolve/import the executor before constructing
    # their engine/client. The custom executor import installs the wrapper here.
    for path, cls_name in [(paths[2], "LLMEngine"), (paths[3], "AsyncLLM")]:
        cls = next(n for n in trees[path].body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        for method in ("from_vllm_config", "from_engine_args"):
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
            returned = next(n.value for n in fn.body if isinstance(n, ast.Return))
            assert isinstance(returned, ast.Call) and ast.unparse(returned.func) == "cls"
            executor = next(k.value for k in returned.keywords if k.arg == "executor_class")
            if method == "from_vllm_config":
                assert ast.unparse(executor) == "Executor.get_class(vllm_config)"
            else:
                assert ast.unparse(executor) == "executor_class"
                assert any(isinstance(n, ast.Assign) and ast.unparse(n.value) == "Executor.get_class(vllm_config)"
                           for n in fn.body[:fn.body.index(next(n for n in fn.body if isinstance(n, ast.Return)))])
