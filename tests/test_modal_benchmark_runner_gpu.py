"""Offline tests for the one-container-per-GPU Modal benchmark app (fake `modal`, no network)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import run_benchmark_campaign as camp

RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "modal_benchmark_runner.py"


class FakeDict(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True

    def contains(self, key):
        return key in self


class FakeCall:
    def __init__(self, object_id, result=None, *, running=False, on_get=None, graph_status=None):
        self.object_id = object_id
        self._result = result
        self._running = running
        self._on_get = on_get
        self._graph_status = graph_status

    def get_call_graph(self):
        if self._graph_status is None:
            raise RuntimeError("no call graph in this fake")
        return [SimpleNamespace(status=self._graph_status, children=[])]

    def get(self, timeout=None):
        if self._on_get:
            self._on_get()
        if self._running:
            raise TimeoutError
        return self._result


def _load(monkeypatch, *, locks=None, spawned=None, calls=None, volume=None):
    """Import the runner with a recording fake `modal`; returns (module, recorded function kwargs, fake)."""
    from flexsds.server import modal_image

    image = Mock()
    for method in ("pip_install", "uv_pip_install", "env", "add_local_dir", "add_local_file", "run_commands"):
        getattr(image, method).return_value = image
    monkeypatch.setattr(modal_image, "serving_image", lambda: image)
    functions: dict[str, dict] = {}
    app = Mock()

    def fake_function(**kwargs):
        def apply(fn):
            functions[fn.__name__] = kwargs
            return fn
        return apply

    app.function.side_effect = fake_function
    locks = FakeDict() if locks is None else locks
    spawned = [] if spawned is None else spawned
    calls = {} if calls is None else calls

    def from_name(app_name, fn_name):
        def spawn(*args):
            call = spawned_results.pop(0) if spawned_results else FakeCall(f"fc-{len(spawned)}", running=True)
            spawned.append((fn_name, args))
            calls[call.object_id] = call
            return call
        return SimpleNamespace(spawn=spawn)

    spawned_results: list[FakeCall] = []
    fake = SimpleNamespace(
        App=Mock(return_value=app), is_local=lambda: True,
        Volume=SimpleNamespace(from_name=Mock(return_value=volume or Mock())),
        Dict=SimpleNamespace(from_name=Mock(return_value=locks)),
        Function=SimpleNamespace(from_name=from_name),
        FunctionCall=SimpleNamespace(from_id=lambda cid: calls[cid]),
        current_function_call_id=lambda: "fc-current",
        web_server=Mock(side_effect=AssertionError("web_server must not be used")),
        concurrent=Mock(side_effect=AssertionError("no web endpoint in the benchmark app")),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    spec = importlib.util.spec_from_file_location("bench_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._spawned_results = spawned_results  # test hook: queue the next spawn results
    return module, functions, fake


def test_runner_defines_one_function_per_gpu_with_exact_resources(monkeypatch):
    module, functions, _ = _load(monkeypatch)
    assert set(functions) == {"run_gpu_T4", "run_gpu_A10", "run_gpu_L40S", "run_gpu_A100"}
    assert functions["run_gpu_A100"]["gpu"] == "A100-80GB"
    assert functions["run_gpu_A10"]["gpu"] == "A10"
    for kwargs in functions.values():
        assert kwargs["max_containers"] == 1
        assert kwargs["scaledown_window"] == 2
        assert kwargs["timeout"] == 20 * 3600


def _record():
    return camp.plan("camp1", "a" * 40, mode="fast", gpus="A10", workloads="W2")


def test_submit_gpu_retries_wrong_card_then_gives_up(monkeypatch):
    module, _, _ = _load(monkeypatch)
    for i in range(12):
        module._spawned_results.append(FakeCall(f"fc-w{i}", {"stopped": "WrongGpu: A10 must be measured on its exact card, got NVIDIA A10G"}))
    with pytest.raises(RuntimeError, match="NVIDIA A10G") as exc:
        module.submit_gpu(_record(), "A10", ["A10__vanilla__W2__c1"], None, tries=12, wait_s=0, poll_s=0)
    assert "12" in str(exc.value)


def test_submit_gpu_returns_call_id_when_card_is_right(monkeypatch):
    locks = FakeDict()
    module, _, _ = _load(monkeypatch, locks=locks)
    module._spawned_results.append(FakeCall("fc-wrong", {"stopped": "WrongGpu: got NVIDIA A10G"}))
    # the container takes the lock once its card check passed
    running = FakeCall("fc-good", running=True, on_get=lambda: locks.__setitem__("camp1:A10", "fc-good"))
    module._spawned_results.append(running)
    assert module.submit_gpu(_record(), "A10", ["A10__vanilla__W2__c1"], None, tries=12, wait_s=0, poll_s=0) == "fc-good"


def test_submit_gpu_clears_stale_lock_of_finished_call(monkeypatch):
    locks = FakeDict({"camp1:A10": "fc-old"})
    calls = {"fc-old": FakeCall("fc-old", {"cells": {}})}
    module, _, _ = _load(monkeypatch, locks=locks, calls=calls)
    module._spawned_results.append(FakeCall("fc-new", {"cells": {"A10__vanilla__W2__c1": "done"}}))
    assert module.submit_gpu(_record(), "A10", ["A10__vanilla__W2__c1"], None, tries=1, wait_s=0, poll_s=0) == "fc-new"
    assert locks.get("camp1:A10") != "fc-old"


def test_submit_gpu_refuses_when_a_live_call_holds_the_gpu(monkeypatch):
    locks = FakeDict({"camp1:A10": "fc-live"})
    calls = {"fc-live": FakeCall("fc-live", running=True)}
    module, _, _ = _load(monkeypatch, locks=locks, calls=calls)
    with pytest.raises(RuntimeError, match="already running"):
        module.submit_gpu(_record(), "A10", ["A10__vanilla__W2__c1"], None, tries=1, wait_s=0, poll_s=0)


def test_gpu_call_state(monkeypatch):
    calls = {"fc-r": FakeCall("fc-r", running=True), "fc-d": FakeCall("fc-d", {"cells": {"k": "done"}}),
             "fc-s": FakeCall("fc-s", {"stopped": "WrongGpu: x"})}
    module, _, _ = _load(monkeypatch, calls=calls)
    assert module.gpu_call_state("fc-r") == "RUNNING"
    assert module.gpu_call_state("fc-d") == "DONE"
    assert module.gpu_call_state("fc-s") == "FAILED"


def test_gpu_call_state_reports_a_dead_call_whose_get_still_times_out(monkeypatch):
    """Seen on Modal 26/11: the container got SIGINT, the call graph says FAILURE, get(timeout=0) still times out."""
    calls = {"fc-dead": FakeCall("fc-dead", running=True, graph_status=2),      # InputStatus.FAILURE
             "fc-term": FakeCall("fc-term", running=True, graph_status=3),      # TERMINATED
             "fc-pend": FakeCall("fc-pend", running=True, graph_status=0)}      # PENDING = still running
    module, _, _ = _load(monkeypatch, calls=calls)
    assert module.gpu_call_state("fc-dead") == "FAILED"
    assert module.gpu_call_state("fc-term") == "FAILED"
    assert module.gpu_call_state("fc-pend") == "RUNNING"


class FakeVolume:
    """Like modal.Volume: recursive listdir returns directory entries (type 2) as well as files (type 1)."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def listdir(self, path, recursive=False):
        prefix = path.rstrip("/") + "/"
        dirs = sorted({p.rsplit("/", i)[0] for p in self.files if p.startswith(prefix) for i in (1, 2)
                       if p.rsplit("/", i)[0].startswith(prefix)})
        return ([SimpleNamespace(path=d, type=SimpleNamespace(value=2, name="DIRECTORY")) for d in dirs]
                + [SimpleNamespace(path=p, type=SimpleNamespace(value=1, name="FILE")) for p in self.files
                   if p.startswith(prefix)])

    def read_file(self, path):
        if path not in self.files:
            raise IsADirectoryError(path)
        yield self.files[path]


def test_collect_gpu_writes_attempt_layout_and_refuses_conflicts(tmp_path, monkeypatch):
    base = "flexsds_bench/camp1/A10/A10__vanilla__W2__c8"
    bench = {"bench_run_id": "camp1-A10-vanilla-W2-c8-a001"}
    volume = FakeVolume({f"{base}/bench.json": json.dumps(bench).encode(), f"{base}/raw.jsonl": b"{}\n",
                         "flexsds_bench/camp1/A10/A10__static__W2__c8/failure.json": b'{"status": "failed:server"}',
                         "flexsds_bench/camp1/A10/_servers/W2__vanilla/vllm.log": b"log"})
    module, _, _ = _load(monkeypatch, volume=volume)
    got = module.collect_gpu("camp1", "A10", tmp_path)
    assert got == ["A10__vanilla__W2__c8"]
    out = tmp_path / "camp1-A10-vanilla-W2-c8-a001"
    assert json.loads((out / "bench.json").read_text(encoding="utf-8")) == bench
    assert (tmp_path.parent / "servers" / "A10" / "W2__vanilla" / "vllm.log").read_bytes() == b"log"
    assert module.collect_gpu("camp1", "A10", tmp_path) == ["A10__vanilla__W2__c8"]  # idempotent
    (out / "raw.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.collect_gpu("camp1", "A10", tmp_path)


def test_stop_releases_locks_of_the_campaign(monkeypatch):
    locks = FakeDict({"camp1:A10": "fc-1", "camp1:T4": "fc-2", "other:A10": "fc-3"})
    module, _, _ = _load(monkeypatch, locks=locks)
    module.release_locks("camp1")
    assert dict(locks) == {"other:A10": "fc-3"}


def test_sdk_bridge_exposes_new_actions():
    from experiments import modal_benchmark_runner as runner

    actions = runner.SDK_ACTIONS
    assert set(actions) == {"submit_gpu", "gpu_call_state", "collect_gpu", "release_locks", "cancel_call"}
    assert not hasattr(runner, "bench_submit")


def test_collect_gpu_keeps_failure_records_of_cells_without_results(tmp_path, monkeypatch):
    volume = FakeVolume({"flexsds_bench/camp1/T4/T4__static__W2__c8/failure.json":
                         b'{"status": "failed:server", "error": "OOM"}'})
    module, _, _ = _load(monkeypatch, volume=volume)
    assert module.collect_gpu("camp1", "T4", tmp_path / "attempts") == []
    saved = tmp_path / "failures" / "T4__static__W2__c8.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["status"] == "failed:server"


def test_collect_gpu_reads_attempt_scoped_directories(tmp_path, monkeypatch):
    base = "flexsds_bench/camp1/A100"
    volume = FakeVolume({
        f"{base}/A100__static__W1_V3__c8/a001/bench.json": json.dumps({"bench_run_id": "camp1-A100-static-W1_V3-c8-a001"}).encode(),
        f"{base}/A100__static__W1_V3__c8/a001/raw.jsonl": b"old\n",
        f"{base}/A100__static__W1_V3__c8/a002/bench.json": json.dumps({"bench_run_id": "camp1-A100-static-W1_V3-c8-a002"}).encode(),
        f"{base}/A100__static__W1_V3__c8/a002/raw.jsonl": b"new\n",
        f"{base}/A100__adaptive__W3__c8/a002/failure.json": b'{"status": "failed:server", "error": "OOM"}',
    })
    module, _, _ = _load(monkeypatch, volume=volume)
    got = module.collect_gpu("camp1", "A100", tmp_path / "attempts")
    assert sorted(got) == ["A100__static__W1_V3__c8", "A100__static__W1_V3__c8"]
    attempts = tmp_path / "attempts"
    assert (attempts / "camp1-A100-static-W1_V3-c8-a001" / "raw.jsonl").read_bytes() == b"old\n"
    assert (attempts / "camp1-A100-static-W1_V3-c8-a002" / "raw.jsonl").read_bytes() == b"new\n"
    saved = tmp_path / "failures" / "A100__adaptive__W3__c8.a002.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["error"] == "OOM"
