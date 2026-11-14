"""Verify shutdown ordering without starting a process or GPU."""
import signal
import subprocess
from types import SimpleNamespace
from experiments import m4_runtime as runtime
from experiments.m4_shutdown import cancel_before_shutdown
import pytest


def test_graceful_shutdown_signals_only_api_server(monkeypatch):
    events = []
    process = SimpleNamespace(pid=123, poll=lambda: None,
        terminate=lambda: events.append("api-term"),
        wait=lambda timeout: events.append(("wait", timeout)))
    monkeypatch.setattr(runtime.os, "killpg", lambda *a: events.append(("group", a)), raising=False)
    runtime.stop_server(process)
    assert events == ["api-term", ("wait", 60)]


def test_hung_server_still_terminates_its_process_group(monkeypatch):
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    events = []
    def wait(timeout):
        events.append(("wait", timeout))
        if timeout == 60:
            raise subprocess.TimeoutExpired("vllm", timeout)
    process = SimpleNamespace(pid=123, poll=lambda: None,
        terminate=lambda: events.append("api-term"), wait=wait)
    monkeypatch.setattr(runtime.os, "killpg", lambda *a: events.append(("group", a)), raising=False)
    runtime.stop_server(process)
    assert events == ["api-term", ("wait", 60), ("group", (123, signal.SIGKILL)), ("wait", 30)]


def test_output_consumer_is_cancelled_before_engine_teardown():
    events = []
    handler = object()
    client = SimpleNamespace(output_handler=handler)
    def original(self, timeout):
        assert self is client and timeout == 7
        events.append("engine-stop")
        return "result"
    def cancel(task):
        assert task is handler
        events.append("cancel-consumer")
    shutdown = cancel_before_shutdown(original, cancel)
    assert shutdown(client, timeout=7) == "result"
    assert events == ["cancel-consumer", "engine-stop"]


def test_lifecycle_adapter_does_not_hide_shutdown_failures():
    def broken(self):
        raise RuntimeError("real-shutdown-failure")
    with pytest.raises(RuntimeError, match="real-shutdown-failure"):
        cancel_before_shutdown(broken, lambda task: None)(SimpleNamespace(output_handler=None))
