"""Verify shutdown ordering without starting a process or GPU."""
import signal
import subprocess
from types import SimpleNamespace
from experiments import m4_runtime as runtime


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
