"""User-run Linux process lifecycle helpers. Importing this module starts nothing."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from experiments.m4_protocol import ROOT, write_new
from experiments.m4_treatments import definition, runtime_env, serve_args


def http_json(url: str, payload=None, timeout=120):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def gpu_snapshot() -> dict:
    result = {}
    for key, fields in (("memory", ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]),
                        ("processes", ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])):
        proc = subprocess.run(["nvidia-smi", *fields], capture_output=True, text=True, timeout=20)
        if proc.returncode:
            raise RuntimeError("GPU lifecycle observation unavailable")
        result[key] = proc.stdout.strip()
    return result


def require_gpu_release(before: dict, after: dict):
    if after["processes"] != before["processes"]:
        raise RuntimeError("Residual/changed GPU process set; stop before next treatment")
    before_mb = float(before["memory"].split(",")[0])
    after_mb = float(after["memory"].split(",")[0])
    if after_mb > before_mb + 128:
        raise RuntimeError("Unexplained residual GPU memory >128 MiB; inspect lifecycle evidence")


@contextmanager
def serving(treatment: str, directory: Path, source_commit: str, port=8000):
    directory.mkdir(parents=True, exist_ok=False)
    before = gpu_snapshot()
    if before["processes"]:
        raise RuntimeError("A single clean GPU is required; existing processes were not terminated")
    command = ["vllm", "serve", *serve_args(treatment, port)]
    # Only record our explicit overrides, never the inherited secret environment.
    overrides = runtime_env(treatment, directory)
    write_new(directory / "launch.json", dict(command=command, env=overrides,
              definition=definition(treatment), source_commit=source_commit, gpu_before=before))
    process = None
    base = f"http://127.0.0.1:{port}/v1"
    with (directory / "server.log").open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **overrides),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_new(directory / "pid.json", dict(pid=process.pid, started_at=time.time()))
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server startup failed ({process.returncode}); see preserved log")
                try:
                    models = http_json(base + "/models", timeout=5)
                    if [m["id"] for m in models["data"]] != [definition(treatment)["served_model_name"]]:
                        raise RuntimeError("Endpoint model identity mismatch")
                    write_new(directory / "models.json", models)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server readiness timeout")
                    time.sleep(1)
            yield base
        finally:
            if process is not None:
                os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                # Observe worker teardown, not CUDA allocator cache in a live process.
                after = gpu_snapshot()
                write_new(directory / "lifecycle.json", dict(returncode=process.returncode,
                          gpu_before=before, gpu_after=after, finished_at=time.time()))
                require_gpu_release(before, after)
