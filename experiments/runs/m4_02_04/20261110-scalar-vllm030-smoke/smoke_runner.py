"""Isolated real-engine smoke; no benchmark or controller changes."""
import concurrent.futures
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import modal
from experiments.modal_runner import image, SOURCE_ROOT, SOURCE_COMMIT, model_volume, vllm_cache_volume
from experiments import modal_lanes as lanes

EXPECTED = "a6cd6ad5cee5374738f890c270ada4ca01f0946b"
OUT = Path(__file__).resolve().parent / "attempt2"
app = modal.App("m40204-scalar-integration-smoke")
smoke_image = image.add_local_dir(SOURCE_ROOT / "experiments", "/root/experiments", ignore=["**/__pycache__/**"])


@app.function(image=smoke_image, gpu="A10G", timeout=1200, retries=0,
              volumes={lanes.HF_HOME: model_volume, lanes.VLLM_CACHE_ROOT: vllm_cache_volume},
              env={"HF_HOME": lanes.HF_HOME, "HF_HUB_OFFLINE": "1",
                   "VLLM_CACHE_ROOT": lanes.VLLM_CACHE_ROOT,
                   "PYTHONPATH": "/root/project:/root"})
def smoke():
    root = Path("/tmp/m40204-smoke")
    root.mkdir(exist_ok=False)
    result = {"source_commit": SOURCE_COMMIT, "phases": {}, "files": {}}
    proc = None
    log = None

    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")

    def command(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return {"argv": args, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

    try:
        import torch
        import vllm
        env = {"uname": command(["uname", "-a"]), "os_release": Path("/etc/os-release").read_text(),
               "nvidia_smi": command(["nvidia-smi"]), "pip": command([sys.executable, "-m", "pip", "--version"]),
               "python_executable": sys.executable, "python": sys.version,
               "torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "cuda_available": torch.cuda.is_available(), "vllm": vllm.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        save("environment.json", env)
        assert vllm.__version__ == "0.30.0", "ENVIRONMENT MISMATCH"
        assert torch.cuda.is_available(), "ENVIRONMENT MISMATCH: CUDA unavailable"
        assert SOURCE_COMMIT == EXPECTED, "Unexpected source commit"
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        from src.server.adaptive_scheduler import AdaptiveAsyncScheduler
        from src.server.budget_planner import AdaptiveBudgetPlanner
        assert issubclass(AdaptiveAsyncScheduler, AsyncScheduler)
        api = {"issubclass": True, "installed_source": inspect.getfile(AsyncScheduler), "signatures": {}}
        for cls in (AsyncScheduler, Scheduler):
            save(cls.__name__ + "_source.json", {"source": inspect.getsource(cls)})
            for name in ("__init__", "schedule", "_update_after_schedule", "make_spec_decoding_stats", "_free_request"):
                api["signatures"][cls.__name__ + "." + name] = str(inspect.signature(getattr(cls, name)))
        save("import_api.json", api)
        hashes = {}
        for base in (Path("/root/project/src"), Path("/root/experiments")):
            for p in base.rglob("*.py"):
                hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        save("worker_source_hashes.json", hashes)
        runtime_env = lanes.lane_env("adaptive", "m40204-isolated-smoke")
        runtime_env["ADAPTIVE_TELEMETRY_PATH"] = str(root / "telemetry.jsonl")
        runtime_env["ADAPTIVE_STEP_TELEMETRY_PATH"] = str(root / "steps.jsonl")
        os.environ.update(runtime_env)
        planner = AdaptiveBudgetPlanner.from_env()
        from src.proposer.controller import AdaptiveDraftController
        assert planner.adapter.controller_cls is AdaptiveDraftController, "Tree controller configuration drift"
        argv = ["vllm", "serve", *lanes.build_vllm_args("adaptive", "l4-bf16")]
        save("deployment.json", {"argv": argv, "env": runtime_env, "profile": "l4-bf16", "gpu_request": "A10G",
                                 "controller": planner.adapter.controller_cls.__name__, "reason": "L4 probe queued with zero runners; supported functional fallback"})
        log = (root / "server.log").open("w")
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True)
        import requests
        deadline = time.monotonic() + 720
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited with code {proc.returncode}")
            try:
                ready = requests.get("http://127.0.0.1:8000/health", timeout=3).status_code == 200
            except requests.RequestException:
                pass
            if ready:
                break
            time.sleep(2)
        assert ready, "Health endpoint did not become ready within smoke limit"
        result["phases"]["A"] = "PASS"

        def request(label, text):
            payload = {"model": "qwen2.5-7b-adaptive", "messages": [{"role": "user", "content": text}],
                       "temperature": 0, "seed": 42, "max_tokens": 48}
            response = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=120)
            record = {"request": payload, "status": response.status_code, "response": response.text}
            save(label + ".json", record)
            response.raise_for_status()
            assert response.json()["choices"][0]["message"]["content"].strip(), "Empty output"
            return record

        request("single_request", "Count from one to twenty, separated by commas.")
        result["phases"]["B"] = "PASS"
        # Natural decisions only; do not force a second K or alter controller thresholds.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request, f"concurrent_{i}", text) for i, text in enumerate([
                "Write a short sentence about the ocean.", "Count down from twenty to one, separated by commas."])]
            for future in futures:
                future.result()
        result["phases"]["D"] = "REQUESTS_COMPLETED_ISOLATION_REQUIRES_TRACE_REVIEW"
        time.sleep(2)
        assert proc.poll() is None, "Engine died after requests"
        result["server_alive_after_requests"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        (root / "error.log").write_text(result["error"])
    finally:
        if proc is not None and proc.poll() is None:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        if log is not None:
            log.close()
        for p in root.iterdir():
            if p.is_file():
                result["files"][p.name] = p.read_text(errors="replace")
    return result


@app.local_entrypoint()
def main():
    assert SOURCE_COMMIT == EXPECTED
    OUT.mkdir(exist_ok=False)
    call = smoke.spawn()
    (OUT / "call.json").write_text(json.dumps({"call_id": call.object_id, "url": call.get_dashboard_url()}, indent=2))
    try:
        result = call.get(timeout=1200)
        for name, content in result.pop("files", {}).items():
            (OUT / name).write_text(content, encoding="utf-8")
        (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        call.cancel()
