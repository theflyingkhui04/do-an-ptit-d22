"""Pinned-runtime validation using the canonical Modal Lane 3 launcher."""
import json
from pathlib import Path

import modal
from experiments.modal_runner import (image, SOURCE_COMMIT, GPU, SERVER_VOLUMES,
                                      lanes, RUN_ID, PROFILE)

app = modal.App("ptit-async-lineage-preflight", image=image)


@app.function(timeout=600)
def probe():
    import importlib.metadata
    import platform
    import sys
    from datetime import datetime, timezone

    sys.path.insert(0, "/root/project")
    from experiments.scalar_pinned_sources import manifest, read_source

    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  hostname=platform.node(), provider="Modal CPU preflight",
                  git_sha=SOURCE_COMMIT, python=sys.version,
                  vllm=importlib.metadata.version("vllm"),
                  torch=importlib.metadata.version("torch"), files=[])
    for record in manifest()["files"]:
        try:
            read_source(record["path"], installed_only=True)
            result["files"].append(dict(**record, status="PASS"))
        except Exception as error:
            result["files"].append(dict(**record, status="FAIL", error=str(error)))
    result["source_gate"] = ("PASS" if result["vllm"] == "0.30.0"
                             and all(r["status"] == "PASS" for r in result["files"])
                             else "FAIL")
    return result


@app.function(gpu=GPU, volumes=SERVER_VOLUMES,
              env=lanes.server_env("adaptive", profile=PROFILE, run_id=RUN_ID), timeout=1200)
def integration():
    import os
    import subprocess
    import time
    import traceback
    import urllib.request
    from experiments.modal_runner import _start_vllm, _stop_vllm

    root = Path(lanes.server_artifact_dir("adaptive", RUN_ID))
    root.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (root / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    process = None
    try:
        sources = probe.local()
        save("source-gate.json", sources)
        if sources["source_gate"] != "PASS":
            raise RuntimeError("BLOCKED — PINNED SOURCE MISMATCH")
        os.environ.update(ADAPTIVE_STARTUP_TELEMETRY_PATH=str(root / "startup.jsonl"),
                          ADAPTIVE_STEP_TELEMETRY_PATH=str(root / "steps.jsonl"))
        save("launch.json", dict(git_sha=SOURCE_COMMIT, run_id=RUN_ID,
             command=["vllm", "serve", *lanes.build_vllm_args("adaptive", profile=PROFILE)],
             env={k: v for k, v in os.environ.items() if k.startswith("ADAPTIVE_")
                  or k in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_USE_V2_MODEL_RUNNER")},
             harness_pid=os.getpid(), model_revision=lanes.TARGET_REVISION,
             draft_revision=lanes.DRAFT_REVISION))
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True)
        (root / "gpu.txt").write_text(gpu.stdout, encoding="utf-8")
        process = _start_vllm("adaptive")
        save("server-process.json", dict(pid=process.pid, parent_pid=os.getpid()))
        deadline = time.monotonic() + 720
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited before readiness: {process.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("Server readiness deadline exceeded")
        save("readiness.json", dict(status="PASS", server_pid=process.pid))
        payload = dict(model=lanes.LANES["adaptive"].served_model_name,
                       messages=[dict(role="user", content="Count from 1 to 50, separated by commas.")],
                       temperature=0, max_tokens=64, stream=False)
        save("request.json", payload)
        request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                   data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            save("response.json", json.load(response))
        time.sleep(2)
        save("outcome.json", dict(status="COLLECTED", git_sha=SOURCE_COMMIT))
    except Exception:
        save("failure.json", dict(error=traceback.format_exc(), git_sha=SOURCE_COMMIT))
    finally:
        if process is not None:
            _stop_vllm(process)
    return {p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in root.iterdir() if p.is_file()}


@app.local_entrypoint()
def main(output: str, stage: str = "preflight"):
    if stage == "preflight":
        result = probe.remote()
        Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif stage == "integration":
        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        for name, content in integration.remote().items():
            (root / name).write_text(content, encoding="utf-8")
        print("Collected runtime evidence: " + str(root))
    else:
        raise ValueError("stage must be preflight or integration")
