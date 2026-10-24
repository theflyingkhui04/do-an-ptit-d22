"""[M3-00] Serverless vLLM runner on Modal.com (GPU L4).

Three authenticated vLLM servers (vanilla / static / adaptive) that mirror
scripts/launch_lane{1,2,3}_*.sh; lane flags live in experiments/modal_lanes.py.

Run from the repository root with the conda env ``doan-d22``:

    modal run -m experiments.modal_runner::image_check   # image + GPU versions
    modal run -m experiments.modal_runner::weights       # fill ptit-model-cache
    modal deploy -m experiments.modal_runner             # 3 lane servers

Deploy-time switches (environment variables on the machine that deploys):
    PTIT_DTYPE_PROFILE   l4-bf16 (default) | fp16-fallback
    PTIT_RUN_ID          evidence folder name (default: UTC timestamp)
    PTIT_KEEP_WARM=1     keep one container per lane up (benchmarks only; costs credit)
    PTIT_SCALEDOWN_SECONDS  idle seconds before a container stops (default 300)
    PTIT_LANES           comma-separated lanes to deploy (default: all three)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import modal

from experiments import modal_lanes as lanes


APP_NAME = lanes.APP_NAME
GPU = "L4"
REPO_ROOT = Path(__file__).resolve().parents[1]

PROFILE = os.environ.get("PTIT_DTYPE_PROFILE", lanes.DEFAULT_PROFILE)
RUN_ID = os.environ.get("PTIT_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
KEEP_WARM = os.environ.get("PTIT_KEEP_WARM") == "1"
SCALEDOWN_SECONDS = int(os.environ.get("PTIT_SCALEDOWN_SECONDS", "300"))
ENABLED_LANES = set(os.environ.get("PTIT_LANES", "vanilla,static,adaptive").split(","))

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .uv_pip_install(f"vllm=={lanes.VLLM_VERSION}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(REPO_ROOT / "src", f"{lanes.REMOTE_PROJECT_ROOT}/src", ignore=["**/__pycache__/**"])
    .add_local_file(REPO_ROOT / "experiments" / "modal_lanes.py", "/root/experiments/modal_lanes.py")
)

model_volume = modal.Volume.from_name("ptit-model-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("ptit-vllm-cache", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("ptit-run-artifacts", create_if_missing=True)
SERVER_VOLUMES = {
    lanes.HF_HOME: model_volume,
    lanes.VLLM_CACHE_ROOT: vllm_cache_volume,
    lanes.ARTIFACTS_ROOT: artifacts_volume,
}
MANIFEST_NAME = "ptit_weights_manifest.json"

app = modal.App(APP_NAME, image=image)


# --------------------------------------------------------------------------
# Image check and weights
# --------------------------------------------------------------------------
@app.function(gpu=GPU, timeout=15 * 60)
def versions() -> dict:
    import torch
    import vllm

    return {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version.split()[0],
    }


@app.function(volumes={lanes.HF_HOME: model_volume}, env={"HF_HOME": lanes.HF_HOME}, timeout=60 * 60)
def download_weights() -> dict:
    from huggingface_hub import HfApi, snapshot_download

    manifest = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "models": []}
    for repo, revision in (
        (lanes.TARGET_MODEL, lanes.TARGET_REVISION),
        (lanes.DRAFT_MODEL, lanes.DRAFT_REVISION),
    ):
        sha = HfApi().model_info(repo, revision=revision).sha
        try:
            snapshot_download(repo, revision=sha, local_files_only=True)
            already_present = True
        except Exception:
            already_present = False
        path = Path(snapshot_download(repo, revision=sha))
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        ref = lanes.pin_main_ref(Path(lanes.HF_HOME) / "hub", repo, sha)
        print(f"{repo}@{sha}: {'already present' if already_present else 'downloaded'} ({size} bytes); {ref} -> {sha}")
        manifest["models"].append(
            {"repo": repo, "revision": sha, "already_present": already_present, "bytes": size, "main_ref_pinned": True}
        )

    # Written only after every model is complete: no manifest = not usable.
    (Path(lanes.HF_HOME) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model_volume.commit()
    return manifest


# --------------------------------------------------------------------------
# Lane servers
# --------------------------------------------------------------------------
def _start_vllm(lane: str) -> subprocess.Popen:
    run_dir = Path(lanes.ARTIFACTS_ROOT) / lane / os.environ["PTIT_RUN_ID"]
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry = os.environ.get("ADAPTIVE_TELEMETRY_PATH")
    if telemetry:
        Path(telemetry).parent.mkdir(parents=True, exist_ok=True)

    cmd = ["vllm", "serve", *lanes.build_vllm_args(lane, profile=os.environ["PTIT_DTYPE_PROFILE"])]
    print(json.dumps({"event": "vllm_start", "lane": lane, "cmd": cmd}), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump() -> None:
        with (run_dir / "vllm.log").open("a", encoding="utf-8") as log:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()

    threading.Thread(target=pump, daemon=True).start()
    return proc


def _stop_vllm(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    artifacts_volume.commit()


def _server_options(lane: str) -> dict:
    return {
        "name": lane,
        "gpu": GPU,
        "env": lanes.server_env(lane, profile=PROFILE, run_id=RUN_ID),
        "volumes": SERVER_VOLUMES,
        "port": lanes.DEFAULT_PORT,
        "startup_timeout": 15 * 60,
        "scaledown_window": SCALEDOWN_SECONDS,
        "min_containers": 1 if KEEP_WARM else 0,
        # One container per lane: measurements must not be split across replicas.
        "max_containers": 1,
        "target_concurrency": 32,
    }


if "vanilla" in ENABLED_LANES:

    @app.server(**_server_options("vanilla"))
    class VanillaServer:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("vanilla")

        @modal.exit()
        def stop(self) -> None:
            _stop_vllm(self.proc)


if "static" in ENABLED_LANES:

    @app.server(**_server_options("static"))
    class StaticServer:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("static")

        @modal.exit()
        def stop(self) -> None:
            _stop_vllm(self.proc)


if "adaptive" in ENABLED_LANES:

    @app.server(**_server_options("adaptive"))
    class AdaptiveServer:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("adaptive")

        @modal.exit()
        def stop(self) -> None:
            _stop_vllm(self.proc)


# --------------------------------------------------------------------------
# Local entrypoints (write evidence under experiments/runs/m3_00/<run-id>/)
# --------------------------------------------------------------------------
def _evidence_dir() -> Path:
    path = REPO_ROOT / "experiments" / "runs" / "m3_00" / RUN_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def image_check() -> None:
    info = {**versions.remote(), "commit": _git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    out = _evidence_dir() / "image.json"
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Wrote {out}")


@app.local_entrypoint()
def weights() -> None:
    manifest = download_weights.remote()
    out = _evidence_dir() / "weights_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {out}")
