"""[M3-00] Serverless vLLM runner on Modal.com (GPU L4).

Three authenticated vLLM servers (vanilla / static / adaptive) that mirror
scripts/launch_lane{1,2,3}_*.sh; lane flags live in experiments/modal_lanes.py.

Run from the repository root with the conda env ``doan-d22``:

    modal run -m experiments.modal_runner::image_check   # image + GPU versions
    modal run -m experiments.modal_runner::weights       # fill ptit-model-cache
    modal deploy -m experiments.modal_runner             # 3 lane servers

Resilient cells (lookup the deployed Function; collection needs no submit process):
    python -m experiments.modal_runner bench_submit --help
    python -m experiments.modal_runner bench_collect --call-id fc-...

Deploy-time switches (environment variables on the machine that deploys):
    PTIT_DTYPE_PROFILE   l4-bf16 (default) | fp16-fallback
    PTIT_RUN_ID          evidence folder name (default: UTC timestamp)
    PTIT_KEEP_WARM=1     keep one container per lane up (benchmarks only; costs credit)
    PTIT_SCALEDOWN_SECONDS  idle seconds before a container stops (default 300)
    PTIT_LANES           comma-separated lanes to deploy (default: all three)
    PTIT_GPU             L4 (default, reference) | A10G | T4 (T4 needs fp16-fallback)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from experiments import modal_lanes as lanes


APP_NAME = lanes.APP_NAME
GPU = lanes.resolve_gpu(os.environ)
REPO_ROOT = Path(__file__).resolve().parents[1]
if modal.is_local():
    from experiments.bench_source import snapshot

    _SOURCE_SNAPSHOT, SOURCE_COMMIT = snapshot(REPO_ROOT)
    SOURCE_ROOT = Path(_SOURCE_SNAPSHOT.name)
else:
    SOURCE_ROOT = REPO_ROOT
    SOURCE_COMMIT = os.environ.get("PTIT_SOURCE_COMMIT", "unknown")

PROFILE = os.environ.get("PTIT_DTYPE_PROFILE", lanes.DEFAULT_PROFILE)
RUN_ID = os.environ.get("PTIT_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
KEEP_WARM = os.environ.get("PTIT_KEEP_WARM") == "1"
SCALEDOWN_SECONDS = int(os.environ.get("PTIT_SCALEDOWN_SECONDS", "300"))
ENABLED_LANES = set(os.environ.get("PTIT_LANES", "vanilla,static,adaptive,eagle").split(","))
if unknown_lanes := ENABLED_LANES - lanes.LANES.keys():
    raise ValueError(f"Unknown deployment lanes: {sorted(unknown_lanes)}")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .uv_pip_install(f"vllm=={lanes.VLLM_VERSION}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "PTIT_SOURCE_COMMIT": SOURCE_COMMIT,
          "PTIT_RUN_ID": RUN_ID, "PTIT_LANES": ",".join(sorted(ENABLED_LANES)),
          "PTIT_DTYPE_PROFILE": PROFILE, "PTIT_KEEP_WARM": "1" if KEEP_WARM else "0",
          "PTIT_SCALEDOWN_SECONDS": str(SCALEDOWN_SECONDS)})
    .add_local_dir(SOURCE_ROOT / "src", f"{lanes.REMOTE_PROJECT_ROOT}/src", ignore=["**/__pycache__/**"])
    .add_local_file(SOURCE_ROOT / "experiments" / "modal_lanes.py", "/root/experiments/modal_lanes.py")
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
SERVER_ENV_FILE = "server_env.json"
RAW_FILE = "raw.jsonl"

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
    models = [
        (lanes.TARGET_MODEL, lanes.TARGET_REVISION),
        (lanes.DRAFT_MODEL, lanes.DRAFT_REVISION),
    ]
    if getattr(lanes, "EAGLE_DRAFT_MODEL", None):
        models.append((lanes.EAGLE_DRAFT_MODEL, lanes.EAGLE_REVISION))
    for repo, revision in models:
        if not revision:
            raise ValueError(f"Missing pinned revision for {repo}")
        sha = HfApi().model_info(repo, revision=revision).sha
        if sha != revision:
            raise ValueError(f"Revision mismatch for {repo}: requested {revision}, resolved {sha}")
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
            {"repo": repo, "revision": sha, "requested_revision": revision,
             "resolved_revision": sha, "already_present": already_present,
             "bytes": size, "main_ref_pinned": True}
        )

    # Written only after every model is complete: no manifest = not usable.
    (Path(lanes.HF_HOME) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model_volume.commit()
    return manifest


# --------------------------------------------------------------------------
# Lane servers
# --------------------------------------------------------------------------
def _write_server_env(lane: str, run_dir: Path) -> None:
    """[M3-01] Record the server's real environment for the remote-mode harness."""
    import pynvml
    import torch
    import vllm

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    driver = pynvml.nvmlSystemGetDriverVersion()
    env = lanes.build_server_env(
        gpu=name.decode() if isinstance(name, bytes) else str(name),
        driver_version=driver.decode() if isinstance(driver, bytes) else str(driver),
        gpu_memory_total_mb=round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2, 2),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        vllm_version=vllm.__version__,
        lane=lane,
        run_id=os.environ["PTIT_RUN_ID"],
        profile=os.environ["PTIT_DTYPE_PROFILE"],
    )
    env["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    env["source_commit"] = os.environ["PTIT_SOURCE_COMMIT"]
    import uuid
    env["instance_id"] = str(uuid.uuid4())
    env["modal_task_id"] = os.environ.get("MODAL_TASK_ID")
    env["vllm_args"] = lanes.build_vllm_args(lane, profile=os.environ["PTIT_DTYPE_PROFILE"])
    (run_dir / SERVER_ENV_FILE).write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    artifacts_volume.commit()


def _sample_vram(run_dir: Path, stop: threading.Event, interval_s: float = 0.25, commit_every_s: float = 10.0) -> None:
    """[M3-01] Append whole-GPU NVML memory samples ({ts_utc, used_mb, total_mb}) to vram.jsonl."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    last_commit = time.monotonic()
    with (run_dir / "vram.jsonl").open("a", encoding="utf-8") as out:
        while not stop.is_set():
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out.write(json.dumps({
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "used_mb": round(mem.used / 1024**2, 2),
                "total_mb": round(mem.total / 1024**2, 2),
            }) + "\n")
            out.flush()
            if time.monotonic() - last_commit >= commit_every_s:
                artifacts_volume.commit()
                last_commit = time.monotonic()
            stop.wait(interval_s)


def _start_vllm(lane: str) -> subprocess.Popen:
    run_dir = Path(lanes.server_artifact_dir(lane, os.environ["PTIT_RUN_ID"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_server_env(lane, run_dir)
    _VRAM_STOP.clear()
    threading.Thread(target=_sample_vram, args=(run_dir, _VRAM_STOP), daemon=True).start()
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


_VRAM_STOP = threading.Event()


def _stop_vllm(proc: subprocess.Popen) -> None:
    _VRAM_STOP.set()
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


if "eagle" in ENABLED_LANES:

    @app.server(**_server_options("eagle"))
    class EagleServer:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("eagle")

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


# --------------------------------------------------------------------------
# [M3-01] Benchmark from inside Modal (same region as the servers)
# --------------------------------------------------------------------------
BENCH_ROOT = "/root/bench"

bench_image = image.add_local_dir(
    SOURCE_ROOT / "experiments",
    f"{BENCH_ROOT}/experiments",
    ignore=["runs/**", "results/**", "notebooks/**", "**/__pycache__/**", "**/*.ipynb"],
)


@app.function(cpu=0.125, timeout=60)
def benchmark_identity() -> dict:
    """Cheap deployed-source handshake; never starts a GPU server."""
    return {"commit": os.environ["PTIT_SOURCE_COMMIT"], "run_id": RUN_ID,
            "lanes": sorted(ENABLED_LANES), "keep_warm": KEEP_WARM,
            "scaledown_seconds": SCALEDOWN_SECONDS}


def _lane_base_url(lane: str) -> str:
    workspace = os.environ.get("PTIT_MODAL_WORKSPACE", "nguyenduonghb940")
    return f"https://{workspace}--{APP_NAME}-{lane}.us-east.modal.direct/v1"


def _wait_server_ready(base_url: str, token: str, timeout_s: float = 1200.0) -> float:
    import urllib.request

    start = time.monotonic()
    health = base_url[: -len("/v1")] + "/health"
    while True:
        try:
            req = urllib.request.Request(health, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    return time.monotonic() - start
        except OSError:  # URLError is an OSError
            pass
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"{health} not ready after {timeout_s:.0f} s")
        time.sleep(10)


@app.function(
    image=bench_image,
    cpu=2,
    timeout=4 * 60 * 60,
    max_containers=1,
    volumes={lanes.ARTIFACTS_ROOT: artifacts_volume},
    secrets=[modal.Secret.from_name("ptit-proxy-token")],
)
def bench(lane: str, server_run_id: str, bench_run_id: str, cell: dict, git_commit: str) -> dict:
    """Run one benchmark_harness cell against a deployed lane and join server VRAM."""
    sys.path.insert(0, BENCH_ROOT)
    from experiments.modal_vram_join import join_vram, load_jsonl

    if os.environ.get("PTIT_SOURCE_COMMIT") != git_commit or RUN_ID != server_run_id:
        raise ValueError("Deployed benchmark source/run identity mismatch")
    token = f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}"
    base_url = _lane_base_url(lane)
    server_dir = Path(lanes.server_artifact_dir(lane, server_run_id))
    out_dir = Path(lanes.ARTIFACTS_ROOT) / "bench" / bench_run_id
    artifacts_volume.reload()
    out_dir.mkdir(parents=True, exist_ok=False)
    artifacts_volume.commit()  # durable reservation before any server request
    ready_s = _wait_server_ready(base_url, token)
    artifacts_volume.reload()
    server_before = json.loads((server_dir / SERVER_ENV_FILE).read_text(encoding="utf-8"))
    if (server_before.get("source_commit") != git_commit or server_before.get("run_id") != server_run_id
            or server_before.get("lane") != lane or not server_before.get("instance_id")):
        raise ValueError("Server source/run/instance mismatch")
    # A run/instance may serve only one scientific attempt, even after client loss.
    with (server_dir / "benchmark_claim.json").open("x", encoding="utf-8") as stream:
        json.dump({"bench_run_id": bench_run_id, "instance_id": server_before["instance_id"]}, stream)
    artifacts_volume.commit()
    import urllib.request
    request = urllib.request.Request(base_url + "/models", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        model_ids = [item["id"] for item in json.load(response)["data"]]
    if model_ids != [lanes.LANES[lane].served_model_name]:
        raise ValueError("Endpoint model identity mismatch")

    args = lanes.bench_harness_args(
        lane,
        base_url=base_url,
        server_env_json=str(server_dir / SERVER_ENV_FILE),
        raw_out=str(out_dir / RAW_FILE),
        summary_csv=str(out_dir / "harness_header.csv"),
        experiment_id=bench_run_id,
        **cell,
    )
    args.extend(["--warmup-policy", "m3-fixed3"])
    env = {**os.environ, "OPENAI_API_KEY": token, "PTIT_GIT_COMMIT": git_commit,
           "PTIT_SERVER_ENV_ROOT": lanes.ARTIFACTS_ROOT, "PYTHONIOENCODING": "utf-8"}
    commit_stop = threading.Event()

    def preserve_partial() -> None:
        while not commit_stop.wait(10):
            artifacts_volume.commit()

    committer = threading.Thread(target=preserve_partial, daemon=True)
    committer.start()
    try:
        with (out_dir / "harness.log").open("x", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-u", "-m", "experiments.benchmark_harness", *args],
                                  cwd=BENCH_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    finally:
        commit_stop.set()
        committer.join(timeout=30)
        artifacts_volume.commit()
    harness_log = (out_dir / "harness.log").read_text(encoding="utf-8")

    result = {"lane": lane, "returncode": proc.returncode, "server_ready_wait_s": round(ready_s, 2),
              "harness_log": harness_log, "raw": None, "joined": None,
              "server_env": None, "join_error": None,
              "server_run_id": server_run_id, "bench_run_id": bench_run_id,
              "cell": cell, "commit": git_commit, "endpoint_model_ids": model_ids,
              "call_id": modal.current_function_call_id()}
    result["server_env"] = server_before
    raw_path = out_dir / RAW_FILE
    warmup_path = out_dir / "warmup.jsonl"
    result["warmup"] = warmup_path.read_text(encoding="utf-8") if warmup_path.exists() else None
    if raw_path.exists():
        result["raw"] = raw_path.read_text(encoding="utf-8")
    if proc.returncode == 0 and raw_path.exists():
        time.sleep(12)  # let the server's periodic commit publish the last VRAM samples
        artifacts_volume.reload()
        final_server = json.loads((server_dir / SERVER_ENV_FILE).read_text(encoding="utf-8"))
        result["server_instance_unchanged"] = final_server == result["server_env"]
        try:
            joined = join_vram(load_jsonl(raw_path), load_jsonl(server_dir / "vram.jsonl"))
            result["joined"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in joined)
        except (OSError, ValueError, KeyError) as exc:
            result["join_error"] = str(exc)
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    # FunctionCall outputs have limited retention. Keep the full return bundle durably too.
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts_volume.commit()
    return result


@app.local_entrypoint()
def bench_cell(lane: str = "vanilla", server_run_id: str = "", workload: str = "W1",
               concurrency: int = 1, repeats: int = 1, num_requests: int = 8) -> None:
    """modal run -m experiments.modal_runner::bench_cell --server-run-id <deploy PTIT_RUN_ID> ..."""
    if not server_run_id:
        raise SystemExit("--server-run-id is required (the PTIT_RUN_ID used at modal deploy)")
    bench_run_id = f"{RUN_ID}-{lane}-{workload}-c{concurrency}"
    cell = {"workload": workload, "concurrency": concurrency, "repeats": repeats, "num_requests": num_requests}
    out = REPO_ROOT / "experiments" / "runs" / "m3_01" / bench_run_id
    out.mkdir(parents=True, exist_ok=False)
    commit = _git_commit()
    result = bench.remote(lane, server_run_id, bench_run_id, cell, commit)

    (out / "harness.log").write_text(result["harness_log"], encoding="utf-8")
    if result["server_env"]:
        (out / SERVER_ENV_FILE).write_text(json.dumps(result["server_env"], indent=2) + "\n", encoding="utf-8")
    if result["raw"]:
        (out / RAW_FILE).write_text(result["raw"], encoding="utf-8")
    if result["joined"]:
        (out / "raw.vram.jsonl").write_text(result["joined"], encoding="utf-8")
    summary = {k: result[k] for k in ("lane", "returncode", "server_ready_wait_s", "join_error")}
    summary.update({"bench_run_id": bench_run_id, "server_run_id": server_run_id, "cell": cell,
                    "commit": commit, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    (out / "bench.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(result["harness_log"][-2500:])
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


def _bench_evidence_root() -> Path:
    return REPO_ROOT / "experiments" / "runs" / "m3_01"


def _validate_submission(record: dict) -> None:
    for key in ("server_run_id", "bench_run_id"):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", record[key]):
            raise ValueError(f"Invalid {key}")
    if record.get("lane") not in lanes.LANES:
        raise ValueError("Invalid lane")
    if not re.fullmatch(r"[0-9a-f]{40}", record.get("commit", "")):
        raise ValueError("Submission requires a committed Git SHA")
    cell = record.get("cell", {})
    if cell.get("workload") not in lanes.BENCH_WORKLOADS:
        raise ValueError("Invalid workload")
    for key in ("concurrency", "repeats", "num_requests"):
        if type(cell.get(key)) is not int or cell[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not record.get("submitted_at_utc"):
        raise ValueError("Missing submission timestamp")


def _committed_bench_source() -> str:
    """Freeze the uploaded code/config, ignoring local run evidence and caches."""
    git = ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}"]
    source = ["src", "experiments", ":(exclude)experiments/runs", ":(exclude)experiments/results",
              ":(exclude)experiments/notebooks", ":(exclude)**/__pycache__/**", ":(exclude)**/*.ipynb"]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    changed = subprocess.check_output([*git, "diff", "--name-only", "HEAD", "--", *source],
                                      cwd=REPO_ROOT, text=True).strip()
    untracked = subprocess.check_output([*git, "ls-files", "--others", "--exclude-standard", "--", *source],
                                        cwd=REPO_ROOT, text=True).strip()
    if changed or untracked:
        raise RuntimeError("Commit uploaded source/config changes before benchmark submission")
    return commit


def _save_submission(path: Path, record: dict) -> None:
    # Atomic receipt replacement: a killed client leaves the previous complete receipt.
    import uuid
    temporary = path.with_name(f".submission-{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def bench_submit(lane: str = "vanilla", server_run_id: str = "", workload: str = "W1",
                 concurrency: int = 1, repeats: int = 1, num_requests: int = 8,
                 bench_run_id: str = "") -> str:
    """Submit exactly one cell from Python CLI; detach is automatic, never optional."""
    record = {
        "lane": lane, "server_run_id": server_run_id,
        "bench_run_id": bench_run_id or f"{RUN_ID}-{lane}-{workload}-c{concurrency}",
        "cell": {"workload": workload, "concurrency": concurrency,
                 "repeats": repeats, "num_requests": num_requests},
        "commit": _committed_bench_source(),
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(), "call_id": None,
    }
    _validate_submission(record)
    out = _bench_evidence_root() / record["bench_run_id"]
    out.mkdir(parents=True, exist_ok=False)
    receipt = out / "submission.json"
    _save_submission(receipt, record)
    identity = modal.Function.from_name(APP_NAME, "benchmark_identity").remote()
    if (identity.get("commit") != record["commit"] or identity.get("run_id") != server_run_id
            or identity.get("lanes") != [lane] or identity.get("keep_warm") is not False
            or identity.get("scaledown_seconds") != 60):
        raise ValueError("Deployed source/run/lifecycle mismatch; deploy the committed single lane first")
    bench_fn = modal.Function.from_name(APP_NAME, "bench")
    # Reserve before the RPC. A disconnect during spawn is ambiguous: never auto-retry.
    # Preserve this separate marker even when call_id is null (NOT_SUBMITTED receipt).
    _save_submission(out / "spawn_attempt.json", identity)
    call = bench_fn.spawn(lane, server_run_id, record["bench_run_id"], record["cell"], record["commit"])
    call_id = call.object_id
    if not isinstance(call_id, str) or not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise RuntimeError("Modal did not return a valid FunctionCall ID")
    # Print immediately as a second recovery channel if persisting the receipt fails.
    print(f"FunctionCall: {call_id}", flush=True)
    record["call_id"] = call_id
    _save_submission(receipt, record)
    print(f"Receipt: {receipt}", flush=True)
    return record["call_id"]


def bench_recover(bench_run_id: str, call_id: str) -> None:
    """Attach an orphaned *completed* call only after validating its full identity."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bench_run_id):
        raise ValueError("Invalid bench_run_id")
    if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise ValueError("Invalid FunctionCall ID")
    path = _bench_evidence_root() / bench_run_id / "submission.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    _validate_submission(record)
    if record["bench_run_id"] != bench_run_id or record.get("call_id") not in (None, call_id):
        raise ValueError("Conflicting recovery receipt")
    result = modal.FunctionCall.from_id(call_id).get(timeout=0)
    _validate_bench_result(result, record)
    record["call_id"] = call_id
    _save_submission(path, record)
    print(f"FunctionCall: {call_id}", flush=True)


def bench_status(bench_run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bench_run_id):
        raise ValueError("Invalid bench_run_id")
    out = _bench_evidence_root() / bench_run_id
    if not out.exists():
        return "NOT_SUBMITTED"
    record = json.loads((out / "submission.json").read_text(encoding="utf-8"))
    _validate_submission(record)
    if (out / "validation_failure.json").exists():
        return "FAILED"
    if not record.get("call_id"):
        return "NOT_SUBMITTED"
    handle = modal.FunctionCall.from_id(record["call_id"])
    from modal.exception import TimeoutError as ModalTimeoutError
    try:
        result = _bench_result(record["call_id"], record, timeout=0)
    except Exception as exc:
        graph = handle.get_call_graph()
        if any(node.status.name in ("FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT") for node in graph):
            return "FAILED"
        if isinstance(exc, (TimeoutError, ModalTimeoutError)):
            return "RUNNING" if any(node.task_id for node in graph) else "SUBMITTED"
        raise  # observation/network errors never justify resubmission
    if result.get("returncode") != 0:
        return "FAILED"
    try:
        _validate_bench_result(result, record)
    except (ValueError, RuntimeError):
        return "FAILED"
    return "COLLECTED" if (out / "bench.json").exists() else "COMPLETED"


def _bench_result(call_id: str, submission: dict, timeout: float | None = None) -> dict:
    from modal.exception import OutputExpiredError
    try:
        call = modal.FunctionCall.from_id(call_id)
        return call.get() if timeout is None else call.get(timeout=timeout)
    except OutputExpiredError:
        content = b"".join(artifacts_volume.read_file(f"bench/{submission['bench_run_id']}/result.json"))
        result = json.loads(content)
        if result.get("call_id") != call_id:
            raise ValueError("Archived result/FunctionCall identity mismatch")
        return result


def _validate_bench_result(result: dict, submission: dict) -> None:
    for key in ("lane", "server_run_id", "bench_run_id", "cell", "commit"):
        if result.get(key) != submission[key]:
            raise ValueError(f"Remote benchmark metadata mismatch: {key}")
    if result.get("returncode") != 0:
        raise RuntimeError(f"Remote harness failed: returncode={result.get('returncode')}; "
                           f"artifacts: /bench/{submission['bench_run_id']}")
    server = result.get("server_env") or {}
    if server.get("lane") != submission["lane"] or server.get("run_id") != submission["server_run_id"]:
        raise ValueError("Remote server environment lane/run mismatch")
    expected = {"experiment_id": submission["bench_run_id"], "lane": submission["lane"],
                "commit": submission["commit"], "workload": submission["cell"]["workload"],
                "concurrency": submission["cell"]["concurrency"]}
    for artifact in ("raw", "joined"):
        content = result.get(artifact)
        if artifact == "joined" and content is None:
            continue
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Missing {artifact} artifact")
        for line in content.splitlines():
            row = json.loads(line)
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{artifact} artifact metadata mismatch")


def bench_collect(call_id: str) -> Path:
    """Recover a submitted call without starting an app or submitting another cell."""
    if not isinstance(call_id, str) or not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise ValueError("A valid FunctionCall ID (fc-...) is required")
    matches = []
    for path in _bench_evidence_root().glob("*/submission.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("call_id") == call_id:
            matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one local submission receipt for {call_id}; found {len(matches)}")
    receipt, submission = matches[0]
    _validate_submission(submission)
    if receipt.parent.name != submission["bench_run_id"]:
        raise ValueError("Submission receipt/run directory mismatch")
    # No cancel(), even if retrieval fails or the collector is interrupted.
    result = _bench_result(call_id, submission)
    _validate_bench_result(result, submission)
    summary = {key: result[key] for key in ("lane", "returncode", "server_ready_wait_s", "join_error")}
    summary.update({key: submission[key] for key in (
        "bench_run_id", "server_run_id", "cell", "commit", "call_id", "submitted_at_utc")})
    summary["timestamp_utc"] = result["timestamp_utc"]
    summary["server_instance_unchanged"] = result.get("server_instance_unchanged", False)
    summary["endpoint_model_ids"] = result.get("endpoint_model_ids")
    files = {"harness.log": result["harness_log"],
             SERVER_ENV_FILE: json.dumps(result["server_env"], indent=2) + "\n",
             RAW_FILE: result["raw"]}
    if result.get("warmup") is not None:
        files["warmup.jsonl"] = result["warmup"]
    if result.get("joined") is not None:
        files["raw.vram.jsonl"] = result["joined"]
    files["bench.json"] = json.dumps(summary, indent=2) + "\n"
    out = receipt.parent
    # Retry collection safely after local interruption; never replace conflicting evidence.
    for name, content in files.items():
        path = out / name
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Conflicting existing evidence: {path}")
    for name, content in files.items():
        path = out / name
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        else:
            print(f"Reusing identical evidence: {path}")
    print(f"Collected {call_id}: {out}")
    return out


# --------------------------------------------------------------------------
# [M3-02] Forward cost of target / draft on L4 (one container per model)
# --------------------------------------------------------------------------
profile_image = image.add_local_file(
    REPO_ROOT / "experiments" / "modal_profiler.py", "/root/experiments/modal_profiler.py"
)


def _kv_capacity_tokens(llm) -> int | None:
    try:
        cache = llm.llm_engine.vllm_config.cache_config
        return int(cache.num_gpu_blocks) * int(cache.block_size)
    except (AttributeError, TypeError, ValueError):
        return None


@app.function(
    image=profile_image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes=SERVER_VOLUMES,
    env={"HF_HOME": lanes.HF_HOME, "HF_HUB_OFFLINE": "1", "VLLM_CACHE_ROOT": lanes.VLLM_CACHE_ROOT},
)
def profile_forward(model: str, run_id: str, contexts: list[int], batches: list[int],
                    warmup: int, repeats: int, decode_tokens: int) -> dict:
    """Measure step_ms for every (L, B) cell of one model with vLLM offline LLM."""
    import pynvml
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from experiments import modal_profiler as prof

    kwargs = prof.engine_kwargs(model)
    started = time.monotonic()
    llm = LLM(**kwargs)
    load_s = time.monotonic() - started
    capacity = _kv_capacity_tokens(llm)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    env = {
        "model": model, "engine_kwargs": kwargs, "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        "gpu_memory_total_mb": round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**20, 1),
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda, "kv_capacity_tokens": capacity,
        "engine_load_s": round(load_s, 2), "run_id": run_id,
    }
    print(json.dumps({"event": "engine_ready", **env}), flush=True)

    def generate(prompts: list[list[int]], max_tokens: int) -> None:
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)

    records = []
    out_dir = Path(lanes.ARTIFACTS_ROOT) / "profile" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{model}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for length, batch in prof.grid(contexts, batches):
            if not prof.fits_kv(length, batch, decode_tokens, capacity):
                cell = [{"model": model, "context_len": length, "batch_size": batch, "status": "skipped",
                         "reason": f"B*(L+N+1)={batch * (length + decode_tokens + 1)} > kv {capacity}",
                         "step_ms": None}]
            else:
                cell = prof.measure_cell(generate, model, length, batch, warmup=warmup, repeats=repeats,
                                         decode_tokens=decode_tokens)
            for record in cell:
                raw.write(json.dumps(record) + "\n")
            raw.flush()
            records.extend(cell)
            medians = [r["step_ms"] for r in cell if r["step_ms"] is not None]
            print(json.dumps({"event": "cell", "model": model, "L": length, "B": batch,
                              "step_ms": sorted(medians)[len(medians) // 2] if medians else None}), flush=True)
            artifacts_volume.commit()
    (out_dir / f"{model}.env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    artifacts_volume.commit()
    return {"env": env, "records": records}


@app.local_entrypoint()
def profile_costs(run_id: str = "", models: str = "target,draft", contexts: str = "", batches: str = "",
                  warmup: int = 3, repeats: int = 5, decode_tokens: int = 64) -> None:
    """modal run -m experiments.modal_runner::profile_costs --run-id <id> [--contexts 256 --batches 1]"""
    from experiments import modal_profiler as prof

    run_id = run_id or RUN_ID
    out = prof.evidence_dir(run_id)
    ctx = prof.parse_int_list(contexts, prof.CONTEXT_LENGTHS)
    bs = prof.parse_int_list(batches, prof.BATCH_SIZES)
    selected = [m for m in models.split(",") if m]
    for model in selected:
        prof.engine_kwargs(model)  # reject unknown names before spending GPU time

    calls = {m: profile_forward.spawn(m, run_id, list(ctx), list(bs), warmup, repeats, decode_tokens)
             for m in selected}
    results = {m: call.get() for m, call in calls.items()}

    out.mkdir(parents=True, exist_ok=True)
    records = [r for m in selected for r in results[m]["records"]]
    (out / "forward_costs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    meta = {"run_id": run_id, "commit": _git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "grid": {"contexts": list(ctx), "batches": list(bs), "warmup": warmup, "repeats": repeats,
                     "decode_tokens": decode_tokens},
            "method": "step_ms = (T(1+N) - T(1)) / N, vLLM offline LLM, prefix caching off",
            "models": {m: results[m]["env"] for m in selected}}
    (out / "env.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    summary = {**meta, "cells": prof.summarize(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    full_grid = ctx == prof.CONTEXT_LENGTHS and bs == prof.BATCH_SIZES and set(selected) == set(prof.MODELS)
    if full_grid and repeats >= prof.REPEATS:
        results_path = REPO_ROOT / "experiments" / "results" / "m3_02_forward_costs_l4.json"
        results_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {results_path}")
    for cell in summary["cells"]:
        print(cell["model"], cell["context_len"], cell["batch_size"], cell["status"],
              cell["step_ms_median"], cell["variance_flag"])
    print(f"Wrote {out}")


def _bench_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Submit/recover one detached Modal benchmark cell")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("bench_submit", help="Spawn one cell against the deployed Function")
    submit.add_argument("--lane", choices=tuple(lanes.LANES), default="vanilla")
    submit.add_argument("--server-run-id", required=True)
    submit.add_argument("--bench-run-id", default="")
    submit.add_argument("--workload", choices=tuple(lanes.BENCH_WORKLOADS), default="W1")
    submit.add_argument("--concurrency", type=int, default=1)
    submit.add_argument("--repeats", type=int, default=1)
    submit.add_argument("--num-requests", type=int, default=8)
    collect = commands.add_parser("bench_collect", help="Recover an existing FunctionCall; never resubmit")
    collect.add_argument("--call-id", required=True)
    status = commands.add_parser("bench_status", help="Read remote state without submitting")
    status.add_argument("--bench-run-id", required=True)
    recover = commands.add_parser("bench_recover", help="Recover orphaned completed call; never spawn")
    recover.add_argument("--bench-run-id", required=True)
    recover.add_argument("--call-id", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "bench_submit":
        bench_submit(**args)
    elif command == "bench_collect":
        bench_collect(**args)
    elif command == "bench_recover":
        bench_recover(**args)
    else:
        print(bench_status(**args))


if modal.is_local() and "deploy" in sys.argv[1:3]:
    _committed_bench_source()


if __name__ == "__main__":
    _bench_cli()
