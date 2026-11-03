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
import time
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
ENABLED_LANES = lanes.enabled_lanes(os.environ.get("PTIT_LANES", ""), local=modal.is_local())

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
    env.update(_gpu_limits(pynvml, handle))
    (run_dir / SERVER_ENV_FILE).write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    artifacts_volume.commit()


def _gpu_limits(pynvml, handle) -> dict:
    """The host's enforced power limit and max SM clock (L4 hosts may differ)."""
    try:
        return {
            "power_limit_w": round(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000, 1),
            "sm_clock_max_mhz": pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM),
        }
    except pynvml.NVMLError:
        return {}


def _gpu_state(pynvml, handle) -> dict:
    """SM clock, board power and temperature; empty if the driver does not report them."""
    try:
        return {
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1),
            "temp_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        }
    except pynvml.NVMLError:
        return {}


def _sample_vram(run_dir: Path, stop: threading.Event, interval_s: float = 0.25, commit_every_s: float = 10.0) -> None:
    """[M3-01] Append whole-GPU NVML samples ({ts_utc, used_mb, total_mb} plus sm_clock_mhz,
    power_w, temp_c) to vram.jsonl."""
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
                **_gpu_state(pynvml, handle),
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


if "adaptive-f1" in ENABLED_LANES:

    @app.server(**_server_options("adaptive-f1"))
    class AdaptiveF1Server:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("adaptive-f1")

        @modal.exit()
        def stop(self) -> None:
            _stop_vllm(self.proc)


if "adaptive-f12" in ENABLED_LANES:

    @app.server(**_server_options("adaptive-f12"))
    class AdaptiveF12Server:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("adaptive-f12")

        @modal.exit()
        def stop(self) -> None:
            _stop_vllm(self.proc)


if "adaptive-v1" in ENABLED_LANES:

    @app.server(**_server_options("adaptive-v1"))
    class AdaptiveV1Server:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("adaptive-v1")

        @modal.exit()
        def stop(self) -> None:
            _stop_vllm(self.proc)


if "adaptive-v2" in ENABLED_LANES:

    @app.server(**_server_options("adaptive-v2"))
    class AdaptiveV2Server:
        @modal.enter()
        def start(self) -> None:
            self.proc = _start_vllm("adaptive-v2")

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
    REPO_ROOT / "experiments",
    f"{BENCH_ROOT}/experiments",
    ignore=["runs/**", "results/**", "notebooks/**", "**/__pycache__/**", "**/*.ipynb"],
)


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
    volumes={lanes.ARTIFACTS_ROOT: artifacts_volume},
    secrets=[modal.Secret.from_name("ptit-proxy-token")],
)
def bench(lane: str, server_run_id: str, bench_run_id: str, cell: dict, git_commit: str) -> dict:
    """Run one benchmark_harness cell against a deployed lane and join server VRAM."""
    return _run_bench(lane, server_run_id, bench_run_id, cell, git_commit)


def _run_bench(lane: str, server_run_id: str, bench_run_id: str, cell: dict, git_commit: str,
               ready_timeout_s: float = 1200.0) -> dict:
    sys.path.insert(0, BENCH_ROOT)
    from experiments.modal_vram_join import join_vram, load_jsonl

    token = f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}"
    base_url = _lane_base_url(lane)
    ready_s = _wait_server_ready(base_url, token, timeout_s=ready_timeout_s)

    server_dir = Path(lanes.server_artifact_dir(lane, server_run_id))
    out_dir = Path(lanes.ARTIFACTS_ROOT) / "bench" / bench_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_volume.reload()

    args = lanes.bench_harness_args(
        lane,
        base_url=base_url,
        server_env_json=str(server_dir / SERVER_ENV_FILE),
        raw_out=str(out_dir / RAW_FILE),
        summary_csv=str(out_dir / "harness_header.csv"),
        experiment_id=bench_run_id,
        **cell,
    )
    env = {**os.environ, "OPENAI_API_KEY": token, "PTIT_GIT_COMMIT": git_commit,
           "PTIT_SERVER_ENV_ROOT": lanes.ARTIFACTS_ROOT, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-m", "experiments.benchmark_harness", *args],
                          cwd=BENCH_ROOT, env=env, capture_output=True, text=True)
    (out_dir / "harness.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")

    result = {"lane": lane, "returncode": proc.returncode, "server_ready_wait_s": round(ready_s, 2),
              "harness_log": proc.stdout[-6000:] + proc.stderr[-4000:], "raw": None, "joined": None,
              "server_env": None, "join_error": None}
    if (server_dir / SERVER_ENV_FILE).exists():
        result["server_env"] = json.loads((server_dir / SERVER_ENV_FILE).read_text(encoding="utf-8"))
    raw_path = out_dir / RAW_FILE
    if proc.returncode == 0 and raw_path.exists():
        result["raw"] = raw_path.read_text(encoding="utf-8")
        time.sleep(12)  # let the server's periodic commit publish the last VRAM samples
        artifacts_volume.reload()
        try:
            joined = join_vram(load_jsonl(raw_path), load_jsonl(server_dir / "vram.jsonl"))
            result["joined"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in joined)
        except (OSError, ValueError, KeyError) as exc:
            result["join_error"] = str(exc)
    return result


@app.local_entrypoint()
def bench_cell(lane: str = "vanilla", server_run_id: str = "", workload: str = "W1",
               concurrency: int = 1, repeats: int = 1, num_requests: int = 8) -> None:
    """modal run -m experiments.modal_runner::bench_cell --server-run-id <deploy PTIT_RUN_ID> ..."""
    if not server_run_id:
        raise SystemExit("--server-run-id is required (the PTIT_RUN_ID used at modal deploy)")
    bench_run_id = f"{RUN_ID}-{lane}-{workload}-c{concurrency}"
    cell = {"workload": workload, "concurrency": concurrency, "repeats": repeats, "num_requests": num_requests}
    result = bench.remote(lane, server_run_id, bench_run_id, cell, _git_commit())

    out = REPO_ROOT / "experiments" / "runs" / "m3_01" / bench_run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "harness.log").write_text(result["harness_log"], encoding="utf-8")
    if result["server_env"]:
        (out / SERVER_ENV_FILE).write_text(json.dumps(result["server_env"], indent=2) + "\n", encoding="utf-8")
    if result["raw"]:
        (out / RAW_FILE).write_text(result["raw"], encoding="utf-8")
    if result["joined"]:
        (out / "raw.vram.jsonl").write_text(result["joined"], encoding="utf-8")
    summary = {k: result[k] for k in ("lane", "returncode", "server_ready_wait_s", "join_error")}
    summary.update({"bench_run_id": bench_run_id, "server_run_id": server_run_id, "cell": cell,
                    "commit": _git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    (out / "bench.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(result["harness_log"][-2500:])
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


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


# --------------------------------------------------------------------------
# [Adaptive v1] Ablation: lanes in parallel, cells sequential within a lane
# --------------------------------------------------------------------------
@app.function(
    image=bench_image,
    cpu=2,
    timeout=12 * 60 * 60,
    volumes={lanes.ARTIFACTS_ROOT: artifacts_volume},
    secrets=[modal.Secret.from_name("ptit-proxy-token")],
)
def bench_series(lane: str, server_run_id: str, campaign_id: str, cells: list[dict], git_commit: str,
                 ready_timeout_s: float = 2700.0) -> dict:
    """All cells of one lane in order, then that lane's adaptive telemetry (if any)."""
    sys.path.insert(0, BENCH_ROOT)
    from experiments import ablation_v1 as ab

    def run_one(cell: dict) -> dict:
        bench_run_id = f"{campaign_id}-{lane}-{cell['workload']}-c{cell['concurrency']}"
        result = _run_bench(lane, server_run_id, bench_run_id, cell, git_commit, ready_timeout_s)
        print(json.dumps({"event": "cell_done", "lane": lane, **cell, "returncode": result["returncode"],
                          "join_error": result["join_error"]}), flush=True)
        return result

    results = ab.run_cells(cells, run_one)
    telemetry = None
    path = lanes.lane_env(lane, server_run_id).get("ADAPTIVE_TELEMETRY_PATH")
    if path:
        time.sleep(12)
        artifacts_volume.reload()
        if Path(path).exists():
            telemetry = Path(path).read_text(encoding="utf-8")
    return {"lane": lane, "results": results, "telemetry": telemetry}


def _write_cell(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if result.get("error"):
        error = {"cell": result["cell"], "error": result["error"]}
        (out / "bench.json").write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
        return
    (out / "harness.log").write_text(result["harness_log"], encoding="utf-8")
    if result["server_env"]:
        (out / SERVER_ENV_FILE).write_text(json.dumps(result["server_env"], indent=2) + "\n", encoding="utf-8")
    if result["raw"]:
        (out / RAW_FILE).write_text(result["raw"], encoding="utf-8")
    if result["joined"]:
        (out / "raw.vram.jsonl").write_text(result["joined"], encoding="utf-8")
    meta = {k: result[k] for k in ("lane", "returncode", "server_ready_wait_s", "join_error", "cell")}
    (out / "bench.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _summarize_cell_tolerant(analyze_m2, raw: Path, harness_lane: str) -> dict:
    """M2-04 cell aggregation over the cell's last complete attempt; spec-lane records whose
    server returned no per-request metrics (vLLM "metrics": null) are excluded and counted
    instead of aborting."""
    import tempfile

    from experiments import ablation_v1 as ab

    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    current = ab.latest_attempt(records)
    missing = [r for r in current if harness_lane in analyze_m2.SPEC_LANES
               and r.get("phase") == "measured" and not isinstance(r.get("server_metrics"), dict)]
    kept = [r for r in current if r not in missing]
    with tempfile.TemporaryDirectory() as tmp:
        filtered = Path(tmp) / raw.name
        filtered.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        row = analyze_m2._summarize_cell(filtered, harness_lane)
    row = {**row, "n_measured": sum(r.get("phase") == "measured" for r in kept),
           "stale_records_dropped": len(records) - len(current), "spec_metrics_missing": len(missing)}
    if missing:
        row["spec_metrics_missing_samples"] = sorted({r.get("sample_id") for r in missing})
    return row


def _summarize_ablation(root: Path, lane_names: list[str], cells: list[dict]) -> dict:
    from experiments import ablation_v1 as ab
    from experiments import analyze_m2

    rows, telemetry = [], {}
    for lane in lane_names:
        harness_lane = lanes.ALL_LANES[lane].harness_lane or lane
        windows = {}
        for cell in cells:
            raw = root / lane / f"{cell['workload']}-c{cell['concurrency']}" / "raw.vram.jsonl"
            if not raw.exists():
                continue
            row = _summarize_cell_tolerant(analyze_m2, raw, harness_lane)
            rows.append({"lane": lane, "workload": cell["workload"], "concurrency": cell["concurrency"],
                         **{k: v for k, v in row.items() if not k.startswith("_") and k not in ("lane", "workload", "concurrency")}})
            records = ab.latest_attempt(
                [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()])
            windows[(cell["workload"], cell["concurrency"])] = ab.cell_window(records)
        tel_path = root / lane / "telemetry.jsonl"
        if tel_path.exists():
            lines = [json.loads(line) for line in tel_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            telemetry[lane] = {f"{w}-c{c}": ab.telemetry_summary(part)
                               for (w, c), part in ab.split_telemetry(lines, windows).items()}
    rows = ab.add_speedups(rows)
    return {"rows": rows, "criteria": ab.evaluate_criteria(rows), "telemetry": telemetry}


@app.local_entrypoint()
def ablation(server_run_id: str = "", campaign_id: str = "", lanes_csv: str = "", workloads: str = "",
             concurrency: str = "", repeats: int = 3, num_requests: int = 16, summary_only: bool = False) -> None:
    """modal run -m experiments.modal_runner::ablation --server-run-id <id> --campaign-id <id>

    Run lanes in batches (--lanes-csv), then --summary-only to aggregate every lane on disk."""
    from experiments import ablation_v1 as ab
    from experiments.modal_profiler import parse_int_list

    if not server_run_id:
        raise SystemExit("--server-run-id is required (the PTIT_RUN_ID used at modal deploy)")
    campaign_id = campaign_id or RUN_ID
    root = ab.campaign_dir(campaign_id)
    lane_names = [x for x in (lanes_csv or ",".join(ab.LANES)).split(",") if x]
    for lane in lane_names:
        lanes._lane(lane)
    cells = ab.plan_cells(tuple(x for x in (workloads or ",".join(ab.WORKLOADS)).split(",") if x),
                          parse_int_list(concurrency, ab.CONCURRENCY), repeats, num_requests)

    commit = _git_commit()
    if summary_only:
        lane_names = [lane for lane in ab.LANES if (root / lane).is_dir()]
        calls = {}
    else:
        calls = {lane: bench_series.spawn(lane, server_run_id, campaign_id, cells, commit, ab.READY_TIMEOUT_S)
                 for lane in lane_names}
    for lane, call in calls.items():
        try:
            out = call.get()
        except Exception as exc:  # keep the lanes that finished
            print(f"[{lane}] FAILED: {type(exc).__name__}: {exc}")
            continue
        for result in out["results"]:
            cell = result["cell"]
            _write_cell(root / lane / f"{cell['workload']}-c{cell['concurrency']}", result)
        if out["telemetry"]:
            tel_path = root / lane / "telemetry.jsonl"
            old = tel_path.read_text(encoding="utf-8") if tel_path.exists() else ""
            tel_path.write_text(ab.merge_jsonl(old, out["telemetry"]), encoding="utf-8")
        print(f"[{lane}] {sum(r['returncode'] == 0 for r in out['results'])}/{len(out['results'])} cells ok")

    summary = {"campaign_id": campaign_id, "server_run_id": server_run_id, "commit": commit,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "lanes": lane_names, "cells": cells,
               **_summarize_ablation(root, lane_names, cells)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for r in summary["rows"]:
        print(f"{r['lane']:13} {r['workload']} c{r['concurrency']:<3} tpot {r['tpot_p50']:7.2f} "
              f"vsV {r['speedup_vs_vanilla'] or 0:5.3f} vsS {r['speed_vs_static'] or 0:5.3f}")
    print(json.dumps(summary["criteria"], indent=2))
    print(f"Wrote {root}")
