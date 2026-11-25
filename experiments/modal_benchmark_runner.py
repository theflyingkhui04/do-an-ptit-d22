"""Dedicated Modal app (`flexsds-benchmark`) for multi-GPU 3-node 3-lane benchmarks.

Isolated from M3/M5 (`ptit-vllm`), web demo (`ptit-vllm-demo`), and interactive serving
(`flexsds-serving`), while sharing the `ptit-model-cache`, `ptit-vllm-cache`, and
`ptit-run-artifacts` volumes. Works on any authenticated Modal workspace without hardcoded
workspace names or custom proxy-token secrets.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from experiments import modal_lanes as lanes
from experiments import vllm_runtime_patch as runtime_patch
from flexsds.core.hardware import DEFAULT_MODAL_GPU, MODAL_GPU_NAMES, HardwareInfo, matches_gpu_label, modal_gpu_resource
from flexsds.server.local_runner import LocalServingRunner

APP_NAME = os.environ.get("FLEXSDS_BENCH_APP_NAME", "flexsds-benchmark")
ARTIFACTS_ROOT = lanes.ARTIFACTS_ROOT
SERVER_ENV_FILE = "server_env.json"
RAW_FILE = "raw.jsonl"
DEFAULT_PORT = 8000

_encoded_cfg = os.environ.get("FLEXSDS_BENCH_CONFIG_B64")
BENCH_CONFIG: dict[str, Any] = (
    json.loads(base64.b64decode(_encoded_cfg, validate=True).decode("utf-8"))
    if _encoded_cfg
    else {
        "target_gpu": DEFAULT_MODAL_GPU,
        "lane": "vanilla",
        "port": DEFAULT_PORT,
        "max_model_len": 4096,
        "vllm_runtime": "backport-55708",
        "run_id": os.environ.get("FLEXSDS_BENCH_RUN_ID", "unconfigured"),
        "source_commit": os.environ.get("FLEXSDS_BENCH_COMMIT", "unknown"),
    }
)

GPU = str(BENCH_CONFIG.get("target_gpu") or DEFAULT_MODAL_GPU).upper()
if GPU not in MODAL_GPU_NAMES:
    raise ValueError(f"Unsupported Modal benchmark GPU {GPU!r}; expected one of {MODAL_GPU_NAMES}")
MODAL_GPU = modal_gpu_resource(GPU)
PORT = int(BENCH_CONFIG.get("port", DEFAULT_PORT))
if not 1 <= PORT <= 65535:
    raise ValueError(f"Invalid benchmark serving port: {PORT}")
RUN_ID = str(os.environ.get("FLEXSDS_BENCH_RUN_ID") or BENCH_CONFIG.get("run_id") or "unconfigured")
SOURCE_COMMIT = str(BENCH_CONFIG.get("source_commit") or os.environ.get("FLEXSDS_BENCH_COMMIT") or "unknown")


def server_artifact_dir(gpu: str, lane: str, run_id: str) -> str:
    return f"{ARTIFACTS_ROOT}/flexsds_bench/{gpu}/{lane}/{run_id}"


def build_harness_args(
    *,
    lane: str,
    gpu: str,
    base_url: str,
    served_model_name: str,
    server_env_json: str,
    raw_out: str,
    summary_csv: str,
    experiment_id: str,
    workload: str,
    concurrency: int,
    repeats: int,
    num_requests: int,
    warmup_requests: int = 3,
) -> list[str]:
    """Build experiments.benchmark_harness argv for one multi-GPU FlexSDS cell."""
    workload_path = lanes.workload_file(workload)
    serving = lanes.workload_serving(workload)
    timeout = [] if serving["timeout_s"] is None else ["--timeout-s", str(serving["timeout_s"])]
    env_profile = lanes.environment_profile_for_gpu(gpu)
    return [
        "--lane", lane,
        "--workload", workload,
        "--endpoint", f"{base_url.rstrip('/')}/chat/completions",
        "--model", served_model_name,
        "--workload-file", workload_path,
        "--sample-policy", "cycle-rotate",
        "--concurrency", str(concurrency),
        "--repeats", str(repeats),
        "--num-requests", str(num_requests),
        "--warmup-requests", str(warmup_requests),
        "--warmup-policy", "m3-fixed3",
        "--experiment-id", experiment_id,
        "--raw-out", raw_out,
        "--summary-csv", summary_csv,
        "--environment-profile", env_profile,
        "--server-env-json", server_env_json,
        *timeout,
    ]


def _committed_bench_source() -> str:
    git = ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}"]
    source = [
        "src",
        "experiments",
        ":(exclude)experiments/runs",
        ":(exclude)experiments/results",
        ":(exclude)experiments/notebooks",
        ":(exclude)**/__pycache__/**",
        ":(exclude)**/*.ipynb",
    ]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    changed = subprocess.check_output([*git, "diff", "--name-only", "HEAD", "--", *source], cwd=REPO_ROOT, text=True).strip()
    untracked = subprocess.check_output(
        [*git, "ls-files", "--others", "--exclude-standard", "--", *source], cwd=REPO_ROOT, text=True
    ).strip()
    if changed or untracked:
        raise RuntimeError("Commit uploaded source/config changes before benchmark submission")
    return commit


def _validate_submission(record: dict[str, Any]) -> None:
    for key in ("server_run_id", "bench_run_id"):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", record[key]):
            raise ValueError(f"Invalid {key}")
    if record.get("gpu") not in MODAL_GPU_NAMES:
        raise ValueError("Invalid GPU")
    if record.get("lane") not in ("vanilla", "static", "adaptive"):
        raise ValueError("Invalid lane")
    if not re.fullmatch(r"[0-9a-f]{40}", record.get("commit", "")):
        raise ValueError("Submission requires a committed Git SHA")
    cell = record.get("cell", {})
    if cell.get("workload") not in lanes.ALL_WORKLOADS:
        raise ValueError("Invalid workload")
    for key in ("concurrency", "repeats", "num_requests"):
        if type(cell.get(key)) is not int or cell[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not record.get("submitted_at_utc"):
        raise ValueError("Missing submission timestamp")


def _save_submission(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(f".submission-{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_bench_result(result: dict[str, Any], submission: dict[str, Any]) -> None:
    for key in ("gpu", "lane", "server_run_id", "bench_run_id", "cell", "commit"):
        if result.get(key) != submission[key]:
            raise ValueError(f"Remote benchmark metadata mismatch: {key}")
    if result.get("returncode") != 0:
        raise RuntimeError(
            f"Remote harness failed: returncode={result.get('returncode')}; "
            f"artifacts: /flexsds_bench/bench/{submission['bench_run_id']}"
        )
    server = result.get("server_env") or {}
    if server.get("lane") != submission["lane"] or server.get("run_id") != submission["server_run_id"]:
        raise ValueError("Remote server environment lane/run mismatch")
    expected = {
        "experiment_id": submission["bench_run_id"],
        "lane": submission["lane"],
        "commit": submission["commit"],
        "workload": submission["cell"]["workload"],
        "concurrency": submission["cell"]["concurrency"],
    }
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


def _wait_server_ready(base_url: str, api_key: str | None = None, timeout_s: float = 1200.0) -> float:
    import urllib.request

    start = time.monotonic()
    health = base_url[: -len("/v1")] + "/health"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while True:
        try:
            req = urllib.request.Request(health, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    return time.monotonic() - start
        except OSError:
            pass
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"{health} not ready after {timeout_s:.0f} s")
        time.sleep(10)


try:
    import modal
except ImportError:
    modal = None


if modal is not None:
    from flexsds.server.modal_image import serving_image

    container_config = dict(BENCH_CONFIG)
    if container_config.get("profile_path"):
        container_config["profile_path"] = "/root/project/hardware_profile.json"

    _base_image = (
        serving_image()
        .uv_pip_install("aiohttp", "nvidia-ml-py")
        .add_local_file(
            str(REPO_ROOT / "experiments" / "vllm_runtime_patch.py"),
            "/root/vllm_runtime_patch.py",
            copy=True,
        )
        .run_commands("python /root/vllm_runtime_patch.py")
        .env(
            {
                "PYTHONPATH": "/root/project:/root/project/src",
                "HF_HOME": lanes.HF_HOME,
                "VLLM_CACHE_ROOT": lanes.VLLM_CACHE_ROOT,
                "FLEXSDS_BENCH_CONFIG_B64": base64.b64encode(
                    json.dumps(container_config).encode("utf-8")
                ).decode("ascii"),
                "FLEXSDS_BENCH_RUN_ID": RUN_ID,
                "FLEXSDS_BENCH_COMMIT": SOURCE_COMMIT,
            }
        )
    )
    if modal.is_local():
        _base_image = _base_image.add_local_dir(
            str(REPO_ROOT / "src"),
            remote_path="/root/project/src",
            ignore=["**/__pycache__/**"],
        ).add_local_dir(
            str(REPO_ROOT / "experiments"),
            remote_path="/root/project/experiments",
            ignore=["runs/**", "results/**", "notebooks/**", "**/__pycache__/**", "**/*.ipynb"],
        )
        selected_profile = BENCH_CONFIG.get("profile_path")
        if BENCH_CONFIG.get("enable_adaptive") and selected_profile:
            profile_file = Path(selected_profile)
            if not profile_file.is_file():
                raise ValueError(f"Selected hardware profile artifact is missing: {selected_profile}")
            _base_image = _base_image.add_local_file(
                str(profile_file),
                remote_path="/root/project/hardware_profile.json",
            )

    model_volume = modal.Volume.from_name("ptit-model-cache", create_if_missing=True)
    vllm_cache_volume = modal.Volume.from_name("ptit-vllm-cache", create_if_missing=True)
    artifacts_volume = modal.Volume.from_name("ptit-run-artifacts", create_if_missing=True)
    SERVER_VOLUMES = {
        lanes.HF_HOME: model_volume,
        lanes.VLLM_CACHE_ROOT: vllm_cache_volume,
        ARTIFACTS_ROOT: artifacts_volume,
    }

    app = modal.App(APP_NAME)
    _VRAM_STOP = threading.Event()

    def _sample_vram(run_dir: Path, stop: threading.Event, interval_s: float = 0.25, commit_every_s: float = 10.0) -> None:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        last_commit = time.monotonic()
        with (run_dir / "vram.jsonl").open("a", encoding="utf-8") as out:
            while not stop.is_set():
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                out.write(
                    json.dumps(
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "used_mb": round(mem.used / 1024**2, 2),
                            "total_mb": round(mem.total / 1024**2, 2),
                        }
                    )
                    + "\n"
                )
                out.flush()
                if time.monotonic() - last_commit >= commit_every_s:
                    artifacts_volume.commit()
                    last_commit = time.monotonic()
                stop.wait(interval_s)

    @app.function(image=_base_image, cpu=0.125, timeout=60)
    def benchmark_identity() -> dict[str, Any]:
        return {
            "commit": SOURCE_COMMIT,
            "run_id": RUN_ID,
            "gpu": GPU,
            "modal_gpu": MODAL_GPU,
            "lane": BENCH_CONFIG.get("lane", "vanilla"),
            "max_model_len": int(BENCH_CONFIG.get("max_model_len", 4096)),
            "vllm_runtime": BENCH_CONFIG.get("vllm_runtime", "backport-55708"),
        }

    @app.function(
        image=_base_image,
        gpu=MODAL_GPU,
        max_containers=1,
        min_containers=0,
        scaledown_window=60,
        timeout=3600,
        volumes=SERVER_VOLUMES,
    )
    @modal.web_server(PORT, startup_timeout=900)
    def serve() -> None:
        import pynvml
        import torch
        import vllm
        from flexsds.core.cost_model import validate_measured_profile

        runtime_cfg = dict(BENCH_CONFIG)
        if runtime_cfg.get("profile_path"):
            runtime_cfg["profile_path"] = "/root/project/hardware_profile.json"

        actual_device = torch.cuda.get_device_name(0)
        if not matches_gpu_label(MODAL_GPU, actual_device):
            raise ValueError(f"Actual Modal GPU mismatch: expected {MODAL_GPU}, got {actual_device}")

        profile_sha256 = None
        if runtime_cfg.get("enable_adaptive", False):
            sm = "sm_%d%d" % torch.cuda.get_device_capability(0)
            data = json.loads(Path(runtime_cfg["profile_path"]).read_text(encoding="utf-8"))
            if not matches_gpu_label(data["metadata"]["device"], actual_device):
                raise ValueError("Profile differs from actual Modal GPU")
            hardware = HardwareInfo(
                device_name=f"Remote {GPU}",
                sm_architecture=sm,
                total_vram_mb=0,
                free_vram_mb=0,
                cuda_version=torch.version.cuda,
                is_cuda_available=True,
                gpu_match_label=MODAL_GPU,
            )
            validate_measured_profile(
                data,
                hardware,
                runtime_cfg.get("target_model"),
                runtime_cfg.get("draft_model"),
            )
            profile_sha256 = data.get("metadata", {}).get("raw_sha256")

        runner = LocalServingRunner()
        cmd = runner.build_command(runtime_cfg, port=PORT)
        env = runner.setup_environment(runtime_cfg)

        lane = str(runtime_cfg.get("lane", "vanilla"))
        run_dir = Path(server_artifact_dir(GPU, lane, RUN_ID))
        run_dir.mkdir(parents=True, exist_ok=True)

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        driver = pynvml.nvmlSystemGetDriverVersion()
        server_env = {
            "gpu": actual_device,
            "requested_gpu": GPU,
            "modal_gpu": MODAL_GPU,
            "driver_version": driver.decode() if isinstance(driver, bytes) else str(driver),
            "gpu_memory_total_mb": round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2, 2),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "vllm_version": vllm.__version__,
            "dtype": runtime_cfg.get("dtype"),
            "quantization": runtime_cfg.get("quantization"),
            "lane": lane,
            "run_id": RUN_ID,
            "instance_id": str(uuid.uuid4()),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": SOURCE_COMMIT,
            "target_model": runtime_cfg.get("target_model"),
            "draft_model": runtime_cfg.get("draft_model"),
            "served_model_name": runtime_cfg.get("served_model_name"),
            "max_model_len": int(runtime_cfg.get("max_model_len", 4096)),
            "runtime_variant": runtime_patch.RUNTIME_LABEL,
            "vllm_outputs_sha256": hashlib.sha256(
                (Path(vllm.__file__).parent / "outputs.py").read_bytes()
            ).hexdigest(),
            "vllm_args": cmd,
            "profile_sha256": profile_sha256,
        }
        (run_dir / SERVER_ENV_FILE).write_text(json.dumps(server_env, indent=2) + "\n", encoding="utf-8")
        artifacts_volume.commit()

        _VRAM_STOP.clear()
        threading.Thread(target=_sample_vram, args=(run_dir, _VRAM_STOP), daemon=True).start()
        subprocess.Popen(cmd, env=env)

    @app.function(
        image=_base_image,
        cpu=2,
        timeout=4 * 60 * 60,
        max_containers=1,
        volumes={ARTIFACTS_ROOT: artifacts_volume},
    )
    def bench(
        gpu: str,
        lane: str,
        server_run_id: str,
        bench_run_id: str,
        cell: dict[str, Any],
        git_commit: str,
        served_model_name: str,
    ) -> dict[str, Any]:
        import urllib.request
        from experiments.modal_vram_join import join_vram, load_jsonl

        if SOURCE_COMMIT != git_commit or RUN_ID != server_run_id or GPU != gpu:
            raise ValueError("Deployed benchmark source/gpu/run identity mismatch")

        serve_fn = modal.Function.from_name(APP_NAME, "serve")
        web_url = serve_fn.get_web_url()
        if not web_url:
            raise RuntimeError(f"Modal app {APP_NAME!r} serve function has no web URL")
        base_url = web_url.rstrip("/") + "/v1"
        api_key = os.environ.get("FLEXSDS_BENCH_API_KEY")

        server_dir = Path(server_artifact_dir(gpu, lane, server_run_id))
        out_dir = Path(ARTIFACTS_ROOT) / "flexsds_bench" / "bench" / bench_run_id
        artifacts_volume.reload()
        out_dir.mkdir(parents=True, exist_ok=False)
        artifacts_volume.commit()

        ready_s = _wait_server_ready(base_url, api_key=api_key)
        artifacts_volume.reload()
        server_before = json.loads((server_dir / SERVER_ENV_FILE).read_text(encoding="utf-8"))
        if (
            server_before.get("source_commit") != git_commit
            or server_before.get("run_id") != server_run_id
            or server_before.get("lane") != lane
            or not server_before.get("instance_id")
        ):
            raise ValueError("Server source/run/instance mismatch")
        with (server_dir / "benchmark_claim.json").open("x", encoding="utf-8") as stream:
            json.dump({"bench_run_id": bench_run_id, "instance_id": server_before["instance_id"]}, stream)
        artifacts_volume.commit()

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        req = urllib.request.Request(base_url + "/models", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            model_ids = [item["id"] for item in json.load(response)["data"]]
        if model_ids != [served_model_name]:
            raise ValueError(f"Endpoint model identity mismatch: expected [{served_model_name}], got {model_ids}")

        args = build_harness_args(
            lane=lane,
            gpu=gpu,
            base_url=base_url,
            served_model_name=served_model_name,
            server_env_json=str(server_dir / SERVER_ENV_FILE),
            raw_out=str(out_dir / RAW_FILE),
            summary_csv=str(out_dir / "harness_header.csv"),
            experiment_id=bench_run_id,
            **cell,
        )
        env = {
            **os.environ,
            "PTIT_GIT_COMMIT": git_commit,
            "PTIT_SERVER_ENV_ROOT": ARTIFACTS_ROOT,
            "PYTHONIOENCODING": "utf-8",
        }
        if api_key:
            env["OPENAI_API_KEY"] = api_key

        commit_stop = threading.Event()

        def preserve_partial() -> None:
            while not commit_stop.wait(10):
                artifacts_volume.commit()

        committer = threading.Thread(target=preserve_partial, daemon=True)
        committer.start()
        try:
            with (out_dir / "harness.log").open("x", encoding="utf-8") as log:
                proc = subprocess.run(
                    [sys.executable, "-u", "-m", "experiments.benchmark_harness", *args],
                    cwd="/root/project",
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        finally:
            commit_stop.set()
            committer.join(timeout=30)
            artifacts_volume.commit()

        harness_log = (out_dir / "harness.log").read_text(encoding="utf-8")
        result: dict[str, Any] = {
            "gpu": gpu,
            "lane": lane,
            "returncode": proc.returncode,
            "server_ready_wait_s": round(ready_s, 2),
            "harness_log": harness_log,
            "raw": None,
            "joined": None,
            "server_env": server_before,
            "join_error": None,
            "server_run_id": server_run_id,
            "bench_run_id": bench_run_id,
            "cell": cell,
            "commit": git_commit,
            "endpoint_model_ids": model_ids,
            "call_id": modal.current_function_call_id(),
        }
        raw_path = out_dir / RAW_FILE
        warmup_path = out_dir / "warmup.jsonl"
        result["warmup"] = warmup_path.read_text(encoding="utf-8") if warmup_path.exists() else None
        if raw_path.exists():
            result["raw"] = raw_path.read_text(encoding="utf-8")
        if proc.returncode == 0 and raw_path.exists():
            time.sleep(12)
            artifacts_volume.reload()
            final_server = json.loads((server_dir / SERVER_ENV_FILE).read_text(encoding="utf-8"))
            result["server_instance_unchanged"] = final_server == server_before
            try:
                joined = join_vram(load_jsonl(raw_path), load_jsonl(server_dir / "vram.jsonl"))
                result["joined"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in joined)
            except (OSError, ValueError, KeyError) as exc:
                result["join_error"] = str(exc)
        result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        artifacts_volume.commit()
        return result
else:
    app = None
    artifacts_volume = None


def _require_modal() -> Any:
    if modal is None:
        raise RuntimeError("Modal Python SDK is required for remote benchmark operations")
    return modal


def stop_app() -> None:
    apps = json.loads(
        subprocess.check_output([sys.executable, "-m", "modal", "app", "list", "--json"], cwd=REPO_ROOT, text=True)
    )
    for item in apps:
        if item.get("description") == APP_NAME and item.get("state") != "stopped":
            subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", item["app_id"], "-y"],
                cwd=REPO_ROOT,
                check=True,
            )


def bench_submit(
    *,
    gpu: str,
    lane: str,
    workload: str,
    concurrency: int,
    repeats: int,
    num_requests: int,
    server_run_id: str,
    bench_run_id: str,
    evidence_root: Path,
    serving_config: dict[str, Any],
) -> str:
    m = _require_modal()
    record = {
        "gpu": gpu,
        "lane": lane,
        "server_run_id": server_run_id,
        "bench_run_id": bench_run_id,
        "cell": {
            "workload": workload,
            "concurrency": concurrency,
            "repeats": repeats,
            "num_requests": num_requests,
        },
        "commit": _committed_bench_source(),
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "call_id": None,
    }
    _validate_submission(record)
    out = evidence_root / bench_run_id
    out.mkdir(parents=True, exist_ok=False)
    receipt = out / "submission.json"
    _save_submission(receipt, record)

    identity = m.Function.from_name(APP_NAME, "benchmark_identity").remote()
    if (
        identity.get("commit") != record["commit"]
        or identity.get("run_id") != server_run_id
        or identity.get("gpu") != gpu
        or identity.get("lane") != lane
        or identity.get("max_model_len") != serving_config["max_model_len"]
        or identity.get("vllm_runtime") != "backport-55708"
    ):
        raise ValueError("Deployed benchmark identity mismatch; redeploy for this cell")

    bench_fn = m.Function.from_name(APP_NAME, "bench")
    _save_submission(out / "spawn_attempt.json", identity)
    call = bench_fn.spawn(
        gpu,
        lane,
        server_run_id,
        bench_run_id,
        record["cell"],
        record["commit"],
        serving_config["served_model_name"],
    )
    call_id = call.object_id
    if not isinstance(call_id, str) or not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise RuntimeError("Modal did not return a valid FunctionCall ID")
    print(f"FunctionCall: {call_id}", flush=True)
    record["call_id"] = call_id
    _save_submission(receipt, record)
    print(f"Receipt: {receipt}", flush=True)
    return call_id


def _bench_result(call_id: str, submission: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    m = _require_modal()
    try:
        call = m.FunctionCall.from_id(call_id)
        return call.get() if timeout is None else call.get(timeout=timeout)
    except Exception as exc:
        if type(exc).__name__ != "OutputExpiredError" or artifacts_volume is None:
            raise
        content = b"".join(artifacts_volume.read_file(f"flexsds_bench/bench/{submission['bench_run_id']}/result.json"))
        result = json.loads(content)
        if result.get("call_id") != call_id:
            raise ValueError("Archived result/FunctionCall identity mismatch")
        return result


def bench_status(bench_run_id: str, *, evidence_root: Path) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bench_run_id):
        raise ValueError("Invalid bench_run_id")
    out = evidence_root / bench_run_id
    if not out.exists():
        return "NOT_SUBMITTED"
    record = json.loads((out / "submission.json").read_text(encoding="utf-8"))
    _validate_submission(record)
    if (out / "validation_failure.json").exists():
        return "FAILED"
    if not record.get("call_id"):
        return "NOT_SUBMITTED"
    m = _require_modal()
    handle = m.FunctionCall.from_id(record["call_id"])
    try:
        result = _bench_result(record["call_id"], record, timeout=0)
    except Exception as exc:
        graph = handle.get_call_graph()
        if any(node.status.name in ("FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT") for node in graph):
            return "FAILED"
        if isinstance(exc, TimeoutError) or type(exc).__name__ == "TimeoutError":
            return "RUNNING" if any(node.task_id for node in graph) else "SUBMITTED"
        raise
    if result.get("returncode") != 0:
        return "FAILED"
    try:
        _validate_bench_result(result, record)
    except (ValueError, RuntimeError):
        return "FAILED"
    return "COLLECTED" if (out / "bench.json").exists() else "COMPLETED"


def bench_collect(call_id: str, *, evidence_root: Path) -> Path:
    if not isinstance(call_id, str) or not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise ValueError("A valid FunctionCall ID (fc-...) is required")
    matches = []
    for path in evidence_root.glob("*/submission.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("call_id") == call_id:
            matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one local submission receipt for {call_id}; found {len(matches)}")
    receipt, submission = matches[0]
    _validate_submission(submission)
    if receipt.parent.name != submission["bench_run_id"]:
        raise ValueError("Submission receipt/run directory mismatch")

    result = _bench_result(call_id, submission)
    _validate_bench_result(result, submission)
    summary = {key: result[key] for key in ("gpu", "lane", "returncode", "server_ready_wait_s", "join_error")}
    summary.update(
        {
            key: submission[key]
            for key in ("bench_run_id", "server_run_id", "cell", "commit", "call_id", "submitted_at_utc")
        }
    )
    summary["timestamp_utc"] = result["timestamp_utc"]
    summary["server_instance_unchanged"] = result.get("server_instance_unchanged", False)
    summary["endpoint_model_ids"] = result.get("endpoint_model_ids")

    files: dict[str, str] = {
        "harness.log": result["harness_log"],
        SERVER_ENV_FILE: json.dumps(result["server_env"], indent=2) + "\n",
        RAW_FILE: result["raw"],
        "bench.json": json.dumps(summary, indent=2) + "\n",
    }
    if result.get("warmup") is not None:
        files["warmup.jsonl"] = result["warmup"]
    if result.get("joined") is not None:
        files["raw.vram.jsonl"] = result["joined"]

    out = receipt.parent
    for name, content in files.items():
        path = out / name
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Conflicting existing evidence: {path}")
    for name, content in files.items():
        path = out / name
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    print(f"Collected {call_id}: {out}")
    return out


def bench_recover(bench_run_id: str, call_id: str, *, evidence_root: Path) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bench_run_id):
        raise ValueError("Invalid bench_run_id")
    if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
        raise ValueError("Invalid FunctionCall ID")
    m = _require_modal()
    path = evidence_root / bench_run_id / "submission.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    _validate_submission(record)
    if record["bench_run_id"] != bench_run_id or record.get("call_id") not in (None, call_id):
        raise ValueError("Conflicting recovery receipt")
    result = m.FunctionCall.from_id(call_id).get(timeout=0)
    _validate_bench_result(result, record)
    record["call_id"] = call_id
    _save_submission(path, record)
    print(f"FunctionCall: {call_id}", flush=True)
