"""Modal app `flexsds-benchmark`: one container per GPU for the multi-GPU 3-node 3-lane benchmark.

Design: openspec/changes/flexsds-bench-one-container-per-gpu/design.md. Each GPU tier has its own Modal
function (`run_gpu_<GPU>`). Inside, `experiments.bench_gpu_session.run_session` starts vLLM per
(workload, lane) on 127.0.0.1:8000 and runs the canonical harness against it, so requests never pass Modal's
web proxy (which serves a web_server container one request at a time). Card identity is checked before any
model is loaded; a wrong card (e.g. NVIDIA A10G for A10) returns in seconds and the laptop asks again.

Laptop side: `submit_gpu` (spawn with wrong-card retries and a per-GPU lock), `gpu_call_state`,
`collect_gpu` (volume -> attempts/<attempt_id>/), `release_locks`, `stop_app`. On machines where Modal only
lives in the `modal` CLI's interpreter (pipx), the same actions run through `--sdk-action`.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from experiments import modal_lanes as lanes
from experiments import vllm_runtime_patch as runtime_patch
from experiments.bench_gpu_session import WrongGpu, build_harness_args  # noqa: F401 (re-exported)
from flexsds.core.hardware import MODAL_GPU_NAMES, modal_gpu_resource

APP_NAME = os.environ.get("FLEXSDS_BENCH_APP_NAME", "flexsds-benchmark")
LOCK_NAME = os.environ.get("FLEXSDS_BENCH_LOCK_NAME", "flexsds-benchmark-lock")
ARTIFACTS_ROOT = lanes.ARTIFACTS_ROOT
RESULTS_PREFIX = "flexsds_bench"
GPU_TIMEOUT_S = 20 * 3600
SOURCE_PATHS = ("src", "experiments")
SOURCE_EXCLUDES = (
    ":(exclude)experiments/runs", ":(exclude)experiments/results", ":(exclude)experiments/notebooks",
    ":(exclude)**/__pycache__/**", ":(exclude)**/*.ipynb",
)


def function_name(gpu: str) -> str:
    if gpu not in MODAL_GPU_NAMES:
        raise ValueError(f"Unsupported Modal benchmark GPU {gpu!r}; expected one of {MODAL_GPU_NAMES}")
    return f"run_gpu_{gpu}"


def lock_key(campaign_id: str, gpu: str) -> str:
    return f"{campaign_id}:{gpu}"


def volume_dir(campaign_id: str, gpu: str) -> str:
    return f"{RESULTS_PREFIX}/{campaign_id}/{gpu}"


# --------------------------------------------------------------------------------------------------
# Source identity: a committed SHA, or (explicitly allowed) a digest of the uploaded working tree.
# --------------------------------------------------------------------------------------------------
def _git(*args: str) -> str:
    git = ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}"]
    return subprocess.check_output([*git, *args], cwd=REPO_ROOT, text=True).strip()


def source_digest(repo_root: Path = REPO_ROOT) -> str:
    """SHA-256 over path + bytes of every uploaded source file (tracked or new, not ignored)."""
    listed = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root.as_posix()}", "ls-files", "--cached", "--others",
         "--exclude-standard", "--", *SOURCE_PATHS, *SOURCE_EXCLUDES], cwd=repo_root, text=True)
    digest = hashlib.sha256()
    for rel in sorted(set(line for line in listed.splitlines() if line.strip())):
        path = repo_root / rel
        if path.is_file():
            digest.update(rel.encode("utf-8") + b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def committed_bench_source(*, allow_uncommitted: bool = False) -> tuple[str, str | None]:
    """(HEAD SHA, None) for a clean tree; (HEAD SHA, working-tree digest) only when explicitly allowed."""
    commit = _git("rev-parse", "HEAD")
    changed = _git("diff", "--name-only", "HEAD", "--", *SOURCE_PATHS, *SOURCE_EXCLUDES)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS, *SOURCE_EXCLUDES)
    if changed or untracked:
        if not allow_uncommitted:
            raise RuntimeError("Commit uploaded source/config changes before benchmark submission "
                               "(or pass --allow-uncommitted to record a source digest instead)")
        return commit, source_digest()
    return commit, None


def _committed_bench_source() -> str:
    """Backward-compatible strict check used by older callers."""
    return committed_bench_source()[0]


try:
    import modal
except ImportError:
    modal = None


if modal is not None:
    from flexsds.server.modal_image import serving_image

    _image = (
        serving_image()
        .uv_pip_install("aiohttp", "nvidia-ml-py")
        .add_local_file(str(REPO_ROOT / "experiments" / "vllm_runtime_patch.py"), "/root/vllm_runtime_patch.py",
                        copy=True)
        .run_commands("python /root/vllm_runtime_patch.py")
        .env({"PYTHONPATH": "/root/project:/root/project/src", "HF_HOME": lanes.HF_HOME,
              "VLLM_CACHE_ROOT": lanes.VLLM_CACHE_ROOT, "VLLM_USE_V2_MODEL_RUNNER": "0",
              "PYTHONIOENCODING": "utf-8"})
    )
    if modal.is_local():
        _image = _image.add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/project/src",
                                      ignore=["**/__pycache__/**"]).add_local_dir(
            str(REPO_ROOT / "experiments"), remote_path="/root/project/experiments",
            ignore=["runs/**", "results/**", "notebooks/**", "**/__pycache__/**", "**/*.ipynb"])

    model_volume = modal.Volume.from_name("ptit-model-cache", create_if_missing=True)
    vllm_cache_volume = modal.Volume.from_name("ptit-vllm-cache", create_if_missing=True)
    artifacts_volume = modal.Volume.from_name("ptit-run-artifacts", create_if_missing=True)
    VOLUMES = {lanes.HF_HOME: model_volume, lanes.VLLM_CACHE_ROOT: vllm_cache_volume,
               ARTIFACTS_ROOT: artifacts_volume}
    app = modal.App(APP_NAME)
    _COMMON = dict(image=_image, max_containers=1, scaledown_window=2, timeout=GPU_TIMEOUT_S, volumes=VOLUMES)

    @app.function(gpu=modal_gpu_resource("T4"), **_COMMON)
    def run_gpu_T4(record_json: str, cells_json: str, profile_text: str | None) -> dict[str, Any]:
        return _gpu_main("T4", record_json, cells_json, profile_text)

    @app.function(gpu=modal_gpu_resource("A10"), **_COMMON)
    def run_gpu_A10(record_json: str, cells_json: str, profile_text: str | None) -> dict[str, Any]:
        return _gpu_main("A10", record_json, cells_json, profile_text)

    @app.function(gpu=modal_gpu_resource("L40S"), **_COMMON)
    def run_gpu_L40S(record_json: str, cells_json: str, profile_text: str | None) -> dict[str, Any]:
        return _gpu_main("L40S", record_json, cells_json, profile_text)

    @app.function(gpu=modal_gpu_resource("A100"), **_COMMON)
    def run_gpu_A100(record_json: str, cells_json: str, profile_text: str | None) -> dict[str, Any]:
        return _gpu_main("A100", record_json, cells_json, profile_text)
else:
    app = None
    artifacts_volume = None


# --------------------------------------------------------------------------------------------------
# Container side
# --------------------------------------------------------------------------------------------------
def _gpu_main(gpu: str, record_json: str, cells_json: str, profile_text: str | None) -> dict[str, Any]:
    import pynvml
    import torch

    from experiments import bench_gpu_session as sess

    record = json.loads(record_json)
    cells = json.loads(cells_json)
    device = torch.cuda.get_device_name(0)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    uuid_raw = pynvml.nvmlDeviceGetUUID(handle)
    gpu_uuid = uuid_raw.decode() if isinstance(uuid_raw, bytes) else str(uuid_raw)
    driver_raw = pynvml.nvmlSystemGetDriverVersion()
    driver = driver_raw.decode() if isinstance(driver_raw, bytes) else str(driver_raw)
    try:
        sess.check_gpu(gpu, device)
    except sess.WrongGpu as exc:
        return {"stopped": str(exc), "gpu_name": device}

    call_id = modal.current_function_call_id()
    locks = modal.Dict.from_name(LOCK_NAME, create_if_missing=True)
    key = lock_key(record["campaign_id"], gpu)
    if not locks.put(key, call_id, skip_if_exists=True) and locks.get(key) != call_id:
        return {"stopped": f"Busy: {key} is held by {locks.get(key)}", "gpu_name": device}

    out_root = Path(ARTIFACTS_ROOT) / volume_dir(record["campaign_id"], gpu)
    try:
        result = sess.run_session(
            record, gpu, cells, out_root=out_root, device_name=device, gpu_uuid=gpu_uuid, driver=driver,
            profile_text=profile_text, call_id=call_id, start_server=_start_vllm, stop_server=_stop_vllm,
            run_harness=_run_harness, sample=lambda: sess.MetricsSampler(vram=lambda: _vram_used_mb(handle)),
            persist=artifacts_volume.commit)
    finally:
        if locks.get(key) == call_id:
            locks.pop(key)
        artifacts_volume.commit()
    return {**result, "gpu_name": device, "gpu_uuid": gpu_uuid, "driver_version": driver}


def _vram_used_mb(handle: Any) -> float:
    import pynvml
    return round(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2, 2)


def _get(url: str) -> int:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status
    except OSError:
        return 0


def _start_vllm(cfg: dict[str, Any], server_dir: Path) -> dict[str, Any]:
    """vLLM on 127.0.0.1:8000 for one (workload, lane); ready when /health and /v1/models answer."""
    import urllib.request

    import pynvml
    import torch
    import vllm
    from flexsds.server.local_runner import LocalServingRunner

    runner = LocalServingRunner()
    cfg = dict(cfg, telemetry_path=str(server_dir / "telemetry" / "serving_trace.jsonl"))
    cmd = runner.build_command(cfg, port=8000)
    env = {**os.environ, **runner.setup_environment(cfg)}
    log = open(server_dir / "vllm.log", "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, cwd=server_dir, stdout=log, stderr=subprocess.STDOUT)
    started = time.monotonic()
    while time.monotonic() - started < 900:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"vLLM exited early with code {proc.returncode} (see {server_dir / 'vllm.log'})")
        if _get("http://127.0.0.1:8000/health") == 200 and _get("http://127.0.0.1:8000/v1/models") == 200:
            break
        time.sleep(5)
    else:
        proc.kill()
        log.close()
        raise TimeoutError("vLLM was not ready within 15 minutes")
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
        model_ids = [item["id"] for item in json.load(response)["data"]]
    pynvml.nvmlInit()
    total = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).total
    model_volume.commit()
    return {"proc": proc, "log": log, "cmd": cmd, "model_ids": model_ids, "ready_s": round(time.monotonic() - started, 1),
            "vllm_version": vllm.__version__, "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda, "gpu_memory_total_mb": round(total / 1024**2, 2),
            "runtime_variant": runtime_patch.RUNTIME_LABEL,
            "vllm_outputs_sha256": hashlib.sha256((Path(vllm.__file__).parent / "outputs.py").read_bytes()).hexdigest()}


def _stop_vllm(server: dict[str, Any]) -> None:
    proc = server["proc"]
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    server["log"].close()
    deadline = time.monotonic() + 60
    while _get("http://127.0.0.1:8000/health") and time.monotonic() < deadline:
        time.sleep(2)
    time.sleep(5)


def _run_harness(args: list[str], env: dict[str, str], log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.run([sys.executable, "-u", "-m", "experiments.benchmark_harness", *args],
                              cwd="/root/project", env=env, stdout=log, stderr=subprocess.STDOUT).returncode


# --------------------------------------------------------------------------------------------------
# Laptop side
# --------------------------------------------------------------------------------------------------
def _modal_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    from flexsds.core.credentials import CredentialsManager
    from flexsds.server import modal_auth

    modal_auth.ensure_modal()
    if not modal_auth.prepare_modal_auth(CredentialsManager()):
        raise RuntimeError("Modal credentials missing; run 'flexsds auth login --provider modal' first.")
    env = dict(os.environ)
    parts = [str(REPO_ROOT), str(SRC_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def _invoke_in_modal_interpreter(action: str, payload: dict[str, Any]) -> Any:
    """Bridge Modal SDK calls through the installed Modal CLI interpreter (e.g. pipx venv)."""
    import shutil
    from flexsds.server import modal_auth

    modal_auth.ensure_modal()
    executable = shutil.which("modal")
    if not executable:
        raise RuntimeError("Modal Python SDK is not installed in this environment and no 'modal' CLI was found.")
    proc = subprocess.run([*modal_auth._modal_interpreter_args(executable), str(Path(__file__).resolve()),
                           "--sdk-action", action], input=json.dumps(payload), cwd=REPO_ROOT, env=_modal_env(),
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"Modal SDK action {action!r} failed: {(proc.stderr or proc.stdout or '').strip()}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


# modal.call_graph.InputStatus: PENDING=0, SUCCESS=1, FAILURE=2, TERMINATED=3, TIMEOUT=4, INIT_FAILURE=5
_DEAD_INPUT_STATUSES = {2, 3, 4, 5}


def _call_graph_status(call: Any) -> int | None:
    """Status of the root input of a function call, or None when Modal cannot tell."""
    try:
        graph = call.get_call_graph()
        return int(graph[0].status) if graph else None
    except Exception:  # noqa: BLE001 - older SDKs / transient errors: fall back to "still running"
        return None


def gpu_call_state(call_id: str) -> str:
    """RUNNING, DONE (session finished) or FAILED (crashed, stopped, wrong card, busy)."""
    if modal is None:
        return str(_invoke_in_modal_interpreter("gpu_call_state", {"call_id": call_id}))
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except Exception as exc:  # noqa: BLE001 - Modal raises its own TimeoutError and remote errors
        if isinstance(exc, TimeoutError) or type(exc).__name__ == "TimeoutError":
            # A call whose container died (e.g. SIGINT) can keep timing out here forever: ask the call graph.
            return "FAILED" if _call_graph_status(call) in _DEAD_INPUT_STATUSES else "RUNNING"
        return "FAILED"
    if not isinstance(result, dict) or result.get("stopped"):
        return "FAILED"
    return "DONE"


def submit_gpu(record: dict[str, Any], gpu: str, cell_keys: list[str], profile_text: str | None, *,
               tries: int = 12, wait_s: float = 30, poll_s: float = 10, attempts: dict[str, int] | None = None,
               max_wait_s: float = 3600) -> str:
    """Spawn the GPU session; ask again while Modal hands out another card. Returns the running call id."""
    if modal is None:
        return str(_invoke_in_modal_interpreter("submit_gpu", {
            "record": record, "gpu": gpu, "cell_keys": cell_keys, "profile_text": profile_text, "tries": tries,
            "wait_s": wait_s, "poll_s": poll_s, "attempts": attempts or {}}))
    key = lock_key(record["campaign_id"], gpu)
    locks = modal.Dict.from_name(LOCK_NAME, create_if_missing=True)
    holder = locks.get(key) if locks.contains(key) else None
    if holder:
        if gpu_call_state(holder) == "RUNNING":
            raise RuntimeError(f"{gpu} is already running in call {holder}; collect it or stop it first")
        locks.pop(key)  # stale: that call is finished or gone
    by_key = {c["key"]: c for c in record["cells"]}
    cells = [dict(by_key[k], attempt=(attempts or {}).get(k, 1)) for k in cell_keys]
    function = modal.Function.from_name(APP_NAME, function_name(gpu))
    last = ""
    for attempt in range(1, tries + 1):
        call = function.spawn(json.dumps(record), json.dumps(cells), profile_text)
        started = time.monotonic()
        while True:
            try:
                result = call.get(timeout=0)
            except Exception as exc:  # noqa: BLE001
                if not (isinstance(exc, TimeoutError) or type(exc).__name__ == "TimeoutError"):
                    raise
                if locks.contains(key) and locks.get(key) == call.object_id:
                    return call.object_id  # right card, session running
                if time.monotonic() - started > max_wait_s:
                    raise RuntimeError(f"{gpu}: call {call.object_id} has not started within {max_wait_s:.0f} s") from exc
                time.sleep(poll_s)
                continue
            stopped = str(result.get("stopped") or "") if isinstance(result, dict) else ""
            if stopped.startswith("WrongGpu"):
                last = stopped
                print(f"{gpu}: try {attempt}/{tries}: {stopped}", flush=True)
                break
            if stopped:
                raise RuntimeError(f"{gpu}: {stopped}")
            return call.object_id  # finished already (e.g. every cell failed fast); collect will show why
        time.sleep(wait_s)
    raise RuntimeError(f"{gpu}: Modal did not hand out the exact card in {tries} tries; last: {last}")


ATTEMPT_DIR = re.compile(r"a\d{3}")


def collect_gpu(campaign_id: str, gpu: str, evidence_root: Path) -> list[str]:
    """Copy finished cells of one GPU from the artifacts volume to <evidence_root>/<attempt_id>/."""
    evidence_root = Path(evidence_root)
    if modal is None:
        return list(_invoke_in_modal_interpreter("collect_gpu", {
            "campaign_id": campaign_id, "gpu": gpu, "evidence_root": str(evidence_root)}))
    volume = modal.Volume.from_name("ptit-run-artifacts")
    base = volume_dir(campaign_id, gpu)
    # Layout: <cell>/<aNNN>/<file> (one directory per attempt); campaigns before 27/11 used <cell>/<file>.
    files: dict[tuple[str, str], dict[str, str]] = {}
    for entry in volume.listdir(base, recursive=True):
        if getattr(getattr(entry, "type", None), "value", 1) != 1:  # FileEntryType.FILE; skip directories
            continue
        rel = entry.path[len(base) + 1:] if entry.path.startswith(base + "/") else entry.path
        top, _, name = rel.partition("/")
        attempt, _, inner = name.partition("/")
        if inner and not top.startswith("_") and ATTEMPT_DIR.fullmatch(attempt):
            files.setdefault((top, attempt), {})[inner] = entry.path
        elif name:
            files.setdefault((top, ""), {})[name] = entry.path

    def read(path: str) -> bytes:
        return b"".join(volume.read_file(path))

    collected: list[str] = []
    for (top, attempt), names in sorted(files.items()):
        if top == "_servers":
            for name, path in names.items():
                target = evidence_root.parent / "servers" / gpu / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(read(path))
            continue
        if top.startswith("_"):
            continue
        if "bench.json" not in names:
            if "failure.json" in names:  # the cell failed inside the container (server, profile, ...)
                target = evidence_root.parent / "failures" / (f"{top}.{attempt}.json" if attempt else f"{top}.json")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(read(names["failure.json"]))
            continue
        bench = json.loads(read(names["bench.json"]))
        out = evidence_root / bench["bench_run_id"]
        out.mkdir(parents=True, exist_ok=True)
        payloads = {name: read(path) for name, path in names.items() if "/" not in name}
        for name, content in payloads.items():
            target = out / name
            if target.exists() and target.read_bytes() != content:
                raise FileExistsError(f"Conflicting existing evidence: {target}")
        for name, content in payloads.items():
            if not (out / name).exists():
                (out / name).write_bytes(content)
        collected.append(top)
    return collected


def release_locks(campaign_id: str) -> None:
    """Drop every GPU lock of one campaign (used by `flexsds benchmark stop`)."""
    if modal is None:
        _invoke_in_modal_interpreter("release_locks", {"campaign_id": campaign_id})
        return
    locks = modal.Dict.from_name(LOCK_NAME, create_if_missing=True)
    for gpu in MODAL_GPU_NAMES:
        key = lock_key(campaign_id, gpu)
        if locks.contains(key):
            locks.pop(key)


def deploy_app() -> None:
    """Deploy `flexsds-benchmark` from this checkout (all four run_gpu_* functions)."""
    from flexsds.server import modal_auth

    subprocess.run([*modal_auth.modal_command(), "deploy", "-m", "experiments.modal_benchmark_runner"],
                   cwd=REPO_ROOT, env=_modal_env(), check=True)


def cancel_call(call_id: str) -> None:
    """Cancel one GPU session (one campaign's call); other campaigns on the same app keep running."""
    if modal is None:
        _invoke_in_modal_interpreter("cancel_call", {"call_id": call_id})
        return
    modal.FunctionCall.from_id(call_id).cancel()


def stop_app() -> None:
    """Stop the deployed benchmark app (cancels every running GPU session of every campaign)."""
    from flexsds.server import modal_auth

    env = _modal_env()
    apps = json.loads(subprocess.check_output([*modal_auth.modal_command(), "app", "list", "--json"],
                                              cwd=REPO_ROOT, env=env, text=True))
    for item in apps:
        if item.get("description") == APP_NAME and item.get("state") != "stopped":
            subprocess.run([*modal_auth.modal_command(), "app", "stop", item["app_id"], "-y"],
                           cwd=REPO_ROOT, env=env, check=True)


SDK_ACTIONS = {
    "submit_gpu": lambda p: submit_gpu(p["record"], p["gpu"], p["cell_keys"], p["profile_text"], tries=p["tries"],
                                       wait_s=p["wait_s"], poll_s=p["poll_s"], attempts=p.get("attempts")),
    "gpu_call_state": lambda p: gpu_call_state(p["call_id"]),
    "collect_gpu": lambda p: collect_gpu(p["campaign_id"], p["gpu"], Path(p["evidence_root"])),
    "release_locks": lambda p: release_locks(p["campaign_id"]),
    "cancel_call": lambda p: cancel_call(p["call_id"]),
}


def _sdk_bridge_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-action", choices=tuple(SDK_ACTIONS), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(SDK_ACTIONS[args.sdk_action](json.loads(sys.stdin.read()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(_sdk_bridge_main())
