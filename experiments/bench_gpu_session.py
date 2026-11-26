"""In-container session of the one-container-per-GPU FlexSDS benchmark (design: openspec/changes/
flexsds-bench-one-container-per-gpu/design.md).

One Modal container per GPU runs every pending cell of that GPU: for each (workload, lane) a fresh vLLM
process on 127.0.0.1:8000, then the canonical harness per concurrency. Requests never pass Modal's web
proxy, so a cell at concurrency c really has c requests in flight; a /metrics sampler proves it.

Container effects (vLLM, harness subprocess, samplers, volume commits) are injected so the session logic is
tested offline (tests/test_bench_gpu_session.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
import urllib.request
import uuid

from experiments import modal_lanes as lanes
from experiments import run_benchmark_campaign as camp

try:
    from flexsds.core.hardware import device_matches_exactly
except ModuleNotFoundError:  # repository checkout without the package on sys.path
    from src.flexsds.core.hardware import device_matches_exactly

LOCAL_BASE_URL = "http://127.0.0.1:8000"
WORKLOAD_ORDER = {w: i for i, w in enumerate(camp.BENCHMARK_WORKLOADS)}
LANE_ORDER = {l: i for i, l in enumerate(camp.BENCHMARK_LANES)}
_METRIC = re.compile(r"^(vllm:num_requests_running|vllm:num_requests_waiting)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


class WrongGpu(RuntimeError):
    """Modal handed out another card than the one this GPU tier is measured on."""


def check_gpu(gpu: str, device_name: str) -> None:
    if not device_matches_exactly(gpu, device_name):
        raise WrongGpu(f"WrongGpu: {gpu} must be measured on its exact card, got {device_name}")


def parse_vllm_metrics(text: str) -> tuple[float, float]:
    """Sum of vllm:num_requests_running and vllm:num_requests_waiting over all label sets."""
    running = waiting = 0.0
    for line in text.splitlines():
        match = _METRIC.match(line.strip())
        if not match:
            continue
        if match.group(1).endswith("running"):
            running += float(match.group(3))
        else:
            waiting += float(match.group(3))
    return running, waiting


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class MetricsSampler:
    """Polls vLLM /metrics (and optionally GPU memory) in a thread while one harness run is in flight."""

    def __init__(self, base_url: str = LOCAL_BASE_URL, interval_s: float = 0.25,
                 fetch: Callable[[str], str] | None = None, vram: Callable[[], float | None] | None = None):
        self.url = base_url.rstrip("/") + "/metrics"
        self.interval_s = interval_s
        self.fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=5).read().decode("utf-8"))
        self.vram = vram
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                running, waiting = parse_vllm_metrics(self.fetch(self.url))
                row: dict[str, Any] = {"ts_utc": _utc_now(), "running": running, "waiting": waiting}
                if self.vram is not None:
                    row["used_mb"] = self.vram()
                self.samples.append(row)
            except OSError:
                pass  # a missed poll is fine; a repeat with no sample at all fails concurrency_check
            self._stop.wait(self.interval_s)

    def start(self) -> "MetricsSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return list(self.samples)


def concurrency_check(samples: list[dict], raw: list[dict], concurrency: int) -> dict[str, Any]:
    """Every repeat must have seen at least `concurrency` requests in vLLM (running + waiting).

    The window of a repeat is the harness' run_started_at_utc..run_finished_at_utc. Running alone may stay
    below c when KV cache is short (e.g. T4 with W1_V3); that is real behaviour and is reported, not failed.
    """
    windows: dict[str, tuple[float, float]] = {}
    for row in raw:
        rid = row["repeat_id"]
        start, end = _ts(row["run_started_at_utc"]), _ts(row["run_finished_at_utc"])
        old = windows.get(rid)
        windows[rid] = (min(start, old[0]), max(end, old[1])) if old else (start, end)
    points = [(_ts(s["ts_utc"]), float(s["running"]), float(s["waiting"])) for s in samples]
    per_repeat: dict[str, dict[str, float]] = {}
    ok = bool(windows)
    for rid in sorted(windows):
        start, end = windows[rid]
        inside = [(r, w) for t, r, w in points if start <= t <= end]
        if not inside:
            per_repeat[rid] = {"peak_inflight": 0.0, "peak_running": 0.0}
            ok = False
            continue
        peak_inflight = max(r + w for r, w in inside)
        per_repeat[rid] = {"peak_inflight": peak_inflight, "peak_running": max(r for r, _ in inside)}
        ok = ok and peak_inflight >= concurrency
    return {"ok": ok, "per_repeat": per_repeat}


def session_order(cells: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """(workload, lane, cells by ascending concurrency): W1_V3, W2, W3 x vanilla, static, adaptive."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for cell in cells:
        groups.setdefault((cell["workload"], cell["lane"]), []).append(cell)
    ordered = sorted(groups, key=lambda wl: (WORKLOAD_ORDER[wl[0]], LANE_ORDER[wl[1]]))
    return [(w, l, sorted(groups[(w, l)], key=lambda c: c["concurrency"])) for w, l in ordered]


def build_harness_args(*, lane: str, gpu: str, base_url: str, served_model_name: str, server_env_json: str,
                       raw_out: str, summary_csv: str, experiment_id: str, workload: str, concurrency: int,
                       repeats: int, num_requests: int, warmup_requests: int = 3) -> list[str]:
    """experiments.benchmark_harness argv for one cell (canonical protocol, per-GPU environment profile)."""
    serving = lanes.workload_serving(workload)
    timeout = [] if serving["timeout_s"] is None else ["--timeout-s", str(serving["timeout_s"])]
    return [
        "--lane", lane, "--workload", workload,
        "--endpoint", f"{base_url.rstrip('/')}/chat/completions", "--model", served_model_name,
        "--workload-file", lanes.workload_file(workload), "--sample-policy", "cycle-rotate",
        "--concurrency", str(concurrency), "--repeats", str(repeats), "--num-requests", str(num_requests),
        "--warmup-requests", str(warmup_requests), "--warmup-policy", "m3-fixed3",
        "--experiment-id", experiment_id, "--raw-out", raw_out, "--summary-csv", summary_csv,
        "--environment-profile", lanes.environment_profile_for_gpu(gpu), "--server-env-json", server_env_json,
        *timeout,
    ]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fail(cell_dir: Path, kind: str, error: str) -> str:
    cell_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cell_dir / "failure.json", {"status": f"failed:{kind}", "error": error, "at_utc": _utc_now()})
    return f"failed:{kind}"


def _adaptive_profile(out_root: Path, gpu: str, device_name: str, profile_text: str | None) -> tuple[str | None, str]:
    """Write the adaptive profile into the container; '' or an error message when it cannot be used."""
    if not profile_text:
        return None, "missing hardware profile"
    data = json.loads(profile_text)
    profile_device = str(data.get("metadata", {}).get("device") or "")
    if profile_device != device_name or not device_matches_exactly(gpu, profile_device):
        return None, f"profile measured on {profile_device!r}, container card is {device_name!r}"
    path = out_root / "_profile" / "hardware_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile_text, encoding="utf-8")
    return str(path), ""


def run_session(record: dict, gpu: str, cells: list[dict], *, out_root: Path, device_name: str, gpu_uuid: str,
                driver: str, profile_text: str | None, call_id: str, start_server: Callable, stop_server: Callable,
                run_harness: Callable, sample: Callable, persist: Callable[[], None]) -> dict[str, Any]:
    """Measure every given cell of one GPU in this container; returns {"cells": {key: status}}."""
    check_gpu(gpu, device_name)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    profile_path, profile_error = (None, "")
    if any(c["lane"] == "adaptive" for c in cells):
        profile_path, profile_error = _adaptive_profile(out_root, gpu, device_name, profile_text)

    for workload, lane, group in session_order(cells):
        if lane == "adaptive" and profile_error:
            for cell in group:
                status[cell["key"]] = _fail(out_root / cell["key"], "profile", profile_error)
            persist()
            continue
        cfg = camp.build_cell_serving_config(record, group[0], profile_path=profile_path if lane == "adaptive" else None)
        instance_id = str(uuid.uuid4())
        server_run_id = f"{record['campaign_id']}-{gpu}-{workload}-{lane}-{instance_id[:8]}"
        cfg["run_id"], cfg["source_commit"] = server_run_id, record["commit"]
        server_dir = out_root / "_servers" / f"{workload}__{lane}"
        server_dir.mkdir(parents=True, exist_ok=True)
        try:
            server = start_server(cfg, server_dir)
        except Exception as exc:  # vLLM could not start (e.g. OOM): the lane fails, the session goes on
            for cell in group:
                status[cell["key"]] = _fail(out_root / cell["key"], "server", f"{type(exc).__name__}: {exc}")
            persist()
            continue
        try:
            for cell in group:
                status[cell["key"]] = _run_cell(record, gpu, cell, cfg, server, out_root, device_name=device_name,
                                                gpu_uuid=gpu_uuid, driver=driver, instance_id=instance_id,
                                                server_run_id=server_run_id, call_id=call_id,
                                                run_harness=run_harness, sample=sample)
                persist()
        finally:
            stop_server(server)
    return {"cells": status}


def _run_cell(record: dict, gpu: str, cell: dict, cfg: dict, server: dict, out_root: Path, *, device_name: str,
              gpu_uuid: str, driver: str, instance_id: str, server_run_id: str, call_id: str,
              run_harness: Callable, sample: Callable) -> str:
    from experiments.modal_vram_join import join_vram

    cell_dir = out_root / cell["key"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    attempt = int(cell.get("attempt", 1))
    bench_run_id = f"{camp.attempt_prefix(record, cell)}{attempt:03d}"
    submission = {
        "gpu": gpu, "lane": cell["lane"], "server_run_id": server_run_id, "bench_run_id": bench_run_id,
        "expected_profile_sha256": cfg.get("profile_sha256") if cell["lane"] == "adaptive" else None,
        "cell": {"workload": cell["workload"], "concurrency": cell["concurrency"], "repeats": camp.REPEATS,
                 "num_requests": camp.NUM_REQUESTS},
        "commit": record["commit"], "source_digest": record.get("source_digest"),
        "submitted_at_utc": _utc_now(), "call_id": call_id,
    }
    _write_json(cell_dir / "submission.json", submission)
    server_env = {
        "gpu": device_name, "gpu_uuid": gpu_uuid, "driver_version": driver, "requested_gpu": gpu,
        "modal_gpu": lanes.modal_gpu_resource(gpu), "gpu_memory_total_mb": server.get("gpu_memory_total_mb"),
        "torch_version": server.get("torch_version"), "torch_cuda_version": server.get("torch_cuda_version"),
        "vllm_version": server.get("vllm_version"), "dtype": cfg["dtype"], "quantization": cfg["quantization"],
        "lane": cell["lane"], "run_id": server_run_id, "instance_id": instance_id, "started_at_utc": _utc_now(),
        "source_commit": record["commit"], "source_digest": record.get("source_digest"),
        "target_model": cfg["target_model"], "draft_model": cfg["draft_model"],
        "served_model_name": cfg["served_model_name"], "max_model_len": cfg["max_model_len"],
        "runtime_variant": server.get("runtime_variant"), "vllm_outputs_sha256": server.get("vllm_outputs_sha256"),
        "vllm_args": server.get("cmd"), "profile_sha256": cfg.get("profile_sha256"),
    }
    env_path = cell_dir / "server_env.json"
    _write_json(env_path, server_env)
    args = build_harness_args(
        lane=cell["lane"], gpu=gpu, base_url=f"{LOCAL_BASE_URL}/v1", served_model_name=cfg["served_model_name"],
        server_env_json=str(env_path), raw_out=str(cell_dir / "raw.jsonl"),
        summary_csv=str(cell_dir / "harness_header.csv"), experiment_id=bench_run_id,
        workload=cell["workload"], concurrency=cell["concurrency"], repeats=camp.REPEATS,
        num_requests=camp.NUM_REQUESTS, warmup_requests=camp.WARMUP_REQUESTS)
    env = {**os.environ, "PTIT_GIT_COMMIT": record["commit"], "PTIT_SERVER_ENV_ROOT": str(out_root),
           "PYTHONIOENCODING": "utf-8"}
    sampler = sample().start()
    started = time.monotonic()
    try:
        returncode = run_harness(args, env, cell_dir / "harness.log")
    finally:
        samples = sampler.stop()
    _jsonl(cell_dir / "concurrency.jsonl", samples)
    raw = _load_jsonl(cell_dir / "raw.jsonl")
    check = concurrency_check(samples, raw, cell["concurrency"]) if raw else {"ok": False, "per_repeat": {}}
    join_error = None
    vram_rows = [{"ts_utc": s["ts_utc"], "used_mb": s["used_mb"], "total_mb": server.get("gpu_memory_total_mb")}
                 for s in samples if s.get("used_mb") is not None]
    if raw and vram_rows:
        try:
            _jsonl(cell_dir / "raw.vram.jsonl", join_vram(raw, vram_rows))
        except (ValueError, KeyError) as exc:
            join_error = str(exc)
    bench = {
        **{k: submission[k] for k in ("gpu", "lane", "server_run_id", "bench_run_id", "cell", "commit", "call_id",
                                      "submitted_at_utc", "source_digest")},
        "returncode": returncode, "join_error": join_error, "timestamp_utc": _utc_now(),
        "server_instance_unchanged": True, "endpoint_model_ids": server.get("model_ids", [cfg["served_model_name"]]),
        "measured_wall_s": round(time.monotonic() - started, 2),
        "concurrency_ok": check["ok"], "concurrency_per_repeat": check["per_repeat"],
    }
    _write_json(cell_dir / "bench.json", bench)
    return "done" if returncode == 0 else "failed:harness"
