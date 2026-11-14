from __future__ import annotations

import argparse
import asyncio
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Any

try:
    import aiohttp
except ImportError:
    class _AiohttpStub:
        ClientError = Exception
        ClientResponse = Any
        ClientSession = Any
        __version__ = "stub"
    aiohttp = _AiohttpStub()  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
        nvmlShutdown,
        nvmlSystemGetDriverVersion,
    )
except ImportError:
    nvmlInit = None  # type: ignore[assignment]
    nvmlShutdown = None  # type: ignore[assignment]
    nvmlDeviceGetHandleByIndex = None  # type: ignore[assignment]
    nvmlDeviceGetMemoryInfo = None  # type: ignore[assignment]
    nvmlDeviceGetName = None  # type: ignore[assignment]
    nvmlSystemGetDriverVersion = None  # type: ignore[assignment]

try:
    from workload_adapter import (
        WorkloadContractError,
        WorkloadIntegrity,
        build_request_payload,
        build_workload_metadata,
        load_workload_rows,
        resolve_repeat_offset,
        resolve_sample_index,
        select_workload_row,
        validate_manifest_consistency,
        validate_workload_label,
        verify_workload_integrity,
    )
except ImportError:
    from experiments.workload_adapter import (
        WorkloadContractError,
        WorkloadIntegrity,
        build_request_payload,
        build_workload_metadata,
        load_workload_rows,
        resolve_repeat_offset,
        resolve_sample_index,
        select_workload_row,
        validate_manifest_consistency,
        validate_workload_label,
        verify_workload_integrity,
    )


ENVIRONMENT_PROFILE = "m1-t4-cu130-v1"

EXPECTED_GPU = "Tesla T4"
EXPECTED_VLLM = "0.30.0"
EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_TORCH_CUDA = "13.0"

L4_MODAL_PROFILE = "m3-l4-modal-v1"

# "local": vLLM runs on this machine; GPU/VRAM read here with NVML (M1/M2, T4).
# "server": vLLM runs elsewhere (Modal L4, [M3-01]); the environment comes from the
# server_env.json written by the server container and VRAM is joined afterwards
# from the server's own NVML samples (experiments/modal_vram_join.py).
ENVIRONMENT_PROFILES = {
    "m4-a100-colab-v1": {
        "gpu": "NVIDIA A100", "vllm_version": "0.30.0",
        "torch_version": "2.13.0+cu130", "torch_cuda_version": "13.0", "mode": "local",
    },
    ENVIRONMENT_PROFILE: {
        "gpu": EXPECTED_GPU,
        "vllm_version": EXPECTED_VLLM,
        "torch_version": EXPECTED_TORCH,
        "torch_cuda_version": EXPECTED_TORCH_CUDA,
        "mode": "local",
    },
    L4_MODAL_PROFILE: {
        "gpu": "NVIDIA L4",
        "vllm_version": "0.30.0",
        "torch_version": "2.13.0+cu130",
        "torch_cuda_version": "13.0",
        "mode": "server",
    },
}

SERVER_ENV_REQUIRED = (
    "gpu",
    "driver_version",
    "gpu_memory_total_mb",
    "torch_version",
    "torch_cuda_version",
    "vllm_version",
    "dtype",
    "quantization",
)


class ServerEnvError(ValueError):
    """The server-reported environment file is missing or incomplete."""

DEFAULT_CONCURRENCY_SWEEP = (1, 2, 4, 8, 16, 32)
CANONICAL_MIN_REPEATS = 5
CANONICAL_MIN_WARMUP_REQUESTS = 3

SUMMARY_COLUMNS = [
    "lane",
    "workload",
    "concurrency",
    "gpu",
    "n_runs",
    "ttft_p50",
    "tpot_p50",
    "tpot_p95",
    "e2e_p50",
    "alpha",
    "tau",
    "vram_peak_mb",
    "throughput",
    "speedup_vs_vanilla",
    "variance_flag",
    "commit",
    "timestamp",
]


class RawJsonlWriter:
    """Manage measured raw JSONL at experiment scope."""

    def __init__(self, path: Path, overwrite: bool) -> None:
        self.path = path
        self.overwrite = overwrite

    def prepare(self) -> None:
        """Prepare the experiment output before any warm-up or measurement."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            self.path.unlink(missing_ok=True)

    def write(self, records: list[dict]) -> None:
        """Append measured records from one repeat."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    # Remote runs (Modal) have no .git; the launcher passes the local HEAD.
    if os.environ.get("PTIT_GIT_COMMIT"):
        return os.environ["PTIT_GIT_COMMIT"]
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_server_env(path: Path) -> dict:
    """Read the environment written by the remote server container.

    The file must sit under PTIT_SERVER_ENV_ROOT (default: the working
    directory); the in-Modal benchmark sets it to the artifacts volume.
    """
    root = Path(os.environ.get("PTIT_SERVER_ENV_ROOT") or Path.cwd()).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ServerEnvError(f"server environment file {path} is outside {root}")
    if not path.is_file():
        raise ServerEnvError(f"server environment file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServerEnvError(f"server environment file is not JSON: {path}: {exc}") from exc
    missing = [name for name in SERVER_ENV_REQUIRED if data.get(name) in (None, "")]
    if missing:
        raise ServerEnvError(f"server environment {path} is missing: {', '.join(missing)}")
    return data


def environment_info(
    gpu_index: int,
    profile: str = ENVIRONMENT_PROFILE,
    server_env_json: Path | None = None,
) -> tuple[dict, object]:
    """Return (environment record, local NVML handle or None for server mode)."""
    if ENVIRONMENT_PROFILES[profile]["mode"] == "server":
        server = load_server_env(server_env_json)
        info = {
            "environment_profile": profile,
            "environment_source": "server",
            "gpu": server["gpu"],
            "gpu_index": None,
            "gpu_memory_total_mb": server["gpu_memory_total_mb"],
            "driver_version": server["driver_version"],
            "torch_version": server["torch_version"],
            "torch_cuda_version": server["torch_cuda_version"],
            "vllm_version": server["vllm_version"],
            "server_dtype": server["dtype"],
            "server_quantization": server["quantization"],
            "server_lane": server.get("lane"),
            "server_run_id": server.get("run_id"),
            "server_instance_id": server.get("instance_id"),
            # Configured launch identities; endpoint identity still needs a runtime probe.
            "target_model": server.get("target_model"),
            "target_revision": server.get("target_revision"),
            "draft_model": server.get("draft_model"),
            "draft_revision": server.get("draft_revision"),
            "served_model_name": server.get("served_model_name"),
            "aiohttp_version": aiohttp.__version__,
            "commit": git_commit(),
        }
        return info, None

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    name = nvmlDeviceGetName(handle)
    driver = nvmlSystemGetDriverVersion()
    mem = nvmlDeviceGetMemoryInfo(handle)

    info = {
        "environment_profile": profile,
        "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "gpu_index": gpu_index,
        "gpu_memory_total_mb": round(mem.total / 1024**2, 2),
        "driver_version": (
            driver.decode() if isinstance(driver, bytes) else str(driver)
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": package_version("vllm"),
        "aiohttp_version": aiohttp.__version__,
        "commit": git_commit(),
    }
    return info, handle


def validate_environment(env: dict, profile: str = ENVIRONMENT_PROFILE) -> list[str]:
    expected = {
        field: value
        for field, value in ENVIRONMENT_PROFILES[profile].items()
        if field != "mode"
    }

    mismatches: list[str] = []
    for field, expected_value in expected.items():
        actual_value = env.get(field)
        if profile == "m4-a100-colab-v1" and field == "gpu":
            if "NVIDIA" not in str(actual_value) or "A100" not in str(actual_value):
                mismatches.append(f"gpu: expected NVIDIA A100, got {actual_value}")
            continue
        if isinstance(expected_value, (tuple, list, set)):
            if actual_value not in expected_value:
                mismatches.append(
                    f"{field}: expected one of {expected_value}, got {actual_value}"
                )
        elif actual_value != expected_value:
            mismatches.append(
                f"{field}: expected {expected_value}, got {actual_value}"
            )
    return mismatches


def ensure_summary(path: Path) -> None:
    """Create the canonical summary header if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(",".join(SUMMARY_COLUMNS) + "\n", encoding="utf-8")


def make_experiment_id(configured_id: str | None) -> str:
    if configured_id:
        return configured_id
    return f"exp-{uuid.uuid4()}"


def warmup_count(args: argparse.Namespace, concurrency: int) -> int:
    if getattr(args, "warmup_policy", "concurrency-matched") == "m3-fixed3":
        if args.warmup_requests != 3:
            raise ValueError("M3 confirmatory warm-up requires exactly 3 requests")
        return 3
    return max(CANONICAL_MIN_WARMUP_REQUESTS, args.warmup_requests, concurrency)


def make_cell_id(lane: str, workload: str, concurrency: int) -> str:
    return f"{lane}__{workload}__c{concurrency}"


def make_repeat_id(repeat_index: int) -> str:
    return f"r{repeat_index:03d}"


def make_request_identity(
    experiment_id: str,
    cell_id: str,
    repeat_id: str | None,
    request_index: int,
    phase: str,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "cell_id": cell_id,
        "repeat_id": repeat_id,
        "request_id": f"req-{uuid.uuid4()}",
        "request_index": request_index,
        "phase": phase,
    }


async def iter_sse(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
    data_lines: list[str] = []

    async for raw_line in response.content:
        line = raw_line.decode("utf-8").rstrip("\r\n")

        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue

        if line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


async def monitor_vram(handle, stop: asyncio.Event, interval: float) -> float:
    peak = 0.0

    while not stop.is_set():
        used = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        peak = max(peak, used)

        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

    used = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    return round(max(peak, used), 2)


def build_legacy_workload_rows(args: argparse.Namespace) -> list[dict]:
    """Preserve the previous one-prompt path for smoke/debug runs only."""
    return [
        {
            "workload_id": args.workload,
            "dataset_version": "legacy-cli",
            "sample_id": "legacy-cli-000",
            "prompt_version": "legacy-cli.v1",
            "request": {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": args.prompt,
                    }
                ],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
        }
    ]


async def run_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    env: dict,
    identity: dict,
    concurrency: int,
    payload: dict,
    workload_metadata: dict,
) -> dict:
    """Execute exactly one already-resolved HTTP request and measure it."""
    queued_at = time.perf_counter()

    record = {
        **identity,
        "server_request_id": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lane": args.lane,
        "workload": args.workload,
        "concurrency": concurrency,
        "model": payload["model"],
        "temperature": payload.get("temperature"),
        "seed": payload.get("seed"),
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "response_format": payload.get("response_format"),
        "workload_id": workload_metadata["workload_id"],
        "dataset_version": workload_metadata["dataset_version"],
        "sample_id": workload_metadata["sample_id"],
        "sample_index": workload_metadata["sample_index"],
        "sample_offset": workload_metadata["sample_offset"],
        "canonical_sample_count": workload_metadata[
            "canonical_sample_count"
        ],
        "sample_policy": workload_metadata["sample_policy"],
        "prompt_version": workload_metadata["prompt_version"],
        "package_model": workload_metadata.get("package_model"),
        "manifest_package_sha256": workload_metadata.get(
            "manifest_package_sha256"
        ),
        "workload_artifact_key": workload_metadata.get("workload_artifact_key"),
        "workload_file_sha256": workload_metadata.get("workload_file_sha256"),
        "workload_schema_version": workload_metadata.get("schema_version"),
        "img_id": workload_metadata.get("img_id"),
        "expected_source": workload_metadata.get("expected_source"),
        "success": False,
        "metrics_complete": False,
        "http_status": None,
        "error_type": None,
        "error_message": None,
        "client_queue_ms": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "itl_ms": [],
        "input_tokens": None,
        "output_tokens": None,
        "generated_text": "",
        "server_metrics": None,
        **env,
    }

    try:
        async with semaphore:
            record["client_queue_ms"] = round(
                (time.perf_counter() - queued_at) * 1000,
                3,
            )

            start = time.perf_counter()
            last_timestamp = start
            first_chunk = False
            generated: list[str] = []

            async with session.post(args.endpoint, json=payload) as response:
                record["http_status"] = response.status

                if response.status != 200:
                    record["error_type"] = f"http_{response.status}"
                    record["error_message"] = (await response.text())[:2000]
                    return record

                async for message in iter_sse(response):
                    if message == "[DONE]":
                        break

                    timestamp = time.perf_counter()
                    data = json.loads(message)

                    if record["server_request_id"] is None:
                        record["server_request_id"] = data.get("id")

                    choices = data.get("choices")
                    if choices:
                        content = (choices[0].get("delta") or {}).get("content")

                        if not first_chunk:
                            first_chunk = True
                            record["ttft_ms"] = round(
                                (timestamp - start) * 1000,
                                3,
                            )
                        else:
                            record["itl_ms"].append(
                                round((timestamp - last_timestamp) * 1000, 3)
                            )

                        if isinstance(content, str):
                            generated.append(content)

                    usage = data.get("usage")
                    if usage:
                        record["input_tokens"] = usage.get("prompt_tokens")
                        record["output_tokens"] = usage.get("completion_tokens")
                        record["server_metrics"] = data.get("metrics")

                    last_timestamp = timestamp

            if not first_chunk:
                record["error_type"] = "no_choice_event"
                record["error_message"] = "No streamed choices event received."
                return record

            record["generated_text"] = "".join(generated)
            record["e2e_ms"] = round((last_timestamp - start) * 1000, 3)

            output_tokens = record["output_tokens"]
            if isinstance(output_tokens, int) and output_tokens > 1:
                record["tpot_ms"] = round(
                    (record["e2e_ms"] - record["ttft_ms"])
                    / (output_tokens - 1),
                    3,
                )

            record["success"] = True
            record["metrics_complete"] = (
                record["ttft_ms"] is not None
                and record["e2e_ms"] is not None
                and isinstance(output_tokens, int)
                and output_tokens >= 1
                and (output_tokens == 1 or record["tpot_ms"] is not None)
            )

    except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as exc:
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)

    return record


async def run_requests(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    env: dict,
    workload_rows: list[dict],
    workload_integrity: WorkloadIntegrity | None,
    sample_policy: str,
    count: int,
    concurrency: int,
    experiment_id: str,
    cell_id: str,
    repeat_id: str | None,
    phase: str,
    request_index_start: int = 0,
    sample_offset: int = 0,
) -> list[dict]:
    """Resolve identity/sample/payload first, then schedule concurrent work."""
    semaphore = asyncio.Semaphore(concurrency)
    resolved_requests: list[tuple[dict, dict, dict]] = []

    for offset in range(count):
        request_index = request_index_start + offset

        sample_index = resolve_sample_index(
            len(workload_rows),
            request_index,
            sample_policy=sample_policy,
            sample_offset=sample_offset,
        )

        identity = make_request_identity(
            experiment_id=experiment_id,
            cell_id=cell_id,
            repeat_id=repeat_id,
            request_index=request_index,
            phase=phase,
        )

        row = select_workload_row(
            workload_rows,
            request_index,
            sample_policy=sample_policy,
            sample_offset=sample_offset,
        )

        workload_metadata = build_workload_metadata(
            row,
            integrity=workload_integrity,
        )
        workload_metadata.update(
            {
                "sample_index": sample_index,
                "sample_offset": sample_offset,
                "canonical_sample_count": len(workload_rows),
                "sample_policy": sample_policy,
            }
        )
        payload = build_request_payload(
            row,
            runtime_model=args.model,
        )

        resolved_requests.append(
            (identity, payload, workload_metadata)
        )

    return await asyncio.gather(
        *[
            run_one(
                session=session,
                semaphore=semaphore,
                args=args,
                env=env,
                identity=identity,
                concurrency=concurrency,
                payload=payload,
                workload_metadata=workload_metadata,
            )
            for identity, payload, workload_metadata in resolved_requests
        ]
    )


async def warm_up_cell(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    env: dict,
    workload_rows: list[dict],
    workload_integrity: WorkloadIntegrity | None,
    sample_policy: str,
    experiment_id: str,
    cell_id: str,
    concurrency: int,
) -> int:
    effective_warmup_count = warmup_count(args, concurrency)

    print("\n===== M1-02 CELL WARM-UP =====")
    print(f"policy               : {getattr(args, 'warmup_policy', 'concurrency-matched')}")
    print(f"experiment_id        : {experiment_id}")
    print(f"cell_id              : {cell_id}")
    print(f"requested_minimum    : {args.warmup_requests}")
    print(f"warmup_requests      : {effective_warmup_count}")
    print(f"warmup_concurrency   : {concurrency}")
    print(f"measured_concurrency : {concurrency}")
    print(f"sample_policy        : {sample_policy}")
    print("==============================")

    warm = await run_requests(
        session=session,
        args=args,
        env=env,
        workload_rows=workload_rows,
        workload_integrity=workload_integrity,
        sample_policy=sample_policy,
        count=effective_warmup_count,
        concurrency=concurrency,
        experiment_id=experiment_id,
        cell_id=cell_id,
        repeat_id=None,
        phase="warmup",
        request_index_start=0,
        sample_offset=0,
    )

    if getattr(args, "warmup_policy", "concurrency-matched") == "m3-fixed3":
        path = Path(args.raw_out).with_name("warmup.jsonl")
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in warm:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    warmup_failures = [record for record in warm if not record["success"]]
    if warmup_failures:
        print("\n===== WARM-UP FAILED =====")
        print(f"cell_id              : {cell_id}")
        print(f"attempted            : {len(warm)}")
        print(f"failed               : {len(warmup_failures)}")
        for record in warmup_failures[:5]:
            print(
                "failure              : "
                f"request_id={record['request_id']} "
                f"sample_id={record.get('sample_id')} "
                f"error_type={record['error_type']} "
                f"error_message={record['error_message']}"
            )
        if len(warmup_failures) > 5:
            print(f"additional_failures   : {len(warmup_failures) - 5}")
        print("==========================")
        return 4

    print("\n===== WARM-UP PASS =====")
    print(f"cell_id              : {cell_id}")
    print(f"requests             : {len(warm)}")
    print(f"successes            : {len(warm)}")
    print(f"concurrency_limit    : {concurrency}")
    print("========================")
    return 0


async def run_repeat(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    env: dict,
    workload_rows: list[dict],
    workload_integrity: WorkloadIntegrity | None,
    sample_policy: str,
    gpu_handle,
    raw_writer: RawJsonlWriter,
    experiment_id: str,
    cell_id: str,
    concurrency: int,
    repeat_index: int,
) -> int:
    repeat_id = make_repeat_id(repeat_index)
    sample_offset = resolve_repeat_offset(
        len(workload_rows),
        args.num_requests,
        repeat_index,
        sample_policy=sample_policy,
    )

    # gpu_handle is None in server mode: VRAM is joined later from the
    # server's own NVML samples using the UTC window recorded below.
    stop_monitor = asyncio.Event()
    monitor_task = None
    if gpu_handle is not None:
        monitor_task = asyncio.create_task(
            monitor_vram(
                gpu_handle,
                stop_monitor,
                args.vram_sample_interval_s,
            )
        )

    started_at_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        records = await run_requests(
            session=session,
            args=args,
            env=env,
            workload_rows=workload_rows,
            workload_integrity=workload_integrity,
            sample_policy=sample_policy,
            count=args.num_requests,
            concurrency=concurrency,
            experiment_id=experiment_id,
            cell_id=cell_id,
            repeat_id=repeat_id,
            phase="measured",
            request_index_start=0,
            sample_offset=sample_offset,
        )
        duration = time.perf_counter() - start
    finally:
        finished_at_utc = datetime.now(timezone.utc).isoformat()
        stop_monitor.set()
        vram_peak_mb = await monitor_task if monitor_task is not None else None

    run_id = f"run-{uuid.uuid4()}"
    successes = sum(record["success"] for record in records)
    complete = sum(record["metrics_complete"] for record in records)
    output_tokens = sum(
        record["output_tokens"]
        for record in records
        if isinstance(record["output_tokens"], int)
    )

    for record in records:
        record["run_id"] = run_id
        record["run_duration_s"] = round(duration, 6)
        record["run_vram_peak_mb"] = vram_peak_mb
        record["run_started_at_utc"] = started_at_utc
        record["run_finished_at_utc"] = finished_at_utc

    raw_writer.write(records)

    print("\n===== M1-02 REPEAT RESULT =====")
    print(f"experiment_id       : {experiment_id}")
    print(f"cell_id             : {cell_id}")
    print(f"repeat_id           : {repeat_id}")
    print(f"sample_offset       : {sample_offset}")
    print(f"run_id              : {run_id}")
    print(f"requests            : {len(records)}")
    print(f"successes           : {successes}")
    print(f"metrics_complete    : {complete}")
    print(f"concurrency         : {concurrency}")
    print(f"duration_s          : {duration:.3f}")
    print(f"request_throughput  : {successes / duration:.3f} req/s")
    print(f"output_throughput   : {output_tokens / duration:.3f} tok/s")
    print(f"run_vram_peak_mb    : {vram_peak_mb}")
    print(f"raw_jsonl           : {raw_writer.path}")
    print("=================================")

    if successes != len(records):
        return 2
    if complete != len(records):
        return 3
    return 0


async def run_cell(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    env: dict,
    workload_rows: list[dict],
    workload_integrity: WorkloadIntegrity | None,
    sample_policy: str,
    gpu_handle,
    raw_writer: RawJsonlWriter,
    experiment_id: str,
    concurrency: int,
) -> int:
    cell_id = make_cell_id(args.lane, args.workload, concurrency)

    print("\n========================================")
    print("START CELL")
    print(f"experiment_id : {experiment_id}")
    print(f"cell_id       : {cell_id}")
    print(f"concurrency   : {concurrency}")
    print(f"repeats       : {args.repeats}")
    print("========================================")

    warmup_code = await warm_up_cell(
        session=session,
        args=args,
        env=env,
        workload_rows=workload_rows,
        workload_integrity=workload_integrity,
        sample_policy=sample_policy,
        experiment_id=experiment_id,
        cell_id=cell_id,
        concurrency=concurrency,
    )
    if warmup_code != 0:
        return warmup_code

    for repeat_index in range(1, args.repeats + 1):
        repeat_code = await run_repeat(
            session=session,
            args=args,
            env=env,
            workload_rows=workload_rows,
            workload_integrity=workload_integrity,
            sample_policy=sample_policy,
            gpu_handle=gpu_handle,
            raw_writer=raw_writer,
            experiment_id=experiment_id,
            cell_id=cell_id,
            concurrency=concurrency,
            repeat_index=repeat_index,
        )
        if repeat_code != 0:
            print(
                "CELL ABORTED: "
                f"{cell_id} failed at {make_repeat_id(repeat_index)} "
                f"with exit code {repeat_code}"
            )
            return repeat_code

    print("\n===== CELL PASS =====")
    print(f"cell_id              : {cell_id}")
    print(f"measured_repeats     : {args.repeats}")
    print("=====================")
    return 0


async def run_sweep(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    env: dict,
    workload_rows: list[dict],
    workload_integrity: WorkloadIntegrity | None,
    sample_policy: str,
    gpu_handle,
    raw_writer: RawJsonlWriter,
    experiment_id: str,
    concurrencies: list[int],
) -> int:
    print("\n===== M1-02 CONCURRENCY SWEEP =====")
    print(f"experiment_id        : {experiment_id}")
    print(f"lane                 : {args.lane}")
    print(f"workload             : {args.workload}")
    print(f"canonical_samples    : {len(workload_rows)}")
    print(f"sample_policy        : {sample_policy}")
    print(f"concurrencies        : {concurrencies}")
    print(f"repeats_per_cell     : {args.repeats}")
    print(f"requests_per_repeat  : {args.num_requests}")
    print("===================================")

    for concurrency in concurrencies:
        cell_code = await run_cell(
            session=session,
            args=args,
            env=env,
            workload_rows=workload_rows,
            workload_integrity=workload_integrity,
            sample_policy=sample_policy,
            gpu_handle=gpu_handle,
            raw_writer=raw_writer,
            experiment_id=experiment_id,
            concurrency=concurrency,
        )
        if cell_code != 0:
            print(
                "\nSWEEP ABORTED: "
                f"concurrency={concurrency} failed with exit code {cell_code}"
            )
            return cell_code

    print("\n===== SWEEP PASS =====")
    print(f"experiment_id        : {experiment_id}")
    print(f"cells_completed      : {len(concurrencies)}")
    print(f"concurrencies        : {concurrencies}")
    print(f"repeats_per_cell     : {args.repeats}")
    print("======================")
    return 0


def resolve_concurrencies(args: argparse.Namespace) -> list[int]:
    if args.concurrency is not None:
        return [args.concurrency]
    if args.concurrency_sweep is not None:
        return list(args.concurrency_sweep)
    return list(DEFAULT_CONCURRENCY_SWEEP)


async def async_main(args: argparse.Namespace) -> int:
    if args.workload_file is not None:
        try:
            workload_integrity = verify_workload_integrity(
                args.workload_file
            )
            workload_rows = load_workload_rows(args.workload_file)
            validate_manifest_consistency(
                workload_integrity,
                workload_rows,
            )
            args.workload = validate_workload_label(
                args.workload,
                workload_rows[0]["workload_id"],
            )
        except WorkloadContractError as exc:
            print("\n===== WORKLOAD CONTRACT FAILED =====")
            print(f"error                : {exc}")
            print("====================================")
            return 6

        sample_policy = args.sample_policy
        workload_mode = "canonical-jsonl"
    else:
        workload_rows = build_legacy_workload_rows(args)
        workload_integrity = None
        sample_policy = "cycle"
        workload_mode = "legacy-single-prompt"

    concurrencies = resolve_concurrencies(args)

    # Strict mode must fail before GPU/server work if either measured requests
    # or the largest cell warm-up would require implicit sample reuse.
    max_warmup_count = warmup_count(args, max(concurrencies))
    required_requests = max(args.num_requests, max_warmup_count)

    if (
        args.workload_file is not None
        and sample_policy == "strict"
        and required_requests > len(workload_rows)
    ):
        print("\n===== WORKLOAD EXPANSION POLICY REQUIRED =====")
        print(f"sample_count         : {len(workload_rows)}")
        print(f"requests_required    : {required_requests}")
        print(f"sample_policy        : {sample_policy}")
        print(
            "Canonical package reuse would be required. "
            "Do not silently cycle/random-sample."
        )
        print("PM DECISION REQUIRED before canonical reuse.")
        print("==============================================")
        return 6

    print("\n===== WORKLOAD INPUT =====")
    print(f"mode                 : {workload_mode}")
    print(f"workload             : {args.workload}")
    print(f"workload_file        : {args.workload_file}")
    print(f"sample_count         : {len(workload_rows)}")
    print(f"sample_policy        : {sample_policy}")
    print(f"package_workload_id  : {workload_rows[0]['workload_id']}")
    print(f"dataset_version      : {workload_rows[0]['dataset_version']}")
    print(f"prompt_version       : {workload_rows[0]['prompt_version']}")
    if workload_integrity is not None:
        print(f"manifest_path        : {workload_integrity.manifest_path}")
        print(f"artifact_key         : {workload_integrity.artifact_key}")
        print(f"workload_file_sha256 : {workload_integrity.actual_sha256}")
        print(
            "manifest_package_sha256: "
            f"{workload_integrity.manifest_package_sha256}"
        )
    print("==========================")

    if args.workload_file is not None and sample_policy == "cycle":
        print("\n===== NON-CANONICAL SAMPLE REUSE =====")
        print("policy               : deterministic-cycle")
        print(
            "mapping              : "
            "sample_index = request_index % sample_count"
        )
        print(
            "WARNING: simple cycle is retained for debug/backward compatibility; "
            "the PM-approved canonical reuse policy is cycle-rotate."
        )
        print("======================================")

    if args.workload_file is not None and sample_policy == "cycle-rotate":
        print("\n===== PM-APPROVED SAMPLE REUSE =====")
        print("policy               : deterministic-cycle-rotate")
        print(
            "repeat_offset        : "
            "((repeat_index - 1) * (num_requests % sample_count)) "
            "% sample_count"
        )
        print(
            "mapping              : "
            "sample_index = (request_index + repeat_offset) % sample_count"
        )
        print("lane_pairing         : exact same mapping for Vanilla and Static")
        print("warmup_offset        : 0")
        print("====================================")

    profile = args.environment_profile
    try:
        env, gpu_handle = environment_info(args.gpu_index, profile, args.server_env_json)
    except ServerEnvError as exc:
        print("\n===== ENVIRONMENT CONTRACT FAILED =====")
        print(f"profile              : {profile}")
        print(f"error                : {exc}")
        print("=======================================")
        return 5

    try:
        mismatches = validate_environment(env, profile)
        if mismatches:
            print("\n===== ENVIRONMENT CONTRACT FAILED =====")
            print(f"profile              : {profile}")
            for mismatch in mismatches:
                print(f"mismatch             : {mismatch}")
            print("=======================================")
            if not args.skip_env_check:
                return 5
            print("[CẢNH BÁO] Tiếp tục chạy vì có cờ --skip-env-check")
        else:
            print("\n===== ENVIRONMENT CONTRACT PASS =====")
            print(f"profile              : {profile}")
            print(f"environment_source   : {env.get('environment_source', 'local')}")
            print(f"gpu                  : {env['gpu']}")
            print(f"vllm_version         : {env['vllm_version']}")
            print(f"torch_version        : {env['torch_version']}")
            print(f"torch_cuda_version   : {env['torch_cuda_version']}")
            print("=====================================")

        ensure_summary(Path(args.summary_csv))

        experiment_id = make_experiment_id(args.experiment_id)
        raw_writer = RawJsonlWriter(Path(args.raw_out), args.overwrite)
        raw_writer.prepare()

        canonical_sweep = list(DEFAULT_CONCURRENCY_SWEEP)
        if concurrencies != canonical_sweep:
            print("\n===== NON-CANONICAL CONCURRENCY SET =====")
            print(f"selected             : {concurrencies}")
            print(f"canonical            : {canonical_sweep}")
            print("Do not use this run as complete M1 sweep evidence.")
            print("=========================================")

        if args.repeats < CANONICAL_MIN_REPEATS:
            print("\n===== NON-CANONICAL SMOKE MODE =====")
            print(
                f"repeats={args.repeats} is below canonical "
                f"M1 minimum {CANONICAL_MIN_REPEATS}."
            )
            print("Do not use this run as scientific M1 benchmark evidence.")
            print("====================================")

        timeout = aiohttp.ClientTimeout(total=args.timeout_s)
        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"

        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
        ) as session:
            try:
                return await run_sweep(
                    session=session,
                    args=args,
                    env=env,
                    workload_rows=workload_rows,
                    workload_integrity=workload_integrity,
                    sample_policy=sample_policy,
                    gpu_handle=gpu_handle,
                    raw_writer=raw_writer,
                    experiment_id=experiment_id,
                    concurrencies=concurrencies,
                )
            except WorkloadContractError as exc:
                print("\n===== WORKLOAD EXECUTION FAILED =====")
                print(f"error                : {exc}")
                print("=====================================")
                return 6
    finally:
        if gpu_handle is not None:
            nvmlShutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M1-02 vLLM benchmark harness with automatic concurrency sweep, "
            "cell-scoped Option-B warm-up, canonical workload replay, "
            "identity tracing, and measured repeats."
        )
    )

    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Trả lời đúng một câu: Hà Nội là thủ đô của nước nào?",
        help=(
            "Legacy smoke/debug prompt. Ignored when --workload-file is supplied."
        ),
    )
    parser.add_argument(
        "--workload-file",
        type=Path,
        default=None,
        help=(
            "Canonical workload chat-request JSONL. When supplied, "
            "messages/temperature/seed/max_tokens/response_format come from "
            "the frozen package rather than legacy CLI semantic fields."
        ),
    )
    parser.add_argument(
        "--sample-policy",
        choices=["strict", "cycle", "cycle-rotate"],
        default="strict",
        help=(
            "Mapping from request_index to canonical samples. 'strict' forbids "
            "sample reuse. 'cycle' is the legacy/debug zero-offset cycle. "
            "'cycle-rotate' is the PM-approved canonical reuse policy: each "
            "measured repeat uses deterministic rotating offset "
            "((repeat_index-1)*(num_requests%%sample_count))%%sample_count, "
            "with the exact same mapping across Vanilla and Static."
        ),
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=32,
        help=(
            "Measured requests per repeat. Must be >= every selected "
            "concurrency so the requested load can be exercised."
        ),
    )

    concurrency_group = parser.add_mutually_exclusive_group()
    concurrency_group.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Run exactly one concurrency cell. Useful for smoke/debug runs. "
            "If omitted, the default automatic sweep is 1 2 4 8 16 32."
        ),
    )
    concurrency_group.add_argument(
        "--concurrency-sweep",
        nargs="+",
        type=int,
        default=None,
        metavar="C",
        help=(
            "Explicit concurrency sweep. Example: "
            "--concurrency-sweep 1 2 4 8 16 32. "
            "If neither this nor --concurrency is supplied, the canonical "
            "default sweep 1 2 4 8 16 32 is used."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=CANONICAL_MIN_REPEATS,
        help=(
            "Measured repeats per cell. Canonical M1 protocol requires >=5. "
            "Smaller values are allowed only for smoke/debug runs and are "
            "explicitly marked non-canonical."
        ),
    )
    parser.add_argument("--warmup-policy", choices=("concurrency-matched", "m3-fixed3"),
                        default="concurrency-matched",
                        help="M3 PM lock: exactly three warm-ups; older protocols retain their default.")
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=CANONICAL_MIN_WARMUP_REQUESTS,
        help=(
            "Requested minimum warm-up request count. PM-approved Option B "
            "enforces max(3, --warmup-requests, current_cell_concurrency). "
            "Warm-up runs once per cell before all measured repeats."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help=(
            "Legacy smoke/debug token budget. Ignored when --workload-file "
            "is supplied."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Legacy smoke/debug temperature. Ignored when --workload-file "
            "is supplied."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Legacy smoke/debug seed. Ignored when --workload-file is supplied."
        ),
    )
    parser.add_argument(
        "--lane",
        choices=["vanilla", "static", "adaptive", "eagle", "eagle_fixed", "eagle_adaptive"],
        default="vanilla",
    )
    parser.add_argument(
        "--workload",
        default="smoke",
        help=(
            "Workload label used in cell identity. Canonical --workload-file "
            "runs require W1, W2, or W3 and validate the label against the "
            "package workload_id. Legacy smoke/debug runs may use other labels."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "Optional experiment execution identity shared by all cells "
            "and repeats in this invocation. If omitted, exp-<UUID4> is generated."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--environment-profile",
        choices=sorted(ENVIRONMENT_PROFILES),
        default=ENVIRONMENT_PROFILE,
        help=(
            "m1-t4-cu130-v1: vLLM on this machine (local NVML). "
            "m3-l4-modal-v1: remote Modal L4 server; requires --server-env-json "
            "and VRAM is joined afterwards from the server's samples."
        ),
    )
    parser.add_argument(
        "--server-env-json",
        type=Path,
        default=None,
        help="server_env.json written by the remote server container.",
    )
    parser.add_argument("--vram-sample-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--raw-out",
        default="experiments/runs/raw_test.jsonl",
    )
    parser.add_argument(
        "--summary-csv",
        default="experiments/results/summary.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Remove any stale raw JSONL once at experiment start, before warm-up; "
            "all measured repeats/cells then append to the same experiment file."
        ),
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help=(
            "Bỏ qua việc dừng chương trình nếu môi trường không khớp hoàn toàn "
            "(dùng khi thử nghiệm/debug)."
        ),
    )

    args = parser.parse_args()
    concurrencies = resolve_concurrencies(args)

    if (
        ENVIRONMENT_PROFILES[args.environment_profile]["mode"] == "server"
        and args.server_env_json is None
    ):
        parser.error(
            f"--environment-profile {args.environment_profile} requires --server-env-json"
        )

    if args.workload_file is not None and args.workload == "smoke":
        parser.error(
            "--workload must be explicitly set when --workload-file is used"
        )

    if args.num_requests < 1:
        parser.error("--num-requests must be >= 1")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be >= 0")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if not concurrencies:
        parser.error("at least one concurrency value is required")
    if any(concurrency < 1 for concurrency in concurrencies):
        parser.error("all concurrency values must be >= 1")
    if len(set(concurrencies)) != len(concurrencies):
        parser.error("concurrency sweep values must be unique")

    max_concurrency = max(concurrencies)
    if args.num_requests < max_concurrency:
        parser.error(
            "--num-requests must be >= the maximum selected concurrency "
            f"({max_concurrency})"
        )

    return args


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(async_main(parse_args())))
