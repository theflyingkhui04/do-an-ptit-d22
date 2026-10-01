from __future__ import annotations

import argparse
import asyncio
import importlib.metadata as metadata
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import aiohttp
import torch
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
    nvmlShutdown,
    nvmlSystemGetDriverVersion,
)

ENVIRONMENT_PROFILE = "m1-t4-cu130-v1"

EXPECTED_GPU = "Tesla T4"
EXPECTED_VLLM = "0.29.0"
EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_TORCH_CUDA = "13.0"

DEFAULT_CONCURRENCY_SWEEP = (1, 2, 4, 8, 16, 32)
CANONICAL_MIN_REPEATS = 5
CANONICAL_MIN_WARMUP_REQUESTS = 3
WORKLOAD_CONFIG_DIR = Path("experiments/configs/workloads")
DEFAULT_WORKLOAD_FILES = {
    "w1": WORKLOAD_CONFIG_DIR / "w1_vat_extractor.jsonl",
    "w1_vat_extractor": WORKLOAD_CONFIG_DIR / "w1_vat_extractor.jsonl",
    "w2": WORKLOAD_CONFIG_DIR / "w2_text_to_sql.jsonl",
    "w2_text_to_sql": WORKLOAD_CONFIG_DIR / "w2_text_to_sql.jsonl",
    "w3": WORKLOAD_CONFIG_DIR / "w3_free_analysis.jsonl",
    "w3_free_analysis": WORKLOAD_CONFIG_DIR / "w3_free_analysis.jsonl",
}

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


def environment_info(gpu_index: int) -> tuple[dict, object]:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    name = nvmlDeviceGetName(handle)
    driver = nvmlSystemGetDriverVersion()
    mem = nvmlDeviceGetMemoryInfo(handle)

    info = {
        "environment_profile": ENVIRONMENT_PROFILE,
        "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "gpu_index": gpu_index,
        "gpu_memory_total_mb": round(mem.total / 1024**2, 2),
        "driver_version": driver.decode() if isinstance(driver, bytes) else str(driver),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": package_version("vllm"),
        "aiohttp_version": aiohttp.__version__,
        "commit": git_commit(),
    }
    return info, handle


def validate_environment(env: dict) -> list[str]:
    expected = {
        "gpu": EXPECTED_GPU,
        "vllm_version": EXPECTED_VLLM,
        "torch_version": EXPECTED_TORCH,
        "torch_cuda_version": EXPECTED_TORCH_CUDA,
    }

    mismatches: list[str] = []
    for field, expected_value in expected.items():
        actual_value = env.get(field)
        if actual_value != expected_value:
            mismatches.append(
                f"{field}: expected {expected_value}, got {actual_value}"
            )
    return mismatches


def ensure_summary(path: Path) -> None:
    """Create the canonical summary header if missing.

    Aggregation into summary.csv is intentionally not implemented in this
    increment; raw measured JSONL remains the output evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(",".join(SUMMARY_COLUMNS) + "\n", encoding="utf-8")


def load_prompt_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: missing non-empty prompt")
            prompt_id = row.get("id") or f"{path.stem}-{line_number:03d}"
            if not isinstance(prompt_id, str):
                raise ValueError(f"{path}:{line_number}: id must be a string")
            records.append(
                {
                    "id": prompt_id,
                    "prompt": prompt,
                    "workload": row.get("workload"),
                    "source_path": path.as_posix(),
                }
            )
    if not records:
        raise ValueError(f"{path}: no prompt records found")
    return records


def resolve_prompt_records(args: argparse.Namespace) -> list[dict]:
    prompt_file = Path(args.prompt_file) if args.prompt_file else None
    if prompt_file is None:
        prompt_file = DEFAULT_WORKLOAD_FILES.get(args.workload.lower())
    if prompt_file is not None:
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt file not found for workload {args.workload}: {prompt_file}"
            )
        return load_prompt_records(prompt_file)
    return [
        {
            "id": "inline_prompt",
            "prompt": args.prompt,
            "workload": args.workload,
            "source_path": None,
        }
    ]


def prompt_record_for_request(args: argparse.Namespace, request_index: int) -> dict:
    records = args.prompt_records
    return records[request_index % len(records)]


def make_experiment_id(configured_id: str | None) -> str:
    if configured_id:
        return configured_id
    return f"exp-{uuid.uuid4()}"


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


async def run_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    env: dict,
    identity: dict,
    concurrency: int,
) -> dict:
    queued_at = time.perf_counter()
    prompt_record = prompt_record_for_request(args, identity["request_index"])

    record = {
        **identity,
        "server_request_id": None,
        "prompt_id": prompt_record["id"],
        "prompt_index": identity["request_index"] % len(args.prompt_records),
        "prompt_source": prompt_record["source_path"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lane": args.lane,
        "workload": args.workload,
        "concurrency": concurrency,
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
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

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_record["prompt"]}],
        "temperature": args.temperature,
        "seed": args.seed,
        "max_completion_tokens": args.max_tokens,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
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
    count: int,
    concurrency: int,
    experiment_id: str,
    cell_id: str,
    repeat_id: str | None,
    phase: str,
    request_index_start: int = 0,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    identities = [
        make_request_identity(
            experiment_id=experiment_id,
            cell_id=cell_id,
            repeat_id=repeat_id,
            request_index=request_index_start + offset,
            phase=phase,
        )
        for offset in range(count)
    ]

    return await asyncio.gather(
        *[
            run_one(
                session=session,
                semaphore=semaphore,
                args=args,
                env=env,
                identity=identity,
                concurrency=concurrency,
            )
            for identity in identities
        ]
    )


async def warm_up_cell(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    env: dict,
    experiment_id: str,
    cell_id: str,
    concurrency: int,
) -> int:
    effective_warmup_count = max(
        CANONICAL_MIN_WARMUP_REQUESTS,
        args.warmup_requests,
        concurrency,
    )

    print("\n===== M1-02 CELL WARM-UP =====")
    print("policy               : option-b-concurrency-matched")
    print(f"experiment_id        : {experiment_id}")
    print(f"cell_id              : {cell_id}")
    print(f"requested_minimum    : {args.warmup_requests}")
    print(f"warmup_requests      : {effective_warmup_count}")
    print(f"warmup_concurrency   : {concurrency}")
    print(f"measured_concurrency : {concurrency}")
    print("==============================")

    warm = await run_requests(
        session=session,
        args=args,
        env=env,
        count=effective_warmup_count,
        concurrency=concurrency,
        experiment_id=experiment_id,
        cell_id=cell_id,
        repeat_id=None,
        phase="warmup",
        request_index_start=0,
    )

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
    gpu_handle,
    raw_writer: RawJsonlWriter,
    experiment_id: str,
    cell_id: str,
    concurrency: int,
    repeat_index: int,
) -> int:
    repeat_id = make_repeat_id(repeat_index)

    stop_monitor = asyncio.Event()
    monitor_task = asyncio.create_task(
        monitor_vram(
            gpu_handle,
            stop_monitor,
            args.vram_sample_interval_s,
        )
    )

    start = time.perf_counter()
    try:
        records = await run_requests(
            session=session,
            args=args,
            env=env,
            count=args.num_requests,
            concurrency=concurrency,
            experiment_id=experiment_id,
            cell_id=cell_id,
            repeat_id=repeat_id,
            phase="measured",
            request_index_start=0,
        )
        duration = time.perf_counter() - start
    finally:
        stop_monitor.set()
        vram_peak_mb = await monitor_task

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

    raw_writer.write(records)

    print("\n===== M1-02 REPEAT RESULT =====")
    print(f"experiment_id       : {experiment_id}")
    print(f"cell_id             : {cell_id}")
    print(f"repeat_id           : {repeat_id}")
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
    gpu_handle,
    raw_writer: RawJsonlWriter,
    experiment_id: str,
    concurrencies: list[int],
) -> int:
    print("\n===== M1-02 CONCURRENCY SWEEP =====")
    print(f"experiment_id        : {experiment_id}")
    print(f"lane                 : {args.lane}")
    print(f"workload             : {args.workload}")
    print(f"concurrencies        : {concurrencies}")
    print(f"repeats_per_cell     : {args.repeats}")
    print(f"requests_per_repeat  : {args.num_requests}")
    print("===================================")

    for concurrency in concurrencies:
        cell_code = await run_cell(
            session=session,
            args=args,
            env=env,
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
    env, gpu_handle = environment_info(args.gpu_index)

    try:
        mismatches = validate_environment(env)
        if mismatches:
            print("\n===== ENVIRONMENT CONTRACT FAILED =====")
            print(f"profile              : {ENVIRONMENT_PROFILE}")
            for mismatch in mismatches:
                print(f"mismatch             : {mismatch}")
            print("=======================================")
            if args.strict_env:
                return 5
            print("WARNING: continuing because --strict-env was not set.")

        print("\n===== ENVIRONMENT CONTRACT PASS =====")
        print(f"profile              : {ENVIRONMENT_PROFILE}")
        print(f"gpu                  : {env['gpu']}")
        print(f"vllm_version         : {env['vllm_version']}")
        print(f"torch_version        : {env['torch_version']}")
        print(f"torch_cuda_version   : {env['torch_cuda_version']}")
        print("=====================================")

        ensure_summary(Path(args.summary_csv))

        concurrencies = resolve_concurrencies(args)
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
            return await run_sweep(
                session=session,
                args=args,
                env=env,
                gpu_handle=gpu_handle,
                raw_writer=raw_writer,
                experiment_id=experiment_id,
                concurrencies=concurrencies,
            )
    finally:
        nvmlShutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M1-02 vLLM benchmark harness with automatic concurrency sweep, "
            "cell-scoped Option-B warm-up, identity tracing, and measured repeats."
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
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "JSONL workload prompt file. If omitted, known workload names "
            "w2/w2_text_to_sql and w3/w3_free_analysis resolve to "
            "experiments/configs/workloads/*.jsonl. Falls back to --prompt "
            "for other workloads."
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
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=CANONICAL_MIN_WARMUP_REQUESTS,
        help=(
            "Requested minimum warm-up request count. "
            "PM-approved Option B enforces "
            "max(3, --warmup-requests, current_cell_concurrency). "
            "Warm-up runs once per cell before all measured repeats."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lane",
        choices=["vanilla", "static", "adaptive"],
        default="vanilla",
    )
    parser.add_argument("--workload", default="smoke")
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
        "--strict-env",
        action="store_true",
        help=(
            "Abort on canonical M1-02 environment mismatch. By default the "
            "harness warns and continues so the M1-01 Colab vLLM 0.30 runtime "
            "can run W2/W3 continuation measurements."
        ),
    )

    args = parser.parse_args()
    concurrencies = resolve_concurrencies(args)
    try:
        args.prompt_records = resolve_prompt_records(args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

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
    raise SystemExit(asyncio.run(async_main(parse_args())))
