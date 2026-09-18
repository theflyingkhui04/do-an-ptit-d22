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

EXPECTED_VLLM = "0.29.0"

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


def ensure_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(",".join(SUMMARY_COLUMNS) + "\n", encoding="utf-8")


async def iter_sse(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
    """Yield `data:` payloads from an SSE response."""
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
    index: int,
) -> dict:
    queued_at = time.perf_counter()

    record = {
        "client_request_id": f"bench-{uuid.uuid4().hex[:12]}",
        "server_request_id": None,
        "request_index": index,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lane": args.lane,
        "workload": args.workload,
        "concurrency": args.concurrency,
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
        "messages": [{"role": "user", "content": args.prompt}],
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
                (time.perf_counter() - queued_at) * 1000, 3
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
                                (timestamp - start) * 1000, 3
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

                    # Match vLLM benchmark timing semantics:
                    # final usage event is part of end-to-end response latency.
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
                    (record["e2e_ms"] - record["ttft_ms"]) / (output_tokens - 1),
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
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    return await asyncio.gather(
        *[
            run_one(session, semaphore, args, env, index)
            for index in range(count)
        ]
    )


async def async_main(args: argparse.Namespace) -> int:
    env, gpu_handle = environment_info(args.gpu_index)
    ensure_summary(Path(args.summary_csv))

    if env["vllm_version"] != EXPECTED_VLLM:
        print(
            f"WARNING: expected vLLM {EXPECTED_VLLM}, "
            f"found {env['vllm_version']}"
        )

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    headers = {"Content-Type": "application/json"}

    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            # Warm-up is not written to raw JSONL.
            for i in range(args.warmup_requests):
                warm = await run_requests(session, args, env, 1, 1)
                if not warm[0]["success"]:
                    print(f"Warm-up {i + 1} failed: {warm[0]['error_message']}")
                    return 4

            stop_monitor = asyncio.Event()
            monitor_task = asyncio.create_task(
                monitor_vram(
                    gpu_handle,
                    stop_monitor,
                    args.vram_sample_interval_s,
                )
            )

            start = time.perf_counter()
            records = await run_requests(
                session,
                args,
                env,
                args.num_requests,
                args.concurrency,
            )
            duration = time.perf_counter() - start

            stop_monitor.set()
            vram_peak_mb = await monitor_task

    finally:
        nvmlShutdown()

    run_id = f"run-{uuid.uuid4().hex[:12]}"
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

    raw_path = Path(args.raw_out)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w" if args.overwrite else "a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n===== M0-03 BENCHMARK RESULT =====")
    print(f"run_id              : {run_id}")
    print(f"requests            : {len(records)}")
    print(f"successes           : {successes}")
    print(f"metrics_complete    : {complete}")
    print(f"concurrency         : {args.concurrency}")
    print(f"duration_s          : {duration:.3f}")
    print(f"request_throughput  : {successes / duration:.3f} req/s")
    print(f"output_throughput   : {output_tokens / duration:.3f} tok/s")
    print(f"run_vram_peak_mb    : {vram_peak_mb}")
    print(f"raw_jsonl           : {raw_path}")
    print("===================================")

    if successes != len(records):
        return 2
    if complete != len(records):
        return 3
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0-03 vLLM benchmark harness")

    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Trả lời đúng một câu: Hà Nội là thủ đô của nước nào?",
    )
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lane",
        choices=["vanilla", "static", "adaptive"],
        default="vanilla",
    )
    parser.add_argument("--workload", default="smoke")
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
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.num_requests < 1:
        parser.error("--num-requests must be >= 1")
    if not 1 <= args.concurrency <= args.num_requests:
        parser.error("--concurrency must be between 1 and --num-requests")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be >= 0")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")

    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))