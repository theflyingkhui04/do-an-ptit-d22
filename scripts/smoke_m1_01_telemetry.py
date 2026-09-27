"""Run M1-01 dual-lane smoke requests and emit normalized telemetry records."""

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
from typing import Any, AsyncIterator

import aiohttp

from check_m1_telemetry import normalize_record, validate_normalized_record
from jsonschema import Draft202012Validator


DEFAULT_SCHEMA = Path("experiments/configs/m1_request_telemetry.schema.json")
DEFAULT_SUMMARY = Path("experiments/results/m1_01_smoke_summary.json")
DEFAULT_PROMPTS = [
    "Trả lời đúng một câu: Speculative Decoding là gì?",
    "Tóm tắt trong một câu: vì sao cần đo TTFT và TPOT?",
    "Trả lời ngắn: GPU T4 thường có bao nhiêu GB VRAM?",
    "Viết một câu mô tả OpenAI-compatible chat completions.",
    "Trả lời một câu: alpha trong speculative decoding biểu thị điều gì?",
    "Trả lời một câu: mean acceptance length là gì?",
    "Nêu một lý do cần seed cố định khi benchmark.",
    "Giải thích ngắn: vanilla decoding khác static draft decoding thế nào?",
    "Trả lời một câu: vì sao không dùng smoke test để claim speedup?",
    "Nêu một câu về vai trò của NVML trong đo VRAM."
]
VRAM_PEAK_LIMIT_MB = 14336.0
VRAM_IDLE_DELTA_LIMIT_MB = 256.0


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


class NvmlSampler:
    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.available = False
        self._nvml = None
        self._handle = None
        self.gpu_name: str | None = None
        self.driver_version: str | None = None

    def __enter__(self) -> "NvmlSampler":
        try:
            from pynvml import (  # type: ignore
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetMemoryInfo,
                nvmlDeviceGetName,
                nvmlInit,
                nvmlSystemGetDriverVersion,
            )
        except Exception:
            return self
        self._nvml = {
            "get_memory": nvmlDeviceGetMemoryInfo,
        }
        nvmlInit()
        self._handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        name = nvmlDeviceGetName(self._handle)
        driver = nvmlSystemGetDriverVersion()
        self.gpu_name = name.decode() if isinstance(name, bytes) else str(name)
        self.driver_version = driver.decode() if isinstance(driver, bytes) else str(driver)
        self.available = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self.available:
            return
        try:
            from pynvml import nvmlShutdown  # type: ignore

            nvmlShutdown()
        except Exception:
            pass

    def used_mb(self) -> float | None:
        if not self.available or self._handle is None or self._nvml is None:
            return None
        mem = self._nvml["get_memory"](self._handle)
        return round(mem.used / 1024**2, 3)


async def monitor_vram(sampler: NvmlSampler, stop: asyncio.Event, interval_s: float) -> float | None:
    peak = sampler.used_mb()
    while not stop.is_set():
        current = sampler.used_mb()
        if current is not None:
            peak = current if peak is None else max(peak, current)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
    current = sampler.used_mb()
    if current is not None:
        peak = current if peak is None else max(peak, current)
    return peak


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


def runtime_info(sampler: NvmlSampler) -> dict[str, Any]:
    return {
        "vllm_version": package_version("vllm"),
        "torch_version": package_version("torch"),
        "aiohttp_version": aiohttp.__version__,
        "driver_version": sampler.driver_version,
        "gpu_name": sampler.gpu_name,
        "git_commit": git_commit(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def prompt_for_index(index: int) -> str:
    return DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)]


async def run_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    index: int,
    sampler: NvmlSampler,
    runtime: dict[str, Any],
    idle_after_warmup_mb: float | None,
) -> dict[str, Any]:
    client_request_id = f"m1-01-{args.lane}-{index + 1:04d}-{uuid.uuid4().hex[:8]}"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_for_index(index)}],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
    }

    start = time.perf_counter()
    first_token_at: float | None = None
    last_event_at = start
    server_request_id: str | None = None
    output_tokens: int | None = None
    final_metrics: dict[str, Any] | None = None
    text_parts: list[str] = []
    vram_before_mb = sampler.used_mb()
    vram_peak_mb = vram_before_mb

    async with session.post(args.endpoint, json=payload) as response:
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {text[:1000]}")
        async for message in iter_sse(response):
            if message == "[DONE]":
                break
            event_at = time.perf_counter()
            data = json.loads(message)
            server_request_id = server_request_id or data.get("id")
            choices = data.get("choices")
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if first_token_at is None:
                        first_token_at = event_at
                    text_parts.append(content)
            usage = data.get("usage")
            if isinstance(usage, dict):
                output_tokens = usage.get("completion_tokens")
                final_metrics = data.get("metrics")
            current_vram = sampler.used_mb()
            if current_vram is not None:
                vram_peak_mb = current_vram if vram_peak_mb is None else max(vram_peak_mb, current_vram)
            last_event_at = event_at

    if first_token_at is None:
        raise RuntimeError("No streamed token event received")
    if not isinstance(output_tokens, int) or output_tokens < 1:
        raise RuntimeError("Final usage chunk did not include completion_tokens")

    e2e_ms = (last_event_at - start) * 1000
    ttft_ms = (first_token_at - start) * 1000
    tpot_ms = 0.0 if output_tokens == 1 else (e2e_ms - ttft_ms) / (output_tokens - 1)
    vram_after_mb = sampler.used_mb()
    if vram_after_mb is not None:
        vram_peak_mb = vram_after_mb if vram_peak_mb is None else max(vram_peak_mb, vram_after_mb)
    if vram_peak_mb is None:
        vram_peak_mb = 0.0
    warmup_baseline = idle_after_warmup_mb if idle_after_warmup_mb is not None else 0.0
    after_smoke = vram_after_mb if vram_after_mb is not None else warmup_baseline
    idle_delta = after_smoke - warmup_baseline

    return normalize_record(
        client_request_id=client_request_id,
        server_request_id=server_request_id,
        lane=args.lane,
        model=args.model,
        server_metrics=final_metrics,
        client_metrics={
            "ttft_ms": round(ttft_ms, 3),
            "tpot_ms": round(tpot_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
            "output_tokens": output_tokens,
        },
        gpu_metrics={
            "gpu_index": args.gpu_index,
            "vram_peak_mb": round(vram_peak_mb, 3),
            "vram_idle_before_warmup_mb": round(args.vram_idle_before_warmup_mb, 3),
            "vram_idle_after_warmup_mb": round(warmup_baseline, 3),
            "vram_idle_after_smoke_mb": round(after_smoke, 3),
            "vram_idle_delta_mb": round(idle_delta, 3),
            "vram_peak_limit_mb": args.vram_peak_limit_mb,
            "vram_peak_pass": vram_peak_mb <= args.vram_peak_limit_mb,
            "vram_stability_pass": idle_delta <= args.vram_idle_delta_limit_mb,
        },
        runtime=runtime,
    )


async def run_smoke(args: argparse.Namespace) -> int:
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    with NvmlSampler(args.gpu_index) as sampler:
        args.vram_idle_before_warmup_mb = sampler.used_mb() or 0.0
        timeout = aiohttp.ClientTimeout(total=args.timeout_s)
        runtime = runtime_info(sampler)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            for index in range(args.warmup):
                await run_one(session, args, index, sampler, runtime, args.vram_idle_before_warmup_mb)
            idle_after_warmup_mb = sampler.used_mb() or args.vram_idle_before_warmup_mb
            stop = asyncio.Event()
            monitor_task = asyncio.create_task(monitor_vram(sampler, stop, args.vram_sample_interval_s))
            records: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            try:
                for index in range(args.requests):
                    try:
                        record = await run_one(
                            session,
                            args,
                            index,
                            sampler,
                            runtime,
                            idle_after_warmup_mb,
                        )
                        errors = validate_normalized_record(record, validator)
                        if errors:
                            failures.append({"index": index + 1, "errors": errors})
                        records.append(record)
                    except Exception as exc:
                        failures.append(
                            {
                                "index": index + 1,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
            finally:
                stop.set()
                sampled_peak = await monitor_task

    with args.raw_out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    alpha_values = [record["alpha"] for record in records if isinstance(record.get("alpha"), (int, float))]
    tau_values = [record["tau"] for record in records if isinstance(record.get("tau"), (int, float))]
    vram_values = [
        record.get("gpu_metrics", {}).get("vram_peak_mb")
        for record in records
        if isinstance(record.get("gpu_metrics", {}).get("vram_peak_mb"), (int, float))
    ]
    if sampled_peak is not None:
        vram_values.append(sampled_peak)
    lane_summary = {
        "task": "M1-01",
        "step": "dual-lane-telemetry-smoke",
        "lane": args.lane,
        "endpoint": args.endpoint,
        "model": args.model,
        "schema_version": "m1_request_telemetry.v1",
        "request_count": args.requests,
        "success_count": len(records),
        "failure_count": len(failures),
        "telemetry_contract_pass": not failures and len(records) == args.requests,
        "alpha_min": min(alpha_values) if alpha_values else None,
        "alpha_max": max(alpha_values) if alpha_values else None,
        "tau_min": min(tau_values) if tau_values else None,
        "tau_max": max(tau_values) if tau_values else None,
        "vram_peak_mb": max(vram_values) if vram_values else None,
        "vram_peak_limit_mb": args.vram_peak_limit_mb,
        "raw_out": args.raw_out.as_posix(),
        "runtime": runtime,
        "failures": failures,
    }
    if args.summary.exists():
        try:
            summary = json.loads(args.summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    if not isinstance(summary, dict):
        summary = {}
    lanes = summary.get("lanes")
    if not isinstance(lanes, dict):
        lanes = {}
    lanes[args.lane] = lane_summary
    completed_lanes = sorted(lane for lane, data in lanes.items() if isinstance(data, dict))
    handoff_ready = (
        {"vanilla", "static"}.issubset(completed_lanes)
        and all(isinstance(lanes[lane], dict) and lanes[lane].get("telemetry_contract_pass") for lane in ("vanilla", "static"))
    )
    summary = {
        "task": "M1-01",
        "step": "dual-lane-telemetry-smoke",
        "schema_version": "m1_request_telemetry.v1",
        "status": "PASS" if handoff_ready else "PARTIAL",
        "completed_lanes": completed_lanes,
        "handoff_ready": handoff_ready,
        "lanes": lanes,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if lane_summary["telemetry_contract_pass"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=["vanilla", "static"], required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--vram-sample-interval-s", type=float, default=0.05)
    parser.add_argument("--vram-peak-limit-mb", type=float, default=VRAM_PEAK_LIMIT_MB)
    parser.add_argument("--vram-idle-delta-limit-mb", type=float, default=VRAM_IDLE_DELTA_LIMIT_MB)
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_smoke(parse_args())))
