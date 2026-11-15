"""Measurements of the web console (page "Đo và so sánh").

* ``speed``: requests of a frozen workload package (W1 = w1_v3, W2 = w2_v2, W3 = w3_v2) are sent to every chosen
  endpoint at once, at each load (number of simultaneous users). Per request: TTFT, TPOT, latency, tokens.
  Speed-up follows the campaign definition: baseline TPOT p50 / endpoint TPOT p50 at the same load.
* ``chain``: one devset image through OCR, then on each endpoint Node 1 → Node 2 (Q1–Q6) → Node 3, with the time
  of each node and whether the outputs equal the baseline's.

Numbers measured here are live and noisy; the web labels them apart from the official campaign numbers.
One measurement runs at a time; the backend may be restarted by Render, which marks the run "interrupted".
"""

from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from src.api.compare import compare_fields

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_FILES = {"W1": "experiments/configs/w1_v3/node1_chat_requests.jsonl",
                  "W2": "experiments/configs/w2_v2/chat_requests.jsonl",
                  "W3": "experiments/configs/w3_v2/chat_requests.jsonl"}
DEVSET_SAMPLES = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
REQUEST_TIMEOUT_S = 600.0


def load_requests(workload: str, n: int, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """First ``n`` requests of the frozen package (cycled when ``n`` is larger), without the ``model`` field."""
    if workload not in WORKLOAD_FILES:
        raise ValueError(f"unknown workload {workload!r}")
    rows = [json.loads(line) for line in (repo_root / WORKLOAD_FILES[workload]).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    requests = [{k: v for k, v in row["request"].items() if k != "model"} for row in rows]
    return [requests[i % len(requests)] for i in range(n)]


def _p50(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize(samples: list[dict[str, Any]], wall_ms: float) -> dict[str, Any]:
    ok = [s for s in samples if s.get("ok")]
    tokens = sum(s.get("completion_tokens") or 0 for s in ok)
    return {"n": len(samples), "n_ok": len(ok),
            "ttft_ms_p50": _p50([s["ttft_ms"] for s in ok if s.get("ttft_ms") is not None]),
            "tpot_ms_p50": _p50([s["tpot_ms"] for s in ok if s.get("tpot_ms") is not None]),
            "latency_ms_p50": _p50([s["latency_ms"] for s in ok if s.get("latency_ms") is not None]),
            "throughput_tps": tokens / (wall_ms / 1000) if wall_ms > 0 else None,
            "errors": [s["error"] for s in samples if not s.get("ok")][:5]}


def speedups(cells: list[dict[str, Any]], baseline_id: str | None) -> None:
    """Add ``speedup_vs_baseline`` = baseline TPOT p50 / endpoint TPOT p50 at the same load (None when unknown)."""
    base = {c["load"]: c.get("tpot_ms_p50") for c in cells if c["endpoint_id"] == baseline_id}
    for cell in cells:
        ref, mine = base.get(cell["load"]), cell.get("tpot_ms_p50")
        cell["speedup_vs_baseline"] = ref / mine if ref and mine else None


async def measure_one(client: httpx.AsyncClient, connection: dict[str, Any], request: dict[str, Any],
                      clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    body = {**request, "model": connection["model_id"], "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Accept": "text/event-stream"}
    if connection.get("api_key"):
        headers["Authorization"] = f"Bearer {connection['api_key']}"
    start = clock()
    ttft = usage = None
    try:
        async with client.stream("POST", f"{connection['base_url']}/chat/completions", json=body, headers=headers,
                                 timeout=REQUEST_TIMEOUT_S) as response:
            if response.status_code != 200:
                await response.aread()
                return {"ok": False, "error": f"HTTP {response.status_code}"}
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if ttft is None and any((c.get("delta") or {}).get("content") for c in chunk.get("choices") or []):
                    ttft = clock() - start
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": type(exc).__name__}
    total = clock() - start
    tokens = (usage or {}).get("completion_tokens")
    tpot = (total - ttft) * 1000 / (tokens - 1) if ttft is not None and tokens and tokens > 1 else None
    return {"ok": True, "ttft_ms": None if ttft is None else ttft * 1000, "tpot_ms": tpot, "latency_ms": total * 1000,
            "completion_tokens": tokens}


class BenchService:
    def __init__(self, store: Any, registry: Any, *, pipeline: Any = None, reports: Any = None,
                 repo_root: Path = REPO_ROOT, async_client: Callable[[], httpx.AsyncClient] | None = None,
                 spawn: Callable[..., Any] | None = None, new_id: Callable[[], str] | None = None,
                 clock: Callable[[], float] = time.perf_counter, redact: Callable[[str], str] = lambda s: s) -> None:
        self.store = store
        self.registry = registry
        self.pipeline = pipeline
        self.reports = reports
        self.repo_root = repo_root
        self._async_client = async_client or (lambda: httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_S,
                                                                                              connect=30.0)))
        self._spawn = spawn or (lambda target, args: threading.Thread(target=target, args=args, daemon=True).start())
        self._new_id = new_id or (lambda: f"bn_{uuid.uuid4().hex[:12]}")
        self._clock = clock
        self._redact = redact
        self._lock = threading.Lock()
        self._active: str | None = None

    def start(self, config: dict[str, Any]) -> str:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Đang có một lần đo chạy; chờ lần đó xong rồi đo tiếp.")
            run_id = self._new_id()
            self._active = run_id
        self.store.create_bench(run_id, config["kind"], config)
        self._spawn(self._execute, (run_id, config))
        return run_id

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.store.get_bench(run_id)

    def list(self) -> list[dict[str, Any]]:
        return self.store.list_bench()

    def _execute(self, run_id: str, config: dict[str, Any]) -> None:
        try:
            self.store.update_bench(run_id, "running", progress={"done": 0, "total": 0})
            run = self._speed if config["kind"] == "speed" else self._chain
            progress, result = run(run_id, config)
            self.store.update_bench(run_id, "done", progress=progress, result=result)
        except Exception as exc:  # noqa: BLE001  (background run: recorded on the run, never raised)
            self.store.update_bench(run_id, "failed", error=self._redact(f"{type(exc).__name__}: {exc}")[:500])
        finally:
            with self._lock:
                self._active = None

    def _baseline_in(self, ids: list[str]) -> str | None:
        baseline = self.registry.baseline()
        return baseline["endpoint_id"] if baseline and baseline["endpoint_id"] in ids else None

    # ---------------------------------------------------------------- speed
    def _speed(self, run_id: str, config: dict[str, Any]) -> tuple[dict, dict]:
        ids = list(config["endpoint_ids"])
        connections = [self.registry.get(i, with_key=True) for i in ids]
        requests = load_requests(config["workload"], config["requests"], self.repo_root)
        loads = sorted(set(config["loads"]))
        total = len(ids) * len(loads) * config["repeats"] * len(requests)
        progress = {"done": 0, "total": total}
        samples: dict[tuple[str, int], list[dict]] = {}
        walls: dict[tuple[str, int], float] = {}

        async def one_endpoint(client: httpx.AsyncClient, conn: dict, load: int) -> tuple[list[dict], float]:
            gate = asyncio.Semaphore(load)

            async def guarded(request: dict) -> dict:
                async with gate:
                    result = await measure_one(client, conn, request, self._clock)
                progress["done"] += 1
                return result

            started = self._clock()
            results = await asyncio.gather(*(guarded(r) for r in requests))
            return list(results), (self._clock() - started) * 1000

        async def main() -> None:
            async with self._async_client() as client:
                for load in loads:
                    for _ in range(config["repeats"]):
                        outcomes = await asyncio.gather(*(one_endpoint(client, c, load) for c in connections))
                        for conn, (results, wall) in zip(connections, outcomes):
                            key = (conn["endpoint_id"], load)
                            samples.setdefault(key, []).extend(results)
                            walls[key] = walls.get(key, 0.0) + wall
                        self.store.update_bench(run_id, "running", progress=dict(progress))

        asyncio.run(main())
        names = {c["endpoint_id"]: c["name"] for c in connections}
        cells = [{"endpoint_id": eid, "name": names[eid], "load": load, **summarize(samples[(eid, load)], walls[(eid, load)])}
                 for load in loads for eid in ids]
        baseline_id = self._baseline_in(ids)
        speedups(cells, baseline_id)
        result = {"kind": "speed", "live": True, "workload": config["workload"], "loads": loads,
                  "requests": config["requests"], "repeats": config["repeats"], "baseline": baseline_id, "cells": cells}
        return progress, result

    # ---------------------------------------------------------------- chain 1 -> 2 -> 3
    def _sample_image(self, sample_id: str) -> bytes:
        for line in (self.repo_root / DEVSET_SAMPLES).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["sample_id"] == sample_id:
                return (self.repo_root / row["image_path"]).read_bytes()
        raise ValueError(f"unknown devset sample {sample_id!r}")

    def _chain_one(self, run_id: str, text: str, endpoint: dict) -> dict[str, Any]:
        started = self._clock()
        node1 = self.pipeline.run_endpoint(run_id, f"chain/{run_id}", text, endpoint)
        node1_ms = (self._clock() - started) * 1000
        report = self.reports.run(self.registry.caller(endpoint["endpoint_id"]))
        timings = {"node1": node1_ms, "node2": report["timings_ms"]["node2"], "node3": report["timings_ms"]["node3"]}
        payload = node1["success"].canonical_payload if node1["success"] else None
        return {"endpoint_id": endpoint["endpoint_id"], "name": endpoint["name"], "timings_ms": timings,
                "total_ms": sum(timings.values()), "node1": node1["report"], "invoice": payload,
                "questions": report["questions"], "report": report["report"], "number_check": report["number_check"]}

    def _chain(self, run_id: str, config: dict[str, Any]) -> tuple[dict, dict]:
        ids = list(config["endpoint_ids"])
        endpoints = [self.registry.get(i) for i in ids]
        sample_id = config.get("sample_id") or "m0_04_001"
        ocr_result: dict[str, Any] = {"timings_ms": {}}
        text = self.pipeline.read_text(self._sample_image(sample_id), ocr_result)
        progress = {"done": 0, "total": len(ids)}
        with ThreadPoolExecutor(max_workers=len(ids)) as pool:
            rows = list(pool.map(lambda e: self._chain_one(run_id, text, e), endpoints))
        progress["done"] = len(ids)
        baseline_id = self._baseline_in(ids)
        base = next((r for r in rows if r["endpoint_id"] == baseline_id), None)
        for row in rows:
            row["same_as_baseline"] = None
            if base is None or row is base:
                continue
            fields = compare_fields(row["invoice"], base["invoice"]) if row["invoice"] and base["invoice"] else None
            row["same_as_baseline"] = {
                "node1": bool(fields) and not fields["differing_fields"],
                "node2": [q.get("rows") for q in row["questions"]] == [q.get("rows") for q in base["questions"]],
                "node3": row["report"].get("text") == base["report"].get("text")}
        result = {"kind": "chain", "live": True, "sample_id": sample_id, "ocr_ms": ocr_result["timings_ms"].get("ocr"),
                  "baseline": baseline_id, "endpoints": rows}
        return progress, result
