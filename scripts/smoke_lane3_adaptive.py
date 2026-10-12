"""Smoke test for Lane 3 (AdaptiveAsyncScheduler) — M2-02 checklist 4.

A T4 cannot host two lanes at once, so the test runs in two passes:

  # Pass 1 — Vanilla reference (lane 1, port 8001)
  bash scripts/switch_lane.sh lane1
  python scripts/smoke_lane3_adaptive.py reference --port 8001 \
      --model qwen2.5-7b-vanilla --out experiments/runs/m2_02/smoke_vanilla_outputs.json

  # Pass 2 — Adaptive lane (port 8003), compared against the reference
  bash scripts/switch_lane.sh lane3
  python scripts/smoke_lane3_adaptive.py adaptive --port 8003 \
      --model qwen2.5-7b-adaptive \
      --reference experiments/runs/m2_02/smoke_vanilla_outputs.json \
      --telemetry experiments/runs/m2_02/lane3_telemetry.jsonl \
      --summary experiments/results/m2_02_lane3_smoke_summary.json

Requests are taken from the frozen W1 / W3 v2 packages (read-only), sent one at a
time at temperature 0. Acceptance (logs/Duong/M2-02/M2-02-openspec.md C4):
no failed request, >= 2 distinct k values in telemetry, outputs equal Vanilla.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
W1_REQUESTS = PROJECT_ROOT / "experiments/configs/w1/node1_chat_requests.jsonl"
W3_REQUESTS = PROJECT_ROOT / "experiments/configs/w3_v2/chat_requests.jsonl"


def load_prompts(per_workload: int) -> List[dict]:
    prompts = []
    for workload, path in (("w1", W1_REQUESTS), ("w3", W3_REQUESTS)):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if len([p for p in prompts if p["workload"] == workload]) >= per_workload:
                    break
                record = json.loads(line)
                prompts.append(
                    {
                        "workload": workload,
                        "sample_id": record["sample_id"],
                        "messages": record["request"]["messages"],
                    }
                )
    return prompts


def send(port: int, model: str, messages: list, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    start = time.time()
    resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=600)
    latency = time.time() - start
    resp.raise_for_status()
    body = resp.json()
    return {
        "content": body["choices"][0]["message"]["content"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "latency_s": round(latency, 3),
    }


def run_prompts(port: int, model: str, prompts: List[dict], max_tokens: int) -> List[dict]:
    results = []
    for prompt in prompts:
        entry = {"workload": prompt["workload"], "sample_id": prompt["sample_id"]}
        try:
            entry.update(send(port, model, prompt["messages"], max_tokens))
            entry["ok"] = True
        except Exception as exc:  # recorded, never hides a failure
            entry["ok"] = False
            entry["error"] = repr(exc)
        print(f"[{entry['workload']}] {entry['sample_id']}: ok={entry['ok']}")
        results.append(entry)
    return results


def gpu_free_mb() -> Optional[float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def read_telemetry(path: Path, since: float) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("finished_at", 0) >= since:
                records.append(record)
    return records


def cmd_reference(args) -> int:
    prompts = load_prompts(args.per_workload)
    results = run_prompts(args.port, args.model, prompts, args.max_tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} reference outputs to {out}")
    return 0 if all(r["ok"] for r in results) else 1


def cmd_adaptive(args) -> int:
    reference = {
        (r["workload"], r["sample_id"]): r
        for r in json.loads(Path(args.reference).read_text(encoding="utf-8"))
    }
    idle_free_mb = gpu_free_mb()
    started = time.time()
    prompts = load_prompts(args.per_workload)
    results = run_prompts(args.port, args.model, prompts, args.max_tokens)
    time.sleep(2)  # let the scheduler free finished requests and flush telemetry

    mismatches = []
    for r in results:
        ref = reference.get((r["workload"], r["sample_id"]))
        if r["ok"] and ref and ref.get("ok") and ref["content"] != r["content"]:
            mismatches.append(r["sample_id"])

    telemetry = read_telemetry(Path(args.telemetry), started)
    k_values = sorted({k for t in telemetry for k in t.get("tree_budgets", [])})

    checks = {
        "all_requests_ok": all(r["ok"] for r in results),
        "telemetry_lines_match_requests": len(telemetry) == len(results),
        "at_least_two_distinct_k": len(k_values) >= 2,
        "outputs_match_vanilla": not mismatches and len(reference) >= len(results),
    }
    summary = {
        "task": "M2-02 checklist 4 — Lane 3 smoke test",
        "port": args.port,
        "model": args.model,
        "num_requests": len(results),
        "max_tokens": args.max_tokens,
        "gpu_free_mb_after_server_load": idle_free_mb,
        "vram_guard_recommendation": (
            None if idle_free_mb is None
            else ("disable (ADAPTIVE_ENABLE_VRAM_GUARD=0)" if idle_free_mb < 1500 else "keep enabled")
        ),
        "distinct_k_values": k_values,
        "mismatched_samples": mismatches,
        "checks": checks,
        "passed": all(checks.values()),
        "requests": results,
        "telemetry": telemetry,
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("checks", "passed", "distinct_k_values",
                                              "gpu_free_mb_after_server_load",
                                              "vram_guard_recommendation")}, indent=2))
    return 0 if summary["passed"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("reference", "adaptive"):
        p = sub.add_parser(name)
        p.add_argument("--port", type=int, required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--per-workload", type=int, default=5, help="prompts per workload (W1, W3)")
        p.add_argument("--max-tokens", type=int, default=256)
    sub.choices["reference"].add_argument("--out", required=True)
    sub.choices["adaptive"].add_argument("--reference", required=True)
    sub.choices["adaptive"].add_argument("--telemetry", required=True)
    sub.choices["adaptive"].add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    return cmd_reference(args) if args.cmd == "reference" else cmd_adaptive(args)


if __name__ == "__main__":
    sys.exit(main())
