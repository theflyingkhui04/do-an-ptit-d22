"""Smoke test for Lane 3 (AdaptiveAsyncScheduler) — M2-02 checklist 4.

A T4 cannot host two lanes at once, so the test runs one lane at a time:

  bash scripts/switch_lane.sh lane1
  python scripts/smoke_lane3_adaptive.py reference --lane vanilla   # Pass 1 — Vanilla
  bash scripts/switch_lane.sh lane2
  python scripts/smoke_lane3_adaptive.py reference --lane static    # Pass 2 — Static (optional)
  bash scripts/switch_lane.sh lane3
  python scripts/smoke_lane3_adaptive.py adaptive                   # Pass 3 — Adaptive + checks

Ports, served-model names and every input/output path are fixed constants below
(no free-form URL or path arguments). Requests come from the frozen W1 / W3 v2
packages (read-only), sent one at a time at temperature 0. Acceptance
(logs/Duong/M2-02/M2-02-openspec.md C4): no failed request, >= 2 distinct k values
in telemetry, outputs equal Vanilla.
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
RUNS_DIR = PROJECT_ROOT / "experiments/runs/m2_02"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"

# Fixed lane table: same ports / served-model names as scripts/launch_lane*.sh.
LANES = {
    "vanilla": {"port": 8001, "model": "qwen2.5-7b-vanilla",
                "outputs": RUNS_DIR / "smoke_vanilla_outputs.json"},
    "static": {"port": 8002, "model": "qwen2.5-7b-static-draft",
               "outputs": RUNS_DIR / "smoke_static_outputs.json"},
    "adaptive": {"port": 8003, "model": "qwen2.5-7b-adaptive",
                 "outputs": None},
}
REFERENCE_OUTPUTS = LANES["vanilla"]["outputs"]
# Default of ADAPTIVE_TELEMETRY_PATH in scripts/launch_lane3_adaptive.sh.
TELEMETRY_PATH = RUNS_DIR / "lane3_telemetry.jsonl"
SUMMARY_PATH = RESULTS_DIR / "m2_02_lane3_smoke_summary.json"
VRAM_GUARD_MIN_FREE_MB = 1500


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


def send(lane: dict, messages: list, max_tokens: int) -> dict:
    payload = {
        "model": lane["model"],
        "messages": messages,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    url = "http://127.0.0.1:%d/v1/chat/completions" % lane["port"]
    start = time.time()
    resp = requests.post(url, json=payload, timeout=600)
    latency = time.time() - start
    resp.raise_for_status()
    body = resp.json()
    return {
        "content": body["choices"][0]["message"]["content"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "latency_s": round(latency, 3),
    }


def run_prompts(lane: dict, prompts: List[dict], max_tokens: int) -> List[dict]:
    results = []
    for prompt in prompts:
        entry = {"workload": prompt["workload"], "sample_id": prompt["sample_id"]}
        try:
            entry.update(send(lane, prompt["messages"], max_tokens))
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


def vram_guard_recommendation(free_mb: Optional[float]) -> Optional[str]:
    if free_mb is None:
        return None
    if free_mb < VRAM_GUARD_MIN_FREE_MB:
        return "disable (ADAPTIVE_ENABLE_VRAM_GUARD=0)"
    return "keep enabled"


def read_telemetry(since: float) -> List[dict]:
    if not TELEMETRY_PATH.exists():
        return []
    records = []
    with TELEMETRY_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("finished_at", 0) >= since:
                records.append(record)
    return records


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_reference(lane_name: str, per_workload: int, max_tokens: int) -> int:
    lane = LANES[lane_name]
    prompts = load_prompts(per_workload)
    results = run_prompts(lane, prompts, max_tokens)
    write_json(lane["outputs"], results)
    print(f"Saved {len(results)} reference outputs to {lane['outputs']}")
    return 0 if all(r["ok"] for r in results) else 1


def cmd_adaptive(per_workload: int, max_tokens: int) -> int:
    lane = LANES["adaptive"]
    reference = {
        (r["workload"], r["sample_id"]): r
        for r in json.loads(REFERENCE_OUTPUTS.read_text(encoding="utf-8"))
    }
    idle_free_mb = gpu_free_mb()
    started = time.time()
    prompts = load_prompts(per_workload)
    results = run_prompts(lane, prompts, max_tokens)
    time.sleep(2)  # let the scheduler free finished requests and flush telemetry

    mismatches = []
    for r in results:
        ref = reference.get((r["workload"], r["sample_id"]))
        if r["ok"] and ref and ref.get("ok") and ref["content"] != r["content"]:
            mismatches.append(r["sample_id"])

    telemetry = read_telemetry(started)
    k_values = sorted({k for t in telemetry for k in t.get("tree_budgets", [])})

    checks = {
        "all_requests_ok": all(r["ok"] for r in results),
        "telemetry_lines_match_requests": len(telemetry) == len(results),
        "at_least_two_distinct_k": len(k_values) >= 2,
        "outputs_match_vanilla": not mismatches and len(reference) >= len(results),
    }
    summary = {
        "task": "M2-02 checklist 4 — Lane 3 smoke test",
        "port": lane["port"],
        "model": lane["model"],
        "num_requests": len(results),
        "max_tokens": max_tokens,
        "gpu_free_mb_after_server_load": idle_free_mb,
        "vram_guard_recommendation": vram_guard_recommendation(idle_free_mb),
        "distinct_k_values": k_values,
        "mismatched_samples": mismatches,
        "checks": checks,
        "passed": all(checks.values()),
        "requests": results,
        "telemetry": telemetry,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({k: summary[k] for k in ("checks", "passed", "distinct_k_values",
                                              "gpu_free_mb_after_server_load",
                                              "vram_guard_recommendation")}, indent=2))
    return 0 if summary["passed"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ref = sub.add_parser("reference", help="collect reference outputs from lane 1 or lane 2")
    ref.add_argument("--lane", choices=["vanilla", "static"], required=True)
    ada = sub.add_parser("adaptive", help="run lane 3 and check it against the Vanilla reference")
    for p in (ref, ada):
        p.add_argument("--per-workload", type=int, default=5, help="prompts per workload (W1, W3)")
        p.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.cmd == "reference":
        return cmd_reference(args.lane, args.per_workload, args.max_tokens)
    return cmd_adaptive(args.per_workload, args.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
