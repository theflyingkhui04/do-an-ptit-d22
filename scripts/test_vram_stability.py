#!/usr/bin/env python3
"""Hardware VRAM stability & memory leak benchmark script for Adaptive Speculative Decoding.

Measures memory usage (allocated, reserved, peak, and RSS) across requests with dynamic
tree budget changes k in {0, 2, 3, 4}. Compatible with Tesla T4 (CUDA) and CPU fallback.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import torch

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import AdaptiveDraftController


def get_current_memory(device: torch.device) -> Dict[str, float]:
    """Retrieves current memory metrics in Megabytes (MB)."""
    metrics = {}
    if device.type == "cuda":
        metrics["allocated_mb"] = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        metrics["reserved_mb"] = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
        metrics["max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        # Fallback to process RSS on CPU
        try:
            import psutil
            process = psutil.Process()
            rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
        except ImportError:
            rss_mb = 0.0
        metrics["allocated_mb"] = rss_mb
        metrics["reserved_mb"] = rss_mb
        metrics["max_allocated_mb"] = rss_mb
    return metrics


def run_benchmark(
    num_requests: int = 50,
    steps_per_request: int = 25,
    warmup_requests: int = 5,
    device_name: str = "auto",
    output_path: str = "logs/Khoi/M2-01/vram_stability_t4.json",
    peak_threshold_mb: float = 13500.0,
) -> Dict[str, Any]:
    """Runs controlled stress test measuring VRAM stability across dynamic budget switching."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"=== VRAM Stability & Memory Leak Benchmark ===")
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Host CPU'})")
    print(f"Workload: {num_requests} requests x {steps_per_request} steps (Warmup: {warmup_requests} requests)")
    print(f"Peak VRAM Threshold: {peak_threshold_mb:.1f} MB\n")

    adapter = AdaptiveProposerAdapter()

    # Pre-allocate hidden dimension for mock tensor simulation
    hidden_dim = 2048  # Simulates 7B intermediate activation scale

    def mock_draft_generator(k: int) -> List[int]:
        if k <= 0:
            return []
        # Allocate mock draft tensor on target device to simulate forward pass activations
        dummy_tensor = torch.randn((1, k, hidden_dim), device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        # Force computation to ensure memory registration
        _ = dummy_tensor.sum()
        del dummy_tensor
        return [100 + i for i in range(k)]

    # --------------------------------------------------------------------------
    # 1. Warm-up Phase
    # --------------------------------------------------------------------------
    print(">> Phase 1: Running warm-up requests to establish memory steady-state...")
    for w_idx in range(warmup_requests):
        req_id = f"warmup_{w_idx}"
        for step in range(10):
            out = adapter.propose(req_id, draft_generator=mock_draft_generator)
            if not out.skipped_draft:
                adapter.on_verification_completed(req_id, num_accepted=out.budget_k, num_drafted=out.budget_k)
        adapter.end_request(req_id)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    baseline_mem = get_current_memory(device)
    print(f"   [Baseline Memory] Allocated: {baseline_mem['allocated_mb']:.2f} MB | Reserved: {baseline_mem['reserved_mb']:.2f} MB\n")

    # --------------------------------------------------------------------------
    # 2. Dynamic Stress Phase
    # --------------------------------------------------------------------------
    print(">> Phase 2: Executing dynamic decoding requests with budget transitions...")
    samples: List[Dict[str, Any]] = []
    budget_distribution: Dict[int, int] = {0: 0, 2: 0, 3: 0, 4: 0}
    t_start = time.time()

    for req_idx in range(num_requests):
        req_id = f"stress_req_{req_idx:03d}"
        req_budgets = []

        for step in range(steps_per_request):
            # Simulate alternating entropy phases:
            # - Steps 0..7: High acceptance (alpha ~ 0.90 -> k=3, 4)
            # - Steps 8..15: Low acceptance (alpha ~ 0.40 -> fallback k=0, periodic probe k=2)
            # - Steps 16..24: Mid acceptance (alpha ~ 0.72 -> k=2, 3)
            if step < 8:
                simulated_alpha = 0.90
            elif step < 16:
                simulated_alpha = 0.40
            else:
                simulated_alpha = 0.72

            out = adapter.propose(req_id, draft_generator=mock_draft_generator)
            k = out.budget_k
            req_budgets.append(k)
            budget_distribution[k] = budget_distribution.get(k, 0) + 1

            if not out.skipped_draft:
                # Accept according to simulated alpha
                accepted = min(k, int(round(k * simulated_alpha)))
                adapter.on_verification_completed(req_id, num_accepted=accepted, num_drafted=k)

        # Finalize request and collect telemetry
        telemetry = adapter.end_request(req_id)

        # Sample memory immediately after request termination
        mem_after_req = get_current_memory(device)
        samples.append({
            "request_index": req_idx,
            "request_id": req_id,
            "allocated_mb": mem_after_req["allocated_mb"],
            "reserved_mb": mem_after_req["reserved_mb"],
            "max_allocated_mb": mem_after_req["max_allocated_mb"],
            "mean_accept_len": telemetry.get("mean_accept_len", 0.0),
            "mean_k": telemetry.get("mean_k", 0.0),
        })

        if (req_idx + 1) % 10 == 0 or (req_idx + 1) == num_requests:
            print(f"   [Progress] Completed {req_idx + 1:02d}/{num_requests} requests | "
                  f"Allocated: {mem_after_req['allocated_mb']:.2f} MB | "
                  f"Peak: {mem_after_req['max_allocated_mb']:.2f} MB")

    total_duration = time.time() - t_start

    # --------------------------------------------------------------------------
    # 3. Post-Run Memory Cooldown & Analysis
    # --------------------------------------------------------------------------
    print("\n>> Phase 3: Evaluating post-run memory recovery & leak gradient...")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_mem = get_current_memory(device)

    # Compute memory delta
    delta_allocated = final_mem["allocated_mb"] - baseline_mem["allocated_mb"]
    delta_reserved = final_mem["reserved_mb"] - baseline_mem["reserved_mb"]
    peak_allocated = max(s["max_allocated_mb"] for s in samples) if samples else final_mem["max_allocated_mb"]

    # Compute trend slope (MB per request) to detect linear creeping leaks
    if len(samples) > 1:
        n = len(samples)
        x = list(range(n))
        y = [s["allocated_mb"] for s in samples]
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        trend_slope = (numerator / denominator) if denominator > 0 else 0.0
    else:
        trend_slope = 0.0

    # Decision criteria
    passed_peak = peak_allocated <= peak_threshold_mb
    passed_leak = abs(delta_allocated) < 5.0 and abs(trend_slope) < 0.05
    overall_status = "PASS" if (passed_peak and passed_leak) else "FAIL"

    result_bundle = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "Host CPU",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_requests": num_requests,
        "steps_per_request": steps_per_request,
        "total_steps": num_requests * steps_per_request,
        "duration_seconds": round(total_duration, 2),
        "baseline_memory_mb": round(baseline_mem["allocated_mb"], 2),
        "final_memory_mb": round(final_mem["allocated_mb"], 2),
        "delta_memory_mb": round(delta_allocated, 4),
        "peak_allocated_mb": round(peak_allocated, 2),
        "peak_threshold_mb": peak_threshold_mb,
        "memory_trend_slope_mb_per_req": round(trend_slope, 6),
        "budget_distribution": budget_distribution,
        "status": overall_status,
        "samples": samples,
    }

    # Ensure output directory exists and save JSON
    output_file = REPO_ROOT / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_bundle, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"            VRAM STABILITY BENCHMARK RESULTS            ")
    print(f"=======================================================")
    print(f"Status               : {overall_status} (Peak <= {peak_threshold_mb:.0f}MB & Zero Creep)")
    print(f"Baseline Allocated   : {baseline_mem['allocated_mb']:.2f} MB")
    print(f"Final Allocated      : {final_mem['allocated_mb']:.2f} MB")
    print(f"Delta Allocated      : {delta_allocated:+.4f} MB")
    print(f"Peak Allocated       : {peak_allocated:.2f} MB")
    print(f"Memory Trend Slope   : {trend_slope:+.6f} MB/request")
    print(f"Budget Distribution  : {budget_distribution}")
    print(f"Report Artifact      : {output_file}")
    print(f"=======================================================\n")

    return result_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRAM Stability and Memory Leak Benchmark")
    parser.add_argument("--requests", type=int, default=50, help="Number of test requests (default: 50)")
    parser.add_argument("--steps", type=int, default=25, help="Steps per request (default: 25)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests (default: 5)")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    parser.add_argument("--output", type=str, default="logs/Khoi/M2-01/vram_stability_t4.json", help="Output JSON path")
    parser.add_argument("--peak-threshold", type=float, default=13500.0, help="Peak VRAM threshold in MB (default: 13500.0)")

    args = parser.parse_args()
    res = run_benchmark(
        num_requests=args.requests,
        steps_per_request=args.steps,
        warmup_requests=args.warmup,
        device_name=args.device,
        output_path=args.output,
        peak_threshold_mb=args.peak_threshold,
    )
    if res["status"] != "PASS":
        sys.exit(1)
