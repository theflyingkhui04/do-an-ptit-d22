"""Script to generate hardware_profile_l4.json from empirical benchmark data.

Task: [M3-02] Hardware Profiler & Mô hình Chi phí C(N, B) trên GPU L4
Input: experiments/results/m3_02_forward_costs_l4.json
Output: experiments/results/hardware_profile_l4.json
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple


def solve_alpha_star(k: int, cost_ratio: float, iterations: int = 60) -> Optional[float]:
    """Solves break-even acceptance rate alpha* in (0, 1) where E[tau] == cost_ratio.
    
    E[tau] = 1 + alpha + alpha^2 + ... + alpha^k.
    If cost_ratio <= 1.0, break-even is 0.0 (always faster).
    If 1 + k < cost_ratio, even 100% acceptance cannot break even, returns None.
    """
    if cost_ratio <= 1.0:
        return 0.0
    if 1 + k < cost_ratio:
        return None

    low, high = 0.0, 1.0
    for _ in range(iterations):
        mid = (low + high) / 2.0
        val = sum(mid**i for i in range(k + 1))
        if val < cost_ratio:
            low = mid
        else:
            high = mid
    return round((low + high) / 2.0, 4)


def build_hardware_profile_l4(
    input_path: Path,
    output_path: Path,
    attn_verify_factor: float = 0.008,
) -> dict:
    """Builds and serializes L4 hardware cost profile matrix C(k, B)."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = data.get("grid", {})
    batches: List[int] = grid.get("batches", [1, 2, 4, 8, 16, 32])
    contexts: List[int] = grid.get("contexts", [256, 512, 1024, 2048, 3072])
    cells = data.get("cells", [])

    # Group cells by (model, batch_size)
    target_by_batch: Dict[int, List[float]] = {b: [] for b in batches}
    draft_by_batch: Dict[int, List[float]] = {b: [] for b in batches}

    # Also keep grid lookup by (model, context_len, batch_size)
    full_grid: Dict[Tuple[str, int, int], float] = {}

    for c in cells:
        m = c["model"]
        b = c["batch_size"]
        l = c["context_len"]
        val = c["step_ms_median"]
        full_grid[(m, l, b)] = val
        if m == "target" and b in target_by_batch:
            target_by_batch[b].append(val)
        elif m == "draft" and b in draft_by_batch:
            draft_by_batch[b].append(val)

    # Compute median cost for each batch size across context lengths
    c_target: Dict[int, float] = {b: round(median(target_by_batch[b]), 3) for b in batches}
    c_draft: Dict[int, float] = {b: round(median(draft_by_batch[b]), 3) for b in batches}

    # Candidate budgets k in {0, 1, 2, 3, 4, 5, 6}
    candidate_k = [0, 1, 2, 3, 4, 5, 6]

    cost_matrix: Dict[str, float] = {}
    verification_costs: Dict[str, float] = {}
    break_even_alpha: Dict[str, Optional[float]] = {}

    for b in batches:
        c_t = c_target[b]
        c_d = c_draft[b]

        # Pure vanilla baseline
        c_vanilla = c_t

        for k in candidate_k:
            key = f"k{k}_b{b}"
            if k == 0:
                # In standalone vanilla, cost is C_T.
                # In dual-model runner with draft sync, cost is C_T + C_D.
                # We store standard step cost as C_T.
                cost_matrix[key] = round(c_t, 3)
                break_even_alpha[key] = 0.0
            else:
                # Multi-token verification attention overhead
                c_v = c_t * (1.0 + attn_verify_factor * (k - 1))
                verification_costs[f"k{k}_b{b}"] = round(c_v, 3)

                # Total speculative cycle time: C(k, B) = k * C_D + C_V
                step_cost = k * c_d + c_v
                cost_matrix[key] = round(step_cost, 3)

                ratio = step_cost / c_vanilla
                break_even_alpha[key] = solve_alpha_star(k, ratio)

    profile_data = {
        "metadata": {
            "device": "NVIDIA L4",
            "sm_architecture": "sm_89",
            "vram_total_mb": 23034.0,
            "vllm_version": "0.30.0",
            "source_run_id": data.get("run_id", "20261025-m302-l4-v2"),
            "source_commit": data.get("commit", ""),
            "timestamp_utc": data.get("timestamp_utc", ""),
            "attn_verify_factor": attn_verify_factor,
        },
        "models": {
            "target": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "grid": {
            "batches": batches,
            "contexts": contexts,
            "candidate_budgets": candidate_k,
        },
        "forward_costs_median_ms": {
            "target_c_t": {str(b): c_target[b] for b in batches},
            "draft_c_d": {str(b): c_draft[b] for b in batches},
            "c_d_over_c_t_ratio": {str(b): round(c_draft[b] / c_target[b], 4) for b in batches},
        },
        "verification_costs_ms": verification_costs,
        "cost_matrix_ms": cost_matrix,
        "break_even_alpha": break_even_alpha,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    return profile_data


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    in_file = repo_root / "experiments" / "results" / "m3_02_forward_costs_l4.json"
    out_file = repo_root / "experiments" / "results" / "hardware_profile_l4.json"
    res = build_hardware_profile_l4(in_file, out_file)
    print(f"Generated L4 Hardware Profile successfully at: {out_file}")
    print(f"Batches: {res['grid']['batches']}")
    print(f"Break-even alpha at B=1: k2={res['break_even_alpha'].get('k2_b1')}, "
          f"k3={res['break_even_alpha'].get('k3_b1')}, k4={res['break_even_alpha'].get('k4_b1')}")
