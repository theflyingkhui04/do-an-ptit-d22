"""[M5-01] Four-lane benchmark campaign on Modal L4 (manual; Đại runs it), workload W1_V3 by default.

Same scientific protocol as M3-05 (experiments/M3_05_CAMPAIGN_RUNBOOK.md) on the
full-receipt workload (W1_V3: library OCR text without marks; W1_V2: earlier Qwen-VL text): 4 lanes x c=1,2,4,8,16,32 = 24 cells, exactly 3
warm-ups + 5 repeats x 32 measured requests, one fresh server per cell. Each
server runs with an 8192-token context and the same patched vLLM runtime as
M3-05 (backport of upstream PR #55708, experiments/vllm_runtime_patch.py).

The deploy/submit/poll/collect/validate/stop loop is run_m3_campaign.drive.
Runbook: experiments/M5_01_W1V2_CAMPAIGN_RUNBOOK.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from experiments import modal_lanes as lanes
from experiments import run_m3_campaign as m3
from experiments import vllm_runtime_patch as runtime_patch
from experiments.validate_m3 import validate_cell

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_WORKLOADS = ("W1_V2", "W1_V3")
WORKLOAD = "W1_V3"  # [M5-01 canonical flow] default; W1_V2 kept for the earlier Qwen-VL package
CONCURRENCIES = (1, 2, 4, 8, 16, 32)
# Riskiest cell first: longest outputs at the highest load (timeout, KV cache, spec metrics).
PILOT_CELL = ("eagle", 32)
GPU = "L4"
PLAN_ROOT = Path("experiments/runs/m5_01")
def summary_path(workload: str) -> Path:
    return Path(f"experiments/results/m5_01_{workload.lower()}_sweep_summary.json")


M3_SUMMARY = Path("experiments/results/m3_sweep_4lanes_summary.json")
def comparison_path(workload: str) -> Path:
    return Path(f"experiments/results/m5_01_{workload.lower()}_vs_m3_05_w1.md")


FIGURES = Path("experiments/results/figures/m5_01")


def serving_env(workload: str = WORKLOAD) -> dict[str, str]:
    """Deploy-time switches of experiments.modal_runner for every cell of the campaign workload."""
    serving = lanes.workload_serving(workload)
    return {"PTIT_GPU": GPU, "PTIT_DTYPE_PROFILE": "l4-bf16",
            "PTIT_MAX_MODEL_LEN": str(serving["max_model_len"]),
            "PTIT_VLLM_RUNTIME": serving["vllm_runtime"]}


def plan(campaign_id: str, sha: str, pilot: bool = False, workload: str = WORKLOAD) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign_id):
        raise ValueError("Invalid campaign_id")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("A full committed execution SHA is required")
    if workload not in CAMPAIGN_WORKLOADS:
        raise ValueError(f"workload must be one of {CAMPAIGN_WORKLOADS}")
    grid = [PILOT_CELL] if pilot else [(lane, c) for lane in lanes.LANES for c in CONCURRENCIES]
    serving = lanes.workload_serving(workload)
    return {"campaign_id": campaign_id, "commit": sha, "task": "M5-01",
            "kind": "pilot" if pilot else "confirmatory", "workload": workload, "gpu": GPU,
            "max_model_len": serving["max_model_len"], "harness_timeout_s": serving["timeout_s"],
            "vllm_runtime": runtime_patch.RUNTIME_LABEL, "warmup": 3,
            "lifecycle": "fresh-server-per-cell", "repeats": 5, "num_requests": 32,
            "cells": [{"key": f"{lane}__{workload}__c{c}", "lane": lane, "workload": workload,
                       "concurrency": c} for lane, c in grid]}


def run(record: dict, retry_failed: bool = False) -> None:
    m3.drive(record, retry_failed, plan_root=PLAN_ROOT, extra_env=serving_env(record["workload"]))


def server_gpu(path: Path) -> str:
    return json.loads((path / "server_env.json").read_text(encoding="utf-8"))["gpu"]


def validate_campaign(record: dict) -> list[Path]:
    selected = [m3.selected_attempt(record, cell) for cell in record["cells"]]
    missing = [c["key"] for c, p in zip(record["cells"], selected) if p is None]
    if missing:
        raise ValueError(f"Incomplete campaign: {missing}")
    instance_ids = [validate_cell(path)["server_instance_id"] for path in selected]
    if len(set(instance_ids)) != len(selected):
        raise ValueError("Server instance reused between scientific cells")
    gpus = {server_gpu(path) for path in selected}
    if gpus != {"NVIDIA L4"}:
        raise ValueError(f"Every cell must run on NVIDIA L4, got {sorted(gpus)}")
    return selected


def aggregate(record: dict, destination: Path | None = None) -> None:
    from experiments import aggregate_results as agg

    if record["kind"] != "confirmatory":
        raise ValueError("Pilot evidence is never aggregated into campaign results")
    destination = destination or ROOT / summary_path(record["workload"])
    paths = validate_campaign(record)
    rows = []
    for path in paths:
        raw_path = path / "raw.vram.jsonl"
        if not raw_path.exists():
            raise ValueError(f"VRAM evidence required for final aggregation: {path}")
        raw = agg._load_jsonl(raw_path)
        row = agg.aggregate_raw_jsonl(raw_path)[0]
        if row["lane"] != "vanilla":
            row["alpha"], row["tau"] = agg.aggregate_static_acceptance(raw)
        row["attempt_id"] = path.name
        row["_experiment_id"] = record["campaign_id"]
        rows.append(row)
    rows = agg.apply_speedup_vs_vanilla(rows)
    result = {"campaign": record, "scientific_status": "VALID", "engineering_gate": "NOT_DEFINED",
              "selected_attempts": [p.relative_to(ROOT).as_posix() for p in paths],
              "all_attempts": {c["key"]: [p.name for p in m3.attempts(record, c)] for c in record["cells"]},
              "cells": rows}
    content = (json.dumps(result, indent=2) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != content:
        raise FileExistsError(f"Conflicting summary: {destination}")
    if not destination.exists():
        destination.write_bytes(content)


def _fmt(value: float | None, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def comparison_markdown(m5_summary: dict, m3_summary: dict) -> str:
    """Speedup vs vanilla of the campaign workload next to the M3-05 W1 cell of the same lane and concurrency."""
    w1 = {(r["lane"], r["concurrency"]): r for r in m3_summary["cells"] if r["workload"] == "W1"}
    order = {lane: i for i, lane in enumerate(lanes.LANES)}
    rows = sorted((r for r in m5_summary["cells"] if r["workload"] in CAMPAIGN_WORKLOADS),
                  key=lambda r: (order[r["lane"]], r["concurrency"]))
    name = rows[0]["workload"] if rows else WORKLOAD
    source = "OCR thư viện (PaddleOCR + VietOCR), không dấu" if name == "W1_V3" else "OCR Qwen2.5-VL toàn tờ"
    lines = [
        f"# [M5-01] {name} so với W1 v1 (M3-05) — tăng tốc so với vanilla",
        "",
        "- **M3-05 (W1 v1):** Google Colab L4, ngữ cảnh 4096, đầu vào 4 dòng nhãn, `max_tokens` 1024.",
        f"- **M5-01 ({name}):** Modal L4, ngữ cảnh 8192, đầu vào {source}, `max_tokens` 3072.",
        "- Cả hai: vLLM 0.30.0 + backport #55708, cùng giao thức (3 warm-up, 5 × 32 request đo, server mới mỗi ô).",
        "- Chỉ so **tỉ lệ tăng tốc trong từng campaign**; TPOT và throughput tuyệt đối đo ở điều kiện khác nhau.",
        "- Không có ngưỡng đạt/không đạt; tăng tốc < 1 là kết quả hợp lệ.",
        "",
        f"| Làn | c | Tăng tốc W1 (M3-05) | Tăng tốc {name} (M5-01) | TPOT p50 W1 (ms) | TPOT p50 {name} (ms) | α {name} |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        ref = w1.get((row["lane"], row["concurrency"]), {})
        lines.append(
            f"| {row['lane']} | {row['concurrency']} | {_fmt(ref.get('speedup_vs_vanilla'), 2)} | "
            f"{_fmt(row.get('speedup_vs_vanilla'), 2)} | {_fmt(ref.get('tpot_p50'), 2)} | "
            f"{_fmt(row.get('tpot_p50'), 2)} | {_fmt(row.get('alpha'), 3)} |")
    return "\n".join(lines) + "\n"


def _load_valid_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("scientific_status") != "VALID" or summary["campaign"].get("kind") != "confirmatory":
        raise ValueError(f"{path} is not a validated confirmatory summary")
    return summary


def compare(workload: str = WORKLOAD, m3_summary: Path = ROOT / M3_SUMMARY) -> None:
    summary, destination = ROOT / summary_path(workload), ROOT / comparison_path(workload)
    text = comparison_markdown(_load_valid_summary(summary),
                               json.loads(m3_summary.read_text(encoding="utf-8")))
    if destination.exists() and destination.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Conflicting comparison: {destination}")
    destination.write_text(text, encoding="utf-8", newline="\n")


def figures(workload: str = WORKLOAD, destination: Path = ROOT / FIGURES) -> None:
    summary = ROOT / summary_path(workload)
    tag = workload.lower().replace("_", "")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = _load_valid_summary(summary)
    rows = record["cells"]
    if len(rows) != len(lanes.LANES) * len(CONCURRENCIES):
        raise ValueError("Figures require the full 24-cell summary")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, metrics in ((f"F1_m5_01_{tag}_tpot_throughput.png", ("tpot_p50", "throughput")),
                              (f"F2_m5_01_{tag}_speedup_vs_vanilla.png", ("speedup_vs_vanilla",))):
        target = destination / filename
        if target.exists():
            raise FileExistsError(target)
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            for lane in lanes.LANES:
                values = sorted((r for r in rows if r["lane"] == lane), key=lambda r: r["concurrency"])
                ax.plot([r["concurrency"] for r in values], [r[metric] for r in values], marker="o", label=lane)
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(CONCURRENCIES), [str(c) for c in CONCURRENCIES])
            ax.set_xlabel("Concurrency")
            ax.set_ylabel({"tpot_p50": "TPOT p50 (ms)", "throughput": "Output tokens/s",
                           "speedup_vs_vanilla": "Speedup vs Vanilla (TPOT ratio)"}[metric])
            if metric == "speedup_vs_vanilla":
                ax.axhline(1, linestyle="--", color="black")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"M5-01 {workload} (L4, ctx 8192): {record['campaign']['campaign_id']} — negative results retained")
        fig.tight_layout()
        fig.savefig(target, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("plan", "run", "status", "validate", "aggregate",
                                           "compare", "figures", "stop"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--pilot", action="store_true", help="one eagle c=32 cell; never aggregated")
    parser.add_argument("--workload", choices=CAMPAIGN_WORKLOADS, default=WORKLOAD)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    record = plan(args.campaign_id, args.execution_sha, pilot=args.pilot, workload=args.workload)
    if args.action == "plan":
        print(json.dumps(record, indent=2))  # read-only; does not create an evidence directory
        return
    m3.save_plan(record, PLAN_ROOT)
    if args.action == "run":
        run(record, args.retry_failed)
    elif args.action == "status":
        from experiments import modal_runner as runner
        for cell in record["cells"]:
            history = m3.attempts(record, cell)
            state = "COLLECTED" if m3.selected_attempt(record, cell) else (
                runner.bench_status(history[-1].name) if history else "NOT_SUBMITTED")
            print(cell["key"], state)
    elif args.action == "validate":
        cells = validate_campaign(record)
        print(f"VALID: {len(cells)} cells, {len(cells) * 160} measured requests")
    elif args.action == "aggregate":
        aggregate(record)
    elif args.action == "compare":
        compare(record["workload"])
    elif args.action == "figures":
        figures(record["workload"])
    else:
        m3.stop()


if __name__ == "__main__":
    main()
