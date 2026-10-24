#!/usr/bin/env python3
"""M2-04 figures F1/F2 for the three-lane comparison.

Input : experiments/results/m2_summary.csv (from ``experiments.analyze_m2``)
Output: experiments/results/figures/m2/
  - F1_m2_tpot_throughput_3lanes.png : throughput + TPOT p50 vs concurrency,
    three lanes, error bars = repeat-level min-max.
  - F2_m2_speedup_vs_vanilla.png     : Static/Adaptive speedup vs matched
    Vanilla with the 1.0x parity line. The y-axis is derived from the data,
    so negative-speedup points are never clipped.

Usage: python -m experiments.plot_m2 [--csv ...] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from experiments.analyze_m2 import CONCURRENCY_LEVELS, LANES, WORKLOADS


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
})

WORKLOAD_TITLES = {
    "W1": "W1: VAT JSON Extraction (constrained)",
    "W2": "W2: Text-to-SQL (semi-constrained)",
    "W3": "W3: Free-text Analysis (unconstrained)",
}

LANE_CONFIG = {
    "vanilla": {"label": "Vanilla (autoregressive)", "color": "#1f77b4",
                "marker": "o", "linestyle": "-"},
    "static": {"label": "Static draft", "color": "#ff7f0e",
               "marker": "s", "linestyle": "--"},
    "adaptive": {"label": "Adaptive proposer", "color": "#2ca02c",
                 "marker": "^", "linestyle": "-."},
}

REQUIRED_COLUMNS = [
    "lane", "workload", "concurrency", "tpot_p50", "tpot_p50_min",
    "tpot_p50_max", "throughput", "throughput_min", "throughput_max",
    "speedup_vs_vanilla", "gpu", "n_runs", "experiment_id",
]


def load_m2_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path}: missing columns {missing_cols}")

    df["concurrency"] = df["concurrency"].astype(int)
    for col in REQUIRED_COLUMNS[3:10]:
        df[col] = df[col].astype(float)

    seen = set(zip(df["lane"], df["workload"], df["concurrency"]))
    missing = [
        (lane, workload, c)
        for workload in WORKLOADS for c in CONCURRENCY_LEVELS for lane in LANES
        if (lane, workload, c) not in seen
    ]
    if missing or len(df) != len(seen) or len(df) != 54:
        raise ValueError(
            f"{csv_path}: expected 54 unique cells; missing={missing}, rows={len(df)}"
        )

    for value in ("tpot_p50", "throughput"):
        bad = df[(df[f"{value}_min"] > df[value]) | (df[value] > df[f"{value}_max"])]
        if not bad.empty:
            cells = list(zip(bad["lane"], bad["workload"], bad["concurrency"]))
            raise ValueError(f"{csv_path}: {value} outside [min, max] for {cells}")
    return df


def _context(df: pd.DataFrame) -> str:
    gpus = ", ".join(sorted(set(df["gpu"].astype(str))))
    runs = int(df["n_runs"].min())
    experiments = ", ".join(sorted(set(df["experiment_id"].astype(str))))
    return f"{gpus} · {runs} repeats/cell · {experiments}"


def _log2_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True)


def _errorbar(ax, sub: pd.DataFrame, lane: str, value: str) -> None:
    cfg = LANE_CONFIG[lane]
    ax.errorbar(
        sub["concurrency"], sub[value],
        yerr=[sub[value] - sub[f"{value}_min"], sub[f"{value}_max"] - sub[value]],
        label=cfg["label"], color=cfg["color"], marker=cfg["marker"],
        linestyle=cfg["linestyle"], capsize=3,
    )


def plot_f1(df: pd.DataFrame, output_dir: Path) -> Path:
    """Row 1: throughput (tok/s); row 2: TPOT p50 (ms/token)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle(
        "Figure F1 (M2): Throughput & TPOT p50 vs Concurrency — Vanilla / Static / Adaptive\n"
        f"{_context(df)} · error bars = repeat-level min-max",
        fontweight="bold", y=0.98,
    )

    for col, workload in enumerate(WORKLOADS):
        wl_df = df[df["workload"] == workload]
        for lane in LANES:
            sub = wl_df[wl_df["lane"] == lane].sort_values("concurrency")
            _errorbar(axes[0, col], sub, lane, "throughput")
            _errorbar(axes[1, col], sub, lane, "tpot_p50")

        axes[0, col].set_title(WORKLOAD_TITLES[workload], fontweight="bold")
        axes[0, col].set_ylabel("Throughput (tokens / s)")
        axes[1, col].set_ylabel("TPOT p50 (ms / token)")
        axes[1, col].set_xlabel("Concurrency")
        _log2_axis(axes[0, col])
        _log2_axis(axes[1, col])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    out_file = output_dir / "F1_m2_tpot_throughput_3lanes.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def speedup_ylim(df: pd.DataFrame, margin: float = 0.08) -> tuple[float, float]:
    """Y-limits covering every non-Vanilla speedup and the 1.0x line."""
    values = df.loc[df["lane"] != "vanilla", "speedup_vs_vanilla"]
    low = min(float(values.min()), 1.0)
    high = max(float(values.max()), 1.0)
    pad = max((high - low) * margin, 0.05)
    return low - pad, high + pad


def label_offsets(static: float, adaptive: float, gap: int = 8) -> tuple[int, int]:
    """Point offsets for value labels: the higher lane above, the lower below."""
    if adaptive > static:
        return -gap - 6, gap
    return gap, -gap - 6


def plot_f2(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    fig.suptitle(
        "Figure F2 (M2): Speedup vs matched Vanilla (TPOT p50 ratio)\n"
        f"{_context(df)} · dashed line 1.0× = parity; below = negative speedup",
        fontweight="bold", y=1.03,
    )
    low, high = speedup_ylim(df)

    for idx, workload in enumerate(WORKLOADS):
        ax = axes[idx]
        ax.axhspan(low, 1.0, color="#d62728", alpha=0.06)
        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.8,
                   label="Parity 1.0× (Vanilla)")

        wl_df = df[df["workload"] == workload]
        speedups = {
            lane: wl_df[wl_df["lane"] == lane].set_index("concurrency")["speedup_vs_vanilla"]
            for lane in ("static", "adaptive")
        }
        for lane in ("static", "adaptive"):
            cfg = LANE_CONFIG[lane]
            series = speedups[lane].sort_index()
            ax.plot(series.index, series.values, label=cfg["label"],
                    color=cfg["color"], marker=cfg["marker"], linestyle=cfg["linestyle"])
            for concurrency, value in series.items():
                offsets = label_offsets(speedups["static"][concurrency],
                                        speedups["adaptive"][concurrency])
                ax.annotate(f"{value:.2f}×", xy=(concurrency, value),
                            xytext=(0, offsets[0] if lane == "static" else offsets[1]),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=cfg["color"])

        ax.set_title(WORKLOAD_TITLES[workload], fontweight="bold")
        ax.set_xlabel("Concurrency")
        if idx == 0:
            ax.set_ylabel("Speedup vs Vanilla (×)")
        _log2_axis(ax)
        ax.set_ylim(low, high)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()

    out_file = output_dir / "F2_m2_speedup_vs_vanilla.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M2-04 figures F1/F2 (three lanes)")
    parser.add_argument("--csv", type=Path, default=Path("experiments/results/m2_summary.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/results/figures/m2"))
    args = parser.parse_args(argv)

    df = load_m2_summary(args.csv)
    print(f"Saved {plot_f1(df, args.output_dir)}")
    print(f"Saved {plot_f2(df, args.output_dir)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
