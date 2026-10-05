#!/usr/bin/env python3
"""
experiments/plot_f1.py
Sinh biểu đồ F1 (Throughput, TPOT vs Concurrency) và F2 (Speedup vs Concurrency)
cho Cột mốc M1 (Baseline Benchmark).

Tuân thủ đặc tả kỹ thuật: docs/06-ke-hoach-thuc-nghiem.md
Dữ liệu đầu vào: experiments/results/summary.csv
Thư mục xuất ảnh: experiments/results/figures/
"""

import os
import sys
import argparse

# Tránh lỗi mã hóa ký tự trên console Windows (cp932 / cp1252 / ...)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib.pyplot as plt

# Định cấu hình đồ họa chất lượng cao chuẩn báo cáo / bài báo khoa học
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
})

WORKLOAD_TITLES = {
    "W1": "W1: VAT JSON Extraction\n(Constrained / High-Alpha ~91.6%)",
    "W2": "W2: Text-to-SQL\n(Semi-Constrained / Mid-Alpha ~80.5%)",
    "W3": "W3: Free Text Analysis\n(Unconstrained / Low-Alpha ~54.5%)",
}

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]

LANE_CONFIG = {
    "vanilla": {
        "label": "Vanilla Baseline (Autoregressive)",
        "color": "#1f77b4",  # Steel Blue
        "marker": "o",
        "linestyle": "-",
    },
    "static": {
        "label": "Static Draft Baseline (k=3)",
        "color": "#ff7f0e",  # Orange
        "marker": "s",
        "linestyle": "--",
    },
}


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file summary.csv tại: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = [
        "lane", "workload", "concurrency", "tpot_p50", "throughput", "speedup_vs_vanilla"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc '{col}' trong {csv_path}")

    # Chuẩn hóa kiểu dữ liệu
    df["concurrency"] = df["concurrency"].astype(int)
    df["tpot_p50"] = df["tpot_p50"].astype(float)
    df["throughput"] = df["throughput"].astype(float)
    df["speedup_vs_vanilla"] = df["speedup_vs_vanilla"].astype(float)

    return df


def plot_f1_throughput_tpot(df: pd.DataFrame, output_dir: str):
    """
    Sinh Biểu đồ F1:
    - Hàng 1: Throughput (tok/s) vs Concurrency cho W1, W2, W3
    - Hàng 2: TPOT p50 (ms/tok) vs Concurrency cho W1, W2, W3
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle(
        "Figure F1: M1 Baseline Benchmark — Throughput & TPOT vs Concurrency (Turing T4 16GB)\n"
        "So sánh Làn 1 (Vanilla: Qwen2.5-7B-AWQ) vs Làn 2 (Static Draft: Target 7B + Draft 0.5B, k=3)",
        fontweight="bold",
        y=0.98,
    )

    workloads = ["W1", "W2", "W3"]

    for col_idx, wl in enumerate(workloads):
        ax_tp = axes[0, col_idx]
        ax_tpot = axes[1, col_idx]

        wl_df = df[df["workload"] == wl]

        # 1. Vẽ Throughput
        for lane in ["vanilla", "static"]:
            sub = wl_df[wl_df["lane"] == lane].sort_values("concurrency")
            if not sub.empty:
                cfg = LANE_CONFIG[lane]
                ax_tp.plot(
                    sub["concurrency"],
                    sub["throughput"],
                    label=cfg["label"],
                    color=cfg["color"],
                    marker=cfg["marker"],
                    linestyle=cfg["linestyle"],
                )

        ax_tp.set_title(WORKLOAD_TITLES[wl], fontweight="bold")
        ax_tp.set_ylabel("Throughput (tokens / sec)")
        ax_tp.grid(True)
        ax_tp.set_xscale("log", base=2)
        ax_tp.set_xticks(CONCURRENCY_LEVELS)
        ax_tp.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # 2. Vẽ TPOT
        for lane in ["vanilla", "static"]:
            sub = wl_df[wl_df["lane"] == lane].sort_values("concurrency")
            if not sub.empty:
                cfg = LANE_CONFIG[lane]
                ax_tpot.plot(
                    sub["concurrency"],
                    sub["tpot_p50"],
                    label=cfg["label"],
                    color=cfg["color"],
                    marker=cfg["marker"],
                    linestyle=cfg["linestyle"],
                )

        ax_tpot.set_xlabel("Concurrency (luồng đồng thời)")
        ax_tpot.set_ylabel("TPOT p50 (ms / token)")
        ax_tpot.grid(True)
        ax_tpot.set_xscale("log", base=2)
        ax_tpot.set_xticks(CONCURRENCY_LEVELS)
        ax_tpot.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Legend chung cho toàn bộ figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.02),
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    out_file = os.path.join(output_dir, "F1_tpot_throughput_concurrency.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"✅ Đã lưu Biểu đồ F1 thành công: {out_file}")


def plot_f2_speedup(df: pd.DataFrame, output_dir: str):
    """
    Sinh Biểu đồ F2:
    - Speedup của Static Draft so với Vanilla Baseline
    - Kẻ vạch chuẩn 1.0x (Baseline)
    - Thể hiện rõ hiện tượng Tăng tốc âm (Negative Speedup) ở W3
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Figure F2: M1 Speedup vs Vanilla Baseline across Workloads (Turing T4 16GB)\n"
        "Vạch đỏ 1.0×: Ngưỡng ranh giới tăng tốc (Dưới 1.0× là tăng tốc âm do chấp nhận nháp thấp)",
        fontweight="bold",
        y=1.02,
    )

    workloads = ["W1", "W2", "W3"]
    colors = {"W1": "#2ca02c", "W2": "#1f77b4", "W3": "#d62728"}

    for idx, wl in enumerate(workloads):
        ax = axes[idx]
        sub = df[(df["workload"] == wl) & (df["lane"] == "static")].sort_values("concurrency")

        if not sub.empty:
            ax.plot(
                sub["concurrency"],
                sub["speedup_vs_vanilla"],
                color=colors[wl],
                marker="D",
                linewidth=2.5,
                markersize=8,
                label=f"Static Draft Speedup ({wl})",
            )

            # Annotate điểm giá trị
            for _, row in sub.iterrows():
                val = row["speedup_vs_vanilla"]
                c = int(row["concurrency"])
                ax.annotate(
                    f"{val:.2f}×",
                    xy=(c, val),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=colors[wl],
                )

        # Kẻ đường 1.0x
        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.8, label="Vạch chuẩn 1.0× (Vanilla)")

        ax.set_title(WORKLOAD_TITLES[wl], fontweight="bold")
        ax.set_xlabel("Concurrency (luồng đồng thời)")
        if idx == 0:
            ax.set_ylabel("Speedup vs. Vanilla (lần)")
        ax.grid(True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONCURRENCY_LEVELS)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0.70, 1.45)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "F2_speedup_vs_concurrency.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Đã lưu Biểu đồ F2 thành công: {out_file}")


def print_summary_table(df: pd.DataFrame):
    """In bảng đối chứng trực quan trên terminal."""
    print("\n" + "=" * 95)
    print(f"{'M1 BASELINE BENCHMARK MATRIX (Turing T4 16GB)':^95}")
    print("=" * 95)
    print(f"{'Workload':<6} | {'Conc':<4} | {'Vanilla Throughput':<18} | {'Static Throughput':<17} | {'Vanilla TPOT':<12} | {'Static TPOT':<11} | {'Speedup':<8}")
    print("-" * 95)

    for wl in ["W1", "W2", "W3"]:
        for c in CONCURRENCY_LEVELS:
            v_row = df[(df["workload"] == wl) & (df["lane"] == "vanilla") & (df["concurrency"] == c)]
            s_row = df[(df["workload"] == wl) & (df["lane"] == "static") & (df["concurrency"] == c)]
            if not v_row.empty and not s_row.empty:
                v_tp = v_row["throughput"].values[0]
                s_tp = s_row["throughput"].values[0]
                v_tpot = v_row["tpot_p50"].values[0]
                s_tpot = s_row["tpot_p50"].values[0]
                sp = s_row["speedup_vs_vanilla"].values[0]
                flag = "🔴 TĂNG TỐC ÂM" if sp < 1.0 else "🟢 TĂNG TỐC"
                print(f"{wl:<6} | {c:<4} | {v_tp:>8.2f} tok/s       | {s_tp:>8.2f} tok/s      | {v_tpot:>6.2f} ms     | {s_tpot:>6.2f} ms    | {sp:>5.2f}× ({flag})")
        print("-" * 95)


def main():
    parser = argparse.ArgumentParser(description="Sinh biểu đồ F1 và F2 cho Cột mốc M1")
    parser.add_argument(
        "--csv",
        default="experiments/results/summary.csv",
        help="Đường dẫn file summary.csv (mặc định: experiments/results/summary.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/figures",
        help="Thư mục lưu ảnh biểu đồ (mặc định: experiments/results/figures)",
    )
    args = parser.parse_args()

    # Chuyển đổi đường dẫn tương đối theo root repo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "experiments" else script_dir

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(repo_root, args.csv)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(repo_root, args.output_dir)

    print(f"📊 Đang nạp dữ liệu từ: {csv_path}")
    df = load_and_validate_data(csv_path)

    print_summary_table(df)

    print(f"\n🎨 Đang xuất biểu đồ vào: {output_dir}")
    plot_f1_throughput_tpot(df, output_dir)
    plot_f2_speedup(df, output_dir)
    print("\n🎉 Hoàn tất sinh toàn bộ biểu đồ thực nghiệm F1 & F2 cho M1 Baseline!")


if __name__ == "__main__":
    main()
