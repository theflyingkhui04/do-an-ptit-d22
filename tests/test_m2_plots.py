from __future__ import annotations

import csv

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from experiments import analyze_m2 as m2
from experiments import plot_m2


def write_summary(path, *, adaptive_w3_speedup=0.6, drop=None):
    rows = []
    for workload in m2.WORKLOADS:
        for concurrency in m2.CONCURRENCY_LEVELS:
            for lane in m2.LANES:
                if drop == (lane, workload, concurrency):
                    continue
                speedup = {"vanilla": 1.0, "static": 1.2, "adaptive": 1.1}[lane]
                if lane == "adaptive" and workload == "W3":
                    speedup = adaptive_w3_speedup
                tpot = 30.0 / speedup
                rows.append({
                    "lane": lane, "workload": workload, "concurrency": concurrency,
                    "tpot_p50": tpot, "tpot_p50_min": tpot - 0.5,
                    "tpot_p50_max": tpot + 0.5, "throughput": 100.0 * concurrency,
                    "throughput_min": 95.0 * concurrency,
                    "throughput_max": 105.0 * concurrency,
                    "speedup_vs_vanilla": speedup, "gpu": "Tesla T4", "n_runs": 5,
                    "experiment_id": "m2-test",
                })
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loads_complete_three_lane_matrix(tmp_path):
    df = plot_m2.load_m2_summary(write_summary(tmp_path / "s.csv"))

    assert len(df) == 54
    assert set(df["lane"]) == set(m2.LANES)


def test_rejects_incomplete_matrix(tmp_path):
    path = write_summary(tmp_path / "s.csv", drop=("adaptive", "W3", 8))

    with pytest.raises(ValueError, match="adaptive.*W3.*8"):
        plot_m2.load_m2_summary(path)


@pytest.mark.parametrize(
    "column, bad_value",
    [("tpot_p50_min", 1000.0), ("throughput_max", 0.0)],
)
def test_rejects_value_outside_min_max(tmp_path, column, bad_value):
    path = write_summary(tmp_path / "s.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    first = lines[1].split(",")
    first[header.index(column)] = str(bad_value)
    lines[1] = ",".join(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=column.rsplit("_", 1)[0]):
        plot_m2.load_m2_summary(path)


@pytest.mark.parametrize(
    "static, adaptive, expected",
    [
        (1.284, 1.314, (-1, 1)),  # W1/c1: adaptive above static
        (0.969, 0.752, (1, -1)),
        (1.0, 1.0, (1, -1)),
    ],
)
def test_higher_lane_label_goes_above(static, adaptive, expected):
    static_offset, adaptive_offset = plot_m2.label_offsets(static, adaptive)

    assert (static_offset > 0, adaptive_offset > 0) == (expected[0] > 0, expected[1] > 0)


@pytest.mark.parametrize("w3_speedup", [0.3, 0.6, 1.9])
def test_speedup_axis_never_clips_points_and_shows_parity(tmp_path, w3_speedup):
    df = plot_m2.load_m2_summary(
        write_summary(tmp_path / "s.csv", adaptive_w3_speedup=w3_speedup)
    )

    low, high = plot_m2.speedup_ylim(df)

    values = df.loc[df["lane"] != "vanilla", "speedup_vs_vanilla"]
    assert low < min(values.min(), 1.0)
    assert high > max(values.max(), 1.0)


def test_writes_f1_and_f2_figures(tmp_path):
    df = plot_m2.load_m2_summary(write_summary(tmp_path / "s.csv"))

    f1 = plot_m2.plot_f1(df, tmp_path / "fig")
    f2 = plot_m2.plot_f2(df, tmp_path / "fig")

    assert f1.name == "F1_m2_tpot_throughput_3lanes.png"
    assert f2.name == "F2_m2_speedup_vs_vanilla.png"
    assert f1.stat().st_size > 0 and f2.stat().st_size > 0
