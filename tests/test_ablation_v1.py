"""[Adaptive v1] Ablation campaign planning and analysis (pure logic, no GPU)."""

import pytest

from experiments import ablation_v1 as ab


def test_default_plan_covers_three_workloads_and_concurrencies():
    cells = ab.plan_cells()

    assert [(c["workload"], c["concurrency"]) for c in cells] == [
        (w, c) for w in ("W1", "W2", "W3") for c in (1, 4, 16)
    ]
    assert {c["repeats"] for c in cells} == {3}
    assert {c["num_requests"] for c in cells} == {16}


def test_lanes_include_baselines_v0_and_variants():
    assert ab.LANES == ("vanilla", "static", "adaptive", "adaptive-f1", "adaptive-f12", "adaptive-v1", "adaptive-v2")


def test_campaign_dir_rejects_path_tricks(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "REPO_ROOT", tmp_path)

    assert ab.campaign_dir("abl-1") == (tmp_path / "experiments" / "runs" / "m2_02_v1" / "abl-1").resolve()
    for bad in ("../x", "a/b", ""):
        with pytest.raises(ValueError):
            ab.campaign_dir(bad)


def row(lane, workload, c, tpot):
    return {"lane": lane, "workload": workload, "concurrency": c, "tpot_p50": tpot}


def test_add_speedups_against_vanilla_and_static():
    rows = [row("vanilla", "W3", 1, 25.0), row("static", "W3", 1, 26.0), row("adaptive-v1", "W3", 1, 24.0)]

    out = {r["lane"]: r for r in ab.add_speedups(rows)}

    assert out["adaptive-v1"]["speedup_vs_vanilla"] == pytest.approx(25 / 24)
    assert out["adaptive-v1"]["speed_vs_static"] == pytest.approx(26 / 24)
    assert out["vanilla"]["speedup_vs_vanilla"] == pytest.approx(1.0)


def test_criteria_pass_and_fail():
    rows = ab.add_speedups([
        row("vanilla", "W3", 1, 25.0), row("static", "W3", 1, 26.0), row("adaptive-v1", "W3", 1, 25.4),
        row("vanilla", "W1", 1, 28.0), row("static", "W1", 1, 21.0), row("adaptive-v1", "W1", 1, 21.5),
        row("vanilla", "W2", 1, 25.0), row("static", "W2", 1, 20.0), row("adaptive-v1", "W2", 1, 21.0),
    ])

    result = ab.evaluate_criteria(rows, lane="adaptive-v1")

    w3, w1, w2 = (next(c for c in result["checks"] if c["workload"] == w) for w in ("W3", "W1", "W2"))
    assert w3["pass"] is True    # 25/25.4 = 0.984 >= 0.98
    assert w1["pass"] is True    # 21/21.5 = 0.977 >= 0.97
    assert w2["pass"] is False   # 20/21 = 0.952 < 0.97
    assert result["status"] == "FAIL"


def test_criteria_missing_cells_are_reported():
    result = ab.evaluate_criteria(ab.add_speedups([row("vanilla", "W3", 1, 25.0)]), lane="adaptive-v1")

    assert result["status"] == "INCOMPLETE"


def test_split_telemetry_by_cell_window():
    windows = {("W1", 1): (100.0, 200.0), ("W2", 1): (300.0, 400.0)}
    lines = [{"finished_at": 150.0, "request_id": "a"}, {"finished_at": 350.0, "request_id": "b"},
             {"finished_at": 250.0, "request_id": "warmup-gap"}]

    parts = ab.split_telemetry(lines, windows)

    assert [r["request_id"] for r in parts[("W1", 1)]] == ["a"]
    assert [r["request_id"] for r in parts[("W2", 1)]] == ["b"]


def test_telemetry_summary_merges_counts():
    lines = [
        {"reason_counts": {"high": 3, "floor": 1}, "batch_k_counts": {"3": 4}, "tree_budgets": [3, 3, 3, 2]},
        {"reason_counts": {"floor": 2}, "batch_k_counts": {"3": 1, "2": 1}, "tree_budgets": [2, 2]},
    ]

    summary = ab.telemetry_summary(lines)

    assert summary["requests"] == 2
    assert summary["reason_counts"] == {"high": 3, "floor": 3}
    assert summary["batch_k_counts"] == {"3": 5, "2": 1}
    assert summary["k_share"] == {"3": pytest.approx(3 / 6), "2": pytest.approx(3 / 6)}


def test_cell_window_from_raw_records():
    raw = [{"run_started_at_utc": "2026-10-25T00:00:10+00:00", "run_finished_at_utc": "2026-10-25T00:01:00+00:00"},
           {"run_started_at_utc": "2026-10-25T00:01:05+00:00", "run_finished_at_utc": "2026-10-25T00:02:00+00:00"}]

    start, end = ab.cell_window(raw)

    assert end - start == pytest.approx(110.0)


def test_summarize_ablation_on_recorded_l4_cell(tmp_path):
    import shutil
    from pathlib import Path

    from experiments import modal_runner

    src = Path(__file__).resolve().parents[1] / "experiments/runs/m3_01/20261025-m301-recheck-vanilla-W1-c1/raw.vram.jsonl"
    cell_dir = tmp_path / "vanilla" / "W1-c1"
    cell_dir.mkdir(parents=True)
    shutil.copy(src, cell_dir / "raw.vram.jsonl")

    summary = modal_runner._summarize_ablation(
        tmp_path, ["vanilla"], [{"workload": "W1", "concurrency": 1, "repeats": 1, "num_requests": 8}])

    assert summary["rows"][0]["tpot_p50"] == pytest.approx(19.19)
    assert summary["rows"][0]["gpu"] == "NVIDIA L4"
    assert summary["criteria"]["status"] == "INCOMPLETE"


# ------------------------------------------------------------- robustness (L4 capacity, 29/10)
def test_run_cells_records_a_failed_cell_and_continues():
    def run_one(cell):
        if cell["workload"] == "W1":
            raise TimeoutError("health not ready after 1200 s")
        return {"returncode": 0}

    results = ab.run_cells(ab.plan_cells(("W1", "W2"), (1,)), run_one)

    assert [r["cell"]["workload"] for r in results] == ["W1", "W2"]
    assert results[0]["error"] == "TimeoutError: health not ready after 1200 s"
    assert results[0]["returncode"] is None
    assert results[1]["returncode"] == 0
    assert "error" not in results[1]


def test_default_ready_timeout_allows_for_l4_queueing():
    assert ab.READY_TIMEOUT_S >= 45 * 60


def test_summary_excludes_and_counts_spec_records_without_server_metrics(tmp_path):
    # 29/10 L4: vLLM returned "metrics": null for sample w2_sql_v2_003 on the static lane.
    import json
    import shutil
    from pathlib import Path

    from experiments import modal_runner

    src = Path(__file__).resolve().parents[1] / "experiments/runs/m2_02_v1/abl-20261029/static/W2-c1/raw.vram.jsonl"
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = sum(r.get("server_metrics") is None for r in records)
    cell_dir = tmp_path / "static" / "W2-c1"
    cell_dir.mkdir(parents=True)
    shutil.copy(src, cell_dir / "raw.vram.jsonl")

    summary = modal_runner._summarize_ablation(
        tmp_path, ["static"], [{"workload": "W2", "concurrency": 1, "repeats": 3, "num_requests": 16}])

    row = summary["rows"][0]
    assert missing == 2
    assert row["spec_metrics_missing"] == 2
    assert row["alpha"] not in (None, "N/A")


def test_merge_jsonl_keeps_existing_lines_and_appends_new_ones():
    old = '{"a": 1}\n{"a": 2}\n'
    new = '{"a": 2}\n{"a": 3}\n'
    assert ab.merge_jsonl(old, new) == '{"a": 1}\n{"a": 2}\n{"a": 3}\n'
    assert ab.merge_jsonl("", new) == new
    assert ab.merge_jsonl(old, "") == old


def test_latest_attempt_keeps_only_the_last_repeat_series():
    def rec(run_id, repeat_id):
        return {"run_id": run_id, "repeat_id": repeat_id, "phase": "measured"}

    # an aborted attempt (r001, r002) appended before the complete one on the same volume file
    records = [rec("a", "r001"), rec("b", "r002"), rec("c", "r001"), rec("c", "r001"),
               rec("d", "r002"), rec("e", "r003")]
    assert [r["run_id"] for r in ab.latest_attempt(records)] == ["c", "c", "d", "e"]
    clean = [rec("x", "r001"), rec("y", "r002")]
    assert ab.latest_attempt(clean) == clean
    assert ab.latest_attempt([]) == []


def test_summary_uses_only_the_complete_attempt_of_a_cell(tmp_path):
    # 29/10-30/10 L4: static W1-c1 raw holds 2 repeats of a cut-off attempt (29/10) and the
    # complete rerun (30/10); only the rerun may enter the cell statistics.
    import shutil
    from pathlib import Path

    from experiments import modal_runner

    src = Path(__file__).resolve().parents[1] / "experiments/runs/m2_02_v1/abl-20261029/static/W1-c1/raw.vram.jsonl"
    cell_dir = tmp_path / "static" / "W1-c1"
    cell_dir.mkdir(parents=True)
    shutil.copy(src, cell_dir / "raw.vram.jsonl")

    summary = modal_runner._summarize_ablation(
        tmp_path, ["static"], [{"workload": "W1", "concurrency": 1, "repeats": 3, "num_requests": 16}])

    row = summary["rows"][0]
    assert row["stale_records_dropped"] == 32
    assert row["n_measured"] == 48
