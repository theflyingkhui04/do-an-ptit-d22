"""CPU-only checks for Dai's M2 execution preparation and noise readiness."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from experiments import aggregate_results as agg
from experiments import run_m2_campaign as campaign


def records():
    result = []
    for lane, _, _, _ in campaign.LANES:
        for workload in campaign.WORKLOADS:
            for concurrency in (1, 2, 4, 8, 16, 32):
                for repeat in range(1, 6):
                    result.append({
                        "phase": "measured", "success": True, "metrics_complete": True,
                        "lane": lane, "workload": workload, "concurrency": concurrency,
                        "repeat_id": f"r{repeat:03}", "run_id": f"run-{repeat}",
                        "tpot_ms": 20.0, "ttft_ms": 50.0, "e2e_ms": 250.0,
                        "run_duration_s": 1.0, "output_tokens": 10,
                        "run_vram_peak_mb": 12000.0,
                        "server_metrics": {"speculative_decoding": {
                            "num_accepted_draft_tokens": 1, "num_draft_tokens": 2,
                            "num_spec_steps": 1,
                        }},
                    })
    return result


def summaries():
    return agg._aggregate_measured_records(records())


def test_plan_is_54_unique_interleaved_cells_with_identical_inputs(tmp_path):
    plan = campaign.build_plan(tmp_path / "new-attempt")
    assert len(plan) == len({item["cell"] for item in plan}) == 54
    for index in range(0, 54, 3):
        group = plan[index:index + 3]
        parsed = []
        for item in group:
            args = item["harness"][2:]
            parsed.append(dict(zip(args[::2], args[1::2])))
        assert [args["--lane"] for args in parsed] == ["vanilla", "static", "adaptive"]
        assert [item["switch"][-1] for item in group] == ["lane1", "lane2", "lane3"]
        for field in ("--workload", "--concurrency", "--workload-file", "--sample-policy",
                      "--repeats", "--num-requests", "--warmup-requests", "--experiment-id"):
            assert len({args[field] for args in parsed}) == 1
        assert parsed[0]["--repeats"] == "5"
        assert parsed[0]["--sample-policy"] == "cycle-rotate"
        assert parsed[0]["--num-requests"] == "32"
        assert parsed[0]["--warmup-requests"] == "3"
        assert all("--skip-env-check" not in item["harness"] for item in group)
        assert all("--overwrite" not in item["harness"] for item in group)
        assert all("--seed" not in item["harness"] and "--temperature" not in item["harness"]
                   for item in group)  # Frozen package supplies these.


@pytest.mark.parametrize("repeats", [0, 1, 4])
def test_plan_rejects_insufficient_repeats(tmp_path, repeats):
    with pytest.raises(ValueError, match=">=5"):
        campaign.build_plan(tmp_path, repeats)


def test_default_cli_prints_plan_without_execution_or_writes(tmp_path, capsys):
    with patch.object(sys, "argv", ["run", "--output-dir", str(tmp_path / "new"),
                                    "--execution-sha", "a" * 40]), \
         patch.object(campaign, "execute") as execute:
        assert campaign.main() == 0
    execute.assert_not_called()
    assert len(json.loads(capsys.readouterr().out)["cells"]) == 54
    assert not (tmp_path / "new").exists()


def test_execute_calls_existing_tools_in_order_and_isolates_logs(tmp_path):
    output = tmp_path / "attempt"
    plan = campaign.build_plan(output)
    with patch.object(campaign, "verify_execution") as verify, \
         patch.object(campaign.subprocess, "run") as run:
        campaign.execute(plan, output, "a" * 40)
    verify.assert_called_once_with("a" * 40)
    calls = run.call_args_list
    assert len(calls) == 109  # switch + harness per cell, then readiness check
    for index, item in enumerate(plan):
        switch, harness = calls[index * 2:index * 2 + 2]
        assert switch.args[0] == item["switch"]
        assert harness.args[0] == item["harness"]
        env = switch.kwargs["env"]
        assert Path(env["LANE_LOG_DIR"]).parent == output / item["cell"]
        assert Path(env["ADAPTIVE_TELEMETRY_PATH"]).parent == output / item["cell"]
        assert switch.kwargs["check"] and harness.kwargs["check"]
    assert "--check-m2" in calls[-1].args[0]
    assert json.loads((output / "plan.json").read_text())["execution_sha"] == "a" * 40


def test_existing_evidence_is_never_overwritten(tmp_path):
    evidence = tmp_path / "raw.jsonl"
    evidence.write_bytes(b"historical evidence\n")
    with patch.object(campaign, "verify_execution") as verify, \
         patch.object(campaign.subprocess, "run") as run:
        with pytest.raises(FileExistsError):
            campaign.execute(campaign.build_plan(tmp_path), tmp_path, "a" * 40)
    verify.assert_not_called()
    run.assert_not_called()
    assert evidence.read_bytes() == b"historical evidence\n"


def test_failed_cell_stops_and_retains_attempt(tmp_path):
    output = tmp_path / "attempt"
    plan = campaign.build_plan(output)
    with patch.object(campaign, "verify_execution"), \
         patch.object(campaign.subprocess, "run", side_effect=[
             subprocess.CompletedProcess([], 0), subprocess.CalledProcessError(1, "harness"),
         ]) as run:
        with pytest.raises(subprocess.CalledProcessError):
            campaign.execute(plan, output, "a" * 40)
    assert run.call_count == 2
    assert (output / "plan.json").exists()
    assert (output / plan[0]["cell"] / "harness.log").exists()
    assert not (output / plan[1]["cell"]).exists()


@pytest.mark.parametrize("responses,message", [
    (["b" * 40], "SHA"),
    (["a" * 40, " M experiments/benchmark_harness.py"], "clean"),
])
def test_execution_rejects_wrong_sha_or_dirty_checkout(responses, message):
    with patch.object(campaign.subprocess, "check_output", side_effect=responses):
        with pytest.raises(ValueError, match=message):
            campaign.verify_execution("a" * 40)


def test_execution_checks_all_artifacts_before_switching():
    from experiments import workload_adapter as adapter
    with patch.object(campaign.subprocess, "check_output", side_effect=["a" * 40, ""]), \
         patch.object(adapter, "verify_workload_integrity") as integrity, \
         patch.object(adapter, "load_workload_rows", return_value=[{"workload_id": "test"}]), \
         patch.object(adapter, "validate_manifest_consistency") as consistency, \
         patch.object(adapter, "validate_workload_label") as label:
        campaign.verify_execution("a" * 40)
    assert integrity.call_count == consistency.call_count == label.call_count == 3
    assert [call.args[0] for call in label.call_args_list] == ["W1", "W2", "W3"]


def test_clean_full_campaign_readiness_does_not_assert_w3_gate():
    report = agg.check_m2_cells(summaries())
    assert report["status"] == "PASS"
    assert report["cell_count"] == 54
    assert report["w3_speedup_gate"] == "REQUIRES_EMPIRICAL_ANALYSIS"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected", "four_repeats"])
def test_readiness_rejects_invalid_campaign(mutation):
    rows = summaries()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "unexpected":
        rows[0]["concurrency"] = 64
    else:
        rows[0]["n_runs"] = 4
    report = agg.check_m2_cells(rows)
    assert report["status"] == "FAIL"
    assert report["errors"]


@pytest.mark.parametrize("maximum,noisy", [(23.0, False), (23.001, True)])
def test_variance_boundary_uses_repeat_medians_and_retains_noisy_evidence(maximum, noisy):
    raw = records()
    raw[4]["tpot_ms"] = maximum  # First cell: 20,20,20,20,max.
    rows = agg._aggregate_measured_records(raw)
    cell = next(row for row in rows if (row["lane"], row["workload"], row["concurrency"])
                == ("vanilla", "W1", 1))
    assert cell["variance_flag"] is noisy
    before = copy.deepcopy(rows)
    report = agg.check_m2_cells(rows)
    assert bool(report["noisy_cells_rerun_required"]) is noisy
    assert report["status"] == ("FAIL" if noisy else "PASS")
    assert rows == before


def test_checker_rejects_stale_ten_percent_flag():
    rows = summaries()
    rows[0].update(_tpot_max=22.5, variance_flag=True)
    assert agg.check_m2_cells(rows)["errors"]


def test_check_cli_emits_readiness_json_and_never_writes_csv(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in records()), encoding="utf-8")
    before = raw.read_bytes()
    output = tmp_path / "must-not-exist.csv"
    stream = io.StringIO()
    with patch.object(sys, "argv", ["aggregate", "--check-m2", "--raw", str(raw),
                                    "--output-csv", str(output)]), contextlib.redirect_stdout(stream):
        assert agg.main() == 0
    assert json.loads(stream.getvalue())["status"] == "PASS"
    assert raw.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("bad_input", ["duplicate_file", "failed_request", "incomplete_metrics"])
def test_check_cli_reports_invalid_raw_without_discarding_it(tmp_path, bad_input):
    raw = tmp_path / "raw.jsonl"
    rows = records()
    if bad_input == "failed_request":
        rows[0]["success"] = False
    elif bad_input == "incomplete_metrics":
        rows[0]["metrics_complete"] = False
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = raw.read_bytes()
    args = ["aggregate", "--check-m2", "--raw", str(raw)]
    if bad_input == "duplicate_file":
        args.append(str(raw))
    stream = io.StringIO()
    with patch.object(sys, "argv", args), contextlib.redirect_stdout(stream):
        assert agg.main() == 1
    report = json.loads(stream.getvalue())
    assert report["status"] == "FAIL" and report["errors"]
    assert raw.read_bytes() == before
