from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments import analyze_m2 as m2


EXPERIMENT_ID = "m2-test-campaign"
EXECUTION_SHA = "a" * 40
WORKLOADS = ("W1", "W2", "W3")
CONCURRENCY = (1, 2, 4, 8, 16, 32)
LANES = ("vanilla", "static", "adaptive")


def make_record(lane, workload, concurrency, repeat, request, *, tpot, spec):
    record = {
        "phase": "measured",
        "success": True,
        "metrics_complete": True,
        "lane": lane,
        "workload": workload,
        "concurrency": concurrency,
        "repeat_id": f"r{repeat:03d}",
        "run_id": f"run-{lane}-{workload}-{concurrency}-{repeat}",
        "request_id": f"req-{lane}-{workload}-{concurrency}-{repeat}-{request}",
        "ttft_ms": 50.0,
        "tpot_ms": tpot,
        "e2e_ms": 500.0,
        "run_vram_peak_mb": 12000.0,
        "run_duration_s": 1.0 + 0.1 * repeat,
        "output_tokens": 100,
        "gpu": "Tesla T4",
        "commit": EXECUTION_SHA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_version": f"{workload}-v1",
        "workload_file_sha256": f"{workload}-sha",
        "server_metrics": None,
    }
    if spec is not None:
        accepted, drafted, steps = spec
        record["server_metrics"] = {
            "speculative_decoding": {
                "num_accepted_draft_tokens": accepted,
                "num_draft_tokens": drafted,
                "num_spec_steps": steps,
                "num_spec_tokens": 3,
            }
        }
    return record


def default_tpot(lane, workload, concurrency):
    # Vanilla 30ms; Static 20ms (1.5x); Adaptive 25ms (1.2x) unless overridden.
    return {"vanilla": 30.0, "static": 20.0, "adaptive": 25.0}[lane]


def build_campaign(root: Path, *, tpot_fn=default_tpot, skip_cells=()):
    cells_dir = root / "cells"
    triplets_dir = root / "triplets"
    cells_dir.mkdir(parents=True)
    triplets_dir.mkdir()

    for workload in WORKLOADS:
        for concurrency in CONCURRENCY:
            hashes = {}
            for lane in LANES:
                cell_id = f"{lane}__{workload}__c{concurrency}"
                if cell_id in skip_cells:
                    continue
                spec = None
                if lane == "static":
                    spec = (6, 9, 3)
                elif lane == "adaptive":
                    spec = (4, 8, 4)
                base = tpot_fn(lane, workload, concurrency)
                rows = [
                    make_record(
                        lane, workload, concurrency, repeat, request,
                        # Repeat-level TPOT spread: base-0.2 .. base+0.2.
                        tpot=base + 0.1 * (repeat - 3),
                        spec=spec,
                    )
                    for repeat in range(1, 6)
                    for request in range(2)
                ]
                cell_dir = cells_dir / cell_id
                cell_dir.mkdir()
                raw = cell_dir / "raw.jsonl"
                raw.write_bytes(
                    "".join(json.dumps(row) + "\n" for row in rows).encode()
                )
                hashes[cell_id] = hashlib.sha256(raw.read_bytes()).hexdigest()

            marker = {
                "status": "COMPLETE_VALID",
                "experiment_id": EXPERIMENT_ID,
                "execution_sha": EXECUTION_SHA,
                "workload": workload,
                "concurrency": concurrency,
                "lanes": list(LANES),
                "cells": [f"{lane}__{workload}__c{concurrency}" for lane in LANES],
                "raw_sha256": hashes,
                "variance_threshold": 0.15,
            }
            (triplets_dir / f"{workload}__c{concurrency}.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
    return root


@pytest.fixture
def campaign(tmp_path):
    return build_campaign(tmp_path / "campaign")


def row_for(rows, lane, workload, concurrency):
    matches = [
        row for row in rows
        if (row["lane"], row["workload"], row["concurrency"])
        == (lane, workload, concurrency)
    ]
    assert len(matches) == 1
    return matches[0]


def test_summary_has_54_rows_with_matched_speedups(campaign):
    rows = m2.summarize_campaign(campaign)

    assert len(rows) == 54
    adaptive = row_for(rows, "adaptive", "W3", 16)
    static = row_for(rows, "static", "W3", 16)
    vanilla = row_for(rows, "vanilla", "W3", 16)
    assert vanilla["speedup_vs_vanilla"] == 1.0
    assert static["speedup_vs_vanilla"] == pytest.approx(30.0 / 20.0)
    assert adaptive["speedup_vs_vanilla"] == pytest.approx(30.0 / 25.0)
    assert {row["experiment_id"] for row in rows} == {EXPERIMENT_ID}


def test_adaptive_alpha_tau_use_native_counters(campaign):
    rows = m2.summarize_campaign(campaign)

    adaptive = row_for(rows, "adaptive", "W2", 4)
    # Summed counters: accepted=4, drafted=8, steps=4 per request.
    assert adaptive["alpha"] == pytest.approx(0.5)
    assert adaptive["tau"] == pytest.approx(2.0)
    static = row_for(rows, "static", "W2", 4)
    assert static["alpha"] == pytest.approx(6 / 9)
    assert row_for(rows, "vanilla", "W2", 4)["alpha"] == "N/A"


def test_summary_reports_repeat_level_min_max(campaign):
    rows = m2.summarize_campaign(campaign)

    vanilla = row_for(rows, "vanilla", "W1", 1)
    assert vanilla["tpot_p50"] == pytest.approx(30.0)
    assert vanilla["tpot_p50_min"] == pytest.approx(29.8)
    assert vanilla["tpot_p50_max"] == pytest.approx(30.2)
    # Throughput per repeat = 200 tokens / (1.0 + 0.1 * repeat) seconds.
    assert vanilla["throughput_max"] == pytest.approx(200 / 1.1, abs=0.01)
    assert vanilla["throughput_min"] == pytest.approx(200 / 1.5, abs=0.01)
    assert vanilla["variance_ratio"] == pytest.approx(0.4 / 30.0, abs=1e-6)
    assert len(vanilla["raw_sha256"]) == 64


def test_raw_sha_mismatch_is_rejected(campaign):
    raw = campaign / "cells" / "adaptive__W3__c8" / "raw.jsonl"
    # CRLF conversion is the realistic corruption on Windows checkouts.
    raw.write_bytes(raw.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(ValueError, match="adaptive__W3__c8"):
        m2.summarize_campaign(campaign)


def test_missing_cell_is_rejected(tmp_path):
    root = build_campaign(tmp_path / "c", skip_cells={"static__W2__c2"})

    with pytest.raises(ValueError, match="static__W2__c2"):
        m2.summarize_campaign(root)


def test_incomplete_triplet_marker_is_rejected(campaign):
    marker = campaign / "triplets" / "W1__c4.json"
    data = json.loads(marker.read_text(encoding="utf-8"))
    data["status"] = "MISSING"
    marker.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="W1__c4"):
        m2.summarize_campaign(campaign)


def _edit_marker(campaign, name, mutate):
    marker = campaign / "triplets" / f"{name}.json"
    data = json.loads(marker.read_text(encoding="utf-8"))
    marker.write_text(json.dumps(mutate(data)), encoding="utf-8")


@pytest.mark.parametrize("field", ["experiment_id", "execution_sha"])
def test_marker_without_campaign_identity_is_rejected(campaign, field):
    for workload in WORKLOADS:
        for concurrency in CONCURRENCY:
            _edit_marker(
                campaign, f"{workload}__c{concurrency}",
                lambda data: {k: v for k, v in data.items() if k != field},
            )

    with pytest.raises(ValueError, match=field):
        m2.summarize_campaign(campaign)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: [data],
        lambda data: {**data, "raw_sha256": list(data["raw_sha256"].values())},
    ],
    ids=["marker-not-object", "raw-sha-not-object"],
)
def test_malformed_marker_fails_cleanly(campaign, tmp_path, mutate):
    _edit_marker(campaign, "W2__c8", mutate)

    code = m2.main([
        "--campaign-root", str(campaign),
        "--output-csv", str(tmp_path / "s.csv"),
        "--gate-json", str(tmp_path / "g.json"),
    ])

    assert code == 1
    with pytest.raises(ValueError, match="W2__c8"):
        m2.summarize_campaign(campaign)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: {**data, "workload": "W2"},
        lambda data: {**data, "concurrency": 8},
        lambda data: {**data, "cells": data["cells"][:2]},
    ],
    ids=["workload", "concurrency", "cells"],
)
def test_marker_metadata_must_match_file_name(campaign, mutate):
    _edit_marker(campaign, "W1__c4", mutate)

    with pytest.raises(ValueError, match="W1__c4"):
        m2.summarize_campaign(campaign)


def test_w3_gate_passes_only_when_all_six_cells_reach_one(campaign):
    gate = m2.evaluate_w3_gate(m2.summarize_campaign(campaign))

    assert gate["status"] == "PASS"
    assert [cell["concurrency"] for cell in gate["cells"]] == list(CONCURRENCY)
    assert gate["threshold"] == 1.0


def test_w3_gate_fails_when_any_cell_below_one(tmp_path):
    def tpot(lane, workload, concurrency):
        if lane == "adaptive" and workload == "W3" and concurrency == 32:
            return 31.0
        return default_tpot(lane, workload, concurrency)

    rows = m2.summarize_campaign(build_campaign(tmp_path / "c", tpot_fn=tpot))
    gate = m2.evaluate_w3_gate(rows)

    assert gate["status"] == "FAIL"
    failing = [cell for cell in gate["cells"] if not cell["pass"]]
    assert [cell["concurrency"] for cell in failing] == [32]
    assert failing[0]["speedup"] == pytest.approx(30.0 / 31.0)


def test_w3_gate_boundary_exactly_one_passes(tmp_path):
    def tpot(lane, workload, concurrency):
        if lane == "adaptive":
            return 30.0
        return default_tpot(lane, workload, concurrency)

    gate = m2.evaluate_w3_gate(
        m2.summarize_campaign(build_campaign(tmp_path / "c", tpot_fn=tpot))
    )

    assert gate["status"] == "PASS"
    assert all(cell["speedup"] == 1.0 for cell in gate["cells"])


def test_cli_writes_csv_and_gate_json(campaign, tmp_path):
    out_csv = tmp_path / "out" / "m2_summary.csv"
    out_gate = tmp_path / "out" / "m2_w3_gate.json"

    code = m2.main([
        "--campaign-root", str(campaign),
        "--output-csv", str(out_csv),
        "--gate-json", str(out_gate),
    ])

    assert code == 0
    with out_csv.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == m2.M2_SUMMARY_COLUMNS
        assert len(list(reader)) == 54
    gate = json.loads(out_gate.read_text(encoding="utf-8"))
    assert gate["status"] == "PASS"
    assert gate["experiment_id"] == EXPERIMENT_ID


def test_cli_gate_fail_is_still_a_successful_analysis(tmp_path):
    def tpot(lane, workload, concurrency):
        return 40.0 if lane == "adaptive" else default_tpot(lane, workload, concurrency)

    root = build_campaign(tmp_path / "c", tpot_fn=tpot)
    out_gate = tmp_path / "gate.json"

    code = m2.main([
        "--campaign-root", str(root),
        "--output-csv", str(tmp_path / "s.csv"),
        "--gate-json", str(out_gate),
    ])

    assert code == 0
    assert json.loads(out_gate.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_cli_validation_error_returns_nonzero(tmp_path):
    root = build_campaign(tmp_path / "c", skip_cells={"vanilla__W1__c1"})

    code = m2.main([
        "--campaign-root", str(root),
        "--output-csv", str(tmp_path / "s.csv"),
        "--gate-json", str(tmp_path / "g.json"),
    ])

    assert code == 1
    assert not (tmp_path / "s.csv").exists()
