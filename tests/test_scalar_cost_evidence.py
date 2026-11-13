"""Synthetic evidence only; these tests do not establish GPU timing validity."""
import copy
import json

import pytest

from experiments import scalar_cost_evidence as p
from experiments import modal_lanes as lanes


def fixture():
    identity = dict(gpu_name="NVIDIA L4", sm_architecture="sm_89", driver="fixture",
                    cuda="13.0", torch="2.13.0+cu130", vllm=lanes.VLLM_VERSION,
                    target_model=lanes.TARGET_MODEL, target_revision=lanes.TARGET_REVISION,
                    draft_model=lanes.DRAFT_MODEL, draft_revision=lanes.DRAFT_REVISION,
                    draft_method="draft_model", dtype="bfloat16", quantization="awq_marlin",
                    effective_quantization="auto_awq",
                    scheduler_cls="src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
                    scheduler_base_cls="src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
                    runtime_config_sha256="a" * 64, protocol=p.PROTOCOL)
    cells = [dict(cell_id=f"k{k}", shared_k=k, request_budgets=[k],
                  context_bin=[32, 256], reference_cell_id="k0") for k in (0, 2, 3, 4)]
    manifest = dict(run_id="synthetic-only", git_sha="a" * 40,
                    timestamp_utc="2026-11-11T00:00:00Z", observed_identity=identity,
                    expected_identity=copy.deepcopy(identity), warmup=5, repeats=50,
                    cycles=8, cells=cells)
    raw = []
    for cell in cells:
        k = cell["shared_k"]
        for phase, count in (("warmup", 5), ("measured", 50)):
            for index in range(count):
                gap = 100_000_000 if phase == "warmup" else 10_000_000
                events = []
                cohort = f"{cell['cell_id']}-{phase}-{index}"
                for cycle in range(8):
                    events.append(dict(verifier_id=f"v{cycle}", end_ns=(cycle + 1) * gap,
                                       shared_k=k, producer_rows=1, proposer_calls=1,
                                       draft_forward_calls=max(1, k), requests=[dict(
                                           request_id="r", effective_k=k, context_tokens=40 + cycle,
                                           query_tokens=k + 1, drafted=k, accepted=k,
                                           emitted=k + 1, prefill=False, terminal=False, stale=False)]))
                raw.append(dict(run_id=manifest["run_id"], git_sha=manifest["git_sha"],
                                cell_id=cell["cell_id"], phase=phase, repeat=index, status="ok",
                                timing_boundary=p.BOUNDARY, clock="perf_counter_ns",
                                cohort_id=cohort, clock_domain=f"synthetic-process-{cohort}", start_ns=0,
                                end_ns=8 * gap, retirements=events, tree_execution_verified=False))
    return manifest, raw


def test_complete_matrix_and_pooled_arithmetic_exclude_warmup():
    manifest, raw = fixture()
    result = p.validate_run(manifest, raw)
    assert result["complete"] and not result["consumer_profile"]
    for row, k in zip(result["summary"], (0, 2, 3, 4)):
        assert row["median_ms_per_retired_batch"] == 10
        assert row["mean_emitted_tokens_per_batch"] == k + 1
        assert row["pooled_emitted_tokens_per_ms"] == (k + 1) / 10
        assert row["variance_ratio"] == 0


@pytest.mark.parametrize("field", p.IDENTITY_FIELDS)
def test_identity_mismatch_fails_closed(field):
    manifest, _ = fixture()
    manifest["observed_identity"][field] = "different"
    with pytest.raises(ValueError, match="Identity mismatch"):
        p.validate_identity(manifest["observed_identity"], manifest["expected_identity"])


@pytest.mark.parametrize("field,value", [
    ("gpu_name", "NVIDIA A10"), ("draft_model", lanes.EAGLE_DRAFT_MODEL),
    ("draft_method", "eagle3"), ("sm_architecture", "sm_75"),
    ("protocol", "m4-component-calibration-v1"), ("vllm", "latest"),
])
def test_matching_but_noncanonical_identity_is_rejected(field, value):
    manifest, _ = fixture()
    for key in ("observed_identity", "expected_identity"):
        manifest[key][field] = value
    with pytest.raises(ValueError):
        p.validate_identity(manifest["observed_identity"], manifest["expected_identity"])


@pytest.mark.parametrize("mutation", [
    "negative_duration", "nan_duration", "boundary", "missing_shape", "query", "context",
    "prefill", "terminal", "stale", "k_label", "request_budget", "acceptance",
    "emission", "missing_k0_sync", "producer_rows", "duplicate_verifier", "tree_claim",
])
def test_invalid_timing_and_shape_are_rejected(mutation):
    manifest, raw = fixture()
    record = raw[5]  # measured K=0
    event = record["retirements"][0]
    row = event["requests"][0]
    if mutation == "negative_duration": event["end_ns"] = -1
    elif mutation == "nan_duration": event["end_ns"] = float("nan")
    elif mutation == "boundary": record["timing_boundary"] = "cpu-dispatch-only"
    elif mutation == "missing_shape": event["requests"] = []
    elif mutation == "query": row["query_tokens"] = 4
    elif mutation == "context": row["context_tokens"] = 1024
    elif mutation in ("prefill", "terminal", "stale"): row[mutation] = True
    elif mutation == "k_label": event["shared_k"] = 2
    elif mutation == "request_budget": row["effective_k"] = 2
    elif mutation == "acceptance": row["accepted"] = 2
    elif mutation == "emission": row["emitted"] = 0
    elif mutation == "missing_k0_sync": event["draft_forward_calls"] = 0
    elif mutation == "producer_rows": event["producer_rows"] = 2
    elif mutation == "duplicate_verifier": event["verifier_id"] = "v1"
    elif mutation == "tree_claim": record["tree_execution_verified"] = True
    with pytest.raises(ValueError):
        p.validate_run(manifest, raw)


@pytest.mark.parametrize("status", ["error", "oom", "timeout", "unsupported_shape"])
def test_failed_attempts_preserved_and_run_incomplete(status):
    manifest, raw = fixture()
    raw[5] = dict(raw[5], status=status, error="synthetic retained failure", retirements=[])
    result = p.validate_run(manifest, raw)
    assert not result["complete"]
    assert result["failures"][0]["status"] == status
    assert result["summary"][0]["successful_measured_cohorts"] == 49
    assert raw[5]["error"] == "synthetic retained failure"


def test_missing_cells_or_warmups_cannot_be_complete():
    manifest, raw = fixture()
    result = p.validate_run(manifest, [r for r in raw if r["cell_id"] != "k2"])
    assert not result["complete"] and len(result["missing_attempts"]) == 55
    result = p.validate_run(manifest, raw[1:])
    assert not result["complete"] and result["missing_attempts"] == [("k0", "warmup", 0)]


def test_reference_required_and_duplicates_rejected():
    manifest, raw = fixture()
    with pytest.raises(ValueError, match="Duplicate attempt"):
        p.validate_run(manifest, raw + raw[:1])
    manifest["cells"][1]["reference_cell_id"] = "absent"
    with pytest.raises(ValueError, match="K=0 reference"):
        p.validate_run(manifest, raw)


def test_raw_json_roundtrip_preserves_errors_and_rejects_nan():
    failure = dict(status="oom", error="GPU out of memory", duration_ms=None)
    assert json.loads(p.json_line(failure)) == failure
    with pytest.raises(ValueError):
        p.json_line(dict(duration_ms=float("nan")))


def test_pooled_rate_uses_elapsed_time_instead_of_average_rates():
    manifest, raw = fixture()
    for record in raw:
        if record["phase"] == "measured" and record["repeat"] % 2:
            for event in record["retirements"]:
                event["end_ns"] *= 3
            record["end_ns"] *= 3
    row = p.validate_run(manifest, raw)["summary"][0]
    assert row["pooled_emitted_tokens_per_ms"] == pytest.approx(.05)
    assert row["median_ms_per_retired_batch"] == 20


def test_heterogeneous_rows_preserve_shared_cost_and_useful_yield():
    manifest, raw = fixture()
    for cell in manifest["cells"]:
        cell["request_budgets"] = [cell["shared_k"], 0]
    for record in raw:
        for event in record["retirements"]:
            event["producer_rows"] = 2  # includes the trimmed K_i=0 row
            second = dict(event["requests"][0], request_id="other", effective_k=0,
                          query_tokens=1, drafted=0, accepted=0, emitted=1)
            event["requests"].append(second)
    report = p.validate_run(manifest, raw)
    assert report["complete"]
    row = report["summary"][-1]
    assert row["mean_emitted_tokens_per_batch"] == 6
    assert row["median_ms_per_retired_batch"] == 10  # shared cost charged once
    assert row["pooled_emitted_tokens_per_ms"] == .6
    # Physical worker compaction/reordering must not swap request budgets.
    for record in raw:
        record["retirements"][1]["requests"].reverse()
    reordered = p.validate_run(manifest, raw)
    assert reordered["complete"] and reordered["summary"] == report["summary"]


def test_t4_identity_is_separate_from_l4():
    manifest, _ = fixture()
    for key in ("observed_identity", "expected_identity"):
        manifest[key].update(gpu_name="Tesla T4", sm_architecture="sm_75",
                             dtype="half", quantization="awq", effective_quantization="awq")
    p.validate_identity(manifest["observed_identity"], manifest["expected_identity"])


def test_offline_cli_reads_without_rewriting_raw(tmp_path, monkeypatch, capsys):
    import sys
    manifest, raw = fixture()
    (tmp_path / "manifest.json").write_text(p.json_line(manifest), encoding="utf-8")
    raw_path = tmp_path / "raw.jsonl"
    original = "".join(map(p.json_line, raw)).encode()
    raw_path.write_bytes(original)
    monkeypatch.setattr(sys, "argv", ["scalar_cost_evidence", str(tmp_path)])
    assert p.main() == 0
    assert json.loads(capsys.readouterr().out)["consumer_profile"] is False
    assert raw_path.read_bytes() == original


def test_independent_cohorts_and_nonoverlapping_windows_required():
    manifest, raw = fixture()
    raw[1]["cohort_id"] = raw[0]["cohort_id"]
    with pytest.raises(ValueError, match="cohort ID"):
        p.validate_run(manifest, raw)
    manifest, raw = fixture()
    raw[1]["clock_domain"] = raw[0]["clock_domain"]
    with pytest.raises(ValueError, match="Overlapping"):
        p.validate_run(manifest, raw)


def test_reconstructed_context_fields_and_forecast_rejection():
    from experiments.scalar_context import VERSION
    manifest, raw = fixture()
    for record in raw:
        record.update(context_source="reconstructed", context_reconstruction_version=VERSION)
        for event in record["retirements"]:
            for row in event["requests"]:
                row.update(pre_execution_context=row["context_tokens"],
                           verification_query_length=row["query_tokens"],
                           raw_valid_sampled_count=row["emitted"],
                           corrected_post_execution_context=row["context_tokens"] + row["emitted"])
    assert p.validate_run(manifest, raw)["complete"]
    raw[5]["retirements"][0]["requests"][0]["corrected_post_execution_context"] += 1
    with pytest.raises(ValueError, match="reconstructed"):
        p.validate_run(manifest, raw)
    raw[5]["context_source"] = "CPU_OPTIMISTIC"
    with pytest.raises(ValueError, match="Forecast"):
        p.validate_run(manifest, raw)
