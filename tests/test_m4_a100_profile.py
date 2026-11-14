"""SYNTHETIC profile fixtures exercise schema only; no A100 measurement."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.m4_a100_profile import BATCHES, KS, hardware_gate, validate_profile
from experiments.m4_treatments import TARGET, TARGET_REVISION, SPECULATOR, SPECULATOR_REVISION

SHA = "a" * 40


def synthetic_profile():
    return dict(metadata=dict(gpu="NVIDIA A100-SXM4-40GB", gpu_compute_capability=[8, 0],
        cuda_available=True, gpu_memory_total_mb=40000, sm_architecture="sm_80",
        torch_version="2.13.0+cu130", vllm_version="0.30.0", cuda_version="13.0",
        source_commit=SHA, measurement_status="measured", scalar_only=True, dtype="bfloat16", quantization="awq_marlin",
        driver_version="SYNTHETIC", python_version="SYNTHETIC", timestamp_utc="SYNTHETIC",
        profiler_command="SYNTHETIC TEST FIXTURE", raw_sha256="b" * 64),
        models=dict(target=TARGET, target_revision=TARGET_REVISION, draft=SPECULATOR, draft_revision=SPECULATOR_REVISION),
        cost_matrix_ms={f"k{k}_b{b}": 1.0 + k for k in KS for b in BATCHES})


def test_profile_serialization_and_exact_loader_lookups():
    p = json.loads(json.dumps(synthetic_profile()))
    model = validate_profile(p, SHA)
    assert model.matrix[(4, 32)] == 5
    assert model.get_step_cost(4, 32) == 5
    with pytest.raises(ValueError, match="scalar"):
        model.get_step_cost("T8", 32)


@pytest.mark.parametrize("change", [{"gpu": "NVIDIA L4"}, {"gpu_compute_capability": [8, 9]},
    {"cuda_available": False}, {"gpu_memory_total_mb": 0}, {"source_commit": "b" * 40},
    {"measurement_status": "analytical"}, {"vllm_version": "latest"}, {"driver_version": None}])
def test_profile_malformed_or_other_hardware_rejected(change):
    p = synthetic_profile(); p["metadata"].update(change)
    with pytest.raises(ValueError): validate_profile(p, SHA)


@pytest.mark.parametrize("change", ["missing", "negative", "nan", "tree"])
def test_profile_grid_rejected(change):
    p = synthetic_profile()
    if change == "missing": p["cost_matrix_ms"].pop("k0_b1")
    if change == "negative": p["cost_matrix_ms"]["k1_b1"] = -1
    if change == "nan": p["cost_matrix_ms"]["k1_b1"] = float("nan")
    if change == "tree": p["cost_matrix_ms"]["topo_T32_b1"] = 1
    with pytest.raises(ValueError): validate_profile(p, SHA)


def test_a100_mapping_cannot_silently_use_l4(tmp_path, monkeypatch):
    from src.flexsds.core import cost_model
    from experiments.m4_cost_profiler import classify_hardware
    assert classify_hardware("NVIDIA A100-SXM4-40GB", [8, 0], "A100")["canonical_hardware"] is True
    # Name tokenization must support real A100 SKU names, not just the bare word.
    monkeypatch.setattr(cost_model, "PROFILES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="profile_sm80"):
        cost_model.load_profile_for_hardware(SimpleNamespace(sm_architecture="sm_80"))
    (tmp_path / "profile_sm80.json").write_text('{}')
    with pytest.raises(ValueError, match="Malformed"):
        cost_model.load_profile_for_hardware(SimpleNamespace(sm_architecture="sm_80"))
    (tmp_path / "profile_sm80.json").write_text(json.dumps(synthetic_profile()))
    model, name = cost_model.load_profile_for_hardware(SimpleNamespace(sm_architecture="sm_80"))
    assert name == "profile_sm80.json" and model.get_step_cost(4, 32) == 5


def test_orchestrator_hardware_probe_is_process_isolated(monkeypatch):
    import experiments.m4_a100_profile as module
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        return '{"gpu":"SYNTHETIC"}'
    monkeypatch.setattr(module.subprocess, "check_output", child)
    assert module.inspect_environment()["gpu"] == "SYNTHETIC"
    assert calls[0][1] == "-c" and "observe_hardware" in calls[0][2]
