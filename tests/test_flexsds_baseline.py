"""Unit test suite for FlexSDS Baseline CLI and Core modules."""
import pytest
try:
    from flexsds import __version__
    from flexsds.cli.banner import render_banner
    from flexsds.core.hardware import detect_hardware, HardwareInfo
    from flexsds.core.compatibility import check_compatibility, estimate_model_vram_mb
    from flexsds.core.cost_model import FlexCostModel, load_profile_for_hardware
    from flexsds.core.controller import TreeAdaptiveController
except ModuleNotFoundError:
    from src.flexsds import __version__
    from src.flexsds.cli.banner import render_banner
    from src.flexsds.core.hardware import detect_hardware, HardwareInfo
    from src.flexsds.core.compatibility import check_compatibility, estimate_model_vram_mb
    from src.flexsds.core.cost_model import FlexCostModel, load_profile_for_hardware
    from src.flexsds.core.controller import TreeAdaptiveController


def test_flexsds_version():
    """Verifies package version format."""
    assert __version__ == "0.2.0-alpha"


def test_banner_rendering():
    """Verifies PTIT logo and FlexSDS banner rendering without errors."""
    mock_hw = HardwareInfo(
        device_name="NVIDIA L4",
        sm_architecture="sm_89",
        total_vram_mb=23034.0,
        free_vram_mb=22450.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    banner_text = render_banner(mock_hw)
    assert "FlexSDS" in banner_text
    assert "NVIDIA L4" in banner_text
    assert "sm_89" in banner_text
    assert "PTIT" in banner_text


def test_hardware_detector():
    """Verifies hardware detector returns valid HardwareInfo object."""
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.total_vram_mb > 0
    assert hw.free_vram_mb > 0
    assert hw.device_name is not None
    assert hw.sm_architecture is not None


def test_compatibility_guard_eagle_pairing():
    """Verifies EAGLE speculator pairing with Qwen target."""
    report = check_compatibility(
        target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        free_vram_mb=23000.0,
    )
    assert report.draft_method == "eagle3"
    assert report.use_heterogeneous_vocab is True
    assert report.oom_risk is False
    assert report.is_compatible is True


def test_compatibility_guard_oom_detection():
    """Verifies detection of VRAM pressure when free memory is insufficient."""
    report = check_compatibility(
        target_model="Qwen/Qwen2.5-7B-Instruct",  # FP16 ~15GB
        draft_model="Qwen/Qwen2.5-0.5B-Instruct",
        free_vram_mb=8000.0,  # Only 8GB free -> OOM risk
        target_quant="fp16",
    )
    assert report.oom_risk is True
    assert len(report.warnings) > 0
    assert "VRAM shortfall" in report.warnings[0]


def test_cost_profile_loader_sm89():
    """Verifies auto-loading of L4 profile for sm_89."""
    mock_hw = HardwareInfo(
        device_name="NVIDIA L4",
        sm_architecture="sm_89",
        total_vram_mb=23034.0,
        free_vram_mb=22000.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    model, filename = load_profile_for_hardware(mock_hw)
    assert "profile_sm89_l4.json" in filename
    assert len(model.forward_target) > 0

    # Test alpha* calculation
    alpha_star = model.calculate_alpha_star(k=3, batch_size=1)
    assert 0.3 < alpha_star < 0.6


def test_tree_adaptive_controller_budget_selection():
    """Verifies TreeAdaptiveController dynamically chooses optimal k based on alpha and load."""
    mock_hw = HardwareInfo(
        device_name="NVIDIA L4",
        sm_architecture="sm_89",
        total_vram_mb=23034.0,
        free_vram_mb=22000.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    model, _ = load_profile_for_hardware(mock_hw)
    controller = TreeAdaptiveController(cost_model=model, initial_alpha=0.85)

    # High acceptance at B=1 should pick k >= 2
    decision_high = controller.select_budget(batch_size=1, free_vram_mb=20000.0)
    assert decision_high.k >= 2

    # Low acceptance should fallback to k=0
    controller.alpha_ema = 0.20
    controller.step_count = 5
    decision_low = controller.select_budget(batch_size=1, free_vram_mb=20000.0)
    assert decision_low.k == 0
    assert "fallback" in decision_low.reason.lower()


def test_flexsds_core_tree_adaptive_controller_vendored():
    from flexsds.core.controller import TreeAdaptiveController, TreeAdaptiveDecision
    ctrl = TreeAdaptiveController(min_free_vram_mb=1500.0)
    decision = ctrl.select_budget(batch_size=1, free_vram_mb=8000.0, kv_cache_usage_pct=20.0)
    assert isinstance(decision, TreeAdaptiveDecision)
    assert hasattr(decision, "topology_name")
    assert hasattr(decision, "alpha_effective")
    assert hasattr(decision, "speedup")
    assert decision.status in ("ok", "memory_guard", "concurrency_throttled", "fallback_gate", "probing", "zero_crash_fallback")


def test_cost_profile_resolution_missing_hardware_fallback():
    """Verifies unknown/unprofiled hardware falls back safely to conservative analytical model."""
    unknown_hw = HardwareInfo(
        device_name="NVIDIA RTX 4090 Custom",
        sm_architecture="sm_99",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.6",
        is_cuda_available=True,
    )
    model, filename = load_profile_for_hardware(unknown_hw)
    assert filename == "conservative_analytical_fallback"
    assert model.provenance == "conservative_analytical_fallback"
    assert model.metadata.get("fallback_mode") is True
    assert model.metadata.get("k_max_safe") == 3


def test_cost_profile_resolution_custom_path_override(tmp_path):
    """Verifies explicit custom profile path takes highest precedence."""
    import json
    custom_file = tmp_path / "custom_rtx4090.json"
    custom_data = {
        "forward_target": {"1": 15.0, "2": 16.0},
        "forward_draft": {"1": 2.5, "2": 2.7},
    }
    custom_file.write_text(json.dumps(custom_data), encoding="utf-8")

    model, filename = load_profile_for_hardware(custom_profile_path=custom_file)
    assert filename == "custom_rtx4090.json"
    assert model.forward_target[1] == 15.0
    assert model.forward_draft[1] == 2.5


def test_save_hardware_profile_and_user_dir_resolution(tmp_path, monkeypatch):
    """Verifies newly calibrated profile is saved and picked up from user profile store."""
    from flexsds.core.cost_model import save_hardware_profile, USER_PROFILES_DIR
    import flexsds.core.cost_model as cm

    # Mock user profiles dir to temporary path
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)

    profile_payload = {
        "forward_target": {"1": 12.0},
        "forward_draft": {"1": 2.0},
        "metadata": {"calibrated": True},
    }
    saved_path = save_hardware_profile(
        profile_data=profile_payload,
        sm_architecture="sm_90",
        device_name="NVIDIA H100",
        target_dir=tmp_path,
    )
    assert saved_path.exists()
    assert "sm_90" in saved_path.name

    h100_hw = HardwareInfo(
        device_name="NVIDIA H100",
        sm_architecture="sm_90",
        total_vram_mb=80000.0,
        free_vram_mb=75000.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    model, filename = load_profile_for_hardware(h100_hw)
    assert "sm_90" in filename
    assert model.forward_target[1] == 12.0


