"""Tests for centralized Modal GPU catalog, pricing footers, and strict hardware matching."""
from __future__ import annotations

import json
import pytest

from flexsds.core import calibration as cal
from flexsds.core import cost_model as cm
from flexsds.core.compatibility import evaluate_serving_preflight
from flexsds.core.hardware import (
    DEFAULT_MODAL_GPU,
    MODAL_GPU_CHOICES,
    MODAL_GPU_NAMES,
    MODAL_GPU_SPECS,
    HardwareInfo,
    build_remote_hardware_info,
    get_modal_gpu_spec,
    matches_gpu_label,
    modal_gpu_choice_label,
    modal_gpu_footer,
    modal_gpu_resource,
    remote_gpu_budget_mb,
)


def _fake_measured_groups(device: str, sm_architecture: str, batches=None):
    batches = batches or [1]
    groups = []
    for k in cal.KS:
        records = [
            dict(
                k=k,
                batch=b,
                context=cal.CONTEXT,
                repeat=r,
                warmup=r < 2,
                samples=dict(target=10.0 + k + b, draft=2.0 * max(1, k)),
                shapes=[dict(batch=b, query=[k + 1] * b, context=[cal.CONTEXT] * b)],
            )
            for b in batches
            for r in range(7)
        ]
        groups.append(
            dict(
                device=device,
                sm_architecture=sm_architecture,
                records=records,
                vllm_version="0.30.0",
                target_revision="a" * 40,
                draft_revision="b" * 40,
            )
        )
    return groups


def test_modal_gpu_catalog_tiers_vram_pricing_and_modal_resources():
    assert MODAL_GPU_NAMES == ("T4", "A10", "L40S", "A100")
    assert DEFAULT_MODAL_GPU == "A10"
    assert len(MODAL_GPU_SPECS) == 4

    by_name = {spec.name: spec for spec in MODAL_GPU_SPECS}
    assert by_name["T4"].tier == "Low"
    assert by_name["T4"].vram_gb == 16
    assert by_name["T4"].sm_architecture == "sm_75"
    assert by_name["T4"].modal_gpu == "T4"
    assert by_name["T4"].price_per_sec == pytest.approx(0.000164)
    assert by_name["T4"].price_per_hour == pytest.approx(0.59)

    assert by_name["A10"].tier == "Medium"
    assert by_name["A10"].vram_gb == 24
    assert by_name["A10"].sm_architecture == "sm_86"
    assert by_name["A10"].modal_gpu == "A10"
    assert by_name["A10"].price_per_sec == pytest.approx(0.000306)
    assert by_name["A10"].price_per_hour == pytest.approx(1.10)

    assert by_name["L40S"].tier == "High"
    assert by_name["L40S"].vram_gb == 48
    assert by_name["L40S"].sm_architecture == "sm_89"
    assert by_name["L40S"].modal_gpu == "L40S"
    assert by_name["L40S"].price_per_sec == pytest.approx(0.000542)
    assert by_name["L40S"].price_per_hour == pytest.approx(1.95)

    assert by_name["A100"].tier == "High"
    assert by_name["A100"].vram_gb == 80
    assert by_name["A100"].sm_architecture == "sm_80"
    assert by_name["A100"].modal_gpu == "A100-80GB"
    assert by_name["A100"].price_per_sec == pytest.approx(0.000694)
    assert by_name["A100"].price_per_hour == pytest.approx(2.50)

    assert modal_gpu_resource("A100") == "A100-80GB"
    assert modal_gpu_resource("L40S") == "L40S"
    assert modal_gpu_resource("A10") == "A10"
    assert modal_gpu_resource("T4") == "T4"


def test_modal_gpu_labels_and_footers_show_tier_vram_and_pricing():
    assert MODAL_GPU_CHOICES == [
        "T4 [Low]",
        "A10 [Medium] (Recommended)",
        "L40S [High]",
        "A100 [High]",
    ]
    for spec in MODAL_GPU_SPECS:
        label = modal_gpu_choice_label(spec)
        assert "A100-80GB" not in label
        assert get_modal_gpu_spec(label) == spec
        footer = modal_gpu_footer(spec)
        assert f"[{spec.tier}]" in footer
        assert f"{spec.vram_gb} GB VRAM" in footer
        assert spec.sm_architecture in footer
        assert f"${spec.price_per_sec:.6f}/s" in footer
        assert f"~${spec.price_per_hour:.2f}/h" in footer


@pytest.mark.parametrize("bad_gpu", ["L4", "A10G", "A100-40GB", "H100", ""])
def test_get_modal_gpu_spec_rejects_removed_or_unknown_gpus(bad_gpu):
    with pytest.raises(ValueError, match="Unsupported Modal GPU"):
        get_modal_gpu_spec(bad_gpu)


@pytest.mark.parametrize(
    "label,device,expected",
    [
        ("A10", "NVIDIA A10", True),
        ("A10G", "NVIDIA A10", True),
        ("A10", "NVIDIA A10G", True),
        ("A10", "NVIDIA RTX 3090", False),
        ("L40S", "NVIDIA L40S", True),
        ("L40S", "NVIDIA L4", False),
        ("L4", "NVIDIA L40S", False),
        ("A100-80GB", "NVIDIA A100-SXM4-80GB", True),
        ("A100-80GB", "NVIDIA A100 80GB PCIe", True),
        ("A100-80GB", "NVIDIA A100-SXM4-40GB", False),
        ("A100-80GB", "NVIDIA A100-PCIE-40GB", False),
        ("A100-40GB", "NVIDIA A100-SXM4-40GB", True),
        ("A100-40GB", "NVIDIA A100-SXM4-80GB", False),
        ("A100", "NVIDIA A100-SXM4-80GB", True),
        ("A100", "NVIDIA A100-SXM4-40GB", True),
    ],
)
def test_matches_gpu_label_enforces_exact_sku_boundaries(label, device, expected):
    assert matches_gpu_label(label, device) is expected


def test_modal_a100_rejects_40gb_profile_on_host_and_accepts_80gb(tmp_path, monkeypatch):
    """Review Focus #1: Modal A100 (80 GB) must reject 40 GB sm_80 profiles before launch."""
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path / "user")
    monkeypatch.setattr(cm, "PROFILES_DIR", tmp_path / "bundled")

    a100_hw = build_remote_hardware_info("modal", "A100")
    assert a100_hw.device_name == "Remote A100"
    assert a100_hw.sm_architecture == "sm_80"
    assert a100_hw.gpu_match_label == "A100-80GB"
    assert a100_hw.free_vram_mb == remote_gpu_budget_mb("modal", "A100") == 78000.0

    # No profile present in strict mode must return (None, ""), never crash with FileNotFoundError.
    assert cm.load_profile_for_hardware(a100_hw, strict=True, target_model="t", draft_model="d") == (None, "")

    # A 40 GB measured profile must NOT match Modal A100 (80 GB).
    cfg_40 = dict(
        target_model="t",
        draft_model="d",
        draft_method="eagle3",
        sm_architecture="sm_80",
        measurement_mode="fast",
    )
    prof_40 = cal.build_profile(cfg_40, _fake_measured_groups("NVIDIA A100-SXM4-40GB", "sm_80"), [1])
    cm.save_hardware_profile(prof_40, "sm_80", "a100_40gb", target_dir=tmp_path / "user")
    assert cm.load_profile_for_hardware(a100_hw, strict=True, target_model="t", draft_model="d") == (None, "")

    # An 80 GB measured profile MUST match Modal A100 (80 GB).
    cfg_80 = dict(
        target_model="t",
        draft_model="d",
        draft_method="eagle3",
        sm_architecture="sm_80",
        target_gpu="A100",
        measurement_mode="fast",
    )
    prof_80 = cal.build_profile(cfg_80, _fake_measured_groups("NVIDIA A100-SXM4-80GB", "sm_80"), [1])
    saved_80 = cm.save_hardware_profile(prof_80, "sm_80", "remote_a100", target_dir=tmp_path / "user")
    model, name = cm.load_profile_for_hardware(a100_hw, strict=True, target_model="t", draft_model="d")
    assert model is not None
    assert name == saved_80.name

    # Calibration build_profile with target_gpu="A100" must reject 40 GB hardware evidence.
    with pytest.raises(ValueError, match="GPU model mismatch"):
        cal.build_profile(cfg_80, _fake_measured_groups("NVIDIA A100-SXM4-40GB", "sm_80"), [1])


def test_modal_a100_strict_mode_rejects_legacy_40gb_scalar_profile(tmp_path, monkeypatch):
    """Strict host resolution must reject non-component-v2 scalar sm_80 profiles (e.g. 40 GB M4)."""
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path / "user")
    monkeypatch.setattr(cm, "PROFILES_DIR", tmp_path / "bundled")
    (tmp_path / "user").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bundled").mkdir(parents=True, exist_ok=True)

    legacy_scalar_40gb = {
        "metadata": {
            "gpu": "NVIDIA A100-SXM4-40GB",
            "sm_architecture": "sm_80",
            "scalar_only": True,
            "measurement_status": "measured",
        },
        "cost_matrix_ms": {
            f"k{k}_b{b}": 1.0 + k
            for k in range(5)
            for b in (1, 2, 4, 8, 16, 32)
        },
    }
    (tmp_path / "user" / "profile_sm_80.json").write_text(json.dumps(legacy_scalar_40gb), encoding="utf-8")
    (tmp_path / "bundled" / "profile_sm80.json").write_text(json.dumps(legacy_scalar_40gb), encoding="utf-8")

    a100_hw = build_remote_hardware_info("modal", "A100")
    assert cm.load_profile_for_hardware(a100_hw, strict=True, target_model="t", draft_model="d") == (None, "")


def test_modal_l40s_never_matches_l4_profile_on_shared_sm89(tmp_path, monkeypatch):
    """Review Focus #2: L40S shares sm_89 with L4 and must never load an L4 profile."""
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path / "user")
    l40s_hw = build_remote_hardware_info("modal", "L40S")
    assert l40s_hw.device_name == "Remote L40S"
    assert l40s_hw.sm_architecture == "sm_89"
    assert l40s_hw.gpu_match_label == "L40S"

    cfg_l4 = dict(
        target_model="t",
        draft_model="d",
        draft_method="eagle3",
        sm_architecture="sm_89",
        measurement_mode="fast",
    )
    prof_l4 = cal.build_profile(cfg_l4, _fake_measured_groups("NVIDIA L4", "sm_89"), [1])
    cm.save_hardware_profile(prof_l4, "sm_89", "remote_l4", target_dir=tmp_path / "user")

    assert cm.load_profile_for_hardware(l40s_hw, strict=True, target_model="t", draft_model="d") == (None, "")


def test_bundled_a10_profile_loads_cleanly_for_modal_a10(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)
    a10_hw = build_remote_hardware_info("modal", "A10")
    model, name = cm.load_profile_for_hardware(
        a10_hw,
        strict=True,
        target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
    )
    assert model is not None
    assert name == "profile_sm_86_remote_a10g_fcda5486b12d.json"


def test_preflight_uses_remote_budgets_for_all_modal_gpus(monkeypatch):
    captured = []

    def fake_check(target_model, draft_model, free_vram_mb, **kwargs):
        captured.append(free_vram_mb)
        from flexsds.core.compatibility import CompatibilityReport

        return CompatibilityReport(
            target_model,
            draft_model,
            "eagle3",
            False,
            5000.0,
            400.0,
            2000.0,
            8000.0,
            free_vram_mb,
            False,
            True,
            vocabulary_status="Shared",
        )

    monkeypatch.setattr("flexsds.core.compatibility.check_compatibility", fake_check)
    hw = HardwareInfo("CPU", "cpu", 0.0, 0.0, "N/A", False)
    for gpu, expected_mb in [("T4", 15000.0), ("A10", 22000.0), ("L40S", 46000.0), ("A100", 78000.0)]:
        result = evaluate_serving_preflight("target", "draft", hw, backend="modal", target_gpu=gpu)
        assert result.download.free_vram_mb == expected_mb
    assert captured == [15000.0, 22000.0, 46000.0, 78000.0]


def test_modal_app_maps_a100_to_a100_80gb_and_rejects_removed_gpus(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    from flexsds.server import modal_image

    source = Path(modal_image.__file__).with_name("modal_app.py")

    def run_app_with_gpu(gpu_name):
        monkeypatch.setenv(
            "FLEXSDS_SERVING_CONFIG",
            json.dumps({"target_gpu": gpu_name, "enable_adaptive": False}),
        )
        monkeypatch.delenv("FLEXSDS_SERVING_CONFIG_B64", raising=False)
        image = Mock()
        image.env.return_value = image
        image.add_local_dir.return_value = image
        image.add_local_file.return_value = image
        app = Mock()
        app.function.return_value = lambda fn: fn
        monkeypatch.setitem(
            sys.modules,
            "modal",
            SimpleNamespace(
                App=Mock(return_value=app),
                is_local=Mock(return_value=False),
                Volume=SimpleNamespace(from_name=Mock()),
                web_server=lambda *a, **kw: lambda fn: fn,
            ),
        )
        monkeypatch.setattr(modal_image, "serving_image", lambda: image)
        runpy.run_path(str(source))
        return app.function.call_args.kwargs["gpu"]

    assert run_app_with_gpu("T4") == "T4"
    assert run_app_with_gpu("A10") == "A10"
    assert run_app_with_gpu("L40S") == "L40S"
    assert run_app_with_gpu("A100") == "A100-80GB"

    for removed in ("L4", "A10G", "A100-40GB"):
        with pytest.raises(ValueError, match="Unsupported Modal GPU"):
            run_app_with_gpu(removed)


def test_modal_calibration_accepts_catalog_gpus_and_rejects_removed(monkeypatch):
    from pathlib import Path
    from flexsds.server import modal_calibration

    monkeypatch.setattr("flexsds.server.modal_auth.ensure_modal", lambda: None)
    monkeypatch.setattr("flexsds.server.modal_auth.modal_command", lambda: ["modal"])

    def fake_run(command, *, env, **kwargs):
        Path(env["FLEXSDS_CALIBRATION_OUTPUT"]).write_text('[{"ok": true}]')

    monkeypatch.setattr(modal_calibration, "run_logged_process", fake_run)
    for gpu in ("T4", "A10", "L40S", "A100"):
        assert modal_calibration.calibrate({"target_gpu": gpu}, [1]) == [{"ok": True}]

    for bad in ("L4", "A10G", "A100-40GB"):
        with pytest.raises(ValueError, match="Unsupported Modal calibration GPU"):
            modal_calibration.calibrate({"target_gpu": bad}, [1])


def test_cli_gpu_flags_and_wizard_footers_match_modal_catalog():
    from flexsds.cli.main import parse_args
    from flexsds.cli import wizard

    assert parse_args(["serve"]).gpu == "A10"
    assert parse_args(["profile"]).gpu == "A10"
    for gpu in ("T4", "A10", "L40S", "A100"):
        assert parse_args(["serve", "--gpu", gpu]).gpu == gpu
        assert parse_args(["profile", "--gpu", gpu]).gpu == gpu

    for rejected in ("L4", "A10G", "A100-40GB"):
        with pytest.raises(SystemExit) as exc:
            parse_args(["serve", "--gpu", rejected])
        assert exc.value.code == 2
        with pytest.raises(SystemExit) as exc:
            parse_args(["profile", "--gpu", rejected])
        assert exc.value.code == 2

    assert wizard.MODAL_GPU_CHOICES == MODAL_GPU_CHOICES
    for spec in MODAL_GPU_SPECS:
        choice = modal_gpu_choice_label(spec)
        desc = wizard.choice_description("Modal GPU:", choice)
        assert f"[{spec.tier}]" in desc
        assert f"{spec.vram_gb} GB VRAM" in desc
        assert f"${spec.price_per_sec:.6f}/s" in desc

    colab_desc = wizard.choice_description("Serving platform:", wizard.PLATFORM_CHOICES[1])
    assert "notebook" in colab_desc.lower()


def test_modal_runner_forwards_eagle_version():
    from flexsds.server.modal_runner import ModalServingRunner

    runner = ModalServingRunner()
    cfg = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "draft_method": "eagle3",
            "eagle_version": "3.1",
            "target_gpu": "A10",
            "enable_adaptive": False,
        }
    }
    forwarded = json.loads(runner.setup_environment(cfg)["FLEXSDS_SERVING_CONFIG"])
    assert forwarded["eagle_version"] == "3.1"
    assert forwarded["target_gpu"] == "A10"


def test_colab_remote_budget_handles_default_and_colab_gpus():
    assert remote_gpu_budget_mb("colab", "A10") == 22000.0
    assert remote_gpu_budget_mb("colab", "T4") == 15000.0
    assert remote_gpu_budget_mb("colab", "L4") == 23000.0
    assert remote_gpu_budget_mb("colab", "A100") == 78000.0
