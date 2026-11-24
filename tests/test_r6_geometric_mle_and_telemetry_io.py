"""Tests for R6 Discounted-Count Geometric MLE alpha estimator and batched step telemetry I/O."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def _load_a10g_profile():
    from src.proposer.controller import EmpiricalCostProfile

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    matrix = {
        (int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): float(v)
        for k, v in data["cost_matrix_ms"].items()
    }
    return EmpiricalCostProfile(matrix=matrix), data["grid"]["candidate_budgets"]


@pytest.mark.parametrize("true_p", [0.40, 0.60, 0.80])
@pytest.mark.parametrize("k_fixed", [1, 2, 3, 4])
def test_geometric_mle_is_unbiased_across_all_k(true_p, k_fixed):
    """AC-01 & NFR-01: get_geometric_alpha() hội tụ về p thật (±0.02) ở mọi K in {1..4}."""
    from src.proposer.controller import AdaptiveDraftController

    rng = random.Random(2026 + k_fixed * 100 + int(true_p * 100))
    ctrl = AdaptiveDraftController(
        initial_alpha=0.8,
        mle_beta=0.90,
        mle_prior_weight=2.0,
        alpha_estimator="geometric_mle",
    )
    estimates = []
    for step in range(4000):
        acc = 0
        for _ in range(k_fixed):
            if rng.random() < true_p:
                acc += 1
            else:
                break
        ctrl.update(num_accepted=acc, num_drafted=k_fixed)
        if step >= 500:
            estimates.append(ctrl.get_geometric_alpha())
    mean_est = sum(estimates) / len(estimates)
    assert abs(mean_est - true_p) <= 0.02


def test_geometric_mle_avoids_step1_deadlock_but_falls_back_on_hard_prompt():
    """AC-02: Bước 1 trượt không gây khóa K=0; prompt khó liên tiếp trượt vẫn về K=0."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_profile()
    ctrl = AdaptiveDraftController(
        initial_alpha=0.80,
        candidate_budgets=budgets,
        cost_profile=profile,
        mle_beta=0.90,
        mle_prior_weight=2.0,
    )
    ctrl.update(num_accepted=0, num_drafted=4)
    assert ctrl.get_geometric_alpha() > 0.50
    d1 = ctrl.select_budget(batch_size=1)
    assert d1.k > 0
    assert d1.alpha_estimator == "geometric_mle"

    # Prompt khó (luôn trượt a=0): B=16 rơi về K=0 trong <= 3 bước; B=1 trong <= 15 bước
    ctrl_b16 = AdaptiveDraftController(initial_alpha=0.80, candidate_budgets=budgets, cost_profile=profile)
    b16_k0_step = None
    for s in range(1, 6):
        ctrl_b16.update(0, 2)
        if ctrl_b16.select_budget(batch_size=16).k == 0:
            b16_k0_step = s
            break
    assert b16_k0_step is not None and b16_k0_step <= 3

    ctrl_b1 = AdaptiveDraftController(initial_alpha=0.80, candidate_budgets=budgets, cost_profile=profile)
    b1_k0_step = None
    for s in range(1, 20):
        ctrl_b1.update(0, 1)
        if ctrl_b1.select_budget(batch_size=1).k == 0:
            b1_k0_step = s
            break
    assert b1_k0_step is not None and b1_k0_step <= 15


def test_geometric_mle_keeps_k3_k4_dominant_at_low_batch_and_k0_at_high_batch():
    """AC-03: Mô phỏng 200 request 60 token trên A10G thật (p=0.60)."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_profile()
    for batch_size in (1, 8, 32):
        rng = random.Random(123 + batch_size)
        counts = Counter()
        for _ in range(200):
            ctrl = AdaptiveDraftController(
                initial_alpha=0.80,
                candidate_budgets=budgets,
                cost_profile=profile,
            )
            gen = 0
            while gen < 60:
                k = ctrl.select_budget(batch_size=batch_size).k
                counts[k] += 1
                acc = 0
                for _ in range(k):
                    if rng.random() < 0.60:
                        acc += 1
                    else:
                        break
                gen += acc + 1
                if k > 0:
                    ctrl.update(acc, k)
        tot = sum(counts.values())
        if batch_size == 1:
            assert (counts[3] + counts[4]) / tot >= 0.80
            assert counts[0] / tot < 0.02
        elif batch_size == 8:
            assert counts[3] / tot >= 0.75
        else:
            assert counts[0] / tot >= 0.90


def test_mle_parameter_validation_clamping_and_reset():
    """Review Focus #1 & #2: Validate tham số bool/out-of-range, kẹp input xấu và reset sạch."""
    from src.proposer.controller import AdaptiveDraftController

    for bad_kwargs in (
        {"mle_beta": True},
        {"mle_beta": 0.0},
        {"mle_beta": 1.0},
        {"mle_beta": float("nan")},
        {"mle_prior_weight": True},
        {"mle_prior_weight": -0.5},
        {"alpha_estimator": "unknown"},
    ):
        with pytest.raises(ValueError):
            AdaptiveDraftController(**bad_kwargs)

    ctrl = AdaptiveDraftController(initial_alpha=0.75, mle_prior_weight=2.0, alpha_estimator="geometric_mle")
    ctrl.update(num_accepted=-5, num_drafted=3)
    assert 0.0 <= ctrl.get_geometric_alpha() <= 1.0
    ctrl.update(num_accepted=10, num_drafted=3)
    assert 0.0 <= ctrl.get_geometric_alpha() <= 1.0
    before_zero = ctrl.get_geometric_alpha()
    ctrl.update(num_accepted=0, num_drafted=0)
    assert ctrl.get_geometric_alpha() == pytest.approx(before_zero)

    ctrl.reset(initial_alpha=0.65)
    assert ctrl.get_geometric_alpha() == pytest.approx(0.65)
    assert ctrl.get_stats()["total_tested_positions"] == 0


def test_planner_batched_step_telemetry_opens_once_and_flushes_on_finish_and_close(tmp_path, monkeypatch):
    """AC-04, NFR-02 & Review Focus #4: Flush theo lô 64 dòng chỉ open() 1 lần và tự flush khi finish/close."""
    import builtins
    from src.server.budget_planner import AdaptiveBudgetPlanner

    step_path = tmp_path / "steps.jsonl"
    real_open = builtins.open
    open_count = 0

    def counting_open(file, *args, **kwargs):
        nonlocal open_count
        if str(file) == str(step_path):
            open_count += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    planner = AdaptiveBudgetPlanner.from_env(
        {
            "FLEXSDS_PROFILE_PATH": str(A10G_PROFILE_PATH),
            "ADAPTIVE_STEP_TELEMETRY_PATH": str(step_path),
            "ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY": "64",
        }
    )
    for _ in range(65):
        planner.plan_step(["r1"], batch_size=1, kv_usage=0.1, free_vram_mb=16000.0)
        planner.record_verification("r1", num_drafted=3, num_accepted=1)

    # 130 bản ghi >= 2 lô 64 -> đã ghi đúng 128 dòng ra đĩa, chỉ open() 1 lần
    assert open_count == 1
    assert len(real_open(step_path, "r", encoding="utf-8").read().splitlines()) == 128

    # finish_request tự flush nốt 2 bản ghi còn lại trong đệm
    planner.finish_request("r1")
    lines = [json.loads(line) for line in real_open(step_path, "r", encoding="utf-8").read().splitlines()]
    assert len(lines) == 130
    assert open_count == 1

    # Kiểm tra các trường telemetry MLE mới (FR-02)
    planned_rec = next(r for r in lines if r["record_type"] == "planned_decision")
    fb_rec = next(r for r in lines if r["record_type"] == "verification_feedback")
    assert planned_rec["alpha_estimator"] == "geometric_mle"
    assert planned_rec["geometric_alpha_mle"] is not None
    assert fb_rec["tested_draft_positions"] == 2
    assert fb_rec["geometric_alpha_mle"] is not None

    planner.close()
    assert open_count == 1


def test_planner_batched_io_error_does_not_crash_or_leak_buffer(tmp_path, caplog):
    """Review Focus #3: Đường dẫn lỗi không làm sập plan_step/flush/close và không tích tụ buffer."""
    from src.server.budget_planner import AdaptiveBudgetPlanner

    bad_path = tmp_path / "missing_dir" / "steps.jsonl"
    planner = AdaptiveBudgetPlanner(step_telemetry_path=str(bad_path), step_telemetry_flush_every=2)
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.1)
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.1)
    assert planner._step_buffer == []
    planner.close()
    assert "step telemetry" in caplog.text.lower()


def test_runner_and_modal_step_telemetry_batching_and_opt_out(monkeypatch):
    """AC-05 & Review Focus #5: Mặc định flush_every=64; opt-out tắt hẳn path và dọn biến rò."""
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.modal_runner import ModalServingRunner

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    base = {
        "target_model": data["models"]["target"],
        "draft_model": data["models"]["draft"],
        "draft_method": data["draft_method"],
        "target_revision": data["models"]["target_revision"],
        "draft_revision": data["models"]["draft_revision"],
        "profile_path": str(A10G_PROFILE_PATH),
        "enable_adaptive": True,
        "target_gpu": "A10G",
    }

    local_runner = LocalServingRunner()
    env_default = local_runner.setup_environment({"serving": dict(base)})
    assert "ADAPTIVE_STEP_TELEMETRY_PATH" in env_default
    assert env_default["ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY"] == "64"

    monkeypatch.setenv("ADAPTIVE_STEP_TELEMETRY_PATH", "/tmp/leaked_steps.jsonl")
    monkeypatch.setenv("ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY", "16")
    env_off = local_runner.setup_environment({"serving": dict(base, enable_step_telemetry=False)})
    assert "ADAPTIVE_STEP_TELEMETRY_PATH" not in env_off
    assert "ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY" not in env_off

    env_static = local_runner.setup_environment({"serving": {"enable_adaptive": False}})
    assert "ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY" not in env_static

    for bad_flush in (True, 0, -4):
        with pytest.raises(ValueError):
            local_runner.setup_environment({"serving": dict(base, step_telemetry_flush_every=bad_flush)})

    modal_env = ModalServingRunner().setup_environment(
        {"serving": dict(base, enable_step_telemetry=False, step_telemetry_flush_every=128)}
    )
    payload = json.loads(modal_env["FLEXSDS_SERVING_CONFIG"])
    assert payload["enable_step_telemetry"] is False
    assert payload["step_telemetry_flush_every"] == 128


def test_cli_no_step_telemetry_flag_forwards_opt_out(monkeypatch, capsys):
    """AC-05: Cờ --no-step-telemetry trên flexsds serve được truyền vào Modal serving config."""
    from types import SimpleNamespace
    import flexsds.cli.main as cli_main

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    fake_report = SimpleNamespace(
        draft_method=data["draft_method"],
        eagle_version="EAGLE-3.1",
        target_revision=data["models"]["target_revision"],
        draft_revision=data["models"]["draft_revision"],
        use_heterogeneous_vocab=False,
        identified_method="eagle3",
        oom_risk=False,
        warnings=[],
        recommendations=[],
        summary=lambda: "Compatibility: OK",
    )
    fake_preflight = SimpleNamespace(report=fake_report, launch_reasons=lambda allow_oom: [])
    monkeypatch.setattr(cli_main, "evaluate_serving_preflight", lambda *a, **kw: fake_preflight)
    monkeypatch.setattr(
        cli_main,
        "load_profile_for_hardware",
        lambda *a, **kw: (object(), A10G_PROFILE_PATH.name),
    )
    monkeypatch.setattr(cli_main, "resolve_profile_path", lambda name: A10G_PROFILE_PATH)

    rc = cli_main.main(
        [
            "serve",
            "--backend",
            "modal",
            "--gpu",
            "A10G",
            "--target",
            data["models"]["target"],
            "--draft",
            data["models"]["draft"],
            "--no-step-telemetry",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"enable_step_telemetry": false' in out


def test_adversarial_edge_cases_adapter_telemetry_env_validation_and_modal_launch():
    """Adversarial review: adapter export MLE stats, env flush_every errors, Modal launch validation."""
    from src.proposer.adapter import AdaptiveProposerAdapter
    from src.server.budget_planner import AdaptiveBudgetPlanner
    from flexsds.server.modal_runner import ModalServingRunner

    profile, budgets = _load_a10g_profile()
    adapter = AdaptiveProposerAdapter(
        controller_kwargs={"cost_profile": profile, "candidate_budgets": budgets}
    )
    adapter.propose("r1", draft_generator=lambda k: [1] * k, batch_size=1)
    adapter.on_verification_completed("r1", num_accepted=2, num_drafted=4)
    summary = adapter.end_request("r1")
    assert summary["alpha_estimator"] == "geometric_mle"
    assert summary["total_tested_positions"] == 3
    assert summary["geometric_alpha_mle"] == pytest.approx((0.90 * 2.0 * 0.8 + 2) / (0.90 * 2.0 + 3))

    for bad_env in ("0", "-3", "not-an-int"):
        with pytest.raises(ValueError):
            AdaptiveBudgetPlanner.from_env({"ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY": bad_env})

    # ModalServingRunner.launch rejects invalid flush_every early, yet accepts enable_step_telemetry=False
    # on a minimal non-adaptive dry-run config without falsely demanding a profile artifact.
    modal_runner = ModalServingRunner()
    with pytest.raises(ValueError):
        modal_runner.launch(
            {"serving": {"target_gpu": "A10G", "enable_adaptive": False, "step_telemetry_flush_every": 0}},
            dry_run=True,
        )
    assert (
        modal_runner.launch(
            {"serving": {"target_gpu": "A10G", "enable_step_telemetry": False}},
            dry_run=True,
        )
        is None
    )


