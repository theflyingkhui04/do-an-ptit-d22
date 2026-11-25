"""Offline tests for matched execution contract across Vanilla, Static EAGLE, and Adaptive."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexsds.server.local_runner import LocalServingRunner

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def _base_pair_cfg(**overrides):
    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    cfg = {
        "target_model": data["models"]["target"],
        "draft_model": data["models"]["draft"],
        "draft_method": data["draft_method"],
        "target_revision": data["models"]["target_revision"],
        "draft_revision": data["models"]["draft_revision"],
        "port": 8000,
        "max_model_len": 4096,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.85,
    }
    cfg.update(overrides)
    return {"serving": cfg}


def test_adaptive_on_and_static_matched_share_identical_v1_eager_contract():
    """AC-01: Adaptive ON và Static matched dùng chung 100% cờ V1 eager và env runner."""
    runner = LocalServingRunner()
    on_cfg = _base_pair_cfg(enable_adaptive=True, profile_path=str(A10G_PROFILE_PATH))
    static_matched_cfg = _base_pair_cfg(
        enable_adaptive=False,
        match_adaptive_runtime=True,
    )

    on_cmd = runner.build_command(on_cfg)
    st_cmd = runner.build_command(static_matched_cfg)
    on_env = runner.setup_environment(on_cfg)
    st_env = runner.setup_environment(static_matched_cfg)

    for flag in ("--enforce-eager", "--compilation-config", "--async-scheduling"):
        assert flag in on_cmd
        assert flag in st_cmd
    assert st_cmd.count("--async-scheduling") == 1
    assert on_cmd.count("--async-scheduling") == 1
    assert st_cmd[st_cmd.index("--compilation-config") + 1] == on_cmd[on_cmd.index("--compilation-config") + 1]
    assert on_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert st_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"

    assert "--scheduler-cls" not in st_cmd
    assert "--distributed-executor-backend" not in st_cmd
    st_spec = json.loads(st_cmd[st_cmd.index("--speculative-config") + 1])
    assert "num_speculative_tokens_per_batch_size" not in st_spec
    assert st_spec["num_speculative_tokens"] == 4
    assert "FLEXSDS_PROFILE_PATH" not in st_env
    assert "ADAPTIVE_STRICT" not in st_env
    assert runner.describe_execution_contract(on_cfg) == "adaptive-eager-v1"
    assert runner.describe_execution_contract(static_matched_cfg) == "static-matched-eager-v1"


def test_vanilla_matched_omits_speculative_and_supports_no_prefix_caching():
    """AC-02 & Review Focus #3: Vanilla matched bỏ hẳn speculative và hỗ trợ cache-off."""
    runner = LocalServingRunner()
    cfg = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "target_revision": "b25037543e9394b818fdfca67ab2a00ecc7dd641",
            "enable_adaptive": False,
            "disable_speculative": True,
            "match_adaptive_runtime": True,
            "disable_prefix_caching": True,
            "draft_method": "none",
            "port": 8000,
        }
    }
    cmd = runner.build_command(cfg)
    env = runner.setup_environment(cfg)

    assert "--enforce-eager" in cmd
    assert "--compilation-config" in cmd
    assert "--async-scheduling" in cmd
    assert "--no-enable-prefix-caching" in cmd
    assert "--speculative-config" not in cmd
    assert "--per-request-spec-decode-metrics" not in cmd
    assert "--scheduler-cls" not in cmd
    assert "--distributed-executor-backend" not in cmd
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert runner.describe_execution_contract(cfg) == "vanilla-matched-eager-v1+no-prefix-cache"


def test_setup_environment_strips_leaked_shell_vars_for_off_modes(monkeypatch):
    """AC-03 & Review Focus #1: Dọn sạch biến rò từ shell cha ở cả default OFF và matched OFF."""
    runner = LocalServingRunner()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("ADAPTIVE_STRICT", "1")
    monkeypatch.setenv("FLEXSDS_PROFILE_PATH", "/tmp/leaked.json")
    monkeypatch.setenv("ADAPTIVE_STEP_TELEMETRY_PATH", "/tmp/steps.jsonl")
    monkeypatch.setenv("ADAPTIVE_STARTUP_TELEMETRY_PATH", "/tmp/lineage.jsonl")
    monkeypatch.setenv("ADAPTIVE_CONTROLLER_KWARGS", '{"initial_alpha": 0.5}')

    default_env = runner.setup_environment({"serving": {"enable_adaptive": False}})
    for var in (
        "VLLM_USE_V2_MODEL_RUNNER",
        "ADAPTIVE_STRICT",
        "FLEXSDS_PROFILE_PATH",
        "ADAPTIVE_STEP_TELEMETRY_PATH",
        "ADAPTIVE_STARTUP_TELEMETRY_PATH",
        "ADAPTIVE_CONTROLLER_KWARGS",
    ):
        assert var not in default_env

    matched_env = runner.setup_environment(
        {"serving": {"enable_adaptive": False, "match_adaptive_runtime": True}}
    )
    assert matched_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    for var in (
        "ADAPTIVE_STRICT",
        "FLEXSDS_PROFILE_PATH",
        "ADAPTIVE_STEP_TELEMETRY_PATH",
        "ADAPTIVE_STARTUP_TELEMETRY_PATH",
        "ADAPTIVE_CONTROLLER_KWARGS",
    ):
        assert var not in matched_env


def test_static_k_customization_and_invalid_option_rejection():
    """Review Focus #2: static_k hợp lệ ghi vào speculative_config; bool/mâu thuẫn bị từ chối."""
    runner = LocalServingRunner()
    ok_cfg = _base_pair_cfg(enable_adaptive=False, match_adaptive_runtime=True, static_k=3)
    cmd = runner.build_command(ok_cfg)
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec["num_speculative_tokens"] == 3

    bad_configs = [
        _base_pair_cfg(enable_adaptive=True, disable_speculative=True),
        _base_pair_cfg(enable_adaptive=True, static_k=3),
        _base_pair_cfg(enable_adaptive=False, disable_speculative=True, static_k=3),
        _base_pair_cfg(enable_adaptive=False, static_k=True),
        _base_pair_cfg(enable_adaptive=False, static_k=0),
        _base_pair_cfg(enable_adaptive=False, static_k=9),
    ]
    for bad in bad_configs:
        with pytest.raises(ValueError):
            runner.build_command(bad)
        with pytest.raises(ValueError):
            runner.setup_environment(bad)


def test_modal_runner_forwards_matched_contract_keys_and_validates_early(capsys):
    """AC-04 & Review Focus #4: ModalServingRunner chuyển tiếp đủ 4 khóa và chặn cấu hình sai sớm."""
    from flexsds.server.modal_runner import ModalServingRunner

    runner = ModalServingRunner()
    cfg = _base_pair_cfg(
        enable_adaptive=False,
        match_adaptive_runtime=True,
        disable_prefix_caching=True,
        static_k=3,
        target_gpu="A10",
    )
    env = runner.setup_environment(cfg)
    payload = json.loads(env["FLEXSDS_SERVING_CONFIG"])
    assert payload["match_adaptive_runtime"] is True
    assert payload["disable_prefix_caching"] is True
    assert payload["static_k"] == 3

    runner.launch(cfg, deploy=False, dry_run=True)
    out = capsys.readouterr().out
    assert "Execution contract: static-matched-eager-v1+no-prefix-cache" in out

    invalid_cfg = _base_pair_cfg(
        enable_adaptive=False,
        disable_speculative=True,
        static_k=3,
        target_gpu="A10",
    )
    with pytest.raises(ValueError):
        runner.launch(invalid_cfg, deploy=False, dry_run=True)


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--match-adaptive-runtime", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--static-k", "3", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--vanilla", "--static-k", "3", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--vanilla", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--no-adaptive", "--static-k", "9", "--dry-run"],
    ],
)
def test_cli_rejects_invalid_matched_contract_flag_combinations(argv, capsys):
    """AC-05 & Review Focus #5: Chặn tổ hợp cờ CLI sai với exit code 2 trước preflight mạng."""
    from flexsds.cli.main import main

    rc = main(argv)
    assert rc == 2
    out = capsys.readouterr().out.lower()
    assert "invalid" in out


def test_cli_vanilla_matched_modal_dry_run_succeeds_offline_without_draft(capsys):
    """AC-05 & Review Focus #5: --vanilla --match-adaptive-runtime --dry-run chạy offline không cần --draft."""
    from flexsds.cli.main import main

    rc = main(
        [
            "serve",
            "--backend",
            "modal",
            "--gpu",
            "A10",
            "--target",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--vanilla",
            "--match-adaptive-runtime",
            "--no-prefix-caching",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Execution contract: vanilla-matched-eager-v1+no-prefix-cache" in out
    assert '"disable_speculative": true' in out
    assert '"match_adaptive_runtime": true' in out
    assert '"disable_prefix_caching": true' in out


def test_cli_static_matched_local_dry_run_builds_expected_contract(monkeypatch, capsys):
    """AC-05: flexsds serve --no-adaptive --match-adaptive-runtime --static-k 3 --dry-run."""
    from types import SimpleNamespace
    import flexsds.cli.main as cli_main

    fake_report = SimpleNamespace(
        draft_method="eagle3",
        eagle_version="EAGLE-3.1",
        target_revision="b25037543e9394b818fdfca67ab2a00ecc7dd641",
        draft_revision="d9016c3860f42cc3f33a87fa5d346fee768f6290",
        use_heterogeneous_vocab=False,
        identified_method="eagle3",
        oom_risk=False,
        warnings=[],
        recommendations=[],
        summary=lambda: "Compatibility: OK",
    )
    fake_preflight = SimpleNamespace(report=fake_report, launch_reasons=lambda allow_oom: [])
    monkeypatch.setattr(cli_main, "evaluate_serving_preflight", lambda *a, **kw: fake_preflight)

    rc = cli_main.main(
        [
            "serve",
            "--backend",
            "local",
            "--target",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--draft",
            "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "--no-adaptive",
            "--match-adaptive-runtime",
            "--no-prefix-caching",
            "--static-k",
            "3",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Execution contract: static-matched-eager-v1+no-prefix-cache" in out
    assert "--enforce-eager" in out
    assert "--async-scheduling" in out
    assert "--no-enable-prefix-caching" in out
    assert '"num_speculative_tokens": 3' in out
    assert "--scheduler-cls" not in out


def test_draft_model_static_k_syncs_batch_lookup_ceiling():
    """Adversarial check: draft_model static_k=3 không bị ghi cứng [[1, 32, 4]]."""
    runner = LocalServingRunner()
    cfg = _base_pair_cfg(
        draft_model="Qwen/Qwen2.5-0.5B-Instruct",
        draft_method="draft_model",
        enable_adaptive=False,
        match_adaptive_runtime=True,
        static_k=3,
    )
    cmd = runner.build_command(cfg)
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec["num_speculative_tokens"] == 3
    assert spec["num_speculative_tokens_per_batch_size"] == [[1, 32, 3]]


def test_cli_vanilla_without_target_on_local_cpu_blocks_instead_of_opening_wizard(monkeypatch, capsys):
    """Adversarial check: --vanilla không bị rơi ngầm vào wizard tương tác."""
    import flexsds.cli.main as cli_main
    from flexsds.core.hardware import HardwareInfo

    cpu_only = HardwareInfo(
        device_name="CPU Only",
        sm_architecture="cpu",
        total_vram_mb=0.0,
        free_vram_mb=0.0,
        cuda_version="N/A",
        is_cuda_available=False,
    )
    monkeypatch.setattr(cli_main, "detect_hardware", lambda: cpu_only)
    monkeypatch.setattr(
        cli_main,
        "run_interactive_wizard",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("wizard should not run for --vanilla")),
    )
    rc = cli_main.main(["serve", "--backend", "local", "--vanilla"])
    assert rc == 2
    assert "cuda gpu unavailable" in capsys.readouterr().out.lower()
