"""Lane configuration for the Modal runner must match the T4 launchers.

The launchers in scripts/launch_lane*.sh stay the source of truth for model,
serving limits and speculative config; only dtype/quantization may differ.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import modal_lanes as lanes


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    "vanilla": ROOT / "scripts" / "launch_lane1_vanilla.sh",
    "static": ROOT / "scripts" / "launch_lane2_static.sh",
    "adaptive": ROOT / "scripts" / "launch_lane3_adaptive.sh",
}
SHARED_FLAGS = (
    "--served-model-name",
    "--generation-config",
    "--gpu-memory-utilization",
    "--max-model-len",
    "--max-num-seqs",
)


def launcher_flags(lane: str) -> dict[str, str]:
    text = LAUNCHERS[lane].read_text(encoding="utf-8")
    return dict(re.findall(r"^\s*(--[a-z-]+)\s+(\S+)\s*\\?\s*$", text, flags=re.M))


def launcher_spec_config(lane: str) -> dict:
    text = LAUNCHERS[lane].read_text(encoding="utf-8")
    match = re.search(r"^SPEC_CONFIG='(\{.*\})'\s*$", text, flags=re.M)
    assert match, f"{LAUNCHERS[lane].name} has no SPEC_CONFIG"
    return json.loads(match.group(1))


def as_dict(args: list[str]) -> dict[str, str]:
    return {flag: value for flag, value in zip(args, args[1:]) if flag.startswith("--")}


@pytest.mark.parametrize("lane", ["vanilla", "static", "adaptive"])
def test_shared_flags_match_launcher(lane):
    args = lanes.build_vllm_args(lane)
    expected = launcher_flags(lane)

    assert args[0] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    for flag in SHARED_FLAGS:
        assert as_dict(args)[flag] == expected[flag], flag


@pytest.mark.parametrize("lane", ["static", "adaptive"])
def test_speculative_config_matches_launcher(lane):
    args = as_dict(lanes.build_vllm_args(lane))
    spec = json.loads(args["--speculative-config"])

    # Launchers add "revision" the same way when PROTOCOL_B_DRAFT_REVISION is set.
    assert spec.pop("revision") == lanes.DRAFT_REVISION
    assert spec == launcher_spec_config(lane)
    assert args["--per-request-spec-decode-metrics"] == "summary"


def test_vanilla_has_no_speculation():
    args = lanes.build_vllm_args("vanilla")

    assert "--speculative-config" not in args
    assert "--scheduler-cls" not in args


@pytest.mark.parametrize(
    "profile, dtype, quantization",
    [("l4-bf16", "bfloat16", "awq_marlin"), ("fp16-fallback", "half", "awq")],
)
def test_dtype_profiles(profile, dtype, quantization):
    args = as_dict(lanes.build_vllm_args("vanilla", profile=profile))

    assert args["--dtype"] == dtype
    assert args["--quantization"] == quantization


def test_fp16_fallback_equals_t4_launcher_kernel():
    args = as_dict(lanes.build_vllm_args("static", profile="fp16-fallback"))
    expected = launcher_flags("static")

    assert args["--dtype"] == expected["--dtype"]
    assert args["--quantization"] == expected["--quantization"]


def test_default_profile_is_l4_bf16():
    assert as_dict(lanes.build_vllm_args("vanilla"))["--dtype"] == "bfloat16"


def test_unknown_lane_or_profile_rejected():
    with pytest.raises(ValueError, match="lane"):
        lanes.build_vllm_args("eagle")
    with pytest.raises(ValueError, match="profile"):
        lanes.build_vllm_args("vanilla", profile="int8")


def test_server_listens_on_all_interfaces():
    args = as_dict(lanes.build_vllm_args("vanilla", port=8000))

    assert args["--host"] == "0.0.0.0"
    assert args["--port"] == "8000"


def test_adaptive_lane_uses_team_scheduler_and_env():
    args = as_dict(lanes.build_vllm_args("adaptive"))
    env = lanes.lane_env("adaptive", run_id="run-1")
    launcher = LAUNCHERS["adaptive"].read_text(encoding="utf-8")

    assert args["--scheduler-cls"] == "src.server.adaptive_scheduler.AdaptiveAsyncScheduler"
    assert args["--scheduler-cls"] in launcher
    for name in (
        "ADAPTIVE_ENABLE_VRAM_GUARD",
        "ADAPTIVE_MAX_K",
        "ADAPTIVE_MIN_BATCH_K",
        "ADAPTIVE_PER_REQUEST",
    ):
        default = re.search(rf'{name}="\$\{{{name}:-([^}}]*)\}}"', launcher).group(1)
        assert env[name] == default, name
    assert env["ADAPTIVE_TELEMETRY_PATH"] == "/artifacts/lane3/run-1/telemetry.jsonl"
    assert env["PYTHONPATH"].split(":")[0] == lanes.REMOTE_PROJECT_ROOT


@pytest.mark.parametrize("lane", ["vanilla", "static"])
def test_non_adaptive_lanes_have_no_adaptive_env(lane):
    assert not any(key.startswith("ADAPTIVE_") for key in lanes.lane_env(lane, run_id="r"))


def test_pinned_revisions_are_passed_when_set(monkeypatch):
    monkeypatch.setattr(lanes, "TARGET_REVISION", "a" * 40)
    monkeypatch.setattr(lanes, "DRAFT_REVISION", "b" * 40)

    args = as_dict(lanes.build_vllm_args("static"))

    assert args["--revision"] == "a" * 40
    assert json.loads(args["--speculative-config"])["revision"] == "b" * 40


def test_revisions_are_pinned_to_full_commit_hashes():
    for revision in (lanes.TARGET_REVISION, lanes.DRAFT_REVISION):
        assert re.fullmatch(r"[0-9a-f]{40}", revision or ""), revision


def _fake_snapshot(hub: Path, repo: str, sha: str) -> None:
    snapshot = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")


def test_offline_main_lookup_fails_without_ref(tmp_path):
    # vLLM loads the draft tokenizer by repo id with no revision (= "main").
    huggingface_hub = pytest.importorskip("huggingface_hub")
    _fake_snapshot(tmp_path, "Qwen/Qwen2.5-0.5B-Instruct", "b" * 40)

    found = huggingface_hub.try_to_load_from_cache(
        "Qwen/Qwen2.5-0.5B-Instruct", "config.json", cache_dir=tmp_path, revision="main"
    )

    assert not isinstance(found, str)


def test_pin_main_ref_makes_offline_main_resolve_to_pinned_snapshot(tmp_path):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    _fake_snapshot(tmp_path, "Qwen/Qwen2.5-0.5B-Instruct", "b" * 40)

    lanes.pin_main_ref(tmp_path, "Qwen/Qwen2.5-0.5B-Instruct", "b" * 40)

    found = huggingface_hub.try_to_load_from_cache(
        "Qwen/Qwen2.5-0.5B-Instruct", "config.json", cache_dir=tmp_path, revision="main"
    )
    assert isinstance(found, str)
    assert ("b" * 40) in found


def test_pin_main_ref_rejects_missing_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="snapshot"):
        lanes.pin_main_ref(tmp_path, "Qwen/Qwen2.5-0.5B-Instruct", "c" * 40)


def test_lane_identities_are_fixed():
    assert {lane: cfg.served_model_name for lane, cfg in lanes.LANES.items()} == {
        "vanilla": "qwen2.5-7b-vanilla",
        "static": "qwen2.5-7b-static-draft",
        "adaptive": "qwen2.5-7b-adaptive",
    }


def test_server_env_carries_profile_run_id_and_lane_env():
    env = lanes.server_env("adaptive", profile="fp16-fallback", run_id="r1")

    assert env["PTIT_LANE"] == "adaptive"
    assert env["PTIT_DTYPE_PROFILE"] == "fp16-fallback"
    assert env["PTIT_RUN_ID"] == "r1"
    assert env["ADAPTIVE_TELEMETRY_PATH"] == "/artifacts/lane3/r1/telemetry.jsonl"
    assert env["HF_HOME"] == lanes.HF_HOME


def test_server_env_rejects_unknown_profile():
    with pytest.raises(ValueError, match="profile"):
        lanes.server_env("vanilla", profile="int8", run_id="r1")


def test_vllm_version_matches_canonical_harness_pin():
    harness = (ROOT / "experiments" / "benchmark_harness.py").read_text(encoding="utf-8")

    assert f'EXPECTED_VLLM = "{lanes.VLLM_VERSION}"' in harness


def test_servers_read_weights_offline():
    assert lanes.server_env("vanilla", profile="l4-bf16", run_id="r")["HF_HUB_OFFLINE"] == "1"


def test_sources_contain_no_proxy_token():
    pattern = re.compile(r"\bw[ks]-[A-Za-z0-9]{16,}")
    sources = [ROOT / "experiments" / "modal_lanes.py"]
    sources += [p for p in (ROOT / "experiments" / "modal_runner.py",) if p.exists()]

    for path in sources:
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_lane_config_import_does_not_need_modal():
    code = "import sys, experiments.modal_lanes; assert 'modal' not in sys.modules"

    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------- [M3-01] server env + bench
def test_bench_workloads_match_m2_campaign():
    from experiments import run_m2_campaign

    assert lanes.BENCH_WORKLOADS == run_m2_campaign.WORKLOADS


def test_server_artifact_dir():
    assert lanes.server_artifact_dir("static", "r1") == "/artifacts/static/r1"


def test_build_server_env_has_harness_required_fields():
    from experiments import benchmark_harness as harness

    env = lanes.build_server_env(
        gpu="NVIDIA L4", driver_version="610.57.04", gpu_memory_total_mb=23034.0,
        torch_version="2.13.0+cu130", torch_cuda_version="13.0", vllm_version="0.30.0",
        lane="static", run_id="r1", profile="l4-bf16",
    )

    assert all(env[field] not in (None, "") for field in harness.SERVER_ENV_REQUIRED)
    assert env["dtype"] == "bfloat16"
    assert env["quantization"] == "awq_marlin"
    assert env["target_revision"] == lanes.TARGET_REVISION
    assert env["lane"] == "static"
    assert harness.validate_environment(env, profile=harness.L4_MODAL_PROFILE) == []


def test_bench_harness_args_are_canonical_and_parse(tmp_path):
    import sys
    from unittest.mock import patch
    from experiments import benchmark_harness as harness

    args = lanes.bench_harness_args(
        "adaptive", base_url="https://x.test/v1", server_env_json="/artifacts/adaptive/r1/server_env.json",
        raw_out="/artifacts/bench/b1/raw.jsonl", summary_csv="/artifacts/bench/b1/header.csv",
        experiment_id="m3-01-b1", workload="W3", concurrency=4, repeats=1, num_requests=8,
    )
    flags = dict(zip(args, args[1:]))

    assert flags["--endpoint"] == "https://x.test/v1/chat/completions"
    assert flags["--model"] == "qwen2.5-7b-adaptive"
    assert flags["--workload-file"] == "experiments/configs/w3_v2/chat_requests.jsonl"
    assert flags["--sample-policy"] == "cycle-rotate"
    assert flags["--warmup-requests"] == "3"
    assert flags["--environment-profile"] == "m3-l4-modal-v1"
    assert "--api-key" not in args  # token goes through OPENAI_API_KEY, never argv
    with patch.object(sys, "argv", ["harness", *args]):
        parsed = harness.parse_args()
    assert parsed.lane == "adaptive"
    assert parsed.concurrency == 4


def test_bench_rejects_unknown_workload():
    with pytest.raises(ValueError, match="workload"):
        lanes.bench_harness_args("vanilla", base_url="u", server_env_json="e", raw_out="r", summary_csv="s",
                                 experiment_id="x", workload="W9", concurrency=1, repeats=1, num_requests=1)


def test_harness_commit_can_come_from_environment(monkeypatch):
    from experiments import benchmark_harness as harness

    monkeypatch.setenv("PTIT_GIT_COMMIT", "a" * 40)

    assert harness.git_commit() == "a" * 40
