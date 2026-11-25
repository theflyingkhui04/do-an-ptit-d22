"""Lane configuration for the Modal runner must match the T4 launchers.

The launchers in scripts/launch_lane*.sh stay the source of truth for model,
serving limits and speculative config; only dtype/quantization may differ.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from experiments import modal_lanes as lanes


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    "vanilla": ROOT / "scripts" / "launch_lane1_vanilla.sh",
    "static": ROOT / "scripts" / "launch_lane2_static.sh",
    "adaptive": ROOT / "scripts" / "launch_lane3_adaptive.sh",
    "eagle": ROOT / "scripts" / "launch_lane4_eagle.sh",
}
SHARED_FLAGS = (
    "--served-model-name",
    "--generation-config",
    "--gpu-memory-utilization",
    "--max-model-len",
    "--max-num-seqs",
)


def launcher_flags(lane: str) -> dict[str, str]:
    if lane == "eagle":
        return as_dict(eagle_launcher_args())
    text = LAUNCHERS[lane].read_text(encoding="utf-8")
    return dict(re.findall(r"^\s*(--[a-z-]+)\s+(\S+)\s*\\?\s*$", text, flags=re.M))


def launcher_spec_config(lane: str) -> dict:
    if lane == "eagle":
        return json.loads(launcher_flags(lane)["--speculative-config"])
    text = LAUNCHERS[lane].read_text(encoding="utf-8")
    match = re.search(r"^SPEC_CONFIG='(\{.*\})'\s*$", text, flags=re.M)
    assert match, f"{LAUNCHERS[lane].name} has no SPEC_CONFIG"
    return json.loads(match.group(1))


def as_dict(args: list[str]) -> dict[str, str]:
    return {flag: value for flag, value in zip(args, args[1:]) if flag.startswith("--")}


def eagle_launcher_args() -> list[str]:
    text = LAUNCHERS["eagle"].read_text(encoding="utf-8")
    code = re.search(r"VLLM_ARGS_TEXT=\$\(python3 - <<'PY'\n(.*?)\nPY\n\)", text, re.S)
    assert code, "launcher must obtain its arguments from the canonical Python contract"
    result = subprocess.run([sys.executable, "-"], input=code.group(1).encode("utf-8"), cwd=ROOT,
                            capture_output=True)
    assert result.returncode == 0, result.stderr
    assert b"\r" not in result.stdout, "Bash mapfile must receive LF-delimited argv, including on Windows"
    assert 'mapfile -t VLLM_ARGS <<< "$VLLM_ARGS_TEXT"' in text
    assert '$VLLM_CMD "${VLLM_ARGS[@]}"' in text
    return result.stdout.decode("utf-8").splitlines()


@pytest.mark.parametrize("lane", ["vanilla", "static", "adaptive", "eagle"])
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


def test_eagle_speculative_config_matches_launcher():
    args = as_dict(lanes.build_vllm_args("eagle"))
    spec = json.loads(args["--speculative-config"])
    assert spec["revision"] == lanes.EAGLE_REVISION
    assert spec == launcher_spec_config("eagle")
    assert args["--per-request-spec-decode-metrics"] == "summary"
    assert eagle_launcher_args() == lanes.build_vllm_args("eagle", profile="fp16-fallback", port=8004)


def test_eagle_uses_frozen_eagle3_contract():
    spec = json.loads(as_dict(lanes.build_vllm_args("eagle"))["--speculative-config"])
    assert spec == {
        "method": "eagle3", "model": lanes.EAGLE_DRAFT_MODEL,
        "revision": lanes.EAGLE_REVISION, "num_speculative_tokens": 3,
        "draft_sample_method": "greedy",
    }
    assert "use_heterogeneous_vocab" not in spec
    assert lanes.EAGLE_REVISION == "c3adca7e211b471d97a519a625df7ffd37b8b461"
    assert as_dict(lanes.build_vllm_args("eagle"))["--revision"] == lanes.TARGET_REVISION


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
        lanes.build_vllm_args("unknown_lane")
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
    for revision in (lanes.TARGET_REVISION, lanes.DRAFT_REVISION, lanes.EAGLE_REVISION):
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
        "eagle": "qwen2.5-7b-eagle3-draft",
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


def test_adaptive_lineage_backend_and_runtime_packaging():
    args = as_dict(lanes.build_vllm_args("adaptive"))
    backend = "src.server.lineage_executor.LineageUniProcExecutor"
    assert args["--distributed-executor-backend"] == backend
    assert launcher_flags("adaptive")["--distributed-executor-backend"] == backend
    assert "--distributed-executor-backend" not in lanes.build_vllm_args("static")
    source = ast.parse((ROOT / "experiments/modal_runner.py").read_text(encoding="utf-8"))
    image = next(n.value for n in source.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "image" for t in n.targets))
    mounts = [n for n in ast.walk(image) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "add_local_dir"]
    assert any(ast.unparse(n.args[0]) == "SOURCE_ROOT / 'experiments'" for n in mounts)
    for name in ("scalar_vllm_source_manifest.json", "protocol_b_boundary_observer.py", "protocol_b.py"):
        assert (ROOT / "experiments" / name).is_file()


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


@pytest.mark.parametrize("lane", ["vanilla", "static", "adaptive", "eagle"])
def test_model_identity_survives_server_env_and_harness(tmp_path, monkeypatch, lane):
    from experiments import benchmark_harness as harness

    server = lanes.build_server_env(
        gpu="NVIDIA L4", driver_version="610.57.04", gpu_memory_total_mb=23034.0,
        torch_version="2.13.0+cu130", torch_cuda_version="13.0", vllm_version="0.30.0",
        lane=lane, run_id="r1", profile="l4-bf16",
    )
    expected_draft = None if lane == "vanilla" else (
        lanes.EAGLE_DRAFT_MODEL if lane == "eagle" else lanes.DRAFT_MODEL)
    expected_revision = None if lane == "vanilla" else (
        lanes.EAGLE_REVISION if lane == "eagle" else lanes.DRAFT_REVISION)
    assert server["target_model"] == lanes.TARGET_MODEL
    assert server["target_revision"] == lanes.TARGET_REVISION
    assert server["draft_model"] == expected_draft
    assert server["draft_revision"] == expected_revision
    assert server["lane"] == lane
    assert server["served_model_name"] == lanes.LANES[lane].served_model_name
    path = tmp_path / "server_env.json"
    path.write_text(json.dumps(server), encoding="utf-8")
    monkeypatch.setenv("PTIT_SERVER_ENV_ROOT", str(tmp_path))
    env, handle = harness.environment_info(0, profile=harness.L4_MODAL_PROFILE, server_env_json=path)
    assert handle is None
    for key in ("target_model", "target_revision", "draft_model", "draft_revision", "served_model_name"):
        assert env[key] == server[key]
    assert env["server_lane"] == lane

    # Exercise the real request record and JSONL writer, without HTTP or GPU calls.
    async def events(response):
        yield json.dumps({"id": "r", "choices": [{"delta": {"content": "ok"}}]})
        yield json.dumps({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
        yield "[DONE]"

    session = Mock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=SimpleNamespace(status=200))
    session.post.return_value = context
    metadata = {key: "fixture" for key in (
        "workload_id", "dataset_version", "sample_id", "sample_index", "sample_offset",
        "canonical_sample_count", "sample_policy", "prompt_version")}
    with patch.object(harness, "iter_sse", events):
        record = asyncio.run(harness.run_one(
            session, asyncio.Semaphore(1),
            SimpleNamespace(lane=lane, workload="W1", endpoint="https://test.invalid"),
            env, {"request_id": "cpu-only"}, 1,
            {"model": server["served_model_name"]}, metadata,
        ))
    assert record["success"]
    raw_path = tmp_path / "raw.jsonl"
    harness.RawJsonlWriter(raw_path, overwrite=False).write([record])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    for key in ("target_model", "target_revision", "draft_model", "draft_revision", "served_model_name"):
        assert raw[key] == server[key]


def test_canonical_lane_enumeration_has_no_fifth_condition(monkeypatch):
    from experiments import benchmark_harness as harness
    from unittest.mock import patch

    assert tuple(lanes.LANES) == ("vanilla", "static", "adaptive", "eagle")
    with pytest.raises(ValueError, match="Unknown lane"):
        lanes.build_vllm_args("eagle-custom")
    with patch.object(sys, "argv", ["harness", "--model", "m", "--lane", "eagle-custom"]):
        with pytest.raises(SystemExit):
            harness.parse_args()
    tree = ast.parse((ROOT / "experiments" / "modal_runner.py").read_text(encoding="utf-8"))
    selection = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ENABLED_LANES"
                                           for t in node.targets)) or (
        isinstance(node, ast.If) and isinstance(node.test, ast.NamedExpr)
        and node.test.target.id == "unknown_lanes")]
    assert len(selection) == 2
    code = compile(ast.Module(body=selection, type_ignores=[]), "modal_runner.py", "exec")
    monkeypatch.delenv("PTIT_LANES", raising=False)
    namespace = {"os": os, "lanes": lanes}
    exec(code, namespace)
    assert namespace["ENABLED_LANES"] == set(lanes.LANES)
    monkeypatch.setenv("PTIT_LANES", "eagle-custom")
    with pytest.raises(ValueError, match="Unknown deployment lanes"):
        exec(code, namespace)


@pytest.fixture
def weight_download(tmp_path, monkeypatch):
    """Execute the real CPU downloader body without importing/deploying Modal."""
    import huggingface_hub

    tree = ast.parse((ROOT / "experiments" / "modal_runner.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "download_weights")
    function.decorator_list = []
    api = Mock()
    pins = {lanes.TARGET_MODEL: lanes.TARGET_REVISION, lanes.DRAFT_MODEL: lanes.DRAFT_REVISION,
            lanes.EAGLE_DRAFT_MODEL: lanes.EAGLE_REVISION}
    api.model_info.side_effect = lambda repo, revision: Mock(sha=pins[repo])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(lanes, "HF_HOME", str(tmp_path))
    hub = tmp_path / "hub"
    for repo, sha in pins.items():
        _fake_snapshot(hub, repo, sha)

    def snapshot(repo, *, revision, local_files_only=False):
        return str(hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision)

    download = Mock(side_effect=snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    volume = Mock()
    namespace = {"lanes": lanes, "Path": Path, "datetime": datetime, "timezone": timezone,
                 "json": json, "MANIFEST_NAME": "ptit_weights_manifest.json", "model_volume": volume}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "modal_runner.py", "exec"), namespace)
    return namespace["download_weights"], api, download, volume


@pytest.mark.parametrize("cached", [True, False])
def test_download_manifest_and_cache_use_exact_pins(weight_download, tmp_path, cached):
    run, api, download, volume = weight_download
    snapshot = download.side_effect

    def fetch(repo, **kwargs):
        if not cached and kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        return snapshot(repo, **kwargs)

    download.side_effect = fetch
    manifest = run()
    assert json.loads((tmp_path / "ptit_weights_manifest.json").read_text()) == manifest
    pins = {lanes.TARGET_MODEL: lanes.TARGET_REVISION, lanes.DRAFT_MODEL: lanes.DRAFT_REVISION,
            lanes.EAGLE_DRAFT_MODEL: lanes.EAGLE_REVISION}
    assert len(manifest["models"]) == 3
    for row in manifest["models"]:
        sha = pins[row["repo"]]
        assert row["requested_revision"] == row["resolved_revision"] == row["revision"] == sha
        assert row["already_present"] is cached
        api.model_info.assert_any_call(row["repo"], revision=sha)
        ref = tmp_path / "hub" / f"models--{row['repo'].replace('/', '--')}" / "refs" / "main"
        assert ref.read_text() == sha
    assert len(download.call_args_list) == 6
    for call in download.call_args_list:
        assert call.kwargs["revision"] == pins[call.args[0]]
    volume.commit.assert_called_once()


def test_download_rejects_resolved_eagle_sha_mismatch(weight_download, tmp_path):
    run, api, download, volume = weight_download
    api.model_info.side_effect = lambda repo, revision: Mock(
        sha="f" * 40 if repo == lanes.EAGLE_DRAFT_MODEL else revision)
    with pytest.raises(ValueError, match="Revision mismatch"):
        run()
    assert not any(call.args[0] == lanes.EAGLE_DRAFT_MODEL for call in download.call_args_list)
    assert not (tmp_path / "ptit_weights_manifest.json").exists()
    volume.commit.assert_not_called()


def test_download_rejects_missing_eagle_pin(weight_download, tmp_path, monkeypatch):
    run, api, download, volume = weight_download
    monkeypatch.setattr(lanes, "EAGLE_REVISION", None)
    with pytest.raises(ValueError, match="Missing pinned revision"):
        run()
    assert not any(call.args[0] == lanes.EAGLE_DRAFT_MODEL for call in download.call_args_list)
    assert not (tmp_path / "ptit_weights_manifest.json").exists()
    volume.commit.assert_not_called()


def test_gpu_defaults_to_l4():
    assert lanes.DEFAULT_GPU == "L4"
    assert lanes.resolve_gpu({}) == "L4"
    assert lanes.resolve_gpu({"PTIT_GPU": ""}) == "L4"


@pytest.mark.parametrize("gpu", ["L4", "A10", "T4"])
def test_gpu_override_from_env(gpu):
    assert lanes.resolve_gpu({"PTIT_GPU": gpu}) == gpu


def test_unknown_gpu_rejected():
    with pytest.raises(ValueError, match="PTIT_GPU"):
        lanes.resolve_gpu({"PTIT_GPU": "H100"})


# ---------------------------------------------------------------- [M5-01] long context + W1_V2
def test_default_max_model_len_is_unchanged():
    assert lanes.DEFAULT_MAX_MODEL_LEN == 4096
    for lane in lanes.LANES:
        assert lanes.build_vllm_args(lane) == lanes.build_vllm_args(lane, max_model_len=4096)
        assert as_dict(lanes.build_vllm_args(lane))["--max-model-len"] == "4096"


@pytest.mark.parametrize("lane", ["vanilla", "static", "adaptive", "eagle"])
def test_long_context_changes_only_max_model_len(lane):
    default = lanes.build_vllm_args(lane)
    long_context = lanes.build_vllm_args(lane, max_model_len=8192)
    index = default.index("--max-model-len") + 1

    assert long_context[index] == "8192"
    assert long_context[:index] + long_context[index + 1:] == default[:index] + default[index + 1:]


def test_resolve_max_model_len():
    assert lanes.resolve_max_model_len({}) == 4096
    assert lanes.resolve_max_model_len({"PTIT_MAX_MODEL_LEN": "8192"}) == 8192
    for bad in ("5000", "abc", ""):
        with pytest.raises(ValueError):
            lanes.resolve_max_model_len({"PTIT_MAX_MODEL_LEN": bad})
    with pytest.raises(ValueError):
        lanes.build_vllm_args("vanilla", max_model_len=5000)


def test_w1_v2_is_a_candidate_workload_outside_the_campaigns():
    assert "W1_V2" not in lanes.BENCH_WORKLOADS
    assert lanes.workload_file("W1_V2") == "experiments/configs/w1_v2/node1_chat_requests.jsonl"
    assert lanes.workload_file("W1") == lanes.BENCH_WORKLOADS["W1"]
    assert (ROOT / lanes.workload_file("W1_V2")).is_file()
    with pytest.raises(ValueError):
        lanes.workload_file("W9")

    args = lanes.bench_harness_args(
        "vanilla", base_url="https://x.test/v1", server_env_json="/a/server_env.json",
        raw_out="/a/raw.jsonl", summary_csv="/a/h.csv", experiment_id="m5-01",
        workload="W1_V2", concurrency=1, repeats=1, num_requests=8,
    )
    flags = dict(zip(args, args[1:]))
    assert flags["--workload"] == "W1_V2"
    assert flags["--workload-file"] == "experiments/configs/w1_v2/node1_chat_requests.jsonl"


def test_w1_v2_label_maps_to_the_w1_package_id():
    from experiments.workload_adapter import validate_workload_label

    assert validate_workload_label("W1_V2", "w1_node1_invoice_extraction") == "W1_V2"
    assert validate_workload_label("w1_v2", "w1_node1_invoice_extraction") == "W1_V2"


# [M5-01 web demo] the demo lanes run as their own Modal application.
def test_app_name_default_and_demo():
    assert lanes.resolve_app_name({}) == "ptit-vllm" == lanes.APP_NAME
    assert lanes.resolve_app_name({"PTIT_APP_NAME": ""}) == "ptit-vllm"
    assert lanes.resolve_app_name({"PTIT_APP_NAME": "ptit-vllm-demo"}) == "ptit-vllm-demo"
    for bad in ("Ptit", "ptit_vllm", "a/b", "-x", "x-", "x" * 41):
        with pytest.raises(ValueError, match="PTIT_APP_NAME"):
            lanes.resolve_app_name({"PTIT_APP_NAME": bad})


def test_runner_image_carries_the_app_name():
    source = (ROOT / "experiments/modal_runner.py").read_text(encoding="utf-8")
    assert "APP_NAME = lanes.resolve_app_name(os.environ)" in source
    assert '"PTIT_APP_NAME": APP_NAME' in source
