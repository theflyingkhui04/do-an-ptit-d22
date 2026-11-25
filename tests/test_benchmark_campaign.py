"""Offline TDD tests for the multi-GPU 3-node 3-lane FlexSDS benchmark flow."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments import benchmark_harness as harness
from experiments import modal_lanes as lanes
from flexsds.core import calibration as cal
from flexsds.server.local_runner import LocalServingRunner


# --------------------------------------------------------------------------
# Task 1: 4-GPU Modal support, harness server profiles, and T4 FP16 AWQ fix
# --------------------------------------------------------------------------
def test_modal_lanes_supports_all_four_production_gpus_plus_legacy_l4():
    for gpu in ("L4", "T4", "A10", "L40S", "A100"):
        assert gpu in lanes.SUPPORTED_GPUS
        assert lanes.resolve_gpu({"PTIT_GPU": gpu}) == gpu
    assert lanes.DEFAULT_GPU == "L4"
    assert lanes.modal_gpu_resource("T4") == "T4"
    assert lanes.modal_gpu_resource("A10") == "A10"
    assert lanes.modal_gpu_resource("L40S") == "L40S"
    assert lanes.modal_gpu_resource("A100") == "A100-80GB"
    assert lanes.modal_gpu_resource("L4") == "L4"

    assert lanes.default_dtype_profile("T4") == "fp16-fallback"
    for gpu in ("A10", "L40S", "A100", "L4"):
        assert lanes.default_dtype_profile(gpu) == "l4-bf16"

    assert lanes.environment_profile_for_gpu("T4") == "modal-t4-v1"
    assert lanes.environment_profile_for_gpu("A10") == "modal-a10-v1"
    assert lanes.environment_profile_for_gpu("L40S") == "modal-l40s-v1"
    assert lanes.environment_profile_for_gpu("A100") == "modal-a100-v1"
    assert lanes.environment_profile_for_gpu("L4") == "m3-l4-modal-v1"

    args = lanes.bench_harness_args(
        "vanilla",
        base_url="https://demo.modal.run/v1",
        server_env_json="/artifacts/server_env.json",
        raw_out="/artifacts/raw.jsonl",
        summary_csv="/artifacts/h.csv",
        experiment_id="exp-1",
        workload="W1_V3",
        concurrency=8,
        repeats=5,
        num_requests=32,
        environment_profile="modal-a100-v1",
    )
    assert args[args.index("--environment-profile") + 1] == "modal-a100-v1"


@pytest.mark.parametrize(
    "profile,good_gpu,bad_gpu",
    [
        ("modal-t4-v1", "Tesla T4", "NVIDIA A10"),
        ("modal-a10-v1", "NVIDIA A10", "NVIDIA RTX 3090"),
        ("modal-a10-v1", "NVIDIA A10G", "NVIDIA L4"),
        ("modal-l40s-v1", "NVIDIA L40S", "NVIDIA L4"),
        ("modal-a100-v1", "NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-40GB"),
        ("modal-a100-v1", "NVIDIA A100 80GB PCIe", "NVIDIA L40S"),
    ],
)
def test_harness_modal_server_profiles_enforce_strict_gpu_sku_matching(profile, good_gpu, bad_gpu):
    assert profile in harness.ENVIRONMENT_PROFILES
    assert harness.ENVIRONMENT_PROFILES[profile]["mode"] == "server"
    base_env = {
        "vllm_version": "0.30.0",
        "torch_version": "2.13.0+cu130",
        "torch_cuda_version": "13.0",
    }
    assert harness.validate_environment({**base_env, "gpu": good_gpu}, profile=profile) == []
    mismatches = harness.validate_environment({**base_env, "gpu": bad_gpu}, profile=profile)
    assert len(mismatches) == 1 and "gpu" in mismatches[0]


def test_local_runner_uses_fp16_awq_on_t4_and_bf16_marlin_on_ampere_ada():
    runner = LocalServingRunner()
    base = {
        "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "disable_speculative": True,
        "enable_adaptive": False,
    }
    t4_cmd = runner.build_command({**base, "target_gpu": "T4"})
    assert t4_cmd[t4_cmd.index("--dtype") + 1] == "half"
    assert t4_cmd[t4_cmd.index("--quantization") + 1] == "awq"

    t4_sm_cmd = runner.build_command({**base, "hardware": {"sm_architecture": "sm_75"}})
    assert t4_sm_cmd[t4_sm_cmd.index("--dtype") + 1] == "half"
    assert t4_sm_cmd[t4_sm_cmd.index("--quantization") + 1] == "awq"

    for gpu in ("A10", "L40S", "A100"):
        cmd = runner.build_command({**base, "target_gpu": gpu})
        assert cmd[cmd.index("--dtype") + 1] == "bfloat16"
        assert cmd[cmd.index("--quantization") + 1] == "awq_marlin"


def test_calibration_awq_dtype_selection_handles_t4_fp16_and_bf16_gpus():
    assert cal.resolve_calibration_dtype_quant("Qwen/Qwen2.5-7B-Instruct-AWQ", "sm_75") == ("float16", "awq")
    assert cal.resolve_calibration_dtype_quant("Qwen/Qwen2.5-7B-Instruct-AWQ", "sm_86") == ("bfloat16", "awq_marlin")
    assert cal.resolve_calibration_dtype_quant("Qwen/Qwen2.5-7B-Instruct-AWQ", "sm_89") == ("bfloat16", "awq_marlin")
    assert cal.resolve_calibration_dtype_quant("Qwen/Qwen2.5-7B-Instruct-AWQ", "sm_80") == ("bfloat16", "awq_marlin")
    assert cal.resolve_calibration_dtype_quant("org/model-gptq", "sm_86") == ("float16", "gptq")
    assert cal.resolve_calibration_dtype_quant("meta-llama/Llama-3.1-8B", "sm_75") == ("float16", None)
    assert cal.resolve_calibration_dtype_quant("meta-llama/Llama-3.1-8B", "sm_86") == ("bfloat16", None)


# --------------------------------------------------------------------------
# Task 2: Multi-GPU 3-node 3-lane campaign planner, validator & aggregator
# --------------------------------------------------------------------------
def test_benchmark_campaign_full_and_fast_matrix_counts_and_protocol(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    from flexsds.core import cost_model as cm

    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path / "empty_user_profiles")

    full = camp.plan("bench-full-v1", "a" * 40, mode="full")
    assert full["mode"] == "full"
    assert tuple(full["gpus"]) == ("T4", "A10", "L40S", "A100")
    assert tuple(full["workloads"]) == ("W1_V3", "W2", "W3")
    assert tuple(full["lanes"]) == ("vanilla", "static", "adaptive")
    assert tuple(full["concurrencies"]) == (1, 2, 4, 8, 16, 32)
    assert (full["warmup"], full["repeats"], full["num_requests"]) == (3, 5, 32)
    assert len(full["cells"]) == 216
    assert len({c["key"] for c in full["cells"]}) == 216
    assert full["total_measured_requests"] == 216 * 160

    fast = camp.plan("bench-fast-v1", "a" * 40, mode="fast")
    assert fast["mode"] == "fast"
    assert tuple(fast["gpus"]) == ("T4", "A10", "L40S", "A100")
    assert tuple(fast["workloads"]) == ("W1_V3", "W2", "W3")
    assert tuple(fast["lanes"]) == ("vanilla", "static", "adaptive")
    assert tuple(fast["concurrencies"]) == (1, 8, 32)
    assert (fast["warmup"], fast["repeats"], fast["num_requests"]) == (3, 5, 32)
    assert len(fast["cells"]) == 108
    assert len({c["key"] for c in fast["cells"]}) == 108
    assert fast["total_measured_requests"] == 108 * 160

    # Check GPU-specific fields and workload context/timeout
    by_key = {c["key"]: c for c in fast["cells"]}
    t4_w1 = by_key["T4__static__W1_V3__c8"]
    assert t4_w1["modal_gpu"] == "T4"
    assert t4_w1["dtype_profile"] == "fp16-fallback"
    assert t4_w1["environment_profile"] == "modal-t4-v1"
    assert t4_w1["max_model_len"] == 8192
    assert t4_w1["timeout_s"] == 600

    a100_w2 = by_key["A100__adaptive__W2__c32"]
    assert a100_w2["modal_gpu"] == "A100-80GB"
    assert a100_w2["dtype_profile"] == "l4-bf16"
    assert a100_w2["environment_profile"] == "modal-a100-v1"
    assert a100_w2["max_model_len"] == 4096

    # Bundled A10 profile is ready; T4/L40S/A100 are reported missing without crashing plan().
    readiness = fast["profile_readiness"]
    assert readiness["A10"]["status"] == "ready"
    assert readiness["T4"]["status"] == "missing"
    assert readiness["L40S"]["status"] == "missing"
    assert readiness["A100"]["status"] == "missing"


def test_build_cell_serving_config_matches_three_lanes_contract(tmp_path):
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("bench-lanes", "a" * 40, mode="fast", static_k=3)
    by_key = {c["key"]: c for c in record["cells"]}
    fake_profile = tmp_path / "profile_a10.json"
    fake_profile.write_text("{}", encoding="utf-8")

    vanilla = camp.build_cell_serving_config(record, by_key["T4__vanilla__W1_V3__c1"])
    assert vanilla["disable_speculative"] is True
    assert vanilla["enable_adaptive"] is False
    assert vanilla["draft_model"] is None
    assert vanilla["dtype"] == "half"
    assert vanilla["quantization"] == "awq"
    assert vanilla["max_model_len"] == 8192

    static = camp.build_cell_serving_config(record, by_key["L40S__static__W2__c8"])
    assert static["disable_speculative"] is False
    assert static["enable_adaptive"] is False
    assert static["draft_model"] == camp.DEFAULT_DRAFT_MODEL
    assert static["static_k"] == 3
    assert static["dtype"] == "bfloat16"
    assert static["quantization"] == "awq_marlin"
    assert static["max_model_len"] == 4096

    adaptive = camp.build_cell_serving_config(
        record, by_key["A100__adaptive__W3__c32"], profile_path=str(fake_profile)
    )
    assert adaptive["disable_speculative"] is False
    assert adaptive["enable_adaptive"] is True
    assert adaptive["draft_model"] == camp.DEFAULT_DRAFT_MODEL
    assert adaptive["max_k"] == 4
    assert adaptive["modal_gpu"] == "A100-80GB"
    assert adaptive["profile_path"] == str(fake_profile)

    with pytest.raises(ValueError, match="profile"):
        camp.build_cell_serving_config(record, by_key["A100__adaptive__W3__c32"], profile_path=None)


def test_ensure_adaptive_profiles_fail_closed_and_auto_calibrate(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    from flexsds.core import cost_model as cm

    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path / "user")
    # Default fail-closed raises listing missing GPUs and calibration commands.
    with pytest.raises(RuntimeError) as exc_info:
        camp.ensure_adaptive_profiles(
            ("T4", "A10", "L40S", "A100"),
            target_model=camp.DEFAULT_TARGET_MODEL,
            draft_model=camp.DEFAULT_DRAFT_MODEL,
            auto_calibrate=False,
        )
    msg = str(exc_info.value)
    assert "T4" in msg and "L40S" in msg and "A100" in msg
    assert "flexsds profile --benchmark --backend modal --gpu" in msg

    # With --auto-calibrate, calls run_hardware_micro_benchmark only for missing GPUs (skips ready A10).
    calibrated: list[tuple[str, str]] = []
    ready_after: set[str] = {"A10"}

    def fake_resolve(gpus, **kwargs):
        return {
            g: {
                "status": "ready" if g in ready_after else "missing",
                "profile_path": str(tmp_path / f"{g}.json") if g in ready_after else None,
                "profile_name": f"{g}.json" if g in ready_after else None,
                "profile_sha256": "f" * 64 if g in ready_after else None,
                "calibrate_command": f"flexsds profile --benchmark --backend modal --gpu {g}",
            }
            for g in gpus
        }

    def fake_micro_bench(hw_info, **kwargs):
        gpu_name = hw_info.device_name.removeprefix("Remote ")
        calibrated.append((gpu_name, kwargs["measurement_mode"]))
        ready_after.add(gpu_name)
        return object(), tmp_path / f"{gpu_name}.json"

    monkeypatch.setattr(camp, "resolve_adaptive_profiles", fake_resolve)
    monkeypatch.setattr(cm, "run_hardware_micro_benchmark", fake_micro_bench)

    resolved = camp.ensure_adaptive_profiles(
        ("T4", "A10", "L40S", "A100"),
        target_model=camp.DEFAULT_TARGET_MODEL,
        draft_model=camp.DEFAULT_DRAFT_MODEL,
        auto_calibrate=True,
        calibrate_mode="fast",
    )
    assert calibrated == [("T4", "fast"), ("L40S", "fast"), ("A100", "fast")]
    assert all(resolved[g]["status"] == "ready" for g in ("T4", "A10", "L40S", "A100"))


def test_aggregate_computes_speedup_per_gpu_independently(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    from experiments import vllm_runtime_patch as patch
    from experiments.bench_source import snapshot
    from experiments.workload_adapter import load_workload_rows, verify_workload_integrity

    packed, commit = snapshot(camp.ROOT)
    try:
        w2_path = Path(packed.name) / lanes.workload_file("W2")
        integrity = verify_workload_integrity(w2_path)
        corpus = load_workload_rows(w2_path)
    finally:
        packed.cleanup()

    record = camp.plan(
        "cpu-multi-gpu",
        commit,
        mode="fast",
        gpus=("T4", "A100"),
        workloads=("W2",),
        lanes_list=("vanilla", "static", "adaptive"),
    )
    # Keep only concurrency=1 so fixture creation stays super fast while exercising multi-GPU aggregation.
    record["concurrencies"] = [1]
    record["cells"] = [c for c in record["cells"] if c["concurrency"] == 1]

    plan_root = tmp_path / "runs"
    monkeypatch.setattr(camp, "ROOT", tmp_path)
    camp.save_plan(record, plan_root=plan_root)

    tpot_map = {
        ("T4", "vanilla"): 40.0,
        ("T4", "static"): 25.0,
        ("T4", "adaptive"): 20.0,
        ("A100", "vanilla"): 10.0,
        ("A100", "static"): 8.0,
        ("A100", "adaptive"): 5.0,
    }
    gpu_device = {"T4": "Tesla T4", "A100": "NVIDIA A100-SXM4-80GB"}
    profile_sha = "d" * 64

    for idx, cell in enumerate(record["cells"]):
        gpu = cell["gpu"]
        lane = cell["lane"]
        attempt_id = f"{camp.attempt_prefix(record, cell)}001"
        out = plan_root / record["campaign_id"] / "attempts" / attempt_id
        out.mkdir(parents=True)
        cfg = camp.build_cell_serving_config(
            record,
            cell,
            profile_path=str(tmp_path / "prof.json") if lane == "adaptive" else None,
        )
        receipt = {
            "gpu": gpu,
            "lane": lane,
            "server_run_id": attempt_id,
            "bench_run_id": attempt_id,
            "call_id": f"fc-test{idx:03d}",
            "commit": commit,
            "cell": {
                "workload": "W2",
                "concurrency": 1,
                "repeats": 5,
                "num_requests": 32,
            },
        }
        server_env = {
            "gpu": gpu_device[gpu],
            "requested_gpu": gpu,
            "modal_gpu": lanes.modal_gpu_resource(gpu),
            "driver_version": "550.0",
            "gpu_memory_total_mb": 15360.0 if gpu == "T4" else 81920.0,
            "torch_version": "2.13.0+cu130",
            "torch_cuda_version": "13.0",
            "vllm_version": "0.30.0",
            "dtype": cfg["dtype"],
            "quantization": cfg["quantization"],
            "lane": lane,
            "run_id": attempt_id,
            "instance_id": f"boot-{idx}",
            "source_commit": commit,
            "target_model": record["target_model"],
            "draft_model": cfg["draft_model"],
            "served_model_name": cfg["served_model_name"],
            "max_model_len": cfg["max_model_len"],
            "runtime_variant": patch.RUNTIME_LABEL,
            "vllm_outputs_sha256": patch.PATCHED_OUTPUTS_SHA256,
            "profile_sha256": profile_sha if lane == "adaptive" else None,
        }
        raw, warm = [], []
        for repeat in range(6):
            phase = "warmup" if repeat == 0 else "measured"
            offset = ((repeat - 1) * (32 % len(corpus))) % len(corpus) if repeat else 0
            for req_idx in range(3 if repeat == 0 else 32):
                sample_index = (req_idx + offset) % len(corpus)
                req = corpus[sample_index]["request"]
                row = {
                    "experiment_id": attempt_id,
                    "lane": lane,
                    "workload": "W2",
                    "concurrency": 1,
                    "phase": phase,
                    "repeat_id": f"r{repeat:03}" if repeat else None,
                    "request_id": f"{attempt_id}-{repeat}-{req_idx}",
                    "request_index": req_idx,
                    "commit": commit,
                    "server_run_id": attempt_id,
                    "server_instance_id": f"boot-{idx}",
                    "sample_policy": "cycle-rotate",
                    "sample_id": corpus[sample_index]["sample_id"],
                    "sample_index": sample_index,
                    "sample_offset": offset,
                    "canonical_sample_count": len(corpus),
                    "success": True,
                    "workload_file_sha256": integrity.actual_sha256,
                    "metrics_complete": True,
                    "http_status": 200,
                    "model": cfg["served_model_name"],
                    "temperature": req["temperature"],
                    "seed": req["seed"],
                    "max_completion_tokens": req["max_tokens"],
                    "response_format": req.get("response_format"),
                    "timestamp_utc": f"2026-11-25T00:0{repeat}:00+00:00",
                    "e2e_ms": 1000.0,
                    "ttft_ms": 12.0,
                    "tpot_ms": tpot_map[(gpu, lane)],
                    "output_tokens": 50,
                    "gpu": gpu_device[gpu],
                    "run_id": f"{attempt_id}-run-{repeat}",
                    "run_duration_s": 32.0,
                    "run_vram_peak_mb": 9000.0,
                }
                if lane != "vanilla":
                    row["server_metrics"] = {
                        "speculative_decoding": {
                            "num_spec_steps": 5,
                            "num_draft_tokens": 15,
                            "num_accepted_draft_tokens": 9,
                            "num_spec_tokens": 3,
                        }
                    }
                (raw if repeat else warm).append(row)

        (out / "submission.json").write_text(json.dumps(receipt), encoding="utf-8")
        (out / "bench.json").write_text(
            json.dumps({**receipt, "returncode": 0, "server_instance_unchanged": True,
                        "endpoint_model_ids": [cfg["served_model_name"]]}),
            encoding="utf-8",
        )
        (out / "server_env.json").write_text(json.dumps(server_env), encoding="utf-8")
        raw_text = "".join(json.dumps(r) + "\n" for r in raw)
        (out / "raw.jsonl").write_text(raw_text, encoding="utf-8")
        (out / "raw.vram.jsonl").write_text(raw_text, encoding="utf-8")
        (out / "warmup.jsonl").write_text("".join(json.dumps(r) + "\n" for r in warm), encoding="utf-8")

    # Need snapshot to read from the real repo root while evidence lives in tmp_path.
    monkeypatch.setattr(camp, "REPO_ROOT", Path(__file__).resolve().parents[1])
    dest = tmp_path / "summary.json"
    report_dest = tmp_path / "report.md"
    result = camp.aggregate(record, plan_root=plan_root, destination=dest, report_destination=report_dest)

    assert dest.is_file() and report_dest.is_file()
    by_gpu_lane = {(r["gpu_tier"], r["lane"]): r for r in result["cells"]}
    assert by_gpu_lane[("T4", "vanilla")]["speedup_vs_vanilla"] == pytest.approx(1.0)
    assert by_gpu_lane[("T4", "static")]["speedup_vs_vanilla"] == pytest.approx(40.0 / 25.0)
    assert by_gpu_lane[("T4", "adaptive")]["speedup_vs_vanilla"] == pytest.approx(40.0 / 20.0)
    assert by_gpu_lane[("A100", "vanilla")]["speedup_vs_vanilla"] == pytest.approx(1.0)
    assert by_gpu_lane[("A100", "static")]["speedup_vs_vanilla"] == pytest.approx(10.0 / 8.0)
    assert by_gpu_lane[("A100", "adaptive")]["speedup_vs_vanilla"] == pytest.approx(10.0 / 5.0)

    md = report_dest.read_text(encoding="utf-8")
    assert "T4" in md and "A100" in md and "W2" in md

    # Tampering A100 server_env to 40GB must fail closed in validate_cell.
    a100_cell = next(c for c in record["cells"] if c["gpu"] == "A100" and c["lane"] == "vanilla")
    a100_dir = camp.attempts(record, a100_cell, plan_root=plan_root)[0]
    env_path = a100_dir / "server_env.json"
    bad_env = json.loads(env_path.read_text(encoding="utf-8"))
    bad_env["gpu"] = "NVIDIA A100-SXM4-40GB"
    env_path.write_text(json.dumps(bad_env), encoding="utf-8")
    with pytest.raises(ValueError, match="Server GPU mismatch"):
        camp.validate_cell(a100_dir, record, a100_cell)


# --------------------------------------------------------------------------
# Task 3: Dedicated Modal benchmark app runner (offline mocked modal SDK)
# --------------------------------------------------------------------------
def _load_modal_benchmark_runner(monkeypatch, cfg: dict):
    import base64
    import importlib.util
    import sys
    from types import SimpleNamespace

    encoded = base64.b64encode(json.dumps(cfg).encode("utf-8")).decode("ascii")
    monkeypatch.setenv("FLEXSDS_BENCH_CONFIG_B64", encoded)
    monkeypatch.setenv("FLEXSDS_BENCH_RUN_ID", cfg.get("run_id", "bench-run-1"))

    image = Mock()
    for method in ("pip_install", "uv_pip_install", "entrypoint", "env", "add_local_dir", "add_local_file", "run_commands"):
        getattr(image, method).return_value = image

    fn_calls: list[dict] = []
    app = Mock()

    def fake_function(**kwargs):
        fn_calls.append(kwargs)
        return lambda f: f

    app.function.side_effect = fake_function
    web_server = Mock(side_effect=lambda *a, **kw: lambda f: f)
    fake_modal = SimpleNamespace(
        App=Mock(return_value=app),
        Image=SimpleNamespace(from_registry=Mock(return_value=image)),
        Volume=SimpleNamespace(from_name=Mock(return_value="volume")),
        web_server=web_server,
        is_local=lambda: True,
        Function=SimpleNamespace(from_name=Mock()),
        FunctionCall=SimpleNamespace(from_id=Mock()),
        current_function_call_id=lambda: "fc-current01",
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    runner_path = Path(__file__).resolve().parents[1] / "experiments" / "modal_benchmark_runner.py"
    spec = importlib.util.spec_from_file_location("test_modal_benchmark_runner_mod", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_modal, image, fn_calls, web_server


def test_modal_benchmark_runner_allocates_a100_80gb_patches_vllm_and_mounts_profile(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    prof = tmp_path / "profile_a100.json"
    prof.write_text(json.dumps({"metadata": {"raw_sha256": "e" * 64}}), encoding="utf-8")
    record = camp.plan("bench-a100", "a" * 40, mode="fast", gpus=("A100",), workloads=("W1_V3",), lanes_list=("adaptive",))
    cfg = camp.build_cell_serving_config(record, record["cells"][0], profile_path=str(prof))
    cfg["run_id"] = "bench-a100-run1"
    cfg["source_commit"] = "a" * 40

    module, fake_modal, image, fn_calls, web_server = _load_modal_benchmark_runner(monkeypatch, cfg)
    assert module.APP_NAME == "flexsds-benchmark"
    assert module.MODAL_GPU == "A100-80GB"
    # Verify server decorator got A100-80GB and max_containers=1
    server_fn_kwargs = next(kw for kw in fn_calls if kw.get("gpu") == "A100-80GB")
    assert server_fn_kwargs["max_containers"] == 1
    assert server_fn_kwargs["scaledown_window"] == 60
    web_server.assert_called_once_with(8000, startup_timeout=900)
    # Verify backport patch command and mounted profile file
    run_cmd_args = [call.args[0] for call in image.run_commands.call_args_list]
    assert any("vllm_runtime_patch.py" in str(arg) for arg in run_cmd_args)
    added_files = [call.kwargs.get("remote_path") or (call.args[1] if len(call.args) > 1 else None)
                   for call in image.add_local_file.call_args_list]
    assert "/root/project/hardware_profile.json" in added_files

    # Verify harness CLI arguments use modal-a100-v1, m3-fixed3, and 600s timeout for W1_V3
    harness_args = module.build_harness_args(
        lane="adaptive",
        gpu="A100",
        base_url="https://any-workspace--flexsds-benchmark-serve.modal.run/v1",
        served_model_name="flexsds-bench-adaptive",
        server_env_json="/artifacts/server_env.json",
        raw_out="/artifacts/raw.jsonl",
        summary_csv="/artifacts/h.csv",
        experiment_id="bench-a100-run1",
        workload="W1_V3",
        concurrency=32,
        repeats=5,
        num_requests=32,
    )
    assert harness_args[harness_args.index("--environment-profile") + 1] == "modal-a100-v1"
    assert harness_args[harness_args.index("--warmup-policy") + 1] == "m3-fixed3"
    assert harness_args[harness_args.index("--timeout-s") + 1] == "600"


def test_modal_benchmark_runner_submit_and_collect_roundtrip(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("bench-t4", "a" * 40, mode="fast", gpus=("T4",), workloads=("W2",), lanes_list=("vanilla",))
    cfg = camp.build_cell_serving_config(record, record["cells"][0])
    cfg["run_id"] = "bench-t4-run1"
    cfg["source_commit"] = "a" * 40

    module, fake_modal, _, _, _ = _load_modal_benchmark_runner(monkeypatch, cfg)
    monkeypatch.setattr(module, "_committed_bench_source", lambda: "a" * 40)

    identity_fn = Mock()
    identity_fn.remote.return_value = {
        "commit": "a" * 40,
        "run_id": "bench-t4-run1",
        "gpu": "T4",
        "modal_gpu": "T4",
        "lane": "vanilla",
        "max_model_len": 4096,
        "vllm_runtime": "backport-55708",
    }
    bench_fn = Mock()
    bench_fn.spawn.return_value = Mock(object_id="fc-bench123")

    def fake_from_name(app_name, fn_name):
        assert app_name == "flexsds-benchmark"
        return identity_fn if fn_name == "benchmark_identity" else bench_fn

    fake_modal.Function.from_name.side_effect = fake_from_name

    evidence_root = tmp_path / "attempts"
    call_id = module.bench_submit(
        gpu="T4",
        lane="vanilla",
        workload="W2",
        concurrency=8,
        repeats=5,
        num_requests=32,
        server_run_id="bench-t4-run1",
        bench_run_id="bench-t4-run1",
        evidence_root=evidence_root,
        serving_config=cfg,
    )
    assert call_id == "fc-bench123"
    receipt = json.loads((evidence_root / "bench-t4-run1" / "submission.json").read_text(encoding="utf-8"))
    assert receipt["call_id"] == "fc-bench123"
    assert receipt["gpu"] == "T4"

    raw_line = json.dumps({
        "experiment_id": "bench-t4-run1",
        "lane": "vanilla",
        "commit": "a" * 40,
        "workload": "W2",
        "concurrency": 8,
    }) + "\n"
    fake_result = {
        "gpu": "T4",
        "lane": "vanilla",
        "server_run_id": "bench-t4-run1",
        "bench_run_id": "bench-t4-run1",
        "cell": {"workload": "W2", "concurrency": 8, "repeats": 5, "num_requests": 32},
        "commit": "a" * 40,
        "call_id": "fc-bench123",
        "returncode": 0,
        "server_ready_wait_s": 42.5,
        "join_error": None,
        "server_instance_unchanged": True,
        "endpoint_model_ids": ["flexsds-bench-vanilla"],
        "server_env": {"lane": "vanilla", "run_id": "bench-t4-run1", "gpu": "Tesla T4"},
        "harness_log": "ok\n",
        "raw": raw_line,
        "joined": raw_line,
        "warmup": raw_line,
        "timestamp_utc": "2026-11-25T16:00:00+00:00",
    }
    fake_modal.FunctionCall.from_id.return_value = Mock(get=Mock(return_value=fake_result))
    out = module.bench_collect("fc-bench123", evidence_root=evidence_root)
    assert (out / "bench.json").is_file()
    assert (out / "raw.vram.jsonl").is_file()
    assert (out / "server_env.json").is_file()



