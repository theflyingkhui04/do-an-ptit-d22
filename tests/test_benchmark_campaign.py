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

def _only_a10_packaged(tmp_path, monkeypatch):
    """Packaged profiles as before 27/11 (A10 only), so a missing-profile path can still be tested."""
    import shutil
    from flexsds.core import cost_model as cm
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    for path in cm.PROFILES_DIR.glob("*.json"):
        if "remote" not in path.name or "a10g" in path.name:
            shutil.copy(path, packaged / path.name)
    monkeypatch.setattr(cm, "PROFILES_DIR", packaged)


def test_benchmark_campaign_full_and_fast_matrix_counts_and_protocol(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    from flexsds.core import cost_model as cm

    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path / "empty_user_profiles")
    _only_a10_packaged(tmp_path, monkeypatch)

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
    _only_a10_packaged(tmp_path, monkeypatch)
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


def _make_valid_campaign(tmp_path, monkeypatch, *, campaign_id, gpus, concurrency, tpot_map, gpu_device,
                         lanes_list=("vanilla", "static", "adaptive")):
    """Write one complete, valid attempt per cell in the one-container-per-GPU evidence layout."""
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
    record = camp.plan(campaign_id, commit, mode="fast", gpus=gpus, workloads=("W2",), lanes_list=lanes_list)
    record["concurrencies"] = [concurrency]
    record["cells"] = [c for c in record["cells"] if c["concurrency"] == concurrency]
    plan_root = tmp_path / "runs"
    monkeypatch.setattr(camp, "ROOT", tmp_path)
    camp.save_plan(record, plan_root=plan_root)
    prof = tmp_path / "prof.json"
    prof.write_text(json.dumps({"models": {"target_revision": "a" * 40, "draft_revision": "b" * 40},
                                "metadata": {"raw_sha256": "d" * 64}}), encoding="utf-8")
    instances: dict[tuple[str, str], str] = {}
    for idx, cell in enumerate(record["cells"]):
        gpu, lane = cell["gpu"], cell["lane"]
        attempt_id = f"{camp.attempt_prefix(record, cell)}001"
        out = plan_root / record["campaign_id"] / "attempts" / attempt_id
        out.mkdir(parents=True)
        cfg = camp.build_cell_serving_config(record, cell, profile_path=str(prof) if lane == "adaptive" else None)
        instance = instances.setdefault((gpu, lane), f"boot-{gpu}-{lane}")
        server_run_id = f"srv-{gpu}-{lane}"
        receipt = {"gpu": gpu, "lane": lane, "server_run_id": server_run_id, "bench_run_id": attempt_id,
                   "call_id": f"fc-{gpu}", "commit": commit,
                   "cell": {"workload": "W2", "concurrency": concurrency, "repeats": 5, "num_requests": 32}}
        server_env = {
            "gpu": gpu_device[gpu], "gpu_uuid": f"GPU-{gpu}", "requested_gpu": gpu,
            "modal_gpu": lanes.modal_gpu_resource(gpu), "driver_version": "610.57.04",
            "gpu_memory_total_mb": 23028.0, "torch_version": "2.13.0+cu130", "torch_cuda_version": "13.0",
            "vllm_version": "0.30.0", "dtype": cfg["dtype"], "quantization": cfg["quantization"], "lane": lane,
            "run_id": server_run_id, "instance_id": instance, "source_commit": commit,
            "target_model": record["target_model"], "draft_model": cfg["draft_model"],
            "served_model_name": cfg["served_model_name"], "max_model_len": cfg["max_model_len"],
            "runtime_variant": patch.RUNTIME_LABEL, "vllm_outputs_sha256": patch.PATCHED_OUTPUTS_SHA256,
            "vllm_args": LocalServingRunner().build_command(cfg, port=8000),
            "profile_sha256": ("d" * 64) if lane == "adaptive" else None,
        }
        if lane == "adaptive":
            receipt["expected_profile_sha256"] = "d" * 64
        raw, warm, samples = [], [], []
        for repeat in range(6):
            phase = "warmup" if repeat == 0 else "measured"
            offset = ((repeat - 1) * (32 % len(corpus))) % len(corpus) if repeat else 0
            start, end = f"2026-11-25T00:0{repeat}:00+00:00", f"2026-11-25T00:0{repeat}:40+00:00"
            if repeat:
                samples.append({"ts_utc": f"2026-11-25T00:0{repeat}:20+00:00", "running": float(concurrency),
                                "waiting": 0.0, "used_mb": 9000.0})
            for req_idx in range(3 if repeat == 0 else 32):
                sample_index = (req_idx + offset) % len(corpus)
                req = corpus[sample_index]["request"]
                row = {
                    "experiment_id": attempt_id, "lane": lane, "workload": "W2", "concurrency": concurrency,
                    "phase": phase, "repeat_id": f"r{repeat:03}" if repeat else None,
                    "request_id": f"{attempt_id}-{repeat}-{req_idx}", "request_index": req_idx, "commit": commit,
                    "server_run_id": server_run_id, "server_instance_id": instance, "sample_policy": "cycle-rotate",
                    "sample_id": corpus[sample_index]["sample_id"], "sample_index": sample_index,
                    "sample_offset": offset, "canonical_sample_count": len(corpus), "success": True,
                    "workload_file_sha256": integrity.actual_sha256, "metrics_complete": True, "http_status": 200,
                    "model": cfg["served_model_name"], "temperature": req["temperature"], "seed": req["seed"],
                    "max_completion_tokens": req["max_tokens"], "response_format": req.get("response_format"),
                    "timestamp_utc": start, "e2e_ms": 1000.0, "ttft_ms": 12.0, "tpot_ms": tpot_map[(gpu, lane)],
                    "output_tokens": 50, "gpu": gpu_device[gpu], "run_id": f"{attempt_id}-run-{repeat}",
                    "run_started_at_utc": start, "run_finished_at_utc": end, "run_duration_s": 32.0,
                    "run_vram_peak_mb": 9000.0,
                }
                if lane != "vanilla":
                    row["server_metrics"] = {"speculative_decoding": {
                        "num_spec_steps": 5, "num_draft_tokens": 15, "num_accepted_draft_tokens": 9,
                        "num_spec_tokens": 3}}
                (raw if repeat else warm).append(row)
        (out / "submission.json").write_text(json.dumps(receipt), encoding="utf-8")
        (out / "bench.json").write_text(json.dumps({**receipt, "returncode": 0, "server_instance_unchanged": True,
                                                    "endpoint_model_ids": [cfg["served_model_name"]]}),
                                        encoding="utf-8")
        (out / "server_env.json").write_text(json.dumps(server_env), encoding="utf-8")
        raw_text = "".join(json.dumps(r) + "\n" for r in raw)
        (out / "raw.jsonl").write_text(raw_text, encoding="utf-8")
        (out / "raw.vram.jsonl").write_text(raw_text, encoding="utf-8")
        (out / "warmup.jsonl").write_text("".join(json.dumps(r) + "\n" for r in warm), encoding="utf-8")
        (out / "concurrency.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples), encoding="utf-8")
    monkeypatch.setattr(camp, "REPO_ROOT", Path(__file__).resolve().parents[1])
    return record, plan_root


def test_aggregate_computes_speedup_per_gpu_independently(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    tpot_map = {
        ("T4", "vanilla"): 40.0,
        ("T4", "static"): 25.0,
        ("T4", "adaptive"): 20.0,
        ("A100", "vanilla"): 10.0,
        ("A100", "static"): 8.0,
        ("A100", "adaptive"): 5.0,
    }
    gpu_device = {"T4": "Tesla T4", "A100": "NVIDIA A100-SXM4-80GB"}
    record, plan_root = _make_valid_campaign(tmp_path, monkeypatch, campaign_id="cpu-multi-gpu", gpus=("T4", "A100"),
                                             concurrency=1, tpot_map=tpot_map, gpu_device=gpu_device)

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
def test_flexsds_benchmark_cli_full_and_fast_dry_run_without_modal(monkeypatch, capsys):
    from flexsds.cli import main as cli
    from flexsds.core.hardware import HardwareInfo

    monkeypatch.setattr(cli, "detect_hardware", lambda: HardwareInfo("CPU", "cpu", 0, 0, "N/A", False))
    monkeypatch.setattr(cli, "print_banner", lambda hw: None)

    assert cli.main(["benchmark", "plan", "--mode", "full", "--dry-run", "--execution-sha", "a" * 40]) == 0
    full_out = capsys.readouterr().out
    assert "216 cells" in full_out
    assert "T4, A10, L40S, A100" in full_out
    assert "W1_V3, W2, W3" in full_out
    assert "vanilla, static, adaptive" in full_out
    assert "[1, 2, 4, 8, 16, 32]" in full_out

    assert cli.main(["benchmark", "--mode", "fast", "--dry-run", "--json", "--execution-sha", "a" * 40]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "fast"
    assert payload["concurrencies"] == [1, 8, 32]
    assert payload["total_cells"] == 108
    assert payload["gpus"] == ["T4", "A10", "L40S", "A100"]
    assert payload["workloads"] == ["W1_V3", "W2", "W3"]
    assert payload["lanes"] == ["vanilla", "static", "adaptive"]


def test_wizard_main_menu_stays_five_items_and_benchmark_lives_under_measure_now(tmp_path, monkeypatch, capsys):
    from flexsds.cli import wizard
    from flexsds.core import cost_model as cm
    from flexsds.core.hardware import HardwareInfo

    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path / "empty")
    seen_main_choices: list[list[str]] = []

    def choose(question, choices, default_idx=0):
        if question == "Select action:":
            seen_main_choices.append(list(choices))
            return "Measure & available profile GPU"
        if question == "Profile action:":
            return "Measure now"
        if question == "Measurement type:":
            assert choices[0].startswith("GPU cost profile")
            assert any("benchmark" in c.lower() for c in choices)
            return choices[1]
        if question == "Benchmark mode:":
            return next(c for c in choices if "Fast benchmark" in c)
        if question == "Benchmark hardware:":
            return next(c for c in choices if "All 4 Modal GPUs" in c)
        if question == "Model pair:":
            return choices[0]
        if question == "Benchmark action:":
            return next(c for c in choices if "Preview plan" in c)
        return choices[default_idx]

    monkeypatch.setattr(wizard, "prompt_choice", choose)
    result = wizard.run_profile_wizard(HardwareInfo("CPU", "cpu", 0, 0, "N/A", False))
    assert result["action"] == "benchmark"
    assert result["mode"] == "fast"
    assert result["total_cells"] == 108
    assert result["dry_run"] is True
    out = capsys.readouterr().out
    assert "108 cells" in out
    assert "T4, A10, L40S, A100" in out

    # Confirm main menu does NOT add a 6th 'Run benchmark' entry.
    def abort_after_main(question, choices, default_idx=0):
        if question == "Select action:":
            seen_main_choices.append(list(choices))
            raise KeyboardInterrupt
        return choices[default_idx]

    monkeypatch.setattr(wizard, "prompt_choice", abort_after_main)
    with pytest.raises(KeyboardInterrupt):
        wizard.run_interactive_wizard(HardwareInfo("CPU", "cpu", 0, 0, "N/A", False))
    assert seen_main_choices[-1] == [
        "Launch serving",
        "Detect hardware",
        "Measure & available profile GPU",
        "Delete Modal token",
        "Running endpoint",
    ]


# --------------------------------------------------------------------------
# Review-Fix Regressions (Critical #1, #2 & Important #1-#6)
# --------------------------------------------------------------------------
def test_flexsds_benchmark_dry_run_json_emits_pure_json_without_banner_or_cuda(monkeypatch, capsys):
    """Important #1: --dry-run --json must not touch detect_hardware or print ASCII banner."""
    from flexsds.cli import main as cli

    monkeypatch.setattr(cli, "detect_hardware", lambda: pytest.fail("benchmark plan/dry-run must not probe local CUDA"))
    assert cli.main(["benchmark", "--mode", "fast", "--dry-run", "--json", "--execution-sha", "a" * 40]) == 0
    raw_out = capsys.readouterr().out
    payload = json.loads(raw_out)
    assert payload["mode"] == "fast" and payload["total_cells"] == 108


def test_flexsds_benchmark_status_requires_campaign_id_and_stop_respects_dry_run(tmp_path, monkeypatch, capsys):
    """Important #3: status/validate/aggregate require --campaign-id and never create orphan plans; stop --dry-run is safe."""
    from flexsds.cli import main as cli
    from experiments import run_benchmark_campaign as camp

    monkeypatch.setattr(camp, "ROOT", tmp_path)
    assert cli.main(["benchmark", "status"]) == 2
    assert cli.main(["benchmark", "validate"]) == 2
    assert cli.main(["benchmark", "aggregate"]) == 2
    assert not (tmp_path / "experiments" / "runs" / "flexsds_bench").exists()

    assert cli.main(["benchmark", "stop", "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out.lower()


def test_adaptive_k0_zero_draft_tokens_accepted_while_static_zero_rejected(tmp_path, monkeypatch):
    """Critical #2: adaptive K=0 cells with 0 drafted tokens must validate and aggregate cleanly."""
    from experiments import run_benchmark_campaign as camp

    tpot = {("A10", "vanilla"): 20.0, ("A10", "static"): 18.0, ("A10", "adaptive"): 19.0}
    record, plan_root = _make_valid_campaign(tmp_path, monkeypatch, campaign_id="k0-adaptive", gpus=("A10",),
                                             concurrency=32, tpot_map=tpot, gpu_device={"A10": "NVIDIA A10"})
    adaptive_cell = next(c for c in record["cells"] if c["lane"] == "adaptive")
    adaptive_dir = camp.attempts(record, adaptive_cell, plan_root=plan_root)[0]
    for name in ("raw.jsonl", "raw.vram.jsonl"):
        rows = [json.loads(l) for l in (adaptive_dir / name).read_text(encoding="utf-8").splitlines()]
        for row in rows:  # genuine K=0 adaptive steps: 0 drafted, 0 accepted
            row["server_metrics"] = {"speculative_decoding": {"num_spec_steps": 0, "num_draft_tokens": 0,
                                                              "num_accepted_draft_tokens": 0, "num_spec_tokens": 0}}
        (adaptive_dir / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    monkeypatch.setattr(camp, "REPO_ROOT", Path(__file__).resolve().parents[1])
    dest = tmp_path / "summary.json"
    rep = tmp_path / "report.md"
    first = camp.aggregate(record, plan_root=plan_root, destination=dest, report_destination=rep)
    # Important #5: re-running aggregate must be idempotent despite wall-clock timestamp.
    second = camp.aggregate(record, plan_root=plan_root, destination=dest, report_destination=rep)
    assert len(first["cells"]) == len(second["cells"]) == 3
    adaptive_row = next(r for r in first["cells"] if r["lane"] == "adaptive")
    assert adaptive_row["alpha"] == "N/A" and adaptive_row["tau"] == "N/A"

    # Important #2: mismatched static_k in vllm_args must fail validate_cell.
    static_cell = next(c for c in record["cells"] if c["lane"] == "static")
    static_dir = camp.attempts(record, static_cell, plan_root=plan_root)[0]
    env_data = json.loads((static_dir / "server_env.json").read_text(encoding="utf-8"))
    spec_idx = env_data["vllm_args"].index("--speculative-config") + 1
    bad_spec = json.loads(env_data["vllm_args"][spec_idx])
    bad_spec["num_speculative_tokens"] = 7
    env_data["vllm_args"][spec_idx] = json.dumps(bad_spec)
    (static_dir / "server_env.json").write_text(json.dumps(env_data), encoding="utf-8")
    with pytest.raises(ValueError, match="vllm_args|static_k|speculative"):
        camp.validate_cell(static_dir, record, static_cell)


def test_modal_benchmark_runner_uses_modal_auth_and_pipx_cli_fallback(monkeypatch):
    """Critical #1: stop_app and SDK bridge use modal_auth.modal_command/prepare_modal_auth."""
    from experiments import modal_benchmark_runner as runner
    from flexsds.server import modal_auth

    calls: list[list[str]] = []
    monkeypatch.setattr(modal_auth, "ensure_modal", lambda: None)
    monkeypatch.setattr(modal_auth, "prepare_modal_auth", lambda mgr: True)
    monkeypatch.setattr(modal_auth, "modal_command", lambda: ["/opt/pipx/bin/modal"])

    def fake_check_output(cmd, **kwargs):
        calls.append(list(cmd))
        return json.dumps([{"app_id": "ap-123", "description": runner.APP_NAME, "state": "deployed"}])

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return Mock(returncode=0)

    monkeypatch.setattr(runner.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.stop_app()
    assert calls[0][:4] == ["/opt/pipx/bin/modal", "app", "list", "--json"]
    assert calls[1][:4] == ["/opt/pipx/bin/modal", "app", "stop", "ap-123"]







# --------------------------------------------------------------------------
# One container per GPU (openspec/changes/flexsds-bench-one-container-per-gpu)
# --------------------------------------------------------------------------
def test_all_lanes_run_eager_async_without_prefix_cache(tmp_path):
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("bench-parity", "a" * 40, mode="fast", gpus="A10", workloads="W2")
    assert record["lifecycle"] == "one-container-per-gpu"
    assert record["match_adaptive_runtime"] is True
    assert record["disable_prefix_caching"] is True
    profile = tmp_path / "p.json"
    profile.write_text("{}", encoding="utf-8")
    for cell in (c for c in record["cells"] if c["concurrency"] == 8):
        cfg = camp.build_cell_serving_config(
            record, cell, profile_path=str(profile) if cell["lane"] == "adaptive" else None)
        cmd = LocalServingRunner().build_command(cfg, port=8000)
        for flag in ("--enforce-eager", "--async-scheduling", "--no-enable-prefix-caching"):
            assert flag in cmd, (cell["lane"], flag)


def test_benchmark_cli_no_longer_offers_match_adaptive_runtime():
    from flexsds.cli.main import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(["benchmark", "plan", "--match-adaptive-runtime"])
    assert exc.value.code == 2
    with pytest.raises(TypeError):
        from experiments import run_benchmark_campaign as camp
        camp.plan("x", "a" * 40, mode="fast", match_adaptive_runtime=False)


def _a10_campaign(tmp_path, monkeypatch, concurrency=8):
    tpot = {("A10", "vanilla"): 20.0, ("A10", "static"): 16.0, ("A10", "adaptive"): 17.0}
    return _make_valid_campaign(tmp_path, monkeypatch, campaign_id="a10-check", gpus=("A10",),
                                concurrency=concurrency, tpot_map=tpot, gpu_device={"A10": "NVIDIA A10"})


def _cell_dir(record, plan_root, lane):
    from experiments import run_benchmark_campaign as camp
    cell = next(c for c in record["cells"] if c["lane"] == lane)
    return cell, camp.attempts(record, cell, plan_root=plan_root)[0]


def _edit_json(path, **changes):
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_validate_cell_accepts_the_new_layout(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    record, plan_root = _a10_campaign(tmp_path, monkeypatch)
    assert len(camp.validate_campaign(record, plan_root=plan_root)) == 3


def test_validate_cell_rejects_a10g_server(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    record, plan_root = _a10_campaign(tmp_path, monkeypatch)
    cell, out = _cell_dir(record, plan_root, "vanilla")
    _edit_json(out / "server_env.json", gpu="NVIDIA A10G")
    with pytest.raises(ValueError, match="NVIDIA A10G"):
        camp.validate_cell(out, record, cell)


def test_validate_cell_rejects_serialized_concurrency(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    record, plan_root = _a10_campaign(tmp_path, monkeypatch, concurrency=8)
    cell, out = _cell_dir(record, plan_root, "static")
    rows = [json.loads(l) for l in (out / "concurrency.jsonl").read_text(encoding="utf-8").splitlines()]
    (out / "concurrency.jsonl").write_text("".join(json.dumps({**r, "running": 1.0}) + "\n" for r in rows),
                                           encoding="utf-8")
    with pytest.raises(ValueError, match="concurrency"):
        camp.validate_cell(out, record, cell)


def test_validate_cell_rejects_missing_prefix_cache_flag_and_missing_args(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    record, plan_root = _a10_campaign(tmp_path, monkeypatch)
    cell, out = _cell_dir(record, plan_root, "vanilla")
    env = json.loads((out / "server_env.json").read_text(encoding="utf-8"))
    _edit_json(out / "server_env.json", vllm_args=[a for a in env["vllm_args"] if a != "--no-enable-prefix-caching"])
    with pytest.raises(ValueError, match="--no-enable-prefix-caching"):
        camp.validate_cell(out, record, cell)
    _edit_json(out / "server_env.json", vllm_args=None)
    with pytest.raises(ValueError, match="vllm_args"):
        camp.validate_cell(out, record, cell)


def test_validate_campaign_requires_one_card_per_gpu(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    record, plan_root = _a10_campaign(tmp_path, monkeypatch)
    _, out = _cell_dir(record, plan_root, "adaptive")
    _edit_json(out / "server_env.json", gpu_uuid="GPU-other")
    with pytest.raises(ValueError, match="more than one card"):
        camp.validate_campaign(record, plan_root=plan_root)


def test_validate_campaign_rejects_one_vllm_process_shared_by_two_lanes(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp
    record, plan_root = _a10_campaign(tmp_path, monkeypatch)
    vanilla = json.loads(_cell_dir(record, plan_root, "vanilla")[1].joinpath("server_env.json").read_text(encoding="utf-8"))
    cell, out = _cell_dir(record, plan_root, "static")
    _edit_json(out / "server_env.json", instance_id=vanilla["instance_id"])
    for name in ("raw.jsonl", "raw.vram.jsonl", "warmup.jsonl"):
        rows = [json.loads(l) for l in (out / name).read_text(encoding="utf-8").splitlines()]
        (out / name).write_text("".join(json.dumps({**r, "server_instance_id": vanilla["instance_id"]}) + "\n"
                                        for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="instance"):
        camp.validate_campaign(record, plan_root=plan_root)


def test_drive_resumes_only_missing_cells_and_skips_running_gpu(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("resume1", "a" * 40, mode="fast", gpus=("T4", "A10"), workloads=("W2",))
    plan_root = tmp_path / "runs"
    monkeypatch.setattr(camp, "ROOT", tmp_path)
    t4_keys = [c["key"] for c in record["cells"] if c["gpu"] == "T4"]
    done = set(t4_keys[:5])
    calls_file = plan_root / "resume1" / "gpu_calls.json"
    calls_file.parent.mkdir(parents=True)
    calls_file.write_text(json.dumps({"A10": "fc-a10", "T4": "fc-t4-old"}), encoding="utf-8")
    submitted: list[tuple[str, list[str]]] = []
    deployed: list[bool] = []

    class FakeRunner:
        def committed_bench_source(self, allow_uncommitted=False):
            return "a" * 40, None

        def gpu_call_state(self, call_id):
            return {"fc-a10": "RUNNING", "fc-t4-old": "DONE"}[call_id]

        def collect_gpu(self, campaign_id, gpu, evidence_root):
            return []

        def submit_gpu(self, rec, gpu, keys, profile_text, **kwargs):
            submitted.append((gpu, list(keys)))
            return f"fc-{gpu}-new"

        def deploy_app(self):
            deployed.append(True)

    monkeypatch.setattr(camp, "_runner", lambda: FakeRunner())
    monkeypatch.setattr(camp, "_ensure_modal_login", lambda: None)
    monkeypatch.setattr(camp, "ensure_adaptive_profiles", lambda *a, **k: {
        g: {"profile_path": None, "status": "ready"} for g in ("T4", "A10")})
    monkeypatch.setattr(camp, "selected_attempt", lambda rec, cell, root=None: object() if cell["key"] in done else None)
    camp.drive(record, plan_root=plan_root, wait=False)
    assert submitted == [("T4", [k for k in t4_keys if k not in done])]
    assert deployed == [True]  # the app is deployed once, before the first submission
    assert json.loads(calls_file.read_text(encoding="utf-8")) == {"A10": "fc-a10", "T4": "fc-T4-new"}


def test_plan_overview_prints_cost_estimate_per_gpu():
    from experiments import run_benchmark_campaign as camp
    record = camp.plan("cost", "a" * 40, mode="fast")
    text = camp.format_plan_overview(record)
    assert "Estimated GPU time" in text
    for gpu in ("T4", "A10", "L40S", "A100"):
        assert f"  - {gpu}" in text
    assert "USD" in text
    est = camp.estimate_cost(record)
    assert set(est) == {"T4", "A10", "L40S", "A100"}
    assert all(v["usd"] > 0 and v["hours"] > 0 for v in est.values())


def test_run_asks_before_spending_unless_yes(monkeypatch, capsys):
    from flexsds.cli import benchmark as bench_cli
    from experiments import run_benchmark_campaign as camp
    from types import SimpleNamespace

    drove: list[bool] = []
    monkeypatch.setattr(camp, "drive", lambda record, **kw: drove.append(True))
    monkeypatch.setattr(camp, "resolve_git_head", lambda: "a" * 40)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    args = SimpleNamespace(bench_action="run", dry_run=False, mode="fast", gpus="A10", workloads="W2", lanes="all",
                           campaign_id="ask1", execution_sha="a" * 40, yes=False)
    assert bench_cli.run_benchmark_command(args) == 1
    assert drove == []
    assert "USD" in capsys.readouterr().out
    args.yes = True
    assert bench_cli.run_benchmark_command(args) == 0
    assert drove == [True]


def test_drive_keeps_going_when_one_gpu_cannot_be_submitted(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("partial1", "a" * 40, mode="fast", gpus=("T4", "A10"), workloads=("W2",))
    plan_root = tmp_path / "runs"
    monkeypatch.setattr(camp, "ROOT", tmp_path)
    submitted: list[str] = []

    class FakeRunner:
        def committed_bench_source(self, allow_uncommitted=False):
            return "a" * 40, None

        def deploy_app(self):
            pass

        def gpu_call_state(self, call_id):
            return "DONE"

        def collect_gpu(self, campaign_id, gpu, evidence_root):
            return []

        def submit_gpu(self, rec, gpu, keys, profile_text, **kwargs):
            if gpu == "T4":
                raise RuntimeError("T4: Modal did not hand out the exact card in 12 tries")
            submitted.append(gpu)
            return "fc-a10"

    monkeypatch.setattr(camp, "_runner", lambda: FakeRunner())
    monkeypatch.setattr(camp, "_ensure_modal_login", lambda: None)
    monkeypatch.setattr(camp, "ensure_adaptive_profiles", lambda *a, **k: {
        g: {"profile_path": None, "status": "ready"} for g in ("T4", "A10")})
    with pytest.raises(RuntimeError, match="T4: Modal did not hand out"):
        camp.drive(record, plan_root=plan_root, wait=False)
    assert submitted == ["A10"]
    assert json.loads((plan_root / "partial1" / "gpu_calls.json").read_text(encoding="utf-8")) == {"A10": "fc-a10"}


def test_drive_wait_survives_transient_modal_errors(tmp_path, monkeypatch):
    """A dropped connection while waiting must not end the run: the GPU session keeps going on Modal."""
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("flaky1", "a" * 40, mode="fast", gpus=("A10",), workloads=("W2",))
    plan_root = tmp_path / "runs"
    monkeypatch.setattr(camp, "ROOT", tmp_path)
    calls_file = plan_root / "flaky1" / "gpu_calls.json"
    calls_file.parent.mkdir(parents=True)
    calls_file.write_text(json.dumps({"A10": "fc-a10"}), encoding="utf-8")
    states = iter(["RUNNING", "RUNNING", ConnectionError("Connection lost"), "RUNNING", "DONE"])
    collects: list[int] = []

    class FakeRunner:
        def committed_bench_source(self, allow_uncommitted=False):
            return "a" * 40, None

        def deploy_app(self):
            pass

        def gpu_call_state(self, call_id):
            state = next(states)
            if isinstance(state, Exception):
                raise state
            return state

        def collect_gpu(self, campaign_id, gpu, evidence_root):
            collects.append(1)
            if len(collects) == 2:
                raise OSError("StreamTerminatedError: Connection lost")
            return []

    monkeypatch.setattr(camp, "_runner", lambda: FakeRunner())
    monkeypatch.setattr(camp, "_ensure_modal_login", lambda: None)
    monkeypatch.setattr(camp, "ensure_adaptive_profiles", lambda *a, **k: {"A10": {"profile_path": None}})
    monkeypatch.setattr(camp, "selected_attempt", lambda rec, cell, root=None: None)
    monkeypatch.setattr(camp.time, "sleep", lambda s: None)
    camp.drive(record, plan_root=plan_root, wait=True, poll_s=0)
    assert len(collects) >= 3  # kept collecting after both errors, and once more at the end


def test_drive_wait_gives_up_after_many_consecutive_errors(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("flaky2", "a" * 40, mode="fast", gpus=("A10",), workloads=("W2",))
    plan_root = tmp_path / "runs"
    monkeypatch.setattr(camp, "ROOT", tmp_path)
    calls_file = plan_root / "flaky2" / "gpu_calls.json"
    calls_file.parent.mkdir(parents=True)
    calls_file.write_text(json.dumps({"A10": "fc-a10"}), encoding="utf-8")
    first = [True]

    class FakeRunner:
        def committed_bench_source(self, allow_uncommitted=False):
            return "a" * 40, None

        def gpu_call_state(self, call_id):
            if first[0]:
                first[0] = False
                return "RUNNING"
            raise ConnectionError("Connection lost")

        def collect_gpu(self, campaign_id, gpu, evidence_root):
            return []

    monkeypatch.setattr(camp, "_runner", lambda: FakeRunner())
    monkeypatch.setattr(camp, "_ensure_modal_login", lambda: None)
    monkeypatch.setattr(camp, "ensure_adaptive_profiles", lambda *a, **k: {"A10": {"profile_path": None}})
    monkeypatch.setattr(camp.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="flexsds benchmark status"):
        camp.drive(record, plan_root=plan_root, wait=True, poll_s=0)


def test_status_lists_cells_that_failed_inside_the_container(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("fails1", "a" * 40, mode="fast", gpus=("T4",), workloads=("W2",))
    plan_root = tmp_path / "runs"
    monkeypatch.setattr(camp, "ROOT", tmp_path)
    failures = plan_root / "fails1" / "failures"
    failures.mkdir(parents=True)
    (failures / "T4__static__W2__c8.json").write_text(json.dumps(
        {"status": "failed:server", "error": "RuntimeError: vLLM exited early with code 1"}), encoding="utf-8")

    class FakeRunner:
        def gpu_call_state(self, call_id):
            return "DONE"

        def collect_gpu(self, campaign_id, gpu, evidence_root):
            return []

    monkeypatch.setattr(camp, "_runner", lambda: FakeRunner())
    lines = camp.format_campaign_status(record, plan_root=plan_root)
    assert lines[0].startswith("T4: 0/9 cells valid")
    assert any("T4__static__W2__c8" in l and "failed:server" in l and "vLLM exited early" in l for l in lines)


# --------------------------------------------------------------------------
# Known vLLM 0.30 counter artifact: grammar post-stop over-count (A10 W1_V3 c32)
# --------------------------------------------------------------------------
_ARTIFACT_ID = "req-05a67192-4f84-44a0-b9d7-ad4f062add0f"


def _spec_row(request_id, *, accepted, drafted, steps, k, histogram, response_format=None, output_tokens=100):
    return {"request_id": request_id, "repeat_id": "r004", "sample_id": "m0_04_006", "output_tokens": output_tokens,
            "max_completion_tokens": 3072,
            "response_format": {"type": "json_object"} if response_format is None else response_format,
            "server_metrics": {"speculative_decoding": {
                "num_accepted_draft_tokens": accepted, "num_draft_tokens": drafted, "num_spec_steps": steps,
                "num_spec_tokens": k, "acceptance_histogram": histogram}}}


def _artifact_row(**overrides):
    return _spec_row(_ARTIFACT_ID, accepted=4, drafted=3, steps=4, k=4, histogram=[1, 2, 1, 0, 0],
                     output_tokens=1073, **overrides)


def _normal_rows(n=19):
    return [_spec_row(f"req-n{i}", accepted=9, drafted=15, steps=5, k=3, histogram=[1, 1, 1, 2]) for i in range(n)]


@pytest.mark.parametrize("lane", ["static", "adaptive"])
def test_summarize_excludes_known_artifact_records(lane):
    from experiments import run_benchmark_campaign as camp

    alpha, tau, excluded = camp.summarize_speculative_acceptance(_normal_rows() + [_artifact_row()], lane)
    assert alpha == pytest.approx(9 / 15)
    assert tau == pytest.approx(1 + 9 / 5)
    assert [e["request_id"] for e in excluded] == [_ARTIFACT_ID]
    assert excluded[0]["reason"] == "grammar_post_stop_overcount"


def test_summarize_artifact_only_adaptive_cell_is_na():
    from experiments import run_benchmark_campaign as camp

    zero = [_spec_row(f"req-z{i}", accepted=0, drafted=0, steps=0, k=4, histogram=[0] * 5) for i in range(19)]
    alpha, tau, excluded = camp.summarize_speculative_acceptance(zero + [_artifact_row()], "adaptive")
    assert (alpha, tau) == (None, None)
    assert len(excluded) == 1


def test_summarize_vanilla_has_no_acceptance():
    from experiments import run_benchmark_campaign as camp

    assert camp.summarize_speculative_acceptance([{"server_metrics": None}], "vanilla") == (None, None, [])


def test_summarize_still_rejects_unknown_overcount():
    from experiments import run_benchmark_campaign as camp

    with pytest.raises(ValueError, match="C3"):
        camp.summarize_speculative_acceptance(_normal_rows() + [_artifact_row(response_format={"type": "text"})],
                                              "adaptive")


def _t4_campaign(tmp_path, monkeypatch, campaign_id):
    tpot_map = {("T4", "vanilla"): 40.0, ("T4", "static"): 25.0, ("T4", "adaptive"): 20.0}
    return _make_valid_campaign(tmp_path, monkeypatch, campaign_id=campaign_id, gpus=("T4",), concurrency=1,
                                tpot_map=tpot_map, gpu_device={"T4": "Tesla T4"})


def test_aggregate_without_exclusions_has_no_new_key(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record, plan_root = _t4_campaign(tmp_path, monkeypatch, "cpu-no-exclusions")
    result = camp.aggregate(record, plan_root=plan_root, destination=tmp_path / "s.json",
                            report_destination=tmp_path / "r.md")
    assert "speculative_counter_exclusions" not in result
    assert "speculative_counter_exclusions" not in json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert "Ghi chú bộ đếm speculative" not in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_aggregate_lists_exclusions_in_summary_and_report(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record, plan_root = _t4_campaign(tmp_path, monkeypatch, "cpu-exclusions")
    original = camp.summarize_speculative_acceptance
    item = {"request_id": _ARTIFACT_ID, "repeat_id": "r004", "sample_id": "m0_04_006",
            "reason": "grammar_post_stop_overcount", "num_accepted_draft_tokens": 4, "num_draft_tokens": 3,
            "num_spec_tokens": 4}

    def with_artifact(raw, lane):
        alpha, tau, excluded = original(raw, lane)
        return alpha, tau, excluded + ([item] if lane == "adaptive" else [])

    monkeypatch.setattr(camp, "summarize_speculative_acceptance", with_artifact)
    result = camp.aggregate(record, plan_root=plan_root, destination=tmp_path / "s.json",
                            report_destination=tmp_path / "r.md")
    assert result["speculative_counter_exclusions"] == {"T4__adaptive__W2__c1": [item]}
    assert result["scientific_status"] == "VALID"
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## Ghi chú bộ đếm speculative" in md
    assert "T4__adaptive__W2__c1" in md and "grammar_post_stop_overcount" in md


def test_plan_overview_says_when_a_profile_stops_below_the_largest_batch():
    from experiments import run_benchmark_campaign as camp

    record = camp.plan("cpu-readiness", "a" * 40, mode="fast", gpus=("T4", "A10"), workloads=("W2",))
    record["profile_readiness"] = {
        "T4": {"status": "ready", "profile_name": "t4.json", "max_batch": 8},
        "A10": {"status": "ready", "profile_name": "a10.json", "max_batch": 32},
    }
    text = camp.format_plan_overview(record)
    assert "T4   [ready] t4.json (measured B<=8; larger batches use B=8 costs)" in text
    assert "A10  [ready] a10.json\n" in text + "\n"


def _stop_fixture(tmp_path, monkeypatch, calls):
    from experiments import modal_benchmark_runner as runner
    from experiments import run_benchmark_campaign as camp

    monkeypatch.setattr(camp, "ROOT", tmp_path)
    record = camp.plan("cpu-stop", "a" * 40, mode="fast", gpus=("T4",), workloads=("W2",))
    camp.save_plan(record)
    camp._save_calls(record, camp.PLAN_ROOT, calls)
    seen: dict[str, list] = {"cancel": [], "stop_app": [], "release": []}
    monkeypatch.setattr(runner, "cancel_call", lambda call_id: seen["cancel"].append(call_id))
    monkeypatch.setattr(runner, "stop_app", lambda: seen["stop_app"].append(True))
    monkeypatch.setattr(runner, "release_locks", lambda cid: seen["release"].append(cid))
    return seen


def test_benchmark_stop_with_campaign_cancels_only_that_campaign(tmp_path, monkeypatch, capsys):
    # One deployed app serves every campaign: stopping it for one campaign killed a parallel A100 run (27/11).
    from flexsds.cli import main as cli

    seen = _stop_fixture(tmp_path, monkeypatch, {"T4": "fc-t4"})
    assert cli.main(["benchmark", "stop", "--campaign-id", "cpu-stop"]) == 0
    assert seen == {"cancel": ["fc-t4"], "stop_app": [], "release": ["cpu-stop"]}
    out = capsys.readouterr().out
    assert "cpu-stop" in out and "other campaigns keep running" in out


def test_benchmark_stop_without_campaign_stops_the_whole_app(tmp_path, monkeypatch, capsys):
    from flexsds.cli import main as cli

    seen = _stop_fixture(tmp_path, monkeypatch, {"T4": "fc-t4"})
    assert cli.main(["benchmark", "stop"]) == 0
    assert seen["stop_app"] == [True] and seen["cancel"] == []
    assert "every campaign" in capsys.readouterr().out


def test_benchmark_stop_dry_run_with_campaign_names_the_calls(tmp_path, monkeypatch, capsys):
    from flexsds.cli import main as cli

    seen = _stop_fixture(tmp_path, monkeypatch, {"T4": "fc-t4"})
    assert cli.main(["benchmark", "stop", "--campaign-id", "cpu-stop", "--dry-run"]) == 0
    assert seen == {"cancel": [], "stop_app": [], "release": []}
    assert "fc-t4" in capsys.readouterr().out


def test_cancel_call_cancels_the_modal_function_call(monkeypatch):
    from experiments import modal_benchmark_runner as runner

    cancelled = []

    class FakeCall:
        def cancel(self):
            cancelled.append(True)

    fake_modal = Mock()
    fake_modal.FunctionCall.from_id = lambda call_id: (cancelled.append(call_id), FakeCall())[1]
    monkeypatch.setattr(runner, "modal", fake_modal)
    runner.cancel_call("fc-abc")
    assert cancelled == ["fc-abc", True]


# --------------------------------------------------------------------------
# Retrying failed cells (A100, 27/11): mark invalid attempts, resubmit, report
# --------------------------------------------------------------------------
def _break_attempt(path):
    """Make the first measured request of an attempt fail, as a dropped connection does."""
    for name in ("raw.jsonl", "raw.vram.jsonl"):
        rows = [json.loads(l) for l in (path / name).read_text(encoding="utf-8").splitlines()]
        rows[0].update(success=False, metrics_complete=False, error_type="ServerDisconnectedError",
                       error_message="Server disconnected")
        (path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_mark_invalid_attempts_writes_the_m3_marker_once(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record, plan_root = _t4_campaign(tmp_path, monkeypatch, "cpu-mark")
    bad_cell = next(c for c in record["cells"] if c["key"] == "T4__static__W2__c1")
    bad = camp.attempts(record, bad_cell, plan_root)[0]
    _break_attempt(bad)
    assert camp.mark_invalid_attempts(record, plan_root) == [bad.name]
    marker = json.loads((bad / "validation_failure.json").read_text(encoding="utf-8"))
    assert marker["status"] == "INVALID-PROTOCOL" and marker["error"]
    assert camp.mark_invalid_attempts(record, plan_root) == []  # idempotent
    assert camp.selected_attempt(record, bad_cell, plan_root) is None
    others = [c for c in record["cells"] if c["key"] != bad_cell["key"]]
    assert all(camp.selected_attempt(record, c, plan_root) is not None for c in others)


def test_drive_resubmits_a_cell_whose_only_attempt_failed(tmp_path, monkeypatch):
    from experiments import run_benchmark_campaign as camp

    record, plan_root = _t4_campaign(tmp_path, monkeypatch, "cpu-retry")
    bad_cell = next(c for c in record["cells"] if c["key"] == "T4__static__W2__c1")
    _break_attempt(camp.attempts(record, bad_cell, plan_root)[0])
    camp._save_calls(record, plan_root, {"T4": "fc-old"})
    submitted = []

    class FakeRunner:
        def committed_bench_source(self, allow_uncommitted=False):
            return record["commit"], None

        def gpu_call_state(self, call_id):
            return "DONE"

        def collect_gpu(self, campaign_id, gpu, evidence_root):
            return []

        def deploy_app(self):
            pass

        def submit_gpu(self, rec, gpu, keys, profile_text, **kwargs):
            submitted.append((gpu, list(keys), kwargs.get("attempts")))
            return "fc-new"

    monkeypatch.setattr(camp, "_runner", lambda: FakeRunner())
    monkeypatch.setattr(camp, "_ensure_modal_login", lambda: None)
    monkeypatch.setattr(camp, "ensure_adaptive_profiles", lambda *a, **k: {"T4": {"profile_path": None}})
    camp.drive(record, plan_root=plan_root, wait=False)
    assert submitted == [("T4", ["T4__static__W2__c1"], {"T4__static__W2__c1": 2})]


def test_status_hides_a_failed_attempt_once_the_cell_has_a_valid_one(tmp_path, monkeypatch):
    import shutil
    from experiments import run_benchmark_campaign as camp

    record, plan_root = _t4_campaign(tmp_path, monkeypatch, "cpu-status")
    cell = next(c for c in record["cells"] if c["key"] == "T4__static__W2__c1")
    first = camp.attempts(record, cell, plan_root)[0]
    second = first.with_name(first.name[:-3] + "002")
    shutil.copytree(first, second)  # the valid retry
    for name in ("bench.json", "submission.json"):
        data = json.loads((second / name).read_text(encoding="utf-8"))
        data["bench_run_id"] = second.name
        (second / name).write_text(json.dumps(data), encoding="utf-8")
    for name in ("raw.jsonl", "raw.vram.jsonl", "warmup.jsonl"):
        rows = [json.loads(l) for l in (second / name).read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["experiment_id"] = second.name
        (second / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _break_attempt(first)
    failures = plan_root / "cpu-status" / "failures"
    failures.mkdir(parents=True)
    (failures / "T4__vanilla__W2__c8.a002.json").write_text(json.dumps(
        {"status": "failed:server", "error": "vLLM exited early"}), encoding="utf-8")

    class FakeRunner:
        def gpu_call_state(self, call_id):
            return "DONE"

        def collect_gpu(self, campaign_id, gpu, evidence_root):
            return []

    monkeypatch.setattr(camp, "_runner", lambda: FakeRunner())
    lines = camp.format_campaign_status(record, plan_root=plan_root)
    assert (first / "validation_failure.json").exists()  # status marked the failed a001
    assert camp.selected_attempt(record, cell, plan_root) == second
    assert not any("T4__static__W2__c1" in l for l in lines[1:])
    assert any(l.strip().startswith("T4__vanilla__W2__c8.a002") and "failed:server" in l for l in lines)
