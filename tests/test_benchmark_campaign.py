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
