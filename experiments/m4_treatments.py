"""M4-02-06 treatments. Historical M2/M3 lane labels keep their meaning.

No serving framework is imported here. Dynamic SD is plumbing driven by the
project scheduler, not a replacement for the project controller.
"""
from __future__ import annotations

import json
from pathlib import Path

TARGET = "Qwen/Qwen2.5-7B-Instruct-AWQ"
TARGET_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
SPECULATOR = "duckhoi/qwen2.5-7b-eagle3.1-speculator"
SPECULATOR_REVISION = "c3adca7e211b471d97a519a625df7ffd37b8b461"
VLLM_VERSION = "0.30.0"
TREATMENTS = ("vanilla", "eagle_fixed", "eagle_adaptive")
SCHEDULER = "src.server.adaptive_scheduler.AdaptiveAsyncScheduler"
EXECUTOR = "src.server.lineage_executor.LineageUniProcExecutor"


def definition(name: str) -> dict:
    if name not in TREATMENTS:
        raise ValueError(f"Not an M4 treatment: {name}")
    spec = None
    if name != "vanilla":
        spec = dict(method="eagle3", model=SPECULATOR, revision=SPECULATOR_REVISION,
                    num_speculative_tokens=3 if name == "eagle_fixed" else 4,
                    draft_sample_method="greedy")
        if name == "eagle_adaptive":
            spec["num_speculative_tokens_per_batch_size"] = [[1, 32, 4]]
    return dict(treatment=name, target_model=TARGET, target_revision=TARGET_REVISION,
                speculative_config=spec, policy="project_adaptive" if name == "eagle_adaptive"
                else "fixed" if spec else "off", served_model_name=f"m4-{name}")


def engine_kwargs(name: str) -> dict:
    config = definition(name)
    result = dict(model=TARGET, revision=TARGET_REVISION, dtype="bfloat16",
                  quantization="awq_marlin", generation_config="vllm",
                  max_model_len=4096, max_num_seqs=32, gpu_memory_utilization=0.85,
                  async_scheduling=True)
    if config["speculative_config"]:
        result.update(speculative_config=config["speculative_config"],
                      per_request_spec_decode_metrics="summary")
    if name == "eagle_adaptive":
        result.update(scheduler_cls=SCHEDULER, distributed_executor_backend=EXECUTOR)
    return result


def serve_args(name: str, port: int = 8000) -> list[str]:
    config = definition(name)
    args = [TARGET, "--revision", TARGET_REVISION, "--host", "127.0.0.1",
            "--port", str(port), "--served-model-name", config["served_model_name"],
            "--dtype", "bfloat16", "--quantization", "awq_marlin",
            "--generation-config", "vllm", "--max-model-len", "4096",
            "--max-num-seqs", "32", "--gpu-memory-utilization", "0.85",
            "--async-scheduling"]
    if config["speculative_config"]:
        args += ["--speculative-config", json.dumps(config["speculative_config"], sort_keys=True),
                 "--per-request-spec-decode-metrics", "summary"]
    if name == "eagle_adaptive":
        args += ["--scheduler-cls", SCHEDULER, "--distributed-executor-backend", EXECUTOR]
    return args


def runtime_env(name: str, directory: Path) -> dict[str, str]:
    definition(name)
    env = {"VLLM_USE_V2_MODEL_RUNNER": "0"}
    if name == "eagle_adaptive":
        env.update(ADAPTIVE_MAX_K="4", ADAPTIVE_MIN_BATCH_K="0",
                   ADAPTIVE_PER_REQUEST="1", ADAPTIVE_ENABLE_VRAM_GUARD="1",
                   ADAPTIVE_STRICT="1", ADAPTIVE_CONTROLLER_KWARGS='{"history_limit":4096}',
                   ADAPTIVE_TELEMETRY_PATH=str(directory / "requests.jsonl"),
                   ADAPTIVE_STEP_TELEMETRY_PATH=str(directory / "steps.jsonl"),
                   ADAPTIVE_STARTUP_TELEMETRY_PATH=str(directory / "lineage.jsonl"))
    return env
