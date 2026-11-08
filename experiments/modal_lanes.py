"""Lane configuration for the Modal L4 vLLM runner ([M3-00]).

Pure Python, no ``modal`` import, so it can be unit-tested anywhere.
``scripts/launch_lane{1,2,3}_*.sh`` remain the source of truth for model,
serving limits and speculative config; this module mirrors them and only
changes dtype/quantization through a named profile
(tests/test_modal_runner_config.py keeps both in sync).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


# Same runtime as the canonical M1/M2 campaigns (experiments/benchmark_harness.py);
# the lane 3 scheduler subclasses vllm.v1 AsyncScheduler from this release.
VLLM_VERSION = "0.30.0"

APP_NAME = "ptit-vllm"
# [M5-01 web demo] Same runner deployed as another Modal application (for example
# ptit-vllm-demo), so benchmark deployments never replace the demo lanes.
_APP_NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?")


def resolve_app_name(environ: Mapping[str, str]) -> str:
    """``PTIT_APP_NAME`` (lowercase letters, digits, dashes); unset means ``ptit-vllm``."""
    name = environ.get("PTIT_APP_NAME") or APP_NAME
    if not _APP_NAME_RE.fullmatch(name):
        raise ValueError(f"PTIT_APP_NAME must match [a-z0-9-] (max 40, no leading/trailing dash), got {name!r}")
    return name

TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EAGLE_DRAFT_MODEL = "duckhoi/qwen2.5-7b-eagle3.1-speculator"
EAGLE_SERVED_MODEL = "qwen2.5-7b-eagle3-draft"

# HuggingFace commit hashes pinned from the first download into ptit-model-cache
# (experiments/runs/m3_00/20261024-m300-setup/weights_manifest.json).
TARGET_REVISION: str | None = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
DRAFT_REVISION: str | None = "7ae557604adf67be50417f59c2c2f167def9a775"
# M3-04 final export: step 15000, HF release v0.3.0-epoch3 (M3-04 worklog §4).
EAGLE_REVISION = "c3adca7e211b471d97a519a625df7ffd37b8b461"

REMOTE_PROJECT_ROOT = "/root/project"
ARTIFACTS_ROOT = "/artifacts"
HF_HOME = "/root/.cache/huggingface"
VLLM_CACHE_ROOT = "/root/.cache/vllm"
DEFAULT_PORT = 8000

DTYPE_PROFILES: dict[str, dict[str, str]] = {
    # L4 (sm_89): native BF16 with the Marlin AWQ kernel.
    "l4-bf16": {"dtype": "bfloat16", "quantization": "awq_marlin"},
    # Same kernel/dtype as the T4 launchers.
    "fp16-fallback": {"dtype": "half", "quantization": "awq"},
}
DEFAULT_PROFILE = "l4-bf16"

# L4 is the reference GPU for M3 measurements. A10G (24 GB, sm_86, BF16) is a
# fallback for functional runs when Modal has no L4 capacity; T4 needs
# PTIT_DTYPE_PROFILE=fp16-fallback. Latency from a non-L4 GPU is not
# comparable with L4 results.
DEFAULT_GPU = "L4"
SUPPORTED_GPUS = ("L4", "A10G", "T4")


def resolve_gpu(environ: Mapping[str, str]) -> str:
    gpu = environ.get("PTIT_GPU") or DEFAULT_GPU
    if gpu not in SUPPORTED_GPUS:
        raise ValueError(f"PTIT_GPU must be one of {SUPPORTED_GPUS}, got {gpu!r}")
    return gpu

COMMON_SERVING_ARGS = [
    "--generation-config", "vllm",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "4096",
    "--max-num-seqs", "32",
]

# [M5-01] W1_v2 (full-receipt OCR text, max_tokens 3072) needs a longer context.
# The default stays 4096 so M2/M3 campaigns, protocol B and the profiler keep
# their serving contract; long-context servers are deployed explicitly.
DEFAULT_MAX_MODEL_LEN = 4096
SUPPORTED_MAX_MODEL_LENS = (4096, 8192)


def resolve_max_model_len(environ: Mapping[str, str]) -> int:
    """``PTIT_MAX_MODEL_LEN`` (4096 or 8192); unset means the default 4096."""
    raw = environ.get("PTIT_MAX_MODEL_LEN")
    if raw is None:
        return DEFAULT_MAX_MODEL_LEN
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"PTIT_MAX_MODEL_LEN must be one of {SUPPORTED_MAX_MODEL_LENS}, got {raw!r}") from None
    if value not in SUPPORTED_MAX_MODEL_LENS:
        raise ValueError(f"PTIT_MAX_MODEL_LEN must be one of {SUPPORTED_MAX_MODEL_LENS}, got {raw!r}")
    return value


# [M5-01] vLLM runtime of the image: stock 0.30.0 (default) or the minimal
# backport of upstream PR #55708 that the M3-05 Colab campaign used
# (experiments/vllm_runtime_patch.py). The patch only preserves speculative
# metrics when streamed outputs are coalesced.
VLLM_RUNTIMES = ("stock", "backport-55708")
# server_env.json ``runtime_variant``; the patched label equals the M3-05 evidence.
VLLM_RUNTIME_LABELS = {"stock": "vllm-0.30.0", "backport-55708": "vllm-0.30.0-plus-backport-55708"}


def resolve_vllm_runtime(environ: Mapping[str, str]) -> str:
    """``PTIT_VLLM_RUNTIME`` (stock or backport-55708); unset means stock."""
    runtime = environ.get("PTIT_VLLM_RUNTIME") or "stock"
    if runtime not in VLLM_RUNTIMES:
        raise ValueError(f"PTIT_VLLM_RUNTIME must be one of {VLLM_RUNTIMES}, got {runtime!r}")
    return runtime


@dataclass(frozen=True)
class LaneConfig:
    served_model_name: str
    speculative_config: dict[str, Any] | None = None
    scheduler_cls: str | None = None
    env: dict[str, str] = field(default_factory=dict)


LANES: dict[str, LaneConfig] = {
    "vanilla": LaneConfig(served_model_name="qwen2.5-7b-vanilla"),
    "static": LaneConfig(
        served_model_name="qwen2.5-7b-static-draft",
        speculative_config={
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 3,
            "use_heterogeneous_vocab": True,
            "draft_sample_method": "greedy",
        },
    ),
    "adaptive": LaneConfig(
        served_model_name="qwen2.5-7b-adaptive",
        # k ceiling 4 plus dynamic SD so the scheduler can change K per step.
        speculative_config={
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 4,
            "num_speculative_tokens_per_batch_size": [[1, 32, 4]],
            "use_heterogeneous_vocab": True,
            "draft_sample_method": "greedy",
        },
        scheduler_cls="src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
        env={
            "ADAPTIVE_ENABLE_VRAM_GUARD": "1",
            "ADAPTIVE_MAX_K": "4",
            "ADAPTIVE_MIN_BATCH_K": "0",
            "ADAPTIVE_PER_REQUEST": "1",
            "ADAPTIVE_CONTROLLER_KWARGS": "",
        },
    ),
    "eagle": LaneConfig(
        served_model_name=EAGLE_SERVED_MODEL,
        speculative_config={
            "method": "eagle3",
            "model": EAGLE_DRAFT_MODEL,
            "num_speculative_tokens": 3,
            "draft_sample_method": "greedy",
        },
    ),
}


def _lane(lane: str) -> LaneConfig:
    if lane not in LANES:
        raise ValueError(f"Unknown lane {lane!r}; expected one of {sorted(LANES)}")
    return LANES[lane]


def build_vllm_args(
    lane: str,
    profile: str = DEFAULT_PROFILE,
    port: int = DEFAULT_PORT,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> list[str]:
    """Arguments for ``vllm serve`` for one lane (model first, then flags)."""
    cfg = _lane(lane)
    if profile not in DTYPE_PROFILES:
        raise ValueError(f"Unknown dtype profile {profile!r}; expected one of {sorted(DTYPE_PROFILES)}")
    if max_model_len not in SUPPORTED_MAX_MODEL_LENS:
        raise ValueError(f"max_model_len must be one of {SUPPORTED_MAX_MODEL_LENS}, got {max_model_len!r}")
    dtype = DTYPE_PROFILES[profile]
    serving = list(COMMON_SERVING_ARGS)
    serving[serving.index("--max-model-len") + 1] = str(max_model_len)

    args = [TARGET_MODEL]
    if TARGET_REVISION:
        args += ["--revision", TARGET_REVISION]
    args += [
        "--host", "0.0.0.0",
        "--port", str(port),
        "--served-model-name", cfg.served_model_name,
        "--quantization", dtype["quantization"],
        "--dtype", dtype["dtype"],
        *serving,
    ]
    if cfg.speculative_config is not None:
        spec = dict(cfg.speculative_config)
        if lane == "eagle":
            spec["revision"] = EAGLE_REVISION
        elif DRAFT_REVISION:
            spec["revision"] = DRAFT_REVISION
        args += ["--speculative-config", json.dumps(spec)]
    if cfg.scheduler_cls is not None:
        args += ["--scheduler-cls", cfg.scheduler_cls]
    if cfg.speculative_config is not None:
        args += ["--per-request-spec-decode-metrics", "summary"]
    return args


def lane_env(lane: str, run_id: str) -> dict[str, str]:
    """Extra environment variables for the vLLM process of one lane."""
    cfg = _lane(lane)
    # /root keeps the runner's own modules importable next to the project code.
    env = {"PYTHONPATH": f"{REMOTE_PROJECT_ROOT}:/root", **cfg.env}
    if lane == "adaptive":
        env["ADAPTIVE_TELEMETRY_PATH"] = f"{ARTIFACTS_ROOT}/lane3/{run_id}/telemetry.jsonl"
    return env


# --------------------------------------------------------------------------
# [M3-01] server-reported environment and in-region benchmark
# --------------------------------------------------------------------------
L4_ENVIRONMENT_PROFILE = "m3-l4-modal-v1"

# Same frozen workload artifacts as the canonical M2-04 campaign (experiments/run_m2_campaign.py).
BENCH_WORKLOADS = {
    "W1": "experiments/configs/w1/node1_chat_requests.jsonl",
    "W2": "experiments/configs/w2_v2/chat_requests.jsonl",
    "W3": "experiments/configs/w3_v2/chat_requests.jsonl",
}
# [M5-01] Accepted for single cells but not part of the M2/M3 campaign loops.
# W1_V2 requests need a server deployed with PTIT_MAX_MODEL_LEN=8192.
CANDIDATE_WORKLOADS = {
    "W1_V2": "experiments/configs/w1_v2/node1_chat_requests.jsonl",
}
ALL_WORKLOADS = {**BENCH_WORKLOADS, **CANDIDATE_WORKLOADS}


def workload_file(workload: str) -> str:
    if workload not in ALL_WORKLOADS:
        raise ValueError(f"Unknown workload {workload!r}; expected one of {sorted(ALL_WORKLOADS)}")
    return ALL_WORKLOADS[workload]


# [M5-01] Serving needs that differ from the M2/M3 contract. W1_V2 prompts are
# ~2k tokens with max_tokens 3072, and outputs reach ~2.1k tokens: a 4096
# context is too short and the harness default timeout (120 s) is close to a
# c=32 request. Its campaign uses the same patched vLLM runtime as M3-05.
WORKLOAD_SERVING: dict[str, dict[str, Any]] = {
    "W1_V2": {"max_model_len": 8192, "timeout_s": 600, "vllm_runtime": "backport-55708"},
}


def workload_serving(workload: str) -> dict[str, Any]:
    """Server context, harness request timeout and required runtime for a workload."""
    workload_file(workload)
    return {"max_model_len": DEFAULT_MAX_MODEL_LEN, "timeout_s": None, "vllm_runtime": None,
            **WORKLOAD_SERVING.get(workload, {})}


def server_artifact_dir(lane: str, run_id: str) -> str:
    """Volume folder holding a server's vllm.log, server_env.json and vram.jsonl."""
    return f"{ARTIFACTS_ROOT}/{lane}/{run_id}"


def build_server_env(
    *,
    gpu: str,
    driver_version: str,
    gpu_memory_total_mb: float,
    torch_version: str,
    torch_cuda_version: str,
    vllm_version: str,
    lane: str,
    run_id: str,
    profile: str,
) -> dict[str, Any]:
    """Runtime environment and configured model identity, not endpoint verification."""
    _lane(lane)
    dtype = DTYPE_PROFILES[profile]
    draft_revision = None
    if LANES[lane].speculative_config:
        draft_revision = EAGLE_REVISION if lane == "eagle" else DRAFT_REVISION
    return {
        "gpu": gpu,
        "driver_version": driver_version,
        "gpu_memory_total_mb": gpu_memory_total_mb,
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "vllm_version": vllm_version,
        "dtype": dtype["dtype"],
        "quantization": dtype["quantization"],
        "dtype_profile": profile,
        "lane": lane,
        "run_id": run_id,
        "served_model_name": LANES[lane].served_model_name,
        "target_model": TARGET_MODEL,
        "target_revision": TARGET_REVISION,
        "draft_model": (LANES[lane].speculative_config or {}).get("model"),
        "draft_revision": draft_revision,
    }


def bench_harness_args(
    lane: str,
    *,
    base_url: str,
    server_env_json: str,
    raw_out: str,
    summary_csv: str,
    experiment_id: str,
    workload: str,
    concurrency: int,
    repeats: int,
    num_requests: int,
    warmup_requests: int = 3,
) -> list[str]:
    """benchmark_harness arguments for one cell against a Modal lane.

    Mirrors the canonical M2-04 command (run_m2_campaign.build_plan) and only swaps
    the endpoint and environment profile. The proxy token is passed through the
    OPENAI_API_KEY environment variable, never on the command line.
    """
    cfg = _lane(lane)
    workload_path = workload_file(workload)
    timeout_s = workload_serving(workload)["timeout_s"]
    timeout = [] if timeout_s is None else ["--timeout-s", str(timeout_s)]
    return [
        "--lane", lane,
        "--workload", workload,
        "--endpoint", f"{base_url.rstrip('/')}/chat/completions",
        "--model", cfg.served_model_name,
        "--workload-file", workload_path,
        "--sample-policy", "cycle-rotate",
        "--concurrency", str(concurrency),
        "--repeats", str(repeats),
        "--num-requests", str(num_requests),
        "--warmup-requests", str(warmup_requests),
        "--experiment-id", experiment_id,
        "--raw-out", raw_out,
        "--summary-csv", summary_csv,
        "--environment-profile", L4_ENVIRONMENT_PROFILE,
        "--server-env-json", server_env_json,
        *timeout,
    ]


def pin_main_ref(hub_cache: Path, repo: str, sha: str) -> Path:
    """Point the cached ``main`` ref of ``repo`` at the pinned ``sha``.

    Downloading by commit hash writes no ``refs/main``. The servers run with
    HF_HUB_OFFLINE=1, and vLLM loads the draft tokenizer by repo id without a
    revision (``"main"``), so ``main`` must resolve locally to the pinned
    snapshot (vllm/v1/spec_decode/draft_model.py, use_heterogeneous_vocab).
    """
    repo_dir = Path(hub_cache) / f"models--{repo.replace('/', '--')}"
    if not (repo_dir / "snapshots" / sha).is_dir():
        raise FileNotFoundError(f"No snapshot {sha} for {repo} under {repo_dir}")
    ref = repo_dir / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(sha, encoding="utf-8")
    return ref


def server_env(lane: str, profile: str, run_id: str) -> dict[str, str]:
    """Container environment for one lane server, fixed at deploy time."""
    if profile not in DTYPE_PROFILES:
        raise ValueError(f"Unknown dtype profile {profile!r}; expected one of {sorted(DTYPE_PROFILES)}")
    return {
        "PTIT_LANE": lane,
        "PTIT_DTYPE_PROFILE": profile,
        "PTIT_RUN_ID": run_id,
        "HF_HOME": HF_HOME,
        # Weights come from the volume filled by download_weights; never re-download.
        "HF_HUB_OFFLINE": "1",
        "VLLM_CACHE_ROOT": VLLM_CACHE_ROOT,
        **lane_env(lane, run_id),
    }
