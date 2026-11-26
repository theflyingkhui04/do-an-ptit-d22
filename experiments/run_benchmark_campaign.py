"""Multi-GPU 3-node 3-lane FlexSDS benchmark campaign controller on Modal.

Matrix contract:
- 4 Modal GPUs: T4 (16 GB, fp16-fallback), A10 (24 GB), L40S (48 GB), A100 (80 GB -> A100-80GB).
- 3 nodes (tasks): W1_V3 (ctx 8192, timeout 600 s), W2 (w2_v2, ctx 4096), W3 (w3_v2, ctx 4096).
- 3 lanes: vanilla (target only), static (fixed-K speculative), adaptive (Hardware-Adaptive scheduler).
- Modes:
  - full: concurrency (1, 2, 4, 8, 16, 32) -> 216 cells.
  - fast: concurrency (1, 8, 32)           -> 108 cells.
  Both modes keep the exact same per-cell protocol: 3 warm-ups + 5 repeats x 32 measured requests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Sequence

from experiments import modal_lanes as lanes
from experiments import vllm_runtime_patch as runtime_patch
from experiments.bench_source import snapshot
from experiments.workload_adapter import load_workload_rows, verify_workload_integrity

try:
    from flexsds.core import cost_model as cm
    from flexsds.core.hardware import (
        MODAL_GPU_NAMES,
        build_remote_hardware_info,
        get_modal_gpu_spec,
        device_matches_exactly,
    )
except ModuleNotFoundError:
    from src.flexsds.core import cost_model as cm
    from src.flexsds.core.hardware import (
        MODAL_GPU_NAMES,
        build_remote_hardware_info,
        get_modal_gpu_spec,
        device_matches_exactly,
    )


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT

BENCHMARK_GPUS: tuple[str, ...] = tuple(MODAL_GPU_NAMES)  # ("T4", "A10", "L40S", "A100")
BENCHMARK_WORKLOADS: tuple[str, ...] = ("W1_V3", "W2", "W3")
BENCHMARK_LANES: tuple[str, ...] = ("vanilla", "static", "adaptive")
CONCURRENCY_MODES: dict[str, tuple[int, ...]] = {
    "full": (1, 2, 4, 8, 16, 32),
    "fast": (1, 8, 32),
}

WARMUP_REQUESTS = 3
REPEATS = 5
NUM_REQUESTS = 32

DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEFAULT_DRAFT_MODEL = "duckhoi/qwen2.5-7b-eagle3.1-speculator"
DEFAULT_DRAFT_METHOD = "eagle3"
DEFAULT_STATIC_K = 3
DEFAULT_ADAPTIVE_MAX_K = 4

PLAN_ROOT = Path("experiments/runs/flexsds_bench")
RESULTS_ROOT = Path("experiments/results")


def concurrencies_for_mode(mode: str) -> tuple[int, ...]:
    key = str(mode or "").strip().lower()
    if key not in CONCURRENCY_MODES:
        raise ValueError(f"Benchmark mode must be one of {tuple(CONCURRENCY_MODES)}, got {mode!r}")
    return CONCURRENCY_MODES[key]


def parse_selection(
    raw: str | Sequence[str] | None,
    allowed: tuple[str, ...],
    *,
    label: str,
) -> tuple[str, ...]:
    """Parse 'all', comma-separated values, or sequences while preserving canonical order."""
    if raw is None:
        return allowed
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() == "all":
            return allowed
        tokens = [part.strip() for part in text.split(",") if part.strip()]
    else:
        tokens = [str(part).strip() for part in raw if str(part).strip()]
        if len(tokens) == 1 and tokens[0].lower() == "all":
            return allowed

    if not tokens:
        raise ValueError(f"At least one {label} must be selected")

    lookup = {item.lower(): item for item in allowed}
    chosen: set[str] = set()
    for token in tokens:
        canonical = lookup.get(token.lower())
        if canonical is None:
            raise ValueError(f"Unsupported {label} {token!r}; expected one of {allowed}")
        chosen.add(canonical)
    return tuple(item for item in allowed if item in chosen)


def _find_profile_in_dir(
    profile_dir: Path,
    hw_info: Any,
    target_model: str,
    draft_model: str,
) -> tuple[Path, dict[str, Any]] | None:
    if not profile_dir.is_dir():
        return None
    candidates = sorted(
        profile_dir.glob("profile_*.json"),
        key=lambda p: (p.stem.endswith("_fast"), p.name),
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("metadata", {}).get("schema") != "flexsds-component-v2":
                continue
            cm.validate_measured_profile(data, hw_info, target_model, draft_model)
            return path.resolve(), data
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def resolve_adaptive_profiles(
    gpus: Sequence[str],
    *,
    target_model: str = DEFAULT_TARGET_MODEL,
    draft_model: str = DEFAULT_DRAFT_MODEL,
    profile_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Inspect adaptive profile readiness for each requested Modal GPU without mutating state."""
    extra_dir = Path(profile_dir).expanduser() if profile_dir else None
    readiness: dict[str, dict[str, Any]] = {}
    for gpu in gpus:
        hw_info = build_remote_hardware_info("modal", gpu)
        cal_cmd = (
            f"flexsds profile --benchmark --backend modal --gpu {gpu} "
            f"--target {target_model} --draft {draft_model}"
        )
        found_path: Path | None = None
        found_data: dict[str, Any] | None = None

        if extra_dir is not None:
            match = _find_profile_in_dir(extra_dir, hw_info, target_model, draft_model)
            if match is not None:
                found_path, found_data = match

        if found_path is None:
            model, name = cm.load_profile_for_hardware(
                hw_info,
                strict=True,
                target_model=target_model,
                draft_model=draft_model,
            )
            if model is not None and name:
                resolved = cm.resolve_profile_path(name)
                if resolved is not None and resolved.is_file():
                    found_path = resolved.resolve()
                    found_data = json.loads(found_path.read_text(encoding="utf-8"))

        if found_path is not None and found_data is not None:
            readiness[gpu] = {
                "status": "ready",
                "profile_path": str(found_path),
                "profile_name": found_path.name,
                "profile_sha256": found_data.get("metadata", {}).get("raw_sha256"),
                "calibrate_command": cal_cmd,
            }
        else:
            readiness[gpu] = {
                "status": "missing",
                "profile_path": None,
                "profile_name": None,
                "profile_sha256": None,
                "calibrate_command": cal_cmd,
            }
    return readiness


def ensure_adaptive_profiles(
    gpus: Sequence[str],
    *,
    target_model: str = DEFAULT_TARGET_MODEL,
    draft_model: str = DEFAULT_DRAFT_MODEL,
    profile_dir: Path | str | None = None,
    auto_calibrate: bool = False,
    calibrate_mode: str = "fast",
) -> dict[str, dict[str, Any]]:
    """Fail closed when adaptive profiles are missing, or calibrate missing GPUs when authorized."""
    if calibrate_mode not in ("fast", "full"):
        raise ValueError(f"calibrate_mode must be 'fast' or 'full', got {calibrate_mode!r}")

    readiness = resolve_adaptive_profiles(
        gpus,
        target_model=target_model,
        draft_model=draft_model,
        profile_dir=profile_dir,
    )
    missing = [gpu for gpu in gpus if readiness[gpu]["status"] != "ready"]
    if not missing:
        return readiness

    if not auto_calibrate:
        commands = "\n".join(f"  - {gpu}: {readiness[gpu]['calibrate_command']}" for gpu in missing)
        raise RuntimeError(
            "Missing verified flexsds-component-v2 adaptive hardware profiles for GPU(s): "
            f"{', '.join(missing)}.\n"
            "Calibrate each missing GPU first (or pass --auto-calibrate):\n"
            f"{commands}"
        )

    target_dir = Path(profile_dir).expanduser() if profile_dir else None
    for gpu in missing:
        hw_info = build_remote_hardware_info("modal", gpu)
        cm.run_hardware_micro_benchmark(
            hw_info,
            target_model=target_model,
            draft_model=draft_model,
            target_dir=target_dir,
            backend="modal",
            measurement_mode=calibrate_mode,
        )

    refreshed = resolve_adaptive_profiles(
        gpus,
        target_model=target_model,
        draft_model=draft_model,
        profile_dir=profile_dir,
    )
    still_missing = [gpu for gpu in gpus if refreshed[gpu]["status"] != "ready"]
    if still_missing:
        raise RuntimeError(
            f"Auto-calibration completed but verified profiles are still missing for: {', '.join(still_missing)}"
        )
    return refreshed


def resolve_git_head(repo_root: Path = REPO_ROOT) -> str:
    git = ["git", "-c", f"safe.directory={repo_root.as_posix()}"]
    return subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def plan(
    campaign_id: str,
    sha: str,
    *,
    mode: str = "full",
    gpus: str | Sequence[str] | None = BENCHMARK_GPUS,
    workloads: str | Sequence[str] | None = BENCHMARK_WORKLOADS,
    lanes_list: str | Sequence[str] | None = BENCHMARK_LANES,
    target_model: str = DEFAULT_TARGET_MODEL,
    draft_model: str = DEFAULT_DRAFT_MODEL,
    draft_method: str = DEFAULT_DRAFT_METHOD,
    static_k: int = DEFAULT_STATIC_K,
    adaptive_max_k: int = DEFAULT_ADAPTIVE_MAX_K,
    use_heterogeneous_vocab: bool = False,
    profile_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build a frozen, deterministic multi-GPU 3-node 3-lane benchmark plan (read-only)."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign_id):
        raise ValueError("Invalid campaign_id")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("A full 40-char committed execution SHA is required")
    if isinstance(static_k, bool) or not isinstance(static_k, int) or not 1 <= static_k <= 8:
        raise ValueError("static_k must be an integer between 1 and 8")
    if isinstance(adaptive_max_k, bool) or not isinstance(adaptive_max_k, int) or not 1 <= adaptive_max_k <= 8:
        raise ValueError("adaptive_max_k must be an integer between 1 and 8")
    if draft_method not in ("eagle", "eagle3", "draft_model"):
        raise ValueError(f"Unsupported draft_method {draft_method!r}")

    concurrencies = concurrencies_for_mode(mode)
    selected_gpus = parse_selection(gpus, BENCHMARK_GPUS, label="GPU")
    selected_workloads = parse_selection(workloads, BENCHMARK_WORKLOADS, label="workload")
    selected_lanes = parse_selection(lanes_list, BENCHMARK_LANES, label="lane")

    readiness = (
        resolve_adaptive_profiles(
            selected_gpus,
            target_model=target_model,
            draft_model=draft_model,
            profile_dir=profile_dir,
        )
        if "adaptive" in selected_lanes
        else {}
    )

    cells: list[dict[str, Any]] = []
    for gpu in selected_gpus:
        modal_gpu = lanes.modal_gpu_resource(gpu)
        dtype_profile = lanes.default_dtype_profile(gpu)
        env_profile = lanes.environment_profile_for_gpu(gpu)
        for lane in selected_lanes:
            for workload in selected_workloads:
                serving = lanes.workload_serving(workload)
                for concurrency in concurrencies:
                    cells.append(
                        {
                            "key": f"{gpu}__{lane}__{workload}__c{concurrency}",
                            "gpu": gpu,
                            "modal_gpu": modal_gpu,
                            "dtype_profile": dtype_profile,
                            "environment_profile": env_profile,
                            "lane": lane,
                            "workload": workload,
                            "concurrency": concurrency,
                            "max_model_len": serving["max_model_len"],
                            "timeout_s": serving["timeout_s"] or 120,
                            "vllm_runtime": "backport-55708",
                        }
                    )

    return {
        "schema_version": "flexsds-bench-v1",
        "campaign_id": campaign_id,
        "commit": sha,
        "mode": mode,
        "gpus": list(selected_gpus),
        "workloads": list(selected_workloads),
        "lanes": list(selected_lanes),
        "concurrencies": list(concurrencies),
        "target_model": target_model,
        "draft_model": draft_model,
        "draft_method": draft_method,
        "static_k": static_k,
        "adaptive_max_k": adaptive_max_k,
        # Same execution contract for every lane: eager + async scheduling, no prefix cache (design 4.4).
        "match_adaptive_runtime": True,
        "disable_prefix_caching": True,
        "use_heterogeneous_vocab": bool(use_heterogeneous_vocab),
        "vllm_runtime": runtime_patch.RUNTIME_LABEL,
        "warmup": WARMUP_REQUESTS,
        "repeats": REPEATS,
        "num_requests": NUM_REQUESTS,
        "sample_policy": "cycle-rotate",
        "lifecycle": "one-container-per-gpu",
        "total_cells": len(cells),
        "total_measured_requests": len(cells) * REPEATS * NUM_REQUESTS,
        "profile_readiness": readiness,
        "cells": cells,
    }


def build_cell_serving_config(
    record: dict[str, Any],
    cell: dict[str, Any],
    *,
    profile_path: str | None = None,
) -> dict[str, Any]:
    """Build the LocalServingRunner-compatible serving config for a single benchmark cell."""
    gpu = cell["gpu"]
    lane = cell["lane"]
    spec = get_modal_gpu_spec(gpu)
    dtype_profile = lanes.default_dtype_profile(gpu)
    dtype_spec = lanes.DTYPE_PROFILES[dtype_profile]
    target_lower = str(record["target_model"]).lower()
    if "awq" in target_lower:
        dtype = dtype_spec["dtype"]
        quantization = dtype_spec["quantization"]
    elif "gptq" in target_lower:
        dtype = "float16"
        quantization = "gptq"
    else:
        dtype = "half" if gpu == "T4" else "bfloat16"
        quantization = "none"

    if lane == "adaptive" and not profile_path:
        raise ValueError(f"Adaptive lane on {gpu} requires a verified hardware profile path")

    profile_sha256: str | None = None
    if lane == "adaptive" and profile_path:
        p = Path(profile_path)
        if p.is_file():
            try:
                profile_sha256 = json.loads(p.read_text(encoding="utf-8")).get("metadata", {}).get("raw_sha256")
            except (OSError, ValueError):
                profile_sha256 = None

    is_vanilla = lane == "vanilla"
    is_adaptive = lane == "adaptive"
    return {
        "backend": "modal",
        "lane": lane,
        "workload": cell["workload"],
        "concurrency": cell["concurrency"],
        "target_gpu": gpu,
        "modal_gpu": lanes.modal_gpu_resource(gpu),
        "dtype_profile": dtype_profile,
        "environment_profile": cell["environment_profile"],
        "target_model": record["target_model"],
        "draft_model": None if is_vanilla else record["draft_model"],
        "draft_method": "none" if is_vanilla else record["draft_method"],
        "served_model_name": f"flexsds-bench-{lane}",
        "dtype": dtype,
        "quantization": quantization,
        "disable_speculative": is_vanilla,
        "enable_adaptive": is_adaptive,
        "static_k": None if (is_vanilla or is_adaptive) else int(record["static_k"]),
        "max_k": int(record.get("adaptive_max_k", DEFAULT_ADAPTIVE_MAX_K)),
        # The adaptive lane is eager + async by construction; vanilla and static are matched to it.
        "match_adaptive_runtime": not is_adaptive,
        "disable_prefix_caching": True,
        "use_heterogeneous_vocab": False if is_vanilla else bool(record.get("use_heterogeneous_vocab", False)),
        "max_model_len": int(cell["max_model_len"]),
        "max_num_seqs": 32,
        "port": 8000,
        "vllm_runtime": "backport-55708",
        "profile_path": profile_path if is_adaptive else None,
        "profile_sha256": profile_sha256 if is_adaptive else None,
        "hardware": {
            "device_name": f"Remote {gpu}",
            "sm_architecture": spec.sm_architecture,
            "vram_gb": spec.vram_gb,
            "free_vram_mb": spec.vram_budget_mb,
        },
    }


def _campaign_dir(record: dict[str, Any], plan_root: Path = PLAN_ROOT) -> Path:
    base = plan_root if plan_root.is_absolute() else ROOT / plan_root
    return base / record["campaign_id"]


def save_plan(record: dict[str, Any], plan_root: Path = PLAN_ROOT) -> Path:
    out = _campaign_dir(record, plan_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    frozen = {k: v for k, v in record.items() if k != "profile_readiness"}
    content = (json.dumps(frozen, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.pop("profile_readiness", None)
        if existing != frozen:
            raise FileExistsError("Campaign plan differs from frozen existing plan")
        return out
    with path.open("xb") as stream:
        stream.write(content)
    return out


def load_plan(campaign_id: str, plan_root: Path = PLAN_ROOT) -> dict[str, Any]:
    """Load an existing frozen campaign plan.json read-only without mutating disk."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign_id):
        raise ValueError("Invalid campaign_id")
    base = plan_root if plan_root.is_absolute() else ROOT / plan_root
    path = base / campaign_id / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen campaign plan not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_speculative_acceptance(raw: Sequence[dict[str, Any]], lane: str) -> tuple[float | None, float | None]:
    """Validate and aggregate speculative counters; allows genuine adaptive K=0 cells with 0 drafts."""
    from experiments import aggregate_results as agg

    if lane == "vanilla":
        return None, None
    if lane == "static":
        return agg.aggregate_static_acceptance(raw)

    # Adaptive lane may legitimately pick K=0 for every step on high-batch or short-output cells.
    drafted_total = 0
    for record in raw:
        server_metrics = record.get("server_metrics")
        if not isinstance(server_metrics, dict):
            continue
        spec = server_metrics.get("speculative_decoding")
        if not isinstance(spec, dict):
            continue
        drafted = spec.get("num_draft_tokens", 0)
        accepted = spec.get("num_accepted_draft_tokens", 0)
        steps = spec.get("num_spec_steps", 0)
        if not all(type(v) is int for v in (drafted, accepted, steps)) or not 0 <= accepted <= drafted or steps < 0:
            raise ValueError(f"Invalid adaptive native speculative counters: {spec!r}")
        drafted_total += drafted

    if drafted_total > 0:
        return agg.aggregate_static_acceptance(raw)
    return None, None


def _validate_server_vllm_args(server: dict[str, Any], expected_cfg: dict[str, Any], lane: str) -> None:
    args = server.get("vllm_args")
    if args is None:
        raise ValueError("Server vllm_args missing; the launch command must be recorded for every cell")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("Server vllm_args must be a list of strings")
    has_spec = "--speculative-config" in args
    if lane == "vanilla":
        if has_spec or "--scheduler-cls" in args:
            raise ValueError("Vanilla server vllm_args must not enable speculation or adaptive scheduler")
    else:
        if not has_spec:
            raise ValueError(f"Lane {lane!r} vllm_args missing --speculative-config")
        spec = json.loads(args[args.index("--speculative-config") + 1])
        if spec.get("method") != expected_cfg["draft_method"]:
            raise ValueError(f"Server vllm_args speculative method mismatch: {spec.get('method')!r}")
        if spec.get("model") != expected_cfg["draft_model"]:
            raise ValueError(f"Server vllm_args speculative model mismatch: {spec.get('model')!r}")
        if lane == "static":
            if spec.get("num_speculative_tokens") != expected_cfg["static_k"]:
                raise ValueError(
                    f"Server vllm_args static_k mismatch: expected {expected_cfg['static_k']}, "
                    f"got {spec.get('num_speculative_tokens')!r}"
                )
            if "--scheduler-cls" in args:
                raise ValueError("Static lane vllm_args must not set --scheduler-cls")
        elif lane == "adaptive":
            if "--scheduler-cls" not in args or "--distributed-executor-backend" not in args:
                raise ValueError("Adaptive lane vllm_args missing scheduler or lineage executor")
    # Every lane runs the matched eager + async runtime without prefix caching (design 4.4).
    for flag in ("--enforce-eager", "--async-scheduling", "--no-enable-prefix-caching"):
        if flag not in args:
            raise ValueError(f"Lane {lane!r} vllm_args missing {flag}")


def attempt_prefix(record: dict[str, Any], cell: dict[str, Any]) -> str:
    return f"{record['campaign_id']}-{cell['gpu']}-{cell['lane']}-{cell['workload']}-c{cell['concurrency']}-a"


def attempts_dir(record: dict[str, Any], plan_root: Path = PLAN_ROOT) -> Path:
    return _campaign_dir(record, plan_root) / "attempts"


def attempts(record: dict[str, Any], cell: dict[str, Any], plan_root: Path = PLAN_ROOT) -> list[Path]:
    root = attempts_dir(record, plan_root)
    if not root.is_dir():
        return []
    return sorted(root.glob(attempt_prefix(record, cell) + "[0-9][0-9][0-9]"))


def validate_cell(
    out: Path,
    record: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Fail-closed validation of one multi-GPU benchmark cell attempt."""
    from experiments import aggregate_results as agg

    receipt = json.loads((out / "submission.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    server = json.loads((out / "server_env.json").read_text(encoding="utf-8"))

    for key in ("gpu", "lane", "server_run_id", "bench_run_id", "cell", "commit", "call_id"):
        if meta.get(key) != receipt.get(key) or receipt.get(key) is None:
            raise ValueError(f"Receipt/collected identity mismatch: {key}")

    if receipt["commit"] != record["commit"]:
        raise ValueError("Campaign/attempt source commit mismatch")
    if (
        receipt["gpu"],
        receipt["lane"],
        receipt["cell"]["workload"],
        receipt["cell"]["concurrency"],
    ) != (cell["gpu"], cell["lane"], cell["workload"], cell["concurrency"]):
        raise ValueError("Attempt in wrong benchmark cell")
    if receipt["cell"]["repeats"] != REPEATS or receipt["cell"]["num_requests"] != NUM_REQUESTS:
        raise ValueError("Noncanonical repeat/request counts")
    if meta.get("returncode") != 0 or meta.get("server_instance_unchanged") is not True:
        raise ValueError("Failed harness or server restarted during cell")

    if not device_matches_exactly(cell["gpu"], str(server.get("gpu") or "")):
        raise ValueError(f"Server GPU mismatch: {cell['gpu']} must be its exact card, got {server.get('gpu')!r}")
    if not server.get("gpu_uuid"):
        raise ValueError("Server gpu_uuid missing")
    if record.get("source_digest") and server.get("source_digest") != record["source_digest"]:
        raise ValueError("Server source_digest differs from the campaign's uncommitted source digest")

    expected_cfg = build_cell_serving_config(
        record,
        cell,
        profile_path="__required__" if cell["lane"] == "adaptive" else None,
    )
    if server.get("dtype") != expected_cfg["dtype"] or server.get("quantization") != expected_cfg["quantization"]:
        raise ValueError(
            f"Server dtype/quantization mismatch for {cell['gpu']}: "
            f"got ({server.get('dtype')}, {server.get('quantization')}), "
            f"expected ({expected_cfg['dtype']}, {expected_cfg['quantization']})"
        )
    if server.get("max_model_len") != cell["max_model_len"]:
        raise ValueError("Server max_model_len mismatch")
    if (
        server.get("runtime_variant") != runtime_patch.RUNTIME_LABEL
        or server.get("vllm_outputs_sha256") != runtime_patch.PATCHED_OUTPUTS_SHA256
    ):
        raise ValueError("Patched vLLM runtime (backport #55708) required for benchmark cells")
    if not server.get("instance_id"):
        raise ValueError("Missing server instance_id")
    if server.get("lane") != cell["lane"] or server.get("run_id") != receipt["server_run_id"]:
        raise ValueError("Server lane/run_id mismatch")
    if server.get("source_commit") != record["commit"]:
        raise ValueError("Server source_commit mismatch")
    if server.get("target_model") != record["target_model"]:
        raise ValueError("Server target_model mismatch")
    if server.get("draft_model") != expected_cfg["draft_model"]:
        raise ValueError("Server draft_model mismatch")
    _validate_server_vllm_args(server, expected_cfg, cell["lane"])
    if cell["lane"] == "adaptive":
        sha = server.get("profile_sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Adaptive benchmark cell missing verified profile_sha256")
        expected_sha = receipt.get("expected_profile_sha256")
        if expected_sha and sha != expected_sha:
            raise ValueError("Adaptive benchmark cell profile_sha256 mismatch")

    if meta.get("endpoint_model_ids") != [expected_cfg["served_model_name"]]:
        raise ValueError("Missing endpoint model verification")

    raw = agg._load_jsonl(out / "raw.jsonl")
    warm = agg._load_jsonl(out / "warmup.jsonl")
    if len(raw) != REPEATS * NUM_REQUESTS or Counter(r.get("repeat_id") for r in raw) != {
        f"r{i:03}": NUM_REQUESTS for i in range(1, REPEATS + 1)
    }:
        raise ValueError("Expected exactly 160 measured requests: 5 repeats x 32")
    if len(warm) != WARMUP_REQUESTS or any(r.get("phase") != "warmup" for r in warm):
        raise ValueError("Expected exactly 3 separate warm-ups")
    if len({r.get("request_id") for r in raw + warm}) != REPEATS * NUM_REQUESTS + WARMUP_REQUESTS:
        raise ValueError("Missing or duplicate request identities")

    packed, sha = snapshot(REPO_ROOT, receipt["commit"])
    try:
        workload_path = Path(packed.name) / lanes.workload_file(cell["workload"])
        integrity = verify_workload_integrity(workload_path)
        corpus = load_workload_rows(workload_path)
    finally:
        packed.cleanup()
    if sha != receipt["commit"]:
        raise ValueError("Invalid submitted source SHA")

    for records, phase in ((raw, "measured"), (warm, "warmup")):
        seen: set[tuple[Any, Any]] = set()
        for row in records:
            identity = (row.get("repeat_id"), row.get("request_index"))
            if identity in seen:
                raise ValueError("Duplicate request index within repeat")
            seen.add(identity)
            index = row["request_index"]
            if type(index) is not int or index not in range(NUM_REQUESTS if phase == "measured" else WARMUP_REQUESTS):
                raise ValueError("Invalid request index")
            repeat = int(row["repeat_id"][1:]) if phase == "measured" else 1
            offset = ((repeat - 1) * (NUM_REQUESTS % len(corpus))) % len(corpus) if phase == "measured" else 0
            sample_index = (index + offset) % len(corpus)
            req = corpus[sample_index]["request"]
            exp = {
                "experiment_id": receipt["bench_run_id"],
                "lane": cell["lane"],
                "workload": cell["workload"],
                "concurrency": cell["concurrency"],
                "phase": phase,
                "commit": sha,
                "server_run_id": receipt["server_run_id"],
                "server_instance_id": server["instance_id"],
                "sample_policy": "cycle-rotate",
                "sample_id": corpus[sample_index]["sample_id"],
                "sample_index": sample_index,
                "sample_offset": offset,
                "canonical_sample_count": len(corpus),
                "workload_file_sha256": integrity.actual_sha256,
                "success": True,
                "metrics_complete": True,
                "http_status": 200,
                "model": expected_cfg["served_model_name"],
                "temperature": req.get("temperature"),
                "seed": req.get("seed"),
                "response_format": req.get("response_format"),
                "max_completion_tokens": req["max_tokens"],
            }
            if any(row.get(k) != v for k, v in exp.items()) or row.get("error_type") or row.get("error_message"):
                raise ValueError(f"Request protocol/identity mismatch: {identity}")

    warm_finished = max(
        datetime.fromisoformat(r["timestamp_utc"]) + timedelta(milliseconds=r["e2e_ms"])
        for r in warm
    )
    if warm_finished > min(datetime.fromisoformat(r["timestamp_utc"]) for r in raw):
        raise ValueError("Measured execution began before warm-up completed")

    from experiments.bench_gpu_session import concurrency_check

    samples = agg._load_jsonl(out / "concurrency.jsonl") if (out / "concurrency.jsonl").exists() else []
    inflight = concurrency_check(samples, raw, cell["concurrency"])
    if not inflight["ok"]:
        worst = min((v["peak_inflight"] for v in inflight["per_repeat"].values()), default=0.0)
        raise ValueError(f"Requested concurrency {cell['concurrency']} not reached in vLLM "
                         f"(lowest per-repeat peak running+waiting: {worst:g})")

    agg.validate_measured_records(raw)
    alpha_tau = summarize_speculative_acceptance(raw, cell["lane"])
    acceptance = None if alpha_tau == (None, None) else alpha_tau

    summary = None
    if (out / "raw.vram.jsonl").exists():
        joined = agg._load_jsonl(out / "raw.vram.jsonl")
        if len(joined) != REPEATS * NUM_REQUESTS or any(
            {k: v for k, v in a.items() if k != "run_vram_peak_mb"}
            != {k: v for k, v in b.items() if k not in ("run_vram_peak_mb", "vram_join_samples", "vram_source")}
            for a, b in zip(raw, joined)
        ):
            raise ValueError("VRAM join changed raw evidence")
        if any(r.get("run_vram_peak_mb") is None for r in joined):
            raise ValueError("Missing VRAM samples")
        summary = agg._aggregate_measured_records(joined)[0]

    duration = sum({r["repeat_id"]: r["run_duration_s"] for r in raw}.values())
    return {
        "protocol_status": "VALID",
        "commit": sha,
        "gpu": cell["gpu"],
        "server_gpu": server["gpu"],
        "lane": cell["lane"],
        "workload": cell["workload"],
        "concurrency": cell["concurrency"],
        "measured_requests": REPEATS * NUM_REQUESTS,
        "repeats": REPEATS,
        "requests_per_repeat": NUM_REQUESTS,
        "warmup_attempted": WARMUP_REQUESTS,
        "warmup_successful": WARMUP_REQUESTS,
        "warmup_excluded": True,
        "request_failures": 0,
        "workload_hash": integrity.actual_sha256,
        "server_instance_id": server["instance_id"],
        "gpu_uuid": server["gpu_uuid"],
        "concurrency_per_repeat": inflight["per_repeat"],
        "speculative_alpha_tau": acceptance,
        "measured_duration_s": duration,
        "output_tokens_per_s": sum(r["output_tokens"] for r in raw) / duration,
        "summary": summary,
    }


def selected_attempt(
    record: dict[str, Any],
    cell: dict[str, Any],
    plan_root: Path = PLAN_ROOT,
) -> Path | None:
    valid: list[Path] = []
    for out in attempts(record, cell, plan_root):
        if (out / "validation_failure.json").exists():
            continue
        if not (out / "bench.json").exists():
            continue
        validate_cell(out, record, cell)
        valid.append(out)
    if len(valid) > 1:
        raise ValueError(f"Duplicate complete canonical cell: {cell['key']}")
    return valid[0] if valid else None


def validate_campaign(record: dict[str, Any], plan_root: Path = PLAN_ROOT) -> list[Path]:
    """Every cell valid; one card per GPU; a vLLM process never serves two (workload, lane) groups."""
    selected = [selected_attempt(record, cell, plan_root) for cell in record["cells"]]
    missing = [c["key"] for c, p in zip(record["cells"], selected) if p is None]
    if missing:
        raise ValueError(f"Incomplete campaign ({len(missing)} missing cells): {missing[:10]}")
    paths = [p for p in selected if p is not None]
    cards: dict[str, set[str]] = defaultdict(set)
    owner: dict[str, tuple[str, str, str]] = {}
    for path, cell in zip(paths, record["cells"]):
        checked = validate_cell(path, record, cell)
        cards[cell["gpu"]].add(checked["gpu_uuid"])
        group = (cell["gpu"], cell["workload"], cell["lane"])
        if owner.setdefault(checked["server_instance_id"], group) != group:
            raise ValueError(f"vLLM instance {checked['server_instance_id']} shared by {owner[checked['server_instance_id']]} and {group}")
    for gpu, uuids in cards.items():
        if len(uuids) > 1:
            raise ValueError(f"{gpu} was measured on more than one card: {sorted(uuids)}")
    return paths


def _fmt(value: Any, digits: int) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.{digits}f}"


def render_markdown_report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    lines = [
        f"# Báo cáo benchmark đa GPU — {campaign['campaign_id']}",
        "",
        f"- **Chế độ:** `{campaign['mode']}` (concurrency `{campaign['concurrencies']}`, "
        f"{campaign['warmup']} warm-up + {campaign['repeats']} × {campaign['num_requests']} request/ô)",
        f"- **Phần cứng Modal:** {', '.join(campaign['gpus'])}",
        f"- **3 node (tác vụ):** {', '.join(campaign['workloads'])}",
        f"- **3 làn:** {', '.join(campaign['lanes'])}",
        f"- **Mô hình:** target `{campaign['target_model']}`, draft `{campaign['draft_model']}` (`{campaign['draft_method']}`, static `K={campaign['static_k']}`, adaptive `max_k={campaign['adaptive_max_k']}`)",
        "- **Quy tắc tăng tốc:** `speedup_vs_vanilla` luôn tính trên **cùng GPU, cùng workload, cùng concurrency**.",
        "",
    ]
    by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary["cells"]:
        by_gpu[row["gpu_tier"]].append(row)

    lane_order = {lane: i for i, lane in enumerate(BENCHMARK_LANES)}
    workload_order = {w: i for i, w in enumerate(BENCHMARK_WORKLOADS)}

    for gpu in campaign["gpus"]:
        rows = sorted(
            by_gpu.get(gpu, []),
            key=lambda r: (workload_order.get(r["workload"], 99), r["concurrency"], lane_order.get(r["lane"], 99)),
        )
        lines.extend(
            [
                f"## GPU {gpu}",
                "",
                "| Workload | Concurrency | Làn | TPOT p50 (ms) | Throughput (tok/s) | Speedup vs vanilla | α | τ | VRAM peak (MB) |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['workload']} | {row['concurrency']} | {row['lane']} | "
                f"{_fmt(row.get('tpot_p50'), 2)} | {_fmt(row.get('throughput'), 1)} | "
                f"{_fmt(row.get('speedup_vs_vanilla'), 3)} | {_fmt(row.get('alpha'), 3)} | "
                f"{_fmt(row.get('tau'), 3)} | {_fmt(row.get('vram_peak_mb'), 1)} |"
            )
        lines.append("")
    return "\n".join(lines)


def aggregate(
    record: dict[str, Any],
    *,
    plan_root: Path = PLAN_ROOT,
    destination: Path | None = None,
    report_destination: Path | None = None,
) -> dict[str, Any]:
    """Aggregate validated cells, computing speedup_vs_vanilla independently per GPU."""
    from experiments import aggregate_results as agg

    paths = validate_campaign(record, plan_root=plan_root)
    rows_by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path, cell in zip(paths, record["cells"]):
        raw_path = path / "raw.vram.jsonl"
        if not raw_path.exists():
            raise ValueError(f"VRAM evidence required for final aggregation: {path}")
        raw = agg._load_jsonl(raw_path)
        row = agg.aggregate_raw_jsonl(raw_path)[0]
        if row["lane"] != "vanilla":
            alpha, tau = summarize_speculative_acceptance(raw, row["lane"])
            row["alpha"] = "N/A" if alpha is None else alpha
            row["tau"] = "N/A" if tau is None else tau
        row["gpu_tier"] = cell["gpu"]
        row["modal_gpu"] = cell["modal_gpu"]
        row["attempt_id"] = path.name
        row["_experiment_id"] = record["campaign_id"]
        rows_by_gpu[cell["gpu"]].append(row)

    has_vanilla = "vanilla" in record["lanes"]
    all_rows: list[dict[str, Any]] = []
    for gpu in record["gpus"]:
        gpu_rows = rows_by_gpu.get(gpu, [])
        if has_vanilla and gpu_rows:
            gpu_rows = agg.apply_speedup_vs_vanilla(gpu_rows)
        all_rows.extend(gpu_rows)

    result = {
        "schema_version": "flexsds-bench-summary-v1",
        "campaign": record,
        "scientific_status": "VALID",
        "selected_attempts": [str(p) for p in paths],
        "all_attempts": {
            cell["key"]: [p.name for p in attempts(record, cell, plan_root)]
            for cell in record["cells"]
        },
        "cells": all_rows,
    }

    dest = destination or (ROOT / RESULTS_ROOT / f"flexsds_bench_{record['campaign_id']}_summary.json")
    report_dest = report_destination or (ROOT / RESULTS_ROOT / f"flexsds_bench_{record['campaign_id']}_report.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    report_dest.parent.mkdir(parents=True, exist_ok=True)

    def _semantic_summary(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "cells": [{k: v for k, v in r.items() if k != "timestamp"} for r in payload.get("cells", [])],
        }

    serialized = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if dest.exists():
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if _semantic_summary(existing) != _semantic_summary(result):
            raise FileExistsError(f"Conflicting summary: {dest}")
    else:
        dest.write_bytes(serialized)

    report_md = render_markdown_report(result)
    if report_dest.exists() and report_dest.read_text(encoding="utf-8") != report_md:
        raise FileExistsError(f"Conflicting report: {report_dest}")
    if not report_dest.exists():
        report_dest.write_text(report_md, encoding="utf-8", newline="\n")

    return result


# Rough A10 seconds per cell (3 warm-up + 5 x 32 requests), averaged over the three lanes; from the
# 25/11/2026 A10 runs (batchk-2511*). Other GPUs scale by a guessed speed factor. Shown as a rough estimate.
_A10_CELL_SECONDS = {
    "W1_V3": {1: 1430, 2: 800, 4: 480, 8: 270, 16: 255, 32: 255},
    "W2": {1: 224, 2: 130, 4: 70, 8: 30, 16: 25, 32: 25},
    "W3": {1: 900, 2: 500, 4: 300, 8: 165, 16: 150, 32: 150},
}
_GPU_SPEED_FACTOR = {"T4": 2.6, "A10": 1.0, "L40S": 0.6, "A100": 0.55}
_SERVER_START_S = 120  # one vLLM start per (workload, lane)
_CONTAINER_START_S = 300  # image pull + model cache per GPU container


def estimate_cost(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Rough GPU hours and USD per GPU for the cells of a plan (Modal catalog prices)."""
    out: dict[str, dict[str, float]] = {}
    for gpu in record["gpus"]:
        cells = [c for c in record["cells"] if c["gpu"] == gpu]
        measure = sum(_A10_CELL_SECONDS[c["workload"]][c["concurrency"]] for c in cells) * 1.1
        groups = {(c["workload"], c["lane"]) for c in cells}
        seconds = measure * _GPU_SPEED_FACTOR[gpu] + len(groups) * _SERVER_START_S + _CONTAINER_START_S
        hours = seconds / 3600
        out[gpu] = {"hours": round(hours, 2), "usd": round(hours * get_modal_gpu_spec(gpu).price_per_hour, 2)}
    return out


def _runner():
    from experiments import modal_benchmark_runner as runner
    return runner


def _ensure_modal_login() -> None:
    try:
        from flexsds.core.credentials import CredentialsManager
        from flexsds.server import modal_auth
    except ModuleNotFoundError:
        from src.flexsds.core.credentials import CredentialsManager
        from src.flexsds.server import modal_auth
    modal_auth.ensure_modal()
    if not modal_auth.prepare_modal_auth(CredentialsManager()):
        raise RuntimeError("Modal credentials missing; run 'flexsds auth login --provider modal' first.")


def _calls_path(record: dict[str, Any], plan_root: Path) -> Path:
    return _campaign_dir(record, plan_root) / "gpu_calls.json"


def _load_calls(record: dict[str, Any], plan_root: Path) -> dict[str, str]:
    path = _calls_path(record, plan_root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_calls(record: dict[str, Any], plan_root: Path, calls: dict[str, str]) -> None:
    path = _calls_path(record, plan_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calls, indent=2) + "\n", encoding="utf-8")


MAX_CONSECUTIVE_POLL_ERRORS = 30


def _wait_and_collect(runner: Any, record: dict[str, Any], calls: dict[str, str], att_root: Path,
                      poll_s: float) -> None:
    """Collect until no GPU call runs. A dropped connection only skips one poll: the sessions keep running
    on Modal, so the laptop gives up only after many consecutive failures."""
    failures = 0
    while True:
        try:
            running = any(runner.gpu_call_state(c) == "RUNNING" for c in calls.values())
            for gpu in calls:
                runner.collect_gpu(record["campaign_id"], gpu, att_root)
            failures = 0
            if not running:
                return
        except Exception as exc:  # network / Modal API hiccup; KeyboardInterrupt still stops the loop
            failures += 1
            print(f"Polling Modal failed ({failures}/{MAX_CONSECUTIVE_POLL_ERRORS}): {type(exc).__name__}: {exc}",
                  flush=True)
            if failures >= MAX_CONSECUTIVE_POLL_ERRORS:
                raise RuntimeError(
                    f"Lost contact with Modal {failures} times in a row; the GPU sessions keep running there. "
                    f"Resume with: flexsds benchmark status --campaign-id {record['campaign_id']}") from exc
        time.sleep(poll_s)


def drive(
    record: dict[str, Any],
    *,
    retry_failed: bool = False,
    plan_root: Path = PLAN_ROOT,
    profile_dir: Path | str | None = None,
    auto_calibrate: bool = False,
    calibrate_mode: str = "fast",
    wait: bool = True,
    poll_s: float = 60.0,
) -> None:
    """One Modal call per GPU for its missing cells; resumable. A GPU whose call still runs is only collected.

    `retry_failed` is accepted for CLI compatibility: missing or invalid cells are always resubmitted on `run`.
    """
    runner = _runner()
    commit, digest = runner.committed_bench_source(allow_uncommitted=bool(record.get("allow_uncommitted")))
    if commit != record["commit"]:
        raise ValueError("Checkout must match the frozen campaign execution SHA")
    if record.get("source_digest") and digest != record["source_digest"]:
        raise ValueError("Uploaded source changed since the campaign was planned (source_digest differs)")
    _ensure_modal_login()
    profiles = (
        ensure_adaptive_profiles(record["gpus"], target_model=record["target_model"], draft_model=record["draft_model"],
                                 profile_dir=profile_dir, auto_calibrate=auto_calibrate, calibrate_mode=calibrate_mode)
        if "adaptive" in record["lanes"] else {}
    )
    save_plan(record, plan_root)
    att_root = attempts_dir(record, plan_root)
    att_root.mkdir(parents=True, exist_ok=True)
    calls = _load_calls(record, plan_root)
    deployed = False
    errors: list[str] = []

    for gpu in record["gpus"]:
        previous = calls.get(gpu)
        if previous:
            runner.collect_gpu(record["campaign_id"], gpu, att_root)
            if runner.gpu_call_state(previous) == "RUNNING":
                print(f"{gpu}: call {previous} still running; collected finished cells", flush=True)
                continue
        cells = [c for c in record["cells"] if c["gpu"] == gpu]
        missing = [c for c in cells if not selected_attempt(record, c, plan_root)]
        if not missing:
            print(f"{gpu}: all {len(cells)} cells collected", flush=True)
            continue
        if not deployed:
            runner.deploy_app()  # once per run, from this checkout (the source the plan records)
            deployed = True
        numbers = {c["key"]: len(attempts(record, c, plan_root)) + 1 for c in missing}
        profile_path = (profiles.get(gpu) or {}).get("profile_path")
        profile_text = Path(profile_path).read_text(encoding="utf-8") if profile_path else None
        try:
            call_id = runner.submit_gpu(record, gpu, [c["key"] for c in missing], profile_text, attempts=numbers)
        except RuntimeError as exc:  # e.g. wrong card 12 times: the other GPUs still run
            errors.append(str(exc))
            print(f"{gpu}: not submitted: {exc}", flush=True)
            continue
        calls[gpu] = call_id
        _save_calls(record, plan_root, calls)
        print(f"{gpu}: submitted {len(missing)} cells as call {call_id}; the run continues on Modal if this "
              f"machine sleeps (resume: flexsds benchmark status --campaign-id {record['campaign_id']})", flush=True)

    if wait:
        _wait_and_collect(runner, record, calls, att_root, poll_s)
    if errors:
        raise RuntimeError("; ".join(errors) + " (run again to retry these GPUs)")


def format_plan_overview(record: dict[str, Any]) -> str:
    """Human-readable benchmark matrix summary for CLI plan / --dry-run output."""
    lines = [
        f"Benchmark campaign : {record['campaign_id']}",
        f"Mode               : {record['mode']} (concurrencies={record['concurrencies']})",
        f"GPUs ({len(record['gpus'])})           : {', '.join(record['gpus'])}",
        f"Workloads ({len(record['workloads'])})      : {', '.join(record['workloads'])}",
        f"Lanes ({len(record['lanes'])})          : {', '.join(record['lanes'])}",
        f"Models             : target={record['target_model']}, draft={record['draft_model']} "
        f"(method={record['draft_method']}, static_k={record['static_k']}, adaptive_max_k={record['adaptive_max_k']})",
        f"Per-cell protocol  : {record['warmup']} warm-up + {record['repeats']} repeats x {record['num_requests']} requests = "
        f"{record['repeats'] * record['num_requests']} measured/cell",
        f"Total matrix       : {record['total_cells']} cells ({record['total_measured_requests']:,} measured requests)",
    ]
    estimate = estimate_cost(record)
    lines.append("Estimated GPU time (rough, +/-50%; Modal list prices):")
    for gpu in record["gpus"]:
        lines.append(f"  - {gpu:<4} {estimate[gpu]['hours']:.1f} h ~ {estimate[gpu]['usd']:.2f} USD")
    lines.append(f"  = total {sum(v['hours'] for v in estimate.values()):.1f} h ~ "
                 f"{sum(v['usd'] for v in estimate.values()):.2f} USD")
    readiness = record.get("profile_readiness") or {}
    if readiness:
        lines.append("Adaptive GPU profile readiness:")
        for gpu in record["gpus"]:
            info = readiness.get(gpu, {})
            status = info.get("status", "missing")
            detail = info.get("profile_name") if status == "ready" else info.get("calibrate_command", "calibration required")
            lines.append(f"  - {gpu:<4} [{status}] {detail}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("plan", "run", "status", "validate", "aggregate", "stop"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--execution-sha", default=None, help="Full 40-char Git SHA (defaults to HEAD for read-only plan)")
    parser.add_argument("--mode", choices=tuple(CONCURRENCY_MODES), default="full")
    parser.add_argument("--gpus", default="all", help="Comma-separated GPUs from T4,A10,L40S,A100 or 'all'")
    parser.add_argument("--workloads", default="all", help="Comma-separated workloads from W1_V3,W2,W3 or 'all'")
    parser.add_argument("--lanes", default="all", help="Comma-separated lanes from vanilla,static,adaptive or 'all'")
    parser.add_argument("--target", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--draft-method", choices=("eagle", "eagle3", "draft_model"), default=DEFAULT_DRAFT_METHOD)
    parser.add_argument("--static-k", type=int, default=DEFAULT_STATIC_K)
    parser.add_argument("--adaptive-max-k", type=int, default=DEFAULT_ADAPTIVE_MAX_K)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--auto-calibrate", action="store_true")
    parser.add_argument("--calibrate-mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-uncommitted", action="store_true")
    return parser


def format_campaign_status(record: dict[str, Any], plan_root: Path = PLAN_ROOT, *, collect: bool = True) -> list[str]:
    """One line per GPU (cells collected, card, call state), then the invalid cells with their reason."""
    runner = _runner()
    att_root = attempts_dir(record, plan_root)
    calls = _load_calls(record, plan_root)
    lines: list[str] = []
    invalid: list[str] = []
    valid_keys: set[str] = set()
    for gpu in record["gpus"]:
        call_id = calls.get(gpu)
        if collect and call_id:
            runner.collect_gpu(record["campaign_id"], gpu, att_root)
        cells = [c for c in record["cells"] if c["gpu"] == gpu]
        done, card = 0, "?"
        for cell in cells:
            for out in attempts(record, cell, plan_root):
                try:
                    checked = validate_cell(out, record, cell)
                except (ValueError, KeyError, OSError) as exc:
                    if (out / "bench.json").exists():
                        invalid.append(f"  {cell['key']} ({out.name}): {exc}")
                    continue
                done += 1
                valid_keys.add(cell["key"])
                card = f"{checked['server_gpu']} ({checked['gpu_uuid'][:12]})"
                break
        state = runner.gpu_call_state(call_id) if call_id else "not submitted"
        lines.append(f"{gpu}: {done}/{len(cells)} cells valid, card {card}, call {state}")
    failures_dir = _campaign_dir(record, plan_root) / "failures"
    failed = sorted(failures_dir.glob("*.json")) if failures_dir.is_dir() else []
    for path in failed:
        if path.stem in valid_keys:  # an earlier container failure, since measured successfully
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        invalid.append(f"  {path.stem}: {data.get('status')}: {data.get('error')}")
    if invalid:
        lines.append("Invalid or failed cells:")
        lines.extend(invalid)
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "stop":
        from experiments import modal_benchmark_runner as runner
        runner.stop_app()
        return 0
    if args.action in ("status", "validate", "aggregate"):
        record = load_plan(args.campaign_id)
        if args.action == "status":
            for line in format_campaign_status(record):
                print(line)
        elif args.action == "validate":
            cells = validate_campaign(record)
            print(f"VALID: {len(cells)} cells, {len(cells) * REPEATS * NUM_REQUESTS} measured requests")
        else:
            aggregate(record)
        return 0

    sha = args.execution_sha or resolve_git_head()
    record = plan(
        args.campaign_id,
        sha,
        mode=args.mode,
        gpus=args.gpus,
        workloads=args.workloads,
        lanes_list=args.lanes,
        target_model=args.target,
        draft_model=args.draft,
        draft_method=args.draft_method,
        static_k=args.static_k,
        adaptive_max_k=args.adaptive_max_k,
        profile_dir=args.profile_dir,
    )
    if args.action == "plan":
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    if args.allow_uncommitted:
        record["allow_uncommitted"] = True
        record["source_digest"] = _runner().committed_bench_source(allow_uncommitted=True)[1]
    drive(
        record,
        retry_failed=args.retry_failed,
        profile_dir=args.profile_dir,
        auto_calibrate=args.auto_calibrate,
        calibrate_mode=args.calibrate_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
