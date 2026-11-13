"""Measured scalar A100 component profiles, independent of dormant tree formulas.

Costs use existing millisecond schema: target verification forward plus full
EAGLE proposal. They exclude target logits/sampling and are NOT end-to-end cycle
latencies. No bandwidth or arbitrary-tree values are invented.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from statistics import median
from datetime import datetime, timezone

from experiments import m4_cost_profiler as prof
from experiments.m4_protocol import ROOT, sha256, source_gate, write_new
from experiments.m4_treatments import TARGET, TARGET_REVISION, SPECULATOR, SPECULATOR_REVISION

BATCHES = (1, 2, 4, 8, 16, 32)
KS = (0, 1, 2, 3, 4)


def inspect_environment():
    """Query CUDA in a short-lived process, keeping the orchestrator GPU-free."""
    command = [sys.executable, "-c", "import json; from experiments.m4_cost_runner import observe_hardware; print(json.dumps(observe_hardware()))"]
    return json.loads(subprocess.check_output(command, cwd=ROOT, text=True))


def hardware_gate(env: dict, *, installed=True) -> None:
    if ("NVIDIA" not in (env.get("gpu") or "") or "A100" not in (env.get("gpu") or "")
            or env.get("gpu_compute_capability") != [8, 0]
            or not env.get("cuda_available") or not (env.get("gpu_memory_total_mb") or 0) > 0):
        raise ValueError("A100_GATE FAIL: actual NVIDIA A100 / capability 8.0 required")
    if installed and (env.get("vllm_version") != "0.30.0"
                      or env.get("torch_version") != "2.13.0+cu130"
                      or env.get("cuda_version") != "13.0"):
        raise ValueError("ENVIRONMENT_GATE FAIL: reproduce pinned runtime; do not upgrade")


def validate_profile(profile: dict, source_commit: str):
    from src.flexsds.core.cost_model import FlexCostModel
    meta = profile["metadata"]
    hardware_gate(meta)
    if (meta.get("sm_architecture") != "sm_80" or meta.get("source_commit") != source_commit
            or meta.get("measurement_status") != "measured"
            or meta.get("scalar_only") is not True
            or meta.get("dtype") != "bfloat16" or meta.get("quantization") != "awq_marlin"
            or not all(meta.get(f) for f in ("driver_version", "python_version", "timestamp_utc",
                                           "profiler_command", "raw_sha256"))):
        raise ValueError("Invalid/incomplete profile provenance")
    if profile.get("models") != dict(target=TARGET, target_revision=TARGET_REVISION,
                                     draft=SPECULATOR, draft_revision=SPECULATOR_REVISION):
        raise ValueError("Profile model identity mismatch")
    expected = {f"k{k}_b{b}" for k in KS for b in BATCHES}
    if set(profile["cost_matrix_ms"]) != expected:
        raise ValueError("Incomplete or non-scalar profile grid")
    for value in profile["cost_matrix_ms"].values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid measured milliseconds")
    model = FlexCostModel.from_json(profile)
    for k in KS:
        for b in BATCHES:
            if model.matrix[(k, b)] != profile["cost_matrix_ms"][f"k{k}_b{b}"]:
                raise ValueError("Profile loader lookup drift")
    return model


def build_profile(env: dict, records: list[dict], raw_hash: str) -> dict:
    hardware_gate(env)
    # Existing validator checks exact warmup/repeat coverage, shape, identity,
    # finite timings, source hashes, and failures; never fits missing cells.
    prof.validate_portable_evidence(env, records)
    for record in records:
        b, k = record["batch_size"], record["K"]
        if not any(s.get("actual_batch_size") == b
                   and s.get("query_tokens_per_request") == [k + 1] * b
                   and s.get("context_lengths") == [1024] * b
                   and not s.get("preempted_req_ids") for s in record.get("observed_shapes", [])):
            raise ValueError("Profile measurement lacks exact observed B/K/context shape")
    matrix = {}
    for b in BATCHES:
        for k in KS:
            measured = [r for r in records if not r["is_warmup"] and r["batch_size"] == b and r["K"] == k]
            grouped = {}
            for r in measured:
                grouped.setdefault(r["repeat"], []).append(r["latency_ms"])
            if len(grouped) != env["repeats"] or any(len(v) != (2 if k else 1) for v in grouped.values()):
                raise ValueError("Incomplete paired component measurements")
            matrix[f"k{k}_b{b}"] = median(sum(v) for v in grouped.values())
    profile = dict(metadata=dict(env, sm_architecture="sm_80", measurement_status="measured",
                                 scalar_only=True,
                                 raw_sha256=raw_hash, profiler_command="python -m experiments.m4_a100_profile measure",
                                 vram_total_mb=env["gpu_memory_total_mb"]),
                   models=dict(target=TARGET, target_revision=TARGET_REVISION,
                               draft=SPECULATOR, draft_revision=SPECULATOR_REVISION),
                   grid=dict(batches=list(BATCHES), candidate_budgets=list(KS), contexts=[1024]),
                   cost_matrix_ms=matrix,
                   cost_semantics="measured target forward plus EAGLE proposal; excludes target logits/sampling",
                   profile_consumption="AdaptiveDraftController uses acceptance/KV/VRAM rules, not this cost matrix",
                   measurement_summary=prof.summarize(records))
    validate_profile(profile, env["source_commit"])
    return profile


def validate_profile_file(path: Path, source_commit: str):
    """A 'measured' label alone is insufficient: reproduce from paired raw data."""
    profile = json.loads(path.read_text())
    validate_profile(profile, source_commit)
    raw = path.parent / "raw.jsonl"
    env = json.loads((path.parent / "env.json").read_text())
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    if profile != build_profile(env, records, sha256(raw.read_bytes())):
        raise ValueError("Profile cannot be reproduced from preserved raw measurements")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("environment", "measure", "validate"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-k", type=int)
    parser.add_argument("--worker-batch", type=int)
    args = parser.parse_args()
    if args.action == "validate":
        validate_profile_file(args.out, args.source_commit)
        return
    source_gate(args.source_commit)
    if args.worker_k is not None:
        from experiments.m4_cost_runner import run_worker
        args.worker_context = 1024
        raise SystemExit(run_worker(args, args.out))
    import platform
    env = dict(inspect_environment(), python_version=platform.python_version(), source_commit=args.source_commit,
               timestamp_utc=prof.utc_now())
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / "environment.json", env)
    hardware_gate(env, installed=args.action != "environment")
    if args.action == "environment":
        return
    env.update(canonical_hardware=True, expected_gpu="A100", run_id=args.out.name,
               run_type="component_benchmark", working_tree_clean=True,
               sm_architecture="sm_80", target_model=TARGET, target_revision=TARGET_REVISION,
               draft_model=SPECULATOR, draft_revision=SPECULATOR_REVISION, draft_method="eagle3",
               dtype="bfloat16", quantization="awq_marlin", timing_method=prof.TIMING,
               contexts=[1024], batches=list(BATCHES), batch_grid=list(BATCHES),
               k_values=list(KS), k_grid=list(KS), warmup=5, repeats=50,
               engine_lifecycle="fresh_process_per_k_batch_context")
    write_new(args.out / "env.json", env)
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw:
        records = []
        for k in KS:
            for b in BATCHES:
                from experiments.m4_runtime import gpu_snapshot, require_gpu_release
                before = gpu_snapshot()
                if before["processes"]:
                    raise RuntimeError("Profile requires a clean GPU before each worker")
                command = [sys.executable, "-m", "experiments.m4_a100_profile", "measure",
                           "--source-commit", args.source_commit, "--out", str(args.out),
                           "--worker-k", str(k), "--worker-batch", str(b)]
                with (args.out / f"k{k}-b{b}.log").open("x", encoding="utf-8") as log:
                    child = subprocess.run(command, cwd=ROOT,
                        env=dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="0"), stdout=log, stderr=subprocess.STDOUT)
                after = gpu_snapshot()
                write_new(args.out / f"k{k}-b{b}-lifecycle.json", dict(gpu_before=before, gpu_after=after,
                          returncode=child.returncode, timestamp_utc=datetime.now(timezone.utc).isoformat()))
                require_gpu_release(before, after)
                if child.returncode:
                    raise RuntimeError("Profile worker failed; retain raw/logs, assess before retry")
                part = args.out / f"k{k}-b{b}-c1024" / "raw.jsonl"
                for line in part.read_text().splitlines():
                    records.append(json.loads(line)); raw.write(line + "\n")
                raw.flush(); os.fsync(raw.fileno())
    profile = build_profile(env, records, sha256((args.out / "raw.jsonl").read_bytes()))
    write_new(args.out / "profile_sm80.json", profile)


if __name__ == "__main__":
    main()
