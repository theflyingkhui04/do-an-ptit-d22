"""M4 component evidence; canonical profile construction awaits alignment.

Headline target boundary is model forward (including attention, excluding logits
and sampling). Proposer boundary is the real EAGLE propose call, including its
input preparation and sampling. Both are synchronized in the worker process.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiments import modal_lanes as lanes
from experiments.modal_profiler import prompt_token_ids, RUN_ID_PATTERN
from src.flexsds.core.tree_topology import CANONICAL_TOPOLOGIES

BATCHES = (1, 2, 4, 8, 16, 32)
KS = (1, 2, 3, 4, 5)
CONTEXT = 1024
TIMING = "worker_cuda_synchronize_perf_counter_wall_ms"
LIMITATION = ("Canonical arbitrary-tree topologies are represented in the cost/controller "
              "layer but are not executed by the current vLLM serving/proposer path.")
GPU_CONFIG = {"L4": ("l4-bf16", "sm_89"), "T4": ("fp16-fallback", "sm_75")}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def evidence_dir(root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id or "") or ".." in run_id:
        raise ValueError("Invalid run_id")
    return root / "experiments" / "runs" / "m4_cost_profiling" / run_id


def validate_config(gpu, batches, ks, context, warmup, repeats, stage):
    if gpu not in GPU_CONFIG or stage not in ("smoke", "pilot", "full"):
        raise ValueError("Unsupported GPU or stage")
    for values, allowed in ((batches, BATCHES), (ks, (0, *KS))):
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError("Invalid/duplicate grid")
    if not 1 <= context <= 3072 or warmup < 5 or repeats < 1:
        raise ValueError("Context/warmup/repeats outside protocol")
    if stage == "full" and (tuple(batches) != BATCHES or tuple(ks) != (0, *KS) or repeats < 50):
        raise ValueError("Full campaign requires complete grid and >=50 repeats")


def engine_kwargs(gpu: str, k: int):
    profile, _ = GPU_CONFIG[gpu]
    result = dict(model=lanes.TARGET_MODEL, revision=lanes.TARGET_REVISION,
                  **lanes.DTYPE_PROFILES[profile], max_model_len=4096,
                  max_num_seqs=32, max_num_batched_tokens=4096,
                  gpu_memory_utilization=0.85, enable_prefix_caching=False,
                  generation_config="vllm", enforce_eager=True,
                  compilation_config={"mode": 0}, async_scheduling=False,
                  worker_extension_cls="experiments.m4_worker_observer.M4ObserverExtension",
                  enable_chunked_prefill=False)
    if k:
        result["speculative_config"] = dict(lanes.LANES["eagle"].speculative_config,
                                             revision=lanes.EAGLE_REVISION,
                                             num_speculative_tokens=k)
    return result


def validate_hardware(gpu, name, architecture):
    if gpu not in GPU_CONFIG or gpu not in name.split() or architecture != GPU_CONFIG[gpu][1]:
        raise ValueError(f"Hardware mismatch: requested {gpu}, observed {name}/{architecture}")


def timed(operation, synchronize, clock):
    synchronize()
    start = clock()
    result = operation()
    synchronize()
    elapsed = (clock() - start) * 1000
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Nonpositive/nonfinite timing")
    return result, elapsed


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["measurement_kind"], r["batch_size"], r["K"], r["context_len"])].append(r)
    rows = []
    for (kind, batch, k, context), group in sorted(groups.items()):
        values = [r["latency_ms"] for r in group if not r["is_warmup"] and r["status"] == "ok"]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Invalid successful latency")
        row = dict(measurement_kind=kind, batch_size=batch, K=k, context_len=context,
                   n_success=len(values), n_failed=sum(r["status"] != "ok" for r in group),
                   n_warmup=sum(r["is_warmup"] for r in group))
        if values:
            med = statistics.median(values)
            ratio = (max(values) - min(values)) / med
            row.update(median_ms=med, p90_ms=sorted(values)[math.ceil(.9 * len(values)) - 1],
                       min_ms=min(values), max_ms=max(values), variance_ratio=ratio,
                       variance_warning=ratio > .15)
        rows.append(row)
    return rows


ALIGNMENT_REQUIRED = (
    "TEAM TECHNICAL ALIGNMENT REQUIRED: TreeAdaptive N/64 versus "
    "FlexCostModel max(0,N-1) verification formulas and derived-cost provenance. "
    "Canonical profile generation and pilot/full submission are disabled."
)


def validate_submission(gpu, batches, ks, context, warmup, repeats, stage):
    """Execution gate during reconciliation: one B/K smoke cell only."""
    validate_config(gpu, batches, ks, context, warmup, repeats, stage)
    if stage != "smoke" or len(batches) != 1 or len(ks) != 1 or repeats > 5:
        raise ValueError(ALIGNMENT_REQUIRED)


def validate_component_evidence(env: dict, records: list[dict]):
    """Validate the declared evidence grid without creating a consumer profile."""
    validate_config(env["requested_gpu"], env["batch_grid"], env["k_grid"],
                    env["context_len"], env["warmup"], env["repeats"], env["stage"])
    validate_hardware(env["requested_gpu"], env["gpu"], env["sm_architecture"])
    if env["vllm_version"] != lanes.VLLM_VERSION:
        raise ValueError("Unpinned vLLM")
    if (env["target_revision"], env["draft_revision"]) != (lanes.TARGET_REVISION, lanes.EAGLE_REVISION):
        raise ValueError("Unpinned model revisions")
    identities = set()
    expected = set()
    for b in env["batch_grid"]:
        for k in env["k_grid"]:
            kinds = ("linear_verify_forward", "eagle_propose") if k else ("target_decode_forward",)
            for kind in kinds:
                for is_warmup, count in ((True, env["warmup"]), (False, env["repeats"])):
                    expected.update((kind, b, k, repeat, is_warmup) for repeat in range(count))
    for r in records:
        identity = (r["measurement_kind"], r["batch_size"], r["K"], r["repeat"], r["is_warmup"])
        if identity in identities:
            raise ValueError("Duplicate repeat")
        identities.add(identity)
        if r["status"] != "ok":
            raise ValueError("Failed component cell retained; report is incomplete")
        if r["context_len"] != env["context_len"]:
            raise ValueError("Mixed contexts")
        for field in ("run_id", "source_commit", "gpu", "sm_architecture", "target_revision", "draft_revision"):
            if r[field] != env[field]:
                raise ValueError(f"Mixed provenance: {field}")
        if r["timing_method"] != TIMING:
            raise ValueError("Mixed timing boundaries")
    if identities != expected:
        raise ValueError("Incomplete/unexpected evidence grid or repeat indices")
    return summarize(records)


def build_component_report(env: dict, records: list[dict]):
    """Measured component diagnostics only: NO topology matrix or gamma fit.

    Linear proposer OLS characterizes sequential K work, without extrapolating
    to branches or claiming that K1 proposal equals a draft-model forward.
    """
    rows = validate_component_evidence(env, records)
    proposal_fits = {}
    for b in env["batch_grid"]:
        points = [(r["K"], r["median_ms"]) for r in rows
                  if r["batch_size"] == b and r["measurement_kind"] == "eagle_propose"]
        if len(points) < 2:
            continue
        xmean = statistics.mean(k for k, _ in points)
        ymean = statistics.mean(ms for _, ms in points)
        slope = sum((k-xmean)*(ms-ymean) for k, ms in points) / sum((k-xmean)**2 for k, _ in points)
        intercept = ymean-slope*xmean
        proposal_fits[str(b)] = dict(intercept_ms=intercept, marginal_step_ms=slope,
                                    rmse_ms=math.sqrt(statistics.mean((ms-intercept-slope*k)**2 for k, ms in points)),
                                    n_points=len(points), model="linear EAGLE K calibration only")
    return dict(schema_version="m4-component-calibration-v1", artifact_kind="component_report",
                not_consumer_profile=True, canonical_profile_generated=False,
                runtime_tree_available=False, scientific_limitation=LIMITATION,
                technical_alignment_status=ALIGNMENT_REQUIRED,
                gamma_attn_status="not_calibrated_pending_formula_alignment",
                environment=env, measurement_summary=rows, linear_proposer_fits=proposal_fits,
                full_repeat_requirement_met=env["stage"] == "full" and env["repeats"] >= 50,
                cost_semantics={"component_latencies": "measured",
                                "proposal_fits": "empirical_linear_regression"})


def build_profile(*args, **kwargs):
    """Fail closed: current consumer treats any loaded matrix as measured."""
    raise ValueError(ALIGNMENT_REQUIRED)


def validate_consumer_lookup(profile):
    """Read-only compatibility check, not authorization to publish a profile."""
    from src.flexsds.core.cost_model import FlexCostModel
    expected = {f"topo_{t}_b{b}" for t in CANONICAL_TOPOLOGIES for b in BATCHES}
    if set(profile["cost_matrix_ms"]) != expected:
        raise ValueError("Exactly 30 canonical cells required")
    model = FlexCostModel.from_json(profile)
    for t in CANONICAL_TOPOLOGIES:
        for b in BATCHES:
            value = profile["cost_matrix_ms"][f"topo_{t}_b{b}"]
            if not math.isfinite(value) or value <= 0 or model.matrix[(t,b)] != value:
                raise ValueError("Invalid cost or missing exact lookup")
            if model.get_step_cost(t,b) != value:
                raise ValueError("Lookup drift")
    return model
