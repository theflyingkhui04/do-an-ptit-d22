"""M4 cost evidence and profile construction; imports without CUDA or Modal.

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


def build_profile(env: dict, records: list[dict], *, tree_ops_prior_ms: float):
    """Require complete full-campaign evidence. Never fill a failed/missing cell.

    Width is uncalibrated. Proposal cost uses empirical linear K=tree depth,
    with no width multiplier. This is a depth-only proxy, NOT an upper bound.
    Tree bookkeeping is an explicit caller-supplied prior, never measured.
    """
    if not math.isfinite(tree_ops_prior_ms) or tree_ops_prior_ms < 0:
        raise ValueError("Invalid tree operations prior")
    validate_config(env["requested_gpu"], env["batch_grid"], env["k_grid"],
                    env["context_len"], env["warmup"], env["repeats"], env["stage"])
    if env["stage"] != "full":
        raise ValueError("Smoke/pilot evidence is not canonical")
    validate_hardware(env["requested_gpu"], env["gpu"], env["sm_architecture"])
    if env["vllm_version"] != lanes.VLLM_VERSION:
        raise ValueError("Unpinned vLLM")
    if (env["target_revision"], env["draft_revision"]) != (lanes.TARGET_REVISION, lanes.EAGLE_REVISION):
        raise ValueError("Unpinned model revisions")
    rows = summarize(records)
    lookup = {(r["measurement_kind"], r["batch_size"], r["K"]): r for r in rows}
    if any(r["context_len"] != env["context_len"] for r in records):
        raise ValueError("Mixed contexts")
    identities = set()
    for r in records:
        identity = (r["measurement_kind"], r["batch_size"], r["K"], r["repeat"], r["is_warmup"])
        if identity in identities:
            raise ValueError("Duplicate repeat")
        identities.add(identity)
        for field in ("run_id", "source_commit", "gpu", "sm_architecture", "target_revision", "draft_revision"):
            if r[field] != env[field]:
                raise ValueError(f"Mixed provenance: {field}")
        if r["timing_method"] != TIMING:
            raise ValueError("Mixed timing boundaries")

    def cell(kind, b, k):
        r = lookup.get((kind, b, k))
        if not r or r["n_failed"] or r["n_success"] != env["repeats"] or r["n_warmup"] != env["warmup"]:
            raise ValueError(f"Incomplete/failed cell: {kind}, B={b}, K={k}")
        expected = set(range(env["repeats"]))
        observed = {x["repeat"] for x in records if x["measurement_kind"] == kind and
                    x["batch_size"] == b and x["K"] == k and not x["is_warmup"]}
        if observed != expected:
            raise ValueError("Missing repeat indices")
        return r["median_ms"]

    target, draft, linear, fits = {}, {}, {}, {}
    matrix, verification, provenance = {}, {}, {}
    for b in BATCHES:
        ct = cell("target_decode_forward", b, 0)
        target[str(b)] = ct
        points = []
        for k in KS:
            cd = cell("eagle_propose", b, k)
            cv = cell("linear_verify_forward", b, k)
            linear[f"linear_eagle_k{k}_b{b}"] = dict(propose_ms=cd, verify_forward_ms=cv,
                                                     effective_target_tokens=b * (k + 1), source="measured")
            points.append((k, cd, cv))
        # OLS diagnostics characterize fixed/marginal sequential proposal work.
        xmean = statistics.mean(p[0] for p in points)
        ymean = statistics.mean(p[1] for p in points)
        slope = sum((k-xmean)*(cd-ymean) for k, cd, _ in points) / sum((k-xmean)**2 for k, _, _ in points)
        intercept = ymean - slope*xmean
        rmse = math.sqrt(statistics.mean((cd-intercept-slope*k)**2 for k, cd, _ in points))
        # Fit the EXISTING project verification approximation, anchored at CT.
        # K=1 contributes a residual but no slope information. No universal law.
        gamma = max(0.0, sum((k-1)*(cv/ct-1) for k, _, cv in points) /
                    sum((k-1)**2 for k, _, _ in points))
        fits[str(b)] = dict(proposal_intercept_ms=intercept, proposal_slope_ms=slope,
                            proposal_rmse_ms=rmse, gamma_attn=gamma,
                            verification_rmse_ms=math.sqrt(statistics.mean(
                                (cv-ct*(1+gamma*(k-1)))**2 for k, _, cv in points)), n_points=5)
        draft[str(b)] = linear[f"linear_eagle_k1_b{b}"]["propose_ms"]
    # Consumer supports one scalar coefficient: least-squares in milliseconds
    # over all batches, rather than silently using a batch-specific coefficient.
    denom = sum((target[str(b)]*(k-1))**2 for b in BATCHES for k in KS)
    gamma = max(0., sum(target[str(b)]*(k-1)*(linear[f"linear_eagle_k{k}_b{b}"]["verify_forward_ms"]-target[str(b)])
                        for b in BATCHES for k in KS) / denom)
    residuals = [linear[f"linear_eagle_k{k}_b{b}"]["verify_forward_ms"] -
                 target[str(b)]*(1+gamma*(k-1)) for b in BATCHES for k in KS]
    for name, topo in CANONICAL_TOPOLOGIES.items():
        for b in BATCHES:
            key = f"topo_{name}_b{b}"
            ct = target[str(b)]
            if name == "T0":
                matrix[key] = ct
                provenance[key] = dict(source="measured", measurement_kind="target_decode_forward")
            else:
                cv = ct*(1+gamma*max(0, topo.num_tokens-1))
                cd = linear[f"linear_eagle_k{topo.depth}_b{b}"]["propose_ms"]
                verification[key] = cv
                matrix[key] = tree_ops_prior_ms + cd + cv
                provenance[key] = dict(source="derived_model_assisted", proposal_proxy_k=topo.depth,
                                       proposal_width_calibrated=False, verification_source="modeled",
                                       verification_extrapolated=topo.num_tokens > max(KS),
                                       tree_ops_source="explicit_prior", tree_ops_ms=tree_ops_prior_ms)
    result = dict(schema_version="m4-cost-profile-v1", metadata={
        **env, "device": env["gpu"], "source_run_id": env["run_id"],
        "profile_provenance": "derived_model_assisted", "gamma_attn": gamma,
        "attn_verify_factor": gamma, "gamma_attn_source": "measured_fit",
        "gamma_attn_fit": dict(method="nonnegative anchored OLS in ms; x=CT(B)*(K-1)",
                               n_points=30, rmse_ms=math.sqrt(statistics.mean(r*r for r in residuals)),
                               residuals_ms=residuals),
        "runtime_tree_available": False, "scientific_limitation": LIMITATION,
        "proposal_model": "empirical linear K=depth proxy; width uncalibrated; not a conservative bound",
        "tree_ops_prior_ms": tree_ops_prior_ms,
        "aggregation": "per-cell median at one context; all post-warmup repeats retained",
        "cost_semantics": dict(target_c_t="measured", draft_c_d="measured_eagle_k1_propose_call",
                               eagle_linear_calibration="measured", gamma_attn="measured_fit",
                               verification_costs_ms="modeled", cost_matrix_ms="derived_model_assisted")},
        models=dict(target=lanes.TARGET_MODEL, draft=lanes.EAGLE_DRAFT_MODEL),
        grid=dict(batches=list(BATCHES), topologies=list(CANONICAL_TOPOLOGIES), contexts=[env["context_len"]]),
        forward_costs_median_ms=dict(target_c_t=target, draft_c_d=draft),
        eagle_linear_calibration=linear, calibration_fits=fits, verification_costs_ms=verification,
        cost_matrix_ms=matrix, cell_provenance=provenance, measurement_summary=rows)
    validate_profile(result)
    return result


def validate_profile(profile):
    from src.flexsds.core.cost_model import FlexCostModel
    expected = {f"topo_{t}_b{b}" for t in CANONICAL_TOPOLOGIES for b in BATCHES}
    if set(profile["cost_matrix_ms"]) != expected or set(profile["cell_provenance"]) != expected:
        raise ValueError("Exactly 30 canonical cells with provenance required")
    model = FlexCostModel.from_json(profile)
    for t in CANONICAL_TOPOLOGIES:
        for b in BATCHES:
            key = f"topo_{t}_b{b}"
            value = profile["cost_matrix_ms"][key]
            if not math.isfinite(value) or value <= 0 or model.matrix[(t, b)] != value:
                raise ValueError("Invalid cost or missing exact lookup")
            if model.get_step_cost(t, b) != value:
                raise ValueError("Lookup drift")
            if t == "T0":
                if value != profile["forward_costs_median_ms"]["target_c_t"][str(b)]:
                    raise ValueError("T0 must equal measured baseline")
            elif profile["cell_provenance"][key]["source"] != "derived_model_assisted":
                raise ValueError("Unavailable tree execution cannot be measured")
