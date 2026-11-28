"""vLLM-free budget planning logic behind the Lane 3 adaptive scheduler.

The vLLM scheduler subclass (``src.server.adaptive_scheduler``) only forwards
runtime signals to this planner, so every decision rule here can be unit-tested
without a GPU or a vLLM install.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import BudgetDecision

# Free VRAM value passed to the controller when no reading is available,
# so the VRAM guard never fires on a missing signal.
NO_VRAM_READING_MB = 1.0e9
K_SELECTION_MODES = frozenset({"max_k", "batch"})
DEFAULT_PROBE_INTERVAL = 20
logger = logging.getLogger(__name__)


@dataclass
class StepPlan:
    """Planned nodes and scalar budgets; neither proves tree execution.

    Existing batch_k/per_request_k retain their capped scalar semantics.
    Decisions are detached snapshots of the project-owned controller types.
    """
    batch_k: int
    per_request_k: Dict[str, int] = field(default_factory=dict)
    batch_node_count: int = 0
    per_request_node_count: Dict[str, int] = field(default_factory=dict)
    per_request_decisions: Dict[str, BudgetDecision] = field(default_factory=dict)
    plan_sequence: Optional[int] = None


class AdaptiveBudgetPlanner:
    """Turns per-request AdaptiveDraftController decisions into a batch plan."""

    def __init__(
        self,
        adapter: Optional[AdaptiveProposerAdapter] = None,
        *,
        max_k: int = 4,
        min_batch_k: int = 0,
        per_request: bool = True,
        telemetry_path: Optional[str] = None,
        step_telemetry_path: Optional[str] = None,
        step_telemetry_flush_every: int = 1,
        hardware_profile: Optional[dict] = None,
        k_selection: str = "max_k",
        probe_alpha_cap: float = 0.80,
    ):
        if k_selection not in K_SELECTION_MODES:
            raise ValueError(f"k_selection must be one of {sorted(K_SELECTION_MODES)}: {k_selection!r}")
        if (
            isinstance(probe_alpha_cap, bool)
            or not isinstance(probe_alpha_cap, (int, float))
            or not math.isfinite(probe_alpha_cap)
            or not 0.0 < probe_alpha_cap <= 1.0
        ):
            raise ValueError("probe_alpha_cap must be finite and in (0, 1]")
        if max_k < 0 or min_batch_k < 0:
            raise ValueError("max_k and min_batch_k must be non-negative")
        if (
            isinstance(step_telemetry_flush_every, bool)
            or not isinstance(step_telemetry_flush_every, int)
            or step_telemetry_flush_every < 1
        ):
            raise ValueError("step_telemetry_flush_every must be a positive integer")
        if (telemetry_path and step_telemetry_path
                and os.path.normcase(os.path.realpath(telemetry_path))
                == os.path.normcase(os.path.realpath(step_telemetry_path))):
            raise ValueError("Step and aggregate telemetry must use separate paths")
        self.adapter = adapter or AdaptiveProposerAdapter()
        self._max_k = max_k
        self.min_batch_k = min_batch_k
        self.per_request = per_request
        self.telemetry_path = telemetry_path
        self.step_telemetry_path = step_telemetry_path
        self.step_telemetry_flush_every = step_telemetry_flush_every
        self.hardware_profile = deepcopy(hardware_profile) if hardware_profile else None
        self._known_requests: Set[str] = set()
        self._last_decisions: Dict[str, dict] = {}
        self._scheduled_runtime_modes: Dict[str, str] = {}
        self._plan_sequence = 0
        self.k_selection = k_selection
        self._probe_alpha_cap = float(probe_alpha_cap)
        # In batch mode probing belongs to the planner: per-request probes would drag the whole batch into a
        # drafting step (batch K = max(k_i)), so the controllers' own probing is disabled.
        self._batch_probe_interval = int(self.adapter.controller_kwargs.get("probe_interval", DEFAULT_PROBE_INTERVAL))
        self._batch_fallback_steps = 0
        profile = self.adapter.controller_kwargs.get("cost_profile")
        if k_selection == "batch" and bool(getattr(profile, "matrix", None)):
            self.adapter.controller_kwargs["probe_interval"] = 0
        self._step_buffer: List[str] = []
        self._step_fh = None
        self._step_io_state: Dict[str, object] = {"fh": None, "buf": self._step_buffer, "path": step_telemetry_path}
        self._finalizer = weakref.finalize(self, _finalize_step_io, self._step_io_state)
        self._sync_candidate_budgets_to_max_k()

    @property
    def max_k(self) -> int:
        return self._max_k

    @max_k.setter
    def max_k(self, value: int) -> None:
        if value < 0:
            raise ValueError("max_k must be non-negative")
        self._max_k = value
        self._sync_candidate_budgets_to_max_k()

    def _sync_candidate_budgets_to_max_k(self) -> None:
        """Keep cost-model candidate budgets within the executable max_k ceiling."""
        budgets = self.adapter.controller_kwargs.get("candidate_budgets")
        if budgets is not None:
            filtered = sorted({0} | {k for k in budgets if 0 <= k <= self._max_k})
            self.adapter.controller_kwargs["candidate_budgets"] = filtered
            for controller in self.adapter._active_sessions.values():
                if hasattr(controller, "candidate_budgets"):
                    controller.candidate_budgets = list(filtered)

    def plan_step(
        self,
        decode_req_ids: Iterable[str],
        batch_size: int,
        kv_usage: float,
        free_vram_mb: Optional[float] = None,
    ) -> StepPlan:
        """Selects k_i for every decoding request and the batch-wide K = max(k_i).

        kv_usage is the vLLM fraction in [0, 1] (Scheduler.get_kv_cache_usage()).
        """
        kv_pct = 100.0 * kv_usage
        vram = NO_VRAM_READING_MB if free_vram_mb is None else free_vram_mb

        per_request_k: Dict[str, int] = {}
        nodes: Dict[str, int] = {}
        decisions: Dict[str, BudgetDecision] = {}
        alpha_signals: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
        for req_id in decode_req_ids:
            self._known_requests.add(req_id)
            controller = self.adapter.get_controller(req_id)
            decision = controller.select_budget(
                batch_size=batch_size,
                free_vram_mb=vram,
                kv_cache_usage_pct=kv_pct,
            )
            decisions[req_id] = deepcopy(decision)
            effective_alpha_fn = getattr(controller, "get_effective_alpha", None)
            geometric_alpha_fn = getattr(controller, "get_geometric_alpha", None)
            geom_alpha = geometric_alpha_fn() if callable(geometric_alpha_fn) else None
            estimator = getattr(decision, "alpha_estimator", None)
            if estimator == "geometric_mle" and geom_alpha is not None:
                eff_alpha = geom_alpha
            else:
                eff_alpha = effective_alpha_fn() if callable(effective_alpha_fn) else None
            alpha_signals[req_id] = (getattr(controller, "alpha_ema", None), eff_alpha, geom_alpha)
            nodes[req_id] = max(0, getattr(decision, "num_nodes", decision.k))
            per_request_k[req_id] = min(nodes[req_id], self.max_k)

        if not per_request_k:
            return StepPlan(batch_k=0, per_request_k={})

        batch = self._select_batch_k(decisions, batch_size) if self.k_selection == "batch" else None
        batch_probe = False
        if batch is not None and batch["best_k"] > 0:
            self._batch_fallback_steps = 0
        if batch is None:
            batch_k = max(max(per_request_k.values()), self.min_batch_k)
            batch_k = min(batch_k, self.max_k)
            if not self.per_request:
                per_request_k = {req_id: batch_k for req_id in per_request_k}
        else:
            batch_k = min(max(batch["best_k"], self.min_batch_k), self.max_k)
            batch_probe = batch["best_k"] == 0 and self._probe_due(batch)
            if batch_probe:
                batch_k = batch["probe_k"]
            guard_cap = {
                req_id: decision.k if decision.decision_policy == "memory_guard" else self.max_k
                for req_id, decision in decisions.items()
            }
            per_request_k = {req_id: min(batch_k, guard_cap[req_id]) for req_id in per_request_k}
            batch_k = max(per_request_k.values())
            nodes = dict(per_request_k)
        batch_node_count = max(nodes.values())
        self._plan_sequence += 1
        for req_id, decision in decisions.items():
            raw_alpha, effective_alpha, geometric_alpha = alpha_signals[req_id]
            record = _decision_record(decision, nodes[req_id], batch_size, kv_pct, free_vram_mb,
                                      raw_alpha=raw_alpha, effective_alpha=effective_alpha,
                                      geometric_alpha=geometric_alpha,
                                      hardware_profile=self.hardware_profile)
            record.update(request_id=req_id, lane="adaptive", record_type="planned_decision",
                          controller_class=f"{type(self.adapter.get_controller(req_id)).__module__}.{type(self.adapter.get_controller(req_id)).__name__}",
                          plan_sequence=self._plan_sequence, scalar_runtime_budget=per_request_k[req_id],
                          batch_scalar_budget=batch_k, batch_node_count=batch_node_count,
                          batch_k_selected=batch_k if batch is not None else None,
                          batch_score=batch["score"] if batch is not None else None,
                          batch_score_k0=batch["score_k0"] if batch is not None else None,
                          batch_probe=batch_probe)
            self._last_decisions[req_id] = record
            self._write_step(record)
        return StepPlan(batch_k=batch_k, per_request_k=per_request_k,
                        batch_node_count=batch_node_count, per_request_node_count=nodes,
                        per_request_decisions=decisions, plan_sequence=self._plan_sequence)

    def _select_batch_k(self, decisions: Dict[str, BudgetDecision], batch_size: int) -> Optional[dict]:
        """One K for the whole batch: argmax_K of sum_i E[tau_i(K)] / C(K, B) (None without a cost matrix)."""
        controllers = {req_id: self.adapter.get_controller(req_id) for req_id in decisions}
        first = next(iter(controllers.values()))
        profile = getattr(first, "cost_profile", None)
        if not bool(getattr(profile, "matrix", None)) or not callable(getattr(first, "decision_alpha", None)):
            return None  # no measured cost table or another controller type: keep the per-request path
        budgets = sorted({0} | {k for k in first.candidate_budgets if 0 < k <= self.max_k})
        costs = {k: profile.get_cost(k, batch_size) for k in budgets}
        c_ref = min(costs.values())  # K=0 is costed at C_ref(B), as in the per-request rule (noisy cells C(k,B) < C(0,B))
        alphas = [controller.decision_alpha()[0] for controller in controllers.values()]
        score = {k: sum(first.estimate_expected_tokens(k, a) for a in alphas) / costs[k] for k in budgets}
        score[0] = len(alphas) / c_ref
        best_k = 0
        for k in budgets[1:]:
            if score[k] >= getattr(first, 'min_speedup_ratio', 1.05) * score[0] and score[k] > score[best_k] + 1e-9:
                best_k = k
        guarded = any(d.decision_policy == "memory_guard" for d in decisions.values())
        return {"best_k": best_k, "score": score[best_k], "score_k0": score[0], "budgets": budgets,
                "costs": costs, "c_ref": c_ref, "controller": first, "guarded": guarded}

    def _probe_due(self, batch: dict) -> bool:
        """Counts batch-wide K=0 steps and decides whether this step may probe (sets batch['probe_k'])."""
        if batch["guarded"] or self._batch_probe_interval <= 0:
            return False
        self._batch_fallback_steps = min(self._batch_fallback_steps + 1, self._batch_probe_interval)
        if self._batch_fallback_steps < self._batch_probe_interval:
            return False
        feasible = [
            (star, k)
            for k in batch["budgets"][1:]
            if (star := batch["controller"].solve_break_even_alpha(k, batch["costs"][k] / batch["c_ref"])) is not None
            and star <= self._probe_alpha_cap
        ]
        if not feasible:
            return False
        batch["probe_k"] = min(feasible)[1]
        self._batch_fallback_steps = 0
        return True

    def record_scheduled_budget(self, req_id: str, batch_k: int, placeholder_k: int,
                                *, dynamic: bool, schedule_sequence: int,
                                plan_sequence: Optional[int]) -> None:
        """Observe this output's proposer K and placeholders for future verification.

        K is consumed by the worker handling THIS output. Its proposed tokens
        are verified by a later output; neither observation proves GPU execution.
        """
        self._known_requests.add(req_id)
        mode = "linear_dynamic_k" if dynamic else "linear_fixed_k"
        self._scheduled_runtime_modes[req_id] = mode
        if not self.step_telemetry_path:
            return
        self._write_step({"record_type": "scheduled_budget", "request_id": req_id,
                          "lane": "adaptive", "executed_runtime_mode": "unverified",
                          "scheduled_runtime_mode": mode,
                          "scheduled_scalar_k": batch_k,
                          "schedule_sequence": schedule_sequence, "plan_sequence": plan_sequence,
                          "budget_target": "drafts_for_future_verification",
                          # Compatibility aliases: 'next' refers to verification,
                          # not to the worker invocation that consumes this K.
                          "next_batch_scalar_budget": batch_k,
                          "next_request_placeholder_budget": placeholder_k,
                          "executed_draft_token_count": None, "tree_execution_verified": False})

    def _write_step(self, record: dict) -> None:
        if not self.step_telemetry_path:
            return
        try:
            line = json.dumps(dict(record, recorded_at=time.time()), allow_nan=False) + "\n"
        except ValueError:
            logger.exception("AdaptiveBudgetPlanner: step telemetry serialization failed")
            return
        self._step_buffer.append(line)
        if len(self._step_buffer) >= self.step_telemetry_flush_every:
            self.flush()

    def flush(self) -> None:
        """Flushes buffered step telemetry lines to the persistent file handle."""
        if not self._step_buffer:
            return
        if not self.step_telemetry_path:
            self._step_buffer.clear()
            return
        payload = "".join(self._step_buffer)
        self._step_buffer.clear()
        self._step_io_state["path"] = self.step_telemetry_path
        try:
            if self._step_fh is None or self._step_fh.closed:
                self._step_fh = open(self.step_telemetry_path, "a", encoding="utf-8", buffering=65536)
                self._step_io_state["fh"] = self._step_fh
            self._step_fh.write(payload)
            self._step_fh.flush()
        except OSError:
            logger.exception("AdaptiveBudgetPlanner: step telemetry write failed")
            if self._step_fh is not None:
                try:
                    self._step_fh.close()
                except OSError:
                    pass
                self._step_fh = None
                self._step_io_state["fh"] = None

    def close(self) -> None:
        """Flushes any pending step telemetry and closes the persistent handle."""
        self.flush()
        if self._step_fh is not None:
            try:
                self._step_fh.close()
            except OSError:
                logger.exception("AdaptiveBudgetPlanner: step telemetry close failed")
            finally:
                self._step_fh = None
                self._step_io_state["fh"] = None

    def record_lineage_failure(self, req_id: str, reason: str) -> None:
        logger.warning("Adaptive feedback skipped for %s: %s", req_id, reason)
        self._write_step({"record_type": "verification_feedback_skipped", "request_id": req_id,
                          "lane": "adaptive", "lineage_status": reason,
                          "source_plan_sequence": None, "source_schedule_sequence": None,
                          "tree_execution_verified": False})

    def record_verification(self, req_id: str, num_drafted: int, num_accepted: int,
                            *, source=None) -> None:
        """Feeds one verification result back to the request's controller.

        Direct planner callers supply synchronous counts. Production scheduler
        callers MUST validate execution provenance before invoking this method.
        A newly completed prefill may have an exact schedule but no decode plan.
        """
        if num_drafted <= 0:
            return
        self._known_requests.add(req_id)
        alpha_before = (self.adapter.get_telemetry(req_id)["alpha_ema"]
                        if self.step_telemetry_path else None)
        self.adapter.on_verification_completed(
            request_id=req_id,
            num_accepted=num_accepted,
            num_drafted=num_drafted,
        )
        if self.step_telemetry_path:
            controller = self.adapter.get_controller(req_id)
            accepted_clamped = min(max(int(num_accepted), 0), int(num_drafted))
            tested_positions = accepted_clamped + (1 if accepted_clamped < num_drafted else 0)
            geom_fn = getattr(controller, "get_geometric_alpha", None)
            geom_alpha = geom_fn() if callable(geom_fn) else None
            if geom_alpha is not None and not math.isfinite(geom_alpha):
                geom_alpha = None
            self._write_step({
                "record_type": "verification_feedback", "request_id": req_id,
                "lane": "adaptive", "executed_draft_token_count": num_drafted,
                "accepted_draft_token_count": num_accepted,
                "tested_draft_positions": tested_positions,
                "executed_runtime_mode": "unverified",
                "source_plan_sequence": source.plan_sequence if source else None,
                "source_schedule_sequence": source.schedule_sequence if source else None,
                "source_request_scalar_k": source.request_k if source else None,
                "source_batch_scalar_k": source.batch_k if source else None,
                "lineage_status": "attributed" if source else "unverified",
                "tree_execution_verified": False,
                "alpha_ema": self.adapter.get_telemetry(req_id)["alpha_ema"],
                "alpha_ema_before": alpha_before,
                "geometric_alpha_mle": geom_alpha,
            })

    def finish_request(self, req_id: str) -> Optional[dict]:
        """Releases the request session and appends its telemetry as one JSONL line."""
        if req_id not in self._known_requests:
            return None
        self._known_requests.discard(req_id)
        last_decision = self._last_decisions.pop(req_id, {})
        mode = self._scheduled_runtime_modes.pop(req_id, "unverified")
        telemetry = self.adapter.end_request(req_id)
        # Top-level v0.2 decision fields describe the LAST plan only. Legacy
        # alpha/history/mean_accept_len remain request aggregates.
        telemetry.update({key: value for key, value in last_decision.items()
                          if key not in telemetry and key != "record_type"})
        telemetry["last_planned_decision"] = deepcopy(last_decision) or None
        telemetry.update(record_type="request_aggregate", decision_scope="last_planned_decision",
                         scheduled_runtime_mode=mode, executed_runtime_mode="unverified",
                         executed_draft_token_count=None,
                         tree_execution_verified=False)
        self.flush()
        if self.telemetry_path:
            record = dict(telemetry)
            record["finished_at"] = time.time()
            with open(self.telemetry_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        return telemetry

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "AdaptiveBudgetPlanner":
        """Builds a planner from ADAPTIVE_* environment variables (read once at server start)."""
        env = os.environ if environ is None else environ

        raw_kwargs = env.get("ADAPTIVE_CONTROLLER_KWARGS", "").strip()
        try:
            controller_kwargs = json.loads(raw_kwargs) if raw_kwargs else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"ADAPTIVE_CONTROLLER_KWARGS is not valid JSON: {exc}") from exc
        if not isinstance(controller_kwargs, dict):
            raise ValueError("ADAPTIVE_CONTROLLER_KWARGS must be a JSON object")

        profile_path = env.get("FLEXSDS_PROFILE_PATH")
        hardware_profile: Optional[dict] = None
        if profile_path:
            import hashlib
            from pathlib import Path
            from flexsds.core.cost_model import (
                inspect_cost_matrix_monotonicity,
                validate_measured_profile,
            )
            from src.proposer.controller import EmpiricalCostProfile
            raw = Path(profile_path).read_bytes()
            data = validate_measured_profile(json.loads(raw))
            matrix = {(int(key.split('_')[0][1:]), int(key.split('_')[1][1:])): value
                      for key, value in data['cost_matrix_ms'].items()}
            controller_kwargs['cost_profile'] = EmpiricalCostProfile(matrix=matrix)
            controller_kwargs['candidate_budgets'] = data['grid']['candidate_budgets']
            digest = hashlib.sha256(raw).hexdigest()
            anomalies = inspect_cost_matrix_monotonicity(data['cost_matrix_ms'])
            below_k0 = [item for item in anomalies if item['below_k0']]
            if below_k0:
                logger.warning(
                    "Measured hardware profile %s has %d non-monotonic cost cells "
                    "(C(k, B) < C(0, B)); lower-bound reference C_ref(B) normalization is active.",
                    profile_path,
                    len(below_k0),
                )
            metadata = data.get("metadata", {})
            hardware_profile = {
                "name": Path(profile_path).name,
                "sha256": digest,
                "device": metadata.get("device"),
                "sm_architecture": metadata.get("sm_architecture"),
                "non_monotonic_cell_count": len(anomalies),
            }
            logger.info("Loaded empirical hardware profile %s sha256=%s", profile_path, digest)

        if "ADAPTIVE_ENABLE_VRAM_GUARD" in env:
            controller_kwargs["enable_vram_guard"] = _env_flag(env["ADAPTIVE_ENABLE_VRAM_GUARD"])

        raw_flush_every = env.get("ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY", "1").strip()
        try:
            flush_every = int(raw_flush_every) if raw_flush_every else 1
        except ValueError as exc:
            raise ValueError(
                f"ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY must be a positive integer: {raw_flush_every!r}"
            ) from exc

        raw_cap = env.get("ADAPTIVE_BATCH_PROBE_ALPHA_CAP", "").strip()
        return cls(
            adapter=AdaptiveProposerAdapter(controller_kwargs=controller_kwargs),
            k_selection=env.get("ADAPTIVE_K_SELECTION", "max_k").strip() or "max_k",
            probe_alpha_cap=float(raw_cap) if raw_cap else 0.80,
            max_k=int(env.get("ADAPTIVE_MAX_K", "4")),
            min_batch_k=int(env.get("ADAPTIVE_MIN_BATCH_K", "0")),
            per_request=_env_flag(env.get("ADAPTIVE_PER_REQUEST", "1")),
            telemetry_path=env.get("ADAPTIVE_TELEMETRY_PATH") or None,
            step_telemetry_path=env.get("ADAPTIVE_STEP_TELEMETRY_PATH") or None,
            step_telemetry_flush_every=flush_every,
            hardware_profile=hardware_profile,
        )


def _finalize_step_io(state: Dict[str, object]) -> None:
    """Best-effort flush and close on garbage collection / interpreter exit."""
    buf = state.get("buf")
    fh = state.get("fh")
    path = state.get("path")
    try:
        if isinstance(buf, list) and buf and path:
            payload = "".join(buf)
            buf.clear()
            if fh is None or getattr(fh, "closed", True):
                fh = open(path, "a", encoding="utf-8", buffering=65536)
            fh.write(payload)
            fh.flush()
    except OSError:
        pass
    finally:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
            state["fh"] = None


def _env_flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _decision_record(decision: BudgetDecision, nodes: int, batch_size: int,
                     kv_pct: float, free_vram_mb: Optional[float], *,
                     raw_alpha: Optional[float], effective_alpha: Optional[float],
                     geometric_alpha: Optional[float] = None,
                     hardware_profile: Optional[dict] = None) -> dict:
    """Additive Interface A v0.2 decision fields (spec §4.6).

    Never derive a topology from a scalar count. Unknown identities keep their
    name/count and null structure. Canonical linear serving has no topology;
    this module must not load the dormant future-tree catalog for telemetry.
    Unmeasured latency, hardware identity and tau are explicitly unavailable.
    """
    def finite(value: Optional[float]) -> Optional[float]:
        return value if value is not None and math.isfinite(value) else None

    name = getattr(decision, "topology_name", None)
    topology = None
    if name is not None:
        topology = {"name": name, "total_nodes_N": nodes,
                    "depth_d": None, "parent_indices": None}
    status = getattr(decision, "status", None)
    return {
        "engine_version": None,
        "hardware_profile": deepcopy(hardware_profile) if hardware_profile else None,
        "concurrency_batch_b": batch_size, "acceptance_rate_ema": finite(raw_alpha),
        "effective_alpha": finite(effective_alpha),
        "geometric_alpha_mle": finite(geometric_alpha),
        "alpha_estimator": getattr(decision, "alpha_estimator", None),
        "dynamic_alpha_star": finite(getattr(decision, "alpha_star", None)),
        "selected_topology": topology, "planned_draft_node_count": nodes,
        "mean_accepted_tokens_tau": None,
        "theoretical_speedup": finite(getattr(decision, "speedup", None)),
        "actual_step_latency_ms": None,
        "decision_policy": getattr(decision, "decision_policy", None),
        "cost_k_ms": finite(getattr(decision, "cost_k_ms", None)),
        "cost_baseline_ms": finite(getattr(decision, "cost_baseline_ms", None)),
        "cost_ref_ms": finite(getattr(decision, "cost_ref_ms", None)),
        "non_monotonic_cost": bool(getattr(decision, "non_monotonic_cost", False)),
        "memory_status": {"free_vram_mb": finite(free_vram_mb),
                          "kv_cache_usage_pct": finite(kv_pct),
                          "guard_triggered": status == "memory_guard" if status else None},
        "decision_reason": decision.reason, "executed_runtime_mode": "unverified",
        "executed_draft_token_count": None, "tree_execution_verified": False,
    }
