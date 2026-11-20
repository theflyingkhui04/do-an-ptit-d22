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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import AdaptiveDraftController, BudgetDecision

# Free VRAM value passed to the controller when no reading is available,
# so the VRAM guard never fires on a missing signal.
NO_VRAM_READING_MB = 1.0e9
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
    per_request_selected_k: Dict[str, int] = field(default_factory=dict)
    fresh_max_k: int = 0
    recovery_cap_k: int = 0
    pending_target_k: Optional[int] = None
    pending_streak: int = 0

    @property
    def worker_k(self) -> int:
        return self.batch_k


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
        replacement_policy: bool = False,
        recovery_hold_steps: int = 0,
    ):
        if max_k < 0 or min_batch_k < 0:
            raise ValueError("max_k and min_batch_k must be non-negative")
        if type(recovery_hold_steps) is not int or recovery_hold_steps < 0:
            raise ValueError("recovery_hold_steps must be a non-negative integer")
        if replacement_policy and not per_request:
            raise ValueError("The replacement policy requires per-request budgeting")
        if recovery_hold_steps and not replacement_policy:
            raise ValueError("Recovery hold requires the replacement policy")
        if (telemetry_path and step_telemetry_path
                and os.path.normcase(os.path.realpath(telemetry_path))
                == os.path.normcase(os.path.realpath(step_telemetry_path))):
            raise ValueError("Step and aggregate telemetry must use separate paths")
        self.adapter = adapter or AdaptiveProposerAdapter()
        if replacement_policy and not issubclass(self.adapter.controller_cls, AdaptiveDraftController):
            raise ValueError("The replacement policy requires the scalar adaptive controller")
        self.max_k = max_k
        self.min_batch_k = min_batch_k
        self.per_request = per_request
        self.telemetry_path = telemetry_path
        self.step_telemetry_path = step_telemetry_path
        self.replacement_policy = replacement_policy
        self.recovery_hold_steps = recovery_hold_steps
        self._known_requests: Set[str] = set()
        self._request_incarnations: Dict[str, int] = {}
        self._last_decisions: Dict[str, dict] = {}
        self._scheduled_runtime_modes: Dict[str, str] = {}
        self._plan_sequence = 0
        self.reset_recovery()

    def reset_recovery(self) -> None:
        """Clear the cap and pending increase for the next eligible cohort."""
        self._recovery_cap_k: Optional[int] = None
        self._pending_target_k: Optional[int] = None
        self._pending_streak = 0
        self._cohort: Set[Tuple[str, Optional[int]]] = set()

    def bind_request(self, req_id: str, incarnation: int) -> None:
        """Retire policy state when a different request object takes the same ID."""
        previous = self._request_incarnations.get(req_id)
        if previous is not None and previous != incarnation:
            self.finish_request(req_id)
        self._request_incarnations[req_id] = incarnation

    def _recovery_cap(self, selected: Dict[str, int], fresh_max_k: int) -> int:
        cohort = {(rid, self._request_incarnations.get(rid)) for rid in selected}
        if self._cohort.isdisjoint(cohort):
            self.reset_recovery()
        self._cohort = cohort
        if not cohort:
            return 0

        if (self._recovery_cap_k is None or fresh_max_k <= self._recovery_cap_k
                or self.recovery_hold_steps <= 1):
            self._recovery_cap_k = fresh_max_k
            self._pending_target_k, self._pending_streak = None, 0
        else:
            if self._pending_target_k != fresh_max_k:
                self._pending_target_k, self._pending_streak = fresh_max_k, 0
            self._pending_streak += 1
            if self._pending_streak >= self.recovery_hold_steps:
                self._recovery_cap_k = fresh_max_k
                self._pending_target_k, self._pending_streak = None, 0
        return self._recovery_cap_k

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
        alpha_signals: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        if self.replacement_policy and self.max_k == 0:
            decode_req_ids = ()
        for req_id in decode_req_ids:
            self._known_requests.add(req_id)
            controller = self.adapter.get_controller(req_id)
            decision = controller.select_budget(
                batch_size=batch_size,
                free_vram_mb=vram,
                kv_cache_usage_pct=kv_pct,
            )
            decisions[req_id] = deepcopy(decision)
            effective_alpha = getattr(controller, "get_effective_alpha", None)
            alpha_signals[req_id] = (getattr(controller, "alpha_ema", None),
                                     effective_alpha() if callable(effective_alpha) else None)
            nodes[req_id] = max(0, getattr(decision, "num_nodes", decision.k))
            per_request_k[req_id] = min(nodes[req_id], self.max_k)
            if self.replacement_policy:
                # A runtime maximum may cut through a sparse candidate set.
                limit = per_request_k[req_id]
                per_request_k[req_id] = max(
                    (k for k in controller.candidate_budgets if 0 < k <= limit), default=0)

        if not per_request_k and not self.replacement_policy:
            return StepPlan(batch_k=0, per_request_k={})

        selected = per_request_k
        fresh_max_k = max(selected.values(), default=0)
        if self.replacement_policy:
            cap = self._recovery_cap(selected, fresh_max_k)
            per_request_k = {rid: min(k, cap) for rid, k in selected.items()}
            batch_k = max(per_request_k.values(), default=0)
        else:
            batch_k = min(max(fresh_max_k, self.min_batch_k), self.max_k)
            cap = batch_k
            if not self.per_request:
                per_request_k = {rid: batch_k for rid in per_request_k}
        batch_node_count = max(nodes.values(), default=0)
        self._plan_sequence += 1
        for req_id, decision in decisions.items():
            raw_alpha, effective_alpha = alpha_signals[req_id]
            record = _decision_record(decision, nodes[req_id], batch_size, kv_pct, free_vram_mb,
                                      raw_alpha=raw_alpha, effective_alpha=effective_alpha)
            record.update(request_id=req_id, lane="adaptive", record_type="planned_decision",
                          controller_class=f"{type(self.adapter.get_controller(req_id)).__module__}.{type(self.adapter.get_controller(req_id)).__name__}",
                          plan_sequence=self._plan_sequence, scalar_runtime_budget=per_request_k[req_id],
                          batch_scalar_budget=batch_k, batch_node_count=batch_node_count)
            if self.replacement_policy:
                record.update(
                    request_incarnation=self._request_incarnations.get(req_id),
                    plan_id=self._plan_sequence, selected_k=selected[req_id],
                    applied_k=per_request_k[req_id], reason=decision.reason,
                    rule_alpha=getattr(decision, "rule_alpha", None),
                    warmup_active=getattr(decision, "warmup_active", False),
                    verification_sample_count=getattr(decision, "verification_sample_count", 0),
                    memory_state=getattr(decision, "memory_state", None),
                )
            self._last_decisions[req_id] = record
            self._write_step(record)
        if self.replacement_policy:
            self._write_step(dict(
                record_type="batch_policy", lane="adaptive", plan_id=self._plan_sequence,
                fresh_max_k=fresh_max_k, recovery_cap_k=cap, worker_k=batch_k,
                planned_worker_k=batch_k,
                eligible_batch_size=len(selected), pending_target_k=self._pending_target_k,
                pending_streak=self._pending_streak, required_hold_steps=self.recovery_hold_steps,
            ))
        return StepPlan(batch_k=batch_k, per_request_k=per_request_k,
                        batch_node_count=batch_node_count, per_request_node_count=nodes,
                        per_request_decisions=decisions, plan_sequence=self._plan_sequence,
                        per_request_selected_k=dict(selected), fresh_max_k=fresh_max_k,
                        recovery_cap_k=cap, pending_target_k=self._pending_target_k,
                        pending_streak=self._pending_streak)

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
        self._write_step({"record_type": "scheduled_budget", "request_id": req_id,
                          "lane": "adaptive", "executed_runtime_mode": "unverified",
                          "scheduled_runtime_mode": mode,
                          "scheduled_scalar_k": batch_k,
                          "planned_worker_k": batch_k,
                          "schedule_sequence": schedule_sequence, "plan_sequence": plan_sequence,
                          "budget_target": "drafts_for_future_verification",
                          # Compatibility aliases: 'next' refers to verification,
                          # not to the worker invocation that consumes this K.
                          "next_batch_scalar_budget": batch_k,
                          "next_request_placeholder_budget": placeholder_k,
                          "executed_draft_token_count": None, "tree_execution_verified": False})

    def _write_step(self, record: dict) -> None:
        if self.step_telemetry_path:
            try:
                line = json.dumps(dict(record, recorded_at=time.time()), allow_nan=False)
                with open(self.step_telemetry_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except (OSError, ValueError):
                logger.exception("AdaptiveBudgetPlanner: step telemetry write failed")

    def record_lineage_failure(self, req_id: str, reason: str) -> None:
        logger.warning("Adaptive feedback skipped for %s: %s", req_id, reason)
        self._write_step({"record_type": "verification_feedback_skipped", "request_id": req_id,
                          "lane": "adaptive", "lineage_status": reason,
                          "source_plan_sequence": None, "source_schedule_sequence": None,
                          "tree_execution_verified": False})

    def record_draft_birth(self, origin, width: int) -> None:
        if not self.step_telemetry_path:
            return
        self._write_step(dict(
            record_type="draft_policy_produced", lane="adaptive",
            request_id=origin.request_id, request_incarnation=origin.incarnation,
            producer_plan_id=origin.plan_sequence, producer_schedule_id=origin.schedule_sequence,
            applied_k=origin.request_k, worker_k=origin.batch_k,
            planned_worker_k=origin.planned_worker_k, physical_draft_width=width,
        ))

    def record_verification(self, req_id: str, num_drafted: int, num_accepted: int,
                            *, source=None, attributed: bool = False) -> None:
        """Feeds one verification result back to the request's controller.

        Legacy direct callers may supply synchronous counts. The replacement
        scheduler must validate and claim the exact consumed origin before
        passing attributed=True.
        """
        if num_drafted <= 0:
            return
        if self.replacement_policy and (not attributed or source is None
                or source.request_id != req_id
                or source.incarnation != self._request_incarnations.get(req_id)
                or not 0 <= num_accepted <= num_drafted <= source.request_k):
            self.record_lineage_failure(req_id, "unvalidated-policy-feedback")
            return
        self._known_requests.add(req_id)
        alpha_before = (self.adapter.get_telemetry(req_id)["alpha_ema"]
                        if self.step_telemetry_path else None)
        options = {"attributed": True} if self.replacement_policy else {}
        self.adapter.on_verification_completed(
            request_id=req_id,
            num_accepted=num_accepted,
            num_drafted=num_drafted,
            **options,
        )
        if self.step_telemetry_path:
            telemetry = self.adapter.get_telemetry(req_id)
            record = {
                "record_type": "verification_feedback", "request_id": req_id,
                "lane": "adaptive", "executed_draft_token_count": num_drafted,
                "accepted_draft_token_count": num_accepted,
                "executed_runtime_mode": "unverified",
                "source_plan_sequence": source.plan_sequence if source else None,
                "source_schedule_sequence": source.schedule_sequence if source else None,
                "source_request_scalar_k": source.request_k if source else None,
                "source_batch_scalar_k": source.batch_k if source else None,
                "lineage_status": "attributed" if source else "unverified",
                "tree_execution_verified": False,
                "alpha_ema": telemetry["alpha_ema"],
                "alpha_ema_before": alpha_before,
            }
            if self.replacement_policy:
                record.update(
                    request_incarnation=source.incarnation,
                    producer_plan_id=source.plan_sequence, producer_schedule_id=source.schedule_sequence,
                    drafted_tokens=num_drafted, accepted_tokens=num_accepted,
                    applied_k=source.request_k, worker_k=source.batch_k,
                    planned_worker_k=source.planned_worker_k,
                    physical_draft_width=source.physical_draft_width,
                    feedback_counted_for_warmup=True,
                    verification_sample_count=telemetry["verification_sample_count"],
                )
            self._write_step(record)

    def finish_request(self, req_id: str) -> Optional[dict]:
        """Releases the request session and appends its telemetry as one JSONL line."""
        incarnation = self._request_incarnations.pop(req_id, None)
        self._cohort.discard((req_id, incarnation))
        if not self._cohort:
            self.reset_recovery()
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
        replacement_policy = _env_flag(env.get("ADAPTIVE_REPLACEMENT_POLICY", "0"))
        if not replacement_policy and any(controller_kwargs.get(key) for key in ("min_k", "min_warmup_samples")):
            raise ValueError("Acceptance floor and warmup require the replacement policy")

        profile_path = env.get("FLEXSDS_PROFILE_PATH")
        if profile_path:
            import hashlib
            from pathlib import Path
            from flexsds.core.cost_model import validate_measured_profile
            from src.proposer.controller import EmpiricalCostProfile
            raw = Path(profile_path).read_bytes()
            data = validate_measured_profile(json.loads(raw))
            matrix = {(int(key.split('_')[0][1:]), int(key.split('_')[1][1:])): value
                      for key, value in data['cost_matrix_ms'].items()}
            controller_kwargs['cost_profile'] = EmpiricalCostProfile(matrix=matrix)
            controller_kwargs['candidate_budgets'] = data['grid']['candidate_budgets']
            logger.info("Loaded empirical hardware profile %s sha256=%s", profile_path,
                        hashlib.sha256(raw).hexdigest())

        if "ADAPTIVE_ENABLE_VRAM_GUARD" in env:
            controller_kwargs["enable_vram_guard"] = _env_flag(env["ADAPTIVE_ENABLE_VRAM_GUARD"])

        return cls(
            adapter=AdaptiveProposerAdapter(controller_kwargs=controller_kwargs),
            max_k=int(env.get("ADAPTIVE_MAX_K", "4")),
            min_batch_k=int(env.get("ADAPTIVE_MIN_BATCH_K", "0")),
            per_request=_env_flag(env.get("ADAPTIVE_PER_REQUEST", "1")),
            telemetry_path=env.get("ADAPTIVE_TELEMETRY_PATH") or None,
            step_telemetry_path=env.get("ADAPTIVE_STEP_TELEMETRY_PATH") or None,
            replacement_policy=replacement_policy,
            recovery_hold_steps=int(env.get("ADAPTIVE_RECOVERY_HOLD_STEPS", "0")),
        )


def _env_flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _decision_record(decision: BudgetDecision, nodes: int, batch_size: int,
                     kv_pct: float, free_vram_mb: Optional[float], *,
                     raw_alpha: Optional[float], effective_alpha: Optional[float]) -> dict:
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
        "engine_version": None, "hardware_profile": None,
        "concurrency_batch_b": batch_size, "acceptance_rate_ema": finite(raw_alpha),
        "effective_alpha": finite(effective_alpha),
        "dynamic_alpha_star": finite(getattr(decision, "alpha_star", None)),
        "selected_topology": topology, "planned_draft_node_count": nodes,
        "mean_accepted_tokens_tau": None,
        "theoretical_speedup": finite(getattr(decision, "speedup", None)),
        "actual_step_latency_ms": None,
        "memory_status": {"free_vram_mb": finite(free_vram_mb),
                          "kv_cache_usage_pct": finite(kv_pct),
                          "guard_triggered": status == "memory_guard" if status else None},
        "decision_reason": decision.reason, "executed_runtime_mode": "unverified",
        "executed_draft_token_count": None, "tree_execution_verified": False,
    }
