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
from src.proposer.controller import BudgetDecision

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
    ):
        if max_k < 0 or min_batch_k < 0:
            raise ValueError("max_k and min_batch_k must be non-negative")
        if (telemetry_path and step_telemetry_path
                and os.path.normcase(os.path.realpath(telemetry_path))
                == os.path.normcase(os.path.realpath(step_telemetry_path))):
            raise ValueError("Step and aggregate telemetry must use separate paths")
        self.adapter = adapter or AdaptiveProposerAdapter()
        self.max_k = max_k
        self.min_batch_k = min_batch_k
        self.per_request = per_request
        self.telemetry_path = telemetry_path
        self.step_telemetry_path = step_telemetry_path
        self._known_requests: Set[str] = set()
        self._last_decisions: Dict[str, dict] = {}
        self._scheduled_runtime_modes: Dict[str, str] = {}
        self._plan_sequence = 0

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

        if not per_request_k:
            return StepPlan(batch_k=0, per_request_k={})

        batch_k = max(max(per_request_k.values()), self.min_batch_k)
        batch_k = min(batch_k, self.max_k)
        if not self.per_request:
            per_request_k = {req_id: batch_k for req_id in per_request_k}
        batch_node_count = max(nodes.values())
        self._plan_sequence += 1
        for req_id, decision in decisions.items():
            raw_alpha, effective_alpha = alpha_signals[req_id]
            record = _decision_record(decision, nodes[req_id], batch_size, kv_pct, free_vram_mb,
                                      raw_alpha=raw_alpha, effective_alpha=effective_alpha)
            record.update(request_id=req_id, lane="adaptive", record_type="planned_decision",
                          controller_class=f"{type(self.adapter.get_controller(req_id)).__module__}.{type(self.adapter.get_controller(req_id)).__name__}",
                          plan_sequence=self._plan_sequence, scalar_runtime_budget=per_request_k[req_id],
                          batch_scalar_budget=batch_k, batch_node_count=batch_node_count)
            self._last_decisions[req_id] = record
            self._write_step(record)
        return StepPlan(batch_k=batch_k, per_request_k=per_request_k,
                        batch_node_count=batch_node_count, per_request_node_count=nodes,
                        per_request_decisions=decisions, plan_sequence=self._plan_sequence)

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
            self._write_step({
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
                "alpha_ema": self.adapter.get_telemetry(req_id)["alpha_ema"],
                "alpha_ema_before": alpha_before,
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

        if "ADAPTIVE_ENABLE_VRAM_GUARD" in env:
            controller_kwargs["enable_vram_guard"] = _env_flag(env["ADAPTIVE_ENABLE_VRAM_GUARD"])

        return cls(
            adapter=AdaptiveProposerAdapter(controller_kwargs=controller_kwargs),
            max_k=int(env.get("ADAPTIVE_MAX_K", "4")),
            min_batch_k=int(env.get("ADAPTIVE_MIN_BATCH_K", "0")),
            per_request=_env_flag(env.get("ADAPTIVE_PER_REQUEST", "1")),
            telemetry_path=env.get("ADAPTIVE_TELEMETRY_PATH") or None,
            step_telemetry_path=env.get("ADAPTIVE_STEP_TELEMETRY_PATH") or None,
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
