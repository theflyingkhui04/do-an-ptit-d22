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
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import AdaptiveDraftController, BudgetDecision
from src.proposer.cost_policy import (
    CostDecision, PendingFeedbackStore, PendingVerifierEvent, ScalarEagleCostGovernor,
    SurvivalHistory, VerifierEventKey, source_decision_mode, EagleConsumerProfile,
)
from src.server.decision_state import (
    CONTEXT_SEMANTICS, STATE_SCHEMA, CandidateDecisionStateV1, FixedCalibrationActionV1,
    DecisionStateSnapshotV1, RuntimeIdentitySnapshotV1, context_bucket,
)

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
    decision_state: Optional[DecisionStateSnapshotV1] = None
    decision_state_status: str = "UNAVAILABLE"
    decision_state_reasons: tuple = ()
    candidate_decision_states: tuple = ()
    cost_decision: object = None
    per_request_governed_k: Dict[str, int] = field(default_factory=dict)
    per_request_base_k: Dict[str, int] = field(default_factory=dict)
    per_request_source_modes: Dict[str, str] = field(default_factory=dict)
    calibration_action: Optional[FixedCalibrationActionV1] = None

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
        cost_policy_mode: str = "off",
        consumer_profile=None,
        consumer_profile_error: Optional[str] = None,
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
        self._runtime_identity = RuntimeIdentitySnapshotV1()
        self._runtime_identity_bound = False
        self._last_decision_state_sequence = 0
        self.cost_governor = ScalarEagleCostGovernor(cost_policy_mode, consumer_profile, consumer_profile_error)
        if cost_policy_mode != "off" and not replacement_policy:
            raise ValueError("SECP cost policy requires the replacement policy")
        self.survival_history = SurvivalHistory()
        self.pending_feedback = PendingFeedbackStore()
        self.reset_recovery()

    @property
    def runtime_identity(self):
        return self._runtime_identity

    def bind_runtime_identity(self, identity: RuntimeIdentitySnapshotV1) -> None:
        """One immutable startup observation per planner; no launch-label fallback."""
        if not isinstance(identity, RuntimeIdentitySnapshotV1):
            raise TypeError("Expected RuntimeIdentitySnapshotV1")
        if self._runtime_identity_bound:
            if identity != self.runtime_identity:
                raise ValueError("Runtime identity is already bound")
            return
        self._runtime_identity = identity
        self._runtime_identity_bound = True
        if self.replacement_policy:
            self._write_step(dict(record_type="runtime_identity_snapshot", **identity.to_record()))

    def _bind_decision_state(self, snapshot, request_ids, batch_size):
        if snapshot is None:
            return None, "UNAVAILABLE", ("missing_decision_state",)
        if not isinstance(snapshot, DecisionStateSnapshotV1):
            raise TypeError("Expected DecisionStateSnapshotV1")
        reasons = []
        if snapshot.schema_version != STATE_SCHEMA or snapshot.context_snapshot_semantics != CONTEXT_SEMANTICS:
            reasons.append("unsupported_snapshot_schema_or_semantics")
        if (type(snapshot.scheduler_sequence) is not int
                or snapshot.scheduler_sequence <= self._last_decision_state_sequence):
            reasons.append("stale_scheduler_sequence")
        full = snapshot.full_scheduled_request_ids
        eligible = snapshot.eligible_decode_request_ids
        if (len(set(full)) != len(full) or len(set(eligible)) != len(eligible)
                or not set(eligible) <= set(full)
                or tuple(request_ids) != eligible or batch_size != len(eligible)):
            reasons.append("cohort_mismatch")
        epochs = dict(snapshot.request_incarnations)
        old = dict(snapshot.old_verification_width_by_request)
        contexts = dict(snapshot.host_context_tokens_after_schedule_by_request)
        if any(set(values) != set(full) for values in (epochs, old, contexts)):
            reasons.append("incomplete_row_metadata")
        if any(type(epochs.get(rid)) is not int or epochs[rid] <= 0 for rid in full):
            reasons.append("invalid_incarnation")
        if any(epochs.get(rid) != self._request_incarnations.get(rid) for rid in eligible):
            reasons.append("request_incarnation_mismatch")
        if any(type(old.get(rid)) is not int or old[rid] < 0 for rid in full):
            reasons.append("invalid_old_verification_width")
        if (type(snapshot.pure_decode) is not bool or type(snapshot.has_scheduled_prefill) is not bool
                or snapshot.pure_decode == snapshot.has_scheduled_prefill):
            reasons.append("invalid_decode_classification")
        if snapshot.runtime_mode_id != self.runtime_identity.runtime_mode_id:
            reasons.append("runtime_mode_mismatch")
        if reasons:
            return None, "UNSUPPORTED", tuple(reasons)
        self._last_decision_state_sequence = snapshot.scheduler_sequence
        missing = list(snapshot.unsupported_reasons)
        if any(type(contexts[rid]) is not int or contexts[rid] < 0 for rid in full):
            missing.append("unavailable_host_context")
        if snapshot.runtime_mode_id is None:
            missing.append("unknown_runtime_mode")
        return snapshot, "UNSUPPORTED" if missing else "SUPPORTED", tuple(dict.fromkeys(missing))

    def _candidate_decision_states(self, snapshot, base, binding_reasons):
        """Planner-owned host candidate views only; no cost lookup or K mutation."""
        if snapshot is None:
            return ()
        ids = snapshot.eligible_decode_request_ids
        old = dict(snapshot.old_verification_width_by_request)
        contexts = dict(snapshot.host_context_tokens_after_schedule_by_request)
        buckets = tuple((rid, context_bucket(contexts[rid])) for rid in ids)
        reasons = list(binding_reasons)
        if not snapshot.pure_decode:
            reasons.append("mixed_prefill_decode")
        if self.runtime_identity.identity_key is None:
            reasons.append("unsupported_runtime_identity")
        if not ids:
            reasons.append("empty_decode_cohort")
        if set(base) != set(ids):
            reasons.append("incomplete_base_vector")
        if any(bucket is None for _, bucket in buckets):
            reasons.append("unsupported_context_bucket")
        caps = sorted({0} | {k for k in (2, 3, 4) if any(k <= b for b in base.values())})
        return tuple((q, CandidateDecisionStateV1(
            self.runtime_identity.identity_key, snapshot.pure_decode, len(ids),
            tuple((rid, old[rid]) for rid in ids),
            tuple((rid, min(base[rid], q)) for rid in ids if rid in base),
            max((min(b, q) for b in base.values()), default=0), buckets,
            snapshot.runtime_mode_id, tuple(dict.fromkeys(reasons)),
        )) for q in caps)

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
        if self.cost_governor.mode != "off":
            controller = self.adapter.get_controller(req_id)
            self._known_requests.add(req_id)
            self.survival_history.bind_request(req_id, incarnation, controller.window_size)
            self.pending_feedback.bind_request(req_id, incarnation)

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
        *,
        decision_state: Optional[DecisionStateSnapshotV1] = None,
        calibration_action: Optional[FixedCalibrationActionV1] = None,
    ) -> StepPlan:
        """Selects k_i for every decoding request and the batch-wide K = max(k_i).

        kv_usage is the vLLM fraction in [0, 1] (Scheduler.get_kv_cache_usage()).
        """
        if calibration_action is not None:
            if (not isinstance(calibration_action, FixedCalibrationActionV1)
                    or not self.replacement_policy or self.cost_governor.mode == "off"):
                raise ValueError("unsupported_calibration_action_path")
        decode_req_ids = tuple(decode_req_ids)
        snapshot, state_status, state_reasons = self._bind_decision_state(
            decision_state, decode_req_ids, batch_size)
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
        base = {rid: decision.k for rid, decision in decisions.items()}
        source_modes = {rid: source_decision_mode(decision) for rid, decision in decisions.items()}
        candidate_states = self._candidate_decision_states(
            snapshot, base, state_reasons)
        cost_decision = None
        governed = dict(selected)
        if self.replacement_policy:
            if calibration_action is None:
                cost_decision = self.cost_governor.decide(
                    runtime=self.runtime_identity, candidate_states=candidate_states,
                    base=base, source_modes=source_modes, history=self.survival_history,
                    incarnations={rid: self._request_incarnations.get(rid) for rid in base})
                if cost_decision.selected_cap is not None:
                    governed = {rid: min(k, cost_decision.selected_cap) for rid, k in selected.items()}
            else:
                # Fixed experimental cap before existing replacement/recovery;
                # controller base and hard safety remain upper bounds.
                governed = {rid: min(k, calibration_action.k) for rid, k in selected.items()}
                source_modes = {rid: "calibration" for rid in source_modes}
                cost_decision = CostDecision(None, "calibration", "frozen_collection_action",
                    False, False, False, json.dumps(dict(base_k=base, candidates=[],
                        profile_identity_status="NOT_USED_CALIBRATION",
                        calibration_action_k=calibration_action.k,
                        calibration_plan_hash=calibration_action.collection_plan_hash)))
        fresh_max_k = max(governed.values(), default=0)
        if self.replacement_policy:
            cap = self._recovery_cap(governed, fresh_max_k)
            per_request_k = {rid: min(k, cap) for rid, k in governed.items()}
            batch_k = max(per_request_k.values(), default=0)
        else:
            batch_k = min(max(fresh_max_k, self.min_batch_k), self.max_k)
            cap = batch_k
            if not self.per_request:
                per_request_k = {rid: batch_k for rid in per_request_k}
        batch_node_count = max(nodes.values(), default=0)
        self._plan_sequence += 1
        if cost_decision is not None:
            self._write_step(dict(
                record_type="hardware_cost_decision", plan_sequence=self._plan_sequence,
                **cost_decision.to_record(), final_applied_k=dict(per_request_k),
                planned_worker_k=batch_k, physical_draft_width=None,
                memory_kv=dict(kv_cache_usage_pct=kv_pct, free_vram_mb=free_vram_mb),
                recovery=dict(fresh_max_k=fresh_max_k, recovery_cap_k=cap,
                              pending_target_k=self._pending_target_k, pending_streak=self._pending_streak)))
        if decision_state is not None:
            self._write_step(dict(
                record_type="decision_state_snapshot", plan_sequence=self._plan_sequence,
                decision_state=snapshot.to_record() if snapshot is not None else None,
                decision_state_status=state_status, decision_state_reasons=state_reasons,
                candidate_decision_states=[dict(cap=q, **asdict(state), status=state.status)
                                           for q, state in candidate_states],
            ))
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
                    K_base=base[req_id], source_decision_mode=source_modes[req_id],
                    governed_k=governed[req_id],
                    cost_policy_applied=cost_decision.cost_policy_applied,
                    cost_optimal=cost_decision.cost_optimal,
                    hardware_cost_claim_allowed=cost_decision.hardware_cost_claim_allowed,
                    cost_selected_cap=cost_decision.selected_cap,
                    decision_mode=cost_decision.decision_mode,
                    cost_decision_reason=cost_decision.decision_reason,
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
                        pending_streak=self._pending_streak, decision_state=snapshot,
                        decision_state_status=state_status, decision_state_reasons=state_reasons,
                        candidate_decision_states=candidate_states, cost_decision=cost_decision,
                        per_request_governed_k=governed, per_request_base_k=base,
                        per_request_source_modes=source_modes, calibration_action=calibration_action)

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
            source_base_k=origin.source_base_k, source_decision_mode=origin.source_decision_mode,
        ))

    def _emit_finalized_cost_feedback(self, event):
        if event is None:
            return
        counted = self.survival_history.add(event)
        observation = event.observation
        self._write_step(dict(
            record_type="cost_verification_finalized", **asdict(observation),
            finalization_status=event.finalization_status, actual_emitted_tokens=event.emitted_tokens,
            validation_emitted_available=event.emitted_tokens is not None,
            normal_survival_counted=counted))

    def _stage_cost_feedback(self, req_id, drafted, accepted, source, verifier_id,
                             lineage_status="attributed"):
        if self.cost_governor.mode == "off":
            return None
        try:
            if (type(source.source_base_k) is not int or not 0 <= source.source_base_k <= 4
                    or source.source_decision_mode not in ("normal", "probe", "memory_override", "calibration")):
                raise ValueError("Unknown producer base/mode")
            key = VerifierEventKey(req_id, source.incarnation, source.plan_sequence,
                                   source.schedule_sequence, verifier_id)
            event = PendingVerifierEvent(key, source.source_base_k, source.request_k,
                                         source.batch_k, accepted, drafted,
                                         source.source_decision_mode, lineage_status)
        except (AttributeError, ValueError, TypeError):
            self._write_step(dict(record_type="cost_verification_rejected", request_id=req_id,
                                  verifier_sequence=verifier_id, finalization_status="AMBIGUOUS_INVALID"))
            return None
        if not self.pending_feedback.stage(event):
            self._write_step(dict(record_type="cost_verification_rejected", **asdict(key),
                                  finalization_status="STALE_INVALID"))
            return None
        self._write_step(dict(record_type="cost_verification_pending", **asdict(event),
                              finalization_status="PENDING"))
        return key

    def record_cost_feedback_invalid(self, req_id, drafted, accepted, *, source, verifier_id, status):
        key = self._stage_cost_feedback(req_id, drafted, accepted, source, verifier_id, "invalid")
        if key is not None:
            self._emit_finalized_cost_feedback(self.pending_feedback.finalize(key, status))

    def finalize_cost_feedback(self, verifier_id, engine_outputs, *, invalid_status=None):
        if self.cost_governor.mode == "off" or verifier_id is None:
            return
        rows = {}
        if isinstance(engine_outputs, dict):
            for batch in engine_outputs.values():
                for row in getattr(batch, "outputs", ()):
                    rows.setdefault(row.request_id, []).append(row)
        for key in list(self.pending_feedback.pending):
            if key.verifier_sequence != verifier_id:
                continue
            matching = rows.get(key.request_id, ())
            status, emitted = invalid_status, None
            if status is None:
                if len(matching) != 1:
                    status = "AMBIGUOUS_INVALID"
                else:
                    row = matching[0]
                    tokens = getattr(row, "new_token_ids", None)
                    if not isinstance(tokens, (list, tuple)):
                        status = "AMBIGUOUS_INVALID"
                    else:
                        emitted = len(tokens)
                        status = ("TERMINAL_CENSORED" if getattr(row, "finish_reason", None) is not None
                                  else "ELIGIBLE_NONTERMINAL")
            self._emit_finalized_cost_feedback(self.pending_feedback.finalize(key, status, emitted))

    def record_verification(self, req_id: str, num_drafted: int, num_accepted: int,
                            *, source=None, attributed: bool = False, verifier_id=None) -> None:
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
        if attributed:
            self._stage_cost_feedback(req_id, num_drafted, num_accepted, source, verifier_id)
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

    def finish_request(self, req_id: str, *, feedback_status="ABORTED_INVALID", defer_verifier=None) -> Optional[dict]:
        """Releases the request session and appends its telemetry as one JSONL line."""
        incarnation = self._request_incarnations.pop(req_id, None)
        if incarnation is not None:
            for event in self.pending_feedback.close_epoch(req_id, incarnation, feedback_status, defer_verifier):
                self._emit_finalized_cost_feedback(event)
            self.survival_history.reset_epoch(req_id, incarnation)
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

        cost_mode = env.get("ADAPTIVE_COST_POLICY_MODE", "off").strip().lower()
        consumer_profile, consumer_error = None, None
        consumer_path = env.get("ADAPTIVE_CONSUMER_PROFILE_PATH")
        if cost_mode != "off" and consumer_path:
            from pathlib import Path
            try:
                consumer_profile = EagleConsumerProfile.from_data(json.loads(Path(consumer_path).read_text()))
            except (OSError, ValueError, TypeError, KeyError):
                consumer_error = "invalid_consumer_profile"

        return cls(
            adapter=AdaptiveProposerAdapter(controller_kwargs=controller_kwargs),
            max_k=int(env.get("ADAPTIVE_MAX_K", "4")),
            min_batch_k=int(env.get("ADAPTIVE_MIN_BATCH_K", "0")),
            per_request=_env_flag(env.get("ADAPTIVE_PER_REQUEST", "1")),
            telemetry_path=env.get("ADAPTIVE_TELEMETRY_PATH") or None,
            step_telemetry_path=env.get("ADAPTIVE_STEP_TELEMETRY_PATH") or None,
            replacement_policy=replacement_policy,
            recovery_hold_steps=int(env.get("ADAPTIVE_RECOVERY_HOLD_STEPS", "0")),
            cost_policy_mode=cost_mode, consumer_profile=consumer_profile, consumer_profile_error=consumer_error,
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
