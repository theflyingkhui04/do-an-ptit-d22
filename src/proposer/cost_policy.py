"""SECP-v1.1: finalized same-base survival and a conservative batch cost cap."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math

try:
    from flexsds.core.eagle_consumer_profile import (
        ConsumerCostCell, ConsumerProfileError, EagleConsumerProfile, canonical, decision_state_payload,
    )
except ModuleNotFoundError as exc:
    if exc.name not in ("flexsds", "flexsds.core", "flexsds.core.eagle_consumer_profile"):
        raise
    from src.flexsds.core.eagle_consumer_profile import (
        ConsumerCostCell, ConsumerProfileError, EagleConsumerProfile, canonical, decision_state_payload,
    )

POLICY_VERSION = "SECP-v1.1-candidate"
FINALIZATION_STATES = frozenset({
    "ELIGIBLE_NONTERMINAL", "TERMINAL_CENSORED", "PREEMPTED_INVALID",
    "STALE_INVALID", "ABORTED_INVALID", "AMBIGUOUS_INVALID",
})


def source_decision_mode(decision):
    if getattr(decision, "memory_state", "clear") != "clear":
        return "memory_override"
    if decision.reason.startswith("Periodic probe"):
        return "probe"
    return "normal"


@dataclass(frozen=True)
class VerifierEventKey:
    request_id: str
    incarnation: int
    source_plan_sequence: int
    source_schedule_sequence: int
    verifier_sequence: int

    def __post_init__(self):
        if (not isinstance(self.request_id, str) or not self.request_id
                or any(type(n) is not int or n <= 0 for n in (
                    self.incarnation, self.source_plan_sequence, self.source_schedule_sequence,
                    self.verifier_sequence))):
            raise ValueError("Pending feedback needs incarnation, producer and verifier identities")

    @property
    def epoch(self):
        return self.request_id, self.incarnation


@dataclass(frozen=True)
class PendingVerifierEvent:
    key: VerifierEventKey
    source_base_k: int
    source_applied_k: int
    source_shared_k: int
    A: int
    D: int
    source_decision_mode: str = "normal"
    lineage_status: str = "attributed"


@dataclass(frozen=True)
class SurvivalEvent:
    observation: PendingVerifierEvent
    finalization_status: str
    emitted_tokens: int | None = None


@dataclass(frozen=True)
class SurvivalEstimate:
    request_id: str
    incarnation: int
    base_k: int
    window_size: int
    count: int
    survival: tuple

    @property
    def ready(self):
        return self.base_k == 0 or self.count == self.window_size

    def predicted_yield(self, k):
        if type(k) is not int or not 0 <= k <= self.base_k:
            raise ValueError("Cannot extrapolate above the source base")
        if k == 0:
            return 1.
        if not self.count:
            raise ValueError("No survival evidence")
        return 1. + sum(self.survival[:k])


class SurvivalHistory:
    def __init__(self):
        self._active = {}
        self._windows = {}

    def bind_request(self, request_id, incarnation, window_size):
        if type(window_size) is not int or window_size < 1:
            raise ValueError("Survival window must be positive")
        old = self._active.get(request_id)
        if old is not None and old[0] != incarnation:
            self.reset_epoch(request_id, old[0])
        elif old is not None and old[1] != window_size:
            raise ValueError("Cannot silently change a live survival window")
        self._active[request_id] = (incarnation, window_size)

    def reset_epoch(self, request_id, incarnation):
        if self._active.get(request_id, (None,))[0] == incarnation:
            self._active.pop(request_id)
        for key in list(self._windows):
            if key[:2] == (request_id, incarnation):
                del self._windows[key]

    def add(self, finalized):
        e = finalized.observation
        epoch = self._active.get(e.key.request_id)
        if (finalized.finalization_status != "ELIGIBLE_NONTERMINAL"
                or e.source_decision_mode != "normal" or e.lineage_status != "attributed"
                or e.source_base_k not in (2, 3, 4) or type(e.D) is not int or type(e.A) is not int
                or not 0 <= e.A <= e.D or e.D < e.source_base_k
                or epoch is None or epoch[0] != e.key.incarnation):
            return False
        key = (*e.key.epoch, e.source_base_k)
        self._windows.setdefault(key, deque(maxlen=epoch[1])).append(e.A)
        return True

    def estimate(self, request_id, incarnation, base_k):
        active = self._active.get(request_id)
        if active is None or active[0] != incarnation:
            raise ValueError("Unbound survival incarnation")
        values = self._windows.get((request_id, incarnation, base_k), ())
        count = len(values)
        tails = tuple(sum(a >= j for a in values) / count for j in range(1, base_k + 1)) if count else ()
        return SurvivalEstimate(request_id, incarnation, base_k, active[1], count, tails)


class PendingFeedbackStore:
    """Per-engine pending events; finalized watermark is O(active requests).

    Native EngineCore retires verifier outputs in order. Several observations
    can be pending and finalize independently; finalization never uses a later
    schedule as evidence. Closed epochs cannot be restaged.
    """
    def __init__(self):
        self.pending = {}
        self._active = {}
        self._finalized_through = {}
        self._closed = {}

    def bind_request(self, request_id, incarnation):
        old = self._active.get(request_id)
        if old is not None and old != incarnation:
            if any(k.epoch == (request_id, old) for k in self.pending):
                raise ValueError("Old pending epoch must be closed before rebind")
            self._finalized_through.pop((request_id, old), None)
        self._active[request_id] = incarnation

    def stage(self, event):
        key = event.key
        if (self._active.get(key.request_id) != key.incarnation
                or key in self.pending
                or key.verifier_sequence <= self._finalized_through.get(key.epoch, 0)
                or type(event.A) is not int or type(event.D) is not int
                or not 0 <= event.A <= event.D):
            return False
        self.pending[key] = event
        return True

    def _gc(self, epoch):
        if self._active.get(epoch[0]) != epoch[1] and not any(k.epoch == epoch for k in self.pending):
            self._closed.pop(epoch, None)
            self._finalized_through.pop(epoch, None)

    def finalize(self, key, status, emitted_tokens=None):
        if status not in FINALIZATION_STATES:
            raise ValueError("Unknown feedback finalization state")
        if emitted_tokens is not None and (type(emitted_tokens) is not int or emitted_tokens < 0):
            raise ValueError("Invalid observed emitted count")
        event = self.pending.pop(key, None)
        if event is None:
            return None
        status = self._closed.get(key.epoch, status)
        if status == "ELIGIBLE_NONTERMINAL" and self._active.get(key.request_id) != key.incarnation:
            status = "STALE_INVALID"
        self._finalized_through[key.epoch] = max(key.verifier_sequence, self._finalized_through.get(key.epoch, 0))
        result = SurvivalEvent(event, status, emitted_tokens)
        self._gc(key.epoch)
        return result

    def close_epoch(self, request_id, incarnation, status="ABORTED_INVALID", defer_verifier=None):
        epoch = request_id, incarnation
        if self._active.get(request_id) == incarnation:
            self._active.pop(request_id)
        self._closed[epoch] = status
        results = []
        for key in list(self.pending):
            if key.epoch == epoch and key.verifier_sequence != defer_verifier:
                results.append(self.finalize(key, status))
        self._gc(epoch)
        return tuple(results)


@dataclass(frozen=True)
class CandidateScore:
    cap: int
    future_vector: tuple
    per_request_yield: tuple
    predicted_yield_batch: float
    C_point: float
    C_low: float
    C_high: float
    R_point: float
    R_lower: float
    R_upper: float
    supported: bool
    decision_state_key_json: str
    raw_sha256: str


@dataclass(frozen=True)
class CostDecision:
    selected_cap: int | None
    decision_mode: str
    decision_reason: str
    cost_policy_applied: bool
    cost_optimal: bool
    hardware_cost_claim_allowed: bool
    evidence_json: str

    def to_record(self):
        return dict(json.loads(self.evidence_json), **{
            k: v for k, v in asdict(self).items() if k != "evidence_json"})


class ScalarEagleCostGovernor:
    def __init__(self, mode="off", profile=None, profile_error=None):
        if mode not in ("off", "strict", "compat"):
            raise ValueError("ADAPTIVE_COST_POLICY_MODE must be off, strict or compat")
        self.mode, self.profile, self.profile_error = mode, profile, profile_error

    def decide(self, *, runtime, candidate_states, base, source_modes, history, incarnations):
        profile = self.profile
        evidence = dict(policy_version=POLICY_VERSION, consumer_profile_protocol=getattr(profile, "protocol", None),
                        profile_id=getattr(profile, "profile_id", None), profile_hash=getattr(profile, "sha256", None),
                        synthetic_profile=getattr(profile, "synthetic", None), base_k=dict(base),
                        request_incarnations=dict(incarnations), source_modes=dict(source_modes),
                        policy_mode=self.mode, profile_identity_status="UNAVAILABLE", survival={}, candidates=[],
                        candidate_states=[dict(cap=q, **asdict(s), status=s.status) for q, s in candidate_states])

        def result(cap, mode, reason, applied=False, optimal=False):
            return CostDecision(cap, mode, reason, applied, optimal,
                                applied and profile is not None and not profile.synthetic, canonical(evidence))

        def degraded(reason):
            return result(None if self.mode == "compat" else 0,
                          "compatibility" if self.mode == "compat" else "normal", reason)

        if any(mode == "memory_override" for mode in source_modes.values()):
            # The controller has already forced affected requests to zero.
            # No cost preference may restore them; preserve other request bounds.
            return result(None, "memory_override", "memory_or_kv_safety")
        if self.mode == "off":
            return result(None, "off", "cost_policy_off")
        if profile is None:
            return degraded(self.profile_error or "missing_profile")
        evidence["profile_identity_status"] = profile.identity_status(runtime)
        if evidence["profile_identity_status"] != "VALID_MATCH":
            return degraded("identity_mismatch")
        states = dict(candidate_states)
        if not states or 0 not in states or any(s.status != "SUPPORTED" for s in states.values()):
            return degraded("unsupported_decision_state")
        costs = {}
        for q, state in states.items():
            try:
                costs[q] = profile.lookup(state)
                if not isinstance(costs[q], ConsumerCostCell):
                    return degraded("invalid_cost_cell")
            except (ConsumerProfileError, ValueError, KeyError, TypeError):
                return degraded("missing_k0_reference" if q == 0 else "unsupported_decision_state")
        evidence["admitted_costs"] = {q: dict(C_point=c.C_point, C_low=c.C_low, C_high=c.C_high,
                                             raw_sha256=c.raw_sha256) for q, c in costs.items()}
        if any(mode == "probe" for mode in source_modes.values()):
            return result(None, "probe", "legacy_probe_exploration")
        estimates = {}
        try:
            for rid, b in base.items():
                estimates[rid] = history.estimate(rid, incarnations[rid], b)
                evidence["survival"][rid] = asdict(estimates[rid])
        except (KeyError, ValueError):
            return degraded("unbound_survival_incarnation")
        if any(not e.ready for e in estimates.values()):
            return result(None, "cost_warmup", "survival_window_underfilled")
        reference_upper = len(base) / costs[0].C_low
        reference_lower = len(base) / costs[0].C_high
        evidence["K0_R_upper"] = reference_upper
        scores = []
        for q, state in states.items():
            yields = tuple((rid, estimates[rid].predicted_yield(k))
                           for rid, k in state.candidate_future_width_by_request)
            y = sum(value for _, value in yields)
            cost = costs[q]
            scores.append(CandidateScore(q, state.candidate_future_width_by_request, yields, y,
                cost.C_point, cost.C_low, cost.C_high, y / cost.C_point, y / cost.C_high,
                y / cost.C_low, q > 0 and y / cost.C_high > reference_upper,
                canonical(decision_state_payload(state)), cost.raw_sha256))
        evidence["candidates"] = [asdict(s) for s in scores]
        best = None
        for score in sorted(scores, key=lambda s: s.cap):
            if score.supported and (best is None or (score.R_lower > best.R_lower
                    and not math.isclose(score.R_lower, best.R_lower, rel_tol=1e-9, abs_tol=1e-12))):
                best = score
        if best is not None:
            return result(best.cap, "normal", "supported_conservative_cap", True, True)
        overlap = any(s.cap > 0 and s.R_lower <= reference_upper and s.R_upper >= reference_lower for s in scores)
        return result(0, "normal", "INCONCLUSIVE_COST" if overlap else "no_profitable_candidate", True, not overlap)
