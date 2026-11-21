"""EAGLE retirement-window consumer costs; never converts component profiles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
import math
import random
import re

from src.server.decision_state import CandidateDecisionStateV1, RuntimeIdentitySnapshotV1

PROTOCOL = "EAGLE-RW-v1.1-candidate"
UNIT = "ms/retired_batch"
K0_SEMANTICS = "same_engine_eagle_zero_width_v1"
CLOCK = "perf_counter_ns"
TIMING_BOUNDARY = "enginecore_retirement_window_v1"


class ConsumerProfileError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def decision_state_payload(state: CandidateDecisionStateV1):
    """Remove request labels, retaining ordered, aligned row attributes exactly."""
    if not isinstance(state, CandidateDecisionStateV1) or state.status != "SUPPORTED":
        raise ConsumerProfileError("unsupported_decision_state")
    rows = state.candidate_future_width_by_request
    ids = tuple(r for r, _ in rows)
    if (not state.pure_decode_state or not ids or len(set(ids)) != len(ids)
            or state.eligible_decode_rows != len(ids)
            or tuple(r for r, _ in state.old_verification_width_by_request) != ids
            or tuple(r for r, _ in state.context_bucket_by_request) != ids
            or not state.runtime_identity_key or not state.runtime_mode_id
            or any(type(k) is not int or k not in (0, 2, 3, 4) for _, k in rows)
            or any(type(k) is not int or not 0 <= k <= 4 for _, k in state.old_verification_width_by_request)
            or state.candidate_shared_worker_k != max(k for _, k in rows)
            or any(b not in ("[0,256)", "[256,1024)", "[1024,2048)", "[2048,4096)")
                   for _, b in state.context_bucket_by_request)):
        raise ConsumerProfileError("unsupported_decision_state")
    return dict(schema="decision_state_key_v1", runtime_identity_key=state.runtime_identity_key,
                pure_decode_state=True, eligible_decode_rows=len(ids),
                old_verification_widths=[k for _, k in state.old_verification_width_by_request],
                candidate_future_widths=[k for _, k in rows],
                candidate_shared_worker_k=state.candidate_shared_worker_k,
                context_buckets=[b for _, b in state.context_bucket_by_request],
                runtime_mode_id=state.runtime_mode_id)


def decision_state_key(state):
    return canonical(decision_state_payload(state))


@dataclass(frozen=True)
class PublicationPlan:
    measured_cohorts: int = 50
    retirements_per_cohort: int = 8
    bootstrap_resamples: int = 5000
    confidence_level: float = .95
    seed: int = 0
    replicate_unit: str = "independent_cohort"
    method: str = "percentile_cohort_bootstrap_v1"
    stopping_rule: str = "fixed_sample_plan_v1"

    def __post_init__(self):
        if (type(self.measured_cohorts) is not int or self.measured_cohorts < 50
                or type(self.retirements_per_cohort) is not int or self.retirements_per_cohort < 1
                or type(self.bootstrap_resamples) is not int or self.bootstrap_resamples < 5000
                or self.confidence_level != .95 or type(self.seed) is not int
                or self.replicate_unit != "independent_cohort"
                or self.method != "percentile_cohort_bootstrap_v1"
                or self.stopping_rule != "fixed_sample_plan_v1"):
            raise ConsumerProfileError("invalid_publication_plan")


@dataclass(frozen=True)
class Cohort:
    cohort_id: str
    clock_domain: str
    start_ns: int
    retirement_end_ns: tuple
    emitted_tokens: tuple
    status: str = "ok"
    error: str | None = None
    clock: str = CLOCK
    added_cuda_sync: bool = False
    steady_pure_decode: bool = True

    def __post_init__(self):
        object.__setattr__(self, "retirement_end_ns", tuple(self.retirement_end_ns))
        object.__setattr__(self, "emitted_tokens", tuple(self.emitted_tokens))


def _percentile(values, fraction):
    at = (len(values) - 1) * fraction
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def publish_cost(cohorts, plan=PublicationPlan()):
    """Fixed-plan cohort bootstrap. Failed attempts remain in the returned raw data."""
    cohorts = tuple(cohorts)
    raw = dict(plan=asdict(plan), cohorts=[asdict(c) for c in cohorts])
    failures, costs, ids, clock_ends = [], [], set(), {}
    for c in cohorts:
        if not c.cohort_id or c.cohort_id in ids:
            raise ConsumerProfileError("duplicate_or_missing_cohort_id")
        ids.add(c.cohort_id)
        if c.status != "ok":
            if c.status not in ("oom", "timeout", "unsupported_shape", "observer_failure", "error", "abort") or not c.error:
                raise ConsumerProfileError("unexplained_failed_cohort")
            failures.append(c.cohort_id)
            continue
        ends = c.retirement_end_ns
        if (c.clock != CLOCK or not c.clock_domain or type(c.start_ns) is not int or c.start_ns < 0
                or c.added_cuda_sync is not False or c.steady_pure_decode is not True
                or len(ends) != plan.retirements_per_cohort or len(c.emitted_tokens) != len(ends)
                or any(type(n) is not int or n <= before for n, before in zip(ends, (c.start_ns,) + ends[:-1]))
                or any(type(n) is not int or n < 0 for n in c.emitted_tokens)
                or c.start_ns < clock_ends.get(c.clock_domain, 0)):
            raise ConsumerProfileError("invalid_retirement_window")
        clock_ends[c.clock_domain] = ends[-1]
        costs.append((ends[-1] - c.start_ns) / 1e6 / len(ends))
    result = dict(protocol=PROTOCOL, unit=UNIT, raw=raw, raw_sha256=digest(raw),
                  cohort_ids=sorted(ids), failures=failures, complete=False,
                  replicate_unit=plan.replicate_unit, performance_claim=False)
    if len(cohorts) != plan.measured_cohorts or failures:
        return result
    rng = random.Random(plan.seed)
    # Each draw chooses complete cohorts. Inner retirements are never sampled.
    samples = sorted(sum(rng.choices(costs, k=len(costs))) / len(costs)
                     for _ in range(plan.bootstrap_resamples))
    tail = (1 - plan.confidence_level) / 2
    result.update(complete=True, C_point=sum(costs) / len(costs),
                  C_low=_percentile(samples, tail), C_high=_percentile(samples, 1 - tail))
    return result


@dataclass(frozen=True)
class ConsumerCostCell:
    C_point: float
    C_low: float
    C_high: float
    evidence_json: str
    raw_sha256: str
    synthetic: bool = False

    def __post_init__(self):
        if (not all(positive(c) for c in (self.C_point, self.C_low, self.C_high))
                or not self.C_low <= self.C_point <= self.C_high):
            raise ConsumerProfileError("invalid_cost_interval")
        raw = json.loads(self.evidence_json)
        if digest(raw) != self.raw_sha256:
            raise ConsumerProfileError("raw_link_mismatch")
        if self.synthetic:
            expected = dict(kind="synthetic_interval_fixture_v1", C_point=self.C_point,
                            C_low=self.C_low, C_high=self.C_high)
            if raw != expected:
                raise ConsumerProfileError("synthetic_fixture_mismatch")
        else:
            if raw.get("kind") != "retirement_cohorts_v1":
                raise ConsumerProfileError("component_cost_not_consumer_cost")
            report = publish_cost([Cohort(**c) for c in raw["cohorts"]], PublicationPlan(**raw["plan"]))
            if not report["complete"] or any(report[k] != getattr(self, k) for k in ("C_point", "C_low", "C_high")):
                raise ConsumerProfileError("unpublishable_or_inconsistent_cost")

    @classmethod
    def synthetic_fixture(cls, point, low=None, high=None):
        low, high = point if low is None else low, point if high is None else high
        if not all(positive(v) for v in (point, low, high)) or not low <= point <= high:
            raise ConsumerProfileError("invalid_cost_interval")
        raw = dict(kind="synthetic_interval_fixture_v1", C_point=point, C_low=low, C_high=high)
        return cls(point, low, high, canonical(raw), digest(raw), True)

    @classmethod
    def from_publication(cls, report):
        if not report.get("complete"):
            raise ConsumerProfileError("publication_incomplete")
        raw = dict(kind="retirement_cohorts_v1", **report["raw"])
        return cls(report["C_point"], report["C_low"], report["C_high"], canonical(raw), digest(raw))

    def to_data(self):
        return dict(C_point=self.C_point, C_low=self.C_low, C_high=self.C_high,
                    evidence=json.loads(self.evidence_json), raw_sha256=self.raw_sha256, synthetic=self.synthetic)


def validate_identity(identity):
    if not isinstance(identity, RuntimeIdentitySnapshotV1) or identity.status != "SUPPORTED":
        raise ConsumerProfileError("unknown_profile_identity")
    observed, configured = dict(identity.observed), dict(identity.configured)
    if (not positive(observed["vram_total_mb"]) or observed["vllm_version"] != "0.30.0"
            or any(not isinstance(observed.get(k), str) or not observed[k] for k in (
                "gpu_name", "sm_architecture", "torch_version", "cuda_version", "driver_version",
                "effective_dtype", "effective_quantization"))
            or any(not isinstance(configured.get(k), str) or not configured[k] for k in ("target_model", "draft_model"))
            or configured["draft_method"] not in ("eagle", "eagle3")
            or not all(isinstance(configured.get(k), str) and re.fullmatch(r"[0-9a-fA-F]{40}", configured[k])
                       for k in ("target_revision", "draft_revision"))
            or any(configured.get(k) is None for k in ("configured_dtype", "configured_quantization"))):
        raise ConsumerProfileError("invalid_profile_identity")


@dataclass(frozen=True)
class EagleConsumerProfile:
    profile_id: str
    identity: RuntimeIdentitySnapshotV1
    cells: tuple
    synthetic: bool = False
    protocol: str = PROTOCOL
    unit: str = UNIT
    k0_semantics: str = K0_SEMANTICS
    alias_rule_version: str = "exact_observed_name_v1"
    driver_policy: str = "exact_observed_driver_v1"
    vram_policy: str = "exact_observed_total_mb_v1"
    timing_boundary: str = TIMING_BOUNDARY
    clock: str = CLOCK
    added_cuda_sync: bool = False

    def __post_init__(self):
        validate_identity(self.identity)
        object.__setattr__(self, "cells", tuple(tuple(pair) for pair in self.cells))
        if (not self.profile_id or self.protocol != PROTOCOL or self.unit != UNIT
                or self.k0_semantics != K0_SEMANTICS or self.alias_rule_version != "exact_observed_name_v1"
                or self.driver_policy != "exact_observed_driver_v1" or self.vram_policy != "exact_observed_total_mb_v1"
                or self.timing_boundary != TIMING_BOUNDARY or self.clock != CLOCK or self.added_cuda_sync is not False
                or len(dict(self.cells)) != len(self.cells)):
            raise ConsumerProfileError("invalid_consumer_profile_contract")
        for key, cell in self.cells:
            state = json.loads(key)
            if (state.get("schema") != "decision_state_key_v1"
                    or state.get("runtime_identity_key") != self.identity.identity_key
                    or not isinstance(cell, ConsumerCostCell) or cell.synthetic != self.synthetic):
                raise ConsumerProfileError("invalid_profile_cell")

    def identity_status(self, runtime):
        if runtime.status != "SUPPORTED":
            return "UNKNOWN_IDENTITY"
        return "VALID_MATCH" if runtime == self.identity else "IDENTITY_MISMATCH"

    def lookup(self, state):
        key = decision_state_key(state)
        try:
            return self._cell_index[key]
        except KeyError as exc:
            raise ConsumerProfileError("unsupported_decision_state") from exc

    def to_data(self):
        return dict(profile_id=self.profile_id, protocol=self.protocol, unit=self.unit,
                    identity=asdict(self.identity), synthetic=self.synthetic, k0_semantics=self.k0_semantics,
                    alias_rule_version=self.alias_rule_version, driver_policy=self.driver_policy,
                    vram_policy=self.vram_policy, timing_boundary=self.timing_boundary,
                    clock=self.clock, added_cuda_sync=self.added_cuda_sync,
                    cells=[dict(state=json.loads(k), cost=c.to_data()) for k, c in self.cells])

    @cached_property
    def _cell_index(self):
        return dict(self.cells)

    @cached_property
    def sha256(self):
        return digest(self.to_data())

    @classmethod
    def from_data(cls, data):
        if not isinstance(data, dict) or data.get("protocol") != PROTOCOL:
            raise ConsumerProfileError("component_or_unknown_profile_protocol")
        data = dict(data)
        data["identity"] = RuntimeIdentitySnapshotV1(**data["identity"])
        cells = []
        for item in data["cells"]:
            if (not isinstance(item, dict) or not isinstance(item.get("state"), dict)
                    or not isinstance(item.get("cost"), dict)):
                raise ConsumerProfileError("invalid_profile_cell")
            cost = dict(item["cost"])
            cost["evidence_json"] = canonical(cost.pop("evidence"))
            cells.append((canonical(item["state"]), ConsumerCostCell(**cost)))
        data["cells"] = tuple(cells)
        return cls(**data)
