"""Opt-in, project-local EAGLE-RW evidence. Never launches or publishes anything.

Canonical CPU evidence is explicitly synthetic. Live installation requires a
clean source checkpoint and the existing source-gated local executor. The legacy
RetirementObserver export is deliberately unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import os
import uuid
from weakref import WeakKeyDictionary

try:
    from flexsds.core.eagle_consumer_profile import (
        PROTOCOL, UNIT, CLOCK, TIMING_BOUNDARY, K0_SEMANTICS, PublicationPlan,
        canonical, digest, decision_state_key, validate_identity,
    )
except ModuleNotFoundError as error:
    if error.name not in ("flexsds", "flexsds.core", "flexsds.core.eagle_consumer_profile"):
        raise
    from src.flexsds.core.eagle_consumer_profile import (
        PROTOCOL, UNIT, CLOCK, TIMING_BOUNDARY, K0_SEMANTICS, PublicationPlan,
        canonical, digest, decision_state_key, validate_identity,
    )
from src.server.decision_state import RuntimeIdentitySnapshotV1


_ISSUED = object()
_COHORT_CLAIMS = WeakKeyDictionary()
_CODE_PATHS = (
    "experiments/eagle_rw_collector.py", "src/server/decision_state.py",
    "src/server/budget_planner.py", "src/server/adaptive_scheduler.py",
    "src/server/lineage.py", "src/server/lineage_executor.py",
    "src/proposer/cost_policy.py", "src/flexsds/core/eagle_consumer_profile.py",
)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def runtime_config_payload(identity, protocol=PROTOCOL):
    """Canonical scalar dictionaries, never repr or caller dictionary order."""
    validate_identity(identity)
    if protocol != PROTOCOL:
        raise ValueError("unsupported_protocol")
    return dict(schema="eagle_rw_runtime_config_v1", profile_protocol_version=protocol,
                observed=dict(identity.observed), configured=dict(identity.configured))


def runtime_config_hash(identity, protocol=PROTOCOL):
    return digest(runtime_config_payload(identity, protocol))


@dataclass(frozen=True)
class EagleRWSourceProvenance:
    source_git_sha: str
    synthetic: bool
    code_manifest: tuple = ()
    checkout_root: str | None = None
    _proof: object = field(default=None, repr=False, compare=False)

    @classmethod
    def cpu_fixture(cls, source_git_sha):
        if not _sha(source_git_sha):
            raise ValueError("missing_source_sha")
        return cls(source_git_sha, True, _proof=_ISSUED)

    @classmethod
    def from_checkout(cls, root):
        """Read-only clean execution-source gate; raw audit outputs are not code.

        Git LF normalization is explicit. No client-supplied 'clean' boolean is
        accepted. Runtime packaging must preserve this verified checkout.
        """
        root = Path(root).resolve()
        def git(*args):
            p = subprocess.run(["git", "-c", "safe.directory="+root.as_posix(), *args],
                               cwd=root, capture_output=True, check=True)
            return p.stdout
        sha = git("rev-parse", "HEAD").decode().strip()
        dirty = git("status", "--porcelain", "--untracked-files=all", "--",
                    "src", "experiments", "tests", "pyproject.toml")
        tracked_dirty = git("diff", "--name-only", "HEAD")
        if not _sha(sha) or dirty or tracked_dirty:
            raise ValueError("dirty_or_unfrozen_execution_source")
        manifest = []
        for name in _CODE_PATHS:
            committed = git("show", sha+":"+name)
            loaded = (root/name).read_bytes().replace(b"\r\n", b"\n")
            if committed != loaded:
                raise ValueError("loaded_source_mismatch:"+name)
            manifest.append((name, hashlib.sha256(loaded).hexdigest()))
        return cls(sha, False, tuple(manifest), str(root), _ISSUED)

    def valid(self):
        if self._proof is not _ISSUED or not _sha(self.source_git_sha):
            return False
        if self.synthetic:
            return True
        try:
            fresh = type(self).from_checkout(self.checkout_root)
            return (fresh.source_git_sha, fresh.code_manifest) == (self.source_git_sha, self.code_manifest)
        except (ValueError, OSError, subprocess.SubprocessError, TypeError):
            return False


@dataclass(frozen=True)
class EagleRWCohortDescriptor:
    cohort_id: str
    request_incarnations: tuple
    replicate_unit: str = "independent_cohort"

    def __post_init__(self):
        pairs = tuple(tuple(row) for row in self.request_incarnations)
        object.__setattr__(self, "request_incarnations", pairs)
        if (not isinstance(self.cohort_id, str) or not self.cohort_id
                or self.replicate_unit != "independent_cohort" or not pairs
                or len(dict(pairs)) != len(pairs)
                or any(not isinstance(r, str) or not r or type(e) is not int or e <= 0 for r,e in pairs)):
            raise ValueError("invalid_cohort_provenance")


@dataclass(frozen=True)
class EagleRWCollectionPlan:
    source: EagleRWSourceProvenance
    runtime_identity: RuntimeIdentitySnapshotV1
    cohort: EagleRWCohortDescriptor
    target_decision_state_key: str
    sample_plan: PublicationPlan = field(default_factory=PublicationPlan)
    profile_protocol_version: str = PROTOCOL

    def __post_init__(self):
        if not isinstance(self.sample_plan, PublicationPlan) or not isinstance(self.cohort, EagleRWCohortDescriptor):
            raise ValueError("invalid_fixed_collection_plan")
        if not self.source.valid():
            raise ValueError("dirty_or_unfrozen_execution_source")
        runtime_config_payload(self.runtime_identity, self.profile_protocol_version)
        key = json.loads(self.target_decision_state_key)
        if (canonical(key) != self.target_decision_state_key
                or key.get("schema") != "decision_state_key_v1"
                or key.get("runtime_identity_key") != self.runtime_identity.identity_key
                or key.get("runtime_mode_id") != self.runtime_identity.runtime_mode_id):
            raise ValueError("invalid_target_cell")

    @property
    def runtime_config_hash(self):
        return runtime_config_hash(self.runtime_identity, self.profile_protocol_version)

    def payload(self):
        return dict(profile_protocol_version=self.profile_protocol_version,
            source_git_sha=self.source.source_git_sha, synthetic=self.source.synthetic,
            source_manifest=list(self.source.code_manifest), runtime_config_hash=self.runtime_config_hash,
            expected_runtime_identity=asdict(self.runtime_identity), cohort=asdict(self.cohort),
            target_decision_state_key=self.target_decision_state_key, fixed_sample_plan=asdict(self.sample_plan))

    @property
    def collection_plan_hash(self):
        return digest(self.payload())


class CollectorCanonicalityStatus(str, Enum):
    CANONICAL = "CANONICAL"
    NONCANONICAL = "NONCANONICAL"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class EagleRWDecisionBinding:
    scheduler_sequence: int
    plan_sequence: int
    payload_json: str
    _proof: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class EagleRWRetirementRecord:
    payload_json: str
    reasons: tuple
    _seal: str = field(default="", repr=False)
    _proof: object = field(default=None, repr=False, compare=False)
    _assessment_seal: str = field(default="", repr=False)

    @property
    def canonicality_status(self):
        try:
            if (self._proof is not _ISSUED or digest(json.loads(self.payload_json)) != self._seal
                    or self._assessment_seal != digest([self._seal,self.reasons])):
                return CollectorCanonicalityStatus.INVALID
        except (ValueError, TypeError):
            return CollectorCanonicalityStatus.INVALID
        if self.reasons:
            return (CollectorCanonicalityStatus.NONCANONICAL if self.reasons == ("anchor_retirement",)
                    else CollectorCanonicalityStatus.UNSUPPORTED)
        try:
            if _record_errors(json.loads(self.payload_json)):
                return CollectorCanonicalityStatus.INVALID
        except (ValueError, TypeError, KeyError, IndexError):
            return CollectorCanonicalityStatus.INVALID
        return CollectorCanonicalityStatus.CANONICAL

    def to_data(self):
        return dict(json.loads(self.payload_json), canonicality_status=self.canonicality_status.value,
                    unsupported_reasons=list(self.reasons), raw_sha256=self._seal)


class EagleRWCollector:
    """Owned instance hooks; no worker/result schema changes or forced waits.

    A plan labels one independently specified cohort; it never creates cohorts
    per retirement. Canonical evidence is not a published consumer profile.
    """
    def __init__(self, core, plan, *, clock=None):
        self.core, self.plan = core, plan
        self.scheduler = core.scheduler
        self.planner = self.scheduler._planner
        self.executor = core.model_executor
        self.runner = self.executor._lineage_runner
        if not plan.source.synthetic:
            # Source-gated project executor already certifies the pinned runner.
            if (type(self.executor).__module__, type(self.executor).__name__) != (
                    "src.server.lineage_executor", "LineageUniProcExecutor"):
                raise ValueError("unsupported_local_executor")
            if self.scheduler._lineage_registration.owner() is not self.executor:
                raise ValueError("unowned_executor")
            if Path(__file__).resolve() != Path(plan.source.checkout_root)/_CODE_PATHS[0]:
                raise ValueError("collector_source_root_mismatch")
            if clock is not None:
                raise ValueError("live_clock_override_prohibited")
        self.clock = time.perf_counter_ns if clock is None else clock
        self.clock_domain = f"enginecore:{os.getpid()}:{uuid.uuid4().hex}"
        self._undo, self._pending, self._records = [], {}, []
        self._scheduling = self._executing = self._retiring = None
        self._anchor = None
        self._installed = False
        self._source_verified = False

    def _patch(self, obj, name, wrapper):
        existed, prior = name in vars(obj), vars(obj).get(name)
        wrapped = wrapper(getattr(obj, name))
        setattr(obj, name, wrapped)
        self._undo.append((obj, name, existed, prior, wrapped))

    def _observe(self, action, schedule, operation):
        try:
            return operation()
        except Exception as error:
            self.failure("observer_failure", schedule, detail=action+":"+type(error).__name__)
            return None

    def install(self):
        if self._installed:
            raise ValueError("collector_already_installed")
        if not self.plan.source.valid():
            raise ValueError("dirty_or_unfrozen_execution_source")
        self._source_verified = True
        expected = dict(self.plan.cohort.request_incarnations)
        active = self.planner._request_incarnations
        if any(active.get(rid) != epoch for rid,epoch in expected.items()):
            raise ValueError("cohort_incarnation_not_active")
        claims = _COHORT_CLAIMS.setdefault(self.scheduler,{})
        for key in list(claims):
            if active.get(key[0]) != key[1]:
                del claims[key]
        claim = (self.plan.cohort.cohort_id,self.plan.collection_plan_hash)
        if any(key in claims and claims[key] != claim for key in self.plan.cohort.request_incarnations):
            raise ValueError("cohort_not_independent_reused_incarnation")
        claims.update({key:claim for key in self.plan.cohort.request_incarnations})
        self._installed = True
        try:
            self._patch(self.planner, "plan_step", self._planning_hook)
            self._patch(self.scheduler, "_update_after_schedule", self._schedule_hook)
            self._patch(self.scheduler, "update_from_output", self._retirement_hook)
            self._patch(self.planner, "_emit_finalized_cost_feedback", self._feedback_hook)
            self._patch(self.executor, "execute_model", self._execution_hook)
            self._patch(self.executor, "sample_tokens", self._sampling_hook)
            self._patch(self.runner.drafter, "propose", self._proposal_hook)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        # In-flight/abandoned attempts remain evidence before witnesses release.
        for entry in list(self._pending.values()):
            self.failure("abort", entry["schedule"], detail="collector_closed_with_inflight")
        for obj, name, existed, prior, wrapped in reversed(self._undo):
            if vars(obj).get(name) is wrapped:
                if existed:
                    setattr(obj, name, prior)
                else:
                    delattr(obj, name)
        self._undo.clear()
        self._installed = False

    def _planning_hook(self, delegate):
        def planning(*args, **kwargs):
            result = delegate(*args, **kwargs)
            if self._scheduling is not None:
                self._observe("pre_action_binding", self._scheduling,
                              lambda: self._capture(self._scheduling, result))
            return result
        return planning

    def _capture(self, schedule, result):
        if self._scheduling is not schedule or self.scheduler._lineage.get(schedule) is not None:
            raise ValueError("decision_state_not_captured_before_action")
        state = result.decision_state
        if state is None or state.scheduler_sequence != self.scheduler._schedule_sequence:
            raise ValueError("missing_pre_action_decision_state")
        decision = result.cost_decision.to_record() if result.cost_decision else {}
        selected = decision.get("selected_cap")
        scores = [s for s in decision.get("candidates", ()) if s["cap"] == selected]
        keys = {q:decision_state_key(s) for q,s in result.candidate_decision_states if s.status == "SUPPORTED"}
        key = scores[0]["decision_state_key_json"] if len(scores) == 1 else None
        if key != keys.get(selected):
            key = None
        payload = dict(capture_phase="pre_action_plan_return", decision_state=state.to_record(),
            decision_state_key=key, decision=decision, request_incarnations=dict(state.request_incarnations),
            K_base=dict(result.per_request_base_k), selected_cost_cap=selected,
            K_governed=dict(result.per_request_governed_k), K_applied=dict(result.per_request_k),
            planned_worker_k=result.worker_k,
            old_verification_width=dict(state.old_verification_width_by_request))
        binding = EagleRWDecisionBinding(state.scheduler_sequence, result.plan_sequence, canonical(payload), _ISSUED)
        if id(schedule) in self._pending:
            raise ValueError("duplicate_schedule_binding")
        self._pending[id(schedule)] = dict(schedule=schedule, binding=binding, output=None,
            submitted=False, physical=None, feedback={}, consumed={}, failure=None)

    def _schedule_hook(self, delegate):
        def scheduled(schedule):
            previous, self._scheduling = self._scheduling, schedule
            try:
                result = delegate(schedule)
                def bind():
                    entry = self._pending.get(id(schedule))
                    execution = self.scheduler._lineage.get(schedule)
                    if (entry is None or entry["schedule"] is not schedule or execution is None
                            or execution.schedule_sequence != entry["binding"].scheduler_sequence):
                        raise ValueError("schedule_lineage_mismatch")
                    for rid, origin in execution.producers.items():
                        if origin.plan_sequence != entry["binding"].plan_sequence:
                            raise ValueError("plan_lineage_mismatch")
                self._observe("schedule_binding", schedule, bind)
                return result
            finally:
                self._scheduling = previous
        return scheduled

    def _entry(self, schedule):
        entry = self._pending.get(id(schedule))
        if entry is None or entry["schedule"] is not schedule:
            raise ValueError("missing_exact_schedule_witness")
        return entry

    def _result(self, schedule, result):
        if result is None:
            return
        if hasattr(result, "req_ids"):
            entry = self._entry(schedule)
            if entry["output"] is not None and entry["output"] is not result:
                raise ValueError("multiple_output_witnesses")
            entry["output"] = result
        elif callable(getattr(result, "result", None)):
            def materialization(delegate):
                def materialize(*args, **kwargs):
                    try:
                        value = delegate(*args, **kwargs)  # only the engine calls this wrapper
                    except BaseException as error:
                        self.failure(self._failure_kind(error), schedule, detail=type(error).__name__)
                        raise
                    self._observe("output_materialization", schedule, lambda:self._result(schedule, value))
                    return value
                return materialize
            self._patch(result, "result", materialization)
            if callable(getattr(result, "add_done_callback", None)):
                def done(future):
                    def terminal():
                        if future.cancelled():
                            self.failure("abort", schedule, detail="future_cancelled")
                        else:
                            error = future.exception()  # completed future only; never materialize output
                            if error is not None:
                                self.failure(self._failure_kind(error), schedule, detail=type(error).__name__)
                    self._observe("completed_future", schedule, terminal)
                result.add_done_callback(done)

    @staticmethod
    def _failure_kind(error):
        if isinstance(error, TimeoutError):
            return "timeout"
        if type(error).__name__ == "OutOfMemoryError":
            return "oom"
        return "error"

    def _execution_hook(self, delegate):
        def execute(schedule, *args, **kwargs):
            previous, self._executing = self._executing, schedule
            self._observe("worker_submission", schedule, lambda:self._entry(schedule).update(submitted=True))
            try:
                result = delegate(schedule, *args, **kwargs)
                self._observe("execution_output", schedule, lambda:self._result(schedule, result))
                return result
            except BaseException as error:
                self.failure(self._failure_kind(error), schedule, detail=type(error).__name__)
                raise
            finally:
                self._executing = previous
        return execute

    def _sampling_hook(self, delegate):
        def sample(*args, **kwargs):
            schedule = self.executor._lineage_pending
            previous, self._executing = self._executing, schedule
            try:
                result = delegate(*args, **kwargs)
                if schedule is not None:
                    self._observe("sampling_output", schedule, lambda:self._result(schedule, result))
                return result
            except BaseException as error:
                self.failure(self._failure_kind(error), schedule, detail=type(error).__name__)
                raise
            finally:
                self._executing = previous
        return sample

    def _proposal_hook(self, delegate):
        def propose(*args, **kwargs):
            result = delegate(*args, **kwargs)
            schedule = self._executing
            def physical():
                shape = tuple(result.shape)
                if len(shape) != 2 or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError("invalid_physical_draft_shape")
                entry = self._entry(schedule)
                if entry["physical"] is not None:
                    raise ValueError("duplicate_physical_proposal")
                entry["physical"] = shape
            self._observe("physical_proposal", schedule, physical)
            return result
        return propose

    def _feedback_hook(self, delegate):
        def finalized(event):
            result = delegate(event)
            if event is not None and self._retiring is not None:
                def capture():
                    entry = self._entry(self._retiring)
                    key = event.observation.key
                    if key.verifier_sequence != entry["binding"].scheduler_sequence or key.request_id in entry["feedback"]:
                        raise ValueError("feedback_verifier_mismatch")
                    entry["feedback"][key.request_id] = asdict(event)
                self._observe("post_stop_feedback", self._retiring, capture)
            return result
        return finalized

    def _retirement_hook(self, delegate):
        def retire(schedule, output):
            previous, self._retiring = self._retiring, schedule
            def before():
                entry = self._entry(schedule)
                ledger = self.scheduler._lineage.get(schedule)
                if ledger is None:
                    raise ValueError("missing_retiring_execution")
                entry["consumed"] = {rid:dict(origin=asdict(origin), verified_slots=count)
                                     for rid,(origin,count) in ledger.consumed.items()}
                entry["raw_sampled_counts"] = {rid:len(output.sampled_token_ids[output.req_id_to_index[rid]])
                                               for rid in output.req_ids}
            self._observe("retiring_origin", schedule, before)
            try:
                result = delegate(schedule, output)
                end = self._observe("monotonic_clock", schedule, self.clock)
                if type(end) is int:
                    self._observe("retirement_record", schedule, lambda:self._finish(schedule,output,result,end))
                elif id(schedule) in self._pending:
                    self.failure("observer_failure", schedule, detail="invalid_monotonic_clock")
                return result
            except BaseException as error:
                self.failure(self._failure_kind(error), schedule, detail=type(error).__name__)
                raise
            finally:
                self._retiring = previous
        return retire

    def _base_payload(self, binding, end):
        b = json.loads(binding.payload_json) if binding else {}
        return dict(record_schema_version="eagle_rw_retirement_v1", source_git_sha=self.plan.source.source_git_sha,
            source_manifest_hash=digest(self.plan.source.code_manifest), synthetic=self.plan.source.synthetic,
            profile_protocol_version=PROTOCOL, runtime_config_hash=self.plan.runtime_config_hash,
            collection_plan_hash=self.plan.collection_plan_hash, runtime_identity_snapshot=asdict(self.planner.runtime_identity),
            cohort_id=self.plan.cohort.cohort_id, cohort_descriptor=asdict(self.plan.cohort),
            retirement_window_id=f"{self.plan.cohort.cohort_id}:{self.clock_domain}:r{len(self._records)+1}",
            clock=CLOCK, clock_domain=self.clock_domain, timing_boundary=TIMING_BOUNDARY,
            monotonic_start=None, monotonic_end=end, consumer_cost_ms=None, consumer_cost_unit=UNIT,
            k0_semantics=K0_SEMANTICS, plan_sequence=binding.plan_sequence if binding else None,
            schedule_sequence=binding.scheduler_sequence if binding else None,
            decision_binding=json.loads(binding.payload_json) if binding else None,
            decision_state_key=b.get("decision_state_key"), request_incarnations=b.get("request_incarnations"),
            K_base=b.get("K_base"), selected_cost_cap=b.get("selected_cost_cap"),
            K_governed=b.get("K_governed"), K_applied=b.get("K_applied"),
            planned_worker_k=b.get("planned_worker_k"), old_verification_width=b.get("old_verification_width"),
            physical_draft_width_when_observed=None, physical_draft_rows_when_observed=None,
            drafted=None, accepted=None, verified_if_available=None, actual_useful_emitted_tokens=None,
            feedback_verifier_origin=None, finalization_status=None, failure_status=None)

    def _append(self, payload, reasons):
        # All canonical records are issued only by this path, from owned hooks.
        encoded = canonical(payload)
        reasons = tuple(dict.fromkeys(reasons))
        raw_hash = digest(payload)
        record = EagleRWRetirementRecord(encoded,reasons,raw_hash,_ISSUED,digest([raw_hash,reasons]))
        self._records.append(record)
        return record

    def failure(self, status, schedule=None, *, detail=None):
        """Retain failed/unsupported attempts; never convert them to samples."""
        entry = self._pending.pop(id(schedule), None) if schedule is not None else None
        if entry is not None and entry["schedule"] is not schedule:
            entry = None
        payload = self._base_payload(entry["binding"] if entry else None, None)
        payload.update(failure_status=status, failure_detail=detail, finalization_status="INVALID")
        if entry is not None:
            payload.update(feedback_verifier_origin=entry["consumed"],feedback_finalizations=entry["feedback"],
                raw_sampled_counts_before_stop=entry.get("raw_sampled_counts"),worker_submitted=entry["submitted"])
            if entry["physical"] is not None:
                rows,width = entry["physical"]
                payload.update(physical_draft_width_when_observed=width,physical_draft_rows_when_observed=rows)
        self._anchor = None
        return self._append(payload, (status,))

    def _finish(self, schedule, output, result, end):
        entry = self._entry(schedule)
        binding = entry["binding"]
        b = json.loads(binding.payload_json)
        payload = self._base_payload(binding,end)
        payload.update({k:b[k] for k in ("decision_state_key","request_incarnations","K_base","selected_cost_cap",
            "K_governed","K_applied","planned_worker_k","old_verification_width")})
        payload.update(raw_sampled_counts_before_stop=entry.get("raw_sampled_counts"),
                       feedback_verifier_origin=entry["consumed"], feedback_finalizations=entry["feedback"])
        reasons = []
        if binding._proof is not _ISSUED or b.get("capture_phase") != "pre_action_plan_return":
            reasons.append("decision_state_not_captured_before_action")
        if self.planner.runtime_identity != self.plan.runtime_identity:
            reasons.append("identity_mismatch")
        if not self._source_verified:
            reasons.append("dirty_or_unfrozen_execution_source")
        if b["request_incarnations"] != dict(self.plan.cohort.request_incarnations):
            reasons.append("cohort_incarnation_mismatch")
        if (not b["decision"].get("cost_policy_applied") or b["decision_state_key"] is None
                or b["decision_state_key"] != self.plan.target_decision_state_key):
            reasons.append("unsupported_decision_state")
        if not entry["submitted"] or entry["output"] is not output:
            reasons.append("pairing_failure")
        if entry["physical"] is not None:
            rows,width = entry["physical"]
            payload.update(physical_draft_width_when_observed=width,physical_draft_rows_when_observed=rows)
            if width != b["planned_worker_k"] or rows != len(b["K_applied"]):
                reasons.append("planned_physical_mismatch")
        elif b["planned_worker_k"] == 0:
            reasons.append("missing_explicit_same_engine_k0_proposal")
        if b["decision_state_key"] is not None:
            key = json.loads(b["decision_state_key"])
            ids = list(b["decision_state"]["eligible_decode_request_ids"])
            if ([b["K_applied"].get(r) for r in ids] != key["candidate_future_widths"]
                    or b["planned_worker_k"] != key["candidate_shared_worker_k"]):
                reasons.append("candidate_action_transition")
        emitted, final_status = {}, {}
        for batch in result.values():
            for row in batch.outputs:
                if row.request_id in emitted:
                    reasons.append("invalid_finalization")
                emitted[row.request_id] = len(row.new_token_ids)
                final_status[row.request_id] = "TERMINAL_CENSORED" if row.finish_reason is not None else "ELIGIBLE_NONTERMINAL"
        if set(emitted) != set(b["K_applied"]):
            reasons.append("invalid_finalization")
        drafted,accepted = {},{}
        for rid,old in b["old_verification_width"].items():
            event = entry["feedback"].get(rid)
            if old == 0 and event is None:
                drafted[rid],accepted[rid] = 0,0
                continue  # explicitly no positive draft origin is applicable
            consumed = entry["consumed"].get(rid)
            if event is None or consumed is None:
                reasons.append("invalid_finalization")
                continue
            observation = event["observation"]
            origin, key = consumed["origin"],observation["key"]
            expected = (rid,b["request_incarnations"].get(rid),origin["plan_sequence"],origin["schedule_sequence"],binding.scheduler_sequence)
            actual = tuple(key[k] for k in ("request_id","incarnation","source_plan_sequence","source_schedule_sequence","verifier_sequence"))
            if (actual != expected or event["emitted_tokens"] != emitted.get(rid)
                    or event["finalization_status"] != final_status.get(rid)):
                reasons.append("invalid_finalization")
            drafted[rid],accepted[rid] = observation["D"],observation["A"]
        payload.update(drafted=drafted,accepted=accepted,actual_useful_emitted_tokens=emitted,
                       finalization_status=final_status)
        signature = digest(dict(key=b["decision_state_key"],applied=b["K_applied"],
            base=b["K_base"],governed=b["K_governed"],cohort=self.plan.cohort.cohort_id))
        if type(end) is not int or end < 0:
            reasons.append("invalid_clock")
        start = self._anchor[1] if self._anchor and self._anchor[0] == signature else None
        if start is None:
            reasons.append("anchor_retirement")
        elif end <= start:
            reasons.append("invalid_clock")
        else:
            payload.update(monotonic_start=start, consumer_cost_ms=(end-start)/1e6)
        record = self._append(payload,reasons)
        self._pending.pop(id(schedule))
        self._anchor = ((signature,end) if not [r for r in reasons if r != "anchor_retirement"]
                        and all(s == "ELIGIBLE_NONTERMINAL" for s in final_status.values()) else None)
        return record

    @property
    def records(self):
        return tuple(self._records)

    def export(self):
        records = [r.to_data() for r in self.records]
        source_valid = self.plan.source.valid()  # outside the canonical timing path
        if not source_valid:
            for row in records:
                row["canonicality_status"] = "INVALID"
                row["unsupported_reasons"].append("dirty_or_unfrozen_execution_source")
        canonical_rows = [r for r in records if r["canonicality_status"] == "CANONICAL"]
        samples = [r for r in canonical_rows if r["finalization_status"]
                   and all(v == "ELIGIBLE_NONTERMINAL" for v in r["finalization_status"].values())]
        errors = [r for r in records if r["canonicality_status"] == "INVALID"
                  or r["unsupported_reasons"] not in ([],["anchor_retirement"])]
        payload = dict(evidence_layer="G4_EAGLE_RW_RETIREMENTS",consumer_profile=False,
            canonical_profile_input=(len(samples)==self.plan.sample_plan.retirements_per_cohort and not errors
                                     and source_valid),
            collection_plan=self.plan.payload(),collection_plan_hash=self.plan.collection_plan_hash,
            records=records,performance_claim=False)
        return dict(payload,raw_sha256=digest(payload))


def _record_errors(row):
    required = ("source_git_sha", "profile_protocol_version", "runtime_config_hash", "collection_plan_hash",
                "runtime_identity_snapshot", "cohort_id", "retirement_window_id", "decision_state_key",
                "request_incarnations", "K_base", "selected_cost_cap", "K_governed", "K_applied",
                "planned_worker_k", "old_verification_width", "actual_useful_emitted_tokens",
                "drafted", "accepted", "feedback_verifier_origin", "finalization_status", "decision_binding", "clock_domain")
    errors = ["missing_"+name for name in required if name not in row or row[name] is None]
    if errors:
        return errors
    if not _sha(row["source_git_sha"]):
        errors.append("invalid_source_sha")
    identity = RuntimeIdentitySnapshotV1(**row["runtime_identity_snapshot"])
    try:
        if row["runtime_config_hash"] != runtime_config_hash(identity):
            errors.append("runtime_config_hash_mismatch")
    except ValueError:
        errors.append("unsupported_identity")
    if (row.get("profile_protocol_version") != PROTOCOL or row.get("timing_boundary") != TIMING_BOUNDARY
            or row.get("consumer_cost_unit") != UNIT or row.get("clock") != CLOCK
            or row.get("k0_semantics") != K0_SEMANTICS or row.get("failure_status") is not None):
        errors.append("invalid_measurement_contract")
    start,end = row.get("monotonic_start"),row.get("monotonic_end")
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start
            or row.get("consumer_cost_ms") != (end-start)/1e6):
        errors.append("invalid_retirement_cost")
    if row["decision_binding"].get("capture_phase") != "pre_action_plan_return":
        errors.append("post_action_decision_state")
    if not row["cohort_id"] or row["cohort_id"] != row.get("cohort_descriptor",{}).get("cohort_id"):
        errors.append("cohort_provenance_mismatch")
    binding = row["decision_binding"]
    for name in ("decision_state_key","request_incarnations","K_base","selected_cost_cap",
                 "K_governed","K_applied","planned_worker_k","old_verification_width"):
        if row[name] != binding.get(name):
            errors.append("binding_mismatch:"+name)
    decision = binding["decision"]
    cap = row["selected_cost_cap"]
    if type(cap) is not int or cap not in (0,2,3,4):
        errors.append("invalid_cost_cap")
    for name in ("K_base","K_governed","K_applied","old_verification_width","drafted","accepted",
                 "actual_useful_emitted_tokens"):
        if any(type(v) is not int or v < 0 for v in row[name].values()):
            errors.append("invalid_scalar_accounting:"+name)
    if row["planned_worker_k"] == 0 and row.get("physical_draft_width_when_observed") != 0:
        errors.append("missing_same_engine_k0_observation")
    if (row.get("physical_draft_width_when_observed") is not None
            and (row["physical_draft_width_when_observed"] != row["planned_worker_k"]
                 or row.get("physical_draft_rows_when_observed") != len(row["K_applied"]))):
        errors.append("planned_physical_mismatch")
    scores = [s for s in decision.get("candidates",()) if s["cap"] == cap]
    if (not decision.get("cost_policy_applied") or decision.get("profile_identity_status") != "VALID_MATCH"
            or len(scores) != 1 or scores[0]["decision_state_key_json"] != row["decision_state_key"]
            or row["K_base"] != decision.get("base_k")):
        errors.append("policy_key_mismatch")
    key = json.loads(row["decision_state_key"])
    ids = binding["decision_state"]["eligible_decode_request_ids"]
    if (key["runtime_identity_key"] != identity.identity_key or key["runtime_mode_id"] != identity.runtime_mode_id
            or key["candidate_future_widths"] != [row["K_applied"][r] for r in ids]
            or key["old_verification_widths"] != [row["old_verification_width"][r] for r in ids]
            or row["planned_worker_k"] != max(row["K_applied"].values())
            or key["candidate_shared_worker_k"] != row["planned_worker_k"]
            or row["request_incarnations"] != dict(row["cohort_descriptor"]["request_incarnations"])
            or any(not 0 <= row["K_applied"][r] <= row["K_governed"][r] <= min(row["K_base"][r],cap) for r in ids)):
        errors.append("state_or_k_provenance_mismatch")
    for rid,old in row["old_verification_width"].items():
        event = row.get("feedback_finalizations",{}).get(rid)
        if old == 0 and event is None:
            if row["drafted"].get(rid) != 0 or row["accepted"].get(rid) != 0:
                errors.append("invalid_zero_width_feedback")
            continue
        origin = row["feedback_verifier_origin"].get(rid,{}).get("origin")
        if event is None or origin is None:
            errors.append("missing_origin_finalization")
            continue
        obs = event["observation"]
        if (event["emitted_tokens"] != row["actual_useful_emitted_tokens"].get(rid)
                or event["finalization_status"] != row["finalization_status"].get(rid)
                or obs["A"] != row["accepted"].get(rid) or obs["D"] != row["drafted"].get(rid)
                or obs["key"] != dict(request_id=rid,incarnation=row["request_incarnations"][rid],
                    source_plan_sequence=origin["plan_sequence"],source_schedule_sequence=origin["schedule_sequence"],
                    verifier_sequence=row["schedule_sequence"])):
            errors.append("post_stop_token_or_origin_mismatch")
    return errors


def _validate_export(data):
    """Validate a serialized evidence envelope; it is never a published profile."""
    data = dict(data)
    raw_hash = data.pop("raw_sha256",None)
    if (data.get("consumer_profile") is not False or data.get("performance_claim") is not False
            or data.get("evidence_layer") != "G4_EAGLE_RW_RETIREMENTS" or raw_hash != digest(data)):
        raise ValueError("noncanonical_evidence_envelope")
    plan = data["collection_plan"]
    if data["collection_plan_hash"] != digest(plan):
        raise ValueError("collection_plan_hash_mismatch")
    for row in data["records"]:
        if (row.get("cohort_id") != plan["cohort"]["cohort_id"]
                or row.get("collection_plan_hash") != data["collection_plan_hash"]
                or row.get("source_git_sha") != plan["source_git_sha"]
                or row.get("runtime_config_hash") != plan["runtime_config_hash"]
                or canonical(row.get("runtime_identity_snapshot")) != canonical(plan["expected_runtime_identity"])):
            raise ValueError("record_provenance_mismatch")
        payload = {k:v for k,v in row.items() if k not in ("canonicality_status","unsupported_reasons","raw_sha256")}
        if digest(payload) != row["raw_sha256"]:
            raise ValueError("record_raw_hash_mismatch")
        if row["canonicality_status"] == "CANONICAL" and (row["unsupported_reasons"] or _record_errors(row)):
            raise ValueError("incomplete_canonical_record")
    samples = [r for r in data["records"] if r["canonicality_status"] == "CANONICAL"
               and all(v == "ELIGIBLE_NONTERMINAL" for v in r["finalization_status"].values())]
    errors = [r for r in data["records"] if r["canonicality_status"] == "INVALID"
              or r["unsupported_reasons"] not in ([],["anchor_retirement"])]
    expected = len(samples) == plan["fixed_sample_plan"]["retirements_per_cohort"] and not errors
    if data["canonical_profile_input"] != expected:
        raise ValueError("manually_forced_canonical_profile_input")
    return True


def validate_export(data):
    """Reject malformed or manually relabelled serialized collector evidence."""
    try:
        return _validate_export(data)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed_collector_evidence") from error
