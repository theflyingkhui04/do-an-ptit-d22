"""SECP-PULSE-2R-v1: exact two-retirement scoring and request-owned leases."""
from collections import deque
from dataclasses import asdict
import json
import math
import time
import os
import uuid

from src.proposer.cost_policy import CostDecision, SurvivalHistory
try:
    from flexsds.core.eagle_consumer_profile import (
        EaglePulseProfile, ConsumerProfileError, PULSE_OBJECTIVE, PULSE_UNIT,
        PULSE_REWARD_UNIT, canonical, digest, decision_state_key, pulse_state_payload,
    )
except ModuleNotFoundError as exc:
    if exc.name not in ("flexsds", "flexsds.core", "flexsds.core.eagle_consumer_profile"):
        raise
    from src.flexsds.core.eagle_consumer_profile import (
        EaglePulseProfile, ConsumerProfileError, PULSE_OBJECTIVE, PULSE_UNIT,
        PULSE_REWARD_UNIT, canonical, digest, decision_state_key, pulse_state_payload,
    )


class PulseSurvivalHistory(SurvivalHistory):
    """An isolated stratum, admitted only by completed normal cycle evidence."""
    def add(self, event):
        return False  # Legacy feedback can never populate this stratum.

    def __init__(self):
        super().__init__()
        self.domain_hash = None

    def bind_domain(self, value):
        if self.domain_hash is not None and self.domain_hash != value:
            raise ValueError("pulse_history_domain_changed")
        self.domain_hash = value

    def add_cycle(self, cycle, event):
        e = event.observation
        if (cycle["domain_hash"] != self.domain_hash
                or cycle["status"] != "COMPLETE" or cycle["purpose"] != "NORMAL"
                or cycle["q"] != 3 or cycle["continuation"] != 0
                or e.source_base_k != 3 or e.source_applied_k != 3
                or e.source_shared_k != 3 or e.D != 3
                or cycle["request_id"] != e.key.request_id or cycle["incarnation"] != e.key.incarnation
                or cycle["retirements"][1]["schedule_sequence"] != e.key.verifier_sequence):
            return False
        return super().add(event)


class PulseCostGovernor:
    objective = PULSE_OBJECTIVE

    def __init__(self, mode="off", profile=None, profile_error=None):
        if mode not in ("off", "strict", "compat"):
            raise ValueError("ADAPTIVE_COST_POLICY_MODE must be off, strict or compat")
        self.mode, self.profile, self.profile_error = mode, profile, profile_error

    def decide(self, *, runtime, candidate_states, base, source_modes, history, incarnations,
               pulse_context=None, snapshot=None):
        profile = self.profile
        evidence = dict(policy_version=PULSE_OBJECTIVE, objective=PULSE_OBJECTIVE, unit=PULSE_UNIT,
            reward_unit=PULSE_REWARD_UNIT, horizon=2, continuation=0, terminal_value=0,
            profile_id=getattr(profile,"profile_id",None), profile_hash=getattr(profile,"sha256",None),
            synthetic_profile=getattr(profile,"synthetic",None), base_k=dict(base),
            request_incarnations=dict(incarnations), source_modes=dict(source_modes), candidates=[],
            profile_identity_status="UNAVAILABLE", policy_mode=self.mode)
        def result(cap, mode, reason, applied=False, optimal=False):
            return CostDecision(cap,mode,reason,applied,optimal,
                applied and profile is not None and not profile.synthetic,canonical(evidence))
        def degraded(reason):
            return result(None if self.mode=="compat" else 0,
                          "compatibility" if self.mode=="compat" else "normal",reason)
        if "memory_override" in source_modes.values():
            return result(None,"memory_override","memory_or_kv_safety")
        if self.mode=="off":
            return result(None,"off","cost_policy_off")
        if not isinstance(profile,EaglePulseProfile):
            return degraded(self.profile_error or "missing_pulse_objective")
        evidence["profile_identity_status"] = profile.identity_status(runtime)
        if evidence["profile_identity_status"] != "VALID_MATCH":
            return degraded("identity_mismatch")
        try:
            domain_hash = pulse_context.validate(snapshot)
            if domain_hash != profile.domain_hash:
                return degraded("pulse_domain_mismatch")
            evidence["domain_hash"] = domain_hash
            evidence["pulse_context"] = asdict(pulse_context)
            states = dict(candidate_states)
            if (list(base.values()) != [3] or set(states) != {0,2,3}
                    or not isinstance(history,PulseSurvivalHistory)):
                return degraded("unsupported_pulse_state_or_history")
            for state in states.values():
                pulse_state_payload(state)
            costs = {q:profile.lookup(s) for q,s in states.items()}
            history.bind_domain(domain_hash)
        except (AttributeError, ValueError, TypeError, KeyError, ConsumerProfileError):
            return degraded("unsupported_exact_pulse_domain_or_cell")
        if "probe" in source_modes.values():
            return result(None,"probe","legacy_probe_exploration")
        rid = next(iter(base))
        try:
            estimate = history.estimate(rid,incarnations[rid],3)
        except (KeyError,ValueError):
            return degraded("unbound_pulse_history")
        evidence["survival"] = {rid:asdict(estimate)}
        if not estimate.ready:
            return result(3,"cost_warmup","pulse_survival_window_underfilled")
        baseline = 2/costs[0].C_low
        evidence["K0_R_upper"] = baseline
        best = None
        for q in sorted(states):
            y = estimate.predicted_cycle_reward(q)
            cost = costs[q]
            row = dict(cap=q, predicted_cycle_reward=y, C_point=cost.C_point,
                C_low=cost.C_low,C_high=cost.C_high,R_point=y/cost.C_point,
                R_lower=y/cost.C_high,R_upper=y/cost.C_low,
                supported=q>0 and y/cost.C_high>baseline,
                decision_state_key_json=decision_state_key(states[q]),raw_sha256=cost.raw_sha256)
            evidence["candidates"].append(row)
            if row["supported"] and (best is None or (row["R_lower"]>best["R_lower"]
                    and not math.isclose(row["R_lower"],best["R_lower"],rel_tol=1e-9,abs_tol=1e-12))):
                best = row
        overlap = any(row["cap"]>0 and row["R_upper"]>=2/costs[0].C_high
                      for row in evidence["candidates"])
        return result(best["cap"] if best else 0,"normal",
            "supported_conservative_pulse" if best else ("INCONCLUSIVE_COST" if overlap else "no_profitable_candidate"),
            True,best is not None)


class PulseCycleLease:
    """One pulse, one forced continuation; completion needs two exact outputs.

    Scheduling may run ahead of retirement. Until completion, later native
    queue-fill work stays neutral but receives no cycle reward credit.
    """
    def __init__(self, step, context, identity, *, purpose="NORMAL", cohort_id=None,
                 source_git_sha=None, synthetic=False):
        state = step.decision_state
        domain_hash = context.validate(state)
        rid = context.request_id
        q = step.per_request_k[rid]
        if (q not in (0,2,3) or step.per_request_base_k != {rid:3}
                or step.worker_k != q or dict(state.old_verification_width_by_request) != {rid:0}):
            raise ValueError("pulse_not_applied_exactly")
        initial = dict(step.candidate_decision_states)[q]
        pulse_state_payload(initial)
        self.context = context
        self.rid, self.epoch, self.q = rid, context.incarnation, q
        self.first = state.scheduler_sequence
        self.second = None
        self.feedback = None
        self.physical = {}
        self.record = dict(objective=PULSE_OBJECTIVE,unit=PULSE_UNIT,reward_unit=PULSE_REWARD_UNIT,
            horizon=2,continuation=0,terminal_value=0,status="AWAIT_CONTINUATION",
            cycle_id=f"{rid}:{self.epoch}:{step.plan_sequence}:{self.first}",
            cohort_id=cohort_id,request_id=rid,incarnation=self.epoch,q=q,purpose=purpose,
            domain_hash=domain_hash,initial_key=decision_state_key(initial),
            runtime_identity_key=identity.identity_key,source_git_sha=source_git_sha,synthetic=synthetic,
            clock_domain=f"enginecore:{os.getpid()}:{uuid.uuid4().hex}",
            decision=step.cost_decision.to_record() if step.cost_decision else None,
            pulse_receipt=dict(plan_sequence=step.plan_sequence,schedule_sequence=self.first,
                decision_state=asdict(state),applied_k=dict(step.per_request_k),worker_k=step.worker_k),
            continuation_receipt=None,retirements=[],failures=[],start_ns=time.perf_counter_ns(),
            end_ns=None,cost_ms=None,useful_tokens=None,clock="perf_counter_ns")
        self.record["added_cuda_sync"] = False

    @property
    def active(self):
        return self.record["status"] not in ("COMPLETE","INVALID","CENSORED")

    def invalidate(self, reason, *, censored=False):
        # A failure is sticky; later completions cannot repair it.
        self.record["failures"].append(reason)
        if self.active:
            self.record["status"] = "CENSORED" if censored else "INVALID"

    def confirm(self, step):
        if self.second is not None:
            return  # Native later neutral queue fill; never becomes a target.
        state = step.decision_state
        if (state is None or state.scheduler_sequence != self.first+1
                or state.status != "SUPPORTED" or state.full_scheduled_request_ids != (self.rid,)
                or not state.pure_decode
                or state.runtime_mode_id != self.record["pulse_receipt"]["decision_state"]["runtime_mode_id"]
                or dict(state.request_incarnations).get(self.rid) != self.epoch
                or dict(state.old_verification_width_by_request).get(self.rid) != self.q
                or step.per_request_k != {self.rid:0} or step.worker_k != 0):
            self.invalidate("continuation_lineage_or_action_mismatch")
            return
        self.second = state.scheduler_sequence
        self.record["continuation_receipt"] = dict(plan_sequence=step.plan_sequence,
            schedule_sequence=self.second,decision_state=asdict(state),applied_k=dict(step.per_request_k),worker_k=0)
        self.record["status"] = "AWAIT_POST_STOP_COMPLETION"

    def proposal(self, sequence, rows, width):
        if sequence not in (self.first,self.first+1) or not self.active:
            return
        if sequence in self.physical or rows != 1 or width != (self.q if sequence==self.first else 0):
            self.invalidate("physical_pulse_or_continuation_mismatch")
            return
        self.physical[sequence] = dict(rows=rows,width=width)

    def verifier(self, event):
        e = event.observation
        if e.key.verifier_sequence != self.first+1 or e.key.request_id != self.rid or not self.active:
            return
        if (self.feedback is not None or e.key.incarnation != self.epoch
                or e.key.source_schedule_sequence != self.first
                or e.key.source_plan_sequence != self.record["pulse_receipt"]["plan_sequence"]
                or e.D != self.q or e.source_applied_k != self.q or e.source_shared_k != self.q):
            self.invalidate("verifier_origin_mismatch")
            return
        self.feedback = event
        self.record["verifier_finalization"] = asdict(event)
        if event.finalization_status != "ELIGIBLE_NONTERMINAL":
            self.invalidate(event.finalization_status,censored=event.finalization_status=="TERMINAL_CENSORED")

    def retire(self, sequence, execution, outputs, invalid_status=None):
        if sequence not in (self.first,self.first+1):
            return
        # Retain clipped/failed native output even when feedback or native
        # request cleanup has already invalidated this cycle. Never repair it.
        raw = [dict(request_id=r.request_id,new_token_ids=list(getattr(r,"new_token_ids",())),
                    finish_reason=getattr(r,"finish_reason",None))
               for batch in (outputs or {}).values() for r in getattr(batch,"outputs",())
               if r.request_id==self.rid]
        self.record.setdefault("raw_poststop_outputs",[]).append(dict(schedule_sequence=sequence,outputs=raw))
        if not self.active:
            self.record["partial_end_ns"] = time.perf_counter_ns()
            return
        expected = self.first if not self.record["retirements"] else self.first+1
        if sequence != expected or execution is None:
            self.invalidate("retirement_order_or_execution_mismatch"); return
        rows = [r for batch in (outputs or {}).values() for r in getattr(batch,"outputs",())
                if r.request_id==self.rid]
        origin = execution.producers.get(self.rid)
        receipt = self.record["pulse_receipt"] if sequence==self.first else self.record["continuation_receipt"]
        if (invalid_status or receipt is None or origin is None or origin.incarnation != self.epoch
                or origin.schedule_sequence != sequence or origin.plan_sequence != receipt["plan_sequence"]
                or origin.request_k != (self.q if sequence==self.first else 0)
                or origin.batch_k != origin.request_k or len(rows)!=1
                or not isinstance(getattr(rows[0],"new_token_ids",None),(list,tuple))):
            self.invalidate(invalid_status or "retirement_lineage_or_output_mismatch"); return
        row = rows[0]
        item = dict(schedule_sequence=sequence,origin=asdict(origin),
                    useful_tokens=len(row.new_token_ids),finish_reason=getattr(row,"finish_reason",None),
                    physical=self.physical.get(sequence))
        self.record["retirements"].append(item)
        if item["finish_reason"] is not None:
            self.invalidate("terminal_or_clipped",censored=True); return
        if item["physical"] is None:
            self.invalidate("missing_actual_proposal_witness"); return
        if sequence==self.first and item["useful_tokens"]!=1:
            self.invalidate("old0_reward_mismatch"); return
        if sequence==self.first+1:
            if self.q and (self.feedback is None or self.feedback.emitted_tokens != item["useful_tokens"]):
                self.invalidate("missing_exact_finalized_verifier"); return
            accepted = self.feedback.observation.A if self.q else 0
            if item["useful_tokens"] != 1+accepted:
                self.invalidate("continuation_reward_mismatch"); return
            self.record.update(status="COMPLETE",end_ns=time.perf_counter_ns(),
                useful_tokens=sum(r["useful_tokens"] for r in self.record["retirements"]))
            self.record["cost_ms"] = (self.record["end_ns"]-self.record["start_ns"])/1e6


class PulseCycleState:
    def __init__(self):
        self.lease = None
        self.records = deque(maxlen=128)

    @property
    def active(self):
        return self.lease is not None and self.lease.active

    def archive(self, planner):
        lease = self.lease
        if lease is None or lease.active:
            return
        counted = False
        if lease.feedback is not None:
            counted = planner.pulse_history.add_cycle(lease.record,lease.feedback)
        lease.record["normal_survival_counted"] = counted
        record = json.loads(canonical(lease.record))
        self.records.append(record)
        planner._write_step(dict(record_type="pulse_cycle_finalized",**record))
        self.lease = None
