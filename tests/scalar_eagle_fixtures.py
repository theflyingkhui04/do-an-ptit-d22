"""Explicitly synthetic CPU fixtures. No observed GPU compatibility is claimed."""
from dataclasses import replace, asdict
from types import SimpleNamespace as NS

from flexsds.core.eagle_consumer_profile import (
    EagleConsumerProfile, ConsumerCostCell, decision_state_key,
)
from src.server.decision_state import (
    CONFIGURED_FIELDS, OBSERVED_FIELDS, RuntimeIdentitySnapshotV1, DecisionStateSnapshotV1,
)
from src.server.budget_planner import AdaptiveBudgetPlanner
from src.server.lineage import DraftOrigin
from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.cost_policy import PendingVerifierEvent, SurvivalEvent, VerifierEventKey
from src.proposer.cost_policy import ScalarEagleCostGovernor


def identity():
    observed = dict(zip(OBSERVED_FIELDS, (
        "SYNTHETIC CPU fixture GPU", "synthetic-sm", 24000, "0.30.0", "synthetic-torch",
        "synthetic-cuda", "synthetic-driver", "bfloat16", "none")))
    configured = {k: False for k in CONFIGURED_FIELDS}
    configured.update(target_model="synthetic-target", target_revision="a" * 40,
        draft_model="synthetic-draft", draft_revision="b" * 40, draft_method="eagle3",
        scheduler_class="synthetic-async", executor_class="synthetic-uniproc",
        tensor_parallel_size=1, async_scheduling=True, compilation_mode="none",
        max_model_len=4096, max_num_seqs=8, gpu_memory_utilization=.9,
        cudagraph_mode="none", cache_dtype="auto", cache_block_size=16,
        speculative_ceiling=4, max_num_batched_tokens=256,
        configured_dtype="bfloat16", configured_quantization="none")
    return RuntimeIdentitySnapshotV1(tuple(observed.items()), tuple(configured.items()))


def snapshot(p, ids=("A", "B"), sequence=1, old=None, contexts=None, pure=True):
    old = old or {r: 4 for r in ids}
    contexts = contexts or {r: 127 for r in ids}
    return DecisionStateSnapshotV1(sequence,
        tuple((r, p._request_incarnations[r]) for r in ids), ids, ids, pure, not pure,
        tuple(old.items()), tuple(contexts.items()), p.runtime_identity.runtime_mode_id)


def planner(base=None, mode="strict", costs=(1., 1.8, 2.3, 3.2), ready=True,
            window=20, hold=0, floor=0, telemetry=None):
    base = base or {"A": 4, "B": 2}
    p = AdaptiveBudgetPlanner(AdaptiveProposerAdapter(controller_kwargs={
        "window_size": window, "initial_alpha": .95, "min_k": floor}),
        replacement_policy=True, recovery_hold_steps=hold, cost_policy_mode=mode,
        step_telemetry_path=str(telemetry) if telemetry else None)
    # Explicit original-contract regression fixture, not the active loader/path.
    # Pulse migration has a separate B1/base3/old0 production causal witness.
    p.cost_governor = ScalarEagleCostGovernor(mode)
    p.bind_runtime_identity(identity())
    for r, b in base.items():
        p.bind_request(r, 1)
        p.adapter.get_controller(r).reset(initial_alpha={0: .4, 2: .7, 3: .85, 4: .95}[b])
    s = snapshot(p, tuple(base))
    states = p._candidate_decision_states(s, base, ())
    p.cost_governor.profile = profile_for(p, states, costs)
    if ready and mode != "off":
        for r, b in base.items():
            values = ([0]*2 + [1]*3 + [2]*4 + [3]*4 + [4]*7 if b == 4 else
                      [0]*4 + [1]*6 + [2]*10 if b == 2 else [b]*20)
            for n, a in enumerate(values[-window:], 1):
                seed(p, r, b, a, n)
            p.adapter.get_controller(r).reset(initial_alpha={0: .4, 2: .7, 3: .85, 4: .95}[b])
    return p, s


def profile_for(p, states, costs):
    return EagleConsumerProfile("synthetic-CPU-policy", p.runtime_identity,
        tuple((decision_state_key(s), ConsumerCostCell.synthetic_fixture(
            *costs[q] if isinstance(costs, dict) and isinstance(costs[q], tuple) else
            (costs[q] if isinstance(costs, dict) else costs[(0, 2, 3, 4).index(q)],)))
            for q, s in states), synthetic=True)


def seed(p, rid, base, a, seq, *, d=None, status="ELIGIBLE_NONTERMINAL", mode="normal"):
    d = base if d is None else d
    epoch = p._request_incarnations[rid]
    origin = DraftOrigin(rid, epoch, seq, seq, max(base, d), max(base, d),
                         source_base_k=base, source_decision_mode=mode)
    p.record_verification(rid, d, a, source=origin, attributed=True, verifier_id=seq)
    row = NS(request_id=rid, new_token_ids=list(range(a + 1)),
             finish_reason=None if status == "ELIGIBLE_NONTERMINAL" else "stop")
    p.finalize_cost_feedback(seq, {0: NS(outputs=[row])},
        invalid_status=None if status in ("ELIGIBLE_NONTERMINAL", "TERMINAL_CENSORED") else status)


def event(a=4, d=4, base=4, seq=1, epoch=1, mode="normal", status="ELIGIBLE_NONTERMINAL", lineage="attributed"):
    return SurvivalEvent(PendingVerifierEvent(VerifierEventKey("A", epoch, seq, seq, seq),
        base, d, d, a, d, mode, lineage), status, a + 1)


def plan(p, state, kv=.1):
    return p.plan_step(state.eligible_decode_request_ids, len(state.eligible_decode_request_ids),
                       kv, decision_state=state)
