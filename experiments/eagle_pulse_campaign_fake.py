"""CPU-only executor using the active planner/collector, never an external API."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace as NS

from experiments.eagle_pulse_campaign import (
    CELLS, PRODUCTS, ROOT, CampaignError, canonical, digest, read_json, require,
)


def make_plan(source_sha, max_allocations=8, synthetic=False):
    contract = read_json(ROOT / "experiments/eagle_pulse_campaign_contract.json")
    domain = dict(
        preparation=dict(procedure="fresh-request-neutral-prefill-old0-v1", prompt=contract["prompt"],
                         tokenizer_revision=contract["engine_args"]["revision"], max_tokens=64),
        workload=dict(prompt=contract["prompt"], tokenizer=contract["engine_args"]["model"],
                      tokenizer_revision=contract["engine_args"]["revision"], add_special_tokens=False),
        warm_runtime_cache="warm-loaded-engine-reset-prefix-cache-between-cohorts-v1",
        sampling="greedy_no_stops_v1",
        queue=dict(capacity=2, driver="native_enginecore_v030", pending="bounded_native_placeholders", continuation=0),
        controller_recovery=dict(window_size=4, initial_alpha=.85, candidate_budgets=[0, 2, 3],
                                 max_k=3, recovery="production-default", vram_guard=True),
        source_observer=dict(source_git_sha=source_sha, collector="EaglePulseCycleCollector-v1"),
        cohort_failure_plan="fixed50-no-replacement-v1", seeds=[731], continuation=0,
        replicate_definition="fresh-request-incarnation-on-stable-worker;not-cross-VM-independent",
    )
    return dict(schema_version="secp-pulse-2r-campaign-plan-v1", objective="SECP-PULSE-2R-v1",
        source_git_sha=source_sha, products=list(PRODUCTS), cells=deepcopy(list(CELLS)),
        cohort_slots_per_cell=50, bootstrap_resamples=5000, confidence_level=.95,
        bootstrap_seed=731, max_allocations=max_allocations,
        modal=dict(requested_gpu_selector="A10G", max_containers=1, retries=0,
                   timeout_seconds=3600, startup_timeout_seconds=None),
        runtime_contract_hash=digest(contract["identities"]),
        preparation_contract_hash=digest(domain["preparation"]), queue_contract_hash=digest(domain["queue"]),
        continuation_contract_hash=digest(domain["continuation"]),
        failure_policy="fixed-slot-no-replacement-v1", identities=contract["identities"],
        domain=domain, engine_args=contract["engine_args"], synthetic=synthetic)


def synthetic_cycle(plan, product, cell, slot):
    from experiments.eagle_rw_collector import EaglePulseCycleCollector, EagleRWSourceProvenance, EagleRWCohortDescriptor
    from src.proposer.adapter import AdaptiveProposerAdapter
    from src.server.budget_planner import AdaptiveBudgetPlanner
    from src.server.decision_state import RuntimeIdentitySnapshotV1, DecisionStateSnapshotV1, PulseContextV1
    from src.server.lineage import DraftOrigin, Execution
    q = next(c["future_k"] for c in CELLS if c["id"] == cell)
    cid = f"{product.removeprefix('NVIDIA ')}:{cell}:{slot}"
    rid = "synthetic-" + cid
    p = AdaptiveBudgetPlanner(AdaptiveProposerAdapter(controller_kwargs=dict(window_size=4,
        initial_alpha=.85, candidate_budgets=[0, 2, 3])), max_k=3,
        replacement_policy=True, cost_policy_mode="strict")
    p.bind_runtime_identity(RuntimeIdentitySnapshotV1(**plan["identities"][product]))
    p.bind_request(rid, 1)
    p.pulse_domain_json = canonical(plan["domain"])
    state = DecisionStateSnapshotV1(1, ((rid, 1),), (rid,), (rid,), True, False,
        ((rid, 0),), ((rid, 127),), p.runtime_identity.runtime_mode_id)
    context = PulseContextV1(p.pulse_domain_json, rid, 1, 1, 64, 1921, 0, 2, True)
    collector = EaglePulseCycleCollector(NS(scheduler=NS(_planner=p)),
        EagleRWSourceProvenance.cpu_fixture(plan["source_git_sha"]),
        EagleRWCohortDescriptor(cid, ((rid, 1),)), q, p.pulse_domain_json).install()
    try:
        first = p.plan_step([rid], 1, .1, decision_state=state, pulse_context=context)
        second_state = replace(state, scheduler_sequence=2, old_verification_width_by_request=((rid, q),))
        second = p.plan_step([rid], 1, .1, decision_state=second_state,
                             pulse_context=replace(context, schedule_sequence=2))
        def execution(step):
            return Execution(object(), {rid: DraftOrigin(rid, 1, step.plan_sequence,
                step.decision_state.scheduler_sequence, step.per_request_k[rid], step.worker_k,
                step.worker_k, step.per_request_base_k[rid], step.per_request_source_modes[rid])},
                schedule_sequence=step.decision_state.scheduler_sequence)
        for step, useful in ((first, 1), (second, 1 + q)):
            seq = step.decision_state.scheduler_sequence
            p.record_pulse_proposal(seq, 1, step.worker_k)
            if step is second and q:
                p.record_verification(rid, q, q, source=execution(first).producers[rid], attributed=True, verifier_id=seq)
            outputs = {0: NS(outputs=[NS(request_id=rid, new_token_ids=[10] * useful, finish_reason=None)])}
            p.finalize_cost_feedback(seq, outputs)
            p.finalize_pulse_cycle(seq, execution(step), outputs)
        collector.close()
        row = deepcopy(collector.export()["records"][0])
        require(row["status"] == "COMPLETE", "fake_native_cycle_incomplete")
        for receipt, step in ((row["pulse_receipt"], first), (row["continuation_receipt"], second)):
            receipt["base_k"] = step.per_request_base_k[rid]
        # Deterministic CPU fixture duration, explicitly synthetic; no measured claim.
        row.update(start_ns=1000000, end_ns=2000000 + q * 100000, cost_ms=1 + q / 10,
                   clock_domain="synthetic:" + cid, campaign_plan_sha256=digest(plan))
        return row
    finally:
        collector.close()


class FakeExecutor:
    synthetic = True

    def __init__(self, sequence, *, slots_per_worker=150, failures=None, cycle_factory=synthetic_cycle):
        self.sequence = iter(sequence)
        self.slots_per_worker = slots_per_worker
        self.failures = failures or {}
        self.cycle_factory = cycle_factory
        self.allocations = []
        self.attempts = []
        self.released = []

    def allocate(self, plan, index):
        label = next(self.sequence, "unsupported")
        product = {"A10": "NVIDIA A10", "A10G": "NVIDIA A10G"}.get(label, label)
        identity = deepcopy(plan["identities"].get(product, dict(observed=[["gpu_name", product]],
                                                               configured=[], schema_version="runtime_identity_snapshot_v1")))
        worker = dict(index=index, identity=identity, remaining=self.slots_per_worker,
                      history=None, stable_worker_id=f"fake-worker-{index}")
        self.allocations.append(worker)
        return worker

    def observe_identity(self, worker):
        return deepcopy(worker["identity"])

    def available(self, worker):
        return worker["remaining"] > 0

    def execute_slot(self, worker, plan, product, cell, slot, attempt_id):
        require(worker["history"] is None, "history_cache_leak")
        require(dict(worker["identity"]["observed"])["gpu_name"] == product, "cross_product_write")
        self.attempts.append((product, cell, slot))
        worker["remaining"] -= 1
        row = self.cycle_factory(plan, product, cell, slot)
        failure = self.failures.get((product, cell, slot))
        if failure:
            row.update(status="INVALID", failures=[failure])
        return dict(record=row, identity=deepcopy(worker["identity"]), failure=failure,
                    log="SYNTHETIC CPU fixture; " + worker["stable_worker_id"])

    def close(self, worker):
        require(worker["index"] not in self.released, "worker_double_release")
        self.released.append(worker["index"])
