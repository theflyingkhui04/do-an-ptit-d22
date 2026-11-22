"""Native pinned retirement witnesses through the actual campaign worker loop."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest

from scalar_eagle_native_fixture import native_runtime, scheduler_module, CpuRequest, retire
from test_adaptive_scheduler_glue import add_running
from experiments.eagle_pulse_campaign_fake import make_plan
from experiments.eagle_pulse_campaign_worker import LiveWorker
from experiments.eagle_rw_collector import EagleRWSourceProvenance
import experiments.eagle_pulse_campaign as c


@pytest.mark.parametrize("q", [0, 2, 3], ids=["CAM-native-q0", "CAM-native-q2", "CAM-native-q3"])
def test_worker_native_queue_fresh_request_reset(native_runtime, q):
    from src.proposer.pulse_policy import PulseCostGovernor, PulseSurvivalHistory
    scheduler, old_request, _, status = native_runtime
    plan = make_plan(c.BASE_SHA, 8, True)
    p = scheduler._planner
    p.finish_request(old_request.request_id)
    scheduler.running.clear()
    scheduler.requests.clear()
    from src.server.budget_planner import AdaptiveBudgetPlanner
    from src.proposer.adapter import AdaptiveProposerAdapter
    p = AdaptiveBudgetPlanner(AdaptiveProposerAdapter(controller_kwargs=dict(window_size=4,
        initial_alpha=.85, candidate_budgets=[0, 2, 3])), max_k=3,
        replacement_policy=True, cost_policy_mode="strict")
    scheduler._planner = p
    p.bind_runtime_identity(c.RuntimeIdentitySnapshotV1(**plan["identities"]["NVIDIA A10"]))
    p.pulse_domain_json = c.canonical(plan["domain"])
    p.max_k = 3
    scheduler._pulse_queue_capacity = 2
    scheduler.has_requests = lambda: bool(scheduler.running)
    core = NS(scheduler=scheduler, batch_queue=deque(), batch_queue_size=2,
              request_block_hasher=None, post_step=lambda executed: None)
    resets, requests, schedules = [], [], []
    core.reset_prefix_cache = lambda: resets.append(True) or True
    def request_factory(rid, prompt_ids, params, *_args, **_kwargs):
        request = CpuRequest(rid, status)
        request.max_tokens = 64
        request.sampling_params = params
        request.num_prompt_tokens = len(prompt_ids)
        request.num_computed_tokens = len(prompt_ids)
        return request
    def add(request):
        add_running(scheduler, request)
        requests.append(request.request_id)
        controller = p.adapter.get_controller(request.request_id)
        controller.reset(initial_alpha=.85)
        controller.candidate_budgets = [0, 2, 3]
    core.add_request = add
    def abort(ids):
        for rid in ids:
            p.finish_request(rid)
            scheduler.requests.pop(rid, None)
        scheduler.running[:] = [r for r in scheduler.running if r.request_id not in ids]
    core.abort_requests = abort
    def step():
        if scheduler.has_requests():
            output = scheduler.schedule()
            step_plan = scheduler._plan
            schedules.append(step_plan.worker_k)
            p.record_pulse_proposal(step_plan.decision_state.scheduler_sequence, 1, step_plan.worker_k)
            core.batch_queue.append(output)
        if len(core.batch_queue) < 2 and scheduler.has_requests():
            return {}, True
        if core.batch_queue:
            output = core.batch_queue.popleft()
            old = max((len(v) for v in output.scheduled_spec_decode_tokens.values()), default=0)
            return retire(scheduler, output, list(range(10, 11 + old))), True
        return {}, False
    core.step_fn = step
    worker = LiveWorker.__new__(LiveWorker)
    worker.plan, worker.core = plan, core
    worker.attempted_slots, worker.attempted_ids = set(), set()
    worker.project = c.ROOT
    worker.source = EagleRWSourceProvenance.cpu_fixture(c.BASE_SHA)
    worker.identity = deepcopy(plan["identities"]["NVIDIA A10"])
    worker.prompt_ids = [10] * 20
    worker.sampling_factory = lambda **kwargs: NS(**kwargs, repetition_detection=None, num_logprobs=None, min_tokens=0)
    worker.request_factory = request_factory
    for slot in ("000", "001"):
        result = worker.execute("NVIDIA A10", f"old0_q{q}", slot, "cpu-" + slot)
        assert result["failure"] is None, result
        row = result["record"]
        assert row["status"] == "COMPLETE"
        assert [r["origin"]["request_k"] for r in row["retirements"]] == [q, 0]
        assert row["useful_tokens"] == 2 + q
        assert len(row["retirements"]) == 2
        assert row["preparation_evidence"]["prefix_cache_reset"] is True
        assert not scheduler.running and not core.batch_queue
        c.validate_cycle(row, plan, "NVIDIA A10", f"old0_q{q}", slot, p.runtime_identity)
    assert len(set(requests)) == len(resets) == 2
    assert schedules[:3] == [q, 0, 0]
    with pytest.raises(c.CampaignError, match="already_attempted"):
        worker.execute("NVIDIA A10", f"old0_q{q}", "000", "different-attempt")
    with pytest.raises(c.CampaignError, match="already_attempted"):
        worker.execute("NVIDIA A10", f"old0_q{q}", "002", "cpu-000")
    assert len(requests) == len(resets) == 2
