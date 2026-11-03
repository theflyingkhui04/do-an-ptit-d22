"""[Adaptive v1] Batch-K hysteresis and per-request batch-K telemetry of AdaptiveBudgetPlanner."""

import json

from src.proposer.adapter import AdaptiveProposerAdapter
from src.server.budget_planner import AdaptiveBudgetPlanner


def make_planner(**kwargs):
    """Planner plus its adapter, so tests can pin each request's next decision."""
    adapter = AdaptiveProposerAdapter(controller_kwargs={"initial_alpha": 0.95})
    return AdaptiveBudgetPlanner(adapter=adapter, **kwargs), adapter


def force_k(adapter, req_id, k):
    """Pin the next decision of one request by setting its controller alpha directly."""
    controller = adapter.get_controller(req_id)
    controller.initial_alpha = {4: 0.95, 3: 0.85, 2: 0.70, 0: 0.10}[k]
    controller.step_count = 0


def test_default_has_no_hold_and_follows_max():
    planner, adapter = make_planner()
    force_k(adapter, "a", 4)
    assert planner.plan_step(["a"], batch_size=1, kv_usage=0.1).batch_k == 4
    force_k(adapter, "a", 2)
    assert planner.plan_step(["a"], batch_size=1, kv_usage=0.1).batch_k == 2


def test_hold_ignores_short_dips():
    planner, adapter = make_planner(k_hold_steps=3)
    force_k(adapter, "a", 3)
    ks = [planner.plan_step(["a"], batch_size=1, kv_usage=0.1).batch_k]
    force_k(adapter, "a", 2)
    ks += [planner.plan_step(["a"], batch_size=1, kv_usage=0.1).batch_k for _ in range(2)]
    force_k(adapter, "a", 3)
    ks += [planner.plan_step(["a"], batch_size=1, kv_usage=0.1).batch_k]

    assert ks == [3, 3, 3, 3]


def test_hold_switches_after_enough_consecutive_steps():
    planner, adapter = make_planner(k_hold_steps=3)
    force_k(adapter, "a", 3)
    planner.plan_step(["a"], batch_size=1, kv_usage=0.1)
    force_k(adapter, "a", 2)

    ks = [planner.plan_step(["a"], batch_size=1, kv_usage=0.1).batch_k for _ in range(4)]

    assert ks == [3, 3, 2, 2]


def test_drop_to_zero_is_immediate_under_hold():
    planner, adapter = make_planner(k_hold_steps=5)
    force_k(adapter, "a", 3)
    planner.plan_step(["a"], batch_size=1, kv_usage=0.1)

    plan = planner.plan_step(["a"], batch_size=1, kv_usage=0.95)  # KV pressure -> 0

    assert plan.batch_k == 0


def test_uniform_batch_uses_held_k_for_every_request():
    planner, adapter = make_planner(k_hold_steps=2, per_request=False)
    force_k(adapter, "a", 3)
    force_k(adapter, "b", 3)
    planner.plan_step(["a", "b"], batch_size=2, kv_usage=0.1)
    force_k(adapter, "b", 2)
    force_k(adapter, "a", 2)

    plan = planner.plan_step(["a", "b"], batch_size=2, kv_usage=0.1)

    assert plan.per_request_k == {"a": 3, "b": 3}


def test_per_request_k_never_exceeds_held_batch_k():
    planner, adapter = make_planner(k_hold_steps=3)
    force_k(adapter, "a", 2)
    planner.plan_step(["a"], batch_size=1, kv_usage=0.1)
    force_k(adapter, "a", 4)

    plan = planner.plan_step(["a"], batch_size=1, kv_usage=0.1)

    assert plan.batch_k == 2
    assert plan.per_request_k["a"] == 2


def test_finish_request_writes_batch_k_counts(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    planner, adapter = make_planner(telemetry_path=str(path))
    force_k(adapter, "a", 3)
    planner.plan_step(["a"], batch_size=1, kv_usage=0.1)
    planner.plan_step(["a"], batch_size=1, kv_usage=0.1)
    force_k(adapter, "a", 2)
    planner.plan_step(["a"], batch_size=1, kv_usage=0.1)
    planner.record_verification("a", num_drafted=3, num_accepted=2)

    planner.finish_request("a")
    line = json.loads(path.read_text(encoding="utf-8").strip())

    assert line["batch_k_counts"] == {"3": 2, "2": 1}
    assert "reason_counts" in line


def test_from_env_reads_hold_steps():
    planner = AdaptiveBudgetPlanner.from_env({"ADAPTIVE_K_HOLD_STEPS": "4"})

    assert planner.k_hold_steps == 4
    assert AdaptiveBudgetPlanner.from_env({}).k_hold_steps == 0
