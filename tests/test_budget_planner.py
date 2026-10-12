"""Unit tests for AdaptiveBudgetPlanner (vLLM-free logic behind Lane 3 scheduler)."""
import json

import pytest

from src.proposer.adapter import AdaptiveProposerAdapter
from src.server.budget_planner import AdaptiveBudgetPlanner, StepPlan


def make_planner(initial_alpha=0.85, **kwargs) -> AdaptiveBudgetPlanner:
    adapter = AdaptiveProposerAdapter(controller_kwargs={"initial_alpha": initial_alpha})
    return AdaptiveBudgetPlanner(adapter=adapter, **kwargs)


def test_plan_step_high_alpha_drafts_k3_for_each_request():
    planner = make_planner()

    plan = planner.plan_step(["r1", "r2"], batch_size=2, kv_usage=0.10)

    assert isinstance(plan, StepPlan)
    assert plan.per_request_k == {"r1": 3, "r2": 3}
    assert plan.batch_k == 3


def test_plan_step_calls_select_budget_once_per_request():
    planner = make_planner()

    planner.plan_step(["r1"], batch_size=1, kv_usage=0.10)
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.10)
    telemetry = planner.finish_request("r1")

    assert telemetry["tree_budgets"] == [3, 3]


def test_plan_step_converts_kv_fraction_to_percent():
    planner = make_planner()

    calm = planner.plan_step(["r1"], batch_size=1, kv_usage=0.50)
    pressured = planner.plan_step(["r2"], batch_size=1, kv_usage=0.95)

    assert calm.per_request_k["r1"] == 3
    assert pressured.per_request_k["r2"] == 0
    assert pressured.batch_k == 0


def test_plan_step_without_decode_requests():
    planner = make_planner()

    plan = planner.plan_step([], batch_size=0, kv_usage=0.10)

    assert plan.batch_k == 0
    assert plan.per_request_k == {}


def test_batch_k_is_max_of_per_request_k():
    planner = make_planner()
    planner.plan_step(["good", "bad"], batch_size=2, kv_usage=0.10)
    planner.record_verification("bad", num_drafted=3, num_accepted=0)

    plan = planner.plan_step(["good", "bad"], batch_size=2, kv_usage=0.10)

    assert plan.per_request_k == {"good": 3, "bad": 0}
    assert plan.batch_k == 3


def test_per_request_k_clamped_to_max_k():
    planner = make_planner(initial_alpha=0.95, max_k=3)

    plan = planner.plan_step(["r1"], batch_size=1, kv_usage=0.10)

    assert plan.per_request_k["r1"] == 3
    assert plan.batch_k == 3


def test_record_verification_ignores_steps_without_drafts():
    planner = make_planner()
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.10)

    planner.record_verification("r1", num_drafted=0, num_accepted=0)
    telemetry = planner.finish_request("r1")

    assert telemetry["step_count"] == 0


def test_record_verification_registers_unplanned_request():
    # A request that just finished prefill can be verified before it is ever planned.
    planner = make_planner()

    planner.record_verification("fresh", num_drafted=3, num_accepted=3)
    telemetry = planner.finish_request("fresh")

    assert telemetry is not None
    assert telemetry["step_count"] == 1


def test_finish_request_writes_one_jsonl_line(tmp_path):
    path = tmp_path / "lane3_telemetry.jsonl"
    planner = make_planner(telemetry_path=str(path))
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.10)
    planner.record_verification("r1", num_drafted=3, num_accepted=2)

    planner.finish_request("r1")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request_id"] == "r1"
    assert record["lane"] == "adaptive"
    assert record["tree_budgets"] == [3]
    assert record["mean_accept_len"] == 2.0
    assert "mean_k" in record


def test_finish_unknown_request_is_noop(tmp_path):
    path = tmp_path / "lane3_telemetry.jsonl"
    planner = make_planner(telemetry_path=str(path))

    assert planner.finish_request("never-planned") is None
    assert not path.exists()


def test_finish_request_releases_session():
    planner = make_planner()
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.10)

    planner.finish_request("r1")

    assert planner.finish_request("r1") is None


def test_per_request_disabled_uses_batch_k_for_all():
    planner = make_planner(per_request=False)
    planner.plan_step(["good", "bad"], batch_size=2, kv_usage=0.10)
    planner.record_verification("bad", num_drafted=3, num_accepted=0)

    plan = planner.plan_step(["good", "bad"], batch_size=2, kv_usage=0.10)

    assert plan.per_request_k == {"good": 3, "bad": 3}
    assert plan.batch_k == 3


def test_min_batch_k_floor_keeps_per_request_zero():
    planner = make_planner(min_batch_k=2)

    plan = planner.plan_step(["r1"], batch_size=1, kv_usage=0.95)

    assert plan.per_request_k["r1"] == 0
    assert plan.batch_k == 2


def test_free_vram_none_means_no_vram_pressure():
    planner = make_planner()

    plan = planner.plan_step(["r1"], batch_size=1, kv_usage=0.10, free_vram_mb=None)

    assert plan.per_request_k["r1"] == 3


def test_low_free_vram_triggers_when_guard_enabled():
    planner = make_planner()

    plan = planner.plan_step(["r1"], batch_size=1, kv_usage=0.10, free_vram_mb=500.0)

    assert plan.per_request_k["r1"] == 0


def test_from_env_reads_configuration(tmp_path):
    path = tmp_path / "t.jsonl"
    env = {
        "ADAPTIVE_CONTROLLER_KWARGS": json.dumps({"initial_alpha": 0.95}),
        "ADAPTIVE_TELEMETRY_PATH": str(path),
        "ADAPTIVE_ENABLE_VRAM_GUARD": "0",
        "ADAPTIVE_MAX_K": "3",
        "ADAPTIVE_MIN_BATCH_K": "1",
        "ADAPTIVE_PER_REQUEST": "1",
    }

    planner = AdaptiveBudgetPlanner.from_env(env)
    plan = planner.plan_step(["r1"], batch_size=1, kv_usage=0.10, free_vram_mb=500.0)

    # VRAM guard disabled -> low free VRAM ignored; max_k=3 clamps the k=4 choice
    assert plan.per_request_k["r1"] == 3
    assert planner.max_k == 3
    assert planner.min_batch_k == 1
    assert planner.telemetry_path == str(path)


def test_from_env_defaults():
    planner = AdaptiveBudgetPlanner.from_env({})

    assert planner.max_k == 4
    assert planner.min_batch_k == 0
    assert planner.per_request is True
    assert planner.telemetry_path is None


def test_from_env_rejects_invalid_controller_json():
    with pytest.raises(ValueError):
        AdaptiveBudgetPlanner.from_env({"ADAPTIVE_CONTROLLER_KWARGS": "{not json"})
