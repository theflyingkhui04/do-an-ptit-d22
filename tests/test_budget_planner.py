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


def decision_planner(decisions, **kwargs):
    """Use real request sessions; override only the decision boundary."""
    planner = make_planner(**kwargs)
    for request_id, decision in decisions.items():
        controller = planner.adapter.get_controller(request_id)
        controller.select_budget = lambda _decision=decision, **signals: _decision
    return planner


def tree_decision(name, n):
    from src.proposer.tree_adaptive import TreeAdaptiveDecision
    return TreeAdaptiveDecision(k=n, num_nodes=n, topology_name=name,
                                alpha_ema=0.8, expected_tokens=2.0, reason="contract")


def test_heterogeneous_planned_nodes_are_separate_from_capped_scalar_k():
    decisions = {"r1": tree_decision("small", 2), "r2": tree_decision("T8", 8),
                 "r3": tree_decision("T0", 0)}
    planner = decision_planner(decisions, max_k=4)
    plan = planner.plan_step(decisions, batch_size=3, kv_usage=0.1)
    assert plan.per_request_node_count == {"r1": 2, "r2": 8, "r3": 0}
    assert plan.batch_node_count == 8
    assert plan.per_request_k == {"r1": 2, "r2": 4, "r3": 0}
    assert plan.batch_k == 4
    assert decisions["r2"].num_nodes == 8


def test_equal_node_counts_keep_distinct_topology_identity():
    decisions = {"a": tree_decision("branching", 4), "b": tree_decision("linear", 4)}
    planner = decision_planner(decisions)
    plan = planner.plan_step(decisions, batch_size=2, kv_usage=0.1)
    assert plan.per_request_decisions["a"].topology_name == "branching"
    assert plan.per_request_decisions["b"].topology_name == "linear"
    decisions["a"].topology_name = "mutated"
    assert plan.per_request_decisions["a"].topology_name == "branching"
    assert planner.finish_request("a")["selected_topology"]["name"] == "branching"


def test_feedback_and_completion_are_request_isolated():
    planner = make_planner()
    planner.plan_step(["r1", "r2"], batch_size=2, kv_usage=0.1)
    r1 = planner.adapter.get_controller("r1")
    before = (r1.alpha_ema, r1.get_window_samples(), r1.get_stats())
    planner.record_verification("r2", 3, 0)
    assert (r1.alpha_ema, r1.get_window_samples(), r1.get_stats()) == before
    planner.finish_request("r1")
    assert planner.adapter.active_request_count == 1
    assert planner.adapter.get_telemetry("r2")["mean_accept_len"] == 0.0
    assert planner.finish_request("r1") is None
    planner.finish_request("r2")
    assert planner._last_decisions == {}


def test_step_telemetry_is_separate_from_final_aggregate(tmp_path):
    steps, final = tmp_path / "steps.jsonl", tmp_path / "final.jsonl"
    planner = decision_planner({"r": tree_decision("T8", 8)}, max_k=4,
                               step_telemetry_path=str(steps), telemetry_path=str(final))
    planner.plan_step(["r"], batch_size=1, kv_usage=0.1)
    planner.record_verification("r", 3, 2)
    aggregate = planner.finish_request("r")
    records = [json.loads(line) for line in steps.read_text().splitlines()]
    assert [r["record_type"] for r in records] == ["planned_decision", "verification_feedback"]
    planned, feedback = records
    assert planned["selected_topology"] == {"name": "T8", "total_nodes_N": 8,
        "depth_d": None, "parent_indices": None}
    assert planned["scalar_runtime_budget"] == 4
    assert planned["executed_draft_token_count"] is None
    assert planned["tree_execution_verified"] is False
    assert planned["actual_step_latency_ms"] is None
    assert feedback["executed_draft_token_count"] == 3
    assert feedback["accepted_draft_token_count"] == 2
    assert "selected_topology" not in feedback  # async feedback is not joined to latest plan
    assert aggregate["record_type"] == "request_aggregate"
    assert aggregate["executed_runtime_mode"] == "unverified"
    assert aggregate["tree_budgets"] == []  # legacy controller history is preserved
    assert len(final.read_text().splitlines()) == 1


def test_all_zero_and_empty_plans_have_neutral_node_count():
    planner = decision_planner({"r": tree_decision("T0", 0)})
    assert planner.plan_step(["r"], 1, 0.1).batch_k == 0
    empty = planner.plan_step([], 0, 0.1)
    assert empty.batch_node_count == empty.batch_k == 0


@pytest.mark.parametrize("kwargs", [{"max_k": -1}, {"min_batch_k": -1}])
def test_negative_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        make_planner(**kwargs)


def test_normalization_preserves_legacy_aggregate_after_feedback():
    planner = make_planner()
    planner.plan_step(["r"], 1, 0.1)
    planner.record_verification("r", 3, 1)
    legacy = planner.adapter.get_telemetry("r")
    final = planner.finish_request("r")
    assert all(final[key] == value for key, value in legacy.items())
    assert final["last_planned_decision"]["acceptance_rate_ema"] == 0.85


def test_required_v02_fields_and_catalog_are_detached(tmp_path):
    from src.proposer.tree_adaptive import get_tree_topology
    steps = tmp_path / "steps.jsonl"
    planner = decision_planner({"r": tree_decision("T8", 8)}, step_telemetry_path=str(steps))
    planner.plan_step(["r"], 1, 0.1, free_vram_mb=None)
    record = json.loads(steps.read_text())
    required = {"request_id", "engine_version", "hardware_profile", "concurrency_batch_b",
                "acceptance_rate_ema", "effective_alpha", "dynamic_alpha_star", "selected_topology",
                "mean_accepted_tokens_tau", "theoretical_speedup", "actual_step_latency_ms",
                "memory_status", "decision_reason"}
    assert required <= record.keys()
    assert record["memory_status"]["free_vram_mb"] is None
    final = planner.finish_request("r")
    assert final["selected_topology"]["parent_indices"] is None
    final["selected_topology"]["name"] = "changed"
    assert record["selected_topology"]["name"] == "T8"
    assert get_tree_topology("T8").parent_indices[1] == 0


def test_canonical_serving_does_not_import_future_tree():
    import subprocess
    import sys
    from pathlib import Path
    code = '''
import sys
class BlockTree:
    def find_spec(self, fullname, path=None, target=None):
        if "tree_adaptive" in fullname or "tree_topology" in fullname:
            raise AssertionError("canonical serving imported " + fullname)
sys.meta_path.insert(0, BlockTree())
from src.server.budget_planner import AdaptiveBudgetPlanner
p = AdaptiveBudgetPlanner()
for rid, alpha in [("two", .7), ("four", .95), ("zero", .4)]:
    p.adapter.start_request(rid, initial_alpha=alpha)
plan = p.plan_step(["two", "four", "zero"], 3, .1)
assert plan.per_request_k == {"two": 2, "four": 4, "zero": 0}
assert plan.batch_k == 4
assert all(r["selected_topology"] is None for r in p._last_decisions.values())
assert all(r["tree_execution_verified"] is False for r in p._last_decisions.values())
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_heterogeneous_two_four_zero_contract():
    decisions = {"r1": tree_decision("two", 2), "r2": tree_decision("T4", 4),
                 "r3": tree_decision("T0", 0)}
    plan = decision_planner(decisions).plan_step(decisions, 3, 0.1)
    assert plan.batch_node_count == plan.batch_k == 4
    assert plan.per_request_k == {"r1": 2, "r2": 4, "r3": 0}


def test_shared_scalar_mode_keeps_individual_planned_nodes():
    decisions = {"r1": tree_decision("two", 2), "r2": tree_decision("T4", 4)}
    plan = decision_planner(decisions, per_request=False).plan_step(decisions, 2, 0.1)
    assert plan.per_request_k == {"r1": 4, "r2": 4}
    assert plan.per_request_node_count == {"r1": 2, "r2": 4}


def test_step_and_aggregate_path_must_be_distinct(tmp_path):
    path = str(tmp_path / "same.jsonl")
    with pytest.raises(ValueError):
        make_planner(step_telemetry_path=path, telemetry_path=path)


def test_step_path_from_env(tmp_path):
    path = str(tmp_path / "steps.jsonl")
    assert AdaptiveBudgetPlanner.from_env({"ADAPTIVE_STEP_TELEMETRY_PATH": path}).step_telemetry_path == path


def test_step_io_failure_does_not_change_budget_or_feedback(tmp_path, caplog):
    planner = make_planner(step_telemetry_path=str(tmp_path / "missing" / "steps.jsonl"))
    assert planner.plan_step(["r"], 1, 0.1).batch_k == 3
    planner.record_verification("r", 3, 2)
    assert planner.finish_request("r")["mean_accept_len"] == 2.0
    assert "step telemetry" in caplog.text


def test_planned_ema_and_effective_alpha_have_distinct_scalar_sources(tmp_path):
    path = tmp_path / "steps.jsonl"
    planner = make_planner(step_telemetry_path=str(path))
    planner.plan_step(["r"], 1, 0.1)
    planner.record_verification("r", 3, 0)
    plan = planner.plan_step(["r"], 1, 0.1)
    planned = json.loads(path.read_text().splitlines()[-1])
    # Scalar BudgetDecision.alpha_ema historically carries the bias-corrected
    # decision signal, NOT the raw EMA stored on the controller.
    assert plan.per_request_decisions["r"].alpha_ema == pytest.approx(0.0)
    assert planned["acceptance_rate_ema"] == pytest.approx(0.68)
    assert planned["effective_alpha"] == pytest.approx(0.0)


def test_real_scalar_controller_is_used_after_metadata_transport():
    from src.proposer.controller import AdaptiveDraftController
    planner = AdaptiveBudgetPlanner.from_env({})
    planner.plan_step(["r"], 1, 0.1)
    controller = planner.adapter.get_controller("r")
    assert type(controller) is AdaptiveDraftController
    planner.record_verification("r", 8, 4)
    assert controller.alpha_ema == pytest.approx(0.74)
