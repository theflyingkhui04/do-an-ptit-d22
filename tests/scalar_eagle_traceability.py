"""Explicit normative requirement to executable CPU/source verification map."""
from pathlib import Path
import ast
import re

ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/"audit/flexsds_v0.2.3_hardware_adaptive/design/scalar-eagle-cost-policy-spec-amendment-candidate.md"
GROUPS=[
    ([1,2],"test_scalar_eagle_k0_source","test_k0_pinned_eagle_forward_precedes_zero_width_return"),
    ([3,53],"test_scalar_eagle_cost_modes","test_memory_safety_dominates_profile_floor_and_mode"),
    ([4,69,70,71,72],"test_budget_planner_scalar_eagle_cost","test_f_causal_01_only_consumer_cost_changes_active_planner"),
    ([5,37,40,41],"test_budget_planner_scalar_eagle_cost","test_shared_batch_rate_not_sum_of_per_request_rates_and_tie_smaller_cap"),
    ([6],"test_budget_planner_scalar_eagle_cost","test_legacy_parent_selector_shortcut_is_rejected"),
    ([7,8,59,60],"test_scalar_eagle_cost_modes","test_cost_cap_beats_floor_and_immediate_downward_recovery"),
    ([9,10,11,13,21],"test_scalar_eagle_survival","test_adaptive_d_counterexample_common_denominator"),
    ([12,15,18,19,20,33],"test_scalar_eagle_survival","test_zero_accept_rolling_full_window_and_incarnation"),
    ([14,16,17],"test_scalar_eagle_survival","test_only_fully_exposed_normal_finalized_same_epoch"),
    ([22,73,74],"test_scalar_eagle_cost_telemetry","test_decision_roundtrip_reconstructs_selected_cap_and_keeps_latency_unknown"),
    ([23,24,25,32,78],"test_scalar_eagle_pending_feedback","test_production_prestop_pending_then_native_nonterminal_commits"),
    ([26],"test_scalar_eagle_pending_feedback","test_production_terminal_native_clipping_censored_even_with_future_schedule"),
    ([27,29],"test_scalar_eagle_pending_feedback","test_pending_only_until_exact_post_stop_row_and_ambiguous_fails_closed"),
    ([28],"test_scalar_eagle_pending_feedback","test_two_pending_origins_no_overwrite_idempotence_future_schedule_not_proof"),
    ([30,77],"test_scalar_eagle_pending_feedback","test_production_stale_origin_and_old_cleanup_cannot_affect_reused_id"),
    ([31],"test_scalar_eagle_pending_feedback","test_binary_nonterminal_finalization_can_be_online_without_emitted_validation"),
    ([34,38],"test_eagle_consumer_cost_statistics","test_fixed_cohort_bootstrap_reproducible_not_inner_iid"),
    ([35],"test_eagle_consumer_cost_statistics","test_invalid_sample_plans_rejected"),
    ([36],"test_eagle_consumer_cost_statistics","test_failed_cohorts_preserved_not_dropped"),
    ([39],"test_scalar_eagle_cost_modes","test_inconclusive_and_unprofitable_cost_force_zero"),
    ([42,44,47],"test_eagle_decision_state_key","test_context_and_runtime_identity_explicit_no_physical_width"),
    ([43,49],"test_eagle_decision_state_key","test_no_nearest_state_and_no_worker_backfill"),
    ([45,46],"test_eagle_decision_state_key","test_exact_aligned_vectors_not_batch_or_max_only"),
    ([48],"test_scalar_eagle_cost_telemetry","test_later_physical_width_variation_does_not_rewrite_decision_key"),
    ([50],"test_decision_state_transport","test_real_scheduler_boundary_transports_full_immutable_snapshot"),
    ([51,52],"test_scalar_eagle_cost_modes","test_default_off_independent_of_adaptive_strict"),
    ([54,58,75],"test_scalar_eagle_cost_modes","test_missing_profile_modes_and_claims"),
    ([55],"test_scalar_eagle_cost_modes","test_strict_warmup_uses_base_then_activates_not_k0_forever"),
    ([56,76],"test_scalar_eagle_cost_modes","test_valid_probe_is_exploration_and_never_normal_history"),
    ([57,68],"test_scalar_eagle_cost_modes","test_degraded_admission_never_claims_cost"),
    ([61,62],"test_eagle_consumer_profile_identity","test_name_sm_driver_vram_software_exact_identity"),
    ([63],"test_eagle_consumer_profile_identity","test_unknown_observed_identity_never_backfilled_by_configured_or_filename"),
    ([64,66,67],"test_eagle_consumer_profile_publication","test_primary_profile_contract_rejects_component_conversion"),
    ([65],"test_eagle_consumer_cost_statistics","test_retirement_window_contract"),
]

NEGATIVE_CONTROLS=[
    ("profile load only","test_budget_planner_scalar_eagle_cost","test_causal_oracle_rejects_disconnected_cost_paths"),
    ("cost lookup discarded","test_budget_planner_scalar_eagle_cost","test_causal_oracle_rejects_disconnected_cost_paths"),
    ("legacy parent selector shortcut","test_budget_planner_scalar_eagle_cost","test_legacy_parent_selector_shortcut_is_rejected"),
    ("A/D geometric alpha","test_scalar_eagle_survival","test_estimator_not_legacy_ema_or_geometric"),
    ("naive variable D pooling","test_scalar_eagle_survival","test_adaptive_d_counterexample_common_denominator"),
    ("batch size only key","test_eagle_decision_state_key","test_exact_aligned_vectors_not_batch_or_max_only"),
    ("max K only key","test_eagle_decision_state_key","test_exact_aligned_vectors_not_batch_or_max_only"),
    ("nearest state lookup","test_eagle_decision_state_key","test_no_nearest_state_and_no_worker_backfill"),
    ("target only K0","test_eagle_consumer_profile_publication","test_primary_profile_contract_rejects_component_conversion"),
    ("request ID only pending key","test_scalar_eagle_pending_feedback","test_two_pending_origins_no_overwrite_idempotence_future_schedule_not_proof"),
    ("future schedule terminal proof","test_scalar_eagle_pending_feedback","test_pending_only_until_exact_post_stop_row_and_ambiguous_fails_closed"),
    ("per retirement IID bootstrap","test_eagle_consumer_cost_statistics","test_fixed_cohort_bootstrap_reproducible_not_inner_iid"),
    ("strict K0 forever during warmup","test_scalar_eagle_cost_modes","test_strict_warmup_uses_base_then_activates_not_k0_forever"),
    ("floor raises cost cap","test_scalar_eagle_cost_modes","test_cost_cap_beats_floor_and_immediate_downward_recovery"),
    ("probe cost optimal claim","test_scalar_eagle_cost_modes","test_valid_probe_is_exploration_and_never_normal_history"),
    ("compat hardware aware claim","test_scalar_eagle_cost_modes","test_missing_profile_modes_and_claims"),
    ("profile cost as actual latency","test_scalar_eagle_cost_telemetry","test_decision_roundtrip_reconstructs_selected_cap_and_keeps_latency_unknown"),
]


def entries():
    titles=dict(re.findall(r"^### SECP-R(\d{3}) — (.+)$",SPEC.read_text(encoding="utf-8"),re.M))
    rows=[]
    for ids,module,test in GROUPS:
        for number in ids:
            key=f"{number:03}"
            rows.append(dict(requirement="SECP-R"+key,title=titles[key],
                test=f"tests/{module}.py::{test}",verification="CPU/source"))
    return titles,rows


def validate():
    titles,rows=entries()
    ids=[r["requirement"] for r in rows]
    assert len(titles)==len(rows)==len(set(ids))==78
    assert {"SECP-R"+k for k in titles}==set(ids)
    for row in rows:
        path,test=row["test"].split("::")
        names={n.name for n in ast.parse((ROOT/path).read_text(encoding="utf-8-sig")).body
               if isinstance(n,ast.FunctionDef)}
        assert test in names,row
    return rows
