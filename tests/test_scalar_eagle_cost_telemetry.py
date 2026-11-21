from dataclasses import asdict
import json
import math
from scalar_eagle_fixtures import planner, plan, seed


def test_decision_roundtrip_reconstructs_selected_cap_and_keeps_latency_unknown(tmp_path):
    path=tmp_path/"decisions.jsonl"
    p,s=planner(telemetry=path)
    result=plan(p,s)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    record=next(r for r in rows if r["record_type"]=="hardware_cost_decision")
    assert record["profile_hash"] == p.cost_governor.profile.sha256
    assert record["profile_identity_status"] == "VALID_MATCH"
    assert record["policy_version"] == "SECP-v1.1-candidate"
    assert record["base_k"] == {"A":4,"B":2}
    assert record["survival"]["A"]["survival"] == [.9,.75,.55,.35]
    assert record["survival"]["B"]["survival"] == [.8,.5]
    assert record["survival"]["A"]["window_size"] == record["survival"]["A"]["count"] == 20
    best=None
    for score in record["candidates"]:
        assert score["R_lower"] == score["predicted_yield_batch"]/score["C_high"]
        if score["cap"] and score["R_lower"] > record["K0_R_upper"]:
            if best is None or (score["R_lower"]>best["R_lower"] and not math.isclose(
                    score["R_lower"],best["R_lower"],rel_tol=1e-9,abs_tol=1e-12)):
                best=score
    assert best["cap"] == record["selected_cap"] == result.batch_k == 2
    assert record["final_applied_k"] == result.per_request_k
    assert record["planned_worker_k"] == 2 and record["physical_draft_width"] is None
    planned=[r for r in rows if r["record_type"]=="planned_decision"]
    assert all(r["actual_step_latency_ms"] is None for r in planned)
    assert all(r["executed_draft_token_count"] is None for r in planned)
    finalized=[r for r in rows if r["record_type"]=="cost_verification_finalized"]
    assert len(finalized)==40 and all(r["normal_survival_counted"] for r in finalized)
    assert all(r["actual_emitted_tokens"] == r["A"]+1 for r in finalized)
    assert all(r["key"]["incarnation"] == 1 and r["key"]["verifier_sequence"]>0 for r in finalized)


def test_probe_and_compat_telemetry_never_hardware_aware(tmp_path):
    for mode in ("compat","strict"):
        p,s=planner(mode=mode,ready=False,base={"A":2})
        if mode=="compat": p.cost_governor.profile=None
        else:
            c=p.adapter.get_controller("A");c.reset(initial_alpha=.4);c.probe_interval=1
        record=plan(p,s).cost_decision.to_record()
        assert not record["hardware_cost_claim_allowed"] and not record["cost_optimal"]
        assert not record["cost_policy_applied"]


def test_later_physical_width_variation_does_not_rewrite_decision_key(tmp_path):
    from src.server.lineage import DraftOrigin
    from flexsds.core.eagle_consumer_profile import decision_state_key
    path=tmp_path/"steps.jsonl"
    p,s=planner(telemetry=path)
    result=plan(p,s)
    before=[decision_state_key(state) for q,state in result.candidate_decision_states]
    origin=DraftOrigin("A",1,1,1,2,2,source_base_k=4,source_decision_mode="normal")
    p.record_draft_birth(origin,2)
    p.record_draft_birth(origin,4)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    widths=[r["physical_draft_width"] for r in rows if r["record_type"]=="draft_policy_produced"]
    assert widths==[2,4]
    assert before==[decision_state_key(state) for q,state in result.candidate_decision_states]
