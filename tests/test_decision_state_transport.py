"""CPU witnesses for the approved host transport, not cost-policy closure."""
from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import pytest

from src.server.budget_planner import AdaptiveBudgetPlanner
from src.server.decision_state import (
    CONFIGURED_FIELDS, CONTEXT_SEMANTICS, OBSERVED_FIELDS,
    DecisionStateSnapshotV1, RuntimeIdentitySnapshotV1, capture_decision_state,
    context_bucket, observe_runtime_identity,
)
from test_adaptive_scheduler_glue import add_running, build, make_request, scheduler_module


def synthetic_identity():
    """Explicitly synthetic complete transport witness, not measured hardware."""
    observed = dict.fromkeys(OBSERVED_FIELDS, "synthetic-observation")
    observed.update(gpu_name="Synthetic A10", sm_architecture="sm_86", vram_total_mb=24000)
    configured = dict.fromkeys(CONFIGURED_FIELDS, "synthetic-config")
    configured.update(tensor_parallel_size=1, async_scheduling=True, enforce_eager=True,
                      compilation_mode=0, target_revision="a" * 40, draft_revision="b" * 40)
    return RuntimeIdentitySnapshotV1(tuple(observed.items()), tuple(configured.items()))


def state(*, sequence=1, epochs=(("r", 1),), ids=("r",), eligible=("r",), old=(("r", 4),),
          contexts=(("r", 127),), prefill=False, identity=None):
    identity = identity or synthetic_identity()
    return DecisionStateSnapshotV1(sequence, epochs, ids, eligible, not prefill, prefill,
                                   old, contexts, identity.runtime_mode_id)


def planned(snapshot, *, identity=None, alpha=.95):
    p = AdaptiveBudgetPlanner(replacement_policy=True)
    p.bind_runtime_identity(identity or synthetic_identity())
    for rid, epoch in snapshot.request_incarnations:
        if rid in snapshot.eligible_decode_request_ids:
            p.bind_request(rid, epoch)
            p.adapter.start_request(rid, initial_alpha=alpha)
    plan = p.plan_step(snapshot.eligible_decode_request_ids, len(snapshot.eligible_decode_request_ids),
                       .1, 16000., decision_state=snapshot)
    return p, plan


def test_c_w1_old_width_distinguishes_previous_identical_inputs():
    a, b = state(), state(old=(("r", 0),))
    _, pa = planned(a)
    _, pb = planned(b)
    assert (a.eligible_decode_request_ids, len(a.eligible_decode_request_ids)) == (
        b.eligible_decode_request_ids, len(b.eligible_decode_request_ids))
    assert pa.per_request_k == pb.per_request_k == {"r": 4}
    assert pa.decision_state != pb.decision_state
    assert dict(pa.candidate_decision_states)[4] != dict(pb.candidate_decision_states)[4]


def test_c_w2_post_schedule_context_bucket_distinguishes():
    _, a = planned(state())
    _, b = planned(state(contexts=(("r", 1024),)))
    assert dict(a.candidate_decision_states)[4].context_bucket_by_request == (("r", "[0,256)"),)
    assert dict(b.candidate_decision_states)[4].context_bucket_by_request == (("r", "[1024,2048)"),)
    assert a.decision_state.context_snapshot_semantics == CONTEXT_SEMANTICS


def test_c_w3_full_scheduled_cohort_not_eligible_only():
    a = state()
    b = state(ids=("r", "p"), epochs=(("r", 1), ("p", 2)),
              old=(("r", 4), ("p", 0)), contexts=(("r", 127), ("p", 100)), prefill=True)
    _, pa = planned(a)
    _, pb = planned(b)
    assert a.eligible_decode_request_ids == b.eligible_decode_request_ids
    assert pa.decision_state.pure_decode and not pa.decision_state.has_scheduled_prefill
    assert not pb.decision_state.pure_decode and pb.decision_state.has_scheduled_prefill
    assert "mixed_prefill_decode" in dict(pb.candidate_decision_states)[4].unsupported_reasons


def test_c_w4_heterogeneous_vectors_retain_row_association():
    candidates = []
    for old in ((4, 4), (4, 0), (4, 2)):
        _, plan = planned(state(ids=("a", "b"), eligible=("a", "b"),
            epochs=(("a", 1), ("b", 2)), old=tuple(zip(("a", "b"), old)),
            contexts=(("a", 127), ("b", 127))))
        candidates.append(dict(plan.candidate_decision_states)[4])
    assert len(set(candidates)) == 3
    assert all(c.eligible_decode_rows == 2 and c.candidate_shared_worker_k == 4 for c in candidates)


def test_c_w5_candidates_are_planner_derived_from_base():
    snapshot = state(ids=("a", "b"), eligible=("a", "b"), epochs=(("a", 1), ("b", 2)),
                     old=(("a", 4), ("b", 2)), contexts=(("a", 127), ("b", 127)))
    p = AdaptiveBudgetPlanner(replacement_policy=True)
    p.bind_runtime_identity(synthetic_identity())
    for rid, epoch, alpha in (("a", 1, .95), ("b", 2, .7)):
        p.bind_request(rid, epoch)
        p.adapter.start_request(rid, initial_alpha=alpha)
    plan = p.plan_step(["a", "b"], 2, .1, decision_state=snapshot)
    assert plan.per_request_selected_k == {"a": 4, "b": 2}
    assert {q: tuple(k for _, k in c.candidate_future_width_by_request)
            for q, c in plan.candidate_decision_states} == {0: (0, 0), 2: (2, 2), 3: (3, 2), 4: (4, 2)}
    assert not hasattr(snapshot, "candidate_future_width_by_request")


def test_c_w6_old_epoch_snapshot_rejected_for_reused_id():
    p, _ = planned(state())
    p.bind_request("r", 2)
    rejected = p.plan_step(["r"], 1, .1, decision_state=replace(state(), scheduler_sequence=2))
    assert rejected.decision_state is None
    assert "request_incarnation_mismatch" in rejected.decision_state_reasons
    fresh = state(sequence=2, epochs=(("r", 2),))
    accepted = p.plan_step(["r"], 1, .1, decision_state=fresh)
    assert accepted.decision_state == fresh


def test_candidate_base_is_controller_width_not_later_replacement_clip():
    p = AdaptiveBudgetPlanner(replacement_policy=True, max_k=2)
    p.bind_runtime_identity(synthetic_identity())
    p.bind_request("r", 1)
    p.adapter.start_request("r", initial_alpha=.95)
    plan = p.plan_step(["r"], 1, .1, decision_state=state())
    assert plan.per_request_decisions["r"].k == 4
    assert dict(plan.candidate_decision_states)[4].candidate_future_width_by_request == (("r", 4),)
    assert plan.per_request_k == {"r": 2} and plan.worker_k == 2


def test_sequence_replay_rejected_without_overwriting_current_binding():
    p, _ = planned(state(sequence=3))
    replay = p.plan_step(["r"], 1, .1, decision_state=state(sequence=3))
    assert replay.decision_state is None and replay.candidate_decision_states == ()
    assert "stale_scheduler_sequence" in replay.decision_state_reasons
    assert p._last_decision_state_sequence == 3


@pytest.mark.parametrize("change,reason", [
    ({"schema_version": "unknown"}, "unsupported_snapshot_schema_or_semantics"),
    ({"context_snapshot_semantics": "worker_context"}, "unsupported_snapshot_schema_or_semantics"),
    ({"runtime_mode_id": "other-runtime"}, "runtime_mode_mismatch"),
    ({"scheduler_sequence": 0}, "stale_scheduler_sequence"),
    ({"old_verification_width_by_request": (("r", -1),)}, "invalid_old_verification_width"),
])
def test_invalid_transport_never_binds_or_becomes_a_candidate(change, reason):
    p = AdaptiveBudgetPlanner(replacement_policy=True)
    p.bind_runtime_identity(synthetic_identity())
    p.bind_request("r", 1)
    plan = p.plan_step(["r"], 1, .1, decision_state=replace(state(), **change))
    assert plan.decision_state is None and plan.candidate_decision_states == ()
    assert reason in plan.decision_state_reasons


@pytest.mark.parametrize("tokens,bucket", [(0, "[0,256)"), (255, "[0,256)"),
    (256, "[256,1024)"), (1023, "[256,1024)"), (1024, "[1024,2048)"),
    (2048, "[2048,4096)"), (4095, "[2048,4096)"), (4096, None), (-1, None), (None, None)])
def test_exact_context_boundaries_without_nearest_fallback(tokens, bucket):
    assert context_bucket(tokens) == bucket


def test_missing_context_unknown_identity_and_mixed_state_cannot_be_supported():
    unknown = RuntimeIdentitySnapshotV1()
    r = make_request("r")
    snapshot = capture_decision_state(1, [r], ["r"], {"r": 0}, {"r": 1}, None)
    _, plan = planned(snapshot, identity=unknown)
    assert snapshot.status == plan.decision_state_status == "UNSUPPORTED"
    assert all(c.status == "UNSUPPORTED" for _, c in plan.candidate_decision_states)
    assert plan.per_request_k == {"r": 4}  # Metadata transport does not change the legacy policy.
    assert unknown.identity_key is None


def test_snapshot_is_detached_immutable_host_metadata():
    r = make_request("r")
    r.num_computed_tokens = 127
    snapshot = capture_decision_state(1, [r], ["r"], {"r": 4}, {"r": 1}, "test-mode")
    r.num_computed_tokens = 999
    assert snapshot.host_context_tokens_after_schedule_by_request == (("r", 127),)
    with pytest.raises(FrozenInstanceError):
        snapshot.scheduler_sequence = 2
    with pytest.raises(TypeError):
        snapshot.old_verification_width_by_request[0] = ("r", 0)


def test_runtime_observation_does_not_initialize_cuda_or_use_cloud_sku():
    class Cuda:
        def is_initialized(self):
            return False
        def __getattr__(self, name):
            raise AssertionError("No CUDA initialization or device query when uninitialized: " + name)
    torch = SimpleNamespace(__version__="cpu-test", version=SimpleNamespace(cuda=None), cuda=Cuda())
    identity = observe_runtime_identity(SimpleNamespace(target_gpu="A10G"), torch_module=torch,
                                       vllm_module=SimpleNamespace(__version__="0.30.0"))
    assert dict(identity.observed)["gpu_name"] is None
    assert identity.status == "UNSUPPORTED" and identity.identity_key is None


def test_observed_cuda_metadata_is_distinct_from_configured_quantization():
    cuda = SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 0,
        get_device_capability=lambda _: (8, 6), get_device_name=lambda _: "NVIDIA A10",
        get_device_properties=lambda _: SimpleNamespace(total_memory=24 * 2**30))
    torch = SimpleNamespace(__version__="synthetic-torch", version=SimpleNamespace(cuda="synthetic-cuda"), cuda=cuda)
    model = SimpleNamespace(model="target", revision="main", hf_config=SimpleNamespace(_commit_hash="a" * 40),
                            dtype="bfloat16", quantization="awq_marlin")
    identity = observe_runtime_identity(SimpleNamespace(model_config=model, target_gpu="L4"),
        runner=SimpleNamespace(model_config=model), torch_module=torch,
        vllm_module=SimpleNamespace(__version__="0.30.0"))
    assert dict(identity.observed)["gpu_name"] == "NVIDIA A10"
    assert dict(identity.observed)["vram_total_mb"] == 24576
    assert dict(identity.observed)["effective_quantization"] is None
    assert dict(identity.observed)["driver_version"] is None
    assert dict(identity.configured)["target_revision"] == "a" * 40
    assert dict(identity.configured)["configured_quantization"] == "awq_marlin"


def test_runtime_identity_is_bound_once():
    p = AdaptiveBudgetPlanner()
    p.bind_runtime_identity(synthetic_identity())
    p.bind_runtime_identity(synthetic_identity())
    with pytest.raises(ValueError, match="already bound"):
        p.bind_runtime_identity(RuntimeIdentitySnapshotV1())
    with pytest.raises(AttributeError):
        p.runtime_identity = RuntimeIdentitySnapshotV1()


def test_version_string_subclasses_remain_immutable_json_scalars():
    class RuntimeVersion(str):
        pass
    identity = RuntimeIdentitySnapshotV1(observed=(("torch_version", RuntimeVersion("2.test")),))
    assert type(dict(identity.observed)["torch_version"]) is str


def test_real_scheduler_boundary_transports_full_immutable_snapshot(scheduler_module, monkeypatch, tmp_path):
    s = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_REPLACEMENT_POLICY="1")
    a, b = make_request("a"), make_request("b", prefill=True)
    a.num_computed_tokens, b.num_computed_tokens = 127, 1024
    add_running(s, a, b)
    first = s.schedule()
    snapshot = s._plan.decision_state
    assert snapshot.full_scheduled_request_ids == ("a", "b")
    assert snapshot.eligible_decode_request_ids == ("a",)
    assert snapshot.has_scheduled_prefill and not snapshot.pure_decode
    assert snapshot.scheduler_sequence == 1
    assert dict(snapshot.old_verification_width_by_request) == {"a": 0, "b": 0}
    a.num_computed_tokens = 1024
    second = s.schedule()
    assert dict(s._plan.decision_state.old_verification_width_by_request)["a"] == first.num_spec_tokens_to_schedule
    assert snapshot.host_context_tokens_after_schedule_by_request == (("a", 127), ("b", 1024))
    assert second.num_spec_tokens_to_schedule == s._plan.worker_k


def test_transport_telemetry_does_not_invent_latency_or_physical_state(scheduler_module, monkeypatch, tmp_path):
    path = tmp_path / "states.jsonl"
    s = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_REPLACEMENT_POLICY="1",
              ADAPTIVE_STEP_TELEMETRY_PATH=str(path))
    r = make_request("r")
    r.num_computed_tokens = 127
    add_running(s, r)
    s.schedule()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len([r for r in records if r["record_type"] == "runtime_identity_snapshot"]) == 1
    state_record = next(r for r in records if r["record_type"] == "decision_state_snapshot")
    assert state_record["decision_state"]["context_snapshot_semantics"] == CONTEXT_SEMANTICS
    assert "physical_draft_width" not in state_record["decision_state"]
    assert next(r for r in records if r["record_type"] == "planned_decision")["actual_step_latency_ms"] is None
