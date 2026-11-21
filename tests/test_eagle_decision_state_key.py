from dataclasses import replace
import json
import pytest
from flexsds.core.eagle_consumer_profile import ConsumerProfileError, decision_state_key
from scalar_eagle_fixtures import planner


def test_exact_aligned_vectors_not_batch_or_max_only():
    p,s = planner()
    state = dict(p._candidate_decision_states(s, {"A":4,"B":2}, ()))[4]
    keys = []
    for vector in ((4,4),(4,0),(4,2)):
        keys.append(decision_state_key(replace(state,
            candidate_future_width_by_request=tuple(zip(("A","B"),vector)))))
    assert len(set(keys)) == 3
    assert json.loads(keys[2])["old_verification_widths"] == [4,4]
    assert json.loads(keys[2])["candidate_future_widths"] == [4,2]
    with pytest.raises(ConsumerProfileError):
        p.cost_governor.profile.lookup(replace(state,
            candidate_future_width_by_request=(("A",4),("B",4))))


@pytest.mark.parametrize("changes", [
    dict(eligible_decode_rows=1), dict(context_bucket_by_request=(("A","[256,1024)"),("B","[0,256)"))),
    dict(runtime_mode_id="other"), dict(pure_decode_state=False),
    dict(unsupported_reasons=("missing_worker_shape",)),
    dict(old_verification_width_by_request=(("A",0),("B",0))),
    dict(candidate_future_width_by_request=(("A",4),("B",0))),
])
def test_no_nearest_state_and_no_worker_backfill(changes):
    p,s = planner()
    state = dict(p._candidate_decision_states(s,{"A":4,"B":2},()))[4]
    with pytest.raises(ConsumerProfileError):
        p.cost_governor.profile.lookup(replace(state,**changes))


def test_context_and_runtime_identity_explicit_no_physical_width():
    p,s = planner()
    state = dict(p._candidate_decision_states(s,{"A":4,"B":2},()))[4]
    data = json.loads(decision_state_key(state))
    assert data["context_buckets"] == ["[0,256)"]*2
    assert data["eligible_decode_rows"] == 2
    assert data["runtime_identity_key"] == p.runtime_identity.identity_key
    assert "physical_draft_width" not in data and "client_concurrency" not in data
    # Request labels are not cost dimensions; row alignment still is.
    renamed = replace(state, old_verification_width_by_request=(("X",4),("Y",4)),
        candidate_future_width_by_request=(("X",4),("Y",2)),
        context_bucket_by_request=(("X","[0,256)"),("Y","[0,256)")))
    assert decision_state_key(renamed) == decision_state_key(state)
