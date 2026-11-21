from dataclasses import replace
import json
import pytest
from flexsds.core.eagle_consumer_profile import (
    ConsumerCostCell, ConsumerProfileError, EagleConsumerProfile, digest, canonical,
    publish_cost, PublicationPlan,
)
from scalar_eagle_fixtures import planner
from test_eagle_consumer_cost_statistics import cohorts


def test_consumer_serialization_raw_digest_and_fixed_metadata():
    p,_=planner()
    profile=p.cost_governor.profile
    restored=EagleConsumerProfile.from_data(json.loads(json.dumps(profile.to_data())))
    assert restored == profile and restored.sha256 == profile.sha256
    data=restored.to_data()
    assert data["unit"] == "ms/retired_batch"
    assert data["timing_boundary"] == "enginecore_retirement_window_v1"
    assert data["clock"] == "perf_counter_ns" and data["added_cuda_sync"] is False
    assert data["synthetic"] is True


@pytest.mark.parametrize("change",[dict(k0_semantics="vanilla_target_only"),dict(unit="ms/token"),
    dict(protocol="component_latency"),dict(added_cuda_sync=True),dict(clock="cuda_event"),
    dict(timing_boundary="sum_components"),dict(alias_rule_version="A10_equals_A10G")])
def test_primary_profile_contract_rejects_component_conversion(change):
    p,_=planner()
    with pytest.raises(ConsumerProfileError):
        replace(p.cost_governor.profile,**change)


@pytest.mark.parametrize("point,low,high",[(0,0,0),(1,2,3),(3,1,2),(-1,-2,-.5),
    (float("inf"),1,float("inf")),(float("nan"),1,2)])
def test_invalid_or_incomplete_cost_intervals(point,low,high):
    with pytest.raises(ConsumerProfileError):
        ConsumerCostCell.synthetic_fixture(point,low,high)


def test_raw_evidence_tamper_and_fake_measured_cell_rejected():
    cell=ConsumerCostCell.synthetic_fixture(2)
    with pytest.raises(ConsumerProfileError):
        replace(cell,raw_sha256="0"*64)
    with pytest.raises(ConsumerProfileError):
        replace(cell,synthetic=False)
    report=publish_cost(cohorts(),PublicationPlan(seed=3))
    measured=ConsumerCostCell.from_publication(report)
    with pytest.raises(ConsumerProfileError):
        replace(measured,C_point=5.1)
    raw=json.loads(measured.evidence_json)
    raw["cohorts"][0]["status"]="timeout"
    raw["cohorts"][0]["error"]="retained failure"
    with pytest.raises(ConsumerProfileError):
        replace(measured,evidence_json=canonical(raw),raw_sha256=digest(raw))


def test_historical_component_profile_not_consumer_profile():
    with pytest.raises(ConsumerProfileError):
        EagleConsumerProfile.from_data({"hardware":"L4", "costs":{"0":1,"4":2}})
