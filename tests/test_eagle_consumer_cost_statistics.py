from dataclasses import replace, asdict
import random
import pytest
from flexsds.core.eagle_consumer_profile import (
    Cohort, PublicationPlan, publish_cost, ConsumerCostCell, ConsumerProfileError,
)


def cohorts():
    # Two correlated inner-retirement blocks; independent synthetic cohorts.
    return tuple(Cohort(str(i), "synthetic-process", i*100_000_000,
        tuple(i*100_000_000+(j+1)*(1_000_000 if i < 25 else 9_000_000) for j in range(8)),
        (3,)*8) for i in range(50))


def test_fixed_cohort_bootstrap_reproducible_not_inner_iid():
    cs, plan = cohorts(), PublicationPlan(seed=731)
    result = publish_cost(cs,plan)
    assert result == publish_cost(cs,plan)
    rng = random.Random(731)
    values = sorted(sum(rng.choices([1.]*25+[9.]*25,k=50))/50 for _ in range(5000))
    def percentile(f):
        x = 4999*f
        i = int(x)
        return values[i]+(values[min(i+1,4999)]-values[i])*(x-i)
    assert result["C_point"] == 5.
    assert result["C_low"] == pytest.approx(percentile(.025))
    assert result["C_high"] == pytest.approx(percentile(.975))
    rng = random.Random(731)
    iid = sorted(sum(rng.choices([1.]*200+[9.]*200,k=400))/400 for _ in range(5000))
    assert result["C_low"] < iid[125] and result["C_high"] > iid[4874]
    assert result["replicate_unit"] == "independent_cohort"
    assert result["performance_claim"] is False
    cell = ConsumerCostCell.from_publication(result)
    assert not cell.synthetic and cell.C_point == 5.


@pytest.mark.parametrize("change", [dict(measured_cohorts=49),dict(bootstrap_resamples=4999),
    dict(confidence_level=.9),dict(seed=None),dict(replicate_unit="retirement"),
    dict(stopping_rule="collect_until_winner"),dict(method="iid_bootstrap")])
def test_invalid_sample_plans_rejected(change):
    with pytest.raises(ConsumerProfileError):
        PublicationPlan(**change)


def test_fixed_plan_incomplete_and_extra_cohorts_never_publish():
    cs=cohorts()
    assert not publish_cost(cs[:-1])["complete"]
    extra=replace(cs[-1],cohort_id="extra",start_ns=6_000_000_000,
                  retirement_end_ns=tuple(6_000_000_000+j*1_000_000 for j in range(1,9)))
    assert not publish_cost(cs+(extra,))["complete"]
    with pytest.raises(ConsumerProfileError):
        ConsumerCostCell.from_publication(publish_cost(cs[:-1]))


@pytest.mark.parametrize("failure",["oom","timeout","unsupported_shape","observer_failure","error","abort"])
def test_failed_cohorts_preserved_not_dropped(failure):
    cs=cohorts()
    report=publish_cost((replace(cs[0],status=failure,error="synthetic failure"),)+cs[1:])
    assert not report["complete"] and report["failures"] == ["0"]
    assert len(report["raw"]["cohorts"]) == 50
    assert report["raw"]["cohorts"][0]["error"] == "synthetic failure"


@pytest.mark.parametrize("changes",[dict(clock="cuda_event"),dict(added_cuda_sync=True),
    dict(steady_pure_decode=False),dict(start_ns=-1),dict(clock_domain=""),
    dict(retirement_end_ns=(1,)*8),dict(emitted_tokens=(-1,)*8)])
def test_retirement_window_contract(changes):
    cs=cohorts()
    with pytest.raises(ConsumerProfileError):
        publish_cost((replace(cs[0],**changes),)+cs[1:])


def test_same_process_windows_no_overlap_and_ids_unique():
    cs=cohorts()
    with pytest.raises(ConsumerProfileError):
        publish_cost((cs[0],cs[0])+cs[2:])
    with pytest.raises(ConsumerProfileError):
        publish_cost((cs[0],replace(cs[1],start_ns=0))+cs[2:])
