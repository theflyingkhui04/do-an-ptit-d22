from dataclasses import asdict
import pytest
from src.proposer.cost_policy import SurvivalHistory
from scalar_eagle_fixtures import event


def history(n=2):
    h = SurvivalHistory()
    h.bind_request("A", 1, n)
    return h


def test_adaptive_d_counterexample_common_denominator():
    h = history()
    assert h.add(event(a=0, d=2, base=2))
    assert h.add(event(a=4, d=4, seq=2))
    # Naive D>=j pooling gives [.5,.5,1,1], an impossible rising tail.
    samples = [(0, 2), (4, 4)]
    naive = [sum(a >= j for a, d in samples if d >= j) / sum(d >= j for a, d in samples)
             for j in range(1, 5)]
    assert naive == [.5, .5, 1., 1.]
    assert h.estimate("A", 1, 4).survival == (1.,)*4
    assert h.estimate("A", 1, 2).survival == (0., 0.)
    assert h.add(event(a=0, seq=3))
    e = h.estimate("A", 1, 4)
    assert e.survival == (.5,)*4 and e.count == 2
    assert e.predicted_yield(4) == 3  # Not geometric A/D alpha (1+.5+.25+...).
    assert e.predicted_yield(0) == 1
    with pytest.raises(ValueError):
        e.predicted_yield(5)


@pytest.mark.parametrize("kwargs", [dict(d=2,a=2), dict(d=0,a=0), dict(mode="probe"),
    dict(status="TERMINAL_CENSORED"), dict(status="PREEMPTED_INVALID"),
    dict(status="STALE_INVALID"), dict(status="ABORTED_INVALID"),
    dict(status="AMBIGUOUS_INVALID"), dict(lineage="ambiguous"), dict(epoch=2)])
def test_only_fully_exposed_normal_finalized_same_epoch(kwargs):
    h = history()
    assert not h.add(event(**kwargs))
    assert h.estimate("A", 1, 4).count == 0


def test_zero_accept_rolling_full_window_and_incarnation():
    h = history(10)
    for n in range(9):
        assert h.add(event(a=0, seq=n+1))
    assert not h.estimate("A", 1, 4).ready
    h.add(event(a=4, seq=10))
    e = h.estimate("A", 1, 4)
    assert e.ready and e.survival == (.1,)*4
    for n in range(11, 21):
        h.add(event(a=4, seq=n))
    assert h.estimate("A", 1, 4).survival == (1.,)*4
    assert not any("confidence" in key or "signific" in key for key in asdict(e))
    h.bind_request("A", 2, 10)
    assert not h.add(event())
    assert h.estimate("A", 2, 4).count == 0
    h.reset_epoch("A", 1)
    assert h.estimate("A", 2, 4).count == 0
    assert h.estimate("A", 2, 0).ready


def test_estimator_not_legacy_ema_or_geometric():
    h = history()
    h.add(event(a=0))
    h.add(event(a=4,seq=2))
    e = h.estimate("A",1,4)
    assert e.predicted_yield(2) == 2.
    assert e.predicted_yield(2) != 1 + .5 + .5**2


def test_rolling_failure_and_success_monotonicity():
    import itertools
    for values in itertools.product(range(5),repeat=3):
        for new,sign in ((0,-1),(4,1)):
            h=history(3)
            for seq,a in enumerate(values,1): h.add(event(a=a,seq=seq))
            before=h.estimate("A",1,4).survival
            h.add(event(a=new,seq=4))
            after=h.estimate("A",1,4).survival
            assert all(sign*(y-x)>=0 for x,y in zip(before,after))
