"""Synthetic timing fixtures test data handling, never serve as GPU evidence."""
import copy
import math
from types import SimpleNamespace

import pytest

from experiments import m4_cost_profiler as p
from experiments import modal_lanes as lanes
from src.flexsds.core.cost_model import FlexCostModel


def evidence(gpu="L4"):
    env = dict(run_id="synthetic-test-only", source_commit="a"*40,
               requested_gpu=gpu, gpu="NVIDIA L4" if gpu == "L4" else "Tesla T4",
               sm_architecture=p.GPU_CONFIG[gpu][1], vllm_version=lanes.VLLM_VERSION,
               target_revision=lanes.TARGET_REVISION, draft_revision=lanes.EAGLE_REVISION,
               batch_grid=list(p.BATCHES), k_grid=[0,*p.KS], context_len=1024,
               warmup=5, repeats=50, stage="full")
    records = []
    for b in p.BATCHES:
        for k in (0,*p.KS):
            kinds = ("linear_verify_forward", "eagle_propose") if k else ("target_decode_forward",)
            for kind in kinds:
                for index in range(55):
                    val = 2+k if kind == "eagle_propose" else (10+b)*(1+.02*max(0,k-1))
                    records.append(dict(env, K=k, batch_size=b, measurement_kind=kind,
                                        repeat=index if index < 5 else index-5, is_warmup=index < 5,
                                        latency_ms=999 if index < 5 else val, status="ok", timing_method=p.TIMING))
    return env, records


@pytest.mark.parametrize("gpu", ["L4", "T4"])
def test_full_profile_provenance_and_exact_lookup(gpu, monkeypatch):
    env, raw = evidence(gpu)
    result = p.build_profile(env, raw, tree_ops_prior_ms=0)
    assert len(result["cost_matrix_ms"]) == 30
    model = FlexCostModel.from_json(result)
    assert model.provenance == "derived_model_assisted"
    monkeypatch.setattr(model, "get_forward_costs", lambda *a, **kw: pytest.fail("fallback"))
    for t in p.CANONICAL_TOPOLOGIES:
        for b in p.BATCHES:
            key = f"topo_{t}_b{b}"
            value = result["cost_matrix_ms"][key]
            assert math.isfinite(value) and value > 0
            assert model.get_step_cost(t, b) == value
            assert result["cell_provenance"][key]["source"] == (
                "measured" if t == "T0" else "derived_model_assisted")
            if t == "T0":
                assert value == 10+b
    assert result["metadata"]["gamma_attn"] == pytest.approx(.02)
    assert result["metadata"]["attn_verify_factor"] == result["metadata"]["gamma_attn"]
    assert result["metadata"]["gamma_attn_fit"]["rmse_ms"] < 1e-10
    assert not result["metadata"]["runtime_tree_available"]
    assert result["calibration_fits"]["1"]["proposal_slope_ms"] == pytest.approx(1)


@pytest.mark.parametrize("mutation", ["missing", "failed", "duplicate", "mixed_context", "mixed_sha", "nan", "warmup_missing", "wrong_index"])
def test_rejects_bad_evidence(mutation):
    env, raw = evidence()
    if mutation == "missing":
        raw.pop()
    elif mutation == "failed":
        raw[-1]["status"] = "oom"
    elif mutation == "duplicate":
        raw.append(copy.deepcopy(raw[-1]))
    elif mutation == "mixed_context":
        raw[-1]["context_len"] = 512
    elif mutation == "mixed_sha":
        raw[-1]["source_commit"] = "b"*40
    elif mutation == "nan":
        raw[-1]["latency_ms"] = float("nan")
    elif mutation == "warmup_missing":
        raw.pop(0)
    elif mutation == "wrong_index":
        raw[-1]["repeat"] = 500
    with pytest.raises(ValueError):
        p.build_profile(env, raw, tree_ops_prior_ms=0)


@pytest.mark.parametrize("field,value", [("stage","smoke"), ("repeats",49),
                                         ("target_revision","main"), ("vllm_version","latest")])
def test_canonical_publication_gate(field,value):
    env, raw = evidence()
    env[field] = value
    with pytest.raises(ValueError):
        p.build_profile(env,raw,tree_ops_prior_ms=0)


def test_timing_synchronizes_before_and_after_work():
    calls = []
    clock = iter([1,1.025])
    value, ms = p.timed(lambda: calls.append("work"), lambda: calls.append("sync"), lambda: next(clock))
    assert calls == ["sync","work","sync"]
    assert ms == pytest.approx(25)


def test_summary_retains_failures_and_excludes_warmups():
    env, raw = evidence()
    raw[5]["status"] = "oom"
    row = next(r for r in p.summarize(raw) if r["measurement_kind"] == "target_decode_forward" and r["batch_size"] == 1)
    assert row["median_ms"] == 11
    assert row["n_warmup"] == 5 and row["n_success"] == 49 and row["n_failed"] == 1


def test_engine_and_hardware_contract():
    for gpu in ("L4","T4"):
        cfg = p.engine_kwargs(gpu,3)
        assert cfg["speculative_config"] == dict(lanes.LANES["eagle"].speculative_config, revision=lanes.EAGLE_REVISION)
        assert cfg["revision"] == lanes.TARGET_REVISION
        assert not cfg["enable_prefix_caching"]
        assert cfg["enforce_eager"] and not cfg["async_scheduling"]
    assert p.engine_kwargs("T4",0)["dtype"] == "half"
    with pytest.raises(ValueError):
        p.validate_hardware("L4","Tesla T4","sm_75")
    with pytest.raises(ValueError):
        p.validate_hardware("T4","Tesla T4","sm_89")


@pytest.mark.parametrize("bad", ["", "../bad", "a/b", "x..y", "a\\b"])
def test_run_id_safety(tmp_path,bad):
    with pytest.raises(ValueError):
        p.evidence_dir(tmp_path,bad)


def test_derived_cells_cannot_be_relabelled_measured():
    env, raw = evidence()
    result = p.build_profile(env,raw,tree_ops_prior_ms=0)
    result["cell_provenance"]["topo_T8_b1"]["source"] = "measured"
    with pytest.raises(ValueError):
        p.validate_profile(result)


@pytest.mark.parametrize("actual_batch,context,query,expect_sample", [
    (2,1024,2,True), (1,1024,2,False), (2,0,1024,False),
    (2,1025,2,False), (2,1024,1,False)])
def test_worker_observer_checks_actual_shapes(monkeypatch,actual_batch,context,query,expect_sample):
    """Exercise the installed hooks and RPC state without importing a GPU stack."""
    import sys
    from experiments import m4_worker_observer as obs

    class Model:
        def register_forward_pre_hook(self, hook):
            self.before = hook
        def register_forward_hook(self, hook):
            self.after = hook
        def __call__(self):
            self.before(self,())
            self.after(self,(),None)

    class Base:
        pass

    class Eagle(Base):
        parallel_drafting = False
        def propose(self, num_speculative_tokens, common_attn_metadata):
            return SimpleNamespace(shape=(actual_batch,num_speculative_tokens))

    class Runner:
        def execute_model(self, scheduler):
            self.model()

    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules,"vllm",SimpleNamespace(__version__="0.30.0"))
    monkeypatch.setitem(sys.modules,"vllm.v1.spec_decode.eagle",SimpleNamespace(EagleProposer=Eagle))
    monkeypatch.setitem(sys.modules,"vllm.v1.spec_decode.llm_base_proposer",SimpleNamespace(SpecDecodeBaseProposer=Base))
    runner = Runner()
    runner.model = Model()
    runner.speculative_config = object()
    runner.drafter = Eagle()
    runner.input_batch = SimpleNamespace(req_ids=[str(i) for i in range(actual_batch)],
                                        num_reqs=actual_batch, num_computed_tokens_cpu=[context]*actual_batch)
    worker = SimpleNamespace(model_runner=runner)
    hashes = obs.install(worker)
    assert set(hashes) == {"runner","eagle","base_proposer"}
    obs.begin(worker,2,1,1024)
    runner.execute_model(SimpleNamespace(num_scheduled_tokens={str(i):query for i in range(actual_batch)},
                                         preempted_req_ids=None))
    runner.drafter.propose(1,SimpleNamespace(batch_size=lambda: actual_batch))
    result = obs.finish(worker)
    assert set(result["samples"]) == ({"target","propose"} if expect_sample else set())
    assert result["observed_shapes"][0]["actual_batch_size"] == actual_batch
