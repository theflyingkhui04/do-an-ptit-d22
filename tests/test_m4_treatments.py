"""CPU configuration tests; no engine or cloud resources."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import m4_treatments as m4


def test_same_trained_speculator_and_target():
    fixed, adaptive = [m4.definition(t) for t in m4.TREATMENTS[1:]]
    for field in ("target_model", "target_revision"):
        assert fixed[field] == adaptive[field]
    for field in ("model", "revision", "method", "draft_sample_method"):
        assert fixed["speculative_config"][field] == adaptive["speculative_config"][field]
    assert fixed["speculative_config"]["model"] == m4.SPECULATOR
    assert fixed["speculative_config"]["revision"] == m4.SPECULATOR_REVISION
    assert fixed["speculative_config"]["method"] == "eagle3"
    assert "0.5B" not in json.dumps(adaptive)


@pytest.mark.parametrize("name", m4.TREATMENTS)
def test_scheduler_metrics_and_deterministic_serialization(name, tmp_path):
    args = m4.serve_args(name)
    assert args == m4.serve_args(name)
    assert ("--scheduler-cls" in args) == (name == "eagle_adaptive")
    assert ("--distributed-executor-backend" in args) == (name == "eagle_adaptive")
    assert ("--per-request-spec-decode-metrics" in args) == (name != "vanilla")
    if name != "vanilla":
        assert json.loads(args[args.index("--speculative-config") + 1]) == m4.engine_kwargs(name)["speculative_config"]
    if name == "eagle_adaptive":
        assert m4.runtime_env(name, tmp_path)["ADAPTIVE_STRICT"] == "1"


@pytest.mark.parametrize("name", ["static", "adaptive", "eagle", "treeadaptive", ""])
def test_no_historical_aliases(name):
    with pytest.raises(ValueError):
        m4.definition(name)


def test_historical_lane_meanings_unchanged():
    from experiments.modal_lanes import LANES
    assert LANES["adaptive"].speculative_config["method"] == "draft_model"
    assert "0.5B" in LANES["adaptive"].speculative_config["model"]
    assert not set(m4.TREATMENTS[1:]) & set(LANES)


def test_eagle_executor_contract(scheduler_module, monkeypatch):
    import ast
    from experiments.m4_protocol import ROOT
    calls = []
    tree = ast.parse((ROOT / "src/server/lineage_executor.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_speculator_contract")
    scope = {"verify_eagle_sources": lambda: calls.append("source-verified")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "executor-contract", "exec"), scope)
    spec = SimpleNamespace(method="eagle3", parallel_drafting=False,
                           num_speculative_tokens_per_batch_size=[[1, 32, 4]])
    scope["validate_speculator_contract"](spec)
    assert calls == ["source-verified"]
    spec.parallel_drafting = True
    with pytest.raises(RuntimeError):
        scope["validate_speculator_contract"](spec)


# Reuse only the CPU vLLM module fixture; it does not create a GPU engine.
from test_adaptive_scheduler_glue import scheduler_module


def test_strict_adaptive_rejects_missing_budget_plumbing(scheduler_module, monkeypatch):
    from test_adaptive_scheduler_glue import paired_cpu_scheduler
    monkeypatch.setenv("ADAPTIVE_STRICT", "1")
    with pytest.raises(RuntimeError, match="dynamic speculative"):
        paired_cpu_scheduler(scheduler_module, dynamic=False)


def test_strict_adaptive_does_not_fall_back_on_planner_failure(scheduler_module, monkeypatch, tmp_path):
    from test_adaptive_scheduler_glue import build
    sched = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_STRICT="1")
    def failed(*args, **kwargs):
        raise ValueError("injected planning failure")
    sched._planner.plan_step = failed
    with pytest.raises(ValueError, match="injected"):
        sched.schedule()
