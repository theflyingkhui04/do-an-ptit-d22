"""No-external-action CLI, source gates and adapter protocol witnesses."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

import experiments.eagle_pulse_campaign as c
from experiments.eagle_pulse_campaign_fake import make_plan, FakeExecutor


@pytest.fixture
def local(tmp_path, monkeypatch):
    root = tmp_path / "cli"
    plan = make_plan(c.BASE_SHA, 2, True)
    c.initialize(root, plan, "cpu-cli")
    original = c.validate
    monkeypatch.setattr(c, "validate", lambda *a, **kw: original(*a, **dict(kw, source=False)))
    return root, plan


@pytest.mark.parametrize("verb", [[], ["dry-run"], ["run", "--dry-run"]], ids=["CAM-L09", "CAM-L01-L02", "CAM-L01-explicit-override"])
def test_non_live_default(local, monkeypatch, capsys, verb):
    root, plan = local
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("external operation attempted: " + name)
    monkeypatch.setitem(sys.modules, "modal", Forbidden())
    assert c.main([*verb, "--campaign-dir", str(root)]) == 0
    import json
    output = json.loads(capsys.readouterr().out)
    assert output["source_git_sha"] == c.BASE_SHA
    assert output["plan_sha256"] == c.digest(plan)
    assert output["remaining_allocations"] == 2
    assert set(output["products"]) == set(c.PRODUCTS)
    for branch in output["products"].values():
        assert branch["next_action"] == ["old0_q0", "000"]
        assert all(cell["UNATTEMPTED"] == 50 for cell in branch["cells"].values())


def test_CAM_L08_invalid_state_nonzero(local):
    root, _ = local
    (root / "campaign-state.json").write_text("corrupt")
    assert c.main(["--campaign-dir", str(root)]) == 2


def test_live_requires_explicit_resume_and_live_plan(local, monkeypatch):
    root, _ = local
    assert c.main(["run", "--campaign-dir", str(root)]) == 2
    assert c.main(["run", "--campaign-dir", str(root), "--resume"]) == 2


def test_CAM_I03_frozen_bound_cli(local):
    root, _ = local
    assert c.main(["--campaign-dir", str(root), "--max-allocations", "3"]) == 2


@pytest.mark.parametrize("defect", ["branch", "sha", "tracked", "untracked"], ids=["CAM-A02", "CAM-A01-H05", "CAM-A01-dirty", "CAM-A01-unfrozen"])
def test_source_gate_before_external(tmp_path, monkeypatch, defect):
    def git(root, *args):
        if args == ("branch", "--show-current"): return "wrong" if defect == "branch" else c.BRANCH
        if args == ("rev-parse", "HEAD"): return "0" * 40 if defect == "sha" else c.BASE_SHA
        if args[0] == "diff": return "src/dirty.py" if defect == "tracked" else ""
        if args[0] == "status": return "?? experiments/dirty.py" if defect == "untracked" else ""
        pytest.fail(str(args))
    monkeypatch.setattr(c, "git", git)
    with pytest.raises(c.CampaignError): c.source_gate(tmp_path, c.BASE_SHA)


def test_CAM_A03_pinned_source():
    import os
    from experiments.m4_pinned_contract import verify
    root = Path(os.environ["M4_VLLM_SOURCE_ROOT"])
    result = verify(root)
    assert result["version"] == "0.30.0" and result["verified_files"] == 19


def test_CAM_A04_production_objective():
    from src.server.budget_planner import AdaptiveBudgetPlanner
    from src.proposer.pulse_policy import PulseCostGovernor
    assert c.PULSE_OBJECTIVE == "SECP-PULSE-2R-v1"
    assert isinstance(AdaptiveBudgetPlanner.from_env({"ADAPTIVE_COST_POLICY_MODE": "strict",
        "ADAPTIVE_REPLACEMENT_POLICY": "1"}).cost_governor, PulseCostGovernor)


def test_adapter_ack_then_release_without_modal():
    from experiments.eagle_pulse_campaign_modal import ModalExecutor
    events = []
    def generator():
        events.append("next-after-command")
        yield dict(kind="slot", attempt_id="a", result=dict(raw="synthetic protocol"))
        events.append("worker-stopped")
    class Commands:
        def put(self, command, timeout):
            events.append(command["kind"])
    class Stack:
        def close(self): events.append("resources-released")
    worker = dict(commands=Commands(), generator=generator(), stack=Stack())
    adapter = ModalExecutor(Path("unused"))
    assert adapter.execute_slot(worker, {}, "NVIDIA A10", "old0_q0", "000", "a") == dict(raw="synthetic protocol")
    events.append("host-raw-state-persisted")
    adapter.close(worker)
    assert events == ["slot", "next-after-command", "host-raw-state-persisted", "stop", "worker-stopped", "resources-released"]


def test_adapter_bounds_static():
    source = (c.ROOT / "experiments/eagle_pulse_campaign_modal.py").read_text()
    assert "max_containers=1" in source and "retries=0" in source and "single_use_containers=True" in source
    assert "include_source=False" in source
    assert "commands.get(timeout=120)" in source and "range(151)" in source
    assert "import modal" not in Path(c.__file__).read_text()


def test_CAM_K08_K09_K10_G04_runtime_source_negative(local):
    root, plan = local
    from experiments.eagle_pulse_campaign_fake import synthetic_cycle
    row = synthetic_cycle(plan, "NVIDIA A10", "old0_q0", "000")
    identity = c.RuntimeIdentitySnapshotV1(**plan["identities"]["NVIDIA A10"])
    for key, wrong in [("source_git_sha", "e" * 40), ("runtime_identity_key", "0" * 64), ("campaign_plan_sha256", "0" * 64)]:
        bad = deepcopy(row)
        bad[key] = wrong
        with pytest.raises(c.CampaignError): c.validate_cycle(bad, plan, "NVIDIA A10", "old0_q0", "000", identity)
    bad = deepcopy(plan)
    bad["identities"]["NVIDIA A10"]["configured"][0][1] = "wrong-model"
    bad["runtime_contract_hash"] = c.digest(bad["identities"])
    with pytest.raises(c.CampaignError): c.validate_plan(bad)
