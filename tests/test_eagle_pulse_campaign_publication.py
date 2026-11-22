"""Synthetic whole-cycle publication, routing, profile and hash admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import experiments.eagle_pulse_campaign as c
from experiments.eagle_pulse_campaign_fake import make_plan, FakeExecutor, synthetic_cycle


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("dual-happy") / "campaign"
    plan = make_plan(c.BASE_SHA, 8, True)
    c.initialize(root, plan, "synthetic-happy")
    executor = FakeExecutor(["unsupported", "A10G", "A10G", "A10"])
    state = c.run(root, executor, source=False)
    return root, plan, state, executor


def test_CAM_M01_M02_M03_M04_I08_I10(completed):
    root, plan, state, executor = completed
    assert state["status"] == "COMPLETE"
    assert state["allocation_count"] == 4 and executor.released == [0, 1, 2, 3]
    assert len(executor.attempts) == 300 == len(set(executor.attempts))
    assert executor.attempts[0] == ("NVIDIA A10G", "old0_q0", "000")
    assert executor.attempts[49] == ("NVIDIA A10G", "old0_q0", "049")
    assert executor.attempts[50] == ("NVIDIA A10G", "old0_q2", "000")
    assert executor.attempts[99] == ("NVIDIA A10G", "old0_q2", "049")
    assert executor.attempts[100] == ("NVIDIA A10G", "old0_q3", "000")
    assert executor.attempts[149] == ("NVIDIA A10G", "old0_q3", "049")
    assert executor.attempts[150] == ("NVIDIA A10", "old0_q0", "000")
    assert executor.allocations[0]["remaining"] == 150
    assert executor.allocations[2]["remaining"] == 150
    assert executor.allocations[1]["remaining"] == 0
    # One worker provided 150 distinct request cohorts, never cross-VM claims.
    request_ids = [c.read_json(root / c.slot_path("NVIDIA A10G", cell["id"], slot) / "raw-record.json")["request_id"]
                   for cell in c.CELLS for slot in c.SLOTS]
    assert len(set(request_ids)) == 150


def test_CAM_H11_H12_H13_C07_complete_skips(completed):
    root, plan, state, prior = completed
    fake = FakeExecutor(["A10", "A10G"])
    assert c.run(root, fake, source=False) == state
    assert fake.allocations == [] and fake.attempts == []
    assert c.next_slot(state, "NVIDIA A10") is None
    assert c.next_slot(state, "NVIDIA A10G") is None


@pytest.mark.parametrize("product", c.PRODUCTS, ids=["CAM-K01", "CAM-K02"])
def test_CAM_K15_roundtrip_admission(completed, product):
    root, plan, state, _ = completed
    branch = state["products"][product]
    profile = c.admit(root, plan, product, branch)
    assert c.EaglePulseProfile.from_data(profile.to_data()).to_data() == profile.to_data()
    other = next(p for p in c.PRODUCTS if p != product)
    assert profile.identity_status(c.RuntimeIdentitySnapshotV1(**plan["identities"][other])) == "IDENTITY_MISMATCH"
    assert len(profile.cells) == 3
    assert (profile.objective, profile.horizon, profile.continuation, profile.terminal_value) == (c.PULSE_OBJECTIVE, 2, 0, 0)
    assert profile.synthetic and "synthetic-publication" in branch["profile_path"]
    assert not (root / "publication").exists()


def test_CAM_J01_J02_J03_J04_J07_J09_reviewed_bootstrap(monkeypatch, completed):
    root, plan, state, _ = completed
    rows = [c.read_json(root / c.slot_path("NVIDIA A10", "old0_q0", s) / "raw-record.json") for s in c.SLOTS]
    rows = c.publication_cycles(rows, True)
    import flexsds.core.eagle_consumer_profile as production
    original = production.random.Random
    calls = []
    class Count(original):
        def __init__(self, seed):
            assert seed == 731
            super().__init__(seed)
        def choices(self, population, *, k):
            assert len(population) == k == 50
            calls.append(1)
            return super().choices(population, k=k)
    monkeypatch.setattr(production.random, "Random", Count)
    first = c.publish_pulse_cost(rows)
    assert len(calls) == 5000
    assert first == c.publish_pulse_cost(rows)
    assert first["confidence_level"] == .95 and first["seed"] == 731
    assert first["C_low"] <= first["C_point"] <= first["C_high"]
    assert first["C_point"] == rows[0]["cost_ms"]
    assert "publish_pulse_cost(publication_cycles(" in Path(c.__file__).read_text()


@pytest.mark.parametrize("count", [49, 51], ids=["CAM-J08", "CAM-J10"])
def test_wrong_fixed_slot_count(completed, count):
    root, plan, state, _ = completed
    rows = c.publication_cycles([c.read_json(root / c.slot_path("NVIDIA A10", "old0_q0", s) / "raw-record.json") for s in c.SLOTS], True)
    if count == 49:
        rows.pop()
    else:
        failed = deepcopy(rows[0])
        failed.update(cohort_id="hidden-failed", request_id="hidden-request", status="INVALID", failures=["OOM"])
        rows.append(failed)
    assert not c.publish_pulse_cost(rows)["complete"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1], ids=["CAM-J05-nan", "CAM-J05-inf", "CAM-J06-zero", "CAM-J06-negative"])
def test_bad_intervals(value):
    with pytest.raises(ValueError): c.PulseCostCell.synthetic_fixture(value)
    with pytest.raises(ValueError): c.PulseCostCell.synthetic_fixture(1, 2, 3)


@pytest.mark.parametrize("field,value", [("objective", "component"), ("horizon", 1), ("continuation", 2),
    ("protocol", "flexsds-component-v2"), ("protocol", "first-response-v1"), ("protocol", "steady-state-v1")],
    ids=["CAM-K05", "CAM-K06", "CAM-K07", "CAM-K12-O14", "CAM-K13", "CAM-K14"])
def test_profile_type_negative(completed, field, value):
    root, plan, state, _ = completed
    data = c.read_json(root / state["products"]["NVIDIA A10"]["profile_path"])
    data[field] = value
    with pytest.raises(ValueError): c.EaglePulseProfile.from_data(data)


@pytest.mark.parametrize("defect", ["raw_missing", "raw_hash", "failed_to_valid", "duplicate", "hidden", "missing_cell"],
    ids=["CAM-H06-K11", "CAM-H07-O07", "CAM-G08-E04-E05", "CAM-O09", "CAM-J10-hidden", "CAM-K04"])
def test_raw_and_state_negative(tmp_path, defect):
    plan = make_plan(c.BASE_SHA, 1, True)
    root = tmp_path / "negative"
    c.initialize(root, plan, "raw-negative")
    failure = {("NVIDIA A10", "old0_q0", "000"): "timeout"} if defect == "failed_to_valid" else None
    state = c.run(root, FakeExecutor(["A10"], slots_per_worker=2, failures=failure), source=False)
    entry = state["products"]["NVIDIA A10"]["cells"]["old0_q0"]
    slot = entry["slots"]["000"]
    directory = root / c.slot_path("NVIDIA A10", "old0_q0", "000")
    if defect == "raw_missing": (directory / "raw-manifest.json").rename(directory / "missing-manifest.json")
    if defect == "raw_hash": (directory / "raw-record.json").write_text("{}")
    if defect == "failed_to_valid": slot.update(status="VALID", failure_code=None)
    if defect == "duplicate": entry["slots"]["001"] = dict(slot)
    if defect == "hidden":
        (root / "raw/hidden").mkdir()
        (root / "raw/hidden/raw-record.json").write_text("{}")
    if defect == "missing_cell": state["products"]["NVIDIA A10"]["cells"].pop("old0_q3")
    c.atomic_json(root / "campaign-state.json", state)
    executor = FakeExecutor(["A10"])
    with pytest.raises((ValueError, OSError, TypeError)):
        c.run(root, executor, source=False)
    assert executor.allocations == []


def test_CAM_M08_O15_synthetic_isolation(completed):
    root, plan, state, _ = completed
    assert not (root / "publication").exists()
    for product in c.PRODUCTS:
        profile = c.read_json(root / state["products"][product]["profile_path"])
        assert profile["synthetic"] is True
        assert not {"speedup", "TPOT", "throughput"} & profile.keys()
    assert "not-cross-VM-independent" in plan["domain"]["replicate_definition"]


def test_CAM_H08_profile_mutation(completed):
    root, plan, state, _ = completed
    path = root / state["products"]["NVIDIA A10"]["profile_path"]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(c.CampaignError, match="profile_hash"):
            c.validate(root, source=False)
    finally:
        path.write_bytes(original)


def test_CAM_K04_missing_published_q(completed, monkeypatch):
    root, plan, state, _ = completed
    branch = state["products"]["NVIDIA A10"]
    path = root / branch["profile_path"]
    original = c.read_json
    def missing_q(candidate):
        data = original(candidate)
        if Path(candidate) == path:
            data["cells"].pop()
        return data
    monkeypatch.setattr(c, "read_json", missing_q)
    with pytest.raises(c.CampaignError, match="profile_cell_set"):
        c.admit(root, plan, "NVIDIA A10", branch)


def test_measured_schema_contract_in_memory_only(completed):
    # Explicit CPU schema exercise, as in the established production T28
    # fixture. No synthetic raw bytes or live profile paths are changed.
    root, plan, state, _ = completed
    product = "NVIDIA A10"
    cells = []
    for cell in c.CELLS:
        cycles = [c.read_json(root / c.slot_path(product, cell["id"], slot) / "raw-record.json") for slot in c.SLOTS]
        copies = c.publication_cycles(cycles, True)
        cost = c.cost_cell(copies, False)
        assert not cost.synthetic
        cells.append((copies[0]["initial_key"], cost))
    identity = c.RuntimeIdentitySnapshotV1(**plan["identities"][product])
    profile = c.EaglePulseProfile("CPU-schema-contract-only", identity, c.canonical(plan["domain"]), tuple(cells), False)
    assert c.EaglePulseProfile.from_data(profile.to_data()).identity_status(identity) == "VALID_MATCH"
    assert not (root / "publication").exists()
    raw = c.read_json(root / c.slot_path(product, "old0_q0", "000") / "raw-record.json")
    assert raw["synthetic"] is True
