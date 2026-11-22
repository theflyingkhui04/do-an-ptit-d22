"""CAM acceptance witnesses: CPU-only, actual production pulse receipts."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

import experiments.eagle_pulse_campaign as c
from experiments.eagle_pulse_campaign_fake import make_plan, FakeExecutor, synthetic_cycle


@pytest.fixture
def plan():
    return make_plan(c.BASE_SHA, 8, True)


@pytest.fixture
def campaign(tmp_path, plan):
    root = tmp_path / "synthetic-campaign"
    c.initialize(root, plan, "cpu-campaign")
    return root


def save(root, state):
    c.atomic_json(root / "campaign-state.json", state)


def test_CAM_B01_canonical_order(plan):
    assert c.canonical(plan) == c.canonical(dict(reversed(list(plan.items()))))
    assert c.digest(plan) == c.digest(dict(reversed(list(plan.items()))))


@pytest.mark.parametrize("key", ["products", "source_git_sha", "cells", "cohort_slots_per_cell", "bootstrap_resamples",
    "confidence_level", "bootstrap_seed", "runtime_contract_hash", "preparation_contract_hash", "queue_contract_hash",
    "continuation_contract_hash", "failure_policy"], ids=lambda s: "CAM-B02-" + s)
def test_scientific_mutation_hash(plan, key):
    bad = deepcopy(plan)
    bad[key] = "changed"
    assert c.digest(bad) != c.digest(plan)
    with pytest.raises((ValueError, TypeError, KeyError)):
        c.validate_plan(bad)


@pytest.mark.parametrize("q", [0, 2, 3], ids=["CAM-D01", "CAM-D02", "CAM-D03"])
def test_cells_exact(plan, q):
    cell = next(cell for cell in plan["cells"] if cell["future_k"] == q)
    assert cell == dict(id=f"old0_q{q}", future_k=q, old_width=0, shared_k=q, continuation_k=0, horizon=2)


@pytest.mark.parametrize("mutator", [
    lambda p: p["products"].append("NVIDIA L4"),
    lambda p: p["cells"].append(dict(p["cells"][2], id="old0_q4", future_k=4)),
    lambda p: p["cells"].pop(),
    lambda p: p["cells"][1].update(old_width=2),
    lambda p: p.update(max_allocations=0),
    lambda p: p.update(max_allocations=-1),
    lambda p: p["modal"].update(max_containers=2),
    lambda p: p["modal"].update(retries=1),
], ids=["CAM-B04", "CAM-B05-D04-O12", "CAM-B05-missing", "CAM-D05-O13", "CAM-I03-zero", "CAM-I03-negative", "CAM-I05", "CAM-I06"])
def test_bad_plan(plan, mutator):
    mutator(plan)
    with pytest.raises(c.CampaignError):
        c.validate_plan(plan)


def test_CAM_B06_E01_E02_fixed_slots(campaign, plan):
    _, state = c.validate(campaign, source=False)
    assert (plan["cohort_slots_per_cell"], plan["bootstrap_resamples"], plan["confidence_level"], plan["bootstrap_seed"]) == (50, 5000, .95, 731)
    for branch in state["products"].values():
        for entry in branch["cells"].values():
            assert tuple(entry["slots"]) == c.SLOTS
    assert c.read_json(campaign / "campaign-state.json") == state


@pytest.mark.parametrize("sequence", [["NVIDIA L4"], ["NVIDIA A10x"], ["NVIDIA A10G extra"], ["unsupported"] * 8],
    ids=["CAM-C06-I07", "CAM-O10-prefix", "CAM-O10-suffix", "CAM-M07-repeated-unsupported"])
def test_unsupported_allocations_do_not_mutate_science(campaign, sequence):
    before = c.read_json(campaign / "campaign-state.json")["products"]
    executor = FakeExecutor(sequence)
    state = c.run(campaign, executor, source=False)
    assert state["status"] == "MAX_ALLOCATIONS_REACHED"
    assert state["allocation_count"] == 8 == len(executor.allocations)
    assert executor.attempts == [] and executor.released == list(range(8))
    assert state["products"] == before
    for item in state["allocations"]:
        allocation = c.read_json(campaign / item["path"])
        assert allocation["status"] == "UNSUPPORTED_ALLOCATION" and allocation["released"]
        assert allocation["observed_identity"]
    assert not (campaign / "raw").exists()


@pytest.mark.parametrize("product", ["A10", "A10G"], ids=["CAM-C01-C05", "CAM-C02"])
def test_routing_and_partial_resume(campaign, product):
    executor = FakeExecutor([product] * 8, slots_per_worker=1)
    state = c.run(campaign, executor, source=False)
    selected = "NVIDIA " + product
    assert {attempt[0] for attempt in executor.attempts} == {selected}
    assert [s for _, _, s in executor.attempts] == [f"{i:03d}" for i in range(8)]
    assert state["products"][next(p for p in c.PRODUCTS if p != selected)]["status"] == "MISSING"
    assert c.next_slot(c.read_json(campaign / "campaign-state.json"), selected) == ("old0_q0", "008")


def test_CAM_H10_M05_interruption_resume(campaign):
    executor = FakeExecutor(["A10"], slots_per_worker=10)
    class Interrupt(FakeExecutor):
        def allocate(self, plan, index):
            if index == 1:
                raise KeyboardInterrupt("simulated interruption between slots")
            return super().allocate(plan, index)
    first = Interrupt(["A10"], slots_per_worker=10)
    with pytest.raises(KeyboardInterrupt):
        c.run(campaign, first, source=False)
    _, partial = c.validate(campaign, source=False)
    assert c.next_slot(partial, "NVIDIA A10") == ("old0_q0", "010")
    resumed = FakeExecutor(["A10", "A10G"])
    final = c.run(campaign, resumed, source=False)
    assert final["status"] == "COMPLETE"
    assert resumed.attempts[0] == ("NVIDIA A10", "old0_q0", "010")
    assert len(first.attempts) + len(resumed.attempts) == 300


@pytest.mark.parametrize("failure", ["OOM", "FAILED_TIMEOUT", "FAILED_TERMINAL", "FAILED_STATE_MISMATCH", "collector_failure", "lineage_failure"],
    ids=["CAM-G01", "CAM-G02", "CAM-G03", "CAM-G05", "CAM-G06", "CAM-G07"])
def test_failure_sticky_failfast(campaign, failure):
    key = ("NVIDIA A10", "old0_q0", "000")
    executor = FakeExecutor(["A10", "A10G"], failures={key: failure})
    state = c.run(campaign, executor, source=False)
    record = state["products"][key[0]]["cells"][key[1]]["slots"][key[2]]
    assert state["status"] == "BLOCKED" and state["allocation_count"] == 1
    assert executor.attempts == [key] and executor.released == [0]
    assert record["status"] == "FAILED" and record["failure_code"] == failure
    assert sum(r["status"] == "VALID" for r in state["products"][key[0]]["cells"][key[1]]["slots"].values()) == 0
    assert len(state["products"][key[0]]["cells"][key[1]]["slots"]) == 50
    assert (campaign / record["raw_manifest"]).exists()
    assert (campaign / Path(record["raw_manifest"]).parent / "failure.json").exists()
    assert c.run(campaign, FakeExecutor(["A10"]), source=False) == state
    with pytest.raises(c.CampaignError, match="slot_transition"):
        c.persist_slot(campaign, make_plan(c.BASE_SHA, 8, True), state, *key, {})


def test_CAM_H01_H02_atomic_replace(tmp_path):
    path = tmp_path / "state.json"
    c.atomic_json(path, {"generation": 1})
    def crash():
        assert c.read_json(path) == {"generation": 1}
        raise RuntimeError("crash before replace")
    with pytest.raises(RuntimeError):
        c.atomic_json(path, {"generation": 2}, crash)
    assert c.read_json(path) == {"generation": 1}


def test_CAM_H09_single_writer(campaign):
    with c.writer(campaign):
        with pytest.raises(c.CampaignError, match="single_writer"):
            with c.writer(campaign):
                pytest.fail("second writer admitted")
        with pytest.raises(c.CampaignError, match="single_writer"):
            c.validate(campaign, source=False)


@pytest.mark.parametrize("defect", ["schema", "enum", "hash", "slots", "forged_valid", "running", "duplicate_json"],
    ids=["CAM-O04", "CAM-O03", "CAM-H04", "CAM-O08", "CAM-O06", "CAM-E03-crash", "CAM-O08-json"])
def test_corrupt_state_stops_before_allocate(campaign, defect):
    state = c.read_json(campaign / "campaign-state.json")
    slot = state["products"]["NVIDIA A10"]["cells"]["old0_q0"]["slots"]["000"]
    if defect == "schema": state["schema_version"] = "unknown"
    if defect == "enum": state["status"] = "MAGIC"
    if defect == "hash": state["plan_sha256"] = "0" * 64
    if defect == "slots": state["products"]["NVIDIA A10"]["cells"]["old0_q0"]["slots"]["050"] = dict(slot)
    if defect == "forged_valid": slot.update(status="VALID")
    if defect == "running": slot.update(status="RUNNING", attempt_id="interrupted")
    save(campaign, state)
    if defect == "duplicate_json":
        (campaign / "campaign-state.json").write_text('{"status":"READY","status":"COMPLETE"}')
    executor = FakeExecutor(["A10"])
    with pytest.raises((ValueError, TypeError, OSError)):
        c.run(campaign, executor, source=False)
    assert executor.allocations == []


def test_CAM_H03_parse_failure(campaign):
    (campaign / "campaign-state.json").write_text("{corrupt")
    with pytest.raises(ValueError): c.validate(campaign, source=False)


def test_CAM_B03_plan_hash_frozen(campaign):
    plan = c.read_json(campaign / "campaign-plan.json")
    plan["source_git_sha"] = "d" * 40
    c.atomic_json(campaign / "campaign-plan.json", plan)
    with pytest.raises(c.CampaignError): c.validate(campaign, source=False)


def test_CAM_O01_path_traversal(campaign):
    for path in ("../escape", str(campaign.parent / "escape")):
        with pytest.raises(c.CampaignError): c.safe(campaign, path)
    with pytest.raises(c.CampaignError): c.slot_path("NVIDIA A10", "../escape", "000")
    with pytest.raises(c.CampaignError): c.new_state(make_plan(c.BASE_SHA, 8, True), "../bad")


def test_CAM_O02_symlink(campaign, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (campaign / "raw").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        # Windows without symlink privilege: junction-independent resolved-path
        # negative control still exercises containment; do not report skipped.
        with pytest.raises(c.CampaignError): c.safe(campaign, outside)
        return
    with pytest.raises(c.CampaignError): c.safe(campaign, "raw/escape.json")


def test_CAM_O05_done_marker_no_authority(campaign):
    (campaign / "A10.done").touch()
    _, state = c.validate(campaign, source=False)
    assert state["products"]["NVIDIA A10"]["status"] == "MISSING"


@pytest.mark.parametrize("defect", ["base", "old", "rows", "context", "pulse", "continuation", "origin1", "origin2", "third", "tokens", "physical", "objective", "timing"],
    ids=["CAM-F03", "CAM-F02", "CAM-F04", "CAM-F05-D06", "CAM-F06", "CAM-F07-F08", "CAM-F09-F11", "CAM-F10", "CAM-F12", "CAM-F15", "CAM-F16", "CAM-F17", "CAM-F13"])
def test_scientific_receipt_negative(plan, defect):
    row = synthetic_cycle(plan, "NVIDIA A10", "old0_q2", "000")
    if defect == "base": row["pulse_receipt"]["base_k"] = 4
    if defect == "old": row["pulse_receipt"]["decision_state"]["old_verification_width_by_request"][0][1] = 2
    if defect == "rows": row["pulse_receipt"]["decision_state"]["eligible_decode_request_ids"].append("extra")
    if defect == "context": row["pulse_receipt"]["decision_state"]["host_context_tokens_after_schedule_by_request"][0][1] = 512
    if defect == "pulse": row["pulse_receipt"]["worker_k"] = 3
    if defect == "continuation": row["continuation_receipt"]["worker_k"] = 2
    if defect == "origin1": row["retirements"][0]["origin"]["schedule_sequence"] = 0
    if defect == "origin2": row["retirements"][1]["origin"]["plan_sequence"] = 99
    if defect == "third": row["retirements"].append(deepcopy(row["retirements"][1]))
    if defect == "tokens": row["retirements"][1].pop("useful_tokens")
    if defect == "physical": row["retirements"][1]["physical"]["width"] = 2
    if defect == "objective": row["objective"] = "component"
    if defect == "timing": row["end_ns"] = row["start_ns"]
    identity = c.RuntimeIdentitySnapshotV1(**plan["identities"]["NVIDIA A10"])
    with pytest.raises((ValueError, KeyError)):
        c.validate_cycle(row, plan, "NVIDIA A10", "old0_q2", "000", identity)


def test_CAM_F14_I09_static_bounds():
    worker = (c.ROOT / "experiments/eagle_pulse_campaign_worker.py").read_text()
    assert "cuda.synchronize" not in worker
    source = Path(c.__file__).read_text()
    assert 'state["allocation_count"] < plan["max_allocations"]' in source


def test_CAM_I11_history_leak_rejected(plan):
    fake = FakeExecutor(["A10"])
    worker = fake.allocate(plan, 0)
    worker["history"] = "previous-incarnation"
    with pytest.raises(c.CampaignError, match="history_cache_leak"):
        fake.execute_slot(worker, plan, "NVIDIA A10", "old0_q0", "000", "attempt")


@pytest.mark.parametrize("product,wrong", [("A10", "NVIDIA A10G"), ("A10G", "NVIDIA A10")], ids=["CAM-C03", "CAM-C04"])
def test_cross_product_worker_write(plan, product, wrong):
    executor = FakeExecutor([product])
    worker = executor.allocate(plan, 0)
    with pytest.raises(c.CampaignError, match="cross_product"):
        executor.execute_slot(worker, plan, wrong, "old0_q0", "000", "attempt")


def test_CAM_I04_exact_allocation_boundary(tmp_path):
    root = tmp_path / "one-allocation"
    c.initialize(root, make_plan(c.BASE_SHA, 1, True), "one-allocation")
    executor = FakeExecutor(["NVIDIA L4", "A10"])
    before = c.read_json(root / "campaign-state.json")["products"]
    state = c.run(root, executor, source=False)
    assert len(executor.allocations) == state["allocation_count"] == 1
    assert state["status"] == "MAX_ALLOCATIONS_REACHED" and state["products"] == before
    assert executor.released == [0] and executor.attempts == []


def test_worker_exception_is_durable_failed_attempt(campaign):
    class Throws(FakeExecutor):
        def execute_slot(self, *args):
            raise TimeoutError("lost worker result")
    fake = Throws(["A10", "A10G"])
    state = c.run(campaign, fake, source=False)
    assert state["status"] == "BLOCKED" and state["allocation_count"] == 1
    record = state["products"]["NVIDIA A10"]["cells"]["old0_q0"]["slots"]["000"]
    assert record["status"] == "FAILED" and "TimeoutError" in record["failure_code"]
    assert fake.released == [0]
    c.validate(campaign, source=False)


def test_allocation_identity_failure_preserved_and_cannot_resume(campaign):
    before = c.read_json(campaign / "campaign-state.json")["products"]
    class Mismatch(FakeExecutor):
        def observe_identity(self, worker):
            identity = super().observe_identity(worker)
            identity["observed"][1][1] = "wrong-architecture"
            return identity
    fake = Mismatch(["A10"])
    with pytest.raises(c.CampaignError, match="identity_mismatch"):
        c.run(campaign, fake, source=False)
    state = c.read_json(campaign / "campaign-state.json")
    assert state["products"] == before and fake.released == [0] and fake.attempts == []
    raw = c.read_json(campaign / state["allocations"][0]["path"])
    assert raw["status"] == "INVARIANT_FAILURE" and raw["observed_identity"]
    second = FakeExecutor(["A10"])
    with pytest.raises(c.CampaignError, match="requires_review"):
        c.run(campaign, second, source=False)
    assert second.allocations == []


def test_CAM_O09_duplicate_request_provenance_blocks_before_next_slot(campaign):
    def cycles(plan, product, cell, slot):
        row = synthetic_cycle(plan, product, cell, slot)
        if slot == "001":
            previous = row["request_id"].removesuffix("001") + "000"
            row = json.loads(c.canonical(row).replace(row["request_id"], previous))
        return row
    fake = FakeExecutor(["A10"], cycle_factory=cycles)
    state = c.run(campaign, fake, source=False)
    slots = state["products"]["NVIDIA A10"]["cells"]["old0_q0"]["slots"]
    assert state["status"] == "BLOCKED" and len(fake.attempts) == 2
    assert slots["000"]["status"] == "VALID" and slots["001"]["status"] == "FAILED"
    assert "duplicate_cohort_provenance" in slots["001"]["failure_code"]
    assert slots["002"]["status"] == "UNATTEMPTED"
    c.validate(campaign, source=False)


@pytest.mark.parametrize("field,value", [("old_width", False), ("future_k", 0.0), ("horizon", 2.0)])
def test_plan_cell_types_are_exact(plan, field, value):
    plan["cells"][0][field] = value
    with pytest.raises(c.CampaignError): c.validate_plan(plan)
