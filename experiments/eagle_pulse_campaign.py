"""Fixed-slot PULSE campaign driver. External work requires the explicit run verb.

All decisions and persistence are local; an executor owns worker lifetimes.
No Modal import or external operation occurs on import, dry-run or validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone

from flexsds.core.eagle_consumer_profile import (
    EaglePulseProfile, PulseCostCell, PULSE_OBJECTIVE, PULSE_UNIT, canonical,
    digest, publish_pulse_cost, validate_pulse_cycle, validate_identity,
)
from src.server.decision_state import RuntimeIdentitySnapshotV1

BRANCH = "audit/flexsds-v023-hardwareadaptive-20261120"
BASE_SHA = "ac395ed035374ea3937e79b4cea077feb59bfa1a"
PRODUCTS = ("NVIDIA A10", "NVIDIA A10G")
CELLS = tuple(dict(id=f"old0_q{q}", old_width=0, future_k=q, shared_k=q,
                   continuation_k=0, horizon=2) for q in (0, 2, 3))
SLOTS = tuple(f"{i:03d}" for i in range(50))
PLAN_SCHEMA = "secp-pulse-2r-campaign-plan-v1"
STATE_SCHEMA = "secp-pulse-2r-campaign-state-v1"
RAW_FILES = ("slot-plan.json", "runtime-identity.json", "decision-state.json",
             "pulse-receipt.json", "retirement-1.json", "continuation-receipt.json",
             "retirement-2.json", "cycle-finalization.json", "raw-record.json", "worker-log.txt")
ROOT = Path(__file__).resolve().parents[1]


class CampaignError(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise CampaignError(reason)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key:" + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(CampaignError("nonfinite_json")))


def safe(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe_path")
    target = root / relative
    require(target.resolve().is_relative_to(root), "path_escape")
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        require(not cursor.is_symlink(), "symlink_path")
    return target


def atomic_json(path, value, before_replace=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if before_replace:
            before_replace()
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def writer(root):
    path = safe(root, "campaign-state.lock")
    # Crash leaves a conservative owner marker. Never automatically steal a lock.
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise CampaignError("single_writer_violation") from exc
    owner = dict(pid=os.getpid(), owner=uuid.uuid4().hex)
    try:
        with stream:
            stream.write(canonical(owner))
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        require(read_json(path) == owner, "lock_owner_changed")
        path.unlink()


def git(root, *args):
    return subprocess.check_output(["git", "-c", "safe.directory=" + Path(root).resolve().as_posix(),
                                    *args], cwd=root).decode("utf-8").strip()


def source_gate(root, sha):
    require(git(root, "branch", "--show-current") == BRANCH, "branch_mismatch")
    require(git(root, "rev-parse", "HEAD") == sha, "source_sha_mismatch")
    require(not git(root, "diff", "--name-only", "HEAD"), "dirty_tracked_source")
    require(not git(root, "status", "--porcelain", "--untracked-files=all", "--",
                    "src", "experiments", "tests", "pyproject.toml"), "unfrozen_execution_source")


def validate_plan(plan):
    require(plan["schema_version"] == PLAN_SCHEMA and plan["objective"] == PULSE_OBJECTIVE, "plan_schema_objective")
    require(plan["products"] == list(PRODUCTS) and canonical(plan["cells"]) == canonical(list(CELLS)), "plan_products_cells")
    require(re.fullmatch("[0-9a-f]{40}", plan["source_git_sha"]) is not None, "plan_source")
    for key, value in dict(cohort_slots_per_cell=50, bootstrap_resamples=5000,
                           confidence_level=.95, bootstrap_seed=731,
                           failure_policy="fixed-slot-no-replacement-v1").items():
        require(type(plan[key]) is type(value) and plan[key] == value, "plan_constant:" + key)
    require(type(plan["max_allocations"]) is int and plan["max_allocations"] > 0, "allocation_bound")
    require(plan["modal"] == dict(requested_gpu_selector="A10G", max_containers=1,
                                 retries=0, timeout_seconds=3600, startup_timeout_seconds=None), "modal_bounds")
    require(type(plan["synthetic"]) is bool, "synthetic_flag")
    contract = read_json(ROOT / "experiments/eagle_pulse_campaign_contract.json")
    require(canonical(plan["identities"]) == canonical(contract["identities"])
            and canonical(plan["engine_args"]) == canonical(contract["engine_args"]), "frozen_runtime_family")
    require(set(plan["identities"]) == set(PRODUCTS), "identity_products")
    for product in PRODUCTS:
        identity = RuntimeIdentitySnapshotV1(**plan["identities"][product])
        validate_identity(identity)
        require(dict(identity.observed)["gpu_name"] == product, "identity_product")
    require(plan["runtime_contract_hash"] == digest(plan["identities"]), "runtime_contract_hash")
    domain = plan["domain"]
    for key, field in (("preparation_contract_hash", "preparation"), ("queue_contract_hash", "queue"),
                       ("continuation_contract_hash", "continuation")):
        require(plan[key] == digest(domain[field]), key)
    require(domain["queue"]["continuation"] == 0 and domain["continuation"] == 0, "domain_continuation")
    from experiments.eagle_pulse_campaign_fake import make_plan
    require(canonical(domain) == canonical(make_plan(plan["source_git_sha"], plan["max_allocations"], plan["synthetic"])["domain"]), "frozen_preparation_domain")
    return plan


def new_state(plan, campaign_id):
    require(re.fullmatch(r"[A-Za-z0-9_-]+", campaign_id) is not None, "campaign_id")
    def product():
        return dict(status="MISSING", cells={cell["id"]: dict(status="PENDING", slots={
            slot: dict(status="UNATTEMPTED", attempt_id=None, raw_manifest=None,
                       raw_sha256=None, failure_code=None) for slot in SLOTS}) for cell in CELLS},
            profile_path=None, profile_sha256=None)
    return dict(schema_version=STATE_SCHEMA, campaign_id=campaign_id, plan_sha256=digest(plan),
                source_git_sha=plan["source_git_sha"], allocation_count=0, status="READY",
                allocations=[], products={name: product() for name in PRODUCTS})


def initialize(root, plan, campaign_id):
    validate_plan(plan)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    require(not any(root.iterdir()), "campaign_directory_not_empty")
    atomic_json(safe(root, "campaign-plan.json"), plan)
    safe(root, "campaign-plan.sha256").write_text(digest(plan) + "\n", encoding="utf-8")
    atomic_json(safe(root, "campaign-state.json"), new_state(plan, campaign_id))


def slot_path(product, cell, slot):
    require(product in PRODUCTS and cell in [c["id"] for c in CELLS] and slot in SLOTS, "slot_domain")
    return f"raw/{product.removeprefix('NVIDIA ')}/{cell}/cohort-{slot}"


def next_slot(state, product):
    branch = state["products"][product]
    if branch["status"] in ("COMPLETE", "PUBLICATION_BLOCKED"):
        return None
    for cell in CELLS:
        for slot in SLOTS:
            if branch["cells"][cell["id"]]["slots"][slot]["status"] == "UNATTEMPTED":
                return cell["id"], slot
    return None


def manifest(root, relative):
    directory = safe(root, relative)
    data = read_json(directory / "raw-manifest.json")
    require(set(data["files"]) == {p.name for p in directory.iterdir() if p.name != "raw-manifest.json"}, "raw_inventory")
    require(set(RAW_FILES) <= set(data["files"]), "raw_files_missing")
    for name, expected in data["files"].items():
        require(Path(name).name == name, "raw_filename")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "raw_file_type")
        require(sha_file(path) == expected, "raw_hash_mismatch")
    return data


def validate_cycle(row, plan, product, cell, slot, identity):
    q = next(c["future_k"] for c in CELLS if c["id"] == cell)
    require(row["cohort_id"] == f"{product.removeprefix('NVIDIA ')}:{cell}:{slot}", "cohort_binding")
    require(row["source_git_sha"] == plan["source_git_sha"] and row["campaign_plan_sha256"] == digest(plan), "cycle_source_plan")
    require(row["runtime_identity_key"] == identity.identity_key and row["domain_hash"] == digest(plan["domain"]), "cycle_identity_domain")
    require(row["synthetic"] is plan["synthetic"], "cycle_synthetic_binding")
    require(row["q"] == q and row["objective"] == PULSE_OBJECTIVE and row["unit"] == PULSE_UNIT,
            "cycle_cell_objective")
    require(row["status"] == "COMPLETE" and row["horizon"] == 2 and row["continuation"] == 0
            and row["terminal_value"] == 0, "cycle_horizon")
    require(all(type(row[k]) is int for k in ("q", "horizon", "continuation", "terminal_value", "useful_tokens")), "cycle_integer_fields")
    require(type(row["start_ns"]) is int and type(row["end_ns"]) is int
            and 0 <= row["start_ns"] < row["end_ns"]
            and row["cost_ms"] == (row["end_ns"] - row["start_ns"]) / 1e6, "cycle_timing")
    validate_pulse_cycle(row)
    # Production receipt semantics include the original controller base3.
    for receipt in (row["pulse_receipt"], row["continuation_receipt"]):
        require(type(receipt["base_k"]) is int and receipt["base_k"] == 3, "base_k_mismatch")
        require(type(receipt["worker_k"]) is int and all(type(k) is int for k in receipt["applied_k"].values()), "receipt_integer_fields")
    for retirement in row["retirements"]:
        require(type(retirement["useful_tokens"]) is int
                and all(type(v) is int for v in retirement["physical"].values()), "retirement_integer_fields")
    return row


def validate(root, *, checkout=ROOT, source=True, complete=False, lock_owned=False):
    root = Path(root).resolve()
    plan = validate_plan(read_json(safe(root, "campaign-plan.json")))
    state = read_json(safe(root, "campaign-state.json"))
    require(state["schema_version"] == STATE_SCHEMA, "state_schema")
    require(state["status"] in ("READY", "RUNNING", "BLOCKED", "COMPLETE", "MAX_ALLOCATIONS_REACHED"), "state_enum")
    require(state["plan_sha256"] == digest(plan) == safe(root, "campaign-plan.sha256").read_text().strip(), "plan_hash_mismatch")
    require(state["source_git_sha"] == plan["source_git_sha"], "state_source")
    require(set(state["products"]) == set(PRODUCTS), "state_products")
    require(lock_owned or not safe(root, "campaign-state.lock").exists(), "single_writer_violation")
    require(type(state["allocation_count"]) is int and 0 <= state["allocation_count"] <= plan["max_allocations"]
            and len(state["allocations"]) == state["allocation_count"], "allocation_accounting")
    if source:
        source_gate(checkout, plan["source_git_sha"])
        verify_source(root, plan)
    seen, attempted_paths = set(), set()
    for index, item in enumerate(state["allocations"]):
        allocation = read_json(safe(root, item["path"]))
        require(item["path"] == f"allocations/{index:04d}/allocation.json"
                and sha_file(safe(root, item["path"])) == item["sha256"], "allocation_integrity")
        require(allocation["plan_sha256"] == digest(plan), "allocation_plan")
        require(allocation["schema_version"] == "pulse-allocation-v1" and allocation["index"] == index, "allocation_schema_index")
        require(allocation["status"] in ("NO_WORKER_ADMITTED", "ADMITTED", "UNSUPPORTED_ALLOCATION")
                and allocation["released"] is True,
                "allocation_hard_invariant_requires_review")
    for product, branch in state["products"].items():
        require(branch["status"] in ("MISSING", "IN_PROGRESS", "PUBLICATION_BLOCKED", "COMPLETE"), "product_enum")
        require(set(branch["cells"]) == {c["id"] for c in CELLS}, "state_cells")
        for cell, entry in branch["cells"].items():
            require(entry["status"] in ("PENDING", "IN_PROGRESS", "PUBLICATION_BLOCKED", "COMPLETE"), "cell_enum")
            require(set(entry["slots"]) == set(SLOTS), "state_slots")
            for slot, record in entry["slots"].items():
                require(record["status"] in ("UNATTEMPTED", "RUNNING", "VALID", "FAILED"), "slot_enum")
                if record["status"] == "UNATTEMPTED":
                    require(all(record[k] is None for k in ("attempt_id", "raw_manifest", "raw_sha256", "failure_code")), "unattempted_metadata")
                    continue
                require(record["status"] != "RUNNING", "interrupted_scientific_attempt_requires_review")
                relative = slot_path(product, cell, slot)
                attempted_paths.add(relative)
                require(record["raw_manifest"] == relative + "/raw-manifest.json"
                        and sha_file(safe(root, record["raw_manifest"])) == record["raw_sha256"], "raw_manifest_link")
                data = manifest(root, relative)
                require(data["plan_sha256"] == digest(plan) and data["attempt_id"] == record["attempt_id"], "raw_attempt_plan")
                row = read_json(safe(root, relative + "/raw-record.json"))
                allocation_index = record["allocation_index"]
                require(type(allocation_index) is int and 0 <= allocation_index < state["allocation_count"], "slot_allocation_binding")
                require(data["allocation_index"] == row["allocation_index"] == allocation_index, "raw_allocation_binding")
                allocation = read_json(safe(root, state["allocations"][allocation_index]["path"]))
                require(allocation["status"] == "ADMITTED" and dict(allocation["observed_identity"]["observed"])["gpu_name"] == product, "raw_observed_product_binding")
                epoch = (row.get("request_id"), row.get("incarnation"))
                if record["status"] == "VALID":
                    require(epoch not in seen and None not in epoch, "duplicate_cohort_provenance")
                    seen.add(epoch)
                identity_artifact = read_json(safe(root, relative + "/runtime-identity.json"))
                require(identity_artifact["plan_sha256"] == digest(plan), "identity_plan")
                if record["status"] == "VALID":
                    identity = RuntimeIdentitySnapshotV1(**identity_artifact["data"])
                    require(identity.identity_key == RuntimeIdentitySnapshotV1(**plan["identities"][product]).identity_key, "raw_runtime_identity")
                    require(record["failure_code"] is None and "failure.json" not in data["files"], "valid_failure")
                    validate_cycle(row, plan, product, cell, slot, identity)
                    parts = {"decision-state.json": row["pulse_receipt"]["decision_state"],
                             "pulse-receipt.json": row["pulse_receipt"], "continuation-receipt.json": row["continuation_receipt"],
                             "retirement-1.json": row["retirements"][0], "retirement-2.json": row["retirements"][1]}
                    for name, part in parts.items():
                        expected = hashlib.sha256((canonical(dict(plan_sha256=digest(plan), data=part)) + "\n").encode()).hexdigest()
                        require(data["files"][name] == expected, "raw_split_record_binding")
                else:
                    require(record["failure_code"] and "failure.json" in data["files"], "failed_evidence")
            statuses = [s["status"] for s in entry["slots"].values()]
            expected = "PUBLICATION_BLOCKED" if "FAILED" in statuses else "COMPLETE" if statuses == ["VALID"] * 50 else "IN_PROGRESS" if "VALID" in statuses else "PENDING"
            require(entry["status"] == expected, "cell_status_inconsistent")
        cell_status = [c["status"] for c in branch["cells"].values()]
        expected = "PUBLICATION_BLOCKED" if "PUBLICATION_BLOCKED" in cell_status else "COMPLETE" if cell_status == ["COMPLETE"] * 3 else "MISSING" if cell_status == ["PENDING"] * 3 else "IN_PROGRESS"
        require(branch["status"] == expected, "product_status_inconsistent")
        if branch["status"] == "COMPLETE":
            require(branch["profile_path"] == profile_path(plan, product), "profile_path")
            require(sha_file(safe(root, branch["profile_path"])) == branch["profile_sha256"], "profile_hash")
            admit(root, plan, product, branch)
        else:
            require(branch["profile_path"] is None and branch["profile_sha256"] is None, "unexpected_profile")
    raw = safe(root, "raw")
    actual = {p.parent.relative_to(root).as_posix() for p in raw.rglob("*") if p.is_file()} if raw.exists() else set()
    require(actual == attempted_paths, "unaccounted_raw_attempt")
    statuses = [p["status"] for p in state["products"].values()]
    require((state["status"] == "COMPLETE") == (statuses == ["COMPLETE"] * 2), "campaign_complete_inconsistent")
    require((state["status"] == "BLOCKED") == ("PUBLICATION_BLOCKED" in statuses), "campaign_blocked_inconsistent")
    if complete:
        require(state["status"] == "COMPLETE", "campaign_not_complete")
    return plan, state


def profile_path(plan, product):
    prefix = "synthetic-publication" if plan["synthetic"] else "publication"
    return f"{prefix}/{product.removeprefix('NVIDIA ')}/eagle-pulse-profile.json"


def publication_cycles(cycles, synthetic):
    if not synthetic:
        return cycles
    # Isolated estimator contract exercise; never modifies synthetic raw evidence.
    copies = deepcopy(cycles)
    for row in copies:
        require(row["synthetic"] is True, "fake_record_not_synthetic")
        row["synthetic"] = False
    return copies


def cost_cell(cycles, synthetic):
    report = publish_pulse_cost(publication_cycles(cycles, synthetic))
    require(report["complete"], "publication_incomplete")
    values = tuple(report[k] for k in ("C_point", "C_low", "C_high"))
    if synthetic:
        return PulseCostCell.synthetic_fixture(*values)
    evidence = dict(kind="pulse_cycle_cohorts_v1", cycles=cycles)
    return PulseCostCell(*values, canonical(evidence), digest(evidence))


def publish(root, plan, state, product):
    branch = state["products"][product]
    identity = RuntimeIdentitySnapshotV1(**plan["identities"][product])
    cells = []
    for cell in CELLS:
        entry = branch["cells"][cell["id"]]
        require(all(s["status"] == "VALID" for s in entry["slots"].values()), "fixed_slots_incomplete")
        cycles = [read_json(safe(root, slot_path(product, cell["id"], slot) + "/raw-record.json")) for slot in SLOTS]
        for slot, row in zip(SLOTS, cycles):
            manifest(root, slot_path(product, cell["id"], slot))
            validate_cycle(row, plan, product, cell["id"], slot, identity)
        cost = cost_cell(cycles, plan["synthetic"])
        cells.append((cycles[0]["initial_key"], cost))
    profile = EaglePulseProfile(state["campaign_id"] + ":" + product, identity, canonical(plan["domain"]), tuple(cells), plan["synthetic"])
    relative = profile_path(plan, product)
    require(not safe(root, relative).exists(), "profile_already_exists")
    atomic_json(safe(root, relative), profile.to_data())
    metadata = dict(schema="pulse-campaign-publication-v1", plan_sha256=digest(plan),
                    source_git_sha=plan["source_git_sha"], synthetic=plan["synthetic"], product=product,
                    profile_sha256=sha_file(safe(root, relative)), raw_manifests={
                        cell["id"]: {slot: branch["cells"][cell["id"]]["slots"][slot]["raw_sha256"] for slot in SLOTS} for cell in CELLS},
                    bootstrap_resamples=5000, confidence_level=.95, bootstrap_seed=731,
                    failure_count=0, published_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(safe(root, str(Path(relative).with_name("publication-manifest.json"))), metadata)
    branch.update(profile_path=relative, profile_sha256=metadata["profile_sha256"], status="COMPLETE")
    admit(root, plan, product, branch)


def admit(root, plan, product, branch):
    profile = EaglePulseProfile.from_data(read_json(safe(root, branch["profile_path"])))
    identity = RuntimeIdentitySnapshotV1(**plan["identities"][product])
    require(profile.identity_status(identity) == "VALID_MATCH" and profile.synthetic is plan["synthetic"]
            and profile.domain_json == canonical(plan["domain"]), "profile_admission")
    require(sorted(json.loads(k)["candidate_shared_worker_k"] for k, _ in profile.cells) == [0, 2, 3], "profile_cell_set")
    meta = read_json(safe(root, str(Path(branch["profile_path"]).with_name("publication-manifest.json"))))
    require(meta["schema"] == "pulse-campaign-publication-v1" and meta["synthetic"] is plan["synthetic"]
            and meta["plan_sha256"] == digest(plan) and meta["profile_sha256"] == branch["profile_sha256"]
            and meta["source_git_sha"] == plan["source_git_sha"] and meta["product"] == product, "publication_binding")
    expected = {cell["id"]: {slot: branch["cells"][cell["id"]]["slots"][slot]["raw_sha256"] for slot in SLOTS} for cell in CELLS}
    require(meta["raw_manifests"] == expected and meta["failure_count"] == 0
            and meta["bootstrap_resamples"] == 5000 and meta["confidence_level"] == .95
            and meta["bootstrap_seed"] == 731, "publication_metadata")
    require(bool(meta["published_at_utc"]), "publication_timestamp")
    if not plan["synthetic"]:
        for key, cost in profile.cells:
            q = json.loads(key)["candidate_shared_worker_k"]
            actual = [read_json(safe(root, slot_path(product, f"old0_q{q}", slot) + "/raw-record.json")) for slot in SLOTS]
            require(json.loads(cost.evidence_json)["cycles"] == actual, "profile_raw_cycle_binding")
    return profile


def persist_slot(root, plan, state, product, cell, slot, result, seen_epochs=None):
    record = state["products"][product]["cells"][cell]["slots"][slot]
    require(record["status"] == "RUNNING", "slot_transition")
    relative = slot_path(product, cell, slot)
    directory = safe(root, relative)
    directory.mkdir(parents=True, exist_ok=False)
    row, failure = result["record"], result.get("failure")
    try:
        identity = RuntimeIdentitySnapshotV1(**result["identity"])
        require(identity.identity_key == RuntimeIdentitySnapshotV1(**plan["identities"][product]).identity_key, "runtime_mismatch")
        require(not failure, failure or "executor_failure")
        validate_cycle(row, plan, product, cell, slot, identity)
        epoch = (row["request_id"], row["incarnation"])
        if seen_epochs is None:
            seen_epochs = valid_epochs(root, state)
        require(epoch not in seen_epochs, "duplicate_cohort_provenance")
    except (ValueError, KeyError, TypeError) as exc:
        failure = failure or str(exc)
    artifacts = {
        "slot-plan.json": dict(plan_sha256=digest(plan), source_git_sha=plan["source_git_sha"], product=product,
                               cell=cell, slot=slot, attempt_id=record["attempt_id"], allocation_index=record["allocation_index"]),
        "runtime-identity.json": result["identity"],
        "decision-state.json": row["pulse_receipt"].get("decision_state") if isinstance(row.get("pulse_receipt"), dict) else None,
        "pulse-receipt.json": row.get("pulse_receipt"),
        "continuation-receipt.json": row.get("continuation_receipt"),
        "cycle-finalization.json": {k: row.get(k) for k in ("status", "start_ns", "end_ns", "useful_tokens", "failures")},
        "raw-record.json": row,
    }
    retirements = row.get("retirements", [])
    if not isinstance(retirements, (list, tuple)):
        retirements = []
    for index in (1, 2):
        artifacts[f"retirement-{index}.json"] = retirements[index - 1] if len(retirements) >= index else None
    if failure:
        artifacts["failure.json"] = dict(code=failure, attempt_id=record["attempt_id"], plan_sha256=digest(plan))
    for name, data in artifacts.items():
        atomic_json(directory / name, data if name in ("raw-record.json", "slot-plan.json", "failure.json")
                    else dict(plan_sha256=digest(plan), data=data))
    with (directory / "worker-log.txt").open("x", encoding="utf-8") as stream:
        stream.write("campaign_plan_sha256=" + digest(plan) + "\n" + result.get("log", ""))
        stream.flush()
        os.fsync(stream.fileno())
    raw = dict(plan_sha256=digest(plan), attempt_id=record["attempt_id"], allocation_index=record["allocation_index"],
               files={p.name: sha_file(p) for p in directory.iterdir()})
    atomic_json(directory / "raw-manifest.json", raw)
    manifest(root, relative)
    record.update(status="FAILED" if failure else "VALID", failure_code=failure,
                  raw_manifest=relative + "/raw-manifest.json", raw_sha256=sha_file(directory / "raw-manifest.json"))
    entry = state["products"][product]["cells"][cell]
    branch = state["products"][product]
    if failure:
        entry["status"] = branch["status"] = "PUBLICATION_BLOCKED"
        state["status"] = "BLOCKED"
    else:
        seen_epochs.add((row["request_id"], row["incarnation"]))
        entry["status"] = "COMPLETE" if all(s["status"] == "VALID" for s in entry["slots"].values()) else "IN_PROGRESS"
        branch["status"] = "IN_PROGRESS"
        if all(c["status"] == "COMPLETE" for c in branch["cells"].values()):
            publish(root, plan, state, product)
    if all(p["status"] == "COMPLETE" for p in state["products"].values()):
        state["status"] = "COMPLETE"
    atomic_json(safe(root, "campaign-state.json"), state)


def valid_epochs(root, state):
    result = set()
    for branch in state["products"].values():
        for cell in branch["cells"].values():
            for slot in cell["slots"].values():
                if slot["status"] == "VALID":
                    row = read_json(safe(root, str(Path(slot["raw_manifest"]).with_name("raw-record.json"))))
                    result.add((row["request_id"], row["incarnation"]))
    return result


def run(root, executor, *, checkout=ROOT, source=True):
    with writer(root):
        plan, state = validate(root, checkout=checkout, source=source, lock_owned=True)
        require(executor.synthetic is plan["synthetic"], "executor_mode_mismatch")
        epochs = valid_epochs(root, state)
        while state["status"] not in ("COMPLETE", "BLOCKED") and state["allocation_count"] < plan["max_allocations"]:
            validate(root, checkout=checkout, source=source, lock_owned=True)
            index = state["allocation_count"]
            # Reserve/count before external work; crash never grants an extra allocation.
            relative = f"allocations/{index:04d}/allocation.json"
            allocation = dict(schema_version="pulse-allocation-v1", index=index,
                              plan_sha256=digest(plan), status="RESERVED", observed_identity=None, released=False)
            atomic_json(safe(root, relative), allocation)
            state["allocations"].append(dict(path=relative, sha256=sha_file(safe(root, relative))))
            state["allocation_count"] += 1
            state["status"] = "RUNNING"
            atomic_json(safe(root, "campaign-state.json"), state)
            worker = None
            try:
                worker = executor.allocate(plan, index)
                observed = executor.observe_identity(worker)
                allocation["observed_identity"] = observed
                product = dict(observed.get("observed", ())).get("gpu_name")
                allocation["status"] = "UNSUPPORTED_ALLOCATION" if product not in PRODUCTS else "ADMITTED"
                atomic_json(safe(root, relative), allocation)
                state["allocations"][-1]["sha256"] = sha_file(safe(root, relative))
                atomic_json(safe(root, "campaign-state.json"), state)
                if product in PRODUCTS:
                    identity = RuntimeIdentitySnapshotV1(**observed)
                    require(identity.identity_key == RuntimeIdentitySnapshotV1(**plan["identities"][product]).identity_key, "allocation_identity_mismatch")
                    while state["status"] not in ("COMPLETE", "BLOCKED"):
                        work = next_slot(state, product)
                        if work is None or not executor.available(worker):
                            break
                        cell, slot = work
                        record = state["products"][product]["cells"][cell]["slots"][slot]
                        require(record["status"] == "UNATTEMPTED", "slot_already_attempted")
                        record.update(status="RUNNING", attempt_id=uuid.uuid4().hex, allocation_index=index)
                        atomic_json(safe(root, "campaign-state.json"), state)
                        try:
                            result = executor.execute_slot(worker, plan, product, cell, slot, record["attempt_id"])
                            require(isinstance(result, dict) and isinstance(result.get("record"), dict)
                                    and isinstance(result.get("identity"), dict)
                                    and isinstance(result.get("log", ""), str)
                                    and (result.get("failure") is None or isinstance(result["failure"], str)), "malformed_worker_result")
                            canonical(result)
                        except Exception as error:
                            failure = f"{type(error).__name__}:{error}"
                            result = dict(identity=observed, failure=failure, log=failure,
                                record=dict(cohort_id=f"{product.removeprefix('NVIDIA ')}:{cell}:{slot}",
                                    request_id=None, incarnation=None, status="INVALID", failures=[failure],
                                    source_git_sha=plan["source_git_sha"], campaign_plan_sha256=digest(plan),
                                    synthetic=plan["synthetic"], objective=PULSE_OBJECTIVE))
                        result["record"]["allocation_index"] = index
                        persist_slot(root, plan, state, product, cell, slot, result, epochs)
            except Exception as error:
                allocation["status"] = "INVARIANT_FAILURE"
                allocation["failure_code"] = f"{type(error).__name__}:{error}"
                raise
            finally:
                try:
                    if worker is not None:
                        executor.close(worker)
                    elif allocation["status"] == "RESERVED":
                        # allocate() owns cleanup until it returns a handle.
                        allocation["status"] = "NO_WORKER_ADMITTED"
                    allocation["released"] = True
                except BaseException as error:
                    allocation["status"] = "INVARIANT_FAILURE"
                    allocation["release_failure"] = f"{type(error).__name__}:{error}"
                    raise
                finally:
                    atomic_json(safe(root, relative), allocation)
                    state["allocations"][-1]["sha256"] = sha_file(safe(root, relative))
                    atomic_json(safe(root, "campaign-state.json"), state)
        if state["status"] not in ("COMPLETE", "BLOCKED"):
            state["status"] = "MAX_ALLOCATIONS_REACHED"
            atomic_json(safe(root, "campaign-state.json"), state)
        validate(root, checkout=checkout, source=source, lock_owned=True)
        return state


def summary(plan, state):
    return dict(source_git_sha=plan["source_git_sha"], plan_sha256=digest(plan),
                status=state["status"], remaining_allocations=plan["max_allocations"] - state["allocation_count"],
                products={product: dict(status=branch["status"], next_action=next_slot(state, product),
                    cells={cell: {s: sum(r["status"] == s for r in entry["slots"].values())
                                  for s in ("UNATTEMPTED", "RUNNING", "VALID", "FAILED")}
                           for cell, entry in branch["cells"].items()}) for product, branch in state["products"].items()})


def verify_source(root, plan):
    package = read_json(safe(root, "source/package-manifest.json"))
    require(package["schema"] == "pulse-source-package-v1" and package["branch"] == BRANCH, "package_schema_branch")
    require(package["source_git_sha"] == plan["source_git_sha"], "package_source")
    require(sha_file(safe(root, "source/reviewed-source.bundle")) == package["bundle_sha256"], "bundle_hash")
    require(package["bundle_sha256"] == safe(root, "source/bundle.sha256").read_text().strip(), "bundle_hash_link")
    verdict = read_json(safe(root, "source/source-verdict.json"))
    require(verdict["source_git_sha"] == plan["source_git_sha"] and verdict["verified"] is True, "source_verdict")
    for name, expected in package["files"].items():
        require(hashlib.sha256(safe(ROOT, name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, "runtime_source_hash")
    require(package["bundle_sha256"] == plan["source_bundle_sha256"]
            and digest(package) == plan["source_package_sha256"], "frozen_source_package")
    names = git(ROOT, "ls-files", "src", "experiments", "tests", "pyproject.toml").splitlines()
    require(set(package["files"]) == {name for name in names if Path(name).suffix in (".py", ".json", ".toml")}, "source_manifest_inventory")
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("dry-run", "run", "validate", "fake"), default="dry-run")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-allocations", type=int)
    parser.add_argument("--fake-sequence", default="A10G,A10")
    args = parser.parse_args(argv)
    try:
        plan, state = validate(args.campaign_dir, complete=args.command == "validate")
        if args.max_allocations is not None:
            require(args.max_allocations == plan["max_allocations"], "frozen_allocation_bound")
        if args.command == "run" and not args.dry_run:
            require(args.resume and not plan["synthetic"], "explicit_live_resume_required")
            from experiments.eagle_pulse_campaign_modal import ModalExecutor
            state = run(args.campaign_dir, ModalExecutor(args.campaign_dir))
        elif args.command == "fake" and not args.dry_run:
            require(plan["synthetic"], "fake_requires_synthetic_plan")
            from experiments.eagle_pulse_campaign_fake import FakeExecutor
            state = run(args.campaign_dir, FakeExecutor(args.fake_sequence.split(",")))
        print(json.dumps(summary(plan, state), indent=2))
        return 0 if state["status"] != "BLOCKED" else 2
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print("SOURCE/STATE/IDENTITY INVARIANT FAILURE: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
