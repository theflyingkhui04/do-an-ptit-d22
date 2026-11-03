"""CPU-only benchmark job lifecycle tests; no Modal app/network/GPU is started."""

import ast
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from experiments import modal_lanes as lanes


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
CALL_ID = "fc-test001"
CELL = {"workload": "W1", "concurrency": 1, "repeats": 5, "num_requests": 32}


@pytest.fixture
def job(tmp_path):
    # Execute the production functions without registering Modal resources.
    tree = ast.parse((ROOT / "experiments/modal_runner.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 or isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    for node in functions:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = ModuleType("bench_jobs_under_test")
    module.__dict__.update(
        json=json, os=os, re=re, subprocess=subprocess, sys=sys, datetime=datetime, timezone=timezone,
        threading=threading,
        Path=Path, lanes=lanes, REPO_ROOT=tmp_path, RUN_ID="pilot", APP_NAME="ptit-vllm",
        SERVER_ENV_FILE="server_env.json", RAW_FILE="raw.jsonl", BENCH_ROOT=str(tmp_path / "remote"),
        modal=SimpleNamespace(FunctionCall=Mock(), Function=Mock(), current_function_call_id=lambda: CALL_ID),
        app=Mock(), artifacts_volume=Mock(), time=Mock(),
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "modal_runner.py", "exec"), module.__dict__)
    module.original_source_check = module._committed_bench_source
    module.original_bench = module.bench
    module._committed_bench_source = Mock(return_value=SHA)
    module._git_commit = Mock(return_value=SHA)
    module.app.run.return_value = MagicMock()
    module.bench = Mock()
    module.deployed = Mock()
    module.identity = Mock()
    module.identity.remote.return_value = {"commit": SHA, "run_id": "server-smoke", "lanes": ["eagle"],
                                         "keep_warm": False, "scaledown_seconds": 60}
    module.modal.Function.from_name.side_effect = lambda app, name: (
        module.identity if name == "benchmark_identity" else module.deployed)
    module.deployed.spawn.return_value.object_id = CALL_ID
    module.deployed.spawn.return_value.get.side_effect = AssertionError("submit must not wait")
    return module


def submit(job):
    return job.bench_submit(lane="eagle", server_run_id="server-smoke", bench_run_id="pilot-cell", **CELL)


def receipt(job):
    return job._bench_evidence_root() / "pilot-cell/submission.json"


def result():
    row = {"experiment_id": "pilot-cell", "lane": "eagle", "workload": "W1", "concurrency": 1,
           "commit": SHA, "success": True}
    return {
        "lane": "eagle", "server_run_id": "server-smoke", "bench_run_id": "pilot-cell", "cell": dict(CELL),
        "commit": SHA, "returncode": 0, "server_ready_wait_s": 0.1, "harness_log": "harness output\n",
        "server_env": {"lane": "eagle", "run_id": "server-smoke", "source_commit": SHA,
                       "instance_id": "test-instance"}, "join_error": None,
        "raw": json.dumps(row) + "\n", "joined": json.dumps({**row, "vram_peak_mb": 100}) + "\n",
        "timestamp_utc": "2026-11-03T03:00:00+00:00",
    }


def prepare_collect(job):
    submit(job)
    job.app.reset_mock()
    job.bench.reset_mock()
    job.deployed.reset_mock()
    job.modal.FunctionCall.from_id.return_value.get.return_value = result()


def test_submit_spawns_detached_persists_id_and_never_waits(job):
    assert submit(job) == CALL_ID
    job.app.run.assert_not_called()
    job.modal.Function.from_name.assert_any_call("ptit-vllm", "bench")
    job.deployed.spawn.assert_called_once_with("eagle", "server-smoke", "pilot-cell", CELL, SHA)
    job.bench.remote.assert_not_called()
    job.deployed.spawn.return_value.get.assert_not_called()
    job.deployed.spawn.return_value.cancel.assert_not_called()
    data = json.loads(receipt(job).read_text())
    assert data["call_id"] == CALL_ID
    assert data["lane"] == "eagle"
    assert data["server_run_id"] == "server-smoke"
    assert data["bench_run_id"] == "pilot-cell"
    assert data["cell"] == CELL
    assert data["commit"] == SHA
    assert datetime.fromisoformat(data["submitted_at_utc"]).tzinfo is not None
    assert {p.name for p in receipt(job).parent.iterdir()} == {"submission.json", "spawn_attempt.json"}


def test_pre_spawn_interrupt_is_not_submitted(job):
    job.identity.remote.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        submit(job)
    assert json.loads(receipt(job).read_text())["call_id"] is None
    assert not (receipt(job).parent / "spawn_attempt.json").exists()
    job.deployed.spawn.assert_not_called()


@pytest.mark.parametrize("call_id", [None, "", "invalid"])
def test_submit_fails_closed_if_sdk_returns_no_valid_call_id(job, call_id):
    job.deployed.spawn.return_value.object_id = call_id
    with pytest.raises(RuntimeError, match="valid FunctionCall ID"):
        submit(job)
    assert json.loads(receipt(job).read_text())["call_id"] is None
    job.deployed.spawn.return_value.cancel.assert_not_called()


def test_submit_receipt_failure_after_spawn_does_not_cancel_job(job):
    save = job._save_submission

    def fail_after_spawn(path, record):
        if record.get("call_id") is not None:
            raise OSError("local disk unavailable")
        save(path, record)

    job._save_submission = Mock(side_effect=fail_after_spawn)
    with pytest.raises(OSError, match="disk unavailable"):
        submit(job)
    job.deployed.spawn.return_value.cancel.assert_not_called()
    assert json.loads(receipt(job).read_text())["call_id"] is None
    # This ambiguous submission is reserved, not automatically submitted twice.
    with pytest.raises(FileExistsError):
        submit(job)
    assert job.deployed.spawn.call_count == 1


def test_submit_does_not_overwrite_or_resubmit_existing_cell(job):
    submit(job)
    before = receipt(job).read_bytes()
    with pytest.raises(FileExistsError):
        submit(job)
    assert receipt(job).read_bytes() == before
    assert job.deployed.spawn.call_count == 1


@pytest.mark.parametrize("changes", [
    {"server_run_id": ""}, {"server_run_id": "../escape"}, {"bench_run_id": "../escape"},
    {"lane": "missing"}, {"workload": "bad"}, {"repeats": 0}, {"num_requests": -1},
])
def test_submit_rejects_invalid_parameters_before_modal(job, changes):
    args = {"lane": "eagle", "server_run_id": "server-smoke", "bench_run_id": "pilot-cell", **CELL}
    with pytest.raises(ValueError):
        job.bench_submit(**{**args, **changes})
    job.app.run.assert_not_called()
    job.bench.spawn.assert_not_called()


def test_collect_recovers_and_writes_canonical_evidence_with_original_sha(job):
    prepare_collect(job)
    job._git_commit.return_value = "b" * 40
    out = job.bench_collect(CALL_ID)
    job.modal.FunctionCall.from_id.assert_called_once_with(CALL_ID)
    job.modal.FunctionCall.from_id.return_value.get.assert_called_once_with()
    job.app.run.assert_not_called()
    job.bench.spawn.assert_not_called()
    assert {path.name for path in out.iterdir()} == {
        "submission.json", "spawn_attempt.json", "harness.log", "server_env.json", "raw.jsonl", "raw.vram.jsonl", "bench.json"}
    assert (out / "harness.log").read_text() == result()["harness_log"]
    assert json.loads((out / "server_env.json").read_text()) == result()["server_env"]
    assert (out / "raw.jsonl").read_text() == result()["raw"]
    assert (out / "raw.vram.jsonl").read_text() == result()["joined"]
    meta = json.loads((out / "bench.json").read_text())
    assert meta["commit"] == SHA
    assert meta["cell"] == CELL
    assert meta["call_id"] == CALL_ID
    assert meta["bench_run_id"] == "pilot-cell" and meta["server_run_id"] == "server-smoke"
    job._git_commit.assert_not_called()


def test_collect_allows_unavailable_vram_join(job):
    prepare_collect(job)
    remote = result()
    remote.update(joined=None, join_error="No samples available")
    job.modal.FunctionCall.from_id.return_value.get.return_value = remote
    out = job.bench_collect(CALL_ID)
    assert not (out / "raw.vram.jsonl").exists()
    assert json.loads((out / "bench.json").read_text())["join_error"] == "No samples available"


def test_collect_surfaces_function_call_lookup_failure(job):
    prepare_collect(job)
    job.modal.FunctionCall.from_id.side_effect = LookupError("unknown FunctionCall")
    with pytest.raises(LookupError, match="unknown FunctionCall"):
        job.bench_collect(CALL_ID)
    assert {p.name for p in receipt(job).parent.iterdir()} == {"submission.json", "spawn_attempt.json"}


@pytest.mark.parametrize("failure", [RuntimeError("remote failed"), ConnectionError("laptop offline"),
                                     KeyboardInterrupt()])
def test_collect_failure_or_disconnect_never_cancels_and_can_retry(job, failure):
    prepare_collect(job)
    handle = job.modal.FunctionCall.from_id.return_value
    handle.get.side_effect = failure
    with pytest.raises(type(failure)):
        job.bench_collect(CALL_ID)
    handle.cancel.assert_not_called()
    assert {p.name for p in receipt(job).parent.iterdir()} == {"submission.json", "spawn_attempt.json"}
    handle.get.side_effect = None
    assert (job.bench_collect(CALL_ID) / "bench.json").exists()
    job.bench.spawn.assert_not_called()


@pytest.mark.parametrize("key,value", [
    ("lane", "static"), ("server_run_id", "other-server"), ("bench_run_id", "other-cell"),
    ("commit", "b" * 40), ("cell", {**CELL, "repeats": 1}),
    ("server_env", {"lane": "eagle", "run_id": "other-server"}),
])
def test_collect_rejects_mismatched_metadata(job, key, value):
    prepare_collect(job)
    remote = {**result(), key: value}
    job.modal.FunctionCall.from_id.return_value.get.return_value = remote
    with pytest.raises(ValueError, match="mismatch"):
        job.bench_collect(CALL_ID)
    assert {p.name for p in receipt(job).parent.iterdir()} == {"submission.json", "spawn_attempt.json"}


@pytest.mark.parametrize("key,value", [("raw", None), ("raw", ""), ("raw", "not json"),
                                       ("raw", json.dumps({"commit": "b" * 40})), ("returncode", 4)])
def test_collect_rejects_failed_harness_or_missing_invalid_raw(job, key, value):
    prepare_collect(job)
    job.modal.FunctionCall.from_id.return_value.get.return_value = {**result(), key: value}
    with pytest.raises((ValueError, RuntimeError)):
        job.bench_collect(CALL_ID)
    assert {p.name for p in receipt(job).parent.iterdir()} == {"submission.json", "spawn_attempt.json"}


@pytest.mark.parametrize("call_id", [None, "", "wrong", "fc-../escape", "fc-unknown"])
def test_collect_rejects_invalid_or_missing_receipt(job, call_id):
    with pytest.raises(ValueError):
        job.bench_collect(call_id)
    job.modal.FunctionCall.from_id.assert_not_called()


def test_collect_rejects_ambiguous_receipts(job):
    prepare_collect(job)
    other = receipt(job).parent.parent / "duplicate/submission.json"
    other.parent.mkdir()
    other.write_bytes(receipt(job).read_bytes())
    with pytest.raises(ValueError, match="exactly one"):
        job.bench_collect(CALL_ID)
    job.modal.FunctionCall.from_id.assert_not_called()


def test_collect_reuses_identical_evidence_without_overwriting(job):
    prepare_collect(job)
    out = job.bench_collect(CALL_ID)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in out.iterdir()}
    assert job.bench_collect(CALL_ID) == out
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in out.iterdir()}


def test_collect_refuses_conflicting_evidence_before_writing_other_files(job):
    prepare_collect(job)
    path = receipt(job).parent / "raw.jsonl"
    path.write_text("existing evidence")
    with pytest.raises(FileExistsError, match="Conflicting"):
        job.bench_collect(CALL_ID)
    assert path.read_text() == "existing evidence"
    assert not (path.parent / "harness.log").exists()


def test_synchronous_cell_preserves_parameters_and_records_launch_sha_once(job):
    job.bench.remote.return_value = result()
    job._git_commit.side_effect = [SHA, "b" * 40]
    job.bench_cell(lane="eagle", server_run_id="server-smoke", **CELL)
    job.bench.remote.assert_called_once_with("eagle", "server-smoke", "pilot-eagle-W1-c1", CELL, SHA)
    job.bench.spawn.assert_not_called()
    meta = json.loads((job._bench_evidence_root() / "pilot-eagle-W1-c1/bench.json").read_text())
    assert meta["commit"] == SHA and meta["cell"] == CELL
    job._git_commit.assert_called_once()


def test_synchronous_defaults_are_unchanged_and_existing_evidence_is_protected(job):
    job.bench.remote.return_value = result()
    job.bench_cell(server_run_id="server-smoke")
    job.bench.remote.assert_called_once_with("vanilla", "server-smoke", "pilot-vanilla-W1-c1",
                                           {"workload": "W1", "concurrency": 1,
                                            "repeats": 1, "num_requests": 8}, SHA)
    with pytest.raises(FileExistsError):
        job.bench_cell(server_run_id="server-smoke")
    assert job.bench.remote.call_count == 1


def test_python_cli_forwards_exact_pilot_parameters(job, monkeypatch):
    job.bench_submit = Mock()
    monkeypatch.setattr(sys, "argv", ["modal_runner", "bench_submit", "--lane", "eagle",
                                     "--server-run-id", "server-smoke", "--bench-run-id", "pilot-cell",
                                     "--workload", "W1", "--concurrency", "1", "--repeats", "5",
                                     "--num-requests", "32"])
    job._bench_cli()
    job.bench_submit.assert_called_once_with(lane="eagle", server_run_id="server-smoke",
                                            bench_run_id="pilot-cell", **CELL)


def test_python_collect_cli_requires_call_id(job, monkeypatch):
    job.bench_collect = Mock()
    monkeypatch.setattr(sys, "argv", ["modal_runner", "bench_collect"])
    with pytest.raises(SystemExit) as error:
        job._bench_cli()
    assert error.value.code == 2
    job.bench_collect.assert_not_called()
    monkeypatch.setattr(sys, "argv", ["modal_runner", "bench_collect", "--call-id", CALL_ID])
    job._bench_cli()
    job.bench_collect.assert_called_once_with(call_id=CALL_ID)


@pytest.mark.parametrize("dirty,untracked", [("experiments/modal_runner.py", ""), ("", "src/new.py")])
def test_source_provenance_rejects_uncommitted_uploaded_code(job, monkeypatch, dirty, untracked):
    fake = Mock()
    fake.check_output.side_effect = [SHA, dirty, untracked]
    monkeypatch.setattr(job, "subprocess", fake)
    with pytest.raises(RuntimeError, match="Commit uploaded source"):
        job.original_source_check()


def test_source_provenance_uses_committed_sha_and_excludes_historical_outputs(job, monkeypatch):
    fake = Mock()
    fake.check_output.side_effect = [SHA, "", ""]
    monkeypatch.setattr(job, "subprocess", fake)
    assert job.original_source_check() == SHA
    assert ":(exclude)experiments/runs" in fake.check_output.call_args.args[0]


def test_collect_in_a_fresh_python_process(job, tmp_path):
    prepare_collect(job)
    bundle = tmp_path / "returned-result.json"
    bundle.write_text(json.dumps(result()), encoding="utf-8")
    # A separate interpreter loads the production functions and only the receipt + ID.
    script = tmp_path / "fresh_collect.py"
    script.write_text('''from __future__ import annotations
import ast,json,os,re,sys
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
from unittest.mock import Mock
root, production, bundle = map(Path, sys.argv[1:])
tree=ast.parse(production.read_text(encoding="utf-8"))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
for node in functions: node.decorator_list=[]
handle=SimpleNamespace(get=lambda: json.loads(bundle.read_text(encoding="utf-8")))
modal=SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda call_id: handle))
lanes=SimpleNamespace(LANES={"eagle":None},BENCH_WORKLOADS={"W1":None})
ns=dict(json=json,os=os,re=re,sys=sys,Path=Path,datetime=datetime,timezone=timezone,
        REPO_ROOT=root,SERVER_ENV_FILE="server_env.json",RAW_FILE="raw.jsonl",modal=modal,lanes=lanes)
exec(compile(ast.Module(body=functions,type_ignores=[]),"modal_runner.py","exec"),ns)
ns["bench_collect"]("fc-test001")
''', encoding="utf-8")
    subprocess.run([sys.executable, str(script), str(job.REPO_ROOT),
                    str(ROOT / "experiments/modal_runner.py"), str(bundle)], check=True)
    assert (receipt(job).parent / "bench.json").exists()


def test_wrong_deployed_source_cannot_spawn(job):
    job.identity.remote.return_value["commit"] = "b" * 40
    with pytest.raises(ValueError, match="Deployed source"):
        submit(job)
    assert json.loads(receipt(job).read_text())["call_id"] is None
    job.deployed.spawn.assert_not_called()


def test_orphaned_completed_call_can_be_recovered_without_resubmission(job):
    job.identity.remote.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        submit(job)
    job.modal.FunctionCall.from_id.return_value.get.return_value = result()
    job.bench_recover("pilot-cell", CALL_ID)
    assert json.loads(receipt(job).read_text())["call_id"] == CALL_ID
    job.deployed.spawn.assert_not_called()


def test_receipt_replace_failure_retains_previous_complete_receipt(job, monkeypatch):
    submit(job)
    before = receipt(job).read_bytes()
    monkeypatch.setattr(job, "os", SimpleNamespace(fsync=os.fsync, replace=Mock(side_effect=OSError("disk"))))
    with pytest.raises(OSError):
        job._save_submission(receipt(job), {**json.loads(before), "call_id": "fc-other"})
    assert receipt(job).read_bytes() == before


@pytest.mark.parametrize("task_id,expected", [("", "SUBMITTED"), ("ta-test", "RUNNING")])
def test_status_polls_existing_call_without_resubmission(job, task_id, expected):
    from modal.exception import TimeoutError as ModalTimeoutError
    prepare_collect(job)
    handle = job.modal.FunctionCall.from_id.return_value
    handle.get.side_effect = ModalTimeoutError()
    handle.get_call_graph.return_value = [SimpleNamespace(task_id=task_id, status=SimpleNamespace(name="PENDING"))]
    assert job.bench_status("pilot-cell") == expected
    job.deployed.spawn.assert_not_called()


def test_status_network_error_is_not_a_failed_job(job):
    prepare_collect(job)
    handle = job.modal.FunctionCall.from_id.return_value
    handle.get.side_effect = ConnectionError("offline")
    handle.get_call_graph.return_value = []
    with pytest.raises(ConnectionError):
        job.bench_status("pilot-cell")


def test_expired_output_can_be_collected_from_bound_volume_archive(job):
    from modal.exception import OutputExpiredError
    prepare_collect(job)
    handle = job.modal.FunctionCall.from_id.return_value
    handle.get.side_effect = OutputExpiredError()
    job.artifacts_volume.read_file.return_value = [json.dumps({**result(), "call_id": CALL_ID}).encode()]
    assert (job.bench_collect(CALL_ID) / "bench.json").exists()
    job.deployed.spawn.assert_not_called()


def test_remote_bench_echoes_submission_identity_and_commits_artifacts(job, tmp_path, monkeypatch):
    monkeypatch.setenv("PTIT_SOURCE_COMMIT", SHA)
    job.RUN_ID = "server-smoke"
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(lanes, "ARTIFACTS_ROOT", str(tmp_path / "volume"))
    server_dir = Path(lanes.server_artifact_dir("eagle", "server-smoke"))
    server_dir.mkdir(parents=True)
    (server_dir / "server_env.json").write_text(json.dumps(result()["server_env"]))
    job._wait_server_ready = Mock(return_value=0.1)
    response = MagicMock()
    response.__enter__.return_value = SimpleNamespace(read=lambda: json.dumps({
        "data": [{"id": lanes.EAGLE_SERVED_MODEL}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen", Mock(return_value=response))
    job.sys = SimpleNamespace(path=Mock(), executable=sys.executable)
    join = SimpleNamespace(join_vram=Mock(return_value=[json.loads(result()["raw"])]),
                           load_jsonl=Mock(return_value=[]))
    monkeypatch.setitem(sys.modules, "experiments.modal_vram_join", join)

    def run(command, **kwargs):
        Path(command[command.index("--raw-out") + 1]).write_text(result()["raw"])
        kwargs["stdout"].write("log")
        assert kwargs["env"]["PTIT_GIT_COMMIT"] == SHA
        assert command[command.index("--repeats") + 1] == "5"
        assert command[command.index("--num-requests") + 1] == "32"
        return SimpleNamespace(returncode=0, stdout="log", stderr="")

    job.subprocess = SimpleNamespace(run=Mock(side_effect=run), STDOUT=subprocess.STDOUT)
    remote = job.original_bench("eagle", "server-smoke", "pilot-cell", CELL, SHA)
    for key in ("lane", "server_run_id", "bench_run_id", "cell", "commit"):
        assert remote[key] == result()[key]
    assert job.artifacts_volume.commit.call_count == 4
    with pytest.raises(FileExistsError):
        job.original_bench("eagle", "server-smoke", "pilot-cell", CELL, SHA)
    assert job.subprocess.run.call_count == 1
