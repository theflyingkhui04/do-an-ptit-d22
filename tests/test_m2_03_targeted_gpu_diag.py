"""CPU-only tests: official source fixtures, fake tensors/processes/HTTP.

No vLLM import, Linux driver, model download, server or real POST is used.
"""
import copy
import ast
from contextlib import redirect_stderr
import json
import numpy as np
import signal
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from experiments.m2_03_targeted import analysis as a, common as c, execution as e, observer as o, sources as s


@pytest.mark.parametrize("name", list(s.manifest()["sha256"]))
def test_official_source_pin_and_inference_ast_unchanged(name):
    raw = s.fixture_sources()[name].encode()
    assert c.sha(raw) == s.manifest()["sha256"][name]
    # transform() compiles AND compares AST after deleting observer-only nodes.
    patched = s.transform(name, raw)
    assert patched != raw if name in s.INSERTIONS else patched == raw
    with pytest.raises(c.Invalid, match="hash drift"):
        s.transform(name, raw + b'\n')


def test_previous_hook_not_accepted(tmp_path):
    for name, text in s.fixture_sources().items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    assert len(s.verify_installed(tmp_path)) == 5
    p = tmp_path / "v1/worker/gpu_model_runner.py"
    p.write_bytes(p.read_bytes() + b'\n# prior v7/v8 hook\n')
    with pytest.raises(c.Invalid, match="installed source mismatch"):
        s.verify_installed(tmp_path)


def test_position_mapping_ignores_optimistic_output_length():
    rows = o.map_rows(189, [247, 248, 249], [0, 1, 2], 2, [0, 1], [2])
    assert [r["generated_index"] for r in rows] == [59, 60, 61]
    assert rows[1]["role"] == "draft_verification"
    assert rows[2]["role"] == "bonus"


@pytest.mark.parametrize("positions,indices,targets", [
    ([10, 12, 13], [0, 1, 2], [0, 1]),
    ([10, 11, 12], [0, 0, 2], [0, 1]),
    ([10, 11, 12], [0, 1, 2], [1, 0]),
])
def test_bad_mapping_rejected(positions, indices, targets):
    with pytest.raises(c.Invalid):
        o.map_rows(5, positions, indices, 2, targets, [2])


@pytest.mark.parametrize("ids", [[1, -1, 2], [-2], []])
def test_invalid_padding_or_empty_output(ids):
    if not ids:
        assert o.valid_selected(ids) == []  # end() rejects empty selected output.
    else:
        with pytest.raises(c.Invalid):
            o.valid_selected(ids)


def forward_fixture(directory, lane, token, *, selected=True, ordinary=False):
    cap = directory / "capture"
    cap.mkdir(parents=True)
    ids = [1] * 70
    ids[60] = token
    prompt = [0] * 5
    if ordinary:
        positions, inputs, drafts, base = [64], [1], 0, 60
    else:
        positions, inputs, drafts, base = [63, 64, 65], [1, 1, token], 2, 59
    runner = NS(input_batch=NS(num_reqs=1, req_ids=["api-1234abcd"], has_structured_output_reqs=False),
                uses_mrope=False, enable_prompt_embeds=False,
                cache_config=NS(enable_prefix_caching=False),
                vllm_config=NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                                                 data_parallel_size=1, decode_context_parallel_size=1)),
                requests={"api-1234abcd": NS(prompt_token_ids=prompt, output_token_ids=[1] * 100 + [-1] * 8)},
                query_start_loc=NS(np=[0, len(positions)]), positions=torch.tensor(positions),
                input_ids=NS(gpu=torch.tensor(inputs)), use_async_scheduling=not ordinary)
    spec = None if ordinary else NS(num_draft_tokens=[2], logits_indices=torch.tensor([0, 1, 2]),
                                     target_logits_indices=torch.tensor([0, 1]), bonus_logits_indices=torch.tensor([2]),
                                     cu_num_sampled_tokens=torch.tensor([3]))
    raw = torch.tensor([[0., 1., 3., 2.]] * len(positions), dtype=torch.float16)
    config = dict(lane=lane, execution_sha="d" * 40, directory=str(cap))
    scheduled = NS(total_num_scheduled_tokens=len(positions),
                   num_scheduled_tokens={"api-1234abcd": len(positions)},
                   scheduled_spec_decode_tokens={"api-1234abcd": [1, token] if drafts else []})
    original = raw.clone()
    o.begin(config, runner, scheduled, raw, spec, None)
    processed = raw.float()
    o.processed("sample_scores", processed[-1:], NS(all_greedy=True))
    if drafts:
        o.processed("verify_scores", processed[:-1], NS(all_greedy=True), spec)
    step = [token] if ordinary else ([1, token, 1] if selected else [1, -1, -1])
    output = NS(sampled_token_ids=torch.tensor([step]))
    original_output = output.sampled_token_ids.clone()
    o.end(config, runner, output)
    assert torch.equal(raw, original) and torch.equal(output.sampled_token_ids, original_output)
    req = NS(request_id="api-1234abcd", _output_token_ids=ids[:base],
             num_computed_tokens=10, num_in_flight_tokens=3, num_output_placeholders=3)
    committed = o.valid_selected(step)
    o.accounting(config, "account_before", req, committed, scheduled, False)
    req._output_token_ids += committed
    if not ordinary:
        req.num_output_placeholders -= len(committed)
    o.accounting(config, "account_after", req, committed, scheduled, False)
    payload = c.frozen_request()
    payload.update(model="qwen2.5-7b-" + lane, return_token_ids=True)
    c.save(directory / "request.json", payload)
    c.claim_request(directory)
    c.save(directory / "response.json", dict(id="api", prompt_token_ids=prompt,
           choices=[dict(token_ids=ids, finish_reason="stop")], usage=dict(completion_tokens=len(ids))))
    c.save(directory / "response_meta.json", dict(status=200))
    c.save(directory / "response_saved.json", dict(sha256=c.sha((directory / "response.json").read_bytes())))
    c.save(directory / "server_exit.json", dict(code=0))
    (directory / "server.log").write_text("")
    return ids, prompt, config, runner, scheduled, raw, spec


def synthetic_shutdown(directory, code=0):
    """Arrange new lifecycle schema in SYNTHETIC fixtures, not historical data."""
    from experiments.m2_03_targeted.lifecycle import response_receipt
    identity = dict(pid=171, start_ticks="17", process_group=171, boot_id="synthetic-boot")
    def put(name, value):
        (directory / name).write_bytes(c.encoded(value))
    put("response_saved.json", dict(sha256=c.file_sha(directory / "response.json"), saved_ns=20))
    put("process.json", identity)
    put("server_started.json", dict(schema="m203.targeted.lifecycle.v1", process=identity,
        lane=directory.name, execution_sha="d" * 40, launch_sha256=c.file_sha(directory / "launch.json"), started_ns=10))
    put("stop_intent.json", dict(process=identity, signal=15, started_sha256=c.file_sha(directory / "server_started.json"),
        response_receipt=response_receipt(directory), intent_ns=30))
    put("stop_sent.json", dict(process=identity, signal=15, intent_sha256=c.file_sha(directory / "stop_intent.json"),
        send_begin_ns=40, send_completed_ns=50))
    put("server_exit.json", dict(code=code, process=identity, observed_ns=60))


def run_fixture(tmp_path, different=True, adaptive_ordinary=False):
    policy = c.load(c.ROOT / "experiments/m2_03_targeted/runtime_policy.example.json")
    # Explicit synthetic metadata, not values collected from a Kaggle session.
    policy["runtime"].update(python="3.13.0", torch="2.13.0+cu130", cuda="13.0", driver="580.178.04")
    policy["runtime_environment"] = {}
    policy["adaptive_env"].update(ADAPTIVE_ENABLE_VRAM_GUARD="0", ADAPTIVE_CONTROLLER_KWARGS="")
    for role, model in policy["models"].items():
        model.update(revision="a" * 40, snapshot="/snapshots/" + "a" * 40)
    c.save(tmp_path / "run.json", dict(schema="m203.targeted.run.v1", case=c.CASE, target=60,
           historical_checkpoint_sha=c.HISTORICAL, historical_execution_sha=c.HISTORICAL_EXECUTION,
           request_sha256=c.REQUEST_HASH, diagnostic_execution_sha="d" * 40, policy=policy))
    forward_fixture(tmp_path / "vanilla", "vanilla", 2, ordinary=True)
    forward_fixture(tmp_path / "adaptive", "adaptive", 3 if different else 2,
                    ordinary=adaptive_ordinary)
    launch = {}
    for lane in c.LANES:
        c.save(tmp_path / lane / "observer.json", {
            "lane": lane, "directory": str(tmp_path / lane / "capture"),
            "execution_sha": "d" * 40,
            "diagnostic_control": policy.get("diagnostic_control", {}),
        })
        argv, env = e.launch_args(tmp_path, lane)
        launch[lane] = argv
        recorded = {k: env[k] for k in env if k.startswith(("ADAPTIVE_", "M203_TARGETED_"))
                    or k in ("VLLM_USE_V2_MODEL_RUNNER", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
        c.save(tmp_path / lane / "launch.json", dict(argv=argv, execution_sha="d" * 40, environment=recorded))
        synthetic_shutdown(tmp_path / lane)
    synthetic_runtime = dict(policy["runtime"], vllm_root="/synthetic/not-installed")
    c.save(tmp_path / "preflight.json", dict(runtime=synthetic_runtime, models=policy["models"],
           execution_sha="d" * 40, source_base_sha256=s.manifest()["sha256"], launcher_argv=launch))
    c.save(tmp_path / "installed.json", dict(runtime=synthetic_runtime, patched_sha256={
        k: c.sha(s.transform(k, v.encode())) for k, v in s.fixture_sources().items() if k in s.INSERTIONS}))
    return tmp_path


@pytest.mark.parametrize("different,state", [(True, "REPRODUCED"), (False, "NOT_REPRODUCED")])
def test_two_lane_capture_states_and_read_only_tensors(tmp_path, different, state):
    root = run_fixture(tmp_path, different)
    report = a.compare(root)
    assert report["state"] == state, report
    assert report["same_kv_state_proven"] is False
    assert report["same_internal_state_proven"] is False
    _, prompt, ids, _ = a.response(root / 'vanilla', 'vanilla')
    assert a.capture(root / 'vanilla', 'vanilla', 'd'*40, 'api', prompt, ids)['selection_role'] == 'ordinary'


def test_piecewise_control_changes_only_vanilla_target_launch(tmp_path):
    root = run_fixture(tmp_path)
    run = c.load(root / "run.json")
    run["policy"]["diagnostic_control"] = {
        "vanilla_cudagraph_mode": "PIECEWISE",
    }
    (root / "run.json").write_bytes(c.encoded(run))
    for lane in c.LANES:
        observer = c.load(root / lane / "observer.json")
        observer["diagnostic_control"] = run["policy"]["diagnostic_control"]
        (root / lane / "observer.json").write_bytes(c.encoded(observer))
    vanilla, _ = e.launch_args(root, "vanilla")
    adaptive, _ = e.launch_args(root, "adaptive")
    assert c.decode(vanilla[vanilla.index("--compilation-config") + 1]) == {
        "cudagraph_mode": "PIECEWISE",
    }
    assert "--compilation-config" not in adaptive
    assert "--speculative-config" in adaptive

    launchers = {"vanilla": vanilla, "adaptive": adaptive}
    for lane, argv in launchers.items():
        launch = c.load(root / lane / "launch.json")
        launch["argv"] = argv
        (root / lane / "launch.json").write_bytes(c.encoded(launch))
        synthetic_shutdown(root / lane)
    preflight = c.load(root / "preflight.json")
    preflight["launcher_argv"] = launchers
    (root / "preflight.json").write_bytes(c.encoded(preflight))

    report = a.compare(root)
    assert report["diagnostic_control"] == {
        "vanilla_cudagraph_mode": "PIECEWISE",
    }


def test_adaptive_kv_recompute_refills_prefix_then_restores_decode_budget(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    request = NS(request_id="api-1234abcd", num_prompt_tokens=5,
                 _output_token_ids=[2] * 60, spec_token_ids=[3, 4],
                 num_computed_tokens=67)

    class Scheduler:
        running = [request]
        requests = {request.request_id: request}
        max_num_scheduled_tokens = 1024
        scheduler_config = NS(max_num_batched_tokens=1024)

        def reset_prefix_cache(self, reset_running_requests=False):
            assert reset_running_requests is True
            self.running = []
            request.num_computed_tokens = 0
            request.spec_token_ids = []
            return True

    scheduler = Scheduler()
    config = {"lane": "adaptive", "directory": str(capture),
              "diagnostic_control": {"adaptive_kv_recompute_output_count": 60}}
    o.kv_recompute_before_schedule(config, scheduler)
    assert scheduler.max_num_scheduled_tokens == 64
    assert request.num_computed_tokens == 0 and request.spec_token_ids == []

    # Simulate the one controlled chunk. The next schedule restores the normal
    # budget and leaves exactly one conditioning token for ordinary decoding.
    request.num_computed_tokens = 64
    scheduler.running = [request]
    o.kv_recompute_before_schedule(config, scheduler)
    assert scheduler.max_num_scheduled_tokens == 1024
    events = {e["kind"]: e for e in
              (c.load(p) for p in (capture / "events").glob("*.json"))}
    assert set(events) == {"kv_recompute_started", "kv_recompute_ready"}
    assert events["kv_recompute_ready"]["recomputed_tokens"] == 64
    assert events["kv_recompute_ready"]["budget_restored"] is True


def test_kv_recompute_hook_runs_before_scheduler_step_mutation():
    patched = s.transform("v1/core/sched/scheduler.py",
                          s.fixture_sources()["v1/core/sched/scheduler.py"].encode()).decode()
    node = next(n for n in ast.walk(ast.parse(patched))
                if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_m203_control"]
    increments = [n for n in ast.walk(node) if isinstance(n, ast.AugAssign)
                  and isinstance(n.target, ast.Attribute)
                  and n.target.attr == "current_step"]
    assert len(calls) == len(increments) == 1
    assert calls[0].lineno < increments[0].lineno


def test_kv_recompute_evidence_is_required_and_preserves_boundary_shape(tmp_path):
    root = run_fixture(tmp_path, adaptive_ordinary=True)
    run = c.load(root / "run.json")
    run["policy"]["diagnostic_control"] = {
        "adaptive_kv_recompute_output_count": 60,
    }
    (root / "run.json").write_bytes(c.encoded(run))
    for lane in c.LANES:
        observer = c.load(root / lane / "observer.json")
        observer["diagnostic_control"] = run["policy"]["diagnostic_control"]
        (root / lane / "observer.json").write_bytes(c.encoded(observer))

    missing = a.compare(root)
    assert missing["state"] == "INVALID_CAPTURE"
    assert "KV recompute control missing" in missing["reason"]

    c.event(root / "adaptive/capture", "kv_recompute_ready",
            request_id="api-1234abcd", output_count=60,
            total_tokens=65, recomputed_tokens=64,
            num_computed_tokens=64, budget_restored=True)
    report = a.compare(root)
    assert report["state"] == "REPRODUCED", report
    assert report["captures"]["adaptive"]["decoding_path"] == "ordinary"
    assert report["captures"]["adaptive"]["active_query_token_count"] == 1


def test_rejected_candidate_not_mistaken_for_selected_row(tmp_path):
    ids, prompt, *_ = forward_fixture(tmp_path / "adaptive", "adaptive", 2, selected=False)
    with pytest.raises(c.Invalid, match="selected boundary capture"):
        a.capture(tmp_path / "adaptive", "adaptive", "d" * 40, "api", prompt, ids)


@pytest.mark.parametrize("problem", ["scores", "mapping", "accounting", "error", "checksum", "response"])
def test_incomplete_or_corrupt_capture_fails_closed(tmp_path, problem):
    root = run_fixture(tmp_path)
    lane = root / "adaptive"
    p = next((lane / "capture").glob("*.forward.json"))
    record = c.load(p)
    if problem == "scores":
        record["scores"]["1"].pop("processed")
    elif problem == "mapping":
        record["rows"][1]["position"] += 1
    elif problem == "accounting":
        next((lane / "capture/events").glob("*.json")).unlink()
    elif problem == "error":
        c.event(lane / "capture", "observer_error", error="synthetic")
    elif problem == "checksum":
        (lane / "capture" / record["scores"]["1"]["raw"]["file"]).write_bytes(b'broken')
    else:
        (lane / "response_saved.json").unlink()
    p.write_bytes(c.encoded(record))
    assert a.compare(root)["state"] == "INVALID_CAPTURE"


def test_observer_exception_suppressed_but_capture_invalidated(tmp_path, monkeypatch):
    cfg = tmp_path / "observer.json"
    c.save(cfg, dict(directory=str(tmp_path / "capture"), lane="adaptive", execution_sha="d" * 40))
    monkeypatch.setenv("M203_TARGETED_CONFIG", str(cfg))
    o.observe("begin", object(), None, None, None, None)
    events = [c.load(p) for p in (tmp_path / "capture/events").glob("*.json")]
    assert len(events) == 1 and events[0]["kind"] == "observer_error"


def test_boundary_not_covered(tmp_path):
    root = run_fixture(tmp_path)
    p = root / "adaptive/response.json"
    body = c.load(p)
    body["choices"][0]["token_ids"][55] = 0
    p.write_bytes(c.encoded(body))
    synthetic_shutdown(root / "adaptive")
    assert a.compare(root)["state"] == "BOUNDARY_NOT_COVERED"


def test_one_post_failure_spends_budget_and_never_retries(tmp_path, monkeypatch):
    root = tmp_path / "run"
    lane = root / "vanilla"
    lane.mkdir(parents=True)
    identity = dict(pid=99, start_ticks="1", process_group=99, boot_id="test-boot")
    c.save(lane / "process.json", identity)
    c.save(root / "run.json", dict(diagnostic_execution_sha="d" * 40))
    c.save(root / 'installed.json', dict(runtime=dict(vllm_root='/synthetic/not-installed')))
    monkeypatch.setattr(s, 'verify_installed', lambda *args, **kwargs: {})
    c.save(lane / "launch.json", dict(execution_sha="d" * 40))
    c.save(lane / "server_started.json", dict(schema="m203.targeted.lifecycle.v1", process=identity,
        lane="vanilla", execution_sha="d" * 40, launch_sha256=c.file_sha(lane / "launch.json"), started_ns=1))
    monkeypatch.setattr(e, "verify_execution", lambda x: None)
    monkeypatch.setattr(e, "process_identity", lambda pid: identity)
    calls = []

    def fake_http(port, method, path, body=None):
        calls.append(method)
        if method == "POST":
            assert (lane / "request.claim").is_file()
            raise TimeoutError("outcome UNKNOWN")
        return 200, c.encoded(dict(data=[dict(id="qwen2.5-7b-vanilla")]))

    monkeypatch.setattr(e, "http_exchange", fake_http)
    with pytest.raises(TimeoutError):
        e.request(root, "vanilla")
    with pytest.raises(c.Invalid, match="budget spent"):
        e.request(root, "vanilla")
    assert calls == ["GET", "GET", "POST"]


def test_empty_claim_still_blocks_resend(tmp_path):
    (tmp_path / "request.claim").touch()
    with pytest.raises(FileExistsError):
        c.claim_request(tmp_path)


def test_snapshot_copies_live_bytes_and_detects_tamper(tmp_path):
    root = tmp_path / "run"
    c.write_new(root / "server.log", b'partial\n')
    dst = c.checkpoint(root, "partial")
    (root / "server.log").write_bytes(b'partial\nmore data\n')
    assert c.verify_checkpoint(dst)["partial"] is True
    assert (dst / "data/server.log").read_bytes() == b'partial\n'
    (dst / "data/server.log").write_bytes(b'tampered')
    with pytest.raises(c.Invalid, match="integrity"):
        c.verify_checkpoint(dst)


def test_no_clobber_and_output_isolation(tmp_path):
    c.save(tmp_path / "one.json", dict(x=1))
    with pytest.raises(FileExistsError):
        c.save(tmp_path / "one.json", dict(x=2))
    with pytest.raises(c.Invalid):
        c.isolated(c.ROOT / "experiments/runs/new")
    assert c.load(tmp_path / "one.json") == dict(x=1)


def test_unreviewed_model_policy_and_source_checkout_fail(tmp_path):
    policy = c.load(c.ROOT / "experiments/m2_03_targeted/runtime_policy.example.json")
    with pytest.raises(c.Invalid, match="unreviewed"):
        e.model_inventory(policy)
    with pytest.raises(c.Invalid, match="wrong execution checkout"):
        e.verify_execution("0" * 40)


def test_export_verifies_snapshot_and_preserves_bytes(tmp_path):
    c.write_new(tmp_path / "run/raw.bin", b'\x00\r\n\xff')
    dst = c.checkpoint(tmp_path / "run", "partial")
    archive = e.export_snapshot(dst)
    assert c.load(archive.with_suffix(archive.suffix + ".sha256.json"))["sha256"] == c.file_sha(archive)
    with pytest.raises(FileExistsError):
        e.export_snapshot(dst)


@pytest.mark.parametrize("file,field,value", [
    ("run.json", "diagnostic_execution_sha", c.HISTORICAL_EXECUTION),
    ("preflight.json", "source_base_sha256", {}),
    ("installed.json", "patched_sha256", {}),
    ("adaptive/launch.json", "execution_sha", "0" * 40),
])
def test_capture_provenance_is_required(tmp_path, file, field, value):
    root = run_fixture(tmp_path)
    p = root / file
    record = c.load(p)
    record[field] = value
    p.write_bytes(c.encoded(record))
    assert a.compare(root)["state"] == "INVALID_CAPTURE"


def test_install_does_not_resume_partial_patch(tmp_path, monkeypatch):
    c.save(tmp_path / "run.json", dict(diagnostic_execution_sha="d" * 40))
    c.save(tmp_path / "preflight.json", dict(runtime={"vllm_root": str(tmp_path / "vllm")}))
    c.write_new(tmp_path / "install.intent", b'partial')
    monkeypatch.setattr(e, "verify_execution", lambda _: None)
    monkeypatch.setattr(e, "runtime", lambda: {"vllm_root": str(tmp_path / "vllm")})
    monkeypatch.setattr(s, "verify_installed", lambda *a, **k: {})
    with pytest.raises(FileExistsError):
        e.install(tmp_path)
    assert not (tmp_path / "installed.json").exists()


def test_occupied_port_fail_closed(monkeypatch):
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def connect_ex(self, addr): return 0
    monkeypatch.setattr(e.socket, "socket", FakeSocket)
    with pytest.raises(c.Invalid, match="occupied"):
        e.free_port(8001)


def test_import_git_bundle_and_checksum_guard(tmp_path, monkeypatch):
    c.write_new(tmp_path / "run/raw.bin", b'\x00\r\n\xff')
    snap = c.checkpoint(tmp_path / "run", "partial")
    archive = e.export_snapshot(snap)
    raw = archive.read_bytes()
    desc = c.encoded(dict(size=len(raw), sha256=c.sha(raw)))
    monkeypatch.setattr(e, "blob", lambda path, commit: raw if path == "bundle" else desc)
    dst = e.import_checkpoint("e" * 40, "bundle", "descriptor", tmp_path / "unpacked")
    assert (dst / "data/raw.bin").read_bytes() == b'\x00\r\n\xff'
    desc = c.encoded(dict(size=len(raw), sha256="0" * 64))
    with pytest.raises(c.Invalid, match="checksum"):
        e.import_checkpoint("e" * 40, "bundle", "descriptor", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_score_snapshot_retains_raw_before_inplace_processor(tmp_path):
    tensor = torch.tensor([0., 2., 1., float('-inf')], dtype=torch.float32)
    desc = o.snapshot(tensor, tmp_path, "raw.f32")
    tensor.add_(10)  # Simulate existing upstream processor, not observer mutation.
    values = a.vector(tmp_path, desc)
    assert list(values[:3]) == [0., 2., 1.]
    assert desc["top1_id"] == 1 and desc["top2_id"] == 2 and desc["margin"] == 1
    assert desc["masked_count"] == 1


@pytest.mark.parametrize("problem", ["version", "ports", "model", "source", "environment"])
def test_preflight_errors_prevent_install_and_requests(tmp_path, monkeypatch, problem):
    policy = c.load(c.ROOT / "experiments/m2_03_targeted/runtime_policy.example.json")
    policy["runtime"].update(python="3.13.0", torch="2.13.0+cu130", cuda="13.0", driver="580.178.04")
    policy["runtime_environment"] = {}
    policy["adaptive_env"].update(ADAPTIVE_ENABLE_VRAM_GUARD="0", ADAPTIVE_CONTROLLER_KWARGS="")
    for model in policy["models"].values():
        model.update(revision="a" * 40, snapshot="/snapshots/" + "a" * 40)
    runtime = dict(policy["runtime"], environment={}, vllm_root=str(tmp_path / "vllm"))
    c.save(tmp_path / "run.json", dict(diagnostic_execution_sha="d" * 40, policy=policy,
           policy_path="approved.json", policy_sha256=c.sha(c.encoded(policy)),
           case=c.CASE, request_sha256=c.REQUEST_HASH))
    monkeypatch.setattr(e, "verify_execution", lambda _: None)
    monkeypatch.setattr(e, "blob", lambda *args: c.encoded(policy))
    monkeypatch.setattr(e, "runtime", lambda: runtime)
    def fail(reason): raise c.Invalid(reason)
    monkeypatch.setattr(e, "free_port", lambda _: fail("port") if problem == "ports" else None)
    monkeypatch.setattr(e, "model_inventory", lambda _: fail("model") if problem == "model" else {})
    monkeypatch.setattr(s, "verify_installed", lambda _: fail("source") if problem == "source" else {})
    if problem == "version": runtime["vllm"] = "wrong"
    if problem == "environment": runtime["environment"] = {"VLLM_USE_V2_MODEL_RUNNER": "1"}
    expected_error = {"version": "runtime differs", "ports": "port", "model": "model", "source": "source", "environment": "runtime environment"}[problem]
    with pytest.raises(c.Invalid, match=expected_error):
        e.preflight(tmp_path)
    assert not (tmp_path / "preflight.json").exists()
    assert not (tmp_path / "install.intent").exists()
    assert not list(tmp_path.glob("*/request.claim"))


def test_actual_model_inventory_and_drift(tmp_path):
    policy = {"models": {}}
    for role, name in e.MODELS.items():
        root = tmp_path / role / ("a" * 40)
        root.mkdir(parents=True)
        for filename in ("config.json", "tokenizer_config.json", "weights.safetensors"):
            c.write_new(root / filename, b'synthetic fixture')
        policy["models"][role] = e.inspect_model(role, root)
    assert e.model_inventory(policy) == policy["models"]
    path = Path(policy["models"]["target"]["snapshot"]) / "weights.safetensors"
    path.write_bytes(b'changed')
    with pytest.raises(c.Invalid, match="model file hash mismatch"):
        e.model_inventory(policy)


def test_restore_validates_all_sources_before_mutation(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    for name, text in s.fixture_sources().items():
        p = package / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(s.transform(name, text.encode()))
        if name in s.INSERTIONS:
            c.write_new(tmp_path / "original_sources" / name, text.encode())
    c.save(tmp_path / "preflight.json", dict(runtime=dict(vllm_root=str(package))))
    monkeypatch.setattr(e, "free_port", lambda _: None)
    altered = package / "v1/sample/sampler.py"
    altered.write_bytes(altered.read_bytes() + b'\n# unknown drift\n')
    before = {p: p.read_bytes() for p in package.rglob('*.py')}
    with pytest.raises(c.Invalid, match="unknown source"):
        e.restore(tmp_path)
    assert {p: p.read_bytes() for p in before} == before


def test_first_lane_validation_blocks_continuing_after_capture_error(tmp_path):
    root = run_fixture(tmp_path)
    assert a.validate_lane(root, "vanilla")["lane_valid"] is True
    c.event(root / "vanilla/capture", "observer_error", error="synthetic")
    with pytest.raises(c.Invalid, match="observer error"):
        a.validate_lane(root, "vanilla")


def test_partial_snapshot_cannot_claim_complete_diagnostic(tmp_path, capsys):
    root = run_fixture(tmp_path / "run")
    snapshot = c.checkpoint(root, "partial")
    assert e.main(["compare", "--snapshot", str(snapshot)]) == 2
    assert json.loads(capsys.readouterr().out)["state"] == "INVALID_CAPTURE"


def test_stop_sigterm_child_minus15_is_expected_shutdown(tmp_path, monkeypatch):
    """Real stage control flow; fake child/killpg, no process or GPU launched."""
    root = tmp_path / "run"
    directory = root / "vanilla"
    directory.mkdir(parents=True)
    rt = {"vllm_root": str(tmp_path / "fake-installed")}
    c.save(root / "run.json", dict(diagnostic_execution_sha="d" * 40, policy={}))
    c.save(root / "installed.json", dict(runtime=rt))
    c.save(root / "preflight.json", dict(launcher_argv={"vanilla": ["fake-vllm"]}))
    monkeypatch.setattr(e, "verify_execution", lambda _: None)
    monkeypatch.setattr(e, "runtime", lambda: rt)
    monkeypatch.setattr(s, "verify_installed", lambda *args, **kwargs: {})
    monkeypatch.setattr(e, "model_inventory", lambda _: {})
    monkeypatch.setattr(e, "free_port", lambda _: None)
    monkeypatch.setattr(e, "launch_args", lambda *args: (["fake-vllm"], {}))
    identity = dict(pid=171, start_ticks="17", process_group=171, boot_id="test-boot")
    monkeypatch.setattr(e, "process_identity", lambda pid: identity)
    sent = []
    monkeypatch.setattr(e.os, "killpg", lambda pid, sig: sent.append((pid, sig)), raising=False)
    frozen = c.frozen_request()  # Read Git before replacing the server Popen boundary.

    class Child:
        pid = 171

        def wait(self):
            payload = copy.deepcopy(frozen)
            payload.update(model="qwen2.5-7b-vanilla", return_token_ids=True)
            c.save(directory / "request.json", payload)
            c.claim_request(directory)
            c.save(directory / "response.json", dict(id="api", prompt_token_ids=[0] * 5,
                   choices=[dict(token_ids=[1] * 70, finish_reason="stop")], usage=dict(completion_tokens=70)))
            c.save(directory / "response_meta.json", dict(status=200))
            c.save(directory / "response_saved.json", dict(sha256=c.file_sha(directory / "response.json"),
                                                           saved_ns=time.monotonic_ns()))
            e.stop(root, "vanilla")
            return -15

    monkeypatch.setattr(e.subprocess, "Popen", lambda *args, **kwargs: Child())
    e.serve(root, "vanilla")
    assert sent == [(171, signal.SIGTERM)]
    assert c.load(directory / "server_exit.json")["code"] == -15
    from experiments.m2_03_targeted.lifecycle import verify_shutdown
    assert verify_shutdown(directory)["classification"] == "EXPECTED_STOP"


def test_restore_rejects_corrupt_backup_even_for_pristine_file(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    first = next(iter(s.INSERTIONS))
    for name, text in s.fixture_sources().items():
        p = package / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(s.transform(name, text.encode()) if name == first else text.encode())
        if name in s.INSERTIONS:
            c.write_new(tmp_path / "original_sources" / name, text.encode())
    c.save(tmp_path / "preflight.json", dict(runtime=dict(vllm_root=str(package))))
    monkeypatch.setattr(e, "free_port", lambda _: None)
    (tmp_path / "original_sources/v1/sample/sampler.py").write_bytes(b'corrupt backup')
    before = {p: p.read_bytes() for p in package.rglob('*.py')}
    with pytest.raises(c.Invalid, match="backup mismatch"):
        e.restore(tmp_path)
    assert {p: p.read_bytes() for p in before} == before


def test_observer_error_event_write_failure_still_invalidates_complete_capture(tmp_path, monkeypatch):
    root = run_fixture(tmp_path)
    assert a.compare(root)["state"] == "REPRODUCED"
    cfg = root / "observer-failure.json"
    c.save(cfg, dict(directory=str(root / "adaptive/capture"), lane="adaptive", execution_sha="d" * 40))
    monkeypatch.setenv("M203_TARGETED_CONFIG", str(cfg))
    def cannot_write_event(*args, **kwargs):
        raise OSError("synthetic event-write failure")
    monkeypatch.setattr(o, "event", cannot_write_event)
    with (root / "adaptive/server.log").open("a") as log, redirect_stderr(log):
        o.observe("begin", object(), None, None, None, None)
    assert not any(c.load(p)["kind"] == "observer_error" for p in (root / "adaptive/capture/events").glob("*.json"))
    report = a.compare(root)
    assert report["state"] == "INVALID_CAPTURE"
    assert "observer error" in report["reason"]


@pytest.mark.parametrize("code", [0, -15])
def test_expected_shutdown_shared_by_validation_and_adaptive_gate(tmp_path, monkeypatch, code):
    root = run_fixture(tmp_path)
    synthetic_shutdown(root / "vanilla", code)
    validated = a.validate_lane(root, "vanilla")
    assert validated["shutdown"]["exit_code"] == code
    c.save(root / "vanilla/validated.json", validated)
    e.require_previous_lane(root)
    # The actual adaptive serve() path must reject changed lifecycle before Popen.
    exit_path = root / "vanilla/server_exit.json"
    exited = c.load(exit_path)
    exited["code"] = -9
    exit_path.write_bytes(c.encoded(exited))
    monkeypatch.setattr(e, "verify_execution", lambda _: None)
    monkeypatch.setattr(e, "runtime", lambda: c.load(root / "installed.json")["runtime"])
    monkeypatch.setattr(s, "verify_installed", lambda *args, **kwargs: {})
    monkeypatch.setattr(e, "model_inventory", lambda _: {})
    monkeypatch.setattr(e, "free_port", lambda _: None)
    spawned = []
    monkeypatch.setattr(e.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))
    with pytest.raises(c.Invalid, match="unexpected crash"):
        e.serve(root, "adaptive")
    assert spawned == []
    with pytest.raises(c.Invalid, match="unexpected crash"):
        a.validate_lane(root, "vanilla")
    assert a.compare(root)["state"] == "INVALID_CAPTURE"


@pytest.mark.parametrize("problem", ["no_intent", "no_ack", "wrong_pid", "wrong_boot", "too_early",
                                   "response_drift", "signal", "interrupted", "crash", "exit_zero_no_intent"])
def test_shutdown_is_not_blanket_nonzero_acceptance(tmp_path, problem):
    from experiments.m2_03_targeted.lifecycle import verify_shutdown
    root = run_fixture(tmp_path)
    d = root / "vanilla"
    synthetic_shutdown(d, -15)
    if problem in ("no_intent", "exit_zero_no_intent"):
        (d / "stop_intent.json").unlink()
    elif problem == "no_ack":
        (d / "stop_sent.json").unlink()
    elif problem == "response_drift":
        (d / "response.json").write_bytes((d / "response.json").read_bytes() + b' ')
    elif problem == "interrupted":
        c.event(d, "server_wrapper_interrupted")
    else:
        p = d / "server_exit.json"
        data = c.load(p)
        if problem == "wrong_pid": data["process"]["pid"] = 999
        if problem == "wrong_boot": data["process"]["boot_id"] = "another-boot"
        if problem == "too_early": data["observed_ns"] = 35  # Before signal dispatch.
        if problem == "crash": data["code"] = 137
        if problem == "signal":
            p = d / "stop_sent.json"
            data = c.load(p)
            data["signal"] = 9
        p.write_bytes(c.encoded(data))
    if problem == "exit_zero_no_intent":
        p = d / "server_exit.json"
        data = c.load(p)
        data["code"] = 0
        p.write_bytes(c.encoded(data))
    with pytest.raises((c.Invalid, OSError)):
        verify_shutdown(d)
    assert a.compare(root)["state"] == "INVALID_CAPTURE"


def test_exit_observation_may_precede_stop_ack_publication(tmp_path):
    from experiments.m2_03_targeted.lifecycle import verify_shutdown
    root = run_fixture(tmp_path)
    p = root / "vanilla/server_exit.json"
    data = c.load(p)
    data["observed_ns"] = 45  # Dispatch began at 40, acknowledgement completed at 50.
    p.write_bytes(c.encoded(data))
    assert verify_shutdown(root / "vanilla")["classification"] == "EXPECTED_STOP"


def test_partial_install_recovers_exact_sources_without_resume(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    originals = {name: text.encode() for name, text in s.fixture_sources().items()}
    for name, raw in originals.items():
        p = package / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    c.save(tmp_path / "run.json", dict(diagnostic_execution_sha="d" * 40))
    runtime = dict(vllm_root=str(package))
    c.save(tmp_path / "preflight.json", dict(runtime=runtime))
    monkeypatch.setattr(e, "verify_execution", lambda _: None)
    monkeypatch.setattr(e, "runtime", lambda: runtime)
    monkeypatch.setattr(e, "free_port", lambda _: None)
    replace = e.os.replace
    calls = []
    def crash_second_replace(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("synthetic crash between pinned-file replacements")
        return replace(src, dst)
    with monkeypatch.context() as m:
        m.setattr(e.os, "replace", crash_second_replace)
        with pytest.raises(OSError, match="synthetic crash"):
            e.install(tmp_path)
    assert (tmp_path / "install.intent").exists() and not (tmp_path / "installed.json").exists()
    assert len(list((tmp_path / "original_sources").rglob('*.py'))) == 2
    for p in (tmp_path / "original_sources").rglob('*.py'):
        assert p.read_bytes() == originals[p.relative_to(tmp_path / "original_sources").as_posix()]
    before = {name: (package / name).read_bytes() for name in originals}
    with pytest.raises(c.Invalid, match="installed source mismatch"):
        e.install(tmp_path)
    assert before == {name: (package / name).read_bytes() for name in originals}
    e.restore(tmp_path)
    assert {name: (package / name).read_bytes() for name in originals} == originals
    assert len(s.verify_installed(package)) == 5


@pytest.mark.parametrize("drafts", [0, 1, 2, 4])
@pytest.mark.parametrize("bonus_at_target", [False, True])
def test_mapping_from_exact_upstream_metadata_methods(drafts, bonus_at_target):
    """Real pinned method bodies; only H2D transfer is replaced by CPU copy.

    This does NOT execute GPU runner scheduling, Triton sampling or CUDA streams.
    """
    text = s.fixture_sources()['v1/worker/gpu_model_runner.py']
    assert c.sha(text.encode()) == s.manifest()['sha256']['v1/worker/gpu_model_runner.py']
    tree = ast.parse(text)
    methods = {n.name: ast.get_source_segment(text, n) for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name in ('_get_cumsum_and_arange', '_calc_spec_decode_metadata')}
    namespace = dict(np=np, torch=torch, SpecDecodeMetadata=NS,
                     async_tensor_h2d=lambda x, device: torch.from_numpy(x.copy()))
    for body in methods.values():
        exec('from __future__ import annotations\n' + body, namespace)
    runner = NS(device='cpu', arange_np=np.arange(16, dtype=np.int32), _arange_scratch=np.zeros(16, dtype=np.int32),
                input_ids=NS(gpu=torch.arange(100, 100 + drafts + 1)))
    runner._get_cumsum_and_arange = lambda *args, **kwargs: namespace['_get_cumsum_and_arange'](runner, *args, **kwargs)
    meta = namespace['_calc_spec_decode_metadata'](runner, np.array([drafts], dtype=np.int32), np.array([drafts + 1], dtype=np.int32))
    base = 60 - drafts if bonus_at_target else 60
    positions = list(range(189 + base - 1, 189 + base + drafts))
    rows = o.map_rows(189, positions, o.ints(meta.logits_indices), drafts,
                      o.ints(meta.target_logits_indices), o.ints(meta.bonus_logits_indices))
    target = next(row for row in rows if row['generated_index'] == 60)
    assert target['row'] == (drafts if bonus_at_target else 0)
    assert target['role'] == ('bonus' if bonus_at_target or drafts == 0 else 'draft_verification')
    assert o.ints(meta.draft_token_ids) == list(range(101, 101 + drafts))


def test_runtime_template_rejected_before_init_or_inventory(tmp_path, monkeypatch):
    name = 'experiments/m2_03_targeted/runtime_policy.example.json'
    raw = (c.ROOT / name).read_bytes()
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'blob', lambda *args: raw)
    def must_not_run(): raise AssertionError('unreviewed template reached runtime/request inspection')
    monkeypatch.setattr(e, 'runtime', must_not_run)
    monkeypatch.setattr(e, 'frozen_request', must_not_run)
    with pytest.raises(c.Invalid, match='template is not executable'):
        e.init_run(tmp_path / 'new', 'd' * 40, name)
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('state', ['R', 'Z', 'X'])
def test_linux_process_identity_refuses_exited_states(monkeypatch, state):
    tail = [state, '1', '171'] + ['0'] * 16 + ['17']
    stat = '171 (python worker with spaces) ' + ' '.join(tail)
    monkeypatch.setattr(e, 'Path', lambda p: NS(read_text=lambda: 'test-boot' if p.endswith('boot_id') else stat))
    if state == 'R':
        assert e.process_identity(171) == dict(pid=171, start_ticks='17', process_group=171, boot_id='test-boot')
    else:
        with pytest.raises(c.Invalid, match='exited/zombie'):
            e.process_identity(171)


def test_explicit_abort_cleans_owned_process_but_cannot_validate_or_resend(tmp_path, monkeypatch):
    root = run_fixture(tmp_path)
    d = root / 'vanilla'
    for name in ('stop_intent.json', 'stop_sent.json', 'server_exit.json', 'response_saved.json'):
        (d / name).unlink()
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'process_identity', lambda _: c.load(d / 'process.json'))
    sent = []
    monkeypatch.setattr(e.os, 'killpg', lambda pid, sig: sent.append((pid, sig)), raising=False)
    e.abort(root, 'vanilla')
    assert sent == [(171, signal.SIGTERM)]
    with pytest.raises(c.Invalid, match='aborted'):
        a.validate_lane(root, 'vanilla')
    with pytest.raises(c.Invalid, match='aborted'):
        e.request(root, 'vanilla')


@pytest.mark.parametrize('problem', ['missing_response', 'changed_identity', 'already_exited'])
def test_stop_refuses_unsafe_intent_before_sending_signal(tmp_path, monkeypatch, problem):
    root = run_fixture(tmp_path)
    d = root / 'vanilla'
    for name in ('stop_intent.json', 'stop_sent.json'):
        (d / name).unlink()
    if problem != 'already_exited': (d / 'server_exit.json').unlink()
    if problem == 'missing_response': (d / 'response_saved.json').unlink()
    identity = c.load(d / 'process.json')
    if problem == 'changed_identity': identity['start_ticks'] = '999'
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'process_identity', lambda _: identity)
    sent = []
    monkeypatch.setattr(e.os, 'killpg', lambda *args: sent.append(args), raising=False)
    with pytest.raises((c.Invalid, OSError)):
        e.stop(root, 'vanilla')
    assert sent == [] and not (d / 'stop_intent.json').exists()


def interrupted_restore_fixture(tmp_path, monkeypatch):
    """Actual pinned file bytes, fake runtime/process/HTTP boundaries only."""
    root, package = tmp_path / 'run', tmp_path / 'vllm'
    for name, text in s.fixture_sources().items():
        p = package / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(s.transform(name, text.encode()))
        if name in s.INSERTIONS:
            c.write_new(root / 'original_sources' / name, text.encode())
    rt = dict(vllm_root=str(package))
    c.save(root / 'run.json', dict(diagnostic_execution_sha='d' * 40, policy={}))
    c.save(root / 'installed.json', dict(runtime=rt))
    c.save(root / 'preflight.json', dict(runtime=rt, launcher_argv={'vanilla': ['fake-vllm']}))
    d = root / 'vanilla'
    c.save(d / 'launch.json', dict(execution_sha='d' * 40, argv=['fake-vllm']))
    identity = dict(pid=171, start_ticks='17', process_group=171, boot_id='synthetic-boot')
    c.save(d / 'process.json', identity)
    c.save(d / 'server_started.json', dict(schema='m203.targeted.lifecycle.v1', process=identity,
           execution_sha='d' * 40, lane='vanilla', started_ns=1, launch_sha256=c.file_sha(d / 'launch.json')))
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'runtime', lambda: rt)
    monkeypatch.setattr(e, 'process_identity', lambda _: identity)
    monkeypatch.setattr(e, 'free_port', lambda _: None)
    monkeypatch.setattr(e, 'model_inventory', lambda _: {})
    replace = e.os.replace
    count = []
    def fail_after_one(src, dst):
        count.append(dst)
        if len(count) == 2: raise OSError('synthetic crash after first restored file')
        return replace(src, dst)
    with monkeypatch.context() as m:
        m.setattr(e.os, 'replace', fail_after_one)
        with pytest.raises(OSError, match='after first restored file'):
            e.restore(root)
    assert len(count) == 2
    for patched in (False, True):
        with pytest.raises(c.Invalid, match='installed source mismatch'):
            s.verify_installed(package, patched=patched)
    return root, package


def test_partial_restore_blocks_serve_and_request(tmp_path, monkeypatch):
    root, _ = interrupted_restore_fixture(tmp_path, monkeypatch)
    http_calls, spawns = [], []
    def forbidden_http(*args, **kwargs):
        http_calls.append(args)
        raise AssertionError('HTTP reached during partial restoration')
    monkeypatch.setattr(e, 'http_exchange', forbidden_http)
    monkeypatch.setattr(e.subprocess, 'Popen', lambda *args, **kwargs: spawns.append(args))
    with pytest.raises(c.Invalid): e.serve(root, 'vanilla')
    with pytest.raises(c.Invalid): e.request(root, 'vanilla')
    assert http_calls == spawns == []
    assert not (root / 'vanilla/request.claim').exists()


def test_partial_restore_has_durable_status_and_explicit_recovery(tmp_path, monkeypatch):
    root, package = interrupted_restore_fixture(tmp_path, monkeypatch)
    assert (root / 'restore_intent.json').exists()
    status = e.restoration_status(root)
    assert status['state'] == 'INCOMPLETE' and status['source_state']['mode'] == 'MIXED'
    assert status['multi_file_atomic'] is False and status['inference_allowed'] is False
    e.restore(root)  # Explicit operator recovery, not an inference retry.
    assert e.restoration_status(root)['state'] == 'RESTORED'
    assert e.restoration_status(root)['inference_allowed'] is False
    assert len(s.verify_installed(package)) == 5


def test_correct_source_row_wrong_emitted_index_is_invalid(tmp_path):
    root = run_fixture(tmp_path)
    p = next((root / 'adaptive/capture').glob('*.forward.json'))
    rec = c.load(p)
    # A valid source row at position P+60-1, but output offset 2 belongs to
    # emitted index 61 in the scheduler step starting at 59. Repeated IDs
    # otherwise make API-prefix equality look convincing.
    for j, row in enumerate(rec['rows']):
        row.update(position=62+j, generated_index=58+j)
    rec['active_query'] = [[62, 1], [63, 1], [64, 1]]
    rec['selected'] = [1, 1, 3]
    p.write_bytes(c.encoded(rec))
    report = a.compare(root)
    assert report['state'] == 'INVALID_CAPTURE', report


def test_scheduler_selected_sequence_must_match_gpu_step(tmp_path):
    root = run_fixture(tmp_path)
    p = next(p for p in (root / 'adaptive/capture/events').glob('*.json') if c.load(p)['kind'] == 'account_before')
    rec = c.load(p)
    rec['output_token_ids'][0] = 0
    p.write_bytes(c.encoded(rec))
    assert a.compare(root)['state'] == 'INVALID_CAPTURE'


def test_partial_restore_unknown_source_refuses_all_recovery_writes(tmp_path, monkeypatch):
    root, package = interrupted_restore_fixture(tmp_path, monkeypatch)
    path = package / next(iter(s.INSERTIONS))
    path.write_bytes(path.read_bytes() + b'\n# unknown external modification\n')
    before = {n: (package / n).read_bytes() for n in s.fixture_sources()}
    assert e.restoration_status(root)['state'] == 'UNKNOWN'
    with pytest.raises(c.Invalid, match='unknown source'):
        e.restore(root)
    assert before == {n: (package / n).read_bytes() for n in before}
    for lane in c.LANES:
        with pytest.raises(c.Invalid): e.serve(root, lane)
    with pytest.raises(c.Invalid): e.request(root, 'vanilla')


def test_mixed_sources_block_inference_even_without_restore_marker(tmp_path, monkeypatch):
    root, _ = interrupted_restore_fixture(tmp_path, monkeypatch)
    (root / 'restore_intent.json').unlink()  # Simulate lost journal; source hashes still guard.
    assert e.restoration_status(root)['state'] == 'UNTRACKED_MIXED'
    for lane in c.LANES:
        with pytest.raises(c.Invalid, match='installed source mismatch'): e.serve(root, lane)
    with pytest.raises(c.Invalid, match='installed source mismatch'): e.request(root, 'vanilla')
    assert not (root / 'vanilla/request.claim').exists()


def test_restore_crash_after_last_replacement_still_requires_explicit_recovery(tmp_path, monkeypatch):
    root, package = interrupted_restore_fixture(tmp_path, monkeypatch)
    save = e.save
    def fail_completion(path, value):
        if path.name == 'restore_complete.json': raise OSError('completion not published')
        return save(path, value)
    with monkeypatch.context() as m:
        m.setattr(e, 'save', fail_completion)
        with pytest.raises(OSError, match='completion not published'): e.restore(root)
    status = e.restoration_status(root)
    assert status['state'] == 'INCOMPLETE' and status['source_state']['mode'] == 'PRISTINE'
    assert not status['inference_allowed']
    for lane in c.LANES:
        with pytest.raises(c.Invalid): e.serve(root, lane)
    e.restore(root)
    assert e.restoration_status(root)['state'] == 'RESTORED'
    assert len(s.verify_installed(package)) == 5
    with pytest.raises(c.Invalid): e.request(root, 'vanilla')


def alter_synthetic_step(root, *, base, selected, inputs, drafts):
    """Arrange alternate CPU-only sampler/accounting paths, never raw evidence."""
    cap = root / 'adaptive/capture'
    path = next(cap.glob('*.forward.json'))
    rec = c.load(path)
    for j, row in enumerate(rec['rows']):
        row.update(position=5 + base + j - 1, generated_index=base+j)
    rec['active_query'] = [[5 + base + j - 1, token] for j, token in enumerate(inputs)]
    rec.update(selected=selected, accepted_drafts=len(selected)-1,
               rejected_drafts=rec['draft_count']-len(selected)+1)
    path.write_bytes(c.encoded(rec))
    for p in (cap / 'events').glob('*.json'):
        event = c.load(p)
        event.update(output_count=base if event['kind'] == 'account_before' else base + len(selected),
                     output_token_ids=selected, scheduled_spec_token_ids=drafts)
        event['num_output_placeholders'] = 3 if event['kind'] == 'account_before' else 3-len(selected)
        p.write_bytes(c.encoded(event))
    return path


@pytest.mark.parametrize('role', ['accepted_draft', 'recovered', 'bonus'])
def test_speculative_boundary_selection_roles(tmp_path, role):
    root = run_fixture(tmp_path)
    if role == 'recovered':
        alter_synthetic_step(root, base=59, selected=[1, 3], inputs=[1, 1, 0], drafts=[1, 0])
    elif role == 'bonus':
        alter_synthetic_step(root, base=58, selected=[1, 1, 3], inputs=[1, 1, 1], drafts=[1, 1])
    report = a.compare(root)
    assert report['state'] == 'REPRODUCED', report
    _, prompt, ids, _ = a.response(root / 'adaptive', 'adaptive')
    assert a.capture(root / 'adaptive', 'adaptive', 'd'*40, 'api', prompt, ids)['selection_role'] == role


@pytest.mark.parametrize('problem', ['stale', 'invalid_spec_tokens', 'processed_point', 'processed_row',
                                    'raw_row', 'path', 'duplicate', 'missing_processed', 'query_placeholder'])
def test_additional_mapping_failures_are_invalid(tmp_path, problem):
    root = run_fixture(tmp_path)
    cap = root / 'adaptive/capture'
    p = next(cap.glob('*.forward.json'))
    rec = c.load(p)
    if problem in ('stale', 'invalid_spec_tokens'):
        event_path = next(p for p in (cap / 'events').glob('*.json') if c.load(p)['kind'] == 'account_before')
        event = c.load(event_path)
        event[problem] = True if problem == 'stale' else 1
        event_path.write_bytes(c.encoded(event))
    elif problem == 'processed_point': rec['scores']['1']['processed']['capture_point'] = 'sampler_pre_greedy'
    elif problem == 'processed_row': rec['scores']['1']['processed']['source_row'] = 0
    elif problem == 'raw_row': rec['scores']['1']['raw']['source_row'] = 0
    elif problem == 'path': rec['decoding_path'] = 'ordinary'
    elif problem == 'duplicate': (cap / 'duplicate.forward.json').write_bytes(p.read_bytes())
    elif problem == 'missing_processed': rec['scores']['1'].pop('processed')
    elif problem == 'query_placeholder': rec['active_query'][0][1] = -1
    p.write_bytes(c.encoded(rec))
    assert a.compare(root)['state'] == 'INVALID_CAPTURE'


def upstream_method(name, method):
    text = s.fixture_sources()[name]
    assert c.sha(text.encode()) == s.manifest()['sha256'][name]
    return next(ast.get_source_segment(text, n) for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == method)


def test_async_placeholder_delta_must_match_actual_emitted_count(tmp_path):
    root = run_fixture(tmp_path)
    path = next(p for p in (root / 'adaptive/capture/events').glob('*.json') if c.load(p)['kind'] == 'account_after')
    event = c.load(path)
    event['num_output_placeholders'] = 3  # Nonnegative, but failed to consume three emitted IDs.
    path.write_bytes(c.encoded(event))
    assert a.compare(root)['state'] == 'INVALID_CAPTURE'


@pytest.mark.parametrize('problem', ['grammar', 'mrope', 'discard', 'parallel', 'positions', 'random'])
def test_unsupported_runtime_paths_invalidate_existing_boundary_capture(tmp_path, monkeypatch, problem):
    directory = tmp_path / 'adaptive'
    ids, prompt, config, runner, scheduled, raw, spec = forward_fixture(directory, 'adaptive', 3)
    path = tmp_path / 'observer.json'
    c.save(path, config)
    monkeypatch.setenv('M203_TARGETED_CONFIG', str(path))
    grammar = object() if problem == 'grammar' else None
    if problem == 'mrope': runner.uses_mrope = True
    if problem == 'discard': runner.discard_request_mask = NS(np=[True])
    if problem == 'parallel': runner.vllm_config.parallel_config.tensor_parallel_size = 2
    if problem == 'positions': runner.positions[0] = -1
    o.observe('begin', runner, scheduled, raw, spec, grammar)
    if problem == 'random': o.observe('sample_scores', raw[-1:], NS(all_greedy=False))
    with pytest.raises(c.Invalid, match='observer error present'):
        a.capture(directory, 'adaptive', 'd'*40, 'api', prompt, ids)


@pytest.mark.parametrize('discard', [False, True])
def test_real_parse_output_filtering_cannot_shift_emitted_index(tmp_path, discard):
    namespace = dict(PLACEHOLDER_TOKEN_ID=-1)
    exec('from __future__ import annotations\n' + upstream_method('v1/sample/rejection_sampler.py', 'parse_output'), namespace)
    output, _ = namespace['parse_output'](torch.tensor([[1, 99, 3, -1]]), 4, [0] if discard else [])
    assert output == ([[]] if discard else [[1, 3]])
    root = run_fixture(tmp_path)
    path = alter_synthetic_step(root, base=58, selected=[1, 99, 3], inputs=[1, 1, 1], drafts=[1, 1])
    # Raw GPU row 2 is still index 60, but parsing removed one earlier ID.
    for p in (path.parent / 'events').glob('*.json'):
        ev = c.load(p)
        ev['output_token_ids'] = output[0]
        ev['output_count'] = 59 if ev['kind'] == 'account_before' else 59+len(output[0])
        p.write_bytes(c.encoded(ev))
    assert a.compare(root)['state'] == 'INVALID_CAPTURE'


@pytest.mark.parametrize('stale,trim', [(False, False), (False, True), (True, False)])
def test_exact_async_accounting_method_consumes_only_emitted_tokens(stale, trim):
    method = upstream_method('v1/core/sched/async_scheduler.py', '_update_request_with_output')
    namespace = dict(RequestStatus=NS(RUNNING='running'))
    class Parent:
        def _update_request_with_output(self, request, output):
            emitted = output[:1] if trim else output
            request._output_token_ids.extend(emitted)
            return emitted, trim
    # Retain the unchanged method inside a class so zero-argument super() works.
    exec('from __future__ import annotations\nclass Actual(Parent):\n' + '\n'.join('    '+line for line in method.splitlines()),
         dict(namespace, Parent=Parent), result := {})
    cached = []
    runner = result['Actual']()
    runner.kv_cache_manager = NS(cache_blocks=lambda req, count: cached.append(count))
    req = NS(status='running', _output_token_ids=[1]*59, num_output_placeholders=0 if stale else 3,
             num_computed_tokens=64)
    emitted, stopped = runner._update_request_with_output(req, [1, 3, 1], stale)
    assert len(req._output_token_ids) == 59 + len(emitted)
    assert req.num_output_placeholders == (0 if stale else 3-len(emitted))
    assert cached == [64-req.num_output_placeholders] and stopped == trim


@pytest.mark.parametrize('name,method,sequence', [
    ('v1/worker/gpu_model_runner.py', 'sample_tokens', ['begin', 'apply_grammar_bitmask', '_sample', 'end', '_update_states_after_model_execute']),
    ('v1/sample/sampler.py', 'sample', ['sample_scores', 'greedy_sample']),
    ('v1/sample/rejection_sampler.py', 'forward', ['apply_sampling_constraints', 'verify_scores', 'rejection_sample']),
    ('v1/core/sched/scheduler.py', 'update_from_output', ['account_before', '_update_request_with_output', 'account_after']),
])
def test_observer_hook_owner_and_order_on_exact_pinned_source(name, method, sequence):
    patched = s.transform(name, s.fixture_sources()[name].encode()).decode()
    node = next(n for n in ast.walk(ast.parse(patched)) if isinstance(n, ast.FunctionDef) and n.name == method)
    calls = {}
    for n in ast.walk(node):
        if not isinstance(n, ast.Call): continue
        fn = n.func
        if isinstance(fn, ast.Name) and fn.id == '_m203_observe': key = n.args[0].value
        elif isinstance(fn, ast.Name): key = fn.id
        elif isinstance(fn, ast.Attribute): key = fn.attr
        else: continue
        calls.setdefault(key, []).append(n.lineno)
    assert all(len(calls[key]) == 1 for key in sequence)
    assert [calls[key][0] for key in sequence] == sorted(calls[key][0] for key in sequence)
