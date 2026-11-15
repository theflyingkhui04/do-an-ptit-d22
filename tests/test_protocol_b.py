"""CPU evidence contract tests; no simulation of GPU nondeterminism."""
import copy
import json
import subprocess
import os
import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import m203_frozen_sources as frozen
from experiments import protocol_b as pb
from experiments import protocol_b_capture as capture


def server_args(lane):
    port, model = capture.LANES[lane]
    args = ["vllm", "serve", capture.TARGET, "--host", "0.0.0.0", "--port", str(port),
            "--revision", "b" * 40, "--served-model-name", model, "--generation-config", "vllm",
            "--quantization", "awq", "--dtype", "half", "--gpu-memory-utilization", "0.85",
            "--max-model-len", "4096", "--max-num-seqs", "32"]
    if lane == "adaptive":
        args += ["--scheduler-cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
                 "--per-request-spec-decode-metrics", "summary", "--speculative-config",
                 json.dumps({"method": "draft_model", "model": capture.DRAFT, "revision": "c" * 40,
                             "num_speculative_tokens": 4, "num_speculative_tokens_per_batch_size": [[1, 32, 4]],
                             "use_heterogeneous_vocab": True, "draft_sample_method": "greedy"})]
    return args


def live_fixture(lane, env):
    settings = copy.deepcopy(env["config"]["environment"])
    settings["PYTHONPATH"] = "/checkout" if lane == "adaptive" else ""
    return {"pid": 123, "port": capture.LANES[lane][0], "argv": server_args(lane),
            "environment": settings, "working_directory": "/checkout",
            "executable": "/usr/bin/python3", "process_start_ticks": "12345",
            "versions": {"torch": env["torch_version"], "cuda": env["cuda_version"], "vllm": env["vllm_version"]},
            "sources": {"argv": "/proc/123/cmdline", "environment": "/proc/123/environ",
                        "port": f"fuser {capture.LANES[lane][0]}/tcp",
                        "versions": "/proc/123/exe + live environment"}}


@pytest.fixture
def evidence():
    prompts = [{"workload": "W1", "sample_id": f"sample-{i}",
                "dataset_version": "fixture.v1", "prompt_version": "fixture.v1",
                "request_sha256": str(i) * 64,
                "request": {"model": "fixture", "messages": [{"role": "user", "content": str(i)}],
                            "temperature": 0, "seed": 42, "max_tokens": 8}} for i in range(3)]
    env = {key: "fixture" for key in pb.ENV_FIELDS}
    env.update(git_sha="a" * 40, target_revision="b" * 40, draft_revision="c" * 40,
               plan_sha256="d" * 64, launcher_hashes={k: "e" * 64 for k in pb.LAUNCHERS},
               source_hashes={k: "f" * 64 for k in pb.REQUIRED_SOURCES},
               config={field: "fixture" for field in pb.CONFIG_FIELDS})
    env.update(gpu_name="Tesla T4", gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
               driver_version="580.1", cuda_version="13.0", torch_version="2.13.0+cu130",
               vllm_version="0.30.0", target_model=capture.TARGET, draft_model=capture.DRAFT)
    env["config"].update(concurrency=1, warmup_requests=0, request_timeout_seconds=600,
                          max_model_len=4096, max_num_seqs=32, gpu_memory_utilization=0.85)
    env["config"].update(pb.FROZEN_CONFIG)
    env["config"]["environment"] = {field: "fixture" for field in pb.CONFIG_ENV}
    env["config"]["environment"].update(CUDA_VISIBLE_DEVICES=env["gpu_uuid"], HF_HUB_OFFLINE="1")
    env["config"]["environment"].update(PROTOCOL_B_TARGET_REVISION=env["target_revision"],
                                        PROTOCOL_B_DRAFT_REVISION=env["draft_revision"], other_overrides={})
    env["config"]["environment"].update(pb.FROZEN_ENV)
    env["config"]["environment"].update(execution_root="/checkout", base_pythonpath="")
    pairs = []
    for i in range(2):
        pair = {"pair_id": f"pair-{i}"}
        for j, lane in enumerate(("vanilla", "adaptive")):
            minute = 2 * i + j
            run_id = f"run-{i}-{lane}"
            records = [{**p, "run_id": run_id, "pair_id": pair["pair_id"], "lane": lane,
                        "timestamp": f"2026-10-18T00:0{minute}:10+00:00", "ok": True,
                        "token_ids": [1, 4], "prompt_token_ids": [8],
                        "completion_tokens": 2, "finish_reason": "stop",
                        "response_id": f"response-{i}-{j}-{p['sample_id']}"} for p in prompts]
            pair[lane] = {"run_id": run_id, "pair_id": pair["pair_id"], "lane": lane,
                          "environment": copy.deepcopy(env), "complete": True,
                          "live_server": live_fixture(lane, env),
                          "started_at": f"2026-10-18T00:0{minute}:00+00:00",
                          "finished_at": f"2026-10-18T00:0{minute}:30+00:00",
                          "records": records}
        pairs.append(pair)
    return prompts, pairs


def summary(evidence):
    return pb.build_summary(*evidence, generated_at="2026-10-18T01:00:00+00:00")


def test_identical_sequences_are_evidence_only(evidence):
    result = summary(evidence)
    assert result["vanilla_repeatability"]["exact_matches"] == 3
    assert result["adaptive_repeatability"]["match_rate"] == 1
    assert result["paired_comparisons"][0]["valid_prompt_pairs"] == 3
    assert result["interpretation_status"] == "NOT_YET_APPROVED"
    assert result["task_done"] is False
    assert "passed" not in pb.canonical_json(result)


def test_first_token_mismatch_and_lengths(evidence):
    record = evidence[1][0]["adaptive"]["records"][0]
    record["token_ids"] = [1, 9, 3]
    record["completion_tokens"] = 3
    detail = summary(evidence)["paired_comparisons"][0]["details"][0]
    assert detail["first_mismatch"]["index"] == 1
    assert detail["first_mismatch"]["expected_token_id"] == 4
    assert detail["first_mismatch"]["actual_token_id"] == 9
    assert detail["completion_lengths"] == [2, 3]


def test_prefix_length_difference_and_truncation(evidence):
    record = evidence[1][0]["adaptive"]["records"][0]
    record.update(token_ids=[1], completion_tokens=1, finish_reason="length")
    metric = summary(evidence)["paired_comparisons"][0]
    detail = metric["details"][0]
    assert detail["first_mismatch"]["index"] == 1
    assert detail["first_mismatch"]["actual_token_id"] is None
    assert detail["completion_lengths"] == [2, 1]
    assert detail["finish_reasons"] == ["stop", "length"]
    assert metric["truncated_pairs"] == 1
    assert metric["full_completion_verified"] is False


def test_truncated_exact_output_does_not_claim_completion(evidence):
    evidence[1][0]["adaptive"]["records"][0]["finish_reason"] = "length"
    result = summary(evidence)
    assert result["paired_comparisons"][0]["exact_matches"] == 3
    assert result["paired_comparisons"][0]["full_completion_verified"] is False
    assert not result["task_done"]


def test_metrics_are_independent(evidence):
    evidence[1][0]["adaptive"]["records"][1]["token_ids"] = [2, 4]
    evidence[1][1]["adaptive"]["records"][1]["token_ids"] = [2, 4]
    evidence[1][1]["adaptive"]["records"][2]["token_ids"] = [3, 4]
    result = summary(evidence)
    assert result["vanilla_repeatability"]["exact_matches"] == 3
    assert result["adaptive_repeatability"]["exact_matches"] == 2
    assert [m["exact_matches"] for m in result["paired_comparisons"]] == [2, 1]
    assert result["paired_comparisons"][1]["mismatched_sample_ids"] == [
        {"workload": "W1", "sample_id": "sample-1"},
        {"workload": "W1", "sample_id": "sample-2"}]


@pytest.mark.parametrize("mutation", [
    lambda r: r["records"].pop(),
    lambda r: r["records"].append(copy.deepcopy(r["records"][0])),
    lambda r: r["records"][0].update(request_sha256="f" * 64),
    lambda r: r.update(pair_id="wrong"),
    lambda r: r["records"][0].update(pair_id="wrong"),
    lambda r: r["records"][0].update(run_id="wrong"),
    lambda r: r.update(lane="vanilla"),
    lambda r: r.update(complete=False),
    lambda r: r["records"][0].update(ok=False, error="timeout"),
    lambda r: r["records"][0].update(token_ids=[]),
    lambda r: r["records"][0].update(token_ids=[True, 4]),
    lambda r: r["records"][0].update(completion_tokens=3),
    lambda r: r["records"][0].update(finish_reason=None),
    lambda r: r["records"][0].update(prompt_token_ids=[9]),
    lambda r: r["records"][0].update(timestamp="2026-10-19T00:00:00+00:00"),
    lambda r: r.update(started_at="2026-10-18T00:00:00+00:00"),
    lambda r: r["environment"].pop("gpu_uuid"),
    lambda r: r["environment"].update(gpu_uuid="other-gpu"),
    lambda r: r["environment"].update(session_id="other-session"),
    lambda r: r["environment"].update(target_revision="e" * 40),
    lambda r: r["environment"].update(config={}),
])
def test_invalid_evidence_fails_closed(evidence, mutation):
    mutation(evidence[1][0]["adaptive"])
    with pytest.raises(pb.EvidenceError):
        summary(evidence)


def test_same_missing_provenance_in_all_runs_is_rejected(evidence):
    for pair in evidence[1]:
        for lane in capture.LANES:
            pair[lane]["environment"].pop("target_revision")
    with pytest.raises(pb.EvidenceError, match="Missing provenance"):
        summary(evidence)


def test_duplicate_run_and_pair_identity(evidence):
    evidence[1][1]["pair_id"] = evidence[1][0]["pair_id"]
    with pytest.raises(pb.EvidenceError, match="pair ID"):
        summary(evidence)


def test_run_id_cannot_be_reused_across_pairs(evidence):
    run = evidence[1][1]["vanilla"]
    run["run_id"] = evidence[1][0]["vanilla"]["run_id"]
    for record in run["records"]:
        record["run_id"] = run["run_id"]
    with pytest.raises(pb.EvidenceError, match="Reused run ID"):
        summary(evidence)


def test_summary_serialization_is_deterministic(evidence):
    before = copy.deepcopy(evidence)
    text = pb.canonical_json(summary(evidence))
    evidence[1][0]["vanilla"]["records"].reverse()
    assert text == pb.canonical_json(summary(evidence))
    assert before[0] == evidence[0]
    assert text == pb.canonical_json(json.loads(text))


def test_exclusive_output_and_duplicate_json_keys(tmp_path):
    path = tmp_path / "result.json"
    pb.write_new(path, {"task_done": False})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        pb.write_new(path, {"task_done": True})
    assert path.read_bytes() == original
    path.write_text('{"pairs": [], "pairs": []}', encoding="utf-8")
    with pytest.raises(pb.EvidenceError, match="Duplicate JSON key"):
        pb.load_json(path)


def test_frozen_plan(tmp_path):
    # Windows autocrlf changes working-tree bytes. Use immutable Git blobs in an
    # isolated fixture, exactly as an LF Linux checkout; never normalize inputs
    # inside the production hash guard or modify the user's frozen packages.
    for relative, _, _, _ in pb.WORKLOADS.values():
        request_path = pb.ROOT / "experiments/configs" / relative
        for path in (request_path, request_path.parent / "package_manifest.json"):
            name = path.relative_to(pb.ROOT).as_posix()
            data = subprocess.check_output([
                "git", "-c", f"safe.directory={pb.ROOT.as_posix()}", "show", f"HEAD:{name}"],
                cwd=pb.ROOT)
            destination = tmp_path / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    with patch.object(pb, "ROOT", tmp_path):
        plan = pb.load_plan()
    assert len(plan["prompts"]) == 54
    assert [sum(p["workload"] == w for p in plan["prompts"]) for w in pb.WORKLOADS] == [30, 12, 12]
    assert all(p["request"]["temperature"] == 0 and p["request"]["seed"] == 42
               for p in plan["prompts"])
    request_path = tmp_path / "experiments/configs/w1/node1_chat_requests.jsonl"
    request_path.write_bytes(request_path.read_bytes() + b"\n")
    with patch.object(pb, "ROOT", tmp_path):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            pb.load_plan()


def test_capture_failure_keeps_evidence_without_retry(tmp_path):
    output = tmp_path / "failed"
    with patch.object(capture, "load_plan", side_effect=pb.EvidenceError("bad plan")) as load:
        with pytest.raises(pb.EvidenceError, match="bad plan"):
            capture.capture(output, "p1", "v1", "vanilla", "session")
    assert load.call_count == 1
    assert pb.load_json(output / "run.json")["complete"] is False
    assert (output / "started.json").exists()
    with pytest.raises(FileExistsError):
        capture.capture(output, "p1", "v1", "vanilla", "session")


def test_transport_timeout_is_journaled_once(evidence, tmp_path):
    # Test transport/error retention only, with no GPU runtime calls or claims.
    output = tmp_path / "timeout"
    plan = {"prompts": evidence[0]}
    model = capture.LANES["vanilla"][1]
    with patch.object(capture, "load_plan", return_value=plan), \
         patch.object(capture, "runtime", return_value=evidence[1][0]["vanilla"]["environment"]), \
         patch.object(capture, "server_processes", return_value=evidence[1][0]["vanilla"]["live_server"]), \
         patch.object(capture, "http_json", return_value={"data": [{"id": model}]}), \
         patch.object(capture, "request_raw", side_effect=TimeoutError("fixture timeout")) as http:
        with pytest.raises(pb.EvidenceError, match="without retry"):
            capture.capture(output, "p1", "v1", "vanilla", "session")
    assert http.call_count == 1
    lines = (output / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    raw = json.loads(lines[1])
    assert "TimeoutError" in raw["error"]
    assert raw["request"] == dict(evidence[0][0]["request"], model=model, return_token_ids=True)
    run = pb.load_json(output / "run.json")
    assert run["complete"] is False and len(run["records"]) == 1
    assert run["responses_sha256"] == pb.file_sha(output / "responses.jsonl")


def test_response_parser_validates_ids_and_lengths():
    body = {"id": "response-1", "model": "fixture", "prompt_token_ids": [8],
            "choices": [{"token_ids": [1, 2], "finish_reason": "length"}],
            "usage": {"completion_tokens": 2}}
    assert capture.parse_completion(body, "fixture")["finish_reason"] == "length"
    body["usage"]["completion_tokens"] = 3
    with pytest.raises(pb.EvidenceError):
        capture.parse_completion(body, "fixture")


def test_server_arguments_preserve_json_argument_boundaries():
    args = server_args("vanilla")
    capture.validate_server_command(args, "vanilla", "b" * 40, "c" * 40)
    args = server_args("adaptive")
    capture.validate_server_command(args, "adaptive", "b" * 40, "c" * 40)
    args[args.index("4096")] = "8192"
    with pytest.raises(pb.EvidenceError, match="config"):
        capture.validate_server_command(args, "adaptive", "b" * 40, "c" * 40)


def test_report_round_trip_and_journal_tamper(evidence, tmp_path):
    prompts, pairs = evidence
    manifest = {"pairs": []}
    for i, pair in enumerate(pairs):
        item = {"pair_id": pair["pair_id"]}
        for lane in capture.LANES:
            directory = tmp_path / f"{i}-{lane}"
            directory.mkdir()
            journal = directory / "responses.jsonl"
            events = []
            for r in pair[lane]["records"]:
                body = {"id": r["response_id"], "model": capture.LANES[lane][1],
                             "prompt_token_ids": r["prompt_token_ids"],
                             "choices": [{"token_ids": r["token_ids"], "finish_reason": r["finish_reason"]}],
                             "usage": {"completion_tokens": r["completion_tokens"]}}
                attempt = {"event": "attempt", "identity": {"sample_id": r["sample_id"]},
                           "request": dict(r["request"], model=capture.LANES[lane][1], return_token_ids=True)}
                events.extend([attempt, {**attempt, "event": "result", "record": r,
                                         "response": body, "http_status": 200,
                                         "raw_response_body": pb.canonical_json(body)}])
            journal.write_text("\n".join(pb.canonical_json(e) for e in events) + "\n", encoding="utf-8")
            pair[lane]["responses_sha256"] = pb.file_sha(journal)
            pb.write_new(directory / "run.json", pair[lane])
            item[lane] = f"{directory.name}/run.json"
        manifest["pairs"].append(item)
    pb.write_new(tmp_path / "experiment.json", manifest)
    plan = {"prompts": prompts, "plan_sha256": "d" * 64}
    with patch.object(pb, "load_plan", return_value=plan):
        result = pb.report(tmp_path / "experiment.json", tmp_path / "summary.json",
                           generated_at="2026-10-18T01:00:00+00:00")
        assert len(result["input_evidence"]) == 4
        journal.write_text("{}\n", encoding="utf-8")
        with pytest.raises(pb.EvidenceError, match="journal hash"):
            pb.report(tmp_path / "experiment.json", tmp_path / "invalid.json",
                      generated_at="2026-10-18T01:00:00+00:00")
        assert not (tmp_path / "invalid.json").exists()


def test_missing_expected_manifest_is_rejected(evidence):
    evidence[0].append(copy.deepcopy(evidence[0][0]))
    with pytest.raises(pb.EvidenceError, match="Duplicate expected prompt"):
        summary(evidence)


def test_fix_duplicate_reservation_precedes_all_runbook_effects(tmp_path):
    output = tmp_path / "pair1-vanilla"
    capture.reserve_run(output, "pair1", "pair1-vanilla", "vanilla", "session")
    (output / "switch.log").write_bytes(b"original evidence")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with patch.object(capture, "server_processes") as live:
        with pytest.raises(FileExistsError):
            capture.capture(output, "pair1", "pair1-vanilla", "vanilla", "session")
        live.assert_not_called()
    book = (pb.ROOT / "docs/runbooks/m2-03-protocol-b.md").read_text(encoding="utf-8")
    function = re.search(r"(?ms)^run_lane\(\) \{.*?^\}", book).group()
    git_bash = Path("C:/Program Files/Git/usr/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    assert bash, "Bash is needed to verify runbook reservation ordering"
    marker = tmp_path / "side-effect"
    script = '''
python() { command "$TEST_PYTHON" -B "$@"; }
assert_gpu() { echo touched >> "$TEST_MARKER"; }
nvidia-smi() { echo touched >> "$TEST_MARKER"; }
bash() { echo touched >> "$TEST_MARKER"; }
''' + function + '\nrun_lane pair1 vanilla lane1 1\n'
    import sys
    env = dict(os.environ, RUN_ROOT=tmp_path.as_posix(), SESSION_ID="session",
               TEST_PYTHON=Path(sys.executable).as_posix(), TEST_MARKER=marker.as_posix())
    result = subprocess.run([bash, "--noprofile", "--norc"], input=script.encode("utf-8"),
                            capture_output=True, cwd=pb.ROOT, env=env)
    assert result.returncode != 0 and not marker.exists()
    assert b"CAPTURE_FAILED" in result.stderr and b"exists" in result.stderr.lower()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("field,value", [
    ("driver_version", False), ("cuda_version", ""), ("torch_version", None),
    ("vllm_version", []), ("gpu_name", " "), ("gpu_uuid", "not-a-uuid"),
    ("target_revision", "main"), ("target_model", ""),
    ("launcher_hashes", {"unrelated": "a" * 64}),
    ("source_hashes", {"unrelated": "a" * 64}),
])
def test_fix_strict_provenance_values(evidence, field, value):
    for pair in evidence[1]:
        for lane in capture.LANES:
            pair[lane]["environment"][field] = value
    with pytest.raises(pb.EvidenceError):
        summary(evidence)


@pytest.mark.parametrize("field", pb.ENV_FIELDS)
def test_fix_missing_mandatory_provenance(evidence, field):
    for pair in evidence[1]:
        for lane in capture.LANES:
            pair[lane]["environment"].pop(field)
    with pytest.raises(pb.EvidenceError):
        summary(evidence)


@pytest.mark.parametrize("field", ["cache_policy", "sampling"])
def test_fix_empty_policy(evidence, field):
    for pair in evidence[1]:
        for lane in capture.LANES:
            pair[lane]["environment"]["config"][field] = ""
    with pytest.raises(pb.EvidenceError):
        summary(evidence)


@pytest.mark.parametrize("field,key", [("launcher_hashes", k) for k in pb.LAUNCHERS]
                         + [("source_hashes", k) for k in pb.REQUIRED_SOURCES])
def test_fix_missing_individual_hash_entry(evidence, field, key):
    for pair in evidence[1]:
        for lane in capture.LANES:
            pair[lane]["environment"][field].pop(key)
    with pytest.raises(pb.EvidenceError):
        summary(evidence)


def test_fix_duplicate_reserved_capture_claim(tmp_path):
    output = tmp_path / "claimed"
    capture.reserve_run(output, "p1", "v1", "vanilla", "session")
    with patch.object(capture, "load_plan", side_effect=pb.EvidenceError("fixture interruption")):
        with pytest.raises(pb.EvidenceError):
            capture.capture(output, "p1", "v1", "vanilla", "session", reserved=True)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with patch.object(capture, "server_processes") as live:
        with pytest.raises(FileExistsError):
            capture.capture(output, "p1", "v1", "vanilla", "session", reserved=True)
        live.assert_not_called()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("extra", [["--revision", "other-revision"], ["--enable-prefix-caching"],
                                  ["--disable-log-requests"], ["--seed", "123"]])
def test_fix_reject_unapproved_live_flags(extra):
    for lane in capture.LANES:
        with pytest.raises(pb.EvidenceError):
            capture.validate_server_command(server_args(lane) + extra, lane, "b" * 40, "c" * 40)


def test_fix_live_revision_environment_and_source_binding(evidence):
    run = evidence[1][0]["adaptive"]
    capture.validate_live_server(run["live_server"], "adaptive", run["environment"])
    for mutate in (
        lambda l: l.update(port=8001),
        lambda l: l["environment"].update(CUDA_VISIBLE_DEVICES="GPU-other"),
        lambda l: l["sources"].pop("environment"),
        lambda l: l["argv"].__setitem__(l["argv"].index("b" * 40), "d" * 40),
    ):
        live = copy.deepcopy(run["live_server"])
        mutate(live)
        with pytest.raises(pb.EvidenceError):
            capture.validate_live_server(live, "adaptive", run["environment"])


@pytest.mark.parametrize("response", [
    {"http_status": 200, "raw_response_body": '{"choices":[null]}'},
    {"http_status": 200, "raw_response_body": '{not-json'},
    {"http_status": 200, "raw_response_body": '{"partial":', "transport_error": "IncompleteRead"},
])
def test_fix_failed_request_always_has_attempt_and_raw_result(evidence, tmp_path, response):
    model = capture.LANES["vanilla"][1]
    response = dict(response)
    if "null" in response["raw_response_body"]:
        response["raw_response_body"] = json.dumps({"model": model, "choices": [None]})
    run = evidence[1][0]["vanilla"]
    with patch.object(capture, "load_plan", return_value={"prompts": evidence[0]}), \
         patch.object(capture, "server_processes", side_effect=lambda lane: copy.deepcopy(run["live_server"])), \
         patch.object(capture, "runtime", return_value=run["environment"]), \
         patch.object(capture, "http_json", return_value={"data": [{"id": model}]}), \
         patch.object(capture, "request_raw", return_value=response) as http:
        with pytest.raises(pb.EvidenceError, match="without retry"):
            capture.capture(tmp_path / "failed", "pair1", "v1", "vanilla", "session")
    assert http.call_count == 1
    events = [json.loads(line) for line in (tmp_path / "failed/responses.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["attempt", "result"]
    result = events[1]
    assert result["raw_response_body"] == response["raw_response_body"]
    assert result["record"]["sample_id"] == evidence[0][0]["sample_id"]
    assert result["record"]["ok"] is False and result["record"]["latency_seconds"] >= 0
    assert result["record"]["completion_status"] == "failed" and result["error"]


def test_fix_partial_transport_bytes_retained():
    import http.client
    from unittest.mock import MagicMock
    response = MagicMock()
    response.__enter__.return_value = response
    response.code = 200
    response.read.side_effect = http.client.IncompleteRead(b'{"partial":', 20)
    with patch.object(capture.urllib.request, "urlopen", return_value=response) as open_http:
        result = capture.request_raw("http://127.0.0.1:8001/fixture", {})
    assert result["raw_response_body"] == '{"partial":'
    assert result["http_status"] == 200 and "IncompleteRead" in result["transport_error"]
    assert open_http.call_count == 1


def test_fix_server_interpreter_retains_virtualenv_path(tmp_path):
    invoked, binary = tmp_path / "venv/bin/python", tmp_path / "system/python"
    with patch.object(capture.shutil, "which", return_value=str(invoked)), \
         patch.object(Path, "resolve", return_value=binary):
        assert capture.server_interpreter("python", {"PATH": "fixture"}, binary) == str(invoked.absolute())
    with patch.object(capture.shutil, "which", return_value=None):
        with pytest.raises(pb.EvidenceError, match="interpreter"):
            capture.server_interpreter("python", {}, binary)


def test_fix_invalid_utf8_is_retained_and_invalid():
    import base64
    from unittest.mock import MagicMock
    response = MagicMock()
    response.__enter__.return_value = response
    response.code = 200
    response.read.return_value = b'{"text":"\xff"}'
    with patch.object(capture.urllib.request, "urlopen", return_value=response):
        result = capture.request_raw("http://127.0.0.1:8001/fixture", {})
    assert base64.b64decode(result["raw_response_base64"]) == response.read.return_value
    assert "UnicodeDecodeError" in result["decoding_error"]


@pytest.mark.parametrize("lane", ["vanilla", "adaptive"])
@pytest.mark.parametrize("pinned", [False, True])
def test_fix_launchers_bind_revisions_only_when_requested(tmp_path, lane, pinned):
    # Exercise shell argument construction with a stub command; never start vLLM.
    if pinned and lane == "adaptive" and "scripts/launch_lane3_adaptive.sh" in frozen.live_drift():
        pytest.skip(frozen.live_tree_reason())  # the approved Protocol B server config is the frozen launcher's
    import sys
    git_bash = Path("C:/Program Files/Git/usr/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    assert bash
    argument_file = tmp_path / "argv.bin"
    env = dict(os.environ, TEST_PYTHON=Path(sys.executable).as_posix(),
               TEST_ARG_FILE=argument_file.as_posix(), ADAPTIVE_TELEMETRY_PATH=(tmp_path / "telemetry.jsonl").as_posix())
    for key in ("PROTOCOL_B_TARGET_REVISION", "PROTOCOL_B_DRAFT_REVISION"):
        env.pop(key, None)
    if pinned:
        env.update(PROTOCOL_B_TARGET_REVISION="b" * 40, PROTOCOL_B_DRAFT_REVISION="c" * 40)
    script = '''
vllm() { printf '%s\\0' "$@" > "$TEST_ARG_FILE"; }
python3() { command "$TEST_PYTHON" "$@"; }
''' + f"source scripts/launch_lane{1 if lane == 'vanilla' else 3}_{'vanilla' if lane == 'vanilla' else 'adaptive'}.sh\n"
    result = subprocess.run([bash, "--noprofile", "--norc"], input=script.encode("utf-8"),
                            capture_output=True, cwd=pb.ROOT, env=env)
    assert result.returncode == 0, result.stderr
    args = ["vllm", *argument_file.read_bytes().decode("utf-8").rstrip("\0").split("\0")]
    if pinned:
        capture.validate_server_command(args, lane, "b" * 40, "c" * 40)
    else:
        assert "--revision" not in args
        if lane == "adaptive":
            assert "revision" not in json.loads(args[args.index("--speculative-config") + 1])


def test_fix_complete_capture_to_report(evidence, tmp_path):
    prompts, fixtures = evidence
    manifest = {"pairs": []}
    for i, pair in enumerate(fixtures):
        entry = {"pair_id": pair["pair_id"]}
        for lane in capture.LANES:
            fixture = pair[lane]
            model = capture.LANES[lane][1]
            bodies = [{"http_status": 200, "raw_response_body": json.dumps({
                "id": r["response_id"], "model": model, "prompt_token_ids": r["prompt_token_ids"],
                "choices": [{"token_ids": r["token_ids"], "finish_reason": "stop"}],
                "usage": {"completion_tokens": len(r["token_ids"])}})} for r in fixture["records"]]
            destination = tmp_path / f"{i}-{lane}"
            with patch.object(capture, "load_plan", return_value={"prompts": prompts}), \
                 patch.object(capture, "server_processes", side_effect=lambda lane, f=fixture: copy.deepcopy(f["live_server"])), \
                 patch.object(capture, "runtime", return_value=fixture["environment"]), \
                 patch.object(capture, "http_json", return_value={"data": [{"id": model}]}), \
                 patch.object(capture, "request_raw", side_effect=bodies):
                captured = capture.capture(destination, pair["pair_id"], fixture["run_id"], lane, "session")
            assert captured["complete"] and all(r["completion_status"] == "success" for r in captured["records"])
            entry[lane] = f"{destination.name}/run.json"
        manifest["pairs"].append(entry)
    pb.write_new(tmp_path / "experiment.json", manifest)
    with patch.object(pb, "load_plan", return_value={"prompts": prompts, "plan_sha256": "d" * 64}):
        result = pb.report(tmp_path / "experiment.json", tmp_path / "summary.json", generated_at=pb.utc_now())
    assert result["vanilla_repeatability"]["exact_matches"] == len(prompts)
    assert result["interpretation_status"] == "NOT_YET_APPROVED" and not result["task_done"]
