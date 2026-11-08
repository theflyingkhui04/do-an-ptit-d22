"""[M5-01] W1_V2 benchmark campaign tooling: CPU-only, no Modal app/network/GPU is started."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import modal_lanes as lanes
from experiments import run_m3_campaign as m3
from experiments import run_m5_campaign as m5
from experiments import vllm_runtime_patch as patch
from experiments.bench_source import snapshot
from experiments.validate_m3 import validate_cell
from experiments.workload_adapter import load_workload_rows, verify_workload_integrity
from tests.test_modal_bench_jobs import (  # noqa: F401  (pytest fixture reuse)
    job,
    submit,
)

ROOT = Path(__file__).resolve().parents[1]

# Same shape as the aggregate branch of vLLM 0.30.0 RequestOutput.add (vllm/outputs.py).
SNIPPET = (
    "                    if aggregate:\n"
    "                        completion.text += next_completion.text\n"
    "                        if next_completion.logprobs:\n"
    "                            completion.logprobs.extend(next_completion.logprobs)\n"
    "                        completion.cumulative_logprob = (\n"
    "                            next_completion.cumulative_logprob\n"
    "                        )\n"
    "                        completion.finish_reason = next_completion.finish_reason\n"
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# vLLM #55708 backport (same bytes as the M3-05 Colab runtime)
# --------------------------------------------------------------------------
def test_runtime_identity_matches_m305_colab_evidence():
    assert patch.RUNTIME_LABEL == "vllm-0.30.0-plus-backport-55708"
    assert lanes.VLLM_RUNTIME_LABELS == {"stock": patch.STOCK_RUNTIME_LABEL, "backport-55708": patch.RUNTIME_LABEL}
    assert patch.STOCK_OUTPUTS_SHA256 == "346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425"
    assert patch.PATCHED_OUTPUTS_SHA256 == "8f58684245f17007863382ffa9de115d371738ae28731b0c876196eb3b3fea47"


def test_backport_preserves_spec_metrics_before_cumulative_logprob():
    out = patch.apply_backport_55708(SNIPPET)
    assert out.count("completion.spec_decode_metrics = (") == 1
    assert out.index("if next_completion.spec_decode_metrics is not None:") < out.index(
        "completion.cumulative_logprob = (")
    assert out.replace(patch.BACKPORT_BLOCK, "") == SNIPPET


@pytest.mark.parametrize("source", ["no anchor here\n", SNIPPET + SNIPPET])
def test_backport_refuses_unknown_source(source):
    with pytest.raises(ValueError, match="anchor"):
        patch.apply_backport_55708(source)


def test_patch_file_is_hash_gated_and_idempotent(tmp_path):
    target = tmp_path / "outputs.py"
    target.write_bytes(SNIPPET.encode())
    patched = patch.apply_backport_55708(SNIPPET)
    hashes = {"stock_sha256": sha(SNIPPET), "patched_sha256": sha(patched)}

    assert patch.patch_file(target, **hashes) == "patched"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == sha(patched)
    assert patch.patch_file(target, **hashes) == "already-patched"

    target.write_bytes(b"something else\n")
    with pytest.raises(ValueError, match="sha256"):
        patch.patch_file(target, **hashes)


def test_wrong_expected_patched_hash_fails_closed(tmp_path):
    target = tmp_path / "outputs.py"
    target.write_bytes(SNIPPET.encode())
    with pytest.raises(ValueError, match="sha256"):
        patch.patch_file(target, stock_sha256=sha(SNIPPET), patched_sha256="0" * 64)
    assert target.read_bytes() == SNIPPET.encode()


# --------------------------------------------------------------------------
# Serving requirements per workload
# --------------------------------------------------------------------------
def test_serving_requirements_default_and_w1_v2():
    assert lanes.workload_serving("W1") == {"max_model_len": 4096, "timeout_s": None, "vllm_runtime": None}
    assert lanes.workload_serving("W3") == lanes.workload_serving("W1")
    assert lanes.workload_serving("W1_V2") == {
        "max_model_len": 8192, "timeout_s": 600, "vllm_runtime": "backport-55708"}
    with pytest.raises(ValueError):
        lanes.workload_serving("W9")


def test_resolve_vllm_runtime():
    assert lanes.resolve_vllm_runtime({}) == "stock"
    assert lanes.resolve_vllm_runtime({"PTIT_VLLM_RUNTIME": "backport-55708"}) == "backport-55708"
    with pytest.raises(ValueError):
        lanes.resolve_vllm_runtime({"PTIT_VLLM_RUNTIME": "nightly"})


def harness_flags(workload: str) -> tuple[list[str], dict[str, str]]:
    args = lanes.bench_harness_args(
        "eagle", base_url="https://x.test/v1", server_env_json="/a/server_env.json", raw_out="/a/raw.jsonl",
        summary_csv="/a/h.csv", experiment_id="m5-01", workload=workload, concurrency=32, repeats=5,
        num_requests=32)
    return args, dict(zip(args, args[1:]))


def test_only_w1_v2_harness_gets_a_longer_request_timeout():
    import sys
    from unittest.mock import patch as mock_patch

    from experiments import benchmark_harness as harness

    args, flags = harness_flags("W1_V2")
    assert flags["--timeout-s"] == "600"
    with mock_patch.object(sys, "argv", ["harness", *args]):
        assert harness.parse_args().timeout_s == 600.0
    args, _ = harness_flags("W1")
    assert "--timeout-s" not in args


def test_submit_refuses_w1_v2_on_a_short_context_server(job):  # noqa: F811
    job.identity.remote.return_value["max_model_len"] = 4096
    with pytest.raises(ValueError, match="max_model_len"):
        job.bench_submit(lane="eagle", server_run_id="server-smoke", bench_run_id="pilot-cell",
                         workload="W1_V2", concurrency=32, repeats=5, num_requests=32)
    job.deployed.spawn.assert_not_called()


def test_submit_accepts_w1_v2_on_a_long_context_server(job):  # noqa: F811
    job.identity.remote.return_value.update(max_model_len=8192, vllm_runtime="backport-55708")
    assert job.bench_submit(lane="eagle", server_run_id="server-smoke", bench_run_id="pilot-cell",
                            workload="W1_V2", concurrency=32, repeats=5, num_requests=32) == "fc-test001"


def test_submit_of_canonical_workloads_is_unchanged(job):  # noqa: F811
    assert submit(job) == "fc-test001"


# --------------------------------------------------------------------------
# Cell validation for W1_V2
# --------------------------------------------------------------------------
def w1_v2_evidence(tmp_path: Path, lane: str = "eagle", **server_overrides) -> Path:
    packed, commit = snapshot(ROOT)
    try:
        path = Path(packed.name) / lanes.workload_file("W1_V2")
        integrity = verify_workload_integrity(path)
        corpus = load_workload_rows(path)
    finally:
        packed.cleanup()
    cell = {"workload": "W1_V2", "concurrency": 32, "repeats": 5, "num_requests": 32}
    receipt = {"lane": lane, "server_run_id": "cpu-server", "bench_run_id": "cpu-cell",
               "call_id": "fc-cpufixture", "commit": commit, "cell": cell}
    server = lanes.build_server_env(gpu="NVIDIA L4", driver_version="fixture", gpu_memory_total_mb=23034,
                                    torch_version="fixture", torch_cuda_version="13.0", vllm_version="0.30.0",
                                    lane=lane, run_id="cpu-server", profile="l4-bf16")
    server.update(source_commit=commit, instance_id="cpu-boot",
                  vllm_args=lanes.build_vllm_args(lane, max_model_len=8192),
                  runtime_variant=patch.RUNTIME_LABEL, vllm_outputs_sha256=patch.PATCHED_OUTPUTS_SHA256)
    server.update(server_overrides)
    draft = lanes.EAGLE_REVISION if lane == "eagle" else (None if lane == "vanilla" else lanes.DRAFT_REVISION)
    raw, warm = [], []
    for repeat in range(6):
        phase = "warmup" if repeat == 0 else "measured"
        offset = ((repeat - 1) * (32 % len(corpus))) % len(corpus) if repeat else 0
        for index in range(3 if repeat == 0 else 32):
            sample_index = (index + offset) % len(corpus)
            request = corpus[sample_index]["request"]
            row = {"experiment_id": "cpu-cell", "lane": lane, "workload": "W1_V2", "concurrency": 32,
                   "phase": phase, "repeat_id": f"r{repeat:03}" if repeat else None,
                   "request_id": f"cpu-{repeat}-{index}", "request_index": index, "commit": commit,
                   "server_run_id": "cpu-server", "server_instance_id": "cpu-boot",
                   "sample_policy": "cycle-rotate", "sample_id": corpus[sample_index]["sample_id"],
                   "sample_index": sample_index, "sample_offset": offset, "canonical_sample_count": len(corpus),
                   "success": True, "workload_file_sha256": integrity.actual_sha256, "metrics_complete": True,
                   "http_status": 200, "target_revision": lanes.TARGET_REVISION, "draft_revision": draft,
                   "model": lanes.LANES[lane].served_model_name, "temperature": request["temperature"],
                   "seed": request["seed"], "max_completion_tokens": request["max_tokens"],
                   "response_format": request.get("response_format"),
                   "timestamp_utc": f"2026-11-09T00:0{repeat}:00+00:00", "e2e_ms": 1000, "ttft_ms": 10,
                   "tpot_ms": 20, "output_tokens": 1400, "gpu": "NVIDIA L4", "run_id": f"cpu-run-{repeat}",
                   "run_duration_s": 60,
                   "server_metrics": {"speculative_decoding": {"num_spec_steps": 5, "num_draft_tokens": 15,
                                                               "num_accepted_draft_tokens": 5,
                                                               "num_spec_tokens": 3}}}
            (raw if repeat else warm).append(row)

    def write(name: str, value: dict) -> None:
        (tmp_path / name).write_bytes((json.dumps(value) + "\n").encode())

    write("submission.json", receipt)
    write("bench.json", {**receipt, "returncode": 0, "server_instance_unchanged": True,
                         "endpoint_model_ids": [lanes.LANES[lane].served_model_name]})
    write("server_env.json", server)
    (tmp_path / "raw.jsonl").write_bytes("".join(json.dumps(r) + "\n" for r in raw).encode())
    (tmp_path / "warmup.jsonl").write_bytes("".join(json.dumps(r) + "\n" for r in warm).encode())
    return tmp_path


def test_validator_accepts_w1_v2_cell_on_patched_long_context_server(tmp_path):
    result = validate_cell(w1_v2_evidence(tmp_path))
    assert result["measured_requests"] == 160
    assert result["speculative_alpha_tau"] == (1 / 3, 2)


@pytest.mark.parametrize("override", [
    {"vllm_args": lanes.build_vllm_args("eagle")},
    {"runtime_variant": "vllm-0.30.0"},
    {"vllm_outputs_sha256": patch.STOCK_OUTPUTS_SHA256},
])
def test_validator_rejects_w1_v2_on_the_wrong_server(tmp_path, override):
    with pytest.raises(ValueError):
        validate_cell(w1_v2_evidence(tmp_path, **override))


# --------------------------------------------------------------------------
# Campaign plan, drive and aggregation
# --------------------------------------------------------------------------
def test_plan_is_four_lanes_by_six_concurrencies_on_w1_v2():
    record = m5.plan("cpu-m5", "a" * 40)
    assert len(record["cells"]) == 24
    assert len({c["key"] for c in record["cells"]}) == 24
    assert {c["workload"] for c in record["cells"]} == {"W1_V2"}
    assert {(c["lane"], c["concurrency"]) for c in record["cells"]} == {
        (lane, c) for lane in lanes.LANES for c in (1, 2, 4, 8, 16, 32)}
    assert (record["warmup"], record["repeats"], record["num_requests"]) == (3, 5, 32)
    assert (record["gpu"], record["max_model_len"], record["vllm_runtime"]) == ("L4", 8192, patch.RUNTIME_LABEL)
    assert record["kind"] == "confirmatory" and record["task"] == "M5-01"


def test_pilot_plan_is_one_eagle_c32_cell():
    record = m5.plan("cpu-m5-pilot", "a" * 40, pilot=True)
    assert record["kind"] == "pilot"
    assert [(c["lane"], c["workload"], c["concurrency"]) for c in record["cells"]] == [("eagle", "W1_V2", 32)]


@pytest.mark.parametrize("campaign_id, sha", [("../x", "a" * 40), ("ok", "HEAD")])
def test_plan_rejects_unsafe_identity(campaign_id, sha):
    with pytest.raises(ValueError):
        m5.plan(campaign_id, sha)


def test_plan_is_saved_under_m5_01_and_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(m3, "ROOT", tmp_path)
    record = m5.plan("cpu-m5", "a" * 40)
    path = m3.save_plan(record, m5.PLAN_ROOT)
    assert path == tmp_path / "experiments/runs/m5_01/cpu-m5"
    assert m3.save_plan(record, m5.PLAN_ROOT) == path
    with pytest.raises(FileExistsError):
        m3.save_plan({**record, "repeats": 4}, m5.PLAN_ROOT)


def test_m3_plan_location_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(m3, "ROOT", tmp_path)
    assert m3.save_plan(m3.plan("cpu-m3", "a" * 40)) == tmp_path / "experiments/runs/m3_05/cpu-m3"


def test_drive_deploys_long_context_patched_l4_lane(monkeypatch):
    import experiments

    record = m5.plan("cpu-m5", "a" * 40)
    record["cells"] = record["cells"][:1]
    deploys: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(m3, "command", lambda *args, env=None: deploys.append((args, env)))
    monkeypatch.setattr(m3, "stop", Mock())
    monkeypatch.setattr(m3, "save_plan", Mock())
    monkeypatch.setattr(m3, "selected_attempt", lambda r, c: None)
    monkeypatch.setattr(m3, "attempts", lambda r, c: [])
    runner = SimpleNamespace(_committed_bench_source=lambda: "a" * 40, bench_submit=Mock(),
                             bench_status=Mock(side_effect=ConnectionError("laptop offline")))
    monkeypatch.setattr(experiments, "modal_runner", runner, raising=False)

    with pytest.raises(ConnectionError):
        m5.run(record)

    (args, env), = deploys
    assert args == ("modal", "deploy", "-m", "experiments.modal_runner")
    assert (env["PTIT_GPU"], env["PTIT_MAX_MODEL_LEN"], env["PTIT_VLLM_RUNTIME"]) == ("L4", "8192", "backport-55708")
    assert env["PTIT_LANES"] == record["cells"][0]["lane"] and env["PTIT_KEEP_WARM"] == "0"
    assert env["PTIT_RUN_ID"] == f"cpu-m5-{record['cells'][0]['lane']}-W1_V2-c1-a001"
    kwargs = runner.bench_submit.call_args.kwargs
    assert (kwargs["workload"], kwargs["repeats"], kwargs["num_requests"]) == ("W1_V2", 5, 32)
    m3.save_plan.assert_called_once_with(record, m5.PLAN_ROOT)


def test_validate_campaign_requires_every_cell_and_fresh_servers(monkeypatch, tmp_path):
    record = m5.plan("cpu-m5", "a" * 40)
    monkeypatch.setattr(m3, "selected_attempt", lambda r, c: None)
    with pytest.raises(ValueError, match="Incomplete campaign"):
        m5.validate_campaign(record)

    monkeypatch.setattr(m3, "selected_attempt", lambda r, c: tmp_path / c["key"])
    monkeypatch.setattr(m5, "validate_cell", lambda p: {"server_instance_id": "same-boot"})
    monkeypatch.setattr(m5, "server_gpu", lambda p: "NVIDIA L4")
    with pytest.raises(ValueError, match="reused"):
        m5.validate_campaign(record)

    ids = iter(range(100))
    monkeypatch.setattr(m5, "validate_cell", lambda p: {"server_instance_id": f"boot-{next(ids)}"})
    monkeypatch.setattr(m5, "server_gpu", lambda p: "NVIDIA A10G")
    with pytest.raises(ValueError, match="L4"):
        m5.validate_campaign(record)


def summary_rows(workload: str, speedups: dict[tuple[str, int], float]) -> list[dict]:
    return [{"lane": lane, "workload": workload, "concurrency": c, "speedup_vs_vanilla": s,
             "tpot_p50": 20.0 / s, "throughput": 100.0 * s, "alpha": None if lane == "vanilla" else 0.9}
            for (lane, c), s in speedups.items()]


def test_comparison_pairs_w1_v2_with_m305_w1_by_lane_and_concurrency():
    m5_summary = {"cells": summary_rows("W1_V2", {("vanilla", 1): 1.0, ("adaptive", 1): 1.30})}
    m3_summary = {"cells": summary_rows("W1", {("vanilla", 1): 1.0, ("adaptive", 1): 1.45})
                  + summary_rows("W2", {("adaptive", 1): 9.9})}
    text = m5.comparison_markdown(m5_summary, m3_summary)
    assert "| adaptive | 1 | 1.45 | 1.30 |" in text
    assert "9.9" not in text
    assert "Colab" in text and "8192" in text
