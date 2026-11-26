"""Offline tests for the in-container session of the one-container-per-GPU benchmark (no GPU, no network)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from experiments import bench_gpu_session as sess
from experiments import run_benchmark_campaign as camp

T0 = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)


def _iso(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


def _raw(repeats: int, *, seconds_each: float = 10.0) -> list[dict]:
    """Two requests per repeat with the harness' run window fields."""
    rows = []
    for r in range(repeats):
        start, end = r * (seconds_each + 1), r * (seconds_each + 1) + seconds_each
        for i in range(2):
            rows.append({"repeat_id": f"r{r + 1:03}", "run_id": f"run-{r}", "request_index": i,
                         "run_started_at_utc": _iso(start), "run_finished_at_utc": _iso(end)})
    return rows


def _samples(repeats: int, running: float, waiting: float = 0.0, *, seconds_each: float = 10.0) -> list[dict]:
    out = []
    for r in range(repeats):
        base = r * (seconds_each + 1)
        for k in range(1, 5):
            out.append({"ts_utc": _iso(base + k * 2), "running": running, "waiting": waiting, "used_mb": 1000.0})
    return out


def test_parse_vllm_metrics_sums_labels():
    text = (
        "# HELP vllm:num_requests_running Number of requests\n"
        'vllm:num_requests_running{model_name="a"} 7.0\n'
        'vllm:num_requests_running{model_name="b"} 1.0\n'
        'vllm:num_requests_waiting{model_name="a"} 3.0\n'
        "vllm:num_requests_waiting_by_reason 9.0\n"
    )
    assert sess.parse_vllm_metrics(text) == (8.0, 3.0)


def test_concurrency_check_counts_waiting_for_kv_limited_gpu():
    result = sess.concurrency_check(_samples(5, running=20, waiting=12), _raw(5), 32)
    assert result["ok"] is True
    assert result["per_repeat"]["r001"] == {"peak_inflight": 32.0, "peak_running": 20.0}


def test_concurrency_check_rejects_serialized_requests():
    result = sess.concurrency_check(_samples(5, running=1, waiting=0), _raw(5), 8)
    assert result["ok"] is False
    assert result["per_repeat"]["r003"]["peak_inflight"] == 1.0


def test_concurrency_check_rejects_a_repeat_without_samples():
    samples = [s for s in _samples(5, running=8) if not s["ts_utc"].startswith(_iso(11 + 2)[:19])]
    samples = [s for s in samples if not (_iso(11) <= s["ts_utc"] <= _iso(21))]
    assert sess.concurrency_check(samples, _raw(5), 8)["ok"] is False


def test_check_gpu_raises_wrong_gpu_for_a10g():
    with pytest.raises(sess.WrongGpu, match="NVIDIA A10G"):
        sess.check_gpu("A10", "NVIDIA A10G")
    sess.check_gpu("A10", "NVIDIA A10")


def test_session_order_groups_by_workload_then_lane():
    record = camp.plan("order", "a" * 40, mode="fast", gpus="A10")
    shuffled = list(reversed(record["cells"]))
    order = sess.session_order(shuffled)
    assert [(w, l) for w, l, _ in order][:4] == [
        ("W1_V3", "vanilla"), ("W1_V3", "static"), ("W1_V3", "adaptive"), ("W2", "vanilla")]
    assert [c["concurrency"] for c in order[0][2]] == [1, 8, 32]
    assert len(order) == 9


class _Fakes:
    """Injected container effects: vLLM processes, harness runs and /metrics samplers."""

    def __init__(self, *, harness_fail_keys=(), server_fail_lanes=(), running=None):
        self.harness_fail_keys = set(harness_fail_keys)
        self.server_fail_lanes = set(server_fail_lanes)
        self.running = running
        self.started: list[str] = []
        self.stopped = 0
        self.persisted = 0

    def start_server(self, cfg, server_dir):
        if cfg["lane"] in self.server_fail_lanes:
            raise RuntimeError("vLLM exited with code 1 (CUDA out of memory)")
        self.started.append(f"{cfg['workload']}/{cfg['lane']}")
        return {"cmd": ["vllm", "serve", "--enforce-eager", "--async-scheduling", "--no-enable-prefix-caching"],
                "vllm_version": "0.30.0", "torch_version": "2.13.0+cu130", "torch_cuda_version": "13.0",
                "gpu_memory_total_mb": 23028.0, "runtime_variant": "vllm-0.30.0-plus-backport-55708",
                "vllm_outputs_sha256": "f" * 64}

    def stop_server(self, server):
        self.stopped += 1

    def run_harness(self, args, env, log_path):
        raw_out = Path(args[args.index("--raw-out") + 1])
        key = raw_out.parent.name
        c = int(args[args.index("--concurrency") + 1])
        raw_out.write_text("".join(json.dumps(r) + "\n" for r in _raw(5)), encoding="utf-8")
        raw_out.with_name("warmup.jsonl").write_text('{"phase": "warmup"}\n' * 3, encoding="utf-8")
        self.last_concurrency = self.running if self.running is not None else c
        return 1 if any(k in key for k in self.harness_fail_keys) else 0

    def sample(self):
        fakes = self

        class _Sampler:
            def start(self):
                return self

            def stop(self):
                return _samples(5, running=fakes.last_concurrency)
        return _Sampler()

    def persist(self):
        self.persisted += 1


def _run(tmp_path, record, fakes, *, profile_text=None, device="NVIDIA A10", workloads=None):
    cells = [c for c in record["cells"] if workloads is None or c["workload"] in workloads]
    return sess.run_session(
        record, "A10", cells, out_root=tmp_path, device_name=device, gpu_uuid="GPU-1234", driver="610.57.04",
        profile_text=profile_text, call_id="fc-test", start_server=fakes.start_server,
        stop_server=fakes.stop_server, run_harness=fakes.run_harness, sample=fakes.sample,
        persist=fakes.persist)


def _profile_text(device: str) -> str:
    return json.dumps({"metadata": {"device": device, "raw_sha256": "a" * 64}})


def test_run_session_writes_cell_evidence_and_one_server_per_workload_lane(tmp_path):
    record = camp.plan("sess", "a" * 40, mode="fast", gpus="A10", workloads="W2")
    fakes = _Fakes()
    result = _run(tmp_path, record, fakes, profile_text=_profile_text("NVIDIA A10"))
    assert set(result["cells"].values()) == {"done"}
    assert fakes.started == ["W2/vanilla", "W2/static", "W2/adaptive"]
    assert fakes.stopped == 3 and fakes.persisted == 9
    cell_dir = tmp_path / "A10__static__W2__c8"
    for name in ("server_env.json", "raw.jsonl", "warmup.jsonl", "concurrency.jsonl", "raw.vram.jsonl",
                 "bench.json", "submission.json"):
        assert (cell_dir / name).is_file(), name
    env = json.loads((cell_dir / "server_env.json").read_text(encoding="utf-8"))
    assert (env["gpu"], env["gpu_uuid"], env["driver_version"]) == ("NVIDIA A10", "GPU-1234", "610.57.04")
    other = json.loads((tmp_path / "A10__static__W2__c32" / "server_env.json").read_text(encoding="utf-8"))
    vanilla = json.loads((tmp_path / "A10__vanilla__W2__c8" / "server_env.json").read_text(encoding="utf-8"))
    assert env["instance_id"] == other["instance_id"] != vanilla["instance_id"]
    assert json.loads((cell_dir / "bench.json").read_text(encoding="utf-8"))["concurrency_ok"] is True


def test_run_session_records_serialized_concurrency_as_not_ok(tmp_path):
    record = camp.plan("sess", "a" * 40, mode="fast", gpus="A10", workloads="W2", lanes_list="vanilla")
    result = _run(tmp_path, record, _Fakes(running=1))
    assert result["cells"]["A10__vanilla__W2__c8"] == "done"
    bench = json.loads((tmp_path / "A10__vanilla__W2__c8" / "bench.json").read_text(encoding="utf-8"))
    assert bench["concurrency_ok"] is False


def test_run_session_continues_after_harness_failure(tmp_path):
    record = camp.plan("sess", "a" * 40, mode="fast", gpus="A10", workloads="W2", lanes_list="vanilla")
    fakes = _Fakes(harness_fail_keys={"__c8"})
    result = _run(tmp_path, record, fakes)
    assert result["cells"] == {"A10__vanilla__W2__c1": "done", "A10__vanilla__W2__c8": "failed:harness",
                               "A10__vanilla__W2__c32": "done"}
    assert fakes.persisted == 3


def test_run_session_marks_lane_failed_when_server_does_not_start(tmp_path):
    record = camp.plan("sess", "a" * 40, mode="fast", gpus="A10", workloads="W2")
    result = _run(tmp_path, record, _Fakes(server_fail_lanes={"static"}), profile_text=_profile_text("NVIDIA A10"))
    statuses = {k.split("__")[1]: v for k, v in result["cells"].items()}
    assert statuses == {"vanilla": "done", "static": "failed:server", "adaptive": "done"}
    failure = json.loads((tmp_path / "A10__static__W2__c1" / "failure.json").read_text(encoding="utf-8"))
    assert "CUDA out of memory" in failure["error"]


def test_run_session_adaptive_profile_must_match_card(tmp_path):
    record = camp.plan("sess", "a" * 40, mode="fast", gpus="A10", workloads="W2", lanes_list="adaptive")
    fakes = _Fakes()
    result = _run(tmp_path, record, fakes, profile_text=_profile_text("NVIDIA A10G"))
    assert set(result["cells"].values()) == {"failed:profile"}
    assert fakes.started == []


def test_run_session_refuses_wrong_card_before_starting_anything(tmp_path):
    record = camp.plan("sess", "a" * 40, mode="fast", gpus="A10", workloads="W2")
    fakes = _Fakes()
    with pytest.raises(sess.WrongGpu):
        _run(tmp_path, record, fakes, device="NVIDIA A10G")
    assert fakes.started == []
