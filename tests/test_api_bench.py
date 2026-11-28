"""[M5-01 web console] Measurements: frozen workload requests on n endpoints at once, and the 1→2→3 chain."""

from __future__ import annotations

import json

import httpx
import pytest

from src.api.bench import BenchService, load_requests, speedups, summarize
from tests.api_fakes import FakeStore, endpoint


def sse(text_tokens: int) -> bytes:
    chunks = [{"choices": [{"delta": {"content": "x"}, "finish_reason": None}]} for _ in range(text_tokens)]
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": text_tokens}})
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


class Registry:
    def __init__(self, rows: list[dict], baseline: str | None = None) -> None:
        self.rows = {r["endpoint_id"]: r for r in rows}
        self._baseline = baseline

    def get(self, endpoint_id, *, with_key=False):
        return {**self.rows[endpoint_id], "api_key": "k" if with_key else None}

    def baseline(self):
        return self.rows.get(self._baseline)

    def caller(self, endpoint_id):
        return endpoint_id


def test_frozen_workload_requests_cycle_and_drop_the_model_field():
    w1 = load_requests("W1", 32)
    assert len(w1) == 32 and w1[30] == w1[0] and "model" not in w1[0]
    assert w1[0]["max_tokens"] == 3072 and w1[0]["response_format"] == {"type": "json_object"}
    assert len(load_requests("W2", 5)) == 5 and load_requests("W3", 1)[0]["max_tokens"] == 768
    with pytest.raises(ValueError):
        load_requests("W9", 1)


def test_summary_and_speedup_follow_the_campaign_definition():
    samples = [{"ok": True, "ttft_ms": 50, "tpot_ms": 20, "latency_ms": 1000, "completion_tokens": 50},
               {"ok": True, "ttft_ms": 70, "tpot_ms": 24, "latency_ms": 1300, "completion_tokens": 50},
               {"ok": False, "error": "HTTP 500"}]
    cell = summarize(samples, wall_ms=2000)
    assert cell == {"n": 3, "n_ok": 2, "ttft_ms_p50": 60.0, "tpot_ms_p50": 22.0, "latency_ms_p50": 1150.0,
                    "throughput_tps": 50.0, "errors": ["HTTP 500"]}
    cells = [{"endpoint_id": "ep_base", "load": 1, "tpot_ms_p50": 20.0}, {"endpoint_id": "ep_fast", "load": 1, "tpot_ms_p50": 14.0},
             {"endpoint_id": "ep_fast", "load": 8, "tpot_ms_p50": 30.0}]
    speedups(cells, "ep_base")
    assert [c["speedup_vs_baseline"] for c in cells] == [1.0, pytest.approx(20 / 14), None]


def service(store, registry, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":  # the wake-up check before measuring
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        body = json.loads(request.content)
        assert body["stream"] is True and body["model"].startswith("model-")
        return httpx.Response(200, content=sse(5), headers={"content-type": "text/event-stream"})

    kwargs.setdefault("spawn", lambda target, args: target(*args))
    return BenchService(store, registry, async_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                        new_id=lambda: "bn_000000000001", **kwargs)


def test_speed_run_measures_every_endpoint_at_every_load():
    store = FakeStore()
    rows = [endpoint("ep_base", "vanilla"), endpoint("ep_fast", "flexsds-serve")]
    for row in rows:
        row["model_id"] = f"model-{row['endpoint_id']}"
    bench = service(store, Registry(rows, baseline="ep_base"))
    run_id = bench.start({"kind": "speed", "endpoint_ids": ["ep_base", "ep_fast"], "workload": "W2", "loads": [1, 2],
                          "requests": 3, "repeats": 1})
    run = bench.get(run_id)
    assert run["status"] == "done" and run["progress"] == {"done": 12, "total": 12}
    cells = run["result"]["cells"]
    assert [(c["endpoint_id"], c["load"], c["n_ok"]) for c in cells] == [
        ("ep_base", 1, 3), ("ep_fast", 1, 3), ("ep_base", 2, 3), ("ep_fast", 2, 3)]
    assert run["result"]["baseline"] == "ep_base" and run["result"]["live"] is True
    assert all(c["speedup_vs_baseline"] is not None for c in cells)


def test_only_one_measurement_at_a_time():
    store = FakeStore()
    bench = service(store, Registry([endpoint("ep_a", "a")]), spawn=lambda target, args: None)
    bench.start({"kind": "speed", "endpoint_ids": ["ep_a"], "workload": "W1", "loads": [1], "requests": 1, "repeats": 1})
    with pytest.raises(RuntimeError, match="Đang có một lần đo"):
        bench.start({"kind": "speed", "endpoint_ids": ["ep_a"], "workload": "W1", "loads": [1], "requests": 1,
                     "repeats": 1})


class ChainPipeline:
    def read_text(self, data, result):
        assert data.startswith(b"\xff\xd8")
        result["timings_ms"]["ocr"] = 900.0
        return "MINIMART ANAN"

    def run_endpoint(self, job_id, storage_path, text, ep):
        from types import SimpleNamespace

        payload = {"seller": {"name": "MINIMART ANAN"},
                   "totals": {"total_amount": {"value": 42000 if ep["endpoint_id"] != "ep_bad" else 1}}}
        success = SimpleNamespace(canonical_payload=payload, latency_ms=1000.0)
        return {"success": success, "report": {"endpoint_id": ep["endpoint_id"], "name": ep["name"], "valid": True,
                                                 "latency_ms": 1000.0, "tpot_ms": 20.0}}


class ChainReports:
    def run(self, caller):
        return {"questions": [{"question_id": "Q1", "status": "ok", "rows": [[42000]]}],
                "report": {"status": "ok", "text": "Tổng 42.000 đ" if caller != "ep_bad" else "Khác"},
                "number_check": {"numbers": 1, "ungrounded": []}, "timings_ms": {"node2": 6000.0, "node3": 4000.0}}


def test_chain_runs_node1_then_q1_q6_then_the_report_on_each_endpoint():
    store = FakeStore()
    rows = [endpoint("ep_base", "vanilla"), endpoint("ep_bad", "khac")]
    bench = service(store, Registry(rows, baseline="ep_base"), pipeline=ChainPipeline(), reports=ChainReports())
    run_id = bench.start({"kind": "chain", "endpoint_ids": ["ep_base", "ep_bad"], "sample_id": "m0_04_001"})
    run = bench.get(run_id)
    assert run["status"] == "done", run["error"]
    by_id = {r["endpoint_id"]: r for r in run["result"]["endpoints"]}
    assert by_id["ep_base"]["timings_ms"]["node2"] == 6000.0 and by_id["ep_base"]["total_ms"] > 0
    assert by_id["ep_bad"]["same_as_baseline"] == {"node1": False, "node2": True, "node3": False}
    assert by_id["ep_base"]["same_as_baseline"] is None and run["result"]["ocr_ms"] == 900.0


def _waking_bench(store, rows, models_answers, **kwargs):
    """Bench whose endpoints answer /models with the given statuses (one per call) before they are ready."""
    seen: list[str] = []
    answers = list(models_answers)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.method == "GET":
            return httpx.Response(answers.pop(0) if answers else 200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, content=sse(5), headers={"content-type": "text/event-stream"})

    bench = BenchService(store, Registry(rows), async_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                         new_id=lambda: "bn_000000000002", spawn=lambda target, args: target(*args),
                         asleep=lambda s: _noop(), **kwargs)
    return bench, seen


async def _noop():
    return None


def test_speed_run_wakes_sleeping_endpoints_before_measuring():
    """A Modal GPU asleep answers 303/503 for minutes; requests sent then would all fail or time the cold start."""
    store = FakeStore()
    rows = [endpoint("ep_a", "a")]
    rows[0]["model_id"] = "model-a"
    bench, seen = _waking_bench(store, rows, [303, 503])
    run = bench.get(bench.start({"kind": "speed", "endpoint_ids": ["ep_a"], "workload": "W2", "loads": [1],
                                 "requests": 2, "repeats": 1}))
    assert run["status"] == "done"
    assert seen[:3] == ["GET /v1/models"] * 3 and seen[3:] == ["POST /v1/chat/completions"] * 2
    assert run["result"]["cells"][0]["n_ok"] == 2


def test_speed_run_fails_when_an_endpoint_never_wakes():
    store = FakeStore()
    rows = [endpoint("ep_a", "a")]
    rows[0]["model_id"] = "model-a"
    ticks = iter(range(0, 100000, 400))
    bench, seen = _waking_bench(store, rows, [503] * 50, clock=lambda: float(next(ticks)))
    run = bench.get(bench.start({"kind": "speed", "endpoint_ids": ["ep_a"], "workload": "W2", "loads": [1],
                                 "requests": 2, "repeats": 1}))
    assert run["status"] == "failed" and "chưa thức dậy" in run["error"]
    assert "POST /v1/chat/completions" not in seen
