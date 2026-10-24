"""Client-side probe for the Modal lane servers: offline tests with fake HTTP."""

from __future__ import annotations

import json

import pytest

from experiments import modal_probe as probe


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeHttp:
    """Scripted responses per (method, path); last response repeats."""

    def __init__(self, routes, clock=None, cost=0.0):
        self.routes = {key: list(value) for key, value in routes.items()}
        self.calls = []
        self.clock = clock
        self.cost = cost

    def __call__(self, method, url, headers=None, body=None, timeout=30.0):
        path = "/" + url.split("/", 3)[3] if url.count("/") >= 3 else "/"
        self.calls.append((method, path, dict(headers or {}), body))
        if self.clock is not None:
            self.clock.now += self.cost
        queue = self.routes[(method, path)]
        status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(payload, Exception):
            raise payload
        return status, json.dumps(payload).encode() if payload is not None else b""


BASE = "https://ws--ptit-vllm-vanilla.modal.run"
MODELS_OK = (200, {"data": [{"id": "qwen2.5-7b-vanilla"}]})
CHAT_OK = (200, {"choices": [{"message": {"content": "Xin chào"}}], "usage": {"completion_tokens": 3}})


def test_load_token_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nMODAL_PROXY_TOKEN_ID=wk-abc\nMODAL_PROXY_TOKEN_SECRET=ws-def\n", encoding="utf-8")

    assert probe.load_token(env) == "wk-abc.ws-def"


def test_load_token_prefers_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-env")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-env")

    assert probe.load_token(tmp_path / "missing.env") == "wk-env.ws-env"


def test_load_token_missing_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="MODAL_PROXY_TOKEN_ID"):
        probe.load_token(tmp_path / "missing.env")


def test_wait_ready_retries_through_503_and_connection_errors():
    clock = FakeClock()
    http = FakeHttp(
        {("GET", "/health"): [(503, None), (0, ConnectionError("reset")), (503, None), (200, None)]},
        clock=clock,
    )

    result = probe.wait_ready(BASE, {}, timeout_s=60, poll_s=5, request=http, clock=clock, sleep=clock.sleep)

    assert result["status"] == "ok"
    assert result["attempts"] == 4
    assert result["first_status"] == 503
    assert result["seconds_to_ready"] == pytest.approx(15)


def test_wait_ready_times_out():
    clock = FakeClock()
    http = FakeHttp({("GET", "/health"): [(503, None)]}, clock=clock)

    result = probe.wait_ready(BASE, {}, timeout_s=20, poll_s=5, request=http, clock=clock, sleep=clock.sleep)

    assert result["status"] == "timeout"
    assert result["last_status"] == 503


def test_check_lane_ok_and_unauthenticated_rejected():
    http = FakeHttp(
        {
            ("GET", "/health"): [(200, None)],
            ("GET", "/v1/models"): [(401, None), MODELS_OK],
            ("POST", "/v1/chat/completions"): [CHAT_OK],
        }
    )

    result = probe.check_lane("vanilla", BASE, "wk-x.ws-y", request=http, timeout_s=10, poll_s=1)

    assert result == {
        "lane": "vanilla",
        "ok": True,
        "health": 200,
        "unauthenticated_status": 401,
        "models": ["qwen2.5-7b-vanilla"],
        "chat_status": 200,
        "chat_content_nonempty": True,
    }
    authed = [c for c in http.calls if c[1] == "/v1/chat/completions"][0]
    assert authed[2]["Authorization"] == "Bearer wk-x.ws-y"
    sent = json.loads(authed[3])
    assert sent["model"] == "qwen2.5-7b-vanilla"
    assert sent["temperature"] == 0


def test_check_lane_fails_when_auth_not_enforced():
    http = FakeHttp(
        {
            ("GET", "/health"): [(200, None)],
            ("GET", "/v1/models"): [MODELS_OK],
            ("POST", "/v1/chat/completions"): [CHAT_OK],
        }
    )

    result = probe.check_lane("vanilla", BASE, "wk-x.ws-y", request=http, timeout_s=10, poll_s=1)

    assert result["ok"] is False
    assert result["unauthenticated_status"] == 200


def test_check_lane_fails_on_wrong_model_name():
    http = FakeHttp(
        {
            ("GET", "/health"): [(200, None)],
            ("GET", "/v1/models"): [(401, None), (200, {"data": [{"id": "qwen2.5-7b-adaptive"}]})],
            ("POST", "/v1/chat/completions"): [CHAT_OK],
        }
    )

    assert probe.check_lane("vanilla", BASE, "t", request=http, timeout_s=10, poll_s=1)["ok"] is False


def test_measure_startup_cold_record_has_contract_fields():
    clock = FakeClock()
    http = FakeHttp(
        {
            ("GET", "/health"): [(503, None), (503, None), (200, None)],
            ("POST", "/v1/chat/completions"): [CHAT_OK],
        },
        clock=clock,
        cost=1.0,
    )

    record = probe.measure_startup(
        "static", BASE, "t", kind="cold", profile="l4-bf16", commit="abc",
        request=http, clock=clock, sleep=clock.sleep, timeout_s=600, poll_s=2,
    )

    for field in (
        "lane", "gpu", "vllm_version", "dtype", "quantization", "kind",
        "seconds_to_ready", "first_response_seconds", "timestamp_utc", "commit",
    ):
        assert field in record, field
    assert record["kind"] == "cold"
    assert record["observed_cold"] is True
    assert record["dtype"] == "bfloat16"
    assert record["quantization"] == "awq_marlin"
    assert record["vllm_version"] == "0.30.0"
    assert 0 < record["seconds_to_ready"] < record["first_response_seconds"]
    assert record["status"] == "ok"


def test_measure_startup_timeout_record():
    clock = FakeClock()
    http = FakeHttp({("GET", "/health"): [(503, None)]}, clock=clock)

    record = probe.measure_startup(
        "vanilla", BASE, "t", kind="cold", profile="fp16-fallback", commit="abc",
        request=http, clock=clock, sleep=clock.sleep, timeout_s=10, poll_s=5,
    )

    assert record["status"] == "timeout"
    assert record["first_response_seconds"] is None


def test_summarize_reports_median_per_lane_and_kind():
    records = [
        {"lane": "vanilla", "kind": "cold", "status": "ok", "seconds_to_ready": s, "first_response_seconds": s + 1}
        for s in (30.0, 10.0, 20.0)
    ]

    summary = probe.summarize(records)

    assert summary == [
        {"lane": "vanilla", "kind": "cold", "n_ok": 3, "median_seconds_to_ready": 20.0,
         "median_first_response_seconds": 21.0}
    ]


def test_token_never_written_to_records():
    clock = FakeClock()
    http = FakeHttp(
        {("GET", "/health"): [(200, None)], ("POST", "/v1/chat/completions"): [CHAT_OK]}, clock=clock
    )

    record = probe.measure_startup(
        "vanilla", BASE, "wk-secret.ws-secret", kind="warm", profile="l4-bf16", commit="abc",
        request=http, clock=clock, sleep=clock.sleep, timeout_s=10, poll_s=1,
    )

    assert "ws-secret" not in json.dumps(record)


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "..", "", r"x\y"])
def test_evidence_path_rejects_unsafe_run_id(run_id):
    with pytest.raises(ValueError, match="run-id"):
        probe._evidence_path(run_id, "check.json")


def test_evidence_path_stays_under_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "REPO_ROOT", tmp_path)

    path = probe._evidence_path("20261025-m300.ok_1", "check.json")

    assert path == tmp_path / "experiments" / "runs" / "m3_00" / "20261025-m300.ok_1" / "check.json"
    assert path.parent.is_dir()


@pytest.mark.parametrize(
    "extra",
    [["--run-id", "../escape"], ["--run-id", "ok", "--env-file", "OUTSIDE"]],
    ids=["bad-run-id", "env-file-outside-repo"],
)
def test_main_rejects_unsafe_paths_before_network(tmp_path, monkeypatch, extra):
    called = []
    monkeypatch.setattr(probe, "resolve_url", lambda lane: called.append(lane))
    args = [str(tmp_path / ".env") if a == "OUTSIDE" else a for a in extra]

    assert probe.main(["check", "--lanes", "vanilla", *args]) == 2
    assert called == []
