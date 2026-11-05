"""Shared Modal lane client: offline tests with a fake HTTP transport."""

from __future__ import annotations

import json

import pytest

from src.client import modal_client as mc


TOKEN = "wk-testid.ws-testsecret"


class FakeResponse:
    def __init__(self, status: int, body=None, lines=None):
        self.status = status
        self._body = json.dumps(body).encode() if isinstance(body, (dict, list)) else (body or b"")
        self._lines = lines or []

    def read(self) -> bytes:
        return self._body

    def lines(self):
        yield from self._lines


class FakeTransport:
    """Returns scripted responses in order; records every call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers,
                           "body": json.loads(body) if body else None})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def sse(*events: dict) -> list[str]:
    return [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]


def client(transport, lane="vanilla", **kwargs):
    clock = kwargs.pop("clock", FakeClock())
    return mc.ModalLaneClient(lane, token=TOKEN, transport=transport, clock=clock, sleep=clock.sleep, **kwargs)


# ---------------------------------------------------------------- addressing
def test_default_url_and_model_per_lane(monkeypatch):
    monkeypatch.delenv("PTIT_MODAL_URL_STATIC", raising=False)
    monkeypatch.delenv("PTIT_MODAL_WORKSPACE", raising=False)

    c = mc.ModalLaneClient("static", token=TOKEN)

    assert c.base_url == "https://nguyenduonghb940--ptit-vllm-static.us-east.modal.direct/v1"
    assert c.model == "qwen2.5-7b-static-draft"


def test_url_override_per_lane(monkeypatch):
    monkeypatch.setenv("PTIT_MODAL_URL_VANILLA", "https://example.test/v1/")

    assert mc.ModalLaneClient("vanilla", token=TOKEN).base_url == "https://example.test/v1"


def test_workspace_override(monkeypatch):
    monkeypatch.delenv("PTIT_MODAL_URL_ADAPTIVE", raising=False)
    monkeypatch.setenv("PTIT_MODAL_WORKSPACE", "otherws")

    assert mc.ModalLaneClient("adaptive", token=TOKEN).base_url.startswith("https://otherws--ptit-vllm-adaptive.")


def test_unknown_lane_rejected():
    with pytest.raises(ValueError, match="vanilla"):
        mc.ModalLaneClient("unknown_lane", token=TOKEN)


def test_removed_eagle_alias_cannot_route_to_an_undeployed_endpoint():
    with pytest.raises(ValueError, match="Unknown lane"):
        mc.ModalLaneClient("eagle-custom", token=TOKEN)


def test_lane_models_match_runner_config():
    from experiments import modal_lanes

    assert mc.LANE_MODELS == {lane: cfg.served_model_name for lane, cfg in modal_lanes.LANES.items()}


# ---------------------------------------------------------------- token
def test_load_token_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)
    env = tmp_path / ".env"
    env.write_text("MODAL_PROXY_TOKEN_ID=wk-a\nMODAL_PROXY_TOKEN_SECRET=ws-b\n", encoding="utf-8")

    assert mc.load_token(env) == "wk-a.ws-b"


def test_missing_token_names_variables(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)

    with pytest.raises(mc.ModalClientError, match="MODAL_PROXY_TOKEN_ID"):
        mc.ModalLaneClient("vanilla", env_file=tmp_path / "none.env")


def test_probe_reuses_shared_token_loader():
    from experiments import modal_probe

    assert modal_probe.load_token is mc.load_token


def test_bearer_header_sent():
    t = FakeTransport(FakeResponse(200, {"choices": [{"message": {"content": "hi"}}], "usage": {}}))

    client(t).chat([{"role": "user", "content": "x"}])

    assert t.calls[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_401_error_does_not_leak_token():
    t = FakeTransport(FakeResponse(401, b"missing credentials"))

    lane = client(t)
    with pytest.raises(mc.ModalAuthError) as err:
        lane.chat([{"role": "user", "content": "x"}])

    assert "ws-testsecret" not in str(err.value)
    assert "wk-testid" not in str(err.value)


# ---------------------------------------------------------------- chat / retry
def test_chat_returns_text_usage_and_payload():
    body = {"choices": [{"message": {"content": "Xin chào"}}], "usage": {"completion_tokens": 3},
            "metrics": {"speculative_decoding": {"num_spec_tokens": 3}}}
    t = FakeTransport(FakeResponse(200, body))

    r = client(t, lane="static").chat([{"role": "user", "content": "x"}], max_tokens=16, temperature=0)

    assert r.text == "Xin chào"
    assert r.usage == {"completion_tokens": 3}
    assert r.metrics["speculative_decoding"]["num_spec_tokens"] == 3
    sent = t.calls[0]["body"]
    assert sent["model"] == "qwen2.5-7b-static-draft"
    assert sent["max_tokens"] == 16
    assert t.calls[0]["url"].endswith("/v1/chat/completions")


def test_503_is_retried_until_ready():
    ok = FakeResponse(200, {"choices": [{"message": {"content": "ok"}}], "usage": {}})
    t = FakeTransport(FakeResponse(503), FakeResponse(503), ok)

    r = client(t, poll_s=5).chat([{"role": "user", "content": "x"}])

    assert r.text == "ok"
    assert len(t.calls) == 3


def test_503_times_out():
    clock = FakeClock()
    t = FakeTransport(FakeResponse(503))

    lane = client(t, clock=clock, poll_s=5, ready_timeout_s=20)
    with pytest.raises(mc.ModalReadyTimeout):
        lane.chat([{"role": "user", "content": "x"}])


def test_other_http_error_raises_client_error():
    t = FakeTransport(FakeResponse(400, b'{"error": "bad model"}'))

    lane = client(t)
    with pytest.raises(mc.ModalClientError, match="400"):
        lane.chat([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------- stream
def test_stream_yields_deltas_then_final_with_usage_and_metrics():
    lines = sse(
        {"choices": [{"delta": {"content": "Xin"}}]},
        {"choices": [{"delta": {"content": " chào"}}]},
        {"choices": [], "usage": {"completion_tokens": 2},
         "metrics": {"speculative_decoding": {"num_spec_tokens": 3}}},
    )
    t = FakeTransport(FakeResponse(200, lines=lines))

    events = list(client(t, lane="static").stream([{"role": "user", "content": "x"}], max_tokens=16))

    deltas = [e.text for e in events if e.kind == "delta"]
    final = events[-1]
    assert deltas == ["Xin", " chào"]
    assert final.kind == "final"
    assert final.text == "Xin chào"
    assert final.usage == {"completion_tokens": 2}
    assert final.metrics["speculative_decoding"]["num_spec_tokens"] == 3
    sent = t.calls[0]["body"]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


def test_stream_records_time_to_first_token():
    clock = FakeClock()
    t = FakeTransport(FakeResponse(200, lines=sse({"choices": [{"delta": {"content": "a"}}]})))

    final = list(client(t, clock=clock).stream([{"role": "user", "content": "x"}]))[-1]

    assert final.ttft_s is not None
    assert final.total_s >= final.ttft_s


# ---------------------------------------------------------------- logit margin
def lp(token, logprob, *top):
    return {"token": token, "logprob": logprob,
            "top_logprobs": [{"token": tk, "logprob": v} for tk, v in top]}


def test_margins_per_token():
    body = {"choices": [{"message": {"content": "ab"}, "logprobs": {"content": [
        lp("a", -0.1, ("a", -0.1), ("b", -2.6)),
        lp("b", -0.5, ("c", -0.9), ("b", -0.5)),  # unsorted top list
    ]}}]}
    t = FakeTransport(FakeResponse(200, body))

    margins = client(t).margins([{"role": "user", "content": "x"}], max_tokens=2)

    assert [m.token for m in margins] == ["a", "b"]
    assert margins[0].margin == pytest.approx(2.5)
    assert margins[1].margin == pytest.approx(0.4)
    assert margins[0].logprob == pytest.approx(-0.1)
    sent = t.calls[0]["body"]
    assert sent["logprobs"] is True
    assert sent["top_logprobs"] == 2
    assert sent["temperature"] == 0


def test_margin_tie_is_zero():
    body = {"choices": [{"message": {"content": "a"}, "logprobs": {"content": [
        lp("a", -0.69, ("a", -0.69), ("b", -0.69))]}}]}

    margins = client(FakeTransport(FakeResponse(200, body))).margins([{"role": "user", "content": "x"}])

    assert margins[0].margin == 0.0


@pytest.mark.parametrize("choice", [
    {"message": {"content": "a"}},
    {"message": {"content": "a"}, "logprobs": None},
    {"message": {"content": "a"}, "logprobs": {"content": [{"token": "a", "logprob": -0.1, "top_logprobs": []}]}},
])
def test_margin_unsupported_when_top_logprobs_missing(choice):
    t = FakeTransport(FakeResponse(200, {"choices": [choice]}))

    lane = client(t)
    with pytest.raises(mc.LogitMarginUnsupported):
        lane.margins([{"role": "user", "content": "x"}])
