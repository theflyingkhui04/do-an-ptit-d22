"""[M5-01 web demo] Streaming model client: OpenAI-shaped response + per-call speed metrics."""

from __future__ import annotations

import json

import httpx
import pytest

from src.api.llm import LaneCaller, LLMError
from src.pipeline.openai_compatible import OpenAICompatibleError

TOKEN = "tok-id.tok-secret"


def sse(*chunks: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


class Clock:
    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


def caller(handler, clock=None, **kwargs) -> LaneCaller:
    return LaneCaller("adaptive", "https://x.test/v1/", TOKEN, "qwen2.5-7b-adaptive",
                      client=httpx.Client(transport=httpx.MockTransport(handler)),
                      clock=clock or Clock(0.0), sleep=lambda s: None, **kwargs)


def test_streamed_answer_is_returned_as_a_chat_completion_with_metrics():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"content": '{"a": '}}]},
            {"choices": [{"delta": {"content": "1}"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 11}}))

    # start, first token at 0.5 s, end at 1.5 s
    lane = caller(handler, clock=Clock(0.0, 0.5, 1.5))
    response = lane.chat_completions({"messages": [{"role": "user", "content": "x"}], "max_tokens": 64})

    assert seen["auth"] == f"Bearer {TOKEN}"
    assert seen["body"]["model"] == "qwen2.5-7b-adaptive" and seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert response["choices"][0]["message"]["content"] == '{"a": 1}'
    assert response["choices"][0]["finish_reason"] == "stop"
    metrics = lane.last_metrics
    assert metrics.lane == "adaptive" and metrics.completion_tokens == 11
    assert metrics.latency_ms == pytest.approx(1500.0)
    assert metrics.ttft_ms == pytest.approx(500.0)
    assert metrics.tpot_ms == pytest.approx(100.0)          # (1500 - 500) / (11 - 1)
    assert metrics.output_tokens_per_s == pytest.approx(11 / 1.5)
    assert metrics.finish_reason == "stop"


def test_cold_start_503_is_retried_until_ready():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))

    assert caller(handler).chat_completions({"messages": []})["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 3


def test_errors_are_transport_errors_without_the_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad {TOKEN}")

    with pytest.raises(LLMError) as info:
        caller(handler).chat_completions({"messages": []})
    assert isinstance(info.value, OpenAICompatibleError)
    assert TOKEN not in str(info.value) and "401" in str(info.value) and "adaptive" in str(info.value)


def test_ready_timeout():
    with pytest.raises(LLMError, match="not ready"):
        caller(lambda r: httpx.Response(503), clock=Clock(0.0, 0.0, 2000.0), ready_timeout_s=1200).chat_completions(
            {"messages": []})


def test_single_token_answer_has_no_tpot():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}))

    lane = caller(handler, clock=Clock(0.0, 0.2, 0.2))
    lane.chat_completions({"messages": []})
    assert lane.last_metrics.tpot_ms is None


def test_connection_reset_during_cold_start_is_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ReadError("connection reset by peer")
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))

    assert caller(handler).chat_completions({"messages": []})["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 3


def test_reset_after_the_stream_started_is_not_retried():
    calls = []

    def broken_stream():
        yield b'data: {"choices": [{"delta": {"content": "{\\"a\\""}}]}\n\n'
        raise httpx.ReadError("connection reset mid-stream")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=broken_stream())

    with pytest.raises(LLMError, match="stream interrupted"):
        caller(handler).chat_completions({"messages": []})
    assert len(calls) == 1


def test_connection_errors_stop_after_the_ready_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="cannot reach"):
        caller(handler, clock=Clock(0.0, 0.0, 2000.0), ready_timeout_s=1200).chat_completions({"messages": []})
