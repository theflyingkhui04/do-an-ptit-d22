"""Tests for FlexSDSClient and OpenAI-compatible SSE streaming."""
import json
import pytest
from flexsds.client.client import FlexSDSClient, EndpointUnreachableError


def test_client_handles_unreachable_endpoint():
    client = FlexSDSClient(base_url="http://127.0.0.1:59999/v1")
    assert client.check_health() is False


def test_client_get_models_mock(monkeypatch):
    fake_models_resp = json.dumps({"data": [{"id": "qwen2.5-7b-adaptive"}]}).encode("utf-8")
    class MockHttpResp:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return fake_models_resp

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=5: MockHttpResp())

    client = FlexSDSClient(base_url="http://localhost:8000/v1")
    models = client.get_models()
    assert "qwen2.5-7b-adaptive" in models


def test_client_stream_chat_parses_sse(monkeypatch):
    chunk1 = json.dumps({"choices": [{"delta": {"content": "Xin"}}]}).encode("utf-8")
    chunk2 = json.dumps({"choices": [{"delta": {"content": " chào"}}]}).encode("utf-8")
    fake_sse_lines = [
        b'data: ' + chunk1 + b'\n\n',
        b'data: ' + chunk2 + b'\n\n',
        b'data: [DONE]\n\n',
    ]
    class MockHttpResp:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self): return iter(fake_sse_lines)

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHttpResp())

    client = FlexSDSClient(base_url="http://localhost:8000/v1")
    tokens = list(client.stream_chat([{"role": "user", "content": "Hi"}], model="qwen2.5-7b-adaptive"))
    assert "".join(tokens) == "Xin chào"
