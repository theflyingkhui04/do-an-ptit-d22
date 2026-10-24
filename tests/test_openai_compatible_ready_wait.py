"""OpenAICompatibleClient waits through 503 (Modal cold start) only when asked to."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from src.pipeline import openai_compatible as oc


class OkResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_503():
    return HTTPError("u", 503, "Service Unavailable", {}, io.BytesIO(b'{"error":"no upstreams available"}'))


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def scripted(*outcomes):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return OkResponse(json.dumps(outcome).encode())

    fake_urlopen.calls = calls
    return fake_urlopen


def make(**kwargs):
    clock = Clock()
    return oc.OpenAICompatibleClient("https://x.test/v1", "tok", clock=clock, sleep=clock.sleep, **kwargs)


def test_default_does_not_wait_on_503():
    opener = scripted(http_503(), {"data": [{"id": "m"}]})
    with patch.object(oc, "urlopen", opener):
        with pytest.raises(oc.OpenAICompatibleError, match="503"):
            make().list_models()
    assert len(opener.calls) == 1


def test_waits_through_503_when_ready_timeout_set():
    opener = scripted(http_503(), http_503(), {"data": [{"id": "m"}]})
    with patch.object(oc, "urlopen", opener):
        assert make(ready_timeout_seconds=600, poll_seconds=10).list_models() == ["m"]
    assert len(opener.calls) == 3


def test_ready_timeout_exceeded_reports_503():
    opener = scripted(http_503())
    with patch.object(oc, "urlopen", opener):
        with pytest.raises(oc.OpenAICompatibleError, match="503"):
            make(ready_timeout_seconds=30, poll_seconds=10).list_models()
    assert 3 <= len(opener.calls) <= 5


def test_other_http_errors_are_not_retried():
    opener = scripted(HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"no")), {"data": []})
    with patch.object(oc, "urlopen", opener):
        with pytest.raises(oc.OpenAICompatibleError, match="401"):
            make(ready_timeout_seconds=600).list_models()
    assert len(opener.calls) == 1


def test_node1_cli_exposes_ready_timeout(monkeypatch):
    sys.path.insert(0, "scripts")
    import run_node1_extractor as node1

    monkeypatch.setattr(sys, "argv", ["run_node1_extractor.py", "--ready-timeout", "1200"])
    assert node1.parse_args().ready_timeout == 1200.0
