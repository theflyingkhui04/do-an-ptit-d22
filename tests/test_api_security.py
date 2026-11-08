"""[M5-01 web demo] Task 4.1: demo key, rate limits, CORS and secret redaction."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from src.api.security import RateLimiter, RedactingJsonFormatter
from src.api.settings import Settings
from tests.api_fakes import DEMO_KEY, HEADERS, SECRET_URL, FakeStore, build, jpeg_bytes


def test_status_needs_no_key_but_everything_else_does():
    app, _, _ = build()
    with TestClient(app) as client:
        assert client.get("/api/status").json()["backend"] == "ready"
        for method, path in (("get", "/api/invoices"), ("get", "/api/jobs/up_000000000000"),
                             ("post", "/api/warmup")):
            assert getattr(client, method)(path).status_code == 401
            assert getattr(client, method)(path, headers={"X-Demo-Key": "wrong-key-000"}).status_code == 401
        assert client.post("/api/questions", json={"question": "x"}).status_code == 401
        assert client.get("/api/invoices", headers=HEADERS).status_code == 200


def test_eleventh_upload_in_ten_minutes_is_429_with_retry_after():
    now = [0.0]
    app, _, _ = build(clock=lambda: now[0])
    with TestClient(app) as client:
        for _ in range(10):
            r = client.post("/api/invoices", headers=HEADERS, files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
            assert r.status_code == 202
        r = client.post("/api/invoices", headers=HEADERS, files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) == 600
        now[0] = 601.0
        r = client.post("/api/invoices", headers=HEADERS, files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
        assert r.status_code == 202


def test_rate_limit_is_per_ip():
    limiter = RateLimiter(1, 600, clock=lambda: 0.0)
    assert limiter.hit("1.1.1.1") is None
    assert limiter.hit("1.1.1.1") == 600
    assert limiter.hit("2.2.2.2") is None


def test_cors_allows_only_the_frontend_origin():
    app, _, _ = build()
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-demo-key"}
    with TestClient(app) as client:
        ok = client.options("/api/invoices", headers={"Origin": "https://demo.vercel.app", **preflight})
        assert ok.headers.get("access-control-allow-origin") == "https://demo.vercel.app"
        bad = client.options("/api/invoices", headers={"Origin": "https://evil.test", **preflight})
        assert "access-control-allow-origin" not in bad.headers


def test_internal_errors_hide_secrets_from_response_and_logs(caplog):
    class BrokenStore(FakeStore):
        def list_invoices(self, limit=200):
            raise RuntimeError(f"cannot connect to {SECRET_URL}")

    app, _, _ = build(store=BrokenStore())
    with caplog.at_level(logging.ERROR, logger="src.api"), TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/api/invoices", headers=HEADERS)
    assert r.status_code == 500
    assert "pw-very-secret" not in r.text
    assert "pw-very-secret" not in caplog.text
    assert "***" in caplog.text


def test_json_log_formatter_redacts_configured_secrets():
    settings = Settings(demo_key=DEMO_KEY, writer_url=SECRET_URL, modal_token="tok-id-123.tok-secret-456")
    formatter = RedactingJsonFormatter(settings.secret_values())
    record = logging.LogRecord("x", logging.INFO, __file__, 1, f"key {DEMO_KEY} token tok-secret-456", None, None)
    line = formatter.format(record)
    assert DEMO_KEY not in line and "tok-secret-456" not in line and line.count("***") == 2


def test_settings_require_a_demo_key_and_valid_lanes():
    import pytest

    with pytest.raises(ValueError, match="DEMO_KEY"):
        Settings.from_env({})
    with pytest.raises(ValueError, match="lane"):
        Settings.from_env({"DEMO_KEY": DEMO_KEY, "DEMO_LANE_NODE1": "turbo"})
    settings = Settings.from_env({"DEMO_KEY": DEMO_KEY, "DEMO_FRONTEND_ORIGINS": "https://a.app/, https://b.app",
                                  "MODAL_PROXY_TOKEN_ID": "id1", "MODAL_PROXY_TOKEN_SECRET": "sec1"})
    assert settings.frontend_origins == ("https://a.app", "https://b.app")
    assert settings.modal_token == "id1.sec1"
    assert settings.lane_map == {"node1": "adaptive", "node2": "static", "answer": "static", "report": "eagle"}
