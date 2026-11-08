"""[M5-01 web demo] Task 4.4: service status, warm-up and bounded keep-alive."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.warmup import WarmupManager
from tests.api_fakes import HEADERS, FakeStore, build


class Probe:
    def __init__(self, error: str | None = None) -> None:
        self.calls = 0
        self.error = error

    def __call__(self) -> None:
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)


def manager(now, **probes):
    return WarmupManager(probes, clock=lambda: now[0], background=False, redact=lambda s: s.replace("tok", "***"))


def test_cold_services_become_ready_after_warmup():
    now = [0.0]
    ocr, adaptive = Probe(), Probe()
    warm = manager(now, ocr=ocr, adaptive=adaptive)
    assert {v["state"] for v in warm.snapshot().values()} == {"cold"}
    snapshot = warm.request()
    assert snapshot == {"ocr": {"state": "ready", "message": None}, "adaptive": {"state": "ready", "message": None}}
    assert (ocr.calls, adaptive.calls) == (1, 1)


def test_one_failing_lane_is_reported_alone():
    now = [0.0]
    warm = manager(now, ocr=Probe(), eagle=Probe(error="HTTP 500 tok-secret"))
    snapshot = warm.request()
    assert snapshot["ocr"]["state"] == "ready"
    assert snapshot["eagle"]["state"] == "error"
    assert "tok-secret" not in snapshot["eagle"]["message"] and "HTTP 500" in snapshot["eagle"]["message"]


def test_keepalive_runs_every_four_minutes_and_stops_after_sixty():
    now = [0.0]
    probe = Probe()
    warm = manager(now, ocr=probe)
    warm.request()
    now[0] = 100.0
    assert warm.tick() is False
    now[0] = 241.0
    assert warm.tick() is True
    calls_inside = probe.calls
    now[0] = 3601.0
    assert warm.tick() is False
    now[0] = 4000.0
    assert warm.tick() is False
    assert probe.calls == calls_inside


def test_mark_records_real_call_outcomes():
    now = [0.0]
    warm = manager(now, adaptive=Probe())
    warm.mark("adaptive", False, "HTTP 503")
    assert warm.snapshot()["adaptive"] == {"state": "error", "message": "HTTP 503"}
    warm.mark("unknown", True)  # ignored
    assert set(warm.snapshot()) == {"adaptive"}


def test_status_reports_database_and_services_without_a_key():
    store = FakeStore()
    app, _, _ = build(store=store)
    with TestClient(app) as client:
        body = client.get("/api/status").json()
        assert body["database"] == "ready" and body["gpu"] == "A10G"
        assert body["services"]["database"]["state"] == "ready"
        assert body["lane_map"]["node1"] == "adaptive"
        store.ping_ok = False
        body = client.get("/api/status").json()
        assert body["database"] == "error" and body["services"]["database"]["state"] == "error"


def test_warmup_endpoint_requires_the_key():
    app, _, _ = build()
    with TestClient(app) as client:
        assert client.post("/api/warmup").status_code == 401
        body = client.post("/api/warmup", headers=HEADERS).json()
    assert body["services"]["database"]["state"] == "ready" and body["max_seconds"] == 3600.0


def test_invoice_list():
    app, _, _ = build()
    with TestClient(app) as client:
        assert client.get("/api/invoices", headers=HEADERS).json() == {"invoices": []}
