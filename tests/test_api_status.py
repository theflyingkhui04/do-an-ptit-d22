"""[M5-01 web console] Status without a key: backend, database and how many pasted endpoints answer.

The web never starts GPUs and never polls endpoints in the background (each call can wake a GPU on Modal),
so the status only reports what the registry last saw.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api_fakes import HEADERS, FakeStore, build, endpoint


def test_status_reports_database_and_endpoints_without_a_key():
    store = FakeStore()
    rows = [endpoint("ep_0000000a", "vanilla"), endpoint("ep_0000000b", "flexsds-serve", status="down")]
    app, _, _ = build(store=store, endpoints=rows)
    with TestClient(app) as client:
        body = client.get("/api/status").json()
        assert body == {"backend": "ready", "database": "ready", "endpoints": {"total": 2, "ok": 1, "down": 1}}
        store.ping_ok = False
        body = client.get("/api/status").json()
        assert body["database"] == "error" and body["endpoints"] == {"total": 0, "ok": 0, "down": 0}


def test_first_run_has_no_endpoint():
    app, _, _ = build(endpoints=[])
    with TestClient(app) as client:
        assert client.get("/api/status").json()["endpoints"] == {"total": 0, "ok": 0, "down": 0}
        assert client.get("/api/endpoints", headers=HEADERS).json() == {"endpoints": []}


def test_warmup_route_is_gone():
    app, _, _ = build()
    with TestClient(app) as client:
        assert client.post("/api/warmup", headers=HEADERS).status_code in (404, 405)


def test_invoice_list():
    app, _, _ = build()
    with TestClient(app) as client:
        assert client.get("/api/invoices", headers=HEADERS).json() == {"invoices": []}
