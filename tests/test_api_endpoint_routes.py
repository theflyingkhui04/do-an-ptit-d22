"""[M5-01 web console] /api/endpoints: paste, detect, edit, baseline, check, delete (design D2, D14)."""

from __future__ import annotations

import socket

import httpx
from fastapi.testclient import TestClient

from src.api.app import Deps, create_app
from src.api.endpoints import EndpointRegistry
from src.api.settings import Settings
from tests.api_fakes import DEMO_KEY, HEADERS, FakeStore

MODAL = "https://canduckhoi--flexsds-serve.modal.run/v1"
COLAB = "https://swift-orchid-ptit.trycloudflare.com/v1"


def app_with_registry(down=()):
    store = FakeStore()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host in down:
            raise httpx.ConnectError("refused")
        if request.url.host == "llm.other-team.example.com" and request.headers.get("authorization") != "Bearer k-2":
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen2.5-7B-Instruct-AWQ", "max_model_len": 4096}]})

    def public_dns(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.16.230.132", port))]

    registry = EndpointRegistry(store, httpx.Client(transport=httpx.MockTransport(handler)), resolve=public_dns)
    settings = Settings(demo_key=DEMO_KEY, writer_url="postgresql://x")
    return create_app(settings, Deps(store=store, endpoints=registry, pipeline=None)), store


def test_paste_detect_and_save_from_the_cli_output():
    app, _store = app_with_registry()
    with TestClient(app) as client:
        detected = client.post("/api/endpoints/detect", headers=HEADERS,
                               json={"urls": [MODAL, "http://127.0.0.1:8000/v1"]}).json()["results"]
        assert [(r["ok"], r["code"]) for r in detected] == [(True, None), (False, "needs_tunnel")]
        assert client.get("/api/endpoints", headers=HEADERS).json() == {"endpoints": []}
        added = client.post("/api/endpoints", headers=HEADERS,
                            json={"endpoints": [{"url": MODAL, "name": "adaptive L4", "api_key": "ak-1.as-2"},
                                                {"url": COLAB}]}).json()
        assert [(e["platform"], e["name"], e["has_key"]) for e in added["saved"]] == [
            ("modal", "adaptive L4", True), ("tunnel", "Qwen/Qwen2.5-7B-Instruct-AWQ", False)]
        listed = client.get("/api/endpoints", headers=HEADERS).json()["endpoints"]
        again = client.post("/api/endpoints/detect", headers=HEADERS, json={"urls": [MODAL]}).json()["results"][0]
    assert len(listed) == 2 and all("api_key" not in e for e in listed)
    assert (again["code"], again["name"], again["endpoint_id"]) == ("exists", "adaptive L4", listed[0]["endpoint_id"])


def test_rename_baseline_check_and_delete():
    down: set[str] = set()
    app, _ = app_with_registry(down=down)
    with TestClient(app) as client:
        saved = client.post("/api/endpoints", headers=HEADERS, json={"endpoints": [{"url": MODAL}, {"url": COLAB}]}).json()["saved"]
        down.add("swift-orchid-ptit.trycloudflare.com")  # the Colab session stops after it was added
        a, b = (e["endpoint_id"] for e in saved)
        renamed = client.patch(f"/api/endpoints/{a}", headers=HEADERS, json={"name": "Modal L4"}).json()
        assert renamed["name"] == "Modal L4"
        assert client.patch(f"/api/endpoints/{a}", headers=HEADERS, json={"is_baseline": True}).json()["is_baseline"]
        client.patch(f"/api/endpoints/{b}", headers=HEADERS, json={"is_baseline": True})
        flags = {e["endpoint_id"]: e["is_baseline"] for e in client.get("/api/endpoints", headers=HEADERS).json()["endpoints"]}
        assert flags == {a: False, b: True}
        assert client.patch(f"/api/endpoints/{b}", headers=HEADERS, json={"is_baseline": False}).json()["is_baseline"] is False
        checked = client.post(f"/api/endpoints/{b}/check", headers=HEADERS).json()
        assert checked["status"] == "down" and "flexsds serve" in checked["status_message"]
        assert client.delete(f"/api/endpoints/{a}", headers=HEADERS).json() == {"deleted": a}
        assert client.delete(f"/api/endpoints/{a}", headers=HEADERS).status_code == 404
        assert client.patch("/api/endpoints/ep_0000ffff", headers=HEADERS, json={"name": "x"}).status_code == 404
        status = client.get("/api/status").json()["endpoints"]
    assert status == {"total": 1, "ok": 0, "down": 1}


def test_paste_limits():
    app, _ = app_with_registry()
    with TestClient(app) as client:
        assert client.post("/api/endpoints", headers=HEADERS, json={"endpoints": []}).status_code == 422
        assert client.post("/api/endpoints", headers=HEADERS, json={"endpoints": [{"url": MODAL}] * 11}).status_code == 422
        assert client.post("/api/endpoints", headers=HEADERS, json={"endpoints": [{"url": MODAL, "name": "x" * 81}]}).status_code == 422
        assert client.post("/api/endpoints", headers=HEADERS, json={"urls": [MODAL]}).status_code == 422  # the old body


def test_edit_address_key_and_notes():
    app, store = app_with_registry(down={"dead--flexsds-serve.modal.run"})
    with TestClient(app) as client:
        ep = client.post("/api/endpoints", headers=HEADERS, json={"endpoints": [{"url": MODAL}]}).json()["saved"][0]
        other = client.post("/api/endpoints", headers=HEADERS, json={"endpoints": [{"url": COLAB}]}).json()["saved"][0]
        path = f"/api/endpoints/{ep['endpoint_id']}"

        refused = client.patch(path, headers=HEADERS, json={"base_url": "https://llm.other-team.example.com/v1"})
        assert refused.status_code == 422 and "401" in refused.json()["detail"]
        moved = client.patch(path, headers=HEADERS, json={"base_url": "https://llm.other-team.example.com/v1",
                                                         "api_key": "k-2"}).json()
        assert (moved["base_url"], moved["platform"], moved["has_key"]) == ("https://llm.other-team.example.com/v1", "other", True)
        assert "api_key" not in moved and store.endpoints[ep["endpoint_id"]]["api_key"] == "k-2"

        dead = client.patch(path, headers=HEADERS, json={"base_url": "https://dead--flexsds-serve.modal.run/v1"})
        assert dead.status_code == 422 and "flexsds serve" in dead.json()["detail"]
        taken = client.patch(path, headers=HEADERS, json={"base_url": other["base_url"]})
        assert taken.status_code == 422 and other["name"] in taken.json()["detail"]

        noted = client.patch(path, headers=HEADERS, json={"meta": {"gpu": "NVIDIA L4", "draft_method": "eagle3", "adaptive": True}}).json()
        assert noted["meta"] == {"gpu": "NVIDIA L4", "draft_method": "eagle3", "adaptive": True} and noted["has_key"] is True
        cleared = client.patch(path, headers=HEADERS, json={"api_key": None, "meta": {"adaptive": None}}).json()
        assert cleared["has_key"] is False and cleared["meta"] == {"gpu": "NVIDIA L4", "draft_method": "eagle3"}
        assert client.patch(path, headers=HEADERS, json={"meta": {"base_model": "x"}}).status_code == 422
        assert client.patch(path, headers=HEADERS, json={"name": "   "}).status_code == 422
        assert client.patch(path, headers=HEADERS, json={"api_key": "x" * 501}).status_code == 422
