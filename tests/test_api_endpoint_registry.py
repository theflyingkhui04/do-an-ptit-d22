"""[M5-01 web console] Endpoint registry: add pasted endpoints, check, baseline, callers (no network, no DB)."""

from __future__ import annotations

import socket

import httpx
import pytest

from src.api.endpoints import EndpointRegistry
from tests.api_fakes import FakeStore

KEY = "ak-123.as-456"
MODELS = {
    "canduckhoi--flexsds-serve.modal.run": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "swift-orchid-ptit.trycloudflare.com": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "u--ptit-vllm-demo-vanilla.us-east.modal.direct": "qwen2.5-7b-vanilla",
}


def public_dns(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.16.230.132", port))]


def make(store=None, down=(), max_endpoints=12):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("authorization")))
        if request.url.host in down:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"data": [{"id": MODELS[request.url.host], "max_model_len": 4096}]})

    ids = iter(f"ep_{i:08x}" for i in range(100))
    registry = EndpointRegistry(store or FakeStore(), httpx.Client(transport=httpx.MockTransport(handler)),
                                resolve=public_dns, max_endpoints=max_endpoints, new_id=lambda: next(ids),
                                now=lambda: "2026-11-12T10:00:00+00:00")
    return registry, seen


def test_add_saves_reachable_endpoints_and_explains_the_rest():
    registry, seen = make()
    out = registry.add(["https://canduckhoi--flexsds-serve.modal.run/v1", "http://127.0.0.1:8000/v1",
                        "https://swift-orchid-ptit.trycloudflare.com/v1/"], api_key=KEY)
    codes = [(r["ok"], r["code"], r["platform"]) for r in out["results"]]
    assert codes == [(True, None, "modal"), (False, "needs_tunnel", "local"), (True, None, "tunnel")]
    assert [e["name"] for e in out["saved"]] == ["Qwen/Qwen2.5-7B-Instruct-AWQ", "Qwen/Qwen2.5-7B-Instruct-AWQ (đường hầm)"]
    assert all(auth == f"Bearer {KEY}" for _, auth in seen)
    listed = registry.list()
    assert len(listed) == 2 and all(e["has_key"] and "api_key" not in e for e in listed)
    assert listed[0]["base_url"] == "https://canduckhoi--flexsds-serve.modal.run/v1" and listed[0]["status"] == "ok"


def test_serving_config_fills_what_v1_models_cannot_tell():
    registry, _ = make()
    config = {"hardware": {"device_name": "NVIDIA L4"},
              "serving": {"draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator", "draft_method": "eagle3",
                          "adaptive_enabled": True}}
    saved = registry.add(["https://canduckhoi--flexsds-serve.modal.run/v1"], serving_config=config)["saved"][0]
    assert saved["meta"] == {"gpu": "NVIDIA L4", "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
                             "draft_method": "eagle3", "adaptive": True}
    assert saved["has_key"] is False


def test_pasting_the_same_address_again_updates_instead_of_duplicating():
    store = FakeStore()
    registry, _ = make(store)
    first = registry.add(["https://canduckhoi--flexsds-serve.modal.run/v1"])["saved"][0]
    again = registry.add(["https://canduckhoi--flexsds-serve.modal.run/v1/"], api_key=KEY)["saved"][0]
    assert again["endpoint_id"] == first["endpoint_id"] and again["has_key"] is True and len(registry.list()) == 1


def test_limit_on_the_number_of_endpoints():
    registry, _ = make(max_endpoints=1)
    out = registry.add(["https://canduckhoi--flexsds-serve.modal.run/v1", "https://swift-orchid-ptit.trycloudflare.com/v1"])
    assert [r["code"] for r in out["results"]] == [None, "too_many"] and len(out["saved"]) == 1


def test_check_marks_a_stopped_session_and_recovers():
    store = FakeStore()
    registry, _ = make(store, down=("swift-orchid-ptit.trycloudflare.com",))
    with pytest.raises(LookupError):
        registry.check("ep_missing")
    ep = make(store)[0].add(["https://swift-orchid-ptit.trycloudflare.com/v1"])["saved"][0]
    checked = registry.check(ep["endpoint_id"])
    assert checked["status"] == "down" and "flexsds serve" in checked["status_message"]
    assert make(store)[0].check(ep["endpoint_id"])["status"] == "ok"


def test_baseline_is_unique_and_callers_use_the_stored_key_and_model():
    registry, _ = make()
    a, b = registry.add(["https://u--ptit-vllm-demo-vanilla.us-east.modal.direct/v1",
                         "https://canduckhoi--flexsds-serve.modal.run/v1"], api_key=KEY)["saved"]
    registry.set_baseline(a["endpoint_id"])
    registry.set_baseline(b["endpoint_id"])
    assert [e["is_baseline"] for e in registry.list()] == [False, True]
    assert registry.baseline()["endpoint_id"] == b["endpoint_id"]
    registry.set_baseline(None)
    assert registry.baseline() is None
    caller = registry.caller(a["endpoint_id"])
    assert (caller.lane, caller.model, caller.base_url) == ("qwen2.5-7b-vanilla", "qwen2.5-7b-vanilla",
                                                            "https://u--ptit-vllm-demo-vanilla.us-east.modal.direct/v1")
    assert registry.rename(a["endpoint_id"], "  vanilla A10G  ")["name"] == "vanilla A10G"
    assert registry.delete(a["endpoint_id"]) is True and registry.delete(a["endpoint_id"]) is False
    assert [e["endpoint_id"] for e in registry.list()] == [b["endpoint_id"]]
