"""[M5-01 web console] Endpoint registry: add pasted endpoints, check, baseline, callers (no network, no DB)."""

from __future__ import annotations

import socket

import httpx
import pytest

from src.api.endpoints import EndpointError, EndpointRegistry
from tests.api_fakes import FakeStore

KEY = "ak-123.as-456"
MODELS = {
    "canduckhoi--flexsds-serve.modal.run": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "swift-orchid-ptit.trycloudflare.com": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "u--ptit-vllm-demo-vanilla.us-east.modal.direct": "qwen2.5-7b-vanilla",
    "u--ptit-vllm-demo-adaptive.us-east.modal.direct": "qwen2.5-7b-adaptive",
}
ROOTS = {  # vLLM ``root``: the model behind the served name; the demo lanes share one
    "u--ptit-vllm-demo-vanilla.us-east.modal.direct": "/models/qwen2.5-7b-awq",
    "u--ptit-vllm-demo-adaptive.us-east.modal.direct": "/models/qwen2.5-7b-awq",
}


def public_dns(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.16.230.132", port))]


LOCKED = "llm.other-team.example.com"  # a vLLM started with --api-key KEY


def make(store=None, down=(), max_endpoints=12):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("authorization")))
        if request.url.host in down:
            raise httpx.ConnectError("refused")
        if request.url.host == LOCKED:
            if request.headers.get("authorization") != f"Bearer {KEY}":
                return httpx.Response(401, json={"error": "unauthorized"})
            return httpx.Response(200, json={"data": [{"id": "meta/llama", "max_model_len": 8192}]})
        model = {"id": MODELS[request.url.host], "max_model_len": 4096}
        if request.url.host in ROOTS:
            model["root"] = ROOTS[request.url.host]
        return httpx.Response(200, json={"data": [model]})

    ids = iter(f"ep_{i:08x}" for i in range(100))
    registry = EndpointRegistry(store or FakeStore(), httpx.Client(transport=httpx.MockTransport(handler)),
                                resolve=public_dns, max_endpoints=max_endpoints, new_id=lambda: next(ids),
                                now=lambda: "2026-11-12T10:00:00+00:00")
    return registry, seen


MODAL = "https://canduckhoi--flexsds-serve.modal.run/v1"
COLAB = "https://swift-orchid-ptit.trycloudflare.com/v1"
VANILLA = "https://u--ptit-vllm-demo-vanilla.us-east.modal.direct/v1"
ADAPTIVE = "https://u--ptit-vllm-demo-adaptive.us-east.modal.direct/v1"


def items(*urls, key=None):
    return [{"url": u, "api_key": key} for u in urls]


def test_add_saves_reachable_endpoints_and_explains_the_rest():
    registry, seen = make()
    out = registry.add(items(MODAL, "http://127.0.0.1:8000/v1", COLAB + "/", key=KEY))
    codes = [(r["ok"], r["code"], r["platform"]) for r in out["results"]]
    assert codes == [(True, None, "modal"), (False, "needs_tunnel", "local"), (True, None, "tunnel")]
    assert [e["name"] for e in out["saved"]] == ["Qwen/Qwen2.5-7B-Instruct-AWQ", "Qwen/Qwen2.5-7B-Instruct-AWQ (đường hầm)"]
    assert all(auth == f"Bearer {KEY}" for _, auth in seen)
    listed = registry.list()
    assert len(listed) == 2 and all(e["has_key"] and "api_key" not in e for e in listed)
    assert listed[0]["base_url"] == MODAL and listed[0]["status"] == "ok" and listed[0]["meta"] == {}


def test_add_takes_a_name_and_a_key_for_each_address():
    registry, seen = make()
    out = registry.add([{"url": "https://" + LOCKED + "/v1", "name": "  llama nhóm khác  ", "api_key": KEY},
                        {"url": COLAB, "name": "   "}])
    assert [e["name"] for e in out["saved"]] == ["llama nhóm khác", "Qwen/Qwen2.5-7B-Instruct-AWQ"]
    assert [e["has_key"] for e in out["saved"]] == [True, False]
    assert seen == [(LOCKED, f"Bearer {KEY}"), ("swift-orchid-ptit.trycloudflare.com", None)]


def test_an_endpoint_that_wants_a_key_says_so():
    registry, _ = make()
    (result,) = registry.detect(["https://" + LOCKED + "/v1"])
    assert (result["ok"], result["code"]) == (False, "auth_failed") and "401" in result["message"]
    assert registry.detect(["https://" + LOCKED + "/v1"], KEY)[0]["ok"] is True


def test_an_address_already_in_the_list_is_named_and_not_called_again():
    store = FakeStore()
    registry, seen = make(store)
    first = registry.add(items(VANILLA))["saved"][0]
    registry.rename(first["endpoint_id"], "vanilla A10G")
    seen.clear()
    (found,) = registry.detect([VANILLA + "/"])
    assert (found["ok"], found["code"], found["endpoint_id"], found["name"]) == (
        False, "exists", first["endpoint_id"], "vanilla A10G")
    assert "vanilla A10G" in found["message"]
    out = registry.add(items(VANILLA, key=KEY))
    assert seen == [] and out["saved"] == [] and out["results"][0]["code"] == "exists"
    assert [e["has_key"] for e in registry.list()] == [False]


def test_limit_on_the_number_of_endpoints():
    registry, _ = make(max_endpoints=1)
    out = registry.add(items(MODAL, COLAB))
    assert [r["code"] for r in out["results"]] == [None, "too_many"] and len(out["saved"]) == 1


def test_check_marks_a_stopped_session_and_recovers():
    store = FakeStore()
    registry, _ = make(store, down=("swift-orchid-ptit.trycloudflare.com",))
    with pytest.raises(LookupError):
        registry.check("ep_missing")
    ep = make(store)[0].add(items(COLAB))["saved"][0]
    checked = registry.check(ep["endpoint_id"])
    assert checked["status"] == "down" and "flexsds serve" in checked["status_message"]
    assert make(store)[0].check(ep["endpoint_id"])["status"] == "ok"


@pytest.mark.parametrize("waking", [
    lambda request: httpx.Response(503, json={"error": "no container ready"}),
    lambda request: (_ for _ in ()).throw(httpx.RemoteProtocolError("closed before a response", request=request)),
    lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("no answer yet", request=request)),
], ids=["http-503", "reset-before-response", "read-timeout"])
def test_check_says_starting_while_a_sleeping_gpu_wakes_up(waking):
    store = FakeStore()
    ep = make(store)[0].add(items(VANILLA, key=KEY))["saved"][0]
    sleeping = EndpointRegistry(store, httpx.Client(transport=httpx.MockTransport(waking)), resolve=public_dns,
                                now=lambda: "2026-11-18T14:42:00+00:00")
    checked = sleeping.check(ep["endpoint_id"])
    assert checked["status"] == "starting" and "khởi động" in checked["status_message"]
    assert checked["checked_at"] == "2026-11-18T14:42:00+00:00"
    assert make(store)[0].check(ep["endpoint_id"])["status"] == "ok"  # the next check finds it up


def test_the_model_behind_the_served_name_is_kept_for_fair_comparisons():
    store = FakeStore()
    registry, _ = make(store)
    a, b = registry.add(items(VANILLA, ADAPTIVE, key=KEY))["saved"]
    assert a["model_id"] != b["model_id"]
    assert a["meta"]["base_model"] == b["meta"]["base_model"] == "/models/qwen2.5-7b-awq"
    # saved before the base model was read: the next check fills it in and keeps the noted fields
    store.update_endpoint(a["endpoint_id"], {"meta": {"gpu": "NVIDIA A10G"}})
    assert registry.check(a["endpoint_id"])["meta"] == {"gpu": "NVIDIA A10G", "base_model": "/models/qwen2.5-7b-awq"}
    # an endpoint without ``root`` gets no base_model
    other = registry.add(items(MODAL))["saved"][0]
    assert "base_model" not in other["meta"]


def test_baseline_is_unique_and_callers_use_the_stored_key_and_model():
    registry, _ = make()
    a, b = registry.add(items(VANILLA, MODAL, key=KEY))["saved"]
    registry.set_baseline(a["endpoint_id"])
    registry.set_baseline(b["endpoint_id"])
    assert [e["is_baseline"] for e in registry.list()] == [False, True]
    assert registry.baseline()["endpoint_id"] == b["endpoint_id"]
    registry.set_baseline(None)
    assert registry.baseline() is None
    caller = registry.caller(a["endpoint_id"])
    assert (caller.lane, caller.model, caller.base_url) == ("qwen2.5-7b-vanilla", "qwen2.5-7b-vanilla", VANILLA)
    assert registry.rename(a["endpoint_id"], "  vanilla A10G  ")["name"] == "vanilla A10G"
    assert registry.delete(a["endpoint_id"]) is True and registry.delete(a["endpoint_id"]) is False
    assert [e["endpoint_id"] for e in registry.list()] == [b["endpoint_id"]]


# ---------------------------------------------------------------- editing a saved endpoint
def test_a_new_address_is_checked_before_it_replaces_the_old_one():
    store = FakeStore()
    registry, seen = make(store, down=("dead--flexsds-serve.modal.run",))
    ep = registry.add(items(VANILLA, key=KEY))["saved"][0]
    store.update_endpoint(ep["endpoint_id"], {"status": "down", "status_message": "HTTP 503", "meta": {"gpu": "A10G"}})
    seen.clear()
    moved = registry.update(ep["endpoint_id"], base_url=ADAPTIVE + "/")
    assert (moved["base_url"], moved["model_id"], moved["platform"], moved["status"], moved["status_message"]) == (
        ADAPTIVE, "qwen2.5-7b-adaptive", "modal", "ok", None)
    assert moved["meta"] == {"gpu": "A10G", "base_model": "/models/qwen2.5-7b-awq"}
    assert seen == [("u--ptit-vllm-demo-adaptive.us-east.modal.direct", f"Bearer {KEY}")]  # the stored key
    for bad, code in (("http://127.0.0.1:8000/v1", "needs_tunnel"), ("https://dead--flexsds-serve.modal.run/v1", "unreachable"),
                      ("ftp://x", "invalid_url")):
        with pytest.raises(EndpointError) as caught:
            registry.update(ep["endpoint_id"], base_url=bad)
        assert caught.value.code == code
    assert registry.get(ep["endpoint_id"])["base_url"] == ADAPTIVE  # nothing changed by the failed tries


def test_a_new_address_may_not_be_another_endpoints():
    registry, _ = make()
    a, b = registry.add(items(VANILLA, ADAPTIVE))["saved"]
    with pytest.raises(EndpointError) as caught:
        registry.update(a["endpoint_id"], base_url=ADAPTIVE)
    assert caught.value.code == "exists" and b["name"] in caught.value.message
    assert registry.update(a["endpoint_id"], base_url=VANILLA + "/")["base_url"] == VANILLA  # its own address is fine


def test_a_new_key_is_used_to_check_the_new_address():
    registry, seen = make()
    ep = registry.add(items(MODAL))["saved"][0]
    with pytest.raises(EndpointError) as caught:
        registry.update(ep["endpoint_id"], base_url="https://" + LOCKED + "/v1")
    assert caught.value.code == "auth_failed"
    moved = registry.update(ep["endpoint_id"], base_url="https://" + LOCKED + "/v1", api_key=KEY)
    assert (moved["model_id"], moved["has_key"], moved["platform"]) == ("meta/llama", True, "other")
    assert seen[-1] == (LOCKED, f"Bearer {KEY}")


def test_the_key_can_be_set_replaced_and_removed():
    registry, _ = make()
    ep = registry.add(items(MODAL))["saved"][0]
    assert registry.update(ep["endpoint_id"], api_key=KEY)["has_key"] is True
    assert registry.get(ep["endpoint_id"], with_key=True)["api_key"] == KEY
    assert registry.update(ep["endpoint_id"], name="giữ khoá")["has_key"] is True  # not sent: kept
    cleared = registry.update(ep["endpoint_id"], api_key=None)
    assert cleared["has_key"] is False and "api_key" not in cleared
    assert registry.get(ep["endpoint_id"], with_key=True)["api_key"] is None


def test_notes_on_the_serving_setup_merge_and_can_be_removed():
    registry, _ = make()
    ep = registry.add(items(VANILLA))["saved"][0]
    noted = registry.update(ep["endpoint_id"], meta={"gpu": "NVIDIA A10G", "draft_method": "eagle3", "adaptive": True,
                                                     "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator"})
    assert noted["meta"] == {"base_model": "/models/qwen2.5-7b-awq", "gpu": "NVIDIA A10G", "draft_method": "eagle3",
                             "adaptive": True, "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator"}
    again = registry.update(ep["endpoint_id"], meta={"draft_method": None, "draft_model": None, "adaptive": None})
    assert again["meta"] == {"base_model": "/models/qwen2.5-7b-awq", "gpu": "NVIDIA A10G"}
    with pytest.raises(EndpointError) as caught:
        registry.update(ep["endpoint_id"], meta={"base_model": "x"})  # read from /v1/models, not typed in
    assert caught.value.code == "bad_meta"
    with pytest.raises(LookupError):
        registry.update("ep_missing", name="x")
