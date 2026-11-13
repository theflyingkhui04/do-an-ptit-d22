"""[M5-01 web console] Endpoints pasted from `flexsds serve`: URL rules, platform, address guard, model probe."""

from __future__ import annotations

import socket

import httpx
import pytest

from src.api.endpoints import (
    EndpointError,
    check_address,
    normalize_url,
    parse_serving_config,
    platform_of,
    probe,
)

TOKEN = "tok-id.tok-secret"


def resolver(*ips):
    def resolve(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]
    return resolve


def test_normalize_keeps_v1_and_strips_noise():
    assert normalize_url("  https://canduckhoi--flexsds-serve.modal.run/v1/  ") == "https://canduckhoi--flexsds-serve.modal.run/v1"
    assert normalize_url("https://x.trycloudflare.com") == "https://x.trycloudflare.com/v1"
    assert normalize_url("https://x.modal.run/v1/chat/completions") == "https://x.modal.run/v1"
    for bad in ("", "canduckhoi.modal.run/v1", "ftp://x/v1", "https:///v1"):
        with pytest.raises(EndpointError) as info:
            normalize_url(bad)
        assert info.value.code == "invalid_url"


def test_platform_comes_from_the_address_the_cli_prints():
    assert platform_of("https://canduckhoi--flexsds-serve.modal.run/v1") == "modal"
    assert platform_of("https://nguyenduonghb940--ptit-vllm-demo-vanilla.us-east.modal.direct/v1") == "modal"
    assert platform_of("https://swift-orchid-ptit.trycloudflare.com/v1") == "tunnel"
    assert platform_of("http://127.0.0.1:8000/v1") == "local"
    assert platform_of("http://localhost:8000/v1") == "local"
    assert platform_of("https://llm.example.org/v1") == "other"


def test_local_addresses_need_a_tunnel_and_private_ones_are_blocked():
    with pytest.raises(EndpointError) as info:
        check_address("http://127.0.0.1:8000/v1", resolve=resolver("127.0.0.1"))
    assert info.value.code == "needs_tunnel" and "cloudflared tunnel --url http://127.0.0.1:8000" in info.value.message
    with pytest.raises(EndpointError) as info:
        check_address("http://llm.example.org/v1", resolve=resolver("93.184.216.34"))
    assert info.value.code == "https_required"
    for ip in ("10.0.0.5", "192.168.1.2", "169.254.169.254", "172.16.0.1", "::1", "0.0.0.0"):
        with pytest.raises(EndpointError) as info:
            check_address("https://evil.example.org/v1", resolve=resolver(ip))
        assert info.value.code == "blocked_address"
    check_address("https://swift-orchid-ptit.trycloudflare.com/v1", resolve=resolver("104.16.230.132"))


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_probe_reads_the_served_model_and_sends_the_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "Qwen/Qwen2.5-7B-Instruct-AWQ", "object": "model", "max_model_len": 4096}]})

    result = probe("https://x.modal.run/v1", TOKEN, client(handler))
    assert seen == {"url": "https://x.modal.run/v1/models", "auth": f"Bearer {TOKEN}"}
    assert result == {"model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ", "models": ["Qwen/Qwen2.5-7B-Instruct-AWQ"],
                      "max_model_len": 4096}

    def no_key(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    assert probe("https://x.modal.run/v1", None, client(no_key))["max_model_len"] is None


@pytest.mark.parametrize(("response", "code"), [
    (httpx.Response(401, text=f"bad {TOKEN}"), "auth_failed"),
    (httpx.Response(500, text="boom"), "bad_response"),
    (httpx.Response(200, text="<html>not json</html>"), "bad_response"),
    (httpx.Response(200, json={"data": []}), "no_model"),
])
def test_probe_failures_are_explained_without_the_key(response, code):
    with pytest.raises(EndpointError) as info:
        probe("https://x.modal.run/v1", TOKEN, client(lambda r: response))
    assert info.value.code == code and TOKEN not in info.value.message


def test_unreachable_endpoint():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(EndpointError) as info:
        probe("https://x.trycloudflare.com/v1", None, client(down))
    assert info.value.code == "unreachable"


def test_serving_config_from_the_cli():
    config = {"flexsds_version": "0.2.0-alpha",
              "hardware": {"device_name": "NVIDIA L4", "sm_architecture": "sm_89", "free_vram_mb": 22450.0},
              "serving": {"backend": "modal", "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                          "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator", "draft_method": "eagle3",
                          "max_model_len": 4096, "adaptive_enabled": True}}
    assert parse_serving_config(config) == {
        "gpu": "NVIDIA L4", "sm": "sm_89", "backend": "modal", "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator", "draft_method": "eagle3", "adaptive": True,
        "max_model_len": 4096, "flexsds_version": "0.2.0-alpha"}
    assert parse_serving_config({"serving": {"draft_model": None}}) == {"draft_model": None}
    with pytest.raises(EndpointError):
        parse_serving_config(["not", "an", "object"])
