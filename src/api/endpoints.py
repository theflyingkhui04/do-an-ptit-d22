"""Endpoints pasted from `flexsds serve` (OpenAI-compatible servers on Modal, Colab, a tunnel or elsewhere).

The web never starts or stops GPUs. It only checks an address before using it:
the platform is guessed from the address the CLI prints, local addresses are
explained (they need a public tunnel because the backend runs on Render), private
network addresses are refused (the backend must not be usable to reach internal
hosts), and ``GET /v1/models`` tells which model the server runs.
Error messages never contain the API key.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
TUNNEL_HINT = "cloudflared tunnel --url http://127.0.0.1:{port}"
PROBE_TIMEOUT_S = 60.0


class EndpointError(Exception):
    """An address that cannot be used, with a Vietnamese message that says how to fix it."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def normalize_url(raw: str) -> str:
    """Base URL ending in ``/v1`` (the form `flexsds serve` prints)."""
    text = (raw or "").strip()
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise EndpointError("invalid_url", "Địa chỉ phải bắt đầu bằng https:// và có tên máy, ví dụ "
                                           "https://…--flexsds-serve.modal.run/v1")
    path = parts.path.rstrip("/")
    for tail in ("/chat/completions", "/completions", "/models"):
        path = path.removesuffix(tail)
    if not path:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _is_local(host: str) -> bool:
    if host.lower() in LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def platform_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    if host.endswith((".modal.run", ".modal.direct")):
        return "modal"
    if host.endswith(".trycloudflare.com"):
        return "tunnel"
    if _is_local(host):
        return "local"
    return "other"


def check_address(url: str, *, resolve: Callable[..., list] = socket.getaddrinfo) -> None:
    """Raise unless the backend may call ``url``: https, public address after DNS resolution."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if _is_local(host):
        raise EndpointError(
            "needs_tunnel",
            "Đây là địa chỉ trên máy của bạn; web chạy trên mạng nên không gọi tới được. Chạy lệnh "
            f"`{TUNNEL_HINT.format(port=parts.port or 8000)}` trên máy đang chạy flexsds rồi dán địa chỉ "
            "https://….trycloudflare.com/v1 mà lệnh in ra.")
    if parts.scheme != "https":
        raise EndpointError("https_required", "Chỉ nhận địa chỉ https://. Địa chỉ Modal và đường hầm "
                                              "cloudflared đều là https.")
    try:
        infos = resolve(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise EndpointError("unreachable", f"Không phân giải được tên máy {host}.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
                or address.is_multicast or address.is_unspecified):
            raise EndpointError("blocked_address", f"{host} trỏ vào một địa chỉ nội bộ; web không gọi địa chỉ này.")


def probe(url: str, api_key: str | None, client: httpx.Client) -> dict[str, Any]:
    """``GET {url}/models``: the first served model id (sent as ``model`` in every request) and its context length."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = client.get(f"{url}/models", headers=headers, timeout=PROBE_TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise EndpointError("unreachable", f"Không kết nối được ({type(exc).__name__}). Kiểm tra phiên "
                                           "flexsds serve còn chạy không.") from exc
    if response.status_code in (401, 403):
        raise EndpointError("auth_failed", f"Endpoint từ chối khoá API (HTTP {response.status_code}).")
    if response.status_code in (502, 503, 504):
        raise EndpointError("not_ready", f"Endpoint chưa sẵn sàng (HTTP {response.status_code}): GPU đang khởi động "
                                         "hoặc đang ngủ, hoặc phiên flexsds serve đã dừng. Chờ vài phút rồi kiểm tra lại.")
    if response.status_code != 200:
        raise EndpointError("bad_response", f"/v1/models trả HTTP {response.status_code}.")
    try:
        models = [m for m in response.json().get("data") or [] if isinstance(m, dict) and m.get("id")]
    except (ValueError, AttributeError) as exc:
        raise EndpointError("bad_response", "/v1/models không trả JSON chuẩn OpenAI.") from exc
    if not models:
        raise EndpointError("no_model", "Endpoint chưa phục vụ model nào.")
    max_len = models[0].get("max_model_len")
    root = models[0].get("root")
    # ``root`` (vLLM) is the model behind the served name: lanes served as qwen2.5-7b-vanilla / -adaptive share it.
    return {"model_id": models[0]["id"], "models": [m["id"] for m in models],
            "max_model_len": max_len if isinstance(max_len, int) else None,
            "root": root if isinstance(root, str) and root else None}


def parse_serving_config(config: Any) -> dict[str, Any]:
    """Fields of the CLI's ``serving_config.json`` that describe an endpoint; missing fields are left out."""
    if not isinstance(config, dict):
        raise EndpointError("bad_config", "serving_config.json phải là một đối tượng JSON.")
    hardware = config.get("hardware") if isinstance(config.get("hardware"), dict) else {}
    serving = config.get("serving") if isinstance(config.get("serving"), dict) else {}
    fields = {"gpu": hardware.get("device_name", ...), "sm": hardware.get("sm_architecture", ...),
              "backend": serving.get("backend", ...), "target_model": serving.get("target_model", ...),
              "draft_model": serving.get("draft_model", ...), "draft_method": serving.get("draft_method", ...),
              "adaptive": serving.get("adaptive_enabled", ...), "max_model_len": serving.get("max_model_len", ...),
              "flexsds_version": config.get("flexsds_version", ...)}
    return {k: v for k, v in fields.items() if v is not ...}


PLATFORM_LABELS = {"modal": "Modal", "tunnel": "đường hầm", "other": "khác"}


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    import uuid

    return f"ep_{uuid.uuid4().hex[:8]}"


class EndpointRegistry:
    """The endpoints people pasted, shared by every viewer of the demo. Keys stay in the database."""

    def __init__(self, store: Any, client: httpx.Client, *, resolve: Callable[..., list] = socket.getaddrinfo,
                 max_endpoints: int = 12, new_id: Callable[[], str] = _new_id, now: Callable[[], str] = _utc_now,
                 llm_client: httpx.Client | None = None) -> None:
        self.store = store
        self._client = client
        self._llm_client = llm_client or client
        self._resolve = resolve
        self.max_endpoints = max_endpoints
        self._new_id = new_id
        self._now = now

    def _inspect(self, raw: str, api_key: str | None) -> dict[str, Any]:
        result: dict[str, Any] = {"input": (raw or "").strip(), "url": None, "ok": False, "code": None,
                                  "message": None, "platform": None, "model_id": None, "max_model_len": None}
        try:
            url = normalize_url(raw)
            result.update(url=url, platform=platform_of(url))
            check_address(url, resolve=self._resolve)
            found = probe(url, api_key, self._client)
        except EndpointError as exc:
            result.update(code=exc.code, message=exc.message)
            return result
        result.update(ok=True, model_id=found["model_id"], max_model_len=found["max_model_len"], root=found["root"])
        return result

    def detect(self, urls: list[str], api_key: str | None = None) -> list[dict[str, Any]]:
        return [self._inspect(raw, api_key or None) for raw in urls if (raw or "").strip()]

    def _unique_name(self, model_id: str, platform: str) -> str:
        names = {e["name"] for e in self.store.list_endpoints()}
        if model_id not in names:
            return model_id
        candidate = f"{model_id} ({PLATFORM_LABELS.get(platform, platform)})"
        index = 2
        while candidate in names:
            candidate = f"{model_id} ({PLATFORM_LABELS.get(platform, platform)} {index})"
            index += 1
        return candidate

    def add(self, urls: list[str], api_key: str | None = None, serving_config: Any = None) -> dict[str, Any]:
        meta = parse_serving_config(serving_config) if serving_config is not None else {}
        results = self.detect(urls, api_key)
        saved = []
        for result in results:
            if not result["ok"]:
                continue
            fields = {"model_id": result["model_id"], "max_model_len": result["max_model_len"] or meta.get("max_model_len"),
                      "status": "ok", "status_message": None, "checked_at": self._now()}
            row_meta = {**meta, **({"base_model": result["root"]} if result.get("root") else {})}
            existing = self.store.find_endpoint_by_url(result["url"])
            if existing is not None:
                if row_meta:
                    fields["meta"] = {**(existing.get("meta") or {}), **row_meta}
                if api_key:
                    fields["api_key"] = api_key
                self.store.update_endpoint(existing["endpoint_id"], fields)
                saved.append(self.store.get_endpoint(existing["endpoint_id"]))
                continue
            if len(self.store.list_endpoints()) >= self.max_endpoints:
                result.update(ok=False, code="too_many",
                              message=f"Đã có {self.max_endpoints} endpoint; xoá bớt endpoint không dùng rồi thêm lại.")
                continue
            endpoint_id = self._new_id()
            self.store.insert_endpoint({
                "endpoint_id": endpoint_id, "name": self._unique_name(result["model_id"], result["platform"]),
                "base_url": result["url"], "platform": result["platform"], "meta": row_meta, "api_key": api_key or None,
                "is_baseline": False, "created_at": self._now(), **fields})
            saved.append(self.store.get_endpoint(endpoint_id))
        return {"results": results, "saved": saved}

    def list(self) -> list[dict[str, Any]]:
        return self.store.list_endpoints()

    def _require(self, endpoint_id: str, *, with_key: bool = False) -> dict[str, Any]:
        row = self.store.get_endpoint(endpoint_id, with_key=with_key)
        if row is None:
            raise LookupError(endpoint_id)
        return row

    def check(self, endpoint_id: str) -> dict[str, Any]:
        row = self._require(endpoint_id, with_key=True)
        try:
            found = probe(row["base_url"], row.get("api_key"), self._client)
        except EndpointError as exc:
            fields = {"status": "down", "status_message": exc.message, "checked_at": self._now()}
        else:
            fields = {"status": "ok", "status_message": None, "checked_at": self._now(), "model_id": found["model_id"]}
            if found["root"]:
                fields["meta"] = {**(row.get("meta") or {}), "base_model": found["root"]}
        self.store.update_endpoint(endpoint_id, fields)
        return self._require(endpoint_id)

    def rename(self, endpoint_id: str, name: str) -> dict[str, Any]:
        self._require(endpoint_id)
        self.store.update_endpoint(endpoint_id, {"name": name.strip()[:80]})
        return self._require(endpoint_id)

    def set_baseline(self, endpoint_id: str | None) -> None:
        if endpoint_id is not None:
            self._require(endpoint_id)
        self.store.set_baseline(endpoint_id)

    def baseline(self) -> dict[str, Any] | None:
        return next((e for e in self.store.list_endpoints() if e["is_baseline"]), None)

    def delete(self, endpoint_id: str) -> bool:
        return self.store.delete_endpoint(endpoint_id)

    def get(self, endpoint_id: str, *, with_key: bool = False) -> dict[str, Any]:
        return self._require(endpoint_id, with_key=with_key)

    def caller(self, endpoint_id: str) -> Any:
        from src.api.llm import LaneCaller

        row = self._require(endpoint_id, with_key=True)
        return LaneCaller(row["name"], row["base_url"], row.get("api_key") or "", row["model_id"],
                          client=self._llm_client)
