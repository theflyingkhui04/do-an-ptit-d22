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
# Notes people may type about how an endpoint is served (shown on the Endpoint page only, design D14).
NOTE_KEYS = frozenset({"gpu", "draft_method", "draft_model", "adaptive", "target_model"})
_KEEP: Any = object()  # update(): leave the stored key as it is


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
    except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
        # The server took the call but did not answer: a GPU waking up on Modal does this before its 503s.
        raise EndpointError("not_ready", f"Endpoint chưa trả lời ({type(exc).__name__}): GPU có thể đang khởi động. "
                                         "Chờ vài phút rồi kiểm tra lại.") from exc
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
            saved = self.store.find_endpoint_by_url(url)
            if saved is not None:  # not called again: a sleeping GPU is not woken for nothing
                result.update(code="exists", endpoint_id=saved["endpoint_id"], name=saved["name"],
                              message=f"Đã có trong danh sách với tên {saved['name']}, không cần thêm lại.")
                return result
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

    def add(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Save each reachable address ``{"url", "name"?, "api_key"?}``; the others come back with their reason."""
        results, saved = [], []
        for item in items:
            if not (item.get("url") or "").strip():
                continue
            api_key = item.get("api_key") or None
            result = self._inspect(item["url"], api_key)
            results.append(result)
            if not result["ok"]:
                continue
            if len(self.store.list_endpoints()) >= self.max_endpoints:
                result.update(ok=False, code="too_many",
                              message=f"Đã có {self.max_endpoints} endpoint; xoá bớt endpoint không dùng rồi thêm lại.")
                continue
            endpoint_id = self._new_id()
            name = (item.get("name") or "").strip()[:80] or result["model_id"]
            self.store.insert_endpoint({
                "endpoint_id": endpoint_id, "name": self._unique_name(name, result["platform"]),
                "base_url": result["url"], "platform": result["platform"], "model_id": result["model_id"],
                "max_model_len": result["max_model_len"], "meta": {"base_model": result["root"]} if result.get("root") else {},
                "api_key": api_key, "is_baseline": False, "status": "ok", "status_message": None,
                "checked_at": self._now(), "created_at": self._now()})
            saved.append(self.store.get_endpoint(endpoint_id))
        return {"results": results, "saved": saved}

    def update(self, endpoint_id: str, *, name: str | None = None, base_url: str | None = None,
               api_key: Any = _KEEP, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Edit a saved endpoint. A new address is checked like a pasted one before anything is written.

        ``api_key``: a string sets it, ``None`` removes it, left out keeps it. ``meta``: notes (``NOTE_KEYS``)
        merged into the stored ones, ``None`` removes a note; what ``/v1/models`` told (``base_model``) stays.
        """
        row = self._require(endpoint_id, with_key=True)
        fields: dict[str, Any] = {}
        notes = dict(row.get("meta") or {})
        if name is not None:
            fields["name"] = name.strip()[:80]
            if not fields["name"]:
                raise EndpointError("bad_name", "Tên không được để trống.")
        if meta is not None:
            unknown = sorted(set(meta) - NOTE_KEYS)
            if unknown:
                raise EndpointError("bad_meta", f"Không ghi chú được {', '.join(unknown)}; chỉ ghi được GPU, "
                                                "cách giải mã, model nháp và model đích.")
            for key, value in meta.items():
                if value is None:
                    notes.pop(key, None)
                else:
                    notes[key] = value
            fields["meta"] = notes
        key = row.get("api_key") if api_key is _KEEP else (api_key or None)
        if api_key is not _KEEP:
            fields["api_key"] = key
        if base_url is not None:
            url = normalize_url(base_url)
            if url != row["base_url"]:
                other = self.store.find_endpoint_by_url(url)
                if other is not None:
                    raise EndpointError("exists", f"Địa chỉ này đã thuộc endpoint {other['name']}.")
                check_address(url, resolve=self._resolve)
                found = probe(url, key, self._client)
                if found["root"]:
                    notes["base_model"] = found["root"]
                else:
                    notes.pop("base_model", None)
                fields.update(base_url=url, platform=platform_of(url), model_id=found["model_id"],
                              max_model_len=found["max_model_len"], meta=notes, status="ok", status_message=None,
                              checked_at=self._now())
        if fields:
            self.store.update_endpoint(endpoint_id, fields)
        return self._require(endpoint_id)

    def list(self) -> list[dict[str, Any]]:
        return self.store.list_endpoints()

    def _require(self, endpoint_id: str, *, with_key: bool = False) -> dict[str, Any]:
        row = self.store.get_endpoint(endpoint_id, with_key=with_key)
        if row is None:
            raise LookupError(endpoint_id)
        return row

    def check(self, endpoint_id: str) -> dict[str, Any]:
        """One ``GET /v1/models``: ``ok``, ``starting`` (a sleeping GPU is waking up; the web asks again) or ``down``."""
        row = self._require(endpoint_id, with_key=True)
        try:
            found = probe(row["base_url"], row.get("api_key"), self._client)
        except EndpointError as exc:
            status = "starting" if exc.code == "not_ready" else "down"
            fields = {"status": status, "status_message": exc.message, "checked_at": self._now()}
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
