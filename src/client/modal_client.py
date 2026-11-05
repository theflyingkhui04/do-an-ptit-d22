"""Shared client for the [M3-00] vLLM lanes on Modal (OpenAI-compatible, stdlib only).

    from src.client.modal_client import ModalLaneClient
    lane = ModalLaneClient("vanilla")                      # token from env or .env
    lane.chat([{"role": "user", "content": "Xin chào"}], max_tokens=16)
    for event in lane.stream(messages): ...                # "delta" events, then one "final"
    lane.margins(messages, max_tokens=8)                   # top-1 minus top-2 logprob per token

No Modal account is needed: the URL follows the deployed app's naming
(``https://<workspace>--ptit-vllm-<lane>.us-east.modal.direct/v1``) and can be
overridden with ``PTIT_MODAL_URL_<LANE>`` or ``PTIT_MODAL_WORKSPACE``.
The proxy token is sent as ``Authorization: Bearer <id>.<secret>`` and is never
included in errors or results.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = "nguyenduonghb940"
APP_NAME = "ptit-vllm"

# Must match experiments.modal_lanes.LANES (checked by tests/test_modal_client.py).
LANE_MODELS = {
    "vanilla": "qwen2.5-7b-vanilla",
    "static": "qwen2.5-7b-static-draft",
    "adaptive": "qwen2.5-7b-adaptive",
    "eagle": "qwen2.5-7b-eagle3-draft",
}


class ModalClientError(RuntimeError):
    """Request to a Modal lane failed."""


class ModalAuthError(ModalClientError):
    """The lane rejected the proxy token (HTTP 401/403)."""


class ModalReadyTimeout(ModalClientError):
    """The lane kept answering 503 (cold start) past the ready timeout."""


class LogitMarginUnsupported(ModalClientError):
    """The lane returned no top-2 logprobs, so a logit margin cannot be computed."""


@dataclass
class ChatResult:
    text: str
    usage: dict[str, Any]
    metrics: dict[str, Any] | None
    raw: dict[str, Any]


@dataclass
class StreamEvent:
    kind: str  # "delta" or "final"
    text: str
    usage: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    ttft_s: float | None = None
    total_s: float | None = None


@dataclass
class TokenMargin:
    token: str
    logprob: float
    margin: float  # logprob(top-1) - logprob(top-2) == logit gap


@dataclass
class _Response:
    status: int
    _read: Callable[[], bytes] = field(repr=False)
    _lines: Callable[[], Iterator[str]] = field(repr=False)

    def read(self) -> bytes:
        return self._read()

    def lines(self) -> Iterator[str]:
        return self._lines()


Transport = Callable[[str, str, dict, "bytes | None", float], Any]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float) -> _Response:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return _Response(exc.code, lambda: payload, lambda: iter(()))
    except urllib.error.URLError as exc:
        raise ModalClientError(f"Cannot reach {url}: {exc.reason}") from exc

    def lines() -> Iterator[str]:
        with resp:
            for raw in resp:
                yield raw.decode("utf-8", errors="replace").rstrip("\r\n")

    def read() -> bytes:
        with resp:
            return resp.read()

    return _Response(resp.status, read, lines)


def _read_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def load_token(env_file: Path = REPO_ROOT / ".env") -> str:
    """Return ``<id>.<secret>`` from the environment (preferred) or ``env_file``."""
    values = _read_env_file(env_file)
    token_id = os.environ.get("MODAL_PROXY_TOKEN_ID") or values.get("MODAL_PROXY_TOKEN_ID")
    secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET") or values.get("MODAL_PROXY_TOKEN_SECRET")
    if not token_id or not secret:
        raise ModalClientError(
            "Set MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET (environment or .env)"
        )
    return f"{token_id}.{secret}"


def lane_base_url(lane: str) -> str:
    override = os.environ.get(f"PTIT_MODAL_URL_{lane.upper()}")
    if override:
        return override.rstrip("/")
    workspace = os.environ.get("PTIT_MODAL_WORKSPACE", DEFAULT_WORKSPACE)
    return f"https://{workspace}--{APP_NAME}-{lane}.us-east.modal.direct/v1"


def _sse_chunks(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parsed JSON payloads of ``data:`` lines until ``[DONE]``."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def _delta_texts(chunk: dict[str, Any]) -> Iterator[str]:
    for choice in chunk.get("choices") or []:
        content = (choice.get("delta") or {}).get("content")
        if content:
            yield content


class ModalLaneClient:
    def __init__(
        self,
        lane: str,
        *,
        base_url: str | None = None,
        token: str | None = None,
        env_file: Path | None = None,
        timeout_s: float = 300.0,
        ready_timeout_s: float = 1200.0,
        poll_s: float = 10.0,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if lane not in LANE_MODELS:
            raise ValueError(f"Unknown lane {lane!r}; expected one of {sorted(LANE_MODELS)}")
        self.lane = lane
        self.model = LANE_MODELS[lane]
        self.base_url = (base_url or lane_base_url(lane)).rstrip("/")
        self._token = token or load_token(env_file or REPO_ROOT / ".env")
        self.timeout_s = timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.poll_s = poll_s
        self._transport = transport or _urllib_transport
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------ plumbing
    def _post(self, payload: dict[str, Any]):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        start = self._clock()
        while True:
            response = self._transport("POST", url, headers, body, self.timeout_s)
            if response.status != 503:
                break
            if self._clock() - start >= self.ready_timeout_s:
                raise ModalReadyTimeout(f"{self.lane}: still 503 after {self.ready_timeout_s:.0f} s ({url})")
            self._sleep(self.poll_s)
        if response.status in (401, 403):
            raise ModalAuthError(f"{self.lane}: HTTP {response.status} — proxy token rejected ({url})")
        if response.status != 200:
            detail = response.read()[:500].decode("utf-8", errors="replace")
            raise ModalClientError(f"{self.lane}: HTTP {response.status} from {url}: {detail}")
        return response

    def _payload(self, messages: list[dict], params: dict[str, Any]) -> dict[str, Any]:
        return {"model": self.model, "messages": messages, **params}

    # ------------------------------------------------------------ public API
    def chat(self, messages: list[dict], **params: Any) -> ChatResult:
        data = json.loads(self._post(self._payload(messages, {**params, "stream": False})).read())
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        return ChatResult(text=text, usage=data.get("usage") or {}, metrics=data.get("metrics"), raw=data)

    def stream(self, messages: list[dict], **params: Any) -> Iterator[StreamEvent]:
        payload = self._payload(messages, {**params, "stream": True, "stream_options": {"include_usage": True}})
        start = self._clock()
        response = self._post(payload)
        parts: list[str] = []
        usage = metrics = None
        ttft = None
        for chunk in _sse_chunks(response.lines()):
            for content in _delta_texts(chunk):
                if ttft is None:
                    ttft = self._clock() - start
                parts.append(content)
                yield StreamEvent("delta", content)
            if chunk.get("usage"):
                usage, metrics = chunk["usage"], chunk.get("metrics")
        yield StreamEvent("final", "".join(parts), usage=usage, metrics=metrics,
                          ttft_s=ttft, total_s=self._clock() - start)

    def margins(self, messages: list[dict], **params: Any) -> list[TokenMargin]:
        params = {"temperature": 0, **params, "logprobs": True, "top_logprobs": 2}
        result = self.chat(messages, **params)
        choices = result.raw.get("choices") or [{}]
        content = ((choices[0].get("logprobs") or {}).get("content")) or []
        if not content:
            raise LogitMarginUnsupported(f"{self.lane}: response has no logprobs (logit margin not supported)")
        margins = []
        for entry in content:
            top = sorted((t["logprob"] for t in entry.get("top_logprobs") or []), reverse=True)
            if len(top) < 2:
                raise LogitMarginUnsupported(f"{self.lane}: fewer than 2 top_logprobs (logit margin not supported)")
            margins.append(TokenMargin(entry.get("token", ""), float(entry["logprob"]), float(top[0] - top[1])))
        return margins
