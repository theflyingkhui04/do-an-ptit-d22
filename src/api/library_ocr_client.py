"""Client of the library OCR server (Modal app ``ptit-ocr-lib``, engine A: PaddleOCR detection + VietOCR).

Same cold-start handling as ``LaneCaller``: HTTP 503 or a connection reset before
any response means "still starting" and is retried until ``ready_timeout_s``.
Errors never contain the proxy token.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.api.llm import LLMError
from src.pipeline.ocr import OcrError


@dataclass(frozen=True)
class LibraryOcrResult:
    text: str
    boxes: int
    latency_ms: float
    engine: str
    versions: dict[str, Any] = field(default_factory=dict)


class LibraryOcrClient:
    def __init__(self, base_url: str, token: str, *, client: httpx.Client | None = None,
                 timeout_s: float = 300.0, ready_timeout_s: float = 1200.0, poll_s: float = 10.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))
        self.ready_timeout_s = ready_timeout_s
        self.poll_s = poll_s
        self._clock = clock
        self._sleep = sleep

    def _error(self, message: str) -> LLMError:
        return LLMError(f"ocr: {message}".replace(self._token, "***"))

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self._token}"}
        start = self._clock()
        while True:
            try:
                response = self._client.request(method, url, headers=headers, **kwargs)
            except httpx.HTTPError as exc:  # cold start: refused or reset before any response
                if self._clock() - start >= self.ready_timeout_s:
                    raise self._error(f"cannot reach {url}: {type(exc).__name__}") from exc
            else:
                if response.status_code != 503:
                    return response
                if self._clock() - start >= self.ready_timeout_s:
                    raise self._error(f"not ready after {self.ready_timeout_s:.0f} s (HTTP 503)")
            self._sleep(self.poll_s)

    def health(self) -> bool:
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise self._error(f"health HTTP {response.status_code}")
        return True

    def read(self, data: bytes) -> LibraryOcrResult:
        response = self._request("POST", "/v1/ocr", json={"image_b64": base64.b64encode(data).decode("ascii")})
        if response.status_code == 422:
            detail = response.json().get("detail", "no text detected")
            raise OcrError(str(detail))
        if response.status_code != 200:
            raise self._error(f"HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        return LibraryOcrResult(text=body["text"], boxes=int(body.get("boxes", 0)),
                                latency_ms=float(body.get("latency_ms", 0.0)), engine=body.get("engine", ""),
                                versions=body.get("versions") or {})
