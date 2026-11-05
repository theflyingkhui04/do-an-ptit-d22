"""Small dependency-free client for an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenAICompatibleError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint cannot serve a request."""


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 180.0,
        ready_timeout_seconds: float = 0.0,
        poll_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        # > 0: keep retrying HTTP 503 (e.g. a Modal lane cold-starting) for this many seconds.
        self.ready_timeout_seconds = ready_timeout_seconds
        self.poll_seconds = poll_seconds
        self._clock = clock
        self._sleep = sleep

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        start = self._clock()
        while True:
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response_body = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                waited = self._clock() - start
                if exc.code == 503 and waited < self.ready_timeout_seconds:
                    self._sleep(self.poll_seconds)
                    continue
                error_body = exc.read().decode("utf-8", errors="replace")
                raise OpenAICompatibleError(
                    f"HTTP {exc.code} from {request.full_url}: {error_body}"
                ) from exc
            except URLError as exc:
                raise OpenAICompatibleError(f"Cannot reach {request.full_url}: {exc.reason}") from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleError(
                f"Endpoint returned non-JSON content from {request.full_url}"
            ) from exc
        if not isinstance(decoded, dict):
            raise OpenAICompatibleError(f"Endpoint returned a non-object JSON value from {request.full_url}")
        return decoded

    def list_models(self) -> list[str]:
        response = self._request("GET", "/models")
        models = response.get("data") or []
        if not isinstance(models, list):
            raise OpenAICompatibleError("The /models response has a non-list data field")
        return [entry["id"] for entry in models if isinstance(entry, dict) and "id" in entry]

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/chat/completions", payload)
