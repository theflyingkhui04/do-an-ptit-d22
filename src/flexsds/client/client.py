"""Lightweight OpenAI-compatible HTTP client with streaming SSE support."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional


class EndpointUnreachableError(Exception):
    """Raised when the serving endpoint cannot be reached."""


class FlexSDSClient:
    """Client for OpenAI-compatible vLLM serving endpoints."""

    base_url: str
    api_key: str

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"

    def _make_request(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 10,
    ) -> urllib.request.Request:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FlexSDS-Client/0.2.0",
        }
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = json.dumps(data).encode("utf-8") if data is not None else None
        return urllib.request.Request(url, data=body, headers=headers)

    def check_health(self, timeout: int = 3) -> bool:
        """Kiểm tra xem máy chủ có phản hồi không (qua /models hoặc /health)."""
        # Thử endpoint models trước
        try:
            req = self._make_request("models", timeout=timeout)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            pass

        # Thử endpoint root health
        try:
            root_url = self.base_url.rsplit("/v1", 1)[0]
            req = urllib.request.Request(f"{root_url}/health", headers={"User-Agent": "FlexSDS-Client/0.2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_models(self) -> List[str]:
        """Lấy danh sách các mô hình đang được phục vụ trên máy chủ."""
        req = self._make_request("models")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            raise EndpointUnreachableError(f"Cannot reach server at {self.base_url}: {e}") from e

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen2.5-7b-adaptive",
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Gửi yêu cầu trò chuyện và trả về từng token văn bản (streaming SSE)."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        req = self._make_request("chat/completions", data=payload)

        try:
            resp = urllib.request.urlopen(req)
        except Exception as e:
            raise EndpointUnreachableError(f"Failed to connect to {self.base_url}: {e}") from e

        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                chunk_str = line[6:].strip()
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue
