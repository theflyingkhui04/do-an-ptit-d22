"""Streaming OpenAI-compatible caller with per-call speed metrics (TTFT, TPOT, tokens/s).

``LaneCaller.chat_completions`` returns a regular (non-streaming) chat-completion
dict, so ``Node1Extractor`` and ``src/pipeline/ocr.run_ocr`` use it unchanged,
while the stream gives the time to first token. Errors are
``OpenAICompatibleError`` subclasses whose message never contains the token.

An optional ``progress`` hook (set by the Node 1 job, design D13) receives plain dicts: ``waiting`` before each
retry of a cold endpoint, ``first_token``, and ``receiving`` with the characters received so far (at most every
``PROGRESS_EVERY_S`` seconds, plus once at the end). A failing hook never breaks the call.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import httpx

from src.pipeline.openai_compatible import OpenAICompatibleError

PROGRESS_EVERY_S = 2.0


class LLMError(OpenAICompatibleError):
    """A model endpoint could not serve the request."""


@dataclass(frozen=True)
class CallMetrics:
    lane: str
    model: str
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    output_tokens_per_s: float | None
    finish_reason: str | None
    spec_decode: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LaneCaller:
    def __init__(
        self,
        lane: str,
        base_url: str,
        token: str,
        model: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 600.0,
        ready_timeout_s: float = 1200.0,
        poll_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.lane = lane
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))
        self.ready_timeout_s = ready_timeout_s
        self.poll_s = poll_s
        self._clock = clock
        self._sleep = sleep
        self.last_metrics: CallMetrics | None = None
        self.progress: Callable[[dict[str, Any]], None] | None = None

    def _report(self, kind: str, **fields: Any) -> None:
        if self.progress is None:
            return
        try:
            self.progress({"kind": kind, **fields})
        except Exception:  # the console is only an observer; never let it break a model call
            pass

    def _error(self, message: str) -> LLMError:
        text = f"{self.lane}: {message}"
        return LLMError(text.replace(self._token, "***") if self._token else text)

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {**payload, "model": self.model, "stream": True, "stream_options": {"include_usage": True}}
        headers = {"Accept": "text/event-stream"}
        if self._token:  # endpoints started without a key accept anonymous calls
            headers["Authorization"] = f"Bearer {self._token}"
        url = f"{self.base_url}/chat/completions"
        start = self._clock()
        tries = 0
        while True:
            streaming = False
            try:
                with self._client.stream("POST", url, json=body, headers=headers) as response:
                    if response.status_code == 503:
                        response.read()
                    elif response.status_code == 303:
                        # Modal held the request past its 150 s web limit while the GPU cold-starts. Do not
                        # re-post now (each post would queue one more generation): wait on /models first.
                        response.read()
                        self._wait_until_ready(headers, start)
                        continue
                    elif response.status_code != 200:
                        detail = response.read().decode("utf-8", errors="replace")[:300]
                        raise self._error(f"HTTP {response.status_code} from {url}: {detail}")
                    else:
                        streaming = True
                        return self._consume(response, start)
            except httpx.HTTPError as exc:
                if streaming:  # never retry a half-generated answer
                    raise self._error(f"stream interrupted from {url}: {type(exc).__name__}") from exc
                # Before any response a cold-starting Modal server may refuse or reset the connection.
                if self._clock() - start >= self.ready_timeout_s:
                    raise self._error(f"cannot reach {url}: {type(exc).__name__}") from exc
                reason = type(exc).__name__
            else:
                if self._clock() - start >= self.ready_timeout_s:
                    raise self._error(f"not ready after {self.ready_timeout_s:.0f} s (HTTP 503)")
                reason = "HTTP 503"
            tries += 1
            self._report("waiting", tries=tries, waited_s=self._clock() - start, reason=reason,
                         retry_in_s=self.poll_s, ready_timeout_s=self.ready_timeout_s)
            self._sleep(self.poll_s)

    def _wait_until_ready(self, headers: dict[str, str], start: float) -> None:
        """Poll GET /models (cheap, no generation) until the cold-starting endpoint answers 200."""
        url = f"{self.base_url}/models"
        tries = 0
        reason = "HTTP 303"
        while True:
            if self._clock() - start >= self.ready_timeout_s:
                raise self._error(f"not ready after {self.ready_timeout_s:.0f} s ({reason})")
            tries += 1
            self._report("waiting", tries=tries, waited_s=self._clock() - start, reason=reason,
                         retry_in_s=self.poll_s, ready_timeout_s=self.ready_timeout_s)
            self._sleep(self.poll_s)
            try:
                response = self._client.get(url, headers={k: v for k, v in headers.items() if k != "Accept"})
            except httpx.HTTPError as exc:
                reason = type(exc).__name__
                continue
            if response.status_code == 200:
                return
            reason = f"HTTP {response.status_code}"

    def _consume(self, response: httpx.Response, start: float) -> dict[str, Any]:
        parts: list[str] = []
        finish_reason = usage = spec = None
        ttft = None
        chars = 0
        reported_at: float | None = None
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise self._error(f"invalid stream chunk: {exc}") from exc
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    now = self._clock()
                    if ttft is None:
                        ttft = now - start
                        reported_at = now
                        self._report("first_token", ttft_s=ttft)
                    parts.append(content)
                    chars += len(content)
                    if reported_at is not None and now - reported_at >= PROGRESS_EVERY_S:
                        reported_at = now
                        self._report("receiving", chars=chars, elapsed_s=now - start)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
                spec = chunk.get("metrics")
        total = self._clock() - start
        if chars:
            self._report("receiving", chars=chars, elapsed_s=total)
        completion = (usage or {}).get("completion_tokens")
        tpot = None
        if ttft is not None and completion and completion > 1:
            tpot = (total - ttft) * 1000 / (completion - 1)
        self.last_metrics = CallMetrics(
            lane=self.lane, model=self.model, latency_ms=total * 1000,
            ttft_ms=None if ttft is None else ttft * 1000, tpot_ms=tpot,
            prompt_tokens=(usage or {}).get("prompt_tokens"), completion_tokens=completion,
            output_tokens_per_s=(completion / total) if completion and total > 0 else None,
            finish_reason=finish_reason, spec_decode=spec)
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                             "finish_reason": finish_reason}],
                "usage": usage or {}, "model": self.model}
