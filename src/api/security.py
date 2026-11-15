"""Demo key check, per-IP rate limits and secret redaction for logs."""

from __future__ import annotations

import hmac
import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable

from fastapi import Request


def key_matches(given: str | None, expected: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding window: at most ``limit`` events per ``window_s`` seconds per key."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> int | None:
        """Record one event; return ``None`` if allowed, else seconds until a slot frees up."""
        now = self._clock()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= self.window_s:
                events.popleft()
            if len(events) >= self.limit:
                return max(1, math.ceil(self.window_s - (now - events[0])))
            events.append(now)
            return None


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


class RedactingJsonFormatter(logging.Formatter):
    """One JSON object per line; any configured secret value is replaced by ``***``."""

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = sorted(set(secrets), key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        entry = {"level": record.levelname, "logger": record.name, "message": record.getMessage()}
        for key in ("job_id", "step", "lane", "duration_ms", "status"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return redact(json.dumps(entry, ensure_ascii=False, default=str), self._secrets)
