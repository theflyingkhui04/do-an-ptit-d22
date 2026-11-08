"""Warm-up and bounded keep-alive of the model endpoints and the database."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

KEEPALIVE_SECONDS = 240.0   # below Modal's 300 s scale-down window
MAX_WARM_SECONDS = 3600.0   # at most 60 minutes per warm-up request
STATES = ("cold", "starting", "ready", "error")


class WarmupManager:
    def __init__(self, probes: dict[str, Callable[[], Any]], *, clock: Callable[[], float] = time.monotonic,
                 keepalive_s: float = KEEPALIVE_SECONDS, max_s: float = MAX_WARM_SECONDS,
                 background: bool = True, redact: Callable[[str], str] = lambda s: s) -> None:
        self._probes = dict(probes)
        self._clock = clock
        self.keepalive_s = keepalive_s
        self.max_s = max_s
        self._background = background
        self._redact = redact
        self._lock = threading.Lock()
        self._states = {name: {"state": "cold", "message": None} for name in probes}
        self.deadline: float | None = None
        self.last_ping: float | None = None
        self._loop_running = False

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: dict(state) for name, state in self._states.items()}

    def mark(self, name: str, ok: bool, message: str | None = None) -> None:
        if name not in self._states:
            return
        with self._lock:
            self._states[name] = {"state": "ready" if ok else "error",
                                  "message": None if ok else self._redact(str(message))[:300]}

    def _probe(self, name: str) -> None:
        try:
            self._probes[name]()
        except Exception as exc:  # noqa: BLE001  (service boundary: reported, never raised to the user)
            self.mark(name, False, f"{type(exc).__name__}: {exc}")
        else:
            self.mark(name, True)

    def ping_all(self) -> None:
        self.last_ping = self._clock()
        if self._background:
            with ThreadPoolExecutor(max_workers=max(1, len(self._probes))) as pool:
                list(pool.map(self._probe, self._probes))
        else:
            for name in self._probes:
                self._probe(name)

    def request(self) -> dict[str, dict[str, Any]]:
        """Start (or extend) a warm window of ``max_s`` seconds and ping every service now."""
        self.deadline = self._clock() + self.max_s
        with self._lock:
            for state in self._states.values():
                if state["state"] != "ready":
                    state.update(state="starting", message=None)
        if self._background:
            threading.Thread(target=self.ping_all, daemon=True).start()
            if not self._loop_running:
                self._loop_running = True
                threading.Thread(target=self._loop, daemon=True).start()
        else:
            self.ping_all()
        return self.snapshot()

    def tick(self) -> bool:
        """Send keep-alive pings when due inside the warm window; return whether it pinged."""
        now = self._clock()
        if self.deadline is None or now >= self.deadline:
            return False
        if self.last_ping is not None and now - self.last_ping < self.keepalive_s:
            return False
        self.ping_all()
        return True

    def _loop(self) -> None:
        try:
            while self.deadline is not None and self._clock() < self.deadline:
                self.tick()
                time.sleep(10)
        finally:
            self._loop_running = False
