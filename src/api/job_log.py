"""Live log of one Node 1 job for the web console (design D13).

Lines ``{t, src, level, msg[, key][, data]}`` are kept in ``result["events"]`` and saved with the job, so a poll of
``GET /api/jobs/{id}`` shows what each part is doing. ``src`` is ``system``, ``ocr`` or an endpoint id;
``level`` is info, ok, warn or error. A line with a ``key`` (a retry counter, the receiving counter) is
updated in place and moved to the end, so the log stays in time order without one line per retry.
``data`` carries the numbers behind a line (tries, characters, tokens) so the console need not parse text.
Endpoints run in a thread pool, hence the lock around every change and save.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

LEVELS = ("info", "ok", "warn", "error")


class JobLog:
    def __init__(self, store: Any, job_id: str, result: dict[str, Any], *, status: str = "queued",
                 clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.job_id = job_id
        self.result = result
        self.events: list[dict[str, Any]] = result.setdefault("events", [])
        self.status = status
        self._clock = clock
        self._lock = threading.Lock()

    def add(self, src: str, level: str, msg: str, *, key: str | None = None,
            data: dict[str, Any] | None = None) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown log level {level!r}")
        with self._lock:
            line: dict[str, Any] = {"t": self._clock(), "src": src, "level": level, "msg": msg}
            if data is not None:
                line["data"] = data
            if key is not None:
                line["key"] = key
                for index, old in enumerate(self.events):
                    if old.get("key") == key:
                        del self.events[index]
                        break
            self.events.append(line)
            self._save()

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status
            self._save()

    def _save(self) -> None:
        self.store.update_job(self.job_id, self.status, result=self.result)
