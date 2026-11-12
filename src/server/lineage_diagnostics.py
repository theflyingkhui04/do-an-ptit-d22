"""Opt-in, best-effort startup observations; never used for authorization."""
import json
import multiprocessing
import os
import time
import uuid
from threading import RLock
from weakref import WeakKeyDictionary

_lock = RLock()
_attempts = WeakKeyDictionary()
_generations = WeakKeyDictionary()


def observe(event, *, attempt=None, registration=None, **fields):
    path = os.environ.get("ADAPTIVE_STARTUP_TELEMETRY_PATH")
    if not path:
        return
    try:
        with _lock:
            if registration is not None:
                attempt = registration.attempt
                if registration not in _generations:
                    _generations[registration] = (uuid.uuid4().hex, uuid.uuid4().hex)
                fields["generation_id"], fields["ledger_id"] = _generations[registration]
            if attempt is not None:
                if attempt not in _attempts:
                    _attempts[attempt] = uuid.uuid4().hex
                fields["attempt_id"] = _attempts[attempt]
            record = dict(event=event, pid=os.getpid(), ppid=os.getppid(),
                          process_name=multiprocessing.current_process().name,
                          start_method=multiprocessing.get_start_method(allow_none=True),
                          timestamp=time.time(), **fields)
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    except Exception:
        # Diagnostics must not turn successful startup into failure.
        pass
