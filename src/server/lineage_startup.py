"""Project-owned, exact-source-gated boundary around pinned EngineCore startup.

Importing the custom executor installs this before EngineCore construction:
pinned LLMEngine/AsyncLLM resolve Executor.get_class before creating the client.
No upstream package files are modified. All constructor invocations get a fresh
context, including plain executors, so nested construction cannot inherit one.
Ordinary engines do not publish or require lineage state.
"""
from functools import wraps
import hashlib
import json
from pathlib import Path
from threading import RLock

from src.server.lineage import _engine_startup


_installation_lock = RLock()


def install_enginecore_startup_binding():
    import vllm
    from vllm.v1.engine.core import EngineCore

    with _installation_lock:
        original = EngineCore.__init__
        if getattr(original, "_lineage_startup_binding", False):
            return
        manifest = Path(__file__).resolve().parents[2] / "experiments/scalar_vllm_source_manifest.json"
        record = next(r for r in json.loads(manifest.read_text())["files"]
                      if r["path"] == "vllm/v1/engine/core.py")
        path = Path(vllm.__file__).resolve().parent / "v1/engine/core.py"
        if (vllm.__version__ != "0.30.0"
                or Path(original.__code__.co_filename).resolve() != path
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise RuntimeError("Unsupported EngineCore source for lineage startup binding")

        @wraps(original)
        def initialize(self, vllm_config, executor_class, *args, **kwargs):
            with _engine_startup(vllm_config):
                return original(self, vllm_config, executor_class, *args, **kwargs)

        initialize._lineage_startup_binding = True
        EngineCore.__init__ = initialize
