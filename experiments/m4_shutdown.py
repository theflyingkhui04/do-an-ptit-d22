"""Order pinned vLLM frontend teardown; never alter inference or filter logs."""
from functools import wraps
import hashlib
from pathlib import Path

ASYNC_LLM_SHA256 = "52cbf404f1dbc99c6bebf4a50a7eb18ff842a1f53106a45f365523192d816be1"


def cancel_before_shutdown(original, cancel):
    @wraps(original)
    def shutdown(self, *args, **kwargs):
        handler = getattr(self, "output_handler", None)
        if handler is not None:
            # Upstream does this only AFTER killing the engine. When shutdown
            # runs in an executor thread, the live loop can observe EngineDead
            # before that cancellation. Cancel the consumer before its producer.
            cancel(handler)
        return original(self, *args, **kwargs)
    return shutdown


def install():
    import vllm.v1.engine.async_llm as module
    from vllm.utils.async_utils import cancel_task_threadsafe
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != ASYNC_LLM_SHA256:
        raise RuntimeError("M4 shutdown adapter requires exact pinned AsyncLLM source")
    module.AsyncLLM.shutdown = cancel_before_shutdown(module.AsyncLLM.shutdown, cancel_task_threadsafe)

