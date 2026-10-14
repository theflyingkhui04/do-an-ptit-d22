"""M2-03 diagnostic: observe V1 KV-cache lookups.

Pinned to vLLM 0.30.0.
No modification to the installed vLLM source.
Not for canonical benchmarks.
"""

import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from vllm.v1.core.kv_cache_manager import KVCacheManager

from src.server.adaptive_scheduler import AdaptiveAsyncScheduler


EXPECTED_SHA = (
    "58800535eda854580d84e1025bf32fe01b2a43"
    "ce0bba501b4f3d125e4f2c8afa"
)


def _verify_runtime():
    version = importlib.metadata.version("vllm")
    if version != "0.30.0":
        raise RuntimeError(f"Unsupported vLLM: {version}")

    import vllm.v1.core.kv_cache_manager as module

    path = Path(module.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()

    if actual != EXPECTED_SHA:
        raise RuntimeError(
            f"KVCacheManager source SHA mismatch: {actual}"
        )

    return actual


SOURCE_SHA = _verify_runtime()
_ORIGINAL = KVCacheManager.get_computed_blocks

if getattr(_ORIGINAL, "_m203_wrapped", False):
    raise RuntimeError("M2-03 cache probe installed twice")


def _write_record(record):
    output = os.environ.get("M203_CACHE_LOOKUP_JSONL")
    if not output:
        raise RuntimeError("M203_CACHE_LOOKUP_JSONL missing")

    path = Path(output)
    if not path.is_absolute():
        raise RuntimeError("Diagnostic output must be absolute")

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _observed_lookup(self, request):
    # Preserve original lookup behavior and exception propagation.
    result = _ORIGINAL(self, request)

    try:
        blocks, cached_tokens, boundary = result

        req_id = getattr(request, "request_id", None)
        if not isinstance(req_id, str) or not req_id:
            raise ValueError("Missing request ID")

        prompt_ids = getattr(request, "prompt_token_ids", None)
        prompt_hash = None

        if prompt_ids is not None:
            prompt_hash = hashlib.sha256(
                json.dumps(
                    list(map(int, prompt_ids)),
                    separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()

        grouped_blocks = getattr(blocks, "blocks", ())

        record = {
            "schema": "m2-03.cache-lookup.v1",
            "timestamp_utc":
                datetime.now(timezone.utc).isoformat(),
            "request_id": req_id,
            "request_id_sha256":
                hashlib.sha256(req_id.encode()).hexdigest(),
            "prompt_token_ids_sha256": prompt_hash,
            "request_num_tokens":
                int(request.num_tokens),
            "request_num_computed_tokens":
                int(request.num_computed_tokens),
            "cached_prefix_tokens": int(cached_tokens),
            "shared_prefix_boundary": int(boundary),
            "cached_blocks_per_group": [
                len(group) for group in grouped_blocks
            ],
            "cache_enabled":
                bool(self.enable_caching),
            "source_sha256": SOURCE_SHA,
        }

        _write_record(record)

    except Exception as exc:
        # Diagnostic failure must not change the cache lookup result.
        print(
            f"[M203_CACHE_PROBE_ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    return result


_observed_lookup._m203_wrapped = True
KVCacheManager.get_computed_blocks = _observed_lookup


class M203CacheLookupScheduler(AdaptiveAsyncScheduler):
    """Adaptive scheduler with read-only cache lookup observation."""
    pass
