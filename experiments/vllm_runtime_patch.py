"""[M5-01] Minimal backport of vLLM PR #55708 into the pinned vLLM 0.30.0.

Stock vLLM 0.30.0 can drop a request's speculative-decoding metrics when it
coalesces streamed outputs (``RequestOutput.add(..., aggregate=True)``); the
M3-05 campaign measured 3/320 such records on stock and 0/320 with this patch
(logs/daijestic/M3-05/M3-05_worklogs.md §12-13). The aggregator then refuses
the whole cell, as it must.

The patch only copies ``spec_decode_metrics`` from the newer chunk, exactly as
upstream commit 6c2925fe5fe8156475affbfa995ed272c96fe4fa does. Both the stock
and the patched ``vllm/outputs.py`` are SHA-256 gated: the patched bytes equal
the M3-05 Colab runtime (``vllm-0.30.0-plus-backport-55708``).

Standalone on purpose (no repository imports): the Modal image build runs it as

    python /root/vllm_runtime_patch.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

RUNTIME_LABEL = "vllm-0.30.0-plus-backport-55708"
STOCK_RUNTIME_LABEL = "vllm-0.30.0"
UPSTREAM_PR = 55708
UPSTREAM_COMMIT = "6c2925fe5fe8156475affbfa995ed272c96fe4fa"
VLLM_BASE_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"  # v0.30.0
STOCK_OUTPUTS_SHA256 = "346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425"
PATCHED_OUTPUTS_SHA256 = "8f58684245f17007863382ffa9de115d371738ae28731b0c876196eb3b3fea47"

ANCHOR = "                        completion.cumulative_logprob = (\n"
BACKPORT_BLOCK = (
    "                        if next_completion.spec_decode_metrics is not None:\n"
    "                            completion.spec_decode_metrics = (\n"
    "                                next_completion.spec_decode_metrics\n"
    "                            )\n"
)


def apply_backport_55708(source: str) -> str:
    """Insert the metrics-preservation block before the single anchor line."""
    count = source.count(ANCHOR)
    if count != 1:
        raise ValueError(f"Expected exactly one patch anchor in vllm/outputs.py, found {count}")
    return source.replace(ANCHOR, BACKPORT_BLOCK + ANCHOR)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_file(
    path: Path,
    *,
    stock_sha256: str = STOCK_OUTPUTS_SHA256,
    patched_sha256: str = PATCHED_OUTPUTS_SHA256,
) -> str:
    """Patch ``path`` in place; return ``"patched"`` or ``"already-patched"``.

    Refuses any file that is neither the expected stock nor the expected
    patched bytes, and never writes a result whose hash is not the expected one.
    """
    data = path.read_bytes()
    actual = _sha256(data)
    if actual == patched_sha256:
        return "already-patched"
    if actual != stock_sha256:
        raise ValueError(f"{path}: unexpected sha256 {actual}; expected stock {stock_sha256}")
    patched = apply_backport_55708(data.decode("utf-8")).encode("utf-8")
    if _sha256(patched) != patched_sha256:
        raise ValueError(f"{path}: patched sha256 {_sha256(patched)} != expected {patched_sha256}")
    path.write_bytes(patched)
    return "patched"


def installed_outputs_path() -> Path:
    """``vllm/outputs.py`` of the installed package, located without importing vLLM."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm is not installed")
    return Path(next(iter(spec.submodule_search_locations))) / "outputs.py"


def file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:  # never open a path from the command line; tests call patch_file() directly
        raise SystemExit("vllm_runtime_patch takes no arguments: it patches the installed vllm/outputs.py")
    path = installed_outputs_path()
    status = patch_file(path, stock_sha256=STOCK_OUTPUTS_SHA256, patched_sha256=PATCHED_OUTPUTS_SHA256)
    print(f"[vllm_runtime_patch] {status}: {path} sha256={file_sha256(path)} ({RUNTIME_LABEL})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
