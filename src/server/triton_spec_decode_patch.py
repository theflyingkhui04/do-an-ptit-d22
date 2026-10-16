"""Reversible, exact-source vLLM 0.30.0 short-query split-KV patch.

Run with stopped workers: python -m src.server.triton_spec_decode_patch
{status,install,restore} [--vllm-root /path/to/vllm]. No runtime observer import.
Guard arithmetic follows vllm-project/vllm#59448; padding follows #56148.
Upstream source is Apache-2.0; its SPDX headers survive this transform.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile

SOURCE = "v1/attention/ops/triton_unified_attention.py"
PRISTINE_SHA256 = "65d0fbcf33d00789df8ca688c0a1c908e710cfaef53471178747ecae11f05829"

# This is inserted after the unchanged q=1 decision. No new allocations, tensor
# reads, tuning, or changes to existing 2D/q=1 arithmetic. Full causal visibility
# is a prefix: an empty tile cannot precede a later visible tile in a segment.
# Thus the SWA masked-first-tile fix from #59448 is deliberately out of scope.
ADMISSION = '''    # Short causal full-attention verification, bounded to the reviewed family.
    short_query_3d = (
        max_seqlen_q == 3 and num_seqs == 1
        and q.shape[0] == max_seqlen_q and max_seqlen_k > 0
        and causal is True and window_size == (-1, -1)
        and mm_prefix_range is None and not mm_prefix_clamp_sliding_window
        and rswa_prefix_lens is None and rswa_window is None
        and not use_td and not tuned_large_head and not is_batch_invariant
        and kv_quant_mode == KVQuantMode.NONE
        and alibi_slopes is None and not use_alibi_sqrt
        and qq_bias is None and sinks is None and softcap == 0
        and output_scale is None
        and q.dtype == k.dtype == v.dtype == out.dtype == torch.float16
        and q.device.type == "cuda" and k.device == v.device == out.device == q.device
        and seq_threshold_3D is not None and num_seqs <= seq_threshold_3D
        and num_par_softmax_segments is not None
        and num_par_softmax_segments > 0
        and (num_par_softmax_segments & (num_par_softmax_segments - 1)) == 0
    )
    if short_query_3d:
        # Kernels use packed pointer offsets, not tensor strides, for scratch.
        shapes = ((num_query_heads, num_par_softmax_segments, head_size_padded),
                  (num_query_heads, num_par_softmax_segments),
                  (num_query_heads, num_par_softmax_segments))
        short_query_3d = all(
            buf is not None and len(buf.shape) == len(tail) + 1
            and buf.shape[0] >= q.shape[0] and tuple(buf.shape[1:]) == tail
            and buf.dtype == torch.float32 and buf.device == q.device
            and buf.is_contiguous()
            for buf, tail in zip(
                (softmax_segm_output, softmax_segm_max, softmax_segm_expsum), shapes)
        )
    use_3d = use_3d or short_query_3d

'''

# Exact anchors plus whole-file digest; neither diagnostics nor unknown sources
# are accepted. Defaults preserve positional callers and old specializations.
INSERTIONS = (
    ("):\n    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.\n",
     "    _SHORT_QUERY_3D_GUARDS: tl.constexpr = False,\n"),
    ("):\n    query_token_idx = tl.program_id(0)\n",
     "    _SHORT_QUERY_3D_GUARDS: tl.constexpr = False,\n"),
    ("        store_segm_reduce_scalars(\n",
     '        if _SHORT_QUERY_3D_GUARDS:\n'
     '            M = tl.where(L > 0.0, M, float("-inf"))\n'),
    ("    seq_idx = find_seq_idx(\n",
     '    if _SHORT_QUERY_3D_GUARDS:\n'
     '        if query_token_idx >= tl.load(query_start_len_ptr + num_seqs):\n'
     '            return\n\n'),
    ("    # number of segments for this particular sequence\n",
     '    if _SHORT_QUERY_3D_GUARDS:\n'
     '        if seq_len <= 0:\n'
     '            return\n\n'),
    ("    # load and rescale segment exp sums\n",
     '    if _SHORT_QUERY_3D_GUARDS:\n'
     '        overall_max = tl.where(overall_max > float("-inf"), overall_max, 0.0)\n\n'),
    ("    # The kernel signature is the same for 2D and 3D ", ADMISSION),
    ("        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,\n",
     "        _SHORT_QUERY_3D_GUARDS=short_query_3d,\n"),
    ("            USE_FP8=output_scale is not None,\n",
     "            _SHORT_QUERY_3D_GUARDS=short_query_3d,\n"),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def transform(raw):
    if digest(raw) != PRISTINE_SHA256:
        raise ValueError("expected pristine pinned vLLM 0.30.0 attention source")
    text = raw.decode("utf-8")
    for anchor, insertion in INSERTIONS:
        if text.count(anchor) != 1:
            raise ValueError("non-unique source anchor")
        text = text.replace(anchor, insertion + anchor, 1)
    result = text.encode("utf-8")
    compile(result, SOURCE, "exec")
    return result


def pristine(raw):
    """Recover only our exact patch; refuse partial/foreign/diagnostic edits."""
    if digest(raw) == PRISTINE_SHA256:
        return raw
    text = raw.decode("utf-8")
    for anchor, insertion in reversed(INSERTIONS):
        if text.count(insertion + anchor) != 1:
            raise ValueError("unknown/partial attention patch; refusing mutation")
        text = text.replace(insertion + anchor, anchor, 1)
    original = text.encode("utf-8")
    if digest(original) != PRISTINE_SHA256 or transform(original) != raw:
        raise ValueError("source hash drift; refusing mutation")
    return original


def apply(root, action):
    """One file, atomic replacement, no installation as a side effect of serving."""
    path = Path(root) / SOURCE
    before = path.read_bytes()
    original = pristine(before)
    desired = transform(original) if action == "install" else original
    if action not in ("status", "install", "restore"):
        raise ValueError("unknown action")
    if action != "status" and before != desired:
        fd, pending = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(desired)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(pending, path.stat().st_mode)
            if path.read_bytes() != before:
                raise ValueError("source changed before replacement")
            os.replace(pending, path)
        finally:
            if os.path.exists(pending):
                os.unlink(pending)
    actual = path.read_bytes()
    if actual != (before if action == "status" else desired):
        raise ValueError("source changed after operation")
    return {"state": "PRISTINE" if actual == original else "PATCHED",
            "source": str(path), "sha256": digest(actual),
            "pristine_sha256": PRISTINE_SHA256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "install", "restore"))
    parser.add_argument("--vllm-root", type=Path)
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        if importlib.metadata.version("vllm") != "0.30.0":
            parser.error("vLLM 0.30.0 required")
        spec = importlib.util.find_spec("vllm")
        root = Path(spec.origin).parent
    print(json.dumps(apply(root, args.action), sort_keys=True))


if __name__ == "__main__":
    main()
