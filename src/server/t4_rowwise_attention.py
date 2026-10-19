"""Exact, opt-in production transform for the pinned vLLM 0.30.0 sources.

This module does not import vLLM or launch a model. Historical observer transforms
remain separate. Render the resulting upstream diff with ``python -m
src.server.t4_rowwise_attention diff --source-root PATH`` before installation.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


PROFILE = "m203.t4-rowwise-segmented3d.v2"
BACKEND = "v1/attention/backends/triton_attn.py"
OPS = "v1/attention/ops/triton_unified_attention.py"
DRAFT = "v1/spec_decode/draft_model.py"
HASHES = {
    BACKEND: "1057663d110e1d5229495a799027b27277c1eca325d5b6780fa2259afbf58a1a",
    OPS: "65d0fbcf33d00789df8ca688c0a1c908e710cfaef53471178747ecae11f05829",
    DRAFT: "6b884f49a75f01d1aa6b2d27159b63fa27f7ba7f0518d2810d1a5b565e14f026",
}
RUNTIME_FIXES = {DRAFT: "draft-tokenizer-revision-v1"}
DRAFT_TOKENIZER_CALL = (
    "            draft_tokenizer = get_tokenizer(\n"
    "                spec.draft_model_config.model,\n"
    "                trust_remote_code=spec.draft_model_config.trust_remote_code,\n"
    "            )\n"
)
DRAFT_REVISION_LINE = "                revision=spec.draft_model_config.revision,\n"
HELPER = "v1/attention/ops/triton_attention_helpers.py"
DEPENDENCY_HASHES = {
    HELPER: "35aa993d01345f535d2641cf9dcabb8b64c51b20a19707bc5987e616aaa87ddd",
}
SOURCE_HASHES = HASHES | DEPENDENCY_HASHES

# Embedded into the upstream ops module, so installed inference has no dependency
# on project code, observers, environment flags, or a Python attention fallback.
ROW_STATE = '''@triton.jit
def _rowwise_decode_state(query_start_len_ptr, seq_lens_ptr, token_idx, seq_idx):
    start = tl.load(query_start_len_ptr + seq_idx)
    stop = tl.load(query_start_len_ptr + seq_idx + 1)
    final_len = tl.load(seq_lens_ptr + seq_idx)
    return final_len - (stop - start) + (token_idx - start) + 1


'''

ROW_ADMISSION = '''    # Opt-in target-only policy. Shapes/configuration are host-known: no
    # GPU scalar reads, scratch allocation, or request metadata mutation.
    rowwise_3d = (
        rowwise_decode_max_q > 1
        and 1 < max_seqlen_q <= rowwise_decode_max_q
        and num_seqs == 1
        and q.shape[0] == max_seqlen_q
        and cu_seqlens_q.shape == (2,)
        and max_seqlen_k >= max_seqlen_q
        and current_platform.is_cuda()
        and current_platform.get_device_capability() is not None
        and current_platform.get_device_capability().to_int() == 75
        and q.device.type == "cuda"
        and q.dtype == k.dtype == v.dtype == out.dtype == torch.float16
        and q.device == k.device == v.device == out.device
        and q.shape[1:] == (28, 128) and num_kv_heads == 4
        and causal is True
        and tuple(window_size) == (-1, -1)
        and kv_quant_mode == KVQuantMode.NONE
        and q_descale is None and output_scale is None
        and alibi_slopes is None and not use_alibi_sqrt
        and qq_bias is None and sinks is None and softcap == 0
        and mm_prefix_range is None
        and rswa_prefix_lens is None and rswa_window is None
        and chunk_lookback == -1 and not use_td
        and not mm_prefix_clamp_sliding_window
        and not is_batch_invariant
        and seq_threshold_3D is not None and num_seqs <= seq_threshold_3D
        and num_par_softmax_segments == 16
        and all(t is not None for t in
                (softmax_segm_output, softmax_segm_max, softmax_segm_expsum))
        and softmax_segm_output.shape[1:] == (28, 16, 128)
        and softmax_segm_max.shape[1:] == softmax_segm_expsum.shape[1:] == (28, 16)
        and all(t.shape[0] >= q.shape[0] and t.dtype == torch.float32
                and t.device == q.device and t.is_contiguous() for t in
                (softmax_segm_output, softmax_segm_max, softmax_segm_expsum))
        and cu_seqlens_q.device == seqused_k.device == block_table.device == q.device
    )
    use_3d = use_3d or rowwise_3d

'''


def _replace(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("Missing/nonunique pinned source anchor: " + old[:100])
    return text.replace(old, new, 1)


def transform(name: str, raw: bytes) -> bytes:
    """Accept only pristine pinned bytes; emit one exact production transform."""
    if name not in SOURCE_HASHES or hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError("Pristine vLLM source drift: " + name)
    if name in DEPENDENCY_HASHES:
        return raw
    text = raw.decode("utf-8")
    if name == DRAFT:
        # Exact previously approved runtime fix, without diagnostic hooks.
        text = _replace(text, DRAFT_TOKENIZER_CALL,
                        DRAFT_TOKENIZER_CALL.replace("                trust_remote_code=",
                                                     DRAFT_REVISION_LINE + "                trust_remote_code="))
    elif name == BACKEND:
        text = _replace(text, "    # Optional aot scheduling\n",
                        "    rowwise_decode_max_q: int = 0\n\n    # Optional aot scheduling\n")
        text = _replace(text, "        model_config = vllm_config.model_config\n", '''        model_config = vllm_config.model_config
        spec = vllm_config.speculative_config
        # Draft configs replace model_config with draft_model_config. Enable
        # only the target; ordinary Vanilla decode retains its original path.
        self.rowwise_decode_max_q = (
            1 + spec.num_speculative_tokens
            if spec is not None and model_config is spec.target_model_config
            and model_config.dtype == torch.float16
            and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
            else 0
        )
''')
        text = _replace(text, "            num_actual_tokens=num_actual_tokens,\n",
                        "            rowwise_decode_max_q=self.rowwise_decode_max_q,\n"
                        "            num_actual_tokens=num_actual_tokens,\n")
        text = _replace(text, "            max_seqlen_q=max_seqlen_q,\n",
                        "            rowwise_decode_max_q=attn_metadata.rowwise_decode_max_q,\n"
                        "            max_seqlen_q=max_seqlen_q,\n")
    else:
        text = _replace(text, "@triton.jit\ndef kernel_unified_attention(",
                        ROW_STATE + "@triton.jit\ndef kernel_unified_attention(")
        text = _replace(text, "    IS_3D: tl.constexpr,\n",
                        "    IS_3D: tl.constexpr,\n    ROWWISE_DECODE: tl.constexpr,\n")
        old = '''    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )
'''
        text = _replace(text, old, '''    if ROWWISE_DECODE:
        # Grid axis 0 is the original flattened query-token index. Present
        # exactly one logical query to the unchanged decode arithmetic below.
        seq_idx = find_seq_idx(
            query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, False
        )
        seq_len = _rowwise_decode_state(
            query_start_len_ptr, seq_lens_ptr, q_block_global_idx, seq_idx
        )
        q_block_local_idx = 0
        cur_batch_in_all_start_index = q_block_global_idx
        cur_batch_query_len = 1
    else:
''' + "".join("    " + line if line.strip() else line for line in old.splitlines(keepends=True)))
        text = _replace(text, "    USE_FP8: tl.constexpr,  # bool\n",
                        "    USE_FP8: tl.constexpr,  # bool\n    ROWWISE_DECODE: tl.constexpr,\n")
        text = _replace(text, "    seq_len = tl.load(seq_lens_ptr + seq_idx)\n", '''    if ROWWISE_DECODE:
        seq_len = _rowwise_decode_state(
            query_start_len_ptr, seq_lens_ptr, query_token_idx, seq_idx
        )
    else:
        seq_len = tl.load(seq_lens_ptr + seq_idx)
''')
        text = _replace(text, "    mm_prefix_clamp_sliding_window: bool = False,\n",
                        "    mm_prefix_clamp_sliding_window: bool = False,\n"
                        "    rowwise_decode_max_q: int = 0,\n")
        text = _replace(text, "    # The kernel signature is the same for 2D and 3D",
                        ROW_ADMISSION + "    # The kernel signature is the same for 2D and 3D")
        text = _replace(text, "    grid: tuple[Any, ...]\n",
                        "    if rowwise_3d:\n        total_num_q_blocks = q.shape[0]\n\n"
                        "    grid: tuple[Any, ...]\n")
        text = _replace(text, "        IS_3D=use_3d,\n",
                        "        IS_3D=use_3d,\n        ROWWISE_DECODE=rowwise_3d,\n")
        text = _replace(text, "            USE_FP8=output_scale is not None,\n",
                        "            USE_FP8=output_scale is not None,\n"
                        "            ROWWISE_DECODE=rowwise_3d,\n")
    result = text.encode("utf-8")
    compile(result, name, "exec")
    return result


def validate_transform(name: str, raw: bytes, patched: bytes) -> None:
    if patched != transform(name, raw):
        raise ValueError("Unapproved production runtime transform: " + name)


def read_sources(root: Path) -> dict[str, bytes]:
    originals = {name: (root / name).read_bytes() for name in SOURCE_HASHES}
    for name, raw in originals.items():
        transform(name, raw)  # Authenticate the complete set before any write.
    return originals


def source_diff(originals: dict[str, bytes]) -> str:
    return "".join("".join(difflib.unified_diff(
        raw.decode().splitlines(keepends=True), transform(name, raw).decode().splitlines(keepends=True),
        fromfile="a/vllm/" + name, tofile="b/vllm/" + name))
        for name, raw in originals.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diff", "install", "verify", "restore"))
    parser.add_argument("--source-root", type=Path, required=True,
                        help="vllm package directory, not repository root")
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if args.command == "diff":
        print(source_diff(read_sources(args.source_root)), end="")
        return
    if args.evidence_dir is None:
        parser.error("--evidence-dir is required")
    # Reuse the project's atomic writer; never weaken historical source profiles.
    from experiments.protocol_b_boundary_sources import _write_atomically

    root = args.source_root.resolve()
    evidence = args.evidence_dir
    if args.command == "install":
        originals = read_sources(root)
        patched = {name: transform(name, originals[name]) for name in HASHES}
        evidence.mkdir(parents=True, exist_ok=False)
        for name, raw in originals.items():
            dest = evidence / "originals" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        manifest = dict(profile=PROFILE, source_root=str(root), original_hashes=SOURCE_HASHES,
                        runtime_fixes=RUNTIME_FIXES,
                        patched_hashes={n: hashlib.sha256(b).hexdigest() for n, b in patched.items()})
        (evidence / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        for name, raw in patched.items():
            _write_atomically(root / name, raw)
    else:
        manifest = json.loads((evidence / "manifest.json").read_text())
        originals = read_sources(evidence / "originals")
        expected = {name: transform(name, originals[name]) for name in HASHES}
        if (manifest.get("profile") != PROFILE or manifest.get("source_root") != str(root)
                or manifest.get("original_hashes") != SOURCE_HASHES
                or manifest.get("runtime_fixes") != RUNTIME_FIXES
                or manifest.get("patched_hashes") != {n: hashlib.sha256(b).hexdigest() for n, b in expected.items()}):
            raise ValueError("Production source manifest drift")
        # Preflight all files before restoring any; unknown mutations fail closed.
        for name, original in originals.items():
            raw = expected.get(name, original)
            accepted = (raw, original) if args.command == "restore" else (raw,)
            if (root / name).read_bytes() not in accepted:
                raise ValueError("Installed production source drift: " + name)
        if args.command == "restore":
            for name in HASHES:
                raw = originals[name]
                _write_atomically(root / name, raw)
    print(PROFILE + ": " + args.command + " complete")


if __name__ == "__main__":
    main()
