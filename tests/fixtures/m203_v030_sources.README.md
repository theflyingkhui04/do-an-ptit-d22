The gzip fixture contains six unchanged UTF-8 source files retrieved from
https://github.com/vllm-project/vllm/tree/v0.30.0 on 2026-10-14.
The R15 addition is `v1/attention/ops/triton_unified_attention.py`, retrieved
from the same official tag on 2026-10-15. Paths and exact SHA-256 digests are in
`experiments/m2_03_targeted/upstream.json`.

The sources retain their SPDX Apache-2.0 and vLLM copyright headers.
Upstream license: https://github.com/vllm-project/vllm/blob/v0.30.0/LICENSE
Apache-2.0 terms are also included in
`experiments/m2_03_targeted/UPSTREAM_LICENSE.txt`.

These files are data, never imported as vLLM. CPU tests decompress them to
verify the exact insertion anchors, compilation and inference AST invariance.
The compressed JSON uses gzip mtime=0 for reproducibility. No model weights
or generated scientific evidence are included.

R16 ports only the non-DiffKV empty-segment epilogue and reducer clamp described
in https://github.com/vllm-project/vllm/pull/59448. The related merged work is
https://github.com/vllm-project/vllm/pull/56148; its general admission policy,
scratch reallocations and helper refactor are not part of this diagnostic.
The pristine fixture remains unchanged.

`sources.py` adds `_M203_Q3_GUARDS=False` constexpr parameters to the main kernel
and reducer, setting them true only on the boundary-only forced launch. The
epilogue replaces the maximum of a zero-mass segment with negative infinity;
the reducer replaces an all-negative-infinity overall maximum with zero before
exponentiation. Existing zero-denominator handling then returns zero. Disabled
specializations retain the original arithmetic.

R16 accepts only its unpadded three-token, causal, full-attention boundary with
559 KV tokens and sufficient existing scratch rows. It excludes sliding windows,
R-SWA, multimodal prefix masks and tensor descriptors. Therefore the additional
masked-first-tile/sliding-window and graph-padding hardening discussed upstream
is outside this diagnostic. No BLOCK_M/BLOCK_Q or segment-count tuning is ported.
CPU tests exercise the inserted guard arithmetic and launch plumbing; they do
not compile Triton or establish numerical correctness on a T4.
