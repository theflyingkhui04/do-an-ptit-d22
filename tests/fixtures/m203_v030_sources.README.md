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

The separate `m203_v030_v2_sources.json.gz` fixture contains eleven official
vLLM v0.30.0 files retrieved on 2026-10-16. Exact paths and SHA-256 hashes are in
`experiments/m2_03_targeted/upstream_v2.json`. All SPDX/license headers remain;
the same Apache-2.0 license above applies. gzip mtime is zero. These sources are
data for source review and CPU tests, never imported as a vLLM runtime.

V2 source inspection identified the input/slot preparation and raw/selected-token
boundaries in `gpu/model_runner.py`, final ordinary processing in
`gpu/sample/sampler.py`, processed verification in
`gpu/spec_decode/rejection_sampler.py`, and actual FULL capture/replay in
`gpu/cudagraph_utils.py`. The remaining fixture files pin the input/state/KV and
attention metadata contracts used by the observer. Stripping only the inserted
observer calls/imports must recover each original inference AST. Production
attention arithmetic/routing is not ported or rewritten.
