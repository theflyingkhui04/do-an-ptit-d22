The gzip fixture contains five unchanged UTF-8 source files retrieved from
https://github.com/vllm-project/vllm/tree/v0.30.0 on 2026-10-14.
Paths and exact SHA-256 digests are in
`experiments/m2_03_targeted/upstream.json`.

The sources retain their SPDX Apache-2.0 and vLLM copyright headers.
Upstream license: https://github.com/vllm-project/vllm/blob/v0.30.0/LICENSE
Apache-2.0 terms are also included in
`experiments/m2_03_targeted/UPSTREAM_LICENSE.txt`.

These files are data, never imported as vLLM. CPU tests decompress them to
verify the exact insertion anchors, compilation and inference AST invariance.
The compressed JSON uses gzip mtime=0 for reproducibility. No model weights
or generated scientific evidence are included.
