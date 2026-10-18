# Boundary observer source fixture

Four unmodified official vLLM v0.30.0 Python sources, selected from archive
`a7b52e7ee7cd1ce315f3c126cb5c0d8fa0e4ca25`'s
`tests/fixtures/m203_v030_sources.json.gz`. No experiment outputs are included.

`experiments/protocol_b_boundary_sources.py` pins their SHA-256 hashes. CPU tests
apply the observation-only insertions to these real sources, compile the result,
and verify that stripping observation calls recovers the original AST.

Upstream: https://github.com/vllm-project/vllm/tree/v0.30.0
License: Apache-2.0, included in `protocol_b_boundary_LICENSE.txt`.
