# Boundary observer source fixture

Four unmodified official vLLM v0.30.0 Python sources, selected from archive
`a7b52e7ee7cd1ce315f3c126cb5c0d8fa0e4ca25`'s
`tests/fixtures/m203_v030_sources.json.gz`. No experiment outputs are included.

`experiments/protocol_b_boundary_sources.py` pins their SHA-256 hashes. CPU tests
apply the observation-only insertions to these real sources, compile the result,
and verify that stripping observation calls recovers the original AST.

Upstream: https://github.com/vllm-project/vllm/tree/v0.30.0
License: Apache-2.0, included in `protocol_b_boundary_LICENSE.txt`.

## Executed V2 sampler

`protocol_b_boundary_v2_v030.json.gz` contains only the unmodified V2 sampler
from checkpoint `31d31f242af55faecb262114e738aa0340c07429`, under
`experiments/runs/m2_03/boundary279_20261018T115933Z_0267339f/vllm-v2-executed-sources/`.
Its SHA-256 is `832c9945d201a1e730ff9b4b52076aa83bd4a5064e1e4cfea952cf84d4a8d1f0`.
The existing V1 fixture and historical experiment outputs are unchanged.
