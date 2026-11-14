# Final layer-0 internal localization (M4-02-06)

## Scope and evidence ceiling

Parent: `83dbfabe953114fc6023b7c8136bc38ca965f150` on
`fix/m4-02-06-greedy-repro`. This is one opt-in diagnostic, not a repair.
No GPU, readiness, benchmark, PR, merge, or Trello action was executed here.
GPU validation pending.

The specified canonical archive
`m4-02-06-layer-boundaries-20261114T065957207026Z-83dbfabe` was not found
in the local evidence, Downloads, or Codex visualization locations checked.
No other run was substituted. Its reported exact embedding equality, first
layer-0 divergence, and sequential/forced 4164/3752 flip are user-supplied
context, not newly audited raw evidence. The future capture must independently
reproduce the historical prefix, generation 549, position 2070 and flip.
`analyze_trajectory` verifies source/model/grammar/config/prefix provenance and
actual greedy sample/raw argmax before the internal analyzer accepts evidence.

## Pinned path and implementation authority

The 40 exact SHA256/byte-length/source URL records are in
`experiments/m4_layer0_probe_sources.json`. All 36 prior layer-probe hashes
are preserved; activation, linear-kernel selection, Marlin, and Marlin utilities
are added. Source copies are local audit material, not committed dependencies.
The requested `rotary_embedding.py` is a package in v0.30.0; its reviewed
`__init__.py` and `base.py` are the applicable files.

[Qwen2 v0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen2.py)
executes embedding, layer 0 with residual None, input RMSNorm, packed QKV
linear (bias handled by the original module), split by q_size/kv_size/kv_size,
RoPE on Q/K, original attention (including original KV update/consumer path),
o_proj, post-attention fused residual add/RMSNorm, gate_up projection,
SiLU-and-multiply, down projection, then return hidden/residual. Remaining
layers and final norm execute their original modules without internal captures.
There is no fused input residual merge at layer 0 on this single-rank path.
QK normalization, dual-chunk, noncausal, multi-rank and alternate implementations
fail closed. Runtime selection must already be AutoAWQMarlinLinearMethod with
MarlinLinearKernel, FlashAttentionImpl/FLASH_ATTN, ordinary RotaryEmbedding,
RMSNorm and SiluAndMul. The diagnostic does not select a different backend.

The copied single-rank forward preserves original operators, arguments and
ordering. It adds only independent selected-row gathers/clones and auxiliary
returns. It never reimplements RMSNorm, quantized GEMM, RoPE, attention or MLP
arithmetic. Module source hashes and installed implementation identities are
verified before capture. Native CUDA numerical execution remains unvalidated.

## Capture and graph safety

Explicit flag: `--diagnostic-mode decode_trajectory --localization-stage layer0_internal`.
Default engines have no worker override. No production benchmark path imports
or enables this observer. Model/tokenizer/runtime/model revisions, scheduler,
chunking, asynchronous scheduling and FULL_AND_PIECEWISE modes remain frozen.

The existing after-KV-allocation/before-warmup GraphLayerRuntime owns the
temporary model subclass, new forward code object, compiler reset, nonpersistent
device row buffer, graph cleanup and restoration. Internal observations are
returned by the real compiled/captured forward, not Python hooks installed
after capture. The row buffer stays dynamic through export. FULL replay and
PIECEWISE execution return the same auxiliary schema. CPU torch.export checks
this tensor surface only; they do not validate CUDA graph replay.

Ordered records (only layer 0): input; input_norm; packed qkv_projection;
q_post_rope; k_post_rope; v; attention_output; o_proj; post_attention_norm;
post_attention_residual; gate_up; activation; down_proj; return_hidden;
return_residual. Input residual is None, documented rather than fabricated.
The last two duplicate the returned components to validate their identity
before subsequent fused mutations. No internal KV-update boundary is exposed:
attention_output includes the KV consumer/kernel interval. No full KV dump.

Packed QKV is independently gathered/cloned BEFORE RoPE can mutate its split
views. Semantic slices are Q=[0:q_size], K=[q_size:q_size+kv_size],
V=[q_size+kv_size:q_size+2*kv_size]. Preserve these unequal widths; for this
model they are 3584/512/512, packed 4608. Post-RoPE Q/K and V are independent
rows. Independent immutable CPU byte slices retain pre-RoPE Q/K/V with their own
semantic names, shapes, offsets and SHA256, validated against the packed row.
They add 9 KiB per arm in this model and require no extra GPU row storage.

Each record carries condition, request ID, generation index, logical position,
logical-prefix hash, source query row, stage, layer index, actual captured row
shape/dtype/stride/layout, exact BF16 bytes, SHA256 and finite counts. Layout
is the independent captured row layout, not a claim about source-view stride.
A device stamp validates row, position, token and padded execution length.
For forced prefill row 22 at position 2070 is selected, not padding row 23;
for sequential decode row 0 is selected. Query ownership uses the request's
scheduled query length, not a padded tensor's last row.

Outside the compiled model but inside the real runner's forward context, the
observer independently clones the layer-0 consumer slot mapping at this same
row, on the same stream. It is appended to the device stamp and frozen with the
first-mismatch latch. This is an observation of the mapping actually supplied to
attention, not a recomputation from intended blocks. Computed-token count,
scheduled-query count, padded shape/graph mode, request-owned block IDs plus
canonical SHA256, and installed attention/backend identities accompany it.
Block IDs identify the scheduler's request table, not an independently dumped
GPU block-table tensor. They must not be interpreted as proof of KV correctness.

All selected surfaces are frozen before later async forwards can reuse graph
outputs. Serialization runs only after abort/barrier. Duplicate/missing/order,
owner, position, dtype/shape/layout, byte-hash and nonfinite failures invalidate
evidence. Cleanup clears owned graphs, restores hooks/model/compiler state,
releases buffers and verifies unchanged KV allocation, including exception and
partial-install paths. Existing run lifecycle gates also require engine release.

Widths h=3584, q=3584, kv=512, m=18944 give 98,304 BF16 elements:
196,608 bytes (192 KiB) per complete layer-0 row table, plus a 40-byte stamp.
Two arms retain 384 KiB of raw internal state. Temporary return/table/latch
copies add a few such buffers; each captured graph can also retain its return
storage. This excludes existing logits, compiler workspaces and allocator
caching. Memory release is verified by existing lifecycle checks, not assumed.

## Analysis and disposition

Primary equality is exact BF16 bytes. Each stage independently reports byte
and numerical differing-element counts, first differing index, max/mean absolute
difference and NaN/Inf counts. Signed zero differs in bytes even if numerically
equal. No tolerance turns differences into equality.

An adjacent observed transition permits FIRST_INTERNAL_DIVERGENCE for input
norm, QKV projection, RoPE, attention output, o_proj, post-attention norm, or a
specific MLP boundary. Incoming-state differences or inconsistent duplicate
return surfaces yield INTERNAL_DIVERGENCE_INTERVAL rather than an invented
operation. Equal QKV and post-RoPE current rows with different attention output
localize the consumer/history interval only: historical KV, metadata, slot
mapping and numerical attention kernels remain distinct possible causes.
Different packed QKV rules out KV contents as the first current-row divergence.
No internal operation may be claimed before the future GPU result.

Source-based next-state recommendation:
`EXPECTED_NUMERICAL_PATH_DEPENDENCE_CANDIDATE`.
[AutoAWQ selection](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/auto_awq.py)
explicitly excludes Marlin from batch-invariant mode; this is evidence that
shape-dependent numerical paths are legitimate candidates, not proof that this
discrepancy is expected. Different query lengths/graph geometry alone establish
no correctness-contract violation. The future result must be dispositioned as
expected numerical behavior or a supported defect hypothesis. If a defect is
supported, specify only the smallest causal counterfactual for the implicated
state/operation; do not add another broad localization scan.

The project criterion remains the frozen greedy lossless contract between
Vanilla and the speculative treatments on their intended benchmark paths,
including original grammar, prompt, sampling and EOS semantics. Bit-identical
forced-prefill/decode internals are not by themselves a stated model requirement.
Accepting expected numerical behavior must explicitly reconcile the observed
mismatch with that project criterion; this diagnostic does not relax lossless
checks or authorize an exception. No production fix without causal evidence.

## Minimum full-sweep exit path

1. Run this single controlled layer-0 diagnostic at its immutable published SHA.
2. Accept a technical disposition consistent with project correctness: documented
   expected numerical behavior, or supported defect with a narrow causal test.
3. Repair only if required, and verify the repair.
4. Use an intended benchmark SHA whose production path has no active diagnostic
   instrumentation; rerun canonical readiness with the frozen protocol unchanged.
5. Only after canonical readiness PASSes may the frozen full sweep be authorized.

Scientific ceiling remains H_TRAJECTORY_SUPPORTED (prior supplied evidence),
FIRST_DIVERGENT_LAYER=layer_0, ROOT_CAUSE_NOT_PROVEN, FULL_SWEEP_BLOCKED.
Prepare the minimal A100 run after publication; do not launch it here.


## Technical review and CPU validation

Findings first: no unresolved BLOCKER or MAJOR. One MINOR serialization finding
was resolved in `serialize_internal`: a contiguous one-row view can retain the
flattened table's parent stride. That made a correct byte snapshot fail its
layout contract. The initial targeted suite exposed it; canonical independent
1-D cloning plus reshape now preserves the declared row stride. Exact-byte,
shape/layout rejection and both-arm analysis tests verify the correction.

Reviewed graph ownership, before-capture installation, operator sequence,
pre-RoPE aliasing, slot ownership, first-mismatch latch, finite byte validation,
exception/partial-install cleanup, offline dispatch, and unchanged default engine
configuration. Remaining GPU graph/noninterference validation is explicitly
pending and is the purpose of the single future controlled diagnostic, not a
CPU PASS claim. CPU host: Python 3.11, torch 2.6.0+cu124; the future engine still
requires the unchanged pinned vLLM 0.30.0 / torch 2.13.0+cu130 runtime.

The initial affected suite ran 315 passed / 0 failed / 4 skipped. These four
optional installed-source fixture tests are not counted as PASS. The final
suite uses a local external source-fixture location with all 15 original scalar
hashes verified before execution; no test assertions or source gates are
bypassed, and no historical evidence/cache directory is overwritten. All 40
internal source hashes and lengths also verify. Final affected-suite result: **327 passed, 0 failed, 0 skipped**. The final
targeted suite also passed all 46 tests. The external JUnit artifact retains
the full-suite counts. `git diff --check` and Python compilation pass.
