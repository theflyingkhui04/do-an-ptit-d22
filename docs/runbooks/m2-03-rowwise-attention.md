# M2-03: bounded T4 row-wise segmented attention

Implementation profile: `m203.t4-rowwise-segmented3d.v1` (2026-10-19).
GPU targeted verification **NOT RUN**. Canonical 54 **NOT RUN**.

## Review boundary

`src/server/t4_rowwise_attention.py` produces an exact production transform of
two pinned vLLM 0.30.0 files. The generated upstream diff is
`patches/vllm-0.30.0-t4-rowwise-attention.patch`. Historical source snapshots,
observer transforms, diagnostic profiles, and raw evidence are unchanged.

The causal result supplied for execution `8c04f232b493c51b352daf841d7390ae62c0f755`
supports attention-path causality at layer0 P177. It does not prove that this
implementation restores full token equality or explains every canonical mismatch.

## Runtime behavior

The metadata builder enables the policy only for a speculative **target** whose
model-config object is `spec.target_model_config`, with FP16 model dtype and the
existing PIECEWISE graph regime. The bound is `1 + num_speculative_tokens` (five
query rows for the frozen configuration). This limits eligibility, and does not
change the configured or scheduled k.

Host dispatch additionally requires sm75/CUDA, one sequence, 2..bound query rows,
the frozen 28-query-head / 4-KV-head / head-size-128 shape, FP16 Q/K/V/output,
ordinary causal attention, and matching persistent FP32 contiguous scratch.
Window, prefix, sink, bias, softcap, quantized-cache, tensor-descriptor, batch
invariance, and unsupported graph configurations retain their existing paths.
Scratch capacity must cover every query token in each of the three buffers.

Grid axis 0 indexes the unchanged flattened query tensor. For token `t` in
sequence `s`, the shared JIT helper computes:

```
r = t - query_start_loc[s]
Qlen = query_start_loc[s+1] - query_start_loc[s]
Lr = seq_lens[s] - Qlen + r + 1
```

Each producer program presents one logical query to the existing decode body:
local query block 0, query length 1, flattened start `t`, and sequence length
`Lr`. The original BLOCK_M=16 and GQA=7 expressions mask lanes 7..15. Only the
seven heads for this query and KV group are valid. Output and scalar scratch
continue to use flattened original token/head/segment indices.

Both producer and reducer call the same helper and use decode tile size 16 and
16 segments. Thus `tiles_per_segment = ceil(Lr / 256)` and active segments are
`ceil(Lr / (tiles_per_segment * 16))`. No earlier row uses the final verify
length for reduction. The causal mask sees query absolute position `Lr-1`.

The existing KV update, compile dependency, block table, sampling, scheduler,
and target-model forward are untouched. All query rows remain in one target
forward and one attention producer launch plus its reduction per layer.

## Source authentication and installation interface

The transformer accepts exact pristine hashes only. `validate_transform` checks
the complete expected output bytes, rather than broadly stripping arbitrary AST
changes. The unmodified imported attention helper is also pinned and backed up.
Its fixture was retrieved from official `vllm-project/vllm`, tag `v0.30.0`, path
`vllm/v1/attention/ops/triton_attention_helpers.py`, Git blob
`dff0b21c64232a72379f46e87c5d5a6991517d84`, SHA-256
`35aa993d01345f535d2641cf9dcabb8b64c51b20a19707bc5987e616aaa87ddd`.
It retains its upstream Apache-2.0 header and the existing fixture license applies.

Review only, against a pristine package directory:

```bash
python -m src.server.t4_rowwise_attention diff --source-root /path/to/vllm
```

The same module has explicit `install`, `verify`, and `restore` commands, each
requiring `--source-root` and `--evidence-dir`. Installation authenticates all
three sources before any installed-source write, requires a fresh evidence
directory, preserves original bytes and hashes, and records exact patched hashes.
Restore accepts only authenticated original/patched bytes, so a partial install
can be recovered but an unknown modification cannot be overwritten.

No installation or server restart was performed during local implementation.
Before any future installation, stop the affected server through its authorized
workflow. Never install into a running server. Review the actual diff first.

Historical diagnostic authentication deliberately rejects this production
transform. Do not bypass that check or reuse an old diagnostic manifest to
describe a patched run. Production verification uses this explicit profile;
any future observer integration must declare its exact composed transform.

## Validation and next gate

The focused tests execute the transformed host dispatch with launch recorders,
the actual per-row integer helper, and pinned causal/index expressions on CPU.
They compare fallback launch arguments to pristine source and row indexing and
segment geometry to native q_len=1 semantics. They do not emulate Triton dots,
softmax arithmetic, CUDA capture, or GPU output equality.

After diff review and explicit GPU authorization, the first experiment remains
`w3_business_v2_004` under the frozen controlled workflow and conditioning.
PASS requires exact Vanilla/Adaptive generated token equality **and** evidence
of real speculative verification: k>0, drafts scheduled, M-row target verify,
no hidden k=0 fallback, no extra target forwards, and no observer/runtime errors.
Preserve all provenance and failures in a fresh run directory. Report baseline
repeatability, adaptive repeatability, and paired comparisons separately.
Only targeted PASS permits considering canonical 54, with separate authorization.

## Local validation record — 2026-10-19

PASS: 70 focused tests. PASS: 391 existing tests and 169 subtests. The latter
exercise historical Protocol-B/observer profiles, which are unchanged by the
production transform. No GPU numerical or capture result is implied.

Exact test commands:

```text
python -m pytest tests/test_t4_rowwise_attention.py -q
python -m pytest tests/test_protocol_b.py tests/test_protocol_b_boundary.py tests/test_protocol_b_boundary_v2.py tests/test_protocol_b_boundary_kv.py tests/test_protocol_b_boundary_early.py tests/test_protocol_b_boundary_postwrite.py tests/test_protocol_b_boundary_layer1.py tests/test_protocol_b_boundary_gpu_resolution.py tests/test_protocol_b_boundary_tokenizer_revision.py tests/test_protocol_b_boundary_p177_k0_control.py tests/test_protocol_b_boundary_p177_attention_substitution.py -q
python -m py_compile src/server/t4_rowwise_attention.py tests/test_t4_rowwise_attention.py tests/fixtures/m203_v030_triton_attention_helpers.py
git -c safe.directory=D:/Document/Four-year/do_an/do-an-ptit-d22 diff --check
git -c safe.directory=D:/Document/Four-year/do_an/do-an-ptit-d22 diff --exit-code HEAD -- experiments/runs experiments/results experiments/configs logs experiments/protocol_b_boundary_sources.py tests/fixtures/protocol_b_boundary_kv_v030.json.gz
```

All commands above exited zero. Transformed upstream sources were also compiled
in memory. New Python/document files were separately checked for trailing
whitespace, since ordinary `git diff --check` does not inspect untracked files.
The helper fixture's Git blob hash matches the official tag's blob exactly.
