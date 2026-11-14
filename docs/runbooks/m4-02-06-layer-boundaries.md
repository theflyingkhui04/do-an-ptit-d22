# M4-02-06 layer boundary localization

This opt-in diagnostic prepares one controlled Vanilla A100 run to compare
`SEQUENTIAL_DECODE_COLD` and `FORCED_PREFILL_COLD` at generation index 549.
It does not launch a GPU run, change model/runtime pins, or enable the full sweep.
CPU/source validation does not establish that a divergent layer has been found.

## Evidence audit

The canonical input archive is
`m4-02-06-lm-head-input-20261114T025541779630Z-48e210cf.tar.gz`, SHA256
`2da8193b86fea3d4d79c83708ae846d21337ba929c0c9fffa48815b110fdaf5f`.
Its source and instrumentation SHA is
`48e210cf991d5670f598abd1ccc1ade442ed0c51`. All 18 manifest hashes passed,
including `worker.log`; there are 19 files including `manifest.json`. This
clarifies the input description's count without changing canonical evidence.

Independent BF16-byte decoding and full-logit calculations reproduced:

| Measurement | Maximum absolute difference | Mean absolute difference | Different elements |
| --- | ---: | ---: | ---: |
| Gathered final-norm row, 3584 BF16 elements | 1.0 | 0.024828363742147173 | 3123 |
| Raw logits, 152064 elements | 0.25 | 0.019997053001991626 | 125178 |

The prefix, generation index, request ownership, cold resets, zero prefix-cache
hits, stage/shape/dtype, runtime, and teardown/release records passed offline
checks. Both conditions' last logical input position is 2070. The historical
sample/raw/masked argmax pair remains 4164 versus 3752. Existing scientific status:
`H_TRAJECTORY_SUPPORTED`, `UPSTREAM_HIDDEN_STATE_DIVERGENCE`,
`ROOT_CAUSE_NOT_PROVEN`, `FULL_SWEEP_BLOCKED`.

## Execution map and state semantics

The immutable model revision is
`Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e9394b818fdfca67ab2a00ecc7dd641`.
Its configuration has 28 layers, hidden width 3584, 28 query heads, 4 KV heads,
head dimension 128, and MLP width 18944. Runtime code reads the actual model
configuration and enumerates its layers; it does not hard-code this layer count.

| Stage | Sequential source | Forced source | Observation |
| --- | --- | --- | --- |
| Input IDs / absolute positions | `[1]`, position 2070 | `[24]`, useful positions 2048..2070 plus padding | Prepared outside the graph |
| Embedding | `[1,3584]`, row 0 | `[24,3584]`, row 22 | Graph-produced row copy before layer 0 |
| Each decoder layer input norm, QKV, RoPE, attention, output projection, post-attention norm, MLP | Query dimension 1 | Padded query dimension 24 | Original operations; no internal-operation attribution |
| Each decoder return | Two `[1,3584]` source tensors | Two `[24,3584]` source tensors | Separate row copies of **both** components |
| Final fused add/norm | `[1,3584]` | `[24,3584]` | Original model operation |
| Gathered final norm / compute_logits entry | `[1,3584]` | `[1,3584]` | Graph-external clone before head projection |
| Raw logits | `[1,152064]` | `[1,152064]` | Existing trajectory capture |

Every model stage above embedding through final norm executes within FULL
capture/replay. Post-capture Python layer hooks cannot observe those operations
on FULL replay. PIECEWISE also replays compiled tensor regions. The diagnostic
therefore installs tensor copies before actual runtime graph capture, rather
than treating a tracing-time Python callback as a measurement.

`Qwen2DecoderLayer.forward` returns `(hidden_states, residual)`: the first is
the MLP output and the second is the post-attention residual merge. The next
input norm (or final norm) performs the next merge. The diagnostic retains the
pair at each return. Neither component is mislabeled as the whole residual
stream, and their BF16 sum is not substituted for the pair: rounding or
cancellation could hide a component difference.

For context, TP=1 embedding delegates to the selected embedding method.
QKV/gate-up/down/output projections delegate to quantization methods;
`AutoAWQMarlinLinearMethod.apply` delegates to its selected kernel. The RoPE
CUDA implementation updates Q/K in place. Attention dispatches slot-mapped KV
updates and attention through forward-context metadata; the FlashAttention
cache update invokes `reshape_and_cache_flash`. RMSNorm selects RMSNorm or
the fused add/norm path according to the presence of `residual`. None of
these implementation facts establishes a causal kernel/component explanation.

## Graph observability proof and installation order

The exact vLLM 0.30.0 sources are hash-gated by
`experiments/m4_layer_probe_sources.json`. It extends the existing 30-file
source manifest with the reviewed compilation wrapper/decorator, model
interface, linear projection and RoPE files. Runtime versions remain
vLLM 0.30.0, PyTorch 2.13.0+cu130, CUDA 13.0, A100 sm80.

1. The standard worker loads and profiles the unmodified model. Its ordinary
   activation/graph-memory calculation and KV allocation finish first.
   `profile_cudagraph_memory` clears its temporary graph entries. The adapter
   rejects missing KV storage, existing graph entries, or another graph-owning
   engine. No sizing formula or allocation policy is overridden.
2. The diagnostic worker installs the adapter at entry to
   `Worker.compile_or_warm_up_model`, after `initialize_from_config`. It saves
   and later checks KV tensor identities, addresses, shapes, dtypes and byte
   sizes. No KV contents are inspected or changed by the observer.
3. `Qwen2Model.forward` already calls `_maybe_add_hidden_state` before layer 0
   and after each decoder layer, and already supports returning a tensor plus
   a list of auxiliary tensors. The instance-local replacement appends only
   `index_select(...).clone()` rows, retaining the pair after every layer.
   This also fixes the built-in embedding auxiliary alias without retaining
   full-query activations or enabling speculative decoding.
4. A diagnostic subclass supplies a distinct forward code object which
   delegates the original Qwen2 forward tensor operations. Its only additional
   output is an integer stamp containing source row, actual absolute position,
   actual input ID and padded query dimension. The row selector is a registered
   **nonpersistent buffer**, so export treats it as model state rather than a
   serialized closure constant. It is not a weight or a sampling input.
5. The pinned `reset_compile_wrapper` rebuilds compiled execution for that
   distinct forward. The prior bytecode hook is removed first. This matters:
   merely changing an auxiliary flag after the memory profile is insufficient,
   because the ordinary compiled wrapper skips guards. The diagnostic forward's
   source-file/line cache key differs from the vanilla model's. Compilation mode,
   CUDA graph mode, capture sizes and original model tensor operations remain
   unchanged. Source tracing includes the diagnostic adapter.
6. Ordinary warmup and runtime capture run with these tensor outputs present.
   `CUDAGraphWrapper.__call__` invokes its runnable during capture, then later
   calls `entry.cudagraph.replay()` and returns the recorded output surface.
   Thus FULL replay executes the captured row-copy/stamp operations without
   requiring Python layer callbacks. PIECEWISE compiles/captures the same
   output-producing operations. The runner's existing auxiliary-output branch
   unpacks `(hidden_states, auxiliary)`; no drafter is configured or invoked.
7. The graph-external `_model_forward` wrapper updates the device row selector
   before invoking the model and checks the exact auxiliary return surface.
   The existing `compute_logits` wrapper copies the returned stage rows into
   independent storage before a later replay can reuse graph outputs. The final
   norm row is cloned before the original head call. No per-layer CPU transfer,
   device reduction to the host, or live synchronization is introduced.

This establishes the source-level route through FULL and PIECEWISE replay.
It is not empirical proof of CUDA/Inductor compatibility or numerical
noninterference. The first controlled A100 run must pass the preserved
historical-flip gate and all ownership/shape/stamp/lifecycle gates. Added
outputs can affect compiler fusion, memory lifetime and timing; a changed or
unreproduced boundary is inconclusive for historical attribution.

Primary implementation references:
[Qwen2](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen2.py),
[runner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py),
[worker](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_worker.py),
[graph wrapper](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/compilation/cuda_graph.py),
[model interface](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/interfaces.py),
[compile reset](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/compilation/wrapper.py),
[compile dispatch/cache](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/compilation/decorators.py).

## Row ownership, latching and comparison

The non-speculative runner uses `logits_indices = query_start_loc[1:] - 1`.
For one request, source row is therefore `scheduled_query_tokens - 1`.
`_prepare_inputs` constructs absolute positions from computed-token offsets.
Sequential: `2070 + 0 = 2070`. Forced: `2048 + 22 = 2070`.
Padding row 23 is excluded. The device-produced stamp must agree with the
request-owned scheduler metadata and the reconstructed logical prefix.

The existing actual-sampler first-mismatch latch freezes the entire stage
table and stamp together with raw/masked logits and sampled token. It cannot
be overwritten by later async forwards. Collection occurs after request abort
and RPC barrier. Prefix reconstruction, cold-cache ownership, source/model/
runtime identity and grammar evidence continue to use the trajectory gates.
An earlier mismatch cannot be reported as a generation-549 layer localization.

Serialization records each stage/layer/component, condition, request ID,
generation index, logical position, prefix hash, source row, retained row shape,
BF16 dtype, stride/layout, exact bytes, SHA256, expanded values and NaN/Inf counts.
The analyzer decodes BF16 bytes independently and compares exact bytes first.
Numerical counts/first indices/max/mean differences are supplemental; signed
zero differences remain byte differences. Missing, duplicate, reordered,
unsupported, nonfinite, mismatched or corrupted evidence fails closed.

For every layer, both return components must be equal to call that boundary
equal. Exact first-layer claims require complete earlier checkpoints. Sparse
evidence reports `DIVERGENCE_INTERVAL_IDENTIFIED` with the preceding equal
checkpoint and first differing checkpoint, even when the last two checkpoints
are adjacent but earlier layers were omitted. Sparse capture cannot exclude
transient differences within unmeasured gaps. The live CLI captures all layers;
sparse support in the analyzer does not silently reduce live measurement.

Classifications distinguish embedding, decoder layer, final norm, sparse
interval, no reproduced captured difference and `INCONCLUSIVE`. The historical
549 flip must reproduce before a first-boundary attribution is returned.
Localization is limited to the selected token's observed state boundaries; it
does not prove global KV-state equality, internal-layer operation, or root cause.

## Memory and teardown

One BF16 row is `3584 * 2 = 7168` bytes. With actual `L=28`, the table contains
embedding + two components per layer + final norm: 58 rows, 415744 bytes per
condition, 831488 bytes for both conditions, plus small integer stamps and
metadata. Live auxiliary outputs, the independent candidate, temporary copies
and the frozen latch add a few such tables; they do not scale with query length.
That is the observer-owned payload budget, not a bound on compiler temporaries
or peak GPU memory. Compiler scheduling can change intermediate lifetimes;
actual peak memory and compatibility remain controlled-run validation items.
The observer does not retain full sequences or KV snapshots. JSON expansions
are larger than these BF16 payload estimates and are produced after abort.

Teardown restores trajectory wrappers even when sampling/collection fails,
clears graph references, restores the original model class and auxiliary flag,
removes the selector buffer and adapter hook, resets vanilla compiled execution,
and restores compilation cache-directory/source-trace metadata. It checks KV
allocation identity again. All restoration owners are attempted on exceptions;
cleanup failure fails the run and the outer lifecycle still shuts the engine
down and checks GPU release. CPU tests exercise partial installation and
exception paths. They do not assert that GPU memory release has been observed
for this new diagnostic before its future controlled run.

## Interface and validation boundary

Use the existing prepare/run/analyze CLI with:

```text
--diagnostic-mode decode_trajectory --localization-stage layer_boundaries
```

The worker class is selected only for this explicit stage. Ordinary trajectory,
LM-head and cache modes retain their existing engine configuration. Prepare is
CPU-only; `run` remains an explicit future operator action against a reviewed
immutable source SHA. Do not launch it as part of local preparation/publication.

Targeted tests cover stage order/indexing, exact bytes, signed zero, row mapping,
request/prefix/generation ownership, both state components, async latch freezing,
missing/duplicate/wrong stages, shape/dtype/layout/provenance failures, sparse
precision, unreproduced flips, export of the dynamic row buffer and cleanup.
The exported graph test is CPU tensor/export validation; the worker fixture is
an eager CPU runner. Neither is labeled a FULL CUDA graph replay test.
Real A100 FULL/PIECEWISE compatibility and first-layer evidence remain pending.

No dependency/model/runtime pin, production repair, scheduler policy, graph mode,
attention backend, quantization, sampling, grammar or benchmark protocol changes
are part of this diagnostic. `ROOT_CAUSE_NOT_PROVEN` and `FULL_SWEEP_BLOCKED`
remain mandatory regardless of localization precision.

## Local review record

Final affected suite: **277 passed, 0 failed, 4 skipped**. The 69 new layer
tests are included. The four skips are existing optional installed-source-cache
tests in treatments/lifecycle; they are not reported as passes or GPU tests.
All 36 source hashes passed, CLI help exposes the new opt-in stage, and both
staged and unstaged `git diff --check` passed. The exact seven-path diff contains
only diagnostic code, its source manifest, tests and this runbook.

Resolved BLOCKER/MAJOR findings: post-capture Python visibility, profiling/KV
coupling, reuse of unguarded vanilla compilation, embedding aliasing, exported
row-buffer identity, sparse precision, unsupported nested graph wrappers, and
partial-installation/cleanup paths. No unresolved BLOCKER/MAJOR code finding
remains after local review. This review does not replace the pending A100
FULL/PIECEWISE validation or establish a first divergent layer.
