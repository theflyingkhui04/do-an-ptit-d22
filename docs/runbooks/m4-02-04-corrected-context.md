# M4-02-04 corrected scalar context proof

2026-11-11. Verdict: **CORRECTED CONTEXT RECONSTRUCTION VERIFIED** at
source-contract/L2 level for the guarded path below. No GPU execution, live
snapshot installer, cost calibration, or profitability result is claimed.

## Baseline and scope

Branch `feat/m4-02-04-scheduler-telemetry-v02`; starting/ending HEAD
`b4c13637149b14b3ebab20412b044639b7e4c40f`. Preserve existing worklog and
adaptive scheduler test edits, all prior measurement helpers, and artifacts.
Production serving source, deployment scripts, and frozen configs are unchanged.

Eligible mode: pinned vLLM 0.30.0, V1 uniprocess executor, TP1/PP1,
asynchronous pure decode, `draft_model`, padded token preparation, absolute
positions. Initial profiling excludes preemption/resume, stale/aborted/discarded
rows, prefill, terminal intervals, unsupported modes, and missing history.
Heterogeneous K_i, invalid speculative slots, and row permutation are covered by
the reconstruction proof; initial reference matching excludes invalid slots.

## Source authority and hashes

All paths below are relative to
`experiments/runs/m4_02_04/20261110-async-lineage-design/installed/`.
The adjacent `installed_source_manifest.json` is checked before extracted tests
execute. No current upstream behavior is substituted for this installed capture.

| Captured source | SHA-256 |
|---|---|
| vllm/v1/worker/gpu_model_runner.py | 87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51 |
| vllm/v1/spec_decode/utils.py | 318067397e6827153b484953b4f245e64b0a04308cfe05fa6760418b83a15ed8 |
| vllm/v1/spec_decode/llm_base_proposer.py | 5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab |
| vllm/v1/sample/rejection_sampler.py | 0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750 |
| vllm/v1/core/sched/scheduler.py | c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884 |
| vllm/v1/executor/uniproc_executor.py | 3cb039b80fe9b03dc1122f93c1f8a4098ef1355104b317ff6df25358f4349494 |

## Exact recurrence and lifecycle

For execution t and current worker row i, retain:

- H_t[i]: host `num_computed_tokens_cpu` actually submitted during preparation;
- P_t[i]: current-to-previous worker row mapping (`_compute_prev_positions`);
- D_t[j]: `prev_num_draft_tokens` as supplied to this preparation, indexed in
  previous row space; this gate is distinct from future proposer shared K;
- B_t: whether the actual async correction branch executes;
- V_(t-1)[j]: previous execution's GPU valid sampled-token count;
- C_(t-1)[j]: previous execution's corrected GPU input context.

Let M_t[i] = B_t AND P_t[i] >= 0 AND D_t[P_t[i]] > 0. The pinned formula is:

```text
C_t[i] = C_(t-1)[P_t[i]] + V_(t-1)[P_t[i]]  if M_t[i]
         H_t[i]                              otherwise
positions_t[i,q] = C_t[i] + q, 0 <= q < Q_t[i]
seq_lens_t[i] = C_t[i] + Q_t[i]
```

Evidence: `gpu_model_runner.py:1775-1790` maps previous rows;
`:2159-2186` selects correction versus CPU copy; `utils.py:822-854`
`update_num_computed_tokens_for_batch_change` implements masked gather, addition,
and `torch.where`; runner `:2195-2202` creates positions/sequence lengths.
The gate is evaluated using the current preparation's previous-draft array,
including K transitions. It must never be replaced by the new shared K.

Induction: a nonparticipating row copies H exactly, giving a valid seed. For a
participating row, previous exact C plus exact V reproduces the tensor function.
The previous-row mapping and request incarnation prevent cross-request reuse.
Missing predecessor facts produce `UNSUPPORTED_SHAPE`, never a forecast.

## Authoritative state and lifetime

| State | Device / producer | Validity and overwrite | Consumer |
|---|---|---|---|
| `num_computed_tokens.gpu` | GPU; CPU copy or correction helper | Corrected C after input preparation; reused/overwritten at subsequent preparation | target positions and seq_lens, next correction |
| `positions.gpu` | GPU; C plus query offset | Current prepared execution; reused next preparation | target forward |
| `seq_lens.gpu` | GPU; C plus scheduled query length | Current prepared execution; reused next preparation | attention metadata |
| `valid_sampled_token_count_gpu` | GPU; padded next-token kernel | Previous sampled output used by next preparation; runner resets/replaces retained reference during sampling | next correction and proposer |
| `prev_positions`, `prev_num_draft_tokens` | CPU metadata copied to GPU | Current preparation, previous-row index space; arrays reused | correction helper |
| host computed-token count / CPU seq_lens | CPU; worker request update and optimistic scheduling | Can run ahead of valid GPU tokens; later deferred correction and row moves mutate it | input preparation and bookkeeping |

Runner `:1372-1415` prepares previous draft counts and optimistic placeholders;
`:1491` sets host counts; `:1519` tracks current verification draft slots.
Deferred CPU correction `:1553-1583`, called at `:4344-4346` and `:4545-4546`,
does not make an arbitrary earlier/later host snapshot authoritative. CPU
`seq_lens` at `:2088-2096` can be optimistic. Standard async paths skip some CPU
accepted-count corrections (`:2117` onward).

## Existing async D2H signals

| Signal | Transfer / fact | Safe read and canonical wait | Observer rule |
|---|---|---|---|
| `sampled_token_ids_cpu` | Async output copy stream waits producer stream; nonblocking D2H, then event record (`gpu_model_runner.py:319-343`) | `AsyncGPUModelRunnerOutput.get_output`, event wait at :351, then CPU list/parse | Read only after existing canonical materialization returns |
| `ModelRunnerOutput.sampled_token_ids` | Parsed raw token lists from that D2H, row map copied at :3727-3728 | Materialized by normal output/future path before scheduler update | Freeze lengths before scheduler stop handling mutates lists |
| `valid_sampled_token_count_cpu` | Reusable pinned CPU buffer, separate copy stream/event at :4956-4976; same GPU V | `_get_valid_sampled_token_count` waits event at :4978-4987 when runtime needs it | Do not call getter or read reusable buffer; not durable per execution |
| `draft_token_ids_cpu` | Conditional async draft copy at :4905-4954 for runtime consumers | Associated canonical getter/event when consumer requires it | Not needed; adding a getter call could force an earlier wait |
| accepted-token counts | GPU accepted count updated by correction; scheduler derives accepted/rejected metrics from raw token length/drafts (:1978 onward) | CPU metrics exist during normal update; CPU accepted arrays may lag | Do not substitute accepted count, rejection metrics, or emitted count for V |

`utils.py:260-320` counts the padded output with the predicate
`(token_ids != -1) & (token_ids < vocab_size) & token_mask`; discard rows have V=0.
`RejectionSampler.parse_output` (:268-306) applies the same validity predicate
and clears discarded rows. Thus raw materialized CPU row length equals GPU V
for eligible non-discarded rows. No accepted-count-to-bonus guess is necessary.
Runner's padded draft-model path (:5190-5206) uses this count producer, including
the K0 path; the proposer performs its initial forward before K0 return.

## Input, carry, committed output

Match C entering target verification. Retain Q, all positions, and C+Q as shape.
Q is scheduled verification tokens, including allocated invalid slots; decode
Q = scheduled speculative slots + 1. Do not remove invalid slots from Q.

The GPU context buffer still represents C after this execution. The record's
`corrected_post_execution_context = C + V` is a **prospective corrected carry**
for a later participating preparation, not a claim of a GPU write now. A later
nonparticipating preparation follows its own exact host-copy branch.

Engine committed output is a separate counter: previous output count plus
actual emitted count. Scheduler `_update_request_with_output` (:2347-2363)
appends tokens, checks EOS/stop/max_tokens, and deletes the remaining tail;
`EngineCoreOutput.new_token_ids` is clipped. Example: C=40, raw V=3, emitted=1
at EOS gives prospective carry43, committed-output increment1, terminal=true.
The terminal request cannot supply a later eligible execution. Never use
clipped emitted length in the GPU recurrence.

## Immutable delayed reconstruction

`experiments/scalar_context.py` defines frozen CPU-only tuples/dataclasses with
execution ID, SchedulerOutput identity, request incarnation/fixture slot, H,
Q, P, D, current verification slots/invalid slots, K_i/shared K, branch flag,
previous count-source execution ID, padding, lifecycle, and optional prior
committed-output counter. It holds no vLLM Request objects or tensors.

Snapshots must be frozen at the actual worker preparation boundary, in worker
execution order. Scheduler N+1 can be ahead of GPU N: retain both snapshots;
after normal output N materializes, freeze its raw counts, resolve its record,
then resolve dependent records as their own canonical outputs arrive. Multiple
in-flight snapshots are independent of mutable latest request state. A missing
dependency raises immediately; there is no observer wait/poll/materialization.

The adapter binds retained preparation snapshot to submitted SchedulerOutput,
verifier ID, incarnation, raw counts, query lengths, host snapshot, and shared K.
Only a matching typed reconstruction can promote CPU_OPTIMISTIC to RECONSTRUCTED;
the original host forecast is retained separately. Final emitted/terminal facts
come from the scheduler's existing result. Timing boundary remains retirement
after the existing delegate returns; serialization stays outside capture.

There is **no live worker snapshot installer** in this change. This proof
establishes recoverability from specified snapshots, not that live transport
already captures them correctly. The next smoke must verify that transport.

## No new synchronization

The pure helper has no torch import, tensor access, device transfer, event wait,
future get/result, runtime get_output, or output-materialization call. The
observer reads only existing CPU lists and immutable metadata. Canonical runtime
already waits for sampled-output D2H; observation occurs after that wait and
before scheduler clipping. It does not invoke the separate valid-count getter.
AST checks and device-object trap tests guard this boundary. CPU work still has
cost: prior observer overhead reports are historical, their source hashes
predate this reconstruction extension, and are not current perturbation bounds.
No L3 timing-overhead result is asserted.

## Shape and exact matching contract

Minimum signature: immutable model/revision/runtime/hardware/config identity;
execution and SchedulerOutput IDs; ordered worker rows and request incarnations;
explicit cross-arm fixture-slot mapping; exact per-slot C trajectory; Q/positions/
sequence length; current valid and invalid speculative slots; shared K and K_i;
proposer rows; padded worker rows/query tokens and graph mode; lifecycle and raw
V provenance. Retain emitted count as outcome, separately from raw V.

Use exact context equality, not context bins, concurrency, running count, or an
arbitrary tolerance. `reference_match` requires the caller to verify the complete
runtime identity explicitly; default is unmatched. It checks mapped per-slot
input trajectories, composition, known equal row padding, equal query padding,
same mode, no terminal/invalid rows, and saturated per-arm decode queries.
K0 Q=1 and K2 Q=3 are the declared treatment, so their literal query lengths
cannot be equal. Every other shape difference needs review, not silent matching.
Acceptance/yield need not be equal. Diverging actual context trajectories remain
`UNMATCHED_REFERENCE`; exact reconstruction alone does not create a reference or
establish profitability. Unknown context always remains unmatched.

## L2 validation

```text
python -m pytest -q tests/test_scalar_context.py tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_modal_profiler.py
230 passed in 17.10s
```

Tests execute the full unchanged pinned tensor correction body on real torch CPU
tensors, removing decorators only. They compare 512 bounded combinations of C,
K, invalid slots, acceptance, previous-row presence, and correction branch.
Additional cases cover all/partial/first rejection, bonus/K0, heterogeneous K_i,
row reorder, K transitions, canonical parse validity, EOS/max_tokens clipping,
two in-flight snapshots, observer binding, immutable metadata, missing counts,
incarnation/source mismatch, lifecycle exclusions, matching, and sync bans.
This is L2, not installed vLLM/GPU execution. Discard V=0 is compared to pinned
parse then rejected by reconstruction; bonus absence is never guessed.

## Prepared tiny L4 smoke specification — awaiting review

Status: **NOT RUN — AWAITING REVIEW**. This is a specification, not a launcher.

- One NVIDIA L4 (sm89), worker batch1, K0 and K2 only; same canonical fixture,
  pure decode; prefill/warmup/terminal windows excluded from timing evidence.
- Keep both target `Qwen/Qwen2.5-7B-Instruct-AWQ` revision
  `b25037543e9394b818fdfca67ab2a00ecc7dd641` and draft
  `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775` loaded in both arms; vLLM0.30.0,
  draft_model, async, TP1/PP1; retain canonical graph/cache/dtype configuration
  and speculative ceiling4. Record actual effective quantization and identity.
- Before allocating, review a measurement-only adapter that snapshots the actual
  preparation branch/H/P/D/query/padding facts above. No production hook changes
  or new synchronization are authorized by this specification.
- Freeze one canonical tokenized prompt and greedy seed/config, store its token
  IDs and hash, and reuse exactly in both arms. Request enough completion tokens
  for a seed and a short nonterminal diagnostic decode segment; do not expand
  into the grid. Both arms must use the same fixture chosen at review.
- Retain raw canonical output counts before scheduler update, final emitted
  counts, immutable execution snapshots, reconstruction version/source hashes,
  per-retirement clocks and unsupported events, runtime manifest and logs.
- Pass only if every eligible execution has exact source-bound reconstruction,
  correct request/incarnation/dependency pairing, and no added GPU barrier.
  Compare actual trajectories exactly; unmatched references stay unmatched.
- Stop on missing metadata, preemption/resume, stale/discarded rows, transport
  mismatch, unsupported shape, or added synchronization. No interpolation,
  synthetic costs, profitability inference, or automatic full-grid escalation.

Full grid: **NOT RUN**. Production source: **UNCHANGED**.
Next action: review this tiny smoke specification and its snapshot-transport gate.

## Git checkpoint source lookup

The committed `experiments/scalar_vllm_source_manifest.json` retains hash-only metadata for 15 reviewed installed files. `scalar_pinned_sources.py` reads historical local captures when present, otherwise the exact installed vLLM 0.30.0 distribution, and rejects differing bytes. No upstream source is vendored by this checkpoint. This lets the same source/L2 tests run in a clean Git clone after pinned dependency installation; missing source still cannot support verification. The current six-file local gate passes 230 tests with zero skips.
