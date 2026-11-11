# M4-02-04 scalar cost observer fidelity audit

Date: 2026-11-11. Scope: Option B, measurement only. Starting/ending HEAD:
`b4c13637149b14b3ebab20412b044639b7e4c40f`, branch
`feat/m4-02-04-scheduler-telemetry-v02`.
The previous integration/worklog edits and three measurement-contract files
were already dirty/untracked at entry and are preserved. No production file,
frozen protocol, launcher or controller policy was changed. No GPU was allocated.

## Result

**NOT VERIFIED — SHAPE GAP**. Pairing, retirement boundary, row mapping and
useful-token accounting have L2 evidence. Actual corrected async context lengths
are not available from the proposed cheap CPU snapshot. Do not run a small GPU
smoke or grid through this ambiguity. **TECHNICAL REVIEW REQUIRED**.

The earlier contract's use of CPU worker context buffers needs qualification:
they can be optimistic, not measured execution context. The retirement-window
walltime definition remains unchanged. Existing real scalar L3 evidence remains
valid for its original functional claim; this audit supplies no installed-engine
or GPU evidence.

## Source authority and pairing proof

Tests read existing `20261110-async-lineage-design/installed/` captures and check
each file against `installed_source_manifest.json` before AST extraction. No
upstream code was vendored. Missing captures explicitly skip the affected tests;
such a run cannot support this audit's verified L2 claims. The current local run
has zero skips. Source links below are the exact official v0.30.0 tag; captured
installed bytes remain the executed-test authority.

| Source | Pinned path and lines | What is established |
| --- | --- | --- |
| E1 | [EngineCore](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py#L589), 589-619 | `step`: schedule S, execute future, materialize O (or sample after None), process aborts, update(S,O). |
| E2 | Same file, 630-744 | Queue stores `(future,S,exec_future)` at686, pops the oldest triple at701, resolves that future and calls update at718. Completion readiness does not change the tuple. |
| E3 | Same file, 676-742 | Structured output retains `deferred_scheduler_output`; older queued work retires first, then deferred grammar/draft validation and sampling queue the retained S with its future. |
| E4 | [UniProcExecutor](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/executor/uniproc_executor.py#L32), 32-49,90-150 | `AsyncOutputFuture` owns one async output and caches materialization. UniProc runs the worker method directly; this does not establish support for MP/Ray/TP/PP transports. |
| E5 | [Scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py#L1870), 1870-2210 | Update consumes paired result using `req_id_to_index`; abort/missing/finished and stale-drop rows are skipped. EngineCoreOutput carries final per-request `new_token_ids`. |
| E6 | Same file, 2347-2363; [stop checks](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/utils.py#L94) | Append/check stop before output creation; trim token tail beyond EOS/stop token/max_tokens/model length. |
| E7 | [GPUModelRunner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py#L1775), 1775-1790,3727-3728,4779-4847 | Explicit previous-row mapping and copied output IDs/index map survive row movement and subsequent input-batch mutation. |
| E8 | Same file, 2088-2202; [correction](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/utils.py#L822), 822-854 | CPU lengths assume all prior drafts accepted. GPU context correction uses prior GPU context plus actual valid sampled count for participating rows. |
| E9 | Same runner, 5019-5296; [proposer](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/llm_base_proposer.py#L515), 515-630 | Scheduler-selected K is passed to propose; it becomes `self.num_speculative_tokens`. K0 still performs initial forward before returning empty drafts. |

Pairing proof concerns executions that reach update. Future failure does not
produce a successful retirement. Empty execution can return a no-op result;
it must not increment decode-window N. Structured deferral is not retirement.
Abort/stale state can make an otherwise correctly paired result ineligible.
No-op, deferred and dropped results must not be reassigned to another S.

## Retirement and observer placement

Retirement = successful return from `scheduler.update_from_output(S_i,O_i)`.
Primary timestamp is taken immediately on return, before observer record
processing. Only supported nonempty pure-decode observations may anchor/count
a scalar window. Zero output, prefill, abort/stale, terminal state, missing
shape and identity mismatch break the segment; they are never skipped across
to form a seemingly continuous window. Failed future/update breaks the segment.

| Option | Pairing and outputs | Shapes | Depth/perturbation | Decision |
| --- | --- | --- | --- | --- |
| EngineCore wrapper | Sees queue path and outer failures | Does not itself expose actual GPU contexts | Broad private engine loop; avoid rewriting it | Outer exception guard only. |
| Scheduler update wrapper | Receives the exact paired arguments and final output | Needs execution-bound worker snapshots | Shallow single method wrapper; CPU metadata overhead | Preferred retirement boundary. |
| Executor/future wrapper | Can tag submission; completion is before scheduler consumption | Worker access possible in UniProc | Future wrapping risks materialization/ordering changes | Submission association only; do not call result early. |

Recommended design: shallow update wrapper plus submission-bound worker
observations and an outer failure guard. `experiments/scalar_cost_capture.py`
implements only the CPU in-memory adapter, not a live installer. It delegates
once, returns the original scheduler result, propagates original execution
exceptions and prevents observation-format errors from replacing engine output.
It never calls future.result(), CUDA, tensor readback, JSON serialization,
file I/O or locks in capture. Bounded-buffer overflow invalidates the observation
and is exported explicitly. A real installation remains blocked.

Capabilities currently checked for the adapter: vLLM0.30.0, UniProc, V1,
TP=PP=1, AdaptiveAsyncScheduler base, explicit async flag. Runtime information
is caller-supplied in L2; a future live installer must inspect actual objects and
installed source identities before enabling collection. It must reject
MP/Ray/PP/V2, unavailable fields and unsupported callback signatures as
`UNSUPPORTED_MEASUREMENT_MODE`. Constructor success is not live capability proof.

## Field observability and the shape blocker

| Field | Exact source | Classification | Qualification |
| --- | --- | --- | --- |
| Actual target row IDs/count | `runner.input_batch.req_ids/num_reqs` after preparation | DIRECTLY OBSERVABLE | Copy CPU values at the paired execution, not at later retirement. |
| Output row identity | `ModelRunnerOutput.req_ids/req_id_to_index` | DIRECTLY OBSERVABLE | Validate mapping; never zip scheduler dictionary and worker rows. |
| Query lengths | Prepared CPU query offsets/counts in `_prepare_inputs` | DIRECTLY OBSERVABLE | Record padding separately and bind to worker row IDs. |
| Proposer rows | `common_attn_metadata.batch_size()`/propose boundary | DIRECTLY OBSERVABLE | CPU shape metadata; not client/running count. |
| Planned batch K/K_i | Project planner snapshots | DIRECTLY OBSERVABLE | A plan, never proof of execution. |
| Scheduled K | `SchedulerOutput.num_spec_tokens_to_schedule` | DIRECTLY OBSERVABLE | Future proposal budget of this output. |
| Proposer K | `propose(num_speculative_tokens=...)` and assigned field | DIRECTLY OBSERVABLE | Requires actual call observation; planned equality cannot stand in for it. |
| Future placeholder counts | Project post-trim `request.spec_token_ids` | DIRECTLY OBSERVABLE | Snapshot before subsequent scheduling mutates it. |
| Current valid scheduled draft slots | S.scheduled_spec_decode_tokens minus S.num_invalid_spec_tokens | DERIVABLE WITHOUT AMBIGUITY | Belongs to current verification, not future placeholders. GPU invalid/sample behavior must also be respected. |
| CPU context lengths | `input_batch.num_computed_tokens_cpu`, `optimistic_seq_lens_cpu` | FORECAST ONLY in async speculative decode | Kernel corrects GPU values; CPU acceptance copy can race with in-flight transfer/row moves. |
| Actual corrected context/positions | `runner.num_computed_tokens`, `positions`, `seq_lens` GPU tensors | NOT AVAILABLE through current safe CPU observer | Reading with `.item()/.cpu()` adds synchronization/copies. No validated delayed snapshot or offline reconstruction exists here. |
| Exact pure-decode context classification | Actual context versus prompt boundary | NOT AVAILABLE in general from this observer | Explicit prefill metadata can exclude known prefills; optimistic contexts cannot prove remaining rows eligible. |

Minimal counterexample from E8: prior GPU context40, valid sampled count1 and
optimistic CPU context43 => corrected GPU context41 for a participating row.
Reporting43 as measured would be wrong even if request IDs, query lengths and K
all match. Tests verify the actual captured correction expression and reject
`CPU_OPTIMISTIC` snapshots. No GPU kernel was run to manufacture this evidence.

An offline reconstruction from complete raw accepted/output history and the
pinned row/correction recurrence may be possible, but requires a separate proof
covering batch changes, invalid drafts, preemption, stale delivery and cached
CPU acceptance. This audit does not implement or claim that reconstruction.

## Request identities and speculative budgets

Use `(run,request_id,incarnation)` for logical state; worker slot is only an
execution-local index. The submission registry keeps a strong S reference until
retirement, preventing accidental object-ID reuse. Paired event records contain
process-local S/O object IDs and a unique verifier ordinal; these are diagnostic
association IDs, not source-plan/source-proposal lineage.

Captured `_compute_prev_positions` tests cover reorder, compaction/removal and
new slot occupants. Request reincarnation changes invalidate an active window.
The evidence validator previously required stable row order and zipped rows
with budgets: fixed to retain first-row request-ID/budget association and accept
later row permutations. Heterogeneous reorder is regression-tested.

Record planned K, scheduled K, actual proposer K, planned K_i, post-trim future
placeholders and current verification slots separately. A captured test uses
planned K4, scheduled/proposer K2, future placeholders2, current verification
slots4. No scalar value is promoted into an arbitrary-tree execution claim.

## Useful output and context-reference semantics

Count `EngineCoreOutput.new_token_ids` after scheduler stop/length handling,
summed over the same retirement window. This includes the bonus and any EOS/
stop token retained by canonical engine token accounting; excludes discarded
tail tokens. It is not the number of accepted drafts or raw sampled row length.
Frontend detokenization/stop-string suppression and network delivery are later
stages; do not equate token-accounting yield with visible text chunks.
Terminal outputs remain diagnostic records and break steady windows.

Trajectory provenance records each request incarnation, initial and per-retirement
actual contexts, shape/composition, final output counts, stop/stale state, query
length and current/future budgets. `trajectory_signature` records exact provenance,
not a MATCHED_REFERENCE predicate. Across independently reset K cohorts, use
explicit token-fixture/logical-slot mapping; raw request IDs need not be equal.
Same initial shape and bin alone are insufficient. Exact mapped actual context
trajectories and supported composition must match, or a reviewed bounded cost
approximation is required. Output yield is an outcome and need not be equal;
its effect on subsequent context must be represented. Unknown corrected contexts
or divergent trajectories => `UNMATCHED_REFERENCE`, no profitability comparison.

## Perturbation and retained failures

CPU-only overhead evidence: `experiments/runs/m4_02_04/20261111-scalar-observer-l2/cpu_overhead_v3.json`.
Seven alternating disabled/enabled repeats, 1000 synthetic calls each, B1/B2.
Includes submission/snapshot/update capture, excludes export/serialization/I/O.
Median added CPU cost: **16.672 us/call at B1**, **19.845 us/call at B2**.
All raw repeat durations, local environment, observer source SHA-256 and dirty
execution identity are retained. This is NOT a GPU perturbation bound or an
engine-performance result. No material-overhead threshold was invented.
Earlier `cpu_overhead.json` and `cpu_overhead_v2.json` attempts are retained;
each identifies its own observer revision. The final source/harness hashes match
v3. Reproduce without overwriting evidence:

```text
python -m experiments.scalar_cost_observer_overhead --output <new-output.json>
```

AST/device-trap tests forbid CUDA synchronization, `.item()`, `.cpu()`, tensor
conversion, future materialization, serialization and file writes in capture.
The runtime's own output materialization synchronization remains untouched.

Failure categories: failed execution/OOM/timeout (outer guard), failed update,
empty schedule, absent/zero useful output, lifecycle-unsafe/terminal, missing
submission pair, output/worker identity mismatch, unavailable/inconsistent shape,
prefill/unknown mode, missing incarnation, state change, observation error and
buffer overflow. Retain them; break the window; never manufacture a successful
duration. A parent watchdog still owns external process timeout evidence.

## Test command and scope

```text
python -m pytest -q tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py
```

Final combined regression command:

```text
python -m pytest -q tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_modal_profiler.py
201 passed, zero skipped
```

The final total includes 40 new captured-method/observer cases, 54 evidence
schema cases and 107 existing profiler cases. `git diff --check` and a separate
whitespace check of new files pass. No runtime source or frozen config diff.

Tests execute captured EngineCore step/queue paths, AsyncOutputFuture,
previous-row mapping, scheduler output trimming and stop logic. GPU/executor
work is controlled CPU scaffolding. Snapshot contexts labelled MEASURED in
positive fixtures are synthetic; only the negative test represents the current
CPU_OPTIMISTIC limitation. Positive fixture success cannot establish a real
worker shape transport or authorize profiling.

Small GPU smoke: **NOT RUN** (shape/reference stop conditions apply).
Full grid: **NOT RUN — SMOKE REVIEW REQUIRED**.
Production source: **UNCHANGED**.

Next action: prove a synchronization-free reconstruction or delayed observation
of corrected per-execution context lengths against the pinned GPU recurrence.

## Corrected-context follow-up (2026-11-11)

The source/L2 reconstruction proof is now in
[m4-02-04-corrected-context.md](m4-02-04-corrected-context.md).
It resolves the mathematical shape gap for the explicitly guarded scalar path;
the earlier audit above remains historical. The new helper and bound observer
integration pass 230 combined tests. Live worker snapshot transport remains
unverified and must be checked by the separately reviewed tiny smoke.
Previous CPU overhead artifact hashes predate this extension. No GPU was run.
