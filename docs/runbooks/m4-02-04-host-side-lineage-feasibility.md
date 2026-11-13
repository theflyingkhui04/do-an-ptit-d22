# M4-02-04 host side async lineage feasibility

For Đại and Khôi. Design investigation only; production remains unchanged.
This refines the earlier [lineage proposal](m4-02-04-async-feedback-lineage-proposal.md):
explicit worker/result fields are not necessary for the narrowly gated,
same-process UniProc/V1 scalar path if the host executor observes existing
worker state at the exact input-preparation and sampling boundaries. A sidecar
built only from scheduling messages and returning counters remains insufficient.

## PINNED FLOW TRACE

Authority: installed vLLM 0.30.0 source, SHA-256 verified against the previous
CPU source collection; project HEAD
`a6cd6ad5cee5374738f890c270ada4ca01f0946b`. Existing L3 evidence establishes
scalar serving, not lineage. No new GPU run occurred.

Illustrative S10/P10/K2 -> V11/P11/K4 flow (labels are not recovered smoke IDs):

| Boundary | Identity and immutability | Rows, reuse and freeing |
|---|---|---|
| Schedule S10 | Existing schedule_sequence plus engine epoch; detached P10 snapshot is immutable in sidecar. SchedulerOutput itself is mutable | Per-request incarnation, not just request string; retain output identity until result handling ends |
| UniProc execute S10 | Same SchedulerOutput passed inline through executor -> WorkerWrapperBase -> GPUWorker -> GPUModelRunner | Target consumes older work; new S10 drafts do not exist yet |
| UniProc sample S10 returns | Host call has enqueued the proposal and bookkeeping; `_draft_token_ids`, its shape, and producing `input_batch.req_ids` are accessible | Create a buffer-generation table stamped S10 even if Tensor object/storage is reused. No need to read token values or synchronize CUDA |
| Scheduler plans P11/K4 | May replace latest planner state before earlier futures materialize | S10 table remains independent of P11; preserve proposed topology/depth/N/K in P10 snapshot |
| Execute V11 returns, before sample V11 | Existing `execute_model_state` retains V11. `prev_positions.np`, current req_ids, old draft tensor and width still describe the input preparation just performed | Copy CPU metadata now. Current rows may be reordered/compacted; each current row selects an old row and a prefix. Freeze `consumed[V11,r] -> origin(S10,r)` |
| Sample V11 returns | Produces a new generation stamped S11/P11/K4 and returns V11's result future | Old mutable runner fields may now be overwritten; V11's frozen consumed-origin map survives independently |
| Future/result -> scheduler | EngineCore retains `(future, SchedulerOutput, exec_future)`; FIFO pop gives exact V11 pairing. ModelRunnerOutput retains req_id_to_index | Callback indexes returned row by request ID, then looks up frozen consumed origin, obtaining S10/P10, never S11/P11 |

The observation window is executable source behavior: UniProcExecutor's
`collective_rpc(non_block=True)` still invokes `run_method` inline; it wraps
the returned async output, not a background worker invocation [E1]. Runner
`execute_model` calls `_update_states`, computes previous positions, prepares
inputs, and saves ExecuteModelState before returning. Draft state is replaced
later in `sample_tokens` [E2,E3].

Scope gates are mandatory: resolved same-process UniProcExecutor-compatible
backend, one worker, V1 GPUModelRunner, scalar draft_model, async scheduling,
supported padded-draft route, serialized executor entry, known source hashes,
and no unobserved runner reset/hot replacement. Missing gates mean unsupported
lineage, not guessed attribution. The previous smoke did not record the
resolved executor class; the future implementation must verify it explicitly.

## INFORMATION PRESERVED ACROSS WORKER BOUNDARY

- SchedulerOutput identity is retained on the EngineCore side with the future;
  no result-schema field is needed to identify the verifier execution [E4].
- UniProc shares the live `driver_worker.worker.model_runner` object with its
  calling process. The host adapter can read existing CPU mappings and tensor
  shape metadata without adding worker fields or changing method signatures.
- Actual current request order is observable after `_update_states` condenses
  and may reorder the batch. `_compute_prev_positions` maps it to previous rows;
  `_prepare_input_ids` uses those exact indices and `prev_num_spec_tokens` [E2].
- Before result creation, bookkeeping copies req_ids and req_id_to_index.
  AsyncGPUModelRunnerOutput retains that ModelRunnerOutput and fills its sampled
  rows after the copy-ready event. Scheduler.update_from_output uses the copied
  request map, not scheduler dict order [E3,E5].
- CPU DraftTokenIds for structured validation already carry request IDs. A
  host adapter can bind the take_draft_token_ids call to the current known buffer
  generation and preserve that sidecar reference through the existing scheduler
  validation hook; no additional DraftTokenIds field is needed locally.

Thus: **yes**, the host in-flight record can retain the producer-origin snapshot
without changing worker tensors or model output schemas. It must freeze actual
consumption before the next sample call; merely adding the new producer's plan
to the verifier's queue entry is wrong.

## INFORMATION LOST ACROSS WORKER BOUNDARY

Ordinary returned message fields omit producer generation, old row mapping,
which input branch copied drafts, and producer decision provenance. Latest
worker fields are overwritten, and an empty callback sequence does not describe
discarded executions. EngineCore's pair identifies V11, not S10.

This information is unavailable to an isolated scheduler/remote EngineCore
using only SchedulerOutput and ModelRunnerOutput. For UniProc, it is not
irretrievably lost: the executor can freeze existing state before overwrite.
Remote/multiprocess/Ray workers do not expose that direct observation window;
they need a worker observation transport, such as Design B. Reconstructing a
remote mapping by repeating batch-compaction logic is not the recommended design.

## HOST-SIDE SIDECAR FEASIBILITY

**PROVABLY FEASIBLE** for the explicitly gated same-process UniProc/V1 scalar
profile above. This is source-level feasibility, not a tested implementation.

Proof obligations and their source basis:

1. **Producer stamp:** after a successful sample call that consumed its pending
   ExecuteModelState (not a no-state/connector-only return), stamp the actual
   current buffer generation with that call's existing execution/schedule ID
   and copy its row-to-request table. Runner sample clears/replaces draft state
   within that call. Treat zero-width, absent and synthetic buffers distinctly.
2. **Actual consumption:** before execute V, retain the active-generation
   descriptor. Immediately after inline execute V returns with ExecuteModelState,
   copy current request IDs, `prev_positions`, previous width, scheduled prefix
   lengths and branch guards. `_prepare_input_ids` selects exactly
   `old_row * previous_width + [0,prefix_length)` from that generation. Those
   relevant CPU fields have no intervening writer before sample V.
3. **Identity checks:** the selected old-row request/incarnation must equal the
   current contribution's incarnation; prefix must fit the recorded generation;
   the buffer witness must still match. No previous row (`-1`) or a CPU-only
   placeholder branch must never be attributed to an arbitrary old generation.
   Explicit real CPU draft lists use their captured DraftTokenIds provenance;
   padding/no actual draft has a distinct non-attributable outcome.
4. **Freeze:** create the consumed-origin map before sample V, which overwrites
   runner state. This handles arbitrarily delayed processing within the existing
   bounded batch queue; result arrival never chooses a producer.
5. **Return:** pair context to the retained SchedulerOutput/execution, then join
   result counts by ModelRunnerOutput.req_id_to_index and full request key.
   The per-request hook gets precisely that immutable source entry.
6. **Retire by execution:** release context on update_from_output completion,
   exception, cancellation or engine shutdown, regardless of callback count.
   Keep active buffer descriptors separately until overwritten/reset. This is
   an execution ledger, not a FIFO popped by feedback.

These rules do not assume scheduler order equals worker order. They observe
the actual CPU indices used by the unchanged worker copy operation. All
ATTRIBUTED events satisfy `origin = origin(actual consumed draft slots)`.
No-buffer/padding records preserve raw vLLM counters but cannot masquerade as
real generated work. Unknown or mixed-origin execution is fail-closed.

Seven CPU inspection cases executed the two **unmodified extracted installed**
input-preparation methods with synthetic surrounding state: reorder, compaction,
P11/K4, P11/K0 verifying old K2, prefix trimming, no speculative slots, and
removed/reused IDs. Results are in
`experiments/runs/m4_02_04/20261110-host-side-lineage-investigation/row_mapping_inspection.json`.
The inspection script requires the optional installed-source cache reconstructed
with the collector and hash inventory described in the
[source map](m4-02-04-async-feedback-lineage-source-map.md); the cache is not a
project runtime or test dependency. This checks
the row-copy rule, not a full sidecar, EngineCore lifecycle or L3 lineage.

## MINIMUM SAFE KEY

For this single-worker, single-row-generation path:

`DraftRowKey = (engine_epoch, producer_schedule_sequence, RequestKey)`

`RequestKey = (internal_request_id, first_dispatched_schedule_sequence)`

The second RequestKey component is an incarnation of the admitted Request
object, retained across preemption. It reuses an existing schedule stamp;
no new draft-generation counter is required. Scope engine_epoch to one executor
instance; a runner replacement invalidates it. A multiworker extension would
also need worker identity and a new proof.

| Candidate | Safe role / failure |
|---|---|
| Request string alone | Reused after abort; insufficient |
| Request incarnation alone | Several generations in flight; insufficient |
| Physical row alone | Reorder/compaction/reuse changes owner; insufficient |
| Producer execution alone | Batch contains multiple requests; insufficient |
| Future/batch object identity | Anchors verifier only; valid process-local lookup while strongly retained, not durable source identity |
| Tensor object/storage pointer | May be reused for another generation; diagnostic witness only |
| New draft-generation number | Could label a birth if observed correctly, but is redundant with the existing producer execution sequence and still needs request identity |
| Key above plus consumed prefix/slot range | Distinguishes producer/request generation; slot range describes trimming/invalid suffixes |

Physical row is a captured binding witness, not part of durable identity. All
slots in one scalar row generation share its DraftOrigin; store the consumed
range with the verifier rather than inventing a per-token counter. KV-cache
block/slot IDs are unrelated reusable storage and are not draft identity.

Candidate-component survival across the ten required cases (a surviving component
is not necessarily a complete key):

| Component | Reorder / compaction | Finish / ID reuse | Preemption / stale output | K=0 / invalid slots | In-flight / slot reuse |
|---|---|---|---|---|---|
| request_id | Survives both as a join | Retained on old result / reused string is ambiguous | Same string cannot distinguish epochs | Does not identify generation or valid slots | Several generations share it / not a physical-slot owner |
| request incarnation | Survives both | Retained context / new incarnation separates reuse | Retain across preemption; capture preemption epoch separately | Still needs generation/outcome | Same request has several executions / protects different admissions |
| physical row | Changes in both | Removed / can be allocated again | Removal/resume invalidates old association | Same index can represent zero/invalid work | Not unique across buffers / immediately reusable |
| producer execution stamp | Stable through both | Retain after finish / new execution separates births | Preserves old source if frozen | Empty/non-generating calls must not stamp a birth | Separates generations / must also identify request |
| draft generation number | Stable if actual birth was observed | Retain until references drain / must not reset ambiguously | Same limitations as producer stamp | Requires explicit no-birth outcomes | Redundant with producer execution; no row/request join by itself |
| tensor/storage slot | Address may stay while owner moves | Allocator may reuse it | May contain old data after reset | Zero-width, padding or invalid contents are not an identity | Shared/reused storage is not a generation stamp |
| verifier future/batch | Retains exact verifier through both | Keep until terminal / no producer incarnation by itself | Carries result context, not old source | Can complete with no callback | Distinct future, but does not select producer |
| safe composite plus actual binding | Uses observed old row, not current index | Immutable key retained; exact-incarnation guard | Preserves source and disposition separately | Outcome/prefix/invalid metadata disambiguate | Independent verifier snapshots prevent reuse or FIFO drift |

## LIFETIME / FREEING RULES

| Lifecycle case | Create / keep / free / stale rule |
|---|---|
| Normal generation | CREATE immutable P10 snapshot at dispatch; register row ownership after sample S10. KEEP active row table until actual overwrite/reset or all of its rows become unreachable, independently of S10's own result |
| Reorder / compaction | KEEP source table in its original generation order; use captured actual previous-row indices. FREE mutable witness after copying consumption metadata |
| Multiple in-flight | CREATE one consumed-origin context per verifier; KEEP until its own paired result is processed. New sample calls cannot mutate old contexts |
| Request finish | FREE active controller normally. KEEP only immutable contexts still referenced by in-flight executions; later skipped output ends its context without controller recreation |
| Request ID or row-slot reuse | CREATE a new incarnation/generation; never match old string/row alone. A removed request is deleted from prev_req_id_to_index [E6] |
| Preemption / stale output | KEEP already frozen consumed origin. Retain preemption epoch and stale disposition; drop-mode output has no feedback callback, but its paired context still terminates |
| K=0 | Zero-column generation creates no positive speculative expectation. A K=0 **new** proposal can verify old K2. FREE unused old suffix on buffer replacement |
| Invalid slots | KEEP original row source through prefix validation/-1 padding; N and invalid I remain separate, D=N-I. Fully invalid/non-draft slots are explicit lifecycle outcomes |
| No callback | FREE at execution terminal handling, not callback arrival. Superseded unconsumed rows/suffixes are discarded explicitly |
| Error / cancellation | Never promote failed staged generation. Invalidate potentially partial buffers; purge its context on caught future error/shutdown. Abrupt process exit releases process memory; never resume attribution across an unknown worker state |

Do not retain GPU tensors, Future device payloads, or mutable Request objects
in long-lived origin records. Copy CPU metadata; an ephemeral non-owning buffer
witness is sufficient during the immediate observation window. Memory is bounded
by active draft-buffer metadata plus in-flight execution contexts and short-lived
request tombstones, not by the number of feedback callbacks.
Before forwarding feedback to the adapter, require the exact active RequestKey.
If it has finished or been replaced, emit its terminal/late-event record without
calling the adapter's lazy get_controller path. This prevents resurrection or
crediting an old incarnation's feedback to a new request.

## PATCH DEPTH — HOST-SIDE

Reference patch plan A, proposed only. Seven production/deployment files:

1. New `src/server/lineage_sidecar.py`: immutable DraftOrigin, execution contexts,
   incarnation keys and lifecycle ownership.
2. New `src/server/lineage_executor.py`: UniProcExecutor subclass; read-only
   observations around existing execute_model/sample_tokens/take_draft_token_ids;
   async-future error and shutdown cleanup.
3. `src/server/adaptive_scheduler.py`: detached plan snapshots, execution-scoped
   update_from_output context, existing validation hooks, feedback and cleanup.
4. `src/server/budget_planner.py`: optional typed verification provenance and labels.
5. `src/proposer/adapter.py`: explicit source-aware event boundary; preserve scalar
   count-based update semantics. No tree estimator changes.
6. `experiments/modal_lanes.py`: select the qualified host executor class and gate
   supported runtime configuration.
7. `scripts/launch_lane3_adaptive.sh`: equivalent explicit executor selection and
   capability checks for local deployment.

**No vLLM source-file, worker method-signature, SchedulerOutput, DraftTokenIds,
ModelRunnerOutput, tensor or kernel change.** Internal interfaces *read/touched*
are UniProcExecutor call boundaries, driver_worker.worker.model_runner,
ExecuteModelState, InputBatch mappings, previous positions/width, draft buffer
metadata, output request maps and existing scheduler hooks.

The shared sidecar can be keyed by the identical VllmConfig object passed to
executor then scheduler by EngineCore, with explicit engine epoch, identity
validation and deregistration at shutdown. This avoids an unscoped global
"latest engine" singleton and does not serialize anything to the worker [E4].

Upgrade fragility remains high because these are private fields; require exact
version/source/capability checks. Cost is O(batch size) CPU metadata copying at
two boundaries, plus bounded context storage. No token-tensor copies or added
CUDA synchronization are required. This is a design estimate, not a benchmark.
Test burden concentrates on observation windows and lifecycle cleanup.

## PATCH DEPTH — EXPLICIT PROPAGATION

Reference patch plan B, proposed only. Project files:

- New `src/server/lineage_contract.py` for typed origin/event semantics.
- `src/server/adaptive_scheduler.py`, `src/server/budget_planner.py`,
  `src/proposer/adapter.py` for stamping, return context and feedback.
- New `patches/vllm-0.30.0-draft-origin.patch` and
  `experiments/modal_runner.py` to apply the reviewed pinned runtime patch.
- `scripts/launch_lane3_adaptive.sh` to assert that patched runtime capability.

The runtime patch would change exactly these interfaces/files in this reference
design:

1. `vllm/v1/core/sched/output.py`: serialized per-request producer references.
2. `vllm/v1/worker/gpu_model_runner.py`: draft-row origins, actual consumption
   snapshot, synthetic/no-draft outcomes and preservation in async output.
3. `vllm/v1/outputs.py`: consumed-origin fields on ModelRunnerOutput and origin
   references on DraftTokenIds for structured validation.

InputBatch and proposer/kernel interfaces need no change: a runner-local parallel
origin table follows actual row mappings. EngineCore's existing pairing remains
usable; the project scheduler reads returned metadata in update_from_output.

This adds serializable host metadata, no GPU math. It has deeper schema/runtime
compatibility and image-patch maintenance burden, but can carry identity across
remote worker boundaries that A cannot observe. Remote/TP/PP support still needs
its own tests; adding fields alone does not prove it. Costs include O(batch size)
metadata transport and snapshots. No measured overhead claim is made.

Future test files for either design: `tests/test_async_lineage_state.py`,
`tests/test_lineage_executor_contract.py`, and a dedicated real-runtime lineage
smoke harness. None is created as production implementation in this session.

## CORRECTNESS COUNTEREXAMPLES

1. **Row-only key:** S10 rows `[r1,r2]`; V11 rows `[r2,r1]`. Current row 0
   consumes previous row 1. A row-0 lookup assigns r1's origin to r2. Extracted
   pinned input preparation instead copied r2's synthetic markers correctly.
2. **Compaction:** remove r1; r2 moves from row 1 to row 0. Reusing row 0's
   old origin is wrong even without a new request.
3. **Slot reuse:** finish r1 and admit r3 at row 0, or resubmit string `r1`.
   remove_request erases the previous mapping. A valid host observer sees -1,
   not S10/r1. Scheduler-only row caching would misattribute it.
4. **Queue pairing alone:** V11's future pairs to V11/P11/K4. Its verification
   consumed S10/P10/K2. Attaching only P11 to the host queue entry is wrong.
5. **Sample-boundary-only reader:** after sample V11, previous positions/buffer
   state may already describe the next generation. Capture consumed origin
   after execute V11 and before sample V11 instead.
6. **Callback FIFO:** abort/drop a contribution, then deliver another. Popping
   only on callbacks shifts the queue. Per-execution terminal retirement does not.
7. **Remote-worker assumption:** a scheduler cannot read a Ray/mp worker's local
   prev_positions through its return schema. Do not enable A on that backend.

## RECOMMENDED DESIGN

Choose **A with executor-side read-only observation** for the explicitly gated
UniProc/V1 scalar deployment. Do not choose a scheduler-message-only sidecar.
The earlier general recommendation for B was conservative; this same-process
observation window makes a shallower design possible without weakening the
actual-consumed-slot invariant.

Use B if deployment requires remote workers or the observation-window gates
cannot be guaranteed. Do not silently approximate A on an unsupported backend.
Neither mechanism is implemented or L3-lineage verified yet.

## REQUIRED SHARED INTERFACE

Project-owned host interface only for A:

- `DraftOrigin`: engine/request incarnation, existing producer schedule/plan IDs,
  detached topology, depth, planned N, request scalar budget and scheduled batch K.
- `PreparedConsumption`: verifier execution key, request key, actual source key,
  row/slot-range witness and input-source kind; immutable before the next sample.
- `HostExecutionRecord`: exact SchedulerOutput/future association and terminal
  lifetime. It is separate from active buffer-generation ownership.
- `VerificationEvent`: consumed origin plus raw/invalid/adjusted counts, accepted
  count, lineage status/reason and lifecycle disposition.

Tree-relevant fields remain planned provenance. They can be carried entirely in
the detached origin; feedback never rereads latest topology/depth. Exact lineage
does not repair L/d probability calibration or prove arbitrary-tree execution.
Keep `tree_execution_verified=false` and preserve current scalar semantics until
the separately reviewed stale-feedback policy is approved.

## TEST GATES

| Level | Smallest required checks |
|---|---|
| L1 | Immutable S10/P10/K2 origin survives P11/K4; independent r1/r2, incarnation and slot reuse, engine epoch, zero-K/no-row/invalid distinctions; terminal retirement without callbacks |
| L2 | Actual UniProc inline invocation and after-execute/before-sample observation window; use pinned row preparation with reorder and compaction; verify buffer-generation stamping even if Tensor identity repeats; three in-flight results and deferred grammar sampling; empty batches, finish, abort, stale-deliver/drop, re-preemption, invalid slots, errors, missing callbacks and shutdown; no live GPU references or controller resurrection |
| L3 | On pinned vLLM/weights, record resolved executor/runner gates, independently audit actual consumed rows against host snapshots, observe natural K change and two requests, check returned P10 origin after P11 exists. Exercise finish/slot reuse/K0/grammar and controlled stale/preempted output where reproducible; unsupported/unexercised cases stay unverified. Confirm one terminal context per execution |

Seven extracted-method CPU checks passed; they do not satisfy full L1/L2/L3
sidecar test gates. Official pinned input-batch swap/condense, worker lifecycle,
invalid-suffix, executor async-support, and earlier scheduler preemption tests
were inspected as source evidence, not claimed as test passes.

## TECHNICAL ALIGNMENT QUESTIONS

1. Is the initial lineage capability explicitly limited to the gated UniProc/V1
   profile, with B required for a future remote-worker deployment?
2. Do Đại and Khôi accept the shared DraftOrigin/PreparedConsumption/Event
   semantics, incarnation key and executor/scheduler ownership split?
3. Which owner maintains private-source capability guards and proves the L2/L3
   observation-window and cleanup gates before enabling attributed telemetry?

## IMPLEMENTATION STATUS

**NOT IMPLEMENTED — ALIGNMENT FIRST**

No production source, original smoke evidence or previous proposal was modified.
Only this investigation's documentation, source-reading cache and temporary CPU
inspection script/results were added. No staging, commit, push, PR, Trello
update, performance benchmark or arbitrary-tree work.

## FIRST ACTION NOW

Đại and Khôi ratify or reject the explicitly gated UniProc host-observation
boundary before approving the shared lineage interface.

[E1]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/executor/uniproc_executor.py#L90
[E2]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py#L1775
[E3]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py#L4529
[E4]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py#L630
[E5]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py#L1972
[E6]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_input_batch.py#L526
