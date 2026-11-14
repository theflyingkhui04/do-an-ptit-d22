# M4-02-04 async feedback lineage proposal

For Đại and Khôi. Status: **TEAM TECHNICAL ALIGNMENT REQUIRED**.
**NOT IMPLEMENTED — ALIGNMENT FIRST.**

Recommend propagating immutable producer metadata with the actual draft buffer
rows, then snapshotting the consumed metadata into the verification result.
Reuse the existing project schedule/plan sequences; do not introduce another
step counter. A queue popped on feedback arrival cannot establish this identity.
This proposal is source-backed; its implementation and L1/L2/L3 tests do not
yet exist and are not verified.

## Baseline and evidence checkpoint

HEAD remains `a6cd6ad5cee5374738f890c270ada4ca01f0946b` on
`feat/m4-02-04-scheduler-telemetry-v02`. Current production files have no diff.
The existing uncommitted L3 smoke establishes scalar runtime compatibility:
vLLM 0.30.0, Ubuntu 22.04.5, A10, Python 3.11.5, torch 2.13.0+cu130;
36 plan/schedule matches, K={0,2,3,4}, 83 drafted/72 accepted, three completions.
All feedback source IDs remain null. These counters establish neither lineage
nor tree execution. `dynamic scalar K != arbitrary-tree runtime execution`.

The original smoke directory and failed attempts were preserved byte-for-byte.
Its 30 files total 486,637 bytes; all 29 historical manifest entries match.
One file is ignored bytecode. Common credential patterns were absent; launch
config and request review found no authentication values. Windows/worker paths,
workspace/app/call IDs and volume paths are non-secret provenance identifiers.
Retain originals; redact a separate public derivative if reviewers require it.

Checkpoint policy (2026-11-10): preserve original smoke artifacts locally,
including full class snapshots and the historical manifest. Commit only the
curated paths in `experiments/runs/m4_02_04/20261110-scalar-vllm030-smoke/checkpoint_manifest.json`.
Full copied installed/upstream source, bytecode and superseded staging lists are
excluded. The 28 KB server log is retained under the repository's run-evidence
convention because it records startup recovery and teardown warnings.
The checkpoint manifest records excluded originals' hashes and pinned source URLs.
The [host-side feasibility refinement](m4-02-04-host-side-lineage-feasibility.md)
is the current narrow UniProc/V1 recommendation; this earlier proposal retains
the general explicit-propagation alternative and its assumptions.

For this analysis, a read-only CPU function collected installed V1 source from
the same pinned project image; no GPU/model launch occurred. Scheduler class
text matches L3 exactly. AsyncScheduler executable AST matches; its sole text
difference is a comment em dash corrupted to U+FFFD in the old snapshot.
No historical artifact was rewritten. Six official v0.30.0 test files were
read, not run. Full source caches are transient and are not checkpoint candidates.
See [source and flow appendix](m4-02-04-async-feedback-lineage-source-map.md).

## Current async flow

Illustrative P10/P11 timeline for one request, allowing later scheduling before
earlier output processing. These are explanatory labels, not attributed IDs
recovered from the L3 smoke:

1. `AdaptiveBudgetPlanner.plan_step` selects P10, k=2 and saves a detached
   decision. AdaptiveAsyncScheduler updates `dynamic_sd_lookup` and advances
   its existing `_schedule_sequence`.
2. `Scheduler.schedule` advances `current_step`, selects the request's
   **existing** speculative placeholders for verification, then chooses the new
   output's `num_spec_tokens_to_schedule` from the dynamic lookup. The output
   contains both old verification slots and the new producer K [S1].
3. `AsyncScheduler._update_after_schedule` reserves output placeholders and
   replaces `request.spec_token_ids` with placeholders for drafts this output
   will generate. The project shrinks those future placeholders per request.
   Current scheduling metadata describes a future verification, not current
   feedback origin [S2].
4. EngineCore pairs `(future, SchedulerOutput, execute_future)` in its batch
   deque. It appends left and pops right; each model result is passed to
   `update_from_output` with that exact output. Deferred grammar sampling
   retains a separate output until the preceding result is processed [S3].
5. The V1 runner maps current request rows to previous rows using
   `prev_req_id_to_index`/`prev_positions`. `_prepare_input_ids` copies the
   scheduled prefix from the previous `_draft_token_ids` tensor. Rejection
   sampling verifies those drafts [S4].
6. After target sampling, the runner clears old draft state and invokes the
   proposer with this output's K=2. `DraftModelProposer` inherits the base
   proposer; the base produces a single batch tensor with a row per request.
   Even K=0 can execute the first forward for draft KV synchronization, then
   returns a tensor with zero draft columns [S5].
7. A later scheduling decision P11/K=4 may already exist. Its target pass
   consumes the earlier draft row; its own proposer may produce K=4 drafts.
   These are distinct producer and verifier roles.
8. `AsyncGPUModelRunnerOutput` retains a per-execution ModelRunnerOutput and
   copies sampled IDs asynchronously. Its future materializes the result;
   EngineCore processes aborts, then calls `Scheduler.update_from_output` [S3,S4].
9. That method handles stale/finished/failed-load cases and, for each eligible
   request, calls `make_spec_decoding_stats` once with counts from this
   verification output. The project subtracts grammar-invalid slots and calls
   `record_verification`, which immediately updates the request controller.
   EngineCoreOutput/EngineCoreOutputs are constructed downstream; they are not
   the carrier that brought per-request verification feedback to the scheduler.

The current `_plan` and `request.spec_token_ids` are overwritten during forward
scheduling. Neither is a reliable source snapshot at step 9.

## Existing upstream identifiers

| Identifier | Producer and consumer | Lifetime | Can solve lineage? |
|---|---|---|---|
| Internal `request_id` | InputProcessor -> scheduler/worker/output | Request lifetime | Request join only; normally randomized suffix, but reuse is explicitly supported by worker cleanup |
| `Scheduler.current_step` | schedule -> V2 eligibility logic | Scheduler instance; every call, including empty | Good schedule stamp if captured; not transported to drafts/results |
| `sched_step_seq` / `processed_step_seq` | scheduler -> deferred block-free fences | Conditional: `defer_block_free`, nonempty outputs | Not universal; counters alone do not identify producer |
| `Request.last_sched_seq` | after-schedule -> block-free check | Latest mutable request fence | Overwritten while older output is in flight |
| `next_decode_eligible_step` | AsyncScheduler -> V2 scheduler | Mutable PP cadence | Eligibility, not draft identity; not the smoke V1 route |
| EngineCore future/output pair | EngineCore -> update_from_output | One in-flight execution | Exact **verification** pairing, not draft producer |
| `prev_req_id_to_index`, row/prev position | runner bookkeeping -> input preparation | Previous worker batch; reordered/overwritten | Locates consumed row; essential binding point, not a durable identifier |
| `num_preemptions`, in-flight/stale token counts | Request/scheduler -> stale handling | Request life | Generation/disposition information; no producer step |
| DP `step_counter` / wave, timestamps, `session_id` | DP/session logic -> coordination/client | Resets/nullable/coarse | Not per-draft identity |
| Project plan/schedule sequences | planner/custom scheduler -> existing telemetry | Planner/scheduler instance | Reuse these existing IDs, but propagate immutable snapshots |

Searching all 392 installed V1 Python files found no existing end-to-end draft
producer identity in SchedulerOutput, DraftTokenIds or ModelRunnerOutput. The
official tests confirm the counters' narrower purposes [T1,T2,T3,T4]. No new
project sequence counter is justified.

## Lineage invariant and cardinality

For one feedback event F(r,V,D,A), every counted valid slot must belong to the
same immutable producer row G(r,S,P). Attribution means:

`origin(F) = origin(actual draft slots consumed by verifier V) = (r,S,P)`.

`0 <= A <= D = verified_slots - grammar_invalid_slots` for this scalar
autoregressive path. K selected for **V's next proposal** must never substitute
for S's K or P. Request identity includes an engine epoch and request incarnation.

**At the project callback: one F -> one request-row producer generation**, when
attributed. Proof: `update_from_output` iterates one request entry per paired
SchedulerOutput and makes one callback using its scheduled-spec list and one
sampled-token row. V1 input preparation selects a prefix of one previous draft
tensor row; it does not concatenate producer generations [S1,S4]. The runner
replaces the draft tensor after sampling. Trimming/grammar filtering remove
slots from that generation, not add another generation.

The relation is partial: producer row -> zero or one feedback event (zero when
unconsumed, zero-budget, aborted, discarded or otherwise skipped). One batch
producer schedule -> N request-row events; upstream SpecDecodingStats folds N
request callbacks into one step aggregate, and request summaries fold multiple
verification steps. Those aggregates are **N:1** and cannot have one scalar
source ID. Attribute before aggregation or retain a list of source contributions.
If another runner ever combines origins, emit UNATTRIBUTED rather than force
this scalar 1:1 contract; supporting contribution lists requires separate review.

## Failure scenarios

| Case | Required handling |
|---|---|
| A: P10/K2, P11/K4, delayed P10 feedback | Carry P10/S10 with produced row; snapshot it at consumption by V11; callback reads V11's immutable consumed origin, never latest P11 |
| B: K=0 between speculative steps | Record scheduling/producer outcome with zero columns; do not create a pending speculative feedback expectation. A K=0 producer can still verify earlier nonzero drafts; associate that feedback with the earlier producer |
| C: invalid slots | Preserve original producer and verified slot range; record N and invalid I separately; D=N-I, A from sampled row minus bonus token. Fully invalid D=0 is a lifecycle outcome, not a positive controller sample |
| D: r1/r2 and row compaction | Join by full request key and actual worker row mapping, never batch row position alone; snapshot before reordering/overwrite |
| E: completion/abort | Process final legitimate feedback before active controller cleanup. Retain immutable execution contexts until their futures drain or are cancelled; later finished-request outputs are terminal drop records, not lazy controller recreation |
| F: preemption/delay | Save producer request epoch. Ordinary stale output may be delivered and reach feedback; drop-mode stale output bypasses callbacks. Preserve exact origin and mark disposition STALE_DELIVERED or STALE_DROPPED; never relabel as resumed work |
| G: error/recovery | Stage metadata until a real SchedulerOutput is dispatched; failure is terminal ERROR, not an item waiting for feedback. Failed planner fallback retains explicit provenance or becomes UNATTRIBUTED; never silently borrows latest plan |

Pinned stale behavior is code, not an assumed FIFO exception: the scheduler
drains stale token counts, skips finished/failed-load/drop-mode results, and
still invokes statistics for eligible delivered stale results [S1,T1].

## Design options and decision matrix

| Mechanism | Correctness | Patch depth | Risks | Necessary proof/tests |
|---|---|---|---|---|
| 1. Per-request FIFO popped by feedback | Reject | Project planner/scheduler only | FIFO **execution completion** is proven, but callback arrivals omit zero, empty, dropped and finished cases; skipped events shift the queue | P10/P11 plus every non-callback outcome; enqueue/dequeue lifecycle proof would effectively turn this into an explicit execution ledger |
| 2. Explicit producer metadata beside draft rows | Recommend, subject to tests | SchedulerOutput metadata, V1 runner host-side draft state, ModelRunnerOutput/async wrapper, callback context, planner/adapter interface | Pinned internal patch and ownership across runtime/project; must copy metadata before mutable buffers reset | Actual row remap, buffer overwrite, grammar, preemption, in-flight and cleanup tests |
| 3. Existing step/fence identity alone | Reject as sufficient contract | Capture counter near scheduler, bind to returning output | Identifies verifier, not producer; conditional fences and mutable last values; DP counter resets | Tests showing verifier ID differs from producer; no arithmetic V-1 shortcut |

Provisional design-review scores, 1 worst to 5 best; these are analysis scores,
not claimed approval or scores supplied by Đại/Khôi. Weights sum to 100.

| Criterion | Weight | FIFO | Explicit metadata | Existing ID alone |
|---|---:|---:|---:|---:|
| Correctness | 35 | 1 | 5 | 2 |
| Async robustness | 15 | 2 | 5 | 2 |
| Preemption robustness | 10 | 1 | 5 | 2 |
| Request isolation | 10 | 3 | 5 | 3 |
| Shallow implementation | 8 | 5 | 2 | 4 |
| Pinned compatibility | 5 | 5 | 3 | 5 |
| Upgrade resilience | 5 | 2 | 2 | 3 |
| Telemetry clarity | 4 | 2 | 5 | 3 |
| Testability | 4 | 3 | 5 | 3 |
| Low overhead | 2 | 5 | 4 | 5 |
| Low shared ownership impact | 2 | 4 | 2 | 4 |
| Weighted / 5 | **100** | **2.18** | **4.43** | **2.64** |

Correctness is a hard gate, not merely a weighted preference. Explicit metadata
is the smallest candidate that proves actual producer identity without assuming
every scheduled expectation becomes a callback. Its payload is host metadata;
no GPU kernel, token, sampling or controller-threshold change is proposed.

## Recommended shared contract

Names below are proposed semantics, not implemented signatures. Use one immutable
`DraftOrigin` per request-row generation and one `VerificationEvent` per returned
request row. Reuse the existing `schedule_sequence` and `plan_sequence` numbers;
qualify them with an engine epoch. `current_step` may be diagnostic metadata,
but is not a second canonical sequence. Preserve old telemetry's schedule meaning.

| Field | Producer -> consumer | Lifetime / class | Nullable rule |
|---|---|---|---|
| `engine_epoch`, internal `request_id`, `request_incarnation` | scheduler admission -> worker/results/adapter | Request and in-flight contexts / identity | Required for a draft row. Incarnation can reuse that Request object's first dispatched schedule sequence, retained across preemption; no extra counter |
| `schedule_sequence`, `plan_sequence` | custom scheduler/planner -> existing scheduling record | Immutable dispatched output / scheduled, planned | Plan may be absent for bootstrap/config/fallback; schedule required on dispatch |
| `source_schedule_sequence`, `source_plan_sequence` | actual produced-row DraftOrigin -> consumed-slot snapshot -> feedback | Until verifier completion / attributed provenance | Source schedule required for ATTRIBUTED; source plan required for planner-origin work. A newly decode-eligible request may have only a batch plan: retain that batch plan ID and mark no request decision. Bootstrap/config has explicit NOT_APPLICABLE plan origin; unknown fallback remains UNATTRIBUTED |
| `verification_schedule_sequence` | returning output context -> event/telemetry | Verifier execution / scheduled identity | Required for new verification events; distinct from source schedule |
| `decision_origin`, detached scalar decision, `source_preemption_epoch` | scheduler -> DraftOrigin -> feedback | Producer generation / planned provenance | Decision origin: REQUEST_PLAN, BATCH_PLAN_ONLY, BOOTSTRAP_CONFIG, RETAINED, UNKNOWN. BATCH_PLAN_ONLY has no invented request depth/topology; retained must point to a known immutable decision or be UNATTRIBUTED |
| Producer batch K, request budget, produced slots; verified slots N, invalid I, drafted D, accepted A | scheduler/runner/verification -> event | Generation/event / scheduled vs measured | Do not infer produced count from planned K; unknown measured count stays unknown with reason. Do not conflate shared batch K with request prefix |
| `producer_outcome` | actual proposal/buffer replacement -> consumed-row result | Producer generation / runtime outcome | Required: GENERATED, ZERO_K, SKIPPED, SYNTHETIC_ZERO_FILL or ERROR; positive vLLM verification-slot counters do not by themselves prove real drafting |
| `lineage_status`, `lineage_reason`, `output_disposition` | binding/return handler -> telemetry/controller | Per event / provenance and lifecycle | Non-null enums/reason. Disposition CURRENT, STALE_DELIVERED, STALE_DROPPED, FINISHED_DROPPED, ERROR |

Statuses: UNKNOWN for legacy/uninstrumented capability; NOT_APPLICABLE for no
speculative contribution; UNATTRIBUTED for a known nonzero contribution with
missing/ambiguous origin; ATTRIBUTED for exact slot binding; STALE for an
invalidated request/preemption generation, retaining any known source IDs and
the explicit disposition. Fully grammar-invalidated slots retain origin and
reason, D=0; this is distinguishable from a K=0 no-draft generation. Null alone
never encodes these different causes.

Binding rules:

1. A dispatched output carries immutable decision references, not `_plan` or a
   mutable Request reference. Planning errors and dispatch failures are explicit.
2. Stamp row origin at the actual buffer-producing branch, with producer_outcome:
   successful generation, K=0, skipped drafter and synthetic zero-fill remain
   distinct. Synthetic rows can retain the creating execution ID but must not
   be represented as successful draft-model proposals; record generated count
   separately from the shape of padded/verified rows.
3. At input preparation, snapshot origin using exactly the same previous/current
   row indices and prefix used to copy draft tokens. Structured-output
   DraftTokenIds transport must retain the same origin through validation/padding.
   Do not calculate source as verifier sequence minus one.
4. Snapshot this consumed origin before the runner clears/replaces its draft
   state. Attach it to that execution's ModelRunnerOutput, preserved by the
   async output wrapper, alongside the exact verifier identity.
5. The result handler establishes an execution-scoped callback context from the
   paired output/result, cleared with try/finally. The existing stats hook reads
   this context by full request key; it never reads a global latest context.
   A typed VerificationEvent passes counts and provenance to planner/adapter.
   A version-gated hook at the native per-request result branch must expose its
   actual stale/drop/failed-load disposition; a wrapper must not reconstruct it
   from mutable counters after update_from_output has changed them.
6. Retire each execution context on completion/drop/error, even if there was no
   stats callback. Free active request/controller state at completion; retain
   only bounded immutable in-flight references/tombstones until output drain.
   A backend without a terminal guarantee needs cancellation/drain or explicit
   UNATTRIBUTED recovery, not silent FIFO eviction.
   Worker draft-row metadata lives independently until consumption or buffer
   overwrite; producer-result completion alone cannot retire that row origin.
   A superseded/unconsumed suffix is terminally discarded, not queued for a
   future callback. This avoids per-request FIFO assumptions during cleanup.
7. Preserve `tree_execution_verified=false`; source topology, if saved, remains
   planned metadata. IDs cannot establish tree execution.

No EngineCoreOutput frontend-schema extension is needed to feed the controller:
feedback is handled inside scheduler update before those client-facing outputs
are built. The proposed transport is SchedulerOutput -> actual worker draft row
-> consumed-row result -> scheduler callback context -> VerificationEvent.

## Controller and scientific impact

Feedback is **not telemetry-only**. `record_verification` calls adapter feedback,
which calls controller.update(A,D), changing EMA/window/debiasing and future K.
The default scalar controller uses measured A/D rather than latest K, so missing
source-plan metadata alone does not prove the verified scalar algorithm wrong.
It does prevent decision-conditioned analysis, exact source-credit assignment
and safe interpretation of stale/incarnation-mismatched updates.

TreeAdaptiveController.update reads `_last_selected_depth`. A delayed sample
can therefore normalize using a different decision's depth; the known L/d
yield-versus-probability issue is independent. Exact **source** depth/topology
and actual execution mode are prerequisites for a source-aware tree estimator,
but lineage alone neither fixes the estimator nor authorizes tree runtime.

Recommended policy for alignment: preserve the current scalar controller's
measured-count semantics during a telemetry-only rollout; expose disposition
and whether an update was applied. Gate any future topology/depth-conditioned
update on exact attributed origin. Decide stale-delivered scalar learning and
missing-origin policy explicitly; do not silently change learning while adding
metadata. No estimator/controller change is made in this session.

## Test plan before implementation

| Level | Tests and exact acceptance assertions |
|---|---|
| L1 pure immutable event model | P10/K2 produces row, P11/K4 exists before return: feedback names P10/S10 and verifier V11; mutation of latest plan has no effect. Two requests/reused ID and engine restart cannot cross incarnations. K0 produces no feedback expectation yet may verify earlier K2. Invalid N3/I1 -> D2/A<=2 retains same source; all-invalid emits lifecycle outcome without alpha sample. Multiple outstanding generations remain independent and terminate exactly once; missing/duplicate/mixed origins remain explicit, never guessed |
| L2 pinned scheduler/runner contract | Queue at least three in-flight outputs and process FIFO; retain each future/output pairing. Reorder/compact/remove/resume worker rows and assert origin matches actual copied tensor prefix. Snapshot before next draft overwrite. Exercise grammar-deferred sampling, synthetic zero-fill/drafter skipped, empty output, request finish/abort, preemption deliver/drop/re-preempt, failed KV load, scheduling/planning/worker errors and cleanup I/O failure. No context leak on exception; no controller resurrection; no callback queue drift. Verify source plan/schedule and verifier ID separately |
| L3 pinned real-engine smoke | After approved implementation, instrument actual production/consumption/result origins on vLLM 0.30.0, exact SHA and pinned weights. One request plus two concurrent requests; natural K change, deliberate delayed result consumption without changing science, verify source against actual worker row stamps. Include K0, invalid slots/grammar and preemption/drop only where safely reproducible; otherwise report those cases unverified. Require one terminal outcome per dispatched request contribution, correct cleanup, no unknown origin on tested real draft feedback, preserved scalar counters/labels, no performance claim |

Existing tests [T1-T6] support upstream lifecycle reasoning; they do not test
this proposed contract. The proposed L1/L2/L3 tests have not been implemented
or run. Source-backed design is not a verified lineage implementation.

## Ownership and unresolved decisions

Role boundaries below are proposed review boundaries, not assigned staffing.
Đại and Khôi should jointly ratify the event semantics and designate the owner
for each interface; no current owner is reassigned by this document.

| Unresolved question | Classification | Shared impact |
|---|---|---|
| Exact metadata carrier and serialization between scheduler, V1 runner and result | TEAM TECHNICAL ALIGNMENT REQUIRED | Runtime/deployment patch owner plus scheduler owner; version-gated internal API adapter |
| VerificationEvent fields, enum/null semantics, old telemetry compatibility and incarnation key | TEAM TECHNICAL ALIGNMENT REQUIRED | Planner/adapter and telemetry consumers; agree schema version and optional-field rollout |
| Delivered-stale/missing-origin scalar updates and future source-aware tree feedback | TECHNICAL REVIEW REQUIRED | Algorithm policy; preserve baseline until explicitly agreed |
| Worker binding coverage for grammar/synthetic/no-draft paths and L2/L3 negative tests | TECHNICAL REVIEW REQUIRED | Runtime/testing owners; explicit unsupported-path outcomes |
| Retention bounds and terminal error/cancellation drain | TECHNICAL REVIEW REQUIRED | Avoid request leaks or false attribution during recovery |
| Tree L/d estimator and source-depth normalization | TECHNICAL REVIEW REQUIRED | Separate calibration review; no tree patch here |
| Reuse existing sequences; reject latest-plan and source=V-1 inference | TECHNICAL DECISION | Proposed invariant and identifier choice |
| Which reviewed evidence paths become a local checkpoint | TECHNICAL DECISION | Explicit list prepared; staging/commit not performed |
| Prior smoke's pending L4 probe cancellation confirmation | TECHNICAL REVIEW REQUIRED | Separate Modal control-plane cleanup uncertainty; not lineage evidence |

**PM DECISION REQUIRED FOR API DESIGN: NONE.** The project-level PM scope
decision remains pending. This technical proposal requests no scope, DoD, owner,
deadline or resource change. If implementing the reviewed design later requires such a change,
raise that specific governance issue rather than asking PM to choose an API.

**First action:** Đại and Khôi review and accept or amend the DraftOrigin ->
VerificationEvent binding invariant and the interface ownership boundary in
one technical alignment review, before assigning implementation.

[S1]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py#L1870
[S2]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/async_scheduler.py#L19
[S3]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py#L630
[S4]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py#L1790
[S5]: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/llm_base_proposer.py#L515
[T1]: https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/core/test_async_scheduler.py#L575
[T2]: https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/engine/test_engine_core.py#L244
[T3]: https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/core/test_deferred_block_free.py#L326
[T4]: https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/spec_decode/test_dynamic_sd.py#L147
[T5]: https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/core/test_scheduler.py#L1690
[T6]: https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/engine/test_engine_core_client.py
