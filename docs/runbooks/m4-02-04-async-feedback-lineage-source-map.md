# M4-02-04 pinned async lineage source map

Companion evidence for [the alignment proposal](m4-02-04-async-feedback-lineage-proposal.md).
Inspected package: vLLM 0.30.0 from the project-pinned Modal image, read on CPU.
The 392-file installed-source SHA-256 inventory and CPU collector are under
`experiments/runs/m4_02_04/20261110-async-lineage-design/`.
Full installed/upstream source caches remain local inspection material, excluded
from Git. No design claim requires a local-only link: use the pinned URLs below
and inventory hashes. To reconstruct installed text, run the retained
`collect_installed_sources.py` against the original project image from
`a6cd6ad5cee5374738f890c270ada4ca01f0946b` in a separate checkout, then verify
every collected path against `installed_source_manifest.json`. Re-running the
collector is an explicit network operation, not part of this checkpoint.

## Source anchors

All line numbers below refer to the collected installed files. Official links
are pinned to v0.30.0; installed files remain the behavior authority.

| Key | File and inspected executable path |
|---|---|
| P1 | `src/server/adaptive_scheduler.py:58-126`: plan, lookup update, hook, feedback subtraction |
| P2 | `src/server/budget_planner.py:71-210`: decision snapshots, scheduled telemetry, verification/controller call, finish |
| P3 | `src/proposer/adapter.py:105-153`: feedback -> controller.update; session/history removal |
| P4 | `src/proposer/controller.py:84-93,251-254`; `src/proposer/tree_adaptive.py:144-166`: scalar A/D and latest-depth tree normalization |
| V1 | [Scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py): current_step 562; placeholder selection 813-826; dynamic K 1392-1398; output construction 1407 onward; conditional fence 1451-1452; request updates 1530-1540 |
| V2 | [AsyncScheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/async_scheduler.py): after-schedule 19-49; stale-aware output update 51-72 |
| V3 | [SchedulerOutput](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/output.py): 228-313; old scheduled_spec_decode_tokens and new num_spec_tokens_to_schedule coexist |
| V4 | [EngineCore](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py): queue type 210-216; simple step 589-619; batch queue 630-744 |
| V5 | [UniProcExecutor](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/executor/uniproc_executor.py): AsyncOutputFuture 32-49; RPC/futures 90-145 |
| V6 | [V1 runner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py): actual token row selection 1778-1925; rejection sampling 3658-3687; prev-row map 3784-3788; draft clearing/proposal 4617-4751; result snapshot 4779-4847; DraftTokenIds 4898-4954; proposer K 5034,5277-5296 |
| V7 | [DraftModelProposer](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/draft_model.py): inheritance 19 onward; [base proposer](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/llm_base_proposer.py): propose 515 onward, K=0 return 623 onward |
| V8 | [ModelRunnerOutput and DraftTokenIds](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/outputs.py): 324-435; no producer step fields |
| V9 | [Scheduler result processing](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py#L1870): paired output 1870 onward; stale/drop/finished 1944-1970; per-request counts/callback 1978-2016; request stop/free 2034-2124; invalid-slot trimming/padding 2425-2460; statistics aggregation 2798-2815 |
| V10 | [Request](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/request.py#L157): placeholder/in-flight/stale/eligibility/fence state 157-181; preemption implementation Scheduler 1477-1517 |
| V11 | [Frontend outputs](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/__init__.py): EngineCoreOutput 195 onward, EngineCoreOutputs 253 onward; built after scheduler callback |
| V12 | [SpecDecodingStats](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/metrics.py#L18): batch aggregation 18-49; [input request IDs](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/input_processor.py#L262): optional randomization 262-279 |

## Object and lifetime trace

P means a planning decision, S a producer schedule, V a verification schedule.
The same SchedulerOutput is simultaneously V for old drafts and S for new drafts.
An absent step identity is recorded as absent, not inferred from an adjacent event.

| Boundary and object/type | Request identity | Step/scheduler identity now | Draft / accepted data | Producer -> consumer; lifetime |
|---|---|---|---|---|
| `BudgetDecision` -> `StepPlan` | dict key request_id | Project plan_sequence; decisions deep-copied | Planned request k/nodes; no measured A | Controller -> planner/custom scheduler; `_plan` replaced next scheduling call |
| `dynamic_sd_lookup: list[int]` | Shared batch lookup, not request-specific | None on list | Planned shared K; no A | Custom scheduler -> Scheduler.schedule; overwritten each adaptive call |
| `Request.spec_token_ids` | Request.request_id | No producer ID | Real IDs or async -1 placeholders; no A | Previous after-schedule -> next schedule; selected prefix then cleared/replaced |
| `SchedulerOutput` | request-keyed dicts, NewRequestData/CachedRequestData | No transported durable schedule field; current_step only on scheduler | scheduled_spec_decode_tokens = old verification slots; num_spec_tokens_to_schedule = new producer K | Scheduler -> executor and retained EngineCore pair; until execution/result completion |
| `AsyncScheduler._update_after_schedule` | Requests looked up by scheduled ID | Reads current_step only for V2 PP eligibility | Reserves sampled+old speculative output count; sets new K placeholders | Scheduler -> future request scheduling; mutable request state |
| `Future`, queue triple | IDs inside retained output | Exact output object retained, FIFO | No new counts | EngineCore/executor -> result handler; one execution in flight, possibly several queued |
| Runner `InputBatch`, `prev_positions` | current internal ID -> previous row index | No durable producer ID | Copy scheduled prefix from prior tensor | Runner bookkeeping -> `_prepare_input_ids`; previous/current batch transition |
| `_draft_token_ids: Tensor`, optional `_draft_token_req_ids` | Row mapping; optional copied IDs | No origin sequence | Shared producer K columns, possibly zero/synthetic; no accepted count yet | Proposer -> next target input; cleared/replaced after target sample, not tied to callback arrival |
| `DraftTokenIds` | req_ids alongside list rows | No producer ID | CPU draft token rows | Runner -> scheduler grammar validation/deferred sampling where used; ephemeral snapshot; not ordinary async feedback channel |
| `SpecDecodeMetadata`, sampler tensors | Batch row order | No producer schedule identity | Per-row verification draft lengths; accepted prefix plus bonus encoded in sampled IDs | Runner input prep -> rejection sampler -> async host copy; one verifier execution |
| `ModelRunnerOutput` / `AsyncGPUModelRunnerOutput` | req_ids plus req_id_to_index, copied lists | Per-execution object pairing only | sampled_token_ids rows; A derived later; no explicit producer metadata | Runner -> executor Future -> EngineCore; async wrapper retains execution output until host materialization |
| `Scheduler.update_from_output(output,result)` | exact output ID -> result row | Exact verifier output via queue; latest scheduler counters may have advanced | N=len(scheduled old spec list); A=max(len(sampled)-bonus,0); invalid I from same verification output | EngineCore -> per-request stats hook; one loop iteration after stale/finish guards |
| `make_spec_decoding_stats` -> `record_verification` | request_id only | **No source plan, producer schedule or verifier ID parameter** | Project D=N-I, A unchanged; ignores D<=0 | Scheduler subclass -> planner -> adapter/controller; immediate state update during returned verifier processing |
| `SpecDecodingStats`, per-request metrics, `EngineCoreOutputs` | Batch aggregate may lose request detail; frontend outputs retain request ID | No producer identifiers | Aggregated drafted/accepted/per-position stats; request metric history | Scheduler -> engine frontend/logging; aggregate lifetime longer than one producer contribution |

## Cardinality proof and limits

Code evidence: V9 loops `num_scheduled_tokens.items()` once per unique request,
indexes one sampled row and calls the statistics hook once if old scheduled-spec
tokens and generated tokens exist. It does not pass the preexisting aggregate
counts as that request's counts. V6 computes all prior draft indices for the
request using one `prev_index * prev_num_spec_tokens` plus its prefix length;
the source is one `_draft_token_ids` allocation/generation, not a concatenation
of steps. V12's observe_draft then sums distinct request contributions. Thus the
project callback has per-request, per-verification granularity; downstream
aggregates do not. This proof is specific to the pinned scalar V1 draft_model
path; it is not evidence for V2, arbitrary-tree verification or a future runner.

Verified slot count can be less than actual generated batch K due to request
trimming, token limits or grammar. Counter A is computed before later request
stop truncation, so it need not equal visible completion text length. A source
decision's planned node count is not a measured generation count.

EngineCore's FIFO pairing establishes result order, not one callback for every
proposal. Zero-token outputs, missing/finished requests, failed KV loads,
drop-mode stale results and invalid generated rows can omit callbacks. Delivered
stale results can reach the callback, whereas drop-mode stale results cannot.
These branches disprove an unconditional per-request callback FIFO pop rule.

## Official pinned tests inspected

These tests were read only. No result here is a claimed test pass or lineage test.

| Test | Relevant evidence |
|---|---|
| `test_async_scheduler.py::test_kv_pressure_preemption_with_inflight_output` | Multiple in-flight steps, delivered stale results, variable acceptance, counter rollback |
| `test_async_scheduler.py::test_reset_prefix_cache_with_inflight_output_under_kv_pressure` | Drop-mode stale output and repeated preemption while stale shares remain |
| `test_async_scheduler.py::test_abort` and `test_preempt` | Request lifetime and absence of a universal callback |
| `test_engine_core.py::test_engine_core_concurrent_batches` | Engine queue/future/output pairing; test explicitly disables async scheduling, so it alone is not an async-lineage proof |
| `test_engine_core_client.py` | Frontend client/output routing and async lifecycle, not a draft producer identity |
| `test_deferred_block_free.py::test_fence_held_across_multiple_inflight_steps` | Conditional fence counters across nonempty in-flight steps |
| `test_dynamic_sd.py::test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests` | New output K from scheduled batch size; also empty-output/static fallback tests |
| `test_scheduler.py::test_per_request_spec_decode_detailed_records_per_step` | One observed draft/accept pair per verification round |
| `test_scheduler.py::test_per_request_spec_decode_subtracts_invalid_drafts` | N3-I1=D2; accepted count is unchanged |

The project's `test_delayed_feedback_does_not_inherit_latest_plan` deliberately
checks null source IDs. It protects scientific honesty; it does not provide
an implementation that attributes delayed feedback.

## Collection diagnostics and checkpoint scope

The first CPU collector failed module import because its analysis harness did
not mount experiments/modal_runner.py. The isolated app
`ap-xI2NYO6H6foA0vyPatRF9V` was explicitly stopped. The corrected collector
mounted the exact committed experiments tree; CPU call
`fc-01M72HV76JR0RYQ07W6JE6WN1Q` succeeded and its app completed. This was a
collector packaging issue, not a new vLLM compatibility failure. No GPU was
requested and no production module was edited.

Current checkpoint selection: this appendix, the proposal and host-side
feasibility document; installed-source/test hash inventories; source identity
comparison; CPU inspection results and source-window audit; small reproduction
scripts and call metadata. Full source caches, bytecode, old staging lists and
historical completion/review snapshots remain local and are excluded. The
current smoke selection is recorded in its `checkpoint_manifest.json`.
