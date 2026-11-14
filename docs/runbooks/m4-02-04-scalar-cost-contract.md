# M4-02-04 canonical scalar cost measurement contract

Protocol proposal: `scalar-retirement-window-v1`, 2026-11-11, Option B.
This is a separate diagnostic profiling protocol, not a change to the frozen
M1/M2/M3 performance/lossless campaigns. No controller policy is connected.
No GPU smoke or full grid was run. Passing the CPU evidence validator does not
prove collector fidelity, matched-reference equivalence, or profitability.

Corrected-context follow-up: [source/L2 proof and tiny smoke specification](m4-02-04-corrected-context.md).
That bounded reconstruction contract supersedes the earlier unresolved context
gap below; live snapshot transport and cross-arm trajectory matching still need
validation. No GPU execution is authorized by passing the helper tests.

## Baseline and protected work

Branch `feat/m4-02-04-scheduler-telemetry-v02`, HEAD
`b4c13637149b14b3ebab20412b044639b7e4c40f`. Existing dirty files before this task:
`tests/test_adaptive_scheduler_glue.py` and
`logs/daijestic/M4-02/M4-02-04_worklogs.md`; preserve both.
Production `src`, canonical launchers and frozen protocols stay unchanged.

## Identity and hardware

`experiments/modal_lanes.py` is the current deployment configuration:

- Target `Qwen/Qwen2.5-7B-Instruct-AWQ`, revision
  `b25037543e9394b818fdfca67ab2a00ecc7dd641`.
- Draft `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, method `draft_model`.
- vLLM 0.30.0; async V1 path and project `AdaptiveAsyncScheduler`.
- Canonical capacity: max model length 4096, max sequences 32, memory fraction
  0.85, dynamic speculative ceiling 4, greedy drafts, heterogeneous vocabulary.
- Next reference profile: **NVIDIA L4 / sm_89**, requested BF16/AWQ Marlin,
  following the M3 reference environment. Record effective quantization too:
  the prior A10 smoke requested AWQ Marlin but engine logs reported auto_awq.
- **Tesla T4 / sm_75**, half/AWQ, is the existing consumer-GPU deployment and
  requires its own later profile for T4 claims. L4 timings cannot be reused.
- A10 was functional smoke fallback hardware, not an L4/T4 calibration profile.

Profile identity must match observed GPU name/architecture, model IDs and immutable
revisions, requested/effective dtype and quantization, draft method, vLLM,
PyTorch, CUDA, driver, scheduler class, executor/TP configuration, compilation and
CUDA-graph modes, async mode, capacity/cache settings and protocol version.
Store exact argv/environment and SHA-256 of the resolved runtime config, not
merely the profile filename or architecture. GPU UUID/boot/process IDs identify
the run; they need not restrict a reviewed profile to one physical device.
Driver/runtime changes require explicit revalidation, not silent acceptance.
Record actual `scheduler_cls` and canonical `scheduler_base_cls` separately if
the measurement adapter subclasses the project scheduler. Record its control
configuration and observer/control source hashes; do not claim an unmodified
runtime class when a diagnostic subclass or instance wrapper is installed.

The existing `load_profile_for_hardware` maps unknown architectures to L4 and
`FlexCostModel.from_json` labels loaded data measured. Neither is an acceptable
identity/provenance guard for this protocol. Leave them unchanged here.
Mismatch => `MISSING_PROFILE`/`UNSUPPORTED_SHAPE`, no calibration claim.
Recommended future behavior is K=0 with the existing memory guards; it avoids
unsupported speculation but does not eliminate draft synchronization cost.
Legacy fixed policy is an explicit opt-in compatibility alternative, labelled
uncalibrated; it cannot satisfy the hardware-aware acceptance gate.

## Cycle, primary timing and async interpretation

A logical execution is one immutable SchedulerOutput paired with the exact
returned ModelRunnerOutput in EngineCore. It verifies **old** draft slots and
also produces **new** drafts for a later execution. It is not a complete
causal proposal-to-verification cycle for one request.

Pinned source: `EngineCore.step_with_batch_queue` queues
`(future, scheduler_output, exec_future)` and later invokes
`scheduler.update_from_output(scheduler_output, model_output)`. See the existing
[pinned source map](m4-02-04-async-feedback-lineage-source-map.md).

Primary observable: a steady-state **retirement window** in the EngineCore
process, containing N consecutive completed nonempty decode batches:

1. Start `t0 = time.perf_counter_ns()` immediately after one
   `scheduler.update_from_output` returns (the anchor batch is excluded).
2. End `tN` immediately after that method returns for the Nth subsequent paired
   completed decode batch. Log each intermediate boundary in the same process.
3. `window_ms = (tN-t0)/1e6`; `C_K = window_ms/N`, ms per retired batch.
4. Sum useful emitted tokens from exactly those N outputs. Never include the
   anchor's output, prefill tokens, speculative placeholders or future drafts.

This is an amortized service interval, not summed submission-to-completion
latency. It includes CPU scheduling, dispatch/waits, overlapped GPU forward and
proposal work, logits/sampling/rejection, state/cache work and scheduler feedback
as they affect the service interval. Outstanding work can cross both boundaries.
Do not charge that work again through component sums. Log in-flight queue depth
at both boundaries; require stationary occupancy/budget after settling, or
retain the sample as a transition diagnostic, not a steady profile cell.

Exclude client/network/admission queueing, startup/model load, compilation,
prefill and final cohort drain. Require an occupied decode cohort, no intentional
sleep inside the window, no preemption/aborts/stale/terminal/grammar-invalid rows.
If any occurs, preserve the full failed/unsupported window. Do not cherry-pick
fast intervals. Running requests waiting on async work are part of service
behavior; unrelated client starvation invalidates a steady cohort.

No `cuda.synchronize()` in the primary timer: it would serialize canonical
overlap. Use a single monotonic EngineCore clock domain; never subtract frontend,
worker or GPU timestamps from it. CPU/GPU component clocks are diagnostics only.
Keep observation overhead measurable (buffer events in memory, flush after the
window). Before treating the result as consumer cost, quantify observer overhead
with repeated instrumented/uninstrumented matched runs; report the difference,
never silently subtract it. No acceptable perturbation margin is established
by current evidence: review it after the small smoke.

## Reference and candidate costs

For K=0,2,3,4 retain the same loaded target AND draft model, speculative ceiling,
executor, scheduler path, cache/graph policy and capacity. Change only declared
profiling budgets in a measurement-only harness. No target-only engine for K=0.
Pinned `SpecDecodeBaseProposer.propose` runs a model forward before its K=0
return; record actual propose/forward counts and output shape (B,0).

Reference `C_0(shape,identity)` uses the same definition and matched initial
context/cohort fixture, hardware/runtime and observation protocol. It includes
draft synchronization. Target-only vanilla is optional explanatory evidence,
never the denominator for this in-engine K decision.

`C_2`, `C_3`, `C_4` are the retirement-window costs with those shared producer
budgets and explicitly recorded request budget vectors. Homogeneous and
heterogeneous vectors are different cells even at the same B/shared K.
Proposal time cannot be attributed to requests simply by their trimmed K_i.

Match reference/candidate context **trajectories**, not only input prompt lengths.
Different K/yield advances context differently during settling and measurement.
Keep exact vectors and histograms; a common context-bin label alone does not
prove equivalence. Reweighting/interpolation requires a separately validated
cost model. Until equivalence or a bounded approximation is demonstrated, mark
the pair `UNMATCHED_REFERENCE` and do not publish a consumer score.

## Shape and decision-time availability

Store one row per actual request per logical verifier: request ID, actual worker
row, prefill/decode flag, context tokens before execution, query tokens, valid
draft slots D, invalid slots, A, emitted tokens, terminal/stale/preemption state.
Also store actual proposer rows/IDs, shared K, per-request effective K_i,
scheduled/running/client counts, total target query tokens, and graph padding.
Record current and future budgets separately; D belongs to current verification.

Cost lookup identity includes ordered/multiset context distribution, query-length
distribution, effective batch, composition, shared K, K_i distribution, padding
and queue mode. Store request mapping before sorting any shape summary. Do not
collapse heterogeneous vectors to B or assume Ki=0 saved proposal compute.

Before `super().schedule`, current code knows running count/flags, past feedback
and memory signals. These are **FORECAST** inputs for the next worker shape.
SchedulerOutput provides admitted IDs/counts later; worker input preparation
provides actual row/context/query/padding metadata (**MEASURED**). Future target
verification rows remain a forecast under async scheduling. Record both with
their own identifiers. Unknown/unsupported forecasts must not become measured
profile matches. Training a forecast model is future work, not this protocol.

The CPU validator accepts only steady pure-decode, saturated valid D=K_i rows;
all other shapes are retained as unsupported evidence, not falsely normalized.
Mixed prefill/decode can be recorded separately but cannot consume this profile.

## Yield, future score and no-root semantics

Preferred numerator is directly observed **useful emitted tokens** after stop/
length processing, summed across the same completed batches as the timer.
For steady nonterminal scalar rows check emitted=A+1; read actual returned
token lists rather than manufacturing counts. K=0 contributes one output per
eligible row, without requiring a speculative feedback event.

Compare possible estimators:

| Estimator | Correctness/data | Current telemetry compatibility |
| --- | --- | --- |
| Conditional p | Under stationary homogeneous Bernoulli assumptions, fit censored prefix likelihood p^A*(1-p)^(1[A<D]); pooled MLE=sum A/(sum A+count(A<D)). Exclude EOS/invalid/stale censoring and prove assumptions. | A,D exist, but legacy EMA A/D is not this estimator. |
| Direct emitted yield | Mean actual emitted tokens per matched K/state/cohort window; no iid geometric assumption. Chosen primary statistic. | New K=0/output observations needed; existing feedback alone omits non-speculative output and terminal clipping. |
| Prefix survival | Estimate P(A>=j) among eligible depth-j trials; keep D>=j and censoring/selection rules explicit. Yield=1+sum survival for nonterminal rows. | A,D support diagnostics, but unobserved depths/variable-K selection need calibration. |

Do not rename EMA A/D or its warm-up-corrected version to theoretical p.
No tree yield estimator is changed. Exact proposal-source lineage remains null;
measurement verifier IDs associate output with its paired execution only.
Theory reference: [Leviathan et al., sections3.1 and3.3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).
Its iid geometric yield and constant-target-verification assumptions are
approximations; the measured wall boundary here takes precedence.

Future batch score: `E[sum useful emitted tokens per retired batch | state,Kvec] /
E[wall ms per retired batch | state,Kvec]`, tokens/ms. Use pooled total tokens /
total elapsed ms for empirical rates, not mean of individual rates or a ratio of
unrelated medians. Keep request yields separate while charging shared cycle cost
once. A throughput objective does not itself establish per-request latency gains.

For a calibrated geometric approximation Y_K(p), compare Y_K/C_K with Y_0/C_0:
`ALWAYS_PROFITABLE` only for strict dominance over the defined domain;
`INTERIOR_BREAK_EVEN` for an interior equality crossing;
`TIE_ONLY` if no strict gain exists but equality does (including equality only
at the maximum feasible yield); `ALWAYS_DOMINATED` when strictly worse
throughout the domain with no equality; `MISSING_PROFILE`, `UNSUPPORTED_SHAPE`, `UNMATCHED_REFERENCE` remain
distinct unavailable states. No feasible root is never encoded as p*=1.
For general empirical yield curves, classify the measured support and uncertainty;
do not assume monotonicity, a unique root or profitability between observations.

## Existing data inventory and reuse

All units below are ms unless noted. Classification applies to every cell in the
named family; failed/OOM/mismatch cells are retained as boundary evidence, never
as successful timing cells. No existing cost cell is REUSABLE_DIRECTLY under the
new full-window contract.

| Artifact | Hardware/model/proposer | K / B / context | Boundary/raw or derived | Reuse status and limitation |
| --- | --- | --- | --- | --- |
| M3 `20261025-m302-l4-v2/forward_costs.jsonl` and summary | L4; canonical target and 0.5B in separate engines | Standalone decode / B=1,2,4,8,16,32 / L=256,512,1024,2048,3072 growing by 64 | Raw differential generate walltime (T65-T1)/64 | REUSABLE_FOR_SANITY_CHECK_ONLY; no canonical dual-engine cycle or verification/sync costs. 300 records/60 cell medians verified in preceding review. |
| Earlier M3 `20261025-m302-l4`, dry fixtures | L4 attempts or synthetic dry run | Declared standalone grids | Raw partial or simulated data | Successful measured cells sanity-only; failed/synthetic cells NOT_COMPATIBLE as cost evidence. |
| `hardware_profile_l4.json`, `profile_sm89_l4.json` | L4; matching named pair | k0..6 / six B / context medians collapsed | Derived k*CD+CT*(1+.008*(k-1)) | REUSABLE_FOR_SANITY_CHECK_ONLY; verification coefficient estimated, K0 target-only. |
| `profile_sm75_t4.json` | T4; named canonical pair | k0..4 / six B / context unspecified | Stored forward costs/approximate roots; no complete matched raw derivation | REUSABLE_FOR_SANITY_CHECK_ONLY; insufficient provenance, no full-cycle evidence. |
| All M4 L4 component reports (`one-cell`, `smoke`, `diagnostic`, `components-29a10c6`, `components-v5`) | L4; target plus EAGLE | k0..5 / declared B subsets through32 / L1024 | Worker synchronized target model forward and EAGLE propose; raw and component summaries | Valid target components sanity-only. EAGLE proposal timings NOT_COMPATIBLE with 0.5B proposer. Incomplete attempts cannot supply missing costs. |
| All M4 T4 diagnostics/components-v3 | T4; target plus EAGLE | k0..5 / B<=16 valid; speculative B32 failed / L1024 | Same component boundaries | Same reuse limits; retain B32 infeasibility, do not extrapolate L4 costs. |
| M4 local-T4 guard attempt | RTX3050Ti observed instead of T4 | No valid grid | Hardware guard/failure | NOT_COMPATIBLE; useful guard evidence only. |
| Existing A10 scalar L3 smoke attempt2 | A10; exact canonical draft-model pair | K0/2/3/4; actual running B1/2; prompts37/39/40 tokens, completion10/48/48 | Raw functional telemetry, HTTP response walltime in seconds | REUSABLE_FOR_SANITY_CHECK_ONLY; no cycle clocks/worker shape traces. Preserves runtime compatibility proof. |
| M1/M2/M3 campaign raw/summary metrics | T4/L4, lane-specific canonical draft or EAGLE | Client concurrency1..32; frozen W1/W2/W3 | Request TTFT/TPOT/E2E, tokens/s; aggregates/response timing | REUSABLE_FOR_SANITY_CHECK_ONLY for workload/load choice; request throughput is not a shared-cycle lookup. No cross-lane conversion. |

## Instrumentation point audit

| Point/start -> end | Clock / CPU or GPU | Sync / granularity | Includes queueing/scheduler? |
| --- | --- | --- | --- |
| Project `schedule` entry -> return | perf_counter_ns / CPU | No GPU sync; one scheduler call | Scheduler CPU only; not GPU completion. Diagnostic only. |
| EngineCore output-update return -> Nth output-update return | perf_counter_ns / CPU wall | No added sync; batch window | Primary service interval including normal engine waits/overlap; excludes external client queueing. |
| Paired execution submission -> materialized future/result and update | Same EngineCore CPU clock | Existing future completion; per batch | Includes execution queue residence; overlaps other batches. Diagnostic latency, never sum as cost. |
| Worker model-forward pre/post hooks | Worker perf_counter + explicit CUDA sync | Per target batch | GPU+host elapsed, excludes logits/sampling; synchronization perturbs pipeline. Separate diagnostic run only. |
| DraftModelProposer entry -> completed propose | Worker CUDA events or synchronized wall | Per shared proposer batch | Includes preparation/sampling if wall boundary encompasses them; K0 forward remains. Separate diagnostic run only. |
| Runner sample/rejection entry -> materialized outputs | CPU clock/CUDA events, separately labelled | Per batch; event completion must be observed | GPU kernel timing excludes CPU dispatch; CPU dispatch timing excludes deferred GPU work. Do not add blindly. |
| `update_from_output` entry -> return | EngineCore CPU clock | Existing result already materialized | Feedback, stopping/cache/request state; no independent GPU inference time. |

CUDA event durations need stream/dependency documentation. A worker clock cannot
be joined arithmetically to an EngineCore clock. Existing component timing sums
are rejected as the primary denominator until disjointness/completeness/overlap
are proven. Current instrumentation has no primary retirement-window collector.

## Minimum matrix and sample rules

The minimal **instrumentation smoke** is K0 and K2, B1, the same short fixture,
on L4. One small raw window per K after settling is enough for hook sanity only,
not calibration. Require identity, ordering, nonzero durations, actual shape,
one proposal call per selected logical execution and observed K0 sync forward.

First required seed matrix: homogeneous K={0,2,3,4} x actual B={1,2}, short
context bin [32,256): 8 cells. Add B2 shared/effective vectors (2;2,0),
(4;4,0), (4;4,2), reusing matched B2 K0 reference: 3 cells. These cover the
real smoke's B1/2 and short contexts plus the shared-cost isolation boundary.
This **11-cell seed is not sufficient for full deployed-domain claims**.

Before a grid, perform a shape census on tokenized frozen W1/W2/W3 inputs and
existing replay configuration. Freeze observed effective B/query/context/Kvec
support in a manifest. Campaign client concurrency includes1,2,4,8,16,32 but
does not establish these as actual worker B. Additional populated worker B,
context distributions and heterogeneous mixes are REQUIRED for claims covering
those shapes; unsupported ones explicitly fall back instead of extrapolating.
Use half-open diagnostic bins [0,256),[256,1024),[1024,2048),[2048,4096), retaining
exact context vectors. Bin boundaries are labels, not evidence of constant cost.
L1024 is an optional M4-comparability anchor until observed workload support
makes it required. Do not multiply every old M3 context by every B blindly.
B3 vector(4,2,0) with matched K0 is an optional integration-stress extension.
T4 is a separate required campaign before T4 calibration claims. Tree/EAGLE,
other GPU substitutions and target-only K0 are out of scope for these profiles.

Separate profiling rule, reusing M4's sampling scale: >=5 excluded warm-up
cohorts and >=50 independent measured cohorts per cell. Each cohort settles
for at least max(5,2*observed queue capacity) completed decode batches after
prefill, then records 8 retirement intervals (B requests, 8 completed batches).
Fresh cohorts, not startup/model reload, are the replicate unit. Balance paired
K order deterministically across replicates; preserve fixture hashes/seeds.
Fixed K for the entire cohort, including settling; any change invalidates it.
Warm-up compilation/caches must be identical across candidates. K0 uses the
same preloaded ceiling4 engine. No empty-cohort or final-drain samples.

Report raw window durations, ms/batch median/p90/min/max, spread/median (flag
>15%, inherited diagnostic threshold), sample SD, per-cohort outputs and pooled
tokens/total ms. Keep exact shape trajectories and queue occupancy. Bootstrap
uncertainty over independent cohorts, not correlated cycles, before consumer
publication. If uncertainty crosses a decision boundary, report inconclusive;
do not add repeats solely until a desired result appears. Extra repeats need a
predeclared precision/stopping rule. Do not remove outliers. Failures remain in
the denominator of completeness; stop after OOM/timeouts and never retry silently.

## Profiler reuse and required implementation surface

- Reuse `bench_source.snapshot` for committed bytes and hashes, `modal_lanes`
  for canonical config/pinned model revisions, and M4 runner helpers for
  hardware inspection, exclusive run reservation, subprocess timeout/failure
  preservation and dependency inventories. Keep existing M4 reports unchanged.
- Do not reuse M3 differential generate timing as cycle cost.
- Do not call M4 `engine_kwargs`/observer for primary measurements: they select
  EAGLE, eager mode and async_scheduling=False, and K0 removes the draft model.
- Prepare a dedicated measurement adapter in `experiments/` using the existing
  Modal image/runner transport. It must retain draft_model/async/graph settings,
  observe EngineCore paired results, capture actual worker shapes, and collect
  token counts after scheduler stop handling. Measurement-only fixed-budget
  control must be logged, never alter committed production methods, and abort
  on memory pressure rather than overriding safety to force K.
- Read-only wrappers may tag SchedulerOutput objects with local verifier IDs
  in the observer; no source-plan lineage sidecar or guessed attribution. Pair
  worker metadata by transported execution identity, never latest request state.
  This observer transport still needs L2 and real small-smoke validation.
- Add no production callback/shared-interface changes without team alignment.
  If observation requires them, stop that collector workstream and review it.

Implemented now: `experiments/scalar_cost_evidence.py`, CPU-only non-consumer
report validator, and `tests/test_scalar_cost_evidence.py`. No GPU collector,
budget override, engine hook or runnable grid launcher was added prematurely.
The proposed adapter work above is the smallest remaining collection surface;
existing M4 collection cannot honestly be relabelled canonical scalar.
Proposed remaining files: `experiments/scalar_cost_capture.py` for observation
and measurement-only control, `experiments/scalar_cost_runner.py` for orchestration
using the existing runner helpers, and `tests/test_scalar_cost_capture.py` for
pinned-contract collector tests. They are a preparation plan, not present or
executable artifacts. Do not launch the current EAGLE runner as a substitute.

## Raw schema and offline reproduction

Future output root: `experiments/runs/m4_02_04/scalar-cost/<unique-run-id>/`.
Reserve with exclusive mkdir; no overwrite. Preserve `manifest.json`, `env.json`,
`argv.json`, `config.json`, `source_manifest.json`, `worker_events.jsonl`,
`raw.jsonl`, stderr/server logs, pip freeze, prompt-token fixtures and checksums.
Append/fsync attempts before execution; flush immutable observations after each
window. A parent watchdog must record timeout/OOM/aborted partial windows even
if the child cannot serialize. Retain raw failures before running a validator.

`manifest.json` specifies run_id, timestamp_utc, full git_sha, source cleanliness
and source hashes; observed_identity/expected_identity (all fields in validator
IDENTITY_FIELDS); warmup, repeats, cycles; cells with cell_id, shared_k,
request_budgets, context_bin, reference_cell_id; plus fixture hashes, seed/order,
initial shape, settle count, queue mode, timeout and observer configuration.

Every `raw.jsonl` window: run_id/git_sha, cell_id, independent cohort_id,
warmup/measured phase, repeat,
status/error; timing_boundary=`enginecore-post-update-to-post-update`,
clock=`perf_counter_ns`, process/boot clock_domain, start_ns/end_ns;
tree_execution_verified=false; and N retirements. Each retirement has verifier_id,
end_ns, shared_k, producer_rows, proposer_calls, draft_forward_calls, and requests
with request_id/effective_k/context_tokens/query_tokens/drafted/accepted/emitted/
prefill/terminal/stale. Attach referenced worker event IDs, queue depths, initial
shape and exact configuration/provenance. Do not fill missing observations with
defaults. Fields outside the CPU validator's coverage still require collector QA.

```text
python -m experiments.scalar_cost_evidence <existing-run-directory>
python -m pytest -q tests/test_scalar_cost_evidence.py
```

The CLI only reads manifest/raw records and prints JSON. `complete=true` means
declared successful attempts and structural invariants are present, NOT that
the run is calibrated. `consumer_profile=false` is unconditional. Missing cells,
failed cells, mixed identities, invalid timing/K/shape and fabricated K0 skip
semantics cannot become a consumer cost matrix.

## Gates, current readiness and next action

CPU tests cover identity, shape/K labels, K0 synchronization, timestamp ordering,
raw serialization, pooled aggregation, warm-up exclusion, missing and failed
cells. Synthetic fixtures are never GPU evidence. Collector-specific L2 must
next prove FIFO output pairing, output/token slicing, same-process timing,
buffer flushing and unchanged async queue behavior against pinned methods.

Verification in this task:

```text
python -m pytest -q tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_modal_profiler.py
107 passed (baseline)

python -m pytest -q tests/test_scalar_cost_evidence.py tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_modal_profiler.py
161 passed (final; 54 new measurement-validator cases, 107 existing cases)
```

These are CPU tests, including synthetic raw records; no timing values from
them are hardware evidence. Production/config/frozen-protocol diff is empty.

**TECHNICAL REVIEW REQUIRED / TEAM TECHNICAL ALIGNMENT REQUIRED** before a GPU
grid: implement and validate observer transport and reference context-trajectory
matching, then quantify observer perturbation. These gaps are not hidden by a
schema validator. The direct walltime definition is specified; fidelity and
reference comparability are not yet demonstrated by current instrumentation.

Small smoke: **NOT RUN**. Full grid: **NOT RUN — CONTRACT FIRST**.
Local environment inspected without model execution: Windows, RTX3050Ti Laptop
GPU (driver592.27), PyTorch2.6.0+cu124, no installed vLLM. It cannot supply L4
canonical evidence; no remote GPU call or capacity reservation was made.
Verdict: **NOT READY — TECHNICAL CONTRACT GAP** for collection approval.
No new PM scope/resource decision is needed to prepare the existing L4 target.
Current L3 functional smoke remains valid; this protocol supplies no new L3 data.

First action: validate the measurement-only paired-retirement/worker-shape
observer design against pinned vLLM before implementing its small-smoke adapter.

## Observer audit follow-up (2026-11-11)

See [the observer fidelity audit](m4-02-04-scalar-observer-audit.md) for captured
pairing/retirement tests and the measurement-only CPU adapter now implemented.
The earlier planned-file list above describes the prior checkpoint; capture.py
and its L2 tests now exist, while a live GPU installer/runner remains absent.
Important correction: worker CPU context buffers can be optimistic under async
speculation. They must not be recorded as measured GPU context. The observer
rejects them, so the current verdict is **NOT VERIFIED — SHAPE GAP** and neither
smoke nor full profiling is authorized by these CPU results.
