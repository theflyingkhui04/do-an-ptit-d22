# M4-02-04 — Safe independent implementation evidence

## Session identity

- Timestamp: 2026-11-10, 16:29 +07:00 (Asia/Bangkok); final checks follow below.
- Branch: `feat/m4-02-04-scheduler-telemetry-v02` (already existed).
- Base and fetched `origin/develop`: `9bc0f8d86e7a1f620a4f8cdaee79230b3bc41b14`.
- No drift from the supplied baseline. Initial working tree was clean; the existing checkout was reused.
- PM decision A/B/C remains pending. This entry does not change task scope or DoD.
- No commit, push, PR, merge, Trello update, dependency change, benchmark, model download or experiment-protocol edit was performed.

## Git and inventory evidence

Commands: `git status --short`, `git branch --show-current`, `git remote -v`,
`git fetch origin`, `git rev-parse HEAD origin/develop`,
`git log -1 --oneline origin/develop`, `git branch -a`.

The sandbox account initially triggered Git's ownership check. Subsequent Git commands
used the per-command option
`-c safe.directory=<repository-root>`;
global configuration was not changed. Fetch initially could not write the linked
worktree's `FETCH_HEAD` in the parent checkout. The approved elevated retry succeeded.

Read the planner, scheduler, proposer adapter/controller, tree controller, topology
catalog, cost model, existing planner/glue tests, and `.agents/rules/vietnamese-capitalization.md`.
Searched `Interface A`, telemetry, dynamic lookup/config, tree decision types,
topology identities, parent indices and deployment pins with `rg`.
An initial unrestricted evidence search produced truncated output; scoped source and
Markdown searches supplied the contract and deployment findings below.

## Pinned vLLM source trace

Deployment pin chain:

- `experiments/modal_lanes.py`: `VLLM_VERSION = "0.30.0"`.
- `experiments/modal_runner.py`: image installs `vllm=={lanes.VLLM_VERSION}`.
- `experiments/m4_modal_runner.py`: reuses that image; validates runtime version.
- `experiments/requirements-m4-cuda.txt` also pins `vllm==0.30.0`.
- Local probe: `python -c "import importlib.util; print('vllm:', importlib.util.find_spec('vllm'))"`
  returned `vllm: None`. No installed source or GPU execution was available.

Read upstream source at the exact tag using Firecrawl retrieval:

- [AsyncScheduler v0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/async_scheduler.py):
  `_update_after_schedule` sizes next-step placeholders from output K, skips prefill,
  and reuses a read-only placeholder list. Project trimming replaces a request's list.
- [Scheduler v0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py):
  initialization builds `dynamic_sd_lookup`; `schedule` reads
  `lookup[len(num_scheduled_tokens)]` for a nonempty scheduled set and invokes
  `_update_after_schedule` before returning. Empty output retains upstream's configured
  K even though the project planner's empty plan is zero.
- The same scheduler's `update_draft_token_ids_in_output` trims returned drafts to
  the already scheduled placeholder count and handles grammar-invalid padding.
  Current scheduled drafts and next-step placeholders are distinct observations.
- `make_spec_decoding_stats` accepts `(stats, drafted, accepted, invalid, request_id)`;
  the project feedback subtracts invalid drafts once before updating its controller,
  then forwards the original arguments to the base implementation.
- `_free_request(request, delay_free_blocks=False)` returns the KV/EC transfer pair;
  the project preserves that call/return path.
- [Dynamic SD utils v0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/dynamic/utils.py):
  `num_speculative_tokens_per_batch_size` expands into a dense 1-indexed scalar K
  table capped at configured speculative tokens. Index zero is unused.

Captured scheduler source under
`experiments/runs/m2_03/boundary279_20261018T132244Z_15f4dae9/vllm-originals/`
also contains the relevant lookup/feedback/free hooks. It was read, not modified.
Latest upstream behavior was not substituted for the pinned tag.

## Implementation and telemetry semantics

Changed implementation files:

- `src/server/budget_planner.py`: additive `StepPlan` fields retain detached existing
  controller decisions, uncapped per-request node counts and `N_batch = max(N_i)`.
  Existing `per_request_k`/`batch_k` remain capped scalar budgets. The legacy
  `min_batch_k` floor and `per_request=False` behavior remain supported. Negative
  configuration is rejected; negative decision budgets cannot execute below zero.
- `src/server/adaptive_scheduler.py`: retain the existing dense lookup update and
  trimming semantics; observe actual scheduled output K and next placeholders after
  the base hook and trimming. Freeing a request removes its plan snapshots even
  when aggregate telemetry I/O fails. Observer/protocol producers are unchanged.

Contract source: `logs/Khoi/spec&plan/treeAdaptive/spec.md` §4.6 and the M4-02-04
entry in its companion `plan.md`. The spec defines per-step decision fields, whereas
the existing adapter emits request-final aggregates. Additions are made at the
planner boundary; the shared adapter API and default scalar controller are unchanged.

- `ADAPTIVE_TELEMETRY_PATH` retains one final aggregate JSONL record per known
  request, including all original adapter fields and values.
- Additive top-level v0.2 decision fields describe the last plan. The detached
  `last_planned_decision` explicitly retains that snapshot. Existing `effective_alpha`
  at the top level stays the legacy final aggregate; its planned counterpart is
  available in `last_planned_decision`. `record_type=request_aggregate` and
  `decision_scope=last_planned_decision` distinguish these scopes.
- Optional `ADAPTIVE_STEP_TELEMETRY_PATH` enables a separate JSONL stream. Paths
  resolving to the same file are rejected. It is disabled by default.
- `planned_decision`: controller output/signals at `plan_step`, with planner-local
  `plan_sequence`, topology metadata, raw N and capped scalar budgets. Following
  the pre-commit audit, raw EMA comes from the controller state and effective alpha
  from its public getter, not the historically overloaded decision field.
- `scheduled_budget`: after the base scheduler hook/trimming; records
  `scheduled_scalar_k` for the proposer handling this output, and
  `next_request_placeholder_budget` for future verification. The earlier
  `next_batch_scalar_budget` remains a compatibility alias of K; its "next" means
  future verification, not a later worker invocation. `budget_target` states this
  explicitly. These are not executed-token measurements. `scheduled_runtime_mode`
  is `linear_dynamic_k` or `linear_fixed_k`; `executed_runtime_mode` remains `unverified`.
- `verification_feedback`: measured drafted/accepted counts at the existing
  verification callback, after invalid-draft adjustment. No latest-plan topology or
  plan sequence is attached: async feedback is not evidence of that plan executing.
  `source_plan_sequence` and `source_schedule_sequence` are explicitly null.
- Stream records carry `recorded_at`; step write errors are logged and do not change
  selected budgets or feedback updates. The stream is diagnostic and synchronous;
  no latency or complete-delivery guarantee is claimed.
- Canonical structure is copied from the existing catalog only when a named
  decision's node count agrees. Unknown topology identities retain name/count and
  null structure. Scalar K never synthesizes a topology identity.
- `engine_version`, `hardware_profile`, `actual_step_latency_ms` and
  `mean_accepted_tokens_tau` are null when unavailable. A theoretical speedup comes
  only from the decision, never from a fabricated measurement.
- `executed_draft_token_count` is null in planning/scheduling/final records and is
  populated only for individual verification callbacks. `tree_execution_verified`
  stays false. Scheduler observations alone never promote execution to verified.

No new topology representation, tree algorithm, mask, position-ID, accepted-path or
KV-cache semantics were introduced. `src/proposer/tree_adaptive.py`,
`src/flexsds/core/tree_topology.py` and `src/flexsds/core/cost_model.py` are unchanged.
Existing planner/glue test files were extended; the plan lists a potential new test
filename, but its DoD does not require that filename. Tests were not duplicated.

## Test evidence

Tests were added before their corresponding implementation:

- Initial planner contract run: **7 failed, 18 passed**, expected missing additions.
- Initial scheduled-telemetry contract: **1 failed, 16 passed**, missing observation hook.
- Legacy aggregate-preservation regression: **1 failed, 30 passed**, exposed planned
  `effective_alpha` overwriting the aggregate. Fixed by preserving original keys.
- Completion regression with `-k free_one_request`: **1 failed, 16 deselected**,
  exposed retained request IDs in `_plan`; fixed with request-local snapshot cleanup.

Passing verification commands (CPU only):

```powershell
python -m pytest -q tests/test_budget_planner.py tests/test_adaptive_scheduler_glue.py
# PASS: 49 passed

python -m pytest -q tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_cost_model_solver.py tests/test_cost_model_l4.py tests/test_budget_controller.py
# PASS: 78 passed

python -m pytest -q tests/test_memory_pressure_rule.py tests/test_modal_runner_config.py tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_m4_colab_preflight.py tests/test_protocol_b.py tests/test_protocol_b_boundary.py
# PASS: 284 passed, 31 subtests passed (109.57 seconds)
```

Coverage includes 2/4/0 heterogeneous requests, raw N above cap, equal counts with
distinct topology names, copied decisions/catalog arrays, alpha/history isolation,
completion cleanup, zero/empty budgets, the legacy floor/shared-scalar mode,
prefill exclusion, actual scheduled-subset lookup indexing, next/current-step
separation, JSON required fields, legacy compatibility and telemetry I/O failure.

- L1: pure planner CPU evidence.
- L2: source-informed scheduler stub/API-contract evidence. The stub now calls the
  update hook inside schedule and indexes by scheduled count, matching the pinned source.
- L3 installed pinned-vLLM smoke: **SKIPPED — vLLM not installed**. No model was loaded.
- No GPU benchmark, measured speedup, latency, race-freedom proof or tree execution
  verification is claimed. Full repository tests were not run.

## Unresolved boundaries

**PM DECISION REQUIRED:** A/B/C scope, arbitrary-tree runtime vs split vs reference
runtime, and resulting task-level acceptance remain pending.

**TECHNICAL REVIEW REQUIRED:** Do not enable tree-controller feedback on a scalar
runtime without reviewing the normalization contract. Minimal source reproducer:

```python
from src.proposer.tree_adaptive import TreeAdaptiveController
c = TreeAdaptiveController(initial_alpha=0.8)
c._make_decision('T8', 8, 0.8, None, 1.0, 'ok', 'source reproducer', 2.0, 3)
print(c.update(num_accepted=4, num_drafted=8))  # observed 0.8400000000000001
print(0.8 * 0.8 + 0.2 * (4 / 8))             # linear EMA: 0.7400000000000001
```

This reflects the controller's existing tree-depth normalization, not a proposed
algorithm change. The pre-commit audit below establishes that A=4 exceeds T8's
depth 3: it is a scalar-injection example, not a valid accepted tree path and not
proof that the intended tree yield metric is wrong. No upstream algorithm file
was edited. The default serving path still uses `AdaptiveDraftController`.

**TEAM TECHNICAL ALIGNMENT REQUIRED:** Khôi's trimming/padding-mask review remains
open. Existing placeholders are for next drafts; newly admitted requests and
preemption may differ from the pre-schedule running set. Current batch planning is
conservative over running decode requests, not a new per-worker execution contract.
Agree on nullability, consumer adoption of the optional step stream, hardware/runtime
identity measurement, and async plan-to-feedback lineage before claiming full v0.2
runtime telemetry conformance. Final snapshots alone are not a per-step trace.

Scientific claim boundary: **dynamic scalar K != arbitrary-tree runtime execution**.
Planner metadata can select T8/T16/T32; CPU/stub evidence does not show a GPU executing
those topologies. Actual latency, measured speedup and arbitrary-tree execution remain
UNKNOWN in this session.

## Review and delivery state

Self-review checked cap/floor behavior, scalar compatibility, request-owned mutable
state, topology identity, prefill exclusion, next/current-step timing, cleanup,
planned/aggregate field collisions, feedback provenance and pinned API assumptions.
At the end of the original implementation session, the diff was left uncommitted
for review and later logical splitting. The later local checkpoint section below
records the subsequent authorization and commits.

Original suggested split: planner contract tests; planner metadata transport; additive
telemetry/scheduler observations; compatibility/lifecycle regressions; this worklog.
After A: review proposer/verifier design against the pinned runtime before tree work.
After B: agree scalar acceptance and explicitly document the deferred runtime boundary.
After C: agree reference-runtime contract and connect these planner metadata snapshots.

Final verification after source self-review:

- `python -m pytest -q tests/test_budget_planner.py`: **PASS, 32 passed**.
- `python -m pytest -q tests/test_adaptive_scheduler_glue.py`: **PASS, 17 passed**.
- `git diff --check`: **PASS**, no whitespace errors.
- `git diff --stat`, `git diff -- src/server/adaptive_scheduler.py src/server/budget_planner.py`,
  and `git diff -- tests/test_budget_planner.py tests/test_adaptive_scheduler_glue.py`:
  inspected all tracked changes; 381 insertions, 23 deletions across four tracked files.
- `git status --short`: exactly those four modified files plus this new worklog.
- Optional Firecrawl quality-feedback submission was rejected by automatic approval
  review because it would send project-specific notes and session identifiers to an
  external service. It was not retried; implementation and source retrieval were unaffected.

## Pre-commit correctness audit — 2026-11-10

### Baseline and scope

Re-ran `git status --short`, `git branch --show-current`, `git rev-parse HEAD`,
`git rev-parse origin/develop`, `git log -1 --oneline HEAD`, `git diff --stat`,
`git diff --check` and inspected the complete tracked diff plus this untracked
worklog. Branch, HEAD and the local origin/develop ref still match the session
identity above. No fetch was required or performed during this audit. No local
changes were discarded. Five dirty files remain: two server modules, their two
existing test modules and this worklog. No algorithm, adapter, deployment or
experiment-protocol file was edited; no commit/push/PR/Trello action was taken.

Rechecked all unchanged producer/consumer modules named in the audit request.
`rg` found the newly introduced telemetry fields only in the server producer,
tests and this worklog; no production reader of those new fields was found.
Legacy readers, including `scripts/smoke_lane3_adaptive.py`, use the preserved
aggregate `tree_budgets` and `finished_at` fields. Direct adapter telemetry remains
unchanged; new normalization is at the server planner boundary.

### Exact v0.30.0 lifecycle

Read the pinned AsyncScheduler, Scheduler, dynamic lookup utility and
[GPUModelRunner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py)
from the tag. Relevant captured Scheduler/runner source agrees with the inspected
tag's statements. Full Scheduler and GPUModelRunner captured contents were also
compared with retrieved tag contents: both matched after newline/outer-whitespace
normalization. Captured Scheduler SHA256:
`c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884`.
No latest/main source was used.

1. Before `super().schedule`, project planning snapshots running decode requests
   and writes capped K to every valid lookup index.
2. Scheduler selects requests, constructs CURRENT verification's
   `scheduled_spec_decode_tokens` from existing request placeholders (scheduler.py
   lines 811–827), and clears those request placeholder lists.
3. Scheduler reads `lookup[len(num_scheduled_tokens)]` (1391–1396), puts K in this
   SchedulerOutput (1407–1427), then invokes the custom update hook (1454–1456).
4. Scheduler's base update advances computed/in-flight counts and recomputes
   `is_prefill_chunk` (1520–1546). AsyncScheduler then allocates K placeholders for
   FUTURE verification, skipping requests still in prefill. Project trimming only
   shrinks those request-owned future placeholder lists.
5. Project scheduling telemetry is emitted before the output is returned to the
   engine. It is an observation of scheduler selection, not GPU execution.
6. GPUModelRunner handles this output's current target verification/sampling, then
   its proposer pass consumes this output's K (runner 5034 and 5277–5289). It does
   not wait for another schedule call before consuming that K. Generated drafts
   are candidates for a later verification output, potentially already in flight.
7. `update_draft_token_ids_in_output` trims returned draft IDs to the already
   scheduled placeholder count and accounts for grammar-invalid slots (2425–2460).
8. On model output, Scheduler computes D_slots from current scheduled drafts and
   A=max(sampled_token_count−bonus_count,0) (1977–2002). Custom feedback uses
   D_valid=D_slots−invalid_count; the adapter forwards A and D_valid unchanged.
   Feedback may arrive after newer plan/schedule events. It cannot safely be
   joined to the latest topology. Request-final telemetry is emitted on free.

Thus "next" is correct for the verification placeholders, but insufficient for
describing proposer execution timing. Source comments and record documentation
were tightened; `scheduled_scalar_k` is the unambiguous current-output field.

### Minimal fixes and tests

- **Fixed: raw/effective alpha collision.** Scalar `select_budget` puts the
  bias-corrected value into `BudgetDecision.alpha_ema`; tree decisions put raw EMA
  there. The earlier normalizer mislabeled scalar effective alpha as raw EMA.
  After one 0/3 observation at initial alpha .85, actual raw EMA is .68 and effective
  alpha is 0. The new regression first failed with reported raw EMA 0, then passed
  after normalizing from controller state/getter. Existing decision types and
  adapter aggregate values are preserved.
- **Fixed: execution label inferred from scheduling.** Scheduling/final records now
  put the observed path in `scheduled_runtime_mode` and keep execution unverified.
  No actual executed topology is emitted. Feedback counts alone do not identify
  runtime mode or source topology.
- **Fixed: missing planning-to-scheduling identifiers.** `StepPlan.plan_sequence`
  travels into scheduled records; scheduler-local `schedule_sequence` distinguishes
  output events. A request completing prefill without a plan gets null plan ID.
  Feedback source IDs remain explicitly null rather than inheriting a newer plan.
- These audit regressions ran before their source fixes: **3 failed, 49 passed**.
- Stub lookup now executes the two lookup statements extracted via AST from the
  committed captured v0.30.0 scheduler. A contract test checks lookup→hook→return
  ordering and feedback/free parameter names against that source.
- An additional L2 test executes the captured upstream draft-update method on
  2/4/0 placeholders, then verifies valid counts reach the real scalar controller.
  It does not reimplement the upstream trim method in the fake.
- Other new checks cover delayed feedback, actual scalar output cap, prefill
  completion, and request-local cleanup even when final aggregate open fails and
  `delay_free_blocks=True` is forwarded to the base.

The stub still models admission, computed counts, preemption, KV blocks and GPU
work in simplified form. Its constructor is a test convenience, not vLLM's config
constructor. Passing it is L2 evidence only; it proves neither races nor GPU behavior.

### Feedback normalization: interpretation and unresolved finding

Let A=accepted draft tokens, D=valid drafted tokens, d=selected tree maximum depth,
beta=.8, and m=raw EMA. Scalar update uses
`x=clip(A/D,0,1); m'=beta*m+(1-beta)*x`. Tree update uses
`x=clip(A/d,0,1)` when d>0 and D>d, otherwise the scalar ratio. Both expose raw m
and bias correction `(m−m0*beta**n)/(1−beta**n)` but do not give those ratios the
same statistical meaning.

`logs/Khoi/spec&plan/treeAdaptiveController/review_alignment_design.md` §5.6
explicitly calls L/d an empirical yield index. Therefore the .84/.74 difference
alone is NOT a proof of a tree arithmetic bug. It also uses A=4>d=3, impossible
for a valid T8 accepted path. A valid example A=2,D=8,d=3 produces scalar EMA
.69 and tree EMA .7733333333333334; these are different normalizations by design.

**TECHNICAL REVIEW REQUIRED:** downstream semantic compatibility is not proven.
`get_effective_alpha` → `select_optimal_topology` → `evaluate_speedup` →
`expected_acceptance` feeds the empirical yield directly into the branch-coverage
probability parameter. Under the project's stated independent-coverage approximation,
two layers with two candidates each and p=.5 have per-layer coverage .75, hence
E[L]/d=(.75+.75²)/2=.65625. Using .65625 as p again predicts E[tau]=2.659470558166504,
where the model at the original p gives 2.3125. This is a calibration mismatch
under that approximation, not evidence that A/N is the correct replacement.

The [original OPT-Tree paper](https://arxiv.org/html/2406.17276v4), §3.2, works with
node/path probabilities approximated from draft-model scores; it does not establish
an identity between empirical L/d and a per-candidate match probability. No claim
that the project's scalar approximation reproduces the paper's full model is made.

The mutable `_last_selected_depth` also belongs to the latest selection, not
necessarily the async feedback's source selection. Fix criteria: agree the estimator
or topology-specific calibration and retain the source topology/depth with its
feedback before enabling that controller in scalar/asynchronous serving. Possible
approaches include an explicitly calibrated yield model or properly attributed
conditional acceptance counters; selecting one requires technical review/alignment.
No speculative formula patch was made.

This finding does not affect the default scalar-path diff: `from_env` constructs
the unchanged adapter default `AdaptiveDraftController`; a new test verifies that
exact class and the 4/8 update to .74. The scalar controller uses measured D_valid,
not latest planned K or tree depth. Explicitly injecting a tree controller remains
outside the verified scalar serving path.

### Telemetry classification inventory

| Fields | Class / source | Limitation |
| --- | --- | --- |
| request_id, lane, record_type | PLANNED/SCHEDULED/MEASURED record context | Lane is a serving label, not execution evidence |
| selected_topology.name/total_nodes_N/depth_d/parent_indices, planned_draft_node_count, batch_node_count | PLANNED; decision/catalog snapshot | Identity and N never imply executed tree |
| scalar_runtime_budget, batch_scalar_budget | PLANNED capped scalar budget | May differ from uncapped N; legacy floor can make K>0 when N=0 |
| concurrency_batch_b | PLANNED running-set size | Not necessarily the final scheduled count |
| acceptance_rate_ema, effective_alpha | PLANNED controller state at selection | Raw and bias-corrected yield signals, not universal probabilities |
| dynamic_alpha_star, theoretical_speedup, decision_reason | PLANNED decision output | Predicted only, nullable |
| memory_status.free_vram_mb/kv_cache_usage_pct | MEASURED signal snapshot, carried in plan | Missing/nonfinite readings become null; VRAM sentinel is not logged as measurement |
| memory_status.guard_triggered | PLANNED controller status | Null when scalar decision has no status field |
| engine_version, hardware_profile, actual_step_latency_ms, mean_accepted_tokens_tau | UNVERIFIED/null | No invented version/profile/timing/tau |
| plan_sequence | PLANNED, referenced by SCHEDULED | Planner-local, nullable for unplanned requests |
| schedule_sequence, scheduled_scalar_k, scheduled_runtime_mode, budget_target | SCHEDULED at update hook | K controls this output's proposer; no GPU claim |
| next_batch_scalar_budget, next_request_placeholder_budget | SCHEDULED compatibility K alias / future placeholder count | "Next" targets future verification; actual admission may delay it |
| executed_runtime_mode, tree_execution_verified | UNVERIFIED / false | False means not verified, not a proved absence of tree work |
| executed_draft_token_count, accepted_draft_token_count | MEASURED only at verification callback | Valid verified slots and accepted drafts, not total raw proposer work; local tests supply synthetic counters |
| source_plan_sequence, source_schedule_sequence | UNVERIFIED/null | No fabricated async lineage |
| alpha_ema in feedback | AGGREGATED controller EMA after measured callback | Not the latest planned signal |
| last_planned_decision, decision_scope, copied top-level decision fields | PLANNED last snapshot inside request aggregate | Not a per-step trace; absent/null if never planned |
| alpha_ema/effective_alpha/tree_budgets/mean_k/mean_accept_len/step_count in final | AGGREGATED legacy adapter fields | Original values preserved, planned/feedback histories differ |
| recorded_at, finished_at | MEASURED host emission timestamps | Not GPU-step latency or verification source IDs |

### Current test execution

```powershell
python -m pytest -q tests/test_budget_planner.py
# PASS: 34 passed
python -m pytest -q tests/test_adaptive_scheduler_glue.py
# PASS: 23 passed
python -m pytest -q tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_cost_model_solver.py tests/test_cost_model_l4.py tests/test_budget_controller.py
# PASS: 78 passed
python -m pytest -q tests/test_memory_pressure_rule.py tests/test_modal_runner_config.py tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_m4_colab_preflight.py tests/test_protocol_b.py tests/test_protocol_b_boundary.py
# PASS: 284 passed, 31 subtests passed (63.06 seconds)
```

Current total: **419 tests passed + 31 subtests**, not a reused count from the
original session. No tree feedback formula was changed. Final whitespace/diff
checks are recorded after documentation review.

### Integration boundary and local commit gate

`python -c "import importlib.util; print('vllm:', importlib.util.find_spec('vllm'))"`
again returned `vllm: None`. **REAL_VLLM_INTEGRATION = UNVERIFIED**. L3/L4 were
not run; GPU execution, speedup, latency, arbitrary-tree runtime, async source
lineage and full concurrent-engine correctness remain unverified.

Future existing smoke entry points, for an approved Linux/CUDA GPU environment
already containing vllm==0.30.0, the pinned target/draft weights, compatible AWQ
dependencies and an isolated disposable checkout/artifact tree on a dedicated
idle worker (the lane switcher stops other vLLM processes):

```bash
python -c 'import vllm; assert vllm.__version__ == "0.30.0"'
bash scripts/switch_lane.sh lane1
python scripts/smoke_lane3_adaptive.py reference --lane vanilla --per-workload 5 --max-tokens 256
bash scripts/switch_lane.sh lane3
python scripts/smoke_lane3_adaptive.py adaptive --per-workload 5 --max-tokens 256
```

These are future commands, not executed evidence. The legacy smoke writes fixed
artifact paths, so it must not be run over canonical historical evidence. It is
a functional/text-comparison smoke, not proof of arbitrary-tree behavior or a
complete validator of the new scheduling/feedback lineage fields.

Scope and shared trimming/padding-mask decisions remain pending. They do not
authorize a tree runtime claim. The scalar-path local commit gate is assessed
separately from task-level completion; no commit is performed by this audit.

Final audit gate: **SAFE TO COMMIT LOCALLY for the scalar-path implementation**.
Targeted tests and current related regressions passed; source changes have no
unresolved scalar correctness blocker found by this audit. Planned/scheduled/
measured semantics, state isolation and cleanup are covered at L1/L2, and the
unresolved tree estimator is demonstrated not to affect the default controller.
Real pinned-engine integration is explicitly UNVERIFIED. `git diff --check`
passed, all tracked source/test diffs were re-read, and protected
algorithm/adapter/deployment/protocol paths have an empty diff. Final status has
exactly the five intended dirty files. This gate does not mean task DONE, tree
support approved, or real engine behavior verified. All changes were still
uncommitted at the end of that audit; the following checkpoint session supersedes
that Git delivery state without changing the scientific boundary.

## Authorized local checkpointing — 2026-11-10

User authorization: create three coherent local commits for the audited scalar
path only. No push, PR, merge, rebase, Trello update, scope change, dependency
installation or GPU benchmark is authorized or performed in this session.

Before staging, verified branch
`feat/m4-02-04-scheduler-telemetry-v02`, starting HEAD
`9bc0f8d86e7a1f620a4f8cdaee79230b3bc41b14`, an empty index, exactly the five
audited dirty files, and the audited tracked diff of 592 insertions/26 deletions.
`git diff --check` passed. Re-read the source/test diffs and the untracked worklog.
No working-tree drift or unrelated file was found. Implementation was not changed
to fit the commit split. Only the historical delivery wording and this checkpoint
evidence were updated in the worklog.

Confirmed before staging: separate raw/effective alpha sources; scheduled mode
distinct from unverified executed mode; preserved plan/schedule IDs; current-output
proposer K distinct from future verification placeholders; unresolved tree
calibration retained; no arbitrary-tree execution or performance claim.

Local implementation commits:

1. `c52cd0ca2a4044fb78376e9100f3d718c38302f0` —
   `feat(m4-02-04): extend planner metadata and telemetry contract`.
   Files: `src/server/budget_planner.py`, `tests/test_budget_planner.py`.
   Purpose: decision identity/N metadata, independent scalar cap, additive
   telemetry, request isolation, legacy aggregates and correct alpha sources.
   Before commit: `python -m pytest -q tests/test_budget_planner.py` — **34 passed**;
   `git diff --cached --check` — **PASS**. Explicitly staged only these two paths.
   Post-commit `git status --short` and `git log --oneline --decorate -5`
   showed only scheduler source/tests and this worklog remaining dirty.
2. `94edea218aa2b48b4035dea4a87b7a7a4d6ceee6` —
   `feat(m4-02-04): integrate scalar adaptive budget with async scheduler`.
   Files: `src/server/adaptive_scheduler.py`, `tests/test_adaptive_scheduler_glue.py`.
   Purpose: pinned source-contract scheduling, observed K classification,
   plan/schedule IDs and lifecycle/feedback regressions, preserving trimming.
   Before commit: `python -m pytest -q tests/test_adaptive_scheduler_glue.py` —
   **23 passed**; `git diff --cached --check` — **PASS**. Explicitly staged only
   these two paths. Post-commit status/log showed only this untracked worklog.

This worklog is prepared as the third checkpoint with message
`docs(m4-02-04): record scalar scheduler evidence and open boundaries`.
It records L1/L2 evidence and preserves the open technical/PM/team boundaries;
it does not change task acceptance or claim runtime verification.

Additional CPU validation during checkpointing:

```powershell
python -m pytest -q tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_cost_model_solver.py tests/test_cost_model_l4.py tests/test_budget_controller.py
# PASS: 78 passed
python -m pytest -q tests/test_memory_pressure_rule.py tests/test_modal_runner_config.py tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_m4_colab_preflight.py tests/test_protocol_b.py tests/test_protocol_b_boundary.py
# PASS: 284 passed, 31 subtests passed (67.54 seconds)
```

Scientific ceiling remains: **dynamic scalar K != arbitrary-tree runtime execution**.

```text
REAL_VLLM_INTEGRATION = UNVERIFIED
GPU_RUNTIME = UNVERIFIED
ARBITRARY_TREE_RUNTIME = UNVERIFIED
MEASURED_SPEEDUP = UNVERIFIED
ASYNC_FEEDBACK_SOURCE_LINEAGE = UNVERIFIED
TREE_YIELD_CALIBRATION = TECHNICAL REVIEW REQUIRED
```

The calibration finding remains unresolved; no tree algorithm was modified.
PM Option A/B/C and shared trimming/padding-mask alignment remain pending.


## L3 real scalar integration smoke — 2026-11-10

Result: **L3 SCALAR PATH VERIFIED** for the bounded three-request functional
smoke. This entry does not mark M4-02-04 DONE.

- Branch `feat/m4-02-04-scheduler-telemetry-v02`; exact tested SHA
  `a6cd6ad5cee5374738f890c270ada4ca01f0946b`. Worktree clean before evidence
  creation. All 63 mounted worker Python source files match the committed Git
  blobs, packaged with `experiments.bench_source.snapshot`.
- Dedicated isolated Modal A10G function, actual GPU NVIDIA A10, Ubuntu 22.04.5,
  driver 610.57.04, CUDA UMD 13.3 / torch CUDA 13.0, Python 3.11.5,
  torch 2.13.0+cu130, **vllm 0.30.0**. Existing pinned image package layer reused.
  Local Windows/torch 2.6.0+cu124/no-vLLM failed the gate and did not serve.
  The L4 probe had no runners; cancellation was requested only for that probe. Existing
  deployed apps were not changed or stopped.
- Exact target `Qwen/Qwen2.5-7B-Instruct-AWQ` revision
  `b25037543e9394b818fdfca67ab2a00ecc7dd641`; draft
  `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Canonical Modal adaptive/l4-bf16 argv: requested awq_marlin/bfloat16,
  draft_model/greedy, K ceiling 4, dynamic lookup [[1,32,4]], heterogeneous
  vocabulary, AdaptiveAsyncScheduler, model length 4096, max sequences 32,
  GPU memory utilization 0.85, generation config vllm, metrics summary.
  Default AdaptiveDraftController; VRAM guard 1, max K 4, min batch K 0,
  per-request 1, empty controller kwargs. Telemetry paths point into the
  isolated worker /tmp/m40204-smoke; step telemetry additionally enabled.
- Harness imports the committed project image and mounts complete committed
  src/experiments trees for project imports. No scheduler/controller/runtime
  methods were monkeypatched. See deployment.json for exact argv/environment.
- Commands: set PYTHONIOENCODING=utf-8, PTIT_GPU=A10G,
  PTIT_DTYPE_PROFILE=l4-bf16, PTIT_LANES=adaptive; then
  `python -m modal run experiments/runs/m4_02_04/20261110-scalar-vllm030-smoke/smoke_runner.py::main`.
  Validate saved evidence with
  `python experiments/runs/m4_02_04/20261110-scalar-vllm030-smoke/validate_evidence.py`.
- Phase A PASS: installed-source imports/signatures verified, custom scheduler
  selected, target/draft initialize, health HTTP 200.
- Phase B PASS: one deterministic small request HTTP 200/nonempty output.
- Phase C PASS: 36 plans matched to SchedulerOutput K observations; 39 scheduled
  records; naturally observed K={0,2,3,4}. Single request progressed K=3 to K=4
  under acceptance feedback, with 37 drafted/37 accepted. Overall real callback
  counters 83 drafted/72 accepted in 25 feedback events. No thresholds forced.
- Phase D PASS for minimal isolation smoke: two concurrent clients completed,
  distinct request histories, independently observed scheduler running batches
  1 and 2. Placeholder trimming observed (shared K=2, one request budget=0).
  Three completion aggregates are emitted after committed end_request removes
  session/history; no subsequent steps for a completed request. No live heap
  inspection or extended concurrency claim. Server alive after all requests.
- Telemetry preserves separate planned/scheduled/measured fields and
  tree_execution_verified=false throughout. planned topology null on default
  scalar path. Aggregate executed count and executed_runtime_mode remain
  unverified; actual counters are in step feedback.
- ASYNC_FEEDBACK_SOURCE_LINEAGE=UNVERIFIED: feedback source plan/schedule IDs
  remain null. **TEAM TECHNICAL ALIGNMENT REQUIRED**. Tree calibration remains
  **TECHNICAL REVIEW REQUIRED**. No arbitrary-tree or performance claim.
- Anomalies: Windows CLI encoding/network/live-log tooling issues; first worker
  harness assertion used the wrong default controller name and stopped before
  launch (preserved root error.log). Corrected harness assertion only, reran in
  attempt2. Draft safetensors offline-discovery errors recovered with cached
  weights. During deliberate teardown, process-manager force-kill and one
  leaked-semaphore warning; cause UNKNOWN, no serving scheduler exception.
- Successful app `ap-g2Y8ylml1O4z0pl2AUuWoh`, call
  `fc-01M72GTCBKZ3SM3G0BWS4DCQBK`; isolated app completed and server terminated.
  Unused queued L4 version call `fc-01M72GJCACZFS7W0PEFCMQZARJ`: cancellation requested twice (including terminate_containers=True); local waiting caller stopped. Final API state still pending=1, cancelled=0, runners=0. Remote cancellation UNCONFIRMED; no active containers. No shared deployed app was stopped.
- Raw evidence root:
  `experiments/runs/m4_02_04/20261110-scalar-vllm030-smoke/`.
  README.md explains attempts/limitations; manifest.json indexes all artifacts;
  attempt2 contains environment, installed scheduler sources, exact deployment,
  server log, three requests/responses, raw steps/request telemetry, source
  hashes, call/result and validation.json. Failed attempts retained.

No production source or compatibility change, commit, push, merge, PR, Trello
change, frozen-protocol edit, benchmark, or arbitrary-tree implementation.
Evidence/worklog are uncommitted for review. Next action: team alignment on
exact async feedback source lineage identifiers and callback contract.


## Evidence-preserving checkpoint and PM closure readiness — 2026-11-10

Status: **READY FOR PM OPTION-B CLOSURE DECISION**. M4-02-04 remains open;
Option B is a technical recommendation, not PM approval or a changed DoD.

### VERIFIED NOW

- Planner/controller L1/L2: previously recorded 419 pytest tests and 31 subtests.
- L3 real vLLM 0.30.0 scalar scheduling on NVIDIA A10, not a performance benchmark.
- Dynamic scalar K={0,2,3,4}; 36 planner/SchedulerOutput matches; 83 drafted,
  72 accepted; three requests, including two concurrent clients.
- Tested request isolation, placeholder trimming, telemetry and completion hooks.
- Raw reproducibility metadata and failed-attempt evidence retained. All 29
  historical smoke manifest entries and 465 hash entries across the research
  manifests were rechecked successfully before curation. The saved validator
  reproduced validation.json in memory without overwriting it; all 63 worker
  source hashes match exact tested Git blobs.

### STILL UNVERIFIED

- Arbitrary-tree runtime, including T8/T16/T32 masks, positions and accepted paths.
- Tree calibration correctness and exact async source-plan lineage.
- Performance improvement and canonical T4/L4 benchmark results.
- Previous queued L4 probe cancellation remains unconfirmed in the saved
  control-plane record; this checkpoint made no remote changes.

`dynamic scalar K != arbitrary-tree runtime execution`.

### PM DECISION STILL REQUIRED

- Approve or reject Option B split scope and evidence-preserving integration.
- Decide arbitrary-tree follow-up timing relative to M4 freeze.
- Assign the follow-up owner and deadline.
- Ratify final DoD and required arbitrary-tree execution claims if scope changes.

### TECHNICAL ALIGNMENT

- Host-side lineage is source-level feasible only for capability-checked V1,
  UniProc, single worker, same process, scalar path. **NOT IMPLEMENTED — ALIGNMENT FIRST**.
- Ratify actual-consumed-slot origin semantics and ownership of the
  compatibility-sensitive executor/scheduler adapter with Đại and Khôi.
- Agree L2/L3 observation-window, lifecycle and negative-case validation owners.
- Review tree calibration separately; lineage does not prove tree execution.

### CHECKPOINT POLICY

- Evidence-only selection: smoke checkpoint_manifest.json and its explicit files,
  scoped ignore/byte-preservation rules, and this worklog.
- Separate design selection: three runbooks, small reproduction scripts,
  installed/test source hash inventories and CPU inspection metadata.
- Full upstream/installed source captures, original historical smoke manifest,
  bytecode and superseded staging/review snapshots remain local and excluded.
- Raw evidence is immutable. No raw artifact or failed attempt was deleted or
  rewritten. Portable report references use repository-relative paths.
- A read-only secret scan and manual request/deployment review found no credential
  values. Raw worker/preflight paths and app/call/volume IDs are provenance;
  no sensitive authentication headers, cookies or signed URLs were found.
- Production/test source remains at the three verified implementation commits;
  no new engine run, lineage implementation or benchmark was performed.

## 2026-11-11 — PM-approved Option B integration validation

Baseline: clean `feat/m4-02-04-scheduler-telemetry-v02` at
`b4c13637149b14b3ebab20412b044639b7e4c40f`. Ending HEAD is unchanged.
The implementation inventory was reported before edits. Only L2 tests and this
worklog were changed; no production fix, commit or external publication.

### PM requirement inventory

| Requirement | Production path and evidence | Status |
| --- | --- | --- |
| hardware/runtime-aware threshold | Scheduler passes running count, KV fraction and optional CUDA free memory to planner. Planner converts KV to percent and VRAM to controller input. Default adapter instantiates `AdaptiveDraftController`, whose scalar rule ignores batch size and hardware cost profile. Runtime memory guards are integrated; hardware-derived alpha-star is not. | PARTIAL |
| adaptive K | `AdaptiveDraftController.select_budget` -> `AdaptiveBudgetPlanner.plan_step` -> `AdaptiveAsyncScheduler.schedule`; L1 rules, L2 natural transition, existing L3 K=0/2/3/4. | IMPLEMENTED_AND_INTEGRATION_TESTED |
| per-request budget | Adapter sessions/history and planner detached decisions preserve K_i while shared K=max(K_i); L1 planner/adapter, L2 heterogeneous lifecycle, L3 concurrent requests. | IMPLEMENTED_AND_INTEGRATION_TESTED |
| AsyncScheduler integration | Lookup -> SchedulerOutput -> placeholders -> pinned draft update -> stats callback; L2 captured v0.30.0 methods and existing installed-engine L3. | IMPLEMENTED_AND_INTEGRATION_TESTED |
| trimming/isolation | `_update_after_schedule` shrinks future placeholders; pinned `update_draft_token_ids_in_output` trims incoming draft IDs. L2 checks 2/4/0, feedback and cleanup; L3 observes shared K=2/request K=0. | IMPLEMENTED_AND_INTEGRATION_TESTED |
| telemetry planned/scheduled/measured | Planner `plan_step`, `record_scheduled_budget`, `record_verification`, `finish_request`; L1 field semantics, L2 complete request lifecycle, L3 saved JSONL. | IMPLEMENTED_AND_INTEGRATION_TESTED |
| real pinned-vLLM smoke evidence | Existing A10/Ubuntu/vLLM 0.30.0 attempt2 revalidated locally; no new GPU run. | IMPLEMENTED_AND_INTEGRATION_TESTED |

### Threshold trace and technical review boundary

- Raw acceptance is accepted/drafted after invalid placeholders are subtracted
  by the scheduler callback. EMA beta defaults to 0.8. Effective alpha removes
  the initial prior using `(ema - initial_alpha * beta**n)/(1-beta**n)`, clamped
  to [0,1], with a small-denominator fallback.
- Serving acceptance thresholds are configured defaults 0.65/0.80/0.90;
  below 0.65 selects K=0 (periodic probes), [0.65,0.80) selects K=2,
  [0.80,0.90) selects K=3, and >=0.90 selects K=4.
- Runtime KV fraction is multiplied by 100; >90% triggers K=0. CUDA bytes are
  divided by 1024**2 (MiB despite the historical `_mb` name); <1500 triggers
  K=0. Equality at the high boundary is safe. After cooldown, KV must be
  strictly <80% to recover. Missing CUDA readings use the planner sentinel,
  so they do not trigger the VRAM guard; KV still applies.
- Batch input is the number of running requests, including prefills, rather
  than client concurrency or the final scheduled subset. The scalar controller
  accepts it but does not use it to select K. No hardware profile is loaded by
  `AdaptiveBudgetPlanner.from_env` or canonical lane configuration.
- `src/proposer/tree_adaptive.py` has batch-aware topology selection and
  `FlexCostModel.solve_alpha_star` solves expected yield = C_spec/C_vanilla
  (both costs in ms, ratio dimensionless). It is not the serving controller.
  `src/flexsds/core/controller.py` also contains a separate hardware-aware
  scalar policy, but it is not wired into canonical serving either.
- **TECHNICAL REVIEW REQUIRED**: the mandatory hardware-aware threshold serving
  gate is unmet. Do not activate a different controller or change its formula
  under the guise of test coverage. Tree yield/probability calibration remains
  deferred. Other scalar integration work can proceed independently.

### L1 review and L2 changes

Existing `test_budget_controller.py` covers rules/EMA, `test_memory_pressure_rule.py`
covers guard hysteresis, `test_budget_planner.py` covers budget aggregation,
caps and telemetry, `test_proposer_adapter.py` covers session feedback/cleanup,
and `test_tree_adaptive_controller.py`/`test_cost_model_solver.py` cover separate
policy/model behavior. Those tests do not prove hardware policy is served.
No duplicate unit cases or production changes were added.

Extended `tests/test_adaptive_scheduler_glue.py`:

- Strengthened `test_trimmed_placeholders_reach_pinned_draft_update_and_feedback`
  into one heterogeneous 2/4/0 lifecycle: actual draft structures, independent
  EMA/windows/history, K=0 completion while others stay active, next-step
  adaptation, all planner/adapter/scheduler-owned request maps cleared, and
  planned/scheduled/measured/aggregate telemetry consistency.
- Added `test_feedback_drives_scheduler_k_transition`: default thresholds,
  real feedback ratios 2/3, 2/2, 2/2, effective alpha 2/3, 23/27, 167/183,
  and scheduler-visible future K=2,2,3,4. Outstanding current drafts remain
  separate from the new proposer K; no exact async source lineage is inferred.
- Added `test_runtime_memory_boundary_fallback_and_recovery` for KV and CUDA
  free-memory inputs through planner units, guard, output K and placeholders.

These are **L2**, using the existing controlled scheduler harness and real
project classes. Upstream lookup and draft trimming execute captured source;
the harness does not execute GPU workers, target verification, token generation
or engine stop detection. Existing L3 supplies real functional serving evidence.

### Current verification results

- Baseline: `python -m pytest -q tests/test_budget_planner.py tests/test_adaptive_scheduler_glue.py tests/test_proposer_adapter.py tests/test_budget_controller.py`
  -> **84 passed**.
- Baseline related suites: `python -m pytest -q tests/test_tree_adaptive_controller.py tests/test_cost_model_solver.py tests/test_memory_pressure_rule.py tests/test_budget_transition_scenarios.py tests/test_budget_fallback_recovery.py -k 'not benchmark'`
  -> **58 passed, 1 deselected**. The timing benchmark was intentionally excluded.
- Updated L2: `python -m pytest -q tests/test_adaptive_scheduler_glue.py`
  -> **26 passed**.
- Combined: `python -m pytest -q tests/test_budget_planner.py tests/test_adaptive_scheduler_glue.py tests/test_proposer_adapter.py tests/test_budget_controller.py tests/test_tree_adaptive_controller.py tests/test_cost_model_solver.py tests/test_memory_pressure_rule.py tests/test_budget_transition_scenarios.py tests/test_budget_fallback_recovery.py tests/test_modal_runner_config.py -k 'not benchmark'`
  -> **195 passed, 1 deselected**.
- `python experiments/runs/m4_02_04/20261110-scalar-vllm030-smoke/validate_evidence.py`
  -> **L3 SCALAR PATH VERIFIED**: 36 plans, 39 scheduled records, 25 feedback
  records, 3 aggregates, 36 plan/schedule joins, K=0/2/3/4, 83 drafted/72 accepted.
- Read-only hash checks: all **63** worker source files match current HEAD;
  **28** non-cache manifest artifacts match SHA-256. AST comparisons of the L2
  fixture and installed L3 captures match for `schedule`,
  `update_draft_token_ids_in_output`, `make_spec_decoding_stats`, `_free_request`.

### Verdict and claim boundary

**NO PRODUCTION FIX REQUIRED — INTEGRATION COVERAGE ADDED** for the existing
scalar path. **STILL VALID** for the existing L3 smoke, limited to its three
requests and recorded environment; production behavior is unchanged.

**NOT READY FOR OPTION-B CLOSURE REVIEW**: hardware-derived threshold is not on
the serving path. The validated behavior is runtime-memory-aware, feedback-driven
scalar K; this must not be expanded into a hardware-cost-aware serving claim.
Exact async source-plan/schedule lineage stays null/unverified by design.
Tree policy/catalog code is kept as DORMANT_TREE_POLICY/RESEARCH_REFERENCE,
not runtime-executed topology. No arbitrary T8/T16/T32 runtime execution or
performance improvement is established.

First action: technical review to resolve the missing hardware-aware scalar
threshold serving connection within Option B.

### 2026-11-11 — corrected scalar context source/L2 proof

Measurement-only follow-up; production behavior and earlier Option-B blockers
are unchanged. Derived the pinned vLLM0.30.0 GPU recurrence from preserved
installed sources: participating rows use previous GPU context plus previous
raw valid sampled count; other rows use the actual preparation's host copy.
Positions and verification sequence length follow corrected C and scheduled Q.
Raw CPU sampled counts are frozen after canonical materialization and before
scheduler EOS/max_tokens clipping. They are distinct from useful emitted tokens.

Added frozen CPU reconstruction snapshots and versioned observer evidence,
execution/incarnation/dependency binding, exact fixture-context matching, and
fail-closed lifecycle/history guards. Compared the unchanged pinned correction
body on CPU tensors over 512 combinations, plus stop/clipping, row reorder,
heterogeneous/invalid slots, K changes, and two-in-flight integration scenarios.
The helper adds no GPU readback, wait, event synchronization, or future access.

Validation: `python -m pytest -q tests/test_scalar_context.py tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py tests/test_m4_cost_profiler.py tests/test_m4_cost_runner.py tests/test_modal_profiler.py`
→ **230 passed in 17.10s**. `git diff --check` passed; production/config/script
diff remains empty. Prior CPU overhead reports predate this observer extension.

Verdict: **CORRECTED CONTEXT RECONSTRUCTION VERIFIED**, guarded source/L2 scope.
Live worker snapshot transport remains unverified; no L3 collector fidelity,
matched reference, profitability, or performance improvement is claimed.
See `docs/runbooks/m4-02-04-corrected-context.md` for source hashes, tensor/D2H
lifetimes, formulas, matching contract, and prepared tiny L4 smoke specification.
Small smoke: **NOT RUN — AWAITING REVIEW**. Full grid: **NOT RUN**.
Production source: **UNCHANGED**. No commit, push, PR, merge, or Trello change.
