# Hardware-Adaptive replacement policy CPU validation

```text
BASELINE:
8006205a6f19790a776d67155a377def053d2736

WORKING BRANCH:
audit/flexsds-v023-hardwareadaptive-20261120

SOURCE BEFORE PATCH:
4c76830706d6d60d715d962764e592b7b184fc42

SOURCE AFTER PATCH:
4c76830706d6d60d715d962764e592b7b184fc42 + uncommitted worktree patch
Source manifest SHA-256 1c34c8bbbc829c13e362138b6733b74f410292247848279f1d06c6148e7e1a17
Test manifest SHA-256   0fec335247df6336cd4a8c883140b13234f3bd5c5ae789f50aeba6076fbaaece
```

`git branch --show-current` and `git rev-parse HEAD` returned the branch and SHA above before and after the work. `git merge-base HEAD 8006205` returned the baseline SHA. The initial `git status --short` showed pre-existing untracked material and zero tracked edits. The six named baseline implementation files matched `8006205` before editing. The frozen audit evidence resolves to `1eb7e594f8f0afa2fc72df8c571ee13cacfa8e06`. The patch changes five production files and leaves the baseline executor implementation intact.

## Design delta

| File/function | Change | Spec invariant |
|---|---|---|
| `controller.py` · `AdaptiveDraftController` | A dedicated attributed-verification counter gates warmup. A decision captures the effective alpha, rule alpha, gate state and memory state together. Reset restores the initial EMA and clears policy state. | I1–I6, I13 |
| `controller.py` · acceptance and memory rules | Positive `min_k` rounds up to the least supported candidate at or above the configured floor and fails when none exists. The floor runs only on low acceptance after the memory gate and before probing. Memory and cooldown leave probe cadence untouched. | I4–I6 |
| `adapter.py` · feedback and telemetry | The attributed update path preserves existing accepted-history limits and reports the sample count plus a fixed-category lifetime reason histogram. | I1–I3, I13, I16 |
| `budget_planner.py` · `plan_step`, recovery and session binding | Runtime clipping rounds down to the greatest supported candidate at or below the scalar decision and runtime limit, falling back to zero. An increase-only cap bounds each eligible row. Empty and disjoint cohorts clear it. Request incarnation changes retire old session state. | I4–I13, I16 |
| `adaptive_scheduler.py` · schedule, cleanup and verification | The pinned post-schedule hook applies the new policy after vLLM settles prefill status. It sets future row placeholders and the output's current proposer K independently of the older drafts this output verifies. Stale, finished and ineligible rows get no new draft allowance. | I4, I7–I14, I16 |
| `lineage.py` · draft birth and claim | A shape-validated birth supplies a weakly retained origin witness. A positive-width consumed origin can be claimed once while it remains in flight, independently of callback order. | I1–I3, I13–I14, I16 |
| `budget_planner.py` · policy records | Decision, batch, physical proposal and verification records join on plan, schedule and request incarnation. Reason counts span the request lifetime even when decision history is bounded. | I6–I8, I14, I16 |
| `tests/test_replacement_policy.py`, `tests/test_adaptive_scheduler_glue.py` | All 37 required IDs plus measured-profile coverage. The existing stub uses the final output K for its simulated worker, matching the pinned dispatch boundary. Existing assertions were preserved. | I1–I16 |

The replacement serving path is enabled with `ADAPTIVE_REPLACEMENT_POLICY=1`. Use `ADAPTIVE_CONTROLLER_KWARGS` for `min_warmup_samples` and `min_k`, and `ADAPTIVE_RECOVERY_HOLD_STEPS` for the upward hold. Each of these values defaults to zero. The replacement path requires per-request mode and never raises a budget using `ADAPTIVE_MIN_BATCH_K`. Save step records through `ADAPTIVE_STEP_TELEMETRY_PATH`. The source-level flag preserves existing launcher behavior and leaves the historical experiment configuration intact.

Shared padded drafting can compute beyond a low-K row's own allowance when a peer needs more drafts. The physical-width record exposes that shared width and the affected row's applied K. The verifier consumes only the producer's allowed row prefix. This distinction also applies to zero-K rows in a mixed batch.

## Red-phase evidence

Before production changes, all 47 cases covering the required IDs ran against exported source at `8006205` and failed as expected. The final suite strengthened several assertions and added one profile preservation case. Its repeat against the same frozen source failed all 48.

| Test ID | Baseline result | Why |
|---|---|---|
| T-WARM-001 | 1 failed | Warmup configuration and decision fields absent |
| T-WARM-002 | 1 failed | Warmup configuration and count absent |
| T-WARM-003 | 1 failed | No attributed sample counter |
| T-WARM-004 | 3 failed | No attributed sample counter |
| T-WARM-005 | 1 failed | No minimum sample configuration |
| T-WARM-006 | 1 failed | No independent rule alpha |
| T-WARM-007 | 1 failed | No replacement count/reset |
| T-FLOOR-001 | 1 failed | No acceptance-floor configuration |
| T-FLOOR-002 | 2 failed | No acceptance-floor configuration |
| T-FLOOR-003 | 1 failed | No acceptance-floor configuration |
| T-FLOOR-004 | 1 failed | No floor normalization |
| T-FLOOR-005 | 1 failed | No floor contract parameter |
| T-FLOOR-006 | 7 failed | No replacement controller parameters |
| T-HOLD-001 | 1 failed | No replacement planner or recovery cap |
| T-HOLD-002 | 1 failed | No recovery-cap contract |
| T-HOLD-003 | 1 failed | No replacement zero/hold contract |
| T-HOLD-004 | 1 failed | No replacement row-cap contract |
| T-HOLD-005 | 1 failed | No replacement row-cap contract |
| T-HOLD-006 | 1 failed | No cohort-aware recovery state |
| T-HOLD-007 | 1 failed | No pending target/streak |
| T-HOLD-008 | 1 failed | No pending target/streak |
| T-LIFE-001 | 1 failed | No replacement planner or lifetime histogram |
| T-LIFE-002 | 1 failed | No explicit recovery reset |
| T-LIFE-003 | 1 failed | No replacement sample telemetry |
| T-LIFE-004 | 2 failed | No replacement incarnation-scoped state |
| T-ACT-001 | 1 failed | Unscheduled high-K row raised worker K to 4 |
| T-ACT-002 | 1 failed | Missing physical-width and warmup records |
| T-ACT-003 | 1 failed | No batch-policy record |
| T-ACT-004 | 1 failed | No optional floor configuration or policy origin fields |
| T-ACT-005 | 1 failed | Baseline resumed drafting immediately after memory cooldown |
| T-ACT-006 | 1 failed | Missing policy applied-K and producer joins |
| T-ACT-007 | 1 failed | Missing policy producer IDs |
| T-ACT-008 | 1 failed | No attributed sample counter |
| T-TEL-001 | 1 failed | Missing warmup/decision fields |
| T-TEL-002 | 1 failed | No batch-policy record |
| T-TEL-003 | 1 failed | No replacement producer/incarnation/count fields |
| T-TEL-004 | 1 failed | No selected/applied/worker decision fields |
| Profile preservation | 1 failed | Baseline controller does not accept the new floor config |

## Green-phase evidence

| Test ID | Layer | Patched result | Evidence |
|---|---|---|---|
| T-WARM-001 | L1 | 1 passed | Two zero-width calls, a direct positive-width estimator call, and repeated selections leave attributed count at zero. |
| T-WARM-002 | L3 | 1 passed | Attributed `(drafted=3, accepted=0)` increments exactly once. |
| T-WARM-003 | L3 | 1 passed | The same producer offered through distinct verifier outputs updates once and logs `duplicate-origin`. |
| T-WARM-004 | L3 | 3 passed | Stale, wrong-incarnation and forged producer events leave EMA and count unchanged. Direct planner calls cannot provide replacement evidence. |
| T-WARM-005 | L1 | 1 passed | Gate stays active at two and releases for the next decision at three. |
| T-WARM-006 | L1 | 1 passed | Decision exposes effective alpha near zero, rule alpha .80, and K=3. |
| T-WARM-007 | L1 | 1 passed | Explicit reset clears sample, fallback and reason state and restores the initial EMA. |
| T-FLOOR-001 | L1 | 1 passed | Low acceptance uses K=2 without advancing probe cadence. |
| T-FLOOR-002 | L1 | 2 passed | KV pressure and low free VRAM individually override the floor and warmup. The safety decision has no rule alpha. |
| T-FLOOR-003 | L1 | 1 passed | The floor resumes only after memory cooldown ends. |
| T-FLOOR-004 | L1/L2 | 1 passed | Floor 1 normalizes to 2. An impossible floor raises. A runtime maximum of 1 does not manufacture unsupported K=1. |
| T-FLOOR-005 | L1 | 1 passed | Memory and cooldown decisions leave acceptance cadence unchanged until normal decisions resume. |
| T-FLOOR-006 | L1 | 7 passed | Seven exact alpha thresholds × warmup × floor × memory × probe due cover 112 combinations. |
| T-HOLD-001 | L2 | 1 passed | Three identical K=4 proposals after K=2 apply 2, 2, 4. |
| T-HOLD-002 | L2 | 1 passed | 4→2 applies immediately. |
| T-HOLD-003 | L2 | 1 passed | All-zero selections clear cap and pending state despite a positive legacy batch floor. |
| T-HOLD-004 | L2 | 1 passed | Fresh `(0,2)` applies `(0,2)` with worker 2. Replacement uniform mode is rejected. |
| T-HOLD-005 | L2 | 1 passed | Exhaustive 16 pairs × four caps × memory on/off cover 128 combinations. |
| T-HOLD-006 | L2 | 1 passed | Empty and fully disjoint request cohorts cannot inherit the previous cap. |
| T-HOLD-007 | L2 | 1 passed | Alternating positive targets restart the streak. |
| T-HOLD-008 | L2 | 1 passed | Zero interrupts a pending increase and recovery earns a fresh streak. |
| T-LIFE-001 | L2 | 1 passed | Finish drops the controller. Reuse starts with original EMA, zero samples and an empty histogram. |
| T-LIFE-002 | L2 | 1 passed | Empty and explicit resets clear cap and pending state. |
| T-LIFE-003 | L3 | 1 passed | Preempted feedback leaves its controller untouched while the peer counts. The stale row gets no future draft placeholder or origin. |
| T-LIFE-004 | L3 | 2 passed | ID reuse with and without prior finish clears state, rejects old feedback and prevents late cleanup from affecting the replacement. |
| T-ACT-001 | L3 | 1 passed | Only actual eligible scheduled rows set worker width. Prefill-only and explicitly disabled proposals use zero. |
| T-ACT-002 | L3 | 1 passed | Mixed `(0,2)` produces shared width 2 and verifies only the width-2 row, which receives feedback. |
| T-ACT-003 | L3 | 1 passed | 4→2 is immediate at the published worker output and in batch telemetry. |
| T-ACT-004 | L3 | 1 passed | A pressure-driven zero produces no new drafts. Older width-4 feedback names its own producer. |
| T-ACT-005 | L3 | 1 passed | The cleared memory guard can select positive K while the upward cap still publishes zero. |
| T-ACT-006 | L3 | 1 passed | Reordered rows retain their own origin widths and update the correct request. |
| T-ACT-007 | L3 | 1 passed | Two in-flight verifiers can retire out of order with exact producer K and plan/schedule IDs. |
| T-ACT-008 | L3 | 1 passed | The repeated callback emits one valid record and one skip with one EMA update. |
| T-TEL-001 | L3 | 1 passed | Decision records contain incarnation, plan, selected/applied K, both alphas, sample count, warmup, reason and memory state. |
| T-TEL-002 | L3 | 1 passed | Batch record identifies the exact fresh max, cap, worker K, eligible size and pending streak. |
| T-TEL-003 | L3 | 1 passed | Feedback distinguishes invalid suffix, valid drafted width, accepted width and counted warmup sample. |
| T-TEL-004 | L3 | 1 passed | One-entry retained history and two lifetime reasons remain distinct. Width streams are separately named. |
| Profile preservation | L2 | 1 passed | Real profile validator loads a deterministic measured fixture as EmpiricalCostProfile. Candidate grid, lookup and bounded history remain usable. |

## Regression table

| Existing suite | Baseline | Patched passed | Failed | Skipped | Regression? |
|---|---:|---:|---:|---:|---|
| `test_budget_controller.py` | 19 passed | 19 | 0 | 0 | No |
| `test_budget_fallback_recovery.py` | 8 passed | 8 | 0 | 0 | No |
| `test_budget_transition_scenarios.py` | 6 passed | 6 | 0 | 0 | No |
| `test_memory_pressure_rule.py` | 13 passed | 13 | 0 | 0 | No |
| `test_proposer_adapter.py` | 8 passed | 8 | 0 | 0 | No |
| `test_budget_planner.py` | 35 passed | 35 | 0 | 0 | No |
| `test_adaptive_scheduler_glue.py` | 67 passed | 67 | 0 | 0 | No |
| `test_lineage_startup.py` | 27 passed | 27 | 0 | 0 | No |
| `test_m4_pinned_contract.py` | Frozen source manifest | 1 | 0 | 0 | No source-contract mismatch |
| `test_replacement_policy.py` | 48 expected failures | 48 | 0 | 0 | New contract |
| **Final patched total** | | **232** | **0** | **0** | |

The confirmed environment used Python 3.11.0, pytest 9.1.1, PyTorch 2.6.0+cu124, NumPy 2.3.4 and psutil 7.1.2. The vLLM source fixtures are 0.30.0. CPU tests extracted from pinned source and used simulated dispatch and CPU tensors where applicable. Commands ran in temporary exports with pytest plugin autoload disabled and source resolved from each export.

```text
python -B -m pytest -p no:cacheprovider --basetemp <scratch> -q --tb=short -rA tests/test_replacement_policy.py
python -B -m pytest -p no:cacheprovider --basetemp <scratch> -q --tb=short -rA tests/test_budget_controller.py tests/test_budget_fallback_recovery.py tests/test_budget_transition_scenarios.py tests/test_memory_pressure_rule.py tests/test_proposer_adapter.py tests/test_budget_planner.py tests/test_adaptive_scheduler_glue.py tests/test_lineage_startup.py
python -B -m pytest -p no:cacheprovider --basetemp <scratch> -q --tb=short -rA tests/test_replacement_policy.py tests/test_budget_controller.py tests/test_budget_fallback_recovery.py tests/test_budget_transition_scenarios.py tests/test_memory_pressure_rule.py tests/test_proposer_adapter.py tests/test_budget_planner.py tests/test_adaptive_scheduler_glue.py tests/test_lineage_startup.py
python -B -m pytest -p no:cacheprovider --basetemp <scratch> -q --tb=short -rA tests/test_m4_pinned_contract.py
```

The first patched run used a partial capture and returned 203 passes, one failure, 23 setup errors and three deselections. Missing executor source and installed-source manifest files blocked startup fixture setup. A complete capture from the sibling M4 checkout and the audit's captured EAGLE protocol file matched all baseline SHA-256 manifests (392 installed, 15 scalar and four EAGLE manifest entries). The rerun restored the three deselections and all startup/source-gate cases. The final EAGLE check used `M4_VLLM_SOURCE_ROOT` pointing to the verified temporary capture. The regression suites also exercised expected step-telemetry I/O errors as part of passing tests.

## Invariant report

| Invariant | Result | Test IDs | Evidence |
|---|---|---|---|
| I1 | PASS | T-WARM-002/003/004/005, T-ACT-008 | Only a live, positive-width, uniquely claimed birth increments. |
| I2 | PASS | T-WARM-001 | Zero-draft calls and decision-only calls leave attributed count unchanged. |
| I3 | PASS | T-WARM-002, T-TEL-003 | Real positive-width zero-acceptance feedback increments. |
| I4 | PASS | T-FLOOR-002/003/005, T-ACT-001/004 | Memory, cooldown, stale/ineligible rows and explicit disable produce applied zero when relevant. |
| I5 | PASS | T-FLOOR-001/002/003/006 | Floor is confined to acceptance fallback. |
| I6 | PASS | T-WARM-006, T-FLOOR-006, T-TEL-001 | Acceptance reports exact rule alpha. Safety decisions report null because they bypass alpha. |
| I7 | PASS | T-HOLD-004/005, T-ACT-001/002/003/005 | Applied row K is bounded by its fresh selection. |
| I8 | PASS | T-HOLD-004/005, T-ACT-001/002, T-TEL-002 | Actual eligible scheduled rows determine max applied K and padded width. |
| I9 | PASS | T-HOLD-003/008, T-ACT-004 | All-zero fresh maximum cuts worker K immediately. |
| I10 | PASS | T-HOLD-002, T-ACT-003 | Downward transition publishes K=2 after K=4 in one schedule. |
| I11 | PASS | T-HOLD-001/007/008, T-ACT-005 | Only positive increases wait for matching eligible proposals. |
| I12 | PASS | T-HOLD-006, T-LIFE-002 | Empty, disjoint and explicitly reset cohorts start fresh. |
| I13 | PASS | T-WARM-007, T-LIFE-001/003/004 | Controller state resets on explicit reset, finish and new request incarnation. |
| I14 | PASS | T-WARM-003/004, T-ACT-004/006/007/008 | Feedback joins to the exact consumed producer, filters stale rows and rejects duplicates. |
| I15 | PASS | Git and audit check | Frozen source and evidence SHA resolved independently. Historical evidence files were left in place. |
| I16 | PASS | 183 existing regressions, pinned EAGLE, profile preservation and T-TEL-004 | Source gates, startup, profiles, topology telemetry, snapshots, bounded histories and legacy lineage interfaces passed. |

## Remaining gaps

- **SOURCE** — The verified worktree has a fingerprint and has not been committed. Record an immutable patched source SHA before collecting new runtime evidence. The manifest hashes UTF-8 lines of `<SHA-256>  <repo-relative path>\n`, sorted by path, using raw worktree bytes for the six inspected source files and separately for the two edited test files.
- **TEST** — All final CPU suites ran. Padded GPU proposal and end-to-end vLLM execution are covered by pinned-source checks and CPU simulation at this phase.
- **GPU** — A separately authorized bounded run is needed for real width, memory recovery, attribution, an effective mixed batch and matched token controls.
- **SCIENTIFIC** — Frozen ON/OFF and target-only evidence remains attributable to its original SHA. Lossless closure and performance claims remain open.
- **OUT-OF-SCOPE** — R6 measured hardware cost consumption, TreeAdaptive, experiment protocol and forensic lossless root-cause work were outside this patch.

## Final verdict

**READY FOR BOUNDED GPU VALIDATION**

No GPU execution or performance benchmark was run.

## Review corrections and final CPU gate

This section supersedes the earlier readiness verdict and fingerprints. HEAD remains 4c76830706d6d60d715d962764e592b7b184fc42 on audit/flexsds-v023-hardwareadaptive-20261120. The patch remains uncommitted.

| Finding | File/function | Root cause | Fix | Test IDs | Result |
|---|---|---|---|---|---|
| reset alpha | controller.py, AdaptiveDraftController.reset and BudgetController.reset | The suspected stale prior was absent. The parent reset already assigns both initial_alpha and alpha_ema. | Added explicit regression coverage for reset(.50), reset(), and the active warmup decision. Existing session reuse coverage remains. | T-WARM-008, T-LIFE-004 | PASS, production reset required no change |
| width telemetry | lineage.py, DraftOrigin and produced; budget_planner.py, record_draft_birth and record_verification | Feedback labeled planned batch K as physical width. Width mismatch returned before a measurement record existed. | Preserve batch_k and worker_k compatibility aliases, expose planned_worker_k, attach observed physical_draft_width to the exact origin at birth, and read that observation at feedback. Mismatch emits a birth record and leaves generation unset so feedback fails closed. | T-TEL-005/006 | PASS |
| request reuse cleanup | adaptive_scheduler.py, _free_request | Native cleanup is keyed partly by request ID. Calling it on an old object could remove the current request. The guard also skips resource cleanup. | Retained incarnation protection and added a dedicated CPU assertion. Native allocator cleanup remains a documented runtime gap under Outcome B. | T-LIFE-005/006 | New incarnation protection PASS; old-resource cleanup BLOCKED BY RUNTIME FIXTURE |

The physical observation is excluded from DraftOrigin equality and hashing. It is written at the producer birth boundary on that same identity witness, preserving existing weak-set duplicate tracking and plan/schedule/incarnation bindings.

| Test ID | Before fix | After fix | Evidence |
|---|---|---|---|
| T-WARM-008 | PASS | PASS | Parent reset already updates the configured prior. Both .50 decisions use .50 and zero samples after reset. |
| T-TEL-005 | FAIL | PASS | Synthetic planned 4 / observed 3 emits both values distinctly and rejects the mismatched generation. |
| T-TEL-006 | FAIL | PASS | Feedback for producer schedule 1 retains its observed width 2 after the newest worker changes to 4. |
| T-LIFE-005 | PASS | PASS | Late old-object cleanup preserves the fresh request table entry, running membership, controller identity and active lineage incarnation. Native cleanup stub is never called. |
| T-LIFE-006 | BLOCKED BY RUNTIME FIXTURE | BLOCKED BY RUNTIME FIXTURE | Current fixtures model request/controller/lineage state but do not instantiate native KV allocator, encoder cache, connector or deferred-free ownership. This is a documented deferred test, outside executable case totals. |

Before production changes the four new executable cases returned two passes and two failures. Final executable count increased by four from 232 to 236.

| Suite | Passed | Failed | Skipped |
|---|---:|---:|---:|
| replacement policy including profile compatibility | 52 | 0 | 0 |
| controller | 19 | 0 | 0 |
| fallback recovery | 8 | 0 | 0 |
| transitions | 6 | 0 | 0 |
| memory rule | 13 | 0 | 0 |
| adapter | 8 | 0 | 0 |
| planner | 35 | 0 | 0 |
| scheduler glue | 67 | 0 | 0 |
| lineage startup | 27 | 0 | 0 |
| pinned vLLM/EAGLE | 1 | 0 | 0 |
| TOTAL | 236 | 0 | 0 |

TOTAL PASSED = 236
TOTAL FAILED = 0
TOTAL SKIPPED = 0

The same temporary-export runner, interpreter, pinned fixtures, manifest verification and pytest options described above were used. The regression assertions remain intact.

The pinned native _free_request would discard the old prefill membership, notify the KV connector, notify the encoder connector, free encoder cache, update finished-request notifications, then conditionally free KV blocks and delete the request-table entry by ID. Deferred block cleanup can instead queue frees behind the in-flight fence. The replacement guard returns before all of those operations and before planner finish, controller release and lineage finish. The prior bind_request transition retires old planner/controller state, while old lineage origins can remain until their normal output retirement. The CPU test proves protection of the new state. It does not prove reclamation of old KV, encoder, connector or deferred resources. Calling native cleanup selectively without those ownership contracts would be unsafe. Outcome B is retained as a GPU/RUNTIME VALIDATION GAP.

| Invariant | Result | Evidence |
|---|---|---|
| I1 | PASS | WARM-002/003/004/005 and ACT-008 count each attributed positive-width birth once. |
| I2 | PASS | WARM-001 rejects zero-width warmup increments. |
| I3 | PASS | WARM-002 counts zero-acceptance real feedback. |
| I4 | PASS | FLOOR-002/003 and ACT-004 preserve safety zero. |
| I5 | PASS | FLOOR-001/002/003/006 preserve safety over floor. |
| I6 | PASS | WARM-006/008 and TEL-001 identify the exact rule alpha, including reset priors. |
| I7 | PASS | HOLD-004/005 exhaust row caps; ACT-001/002 verify actuation. |
| I8 | PASS | ACT-001/002/003 and TEL-005/006 distinguish current max-applied worker K from measured birth width. |
| I9 | PASS | HOLD-003/008 and ACT-004 force immediate all-zero. |
| I10 | PASS | HOLD-002 and ACT-003 apply decreases immediately. |
| I11 | PASS | HOLD-001/007/008 and ACT-005 hold increases only. |
| I12 | PASS | HOLD-006 and LIFE-002 reset empty-cohort state. |
| I13 | PASS | LIFE-004/005 protect the current incarnation and reject old feedback. Native old-resource reclamation remains deferred under LIFE-006. |
| I14 | PASS | ACT-006/007/008, WARM-003/004 and TEL-006 bind feedback and observed width to the consumed producer. |
| I15 | PASS | Frozen baseline and evidence SHA remain as recorded. No historical evidence was edited. |
| I16 | PASS | All 183 baseline regressions plus profile preservation and pinned source gates pass. Existing origin and telemetry aliases remain. |

The remaining GPU/runtime-dependent gaps are actual padded proposal widths on a live vLLM worker, native old-incarnation KV/encoder/connector/deferred-resource reclamation, real memory fallback and recovery, and an effective mixed runtime batch. CPU-closed reset, measurement provenance and new-incarnation protection require no further CPU deferral.

**READY TO COMMIT**

The human reviewer can inspect the uncommitted diff before freezing a new SHA. No commit, push, GPU execution or performance benchmark was performed.

