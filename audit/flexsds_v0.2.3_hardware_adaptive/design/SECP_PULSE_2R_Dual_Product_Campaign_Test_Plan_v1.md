# SECP-PULSE-2R-v1 Dual-Product Full Profile Campaign Test Plan

**Document type:** Verification and acceptance test plan<br>
**Version:** 1.0<br>
**Companion specification:** `SECP_PULSE_2R_Dual_Product_Campaign_Spec_v1.md`<br>
**Target:** Campaign harness implementation only<br>
**Live GPU execution by Codex:** FORBIDDEN<br>
**Required starting executable SHA:** `ac395ed035374ea3937e79b4cea077feb59bfa1a`

---

## 1. Test objective

This plan proves that the campaign harness can safely and deterministically produce two separate `SECP-PULSE-2R-v1` profile campaigns for `NVIDIA A10` and `NVIDIA A10G`, while preserving the frozen scientific protocol and without requiring live GPU execution during implementation review.

The test plan is intentionally prescriptive. If a test contradicts implementation convenience, the implementation SHALL change; the test requirement SHALL not be weakened without technical review.

---

## 2. Test authority and precedence

Order of authority:

1. applied `SECP-PULSE-2R-v1` specification/test/G0 documents in the repository;
2. active production code at the reviewed checkpoint and subsequent exact implementation commit;
3. this campaign specification;
4. pinned vLLM `0.30.0` source/tests;
5. existing A10/A10G live smoke evidence;
6. external official documentation as cross-check only.

If this test plan conflicts with the applied P2R clauses, STOP with:

```text
TECHNICAL REVIEW REQUIRED — CAMPAIGN SPEC CONFLICT
```

---

## 3. Environment

CPU/local test environment:

```bash
export M4_VLLM_SOURCE_ROOT="/d/Document/Four-year/do_an/do-an-ptit-d22-m4-02-02/experiments/qualification/frozen_v1_audit/upstream"
export PYTHONPATH="$PWD:$PWD/src:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Pinned runtime:

```text
vLLM 0.30.0
```

No test in the implementation task may execute Modal, allocate a GPU, upload source externally, or perform live inference.

---

## 4. Test categories

Mandatory categories:

```text
A. Baseline / source provenance
B. Plan schema / immutability
C. Product routing
D. Cell set
E. Cohort-slot semantics
F. Pulse-cycle scientific record
G. Failure handling
H. Durable state / atomic resume
I. Scheduling / bounded loop
J. Publication statistics
K. Profile publication/admission
L. Dry-run
M. Fake executor end-to-end
N. Git bundle handoff
O. Security / negative controls
P. Regression
```

---

# A. Baseline and source provenance

## CAM-A01 — Expected starting SHA

**Given:** repository before campaign implementation.<br>
**Assert:** `git rev-parse HEAD` equals the expected starting checkpoint unless the implementation task has already created its reviewed successor commit.<br>
**Failure:** stop and report source mismatch.

## CAM-A02 — Branch invariant

Assert branch is:

```text
audit/flexsds-v023-hardwareadaptive-20261120
```

No branch switch is permitted.

## CAM-A03 — Pinned vLLM manifest

Verify all 19 pinned files and version `0.30.0` using existing project verifier.

## CAM-A04 — Applied PULSE objective exists

Production/admission path must declare `SECP-PULSE-2R-v1`.

## CAM-A05 — Historical objective rejection remains

Legacy component, steady-state, and first-response profiles must remain rejected by active pulse admission.

---

# B. Campaign plan schema and immutability

## CAM-B01 — Canonical plan serialization

Create logically identical plan dictionaries with different insertion orders.<br>
Assert canonical serialized bytes and SHA-256 are identical.

## CAM-B02 — Scientific-field mutation changes hash

For each of the following, mutate exactly one field and assert plan hash changes:

- product set,
- source SHA,
- any cell q,
- old width,
- continuation K,
- slot count,
- bootstrap resamples,
- confidence,
- seed,
- runtime hash,
- preparation hash,
- queue hash,
- continuation hash,
- failure policy.

## CAM-B03 — Plan frozen after first attempt

After state contains any attempted slot, a changed plan SHALL be rejected before external allocation.

## CAM-B04 — Exact products only

Plan containing product outside `{NVIDIA A10, NVIDIA A10G}` is rejected.

## CAM-B05 — Exact cell set only

Plan must contain exactly:

```text
old0_q0
old0_q2
old0_q3
```

No missing cell and no additional K4.

## CAM-B06 — Exact publication constants

Assert:

```text
slots_per_cell = 50
bootstrap_resamples = 5000
confidence = 0.95
seed = 731
```

---

# C. Product routing

## CAM-C01 — A10 routes only A10

Synthetic observed identity `NVIDIA A10` selects only A10 state.

## CAM-C02 — A10G routes only A10G

Synthetic observed identity `NVIDIA A10G` selects only A10G state.

## CAM-C03 — A10 cannot write A10G raw path

Attempt cross-product write; must fail.

## CAM-C04 — A10G cannot write A10 raw path

Attempt cross-product write; must fail.

## CAM-C05 — Requested selector is not observed product

Set requested selector `A10G`, observed product `NVIDIA A10`.<br>
Assert route is A10.

## CAM-C06 — Unsupported product

Observed product `NVIDIA L4` or arbitrary value: allocation recorded, zero cohort slots executed.

## CAM-C07 — Completed product skip

If A10 is complete and worker reports A10, no new slot is attempted.

---

# D. Exact cell-set semantics

## CAM-D01 — q0 cell

Assert cell payload binds:

```text
old_width=0
future_k=0
shared_k=0
continuation_k=0
horizon=2
```

## CAM-D02 — q2 cell

Assert same with `future_k=shared_k=2`.

## CAM-D03 — q3 cell

Assert same with `future_k=shared_k=3`.

## CAM-D04 — K4 rejected

Any plan/profile cell q4 is rejected in this campaign domain.

## CAM-D05 — Diagonal substitution rejected

`(old2,q2)` cannot satisfy `old0_q2`; `(old3,q3)` cannot satisfy `old0_q3`.

## CAM-D06 — Nearest-state substitution rejected

A cell with mismatching context bucket/old width may not be selected as nearest compatible cell.

---

# E. Cohort-slot semantics

## CAM-E01 — Exactly 50 deterministic IDs

Each product/cell initializes slot IDs `000` through `049` and no others.

## CAM-E02 — Stable slot IDs across restarts

Reconstruct state from disk; slot IDs/order unchanged.

## CAM-E03 — One scientific attempt per slot

After slot leaves `UNATTEMPTED`, second attempt request is rejected.

## CAM-E04 — Valid slot cannot be rerun

`VALID` is immutable.

## CAM-E05 — Failed slot cannot be replaced

`FAILED` is immutable and a new slot `050` must not be created.

## CAM-E06 — Failed slot does not count valid

Counters remain separate.

## CAM-E07 — Failure blocks publication under v1

A cell with one failed predetermined slot cannot publish.

## CAM-E08 — Fail-fast campaign behavior

On a scientific slot failure, campaign persists evidence and stops before spending further allocations under the v1 no-replacement rule.

---

# F. Pulse-cycle scientific record

Use the existing project fake/native fixtures, not new ad hoc semantics.

## CAM-F01 — Fresh incarnation

Each slot uses a new request incarnation ID.

## CAM-F02 — Old width = 0 at pre-action receipt

A slot whose pre-action state has old width !=0 is rejected.

## CAM-F03 — Base K = 3

Mismatch rejected.

## CAM-F04 — B=1

More than one eligible row rejected for this campaign version.

## CAM-F05 — Context bucket

Only `[0,256)` accepted.

## CAM-F06 — Pulse applied once

q is applied exactly one scheduling action.

## CAM-F07 — Continuation exactly K0

Second action is 0.

## CAM-F08 — No ordinary planner override

A fixture attempting to replace continuation with q2/q3 must invalidate the cycle.

## CAM-F09 — Retirement 1 lineage

Retirement 1 origin equals pulse origin.

## CAM-F10 — Retirement 2 lineage

Retirement 2 origin equals continuation origin.

## CAM-F11 — Predecessor exclusion

A predecessor output cannot enter reward/timing/finalization.

## CAM-F12 — Retirement 3 exclusion

Third retirement cannot enter the pulse cycle.

## CAM-F13 — Timing ordering

`t1 > t0`; nonfinite/negative/zero invalid timing rejected according to production validator.

## CAM-F14 — No added synchronization

Static/source assertion: campaign path must not add `torch.cuda.synchronize` or equivalent timing synchronization not already required by active design.

## CAM-F15 — Useful token fields present

Both retirements retain actual useful post-stop emitted-token evidence.

## CAM-F16 — Physical width evidence

Applied K, worker K, and observed physical proposal width are distinct fields and preserved.

## CAM-F17 — Canonical objective tag

Record objective must be `SECP-PULSE-2R-v1`.

---

# G. Failure and censoring

## CAM-G01 — OOM retained

Simulated OOM writes immutable failure record and marks slot FAILED.

## CAM-G02 — Timeout retained

Same for timeout.

## CAM-G03 — Terminal/censored retained

Unexpected terminal/censored cycle remains raw evidence and cannot be VALID.

## CAM-G04 — Identity mismatch retained

Mismatch produces allocation/slot evidence as applicable, never canonical sample.

## CAM-G05 — DecisionState mismatch retained

No relabeling to expected cell.

## CAM-G06 — Collector failure retained

Raw partial evidence preserved; slot FAILED.

## CAM-G07 — Lineage failure retained

No post-hoc repair/fabrication.

## CAM-G08 — Failure cannot mutate to VALID

State transition `FAILED -> VALID` rejected.

---

# H. Durable state and resume

## CAM-H01 — Atomic state write path

Unit-test temporary-write + replace mechanism; no in-place partial JSON writes.

## CAM-H02 — Interrupted temp write

Simulate crash before replace. Previous state remains valid.

## CAM-H03 — State file parse failure

Corrupt canonical state -> stop, no external allocation.

## CAM-H04 — Plan hash mismatch

Stop before external allocation.

## CAM-H05 — Source SHA mismatch

Stop before external allocation.

## CAM-H06 — Raw manifest missing

Stop before external allocation.

## CAM-H07 — Raw hash mismatch

Stop before external allocation.

## CAM-H08 — Profile hash mismatch

Completed product with mutated profile -> stop.

## CAM-H09 — Concurrent writer

Second campaign process using same campaign directory is rejected.

## CAM-H10 — Resume partial slot set

Given slots 000-009 valid, scheduler chooses 010 next.

## CAM-H11 — Resume complete cell

Scheduler skips all slots in completed cell.

## CAM-H12 — Resume complete product

Scheduler skips product.

## CAM-H13 — Complete dual product exits immediately

No external executor invocation.

---

# I. Scheduling and bounded allocation loop

## CAM-I01 — Deterministic cell order

Order is q0 -> q2 -> q3.

## CAM-I02 — Deterministic slot order

Smallest unattempted ID first.

## CAM-I03 — MAX_ALLOCATIONS zero/invalid

Nonpositive live limit rejected before run.

## CAM-I04 — MAX_ALLOCATIONS boundary

If limit N reached, allocation N+1 is never requested.

## CAM-I05 — `max_containers=1`

Static/runtime config assertion.

## CAM-I06 — `retries=0`

Static/runtime config assertion.

## CAM-I07 — Unsupported allocations count against allocation budget

Prevents infinite hunting.

## CAM-I08 — Completed-product allocations count against budget

Allocation occurred; budget decreases even when scientific slots are skipped.

## CAM-I09 — No `while true` without bound

Static test or structural assertion verifies loop condition includes allocation bound.

## CAM-I10 — Multiple fresh cohorts on one worker

Fake executor demonstrates multiple slots can execute with fresh request/incarnation while stable worker identity is preserved, if allowed by protocol.

## CAM-I11 — No history leakage between slots

Fixture deliberately leaves history/cache marker; next slot setup must reset/reconstruct or reject.

---

# J. Publication statistics

## CAM-J01 — Cohort-level resampling

Bootstrap receives whole cycle/cohort values; it never expands retirement1/retirement2 into IID samples.

## CAM-J02 — 5000 resamples exactly

Instrument statistic helper; count exactly 5000.

## CAM-J03 — Seed 731 reproducibility

Same 50 values -> byte/float-equivalent published interval under deterministic implementation.

## CAM-J04 — 95% interval

Assert configured confidence is 0.95.

## CAM-J05 — Point/low/high finite

NaN/Inf publication rejected.

## CAM-J06 — Interval ordering

Require `C_low <= C_point <= C_high` if this is an invariant of the existing reviewed publisher; otherwise test the exact existing publisher invariant rather than inventing one.

## CAM-J07 — Uses reviewed publisher

Static/source witness confirms campaign calls existing `publish_pulse_cost`/approved function rather than duplicating statistical logic.

## CAM-J08 — 49 valid slots rejected

No profile publication.

## CAM-J09 — 50 valid slots accepted

Synthetic cell publishes.

## CAM-J10 — 50 valid + hidden failed extra rejected/flagged

Harness cannot conceal failure artifacts outside plan.

---

# K. Profile publication and admission

## CAM-K01 — Separate A10 profile

Product exactly `NVIDIA A10`.

## CAM-K02 — Separate A10G profile

Product exactly `NVIDIA A10G`.

## CAM-K03 — Product cross-admission rejected

A10 profile rejected on A10G and vice versa.

## CAM-K04 — All three cells mandatory

Missing q0/q2/q3 rejects product publication.

## CAM-K05 — Objective mandatory

Missing/wrong objective rejected.

## CAM-K06 — Horizon mandatory

Horizon !=2 rejected.

## CAM-K07 — Continuation mandatory

Continuation !=0 rejected.

## CAM-K08 — Source SHA mandatory

Mismatch rejected.

## CAM-K09 — Runtime/config hash mandatory

Mismatch rejected.

## CAM-K10 — Model revisions mandatory

Mismatch rejected.

## CAM-K11 — Raw hashes mandatory

Missing provenance rejects publication/admission.

## CAM-K12 — Legacy component profile rejected

Existing component profile cannot enter campaign publication/admission.

## CAM-K13 — First-response profile rejected

## CAM-K14 — Steady-state profile rejected

## CAM-K15 — Round-trip admission

Synthetic published profile reloads through active production admission and matches expected product/domain.

---

# L. Dry-run behavior

## CAM-L01 — Dry-run performs zero Modal call

Use fake/mocked Modal adapter that fails test if called.

## CAM-L02 — Dry-run performs zero upload

Same guard.

## CAM-L03 — Dry-run prints source SHA

## CAM-L04 — Dry-run prints plan hash

## CAM-L05 — Dry-run prints both product states

## CAM-L06 — Dry-run prints cell/slot progress

## CAM-L07 — Dry-run prints deterministic next action

## CAM-L08 — Dry-run invalid state exits nonzero

## CAM-L09 — No subcommand defaults to non-live

CLI with no `run` must not invoke external executor.

---

# M. Fake executor end-to-end

## CAM-M01 — Dual-product happy path

Synthetic executor supplies enough A10/A10G allocations and valid slots for all cells. Assert:

```text
A10 COMPLETE
A10G COMPLETE
campaign COMPLETE
```

No network/GPU.

## CAM-M02 — A10G-first sequence

Allocation sequence begins A10G then A10. Correct routing and eventual completion.

## CAM-M03 — Repeated same product

Many A10G allocations before A10; completed A10G is skipped while allocation budget decreases.

## CAM-M04 — Unsupported product intermixed

Unsupported allocations preserved, no scientific slots executed, eventual valid products still route correctly within budget.

Explicit user resolution (2026-11-22): each unsupported product consumes
exactly one allocation, records full observed identity, releases the worker,
and continues within MAX_ALLOCATIONS. Assert zero slot attempts and unchanged
A10/A10G scientific slot state. Repeated unsupported allocations terminate at
MAX_ALLOCATIONS_REACHED. Unsupported product alone is not a campaign hard stop.

## CAM-M05 — Interruption/resume

Stop synthetic run mid-cell, reload from disk, finish without repeating prior slots.

## CAM-M06 — Failed slot blocks campaign

Inject one failed predetermined slot. Assert campaign BLOCKED/PUBLICATION_BLOCKED and no replacement slot.

## CAM-M07 — MAX_ALLOCATIONS termination

Insufficient product availability results in clean `MAX_ALLOCATIONS_REACHED`, not infinite loop.

## CAM-M08 — Synthetic outputs isolated

Synthetic profiles/raw data cannot appear in canonical live publication path.

---

# N. Git bundle handoff

These tests run after implementation commit; they are local only.

## CAM-N01 — Exact implementation SHA

Record new local commit SHA.

## CAM-N02 — Self-contained bundle creation

Bundle exact ref pointing to new SHA.

## CAM-N03 — `git bundle verify`

PASS.

## CAM-N04 — Empty-repository clone/reconstruction

PASS with no prerequisite object database.

## CAM-N05 — Exact HEAD

Clone HEAD equals new SHA.

## CAM-N06 — Clean checkout

`git status --short` empty.

## CAM-N07 — `git fsck --full`

Exit 0, no missing reachable objects.

## CAM-N08 — No alternates

No `.git/objects/info/alternates` dependency.

## CAM-N09 — Runtime manifest

Every execution-source file hash matches reviewed source bytes.

## CAM-N10 — Bundle hash/size recorded

SHA-256 and bytes written to evidence metadata.

No upload is permitted in the implementation task.

---

# O. Security and negative controls

## CAM-O01 — Path traversal rejected

Campaign/cell/product IDs cannot write outside campaign root.

## CAM-O02 — Symlink escape rejected

Raw/publication paths resolving outside root rejected where applicable.

## CAM-O03 — Unknown state enum rejected

## CAM-O04 — Unknown schema version rejected

## CAM-O05 — Manual `.done` marker has no authority

Creating a marker cannot make product/cell complete.

## CAM-O06 — Edited state cannot invent raw record

Missing raw/hash verification rejects forged VALID state.

## CAM-O07 — Edited raw record invalidates hash

## CAM-O08 — Duplicate slot ID rejected

## CAM-O09 — Duplicate cohort provenance rejected

## CAM-O10 — Product string normalization cannot alias A10/A10G

For example, prefix matching must not make `NVIDIA A10` equal `NVIDIA A10G`.

## CAM-O11 — Cloud selector cannot override observed product

## CAM-O12 — K4 injection rejected

## CAM-O13 — q2/q3 diagonal substitution rejected

## CAM-O14 — component profile promotion rejected

## CAM-O15 — performance fields not generated as claims

Campaign output may contain measured cycle costs, but no speedup/TPOT/throughput conclusion.

---

# P. Existing regression suite

After campaign tests pass, run existing relevant regressions in this order:

1. pulse policy/profile/planner tests;
2. pulse collector/bootstrap tests;
3. DecisionState transport;
4. pending feedback/finalization;
5. lineage/executor;
6. identity observation/admission;
7. safety/recovery/controller;
8. telemetry;
9. prior 746 distinct PULSE migration baseline set;
10. pinned vLLM 19/19 verifier;
11. `git diff --check`.

Do not double-count repeated tests in final distinct count.

Any regression failure is at least MAJOR; correctness/profile/source-provenance failures are BLOCKER.

---

## 5. Test implementation guidance

Preferred new test files:

```text
tests/test_eagle_pulse_campaign.py
tests/test_eagle_pulse_campaign_publication.py
tests/test_eagle_pulse_campaign_cli.py
```

Use `tmp_path` for isolated state/raw directories.

Use a fake executor interface; do not patch real Modal calls deeply into business logic. The campaign scheduler/state machine should be testable with a small executor protocol such as:

```python
class CampaignExecutor(Protocol):
    def allocate(self, ...): ...
    def observe_identity(self, ...): ...
    def execute_slot(self, ...): ...
    def close(self, ...): ...
```

Exact interface may differ, but external side effects must be injectable so `--dry-run` and tests can prove zero external actions.

---

## 6. Required test evidence artifacts

The implementation task SHALL produce an evidence directory containing at least:

```text
baseline.json
changed-files.txt
campaign-plan-fixture.json
campaign-state-schema.json
test-commands.txt
test-results-summary.json
campaign-tests.xml
regression-tests.xml
dry-run-output.txt
fake-happy-path-output.txt
fake-resume-output.txt
fake-failure-output.txt
traceability.json
source-diff.patch
source-freeze.json
bundle-verification.json
final-checks.json
final-review.md
```

No live GPU raw data is expected because Codex is forbidden from running the campaign.

---

## 7. Requirements-to-tests traceability

Codex SHALL create a machine-readable map from every normative campaign requirement to one or more test IDs.

Minimum required result:

```text
requirements mapped: 100%
unmapped: 0
duplicate requirement IDs: 0
```

The campaign specification SHOULD be assigned stable requirement IDs during implementation if they do not already exist, e.g. `P2RC-001...`.

---

## 8. Acceptance gate

The campaign harness is ready for user execution only if:

```text
all mandatory CAM-* tests PASS
all relevant existing regressions PASS
pinned vLLM verification PASS
git diff --check PASS
tracked/index state matches source-freeze evidence
new implementation SHA exists
self-contained Git bundle PASS
empty-repo reconstruction PASS
no Modal/GPU/upload/live inference occurred
```

Required final status:

```text
FULL DUAL-PRODUCT CAMPAIGN HARNESS READY — USER RUN REQUIRED
```

Any failure must use the narrowest blocker, e.g.:

```text
TECHNICAL REVIEW REQUIRED — CAMPAIGN IMPLEMENTATION: <exact issue>
TEAM TECHNICAL ALIGNMENT REQUIRED — <exact shared interface>
```

---

## 9. Manual-command output requirements

Only after the acceptance gate passes, Codex SHALL output:

### 9.1 Safe dry-run command

A copy/paste Git Bash command that invokes the implemented `--dry-run` mode and cannot allocate a GPU.

### 9.2 Exactly one primary live campaign command

Prefix exactly:

```text
DO NOT RUN IN CODEX — USER MANUAL COMMAND
```

The command SHALL:

- point to the exact frozen campaign directory/plan,
- resume existing valid state,
- accept/require `MAX_ALLOCATIONS`,
- use the reviewed source handoff,
- automatically route observed A10 vs A10G,
- skip completed products/cells,
- stop on campaign completion, blocker, or allocation budget.

### 9.3 Post-run validation command

A local command that verifies both final profiles, state, raw hashes, publication manifests, and active profile admission without running another GPU job.

---

## 10. Explicit prohibition during Codex implementation task

No test or implementation step may execute:

```text
modal run
modal deploy
modal shell
live Modal function call
GPU allocation
remote upload
live inference
live profile collection
live profile publication
POLICY_VALIDATION
TPOT/throughput/performance benchmark
```

A test that accidentally calls an external Modal operation is an immediate task failure and must preserve the incident as evidence.

---

## 11. External references checked

Cross-checks only; project sources remain authoritative.

- vLLM Ascend v0.30.0 speculative decoding:<br>
  https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/speculative_decoding.html
- Git bundle:<br>
  https://git-scm.com/docs/git-bundle.html
- Modal SDK function options:<br>
  https://modal.com/docs/sdk/py/latest/App
- Modal GPU selector example:<br>
  https://modal.com/docs/cli/latest/shell
- Python atomic replace primitive:<br>
  https://docs.python.org/3/library/os.html#os.replace

---

## 12. Final implementation reviewer checklist

Before accepting `READY`, reviewer answers YES to all:

```text
[ ] Objective remains SECP-PULSE-2R-v1
[ ] A10 and A10G remain separate
[ ] Exact cells are q0/q2/q3 only
[ ] 50 predetermined slots/cell
[ ] Failed slots are retained and not replaced
[ ] Resume validates source/plan/raw hashes
[ ] State writes are atomic/single-writer
[ ] Loop is allocation-bounded
[ ] max_containers=1
[ ] retries=0
[ ] Dry-run cannot invoke Modal
[ ] Fake executor reaches dual-product COMPLETE
[ ] Failure fake blocks publication
[ ] Bootstrap is cohort-level, 5000, 95%, seed 731
[ ] Reviewed publisher reused
[ ] Separate A10/A10G typed profiles round-trip admit
[ ] Component/steady/first-response profiles reject
[ ] New source SHA frozen
[ ] Self-contained Git bundle verifies in empty repo
[ ] No GPU/Modal/upload was executed by Codex
[ ] Manual dry-run command returned
[ ] Exactly one manual live command returned
[ ] Post-run validation command returned
```
