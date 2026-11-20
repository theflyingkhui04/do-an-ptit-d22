# FlexSDS Hardware-Adaptive Replacement Policy Test Plan

- **Companion spec:** `replacement-policy-spec.md`
- **Baseline:** `8006205a6f19790a776d67155a377def053d2736`
- **Purpose:** Define tests that must exist before and after implementing the replacement warmup / batch-K stabilization policy
- **Rule:** Tests define the contract first; production code must not be changed until the failing tests are reviewed
- **GPU performance benchmark:** forbidden in this phase

---

## 1. Test strategy

Use four layers:

```text
L1 — pure controller tests
L2 — planner tests
L3 — scheduler / lineage integration tests
L4 — bounded GPU correctness smoke after CPU closure
```

This plan intentionally separates:

```text
selected K
applied/scheduled K
worker physical K
verified draft width
feedback destination
```

because they are not interchangeable.

---

## 2. Required test fixtures

Create deterministic fixtures for:

```text
candidate Ks = [0, 2, 3, 4]
initial_alpha = 0.80
max_k = 4
probe_k = 2
probe_interval = 20
memory trigger:
  KV > 90% OR free VRAM < 1500 MiB
memory recovery:
  KV < 80% and VRAM clear after cooldown
cooldown = 3 decisions
```

Use request identities that include incarnation where supported:

```text
req-A / incarnation-1
req-B / incarnation-1
req-A / incarnation-2
```

---

## 3. L1 — controller warmup tests

### T-WARM-001 — zero-width update does not count

Setup:

```text
min_warmup_samples = 3
initial_alpha = 0.80
```

Actions:

```text
update(0, 0)
update(0, 0)
```

Expected:

```text
verification_sample_count = 0
warmup_active = true
rule_alpha = 0.80
```

Severity if failed:

```text
BLOCKER
```

### T-WARM-002 — zero acceptance with real drafts counts

Actions:

```text
feedback(drafted=3, accepted=0, valid lineage)
```

Expected:

```text
verification_sample_count += 1
```

Rationale:

```text
accepted=0 is still a real verifier observation
```

### T-WARM-003 — duplicate feedback counts once

Deliver the same feedback origin twice.

Expected:

```text
verification_sample_count incremented once
EMA updated once
duplicate logged/skipped
```

### T-WARM-004 — stale/wrong-incarnation feedback does not count

Expected:

```text
verification_sample_count unchanged
EMA unchanged
```

### T-WARM-005 — N-1 / N boundary

With:

```text
min_warmup_samples = 3
```

After two valid samples:

```text
warmup_active = true
rule_alpha = initial_alpha
```

After third valid sample and before next decision, define expected transition explicitly.

Preferred contract:

```text
the next decision observes warmup_active = false
```

### T-WARM-006 — telemetry distinguishes alphas

Create state where:

```text
effective_alpha != initial_alpha
warmup_active = true
```

Expected decision telemetry:

```text
effective_alpha = estimator value
rule_alpha = initial_alpha
selected_k derived from rule_alpha
```

### T-WARM-007 — reset / request reuse

Finish `req-A/incarnation-1`, create `req-A/incarnation-2`.

Expected:

```text
verification_sample_count = 0
EMA reset
fallback/probe state reset
old feedback rejected
```

---

## 4. L1 — floor / safety tests

### T-FLOOR-001 — acceptance floor applies only to acceptance fallback

For:

```text
alpha = 0.50
min_k = 2
no memory pressure
```

Expected:

```text
selected_k = normalized floor K
```

### T-FLOOR-002 — memory overrides floor

For:

```text
alpha = 0.50
min_k = 2
KV = 95%
```

Expected:

```text
selected_k = 0
reason = memory...
```

### T-FLOOR-003 — cooldown overrides floor

Trigger memory fallback, then clear raw pressure before cooldown ends.

Expected during cooldown:

```text
selected_k = 0
```

### T-FLOOR-004 — candidate normalization

Cases:

```text
candidates=[0,2,3,4], min_k=1
candidates=[0,2,3],   min_k=4
```

Expected behavior must match the spec-defined normalization/failure rule.

No silent unsupported K.

### T-FLOOR-005 — probe cadence independent from memory fallback

Memory-forced K=0 decisions must not advance acceptance-fallback probe cadence unless explicitly specified.

### T-FLOOR-006 — acceptance boundaries

Test exact alphas:

```text
0.50
0.64
0.65
0.79
0.80
0.89
0.90
```

For combinations:

```text
warmup on/off
min_k 0/2
memory on/off
probe due/not due
```

Generate an explicit expected matrix.

---

## 5. L2 — safe recovery-cap tests

These tests assume the replacement uses:

```text
applied_k_i = min(selected_k_i, recovery_cap_k)
worker_k = max(applied_k_i)
```

### T-HOLD-001 — upward recovery delayed

Initial:

```text
cap=2
hold_steps=3
```

Fresh selected max sequence:

```text
4, 4, 4
```

Expected cap sequence:

```text
2, 2, 4
```

### T-HOLD-002 — decrease is immediate

Initial:

```text
cap=4
```

Fresh selected max:

```text
2
```

Expected:

```text
cap=2 immediately
worker_k<=2
```

This test specifically rejects the historical symmetric-hold behavior.

### T-HOLD-003 — all-zero is immediate

Initial:

```text
cap=4
```

Fresh selections:

```text
A=0, B=0
```

Expected:

```text
cap=0
worker_k=0
pending increase cleared
```

### T-HOLD-004 — mixed rows cannot be raised

Initial cap may be 4.

Fresh:

```text
A selected=0
B selected=2
```

Expected:

```text
A applied=0
B applied=2
worker_k=2
```

Forbidden:

```text
worker_k=4
A applied>0
B applied>2
```

### T-HOLD-005 — request-specific cap invariant

Property test across all combinations:

```text
selected_i ∈ {0,2,3,4}
cap ∈ {0,2,3,4}
```

Assert:

```text
applied_i == min(selected_i, cap)
applied_i <= selected_i
```

### T-HOLD-006 — empty cohort resets batch state

Sequence:

```text
cap=4
active requests finish
next plan has no eligible decode rows
new unrelated cohort arrives selecting max=2
```

Expected:

```text
old cap/pending streak does not force new cohort
new worker_k <= 2
```

### T-HOLD-007 — pending target change resets streak

Sequence:

```text
cap=2
fresh max: 4, 3, 4
hold_steps=3
```

Expected:

```text
no promotion until same target satisfies defined consecutive rule
```

### T-HOLD-008 — zero interrupts pending increase

Sequence:

```text
cap=2
fresh max: 4, 4, 0, 4, 4, 4
```

Expected:

```text
zero applied immediately
pending prior increase cleared
later 4 must earn a fresh streak
```

---

## 6. L2 — lifecycle tests

### T-LIFE-001 — request finish resets request-local policy state

Assert reset of:

```text
warmup count
EMA
fallback/probe counters
reason histogram
```

### T-LIFE-002 — planner state resets on empty cohort

Assert:

```text
recovery cap
pending target
pending streak
```

reset according to spec.

### T-LIFE-003 — preemption does not misroute feedback

Old producer feedback after preemption:

```text
accepted only if lineage says consumed origin remains valid
otherwise skipped
```

### T-LIFE-004 — request ID reuse

Old feedback for:

```text
req-A/incarnation-1
```

must not update:

```text
req-A/incarnation-2
```

---

## 7. L3 — scheduler / worker / verifier integration tests

### T-ACT-001 — selected → applied → worker → verified chain

For each request, capture:

```text
selected_k
applied_k
worker_k
verified_width
```

Assert:

```text
verified_width_i <= applied_k_i <= selected_k_i
worker_k == max(applied_k_i)
```

If physical padded drafting computes shared width `worker_k`, record that separately.

### T-ACT-002 — mixed `(0,2)` case

Fresh selections:

```text
A=0
B=2
```

Assert:

```text
worker_k=2
A verified width=0
B verified width<=2
A receives no draft feedback from this step
B feedback attributed correctly
```

### T-ACT-003 — downward transition from 4 to 2

Previous step:

```text
worker_k=4
```

Current fresh selected max:

```text
2
```

Assert current:

```text
worker_k=2
```

No stale held 4.

### T-ACT-004 — memory zero during previous high K

Previous:

```text
worker_k=4
```

Current:

```text
memory guard forces every request K=0
```

Assert:

```text
worker_k=0 immediately
no new drafts for current step
old in-flight feedback remains bound to old origin only
```

### T-ACT-005 — recovery after memory cooldown

After valid cooldown/recovery:

```text
selected K becomes positive
```

If upward hold is configured, assert:

```text
applied K follows cap
never exceeds selected K
```

### T-ACT-006 — reordered rows

Scheduler row order differs from planner order.

Assert all:

```text
selected/applied K
lineage
feedback destination
```

remain request-correct.

### T-ACT-007 — multiple in-flight producers

Two producer origins for same request are in flight.

Assert feedback binds to the consumed origin and uses that origin's:

```text
plan_id
schedule_id
applied_k
worker_k
```

not the latest controller state.

### T-ACT-008 — duplicate feedback fail-closed

Duplicate verification event:

```text
no second EMA update
no second warmup increment
telemetry records skip
```

---

## 8. L3 — telemetry contract tests

### T-TEL-001 — decision fields

Assert presence and correct meaning of:

```text
selected_k
applied_k
effective_alpha
rule_alpha
warmup_active
verification_sample_count
reason
```

### T-TEL-002 — batch fields

Assert:

```text
fresh_max_k
recovery_cap_k
worker_k
eligible_batch_size
pending_target_k
pending_streak
```

when applicable.

### T-TEL-003 — feedback fields

Assert:

```text
request_id
incarnation
producer_plan_id
producer_schedule_id
drafted
accepted
feedback_counted_for_warmup
```

### T-TEL-004 — histogram semantics

If histograms are included, assert distinct counters for:

```text
selected K
applied K
worker K
reason
```

Do not accept one ambiguous `batch_k_counts` field.

---

## 9. Property-based / exhaustive small-state checks

For candidate set:

```text
{0,2,3,4}
```

enumerate all two-request pairs:

```text
16 combinations
```

for caps:

```text
{0,2,3,4}
```

Assert:

```text
applied_A <= selected_A
applied_B <= selected_B
worker_k == max(applied_A, applied_B)
```

Repeat for memory-forced zero.

This is a cheap invariant test and should run without GPU.

---

## 10. Regression tests to preserve from baseline

At minimum rerun relevant baseline suites covering:

```text
controller thresholds / EMA
fallback / probe
memory pressure rule
budget planner
adaptive scheduler glue
lineage startup
request isolation
stale feedback
finished request cleanup
request ID reuse
row trimming
captured upstream source gates
```

Do not delete or weaken baseline assertions to make the replacement pass.

---

## 11. Test-first implementation order

Required order:

```text
1. Add failing controller warmup tests
2. Add failing safety/floor tests
3. Add failing planner safe-cap tests
4. Add failing lifecycle tests
5. Add failing scheduler/lineage integration tests
6. Review failures
7. Only then modify production code
8. Rerun focused suites
9. Run full relevant baseline regression set
10. Produce before/after evidence table
```

Do not write production code before steps 1–5 are reviewed.

---

## 12. CPU acceptance gate

Before any GPU run:

```text
0 failed in all new contract tests
0 regressions in relevant baseline tests
no ignored failure converted into pass
no network-dependent test used as policy proof
```

Also produce:

| Invariant | Test IDs | Result |
|---|---|---|
| I1–I3 warmup evidence | T-WARM-* | PASS/FAIL |
| I4–I5 safety priority | T-FLOOR-* | PASS/FAIL |
| I6 rule alpha telemetry | T-WARM-006, T-TEL-* | PASS/FAIL |
| I7 row cap | T-HOLD-*, T-ACT-* | PASS/FAIL |
| I8 worker K | T-HOLD-*, T-ACT-* | PASS/FAIL |
| I9 zero immediate | T-HOLD-003, T-ACT-004 | PASS/FAIL |
| I10 decrease immediate | T-HOLD-002, T-ACT-003 | PASS/FAIL |
| I11 increase-only hold | T-HOLD-001/007/008 | PASS/FAIL |
| I12 cohort reset | T-HOLD-006, T-LIFE-002 | PASS/FAIL |
| I13 request reuse | T-WARM-007, T-LIFE-004 | PASS/FAIL |
| I14 lineage binding | T-ACT-006/007/008 | PASS/FAIL |
| I15 provenance | report check | PASS/FAIL |
| I16 baseline interfaces | regression suite | PASS/FAIL |

Any FAIL:

```text
NOT READY FOR GPU
```

---

## 13. L4 — bounded GPU validation after CPU closure

GPU is allowed only after CPU gate passes.

This is not a performance benchmark.

### GPU-G1 — warmup evidence

One request with controlled output long enough to produce at least:

```text
min_warmup_samples + 2
```

real positive-width verifier events.

Capture:

```text
verification_sample_count
effective_alpha
rule_alpha
selected_k
drafted
accepted
```

### GPU-G2 — dynamic transition

Exercise at least:

```text
K=3/4 → K=2 → K=0 → recovery
```

if the policy/environment can deterministically trigger them.

### GPU-G3 — memory fallback/recovery

Trigger bounded memory/KV guard if safe and feasible.

Need trace:

```text
normal
→ pressure
→ K=0
→ cooldown
→ recovery
```

### GPU-G4 — mixed effective runtime batch

Require measured:

```text
effective runtime batch > 1
```

not just client concurrency.

Need at least one step with heterogeneous request selections if feasible.

Capture:

```text
selected_k_i
applied_k_i
worker_k
actual proposal width
verified width
feedback destination
```

### GPU-G5 — token correctness control

Run matched control(s) under the agreed lossless protocol.

Do not claim performance.

---

## 14. GPU acceptance gate

Replacement may be classified:

```text
READY FOR NEXT VALIDATION PHASE
```

only if:

```text
no lineage mismatch
no applied_k > selected_k
no worker_k above current max(applied_k_i)
K=0 immediate when all selected zero
warmup samples match real verifier feedback count
no stale/duplicate feedback updates controller
token correctness protocol has no new unexplained regression
```

Otherwise:

```text
NOT READY — CORRECTNESS BLOCKER
```

---

## 15. Required test report

Codex must produce:

| Test ID | Layer | Behavior | Result | Evidence | Notes |
|---|---|---|---|---|---|

And a final invariant table:

| Invariant | PASS/FAIL | Evidence |
|---|---|---|

No prose-only "looks good" conclusion is acceptable.

---

## 16. Stop condition

For the test-design phase:

```text
STOP after tests/spec are implemented or drafted and reviewed.
```

Do not:

```text
run performance benchmark
change experiment protocol
redesign R6 hardware-cost selection
merge/rebase main
modify Trello
cherry-pick 59f18a9
cherry-pick 0d4a9fa
```
