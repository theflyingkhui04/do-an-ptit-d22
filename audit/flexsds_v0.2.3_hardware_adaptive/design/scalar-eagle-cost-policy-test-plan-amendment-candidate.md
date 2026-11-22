# Scalar EAGLE Hardware-Cost Policy — Amendment Candidate Test Plan

> **Active migration (2026-11-22): SECP-PULSE-2R-v1 applied.** The user authorized this scoped production/CPU migration. The appended P2R clauses supersede earlier candidate text for the B1/base3/old0/[0,256) pulse domain. Earlier fixtures/claims remain original-contract context, not evidence for the new objective. No GPU, campaign, publication, push or benchmark is authorized by this application.


**Document type:** Verification and acceptance plan  
**Status:** `AMENDMENT CANDIDATE — NOT APPROVED`  
**Policy under test:** `SECP-v1.1-candidate`  
**Consumer-profile protocol:** `EAGLE-RW-v1.1-candidate`  
**Audited local source:** `4bb093ad4272aa330679d3baa9f5b1797ea397da`  
**Pinned runtime:** `vLLM 0.30.0`  
**Date:** 2026-11-21  
**Parent test-plan SHA-256:** `ec71b0da0adb73b685a665040e02253f59b4f0200a2f3dbdf90d178060b3b9dc`

> Passing tests from this plan does not itself prove GPU performance. The plan separates contract review, CPU causal closure, live runtime validation and performance benchmarking.

---

## 1. Purpose

This plan verifies the SECP-v1.1 amendment candidate and specifically closes:

- A — survival/history/selection bias;
- B — uncertainty/ties/stopping;
- C — DecisionStateKeyV1 feasibility and matching;
- D — off/strict/compat/cost-warmup/probe state machine;
- E — two-phase verifier feedback finalization.

Primary principle:

```text
test the active path, not merely helper availability
```

A profile loading successfully is not R6 evidence. A `get_cost()` call whose return value is ignored is not R6 evidence.

---

## 2. Test layers

| Layer | Purpose | GPU? |
|---|---|---:|
| G0 review fixtures | contract unambiguity | no |
| L1 math/estimator | survival/yield/counterexamples | no |
| L2 pending-feedback lifecycle | pre-stop/finalized eligibility | no |
| L3 identity/profile/state key | exact admission rules | no |
| L4 policy/state machine | strict/compat/warmup/probe | no |
| L5 planner causal integration | active measured-cost consumption | no |
| L6 regression | preserve current invariants | no |
| L7 collector fidelity | consumer-profile measurement validity | bounded after CPU |
| L8 live R6-GPU | real profile consumption + lineage | yes |
| L9 performance | TPOT/throughput/etc. | yes, separate protocol |

---

# PART I — CANONICAL FIXTURES

## 3. F-A01 — adaptive-D selection-bias counterexample

Underlying latent population, equal probability:

```text
low:  selected D=2, A=0
high: selected D=4, A=4
```

Wrong variable-D estimator:

```text
S1=.5
S2=.5
S3=1
S4=1
Y4=4
```

Counterfactual true K4 useful yield for equal latent population:

```text
low useful=1
high useful=5
E=3
```

Expected: naive `D>=j` pooling is rejected.

---

## 4. F-A02 — same-base fully-exposed cohort

Input:

```text
e1: base4, normal, finalized nonterminal, D4,A0
e2: base4, normal, finalized nonterminal, D4,A4
e3: base2, normal, finalized nonterminal, D2,A0
e4: base4, normal, finalized nonterminal, D2,A2   # clipped
e5: base2, probe,  finalized nonterminal, D2,A2
```

Expected K4 normal cohort:

```text
[e1,e2]
```

Expected:

```text
S=[.5,.5,.5,.5]
Y4=3
```

---

## 5. F-A03 — zero accept is valid

```text
base4
normal
finalized nonterminal
D=4
A=0
```

Expected:

```text
one K4 cohort event
survival contribution [0,0,0,0]
```

---

## 6. F-A04 — D0 is not positive-width evidence

```text
D=0,A=0
```

Expected: no positive survival-stratum update.

---

## 7. F-E01 — terminal censoring

Two events:

```text
event N:
  A=2,D=4
  finalized request remains active

event T:
  A=2,D=4
  finalized terminal due EOS/length
```

Expected:

```text
N -> ELIGIBLE_NONTERMINAL if all other predicates pass
T -> TERMINAL_CENSORED
```

---

## 8. F-E02 — future schedule is not terminal proof

Trace:

```text
verifier output arrives
future schedule already prequeued
final stop processing marks request terminal
```

Expected: future schedule does not commit the pending sample.

---

## 9. F-E03 — multiple outstanding async events

Pending:

```text
(req7,epoch3,plan101,schedule201)
(req7,epoch3,plan102,schedule202)
```

Finalize event 201.

Expected:

```text
only event201 transitions
event202 remains pending
```

---

## 10. F-E04 — request ID reuse

Old pending:

```text
req7 epoch3
```

New request:

```text
req7 epoch4
```

Expected: epoch4 cannot consume/finalize epoch3 state.

---

## 11. F-C01 — heterogeneous width-vector distinction

```text
State A:
  B=2
  old widths=[4,4]
  future candidate=[4,4]

State B:
  B=2
  old widths=[4,0]
  future candidate=[4,4]
```

Expected:

```text
DecisionStateKey(A) != DecisionStateKey(B)
```

---

## 12. F-C02 — same B/maxK is insufficient

Provide:

```text
[4,4]
[4,0]
[4,2]
```

All may share B=2 and maxK=4.

Expected: distinct decision keys unless a later frozen rule proves a dimension irrelevant.

---

## 13. F-C03 — later physical variation does not rewrite forecast

Two executions share identical DecisionStateKey but later show different physical padding.

Expected:

```text
same decision key
distinct measured physical-shape telemetry
```

Cell remains publishable only if uncertainty still passes.

---

## 14. F-B01 — cost uncertainty supports K2

```text
Y2=3.0
C2=[1.9,2.2]

Y0=1.0
C0=[0.9,1.0]
```

Expected:

```text
R_lower2=3/2.2
R_upper0=1/0.9
R_lower2 > R_upper0
```

K2 may be cost-supported.

---

## 15. F-B02 — overlapping cost interval is inconclusive

```text
Y2=3.0
C2=[1.9,2.9]

Y0=1.0
C0=[0.9,1.0]
```

Expected:

```text
R_lower2 <= R_upper0
```

K2 MUST NOT be called cost-supported.

---

## 16. F-B03 — cluster bootstrap unit

Synthetic data:

```text
50 cohort IDs
8 inner retirements/cohort
```

Expected: bootstrap resamples cohort IDs and preserves all inner records of each selected cohort.

Per-retirement IID bootstrap is a failing negative control.

---

## 17. F-D01 — strict cost warmup

```text
mode=strict
profile valid
decision state supported
base K4
K4 survival stratum 7/10
no safety override
normal base mode
```

Expected:

```text
decision_mode=cost_warmup
cost_policy_applied=false
final behavior follows current base/replacement path
```

---

## 18. F-D02 — strict missing profile

Expected:

```text
K0
cost_policy_applied=false
reason=missing_profile
```

---

## 19. F-D03 — compat missing profile

Expected:

```text
legacy base/replacement behavior
cost_policy_applied=false
hardware_cost_claim_allowed=false
```

---

## 20. F-D04 — memory dominates

Profile strongly favors K4; memory/KV guard fires.

Expected:

```text
K0
decision_mode=memory_override
```

---

## 21. F-D05 — probe is exploration

Existing controller emits K2 probe.

Expected:

```text
decision_mode=probe
cost_policy_applied=false
cost_optimal=false
probe excluded from normal survival
```

---

## 22. F-D06 — lower cost cap defeats stale upward hold

Previous worker K4, new base K4, cost cap K2.

Expected:

```text
applied<=2
worker<=2 immediately
```

---

## 23. F-K0-01 — same-engine zero-width

Pinned/source fixture proves K0:

```text
executes EAGLE forward/cache synchronization
returns width zero
is not target-only vanilla
```

---

## 24. F-CAUSAL-01 — active cost-surface flip

Non-cost inputs fixed:

```text
requests=[A,B]
K_base=[4,2]
A survival=[.90,.75,.55,.35]
B survival=[.80,.50]
candidate caps=[0,2,3,4]
```

Predicted yields:

```text
q0=2.00
q2=4.95
q3=5.50
q4=5.85
```

Define two exact DecisionStateKey-compatible synthetic consumer profiles:

```text
surface A -> conservative selection q2
surface B -> conservative selection q4
```

Only cost values may differ.

Expected active planner result:

```text
surface A -> final governed cap 2
surface B -> final governed cap 4
```

---

# PART II — SURVIVAL TESTS

## 25. Recommended file

```text
tests/test_scalar_eagle_survival.py
```

| ID | Requirement(s) | Test |
|---|---|---|
| MATH-A001 | R009 | naive adaptive-D counterexample |
| MATH-A002 | R010 | common denominator same-base cohort |
| MATH-A003 | R011 | no depth above base width |
| MATH-A004 | R012 | request-incarnation isolation |
| MATH-A005 | R013 | base-width strata isolation |
| MATH-A006 | R014 | clipped D<base excluded |
| MATH-A007 | R015 | A0,Dpositive valid |
| MATH-A008 | R016 | D0 excluded |
| MATH-A009 | R017 | probe isolated |
| MATH-A010 | R018-R020 | fixed window readiness |
| MATH-A011 | R021 | yield formula |
| MATH-A012 | R022 | predicted vs actual distinction |

Properties:

- adding a valid failure event cannot increase any survival depth in that stratum;
- adding a valid full-accept event cannot decrease survival;
- another base-width stratum cannot alter the current stratum;
- another incarnation cannot alter current request state;
- probe samples never change normal-stratum counts.

---

# PART III — PENDING FEEDBACK TESTS

## 26. Recommended file

```text
tests/test_scalar_eagle_pending_feedback.py
```

| ID | Requirement(s) | Test |
|---|---|---|
| PEND-001 | R023 | pre-stop feedback creates pending only |
| PEND-002 | R024-R025 | nonterminal finalization commits |
| PEND-003 | R026 | terminal finalization censors |
| PEND-004 | R027 | future schedule not proof |
| PEND-005 | R028 | key includes incarnation/origin |
| PEND-006 | R029 | ambiguous origin invalid |
| PEND-007 | R030 | ID reuse cannot consume old pending |
| PEND-008 | R031 | actual emitted optional online but required validation |
| PEND-009 | R032 | project-local feasibility source test |
| PEND-010 | R078 | duplicate finalization rejected |

Required traces:

1. normal single outstanding event;
2. two outstanding events same request;
3. terminal after prequeued future schedule;
4. preemption/resume;
5. stale producer;
6. request-ID reuse.

If current hooks cannot unambiguously express these, G1 fails.

---

# PART IV — COST UNCERTAINTY TESTS

## 27. Recommended files

```text
tests/test_eagle_consumer_cost_statistics.py
tests/test_eagle_consumer_profile_publication.py
```

| ID | Requirement(s) | Test |
|---|---|---|
| STAT-001 | R033 | online survival has no fake IID CI |
| STAT-002 | R034 | bootstrap resamples cohort IDs |
| STAT-003 | R034 | inner retirements stay clustered |
| STAT-004 | R035 | fixed sample plan required |
| STAT-005 | R035 | no collect-until-winner stopping |
| STAT-006 | R036 | failed cohorts preserved |
| STAT-007 | R037 | conservative lower vs K0 upper |
| STAT-008 | R038 | no TPOT/speedup interpretation |
| STAT-009 | R039 | overlap -> INCONCLUSIVE_COST |
| STAT-010 | R040 | max conservative supported candidate |
| STAT-011 | R041 | deterministic smaller-K tie |

Reproducibility fixture records:

```text
seed
resample count
confidence level
cohort IDs
raw fixture hash
```

Same input/seed -> same interval.

---

# PART V — DECISION-STATE KEY TESTS

## 28. Recommended file

```text
tests/test_eagle_decision_state_key.py
```

| ID | Requirement(s) | Test |
|---|---|---|
| SHAPE-001 | R042 | only decision-time fields permitted |
| SHAPE-002 | R042 | worker-only field cannot be backfilled |
| SHAPE-003 | R048 | forecast/observed telemetry distinct |
| SHAPE-004 | R048 | residual physical variation retained |
| SHAPE-005 | R049 | no nearest-B fallback |
| SHAPE-006 | R049 | no nearest-context fallback |
| SHAPE-007 | R046 | [4,4] != [4,0] |
| SHAPE-008 | R046 | [4,4] != [4,2] |
| SHAPE-009 | R050 | required key fields available or fail G1 |
| SHAPE-010 | R047,R050 | missing context transport -> alignment gate |
| SHAPE-011 | R068 | matching K0 state required |

Source feasibility artifact:

```text
field
available before decision?
source path
semantics
forecast/measured
```

Every required field needs an exact source.

---

# PART VI — MODE / STATE MACHINE TESTS

## 29. Recommended file

```text
tests/test_scalar_eagle_cost_modes.py
```

| ID | Requirement(s) | Test |
|---|---|---|
| MODE-001 | R051 | ADAPTIVE_STRICT not reused |
| MODE-002 | R052 | default off |
| MODE-003 | R053 | memory strict -> K0 |
| MODE-004 | R053 | memory compat -> K0 |
| MODE-005 | R054 | off unchanged |
| MODE-006 | R055 | strict cost warmup |
| MODE-007 | R056 | probe explicit exploration |
| MODE-008 | R057 | strict missing profile |
| MODE-009 | R057 | strict mismatch |
| MODE-010 | R057 | strict unsupported state |
| MODE-011 | R057 | strict inconclusive cost |
| MODE-012 | R058 | compat fallback |
| MODE-013 | R059 | cost cap beats performance floor |
| MODE-014 | R060 | upward hold only |
| MODE-015 | R075 | compat no hardware claim |
| MODE-016 | R076 | probe no optimality claim |

---

# PART VII — IDENTITY / PROFILE TESTS

## 30. Recommended file

```text
tests/test_eagle_consumer_profile_identity.py
```

| ID | Requirement(s) | Test |
|---|---|---|
| PROF-001 | R061 | requested A10G label alone insufficient |
| PROF-002 | R061 | explicit reviewed alias required |
| PROF-003 | R062 | L4 rejected for A10 |
| PROF-004 | R063 | missing field stays unknown |
| PROF-005 | R064 | component profile diagnostic only |
| PROF-006 | R065 | timing clock-domain validation |
| PROF-007 | R066 | component sum cannot become canonical cost |
| PROF-008 | R067 | collector no added sync |
| PROF-009 | R068 | matched K0 required |

---

# PART VIII — ACTIVE PLANNER CAUSAL TESTS

## 31. Recommended file

```text
tests/test_budget_planner_scalar_eagle_cost.py
```

### INT-CAUSAL-001

Arrange identical:

- request IDs/incarnations;
- current controller state;
- K_base;
- memory/KV state;
- pending/survival state;
- DecisionStateKey;
- replacement state;
- policy mode;
- profile identity.

Only swap cost surface A/B.

Assert:

```text
A -> final cap2
B -> final cap4
profile lookup count > 0
cost values enter score computation
final planner output differs
telemetry reconstructs both decisions
```

Fail if:

- only helper output changes;
- profile logged but ignored;
- return value discarded;
- parent selector replaces current controller;
- K raised above K_base.

### INT-SAFE-002

Profile favors K4; memory/KV guard active -> K0.

### INT-TRANS-003

Previous K4; new valid cost cap2 -> immediate <=2 worker.

### INT-WARM-004

Valid profile but survival underfilled -> cost_warmup, no cost claim.

### INT-COMPAT-005

Profile mismatch in compat -> legacy output, `cost_policy_applied=false`.

### INT-SHAPE-006

Batch count matches but state key does not -> unsupported, no nearest lookup.

---

# PART IX — NEGATIVE CONTROLS

## 32. Superficial fixes that MUST fail

1. profile-load-only;
2. `get_cost()` called but result ignored;
3. legacy parent-selector shortcut;
4. A/D used as geometric alpha;
5. naive variable-D survival;
6. batch-size-only profile key;
7. max-K-only profile key;
8. nearest-B lookup;
9. target-only K0;
10. pending keyed only by request ID;
11. future schedule used as terminal proof;
12. per-retirement IID bootstrap;
13. strict K0 forever during warmup;
14. cost cap raised again by performance floor;
15. probe counted as normal cost-optimal evidence;
16. compat fallback labelled hardware-aware;
17. profile cost written into actual live-latency field.

---

# PART X — TELEMETRY TESTS

## 33. Recommended file

```text
tests/test_scalar_eagle_cost_telemetry.py
```

Required reconstruction fields:

- policy/profile versions;
- profile identity state;
- DecisionStateKey;
- K_base;
- candidate vectors;
- survival stratum snapshot;
- yields;
- cost intervals;
- conservative rates;
- selected cap;
- decision mode;
- `cost_policy_applied`;
- replacement/recovery;
- planned worker width;
- later physical width;
- pending/finalized feedback origin.

### TEL-ROUNDTRIP-001

Telemetry + immutable profile cell + survival snapshot reconstruct the same selected cap.

### TEL-NOFAKE-002

Profile cost cannot populate `actual_step_latency_ms`.

### TEL-MODE-003

Probe/compat/warmup cannot report `cost_optimal=true`.

---

# PART XI — REGRESSION INVARIANTS

## 34. Existing behavior to protect

Rerun suites protecting:

- attributed positive-width warmup;
- A0,Dpositive validity;
- D0 no positive evidence;
- memory/KV priority;
- applied<=selected/base;
- worker=max applied;
- immediate all-zero/downward;
- upward-only hold;
- request-incarnation isolation;
- empty-cohort isolation;
- consumed-origin feedback;
- planned vs physical K distinction.

Historical 190/236/162 counts are evidence records, not future expected exact counts.

Any unexplained regression blocks R6-CPU.

---

# PART XII — COLLECTOR FIDELITY

## 35. Consumer-profile collector tests

Only after G0/G1/G2/G3.

| ID | Test |
|---|---|
| COL-001 | same-process monotonic clock |
| COL-002 | exact retirement count |
| COL-003 | exact useful-token slicing |
| COL-004 | no canonical CUDA synchronize |
| COL-005 | observer trajectory neutrality |
| COL-006 | K0 same-engine behavior |
| COL-007 | transition windows excluded |
| COL-008 | timeout/OOM/abort preserved |
| COL-009 | cohort IDs are replicate unit |
| COL-010 | raw -> published interval reproducible |
| COL-011 | DecisionStateKey attached before action |
| COL-012 | physical state attached later/separately |

---

# PART XIII — GPU AUTHORIZATION

## 36. Gate

No GPU until:

```text
G0 human freeze PASS
G1 DecisionStateKey/PendingFeedback feasibility PASS
G2 causal planner CPU PASS
G3 relevant CPU regression PASS
G4 collector fidelity PASS
```

First GPU work is bounded R6-GPU correctness evidence, not benchmark.

---

# PART XIV — LIVE R6-GPU

## 37. Required live evidence

At least one normal decision must show:

```text
matching observed runtime identity
matching EAGLE-RW profile
DecisionStateKey valid
cost_policy_applied=true
cost values/rates reconstruct
K_applied<=K_base
worker=max(applied)
physical width observed
feedback finalized to correct origin/incarnation
```

A second scenario demonstrates K0 from cost or independent safety.

No speedup claim.

---

# PART XV — PERFORMANCE

## 38. Separate benchmark gate

Only after R6-GPU.

Required measured dimensions include:

- TTFT;
- TPOT/ITL under frozen definitions;
- E2E;
- throughput;
- VRAM;
- errors/OOM/timeouts;
- actual runtime batch/state;
- selected/applied/physical K;
- repeats and variability.

Predicted useful-tokens/ms is not measured TPOT.

Negative/neutral result is valid.

---

# PART XVI — TRACEABILITY

## 39. Requirement map

| Requirement | Primary verification |
|---|---|
| R001-R002 | F-K0-01, PROF-009 |
| R003 | INT-SAFE-002 |
| R004 | base-bound property, INT-CAUSAL-001 |
| R005 | batch-shared negative control |
| R006 | existing regression suite |
| R007 | INT-TRANS-003 |
| R008 | upward-recovery regression |
| R009 | MATH-A001 |
| R010 | MATH-A002 |
| R011 | no-above-base test |
| R012 | MATH-A004 |
| R013 | MATH-A005 |
| R014 | MATH-A006 |
| R015 | MATH-A007 |
| R016 | MATH-A008 |
| R017 | MATH-A009 |
| R018-R020 | MATH-A010 |
| R021 | MATH-A011 |
| R022 | predicted-vs-actual validation |
| R023 | PEND-001 |
| R024-R025 | PEND-002 |
| R026 | PEND-003 |
| R027 | PEND-004 |
| R028 | PEND-005 |
| R029 | PEND-006 |
| R030 | PEND-007 |
| R031 | PEND-008 |
| R032 | PEND-009 |
| R033 | STAT-001 |
| R034 | STAT-002, STAT-003 |
| R035 | STAT-004, STAT-005 |
| R036 | STAT-006 |
| R037 | STAT-007 |
| R038 | STAT-008 |
| R039 | STAT-009 |
| R040 | STAT-010 |
| R041 | STAT-011 |
| R042 | SHAPE-001, SHAPE-002 |
| R043 | pure-decode state tests |
| R044 | batch-vs-client negative control |
| R045 | old-vs-future width tests |
| R046 | SHAPE-007, SHAPE-008 |
| R047 | SHAPE-010 |
| R048 | SHAPE-003, SHAPE-004 |
| R049 | SHAPE-005, SHAPE-006 |
| R050 | SHAPE-009, SHAPE-010 |
| R051 | MODE-001 |
| R052 | MODE-002 |
| R053 | MODE-003, MODE-004 |
| R054 | MODE-005 |
| R055 | MODE-006 |
| R056 | MODE-007 |
| R057 | MODE-008..011 |
| R058 | MODE-012 |
| R059 | MODE-013 |
| R060 | MODE-014 |
| R061 | PROF-001, PROF-002 |
| R062 | PROF-003 |
| R063 | PROF-004 |
| R064 | PROF-005 |
| R065 | PROF-006 |
| R066 | PROF-007 |
| R067 | PROF-008 |
| R068 | PROF-009 |
| R069-R072 | INT-CAUSAL-001 + score tests |
| R073 | TEL-NOFAKE-002 |
| R074 | SHAPE-003 + telemetry |
| R075 | MODE-015 |
| R076 | MODE-016 |
| R077 | lifecycle/pending isolation |
| R078 | PEND-010 |

Traceability gate:

```text
every normative requirement -> test/review/runtime evidence
every test expectation -> normative requirement
```

---

# PART XVII — G0 REVIEW CHECKLIST

## 40. Before freeze reviewers answer YES

- [ ] same-base fully-exposed survival approved;
- [ ] request-incarnation/base-width history approved;
- [ ] rolling window policy approved;
- [ ] probe isolation approved;
- [ ] two-phase pending feedback approved;
- [ ] future schedule rejected as terminal proof;
- [ ] cohort-level bootstrap approved;
- [ ] fixed sample plan approved;
- [ ] conservative cost-bound decision approved;
- [ ] DecisionStateKeyV1 approved;
- [ ] exact/no-nearest-state behavior approved;
- [ ] off/strict/compat modes approved;
- [ ] cost_warmup approved;
- [ ] cost-cap/floor/recovery precedence approved;
- [ ] K0 same-engine semantics approved;
- [ ] G1 feasibility criteria approved;
- [ ] R001-R078 traceability complete.

If any answer is NO:

```text
SPEC FROZEN = NO
TEST PLAN APPROVED = NO
```

---

## 41. Exactly one next action

**Review this test plan together with the SECP-v1.1 amendment candidate and freeze both only after the G1 source-feasibility criteria are accepted.**


# Applied pulse verification plan

The complete normative P2R-01 through P2R-20 text is in the companion active specification. Required production test IDs are T01 through T32 in `tests/test_pulse_cycle_policy.py`. Historical 704 cases remain regression evidence for their explicitly declared original contracts; active pulse closure requires the new B1/base3 witness. Publication tests use synthetic CPU fixtures only.

| Clause | Implementation | Test IDs |
|---|---|---|
| P2R-01 | src/proposer/pulse_policy.py:PulseCostGovernor | T01,T02,T03,T28 |
| P2R-02 | src/server/decision_state.py:PulseContextV1; eagle_consumer_profile.py:pulse_state_payload | T15,T27,T31 |
| P2R-03 | src/server/budget_planner.py:plan_step; pulse_policy.py:PulseCycleLease | T16,T17,T18,T19,T32 |
| P2R-04 | pulse_policy.py:PulseCostGovernor; PulseCycleLease | T01,T05,T16,T19 |
| P2R-05 | pulse_policy.py:PulseCycleLease.__init__/retire | T20,T21,T28,T32 |
| P2R-06 | pulse_policy.py:PulseCycleLease.confirm/verifier/retire | T20,T21,T22,T23,T24 |
| P2R-07 | pulse_policy.py:PulseCycleLease.retire; adaptive_scheduler.py:update_from_output | T20,T21,T28,T32 |
| P2R-08 | eagle_consumer_profile.py:EaglePulseProfile/PulseCostCell/publish_pulse_cost | T12,T13,T14,T28 |
| P2R-09 | pulse_policy.py:PulseCycleLease.retire | T23,T28 |
| P2R-10 | src/proposer/cost_policy.py:SurvivalEstimate.predicted_cycle_reward | T01,T02,T03 |
| P2R-11 | pulse_policy.py:PulseSurvivalHistory; budget_planner.py:_emit_finalized_cost_feedback | T26,T32 |
| P2R-12 | decision_state.py:capture_pulse_context/PulseContextV1.validate | T24,T31 |
| P2R-13 | pulse_policy.py:PulseCycleLease.invalidate/retire | T24,T25 |
| P2R-14 | budget_planner.py:plan_step; lineage.py:clear; lineage_executor.py:execute_model/sample_tokens | T25,T29,T30 |
| P2R-15 | experiments/eagle_rw_collector.py:EaglePulseCycleCollector; budget_planner.py:finish_request | T26,T28 |
| P2R-16 | eagle_consumer_profile.py:publish_pulse_cost/PulseCostCell | T25,T28 |
| P2R-17 | eagle_consumer_profile.py:EaglePulseProfile/pulse_state_payload | T12,T13,T14,T15,T27,T31 |
| P2R-18 | pulse_policy.py:PulseCostGovernor.decide | T04,T05,T06,T07,T08,T09,T10,T11 |
| P2R-19 | pulse_policy.py:PulseCycleLease; adaptive_scheduler.py:update_from_output | T16,T17,T18,T19,T20,T21,T32 |
| P2R-20 | budget_planner.py:__init__/from_env; tests/scalar_eagle_fixtures.py | T09,T10,T11,T12,T13,T14,T29,T30,T31,T32 |
