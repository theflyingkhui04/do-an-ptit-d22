# Scalar EAGLE Hardware-Cost Policy Specification — Amendment Candidate

> **Active migration (2026-11-22): SECP-PULSE-2R-v1 applied.** The user authorized this scoped production/CPU migration. The appended P2R clauses supersede earlier candidate text for the B1/base3/old0/[0,256) pulse domain. Earlier fixtures/claims remain original-contract context, not evidence for the new objective. No GPU, campaign, publication, push or benchmark is authorized by this application.


**Document type:** Normative technical specification candidate  
**Status:** `AMENDMENT CANDIDATE — TECHNICAL REVIEW REQUIRED — NOT FROZEN`  
**Policy ID:** `SECP-v1.1-candidate`  
**Parent policy:** `SECP-v1`  
**Consumer-profile protocol:** `EAGLE-RW-v1.1-candidate`  
**Project:** `theflyingkhui04/do-an-ptit-d22`  
**Audited local source:** `4bb093ad4272aa330679d3baa9f5b1797ea397da`  
**Historical baseline:** `8006205a6f19790a776d67155a377def053d2736`  
**Pinned serving dependency:** `vLLM 0.30.0`  
**Review date:** 2026-11-21 (Asia/Bangkok)  
**Parent spec SHA-256:** `02d079ecceb2bfdae119d408648c514dde10869f93d94f1dce2e1173b4764774`  
**Parent test-plan SHA-256:** `ec71b0da0adb73b685a665040e02253f59b4f0200a2f3dbdf90d178060b3b9dc`

> **Normative status.** This document is a review candidate. It resolves the five previously underspecified G0 areas with concrete proposed rules, but it is **not frozen**, does **not** authorize production implementation, GPU calibration, live serving validation, or performance benchmarking. Human technical approval is still required.

---

## 1. Purpose

SECP-v1.1-candidate defines a scientifically defensible scalar EAGLE/EAGLE-3 hardware-cost governor for FlexSDS.

The governor consumes a measured consumer-cost profile only when:

1. hard memory/KV safety permits speculation;
2. the existing project controller has already produced a per-request base width;
3. the decision state is representable by an approved, decision-observable profile key;
4. sufficient finalized, nonterminal, lineage-valid survival evidence exists;
5. the consumer cost profile is identity-compatible and statistically publishable;
6. the selected cost cap does not exceed the current controller's base decision;
7. existing replacement/downward/upward transition invariants remain intact.

This candidate specifically resolves five G0 blockers:

- **A — survival/history/selection bias**
- **B — uncertainty, ties and stopping**
- **C — decision-state forecasting/profile key**
- **D — strict/compat/warmup/probe state machine**
- **E — feedback finalization and terminal clipping**

The current implementation boundary remains:

```text
R6-SOURCE      = NOT FIXED
R6-CPU         = NOT SATISFIED
R6-GPU         = UNVERIFIED
R6-PERFORMANCE = UNVERIFIED
```

The current active controller still does not query measured costs. This document does not change that fact.

---

## 2. Evidence basis and source hierarchy

### 2.1 Project evidence

The candidate is grounded in the read-only evidence package produced from:

```text
branch: audit/flexsds-v023-hardwareadaptive-20261120
HEAD:   4bb093ad4272aa330679d3baa9f5b1797ea397da
```

Fresh evidence package status:

```text
190 passed
0 failed
0 skipped
0 errors
19 / 19 pinned vLLM files matched their manifests
3,789 protected files unchanged
production diff = NONE
```

Important project facts from that package:

- attributed `(A,D)` feedback reaches FlexSDS before final stop/length clipping;
- current planner state can only partially forecast future physical worker shape;
- padded EAGLE uses a shared K; `K_i=0` does not prove proposal compute is physically eliminated;
- K0 retains drafter forward/cache work and returns zero-width draft output;
- profile/component calibration and production async serving do not yet share a validated consumer-cost contract;
- existing memory/replacement/lineage invariants remain the hard compatibility boundary.

### 2.2 Pinned runtime authority

For runtime/API behavior:

```text
pinned vLLM 0.30.0 source/tests
> same-version docs
> release notes
> latest vLLM only as design precedent
```

Latest vLLM Dynamic SD and Adaptive Verification MUST NOT be used as API proof for pinned vLLM 0.30.0.

### 2.3 Research constraints

#### RF-01 — acceptance and execution cost are separate variables

Leviathan, Kalman and Matias model speculative acceptance and hardware/software cost separately. Their geometric expected-token expression assumes a particular acceptance model; FlexSDS legacy `accepted/drafted` EMA is not proven equivalent.

Primary source:

https://proceedings.mlr.press/v202/leviathan23a.html

#### RF-02 — acceptance is context-dependent

EAGLE-2 reports that draft-token acceptance depends on context as well as position. Therefore pooling adaptively selected draft depths without a conditioning contract can create selection bias.

Primary source:

https://aclanthology.org/2024.emnlp-main.422/

#### RF-03 — verification cost changes with load and K

Current vLLM Dynamic Speculative Decoding documents that effective verification work grows with batch size and speculative width, and K may need to decrease as load rises.

Design precedent only:

https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/

#### RF-04 — per-position survival plus profiled cost is a valid design precedent

Current vLLM Adaptive Verification scores draft slots by survival probability and combines that with a profiled step-cost model.

Design precedent only:

https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/

#### RF-05 — dependent data require cluster-aware uncertainty

When observations are clustered/correlated, treating each inner observation as independent can understate uncertainty. EAGLE-RW publication therefore defines the replicate unit explicitly at the cohort level.

Reference:

https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/

---

## 3. Normative language

**MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, **MAY** are normative.

Evidence labels:

- **FACT** — current source/test/raw evidence.
- **PROPOSAL** — behavior introduced by this candidate.
- **UNKNOWN** — evidence is insufficient.
- **RUNTIME OBSERVATION REQUIRED** — cannot be settled by static/CPU source evidence.
- **TEAM TECHNICAL ALIGNMENT REQUIRED** — shared interface semantics must be approved across owners.

---

## 4. Scope

### 4.1 In scope

This candidate applies only to:

- scalar linear EAGLE/EAGLE-3 speculative width;
- candidate grid `K ∈ {0,2,3,4}`;
- hardware/runtime-aware **upper-bound** governance;
- request-incarnation-local survival history;
- batch-shared consumer cost;
- decision-observable state/profile matching;
- strict/compat/off modes;
- cost warmup and legacy probe compatibility;
- replacement/downward/upward transition compatibility;
- causal CPU closure before GPU work.

### 4.2 Non-goals

This candidate does not define:

- arbitrary speculative trees;
- tree parent/mask/position semantics;
- global per-token batch optimization;
- stochastic distribution-preservation proof;
- nearest-B/context interpolation;
- target-only K0;
- automatic vLLM upgrade;
- performance speedup from predicted scores;
- model training;
- modification of historical raw evidence.

---

## 5. Core definitions

| Symbol / field | Definition |
|---|---|
| `K_base_i` | existing controller width for request `i` before cost governance |
| `q` | candidate shared cost cap |
| `K_i(q)` | governed request width `min(K_base_i, q)` after candidate cap |
| `K_applied_i` | width after all current replacement/transition semantics |
| `K_worker` | `max_i(K_applied_i)` |
| `D` | actual verified draft slots for one attributed verifier event |
| `A` | accepted drafts from the same verifier event |
| `j` | draft depth, one-indexed |
| `H_i,b` | finalized normal-mode survival history for request incarnation `i`, base-width stratum `b` |
| `S_i,b,j` | survival probability estimate at depth `j` from `H_i,b` |
| `Y_i(k)` | predicted nonterminal useful-token advance for request `i` at width `k` |
| `Y_batch(q)` | sum of per-request predicted useful advances under cap `q` |
| `DecisionStateKeyV1` | profile key made only from information available before the governed action is committed |
| `C_point(s,q)` | measured mean/central consumer cost for decision state `s`, ms/retired batch |
| `C_low(s,q)` | lower published cost bound |
| `C_high(s,q)` | upper published cost bound |
| `R_point(q)` | `Y_batch(q)/C_point(s,q)` |
| `R_lower(q)` | `Y_batch(q)/C_high(s,q)` |
| `R_upper(q)` | `Y_batch(q)/C_low(s,q)` |

---

## 6. K0 contract

### SECP-R001 — K0 is same-engine zero-width EAGLE

K0 MUST preserve the same target/EAGLE serving engine, scheduler/executor, cache policy, speculative ceiling and runtime mode defined by the consumer profile.

K0 MUST NOT be labelled vanilla, target-only, or zero-drafter-overhead.

### SECP-R002 — K0 is the canonical in-engine reference

Cost comparisons MUST use the matching K0 consumer-profile cell, not an unrelated target-only engine.

---

## 7. Safety and governor role

### SECP-R003 — safety dominates cost

Memory/KV safety MUST execute before any cost preference.

### SECP-R004 — governor may keep or reduce, never raise

```text
K_governed_i <= K_base_i
```

### SECP-R005 — batch cost is shared

A consumer cost cell is charged once to the whole decision state/batch. It MUST NOT be duplicated per request.

### SECP-R006 — existing controller remains authoritative for base behavior

The governor MUST NOT replace or bypass controller warmup, memory/KV safety, fallback, probe accounting, request-local state, feedback lineage, or replacement-policy transitions.

### SECP-R007 — downward cost caps are immediate

A stale higher worker width MUST NOT survive a newly lower valid cap.

### SECP-R008 — upward behavior remains replacement-policy-owned

The cost governor never raises above `K_base_i`.

---

# PART I — BLOCKER A: SURVIVAL / HISTORY

## 8. Naive variable-D pooling is rejected

### Counterexample A-CEX-01

```text
low state:  D=2, A=0
high state: D=4, A=4
equal frequency
```

Naive:

```text
S1=.5
S2=.5
S3=1
S4=1
Y4=4
```

Counterfactual K4 population useful yield:

```text
(1 + 5) / 2 = 3
```

The estimator remains biased with infinite sample size.

### SECP-R009 — naive variable-D pooling prohibited

Normal online survival MUST NOT pool every event satisfying only `D >= j`.

---

## 9. Same-base fully-exposed survival cohort

For request incarnation `i` and source base width `b`:

```text
H_i,b =
  finalized verifier events e such that:
    e.request_incarnation == i
    e.source_decision_mode == normal
    e.source_base_k == b
    e.actual_D >= b
    e.lineage_status == attributed
    e.finalization_status == eligible_nonterminal
    e.was_probe == false
    e.was_stale == false
    e.was_preempted == false
    e.was_aborted == false
```

### SECP-R010 — common denominator within base-width stratum

For `j <= b`:

```text
S_i,b,j =
sum_{e in H_i,b} 1[A_e >= j] / |H_i,b|
```

### SECP-R011 — no extrapolation beyond base width

Normal scoring MUST NOT estimate widths greater than current `K_base`.

### SECP-R012 — request-incarnation-local history

History MUST reset across request incarnation/epoch changes.

### SECP-R013 — base-width stratification

Different source base-width strata MUST NOT be silently pooled.

### SECP-R014 — clipped D<base excluded

If source base width is `b` but actual `D<b`, the event does not enter `H_i,b`.

### SECP-R015 — A0,Dpositive is valid

An otherwise eligible `A=0,D>=b>0` event enters the cohort and contributes zero survival at every depth.

### SECP-R016 — D0 is not positive evidence

`D=0` enters no positive-width survival stratum.

### SECP-R017 — probe isolation

Probe observations MUST NOT enter normal survival strata.

---

## 10. Survival window and readiness

Candidate default:

```text
survival_window_size = existing controller window_size
```

Audited default: `10`.

### SECP-R018 — fixed rolling window

Keep at most the latest `survival_window_size` eligible events per request-incarnation/base-width stratum.

### SECP-R019 — full-window activation

Normal cost optimization for `b>0` activates only when the required stratum has a full window.

Otherwise:

```text
decision_mode=cost_warmup
cost_policy_applied=false
```

### SECP-R020 — readiness is not significance

The full-window rule is operational readiness, not a statistical confidence claim.

---

## 11. Predicted useful yield

```text
Y_i(k) = 1 + sum_{j=1..k} S_i,b,j
Y_i(0) = 1
K_i(q) = min(K_base_i,q)
Y_batch(q)=sum_i Y_i(K_i(q))
```

### SECP-R021 — unit

`Y_batch(q)` is predicted useful tokens per retired batch.

### SECP-R022 — actual emitted tokens remain validation ground truth

Online survival does not replace post-stop emitted-token measurement.

---

# PART II — BLOCKER E: FEEDBACK FINALIZATION

## 12. Two-phase verifier feedback

Pre-stop `(A,D)` MUST first become:

```text
PendingVerifierEvent {
  request_id
  request_epoch_or_incarnation
  source_plan_sequence
  source_schedule_sequence
  verifier_or_consumed_origin_id_if_available
  source_base_k
  source_applied_k
  source_shared_k
  A
  D
  source_decision_mode
}
```

### SECP-R023 — pre-stop feedback is pending

Pre-stop `(A,D)` MUST NOT directly update normal survival.

### SECP-R024 — finalization states

Every pending event transitions to exactly one:

```text
ELIGIBLE_NONTERMINAL
TERMINAL_CENSORED
PREEMPTED_INVALID
STALE_INVALID
ABORTED_INVALID
AMBIGUOUS_INVALID
```

### SECP-R025 — only eligible nonterminal updates survival

Only finalized nonterminal normal events satisfying Part I enter `H_i,b`.

### SECP-R026 — terminal events censored

EOS/length/stop-terminal events MUST NOT be normalized to `A+1` for ordinary online survival.

### SECP-R027 — later schedule is not proof

A future schedule alone MUST NOT prove an earlier verifier output nonterminal.

### SECP-R028 — pending key must include incarnation and origin

`request_id` alone is insufficient.

### SECP-R029 — ambiguous association fails closed

Ambiguous event association becomes `AMBIGUOUS_INVALID`.

### SECP-R030 — no cross-incarnation consumption

New request incarnations MUST NOT consume old pending state.

### SECP-R031 — online vs validation evidence

Exact post-stop emitted count is optional for online survival but required for predicted-vs-actual validation claims.

### SECP-R032 — project-local feasibility gate

CPU/source tests MUST prove existing project hooks can uniquely finalize pending events under the supported async domain. Otherwise: `TEAM TECHNICAL ALIGNMENT REQUIRED`.

---

# PART III — BLOCKER B: UNCERTAINTY

## 13. Separate uncertainty layers

1. online survival: point estimate + readiness rule;
2. consumer cost: publication interval;
3. performance: separate benchmark uncertainty.

### SECP-R033 — no fake IID interval for serial survival

Do not assign IID binomial/Wilson semantics to serial request-local survival without a separate sampling proof.

---

## 14. Consumer-cost publication uncertainty

Replicate unit:

```text
independent cohort
```

Inner retirements are correlated inner observations.

### SECP-R034 — cluster-level resampling

Bootstrap independent cohort IDs and keep each cohort's inner retirements together.

Candidate defaults:

```text
measured cohorts/cell = 50
bootstrap resamples    = 5000
confidence level       = 95%
method                 = percentile bootstrap over cohort IDs
RNG seed               = fixed and recorded
```

### SECP-R035 — fixed sampling plan

Sample count is declared before collection. No collect-until-winner loop.

### SECP-R036 — failures preserved

OOM/timeout/invalid-shape/observer-failure cohorts remain raw evidence.

---

## 15. Published cost interval and conservative score

```text
C_point(s,q)
C_low(s,q)
C_high(s,q)
```

Unit: ms/retired batch.

### SECP-R037 — conservative candidate support

```text
R_lower(q)=Y_batch(q)/C_high(s,q)
R_upper(0)=Y_batch(0)/C_low(s,0)
```

Positive q is supported only if:

```text
R_lower(q) > R_upper(0)
```

### SECP-R038 — no performance overclaim

The interval does not prove TPOT/throughput/speedup.

### SECP-R039 — inconclusive cost

Overlap yields `INCONCLUSIVE_COST`.

### SECP-R040 — candidate selection

Among supported nonzero candidates choose maximum `R_lower(q)`. If none, strict normal mode selects 0.

### SECP-R041 — deterministic tie

Use:

```text
math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12)
```

Tie -> smaller cap.

---

# PART IV — BLOCKER C: DECISION-OBSERVABLE PROFILE STATE

## 16. Decision-state rather than exact CUDA-shape prediction

### SECP-R042 — pre-action fields only

A profile lookup key MUST use only information deterministically available before the governed action is committed.

Worker-only fields MUST NOT be backfilled as if known at decision time.

---

## 17. DecisionStateKeyV1

```text
DecisionStateKeyV1 {
  runtime_identity_key
  pure_decode_state
  eligible_decode_rows
  old_verification_width_multiset
  candidate_future_width_multiset
  candidate_shared_worker_k
  context_bucket_multiset
  queue_runtime_mode
}
```

### SECP-R043 — pure decode only

Mixed prefill/decode is unsupported in v1.1 unless later explicitly profiled.

### SECP-R044 — eligible rows are not client concurrency

External concurrency MUST NOT substitute for decision-time eligible decode rows.

### SECP-R045 — old verification and future proposal widths distinct

Async old-draft verification state and candidate future widths are separate key components.

### SECP-R046 — candidate width vector retained

`[4,4]`, `[4,0]`, `[4,2]`, `[2,2]` MUST NOT collapse only because B/maxK match.

### SECP-R047 — context is explicit

Use a reviewed deterministic context bucket from scheduler-visible state. Candidate diagnostic labels:

```text
[0,256)
[256,1024)
[1024,2048)
[2048,4096)
```

If context is unavailable at decision time, the state is unsupported until reviewed transport exists.

### SECP-R048 — physical shape recorded later

Actual worker rows/query/positions/padding/physical width are validation metadata and within-cell variability, not fabricated decision-time facts.

### SECP-R049 — no nearest-state fallback

No nearest B/context/vector lookup in v1.1.

### SECP-R050 — DecisionStateKey feasibility gate

If required key fields cannot be produced at the decision point without changing a shared interface: `TEAM TECHNICAL ALIGNMENT REQUIRED`.

---

# PART V — BLOCKER D: MODE / WARMUP / PROBE

## 18. Dedicated mode

```text
ADAPTIVE_COST_POLICY_MODE = off | strict | compat
```

### SECP-R051 — do not overload ADAPTIVE_STRICT

Existing scheduler/plumbing strictness is not SECP mode.

### SECP-R052 — default off

Backward-compatible code default is `off`. R6 validation explicitly enables `strict`.

---

## 19. Decision modes and precedence

Every decision has one primary mode:

```text
off
memory_override
cost_warmup
probe
normal
compatibility
```

Precedence:

```text
1. hard memory/KV safety
2. policy off?
3. profile/state admission
4. survival readiness
5. base source mode
6. normal cost evaluation
7. replacement/recovery
8. worker aggregation
```

### SECP-R053 — memory override

Memory/KV safety -> K0 in all modes.

### SECP-R054 — off

Existing behavior unchanged; `cost_policy_applied=false`.

### SECP-R055 — cost warmup

When profile/state valid, safety clear, source mode normal, but survival underfilled:

```text
decision_mode=cost_warmup
cost_policy_applied=false
cost_optimal=false
```

Run existing base/replacement path to avoid strict-mode self-deadlock.

### SECP-R056 — probe is exploration

Existing legacy probe remains explicit:

```text
decision_mode=probe
cost_policy_applied=false
cost_optimal=false
```

Probe events do not enter normal survival.

### SECP-R057 — strict degraded table

| Condition | Strict result |
|---|---|
| memory/KV pressure | K0 |
| missing consumer profile | K0 |
| identity mismatch | K0 |
| unsupported DecisionStateKey | K0 |
| missing K0 reference | K0 |
| invalid/incomplete cost cell | K0 |
| survival underfilled | cost_warmup/base |
| probe due | explicit probe |
| INCONCLUSIVE_COST | K0 |
| no supported positive | K0 |
| normal supported | governed cap |

### SECP-R058 — compatibility degraded table

When cost policy valid, use it. Otherwise legacy base behavior may continue, but:

```text
cost_policy_applied=false
hardware_cost_claim_allowed=false
```

Safety still forces K0.

### SECP-R059 — cost cap beats performance floor

In strict normal mode a lower valid cost cap MUST NOT be raised by a performance-oriented floor.

### SECP-R060 — upward-only recovery hold

Recovery may delay upward movement within bounds; it MUST NOT preserve stale K above a newly lower cap.

---

# PART VI — PROFILE IDENTITY / COST

## 20. Required identity

Consumer profile identity includes:

- observed GPU name, SM, VRAM class, alias-rule version;
- target model + immutable revision;
- EAGLE model + immutable revision;
- draft method;
- vLLM, PyTorch, CUDA, driver policy;
- requested/effective dtype and quantization;
- scheduler/executor/TP;
- async/eager/compile/graph mode;
- max model length/max sequences;
- cache/memory-utilization configuration;
- profile protocol version.

### SECP-R061 — requested cloud label insufficient

Requested `A10G` is not observed hardware identity.

### SECP-R062 — L4 mismatch

L4 evidence does not match A10-family runtime.

### SECP-R063 — unknown stays unknown

Missing compatibility fields MUST NOT be filled from current launch config and relabelled measured.

### SECP-R064 — component profile diagnostic-only

Existing synchronized target-forward + drafter-propose matrix is not canonical EAGLE-RW consumer cost.

---

## 21. Retirement-window cost

Primary cost:

```text
steady-state service wall time / retired valid decode batches
```

### SECP-R065 — same-process monotonic clock

### SECP-R066 — no blind component sum

### SECP-R067 — no added canonical CUDA synchronization

### SECP-R068 — matching K0 reference required

Every positive candidate state requires a matching K0 cell.

---

# PART VII — CANDIDATE SELECTION

## 22. Candidate set and scoring

```text
Q = sorted({0} ∪ {k in grid | exists i: k <= K_base_i})
K_i(q)=min(K_base_i,q)
Y_batch(q)=sum_i Y_i(K_i(q))
R_lower(q)=Y_batch(q)/C_high(s,q)
```

### SECP-R069 — positive eligibility

Positive q requires safety, strict/normal eligibility, identity match, exact DecisionStateKey, matching K0, ready survival, valid cost interval.

### SECP-R070 — support criterion

```text
R_lower(q) > R_upper(0)
```

### SECP-R071 — final normal cap

Choose supported q with maximum `R_lower`; ties smaller q; none -> q0 in strict normal mode.

### SECP-R072 — causal consumption

Changing only valid consumer-cost values MUST be capable of changing final planner-visible governed cap.

---

# PART VIII — TELEMETRY

## 23. Required decision telemetry

At minimum:

```text
policy_version
consumer_profile_protocol
profile_id/hash
profile_identity_status
decision_state_key/status

request_id/incarnation
source/base decision mode
K_base_i
candidate caps/vectors

survival stratum ID
window size/count
S values
candidate Y values

C_point/C_low/C_high
R_point/R_lower/K0 R_upper

selected cost cap
cost_policy_applied
cost_optimal
decision_mode/reason

K_applied_i
planned_worker_k
physical_draft_width_when_observed

memory/KV
replacement/recovery

producer/verifier lineage
pending/finalized feedback status
```

### SECP-R073 — no fake live latency

Profile cost cannot populate actual/live step-latency fields.

### SECP-R074 — decision vs measured physical state distinct

### SECP-R075 — compatibility is not hardware-cost evidence

### SECP-R076 — probe is not cost-optimal evidence

---

# PART IX — INVARIANTS

## 24. Required preserved invariants

- attributed positive-width feedback only;
- D0 does not create positive-width evidence;
- A0,Dpositive is valid;
- memory/KV safety dominates;
- applied <= allowed/base;
- worker=max applied;
- all-zero immediate;
- downward immediate;
- upward hold only;
- request-incarnation isolation;
- empty-cohort isolation;
- consumed-origin feedback;
- planned worker K != observed physical width;
- historical evidence immutable.

### SECP-R077 — pending state follows incarnation ownership

### SECP-R078 — one verifier origin finalizes at most once

---

# PART X — CLAIM CEILING / GATES

## 25. Compliance levels

```text
L0 artifact exists
L1 matching consumer profile valid
L2 R6-SOURCE candidate
L3 R6-CPU candidate
L4 R6-GPU
L5 R6-PERFORMANCE
```

Each level requires its own evidence. Predicted score never implies L5.

---

## 26. Human G0 freeze

Approve/reject:

1. same-base fully-exposed survival;
2. incarnation/base strata;
3. `survival_window_size=controller window_size`;
4. probe isolation;
5. pending-feedback finalization;
6. cluster bootstrap;
7. 50 cohorts / 5000 resamples / 95%;
8. conservative cost-bound scoring;
9. DecisionStateKeyV1;
10. no nearest-state;
11. off/strict/compat;
12. cost_warmup;
13. cap/floor/recovery precedence;
14. C/E source-feasibility gates;
15. K0 semantics.

Current status:

```text
SPEC FROZEN = NO
PRODUCTION IMPLEMENTATION AUTHORIZED = NO
GPU AUTHORIZED = NO
BENCHMARK AUTHORIZED = NO
```

---

## 27. G1 source feasibility

Before implementation:

- prove all DecisionStateKey fields available at decision time;
- prove pending verifier can be uniquely finalized post-stop.

Failure requiring shared contract:

```text
TEAM TECHNICAL ALIGNMENT REQUIRED
```

---

## 28. G2 causal CPU

Canonical planner A/B cost-surface flip with all non-cost inputs fixed.

---

## 29. G3 relevant CPU regression

New tests plus existing controller/planner/replacement/lineage/pinned gates, with no unexplained regression.

---

## 30. G4 collector fidelity

Only after G0-G3.

---

# PART XI — CANONICAL FIXTURES

## 31. F-A01 adaptive-D bias

```text
D2,A0
D4,A4
```

Naive tail estimator rejected.

## 32. F-A02 same-base cohort

```text
base4,D4,A0 normal finalized
base4,D4,A4 normal finalized
base2,D2,A0
base4,D2,A2 clipped
base2,D2,A2 probe
```

K4 stratum includes first two only.

## 33. F-E01 terminal censoring

Same A2,D4; nonterminal commits, terminal censors.

## 34. F-E02 multiple pending

Two origin keys same request/epoch finalize independently.

## 35. F-C01 heterogeneous state

`[4,4]` != `[4,0]` despite same B/maxK.

## 36. F-B01 uncertainty pass/fail

Use conservative cost-bound examples from the companion test plan.

## 37. F-D01 cost warmup

7/10 K4 events -> `cost_warmup`, not normal cost policy.

## 38. F-D02 immediate cap

previous K4, new cap2 -> worker <=2 immediately.

---

# PART XII — REFERENCES

1. Leviathan et al., ICML 2023  
   https://proceedings.mlr.press/v202/leviathan23a.html

2. EAGLE-2, EMNLP 2024  
   https://aclanthology.org/2024.emnlp-main.422/

3. vLLM Dynamic Speculative Decoding — design precedent only  
   https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/

4. vLLM Adaptive Verification — design precedent only  
   https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/

5. Cluster bootstrap reference  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/

---

## 39. Exactly one next action

**Human technical review of this candidate and the companion test plan; if accepted, freeze both before any production implementation.**


# Applied SECP-PULSE-2R-v1 normative amendment

## P2R-01 — Objective and claim ceiling

For this opt-in protocol, choose one two-retirement macro-action to maximize predicted useful emitted tokens per expected wall-clock millisecond over that same finite horizon. Let G be a predeclared preparation, queue-driving, sampling, safety/configuration and continuation domain. Let s be the unchanged exact pre-action DecisionState at the pulse decision. Define:

    C_cycle(s,q;G) = E[t_end(sigma2) - t_precommit(sigma1) | S=s, do(q then0), G]
    Y_cycle(s,q;G) = E[U_poststop(sigma1) + U_poststop(sigma2) | S=s, do(q then0), G]
    J_2R(s,q;G)    = Y_cycle(s,q;G) / C_cycle(s,q;G).

Use the ratio of expected reward to expected elapsed duration, not the mean of per-cohort reward/time ratios. This is a finite macro-action completion-efficiency objective. It is not a theorem or claim about stationary throughput, TPOT, global optimality, or long-run reward rate.

## P2R-02 — Scope and conditioning

V1 supports only pure-decode B1, original controller base K3, old verification width0 and initial context bucket[0,256), with exact observed product/runtime/config identity, immutable target/draft revisions and qualified request-local history. Future actions are exactly q in{0,2,3}, as derived from the existing candidate rule for base3. No K4 cell is required for this domain. Other states are unsupported until separately specified/profiled; no nearest-state, diagonal, architecture-family or product-name substitution is permitted.

G MUST bind exact workload/tokenization/preparation distribution, warm runtime/cache procedure, sampling domain, resolved native queue driver/capacity and pre-action pending-work contract, controller/recovery settings, common neutral continuation, source/observer versions, cohort/failure plan and fixed seeds. G is fixed before data collection and common across candidate arms. Each actual plan MUST serialize and hash its resolved G; the objective definition is parameterized by that frozen domain, not permission to choose a different G per q. Unknown or mismatching G fails closed before a new cost decision. Queue configuration and pending-work guards MUST NOT be inferred from requested hardware labels or static runtime_mode_id alone.

## P2R-03 — Action definition

The online/collection action is the ordered macro-action(q,0), not an unconstrained sequence of independent per-retirement cost decisions. Freeze q before execution; apply q exactly once to sigma1. The next schedule for the same request incarnation is sigma2 and MUST apply same-engine K0. Subsequent scheduling before sigma2 retires MUST use the same predeclared neutral K0 continuation/native queue-driving rule for every arm. No new cost election may override that lease. No unrelated client workload may be introduced silently.

Producer/consumer order is logical execution order, not an added host-side wait: native async queue filling MAY schedule/submit sigma2 before sigma1's host retirement. Sigma2 must still consume sigma1's exact producer, and both host retirements must finalize in their source-bound order. Artificially serializing the host loop would define a different G and require its own matching profile/domain; this amendment does not silently introduce that intervention.

## P2R-04 — K0 matching

The baseline is the identical two-retirement macro-action(0,0), with the same G, initial state, clocks, receipt/observer implementation, source/configuration and strict nonterminal support. K0 retains the same EAGLE engine/drafter/cache-forward semantics. A one-retirement K0 cost or a target-only baseline cannot substitute. The neutral K0 continuation is a declared part of the objective; it can change ranking relative to other continuations without being estimator bias for this objective.

## P2R-05 — Timing start/end

At sigma1's actual decision hook, capture a detached immutable source-bound receipt after the exact pre-action state is available but before pulse K/placeholders are committed and before worker submission. Sample t0 using same-process perf_counter_ns at this boundary. Sample t1 only after natural materialization, native append/stop/length handling and feedback finalization of exact sigma2. C for one cohort is(t1-t0)/1e6. Do not sum overlapping response latencies or use an arbitrary preceding retirement as t0. No added CUDA synchronization, forced future materialization or tensor readback is allowed.

## P2R-06 — Exact two-target lineage

Bind sigma1 and sigma2 using exact request/incarnation, plan sequence, schedule sequence and source/runtime/plan hashes. In this single-request V1 domain sigma2 is the next scheduled output for that same incarnation. Its old verification width MUST equal the actual pulse width q. For q>0, its consumed verifier origin MUST be sigma1's producer and actual verified/drafted width MUST be q, with0<=A<=D=q. For q0, retain explicit zero-width same-engine proposal witnesses; do not fabricate a positive origin or verifier observation. A predecessor retirement, later output, duplicate, stale incarnation or substituted origin cannot occupy either target slot.

## P2R-07 — Reward accounting

Reward is the sum of actual useful token counts from the two native post-stop outputs, recorded separately before summing. Drafted, accepted, verified slots, useful emitted output, original base, governed/applied/worker/physical K and old verification width remain distinct. Pre-stop A+1 is not a data-source substitute. In the supported nonterminal domain, pinned semantics imply U1=1 and U2=1+A_q only after the exact post-stop outputs and full-width origin have been validated.

## P2R-08 — Units and normalization

Profile cost unit is `ms/completed_two_retirement_cycle`; reward unit is `useful_tokens/completed_two_retirement_cycle`. The ratio has units useful_tokens/ms for this finite horizon. These values MUST NOT be relabelled as existing ms/retired_batch measurements. Normalizing both reward and cost by2 would preserve the objective, but V1 publishes the unnormalized cycle quantities to avoid unit ambiguity. Never normalize only one side.

## P2R-09 — Finite horizon and residual value

The horizon is exactly the two target retirements and terminal continuation value is fixed to zero before collection. Sigma2 verifies every pulse draft; sigma2's proposal width is0, so no unconsumed positive pulse width remains to be credited in a later retirement. Nevertheless q changes useful progress, context/KV state and potentially all later rewards/costs. Those residual effects are intentionally truncated by this finite-horizon approximation; the endpoint is not claimed to equal the baseline's full future state. Native prefetch/queue interference within t0..t1 is included under G even when it involves later neutral work; later outputs receive no reward credit. No third retirement, longer horizon or tail-value estimate may be added after seeing rankings.

## P2R-10 — Minimal predicted numerator

For request i and q in{0,2,3}, define the qualified normal-cycle prefix survival estimate S_hat_(i,base3,G,j). The proposed numerator is:

    Y_cycle_hat(s,q;G) = 2 + sum_(j=1..q) S_hat_(i,3,G,j)
    Y_cycle_hat(s,0;G) = 2.

Under the prefix-survival model this is1+the existing B1 predicted_yield(q), not the unchanged Y_batch(q). More generally the existing immediate old0 reward contributes B, but B>1 is not authorized by this V1. A change in numerator is mandatory; using C_cycle with unchanged Y_batch can change rankings and has the wrong reward horizon.

## P2R-11 — Survival qualification and assumptions

Reuse the current estimator's request-incarnation ownership, full rolling-window readiness, same-base full-exposure tail-count arithmetic and finalized-normal-event exclusions. Do NOT retroactively treat its legacy base-only history as cycle-qualified. A separate objective/domain stratum MUST admit only trusted completed normal pulse3/K0 cycles whose initial state is old0/base3 in the same G, whose continuation consumes all3 pulse slots and whose post-stop finalization is eligible. Retain the initial pulse receipt/domain with the accepted-prefix observation. q2/D2 does not expose the full base3 tail; q0 does not supply positive evidence. Calibration/probe/stale/preempted/terminal/aborted/ambiguous events do not enter this normal stratum.

The arithmetic identity E[min(A3,q)]=sum_(j<=q)P(A3>=j) holds for the same full-exposure population. Transporting it to a newly chosen q assumes common-prefix behavior and local representativeness/stability within the qualified history/window/G. Greedy sequential EAGLE and matching conditioning support that model, but source/CPU tests do not prove GPU numerical prefix invariance, future stationarity, statistical IID samples or unbiased prediction for arbitrary prompts. These are explicit model assumptions; no online survival confidence interval is claimed. If history is missing/unqualified, perform safe existing-base warmup with the same two-phase continuation and no cost claim, or fail closed. Normal warmup provenance may qualify only after a genuine full, matched, nonterminal cycle; calibration samples may not seed it.

A fixed rolling window is a point-prediction/readiness rule, not an asymptotically consistent estimator as the total lifetime sample count grows. Under a genuinely representative growing full-exposure sample, the same tail-count arithmetic has its usual conditional-frequency interpretation. This decision does not add an unsupported statistical-consistency or accuracy guarantee to the existing finite-window estimator.

## P2R-12 — Nonterminal interior support

Before each pulse, verify effective EOS stopping is disabled, stop-token/string/grammar/repetition termination is absent, sampling is greedy, and both remaining output budget and model-context capacity have at least6 tokens of unreserved slack after accounting for the maximum pending pre-action output. Five is insufficient because the maximum two-retirement advance is1+(3+1)=5 and equality can trigger a length cap. These guards are common across all arms, independent of q. Unknown pending bounds or effective stop settings are unsupported. This V1 does not estimate near-EOS or near-length-limit utility; it makes no deployment-wide assertion about those requests.

## P2R-13 — Terminal/censoring rules

Unexpected terminal/length/stop/clipped output in either target invalidates the complete cycle as a profile sample. Preserve the real clipped outputs and partial duration/receipts as censored raw evidence. Do not create an artificial zero-cost absorbing second retirement, relabel a later output as sigma2, estimate clipped reward from A+1, silently condition on surviving fast cycles, or replace invalid slots until a desired result appears. Near-terminal/absorbing-state objectives require a separate tail-aware reward model and amendment.

## P2R-14 — Failures and macro-action safety

Memory/KV safety, applied<=base and current replacement/downward/recovery bounds remain authoritative. A changed/clipped/rejected pulse, missing full exposure, unknown origin, unsupported state, identity/plan mismatch, OOM, timeout, abort, observer error, stale/preempted/ambiguous finalization or lease mismatch leaves raw evidence and does not produce a valid cycle. Safety can interrupt a macro-action; the policy MUST NOT preserve speculation merely to complete it. Only a fully source-bound completed cycle can support a completed-cycle hardware-cost claim. A forced continuation K0 is not a new independently cost-selected optimum.

## P2R-15 — Cohort reconstruction

Assign cohort IDs, arm/order and fixed plan before preparation. Each measured cohort contains one matched two-retirement cycle and starts from a newly reconstructed old0 state/request incarnation under the same G. Request-owned survival/history cannot be carried across incarnations or relabelled into independent cohorts. Unmeasured preparation/warmup and post-t1 cleanup remain outside the cost span. Existing normal online history may be used for prediction only within its valid incarnation/objective/domain stratum; cost calibration requires no online survival history. Fresh operational cohorts do not by themselves prove statistical independence of hardware/environmental nuisance; the eventual live protocol must preserve and assess temporal/block effects.

## P2R-16 — Publication

Per product and initial state/action cell, predeclare50 independent cohort slots, one cycle per cohort,5000 percentile bootstrap resamples of complete cohort IDs,95% confidence and seed731. Compute C_point as the mean cycle duration, and C_low/C_high as2.5/97.5 percentiles of cohort-resampled means. Keep each pair of target outputs, timing and all raw attempts together. Do not bootstrap individual retirements. Retain per-cycle actual reward and full K/token/origin provenance as validation data; online reward prediction remains request-local. Any failed/missing/censored slot leaves the cell incomplete under this conservative fixed-slot rule; no implicit replacement cohort protocol is granted.

Reuse the existing cohort-bootstrap mathematics, not its retirement-denominator/schema labels. A future implementation needs explicit cycle raw/publication types; it MUST NOT feed two retirements into the old publisher and relabel its per-retirement mean as milliseconds per cycle.

## P2R-17 — Profile key and admission

Use a versioned typed envelope containing objective `SECP-PULSE-2R-v1`, cost/reward units, horizon2, terminal_value0, continuationK0, resolved G hash and the unchanged exact initial DecisionStateKeyV1 for(0,q). Preserve all of its fields; add guards rather than weakening them. The two raw constituent keys(0,q)and(q,0)MUST be retained, but(q,0)is a continuation witness, not a replacement for the initial(0,q)cell and not a separately required cost lookup. The three initial cost cells remain(0,0),(0,2),(0,3), separately for A10 and A10G. Profile provenance includes exact source/runtime/product/model revisions, effective dtype/quantizer, domain guards, cycle counts, raw hashes, bootstrap parameters and failures.

Legacy component, steady-state, first-response and cycle profiles MUST be distinct and fail closed across types/units. An old EAGLE-RW loader MUST NOT admit cycle costs by label substitution. Unknown or mismatching objective, G, queue/sampling/tail/history guards or exact keys rejects a cost claim. Existing requested/observed product, driver/VRAM matching and immutable source rules remain unchanged.

## P2R-18 — Online score and uncertainty

For each admissible q>0 define:

    R_point(q) = Y_cycle_hat(q) / C_point(q)
    R_lower(q) = Y_cycle_hat(q) / C_high(q)
    R_upper(0) = 2 / C_low(0).

Support q only if R_lower(q)>R_upper(0). Select the supported q with greatest R_lower. Ties use the existing math.isclose(rel_tol=1e-9,abs_tol=1e-12)rule and the smaller q. If none is supported, select macro-action(0,0); preserve an inconclusive-cost explanation when appropriate. Keep safety, exact admission and history readiness before scoring. This asymmetry is conservative with respect to published cost intervals only. It does not account for uncertainty/model error in survival predictions, establish a simultaneous95% ranking guarantee or prove performance speedup.

## P2R-19 — Online lease and feedback

After the pulse's actual StepPlan is confirmed to apply the selected q exactly, reserve the same incarnation's next scheduling slot for K0; retain the parent cycle ID and exact expected origins. Do not rescore that slot or rearm a cycle while prior target outputs remain unfinalized. Native queue filling may submit later neutral K0 work under G; do not force a GPU wait merely to simplify attribution. Complete the lease only after both exact post-stop events. Reject duplicates/wrong epochs and invalidate interrupted cycles. No later output may leak into the pair. Scheduler-owned tail/stop/queue metadata and objective/domain-qualified history receipts require project-local transport/guards; the pinned Request, StepPlan and existing native completion hooks expose the needed facts, so no new upstream callback is required by this review.

## P2R-20 — Migration and compatibility

This is a versioned opt-in scientific objective change, not a calibration-only relabelling. Existing ordinary steady-state and POLICY_VALIDATION behavior remains unchanged until explicitly migrated. Under the new objective, POLICY_VALIDATION still requires an admitted matching profile, real deterministic candidate scores, correct macro-action lease, all existing safety/identity/source gates, physical execution/verifier/finalization evidence and false claims in calibration/probe/warmup/degraded modes.

Preserve the old R6 source/CPU evidence as evidence for its original contract. Do not reinterpret its old B2/K2-versusK4 fixture numbers as pulse-cycle results. Add new fixed-noncost cycle causal fixtures, numerator/baseline/horizon/lease tests and explicit type/domain/history rejection controls. Production implementation requires source review, all regression gates and a new immutable executable checkpoint/bundle before any later live execution. This audit does not perform that migration, campaign, GPU validation or performance benchmark.
