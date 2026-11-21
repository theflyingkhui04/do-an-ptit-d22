# G0 Decision Review — Scalar EAGLE Hardware-Cost Policy

**Status:** `TECHNICAL REVIEW PACKAGE — AMENDMENT CANDIDATE READY`  
**Date:** 2026-11-21  
**Repository source reviewed:** `4bb093ad4272aa330679d3baa9f5b1797ea397da`  
**Pinned runtime:** `vLLM 0.30.0`  
**Production code changed:** NO  
**GPU/benchmark/calibration performed:** NO

---

## 1. Executive verdict

The five original G0 blocker areas no longer require a generic verdict of:

```text
HARDWARE COST POLICY UNDERSPECIFIED
```

A deterministic amendment candidate can now be reviewed.

Recommended status:

```text
A survival/history     RESOLVED CANDIDATE
B uncertainty          RESOLVED CANDIDATE
C decision state       RESOLVED CANDIDATE; SOURCE FEASIBILITY GATE REMAINS
D modes/probe          RESOLVED CANDIDATE
E feedback/terminal    RESOLVED CANDIDATE; SOURCE FEASIBILITY GATE REMAINS

G0 AMENDMENT CANDIDATE READY FOR HUMAN TECHNICAL REVIEW
SPEC FROZEN = NO
```

The two remaining risks are no longer mathematical-policy ambiguity:

1. can the current decision point construct every required `DecisionStateKeyV1` field?
2. can current project-local hooks uniquely finalize pre-stop verifier feedback after stop processing under async execution?

If either answer is no, the correct escalation is:

```text
TEAM TECHNICAL ALIGNMENT REQUIRED
```

for the smallest shared interface needed.

---

## 2. Baseline evidence

Evidence package reports:

```text
branch: audit/flexsds-v023-hardwareadaptive-20261120
HEAD: 4bb093ad4272aa330679d3baa9f5b1797ea397da

190 CPU tests passed
0 failed
0 skipped
0 errors

19/19 pinned vLLM 0.30.0 files matched
3,789 protected files unchanged
production diff NONE
```

Current known implementation facts:

- `(A,D)` enters project feedback before final stop/length clipping;
- current planner only partially forecasts future worker shape;
- shared K/padded EAGLE means a zero request allowance does not prove proposal compute disappears;
- K0 still performs EAGLE forward/cache work;
- no current active measured-cost lookup changes K.

---

# 3. Decision A — survival/history

## 3.1 Problem

The parent proposal used per-depth eligible observations:

```text
D >= j
```

This becomes biased when draft width `D` is chosen adaptively from state correlated with future acceptance.

EAGLE-2 independently supports the premise that acceptance is context-dependent, so selection by adaptive controller state cannot be assumed ignorable.

Source:

https://aclanthology.org/2024.emnlp-main.422/

## 3.2 Options considered

### A1 — naive variable-D pooling

Rejected.

Pros:

- easy;
- uses all samples.

Failure:

- selected-depth populations differ;
- sample size does not eliminate selection bias.

### A2 — inverse propensity weighting

Rejected for v1.

Reason:

- current controller is largely deterministic;
- action overlap can be zero;
- propensity model would introduce a second unvalidated model.

### A3 — randomized exploration

Scientifically attractive, but rejected for v1 production semantics.

Reason:

- modifies controller behavior;
- increases runtime experimentation surface;
- requires a new exploration contract.

Possible future v2.

### A4 — same-base, fully-exposed request-local cohort

Recommended.

For current base `b`:

- use only normal events sourced from base `b`;
- require actual `D >= b`;
- require finalized nonterminal eligibility;
- use one common event denominator for all depths `j<=b`;
- do not extrapolate above `b`;
- isolate probe samples;
- reset on request incarnation.

## 3.3 Counterexample

Equal population:

```text
low  -> D2,A0
high -> D4,A4
```

Naive estimator:

```text
S=[.5,.5,1,1]
Y4=4
```

True K4 population useful yield:

```text
(1+5)/2=3
```

Bias remains with infinite sample count.

## 3.4 Recommendation

Freeze:

```text
same-base fully-exposed survival
request-incarnation-local
base-width-stratified
fixed rolling window
probe excluded
terminal finalized before admission
```

Use existing controller `window_size` as candidate survival window to avoid inventing a second unrelated warmup timescale.

Current audited default: 10.

## 3.5 Minimal proof

Fixture:

```text
base4,D4,A0 normal finalized
base4,D4,A4 normal finalized
base2,D2,A0 normal finalized
base4,D2,A2 clipped
base2,D2,A2 probe
```

Only first two enter K4 normal survival.

Expected:

```text
S=[.5,.5,.5,.5]
Y4=3
```

---

# 4. Decision B — uncertainty

## 4.1 Problem

Existing component-profile repeats are not proven independent cohorts.

Serial verifier events are context/history-correlated.

A simple binomial confidence interval over all inner events would overstate what the data establish.

## 4.2 Options

### B1 — Wilson/binomial interval over verifier events

Rejected for v1 online survival.

Wilson is useful under the right proportion-sampling assumptions, but those assumptions are not established for serial context-dependent request events.

### B2 — IID bootstrap over retirements

Rejected.

It breaks within-cohort correlation.

### B3 — cluster bootstrap over independent cohorts

Recommended for consumer-cost profile publication.

Resample cohort IDs, keep their inner retirements together.

Reference:

https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/

### B4 — full online bootstrap over request history

Rejected for v1.

Small rolling windows and per-request serial correlation make it complex without strengthening current claims.

## 4.3 Recommendation

Separate:

```text
online survival -> point estimate + fixed readiness rule
consumer cost   -> cohort-level uncertainty interval
performance     -> separate benchmark statistics
```

Candidate publication protocol:

```text
50 measured independent cohorts/cell
5000 bootstrap resamples
95% percentile interval
fixed recorded RNG seed
no adaptive collect-until-win behavior
```

These are engineering defaults to review, not theoretical constants.

## 4.4 Conservative decision

```text
R_lower(q)=Y(q)/C_high(q)
R_upper(0)=Y(0)/C_low(0)
```

Positive q is supported only if:

```text
R_lower(q)>R_upper(0)
```

This is conservative about consumer-cost uncertainty only, not proof of TPOT speedup.

## 4.5 Minimal proof

```text
Y2=3
C2=[1.9,2.2]

Y0=1
C0=[0.9,1.0]
```

passes.

If `C2=[1.9,2.9]`, it fails the conservative boundary.

---

# 5. Decision C — DecisionStateKey

## 5.1 Problem

Current planner has only partial forecast.

It cannot honestly know every worker-only execution field before action.

Using `batch_size=len(running)` or `(B,maxK)` would collapse scientifically different states.

Current vLLM Dynamic SD is useful design precedent that load and K interact, but is not pinned-v0.30 API proof.

Reference:

https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/

## 5.2 Options

### C1 — client-concurrency key

Rejected.

Client load is not worker execution state.

### C2 — B + max(K)

Rejected.

`[4,4]`, `[4,0]`, `[4,2]` can share B/maxK but differ in work.

### C3 — predict exact future physical CUDA shape

Rejected for v1.

Several fields are worker-only.

### C4 — decision-observable empirical state

Recommended.

Profile costs condition on information genuinely available before action.

Later physical shape becomes validation metadata/within-cell variability.

## 5.3 Recommended DecisionStateKeyV1

At minimum:

```text
runtime identity
pure-decode status
eligible decode row count
old verification-width vector/multiset
candidate future width vector/multiset
candidate shared K
context bucket multiset
queue/runtime mode
```

No nearest-state interpolation.

## 5.4 Why this is cleaner

A decision can only condition on variables available when it acts.

Unknown later physical variation should appear as measured profile variability, not fabricated forecast precision.

If variation makes the profile interval too wide:

```text
INCONCLUSIVE_COST
```

is the correct state.

## 5.5 Minimal proof

```text
A: old=[4,4], future=[4,4]
B: old=[4,0], future=[4,4]
```

same B/maxK.

Required:

```text
key(A)!=key(B)
```

## 5.6 Residual gate

Source feasibility must establish the required context/state fields.

If unavailable without a shared scheduler/planner contract:

```text
TEAM TECHNICAL ALIGNMENT REQUIRED
```

---

# 6. Decision D — modes/probe

## 6.1 Problem

Existing project has safety, warmup, fallback/probe and replacement rules, but no SECP-specific off/strict/compat contract.

## 6.2 Recommendation

Dedicated mode:

```text
ADAPTIVE_COST_POLICY_MODE =
  off
  strict
  compat
```

Default `off`.

Do not reuse existing `ADAPTIVE_STRICT`.

Scientific R6 validation explicitly uses `strict`.

## 6.3 State precedence

```text
memory/KV safety
>
mode off
>
profile/state admission
>
survival readiness
>
base source mode
>
normal cost evaluation
>
replacement/recovery
>
worker aggregation
```

## 6.4 Strict

- safety -> K0
- missing profile -> K0
- identity mismatch -> K0
- unsupported state -> K0
- valid state but insufficient survival -> `cost_warmup`, run base path, no cost claim
- probe -> explicit probe
- inconclusive cost -> K0
- no positive supported -> K0
- normal supported -> governed cap

## 6.5 Compat

When valid, use cost policy.

Otherwise run legacy base path with:

```text
cost_policy_applied=false
hardware_cost_claim_allowed=false
```

Safety still forces K0.

## 6.6 Why strict warmup is positive

If strict mode used K0 whenever survival was missing, it could never collect positive-depth normal evidence.

Therefore bounded `cost_warmup` runs current base behavior but cannot be called cost-optimal.

## 6.7 Probe

Preserve existing legacy probe semantics in v1.

```text
cost_policy_applied=false
cost_optimal=false
```

Probe samples do not enter normal survival.

## 6.8 Cap vs floor

Cost cap must dominate performance floors in strict normal mode.

Otherwise measured cost could be immediately undone.

---

# 7. Decision E — feedback/terminal

## 7.1 Problem

Pinned/project evidence shows:

```text
A,D generated
before
final append/check_stop/trim
```

Thus `(A,D)` alone cannot establish nonterminal useful yield.

## 7.2 Options

### E1 — update survival immediately

Rejected.

### E2 — later schedule proves nonterminal

Rejected.

Async prequeue/preemption/resume can violate the inference.

### E3 — new shared upstream finalized-verifier callback

Correct but intrusive.

Keep as fallback if project-local solution fails.

### E4 — two-phase pending event

Recommended.

## 7.3 Pending flow

```text
pre-stop A,D
  -> PENDING

post-stop finalization
  -> ELIGIBLE_NONTERMINAL
  -> TERMINAL_CENSORED
  -> PREEMPTED_INVALID
  -> STALE_INVALID
  -> ABORTED_INVALID
  -> AMBIGUOUS_INVALID
```

Only eligible nonterminal normal events enter survival.

## 7.4 Pending key

Must include at least:

```text
request_id
request epoch/incarnation
source plan sequence
source schedule sequence
verifier/consumed-origin ID if available
```

Never request ID alone.

## 7.5 Minimal proof

Same A2,D4:

- nonterminal -> commit;
- EOS/length -> censor.

Two outstanding events same request finalize independently.

New incarnation cannot consume old pending state.

## 7.6 Residual gate

CPU/source tests must prove current project hooks can close association.

If not:

```text
TEAM TECHNICAL ALIGNMENT REQUIRED
```

---

# 8. Cross-blocker architecture

Recommended conceptual flow:

```text
current controller
   ↓
K_base + source mode
   ↓
hard safety
   ↓
DecisionStateKeyV1
   ↓
consumer-profile admission
   ↓
same-base fully-exposed survival
   ↓
conservative consumer-cost scoring
   ↓
q <= K_base
   ↓
replacement/recovery
   ↓
worker K
   ↓
verification A,D
   ↓
pending event
   ↓
post-stop finalization
   ↓
normal survival history
```

Supported claim:

```text
hardware-cost-aware constrained governor
```

Unsupported claim:

```text
globally hardware-optimal speculative decoder
```

---

# 9. Canonical review decisions

Human technical reviewer must explicitly approve/reject:

1. same-base fully-exposed survival;
2. request-local/base-stratified history;
3. `survival_window_size = controller window_size`;
4. probe isolation;
5. two-phase pending feedback;
6. cluster-bootstrap publication;
7. 50 cohorts / 5000 bootstrap / 95% candidate defaults;
8. conservative `R_lower(q)>R_upper(0)` rule;
9. DecisionStateKeyV1;
10. no nearest-state interpolation;
11. off/strict/compat modes;
12. cost_warmup;
13. cost-cap/floor/recovery precedence;
14. project-local feasibility gates for C/E;
15. K0 same-engine semantics.

---

# 10. Minimal proof set before freeze

Two independent reviewers should derive the same result for:

- adaptive-D selection bias;
- same-base fully-exposed cohort;
- terminal censoring;
- two outstanding pending events;
- heterogeneous state key;
- uncertainty-overlap decision;
- strict warmup;
- strict missing profile;
- compat missing profile;
- memory override;
- probe;
- immediate downward cap;
- causal A/B cost-surface flip.

If not:

```text
G0 NOT READY
```

---

# 11. Research sources

Leviathan et al., ICML 2023  
https://proceedings.mlr.press/v202/leviathan23a.html

EAGLE-2, EMNLP 2024  
https://aclanthology.org/2024.emnlp-main.422/

vLLM Dynamic Speculative Decoding — design precedent only  
https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/

vLLM Adaptive Verification — design precedent only  
https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/

Cluster bootstrap reference  
https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/

---

# 12. Final verdict

```text
G0 AMENDMENT CANDIDATE READY FOR TECHNICAL REVIEW

SPEC FROZEN = NO
TEST PLAN APPROVED = NO
PRODUCTION IMPLEMENTATION AUTHORIZED = NO
GPU AUTHORIZED = NO
BENCHMARK AUTHORIZED = NO
```

Exactly one next action:

**Review and approve/reject the amendment candidate; then run the C/E source-feasibility tests before production implementation.**
