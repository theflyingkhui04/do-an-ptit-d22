# FlexSDS Hardware-Adaptive Replacement Policy Specification

- **Document type:** Technical design contract
- **Purpose:** Define correctness invariants for a replacement of the divergent behaviors in `59f18a9` and `0d4a9fa`
- **Primary source baseline:** `8006205a6f19790a776d67155a377def053d2736`
- **Frozen evidence snapshot:** `1eb7e594f8f0afa2fc72df8c571ee13cacfa8e06`
- **Audit branch:** `audit/flexsds-v023-hardwareadaptive-20261120`
- **Review result that motivates this spec:** `INTEGRATE NEITHER AS WRITTEN`
- **Scope:** warmup-feedback contract, low-acceptance floor/probing semantics, batch-K transition semantics, lineage-safe telemetry
- **Out of scope:** measured hardware-cost policy (R6), performance optimization, tree-adaptive redesign, experiment protocol changes, Trello changes

---

## 1. Problem statement

Two historical commits were reviewed against baseline `8006205`:

```text
59f18a9  acceptance floor + warm-up guard + reason counts
0d4a9fa  batch-K hysteresis + per-request K counts
```

Neither commit is an ancestor of `8006205`. Both diverge from merge base:

```text
c8fda862c90ebc2156a7d058dd7437ea6fe52e70
```

The review concluded:

```text
INTEGRATE NEITHER AS WRITTEN
```

because:

1. `59f18a9` uses controller `update()` count as its warmup sample count, so direct `update(0, 0)` calls can advance the gate without a real positive-width verifier event.
2. `59f18a9` can use one alpha to choose K and expose another alpha in telemetry unless the decision contract is explicit.
3. `0d4a9fa` uses planner-wide held batch K. During a downward transition, held K may exceed fresh request selections and may increase physical draft work.
4. In uniform mode, a held high shared K can also increase verification width.
5. Hold state can survive request finish / cohort change because it is planner-wide.
6. Whole-file replacement would regress newer lineage, profile, history, and telemetry interfaces already present in `8006205`.

This specification defines the behavior a replacement patch must satisfy before production code is changed.

---

## 2. Design principles

### P1 — Request decisions remain request-local

Each request has an independently computed requested K:

```text
request_id
→ request-local controller state
→ selected_k
```

No batch-level stabilization mechanism may rewrite the semantic decision of another request.

### P2 — Safety decisions dominate policy stabilization

The following must always be able to force request K to zero:

```text
memory / KV safety fallback
explicit no-draft condition
finished / stale request
ineligible request
```

No `min_k`, warmup policy, batch floor, hysteresis, or hold may override a safety K=0.

### P3 — Warmup is based on real verification evidence

Warmup samples are not generic controller updates.

A warmup sample is counted only when all of the following are true:

```text
1. feedback is attributed to the correct live request incarnation
2. feedback corresponds to consumed drafts from a real producer step
3. drafted_tokens > 0
4. verification completed for that draft origin
5. the feedback event has not already been counted
```

A feedback event with:

```text
drafted_tokens > 0
accepted_tokens = 0
```

IS a valid warmup sample.

A direct call such as:

```text
update(0, 0)
```

MUST NOT advance warmup.

### P4 — One alpha must be identified as the rule alpha

For every decision, telemetry must distinguish:

```text
effective_alpha
rule_alpha
```

where:

```text
rule_alpha = the exact alpha value used to choose K
```

If warmup uses `initial_alpha`, then telemetry must show:

```text
effective_alpha = current estimator output
rule_alpha      = initial_alpha
warmup_active   = true
```

Do not overload a single `alpha` field with two meanings.

### P5 — Stabilization may suppress work, but must not create extra work

A batch-level stabilization mechanism may cap a request below its fresh requested K:

```text
applied_k_i <= selected_k_i
```

It MUST NOT increase that request above its fresh selection.

For every request row `i`:

```text
0 <= applied_k_i <= selected_k_i <= max_k
```

This invariant applies to:

```text
scheduler-visible K
physical proposal width attributable to the row
verification prefix width
telemetry-reported applied K
```

### P6 — Shared worker width is an execution detail, not a request decision

If the underlying worker requires a shared physical K for the batch, define:

```text
worker_k = max(applied_k_i for scheduled eligible rows)
```

The worker may allocate a padded/shared buffer at `worker_k`, but:

```text
request i must consume / verify no more than applied_k_i
```

If a worker physically computes padded tokens beyond `applied_k_i`, that overhead must be explicitly observable as:

```text
worker_k
physical_draft_width
row_applied_k
```

and the policy must not claim that those extra physical tokens were requested by the row.

Preferred target:

```text
worker_k = max(applied_k_i)
```

and never a stale held K above that maximum.

### P7 — K=0 is immediate and non-hysteretic

If the fresh aggregate result is zero because every eligible request is zero:

```text
worker_k = 0
```

immediately.

No positive held K may delay an all-zero transition.

This preserves:

```text
memory fallback
low-acceptance fallback
explicit disable
```

### P8 — Recovery hysteresis must be monotone-safe

If hysteresis is retained, it may delay an increase:

```text
0 → 2
2 → 3
3 → 4
```

but must not delay a safety decrease.

Allowed pattern:

```text
decrease: immediate
increase: optionally delayed
```

Forbidden pattern:

```text
fresh selected max = 2
held worker K      = 4
→ continue worker K=4
```

### P9 — Batch stabilization state must have a defined lifetime

Any batch-level stabilization state must reset when one of these occurs:

```text
no eligible decode rows remain
planner/session generation changes
explicit planner reset
runtime restart
```

It must not silently inherit policy state from an unrelated finished cohort.

If request finish alone does not empty the active cohort, state may remain, but the definition must be explicit and testable.

### P10 — Historical evidence remains historical

Any new replacement patch changes source identity.

Historical evidence collected under:

```text
SOURCE_COMMIT=8006205...
EVIDENCE_COMMIT=1eb7e59
```

must remain attached to that source.

Do not relabel historical runs as evidence for the replacement patch.

---

## 3. Warmup contract

### 3.1 State

Per request:

```text
verification_sample_count: int >= 0
min_warmup_samples: int >= 0
initial_alpha: float
effective_alpha: float
```

### 3.2 Gate

Define:

```text
warmup_active =
    verification_sample_count < min_warmup_samples
```

Then:

```text
rule_alpha =
    initial_alpha   if warmup_active
    effective_alpha otherwise
```

### 3.3 Counting event

Increment `verification_sample_count` exactly once for each unique attributed feedback event satisfying:

```text
drafted_tokens > 0
```

Do not increment for:

```text
decision-only calls
update(0, 0)
duplicate feedback
stale feedback
finished-request feedback
wrong incarnation
wrong draft origin
telemetry-only events
```

### 3.4 Acceptance estimator

The acceptance estimator may update from the same valid event.

For a valid event:

```text
acceptance_sample = accepted_tokens / drafted_tokens
```

with:

```text
0 <= accepted_tokens <= drafted_tokens
drafted_tokens > 0
```

### 3.5 Reset

Request session reset / finish + reuse must create:

```text
verification_sample_count = 0
EMA state = initial state
fallback/probe state = initial state
```

No warmup count may leak between request incarnations.

---

## 4. Low-acceptance floor and probing contract

A replacement may support a configurable acceptance floor, but the semantics must be constrained.

### 4.1 Priority order

Decision priority must be:

```text
1. invalid / finished / ineligible request
2. memory/KV safety rule
3. explicit cooldown safety state
4. warmup rule-alpha selection
5. acceptance policy
6. optional non-safety min_k floor
7. probe logic
8. candidate-set / max_k cap
```

The exact implementation may structure this differently, but observed behavior must match the priority.

### 4.2 `min_k`

If retained:

```text
min_k >= 0
```

and it only applies to acceptance-policy fallback.

It MUST NOT override:

```text
memory K=0
KV K=0
request-ineligible K=0
explicit no-draft K=0
```

### 4.3 Candidate compatibility

A positive `min_k` must be normalized to the allowed candidate set.

Example:

```text
candidates = [0, 2, 3, 4]
min_k = 1
```

must not create unsupported K=1.

The normalization rule must be documented and deterministic.

Preferred:

```text
floor_k = smallest candidate >= min_k
```

with failure if none exists.

### 4.4 Probing

Probe cadence must be defined using eligible low-acceptance decisions, not arbitrary controller calls.

Probe state must specify:

```text
fallback_decision_count
probe_interval
probe_k
```

Safety K=0 decisions caused by memory pressure must not accidentally count as acceptance fallback decisions unless explicitly designed and tested.

---

## 5. Batch-K transition contract

The historical planner-wide symmetric hold is rejected.

If a replacement keeps stabilization, use a safe cap model.

### 5.1 Fresh request decisions

For scheduled eligible rows:

```text
selected_k_i
```

comes from each request-local controller.

### 5.2 Optional recovery cap

Define one planner-level cap:

```text
recovery_cap_k
```

whose only allowed effect is:

```text
applied_k_i = min(selected_k_i, recovery_cap_k)
```

Therefore:

```text
applied_k_i <= selected_k_i
```

always.

### 5.3 Decreases

If the fresh selected maximum decreases:

```text
fresh_max_k = max(selected_k_i)
```

then:

```text
recovery_cap_k = min(recovery_cap_k, fresh_max_k)
```

immediately.

No downward hold.

### 5.4 Increases

An increase may require `N` consecutive eligible planner decisions.

Example:

```text
current recovery_cap_k = 2
fresh_max_k = 4
hold_steps = 3
```

Then:

```text
step 1 → cap 2
step 2 → cap 2
step 3 → cap 4
```

while each row remains:

```text
applied_k_i = min(selected_k_i, cap)
```

### 5.5 Zero aggregate

If:

```text
fresh_max_k = 0
```

then immediately:

```text
recovery_cap_k = 0
worker_k = 0
pending increase reset
```

### 5.6 Worker K

For the current scheduled eligible batch:

```text
worker_k = max(applied_k_i)
```

If there are no eligible rows:

```text
worker_k = 0
```

### 5.7 Uniform mode

Uniform mode MUST NOT assign a request a K above its own controller selection.

If uniform mode cannot preserve:

```text
applied_k_i <= selected_k_i
```

then it is incompatible with this replacement policy and must be disabled for the adaptive path.

Do not silently retain the historical behavior:

```text
row_i = shared held K
```

---

## 6. Cohort / lifecycle contract

Planner-level recovery state is execution state, not request identity state.

It must reset on:

```text
planner construction
explicit planner reset
transition to zero eligible decode requests
runtime restart
```

Request-level counters reset on:

```text
request finish + cleanup
request incarnation change
explicit controller reset
```

Preemption semantics must preserve lineage correctness:

```text
feedback from old producer origin
→ accepted only if origin/incarnation binding remains valid
```

A new request reusing the same request ID must not receive:

```text
old warmup count
old EMA
old reason counts
old feedback
```

---

## 7. Telemetry contract

### 7.1 Decision telemetry

Per request decision must expose at least:

```text
request_id
request_incarnation
plan_id
selected_k
applied_k
effective_alpha
rule_alpha
warmup_active
verification_sample_count
reason
memory_state
```

### 7.2 Batch telemetry

Per planner/scheduler step:

```text
plan_id
fresh_max_k
recovery_cap_k
worker_k
eligible_batch_size
```

If a pending increase exists:

```text
pending_target_k
pending_streak
required_hold_steps
```

### 7.3 Feedback telemetry

Each accepted feedback event:

```text
request_id
request_incarnation
producer_plan_id
producer_schedule_id
drafted_tokens
accepted_tokens
feedback_counted_for_warmup = true/false
```

### 7.4 Histograms

If `reason_counts` or `batch_k_counts` are retained, their semantics must be explicit.

Preferred:

```text
reason_counts:
  request-local count of controller decisions by reason

selected_k_counts:
  request-local count of controller-selected K

applied_k_counts:
  request-local count of scheduler/planner applied K

worker_k_counts:
  step-level count of shared worker K
```

Do not use one `batch_k_counts` field ambiguously for selected, scheduled, physical, and verified K.

---

## 8. Compatibility constraints with baseline `8006205`

A replacement patch must preserve:

```text
EmpiricalCostProfile plumbing
bounded request histories
topology telemetry
StepPlan / PlanSequence interfaces
decision snapshots
record_scheduled_budget
record_lineage_failure
record_verification(..., source=...)
request incarnation checks
stale / finished filtering
draft-origin binding
accepted <= drafted validation
```

Do not replace baseline files wholesale with historical versions.

The implementation must be an additive / targeted patch against current baseline interfaces.

---

## 9. Scientific interpretation

A replacement patch changes policy semantics if any of the following is enabled:

```text
min_warmup_samples > baseline-equivalent behavior
min_k > 0
recovery hold / cap > 0
new candidate normalization
new lifecycle reset semantics
```

Therefore:

```text
historical Adaptive ON evidence
!=
replacement-policy evidence
```

A new source SHA requires:

```text
focused CPU/unit/integration validation
then bounded GPU smoke for changed semantics
then separate lossless validation
```

Performance remains out of scope until correctness and comparability are explicit.

---

## 10. Non-goals

This replacement design does NOT solve:

```text
R6 measured hardware cost not driving K
ON/OFF policy-isolation
lossless root-cause localization
TreeAdaptive
performance benchmarking
```

Do not use this patch as an excuse to address those issues in the same change.

---

## 11. Acceptance invariants

A production patch is acceptable only if all invariants below hold:

```text
I1  warmup count == unique attributed positive-width verifier feedback count
I2  update(0,0) cannot advance warmup
I3  accepted=0, drafted>0 does advance warmup
I4  memory/KV safety can always force K=0
I5  min_k cannot override safety K=0
I6  rule_alpha is explicitly observable
I7  applied_k_i <= selected_k_i for every request
I8  worker_k == max(applied_k_i) for current eligible scheduled rows
I9  fresh all-zero aggregate forces worker_k=0 immediately
I10 downward K changes are never delayed by hysteresis
I11 upward K changes may be delayed only by an explicit recovery hold
I12 planner recovery state cannot leak across an empty active cohort
I13 request state cannot leak across request incarnation reuse
I14 feedback remains bound to consumed draft origin
I15 historical evidence remains mapped to its original SHA
I16 no baseline lineage/profile/telemetry interface is regressed
```

Any violation of I1–I16 is a **BLOCKER** for the replacement patch.

---

## 12. Decision expected after implementation

After tests pass, classify the replacement patch as exactly one:

```text
READY FOR BOUNDED GPU VALIDATION
NOT READY — CORRECTNESS BLOCKER
NEED TEAM TECHNICAL ALIGNMENT
```

`NEED TEAM TECHNICAL ALIGNMENT` is appropriate only for a shared runtime/interface policy decision, not for missing tests.
