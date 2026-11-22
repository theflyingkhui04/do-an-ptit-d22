# SECP-PULSE-2R-v1 Dual-Product Full Profile Campaign Specification

**Document type:** Implementation specification (candidate for Codex execution)<br>
**Version:** 1.0<br>
**Status:** READY FOR IMPLEMENTATION REVIEW — CPU/SYNTHETIC ONLY<br>
**Real GPU campaign authorization:** NOT GRANTED BY THIS DOCUMENT<br>
**Repository:** `D:/Document/Four-year/do_an/do-an-ptit-d22-flexsds-audit`<br>
**Branch:** `audit/flexsds-v023-hardwareadaptive-20261120`<br>
**Required starting executable SHA:** `ac395ed035374ea3937e79b4cea077feb59bfa1a`<br>
**Pinned vLLM:** `0.30.0`<br>
**Objective:** `SECP-PULSE-2R-v1`

---

## 1. Purpose

This specification defines the **full, resumable, dual-product EAGLE pulse-cost profile campaign** for:

- `NVIDIA A10`
- `NVIDIA A10G`

The campaign exists to produce **two separate canonical measured pulse-cost profiles** that may later be admitted by the already-migrated `SECP-PULSE-2R-v1` online policy.

This document deliberately removes implementation discretion from the campaign harness. The implementation SHALL follow the state machine, cell set, slot rules, publication rules, failure behavior, source-provenance gates, and CLI behavior specified below.

This task is **not** a performance benchmark and SHALL NOT make TPOT, throughput, latency-speedup, or performance claims.

---

## 2. Governing prior decisions

The following decisions are already frozen and MUST NOT be reopened by the campaign implementation task.

### 2.1 Objective

For a frozen pre-action state `s`, action `q`, and preparation/queue domain `G`:

\[
J_{2R}(s,q;G)=
\frac{\mathbb{E}[U_1+U_2\mid s,\operatorname{do}(q,0),G]}
{\mathbb{E}[t_1-t_0\mid s,\operatorname{do}(q,0),G]}
\]

Where:

- retirement 1 is causally attributable to the one-shot pulse action `q`,
- retirement 2 is causally attributable to the mandatory `K0` continuation,
- `t0` is the immutable pre-action commit boundary,
- `t1` is after native materialization, stop handling, and finalization of retirement 2,
- useful-token reward and time cost use the **same two-retirement horizon**.

### 2.2 Predicted reward used later by policy validation

\[
\widehat{Y}_{cycle}(q)=2+\sum_{j=1}^{q}\widehat{S}_j
\]

Thus for this campaign domain:

- `q=0`: reward = `2`
- `q=2`: reward = `2 + S1 + S2`
- `q=3`: reward = `2 + S1 + S2 + S3`

The campaign itself measures **cost**, not survival reward.

### 2.3 Online conservative scoring already implemented

For later policy validation:

\[
R_{lower}(q)=\frac{\widehat{Y}_{cycle}(q)}{C_{high}(q)}
\]

and K0 comparator:

\[
R_{upper}(0)=\frac{2}{C_{low}(0)}
\]

A positive action is supportable only when `R_lower(q) > R_upper(0)`.

This campaign MUST publish the exact `C_point`, `C_low`, and `C_high` values required by that implementation.

---

## 3. Exact campaign scope

### 3.1 Product set

The only supported observed products are:

```text
NVIDIA A10
NVIDIA A10G
```

They are **distinct products**. No A10/A10G alias or shared `A10-family` profile is permitted.

The cloud/resource request string is provenance only. It SHALL NOT be used as observed product identity.

### 3.2 Frozen runtime/model family

Each product campaign SHALL use the currently reviewed runtime family:

```text
target_model = Qwen/Qwen2.5-7B-Instruct-AWQ
target_revision = b25037543e9394b818fdfca67ab2a00ecc7dd641

draft_model = duckhoi/qwen2.5-7b-eagle3.1-speculator
draft_revision = d9016c3860f42cc3f33a87fa5d346fee768f6290
draft_method = eagle3

vLLM = 0.30.0
effective_dtype = torch.bfloat16
effective_quantizer = auto_awq
scheduler = AdaptiveAsyncScheduler
executor = LineageUniProcExecutor
tensor_parallel_size = 1
async_scheduling = true
enforce_eager = true
compilation_mode = 0
cudagraph_mode = NONE
max_model_len = 2048
max_num_seqs = 8
prefix_caching = true
chunked_prefill = true
cache_dtype = auto
cache_block_size = 16
gpu_memory_utilization = 0.85
speculative_ceiling = 4
max_num_batched_tokens = 2048
```

If any required runtime field changes, the affected slot SHALL fail identity/config admission and SHALL NOT be silently reused.

### 3.3 Exact DecisionState domain

The campaign is scoped to:

```text
eligible_decode_rows = 1
base_k = 3
old_verification_width = 0
context_bucket = [0,256)
pure_decode = true
```

All other `DecisionStateKey` fields SHALL use the exact typed semantics already implemented in the repository.

### 3.4 Exact required cell set

Each product has exactly three required cost cells:

| Cell ID | Old verification width | Future pulse action | Shared K | Continuation | Horizon |
|---|---:|---:|---:|---:|---:|
| `old0_q0` | 0 | 0 | 0 | 0 | 2 retirements |
| `old0_q2` | 0 | 2 | 2 | 0 | 2 retirements |
| `old0_q3` | 0 | 3 | 3 | 0 | 2 retirements |

No `q=4` cell is part of this campaign.

The following SHALL NOT satisfy any required cell:

- `(old=2,future=2)`
- `(old=3,future=3)`
- steady-state cost records
- first-response-only records
- component timing records
- nearest-state substitutions
- reconstructed/fabricated old width

---

## 4. External technical facts verified for implementation

### 4.1 EAGLE-3 K semantics

Pinned/current official vLLM documentation confirms that:

- EAGLE-3 uses `method="eagle3"`.
- `num_speculative_tokens` is the number of draft tokens K.
- target verification processes `(1 + K)` tokens in an EAGLE decode step.
- dynamic K affects verification compute and effective batch size.

Therefore q-specific pulse cells are not interchangeable.

### 4.2 Modal bounding controls

Modal SDK/CLI supports:

- GPU resource selectors such as `A10G`,
- `max_containers`,
- `retries`,
- function timeouts.

The live campaign SHALL configure:

```text
max_containers = 1
retries = 0
```

No automatic Modal retry loop is permitted.

### 4.3 Git source provenance

Git bundle supports a self-contained reachable-object bundle that can be verified and cloned into an empty repository. The campaign source handoff SHALL use a self-contained bundle for the exact executable SHA, and SHALL validate:

```text
git bundle verify
empty-repository clone/reconstruction
exact HEAD
git status --short == empty
git fsck --full == PASS
no alternates
runtime-file manifest hashes
```

### 4.4 Durable state writes

Campaign state SHALL be written via:

1. write complete new JSON to a temporary file in the same state directory;
2. flush and close;
3. replace the prior state using same-filesystem `os.replace()`/equivalent.

The harness SHALL NOT mutate the state file in-place.

Single-writer semantics are mandatory; concurrent campaign-driver processes against the same state directory are unsupported and SHALL be rejected by a lock/owner marker or equivalent mechanism.

---

## 5. Definitions

### 5.1 Allocation

One externally scheduled Modal worker/container lifetime admitted for campaign use.

### 5.2 Cohort slot

A predeclared independent replicate identifier for one product and one cell.

Each required cell SHALL contain exactly:

```text
50 predetermined slots
```

Numbered deterministically:

```text
000 .. 049
```

### 5.3 Attempt

The execution corresponding to a predetermined cohort slot.

There SHALL be at most one scientific attempt per slot.

Infrastructure failure before any scientific execution may be classified separately only if the existing bounded protocol explicitly defines it as a non-attempt. Otherwise it remains the slot's failed attempt.

### 5.4 Valid slot

A slot whose pulse cycle satisfies all identity, state, lineage, timing, token, and finalization predicates.

### 5.5 Failed slot

Any predetermined slot that was attempted but fails canonical pulse-cycle predicates.

A failed slot SHALL NOT be replaced by a new successful slot.

### 5.6 Complete cell

A cell is publishable only if the frozen publication validator accepts its exact 50-slot cohort set.

Under the current contract, a failed predetermined slot makes that cell **publication-blocked**, not “49/50 plus replacement”.

### 5.7 Complete product

All three required cells are published and the final typed `EaglePulseProfile` passes exact local admission validation for that product.

---

## 6. Campaign plan

A campaign plan SHALL be immutable once the first live allocation is attempted.

The canonical plan JSON SHALL include at least:

```json
{
  "schema_version": "secp-pulse-2r-campaign-plan-v1",
  "objective": "SECP-PULSE-2R-v1",
  "source_git_sha": "<exact reviewed SHA>",
  "products": ["NVIDIA A10", "NVIDIA A10G"],
  "cells": [
    {"id":"old0_q0","old_width":0,"future_k":0,"shared_k":0,"continuation_k":0},
    {"id":"old0_q2","old_width":0,"future_k":2,"shared_k":2,"continuation_k":0},
    {"id":"old0_q3","old_width":0,"future_k":3,"shared_k":3,"continuation_k":0}
  ],
  "cohort_slots_per_cell": 50,
  "bootstrap_resamples": 5000,
  "confidence_level": 0.95,
  "bootstrap_seed": 731,
  "max_allocations": "<positive integer supplied before live run>",
  "modal": {
    "requested_gpu_selector": "A10G",
    "max_containers": 1,
    "retries": 0,
    "timeout_seconds": "<reviewed value>",
    "startup_timeout_seconds": "<reviewed value or null>"
  },
  "runtime_contract_hash": "<hash>",
  "preparation_contract_hash": "<hash>",
  "queue_contract_hash": "<hash>",
  "continuation_contract_hash": "<hash>",
  "failure_policy": "fixed-slot-no-replacement-v1"
}
```

The plan SHALL be canonicalized and SHA-256 hashed. The hash SHALL be stored in every state/raw/publication artifact.

Changing any scientific field after the first attempted slot requires a new campaign ID and new state directory.

---

## 7. Campaign directory layout

The implementation SHALL create one campaign root, for example:

```text
audit/r6_secp_v11_live_gpu_20261121/
  pulse_profile_campaign_<UTC>/
    campaign-plan.json
    campaign-plan.sha256
    campaign-state.json
    campaign-state.lock
    source/
      reviewed-source.bundle
      bundle.sha256
      package-manifest.json
      source-verdict.json
    allocations/
      0000/
      0001/
      ...
    raw/
      A10/
        old0_q0/
          cohort-000/
          ...
        old0_q2/
        old0_q3/
      A10G/
        ...
    publication/
      A10/
      A10G/
    logs/
```

No live artifact SHALL overwrite a historical smoke/audit directory.

---

## 8. Campaign state schema

The durable state SHALL include at minimum:

```json
{
  "schema_version": "secp-pulse-2r-campaign-state-v1",
  "campaign_id": "<stable id>",
  "plan_sha256": "<hash>",
  "source_git_sha": "<sha>",
  "allocation_count": 0,
  "status": "READY|RUNNING|BLOCKED|COMPLETE|MAX_ALLOCATIONS_REACHED",
  "products": {
    "NVIDIA A10": {
      "status": "MISSING|IN_PROGRESS|PUBLICATION_BLOCKED|COMPLETE",
      "cells": {
        "old0_q0": {
          "status": "PENDING|IN_PROGRESS|PUBLICATION_BLOCKED|COMPLETE",
          "slots": {
            "000": {
              "status": "UNATTEMPTED|RUNNING|VALID|FAILED",
              "attempt_id": null,
              "raw_manifest": null,
              "raw_sha256": null,
              "failure_code": null
            }
          }
        }
      },
      "profile_path": null,
      "profile_sha256": null
    },
    "NVIDIA A10G": {}
  }
}
```

State SHALL NOT be considered trustworthy until all referenced raw hashes are verified.

---

## 9. Resume invariants

Before any live work on resume, the driver SHALL verify:

1. current branch is the expected branch;
2. current executable SHA equals `plan.source_git_sha`;
3. tracked executable files/index satisfy the reviewed cleanliness rule;
4. campaign plan hash matches state;
5. source bundle hash matches plan/evidence;
6. every `VALID` or `FAILED` slot's raw manifest exists;
7. every stored raw hash matches current bytes;
8. no slot marked `VALID` lacks a canonical cycle;
9. no slot marked `FAILED` has been overwritten;
10. no completed profile hash/path has changed;
11. product/cell definitions equal the plan exactly;
12. no second campaign writer holds the same campaign lock.

Any failure SHALL stop before Modal invocation.

---

## 10. Product routing

After a worker starts but before inference, the worker SHALL observe the exact hardware/runtime identity.

Routing rule:

```text
if observed_gpu_name == "NVIDIA A10":
    route only to A10
elif observed_gpu_name == "NVIDIA A10G":
    route only to A10G
else:
    record unsupported allocation
    execute zero scientific cohort slots
```

Additional required identity/config fields SHALL be checked against the reviewed product/runtime admission rules.

No profile data may cross products.

If the routed product is already `COMPLETE`, the allocation SHALL be recorded and no cohort slot SHALL execute.

---

## 11. Cell/slot scheduling policy

Within a routed incomplete product, the next work SHALL be deterministic:

1. cells in order: `old0_q0`, `old0_q2`, `old0_q3`;
2. within a cell, smallest `UNATTEMPTED` slot ID first;
3. do not revisit `VALID` or `FAILED` slots;
4. if a cell is `PUBLICATION_BLOCKED`, do not spend GPU on later slots for that cell unless the plan explicitly requires completing all raw slots despite publication impossibility;
5. if a product cannot possibly publish under the fixed-slot rule, product SHALL become `PUBLICATION_BLOCKED` and the campaign SHALL stop for technical review rather than waste allocations.

Default fail-fast rule for v1:

> A failed scientific cohort slot marks the cell `PUBLICATION_BLOCKED` immediately, and the driver SHALL stop the campaign after durably recording the failure.

This rule avoids spending the remaining campaign budget on a product that cannot publish under the frozen no-replacement contract.

---

## 12. Scientific cohort execution

For each selected slot:

### 12.1 Pre-action reconstruction

The worker SHALL create a fresh request incarnation and establish the exact campaign domain:

```text
B = 1
base_k = 3
old_width = 0
context_bucket = [0,256)
pure_decode = true
supported nonterminal domain = true
```

The worker SHALL capture:

- exact request/incarnation identity,
- immutable pre-action `DecisionStateKey`,
- runtime identity,
- runtime/config hash,
- preparation/queue/continuation contract hashes,
- target cell q,
- pre-action timestamp/receipt.

### 12.2 Pulse

Apply q exactly once.

The ordinary controller SHALL NOT replace the chosen calibration pulse.

### 12.3 Retirement 1

Retirement 1 must be associated with the exact pulse origin.

Record:

- applied K,
- worker K,
- physical draft width,
- drafted tokens,
- accepted tokens,
- verified tokens if available,
- actual useful post-stop emitted tokens,
- lineage origin,
- native completion/finalization status.

### 12.4 Mandatory continuation

The next scheduling action SHALL be K0.

This continuation is part of the objective, not a new adaptive decision.

### 12.5 Retirement 2

Retirement 2 must be associated with the exact continuation origin.

Record the same provenance fields as retirement 1.

### 12.6 Cycle finalization

The canonical cycle ends only after retirement 2 has naturally completed materialization, stop handling, and project finalization.

No retirement 3 may enter the cycle.

No predecessor retirement may enter the cycle.

### 12.7 Timing

Cycle cost SHALL use the reviewed monotonic host timing boundary already implemented for PULSE-2R.

No added `torch.cuda.synchronize()` or equivalent synchronization is allowed merely to obtain cleaner numbers.

### 12.8 Token reward evidence

The raw record SHALL preserve actual useful post-stop tokens for both retirements even though campaign publication is cost-only.

This is required for later lineage/audit validation.

---

## 13. Canonical slot predicates

A slot is `VALID` only if all required predicates pass, including:

- exact observed product,
- exact product/runtime admission,
- exact source SHA,
- exact plan hash,
- exact initial DecisionState key,
- exact intended q,
- q applied once,
- K0 continuation applied exactly once,
- no ordinary planner override,
- correct retirement 1 origin,
- correct retirement 2 origin,
- no predecessor leakage,
- no retirement 3 leakage,
- correct timing boundaries,
- complete finalization,
- no terminal/censoring condition forbidden by the pulse contract,
- no failure/OOM/timeout,
- nonnegative/finite timing,
- required K/token/proposal provenance present,
- raw manifest/hash successfully persisted.

Any failed predicate makes the slot `FAILED`.

---

## 14. Raw artifact contract

Every attempted slot SHALL have an immutable raw directory containing at least:

```text
slot-plan.json
runtime-identity.json
decision-state.json
pulse-receipt.json
retirement-1.json
continuation-receipt.json
retirement-2.json
cycle-finalization.json
raw-record.json
raw-manifest.json
worker-log.txt
failure.json (if failed)
```

`raw-manifest.json` SHALL contain SHA-256 for every immutable raw file.

The slot status SHALL be changed from `RUNNING` to `VALID`/`FAILED` only after raw bytes and manifest are durably written and hash-verified.

---

## 15. Publication statistics

For each product/cell, the publication input is the set of **50 predetermined cohort slots**.

Publication parameters:

```text
N = 50 cohort slots per cell
bootstrap_resamples = 5000
confidence_level = 0.95
seed = 731
bootstrap unit = cohort ID / whole cycle
```

Individual retirement records SHALL NOT be resampled independently.

For each bootstrap resample:

1. sample 50 cohort IDs with replacement from the 50 valid cohort IDs;
2. compute the resample cell cost statistic using the project-approved pulse cost aggregation;
3. collect 5000 resample statistics.

The profile SHALL publish:

```text
C_point
C_low
C_high
```

using the project-approved estimator and 95% interval already implemented by `publish_pulse_cost`/equivalent.

The campaign implementation SHALL reuse that reviewed function rather than create a second statistical definition.

---

## 16. Publication gating

A cell SHALL NOT publish unless:

- all 50 predetermined slots satisfy the publication validator,
- all raw hashes verify,
- all slots have the same exact product/runtime/config/domain/cell definition,
- the objective is `SECP-PULSE-2R-v1`,
- horizon is exactly 2,
- continuation K is exactly 0,
- terminal value is exactly 0,
- the cost unit matches the active typed profile contract,
- the fixed bootstrap seed/resample count/CI match the plan.

A product SHALL NOT publish unless all three cells publish.

---

## 17. Canonical profile output

Two independent profile artifacts are expected:

```text
publication/A10/eagle-pulse-profile.json
publication/A10G/eagle-pulse-profile.json
```

The exact repository naming may differ if existing profile utilities dictate a canonical filename, but products SHALL remain separate.

Each profile SHALL include at minimum:

- schema version,
- objective `SECP-PULSE-2R-v1`,
- product identity,
- exact runtime/config identity,
- source Git SHA,
- target/draft model revisions,
- DecisionState payload/key for each cell,
- q and K0 continuation semantics,
- horizon = 2,
- cost unit,
- `C_point`, `C_low`, `C_high`,
- cohort slot count,
- bootstrap resamples = 5000,
- confidence = 95%,
- seed = 731,
- raw manifests/hashes,
- failure accounting,
- publication timestamp,
- campaign-plan hash.

After publication the profile SHALL be reloaded by the production admission code and validated for its own product/runtime domain.

---

## 18. Allocation loop

The manual live driver SHALL implement a bounded loop:

```text
while campaign not COMPLETE:
    if allocation_count >= MAX_ALLOCATIONS:
        status = MAX_ALLOCATIONS_REACHED
        stop cleanly

    perform all local resume/source/state validations
    allocate exactly one worker
    allocation_count += 1

    observe identity
    route A10 / A10G / unsupported

    if unsupported:
        preserve allocation evidence
        stop worker
        continue

    if routed product COMPLETE:
        preserve allocation evidence
        stop worker
        continue

    select deterministic next missing slot
    execute one or more slots on this same admitted worker only if
    the campaign protocol permits fresh request-incarnation cohorts
    on a stable worker

    persist each slot before proceeding

    if cell becomes publishable:
        publish/validate cell

    if product becomes publishable:
        publish/validate product profile

    if any hard blocker/publication-blocking failed slot occurs:
        stop campaign

    stop worker

if A10 COMPLETE and A10G COMPLETE:
    status = COMPLETE
```

### 18.1 Multiple cohorts per allocation

A cohort is defined by a fresh request/incarnation and frozen preparation contract, not automatically by a fresh VM.

Therefore the harness MAY execute multiple predetermined cohort slots on one admitted worker if and only if:

- the applied design does not require environment reallocation per cohort,
- each slot reconstructs a fresh request/incarnation,
- required cache/preparation reset semantics are executed and evidenced,
- runtime identity remains stable,
- no state/history leakage occurs,
- each cohort receives a unique predetermined slot ID.

The harness SHALL NOT claim cross-VM independence from multiple request cohorts on one worker. It SHALL preserve the exact replicate definition used by the frozen protocol.

---

## 19. CLI contract

The implementation SHALL expose one local campaign driver with these capabilities.

Required conceptual CLI arguments (exact spellings may use repository conventions, but tests SHALL pin them):

```text
--campaign-dir PATH
--max-allocations INT
--resume
--dry-run
--fake-sequence <fixture>   # test-only / local synthetic mode
```

### 19.1 Dry-run

`--dry-run` SHALL:

- execute zero Modal/API/GPU/upload operations,
- validate plan/state/source locally,
- print source SHA,
- print campaign-plan hash,
- print A10 status,
- print A10G status,
- print each cell's slot counts,
- print next deterministic action,
- print remaining allocation budget,
- exit 0 if locally runnable,
- exit nonzero on invalid state.

### 19.2 Live command

The real command SHALL require an explicit live flag or subcommand so accidental execution is difficult.

Recommended shape:

```text
python -B <driver> run --campaign-dir ... --max-allocations N --resume
```

`run` must not be the default behavior when no subcommand is given.

---

## 20. Fake executor / synthetic mode

A local fake executor SHALL be implemented to test the entire state machine without network or GPU.

It SHALL support deterministic scripted allocations such as:

```text
[A10G, A10, A10G, unsupported, A10]
```

and deterministic slot outcomes:

```text
VALID
FAILED_TIMEOUT
FAILED_TERMINAL
FAILED_STATE_MISMATCH
```

The fake executor SHALL be able to generate valid synthetic cycle records sufficient for local publication tests while clearly marking them synthetic and preventing confusion with live profiles.

Synthetic profiles SHALL never be written to the live canonical profile output path.

---

## 21. Source handoff

Because the current reviewed source SHA is:

```text
ac395ed035374ea3937e79b4cea077feb59bfa1a
```

and the campaign implementation itself will likely change executable files, the final implementation task SHALL:

1. implement the harness;
2. run all CPU tests;
3. review diff;
4. create a new exact-path local commit;
5. record new SHA;
6. create a new self-contained Git bundle from an exact ref pointing to that SHA;
7. verify the bundle locally;
8. reconstruct into an empty repository;
9. verify exact HEAD, clean status, `git fsck --full`, no alternates;
10. record bundle SHA-256 and size;
11. DO NOT upload or invoke Modal in the implementation task.

---

## 22. Stop conditions

The campaign driver SHALL stop cleanly on any of the following:

- source SHA mismatch,
- source bundle verification failure,
- dirty/unfrozen executable source,
- plan hash mismatch,
- state schema/version mismatch,
- state/raw hash mismatch,
- concurrent writer detected,
- observed product/runtime identity mismatch,
- required DecisionState mismatch,
- pulse/continuation lineage mismatch,
- timing/finalization failure,
- predetermined cohort slot failure under no-replacement policy,
- publication validator failure,
- profile re-admission failure,
- `MAX_ALLOCATIONS` exhausted.

No stop condition may be bypassed to force completion.

### 22.1 Explicit user resolution: unsupported allocations (2026-11-22)

An observed product outside `{NVIDIA A10, NVIDIA A10G}` is an unsupported
allocation, not a scientific cohort attempt. Record the allocation and full
observed identity, increment allocation_count exactly once, execute zero
scientific slots, create no VALID/FAILED cohort, and release that worker.
Unsupported allocations cannot create or mutate either product's scientific
slot state. If no other hard invariant has failed, continue under §18/CAM-M04
while allocation_count < MAX_ALLOCATIONS; otherwise terminate with
MAX_ALLOCATIONS_REACHED when the bound is reached. An unsupported product by
itself is not a campaign-wide hard stop. Source, schema/objective, state/hash,
raw integrity, single-writer, scientific-slot and publication invariants remain
campaign-wide stop conditions. This explicit user resolution governs §22.

---

## 23. Forbidden behaviors

The implementation SHALL NOT:

- auto-alias A10 and A10G,
- use requested cloud selector as observed product,
- reuse legacy component profiles,
- use steady-state/diagonal cells for off-diagonal pulse cells,
- replace failed cohort slots with new successes,
- drop failed raw data,
- recompute/overwrite prior raw slot data,
- silently change campaign plan after first attempt,
- use nearest-state profile lookup,
- add K4,
- use more than one concurrent GPU container,
- configure Modal retries above zero,
- run unbounded allocation loops,
- add synchronization purely for timing cleanup,
- run POLICY_VALIDATION as part of the profile campaign,
- run TPOT/throughput/performance benchmarking,
- claim R6-GPU PASS from profile collection alone.

---

## 24. Implementation files — preferred decomposition

Codex SHOULD prefer a small separation of concerns:

```text
experiments/eagle_pulse_campaign.py
    campaign plan/state/domain
    deterministic scheduling
    resume validation
    publication orchestration

experiments/eagle_pulse_campaign_modal.py
    external allocation / worker entrypoint adapter
    source reconstruction
    identity observation

experiments/eagle_pulse_campaign_fake.py
    CPU-only fake executor

tests/test_eagle_pulse_campaign.py
    campaign state-machine tests

tests/test_eagle_pulse_campaign_publication.py
    publication/statistics tests
```

If repository conventions support a better minimal structure, Codex may adapt names, but SHALL preserve these responsibilities.

Production PULSE policy modules SHALL NOT be redesigned in this task.

---

## 25. Definition of Done

The implementation task is PASS only when all are true:

1. frozen objective unchanged;
2. exact cell set unchanged;
3. A10/A10G separate;
4. 50 fixed slots/cell implemented;
5. no-replacement failure semantics implemented;
6. 5000 cohort bootstrap / 95% / seed 731 implemented through reviewed publisher;
7. durable atomic campaign state implemented;
8. resume hash validation implemented;
9. deterministic next-work selection implemented;
10. bounded `MAX_ALLOCATIONS` loop implemented;
11. `max_containers=1`, `retries=0` pinned;
12. dry-run invokes zero external action;
13. fake executor covers complete dual-product lifecycle;
14. local synthetic publication validated;
15. all test-plan mandatory cases PASS;
16. relevant existing 746-test baseline remains green;
17. pinned vLLM 19/19 verification remains green;
18. `git diff --check` passes;
19. reviewed local source commit created;
20. self-contained Git bundle rebuilt and empty-repo verified;
21. Codex performs zero Modal/GPU/upload/live inference;
22. Codex outputs exact manual dry-run command;
23. Codex outputs exactly one primary manual live campaign command;
24. Codex outputs exact post-run validation command.

Required completion status:

```text
FULL DUAL-PRODUCT CAMPAIGN HARNESS READY — USER RUN REQUIRED
```

---

## 26. Required Codex final answer

Codex SHALL return:

```text
## STATUS

## BASELINE
branch
starting SHA
final SHA
source status
pinned vLLM verification

## CAMPAIGN PLAN
products
cells
slots/statistics
MAX_ALLOCATIONS semantics
failure rule

## IMPLEMENTATION
changed files + responsibility

## STATE / RESUME
state schema
atomic write rule
hash checks
next-work rule

## CPU TESTS
passed
failed
errors
skipped
deselected

## DRY-RUN
synthetic sequence/result
explicit confirmation: no Modal/GPU

## SOURCE FREEZE
final SHA
bundle path/hash/size
empty-repo validation

## MANUAL DRY-RUN COMMAND
<copy/paste ready>

## MANUAL FULL CAMPAIGN COMMAND
DO NOT RUN IN CODEX — USER MANUAL COMMAND
<exactly one primary live command>

## EXPECTED PROFILE PATHS
A10
A10G

## POST-RUN VALIDATION COMMAND
<copy/paste ready>

## R6 STATUS
R6-SOURCE PASS
R6-CPU PASS
R6-GPU UNVERIFIED
R6-PERFORMANCE UNVERIFIED

## NEXT ACTION
User manually runs the full campaign command and returns the resulting profile/evidence status for R6-GPU POLICY_VALIDATION.
```

---

## 27. External references checked

These are cross-checks only; repository pinned source and applied SECP design remain authoritative.

1. vLLM Ascend 0.30.0 Speculative Decoding Guide<br>
   https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/speculative_decoding.html

2. Git bundle documentation<br>
   https://git-scm.com/docs/git-bundle.html

3. Modal Python SDK — App/function options (`max_containers`, `retries`, `timeout`)<br>
   https://modal.com/docs/sdk/py/latest/App

4. Modal CLI shell GPU selector examples (`a10g`)<br>
   https://modal.com/docs/cli/latest/shell

5. Python `os.replace` documentation<br>
   https://docs.python.org/3/library/os.html#os.replace

---

## 28. Non-goals

This specification does not authorize or define:

- live campaign execution by Codex,
- live policy validation,
- performance benchmarking,
- generalization beyond B1/base3/old0/context `[0,256)`,
- A10/A10G profile equivalence,
- alternative objective horizons,
- changing the 50/5000/95%/seed-731 publication contract,
- replacing failed predetermined slots.

Any such change requires a separate technical review.
