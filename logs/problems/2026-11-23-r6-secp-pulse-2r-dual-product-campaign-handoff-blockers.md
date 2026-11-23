# R6 SECP-PULSE-2R Dual-Product Campaign — Protocol, Harness, Provenance và Handoff Blockers

- **Mã vấn đề:** `PROB-R6-SECP-PULSE-2R-DUAL-CAMPAIGN-HANDOFF`
- **Ngày tổng hợp:** 23/11/2026
- **Owner:** Bùi Thiên Đại / `daijestic`
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Branch:** `audit/flexsds-v023-hardwareadaptive-20261120`
- **Current local HEAD:** `218a9b1e2fd78118f1ca3c387c3b162bd9c2ff2b`
- **Current remote branch HEAD observed:** `4c76830706d6d60d715d962764e592b7b184fc42`
- **Branch relation at latest local handoff check:** `behind=0`, `ahead=10`
- **Current active policy family:** `SECP-PULSE-2R-v1`
- **Current scientific scope:** separate measured pulse-cycle consumer profiles for `NVIDIA A10` and `NVIDIA A10G`
- **Current R6 status:**
  - `R6-SOURCE PASS`
  - `R6-CPU PASS`
  - `R6-GPU UNVERIFIED`
  - `R6-PERFORMANCE UNVERIFIED`
- **Live full campaign:** **NOT RUN**
- **Live profile publication:** **NOT RUN**
- **Live POLICY_VALIDATION:** **NOT RUN**
- **Performance benchmark:** **NOT RUN**
- **Push status:** **NOT YET VERIFIED COMPLETE**
- **Current handoff state:** `CAMPAIGN HARNESS IMPLEMENTED/CPU-TESTED LOCALLY — FINAL HANDOFF/PUSH STILL OPEN`

> File này ghi lại chuỗi blocker, root cause, quyết định kỹ thuật, fix, test evidence, source provenance và handoff state của nhánh R6 từ giai đoạn “measured hardware cost chưa điều khiển K” đến khi campaign harness dual-product được xây dựng.
>
> File này **không biến smoke thành profile**, **không biến CPU proof thành GPU proof**, **không coi A10 và A10G là cùng hardware product**, và **không coi local commits là đã handoff xong cho tới khi remote branch được xác minh trùng local HEAD**.

---

# 1. Bối cảnh

Mục tiêu R6 ban đầu là chứng minh chuỗi causal:

```text
measured hardware cost
→ exact runtime/profile admission
→ cost-aware candidate scoring
→ selected K
→ applied K
→ worker behavior
→ finalized feedback
```

và sau đó mới tiến tới:

```text
canonical measured consumer profile
→ live policy consumption
→ R6-GPU evidence
→ performance experiment
```

Trong quá trình audit, vấn đề ban đầu tưởng là “thiếu hardware profile” nhưng thực tế tách thành nhiều lớp:

```text
DecisionState transport
+
runtime identity
+
feedback finalization
+
profile provenance
+
cost-policy causality
+
collector bootstrapping
+
measurement estimand
+
reward/cost horizon
+
campaign reproducibility
+
source packaging/provenance
```

Do đó mỗi GPU step chỉ được chạy sau khi blocker source/CPU trước nó đã được đóng.

---

# 2. Initial R6 blocker — measured hardware cost không drive K

## Observation

Controller/planner ban đầu có thể load measured profile, nhưng active serving path không chứng minh rằng cost lookup thực sự thay đổi final K.

Negative control cho thấy:

```text
cost surface A
cost surface B
→ same final K
```

với:

```text
zero effective cost lookups in active decision
```

### Classification

```text
IMPLEMENTATION / SCIENTIFIC CAUSALITY BLOCKER
R6-SOURCE = NOT FIXED
R6-CPU = NOT SATISFIED
```

### Impact

Không được claim:

> Hardware-Adaptive chọn K từ measured hardware cost.

Allowed claim tại thời điểm đó chỉ là:

> Hardware profile có thể được load/validate, nhưng active K path chưa chứng minh cost causality.

### Resolution direction

Tách rõ:

```text
profile existence
!=
profile admission
!=
profile lookup
!=
cost affects final K
```

và buộc phải có CPU causal witness qua active planner.

---

# 3. G0 research blockers — policy underspecified

Full research package xác định 5 nhóm blocker chính:

```text
A. survival/history
B. uncertainty
C. decision-shape forecasting
D. off/warmup/probe/degraded modes
E. feedback finalization
```

Các finding quan trọng:

- feedback có thể tồn tại trước stop/length clipping;
- pending feedback là structurally possible;
- planner chỉ có partial future-shape knowledge;
- padded EAGLE dùng shared K;
- K0 không đồng nghĩa zero drafter/system overhead;
- runtime/product identity chưa đủ để exact-match consumer profile;
- A10/A10G component artifacts tồn tại nhưng không phải admitted consumer profiles.

### Classification

```text
TECHNICAL REVIEW REQUIRED
```

### Result

Không patch production ngay. Thay vào đó tạo design/spec/test amendment candidate trước.

---

# 4. G1-C blocker — DecisionState transport thiếu dữ liệu

## Observation

Scheduler biết nhiều field nhưng active planner không nhận đủ:

```text
observed runtime identity
full scheduled cohort / pure-decode state
old verification widths
context snapshot/bucket
runtime/queue mode
```

CPU witnesses cho thấy các runtime situations khác nhau có thể collapse thành cùng planner input.

### Impact

Một deterministic cost key không thể phân biệt state nếu channel input không mang thông tin đó.

### Classification

```text
TEAM TECHNICAL ALIGNMENT REQUIRED — DECISION STATE TRANSPORT
```

### Fix

Thêm typed host-side transport:

```text
src/server/decision_state.py
src/server/adaptive_scheduler.py
src/server/budget_planner.py
tests/test_decision_state_transport.py
```

Nguyên tắc:

```text
unknown remains unknown
requested hardware != observed hardware
configured quantizer != loaded/effective quantizer
old verification width != future proposal width
```

### Result

```text
G1-C PASS
G1-E PASS
```

Feedback finalization cũng được chứng minh source/CPU trong supported domain mà không cần shared native callback mới.

---

# 5. SECP-v1.1 CPU implementation — measured cost bắt đầu causally drive K

Production path được bổ sung:

```text
survival history
pending feedback
consumer profiles
cost governor
planner integration
telemetry
exact state lookup
strict/off/compat/warmup/probe behavior
```

Fresh causal witness đã chứng minh:

```text
same non-cost state
same identity/runtime
same candidates
only consumer-cost surface changes
→ final K changes
```

### Result

```text
R6-SOURCE PASS
R6-CPU PASS
R6-GPU UNVERIFIED
R6-PERFORMANCE UNVERIFIED
```

### Scientific limit

CPU synthetic cost fixtures không phải measured GPU cost.

---

# 6. G4 identity blocker — configured metadata chưa phải observed identity

## Problem

Live compatibility cần worker-bound observations:

```text
GPU product
SM
VRAM
driver
CUDA
PyTorch
vLLM
effective dtype
loaded/effective quantizer
```

Không được suy:

```text
requested selector "A10G"
→ observed GPU = A10G
```

hoặc:

```text
configured awq_marlin
→ loaded quantizer = awq_marlin
```

### Fix

Thêm worker-owned identity observation và exact admission logic.

### Result

Focused identity CPU tests pass.

```text
G4-I SOURCE/CPU PASS
LIVE still unverified at that stage
```

---

# 7. Canonical collector gap

## Observation

Legacy adapter output:

```text
consumer_profile=false
```

và thiếu complete binding tới:

```text
DecisionStateKey
source SHA
runtime/config hash
cohort provenance
exact K/token/finalization
retirement timing
```

### Fix

Project-local collector được thêm:

```text
experiments/eagle_rw_collector.py
```

với typed immutable records:

```text
source
plan
cohort
decision binding
retirement record
canonicality validation
```

### Result

```text
G4-C SOURCE/CPU PASS
G4 LIVE UNVERIFIED
```

---

# 8. Historical live blocker — external upload authorization

Prepared identity harness cần upload reviewed source sang Modal.

Automatic authorization review chặn external upload vì user chưa explicit authorize.

### Classification

```text
GOVERNANCE / EXTERNAL SIDE-EFFECT BOUNDARY
```

### Resolution

User explicit authorize bounded identity-only preflight.

### Status

```text
RESOLVED
```

---

# 9. Historical live blocker — remote harness path assumption

## Failure

Remote import crash:

```text
/root/identity_modal.py:5
OUT.parents[1]
IndexError: 1
```

## Root cause

Harness giả định local checkout depth tồn tại trên remote `/root`.

### Classification

```text
EVIDENCE HARNESS BUG
NOT A PRODUCTION BUG
```

### Fix

Dùng explicit local inputs và explicit remote project root.

### Result

Path regression CPU tests pass.

---

# 10. Live identity PASS và A10/A10G product drift

Các bounded live runs cho thấy cùng requested Modal selector có thể trả về hai observed product identities khác nhau.

Observed example A:

```text
NVIDIA A10
SM86
VRAM ~22587 MiB
driver 610.57.04
```

Observed example B:

```text
NVIDIA A10G
SM86
VRAM ~22609.1875 MiB
driver 620.70.06
```

### Important decision

```text
A10 != A10G
```

Không alias theo:

```text
same SM86
same approximate VRAM
same cloud selector family
```

### Classification

```text
PRODUCT IDENTITY / PROFILE COMPATIBILITY
```

### Final rule

Observed product chọn branch/profile.

Requested selector chỉ là provenance.

---

# 11. Collector bootstrap circularity

## Problem

Canonical collector path lúc đầu yêu cầu:

```text
admitted consumer profile
+
scored cost candidate
+
cost_policy_applied=true
```

nhưng initial consumer profile lại cần canonical measured records.

Circular dependency:

```text
profile
→ needed for canonical record
→ canonical record
→ needed to create profile
```

### Fix

Tách:

```text
CollectionPurpose.CALIBRATION
CollectionPurpose.POLICY_VALIDATION
```

`CALIBRATION`:

- profile-independent;
- frozen action;
- pre-action DecisionState key;
- no cost-policy claim;
- no contamination of normal survival.

`POLICY_VALIDATION`:

- requires admitted profile;
- requires actual scored candidate;
- requires cost-policy application.

### Result

Bootstrap CPU PASS.

Reviewed checkpoint sau phase này:

```text
1349aef22aa801496df2b709a53838e6d80d23d2
```

---

# 12. Git provenance packaging failure

## Failure

Runtime-only Git package omitted a tracked blob:

```text
.agents/rules/vietnamese-capitalization.md
```

Remote Git reported invalid/missing object.

### Root cause

Sai assumption:

```text
runtime file allowlist
==
Git object closure
```

### Fix

Dùng self-contained Git bundle.

Verification:

```text
bundle verify
empty repository reconstruction
exact HEAD
clean checkout
git fsck --full
missing blob witness
runtime manifest
```

### Result

Source provenance gate PASS.

### Lesson

```text
files needed at runtime
!=
objects needed to reconstruct commit
```

---

# 13. Existing profile inventory — no admissible canonical A10/A10G profile

Repository-wide inventory phân loại artifacts thành:

```text
canonical measured consumer profile
legacy component profile
synthetic diagnostic profile
reference only
incomplete/unknown
```

### Result

```text
A10 canonical profile: MISSING
A10G canonical profile: MISSING
```

Legacy/component timing artifacts không có đủ:

```text
pulse objective
exact DecisionState
matched K0
positive cells
canonical cohort provenance
current source/runtime binding
```

### Rule

Không promote legacy profile chỉ vì hardware name giống.

---

# 14. Product-specific smoke PASS

Bounded manual smoke đã thành công cho cả hai observed products.

## A10

```text
manual_A10_smoke_20261122T120042Z
LIVE CALIBRATION SMOKE PASS
canonical records: 7
```

## A10G

```text
manual_A10G_smoke_20261122T115552Z
LIVE CALIBRATION SMOKE PASS
canonical records: 7
```

### What this proves

```text
source provenance works
live identity works
product routing works
collector calibration path works
OFF/ON observer-fidelity smoke works
```

### What this does NOT prove

```text
publishable profile exists
50 cohorts/cell complete
live policy consumes measured profile
R6-GPU PASS
performance improvement
```

---

# 15. Full campaign protocol blocker — off-diagonal cells không đo được bằng steady-state fixed-K

For current domain:

```text
B1
base K3
old verification width = 0
context [0,256)
```

active scoring cần exact cells:

```text
(0,0)
(0,2)
(0,3)
```

Nhưng fixed-action steady-state path tạo:

```text
old0 → q2 transition → later (2,2)
old0 → q3 transition → later (3,3)
```

Collector cũ loại transition windows.

### CPU proof

```text
(0,2) / (0,3) first transition
→ excluded anchor

later steady records
→ (2,2) / (3,3)
```

Diagonal-only profile bị active planner exact admission reject.

### Classification

```text
TECHNICAL REVIEW REQUIRED — CAMPAIGN PROTOCOL
```

### Result

Campaign harness chưa được build ở phase này.

---

# 16. First-response estimand — valid measurement nhưng wrong objective horizon

Candidate estimand:

```text
pre-action commit
→ exact first post-action retirement
```

là valid controlled response-latency measurement.

Nhưng reward của speculative q có thể xuất hiện ở retirement kế tiếp.

### Counterexample class

```text
future predicted yield / first-response latency
```

có thể rank q khác với:

```text
actual/matched useful tokens / matched common-horizon time
```

### Classification

```text
TECHNICAL REVIEW REQUIRED — ESTIMAND STILL AMBIGUOUS
```

### Decision

Không dùng first-response latency làm final consumer cost.

---

# 17. Objective frozen — SECP-PULSE-2R-v1

Final chosen finite-horizon objective:

```text
old0
→ pulse q
→ retirement #1
→ mandatory K0 continuation
→ retirement #2
```

Objective:

```text
J_2R(s,q;G)
=
E[U1 + U2 | s, do(q,0), G]
/
E[t1 - t0 | s, do(q,0), G]
```

Predicted cycle reward:

```text
Y_cycle(q) = 2 + sum(S_j, j=1..q)
```

Thus:

```text
q0: 2
q2: 2 + S1 + S2
q3: 2 + S1 + S2 + S3
```

Conservative score:

```text
R_lower(q) = Y_cycle(q) / C_high(q)
R_upper(0) = 2 / C_low(0)
```

Positive q supported only if:

```text
R_lower(q) > R_upper(0)
```

### Exact campaign cells

```text
(old=0, future=0)
(old=0, future=2)
(old=0, future=3)
```

No K4 for current base3 domain.

---

# 18. Production migration to SECP-PULSE-2R-v1

Production migration applied P2R-01..P2R-20.

Key modules:

```text
src/flexsds/core/eagle_consumer_profile.py
src/proposer/cost_policy.py
src/proposer/pulse_policy.py
src/server/budget_planner.py
src/server/decision_state.py
src/server/adaptive_scheduler.py
src/server/lineage.py
src/server/lineage_executor.py
experiments/eagle_rw_collector.py
```

### Causal CPU witness

Fixed:

```text
state
identity
runtime
survival = [.75,.5,.25]
candidates = [0,2,3]
predicted rewards = [2,3.25,3.5]
```

Surface A:

```text
costs = [1,1.1,2]
→ K2
```

Surface B:

```text
costs = [1,2,1.1]
→ K3
```

Only costs change.

### Result

Reported migration gate:

```text
746 distinct PASS
0 failed
0 errors
0 skipped
0 deselected
```

Traceability:

```text
20 requirements
20 mapped
0 unmapped
0 duplicates
```

Reviewed source checkpoint:

```text
ac395ed035374ea3937e79b4cea077feb59bfa1a
```

---

# 19. Dual-product campaign specification package

Campaign implementation authority được tách thành hai documents:

```text
SECP_PULSE_2R_Dual_Product_Campaign_Spec_v1.md

SECP_PULSE_2R_Dual_Product_Campaign_Test_Plan_v1.md
```

Frozen campaign semantics:

```text
products = A10, A10G separately
cells = q0, q2, q3
50 predetermined slots/cell
no replacement-until-success
5000 cohort-level bootstrap resamples
95% CI
seed 731
durable resumable state
raw hash verification
single writer
bounded allocation loop
dry-run
fake executor
separate product profiles
```

---

# 20. Campaign spec conflict — unsupported product behavior

## Conflict

One section/test contract yêu cầu:

```text
unsupported observed GPU
→ consume one allocation
→ zero scientific slots
→ continue within budget
```

Trong khi một section khác có thể được đọc là:

```text
unsupported product
→ campaign-wide stop
```

### Resolution

Authoritative behavior:

```text
unsupported product
!= scientific slot failure
!= hard invariant failure
```

Required behavior:

```text
record observed identity
allocation_count += 1
attempt zero scientific slots
mutate no A10/A10G slot
release worker
continue if budget remains
```

Campaign stops only at:

```text
MAX_ALLOCATIONS_REACHED
```

nếu repeated unsupported allocations consume hết budget.

### Hard-stop examples remain

```text
source provenance mismatch
plan/state/hash corruption
unsupported campaign schema/objective
raw evidence integrity failure
single-writer violation
scientific slot invariant failure after slot begins
publication-blocking fixed-slot failure
```

---

# 21. Campaign harness implementation — current local state

Codex implemented campaign machinery including:

```text
experiments/eagle_pulse_campaign.py
experiments/eagle_pulse_campaign_worker.py
experiments/eagle_pulse_campaign_modal.py
experiments/eagle_pulse_campaign_freeze.py
tests/test_eagle_pulse_campaign.py
tests/test_eagle_pulse_campaign_worker.py
tests/test_eagle_pulse_campaign_cli.py
tests/test_eagle_pulse_campaign_publication.py
tests/test_eagle_pulse_campaign_source_handoff.py
docs/SECP-PULSE-2R-campaign.md
```

plus local audit/evidence helpers under:

```text
audit/r6_secp_v11_gpu_readiness_20261122/
dual_product_campaign_harness_v1/
```

### Implemented concepts

```text
plan/state validation
deterministic scheduling
product routing
fixed scientific slots
raw evidence persistence
resume
profile publication/admission
fake executor
dry-run
live adapter
worker acknowledgment
source handoff
bundle verification
```

---

# 22. Campaign CPU verification progression

Reported progression:

```text
69 tests PASS
→ 115 tests PASS
→ 121 campaign tests PASS under external-action guard
```

Then ordered regression gates were run.

### Guard bug found

Replacing `subprocess.Popen` with a function broke an asyncio subclass during CPU/PyTorch import on Windows.

### Fix

Guard changed to subclass original `Popen` instead of replacing it with a plain function.

### Result

Remaining regression gates pass.

Reported:

```text
unchanged 746-test PULSE baseline PASS
```

No Modal/GPU action occurred.

---

# 23. Bundle / source-manifest Windows issues

## Problem A — Windows path-length

Clean-checkout verification hit Windows path-length limits.

### Fix

Enable long-path support for verification process only.

Global Git config left unchanged.

---

## Problem B — CRLF normalization mismatch

Manifest verification compared:

```text
working source normalized to Git LF
vs
reconstructed checkout raw CRLF
```

and reported mismatch.

A retained historical JSON also had CRLF bytes stored in Git.

### Root cause

Source verification rule was not applying the same normalization contract to all compared text representations.

### Fix

Make Git-LF/source-text normalization explicit and symmetric while preserving raw Git object/bundle verification.

### Result

Reported:

```text
bundle reconstruction PASS
1,095 source-manifest checks PASS
```

---

# 24. Final worker timing bug — export happened while pulse cycle in flight

## Observation

Final worker review found source-validating export was invoked before native pulse cycle fully finalized.

### Impact

Export could observe/validate data while timed scientific cycle was still active.

That violates desired boundary:

```text
measurement
→ t1 finalization
→ only then export/validation
```

### Fix

Move export after cycle finalization.

Add focused assertions:

```text
no export inside timed span
noncanonical completed export → failed slot
```

### Result

Reported native CPU checks pass for:

```text
q0
q2
q3
```

---

# 25. Synthetic full-campaign verification

Reported final local non-live checks:

```text
exact Git Bash dry-run PASS
source-bound CPU fake campaign PASS
both A10 and A10G synthetic completion PASS
post-run validator PASS
```

Live campaign state remained:

```text
READY
0 live allocations
0 live raw profile records
```

### Important limit

Synthetic completion proves orchestration behavior, not real GPU profile validity.

---

# 26. Final evidence-finalization interruption

## Observation

Codex attempted one final evidence/audit command to update:

```text
traceability.json
final-review.md
test-results-summary.json
artifact-sha256.json
final-checks.json
```

and re-check:

```text
tracked/index clean
git diff --check
required artifacts present
live state READY / 0 allocations
```

But the command was declined by auto-review, then Codex quota ended.

### Classification

```text
HANDOFF / EVIDENCE FINALIZATION INCOMPLETE
NOT A PROVEN SOURCE FAILURE
```

### Impact

Tracked implementation may already be committed/clean, but final audit metadata may not fully reflect the last targeted fixes.

Do not claim final artifact ledger is complete unless regenerated/verified.

---

# 27. Current local Git state

Latest user-run handoff script observed:

```text
branch:
audit/flexsds-v023-hardwareadaptive-20261120

local HEAD:
218a9b1e2fd78118f1ca3c387c3b162bd9c2ff2b

remote HEAD:
4c76830706d6d60d715d962764e592b7b184fc42

behind:
0

ahead:
10
```

Tracked/index cleanliness gate passed before diff export.

### Interpretation

```text
remote is ancestor of local
→ normal fast-forward push is structurally possible
```

provided no remote update occurs before push.

Git fast-forward semantics preserve existing remote history because remote tip remains ancestor of local tip.

---

# 28. Handoff diff artifacts

Exact committed tree diff was exported before push attempt:

```text
logs/problems/r6_secp_pulse_2r_20261123/
handoff-218a9b1e2fd7.diff
```

Supporting files:

```text
handoff-218a9b1e2fd7-commits.txt
handoff-218a9b1e2fd7-files.txt
handoff-218a9b1e2fd7-stat.txt
```

Purpose:

```text
preserve exact committed local-vs-remote handoff delta
before moving remote ref
```

---

# 29. Current push blocker — `git diff --check` historical whitespace

## Observation

Full 10-commit range:

```text
origin/audit/... .. HEAD
```

fails `git diff --check`.

Examples reported:

```text
trailing whitespace
new blank line at EOF
```

in design documents including:

```text
g0-decision-review.md
replacement-policy-cpu-validation.md
scalar-eagle-cost-policy-spec-amendment-candidate.md
```

### Git semantics

`git diff --check` treats trailing whitespace and related whitespace patterns as errors and exits nonzero.

### Classification

```text
HANDOFF QUALITY GATE
LIKELY HISTORICAL FORMATTING DEBT
CURRENT HEAD-ONLY ATTRIBUTION NOT YET PROVEN IN THIS LOG
```

### Important distinction

Full-range whitespace failure:

```text
does NOT automatically prove
campaign HEAD commit is semantically wrong
```

It also does not prove the errors belong only to historical commits until HEAD-only checks are run.

### Required verification

Before push:

```text
git show --check --oneline HEAD
git diff --check HEAD^ HEAD
```

If HEAD-only is clean, classify remaining full-range errors as historical formatting debt for this handoff rather than campaign correctness blocker.

If HEAD-only is not clean, add a narrow cleanup commit; do not rewrite all 10 commits merely to make the range clean.

---

# 30. Untracked evidence / raw-artifact situation

Repository contains many untracked paths:

```text
audit/
historical GPU runs
raw logs
smoke state
old handoff zips
replacement-policy diff artifacts
```

### Rule

Do not run:

```text
git add .
git add audit/
git clean
git reset --hard
```

A normal `git push` sends committed Git objects/references, not arbitrary untracked working-tree files.

### Current handoff principle

```text
push committed history only
preserve untracked historical evidence locally
do not accidentally bulk-commit raw artifacts
```

---

# 31. External technical cross-checks

## Git whitespace gate

Official Git documentation defines `git diff --check` as a check for conflict markers and whitespace errors, with nonzero exit when such problems are found.

This supports the interpretation of the current handoff script stopping at the whitespace gate.

## Git push

Official Git documentation defines a branch update as fast-forward when the new tip descends from the old remote tip.

Current observed:

```text
behind=0
ahead=10
```

is compatible with normal fast-forward push if the remote branch remains unchanged.

No `--force` is required or recommended.

## Atomic durable state

Python `os.replace()` replaces the destination and, when successful on the same filesystem, provides atomic rename semantics on supported platforms.

This supports the campaign state's temp-file → replace design.

## Modal bounding

The campaign design intentionally keeps:

```text
max_containers = 1
retries = 0
```

so product allocation behavior is bounded by campaign state rather than hidden retry loops.

---

# 32. Current blocker summary

| Finding | Type | Severity | Status |
|---|---|---:|---|
| Initial measured-cost path did not affect K | implementation/scientific | BLOCKER historically | RESOLVED CPU |
| DecisionState transport insufficient | shared interface | BLOCKER historically | RESOLVED |
| Feedback finalization association | lifecycle | BLOCKER historically | RESOLVED CPU |
| Worker-bound identity incomplete | provenance | MAJOR historically | RESOLVED |
| Canonical collector provenance incomplete | measurement | MAJOR historically | RESOLVED CPU |
| Profile-before-collector circularity | protocol | BLOCKER historically | RESOLVED via CALIBRATION/POLICY_VALIDATION split |
| Git source package missing object closure | provenance | BLOCKER historically | RESOLVED via bundle |
| A10/A10G product drift | hardware identity | MAJOR | RESOLVED semantically: separate products |
| No canonical pre-existing A10/A10G profile | evidence | BLOCKER for R6-GPU | STILL TRUE |
| Fixed-K steady-state cannot measure `(0,2)/(0,3)` | scientific protocol | BLOCKER historically | RESOLVED by PULSE objective |
| First-response cost horizon mismatch | scientific estimand | BLOCKER historically | RESOLVED |
| SECP-PULSE-2R objective | scientific contract | INFO | FROZEN |
| Active PULSE CPU migration | implementation | MAJOR | PASS |
| Dual-product campaign harness | implementation | MAJOR | LOCALLY IMPLEMENTED / CPU-TESTED |
| Unsupported-GPU spec conflict | spec | MAJOR | RESOLVED |
| Windows Popen guard bug | test infrastructure | MINOR/MAJOR for gate | RESOLVED |
| Windows path-length/CRLF bundle issues | packaging | MAJOR for handoff | RESOLVED locally |
| Export-before-cycle-finalization | measurement boundary | MAJOR | RESOLVED locally |
| Final evidence metadata regeneration | handoff | MAJOR | OPEN |
| Full 10-commit whitespace gate | handoff quality | MAJOR | OPEN REVIEW |
| Remote branch update | handoff/provenance | BLOCKER for moving workspace | NOT YET VERIFIED |
| Real A10 profile | GPU evidence | BLOCKER for R6-GPU | NOT COLLECTED |
| Real A10G profile | GPU evidence | BLOCKER for R6-GPU | NOT COLLECTED |
| Live POLICY_VALIDATION | GPU evidence | BLOCKER for R6-GPU | NOT RUN |
| Performance claim | experiment | BLOCKER | NOT AUTHORIZED / NOT RUN |

---

# 33. Current claim ceiling

Allowed:

> The active source has migrated to `SECP-PULSE-2R-v1`, and CPU evidence demonstrates exact pulse-profile admission, deterministic K0 continuation, cost-causal K selection, lineage/finalization handling and a resumable dual-product campaign harness. Both A10 and A10G bounded live calibration smoke paths have passed historically. The real full profile campaigns and live policy-consumption validation remain unexecuted.

Not allowed:

> Canonical A10 and A10G consumer profiles now exist.

Not allowed:

> R6-GPU passes.

Not allowed:

> Hardware-aware PULSE improves throughput/TPOT.

Not allowed:

> The campaign harness has been fully handed off remotely until remote branch SHA equals the reviewed local HEAD.

---

# 34. Exactly one current WIP

Current WIP is **not** more policy research.

Current WIP:

```text
finish source/evidence handoff
→ verify HEAD-only whitespace
→ push committed 10-commit fast-forward history
→ verify remote SHA == local SHA
```

Only after source handoff is closed:

```text
move to another workspace
→ restore branch at exact remote SHA
→ regenerate/verify final campaign evidence metadata if needed
→ user manually runs real dual-product campaign
```

---

# 35. Immediate next action

Run from repository:

```bash
git show --check --oneline HEAD
git diff --check HEAD^ HEAD
```

Decision:

```text
HEAD-only clean
→ normal push --dry-run
→ normal push
→ fetch
→ verify remote SHA == local HEAD

HEAD-only dirty
→ create narrow cleanup commit
→ rerun HEAD-only check
→ then push
```

Do not force-push.

Do not rewrite the 10-commit history merely to remove historical whitespace unless technical review explicitly decides that history cleanup is required.

---

# 36. Handoff completion criterion

Handoff is complete only when all are true:

```text
1. local HEAD known;
2. tracked/index clean;
3. final committed diff preserved;
4. remote is not ahead/diverged;
5. HEAD-only whitespace decision recorded;
6. normal push succeeds;
7. fetch succeeds;
8. remote branch SHA == local HEAD;
9. no raw/untracked evidence accidentally committed;
10. next workspace checks out exact remote SHA.
```

Target local HEAD currently:

```text
218a9b1e2fd78118f1ca3c387c3b162bd9c2ff2b
```

---

# 37. Relationship to historical problem log

Đọc cùng:

```text
logs/problems/2026-11-20-flexsds-v023-hardware-adaptive-audit-blockers.md
```

Historical file đó ghi phase:

```text
Hardware-Adaptive audit
→ measured costs not active
→ correctness/runtime/scientific blockers
```

File hiện tại nối tiếp timeline:

```text
R6 technical closure
→ identity/collector/bootstrap
→ live provenance blockers
→ A10/A10G smoke
→ estimand redesign
→ SECP-PULSE-2R migration
→ dual-product campaign harness
→ current Git handoff/push gate
```

Không dùng file hiện tại để rewrite historical evidence của file 20/11.

---

# 38. Defense-ready narrative

Nếu được hỏi “tại sao task kéo dài?”:

Ban đầu thiếu cost-driven decision có vẻ chỉ là implementation gap. Tuy nhiên một measured-cost policy chỉ có scientific validity khi:

```text
runtime state được định danh đúng
+
profile key match đúng state
+
measurement đo đúng causal horizon
+
reward và cost dùng cùng horizon
+
profile provenance reconstruct được
+
A10/A10G không bị alias
+
campaign không replacement-until-success
+
source chạy live được freeze/reconstruct
```

Mỗi blocker được đóng theo thứ tự:

```text
state transport
→ cost causality
→ identity
→ collector
→ bootstrap
→ source bundle
→ product separation
→ smoke
→ off-diagonal measurement
→ estimand
→ PULSE objective
→ active production migration
→ campaign harness
→ handoff
```

Negative results không bị bỏ; chúng trở thành evidence xác định boundary cho bước kế tiếp.

---

# 39. Current one-line status

```text
SECP-PULSE-2R SOURCE/CPU PASS
→ A10/A10G LIVE SMOKE PASS
→ DUAL-PRODUCT CAMPAIGN HARNESS LOCALLY CPU-TESTED
→ FULL PROFILES NOT YET COLLECTED
→ LIVE POLICY_VALIDATION NOT RUN
→ LOCAL BRANCH 10 COMMITS AHEAD / 0 BEHIND
→ FULL-RANGE DIFF CHECK BLOCKED BY WHITESPACE
→ HEAD-ONLY ATTRIBUTION + REMOTE PUSH VERIFICATION STILL REQUIRED
```

---

# 40. External references used for handoff semantics

- Git `git diff --check`: https://git-scm.com/docs/git-diff
- Git fast-forward / push rules: https://git-scm.com/docs/git-push
- Python `os.replace`: https://docs.python.org/3/library/os.html
- Modal Python SDK / container controls: https://modal.com/docs/reference/modal.App

These references only interpret tooling semantics. They do not replace local source, tests, Git history or run artifacts as project source of truth.
