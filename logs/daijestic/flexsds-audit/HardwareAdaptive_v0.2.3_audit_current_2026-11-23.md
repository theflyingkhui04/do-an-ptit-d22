# FlexSDS Hardware-Adaptive Audit — Báo cáo cập nhật hiện trạng

- **Ngày tổng hợp:** 23/11/2026
- **Task:** Cập nhật audit Hardware-Adaptive / SECP-PULSE-2R theo hiện trạng mới nhất, giữ nguyên provenance và claim discipline
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Working branch:** `audit/flexsds-v023-hardwareadaptive-20261120`
- **Current local HEAD:** `218a9b1e2fd78118f1ca3c387c3b162bd9c2ff2b`
- **Remote branch HEAD quan sát gần nhất:** `4c76830706d6d60d715d962764e592b7b184fc42`
- **Git relation tại handoff check gần nhất:** `0 behind / 10 ahead`
- **Current active policy objective:** `SECP-PULSE-2R-v1`
- **Current production migration checkpoint lineage:** PULSE migration đã được CPU-verified; campaign harness nằm trên các local successor commits
- **Pinned runtime authority:** `vLLM 0.30.0`
- **Current audit status:** `SOURCE/CPU VERIFIED — LIVE PROFILE CAMPAIGN READY BUT NOT EXECUTED`
- **Current R6 status:**
  - `R6-SOURCE PASS`
  - `R6-CPU PASS`
  - `R6-GPU UNVERIFIED`
  - `R6-PERFORMANCE UNVERIFIED`
- **Performance conclusion:** **NOT YET AUTHORIZED / NOT EVALUATED**
- **Canonical full A10 profile:** **NOT YET COLLECTED**
- **Canonical full A10G profile:** **NOT YET COLLECTED**
- **Live POLICY_VALIDATION:** **NOT RUN**
- **Real dual-product full campaign:** **NOT RUN**
- **Branch push/handoff:** **LOCAL COMMITS EXIST; REMOTE FAST-FORWARD PUSH NOT YET VERIFIED COMPLETE**

> Đây là bản cập nhật trực tiếp từ interim audit ngày 20/11/2026.
>
> Báo cáo này **không hạ bất kỳ finding đã được verified trước đó**. Các hạng mục chỉ được nâng trạng thái khi có evidence mới. Hạng mục chưa có live/GPU evidence vẫn giữ claim ceiling tương ứng.
>
> Interim audit không định nghĩa một numeric score chính thức, nên bản cập nhật này **không tự bịa một điểm tổng mới**. Thay vào đó, audit dùng status uplift có truy vết: `NOT VERIFIED → SOURCE VERIFIED → TEST/CPU VERIFIED → HISTORICAL/LIVE GPU VERIFIED → CLOSED`.

---

# 1. Executive summary

So với interim audit 20/11, trạng thái đã tiến bộ đáng kể ở blocker quan trọng nhất: **R6 — measured hardware cost có tham gia active K decision hay không**.

Interim finding:

```text
profile loaded/validated/injected
BUT
active controller did not consume measured cost to select K
```

Current finding:

```text
typed pulse profile
→ exact DecisionState admission
→ conservative measured-cost scoring
→ active planner
→ selected pulse K
→ deterministic K0 continuation
→ lineage/finalization/telemetry
```

đã được chứng minh bằng CPU causal witness với chỉ cost surface thay đổi:

```text
same state
same identity/runtime
same survival
same candidate set
only pulse cost changes
→ final K changes
```

Do đó finding cũ:

```text
R6 — Hardware profile: CHƯA ĐẠT
```

được nâng thành:

```text
R6-SOURCE PASS
R6-CPU PASS
R6-GPU UNVERIFIED
```

Đồng thời, một số vấn đề trước đây chưa tồn tại trong baseline cũ đã được phát hiện và xử lý:

1. DecisionState transport không đủ cho exact profile lookup.
2. Worker-observed runtime identity thiếu driver/effective quantizer.
3. Collector canonicality từng phụ thuộc circularly vào profile đã tồn tại.
4. Runtime Git package từng thiếu Git object closure.
5. Modal selector `A10G` không đảm bảo observed product là `NVIDIA A10G`; đã quan sát cả A10 và A10G.
6. Fixed-K steady-state measurement không tạo được exact off-diagonal cells `(old=0,future=2/3)`.
7. First-response latency không cùng reward horizon với speculative benefit.
8. Objective được redesign/freeze thành matched two-retirement `SECP-PULSE-2R-v1`.
9. Active production path đã migrate sang objective mới.
10. Dual-product resumable full-profile campaign harness đã được implement và CPU/synthetic-test local.
11. Git/source handoff vẫn còn bước cuối: remote branch chưa được xác minh trùng local HEAD.

Điểm quan trọng nhất: current project **không còn bị block bởi việc “cost không drive K” ở source/CPU layer**. Blocker hiện tại chuyển sang **measured live profile collection + live policy consumption**, sau khi hoàn tất handoff Git.

---

# 2. Scope và claim boundary

Audit vẫn tách ba tầng evidence:

1. **SOURCE** — code/runtime contract tồn tại và active path thực sự sử dụng nó.
2. **CPU / UNIT / INTEGRATION** — invariant, causal behavior, admission, lineage, resume, publication logic được test.
3. **LIVE GPU** — behavior thực sự xảy ra với exact product/runtime/model/source trên GPU.

Tám yêu cầu PM từ interim audit vẫn giữ:

1. CLI → runner → scheduler → planner → controller.
2. Acceptance feedback, drafted/accepted, request attribution, EMA/history/warmup.
3. Dynamic K từ decision xuống drafting thực.
4. Memory/VRAM/KV guard.
5. Fallback/probe/recovery.
6. Hardware profile / measured cost thực sự drive K.
7. Per-request/batch semantics.
8. Telemetry reconstruct decision → execution → feedback.

Bổ sung cho current phase:

9. Exact runtime identity / DecisionState profile admission.
10. Canonical cost measurement protocol.
11. Source provenance / self-contained execution package.
12. Resumable fixed-slot campaign / publication protocol.
13. A10/A10G product separation.
14. Matched reward/cost horizon.

---

# 3. Evidence classification hiện tại

- **SOURCE VERIFIED:** đọc/traced source đúng checkpoint.
- **CPU VERIFIED:** unit/integration/native-method/fake-executor tests chứng minh invariant.
- **CAUSAL CPU VERIFIED:** thay đúng một biến causal và active result đổi theo dự đoán.
- **HISTORICAL GPU VERIFIED:** retained prior GPU evidence tồn tại và provenance đủ cho claim hẹp.
- **LIVE SMOKE VERIFIED:** bounded current live smoke đã chạy, nhưng chưa phải campaign/benchmark.
- **LIVE PROFILE VERIFIED:** chỉ dùng khi full fixed-slot profile campaign hoàn thành và publication/admission pass.
- **LIVE POLICY VERIFIED:** chỉ dùng sau POLICY_VALIDATION với measured admitted profile.
- **PERFORMANCE VERIFIED:** chỉ dùng sau benchmark protocol hợp lệ.
- **UNKNOWN / UNVERIFIED:** chưa có evidence đủ.
- **INFERENCE:** suy luận có căn cứ nhưng không thay runtime evidence.

Không đồng nhất:

```text
dry-run != live evidence
smoke != full campaign
smoke != benchmark
client concurrency != effective runtime batch
profile loaded != profile admitted
profile admitted != profile causally consumed
component timing != pulse-cycle consumer cost
requested cloud GPU != observed product
same SM architecture != same hardware-cost profile
CPU synthetic cost != measured GPU cost
```

---

# 4. Audit uplift so với 20/11

| Hạng mục | Interim 20/11 | Current 23/11 | Uplift |
|---|---|---|---|
| Runtime wiring | Đạt một phần | Source/CPU verified; historical live retained | ↑ |
| Feedback/lineage | Đạt một phần | G1-E source/CPU pass; pulse-cycle lineage/finalization CPU verified | ↑ |
| Dynamic K | Đạt một phần | Active PULSE K scoring/continuation CPU verified; historical live K retained | ↑ |
| Memory guard | Đạt một phần | Không giảm; vẫn cần live pressure→recovery closure | = |
| Fallback/probe | Đạt một phần | Không giảm; historical live retained, exit/recovery vẫn mở | = |
| Hardware profile drives K | **Chưa đạt** | **R6-SOURCE PASS / R6-CPU PASS** | **↑↑ major** |
| Request/batch | Đạt một phần | DecisionState/incarnation isolation + campaign fresh-cohort CPU verified | ↑ |
| Telemetry | Đạt một phần | PULSE cycle telemetry reconstructable on CPU; live campaign pending | ↑ |
| Runtime identity | Chưa closed | Worker-bound identity contract + A10/A10G separation | ↑ |
| Profile provenance | Chưa canonical | Typed PULSE profile + exact admission/publication CPU verified | ↑ |
| Measurement estimand | Chưa explicit | `SECP-PULSE-2R-v1` frozen | ↑ |
| Full profile campaign | Chưa tồn tại | Harness implemented + CPU/synthetic-tested; live not run | ↑ |
| R6-GPU | Unverified | Unverified | = |
| Performance | Not evaluated | Not evaluated | = |

Không có downgrade nào được áp dụng chỉ vì current audit nghiêm ngặt hơn. Các claim cũ hợp lệ vẫn được giữ dưới nhãn historical/current tương ứng.

---

# 5. Current source/provenance state

## Historical primary audit baseline

Interim audit dùng:

```text
develop@8006205a6f19790a776d67155a377def053d2736
package metadata = 0.2.3
```

Historical evidence handoff:

```text
EVIDENCE_COMMIT=1eb7e59
```

## Intermediate R6 checkpoints

Các checkpoint quan trọng sau interim:

```text
4bb093ad4272aa330679d3baa9f5b1797ea397da
→ early R6 policy/transport research baseline

7f38f36de1aec526b5f437dec79dbd51b5e4db88
→ identity/collector reviewed source

1349aef22aa801496df2b709a53838e6d80d23d2
→ calibration bootstrap / live-calibration source

ac395ed035374ea3937e79b4cea077feb59bfa1a
→ active SECP-PULSE-2R production migration checkpoint

218a9b1e2fd78118f1ca3c387c3b162bd9c2ff2b
→ current local campaign-harness successor HEAD
```

Không được hồi tố evidence từ SHA cũ sang SHA mới nếu code/runtime contract thay đổi.

## Current branch handoff relation

Latest local check:

```text
LOCAL HEAD  = 218a9b1e2fd78118f1ca3c387c3b162bd9c2ff2b
REMOTE HEAD = 4c76830706d6d60d715d962764e592b7b184fc42

behind = 0
ahead  = 10
```

Interpretation:

```text
remote tip is ancestor of local tip
→ normal fast-forward push is structurally possible
```

nhưng remote handoff chưa được gọi là complete cho tới khi push + fetch + SHA equality được verify.

---

# 6. Bảng bàn giao theo 8 yêu cầu PM — cập nhật

| Yêu cầu | Current implementation/evidence | Current verdict | Gap còn lại |
|---|---|---|---|
| **1. CLI → runner → scheduler → planner → controller** | Historical live wiring retained. Current source có custom scheduler/planner/pulse-policy path; campaign launcher/source bundle path đã CPU/dry-run verify. | **SOURCE/CPU VERIFIED + HISTORICAL GPU VERIFIED** | Current full campaign chưa live-run trên `218a9b1...`. |
| **2. Feedback, drafted/accepted, history/warmup** | G1-E feasibility PASS; lineage/incarnation/finalization tests; separate PULSE survival/history stratum; calibration excluded from normal history. | **SOURCE/CPU VERIFIED; historical GPU feedback retained** | Live full PULSE campaign/policy feedback chưa run; survival confidence không được suy từ cost CI. |
| **3. Dynamic K xuống drafting thật** | Active PULSE governor chọn q0/q2/q3; one-shot pulse + deterministic K0 continuation; executor records proposal shape including K0. Historical live K 3→0→2 remains valid for old path. | **SOURCE/CPU VERIFIED; historical GPU partially verified** | Current PULSE q2/q3 live policy consumption chưa run. |
| **4. Memory / VRAM / KV guard** | Existing safety/recovery invariants preserved by regression; memory safety dominates pulse selection. | **SOURCE/TEST VERIFIED; historical GPU telemetry** | Full live pressure→restriction→cooldown→recovery vẫn chưa closed. |
| **5. Fallback / probing / recovery** | Existing historical fallback/probe evidence retained; PULSE migration không xóa safety/degraded semantics. | **PARTIALLY VERIFIED** | Live fallback exit/recovery chưa closed; K0 vẫn không đồng nghĩa target-only/zero EAGLE overhead. |
| **6. Hardware profile / measured cost drives K** | Typed pulse profile exact admission; `C_low/C_high`; active `PulseCostGovernor`; CPU causal fixture cost-only change K2→K3. | **R6-SOURCE PASS / R6-CPU PASS** | Chưa có full measured A10/A10G PULSE profile; live cost consumption chưa run. |
| **7. Request / batch** | DecisionState transport, incarnation isolation, fresh-cohort rule, deterministic fixed slots, product-specific state, fake-executor resume tests. | **SOURCE/CPU VERIFIED** | Heterogeneous live effective batch behavior ngoài scoped B1 domain chưa closed. |
| **8. Telemetry** | Pulse decision/profile cell/reward/cost/selected q/retirement1/K0 continuation/retirement2/finalization có reconstructable CPU contract. | **SOURCE/CPU VERIFIED** | Need live campaign records + live POLICY_VALIDATION reconstruction. |

---

# 7. R1 — Runtime wiring

Historical chain:

```text
CLI
→ serving runner
→ AdaptiveAsyncScheduler
→ AdaptiveBudgetPlanner
→ proposer/controller
→ EAGLE proposal
→ target verification
→ feedback
```

Current PULSE path mở rộng thành:

```text
DecisionState
→ exact EaglePulseProfile lookup
→ PulseCostGovernor
→ q pulse
→ exact retirement #1
→ mandatory K0 continuation
→ exact retirement #2
→ pulse-cycle finalization
→ pulse telemetry/history
```

Current source không còn chỉ “có module”; active planner đã được migrate sang PULSE objective.

### Verdict

```text
SOURCE/CPU VERIFIED
HISTORICAL GPU WIRING RETAINED
CURRENT FULL CAMPAIGN LIVE UNVERIFIED
```

---

# 8. R2 — Feedback, history và finalization

Interim evidence:

```text
26 drafted
3 accepted
10 speculative steps
10 attributed feedback records
```

vẫn giữ giá trị historical.

Current improvements:

- pending feedback lifecycle được source/CPU trace;
- unique producer/verification association được test;
- preemption/resume/stale-origin/request-ID reuse cases có CPU witnesses;
- pulse cycle có exact retirement #1 + K0 continuation + retirement #2;
- calibration records không seed normal survival;
- legacy history không được relabel thành pulse history.

### Verdict

```text
G1-E SOURCE/CPU PASS
PULSE FINALIZATION CPU VERIFIED
LIVE FULL-PROFILE FEEDBACK UNVERIFIED
```

---

# 9. R3 — Dynamic K / pulse actuation

Historical live:

```text
K=3
→ K=0
→ K=2 probe
```

Current active PULSE scope:

```text
candidate set = {0,2,3}
```

for:

```text
B=1
base K=3
old width=0
context=[0,256)
```

Current action semantics:

```text
old0
→ q exactly once
→ retirement1
→ forced K0 continuation
→ retirement2
→ lease release
```

The continuation cannot be silently re-decided by ordinary adaptive selection.

### Verdict

```text
SOURCE/CPU VERIFIED
LIVE PULSE POLICY CONSUMPTION UNVERIFIED
```

---

# 10. R4 — Memory / VRAM / KV safety

Current audit does **not** downgrade historical memory evidence.

Existing safety/recovery logic remains covered by regression and continues to dominate PULSE selection.

Still not closed:

```text
normal
→ real live memory pressure
→ restriction
→ cooldown
→ recovery
```

on the current PULSE checkpoint.

### Verdict

```text
PARTIALLY VERIFIED
NO REGRESSION EVIDENCE
LIVE CLOSED-LOOP RECOVERY STILL OPEN
```

---

# 11. R5 — Fallback / probe / recovery

Historical live fallback/probe evidence remains usable.

Important semantic boundary remains:

```text
K0 != Vanilla
K0 != target-only
K0 != zero EAGLE system overhead
```

Current PULSE K0 is additionally meaningful as:

```text
same-engine matched continuation/baseline action
```

but this does not erase residual drafter/runtime residency overhead.

### Verdict

```text
PARTIALLY VERIFIED
```

No downgrade; full live recovery exit remains future work.

---

# 12. R6 — Hardware profile: major uplift

## Interim state

Interim audit correctly concluded:

```text
profile loaded
profile validated
profile injected
BUT
measured cost did not drive active K
```

## Current state

Production migration now uses typed pulse profiles with exact admission.

Required profile semantics include:

```text
objective = SECP-PULSE-2R-v1
exact DecisionState key
observed product identity
runtime/config binding
two-retirement horizon
continuation K0
C_point
C_low
C_high
```

Conservative decision rule:

```text
Y_cycle(q) = 2 + Σ S_j

R_lower(q) = Y_cycle(q) / C_high(q)
R_upper(0) = 2 / C_low(0)
```

A positive q is supported only when:

```text
R_lower(q) > R_upper(0)
```

## Causal CPU witness

Fixed:

```text
survival = [.75,.5,.25]
candidate actions = [0,2,3]
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

Only pulse-cycle costs differ.

### Current R6 verdict

```text
R6-SOURCE PASS
R6-CPU PASS
R6-GPU UNVERIFIED
R6-PERFORMANCE UNVERIFIED
```

This is the largest verified improvement since interim audit.

---

# 13. DecisionState transport và exact profile admission

Earlier planner input collapsed states that differed in:

```text
old verification width
context
pure/mixed cohort
observed runtime identity
```

Typed DecisionState transport now preserves the required distinctions.

Unknown values fail closed rather than becoming “nearest valid” state.

Forbidden substitutions:

```text
A10 → A10G
A10G → A10
old2/old3 diagonal → old0 candidate
nearest context bucket
configured quantizer → effective quantizer
requested GPU selector → observed GPU
```

### Verdict

```text
SOURCE/CPU VERIFIED
```

---

# 14. Runtime identity — A10 và A10G phải tách riêng

Live bounded identity observations đã chứng minh requested resource selector không đủ để xác định scientific product.

Observed examples:

```text
NVIDIA A10
SM86
~22587 MiB
driver 610.57.04
```

và:

```text
NVIDIA A10G
SM86
~22609.1875 MiB
driver 620.70.06
```

Do đó:

```text
same selector
same SM86
≈ same VRAM family
```

không tạo profile equivalence.

### Current rule

```text
observed product selects branch/profile
requested selector is provenance only
```

### Verdict

```text
LIVE IDENTITY BEHAVIOR VERIFIED
PRODUCT ALIASING FORBIDDEN
```

---

# 15. Canonical collector và bootstrap circularity

Historical collector gap:

```text
consumer_profile=false
```

và thiếu complete source/runtime/cohort/DecisionState provenance.

Project collector đã được mở rộng.

Sau đó phát hiện circular prerequisite:

```text
canonical record required admitted profile
profile required canonical records
```

Resolution:

```text
CALIBRATION
!=
POLICY_VALIDATION
```

`CALIBRATION` được phép profile-independent nhưng không claim online cost policy.

`POLICY_VALIDATION` giữ exact admitted profile + scored candidate requirement.

### Verdict

```text
SOURCE/CPU RESOLVED
```

---

# 16. Git source provenance

Runtime file allowlist từng fail vì không bao phủ Git object closure.

Remote error cho một tracked blob thiếu đã chứng minh:

```text
runtime files
!=
commit reconstruction closure
```

Resolution dùng self-contained Git bundle.

Verification contract:

```text
bundle verify
empty-repo reconstruction
exact HEAD
clean checkout
git fsck --full
runtime manifest verification
```

Campaign harness source-handoff path tiếp tục preserve contract này.

### Current local campaign verification

Reported:

```text
bundle reconstruction PASS
1,095 source-manifest checks PASS
```

after Windows long-path and CRLF normalization issues were addressed.

### Verdict

```text
SOURCE PACKAGING/PROVENANCE CPU-VERIFIED LOCALLY
LIVE CAMPAIGN PACKAGE NOT YET USED
```

---

# 17. Live calibration smoke — both products

Successful bounded smoke evidence exists for both:

## A10

```text
LIVE CALIBRATION SMOKE PASS
canonical smoke records observed
```

## A10G

```text
LIVE CALIBRATION SMOKE PASS
canonical smoke records observed
```

These smokes validate:

```text
source gate
identity gate
collector path
product routing
bounded live execution
observer-fidelity smoke
```

They do **not** establish:

```text
50 slots/cell
published canonical profile
policy consumption
performance
```

### Verdict

```text
A10 LIVE SMOKE PASS
A10G LIVE SMOKE PASS
FULL PROFILES MISSING
```

---

# 18. Measurement protocol blocker và why PULSE objective was necessary

Fixed-K steady-state recipe could not measure exact cells required by active decision:

```text
(old=0,future=0)
(old=0,future=2)
(old=0,future=3)
```

because after a q pulse, subsequent steady state becomes diagonal:

```text
(2,2)
(3,3)
```

Transition windows were excluded by the old collector contract.

A first-response latency alternative was then tested conceptually/CPU-wise, but reward benefit could land later than the first retirement.

Therefore:

```text
first-response cost horizon
!=
speculative reward horizon
```

and candidate ranking could change.

### Resolution

Freeze matched two-retirement objective:

```text
SECP-PULSE-2R-v1
```

---

# 19. SECP-PULSE-2R-v1 objective

For state `s`, pulse q, common preparation/runtime domain `G`:

```text
J_2R(s,q;G)
=
E[U1 + U2 | s, do(q,0), G]
/
E[t1 - t0 | s, do(q,0), G]
```

Exact cycle:

```text
pre-action state
→ q pulse
→ retirement #1
→ K0 continuation
→ retirement #2
```

K0 baseline uses the same two-retirement horizon.

Terminal continuation value:

```text
0
```

### External technical consistency

Official vLLM speculative-decoding documentation states that EAGLE-3 uses `method="eagle3"` and that verification workload depends on `(1+K)` tokens. This supports treating different K actions as different verification workloads, but the project’s pinned local `vLLM 0.30.0` source remains the implementation authority.

---

# 20. PULSE production migration

Migration applied P2R-01..P2R-20 across:

```text
eagle_consumer_profile.py
cost_policy.py
pulse_policy.py
budget_planner.py
decision_state.py
adaptive_scheduler.py
lineage.py
lineage_executor.py
eagle_rw_collector.py
```

Reported migration evidence:

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

### Verdict

```text
ACTIVE PULSE SOURCE/CPU PASS
```

---

# 21. Dual-product full-profile campaign contract

Campaign v1 freezes:

```text
products:
  NVIDIA A10
  NVIDIA A10G

required cells/product:
  old0_q0
  old0_q2
  old0_q3

slots/cell:
  50 predetermined slots

replacement-until-success:
  forbidden

bootstrap:
  cohort/slot unit
  5000 resamples
  95% interval
  seed 731
```

Failure slot:

```text
FAILED remains FAILED
not replaced
not counted VALID
```

Profiles are product-specific.

---

# 22. Campaign harness — current implementation status

Local campaign harness now includes:

```text
plan validation
durable state
atomic state replacement
single-writer protection
raw manifests/hashes
deterministic product/cell/slot scheduler
resume
fake executor
safe dry-run
live adapter
worker acknowledgment
profile publication/admission
source freeze/bundle handoff
post-run validator
```

Reported campaign test progression:

```text
69 PASS
→ 115 PASS
→ 121 campaign tests PASS under external-action guard
```

Existing PULSE regression:

```text
746-test baseline PASS
```

The two counts are reported separately; this audit does not manufacture a combined “distinct” total without a sealed manifest proving no overlap.

### External technical consistency

Python documents `os.replace()` as an atomic rename/replace operation when successful on the same filesystem, matching the campaign’s temp-file → replace state-write design.

---

# 23. Campaign issues found and fixed during CPU implementation

## 23.1 Shared mutable cell-definition bug

Detected by first state-machine tests.

Fixed before later campaign suites.

## 23.2 Unsupported-GPU spec conflict

Final rule:

```text
unsupported product
→ record allocation
→ zero scientific slots
→ allocation_count += 1
→ release worker
→ continue while budget remains
```

Unsupported allocation is not a scientific slot failure.

## 23.3 Replay / slot binding

Final tightening:

```text
slot bound to allocation
duplicate slot/attempt rejected
unreleased worker blocks unsafe resume
```

## 23.4 External-action test guard

Initial monkeypatch replaced `subprocess.Popen` with a function and broke Windows asyncio/PyTorch import behavior.

Fixed by preserving Popen class semantics.

## 23.5 Windows path-length

Clean checkout verification hit long-path limits.

Fixed in verification process without changing global Git config.

## 23.6 CRLF/LF manifest normalization

Source manifest comparison initially applied inconsistent normalization.

Fixed by explicit symmetric Git-LF/source-text rules while preserving raw Git object verification.

## 23.7 Export timing bug

Worker initially called source-validating export while native cycle was still in flight.

Fixed:

```text
cycle finalization / t1
→ then export
```

Focused q0/q2/q3 native CPU checks pass after fix.

Noncanonical completed export becomes failed evidence.

---

# 24. Synthetic end-to-end campaign verification

Reported final non-live checks:

```text
exact Git Bash dry-run PASS
source-bound CPU fake campaign PASS
A10 synthetic campaign completion PASS
A10G synthetic campaign completion PASS
post-run validator PASS
```

Live state remained:

```text
READY
0 allocations
0 live raw profile records
```

### Verdict

```text
CAMPAIGN ORCHESTRATION CPU/SYNTHETIC VERIFIED
REAL CAMPAIGN NOT RUN
```

---

# 25. Current Git handoff blocker

A handoff script successfully established:

```text
branch correct
tracked/index clean
behind=0
ahead=10
exact committed diff exported
```

but full-range:

```text
git diff --check origin/... HEAD
```

returned nonzero due whitespace findings in design docs.

Git documentation defines `git diff --check` as failing on whitespace errors such as trailing whitespace.

Current required attribution check:

```text
git show --check --oneline HEAD
git diff --check HEAD^ HEAD
```

If HEAD-only is clean:

```text
historical whitespace debt
→ NON-BLOCKING for current semantic handoff
```

subject to explicit review.

If HEAD-only is dirty:

```text
narrow cleanup commit
→ recheck
```

No force push.

---

# 26. R7 — Request/batch semantics update

Current campaign domain deliberately scopes:

```text
eligible_decode_rows = 1
base K = 3
old width = 0
context [0,256)
```

This is not evidence for arbitrary heterogeneous live batch behavior.

Current improvements:

- exact request incarnation binding;
- fresh cohort reconstruction;
- old width preserved separately from future q;
- deterministic continuation;
- A10/A10G state separated;
- resume cannot invent scientific slots.

### Verdict

```text
SOURCE/CPU VERIFIED IN SCOPED B1 DOMAIN
GENERAL LIVE MIXED-K EFFECTIVE BATCH >1 UNVERIFIED
```

No downgrade relative to interim; scoped correctness improved.

---

# 27. R8 — Telemetry traceability update

Current PULSE telemetry contract includes enough fields to reconstruct:

```text
DecisionState
profile cell
survival/predicted cycle reward
C_point/C_low/C_high
candidate score
K0 comparator
selected q
applied pulse
retirement #1
K0 continuation
retirement #2
cycle status/finalization
failure/censoring
```

Campaign raw evidence additionally binds:

```text
source SHA
plan hash
product identity
slot/cohort ID
raw manifest/hash
```

### Verdict

```text
SOURCE/CPU VERIFIED
LIVE CAMPAIGN TRACE NOT YET AVAILABLE
```

---

# 28. Correctness / lossless status

Interim historical canonical relation remains:

```text
Target-only != Adaptive == Static
```

This still does not support:

```text
HardwareAdaptive alone causes divergence
```

and does not close general lossless/distribution-preservation questions.

PULSE migration does not retroactively solve this separate correctness issue.

### Current status

```text
ADAPTIVE-SPECIFIC CORRECTNESS FAILURE:
NOT SUPPORTED BY RETAINED EVIDENCE

SHARED SPECULATIVE CORRECTNESS LIMITATION:
HISTORICALLY SUPPORTED

GENERAL LOSSLESS CLAIM:
NOT CLOSED
```

No downgrade.

---

# 29. Observer validity

Historical forensic logit observer remained perturbing.

That limitation still stands.

Current canonical campaign design intentionally avoids promoting perturbing instrumentation into canonical scientific evidence.

### Status

```text
OLD FORENSIC OBSERVER:
NOT NEUTRAL

CURRENT CAMPAIGN:
DESIGNED AROUND NATIVE FINALIZATION / NO EXTRA SYNC CLAIM
```

A live campaign will still need to preserve observer neutrality as measured, not merely assumed.

---

# 30. ON/OFF comparability and performance

Interim audit correctly warned that historical ON/OFF treatment differed beyond policy toggle.

Nothing in current PULSE source/CPU work authorizes a retrospective policy-isolation performance claim.

Current campaign is **profile calibration**, not performance benchmark.

### Status

```text
PROFILE COLLECTION:
READY TO RUN AFTER HANDOFF

POLICY_VALIDATION:
AFTER PROFILE PUBLICATION

PERFORMANCE:
AFTER CORRECTNESS + PROFILE + POLICY CONSUMPTION GATES
```

---

# 31. Current claim matrix

| Claim | Current status |
|---|---|
| Hardware-Adaptive/PULSE source exists | **VERIFIED** |
| Runtime wiring exists | **VERIFIED** |
| Custom scheduler/planner path historically runs on GPU | **HISTORICAL GPU VERIFIED** |
| DecisionState exact transport | **SOURCE/CPU VERIFIED** |
| Feedback/finalization association | **SOURCE/CPU VERIFIED** |
| Dynamic pulse actuation semantics | **SOURCE/CPU VERIFIED** |
| Historical adaptive K changes reached drafting | **HISTORICAL GPU VERIFIED** |
| Memory guard logic preserved | **SOURCE/TEST VERIFIED** |
| Live memory pressure recovery | **NOT CLOSED** |
| Historical fallback/probe | **HISTORICAL GPU PARTIAL** |
| Fallback exit/recovery | **NOT CLOSED** |
| Measured hardware cost can causally change active K | **CPU CAUSAL VERIFIED** |
| Full measured A10 pulse profile exists | **NO** |
| Full measured A10G pulse profile exists | **NO** |
| A10/A10G are separate products | **LIVE IDENTITY SUPPORTED + CONTRACT VERIFIED** |
| Calibration bootstrap circularity resolved | **SOURCE/CPU VERIFIED** |
| PULSE profile exact admission | **SOURCE/CPU VERIFIED** |
| PULSE reward/cost horizon coherent | **TECHNICAL DECISION FROZEN + CPU VERIFIED** |
| Dual-product campaign harness | **CPU/SYNTHETIC VERIFIED LOCALLY** |
| Campaign source bundle/handoff | **LOCALLY VERIFIED, REMOTE HANDOFF OPEN** |
| Live measured profile publication | **NOT RUN** |
| Live POLICY_VALIDATION cost→K | **NOT RUN** |
| Live mixed-K same effective batch | **NOT VERIFIED** |
| Telemetry contract reconstructable | **SOURCE/CPU VERIFIED** |
| Live campaign telemetry | **NOT YET AVAILABLE** |
| Adaptive-specific correctness failure | **NOT SUPPORTED** |
| General lossless | **NOT CLOSED** |
| Old forensic observer neutral | **NO** |
| Policy-isolated performance | **NOT EVALUATED** |
| Performance speedup | **NO CLAIM** |

---

# 32. Updated temporary delivery table

| Yêu cầu | Current evidence | Kết luận | Việc còn thiếu |
|---|---|---|---|
| 1. Wiring | source + regression + historical live | **Source/CPU verified; historical GPU verified** | Full current campaign live execution |
| 2. Feedback/history | G1-E + pulse lease/finalization/history tests | **Source/CPU verified** | Live profile/policy feedback |
| 3. Dynamic K | PULSE governor/lease/worker tests + historical live K | **Source/CPU verified; historical GPU partial** | Live PULSE measured-profile actuation |
| 4. Memory | existing safety regression + historical telemetry | **Đạt một phần** | pressure→recovery GPU closure |
| 5. Fallback/probe | historical evidence + preserved semantics | **Đạt một phần** | exit/recovery |
| 6. Hardware profile | exact PULSE admission + causal cost fixture | **R6-SOURCE PASS / R6-CPU PASS** | measured profiles + POLICY_VALIDATION |
| 7. Request/batch | DecisionState/incarnation/fresh cohorts/resume | **Source/CPU verified in scoped domain** | general live heterogeneous batch |
| 8. Telemetry | full PULSE reconstruction fields + raw manifests | **Source/CPU verified** | live trace |
| 9. Identity | worker-bound product/runtime contract | **Source/CPU + live smoke identity evidence** | exact profile campaigns |
| 10. Measurement | PULSE 2R horizon | **Frozen/CPU verified** | measured GPU intervals |
| 11. Provenance | self-contained bundle/source manifest | **Local verification pass** | actual campaign execution from final handoff source |
| 12. Campaign reproducibility | fixed slots/resume/hash/single-writer | **CPU/synthetic verified** | real campaign |

---

# 33. Current blockers — only the ones that are still current

## BLOCKER A — Git handoff not finished

```text
local commits not yet verified pushed to remote
```

Need:

```text
HEAD-only whitespace attribution
→ normal fast-forward push
→ fetch
→ remote SHA == local HEAD
```

## BLOCKER B — Canonical A10 profile missing

Need real full campaign.

## BLOCKER C — Canonical A10G profile missing

Need real full campaign.

## BLOCKER D — R6-GPU cost consumption unverified

After both profiles exist:

```text
admit exact measured profile
→ run POLICY_VALIDATION
→ reconstruct cost lookup/score/q/applied/continuation/finalization
```

## BLOCKER E — Performance remains outside current evidence

No benchmark until previous gates close.

---

# 34. Resolved blockers that must not be reopened without contrary evidence

- measured-cost causality at source/CPU;
- DecisionState transport;
- feedback finalization feasibility;
- CALIBRATION vs POLICY_VALIDATION split;
- self-contained Git bundle approach;
- A10 vs A10G separation;
- no admissible legacy consumer profile;
- both product smoke paths;
- steady-state diagonal substitution invalidity;
- first-response objective insufficiency;
- `SECP-PULSE-2R-v1` objective;
- PULSE production migration;
- unsupported-GPU allocation semantics;
- fixed-slot no-replacement campaign design;
- local synthetic campaign state/resume path.

---

# 35. Current claim ceiling

Allowed:

> FlexSDS now has a source/CPU-verified SECP-PULSE-2R hardware-cost decision path in which exact admitted pulse-cycle costs can causally change active K. A resumable dual-product A10/A10G fixed-slot campaign harness has been implemented and locally CPU/synthetic-tested. Bounded live calibration smoke has passed for both observed products.

Not yet allowed:

> FlexSDS has completed canonical measured A10 and A10G profiles.

Not yet allowed:

> Measured A10/A10G pulse costs have changed K in live serving.

Not yet allowed:

> R6-GPU passes.

Not yet allowed:

> SECP-PULSE-2R improves TPOT/throughput.

---

# 36. Current audit verdict

## Interim 20/11 verdict

```text
AUDIT IN PROGRESS — PARTIALLY VERIFIED
R6 hardware cost path not active
```

## Current verdict

```text
HARDWARE-ADAPTIVE / SECP-PULSE-2R:
SOURCE/CPU VERIFIED FOR THE ACTIVE HARDWARE-COST POLICY PATH

LIVE PRODUCT IDENTITY:
BOUNDED SMOKE VERIFIED FOR A10 AND A10G

DUAL-PRODUCT PROFILE CAMPAIGN:
HARNESS IMPLEMENTED + CPU/SYNTHETIC VERIFIED LOCALLY

CANONICAL A10/A10G FULL PROFILES:
NOT YET COLLECTED

R6-GPU:
UNVERIFIED

R6-PERFORMANCE:
UNVERIFIED
```

This is a material audit uplift relative to 20/11 without relaxing evidence standards.

---

# 37. Exactly one next action

Current WIP is handoff, not more algorithm research.

Run:

```bash
git show --check --oneline HEAD
git diff --check HEAD^ HEAD
```

If current HEAD-only diff is clean:

```text
normal push --dry-run
→ normal fast-forward push
→ fetch
→ verify remote branch SHA == local HEAD
```

Then move to the next environment and restore exact source.

Only after handoff:

```text
run real bounded dual-product profile campaign manually
```

The campaign itself remains user-owned live execution; Codex/automation should not silently run it without explicit authorization.

---

# 38. Post-handoff sequence

```text
remote source handoff verified
→ checkout exact SHA in next workspace
→ verify bundle/plan/state
→ manual real campaign
→ A10 profile publication/admission
→ A10G profile publication/admission
→ POLICY_VALIDATION
→ R6-GPU verdict
→ only then performance protocol
```

---

# 39. Tooling cross-checks

These are tooling semantics only; local source/tests remain authority for project behavior.

## Git whitespace

`git diff --check` reports conflict markers and whitespace errors; default whitespace errors include trailing whitespace, and it exits nonzero when problems are found.

Reference:
https://git-scm.com/docs/git-diff

## Python atomic state replacement

`os.replace(src,dst)` replaces the destination and, if successful on the same filesystem, uses atomic rename semantics on POSIX.

Reference:
https://docs.python.org/3/library/os.html#os.replace

## vLLM speculative workload

Official vLLM EAGLE-3 documentation states:

- EAGLE-3 uses `method="eagle3"`;
- `num_speculative_tokens` controls K;
- target verification handles `(1+K)` token work in the documented EAGLE verification path.

Reference:
https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/speculative_decoding.html

This is contextual support only; pinned local vLLM 0.30.0 source/tests remain implementation authority.

---

# 40. Handoff note cho PM

Bản audit mới thay thế interim report **cho current-state summary**, nhưng không xóa historical audit.

Điểm tiến bộ chính:

```text
20/11:
profile loaded but not causally consumed

23/11:
PULSE profile contract
→ active cost governor
→ CPU causal K change
→ matched pulse measurement objective
→ dual-product campaign machinery
```

Điểm chưa đạt vẫn được giữ:

```text
full A10 profile
full A10G profile
live policy validation
general live lossless closure
memory recovery closure
fallback exit
general mixed-batch closure
performance conclusion
```

Không merge/rebase chỉ để làm history “đẹp”.
Không force push.
Không commit bulk raw audit artifacts bằng `git add .`.

---

# 41. One-line state

```text
SECP-PULSE-2R SOURCE/CPU PASS
→ A10/A10G LIVE SMOKE PASS
→ DUAL-PRODUCT FULL-PROFILE CAMPAIGN HARNESS CPU/SYNTHETIC PASS
→ LOCAL SOURCE HANDOFF STILL NEEDS REMOTE SHA VERIFICATION
→ FULL MEASURED PROFILES NOT YET COLLECTED
→ R6-GPU / PERFORMANCE UNVERIFIED
```
