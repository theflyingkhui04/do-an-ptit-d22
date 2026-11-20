# FlexSDS v0.2.3 Hardware-Adaptive Audit — Runtime, Correctness và Scientific-Validity Blockers

- **Mã vấn đề:** `PROB-FLEXSDS-V023-HWADAPT-AUDIT-BLOCKERS`
- **Ngày tổng hợp:** 20/11/2026
- **Owner audit:** Bùi Thiên Đại / `daijestic`
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Phạm vi:** FlexSDS CLI → serving runner → speculative runtime → Hardware-Adaptive scheduler/planner/controller → verifier feedback → telemetry.
- **Primary audit baseline:** `develop@8006205a6f19790a776d67155a377def053d2736`
- **Package metadata tại primary baseline:** `0.2.3`
- **Historical release identity cần phân biệt:** Git tag `v0.2.3@450bf200498a699ae5442f14e502d0159a127f1e`, internal package metadata `0.2.2`.
- **Trạng thái:** `PARTIALLY VERIFIED — SCIENTIFIC CLOSURE BLOCKED`
- **Không benchmark hiệu năng trong problem log này.**
- **Không sửa controller/scheduler/planner để làm audit xanh.**
- **Negative result được giữ nguyên làm evidence.**

> File này ghi các **vấn đề/blocker và root-cause findings** phát hiện trong quá trình audit Hardware-Adaptive.  
> Báo cáo bàn giao theo 8 tiêu chí PM nằm riêng trong `logs/daijestic/`; problem log này không thay thế worklog/report đó.

---

## 1. Bối cảnh

Task audit yêu cầu kiểm chứng Hardware-Adaptive theo ba tầng:

```text
SOURCE
→ UNIT / INTEGRATION
→ GPU RUNTIME
```

với tám nhóm yêu cầu:

1. CLI → serving runner → scheduler → planner → controller.
2. Verification feedback → drafted/accepted → request/step → EMA/warmup.
3. Dynamic K → selected → scheduled → proposal → actual drafted.
4. Memory guard → live VRAM/KV → threshold → cooldown → recovery.
5. Fallback/probing → K=0 → probe → recovery; residual EAGLE cost.
6. Hardware profile → validation → measured cost consumption → K decision.
7. Request/batch state → per-request state → batch aggregation.
8. Telemetry → decision → schedule → draft → verify → feedback.

Audit không được suy performance trước correctness.

---

## 2. Source identity / provenance problem

### Observation

Trong lịch sử hiện có hai đối tượng thường được gọi bằng từ “v0.2.3” nhưng không phải cùng một revision:

```text
Primary audit baseline:
develop@8006205
package metadata = 0.2.3
```

và:

```text
Historical Git tag:
v0.2.3@450bf200498a699ae5442f14e502d0159a127f1e
internal package metadata = 0.2.2
```

Ngoài ra, các fix lịch sử sau không cùng ancestry:

```text
Có trong ancestry của 8006205:
94edea2  scalar adaptive budget integration
5563e16  async feedback lineage binding

Không nằm trong ancestry của 8006205:
59f18a9  acceptance floor + warm-up guard + reason counts
0d4a9fa  batch-K hysteresis + per-request K counts
```

### Impact

Không thể gọi một cách chính xác:

```text
"patched FlexSDS v0.2.3"
```

nếu chưa freeze exact SHA chứa behavior muốn kiểm thử.

Nếu lấy evidence từ một revision nhưng report dưới revision khác, traceability và reproducibility bị phá.

### Classification

```text
PROVENANCE GAP
TECHNICAL REVIEW REQUIRED
```

### Fix criterion

Trước post-audit validation phải có:

```text
SOURCE_COMMIT=<exact SHA>
EVIDENCE_COMMIT=<exact SHA storing artifacts>
```

và mọi patch mới phải được ghi là post-baseline validation, không hồi tố vào historical evidence.

---

## 3. Historical Problem A — Modal client / image-builder incompatibility

### Symptom

Một calibration attempt dừng trước measurement với:

```text
VersionError:
local Modal client only supports older image builder versions
but requested image builder = 2026.10
```

Evidence:

```text
modal-calibration-kietgor0.log
```

### Root cause

Local Modal client quá cũ so với image-builder generation được runtime yêu cầu.

### Classification

```text
ENVIRONMENT / TOOLCHAIN COMPATIBILITY
```

Không phải:

```text
GPU incompatibility
model incompatibility
Hardware-Adaptive algorithm failure
```

### Resolution

Modal client được upgrade và calibration có thể tiến tiếp.

### Status

```text
RESOLVED HISTORICAL BLOCKER
```

---

## 4. Historical Problem B — Raw Windows path truyền trực tiếp qua JSON environment

### Symptom

Một serving attempt dùng raw JSON environment:

```text
FLEXSDS_SERVING_CONFIG='... "profile_path": "C:\Users\..." ...'
```

và container fail khi import:

```text
JSONDecodeError: Invalid \escape
```

Evidence:

```text
modal-serving-5m98xkkh.log
```

### Root cause

Windows backslashes được truyền qua JSON text không có encoding/escaping layer đủ an toàn.

### Resolution direction

Serving config sau đó được đưa qua encoded transport (`FLEXSDS_SERVING_CONFIG_B64`) thay vì raw JSON path.

### Classification

```text
SERIALIZATION / CROSS-PLATFORM RUNTIME BUG
```

### Status

```text
RESOLVED HISTORICAL BLOCKER
```

Không dùng failure này để kết luận scheduler/controller sai.

---

## 5. Historical Problem C — Packaging / runtime source mismatch

### Observation

Các Modal serving attempts cho thấy image phải có đủ:

```text
src/flexsds/server
src/server
src/proposer
experiments
hardware profile
```

để custom Adaptive scheduler/executor và supporting lineage code import được.

Runtime logs sau correction ghi rõ checkout source được upload, ví dụ:

```text
D:\Document\Four-year\do_an\do-an-ptit-d22-flexsds-audit\src
D:\Document\Four-year\do_an\do-an-ptit-d22-flexsds-audit\experiments
```

### Problem

Pristine packaging path trước đó không bảo đảm toàn bộ sibling modules của Adaptive runtime có trong image.

### Impact

Server có thể fail trước inference dù source classes tồn tại trên developer machine.

### Classification

```text
RUNTIME PACKAGING CONTRACT BUG
```

### Status

```text
FIXED LOCALLY FOR AUDIT RUNTIME
```

Local packaging fix phải được tách khỏi Hardware-Adaptive algorithm semantics.

---

## 6. Verified runtime wiring after infrastructure fixes

Sau các infrastructure fixes, Adaptive ON log cho thấy vLLM nhận trực tiếp:

```text
scheduler_cls =
src.server.adaptive_scheduler.AdaptiveAsyncScheduler

distributed_executor_backend =
src.server.lineage_executor.LineageUniProcExecutor

async_scheduling = True
enforce_eager = True
```

và speculative config:

```text
method = eagle3
draft = duckhoi/qwen2.5-7b-eagle3.1-speculator
num_speculative_tokens = 4
num_speculative_tokens_per_batch_size = [[1,32,4]]
```

Evidence examples:

```text
modal-serving-sy0zx5j4.log
modal-serving-e7vk68_g.log
modal-serving-f4bubpsg.log
modal-serving-kuzm6jha.log
modal-serving-m2_r5sqq.log
```

### Interpretation

Hardware-Adaptive serving path không phải orphan code.

Tuy nhiên:

```text
runtime wiring works
!=
all Hardware-Adaptive requirements are correct
```

---

# 7. Current Major Problem 1 — Adaptive ON/OFF không policy-isolated

## 7.1 Adaptive ON

Observed Adaptive ON logs:

```text
enforce_eager=True
AdaptiveAsyncScheduler
LineageUniProcExecutor
async_scheduling=True
compilation mode = NONE
CUDAGraph mode = NONE
```

Runtime log còn ghi trực tiếp:

```text
Enforce eager set, disabling torch.compile and CUDAGraphs.
```

Examples:

```text
modal-serving-sy0zx5j4.log
modal-serving-e7vk68_g.log
modal-serving-f4bubpsg.log
modal-serving-kuzm6jha.log
modal-serving-m2_r5sqq.log
```

## 7.2 Adaptive OFF

Observed OFF/static EAGLE logs:

```text
enforce_eager=False
V2 Model Runner
compilation mode = VLLM_COMPILE
CUDAGraph mode = FULL_AND_PIECEWISE
torch.compile cache active
```

Examples:

```text
modal-serving-4vw92j8r.log
modal-serving-h_1bu7rk.log
modal-serving-i5kyg8ej.log
modal-serving-i1z5qhqb.log
modal-serving-1krek1qb.log
```

## 7.3 Root cause / interpretation

Enabling Adaptive changes more than K-policy:

```text
Adaptive policy
+
custom scheduler
+
custom lineage executor
+
eager execution
+
compile mode
+
CUDA Graph mode
+
model-runner generation / execution machinery
```

### Classification

```text
SCIENTIFIC COMPARABILITY CONFOUND
MAJOR
```

### Impact

As-shipped comparison hiện chỉ hỗ trợ claim:

> Adaptive ON treatment was faster/slower than Adaptive OFF under the measured environment.

Không hỗ trợ causal claim:

> Adaptive policy itself caused the performance difference.

### Fix criterion

Muốn claim policy-only effect phải có controlled policy-isolation treatment hoặc phải giữ claim ở mức whole-treatment.

---

# 8. Current Major Problem 2 — Hardware profile được load nhưng measured costs không drive K

## Observation

Planner path:

```text
profile path
→ validate measured profile
→ parse cost_matrix_ms
→ construct EmpiricalCostProfile
→ inject into controller kwargs
```

Nhưng active runtime controller:

```text
AdaptiveDraftController.select_budget()
```

chọn K theo:

```text
acceptance EMA / threshold rules
+
memory pressure guard
```

không query measured latency matrix trong active selection path.

### Current scalar policy

Observed source behavior:

```text
alpha < 0.65       → K=0
0.65 <= alpha < .8 → K=2
.8 <= alpha < .9   → K=3
alpha >= .9        → K=4
```

### Scientific distinction

```text
profile loaded
!=
profile used to choose K
```

### Classification

```text
PROFILE-LOADED-BUT-COST-IGNORED
MAJOR
R6 = CHƯA ĐẠT
```

### Impact

Không được claim:

> Hardware-Adaptive chooses K from measured hardware latency cost.

Allowed claim hiện tại:

> FlexSDS validates a measured hardware profile, while active Adaptive K selection is driven primarily by acceptance thresholds and memory signals.

### Fix criterion

Nếu DoD yêu cầu hardware-cost-driven selection:

```text
measured profile
→ explicit runtime cost decision
→ selected K
```

phải được trace và test end-to-end.

---

# 9. Current Problem 3 — Memory guard chưa closed-loop trên GPU

## Source behavior

Current audit finding:

```text
free VRAM < 1500 MiB
OR
KV occupancy > 90%
→ K=0
```

Cooldown:

```text
3 decisions
```

Recovery threshold:

```text
KV occupancy < 80%
```

Missing VRAM telemetry:

```text
fail-open
```

vì missing reading được thay bằng sentinel free-VRAM lớn.

## Evidence

Historical GPU logs cho thấy live GPU runtime tồn tại, nhưng retained campaign chưa chứng minh đầy đủ:

```text
normal state
→ threshold crossed
→ K restricted
→ cooldown
→ telemetry recovers
→ K restored
```

### Classification

```text
PARTIAL IMPLEMENTATION EVIDENCE
R4 = ĐẠT MỘT PHẦN
```

### Open question

Fail-open khi VRAM telemetry mất có phải intended safety contract không?

Đây là technical decision/review, không phải PM correctness source.

---

# 10. Current Problem 4 — Fallback/probing có bằng chứng nhưng recovery chưa đóng

## Source behavior

```text
low acceptance
→ K=0

periodic probe
→ K=2
```

Historical runtime evidence đã thấy:

```text
K=0 fallback
K=2 probe
```

## Gap

Chưa có complete trace:

```text
fallback
→ probe
→ good verification feedback
→ exit fallback
→ stable nonzero K
```

### Important semantic distinction

```text
K=0 != Vanilla
K=0 != target-only
K=0 != zero EAGLE treatment overhead
```

K=0 có nghĩa no draft token produced cho step tương ứng, nhưng treatment vẫn có thể giữ:

```text
draft model residency
custom scheduler
lineage executor
planner/controller
telemetry
```

### Classification

```text
R5 = ĐẠT MỘT PHẦN
```

---

# 11. Current Problem 5 — Batch/client concurrency chưa chứng minh heterogeneous runtime batching

## Source behavior

Per-request controller state tồn tại.

Planner tổng hợp:

```text
batch_k = max(k_i)
```

và scheduler có row cap/per-request trimming.

## Gap

Historical campaign có client concurrency >1 nhưng effective planning batch từng vẫn bằng 1.

Do đó không được suy:

```text
client concurrency = 2
⇒ runtime batch = 2
```

Chưa có GPU evidence kiểu:

```text
same effective batch:
request A → K=2
request B → K=4
```

rồi trace actual drafted rows tương ứng.

### Classification

```text
R7 = ĐẠT MỘT PHẦN
```

---

# 12. Current Problem 6 — Correctness / lossless chưa đóng

## Canonical retained observation

Historical same-SKU cache-off evidence:

```text
Target-only token 8 = 3015
Adaptive ON token 8 = 11174
Static OFF token 8  = 11174
```

và:

```text
Adaptive ON == Static OFF
for 64 / 64 output token positions
```

Canonical relation:

```text
Target-only != Adaptive == Static
```

## Interpretation

Evidence hiện tại không ủng hộ hypothesis:

```text
HardwareAdaptive riêng gây divergence
```

vì Static EAGLE khi Adaptive OFF vẫn match Adaptive ON.

### Classification

```text
KNOWN SHARED SPECULATIVE CORRECTNESS LIMITATION
LOSSLESS = NOT CLOSED
```

Không được claim:

```text
Adaptive is incorrect
EAGLE is universally incorrect
exact upstream root cause has been found
distribution preservation is verified
```

### Required next diagnostic

Tách:

```text
baseline repeatability
→ target-only
→ fixed speculative
→ adaptive
```

trên cùng source/runtime contract trước khi localize root cause.

---

# 13. Current Problem 7 — Forensic observer perturbation

## Observation

Historical measurement-validity check cho:

```text
uninstrumented target token 8 = 3015
instrumented target token 8   = 11174
```

Observer capture từng ghi:

```text
logit(11174) = 24.75
logit(3015)  = 24.625
margin       = 0.125
```

nhưng instrumentation tự làm đổi execution trajectory.

### Classification

```text
TRACE PERTURBATION
INSTRUMENTATION NOT NEUTRAL
MAJOR FOR FORENSIC CLAIMS
```

### Impact

Captured logits chỉ mô tả instrumented execution.

Không được ghi:

> canonical target margin = 0.125

### Fix criterion

Forensic instrumentation phải pass:

```text
instrumented output tokens
==
uninstrumented output tokens
```

trước khi dùng logits/intermediate states làm canonical evidence.

---

# 14. Telemetry status

Historical retained evidence đã có khả năng reconstruct nhiều step:

```text
plan
→ schedule
→ proposal birth
→ verification feedback
→ controller update
```

và runtime wiring xác nhận custom scheduler/executor thực sự được load.

Tuy nhiên full telemetry contract vẫn thiếu nhất quán ở một số trường như consuming verifier schedule/incarnation.

### Classification

```text
R8 = ĐẠT MỘT PHẦN
```

Telemetry đủ cho một số lineage joins, chưa đủ để gọi mọi request/step fully traceable.

---

# 15. Test evidence

Focused audit reconstruction tại primary baseline ghi nhận các nhóm non-overlapping:

```text
controller / planner / scheduler / memory:
81 passed
61 skipped
```

và:

```text
lineage / lifecycle / transitions / modal config:
122 passed
27 skipped
```

Tổng hai nhóm:

```text
203 passed
88 skipped
```

Một broader set:

```text
198 passed
27 skipped
2 failed
```

Hai failure bị environment/network model-metadata resolution chặn trước expected profile-guard assertion; hiện không đủ evidence để gọi chúng là Hardware-Adaptive controller regression.

### Interpretation

```text
unit/integration PASS
!=
GPU closed-loop PASS
```

---

# 16. Eight-requirement current status

| # | Requirement | Current verdict | Main open gap |
|---|---|---|---|
| 1 | CLI → runner → scheduler → planner → controller | **Đạt một phần** | exact source/deployment provenance |
| 2 | Verification feedback / EMA / warmup | **Đạt một phần** | full warmup + lifecycle edge cases |
| 3 | Dynamic K → actual drafting | **Đạt một phần** | K=4 + heterogeneous effective batch |
| 4 | VRAM/KV memory guard | **Đạt một phần** | GPU pressure/cooldown/recovery |
| 5 | Fallback / probing / recovery | **Đạt một phần** | fallback exit + residual cost |
| 6 | Measured hardware cost drives K | **Chưa đạt** | active policy ignores measured cost matrix |
| 7 | Per-request state / batch aggregation | **Đạt một phần** | mixed-K same runtime batch |
| 8 | Telemetry traceability | **Đạt một phần** | complete verifier lineage fields |

Correctness:

```text
LOSSLESS = NOT CLOSED
```

Performance:

```text
NOT READY FOR POLICY-ISOLATED PERFORMANCE CLAIM
```

---

# 17. Evidence map

## Modal calibration / infrastructure

```text
modal-calibration-kietgor0.log
  → Modal client/image-builder VersionError

modal-calibration-w0_y6wan.log
modal-calibration-tbn8czbm.log
  → calibration/runtime attempts, eager execution, resource behavior
```

## Adaptive ON serving

```text
modal-serving-sy0zx5j4.log
modal-serving-e7vk68_g.log
modal-serving-f4bubpsg.log
modal-serving-kuzm6jha.log
modal-serving-m2_r5sqq.log
```

Key evidence:

```text
AdaptiveAsyncScheduler
LineageUniProcExecutor
enforce_eager=True
compile NONE
CUDAGraph NONE
```

## Adaptive OFF / static EAGLE

```text
modal-serving-4vw92j8r.log
modal-serving-h_1bu7rk.log
modal-serving-i5kyg8ej.log
modal-serving-i1z5qhqb.log
modal-serving-1krek1qb.log
```

Key evidence:

```text
enforce_eager=False
V2 Model Runner
VLLM_COMPILE
CUDA Graph capture
```

## Historical serialization failure

```text
modal-serving-5m98xkkh.log
```

Key evidence:

```text
JSONDecodeError: Invalid \escape
```

---

# 18. External version-compatible interpretation

vLLM documents:

```text
enforce_eager=True
→ disables torch.compile
→ disables CUDA Graphs
```

This matches the Adaptive ON runtime logs.

vLLM Dynamic Speculative Decoding documents:

```text
num_speculative_tokens_per_batch_size
```

as the batch-size schedule used to choose speculative K, and K=0 means no draft tokens are produced for that schedule range.

These upstream semantics are used only to interpret observed runtime configuration; they do not replace local source/log evidence.

---

# 19. What this problem log does NOT prove

This file does not prove:

```text
all Hardware-Adaptive requirements pass
lossless correctness
hardware-cost-optimal K
policy-only speedup
memory recovery works end-to-end
fallback exit works end-to-end
mixed-K batch behavior works on GPU
observer logits represent canonical execution
```

It also does not rewrite historical failed runs as successes.

---

# 20. Current root-cause / blocker summary

| Finding | Type | Severity | Status |
|---|---|---:|---|
| `v0.2.3` source identity ambiguity | provenance | MAJOR | OPEN |
| Modal client/image-builder mismatch | environment | BLOCKER historically | RESOLVED |
| Windows JSON path escaping | serialization | BLOCKER historically | RESOLVED |
| Adaptive runtime packaging omission | deployment | BLOCKER historically | FIXED FOR AUDIT |
| ON/OFF execution-contract mismatch | scientific comparability | MAJOR | OPEN |
| Profile loaded but measured costs ignored | implementation/scientific | MAJOR | OPEN |
| Missing VRAM telemetry fail-open | safety/behavior contract | MAJOR | OPEN REVIEW |
| Fallback recovery not observed | runtime evidence | MAJOR | OPEN |
| Mixed-K batch not observed | runtime evidence | MAJOR | OPEN |
| Target-only != Adaptive == Static | correctness | BLOCKER for lossless claim | OPEN |
| Observer perturbs trajectory | instrumentation | MAJOR | OPEN |

---

# 21. Current claim ceiling

Allowed:

> FlexSDS primary audit baseline has a real Hardware-Adaptive serving path: custom scheduler/executor wiring, dynamic K, verification feedback and runtime telemetry have partial source/test/historical-GPU evidence. However, measured profile costs do not currently drive the active K decision, lossless correctness remains unresolved, and the shipped Adaptive ON/OFF treatments use materially different execution contracts.

Not allowed:

> Hardware-Adaptive is fully correct and hardware-cost optimal.

Not allowed:

> Hardware-Adaptive itself causes the measured speed difference.

Not allowed:

> Lossless speculative decoding is verified.

---

# 22. Exactly one next action

```text
Freeze the exact source identity intended for the next validation phase,
then review divergent fixes 59f18a9 and 0d4a9fa against 8006205
before creating any patched baseline.
```

Acceptance criteria:

```text
1. Exact SOURCE_COMMIT recorded.
2. Every integrated diff classified.
3. No historical evidence rewritten.
4. Focused regression tests rerun.
5. Only then run bounded GPU checks for remaining R1–R8 gaps.
6. Lossless validation remains separate from performance.
```

---

# 23. Relationship to other logs

Đọc cùng các historical problem/work logs đã có trong repo, đặc biệt:

```text
logs/problems/2026-11-14-flexsds-cli-modal-calibration-and-serving.md
```

Log đó ghi integration/calibration/Modal/GPU history trước đây.

File hiện tại tập trung vào:

```text
Hardware-Adaptive v0.2.3 audit
+
scientific validity blockers
+
runtime-contract differences
+
correctness/provenance gaps
```

Nó không phủ định historical A10G/L4 smoke; mỗi claim phải gắn đúng source, hardware, runtime config và evidence của run tương ứng.

---

# 24. One-line status

```text
RUNTIME WIRING EXISTS
→ PARTIAL CLOSED-LOOP EVIDENCE EXISTS
→ PROFILE COST IS NOT ACTIVE K POLICY
→ ON/OFF IS WHOLE-TREATMENT, NOT POLICY-ISOLATED
→ TARGET-ONLY != ADAPTIVE == STATIC REMAINS UNRESOLVED
→ OBSERVER IS NOT NEUTRAL
→ LOSSLESS/PERFORMANCE CLAIMS REMAIN BLOCKED
```
