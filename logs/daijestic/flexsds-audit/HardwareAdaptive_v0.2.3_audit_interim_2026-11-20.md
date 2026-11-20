# FlexSDS Hardware-Adaptive Audit — Báo cáo bàn giao tạm thời

- **Ngày tổng hợp:** 20/11/2026
- **Task:** Kiểm chứng lớp Hardware-Adaptive của FlexSDS theo 8 yêu cầu PM
- **Primary audit baseline:** `develop@8006205a6f19790a776d67155a377def053d2736`
- **Package metadata tại primary baseline:** `0.2.3`
- **Historical Git release tag cần phân biệt:** `v0.2.3@450bf200498a699ae5442f14e502d0159a127f1e`
- **Internal package metadata tại historical tag:** `0.2.2`
- **Trạng thái:** `AUDIT IN PROGRESS — PARTIALLY VERIFIED`
- **Nguyên tắc:** báo cáo hiện trạng trước; không sửa controller/scheduler/planner để làm audit pass.
- **Performance:** chưa kết luận.
- **Lossless/correctness:** chưa đóng.
- **Code changes trong lượt audit report này:** không sửa production behavior.

---

## 1. Scope và claim boundary

PM yêu cầu kiểm chứng Hardware-Adaptive theo ba tầng bằng chứng:

1. **SOURCE** — code có tồn tại và được nối vào runtime không.
2. **UNIT/INTEGRATION** — logic có được kiểm thử không.
3. **GPU RUNTIME** — behavior có thực sự xảy ra khi serving thật không.

Tám yêu cầu:

1. CLI → serving runner → scheduler → planner → controller.
2. Acceptance feedback, drafted/accepted, request attribution, EMA/warmup.
3. Dynamic K từ quyết định xuống drafting thực tế.
4. Memory/VRAM/KV guard, threshold, cooldown, recovery và missing telemetry.
5. Fallback K=0, probing, recovery và residual EAGLE cost.
6. Hardware profile và measured cost có thật sự tham gia chọn K không.
7. Per-request state và batch aggregation.
8. Telemetry có đủ để reconstruct decision → execution → feedback không.

### Important provenance distinction

Không đồng nhất hai source identity sau:

```text
Primary PM-audit baseline:
develop@8006205
package metadata = 0.2.3

Historical Git release tag:
v0.2.3@450bf200...
internal package metadata = 0.2.2
```

Do đó báo cáo này dùng `8006205` làm **primary audit baseline**, nhưng không gọi nó là exact Git tag `v0.2.3`.

---

## 2. Evidence classification

- **SOURCE VERIFIED:** đọc source tại exact baseline.
- **TEST VERIFIED:** focused unit/integration test đã chạy.
- **HISTORICAL GPU VERIFIED:** evidence GPU đã được giữ trong audit artifacts/problem logs.
- **UNKNOWN / NOT VERIFIED:** chưa đủ bằng chứng.
- **INFERENCE:** suy luận có căn cứ nhưng chưa phải runtime proof.

Không coi:

```text
dry-run = runtime evidence
smoke = benchmark
client concurrency = effective runtime batch
profile loaded = profile consumed
temperature=0 equality = stochastic proof
```

---

## 3. Executive summary

Hardware-Adaptive tại primary baseline `8006205` không phải orphan module. Source và historical runtime evidence cho thấy đường:

```text
CLI
→ serving runner
→ AdaptiveAsyncScheduler
→ AdaptiveBudgetPlanner
→ AdaptiveProposerAdapter
→ AdaptiveDraftController
→ EAGLE proposal
→ target verification
→ feedback
→ next decision
```

đã tồn tại và từng chạy thật.

Tuy nhiên chưa đủ evidence để đóng toàn bộ 8 yêu cầu.

Các finding chính:

1. **R6 — MAJOR:** measured hardware profile được load/validate/inject nhưng active `AdaptiveDraftController.select_budget()` không dùng measured cost matrix để drive K; active decision chủ yếu dùng acceptance thresholds + live memory/KV signals.
2. **Correctness:** historical canonical same-SKU cache-off evidence cho `Target-only != Adaptive == Static`, nên hiện chưa có evidence HardwareAdaptive riêng gây divergence.
3. **Observer validity:** forensic logit observer hiện tại đã được chứng minh làm đổi target trajectory; instrumented logits không được coi là canonical logits.
4. **ON/OFF comparability:** as-shipped Adaptive ON và OFF khác execution contract; performance hiện chỉ đủ cho whole-treatment claim, không policy-isolation.
5. **R4/R5/R7:** memory recovery, fallback exit và heterogeneous effective runtime batch >1 chưa được GPU-verified đầy đủ.
6. Một số fix lịch sử quan trọng nằm trên divergent branch và không nằm trong ancestry của `8006205`.

---

## 4. Provenance của các fix lịch sử

Các commit sau đã nằm trong ancestry của `8006205`:

```text
94edea2  feat(m4-02-04): integrate scalar adaptive budget with async scheduler
5563e16  feat(m4-02-04): bind async feedback to consumed draft origins and keep canonical serving linear
```

Các commit sau không nằm trong ancestry của `8006205`:

```text
59f18a9  feat(m2-02): add acceptance floor, warm-up guard and reason counts to adaptive controller
0d4a9fa  feat(m2-02): add batch-K hysteresis and per-request K counts to budget planner
```

Vì vậy hiện chưa có một `PATCHED_V023_SHA` duy nhất, reproducible, chứa toàn bộ behavior từng được gọi là “fix trước đó”.

---

## 5. Bảng bàn giao theo 8 yêu cầu

| Yêu cầu | File/dòng code | Test hoặc log minh chứng | Kết luận | Việc còn thiếu |
|---|---|---|---|---|
| **1. CLI → runner → scheduler → planner → controller** | `src/flexsds/server/local_runner.py` khoảng `130–164`; `src/server/adaptive_scheduler.py`; `src/server/budget_planner.py` khoảng `72+`; `src/proposer/adapter.py` khoảng `58+` | **SOURCE VERIFIED:** command builder nối custom scheduler/executor. **HISTORICAL GPU VERIFIED:** Adaptive ON từng load custom scheduler/executor/controller và trả completion thành công. | **Đạt một phần** | Pristine baseline từng có packaging/launcher blockers; runtime thành công dùng deployment/packaging fixes. Mỗi rerun phải ghi exact source + deployment diff. |
| **2. Acceptance feedback, drafted/accepted, EMA, warmup** | scheduler khoảng `140–179`; planner khoảng `167–199`; controller khoảng `97–106`, `266–292` | **HISTORICAL GPU VERIFIED:** 26 drafted / 3 accepted / 10 speculative steps; 10 attributed feedback records; lineage join cho request/step. EMA recomputation khớp retained evidence. | **Đạt một phần** | Full warmup completion, stale/preemption/request-reuse chưa đủ GPU evidence. Divergent fix `59f18a9` không nằm trong baseline. |
| **3. Dynamic K xuống drafting thật** | scheduler khoảng `71–125`; planner dùng batch K và row caps | **HISTORICAL GPU VERIFIED:** quan sát K `3 → 0 → 2`; retained joins khớp `selected → scheduled → proposal width → drafted → accepted`. | **Đạt một phần** | Chưa GPU-verified K=4; chưa có heterogeneous per-request K trong cùng effective batch >1. |
| **4. Memory / VRAM / KV guard** | scheduler `_free_vram_mb()` khoảng `212+`; planner KV usage; controller memory rule khoảng `323–377` | **SOURCE/TEST VERIFIED:** có threshold/cooldown/hysteresis. **HISTORICAL GPU VERIFIED:** live VRAM/KV telemetry tồn tại. | **Đạt một phần** | Chưa có end-to-end GPU trace `normal → pressure → K restriction → cooldown → recovery`. Missing VRAM hiện fail-open. |
| **5. Fallback / probing / recovery** | controller khoảng `379–451` | **HISTORICAL GPU VERIFIED:** fallback K=0 và probe K=2 đã xảy ra. | **Đạt một phần** | Chưa quan sát fallback exit/recovery. K=0 không đồng nghĩa Vanilla/target-only/zero EAGLE overhead. |
| **6. Hardware profile** | planner khoảng `238–251`; active controller `select_budget()` khoảng `415–451` | **SOURCE VERIFIED:** profile load/validate/inject. Active `AdaptiveDraftController` không query measured latency matrix để chọn K. | **Chưa đạt** | Nếu DoD yêu cầu hardware-cost-driven K thì production policy hiện chưa đạt. |
| **7. Batch và request** | request-local controller trong adapter; planner khoảng `90–126`; scheduler row trimming khoảng `96–115` | **TEST VERIFIED:** state isolation có coverage. **HISTORICAL GPU VERIFIED:** request lifecycle/cleanup quan sát được. | **Đạt một phần** | Client concurrency từng >1 nhưng effective planning batch vẫn =1; chưa có real mixed-K batch. Divergent fix `0d4a9fa` không nằm trong baseline. |
| **8. Telemetry** | planner/scheduler/lineage telemetry path | **HISTORICAL GPU VERIFIED:** planned/scheduled/proposal/feedback records có thể join ở các step đã audit. | **Đạt một phần** | Consuming verifier schedule/incarnation chưa đầy đủ ở mọi record; cần per-step trace completeness. |

---

## 6. R1 — Runtime wiring

Trace chính:

```text
CLI
→ Modal/LocalServingRunner
→ vLLM serve
→ AdaptiveAsyncScheduler
→ AdaptiveBudgetPlanner
→ AdaptiveProposerAdapter
→ AdaptiveDraftController
```

Hardware-Adaptive đã được runtime sử dụng; không chỉ tồn tại dưới dạng class/module.

Historical deployment blockers như packaging thiếu sibling modules và Windows Modal interpreter selection đã từng tồn tại, nhưng đó là **historical blockers**, không phải current proof rằng core HardwareAdaptive không chạy.

---

## 7. R2 — Feedback / EMA / warmup

Historical retained evidence:

```text
TOTAL drafted = 26
TOTAL accepted = 3
TOTAL speculative steps = 10
```

Có 10 feedback records được bind theo lineage.

Observed logical chain:

```text
plan
→ schedule
→ proposal birth
→ verifier result
→ drafted/accepted attribution
→ planner/controller update
→ EMA update
→ later K decision
```

Kết luận: verifier feedback không chỉ là simulated test behavior; có historical GPU evidence.

Gap còn mở: warmup completion đầy đủ và lifecycle edge cases.

---

## 8. R3 — Dynamic K actuation

Historical runtime đã quan sát:

```text
K=3
→ K=0
→ K=2 probe
```

và selected/scheduled/proposal/drafted lineage khớp ở retained joins.

Do đó K không chỉ xuất hiện trong planner telemetry.

Gap:

- K=4 chưa GPU-verified trong campaign hiện dùng cho audit.
- heterogeneous same-batch K chưa được chứng minh.

---

## 9. R4 — Memory / VRAM / KV

Source behavior hiện audit được:

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

KV recovery condition:

```text
KV occupancy < 80%
```

Missing VRAM telemetry hiện bị thay bằng sentinel free-memory rất lớn, nên VRAM protection **fail-open** khi reading thiếu.

Historical GPU evidence xác nhận live VRAM/KV telemetry có tồn tại, nhưng chưa exercise toàn chuỗi pressure → cooldown → recovery.

---

## 10. R5 — Fallback / probing

Policy có:

```text
low acceptance
→ fallback K=0

periodic probe
→ K=2
```

Historical GPU evidence đã có fallback và probe.

Nhưng:

```text
K=0 != Vanilla
K=0 != target-only
K=0 != zero EAGLE treatment overhead
```

EAGLE model/runtime treatment vẫn có thể giữ:

- draft model residency;
- custom scheduler;
- lineage executor;
- planner/controller;
- telemetry;
- other serving overhead.

Chưa quan sát sustained recovery thoát fallback.

---

## 11. R6 — Hardware profile

Observed source flow:

```text
FLEXSDS_PROFILE_PATH
→ measured-profile validation
→ cost_matrix_ms
→ EmpiricalCostProfile
→ controller kwargs
```

Nhưng active controller flow:

```text
effective alpha
→ memory pressure rule
→ fixed acceptance-threshold budget rule
→ K
```

Không thấy active `AdaptiveDraftController.select_budget()` dùng measured latency matrix để thực hiện cost argmax/break-even selection.

### Classification

```text
PROFILE VALUES NOT CONSUMED BY ACTIVE DECISION PATH
```

### Allowed claim

> FlexSDS loads and validates a measured hardware profile and uses live acceptance/memory/KV signals for Adaptive-K decisions.

### Prohibited claim

> Measured hardware latency costs currently caused the selected K.

Verdict: **Chưa đạt**.

---

## 12. R7 — Per-request / batch

Source có controller state theo request.

Planner tổng hợp batch K, scheduler có row caps/per-request trimming.

Test evidence support request isolation.

Nhưng không được suy:

```text
client concurrency = 2
⇒ effective runtime batch = 2
```

Historical campaign từng có client concurrency >1 nhưng effective planning batch =1.

Do đó mixed-K behavior trong cùng batch thật vẫn chưa được chứng minh.

---

## 13. R8 — Telemetry traceability

Telemetry hiện đủ mạnh để reconstruct nhiều step:

```text
decision
→ schedule
→ proposal birth
→ verification feedback
→ controller update
```

Retained artifacts có planned/scheduled/proposal/feedback lineage.

Gap: consuming verifier schedule/incarnation chưa đầy đủ ở tất cả record, nên chưa gọi telemetry contract hoàn toàn closed.

---

## 14. Correctness / lossless — cập nhật sau historical reference review

### Canonical retained same-SKU cache-off result

Historical evidence ghi:

```text
Target-only token 8 = 3015
Adaptive ON token 8 = 11174
Static OFF token 8  = 11174

Adaptive ON == Static OFF
at 64 / 64 output token positions
```

Canonical relation:

```text
Target-only != Adaptive == Static
```

### Interpretation

Evidence này làm yếu mạnh hypothesis:

```text
HardwareAdaptive riêng gây correctness divergence
```

vì khi tắt HardwareAdaptive nhưng giữ EAGLE, Static vẫn match Adaptive.

Classification hiện tại:

```text
KNOWN SHARED SPECULATIVE CORRECTNESS LIMITATION
```

Không được claim:

- Adaptive causes incorrect tokens;
- EAGLE is universally non-lossless;
- exact upstream root cause đã biết;
- distribution preservation đã verified.

Lossless vẫn:

```text
NOT CLOSED
```

---

## 15. Forensic instrumentation validity

Historical measurement-validity evidence:

```text
uninstrumented target token 8 = 3015
instrumented target token 8   = 11174
```

Instrumented observer capture từng thấy:

```text
logit(11174) = 24.75
logit(3015)  = 24.625
margin       = 0.125
```

Nhưng vì instrumentation tự làm đổi trajectory:

```text
TRACE PERTURBATION — INSTRUMENTATION NOT NEUTRAL
```

Do đó các logits/margin này chỉ mô tả **instrumented execution**.

Không được dùng `0.125` như canonical target margin.

Internal root cause của target-only vs speculative divergence vẫn:

```text
NOT LOCALIZED
```

---

## 16. ON/OFF execution contract

As-shipped OFF và ON không chỉ khác `enable_adaptive`.

### OFF

```text
static EAGLE
V2 model runner
non-eager default
torch.compile ON
CUDA Graphs ON
native scheduler/executor
```

### ON

```text
dynamic Adaptive K
AdaptiveAsyncScheduler
LineageUniProcExecutor
V1/legacy model runner
enforce_eager=True
torch.compile OFF
CUDA Graphs OFF
```

Do đó experiment hiện tại là:

```text
AS-SHIPPED WHOLE-TREATMENT
```

không phải:

```text
POLICY-ISOLATION
```

Allowed claim:

> Adaptive ON treatment was faster/slower than Adaptive OFF under the recorded environment.

Prohibited claim:

> Adaptive policy itself caused the speedup/slowdown.

Performance chưa được kết luận trong audit này.

---

## 17. Focused tests đã chạy trong current reconstruction

Exact primary baseline:

```text
8006205a6f19790a776d67155a377def053d2736
```

Focused controller/planner/scheduler/memory set:

```text
81 passed
61 skipped
```

Lineage/lifecycle/transitions/modal-config set:

```text
122 passed
27 skipped
```

Hai nhóm non-overlapping:

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

Hai failure bị chặn bởi environment/network model-metadata resolution trước expected profile-guard assertion; hiện chưa đủ bằng chứng để gọi là HardwareAdaptive regression.

---

## 18. Historical runtime blockers — resolved vs current

Các blocker từng gặp trong quá trình điều tra:

- Modal packaging thiếu Adaptive support modules;
- Windows Modal interpreter discovery;
- Modal client/image-builder version;
- hardware selector vs actual GPU identity;
- target-only prefix-cache repeatability;
- audit evidence durability;
- readiness wrapper arm identity;
- wrong runner hook;
- perturbing forensic observer.

Không được coi tất cả là current blockers.

### Current scientific gaps

1. exact intended patched-v0.2.3 provenance chưa freeze;
2. measured cost profile không drive active K;
3. lossless root cause chưa localized;
4. observer neutral chưa có;
5. memory recovery chưa GPU-closed;
6. fallback exit chưa GPU-closed;
7. mixed-K effective batch chưa GPU-closed;
8. ON/OFF chưa policy-isolated.

---

## 19. Current claim matrix

| Claim | Status |
|---|---|
| Hardware-Adaptive source tồn tại | **VERIFIED** |
| Runtime wiring tồn tại | **VERIFIED** |
| Adaptive custom scheduler chạy thật | **HISTORICAL GPU VERIFIED** |
| Real verifier feedback quay lại controller | **PARTIALLY VERIFIED** |
| Dynamic K tác động draft width | **PARTIALLY VERIFIED** |
| Memory guard nhận runtime VRAM/KV | **PARTIALLY VERIFIED** |
| Memory pressure recovery closed-loop | **NOT VERIFIED** |
| Fallback/probe chạy thật | **PARTIALLY VERIFIED** |
| Fallback exit | **NOT VERIFIED** |
| Measured profile costs drive K | **NOT IMPLEMENTED IN ACTIVE PATH / CHƯA ĐẠT** |
| Per-request isolation | **TEST VERIFIED** |
| Mixed-K same effective batch | **NOT VERIFIED** |
| Decision→feedback telemetry | **PARTIALLY VERIFIED** |
| Adaptive-specific correctness failure | **NOT SUPPORTED** |
| Shared speculative correctness limitation | **SUPPORTED BY HISTORICAL CANONICAL EVIDENCE** |
| Canonical lossless correctness | **NOT CLOSED** |
| Current forensic observer neutral | **NO** |
| Policy-isolated performance | **NOT VALID CURRENTLY** |

---

## 20. Final temporary delivery table

| Yêu cầu | File/dòng code | Test hoặc log minh chứng | Kết luận | Việc còn thiếu |
|---|---|---|---|---|
| 1. Wiring | `local_runner.py`, `adaptive_scheduler.py`, `budget_planner.py`, `adapter.py` | source + historical GPU custom scheduler startup/completion | **Đạt một phần** | provenance/deployment state phải ghi exact |
| 2. Feedback/EMA | scheduler/planner/controller feedback path | historical 10 feedback joins, 26 drafted/3 accepted | **Đạt một phần** | full warmup + stale/preemption |
| 3. Dynamic K | scheduler dynamic lookup/row caps | historical K=3/0/2 selected→drafted joins | **Đạt một phần** | K=4 + mixed batch |
| 4. Memory | runtime VRAM/KV path + controller rule | source/tests + historical live telemetry | **Đạt một phần** | trigger/cooldown/recovery GPU |
| 5. Fallback/probe | controller fallback/probing path | historical K=0 + K=2 probe | **Đạt một phần** | recovery exit + residual cost measurement |
| 6. Hardware profile | planner load/validate + controller select path | source audit | **Chưa đạt** | measured cost not consumed by active K decision |
| 7. Request/batch | request-local controller + batch max-K/row cap | tests + historical lifecycle | **Đạt một phần** | effective mixed-K batch >1 |
| 8. Telemetry | planner/scheduler/lineage | historical decision→proposal→feedback joins | **Đạt một phần** | complete verifier schedule/incarnation lineage |

---

## 21. Current status

```text
HARDWARE-ADAPTIVE AUDIT:
PARTIALLY VERIFIED

PRIMARY AUDIT BASELINE:
develop@8006205a6f19790a776d67155a377def053d2736
package metadata = 0.2.3

HISTORICAL GIT TAG:
v0.2.3@450bf200498a699ae5442f14e502d0159a127f1e
internal package metadata = 0.2.2

FOCUSED TESTS:
203 passed
88 skipped

HISTORICAL GPU:
usable for wiring / K / feedback / fallback-probe evidence

CORRECTNESS:
Target-only != Adaptive == Static
shared speculative limitation; root cause not localized

FORENSIC OBSERVER:
not neutral

LOSSLESS:
NOT CLOSED

PERFORMANCE:
NOT EVALUATED FOR SCIENTIFIC CONCLUSION
```

---

## 22. Exactly one next technical action

**Freeze the exact source identity intended by PM for the next phase, then technically review divergent fixes `59f18a9` and `0d4a9fa` against `8006205` before constructing any patched baseline.**

Không cherry-pick chỉ vì tên commit khớp finding.

Flow sau khi source freeze:

```text
source freeze
→ classify every diff
→ focused regression tests
→ before/after R1–R8 audit
→ bounded GPU checks only for open gaps
→ separate lossless validation
→ comparability review
→ performance only after correctness/claim boundary is explicit
```

---

## 23. External references

- Git worktree/switch/branch operations phải giữ source provenance rõ ràng.
- vLLM Dynamic Speculative Decoding: dynamic K cần được diễn giải theo effective runtime batch, không chỉ client concurrency.
- vLLM eager execution: `enforce_eager=True` disables `torch.compile` and CUDA Graphs.
- Historical upstream issues về speculative numerical divergence/instrumentation chỉ được dùng như precedent/methodology reference, không phải proof của exact local root cause.

---

## 24. Handoff note cho PM

Báo cáo này là **interim audit report**, không phải final correctness/performance conclusion.

Nó giữ nguyên các negative findings thay vì sửa code để làm report xanh:

- measured-cost profile chưa drive K;
- exact patched source chưa freeze;
- correctness divergence chưa localized;
- observer hiện tại không neutral;
- ON/OFF hiện không policy-isolated.

Không merge branch audit này vào `develop` chỉ để lưu log. Branch được push riêng để PM review provenance/evidence trước.
