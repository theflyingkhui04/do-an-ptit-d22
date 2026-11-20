# FlexSDS Hardware-Adaptive Audit â€” BÃ¡o cÃ¡o bÃ n giao táº¡m thá»i

- **NgÃ y tá»•ng há»£p:** 20/11/2026
- **Task:** Kiá»ƒm chá»©ng lá»›p Hardware-Adaptive cá»§a FlexSDS theo 8 yÃªu cáº§u PM
- **Primary audit baseline / SOURCE_COMMIT:** `develop@8006205a6f19790a776d67155a377def053d2736`
- **Package metadata táº¡i primary baseline:** `0.2.3`
- **Evidence branch:** `audit/flexsds-v023-hardwareadaptive-20261120`
- **EVIDENCE_COMMIT / handoff commit:** `1eb7e59`
- **GitHub handoff relation to `main`:** `1 commit ahead / 7 commits behind` táº¡i thá»i Ä‘iá»ƒm push
- **Historical Git release tag cáº§n phÃ¢n biá»‡t:** `v0.2.3@450bf200498a699ae5442f14e502d0159a127f1e`
- **Internal package metadata táº¡i historical tag:** `0.2.2`
- **Tráº¡ng thÃ¡i:** `AUDIT IN PROGRESS â€” PARTIALLY VERIFIED`
- **Branch policy:** audit branch Ä‘Æ°á»£c push riÃªng Ä‘á»ƒ review; **khÃ´ng merge/rebase chá»‰ Ä‘á»ƒ lÃ m máº¥t tráº¡ng thÃ¡i behind**.
- **NguyÃªn táº¯c:** bÃ¡o cÃ¡o hiá»‡n tráº¡ng trÆ°á»›c; khÃ´ng sá»­a controller/scheduler/planner Ä‘á»ƒ lÃ m audit pass.
- **Performance:** chÆ°a káº¿t luáº­n.
- **Lossless/correctness:** chÆ°a Ä‘Ã³ng.
- **Code changes trong lÆ°á»£t audit report nÃ y:** khÃ´ng sá»­a production behavior.

---

## 1. Scope vÃ  claim boundary

PM yÃªu cáº§u kiá»ƒm chá»©ng Hardware-Adaptive theo ba táº§ng báº±ng chá»©ng:

1. **SOURCE** â€” code cÃ³ tá»“n táº¡i vÃ  Ä‘Æ°á»£c ná»‘i vÃ o runtime khÃ´ng.
2. **UNIT/INTEGRATION** â€” logic cÃ³ Ä‘Æ°á»£c kiá»ƒm thá»­ khÃ´ng.
3. **GPU RUNTIME** â€” behavior cÃ³ thá»±c sá»± xáº£y ra khi serving tháº­t khÃ´ng.

TÃ¡m yÃªu cáº§u:

1. CLI â†’ serving runner â†’ scheduler â†’ planner â†’ controller.
2. Acceptance feedback, drafted/accepted, request attribution, EMA/warmup.
3. Dynamic K tá»« quyáº¿t Ä‘á»‹nh xuá»‘ng drafting thá»±c táº¿.
4. Memory/VRAM/KV guard, threshold, cooldown, recovery vÃ  missing telemetry.
5. Fallback K=0, probing, recovery vÃ  residual EAGLE cost.
6. Hardware profile vÃ  measured cost cÃ³ tháº­t sá»± tham gia chá»n K khÃ´ng.
7. Per-request state vÃ  batch aggregation.
8. Telemetry cÃ³ Ä‘á»§ Ä‘á»ƒ reconstruct decision â†’ execution â†’ feedback khÃ´ng.

### Important provenance distinction

KhÃ´ng Ä‘á»“ng nháº¥t hai source identity sau:

```text
Primary PM-audit baseline:
develop@8006205
package metadata = 0.2.3

Historical Git release tag:
v0.2.3@450bf200...
internal package metadata = 0.2.2
```

Do Ä‘Ã³ bÃ¡o cÃ¡o nÃ y dÃ¹ng `8006205` lÃ m **primary audit baseline**, nhÆ°ng khÃ´ng gá»i nÃ³ lÃ  exact Git tag `v0.2.3`.

### Handoff branch vÃ  branch divergence

Audit evidence Ä‘Ã£ Ä‘Æ°á»£c Ä‘Ã³ng gÃ³i vÃ o:

```text
EVIDENCE_BRANCH=audit/flexsds-v023-hardwareadaptive-20261120
EVIDENCE_COMMIT=1eb7e59
SOURCE_COMMIT=8006205a6f19790a776d67155a377def053d2736
```

Táº¡i thá»i Ä‘iá»ƒm handoff, GitHub hiá»ƒn thá»‹:

```text
audit/flexsds-v023-hardwareadaptive-20261120
â†’ 1 commit ahead of main
â†’ 7 commits behind main
```

Interpretation:

- `1 ahead` lÃ  commit audit/evidence `1eb7e59` chá»‰ cÃ³ trÃªn branch audit.
- `7 behind main` nghÄ©a lÃ  `main` cÃ³ cÃ¡c commit má»›i khÃ´ng náº±m trong ancestry cá»§a branch audit.
- ÄÃ¢y lÃ  **branch divergence cÃ³ chá»§ Ä‘Ã­ch** vÃ¬ branch audit Ä‘Æ°á»£c táº¡o tá»« frozen source baseline `8006205`, khÃ´ng pháº£i dáº¥u hiá»‡u audit commit bá»‹ thiáº¿u hoáº·c push lá»—i.
- KhÃ´ng merge/rebase `main` vÃ o branch nÃ y chá»‰ Ä‘á»ƒ lÃ m GitHub háº¿t bÃ¡o `behind`; lÃ m váº­y sáº½ thay Ä‘á»•i source ancestry sau khi evidence Ä‘Ã£ Ä‘Æ°á»£c thu tháº­p.
- Náº¿u cáº§n má»™t validation branch má»›i trÃªn source má»›i hÆ¡n, pháº£i táº¡o/freeze source identity má»›i vÃ  khÃ´ng há»“i tá»‘ evidence cÅ© sang source má»›i.

Classification:

```text
PROVENANCE / BRANCH DIVERGENCE
EXPECTED FOR FROZEN AUDIT BRANCH
NOT A CORRECTNESS FAILURE
```

---

## 2. Evidence classification

- **SOURCE VERIFIED:** Ä‘á»c source táº¡i exact baseline.
- **TEST VERIFIED:** focused unit/integration test Ä‘Ã£ cháº¡y.
- **HISTORICAL GPU VERIFIED:** evidence GPU Ä‘Ã£ Ä‘Æ°á»£c giá»¯ trong audit artifacts/problem logs.
- **UNKNOWN / NOT VERIFIED:** chÆ°a Ä‘á»§ báº±ng chá»©ng.
- **INFERENCE:** suy luáº­n cÃ³ cÄƒn cá»© nhÆ°ng chÆ°a pháº£i runtime proof.

KhÃ´ng coi:

```text
dry-run = runtime evidence
smoke = benchmark
client concurrency = effective runtime batch
profile loaded = profile consumed
temperature=0 equality = stochastic proof
```

---

## 3. Executive summary

Hardware-Adaptive táº¡i primary baseline `8006205` khÃ´ng pháº£i orphan module. Source vÃ  historical runtime evidence cho tháº¥y Ä‘Æ°á»ng:

```text
CLI
â†’ serving runner
â†’ AdaptiveAsyncScheduler
â†’ AdaptiveBudgetPlanner
â†’ AdaptiveProposerAdapter
â†’ AdaptiveDraftController
â†’ EAGLE proposal
â†’ target verification
â†’ feedback
â†’ next decision
```

Ä‘Ã£ tá»“n táº¡i vÃ  tá»«ng cháº¡y tháº­t.

Tuy nhiÃªn chÆ°a Ä‘á»§ evidence Ä‘á»ƒ Ä‘Ã³ng toÃ n bá»™ 8 yÃªu cáº§u.

CÃ¡c finding chÃ­nh:

1. **R6 â€” MAJOR:** measured hardware profile Ä‘Æ°á»£c load/validate/inject nhÆ°ng active `AdaptiveDraftController.select_budget()` khÃ´ng dÃ¹ng measured cost matrix Ä‘á»ƒ drive K; active decision chá»§ yáº¿u dÃ¹ng acceptance thresholds + live memory/KV signals.
2. **Correctness:** historical canonical same-SKU cache-off evidence cho `Target-only != Adaptive == Static`, nÃªn hiá»‡n chÆ°a cÃ³ evidence HardwareAdaptive riÃªng gÃ¢y divergence.
3. **Observer validity:** forensic logit observer hiá»‡n táº¡i Ä‘Ã£ Ä‘Æ°á»£c chá»©ng minh lÃ m Ä‘á»•i target trajectory; instrumented logits khÃ´ng Ä‘Æ°á»£c coi lÃ  canonical logits.
4. **ON/OFF comparability:** as-shipped Adaptive ON vÃ  OFF khÃ¡c execution contract; performance hiá»‡n chá»‰ Ä‘á»§ cho whole-treatment claim, khÃ´ng policy-isolation.
5. **R4/R5/R7:** memory recovery, fallback exit vÃ  heterogeneous effective runtime batch >1 chÆ°a Ä‘Æ°á»£c GPU-verified Ä‘áº§y Ä‘á»§.
6. Má»™t sá»‘ fix lá»‹ch sá»­ quan trá»ng náº±m trÃªn divergent branch vÃ  khÃ´ng náº±m trong ancestry cá»§a `8006205`.
7. Audit handoff Ä‘Ã£ Ä‘Æ°á»£c freeze á»Ÿ `EVIDENCE_COMMIT=1eb7e59`; branch hiá»‡n `1 ahead / 7 behind main`. ÄÃ¢y lÃ  provenance state cÃ³ chá»§ Ä‘Ã­ch, khÃ´ng pháº£i correctness failure.

---

## 4. Provenance cá»§a cÃ¡c fix lá»‹ch sá»­

CÃ¡c commit sau Ä‘Ã£ náº±m trong ancestry cá»§a `8006205`:

```text
94edea2  feat(m4-02-04): integrate scalar adaptive budget with async scheduler
5563e16  feat(m4-02-04): bind async feedback to consumed draft origins and keep canonical serving linear
```

CÃ¡c commit sau khÃ´ng náº±m trong ancestry cá»§a `8006205`:

```text
59f18a9  feat(m2-02): add acceptance floor, warm-up guard and reason counts to adaptive controller
0d4a9fa  feat(m2-02): add batch-K hysteresis and per-request K counts to budget planner
```

VÃ¬ váº­y hiá»‡n chÆ°a cÃ³ má»™t `PATCHED_V023_SHA` duy nháº¥t, reproducible, chá»©a toÃ n bá»™ behavior tá»«ng Ä‘Æ°á»£c gá»i lÃ  â€œfix trÆ°á»›c Ä‘Ã³â€.

---

## 5. Báº£ng bÃ n giao theo 8 yÃªu cáº§u

| YÃªu cáº§u | File/dÃ²ng code | Test hoáº·c log minh chá»©ng | Káº¿t luáº­n | Viá»‡c cÃ²n thiáº¿u |
|---|---|---|---|---|
| **1. CLI â†’ runner â†’ scheduler â†’ planner â†’ controller** | `src/flexsds/server/local_runner.py` khoáº£ng `130â€“164`; `src/server/adaptive_scheduler.py`; `src/server/budget_planner.py` khoáº£ng `72+`; `src/proposer/adapter.py` khoáº£ng `58+` | **SOURCE VERIFIED:** command builder ná»‘i custom scheduler/executor. **HISTORICAL GPU VERIFIED:** Adaptive ON tá»«ng load custom scheduler/executor/controller vÃ  tráº£ completion thÃ nh cÃ´ng. | **Äáº¡t má»™t pháº§n** | Pristine baseline tá»«ng cÃ³ packaging/launcher blockers; runtime thÃ nh cÃ´ng dÃ¹ng deployment/packaging fixes. Má»—i rerun pháº£i ghi exact source + deployment diff. |
| **2. Acceptance feedback, drafted/accepted, EMA, warmup** | scheduler khoáº£ng `140â€“179`; planner khoáº£ng `167â€“199`; controller khoáº£ng `97â€“106`, `266â€“292` | **HISTORICAL GPU VERIFIED:** 26 drafted / 3 accepted / 10 speculative steps; 10 attributed feedback records; lineage join cho request/step. EMA recomputation khá»›p retained evidence. | **Äáº¡t má»™t pháº§n** | Full warmup completion, stale/preemption/request-reuse chÆ°a Ä‘á»§ GPU evidence. Divergent fix `59f18a9` khÃ´ng náº±m trong baseline. |
| **3. Dynamic K xuá»‘ng drafting tháº­t** | scheduler khoáº£ng `71â€“125`; planner dÃ¹ng batch K vÃ  row caps | **HISTORICAL GPU VERIFIED:** quan sÃ¡t K `3 â†’ 0 â†’ 2`; retained joins khá»›p `selected â†’ scheduled â†’ proposal width â†’ drafted â†’ accepted`. | **Äáº¡t má»™t pháº§n** | ChÆ°a GPU-verified K=4; chÆ°a cÃ³ heterogeneous per-request K trong cÃ¹ng effective batch >1. |
| **4. Memory / VRAM / KV guard** | scheduler `_free_vram_mb()` khoáº£ng `212+`; planner KV usage; controller memory rule khoáº£ng `323â€“377` | **SOURCE/TEST VERIFIED:** cÃ³ threshold/cooldown/hysteresis. **HISTORICAL GPU VERIFIED:** live VRAM/KV telemetry tá»“n táº¡i. | **Äáº¡t má»™t pháº§n** | ChÆ°a cÃ³ end-to-end GPU trace `normal â†’ pressure â†’ K restriction â†’ cooldown â†’ recovery`. Missing VRAM hiá»‡n fail-open. |
| **5. Fallback / probing / recovery** | controller khoáº£ng `379â€“451` | **HISTORICAL GPU VERIFIED:** fallback K=0 vÃ  probe K=2 Ä‘Ã£ xáº£y ra. | **Äáº¡t má»™t pháº§n** | ChÆ°a quan sÃ¡t fallback exit/recovery. K=0 khÃ´ng Ä‘á»“ng nghÄ©a Vanilla/target-only/zero EAGLE overhead. |
| **6. Hardware profile** | planner khoáº£ng `238â€“251`; active controller `select_budget()` khoáº£ng `415â€“451` | **SOURCE VERIFIED:** profile load/validate/inject. Active `AdaptiveDraftController` khÃ´ng query measured latency matrix Ä‘á»ƒ chá»n K. | **ChÆ°a Ä‘áº¡t** | Náº¿u DoD yÃªu cáº§u hardware-cost-driven K thÃ¬ production policy hiá»‡n chÆ°a Ä‘áº¡t. |
| **7. Batch vÃ  request** | request-local controller trong adapter; planner khoáº£ng `90â€“126`; scheduler row trimming khoáº£ng `96â€“115` | **TEST VERIFIED:** state isolation cÃ³ coverage. **HISTORICAL GPU VERIFIED:** request lifecycle/cleanup quan sÃ¡t Ä‘Æ°á»£c. | **Äáº¡t má»™t pháº§n** | Client concurrency tá»«ng >1 nhÆ°ng effective planning batch váº«n =1; chÆ°a cÃ³ real mixed-K batch. Divergent fix `0d4a9fa` khÃ´ng náº±m trong baseline. |
| **8. Telemetry** | planner/scheduler/lineage telemetry path | **HISTORICAL GPU VERIFIED:** planned/scheduled/proposal/feedback records cÃ³ thá»ƒ join á»Ÿ cÃ¡c step Ä‘Ã£ audit. | **Äáº¡t má»™t pháº§n** | Consuming verifier schedule/incarnation chÆ°a Ä‘áº§y Ä‘á»§ á»Ÿ má»i record; cáº§n per-step trace completeness. |

---

## 6. R1 â€” Runtime wiring

Trace chÃ­nh:

```text
CLI
â†’ Modal/LocalServingRunner
â†’ vLLM serve
â†’ AdaptiveAsyncScheduler
â†’ AdaptiveBudgetPlanner
â†’ AdaptiveProposerAdapter
â†’ AdaptiveDraftController
```

Hardware-Adaptive Ä‘Ã£ Ä‘Æ°á»£c runtime sá»­ dá»¥ng; khÃ´ng chá»‰ tá»“n táº¡i dÆ°á»›i dáº¡ng class/module.

Historical deployment blockers nhÆ° packaging thiáº¿u sibling modules vÃ  Windows Modal interpreter selection Ä‘Ã£ tá»«ng tá»“n táº¡i, nhÆ°ng Ä‘Ã³ lÃ  **historical blockers**, khÃ´ng pháº£i current proof ráº±ng core HardwareAdaptive khÃ´ng cháº¡y.

---

## 7. R2 â€” Feedback / EMA / warmup

Historical retained evidence:

```text
TOTAL drafted = 26
TOTAL accepted = 3
TOTAL speculative steps = 10
```

CÃ³ 10 feedback records Ä‘Æ°á»£c bind theo lineage.

Observed logical chain:

```text
plan
â†’ schedule
â†’ proposal birth
â†’ verifier result
â†’ drafted/accepted attribution
â†’ planner/controller update
â†’ EMA update
â†’ later K decision
```

Káº¿t luáº­n: verifier feedback khÃ´ng chá»‰ lÃ  simulated test behavior; cÃ³ historical GPU evidence.

Gap cÃ²n má»Ÿ: warmup completion Ä‘áº§y Ä‘á»§ vÃ  lifecycle edge cases.

---

## 8. R3 â€” Dynamic K actuation

Historical runtime Ä‘Ã£ quan sÃ¡t:

```text
K=3
â†’ K=0
â†’ K=2 probe
```

vÃ  selected/scheduled/proposal/drafted lineage khá»›p á»Ÿ retained joins.

Do Ä‘Ã³ K khÃ´ng chá»‰ xuáº¥t hiá»‡n trong planner telemetry.

Gap:

- K=4 chÆ°a GPU-verified trong campaign hiá»‡n dÃ¹ng cho audit.
- heterogeneous same-batch K chÆ°a Ä‘Æ°á»£c chá»©ng minh.

---

## 9. R4 â€” Memory / VRAM / KV

Source behavior hiá»‡n audit Ä‘Æ°á»£c:

```text
free VRAM < 1500 MiB
OR
KV occupancy > 90%
â†’ K=0
```

Cooldown:

```text
3 decisions
```

KV recovery condition:

```text
KV occupancy < 80%
```

Missing VRAM telemetry hiá»‡n bá»‹ thay báº±ng sentinel free-memory ráº¥t lá»›n, nÃªn VRAM protection **fail-open** khi reading thiáº¿u.

Historical GPU evidence xÃ¡c nháº­n live VRAM/KV telemetry cÃ³ tá»“n táº¡i, nhÆ°ng chÆ°a exercise toÃ n chuá»—i pressure â†’ cooldown â†’ recovery.

---

## 10. R5 â€” Fallback / probing

Policy cÃ³:

```text
low acceptance
â†’ fallback K=0

periodic probe
â†’ K=2
```

Historical GPU evidence Ä‘Ã£ cÃ³ fallback vÃ  probe.

NhÆ°ng:

```text
K=0 != Vanilla
K=0 != target-only
K=0 != zero EAGLE treatment overhead
```

EAGLE model/runtime treatment váº«n cÃ³ thá»ƒ giá»¯:

- draft model residency;
- custom scheduler;
- lineage executor;
- planner/controller;
- telemetry;
- other serving overhead.

ChÆ°a quan sÃ¡t sustained recovery thoÃ¡t fallback.

---

## 11. R6 â€” Hardware profile

Observed source flow:

```text
FLEXSDS_PROFILE_PATH
â†’ measured-profile validation
â†’ cost_matrix_ms
â†’ EmpiricalCostProfile
â†’ controller kwargs
```

NhÆ°ng active controller flow:

```text
effective alpha
â†’ memory pressure rule
â†’ fixed acceptance-threshold budget rule
â†’ K
```

KhÃ´ng tháº¥y active `AdaptiveDraftController.select_budget()` dÃ¹ng measured latency matrix Ä‘á»ƒ thá»±c hiá»‡n cost argmax/break-even selection.

### Classification

```text
PROFILE VALUES NOT CONSUMED BY ACTIVE DECISION PATH
```

### Allowed claim

> FlexSDS loads and validates a measured hardware profile and uses live acceptance/memory/KV signals for Adaptive-K decisions.

### Prohibited claim

> Measured hardware latency costs currently caused the selected K.

Verdict: **ChÆ°a Ä‘áº¡t**.

---

## 12. R7 â€” Per-request / batch

Source cÃ³ controller state theo request.

Planner tá»•ng há»£p batch K, scheduler cÃ³ row caps/per-request trimming.

Test evidence support request isolation.

NhÆ°ng khÃ´ng Ä‘Æ°á»£c suy:

```text
client concurrency = 2
â‡’ effective runtime batch = 2
```

Historical campaign tá»«ng cÃ³ client concurrency >1 nhÆ°ng effective planning batch =1.

Do Ä‘Ã³ mixed-K behavior trong cÃ¹ng batch tháº­t váº«n chÆ°a Ä‘Æ°á»£c chá»©ng minh.

---

## 13. R8 â€” Telemetry traceability

Telemetry hiá»‡n Ä‘á»§ máº¡nh Ä‘á»ƒ reconstruct nhiá»u step:

```text
decision
â†’ schedule
â†’ proposal birth
â†’ verification feedback
â†’ controller update
```

Retained artifacts cÃ³ planned/scheduled/proposal/feedback lineage.

Gap: consuming verifier schedule/incarnation chÆ°a Ä‘áº§y Ä‘á»§ á»Ÿ táº¥t cáº£ record, nÃªn chÆ°a gá»i telemetry contract hoÃ n toÃ n closed.

---

## 14. Correctness / lossless â€” cáº­p nháº­t sau historical reference review

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

Evidence nÃ y lÃ m yáº¿u máº¡nh hypothesis:

```text
HardwareAdaptive riÃªng gÃ¢y correctness divergence
```

vÃ¬ khi táº¯t HardwareAdaptive nhÆ°ng giá»¯ EAGLE, Static váº«n match Adaptive.

Classification hiá»‡n táº¡i:

```text
KNOWN SHARED SPECULATIVE CORRECTNESS LIMITATION
```

KhÃ´ng Ä‘Æ°á»£c claim:

- Adaptive causes incorrect tokens;
- EAGLE is universally non-lossless;
- exact upstream root cause Ä‘Ã£ biáº¿t;
- distribution preservation Ä‘Ã£ verified.

Lossless váº«n:

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

Instrumented observer capture tá»«ng tháº¥y:

```text
logit(11174) = 24.75
logit(3015)  = 24.625
margin       = 0.125
```

NhÆ°ng vÃ¬ instrumentation tá»± lÃ m Ä‘á»•i trajectory:

```text
TRACE PERTURBATION â€” INSTRUMENTATION NOT NEUTRAL
```

Do Ä‘Ã³ cÃ¡c logits/margin nÃ y chá»‰ mÃ´ táº£ **instrumented execution**.

KhÃ´ng Ä‘Æ°á»£c dÃ¹ng `0.125` nhÆ° canonical target margin.

Internal root cause cá»§a target-only vs speculative divergence váº«n:

```text
NOT LOCALIZED
```

---

## 16. ON/OFF execution contract

As-shipped OFF vÃ  ON khÃ´ng chá»‰ khÃ¡c `enable_adaptive`.

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

Do Ä‘Ã³ experiment hiá»‡n táº¡i lÃ :

```text
AS-SHIPPED WHOLE-TREATMENT
```

khÃ´ng pháº£i:

```text
POLICY-ISOLATION
```

Allowed claim:

> Adaptive ON treatment was faster/slower than Adaptive OFF under the recorded environment.

Prohibited claim:

> Adaptive policy itself caused the speedup/slowdown.

Performance chÆ°a Ä‘Æ°á»£c káº¿t luáº­n trong audit nÃ y.

---

## 17. Focused tests Ä‘Ã£ cháº¡y trong current reconstruction

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

Hai nhÃ³m non-overlapping:

```text
203 passed
88 skipped
```

Má»™t broader set:

```text
198 passed
27 skipped
2 failed
```

Hai failure bá»‹ cháº·n bá»Ÿi environment/network model-metadata resolution trÆ°á»›c expected profile-guard assertion; hiá»‡n chÆ°a Ä‘á»§ báº±ng chá»©ng Ä‘á»ƒ gá»i lÃ  HardwareAdaptive regression.

---

## 18. Historical runtime blockers â€” resolved vs current

CÃ¡c blocker tá»«ng gáº·p trong quÃ¡ trÃ¬nh Ä‘iá»u tra:

- Modal packaging thiáº¿u Adaptive support modules;
- Windows Modal interpreter discovery;
- Modal client/image-builder version;
- hardware selector vs actual GPU identity;
- target-only prefix-cache repeatability;
- audit evidence durability;
- readiness wrapper arm identity;
- wrong runner hook;
- perturbing forensic observer.

KhÃ´ng Ä‘Æ°á»£c coi táº¥t cáº£ lÃ  current blockers.

### Current scientific gaps

1. current audit baseline/evidence Ä‘Ã£ freeze (`SOURCE_COMMIT=8006205...`, `EVIDENCE_COMMIT=1eb7e59`), nhÆ°ng **next patched-validation source** váº«n chÆ°a Ä‘Æ°á»£c chá»‘t;
2. measured cost profile khÃ´ng drive active K;
3. lossless root cause chÆ°a localized;
4. observer neutral chÆ°a cÃ³;
5. memory recovery chÆ°a GPU-closed;
6. fallback exit chÆ°a GPU-closed;
7. mixed-K effective batch chÆ°a GPU-closed;
8. ON/OFF chÆ°a policy-isolated.

---

## 19. Current claim matrix

| Claim | Status |
|---|---|
| Hardware-Adaptive source tá»“n táº¡i | **VERIFIED** |
| Runtime wiring tá»“n táº¡i | **VERIFIED** |
| Adaptive custom scheduler cháº¡y tháº­t | **HISTORICAL GPU VERIFIED** |
| Real verifier feedback quay láº¡i controller | **PARTIALLY VERIFIED** |
| Dynamic K tÃ¡c Ä‘á»™ng draft width | **PARTIALLY VERIFIED** |
| Memory guard nháº­n runtime VRAM/KV | **PARTIALLY VERIFIED** |
| Memory pressure recovery closed-loop | **NOT VERIFIED** |
| Fallback/probe cháº¡y tháº­t | **PARTIALLY VERIFIED** |
| Fallback exit | **NOT VERIFIED** |
| Measured profile costs drive K | **NOT IMPLEMENTED IN ACTIVE PATH / CHÆ¯A Äáº T** |
| Per-request isolation | **TEST VERIFIED** |
| Mixed-K same effective batch | **NOT VERIFIED** |
| Decisionâ†’feedback telemetry | **PARTIALLY VERIFIED** |
| Adaptive-specific correctness failure | **NOT SUPPORTED** |
| Shared speculative correctness limitation | **SUPPORTED BY HISTORICAL CANONICAL EVIDENCE** |
| Canonical lossless correctness | **NOT CLOSED** |
| Current forensic observer neutral | **NO** |
| Policy-isolated performance | **NOT VALID CURRENTLY** |

---

## 20. Final temporary delivery table

| YÃªu cáº§u | File/dÃ²ng code | Test hoáº·c log minh chá»©ng | Káº¿t luáº­n | Viá»‡c cÃ²n thiáº¿u |
|---|---|---|---|---|
| 1. Wiring | `local_runner.py`, `adaptive_scheduler.py`, `budget_planner.py`, `adapter.py` | source + historical GPU custom scheduler startup/completion | **Äáº¡t má»™t pháº§n** | provenance/deployment state pháº£i ghi exact |
| 2. Feedback/EMA | scheduler/planner/controller feedback path | historical 10 feedback joins, 26 drafted/3 accepted | **Äáº¡t má»™t pháº§n** | full warmup + stale/preemption |
| 3. Dynamic K | scheduler dynamic lookup/row caps | historical K=3/0/2 selectedâ†’drafted joins | **Äáº¡t má»™t pháº§n** | K=4 + mixed batch |
| 4. Memory | runtime VRAM/KV path + controller rule | source/tests + historical live telemetry | **Äáº¡t má»™t pháº§n** | trigger/cooldown/recovery GPU |
| 5. Fallback/probe | controller fallback/probing path | historical K=0 + K=2 probe | **Äáº¡t má»™t pháº§n** | recovery exit + residual cost measurement |
| 6. Hardware profile | planner load/validate + controller select path | source audit | **ChÆ°a Ä‘áº¡t** | measured cost not consumed by active K decision |
| 7. Request/batch | request-local controller + batch max-K/row cap | tests + historical lifecycle | **Äáº¡t má»™t pháº§n** | effective mixed-K batch >1 |
| 8. Telemetry | planner/scheduler/lineage | historical decisionâ†’proposalâ†’feedback joins | **Äáº¡t má»™t pháº§n** | complete verifier schedule/incarnation lineage |

---

## 21. Current status

```text
HARDWARE-ADAPTIVE AUDIT:
PARTIALLY VERIFIED

PRIMARY AUDIT BASELINE / SOURCE_COMMIT:
develop@8006205a6f19790a776d67155a377def053d2736
package metadata = 0.2.3

EVIDENCE BRANCH:
audit/flexsds-v023-hardwareadaptive-20261120

EVIDENCE_COMMIT:
1eb7e59

GITHUB RELATION OBSERVED AFTER INITIAL HANDOFF `1eb7e59`:
1 commit ahead / 7 commits behind
(historical observation at initial handoff; expected frozen-audit divergence; not a correctness failure; may change after later commits)

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

**Giá»¯ nguyÃªn current audit baseline/evidence (`SOURCE_COMMIT=8006205...`, `EVIDENCE_COMMIT=1eb7e59`) vÃ  technical-review hai divergent fixes `59f18a9` + `0d4a9fa` against `8006205` Ä‘á»ƒ quyáº¿t Ä‘á»‹nh cÃ³ táº¡o má»™t patched-validation baseline má»›i hay khÃ´ng.**

KhÃ´ng cherry-pick chá»‰ vÃ¬ tÃªn commit khá»›p finding. KhÃ´ng rebase/merge `main` vÃ o audit branch hiá»‡n táº¡i chá»‰ Ä‘á»ƒ lÃ m máº¥t tráº¡ng thÃ¡i `7 behind`.

Flow:

```text
current audit evidence frozen
â†’ review 59f18a9 + 0d4a9fa against 8006205
â†’ decide intended patched behavior
â†’ if needed: create NEW patched-validation branch/SHA
â†’ classify every integrated diff
â†’ focused regression tests
â†’ before/after R1â€“R8 audit
â†’ bounded GPU checks only for open gaps
â†’ separate lossless validation
â†’ comparability review
â†’ performance only after correctness/claim boundary is explicit
```

---

## 23. External references

- GitHub Compare/Branches: ahead/behind lÃ  quan há»‡ commit giá»¯a base vÃ  compare branch; branch audit divergent khÃ´ng tá»± Ä‘á»™ng Ä‘á»“ng nghÄ©a vá»›i lá»—i.
- Git worktree/switch/branch operations pháº£i giá»¯ source provenance rÃµ rÃ ng.
- vLLM Dynamic Speculative Decoding: dynamic K cáº§n Ä‘Æ°á»£c diá»…n giáº£i theo effective runtime batch, khÃ´ng chá»‰ client concurrency.
- vLLM eager execution: `enforce_eager=True` disables `torch.compile` and CUDA Graphs.
- Historical upstream issues vá» speculative numerical divergence/instrumentation chá»‰ Ä‘Æ°á»£c dÃ¹ng nhÆ° precedent/methodology reference, khÃ´ng pháº£i proof cá»§a exact local root cause.

---

## 24. Handoff note cho PM

BÃ¡o cÃ¡o nÃ y lÃ  **interim audit report**, khÃ´ng pháº£i final correctness/performance conclusion.

NÃ³ giá»¯ nguyÃªn cÃ¡c negative findings thay vÃ¬ sá»­a code Ä‘á»ƒ lÃ m report xanh:

- measured-cost profile chÆ°a drive K;
- exact patched source chÆ°a freeze;
- correctness divergence chÆ°a localized;
- observer hiá»‡n táº¡i khÃ´ng neutral;
- ON/OFF hiá»‡n khÃ´ng policy-isolated.

KhÃ´ng merge branch audit nÃ y vÃ o `develop` chá»‰ Ä‘á»ƒ lÆ°u log. Branch Ä‘Æ°á»£c push riÃªng Ä‘á»ƒ PM review provenance/evidence trÆ°á»›c.

Handoff state hiá»‡n táº¡i:

```text
branch  = audit/flexsds-v023-hardwareadaptive-20261120
commit  = 1eb7e59
source  = 8006205a6f19790a776d67155a377def053d2736
GitHub  = 1 ahead / 7 behind main immediately after initial handoff `1eb7e59`
merge   = NOT PERFORMED
```

Tráº¡ng thÃ¡i `7 behind main` Ä‘Æ°á»£c giá»¯ nguyÃªn cÃ³ chá»§ Ä‘Ã­ch Ä‘á»ƒ khÃ´ng lÃ m thay Ä‘á»•i ancestry cá»§a source Ä‘Ã£ audit.

