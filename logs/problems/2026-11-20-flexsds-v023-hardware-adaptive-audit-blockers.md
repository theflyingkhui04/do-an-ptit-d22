# FlexSDS v0.2.3 Hardware-Adaptive Audit â€” Runtime, Correctness vÃ  Scientific-Validity Blockers

- **MÃ£ váº¥n Ä‘á»:** `PROB-FLEXSDS-V023-HWADAPT-AUDIT-BLOCKERS`
- **NgÃ y tá»•ng há»£p:** 20/11/2026
- **Owner audit:** BÃ¹i ThiÃªn Äáº¡i / `daijestic`
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Pháº¡m vi:** FlexSDS CLI â†’ serving runner â†’ speculative runtime â†’ Hardware-Adaptive scheduler/planner/controller â†’ verifier feedback â†’ telemetry.
- **Primary audit baseline / SOURCE_COMMIT:** `develop@8006205a6f19790a776d67155a377def053d2736`
- **Package metadata táº¡i primary baseline:** `0.2.3`
- **Evidence branch:** `audit/flexsds-v023-hardwareadaptive-20261120`
- **EVIDENCE_COMMIT:** `1eb7e59`
- **GitHub relation observed immediately after initial handoff (`1eb7e59`):** `1 commit ahead / 7 commits behind main`; historical observation, not a permanent branch-state invariant
- **Historical release identity cáº§n phÃ¢n biá»‡t:** Git tag `v0.2.3@450bf200498a699ae5442f14e502d0159a127f1e`, internal package metadata `0.2.2`.
- **Tráº¡ng thÃ¡i:** `PARTIALLY VERIFIED â€” SCIENTIFIC CLOSURE BLOCKED`
- **Branch policy:** audit evidence branch Ä‘Æ°á»£c giá»¯ riÃªng; khÃ´ng merge/rebase chá»‰ Ä‘á»ƒ xÃ³a tráº¡ng thÃ¡i `behind`.
- **KhÃ´ng benchmark hiá»‡u nÄƒng trong problem log nÃ y.**
- **KhÃ´ng sá»­a controller/scheduler/planner Ä‘á»ƒ lÃ m audit xanh.**
- **Negative result Ä‘Æ°á»£c giá»¯ nguyÃªn lÃ m evidence.**

> File nÃ y ghi cÃ¡c **váº¥n Ä‘á»/blocker vÃ  root-cause findings** phÃ¡t hiá»‡n trong quÃ¡ trÃ¬nh audit Hardware-Adaptive.  
> BÃ¡o cÃ¡o bÃ n giao theo 8 tiÃªu chÃ­ PM náº±m riÃªng trong `logs/daijestic/`; problem log nÃ y khÃ´ng thay tháº¿ worklog/report Ä‘Ã³.

---

## 1. Bá»‘i cáº£nh

Task audit yÃªu cáº§u kiá»ƒm chá»©ng Hardware-Adaptive theo ba táº§ng:

```text
SOURCE
â†’ UNIT / INTEGRATION
â†’ GPU RUNTIME
```

vá»›i tÃ¡m nhÃ³m yÃªu cáº§u:

1. CLI â†’ serving runner â†’ scheduler â†’ planner â†’ controller.
2. Verification feedback â†’ drafted/accepted â†’ request/step â†’ EMA/warmup.
3. Dynamic K â†’ selected â†’ scheduled â†’ proposal â†’ actual drafted.
4. Memory guard â†’ live VRAM/KV â†’ threshold â†’ cooldown â†’ recovery.
5. Fallback/probing â†’ K=0 â†’ probe â†’ recovery; residual EAGLE cost.
6. Hardware profile â†’ validation â†’ measured cost consumption â†’ K decision.
7. Request/batch state â†’ per-request state â†’ batch aggregation.
8. Telemetry â†’ decision â†’ schedule â†’ draft â†’ verify â†’ feedback.

Audit khÃ´ng Ä‘Æ°á»£c suy performance trÆ°á»›c correctness.

---

## 2. Source identity / provenance problem

### Observation

Trong lá»‹ch sá»­ hiá»‡n cÃ³ hai Ä‘á»‘i tÆ°á»£ng thÆ°á»ng Ä‘Æ°á»£c gá»i báº±ng tá»« â€œv0.2.3â€ nhÆ°ng khÃ´ng pháº£i cÃ¹ng má»™t revision:

```text
Primary audit baseline:
develop@8006205
package metadata = 0.2.3
```

vÃ :

```text
Historical Git tag:
v0.2.3@450bf200498a699ae5442f14e502d0159a127f1e
internal package metadata = 0.2.2
```

NgoÃ i ra, cÃ¡c fix lá»‹ch sá»­ sau khÃ´ng cÃ¹ng ancestry:

```text
CÃ³ trong ancestry cá»§a 8006205:
94edea2  scalar adaptive budget integration
5563e16  async feedback lineage binding

KhÃ´ng náº±m trong ancestry cá»§a 8006205:
59f18a9  acceptance floor + warm-up guard + reason counts
0d4a9fa  batch-K hysteresis + per-request K counts
```

### Impact

KhÃ´ng thá»ƒ gá»i má»™t cÃ¡ch chÃ­nh xÃ¡c:

```text
"patched FlexSDS v0.2.3"
```

náº¿u chÆ°a freeze exact SHA chá»©a behavior muá»‘n kiá»ƒm thá»­.

Náº¿u láº¥y evidence tá»« má»™t revision nhÆ°ng report dÆ°á»›i revision khÃ¡c, traceability vÃ  reproducibility bá»‹ phÃ¡.

### Classification

```text
PROVENANCE GAP
TECHNICAL REVIEW REQUIRED
```

### Fix criterion

TrÆ°á»›c post-audit validation pháº£i cÃ³:

```text
SOURCE_COMMIT=<exact SHA>
EVIDENCE_COMMIT=<exact SHA storing artifacts>
```

vÃ  má»i patch má»›i pháº£i Ä‘Æ°á»£c ghi lÃ  post-baseline validation, khÃ´ng há»“i tá»‘ vÃ o historical evidence.

### Current frozen identities after handoff

Sau commit/push audit:

```text
SOURCE_COMMIT=8006205a6f19790a776d67155a377def053d2736
EVIDENCE_BRANCH=audit/flexsds-v023-hardwareadaptive-20261120
EVIDENCE_COMMIT=1eb7e59
```

Äiá»u Ä‘Ã£ freeze lÃ  **current audit source/evidence identity**. Äiá»u **chÆ°a freeze** lÃ  source cho má»™t possible patched-validation phase sau khi review `59f18a9` vÃ  `0d4a9fa`.

---

## 2.1. Branch divergence at handoff

### Observation

Táº¡i thá»i Ä‘iá»ƒm handoff, GitHub hiá»ƒn thá»‹ branch:

```text
audit/flexsds-v023-hardwareadaptive-20261120
â†’ 1 commit ahead of main
â†’ 7 commits behind main
```

Audit commit Ä‘Ã£ push:

```text
1eb7e59
audit(flexsds): preserve v0.2.3 HardwareAdaptive findings, fixes, and evidence
```

Branch audit Ä‘Æ°á»£c táº¡o trá»±c tiáº¿p tá»«:

```text
8006205a6f19790a776d67155a377def053d2736
```

### Interpretation

`ahead/behind` á»Ÿ Ä‘Ã¢y lÃ  quan há»‡ commit giá»¯a branch audit vÃ  `main`:

- `1 ahead`: audit branch cÃ³ commit evidence/handoff `1eb7e59` mÃ  `main` khÃ´ng cÃ³.
- `7 behind`: `main` Ä‘Ã£ tiáº¿n thÃªm cÃ¡c commit mÃ  branch audit khÃ´ng chá»©a.
- ÄÃ¢y lÃ  há»‡ quáº£ dá»± kiáº¿n cá»§a viá»‡c freeze audit branch tá»« source baseline cÅ© hÆ¡n.
- Tráº¡ng thÃ¡i nÃ y **khÃ´ng chá»©ng minh code audit sai**, khÃ´ng chá»©ng minh push thiáº¿u, vÃ  khÃ´ng pháº£i test failure.

### Why not update the branch just to remove "behind"

Náº¿u merge hoáº·c rebase `main` vÃ o audit branch sau khi evidence Ä‘Ã£ Ä‘Æ°á»£c thu:

```text
source ancestry changes
â†’ branch no longer represents exactly the audited source lineage
â†’ historical evidence can be mis-associated with newer code
```

Do Ä‘Ã³ khÃ´ng sync branch chá»‰ Ä‘á»ƒ lÃ m GitHub háº¿t hiá»ƒn thá»‹ `7 behind`.

Náº¿u team muá»‘n validation trÃªn current `main` hoáº·c source má»›i hÆ¡n:

```text
create/freeze a NEW validation branch/SHA
â†’ rerun relevant tests/evidence
â†’ keep old evidence mapped to 8006205/1eb7e59
```

### Classification

```text
PROVENANCE / BRANCH DIVERGENCE
EXPECTED FOR FROZEN AUDIT BRANCH
NOT A CORRECTNESS FAILURE
NOT A PUSH FAILURE
```

### Status

```text
ACCEPTED HANDOFF STATE
```

### Fix criterion

KhÃ´ng cÃ³ "fix" báº¯t buá»™c cho current audit branch.

Chá»‰ cáº§n thay Ä‘á»•i ancestry khi PM/technical owner quyáº¿t Ä‘á»‹nh má»Ÿ **new validation baseline**; khi Ä‘Ã³ pháº£i táº¡o source/evidence identity má»›i thay vÃ¬ sá»­a lá»‹ch sá»­ evidence hiá»‡n táº¡i.

---

## 3. Historical Problem A â€” Modal client / image-builder incompatibility

### Symptom

Má»™t calibration attempt dá»«ng trÆ°á»›c measurement vá»›i:

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

Local Modal client quÃ¡ cÅ© so vá»›i image-builder generation Ä‘Æ°á»£c runtime yÃªu cáº§u.

### Classification

```text
ENVIRONMENT / TOOLCHAIN COMPATIBILITY
```

KhÃ´ng pháº£i:

```text
GPU incompatibility
model incompatibility
Hardware-Adaptive algorithm failure
```

### Resolution

Modal client Ä‘Æ°á»£c upgrade vÃ  calibration cÃ³ thá»ƒ tiáº¿n tiáº¿p.

### Status

```text
RESOLVED HISTORICAL BLOCKER
```

---

## 4. Historical Problem B â€” Raw Windows path truyá»n trá»±c tiáº¿p qua JSON environment

### Symptom

Má»™t serving attempt dÃ¹ng raw JSON environment:

```text
FLEXSDS_SERVING_CONFIG='... "profile_path": "C:\Users\..." ...'
```

vÃ  container fail khi import:

```text
JSONDecodeError: Invalid \escape
```

Evidence:

```text
modal-serving-5m98xkkh.log
```

### Root cause

Windows backslashes Ä‘Æ°á»£c truyá»n qua JSON text khÃ´ng cÃ³ encoding/escaping layer Ä‘á»§ an toÃ n.

### Resolution direction

Serving config sau Ä‘Ã³ Ä‘Æ°á»£c Ä‘Æ°a qua encoded transport (`FLEXSDS_SERVING_CONFIG_B64`) thay vÃ¬ raw JSON path.

### Classification

```text
SERIALIZATION / CROSS-PLATFORM RUNTIME BUG
```

### Status

```text
RESOLVED HISTORICAL BLOCKER
```

KhÃ´ng dÃ¹ng failure nÃ y Ä‘á»ƒ káº¿t luáº­n scheduler/controller sai.

---

## 5. Historical Problem C â€” Packaging / runtime source mismatch

### Observation

CÃ¡c Modal serving attempts cho tháº¥y image pháº£i cÃ³ Ä‘á»§:

```text
src/flexsds/server
src/server
src/proposer
experiments
hardware profile
```

Ä‘á»ƒ custom Adaptive scheduler/executor vÃ  supporting lineage code import Ä‘Æ°á»£c.

Runtime logs sau correction ghi rÃµ checkout source Ä‘Æ°á»£c upload, vÃ­ dá»¥:

```text
D:\Document\Four-year\do_an\do-an-ptit-d22-flexsds-audit\src
D:\Document\Four-year\do_an\do-an-ptit-d22-flexsds-audit\experiments
```

### Problem

Pristine packaging path trÆ°á»›c Ä‘Ã³ khÃ´ng báº£o Ä‘áº£m toÃ n bá»™ sibling modules cá»§a Adaptive runtime cÃ³ trong image.

### Impact

Server cÃ³ thá»ƒ fail trÆ°á»›c inference dÃ¹ source classes tá»“n táº¡i trÃªn developer machine.

### Classification

```text
RUNTIME PACKAGING CONTRACT BUG
```

### Status

```text
FIXED LOCALLY FOR AUDIT RUNTIME
```

Local packaging fix pháº£i Ä‘Æ°á»£c tÃ¡ch khá»i Hardware-Adaptive algorithm semantics.

---

## 6. Verified runtime wiring after infrastructure fixes

Sau cÃ¡c infrastructure fixes, Adaptive ON log cho tháº¥y vLLM nháº­n trá»±c tiáº¿p:

```text
scheduler_cls =
src.server.adaptive_scheduler.AdaptiveAsyncScheduler

distributed_executor_backend =
src.server.lineage_executor.LineageUniProcExecutor

async_scheduling = True
enforce_eager = True
```

vÃ  speculative config:

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

Hardware-Adaptive serving path khÃ´ng pháº£i orphan code.

Tuy nhiÃªn:

```text
runtime wiring works
!=
all Hardware-Adaptive requirements are correct
```

---

# 7. Current Major Problem 1 â€” Adaptive ON/OFF khÃ´ng policy-isolated

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

Runtime log cÃ²n ghi trá»±c tiáº¿p:

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

As-shipped comparison hiá»‡n chá»‰ há»— trá»£ claim:

> Adaptive ON treatment was faster/slower than Adaptive OFF under the measured environment.

KhÃ´ng há»— trá»£ causal claim:

> Adaptive policy itself caused the performance difference.

### Fix criterion

Muá»‘n claim policy-only effect pháº£i cÃ³ controlled policy-isolation treatment hoáº·c pháº£i giá»¯ claim á»Ÿ má»©c whole-treatment.

---

# 8. Current Major Problem 2 â€” Hardware profile Ä‘Æ°á»£c load nhÆ°ng measured costs khÃ´ng drive K

## Observation

Planner path:

```text
profile path
â†’ validate measured profile
â†’ parse cost_matrix_ms
â†’ construct EmpiricalCostProfile
â†’ inject into controller kwargs
```

NhÆ°ng active runtime controller:

```text
AdaptiveDraftController.select_budget()
```

chá»n K theo:

```text
acceptance EMA / threshold rules
+
memory pressure guard
```

khÃ´ng query measured latency matrix trong active selection path.

### Current scalar policy

Observed source behavior:

```text
alpha < 0.65       â†’ K=0
0.65 <= alpha < .8 â†’ K=2
.8 <= alpha < .9   â†’ K=3
alpha >= .9        â†’ K=4
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
R6 = CHÆ¯A Äáº T
```

### Impact

KhÃ´ng Ä‘Æ°á»£c claim:

> Hardware-Adaptive chooses K from measured hardware latency cost.

Allowed claim hiá»‡n táº¡i:

> FlexSDS validates a measured hardware profile, while active Adaptive K selection is driven primarily by acceptance thresholds and memory signals.

### Fix criterion

Náº¿u DoD yÃªu cáº§u hardware-cost-driven selection:

```text
measured profile
â†’ explicit runtime cost decision
â†’ selected K
```

pháº£i Ä‘Æ°á»£c trace vÃ  test end-to-end.

---

# 9. Current Problem 3 â€” Memory guard chÆ°a closed-loop trÃªn GPU

## Source behavior

Current audit finding:

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

Recovery threshold:

```text
KV occupancy < 80%
```

Missing VRAM telemetry:

```text
fail-open
```

vÃ¬ missing reading Ä‘Æ°á»£c thay báº±ng sentinel free-VRAM lá»›n.

## Evidence

Historical GPU logs cho tháº¥y live GPU runtime tá»“n táº¡i, nhÆ°ng retained campaign chÆ°a chá»©ng minh Ä‘áº§y Ä‘á»§:

```text
normal state
â†’ threshold crossed
â†’ K restricted
â†’ cooldown
â†’ telemetry recovers
â†’ K restored
```

### Classification

```text
PARTIAL IMPLEMENTATION EVIDENCE
R4 = Äáº T Má»˜T PHáº¦N
```

### Open question

Fail-open khi VRAM telemetry máº¥t cÃ³ pháº£i intended safety contract khÃ´ng?

ÄÃ¢y lÃ  technical decision/review, khÃ´ng pháº£i PM correctness source.

---

# 10. Current Problem 4 â€” Fallback/probing cÃ³ báº±ng chá»©ng nhÆ°ng recovery chÆ°a Ä‘Ã³ng

## Source behavior

```text
low acceptance
â†’ K=0

periodic probe
â†’ K=2
```

Historical runtime evidence Ä‘Ã£ tháº¥y:

```text
K=0 fallback
K=2 probe
```

## Gap

ChÆ°a cÃ³ complete trace:

```text
fallback
â†’ probe
â†’ good verification feedback
â†’ exit fallback
â†’ stable nonzero K
```

### Important semantic distinction

```text
K=0 != Vanilla
K=0 != target-only
K=0 != zero EAGLE treatment overhead
```

K=0 cÃ³ nghÄ©a no draft token produced cho step tÆ°Æ¡ng á»©ng, nhÆ°ng treatment váº«n cÃ³ thá»ƒ giá»¯:

```text
draft model residency
custom scheduler
lineage executor
planner/controller
telemetry
```

### Classification

```text
R5 = Äáº T Má»˜T PHáº¦N
```

---

# 11. Current Problem 5 â€” Batch/client concurrency chÆ°a chá»©ng minh heterogeneous runtime batching

## Source behavior

Per-request controller state tá»“n táº¡i.

Planner tá»•ng há»£p:

```text
batch_k = max(k_i)
```

vÃ  scheduler cÃ³ row cap/per-request trimming.

## Gap

Historical campaign cÃ³ client concurrency >1 nhÆ°ng effective planning batch tá»«ng váº«n báº±ng 1.

Do Ä‘Ã³ khÃ´ng Ä‘Æ°á»£c suy:

```text
client concurrency = 2
â‡’ runtime batch = 2
```

ChÆ°a cÃ³ GPU evidence kiá»ƒu:

```text
same effective batch:
request A â†’ K=2
request B â†’ K=4
```

rá»“i trace actual drafted rows tÆ°Æ¡ng á»©ng.

### Classification

```text
R7 = Äáº T Má»˜T PHáº¦N
```

---

# 12. Current Problem 6 â€” Correctness / lossless chÆ°a Ä‘Ã³ng

## Canonical retained observation

Historical same-SKU cache-off evidence:

```text
Target-only token 8 = 3015
Adaptive ON token 8 = 11174
Static OFF token 8  = 11174
```

vÃ :

```text
Adaptive ON == Static OFF
for 64 / 64 output token positions
```

Canonical relation:

```text
Target-only != Adaptive == Static
```

## Interpretation

Evidence hiá»‡n táº¡i khÃ´ng á»§ng há»™ hypothesis:

```text
HardwareAdaptive riÃªng gÃ¢y divergence
```

vÃ¬ Static EAGLE khi Adaptive OFF váº«n match Adaptive ON.

### Classification

```text
KNOWN SHARED SPECULATIVE CORRECTNESS LIMITATION
LOSSLESS = NOT CLOSED
```

KhÃ´ng Ä‘Æ°á»£c claim:

```text
Adaptive is incorrect
EAGLE is universally incorrect
exact upstream root cause has been found
distribution preservation is verified
```

### Required next diagnostic

TÃ¡ch:

```text
baseline repeatability
â†’ target-only
â†’ fixed speculative
â†’ adaptive
```

trÃªn cÃ¹ng source/runtime contract trÆ°á»›c khi localize root cause.

---

# 13. Current Problem 7 â€” Forensic observer perturbation

## Observation

Historical measurement-validity check cho:

```text
uninstrumented target token 8 = 3015
instrumented target token 8   = 11174
```

Observer capture tá»«ng ghi:

```text
logit(11174) = 24.75
logit(3015)  = 24.625
margin       = 0.125
```

nhÆ°ng instrumentation tá»± lÃ m Ä‘á»•i execution trajectory.

### Classification

```text
TRACE PERTURBATION
INSTRUMENTATION NOT NEUTRAL
MAJOR FOR FORENSIC CLAIMS
```

### Impact

Captured logits chá»‰ mÃ´ táº£ instrumented execution.

KhÃ´ng Ä‘Æ°á»£c ghi:

> canonical target margin = 0.125

### Fix criterion

Forensic instrumentation pháº£i pass:

```text
instrumented output tokens
==
uninstrumented output tokens
```

trÆ°á»›c khi dÃ¹ng logits/intermediate states lÃ m canonical evidence.

---

# 14. Telemetry status

Historical retained evidence Ä‘Ã£ cÃ³ kháº£ nÄƒng reconstruct nhiá»u step:

```text
plan
â†’ schedule
â†’ proposal birth
â†’ verification feedback
â†’ controller update
```

vÃ  runtime wiring xÃ¡c nháº­n custom scheduler/executor thá»±c sá»± Ä‘Æ°á»£c load.

Tuy nhiÃªn full telemetry contract váº«n thiáº¿u nháº¥t quÃ¡n á»Ÿ má»™t sá»‘ trÆ°á»ng nhÆ° consuming verifier schedule/incarnation.

### Classification

```text
R8 = Äáº T Má»˜T PHáº¦N
```

Telemetry Ä‘á»§ cho má»™t sá»‘ lineage joins, chÆ°a Ä‘á»§ Ä‘á»ƒ gá»i má»i request/step fully traceable.

---

# 15. Test evidence

Focused audit reconstruction táº¡i primary baseline ghi nháº­n cÃ¡c nhÃ³m non-overlapping:

```text
controller / planner / scheduler / memory:
81 passed
61 skipped
```

vÃ :

```text
lineage / lifecycle / transitions / modal config:
122 passed
27 skipped
```

Tá»•ng hai nhÃ³m:

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

Hai failure bá»‹ environment/network model-metadata resolution cháº·n trÆ°á»›c expected profile-guard assertion; hiá»‡n khÃ´ng Ä‘á»§ evidence Ä‘á»ƒ gá»i chÃºng lÃ  Hardware-Adaptive controller regression.

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
| 1 | CLI â†’ runner â†’ scheduler â†’ planner â†’ controller | **Äáº¡t má»™t pháº§n** | exact source/deployment provenance |
| 2 | Verification feedback / EMA / warmup | **Äáº¡t má»™t pháº§n** | full warmup + lifecycle edge cases |
| 3 | Dynamic K â†’ actual drafting | **Äáº¡t má»™t pháº§n** | K=4 + heterogeneous effective batch |
| 4 | VRAM/KV memory guard | **Äáº¡t má»™t pháº§n** | GPU pressure/cooldown/recovery |
| 5 | Fallback / probing / recovery | **Äáº¡t má»™t pháº§n** | fallback exit + residual cost |
| 6 | Measured hardware cost drives K | **ChÆ°a Ä‘áº¡t** | active policy ignores measured cost matrix |
| 7 | Per-request state / batch aggregation | **Äáº¡t má»™t pháº§n** | mixed-K same runtime batch |
| 8 | Telemetry traceability | **Äáº¡t má»™t pháº§n** | complete verifier lineage fields |

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
  â†’ Modal client/image-builder VersionError

modal-calibration-w0_y6wan.log
modal-calibration-tbn8czbm.log
  â†’ calibration/runtime attempts, eager execution, resource behavior
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
â†’ disables torch.compile
â†’ disables CUDA Graphs
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
| Current audit source/evidence identity | provenance | INFO | FROZEN: `8006205` / `1eb7e59` |
| Possible patched-validation source identity | provenance | MAJOR | OPEN |
| Audit branch relation immediately after `1eb7e59`: `1 ahead / 7 behind main` | provenance / branch divergence | INFO | HISTORICAL HANDOFF OBSERVATION |
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
Keep the current audit evidence frozen at:
SOURCE_COMMIT=8006205...
EVIDENCE_COMMIT=1eb7e59

Then technically review divergent fixes
59f18a9 + 0d4a9fa against 8006205
before deciding whether to create a NEW patched-validation baseline.
```

Acceptance criteria:

```text
1. Do not rebase/merge main into the current audit branch merely to remove "behind".
2. Review each divergent fix for correctness, compatibility and intended semantics.
3. If integration is required, create/freeze a NEW SOURCE_COMMIT.
4. Every integrated diff classified.
5. No historical evidence rewritten or reassigned to new code.
6. Focused regression tests rerun.
7. Only then run bounded GPU checks for remaining R1â€“R8 gaps.
8. Lossless validation remains separate from performance.
```

---

# 23. Relationship to other logs

Äá»c cÃ¹ng cÃ¡c historical problem/work logs Ä‘Ã£ cÃ³ trong repo, Ä‘áº·c biá»‡t:

```text
logs/problems/2026-11-14-flexsds-cli-modal-calibration-and-serving.md
```

Log Ä‘Ã³ ghi integration/calibration/Modal/GPU history trÆ°á»›c Ä‘Ã¢y.

File hiá»‡n táº¡i táº­p trung vÃ o:

```text
Hardware-Adaptive v0.2.3 audit
+
scientific validity blockers
+
runtime-contract differences
+
correctness/provenance gaps
```

NÃ³ khÃ´ng phá»§ Ä‘á»‹nh historical A10G/L4 smoke; má»—i claim pháº£i gáº¯n Ä‘Ãºng source, hardware, runtime config vÃ  evidence cá»§a run tÆ°Æ¡ng á»©ng.

---

# 24. Handoff branch state

```text
SOURCE_COMMIT    = 8006205a6f19790a776d67155a377def053d2736
EVIDENCE_BRANCH  = audit/flexsds-v023-hardwareadaptive-20261120
EVIDENCE_COMMIT  = 1eb7e59
GITHUB RELATION  = 1 ahead / 7 behind main immediately after initial handoff `1eb7e59`
MERGE STATUS     = NOT MERGED
INTERPRETATION   = HISTORICAL HANDOFF OBSERVATION; EXPECTED FROZEN-AUDIT DIVERGENCE
```

Do not merge/rebase this branch solely to make the GitHub ahead/behind counters look clean. The counters can change after later commits; provenance is anchored by the exact source/evidence SHAs.

---

# 25. One-line status

```text
RUNTIME WIRING EXISTS
â†’ PARTIAL CLOSED-LOOP EVIDENCE EXISTS
â†’ PROFILE COST IS NOT ACTIVE K POLICY
â†’ ON/OFF IS WHOLE-TREATMENT, NOT POLICY-ISOLATED
â†’ TARGET-ONLY != ADAPTIVE == STATIC REMAINS UNRESOLVED
â†’ OBSERVER IS NOT NEUTRAL
â†’ LOSSLESS/PERFORMANCE CLAIMS REMAIN BLOCKED
```

