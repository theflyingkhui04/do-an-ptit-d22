# Worklog Kỹ Thuật: [M4-02-04] Adaptive Scheduler Telemetry & Scalar Runtime Verification trên NVIDIA L4

## 0. Thông Tin Công Việc

- **Task:** `[M4-02-04] Scheduler Telemetry / Adaptive Scalar Budget Integration`
- **Người thực hiện:** Bùi Thiên Đại (`daijestic`)
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Branch:** `feat/m4-02-04-scheduler-telemetry-v02`
- **Frozen implementation / Colab source SHA:** `16528241468e2624ccfe07a69a67fcc123bfe4a3`
- **Final L4 evidence commit:** `6906985eeb0b53bfce312847dbcf104d6254b1ed`
- **Canonical live-smoke evidence root:**  
  `experiments/runs/m4_02_04/20261111T132628Z-scalar-observer-l4-b45a0afc`
- **Backend:** Google Colab
- **GPU:** NVIDIA L4
- **Driver:** `580.82.07`
- **Python canonical execution:** `3.11.18`
- **PyTorch:** `2.13.0+cu130`
- **CUDA runtime:** `13.0`
- **vLLM:** `0.30.0`
- **Target:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Target revision:** `b25037543e9394b818fdfca67ab2a00ecc7dd641`
- **Draft:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Draft revision:** `7ae557604adf67be50417f59c2c2f167def9a775`
- **Trạng thái kỹ thuật hiện tại:** `READY FOR CLOSURE REVIEW`
- **Production source sau live smoke:** `UNCHANGED`
- **Profitability benchmark:** `NOT EVALUATED`
- **Full profiling grid:** `NOT RUN`
- **Arbitrary-tree execution:** `NOT CLAIMED`

---

# 1. Executive Summary

M4-02-04 tập trung vào việc đưa scalar adaptive speculative budget vào serving path thực của vLLM `0.30.0`, kiểm chứng scheduler integration, telemetry và khả năng quan sát runtime đủ chính xác để chuẩn bị cho bước hardware-aware cost profiling.

Phần implementation trước live GPU smoke đã xác minh:

- adaptive scalar `K`;
- per-request budget `K_i`;
- `AdaptiveAsyncScheduler` integration;
- shared batch `K = max(K_i)`;
- per-request trimming / isolation;
- planned / scheduled / measured telemetry;
- request lifecycle cleanup;
- runtime scalar K propagation.

Arbitrary tree topology không được claim là đã execute.

Final live-smoke trên NVIDIA L4 kiểm tra hai treatment nhỏ:

```text
K=0 / B=1
K=2 / B=1
```

Kết quả cuối:

```text
SOURCE GATE          PASS
PREFLIGHT            PASS
K=0 / B=1            PASS
K=2 / B=1            PASS
OFFLINE VALIDATOR    PASS
OBSERVER VERDICT     LIVE OBSERVER VERIFIED
EVIDENCE PUSH        PASS
```

Evidence commit:

```text
6906985eeb0b53bfce312847dbcf104d6254b1ed
```

Đây là **instrumentation smoke**, không phải performance campaign.

Không kết luận:

```text
K=2 nhanh hơn K=0
hardware-aware policy đã optimal
profitability đã chứng minh
full grid đã hoàn tất
arbitrary tree đã execute
```

---

# 2. Governance Và Scope

## 2.1. PM decision

Trong quá trình M4-02-04, arbitrary-tree runtime audit cho thấy pinned vLLM path hiện tại không cung cấp đầy đủ semantics để execute arbitrary external tree topology.

PM chọn phương án:

```text
Option B
```

Tức:

```text
đóng M4-02-04 ở scalar adaptive-K path đã verify
+
tách arbitrary-tree runtime thành follow-up
```

Không chuyển sang SGLang cho canonical path.

Do đó scope closure của M4-02-04 là:

- adaptive scalar K;
- per-request budget;
- AsyncScheduler integration;
- trimming/isolation;
- telemetry;
- live pinned-vLLM smoke;
- instrumentation readiness.

Không bắt buộc:

- arbitrary tree execution;
- SGLang migration;
- full profiling grid;
- profitability benchmark.

---

# 3. Implementation Baseline

Frozen implementation trước live evidence:

```text
CODE_SHA =
16528241468e2624ccfe07a69a67fcc123bfe4a3
```

Các implementation checkpoint chính trước đó:

```text
c52cd0ca2a4044fb78376e9100f3d718c38302f0
feat(m4-02-04): extend planner metadata and telemetry contract

94edea218aa2b48b4035dea4a87b7a7a4d6ceee6
feat(m4-02-04): integrate scalar adaptive budget with async scheduler

a6cd6ad5cee5374738f890c270ada4ca01f0946b
docs(m4-02-04): record scalar scheduler evidence and open boundaries

3dd7735e53b6b46b3fe745cb21df87e61f36901d
evidence(m4-02-04): record pinned-vllm scalar runtime smoke

b4c13637149b14b3ebab20412b044639b7e4c40f
docs(m4-02-04): document async feedback lineage design

d517206ccc7d9febfd0f6d8c6a824f57a2774657
test(m4-02-04): strengthen adaptive-k integration coverage

1ea823662a9bd1289912cb499c8703873c6fa7ee
feat(m4-02-04): add scalar cost observer and context reconstruction

16528241468e2624ccfe07a69a67fcc123bfe4a3
chore(m4-02-04): add reproducible Colab L4 observer smoke workflow
```

---

# 4. Scalar Runtime Path Được Verify

Serving path:

```text
verification output
→ acceptance feedback
→ controller
→ planner per-request K_i
→ shared batch K=max(K_i)
→ AdaptiveAsyncScheduler
→ dynamic speculative lookup / scheduler output
→ proposer
→ verifier
→ emitted output
→ telemetry
```

Các property đã kiểm tra trước L4 live observer:

- scalar budget truyền tới scheduler;
- per-request budget giữ riêng;
- batch-level K không ghi đè semantic của K_i;
- trimming request placeholder hoạt động;
- request isolation;
- lifecycle cleanup;
- telemetry phân biệt planned / scheduled / measured.

Claim boundary:

```text
scalar K verified
≠
arbitrary topology verified
```

---

# 5. Hardware-Aware Threshold Review

Canonical serving controller tại thời điểm review vẫn dùng fixed acceptance bands:

```text
alpha < 0.65       -> K=0
0.65 <= alpha<0.80 -> K=2
0.80 <= alpha<0.90 -> K=3
alpha >= 0.90      -> K=4
```

Technical review xác định:

- policy trên chưa derive threshold từ measured hardware cost;
- existing scalar `HardwareCostModel` chưa nối vào canonical serving decision;
- batch size / hardware cost profile chưa quyết định break-even criterion;
- acceptance statistic A/D không tự động đồng nghĩa theoretical independent per-token probability;
- shared proposer cost phải được tính ở batch scope.

Kết luận:

```text
hardware-aware policy:
NOT YET IMPLEMENTED AS A MEASURED COST-BASED SERVING RULE
```

Do đó bước tiếp theo trước policy implementation là đo scalar runtime cost bằng observer có contract rõ ràng.

---

# 6. Measurement Contract

Retirement-window contract được chọn để tránh cộng các latency overlap không hợp lệ.

Boundary:

```text
start:
immediately after successful scheduler.update_from_output()

end:
after N subsequent completed decode batches
finish update_from_output()
```

Metric:

```text
elapsed_ms / completed_batch
```

Contract phải ghi:

- shared K;
- per-request K_i;
- actual worker batch;
- request identity;
- context length;
- query length;
- padding;
- proposer rows;
- lifecycle;
- useful emitted token count.

Không đồng nhất:

```text
client concurrency
=
runtime worker batch
```

Không dùng:

```text
sum(overlapping execution latencies)
```

làm cycle cost.

---

# 7. Corrected Context Reconstruction

Pinned vLLM `0.30.0` async speculative path có correction semantics mà CPU-side `num_computed_tokens` có thể chứa optimistic placeholders.

Observer reconstruction dùng recurrence tương ứng với pinned runtime:

```text
if correction participates:

C_t[i]
=
C_(t-1)[P_i]
+
V_(t-1)[P_i]

otherwise:

C_t[i]
=
H_t[i]
```

Trong đó:

- `C_t`: corrected pre-execution context;
- `P_i`: previous-row mapping;
- `V`: prior raw valid sampled count;
- `H_t`: current host count.

Position:

```text
position[i,q] = C_t[i] + q
```

Sequence length:

```text
seq_len[i] = C_t[i] + Q_t[i]
```

Đã kiểm tra các case:

- all accepted;
- partial rejection;
- first-token rejection;
- K=0;
- bonus token;
- heterogeneous budget;
- invalid slots;
- row reorder;
- multiple in-flight execution.

Preemption/resume vẫn fail-closed nếu nằm ngoài supported observer contract.

---

# 8. Colab L4 Environment

## 8.1. Host runtime issue

Colab hiện tại không cung cấp Python 3.11 trực tiếp trong host runtime sử dụng cho notebook.

Host được quan sát:

```text
Python 3.13.x
```

Do canonical smoke cần Python 3.11, notebook dùng managed CPython qua `uv`.

Final execution:

```text
Python = 3.11.18
```

---

## 8.2. Frozen constraints incompatibility

File:

```text
experiments/scalar_smoke_constraints.txt
```

không replay được nguyên trạng trên canonical Python 3.11.

Hai incompatibility được phát hiện trong quá trình setup:

```text
networkx==3.7
```

không compatible với Python 3.11;

và:

```text
numpy==2.4.6
```

xung đột dependency với runtime dependency graph của vLLM / mistral-common trên Python 3.11.

Do đó final environment contract dùng:

```text
known-good top-level pins
+
resolved transitive dependencies
```

thay vì claim:

```text
exact full frozen constraint snapshot replayed
```

Final resolved versions được lưu trong evidence.

Claim bắt buộc:

```text
constraint snapshot replay:
SKIPPED_INVALID_FOR_PYTHON_3_11
```

Không được viết rằng committed full constraint set đã reproduce thành công.

---

# 9. JIT Runtime Dependencies

Trong lần smoke đầu, FlashInfer JIT fail vì không tìm thấy executable:

```text
ninja
```

Root cause:

```text
minimal venv dependency set
không đảm bảo executable ninja nằm trên PATH
```

Final runtime:

```text
ninja = 1.13.2
```

Venv bin được đưa vào `PATH`.

Ngoài ra preflight xác minh:

- `c++` / `g++`;
- `nvcc`;
- CUDA runtime;
- PyTorch CUDA;
- L4 / sm89 identity.

Sau fix, FlashInfer JIT warmup và CUDA graph capture hoàn tất.

---

# 10. AutoAWQ Identity Correction

Initial smoke harness giả định:

```text
cfg.model_config.quantization == "awq_marlin"
```

và:

```text
quant_config.get_name() == "awq_marlin"
```

Live vLLM `0.30.0` cho thấy assumption này không đúng.

Canonical runtime identity:

```text
quantization config identity = auto_awq
```

Marlin là selected execution backend/kernel, không phải config identity.

Runtime-only overlay được dùng để sửa gate:

```text
awq_marlin
→
auto_awq
```

và tách riêng:

```text
target dtype
draft dtype
quantization config
```

Production source không bị sửa.

Repair metadata được giữ trong evidence.

---

# 11. Live Observer Install-Boundary Bug

Sau environment và runtime-object gate PASS, K=0 từng fail:

```text
live observer transport failure
```

Observer evidence cho thấy failure:

```text
stage = before_retirement
error = 'NoneType' object is not subscriptable
```

Root cause:

Async EngineCore có thể đã chứa scheduler outputs trong `batch_queue` trước thời điểm observer được install.

Khi output pre-install đó retirement:

```text
self.facts.get(id(schedule))
→ None
```

nhưng original hook vẫn dereference:

```text
fact["materialized_output_id"]
```

Runtime-only observer repair:

1. snapshot `schedule` identities đã nằm trong queue lúc install;
2. chỉ bỏ qua các retirement đã được chứng minh là pre-install carry-over;
3. mọi unmatched retirement khác vẫn được ghi failure;
4. không globally suppress missing-fact errors.

Production implementation không thay đổi.

Repair artifact:

```text
observer-install-boundary-repair.json
```

---

# 12. Final K=0 Live Smoke

Treatment:

```text
K=0
B=1
```

Kết quả:

```text
PASS
```

Observer thu được đủ eligible decode retirements.

K=0 vẫn giữ draft-side synchronization behavior của pinned path; không diễn giải K=0 thành target-only vanilla inference.

Đây là scalar reference treatment trong cùng target+draft engine.

---

# 13. Final K=2 Live Smoke

Treatment:

```text
K=2
B=1
```

Kết quả:

```text
PASS
```

Smoke xác minh:

- requested scalar K;
- planned K;
- scheduled K;
- proposer K;
- assigned proposer K;
- draft width;
- worker row;
- output pairing;
- corrected context;
- emitted output.

Không dùng kết quả để claim speedup.

---

# 14. Offline Validation

Final:

```text
OFFLINE VALIDATOR = PASS
OBSERVER VERDICT  = LIVE OBSERVER VERIFIED
```

Validator kiểm tra:

- artifact hashes;
- source identity;
- pinned vLLM source hashes;
- environment identity;
- K0/K2 cell status;
- observer failure/overflow;
- output equality structural check;
- fixture equality;
- S/future/O pairing;
- worker/proposer rows;
- planned/scheduled/proposer K;
- placeholder width;
- verification shape;
- context reconstruction;
- selected retirement window;
- cross-cell trajectory/reference matching.

---

# 15. Historical Failed Attempts

Không xóa các attempt fail.

Canonical evidence root giữ:

```text
previous-smoke-attempts/
```

Các failure bao gồm:

1. environment/constraint setup failures;
2. missing pip in venv;
3. missing Ninja;
4. incorrect quantization identity gate;
5. observer install-boundary bookkeeping failure;
6. old offline validator assumption failure.

Các attempt fail được giữ làm troubleshooting provenance.

Không aggregate chúng thành final PASS result.

---

# 16. Validator Reclassification

K0 và K2 đã thực sự collection PASS, nhưng original offline validator fail sau collection vì assumption cũ:

```text
L4 quantization identity = awq_marlin
```

Do đó original `run()` đã ghi:

```text
manifest.status = FAIL
failure.json
```

Original failure được preserve tại:

```text
validation-repair-audit/
```

Sau khi chứng minh:

```text
K0 PASS
K2 PASS
raw records = 2
raw statuses = ok
failure originated from post-collection validate(root)
```

run được reclassify:

```text
FAIL
→ COLLECTED
```

Artifacts được reseal và validator chạy lại.

Final:

```text
OFFLINE VALIDATOR PASS
LIVE OBSERVER VERIFIED
```

Không rerun GPU.

---

# 17. Evidence Integrity Và Push

Final evidence root:

```text
experiments/runs/m4_02_04/
20261111T132628Z-scalar-observer-l4-b45a0afc/
```

Trước push:

```text
SOURCE GATE             PASS
BYTE-INTEGRITY          PASS
CREDENTIAL SCAN         PASS
STRUCTURED GIT --CHECK  PASS
```

Raw `.log` / `.txt` được giữ byte-for-byte.

Không strip trailing whitespace từ raw logs chỉ để thỏa Git whitespace checker.

Structured files được kiểm riêng:

```text
.json
.jsonl
.md
.csv
```

Final commit:

```text
6906985eeb0b53bfce312847dbcf104d6254b1ed
```

Commit identity:

```text
BUI DUC DAI <daibui1208@gmail.com>
```

Remote verification:

```text
local evidence commit
==
origin/feat/m4-02-04-scheduler-telemetry-v02

PASS
```

---

# 18. Claim Matrix

| Claim | Status |
|---|---|
| Adaptive scalar K implemented | VERIFIED |
| Per-request K_i implemented | VERIFIED |
| AsyncScheduler integration | VERIFIED |
| Trimming / isolation | VERIFIED |
| Planned/scheduled/measured telemetry | VERIFIED |
| Pinned-vLLM live scalar path | VERIFIED |
| L4 live observer K=0 | VERIFIED |
| L4 live observer K=2 | VERIFIED |
| Context reconstruction | VERIFIED within guarded scalar scope |
| Arbitrary tree runtime | NOT VERIFIED / OUT OF CURRENT OPTION-B SCOPE |
| Hardware-derived serving threshold | NOT IMPLEMENTED AS FINAL POLICY |
| K profitability | NOT EVALUATED |
| Full scalar cost profile | NOT RUN |
| Consumer-GPU performance conclusion | NOT AVAILABLE FROM THIS SMOKE |

---

# 19. Scientific Interpretation

## FACT

```text
K=0/B1 PASS
K=2/B1 PASS
offline validation PASS
live observer VERIFIED
```

## FACT

Observer có thể thu thập runtime shape/context/retirement evidence trên pinned vLLM `0.30.0` và NVIDIA L4 cho tiny scalar smoke.

## FACT

Production source tại `CODE_SHA` không bị thay đổi bởi các runtime-only Colab repairs.

## OBSERVATION

Một số assumptions trong initial reproducibility package không đúng với environment hiện tại:

- full constraint snapshot không replay được trên Python 3.11;
- AutoAWQ config identity khác assumption;
- async pre-install queue cần xử lý observer install boundary.

## INFERENCE

Instrumentation path hiện đủ để tiếp tục technical review cho scalar cost measurement.

## UNKNOWN

Tiny smoke không trả lời:

```text
K nào tối ưu?
break-even acceptance là bao nhiêu?
K=2 có nhanh hơn K=0 không?
batch B=2/4/... thay cost như thế nào?
hardware-aware controller cuối cùng nên chọn K nào?
```

Các câu hỏi này cần controlled profiling campaign riêng.

---

# 20. Những Claim Không Được Viết

Không được viết:

```text
M4-02-04 chứng minh speculative decoding nhanh hơn
```

Không được viết:

```text
K=2 tốt hơn K=0 trên L4
```

Không được viết:

```text
hardware-aware threshold đã hoàn thành
```

Không được viết:

```text
arbitrary tree đã execute trên vLLM
```

Không được viết:

```text
full profiling grid đã hoàn thành
```

Không được viết:

```text
scalar_smoke_constraints.txt được reproduce nguyên trạng
```

Được viết:

```text
Live scalar observer instrumentation for K=0/B1 and K=2/B1
was verified on NVIDIA L4 with pinned vLLM 0.30.0, with
runtime deviations and validator repairs explicitly preserved
in the evidence.
```

---

# 21. Reproducibility Artifacts

Canonical evidence:

```text
experiments/runs/m4_02_04/
└── 20261111T132628Z-scalar-observer-l4-b45a0afc/
```

Các nhóm artifact quan trọng:

```text
setup/
smoke/
previous-smoke-attempts/
validation-repair-audit/

runtime-overlay.json
observer-install-boundary-repair.json
validator-quantization-repair.json
offline-validator.log
orchestrator.log
```

Trong `smoke/`:

```text
manifest.json
environment.json
source_manifest.json
SHA256SUMS.json
raw.jsonl
validation.json

k0/
├── cell.json
├── environment.json
├── fixture.json
├── observer.json
└── stdout-stderr.log

k2/
├── cell.json
├── environment.json
├── fixture.json
├── observer.json
└── stdout-stderr.log
```

---

# 22. Definition of Done / Closure Review

## Đã đạt

- [x] Scalar adaptive K implementation.
- [x] Per-request budget.
- [x] AsyncScheduler integration.
- [x] Trimming/isolation integration coverage.
- [x] Planned/scheduled/measured telemetry.
- [x] Source gate.
- [x] Pinned vLLM source verification.
- [x] L4 hardware gate.
- [x] K=0/B1 live smoke.
- [x] K=2/B1 live smoke.
- [x] Offline validation.
- [x] Context reconstruction evidence.
- [x] Evidence integrity.
- [x] Credential scan.
- [x] Evidence pushed to remote.
- [x] Failed attempts preserved.
- [x] No profitability/tree overclaim.

## Không thuộc completion claim hiện tại

- [ ] Full scalar profiling grid.
- [ ] Hardware-aware profitability policy.
- [ ] Arbitrary-tree execution.
- [ ] Performance speedup conclusion.

---

# 23. Final Technical Status

```text
CODE_SHA:
16528241468e2624ccfe07a69a67fcc123bfe4a3

FINAL EVIDENCE COMMIT:
6906985eeb0b53bfce312847dbcf104d6254b1ed

LIVE L4:
K=0/B1 PASS
K=2/B1 PASS

OFFLINE VALIDATOR:
PASS

OBSERVER:
LIVE OBSERVER VERIFIED

PROFITABILITY:
NOT EVALUATED

FULL GRID:
NOT RUN

TREE EXECUTION:
NOT CLAIMED
```

Kết luận:

> **M4-02-04 hiện ở trạng thái `READY FOR CLOSURE REVIEW`.**
>
> Scalar serving integration và live observer path đã có evidence thực trên NVIDIA L4.
> Tiny smoke không phải benchmark hiệu năng và không cung cấp profitability conclusion.
> Arbitrary-tree runtime vẫn nằm ngoài claim của Option B hiện tại.

---

# 24. Next Action

Không chạy thêm GPU smoke chỉ để tăng số lượng evidence.

Bước tiếp theo:

```text
PM / technical closure review
```

Sau closure, nếu scope tiếp theo yêu cầu hardware-aware policy:

```text
review live observer evidence
→ freeze scalar cost profiling protocol
→ pilot variance
→ quyết repeats
→ chạy controlled cost matrix
→ aggregate measured C_K(B, context, ...)
→ thiết kế hardware-aware K selection
→ rerun serving validation
```

Không implement policy trước khi measured cost contract được duyệt.

---

# 25. Item 8 async lineage + canonical linear cleanup (2026-11-11)

## Baseline and read-only audit

- Refreshed `origin/develop`: `53b813a0b6ce8fef530c4286fbd8bf4da5b4a149` (PR #36 merge).
- Original worktree branch: `feat/m4-02-04-scheduler-telemetry-v02`, HEAD
  `b6ef6fa9aa9b6e41eb792eb5968703f751a72576`; no tracked modifications.
- Existing untracked evidence directories were preserved untouched:
  `20261111-checkpoint-preparation/`, `20261111T115100Z-colab-observer-package/`,
  `20261111T120000Z-colab-observer-package/` under `experiments/runs/m4_02_04/`.
- Git initially rejected sandbox ownership; a command-scoped safe.directory
  exception was used. Fetch and local ref/index writes required elevated access
  because worktree Git metadata lives in the parent checkout. No global Git
  configuration was changed.
- Audited source/tests/worklog matched the original checkout's tracked bytes;
  `git diff HEAD origin/develop -- src tests ...` was empty. Repository-wide
  lineage/tree searches found 291 matching tracked paths. No AGENTS.md found.
- **Pre-edit classification: PARTIAL.** Scheduler/planner already stamp plan
  and schedule sequences, but `make_spec_decoding_stats` forwarded only request
  ID/counts; planner source IDs were null. The delayed-feedback test protected
  null attribution while still mutating request state. It did not join a draft
  producer to the execution actually consuming its rows.
- New branch `feat/m4-02-04-async-lineage-v02` was created from the exact audited
  remote SHA. Immediately before edits, branch HEAD and origin/develop both
  equaled `53b813a0b6ce8fef530c4286fbd8bf4da5b4a149`.
- Local `develop` remained `9bc0f8d86e7a1f620a4f8cdaee79230b3bc41b14`.
  No develop edit/commit/merge, reset, rebase, clean, force update, PR or Trello write.

## Pinned authority and implementation

**IMPLEMENTED / source-contract VERIFIED**, implementation commit
`5563e16eebacf35e0b0ced9866919b8656a87d42`.

Authority: vLLM **0.30.0**, V1, UniProc, one worker, async padded `draft_model`.
Inspected the committed captured scheduler at
`experiments/runs/m2_03/boundary279_20261018T132244Z_15f4dae9/vllm-originals/v1/core/sched/scheduler.py`,
and the local installed-source capture at
`experiments/runs/m4_02_04/20261110-async-lineage-design/installed/vllm/`.
All 15 captured files in `experiments/scalar_vllm_source_manifest.json` matched
their preserved SHA-256 values. No installed vLLM package is available in the
local Python environment. Captures remain unchanged evidence, not runtime code.

The same SchedulerOutput proposes NEW drafts while verifying OLD drafts. The
implementation therefore does not label old feedback with that output's new K:

1. `LineageLedger.register` stores detached frozen producer plan/schedule/K
   records and request incarnations for each exact output object.
2. `LineageUniProcExecutor` gates version, source bytes, single-worker config,
   async padded draft_model, and exact V1 GPUModelRunner class. It observes the
   existing `_copy_draft_token_ids_to_cpu` call after real proposal, distinguishes
   synthetic zeros, and stamps every buffer generation even on tensor reuse.
3. Immediately after inline execute_model, before sample_tokens overwrites old
   fields, it freezes actual `prev_positions` row/prefix bindings to the prior
   generation. Buffer witness, previous width, row identity and prefix bounds
   must agree. CPU-only/no-buffer/unobserved paths remain unattributed.
4. `update_from_output(S, O)` scopes feedback to S's frozen consumed origins;
   source IDs identify the producer of those drafts, not S's new proposal.
   Model output row, request object/incarnation, counts and stale state are checked.
5. Each binding is consumed once. Missing/corrupt/stale/duplicate provenance
   emits a warning plus `verification_feedback_skipped`; it never calls the
   adaptive updater. Upstream statistics are forwarded unchanged. Invalid
   tokens retain upstream D=N-I accounting and accepted count A unchanged.
6. Retirement finally blocks release output records even without callbacks;
   request completion purges its origins without affecting other rows. Future
   errors/cancellation and shutdown clear the ledger. Config identity isolates
   engine instances; shutdown deregisters the process-local rendezvous.

No worker/result/SchedulerOutput shared schema, upstream fixture, algorithm,
tree estimator, CUDA tensor values or GPU kernel was changed. No added tensor
readback or CUDA synchronization. Legacy direct planner simulations can still
supply synchronous counts; production scheduler feedback requires provenance.

## Tree preservation and canonical serving

- **VERIFIED preservation:** new local `future/tree-adaptive-v02` at
  `53b813a0b6ce8fef530c4286fbd8bf4da5b4a149`, created before cleanup. This contains
  all audited tree source/history. No existing future-work branch was overwritten.
  It includes the tree controller tip `a2510f2` through develop's merge history.
  Preservation was **NOT PUSHED**.
- **VERIFIED canonical path:** local launcher / Modal adaptive lane ->
  AdaptiveAsyncScheduler + LineageUniProcExecutor -> AdaptiveBudgetPlanner ->
  AdaptiveProposerAdapter -> AdaptiveDraftController -> per-request scalar K_i,
  shared scalar K=max(K_i) -> upstream linear draft_model proposer -> exact
  output retirement -> provenance-validated scalar acceptance update.
- Removed unused eager TreeAdaptive imports from adapter and planner's catalog
  reconstruction. Package tree exports are lazy for explicit future users.
  `selected_topology` remains present (null for canonical scalar decisions).
  Explicit injected future-tree decision names/counts retain unknown/null
  structure; parent arrays are not reconstructed by canonical telemetry.
  `tree_execution_verified` remains false.
- Dormant tree controllers/catalog/tests remain because explicit future-work
  imports and existing consumers can continue to inspect them. Canonical
  serving neither selects these controllers nor builds parent arrays, ancestor
  masks, tree position IDs or accepted-tree paths. Keeping dormant source and
  lazy compatibility exports avoids deleting another owner's future work.
- Modal serving now mounts committed experiment source (excluding run/result/
  notebook evidence) for the source manifest and existing scheduler observer
  imports. Launcher/config tests verify custom executor wiring. CLI/README
  TreeAdaptive marketing text remains historical/non-evidence; CLI inspected
  here is a configuration wizard, not this scheduler's execution path.

## Test matrix and execution ledger

Environment for all tests: Windows PowerShell, CPython 3.11.0, local CPU pytest;
torch CPU tensors where noted. No installed vLLM engine, remote requests,
profitability benchmark or GPU smoke. Test-local Git remotes in existing
scalar Git tests are temporary fixtures, not a project branch push.

| Requirement | Result and evidence in test_adaptive_scheduler_glue.py |
| --- | --- |
| T1 single source/feedback | VERIFIED: feedback_reaches_controller_and_super; exact producer IDs in delayed-feedback test |
| T2 multiple in flight | VERIFIED: multiple_inflight_changing_k_exact_producer |
| T3 normal retirement | VERIFIED: FIFO parameter, interleaved source sequences 1/2 |
| T4 delayed identity stress | VERIFIED at L1/L2: reverse parameter is synthetic ledger stress, NOT true out-of-order pinned runtime (EngineCore FIFO) |
| T5 interleaved requests | VERIFIED: interleaved_requests_and_reordered_rows; a/b counts and sources isolated |
| T6 K=2 -> K=4 | VERIFIED: old verifier retains source K=2 after K=4 plan/proposal exists |
| T7 K=0 | VERIFIED: positive old feedback during new K=0; 30 empty-feedback retirements leave no output orphan |
| T8 invalid/trimmed | VERIFIED: captured upstream grammar trimming/padding, D=N-I, exact prior source |
| T9 completion/reuse | VERIFIED: completed request cannot update reused ID/controller; other request unaffected; shutdown clears |
| T10 missing/corrupt | VERIFIED: missing/count/incarnation/request/stale/accepted corruption; duplicate callbacks/retirement; no adaptive mutation, upstream stats preserved, observable warning/event |

Additional source-contract tests execute unchanged captured `_compute_prev_positions`
and `_prepare_input_ids` on CPU tensors (reorder/compaction/prefix identity), and
unchanged UniProc `collective_rpc` / AsyncOutputFuture (inline worker invocation,
deferred host materialization). Production executor wrapper tests simulate the
worker and cover deferred grammar sampling, same-buffer generations, synthetic
zeros, future error cleanup, registry shutdown and unsupported version/worker gates.
These are NOT real-runtime integration verification. Four capture-dependent tests
can skip on another machine without the optional local installed-source cache;
they ran in the final local checks.

Canonical cleanup C1-C8: VERIFIED at source/unit level. A fresh-process import
blocker rejects tree module/catalog imports while scalar {2,4,0}, batch K=4 and
null topology/false tree-execution telemetry remain intact. Existing scalar,
observer, scheduler, adapter and dormant-tree functional regressions passed.
Preservation ref/SHA recorded above. Real serving remains IMPLEMENTED BUT UNVERIFIED.

Commands below use these exact groups:

```text
TARGET = python -m pytest -q tests/test_adaptive_scheduler_glue.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py
GLUE = python -m pytest -q tests/test_adaptive_scheduler_glue.py
BROAD = python -m pytest -q tests/test_adaptive_scheduler_glue.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_budget_controller.py tests/test_budget_fallback_recovery.py tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_cost_model_l4.py tests/test_cost_model_solver.py tests/test_modal_runner_config.py tests/test_modal_smoke.py tests/test_scalar_context.py tests/test_scalar_colab_smoke.py tests/test_scalar_colab_git.py tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py
FUNCTIONAL = BROAD with tests/test_modal_smoke.py omitted and -k "not test_decision_latency_benchmark and not test_cache_alpha_star_performance"
FINAL = TARGET plus tests/test_modal_runner_config.py
```

| Run | Command | Exit | Passed | Failed | Skipped | Deselected | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | TARGET | 1 | 61 | 5 | 0 | 0 | Legacy direct callback tests correctly fail closed without execution context |
| 2 | TARGET | 0 | 66 | 0 | 0 | 0 | Legacy tests now use retirement contexts |
| 3 | TARGET | 0 | 79 | 0 | 0 | 0 | Added T1-T10 stress coverage |
| 4 | GLUE | 0 | 41 | 0 | 2 | 0 | Wrong cache path caused temporary skips; corrected |
| 5 | TARGET | 1 | 82 | 1 | 0 | 0 | CPU fixture sampled-token dtype int64 vs upstream int32; corrected |
| 6 | TARGET | 0 | 83 | 0 | 0 | 0 | Installed-source checks and executor wrapper stable before tree cleanup |
| 7 | TARGET | 0 | 84 | 0 | 0 | 0 | Canonical import isolation added |
| 8 | BROAD | 1 | 385 | 8 | 0 | 0 | Six inherited workload-byte failures + two unchanged timing assertions; details below |
| 9 | FUNCTIONAL | 0 | 376 | 0 | 0 | 2 | Timing checks explicitly excluded; no test thresholds changed |
| 10 | TARGET | 0 | 85 | 0 | 0 | 0 | Added actual captured inline-RPC contract |
| 11 | TARGET | 0 | 85 | 0 | 0 | 0 | Explicit same-buffer birth stamp assertion |
| 12 | FINAL | 0 | 136 | 0 | 0 | 0 | Final production packaging and backend configuration included |

Broad failures retained:

- `test_tree_adaptive_controller.py::test_decision_latency_benchmark`: measured
  0.064274 ms against <0.05 ms; tree implementation unchanged.
- `test_cost_model_solver.py::test_cache_alpha_star_performance`: measured
  0.010800 ms against <0.01 ms; cost model unchanged. No performance pass claimed.
- Six `test_modal_smoke.py` cases: frozen W1 SHA mismatch. Baseline Git blob
  SHA-256 is `0de2387c4e6a8c39523bf869788a55ee7487c6a32318a949ce22e1f8f7b17360`
  (the manifest value), whereas existing CRLF checkout bytes hash to
  `19808e79c743443bf9c8b989b797becc23c9f0483282b93e2c967f416b493386`.
  Read-only comparison proved `worktree.replace(CRLF, LF) == baseline_blob`;
  Git core.autocrlf is true. Frozen workloads/manifests were not modified.

## Changed files, claim ceiling and remaining work

- `src/server/lineage.py`: immutable origins, identity ledger, request lifecycle.
- `src/server/lineage_executor.py`: pinned UniProc observation/source/config gates.
- `src/server/adaptive_scheduler.py`: exact retirement scope, validation, fail-safe.
- `src/server/budget_planner.py`: source-aware telemetry; no tree catalog import.
- `src/proposer/adapter.py`, `src/proposer/__init__.py`: remove eager tree dependency,
  retain explicit lazy compatibility exports.
- `scripts/launch_lane3_adaptive.sh`, `experiments/modal_lanes.py`: executor wiring.
- `experiments/modal_runner.py`: serving runtime source/manifest packaging.
- `tests/test_adaptive_scheduler_glue.py`, `tests/test_budget_planner.py`,
  `tests/test_modal_runner_config.py`: lineage, canonical isolation and launch checks.
- This worklog: audit, preservation, failures/skips, commands and claim ceiling.

**Claim ceiling:** async feedback lineage is IMPLEMENTED and source-contract
tested within pinned V1 / UniProc / single-worker padded draft_model assumptions.
Real pinned-runtime integration is **IMPLEMENTED BUT UNVERIFIED**. Strict source
hash gates intentionally reject other installations. Old launchers using the
scheduler without the lineage executor retain upstream stats but skip adaptive
feedback with observable missing-lineage warnings; they must use the paired backend.
Private vLLM hooks and deployment startup still require real-runtime validation.

Historical L4 scalar/observer evidence is unchanged and does not verify this
new lineage mechanism. Distributed/multiworker lineage and arbitrary-tree
execution are OUT OF SCOPE; profitability/hardware-derived policy NOT EVALUATED.
No expensive GPU rerun was performed merely for cleanup. Full suite is not green:
the eight broad failures above remain unresolved outside this implementation.

Self-review checked latest-plan attribution, duplicate/missing/stale lineage,
request reuse, no-callback/K=0 cleanup, invalid-token counts, source gates,
scalar behavior, dormant-tree reachability and unsupported claims. `git diff
--check` and staged diff checks passed. Only exact task paths were staged;
original untracked evidence remains untouched. Implementation is committed on
the new feature branch; nothing pushed or merged. Worklog evidence follows in
a separate local commit.

Next action: review the local feature diff before authorizing pinned-runtime
integration validation.

---

# 26. F1 startup pairing enforcement (2026-11-12)

**Historical F1 verdict: MAJOR -> RESOLVED at CPU/source-contract level for
fresh unpaired startup only; superseded by the R1/R2 ownership review below.**
The previous tests passed but did not establish generation ownership. Real pinned-runtime
integration remains UNVERIFIED; no GPU, deployment or runtime benchmark was run.
Fix commit: `eeb62ee32ac033a7e64bf3aa1b870dd70767f102`.

## Verified starting state and root cause

- Existing branch retained: `feat/m4-02-04-async-lineage-v02`.
- Pre-fix HEAD: `9f80960b8f86ad0fc425db2ff9e8d720fef926de`.
- Local origin/develop and merge base: `53b813a0b6ce8fef530c4286fbd8bf4da5b4a149`.
- No tracked changes before edits. The three previously recorded untracked
  evidence directories were preserved. No branch switch/new branch or fetch
  was needed for this expected local feature-state fix.
- Scheduler previously called `ledger_for(config)`, which created an empty
  ledger without proving executor readiness. Its ready log preceded the first
  missing-lineage warning; ordinary UniProc serving could therefore start
  without project adaptive feedback.
- Pinned authority remains captured vLLM **0.30.0**, V1, UniProc, one worker,
  padded draft_model. Inspected the installed capture under
  `experiments/runs/m4_02_04/20261110-async-lineage-design/installed/vllm/`,
  especially `v1/engine/core.py`, `v1/executor/abstract.py`,
  `v1/executor/uniproc_executor.py`, and scheduler configuration. Existing
  source-hash checks passed; no installed local vLLM engine was used.

## Startup boundary and chosen handshake

Pinned `EngineCore.__init__` initializes `executor_class(vllm_config)` BEFORE
constructing `Scheduler(vllm_config=vllm_config, ...)`. The same config object
survives both calls. Configured backend resolution is available via pinned
`Executor.get_class`, including a qualified class name, but a configured name
alone does not establish that a live observer was installed. The existing
config-scoped ledger rendezvous supplied the minimal project-owned boundary;
there was no existing readiness registration to reuse.

- `register_lineage_executor(config, executor)` publishes a weak owner reference
  on that exact ledger/config only after the executor passes source/config/
  runner gates and installs its observer hook. Duplicate live owners and wrong
  config/ledger identities are rejected.
- `AdaptiveAsyncScheduler.__init__` calls `require_lineage_executor(config)`
  BEFORE base scheduler initialization, controller creation, readiness logging,
  scheduling or verification feedback. Missing/expired/unregistered capability
  raises RuntimeError explaining the required lineage-capable executor and the
  observed `distributed_executor_backend`.
- Merely configuring the right class string, supplying a lookalike class or
  pre-creating a ledger cannot satisfy startup. Neither an environment bypass
  nor an implicit diagnostic mode exists. CPU fixtures explicitly register a
  simulated executor through the same production contract; the executor-wrapper
  test exercises real registration after its initialization gates.
- The previous design described registration as engine-local, but it was only
  config-scoped: independent review subsequently disproved exclusive ownership
  and safe revocation when that config was reused (R1/R2 below). A weak
  owner prevented destroyed executors from leaving usable readiness. Existing
  feedback, origin, count, trimming and controller algorithms are unchanged.
- No circular initialization: executor needs only config, never the scheduler.
  This check is early scheduler initialization, not a claim that invalid config
  is rejected before upstream executor/model allocation.

## Launch paths and historical Colab handling

Shell Lane 3 and Modal generated args already select AdaptiveAsyncScheduler +
LineageUniProcExecutor; their runtime packaging remains unchanged. Tests verify
both, and the wrapper test resolves the actual generated backend using the
captured pinned `Executor.get_class` before exercising paired startup.
Non-adaptive schedulers/executors do not call the new validation.

The Colab script is the historical scalar observer reproduction, not current
lineage validation. Its CONFIG and exact plain-UniProc expectation remain intact.
Its live CLI stages now require the original immutable source SHA
`16528241468e2624ccfe07a69a67fcc123bfe4a3` before hardware checks. Reproduction
must use that frozen historical checkout; current lineage source has no bypass.
Offline validation of existing evidence remains available. Module documentation
and the runbook explicitly distinguish this historical profile from current
async-lineage QA. No historical raw/evidence files were modified or relabeled.

## Tests and results

Environment: Windows PowerShell, CPython 3.11.0, CPU pytest/source contracts,
using existing local pinned-source captures. No GPU/runtime execution.

| Gate | Evidence | Result |
| --- | --- | --- |
| P1 paired startup | keyword/positional config plus production executor-wrapper registration and shared-ledger assertion | PASS |
| P2 plain UniProc | `uni` backend + unregistered ledger rejected | PASS |
| P3 unrelated executor | unrelated name, uninitialized correct name, lookalike class with marker rejected | PASS |
| P4 non-adaptive | normal scheduler construction with `uni`; no global executor requirement | PASS (CPU stub) |
| P5 shell pairing | existing launcher/config tests | PASS |
| P6 Modal pairing | generated backend, source packaging and pinned class resolution | PASS (source/config) |
| P7 early failure | base-init/feedback sentinels prove failure before either boundary | PASS |
| P8 lineage regression | existing T1-T10 tests retain their origin/count/isolation/fail-safe assertions | PASS (CPU/source-contract) |

Additional tests cover config identity isolation, revoked/garbage-collected
owners, missing config, failed executor initialization, shutdown, pinned startup
order, and historical source rejection before live stages. No production check
is monkeypatched away in scheduler tests.

Commands/results (all local CPU; skipped tests are not counted as passes):

```text
1. python -m pytest -q tests/test_adaptive_scheduler_glue.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_modal_runner_config.py
   exit 0; 145 passed, 0 failed, 0 skipped (initial pairing tests).

2. python -m pytest -q tests/test_adaptive_scheduler_glue.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_modal_runner_config.py tests/test_scalar_colab_smoke.py
   exit 0; 162 passed, 0 failed, 0 skipped (final source-resolution, lookalike and historical-profile coverage included).

3. python -m pytest -q --tb=short tests/test_adaptive_scheduler_glue.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_budget_controller.py tests/test_budget_fallback_recovery.py tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_cost_model_l4.py tests/test_cost_model_solver.py tests/test_modal_runner_config.py tests/test_modal_smoke.py tests/test_scalar_context.py tests/test_scalar_colab_smoke.py tests/test_scalar_colab_git.py tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py
   exit 1; 400 passed, 7 failed, 0 skipped; no timing tests deselected.
```

Broad failures are the six known W1 CRLF/hash cases in test_modal_smoke.py and
test_tree_adaptive_controller.py::test_decision_latency_benchmark
(0.050354 ms against <0.05 ms). The cost-model timing test passed this time.
CRLF failures remain PROVEN PRE-EXISTING / UNRELATED by the preceding review's
baseline-module replay and raw-byte comparison. Timing failures remain LIKELY
PRE-EXISTING BUT NOT PROVEN; their timed implementation/tests are unchanged.
No new failure category appeared. Frozen workloads, hashes and timing thresholds
were not normalized or changed. The broader suite is NOT green.

## Changed files, self-review and ceiling

- `src/server/lineage.py`: explicit executor registration/require/revocation.
- `src/server/lineage_executor.py`: publish capability after gates/hook installation.
- `src/server/adaptive_scheduler.py`: require capability before base initialization.
- `tests/test_adaptive_scheduler_glue.py`: explicit CPU pairing fixtures, P1-P8,
  actual pinned construction-order/class-selection contracts.
- `experiments/scalar_colab_smoke.py`, `tests/test_scalar_colab_smoke.py`:
  historical-only live-source guard and offline-validation coverage.
- `docs/runbooks/m4-02-04-colab-observer-smoke.md`: historical/current boundary.
- This worklog: F1 root cause, fix, tests, failures and evidence ceiling.

Self-review: unpaired adaptive startup now fails; valid registered startup
passes; ordinary non-adaptive paths remain outside the check; canonical Lane 3
still selects both classes; there is no default-enabled bypass. No changes to
TreeAdaptive, hardware cost policy, acceptance thresholds, profiler, origin
algorithm, benchmark protocol or canonical scalar topology telemetry.
`future/tree-adaptive-v02` remains at the audited baseline SHA. Diff checks pass.

**Maximum claim:** adaptive async lineage implementation and scheduler/executor
pairing are source-contract tested for pinned V1 / UniProc / single-worker
configuration. Real pinned-runtime lineage, deployment, GPU, distributed/tree
execution, profitability and hardware-aware policy are NOT VERIFIED by this work.

Local feature-branch commits only, with exact-path staging. No push, merge, PR,
Trello mutation or historical evidence rewrite. Next action: technical review
of the F1 startup-pairing fix before real pinned-runtime integration validation.


---

# 27. R1/R2 startup-generation ownership correction (2026-11-12)

**R1 BLOCKER: RESOLVED at CPU/source-contract level.**
**R2 MAJOR: RESOLVED at CPU/source-contract level.**
Implementation commit: `8912950` (local feature branch).
Independent technical review remains the next gate. The earlier F1 implementation
passed its tests but FAILED independent ownership review; its historical fresh
startup results above do not establish the current ownership properties.

## Baseline and pre-edit reproductions

Verified before writes:

- Branch: `feat/m4-02-04-async-lineage-v02` (retained).
- HEAD: `afc4f13dc89367fa845cea00d8e014879119f31a`.
- `origin/develop`: `53b813a0b6ce8fef530c4286fbd8bf4da5b4a149`.
- Clean tracked state; exactly the original three untracked evidence directories.
- Command-scoped `git -c safe.directory=...` handles sandbox UID ownership;
  no global Git configuration changes, reset, clean, rebase or branch switch.

Executed a minimal CPU reproduction using the actual pre-fix registration and
AdaptiveAsyncScheduler code, with a stubbed upstream base, BEFORE editing:

```text
R1 config id:                1614188539056
   owner A id:               1614180709840
   registration return:      None (no generation handle existed)
   registration weakref id:  1614188085792
   executor A ledger id:     1614188039568
   scheduler A ledger id:    1614188039568
   scheduler B ledger id:    1614188039568
   backend after reuse:      'uni'
   inherited:                True (incorrect startup success)
R2 owner B id:               1614188180944
   ledger B id:              1614188180304
   B valid after A release/new B registration: True
   repeated release(A.vllm_config): removed B
   subsequent require(C): RuntimeError, no ready executor registration
```

R1 root cause: `ledger_for(config)` returned the same ledger and readiness was
reusable indefinitely. The old executor remaining alive made scheduler B appear
paired, even after ordinary UniProc backend selection.
R2 root cause: `release_ledger(config)` unconditionally popped the config slot;
it could not distinguish A teardown from the currently registered B generation.
These sequences are now explicit regression tests with actual object identities.

## Pinned construction authority and scope

Used the local captured installed **vLLM 0.30.0** source under
`experiments/runs/m4_02_04/20261110-async-lineage-design/installed/vllm/`.
The lifecycle test verifies raw SHA-256 against the captured
`installed_source_manifest.json` for EngineCore, Executor, UniProcExecutor and
Scheduler. Existing source-gate tests also verify the committed scalar manifest.
No installed engine, external latest-upstream assumptions, or GPU were used.

- `EngineCore.__init__` directly constructs `executor_class(vllm_config)` at
  core.py:134, then `Scheduler(vllm_config=vllm_config, ...)` at :162-163 on the
  same Python initialization thread. Executor.__init__ retains the config and
  synchronously calls `_init_executor`. The config is passed through directly;
  there is NO assumption that it is unique to an engine.
- There is exactly one assignment to `self.scheduler` in captured EngineCore;
  no scheduler recreation/multiple-consumer path exists in this pinned scope.
  Failed scheduler initialization does not restore consumed readiness: retry
  requires teardown and fresh executor publication.
- EngineCore.shutdown at :756-762 shuts down executor before scheduler.
  UniProc shutdown calls worker.shutdown; scheduler shutdown stops its own
  publishers/connectors. Neither creates a replacement scheduler.
- Inspected pinned Executor.get_class: `uni` resolves ordinary UniProcExecutor;
  qualified names resolve a subclass. Backend names/classes are configuration,
  never generation ownership evidence. Existing canonical resolution tests stay.
- Config class implementation itself is not in this local capture (only
  scheduler/speculative config modules are captured). No uniqueness guarantee
  from VllmConfig internals is needed or claimed.

## Current corrected ownership model

`register_lineage_executor(config, executor)` now allocates a fresh ledger and
returns an immutable, identity-based `LineageRegistration` with an opaque token,
exact config, weak executor owner and publishing Thread object. The executor
retains both handle and its exact ledger after source/config/runner gates and
observer installation. The old config-indexed ledger factory is removed.

Config identity indexes a process-local rendezvous slot; it does not authorize
consumption more than once, identify a generation, or authorize release.
`require_lineage_executor(config)` atomically consumes that slot ONCE under an
RLock, checks live owner/config/ledger identities and publishing-thread identity,
and returns the exact handle to the scheduler before upstream scheduler init,
controller construction, readiness logging or feedback. The scheduler retains
that handle and binds only `handle.ledger`; it never creates a fallback ledger.

This uses the pinned synchronous executor-before-one-scheduler sequence, not
config uniqueness. A later scheduler cannot consume A's already bound token,
even if A remains live or the backend changes. Cross-thread pending-generation
consumption fails without consuming the rightful thread's publication. Live
same-config replacement/overlap is explicitly rejected; same-config restart is
supported only after old handle release. Concurrent different-config initializations
are isolated. This is a process-local handshake, not distributed synchronization.

`release_ledger(handle)` removes the slot only when its current handle IS that
handle, then clears only the supplied generation's ledger. Passing a config now
raises TypeError; None is harmless for pre-publication executor init failure.
Repeated/stale scheduler or executor shutdown cannot remove B or clear B's ledger.
Owner GC callbacks capture only integer config key + opaque token; they compare
the current token before deletion/clear. They retain neither owner nor config
nor handle. A GC test proves current-owner expiration clears its ledger and
releases the config; stale callbacks after B publication leave B intact.

The registry lock protects publication, consumption, explicit release and weak
expiration. It does not claim to make upstream worker execution or speculative
ledger mutation generally thread-safe. Runtime origins, row binding, feedback,
request retirement and scalar controller behavior are unchanged.

## Ownership and startup regression evidence

| Gate | Regression evidence | Result |
| --- | --- | --- |
| E1 | Two configs, actual distinct ledger/token identities and exact owner bindings | PASS |
| E2 / R1 | Bound A; same config switched to uni; B rejected before base init/feedback | PASS |
| E3 | A released, B valid restart; distinct handle/token/ledger | PASS |
| E4 / R2 | Repeated stale A release and A scheduler teardown preserve B marker/registration | PASS |
| E5 | Repeated executor/scheduler shutdown, before and after restart | PASS |
| E6 | Real old-owner GC plus deterministic old callback after B publication; B still binds | PASS |
| E7 | Second scheduler rejected with consumed diagnostic, even after first scheduler deletion | PASS |
| E8 | Four concurrent distinct-config initializations; all ledger/token identities unique | PASS |

Additional ownership tests prove live same-config replacement fails without
changing A, config-only release is invalid, wrong-thread binding fails without
consumption, concurrent same-config publications yield exactly one winner,
and registry weak ownership does not retain a collected config/owner.
Production executor and scheduler shutdown methods are exercised with CPU upstream
stubs for both executor-first (pinned) and scheduler-first (defensive) ordering;
B starts between old component shutdown calls, and both old shutdowns repeat.

P1-P7 remain PASS: valid keyword/positional paired startup; fresh plain and
unrelated/uninitialized/lookalike executor rejection; ordinary non-adaptive
scheduler unaffected; shell and Modal canonical pairing/packaging/resolution;
rejection before base init and feedback. T1-T10 all remain PASS in the existing
CPU/source-contract tests, retaining exact origin/count/identity assertions.
Historical scalar Colab guard and its scientific evidence were not modified.

Environment: Windows PowerShell / CPython 3.11.0 / CPU pytest + captured-source
contracts. Exact commands and totals:

```text
python -m pytest -q tests/test_adaptive_scheduler_glue.py -k ownership --tb=short
13 passed, 0 failed, 0 skipped (54 deselected), exit 0.

python -m pytest -q tests/test_adaptive_scheduler_glue.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_modal_runner_config.py tests/test_scalar_colab_smoke.py --tb=short
175 passed, 0 failed, 0 skipped, exit 0 (final code).

python -m pytest -q --tb=short tests/test_adaptive_scheduler_glue.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_budget_controller.py tests/test_budget_fallback_recovery.py tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_cost_model_l4.py tests/test_cost_model_solver.py tests/test_modal_runner_config.py tests/test_modal_smoke.py tests/test_scalar_context.py tests/test_scalar_colab_smoke.py tests/test_scalar_colab_git.py tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py
413 passed, 7 failed, 0 skipped, exit 1 (final code; no deselection).
```

The first broad run, before adding the final GC-retention test/publication-window
guard, reported 412 passed, 7 failed, 0 skipped; timing was 53.627 us > 50 us.
Final broad failures exactly match the known categories: six W1 CRLF/hash
failures in test_modal_smoke.py and the unchanged tree-controller decision
latency benchmark at 50.540 us > 50 us. No new failure category or unexplained
regression appeared. The cost-model timing test passed. The historical baseline
was 400 passed / 7 failed / 0 skipped; the 13 added ownership cases all pass.
No frozen CRLF inputs, workload hashes, controller implementations or timing
thresholds were changed. Broader regression is NOT green.

## Self-review, changed files and claim ceiling

Reviewed full diff/status and exact-path scope; `git diff --check` passes.
Checked: consumed readiness cannot be reused, config-only release is impossible,
stale owner callback/shutdown cannot touch a newer ledger, no scheduler/global
ledger fallback, actual identity assertions, no new registry strong-reference
cycle, atomic registry mutations, and non-adaptive isolation. No shared external
interface owner/caller was found; all local API callers were migrated together.

Changed files: `src/server/lineage.py`, `src/server/lineage_executor.py`,
`src/server/adaptive_scheduler.py`, `tests/test_adaptive_scheduler_glue.py`, and
this worklog. Shell/Modal launch wiring, historical Colab source handling,
TreeAdaptive, topology, scalar policy, cost model and benchmark protocol unchanged.

**Maximum claim:** Adaptive async lineage startup pairing, engine-generation
ownership, and owner-specific revocation are source-contract tested for the
pinned V1 / UniProc / single-worker assumptions.

Real pinned-runtime, deployment, GPU, distributed/multi-process ownership,
TreeAdaptive runtime, profitability and hardware-aware policy remain UNVERIFIED.
Historical L4 evidence does not validate this handshake. Local exact-path commits
only; three untracked evidence directories preserved; no push, merge, PR, Trello
mutation, GPU, live Colab or Modal execution.

Next action: independent technical review of the corrected ownership handshake.
