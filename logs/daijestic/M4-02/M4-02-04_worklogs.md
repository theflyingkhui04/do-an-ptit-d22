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