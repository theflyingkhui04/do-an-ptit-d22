# Worklog Kỹ Thuật: [M3-05] Benchmark Đối Chứng 4 Làn trên NVIDIA L4

## 0. Thông Tin Công Việc

- **Task:** `[M3-05] Mở rộng Benchmark Harness & Đo kiểm Thực nghiệm Đối chứng 4 Làn`
- **Người thực hiện chính:** Bùi Thiên Đại (`daijestic`) — Lead 70%, Benchmark / MLOps / Experiment Execution / Evidence
- **Partner / Reviewer:** Cấn Đức Khôi — Partner 30%, PM + Technical Review
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Frozen implementation SHA:** `62fe60e2fab16a1770ee96743d3b175727b3fa14`
- **Evidence branch:** `evidence/m3-05-colab-confirmatory`
- **Final campaign evidence commit trước worklog:** `0114379df957e57dd43c663c3b95339fee4dada0`
- **Backend thực nghiệm:** Google Colab
- **GPU:** NVIDIA L4
- **vLLM:** `0.30.0`
- **PyTorch:** `2.13.0+cu130`
- **CUDA:** `13.0`
- **Thời gian thực hiện / điều tra chính:** 31/10/2026 – 05/11/2026
- **Trạng thái kỹ thuật:** `DONE WITH VALID EXPERIMENT EVIDENCE`
- **Trạng thái hypothesis ban đầu:** `NOT SUPPORTED BY EVIDENCE`
- **PM decision mới nhất:** đóng M3-05 với evidence hiện tại; phần tích hợp Adaptive tree/controller vào EAGLE Lane 4 chuyển sang phase tiếp theo.

Worklog này ghi lại đầy đủ luồng:

```text
scope
→ source freeze
→ 4-lane integration
→ runtime/backend validation
→ pilot
→ campaign
→ telemetry blocker
→ root-cause investigation
→ upstream fix
→ controlled patch validation
→ campaign resume
→ aggregate
→ scientific conclusion
→ PM handoff
```

---

# 1. Executive Summary

M3-05 hoàn thành một campaign confirmatory gồm **4 làn**:

1. Vanilla.
2. Static Draft 0.5B.
3. Adaptive Draft 0.5B.
4. Custom EAGLE-3.1.

Ma trận thực nghiệm:

```text
4 lanes
× 3 workloads: W1, W2, W3
× 6 concurrency levels: 1, 2, 4, 8, 16, 32
= 72 cells
```

Mỗi cell:

```text
3 warmups
5 repeats
32 measured requests / repeat
= 160 measured requests / cell
```

Toàn campaign:

```text
72 × 160 = 11,520 measured requests
72 × 3   = 216 warmup requests
```

Final aggregate:

```text
scientific_status = VALID
cells             = 72 / 72 VALID
backend           = google-colab
core_sha          = 62fe60e2fab16a1770ee96743d3b175727b3fa14
```

Artifacts chính:

```text
experiments/results/m3_sweep_4lanes_summary.json

experiments/results/figures/m3/
├── F1_m3_tpot_throughput_4lanes.png
└── F2_m3_speedup_vs_vanilla.png
```

Kết luận khoa học quan trọng:

- Campaign **VALID** không đồng nghĩa hypothesis phải đúng.
- Adaptive Lane 3 không triệt tiêu negative speedup ở W3.
- EAGLE Lane 4 tốt ở low/medium concurrency nhưng cũng crossover xuống dưới Vanilla ở tải cao.
- EAGLE Lane 4 trong 72-cell campaign là **EAGLE3 fixed `K=3`**, chưa tích hợp Adaptive tree/controller của nhóm.
- PM quyết định **đóng M3-05 với negative/boundary result hợp lệ**.
- Phase tiếp theo sẽ tích hợp Adaptive tree/controller vào EAGLE rồi đo contribution trực tiếp.

Không rerun hoặc tune M3-05 chỉ để làm số đẹp.

---

# 2. Source of Truth Và Conflict Resolution

## 2.1. Trello M3-05

Card M3-05 ban đầu đặt mục tiêu:

- mở rộng harness từ 3 sang 4 lanes;
- thêm Custom EAGLE-3.1;
- chạy W1/W2/W3;
- xuất summary;
- tạo F1/F2;
- kỳ vọng Lane 3 và Lane 4 triệt tiêu hoàn toàn negative speedup ở W3.

## 2.2. Evidence thực nghiệm

Final evidence cho thấy:

- Adaptive Lane 3 vẫn có speedup `< 1.0`.
- EAGLE Lane 4 cũng có speedup `< 1.0` tại concurrency cao.

## 2.3. PM decision mới nhất

PM quyết định:

1. Đóng M3-05 với evidence hiện tại.
2. Không rerun để ép kết quả đạt expectation cũ.
3. Giữ 72-cell result làm baseline.
4. Tiếp theo tích hợp Adaptive tree/controller vào EAGLE.
5. Benchmark lại để đo:

```text
EAGLE fixed K
→ EAGLE + Adaptive
→ cải thiện từ bao nhiêu lên bao nhiêu
```

## 2.4. CONFLICT

```text
Source A:
Trello wording cũ yêu cầu “triệt tiêu hoàn toàn negative speedup”.

Source B:
72-cell evidence không hỗ trợ claim đó.

Source C:
PM latest decision cho phép đóng M3-05 và chuyển Adaptive-EAGLE sang phase sau.
```

Theo governance:

```text
PM latest decision > stale Trello wording
```

Vì vậy M3-05 được đóng bằng **negative result hợp lệ**, không sửa số và không cherry-pick.

---

# 3. Kiến Trúc 4 Làn Được Đo

## 3.1. Lane 1 — Vanilla

Target model:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

Revision:

```text
b25037543e9394b818fdfca67ab2a00ecc7dd641
```

Không bật speculative decoding.

Vai trò:

```text
baseline
```

Speedup canonical:

```text
speedup_vs_vanilla =
    Vanilla TPOT p50
    /
    Lane TPOT p50
```

---

## 3.2. Lane 2 — Static Draft

Draft model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Revision:

```text
7ae557604adf67be50417f59c2c2f167def9a775
```

Configuration:

```text
method = draft_model
K      = 3 fixed
```

Vai trò:

```text
fixed-depth speculative decoding baseline
```

---

## 3.3. Lane 3 — Adaptive Draft

Draft:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Components:

```text
AdaptiveDraftController
AdaptiveBudgetPlanner
AdaptiveAsyncScheduler
```

Runtime behavior:

```text
per-request k_i
→ batch planner
→ batch K = max(k_i)
```

Có:

- acceptance tracking;
- periodic probing;
- memory-pressure rule;
- adaptive telemetry;
- runtime tree budget.

Lane 3 là lane trong campaign thực sự sử dụng Adaptive controller của nhóm.

---

## 3.4. Lane 4 — EAGLE-3.1

Model:

```text
duckhoi/qwen2.5-7b-eagle3.1-speculator
```

Revision:

```text
c3adca7e211b471d97a519a625df7ffd37b8b461
```

Configuration:

```text
method = eagle3
num_speculative_tokens = 3
```

Tức:

```text
K = 3 fixed
```

Quan trọng:

Lane 4 **KHÔNG có**:

```text
AdaptiveAsyncScheduler
AdaptiveDraftController
AdaptiveBudgetPlanner
ADAPTIVE_CONTROLLER_KWARGS
ADAPTIVE_TELEMETRY_PATH
```

Do đó M3-05 đo:

```text
EAGLE3 + fixed K=3
```

chứ chưa đo:

```text
EAGLE3 + Adaptive tree/controller
```

---

# 4. Experiment Contract

Campaign được freeze với:

```text
GPU              = NVIDIA L4
vLLM             = 0.30.0
torch            = 2.13.0+cu130
CUDA             = 13.0
target revision  = pinned
draft revisions  = pinned
sample policy    = cycle-rotate
warmups          = 3 / cell
repeats          = 5
requests/repeat  = 32
measured/cell    = 160
```

Các nguyên tắc:

- raw evidence trước aggregate;
- failed attempt phải giữ;
- invalid attempt không được trộn vào aggregate;
- negative speedup phải giữ;
- không loại request vì số xấu;
- environment provenance phải ghi rõ;
- một cell chỉ VALID khi chính evidence của cell đó đủ.

---

# 5. Luồng Hệ Thống M3-05

## 5.1. Source Freeze

Campaign core được freeze tại:

```text
62fe60e2fab16a1770ee96743d3b175727b3fa14
```

Các file chính:

```text
experiments/benchmark_harness.py
experiments/modal_lanes.py
experiments/modal_runner.py
experiments/run_m3_campaign.py
experiments/aggregate_results.py

src/proposer/
src/server/
```

Mục đích:

```text
mọi cell
→ cùng implementation
→ cùng model revision
→ cùng workload contract
→ trace được về một SHA
```

---

## 5.2. Mở rộng Harness Sang EAGLE

Các thay đổi chính:

- thêm EAGLE model identity;
- pinned revision;
- thêm Lane 4;
- runner support;
- aggregate support;
- plot 4-lane F1/F2;
- giữ các phần generic theo `lane`.

---

## 5.3. Backend Migration Modal → Colab

Backend ban đầu:

```text
Modal L4
```

Sau blocker execution / operational constraints, PM duyệt:

```text
Google Colab L4
```

Điều kiện giữ tính khoa học:

- vẫn NVIDIA L4;
- versions pinned;
- same source SHA;
- same workload;
- same harness;
- same model revisions;
- backend ghi rõ trong evidence.

Final summary:

```json
{
  "backend": "google-colab",
  "pm_backend_decision": "APPROVED"
}
```

---

# 6. Pilot

Pilot:

```text
m305-colab-pilot-v1-eagle-W1-c1-a001
```

Configuration:

```text
Lane        = eagle
Workload    = W1
Concurrency = 1
Warmups     = 3
Repeats     = 5
Requests    = 32 / repeat
Measured    = 160
```

Pilot xác minh:

- model load được;
- request success;
- metrics complete;
- raw JSONL đúng;
- evidence manifest đúng;
- Colab runtime contract PASS.

Pilot được checkpoint lên Git trước full campaign.

---

# 7. Reuse Gate

Trước blocker telemetry, các cell sau đã hoàn thành hợp lệ:

```text
Vanilla:   18 / 18
Static W1:  6 / 6
------------------
Reusable:  24 / 72
```

Một cell chỉ reuse nếu:

```text
N = 160 measured

repeat structure:
r001 = 32
r002 = 32
r003 = 32
r004 = 32
r005 = 32

success = 160
metrics_complete = 160

spec metrics complete
nếu lane cần acceptance telemetry
```

Không rerun 24 valid cells chỉ vì một cell sau đó gặp lỗi.

---

# 8. Telemetry Blocker tại Static W2 c1

Campaign dừng tại:

```text
static × W2 × concurrency=1
```

Attempt a001:

```text
N       = 160
success = 160
spec    = 159
missing = 1
```

Attempt a002:

```text
N       = 160
success = 160
spec    = 158
missing = 2
```

Aggregator fail:

```text
Static measured record missing server_metrics
```

Đây là fail-closed đúng:

```text
missing telemetry
→ không aggregate subset
→ không bỏ request lỗi
→ không cherry-pick
```

---

# 9. Diagnostic Investigation

## 9.1. Diagnostic ngắn

Một short diagnostic 20 requests:

```text
0 missing
```

Điều này không bác bỏ bug vì failure intermittent.

---

## 9.2. Diagnostic long lifecycle

Diagnostic dài tái hiện omission.

Request lỗi vẫn:

```text
HTTP 200
usage present
output tokens present
finish_reason valid
```

Nhưng raw final SSE thiếu:

```text
metrics.speculative_decoding
```

Parser mismatch:

```text
0
```

Do đó:

```text
raw server response
đã thiếu telemetry
trước application parser
```

---

# 10. Diagnostic #3 — Strong Reproduction

Protocol được định trước:

```text
2 independent fresh vLLM servers

per server:
3 warmups
5 repeats
32 measured / repeat
160 measured

total:
320 measured requests
```

Capture:

- raw SSE;
- Prometheus before request;
- Prometheus after request;
- không dừng khi gặp anomaly.

Kết quả stock vLLM 0.30.0:

```text
Server 1:
2 / 160 missing

Server 2:
1 / 160 missing

TOTAL:
3 / 320 missing
```

Các anomaly đều:

```text
HTTP 200
usage present
final usage chunk present
metrics absent
```

Trong khi:

```text
vllm:spec_decode_num_drafts_total          tăng
vllm:spec_decode_num_draft_tokens_total    tăng
vllm:spec_decode_num_accepted_tokens_total tăng
```

Theo tài liệu chính thức của vLLM, `metrics.speculative_decoding` cung cấp per-request acceptance metrics như:

```text
num_spec_steps
num_draft_tokens
num_accepted_draft_tokens
mean_acceptance_length
draft_acceptance_rate
acceptance_histogram
```

và các giá trị per-request này có counterpart ở server-level Prometheus metrics.

---

# 11. Upstream Issue

Issue được tạo:

```text
https://github.com/vllm-project/vllm/issues/65644
```

Title:

```text
[Bug]: Streaming final usage chunk intermittently omits
metrics.speculative_decoding while spec-decode Prometheus
counters advance
```

Issue bao gồm:

- environment;
- exact command;
- request conditions;
- raw SSE;
- Prometheus delta;
- predefined reproduction protocol;
- 3/320 result.

---

# 12. Root Cause Và Upstream Fix

Tìm được upstream fix:

```text
PR #55708
```

Commit:

```text
6c2925fe5fe8156475affbfa995ed272c96fe4fa
```

Fix:

```text
Preserve speculative metrics when coalescing outputs
```

vLLM `0.30.0` chưa có phần preservation này trong:

```text
RequestOutput.add(..., aggregate=True)
```

Minimal backport:

```python
if next_completion.spec_decode_metrics is not None:
    completion.spec_decode_metrics = (
        next_completion.spec_decode_metrics
    )
```

Local patch commit:

```text
8c20ed952e20db8c04fdf9a6aa51585e285079b9
```

Patched `outputs.py` SHA256:

```text
8f58684245f17007863382ffa9de115d371738ae28731b0c876196eb3b3fea47
```

Runtime gate:

```text
vLLM version = 0.30.0
patch present = True
loaded code   = True
SHA256        = expected
```

---

# 13. Patch Validation

Cùng predefined protocol:

## Stock vLLM 0.30.0

```text
2 servers
320 measured
3 / 320 missing
```

## vLLM 0.30.0 + #55708 backport

```text
2 servers
320 measured
0 / 320 missing
```

Claim discipline:

Không viết:

```text
“bug đã biến mất vĩnh viễn”
```

Được viết:

```text
Trong predefined 320-request validation protocol,
failure tái hiện trên stock v0.30.0 nhưng không tái hiện
sau minimal backport #55708.
```

Per-request speculative metrics là experimental telemetry của vLLM; các field summary gồm:

```text
num_spec_steps
num_draft_tokens
num_accepted_draft_tokens
draft_acceptance_rate
mean_acceptance_length
acceptance_histogram
```

---

# 14. Campaign Resume

## 14.1. 24 cells được reuse

```text
Vanilla:    18 / 18
Static W1:   6 / 6
Total:      24 / 72
```

Historical failed attempts:

```text
static-W2-c1-a001
static-W2-c1-a002
```

được giữ lại.

Không đưa vào aggregate.

---

## 14.2. Resume Point

Canonical resume:

```text
static × W2 × c=1
```

Final selected attempt:

```text
m305-confirmatory-colab-v1-static-W2-c1-a006
```

Sau đó chạy toàn bộ remaining grid.

---

# 15. Final Campaign Result

Final:

```text
72 / 72 VALID
```

Measured:

```text
11,520
```

Warmups:

```text
216
```

Summary:

```text
experiments/results/m3_sweep_4lanes_summary.json
```

Final evidence commit:

```text
0114379df957e57dd43c663c3b95339fee4dada0
```

Final remote gate:

```text
LOCAL == REMOTE
PASS
```

---

# 16. W3 Performance Result

## TPOT speedup vs Vanilla

| Concurrency | Static K=3 | Adaptive 0.5B | EAGLE3 K=3 |
|---:|---:|---:|---:|
| 1  | 1.022x | 0.745x | 1.238x |
| 2  | 1.027x | 0.709x | 1.222x |
| 4  | 1.028x | 0.637x | 1.183x |
| 8  | 0.994x | 0.581x | 1.088x |
| 16 | 0.928x | 0.522x | 0.931x |
| 32 | 0.754x | 0.474x | 0.669x |

---

# 17. Scientific Interpretation

## FACT

EAGLE fixed K=3:

```text
W3 c1  = 1.238x
W3 c2  = 1.222x
W3 c4  = 1.183x
W3 c8  = 1.088x
W3 c16 = 0.931x
W3 c32 = 0.669x
```

Tức:

```text
c1-c8  : faster than Vanilla
c16-c32: slower than Vanilla
```

## FACT

Adaptive Lane 3:

```text
W3 c1  = 0.745x
W3 c2  = 0.709x
W3 c4  = 0.637x
W3 c8  = 0.581x
W3 c16 = 0.522x
W3 c32 = 0.474x
```

## OBSERVATION

Negative speedup tăng mạnh khi concurrency tăng.

## INFERENCE

Có khả năng speculative cost tăng nhanh hơn gain khi active batch lớn.

## UNKNOWN

M3-05 chưa cô lập được causal contribution của:

- draft model quality;
- K;
- verification cost;
- scheduler;
- batching;
- memory pressure;
- custom controller overhead.

---

# 18. W1 / W2 Boundary Results

## Adaptive W1

```text
c1  = 1.448x
c2  = 1.353x
c4  = 1.309x
c8  = 1.201x
c16 = 1.017x
c32 = 0.863x
```

## Adaptive W2

```text
c1  = 1.253x
c2  = 1.114x
c4  = 0.874x
c8  = 0.756x
c16 = 0.729x
c32 = 0.669x
```

## EAGLE W1

```text
c1  = 1.415x
c2  = 1.405x
c4  = 1.303x
c8  = 1.123x
c16 = 0.911x
c32 = 0.691x
```

## EAGLE W2

```text
c1  = 1.245x
c2  = 1.214x
c4  = 1.167x
c8  = 1.063x
c16 = 0.868x
c32 = 0.647x
```

Observed pattern:

```text
low / medium concurrency
→ speculative lanes có lợi

high concurrency
→ crossover
→ negative speedup
```

Cơ chế này phù hợp với mô tả Dynamic Speculative Decoding của vLLM: khi active batch size tăng, verification workload hiệu dụng tăng theo khoảng `BS × K`; nếu vượt ngưỡng compute phù hợp thì speculative decoding có thể làm TPOT xấu hơn. Dynamic SD vì vậy giảm K khi batch/load tăng.

---

# 19. Audit Adaptive Lane 3 Sau Campaign

Source audit cho thấy:

```text
AdaptiveDraftController.select_budget(...)
```

nhận `batch_size`, nhưng current runtime policy chưa thực sự dùng measured:

```text
CostProfile C(k, B)
```

để chọn K theo batch.

Current policy chủ yếu dựa trên:

- alpha;
- fallback threshold;
- high threshold;
- memory pressure;
- periodic probing.

`AdaptiveBudgetPlanner` dùng:

```text
batch_k = max(per_request_k)
```

Tức:

```text
chỉ cần một request cần K>0
→ cả batch vẫn đi speculative path
```

Finding:

```text
Current implementation chưa hoàn toàn phản ánh
design narrative “hardware-aware C(k,B) runtime decision”.
```

Finding này không invalidate 72-cell campaign.

Campaign vẫn VALID cho **implementation thực tế đã chạy**.

---

# 20. Xác Minh Lane 4 EAGLE Chưa Có Adaptive

Frozen config EAGLE:

```text
method = eagle3
num_speculative_tokens = 3
```

Không có:

```text
scheduler_cls = AdaptiveAsyncScheduler
```

Không có:

```text
AdaptiveDraftController
```

Không có:

```text
AdaptiveBudgetPlanner
```

Không có:

```text
ADAPTIVE_TELEMETRY_PATH
ADAPTIVE_CONTROLLER_KWARGS
```

Command thực tế ở:

```text
experiments/runs/m3_01/
m305-confirmatory-colab-v1-eagle-W3-c32-a001/
vllm_command.json
```

xác nhận EAGLE chạy:

```text
K = 3 fixed
```

Kết luận:

```text
M3-05 Lane 4
=
EAGLE fixed K=3 baseline
```

chưa phải:

```text
EAGLE + Adaptive tree/controller
```

---

# 21. Vì Sao Điều Này Quan Trọng

Trong M3-05:

```text
Adaptive Lane 3
=
0.5B draft model
+
Adaptive controller
```

Trong khi:

```text
EAGLE Lane 4
=
EAGLE3
+
fixed K=3
```

Nếu so trực tiếp:

```text
Lane 3 vs Lane 4
```

thì có ít nhất hai biến thay đổi:

1. draft model;
2. adaptive policy.

Không thể cô lập contribution của Adaptive.

Scientific comparison sạch hơn phải là:

```text
EAGLE fixed K=3
vs
EAGLE + Adaptive controller
```

Khi đó:

```text
draft model = same
target model = same
workload = same
GPU = same
harness = same

chỉ thay:
fixed K
→ Adaptive K/tree
```

---

# 22. PM Decision Sau M3-05

PM chốt:

1. Đóng M3-05.
2. Giữ 72-cell campaign làm baseline.
3. Không rerun M3-05 để “săn” positive speedup.
4. Hoàn thiện / tích hợp Adaptive tree/controller.
5. Apply Adaptive vào EAGLE Lane 4.
6. Benchmark lại.
7. Báo:

```text
EAGLE fixed:
A.x

EAGLE + Adaptive:
B.x

Adaptive contribution:
A.x → B.x
```

Đây là phase mới.

Không sửa canonical M3-05 summary bằng future result.

---

# 23. Follow-up Experiment

## 23.1. Baseline

Đã có:

```text
EAGLE3 fixed K=3
```

trên:

```text
W1/W2/W3
c1/c2/c4/c8/c16/c32
```

## 23.2. Intervention

Future:

```text
EAGLE3
+
Adaptive tree/controller
```

## 23.3. Comparison

Primary:

```text
EAGLE fixed
vs
EAGLE Adaptive
```

Metric:

- TPOT p50/p95.
- TTFT.
- E2E.
- throughput.
- VRAM.
- alpha.
- tau.
- `num_draft_tokens`.
- `num_accepted_draft_tokens`.
- effective K distribution.
- speedup vs Vanilla.
- improvement vs fixed EAGLE.

Ví dụ báo cáo:

```text
W3 c32

EAGLE fixed:
speedup = 0.669x
TPOT    = 34.113 ms

EAGLE Adaptive:
speedup = X.XXXx
TPOT    = X ms

Improvement:
0.669x → X.XXXx
```

Không điền X trước khi có experiment.

---

# 24. Evidence Inventory

## 24.1. Final Summary

```text
experiments/results/m3_sweep_4lanes_summary.json
```

## 24.2. Figures

```text
experiments/results/figures/m3/
```

## 24.3. Campaign Raw Evidence

```text
experiments/runs/m3_01/
m305-confirmatory-colab-v1-*
```

Artifacts mỗi cell gồm:

```text
raw.jsonl
warmup.jsonl
raw.vram.jsonl
vram.jsonl
server_env.json
vllm_command.json
harness.log
vllm.log
identity_check.json
evidence_manifest.json
summary.csv
```

Adaptive lane có thêm:

```text
adaptive_telemetry.jsonl
```

## 24.4. Debug Evidence

```text
experiments/runs/m3_05_debug/
```

## 24.5. Upstream Issue

```text
https://github.com/vllm-project/vllm/issues/65644
```

## 24.6. Upstream Fix

```text
https://github.com/vllm-project/vllm/pull/55708
```

---

# 25. Key SHAs

Campaign source:

```text
62fe60e2fab16a1770ee96743d3b175727b3fa14
```

vLLM v0.30.0 base:

```text
ced6857afa0ea7b2e3f0846a62e1394e90f15607
```

Upstream spec-metrics preservation fix:

```text
6c2925fe5fe8156475affbfa995ed272c96fe4fa
```

Local minimal backport:

```text
8c20ed952e20db8c04fdf9a6aa51585e285079b9
```

Patched `outputs.py` SHA256:

```text
8f58684245f17007863382ffa9de115d371738ae28731b0c876196eb3b3fea47
```

Final campaign evidence commit:

```text
0114379df957e57dd43c663c3b95339fee4dada0
```

---

# 26. Claim Discipline

## FACT

- 72/72 cells VALID.
- 11,520 measured requests.
- 216 warmups.
- backend = Google Colab L4.
- final summary `scientific_status = VALID`.
- EAGLE Lane 4 = fixed K=3.
- Lane 4 chưa dùng Adaptive controller.
- stock vLLM 0.30.0 tái hiện telemetry omission 3/320.
- patched validation 0/320 trong predefined protocol.
- final evidence được push và remote verification PASS.

## OBSERVATION

- Adaptive W3 negative ở mọi concurrency.
- EAGLE W3 positive tới c8 nhưng negative tại c16/c32.
- degradation xuất hiện khi load tăng.

## INFERENCE

- speculative overhead có khả năng tăng không thuận lợi khi batch/load lớn.
- current Adaptive path có runtime overhead đáng kể.
- fixed K có thể không phù hợp mọi load.

## HYPOTHESIS

```text
EAGLE + Adaptive tree/controller
sẽ giảm wasted speculation ở high concurrency
so với EAGLE fixed K=3.
```

Chưa được verify.

## KHÔNG ĐƯỢC CLAIM

```text
Adaptive algorithm đã tối ưu.
EAGLE Adaptive chắc chắn >1x.
M3-05 chứng minh Hardware-Aware controller thành công hoàn toàn.
0/320 nghĩa bug không bao giờ quay lại.
```

---

# 27. Protocol Deviations / Risks

## 27.1. Backend Change

```text
Modal → Google Colab
```

Status:

```text
PM approved
```

Mitigation:

- same GPU class;
- pinned versions;
- provenance recorded.

---

## 27.2. Runtime Telemetry Patch

Campaign gồm:

```text
24 reusable cells trước patch
+
remaining cells sau minimal patch
```

Risk:

mixed telemetry propagation implementation.

Mitigation:

- patch chỉ sửa metrics preservation;
- không đổi decoding algorithm;
- old cells audit riêng;
- failed attempts giữ;
- runtime provenance lưu lại.

---

## 27.3. Campaign Ordering

Execution phần lớn theo grid / lane order.

Nếu protocol lý tưởng mong muốn interleaving lanes để giảm temporal drift thì đây là deviation cần ghi.

Không có evidence đủ mạnh để kết luận ordering là nguyên nhân performance crossover.

---

## 27.4. TTFT Semantics

Streaming chunk đầu có thể là role / metadata event trước actual first generated token.

Do đó TTFT semantics cần review kỹ trước publication.

Primary speedup trong campaign dùng TPOT.

---

# 28. Definition of Done Assessment

## DoD 1 — 4-lane benchmark

```text
PASS
```

## DoD 2 — Summary JSON

```text
PASS
```

Artifact:

```text
experiments/results/m3_sweep_4lanes_summary.json
```

## DoD 3 — F1/F2

```text
PASS
```

## DoD 4 — Lane 3 & 4 triệt tiêu hoàn toàn negative speedup

```text
NOT SUPPORTED BY EVIDENCE
```

Đây là hypothesis / expected outcome, không phải lý do để làm invalid experiment.

PM quyết định giữ negative result.

---

# 29. Checklist Của Đại

- [x] Inventory source và experiment dependencies.
- [x] Freeze source SHA.
- [x] Mở rộng harness sang 4 lanes.
- [x] Add EAGLE Lane 4.
- [x] Pin EAGLE revision.
- [x] Validate Colab L4 runtime.
- [x] Run pilot.
- [x] Audit reusable cells.
- [x] Preserve failed attempts.
- [x] Diagnose missing speculative telemetry.
- [x] Capture raw SSE.
- [x] Capture Prometheus counters.
- [x] Reproduce on two fresh servers.
- [x] Open vLLM issue #65644.
- [x] Find upstream #55708.
- [x] Backport minimal fix.
- [x] Validate patch with predefined protocol.
- [x] Resume campaign.
- [x] Complete 72/72 cells.
- [x] Aggregate 11,520 measured requests.
- [x] Generate F1/F2.
- [x] Verify LOCAL == REMOTE.
- [x] Audit Lane 3 implementation behavior.
- [x] Verify EAGLE Lane 4 fixed K=3.
- [x] Report negative result.
- [x] PM approved closing M3-05.
- [x] Final technical worklog.
- [ ] Integrate Adaptive tree/controller into EAGLE — follow-up.
- [ ] Benchmark EAGLE fixed vs EAGLE Adaptive — follow-up.

---

# 30. Retrospective

## 30.1. Làm Đúng

1. Validator fail-closed khi telemetry thiếu.
2. Failed attempts được giữ.
3. Không cherry-pick.
4. Không rerun valid cells vô lý.
5. Bug được xác minh ở raw SSE.
6. Có independent reproduction.
7. Có upstream issue.
8. Có traceable upstream patch.
9. Patch được validate trước campaign resume.
10. Negative result được giữ.
11. Lane 4 fixed-K và future Adaptive-EAGLE được phân biệt rõ.

## 30.2. Cần Cải Thiện

1. Runtime dependency bug nên được qualification sớm hơn.
2. Hardware CostProfile phải thực sự đi vào adaptive decision nếu muốn claim Hardware-Aware mạnh.
3. Cần paired experiment cùng EAGLE checkpoint để cô lập Adaptive contribution.
4. Experiment ordering nên được freeze rõ hơn.
5. Worklog nên cập nhật song song thay vì cuối task.

---

# 31. Defense Notes

## 31.1. “Vì sao benchmark valid nhưng kết quả không đạt kỳ vọng?”

Vì:

```text
experiment validity
!=
hypothesis truth
```

Một experiment hợp lệ có thể bác bỏ hypothesis.

Nếu rerun cho tới khi số đẹp thì đó là cherry-picking.

---

## 31.2. “EAGLE tại sao c1 nhanh nhưng c32 chậm?”

Evidence cho thấy crossover theo concurrency.

M3-05 chưa đủ để kết luận nguyên nhân duy nhất.

Follow-up sẽ giữ cùng EAGLE model và chỉ thay:

```text
fixed K
→ Adaptive tree/controller
```

để kiểm tra causal contribution.

vLLM mô tả Dynamic Speculative Decoding như một cơ chế điều chỉnh số speculative tokens theo batch size vì verification cost có thể tăng đến mức làm TPOT xấu đi khi batch lớn.

---

## 31.3. “Lane 3 và Lane 4 có phải cùng Adaptive algorithm?”

Không.

Lane 3:

```text
0.5B draft
+
Adaptive controller
```

Lane 4:

```text
EAGLE3
+
fixed K=3
```

---

## 31.4. “Vậy đo contribution Adaptive như thế nào?”

So:

```text
EAGLE fixed K=3
vs
EAGLE + Adaptive tree
```

Giữ các yếu tố khác cố định.

---

# 32. Final Status

```text
M3-05:
DONE WITH VALID EVIDENCE
```

```text
72 / 72 cells VALID
```

```text
11,520 measured requests
```

Hypothesis cũ:

```text
“Lane 3 & Lane 4 eliminate all W3 negative speedup”
```

Kết luận:

```text
REJECTED / NOT SUPPORTED
```

PM decision:

```text
CLOSE M3-05
```

Next direction:

```text
Integrate Adaptive tree/controller into EAGLE Lane 4
→ benchmark fixed EAGLE vs adaptive EAGLE
→ quantify improvement
```

Không sửa canonical M3-05 summary bằng result của phase sau.

Phase sau phải có:

- experiment ID riêng;
- source SHA riêng;
- integration evidence riêng;
- protocol declaration trước run;
- raw evidence riêng;
- aggregate riêng;
- report riêng.

---

# 33. References

## 33.1. Repository

```text
experiments/results/m3_sweep_4lanes_summary.json
experiments/results/figures/m3/
experiments/runs/m3_01/
experiments/runs/m3_05_debug/

experiments/modal_lanes.py
experiments/benchmark_harness.py

src/proposer/controller.py
src/server/budget_planner.py
src/server/adaptive_scheduler.py
```

## 33.2. Upstream

vLLM issue:

```text
https://github.com/vllm-project/vllm/issues/65644
```

vLLM fix:

```text
https://github.com/vllm-project/vllm/pull/55708
```

Per-request speculative metrics:

```text
https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/
```

Dynamic Speculative Decoding:

```text
https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/
```

Các tài liệu chính thức của vLLM xác nhận:

- `metrics.speculative_decoding` chứa per-request speculative-decoding metrics.
- Các field gồm `num_spec_steps`, `num_draft_tokens`, `num_accepted_draft_tokens`, `mean_acceptance_length`, `draft_acceptance_rate` và `acceptance_histogram`.
- Các per-request counters có counterpart ở server-level Prometheus counters.
- Dynamic SD thay đổi K theo active batch-size range.
- K=0 nghĩa là không sinh draft tokens.
- Dynamic SD đã được upstream test với EAGLE, EAGLE3 và DFlash.

---

# 34. Closure Statement

M3-05 được đóng với vai trò một **evidence-producing benchmark task**.

Experiment không xác nhận kỳ vọng ban đầu rằng Adaptive và EAGLE sẽ triệt tiêu mọi negative speedup, nhưng nó đã tạo được một baseline 4-lane hoàn chỉnh, reproducible và traceable.

Giá trị quan trọng nhất của M3-05 là xác định rõ boundary condition:

```text
speculative decoding có thể tăng tốc ở low load
nhưng trở nên không profitable ở high load
```

đồng thời phát hiện và xử lý một blocker telemetry ở vLLM mà nếu bỏ qua sẽ làm acceptance metrics không còn đáng tin cậy.

M3-05 cũng xác minh một điểm kiến trúc quan trọng:

```text
Lane 3 Adaptive
=
0.5B draft
+
Adaptive controller

Lane 4 EAGLE
=
EAGLE3
+
fixed K=3
```

Do đó 72-cell campaign chưa phải phép thử cô lập contribution của Adaptive controller trên EAGLE.

Câu hỏi nghiên cứu tiếp theo đã được làm rõ:

```text
Adaptive tree/controller của nhóm
có cải thiện EAGLE fixed K=3 hay không,
và cải thiện từ bao nhiêu lên bao nhiêu?
```

Phase tiếp theo phải giữ EAGLE làm baseline cố định và chỉ thay Adaptive integration để đo contribution một cách controlled, reproducible và bảo vệ được trước hội đồng.
