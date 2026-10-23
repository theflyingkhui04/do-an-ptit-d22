# [M2-04] Worklog Đại — Three-Lane T4 Benchmark Campaign

> **Task:** `[M2-04] Mở rộng Benchmark Harness & Đo đạc Thực nghiệm Đối chứng 3 Làn`
> **Người thực hiện phần này:** Bùi Thiên Đại / `daijestic` — Lead 60%
> **Partner:** Dương Nguyễn Thế — 40%
> **Ngày hoàn tất phần execution/readiness của Đại:** 23/10/2026
> **Trạng thái phần Đại:** **DONE WITH EVIDENCE**
> **Trạng thái toàn M2-04:** **CHƯA DONE — đang chờ Dương tổng hợp số liệu, F1/F2 và đánh giá W3 gate**

---

## 0. Executive Summary

M2-04 thực hiện benchmark đối chứng ba làn inference trên Tesla T4:

1. `vanilla`
2. `static`
3. `adaptive`

trên:

- 3 workload: `W1`, `W2`, `W3`;
- 6 mức concurrency: `1, 2, 4, 8, 16, 32`;
- 5 measured repeats/cell;
- 32 measured requests/repeat;
- warm-up theo concurrency;
- tổng cộng **54 cells** và tối thiểu **270 measured repeats**.

Phần Đại đã hoàn thành:

- chuẩn bị orchestration cho benchmark ba làn;
- chạy đủ canonical matrix trên Tesla T4;
- lưu raw/log evidence theo từng cell;
- checkpoint/resume campaign qua nhiều Kaggle sessions;
- kiểm completeness và variance;
- xác nhận cuối cùng:
  - `18/18` triplets complete;
  - `54/54` cells complete;
  - `>=270` measured repeats;
  - `errors = []`;
  - `noisy_cells_rerun_required = []`;
  - variance threshold = `15%`.

**Không có kết luận khoa học về W3 speedup trong phần Đại.**

Final readiness record ghi rõ:

```text
status = COMPLETE_VALID
scope = DAI_EXECUTION_AND_NOISE_READINESS
w3_speedup_gate = NOT_EVALUATED_BY_DAI_READINESS
downstream_owner = DUONG_ANALYSIS_SUMMARY_PLOTS
```

Do đó:

> **Phần Đại của M2-04 đã DONE WITH EVIDENCE.
> Toàn bộ M2-04 chưa DONE cho đến khi Dương hoàn thành aggregation, m2_summary.csv, F1/F2 và đánh giá W3 gate.**

---

# 1. Scope và phân chia trách nhiệm

## 1.1. Phần Đại theo task

Phần Đại gồm ba trách nhiệm chính:

1. Nâng cấp/hoàn thiện benchmark workflow để hỗ trợ ba làn:
   - Vanilla
   - Static
   - Adaptive

2. Chạy toàn bộ ma trận:

```text
3 lanes
× 3 workloads
× 6 concurrency
= 54 cells
```

với:

```text
>=5 measured repeats/cell
```

trên Tesla T4.

3. Rà soát variance gate và chuẩn bị execution evidence/readiness cho Gate M2.

---

## 1.2. Phần Dương — explicit out-of-scope của Đại

Các đầu việc sau **không được Đại triển khai thay**:

- tổng hợp raw JSONL thành:
  `experiments/results/m2_summary.csv`;
- tổng hợp median/min-max cuối cùng;
- hoàn thiện acceptance metrics phục vụ bảng kết quả;
- sinh F1:
  Throughput / TPOT ba làn;
- sinh F2:
  Speedup vs Vanilla;
- đánh giá scientific gate:
  `Adaptive W3 >= 1.00x`;
- viết phần phân tích kết quả cho Chương 5.

Raw evidence và readiness từ phần Đại là input bàn giao cho Dương.

---

# 2. Bối cảnh bắt đầu M2-04

M2-04 được bắt đầu sau khi M2-03 bị PM defer/block.

M2-03 không được dùng làm lý do thay đổi workload, seed, cache policy hoặc protocol của M2-04.

Một worktree riêng được tạo để tránh mang dirty diagnostic state của M2-03 sang M2-04:

```text
/d/Document/Four-year/do_an/do-an-ptit-d22-m2-04
```

Branch:

```text
feat/m2-04-three-lane-benchmark
```

Base ban đầu:

```text
origin/develop
82aa55569e381e938a25475c279f1acd13408433
```

M2-03 tiếp tục nằm riêng trong worktree:

```text
do-an-ptit-d22
branch: fix/m2-03-lossless-clean
```

Không cherry-pick diagnostic M2-03 vào M2-04.

---

# 3. Source-of-truth và quyết định protocol

## 3.1. Hardware

Canonical M2-04 được giữ trên:

```text
Tesla T4
```

Modal/L4 là downstream work/campaign khác, không thay hardware của canonical M2-04.

---

## 3.2. Variance conflict và quyết định PM

Trong quá trình readiness review phát hiện conflict:

### Source A — `docs/06-ke-hoach-thuc-nghiem.md`

Noisy cell khi:

```text
(max repeat-level TPOT p50 - min repeat-level TPOT p50)
-------------------------------------------------------
             median repeat-level TPOT p50

> 0.15
```

### Source B — `experiments/aggregate_results.py`

Implementation cũng đang dùng:

```python
variance_flag = variance_ratio > 0.15
```

### Source C — wording trên checklist M2-04

Có wording `<10%`.

Đại không tự chọn threshold.

PM đã được hỏi và xác nhận phương án:

```text
GIỮ 15%
```

Do đó frozen contract cho M2-04 là:

```text
variance_ratio =
(max repeat TPOT p50 - min repeat TPOT p50)
/
median repeat TPOT p50
```

và:

```text
NOISY iff variance_ratio > 0.15
```

**Exactly 15% không bị coi là noisy.**

Historical/noisy evidence không được xóa hoặc ghi đè.

---

# 4. Frozen execution contract

## 4.1. Reviewed execution SHA

Code canonical được khóa tại:

```text
6424c6ab309749f6159250fd3fd74d3dafb2cdd2
```

Commit:

```text
feat(m2-04): add three-lane T4 campaign orchestration
```

Branch phát triển:

```text
feat/m2-04-three-lane-benchmark
```

Execution trên Kaggle dùng exact SHA bằng detached HEAD.

Không chạy theo moving branch HEAD.

---

## 4.2. Runtime

Canonical runtime đã được verify trực tiếp trên Kaggle T4:

```text
GPU          = Tesla T4
PyTorch      = 2.13.0+cu130
CUDA runtime = 13.0
torchvision  = 0.28.0+cu130
vLLM         = 0.30.0
aiohttp      = 3.13.5
nvidia-ml-py = 13.595.45
```

`CUDA_VISIBLE_DEVICES=0`.

Kaggle có thể cấp nhiều T4 nhưng campaign chỉ sử dụng **một Tesla T4** để giữ hardware contract giống baseline.

---

## 4.3. Canonical workload artifacts

Workload byte integrity được verify trước campaign.

### W1

```text
experiments/configs/w1/node1_chat_requests.jsonl

SHA256:
0de2387c4e6a8c39523bf869788a55ee7487c6a32318a949ce22e1f8f7b17360
```

### W2

```text
experiments/configs/w2_v2/chat_requests.jsonl

SHA256:
bc26be01a4f5a813a4f6e843e2533e2f56d2e663918be2bd0e7b231e6385685c
```

### W3

```text
experiments/configs/w3_v2/chat_requests.jsonl

SHA256:
57341b0fc8c2ebbdfcfc579a6343aea66b08ca11fe2bfed5bc53a2bd49f5d83b
```

---

## 4.4. CRLF issue

Trên Git-for-Windows đã từng phát hiện canonical JSONL bị checkout thành CRLF, làm byte-exact manifest hash mismatch.

Đây không phải lỗi workload.

Canonical Linux/Kaggle checkout được tạo với:

```bash
git -c core.autocrlf=false clone ...
```

Không:

- sửa manifest hash;
- normalize workload runtime;
- relax integrity validator.

Canonical campaign chỉ chạy khi workload bytes match committed manifest.

---

# 5. Benchmark matrix

## 5.1. Lanes

```text
vanilla
static
adaptive
```

Server identity:

```text
Vanilla  -> port 8001 -> qwen2.5-7b-vanilla
Static   -> port 8002 -> qwen2.5-7b-static-draft
Adaptive -> port 8003 -> qwen2.5-7b-adaptive
```

---

## 5.2. Workloads

```text
W1
W2
W3
```

---

## 5.3. Concurrency

```text
1
2
4
8
16
32
```

---

## 5.4. Cell count

```text
3 × 3 × 6 = 54 cells
```

---

## 5.5. Repeat contract

Mỗi cell:

```text
5 measured repeats
```

Toàn campaign:

```text
54 × 5 = 270 measured repeats
```

---

## 5.6. Requests per repeat

Runner canonical truyền explicit:

```text
--num-requests 32
```

Do đó:

```text
1 repeat = 32 measured requests
```

Mỗi cell:

```text
5 × 32 = 160 measured requests
```

Toàn campaign:

```text
54 × 160
= 8640 measured requests
```

Lưu ý:

`32 requests/repeat` là execution configuration được runner khóa cho campaign này.

Docs/task chủ yếu freeze `>=5 repeats`; docs không đưa ra một acceptance rule độc lập rằng mọi campaign bắt buộc phải là 32 requests/repeat.

Tuy nhiên sau khi canonical M2-04 bắt đầu với 32 requests/repeat, giá trị này được giữ nguyên cho toàn bộ campaign để đảm bảo reproducibility.

---

# 6. Warm-up discipline

Warm-up chạy một lần trước measured repeats của mỗi cell.

Rule:

```text
effective_warmup =
max(
    3,
    requested warmup,
    current concurrency
)
```

Runner truyền:

```text
--warmup-requests 3
```

Do đó:

| Concurrency | Warm-up requests/cell |
|---:|---:|
| 1 | 3 |
| 2 | 3 |
| 4 | 4 |
| 8 | 8 |
| 16 | 16 |
| 32 | 32 |

Tổng warm-up requests:

```text
(3 + 3 + 4 + 8 + 16 + 32)
× 3 lanes
× 3 workloads

= 594 warm-up requests
```

Measured + warm-up:

```text
8640 + 594 = 9234 requests
```

Không dùng warm-up làm measured evidence.

---

# 7. Interleaving strategy

Campaign không chạy:

```text
all Vanilla
→ all Static
→ all Adaptive
```

Mà chạy theo triplet:

```text
same workload + same concurrency

Vanilla
→ Static
→ Adaptive
```

Ví dụ:

```text
W1/c1:
  vanilla__W1__c1
  static__W1__c1
  adaptive__W1__c1

W1/c2:
  vanilla__W1__c2
  static__W1__c2
  adaptive__W1__c2
```

Mục tiêu:

- giữ paired comparison gần nhau về thời gian;
- giảm temporal drift;
- dùng cùng workload artifact;
- cùng concurrency;
- cùng sample policy;
- cùng experiment identity;
- cùng execution SHA.

Tổng cộng:

```text
18 triplets
```

với mỗi triplet chứa 3 cells.

---

# 8. Code thay đổi cho phần Đại

Commit canonical:

```text
6424c6ab309749f6159250fd3fd74d3dafb2cdd2
```

gồm 6 files.

---

## 8.1. `experiments/run_m2_campaign.py`

Thin orchestration wrapper.

Không rewrite benchmark harness.

Nhiệm vụ:

- tạo chính xác 54 cells;
- thứ tự:
  - W1 → W2 → W3;
  - c1 → c2 → c4 → c8 → c16 → c32;
  - Vanilla → Static → Adaptive;
- truyền:
  - `--repeats 5`;
  - `--num-requests 32`;
  - `--warmup-requests 3`;
  - `--sample-policy cycle-rotate`;
- fail nếu:
  - execution SHA sai;
  - checkout dirty;
  - workload integrity fail;
  - evidence directory bị reuse;
- không truyền:
  - `--overwrite`;
  - `--skip-env-check`.

Default của runner là dry-run.

GPU execution phải explicit.

---

## 8.2. `scripts/switch_lane.sh`

Thêm:

```bash
LANE_LOG_DIR
```

để server logs từng cell nằm trong evidence directory riêng.

Không thay đổi model/scheduler protocol.

---

## 8.3. `experiments/aggregate_results.py`

Thêm readiness mode:

```text
--check-m2
```

Mục tiêu của mode này chỉ là kiểm phần Đại:

- đúng 54 cells;
- không duplicate cells;
- `>=5 repeats`;
- variance đúng `>15%`;
- liệt kê noisy cells.

Mode này không:

- tạo final M2 summary;
- kết luận W3 gate;
- tạo F1/F2.

---

## 8.4. `experiments/M2_04_RUNBOOK.md`

Ghi execution contract:

- T4;
- exact SHA;
- LF workload;
- 54 cells;
- interleaving;
- evidence preservation;
- variance gate;
- handoff cho Dương.

---

## 8.5. Tests

Thêm:

```text
tests/test_m2_campaign.py
tests/test_m2_harness_contract.py
```

Kiểm:

- 54 unique cells;
- V → S → A ordering;
- same inputs across paired lanes;
- >=5 repeats;
- no overwrite;
- no skip-env-check;
- exact SHA/clean checkout enforcement;
- workload verification;
- warm-up;
- variance boundary;
- readiness checker không ghi final CSV.

---

# 9. CPU verification trước GPU

Codex implementation review chạy targeted tests:

```text
42 PASS
0 FAIL
```

Broader relevant subset trên Windows:

```text
64 PASS
1 FAIL
```

Failure duy nhất là canonical W1 artifact bị CRLF trong Windows checkout nên byte hash mismatch.

Không coi đây là implementation failure.

Sau khi test trên snapshot với byte-identical committed LF artifacts:

```text
65 PASS
0 FAIL
```

Ngoài ra:

```text
bash -n scripts/switch_lane.sh
PASS
```

và:

```text
git diff --check
PASS
```

---

# 10. Canonical campaign identity

Experiment ID:

```text
m2-04-6424c6a-t4-canonical-v1
```

Evidence branch:

```text
evidence/m2-04-6424c6a
```

Canonical evidence root:

```text
experiments/runs/m2_04/
└── m2-04-6424c6a-t4-canonical-v1/
```

Selected cell evidence:

```text
cells/
├── vanilla__W1__c1/
├── static__W1__c1/
├── adaptive__W1__c1/
├── ...
└── adaptive__W3__c32/
```

Mỗi cell giữ ít nhất:

```text
raw.jsonl
harness.log
switch.log
server/
```

Adaptive additionally có:

```text
adaptive_telemetry.jsonl
```

---

# 11. Checkpoint/resume trên Kaggle

Full campaign dài hơn giới hạn practical của một Kaggle session.

Do đó campaign được checkpoint theo:

```text
(workload, concurrency)
```

tức **triplet**, không checkpoint từng lane riêng.

Ví dụ:

```text
W1/c1
=
Vanilla + Static + Adaptive
```

Triplet chỉ được chọn làm canonical evidence khi cả ba lanes hoàn tất và readiness/noise check của triplet đạt yêu cầu.

Marker:

```text
triplets/W1__c1.json
triplets/W1__c2.json
...
triplets/W3__c32.json
```

Marker `COMPLETE_VALID` chứa:

- experiment ID;
- execution SHA;
- workload;
- concurrency;
- three cell IDs;
- SHA256 của từng raw JSONL;
- variance threshold;
- completion timestamp.

---

## 11.1. Durable checkpoint

Sau một triplet hợp lệ:

```text
run V/S/A
→ validate
→ copy selected evidence
→ create COMPLETE_VALID marker
→ git add
→ git commit
→ git push evidence branch
→ mới chuyển triplet tiếp theo
```

Do đó Kaggle session có thể dừng mà không mất các triplet đã checkpoint.

Lần chạy tiếp theo:

```text
clone evidence branch
→ restore COMPLETE_VALID
→ skip complete triplets
→ chỉ chạy phần còn thiếu
```

---

## 11.2. Important distinction

`experiments/run_m2_campaign.py` tại execution SHA không có một resume framework lớn.

Checkpoint/resume thực tế được thực hiện ở operational Kaggle notebook bằng cách:

- dùng exact command plan sinh từ frozen runner;
- không thay benchmark harness;
- không thay code execution SHA;
- không thay model/controller/workload;
- chỉ điều phối và persist evidence giữa các Kaggle sessions.

Do đó execution logic của từng cell vẫn đến từ frozen SHA:

```text
6424c6ab309749f6159250fd3fd74d3dafb2cdd2
```

---

# 12. Sự cố 1 — runtime dài

W1/c1 đặc biệt lâu.

Nguyên nhân:

Mỗi lane tại c1 gồm:

```text
3 warm-up requests
+
5 × 32 measured requests
=
163 requests
```

Một triplet W1/c1:

```text
3 lanes × 163
≈ 489 total requests
```

Ở concurrency 1 các requests gần như tuần tự.

Đây không được coi là harness hang chỉ từ wall-clock dài.

Khi concurrency tăng, wall-clock của triplets giảm đáng kể do requests được xử lý song song hơn.

Không giảm:

- repeat count;
- requests/repeat;
- workloads;
- output workload;
- protocol

để ép campaign nhanh hơn.

---

# 13. Sự cố 2 — W2/c1 missing server_metrics

Trong một Kaggle version đầu tiên, W2/c1 đã chạy đủ:

```text
Vanilla
→ Static
→ Adaptive
```

nhưng notebook chết ở bước post-run aggregation với:

```text
ValueError:
Static measured record missing server_metrics:
request_id='req-4b1d93eb-b8c6-4a15-8de3-d6561d72715c'
```

Trace nằm ở:

```text
aggregate_results.py
→ aggregate_static_acceptance()
→ _extract_spec_metrics()
```

## 13.1. Ý nghĩa

Đây không phải bằng chứng GPU cell crash.

GPU execution ba lanes đã chạy.

Failure xảy ra khi full aggregator cố reconstruct Static acceptance metrics và một request không có:

```text
server_metrics
```

Harness chỉ set field này khi final streamed usage chunk chứa:

```text
data["metrics"]
```

Không:

- giả metrics = 0;
- bỏ request;
- impute missing value;
- thay alpha/tau;
- chọn một phần run để PASS.

---

## 13.2. Cách xử lý

Không sửa frozen execution SHA.

Operational notebook được sửa để:

```text
validation failure
→ preserve attempt
→ không chọn làm canonical evidence
→ triplet vẫn MISSING
→ tiếp tục campaign
```

Sau đó W2/c1 được chạy lại nguyên triplet.

Canonical rerun W2/c1 cuối cùng:

```text
COMPLETE_VALID
```

và được chọn vào final 54-cell evidence.

Điều này đảm bảo không cherry-pick lane riêng lẻ.

---

# 14. Final triplet completeness

Final evidence branch có đủ 18 markers:

| Workload | c | Status | Completed (UTC) |
|---|---:|---|---|
| W1 | 1 | COMPLETE_VALID | 2026-10-23 03:17 |
| W1 | 2 | COMPLETE_VALID | 2026-10-23 04:38 |
| W1 | 4 | COMPLETE_VALID | 2026-10-23 05:29 |
| W1 | 8 | COMPLETE_VALID | 2026-10-23 06:03 |
| W1 | 16 | COMPLETE_VALID | 2026-10-23 06:31 |
| W1 | 32 | COMPLETE_VALID | 2026-10-23 06:58 |
| W2 | 1 | COMPLETE_VALID | 2026-10-23 08:07 |
| W2 | 2 | COMPLETE_VALID | 2026-10-23 08:23 |
| W2 | 4 | COMPLETE_VALID | 2026-10-23 08:32 |
| W2 | 8 | COMPLETE_VALID | 2026-10-23 08:40 |
| W2 | 16 | COMPLETE_VALID | 2026-10-23 08:46 |
| W2 | 32 | COMPLETE_VALID | 2026-10-23 08:52 |
| W3 | 1 | COMPLETE_VALID | 2026-10-23 10:49 |
| W3 | 2 | COMPLETE_VALID | 2026-10-23 11:55 |
| W3 | 4 | COMPLETE_VALID | 2026-10-23 12:33 |
| W3 | 8 | COMPLETE_VALID | 2026-10-23 12:59 |
| W3 | 16 | COMPLETE_VALID | 2026-10-23 13:18 |
| W3 | 32 | COMPLETE_VALID | 2026-10-23 13:37 |

Mỗi marker còn lưu SHA256 của ba `raw.jsonl`, cho phép đối chiếu raw evidence được Dương sử dụng sau này có đúng selected evidence hay không.

---

# 15. Final readiness evidence

Final readiness file:

```text
experiments/runs/m2_04/
m2-04-6424c6a-t4-canonical-v1/
M2_04_DAI_READINESS.json
```

Nội dung chính:

```json
{
  "status": "COMPLETE_VALID",
  "experiment_id": "m2-04-6424c6a-t4-canonical-v1",
  "execution_sha": "6424c6ab309749f6159250fd3fd74d3dafb2cdd2",
  "triplets": 18,
  "cells": 54,
  "minimum_measured_repeats": 270,
  "variance_threshold": 0.15,
  "scope": "DAI_EXECUTION_AND_NOISE_READINESS",
  "w3_speedup_gate": "NOT_EVALUATED_BY_DAI_READINESS",
  "downstream_owner": "DUONG_ANALYSIS_SUMMARY_PLOTS",
  "readiness_report": {
    "status": "PASS",
    "cell_count": 54,
    "errors": [],
    "noisy_cells_rerun_required": [],
    "variance_threshold": 0.15,
    "w3_speedup_gate": "REQUIRES_EMPIRICAL_ANALYSIS"
  }
}
```

Completion time:

```text
2026-10-23T13:37:48Z
```

tương đương khoảng:

```text
20:37:48 ICT (+07)
```

ngày 23/10/2026.

---

# 16. Kết quả phần Đại

## 16.1. Acceptance — execution completeness

```text
18/18 triplets
PASS
```

```text
54/54 canonical cells
PASS
```

```text
>=5 measured repeats/cell
PASS
```

```text
>=270 measured repeats
PASS
```

---

## 16.2. Variance readiness

Frozen threshold:

```text
variance_ratio > 0.15
```

Final result:

```text
errors = []
noisy_cells_rerun_required = []
```

Do đó:

```text
VARIANCE READINESS: PASS
```

---

## 16.3. Evidence integrity

Selected evidence có:

- exact execution SHA;
- experiment ID;
- raw JSONL;
- server log;
- harness log;
- switch log;
- Adaptive telemetry;
- per-triplet marker;
- raw SHA256;
- final readiness seal.

Không ghi đè selected historical evidence.

---

# 17. Điều chưa được kết luận

## 17.1. W3 speedup

Không được nói:

```text
Adaptive W3 gate PASS
```

vì phần Đại chưa aggregate matched W3 speedups.

Final readiness cố tình ghi:

```text
REQUIRES_EMPIRICAL_ANALYSIS
```

---

## 17.2. M2-04 overall

Không được nói:

```text
M2-04 DONE
```

chỉ từ readiness của Đại.

Definition of Done toàn task còn phụ thuộc:

- `m2_summary.csv`;
- F1/F2;
- W3 Adaptive speedup ở cả sáu concurrency;
- Chương 5 / scientific analysis.

---

# 18. Known downstream gaps cho Dương

## 18.1. Summary/aggregation

Dương cần đọc selected raw evidence tại:

```text
experiments/runs/m2_04/
m2-04-6424c6a-t4-canonical-v1/
cells/
```

và tạo:

```text
experiments/results/m2_summary.csv
```

theo paired lane/workload/concurrency provenance.

---

## 18.2. Adaptive alpha/tau

Tại execution SHA của Đại, existing aggregator tập trung acceptance aggregation của Static.

Phần Đại cố tình không mở rộng Adaptive alpha/tau vì thuộc downstream analysis scope.

Nếu M2 summary yêu cầu Adaptive acceptance metrics, cần dùng source phù hợp từ raw/native speculative counters và document formula/source rõ ràng.

Không dùng controller `alpha_ema` như thể nó là canonical empirical acceptance rate.

---

## 18.3. F1/F2

Dương chịu trách nhiệm sinh:

### F1

Three-lane:

```text
Vanilla
Static
Adaptive
```

theo:

- TPOT;
- throughput;
- concurrency;
- workload.

### F2

Speedup vs matched Vanilla.

Negative speedup phải được giữ nguyên.

Không cắt/ẩn các điểm dưới `1.00x`.

---

# 19. Handoff contract cho Dương

Dương nên lấy dữ liệu canonical từ:

```text
branch:
evidence/m2-04-6424c6a
```

Experiment:

```text
m2-04-6424c6a-t4-canonical-v1
```

Execution SHA:

```text
6424c6ab309749f6159250fd3fd74d3dafb2cdd2
```

Selected cells:

```text
experiments/runs/m2_04/
m2-04-6424c6a-t4-canonical-v1/
cells/
```

Final readiness:

```text
experiments/runs/m2_04/
m2-04-6424c6a-t4-canonical-v1/
M2_04_DAI_READINESS.json
```

Triplet manifests:

```text
experiments/runs/m2_04/
m2-04-6424c6a-t4-canonical-v1/
triplets/
```

---

## 19.1. Pairing rule

Mọi non-Vanilla cell phải so với đúng Vanilla:

```text
same workload
same concurrency
same canonical campaign
```

Ví dụ:

```text
adaptive__W3__c16
```

chỉ pair với:

```text
vanilla__W3__c16
```

Không pair với cell khác concurrency hoặc historical M1 measurement.

---

## 19.2. W3 gate

Scientific acceptance criterion cần Dương evaluate:

```text
speedup =
Vanilla TPOT p50 / Adaptive TPOT p50
```

cho:

```text
W3/c1
W3/c2
W3/c4
W3/c8
W3/c16
W3/c32
```

PASS chỉ khi:

```text
all six speedups >= 1.00
```

Nếu bất kỳ cell nào:

```text
<1.00
```

thì empirical gate:

```text
FAIL
```

Đó vẫn là kết quả khoa học hợp lệ.

Không:

- đổi controller;
- đổi workload;
- đổi requests;
- loại outlier tùy ý;
- chọn run đẹp hơn

sau khi đã nhìn kết quả để ép PASS.

---

# 20. Claim discipline

Các claim hiện được phép:

### FACT

Canonical execution SHA:

```text
6424c6ab309749f6159250fd3fd74d3dafb2cdd2
```

### FACT

M2-04 execution matrix của Đại:

```text
54/54 cells
```

### FACT

Measured repeats:

```text
>=270
```

### FACT

Final readiness:

```text
PASS
```

### FACT

Variance:

```text
0 noisy selected cells
```

### FACT

Hardware:

```text
Tesla T4
```

### UNKNOWN / NOT YET EVALUATED

```text
W3 Adaptive >=1.00x ở mọi concurrency
```

### UNKNOWN / NOT YET DELIVERED

```text
final m2_summary.csv
F1
F2
Chapter 5 conclusion
```

---

# 21. Trạng thái task tại thời điểm handoff

## Phần Đại

```text
DONE WITH EVIDENCE
```

Acceptance phần execution/readiness đã đạt.

Không cần chạy thêm canonical GPU benchmark nếu downstream validation không phát hiện evidence defect.

---

## Phần Dương

```text
PENDING
```

Cần:

1. validate raw provenance;
2. aggregate median/min-max;
3. tạo `m2_summary.csv`;
4. tính matched speedups;
5. evaluate W3 gate;
6. sinh F1/F2;
7. cung cấp scientific interpretation.

---

## M2-04 overall

```text
NOT DONE YET
```

Lý do:

```text
execution/readiness complete
≠
scientific analysis complete
```

---

# 22. Reproduction summary

Để tái lập execution contract:

```text
checkout exact SHA
6424c6ab309749f6159250fd3fd74d3dafb2cdd2

verify:
Tesla T4
torch 2.13.0+cu130
CUDA 13.0
vLLM 0.30.0
aiohttp 3.13.5
nvidia-ml-py 13.595.45

verify W1/W2/W3 byte hashes

run matrix:
3 lanes
× 3 workloads
× 6 concurrency

order per triplet:
Vanilla → Static → Adaptive

each cell:
warm-up max(3, concurrency)
5 measured repeats
32 measured requests/repeat

noise:
(max-min)/median > 0.15
```

Raw evidence must be preserved.

---

# 23. Final handoff statement

**Ngày 23/10/2026, phần execution/readiness của Đại cho M2-04 đã hoàn tất với canonical Tesla T4 evidence.**

Final evidence confirms:

```text
18/18 triplets
54/54 cells
>=270 measured repeats
0 validation errors
0 selected noisy cells
variance threshold = 15%
```

Không có claim rằng Adaptive đã đạt W3 speedup gate.

Bộ raw evidence được bàn giao nguyên trạng cho Dương để thực hiện aggregation và scientific analysis.

**Next owner:** Dương Nguyễn Thế
**Next stage:** `raw evidence → m2_summary.csv → F1/F2 → W3 gate → Chapter 5`

Không chạy thêm benchmark hoặc sửa protocol nếu chưa phát hiện evidence defect cụ thể.
