# M1-02 — Final Baseline Benchmark Worklog

## 1. Tổng quan

Worklog này ghi lại quá trình triển khai, vận hành, kiểm chứng và lưu trữ evidence cho phần việc của **Đại** trong M1-02.

Mục tiêu chính:

1. khóa môi trường thực nghiệm;
2. hoàn thiện automatic concurrency sweep;
3. chạy đầy đủ baseline **Vanilla vs Static Speculative Decoding** trên `W1`, `W2`, `W3`;
4. lưu raw telemetry có provenance;
5. validate theo fail-closed policy;
6. checkpoint evidence bền vững qua Git;
7. aggregate theo hierarchy `request → repeat → cell`;
8. sinh `summary.csv`;
9. giữ nguyên cả positive và negative result để phục vụ analysis, report và defense.

Đề tài:

> **Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference on Consumer GPUs**

---

# 2. Final Task Status

## Task 1

```text
[Đại] Chạy sweep benchmark đầy đủ 2 làn
(Vanilla & Static)
trên 3 tập workload W1, W2, W3

STATUS: DONE WITH EVIDENCE
```

## Task 2

```text
[Đại] Lưu trữ raw telemetry JSONL tại experiments/runs/
và trích xuất bảng tổng hợp summary.csv

STATUS: DONE WITH EVIDENCE
```

Canonical campaign:

```text
experiment_id =
m1-02-f307e33-20261004T102725Z

benchmark_commit =
f307e33f9aea4bbd45499f548480cb7d6f90a4b9

evidence_branch =
evidence/m1-02-f307e33

RUN_COMPLETE.status =
COMPLETE_VALID
```

Final coverage:

```text
6 sweeps
36 cells
180 measured repeats
5760 measured requests
36 summary rows
```

Warm-up traffic không được tính vào `5760 measured requests`.

---

# 3. Source of Truth và Frozen Environment

Scientific benchmark cuối được chạy từ frozen benchmark commit:

```text
f307e33f9aea4bbd45499f548480cb7d6f90a4b9
```

Canonical runtime:

```text
GPU                 = Tesla T4
CUDA                = 13.0
PyTorch             = 2.13.0+cu130
vLLM                = 0.30.0
aiohttp             = 3.13.5
environment_profile = m1-t4-cu130-v1
```

Target model:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

Static draft model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Static configuration:

```text
num_speculative_tokens = 3
```

Các validation notebook cũ có thể chứa historical revision hoặc runtime trước khi canonical campaign được freeze.

Final scientific evidence phải ưu tiên:

```text
actual benchmark commit
→ raw JSONL
→ done markers
→ server / harness logs
→ RUN_COMPLETE.json
→ summary.csv
→ worklog
```

Historical value như:

```text
vLLM 0.29.0
```

không phải canonical runtime của final campaign.

---

# 4. Luồng thực hiện Task

Luồng thực tế của M1-02:

```text
Requirement / Research Question
        ↓
Protocol decisions
        ↓
Freeze workload + runtime + metric definition
        ↓
Implement benchmark harness
        ↓
Smoke / validation tests
        ↓
Encounter operational failures
        ↓
Collect forensic evidence
        ↓
Evaluate engineering alternatives
        ↓
Build durable checkpoint architecture
        ↓
Run canonical 2-lane × 3-workload campaign
        ↓
Validate each complete sweep
        ↓
Checkpoint + commit + push durable evidence
        ↓
Resume remaining sweep after compute-session reset
        ↓
Validate all six sweeps
        ↓
Aggregate request → repeat → cell
        ↓
Generate summary.csv
        ↓
Validate final campaign
        ↓
RUN_COMPLETE = COMPLETE_VALID
        ↓
Analyze positive + negative results
        ↓
Report / Defense
```

Các nguyên tắc giữ xuyên suốt:

```text
planned work != implemented evidence

smoke test != scientific benchmark

diagnostic run != canonical measurement

partial raw != completed sweep

negative result != failed research
```

---

# 5. Problem / Incident Timeline

![Ảnh 1 — Problem Incident Timeline](<./Ảnh 1 — Problem  Incident Timeline.png>)

Canonical campaign là một workload dài:

```text
2 lanes
× 3 workloads
× 6 concurrency levels
× 5 repeats
× 32 requests
=
5760 measured requests
```

Trong quá trình triển khai, compute session có thể bị reset hoặc hết thời gian.

Local workspace:

```text
/kaggle/working
```

không được xem là durable scientific evidence nếu artifact chưa được push ra remote.

Vì vậy:

```text
partial local progress
!=
durable canonical evidence
```

Engineering requirement được rút ra:

```text
complete sweep
→ validate
→ hash
→ checkpoint
→ commit
→ push
→ verify remote
```

Checkpoint chỉ có giá trị nếu survive compute-session failure.

---

# 6. PM Decision Log & Canonical Protocol Freeze

![Ảnh 2 — PM Decision Log & Canonical Protocol Freeze](<./Ảnh 2 — PM Decision Log & Canonical Protocol Freeze.png>)

Các quyết định protocol quan trọng được freeze trước canonical campaign.

## 6.1 W2 Canonical Workload

Selected design:

```text
pipeline-aligned + diverse SQL patterns
```

Canonical package:

```text
dataset_version = w2.m1_03.v2
sample_count    = 12
max_tokens      = 96
```

Canonical file:

```text
experiments/configs/w2_v2/chat_requests.jsonl
```

Vanilla và Static phải dùng cùng frozen W2 package.

---

## 6.2 W3 Canonical Workload

Selected design:

```text
business-analysis-only
+ diverse scenarios
```

Canonical package:

```text
dataset_version = w3.m1_03.v2
sample_count    = 12
max_tokens      = 768
```

Canonical file:

```text
experiments/configs/w3_v2/chat_requests.jsonl
```

Generation cap `768` được chọn sau truncation diagnostics.

Observed diagnostic evidence:

```text
max_tokens = 192
→ 12 / 12 requests finish_reason = length

max_tokens = 512
→ 5 / 12 requests vẫn bị truncation

diagnostic max_tokens = 1024
→ longest observed natural completion = 681 tokens
```

Canonical choice:

```text
max_tokens = 768
```

Rationale:

```text
768 > 681 observed natural completion

và

768 < diagnostic cap 1024
```

Không được tự thay generation cap trong canonical benchmark.

---

## 6.3 Sample Reuse Policy

Canonical policy:

```text
cycle-rotate
```

Formula:

```text
repeat_offset =
((repeat_index - 1) * (num_requests % sample_count))
% sample_count
```

```text
sample_index =
(request_index + repeat_offset)
% sample_count
```

Mục đích:

- deterministic;
- reproducible;
- Vanilla và Static nhận cùng sample mapping;
- không dùng random sample selection độc lập theo lane.

---

## 6.4 Canonical vLLM Version

Historical material từng ghi:

```text
vLLM 0.29.0
```

Final canonical runtime:

```text
vLLM 0.30.0
```

Runtime contract:

```text
EXPECTED_VLLM = "0.30.0"
```

Nếu runtime không khớp:

```text
environment mismatch
→ benchmark abort
→ no scientific measurement
```

---

## 6.5 Primary Speedup Metric

Canonical definition:

```text
speedup_vs_vanilla =
vanilla_tpot_p50
/
static_tpot_p50
```

Interpretation:

```text
> 1.0
→ Static có TPOT thấp hơn
→ Static faster theo TPOT

= 1.0
→ equivalent

< 1.0
→ negative TPOT speedup
```

Không được đọc `speedup_vs_vanilla` như throughput ratio.

---

# 7. Engineering Alternatives & Durable Checkpoint Decision

![Ảnh 3 — Engineering Alternatives + Decision Matrix](<./Ảnh 3 — Engineering Alternatives + Decision Matrix.png>)

Sau incident mất local progress, ba engineering alternatives được cân nhắc.

## Option A — Local checkpoint only

```text
chỉ lưu /kaggle/working
```

Ưu điểm:

- implementation đơn giản;
- ít Git operations.

Nhược điểm:

- VM reset có thể xóa toàn bộ checkpoint;
- không có durable remote evidence;
- recovery cost lớn.

Decision:

```text
REJECT
```

---

## Option B — Push once after the whole campaign

```text
run all six sweeps
→ push cuối campaign
```

Ưu điểm:

- workflow đơn giản;
- ít evidence commits.

Nhược điểm:

- failure gần cuối có thể làm mất gần toàn campaign;
- durability có một single failure point.

Decision:

```text
NOT SELECTED
```

---

## Option C — Durable per-sweep checkpoint

```text
complete sweep
→ validate
→ SHA256
→ copy evidence
→ commit
→ push
→ verify remote
```

Ưu điểm:

- survive Kaggle reset;
- resume theo whole sweep;
- immutable remote evidence;
- audit provenance rõ;
- không ghép partial runs.

Nhược điểm:

- orchestration phức tạp hơn;
- cần second checkout;
- cần remote verification.

Decision:

```text
SELECTED
```

Các score trong infographic là engineering decision heuristic, không phải experimental measurement.

---

# 8. Two-Checkout Architecture

![Ảnh 4 — Two-Checkout Architecture](<./Ảnh 4 — Two-Checkout Architecture.png>)

Canonical campaign sử dụng hai checkout độc lập.

## 8.1 Frozen Benchmark Checkout

Path:

```text
/kaggle/working/do-an-ptit-d22
```

Branch:

```text
feat/m1-02-concurrency-sweep-baseline
```

Frozen commit:

```text
f307e33f9aea4bbd45499f548480cb7d6f90a4b9
```

Trong canonical campaign:

```text
NO source edit
NO benchmark commit
NO changing HEAD
```

Checkout này chịu trách nhiệm:

- chạy `benchmark_harness.py`;
- chạy lane launcher;
- đọc canonical workload;
- tạo temporary measurement output.

---

## 8.2 Durable Evidence Checkout

Path:

```text
/kaggle/working/do-an-ptit-d22-evidence
```

Branch:

```text
evidence/m1-02-f307e33
```

Checkout này nhận:

```text
raw JSONL
*.done.json
*.run_config.json
harness logs
server logs
artifact hashes
aggregate_results.log
summary.csv
RUN_COMPLETE.json
```

Allowed flow:

```text
benchmark sweep
→ validate exactly 960 measured records
→ SHA256
→ copy artifacts
→ evidence checkout
→ Git commit
→ Git push
→ verify remote SHA
```

Resume behavior:

```text
COMPLETE_VALID
→ SKIP

NOT_STARTED
→ RUN WHOLE SWEEP

INVALID
→ STOP
```

Không commit raw trực tiếp vào frozen benchmark checkout giữa campaign vì sẽ làm benchmark `HEAD` thay đổi giữa các measured sweeps.

---

# 9. PM Decision Traceability

![Ảnh 5 — PM Decision Traceability](<./Ảnh 5 — PM Decision Traceability.png>)

Nguyên tắc:

```text
Decision
→ Implementation
→ Runtime / Semantic Guard
→ Raw Evidence
→ Summary / Report
```

## vLLM

```text
Decision:
vLLM = 0.30.0

Implementation:
requirements-m1.txt
EXPECTED_VLLM

Guard:
runtime mismatch → abort

Raw:
vllm_version

Report:
environment provenance
```

## W2

```text
Decision:
W2 v2 pipeline-aligned workload

Implementation:
experiments/configs/w2_v2/

Raw:
dataset_version
workload_file_sha256
sample_id

Aggregation:
only canonical W2 evidence
```

## W3

```text
Decision:
business-analysis W3 v2
max_tokens = 768

Implementation:
experiments/configs/w3_v2/

Raw:
dataset_version
workload_file_sha256
generation provenance

Report:
all W3 conclusions conditioned on frozen W3 configuration
```

## Sample Reuse

```text
Decision:
cycle-rotate

Implementation:
sample_policy = cycle-rotate

Validator:
expected offset/index/id

Raw:
repeat_id
sample_index
sample_id
sample_offset
```

## Speedup

```text
Decision:
TPOT-based speedup

Implementation:
aggregate_results.py

Formula:
vanilla_tpot_p50 / static_tpot_p50

Summary:
speedup_vs_vanilla
```

Traceability requirement:

```text
WHAT was decided
→ WHERE it is implemented
→ HOW it is enforced
→ WHICH evidence proves it
```

---

# 10. Full Canonical Campaign

![Ảnh 6 — Full Canonical Campaign](<./Ảnh 6 — Full Canonical Campaign.png>)

Canonical campaign order:

```text
1. Vanilla / W1
2. Static  / W1
3. Vanilla / W2
4. Static  / W2
5. Vanilla / W3
6. Static  / W3
```

Mỗi lane/workload sweep sử dụng fresh server.

Before measurement:

```text
GPU/process cleanup
→ start vLLM server
→ /health
→ /v1/models
→ verify model identity
→ verify live launch configuration
→ final health gate
→ benchmark
```

Concurrency:

```text
[1, 2, 4, 8, 16, 32]
```

Mỗi cell:

```text
warm-up once
→ r001
→ r002
→ r003
→ r004
→ r005
```

Canonical warm-up:

```text
effective_warmup_count =
max(
    3,
    configured_warmup_requests,
    current_concurrency
)
```

Equivalent behavior:

```text
c1  → warm 3  @ c1
c2  → warm 3  @ c2
c4  → warm 4  @ c4
c8  → warm 8  @ c8
c16 → warm 16 @ c16
c32 → warm 32 @ c32
```

Mỗi repeat:

```text
32 measured requests
```

Per sweep:

```text
6 cells
× 5 repeats
× 32 requests
=
960 measured requests
```

Whole campaign:

```text
6 sweeps
× 960
=
5760 measured requests
```

Total:

```text
36 cells
180 measured repeats
5760 measured requests
```

Warm-up không được ghi vào measured raw count.

Prefix caching thuộc frozen server configuration và không được reset giữa cells/repeats trong cùng sweep.

Do đó benchmark này không được mô tả là cold-cache benchmark.

---

# 11. Sweep Validation Gate

![Ảnh 7 — Sweep Validation Gate](<./Ảnh 7 — Sweep Validation Gate.png>)

Một sweep chỉ được tính là complete khi có:

```text
6 concurrency
× 5 repeats
× 32 requests
=
960 measured records
```

## 11.1 Identity Validation

Kiểm tra:

```text
experiment_id
benchmark commit
lane
workload
concurrency
cell_id
repeat_id
request_id
```

`request_id` phải unique.

---

## 11.2 Request Validation

Measured record phải có:

```text
phase = measured
success = true
metrics_complete = true
```

Không silently drop failed requests.

---

## 11.3 Workload Provenance

Kiểm tra:

```text
dataset_version
workload_file_sha256
sample_count
sample_policy
sample offset
sample index
sample id
```

Sample mapping phải đúng `cycle-rotate`.

---

## 11.4 Runtime Provenance

Canonical values:

```text
GPU             = Tesla T4
CUDA            = 13.0
PyTorch         = 2.13.0+cu130
vLLM            = 0.30.0
aiohttp         = 3.13.5
```

---

## 11.5 Metric Contract

Measured evidence phải chứa metric hợp lệ:

```text
TTFT
TPOT
E2E
run_duration
input_tokens
output_tokens
VRAM
```

Static lane còn phải có speculative-decoding telemetry hợp lệ.

---

## 11.6 PASS / FAIL Policy

PASS:

```text
960 / 960 valid
→ SHA256
→ *.done.json
→ copy evidence
→ commit
→ push
→ remote verification
→ COMPLETE_VALID
```

FAIL:

```text
failed request
OR
incomplete metrics
OR
wrong count
OR
provenance mismatch
OR
invalid sample mapping

→ abort
```

Không được:

```text
aggregate survivors

append partial run

merge partial campaign attempts

silently delete outlier

rerun until numbers look good
```

---

# 12. Static/W3 Failure Investigation

Static/W3 từng fail trong canonical execution tại high concurrency.

## Attempt 1

Observed:

```text
c8 measured phase

requests         = 32
successes        = 31
metrics_complete = 31
```

Result:

```text
CELL ABORTED
SWEEP ABORTED
```

Root cause tại thời điểm đó:

```text
UNKNOWN
```

---

## Attempt 2

Observed:

```text
c8
→ PASS
```

Sau đó tại `c16` warm-up:

```text
attempted = 16
success   = 15

error_type =
ServerDisconnectedError
```

Result:

```text
warm-up abort
sweep abort
```

Forensic server evidence không cho thấy bằng chứng cho:

```text
GPU OOM
CUDA illegal memory access
fatal traceback
EngineCore crash
```

Server-side log cho thấy process vẫn hoạt động gần thời điểm failure và controlled shutdown xảy ra sau harness abort.

---

## Non-Canonical Diagnostics

Fresh-server diagnostic:

```text
Static/W3/c16
→ PASS
```

Fresh-server diagnostic:

```text
Static/W3/c32
→ PASS
```

Các diagnostics này chứng minh:

```text
c16/c32 có thể hoàn thành
trên ít nhất một fresh-server execution
```

Nhưng không chứng minh exact root cause.

Diagnostic raw:

```text
NOT canonical evidence
NOT included in summary.csv
```

Claim discipline:

```text
FACT:
canonical failures đã xảy ra

OBSERVATION:
fresh c16 và c32 có thể PASS

HYPOTHESIS:
transient transport / session / accumulated-history effect

UNKNOWN:
exact disconnect mechanism

UNSUPPORTED:
proven deterministic GPU OOM
proven hard c16/c32 capacity limit
proven EngineCore crash
```

Sau diagnostic, canonical protocol không bị thay đổi.

Static/W3 được chạy lại theo frozen protocol và cuối cùng đạt complete valid sweep.

---

# 13. Raw Evidence → Summary → Scientific Result

![Ảnh 8 — Raw Evidence Summary Scientific Result](<./Ảnh 8 — Raw Evidence  Summary  Scientific Result.png>)

Canonical evidence root:

```text
experiments/runs/m1_02/
└── m1-02-f307e33-20261004T102725Z/
```

Canonical raw files:

```text
vanilla_w1.jsonl
static_w1.jsonl

vanilla_w2.jsonl
static_w2.jsonl

vanilla_w3.jsonl
static_w3.jsonl
```

Mỗi raw:

```text
960 measured records
```

Total:

```text
6 × 960
=
5760 measured records
```

Mỗi sweep còn có:

```text
*.done.json
*.run_config.json
*.harness.log
*.server.log
```

Campaign artifacts:

```text
aggregate_results.log
RUN_COMPLETE.json
```

Final summary:

```text
experiments/results/summary.csv
```

---

# 14. Aggregation Contract

Canonical hierarchy:

```text
request
→ repeat
→ cell
```

Không pool trực tiếp tất cả request của năm repeats vào một distribution duy nhất.

## Repeat-Level Latency

Request observations trong một repeat được summarize trước:

```text
TTFT p50
TPOT p50
TPOT p95
E2E p50
```

Sau đó cell-level value được lấy từ repeat-level summaries.

---

## System Output Throughput

Per repeat:

```text
throughput =
sum(output_tokens)
/
run_duration_s
```

Cell throughput:

```text
median of repeat throughputs
```

---

## Static Acceptance Metrics

```text
alpha =
sum(num_accepted_draft_tokens)
/
sum(num_draft_tokens)
```

```text
tau =
1
+
sum(num_accepted_draft_tokens)
/
sum(num_spec_steps)
```

Không average request-level ratios.

---

## TPOT Speedup

```text
speedup_vs_vanilla =
vanilla_tpot_p50
/
static_tpot_p50
```

Interpretation:

```text
> 1.0
→ Static faster

= 1.0
→ equivalent

< 1.0
→ Static slower / negative speedup
```

---

# 15. Final Completion Marker

Final marker:

```text
experiments/runs/m1_02/
m1-02-f307e33-20261004T102725Z/
RUN_COMPLETE.json
```

Final status:

```text
status            = COMPLETE_VALID
sweeps            = 6
cells             = 36
measured_repeats  = 180
measured_requests = 5760
summary_rows      = 36
```

Final summary:

```text
experiments/results/summary.csv
```

Expected coverage:

```text
2 lanes
× 3 workloads
× 6 concurrency
=
36 rows
```

Each cell:

```text
n_runs = 5
```

---

# 16. Final Scientific Results

## 16.1 W1

TPOT speedup:

| Concurrency | Static speedup |
|---:|---:|
| 1 | 1.342x |
| 2 | 1.290x |
| 4 | 1.334x |
| 8 | 1.262x |
| 16 | 1.225x |
| 32 | 1.160x |

### OBSERVATION

Static có TPOT thấp hơn Vanilla trên toàn bộ measured concurrency set của W1.

Lợi ích giảm dần ở high concurrency nhưng vẫn:

```text
speedup > 1
```

tại `c32`.

Representative Static acceptance:

```text
alpha ≈ 0.916

tau ≈ 3.74–3.75
```

---

## 16.2 W2

TPOT speedup:

| Concurrency | Static speedup |
|---:|---:|
| 1 | 1.338x |
| 2 | 1.317x |
| 4 | 1.334x |
| 8 | 1.229x |
| 16 | 1.168x |
| 32 | 1.040x |

### OBSERVATION

Static có positive TPOT speedup trên W2 ở toàn bộ measured concurrency set.

Ở `c32`:

```text
speedup ≈ 1.040x
```

nên lợi ích đã tiến gần break-even.

Representative Static acceptance:

```text
alpha ≈ 0.80–0.81

tau ≈ 3.41–3.44
```

---

## 16.3 W3

TPOT speedup:

| Concurrency | Static speedup |
|---:|---:|
| 1 | 0.999x |
| 2 | 0.977x |
| 4 | 0.998x |
| 8 | 0.933x |
| 16 | 0.899x |
| 32 | 0.806x |

Representative Static acceptance:

```text
alpha ≈ 0.54–0.56

tau ≈ 2.62–2.67
```

### OBSERVATION

W3 gần break-even ở low concurrency nhưng negative TPOT speedup tăng rõ ở high concurrency.

Đặc biệt:

```text
W3/c16
=
0.899x
```

```text
W3/c32
=
0.806x
```

### INFERENCE

Trong frozen M1 baseline:

```text
speculative-decoding benefit
phụ thuộc mạnh vào workload characteristics
và concurrency
```

W1 và W2 không được dùng để suy ra rằng Static luôn nhanh hơn.

W3 cung cấp một empirical boundary condition quan trọng.

### CLAIM LIMIT

Kết quả trên không phải theoretical guarantee cho:

- GPU khác;
- target model khác;
- draft model khác;
- speculative token count khác;
- workload khác;
- runtime version khác;
- cache policy khác.

Negative result được giữ nguyên như scientific evidence.

---

# 17. Bonus — Performance Diagnosis

![Bonus 9 — Performance Diagnosis](<./Bonus 9 — Performance Diagnosis.png>)

Một vấn đề từng cần kiểm tra là Vanilla/W1/c1 có repeat runtime khoảng hơn 10 phút.

Representative observations:

```text
GPU:
Tesla T4

Model:
Qwen2.5-7B-Instruct-AWQ

Concurrency:
1

Requests/repeat:
32
```

Observed repeat durations:

```text
r001 ≈ 631.2 s
r002 ≈ 626.2 s
r003 ≈ 628.4 s
r004 ≈ 628.5 s
```

Representative request metrics:

```text
input_tokens median  ≈ 1523
output_tokens median ≈ 641.5

TTFT median ≈ 79.6 ms
TPOT median ≈ 30.18 ms/token
E2E median  ≈ 19.43 s
```

TPOT-derived decode rate:

```text
1000 ms
/
30.18 ms/token
≈
33.14 token/s
```

Measured system output throughput cũng xấp xỉ:

```text
33 token/s
```

### INFERENCE

TPOT-derived rate và observed throughput align với nhau.

Evidence nhất quán với:

```text
GPU/model decode throughput
```

là dominant runtime cost, thay vì Python benchmark harness overhead.

---

## client_queue_ms

Ở concurrency `1`:

```text
32 async tasks submitted
```

nhưng semaphore chỉ cho:

```text
1 active request
```

nên nhiều task phải đợi client-side queue.

Do đó:

```text
large client_queue_ms
!=
per-request model E2E latency
```

Backend warning hoặc attention fallback trên T4 cũng không tự động chứng minh benchmark invalid.

---

# 18. Evidence Inventory

## Vanilla / W1

```text
path:
experiments/runs/m1_02/
m1-02-f307e33-20261004T102725Z/
vanilla_w1.jsonl

measured_records:
960

SHA256:
806cde2769728da62e497aa57deb84a3de571cf7f714ffb0711ba196a71bbd57
```

## Static / W1

```text
path:
experiments/runs/m1_02/
m1-02-f307e33-20261004T102725Z/
static_w1.jsonl

measured_records:
960

SHA256:
e21c245f4f7a93e8083c274e582e650fbbe45f5897fe8bae121a178720cdf0e4
```

## Vanilla / W2

```text
path:
experiments/runs/m1_02/
m1-02-f307e33-20261004T102725Z/
vanilla_w2.jsonl

measured_records:
960

SHA256:
d854d57af0f04eabdf1bbec53ae11c16a33af732cf8521e8ec150a3ebfe3fbd1
```

## Static / W2

```text
path:
experiments/runs/m1_02/
m1-02-f307e33-20261004T102725Z/
static_w2.jsonl

measured_records:
960

SHA256:
b0fbf4c13dc485da2186cd5147226cfeb702389ec4f334e396c6f516c2989732
```

## Vanilla / W3

```text
path:
experiments/runs/m1_02/
m1-02-f307e33-20261004T102725Z/
vanilla_w3.jsonl

measured_records:
960

SHA256:
8f022ab999e8ad4997b3ffb6a57912c5fa890887aab1fd0ab9f78306119e2854
```

## Static / W3

```text
path:
experiments/runs/m1_02/
m1-02-f307e33-20261004T102725Z/
static_w3.jsonl

measured_records:
960

SHA256:
42e2b05e3242fb95692d7c1af9584422ecd56d316e66353081bc7e1bb8814e09
```

---

# 19. Summary Evidence

Canonical summary:

```text
experiments/results/summary.csv
```

Rows:

```text
36
```

SHA256:

```text
bf6eaa9c7ae730c7e0dc3086e0b3539851c6816d1f39317c789fefd7c8c3c2a7
```

Campaign marker:

```text
RUN_COMPLETE.status =
COMPLETE_VALID
```

---

# 20. Final M1-02 Operational Flow

```text
PM / Protocol Decisions
        ↓
Frozen W1 / W2 / W3
        ↓
Frozen Runtime
        ↓
Frozen Benchmark SHA
        ↓
Kaggle Bootstrap
        ↓
Verify Environment
        ↓
Fresh Lane Server
        ↓
Health Gate
        ↓
Model Identity Gate
        ↓
Live Configuration Gate
        ↓
Concurrency Sweep
        ↓
Warm-up Once Per Cell
        ↓
5 Measured Repeats
        ↓
32 Requests / Repeat
        ↓
960 Records / Sweep
        ↓
Fail-Closed Validation
        ↓
SHA256
        ↓
Durable Evidence Checkout
        ↓
Git Commit
        ↓
Git Push
        ↓
Remote HEAD Verification
        ↓
Next Sweep
        ↓
6 / 6 Sweeps COMPLETE_VALID
        ↓
Request → Repeat → Cell Aggregation
        ↓
36-row summary.csv
        ↓
RUN_COMPLETE.json
        ↓
Positive + Negative Result Analysis
        ↓
Report / Defense
```

---

# 21. Definition of Done

Hai task của Đại được đánh giá:

```text
DONE WITH EVIDENCE
```

vì đã có đầy đủ:

- frozen benchmark revision;
- canonical runtime provenance;
- full Vanilla/Static × W1/W2/W3 campaign;
- concurrency `[1,2,4,8,16,32]`;
- `5` measured repeats/cell;
- `32` measured requests/repeat;
- `5760` measured requests;
- `6` canonical raw JSONL;
- mỗi raw có `960` measured records;
- per-sweep validation PASS;
- SHA256 cho raw evidence;
- durable remote checkpoint;
- `36` summary rows;
- `RUN_COMPLETE.status = COMPLETE_VALID`;
- positive result được giữ lại;
- negative result được giữ lại;
- không cherry-pick để làm đẹp số.

---

# 22. Trello / PM Boundary

Technical evidence trong repository không tự thay đổi PM-controlled state.

Worklog này không tự sửa:

```text
Trello card status
checklist
labels
assignee
due date
scope
acceptance criteria
```

Trello vẫn do PM quản lý.

---

# 23. Defense Summary

Nếu cần trình bày M1-02 ngắn gọn:

> M1-02 bắt đầu từ việc freeze workload, runtime và measurement contract. Khi long-running benchmark gặp compute-session failure, campaign được thiết kế lại theo hướng frozen benchmark checkout kết hợp durable evidence checkout. Mỗi lane/workload sweep chỉ được checkpoint sau khi đủ 960 measured records và pass toàn bộ semantic/provenance validation. Evidence sau đó được commit, push và verify remote trước khi chạy sweep tiếp theo. Final campaign hoàn thành 6/6 sweeps, 36 cells, 180 measured repeats và 5760 measured requests. Aggregation giữ đúng hierarchy request → repeat → cell. Kết quả cho thấy Static speculative decoding cải thiện TPOT rõ trên W1 và W2, nhưng W3 gần break-even ở low concurrency và xuất hiện negative speedup ngày càng lớn ở high concurrency. Kết quả W3 được giữ nguyên như một empirical boundary condition thay vì bị loại bỏ hay tối ưu lại để tạo số đẹp.

M1-02 vì vậy cung cấp cả:

```text
implementation evidence
+
reproducibility evidence
+
scientific baseline
+
negative-result boundary condition
```

cho các milestone tiếp theo của đồ án.