# M1-02 — Environment Lock & Automatic Concurrency Sweep

## 1. Tổng quan

Worklog này ghi lại evidence cho hai task kỹ thuật do **Bùi Thiên Đại** thực hiện trong M1-02:

1. **[Đại] Khóa cứng môi trường thực nghiệm chuẩn (CUDA 13, vLLM 0.29.0, torch) trên Kaggle/Colab T4**
2. **[Đại] Nâng cấp `benchmark_harness.py` hỗ trợ sweep tự động các mức concurrency `[1, 2, 4, 8, 16, 32]`**

Branch:

```text
feat/m1-02-concurrency-sweep-baseline
```

Latest validated implementation commit:

```text
0e7a5a521d6f4c929d958a9eb4f65ead6cb33ece
feat(m1-02): add automatic concurrency sweep
```

Hai task trên đã được kiểm tra runtime trên Kaggle Tesla T4.

---

# TASK 1 — Khóa cứng môi trường thực nghiệm chuẩn

## 2. Mục tiêu

Mục tiêu của task này là bảo đảm benchmark M1 chỉ chạy trên runtime phù hợp với environment contract đã chốt, tránh việc thay đổi GPU / CUDA / PyTorch / vLLM làm sai lệch benchmark.

Canonical environment:

```text
GPU              = Tesla T4
CUDA             = 13.0
PyTorch          = 2.13.0+cu130
vLLM             = 0.29.0
aiohttp          = 3.13.5
nvidia-ml-py     = 13.595.45
```

Environment requirements được lưu tại:

```text
experiments/requirements-m1.txt
```

và environment validation được thực hiện trực tiếp trong:

```text
experiments/benchmark_harness.py
```

vLLM khuyến nghị dùng environment được kiểm soát/fresh vì các CUDA kernels và PyTorch builds có thể không binary-compatible giữa các CUDA/PyTorch configuration khác nhau. Đây là lý do environment pinning được coi là một phần của experimental reproducibility, không chỉ là setup tiện lợi.

---

## 3. Environment contract trong harness

Harness thu thập runtime metadata:

```text
environment_profile
gpu
gpu_index
gpu_memory_total_mb
driver_version
torch_version
torch_cuda_version
vllm_version
aiohttp_version
commit
```

Canonical constants:

```python
ENVIRONMENT_PROFILE = "m1-t4-cu130-v1"

EXPECTED_GPU = "Tesla T4"
EXPECTED_VLLM = "0.29.0"
EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_TORCH_CUDA = "13.0"
```

Trước khi benchmark bắt đầu, harness gọi:

```text
environment_info()
→ validate_environment()
```

Nếu một trong các giá trị canonical không khớp, benchmark phải dừng thay vì tiếp tục đo trong environment khác.

Exit code cho environment contract failure:

```text
5
```

Invariant:

```text
environment mismatch
→ benchmark abort
→ no scientific measurement
```

---

## 4. Kaggle validation evidence

Environment validation được chạy trên Kaggle.

Observed:

```text
GPU               : Tesla T4
Driver            : 580.159.04
CUDA              : 13.0
Python            : 3.12.13
PyTorch           : 2.13.0+cu130
torch.version.cuda: 13.0
CUDA available    : True
Compute capability: 7.5
vLLM              : 0.29.0
aiohttp           : 3.13.5
nvidia-ml-py      : 13.595.45
```

Notebook cũng xác nhận source revision được chạy là:

```text
0e7a5a521d6f4c929d958a9eb4f65ead6cb33ece
```

và:

```text
python -m py_compile experiments/benchmark_harness.py
```

PASS trước runtime benchmark.

---

## 5. Server validation

Target model:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

Vanilla server được khởi động bằng vLLM OpenAI-compatible server.

Server log xác nhận:

```text
vLLM version = 0.29.0
device_config = cuda
quantization = auto_awq
GPU = Tesla T4
```

vLLM engine load thành công và server sẵn sàng nhận request.

Các log về FlashAttention 2 không dùng được trên T4 không được coi là benchmark failure; runtime chuyển sang attention backend phù hợp và server vẫn start thành công.

---

## 6. Kết luận Task 1

### FACT

- Canonical M1 environment đã được định nghĩa.
- Required package versions đã được pin.
- Harness kiểm tra runtime environment trước benchmark.
- Kaggle Tesla T4 chạy đúng pinned stack.
- Source revision được ghi lại và trace được tới Git commit.

### Status

```text
[Đại] Khóa cứng môi trường thực nghiệm chuẩn
(CUDA 13, vLLM 0.29.0, torch) trên Kaggle/Colab T4

STATUS: DONE WITH EVIDENCE
```

---

# TASK 2 — Automatic Concurrency Sweep

## 7. Mục tiêu

Nâng cấp:

```text
experiments/benchmark_harness.py
```

từ single-concurrency runner thành automatic concurrency sweep:

```text
[1, 2, 4, 8, 16, 32]
```

với:

```text
>= 3 warm-up requests
>= 5 measured repeats / cell
```

Concurrency là một parameter của benchmark cell.

vLLM cũng có benchmark tooling riêng cho parameter sweeps, tức việc benchmark nhiều configurations trong cùng sweep là một mô hình benchmark hợp lệ.

---

## 8. Experiment hierarchy

Hierarchy hiện tại:

```text
experiment
└── cell = (lane, workload, concurrency)
    ├── warm-up
    ├── repeat r001
    ├── repeat r002
    ├── repeat r003
    ├── repeat r004
    └── repeat r005
        └── measured requests
```

Một automatic sweep sử dụng một `experiment_id` chung.

Ví dụ:

```text
experiment_id = exp-001

vanilla__smoke__c1
vanilla__smoke__c2
vanilla__smoke__c4
vanilla__smoke__c8
vanilla__smoke__c16
vanilla__smoke__c32
```

Concurrency thuộc **cell identity**, không thuộc experiment identity.

---

## 9. Automatic concurrency set

Harness định nghĩa:

```python
DEFAULT_CONCURRENCY_SWEEP = (
    1,
    2,
    4,
    8,
    16,
    32,
)
```

Nếu không truyền:

```text
--concurrency
```

hoặc:

```text
--concurrency-sweep
```

thì harness tự chạy canonical set trên.

Single-cell và partial-sweep mode vẫn tồn tại để smoke/debug:

```text
--concurrency 4
```

hoặc:

```text
--concurrency-sweep 1 2 4
```

nhưng harness in warning để các run này không bị hiểu nhầm thành complete canonical M1 sweep.

---

## 10. Repeat policy

Canonical minimum:

```python
CANONICAL_MIN_REPEATS = 5
```

Default:

```text
--repeats 5
```

Mỗi cell có repeat IDs:

```text
r001
r002
r003
r004
r005
```

Repeat được hiểu là measured replicate của cùng cell trong cùng server activation.

---

## 11. Warm-up policy

PM-approved policy:

```text
Option B — concurrency-matched warm-up
```

Formula:

```python
effective_warmup_count = max(
    3,
    configured_warmup_requests,
    current_cell_concurrency,
)
```

Warm-up chạy **một lần trước measured repeats của mỗi cell**.

Canonical behavior:

```text
c1  → warm 3  @ c1
c2  → warm 3  @ c2
c4  → warm 4  @ c4
c8  → warm 8  @ c8
c16 → warm 16 @ c16
c32 → warm 32 @ c32
```

Timeline một cell:

```text
enter cell
→ warm-up
→ r001
→ r002
→ r003
→ r004
→ r005
```

Không warm-up lại giữa repeats nếu server không restart.

---

## 12. Raw JSONL lifecycle

Measured output được quản lý ở experiment scope.

Khi dùng:

```text
--overwrite
```

harness xóa stale raw JSONL ngay khi experiment bắt đầu.

Flow:

```text
experiment start
→ remove stale raw file
→ warm-up cell
→ repeat 1 append
→ repeat 2 append
→ ...
→ next cell append
```

Warm-up records không được ghi vào measured JSONL.

Mục đích:

- không overwrite previous repeats;
- không trộn measured và warm-up data;
- không để stale file từ experiment cũ tồn tại nếu run mới fail trước measurement.

---

# 13. Runtime validation

## TEST A — Single cell, multiple repeats

Configuration:

```text
lane         = vanilla
workload     = smoke
concurrency  = 4
requests     = 8 / repeat
repeats      = 2
```

Expected:

```text
warm-up once
→ r001
→ r002
```

Observed:

```text
warmup_requests    = 4
warmup_concurrency = 4

r001 PASS
r002 PASS

CELL PASS
SWEEP PASS
```

Measured raw count:

```text
2 × 8 = 16 rows
```

Validated:

```text
one experiment_id
one cell_id
repeat IDs = r001, r002
phase = measured only
```

### Result

```text
PASS
```

---

## TEST B — Mini sweep

Configuration:

```text
concurrency = [1, 2, 4]
repeats     = 2
requests    = 8
```

Expected:

```text
3 cells × 2 repeats × 8 requests
= 48 measured rows
```

Observed hierarchy và count đúng expectation.

### Result

```text
PASS
```

---

## TEST C — Full automatic sweep

Không truyền explicit concurrency set.

Harness tự resolve:

```text
[1, 2, 4, 8, 16, 32]
```

Smoke configuration:

```text
6 cells
1 repeat / cell
32 requests / repeat
```

Expected:

```text
6 × 1 × 32
= 192 measured rows
```

Observed:

```text
SWEEP PASS

cells_completed = 6
concurrencies   = [1, 2, 4, 8, 16, 32]
```

### Result

```text
PASS
```

---

## TEST D — Full sweep with canonical repeat count

Full orchestration được chạy với:

```text
concurrencies    = [1, 2, 4, 8, 16, 32]
repeats_per_cell = 5
requests/repeat  = 32
```

Từng cell từ `c1` đến `c32` hoàn thành:

```text
r001
r002
r003
r004
r005
```

Final result:

```text
===== SWEEP PASS =====

cells_completed      : 6
concurrencies        : [1, 2, 4, 8, 16, 32]
repeats_per_cell     : 5
```

Implied measured requests:

```text
6 cells
× 5 repeats
× 32 requests
= 960 measured requests
```

Warm-up traffic không nằm trong 960 measured requests.

Notebook evidence xác nhận full sweep kết thúc thành công với cả sáu concurrency levels và 5 repeats/cell.

### Result

```text
PASS
```

---

# 14. Evidence classification

## FACT

- Canonical environment đã được lock và runtime validation PASS.
- Automatic concurrency set `[1,2,4,8,16,32]` đã được implement.
- One experiment chứa nhiều concurrency cells.
- Option B warm-up chạy một lần/cell.
- Repeat orchestration hoạt động.
- Raw JSONL append xuyên repeats/cells.
- Full automatic sweep PASS.
- Full sweep 5 repeats/cell PASS.

## OBSERVATION

Trong validation run, observed request throughput tăng khi concurrency tăng.

Observation này chỉ mô tả execution đã chạy.

## NOT YET SCIENTIFIC EVIDENCE

Không sử dụng các throughput/latency values trong validation notebook để kết luận:

```text
Vanilla W1 performance
Static performance
Vanilla vs Static speedup
negative speedup
M1 final baseline
```

Run có:

```text
--workload W1
```

nhưng nếu canonical W1 prompt/dataset chưa được truyền vào harness thì `W1` ở đây chỉ là identity label.

Không được coi label đó là bằng chứng đã benchmark canonical W1.

---

## Reproduction artifacts

### 1. Environment + warm-up validation

`logs/daijestic/M1-02/m1-02-environment-warmup-validation.ipynb`

Validated revision:

`11064f15cb8d52e6d58d3612c6a535a8e4428adf`

Evidence:
- Tesla T4 / CUDA 13.0 / PyTorch 2.13.0+cu130 / vLLM 0.29.0
- environment contract PASS
- Option B concurrency-matched warm-up PASS
- warm-up protocol floor PASS
- warm-up failure gate PASS
- Git SHA traceability PASS

### 2. Automatic concurrency sweep validation

`logs/daijestic/M1-02/m1-02-concurrency-sweep-validation.ipynb`

Validated revision:

`0e7a5a521d6f4c929d958a9eb4f65ead6cb33ece`

Evidence:
- exact Git SHA and pinned environment verification
- single-cell repeat orchestration PASS
- mini sweep `[1,2,4]` PASS
- automatic full sweep `[1,2,4,8,16,32]` PASS
- full sweep with 5 repeats/cell PASS

---

# 16. Task status

## Task 1

```text
[Đại] Khóa cứng môi trường thực nghiệm chuẩn
(CUDA 13, vLLM 0.29.0, torch) trên Kaggle/Colab T4

STATUS: DONE WITH EVIDENCE
```

## Task 2

```text
[Đại] Nâng cấp benchmark_harness.py hỗ trợ sweep tự động
các mức concurrency [1, 2, 4, 8, 16, 32]

STATUS: DONE WITH EVIDENCE
```

---

# 17. M1-02 card chưa hoàn thành

Hai task của Đại ở trên đã có implementation + runtime evidence.

Tuy nhiên **toàn bộ M1-02 chưa DONE**.

Các phần còn lại:

```text
canonical W1/W2/W3 integration
Vanilla scientific sweep
Static scientific sweep
lane interleaving / restart orchestration
raw-data validation
median aggregation
min-max aggregation
variance flag
summary.csv update
Vanilla vs Static comparison
report / defense evidence
```

Không mark toàn card M1-02 Done chỉ dựa trên hai increment này.

---

# 18. Handoff cho task tiếp theo

Harness hiện cung cấp foundation:

```text
controlled environment
→ experiment
→ concurrency cell
→ warm-up once
→ >=5 repeats
→ measured raw JSONL
```

Task downstream có thể dựa vào interface này để đưa canonical workload W1/W2/W3 vào benchmark.

Scientific comparison phải giữ tương đương giữa các lane:

```text
same workload
same prompt/dataset version
same concurrency
same requests/repeat
same repeat policy
same warm-up policy
same sampling config
same environment
same cache/restart protocol
```

Nếu một trong các condition trên thay đổi, phải ghi rõ trong experiment metadata/worklog trước khi so sánh kết quả.