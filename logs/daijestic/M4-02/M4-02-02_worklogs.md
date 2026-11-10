# M4-02-02 — Đo ma trận chi phí GPU và profiling phần cứng cho TreeAdaptive

## 0. Thông tin task

- **Task:** M4-02-02 — Đo ma trận chi phí thực nghiệm và sinh evidence phần cứng phục vụ TreeAdaptive v0.2.
- **Owner:** Bùi Thiên Đại / daijestic.
- **Project:** Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference on Consumer GPUs.
- **Branch:** `feat/m4-02-02-gpu-cost-profiler`.
- **Base ban đầu:** `develop` tại commit `f7253125361635d6e5184b76c39ea248f2323fc0`.
- **Benchmark source cuối cùng:** `de369da6f494c4ee2dd980a82b0936c96c88d6cc`.
- **Branch HEAD tại thời điểm đóng evidence:** `ead7c09330734ea953963500da07c3d6f4fe2e8b`.
- **Trạng thái cuối:** `DONE WITH EVIDENCE — PM Option A`.
- **Nguyên tắc đóng task:** Có evidence đến đâu, nghiệm thu đến đó; không tạo số liệu giả cho phần runtime chưa đo được.

---

# 1. Mục tiêu

Mục tiêu ban đầu của M4-02-02 là xây dựng evidence thực nghiệm về chi phí GPU cho cost model của TreeAdaptive v0.2.

Các biến quan trọng cần đo:

- GPU architecture;
- effective runtime batch size `B`;
- số speculative tokens tuyến tính `K`;
- context length;
- target model;
- EAGLE draft model;
- dtype / quantization;
- warm-up;
- số repeats;
- timing boundary;
- source Git SHA.

Task tập trung vào việc đo các primitive đang thực sự chạy trong runtime:

1. target model forward;
2. EAGLE proposer;
3. linear verification forward.

Các số đo này phục vụ hardware-aware cost modelling.

Task **không mặc nhiên chứng minh** rằng runtime đã thực thi arbitrary-tree T8/T16/T32 chỉ vì các topology đó tồn tại trong representation/controller code.

---

# 2. Research question

Câu hỏi kỹ thuật của task:

> Chi phí target forward, EAGLE proposal và linear verification thay đổi như thế nào theo effective runtime batch size trên NVIDIA L4 và Tesla T4?

Câu hỏi phụ:

- Batch size lớn đến đâu thì runtime vẫn tạo được exact requested batch?
- Cost có khác đáng kể giữa L4 và T4 không?
- Những primitive nào có thể dùng làm empirical evidence cho TreeAdaptive?
- Những phần nào vẫn chỉ là modelled/derived vì arbitrary-tree runtime chưa tồn tại?

Task này đo **component cost**.

Task này không trực tiếp kết luận:

```text
TreeAdaptive speedup = X
```

và không được dùng một component run để claim measured end-to-end arbitrary-tree latency.

---

# 3. Cơ sở lý thuyết

## 3.1 Speculative decoding

Speculative decoding dùng cơ chế:

```text
draft
→ target verify
→ accept prefix
→ phát nhiều token trong một decoding step
```

Trong trường hợp linear draft length `K`, nếu draft token có acceptance probability độc lập xấp xỉ `alpha`, expected emitted tokens có dạng:

```text
E[tau(K, alpha)]
=
1 + alpha + alpha^2 + ... + alpha^K
```

Đây là linear-chain special case.

---

## 3.2 OPT-Tree

OPT-Tree mở rộng draft từ một chuỗi tuyến tính sang một cây candidate.

Expected acceptance theo node-prefix probability:

```text
E[tau(T)]
=
1 + sum q(v)
```

với `q(v)` là xác suất prefix dẫn tới node `v` được accept.

OPT-Tree tối ưu cấu trúc cây theo node budget để tăng expected acceptance length.

Nguồn chính:

- TACL 2025:  
  https://aclanthology.org/2025.tacl-1.8/
- arXiv:  
  https://arxiv.org/abs/2406.17276
- official code:  
  https://github.com/Jikai0Wang/OPT-Tree

---

## 3.3 Project scalar-alpha approximation

TreeAdaptive spec của project không dùng trực tiếp vector node probability `q(v)` tại runtime.

Controller dùng scalar acceptance signal `alpha`.

Project đã định nghĩa approximation contract:

```text
P_step(l, alpha)
=
1 - (1 - alpha)^n_l
```

```text
P_path(l, alpha)
=
product P_step(k, alpha), k=1..l
```

```text
E[tau(T, alpha)]
=
1 + sum P_path(l, alpha)
```

Đây là **project approximation contract**, không phải exact OPT-Tree identity.

Task M4-02-02 không sửa contract này.

---

# 4. Audit runtime trước khi benchmark

Trước khi chạy benchmark, repository được kiểm tra để phân biệt:

```text
planned
represented
implemented
actually executed
```

Canonical topology code có:

- `parent_indices`;
- `position_ids`;
- `tree_mask`;
- `nodes_at_depth`;
- T0;
- T4;
- T8;
- T16;
- T32.

Tuy nhiên tại source base được audit, serving/proposer path chưa chứng minh có code thực sự lấy:

```text
T8 / T16 / T32
```

rồi truyền:

```text
parent_indices
tree mask
position ids
```

vào EAGLE proposal + tree verification runtime.

Vì vậy:

```text
runtime_tree_available = false
```

là deliberate scientific guard.

Không được gọi linear EAGLE K3 là “measured T8”.

Không được gọi component measurement là “measured C(T16,B)”.

---

# 5. Conflict verification-cost formula

Trong quá trình audit phát hiện một technical contract conflict.

## TreeAdaptive spec

Công thức trong:

```text
logs/Khoi/spec&plan/treeAdaptive/spec.md
```

là:

```text
C_V(T,B)
=
C_T(B)
*
(1 + gamma_attn * N/64)
```

---

## FlexCostModel fallback

Current implementation:

```text
src/flexsds/core/cost_model.py
```

dùng:

```text
C_V(T,B)
=
C_T(B)
*
(1 + attn_verify_factor * max(0, N-1))
```

Hai parameterization này không tương đương.

Ví dụ:

```text
N = 32

N/64 = 0.5

N-1 = 31
```

Không thể dùng cùng một numeric coefficient cho hai formula.

Ảnh hưởng có thể lan tới:

- predicted verification cost;
- predicted total tree cost;
- speedup;
- `alpha*`;
- topology selection.

Do đó từ đầu task đã đặt guard:

```text
gamma_attn_status
=
not_calibrated_pending_formula_alignment
```

Không fit gamma trước khi team align formula.

---

# 6. Thiết kế measurement pipeline ban đầu

Measurement pipeline được xây với yêu cầu:

1. đúng pinned model revision;
2. đúng GPU type;
3. synchronized CUDA timing;
4. raw evidence retention;
5. exact observed batch shape;
6. không auto-retry;
7. không cherry-pick;
8. environment provenance;
9. no canonical tree profile generation nếu runtime tree chưa tồn tại.

Artifacts chính:

```text
experiments/m4_cost_profiler.py
experiments/m4_modal_runner.py
experiments/m4_worker_observer.py
scripts/build_m4_cost_profiles.py
tests/test_m4_cost_profiler.py
```

---

# 7. Modal phase

## 7.1 Mục tiêu

Modal được chọn đầu tiên vì repo đã có infrastructure/model volumes và L4 environment.

Các run đầu chủ yếu để test:

- image setup;
- target checkpoint loading;
- EAGLE loading;
- observer injection;
- synchronized timer;
- artifact persistence.

---

## 7.2 Failed attempts

Một số attempt ban đầu fail trước khi có measurement hợp lệ.

Các failure gồm:

- Windows console encoding;
- Modal image build ordering;
- observer RPC serialization;
- deployment env/config issue.

Những run này được giữ lại nhưng được label:

```text
NO MEASUREMENT
```

Không dùng làm benchmark result.

---

## 7.3 Successful Modal one-cell smoke

Run:

```text
20261108-m40202-l4-one-cell-smoke-05
```

Source:

```text
11d0b412d18be2bedcb6c2073c949674d7a6d195
```

Cell:

```text
GPU = L4
B = 1
K = 1
context = 1024
warmup = 5
diagnostic repeats = 2
```

Result:

```text
PASS
```

Measured components:

```text
EAGLE K1 propose
target linear verification
```

Không có:

- baseline target-only cost;
- full B grid;
- K grid;
- tree cost;
- gamma.

Do đó kết luận:

```text
Modal one-cell smoke
=
infrastructure diagnostic only
```

---

# 8. Chuyển execution backend từ Modal sang Colab/Kaggle

Modal smoke chứng minh pipeline/timer hoạt động nhưng L4 allocation không đủ ổn định cho full controlled campaign.

Measurement core được tách khỏi Modal thành backend-independent runner:

```text
experiments/m4_cost_runner.py
```

Các artifact hỗ trợ:

```text
experiments/M4_COLAB.md
experiments/colab_m4_cost_profile.ipynb
experiments/requirements-m4-cuda.txt
```

Pinned stack:

```text
vLLM = 0.30.0
Torch = 2.13.0+cu130
CUDA = 13.0
nvidia-ml-py = pinned runtime package
```

Mục tiêu là giữ measurement semantics giống nhau dù backend thay đổi.

---

# 9. Hardware guard

Runner kiểm tra hardware trước model load.

Canonical pairs:

```text
L4
compute capability = (8,9)
sm_89
```

```text
T4
compute capability = (7,5)
sm_75
```

Nếu mismatch:

```text
canonical_hardware = false
```

runner giữ:

- `env.json`;
- `component_report.json`;
- `run_manifest.json`;
- hardware inspection;

sau đó stop trước model load.

Ví dụ local RTX 3050 Ti guard test đã reject đúng T4 request.

Đây chỉ là hardware-guard evidence, không phải benchmark.

---

# 10. Frozen measurement protocol cuối cùng

Final component protocol:

```text
context = 1024

B =
1,2,4,8,16,32

K =
0,1,2,3,4,5

warmup =
5

measured repeats =
50
```

`K=0`:

```text
target-only baseline
```

`K>=1`:

```text
eagle_propose
linear_verify_forward
```

---

# 11. Model revisions

Target:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

revision:

```text
b25037543e9394b818fdfca67ab2a00ecc7dd641
```

Draft:

```text
duckhoi/qwen2.5-7b-eagle3.1-speculator
```

revision:

```text
c3adca7e211b471d97a519a625df7ffd37b8b461
```

Method:

```text
eagle3
```

Sampling:

```text
greedy
```

---

# 12. GPU-specific runtime

## L4

```text
GPU = NVIDIA L4
SM = sm_89
dtype = bfloat16
quantization = awq_marlin
```

## T4

```text
GPU = Tesla T4
SM = sm_75
dtype = half
quantization = awq
```

Không scale số L4 thành T4 hoặc ngược lại.

---

# 13. Timing methodology

CUDA execution asynchronous nên wall-clock timer phải synchronize.

Timing:

```text
torch.cuda.synchronize()

start = perf_counter()

operation()

torch.cuda.synchronize()

stop = perf_counter()
```

Recorded timing method:

```text
worker_cuda_synchronize_perf_counter_wall_ms
```

Target timing boundary:

```text
model forward only
excluding logits/sampling
```

EAGLE boundary:

```text
actual propose call
including preparation + draft sampling
```

Do boundary khác nhau, không được cộng tùy tiện rồi gọi là full serving latency nếu chưa có contract tương ứng.

---

# 14. Full campaign attempt đầu tiên và bug scheduler

Full L4 attempt đầu tiên fail từ B=2.

Observer mong:

```text
B = 2
context lengths =
[1024, 1024]
```

nhưng thực tế thấy:

```text
[1025, 1024]
```

Một request đã decode trước request còn lại.

Điều này chứng minh:

```text
queued requests
!=
effective runtime batch
```

Nếu lấy latency đó làm `C(B=2)` thì experiment invalid.

---

# 15. Root cause scheduler/admission

Audit vLLM 0.30.0 source cho thấy individual request submission không đảm bảo tất cả requests được admission cùng iteration.

Old flow có thể là:

```text
enqueue A
scheduler chạy A
enqueue B
```

khi đó:

```text
A context = 1025
B context = 1024
```

Observer reject là đúng.

---

# 16. Scheduler fix

Fix source:

```text
de369da6f494c4ee2dd980a82b0936c96c88d6cc
```

Các thay đổi chính:

## 16.1 Fresh engine per cell

Mỗi:

```text
(K, B, context)
```

được chạy trong fresh process/engine.

Lý do:

- tránh cross-cell state;
- tránh provision toàn grid theo B lớn nhất;
- dễ cô lập failure;
- tránh poisoned CUDA context sau failure.

---

## 16.2 Per-cell sizing

```text
max_num_seqs = B
```

```text
max_num_batched_tokens
=
max(4096, B * context)
```

Giữ:

```text
gpu_memory_utilization = 0.85
```

---

## 16.3 Scheduling barrier

Flow:

```text
pause scheduler
→ enqueue đủ B requests
→ wake scheduling
→ drain
→ observer kiểm tra exact shape
```

---

## 16.4 Exact shape invariant

Một sample chỉ hợp lệ nếu:

```text
actual_batch_size = requested B
```

và:

```text
context = requested context
```

và query shape đúng K.

Không dùng partial batch thay thế.

---

## 16.5 Failure retention

Nếu cell fail:

- stop cell ở failure đầu;
- retain raw/log/env;
- continue later cells bằng fresh process;
- không auto-retry.

---

# 17. T4 minimal diagnostic

Trước khi dùng L4 credit, fix được test trên T4.

Run:

```text
t4-cell-diagnostic-v2-20261108-100821
```

Source:

```text
de369da6f494c4ee2dd980a82b0936c96c88d6cc
```

Grid:

```text
K = 1
B = 1,2
context = 1024
warmup = 2
repeats = 2
```

Result:

```text
process_returncode = 0
environment_status = complete
canonical_hardware = true
failed_rows = []
result = PASS
```

Evidence commit:

```text
4fd4709748ef9ba8958c2147ceb44c0a0dffb34b
```

Kết luận:

```text
scheduler/admission fix
PASS on T4 B1/B2
```

---

# 18. L4 minimal diagnostic

Sau T4 PASS, cùng diagnostic được chạy trên L4.

Evidence commit:

```text
ca2a4898af9f72d5e7a331b08166bb0b139f2bb0
```

Result:

```text
DIAG_PASS = true
failed_rows = []
```

Kết luận:

```text
scheduler/admission fix
PASS on L4 B1/B2
```

Full campaign chỉ được chạy sau diagnostic này.

---

# 19. Full L4 component campaign

Run:

```text
l4-components-v5-20261108-104613
```

Evidence commit:

```text
e1dc6fffa02239a230658542253475e07c73ba30
```

Source:

```text
de369da6f494c4ee2dd980a82b0936c96c88d6cc
```

Environment:

```text
GPU = NVIDIA L4
sm_89

Torch = 2.13.0+cu130
CUDA = 13.0
vLLM = 0.30.0

working_tree_clean = true
canonical_hardware = true
```

Grid:

```text
B =
1,2,4,8,16,32

K =
0,1,2,3,4,5

context =
1024

warmup =
5

repeats =
50
```

Result:

```text
status = complete
manifest_status = complete
canonical_component_run = true
full_repeat_requirement_met = true
```

Raw count:

```text
3630
```

Failed rows:

```text
0
```

---

# 20. Kiểm tra completeness L4

Expected records:

## K0

```text
6 batches
*
(5 warmups + 50 measured)
=
330
```

## K1..5

Mỗi K có:

```text
6 batches
*
55 attempts
*
2 measurement kinds
=
660
```

5 K:

```text
5 * 660
=
3300
```

Total:

```text
330 + 3300
=
3630
```

Observed:

```text
3630
```

=> completeness PASS.

---

# 21. L4 variance warnings

Summary có:

```text
66 rows
```

Trong đó:

```text
6 variance warnings
```

Warning criterion:

```text
(max - min) / median > 0.15
```

Các warning được giữ nguyên.

Không:

- delete sample;
- replace value;
- rerun tới khi đẹp;
- cherry-pick.

Một số local median giảm nhẹ khi B tăng.

Các dip có magnitude nhỏ.

Điều này được coi là:

```text
quality warning
```

không phải correctness failure.

GPU latency không bắt buộc strictly monotonic từng B vì:

- occupancy;
- kernel selection;
- fixed-cost amortization;
- scheduling behavior;
- memory regime.

---

# 22. Final L4 classification

```text
MEASURED — COMPLETE COMPONENT EVIDENCE
```

Có thể dùng cho:

- hardware comparison;
- component cost analysis;
- later calibration.

Không được gọi là:

```text
measured arbitrary-tree T16/T32 profile
```

---

# 23. Full T4 campaign

Run:

```text
t4-components-v3-20261108-103626
```

Evidence commit:

```text
ead7c09330734ea953963500da07c3d6f4fe2e8b
```

Source:

```text
de369da6f494c4ee2dd980a82b0936c96c88d6cc
```

Environment:

```text
GPU = Tesla T4
sm_75

dtype = half
quantization = awq

Torch = 2.13.0+cu130
CUDA = 13.0
vLLM = 0.30.0

working_tree_clean = true
canonical_hardware = true
```

Raw records:

```text
3090
```

Summary rows:

```text
66
```

Failed summary rows:

```text
10
```

---

# 24. T4 failed rows thực chất là bao nhiêu cells?

10 failed summary rows không đồng nghĩa 10 physical experiment cells.

Failed physical cells:

```text
K1 / B32
K2 / B32
K3 / B32
K4 / B32
K5 / B32
```

Mỗi speculative cell có:

```text
eagle_propose
linear_verify_forward
```

nên:

```text
5 cells
*
2 measurement kinds
=
10 summary rows
```

---

# 25. T4 cells đã đo được

Valid measured cells:

## K0

```text
B =
1,2,4,8,16,32
```

## K1..5

```text
B =
1,2,4,8,16
```

Các cells này có actual measurement.

---

# 26. T4 B32 speculative boundary

Failed cells:

```text
K1..5
B = 32
```

Error:

```text
No cycle with exact requested batch/context/query shape
```

Category:

```text
shape_mismatch
```

---

# 27. K1/B32 evidence

Requested engine config:

```text
B = 32
context = 1024

max_num_seqs = 32
max_num_batched_tokens = 32768
gpu_memory_utilization = 0.85
```

vLLM runtime report:

```text
GPU KV cache size
≈ 8,256 tokens
```

Observed batch cycles có:

```text
actual_batch_size = 7
```

và một số smaller batches.

Không có cycle:

```text
actual_batch_size = 32
```

thỏa exact measurement condition.

---

# 28. Đây có phải OOM không?

Allocator stats:

```text
num_ooms = 0
num_oom_rejections = 0
```

Vì vậy không classify thành direct CUDA OOM.

Đúng hơn:

```text
runtime/KV admission infeasibility
```

under frozen config.

---

# 29. Giải thích scientific boundary

`max_num_seqs = 32` nghĩa là:

```text
maximum allowed sequences per iteration
```

không nghĩa:

```text
runtime chắc chắn xử lý đủ 32 sequence cùng lúc
```

Scheduler vẫn bị giới hạn bởi:

- KV cache;
- running request capacity;
- token budget;
- model memory;
- speculative runtime overhead.

Do đó observer reject actual batch 7 thay vì 32 là đúng.

---

# 30. Final T4 conclusion

Đúng:

> Với frozen single-T4 configuration của experiment này, speculative `B=32`, `K=1..5` không thể được đo như một exact runtime batch 32.

Không đúng:

> Tesla T4 nói chung không hỗ trợ B32.

Câu thứ hai vượt quá evidence.

---

# 31. Treatment của T4 B32

Không:

```text
set 0
```

Không:

```text
set Infinity
```

Không:

```text
B16 * 2
```

Không:

```text
scale L4 → T4
```

Không:

```text
4 x microbatch B8
```

Không:

```text
2 GPUs T4
```

Không retry đến khi có số đẹp.

Final state:

```text
INFEASIBLE UNDER FROZEN SINGLE-T4 CONFIG
```

---

# 32. PM Decision cuối cùng

Sau technical review, PM chọn:

```text
Option A
```

Tên quyết định:

> Có bằng chứng đến đâu, nghiệm thu đến đó.

Nội dung:

1. công nhận toàn bộ L4 evidence;
2. công nhận T4 evidence đến speculative B16;
3. T4 B32 ghi thẳng là infeasible trên frozen config;
4. không ép dense profile;
5. đóng M4-02-02;
6. formula alignment và arbitrary-tree measurement chuyển follow-up.

---

# 33. Tác động của PM decision lên DoD

Trước PM decision, task từng bị:

```text
BLOCKED — TECHNICAL ALIGNMENT REQUIRED
```

vì:

- gamma formula conflict;
- arbitrary-tree runtime absent;
- canonical tree profile không thể generate khoa học.

Sau PM decision, các item đó không còn là blocker của current task.

Chúng trở thành:

```text
DEFERRED BY PM DECISION
```

---

# 34. Verification formula conflict — final handling

Conflict vẫn tồn tại:

## Formula A

TreeAdaptive spec:

```text
C_V
=
C_T
*
(1 + gamma_attn*N/64)
```

## Formula B

FlexCostModel:

```text
C_V
=
C_T
*
(1 + factor*max(0,N-1))
```

Không giải conflict trong M4-02-02.

Không fit gamma.

Không sửa cost model để chọn một formula tùy ý.

Final status:

```text
DEFERRED
TEAM TECHNICAL ALIGNMENT REQUIRED
```

---

# 35. Arbitrary-tree runtime — final handling

Canonical structures:

```text
T0
T4
T8
T16
T32
```

vẫn chỉ có representation/controller evidence.

Task này không chứng minh:

```text
runtime executes arbitrary tree T8
runtime executes arbitrary tree T16
runtime executes arbitrary tree T32
```

Final status:

```text
runtime_tree_available = false
```

---

# 36. Canonical profile generation

Current component reports intentionally contain:

```text
not_consumer_profile = true
```

và:

```text
canonical_profile_generated = false
```

M4-02-02 không generate:

```text
profile_sm89_l4.json
profile_sm75_t4.json
```

với arbitrary-tree values derived từ linear measurements.

Lý do:

```text
measured component
!=
measured arbitrary tree
```

---

# 37. Evidence summary

## Scheduler fix source

```text
de369da6f494c4ee2dd980a82b0936c96c88d6cc
```

## T4 diagnostic

```text
4fd4709748ef9ba8958c2147ceb44c0a0dffb34b
```

## L4 diagnostic

```text
ca2a4898af9f72d5e7a331b08166bb0b139f2bb0
```

## L4 full

```text
e1dc6fffa02239a230658542253475e07c73ba30
```

## T4 full

```text
ead7c09330734ea953963500da07c3d6f4fe2e8b
```

---

# 38. Raw artifacts

Evidence được lưu trong:

```text
experiments/runs/m4_cost_profiling/
```

Run directories chứa:

```text
env.json
run_manifest.json
component_report.json
raw.jsonl
per-cell raw.jsonl
worker logs
pip/environment evidence
```

Failed evidence được giữ nguyên.

Không overwrite run cũ.

Không xóa negative result.

---

# 39. Reproducibility information

Final evidence giữ được:

- GPU name;
- architecture;
- VRAM;
- driver;
- CUDA;
- PyTorch;
- vLLM;
- target model revision;
- draft model revision;
- dtype;
- quantization;
- context;
- B grid;
- K grid;
- warm-up;
- repeats;
- source SHA;
- timestamp;
- measurement boundary;
- timing method;
- engine sizing;
- runtime batch observation;
- failure category.

---

# 40. Scientific interpretation

## FACT

L4 full component campaign complete:

```text
B = 1,2,4,8,16,32
K = 0,1,2,3,4,5
```

with:

```text
3630 raw records
0 failed rows
```

---

## FACT

T4 speculative measurement complete through:

```text
B = 16
```

for:

```text
K = 1..5
```

K0 baseline also measured at B32.

---

## OBSERVATION

T4 speculative:

```text
B=32
K=1..5
```

không tạo exact runtime batch 32.

---

## INFERENCE

Under frozen single-T4 config:

```text
B32 speculative
=
infeasible exact-runtime-batch condition
```

---

## LIMITATION

Current evidence does not establish direct arbitrary-tree execution.

---

## DEFERRED

- gamma formula;
- gamma calibration;
- arbitrary-tree implementation;
- arbitrary-tree direct measurement;
- canonical tree profiles.

---

# 41. What this task proves

Task này chứng minh:

1. component GPU cost phụ thuộc hardware;
2. requested client/request count không đủ để chứng minh runtime batch;
3. scheduler admission cần được instrument/verified;
4. T4 và L4 có feasible region khác nhau;
5. batch lớn có thể không materialize trên GPU yếu hơn;
6. negative feasibility boundary là useful hardware-aware evidence;
7. raw component measurement phải được tách khỏi modelled tree cost.

---

# 42. What this task does NOT prove

Task này không chứng minh:

```text
TreeAdaptive end-to-end speedup
```

Không chứng minh:

```text
T16 trực tiếp nhanh hơn vanilla
```

Không chứng minh:

```text
gamma_attn đã đúng
```

Không chứng minh:

```text
arbitrary-tree verification đã chạy
```

Không chứng minh:

```text
T4 không bao giờ chạy B32
```

---

# 43. Deferred follow-up work

Các việc sau được tách sang task sau:

1. chọn canonical verification-cost formula;
2. align spec + implementation;
3. define coefficient semantics;
4. implement arbitrary-tree proposer/verifier;
5. trace tree mask/parents/positions;
6. benchmark T4/T8/T16/T32 trực tiếp;
7. fit hardware-specific verification model;
8. quantify model residual;
9. generate canonical consumer profile;
10. validate predicted speedup vs measured speedup.

---

# 44. Final DoD mapping

| Hạng mục | Trạng thái |
|---|---|
| Xây profiler component | DONE |
| GPU guard | DONE |
| CUDA synchronized timing | DONE |
| Scheduler exact-batch validation | DONE |
| Scheduler bug fix | DONE |
| T4 minimal diagnostic | DONE |
| L4 minimal diagnostic | DONE |
| L4 full component matrix | DONE |
| T4 component matrix B<=16 | DONE |
| T4 K0/B32 | DONE |
| T4 speculative B32 | INFEASIBLE BOUNDARY |
| Raw evidence | DONE |
| Provenance | DONE |
| Worklog | DONE |
| gamma_attn calibration | DEFERRED |
| Arbitrary-tree direct measurement | DEFERRED |
| Canonical tree profile | DEFERRED |

---

# 45. Final task status

```text
M4-02-02
DONE WITH EVIDENCE
```

Closure basis:

> Có bằng chứng đến đâu, nghiệm thu đến đó.

Accepted evidence:

```text
L4:
full component matrix complete

T4:
component measurements complete through speculative B16
target-only B32 complete
speculative B32 retained as an infeasible hardware/runtime boundary
```

Not claimed:

```text
measured T8/T16/T32 cost
gamma_attn
canonical arbitrary-tree profile
```

No additional GPU benchmark is required to close M4-02-02 under PM Option A.

---

# 46. Merge / handoff checklist

Trước commit closure:

```bash
git status
git diff --check
```

Chạy focused M4 tests.

Kiểm tra:

```text
no raw evidence deleted
no profile fabricated
no gamma added
no Trello modifications
```

Suggested closure commit:

```text
docs(m4-02-02): close GPU component profiling with L4/T4 evidence
```

PR target:

```text
develop
```

PR summary cần nói rõ:

```text
Measured:
- L4 full component grid
- T4 valid component grid

Boundary:
- T4 speculative B32 infeasible under frozen single-T4 config

Deferred:
- gamma/formula alignment
- arbitrary-tree runtime
- direct T8/T16/T32 benchmark
- canonical tree profiles
```

Sau khi review/merge readiness được chấp nhận, PM cập nhật Trello theo PM decision.

Trello không được thay đổi từ phía technical owner trong quá trình đóng task.