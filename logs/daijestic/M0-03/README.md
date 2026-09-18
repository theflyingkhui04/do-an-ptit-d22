# Worklog Kỹ Thuật: [M0-03] Thiết lập GPU T4 & Benchmark Harness ghi log JSONL

- **Mã công việc:** `[M0-03]` (Sprint 1 · M0 PoC Nền tảng)
- **Người thực hiện:** Đại (`daijestic`) — Dev + IT, MLOps & Hạ tầng
- **Thời gian thực hiện:** 17–18/09/2026
- **Môi trường thực thi chính:** Kaggle Notebook — NVIDIA Tesla T4 16GB
- **Branch:** `feat/m0-03-benchmark-harness`
- **Benchmark code revision:** `397f48eb9b928746c7cb2e8464e6bb9212f00754`
- **Trạng thái kỹ thuật:** PASS toàn bộ acceptance criteria của card M0-03
- **Trạng thái quy trình:** Chờ PR review / PM merge vào `main`

---

## 1. Mục tiêu công việc & Bối cảnh

Mục tiêu của M0-03 là xây dựng hạ tầng đo benchmark tối thiểu nhưng có khả năng tái lập cho các thí nghiệm inference ở các giai đoạn sau. Task cần trả lời hai câu hỏi chính:

1. **Hạ tầng GPU T4 có đủ ổn định để chạy vLLM + Qwen2.5-7B-Instruct-AWQ và phục vụ OpenAI-compatible API hay không?**
2. **Benchmark harness có thể gửi nhiều request đồng thời, đo các metric latency chính và ghi lại từng request dưới dạng JSONL hay không?**

Theo card Trello, các đầu ra chính gồm:

- Môi trường GPU T4 với `pynvml` và `aiohttp` sẵn sàng.
- Script trong `experiments/` gọi endpoint `/v1/chat/completions`.
- Đo `TTFT` (Time to First Token) và `TPOT` (Time per Output Token).
- Ghi mỗi request thành một dòng JSONL tại `experiments/runs/raw_test.jsonl`.
- Tạo `experiments/results/summary.csv` theo schema trong `docs/06`.
- Kiểm chứng harness chạy đồng thời nhiều request mà không timeout.

---

## 2. Môi trường thực nghiệm

Môi trường sau được ghi nhận từ runtime Kaggle dùng cho nghiệm thu M0-03.

| Thành phần | Giá trị thực tế |
| :--- | :--- |
| **GPU** | NVIDIA Tesla T4 |
| **VRAM** | 15,360 MiB |
| **Compute Capability** | 7.5 (`sm_75`) |
| **PyTorch runtime** | `2.13.0+cu130` |
| **CUDA runtime của PyTorch** | `13.0` |
| **vLLM** | `0.29.0` |
| **aiohttp** | `3.13.5` |
| **nvidia-ml-py** | `13.595.45` |
| **Target model** | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| **Số GPU dùng cho benchmark** | 1 GPU (`CUDA_VISIBLE_DEVICES=0`) |
| **OpenAI-compatible endpoint** | `http://127.0.0.1:8000/v1/chat/completions` |

### SOURCE CONFLICT — CUDA version

Card M0-03 mô tả mục tiêu môi trường là **CUDA 12.x**, nhưng runtime Kaggle dùng để nghiệm thu thực tế resolve thành:

```text
torch 2.13.0+cu130
torch.version.cuda = 13.0
vLLM = 0.29.0
```

Không tự sửa card hoặc protocol trong task này. Kết quả dưới đây được ghi đúng theo môi trường thực tế đã chạy. Trước các benchmark khoa học ở M1 cần pin lại version/runtime chính thức theo quyết định của PM và `docs/06`.

---

## 3. Khởi chạy vLLM server trên Tesla T4

### 3.1. Command thực tế đã sử dụng

Server được khởi chạy bằng `vllm serve` trên chính Kaggle runtime:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8000
```

Log server được lưu tại:

```text
/kaggle/working/vllm_server.log
```

### 3.2. Cấu hình vLLM thực tế được resolve

Trong run M0-03, các tham số bảo thủ từ handoff M0-01 **không được truyền explicit**. vLLM 0.29.0 tự resolve cấu hình và log cho thấy:

- `tensor_parallel_size = 1`
- quantization được nhận dạng theo AWQ / AutoAWQ
- dtype chạy ở FP16 trên T4
- `max_model_len = 32768`
- `gpu_memory_utilization = 0.92`
- FlashAttention-2 không dùng được trên `sm_75`; backend chuyển sang `TRITON_ATTN`
- CUDA Graph vẫn được bật
- model load chiếm khoảng `5.29 GiB`
- CUDA Graph chiếm khoảng `0.22 GiB`
- KV cache khả dụng khoảng `6.38 GiB`

### 3.3. Kiểm tra API

Sau khi server sẵn sàng:

```text
GET /health -> HTTP 200
```

Endpoint `/v1/models` trả về model `Qwen/Qwen2.5-7B-Instruct-AWQ`.

Một request thử tới `/v1/chat/completions` trả lời thành công và có usage token, xác nhận server có thể thực hiện inference trước khi chạy benchmark harness.

---

## 4. Benchmark Harness

File chính:

```text
experiments/benchmark_harness.py
```

Harness dùng:

- `aiohttp` để gọi HTTP bất đồng bộ.
- OpenAI-compatible endpoint `/v1/chat/completions`.
- `stream=true` để nhận streamed response.
- `asyncio.Semaphore` để giới hạn số request chạy đồng thời.
- `pynvml` để lấy thông tin GPU và theo dõi VRAM.
- JSONL để lưu raw record theo từng request.
- Git commit SHA và version runtime để tăng khả năng truy vết / reproducibility.

### 4.1. Luồng xử lý chính

```text
CLI/config
   ↓
tạo HTTP request
   ↓
chờ semaphore slot
   ↓
ghi timestamp bắt đầu
   ↓
POST /v1/chat/completions với stream=true
   ↓
đọc SSE stream
   ↓
ghi TTFT tại streamed choices event đầu tiên
   ↓
ghi các khoảng thời gian giữa các streamed outputs
   ↓
kết thúc stream
   ↓
tính E2E / TPOT
   ↓
lấy token usage + GPU/VRAM + environment metadata
   ↓
ghi 1 JSON object / 1 dòng vào raw_test.jsonl
```

### 4.2. Metric được ghi

Các metric chính gồm:

- `ttft_ms`
- `tpot_ms`
- `e2e_ms`
- `itl_ms`
- `input_tokens`
- `output_tokens`
- `run_vram_peak_mb`
- `http_status`
- `success`
- `metrics_complete`
- `gpu`
- `vllm_version`
- `commit`

### 4.3. Công thức TPOT

Với request có nhiều hơn một output token:

\[
TPOT = \frac{E2E - TTFT}{output\_tokens - 1}
\]

Nếu chỉ có một output token thì TPOT không được gán bằng 0 một cách giả tạo.

---

## 5. Kiểm tra mã nguồn trước khi benchmark

Trước khi chạy nghiệm thu trên Kaggle, script đã được kiểm tra cơ bản ở local:

```bash
python -m py_compile experiments/benchmark_harness.py
git diff --check
```

Kết quả:

```text
PASS
```

`.gitignore` cũng được kiểm tra để bảo đảm raw benchmark không bị commit:

```bash
git check-ignore -v experiments/runs/raw_test.jsonl
```

Raw log trong `experiments/runs/` được ignore, trong khi `experiments/results/summary.csv` vẫn được giữ lại trong repo.

---

## 6. Pin đúng revision trên Kaggle

Repo private được clone lên Kaggle, sau đó checkout đúng revision dùng cho benchmark:

```bash
git -C /kaggle/working/do-an-ptit-d22 checkout --detach \
  397f48eb9b928746c7cb2e8464e6bb9212f00754
```

Kiểm tra:

```bash
git -C /kaggle/working/do-an-ptit-d22 rev-parse HEAD
```

Kết quả:

```text
397f48eb9b928746c7cb2e8464e6bb9212f00754
```

Việc checkout detached HEAD ở đây là chủ động để bảo đảm code benchmark đang chạy đúng revision cần nghiệm thu.

---

## 7. Single-request Smoke Test

Trước khi chạy concurrent acceptance test, một request được chạy để xác nhận pipeline đo hoạt động end-to-end.

Kết quả:

```text
success          : True
metrics_complete : True
ttft_ms          : 151.240
tpot_ms          : 27.801
e2e_ms           : 429.246
input_tokens     : 45
output_tokens    : 11
gpu              : Tesla T4
vllm_version     : 0.29.0
commit           : 397f48eb9b928746c7cb2e8464e6bb9212f00754
run_vram_peak_mb : 14343.19
```

Kiểm tra thủ công công thức TPOT:

\[
\frac{429.246 - 151.240}{11 - 1}
= 27.8006
\approx 27.801 \text{ ms/token}
\]

Kết quả tính tay khớp với giá trị harness ghi lại.

---

## 8. Concurrent Acceptance Test

### 8.1. Command nghiệm thu

```bash
cd /kaggle/working/do-an-ptit-d22

python experiments/benchmark_harness.py \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --num-requests 10 \
  --concurrency 4 \
  --warmup-requests 3 \
  --max-tokens 32 \
  --lane vanilla \
  --workload smoke \
  --overwrite
```

### 8.2. Validation raw JSONL

Sau run, file:

```text
experiments/runs/raw_test.jsonl
```

có đúng 10 record.

Validation:

```text
rows             : 10
successes        : 10
metrics_complete : 10
HTTP 200         : 10
commit unique    : {'397f48eb9b928746c7cb2e8464e6bb9212f00754'}
concurrency      : {4}
GPU              : {'Tesla T4'}

M0-03 concurrent acceptance: PASS
```

### 8.3. Ý nghĩa kết quả

Run này xác nhận rằng:

- Harness có thể gửi nhiều request đồng thời.
- 10/10 request hoàn thành thành công.
- Không có timeout trong acceptance run.
- 10/10 request có đủ metric theo kiểm tra của harness.
- Mọi record đều ghi đúng GPU và đúng benchmark code revision.

**Lưu ý:** Đây là smoke/acceptance test của M0-03, **không phải kết quả benchmark khoa học của M1**. Không dùng các số latency của run này để kết luận speedup hoặc hiệu năng cuối cùng của đề tài.

---

## 9. `summary.csv`

File:

```text
experiments/results/summary.csv
```

được tạo với schema theo `docs/06`:

```csv
lane,workload,concurrency,gpu,n_runs,ttft_p50,tpot_p50,tpot_p95,e2e_p50,alpha,tau,vram_peak_mb,throughput,speedup_vs_vanilla,variance_flag,commit,timestamp
```

Trong M0-03, file này **chỉ là template/header**.

Harness hiện chưa aggregate 10 raw request thành một row thống kê trong `summary.csv`. Điều này phù hợp với scope của card M0-03, vốn yêu cầu tạo file mẫu theo schema chứ chưa yêu cầu pipeline phân tích kết quả hoàn chỉnh.

Không ghi thủ công kết quả smoke test vào `summary.csv` để tránh nhầm lẫn với benchmark chính thức ở M1.

---

## 10. Liên hệ với handoff từ M0-01

M0-01 khuyến nghị một cấu hình bảo thủ cho T4:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enforce-eager
```

M0-03 **không áp dụng nguyên xi cấu hình này**.

Trong M0-03, server được khởi chạy bằng `vllm serve` với model, host và port; vLLM 0.29.0 tự resolve các tham số còn lại. Log thực tế cho thấy:

| Thành phần | Khuyến nghị M0-01 | M0-03 thực tế |
| :--- | :--- | :--- |
| Quantization | `awq` explicit | vLLM tự nhận AWQ / AutoAWQ |
| dtype | `half` explicit | resolve FP16 |
| GPU memory utilization | `0.85` | `0.92` |
| max model length | `4096` | `32768` |
| Eager mode | bật | không bật explicit |
| CUDA Graph | tắt khi eager | vẫn bật |
| Attention backend | cần fallback trên T4 | `TRITON_ATTN` |
| Kết quả | cấu hình an toàn đề xuất | server chạy và 10/10 request PASS |

### Kết luận từ M0-03

Kết quả này cho thấy cấu hình trong M0-01 là một **cấu hình an toàn được đề xuất**, nhưng không nên diễn đạt là điều kiện bắt buộc duy nhất để vLLM 0.29.0 chạy trên T4 trong mọi trường hợp.

Trong môi trường M0-03, server vẫn khởi động và vượt acceptance test dù không bật `--enforce-eager`, dùng `gpu_memory_utilization=0.92` và `max_model_len=32768`.

Tuy nhiên, đây chỉ là một run nghiệm thu trên một môi trường cụ thể. Không suy rộng rằng mọi workload/concurrency đều an toàn với cấu hình này và không dùng kết quả này để phủ định nhu cầu kiểm soát VRAM ở M1.

---

## 11. Đánh giá Checklist M0-03

### Tiêu chí nghiệm thu trên Trello

- [x] **Môi trường GPU T4 và thư viện đo lường sẵn sàng**
  - Tesla T4 hoạt động.
  - `pynvml` / `nvidia-ml-py` hoạt động.
  - `aiohttp` hoạt động.
  - vLLM 0.29.0 phục vụ model thành công.

- [x] **Script gọi API chạy ổn định khi có nhiều request đồng thời**
  - Run nghiệm thu `10 requests`, `concurrency=4`.
  - `10/10` request success.
  - `10/10` HTTP 200.
  - Không timeout trong acceptance run.

- [x] **Ghi tối thiểu 10 dòng JSONL có đầy đủ metrics**
  - `raw_test.jsonl` có 10 record.
  - `metrics_complete = True` cho cả 10 record.
  - Commit SHA được ghi đúng trong cả 10 record.

- [x] **`.gitignore` bỏ qua raw logs nhưng giữ `summary.csv`**
  - `experiments/runs/raw_test.jsonl` không được commit.
  - `experiments/results/summary.csv` được giữ trong repo.

### Kết luận kỹ thuật

```text
M0-03 TECHNICAL ACCEPTANCE: PASS
```

---

## 12. Giới hạn và rủi ro còn lại

### 12.1. Run hiện tại chưa phải benchmark M1

Acceptance run chỉ dùng để xác nhận correctness của harness và khả năng concurrent request. Chưa đủ cơ sở để kết luận:

- speedup của speculative decoding,
- throughput ổn định,
- p50/p95 đáng tin cậy,
- ảnh hưởng của concurrency,
- ảnh hưởng của prefix cache,
- variance giữa nhiều repeat.

Các kết luận trên phải tuân theo protocol trong `docs/06`.

### 12.2. Prompt/workload của smoke test chưa đại diện workload nghiên cứu

Nếu lặp cùng prompt, prefix caching hoặc cache reuse có thể tác động tới latency. Đây chưa phải dataset/workload chính thức để so sánh các lane.

### 12.3. Generation config

Khi vLLM server khởi động, log cho thấy model generation config có thể override một số default sampling của vLLM.

Harness M0-03 truyền explicit:

```text
temperature = 0
seed = 42
```

nhưng trước M1 vẫn cần pin toàn bộ sampling/generation configuration theo protocol để bảo đảm so sánh các lane công bằng.

### 12.4. Version-sensitive semantics

Các metric streaming và speculative decoding có thể thay đổi theo version vLLM. Vì vậy mọi benchmark sau M0 cần tiếp tục ghi:

- vLLM version / commit
- model/checkpoint
- GPU
- sampling config
- code commit
- timestamp
- workload
- concurrency

---

## 13. Evidence & File bàn giao

### Code

```text
experiments/benchmark_harness.py
```

### Summary schema

```text
experiments/results/summary.csv
```

### Raw evidence

```text
experiments/runs/raw_test.jsonl
```

Raw evidence được `.gitignore` theo policy và không commit vào Git.

### Server log

```text
/kaggle/working/vllm_server.log
```

Server log là runtime evidence và không được dùng thay cho source code/config đã commit.

### Benchmark revision

```text
397f48eb9b928746c7cb2e8464e6bb9212f00754
```

---

## 14. Bàn giao & Bước tiếp theo

M0-03 đã hoàn thành về mặt kỹ thuật. Các bước còn lại thuộc quy trình hoàn tất task:

1. Commit worklog vào:

   ```text
   logs/daijestic/M0-03/README.md
   ```

2. Push branch:

   ```text
   feat/m0-03-benchmark-harness
   ```

3. Mở/cập nhật Pull Request vào `main`.

4. PM review và merge theo workflow của nhóm.

5. Chỉ chuyển card sang Done sau khi các yêu cầu Definition of Done chung trong `docs/04` đã được đáp ứng.

---

## 15. Tóm tắt kết quả

M0-03 đã xây dựng thành công benchmark harness bất đồng bộ cho OpenAI-compatible `/v1/chat/completions`, có khả năng đo TTFT, TPOT, E2E, ITL, token usage và VRAM, đồng thời ghi metadata môi trường và Git commit cho từng request.

Trên NVIDIA Tesla T4 với Qwen2.5-7B-Instruct-AWQ và vLLM 0.29.0, harness vượt acceptance test `10 requests / concurrency=4` với `10/10` success, `10/10` HTTP 200 và `10/10` record có đầy đủ metric. Kết quả này xác nhận hạ tầng đo M0 đã sẵn sàng để làm nền cho benchmark có kiểm soát ở M1, nhưng bản thân run này không được xem là kết quả hiệu năng khoa học của đề tài.
