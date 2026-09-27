# OpenSpec M1-01 — vLLM Dual-Lane Server & Telemetry Interface

> **Task:** M1-01 — Đóng gói Server vLLM 2 làn (Vanilla & Static Draft) & Chuẩn hóa Endpoint Telemetry  
> **Sprint:** Sprint 2 · M1 Baseline Benchmark  
> **Branch:** `feat/m1-01-vllm-dual-lane-server`  
> **Lead:** Cấn Đức Khôi (70%)  
> **Partner:** Dương Nguyễn Thế (30%)  
> **Trọng tâm tài liệu:** Phần việc của Dương — Telemetry Interface A + Smoke Test + Handoff cho Đại  
> **Ngày rà soát:** 26/09/2026  
> **Trạng thái:** Spec trước khi Dương bắt đầu apply  
> **Nguyên tắc:** Chưa tạo commit, chưa sửa Git trong bước lập đặc tả này.

---

# 0. Kết luận rà soát nhanh

Branch M1-01 hiện có head đã kiểm tra:

```text
737f2e8b3dc3a61ba1614d29b428dc91567a9627
```

So với `main`, branch đang ahead 1 commit và đã có:

```text
requirements-server.txt
scripts/launch_lane1_vanilla.sh
scripts/launch_lane2_static.sh
scripts/switch_lane.sh
scripts/test_lane1.py
scripts/test_lane2.py
```

Checklist Trello hiện tại:

```text
[x] requirements
[x] Khôi — launcher Vanilla
[x] Khôi — Static Draft k=3
[x] Khôi — switch lane / ports
[ ] Dương — telemetry JSON contract
[ ] Dương — smoke test 10 requests + VRAM leak + handoff Đại
[ ] Khôi — worklog + PR
```

Tức là phần còn thiếu để Dương hoàn thành contribution M1-01 là: telemetry contract, validator, smoke-test 10 request, VRAM-stability evidence và handoff artifact cho Đại.

---

# 1. Nguồn sự thật

## 1.1 Trello M1-01

Dương có hai checklist item chính:

1. Chuẩn hóa payload telemetry JSON:
   - alpha / acceptance rate;
   - `mean_accept_len`;
   - `tpot_ms`;
   - `vram_peak_mb`;
   - bám Frozen Interface A.

2. Smoke test:
   - 10 requests;
   - kiểm tra ổn định;
   - đo VRAM / rò rỉ VRAM;
   - nghiệm thu và bàn giao cho Đại.

## 1.2 Frozen Interface A trong tài liệu dự án

`docs/04-quy-trinh-va-ke-hoach.md` khóa:

```text
Engine -> Measurement
acceptance_rate
tree_budgets
tpot_ms
vram_peak_mb
```

`docs/05-thiet-ke-he-thong.md` mở rộng canonical telemetry thành:

```text
request_id
lane
ttft_ms
tpot_ms
itl_ms
output_tokens
acceptance_rate
mean_accept_len
tree_budgets
vram_peak_mb
batch_size_avg
```

M1-01 SHALL không tạo schema thứ ba mâu thuẫn với hai tài liệu này.

## 1.3 M0-02 upstream

M0-02 đã xác nhận trên T4:

```text
target: Qwen/Qwen2.5-7B-Instruct-AWQ
draft:  Qwen/Qwen2.5-0.5B-Instruct
method: draft_model
k:      3
use_heterogeneous_vocab: true
draft_sample_method: greedy
vLLM:   0.29.0
```

M0-02 đã chứng minh Target + Draft chạy được trên T4 và đủ nền tảng để tiến sang M1.

## 1.4 M0-03 / M1-02 measurement contract

M0-03 đã đo:

```text
ttft_ms
tpot_ms
e2e_ms
itl_ms
output_tokens
VRAM via NVML
environment metadata
```

M1-02 hiện đã có logic:

```python
record["server_metrics"] = data.get("metrics")
```

trên streamed usage event.

Hệ quả: Interface A của M1-01 nên tương thích trực tiếp với `data["metrics"]`.

---

# 2. Ranh giới ownership

## 2.1 Khôi sở hữu

Khôi chịu trách nhiệm:

- launcher Vanilla;
- launcher Static Draft;
- target/draft configuration;
- port/server lifecycle;
- launcher/runtime configuration;
- review việc bật native `--per-request-spec-decode-metrics summary`;
- chỉ sửa server internals nếu native vLLM 0.29.0 path thực sự fail;
- worklog/PR cuối task.

## 2.2 Dương sở hữu

Dương chịu trách nhiệm:

- canonical telemetry contract;
- field names, type, range, null semantics;
- phân biệt server-owned vs harness-owned metrics;
- telemetry schema/validator;
- smoke client;
- test 10 request trên cả hai lane;
- VRAM stability/leak check;
- compact result summary;
- handoff cho Đại/M1-02;
- Interface B regression `/v1/chat/completions`.

## 2.3 Đại sở hữu sau handoff

Đại sở hữu:

- concurrency sweep;
- warm-up/repeats benchmark khoa học;
- TTFT/TPOT aggregation;
- `experiments/results/summary.csv`;
- performance comparison.

## 2.4 Dương không sở hữu

Dương SHALL NOT:

- sửa speculative algorithm để “tạo” acceptance metric;
- suy acceptance rate từ latency/output text;
- thay k=3;
- thay target/draft model;
- biến smoke thành benchmark sweep;
- sửa scheduler M1-02;
- claim speedup từ smoke.

---

# 2A. Git working rule cho Dương

Branch task đã tồn tại:

```text
feat/m1-01-vllm-dual-lane-server
```

Dương SHALL **tiếp tục trên branch M1-01 này**, không tạo một task branch M1-01 thứ hai.

Trước khi code:

```bash
git fetch origin
git switch feat/m1-01-vllm-dual-lane-server
git pull --ff-only origin feat/m1-01-vllm-dual-lane-server
git rev-parse HEAD
```

Nếu Khôi push commit mới trong lúc Dương làm, Dương phải sync/reconcile trước khi push; không force-push làm mất commit của Khôi.

---

# 3. Telemetry architecture — dùng native vLLM 0.29.0

M1-01 **SHALL ưu tiên telemetry native của vLLM 0.29.0**, không thiết kế custom response payload nếu chưa cần.

Static launcher SHALL bật:

```text
--per-request-spec-decode-metrics summary
```

Với request `n=1`, speculative decoding bật và streaming có:

```json
"stream_options": {"include_usage": true}
```

final usage chunk sẽ mang:

```text
metrics.speculative_decoding
```

## 3.1 Native speculative metrics — authoritative

Raw object dùng đúng shape vLLM:

```json
{
  "metrics": {
    "speculative_decoding": {
      "mean_acceptance_length": 2.5,
      "draft_acceptance_rate": 0.5,
      "acceptance_histogram": [1, 1, 1, 1],
      "num_spec_steps": 4,
      "num_accepted_draft_tokens": 6,
      "num_draft_tokens": 12,
      "num_spec_tokens": 3
    }
  }
}
```

Raw counters authoritative:

```text
num_spec_steps
num_accepted_draft_tokens
num_draft_tokens
num_spec_tokens
acceptance_histogram
```

Derived values:

```text
draft_acceptance_rate =
  num_accepted_draft_tokens / num_draft_tokens

mean_acceptance_length =
  1 + num_accepted_draft_tokens / num_spec_steps
```

`mean_acceptance_length` **bao gồm bonus token** theo convention của vLLM.

Với Static `k=3`:

```text
0 <= draft_acceptance_rate <= 1
1 <= mean_acceptance_length <= 4
num_spec_tokens == 3
```

## 3.2 Vanilla semantics

Vanilla không speculative decode.

Project-normalized view:

```text
speculative_decoding = null
alpha = null
tau = null
tree_budgets = null
```

Raw Vanilla MAY không có `metrics` hoặc `metrics.speculative_decoding=null`; normalizer coi hai trường hợp tương đương.

## 3.3 Metric ownership

```text
vLLM native:
  speculative-decoding acceptance counters/rates

Harness/client:
  ttft_ms
  tpot_ms
  itl_ms
  e2e_ms
  output_tokens

NVML:
  run_vram_peak_mb
```

Không bật thêm server timing metrics để thay thế measurement hiện có của M0-03/M1-02.

## 3.4 Project-normalized record

Dương chuẩn hóa native response thành:

```json
{
  "schema_version": "m1_request_telemetry.v1",
  "client_request_id": "req-local-001",
  "server_request_id": "chatcmpl-...",
  "lane": "static",
  "speculative_decoding": {
    "mean_acceptance_length": 2.5,
    "draft_acceptance_rate": 0.5,
    "acceptance_histogram": [1, 1, 1, 1],
    "num_spec_steps": 4,
    "num_accepted_draft_tokens": 6,
    "num_draft_tokens": 12,
    "num_spec_tokens": 3
  },
  "alpha": 0.5,
  "tau": 2.5,
  "tree_budgets": null,
  "client_metrics": {
    "ttft_ms": 151.2,
    "tpot_ms": 27.8,
    "e2e_ms": 429.2,
    "output_tokens": 11
  },
  "gpu_metrics": {
    "vram_peak_mb": 13820.0
  }
}
```

`tree_budgets=null` là đúng cho M1 Static `summary` mode; không fabricate `[3,3,...]`.

`client_metrics.tpot_ms` và `gpu_metrics.vram_peak_mb` được giữ trong cùng normalized handoff record để đáp ứng đầy đủ checklist Dương/Frozen Interface A, nhưng nguồn đo vẫn lần lượt là harness và NVML.

---

# 4. Mapping sang Frozen Interface A

Frozen docs dùng:

```text
acceptance_rate
mean_accept_len
tree_budgets
```

M1-01 khóa mapping:

```text
acceptance_rate / alpha
  = metrics.speculative_decoding.draft_acceptance_rate

mean_accept_len / tau
  = metrics.speculative_decoding.mean_acceptance_length

tree_budgets
  = null trong M1 Static summary mode
```

`tau` theo convention native vLLM và **bao gồm bonus token**.

Nếu cần draft-only mean phụ trợ:

```text
mean_accepted_draft_tokens =
  num_accepted_draft_tokens / num_spec_steps
  = tau - 1
```

---

# 5. Request identity

Handoff giữ riêng:

```text
client_request_id
server_request_id
```

`server_request_id` lấy từ response `id`.

Không yêu cầu custom `request_id` bên trong `metrics.speculative_decoding`.

---

# 6. Telemetry transport

Canonical M1:

```text
Static launcher:
  --per-request-spec-decode-metrics summary

Request:
  n=1
  stream=true
  stream_options.include_usage=true
```

Final usage SSE chunk được đọc bằng:

```python
data.get("metrics", {}).get("speculative_decoding")
```

M1-02 hiện đã giữ:

```python
record["server_metrics"] = data.get("metrics")
```

nên không cần đổi transport.

**Native flag là default path.** Chỉ xem xét custom server patch nếu pinned vLLM 0.29.0 không trả metrics dù config/request hợp lệ.

Prometheus `/metrics` SHOULD được chụp before/after smoke để cross-check aggregate speculative counters khi tiện.

---

# 7. Machine-readable normalized schema

Dương SHOULD tạo:

```text
experiments/configs/m1_request_telemetry.schema.json
experiments/configs/m1_request_telemetry.example.json
```

Schema validate **normalized project record**, không redefine native response.

Required:

```text
schema_version
client_request_id
server_request_id
lane
speculative_decoding
alpha
tau
tree_budgets
client_metrics
gpu_metrics
```

Static:

```text
alpha in [0,1]
tau in [1,4]
num_spec_tokens == 3
alpha == draft_acceptance_rate
tau == mean_acceptance_length
tree_budgets == null
client_metrics.tpot_ms >= 0
gpu_metrics.vram_peak_mb >= 0
```

Vanilla:

```text
speculative_decoding == null
alpha == null
tau == null
tree_budgets == null
client_metrics.tpot_ms >= 0
gpu_metrics.vram_peak_mb >= 0
```

---

# 8. Validator

Dương SHALL tạo:

```text
scripts/check_m1_telemetry.py
```

Checker validate:

- normalized JSON/schema;
- non-empty client/server request IDs;
- lane;
- Static native fields đầy đủ;
- `num_spec_tokens == 3`;
- histogram length `k+1`;
- `sum(histogram) == num_spec_steps`;
- recompute accepted draft tokens từ histogram;
- recompute alpha từ counters;
- recompute tau = `1 + accepted/spec_steps`;
- derived/native values match trong float tolerance;
- Vanilla null semantics;
- no NaN/Infinity;
- no negative timing/VRAM trong smoke record.

Exit non-zero khi fail.

---

# 9. Runtime capability requirement

Live acceptance SHALL pin:

```text
vLLM == 0.29.0
```

Dương verify bằng:

```text
vllm --version
```

Static launcher cần native flag:

```text
--per-request-spec-decode-metrics summary
```

Nếu flag không tồn tại, đó là runtime drift/blocker.

Known limitation phải ghi worklog:

- per-request speculative metrics là experimental;
- `mean_acceptance_length` bao gồm bonus token;
- vLLM 0.29.0 có upstream report về terminal EOS có thể làm `mean_acceptance_length` over-count ở bước cuối; do đó `tau` phải được ghi rõ là **native vLLM metric**, không diễn giải quá mức như số token thực tế commit tuyệt đối;
- raw counters/histogram phải được lưu để audit;
- telemetry mode (`summary`) là một phần runtime identity: không trộn Static benchmark runs bật/tắt flag trong cùng ma trận M1.


# 9A. Execution environment — VSCode local → T4 live → VSCode

## Local VSCode

Làm trước, không cần GPU:

```text
schema
normalizer
checker
SSE parser
fixture tests
summary builder
docs
```

## T4 live acceptance

Chạy trên Linux + NVIDIA Tesla T4 16GB (Kaggle/Colab), ưu tiên cùng environment M1/M1-02.

Chỉ live T4 dùng để kết luận:

```text
server startup
native spec metrics
10/10 lane stability
VRAM peak/stability
OOM
```

## Quay lại local

Sau live run, hoàn thiện compact summary/worklog trong VSCode.

---

# 10. Smoke-test protocol

## 10.1 Mục tiêu

Smoke test trả lời:

1. Hai lane serve ổn định?
2. Interface B OpenAI-compatible còn đúng?
3. Interface A telemetry đủ?
4. Static alpha/tau hợp lệ?
5. VRAM có vượt trần/OOM?
6. Có dấu hiệu VRAM leak?
7. Có đủ để bàn giao Đại?

Không dùng smoke để claim speedup.

## 10.2 Số request

Mỗi lane:

```text
3 warm-up + 10 measured
```

Tổng measured:

```text
10 Vanilla
10 Static
```

Spec hiểu checklist “10 requests” là minimum mỗi lane để hai lane đều được nghiệm thu.

## 10.3 Thứ tự chạy

Current `switch_lane.sh` gọi `stop_all()` trước khi bật lane mới, nên hai lane đang chạy luân phiên trên một T4.

```text
lane1 -> warmup -> 10 -> stop
lane2 -> warmup -> 10 -> stop
```

Không mô tả branch hiện tại như hai process simultaneous.

## 10.4 Request settings

```text
temperature = 0
seed = 42  # REQUIRED trên pinned vLLM 0.29.0
max_tokens >=64
stream = true
include_usage = true
concurrency = 1
```

Same prompt/settings ở hai lane.

---

# 11. VRAM stability / leak check

Ghi:

```text
vram_idle_before_warmup_mb
vram_idle_after_warmup_mb
vram_peak_measured_mb
vram_idle_after_smoke_mb
```

Baseline leak dùng post-warmup idle.

MUST:

```text
0 OOM
10/10 measured success
peak <= 14336 MiB
```

Sau cooldown >=5s:

```text
idle_delta_mb =
  vram_idle_after_smoke_mb - vram_idle_after_warmup_mb
```

Gate heuristic:

```text
idle_delta_mb <= 256 MiB
```

`256 MiB` là **project stability heuristic**, không phải bằng chứng toán học về leak. Nếu vượt ngưỡng phải chạy batch thứ hai và nhìn xu hướng monotonic trước khi kết luận suspected leak.

Ngưỡng Trello `VRAM <= 14GB` được operationalize trong M1-01 là:

```text
14 GiB = 14336 MiB
```

vì NVML / `nvidia-smi` báo bộ nhớ theo MiB.

Nếu vượt:

- chạy batch thứ hai;
- nếu baseline tiếp tục tăng monotonic -> `VRAM_STABILITY_FAIL`;
- trả blocker cho Khôi.

Dương không sửa allocator/server internals.

---

# 12. Raw smoke record

Raw JSONL:

```text
experiments/runs/m1_01_smoke/<lane>.jsonl
```

Mỗi row SHOULD có:

```text
smoke_run_id
request_index
lane
endpoint
model
client_request_id
server_request_id
http_status
success
ttft_ms
tpot_ms
e2e_ms
input_tokens
output_tokens
server_metrics
vram_before_mb
vram_peak_mb
vram_after_mb
gpu
vllm_version
commit
timestamp
```

`experiments/runs/` gitignored.

---

# 13. Tracked summary

Tạo:

```text
experiments/results/m1_01_smoke_summary.json
```

Tối thiểu:

```text
branch_commit
gpu
driver
torch_version
cuda_version
vllm_version

vanilla:
  measured_requests
  successes
  telemetry_contract_pass
  vram_peak_mb
  idle_delta_mb

static:
  measured_requests
  successes
  telemetry_contract_pass
  draft_acceptance_rate descriptive stats
  mean_acceptance_length descriptive stats
  num_spec_steps
  num_accepted_draft_tokens
  num_draft_tokens
  vram_peak_mb
  idle_delta_mb

handoff_ready
```

Không đưa raw responses nặng vào tracked summary.

---

# 14. Handoff cho Đại

Dương bàn giao:

```text
telemetry schema
example fixture
checker
smoke summary
raw path convention
field ownership
transport mode
served model names
ports
known limitations
```

Mapping cho M1-02:

Per-request raw:

```text
server_metrics.speculative_decoding.draft_acceptance_rate
server_metrics.speculative_decoding.mean_acceptance_length
```

Canonical **cell-level** aggregation SHALL dùng raw counters, không average ratio naïve:

```text
summary.alpha =
  sum(num_accepted_draft_tokens) / sum(num_draft_tokens)

summary.tau =
  1 + sum(num_accepted_draft_tokens) / sum(num_spec_steps)
```

Vanilla:

```text
summary.alpha = null
summary.tau = null
```

Timing/VRAM tiếp tục theo M1-02 measurement protocol:

```text
record.tpot_ms -> TPOT aggregation
record.run_vram_peak_mb -> VRAM aggregation
```

---

# 15. Interface B regression

Smoke MUST xác nhận:

```text
POST /v1/chat/completions
```

vẫn là standard OpenAI-compatible request.

Client không phải gửi:

```text
lane
speculative=true
draft_model
internal telemetry flags
```

Lane chỉ cấu hình server-side.

---

# 16. Runtime pinning

Current `requirements-server.txt` chưa pin và chưa có NVML dependency rõ:

```text
vllm
huggingface_hub
requests
```

Đây là gap trước live acceptance.

Trong khi M0-02/M0-03/M1-02 đang dùng:

```text
vLLM 0.29.0
PyTorch 2.13.0+cu130
CUDA 13.0
```

Dương SHALL:

- ghi exact runtime versions vào smoke summary;
- không nghiệm thu trên “latest vLLM” không xác định;
- báo Khôi nếu drift;
- yêu cầu `nvidia-ml-py` cho NVML smoke nếu chưa có.

Khôi sở hữu việc pin/fix server environment.

---

# 17. Current branch discrepancies

## 17.1 “Custom proposer” vs implementation

Trello nói custom proposer, nhưng branch hiện dùng:

```text
method = draft_model
Qwen2.5-0.5B-Instruct
k=3
```

Worklog phải mô tả đúng implementation thực tế.

## 17.2 “2 port độc lập”

Port identities là 8001/8002, nhưng switcher hiện chỉ chạy một lane tại một thời điểm trên một GPU.

## 17.3 Telemetry chưa tồn tại

Current branch diff chưa bật native `--per-request-spec-decode-metrics summary` trên Static launcher; đây là default fix trước mọi custom instrumentation.

---

# 18. Stop conditions của Dương

Dương dừng và trả blocker cho Khôi nếu:

- Static không trả `metrics.speculative_decoding` dù đã bật native flag;
- request id không join;
- alpha/tau không recompute được từ native counters;
- Interface B bị phá;
- Static crash/OOM;
- VRAM baseline tăng qua hai batch;
- runtime drift không kiểm soát;
- telemetry transport không ổn định.

Dương dừng và trả integration issue cho Đại nếu:

- harness bỏ `server_metrics`;
- alpha/tau mapping sai;
- TPOT definition khác;
- VRAM source khác;
- server request id bị discard.

---

# 19. Definition of Done — riêng Dương

```text
[ ] telemetry v1 schema khóa
[ ] field ownership rõ
[ ] vanilla null semantics pass
[ ] Static native `metrics.speculative_decoding` pass
[ ] checker pass
[ ] Interface B regression pass
[ ] Vanilla 3 warmup + 10/10 measured pass
[ ] Static 3 warmup + 10/10 measured pass
[ ] peak <=14 GiB cả hai lane
[ ] VRAM stability pass
[ ] tracked smoke summary tồn tại
[ ] runtime/commit recorded
[ ] handoff Đại đủ mapping
```

Sau đó Dương **dừng M1-01** và không chạy concurrency sweep.

---

# 20. Acceptance Matrix

| ID | Acceptance | Owner |
|---|---|---|
| D1 | Canonical telemetry schema v1 | Dương |
| D2 | `draft_acceptance_rate` -> alpha semantics | Dương / native vLLM |
| D3 | `mean_acceptance_length` (bonus included) -> tau | Dương / native vLLM |
| D4 | tree budget limitation documented | Dương |
| D5 | Native `metrics.speculative_decoding` transport enabled | Dương + Khôi |
| D6 | request id joinable | Khôi producer / Dương verify |
| D7 | TPOT harness-owned | Dương |
| D8 | VRAM NVML/harness-owned | Dương |
| D9 | Interface B unchanged | Dương |
| D10 | 10 Vanilla pass | Dương |
| D11 | 10 Static pass | Dương |
| D12 | peak <=14336 MiB | Dương |
| D13 | VRAM stability checked | Dương |
| D14 | exact runtime recorded | Dương |
| D15 | handoff Đại documented | Dương |
| D16 | Dương stops before M1-02 sweep | Dương |

---

# 21. Trạng thái

```text
[x] current branch reviewed
[x] M0-02/M0-03 dependencies reviewed
[x] Frozen Interface A/B reviewed
[x] Dương ownership defined
[x] telemetry semantics defined
[x] smoke protocol defined
[x] VRAM gate defined
[x] handoff boundary defined
[ ] implementation
[ ] live T4 smoke
[ ] Đại handoff acceptance
```
