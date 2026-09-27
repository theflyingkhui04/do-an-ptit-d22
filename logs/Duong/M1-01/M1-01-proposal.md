# Proposal M1-01 — Telemetry Contract & Smoke-Test Handoff

> **Task:** M1-01 — vLLM Dual-Lane Server  
> **Branch:** `feat/m1-01-vllm-dual-lane-server`  
> **Lead:** Khôi 70%  
> **Partner:** Dương 30%  
> **Proposal focus:** Phần việc Dương  
> **Ngày:** 26/09/2026  
> **Trạng thái:** Ready for review, chưa apply Git.

---

# 1. Why

M1-01 đã có launcher cho Vanilla và Static Draft, nhưng hiện chưa có bằng chứng rằng server đã đạt **Frozen Interface A** để M1-02 benchmark một cách đúng nghĩa.

Text generation thành công chưa đủ. M1 cần biết:

```text
Static speculative decoding accept được bao nhiêu token?
tau trung bình là bao nhiêu?
TPOT được đo theo định nghĩa nào?
VRAM peak lấy từ nguồn nào?
request nào ứng với telemetry nào?
```

Nếu không khóa những điểm này trước khi Đại chạy concurrency sweep, M1 có thể sinh ra bảng benchmark nhưng thiếu provenance metric đáng tin cậy.

Dương được phân Partner 30% chính để đóng khoảng trống này.

---

# 2. Current State

Branch hiện tại:

```text
head: 737f2e8b3dc3a61ba1614d29b428dc91567a9627
ahead main: 1 commit
```

Đã có:

```text
Vanilla launcher
Static draft launcher
lane switcher
basic lane tests
minimal server requirements
```

Chưa có:

```text
telemetry schema
telemetry validator
server-metrics acceptance test
10-request stability test
VRAM leak/stability evidence
M1-02 telemetry handoff
M1-01 worklog
```

Checklist Trello hiện là 4/7 complete; hai item của Dương vẫn incomplete.

---

# 3. Proposal Summary

Đề xuất Dương thực hiện một workstream nhỏ, tách khỏi server implementation của Khôi:

```text
1. Telemetry Contract
2. Contract Validator
3. Dual-Lane Smoke Client
4. VRAM Stability Check
5. Compact Evidence
6. Handoff Note cho Đại
```

Dương không implement speculative decoding.

---

# 4. Ownership

## 4.1 Khôi

Khôi sở hữu:

```text
server process
target/draft config
static k=3
server lifecycle
server/lane lifecycle
native vLLM telemetry flag/config review
```

Khôi phải bảo đảm Static runtime được khởi chạy với native vLLM acceptance metrics enabled. Custom engine patch không phải default path.

## 4.2 Dương

Dương sở hữu:

```text
schema
validator
metric definitions
null semantics
client/server metric ownership
smoke runner
VRAM stability protocol
evidence
handoff
```

## 4.3 Đại

Đại consume:

```text
server_metrics
client timings
NVML metrics
```

rồi chạy benchmark M1-02.

---

# 5. Key Design Proposal — native vLLM 0.29.0 first

## 5.1 Dùng native telemetry, không custom nếu chưa cần

Static launcher SHOULD bật:

```text
--per-request-spec-decode-metrics summary
```

Request smoke/M1-02:

```text
n=1
stream=true
stream_options.include_usage=true
```

và đọc:

```text
data["metrics"]["speculative_decoding"]
```

Native object có:

```text
mean_acceptance_length
draft_acceptance_rate
acceptance_histogram
num_spec_steps
num_accepted_draft_tokens
num_draft_tokens
num_spec_tokens
```

## 5.2 Metric ownership

```text
vLLM native:
  speculative acceptance metrics

Harness:
  TTFT / TPOT / ITL / E2E / output tokens

NVML:
  VRAM peak
```

Dương's normalized handoff record SHALL combine cả ba nguồn:

```text
speculative_decoding
alpha / tau
client_metrics.tpot_ms
gpu_metrics.vram_peak_mb
```

Như vậy payload bàn giao đáp ứng đúng checklist `alpha, mean_accept_len, tpot_ms, vram_peak_mb` mà không làm sai ownership của metric.

## 5.3 Project mapping

```text
alpha = metrics.speculative_decoding.draft_acceptance_rate
tau   = metrics.speculative_decoding.mean_acceptance_length
```

`tau` bao gồm bonus token.

Static `k=3`:

```text
alpha: 0..1
tau:   1..4
```

Vanilla normalized:

```text
speculative_decoding = null
alpha = null
tau = null
```

## 5.4 Raw counters

Dương giữ:

```text
num_spec_steps
num_accepted_draft_tokens
num_draft_tokens
num_spec_tokens
acceptance_histogram
```

để audit/recompute alpha/tau.

## 5.5 tree_budgets

M1 Static `summary` mode:

```text
tree_budgets = null
```

Không fabricate per-step values.

## 5.6 Cell-level aggregation contract cho Đại

Dương bàn giao công thức; Đại implement trong M1-02:

```text
alpha_cell =
  sum(num_accepted_draft_tokens) / sum(num_draft_tokens)

tau_cell =
  1 + sum(num_accepted_draft_tokens) / sum(num_spec_steps)
```

Không lấy arithmetic mean của per-request ratios làm canonical value.

Vanilla `alpha/tau = null`.

Telemetry `summary` mode phải được giữ cố định cho mọi Static run trong M1; không benchmark một số ô với metrics bật và một số ô tắt.

---

# 6. Proposed Files

```text
experiments/configs/m1_request_telemetry.schema.json
experiments/configs/m1_request_telemetry.example.json

scripts/check_m1_telemetry.py
scripts/smoke_m1_01_telemetry.py

tests/test_m1_telemetry_contract.py
tests/test_m1_01_smoke_helpers.py

experiments/results/m1_01_telemetry_contract_summary.json
experiments/results/m1_01_smoke_summary.json
```

Schema validate normalized project record.

Raw output:

```text
experiments/runs/m1_01_smoke/
```

không commit.

Tài liệu:

```text
logs/Khoi/M1-01/openspec.md
logs/Khoi/M1-01/proposal.md
logs/Khoi/M1-01/plan.md
logs/Khoi/M1-01/M1-01_worklogs.md
```

---

# 7. Server Integration Proposal

Default shared change:

```text
scripts/launch_lane2_static.sh
  + --per-request-spec-decode-metrics summary
```

Khôi review vì launcher thuộc track server của Khôi.

M1-02 đã có:

```python
record["server_metrics"] = data.get("metrics")
```

Dương chỉ cần chứng minh nested `server_metrics.speculative_decoding` tồn tại.

Custom patch chỉ là fallback nếu pinned vLLM 0.29.0 native path fail.


# 7A. Execution Environment

Development bắt đầu trong VSCode local:

```text
schema / checker / SSE parser / fixtures / unit tests / summary logic
```

Sau đó mới chạy live trên Linux + Tesla T4 16GB (Kaggle/Colab).

Final evidence về native acceptance metrics, VRAM, OOM và 10-request stability chỉ lấy từ T4.

---

# 8. Smoke Proposal

Mỗi lane:

```text
3 warm-up
10 measured
temperature=0
seed=42 (required on pinned vLLM 0.29.0)
same prompt
stream=true
```

Run sequential vì `switch_lane.sh` hiện stop lane cũ trước khi bật lane mới.

Vanilla:

```text
8001
qwen2.5-7b-vanilla
```

Static:

```text
8002
qwen2.5-7b-static-draft
```

---

# 9. VRAM Proposal

Sau warm-up:

```text
capture idle baseline
run 10
capture peak
cooldown >=5s
capture idle after
```

MUST:

```text
peak <=14 GiB
no OOM
10/10 success
```

Stability heuristic:

```text
after - baseline <=256 MiB
```

`256 MiB` chỉ là ngưỡng phát hiện bất thường; nếu vượt phải chạy batch thứ hai trước khi gọi suspected leak.

Trello `<=14GB` được hiểu operationally là `<=14336 MiB` (14 GiB) vì NVML báo MiB.

Nếu fail, rerun second batch. Chỉ gọi “suspected leak” nếu baseline tiếp tục tăng.

---

# 10. Dương Handoff Point

Dương bàn giao cho Đại khi:

```text
schema PASS
static native `metrics.speculative_decoding` PASS
vanilla null semantics PASS
10/10 each lane PASS
VRAM gate PASS
summary artifact committed
transport documented
mapping alpha/tau documented
```

Sau đó Dương **dừng M1-01**.

Dương không chạy:

```text
concurrency 1,2,4,8,16,32
5 repeats per cell
performance sweep
summary.csv final aggregation
```

Đó là M1-02.

---

# 11. Blockers Returned to Khôi

Dương trả blocker cho Khôi nếu:

```text
Static has no `metrics.speculative_decoding` after native flag
server response ID cannot join telemetry
metrics transport is unstable
server crashes/OOM
runtime environment drifts unexpectedly
Interface B requires engine-specific client fields
```

Dương không workaround bằng cách sửa metric definition.

---

# 12. Blockers Returned to Đại

Dương trả integration issue cho Đại nếu:

```text
harness ignores server_metrics
alpha/tau mapping differs from native fields
TPOT definition differs
VRAM source differs
server_request_id is discarded
```

---

# 13. Risk Register

## R1 — Alpha giả

Nguy cơ: suy alpha từ throughput hoặc token count.

Mitigation:

```text
only server-native accepted/proposed counters
```

## R2 — Vanilla alpha=0

Nguy cơ: summary hiểu vanilla đã reject 100%.

Mitigation:

```text
null = not applicable
```

## R3 — Hai nguồn TPOT

Mitigation:

```text
harness authoritative
```

## R4 — VRAM leak false positive do warmup/cache

Mitigation:

```text
compare post-warmup idle baseline
```

## R5 — “2 ports” bị hiểu là simultaneous servers

Mitigation:

```text
document lane switching is sequential on one T4
```

## R6 — Version drift

Current server requirements chưa pin.

Mitigation:

```text
record exact versions
Khôi pins M1 runtime; Dương verifies exact vLLM 0.29.0
```

---

## R7 — Native tau convention / upstream EOS accounting

`mean_acceptance_length` của vLLM gồm bonus token. Với v0.29.0 còn có upstream report về terminal EOS accounting.

Mitigation:

```text
preserve raw counters/histogram
label tau as native vLLM metric
avoid over-interpreting tau as exact committed-token count
```

# 14. Acceptance Proposal

Dương’s part PASS only when:

```text
telemetry_schema_valid = true
vanilla_success = 10/10
static_success = 10/10
static_draft_acceptance_rate_present = true
static_mean_acceptance_length_present = true
vram_peak_vanilla <= 14336 MiB
vram_peak_static <= 14336 MiB
vram_stability_pass = true
m1_02_handoff_ready = true
```

---

# 15. Handoff Gates

## H0 — Khôi -> Dương: Server Ready

Input:

```text
Vanilla launcher
Static launcher
served model names
ports
health
server-side metrics transport
```

Dương chỉ bắt đầu live acceptance khi Static telemetry producer tồn tại.

## H1 — Dương Contract Ready

Dương bàn giao:

```text
schema
fixture
checker
metric definitions
```

Khôi phải implement đúng contract.

## H2 — Khôi Telemetry Producer Ready

Khôi bàn giao một sample Vanilla + Static có metrics.

Dương verify schema/request-id semantics.

Nếu H2 fail, Dương dừng live smoke.

## H3 — Dương Smoke Ready

Dương chạy:

```text
10 Vanilla
10 Static
VRAM stability
Interface B regression
```

## H4 — Dương -> Đại

Dương bàn giao:

```text
contract
mapping
transport
summary
known limitations
```

Đây là **điểm dừng chính của Dương trong M1-01**.

## H5 — Đại / M1-02

Đại tiếp tục benchmark sweep; không còn là scope của Dương M1-01.

---

# 16. Current Gaps to Call Out

1. Branch chưa bật native per-request speculative metrics trên Static launcher.
2. `requirements-server.txt` chưa pin version.
3. Trello nói “custom proposer”, nhưng branch thực tế dùng static `draft_model`.
4. Trello nói hai port độc lập, nhưng switcher hiện chạy lane luân phiên trên một GPU.
5. Existing `test_lane1.py/test_lane2.py` chỉ test inference cơ bản, chưa test streaming metrics/VRAM leak.
6. M1-02 harness đã có nơi nhận `server_metrics`, nhưng chưa có final alpha/tau aggregation.

---

# 17. Proposal Decision

M1-01 nên được xem như ba tầng bàn giao:

```text
KHÔI
Server & lane runtime
        |
        v
DƯƠNG
Telemetry contract + validation + smoke
        |
        v
ĐẠI / M1-02
Benchmark sweep
```

Ranh giới trách nhiệm:

```text
Khôi = server/lane config + review native metrics enablement
Dương = normalize + verify + smoke + handoff
Đại  = benchmark with metrics
```

Dương dừng ở **handoff-ready**, không ở **benchmark-done**.
