# Worklog Kỹ Thuật: [M2-01] Kiểm Thử BudgetController v0 — Chuyển Đổi Tree Budget Theo Acceptance Rate

## Thông tin công việc

- **Mã công việc:** `[M2-01]`
- **Cột mốc:** M2 — Adaptive v0 & Lossless Verification
- **Người thực hiện:** Bùi Thiên Đại (daijestic) — Partner 30%, Unit Test & Technical Review
- **Người phụ trách chính:** Cấn Đức Khôi — Lead 70%, Implementation & Integration
- **Ngày ghi nhận:** 10/10/2026
- **Branch kiểm thử:** `feat/m2-01-budget-controller-v0`
- **Base implementation SHA:** `ddf24d194e13b08b5033109d192184f7f55518d5`
- **Môi trường:** Google Colab T4, Linux, Python 3.13.15, pytest 8.4.2
- **Trạng thái:** Unit tests PASS; chờ commit/push, review và nghiệm thu checklist

### Phạm vi được phân công

**[Đại] Viết unit test mô phỏng thay đổi tree budget theo các kịch bản alpha cao/thấp.**

Phần việc tập trung xác minh logic của `BudgetController v0` và `AdaptiveProposerAdapter` do Khôi hiện thực, đặc biệt là khả năng thay đổi draft budget theo acceptance telemetry trong cùng một request.

Đây là phần tiếp nối implementation và unit tests nền tảng được Khôi ghi nhận tại `logs/Khoi/M2-01/M2-01_worklogs.md`.

---

## 1. Bối Cảnh & Mục Tiêu Kiểm Thử

### 1.1. Bối cảnh

Kết quả baseline M1 cho thấy hiệu quả Speculative Decoding phụ thuộc vào workload và acceptance rate. Một cấu hình draft depth cố định có thể đem lại speedup trong trường hợp acceptance cao nhưng gây tăng chi phí không cần thiết khi acceptance thấp.

BudgetController v0 được thiết kế để điều chỉnh ngân sách nháp `k` dựa trên:

- Acceptance rate trung bình hàm mũ (`alpha_ema`).
- Mô hình chi phí phần cứng `C(k, B)`.
- Ngưỡng fallback khi acceptance thấp.
- Giới hạn bộ nhớ VRAM.

Khôi đã xây dựng controller và adapter cùng 18 unit tests nền tảng.

Phần kiểm thử của Đại tiếp tục xác minh **state transitions**: khi acceptance thay đổi qua các bước decoding, quyết định budget và dữ liệu telemetry có phản ánh chính xác thay đổi đó hay không.

### 1.2. Mục tiêu

1. Xác minh quyết định budget tại các mức acceptance cao, trung bình và thấp.
2. Mô phỏng chuỗi acceptance thay đổi trong cùng một request.
3. Xác minh controller tự động chuyển sang Vanilla khi acceptance xuống dưới ngưỡng.
4. Xác minh VRAM guard được ưu tiên cao hơn quyết định mở rộng draft.
5. Kiểm tra consistency giữa budget thực thi và telemetry `tree_budgets`.
6. Phát hiện những giới hạn logic chưa được các unit tests nền tảng bao phủ.

### 1.3. Test oracle và giới hạn

Các kết quả kỳ vọng được xác định từ công thức EMA, mô hình chi phí và controller policy hiện hành.

Các cost matrix trong unit tests là dữ liệu giả lập (*synthetic fixture*) được chọn có chủ đích để kiểm tra thuật toán, **không phải số đo phần cứng Tesla T4**.

---

## 2. Phân Tích Thuật Toán Trước Khi Viết Test

### 2.1. Rolling acceptance rate

Controller sử dụng công thức:

```text
alpha_step = num_accepted / num_drafted

alpha_ema(t)
    = beta * alpha_ema(t-1)
      + (1 - beta) * alpha_step
```

Với `beta = 0.8`, EMA phản ánh dữ liệu acceptance mới nhưng vẫn giữ một phần ảnh hưởng của các bước trước.

### 2.2. Chi phí và hiệu quả dự kiến

Số token được tạo kỳ vọng mỗi verification step:

```text
E[tau(k)] = 1 + alpha + alpha^2 + ... + alpha^k
```

Hiệu quả dự kiến:

```text
efficiency(k, B) = E[tau(k)] / C(k, B)
```

Controller xét các budget `k ∈ {0, 2, 3, 4}`, so sánh efficiency với Vanilla và áp dụng các safety guard.

Vì vậy, **alpha cao không tự động chứng minh `k=4` tối ưu**. Test cần sử dụng cost profile xác định để kiểm chứng các budget được chọn.

### 2.3. Các ngưỡng chính

| Điều kiện | Hành vi cần kiểm tra |
|---|---|
| `alpha_ema < 0.65` | Fallback về `k=0` |
| `0.65 ≤ alpha_ema < 0.80` | Chỉ xem xét budget được policy cho phép |
| `alpha_ema ≥ 0.80` | Có thể xem xét `k=4` |
| Free VRAM dưới ngưỡng | Bắt buộc `k=0` |

---

## 3. Thiết Kế Bộ Kiểm Thử Của Đại

### 3.1. File kiểm thử

Hai file mới được xây dựng:

```text
tests/
├── test_budget_fallback_recovery.py
└── test_budget_transition_scenarios.py
```

File `test_budget_transition_scenarios.py` là phần kiểm thử trực tiếp cho checklist alpha cao/thấp.

File `test_budget_fallback_recovery.py` bổ sung regression evidence về một failure mode được phát hiện trong quá trình review.

### 3.2. Kịch bản A — Budget selection theo các mức alpha

Sử dụng `CostProfile` giả lập:

```python
CostProfile(
    matrix={
        (0, 1): 20.0,
        (2, 1): 21.0,
        (3, 1): 24.0,
        (4, 1): 27.0,
    }
)
```

Đơn vị chi phí là millisecond, chỉ dùng làm test fixture.

Các trường hợp kỳ vọng:

| Alpha đầu vào | Budget kỳ vọng |
|---|---|
| `0.90` | `k=4` |
| `0.72` | `k=3` |
| `0.50` | `k=0` |

Các lựa chọn này là kết quả của cả cost profile và policy hiện hành. Chúng không phải quy tắc bắt buộc cho mọi workload hoặc mọi GPU.

### 3.3. Kịch bản B — State transitions trong cùng request

Mô phỏng một request có acceptance cao ở giai đoạn đầu, sau đó giảm dần.

**Bước 1 — Acceptance cao**

```text
Initial alpha_ema = 0.90
Selected budget   = 4
Verification      = 4 accepted / 4 drafted

New alpha_ema:
0.8 * 0.90 + 0.2 * 1.0 = 0.92
```

**Bước 2 — Acceptance bắt đầu giảm**

```text
alpha_ema         = 0.92
Selected budget   = 4
Verification      = 0 accepted / 4 drafted

New alpha_ema:
0.8 * 0.92 + 0.2 * 0.0 = 0.736
```

**Bước 3 — Acceptance trung bình**

```text
alpha_ema         = 0.736
Selected budget   = 3
Verification      = 0 accepted / 3 drafted

New alpha_ema:
0.8 * 0.736 + 0.2 * 0.0 = 0.5888
```

**Bước 4 — Acceptance thấp**

```text
alpha_ema         = 0.5888
Fallback threshold = 0.65
Selected budget    = 0

Draft generator skipped
```

Chuỗi budget kỳ vọng:

```text
tree_budgets = [4, 4, 3, 0]
```

Kết quả telemetry kỳ vọng:

```text
alpha_ema = 0.5888
step_count = 3
mean_k = (4 + 4 + 3 + 0) / 4 = 2.75
```

Test còn kiểm tra `draft_generator` chỉ được gọi với:

```text
[4, 4, 3]
```

Điều này xác nhận không sinh nháp khi controller quyết định `k=0`.

### 3.4. Kịch bản C — VRAM safety override

Khởi tạo request có acceptance cao:

```text
alpha_ema = 0.95
min_free_vram_mb = 1500
free_vram_mb = 1000
```

Hành vi kỳ vọng:

```text
Selected budget = 0
skipped_draft = True
draft_tokens = []
```

Mục tiêu là xác minh safety guard được ưu tiên ngay cả khi acceptance rất cao.

Đây là mô phỏng tín hiệu bộ nhớ, không phải phép đo VRAM thực tế trên GPU.

---

## 4. Regression Tests — Phát Hiện Fallback Recovery Limitation

### 4.1. Phát hiện

Qua việc tracing `BudgetController.update()` và `AdaptiveProposerAdapter.propose()`, phát hiện trạng thái fallback có thể bị duy trì vô thời hạn trong cùng request.

Tại:

```text
alpha_ema = 0.60
fallback_threshold = 0.65
```

Controller chọn `k=0`.

Adapter không gọi draft generator, nên không phát sinh acceptance observation mới từ speculative verification.

Nếu không có nguồn observation khác:

```text
alpha_ema remains 0.60
next selected budget = 0
```

Ngay cả khi gọi:

```python
controller.update(
    num_accepted=0,
    num_drafted=0,
)
```

logic hiện tại cũng giữ nguyên EMA.

**Phân loại:** MAJOR — fallback recovery limitation.

### 4.2. Regression evidence

Bảy test bổ sung kiểm tra:

1. Fallback tiếp tục duy trì khi không có feedback mới.
2. `update(0, 0)` không làm EMA phục hồi.
3. Adapter không gọi draft generator trong fallback.
4. Acceptance observation giả lập có thể cập nhật EMA.
5. VRAM guard vẫn có ưu tiên sau khi EMA vượt ngưỡng.
6. Cost profile thuận lợi có thể cho phép chọn draft budget trở lại.
7. Cost profile bất lợi vẫn có thể khiến Vanilla là lựa chọn đúng.

Với:

```text
alpha_ema = 0.60
ema_beta = 0.8
new acceptance observation = 1.0
```

EMA sau cập nhật:

```text
0.8 * 0.60 + 0.2 * 1.0 = 0.68
```

Như vậy EMA vượt ngưỡng fallback `0.65`.

Tuy nhiên cần phân biệt ba sự kiện:

```text
Probe executed
    !=
EMA recovered
    !=
Adaptive decoding resumed
```

Các test hiện tại chỉ đưa observation giả lập vào controller. **Không có cơ chế periodic probing được hiện thực trong production code.**

### 4.3. Đề xuất kỹ thuật — chưa được duyệt

Đề xuất nghiên cứu *bounded periodic probing*: định kỳ cho phép draft nhỏ để thu thập acceptance telemetry mới, với điều kiện kiểm soát VRAM và overhead.

Các tham số như chu kỳ probe, draft depth, giới hạn chi phí và chính sách recovery chưa được chốt.

**PM DECISION REQUIRED:** Chỉ triển khai nếu PM chấp thuận scope và phương án đo kiểm. Không tự sửa controller policy trong checklist unit test này.

---

## 5. Kết Quả Thực Thi Trên Colab T4

### 5.1. Lệnh chạy

```bash
cd /root/do-an-ptit-d22

git rev-parse HEAD

python -m pytest \
  tests/test_budget_controller.py \
  tests/test_proposer_adapter.py \
  tests/test_budget_fallback_recovery.py \
  tests/test_budget_transition_scenarios.py \
  -v

git status --short
```

### 5.2. Git identity

```text
ddf24d194e13b08b5033109d192184f7f55518d5
```

### 5.3. Kết quả

```text
Python 3.13.15
pytest 8.4.2
Linux

collected 30 items

tests/test_budget_controller.py
    12 passed

tests/test_proposer_adapter.py
     6 passed

tests/test_budget_fallback_recovery.py
     7 passed

tests/test_budget_transition_scenarios.py
     5 passed

========================
30 passed in 0.08s
========================
```

| Thành phần | Số test | Kết quả |
|---|---:|---|
| BudgetController baseline | 12 | PASS |
| ProposerAdapter baseline | 6 | PASS |
| Fallback regression | 7 | PASS |
| Alpha transition scenarios | 5 | PASS |
| **Tổng** | **30** | **PASS** |

Tất cả các assertions trong bốn file đã PASS trên implementation SHA được ghi nhận.

### 5.4. Kiểm tra thay đổi Git

Đã sử dụng:

```bash
git add -N \
  tests/test_budget_fallback_recovery.py \
  tests/test_budget_transition_scenarios.py

git diff --check
git diff --stat
git diff -- \
  tests/test_budget_fallback_recovery.py \
  tests/test_budget_transition_scenarios.py
```

Kết quả:

```text
tests/test_budget_fallback_recovery.py
    185 lines added

tests/test_budget_transition_scenarios.py
    203 lines added

2 files changed, 388 insertions(+)
```

`git diff --check` không báo lỗi trong output đã cung cấp.

Tuy nhiên, `git add -N` mới chỉ đánh dấu intent-to-add. Tại thời điểm ghi log, **hai file chưa có commit/push riêng của Đại**.

---

## 6. Phân Tích Kết Quả Và Giới Hạn

### 6.1. Những điều đã được chứng minh

**FACT 1:** Các test hiện có đã chạy thành công trên Git SHA xác định.

**FACT 2:** BudgetController có thể lựa chọn các budget khác nhau trong các synthetic scenarios có alpha và chi phí được kiểm soát.

**FACT 3:** Test mô phỏng trong cùng request xác nhận chuỗi `tree_budgets = [4, 4, 3, 0]`.

**FACT 4:** Khi `k=0`, adapter bỏ qua draft generator trong kịch bản đã kiểm thử.

**FACT 5:** Memory-pressure guard giả lập có thể ép controller về `k=0` ngay cả khi acceptance cao.

### 6.2. Những điều chưa được chứng minh

- Controller và adapter đã hoạt động bên trong decoding loop thực tế của vLLM.
- Adaptive lane đã được khởi động và vượt smoke test trên GPU.
- Không có VRAM leak hoặc RAM leak trong quá trình vận hành dài.
- Cost profile thực tế trên Tesla T4 đã được đo, xác minh hoặc hiệu chuẩn.
- Adaptive decoding cải thiện latency, TPOT hoặc throughput trên benchmark chuẩn.
- Periodic probing đã được hiện thực.
- Thuật toán luôn khôi phục được speculative mode trong mọi request.

### 6.3. Vấn đề telemetry cần lưu ý

Trong adapter hiện tại, trường `mean_accept_len` được lấy từ `mean_k`.

Hai khái niệm khác nhau:

```text
mean_k
    = average selected draft budget

mean_accept_len
    = average accepted draft length
```

Không thể dùng trực tiếp `mean_k` thay cho accepted length khi phân tích hiệu quả speculative decoding.

**Review finding:** Cần đối chiếu với telemetry contract trước khi dùng trường này để tính acceptance metric trong benchmark.

### 6.4. Giới hạn memory testing

Việc `active_request_count == 0` sau `end_request()` xác nhận session được loại khỏi dictionary quản lý của adapter.

Nó không chứng minh Python heap hoặc GPU VRAM đã được giải phóng hoàn toàn.

Ngoài ra, `_history` của controller là list có thể tăng theo số lần `select_budget()` trong một request. Cần kiểm tra tác động đến memory footprint đối với request dài.

---

## 7. Bàn Giao Và Nghiệm Thu

### 7.1. Artifacts

**Files do Đại bổ sung:**

```text
tests/test_budget_fallback_recovery.py
tests/test_budget_transition_scenarios.py
```

**Files upstream của Khôi được kiểm thử:**

```text
src/proposer/controller.py
src/proposer/adapter.py
tests/test_budget_controller.py
tests/test_proposer_adapter.py
```

**Worklog của Đại:**

```text
logs/daijestic/M2-01/M2-01_worklogs.md
```

### 7.2. Trạng thái checklist

**Checklist:** `[Đại] Viết unit test mô phỏng thay đổi tree budget theo các kịch bản alpha cao/thấp`

- [x] Đọc và tracing controller/adapter.
- [x] Xác định các test scenarios và expected results.
- [x] Viết test cho acceptance cao, trung bình và thấp.
- [x] Viết test cho chuỗi chuyển đổi budget trong cùng request.
- [x] Xác minh VRAM guard bằng mock input.
- [x] Chạy 30/30 unit tests PASS.
- [x] Review diff của hai file test mới.
- [ ] Commit và push artifacts lên GitHub.
- [ ] Handoff để Khôi review và PM xác nhận nghiệm thu.

**Kết luận:** Phần unit test alpha cao/thấp đã PASS về chức năng. Chưa ghi nhận `DONE WITH EVIDENCE` ở cấp repository cho đến khi có commit/PR, review và handoff đầy đủ.

### 7.3. Những việc không thuộc bằng chứng nghiệm thu này

VRAM leak benchmark thực tế trên T4 và integration smoke test cần được xác minh riêng theo acceptance criteria, phụ thuộc runtime và implementation tích hợp.

Không được dùng kết quả 30 unit tests để thay thế benchmark khoa học hoặc xác nhận toàn bộ M2-01 hoàn thành.

---

## 8. Retrospective

### 8.1. Điều đã học được

1. Một test PASS chỉ chứng minh hành vi đã được assertion kiểm tra, không tự chứng minh mọi yêu cầu hệ thống đều đúng.
2. Một chuỗi state transitions cần được kiểm tra theo từng bước, không chỉ kiểm tra điểm đầu và cuối.
3. EMA recovery và adaptive recovery là hai khái niệm khác nhau.
4. Fallback có thể tạo ra trạng thái thiếu feedback khiến controller không thể tự đánh giá điều kiện quay lại speculative mode.
5. Một quyết định budget đúng phải kết hợp acceptance, chi phí và memory safety.
6. Synthetic unit tests phù hợp cho correctness của policy nhưng không thay thế measurement trên GPU thực.

### 8.2. Kết luận kỹ thuật

Phần kiểm thử do Đại thực hiện đã mở rộng baseline test suite từ 18 lên 30 test và tái hiện được cả budget transitions lẫn một giới hạn của fallback recovery.

Kết quả là **unit-test evidence có khả năng tái hiện trên commit xác định**, không phải scientific evidence về hiệu năng Adaptive Speculative Decoding trên T4.

### 8.3. Next handoff

Bàn giao hai file test, worklog, Git SHA và pytest output cho Khôi review.

Đề xuất PM xem xét finding về fallback recovery và xác nhận các yêu cầu còn lại trước khi chuyển sang integration/benchmark.

**Không tự thay đổi Trello, scope hoặc controller policy.**