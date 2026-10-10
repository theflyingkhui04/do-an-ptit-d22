# Worklog Kỹ Thuật: [M2-01] Kiểm Thử Chuyển Đổi Tree Budget Theo Acceptance Rate

## 0. Thông Tin Công Việc

- **Task:** `[M2-01] Hardware Profiler & BudgetController v0`
- **Phần được phân công:** `[Đại] Viết unit test mô phỏng thay đổi tree budget theo các kịch bản alpha cao/thấp`
- **Người thực hiện:** Bùi Thiên Đại (daijestic) — Partner 30%, Testing & Technical Review
- **Lead:** Cấn Đức Khôi — Dev & PM
- **Ngày thực hiện và xác minh:** 10/10/2026
- **Repository:** `theflyingkhui04/do-an-ptit-d22`
- **Branch:** `feat/m2-01-budget-controller-v0`
- **Implementation SHA ban đầu:** `ddf24d194e13b08b5033109d192184f7f55518d5`
- **Upstream M2-02 SHA:** `bba730d`
- **Commit của Đại sau rebase:** `37225b7` (trước bản sửa test mới nhất)
- **Môi trường:** Google Colab T4, Linux, Python 3.13.15, pytest 8.4.2
- **Trạng thái:** Unit-test verification PASS; bản sửa cuối và worklog chưa commit/push, chưa được PM nghiệm thu.

Tài liệu này ghi nhận phần việc do Đại thực hiện, tiếp nối implementation và worklog của Khôi tại `logs/Khoi/M2-01/M2-01_worklogs.md`.

---

## 1. Bối Cảnh Và Mục Tiêu

### 1.1. Bối cảnh nghiên cứu

Đề tài nghiên cứu *Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference on Consumer GPUs* hướng tới điều chỉnh speculative decoding theo đặc điểm phần cứng và workload.

Ở baseline M1, nhóm đã quan sát sự khác biệt về acceptance và speedup giữa các workload. Điều này đặt ra yêu cầu không sử dụng một draft depth cố định trong mọi tình huống.

`BudgetController v0` do Khôi hiện thực nhằm lựa chọn ngân sách nháp `k` từ acceptance telemetry, cost profile và memory pressure.

Để kiểm tra controller không chỉ trả về những giá trị hợp lệ mà còn thực sự chuyển trạng thái khi acceptance thay đổi, Đại thực hiện thêm các test mô phỏng theo chuỗi.

### 1.2. Mục tiêu của Đại

1. Xác minh budget được lựa chọn đúng tại các mức alpha cao, trung bình và thấp.
2. Mô phỏng acceptance thay đổi liên tiếp trong cùng request.
3. Kiểm tra chuỗi `tree_budgets` và telemetry tương ứng.
4. Xác minh adapter bỏ qua draft generator khi `k=0`.
5. Kiểm tra VRAM guard được ưu tiên.
6. Tìm và chứng minh bằng test các trường hợp biên chưa được baseline tests bao phủ.
7. Đảm bảo tests của M2-01 tiếp tục chạy đúng khi M2-02 được phát triển trên branch chung.

### 1.3. Phạm vi

Đây là **CPU-level unit testing với synthetic cost fixtures**.

Phạm vi không bao gồm GPU benchmark, đo VRAM leak thực tế, chạy vLLM adaptive server hoặc đánh giá speedup.

---

## 2. Source Reading Và Phân Tích Thuật Toán

### 2.1. Các thành phần được đọc và kiểm tra

- `src/proposer/controller.py`
- `src/proposer/adapter.py`
- `src/proposer/__init__.py`
- `tests/test_budget_controller.py`
- `tests/test_proposer_adapter.py`

`BudgetController` quản lý acceptance state và lựa chọn budget.

`AdaptiveProposerAdapter` quản lý controller theo từng request, gọi draft generator và nhận verification feedback.

### 2.2. Rolling acceptance

Công thức EMA:

```text
alpha_step = num_accepted / num_drafted

alpha_ema(t) =
    beta * alpha_ema(t-1)
    + (1 - beta) * alpha_step
```

Cấu hình được kiểm thử:

```text
ema_beta = 0.8
fallback_threshold = 0.65
high_threshold = 0.80
```

### 2.3. Expected accepted-token progress và cost model

Controller v0 sử dụng ước lượng:

```text
E[tau(k)] = 1 + alpha + alpha^2 + ... + alpha^k
```

Efficiency dự kiến:

```text
efficiency(k, B) = E[tau(k)] / C(k, B)
```

Ngân sách được xem xét:

```text
k in {0, 2, 3, 4}
```

Quyết định còn chịu ảnh hưởng của:

- Cost profile.
- Acceptance threshold.
- VRAM guard.
- Candidate budgets.

Do đó, không thể khẳng định một mức alpha luôn tương ứng với một giá trị `k` trong mọi cost profile.

---

## 3. Thiết Kế Bộ Test Của Đại

### 3.1. Artifacts

Hai file test được bổ sung:

```text
tests/test_budget_transition_scenarios.py
tests/test_budget_fallback_recovery.py
```

`test_budget_transition_scenarios.py` phục vụ trực tiếp checklist mô phỏng high/low alpha.

`test_budget_fallback_recovery.py` ghi nhận một failure mode được phát hiện trong quá trình review.

### 3.2. Test fixture

Sử dụng cost matrix giả lập, đơn vị millisecond:

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

Ma trận trên **không phải T4 hardware measurement**. Các giá trị được chọn để tạo ra test oracle xác định cho cost-model selection.

### 3.3. Test theo các mức alpha

| Initial alpha | Expected budget |
|---|---|
| 0.90 | k=4 |
| 0.72 | k=3 |
| 0.50 | k=0 |

Test dùng `pytest.mark.parametrize` để tách ba trường hợp độc lập.

Mỗi trường hợp kiểm tra `decision.k` và `decision.alpha_ema`.

### 3.4. Test state transition trong cùng request

Mục tiêu là xác minh chuỗi high → medium → low acceptance.

**Stage 1 — High acceptance**

```text
Initial alpha = 0.90
Selected budget = 4
Verification = 4 accepted / 4 drafted

Next EMA =
0.8 * 0.90 + 0.2 * 1.0
= 0.92
```

**Stage 2 — Acceptance giảm**

```text
Current EMA = 0.92
Selected budget = 4
Verification = 0 accepted / 4 drafted

Next EMA =
0.8 * 0.92 + 0.2 * 0
= 0.736
```

**Stage 3 — Medium acceptance**

```text
Current EMA = 0.736
Selected budget = 3
Verification = 0 accepted / 3 drafted

Next EMA =
0.8 * 0.736
= 0.5888
```

**Stage 4 — Low acceptance**

```text
Current EMA = 0.5888
Fallback threshold = 0.65

Selected budget = 0
Draft generation skipped
```

Expected sequence:

```text
tree_budgets = [4, 4, 3, 0]
```

Expected telemetry:

```text
alpha_ema = 0.5888
step_count = 3
mean_k = 2.75
```

Expected draft generator calls:

```text
[4, 4, 3]
```

Test cũng xác minh request session được xóa sau `end_request()`.

### 3.5. Test VRAM guard

Input giả lập:

```text
initial_alpha = 0.95
min_free_vram_mb = 1500
free_vram_mb = 1000
```

Expected:

```text
budget_k = 0
skipped_draft = True
draft_tokens = []
```

Test xác nhận memory guard có ưu tiên cao hơn acceptance-based selection.

Nó không đo VRAM allocation hoặc chứng minh không có OOM.

---

## 4. Regression Finding — Fallback Recovery Limitation

### 4.1. Vấn đề phát hiện

Trong controller v0, khi acceptance giảm dưới ngưỡng, hệ thống có thể rơi vào trạng thái fallback và không tự phục hồi trong cùng request.

Ví dụ:

```text
alpha_ema = 0.60
fallback_threshold = 0.65

select_budget() -> k=0
```

Khi adapter nhận `k=0`, nó bỏ qua draft generator.

Do không còn speculative verification observation, alpha có thể không được cập nhật.

Trong implementation M2-01, `update(0, 0)` cũng không làm EMA thay đổi.

Khi đó:

```text
alpha_ema remains 0.60
next budget remains 0
```

### 4.2. Phân loại

- **Severity:** MAJOR
- **Location:** `src/proposer/controller.py`, `src/proposer/adapter.py`
- **Problem:** Low-acceptance fallback không có recovery mechanism nội tại.
- **Impact:** Controller có thể tiếp tục chọn Vanilla trong cùng request dù đặc điểm sinh token về sau thay đổi.
- **Evidence:** CPU-level regression tests.

Đây là failure mode của logic hiện có, chưa phải lỗi được quan sát từ vLLM production runtime.

### 4.3. Regression tests được thêm

Bảy trường hợp bao phủ:

1. Low-alpha fallback duy trì khi không có feedback.
2. Zero-draft update không làm EMA phục hồi.
3. Adapter bỏ qua draft generator ở `k=0`.
4. Simulated verification observation làm EMA thay đổi.
5. VRAM guard vẫn được áp dụng sau khi EMA vượt ngưỡng.
6. Favorable cost profile cho phép chọn draft budget trở lại.
7. Unfavorable cost profile có thể tiếp tục chọn Vanilla.

### 4.4. Kiểm chứng khả năng EMA phục hồi

Với:

```text
Initial EMA = 0.60
beta = 0.8
new observed acceptance = 1.0
```

Kết quả:

```text
new EMA =
0.8 * 0.60 + 0.2 * 1.0
= 0.68
```

EMA vượt ngưỡng `0.65`, nhưng điều này không tự đảm bảo `k>0`.

Cần phân biệt:

```text
Verification observation received
    !=
EMA crosses fallback threshold
    !=
Adaptive decoding resumes
```

### 4.5. Đề xuất, chưa triển khai

Phương án được xem xét là **bounded periodic probing** nhằm thu thập acceptance feedback mới sau khi rơi vào Vanilla fallback.

Các ràng buộc cần xem xét:

- Giới hạn tần suất probing.
- Giới hạn draft depth khi probe.
- Không probing khi memory pressure không an toàn.
- Đo cả probing overhead.
- Đánh giá bằng latency/throughput, không chỉ acceptance.

**PM DECISION REQUIRED:** Thay đổi controller policy vượt khỏi việc chỉ viết unit test. Chưa được tự triển khai hay điều chỉnh experiment protocol.

---

## 5. Baseline Test Verification Trước Rebase

### 5.1. Git identity

```text
Branch:
feat/m2-01-budget-controller-v0

HEAD:
ddf24d194e13b08b5033109d192184f7f55518d5
```

### 5.2. Lệnh kiểm thử

```bash
python -m pytest \
  tests/test_budget_controller.py \
  tests/test_proposer_adapter.py \
  tests/test_budget_fallback_recovery.py \
  tests/test_budget_transition_scenarios.py \
  -v
```

### 5.3. Kết quả

```text
Python 3.13.15
pytest 8.4.2

collected 30 items

12 BudgetController tests PASSED
 6 ProposerAdapter tests PASSED
 7 fallback regression tests PASSED
 5 budget transition tests PASSED

30 passed in 0.08s
```

Kết quả này xác minh test suite trên code trước thay đổi M2-02.

### 5.4. Git review

Hai file test mới đã được review bằng:

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

Diff tại thời điểm review:

```text
tests/test_budget_fallback_recovery.py       185 lines
tests/test_budget_transition_scenarios.py    203 lines

Total: 388 lines
```

`git diff --check` không báo lỗi.

---

## 6. Commit Đầu Tiên Và Đồng Bộ Branch

### 6.1. Commit local ban đầu

Commit được tạo trên branch M2-01:

```text
0511abc
test(m2-01): add budget transition and fallback regression tests
```

Author:

```text
BUI DUC DAI
```

Commit bao gồm:

```text
logs/daijestic/M2-01/M2-01_worklogs.md
tests/test_budget_fallback_recovery.py
tests/test_budget_transition_scenarios.py
```

### 6.2. Branch divergence

Sau `git fetch origin`, Git báo:

```text
[ahead 1, behind 3]
```

Trong thời gian Đại thực hiện unit tests, Khôi đã thêm ba commit M2-02 lên branch dùng chung:

```text
c00ff21  docs(m2-02): initialize M2-02 worklog
9d93cf0  docs(m2-02): update dual-track strategy
bba730d  feat(m2-02): implement AdaptiveDraftController
```

Lần push trực tiếp bị từ chối:

```text
[rejected] (non-fast-forward)
```

Đây là cơ chế bảo vệ lịch sử Git, không phải failure của unit tests.

### 6.3. Backup và rebase

Trước khi tích hợp remote commits, đã tạo backup reference:

```bash
git branch backup/daijestic-m2-01-0511abc 0511abc
```

Sau đó chạy:

```bash
git rebase origin/feat/m2-01-budget-controller-v0
```

Kết quả:

```text
Successfully rebased and updated refs/heads/feat/m2-01-budget-controller-v0.
```

Commit của Đại sau rebase:

```text
37225b7
```

Lịch sử mới:

```text
37225b7  test(m2-01): add budget transition and fallback regression tests
bba730d  feat(m2-02): implement AdaptiveDraftController
9d93cf0  docs(m2-02): update dual-track strategy
c00ff21  docs(m2-02): initialize M2-02 worklog
ddf24d1  docs(m2-01): finalize adapter worklog
```

---

## 7. Phát Hiện Failure Sau Rebase

### 7.1. Kết quả kiểm thử lần đầu

Sau rebase, chạy lại bốn file test.

Kết quả:

```text
collected 37 items

36 passed
1 failed

FAILED:
tests/test_budget_transition_scenarios.py::
test_tree_budget_changes_with_acceptance
```

Test kỳ vọng:

```text
Third selected budget = 3
```

Nhưng thực tế:

```text
Actual budget = 0

Low acceptance fallback:
alpha 0.444 < threshold 0.650
```

### 7.2. Root cause

M2-02 thay đổi default controller được tạo bởi `AdaptiveProposerAdapter`.

Trước M2-02, adapter dùng `BudgetController` theo mặc định.

Sau commit `bba730d`, constructor của adapter sử dụng:

```python
controller_cls: type[BudgetController] = AdaptiveDraftController
```

`AdaptiveDraftController` dùng *bias-corrected EMA* cho budget selection.

Với hai acceptance samples:

```text
[1.0, 0.0]
```

Raw EMA:

```text
0.736
```

Nhưng effective alpha:

```text
(0.736 - 0.90 * 0.8^2) / (1 - 0.8^2)
= 0.444444...
```

Vì effective alpha thấp hơn `0.65`, M2-02 chọn `k=0`.

Ngược lại, test M2-01 ban đầu được thiết kế cho `BudgetController`, dùng raw EMA `0.736` và synthetic cost model, nên kỳ vọng `k=3`.

### 7.3. Kết luận nguyên nhân

**Observation:** 1/37 tests thất bại sau rebase.

**Root cause:** Test M2-01 phụ thuộc ngầm vào default controller class, trong khi default của adapter đã được M2-02 thay đổi.

**Inference:** Failure này là vấn đề tương thích test với upstream implementation. Chưa đủ căn cứ kết luận M2-02 production logic bị sai.

Không được sửa production controller hoặc nới assertion để ép test PASS.

---

## 8. Cách Khắc Phục Và Verification Cuối

### 8.1. Giải pháp

Sử dụng explicit dependency injection khi tạo adapter để test M2-01 luôn kiểm tra `BudgetController` đúng phạm vi.

Đảm bảo phần import chỉ có:

```python
from src.proposer.adapter import AdaptiveProposerAdapter
from src.proposer.controller import BudgetController, CostProfile
```

Trong `test_tree_budget_changes_with_acceptance()`:

```python
adapter = AdaptiveProposerAdapter(
    controller_cls=BudgetController,
    controller_kwargs={
        "initial_alpha": 0.90,
        "ema_beta": 0.80,
        "fallback_threshold": 0.65,
        "high_threshold": 0.80,
        "cost_profile": make_profile(),
    },
)
```

Không thay đổi test oracle hoặc expected budget sequence.

### 8.2. Verification sau sửa

Lệnh đã thực hiện:

```bash
python -m pytest \
  tests/test_budget_controller.py \
  tests/test_proposer_adapter.py \
  tests/test_budget_fallback_recovery.py \
  tests/test_budget_transition_scenarios.py \
  -q

git diff --check
git diff -- tests/test_budget_transition_scenarios.py
```

Kết quả thực tế:

```text
..................................... [100%]

37 passed in 0.10s
```

`git diff --check` không báo lỗi.

### 8.3. Final code review

Diff cuối cùng của test chỉ gồm:

- Chỉ định `controller_cls=BudgetController`.
- Chuẩn hóa dấu phẩy và cách đóng constructor.
- Loại bỏ dòng trống dư giữa import và helper function.

Không sửa:

- Controller production logic.
- Adapter production logic.
- Cost profile.
- Test oracle.
- Expected transition sequence.
- Acceptance thresholds.

Không còn duplicate import.

### 8.4. Kết luận verification

**FACT:** Cả 37 unit tests đã PASS sau rebase và sửa phụ thuộc controller class.

**FACT:** 12 tests mới của Đại tiếp tục PASS cùng với 25 baseline/upstream tests.

**FACT:** Không cần thay đổi production code để khôi phục kết quả PASS.

**LIMITATION:** Kết quả này xác minh CPU-level controller/adapter logic trên working tree sau rebase, không chứng minh vLLM integration hoặc runtime GPU correctness.

---

## 9. Tổng Hợp Evidence

| Evidence | Kết quả |
|---|---|
| M2-01 implementation ban đầu | `ddf24d1` |
| M2-02 upstream implementation | `bba730d` |
| Commit test của Đại sau rebase | `37225b7` |
| Test trước rebase | 30/30 PASS |
| Test ngay sau rebase | 36 PASS, 1 FAIL |
| Root cause | Default adapter controller đổi sang `AdaptiveDraftController` |
| Fix | Explicit `controller_cls=BudgetController` |
| Test sau fix | 37/37 PASS |
| Python | 3.13.15 |
| pytest | 8.4.2 |
| GPU | Colab T4 được cấp phát; unit tests không sử dụng GPU computation |
| Whitespace check | PASS |
| Final fix commit | Chưa tạo |
| Push GitHub | Chưa xác minh |
| PM acceptance | Chưa xác minh |

Việc giữ lại cả kết quả FAIL trung gian là cần thiết để tái hiện quá trình phát hiện, phân tích và khắc phục vấn đề tương thích.

Không được xóa thất bại trung gian khỏi nhật ký chỉ vì lần chạy cuối đã PASS.

---

## 10. Technical Findings Chưa Được Giải Quyết

### Finding 1 — Fallback Recovery Limitation

**Severity:** MAJOR

Controller có thể duy trì `k=0` trong cùng request khi thiếu acceptance observation mới.

Regression tests chứng minh failure mode ở mức logic.

**Recommendation:** Đề xuất PM xem xét bounded periodic probing hoặc recovery policy khác.

**Status:** Chưa được chấp thuận hoặc implemented.

### Finding 2 — Telemetry Semantics

**Severity:** MAJOR nếu trường này được dùng tính acceptance metric trong benchmark.

Adapter hiện ánh xạ:

```text
mean_accept_len <- mean_k
```

Nhưng `mean_k` phản ánh budget trung bình, không phải accepted draft length thực tế.

**Recommendation:** Đối chiếu Interface A và metric formulas trước khi dùng dữ liệu này trong experimental analysis.

**Status:** Chưa thay đổi production code.

### Finding 3 — Memory Lifecycle

**Severity:** Chưa đủ bằng chứng để khẳng định memory leak.

`_history` lưu một entry cho mỗi lần budget selection và có thể tăng theo độ dài request.

`end_request()` xóa controller khỏi `_active_sessions`, nhưng `active_request_count == 0` không chứng minh RAM/VRAM đã trở về baseline.

**Recommendation:** Chỉ kết luận memory safety sau controlled measurement trên T4 với warm-up, repeats, request lifecycle và memory sampling được ghi rõ.

---

## 11. Checklist Và Trạng Thái Bàn Giao

### Checklist được phân công

**[Đại] Viết unit test mô phỏng thay đổi tree budget theo các kịch bản alpha cao/thấp**

- [x] Đọc implementation và tracing controller/adapter.
- [x] Xác định test scenarios và expected transitions.
- [x] Kiểm thử acceptance cao, trung bình và thấp.
- [x] Kiểm thử chuỗi high → medium → low trong cùng request.
- [x] Kiểm thử `tree_budgets` và telemetry.
- [x] Kiểm thử VRAM guard bằng simulated input.
- [x] Bổ sung regression tests cho fallback limitation.
- [x] Chạy 30/30 tests PASS trước rebase.
- [x] Backup commit trước khi đồng bộ branch.
- [x] Rebase thành công lên upstream M2-02.
- [x] Phân tích 36 PASS / 1 FAIL sau rebase.
- [x] Khắc phục phụ thuộc ngầm vào default controller.
- [x] Chạy 37/37 tests PASS sau sửa.
- [x] Kiểm tra final diff và whitespace.
- [ ] Commit bản sửa test và worklog cuối cùng.
- [ ] Push artifacts lên GitHub.
- [ ] Handoff cho Khôi review.
- [ ] PM xác nhận nghiệm thu checklist.

**Engineering status:** FUNCTIONALLY DONE — FINAL COMMIT/PUSH PENDING.

**Repository acceptance status:** Chưa có bằng chứng push và review cuối cùng.

Không tự cập nhật Trello hoặc tự đánh dấu PM checklist hoàn thành.

---

## 12. Retrospective Và Bài Học

### 12.1. Điều đã học

1. Một passing test chỉ xác nhận hành vi nằm trong phạm vi assertion.
2. Synthetic unit tests kiểm tra controller logic, không đại diện cho GPU performance.
3. State transitions cần được kiểm chứng trong cùng request chứ không chỉ ở các giá trị alpha độc lập.
4. Raw EMA và bias-corrected EMA có thể dẫn đến quyết định budget khác nhau.
5. Thay đổi default dependency có thể làm test cũ thất bại dù API vẫn tương thích.
6. Explicit controller injection làm mục tiêu unit test rõ ràng và ít phụ thuộc implementation khác.
7. Git rebase thành công không đồng nghĩa toàn bộ tests vẫn đúng.
8. Failure trung gian phải được ghi lại như engineering evidence, không bỏ qua.
9. Acceptance recovery, budget recovery và speculative-runtime recovery là các khái niệm khác nhau.
10. Hoàn thành code, commit local, push remote và PM acceptance là các mốc riêng biệt.

### 12.2. Kết luận

Phần việc của Đại đã bổ sung 12 test cases tập trung vào budget transitions, fallback behavior và controller edge cases.

Trước rebase, toàn bộ 30 tests PASS trên M2-01 implementation.

Sau khi Khôi triển khai M2-02, test suite mở rộng lên 37 tests. Một test của Đại phát hiện sự thay đổi default adapter controller và được sửa bằng explicit dependency injection.

**Final local verification: 37/37 PASS.**

Kết quả hiện tại đủ để chứng minh phần kiểm thử logic high/low alpha đã được triển khai và hoạt động theo controller contract được chọn.

Chưa đủ để kết luận toàn bộ M2-01 hoặc Adaptive Speculative Decoding runtime đã nghiệm thu.

### 12.3. Next Handoff

Lưu worklog phiên bản mới, commit bản sửa test và tài liệu, đồng bộ branch remote an toàn, sau đó gửi Khôi/PM các bằng chứng:

- Git SHA cuối cùng.
- Test artifacts.
- Worklog.
- pytest output `37 passed in 0.10s`.
- Finding fallback recovery và telemetry semantics.
- Giới hạn của CPU-level verification.

**Không tự sửa Trello, đổi scope hoặc điều chỉnh benchmark protocol.**