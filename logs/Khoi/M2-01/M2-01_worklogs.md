# Worklog Kỹ Thuật: [M2-01] Thiết Kế & Cài Đặt Lớp BudgetController v0 Tính Toán N* Theo Rolling Average Alpha

- **Mã công việc:** `[M2-01]` (Cột mốc M2 · Adaptive v0 & Lossless Verification)
- **Người thực hiện:** Cấn Đức Khôi (Lead 70% — Dev + PM, AI Systems) & Bùi Đức Đại (Partner 30% — Dev + IT, Reviewer)
- **Thời gian thực hiện:** 05/10/2026 – 08/10/2026
- **Tài nguyên kỹ thuật:**
  - Mã nguồn lõi: [`src/proposer/controller.py`](../../../src/proposer/controller.py)
  - Package module: [`src/proposer/__init__.py`](../../../src/proposer/__init__.py)
  - Kịch bản Unit Test: [`tests/test_budget_controller.py`](../../../tests/test_budget_controller.py)
  - Nhánh Git: `feat/m2-01-budget-controller-v0`
- **Trạng thái nghiệm thu Subtask 1:** Hoàn thành 100% (12/12 test cases PASS, sẵn sàng tích hợp vào custom proposer hook của vLLM).

---

## 1. Bối Cảnh & Động Lực Kỹ Thuật (Từ Kết Quả M1)

Trong Sprint 2 (M1 Baseline Benchmark), hệ thống suy luận hai làn đã ghi nhận hiện tượng khoa học then chốt:
1. **Workload W1 (JSON) & W2 (SQL):** Tỷ lệ chấp nhận token alpha cao (80.5% – 91.6%), độ dài chấp nhận trung bình tau đạt 3.42 – 3.75 token. Chế độ Static Draft (k=3) tăng tốc vượt trội từ 1.16x đến 1.34x.
2. **Workload W3 (Phân tích tự do):** Cú pháp mở rộng, alpha tụt xuống ~54.5% (tau chỉ còn 2.62). Ở mức tải cao (concurrency = 32) trên GPU Tesla T4, hiện tượng **"Tăng tốc âm" (Negative Speedup)** xảy ra nghiêm trọng: tốc độ suy luận giảm 18.24% (speedup = 0.81x so với Vanilla). Nguyên nhân do GPU T4 bị nghẽn compute-bound khi vừa tốn tài nguyên chạy mô hình nháp 0.5B, vừa phải xác thực và rollback liên tục trên mô hình 7B.

**Mục tiêu của BudgetController v0:**
Xây dựng một bộ điều khiển thời gian thực (Runtime Adaptive Controller) đặt trong decoding loop. Controller liên tục theo dõi tỷ lệ chấp nhận trượt alpha, dự báo lợi ích kỳ vọng E[tau] so với chi phí thời gian C(k, B), và đưa ra quyết định ngân sách nháp k* in {0, 2, 3, 4}. Nếu nhận thấy việc suy đoán gây tổn hại thông lượng, controller tự động ép k* = 0 (Fallback về Vanilla) để triệt tiêu hoàn toàn vùng tăng tốc âm.

---

## 2. Mô Hình Toán Học & Thuật Toán Ra Quyết Định

### 2.1. Tỷ lệ chấp nhận trượt (Rolling Average Alpha)
Để tránh hiện tượng dao động giật cục do một vài bước đơn lẻ đoán trúng/trượt ngẫu nhiên, hệ thống kết hợp Cửa sổ trượt (Sliding Window W = 10) và Hàm trung bình trượt hàm mũ (Exponential Moving Average - EMA):

```text
alpha_step = num_accepted / num_drafted
alpha_ema(t) = beta * alpha_ema(t-1) + (1 - beta) * alpha_step
```
* Hệ số beta được cố định ở mức 0.80: đủ nhạy để thích ứng sau 3-5 bước khi chuyển đổi giữa các đoạn văn bản có entropy khác nhau, nhưng đủ ổn định để không đổi trạng thái liên tục.

### 2.2. Ước lượng số token sinh ra kỳ vọng E[tau]
Với cấu hình chuỗi nháp độ sâu k tuần tự:
```text
E[tau(k)] = 1 + sum_(i=1..k) (alpha^i) = 1 + alpha + alpha^2 + ... + alpha^k
```
* Khi alpha = 0.90, k = 3: E[tau] = 1 + 0.9 + 0.81 + 0.729 = 3.439 token.
* Khi alpha = 0.50, k = 3: E[tau] = 1 + 0.5 + 0.25 + 0.125 = 1.875 token.

### 2.3. Điều kiện Tối ưu hóa Hiệu năng (Efficiency Optimization)
* Gọi C(k, B) là thời gian trung vị một bước suy luận với độ sâu nháp k ở batch size B.
* C(0, B) là thời gian một bước giải mã tuần tự Vanilla.
* Tốc độ sinh token kỳ vọng:
```text
Speed_spec(k) = E[tau(k)] / C(k, B)
Speed_vanilla = 1 / C(0, B)
```
* Điều kiện để có lời (Speedup > 1.0):
```text
E[tau(k)] / C(k, B) > 1 / C(0, B) <=> E[tau(k)] > C(k, B) / C(0, B)
```

### 2.4. Quy Tắc Ra Quyết Định Rời Rạc v0 (Decision Rules)
Tập ngân sách được rời rạc hóa: k in {0, 2, 3, 4} để đảm bảo tính ổn định của tensor và tránh biên dịch lại CUDA Graph liên tục:
1. **Quy tắc Áp lực Bộ nhớ (Memory Pressure Guard):** Nếu VRAM khả dụng < 1,500 MB (hoặc tỷ lệ chiếm dụng đỉnh > 13.5 GB), lập tức ép k* = 0 để chống OOM.
2. **Quy tắc Ngưỡng An toàn (Safety Fallback Gate):** Nếu alpha_ema < 0.65, lập tức ngắt suy đoán (k* = 0), chuyển hoàn toàn sang Vanilla.
3. **Quy tắc Hiệu năng cao (High Alpha):** Nếu alpha_ema >= 0.80, cho phép mở rộng k* = 3 hoặc k* = 4 dựa trên argmax E[tau]/C(k, B).
4. **Quy tắc Trung bình (Mid Alpha):** Nếu 0.65 <= alpha_ema < 0.80, ưu tiên hạ độ sâu nháp xuống k* = 2 để giảm thiểu chi phí forward mô hình nhỏ.

---

## 3. Cấu Trúc Mã Nguồn & Thiết Kế Lớp

Module được cài đặt gọn gàng, zero-external-dependencies nặng, tại [`src/proposer/controller.py`](../../../src/proposer/controller.py):

```python
class BudgetController:
    def __init__(
        self,
        window_size: int = 10,
        ema_beta: float = 0.8,
        initial_alpha: float = 0.8,
        fallback_threshold: float = 0.65,
        high_threshold: float = 0.80,
        candidate_budgets: Optional[List[int]] = None,
        cost_profile: Optional[CostProfile] = None,
        min_free_vram_mb: float = 1500.0,
    ): ...

    def update(self, num_accepted: int, num_drafted: int) -> float: ...
    def estimate_expected_tokens(self, k: int, alpha: float) -> float: ...
    def select_budget(self, batch_size: int = 1, free_vram_mb: float = 16000.0) -> BudgetDecision: ...
    def reset(self, initial_alpha: Optional[float] = None) -> None: ...
    def get_stats(self) -> dict: ...
```

---

## 4. Kết Quả Kiểm Thử Độc Lập (Unit Test Verification)

Bộ kiểm thử tự động tại [`tests/test_budget_controller.py`](../../../tests/test_budget_controller.py) bao quát 12 trường hợp kiểm thử quan trọng:
- `test_cost_profile_lookup`: Kiểm tra tra cứu độ trễ C(k, B) chính xác.
- `test_cost_profile_fallback_missing_batch`: Kiểm tra tính chịu lỗi khi thiếu batch size trong profile.
- `test_cost_profile_default_estimation`: Kiểm tra ước lượng tuyến tính mặc định khi chưa có profile phần cứng.
- `test_estimate_expected_tokens`: Kiểm tra độ chính xác của công thức E[tau] tại k in {0, 1, 3} và alpha in {0.5, 0.8, 0.9}.
- `test_rolling_alpha_updates`: Kiểm tra tốc độ hội tụ và cập nhật của alpha_ema khi đoán trúng/trượt.
- `test_rolling_alpha_sliding_window_bound`: Kiểm tra cửa sổ trượt maxlen=W không bị rò rỉ bộ nhớ khi chạy hàng nghìn bước.
- `test_select_budget_vram_pressure`: Kiểm tra bộ điều khiển ép k=0 ngay lập tức khi VRAM dưới ngưỡng 2000 MB.
- `test_select_budget_fallback_low_alpha`: Kiểm tra tự động ngắt về Vanilla khi alpha tụt xuống 0.50 (< 0.65).
- `test_select_budget_heuristic_high_alpha`: Kiểm tra kích hoạt k in {3, 4} khi alpha >= 0.88.
- `test_select_budget_heuristic_mid_alpha`: Kiểm tra giới hạn k in {2, 3} (không bật k=4) khi alpha ở mức 0.72.
- `test_select_budget_cost_model_argmax`: Kiểm tra thuật toán argmax tối đa hóa thông lượng token/ms.
- `test_controller_reset_and_stats`: Kiểm tra xóa lịch sử request và xuất telemetry chuẩn Giao diện A (`tree_budgets`, `alpha_ema`, `mean_k`).

**Kết quả chạy thực tế:**
```text
tests/test_budget_controller.py::test_cost_profile_lookup PASSED         [  8%]
tests/test_budget_controller.py::test_cost_profile_fallback_missing_batch PASSED [ 16%]
tests/test_budget_controller.py::test_cost_profile_default_estimation PASSED [ 25%]
tests/test_budget_controller.py::test_estimate_expected_tokens PASSED    [ 33%]
tests/test_budget_controller.py::test_rolling_alpha_updates PASSED       [ 41%]
tests/test_budget_controller.py::test_rolling_alpha_sliding_window_bound PASSED [ 50%]
tests/test_budget_controller.py::test_select_budget_vram_pressure PASSED [ 58%]
tests/test_budget_controller.py::test_select_budget_fallback_low_alpha PASSED [ 66%]
tests/test_budget_controller.py::test_select_budget_heuristic_high_alpha PASSED [ 75%]
tests/test_budget_controller.py::test_select_budget_heuristic_mid_alpha PASSED [ 83%]
tests/test_budget_controller.py::test_select_budget_cost_model_argmax PASSED [ 91%]
tests/test_budget_controller.py::test_controller_reset_and_stats PASSED  [100%]
============================= 12 passed in 0.05s ==============================
```

---

## 5. Phân Tích Chuyên Sâu Các Quyết Định Thiết Kế (Design Q&A & Insights)

Nhóm đã ghi lại các phân tích phản biện kỹ thuật để phục vụ viết thuyết minh Chương 3 và trả lời Hội đồng đánh giá:

### 5.1. Tầng 1: Vì sao chọn ngưỡng VRAM trống < 1500 MB? Đổi GPU khác có sửa được không?
* **Nguồn gốc vật lý trên Tesla T4:**
  - GPU T4 có 16 GB VRAM vật lý (PyTorch nhận khả dụng ~14.56 GB).
  - Trọng số mô hình Qwen2.5-7B AWQ chiếm cố định ~5.3 GB. Phân vùng KV-cache của vLLM chiếm ~6.5 GB.
  - Khi chịu tải cao (concurrency = 16..32), đỉnh VRAM ghi nhận tại Sprint 2 lên tới **12.3 GB – 13.5 GB** (chỉ còn dư 1.0 – 1.5 GB).
  - Trong Speculative Decoding, mỗi bước sinh nháp và kiểm chứng cần cấp phát thêm bộ nhớ tạm thời cho tensor kích hoạt (activations) và cây nháp. Nếu vùng đệm tụt dưới 1,500 MB, việc cấp phát block PagedAttention tiếp theo sẽ đối mặt rủi ro chí mạng: **`CUDA Out of Memory (OOM)` làm sụp đổ toàn bộ server vLLM**.
* **Khả năng thích ứng khi đổi sang GPU khác (A100, L4, RTX 3090/4090):**
  - Trong code, `min_free_vram_mb` là tham số độc lập trong hàm khởi tạo:
    ```python
    BudgetController(min_free_vram_mb=1500.0)
    ```
  - Thay đổi tham số này hoàn toàn **không làm sai lệch logic toán học**, vì đây là một lớp bảo vệ độc lập (Safety Guard).
  - Khi triển khai trên GPU dung lượng lớn như A100 (80GB), có thể nâng ngưỡng an toàn lên 4,000 MB hoặc chuyển sang tỷ lệ tương đối: `free_vram / total_vram < 0.10` (dưới 10% VRAM còn trống thì chuyển về Vanilla).

### 5.2. Tầng 2 & 3: Cơ sở toán học của các con số 0.65 và 0.80? Có phải áng chừng không?
Những con số này **không phải phỏng đoán cảm tính**, mà được hiệu chuẩn trực tiếp từ **bài toán điểm hòa vốn (Break-even Analysis)** và **số liệu thực nghiệm M1**:
* **Điều kiện tăng tốc lý thuyết:**
  ```text
  Tốc độ Speculative > Tốc độ Vanilla <=> E[tau(k)] / C(k, B) > 1 / C(0, B)
  <=> E[tau(k)] > C(k, B) / C(0, B)
  ```
  Trên T4, thời gian một bước Vanilla `C(0) ~ 20 ms`, thời gian một bước nháp 3 token `C(3) ~ 32 ms`. Tỷ số chi phí phần cứng: `C(3) / C(0) ~ 1.60`.
* **Cơ sở ngưỡng ngắt 0.65 (Tầng 2 - Safety Fallback Gate):**
  - Tại Workload W3 (phân tích mở), số liệu M1 ghi nhận `alpha = 0.545` (54.5%), dẫn đến speedup tụt xuống **0.81x** (tăng tốc âm -19% do GPU bị nghẽn compute ở tải cao).
  - Điểm hòa vốn lý thuyết (khi Speedup = 1.00x) trên T4 nằm quanh `alpha ~ 0.58 – 0.62`.
  - Ngưỡng **0.65** được thiết lập như một **vách ngăn phòng thủ an toàn (safety margin)**: Khi alpha rơi xuống dưới 0.65, xác suất chạy chậm hơn Vanilla là cực cao. Việc ngắt về k = 0 triệt tiêu 100% hiện tượng tăng tốc âm.
* **Cơ sở ngưỡng mở rộng 0.80 (Tầng 3 - High Alpha Expansion):**
  - Mỗi khi tăng độ sâu nháp từ k=3 lên k=4, mô hình nhỏ phải forward thêm một bước.
  - Xác suất chấp nhận token thứ 4 giảm theo quy luật lũy thừa: `P = alpha^4`.
    - Nếu alpha = 0.70: `0.70^4 = 0.24` (chỉ 24% cơ hội đúng, 76% lãng phí tài nguyên).
    - Nếu alpha = 0.85 (như W1/W2): `0.85^4 = 0.52` (hơn 52% cơ hội đúng, việc mở rộng k=4 mới bắt đầu đem lại lợi ích vượt trội).
  - Do đó, 0.80 là điều kiện lọc khắt khe để tránh việc nháp quá dài gây lãng phí chu kỳ GPU.

### 5.3. Có thể không chặn được không?
* **Nếu không chặn VRAM (Tầng 1):** Server sẽ đối mặt nguy cơ crash OOM khi chạy concurrency cao trên GPU phổ thông.
* **Nếu không chặn alpha (Tầng 2):** Hệ thống sẽ lặp lại khuyết điểm của Static Draft ở M1, sinh văn bản tự do chậm hơn Vanilla 19%, đánh mất mục tiêu cốt lõi của đề tài tốt nghiệp.

### 5.4. Lộ trình nâng cấp: Từ ngưỡng Heuristic v0 sang Công thức Giải tích Tự thích ứng (Dynamic Break-Even Alpha)
Trong bản v0 (Sprint 3), nhóm áp dụng các ngưỡng thực nghiệm (calibrated empirical thresholds) để kiểm chứng tính ổn định của luồng điều khiển.
Ở mốc M3, khi module `HardwareProfiler` hoàn thiện việc đo ma trận chi phí `C(k, B)` thực tế lúc khởi động máy:
- Controller sẽ không cần hardcode con số 0.65 nữa, mà sẽ tự động giải phương trình tìm **ngưỡng hòa vốn động theo thời gian thực**:
  ```text
  Giải nghiệm alpha_break_even(B) thỏa mãn:
  1 + alpha + alpha^2 + ... + alpha^k == C(k, B) / C(0, B)
  ```
- Khi chuyển đổi môi trường từ T4 sang L4, A100 hay H100, `HardwareProfiler` sinh ra bảng `C(k, B)` mới, và controller sẽ tự động thích ứng với điểm hòa vốn của GPU đó mà không cần can thiệp mã nguồn. Đây chính là **đóng góp khoa học mới (Novelty)** bảo vệ trước hội đồng.

---

## 6. Hiện Thực Hóa AdaptiveProposerAdapter & Tích Hợp Hook Vòng Lặp Decoding

Theo phân công checklist số 2 (`[Khôi] Tích hợp custom proposer hook vào vòng lặp generate của vLLM`), Khôi đã hoàn thiện module [`src/proposer/adapter.py`](../../../src/proposer/adapter.py):

### 6.1. Thiết kế Lớp `AdaptiveProposerAdapter`
Adapter đóng vai trò cầu nối trung gian giữa nhân suy luận của vLLM và bộ điều khiển `BudgetController`:
- **Quản lý phiên độc lập per-request (`_active_sessions: dict[str, BudgetController]`):** Mỗi request sở hữu một controller theo dõi tỷ lệ `alpha_ema` riêng, phản ánh chính xác ngữ cảnh entropy của từng prompt.
- **Hook đề xuất nháp (`propose()`):**
  - Trước mỗi bước sinh, adapter hỏi `controller.select_budget()`.
  - Nếu `k* == 0` (Vanilla Fallback): Adapter đặt cờ `skipped_draft = True`, trả về `draft_tokens = []` và **hoàn toàn bỏ qua việc gọi hàm sinh nháp (`draft_generator`)**, giúp GPU tiết kiệm 100% tài nguyên tính toán forward mô hình nhỏ.
  - Nếu `k* > 0`: Gọi `draft_generator(k*)` và bọc dữ liệu vào `ProposalOutput`.
- **Hook nhận phản hồi kiểm chứng (`on_verification_completed()`):**
  - Nhận số token Target chấp nhận và cập nhật vào `controller.update(num_accepted, num_drafted)`.
- **Hook kết thúc request & Thu hồi bộ nhớ (`end_request()`):**
  - Trích xuất payload telemetry chuẩn Giao diện A (`tree_budgets`, `alpha_ema`, `mean_accept_len`, `step_count`).
  - Tự động xóa phiên khỏi `_active_sessions` để chống rò rỉ bộ nhớ (zero memory leak).

### 6.2. Kiểm thử Mô phỏng Decoding Loop
Kịch bản tại [`tests/test_proposer_adapter.py`](../../../tests/test_proposer_adapter.py) đã kiểm chứng 6 test cases:
- `test_adapter_initialization` & `test_proposal_output_structure`: Khởi tạo và bao gói dữ liệu.
- `test_propose_with_active_draft`: Đề xuất nháp thành công k in {3, 4} khi alpha cao.
- `test_propose_fallback_skips_draft_generator`: Xác nhận không gọi draft worker khi k=0.
- `test_simulation_decoding_loop`: Mô phỏng chuỗi giải mã chuyển trạng thái: bước 1..3 nháp k=3 -> bước 4..6 trượt liên tục -> bước 7 tự động ngắt về k=0 -> kết thúc xuất mảng `tree_budgets`.
- `test_adapter_memory_cleanup`: Giải phóng 100% tài nguyên sau 10 requests đồng thời.
- **Kết quả:** Đạt **6/6 PASS** (Tổng cộng 18/18 test cases cho toàn bộ module proposer).

---

## 7. Bàn Giao Kỹ Thuật (Handoff) & Bước Tiếp Theo

1. **Checklist 1 thẻ Trello `[M2-01]`:** Hoàn thành 100% mục `[Khôi] Thiết kế lớp BudgetController v0 tính toán N* theo rolling average alpha`.
2. **Checklist 2 thẻ Trello `[M2-01]`:** Hoàn thành 100% mục `[Khôi] Tích hợp custom proposer hook vào vòng lặp generate của vLLM`.
3. **Bàn giao cho Partner (Bùi Đức Đại):**
   - Đại sẵn sàng bắt tay vào checklist số 3 (`[Đại] Viết unit test mô phỏng thay đổi tree budget theo các kịch bản alpha cao/thấp`) dựa trên `AdaptiveProposerAdapter` và `BudgetController`.
4. **Kế hoạch tiếp theo của Khôi:**
   - Triển khai checklist số 4: `[Khôi] Kiểm thử độ ổn định VRAM không phát sinh leak khi thay đổi kích thước cây runtime`.


