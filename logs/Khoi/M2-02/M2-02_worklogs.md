# Worklog Kỹ Thuật: [M2-02] Hiện Thực Hóa Bộ Điều Khiển Cây Nháp Thích Ứng (Adaptive Controller v0)

- **Mã công việc:** `[M2-02]` (Cột mốc M2 · Adaptive v0 & Lossless Verification)
- **Người thực hiện:** Cấn Đức Khôi (Lead 60% — AI Systems & PM) & Dương Nguyễn Thế (Partner 40% — Core Engine & Data)
- **Thời gian thực hiện:** 07/10/2026 – 10/10/2026 (Ép tiến độ Sprint 3)
- **Tài nguyên kỹ thuật:**
  - Mã nguồn lõi: [`src/proposer/controller.py`](../../../src/proposer/controller.py)
  - Lớp adapter vLLM: [`src/proposer/adapter.py`](../../../src/proposer/adapter.py)
  - Kiểm thử tự động: [`tests/test_budget_controller.py`](../../../tests/test_budget_controller.py) & [`tests/test_proposer_adapter.py`](../../../tests/test_proposer_adapter.py)
  - Launcher Làn 3: [`scripts/launch_lane3_adaptive.sh`](../../../scripts/launch_lane3_adaptive.sh)
- **Trạng thái:** In Progress (Đang thực hiện phân rã checklist Khôi & Dương)

---

## 1. Mục Tiêu Kỹ Thuật & Phân Chia Công Việc Cặp 60/40

Mục tiêu cốt lõi của task M2-02 là hoàn thiện trọn vẹn bộ điều khiển runtime `AdaptiveDraftController` cho vLLM, đảm bảo thích ứng động theo từng bước giải mã và loại bỏ hoàn toàn hiện tượng tăng tốc âm:

### 1.1. Khôi (Lead 60%):
1. **Xây dựng lớp `AdaptiveDraftController`:** Kế thừa và mở rộng từ `BudgetController`, chuyên biệt hóa cửa sổ trượt đúng 10 bước (`window_size = 10`), tính toán `alpha_ema` kết hợp cơ chế Warm-up Unbiased.
2. **Cài đặt Budget Rule:** Co giãn độ sâu nháp `k in {0, 2, 3, 4}` theo các ngưỡng thực nghiệm:
   - `alpha_ema >= 0.80`: Duy trì `k = 3` hoặc mở rộng `k = 4`.
   - `0.65 <= alpha_ema < 0.80`: Hạ độ sâu nháp xuống `k = 2` để giảm tải GPU.
   - `alpha_ema < 0.65`: Ngắt hoàn toàn suy đoán (`k = 0`), chuyển về Vanilla để chống tụt tốc độ.
3. **Smoke Test 10 requests:** Kiểm chứng chuyển trạng thái động của mảng `tree_budgets` và cập nhật tài liệu Chương 3.

### 1.2. Dương (Partner 40%):
1. **Cài đặt Memory Pressure Rule:** Giám sát tỷ lệ chiếm dụng KV-cache và VRAM Peak; nếu VRAM Peak > 13.5 GB hoặc KV-cache trống < 10%, tự động ép `k = 0` chống OOM.
2. **Đóng gói Launcher Làn 3:** Viết script [`scripts/launch_lane3_adaptive.sh`](../../../scripts/launch_lane3_adaptive.sh) mở cổng `8003` tích hợp telemetry adapter.

---

## 2. Phân Tích Chuyên Sâu Các Câu Hỏi Kiến Trúc & Chiến Lược Đồ Án (Design Q&A & Strategic Insights)

Nhóm đã ghi lại các phân tích phản biện kỹ thuật sâu sắc giữa Khôi và AI Assistant nhằm phục vụ trực tiếp cho báo cáo và bảo vệ trước Hội đồng:

### 2.1. Phân biệt `BudgetController` vs `AdaptiveDraftController`: Có phải chỉ khác nhau ở Warm-up?
* **Khái niệm theo lý thuyết tổng quát (EAGLE Tree Speculative Decoding):**
  - Trong phương pháp cây nháp (Tree-based Speculative Decoding như EAGLE-2), mô hình nháp sinh ra một cây gồm nhiều nhánh rẽ song song. Tổng số lượng node (tổng số token trên toàn cây) được gọi là **Ngân sách cây (Tree Budget N)**, ví dụ `N in {0, 8, 16, 32, 60}`.
  - Target model dùng cơ chế *Tree Attention* để xác thực đồng thời toàn bộ `N` token này trong 1 lượt forward.
  - 👉 **`BudgetController`** trong tài liệu gốc [`docs/05-thiet-ke-he-thong.md`](../../../docs/05-thiet-ke-he-thong.md) là bộ não đưa ra quyết định vĩ mô: **Nên chọn ngân sách cây `N*` bằng bao nhiêu** dựa trên mô hình chi phí phần cứng.
* **Khái niệm theo thực tế vLLM ở Sprint 3 (Rủi ro kỹ thuật R3.1):**
  - Nhân vLLM hiện tại chưa cho phép thay đổi cấu trúc cây topology phức tạp động theo từng bước qua `custom_proposer`.
  - Nhóm quyết định giải pháp kỹ thuật cho mốc M2 v0: **"Ép phẳng" cây nháp thành chuỗi tuần tự (Single-branch degenerate tree)**. Khi đó, cây chỉ có 1 nhánh thẳng, nên:
    ```text
    Ngân sách cây N* == Độ sâu nháp k* in {0, 2, 3, 4}
    ```
  - 👉 **`AdaptiveDraftController`** chính là lớp runtime cụ thể hóa của `BudgetController` cho vLLM, điều phối trực tiếp từng bước xem **nên nháp mấy token (`k`)**.
* **Bốn điểm khác biệt cụ thể:**
  1. *Theo dõi 2 cấp độ:* `AdaptiveDraftController` theo dõi song song mảng 10 giá trị trượt nguyên thủy `[alpha_t-9, ..., alpha_t]` (để đo độ phân tán, xu hướng trend) và chỉ số mượt mà `alpha_ema`.
  2. *Xử lý Warm-up Unbiased:* Trong 1–3 bước đầu tiên khi cửa sổ chưa đầy, EMA thông thường bị quán tính kéo về `initial_alpha = 0.8`. `AdaptiveDraftController` áp dụng hiệu chỉnh thiên lệch `alpha_unbiased = alpha_ema / (1 - beta^step)` hoặc SMA để nếu gặp câu W3 trượt ngay từ đầu, hệ thống nhận diện và ngắt về `k = 0` sớm hơn 2-3 bước.
  3. *Tách bạch API phân vai:* Tách riêng `apply_budget_rule()` (Khôi làm) và `apply_memory_pressure_rule()` (Dương cắm hook) thay vì gộp chung một khối if/else.
  4. *Khả năng tự chẩn đoán:* Cung cấp `get_window_stats()` (mean, min, max, variance, is_warmed_up).

---

### 2.2. Vấn đề mô hình nháp: Dùng `Qwen2.5-0.5B` thay vì EAGLE ở Sprint 1 & 2 có tương đương và thay thế được không?
* **Về mặt lý thuyết khoa học:**
  - Phương thức dùng mô hình nhỏ độc lập (`Qwen2.5-0.5B-Instruct`) kết hợp mô hình lớn (`Qwen2.5-7B-Instruct-AWQ`) chính là **dạng thức nguyên bản, chuẩn mực kinh điển nhất (Standard Speculative Decoding)** của ngành công nghiệp (được định nghĩa trong paper gốc của Leviathan et al., 2023).
  - EAGLE thực chất chỉ là một biến thể kiến trúc đầu nháp (Draft Head). Cả hai đều tuân theo cùng một vòng lặp: Đề xuất chuỗi token nháp -> Target Model verify -> Rollback hoặc Accept.
* **Vì sao không dùng checkpoint EAGLE-2 của Qwen2.5 lúc này?**
  - Trong thực nghiệm Sprint 1 ([`logs/Khoi/M0-02/M0-02_worklogs.md`](../../../logs/Khoi/M0-02/M0-02_worklogs.md#L45-L53)):
    - Nạp `leptonai/EAGLE-Qwen2.5-7B-Instruct` vào vLLM bị lỗi `ValidationError: Model architectures ['EagleQwen2ForCausalLM'] are not supported`.
    - Chuyển sang SGLang thì PyTorch CUDA 13 đã loại bỏ mã nhị phân cho GPU Turing `sm_75` của Tesla T4 (`cudaErrorNoKernelImageForDevice`).
  - Do đó, **Draft Model 0.5B là lựa chọn duy nhất chạy ổn định 100% trên vLLM cho GPU T4**. Bảng số liệu M1 hoàn toàn minh bạch, chuẩn xác và có giá trị khoa học vững chắc.

---

### 2.3. Chiến Lược Hai Luồng Song Song (Dual-Track Strategy): Bảo Vệ Tiến Độ & Huấn Luyện Mở Rộng EAGLE-3.1

Sau khi thảo luận sâu sắc, Team Lead (Khôi) đã chính thức chốt **Quyết định Chiến lược Hai Luồng Song Song (Dual-Track Pivot)**:

```text
                               CHIẾN LƯỢC HAI LUỒNG SONG SONG (DUAL-TRACK)
+----------------------------------------------------------------------------------------------------+
| LUỒNG 1: MAIN PRODUCTION TRACK (ĐƯỜNG GĂNG TỐT NGHIỆP - CRITICAL PATH)                              |
| - Cấu hình: Qwen2.5-7B-AWQ + Draft Model Qwen2.5-0.5B trên GPU T4 16GB.                            |
| - Tiến độ: Bám sát Sprint 3 (M2 Adaptive Controller v0 & Lossless Verification).                  |
| - Số liệu: Kế thừa 36 ô thực nghiệm M1, hoàn thiện full 54 ô đối chứng 3 làn M2.                  |
| - Mục tiêu: ĐẢM BẢO 100% TIẾN ĐỘ BẢO VỆ TỐT NGHIỆP, an toàn, ổn định production.                   |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  │ Chạy song song không chặn tiến độ
                                                  ▼
+----------------------------------------------------------------------------------------------------+
| LUỒNG 2: R&D / OUTSOURCE EFFORT TRACK (TỰ HUẤN LUYỆN EAGLE-3.1 CHO QWEN2.5)                       |
| - Mục tiêu: Tự train 1 Draft Speculator Head EAGLE-3.1 chuyên biệt cho Qwen2.5-7B.                |
| - Hạ tầng: Mượn/thuê GPU A100 (theo QĐ-02), sử dụng framework SafeAILab/EAGLE hoặc SpecForge.      |
| - Vai trò trong đồ án:                                                                             |
|   1. Trình diễn Demo tốc độ bứt phá tại buổi bảo vệ tốt nghiệp.                                    |
|   2. Minh chứng kỹ năng MLOps / Model Training chuyên sâu của nhóm sinh viên PTIT.                 |
|   3. Cung cấp bài đo Ablation Study so sánh chéo: Đánh giá bộ điều khiển AdaptiveDraftController   |
|      khi cắm vào Draft Model độc lập 0.5B vs khi cắm vào EAGLE-3.1 Head tự train.                  |
+----------------------------------------------------------------------------------------------------+
```

* **Vì sao tách 2 luồng là quyết định đúng đắn nhất?**
  1. *Triệt tiêu rủi ro vỡ tiến độ:* Nếu dồn toàn bộ nguồn lực vào việc train EAGLE-3.1 ngay lúc này, chỉ cần checkpoint train bị lỗi convergence, lệch loss hoặc kernel vLLM chưa tương thích là toàn bộ Sprint 3 sẽ bị trễ hạn. Tách luồng 2 thành "luồng ngoài" giúp Luồng 1 về đích đúng hẹn 100%.
  2. *Bảo toàn 100% giá trị số đo M1 & M2:* Số liệu đo trên Draft Model 0.5B là luận cứ khoa học chính thức cho Chương 4. Số liệu của luồng EAGLE-3.1 sẽ là phần điểm cộng vượt trội (Extra Academic Value) trong Chương 5.
  3. *Chứng minh tính độc lập của kiến trúc (Architectural Independence):* Nhờ thiết kế tách rời (Decoupling) của `AdaptiveDraftController` và `AdaptiveProposerAdapter`, cả 2 luồng đều dùng chung 100% bộ điều khiển của Khôi. Khi checkpoint EAGLE-3.1 train xong, nhóm chỉ cần cắm vào launcher mà không phải sửa một dòng logic điều khiển nào!

---

## 3. Kế Hoạch Triển Khai Tiếp Theo (Implementation Roadmap)

1. **Checklist 1 M2-02 (Khôi):** Cài đặt lớp `AdaptiveDraftController(BudgetController)` trong [`src/proposer/controller.py`](../../../src/proposer/controller.py) với cửa sổ trượt 10 bước và Warm-up Unbiased. (✅ Hoàn thành 10/10/2026)
2. **Checklist 2 M2-02 (Khôi):** Tách bạch phương thức `apply_budget_rule()` với các ngưỡng `0.65` và `0.80`. (✅ Hoàn thành 10/10/2026)
3. **Checklist 3 & 4 M2-02 (Dương):** Cài đặt Memory Pressure Rule và Launcher Làn 3 `launch_lane3_adaptive.sh`.
4. **Viết Unit Test kiểm chứng:** Bổ sung test cases chuyên sâu trong [`tests/test_budget_controller.py`](../../../tests/test_budget_controller.py) cho `AdaptiveDraftController`. (✅ 25/25 tests PASS 100%)
5. **Kích hoạt Luồng 2 (R&D Track):** Chuẩn bị kịch bản training EAGLE-3.1 trên môi trường GPU A100.

---

## 4. Báo Cáo Nghiệm Thu Checklist 1 & 2 (Khôi 60% — Hoàn Thành)

### 4.1. Mã nguồn triển khai
Đã cài đặt lớp `AdaptiveDraftController` trong [`src/proposer/controller.py`](../../../src/proposer/controller.py) và export tại [`src/proposer/__init__.py`](../../../src/proposer/__init__.py):
- **Cửa sổ trượt 10 bước (`window_size = 10`):** Sử dụng `collections.deque(maxlen=10)` lưu trữ các mẫu chấp nhận thực tế `step_alpha in [0.0, 1.0]`. Cung cấp phương thức `get_window_samples() -> list[float]` để phục vụ phân tích phương sai.
- **Cơ chế Warm-up Unbiased:**
  - Trong các bước đầu khi `step_count < window_size`, loại bỏ hoàn toàn thiên lệch khởi tạo `initial_alpha` theo công thức:
    ```text
    alpha_unbiased = (alpha_ema - initial_alpha * (beta^step_count)) / (1.0 - beta^step_count)
    ```
  - Nếu gặp prompt W3 khó khiến bước 1 trượt sạch (`0/3 accepted`), `alpha_unbiased` rơi thẳng về `0.0`, kích hoạt ngắt về `k = 0` (Vanilla Fallback) ngay lập tức thay vì bị ghìm ở mức `0.64` và mất thêm 2–3 bước tụt tốc độ.
- **Phương thức tự chẩn đoán `get_window_stats()`:**
  - Trả về payload giám sát: `mean`, `min`, `max`, `variance`, `is_warmed_up` (`step_count >= window_size`), `samples`, `step_count`.
- **Modular Budget Rule (`apply_budget_rule`):**
  - `alpha >= 0.80`: Duy trì `k = 3` (hoặc mở rộng `k = 4` khi `alpha >= 0.90`).
  - `0.65 <= alpha < 0.80`: Hạ độ sâu nháp xuống `k = 2` để giảm chi phí tính toán GPU.
  - `alpha < 0.65`: Ngắt toàn bộ về `k = 0` (Vanilla Fallback).
- **Hook sẵn sàng cho Dương (`apply_memory_pressure_rule`):**
  - Mặc định kiểm tra `free_vram_mb < min_free_vram_mb`. Tách riêng làm hook độc lập để Dương cắm thêm kiểm tra `kv_cache_usage_pct > 90%` và `peak_vram > 13.5 GB`.
- **Tương thích ngược Adapter:**
  - Lớp [`AdaptiveProposerAdapter`](../../../src/proposer/adapter.py) hỗ trợ tham số `controller_cls` (mặc định trỏ thẳng vào `AdaptiveDraftController`), tích hợp telemetry mở rộng `effective_alpha` và `is_warmed_up`.

### 4.2. Kết quả kiểm thử tự động (Unit Tests)
Đã chạy toàn bộ bộ kiểm thử với kết quả **25/25 test cases PASS 100%**:
- `tests/test_budget_controller.py`: 18/18 tests PASS (bao gồm 6 tests mới cho `AdaptiveDraftController`: kế thừa, theo dõi cửa sổ trượt, warm-up unbiased, thống kê phương sai, quy tắc ngân sách, và luồng select_budget).
- `tests/test_proposer_adapter.py`: 7/7 tests PASS (kiểm chứng tương thích phiên per-request, skip draft khi k=0, mô phỏng vòng lặp decode, giải phóng bộ nhớ, và tích hợp `AdaptiveDraftController`).

### 4.3. Cập nhật thẻ Trello
- Thẻ `[M2-02]` (`6ac462bf8076e675bed68739`):
  - ✅ `[Khôi] Xây dựng lớp AdaptiveDraftController theo dõi alpha_ema qua cửa sổ trượt 10 bước`: `complete`.
  - ✅ `[Khôi] Cài đặt Budget Rule co giãn k = 0, 2, 3, 4 theo các ngưỡng alpha (0.80, 0.65)`: `complete`.
- Trạng thái hiện tại: 2/5 checkItems hoàn thành. Sẵn sàng handoff cho Dương làm Checklist 3 (`Memory Pressure Rule`) và Checklist 4 (`Launcher Làn 3`).

---

## 5. Thống Nhất Kỹ Thuật Với Dương & Xử Lý 2 Lỗi Trọng Yếu (Critical Fixes — 11/10/2026)

### 5.1. Thống nhất kế hoạch với Dương (Partner 40% — Core Engine & Data)
Nhóm đã đối chiếu phản hồi kỹ thuật từ Dương và chốt các điểm then chốt:
1. **Kiến trúc Launcher Làn 3 trên vLLM 0.30.0:**
   - Xác nhận vLLM 0.30.0 (V1 Engine) không còn file `proposer_worker.py` (vốn thuộc engine V0 cũ).
   - Tên metric chính xác phơi qua Prometheus `/metrics` là `vllm:kv_cache_usage_perc`.
   - Đồng thuận phương án của Dương: Xây dựng `AdaptiveScheduler` thông qua điểm mở rộng `scheduler_cls` (`vllm/config/scheduler.py:159`), kết hợp cơ chế dynamic Speculative Decoding (`num_speculative_tokens_per_batch_size`).
2. **Quy tắc Memory Pressure trên GPU T4 (16GB):**
   - Trên GPU T4, do vLLM pre-allocate 85% VRAM khi khởi động (`gpu_memory_utilization = 0.85`), dung lượng `free_vram` lúc rảnh thường đã rơi xuống dưới 1500 MB.
   - Do đó, đối với launcher Làn 3 trên T4, cho phép tắt điều kiện `free_vram < 1500 MB`, lấy `vllm:kv_cache_usage_perc > 90%` làm nguồn chân lý duy nhất (source-of-truth).
   - Hành vi chuẩn khi áp lực xảy ra: Ép `k = 0` (ngắt suy đoán nhị phân), áp dụng cooldown 3 bước, và chỉ phục hồi suy đoán khi KV-cache giảm xuống dưới 80%.
3. **Phê duyệt Spike Colab:** Khôi (Lead) phê duyệt cho Dương thực hiện phiên chạy thử nghiệm spike trên Colab 2–3 giờ trong ngày 12/10 để kiểm chứng tích hợp `AdaptiveScheduler` với vLLM.

### 5.2. Sửa Lỗi Chí Mạng 1: Thoát Bẫy Bế Tắc (Deadlock Bug) Bằng Periodic Probing
- **Bản chất lỗi:**
  - Khi `alpha < 0.65`, bộ điều khiển kích hoạt Vanilla Fallback (`k = 0`).
  - Khi `k = 0`, worker bỏ qua bước sinh nháp (`skipped_draft = True`), không thực hiện kiểm chứng nháp nên không có phản hồi mới từ Target Model.
  - Hệ quả: Chỉ số `alpha_ema` bị đóng băng vĩnh viễn ở mức `< 0.65`, khiến request bị kẹt cứng ở chế độ Vanilla suốt toàn bộ phần còn lại của câu giải mã dù các token tiếp theo có thể rất dễ đoán.
- **Giải pháp xử lý (Periodic Probing):**
  - Cập nhật phương thức `apply_budget_rule()` trong [`src/proposer/controller.py`](../../../src/proposer/controller.py): Bổ sung hai tham số `probe_interval = 5` và `probe_k = 2`.
  - Bộ đếm `consecutive_fallback_steps` theo dõi số bước fallback liên tiếp. Cứ sau đúng 5 bước fallback (`k = 0`), bộ điều khiển chủ động phát động một bước thăm dò nháp nhỏ với `probe_k = 2`.
  - Nếu bước thăm dò mang lại kết quả chấp nhận tốt khiến `alpha` phục hồi `>= 0.65`, bộ đếm được reset về 0 và hệ thống lập tức thoát khỏi fallback để tiếp tục suy đoán bình thường.

### 5.3. Sửa Lỗi Chí Mạng 2: Chuẩn Hóa Telemetry Giao Diện A (`mean_accept_len` vs `mean_k`)
- **Bản chất lỗi:**
  - Trong [`src/proposer/adapter.py`](../../../src/proposer/adapter.py), trường `mean_accept_len` trước đó lấy nhầm `stats.get("mean_k", 0.0)`.
  - `mean_k` phản ánh trung bình ngân sách đề xuất (`k`), trong khi hợp đồng Giao diện A quy định `mean_accept_len` phải là trung bình số token thực tế được Target Model chấp nhận (`num_accepted` hay `tau`).
- **Giải pháp xử lý:**
  - Bổ sung cấu trúc dữ liệu `_accepted_history: Dict[str, List[int]]` trong adapter để ghi nhận chính xác mảng `num_accepted` qua từng bước có sinh nháp (`num_drafted > 0`).
  - Trong `get_telemetry()`: Tính toán `mean_accept_len` là trung bình cộng của mảng `_accepted_history[request_id]`, đồng thời giữ riêng trường `mean_k` (`stats.get("mean_k", 0.0)`).
  - Trong `end_request()`: Giải phóng triệt để cả `_active_sessions[request_id]` lẫn `_accepted_history[request_id]`, đảm bảo an toàn bộ nhớ và ngăn chặn rò rỉ RAM/VRAM theo Finding 3 của Đại.

### 5.4. Kết Quả Kiểm Thử Toàn Diện (Unit Tests Parity)
Đã chạy kiểm chứng tự động toàn bộ 4 file test liên quan với kết quả **40/40 tests PASS 100%**:
1. `tests/test_budget_controller.py`: 19/19 tests PASS (bổ sung test `test_adaptive_draft_controller_periodic_probing_prevents_deadlock`).
2. `tests/test_proposer_adapter.py`: 8/8 tests PASS (bổ sung test `test_adapter_telemetry_semantics_mean_accept_len_vs_mean_k`).
3. `tests/test_budget_transition_scenarios.py`: 5/5 tests PASS.
4. `tests/test_budget_fallback_recovery.py`: 8/8 tests PASS (cập nhật test `test_adapter_periodic_probing_triggers_and_recovers` chứng minh cơ chế probing giải phóng bế tắc).

*Ghi chú:* Toàn bộ thay đổi mã nguồn và tài liệu được lưu giữ trong working directory (chưa chạy `git commit`), sẵn sàng để Team Lead xem xét `git diff` trước khi commit.


