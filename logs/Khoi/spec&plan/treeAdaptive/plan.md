# Kế hoạch triển khai nâng cấp TreeAdaptive từ phiên bản v0.1 lên v0.2

> **Tài liệu:** `plan.md`  
> **Dự án:** Kế hoạch thực thi nâng cấp bộ điều khiển thích ứng TreeAdaptive (Sprint 5 / M4)  
> **Thành viên:** Cấn Đức Khôi (Lead 70%), Bùi Đức Đại (Reviewer 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptive/plan.md`  
> **Hạn chốt hoàn thành:** 12/11/2026 (trước mốc đóng băng toàn diện 15/11/2026)  

---

## 1. Mục tiêu và lộ trình tổng thể

Kế hoạch này vạch ra các bước kỹ thuật chi tiết để hiện thực hóa bản nâng cấp **TreeAdaptive v0.2**, đưa toàn bộ các nghiên cứu về ngưỡng hòa vốn động $\alpha^*(N, B)$, mô hình chi phí thực nghiệm phần cứng và cấu trúc cây phân nhánh vào mã nguồn phục vụ chính thức của đồ án FlexSDS.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      LỘ TRÌNH THỰC HIỆN CÁC TASK                       │
├────────────────────────────────────────────────────────────────────────┤
│ [Task 1] Nâng cấp mô hình chi phí và bộ giải ngưỡng động alpha*(N, B)   │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 2] Hiện thực hóa module cấu trúc cây nháp (tree_topology.py)     │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 3] Tái cấu trúc TreeAdaptiveController v0.2 (core/controller.py) │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 4] Nâng cấp bộ đệm lập lịch và tích hợp vLLM AsyncScheduler     │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 5] Xây dựng bộ kiểm thử tự động toàn diện (unit & regression)    │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 6] Đo kiểm thực nghiệm đối chứng v0.1 vs v0.2 trên Modal GPU L4  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết phân rã công việc (Task breakdown)

### Task 1: Nâng cấp mô hình chi phí và bộ giải nghiệm hòa vốn động $\alpha^*(N, B)$
* **Mục tiêu:**
  * Chuyển hóa lớp `FlexCostModel` trong `src/flexsds/core/cost_model.py` để hỗ trợ tra cứu chi phí theo cấu trúc cây $C(N, B)$.
  * Hiện thực hóa thuật toán tìm kiếm nhị phân 50 bước để giải nhanh ngưỡng hòa vốn $\alpha^*(N, B)$ dựa trên hàm kỳ vọng token chấp nhận $\mathbb{E}[\tau(N, \alpha)]$.
  * Tích hợp hệ số phạt xác thực song song qua attention mask $\gamma_{\text{attn}}$ theo kiến trúc phần cứng (sm_89 và sm_75).
* **Các tệp cần tạo/sửa đổi:**
  * `src/flexsds/core/cost_model.py`
  * `tests/test_cost_model_solver.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Thời gian giải $\alpha^*$ cho một cấu hình $< 0.01\text{ ms}$.
  - Với $B=1$ trên GPU L4, $\alpha^*$ giải ra nằm trong khoảng hợp lý $[0.55, 0.65]$. Với $B=32$, $\alpha^*$ tự động tịnh tiến lên khoảng $[0.75, 0.82]$.
  - 100% Pass các kiểm thử đơn vị với sai số tuyệt đối $< 10^{-4}$.
* **Người thực hiện:** Khôi (Lead) — **Thời gian:** 3 giờ.

---

### Task 2: Hiện thực hóa module cấu trúc topo cây nháp (`tree_topology.py`)
* **Mục tiêu:**
  * Xây dựng module quản lý và biểu diễn các cấu hình cây rời rạc trong không gian $(d, b, N)$.
  * Xây dựng danh mục các topo chuẩn hóa: $T_0$ (Vanilla), $T_4$ (Chuỗi sâu), $T_8$ (Cây cân bằng), $T_{16}$ (Cây rộng), $T_{32}$ (Cây mở rộng).
  * Viết hàm sinh ma trận mặt nạ chú ý 2D (Tree Attention Mask) và mảng chỉ số vị trí (Tree Position IDs) sẵn sàng tương thích với kernel xác thực của vLLM và đầu nháp EAGLE-3.1.
  * Cung cấp bảng tra cứu nhanh hàm kỳ vọng $\mathbb{E}[\tau(T, \alpha)]$ tránh tính toán lại nhiều lần.
* **Các tệp cần tạo:**
  * `src/flexsds/core/tree_topology.py`
  * `tests/test_tree_topology.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Khởi tạo đúng các cấu hình cây chuẩn với tổng số node $N$ khớp với định nghĩa.
  - Ma trận Tree Attention Mask sinh ra đúng chuẩn tam giác dưới mở rộng (causal mask with tree branching), không bị rò rỉ token tương lai giữa các nhánh song song.
  - Hàm tính kỳ vọng trả về kết quả tăng đơn điệu theo $\alpha$.
* **Người thực hiện:** Khôi (Lead), Đại (Review thuật toán mặt nạ) — **Thời gian:** 4 giờ.

---

### Task 3: Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 (`core/controller.py`)
* **Mục tiêu:**
  * Thay thế logic heuristic ngưỡng tĩnh cũ bằng bộ điều khiển đa tầng v0.2.
  * Tích hợp cơ chế phòng vệ hai tầng (Memory Guard): VRAM vật lý $< 1500\text{ MB}$ hoặc KV-cache $> 90\%$ đi kèm cơ chế trễ (Hysteresis) giữ 3 bước cooldown và phục hồi ở mức $80\%$.
  * Tích hợp quy tắc kẹp trần kích thước cây theo luồng đồng thời (Concurrency Throttling) khi $B \ge 16$.
  * Tích hợp bộ giải ngưỡng động $\alpha^*$ để ra quyết định ngắt về $N=0$ khi chất lượng nháp không đạt điểm hòa vốn.
  * Hiện thực hóa cơ chế thăm dò chu kỳ thích ứng tải $T_{\text{probe}}(B)$ để thoát khỏi trạng thái đóng băng tín hiệu.
* **Các tệp cần sửa đổi:**
  * `src/flexsds/core/controller.py`
  * `tests/test_tree_adaptive_controller.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Ra quyết định chọn cấu hình cây tối ưu $T^*$ trong thời gian $< 0.05\text{ ms}$ cho mỗi lượt gọi.
  - Khi giả lập điều kiện bộ nhớ căng thẳng, bộ điều khiển lập tức trả về $N=0$ với lý do `Memory pressure`.
  - Khi $\alpha$ tụt xuống dưới $\alpha^*$, chuyển về $N=0$ và kích hoạt thăm dò đúng chu kỳ $T_{\text{probe}}(B)$.
  - Đạt 100% Pass các kịch bản kiểm thử chuyển trạng thái (State Transition Scenarios).
* **Người thực hiện:** Khôi (Lead) — **Thời gian:** 4.5 giờ.

---

### Task 4: Nâng cấp bộ lập lịch và tích hợp vLLM AsyncScheduler
* **Mục tiêu:**
  * Nâng cấp `AdaptiveBudgetPlanner` trong `src/server/budget_planner.py` để hỗ trợ nhận diện và lập kế hoạch cho cấu trúc cây $N_i$ cho từng request.
  * Cập nhật `AdaptiveAsyncScheduler` trong `src/server/adaptive_scheduler.py` để đồng bộ hóa kích thước cây lớn nhất trong batch $N_{\text{batch}} = \max(N_i)$ vào bảng tra cứu `dynamic_sd_lookup` của vLLM.
  * Đảm bảo việc cắt tỉa placeholder nháp (per-request trimming) hoạt động chính xác với kích thước cây của từng luồng.
  * Cập nhật định dạng log telemetry theo chuẩn Interface A v0.2.
* **Các tệp cần sửa đổi:**
  * `src/server/budget_planner.py`
  * `src/server/adaptive_scheduler.py`
  * `src/proposer/adapter.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - `AdaptiveBudgetPlanner` chạy độc lập không cần GPU, vượt qua toàn bộ unit test mô phỏng multi-request batch.
  - Không gây xung đột luồng khi chạy vLLM ở chế độ async scheduling.
  - Tệp log telemetry ghi nhận đầy đủ các trường `dynamic_alpha_star`, `selected_topology` và `decision_reason`.
* **Người thực hiện:** Khôi (Lead), Đại (Hỗ trợ tích hợp scheduler) — **Thời gian:** 4 giờ.

---

### Task 5: Xây dựng bộ kiểm thử tự động toàn diện (Test suite & Regression tests)
* **Mục tiêu:**
  * Thiết kế bộ kiểm thử hồi quy bao phủ toàn bộ các module từ tầng toán học đến tầng serving.
  * Kiểm tra tính toàn vẹn đầu ra (Lossless verification): đảm bảo việc đổi cây không làm biến đổi token được sinh ra.
  * Kiểm tra độ ổn định khi chuyển làn liên tục giữa Vanilla, Static và Adaptive v0.2.
* **Các tệp kiểm thử cần xây dựng:**
  * `tests/test_tree_adaptive_v02_comprehensive.py`
  * `tests/test_lossless_tree_verification.py`
  * `tests/test_scheduler_tree_glue.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Chạy `pytest tests/` đạt 100% Pass toàn bộ test suite hiện có và các bài test mới (dự kiến $> 200$ tests).
  - Không có cảnh báo bộ nhớ hoặc memory leak trong các bài test chạy vòng lặp 1.000 bước.
* **Người thực hiện:** Đại (Lead 70%), Khôi (Hỗ trợ 30%) — **Thời gian:** 3.5 giờ.

---

### Task 6: Đo kiểm thực nghiệm đối chứng v0.1 vs v0.2 trên Modal GPU L4 và lập báo cáo
* **Mục tiêu:**
  * Khởi chạy chiến dịch đo kiểm đối chứng trên nền tảng Modal (GPU NVIDIA L4 24GB).
  * Chạy sweep toàn diện trên cả 3 tập Workload (W1, W2, W3) với các mức concurrency $c \in \{1, 4, 8, 16, 32\}$.
  * Thu thập số liệu TPOT, Throughput, TTFT, tỉ lệ $\alpha$ và vẽ biểu đồ so sánh giữa 4 chế độ:
    1. Vanilla cơ sở ($1.0\times$).
    2. Static Draft cố định ($k=3$).
    3. Adaptive v0.1 (Linear chain).
    4. TreeAdaptive v0.2 (Cây thích ứng).
  * Đóng gói số liệu vào bảng tổng hợp phục vụ Chương 4 của quyển báo cáo tốt nghiệp.
* **Các tệp cần cập nhật:**
  * `experiments/runs/m4_treeadaptive/`
  * `experiments/results/summary_treeadaptive_v02.csv`
  * `docs/chuong-4-thuc-nghiem.md`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - TreeAdaptive v0.2 đạt tốc độ cao hơn hoặc bằng v0.1 trên tất cả 36 ô đo lường.
  - Trên Workload W3 ở $c=32$, TreeAdaptive v0.2 duy trì hệ số tăng tốc $\ge 1.00\times$, triệt tiêu hoàn toàn mức tụt $0.81\times$ của Static Draft.
  - Hoàn tất bảng số liệu và đóng băng toàn bộ mã nguồn trước ngày 15/11.
* **Người thực hiện:** Đại (Lead đo đạc 70%), Khôi (Phân tích log và fix bug 30%) — **Thời gian:** 5 giờ.

---

## 3. Kế hoạch phối hợp và ma trận trách nhiệm (RACI Matrix)

Tuân thủ nguyên tắc ghép cặp luân phiên 70/30 đã cam kết trong đề tài:

| Hạng mục công việc | Cấn Đức Khôi | Bùi Đức Đại | Vai trò phối hợp |
| :--- | :---: | :---: | :--- |
| **Task 1: Nâng cấp mô hình chi phí & solver** | **A / R (70%)** | C / I (30%) | Khôi code thuật toán; Đại thẩm định tính đúng đắn toán học |
| **Task 2: Hiện thực hóa cấu trúc topo cây** | **A / R (70%)** | C / I (30%) | Khôi code cấu trúc dữ liệu; Đại review ma trận attention mask |
| **Task 3: Tái cấu trúc TreeAdaptiveController** | **A / R (70%)** | C / I (30%) | Khôi hoàn thiện logic điều khiển; Đại chạy thử nghiệm độc lập |
| **Task 4: Tích hợp AsyncScheduler vLLM** | **A / R (70%)** | C / I (30%) | Khôi phụ trách glue code; Đại kiểm tra tính tương thích endpoint |
| **Task 5: Xây dựng bộ test suite & regression** | C / I (30%) | **A / R (70%)** | Đại thiết kế ca kiểm thử; Khôi hỗ trợ sửa lỗi nếu phát sinh |
| **Task 6: Đo kiểm sweep thực nghiệm trên L4** | C / I (30%) | **A / R (70%)** | Đại điều phối harness đo lường; Khôi trực xử lý sự cố server |

*Chú thích ma trận:*  
* **R (Responsible):** Người trực tiếp triển khai chính.  
* **A (Accountable):** Người chịu trách nhiệm cao nhất về chất lượng đầu ra.  
* **C (Consulted):** Người được tham vấn chuyên môn và cùng review mã nguồn.  
* **I (Informed):** Người được cập nhật tiến độ và kết quả bàn giao.  

---

## 4. Quản trị rủi ro và phương án dự phòng (Risk mitigation)

1. **Rủi ro 1: Chi phí sinh mặt nạ chú ý cây làm tăng độ trễ bước giải mã:**
   * *Nguyên nhân:* Tính toán ma trận attention mask 2D động cho từng request có thể tốn thời gian tính toán của CPU scheduler.
   * *Phương án giải quyết:* Tiền tính toán (Pre-compute) toàn bộ các ma trận mặt nạ chú ý cho 5 cấu hình cây chuẩn ($T_0, T_4, T_8, T_{16}, T_{32}$) và nạp vào bộ nhớ dưới dạng tensor hằng số (Static Buffers). Quá trình phục vụ chỉ cần tra cứu con trỏ bảng trang mà không phải cấp phát bộ nhớ mới.

2. **Rủi ro 2: vLLM V1 Engine không hỗ trợ co giãn cây động theo từng request riêng lẻ:**
   * *Nguyên nhân:* Phiên bản vLLM v0.30.0 có thể yêu cầu kích thước tensor nháp đồng nhất trong cùng một micro-batch.
   * *Phương án giải quyết:* Sử dụng cơ chế gộp batch bằng giá trị trần $N_{\text{batch}} = \max(N_i)$ thông qua bảng tra cứu `dynamic_sd_lookup`, sau đó áp dụng cơ chế cắt tỉa placeholder nháp (`per-request trimming`) tại hook `_update_after_schedule`. Nếu luồng nào có $N_i < N_{\text{batch}}$, các vị trí dư thừa được gán mặt nạ che (padding mask) để không tham gia vào hàm mất mát và tính toán xác thực.

3. **Rủi ro 3: Trôi dạt chi phí trên GPU mới (ví dụ RTX 4090 chưa có file profile):**
   * *Nguyên nhân:* Không có sẵn ma trận đo thực tế $C(N, B)$ trong danh mục `profiles/`.
   * *Phương án giải quyết:* Kích hoạt chế độ phân tích giải tích (Analytical Fallback Mode) sử dụng công thức hồi quy xấp xỉ đã kiểm chứng trong Task 1, đồng thời lưu cảnh báo log để người vận hành chạy lệnh `flexsds profile` nhằm hiệu chuẩn tự động trong 15 giây.
