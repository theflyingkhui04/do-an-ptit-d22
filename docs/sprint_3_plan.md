# Kế Hoạch Phân Rã Chi Tiết Sprint 3: Cột Mốc M2 — Hardware-Aware Adaptive Speculative Decoding (v0) & Lossless Verification

- **Thời gian thực hiện:** 05/10/2026 – 14/10/2026 (10 ngày) *(Kế hoạch ép tiến độ để tối ưu hóa quỹ thời gian cho Sprint 5 mở rộng huấn luyện Qwen 3.5 9B)*
- **Cột mốc mục tiêu:** **M2 (Adaptive v0 & Lossless Verification)**
- **Mục tiêu cốt lõi của Sprint 3:**
  1. Hiện thực hóa **Bộ điều khiển Cây nháp Thích ứng phần cứng (Hardware-Aware Adaptive Speculative Controller v0)** tích hợp vào vLLM engine nhằm tự động điều chỉnh độ sâu nháp $k$ hoặc ngắt về Vanilla khi $\alpha$ thấp, triệt tiêu 100% hiện tượng "tăng tốc âm" đã ghi nhận ở W3.
  2. Xây dựng bộ công cụ kiểm chứng **Lossless Verification**: chứng minh thuật toán sinh văn bản bảo toàn 100% từng token so với Vanilla ở `temperature = 0`.
  3. Mở rộng hệ thống đo kiểm tự động lên **3 làn suy luận đối chứng**: `Vanilla Baseline`, `Static Draft (k=3)`, và `Adaptive Speculative Decoding (v0)`.

---

## 1. Phân Rã Công Việc Thành 4 User Story / Task Con Chi Tiết

Dựa trên nguyên tắc quản trị MoSCoW và phân chia vai trò chuyên môn, Sprint 3 được phân rã thành 4 task kỹ thuật trọng tâm:

```
                      CẤU TRÚC PHÂN RÃ CÔNG VIỆC SPRINT 3 (M2)
+---------------------------------------------------------------------------------+
|                                SPRINT 3 (05/10 - 14/10)                         |
+---------------------------------------------------------------------------------+
          |                                      |
          v                                      v
 [M2-01] Hardware Profiler              [M2-02] Adaptive Controller v0
 - Đo chi phí forward Target vs Draft   - Heuristic điều khiển k theo alpha
 - Lập bảng ma trận chi phí C_T, C_D    - Cơ chế ngắt tự động (Fallback gate)
 (Lead: Khôi 70%, Đại 30%)              (Lead: Khôi 60%, Dương 40%)
          |                                      |
          +------------------+-------------------+
                             |
                             v
 [M2-03] Lossless Verification          [M2-04] Benchmark Đối Chứng 3 Làn
 - Bộ test so khớp 100% token @ temp=0  - Đo 3 làn x 3 workloads x 6 concurrency
 - Kiểm thử trên 50 prompt chuẩn        - Xuất bản bảng M2 summary.csv & Biểu đồ
 (Lead: Đại 70%, Khôi 30%)              (Lead: Đại 60%, Dương 40%)
```

---

### Task 1: `[M2-01] Xây dựng Module Hardware Profiler & Mô hình Chi phí Suy luận (Cost Model) trên GPU T4`
- **Mã công việc:** `[M2-01]`
- **Thời gian:** 05/10/2026 – 08/10/2026 (4 ngày)
- **Người thực hiện:** Cấn Đức Khôi (Lead 70%) & Bùi Đức Đại (Partner 30%)
- **Mục tiêu kỹ thuật:**
  - Viết module độc lập [`src/engine/hardware_profiler.py`](../../../src/engine/hardware_profiler.py) tự động chạy đo vi benchmark lúc khởi động (startup warmup):
    - Đo thời gian forward đơn bước của Target 7B-AWQ ($C_T$) theo các kích thước batch size ($B \in [1, 2, 4, 8, 16, 32]$).
    - Đo thời gian forward của Draft Model 0.5B ($C_D$) với độ dài nháp $k \in [1, 2, 3, 4, 5]$.
    - Đo thời gian overhead xác thực song song (Verification Overhead $C_V$).
  - Xuất bảng cấu hình chi phí phần cứng [`experiments/configs/hardware_profile_t4.json`](../../../experiments/configs/hardware_profile_t4.json) để nạp làm tham số đầu vào cho bộ điều khiển Adaptive.
- **Tiêu chí nghiệm thu (DoD):**
  - [ ] Module đo tự động hoàn thành trong $< 60$ giây lúc khởi động server.
  - [ ] File JSON chứa ma trận chi phí chuẩn xác, có tính lặp lại (độ lệch $< 5\%$).

---

### Task 2: `[M2-02] Hiện thực hóa Bộ điều khiển Cây nháp Thích ứng (Adaptive Speculative Controller v0)`
- **Mã công việc:** `[M2-02]`
- **Thời gian:** 07/10/2026 – 10/10/2026 (4 ngày)
- **Người thực hiện:** Cấn Đức Khôi (Lead 60%) & Dương Nguyễn Thế (Partner 40%)
- **Mục tiêu kỹ thuật:**
  - Xây dựng lớp điều khiển `AdaptiveDraftController` tích hợp vào cơ chế `custom_proposer` của vLLM:
    - Thuật toán cửa sổ trượt (Exponential Moving Average / Rolling Window 10 bước) theo dõi tức thời $\alpha_{\text{ema}}$.
    - **Quy tắc điều chỉnh ngân sách nháp (Budget Rule):**
      - Nếu $\alpha_{\text{ema}} \ge 0.80$ (tác vụ có cấu trúc cao như W1, W2): Duy trì $k=3$ hoặc mở rộng $k=4$.
      - Nếu $0.65 \le \alpha_{\text{ema}} < 0.80$: Hạ độ sâu nháp xuống $k=2$ để giảm chi phí forward mô hình nhỏ.
      - Nếu $\alpha_{\text{ema}} < 0.65$ (tác vụ tự do như W3): Ngắt hoàn toàn mô hình nháp ($k=0$), chuyển sang chế độ giải mã tuần tự Vanilla để triệt tiêu tăng tốc âm!
    - **Quy tắc áp lực bộ nhớ (Memory Pressure Rule):** Theo dõi tỉ lệ chiếm dụng KV-cache; nếu VRAM Peak $> 13.5$ GB hoặc KV Cache còn trống $< 10\%$, tự động ép $k=0$ để chống OOM.
  - Đóng gói Launcher Làn 3: [`scripts/launch_lane3_adaptive.sh`](../../../scripts/launch_lane3_adaptive.sh) mở trên cổng `8003`.
- **Tiêu chí nghiệm thu (DoD):**
  - [ ] Controller chuyển trạng thái linh hoạt $k \in [0, 2, 3, 4]$ mà không làm crash vLLM worker.
  - [ ] Đạt smoke test 10 requests với telemetry ghi nhận sự thay đổi động của `tree_budget` theo từng request.

---

### Task 3: `[M2-03] Xây dựng Kịch bản Kiểm chứng Lossless Verification (100% Token Match @ temp=0)`
- **Mã công việc:** `[M2-03]`
- **Thời gian:** 09/10/2026 – 12/10/2026 (4 ngày)
- **Người thực hiện:** Bùi Đức Đại (Lead 70%) & Cấn Đức Khôi (Partner 30%)
- **Mục tiêu kỹ thuật:**
  - Viết kịch bản kiểm tra tự động [`tests/verify_lossless.py`](../../../tests/verify_lossless.py) thực thi so sánh đầu ra:
    - Chạy đồng thời 50 prompt chuẩn (trích xuất từ W1, W2, W3) trên cả 2 làn: Vanilla vs Adaptive ở thiết lập suy luận chuẩn xác (`temperature = 0.0`, `greedy`).
    - So sánh chuỗi token ID đầu ra bằng hàm kiểm tra tương đương tuyệt đối:
      $$\text{TokenMatchRate} = \frac{\sum_{i=1}^N \mathbb{I}(\text{token}_{\text{adaptive}}^{(i)} == \text{token}_{\text{vanilla}}^{(i)})}{N} == 100.0\%$$
    - Báo cáo số lượng prompt khớp chính xác từng ký tự: $50 / 50$ (Không được phép sai lệch bất kỳ token nào).
  - Xuất file kết quả nghiệm thu [`experiments/results/m2_lossless_verification_summary.json`](../../../experiments/results/m2_lossless_verification_summary.json).
- **Tiêu chí nghiệm thu (DoD):**
  - [ ] 50/50 prompt đạt độ tương đồng 100% token ID (Zero token discrepancy).
  - [ ] Đập tan hoàn toàn nghi ngại của hội đồng: *"Tăng tốc độ sinh chữ thì chất lượng câu văn có bị giảm sút không?"*.

---

### Task 4: `[M2-04] Mở rộng Benchmark Harness & Đo đạc Thực nghiệm Đối chứng 3 Làn (M2 Sweep)`
- **Mã công việc:** `[M2-04]`
- **Thời gian:** 11/10/2026 – 13/10/2026 (3 ngày)
- **Người thực hiện:** Bùi Đức Đại (Lead 60%) & Dương Nguyễn Thế (Partner 40%)
- **Mục tiêu kỹ thuật:**
  - Nâng cấp `benchmark_harness.py` hỗ trợ đo đạc luân phiên cả 3 làn: `vanilla`, `static`, và `adaptive`.
  - Quét toàn diện ma trận: 3 làn × 3 workloads (W1, W2, W3) × 6 mức concurrency (`1, 2, 4, 8, 16, 32`) = **54 ô thực nghiệm** (mỗi ô lặp lại 5 lần, tổng cộng 270 lượt đo).
  - Thu thập và cập nhật bảng tổng hợp [`experiments/results/m2_summary.csv`](../../../experiments/results/m2_summary.csv).
  - Sinh biểu đồ khoa học đối chứng 3 làn:
    - Figure F1 (Mở rộng 3 đường): Throughput và TPOT của cả 3 làn.
    - Figure F2 (Mở rộng 3 làn): Chứng minh đường Adaptive **luôn nằm trên hoặc bằng vạch 1.0×**, loại bỏ hoàn toàn vùng tăng tốc âm của Static Draft ở W3!
- **Tiêu chí nghiệm thu (DoD):**
  - [ ] Hoàn thành trọn vẹn 54 ô đo lường tuân thủ kỷ luật Warm-up và Lặp lại $\ge 5$ lần.
  - [ ] Chứng minh trên W3: Speedup của Adaptive đạt $\ge 1.00\times$ ở mọi mức concurrency (triệt tiêu mức sụt giảm 0.81x của Static).

---

## 2. Lịch Trình Thực Hiện & Timeline Chi Tiết (05/10 – 14/10)

| Giai đoạn | Thời gian | Nhiệm vụ trọng tâm | Thành viên phụ trách |
| :--- | :---: | :--- | :--- |
| **Giai đoạn 1: Đo lường & Engine lõi** | 05/10 – 08/10 | Setup môi trường M2, viết `hardware_profiler.py` trích xuất chi phí T4 | Khôi, Đại |
| | 07/10 – 10/10 | Hiện thực hóa lõi `AdaptiveDraftController` v0 tích hợp vLLM | Khôi, Dương |
| | 09/10 – 10/10 | Smoke test làn Adaptive trên Colab T4, kiểm tra chuyển trạng thái $k$ | Cả nhóm |
| **Giai đoạn 2: Kiểm chứng & Đo đối chứng** | 09/10 – 12/10 | Thực hiện kiểm chứng Lossless Verification (50/50 prompts match) | Đại, Khôi |
| | 11/10 – 13/10 | Chạy full sweep benchmark 3 làn (54 cells) trên Colab/Kaggle T4 | Đại, Dương |
| | 13/10 – 14/10 | Tổng hợp `m2_summary.csv`, xuất bản biểu đồ F1/F2 3 làn, cập nhật Chương 5 | Khôi, Dương |
| | **14/10/2026** | **Họp Cổng Kiểm Soát M2 (M2 Gate Review): Nghiệm thu đóng Sprint 3** | Cả nhóm |

---

## 3. Rủi Ro Kỹ Thuật Dự Kiến & Biện Pháp Kiểm Soát

| Mã rủi ro | Mô tả rủi ro | Mức độ | Biện pháp phòng ngừa & kiểm soát |
| :---: | :--- | :---: | :--- |
| **R3.1** | vLLM không cho phép thay đổi cấu trúc cây nháp linh hoạt từng bước qua `custom_proposer` | Cao | Đã kiểm chứng ở M0-02: Thay vì thay đổi topology cây phức tạp, v0 sẽ điều chỉnh độ sâu nháp $k \in [0, 2, 3, 4]$ tuần tự và dùng cờ boolean để kích hoạt/ngắt draft worker. |
| **R3.2** | Kiểm chứng Lossless bị lệch ký tự do hiện tượng không xác định (non-deterministic CUDA operations) | Vừa | Cố định `seed = 42`, `temperature = 0.0`, ép cờ `torch.use_deterministic_algorithms(True)` và tắt các kernel tính toán gần đúng. |
| **R3.3** | Quá tải hạn mức GPU Colab/Kaggle khi chạy 54 cells lặp lại 5 lần (270 runs) | Vừa | Chia nhỏ kịch bản chạy theo từng workload (`--workload w1`, `--workload w2`, `--workload w3`), lưu checkpoint theo từng repeat để tránh mất số liệu khi bị ngắt session. |

---

## 4. Kế Hoạch Quản Trị Thẻ Task trên Trello

Sau khi kế hoạch này được phê duyệt, Team Lead (Khôi) sẽ tạo các thẻ công việc tương ứng trên Trello tại cột **`📌 Sprint 3 · To-Do`**:
1. `[M2-01] Xây dựng Module Hardware Profiler & Mô hình Chi phí Suy luận trên T4 (05/10 – 08/10)`
2. `[M2-02] Hiện thực hóa Bộ điều khiển Cây nháp Thích ứng (Adaptive Controller v0) (07/10 – 10/10)`
3. `[M2-03] Xây dựng Kịch bản Kiểm chứng Lossless Verification (100% Token Match @ temp=0) (09/10 – 12/10)`
4. `[M2-04] Mở rộng Benchmark Harness & Đo đạc Thực nghiệm Đối chứng 3 Làn (11/10 – 13/10)`
5. `[M2-05] Đánh giá Cổng Kiểm Soát M2 (14/10): Nghiệm thu Thuật toán Adaptive v0 & Báo cáo`
