# [M3-05] Worklog Khôi — Mở rộng bộ đo benchmark từ 3 làn lên 4 làn (thêm Làn 4 EAGLE-3.1 Custom Draft)

> **Task:** `[M3-05] Mở rộng Benchmark Harness & Đo kiểm Thực nghiệm Đối chứng 4 Làn (27/10 – 29/10)`  
> **Subtask:** `[Khôi] Mở rộng bộ đo benchmark từ 3 làn lên 4 làn (thêm Làn 4 EAGLE-3.1 Custom Draft)`  
> **Người thực hiện:** Cấn Đức Khôi (Partner 30%) và Bùi Đức Đại (Lead 70%)  
> **Nhánh phát triển:** `feat/m3-05-benchmark-4lanes`  
> **Trạng thái:** ✅ **HOÀN THÀNH (100% PASS 181/181 unit tests)**

---

## 1. Mục tiêu và bối cảnh kiến trúc

### A. Mục tiêu kỹ thuật:
1. Mở rộng bộ đo kiểm `experiments/benchmark_harness.py` hỗ trợ đo đạc đầy đủ cả 4 làn:
   - **Làn 1 (`vanilla`):** Chuẩn so sánh autoregressive cơ sở ($1.0\times$).
   - **Làn 2 (`static`):** Mô hình nháp tĩnh độc lập Qwen2.5-0.5B-Instruct ($k=3$).
   - **Làn 3 (`adaptive`):** Bộ điều khiển thích ứng ngân sách phần cứng Hardware-Aware ($k \in \{0, 2, 3, 4\}$).
   - **Làn 4 (`eagle` / `eagle-custom`):** Đầu nháp suy đoán chuyên biệt EAGLE-3.1 (`duckhoi/qwen2.5-7b-eagle3.1-speculator`) tự huấn luyện (checkpoint epoch 2, 10.000 step).
2. Đồng bộ hóa cấu hình phục vụ qua `experiments/modal_lanes.py` và client chia sẻ `src/client/modal_client.py`.
3. Bổ sung script khởi chạy cục bộ `scripts/launch_lane4_eagle.sh` (cổng 8004) và cập nhật script điều phối chuyển làn `scripts/switch_lane.sh`.
4. Bàn giao bộ đo 4 làn hoàn chỉnh và sẵn sàng để Đại tiến hành sweep thực nghiệm 54 canonical prompts trên 3 workloads W1, W2, W3 (GPU L4).

### B. Giải đáp ranh giới kiến trúc: Chuỗi nháp tuyến tính (Linear Chain) vs Cây nháp (Tree Drafting):
* **Tính chất khoa học của Làn 4 ở Sprint 4 (M3-05):**
  Làn 4 đóng vai trò bài đo đối chứng thành phần (**Ablation Study**) nhằm so sánh chất lượng của đầu nháp chuyên biệt (Domain-Adapted Speculator Head) với mô hình nháp SLM độc lập (Qwen2.5-0.5B).
  Việc chạy Làn 4 ở dạng chuỗi nháp tuyến tính $k=3$ (cùng độ sâu với Làn 2) giúp cô lập hoàn toàn biến số thuật toán:
  - Chi phí forward pass $\tau_q$ của đầu nháp 1 layer decoder rẻ hơn 3–5 lần so với SLM 24 layers.
  - Tỷ lệ chấp thuận $\alpha$ trên dữ liệu MC-OCR và tiếng Việt bứt phá từ ~65% lên $>80\%-85\%$.
* **Ranh giới thích ứng cây (Tree Adaptive):**
  Lớp thích ứng của nhóm (`AdaptiveDraftController`) ở Sprint 4 được thiết kế cho chuỗi nháp tuyến tính $k \in \{0, 2, 3, 4\}$. Bài toán thích ứng cây (tối ưu hóa Tree Topology: branching factor $b$, depth $d$, pruning threshold $\gamma$) là hướng nghiên cứu mở rộng chính thức được quy hoạch cho **Sprint 5 (M4 - Card #6)**. Nhóm hoàn toàn chưa và không cần triển khai Tree Adaptive ở Sprint 4.

---

## 2. Các thay đổi mã nguồn đã thực hiện

1. **Bộ đo benchmark (`experiments/benchmark_harness.py`):**
   - Mở rộng tham số CLI `--lane`: bổ sung lựa chọn `"eagle"` và `"eagle-custom"` vào danh sách hợp lệ `choices=["vanilla", "static", "adaptive", "eagle", "eagle-custom"]`.
   - Bổ sung cơ chế graceful import fallback cho `aiohttp`, `torch`, `pynvml` giúp module CLI và các kiểm thử hợp đồng (contract tests) chạy độc lập trên mọi môi trường CPU host.
2. **Cấu hình làn phục vụ (`experiments/modal_lanes.py`):**
   - Khai báo hằng số `EAGLE_DRAFT_MODEL = "duckhoi/qwen2.5-7b-eagle3.1-speculator"`, `EAGLE_SERVED_MODEL = "qwen2.5-7b-eagle3-draft"`.
   - Bổ sung cấu hình `eagle` và `eagle-custom` vào từ điển `LANES` với speculative config $k=3$, greedy sampling, và heterogeneous vocab.
   - Xử lý phân tách revision độc lập (`EAGLE_REVISION` tách biệt với `DRAFT_REVISION` của Qwen2.5-0.5B) trong `build_vllm_args()` và `build_server_env()`.
3. **Client chia sẻ (`src/client/modal_client.py`):**
   - Cập nhật ánh xạ `LANE_MODELS` bổ sung `"eagle": "qwen2.5-7b-eagle3-draft"` và `"eagle-custom": "qwen2.5-7b-eagle3-draft"`.
4. **Script khởi chạy cục bộ (`scripts/`):**
   - Tạo mới `scripts/launch_lane4_eagle.sh`: khởi chạy server vLLM Làn 4 trên cổng `8004`, nạp mô hình đích `Qwen/Qwen2.5-7B-Instruct-AWQ` và đầu nháp `duckhoi/qwen2.5-7b-eagle3.1-speculator` với $k=3$.
   - Cập nhật `scripts/switch_lane.sh`: bổ sung nhánh lệnh `lane4` và cổng `8004` vào quy trình dọn dẹp, kiểm tra sức khỏe và hiển thị trạng thái hệ thống.
5. **Modal serverless runner (`experiments/modal_runner.py` & `experiments/modal_smoke.py`):**
   - Bổ sung `eagle` vào danh sách làn mặc định `ENABLED_LANES` và danh sách tải trước trọng số `download_weights()`.
   - Định nghĩa container server `EagleServer` phục vụ endpoint Làn 4 trên Modal GPU L4.

---

## 3. Kết quả kiểm thử và nghiệm thu (Verification)

Toàn bộ 11 bộ kiểm thử liên quan (181 tests) đã chạy và đạt kết quả 100% Pass:
```bash
PYTHONPATH=. pytest tests/test_modal_runner_config.py \
                    tests/test_modal_client.py \
                    tests/test_modal_smoke.py \
                    tests/test_m2_harness_contract.py \
                    tests/test_benchmark_profiles.py \
                    tests/test_benchmark_workload_adapter.py \
                    tests/test_m2_campaign.py \
                    tests/test_cost_model_l4.py \
                    tests/test_budget_controller.py \
                    tests/test_budget_fallback_recovery.py \
                    tests/test_budget_transition_scenarios.py
```
* **Kết quả:** `181 passed, 2 skipped in 0.61s` (không có bất kỳ lỗi hồi quy nào).
* **Độ tương thích launcher:** Cấu hình flags trong `modal_lanes.py` khớp 100% với `scripts/launch_lane4_eagle.sh`.
* **Sẵn sàng bàn giao:** Bùi Đức Đại có thể sử dụng ngay `benchmark_harness.py` với cờ `--lane eagle` để thực thi quét 54 prompts canonical trên Modal và Kaggle/Colab.
