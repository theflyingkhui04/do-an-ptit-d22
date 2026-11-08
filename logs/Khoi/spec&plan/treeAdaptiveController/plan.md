# Kế hoạch thực thi: Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng

> **Mã task Trello:** `#50 [M4-02-05] Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng`  
> **Dự án:** FlexSDS — Hardware-Aware Speculative Decoding for Self-Hosted LLMs on Consumer GPUs  
> **Người thực hiện:** Cấn Đức Khôi (Lead 70%) và Bùi Đức Đại (Partner 30%)  
> **Nhánh phát triển:** `feat/m4-02-05-tree-adaptive-controller`  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptiveController/plan.md`  
> **Thời gian thực hiện:** 08/11/2026 – 10/11/2026  
> **Trạng thái:** Kế hoạch thực thi chính thức (Revision 1.0 — 08/11/2026)

---

## 1. Mục tiêu và kết quả bàn giao (Deliverables)

Kế hoạch này chi tiết hóa việc thực hiện Task `[M4-02-05]` thuộc Chặng 3 của Sprint 5 nhằm hoàn thiện bộ điều khiển suy đoán thích ứng cây `TreeAdaptiveController` v0.2 phục vụ serving online, loại bỏ hiện tượng tăng tốc âm (negative speedup) và bảo vệ an toàn tài nguyên GPU.

### Các kết quả bàn giao chính:
1. **Lớp dữ liệu `TreeAdaptiveDecision`:** Kế thừa từ `BudgetDecision`, bổ sung các trường siêu dữ liệu cây (`topology_name`, `num_nodes`, `alpha_effective`, `alpha_star`, `speedup`, `status`).
2. **Bộ điều khiển `TreeAdaptiveController` trong `src/proposer/tree_adaptive.py`:**
   - Kế thừa `BudgetController` trong `src/proposer/controller.py`.
   - Chuẩn hóa bước $\alpha$ theo độ sâu $d$ của cây nháp: $\alpha_{\text{step}} = \min(1.0, \text{num\_accepted} / \max(d, 1))$.
   - Khử lệch khởi động: Unbiased EMA với $\beta = 0.8$.
   - **Tầng 1 — Dual-Tier Memory Guard:** Ngắt $T_0$ khi VRAM $< 1500\text{ MB}$ hoặc KV-cache $> 90\%$; hysteresis cooldown 3 bước; chỉ hồi phục khi KV-cache $\le 80\%$ VÀ VRAM $\ge 1500\text{ MB}$; cấm probe khi Memory Guard kích hoạt và reset bộ đếm probe.
   - **Tầng 2 & 3 — Concurrency Throttling:** Kẹp trần topo theo batch size $B$; ngắt $T_0$ khi $B > 24$ và $\alpha < 0.85$.
   - **Tầng 4 — Dynamic Fallback Gate & Adaptive Probing:** Chọn topo tối ưu qua `FlexCostModel.select_optimal_topology()`; ngắt $T_0$ khi $\text{Speedup} \le 1.0$; thăm dò $T_4$ định kỳ theo chu kỳ $T_{\text{probe}}(B) \in \{15, 20, 30\}$.
   - **Tầng 5 — Zero-Crash Resilience:** Bắt ngoại lệ và fallback an toàn về $T_0$.
   - Thời gian quyết định $< 0.05\text{ ms}$.
3. **Đấu nối vào `src/proposer/adapter.py`:** Hỗ trợ `TreeAdaptiveController` làm controller điều phối cho vLLM speculative proposal loop.
4. **Bộ kiểm thử toàn diện `tests/test_tree_adaptive_controller.py`:** 12 ca kiểm thử bao phủ toàn bộ trạng thái chuyển làn, áp lực bộ nhớ và benchmark độ trễ.
5. **Nhật ký kỹ thuật và nghiệm thu:** `logs/Khoi/M4-02/M4-02-05_worklogs.md`.

---

## 2. Lộ trình triển khai chi tiết theo phương pháp TDD

```text
┌────────────────────────────────────────────────────────────────────────┐
│             LỘ TRÌNH TRIỂN KHAI TDD TASK [M4-02-05]                    │
├────────────────────────────────────────────────────────────────────────┤
│ GIAI ĐOẠN 1: THIẾT LẬP BỘ KIỂM THỬ KHUNG (TEST HARNESS)                │
│  • Tạo file tests/test_tree_adaptive_controller.py                     │
│  • Khai báo 12 ca kiểm thử với các mock và assertion chặt chẽ          │
│                                                                        │
│ GIAI ĐOẠN 2: HIỆN THỰC HÓA CORE CONTROLLER V0.2                        │
│  • Tạo file src/proposer/tree_adaptive.py                              │
│  • Hiện thực hóa TreeAdaptiveDecision                                  │
│  • Cài đặt logic Dual-Tier Memory Guard, Cooldown & Probe Suppression  │
│  • Cài đặt Concurrency Throttling & Dynamic Fallback Gate              │
│  • Cài đặt Depth-Normalized Alpha Update & Zero-Crash Handler          │
│                                                                        │
│ GIAI ĐOẠN 3: ĐẤU NỐI VÀO ADAPTER SERVING                               │
│  • Cập nhật src/proposer/adapter.py tương thích TreeAdaptiveController │
│  • Viết integration test giữa adapter và controller mới                │
│                                                                        │
│ GIAI ĐOẠN 4: BENCHMARK ĐỘ TRỄ VÀ KIỂM CHỨNG HỒI QUY                    │
│  • Chạy benchmark 10.000 lượt đo độ trễ select_budget (< 0.05 ms)      │
│  • Chạy toàn bộ test suite FlexSDS (37/37 + 12/12 = 49 tests)          │
│                                                                        │
│ GIAI ĐOẠN 5: TÀI LIỆU HÓA VÀ NGHIỆM THU                                │
│  • Viết worklog logs/Khoi/M4-02/M4-02-05_worklogs.md                   │
│  • Kiểm tra DoD trên thẻ Trello và chuẩn bị PR                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi tiết các nhiệm vụ cụ thể

### Nhiệm vụ 1: Xây dựng bộ unit test khung (`tests/test_tree_adaptive_controller.py`)
- **Mục tiêu:** Tạo test suite bao phủ toàn bộ các yêu cầu trong đặc tả kỹ thuật trước khi hoàn thiện implementation (chuẩn TDD).
- **Các test case:**
  1. `test_tree_adaptive_controller_initialization`: Kiểm tra khởi tạo tham số mặc định và tham số tùy biến.
  2. `test_unbiased_ema_alpha_tracking`: Kiểm tra tính hội tụ của EMA với công thức bù lệch khởi động.
  3. `test_depth_normalized_alpha_update`: Kiểm tra việc tính $\alpha_{\text{step}} = \text{num\_accepted} / d$ không bị sụt giảm khi $N > d$.
  4. `test_dual_tier_memory_guard_vram_trigger`: Kích hoạt ngắt khi VRAM $< 1500\text{ MB}$.
  5. `test_dual_tier_memory_guard_kv_cache_trigger`: Kích hoạt ngắt khi KV-cache $> 90\%$.
  6. `test_memory_guard_cooldown_and_recovery_conditions`: Kiểm tra đếm trễ 3 bước và điều kiện kép hồi phục.
  7. `test_no_probing_under_memory_guard`: Xác nhận không bao giờ phát $T_4$ khi đang ở trong Memory Guard và reset probe counter.
  8. `test_concurrency_throttling_limits`: Kiểm tra kẹp trần theo $B \in \{1, 4, 8, 16, 32\}$ và ngắt $T_0$ khi $B=32, \alpha < 0.85$.
  9. `test_dynamic_fallback_gate_and_adaptive_probing_cycle`: Kiểm tra ngắt khi $\text{Speedup} \le 1.0$ và phát probe đúng nhịp $T_{\text{probe}}(B)$.
  10. `test_zero_crash_resilience`: Bắt ngoại lệ chia cho 0 hoặc lỗi cost model mà không crash.
  11. `test_decision_latency_benchmark`: Đo 10.000 lượt gọi `select_budget()`, bảo đảm thời gian trung bình $< 0.05\text{ ms}$.
  12. `test_proposer_adapter_integration`: Kiểm tra `AdaptiveProposerAdapter.propose()` và `on_verification_completed()` với `TreeAdaptiveController`.

---

### Nhiệm vụ 2: Hiện thực hóa `TreeAdaptiveController` trong `src/proposer/tree_adaptive.py`
- **Mục tiêu:** Lập trình module điều khiển cây nháp v0.2 phục vụ môi trường serving.
- **Các bước cài đặt:**
  - Khai báo lớp dữ liệu `TreeAdaptiveDecision(BudgetDecision)` kế thừa `k` từ `BudgetDecision` và lưu các trường siêu dữ liệu mở rộng.
  - Khai báo lớp `TreeAdaptiveController(BudgetController)`:
    - Lưu trữ `cost_model` (mặc định khởi tạo `FlexCostModel`), `candidate_topologies` (mặc định `["T0", "T4", "T8", "T16", "T32"]`), `cooldown_steps = 3`.
    - Duy trì các biến trạng thái: `_step_count`, `_fallback_steps`, `_memory_guard_active`, `_cooldown_counter`, `_last_selected_depth`, `_last_selected_topology`.
    - Cài đặt `update(num_accepted, num_drafted)`: tính $\alpha_{\text{step}}$ chuẩn hóa theo `_last_selected_depth` và cập nhật EMA.
    - Cài đặt `get_effective_alpha()`: tính unbiased EMA với $\beta = 0.8$.
    - Cài đặt phương thức chính `select_budget(batch_size, free_vram_mb, kv_cache_usage_pct)` chia thành 5 tầng rõ ràng:
      - **Tầng 1 (Memory Guard):** Kiểm tra điều kiện ngắt, xử lý cooldown 3 bước và điều kiện kép hồi phục.
      - **Tầng 2 (Tín hiệu):** Lấy `alpha_effective`.
      - **Tầng 3 (Concurrency Throttling):** Lọc danh sách candidate theo $B$. Nếu $B > 24$ và $\alpha < 0.85$, ngắt về $T_0$ kèm `status="concurrency_throttled"`.
      - **Tầng 4 (Solver & Probing):** Gọi `cost_model.select_optimal_topology()`. Nếu $\text{Speedup} \le 1.0$, kiểm tra nhịp `probe_interval(B)` để phát $T_4$ thăm dò hoặc giữ $T_0$.
      - **Tầng 5 (Zero-Crash):** Khối `try ... except` bao quanh toàn bộ logic, trả về $T_0$ an toàn nếu gặp sự cố.

---

### Nhiệm vụ 3: Đấu nối vào `src/proposer/adapter.py`
- **Mục tiêu:** Cho phép `AdaptiveProposerAdapter` phối hợp trực tiếp với `TreeAdaptiveController`.
- **Nội dung thay đổi:**
  - Import `TreeAdaptiveController` và `TreeAdaptiveDecision` trong `src/proposer/adapter.py`.
  - Cập nhật hàm `propose()` để lấy `decision.k` (số node nháp $N$) truyền vào `draft_generator`.
  - Đảm bảo tương thích ngược 100% với các test cũ của `adapter.py`.

---

### Nhiệm vụ 4: Kiểm thử, đo benchmark hiệu năng và kiểm tra hồi quy
- **Mục tiêu:** Thẩm định chất lượng code đạt chuẩn DoD của dự án.
- **Thực thi:**
  - Chạy `pytest tests/test_tree_adaptive_controller.py -v`.
  - Kiểm tra tiêu chí thời gian ra quyết định: `avg_latency < 0.05 ms` (mục tiêu thực tế $< 0.01\text{ ms}$).
  - Chạy toàn bộ test suite của FlexSDS:
    ```bash
    PYTHONPATH=. pytest tests/test_flexsds_baseline.py tests/test_cost_model_l4.py tests/test_cost_model_solver.py tests/test_tree_topology.py tests/test_tree_adaptive_controller.py
    ```

---

### Nhiệm vụ 5: Viết tài liệu nghiệm thu và hoàn tất bàn giao
- **Mục tiêu:** Cập nhật tài liệu minh bạch và theo dõi tiến độ trên Trello.
- **Thực thi:**
  - Viết worklog chi tiết tại `logs/Khoi/M4-02/M4-02-05_worklogs.md`.
  - Cập nhật checklist DoD trong thẻ Trello `#50`.
  - Tạo commit chuẩn conventional commits và đẩy nhánh `feat/m4-02-05-tree-adaptive-controller`.

---

## 4. Quản lý rủi ro kỹ thuật

| Rủi ro | Mức độ | Biện pháp giảm thiểu |
| :--- | :---: | :--- |
| **Sập tín hiệu $\alpha$ do chia cho $N$ thay vì $d$** | Nghiêm trọng | Lưu biến `_last_selected_depth` trong controller và chuẩn hóa $\alpha_{\text{step}} = \text{num\_accepted} / \max(d, 1)$. |
| **Trễ tính toán vượt quá trần 0.05 ms** | Trung bình | Tận dụng cache $O(1)$ của `FlexCostModel` và bảng tra cứu $E[\tau]$ tiền tính toán của `TreeTopology`. |
| **OOM do phát probe khi bộ nhớ cạn kiệt** | Cao | Đặt kiểm tra Memory Guard ở Tầng 1, cấm tuyệt đối mọi bước thăm dò khi Guard đang kích hoạt và reset bộ đếm probe về 0. |
| **Lỗi runtime bất ngờ làm crash luồng serving** | Nghiêm trọng | Bọc toàn bộ phương thức `select_budget()` trong khối `try ... except Exception`, trả về $T_0$ an toàn nếu có lỗi. |
