# Kế hoạch thực thi: Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng

> **Mã task Trello:** `#50 [M4-02-05] Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng`  
> **Dự án:** FlexSDS — Hardware-Aware Speculative Decoding for Self-Hosted LLMs on Consumer GPUs  
> **Người thực hiện:** Cấn Đức Khôi (Lead 70%) và Bùi Đức Đại (Partner 30%)  
> **Nhánh phát triển:** `feat/m4-02-05-tree-adaptive-controller`  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptiveController/plan.md`  
> **Thời gian thực hiện:** 08/11/2026 – 10/11/2026  
> **Trạng thái:** Kế hoạch thực thi hoàn thiện (Revision 1.1 — 09/11/2026, cập nhật sau thẩm định kỹ thuật)

---

## 1. Mục tiêu và kết quả bàn giao (Deliverables)

Kế hoạch này chi tiết hóa việc thực hiện Task `[M4-02-05]` thuộc Chặng 3 của Sprint 5 nhằm hoàn thiện bộ điều khiển suy đoán thích ứng cây `TreeAdaptiveController` v0.2 phục vụ serving online, loại bỏ hiện tượng tăng tốc âm (negative speedup) và bảo vệ an toàn tài nguyên GPU.

### Các kết quả bàn giao chính:
1. **Lớp dữ liệu `TreeAdaptiveDecision` và `ProposalOutput`:**
   - Kế thừa từ `BudgetDecision`, bổ sung các trường siêu dữ liệu cây (`topology_name`, `num_nodes`, `alpha_effective`, `alpha_star`, `speedup`, `status`).
   - Mở rộng `ProposalOutput.topology_name` tại `src/proposer/adapter.py` để truyền thông suốt thông tin cấu trúc cây cho scheduler và verifier.
2. **Bộ điều khiển `TreeAdaptiveController` trong `src/proposer/tree_adaptive.py`:**
   - Kế thừa `BudgetController` trong `src/proposer/controller.py`.
   - Chuẩn hóa bước $\alpha$ theo độ sâu $d$ của cây nháp: $\alpha_{\text{step}} = \min(1.0, \text{num\_accepted} / \max(d, 1))$ (empirical yield index).
   - Tách biệt `_observation_count` phục vụ debiasing EMA chính xác; tự phục hồi trạng thái khi gặp `NaN`/`Inf`.
   - **Tầng 1 — Dual-Tier Memory Guard & Sentinel Filter:** Nhận diện sentinel `1e9`, siết ngưỡng an toàn KV ở 75%/65%; ngắt $T_0$ khi thiếu VRAM hoặc KV cao; hysteresis cooldown 3 bước; máy trạng thái phục hồi an toàn theo `_guard_trigger_source`; cấm probe khi Memory Guard kích hoạt.
   - **Tầng 2 & 3 — Concurrency Throttling:** Kẹp trần topo theo batch size $B$; ngắt $T_0$ khi $B > 24$ và $\alpha < 0.85$.
   - **Tầng 4 — Dynamic Fallback Gate & Adaptive Probing:** Chọn topo tối ưu qua `FlexCostModel.select_optimal_topology()`; ngắt $T_0$ (`fallback_gate`) khi $\text{Speedup} \le 1.0$; thăm dò $T_4$ định kỳ theo chu kỳ $T_{\text{probe}}(B) \in \{15, 20, 30\}$.
   - **Tầng 5 — Zero-Crash Resilience:** Bắt ngoại lệ và fallback an toàn về $T_0$ (`zero_crash_fallback`).
   - Thời gian ra quyết định trung bình $< 0.05\text{ ms}$ (thực tế đo được $0.0164\text{ ms}$).
3. **Đấu nối vào `src/proposer/adapter.py`:** Hỗ trợ `TreeAdaptiveController` làm controller điều phối cho vLLM speculative proposal loop.
4. **Bộ kiểm thử toàn diện `tests/test_tree_adaptive_controller.py`:** 19 ca kiểm thử bao phủ toàn bộ trạng thái chuyển làn, áp lực bộ nhớ, sentinel VRAM, self-healing và benchmark độ trễ.
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
│  • Chạy toàn bộ test suite FlexSDS (75/75 tests pass)                  │
│                                                                        │
│ GIAI ĐOẠN 5: ĐIỀU CHỈNH KỸ THUẬT SAU THẨM ĐỊNH (REVISION 1.1)          │
│  • Bổ sung bộ lọc sentinel VRAM và máy trạng thái phục hồi an toàn     │
│  • Tách biệt _observation_count và cơ chế self-healing cho alpha       │
│  • Đóng gói topology_name trong ProposalOutput                         │
│  • Bổ sung 4 ca kiểm thử biên (nâng tổng số test lên 19/19)           │
│  • Đồng bộ tài liệu đặc tả, kế hoạch và worklog                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Nhật ký nghiệm thu và bằng chứng kiểm thử

- **19/19 tests** trong `tests/test_tree_adaptive_controller.py` PASS (0.28s).
- **75/75 tests** toàn hệ thống FlexSDS PASS (0.29s).
- **Mean decision latency:** **$0.0164\text{ ms}$ ($16.43\ \mu\text{s}$)**, thỏa mãn điều kiện SLA $< 0.05\text{ ms}$.
- Toàn bộ 10 phát hiện thẩm định từ Bùi Đức Đại (C01–C10, P01–P05) đã được giải quyết triệt để.
