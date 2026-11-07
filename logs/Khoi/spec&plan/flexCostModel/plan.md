# Kế hoạch thực thi: Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động

> **Mã task Trello:** `#48 [M4-02-03] Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động alpha*(T, B)`  
> **Dự án:** FlexSDS — Hardware-Aware Speculative Decoding for Self-Hosted LLMs on Consumer GPUs  
> **Người thực hiện:** Cấn Đức Khôi (Lead 70%) và Bùi Đức Đại (Partner 30%)  
> **Nhánh phát triển:** `feat/m4-02-03-cost-model-solver`  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/flexCostModel/plan.md`  
> **Thời gian thực hiện:** 07/11/2026 – 09/11/2026  
> **Trạng thái:** Kế hoạch thực thi chính thức (Revision 1.1 — 07/11/2026)

---

## 1. Mục tiêu và phạm vi công việc

Kế hoạch này chi tiết hóa việc thực hiện Task `[M4-02-03]` thuộc Chặng 2 của Sprint 5 nhằm cung cấp bộ giải nghiệm hòa vốn động $\alpha^*(\mathcal{T}, B)$ chính xác, ổn định và tối ưu hiệu năng cao cho lớp điều khiển `TreeAdaptiveController` (Task `[M4-02-05]`).

### Các kết quả bàn giao chính (Deliverables):
1. **Lớp dữ liệu `SolverResult`:** Trả về kết quả giải nghiệm minh bạch kèm trạng thái (`converged`, `always_profitable`, `infeasible`), tỉ số chi phí và sai số phần dư.
2. **Nâng cấp `FlexCostModel` trong `src/flexsds/core/cost_model.py`:**
   - Tích hợp hàm kỳ vọng chấp nhận token $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$ từ module `TreeTopology`.
   - Thuật toán bisection search 50 bước giải nghiệm với sai số $< 10^{-4}$ và thời gian $< 0.01\text{ ms}$.
   - Khắc phục triệt để lỗi S08: Chặn trên $1 + d$ và xử lý vô nghiệm trả về `is_feasible = False`.
   - Cơ chế nạp ma trận thực nghiệm đa chiều kết hợp phân tích giải tích (Analytical Fallback).
   - Hàm chọn cây tối ưu `select_optimal_topology` kèm Fallback Gate toàn catalog.
   - Hàm tính tốc độ dự báo `evaluate_speedup(topo, batch_size, alpha)`.
   - Bảo toàn 100% các hàm tương thích ngược (`calculate_alpha_star`, `estimate_expected_tokens`, `select_optimal_budget`).
3. **Bộ kiểm thử tự động `tests/test_cost_model_solver.py`:** Kiểm tra độ hội tụ, sai số, trường hợp biên và benchmark tốc độ giải.
4. **Nhật ký kỹ thuật:** `logs/Khoi/M4-02/M4-02-03_worklogs.md`.

---

## 2. Phân rã công việc chi tiết (Work Breakdown Structure)

```text
┌────────────────────────────────────────────────────────────────────────┐
│               LỘ TRÌNH THỰC THI TASK M4-02-03 (07/11 – 09/11)          │
├────────────────────────────────────────────────────────────────────────┤
│ Bước 1: Thiết kế cấu trúc dữ liệu và nâng cấp nạp Profile              │
│   • Thêm dataclass SolverResult                                        │
│   • Mở rộng from_json nạp ma trận cây topo_{name}_b{b} và k{k}_b{b}    │
│   • Thêm helper _resolve_topology(topo)                                │
│   • Hoàn thiện cơ chế Analytical Fallback (EAGLE draft + gamma_attn)   │
│                                                                        │
│ Bước 2: Cài đặt thuật toán Bisection Solver và Caching                 │
│   • Cài đặt solve_alpha_star(topology, batch_size, max_iter=50)        │
│   • Kiểm tra điều kiện biên: R <= 1.0 (profitable) & R > 1+d (no-root) │
│   • Cài đặt evaluate_speedup(topo, batch_size, alpha)                  │
│   • Tích hợp bảng cache nghiệm O(1) đạt thời gian tra cứu < 0.001 ms   │
│   • Bảo đảm tương thích ngược với calculate_alpha_star(k, B)           │
│                                                                        │
│ Bước 3: Lựa chọn cấu hình cây tối ưu & Fallback Gate                   │
│   • Cài đặt select_optimal_topology(alpha, batch_size, candidates)     │
│   • Ngắt an toàn về T0 khi max Speedup <= 1.0                          │
│   • Giữ select_optimal_budget & estimate_expected_tokens adapter       │
│                                                                        │
│ Bước 4: Viết unit test & thẩm định hội tụ nghiệm (Khôi + Đại)          │
│   • Tạo tests/test_cost_model_solver.py (10 test cases)                │
│   • Kiểm tra sai số số học |Speedup - 1.0| < 10^-4                     │
│   • Chạy kiểm thử hồi quy bảo đảm 18/18 tests cũ tiếp tục PASS         │
│   • Đo benchmark độ trễ giải nghiệm < 0.01 ms                          │
│                                                                        │
│ Bước 5: Đồng bộ tài liệu, viết worklog và tạo PR                       │
│   • Viết logs/Khoi/M4-02/M4-02-03_worklogs.md                          │
│   • Cập nhật checklist Trello và chuẩn bị PR vào develop               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi tiết từng bước thực hiện

### Bước 1: Mở rộng mô hình chi phí và cấu trúc dữ liệu (`cost_model.py`)
- **Nhiệm vụ:**
  1. Định nghĩa `SolverResult` (các trường: `alpha_star`, `is_feasible`, `status`, `cost_ratio`, `residual`, `iterations`, `topology_name`, `batch_size`).
  2. Cài đặt `_resolve_topology(topo: Union[str, TreeTopology, int]) -> TreeTopology`.
  3. Nâng cấp `from_json`: Phân tích cả khóa truyền thống dạng `"k3_b4"` lẫn khóa cấu hình cây dạng `"topo_T8_b4"`.
  4. Cập nhật `get_step_cost(topology, batch_size)`: Nhận tham số `topology` linh hoạt ở dạng chuỗi tên cây (`"T8"`), đối tượng `TreeTopology` hoặc số nguyên $k$ (tương thích ngược).
  5. Cài đặt chế độ tính toán giải tích khi chưa có profile thực nghiệm:
     $$C(\mathcal{T}, B) = N \cdot t_{\text{step}}(B) + C_T(B) \cdot \left(1 + \gamma_{\text{attn}} \cdot \max(0, N - 1)\right)$$
- **Người thực hiện:** Khôi (Lead 70%).

### Bước 2: Cài đặt bộ giải nghiệm hòa vốn động (`solve_alpha_star`)
- **Nhiệm vụ:**
  1. Tính tỉ số chi phí $R(\mathcal{T}, B) = \frac{C(\mathcal{T}, B)}{C(0, B)}$.
  2. Kiểm tra chặn trên lý thuyết:
     - Nếu $R \le 1.0 \implies$ trả về `SolverResult(alpha_star=0.0, is_feasible=True, status="always_profitable", residual=abs(1.0 - R), iterations=0, ...)`.
     - Nếu $R > 1 + d \implies$ trả về `SolverResult(alpha_star=None, is_feasible=False, status="infeasible", residual=float("inf"), iterations=0, ...)`.
  3. Vùng có nghiệm ($1.0 < R \le 1 + d$): Thực thi tìm kiếm nhị phân 50 bước trên đoạn $[0.0, 1.0]$ sử dụng `topo.expected_acceptance(mid)`.
  4. Cài đặt `evaluate_speedup(topo, batch_size, alpha)`: Tính $S(\mathcal{T}, B, \alpha) = \frac{\mathbb{E}[\tau(\mathcal{T}, \alpha)] \cdot C(0, B)}{C(\mathcal{T}, B)}$.
  5. Lưu cache kết quả theo khóa `(topo_name, batch_size)` để phục vụ tra cứu tức thì ở các bước suy luận sau.
  6. Cung cấp hàm `calculate_alpha_star(k_or_topo, batch_size)` bọc ngoài để giữ tương thích ngược 100% với code và bài test cũ.
- **Người thực hiện:** Khôi (Lead 70%).

### Bước 3: Lựa chọn cây tối ưu (`select_optimal_topology`)
- **Nhiệm vụ:**
  1. Nhận danh sách cây ứng viên (mặc định `["T0", "T4", "T8", "T16", "T32"]`).
  2. Tính hệ số tăng tốc dự báo cho từng cây:
     $$S(\mathcal{T}, B, \alpha) = \frac{\mathbb{E}[\tau(\mathcal{T}, \alpha)] \cdot C(0, B)}{C(\mathcal{T}, B)}$$
  3. Áp dụng quy tắc ngắt toàn catalog: Nếu $\max S \le 1.0$, trả về `("T0", 1.0)`.
  4. Bảo lưu `select_optimal_budget` và `estimate_expected_tokens` cho code controller cũ.
- **Người thực hiện:** Khôi (Lead 70%).

### Bước 4: Xây dựng bộ kiểm thử tự động (`tests/test_cost_model_solver.py`)
- **Nhiệm vụ:**
  1. Kiểm tra nghiệm của chuỗi tuyến tính $T_4$ so với nghiệm giải tích tính tay.
  2. Kiểm tra tính đơn điệu của $\alpha^*(T, B)$ theo batch size $B$.
  3. Kiểm tra trường hợp vô nghiệm khi ép tỉ số chi phí nhân tạo $R > 1 + d$, bảo đảm trả về `is_feasible = False`.
  4. Kiểm tra điều kiện $R \le 1.0$ trả về `always_profitable` và $\alpha^* = 0.0$.
  5. Kiểm tra Fallback Gate chuyển về $T_0$ khi tốc độ $\le 1.0$.
  6. Đo đạc độ trễ giải nghiệm trên 1,000 lần lặp, bảo đảm $< 0.01\text{ ms}$ (và cache hit $< 0.001\text{ ms}$).
  7. Kiểm tra sai số phần dư $|S(\mathcal{T}, B, \alpha^*) - 1.0| < 10^{-4}$.
  8. Kiểm tra tương thích ngược hồi quy 100% với `tests/test_flexsds_baseline.py`, `tests/test_cost_model_l4.py`.
- **Người thực hiện:** Đại (Lead phần test 70%), Khôi (Hỗ trợ 30%).

### Bước 5: Thẩm định kỹ thuật và hoàn tất nghiệm thu
- **Nhiệm vụ:**
  1. Chạy toàn bộ test suite (`pytest tests/test_cost_model_solver.py tests/test_flexsds_baseline.py tests/test_tree_topology.py`).
  2. Viết worklog `logs/Khoi/M4-02/M4-02-03_worklogs.md`.
  3. Cập nhật trạng thái và checklist trên Trello card #48.
- **Người thực hiện:** Khôi và Đại.

---

## 4. Ma trận trách nhiệm (RACI Matrix)

| Hạng mục công việc | Cấn Đức Khôi | Bùi Đức Đại | Vai trò và sản phẩm cụ thể |
| :--- | :---: | :---: | :--- |
| **Đặc tả toán học và kiến trúc lớp** | **A / R (70%)** | C / I (30%) | Khôi viết spec/plan; Đại thẩm định tính đúng đắn toán học |
| **Nâng cấp `FlexCostModel` & Analytical Fallback** | **A / R (70%)** | C / I (30%) | Khôi code module nguồn `cost_model.py` |
| **Cài đặt Bisection Solver & No-Root Guard** | **A / R (70%)** | C / I (30%) | Khôi cài đặt thuật toán; Đại review điều kiện biên $1+d$ |
| **Xây dựng unit test suite** | C / I (30%) | **A / R (70%)** | Đại thiết kế test cases; Khôi hỗ trợ fixture dữ liệu |
| **Đo benchmark tốc độ và sai số** | C / I (30%) | **A / R (70%)** | Đại đo đạc latency, residual; Khôi tối ưu cache |
| **Viết worklog và PR review** | **A / R (70%)** | C / I (30%) | Khôi viết worklog; Đại review và duyệt PR |

*(Ghi chú: A = Accountable, R = Responsible, C = Consulted, I = Informed)*

---

## 5. Tiêu chí nghiệm thu (Definition of Done — DoD)

- [x] Lớp `FlexCostModel` nạp trơn tru cả tệp profile JSON đo thực nghiệm lẫn chế độ fallback giải tích.
- [x] Thuật toán bisection hội tụ nghiệm với sai số phần dư $|S - 1.0| < 10^{-4}$ trên toàn bộ các ca có nghiệm.
- [x] Độ trễ giải nghiệm trung bình $< 0.01\text{ ms}$ (và cache hit $< 0.001\text{ ms}$).
- [x] Xử lý chính xác các trường hợp biên ($R \le 1.0$ và $R > 1 + d$), không ném exception và không gán $\alpha^* = 1.0$ giả tạo.
- [x] 100% Pass bộ unit test mới `tests/test_cost_model_solver.py` và không gây hồi quy (regression) với các test cũ.
- [x] Hoàn tất worklog và cập nhật đầy đủ checklist trên Trello card #48.
