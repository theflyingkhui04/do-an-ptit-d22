# [M4-02-03] Worklog Khôi — Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động alpha*(T, B)

> **Task Trello:** `#48 [M4-02-03] Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động alpha*(T, B)`  
> **Người thực hiện:** Cấn Đức Khôi (Lead 70%) và Bùi Đức Đại (Partner 30%)  
> **Nhánh phát triển:** `feat/m4-02-03-cost-model-solver`  
> **Trạng thái:** ✅ **HOÀN THÀNH (100% PASS 37/37 unit tests & solver verification)**

---

## 1. Mục tiêu và bối cảnh kỹ thuật

Khắc phục triệt để các tồn đọng kỹ thuật từ Báo cáo thẩm định Revision 2 (`tree (1).pdf`):
1. **Khắc phục lỗi S08 (No-root handling và chặn trên lý thuyết):**  
   - Thiết lập chặn trên chặt chẽ: $\mathbb{E}[\tau(\mathcal{T}, \alpha)] \le 1 + d$.
   - Khi tỉ số chi phí $R(\mathcal{T}, B) > 1 + d$, phương trình hòa vốn chắc chắn vô nghiệm. Trả về cấu trúc `SolverResult` với `is_feasible = False`, `alpha_star = None`, `status = "infeasible"`. Tuyệt đối không gán $\alpha^* = 1.0$ giả định.
2. **Tích hợp mô hình kỳ vọng cây OPT-Tree (Đóng vết S07):**  
   - Kết nối trực tiếp hàm $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$ từ module `TreeTopology` (M4-02-01) với phương pháp `branch_aware`, bảo đảm tính đơn điệu tăng và tính chất biên.
3. **Phân định rõ nguồn gốc chi phí (Cost Provenance):**  
   - Nạp linh hoạt ma trận thực nghiệm đa chiều từ profile JSON GPU qua cả hai dạng khóa `"topo_{name}_b{b}"` và `"k{k}_b{b}"` (`provenance = "measured"`).
   - Tự động kích hoạt cơ chế tính giải tích `Analytical Fallback` ($C = t_{\text{draft}} + C_V$) khi chạy trên GPU chưa có profile.
4. **Quy tắc chuyển làn an toàn toàn catalog (S09):**  
   - Hàm `select_optimal_topology` quét toàn bộ danh mục cây ứng viên. Nếu $\max_{\mathcal{T}} S(\mathcal{T}, B, \alpha) \le 1.0$, ngắt ngay về `("T0", 1.0)`.
5. **Hiệu năng giải nghiệm thời gian thực:**  
   - Bisection search 50 bước đạt độ chính xác float64 với sai số phần dư $|S - 1.0| < 10^{-4}$ (thực tế $< 10^{-14}$).
   - Tích hợp bảng cache $O(1)$ giảm độ trễ tra cứu xuống $< 0.001\text{ ms}$.
6. **Bảo toàn 100% tương thích ngược:**  
   - Giữ nguyên các hàm `calculate_alpha_star`, `estimate_expected_tokens`, `select_optimal_budget` để toàn bộ test cũ của `TreeAdaptiveController` và baseline tiếp tục PASS.

---

## 2. Kết quả hiện thực hóa và kiểm chứng

### A. Cấu trúc dữ liệu `SolverResult`

```python
@dataclass(frozen=True)
class SolverResult:
    alpha_star: Optional[float]
    is_feasible: bool
    status: str  # "converged" | "always_profitable" | "infeasible"
    cost_ratio: float
    residual: float
    iterations: int
    topology_name: str
    batch_size: int
```

### B. Bảng phân tích miền nghiệm và điều kiện biên

| Trường hợp | Điều kiện toán học | Trạng thái nghiệm (`status`) | Giá trị $\alpha^*$ trả về | Trạng thái khả thi |
| :--- | :--- | :--- | :---: | :---: |
| **Cực biên dưới** | $R(\mathcal{T}, B) \le 1.0$ | `always_profitable` | `0.0` | `True` |
| **Vùng nghiệm duy nhất** | $1.0 < R(\mathcal{T}, B) \le 1 + d$ | `converged` | $\alpha^* \in (0.0, 1.0]$ | `True` |
| **Vô nghiệm (Lỗi S08)** | $R(\mathcal{T}, B) > 1 + d$ | `infeasible` | `None` | `False` |

---

## 3. Các artifact đã tạo và chỉnh sửa

1. **Module mã nguồn cốt lõi:**
   - `src/flexsds/core/cost_model.py`: Bổ sung `SolverResult`, `_resolve_topology`, `solve_alpha_star`, `evaluate_speedup`, `select_optimal_topology`, cache $O(1)$, Analytical Fallback.
   - `src/flexsds/core/__init__.py`: Export `FlexCostModel`, `SolverResult`, `load_profile_for_hardware`.
2. **Bộ kiểm thử tự động:**
   - `tests/test_cost_model_solver.py`: 11 test cases bao phủ toàn diện boundary conditions, S08 infeasibility, bisection convergence, speedup evaluation, cache latency, fallback gate và backward compatibility.
3. **Tài liệu đặc tả và kế hoạch:**
   - `logs/Khoi/spec&plan/flexCostModel/spec.md`: Cập nhật Revision 1.1 đồng bộ $C_V$, đóng vết S07, chuẩn hóa `SolverResult`.
   - `logs/Khoi/spec&plan/flexCostModel/plan.md`: Cập nhật Revision 1.1 chi tiết hóa các bước TDD.
4. **Nhật ký kỹ thuật:**
   - `logs/Khoi/M4-02/M4-02-03_worklogs.md`: Báo cáo nghiệm thu task M4-02-03.

---

## 4. Biên bản kiểm thử (Test Execution Summary)

```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22
configfile: pyproject.toml
plugins: typeguard-4.4.4
collected 37 items

tests/test_flexsds_baseline.py::test_flexsds_version PASSED               [  2%]
tests/test_flexsds_baseline.py::test_banner_rendering PASSED              [  5%]
tests/test_flexsds_baseline.py::test_hardware_detector PASSED            [  8%]
tests/test_flexsds_baseline.py::test_compatibility_guard_eagle_pairing PASSED [ 10%]
tests/test_flexsds_baseline.py::test_compatibility_guard_oom_detection PASSED [ 13%]
tests/test_flexsds_baseline.py::test_cost_profile_loader_sm89 PASSED     [ 16%]
tests/test_flexsds_baseline.py::test_tree_adaptive_controller_budget_selection PASSED [ 18%]
tests/test_cost_model_l4.py::test_hardware_cost_model_loading PASSED     [ 21%]
tests/test_cost_model_l4.py::test_forward_and_verification_costs PASSED  [ 24%]
tests/test_cost_model_l4.py::test_cost_lookup_and_fallbacks PASSED       [ 27%]
tests/test_cost_model_l4.py::test_expected_tokens_calculation PASSED     [ 29%]
tests/test_cost_model_l4.py::test_speedup_calculation PASSED             [ 32%]
tests/test_cost_model_l4.py::test_break_even_alpha_star PASSED          [ 35%]
tests/test_cost_model_l4.py::test_optimal_budget_selection PASSED        [ 37%]
tests/test_cost_model_l4.py::test_controller_integration PASSED          [ 40%]
tests/test_tree_topology.py::test_list_and_get_topologies PASSED         [ 43%]
tests/test_tree_topology.py::test_canonical_catalog_invariants PASSED    [ 45%]
tests/test_tree_topology.py::test_acyclic_and_topological_ordering PASSED [ 48%]
tests/test_tree_topology.py::test_tree_16_depth_invariant_fixed PASSED   [ 51%]
tests/test_tree_topology.py::test_tree_32_unique_topology_and_depth PASSED [ 54%]
tests/test_tree_topology.py::test_causal_mask_invariants PASSED          [ 56%]
tests/test_tree_topology.py::test_branch_exclusivity_no_sibling_leakage PASSED [ 59%]
tests/test_tree_topology.py::test_expected_acceptance_bounds_and_monotonicity PASSED [ 62%]
tests/test_tree_topology.py::test_fast_lookup_table_speed_and_accuracy PASSED [ 64%]
tests/test_tree_topology.py::test_serialization_roundtrip PASSED         [ 67%]
tests/test_tree_topology.py::test_invalid_topologies_raise_error PASSED  [ 70%]
tests/test_cost_model_solver.py::test_solver_result_dataclass PASSED     [ 72%]
tests/test_cost_model_solver.py::test_resolve_topology_support PASSED     [ 75%]
tests/test_cost_model_solver.py::test_cost_model_from_json_tree_and_legacy_keys PASSED [ 78%]
tests/test_cost_model_solver.py::test_analytical_fallback_calculation PASSED [ 81%]
tests/test_cost_model_solver.py::test_solve_alpha_star_always_profitable PASSED [ 83%]
tests/test_cost_model_solver.py::test_solve_alpha_star_infeasible_s08_handling PASSED [ 86%]
tests/test_cost_model_solver.py::test_solve_alpha_star_convergence_and_residual PASSED [ 89%]
tests/test_cost_model_solver.py::test_evaluate_speedup_values PASSED     [ 91%]
tests/test_cost_model_solver.py::test_cache_alpha_star_performance PASSED [ 94%]
tests/test_cost_model_solver.py::test_select_optimal_topology_and_fallback_gate PASSED [ 97%]
tests/test_cost_model_solver.py::test_backward_compatibility_wrappers PASSED [100%]

============================== 37 passed in 0.10s ==============================
```
