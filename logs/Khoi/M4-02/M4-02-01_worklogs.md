# [M4-02-01] Worklog Khôi — Hiện thực hóa module cấu trúc topo cây nháp tree_topology.py

> **Task Trello:** `#46 [M4-02-01] Hiện thực hóa module cấu trúc topo cây nháp tree_topology.py`  
> **Người thực hiện:** Cấn Đức Khôi (Lead 70%) và Bùi Đức Đại (Partner 30%)  
> **Nhánh phát triển:** `feat/m4-02-01-tree-topology`  
> **Trạng thái:** ✅ **HOÀN THÀNH (100% PASS 11/11 unit tests & validator ground truth)**

---

## 1. Mục tiêu và bối cảnh kỹ thuật

Thực hiện khuyến nghị từ báo cáo thẩm định kỹ thuật Revision 2 của Bùi Đức Đại (`tree (1).pdf`):
1. **Khắc phục lỗi S04 ($T_{16}$):** Node 16 ở bản thiết kế cũ có cha là node 14 (ở độ sâu 4), khiến độ sâu thực tế của node 16 bị đẩy lên 5 thay vì 4. Cần đổi nút cha sang node ở độ sâu 3 (node 13) để đảm bảo độ sâu tối đa $d_{\max} = 4$.
2. **Khắc phục lỗi S05 ($T_{32}$):** Xác định mảng quan hệ cha-con (`parent_indices`) 33 phần tử duy nhất và tất định, tuân thủ đúng phân bố tầng $[1, 4, 8, 8, 6, 6]$, độ sâu $d_{\max} = 5$.
3. **Hiện thực hóa lớp `TreeTopology`:** Quản lý toàn bộ 5 cấu hình cây rời rạc chuẩn ($T_0, T_4, T_8, T_{16}, T_{32}$), tự động tính toán `node_depths`, `position_ids`, ma trận chú ý cây 2 chiều (`tree_mask`) và bảng tra cứu kỳ vọng $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$.
4. **Kiểm chứng tính chất nhân quả (Causal Mask Invariants):** Đảm bảo tính phản xạ, tam giác dưới và tính xung khắc giữa các nhánh anh em (branch exclusivity / no sibling leakage).
5. **Tra cứu kỳ vọng token siêu nhanh:** Bảng tra cứu $O(1)$ đạt độ trễ $< 0.005\text{ ms}$ cho mỗi lượt tra cứu.

---

## 2. Kết quả hiện thực hóa và kiểm chứng

### A. Cấu trúc 5 cây nháp chuẩn hóa (Canonical Catalog)

| Topo | Số token nháp ($N$) | Độ sâu ($d_{\max}$) | Phân bố tầng $[l_0, \dots, l_d]$ | Chi tiết mảng cha `parent_indices` |
| :--- | :---: | :---: | :--- | :--- |
| **$T_0$** | 0 | 0 | `[1]` | `[-1]` |
| **$T_4$** | 4 | 4 | `[1, 1, 1, 1, 1]` | `[-1, 0, 1, 2, 3]` |
| **$T_8$** | 8 | 3 | `[1, 2, 4, 2]` | `[-1, 0, 0, 1, 1, 2, 2, 3, 5]` |
| **$T_{16}$** | 16 | 4 | `[1, 3, 6, 4, 3]` | `[-1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 6, 8, 10, 12, 13]` |
| **$T_{32}$** | 32 | 5 | `[1, 4, 8, 8, 6, 6]` | `[-1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]` |

* **Đường dẫn node 16 trên $T_{16}$:** $0 \to 3 \to 8 \to 13 \to 16$ (chiều dài 4 cạnh, độ sâu đúng bằng 4).
* **Kiểm tra tính phi chu trình (Acyclicity):** $p(i) < i$ với mọi $i \ge 1$.

### B. Ma trận mặt nạ chú ý cây 2 chiều (2D Tree Attention Mask)

* Ma trận $M \in \{0, 1\}^{(N+1) \times (N+1)}$ thỏa mãn chặt chẽ các bất biến:
  1. **Tính phản xạ (Reflexive):** $M_{i, i} = 1$ với mọi $i \in [0, N]$.
  2. **Tính nhân quả (Causality):** $M_{i, j} = 0$ với mọi $j > i$ (không nhìn thấy các node trong tương lai).
  3. **Tổ tiên duy nhất (Ancestor trace):** $M_{i, j} = 1$ khi và chỉ khi $j$ nằm trên đường dẫn từ gốc $0$ tới $i$.
  4. **Tính xung khắc nhánh (Branch Exclusivity):** Hai node $i \neq j$ có cùng cha $p$ tuyệt đối không thể chú ý đến nhau ($M_{i, j} = 0$ và $M_{j, i} = 0$).

### C. Hàm kỳ vọng chấp thuận token $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$ và Lookup Table

* Áp dụng mô hình bao phủ đa ứng viên theo OPT-Tree (*Wang et al., 2025*):
  $$P_{\text{step}}(\ell, \alpha) = 1 - (1 - \alpha)^{n_\ell}$$
  $$P_{\text{path}}(\ell, \alpha) = \prod_{k=1}^\ell P_{\text{step}}(k, \alpha)$$
  $$\mathbb{E}[\tau(\mathcal{T}, \alpha)] = 1 + \sum_{\ell=1}^d P_{\text{path}}(\ell, \alpha)$$
* **Bảo đảm giới hạn toán học:**
  * $\alpha = 0.0 \implies \mathbb{E}[\tau] = 1.0$.
  * $\alpha = 1.0 \implies \mathbb{E}[\tau] = 1 + d$.
  * Đơn điệu tăng trên toàn bộ miền $\alpha \in [0.0, 1.0]$.
* **Hiệu năng bảng tra cứu (Lookup Latency):**
  * Đo đạc thực tế qua 10,000 lần lặp: độ trễ trung bình đạt **$0.000266\text{ ms}$ ($0.266\ \mu\text{s}$)**, vượt xa chỉ tiêu DoD ($< 0.005\text{ ms}$).

---

## 3. Các artifact đã tạo và kiểm thử

1. **Script kiểm chuẩn độc lập (Validator Spike):** `scripts/tree_validator.py`
2. **Module mã nguồn cốt lõi:** `src/flexsds/core/tree_topology.py`
3. **Cập nhật module core:** `src/flexsds/core/__init__.py`
4. **Bộ kiểm thử tự động:** `tests/test_tree_topology.py` (11 test cases, 100% PASS)
5. **Cập nhật tài liệu kỹ thuật:** `logs/Khoi/spec&plan/treeAdaptive/spec.md` (đồng bộ mảng cha $T_{16}, T_{32}$, chặn hòa vốn $1+d$, xử lý vô nghiệm no-root và quy tắc fallback).

---

## 4. Biên bản kiểm thử (Test Execution Summary)

```text
tests/test_tree_topology.py::test_list_and_get_topologies PASSED         [  9%]
tests/test_tree_topology.py::test_canonical_catalog_invariants PASSED    [ 18%]
tests/test_tree_topology.py::test_acyclic_and_topological_ordering PASSED [ 27%]
tests/test_tree_topology.py::test_tree_16_depth_invariant_fixed PASSED   [ 36%]
tests/test_tree_topology.py::test_tree_32_unique_topology_and_depth PASSED [ 45%]
tests/test_tree_topology.py::test_causal_mask_invariants PASSED          [ 54%]
tests/test_tree_topology.py::test_branch_exclusivity_no_sibling_leakage PASSED [ 63%]
tests/test_tree_topology.py::test_expected_acceptance_bounds_and_monotonicity PASSED [ 72%]
tests/test_tree_topology.py::test_fast_lookup_table_speed_and_accuracy PASSED [ 81%]
tests/test_tree_topology.py::test_serialization_roundtrip PASSED         [ 90%]
tests/test_tree_topology.py::test_invalid_topologies_raise_error PASSED  [100%]
============================== 11 passed in 0.04s ==============================
```
