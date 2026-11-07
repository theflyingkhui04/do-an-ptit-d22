# Đặc tả kỹ thuật: Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động

> **Mã task Trello:** `#48 [M4-02-03] Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động alpha*(T, B)`  
> **Dự án:** FlexSDS — Hardware-Aware Speculative Decoding for Self-Hosted LLMs on Consumer GPUs  
> **Tác giả:** Cấn Đức Khôi (Lead 70%) & Bùi Đức Đại (Partner 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/flexCostModel/spec.md`  
> **Cơ sở đối chiếu:** Báo cáo thẩm định kỹ thuật Revision 2 (`tree (1).pdf`), chuẩn OPT-Tree (*Wang et al., 2025, TACL*) và module `tree_topology.py` đã nghiệm thu  
> **Trạng thái:** Bản đặc tả kỹ thuật chính thức (Revision 1.0 — 07/11/2026)

---

## 1. Tổng quan và bối cảnh kỹ thuật

Trong kiến trúc suy đoán thích ứng phần cứng (FlexSDS v0.2), **bộ mô hình chi phí phần cứng (`FlexCostModel`)** đóng vai trò trái tim tính toán của lớp điều khiển. Thay vì sử dụng các ngưỡng heuristic ghim cứng chủ quan (như $0.65$ và $0.80$ ở phiên bản v0.1), hệ thống chuyển sang **giải nghiệm hòa vốn động theo thời gian thực $\alpha^*(\mathcal{T}, B)$** cho từng cấu hình cây nháp $\mathcal{T} \in \{T_0, T_4, T_8, T_{16}, T_{32}\}$ tại mỗi mức tải đồng thời (batch size $B$).

### Các vấn đề cần giải quyết triệt để từ phiên bản cũ (theo Báo cáo thẩm định Revision 2):
1. **Khắc phục lỗi S08 (No-root handling và chặn trên lý thuyết):**  
   Phiên bản cũ kiểm tra điều kiện vô nghiệm lỏng lẻo bằng chặn $1 + N$. Tuy nhiên, trong giải mã tham lam (greedy decoding), độ dài chuỗi token được chấp nhận bị chặn trên bởi độ sâu tối đa của cây:
   $$\mathbb{E}[\tau(\mathcal{T}, \alpha)] \le 1 + d$$
   Do đó, nếu tỉ số chi phí $R(\mathcal{T}, B) \triangleq \frac{C(\mathcal{T}, B)}{C(0, B)} > 1 + d$, phương trình hòa vốn chắc chắn vô nghiệm trên $[0, 1]$. Phiên bản cũ gán $\alpha^* = 1.0$ giả định là sai bản chất toán học, khiến controller hiểu nhầm đây là một nghiệm hợp lệ và không kích hoạt cơ chế ngắt an toàn.
2. **Tích hợp mô hình kỳ vọng cây OPT-Tree và chốt hợp đồng toán học (Đóng vết S07):**  
   Thay thế hàm tính kỳ vọng chuỗi tuyến tính 1D cũ bằng hàm kỳ vọng đa ứng viên $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$ theo phương pháp `branch_aware` từ module `TreeTopology` đã kiểm chứng ở task `[M4-02-01]`, thiết lập mối liên kết toán học tất định giữa scalar acceptance $\alpha$ và xác suất chấp nhận phân nhánh.
3. **Phân định rõ nguồn gốc dữ liệu chi phí (Cost Provenance):**  
   Hỗ trợ nạp linh hoạt ma trận đo thực nghiệm đa chiều $C(\mathcal{T}, B)$ từ tệp profile JSON GPU (được đo bởi Đại ở task `[M4-02-02]`), đồng thời cung cấp cơ chế phân tích giải tích tự động (Analytical Fallback) khi chạy trên dòng GPU mới chưa có profile.
4. **Đạt chỉ tiêu hiệu năng siêu tốc (< 0.01 ms):**  
   Thời gian giải nghiệm $\alpha^*$ cho một cấu hình cây phải hoàn tất trong $< 0.01\text{ ms}$ ($10\ \mu\text{s}$) để không tạo ra độ trễ phụ trong vòng lặp serving.

---

## 2. Cơ sở toán học và thuật toán giải nghiệm

### 2.1. Phương trình điểm hòa vốn (Break-Even Equation)

Tại mức tải $B$ và cấu hình cây $\mathcal{T}$, hệ số tăng tốc dự báo lý thuyết $S(\mathcal{T}, B, \alpha)$ so với giải mã tuần tự cơ sở $T_0$ là:
$$S(\mathcal{T}, B, \alpha) = \frac{\mathbb{E}[\tau(\mathcal{T}, \alpha)] \cdot C(0, B)}{C(\mathcal{T}, B)}$$

Trong đó:
* $C(0, B)$ là chi phí forward một token tuần tự của mô hình đích (Target Model) tại batch size $B$.
* $C(\mathcal{T}, B)$ là tổng chi phí thực hiện một bước suy đoán (sinh nháp bằng đầu nháp EAGLE và xác thực song song bằng mô hình đích qua `tree_mask`).
* $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$ là kỳ vọng số token được phát sinh (gồm token gốc và các token nháp được chấp nhận).

Điểm hòa vốn xuất hiện khi tốc độ suy đoán bằng đúng tốc độ tuần tự, tức $S(\mathcal{T}, B, \alpha^*) = 1.0$:
$$\mathbb{E}[\tau(\mathcal{T}, \alpha^*)] = R(\mathcal{T}, B)$$
Với $R(\mathcal{T}, B) \triangleq \frac{C(\mathcal{T}, B)}{C(0, B)}$ là **tỉ số chi phí suy đoán (Cost Ratio)**.

### 2.2. Phân tích miền nghiệm và điều kiện biên (Boundary Analysis)

Theo đặc tả của `TreeTopology`, hàm $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$ có các tính chất:
1. Liên tục và đơn điệu tăng thực sự theo $\alpha \in [0.0, 1.0]$.
2. $\mathbb{E}[\tau(\mathcal{T}, 0.0)] = 1.0$ (chỉ token gốc được chấp nhận).
3. $\mathbb{E}[\tau(\mathcal{T}, 1.0)] = 1 + d$ (đường dẫn dài nhất đạt độ sâu tối đa $d$).

Từ đó, xét phương trình sai số $f(\alpha) \triangleq \mathbb{E}[\tau(\mathcal{T}, \alpha)] - R(\mathcal{T}, B) = 0$ trên đoạn $[0.0, 1.0]$:

| Trường hợp | Điều kiện toán học | Trạng thái nghiệm (`status`) | Giá trị $\alpha^*$ trả về | Ý nghĩa thực tiễn |
| :--- | :--- | :--- | :---: | :--- |
| **Cực biên dưới** | $R(\mathcal{T}, B) \le 1.0$ | `always_profitable` | `0.0` | Bước suy đoán có chi phí bằng hoặc rẻ hơn cả tuần tự. Mọi mức chấp thuận $\alpha \ge 0$ đều sinh lời. |
| **Vùng nghiệm duy nhất** | $1.0 < R(\mathcal{T}, B) \le 1 + d$ | `converged` | $\alpha^* \in (0.0, 1.0]$ | Tồn tại duy nhất một nghiệm theo định lý giá trị trung gian (IVT) vì $f(0) < 0$ và $f(1) \ge 0$. |
| **Vô nghiệm (Infeasible)** | $R(\mathcal{T}, B) > 1 + d$ | `infeasible` | `None` (`is_feasible=False`) | Tỉ số chi phí vượt quá giới hạn lý thuyết tối đa của cây. Tuyệt đối không sinh lời với bất kỳ $\alpha \le 1.0$. |

> [!IMPORTANT]
> **Quy tắc xử lý vô nghiệm (S08):** Khi $R(\mathcal{T}, B) > 1 + d$, bộ giải bắt buộc trả về cấu trúc kết quả có cờ `is_feasible = False` và trạng thái `status = "infeasible"`. Tuyệt đối **không gán $\alpha^* = 1.0$** để tránh che giấu thông tin cấu hình không khả thi đối với bộ điều khiển.

### 2.3. Thuật toán tìm kiếm nhị phân (Bisection Root Solver)

Đối với trường hợp có nghiệm duy nhất ($1.0 < R(\mathcal{T}, B) \le 1 + d$), thuật toán chia đôi được cài đặt như sau:
1. Đặt khoảng nghiệm $[low, high] = [0.0, 1.0]$.
2. Thực hiện vòng lặp $K = 50$ bước:
   $$mid = \frac{low + high}{2.0}$$
   Tính giá trị hàm mục tiêu: $y = \mathbb{E}[\tau(\mathcal{T}, mid)]$.
   - Nếu $y < R(\mathcal{T}, B)$: gán $low = mid$.
   - Ngược lại: gán $high = mid$.
3. Nghiệm xấp xỉ: $\alpha^* = \frac{low + high}{2.0}$.

**Đánh giá sai số số học:**
* Độ rộng khoảng nghiệm sau 50 bước:
  $$\Delta = \frac{1 - 0}{2^{50}} \approx 8.88 \times 10^{-16}$$
  Đạt mức giới hạn độ chính xác của số thực dấu phẩy động 64-bit (IEEE 754 float64).
* Sai số phần dư số học (Numerical Residual):
  $$|\mathbb{E}[\tau(\mathcal{T}, \alpha^*)] - R(\mathcal{T}, B)| < 10^{-4}$$
  (Thực tế đạt mức $\le 10^{-14}$, vượt xa yêu cầu của DoD).

---

## 3. Kiến trúc phần mềm và mô hình dữ liệu

### 3.1. Cấu trúc kết quả giải nghiệm (`SolverResult`)

```python
@dataclass(frozen=True)
class SolverResult:
    """Represents the outcome of a break-even acceptance rate solver."""
    alpha_star: Optional[float]
    is_feasible: bool
    status: str  # "converged" | "always_profitable" | "infeasible"
    cost_ratio: float
    residual: float
    iterations: int
    topology_name: str
    batch_size: int
```

### 3.2. Lớp `FlexCostModel` nâng cấp

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LỚP FLEXCOSTMODEL V0.2                          │
├────────────────────────────────────────────────────────────────────────┤
│ Thuộc tính:                                                            │
│   • matrix: Dict[Tuple[Union[str, int], int], float] # Chi phí C(T, B) │
│   • forward_target: Dict[int, float]          # Chi phí C_T(B)         │
│   • forward_draft: Dict[int, float]           # Chi phí C_D(B)         │
│   • metadata: Dict[str, Any]                  # Metadata phần cứng     │
│   • provenance: str                           # "measured" | "model"   │
│   • _cache_alpha_star: Dict[...]              # Bảng cache nghiệm      │
├────────────────────────────────────────────────────────────────────────┤
│ Phương thức công khai:                                                 │
│   • get_step_cost(topo, batch_size) -> float                           │
│   • solve_alpha_star(topo, batch_size, max_iter) -> SolverResult       │
│   • evaluate_speedup(topo, batch_size, alpha) -> float                 │
│   • select_optimal_topology(alpha, batch_size, candidates) -> Tuple    │
│   • calculate_alpha_star(k_or_topo, batch_size) -> float (compat)      │
│   • estimate_expected_tokens(k, alpha) -> float (compat)               │
│   • select_optimal_budget(alpha, batch_size, candidates) (compat)      │
│   • from_json(path_or_dict) -> FlexCostModel                           │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.3. Cơ chế xuất xứ dữ liệu chi phí (Cost Provenance)

Hệ thống phân định ranh giới rõ ràng giữa 2 chế độ chi phí:

1. **Chế độ đo thực nghiệm (Measured Mode — `provenance: "measured"`):**
   * Nạp từ tệp JSON profile do task `[M4-02-02]` đo trên GPU NVIDIA L4 hoặc Tesla T4 (`src/flexsds/profiles/profile_sm89_l4.json`).
   * Schema hỗ trợ cả khóa truyền thống `"k{k}_b{b}"` lẫn khóa cấu hình cây chuẩn `"topo_{name}_b{b}"` (ví dụ `"topo_T16_b4": 24.85`).
2. **Chế độ phân tích giải tích (Analytical Fallback — `provenance: "analytical"`):**
   * Kích hoạt tự động khi chạy trên GPU chưa có tệp profile tương thích.
   * Công thức mô hình hóa chi phí suy đoán:
     $$C(\mathcal{T}, B) = t_{\text{draft}}(\mathcal{T}, B) + C_V(\mathcal{T}, B)$$
     - Chi phí sinh nháp EAGLE: $t_{\text{draft}}(\mathcal{T}, B) = N \cdot t_{\text{step}}(B)$ với $t_{\text{step}} = C_D(B)$ (hoặc xấp xỉ $1.0 - 1.5\text{ ms}$).
     - Chi phí xác thực song song qua Tree Attention Mask:
       $$C_V(\mathcal{T}, B) = C_T(B) \cdot \left( 1 + \gamma_{\text{attn}} \cdot \max(0, N - 1) \right)$$
       Với $\gamma_{\text{attn}} = 0.008$ (mức phạt $0.8\%$ trên mỗi token nháp trên kiến trúc Ada Lovelace sm_89, hoặc $0.012$ trên sm_75 Turing).

---

## 4. Đặc tả lựa chọn cây tối ưu (Topology Selection Logic)

Tại bước điều khiển runtime, bộ điều khiển gọi hàm:
```python
select_optimal_topology(
    alpha: float,
    batch_size: int = 1,
    candidate_topologies: Optional[List[str]] = None,
) -> Tuple[str, float]
```

Quy tắc ra quyết định:
1. Danh mục ứng viên mặc định: `["T0", "T4", "T8", "T16", "T32"]`.
2. Duyệt qua từng cấu hình ứng viên $\mathcal{T}$:
   * Tính $S(\mathcal{T}, B, \alpha)$.
   * Cập nhật cấu hình có $S$ cao nhất.
3. **Quy tắc chuyển làn an toàn (Catalog-Wide Fallback Gate):**
   * Nếu $\max_{\mathcal{T}} S(\mathcal{T}, B, \alpha) \le 1.0$: Trả về `("T0", 1.0)`.
   * Tuyệt đối không chọn cây có dự báo tăng tốc âm để phục vụ suy luận.

---

## 5. Tiêu chuẩn nghiệm thu (Definition of Done — DoD)

- [x] **Tương thích ngược 100%:** Giữ nguyên các hàm `from_json`, `get_forward_costs`, `calculate_alpha_star(k, batch_size)` để toàn bộ 18 test cases hiện hành của `test_flexsds_baseline.py` và `test_cost_model_l4.py` tiếp tục PASS mà không cần sửa code kiểm thử cũ.
- [x] **Thời gian thực thi:** Gọi `solve_alpha_star` hoàn tất trong $< 0.01\text{ ms}$ cho mỗi lượt tính toán (có bảng cache nghiệm tức thì $< 0.001\text{ ms}$).
- [x] **Độ chính xác nghiệm:** Residual sai số $|S(\mathcal{T}, B, \alpha^*) - 1.0| < 10^{-4}$ trên toàn bộ các ca kiểm thử có nghiệm.
- [x] **Bảo đảm không sập (Zero-Crash):** Xử lý chính xác các ca vô nghiệm $R > 1 + d$, trả về `is_feasible = False`, không ném Exception ra ngoài.
- [x] **Bộ unit test độc lập:** Viết `tests/test_cost_model_solver.py` bao phủ đầy đủ các ca kiểm thử tổng hợp (synthetic cases), kiểm tra điều kiện biên và so sánh với nghiệm giải tích biết trước.
