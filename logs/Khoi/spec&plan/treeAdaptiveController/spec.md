# Đặc tả kỹ thuật: Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng

> **Mã task Trello:** `#50 [M4-02-05] Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng`  
> **Dự án:** FlexSDS — Hardware-Aware Speculative Decoding for Self-Hosted LLMs on Consumer GPUs  
> **Tác giả:** Cấn Đức Khôi (Lead 70%) & Bùi Đức Đại (Partner 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptiveController/spec.md`  
> **Cơ sở đối chiếu:** Báo cáo thẩm định kỹ thuật Revision 2 (`tree (1).pdf`), chuẩn OPT-Tree (*Wang et al., 2025, TACL*), module `tree_topology.py` (M4-02-01) và `FlexCostModel` (M4-02-03) đã nghiệm thu  
> **Trạng thái:** Bản đặc tả kỹ thuật chính thức (Revision 1.0 — 08/11/2026)

---

## 1. Tổng quan và bối cảnh kỹ thuật (Intent & The Why)

Trong chiến dịch đo kiểm hiệu năng tại mốc M3-05, bộ điều khiển thích ứng cơ sở v0.1 (`AdaptiveDraftController`) gặp hiện tượng suy giảm tốc độ (negative speedup) nghiêm trọng trên tập dữ liệu hội thoại tự do (Workload W3) và tại các mức concurrency cao ($c \ge 16$). Hệ số tăng tốc đo thực tế trên GPU NVIDIA L4 rơi xuống dưới 1.0 (dao động từ $0.745\times$ xuống $0.474\times$).

### Các nguyên nhân cốt lõi:
1. **Không gian nháp tuyến tính 1D ($k \in \{0, 2, 3, 4\}$):** Xác suất chấp nhận chuỗi thẳng suy giảm theo cấp số nhân $\alpha^k$. Chỉ cần một token ở đầu chuỗi bị từ chối, toàn bộ chi phí sinh nháp và kiểm chứng phía sau bị lãng phí.
2. **Ngưỡng quyết định tĩnh ghim cứng (`0.65` và `0.80`):** Không phản ánh đúng sự biến thiên phi tuyến của chi phí tính toán trên GPU khi kích thước batch thay đổi.
3. **Thiếu cơ chế kẹp tải đồng thời:** Khi nhiều yêu cầu cùng chạy, việc giữ kích thước nháp lớn gây quá tải tính toán và tranh chấp băng thông bộ nhớ.
4. **Bảo vệ bộ nhớ lỏng lẻo:** Chưa có cơ chế bảo vệ hai tầng kết hợp giữa dung lượng VRAM thực tế và bộ đệm KV-cache.

Task **[M4-02-05]** hiện thực hóa lớp `TreeAdaptiveController` tại đường dẫn chuẩn runtime `src/proposer/tree_adaptive.py` phục vụ việc serving trực tiếp qua `src/proposer/adapter.py`. Bộ điều khiển v0.2 nâng cấp toàn diện lên cấu trúc cây nháp 2D ($T_0, T_4, T_8, T_{16}, T_{32}$), tích hợp giải nghiệm hòa vốn động theo mô hình chi phí phần cứng và cơ chế phòng vệ hai tầng.

---

## 2. Mục tiêu, phi mục tiêu và phản mục tiêu (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục tiêu (Goals)
- **Hiện thực hóa `TreeAdaptiveController` tại `src/proposer/tree_adaptive.py`:** Kế thừa trực tiếp từ `BudgetController` để bảo toàn tính tương thích của hệ thống.
- **Mở rộng cấu trúc quyết định `TreeAdaptiveDecision`:** Kế thừa từ `BudgetDecision`, duy trì trường `k` (gán bằng số node $N$) đồng thời bổ sung `topology_name`, `num_nodes`, `alpha_effective`, `alpha_star`, `speedup`.
- **Chuẩn hóa tỷ lệ chấp nhận theo độ sâu cây:** Sử dụng độ sâu $d$ của topo vừa phát hành để chuẩn hóa bước $\alpha_{\text{step}} = \min(1.0, \text{num\_accepted} / \max(d, 1))$, tránh việc chia cho tổng số node $N$ làm sập tín hiệu $\alpha$.
- **Cơ chế phòng vệ bộ nhớ hai tầng (Dual-Tier Memory Guard):**
  - Ngắt về $T_0$ ($N=0$) ngay lập tức khi: $\text{free\_vram\_mb} < 1500\text{ MB}$ HOẶC $\text{kv\_cache\_usage\_pct} > 90.0\%$.
  - Duy trì độ trễ hồi phục (hysteresis cooldown) tối thiểu 3 bước liên tiếp ở $T_0$.
  - Chỉ cho phép khôi phục suy đoán khi: $\text{kv\_cache\_usage\_pct} \le 80.0\%$ VÀ $\text{free\_vram\_mb} \ge 1500\text{ MB}$. Nếu VRAM vẫn dưới 1500 MB, tuyệt đối không khôi phục.
  - Cấm hoàn toàn việc phát bước thăm dò (probing) khi Memory Guard đang hoạt động dưới bất kỳ nguyên nhân nào, đồng thời reset bộ đếm thăm dò về 0.
- **Kẹp trần kích thước cây theo luồng đồng thời (Concurrency Throttling):**
  - $B \le 4$: Cho phép cấu hình tối đa $T_{32}$ ($N \le 32$).
  - $4 < B \le 12$: Kẹp trần tối đa $T_{16}$ ($N \le 16$).
  - $12 < B \le 24$: Kẹp trần tối đa $T_8$ ($N \le 8$).
  - $B > 24$: Kẹp trần tối đa $T_4$ ($N \le 4$), đồng thời ngắt về $T_0$ nếu $\alpha_{\text{effective}} < 0.85$.
- **Cơ chế chuyển làn an toàn và thăm dò thích ứng (Dynamic Fallback Gate & Adaptive Probing):**
  - Sử dụng hàm `FlexCostModel.select_optimal_topology()` để chọn topo có hệ số tăng tốc dự báo lớn nhất $S^*$. Nếu $S^* \le 1.0$, ngắt an toàn về $T_0$.
  - Khi đang ở trạng thái ngắt an toàn $T_0$, định kỳ phát bước thăm dò với $T_4$ sau mỗi $T_{\text{probe}}(B)$ bước giải mã:
    $$T_{\text{probe}}(B) = \begin{cases} 15 \text{ bước}, & B \le 4 \\ 20 \text{ bước}, & 4 < B \le 16 \\ 30 \text{ bước}, & B > 16 \end{cases}$$
- **Nguyên tắc không sập hệ thống (Zero-Crash):** Bọc toàn bộ logic ra quyết định trong khối bắt ngoại lệ, tự động fallback an toàn về $T_0$ nếu có lỗi tính toán phát sinh.
- **Đấu nối vào `src/proposer/adapter.py`:** Cho phép `AdaptiveProposerAdapter` khởi tạo và điều phối `TreeAdaptiveController`.
- **Hiệu năng ra quyết định thời gian thực:** Thời gian thực thi `select_budget()` phải đạt $< 0.05\text{ ms}$ cho mỗi lượt gọi.

### 2.2. Phi mục tiêu (Non-Goals)
- Không can thiệp vào logic lập lịch bất đồng bộ của vLLM (`AdaptiveAsyncScheduler` và `AdaptiveBudgetPlanner` thuộc phạm vi task M4-02-04).
- Không tự thu thập hay chạy micro-benchmark đo ma trận chi phí mới trên GPU (thuộc phạm vi task M4-02-02).
- Không thực hiện chiến dịch đo kiểm quét tải sweep toàn diện 4 làn trên hạ tầng đám mây Modal (thuộc phạm vi task M4-02-06).

### 2.3. Phản mục tiêu (Anti-Goals)
- Tuyệt đối không để xảy ra hiện tượng tràn bộ nhớ VRAM (OOM) trong quá trình serving do việc sinh cây nháp.
- Tuyệt đối không thăm dò vượt rào khi bộ nhớ GPU đang ở mức nguy hiểm.
- Tuyệt đối không phá vỡ giao diện công khai của `BudgetController` khiến các unit test cũ hoặc adapter bị lỗi biên dịch.

---

## 3. Các ràng buộc kỹ thuật (Constraints)

- **Môi trường chạy:** Hoạt động hoàn toàn trên CPU host trong vòng lặp đề xuất nháp của proposer, không được tạo thêm kernel GPU phụ trợ gây trễ luồng.
- **Danh mục topo cây:** Sử dụng chính xác 5 cấu hình chuẩn đã được đóng băng từ `src.flexsds.core.tree_topology`:
  - $T_0$ ($N=0, d=0$)
  - $T_4$ ($N=4, d=4$)
  - $T_8$ ($N=8, d=3$)
  - $T_{16}$ ($N=16, d=4$)
  - $T_{32}$ ($N=32, d=5$)
- **Mô hình chi phí:** Tích hợp với `FlexCostModel` từ `src.flexsds.core.cost_model` (đã hỗ trợ cả nạp profile thực nghiệm lẫn tính toán giải tích Analytical Fallback).
- **Bộ khử lệch khởi động (Unbiased EMA):** Sử dụng hệ số $\beta = 0.8$ với công thức hiệu chỉnh:
  $$\alpha_{\text{unbiased}} = \frac{\alpha_{\text{ema}} - \alpha_{\text{init}} \cdot \beta^t}{1 - \beta^t}$$

---

## 4. Các hướng tiếp cận kỹ thuật (Approaches Considered)

| Tiêu chí | Hướng 1: Kế thừa mở rộng và bảo toàn tương thích (Đã chọn) | Hướng 2: Xây dựng interface độc lập hoàn toàn |
| :--- | :--- | :--- |
| **Cấu trúc lớp** | `TreeAdaptiveController(BudgetController)` và `TreeAdaptiveDecision(BudgetDecision)`. | Lớp độc lập hoàn toàn `TreeAdaptiveController` với kiểu trả về riêng biệt. |
| **Ảnh hưởng tới `adapter.py`** | Rất nhỏ. Adapter chỉ cần truyền thêm cấu hình, giữ nguyên lời gọi hàm `select_budget()`. | Lớn. Phải viết lại các nhánh xử lý trong `AdaptiveProposerAdapter` và đổi signature. |
| **Tương thích ngược** | Đạt 100%. Các bài test cũ kiểm tra `isinstance(decision, BudgetDecision)` và `decision.k` đều tiếp tục pass. | Phá vỡ tương thích ngược với các bài test baseline M2/M3. |
| **Độ phức tạp bảo trì** | Thấp, tái sử dụng được giao diện chung của hệ thống. | Cao, tạo ra hai nhánh điều khiển song song không đồng nhất. |

Nhóm kỹ thuật đã thống nhất chọn **Hướng 1**.

---

## 5. Thiết kế kiến trúc chi tiết (Detailed Design)

### 5.1. Cấu trúc dữ liệu `TreeAdaptiveDecision`

```python
@dataclass
class TreeAdaptiveDecision(BudgetDecision):
    """Quyết định chọn topo cây từ TreeAdaptiveController v0.2."""
    topology_name: str = "T0"
    num_nodes: int = 0
    alpha_effective: float = 0.8
    alpha_star: Optional[float] = None
    speedup: float = 1.0
    status: str = "ok"  # "ok" | "memory_guard" | "concurrency_throttled" | "fallback_gate" | "probing" | "zero_crash_fallback"
```

*Ghi chú:* Thuộc tính `k` của lớp cha `BudgetDecision` được gán giá trị bằng `num_nodes` ($N \in \{0, 4, 8, 16, 32\}$) để các thành phần tiêu thụ kiểu cũ (`decision.k`) hoạt động bình thường.

### 5.2. Chuẩn hóa tỷ lệ chấp nhận trong `update()`
Khi bước xác thực hoàn thành, caller gọi `update(num_accepted, num_drafted)`:
- Bộ điều khiển ghi nhớ độ sâu $d$ của topo gần nhất được cấp phát.
- Nếu $d > 0$:
  $$\alpha_{\text{step}} = \min\left(1.0, \frac{\text{num\_accepted}}{\max(d, 1)}\right)$$
- Nếu `num_drafted <= d`: tính theo `min(1.0, num_accepted / num_drafted)` để tương thích với chuỗi tuyến tính cũ.
- Tích lũy vào cửa sổ trượt và cập nhật EMA:
  $$\alpha_{\text{ema}} \leftarrow \beta \cdot \alpha_{\text{ema}} + (1 - \beta) \cdot \alpha_{\text{step}}$$

### 5.3. Máy trạng thái và thứ tự ưu tiên ra quyết định

```text
┌────────────────────────────────────────────────────────────────────────┐
│ select_budget(batch_size, free_vram_mb, kv_cache_usage_pct)            │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: DUAL-TIER MEMORY GUARD & HYSTERESIS COOLDOWN                   │
│ • Kiểm tra VRAM < 1500 MB HOẶC KV > 90% -> Kích hoạt Guard, trả T0     │
│ • Nếu đang Guard: Cooldown >= 3 bước. Chỉ phục hồi khi KV <= 80%      │
│   VÀ VRAM >= 1500 MB. Cấm probing hoàn toàn, reset probe counter.      │
└────────────────────────────────────────────────────────────────────────┘
                                    │ (Bộ nhớ an toàn)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: TÍNH TÍN HIỆU HIỆU DỤNG (UNBIASED EMA ALPHA)                   │
│ • alpha_effective = (alpha_ema - initial_alpha * beta^t) / (1 - beta^t)│
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: CONCURRENCY THROTTLING (KẸP TRẦN TOPO THEO BATCH SIZE)         │
│ • B <= 4   -> Cho phép {T0, T4, T8, T16, T32}                          │
│ • 4 < B<=12-> Kẹp trần T16 ({T0, T4, T8, T16})                         │
│ • 12< B<=24-> Kẹp trần T8  ({T0, T4, T8})                              │
│ • B > 24   -> Kẹp trần T4. Nếu alpha_effective < 0.85 -> Ngắt về T0,  │
│               gán status="concurrency_throttled" và vào nhịp probe 30b │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: DYNAMIC FALLBACK GATE & ADAPTIVE PROBING                       │
│ • Gọi cost_model.select_optimal_topology(B, alpha_effective, candidates│
│ • Nếu max Speedup <= 1.0 (hoặc chọn T0):                               │
│     - Kiểm tra chu kỳ Adaptive Probing T_probe(B) (15 / 20 / 30 bước)  │
│     - Nếu đến hạn thăm dò: Phát T4 để đo lại tín hiệu (status="probing"│
│     - Nếu chưa: Giữ T0 fallback an toàn (status="fallback_gate")       │
│ • Nếu max Speedup > 1.0: Trả về topo tối ưu T* (status="ok")           │
└────────────────────────────────────────────────────────────────────────┘
                                    │ (Bọc toàn bộ)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 5: ZERO-CRASH EXCEPTION HANDLER                                   │
│ • Bắt Exception bất ngờ -> Log warning và trả về T0 an toàn            │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.4. Đấu nối vào `src/proposer/adapter.py`

Cập nhật `AdaptiveProposerAdapter` để:
- Hỗ trợ tham số `controller_cls` nhận `TreeAdaptiveController`.
- Chuyển tiếp các thông tin runtime (`batch_size`, `free_vram_mb`, `kv_cache_usage_pct`) vào phương thức `select_budget()`.
- Trong `propose()`, đối tượng `ProposalOutput` ghi nhận `budget_k = decision.k` (số lượng node nháp của cây) và đóng gói thông tin lý do quyết định.

---

## 6. Kế hoạch kiểm thử và nghiệm thu (Verification & DoD)

### 6.1. Tiêu chí hoàn thành (Definition of Done)
1. **Độ trễ quyết định:** Hoàn tất lời gọi `select_budget()` trong thời gian $< 0.05\text{ ms}$ (được kiểm chứng qua bài test lặp 10.000 lượt).
2. **Kích hoạt phòng vệ bộ nhớ:** Khi $\text{free\_vram\_mb} < 1500\text{ MB}$ hoặc $\text{kv\_cache} > 90.0\%$, bộ điều khiển lập tức trả về $T_0$ và đếm cooldown tối thiểu 3 bước.
3. **Bất biến cấm thăm dò:** Khi Memory Guard được kích hoạt, tuyệt đối không phát sinh bước thăm dò $T_4$ dù đã chạm chu kỳ $T_{\text{probe}}$.
4. **Hồi phục trễ đúng chuẩn:** Chỉ hồi phục khỏi Guard khi đồng thời thỏa mãn $\text{kv\_cache} \le 80.0\%$ VÀ $\text{free\_vram\_mb} \ge 1500\text{ MB}$.
5. **Kẹp tải đồng thời:** Khi $B = 32$, topo được chọn không bao giờ vượt quá $T_4$; khi $\alpha < 0.85$, tự động ngắt về $T_0$.
6. **Dynamic Fallback Gate:** Khi $\alpha$ thấp khiến mọi cây đều có $\text{Speedup} \le 1.0$, hệ thống tự động fallback về $T_0$ và chỉ phát $T_4$ theo đúng nhịp $T_{\text{probe}}(B)$.
7. **Zero-Crash:** Khi truyền các tham số dị thường hoặc giả lập lỗi cost model, bộ điều khiển không bị ném ngoại lệ ra ngoài và trả về $T_0$.
8. **100% Pass test suite:** Vượt qua toàn bộ bài test trong `tests/test_tree_adaptive_controller.py` và bảo toàn kết quả của toàn bộ test suite cũ.

### 6.2. Danh mục ca kiểm thử trong `tests/test_tree_adaptive_controller.py`
1. `test_tree_adaptive_controller_initialization`: Kiểm tra khởi tạo mặc định và các tham số.
2. `test_unbiased_ema_alpha_tracking`: Kiểm tra tính hội tụ và bù lệch khởi động của EMA.
3. `test_depth_normalized_alpha_update`: Kiểm tra chuẩn hóa bước $\alpha$ theo độ sâu $d$ cho cây 2D.
4. `test_dual_tier_memory_guard_vram_trigger`: Kiểm tra ngắt về $T_0$ khi thiếu VRAM.
5. `test_dual_tier_memory_guard_kv_cache_trigger`: Kiểm tra ngắt về $T_0$ khi KV-cache quá tải.
6. `test_memory_guard_cooldown_and_recovery_conditions`: Kiểm tra trễ 3 bước và điều kiện kép hồi phục.
7. `test_no_probing_under_memory_guard`: Kiểm tra tuyệt đối không probe khi đang trong trạng thái Memory Guard.
8. `test_concurrency_throttling_limits`: Kiểm tra kẹp trần ngân sách cây theo $B \in \{1, 4, 8, 16, 32\}$.
9. `test_dynamic_fallback_gate_and_adaptive_probing_cycle`: Kiểm tra ngắt khi $\text{Speedup} \le 1.0$ và nhịp probe $T_{\text{probe}}(B)$.
10. `test_zero_crash_resilience`: Kiểm tra khả năng bắt ngoại lệ và fallback an toàn.
11. `test_decision_latency_benchmark`: Đo benchmark 10.000 lượt gọi bảo đảm $< 0.05\text{ ms}$.
12. `test_proposer_adapter_integration`: Kiểm tra đấu nối thông suốt với `AdaptiveProposerAdapter`.
