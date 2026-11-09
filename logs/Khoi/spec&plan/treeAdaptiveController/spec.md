# Đặc tả kỹ thuật: Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng

> **Mã task Trello:** `#50 [M4-02-05] Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng`  
> **Dự án:** FlexSDS — Hardware-Aware Speculative Decoding for Self-Hosted LLMs on Consumer GPUs  
> **Tác giả:** Cấn Đức Khôi (Lead 70%) & Bùi Đức Đại (Partner 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptiveController/spec.md`  
> **Cơ sở đối chiếu:** Báo cáo thẩm định kỹ thuật 22 trang của Bùi Đức Đại (`tree (2).pdf`, 08/11/2026), chuẩn OPT-Tree (*Wang et al., 2025, TACL*), module `tree_topology.py` (M4-02-01) và `FlexCostModel` (M4-02-03) đã nghiệm thu  
> **Trạng thái:** Bản đặc tả kỹ thuật chính thức (Revision 1.1 — 09/11/2026, cập nhật sau thẩm định kỹ thuật)

---

## 1. Tổng quan và bối cảnh kỹ thuật (Intent & The Why)

Trong chiến dịch đo kiểm hiệu năng tại mốc M3-05, bộ điều khiển thích ứng cơ sở v0.1 (`AdaptiveDraftController`) gặp hiện tượng suy giảm tốc độ (negative speedup) nghiêm trọng trên tập dữ liệu hội thoại tự do (Workload W3) và tại các mức concurrency cao ($c \ge 16$). Hệ số tăng tốc đo thực tế trên GPU NVIDIA L4 rơi xuống dưới 1.0 (dao động từ $0.745\times$ xuống $0.474\times$).

### Các nguyên nhân cốt lõi:
1. **Không gian nháp tuyến tính 1D ($k \in \{0, 2, 3, 4\}$):** Xác suất chấp nhận chuỗi thẳng suy giảm theo cấp số nhân $\alpha^k$. Chỉ cần một token ở đầu chuỗi bị từ chối, toàn bộ chi phí sinh nháp và kiểm chứng phía sau bị lãng phí.
2. **Ngưỡng quyết định tĩnh ghim cứng (`0.65` và `0.80`):** Không phản ánh đúng sự biến thiên phi tuyến của chi phí tính toán trên GPU khi kích thước batch thay đổi.
3. **Thiếu cơ chế kẹp tải đồng thời:** Khi nhiều yêu cầu cùng chạy, việc giữ kích thước nháp lớn gây quá tải tính toán và tranh chấp băng thông bộ nhớ.
4. **Bảo vệ bộ nhớ lỏng lẻo:** Chưa có cơ chế bảo vệ hai tầng kết hợp giữa dung lượng VRAM thực tế và bộ đệm KV-cache.

Task **[M4-02-05]** hiện thực hóa lớp `TreeAdaptiveController` tại đường dẫn chuẩn runtime `src/proposer/tree_adaptive.py` phục vụ việc serving trực tiếp qua `src/proposer/adapter.py`. Bộ điều khiển v0.2 nâng cấp toàn diện lên cấu trúc cây nháp 2D ($T_0, T_4, T_8, T_{16}, T_{32}$), tích hợp giải nghiệm hòa vốn động theo mô hình chi phí phần cứng, cơ chế phòng vệ hai tầng và bộ lọc telemetry sentinel.

---

## 2. Mục tiêu, phi mục tiêu và phản mục tiêu (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục tiêu (Goals)
- **Hiện thực hóa `TreeAdaptiveController` tại `src/proposer/tree_adaptive.py`:** Kế thừa trực tiếp từ `BudgetController` để bảo toàn tính tương thích của hệ thống.
- **Mở rộng cấu trúc quyết định `TreeAdaptiveDecision`:** Kế thừa từ `BudgetDecision`, duy trì trường `k` (gán bằng số node $N$) đồng thời bổ sung `topology_name`, `num_nodes`, `alpha_effective`, `alpha_star`, `speedup`.
- **Chuẩn hóa tỷ lệ chấp nhận theo độ sâu cây (Empirical Yield Index):** Sử dụng độ sâu $d$ của topo vừa phát hành để chuẩn hóa bước $\alpha_{\text{step}} = \min(1.0, \text{num\_accepted} / \max(d, 1))$. Đây là chỉ số thu hoạch thực nghiệm đưa tín hiệu phản hồi về dải $[0, 1]$ cho bộ điều khiển mà không gây sập tín hiệu khi $N > d$.
- **Cơ chế phòng vệ bộ nhớ hai tầng (Dual-Tier Memory Guard) kèm bộ lọc sentinel VRAM:**
  - Nhận diện giá trị telemetry VRAM bị thiếu, âm, hoặc mang giá trị sentinel ($\ge 10^8\text{ MB}$). Khi mất tín hiệu VRAM, kích hoạt chế độ bảo vệ tăng cường bằng cách siết chặt ngưỡng kích hoạt KV-cache từ 90% xuống 75%, và ngưỡng phục hồi từ 80% xuống 65%.
  - Ngắt về $T_0$ ($N=0$) ngay lập tức khi: $\text{free\_vram\_mb} < 1500\text{ MB}$ HOẶC $\text{kv\_cache\_usage\_pct} > \text{ngưỡng an toàn}$.
  - Duy trì máy trạng thái ghi nhận nguồn gốc kích hoạt `_guard_trigger_source` (`"vram"`, `"kv"`, hoặc `"both"`).
  - Độ trễ hồi phục (hysteresis cooldown) tối thiểu 3 bước liên tiếp ở $T_0$.
  - Nếu kích hoạt có nguyên nhân do cạn kiệt VRAM vật lý, bắt buộc phải có telemetry VRAM hợp lệ $\ge 1500\text{ MB}$ mới cho phép phục hồi; nếu mất tín hiệu khi đang đói bộ nhớ, tiếp tục giữ guard để chống crash GPU.
  - Cấm hoàn toàn việc phát bước thăm dò (probing) khi Memory Guard đang hoạt động, đồng thời reset bộ đếm thăm dò về 0.
- **Tách biệt bộ đếm quan sát và tự phục hồi cho $\alpha$:**
  - Tách biến đếm số quan sát thực nghiệm `_observation_count` (chỉ tăng khi `num_drafted > 0`) khỏi tổng số chu kỳ điều phối `_step_count`.
  - Công thức khử lệch khởi động: $\beta_{\text{power}} = \beta^{N_{\text{obs}}}$, triệt tiêu hiện tượng trôi trọng số khi hệ thống chạy ở chế độ $T_0$.
  - Tự phục hồi trạng thái: tự động chuyển `alpha_ema` về `initial_alpha` nếu bị nhiễm giá trị `NaN` hoặc `Inf`.
- **Kẹp trần kích thước cây theo luồng đồng thời (Concurrency Throttling):**
  - $B \le 4$: Cho phép cấu hình tối đa $T_{32}$ ($N \le 32$).
  - $4 < B \le 12$: Kẹp trần tối đa $T_{16}$ ($N \le 16$).
  - $12 < B \le 24$: Kẹp trần tối đa $T_8$ ($N \le 8$).
  - $B > 24$: Kẹp trần tối đa $T_4$ ($N \le 4$), đồng thời ngắt về $T_0$ nếu $\alpha_{\text{effective}} < 0.85$.
- **Cơ chế chuyển làn an toàn và thăm dò thích ứng (Dynamic Fallback Gate & Adaptive Probing):**
  - Sử dụng hàm `FlexCostModel.select_optimal_topology()` để chọn topo có hệ số tăng tốc dự báo lớn nhất $S^*$. Nếu $S^* \le 1.0$, ngắt an toàn về $T_0$ với `status="fallback_gate"`.
  - Khi đang ở trạng thái ngắt an toàn $T_0$, định kỳ phát bước thăm dò với $T_4$ sau mỗi $T_{\text{probe}}(B)$ bước giải mã:
    $$T_{\text{probe}}(B) = \begin{cases} 15 \text{ bước}, & B \le 4 \\ 20 \text{ bước}, & 4 < B \le 16 \\ 30 \text{ bước}, & B > 16 \end{cases}$$
- **Nguyên tắc bọc thép không sập hệ thống (Zero-Crash):** Bọc toàn bộ logic ra quyết định trong khối bắt ngoại lệ, tự động fallback an toàn về $T_0$ (`status="zero_crash_fallback"`) nếu có lỗi tính toán nội bộ phát sinh.
- **Đóng gói `topology_name` trong `ProposalOutput` tại `src/proposer/adapter.py`:**
  - Bổ sung trường `topology_name: Optional[str] = None` trong `ProposalOutput` để truyền cấu trúc cây nháp ra bên ngoài cho verifier và scheduler.
- **Hiệu năng ra quyết định thời gian thực:** Thời gian ra quyết định trung bình (mean latency) của `select_budget()` phải đạt $< 0.05\text{ ms}$ ($50\ \mu\text{s}$) trên CPU.

### 2.2. Phi mục tiêu (Non-Goals)
- Không can thiệp sửa mã nguồn của scheduler và planner thuộc phạm vi M4-02-04 của Đại (`src/server/budget_planner.py` và `src/server/adaptive_scheduler.py`). Việc gỡ bỏ trần cứng `max_k = 4` sẽ do Đại chủ động cập nhật trong task M4-02-04 dựa trên `ProposalOutput.topology_name`.
- Không tự thu thập hay chạy micro-benchmark đo ma trận chi phí mới trên GPU (thuộc phạm vi task M4-02-02).
- Không sửa đổi mô hình giải tích kỳ vọng nhánh trong `tree_topology.py` (vấn đề xung khắc độc lập nhánh C03 được chuyển thành đề tài nghiên cứu cho Sprint 6).

### 2.3. Phản mục tiêu (Anti-Goals)
- Giảm thiểu tối đa rủi ro tràn bộ nhớ VRAM (OOM) trong quá trình serving bằng cơ chế bảo vệ hai tầng.
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
- **Mô hình chi phí:** Tích hợp với `FlexCostModel` từ `src.flexsds.core.cost_model`.
- **Bộ khử lệch khởi động (Unbiased EMA):** Sử dụng hệ số $\beta = 0.8$ với công thức hiệu chỉnh dựa trên số lần quan sát thực tế $N_{\text{obs}}$:
  $$\alpha_{\text{unbiased}} = \frac{\alpha_{\text{ema}} - \alpha_{\text{init}} \cdot \beta^{N_{\text{obs}}}}{1 - \beta^{N_{\text{obs}}}}$$

---

## 4. Các hướng tiếp cận kỹ thuật (Approaches Considered)

| Tiêu chí | Hướng 1: Kế thừa mở rộng và bảo toàn tương thích (Đã chọn) | Hướng 2: Xây dựng interface độc lập hoàn toàn |
| :--- | :--- | :--- |
| **Cấu trúc lớp** | `TreeAdaptiveController(BudgetController)` và `TreeAdaptiveDecision(BudgetDecision)`. | Lớp độc lập hoàn toàn `TreeAdaptiveController` với kiểu trả về riêng biệt. |
| **Ảnh hưởng tới `adapter.py`** | Rất nhỏ. Adapter chỉ cần truyền thêm cấu hình, mở rộng `ProposalOutput.topology_name`. | Lớn. Phải viết lại các nhánh xử lý trong `AdaptiveProposerAdapter` và đổi signature. |
| **Tương thích ngược** | Đạt 100%. Các bài test cũ kiểm tra `isinstance(decision, BudgetDecision)` và `decision.k` đều tiếp tục pass. | Phá vỡ tương thích ngược với các bài test baseline M2/M3. |
| **Độ phức tạp bảo trì** | Thấp, tái sử dụng được giao diện chung của hệ thống. | Cao, tạo ra hai nhánh điều khiển song song không đồng nhất. |

---

## 5. Thiết kế kiến trúc chi tiết (Detailed Design)

### 5.1. Cấu trúc dữ liệu `TreeAdaptiveDecision` và `ProposalOutput`

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

@dataclass
class ProposalOutput:
    """Đóng gói đầu ra từ adapter gửi tới scheduler và verifier."""
    draft_tokens: List[int]
    budget_k: int
    skipped_draft: bool
    reason: str
    topology_name: Optional[str] = None
```

### 5.2. Chuẩn hóa tỷ lệ chấp nhận trong `update()`
Khi bước xác thực hoàn thành, caller gọi `update(num_accepted, num_drafted)`:
- Nếu `num_drafted <= 0` hoặc `num_accepted < 0`: bỏ qua cập nhật EMA, giữ nguyên `_observation_count`.
- Nếu `num_drafted > 0`: tăng `_observation_count += 1`.
- Chuẩn hóa:
  * Nếu $d > 0$ và $N > d$: $\alpha_{\text{step}} = \min(1.0, \text{num\_accepted} / d)$.
  * Ngược lại: $\alpha_{\text{step}} = \min(1.0, \text{num\_accepted} / \text{num\_drafted})$.
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
│ TẦNG 1: DUAL-TIER MEMORY GUARD & SENTINEL VRAM FILTER                  │
│ • Kiểm tra VRAM sentinel (>= 1e8) -> Siết ngưỡng KV ở 75%/65%          │
│ • Kiểm tra VRAM < 1500 MB HOẶC KV vượt ngưỡng -> Kích hoạt Guard       │
│ • Lưu _guard_trigger_source ('vram' | 'kv' | 'both')                   │
│ • Cooldown >= 3 bước. Phục hồi an toàn theo nguồn kích hoạt            │
│ • Cấm probing hoàn toàn khi đang Guard                                 │
└────────────────────────────────────────────────────────────────────────┘
                                     │ (Bộ nhớ an toàn)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: TÍNH TÍN HIỆU HIỆU DỤNG & SELF-HEALING                         │
│ • Tự phục hồi: Nếu alpha_ema là NaN/Inf -> Đưa về initial_alpha        │
│ • Debiasing: alpha_eff = (alpha_ema - alpha_0 * beta^N_obs)/(1 - beta) │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: CONCURRENCY THROTTLING (KẸP TRẦN TOPO THEO BATCH SIZE)         │
│ • B <= 4   -> Cho phép {T0, T4, T8, T16, T32}                          │
│ • 4 < B<=12-> Kẹp trần T16 ({T0, T4, T8, T16})                         │
│ • 12< B<=24-> Kẹp trần T8  ({T0, T4, T8})                              │
│ • B > 24   -> Kẹp trần T4. Nếu alpha_eff < 0.85 -> Ngắt về T0          │
│               (status='concurrency_throttled') và nhịp probe 30 bước   │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: DYNAMIC FALLBACK GATE & ADAPTIVE PROBING                       │
│ • Gọi cost_model.select_optimal_topology(B, alpha_eff, candidates)     │
│ • Nếu max Speedup <= 1.0 (hoặc chọn T0):                               │
│     - Kiểm tra chu kỳ Adaptive Probing T_probe(B) (15 / 20 / 30 bước)  │
│     - Nếu đến hạn thăm dò: Phát T4 (status='probing')                  │
│     - Nếu chưa: Giữ T0 fallback an toàn (status='fallback_gate')       │
│ • Nếu max Speedup > 1.0: Trả về topo tối ưu T* (status='ok')           │
└────────────────────────────────────────────────────────────────────────┘
                                     │ (Bọc toàn bộ)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 5: ZERO-CRASH EXCEPTION HANDLER                                   │
│ • Bắt Exception bất ngờ -> Log warning và trả về T0 an toàn            │
│   (status='zero_crash_fallback')                                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Kế hoạch kiểm thử và nghiệm thu (Verification & DoD)

### 6.1. Tiêu chí hoàn thành (Definition of Done)
1. **Độ trễ quyết định:** Hoàn tất lời gọi `select_budget()` với thời gian trung bình (mean latency) $< 0.05\text{ ms}$ (được kiểm chứng qua bài test lặp 10.000 lượt, đạt $0.0164\text{ ms}$).
2. **Kích hoạt phòng vệ bộ nhớ và sentinel:** Phát hiện sentinel $10^9\text{ MB}$, tự động siết ngưỡng KV ở $75\%$; ngắt về $T_0$ và đếm cooldown tối thiểu 3 bước.
3. **Phục hồi an toàn theo nguồn kích hoạt:** Nếu cạn VRAM vật lý, bắt buộc phải có VRAM hợp lệ $\ge 1500\text{ MB}$ mới phục hồi, không tự ý phục hồi khi mất cảm biến.
4. **Bất biến cấm thăm dò:** Khi Memory Guard được kích hoạt, tuyệt đối không phát sinh bước thăm dò $T_4$.
5. **Kẹp tải đồng thời:** Khi $B = 32$, topo được chọn không bao giờ vượt quá $T_4$; khi $\alpha < 0.85$, tự động ngắt về $T_0$.
6. **Dynamic Fallback Gate:** Khi $\alpha$ thấp khiến mọi cây đều có $\text{Speedup} \le 1.0$, hệ thống tự động fallback về $T_0$ (`fallback_gate`) và chỉ phát $T_4$ theo đúng nhịp $T_{\text{probe}}(B)$.
7. **Zero-Crash Resilience:** Khi mô hình chi phí ném ngoại lệ bất ngờ, controller trả về $T_0$ an toàn với `status="zero_crash_fallback"`.
8. **Đóng gói `topology_name`:** `ProposalOutput` mang đúng tên cấu trúc cây (`"T0"`, `"T4"`, v.v.) và `None` đối với linear controller.
9. **Tách biệt quan sát và self-healing:** $T_0$ không làm trôi debiasing; giá trị `NaN` tự động được phục hồi về `initial_alpha`.
10. **100% Pass test suite:** 19/19 tests trong `tests/test_tree_adaptive_controller.py` và 75/75 tests toàn hệ thống FlexSDS đều PASS.
