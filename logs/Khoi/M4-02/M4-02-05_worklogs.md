# [M4-02-05] Worklog Khôi — Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng

> **Task Trello:** `#50 [M4-02-05] Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng`  
> **Người thực hiện:** Cấn Đức Khôi (Lead 70%) và Bùi Đức Đại (Partner 30%)  
> **Nhánh phát triển:** `feat/m4-02-05-tree-adaptive-controller`  
> **Trạng thái:** ✅ **HOÀN THÀNH VÀ ĐÃ ĐIỀU CHỈNH TOÀN DIỆN THEO THẨM ĐỊNH KỸ THUẬT (100% PASS 19/19 unit tests & 75/75 FlexSDS core suite pass)**

---

## 1. Mục tiêu và bối cảnh kỹ thuật

Khắc phục triệt để hiện tượng suy giảm hiệu năng (negative speedup) của bộ điều khiển tuyến tính cơ sở v0.1 (`AdaptiveDraftController`) trên tập dữ liệu tự do (W3) và ở các mức concurrency cao ($c \ge 16$):
1. **Nâng cấp không gian nháp lên cây 2D:**
   - Thay thế chuỗi nháp 1D ($k \in \{0, 2, 3, 4\}$) bằng danh mục 5 cấu hình cây rời rạc chuẩn: $T_0$ ($N=0$), $T_4$ ($N=4$), $T_8$ ($N=8$), $T_{16}$ ($N=16$), $T_{32}$ ($N=32$) từ module `TreeTopology` (M4-02-01).
2. **Khắc phục hiện tượng sập tín hiệu $\alpha$ trên cây 2D:**
   - Trong giải mã tham lam, số token được chấp nhận tối đa dọc theo một nhánh cây luôn bị chặn bởi độ sâu $d$ ($\text{num\_accepted} \le d \le 5$). Nếu chia cho tổng số node $N$ ($16$ hoặc $32$), tỷ lệ $\alpha$ sẽ bị kéo tụt xuống dưới $0.25$, gây sập Fallback Gate vĩnh viễn.
   - Chuẩn hóa bước theo độ sâu đường dẫn: $\alpha_{\text{step}} = \min(1.0, \text{num\_accepted} / \max(d, 1))$. Đây là empirical yield index giúp đưa tín hiệu về $[0, 1]$ cho bộ điều khiển phản hồi.
3. **Bộ phòng vệ hai tầng (Dual-Tier Memory Guard) kèm lọc sentinel VRAM:**
   - Nhận diện sentinel `1e9` từ scheduler để tự động siết ngưỡng an toàn KV-cache ở mức 75% / 65%.
   - Tự động ngắt về $T_0$ ($N=0$) ngay khi: $\text{free\_vram\_mb} < 1500\text{ MB}$ HOẶC $\text{kv\_cache\_usage\_pct} > \text{ngưỡng an toàn}$.
   - Duy trì trễ hồi phục (hysteresis cooldown) tối thiểu 3 bước liên tiếp ở $T_0$.
   - Ghi nhận nguồn gốc kích hoạt `_guard_trigger_source`: nếu đói VRAM vật lý, bắt buộc phải có telemetry VRAM hợp lệ $\ge 1500\text{ MB}$ mới phục hồi, không tự ý hồi phục khi mất tín hiệu cảm biến.
   - Cấm phát bước thăm dò (probing) dưới bất kỳ trạng thái Memory Guard nào và reset bộ đếm probe về 0 để triệt tiêu nguy cơ OOM.
4. **Kẹp trần kích thước cây theo luồng đồng thời (Concurrency Throttling):**
   - $B \le 4$: Cho phép tối đa $T_{32}$ ($N \le 32$).
   - $4 < B \le 12$: Kẹp trần tối đa $T_{16}$ ($N \le 16$).
   - $12 < B \le 24$: Kẹp trần tối đa $T_8$ ($N \le 8$).
   - $B > 24$: Kẹp trần tối đa $T_4$ ($N \le 4$); tự động ngắt về $T_0$ (`status="concurrency_throttled"`) nếu $\alpha_{\text{effective}} < 0.85$.
5. **Dynamic Fallback Gate & Adaptive Probing:**
   - Tích hợp hàm `FlexCostModel.select_optimal_topology()`. Khi $\max S \le 1.0$, ngắt an toàn về $T_0$ (`status="fallback_gate"`).
   - Phát probe $T_4$ định kỳ theo chu kỳ thích ứng tải: $T_{\text{probe}}(B) \in \{15, 20, 30\}$ bước.
6. **Bảo đảm nguyên tắc Zero-Crash:**
   - Bắt mọi ngoại lệ tính toán trong vòng lặp đề xuất nháp và fallback an toàn về $T_0$ (`status="zero_crash_fallback"`) mà không gián đoạn luồng serving.
7. **Đạt chuẩn hiệu năng thời gian thực:**
   - Thời gian ra quyết định trung bình (mean latency) $< 0.05\text{ ms}$ (thực tế đo được **$0.0164\text{ ms}$**, nhanh gấp hơn 3 lần yêu cầu).

---

## 2. Kết quả hiện thực hóa và kiểm chứng

### A. Cấu trúc dữ liệu `TreeAdaptiveDecision` và `ProposalOutput`

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

### B. Bảng phân tích trạng thái ra quyết định (Decision State Matrix)

| Trạng thái (`status`) | Điều kiện kích hoạt | Topo trả về | $k$ | Hành vi thăm dò (Probing) |
| :--- | :--- | :---: | :---: | :--- |
| **`memory_guard`** | VRAM $< 1500\text{ MB}$ hoặc KV quá ngưỡng (hoặc cooldown $< 3$) | $T_0$ | 0 | Cấm hoàn toàn; reset bộ đếm probe về 0 |
| **`concurrency_throttled`** | $B > 24$ và $\alpha_{\text{effective}} < 0.85$ | $T_0$ | 0 | Phát probe $T_4$ sau mỗi 30 bước nếu bộ nhớ an toàn |
| **`fallback_gate`** | $\max_{\mathcal{T}} S(\mathcal{T}, B, \alpha) \le 1.0$ | $T_0$ | 0 | Phát probe $T_4$ sau mỗi $T_{\text{probe}}(B)$ bước |
| **`probing`** | Chạm mốc chu kỳ $T_{\text{probe}}(B)$ bước ở trạng thái ngắt an toàn | $T_4$ | 4 | Thăm dò một bước để đo lại tín hiệu $\alpha$ |
| **`ok`** | Tồn tại topo có $\text{Speedup} > 1.0$ và bộ nhớ an toàn | $T^* \in \{T_4, T_8, T_{16}, T_{32}\}$ | $N^*$ | Trạng thái suy đoán tối ưu |
| **`zero_crash_fallback`** | Bắt ngoại lệ bất ngờ trong quá trình tính toán | $T_0$ | 0 | Fallback an toàn, log cảnh báo |

### C. Đo lường hiệu năng ra quyết định (Decision Latency Benchmark)

- Phương pháp đo: 10.000 lượt gọi liên tiếp hàm `select_budget(batch_size=1, free_vram_mb=8000.0, kv_cache_usage_pct=40.0)` sau khi warm-up 100 lượt.
- Kết quả:
  - Thời gian đo trung bình: **$0.0164\text{ ms}$ ($16.43\ \mu\text{s}$)**.
  - Ngưỡng yêu cầu tối đa theo đặc tả: **$< 0.050\text{ ms}$ ($50\ \mu\text{s}$)**.
  - Kết luận: **ĐẠT CHUẨN HIỆU NĂNG VƯỢT MỨC (nhanh gấp hơn 3 lần trần SLA)**.

---

## 3. Các artifact đã tạo và chỉnh sửa

1. **Mã nguồn chức năng cốt lõi:**
   - `src/proposer/tree_adaptive.py`: Hiện thực hóa `TreeAdaptiveDecision` và `TreeAdaptiveController` với đầy đủ 5 tầng logic, bộ lọc sentinel VRAM, máy trạng thái phục hồi và tự phục hồi `NaN`/`Inf`.
   - `src/proposer/adapter.py`: Import và hỗ trợ `TreeAdaptiveController`, truyền suốt `topology_name` trong `ProposalOutput` và chuyển tiếp telemetry.
   - `src/proposer/__init__.py`: Export `TreeAdaptiveController` và `TreeAdaptiveDecision`.
2. **Bộ kiểm thử tự động:**
   - `tests/test_tree_adaptive_controller.py`: 19 ca kiểm thử bao phủ toàn diện khởi tạo, EMA, chuẩn hóa độ sâu, Memory Guard, Cooldown, Sentinel VRAM, Concurrency Throttling, Dynamic Fallback Gate, Zero-Crash và Benchmark latency.
3. **Tài liệu đặc tả và kế hoạch:**
   - `logs/Khoi/spec&plan/treeAdaptiveController/spec.md`: Bản đặc tả kỹ thuật chi tiết Revision 1.1.
   - `logs/Khoi/spec&plan/treeAdaptiveController/plan.md`: Bản kế hoạch thực thi chi tiết Revision 1.1.
   - `logs/Khoi/spec&plan/treeAdaptiveController/review_alignment_design.md`: Bản thiết kế kỹ thuật sau thẩm định.
   - `logs/Khoi/spec&plan/treeAdaptiveController/review_alignment_plan.md`: Bản kế hoạch thực thi sau thẩm định.
4. **Nhật ký kỹ thuật:**
   - `logs/Khoi/M4-02/M4-02-05_worklogs.md`: Báo cáo nghiệm thu kỹ thuật task M4-02-05.

---

## 4. Biên bản kiểm thử (Test Execution Summary)

```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22
configfile: pyproject.toml
plugins: typeguard-4.4.4
collected 75 items

tests/test_budget_controller.py ...................                      [ 25%]
tests/test_budget_planner.py ...................                         [ 50%]
tests/test_proposer_adapter.py ........                                  [ 61%]
tests/test_tree_topology.py ...........                                  [ 76%]
tests/test_tree_adaptive_controller.py ...................               [100%]

============================== 75 passed in 0.29s ==============================
```

---

## 5. Kết quả điều chỉnh kỹ thuật sau thẩm định độc lập từ Bùi Đức Đại

Qua báo cáo thẩm định 22 trang của Bùi Đức Đại (`tree (2).pdf`), các điều chỉnh sau đã được hoàn thành 100%:
1. **Khắc phục lỗ hổng VRAM telemetry sentinel (C07):** Bổ sung hàm kiểm tra `_is_valid_vram()`, nhận diện sentinel `1e9`, tự động siết ngưỡng an toàn KV ở 75% / 65%; bảo đảm trạng thái đói VRAM vật lý không bị phục hồi sai khi mất cảm biến (`_guard_trigger_source`).
2. **Tách biệt quan sát EMA và tự phục hồi (C05):** Thêm biến `_observation_count` để tính debiasing chính xác, không bị trôi khi chạy $T_0$; tự phục hồi `alpha_ema` về `initial_alpha` khi phát hiện `NaN`/`Inf`.
3. **Đóng gói `topology_name` trong `ProposalOutput` (C01, C08):** Giúp scheduler và verifier của Đại nhận biết trực tiếp cấu trúc cây nháp để gỡ bỏ trần cứng `max_k = 4` và thiết lập tree mask.
4. **Hài hòa kẹp tải và probing chu kỳ (C06):** Tuyệt đối không probe khi đang trong Memory Guard; chỉ probe khi bộ nhớ an toàn.
5. **Chuẩn hóa văn phong và số liệu (C02, C09, P01, P03, P04):** Định nghĩa rõ ràng SLA thời gian ra quyết định trung bình $< 0.05\text{ ms}$; phân định giữa predicted speedup và measured speedup; làm rõ $\alpha_{\text{step}}$ là empirical yield index.
