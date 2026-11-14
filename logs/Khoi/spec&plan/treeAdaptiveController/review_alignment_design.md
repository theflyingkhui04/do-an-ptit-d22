# Thiết kế điều chỉnh kỹ thuật và hoàn thiện TreeAdaptiveController v0.2 sau thẩm định

- Ngày lập: 09/11/2026 (Cập nhật sau Builder Check)
- Tác giả: Cấn Đức Khôi (Lead M4-02-05)
- Tài liệu đối chiếu: Báo cáo thẩm định kỹ thuật của Bùi Đức Đại (`logs/Khoi/spec&plan/treeAdaptiveController/tree (2).pdf`, 22 trang, ngày 08/11/2026)
- Nhánh công việc: `feat/m4-02-05-tree-adaptive-controller`
- Pull Request: [PR #34](https://github.com/theflyingkhui04/do-an-ptit-d22/pull/34)

---

## 1. Mục đích và bối cảnh (Intent & why)

Bộ điều khiển `TreeAdaptiveController` v0.2 đã được hiện thực hóa tại nhánh `feat/m4-02-05-tree-adaptive-controller` và vượt qua 14/14 unit test cùng 59/59 regression test. Tuy nhiên, qua đợt thẩm định độc lập 22 trang từ Bùi Đức Đại, một số điểm hở giao diện và ranh giới an toàn hệ thống đã được chỉ ra:
1. Thiếu thông tin cấu trúc cây (`topology_name`) trong `ProposalOutput` dẫn đến việc tầng verifier/scheduler chỉ nhận được số nguyên `budget_k` vô hướng mà không biết được hình dạng cây nháp.
2. Lỗ hổng sentinel `NO_VRAM_READING_MB = 1e9` từ phía scheduler có thể vô hiệu hóa tầng phòng vệ bộ nhớ VRAM nếu tín hiệu bị mất hoặc không đo được.
3. Cần làm sạch công thức hiệu chỉnh EMA tại $t=0$, loại trừ rủi ro `NaN`/`Inf`, và làm rõ xung đột giữa cơ chế kẹp tải nặng ($B > 24$) và chu kỳ probing thích ứng.
4. Cần chuẩn hóa lại các phát biểu học thuật và phân định ranh giới phối hợp với task M4-02-04 của Bùi Đức Đại (`AdaptiveBudgetPlanner` và `AdaptiveAsyncScheduler`).

Tài liệu này xác định thiết kế chi tiết để khắc phục triệt để các phản hồi trên, hoàn thiện mã nguồn và cập nhật PR #34.

---

## 2. Mục tiêu, phi mục tiêu và chống mục tiêu (Goals, non-goals, anti-goals)

### Mục tiêu (Goals)
- Đảm bảo an toàn bộ nhớ khi mất tín hiệu VRAM telemetry: phát hiện sentinel và kích hoạt chế độ phòng vệ tăng cường qua KV-cache.
- Bổ sung trường `topology_name: Optional[str] = None` vào `ProposalOutput` trong `src/proposer/adapter.py`, truyền suốt từ quyết định `TreeAdaptiveDecision` ra ngoài adapter (`"T0"`, `"T4"`, v.v. cho cây nháp; `None` cho linear draft).
- Tách bạch `_step_count` (tổng bước lập lịch) và `_observation_count` (số lần thực nhận verification telemetry), triệt tiêu sai số debiasing EMA khi chạy $T_0$.
- Tự phục hồi dữ liệu (self-healing) khi phát hiện `NaN` hoặc `Inf` trong trạng thái nội bộ.
- Hài hòa cơ chế kẹp tải với probing chu kỳ khi $B > 24$.
- Bổ sung bộ kiểm thử biên đầy đủ trong `tests/test_tree_adaptive_controller.py` đạt 100% tỷ lệ vượt qua.
- Hiệu chỉnh tài liệu đặc tả và nhật ký công việc theo đúng quy chuẩn học thuật và quy tắc chính tả tiếng Việt.

### Phi mục tiêu (Non-goals)
- Không sửa đổi mã nguồn của scheduler hay planner thuộc phạm vi M4-02-04 của Đại (ví dụ `src/server/budget_planner.py` hoặc `src/server/adaptive_scheduler.py`). Các điều chỉnh gỡ bỏ trần cứng `max_k = 4` sẽ do Đại chủ động thực hiện trong M4-02-04 dựa trên giao diện thống nhất.
- Không viết lại mô hình giải tích kỳ vọng nhánh trong `tree_topology.py` (vấn đề xung khắc độc lập nhánh C03 được chuyển thành đề tài nghiên cứu cho Sprint 6).

### Chống mục tiêu (Anti-goals)
- Tuyệt đối không làm tăng độ trễ ra quyết định trung bình của controller vượt quá SLA $0.05\text{ ms}$ ($50\ \mu\text{s}$).
- Không phá vỡ tương thích ngược với bộ điều khiển tuyến tính `AdaptiveDraftController` v0.1 và hợp đồng Interface A hiện có.

---

## 3. Ràng buộc kỹ thuật (Constraints)
- Môi trường thực thi: Linux, Python 3.10+, PyTorch/vLLM stack.
- Thời gian ra quyết định trung bình (mean latency) trên CPU: $\le 0.05\text{ ms}$.
- Toàn bộ suite test FlexSDS (59 tests hiện tại và các test mới) phải vượt qua tuyệt đối (PASS).
- Tuân thủ quy tắc chính tả tiếng Việt: không viết hoa kiểu Title Case của tiếng Anh.

---

## 4. Các phương án cân nhắc (Approaches)

### Vấn đề 1: Xử lý VRAM telemetry bị thiếu hoặc sentinel (C07)
- **Phương án 1 (Strict fallback):** Khi phát hiện `free_vram_mb` rơi vào sentinel ($\ge 10^8$), giá trị âm hoặc `None/NaN`, controller lập tức ngắt toàn bộ về $T_0$ ở mọi bước.
  * *Ưu điểm:* Cực kỳ an toàn.
  * *Nhược điểm:* Quá bi quan; nếu cụm máy chủ vLLM không cài đặt `nvml` hoặc driver không đọc được VRAM thời gian thực, hệ thống speculative decoding sẽ bị tê liệt hoàn toàn mặc dù KV-cache vẫn còn nhiều chỗ trống.
- **Phương án 2 (Dual-guard adaptive fail-safe - Phương án được chọn):** Khi `free_vram_mb` không hợp lệ hoặc rơi vào sentinel ($\ge 10^8$), controller ghi nhận trạng thái mất tín hiệu VRAM, ghi log warning và chuyển sang chế độ phòng vệ tăng cường: tạm thời bỏ qua điều kiện VRAM nhưng siết chặt ngưỡng kích hoạt KV-cache guard từ 90% xuống 75%, và ngưỡng phục hồi từ 80% xuống 65%.
  * *Ưu điểm:* Duy trì khả năng suy luận thích ứng khi hệ thống ổn định, nhưng lập tức ngắt về $T_0$ sớm hơn nhiều nếu KV-cache bắt đầu đầy, bảo vệ an toàn mà không làm tê liệt toàn bộ hệ thống.
  * *Lý do chọn:* Cân bằng hoàn hảo giữa hiệu năng và an toàn, giải quyết trọn vẹn mối lo của Đại mà không gây gián đoạn suy luận không cần thiết.

### Vấn đề 2: Truyền cấu trúc cây qua `ProposalOutput` (C01)
- **Phương án 1 (Đổi kiểu trả về thành một đối tượng mới hoàn toàn):** Tạo `TreeProposalOutput` riêng biệt.
  * *Nhược điểm:* Phá vỡ mã nguồn của các module gọi cũ phụ thuộc `ProposalOutput`.
- **Phương án 2 (Mở rộng trường tùy chọn tương thích ngược - Phương án được chọn):** Bổ sung trường `topology_name: Optional[str] = None` vào `ProposalOutput` hiện tại.
  * *Ưu điểm:* Giữ tương thích 100% với `AdaptiveDraftController` (giá trị mặc định là `None`), các caller cũ không bị ảnh hưởng, trong khi caller mới nhận biết được chính xác cấu trúc cây $T_4, T_8, T_{16}, T_{32}$. Không gây lỗi `KeyError` tra cứu topology.

---

## 5. Chi tiết kỹ thuật đã quyết định (Detailed technical design)

### 5.1. Khắc phục lỗ hổng VRAM telemetry và máy trạng thái phục hồi trong `tree_adaptive.py`
- Định nghĩa ngưỡng sentinel: `SENTINEL_VRAM_THRESHOLD_MB = 1.0e8`.
- Bổ sung hàm kiểm tra `_is_valid_vram(free_vram_mb: Optional[float]) -> bool`:
  * Trả về `False` nếu `free_vram_mb is None` hoặc `math.isnan(free_vram_mb)` hoặc `free_vram_mb < 0` hoặc `free_vram_mb >= SENTINEL_VRAM_THRESHOLD_MB`.
- Bổ sung hàm kiểm tra `_is_valid_kv(kv_pct: Optional[float]) -> bool`:
  * Trả về `False` nếu `kv_pct is None` hoặc `math.isnan(kv_pct)` hoặc `kv_pct < 0.0` hoặc `kv_pct > 100.0`. Khi KV không hợp lệ, lập tức kích hoạt Memory Guard phòng vệ.
- Lưu trữ nguồn gốc kích hoạt guard: `_guard_trigger_source: str = "none"` (`"vram"`, `"kv"`, hoặc `"both"`).
- Điều kiện phục hồi an toàn:
  * Nếu guard kích hoạt thuần túy do `kv` (hoặc VRAM vốn đã mất tín hiệu từ đầu): phục hồi khi $KV \le 65.0\%$.
  * Nếu guard kích hoạt có nguyên nhân do cạn kiệt VRAM vật lý (`vram` hoặc `both`): chỉ cho phép phục hồi khi VRAM có telemetry hợp lệ và $\ge 1500\text{ MB}$ đồng thời $KV \le 65.0\%$. Nếu telemetry VRAM bị mất khi đang đói VRAM, tiếp tục giữ guard (giữ an toàn chống crash OOM).

### 5.2. Tách biệt bộ đếm và tự phục hồi cho $\alpha$ (C05)
- Thêm biến đếm số quan sát: `self._observation_count: int = 0`.
- Trong `get_effective_alpha()`:
  * Tự phục hồi: nếu `math.isnan(self.alpha_ema)` hoặc `math.isinf(self.alpha_ema)`, gán lại `self.alpha_ema = self.initial_alpha`.
  * Nếu `self._observation_count == 0`: trả về ngay `self.initial_alpha`.
  * Tính debiasing theo số lượng quan sát thực tế:
    $$\beta_{\text{pow}} = \beta^{N_{\text{obs}}},\quad \text{den} = 1.0 - \beta_{\text{pow}}$$
    Nếu $\text{den} \le 10^{-6}$, trả về `self.alpha_ema`. Ngược lại trả về $(\text{alpha\_ema} - \alpha_0 \cdot \beta_{\text{pow}}) / \text{den}$ kẹp trong $[0.0, 1.0]$.
- Trong `update(num_accepted, num_drafted)`:
  * `self._step_count += 1` luôn tăng để ghi nhận tổng số chu kỳ điều phối phục vụ Interface A.
  * Nếu `num_drafted <= 0` hoặc `num_accepted < 0`: bỏ qua cập nhật EMA, không tăng `_observation_count`.
  * Nếu `num_drafted > 0`: tính $\alpha_{\text{step}}$, cập nhật `self.alpha_ema`, tăng `self._observation_count += 1`.

### 5.3. Hài hòa cơ chế kẹp tải với probing chu kỳ (C06)
- Khi $B > 24$ và $\alpha_{\text{eff}} < 0.85$:
  * Nếu Memory Guard đang hoạt động: tuyệt đối không probe, luôn trả về $T_0$.
  * Nếu bộ nhớ an toàn: tăng bộ đếm fallback `_fallback_steps`. Khi đạt chu kỳ `probe_interval = 30`, thực hiện probe cấu trúc $T_4$ tối thiểu để kiểm tra xem phân phối văn bản có phục hồi không.

### 5.4. Đóng gói `topology_name` trong `ProposalOutput` và `adapter.py` (C01)
- Trong `src/proposer/adapter.py`:
  ```python
  @dataclass
  class ProposalOutput:
      draft_tokens: List[int]
      budget_k: int
      skipped_draft: bool
      reason: str
      topology_name: Optional[str] = None
  ```
- Trong `AdaptiveProposerAdapter.propose()`:
  * Trích xuất `topology_name`:
    `topo = getattr(decision, "topology_name", None)`
    Nếu controller trả về `TreeAdaptiveDecision`, `topo` mang giá trị `"T0"`, `"T4"`, v.v. Nếu là linear controller cũ, `topo` mang giá trị `None`.
  * Truyền `topology_name=topo` vào đối tượng `ProposalOutput`.

### 5.5. Phân định rõ ràng trong bộ kiểm thử biên `test_tree_adaptive_controller.py` (P02)
- Test 15: Kiểm tra missing/sentinel VRAM (`free_vram_mb = 1e9` hoặc `None`), xác nhận hệ thống siết ngưỡng KV-cache ở 75% thay vì crash hoặc bỏ qua.
- Test 16: Kiểm tra đầu vào bất thường của $\alpha$ (`num_drafted <= 0`, `num_accepted < 0`, `math.nan`), xác nhận cơ chế self-healing và tách biệt observation count.
- Test 17: Kiểm tra phân định rõ giữa kinh tế không khả thi và ngoại lệ:
  * Tuyến A: Candidate topologies không khả thi (speedup $\le 1.0$) $\implies$ trả về `fallback_gate` và probing theo chu kỳ.
  * Tuyến B: Exception ném ra từ cost model / solver $\implies$ rơi vào ngoại lệ tầng 5 bọc thép `zero_crash_fallback`.
- Test 18: Kiểm tra `ProposalOutput` mang đúng `topology_name` từ adapter (mang tên cấu trúc với `TreeAdaptiveController` và `None` với linear controller).

### 5.6. Hiệu chỉnh tài liệu đặc tả và nhật ký công việc (C02, C08, C09, P01, P03, P04)
- Cập nhật `spec.md` và `plan.md` trong thư mục `logs/Khoi/spec&plan/treeAdaptiveController/`:
  * Sửa định nghĩa SLA: ghi rõ "thời gian ra quyết định trung bình (mean latency) $< 0.05\text{ ms}$".
  * Hạ cấp câu chữ: thay "tuyệt đối không OOM" thành "giảm thiểu rủi ro OOM qua cơ chế phòng vệ hai tầng".
  * Làm rõ $\alpha_{\text{step}} = L/d$ là empirical yield index cho bộ điều khiển phản hồi.
  * Phân định ranh giới phối hợp với M4-02-04 của Đại về việc xử lý `max_k = 4`.

---

## 6. Những gì để lại cho người lập trình thực thi (What's left to the builder)
- Thứ tự triển khai các hàm kiểm tra nhỏ bên trong `tree_adaptive.py`.
- Tên cụ thể của các hàm test bổ sung trong `test_tree_adaptive_controller.py`.
- Chi tiết câu chữ cập nhật trong `spec.md`, `plan.md` và `M4-02-05_worklogs.md`.
