# Kế hoạch triển khai: Kết nối ma trận chi phí phần cứng thực nghiệm vào bộ chọn ngân sách Hardware-Adaptive

> **Dành cho kỹ sư AI:** KỸ NĂNG BẮT BUỘC: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi kế hoạch này từng tác vụ một. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu (Goal):** Đưa ma trận chi phí phần cứng đo đạc (`EmpiricalCostProfile` / `cost_matrix_ms`) vào đường quyết định thực tế của `AdaptiveDraftController.select_budget()`, xử lý an toàn hiện tượng không đơn điệu trên profile A10G (`C(1, B) < C(0, B)` tại `B ∈ {1, 2, 4}`) bằng chuẩn hóa tham chiếu $C_{\text{ref}}(B)$, và xuất đầy đủ telemetry truy vết quyết định chi phí trong `AdaptiveBudgetPlanner`.

**Kiến trúc (Architecture):**
1. Bổ sung hàm kiểm tra đơn điệu thuần túy `inspect_cost_matrix_monotonicity` trong `src/flexsds/core/cost_model.py` để chẩn đoán các ô `C(k, B) < C(0, B)` hoặc `C(k, B) < C(k_prev, B)` mà không sửa dữ liệu đo gốc.
2. Mở rộng `BudgetDecision` và thêm luật chọn `K` dựa trên chi phí `apply_cost_driven_budget_rule(alpha, batch_size)` trong `AdaptiveDraftController` (`src/proposer/controller.py`), áp dụng khi `self.cost_profile.matrix` không rỗng, giữ nguyên ưu tiên cao nhất cho memory pressure guard và tương thích ngược cho unit test legacy không nạp profile.
3. Nối metadata hồ sơ phần cứng và các trường quyết định chi phí (`decision_policy`, `theoretical_speedup`, `dynamic_alpha_star`, `cost_k_ms`, `cost_baseline_ms`, `cost_ref_ms`, `non_monotonic_cost`) vào `AdaptiveBudgetPlanner.from_env()` và `_decision_record()` (`src/server/budget_planner.py`).

**Công nghệ sử dụng (Tech Stack):** Python 3.11+, Pytest, JSON/JSONL telemetry.

**Đặc tả kỹ thuật (Spec):** `do-an-ptit-d22/logs/bugs/hardwareProfileCostDecisionBug/2026-11-22-hardware-adaptive-cost-driven-budget-spec.md`

## Các ràng buộc toàn cục (Global Constraints)

- **NFR-01 (Trung thực dữ liệu):** Tuyệt đối không sửa tay file `src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`; `validate_measured_profile()` phải tiếp tục pass 100% với `raw_sha256` gốc.
- **NFR-02 (Độ trễ quyết định):** `AdaptiveDraftController.select_budget()` chạy hoàn toàn trên CPU, không gọi CUDA, không cấp phát tensor, độ trễ `< 0.05 ms` mỗi bước.
- **NFR-03 (Tương thích ngược):** Toàn bộ unit test hiện có khởi tạo `AdaptiveDraftController()` không truyền `cost_profile` tiếp tục pass 100% với luật ngưỡng cũ.
- **NFR-04 (Nhánh Git):** Không chỉnh sửa nhánh audit đóng băng `audit/flexsds-v023-hardwareadaptive-20261120`. Tạo nhánh phát triển mới tách từ `develop` (ví dụ `fix/r6-hardware-cost-driven-budget`) trước khi commit.
- **Chính tả tiếng Việt:** Không viết hoa kiểu Title Case tiếng Anh trong tài liệu và comment tiếng Việt; chỉ viết hoa chữ cái đầu câu/tiêu đề, danh từ riêng (FlexSDS, NVIDIA A10, A10G, EAGLE3, vLLM, Modal, Hardware-Adaptive) và từ viết tắt chuẩn (GPU, VRAM, KV, OOM, JSON, CLI, EMA, R1–R8, FR, NFR, AC).
- **Ngoài phạm vi plan này:** Khuyến nghị FR-05 (thêm GPU warmup/clock metadata cho lần đo calibration mới) và gộp bản vá đóng gói Modal/Windows + chế độ Static EAGLE matched-config sẽ thực hiện ở bước tiếp theo sau khi đóng trọn bộ kiểm thử R6 trên CPU.

## Tiêu điểm kiểm tra (Review Focus)

1. **Ma trận không đơn điệu ở batch nhỏ (`C(1, B) < C(0, B)` và `C(2, B) < C(0, B)` trên A10G thật tại `B ∈ {1, 2, 4}`):** Tại `alpha = 0.0`, controller tuyệt đối không được chọn `K = 1` hoặc `K = 2`; phải trả `K = 0` và gắn `non_monotonic_cost = True` (kiểm tra trong Tác vụ 2).
2. **Chi phí hỏng (`<= 0`, `NaN`, `inf`) hoặc thiếu `k=0` trong ma trận:** `apply_cost_driven_budget_rule` phải ném `ValueError` tường minh thay vì chia cho 0 hoặc ghi `NaN` làm sập `json.dumps(..., allow_nan=False)` ở `_write_step` (kiểm tra trong Tác vụ 2).
3. **`probe_k` không nằm trong `candidate_budgets` (ví dụ `candidate_budgets = [0, 1]` khi `probe_k = 2`):** Periodic probing phải tự kẹp về ứng viên dương hợp lệ trong `candidate_budgets` thay vì trả `k = 2` ngoài lưới đo (kiểm tra trong Tác vụ 2).
4. **Hòa hiệu suất giữa hai ứng viên (`|S(k1) - S(k2)| < 1e-6`):** Phải ưu tiên chọn `k` nhỏ hơn để tránh tốn tính toán draft dư thừa (kiểm tra trong Tác vụ 2).
5. **Batch trung gian không nằm đúng trên lưới đo (`B = 3` hoặc `B = 6` với lưới `[1, 2, 4, 8, 16, 32]`):** Tra cứu nhất quán qua `EmpiricalCostProfile.get_cost(k, batch_size)` (batch gần nhất, hòa chọn batch nhỏ hơn) mà không phát sinh `KeyError` (kiểm tra trong Tác vụ 2).

---

### Tác vụ 1: Bổ sung hàm chẩn đoán đơn điệu `inspect_cost_matrix_monotonicity` trong `cost_model.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/core/cost_model.py:379-409`
- Create: `do-an-ptit-d22/tests/test_hardware_cost_driven_controller.py`

**Interfaces:**
- Consumes: `cost_matrix_ms: Mapping[str, float]` theo định dạng khóa `k{k}_b{b}` (ví dụ từ `validate_measured_profile()`).
- Produces: `inspect_cost_matrix_monotonicity(cost_matrix_ms: Mapping[str, float]) -> List[Dict[str, Any]]` trả về danh sách các bất thường đã sắp xếp theo `(batch, k)`, mỗi phần tử gồm:
  `{"batch": int, "k": int, "cost_ms": float, "baseline_k0_ms": Optional[float], "prev_k": Optional[int], "prev_cost_ms": Optional[float], "below_k0": bool, "below_prev_k": bool}`.

- [ ] **Bước 1: Viết bài test thất bại cho `inspect_cost_matrix_monotonicity`**

Tạo `do-an-ptit-d22/tests/test_hardware_cost_driven_controller.py` với 2 test:
```python
import json
from pathlib import Path
import pytest

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "flexsds" / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def test_inspect_monotonicity_detects_a10g_small_batch_inversions():
    from flexsds.core.cost_model import inspect_cost_matrix_monotonicity

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    anomalies = inspect_cost_matrix_monotonicity(data["cost_matrix_ms"])
    below_k0_pairs = {(item["batch"], item["k"]) for item in anomalies if item["below_k0"]}
    assert (1, 1) in below_k0_pairs
    assert (2, 1) in below_k0_pairs
    assert (2, 2) in below_k0_pairs
    assert (4, 1) in below_k0_pairs
    assert (4, 2) in below_k0_pairs
    assert all(item["batch"] not in {8, 16, 32} for item in anomalies if item["below_k0"])


def test_inspect_monotonicity_returns_empty_for_strictly_increasing_matrix():
    from flexsds.core.cost_model import inspect_cost_matrix_monotonicity

    clean = {"k0_b1": 15.0, "k1_b1": 16.0, "k2_b1": 18.0, "k0_b8": 16.0, "k1_b8": 20.0, "k2_b8": 25.0}
    assert inspect_cost_matrix_monotonicity(clean) == []
```

- [ ] **Bước 2: Chạy test để xác nhận thất bại**

Chạy từ `do-an-ptit-d22/`: `PYTHONPATH=src:. pytest tests/test_hardware_cost_driven_controller.py -v`
Kết quả mong đợi: `FAIL` với `ImportError: cannot import name 'inspect_cost_matrix_monotonicity'`.

- [ ] **Bước 3: Triển khai `inspect_cost_matrix_monotonicity` trong `src/flexsds/core/cost_model.py`**

Đặt hàm thuần túy ngay trước `validate_measured_profile` trong `src/flexsds/core/cost_model.py`:
- Signature: `def inspect_cost_matrix_monotonicity(cost_matrix_ms: Dict[str, float]) -> List[Dict[str, Any]]:`
- Parse các khóa dạng `k{k}_b{b}` (bỏ qua khóa topology `topo_*`), gom theo `batch`, duyệt các `k` theo thứ tự tăng dần.
- Nếu `k > 0` và `cost < c0` (`below_k0 = True`) hoặc `cost < prev_cost` (`below_prev_k = True`), thêm dict mô tả vào danh sách kết quả. Không thay đổi dữ liệu đầu vào.

- [ ] **Bước 4: Chạy test xác nhận pass**

Chạy: `PYTHONPATH=src:. pytest tests/test_hardware_cost_driven_controller.py -v`
Kết quả mong đợi: `2 passed`.

- [ ] **Bước 5: Commit Tác vụ 1**

```bash
git add src/flexsds/core/cost_model.py tests/test_hardware_cost_driven_controller.py
git commit -m "feat(cost_model): add non-monotonicity diagnostic helper for measured profiles"
```

---

### Tác vụ 2: Triển khai quyết định chọn `K` theo ma trận chi phí trong `AdaptiveDraftController`

**Files:**
- Modify: `do-an-ptit-d22/src/proposer/controller.py:9-16,216-265,415-472`
- Modify: `do-an-ptit-d22/tests/test_hardware_cost_driven_controller.py`

**Interfaces:**
- Consumes: `self.cost_profile` (`EmpiricalCostProfile` hoặc `CostProfile` có `matrix` không rỗng), `self.candidate_budgets: List[int]`, `batch_size: int`, `effective_alpha: float`.
- Produces:
  - `BudgetDecision` mở rộng thêm các trường mặc định:
    - `speedup: Optional[float] = None`
    - `alpha_star: Optional[float] = None`
    - `status: str = "ok"`
    - `decision_policy: str = "threshold"`
    - `cost_k_ms: Optional[float] = None`
    - `cost_baseline_ms: Optional[float] = None`
    - `cost_ref_ms: Optional[float] = None`
    - `non_monotonic_cost: bool = False`
  - `AdaptiveDraftController.__init__(..., min_speedup_ratio: float = 1.05)` (validate `math.isfinite(min_speedup_ratio) and min_speedup_ratio >= 1.0`, sai thì raise `ValueError`).
  - `AdaptiveDraftController.solve_break_even_alpha(self, k: int, cost_ratio: float, min_speedup_ratio: Optional[float] = None) -> Optional[float]`
  - `AdaptiveDraftController.apply_cost_driven_budget_rule(self, alpha: float, batch_size: int) -> BudgetDecision`
  - `AdaptiveDraftController.select_budget(...) -> BudgetDecision` chọn theo 3 tầng: Memory Guard (`decision_policy="memory_guard"`, `status="memory_guard"`) → Cost-Driven khi `bool(getattr(self.cost_profile, "matrix", None))` (`decision_policy="cost_model"`) → Legacy threshold (`decision_policy="threshold"`).

- [ ] **Bước 1: Viết đầy đủ các bài test thất bại cho AC-01 → AC-04 và 5 trường hợp Review Focus**

Bổ sung vào `do-an-ptit-d22/tests/test_hardware_cost_driven_controller.py`:
```python
def _load_a10g_cost_profile():
    from src.proposer.controller import EmpiricalCostProfile

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    matrix = {
        (int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): float(v)
        for k, v in data["cost_matrix_ms"].items()
    }
    return EmpiricalCostProfile(matrix=matrix), data["grid"]["candidate_budgets"]


def test_cost_driven_adapts_k_across_batch_sizes_on_real_a10g_profile():
    """AC-01: Cùng alpha=0.70, K giảm dần theo batch_size trên profile A10G thật."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    expected = {1: 4, 8: 3, 16: 2, 32: 0}
    for batch_size, expected_k in expected.items():
        ctrl = AdaptiveDraftController(
            initial_alpha=0.70,
            candidate_budgets=budgets,
            cost_profile=profile,
            min_speedup_ratio=1.05,
        )
        decision = ctrl.select_budget(batch_size=batch_size, free_vram_mb=16000.0, kv_cache_usage_pct=10.0)
        assert decision.k == expected_k
        assert decision.decision_policy == "cost_model"
        assert decision.cost_k_ms is not None and decision.cost_k_ms > 0
        assert decision.cost_baseline_ms == pytest.approx(profile.get_cost(0, batch_size))
        assert decision.speedup is not None and decision.speedup >= 1.0


def test_cost_driven_changes_k_for_different_hardware_profiles():
    """AC-02: Cùng alpha=0.70 và B=8, hai GPU profile khác nhau cho ra K khác nhau."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    fast_verify = EmpiricalCostProfile(matrix={(0, 8): 18.0, (1, 8): 18.5, (2, 8): 20.0, (3, 8): 22.0, (4, 8): 25.0})
    slow_verify = EmpiricalCostProfile(matrix={(0, 8): 18.0, (1, 8): 42.0, (2, 8): 56.0, (3, 8): 68.0, (4, 8): 75.0})
    c_fast = AdaptiveDraftController(initial_alpha=0.70, candidate_budgets=[0, 1, 2, 3, 4], cost_profile=fast_verify)
    c_slow = AdaptiveDraftController(initial_alpha=0.70, candidate_budgets=[0, 1, 2, 3, 4], cost_profile=slow_verify)
    assert c_fast.select_budget(batch_size=8).k >= 3
    assert c_slow.select_budget(batch_size=8).k == 0


def test_cost_driven_non_monotonic_a10g_falls_back_at_zero_alpha():
    """AC-03 & Review Focus #1: Không bao giờ chọn K>0 ở alpha=0 dù C(1,B) < C(0,B)."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    for batch_size in (1, 2, 4, 8, 16, 32):
        ctrl = AdaptiveDraftController(initial_alpha=0.0, candidate_budgets=budgets, cost_profile=profile)
        decision = ctrl.select_budget(batch_size=batch_size)
        assert decision.k == 0
        assert decision.status == "fallback_gate"
        assert decision.non_monotonic_cost is (batch_size in {1, 2, 4})


def test_cost_driven_preserves_memory_guard_priority_and_probing():
    """AC-04: Memory guard ưu tiên cao hơn cost model; cost fallback kích hoạt periodic probing."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    ctrl = AdaptiveDraftController(
        initial_alpha=0.95,
        candidate_budgets=budgets,
        cost_profile=profile,
        probe_interval=3,
        probe_k=2,
        memory_cooldown_steps=3,
    )
    mem_dec = ctrl.select_budget(batch_size=1, free_vram_mb=500.0, kv_cache_usage_pct=95.0)
    assert mem_dec.k == 0
    assert mem_dec.status == "memory_guard"
    assert mem_dec.decision_policy == "memory_guard"

    ctrl_probe = AdaptiveDraftController(
        initial_alpha=0.50,
        candidate_budgets=budgets,
        cost_profile=profile,
        probe_interval=3,
        probe_k=2,
    )
    assert ctrl_probe.select_budget(batch_size=32).k == 0
    assert ctrl_probe.select_budget(batch_size=32).k == 0
    probe_dec = ctrl_probe.select_budget(batch_size=32)
    assert probe_dec.k == 2
    assert probe_dec.status == "probing"
    assert probe_dec.decision_policy == "cost_model"


def test_cost_driven_probing_clamps_to_candidate_budgets():
    """Review Focus #3: probe_k=2 bị kẹp về k=1 nếu candidate_budgets chỉ có [0, 1]."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    profile = EmpiricalCostProfile(matrix={(0, 1): 15.0, (1, 1): 40.0})
    ctrl = AdaptiveDraftController(
        initial_alpha=0.2,
        candidate_budgets=[0, 1],
        cost_profile=profile,
        probe_interval=2,
        probe_k=2,
    )
    assert ctrl.select_budget(batch_size=1).k == 0
    probe_dec = ctrl.select_budget(batch_size=1)
    assert probe_dec.k == 1
    assert probe_dec.status == "probing"


def test_cost_driven_tie_breaks_toward_smaller_k():
    """Review Focus #4: Khi hai ứng viên có speedup chênh < 1e-6, chọn k nhỏ hơn."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    # alpha = 0.5 -> E[tau(1)] = 1.5, E[tau(2)] = 1.75; tỷ lệ C(2)/C(1) = 1.75/1.5 tạo hòa tuyệt đối
    profile = EmpiricalCostProfile(matrix={(0, 1): 15.0, (1, 1): 15.0, (2, 1): 17.5})
    ctrl = AdaptiveDraftController(initial_alpha=0.5, candidate_budgets=[0, 1, 2], cost_profile=profile)
    assert ctrl.select_budget(batch_size=1).k == 1


def test_cost_driven_rejects_non_positive_or_non_finite_costs():
    """Review Focus #2: Chi phí <= 0 hoặc NaN/inf phải ném ValueError."""
    from src.proposer.controller import AdaptiveDraftController, EmpiricalCostProfile

    bad = EmpiricalCostProfile(matrix={(0, 1): 15.0, (1, 1): float("nan")})
    ctrl = AdaptiveDraftController(initial_alpha=0.8, candidate_budgets=[0, 1], cost_profile=bad)
    with pytest.raises(ValueError, match="finite"):
        ctrl.select_budget(batch_size=1)


def test_cost_driven_handles_intermediate_batch_size_via_closest_grid_point():
    """Review Focus #5: Batch trung gian B=3 tra lưới A10G gần nhất (B=2) nhất quán."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_cost_profile()
    ctrl = AdaptiveDraftController(initial_alpha=0.70, candidate_budgets=budgets, cost_profile=profile)
    d3 = ctrl.select_budget(batch_size=3)
    d2 = AdaptiveDraftController(initial_alpha=0.70, candidate_budgets=budgets, cost_profile=profile).select_budget(batch_size=2)
    assert d3.k == d2.k
    assert d3.cost_k_ms == pytest.approx(d2.cost_k_ms)
```

- [ ] **Bước 2: Chạy test để xác nhận thất bại**

Chạy: `PYTHONPATH=src:. pytest tests/test_hardware_cost_driven_controller.py -v`
Kết quả mong đợi: FAIL trên các test mới (do `AdaptiveDraftController` chưa hỗ trợ `min_speedup_ratio` và chưa gọi `cost_profile`).

- [ ] **Bước 3: Cập nhật `BudgetDecision` và `AdaptiveDraftController` trong `src/proposer/controller.py`**

Thực hiện đúng các thay đổi sau trong `src/proposer/controller.py`:
1. Mở rộng dataclass `BudgetDecision` với 8 trường tùy chọn có giá trị mặc định như trong khối **Interfaces**.
2. Trong `AdaptiveDraftController.__init__`, thêm `min_speedup_ratio: float = 1.05` (kiểm tra hữu hạn và `>= 1.0`, lưu vào `self.min_speedup_ratio`).
3. Thêm phương thức `solve_break_even_alpha(self, k: int, cost_ratio: float, min_speedup_ratio: Optional[float] = None) -> Optional[float]`:
   - Tìm $\alpha \in [0, 1]$ thỏa $E[\tau(k, \alpha)] = \text{threshold} \cdot \text{cost\_ratio}$ (với `threshold = min_speedup_ratio or self.min_speedup_ratio`).
   - Nếu `k <= 0`: trả `None`. Nếu `target <= 1.0`: trả `0.0`. Nếu `self.estimate_expected_tokens(k, 1.0) < target`: trả `None` (infeasible). Ngược lại giải chia đôi 40 vòng trên `[0.0, 1.0]`.
4. Thêm phương thức `apply_cost_driven_budget_rule(self, alpha: float, batch_size: int) -> BudgetDecision`:
   - Thu thập tập ứng viên `all_k = sorted(set(self.candidate_budgets) | {0})`, tra cứu `costs[k] = self.cost_profile.get_cost(k, batch_size)`. Kiểm tra tất cả `math.isfinite(c) and c > 0` (nếu sai, raise `ValueError("Cost profile entries must be finite and positive")`).
   - Tính `c_baseline = costs[0]`, `c_ref = min(costs.values())`, `non_monotonic = c_ref < c_baseline`.
   - Duyệt `cand_k` dương theo thứ tự tăng dần trong `sorted(k for k in self.candidate_budgets if k > 0)`:
     - `e_tau = self.estimate_expected_tokens(cand_k, alpha)`
     - `speedup = e_tau * (c_ref / costs[cand_k])`
     - Cập nhật `best_k` nếu `speedup >= self.min_speedup_ratio` và `speedup > best_speedup + 1e-6` (đảm bảo hòa trong biên `1e-6` ưu tiên `k` nhỏ hơn).
   - Nếu tìm được `best_k > 0`: reset `self.consecutive_fallback_steps = 0`, tính `alpha_star = self.solve_break_even_alpha(best_k, costs[best_k] / c_ref)`, trả `BudgetDecision(..., status="ok", decision_policy="cost_model", ...)`.
   - Nếu không tìm được `best_k > 0`: tăng `self.consecutive_fallback_steps += 1`; nếu kích hoạt probe (`probe_interval > 0` và `consecutive_fallback_steps >= probe_interval`), reset counter, kẹp `probe_k` vào tập dương của `candidate_budgets` (nếu `probe_k` thuộc tập dương thì dùng `probe_k`; nếu không, chọn phần tử lớn nhất `<= probe_k`, fallback về phần tử dương nhỏ nhất; nếu không có phần tử dương nào thì giữ `k = 0`), trả `status="probing"`. Ngược lại trả `k = 0`, `speedup = 1.0`, `status="fallback_gate"`. Tính `alpha_star` theo ứng viên dương có `alpha_star` nhỏ nhất còn khả thi (đại diện ngưỡng thoát fallback ở `batch_size` hiện tại).
5. Trong `AdaptiveDraftController.select_budget(...)`:
   - Tầng 1: `apply_memory_pressure_rule` → trả `BudgetDecision(..., status="memory_guard", decision_policy="memory_guard")`.
   - Tầng 2: Nếu `bool(getattr(self.cost_profile, "matrix", None))` → gọi `apply_cost_driven_budget_rule(alpha=effective_alpha, batch_size=batch_size)`.
   - Tầng 3: Legacy `apply_budget_rule(alpha=effective_alpha)` → giữ nguyên `decision_policy="threshold"`.
   - Đẩy thêm `decision_policy`, `speedup`, `cost_k_ms` vào `self._history` (giữ nguyên các khóa cũ `"k"`, `"alpha"`, `"reason"`).

- [ ] **Bước 4: Chạy test mới và test controller cũ để xác nhận pass và không hồi quy**

Chạy: `PYTHONPATH=src:. pytest tests/test_hardware_cost_driven_controller.py tests/test_budget_controller.py tests/test_memory_pressure_rule.py -v`
Kết quả mong đợi: 100% PASS.

- [ ] **Bước 5: Commit Tác vụ 2**

```bash
git add src/proposer/controller.py tests/test_hardware_cost_driven_controller.py
git commit -m "feat(controller): drive AdaptiveDraftController K selection from EmpiricalCostProfile"
```

---

### Tác vụ 3: Nối metadata hồ sơ, cảnh báo không đơn điệu và telemetry chi phí trong `AdaptiveBudgetPlanner`

**Files:**
- Modify: `do-an-ptit-d22/src/server/budget_planner.py:45-71,114-126,225-303`
- Modify: `do-an-ptit-d22/tests/test_hardware_cost_driven_controller.py`

**Interfaces:**
- Consumes:
  - `inspect_cost_matrix_monotonicity` từ Tác vụ 1 (`src/flexsds/core/cost_model.py`).
  - Các trường mở rộng của `BudgetDecision` từ Tác vụ 2 (`speedup`, `alpha_star`, `status`, `decision_policy`, `cost_k_ms`, `cost_baseline_ms`, `cost_ref_ms`, `non_monotonic_cost`).
- Produces:
  - `AdaptiveBudgetPlanner.__init__(..., hardware_profile: Optional[dict] = None)`
  - Bản ghi JSONL `planned_decision` (qua `_decision_record`) điền đủ:
    `hardware_profile`, `dynamic_alpha_star`, `theoretical_speedup`, `decision_policy`, `cost_k_ms`, `cost_baseline_ms`, `cost_ref_ms`, `non_monotonic_cost`.
  - Cảnh báo `logger.warning(...)` trong `AdaptiveBudgetPlanner.from_env()` khi profile nạp vào có ô không đơn điệu.

- [ ] **Bước 1: Viết bài test thất bại cho AC-05 và cảnh báo không đơn điệu**

Bổ sung vào `do-an-ptit-d22/tests/test_hardware_cost_driven_controller.py`:
```python
def test_planner_from_env_records_cost_driven_telemetry_and_warns_non_monotonic(tmp_path, caplog):
    """AC-05: Planner nạp FLEXSDS_PROFILE_PATH, log cảnh báo không đơn điệu và ghi telemetry R6/R8."""
    import logging
    from src.server.budget_planner import AdaptiveBudgetPlanner

    step_path = tmp_path / "serving_steps.jsonl"
    with caplog.at_level(logging.WARNING, logger="src.server.budget_planner"):
        planner = AdaptiveBudgetPlanner.from_env({
            "FLEXSDS_PROFILE_PATH": str(A10G_PROFILE_PATH),
            "ADAPTIVE_STEP_TELEMETRY_PATH": str(step_path),
            "ADAPTIVE_CONTROLLER_KWARGS": json.dumps({"initial_alpha": 0.70}),
        })

    assert "non-monotonic" in caplog.text.lower()

    plan_b1 = planner.plan_step(["req-a"], batch_size=1, kv_usage=0.1, free_vram_mb=16000.0)
    plan_b32 = planner.plan_step(["req-b"], batch_size=32, kv_usage=0.1, free_vram_mb=16000.0)
    assert plan_b1.batch_k == 4
    assert plan_b32.batch_k == 0

    lines = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    rec_b1, rec_b32 = lines[0], lines[1]

    assert rec_b1["decision_policy"] == "cost_model"
    assert rec_b1["hardware_profile"]["device"] == "NVIDIA A10"
    assert rec_b1["hardware_profile"]["sm_architecture"] == "sm_86"
    assert rec_b1["hardware_profile"]["sha256"]
    assert rec_b1["theoretical_speedup"] > 1.05
    assert rec_b1["dynamic_alpha_star"] is not None
    assert rec_b1["cost_k_ms"] > 0
    assert rec_b1["cost_baseline_ms"] > 0
    assert rec_b1["cost_ref_ms"] > 0
    assert rec_b1["non_monotonic_cost"] is True

    assert rec_b32["decision_policy"] == "cost_model"
    assert rec_b32["scalar_runtime_budget"] == 0
    assert rec_b32["non_monotonic_cost"] is False
```

- [ ] **Bước 2: Chạy test để xác nhận thất bại**

Chạy: `PYTHONPATH=src:. pytest tests/test_hardware_cost_driven_controller.py::test_planner_from_env_records_cost_driven_telemetry_and_warns_non_monotonic -v`
Kết quả mong đợi: FAIL (chưa có cảnh báo non-monotonic, `plan_b1.batch_k == 2` thay vì `4`, `hardware_profile is None`).

- [ ] **Bước 3: Cập nhật `AdaptiveBudgetPlanner` và `_decision_record` trong `src/server/budget_planner.py`**

1. Trong `AdaptiveBudgetPlanner.__init__`: thêm tham số `hardware_profile: Optional[dict] = None` và lưu `self.hardware_profile = deepcopy(hardware_profile) if hardware_profile else None`.
2. Trong `AdaptiveBudgetPlanner.from_env`:
   - Khi có `FLEXSDS_PROFILE_PATH`, gọi `inspect_cost_matrix_monotonicity(data["cost_matrix_ms"])`.
   - Nếu có bất thường `below_k0`, gọi `logger.warning("Measured hardware profile %s has %d non-monotonic cost cells (C(k, B) < C(0, B)); lower-bound reference C_ref(B) normalization is active.", profile_path, len(below_k0))`.
   - Tạo `hardware_profile = {"name": Path(profile_path).name, "sha256": digest, "device": data["metadata"].get("device"), "sm_architecture": data["metadata"].get("sm_architecture"), "non_monotonic_cell_count": len(anomalies)}` và truyền vào `cls(..., hardware_profile=hardware_profile)`.
3. Trong `plan_step` và `_decision_record`:
   - Truyền `hardware_profile=self.hardware_profile` vào `_decision_record(...)`.
   - Trong `_decision_record(...)`, điền `"hardware_profile": deepcopy(hardware_profile)` (hoặc `None` nếu không nạp profile) và bổ sung các khóa `"decision_policy"`, `"cost_k_ms"`, `"cost_baseline_ms"`, `"cost_ref_ms"`, `"non_monotonic_cost"` (dùng helper `finite(...)` cho mọi số thực để không bao giờ gây lỗi `allow_nan=False`).

- [ ] **Bước 4: Chạy test mới và test planner hiện hữu**

Chạy: `PYTHONPATH=src:. pytest tests/test_hardware_cost_driven_controller.py tests/test_budget_planner.py -v`
Kết quả mong đợi: 100% PASS.

- [ ] **Bước 5: Commit Tác vụ 3**

```bash
git add src/server/budget_planner.py tests/test_hardware_cost_driven_controller.py
git commit -m "feat(planner): emit hardware cost decision telemetry and warn on non-monotonic profiles"
```

---

### Tác vụ 4: Kiểm thử hồi quy toàn diện, xác minh toàn vẹn profile A10G và chốt tiến độ

**Files:**
- Verify (read-only): `do-an-ptit-d22/src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`
- Create: `do-an-ptit-d22/logs/bugs/hardwareProfileCostDecisionBug/progress.md`

- [ ] **Bước 1: Xác minh file profile A10G còn nguyên vẹn và pass `validate_measured_profile()`**

Chạy từ `do-an-ptit-d22/`:
```bash
git diff -- src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json
PYTHONPATH=src:. python3 -c "import json; from pathlib import Path; from flexsds.core.cost_model import validate_measured_profile; validate_measured_profile(json.loads(Path('src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json').read_text())); print('OK')"
```
Kết quả mong đợi: `git diff` rỗng và in ra `OK`.

- [ ] **Bước 2: Chạy toàn bộ bộ test hồi quy liên quan (AC-06)**

Chạy từ `do-an-ptit-d22/`:
```bash
PYTHONPATH=src:. pytest \
  tests/test_hardware_cost_driven_controller.py \
  tests/test_budget_controller.py \
  tests/test_budget_planner.py \
  tests/test_memory_pressure_rule.py \
  tests/test_proposer_adapter.py \
  tests/test_vram_memory_lifecycle.py \
  tests/test_serving_profile_guard.py \
  tests/test_tree_adaptive_controller.py -v
```
Kết quả mong đợi: 100% PASS (0 failed).

- [ ] **Bước 3: Tạo `logs/bugs/hardwareProfileCostDecisionBug/progress.md` ghi nhận bằng chứng hoàn thành**

Ghi lại:
- Danh sách commit SHA của các Tác vụ 1–3 trên nhánh mới tách từ `develop`.
- Kết quả pytest đầy đủ (số test pass/skip/fail).
- Danh sách việc tiếp theo để đóng kiểm chứng GPU cho báo cáo (1. cherry-pick 3 file vá đóng gói Modal/Windows từ nhánh audit; 2. bổ sung chế độ Static EAGLE matched-config cùng eager/V1 runner; 3. chạy 1 đợt GPU verification + lossless sạch + benchmark batch sweep).

- [ ] **Bước 4: Commit tài liệu tiến độ**

```bash
git add logs/bugs/hardwareProfileCostDecisionBug/progress.md
git commit -m "docs(progress): record completion of hardware cost-driven budget controller"
```
