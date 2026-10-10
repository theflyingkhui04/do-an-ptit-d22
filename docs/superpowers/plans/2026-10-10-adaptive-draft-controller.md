# AdaptiveDraftController Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hiện thực hóa lớp `AdaptiveDraftController` kế thừa từ `BudgetController`, chuyên biệt hóa cửa sổ trượt 10 bước, cơ chế Warm-up Unbiased phản ứng tức thời với prompt khó, phân rã modular rules co giãn ngân sách nháp k in {0, 2, 3, 4} theo các ngưỡng thực nghiệm và xuất telemetry Giao diện A.

**Architecture:** Lớp `AdaptiveDraftController` được đặt trong [`src/proposer/controller.py`](../../../src/proposer/controller.py), mở rộng từ `BudgetController`. Controller duy trì hàng đợi cố định 10 mẫu chấp nhận thô (`_window = deque(maxlen=10)`), tính toán `alpha_unbiased` để triệt tiêu thiên lệch khởi tạo `initial_alpha` trong các bước giải mã đầu tiên, cung cấp phương thức tự chẩn đoán `get_window_stats()`, và tách bạch hai hàm quy tắc: `apply_budget_rule(alpha)` (Khôi phụ trách) và `apply_memory_pressure_rule(free_vram, kv_cache)` (Dương phụ trách).

**Tech Stack:** Python 3.10+, dataclasses, collections.deque, pytest.

**Spec:** [`docs/05-thiet-ke-he-thong.md`](../../../docs/05-thiet-ke-he-thong.md) (§2, §3.1), [`logs/Khoi/M2-02/M2-02_worklogs.md`](../../../logs/Khoi/M2-02/M2-02_worklogs.md), và Thẻ Trello `[M2-02]` (`6ac462bf8076e675bed68739`).

## Global Constraints
- Tuyệt đối không dùng mã LaTeX hay ký hiệu $: Tất cả công thức toán học phải biểu diễn bằng chữ thường và ký tự trực quan Unicode (ví dụ: alpha, beta, k, E[tau], C(k, B), sum, <=, >=, !=).
- Kế thừa tương thích ngược 100%: `AdaptiveDraftController` phải là subclass của `BudgetController`, hoạt động trơn tru với `AdaptiveProposerAdapter` mà không yêu cầu sửa đổi adapter.
- Quản lý bộ nhớ an toàn: Cửa sổ trượt dùng `deque(maxlen=window_size)` để tự động giải phóng các mẫu cũ, tránh rò rỉ RAM.
- Độc lập phần cứng khi chạy test: Mã nguồn chạy được 100% trên CPU mà không cần GPU/CUDA.

---

### Task 1: Khởi tạo lớp `AdaptiveDraftController` với cửa sổ trượt 10 bước và Warm-up Unbiased

**Files:**
- Modify: `src/proposer/controller.py`
- Modify: `src/proposer/__init__.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `AdaptiveDraftController(window_size=10, ema_beta=0.8, initial_alpha=0.8, ...)`: Subclass của `BudgetController`.
- `AdaptiveDraftController.step_count`: Thuộc tính int đếm số bước xác thực đã thực hiện.
- `AdaptiveDraftController.get_window_samples() -> list[float]`: Trả về danh sách các giá trị `step_alpha` hiện có trong cửa sổ.
- `AdaptiveDraftController.get_effective_alpha() -> float`: Trả về giá trị `alpha_unbiased` đã khử thiên lệch khởi tạo khi `step_count < window_size`.

- [ ] **Step 1: Viết test thất bại kiểm tra kế thừa, cửa sổ trượt và cơ chế Warm-up Unbiased**

```python
# Thêm vào tests/test_budget_controller.py
def test_adaptive_draft_controller_inheritance_and_defaults():
    from src.proposer.controller import AdaptiveDraftController, BudgetController

    controller = AdaptiveDraftController()
    assert isinstance(controller, BudgetController)
    assert controller.window_size == 10
    assert controller.ema_beta == 0.8
    assert controller.initial_alpha == 0.8
    assert controller.get_window_samples() == []
    assert controller.step_count == 0

def test_adaptive_draft_controller_window_tracking():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController(window_size=10)
    for i in range(15):
        controller.update(num_accepted=i % 4, num_drafted=3)

    samples = controller.get_window_samples()
    assert len(samples) == 10
    assert controller.step_count == 15
    assert len(controller._window) == 10

def test_adaptive_draft_controller_unbiased_warmup():
    from src.proposer.controller import AdaptiveDraftController

    # Test trường hợp prompt khó: bước 1 trượt sạch (0/3 accepted)
    controller = AdaptiveDraftController(window_size=10, ema_beta=0.8, initial_alpha=0.8)
    assert controller.get_effective_alpha() == 0.8

    # Bước 1: 0/3 -> step_alpha = 0.0
    controller.update(num_accepted=0, num_drafted=3)
    # Unbiased alpha loại bỏ ảnh hưởng của initial_alpha=0.8, bằng chính xác 0.0
    assert pytest.approx(controller.get_effective_alpha(), 0.01) == 0.0

    # Bước 2: 3/3 -> step_alpha = 1.0
    controller.update(num_accepted=3, num_drafted=3)
    # Mẫu [0.0, 1.0], unbiased alpha = (0.8*0 + 0.2*1.0) / (1 - 0.8^2) = 0.2 / 0.36 = 0.555
    assert pytest.approx(controller.get_effective_alpha(), 0.01) == 0.555
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -k "test_adaptive_draft_controller_inheritance_and_defaults"`
Expected: FAIL với "ImportError / NameError"

- [ ] **Step 3: Cài đặt khung lớp và logic Warm-up Unbiased**

```python
# Cập nhật trong src/proposer/controller.py
class AdaptiveDraftController(BudgetController):
    def __init__(
        self,
        window_size: int = 10,
        ema_beta: float = 0.8,
        initial_alpha: float = 0.8,
        fallback_threshold: float = 0.65,
        high_threshold: float = 0.80,
        candidate_budgets: Optional[List[int]] = None,
        cost_profile: Optional[CostProfile] = None,
        min_free_vram_mb: float = 1500.0,
    ):
        super().__init__(
            window_size=window_size,
            ema_beta=ema_beta,
            initial_alpha=initial_alpha,
            fallback_threshold=fallback_threshold,
            high_threshold=high_threshold,
            candidate_budgets=candidate_budgets,
            cost_profile=cost_profile,
            min_free_vram_mb=min_free_vram_mb,
        )
        self.step_count = 0

    def update(self, num_accepted: int, num_drafted: int) -> float:
        self.step_count += 1
        return super().update(num_accepted=num_accepted, num_drafted=num_drafted)

    def get_window_samples(self) -> List[float]:
        return list(self._window)

    def get_effective_alpha(self) -> float:
        if self.step_count == 0:
            return self.initial_alpha

        beta_power = self.ema_beta ** self.step_count
        denominator = 1.0 - beta_power
        if denominator <= 1e-6:
            return self.alpha_ema

        unbiased = (self.alpha_ema - self.initial_alpha * beta_power) / denominator
        return min(max(unbiased, 0.0), 1.0)
```

Export trong `src/proposer/__init__.py`:
```python
from src.proposer.controller import BudgetController, AdaptiveDraftController, CostProfile, BudgetDecision
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -k "test_adaptive_draft_controller_inheritance_and_defaults or test_adaptive_draft_controller_window_tracking or test_adaptive_draft_controller_unbiased_warmup"`
Expected: PASS 100%

- [ ] **Step 5: Commit**

```bash
git add src/proposer/controller.py src/proposer/__init__.py tests/test_budget_controller.py
git commit -m "feat(proposer): add AdaptiveDraftController with 10-step window and unbiased warmup"
```

---

### Task 2: Cài đặt bộ chẩn đoán thống kê cửa sổ `get_window_stats()`

**Files:**
- Modify: `src/proposer/controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `AdaptiveDraftController.get_window_stats() -> dict`: Trả về dict chứa `mean`, `min`, `max`, `variance`, `is_warmed_up`, `samples`, `step_count`.

- [ ] **Step 1: Viết test cho phương thức `get_window_stats()`**

```python
# Thêm vào tests/test_budget_controller.py
def test_adaptive_draft_controller_window_stats():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController(window_size=5, initial_alpha=0.8)
    stats_empty = controller.get_window_stats()
    assert stats_empty["is_warmed_up"] is False
    assert stats_empty["mean"] == 0.8
    assert stats_empty["variance"] == 0.0
    assert stats_empty["samples"] == []

    # Bơm 3 mẫu: 0.6, 0.8, 1.0
    controller.update(num_accepted=3, num_drafted=5)  # 0.6
    controller.update(num_accepted=4, num_drafted=5)  # 0.8
    controller.update(num_accepted=5, num_drafted=5)  # 1.0

    stats = controller.get_window_stats()
    assert stats["is_warmed_up"] is False  # 3 < 5
    assert pytest.approx(stats["mean"], 0.001) == 0.8
    assert pytest.approx(stats["min"], 0.001) == 0.6
    assert pytest.approx(stats["max"], 0.001) == 1.0
    assert stats["variance"] == pytest.approx(0.02667, abs=1e-4)
    assert len(stats["samples"]) == 3

    # Bơm thêm 2 mẫu để đủ 5 bước
    controller.update(num_accepted=5, num_drafted=5)
    controller.update(num_accepted=5, num_drafted=5)
    stats_full = controller.get_window_stats()
    assert stats_full["is_warmed_up"] is True
    assert len(stats_full["samples"]) == 5
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py::test_adaptive_draft_controller_window_stats`
Expected: FAIL với "AttributeError: 'AdaptiveDraftController' object has no attribute 'get_window_stats'"

- [ ] **Step 3: Cài đặt phương thức `get_window_stats()`**

```python
# Cập nhật trong src/proposer/controller.py
    def get_window_stats(self) -> dict:
        samples = list(self._window)
        is_warmed_up = self.step_count >= self.window_size

        if not samples:
            return {
                "mean": self.initial_alpha,
                "min": self.initial_alpha,
                "max": self.initial_alpha,
                "variance": 0.0,
                "is_warmed_up": False,
                "samples": [],
                "step_count": self.step_count,
            }

        mean_val = sum(samples) / len(samples)
        var_val = sum((x - mean_val) ** 2 for x in samples) / len(samples)

        return {
            "mean": mean_val,
            "min": min(samples),
            "max": max(samples),
            "variance": var_val,
            "is_warmed_up": is_warmed_up,
            "samples": samples,
            "step_count": self.step_count,
        }
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py::test_adaptive_draft_controller_window_stats`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/controller.py tests/test_budget_controller.py
git commit -m "feat(proposer): add get_window_stats diagnostic metrics in AdaptiveDraftController"
```

---

### Task 3: Cài đặt Modular Rules (`apply_budget_rule` & `apply_memory_pressure_rule`)

**Files:**
- Modify: `src/proposer/controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `AdaptiveDraftController.apply_budget_rule(alpha: float) -> tuple[int, str]`: Quy tắc ngưỡng `0.65` và `0.80` điều chỉnh `k in {0, 2, 3, 4}`.
- `AdaptiveDraftController.apply_memory_pressure_rule(free_vram_mb: float, kv_cache_usage_pct: float = 0.0) -> Optional[tuple[int, str]]`: Hook kiểm tra áp lực bộ nhớ.

- [ ] **Step 1: Viết test cho phương thức `apply_budget_rule()`**

```python
# Thêm vào tests/test_budget_controller.py
def test_adaptive_draft_controller_budget_rules():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController(fallback_threshold=0.65, high_threshold=0.80)

    # 1. High acceptance (>= 0.80) -> duy trì k = 3 (hoặc 4 nếu rất cao >= 0.90)
    k, reason = controller.apply_budget_rule(alpha=0.85)
    assert k == 3
    assert "high" in reason.lower()

    k_high, _ = controller.apply_budget_rule(alpha=0.92)
    assert k_high in [3, 4]

    # 2. Mid acceptance [0.65, 0.80) -> co về k = 2
    k_mid, reason_mid = controller.apply_budget_rule(alpha=0.72)
    assert k_mid == 2
    assert "mid" in reason_mid.lower() or "throttle" in reason_mid.lower()

    # 3. Low acceptance (< 0.65) -> ngắt về k = 0
    k_low, reason_low = controller.apply_budget_rule(alpha=0.55)
    assert k_low == 0
    assert "low" in reason_low.lower() or "fallback" in reason_low.lower()
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py::test_adaptive_draft_controller_budget_rules`
Expected: FAIL với "AttributeError: 'AdaptiveDraftController' object has no attribute 'apply_budget_rule'"

- [ ] **Step 3: Cài đặt `apply_budget_rule()` và `apply_memory_pressure_rule()`**

```python
# Cập nhật trong src/proposer/controller.py
    def apply_memory_pressure_rule(
        self,
        free_vram_mb: float,
        kv_cache_usage_pct: float = 0.0,
    ) -> Optional[Tuple[int, str]]:
        if free_vram_mb < self.min_free_vram_mb:
            return (
                0,
                f"VRAM pressure: free {free_vram_mb:.1f}MB < min {self.min_free_vram_mb:.1f}MB",
            )
        return None

    def apply_budget_rule(self, alpha: float) -> Tuple[int, str]:
        if alpha < self.fallback_threshold:
            return (
                0,
                f"Low acceptance fallback: alpha {alpha:.3f} < threshold {self.fallback_threshold:.3f}",
            )

        if alpha >= self.high_threshold:
            if alpha >= 0.90 and 4 in self.candidate_budgets:
                return (4, f"High acceptance (aggressive): alpha {alpha:.3f} >= 0.90, selected k=4")
            return (3, f"High acceptance: alpha {alpha:.3f} >= threshold {self.high_threshold:.3f}, maintain k=3")

        return (
            2,
            f"Mid acceptance: alpha {alpha:.3f} in [{self.fallback_threshold:.2f}, {self.high_threshold:.2f}), throttled to k=2",
        )
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py::test_adaptive_draft_controller_budget_rules`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/controller.py tests/test_budget_controller.py
git commit -m "feat(proposer): add apply_budget_rule and apply_memory_pressure_rule in AdaptiveDraftController"
```

---

### Task 4: Cài đặt luồng tổng thể `select_budget()`, `reset()`, và Telemetry `get_stats()`

**Files:**
- Modify: `src/proposer/controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `AdaptiveDraftController.select_budget(batch_size, free_vram_mb, kv_cache_usage_pct) -> BudgetDecision`
- `AdaptiveDraftController.reset(initial_alpha) -> None`
- `AdaptiveDraftController.get_stats() -> dict`

- [ ] **Step 1: Viết test cho luồng tích hợp `select_budget()`**

```python
# Thêm vào tests/test_budget_controller.py
def test_adaptive_draft_controller_select_budget_flow():
    from src.proposer.controller import AdaptiveDraftController

    controller = AdaptiveDraftController()

    # Khởi tạo (effective_alpha = 0.80) -> k=3
    decision = controller.select_budget()
    assert decision.k == 3

    # VRAM áp lực -> ép k=0
    decision_vram = controller.select_budget(free_vram_mb=1000.0)
    assert decision_vram.k == 0
    assert "vram" in decision_vram.reason.lower()

    # Bước 1 trượt sạch -> k lập tức rơi về 0 để triệt tiêu tụt tốc độ
    controller.update(num_accepted=0, num_drafted=3)
    decision_step1 = controller.select_budget(free_vram_mb=16000.0)
    assert decision_step1.k == 0
    assert decision_step1.alpha_ema < 0.65
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py::test_adaptive_draft_controller_select_budget_flow`
Expected: FAIL (nếu chưa tích hợp `apply_memory_pressure_rule` hoặc `effective_alpha`)

- [ ] **Step 3: Cài đặt `select_budget()`, `reset()`, và `get_stats()`**

```python
# Cập nhật trong src/proposer/controller.py
    def select_budget(
        self,
        batch_size: int = 1,
        free_vram_mb: float = 16000.0,
        kv_cache_usage_pct: float = 0.0,
    ) -> BudgetDecision:
        effective_alpha = self.get_effective_alpha()

        # 1. Memory Pressure Rule (Chống OOM)
        mem_decision = self.apply_memory_pressure_rule(
            free_vram_mb=free_vram_mb,
            kv_cache_usage_pct=kv_cache_usage_pct,
        )
        if mem_decision is not None:
            k, reason = mem_decision
            decision = BudgetDecision(
                k=k,
                alpha_ema=effective_alpha,
                expected_tokens=1.0,
                reason=reason,
            )
            self._history.append({"k": k, "alpha": effective_alpha, "reason": reason})
            return decision

        # 2. Budget Rule (Co giãn k theo alpha)
        k, reason = self.apply_budget_rule(alpha=effective_alpha)
        e_tau = self.estimate_expected_tokens(k, effective_alpha)

        decision = BudgetDecision(
            k=k,
            alpha_ema=effective_alpha,
            expected_tokens=e_tau,
            reason=reason,
        )
        self._history.append({"k": k, "alpha": effective_alpha, "reason": reason})
        return decision

    def reset(self, initial_alpha: Optional[float] = None) -> None:
        super().reset(initial_alpha=initial_alpha)
        if initial_alpha is not None:
            self.initial_alpha = initial_alpha
        self.step_count = 0

    def get_stats(self) -> dict:
        base_stats = super().get_stats()
        base_stats["effective_alpha"] = self.get_effective_alpha()
        base_stats["is_warmed_up"] = self.step_count >= self.window_size
        base_stats["step_count"] = self.step_count
        return base_stats
```

- [ ] **Step 4: Chạy toàn bộ test suite để xác nhận PASS 100%**

Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest.exe tests/test_budget_controller.py tests/test_proposer_adapter.py -v`
Expected: 24/24 tests PASS 100%

- [ ] **Step 5: Commit**

```bash
git add src/proposer/controller.py tests/test_budget_controller.py
git commit -m "feat(proposer): complete select_budget integration and telemetry in AdaptiveDraftController"
```

---

### Task 5: Cập nhật Trello Card, Worklog M2-02 và Handoff

**Files:**
- Modify: `logs/Khoi/M2-02/M2-02_worklogs.md`
- Update: Trello Card `[M2-02]` (`6ac462bf8076e675bed68739`)

- [ ] **Step 1: Đánh dấu hoàn thành Checklist 1 và Checklist 2 trên Trello**
- Check item `6ac4683f5b9ad789868c6e7a`: `[Khôi] Xây dựng lớp AdaptiveDraftController theo dõi alpha_ema qua cửa sổ trượt 10 bước` -> `complete`.
- Check item `6ac4684512d2335186127f6d`: `[Khôi] Cài đặt Budget Rule co giãn k = 0, 2, 3, 4 theo các ngưỡng alpha (0.80, 0.65)` -> `complete`.

- [ ] **Step 2: Cập nhật nhật ký kỹ thuật M2-02**
Ghi lại kết quả nghiệm thu checklist của Khôi, bảng so sánh hành vi giữa `BudgetController` và `AdaptiveDraftController`, và trạng thái sẵn sàng cho Dương cài đặt checklist 3 & 4.

- [ ] **Step 3: Commit tài liệu**

```bash
git add logs/Khoi/M2-02/M2-02_worklogs.md docs/superpowers/plans/2026-10-10-adaptive-draft-controller.md
git commit -m "docs(m2-02): update worklog with AdaptiveDraftController completion and test results"
```
