# BudgetController v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hiện thực hóa lớp BudgetController v0 tính toán ngân sách cây nháp N* (k*) theo tỷ lệ chấp nhận trượt (rolling average alpha / EMA), tích hợp mô hình chi phí suy luận T4 và cơ chế fallback tự động ngắt về Vanilla (k=0) để triệt tiêu hiện tượng tăng tốc âm.

**Architecture:** Module `src/proposer/controller.py` hoạt động như runtime controller độc lập. Controller theo dõi telemetry của từng bước giải mã (số token nháp vs số token được chấp nhận), tính toán `alpha_ema` qua cửa sổ trượt `W=10`, tính toán số token kỳ vọng `E[tau(k)] = 1 + alpha + ... + alpha^k`, đối chiếu với bảng chi phí phần cứng `C(k, B)` và áp lực bộ nhớ VRAM để ra quyết định `k* in {0, 2, 3, 4}`.

**Tech Stack:** Python 3.10+, dataclasses, collections.deque, math, typing, pytest.

**Spec:** [`docs/05-thiet-ke-he-thong.md`](file:///d:/DoAn/do-an-ptit-d22/docs/05-thiet-ke-he-thong.md) (§2, §3.1) và [`docs/sprint_3_plan.md`](file:///d:/DoAn/do-an-ptit-d22/docs/sprint_3_plan.md) (Task M2-01, M2-02).

## Global Constraints
- Khung chat và code docstring không dùng ký hiệu LaTeX: Mọi công thức toán biểu diễn bằng chữ thường và ký tự trực quan (vi dụ: alpha, tau, E[tau], C(k, B), sum).
- Rời rạc hóa ngân sách: k in {0, 2, 3, 4} để đảm bảo tính ổn định của tensor và CUDA Graph trên vLLM.
- Chống rò rỉ bộ nhớ (VRAM & RAM): Sử dụng `deque(maxlen=10)` cố định cho rolling window, không giữ reference rác.
- Độc lập phần cứng khi unit test: Controller không phụ thuộc trực tiếp vào GPU CUDA, có thể chạy và test 100% trên CPU.

---

### Task 1: Khởi tạo module `src/proposer/` và cấu trúc dữ liệu `CostProfile`, `BudgetDecision`

**Files:**
- Create: `src/proposer/__init__.py`
- Create: `src/proposer/controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `CostProfile`: Dataclass chứa ma trận chi phí `cost_matrix: dict[tuple[int, int], float]` và phương thức `get_cost(k: int, batch_size: int) -> float`.
- `BudgetDecision`: Dataclass chứa `k: int`, `alpha_ema: float`, `expected_tokens: float`, `reason: str`.

- [ ] **Step 1: Viết test thất bại kiểm tra khởi tạo CostProfile và BudgetDecision**

```python
# tests/test_budget_controller.py
import pytest
from src.proposer.controller import CostProfile, BudgetDecision

def test_cost_profile_lookup():
    matrix = {
        (0, 1): 20.0,
        (2, 1): 25.0,
        (3, 1): 28.0,
        (4, 1): 32.0,
    }
    profile = CostProfile(matrix=matrix)
    assert profile.get_cost(k=3, batch_size=1) == 28.0
    assert profile.get_cost(k=0, batch_size=1) == 20.0

def test_cost_profile_fallback_missing_batch():
    matrix = {(0, 1): 20.0, (3, 1): 28.0}
    profile = CostProfile(matrix=matrix)
    # Nếu batch size chưa được profile, fallback về batch gần nhất
    assert profile.get_cost(k=3, batch_size=4) == 28.0
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py::test_cost_profile_lookup`
Expected: FAIL với "ModuleNotFoundError: No module named 'src.proposer'"

- [ ] **Step 3: Viết mã nguồn tối thiểu cho Task 1**

```python
# src/proposer/__init__.py
"""Adaptive Speculative Proposer Module."""
from src.proposer.controller import BudgetController, CostProfile, BudgetDecision

__all__ = ["BudgetController", "CostProfile", "BudgetDecision"]
```

```python
# src/proposer/controller.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from collections import deque

@dataclass
class BudgetDecision:
    k: int
    alpha_ema: float
    expected_tokens: float
    reason: str

@dataclass
class CostProfile:
    matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def get_cost(self, k: int, batch_size: int = 1) -> float:
        if (k, batch_size) in self.matrix:
            return self.matrix[(k, batch_size)]
        
        # Fallback 1: Tìm key có cùng k
        matching_k = [cost for (cand_k, _), cost in self.matrix.items() if cand_k == k]
        if matching_k:
            return matching_k[0]
            
        # Fallback 2: Ước lượng tuyến tính mặc định nếu chưa có profile
        base_t = 20.0  # ms
        draft_step_t = 3.0  # ms
        return base_t if k == 0 else (base_t + k * draft_step_t)
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -k "test_cost_profile"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/__init__.py src/proposer/controller.py tests/test_budget_controller.py
git commit -m "feat(proposer): add CostProfile and BudgetDecision data structures"
```

---

### Task 2: Cài đặt thuật toán Rolling Alpha & Ước lượng Token kỳ vọng `E[tau]`

**Files:**
- Modify: `src/proposer/controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `BudgetController.update(num_accepted: int, num_drafted: int) -> float`: Cập nhật telemetry và trả về `alpha_ema`.
- `BudgetController.get_rolling_alpha() -> float`: Lấy giá trị `alpha_ema` hiện tại.
- `BudgetController.estimate_expected_tokens(k: int, alpha: float) -> float`: Tính `E[tau(k)] = 1 + alpha + ... + alpha^k`.

- [ ] **Step 1: Viết test kiểm tra tính toán Rolling Alpha (EMA & Window) và E[tau]**

```python
# Thêm vào tests/test_budget_controller.py
from src.proposer.controller import BudgetController

def test_estimate_expected_tokens():
    controller = BudgetController()
    # Khi k = 0, chỉ có 1 token của vanilla
    assert controller.estimate_expected_tokens(k=0, alpha=0.9) == 1.0
    
    # Khi k = 1, E[tau] = 1 + alpha
    assert pytest.approx(controller.estimate_expected_tokens(k=1, alpha=0.8), 0.001) == 1.8
    
    # Khi k = 3, alpha = 0.5: E[tau] = 1 + 0.5 + 0.25 + 0.125 = 1.875
    assert pytest.approx(controller.estimate_expected_tokens(k=3, alpha=0.5), 0.001) == 1.875

def test_rolling_alpha_updates():
    controller = BudgetController(window_size=5, ema_beta=0.8, initial_alpha=0.8)
    assert controller.get_rolling_alpha() == 0.8
    
    # Bước 1: trúng 3/3 (alpha_curr = 1.0)
    # alpha_ema = 0.8 * 0.8 + 0.2 * 1.0 = 0.64 + 0.2 = 0.84
    new_alpha = controller.update(num_accepted=3, num_drafted=3)
    assert pytest.approx(new_alpha, 0.001) == 0.84
    assert pytest.approx(controller.get_rolling_alpha(), 0.001) == 0.84
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -k "test_estimate_expected_tokens or test_rolling_alpha_updates"`
Expected: FAIL với "AttributeError / BudgetController not defined"

- [ ] **Step 3: Cài đặt logic toán học trong BudgetController**

```python
# Cập nhật trong src/proposer/controller.py
class BudgetController:
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
        self.window_size = window_size
        self.ema_beta = ema_beta
        self.alpha_ema = initial_alpha
        self.fallback_threshold = fallback_threshold
        self.high_threshold = high_threshold
        self.candidate_budgets = candidate_budgets or [0, 2, 3, 4]
        self.cost_profile = cost_profile or CostProfile()
        self.min_free_vram_mb = min_free_vram_mb

        self._window: deque[float] = deque(maxlen=window_size)
        self._history: List[dict] = []

    def get_rolling_alpha(self) -> float:
        return self.alpha_ema

    def estimate_expected_tokens(self, k: int, alpha: float) -> float:
        if k <= 0:
            return 1.0
        # E[tau(k)] = 1 + alpha + alpha^2 + ... + alpha^k
        gain = 1.0
        curr_power = 1.0
        for _ in range(k):
            curr_power *= alpha
            gain += curr_power
        return gain

    def update(self, num_accepted: int, num_drafted: int) -> float:
        if num_drafted <= 0:
            step_alpha = self.alpha_ema
        else:
            step_alpha = min(max(num_accepted / num_drafted, 0.0), 1.0)

        self._window.append(step_alpha)
        self.alpha_ema = self.ema_beta * self.alpha_ema + (1.0 - self.ema_beta) * step_alpha
        return self.alpha_ema
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -k "test_estimate_expected_tokens or test_rolling_alpha_updates"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/controller.py tests/test_budget_controller.py
git commit -m "feat(proposer): implement rolling alpha EMA and expected token estimation"
```

---

### Task 3: Cài đặt logic ra quyết định ngân sách `select_budget()` (Heuristic & Cost Model Argmax)

**Files:**
- Modify: `src/proposer/controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `BudgetController.select_budget(batch_size: int = 1, free_vram_mb: float = 16000.0) -> BudgetDecision`

- [ ] **Step 1: Viết test cho các kịch bản chuyển đổi ngân sách**

```python
# Thêm vào tests/test_budget_controller.py
def test_select_budget_vram_pressure():
    controller = BudgetController(min_free_vram_mb=2000.0)
    # VRAM chỉ còn 1000MB -> Phải ép k = 0
    decision = controller.select_budget(batch_size=1, free_vram_mb=1000.0)
    assert decision.k == 0
    assert "vram" in decision.reason.lower()

def test_select_budget_heuristic_high_alpha():
    controller = BudgetController(initial_alpha=0.88)
    decision = controller.select_budget(batch_size=1)
    assert decision.k in [3, 4]
    assert decision.k > 0

def test_select_budget_fallback_low_alpha():
    controller = BudgetController(initial_alpha=0.50, fallback_threshold=0.65)
    decision = controller.select_budget(batch_size=1)
    # alpha = 0.50 < 0.65 -> ngắt về 0 (Vanilla)
    assert decision.k == 0
    assert "fallback" in decision.reason.lower() or "alpha" in decision.reason.lower()

def test_select_budget_cost_model_argmax():
    # Giả định T4 cost: Vanilla C(0)=20ms. k=3 C(3)=30ms.
    # Với alpha=0.9: E[tau(3)] = 3.439. Tốc độ spec = 3.439 / 30 = 0.1146 tok/ms
    # Tốc độ vanilla = 1 / 20 = 0.05 tok/ms -> Chọn k=3 hoặc 4.
    cost_matrix = {
        (0, 1): 20.0,
        (2, 1): 25.0,
        (3, 1): 30.0,
        (4, 1): 40.0,
    }
    profile = CostProfile(matrix=cost_matrix)
    controller = BudgetController(cost_profile=profile, initial_alpha=0.90)
    decision = controller.select_budget(batch_size=1)
    assert decision.k > 0
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -k "test_select_budget"`
Expected: FAIL với "AttributeError: 'BudgetController' object has no attribute 'select_budget'"

- [ ] **Step 3: Cài đặt phương thức `select_budget()`**

```python
# Cập nhật trong src/proposer/controller.py
    def select_budget(
        self,
        batch_size: int = 1,
        free_vram_mb: float = 16000.0,
    ) -> BudgetDecision:
        alpha = self.alpha_ema

        # 1. Memory Pressure Rule
        if free_vram_mb < self.min_free_vram_mb:
            decision = BudgetDecision(
                k=0,
                alpha_ema=alpha,
                expected_tokens=1.0,
                reason=f"VRAM pressure: free {free_vram_mb:.1f}MB < min {self.min_free_vram_mb:.1f}MB",
            )
            self._history.append({"k": 0, "alpha": alpha, "reason": decision.reason})
            return decision

        # 2. Safety Fallback Gate
        if alpha < self.fallback_threshold:
            decision = BudgetDecision(
                k=0,
                alpha_ema=alpha,
                expected_tokens=1.0,
                reason=f"Low acceptance fallback: alpha {alpha:.3f} < threshold {self.fallback_threshold}",
            )
            self._history.append({"k": 0, "alpha": alpha, "reason": decision.reason})
            return decision

        # 3. Cost-Model Argmax hoặc Heuristic Thresholds
        c_vanilla = self.cost_profile.get_cost(k=0, batch_size=batch_size)
        vanilla_efficiency = 1.0 / max(c_vanilla, 1e-4)

        best_k = 0
        best_efficiency = vanilla_efficiency
        best_expected_tokens = 1.0

        for cand_k in self.candidate_budgets:
            if cand_k == 0:
                continue
            e_tau = self.estimate_expected_tokens(cand_k, alpha)
            c_cand = self.cost_profile.get_cost(k=cand_k, batch_size=batch_size)
            efficiency = e_tau / max(c_cand, 1e-4)

            # Heuristic tie-breaker / threshold guard
            if cand_k == 4 and alpha < self.high_threshold:
                # Chỉ mở k=4 nếu alpha đạt high_threshold (>= 0.80)
                continue

            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_k = cand_k
                best_expected_tokens = e_tau

        reason = (
            "Vanilla fallback (efficiency bound)"
            if best_k == 0
            else f"Optimal budget k={best_k} with eff={best_efficiency:.4f}"
        )

        decision = BudgetDecision(
            k=best_k,
            alpha_ema=alpha,
            expected_tokens=best_expected_tokens,
            reason=reason,
        )
        self._history.append({"k": best_k, "alpha": alpha, "reason": decision.reason})
        return decision
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -k "test_select_budget"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/controller.py tests/test_budget_controller.py
git commit -m "feat(proposer): add select_budget with VRAM guard, fallback gate, and cost argmax"
```

---

### Task 4: Quản lý vòng đời và xuất dữ liệu Telemetry (`reset`, `get_stats`)

**Files:**
- Modify: `src/proposer/controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- `BudgetController.reset(initial_alpha: Optional[float] = None) -> None`: Khôi phục trạng thái cho request mới.
- `BudgetController.get_stats() -> dict`: Xuất dữ liệu thống kê phục vụ Giao diện A (`tree_budgets`, `step_count`, `mean_k`).

- [ ] **Step 1: Viết test cho `reset` và `get_stats`**

```python
# Thêm vào tests/test_budget_controller.py
def test_controller_reset_and_stats():
    controller = BudgetController(initial_alpha=0.8)
    controller.select_budget()
    controller.update(num_accepted=3, num_drafted=3)
    controller.select_budget()

    stats = controller.get_stats()
    assert "tree_budgets" in stats
    assert len(stats["tree_budgets"]) == 2
    assert stats["step_count"] == 1

    controller.reset(initial_alpha=0.75)
    assert controller.get_rolling_alpha() == 0.75
    assert controller.get_stats()["tree_budgets"] == []
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py::test_controller_reset_and_stats`
Expected: FAIL với "AttributeError: 'BudgetController' object has no attribute 'get_stats'"

- [ ] **Step 3: Cài đặt phương thức `reset` và `get_stats`**

```python
# Cập nhật trong src/proposer/controller.py
    def reset(self, initial_alpha: Optional[float] = None) -> None:
        self.alpha_ema = initial_alpha if initial_alpha is not None else 0.8
        self._window.clear()
        self._history.clear()

    def get_stats(self) -> dict:
        tree_budgets = [h["k"] for h in self._history]
        return {
            "alpha_ema": self.alpha_ema,
            "tree_budgets": tree_budgets,
            "step_count": len(self._window),
            "mean_k": (sum(tree_budgets) / len(tree_budgets)) if tree_budgets else 0.0,
        }
```

- [ ] **Step 4: Chạy toàn bộ test suite để xác nhận PASS 100%**

Run: `.\.venv\Scripts\pytest.exe tests/test_budget_controller.py -v`
Expected: 100% PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/controller.py tests/test_budget_controller.py
git commit -m "feat(proposer): add reset and telemetry export matching Interface A"
```

---

### Task 5: Viết nhật ký kỹ thuật và chuẩn bị nghiệm thu M2-01

**Files:**
- Create: `logs/Khoi/M2-01/README.md`
- Create: `logs/Khoi/M2-01/M2-01_worklogs.md`

- [ ] **Step 1: Viết tài liệu tổng kết thiết kế toán học và kết quả kiểm thử**
Ghi lại chi tiết:
- Động lực bài toán và mục tiêu giải quyết tăng tốc âm W3.
- Công thức toán học (không LaTeX): `E[tau]`, `alpha_ema`, tỷ số hiệu năng `E[tau]/C(k, B)`.
- Kết quả chạy test độc lập của `BudgetController v0`.
- Handoff bàn giao cho Bùi Đại viết unit test mô phỏng kịch bản (Checklist 3 của thẻ Trello M2-01).

- [ ] **Step 2: Commit và push tài liệu lên nhánh tính năng**

```bash
git add logs/Khoi/M2-01/
git commit -m "docs(m2-01): add technical worklog and BudgetController v0 design documentation"
```
