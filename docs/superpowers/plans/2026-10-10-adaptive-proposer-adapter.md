# AdaptiveProposerAdapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hiện thực hóa lớp AdaptiveProposerAdapter trong `src/proposer/adapter.py` đóng vai trò adapter/hook tích hợp `BudgetController` vào vòng lặp giải mã (Decoding Loop) của vLLM, quản lý trạng thái per-request, điều phối sinh nháp `k*` động và xuất telemetry chuẩn Giao diện A.

**Architecture:** `AdaptiveProposerAdapter` quản lý vòng đời từng request (`request_id`), sở hữu instance `BudgetController` độc lập cho mỗi chuỗi giải mã, cung cấp các hook chuẩn: `propose()` (ra quyết định `k*` và gọi/ngắt draft model), `on_verification_completed()` (nhận phản hồi token đúng và cập nhật `alpha_ema`), và `on_request_end()` (trích xuất `tree_budgets`, `mean_k`, `acceptance_rate` tương thích 100% Giao diện A).

**Tech Stack:** Python 3.10+, dataclasses, typing, pytest.

**Spec:** [`docs/05-thiet-ke-he-thong.md`](file:///d:/DoAn/do-an-ptit-d22/docs/05-thiet-ke-he-thong.md) (§1, §3.1, §4) và [`docs/sprint_3_plan.md`](file:///d:/DoAn/do-an-ptit-d22/docs/sprint_3_plan.md) (Task M2-01 Checklist 2, Task M2-02).

## Global Constraints
- Khung chat và code docstring không dùng ký hiệu LaTeX: Mọi công thức toán biểu diễn bằng chữ thường và ký tự trực quan.
- Tuân thủ hợp đồng Giao diện A: Telemetry xuất ra phải chứa `tree_budgets`, `alpha_ema`, `step_count`, `mean_k`.
- Chống rò rỉ bộ nhớ per-request: Tự động giải phóng trạng thái request sau khi hoàn thành (`on_request_end`) để tránh rò rỉ RAM khi phục vụ hàng nghìn request.
- TDD & Khả năng kiểm thử không cần GPU: Thiết kế theo cơ chế dependency injection / callable để unit test có thể mock draft generator và target verifier dễ dàng.

---

### Task 1: Khởi tạo module `src/proposer/adapter.py` và cấu trúc dữ liệu `ProposalOutput`

**Files:**
- Create: `src/proposer/adapter.py`
- Modify: `src/proposer/__init__.py`
- Test: `tests/test_proposer_adapter.py`

**Interfaces:**
- `ProposalOutput`: Dataclass chứa `draft_tokens: list[int]`, `budget_k: int`, `skipped_draft: bool`, `reason: str`.
- `AdaptiveProposerAdapter`: Khởi tạo với cấu hình `controller_config: Optional[dict] = None`.

- [ ] **Step 1: Viết test thất bại kiểm tra khởi tạo AdaptiveProposerAdapter và ProposalOutput**

```python
# tests/test_proposer_adapter.py
import pytest
from src.proposer.adapter import AdaptiveProposerAdapter, ProposalOutput

def test_adapter_initialization():
    adapter = AdaptiveProposerAdapter()
    assert adapter.active_request_count == 0

def test_proposal_output_structure():
    output = ProposalOutput(
        draft_tokens=[10, 20, 30],
        budget_k=3,
        skipped_draft=False,
        reason="Optimal budget k=3",
    )
    assert output.budget_k == 3
    assert len(output.draft_tokens) == 3
    assert not output.skipped_draft
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_proposer_adapter.py`
Expected: FAIL với "ModuleNotFoundError: No module named 'src.proposer.adapter'"

- [ ] **Step 3: Viết mã nguồn tối thiểu cho Task 1**

```python
# src/proposer/adapter.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from src.proposer.controller import BudgetController, BudgetDecision, CostProfile

@dataclass
class ProposalOutput:
    draft_tokens: List[int]
    budget_k: int
    skipped_draft: bool
    reason: str

class AdaptiveProposerAdapter:
    def __init__(self, controller_kwargs: Optional[dict] = None):
        self.controller_kwargs = controller_kwargs or {}
        self._active_sessions: Dict[str, BudgetController] = {}

    @property
    def active_request_count(self) -> int:
        return len(self._active_sessions)
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_proposer_adapter.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/adapter.py src/proposer/__init__.py tests/test_proposer_adapter.py
git commit -m "feat(proposer): add AdaptiveProposerAdapter skeleton and ProposalOutput"
```

---

### Task 2: Quản lý vòng đời request & Hook đề xuất nháp `propose()`

**Files:**
- Modify: `src/proposer/adapter.py`
- Test: `tests/test_proposer_adapter.py`

**Interfaces:**
- `AdaptiveProposerAdapter.start_request(request_id: str, initial_alpha: Optional[float] = None) -> BudgetController`
- `AdaptiveProposerAdapter.propose(request_id: str, draft_generator: Callable[[int], list[int]], batch_size: int = 1, free_vram_mb: float = 16000.0) -> ProposalOutput`

- [ ] **Step 1: Viết test cho `start_request` và `propose` (cả trường hợp sinh nháp k>0 và bỏ qua khi k=0)**

```python
# Thêm vào tests/test_proposer_adapter.py
def test_propose_with_active_draft():
    adapter = AdaptiveProposerAdapter()
    req_id = "req_001"
    
    # Mock draft generator sinh đúng k tokens
    mock_draft_fn = lambda k: [100 + i for i in range(k)]
    
    # Mặc định initial_alpha=0.8, k* in {3, 4}
    output = adapter.propose(request_id=req_id, draft_generator=mock_draft_fn, batch_size=1)
    assert output.budget_k in [3, 4]
    assert len(output.draft_tokens) == output.budget_k
    assert not output.skipped_draft
    assert adapter.active_request_count == 1

def test_propose_fallback_skips_draft_generator():
    adapter = AdaptiveProposerAdapter(controller_kwargs={"initial_alpha": 0.40, "fallback_threshold": 0.65})
    req_id = "req_fallback"
    
    called = False
    def mock_draft_fn(k):
        nonlocal called
        called = True
        return [999] * k
        
    output = adapter.propose(request_id=req_id, draft_generator=mock_draft_fn)
    assert output.budget_k == 0
    assert output.draft_tokens == []
    assert output.skipped_draft
    # Hàm draft_generator không được phép chạy khi k=0 để tiết kiệm GPU compute
    assert not called
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_proposer_adapter.py -k "test_propose"`
Expected: FAIL với "AttributeError: 'AdaptiveProposerAdapter' object has no attribute 'propose'"

- [ ] **Step 3: Cài đặt phương thức `start_request` và `propose`**

```python
# Cập nhật trong src/proposer/adapter.py
    def start_request(self, request_id: str, initial_alpha: Optional[float] = None) -> BudgetController:
        kwargs = dict(self.controller_kwargs)
        if initial_alpha is not None:
            kwargs["initial_alpha"] = initial_alpha
        controller = BudgetController(**kwargs)
        self._active_sessions[request_id] = controller
        return controller

    def get_controller(self, request_id: str) -> BudgetController:
        if request_id not in self._active_sessions:
            return self.start_request(request_id)
        return self._active_sessions[request_id]

    def propose(
        self,
        request_id: str,
        draft_generator: Callable[[int], List[int]],
        batch_size: int = 1,
        free_vram_mb: float = 16000.0,
    ) -> ProposalOutput:
        controller = self.get_controller(request_id)
        decision: BudgetDecision = controller.select_budget(
            batch_size=batch_size,
            free_vram_mb=free_vram_mb,
        )

        if decision.k == 0:
            return ProposalOutput(
                draft_tokens=[],
                budget_k=0,
                skipped_draft=True,
                reason=decision.reason,
            )

        draft_tokens = draft_generator(decision.k)
        return ProposalOutput(
            draft_tokens=draft_tokens,
            budget_k=decision.k,
            skipped_draft=False,
            reason=decision.reason,
        )
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `.\.venv\Scripts\pytest.exe tests/test_proposer_adapter.py -k "test_propose"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/adapter.py tests/test_proposer_adapter.py
git commit -m "feat(proposer): implement request lifecycle and propose hook in adapter"
```

---

### Task 3: Hook xác thực `on_verification_completed()` & Xuất Telemetry Giao diện A `end_request()`

**Files:**
- Modify: `src/proposer/adapter.py`
- Test: `tests/test_proposer_adapter.py`

**Interfaces:**
- `AdaptiveProposerAdapter.on_verification_completed(request_id: str, num_accepted: int, num_drafted: int) -> float`: Cập nhật `alpha_ema`.
- `AdaptiveProposerAdapter.end_request(request_id: str) -> dict`: Trích xuất telemetry chuẩn Giao diện A và giải phóng bộ nhớ.

- [ ] **Step 1: Viết test cho chu trình hoàn chỉnh (Simulation Loop)**

```python
# Thêm vào tests/test_proposer_adapter.py
def test_simulation_decoding_loop():
    adapter = AdaptiveProposerAdapter(controller_kwargs={"fallback_threshold": 0.65})
    req_id = "req_sim_loop"
    adapter.start_request(req_id, initial_alpha=0.9)

    # Bước 1..3: alpha cao (3/3 đúng) -> giữ k=3 hoặc 4
    for _ in range(3):
        out = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
        assert out.budget_k > 0
        adapter.on_verification_completed(req_id, num_accepted=out.budget_k, num_drafted=out.budget_k)

    # Bước 4..6: alpha tụt dốc (0/3 đúng như W3) -> adapter nhận diện và chuyển về k=0
    for _ in range(4):
        out = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
        if not out.skipped_draft:
            adapter.on_verification_completed(req_id, num_accepted=0, num_drafted=out.budget_k)

    # Bước 7: Phải rơi vào Vanilla Fallback (k=0)
    out_last = adapter.propose(req_id, draft_generator=lambda k: [1] * k)
    assert out_last.budget_k == 0
    assert out_last.skipped_draft

    # Kết thúc request: xuất telemetry và giải phóng bộ nhớ
    telemetry = adapter.end_request(req_id)
    assert "tree_budgets" in telemetry
    assert 0 in telemetry["tree_budgets"]  # Đã có bước k=0
    assert telemetry["tree_budgets"][0] > 0  # Bước đầu tiên có nháp
    assert adapter.active_request_count == 0  # Đã giải phóng RAM
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `.\.venv\Scripts\pytest.exe tests/test_proposer_adapter.py -k "test_simulation_decoding_loop"`
Expected: FAIL với "AttributeError: 'AdaptiveProposerAdapter' object has no attribute 'on_verification_completed'"

- [ ] **Step 3: Cài đặt `on_verification_completed()` và `end_request()`**

```python
# Cập nhật trong src/proposer/adapter.py
    def on_verification_completed(
        self,
        request_id: str,
        num_accepted: int,
        num_drafted: int,
    ) -> float:
        controller = self.get_controller(request_id)
        return controller.update(num_accepted=num_accepted, num_drafted=num_drafted)

    def get_telemetry(self, request_id: str) -> dict:
        controller = self.get_controller(request_id)
        stats = controller.get_stats()
        return {
            "request_id": request_id,
            "lane": "adaptive",
            "alpha_ema": stats["alpha_ema"],
            "tree_budgets": stats["tree_budgets"],
            "mean_accept_len": stats.get("mean_k", 0.0),
            "step_count": stats["step_count"],
        }

    def end_request(self, request_id: str) -> dict:
        telemetry = self.get_telemetry(request_id)
        if request_id in self._active_sessions:
            del self._active_sessions[request_id]
        return telemetry
```

- [ ] **Step 4: Chạy test để xác nhận PASS 100%**

Run: `.\.venv\Scripts\pytest.exe tests/test_proposer_adapter.py -v`
Expected: 100% PASS

- [ ] **Step 5: Commit**

```bash
git add src/proposer/adapter.py tests/test_proposer_adapter.py
git commit -m "feat(proposer): add verification feedback and Interface A telemetry export in adapter"
```

---

### Task 4: Cập nhật tài liệu kỹ thuật & Đánh dấu checklist Trello

**Files:**
- Modify: `logs/Khoi/M2-01/M2-01_worklogs.md`
- Modify: `logs/Khoi/M2-01/README.md`

- [ ] **Step 1: Viết mô tả tích hợp adapter vào worklog**
- [ ] **Step 2: Đánh dấu hoàn thành checklist item 2 trên Trello: `[Khôi] Tích hợp custom proposer hook vào vòng lặp generate của vLLM`**
- [ ] **Step 3: Commit và đồng bộ**
