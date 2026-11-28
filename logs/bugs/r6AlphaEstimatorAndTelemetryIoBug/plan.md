# Kế hoạch triển khai: Ước lượng tỉ lệ chấp nhận Discounted Geometric MLE và ghi telemetry theo lô cho R6 Hardware-Adaptive

> **Dành cho kỹ sư AI:** KỸ NĂNG BẮT BUỘC: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi kế hoạch này từng tác vụ một. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu (Goal):** Khắc phục đồng thời hai nguyên nhân khiến R6 Hardware-Adaptive thua Static `K=3` ở tải thấp trên NVIDIA A10 (`ablation-2311`): (1) thay ước lượng $\alpha$ bị lệch (`accepted/drafted`) bằng Discounted-Count Geometric MLE có prior Beta yếu ($w_0 = 2.0, \beta_{\text{mle}} = 0.90$) để triệt tiêu tình trạng chọn `K=1` ~50% bước và bẫy khóa `K=0` ở bước 1; (2) đổi `_write_step` sang persistent file handle + ghi theo lô và thêm cờ `--no-step-telemetry` để loại bỏ chi phí `+2..+4 ms/bước` của I/O telemetry.

**Kiến trúc (Architecture):**
1. Trong `src/proposer/controller.py`: bổ sung bộ tích lũy suy giảm `(_mle_accepted, _mle_tested)` và `get_geometric_alpha()` vào `AdaptiveDraftController`, tự động kích hoạt (`alpha_estimator="auto"`) khi có `cost_profile.matrix` trong khi giữ nguyên `self.alpha_ema` legacy để tương thích 100% với `m4_readiness.py` và unit test cũ.
2. Trong `src/server/budget_planner.py` và `src/server/adaptive_scheduler.py`: mở rộng telemetry với `tested_draft_positions` / `geometric_alpha_mle` / `alpha_estimator`, chuyển `_write_step` sang persistent file handle + đệm theo lô (`step_telemetry_flush_every`), tự flush trong `finish_request()` và đóng sạch trong `close()` / `shutdown()`.
3. Trong `src/flexsds/server/local_runner.py`, `src/flexsds/server/modal_runner.py` và `src/flexsds/cli/main.py`: bật mặc định `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY="64"` khi chạy Adaptive serving, thêm cờ opt-out `--no-step-telemetry` và chuyển tiếp đầy đủ qua `ModalServingRunner`.

**Công nghệ sử dụng (Tech Stack):** Python 3.11+, Pytest, JSON/JSONL buffered I/O, Argparse.

**Đặc tả kỹ thuật (Spec):** `do-an-ptit-d22/logs/bugs/r6AlphaEstimatorAndTelemetryIoBug/2026-11-24-r6-geometric-mle-alpha-and-batched-telemetry-spec.md`

## Các ràng buộc toàn cục (Global Constraints)

- **NFR-01 (Không lệch thống kê):** Trên chuỗi mẫu hình học cụt với $p \in \{0.4, 0.6, 0.8\}$ và mọi $K \in \{1, 2, 3, 4\}$, `get_geometric_alpha()` hội tụ trong biên `±0.02` quanh $p$ thật.
- **NFR-02 (Hiệu năng I/O):** Khi `step_telemetry_flush_every = 64` và ghi 130 bản ghi liên tiếp, chỉ gọi `open()` đúng **1 lần**.
- **NFR-03 (Tương thích ngược tuyệt đối):** Không làm vỡ `self.alpha_ema` legacy, `experiments/m4_readiness.py:114`, `test_cli_serving_readiness.py:123` hay bất kỳ test nào trong 236+ test offline hiện hành.
- **NFR-04 (Chiến lược Git):** Thực hiện trên nhánh mới `fix/r6-geometric-mle-and-batched-telemetry` tách từ `develop@229b3e4`; không sửa nhánh audit cũ và không sửa tay file `profile_sm_86_remote_a10g_fcda5486b12d.json`.
- **Chính tả tiếng Việt:** Không viết hoa kiểu Title Case tiếng Anh trong tài liệu/comment tiếng Việt; chỉ viết hoa chữ cái đầu câu/tiêu đề, danh từ riêng (FlexSDS, NVIDIA A10, A10G, EAGLE3, vLLM, Modal, Hardware-Adaptive) và từ viết tắt chuẩn (GPU, CPU, I/O, VRAM, KV, JSON, JSONL, CLI, EMA, MLE, FR, NFR, AC).

## Tiêu điểm kiểm tra (Review Focus)

1. **Tham số không hợp lệ (`bool`, `NaN`, ngoài miền) cho `mle_beta`, `mle_prior_weight`, `alpha_estimator`:** `AdaptiveDraftController.__init__` phải raise `ValueError` tường minh (kiểm tra trong Tác vụ 1).
2. **Đầu vào biên của `update(num_accepted, num_drafted)` (`< 0`, `> num_drafted`, `num_drafted <= 0`) và `reset()`:** Không bao giờ làm `_mle_tested <= 0` hoặc `get_geometric_alpha()` ra ngoài `[0, 1]`; `reset()` khôi phục đúng `w_0 * initial_alpha` và `w_0` (kiểm tra trong Tác vụ 1).
3. **Lỗi I/O khi ghi đệm (`step_telemetry_path` trỏ tới thư mục không tồn tại):** `flush()` / `close()` ghi log cảnh báo, xả/xóa buffer tránh rò RAM, tuyệt đối không ném ngoại lệ ra ngoài `plan_step()` hay `finish_request()` (kiểm tra trong Tác vụ 2).
4. **Bảo toàn dữ liệu cuối request khi chưa đủ lô (`len(buffer) < step_telemetry_flush_every`):** `finish_request()` phải gọi `self.flush()` để đảm bảo toàn bộ step của request đã xuống đĩa (kiểm tra trong Tác vụ 2).
5. **Rò biến `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY` từ shell cha & `step_telemetry_flush_every` sai kiểu (`True`, `0`):** Bị xóa sạch ở nhánh `--no-adaptive` hoặc `--no-step-telemetry`, và ném `ValueError` khi cấu hình sai kiểu/giá trị (kiểm tra trong Tác vụ 3).

---

### Tác vụ 1: Triển khai Discounted-Count Geometric MLE trong `AdaptiveDraftController`

**Files:**
- Modify: `do-an-ptit-d22/src/proposer/controller.py:10-25,215-315,580-695`
- Create: `do-an-ptit-d22/tests/test_r6_geometric_mle_and_telemetry_io.py`

**Interfaces:**
- Consumes: `num_accepted: int`, `num_drafted: int`, `self.cost_profile`, `self.initial_alpha`.
- Produces:
  - `BudgetDecision.alpha_estimator: Optional[str] = None`
  - `AdaptiveDraftController.__init__(..., mle_beta: float = 0.90, mle_prior_weight: float = 2.0, alpha_estimator: str = "auto")`
  - `AdaptiveDraftController.get_geometric_alpha(self) -> float`
  - `AdaptiveDraftController._resolved_alpha_estimator(self) -> str`
  - `AdaptiveDraftController.get_stats(self) -> dict` bổ sung `"geometric_alpha_mle"`, `"alpha_estimator"`, `"total_tested_positions"`.

- [x] **Bước 1: Viết các bài test thất bại cho AC-01, AC-02, AC-03 và Review Focus #1–#2**

Tạo `do-an-ptit-d22/tests/test_r6_geometric_mle_and_telemetry_io.py`:
```python
"""Tests for R6 Discounted-Count Geometric MLE alpha estimator and batched step telemetry I/O."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def _load_a10g_profile():
    from src.proposer.controller import EmpiricalCostProfile

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    matrix = {
        (int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): float(v)
        for k, v in data["cost_matrix_ms"].items()
    }
    return EmpiricalCostProfile(matrix=matrix), data["grid"]["candidate_budgets"]


@pytest.mark.parametrize("true_p", [0.40, 0.60, 0.80])
@pytest.mark.parametrize("k_fixed", [1, 2, 3, 4])
def test_geometric_mle_is_unbiased_across_all_k(true_p, k_fixed):
    """AC-01 & NFR-01: get_geometric_alpha() hội tụ về p thật (±0.02) ở mọi K in {1..4}."""
    from src.proposer.controller import AdaptiveDraftController

    rng = random.Random(2026 + k_fixed * 100 + int(true_p * 100))
    ctrl = AdaptiveDraftController(
        initial_alpha=0.8,
        mle_beta=0.90,
        mle_prior_weight=2.0,
        alpha_estimator="geometric_mle",
    )
    estimates = []
    for step in range(4000):
        acc = 0
        for _ in range(k_fixed):
            if rng.random() < true_p:
                acc += 1
            else:
                break
        ctrl.update(num_accepted=acc, num_drafted=k_fixed)
        if step >= 500:
            estimates.append(ctrl.get_geometric_alpha())
    mean_est = sum(estimates) / len(estimates)
    assert abs(mean_est - true_p) <= 0.02


def test_geometric_mle_avoids_step1_deadlock_but_falls_back_on_hard_prompt():
    """AC-02: Bước 1 trượt không gây khóa K=0; prompt khó liên tiếp trượt vẫn về K=0."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_profile()
    ctrl = AdaptiveDraftController(
        initial_alpha=0.80,
        candidate_budgets=budgets,
        cost_profile=profile,
        mle_beta=0.90,
        mle_prior_weight=2.0,
    )
    ctrl.update(num_accepted=0, num_drafted=4)
    assert ctrl.get_geometric_alpha() > 0.50
    d1 = ctrl.select_budget(batch_size=1)
    assert d1.k > 0
    assert d1.alpha_estimator == "geometric_mle"

    # Prompt khó (luôn trượt a=0): B=16 rơi về K=0 trong <= 3 bước; B=1 trong <= 15 bước
    ctrl_b16 = AdaptiveDraftController(initial_alpha=0.80, candidate_budgets=budgets, cost_profile=profile)
    b16_k0_step = None
    for s in range(1, 6):
        ctrl_b16.update(0, 2)
        if ctrl_b16.select_budget(batch_size=16).k == 0:
            b16_k0_step = s
            break
    assert b16_k0_step is not None and b16_k0_step <= 3

    ctrl_b1 = AdaptiveDraftController(initial_alpha=0.80, candidate_budgets=budgets, cost_profile=profile)
    b1_k0_step = None
    for s in range(1, 20):
        ctrl_b1.update(0, 1)
        if ctrl_b1.select_budget(batch_size=1).k == 0:
            b1_k0_step = s
            break
    assert b1_k0_step is not None and b1_k0_step <= 15


def test_geometric_mle_keeps_k3_k4_dominant_at_low_batch_and_k0_at_high_batch():
    """AC-03: Mô phỏng 200 request 60 token trên A10G thật (p=0.60)."""
    from src.proposer.controller import AdaptiveDraftController

    profile, budgets = _load_a10g_profile()
    for batch_size in (1, 8, 32):
        rng = random.Random(123 + batch_size)
        counts = Counter()
        for _ in range(200):
            ctrl = AdaptiveDraftController(
                initial_alpha=0.80,
                candidate_budgets=budgets,
                cost_profile=profile,
            )
            gen = 0
            while gen < 60:
                k = ctrl.select_budget(batch_size=batch_size).k
                counts[k] += 1
                acc = 0
                for _ in range(k):
                    if rng.random() < 0.60:
                        acc += 1
                    else:
                        break
                gen += acc + 1
                if k > 0:
                    ctrl.update(acc, k)
        tot = sum(counts.values())
        if batch_size == 1:
            assert (counts[3] + counts[4]) / tot >= 0.80
            assert counts[0] / tot < 0.02
        elif batch_size == 8:
            assert counts[3] / tot >= 0.75
        else:
            assert counts[0] / tot >= 0.90


def test_mle_parameter_validation_clamping_and_reset():
    """Review Focus #1 & #2: Validate tham số bool/out-of-range, kẹp input xấu và reset sạch."""
    from src.proposer.controller import AdaptiveDraftController

    for bad_kwargs in (
        {"mle_beta": True},
        {"mle_beta": 0.0},
        {"mle_beta": 1.0},
        {"mle_beta": float("nan")},
        {"mle_prior_weight": True},
        {"mle_prior_weight": -0.5},
        {"alpha_estimator": "unknown"},
    ):
        with pytest.raises(ValueError):
            AdaptiveDraftController(**bad_kwargs)

    ctrl = AdaptiveDraftController(initial_alpha=0.75, mle_prior_weight=2.0, alpha_estimator="geometric_mle")
    ctrl.update(num_accepted=-5, num_drafted=3)
    assert 0.0 <= ctrl.get_geometric_alpha() <= 1.0
    ctrl.update(num_accepted=10, num_drafted=3)
    assert 0.0 <= ctrl.get_geometric_alpha() <= 1.0
    before_zero = ctrl.get_geometric_alpha()
    ctrl.update(num_accepted=0, num_drafted=0)
    assert ctrl.get_geometric_alpha() == pytest.approx(before_zero)

    ctrl.reset(initial_alpha=0.65)
    assert ctrl.get_geometric_alpha() == pytest.approx(0.65)
    assert ctrl.get_stats()["total_tested_positions"] == 0
```

- [x] **Bước 2: Chạy test để xác nhận thất bại**

Chạy từ `do-an-ptit-d22/`: `PYTHONPATH=src:. pytest tests/test_r6_geometric_mle_and_telemetry_io.py -v`
Kết quả mong đợi: FAIL (`TypeError: unexpected keyword argument 'mle_beta'`).

- [x] **Bước 3: Triển khai FR-01 trong `src/proposer/controller.py`**

1. Thêm `alpha_estimator: Optional[str] = None` vào `BudgetDecision`.
2. Trong `AdaptiveDraftController.__init__`: thêm và kiểm tra `mle_beta: float = 0.90`, `mle_prior_weight: float = 2.0`, `alpha_estimator: str = "auto"`; khởi tạo `_mle_accepted`, `_mle_tested`, `_total_accepted_tokens`, `_total_tested_positions`.
3. Thêm `_resolved_alpha_estimator()` và `get_geometric_alpha()`.
4. Trong `update()`: giữ nguyên `super().update(...)`; khi `num_drafted > 0`, kẹp `accepted = min(max(int(num_accepted), 0), int(num_drafted))`, `tested = accepted + (1 if accepted < num_drafted else 0)`, cập nhật `_mle_accepted`, `_mle_tested`, và bộ đếm tổng.
5. Trong `select_budget()`: tính `estimator = self._resolved_alpha_estimator()` và `decision_alpha = self.get_geometric_alpha() if estimator == "geometric_mle" else self.get_effective_alpha()`; gắn `alpha_estimator=estimator` trên mọi `BudgetDecision` trả về (kể cả `memory_guard`, `cost_model`, `threshold`).
6. Cập nhật `reset()` và `get_stats()`.

- [x] **Bước 4: Chạy test mới và bộ controller cũ để xác nhận pass và không hồi quy**

Chạy: `PYTHONPATH=src:. pytest tests/test_r6_geometric_mle_and_telemetry_io.py tests/test_hardware_cost_driven_controller.py tests/test_budget_controller.py tests/test_memory_pressure_rule.py -v`
Kết quả mong đợi: 100% PASS.

- [x] **Bước 5: Commit Tác vụ 1**

```bash
git add src/proposer/controller.py tests/test_r6_geometric_mle_and_telemetry_io.py
git commit -m "feat(controller): add discounted-count geometric MLE alpha estimator for R6"
```

---

### Tác vụ 2: Ghi step telemetry theo lô bằng persistent handle và mở rộng trường MLE trong `AdaptiveBudgetPlanner` & `AdaptiveAsyncScheduler`

**Files:**
- Modify: `do-an-ptit-d22/src/server/budget_planner.py:45-357`
- Modify: `do-an-ptit-d22/src/server/adaptive_scheduler.py:206-211`
- Modify: `do-an-ptit-d22/tests/test_r6_geometric_mle_and_telemetry_io.py`

**Interfaces:**
- Consumes: `AdaptiveDraftController.get_geometric_alpha()` và `BudgetDecision.alpha_estimator` từ Tác vụ 1.
- Produces:
  - `AdaptiveBudgetPlanner.__init__(..., step_telemetry_flush_every: int = 1)`
  - `AdaptiveBudgetPlanner.from_env()` đọc `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY` (mặc định `"1"`).
  - `AdaptiveBudgetPlanner.flush(self) -> None` và `AdaptiveBudgetPlanner.close(self) -> None`.
  - `_decision_record()` bổ sung `"geometric_alpha_mle"` và `"alpha_estimator"`; `record_verification()` bổ sung `"tested_draft_positions"` và `"geometric_alpha_mle"`.
  - `AdaptiveAsyncScheduler.shutdown()` gọi `self._planner.close()`.

- [x] **Bước 1: Viết các bài test thất bại cho AC-04, FR-02 và Review Focus #3–#4**

Bổ sung vào `do-an-ptit-d22/tests/test_r6_geometric_mle_and_telemetry_io.py`:
```python
def test_planner_batched_step_telemetry_opens_once_and_flushes_on_finish_and_close(tmp_path, monkeypatch):
    """AC-04, NFR-02 & Review Focus #4: Flush theo lô 64 dòng chỉ open() 1 lần và tự flush khi finish/close."""
    import builtins
    from src.server.budget_planner import AdaptiveBudgetPlanner

    step_path = tmp_path / "steps.jsonl"
    real_open = builtins.open
    open_count = 0

    def counting_open(file, *args, **kwargs):
        nonlocal open_count
        if str(file) == str(step_path):
            open_count += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    planner = AdaptiveBudgetPlanner.from_env(
        {
            "FLEXSDS_PROFILE_PATH": str(A10G_PROFILE_PATH),
            "ADAPTIVE_STEP_TELEMETRY_PATH": str(step_path),
            "ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY": "64",
        }
    )
    for _ in range(65):
        planner.plan_step(["r1"], batch_size=1, kv_usage=0.1, free_vram_mb=16000.0)
        planner.record_verification("r1", num_drafted=3, num_accepted=1)

    # 130 bản ghi >= 2 lô 64 -> đã ghi ít nhất 128 dòng, chỉ mở file đúng 1 lần
    assert open_count == 1
    assert len(step_path.read_text(encoding="utf-8").splitlines()) == 128

    # finish_request tự flush nốt 2 bản ghi còn lại trong đệm
    planner.finish_request("r1")
    lines = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 130
    assert open_count == 1

    # Kiểm tra các trường telemetry MLE mới (FR-02)
    planned_rec = next(r for r in lines if r["record_type"] == "planned_decision")
    fb_rec = next(r for r in lines if r["record_type"] == "verification_feedback")
    assert planned_rec["alpha_estimator"] == "geometric_mle"
    assert planned_rec["geometric_alpha_mle"] is not None
    assert fb_rec["tested_draft_positions"] == 2
    assert fb_rec["geometric_alpha_mle"] is not None

    planner.close()
    assert open_count == 1


def test_planner_batched_io_error_does_not_crash_or_leak_buffer(tmp_path, caplog):
    """Review Focus #3: Đường dẫn lỗi không làm sập plan_step/flush/close và không tích tụ buffer."""
    from src.server.budget_planner import AdaptiveBudgetPlanner

    bad_path = tmp_path / "missing_dir" / "steps.jsonl"
    planner = AdaptiveBudgetPlanner(step_telemetry_path=str(bad_path), step_telemetry_flush_every=2)
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.1)
    planner.plan_step(["r1"], batch_size=1, kv_usage=0.1)
    assert planner._step_buffer == []
    planner.close()
    assert "step telemetry" in caplog.text.lower()
```

- [x] **Bước 2: Chạy test để xác nhận thất bại**

Chạy: `PYTHONPATH=src:. pytest tests/test_r6_geometric_mle_and_telemetry_io.py -k "test_planner_" -v`
Kết quả mong đợi: FAIL (`TypeError: unexpected keyword argument 'step_telemetry_flush_every'`).

- [x] **Bước 3: Triển khai FR-02 và FR-03 trong `src/server/budget_planner.py` và `src/server/adaptive_scheduler.py`**

1. Trong `AdaptiveBudgetPlanner.__init__`:
   - Thêm `step_telemetry_flush_every: int = 1` (validate `not isinstance(..., bool) and isinstance(..., int) and >= 1`).
   - Khởi tạo `self._step_fh = None`, `self._step_buffer: List[str] = []`, và đăng ký `weakref.finalize(self, _close_handle_safely, ...)` hoặc đóng an toàn trong `close()`.
2. Trong `_write_step(self, record: dict)`:
   - Serialize `line = json.dumps(dict(record, recorded_at=time.time()), allow_nan=False) + "\n"`, append vào `self._step_buffer`; nếu `len(self._step_buffer) >= self.step_telemetry_flush_every` thì gọi `self.flush()`.
3. Thêm `flush(self)` và `close(self)`:
   - `flush()`: nếu không có `self._step_buffer` hoặc không có `self.step_telemetry_path`, return. Mở lười `self._step_fh = open(self.step_telemetry_path, "a", encoding="utf-8", buffering=65536)` nếu chưa mở; `write("".join(self._step_buffer))` và `self._step_fh.flush()`. Trong khối `finally` (hoặc khi `OSError`), luôn `self._step_buffer.clear()` để không bao giờ phình RAM khi đĩa lỗi; nếu `OSError`, đóng/reset `self._step_fh = None` và `logger.exception(...)`.
   - `close()`: gọi `self.flush()`, đóng `self._step_fh` nếu đang mở và đặt lại `None`.
4. Trong `plan_step()`:
   - Chỉ dựng `_decision_record` khi `self.step_telemetry_path or self.telemetry_path` (giữ `_last_decisions` khi cần `finish_request`).
   - Truyền `geometric_alpha` của controller vào `_decision_record(...)`.
5. Trong `record_verification()`:
   - Bổ sung `"tested_draft_positions"` và `"geometric_alpha_mle"` vào bản ghi `verification_feedback`.
6. Trong `finish_request()`:
   - Gọi `self.flush()` trước khi trả về.
7. Trong `from_env()`:
   - Đọc `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY` (mặc định `"1"`), parse `int` (báo `ValueError` nếu `< 1`).
8. Trong `src/server/adaptive_scheduler.py` (`shutdown()`):
   - Gọi `self._planner.close()` trong khối `finally:` cùng `release_ledger(...)`.

- [x] **Bước 4: Chạy test mới và bộ test planner/readiness cũ để xác nhận pass**

Chạy: `PYTHONPATH=src:. pytest tests/test_r6_geometric_mle_and_telemetry_io.py tests/test_budget_planner.py tests/test_m4_readiness.py tests/test_adaptive_scheduler_glue.py -v`
Kết quả mong đợi: 100% PASS.

- [x] **Bước 5: Commit Tác vụ 2**

```bash
git add src/server/budget_planner.py src/server/adaptive_scheduler.py tests/test_r6_geometric_mle_and_telemetry_io.py
git commit -m "perf(planner): batch step telemetry with persistent handle and log geometric MLE fields"
```

---

### Tác vụ 3: Bật batch flush mặc định `64` và thêm cờ `--no-step-telemetry` ở Runner & CLI

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/server/local_runner.py:67-90,265-295`
- Modify: `do-an-ptit-d22/src/flexsds/server/modal_runner.py:73-86`
- Modify: `do-an-ptit-d22/src/flexsds/cli/main.py:181-215,435-545`
- Modify: `do-an-ptit-d22/tests/test_r6_geometric_mle_and_telemetry_io.py`

**Interfaces:**
- Consumes: `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY` từ Tác vụ 2.
- Produces:
  - `LocalServingRunner.setup_environment()` đọc `enable_step_telemetry: bool = True` và `step_telemetry_flush_every: int = 64`.
  - `ModalServingRunner.setup_environment()` chuyển tiếp `"enable_step_telemetry"` và `"step_telemetry_flush_every"`.
  - CLI `flexsds serve --no-step-telemetry`.

- [x] **Bước 1: Viết các bài test thất bại cho AC-05 và Review Focus #5**

Bổ sung vào `do-an-ptit-d22/tests/test_r6_geometric_mle_and_telemetry_io.py`:
```python
def test_runner_and_modal_step_telemetry_batching_and_opt_out(monkeypatch):
    """AC-05 & Review Focus #5: Mặc định flush_every=64; opt-out tắt hẳn path và dọn biến rò."""
    from flexsds.server.local_runner import LocalServingRunner
    from flexsds.server.modal_runner import ModalServingRunner

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    base = {
        "target_model": data["models"]["target"],
        "draft_model": data["models"]["draft"],
        "draft_method": data["draft_method"],
        "target_revision": data["models"]["target_revision"],
        "draft_revision": data["models"]["draft_revision"],
        "profile_path": str(A10G_PROFILE_PATH),
        "enable_adaptive": True,
        "target_gpu": "A10G",
    }

    local_runner = LocalServingRunner()
    env_default = local_runner.setup_environment({"serving": dict(base)})
    assert "ADAPTIVE_STEP_TELEMETRY_PATH" in env_default
    assert env_default["ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY"] == "64"

    monkeypatch.setenv("ADAPTIVE_STEP_TELEMETRY_PATH", "/tmp/leaked_steps.jsonl")
    monkeypatch.setenv("ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY", "16")
    env_off = local_runner.setup_environment({"serving": dict(base, enable_step_telemetry=False)})
    assert "ADAPTIVE_STEP_TELEMETRY_PATH" not in env_off
    assert "ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY" not in env_off

    env_static = local_runner.setup_environment({"serving": {"enable_adaptive": False}})
    assert "ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY" not in env_static

    for bad_flush in (True, 0, -4):
        with pytest.raises(ValueError):
            local_runner.setup_environment({"serving": dict(base, step_telemetry_flush_every=bad_flush)})

    modal_env = ModalServingRunner().setup_environment(
        {"serving": dict(base, enable_step_telemetry=False, step_telemetry_flush_every=128)}
    )
    payload = json.loads(modal_env["FLEXSDS_SERVING_CONFIG"])
    assert payload["enable_step_telemetry"] is False
    assert payload["step_telemetry_flush_every"] == 128


def test_cli_no_step_telemetry_flag_forwards_opt_out(monkeypatch, capsys):
    """AC-05: Cờ --no-step-telemetry trên flexsds serve được truyền vào Modal serving config."""
    from types import SimpleNamespace
    import flexsds.cli.main as cli_main

    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    fake_report = SimpleNamespace(
        draft_method=data["draft_method"],
        eagle_version="EAGLE-3.1",
        target_revision=data["models"]["target_revision"],
        draft_revision=data["models"]["draft_revision"],
        use_heterogeneous_vocab=False,
        identified_method="eagle3",
        oom_risk=False,
        warnings=[],
        recommendations=[],
        summary=lambda: "Compatibility: OK",
    )
    fake_preflight = SimpleNamespace(report=fake_report, launch_reasons=lambda allow_oom: [])
    monkeypatch.setattr(cli_main, "evaluate_serving_preflight", lambda *a, **kw: fake_preflight)
    monkeypatch.setattr(
        cli_main,
        "load_profile_for_hardware",
        lambda *a, **kw: (object(), A10G_PROFILE_PATH.name),
    )
    monkeypatch.setattr(cli_main, "resolve_profile_path", lambda name: A10G_PROFILE_PATH)

    rc = cli_main.main(
        [
            "serve",
            "--backend",
            "modal",
            "--gpu",
            "A10G",
            "--target",
            data["models"]["target"],
            "--draft",
            data["models"]["draft"],
            "--no-step-telemetry",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"enable_step_telemetry": false' in out
```

- [x] **Bước 2: Chạy test để xác nhận thất bại**

Chạy: `PYTHONPATH=src:. pytest tests/test_r6_geometric_mle_and_telemetry_io.py -k "runner_and_modal or cli_no_step" -v`
Kết quả mong đợi: FAIL.

- [x] **Bước 3: Cập nhật `local_runner.py`, `modal_runner.py`, và `cli/main.py`**

1. Trong `src/flexsds/server/local_runner.py`:
   - Validate `step_telemetry_flush_every` trong `_validate_execution_options()` (mặc định `64`; nếu được truyền phải là `int` không-`bool` `>= 1`).
   - Trong `setup_environment()`:
     - Ở nhánh `enable_adaptive=True`: nếu `serving_cfg.get("enable_step_telemetry", True)` là `True`, đặt `ADAPTIVE_STEP_TELEMETRY_PATH` và `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY = str(flush_every)`; nếu `False`, xóa cả hai biến này khỏi `env`.
     - Ở nhánh `enable_adaptive=False`: thêm `"ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY"` vào danh sách `leaked` được `env.pop(...)`.
2. Trong `src/flexsds/server/modal_runner.py`:
   - Thêm `"enable_step_telemetry"`, `"step_telemetry_flush_every"` vào tuple `keys`.
3. Trong `src/flexsds/cli/main.py`:
   - Thêm `--no-step-telemetry` (`action="store_true"`) vào `serve_parser`.
   - Truyền `"enable_step_telemetry": not args.no_step_telemetry` vào `cfg["serving"]`.

- [x] **Bước 4: Chạy test mới và test runner/matched cũ**

Chạy: `PYTHONPATH=src:. pytest tests/test_r6_geometric_mle_and_telemetry_io.py tests/test_matched_execution_contract.py tests/test_local_runner.py tests/test_modal_runner_cli.py -v`
Kết quả mong đợi: 100% PASS.

- [x] **Bước 5: Commit Tác vụ 3**

```bash
git add src/flexsds/server/local_runner.py src/flexsds/server/modal_runner.py src/flexsds/cli/main.py tests/test_r6_geometric_mle_and_telemetry_io.py
git commit -m "feat(cli,runner): default batched step telemetry flush=64 and add --no-step-telemetry flag"
```

---

### Tác vụ 4: Kiểm thử hồi quy toàn diện, rà soát đối nghịch và chốt tiến độ

**Files:**
- Create: `do-an-ptit-d22/logs/bugs/r6AlphaEstimatorAndTelemetryIoBug/progress.md`
- Modify: `do-an-ptit-d22/logs/bugs/r6AlphaEstimatorAndTelemetryIoBug/plan.md` & `2026-11-24-r6-geometric-mle-alpha-and-batched-telemetry-spec.md`

- [x] **Bước 1: Chạy toàn bộ bộ test hồi quy offline (AC-06) và kiểm tra toàn vẹn profile A10G**

Chạy từ `do-an-ptit-d22/`:
```bash
PYTHONPATH=src:. pytest \
  tests/test_r6_geometric_mle_and_telemetry_io.py \
  tests/test_matched_execution_contract.py \
  tests/test_hardware_cost_driven_controller.py \
  tests/test_budget_controller.py \
  tests/test_budget_planner.py \
  tests/test_memory_pressure_rule.py \
  tests/test_proposer_adapter.py \
  tests/test_tree_adaptive_controller.py \
  tests/test_modal_native_interpreter.py \
  tests/test_modal_app_container_paths.py \
  tests/test_modal_runner_cli.py \
  tests/test_local_runner.py \
  tests/test_m4_readiness.py \
  tests/test_cost_model*.py \
  tests/test_lineage*.py \
  tests/test_adaptive*.py \
  tests/test_hardware_profiler_benchmark.py \
  tests/test_profile_import.py -q
git diff -- src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json
```
Kết quả mong đợi: 100% PASS (0 failed) và `git diff` profile A10G rỗng.

- [x] **Bước 2: Ghi `progress.md` và cập nhật trạng thái spec/plan kèm hướng dẫn chạy kiểm chứng lại trên cùng 1 container A10 cho Dương**

- [x] **Bước 3: Commit tài liệu tiến độ**

```bash
git add logs/bugs/r6AlphaEstimatorAndTelemetryIoBug/
git commit -m "docs(progress): record completion of R6 geometric MLE alpha estimator and batched telemetry"
```
