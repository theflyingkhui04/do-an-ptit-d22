# Kế hoạch thực thi điều chỉnh TreeAdaptiveController v0.2 sau thẩm định

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Khắc phục triệt để các phát hiện thẩm định kỹ thuật từ Bùi Đức Đại, hoàn thiện cơ chế an toàn VRAM, tách biệt số lần quan sát $\alpha$, truyền cấu trúc cây qua adapter và nâng cao độ bao phủ kiểm thử.

**Architecture:** Bổ sung bộ lọc sentinel VRAM và máy trạng thái phục hồi theo nguồn kích hoạt trong tầng 1; tách bộ đếm quan sát thực nghiệm $\alpha$ và bổ sung cơ chế self-healing tại tầng 2; chuẩn hóa trường `topology_name` trong `ProposalOutput` tại tầng adapter; mở rộng kiểm thử biên đạt 100% tỷ lệ vượt qua.

**Tech Stack:** Python 3.10+, PyTorch/vLLM stack, pytest, dataclasses, math.

**Spec:** [`logs/Khoi/spec&plan/treeAdaptiveController/review_alignment_design.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec&plan/treeAdaptiveController/review_alignment_design.md)

## Global Constraints

- Độ trễ ra quyết định trung bình (mean latency) trên CPU: $\le 0.05\text{ ms}$ ($50\ \mu\text{s}$).
- Toàn bộ suite test FlexSDS (59 tests hiện tại cộng các test mới) phải vượt qua tuyệt đối.
- Giữ tương thích ngược 100% với `AdaptiveDraftController` v0.1 và hợp đồng Interface A hiện hữu.
- Không sửa mã nguồn scheduler của M4-02-04; chỉ cung cấp giao diện chuẩn để Đại tích hợp.
- Tuân thủ nghiêm ngặt quy tắc chính tả tiếng Việt: không viết hoa kiểu Title Case của tiếng Anh.

## Review Focus

1. **Sensor flap/outage during starvation:** Mất tín hiệu VRAM (giá trị $10^9$, `None`, `NaN`) khi hệ thống vừa bị cạn kiệt VRAM vật lý $\implies$ controller phải giữ vững trạng thái an toàn $T_0$, không được phục hồi sớm chỉ dựa vào KV-cache.
2. **Invalid KV-cache reading:** Tín hiệu KV-cache nhận giá trị bất thường (`None`, `NaN`, $<0$, $>100$) $\implies$ kích hoạt Memory Guard phòng vệ ngay lập tức.
3. **Cold-start & all-vanilla sequence:** Chuỗi suy luận liên tục chạy $T_0$ (không có token nháp nào được sinh và xác thực) $\implies$ không làm trôi trọng số debiasing $\beta^{N_{\text{obs}}}$ của EMA.
4. **NaN/Inf telemetry corruption:** Tín hiệu chấp thuận bị nhiễm `NaN` hoặc `Inf` $\implies$ controller tự phục hồi về `initial_alpha`, không làm sập tiến trình suy luận.
5. **Cost model failure injection:** Bộ giải mô hình chi phí ném ngoại lệ bất kỳ $\implies$ kích hoạt tầng 5 bọc thép `zero_crash_fallback` trả về $T_0$ an toàn.

---

### Task 1: Bộ lọc sentinel VRAM và máy trạng thái phục hồi phòng vệ bộ nhớ

**Files:**
- Modify: `src/proposer/tree_adaptive.py:40-265`
- Test: `tests/test_tree_adaptive_controller.py`

**Interfaces:**
- Consumes: `TreeAdaptiveController.select_budget(batch_size, free_vram_mb, kv_cache_usage_pct)`
- Produces: `_is_valid_vram()`, `_is_valid_kv()`, `_guard_trigger_source`, điều kiện chuyển trạng thái Memory Guard có kiểm tra nguồn gốc kích hoạt.

- [ ] **Step 1: Viết test kiểm tra trạng thái missing/sentinel VRAM và máy trạng thái phục hồi**

Thêm test vào `tests/test_tree_adaptive_controller.py`:
```python
def test_memory_guard_sentinel_vram_and_recovery_safety():
    ctrl = TreeAdaptiveController(min_free_vram_mb=1500.0, kv_cache_high_pct=90.0, kv_cache_recover_pct=80.0)
    # Case A: VRAM sentinel 1e9 but KV high (78% > tight threshold 75%) -> guard triggers
    dec = ctrl.select_budget(batch_size=1, free_vram_mb=1.0e9, kv_cache_usage_pct=78.0)
    assert dec.topology_name == "T0"
    assert dec.status == "memory_guard"

    # Case B: Starved by physical VRAM then telemetry flaps to 1e9 -> must hold guard
    ctrl.reset()
    ctrl.select_budget(batch_size=1, free_vram_mb=1000.0, kv_cache_usage_pct=50.0)
    assert ctrl._memory_guard_active is True
    # Telemetry flaps to sentinel 1e9 with low KV (50%) -> should NOT prematurely recover
    dec2 = ctrl.select_budget(batch_size=1, free_vram_mb=1.0e9, kv_cache_usage_pct=50.0)
    assert dec2.topology_name == "T0"
    assert dec2.status == "memory_guard"
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py::test_memory_guard_sentinel_vram_and_recovery_safety -v`
Kết quả mong đợi: FAIL do logic hiện tại chưa kiểm tra sentinel `1e9` và chưa có `_guard_trigger_source`.

- [ ] **Step 3: Hiện thực hóa bộ lọc sentinel và máy trạng thái phục hồi trong `tree_adaptive.py`**

- Định nghĩa hằng số `SENTINEL_VRAM_THRESHOLD_MB = 1.0e8`.
- Thêm `_is_valid_vram(val)` và `_is_valid_kv(val)`.
- Khởi tạo `self._guard_trigger_source: str = "none"` trong `__init__()` và `reset()`.
- Cập nhật Tầng 1:
  - Nếu `not _is_valid_kv(kv_cache_usage_pct)`: kích hoạt guard ngay.
  - Nếu VRAM hợp lệ: `is_vram_starved = free_vram_mb < self.min_free_vram_mb`, `kv_threshold = self.kv_cache_high_pct`.
  - Nếu VRAM không hợp lệ: `is_vram_starved = False`, áp dụng ngưỡng siết chặt `kv_threshold = 75.0%`.
  - Ghi nhận `_guard_trigger_source` là `"vram"`, `"kv"`, hoặc `"both"`.
  - Trong điều kiện phục hồi: nếu `_guard_trigger_source` có `"vram"`, bắt buộc phải có VRAM hợp lệ $\ge 1500\text{ MB}$ và $KV \le 65.0\%$ (hoặc `kv_cache_recover_pct`).

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py::test_memory_guard_sentinel_vram_and_recovery_safety -v`
Kết quả mong đợi: PASS.

- [ ] **Step 5: Commit mã nguồn**

```bash
git add src/proposer/tree_adaptive.py tests/test_tree_adaptive_controller.py
git commit -m "fix(controller): enforce sentinel vram filtering and safe recovery state machine"
```

---

### Task 2: Tách biệt bộ đếm quan sát và cơ chế tự phục hồi cho $\alpha$

**Files:**
- Modify: `src/proposer/tree_adaptive.py:74-135`
- Test: `tests/test_tree_adaptive_controller.py`

**Interfaces:**
- Consumes: `TreeAdaptiveController.update(num_accepted, num_drafted)`, `TreeAdaptiveController.get_effective_alpha()`
- Produces: `self._observation_count`, self-healing logic xử lý `NaN`/`Inf`, debiasing dựa trên `_observation_count`.

- [ ] **Step 1: Viết test kiểm tra tách biệt observation count và tự phục hồi `NaN`**

Thêm test vào `tests/test_tree_adaptive_controller.py`:
```python
def test_alpha_observation_count_and_nan_self_healing():
    ctrl = TreeAdaptiveController(initial_alpha=0.8, ema_beta=0.8)
    # Cold-start or T0 steps with num_drafted == 0
    ctrl.update(num_accepted=0, num_drafted=0)
    assert ctrl._step_count == 1
    assert ctrl._observation_count == 0
    assert ctrl.get_effective_alpha() == 0.8  # No decay of beta_power

    # Valid step
    ctrl.update(num_accepted=3, num_drafted=4)
    assert ctrl._observation_count == 1
    assert 0.0 <= ctrl.get_effective_alpha() <= 1.0

    # Inject NaN into alpha_ema
    import math
    ctrl.alpha_ema = float("nan")
    alpha_recovered = ctrl.get_effective_alpha()
    assert not math.isnan(alpha_recovered)
    assert alpha_recovered == 0.8
    assert not math.isnan(ctrl.alpha_ema)  # Self-healed
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py::test_alpha_observation_count_and_nan_self_healing -v`
Kết quả mong đợi: FAIL do chưa có `_observation_count` và chưa tự phục hồi `NaN`.

- [ ] **Step 3: Hiện thực hóa `_observation_count` và self-healing trong `tree_adaptive.py`**

- Thêm `self._observation_count: int = 0` trong `__init__()` và `reset()`.
- Trong `get_effective_alpha()`:
  - Kiểm tra `math.isnan(self.alpha_ema)` hoặc `math.isinf(self.alpha_ema)` $\implies$ gán `self.alpha_ema = self.initial_alpha`.
  - Nếu `self._observation_count == 0`: trả về `self.initial_alpha`.
  - Tính debiasing dựa trên `beta_power = self.ema_beta ** self._observation_count`.
- Trong `update()`:
  - Luôn tăng `self._step_count += 1`.
  - Nếu `num_drafted <= 0` hoặc `num_accepted < 0`: return `self.alpha_ema` (không tăng `_observation_count`).
  - Nếu `num_drafted > 0`: tăng `self._observation_count += 1`, tính `step_alpha`, cập nhật EMA.

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py::test_alpha_observation_count_and_nan_self_healing -v`
Kết quả mong đợi: PASS.

- [ ] **Step 5: Commit mã nguồn**

```bash
git add src/proposer/tree_adaptive.py tests/test_tree_adaptive_controller.py
git commit -m "fix(controller): decouple observation count for ema debiasing and add nan self-healing"
```

---

### Task 3: Hài hòa cơ chế kẹp tải với probing chu kỳ khi $B > 24$

**Files:**
- Modify: `src/proposer/tree_adaptive.py:270-310`
- Test: `tests/test_tree_adaptive_controller.py`

**Interfaces:**
- Consumes: `TreeAdaptiveController.select_budget()` tại Tầng 3
- Produces: Hành vi ưu tiên Memory Guard tuyệt đối và chỉ probe $T_4$ khi bộ nhớ an toàn.

- [ ] **Step 1: Viết test kiểm tra probing không kích hoạt khi Memory Guard đang chặn**

Thêm test vào `tests/test_tree_adaptive_controller.py`:
```python
def test_concurrency_throttling_and_probing_safety():
    ctrl = TreeAdaptiveController()
    # High load B=32, low alpha -> throttled to T0
    ctrl.alpha_ema = 0.5
    ctrl._observation_count = 10
    # Simulate steps up to probing interval (30) under memory pressure
    for _ in range(35):
        dec = ctrl.select_budget(batch_size=32, free_vram_mb=500.0, kv_cache_usage_pct=95.0)
        assert dec.topology_name == "T0"
        assert dec.status == "memory_guard"  # Never probe under memory guard!
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử vượt qua hoặc điều chỉnh**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py::test_concurrency_throttling_and_probing_safety -v`

- [ ] **Step 3: Tinh chỉnh logic Tầng 3 trong `tree_adaptive.py`**

Đảm bảo cấu trúc lệnh: Tầng 1 (Memory Guard) luôn return trước Tầng 3 (Concurrency Throttling). Khi ở Tầng 3, bộ nhớ đã được xác nhận an toàn. Khi đạt `probe_interval = 30`, probe tối thiểu $T_4$ kèm lý do rõ ràng.

- [ ] **Step 4: Chạy lại toàn bộ test suite hiện có để xác nhận không hồi quy**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py -v`
Kết quả mong đợi: PASS toàn bộ.

- [ ] **Step 5: Commit mã nguồn**

```bash
git add src/proposer/tree_adaptive.py tests/test_tree_adaptive_controller.py
git commit -m "refactor(controller): harmonize concurrency throttling and adaptive probing rules"
```

---

### Task 4: Đóng gói `topology_name` trong `ProposalOutput` tại adapter

**Files:**
- Modify: `src/proposer/adapter.py:18-100`
- Test: `tests/test_tree_adaptive_controller.py`

**Interfaces:**
- Consumes: `TreeAdaptiveDecision.topology_name`, `AdaptiveDraftController`
- Produces: `ProposalOutput.topology_name: Optional[str] = None`

- [ ] **Step 1: Viết test kiểm tra `ProposalOutput` mang đúng `topology_name`**

Thêm test vào `tests/test_tree_adaptive_controller.py`:
```python
def test_adapter_proposal_output_carries_topology_name():
    from src.proposer.adapter import AdaptiveProposerAdapter
    from src.proposer.tree_adaptive import TreeAdaptiveController
    from src.proposer.controller import AdaptiveDraftController

    # With TreeAdaptiveController
    tree_adapter = AdaptiveProposerAdapter(controller_cls=TreeAdaptiveController)
    out_tree = tree_adapter.propose("req_tree", lambda k: [1] * k, batch_size=1)
    assert hasattr(out_tree, "topology_name")
    assert out_tree.topology_name in ["T0", "T4", "T8", "T16", "T32"]

    # With legacy linear AdaptiveDraftController
    linear_adapter = AdaptiveProposerAdapter(controller_cls=AdaptiveDraftController)
    out_linear = linear_adapter.propose("req_linear", lambda k: [1] * k, batch_size=1)
    assert hasattr(out_linear, "topology_name")
    assert out_linear.topology_name is None or out_linear.topology_name == "T0"
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py::test_adapter_proposal_output_carries_topology_name -v`
Kết quả mong đợi: FAIL do `ProposalOutput` hiện chưa có trường `topology_name`.

- [ ] **Step 3: Cập nhật `ProposalOutput` và hàm `propose()` trong `adapter.py`**

- Trong `ProposalOutput`:
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
  - Trích xuất `topology_name`:
    `topo = getattr(decision, "topology_name", None)`
  - Đưa `topology_name=topo` vào cả hai nhánh trả về (`decision.k == 0` và `decision.k > 0`).

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py::test_adapter_proposal_output_carries_topology_name -v`
Kết quả mong đợi: PASS.

- [ ] **Step 5: Commit mã nguồn**

```bash
git add src/proposer/adapter.py tests/test_tree_adaptive_controller.py
git commit -m "feat(adapter): expose topology_name in ProposalOutput for downstream scheduler"
```

---

### Task 5: Bổ sung bộ kiểm thử biên đầy đủ và đo lường độ trễ SLA

**Files:**
- Modify: `tests/test_tree_adaptive_controller.py`

**Interfaces:**
- Consumes: Toàn bộ API của `TreeAdaptiveController` và `AdaptiveProposerAdapter`
- Produces: Test 15 (Missing/Sentinel VRAM), Test 16 (Invalid input telemetry), Test 17 (Infeasible candidates vs Zero-crash exception), Test 18 (Topology name propagation), SLA latency verification.

- [ ] **Step 1: Viết test cho trường hợp toàn bộ candidate không khả thi và ngoại lệ mô hình chi phí**

Thêm test vào `tests/test_tree_adaptive_controller.py`:
```python
def test_fallback_gate_vs_zero_crash_exception():
    ctrl = TreeAdaptiveController()
    # Path A: Infeasible speedup (force mock or candidate with speedup <= 1.0)
    # Controller returns fallback_gate, not crashing
    dec = ctrl.select_budget(batch_size=1)
    assert dec.status in ["ok", "fallback_gate"]

    # Path B: Exception injection in cost model
    class FaultyCostModel:
        def select_optimal_topology(self, *args, **kwargs):
            raise RuntimeError("CUDA device communication failure")
        def solve_alpha_star(self, *args, **kwargs):
            raise RuntimeError("Cost model solver failure")

    faulty_ctrl = TreeAdaptiveController(cost_model=FaultyCostModel())
    dec_crash = faulty_ctrl.select_budget(batch_size=1)
    assert dec_crash.topology_name == "T0"
    assert dec_crash.status == "zero_crash_fallback"
    assert "Zero-crash fallback" in dec_crash.reason
```

- [ ] **Step 2: Chạy toàn bộ test suite của `TreeAdaptiveController`**

Chạy: `cd do-an-ptit-d22 && pytest tests/test_tree_adaptive_controller.py -v`
Kết quả mong đợi: Tất cả các test (bao gồm các test mới) đều PASS 100%.

- [ ] **Step 3: Chạy full regression suite của toàn bộ FlexSDS**

Chạy: `cd do-an-ptit-d22 && pytest tests/ -v`
Kết quả mong đợi: Tất cả các test suite của FlexSDS PASS.

- [ ] **Step 4: Chạy benchmark kiểm tra lại SLA độ trễ ra quyết định**

Chạy: `cd do-an-ptit-d22 && python3 -c "
import time
from src.proposer.tree_adaptive import TreeAdaptiveController
ctrl = TreeAdaptiveController()
# Warmup
for _ in range(100): ctrl.select_budget(batch_size=1)
start = time.perf_counter()
N = 10000
for _ in range(N):
    ctrl.select_budget(batch_size=1, free_vram_mb=8000.0, kv_cache_usage_pct=40.0)
elapsed = time.perf_counter() - start
mean_ms = (elapsed / N) * 1000
print(f'Mean decision latency: {mean_ms:.4f} ms')
assert mean_ms < 0.05, f'SLA violated: {mean_ms} ms >= 0.05 ms'
print('SLA PASS!')
"`
Kết quả mong đợi: `Mean decision latency < 0.05 ms`, `SLA PASS!`.

- [ ] **Step 5: Commit mã nguồn test**

```bash
git add tests/test_tree_adaptive_controller.py
git commit -m "test(controller): complete boundary test suite and verify decision latency SLA"
```

---

### Task 6: Cập nhật tài liệu đặc tả, worklogs và ranh giới phối hợp liên task

**Files:**
- Modify: `logs/Khoi/spec&plan/treeAdaptiveController/spec.md`
- Modify: `logs/Khoi/spec&plan/treeAdaptiveController/plan.md`
- Modify: `logs/Khoi/spec&plan/treeAdaptiveController/M4-02-05_worklogs.md`

**Interfaces:**
- Consumes: Bằng chứng kiểm thử và các quyết định kỹ thuật đã thống nhất với review của Đại
- Produces: Bản đặc tả Revision 1.1, kế hoạch cập nhật, và nhật ký kỹ thuật chuẩn văn phong tiếng Việt.

- [ ] **Step 1: Cập nhật `spec.md` sang Revision 1.1**

- Điều chỉnh định nghĩa SLA: ghi rõ "thời gian ra quyết định trung bình (mean latency) $< 0.05\text{ ms}$ trên CPU".
- Hạ cấp câu chữ: thay "tuyệt đối không OOM" thành "giảm thiểu rủi ro OOM qua cơ chế phòng vệ hai tầng và bộ lọc sentinel".
- Làm rõ $\alpha_{\text{step}} = L/d$ là empirical yield index cho bộ điều khiển phản hồi.
- Bổ sung trường `topology_name` trong đặc tả `ProposalOutput`.
- Ghi nhận phân định ranh giới phối hợp với M4-02-04 của Đại (scheduler gỡ bỏ `max_k = 4`).

- [ ] **Step 2: Cập nhật `plan.md` và `M4-02-05_worklogs.md`**

- Ghi lại các bước điều chỉnh kỹ thuật theo thẩm định của Đại (C01–C10, P01–P05).
- Đính kèm kết quả benchmark độ trễ và tỷ lệ vượt qua của toàn bộ suite kiểm thử.
- Tuân thủ nghiêm ngặt quy tắc chính tả tiếng Việt: không viết hoa kiểu Title Case.

- [ ] **Step 3: Commit tài liệu và đẩy lên nhánh remote**

```bash
git add logs/Khoi/spec&plan/treeAdaptiveController/
git commit -m "docs(controller): update spec to rev 1.1 and sync worklogs post technical review"
git push origin feat/m4-02-05-tree-adaptive-controller
```
