# Đặc tả kỹ thuật: Sửa ước lượng tỉ lệ chấp nhận theo hình học (Discounted Geometric MLE) và tối ưu ghi telemetry theo lô cho R6 Hardware-Adaptive

- **Mã tài liệu**: SPEC-R6-ALPHA-IO-20261124
- **Ngày tạo**: 2026-11-24
- **Tác giả**: Đội ngũ phát triển FlexSDS (D22 PTIT)
- **Trạng thái**: Bản nháp đề xuất chờ phê duyệt (Proposed — chưa sửa code)
- **Liên kết thực nghiệm**: Báo cáo ablation ngày 23/11/2026 trên NVIDIA A10 (`ablation-2311`, `r6-2311b`, `probe-2311`, mốc mã nguồn `develop@229b3e4`) và câu hỏi mở `M-09` trong `docs/03-quyet-dinh-thiet-ke.md`
- **Phạm vi tác động**:
  - `src/proposer/controller.py` (`BudgetDecision`, `AdaptiveDraftController`)
  - `src/server/budget_planner.py` (`AdaptiveBudgetPlanner`, `_write_step`, `flush`, `close`, `from_env`, `_decision_record`)
  - `src/server/adaptive_scheduler.py` (`AdaptiveAsyncScheduler.shutdown`)
  - `src/flexsds/server/local_runner.py` (`LocalServingRunner.setup_environment`)
  - `src/flexsds/server/modal_runner.py` (`ModalServingRunner.setup_environment`)
  - `src/flexsds/cli/main.py` (`parse_args`, cờ `--no-step-telemetry`)
  - `tests/test_r6_geometric_mle_and_telemetry_io.py` (tạo mới)

---

## 1. Bối cảnh và bằng chứng thực nghiệm

Trong chiến dịch đo đối chứng cùng một container GPU NVIDIA A10 (`ablation-2311`, workload W1, `develop@229b3e4`, vLLM `0.30.0`, `Qwen/Qwen2.5-7B-Instruct-AWQ` + `duckhoi/qwen2.5-7b-eagle3.1-speculator`, chế độ V1 eager tương đồng), kết quả tốc độ so với Vanilla như sau:

| Nhánh | Tải 1 (`c=1`) | Tải 8 (`c=8`) | Tải 32 (`c=32`) |
|---|---:|---:|---:|
| Static EAGLE `K=3` cố định | **1,77×** | **1,39×** | 0,59× |
| R6 Hardware-Adaptive (bản hiện hành) | 1,43× | 1,09× | 0,67× |
| R6 tắt ghi telemetry từng bước | 1,57× | 1,14× | **0,77×** |

Các xác minh thực nghiệm đã chốt từ đợt đo 23/11/2026:
1. **Tính đúng đắn (lossless) đã khép kín:** 100% vị trí lệch đầu ra so với Vanilla rơi đúng vào các vị trí hai token hòa điểm trong `bfloat16` (biên độ `0` hoặc `0.125` = `1 ULP`); token EAGLE chọn luôn là hạng 1 của mô hình đích (`44/44`); chất lượng W1 MC-OCR không suy giảm (`29/30` người bán, `30/30` thời gian, `30/30` tổng tiền, `27–28/30` địa chỉ); lặp lại khớp `27/27`.
2. **Stack Adaptive không phải nút thắt chính:** Khi khóa cứng `K = 3`, toàn bộ stack (`AdaptiveAsyncScheduler`, `LineageUniProcExecutor`, `AdaptiveBudgetPlanner`, telemetry) chỉ tăng thêm `~0 ms` (`c=1`), `+3 ms` (`c=8`) và `+5 ms` (`c=32`), trong đó khoảng một nửa ở tải cao đến từ I/O ghi file telemetry.
3. **Ở tải cao (`c=32`), R6 đã hoạt động đúng hướng:** Tự tắt nháp (`K=0`) ở 88% số bước, giúp nhanh hơn Static `K=3` tới **+30%** (`0,77×` so với `0,59×` khi tắt step telemetry).
4. **Khoảng trống duy nhất cần vá để thắng thuyết phục ở cả dải tải:** Ở tải thấp và trung bình (`c=1`, `c=8`), R6 vẫn thua Static `K=3` khoảng 12–20% vì hai nguyên nhân trực tiếp dưới đây.

---

## 2. Phân tích nguyên nhân gốc rễ và 2 bẫy toán học cần tránh

### 2.1. Nguyên nhân 1: Ước lượng $\alpha$ bị lệch xuống nghiêm trọng (`M-09`)

Trong `src/proposer/controller.py:106-115`, `BudgetController.update()` hiện tính:
$$\text{step\_alpha} = \frac{\text{num\_accepted}}{\text{num\_drafted}}$$

Mẫu số `num_drafted = K` đếm cả những token nháp nằm phía sau vị trí bị từ chối đầu tiên dù chúng không hề được kiểm chứng. Trong khi đó, hàm mục tiêu chọn `K` của R6 lại dùng công thức chuỗi hình học:
$$E[\tau(K, \alpha)] = 1 + \sum_{i=1}^{K} \alpha^i$$
với $\alpha$ là **xác suất chấp nhận có điều kiện trên từng vị trí thực sự được kiểm chứng** ($p$).

Dữ liệu thực đo trên A10 xác nhận: ở `K = 3` và `K = 4`, xác suất chấp nhận thật theo từng vị trí là `~0,61`, nhưng `accepted / drafted` chỉ trả `0,42` (ở `K = 3`) và `0,36` (ở `K = 4`). Kết hợp với ô nhiễu `C(1) < C(0)` trong profile A10G, $\alpha$ bị kéo xuống dưới `0,50`, khiến R6 chọn `K = 1` tới **49% số bước ở `c=1`** và **54% số bước ở `c=8`** (trong khi trên A10, `K = 1` tốn `20,2 ms` chỉ sinh `1,54` token, còn `K = 3` tốn `22,6 ms` sinh `2,20` token).

### 2.2. Hai bẫy toán học khi sửa công thức $\alpha$ (đã kiểm chứng bằng phân tích & mô phỏng)

Với mỗi bước có $K_t = \text{num\_drafted} > 0$ và $a_t = \text{num\_accepted} \in [0, K_t]$, số vị trí thực sự được kiểm chứng là:
$$n_t = a_t + \mathbb{I}(a_t < K_t) = \min(a_t + 1, K_t)$$

Ta có kết quả chính xác cho phân phối hình học cụt $a_t \sim \text{TruncGeom}(p, K_t)$:
$$\mathbb{E}[a_t] = \sum_{i=1}^{K_t} p^i = p \sum_{i=0}^{K_t - 1} p^i, \qquad \mathbb{E}[n_t] = \sum_{i=0}^{K_t - 1} p^i \implies \frac{\mathbb{E}[a_t]}{\mathbb{E}[n_t]} \equiv p \quad (\forall K_t \ge 1,\; \forall p \in [0, 1])$$

Tuy nhiên, mô phỏng 200 request thực tế (60 token/request, $p = 0,60$, độ trễ A10 đo thật) phát hiện **hai bẫy lớn nếu cài đặt đơn giản**:

1. **Bẫy 1 — Lấy EMA của tỉ số từng bước `a_t / n_t`:** Do $\mathbb{E}[a_t / n_t] \neq \mathbb{E}[a_t] / \mathbb{E}[n_t]$, ở $p = 0,60, K = 3$, kỳ vọng $\mathbb{E}[a_t / n_t]$ chỉ là **`0,432`** (so với `0,392` của code cũ). Tỉ lệ chọn `K = 1` ở `B = 1` chỉ giảm nhẹ từ `44,2%` xuống `38,7%` — **vẫn thất bại**.
2. **Bẫy 2 — Khởi tạo không có prior ($A_0 = 0, N_0 = 0$) cho request ngắn:** Với $p = 0,60$, có tới **40% request** gặp `accepted = 0` ngay bước đầu tiên ($a_1 = 0, n_1 = 1$). Nếu không có prior, $\hat{\alpha}_1 = 0 / 1 = 0,0$, lập tức kéo request rơi vào `K = 0` và bị giam trong `probe_interval = 20` bước (chiếm `27,7%` tổng số bước, làm speedup chỉ đạt `1,61×` thay vì `1,85×`).

**Giải pháp toán học chuẩn (Discounted-Count Geometric MLE với prior Beta yếu):**
Duy trì hai bộ tích lũy suy giảm theo hàm mũ cho tử số và mẫu số:
$$A_0 = w_0 \cdot \alpha_0, \qquad N_0 = w_0$$
$$A_t = \beta_{\text{mle}} A_{t-1} + a_t, \qquad N_t = \beta_{\text{mle}} N_{t-1} + n_t, \qquad \hat{\alpha}_{\text{mle}, t} = \operatorname{clip}\!\left(\frac{A_t}{N_t},\, 0,\, 1\right)$$

Với giá trị mặc định đã hiệu chỉnh bằng mô phỏng: `mle_prior_weight` $w_0 = 2.0$, `mle_beta` $\beta_{\text{mle}} = 0.90$, $\alpha_0 = \text{initial\_alpha}$:

| Phương pháp ước lượng | Kỳ vọng ở $K=3$ ($p=0,60$) | Mô phỏng 200 request 60 token (`B=1`) | Tỉ lệ `K=0` / `K=1` / `K=3..4` | Số bước về `K=0` ở prompt khó ($p=0$) |
|---|---:|---:|---|---:|
| Code cũ `EMA(a/K)` (`β=0,8`, không prior) | `0,392` | `1,53×` | `31,4%` / `30,4%` / `36,7%` | 1 bước |
| Bẫy 1: `EMA(a/n)` (`β=0,8`) | `0,432` | — | `K=1` vẫn `38,7%` (`B=1`), `44,1%` (`B=8`) | — |
| Bẫy 2: `EMA(a)/EMA(n)` không prior (`w0=0`) | `0,600` | `1,61×` | `27,7%` / `16,6%` / `54,4%` | 1 bước |
| **Đề xuất: Discounted MLE (`w0=2,0, β=0,90`)** | **`0,600`** | **`1,85×`** *(Static `K=3`: `1,90×`)* | **`0,0%` / `15,6%` / `84,4%`** | **14 bước (`B=1`), 2 bước (`B=16`), 0 bước (`B=32`)** |

### 2.3. Nguyên nhân 2: Nút thắt I/O ghi file đồng bộ từng dòng ở `_write_step`

Trong `src/server/budget_planner.py:176-183`:
```python
with open(self.step_telemetry_path, "a", encoding="utf-8") as fh:
    fh.write(line + "\n")
```
Mỗi bước decode tại `B = 32` ghi tối đa ~96 bản ghi (`32 planned_decision + 32 scheduled_budget + 32 verification_feedback`), tương đương ~96 cặp syscall `open() / write() / close()` đồng bộ ngay trên luồng scheduler chính của vLLM, gây tốn `+2..+4 ms` mỗi bước. Ngoài ra `LocalServingRunner.setup_environment()` hiện luôn bật `ADAPTIVE_STEP_TELEMETRY_PATH` và chưa có cờ tắt nhanh cho chế độ chạy benchmark hiệu năng.

---

## 3. Yêu cầu chức năng (Functional Requirements)

### 3.1. FR-01: Discounted-Count Geometric MLE trong `AdaptiveDraftController`
Trong `src/proposer/controller.py`:
1. Bổ sung 3 tham số cấu hình tùy chọn cho `AdaptiveDraftController.__init__`:
   - `mle_beta: float = 0.90` — hệ số suy giảm cho tích lũy đếm `(A_t, N_t)`. Validate không-`bool`, hữu hạn, `0.0 < mle_beta < 1.0`.
   - `mle_prior_weight: float = 2.0` — trọng số prior $w_0$ tính theo số vị trí giả định. Validate không-`bool`, hữu hạn, `>= 0.0`.
   - `alpha_estimator: str = "auto"` — nhận một trong `{"auto", "geometric_mle", "legacy_ratio"}`, sai ném `ValueError`.
2. Khởi tạo trạng thái MLE nội bộ:
   - `self._mle_accepted: float = self.mle_prior_weight * self.initial_alpha`
   - `self._mle_tested: float = self.mle_prior_weight`
   - `self._total_tested_positions: int = 0`
   - `self._total_accepted_tokens: int = 0`
3. Thêm helper xác định chế độ đang hoạt động `_resolved_alpha_estimator(self) -> str`:
   - Nếu `self.alpha_estimator == "auto"`: trả `"geometric_mle"` khi `bool(getattr(self.cost_profile, "matrix", None))` là `True`, ngược lại trả `"legacy_ratio"`.
   - Nếu tường minh `"geometric_mle"` hoặc `"legacy_ratio"`: trả đúng giá trị đó.
4. Cập nhật `AdaptiveDraftController.update(self, num_accepted: int, num_drafted: int) -> float`:
   - Vẫn giữ nguyên lời gọi `super().update(num_accepted=num_accepted, num_drafted=num_drafted)` để bảo toàn `self.alpha_ema` legacy (tương thích 100% với `experiments/m4_readiness.py:114` và bộ unit test cũ).
   - Khi `num_drafted > 0`:
     - `accepted = min(max(int(num_accepted), 0), int(num_drafted))`
     - `tested = accepted + (1 if accepted < num_drafted else 0)`
     - `self._mle_accepted = self.mle_beta * self._mle_accepted + accepted`
     - `self._mle_tested = self.mle_beta * self._mle_tested + tested`
     - `self._total_accepted_tokens += accepted`
     - `self._total_tested_positions += tested`
5. Bổ sung phương thức `get_geometric_alpha(self) -> float`:
   - Nếu `self._mle_tested <= 1e-12`: trả `self.initial_alpha`.
   - Ngược lại trả `min(max(self._mle_accepted / self._mle_tested, 0.0), 1.0)`.
6. Trong `select_budget(...)`:
   - Xác định `estimator = self._resolved_alpha_estimator()`.
   - Tính `decision_alpha = self.get_geometric_alpha() if estimator == "geometric_mle" else self.get_effective_alpha()`.
   - Dùng `decision_alpha` cho `apply_cost_driven_budget_rule` (và ghi nhận `alpha_estimator=estimator` vào `BudgetDecision`).
7. Trong `reset(self, initial_alpha=None)`:
   - Reset `self._mle_accepted = self.mle_prior_weight * self.initial_alpha`, `self._mle_tested = self.mle_prior_weight`, `self._total_accepted_tokens = 0`, `self._total_tested_positions = 0`.
8. Trong `get_stats(self)`:
   - Bổ sung `"geometric_alpha_mle"`, `"alpha_estimator"`, `"total_tested_positions"`.

### 3.2. FR-02: Mở rộng telemetry cho cả hai thước đo $\alpha$ (tương thích ngược)
Trong `src/proposer/controller.py` và `src/server/budget_planner.py`:
1. `BudgetDecision` thêm trường tùy chọn `alpha_estimator: Optional[str] = None`.
2. Bản ghi `planned_decision` trong `_decision_record()` giữ nguyên `"acceptance_rate_ema"` (raw `alpha_ema`), cập nhật `"effective_alpha"` phản ánh đúng $\alpha$ đầu vào của luật quyết định, và bổ sung:
   - `"geometric_alpha_mle"`: `finite(geometric_alpha)`
   - `"alpha_estimator"`: `getattr(decision, "alpha_estimator", None)`
3. Bản ghi `verification_feedback` trong `AdaptiveBudgetPlanner.record_verification()` giữ nguyên `"alpha_ema"` và `"alpha_ema_before"` (để không phá `m4_readiness.py` và `validate_async_lineage_runtime.py`), đồng thời bổ sung:
   - `"tested_draft_positions"`: `min(max(num_accepted, 0), num_drafted) + (1 if min(max(num_accepted, 0), num_drafted) < num_drafted else 0)`
   - `"geometric_alpha_mle"`: `finite(controller.get_geometric_alpha())` nếu controller có phương thức này.

### 3.3. FR-03: Ghi step telemetry theo lô với persistent file handle trong `AdaptiveBudgetPlanner`
Trong `src/server/budget_planner.py` và `src/server/adaptive_scheduler.py`:
1. Thêm tham số `step_telemetry_flush_every: int = 1` vào `AdaptiveBudgetPlanner.__init__` (validate không-`bool`, `isinstance(..., int)`, `>= 1`).
2. `AdaptiveBudgetPlanner.from_env()` đọc `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY` (mặc định `"1"` nếu biến môi trường không được đặt, giữ 100% tương thích với các unit test gọi `from_env` rồi đọc `steps.jsonl` ngay lập tức).
3. Thay vì `with open(...)` ở mỗi lần gọi `_write_step(record)`:
   - Giữ file handle mở sẵn `self._step_fh` (mở lười ở lần ghi đầu tiên) và danh sách đệm `self._step_buffer: List[str]`.
   - `_write_step(record)` serialize JSONL, đưa vào `self._step_buffer`, chỉ ghi một lần và `flush()` khi `len(self._step_buffer) >= self.step_telemetry_flush_every`.
   - Bổ sung `flush(self) -> None` (ghi sạch buffer ra `self._step_fh` và `flush()`, bắt `OSError` ghi log mà không làm sập engine) và `close(self) -> None` (`flush()` + đóng `self._step_fh`).
   - Gọi `self.flush()` trong `finish_request()` (đảm bảo khi một request kết thúc, toàn bộ log của request đó đã xuống đĩa) và gọi `self._planner.close()` trong `AdaptiveAsyncScheduler.shutdown()` (kèm `weakref.finalize` dự phòng).
4. Khi `self.step_telemetry_path is None` và `self.telemetry_path is None`:
   - Bỏ qua hoàn toàn việc tạo `_decision_record(...)` trong `plan_step()` và bỏ qua `record_scheduled_budget` write để triệt tiêu cả CPU overhead serialize JSON lẫn disk I/O.

### 3.4. FR-04: Bật batch flush mặc định `64` và thêm cờ `--no-step-telemetry` ở Runner & CLI
1. `LocalServingRunner.setup_environment()` (`src/flexsds/server/local_runner.py`):
   - Đọc `enable_step_telemetry = bool(serving_cfg.get("enable_step_telemetry", True))` và `step_telemetry_flush_every = serving_cfg.get("step_telemetry_flush_every", 64)` (validate `int` không-`bool` `>= 1`).
   - Khi `enable_adaptive=True` và `enable_step_telemetry=True`: đặt `ADAPTIVE_STEP_TELEMETRY_PATH` (giữ pass `test_cli_serving_readiness.py:123`) **và** `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY = str(step_telemetry_flush_every)`.
   - Khi `enable_adaptive=True` và `enable_step_telemetry=False`: `env.pop("ADAPTIVE_STEP_TELEMETRY_PATH", None)` và `env.pop("ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY", None)` để chạy ở chế độ zero-step-I/O cho benchmark/production.
   - Ở nhánh `enable_adaptive=False`: bổ sung `"ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY"` vào danh sách biến môi trường được dọn sạch tránh rò từ shell cha.
2. `ModalServingRunner.setup_environment()` (`src/flexsds/server/modal_runner.py`):
   - Thêm `"enable_step_telemetry"` và `"step_telemetry_flush_every"` vào tuple `keys` chuyển tiếp sang `FLEXSDS_SERVING_CONFIG`.
3. `src/flexsds/cli/main.py`:
   - Thêm cờ `--no-step-telemetry` (`action="store_true"`) cho `flexsds serve` để tắt ghi `serving_steps.jsonl` khi chạy benchmark hiệu năng hoặc production tốc độ cao.
   - Gắn `"enable_step_telemetry": not args.no_step_telemetry` vào `cfg["serving"]`.

### 3.5. FR-05: Ranh giới phạm vi (không làm trong spec này)
- **Không** sửa cơ chế đồng bộ KV cache của bước `K = 0` bên trong mã nguồn vLLM EAGLE3 (ghi nhận làm giới hạn hệ thống và hướng phát triển).
- **Không** sửa tay file `profile_sm_86_remote_a10g_fcda5486b12d.json` (giữ nguyên toàn vẹn `raw_sha256`).

---

## 4. Yêu cầu phi chức năng (Non-Functional Requirements)

- **NFR-01 (Không lệch thống kê ở trạng thái dừng):** Với chuỗi mẫu dài từ phân phối hình học cụt $p \in \{0.4, 0.6, 0.8\}$ ở mọi $K \in \{1, 2, 3, 4\}$, trung bình của `get_geometric_alpha()` phải nằm trong biên `±0.02` quanh $p$ thật (không tụt khi $K$ tăng).
- **NFR-02 (Hiệu năng I/O):** Khi `step_telemetry_flush_every = 64` và ghi 128 bản ghi liên tiếp, số lần gọi `open()` chỉ đúng **1 lần** và số lần ghi đĩa giảm tối thiểu **32 lần** so với bản cũ.
- **NFR-03 (Tương thích ngược tuyệt đối):** Toàn bộ bộ test hiện hành (236+ tests offline, bao gồm `test_budget_controller.py`, `test_budget_planner.py`, `test_m4_readiness.py`, `test_hardware_cost_driven_controller.py`, `test_matched_execution_contract.py`) phải tiếp tục pass 100%.
- **NFR-04 (Chiến lược Git):** Thực hiện trên một nhánh mới tách từ `develop@229b3e4` (ví dụ `fix/r6-geometric-mle-and-batched-telemetry`); không chỉnh sửa nhánh audit cũ.

---

## 5. Tiêu chí nghiệm thu (Acceptance Criteria)

1. **AC-01 — Công thức `tested_positions` và bất biến theo `K`:**
   - Kiểm chứng trên phân phối chính xác/Monte Carlo ở $p = 0,60$ cho cả `K = 1, 2, 3, 4`: `get_geometric_alpha()` hội tụ về `0,60 ± 0,02` tại mọi `K` (không tụt xuống `0,42` hay `0,36` như `alpha_ema` cũ).
2. **AC-02 — Không bị khóa `K=0` sau 1 bước trượt ở request mới, nhưng vẫn fallback ở prompt khó:**
   - Với `cost_profile` A10G thật, `initial_alpha = 0.80`, `mle_prior_weight = 2.0`, `mle_beta = 0.90`: sau bước 1 trượt (`accepted=0, drafted=4`), `get_geometric_alpha()` vẫn `> 0.50` (không rơi về `0.0` gây khóa `K=0` suốt 20 bước).
   - Khi liên tiếp trượt (`accepted=0`) trên prompt khó, `get_geometric_alpha()` giảm đơn điệu và rơi về `K = 0` (`B=16` sau ≤ 3 bước, `B=1` sau ≤ 15 bước).
3. **AC-03 — Cải thiện phân bố `K` trên profile A10G thật ở `B = 1` và `B = 8`:**
   - Mô phỏng 200 request 60 token ($p = 0,60$) trên profile A10G thật: tỉ lệ chọn `K = 3..4` tại `B = 1` đạt `>= 80%` (và `K = 0` dưới `2%`), tại `B = 8` tỉ lệ `K = 3` đạt `>= 75%`, tại `B = 32` tỉ lệ `K = 0` vẫn `>= 90%`.
4. **AC-04 — Ghi telemetry theo lô giảm số lần `open/write` và bảo toàn dữ liệu khi flush/close:**
   - Với `step_telemetry_flush_every = 64`, ghi 130 bước chỉ gọi `open()` **đúng 1 lần**, ghi ra đĩa 2 đợt trong lúc chạy và đủ 130 dòng JSON hợp lệ sau khi gọi `planner.close()` (hoặc `finish_request()`).
5. **AC-05 — Cờ `--no-step-telemetry` và `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY` thông suốt Local/Modal/CLI:**
   - Mặc định `LocalServingRunner.setup_environment(adaptive=True)` đặt `ADAPTIVE_STEP_TELEMETRY_PATH` và `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY="64"`.
   - Khi `enable_step_telemetry=False` (hoặc CLI `--no-step-telemetry`): `ADAPTIVE_STEP_TELEMETRY_PATH` bị loại bỏ hoàn toàn; `ModalServingRunner` chuyển tiếp đúng `enable_step_telemetry` và `step_telemetry_flush_every`.
6. **AC-06 — Không hồi quy (NFR-03):**
   - Toàn bộ test offline hiện hữu (bao gồm `test_budget_controller.py`, `test_budget_planner.py`, `test_m4_readiness.py`, `test_hardware_cost_driven_controller.py`, `test_matched_execution_contract.py`, `test_local_runner.py`, `test_modal_runner_cli.py`) pass 100%.
7. **AC-07 — Xác nhận GPU sau khi triển khai (do Dương/Khôi chạy trên 1 container A10):**
   - Chạy kiểm chứng nhanh `ablation` 3 nhánh (`Vanilla matched`, `Static K=3 matched`, `R6 mới`) trên cùng 1 container NVIDIA A10 (W1, `c = 1 / 8 / 32`): R6 thu hẹp rõ khoảng cách ở `c = 1, 8` (tỉ lệ `K = 1` giảm mạnh xuống `< 20%`) và tiếp tục vượt Static `K=3` ở `c = 32`.
