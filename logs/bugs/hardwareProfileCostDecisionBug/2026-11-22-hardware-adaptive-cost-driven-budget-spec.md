# Đặc tả kỹ thuật: Kết nối ma trận chi phí phần cứng thực nghiệm vào bộ chọn ngân sách Hardware-Adaptive và xử lý bất thường không đơn điệu của profile đo đạc

- **Mã tài liệu**: SPEC-HWCOST-20261122
- **Ngày tạo**: 2026-11-22
- **Tác giả**: Đội ngũ phát triển FlexSDS (D22 PTIT)
- **Trạng thái**: Đã triển khai và kiểm thử xong trên CPU (Implemented & CPU-verified — chờ kiểm chứng GPU)
- **Liên kết kiểm toán**: Audit R6 (`Chưa đạt`) trên nhánh `audit/flexsds-v023-hardwareadaptive-20261120` (`logs/daijestic/flexsds-audit/HardwareAdaptive_v0.2.3_audit_interim_2026-11-20.md`)
- **Phạm vi tác động**:
  - `src/proposer/controller.py` (`BudgetDecision`, `EmpiricalCostProfile`, `AdaptiveDraftController`)
  - `src/server/budget_planner.py` (`AdaptiveBudgetPlanner.from_env`, `_decision_record`)
  - `src/flexsds/core/cost_model.py` (`validate_measured_profile`, chẩn đoán đơn điệu)
  - `src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json` (mốc dữ liệu đối chứng, giữ nguyên không sửa tay)
  - `tests/test_budget_controller.py`, `tests/test_budget_planner.py`, và bổ sung `tests/test_hardware_cost_driven_controller.py`

---

## 1. Bối cảnh và mục đích

Trong đợt kiểm toán lớp Hardware-Adaptive tại mốc `develop@8006205` (gói `v0.2.3`), hạng mục **R6 — Hardware profile** nhận kết luận **`Chưa đạt`** — đây là lỗ hổng nghiêm trọng nhất ảnh hưởng trực tiếp tới luận điểm khoa học cốt lõi của đồ án.

Cụ thể, hệ thống hiện đang tồn tại **hai vấn đề gắn chặt với nhau**:

1. **Đứt gãy giữa hồ sơ phần cứng và quyết định chọn `K` (R6):** `AdaptiveBudgetPlanner.from_env()` đã đọc biến môi trường `FLEXSDS_PROFILE_PATH`, xác thực chữ ký bằng chứng `raw_sha256`, dựng đối tượng `EmpiricalCostProfile(matrix=matrix)` từ `cost_matrix_ms`, rồi truyền vào `AdaptiveDraftController`. Tuy nhiên, phương thức `AdaptiveDraftController.select_budget()` lại bỏ qua hoàn toàn `self.cost_profile` lẫn `batch_size`, chỉ sử dụng 3 ngưỡng cứng theo tỷ lệ chấp nhận (`0.65 / 0.80 / 0.90`).
2. **Bất thường không đơn điệu trong ma trận chi phí thực nghiệm A10G (`profile_sm_86_remote_a10g_fcda5486b12d.json`, dòng 42–61):** Ngay trong hồ sơ đo thật trên NVIDIA A10 (`sm_86`), các giá trị độ trễ tại batch nhỏ `B ∈ {1, 2, 4}` xuất hiện hiện tượng $C(1, B) < C(0, B)$ (và tại `B ∈ {2, 4}` cả $C(2, B) < C(0, B)$). Nếu nối trực tiếp công thức cực đại hóa hiệu suất ngây thơ $\arg\max_k \frac{E[\tau(k, \alpha)]}{C(k, B)}$ vào ma trận này mà không có cơ chế phòng vệ, bộ điều khiển sẽ **chọn `K = 1` ngay cả khi $\alpha = 0$**, phá hỏng hoàn toàn cơ chế fallback về `K = 0`.

Mục đích của tài liệu đặc tả này là:
1. Kết nối thực sự ma trận chi phí đo đạc `EmpiricalCostProfile` vào đường quyết định chính thức của `AdaptiveDraftController.select_budget()` để `K` thực sự thay đổi theo cả **tỷ lệ chấp nhận $\alpha_{\text{eff}}$**, **kích thước lô `batch_size`**, và **hồ sơ GPU thực nghiệm**.
2. Xây dựng công thức chuẩn hóa baseline phòng vệ tính không đơn điệu ở tầng quyết định mà **không sửa đổi hay làm sai lệch dữ liệu thực nghiệm gốc** trong file JSON profile.
3. Mở rộng telemetry quyết định để mỗi bước có thể truy vết ngược chính xác profile nào và ô chi phí `C(k, B)` nào đã sinh ra quyết định `K`.

---

## 2. Phân tích nguyên nhân gốc rễ

### 2.1. Ngắt kết nối trong `AdaptiveDraftController.select_budget()`

Chuỗi truyền dữ liệu hiện tại bị đứt ngay ở bước cuối cùng:

```text
FLEXSDS_PROFILE_PATH
  → src/server/budget_planner.py:238-251 (validate_measured_profile → EmpiricalCostProfile)
  → src/proposer/adapter.py:49-52 (truyền controller_kwargs vào AdaptiveDraftController)
  → src/proposer/controller.py:234-244 (lưu vào self.cost_profile)
  ✗ src/proposer/controller.py:415-451 (select_budget() gọi apply_budget_rule(alpha) — KHÔNG đọc self.cost_profile và KHÔNG dùng batch_size)
```

Trong khi đó, lớp cha `BudgetController.select_budget()` (`src/proposer/controller.py:143-175`) vốn đã cài đặt vòng lặp tìm `best_k` tối đa hóa $\frac{E[\tau(k, \alpha)]}{C(k, B)}$. Khi `AdaptiveDraftController` kế thừa để bổ sung unbiased EMA, memory pressure guard và periodic probing, nó đã ghi đè hoàn toàn `select_budget()` bằng luật ngưỡng cứng (`apply_budget_rule`):
- `alpha < 0.65` → `K = 0` (hoặc probe `K = 2` mỗi 20 bước)
- `0.65 <= alpha < 0.80` → `K = 2`
- `0.80 <= alpha < 0.90` → `K = 3`
- `alpha >= 0.90` → `K = 4`

Hệ quả là:
- Tham số `batch_size` trong `select_budget()` trở thành tham số thừa (unused).
- `candidate_budgets` từ profile (ví dụ `[0, 1, 2, 3, 4]` có `K = 1`) không được đánh giá; `K = 1` không bao giờ được chọn.
- Trên A10G tại `B = 32` và $\alpha = 0.70$, luật ngưỡng cứng chọn `K = 2` (`55.79 ms`, $E[\tau] = 2.19$, thông lượng kỳ vọng `0.0393 tok/ms`), chậm hơn **~26%** so với `K = 0` (`18.72 ms`, `0.0534 tok/ms`).

### 2.2. Bất thường không đơn điệu trong `cost_matrix_ms` của profile A10G (dòng 42–61)

Trích xuất từ `src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json` (mỗi ô là trung vị của 50 lần lặp sau 5 lần warmup trên NVIDIA A10, `Qwen/Qwen2.5-7B-Instruct-AWQ` + `duckhoi/qwen2.5-7b-eagle3.1-speculator`, vLLM `0.30.0`, `enforce_eager=True`):

| `B` | `K=0` (`target` + `draft` = tổng ms) | `K=1` (`target` + `draft` = tổng ms) | `K=2` (`target` + `draft` = tổng ms) | `K=3` (tổng ms) | `K=4` (tổng ms) | Quan sát bất thường |
|---:|---:|---:|---:|---:|---:|---|
| `1` | `15.42 + 2.02 = 17.43` | `13.08 + 2.14 = 15.20` | `13.92 + 3.76 = 17.69` | `18.47` | `19.69` | `C(1,1) < C(0,1)` (-2.23 ms) |
| `2` | `15.86 + 2.04 = 17.91` | `13.19 + 2.14 = 15.34` | `13.91 + 3.80 = 17.72` | `18.14` | `20.09` | `C(1,2) < C(2,2) < C(0,2)` |
| `4` | `16.22 + 2.10 = 18.32` | `13.19 + 2.20 = 15.39` | `13.95 + 3.87 = 17.82` | `19.05` | `21.08` | `C(1,4) < C(2,4) < C(0,4)` |
| `8` | `16.16 + 2.09 = 18.24` | `16.14 + 2.38 = 18.56` | `17.76 + 4.02 = 21.76` | `23.24` | `28.09` | Đơn điệu tăng theo `K` |
| `16` | `16.14 + 2.09 = 18.23` | `22.70 + 2.53 = 25.25` | `25.68 + 4.31 = 29.99` | `35.28` | `45.44` | Đơn điệu tăng mạnh theo `K` |
| `32` | `16.63 + 2.08 = 18.72` | `40.42 + 3.01 = 43.46` | `50.85 + 4.86 = 55.79` | `68.06` | `71.30` | Đơn điệu tăng rất mạnh (`K=4` gấp 3.8x `K=0`) |

Kiểm tra phân bố 50 mẫu (`p25–p75`) trong `evidence.groups` xác nhận đây **không phải ngoại lai ngẫu nhiên**:
- Tại `K=0, B=1`: `target` có `min=14.33`, `p25=15.04`, `med=15.42`, `p75=15.70 ms`.
- Tại `K=1, B=1`: `target` có `min=12.70`, `p25=13.00`, `med=13.08`, `p75=13.64 ms`.

Nguyên nhân kỹ thuật của chênh lệch `~2.2–2.9 ms` này gồm hai yếu tố:
1. **Khác biệt đường thực thi kernel eager trong vLLM `0.30.0` + `awq_marlin`:** Với `K = 0`, mỗi request chỉ có `query = 1` token (nhánh single-token decode giữ gần như phẳng `15.4–16.6 ms` từ `B=1` đến `B=32`); với `K >= 1`, `query = K + 1 >= 2` token đi qua nhánh multi-token verify/GEMM khác (nhanh hơn nhẹ ~2 ms ở `B <= 4` nhưng tăng rất mạnh khi `B >= 16`). Đồng thời `K = 0` vẫn phải chạy `drafter.propose` để đồng bộ KV cache của EAGLE3 (`~2.02–2.10 ms`), gần bằng chi phí draft của `K = 1` (`2.14–2.20 ms`).
2. **Cách ly tiến trình theo thứ tự cố định trong `calibration.py`:** `modal_calibration_app.py` chạy tuần tự `[run_local_group(settings, k, batches) for k in [0, 1, 2, 3, 4]]` trên cùng container GPU; `K = 0` luôn là tiến trình con đầu tiên khởi chạy nên có thể chịu thêm khác biệt trạng thái xung nhịp GPU / cấp phát bộ nhớ ban đầu so với các tiến trình `K = 1..4` chạy ngay sau.

**Rủi ro khi nối công thức argmax ngây thơ:**
Nếu tính trực tiếp $S_{\text{naive}}(k) = \frac{E[\tau(k, \alpha)] / C(k, B)}{1 / C(0, B)}$ trên dữ liệu thật ở trên:
- Tại `B = 1`, $\alpha = 0.00$: `eff(K=1) = 1.0 / 15.20 = 0.0658 > eff(K=0) = 1.0 / 17.43 = 0.0574` → **chọn `K = 1` dù tỷ lệ chấp nhận bằng 0%**.
- Tại `B = 2` và `B = 4`, $\alpha = 0.00$: cả `K = 1` và `K = 2` đều thắng `K = 0` → **không bao giờ fallback về `K = 0` ở batch nhỏ**.

### 2.3. Khoảng trống telemetry truy vết quyết định chi phí

Trong `src/server/budget_planner.py:270-303`, hàm `_decision_record()` hiện cố định:
- `"hardware_profile": None`
- `"dynamic_alpha_star": None`
- `"theoretical_speedup": None`

Do đó, không thể đối chiếu từ log `serving_steps.jsonl` xem tại mỗi bước thực thi, bộ điều khiển đã đọc `C(k, B)` nào, tính speedup kỳ vọng bao nhiêu, và có thực sự dùng profile hay không.

---

## 3. Phân tích các hướng tiếp cận kỹ thuật

### Phương án 1: Ghi đè/làm phẳng `cost_matrix_ms` ngay khi tạo hoặc lưu file profile JSON
- **Cơ chế:** Ép `cost_matrix_ms[f"k{k}_b{b}"] = max(measured, cost_matrix_ms[f"k0_b{b}"])` trong `build_profile()`.
- **Nhược điểm (bị loại):**
  - Phá vỡ hợp đồng kiểm chứng trung thực của `validate_measured_profile()` (vốn yêu cầu tái dựng chính xác `cost_matrix_ms` từ mẫu đo thô trong `evidence.groups` và khớp `raw_sha256`).
  - Che giấu hiện tượng đo đạc thật của phần cứng/vLLM thay vì ghi nhận minh bạch.

### Phương án 2: Giữ nguyên dữ liệu đo thực nghiệm + chuẩn hóa tham chiếu $C_{\text{ref}}(B)$ ngay trong luật quyết định của `AdaptiveDraftController` (Đề xuất)
- **Cơ chế:**
  - Giữ nguyên 100% file JSON profile và `validate_measured_profile()`.
  - Trong `AdaptiveDraftController`, khi có `cost_profile` thực nghiệm (có ma trận chi phí hợp lệ), dùng hàm mục tiêu chuẩn hóa theo ngưỡng tham chiếu cận dưới của cùng batch $B$:
    $$C_{\text{ref}}(B) = \min_{k \in \mathcal{K}} C(k, B)$$
    $$S(k, \alpha, B) = \frac{E[\tau(k, \alpha)] / C(k, B)}{1.0 / C_{\text{ref}}(B)} = E[\tau(k, \alpha)] \cdot \frac{C_{\text{ref}}(B)}{C(k, B)}$$
  - Chỉ chấp nhận ứng viên $k > 0$ khi $S(k, \alpha, B) \ge S_{\min}$ (với biên an toàn mặc định `min_speedup_ratio = 1.05`), và chọn:
    $$k^* = \arg\max_{k \in \mathcal{K} \setminus \{0\},\; S(k, \alpha, B) \ge S_{\min}} S(k, \alpha, B)$$
    Nếu không có $k > 0$ nào đạt $S(k, \alpha, B) \ge S_{\min}$, quyết định `K = 0` (cost-model fallback) và kích hoạt bộ đếm periodic probing (`probe_interval`, `probe_k`).
- **Tính chất toán học then chốt:**
  1. **Tương đương tuyệt đối với công thức chuẩn khi ma trận đơn điệu:** Khi $C(0, B) \le C(1, B) \le \dots \le C(K_{\max}, B)$, ta có $C_{\text{ref}}(B) \equiv C(0, B)$, nên $S(k, \alpha, B) = \frac{E[\tau(k, \alpha)] / C(k, B)}{1 / C(0, B)}$ đúng 100% công thức lý thuyết.
  2. **Chống bẫy không đơn điệu tại $\alpha = 0$:** Vì $C_{\text{ref}}(B) \le C(k, B)$ với mọi $k$, tại $\alpha = 0$ ta luôn có $E[\tau(k, 0)] = 1 \implies S(k, 0, B) \le 1.0 < S_{\min}$. Do đó không ứng viên $k > 0$ nào có thể thắng `K = 0` chỉ nhờ sai số $C(1, B) < C(0, B)$.
  3. **Thích ứng rõ rệt theo cả `batch_size` lẫn GPU:** Mô phỏng trực tiếp trên `profile_sm_86_remote_a10g_fcda5486b12d.json` với `min_speedup_ratio = 1.05`:

| Batch `B` | $\alpha = 0.0$ | $\alpha = 0.2$ | $\alpha = 0.4$ | $\alpha = 0.5$ | $\alpha = 0.6$ | $\alpha = 0.7$ | $\alpha = 0.8$ | $\alpha = 0.9$ |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1` | `K=0` | `K=1` | `K=1` | `K=3` | `K=3` | `K=4` | `K=4` | `K=4` |
| `2` | `K=0` | `K=1` | `K=1` | `K=3` | `K=3` | `K=3` | `K=4` | `K=4` |
| `4` | `K=0` | `K=1` | `K=1` | `K=3` | `K=3` | `K=3` | `K=4` | `K=4` |
| `8` | `K=0` | `K=1` | `K=1` | `K=1` | `K=3` | `K=3` | `K=3` | `K=3` |
| `16` | `K=0` | `K=0` | `K=0` | `K=1` | `K=2` | `K=2` | `K=3` | `K=3` |
| `32` | `K=0` | `K=0` | `K=0` | `K=0` | `K=0` | `K=0` | `K=0` | `K=4` |

  Nhìn vào cột $\alpha = 0.70$: cùng một tỷ lệ chấp nhận, bộ điều khiển tự động co giãn **`K=4` (`B=1`) → `K=3` (`B=2..8`) → `K=2` (`B=16`) → `K=0` (`B=32`)**, triệt tiêu hoàn toàn hiện tượng tăng tốc âm `-26%` ở `B=32` của luật ngưỡng cứng hiện tại.

**Lựa chọn:** Áp dụng **Phương án 2**, kết hợp thêm kiểm tra chẩn đoán không đơn điệu (không chặn cứng nếu chỉ lệch nhẹ ở batch nhỏ, nhưng cảnh báo rõ trong log và telemetry) và quy trình đo kiểm chuẩn hóa hơn cho các lần hiệu chuẩn mới.

---

## 4. Yêu cầu chức năng (Functional Requirements)

### 4.1. FR-01: Phân tách rõ hai chế độ quyết định và ưu tiên tuyệt đối ma trận thực nghiệm
- Trong `AdaptiveDraftController.select_budget(batch_size, free_vram_mb, kv_cache_usage_pct)`:
  1. **Tầng 1 (Ưu tiên cao nhất — không đổi):** Chạy `apply_memory_pressure_rule(free_vram_mb, kv_cache_usage_pct)`. Nếu kích hoạt hoặc đang trong cooldown/hold → trả `K = 0` với `status = "memory_guard"`.
  2. **Tầng 2 (Quyết định theo hồ sơ phần cứng):** Nếu `self.cost_profile` chứa ma trận chi phí hợp lệ (`bool(getattr(self.cost_profile, "matrix", None))`):
     - Bắt buộc gọi `apply_cost_driven_budget_rule(alpha=effective_alpha, batch_size=batch_size)` dùng `self.cost_profile.get_cost(k, batch_size)` trên tập `self.candidate_budgets`.
     - Tuyệt đối không rơi lại về ngưỡng cứng `0.65 / 0.80 / 0.90` khi đã có profile thực nghiệm.
  3. **Tầng 3 (Tương thích ngược chỉ cho offline/unit test không nạp profile):** Chỉ khi `self.cost_profile.matrix` rỗng (khởi tạo mặc định `AdaptiveDraftController()` không truyền profile) mới dùng `apply_budget_rule(alpha=effective_alpha)` cũ. Trong production có `ADAPTIVE_STRICT=1` và `FLEXSDS_PROFILE_PATH`, `AdaptiveBudgetPlanner.from_env()` phải đảm bảo `cost_profile` là `EmpiricalCostProfile` có ma trận không rỗng.

### 4.2. FR-02: Công thức chọn `K` theo chi phí và phòng vệ không đơn điệu
- Với mỗi lần gọi `apply_cost_driven_budget_rule(alpha, batch_size)`:
  - Lấy tập ứng viên $K_{\text{all}} = \text{sorted}(\text{set}(self.\text{candidate\_budgets}) \cup \{0\})$.
  - Tra cứu chi phí `costs[k] = self.cost_profile.get_cost(k, batch_size)` cho mọi $k \in K_{\text{all}}$. Kiểm tra mọi `costs[k]` phải hữu hạn và `> 0`.
  - Tính $C_0 = \text{costs}[0]$ và $C_{\text{ref}} = \min_{k \in K_{\text{all}}} \text{costs}[k]$.
  - Ghi nhận cờ `non_monotonic_cost = (C_ref < C_0)`.
  - Với mỗi ứng viên $k \in self.\text{candidate\_budgets}$ với $k > 0$:
    - Tính $E[\tau(k, \alpha)] = 1 + \sum_{i=1}^{k} \alpha^i$.
    - Tính $S(k) = E[\tau(k, \alpha)] \cdot \frac{C_{\text{ref}}}{\text{costs}[k]}$.
    - Chọn $k^*$ tối đa hóa $S(k)$ trong nhóm thỏa $S(k) \ge \text{min\_speedup\_ratio}$ (mặc định `1.05`, có thể cấu hình qua `controller_kwargs`). Khi hai giá trị $S(k)$ chênh lệch dưới `1e-6`, ưu tiên `k` nhỏ hơn để giảm rủi ro dư thừa tính toán.
  - Tính ngưỡng hòa vốn động $\alpha^*(B)$ của ứng viên tốt nhất hoặc của cấu hình đang xét (bằng giải phương trình $E[\tau(k, \alpha)] = S_{\min} \cdot \frac{C(k, B)}{C_{\text{ref}}(B)}$ trên $[0, 1]$ bằng chia đôi bisection) để ghi vào telemetry `dynamic_alpha_star`.
  - Nếu có $k^* > 0$:
    - Reset `self.consecutive_fallback_steps = 0`.
    - Trả quyết định `k = k*`, `speedup = S(k*)`, `status = "ok"`, kèm lý do chứa `k`, `batch_size`, `alpha`, `c_k_ms`, `c_ref_ms`, `speedup`.
  - Nếu không có $k > 0$ nào đạt `min_speedup_ratio`:
    - Tăng `self.consecutive_fallback_steps += 1`.
    - Nếu `self.probe_interval > 0` và `self.consecutive_fallback_steps >= self.probe_interval`:
      - Reset `self.consecutive_fallback_steps = 0`.
      - Chọn `probe_k` hợp lệ nằm trong `candidate_budgets` (nếu `self.probe_k` không nằm trong `candidate_budgets`, chọn ứng viên dương gần nhất $\le \text{probe\_k}$, fallback về `min(k > 0)`).
      - Trả quyết định `k = effective_probe_k`, `status = "probing"`.
    - Ngược lại trả `k = 0`, `speedup = 1.0`, `status = "fallback_gate"`.

### 4.3. FR-03: Mở rộng `BudgetDecision` và telemetry truy vết R6 + R8
- Mở rộng `BudgetDecision` (giữ giá trị mặc định để không làm vỡ caller cũ):
  - `speedup: Optional[float] = None`
  - `alpha_star: Optional[float] = None`
  - `status: str = "ok"`
  - `decision_policy: str = "threshold"` (`"cost_model"` khi dùng `cost_profile`, `"threshold"` khi ở chế độ legacy, `"memory_guard"` khi chặn bộ nhớ)
  - `cost_k_ms: Optional[float] = None`
  - `cost_baseline_ms: Optional[float] = None` (giá trị $C_0(B)$ đo gốc)
  - `cost_ref_ms: Optional[float] = None` (giá trị $C_{\text{ref}}(B)$ dùng chuẩn hóa)
  - `non_monotonic_cost: bool = False`
- Trong `AdaptiveBudgetPlanner`:
  - Lưu siêu dữ liệu định danh profile khi `from_env()` nạp `FLEXSDS_PROFILE_PATH`: `profile_name`, `profile_sha256`, `profile_device`, `profile_sm_architecture`.
  - Trong `_decision_record()` (`src/server/budget_planner.py:270-303`), điền đầy đủ:
    - `"hardware_profile"` (không còn `None` khi có profile)
    - `"dynamic_alpha_star"` (từ `decision.alpha_star`)
    - `"theoretical_speedup"` (từ `decision.speedup`)
    - `"decision_policy"`, `"cost_k_ms"`, `"cost_baseline_ms"`, `"cost_ref_ms"`, `"non_monotonic_cost"`.

### 4.4. FR-04: Chẩn đoán tính không đơn điệu khi nạp profile và cảnh báo kiểm toán
- Trong `src/flexsds/core/cost_model.py`, bổ sung hàm thuần túy `inspect_cost_matrix_monotonicity(cost_matrix_ms)` trả về danh sách các cặp `(b, k, c_k, c_0)` nơi $C(k, b) < C(0, b)$ hoặc $C(k, b) < C(k - 1, b)$.
- Trong `AdaptiveBudgetPlanner.from_env()`:
  - **Không** chặn lỗi hay sửa đổi dữ liệu profile hợp lệ khi phát hiện `C(k, b) < C(0, b)` (để vẫn dùng được profile đo thật như A10G hiện tại mà không mất chứng cứ).
  - Ghi log mức `WARNING` liệt kê rõ các ô không đơn điệu và xác nhận cơ chế chuẩn hóa $C_{\text{ref}}(B)$ đang được kích hoạt.

### 4.5. FR-05: Cải thiện tính lặp lại của quy trình hiệu chuẩn (cho các lần đo mới)
- Ghi nhận rõ hai khuyến nghị cho `src/flexsds/core/calibration.py` (có thể triển khai ở PR kế tiếp để không ảnh hưởng mốc đối chứng hiện tại):
  1. Trước khi đo `K = 0`, chạy một vòng GPU warmup ngắn (ví dụ vài bước forward nháp bị loại bỏ khỏi mẫu) để ổn định xung nhịp GPU trước khi bắt đầu tiến trình đo đầu tiên.
  2. Ghi thêm `gpu_clock_mhz` / `process_start_utc` vào metadata mỗi `group` trong `evidence.groups` để phân biệt rạch ròi giữa chênh lệch xung nhịp GPU và chênh lệch nhánh kernel vLLM.

---

## 5. Yêu cầu phi chức năng và ràng buộc tương thích

- **NFR-01 (Trung thực dữ liệu thực nghiệm):** Không sửa tay file `profile_sm_86_remote_a10g_fcda5486b12d.json`; mọi bản ghi `validate_measured_profile()` phải tiếp tục vượt qua kiểm tra `raw_sha256`.
- **NFR-02 (Độ trễ quyết định):** Hàm `select_budget()` chạy hoàn toàn trên CPU trong `< 0.05 ms` mỗi bước cho mỗi request (tối đa 5 ứng viên `K`, không cấp phát tensor, không gọi CUDA).
- **NFR-03 (Tương thích ngược unit test):** Toàn bộ test hiện hữu khởi tạo `AdaptiveDraftController()` không truyền `cost_profile` tiếp tục pass mà không cần sửa hành vi mặc định legacy.
- **NFR-04 (Chiến lược nhánh Git):** Không sửa code trực tiếp lên nhánh audit đóng băng `audit/flexsds-v023-hardwareadaptive-20261120`. Mọi thay đổi theo đặc tả này phải được thực hiện trên một nhánh phát triển mới tách từ `develop` / `main`.

---

## 6. Kế hoạch kiểm thử và tiêu chí nghiệm thu (Acceptance Criteria)

### 6.1. Unit tests bắt buộc (`tests/test_hardware_cost_driven_controller.py`)
1. **AC-01 — Nhạy theo `batch_size` trên profile A10G thật:** Nạp trực tiếp `profile_sm_86_remote_a10g_fcda5486b12d.json` vào `EmpiricalCostProfile` → cố định $\alpha = 0.70$, kiểm chứng `select_budget(batch_size=1).k == 4`, `select_budget(batch_size=8).k == 3`, `select_budget(batch_size=16).k == 2`, `select_budget(batch_size=32).k == 0`.
2. **AC-02 — Nhạy theo phần cứng (hai profile khác nhau):** Cùng $\alpha = 0.70$ và cùng `batch_size = 8`, đối chiếu hai `EmpiricalCostProfile` khác nhau (một GPU verify rẻ như A10G, một GPU mô phỏng draft/verify đắt gấp 3 lần) → xác nhận chọn ra hai giá trị `K` khác nhau.
3. **AC-03 — Chống bẫy không đơn điệu tại $\alpha = 0$:** Trên chính `profile_sm_86_remote_a10g_fcda5486b12d.json` (nơi `k1_b1 = 15.20 < k0_b1 = 17.43` và `k2_b4 = 17.82 < k0_b4 = 18.32`), kiểm chứng với $\alpha = 0.0$ tại **mọi** `B ∈ {1, 2, 4, 8, 16, 32}`, `select_budget()` đều trả `K = 0` và gắn `non_monotonic_cost = True` cho `B ∈ {1, 2, 4}`.
4. **AC-04 — Bảo toàn thứ tự ưu tiên Memory Guard và Probing:**
   - Khi `kv_cache_usage_pct = 95.0` hoặc `free_vram_mb = 500.0`, bất kể $\alpha = 0.95$ và `B = 1`, vẫn ép `K = 0` (`status = "memory_guard"`), giữ cooldown 3 bước và chỉ thoát khi `KV < 80%`.
   - Khi rơi vào `K = 0` do cost model (ví dụ `B = 32, alpha = 0.50` hoặc `B = 1, alpha = 0.0`), sau đúng `probe_interval` bước phải phát probe hợp lệ (`status = "probing"`, `k in candidate_budgets`).
5. **AC-05 — Truy vết telemetry end-to-end trong `AdaptiveBudgetPlanner`:** Khi khởi tạo qua `AdaptiveBudgetPlanner.from_env()` với `FLEXSDS_PROFILE_PATH` trỏ tới profile A10G thật, bản ghi `planned_decision` trong `serving_steps.jsonl` phải chứa `hardware_profile != None`, `decision_policy == "cost_model"`, `theoretical_speedup`, `dynamic_alpha_star`, `cost_k_ms`, `cost_baseline_ms`, `cost_ref_ms`.
6. **AC-06 — Không hồi quy (regression):** Toàn bộ bộ test hiện tại (`tests/test_budget_controller.py`, `tests/test_budget_planner.py`, `tests/test_memory_pressure_rule.py`, `tests/test_serving_profile_guard.py`) pass 100%.

### 6.2. Kiểm chứng GPU sau khi chốt code (đóng R6 trên runtime thật)
- Chạy cùng 1 prompt trên Modal A10G ở hai mức effective batch khác nhau (`B = 1` và `B >= 8` hoặc `B >= 16`) với cùng profile A10G.
- Đối chiếu `serving_steps.jsonl`: xác nhận `decision_policy == "cost_model"`, giá trị `cost_k_ms` khớp với `profile_sm_86_remote_a10g_fcda5486b12d.json`, và `K` được chọn thực sự thay đổi theo `batch_size` + `alpha_eff`.
- Lưu ý tách riêng kiểm chứng **R6 (quyết định `K` có dùng chi phí phần cứng)** với benchmark hiệu năng ON/OFF (đòi hỏi mốc Static matched-config cùng eager/V1 runner).
