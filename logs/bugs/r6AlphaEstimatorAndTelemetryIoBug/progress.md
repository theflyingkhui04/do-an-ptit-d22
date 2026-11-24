# Báo cáo tiến độ: Ước lượng tỉ lệ chấp nhận Discounted Geometric MLE và ghi telemetry theo lô cho R6 Hardware-Adaptive

- **Đặc tả kỹ thuật**: [2026-11-24-r6-geometric-mle-alpha-and-batched-telemetry-spec.md](./2026-11-24-r6-geometric-mle-alpha-and-batched-telemetry-spec.md) (`SPEC-R6-ALPHA-IO-20261124`)
- **Kế hoạch triển khai**: [plan.md](./plan.md)
- **Nhánh Git**: `fix/r6-geometric-mle-and-batched-telemetry` (tách từ `develop@229b3e4`)
- **Ngày hoàn thành phần mã nguồn và kiểm thử CPU**: 2026-11-24
- **Trạng thái hiện tại**: Hoàn tất 100% mã nguồn, rà soát đối nghịch và kiểm thử hồi quy offline (`AC-01` đến `AC-06` đạt); chờ đo kiểm chứng GPU trên cùng một container NVIDIA A10 (`AC-07`).

---

## 1. Các commit đã thực hiện

| Commit | Nội dung | Phạm vi tệp |
|---|---|---|
| `9969401` | Thêm đặc tả kỹ thuật và kế hoạch triển khai `SPEC-R6-ALPHA-IO-20261124` | `logs/bugs/r6AlphaEstimatorAndTelemetryIoBug/` |
| `9aba6a1` | Triển khai Discounted-Count Geometric MLE (`mle_beta=0.90`, `mle_prior_weight=2.0`, `alpha_estimator="auto"`) trong `AdaptiveDraftController` | `src/proposer/controller.py`, `tests/test_r6_geometric_mle_and_telemetry_io.py` |
| `5c41e83` | Chuyển `_write_step` sang persistent file handle + ghi theo lô (`step_telemetry_flush_every`), tự `flush()` ở `finish_request()`, `close()` ở `AdaptiveAsyncScheduler.shutdown()`, bổ sung trường telemetry MLE | `src/server/budget_planner.py`, `src/server/adaptive_scheduler.py`, `tests/test_r6_geometric_mle_and_telemetry_io.py` |
| `cfc9b3a` | Đặt mặc định `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY="64"` trong `LocalServingRunner`, thêm cờ `--no-step-telemetry` trên `flexsds serve` và chuyển tiếp qua `ModalServingRunner` | `src/flexsds/server/local_runner.py`, `src/flexsds/server/modal_runner.py`, `src/flexsds/cli/main.py`, `tests/test_r6_geometric_mle_and_telemetry_io.py` |
| `cdce66c` | Rà soát đối nghịch: xuất bổ sung `geometric_alpha_mle`, `alpha_estimator`, `total_tested_positions` trong `AdaptiveProposerAdapter.get_telemetry()` (hữu ích cả khi tắt step telemetry); tách kiểm tra sớm `_validate_execution_options()` trong `ModalServingRunner.launch()` để không đòi hỏi `profile_path` giả khi truyền cấu hình tối giản; đồng bộ đường dẫn vào `_step_io_state` khi `flush()` | `src/proposer/adapter.py`, `src/flexsds/server/modal_runner.py`, `src/server/budget_planner.py`, `tests/test_r6_geometric_mle_and_telemetry_io.py` |

---

## 2. Kết quả kiểm thử và kiểm chứng toàn vẹn

### 2.1. Kiểm thử đơn vị và hồi quy offline (`AC-01` – `AC-06`)

1. **Bộ kiểm thử hồi quy chính (18 nhóm test, bao gồm 20 test mới trong `tests/test_r6_geometric_mle_and_telemetry_io.py`):**
   - Kết quả: **`256 passed, 88 skipped in 12.03s`** (`0 failed`).
   - Ghi chú: 88 bài test bị skip là các test phụ thuộc vLLM runtime / CUDA trong môi trường container CPU hiện tại.
2. **Các bài test offline trong `tests/test_cli_serving_readiness.py` (`-k "not cli_eagle_adaptive and not default_eagle_wizard"`):**
   - Kết quả: **`7 passed, 2 deselected in 0.82s`** (`0 failed`).
   - Ghi chú: 2 bài test bị loại trừ cần kết nối mạng ra HuggingFace (sandbox offline không có DNS ra ngoài).
3. **Tổng cộng đã xác minh offline:** **`263 passed, 0 failed`**.

### 2.2. Kiểm tra toàn vẹn profile A10G (`FR-05` / `NFR-04`)

- `git diff develop..HEAD -- src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`: **rỗng** (không có bất kỳ sửa đổi nào đối với artifact profile đã đo).
- `validate_measured_profile(...)` trên `profile_sm_86_remote_a10g_fcda5486b12d.json`: trả kết quả **`A10G_PROFILE_VALID_OK`** (`raw_sha256` nguyên vẹn).

### 2.3. Đối chiếu tiêu chí nghiệm thu

| Tiêu chí | Trạng thái | Bằng chứng xác minh |
|---|---|---|
| `AC-01` (`tested_positions` & bất biến theo `K`) | Đạt | 12 test tham số hóa `test_geometric_mle_is_unbiased_across_all_k` ($p \in \{0.4, 0.6, 0.8\} \times K \in \{1..4\}$) đạt sai số $\le 0.02$. |
| `AC-02` (Tránh khóa `K=0` ở bước 1 & vẫn fallback ở prompt khó) | Đạt | `test_geometric_mle_avoids_step1_deadlock_but_falls_back_on_hard_prompt` pass. |
| `AC-03` (Phân bố `K` trên profile A10G thật, 200 request × 60 token) | Đạt | `test_geometric_mle_keeps_k3_k4_dominant_at_low_batch_and_k0_at_high_batch` pass (`B=1`: `K=3..4 >= 80%`, `K=0 < 2%`; `B=8`: `K=3 >= 75%`; `B=32`: `K=0 >= 90%`). |
| `AC-04` (Ghi theo lô 1 lần `open()` & bảo toàn dữ liệu) | Đạt | `test_planner_batched_step_telemetry_opens_once_and_flushes_on_finish_and_close` & `test_planner_batched_io_error_does_not_crash_or_leak_buffer` pass. |
| `AC-05` (Cờ `--no-step-telemetry` và `flush=64` qua Local/Modal/CLI) | Đạt | `test_runner_and_modal_step_telemetry_batching_and_opt_out`, `test_cli_no_step_telemetry_flag_forwards_opt_out`, `test_adversarial_edge_cases_adapter_telemetry_env_validation_and_modal_launch` pass. |
| `AC-06` (Không hồi quy trên bộ test hiện hành) | Đạt | `263 passed, 0 failed`. |
| `AC-07` (Đo xác nhận trên cùng một container NVIDIA A10) | Chờ chạy GPU | Xem hướng dẫn chạy ở mục 3. |

---

## 3. Ranh giới khẳng định và hướng dẫn chạy kiểm chứng GPU cho Dương (`AC-07`)

> **Ranh giới trung thực:** Các bài kiểm thử trên CPU chỉ chứng minh tính đúng đắn toán học của bộ ước lượng Geometric MLE, việc giảm syscall I/O từ `130` xuống `1` lần `open()`, và tính tương thích ngược. Độ tăng tốc thực tế trên GPU **phải được xác nhận bằng phép đo ablation trên cùng một container NVIDIA A10** (tránh lệch loại GPU/driver A10 vs A10G như bài học ngày 23/11/2026).

### Lệnh chạy 4 nhánh đối chứng trong cùng một container NVIDIA A10 (W1, `c = 1 / 8 / 32`)

Trên nhánh `fix/r6-geometric-mle-and-batched-telemetry`:

1. **Nhánh 1 — Vanilla matched (`V1 eager`):**
   ```bash
   flexsds serve --backend local --vanilla --match-adaptive-runtime --no-prefix-caching \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ
   ```
2. **Nhánh 2 — Static EAGLE `K=3` matched (`V1 eager`):**
   ```bash
   flexsds serve --backend local --no-adaptive --match-adaptive-runtime --no-prefix-caching \
     --static-k 3 \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ \
     --draft duckhoi/qwen2.5-7b-eagle3.1-speculator
   ```
3. **Nhánh 3a — R6 mới (mặc định batched step telemetry `ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY=64`):**
   ```bash
   flexsds serve --backend local --no-prefix-caching \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ \
     --draft duckhoi/qwen2.5-7b-eagle3.1-speculator
   ```
4. **Nhánh 3b — R6 mới (tắt hẳn step telemetry bằng `--no-step-telemetry`):**
   ```bash
   flexsds serve --backend local --no-prefix-caching --no-step-telemetry \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ \
     --draft duckhoi/qwen2.5-7b-eagle3.1-speculator
   ```

### Chỉ số cần kiểm tra sau khi đo

- Trong `telemetry/serving_steps.jsonl` (nhánh 3a): xác nhận `alpha_estimator == "geometric_mle"`, `geometric_alpha_mle` dao động quanh `~0,60` (không còn bị kéo xuống `0,36–0,42`), và tỉ lệ bước chọn `K = 1` ở `c = 1`, `c = 8` giảm mạnh từ `~50%` xuống `< 20%`.
- Trong `telemetry/serving_trace.jsonl` (cả nhánh 3a và 3b): xác nhận có các trường `geometric_alpha_mle`, `alpha_estimator`, `total_tested_positions`.
- So sánh thông lượng nhánh 3a vs 3b để đo phần chênh lệch còn lại giữa ghi lô `flush=64` và tắt hoàn toàn step telemetry.
- Nếu cần làm sạch nốt ô nhiễu `C(1) < C(0)` trên đúng container A10, có thể chạy lại `flexsds profile --benchmark` sau khi hoàn tất vòng kiểm chứng nhanh này.
