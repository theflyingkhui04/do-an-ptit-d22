# Báo cáo tiến độ: Ước lượng tỉ lệ chấp nhận Discounted Geometric MLE và ghi telemetry theo lô cho R6 Hardware-Adaptive

- **Đặc tả kỹ thuật**: [2026-11-24-r6-geometric-mle-alpha-and-batched-telemetry-spec.md](./2026-11-24-r6-geometric-mle-alpha-and-batched-telemetry-spec.md) (`SPEC-R6-ALPHA-IO-20261124`)
- **Kế hoạch triển khai**: [plan.md](./plan.md)
- **Nhánh Git**: `fix/r6-geometric-mle-and-batched-telemetry` (tách từ `develop@229b3e4`)
- **Ngày hoàn thành phần mã nguồn và kiểm thử CPU**: 2026-11-24
- **Trạng thái hiện tại**: Hoàn tất mã nguồn, rà soát đối nghịch và kiểm thử hồi quy offline (`AC-01` đến `AC-06` đạt). Đã đo `AC-07` trên một container NVIDIA A10 ngày 24/11/2026: **đạt phần lớn**. Riêng tiêu chí "K = 1 dưới 20%" chưa đạt ở `c = 8` (28%). Xem mục 4.

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

---

## 4. Kết quả đo `AC-07` (24/11/2026, Dương chạy)

### 4.1. Cấu hình đo

**Phần cứng và phần mềm**
- Một container Modal duy nhất, GPU "NVIDIA A10" (driver 610.57.04, uuid `GPU-55757f00…`), vLLM 0.30.0.
- Mọi nhánh chạy lần lượt trên cùng GPU này.
- Mã nguồn không sửa:
  - nhánh này ở commit `56513b8` cho các cấu hình mới;
  - `develop@229b3e4` cho "R6 cũ".

**Lệnh và cấu hình server**
- Server dựng bằng chính `LocalServingRunner.build_command()` / `setup_environment()` của từng cây mã, tức đúng thứ `flexsds serve --backend local` chạy.
- Các cờ giống mục 3 (đều có `--no-prefix-caching`).
- Thêm 2 nhánh đối chứng:
  - `r6_old`: R6 trước khi sửa, cùng cờ;
  - `r6_new_nostep_nolineage`: như 3b nhưng tắt thêm file `serving_lineage.jsonl`.

**Tải đo**
- W1 (`experiments/configs/w1`), `c = 1 / 8 / 32`, 32 request, 3 request khởi động, 2 lần lặp.
- Đây là phép đo kiểm chứng, chưa phải giao thức M3-05 đầy đủ (W1–W3, 6 mức tải, 5 lần lặp).

**Dữ liệu thô:** volume Modal `flexsds-r6-campaign`, thư mục `bench-r6fix-2411/`, workspace `daibui1208`. Đã tải về máy Dương, nằm ngoài repo.

### 4.2. Tốc độ (TPOT p50, × = TPOT vanilla / TPOT nhánh)

| Nhánh | `c=1` | `c=8` | `c=32` | Request thành công |
|---|---:|---:|---:|---|
| Vanilla matched (TPOT ms) | 15,9 | 20,4 | 28,6 | 192/192 |
| Static EAGLE `K=3` matched | **1,64×** | **1,22×** | 0,63× | 192/192 |
| R6 cũ (`develop@229b3e4`) | 1,49× | 1,01× | 0,55× | 191/192 (*) |
| **R6 mới, `flush=64` (3a)** | 1,59× | 1,09× | 0,67× | 192/192 |
| R6 mới, `--no-step-telemetry` (3b) | 1,63× | 1,10× | **0,68×** | 192/192 |
| R6 mới, tắt cả step + lineage | 1,60× | 1,11× | 0,67× | 192/192 |

(*) Một request của R6 cũ ở `c=32` lỗi phía client harness (`ClientOSError: Can not write request body`) ngay lúc gửi. Đây là lỗi kết nối, không phải server lỗi.

### 4.3. Telemetry của R6 mới (3a)

**Bộ ước lượng α**
- `alpha_estimator = "geometric_mle"` ở 100% bản ghi `planned_decision` (101.914 bản ghi).
- Tỉ lệ chấp nhận thật đo được:
  - tính theo vị trí được kiểm (accepted / tested): 0,611;
  - tính theo cách cũ (accepted / drafted): 0,416.
- α bộ điều khiển dùng: trung vị 0,59 ở `B=1` và 0,60 ở `B=8`. R6 cũ dùng 0,47 và 0,48.

**Phân bố K** (B = số request trong batch):

| `B` | R6 cũ | R6 mới |
|---|---|---|
| 1 | K1 49%, K3 35%, K4 16% | **K1 19%**, K3 39%, K4 42% |
| 8 | K1 54%, K3 46% | **K1 28%**, K3 72% |
| 32 | K0 95% | K0 95% (không đổi) |

**Trường telemetry mới:** `geometric_alpha_mle`, `alpha_estimator`, `total_tested_positions` có trong 235/235 bản ghi `serving_trace.jsonl`, ở cả 3a và 3b.

### 4.4. Đối chiếu `AC-07`

| Ý | Kết quả |
|---|---|
| R6 thu hẹp khoảng cách với Static `K=3` ở `c = 1, 8` | Đạt. Ở `c=1`: R6 cũ kém static 9%, R6 mới kém 3%. Ở `c=8`: R6 cũ kém 17%, R6 mới kém 11%. |
| Tỉ lệ `K = 1` dưới 20% | Đạt ở `B=1` (19%). **Chưa đạt ở `B=8` (28%).** |
| Tiếp tục vượt Static `K=3` ở `c = 32` | Đạt: 0,67× so với 0,63× (3b: 0,68×). |
| Không hồi quy so với R6 cũ | Đạt. Nhanh hơn ở mọi mức tải: +7% (`c=1`), +8% (`c=8`), +22% (`c=32`). |

### 4.5. Nhận xét

**Telemetry**
- Ghi `flush=64` chỉ tốn khoảng 0,4–0,9 ms mỗi bước so với tắt hẳn step telemetry (1,59× so với 1,63× ở `c=1`).
- Tắt thêm file `serving_lineage.jsonl` không tạo khác biệt đo được (nằm trong nhiễu).

**Còn thua static ở `c=8`:**
- Ở `B=8`, profile cho `C3/C1 = 23,2 / 18,6`.
- Khi α của một request tụt dưới khoảng 0,5, K=1 lại thắng trong công thức chọn K. Ước lượng MLE theo từng request với `β = 0,90` dao động đủ để điều này xảy ra ở khoảng 28% quyết định.
- Hướng cải thiện (chưa làm): đo lại profile trên đúng container A10 bằng `flexsds profile --benchmark`, hoặc tăng `mle_beta`.

**Ở `c=32` mọi nhánh speculative vẫn chậm hơn vanilla.** Đây là giới hạn "bước K=0 không miễn phí" phía vLLM, ngoài phạm vi theo FR-05.

**Kiểm thử offline lại trên máy Dương (Windows, conda `doan-d22`), trỏ `PYTHONPATH` vào mã của nhánh này:**
- Các file test liên quan: 195 pass, 1 skip.
- Toàn bộ `tests/`: 76 fail và 58 lỗi, **giống hệt** `develop`. Nguyên nhân là môi trường: bản xuất không có `.git`, không có vLLM trên Windows, khác cách xử lý đường dẫn.
- Không có test nào chỉ fail trên nhánh này. Nhánh này thêm 20 test pass.

**Hai điểm lệch nhỏ giữa tài liệu và mã:**
1. `plan.md` (Tác vụ 2, Bước 3) ghi "chỉ dựng `_decision_record` khi có telemetry". Trong mã, `plan_step` vẫn luôn dựng bản ghi này. Không ảnh hưởng thực tế, vì runner luôn đặt `ADAPTIVE_TELEMETRY_PATH`.
2. `--no-step-telemetry` không tắt `serving_lineage.jsonl`. Phép đo ở 4.2 cho thấy điều này không ảnh hưởng tốc độ.
