# Kế hoạch triển khai luồng benchmark đa GPU (3 node × 4 GPU × 3 làn) cho FlexSDS CLI

> **Dành cho triển khai:** Thực hiện tuần tự theo TDD (viết bài kiểm thử lỗi trước → cài đặt tối giản vượt kiểm thử → rà soát → commit theo từng phần), giữ nguyên nhánh `feat/modal-gpu-catalog-and-config-export`, tuyệt đối không chỉnh sửa `web/` hay `src/api/`, không phá hợp đồng M3-05 và M5-01 cũ.

**Đặc tả tham chiếu:** `docs/superpowers/specs/2026-11-25-flexsds-benchmark-campaign-design.md`

---

## Task 1: Hỗ trợ 4 GPU Modal, server profile cho `benchmark_harness` và sửa AWQ FP16 trên `T4`

**Tệp sửa đổi:**
- `experiments/modal_lanes.py`
- `experiments/modal_runner.py`
- `experiments/benchmark_harness.py`
- `src/flexsds/server/local_runner.py`
- `src/flexsds/core/calibration.py`
- `tests/test_benchmark_campaign.py` (tạo mới, bắt đầu bằng bài test cho Task 1)

**Yêu cầu chính:**
- `modal_lanes.py`: mở rộng `SUPPORTED_GPUS` thêm `L40S`, `A100` (giữ `L4` và `DEFAULT_GPU = "L4"`), thêm `modal_gpu_resource()`, `default_dtype_profile()`, `environment_profile_for_gpu()`, và tham số tùy chọn `environment_profile` cho `bench_harness_args()`.
- `modal_runner.py`: dùng `lanes.modal_gpu_resource(GPU)` khi truyền `gpu=` vào decorator Modal.
- `benchmark_harness.py`: thêm 4 server profile `modal-t4-v1`, `modal-a10-v1`, `modal-l40s-v1`, `modal-a100-v1` và dùng bộ khớp SKU chặt trong `validate_environment()` (`L40S ≠ L4`, `A100-80GB ≠ 40GB`, `A10` khớp `A10`/`A10G`, `T4` khớp `Tesla T4`/`NVIDIA T4`).
- `local_runner.py` & `calibration.py`: khi target là AWQ và chạy trên `T4` (`target_gpu == "T4"` hoặc `sm_architecture == "sm_75"`), tự chuyển sang `quantization="awq"` và `dtype="half"` (`local_runner`) / `dtype="float16"` (`calibration`).

## Task 2: Bộ điều phối chiến dịch đa phần cứng (`experiments/run_benchmark_campaign.py`)

**Tệp tạo mới / cập nhật:**
- `experiments/run_benchmark_campaign.py`
- `tests/test_benchmark_campaign.py`

**Yêu cầu chính:**
- Định nghĩa ma trận chuẩn:
  - `BENCHMARK_GPUS = ("T4", "A10", "L40S", "A100")`
  - `BENCHMARK_WORKLOADS = ("W1_V3", "W2", "W3")`
  - `BENCHMARK_LANES = ("vanilla", "static", "adaptive")`
  - `CONCURRENCY_MODES = {"full": (1, 2, 4, 8, 16, 32), "fast": (1, 8, 32)}`
  - `WARMUP_REQUESTS = 3`, `REPEATS = 5`, `NUM_REQUESTS = 32` cho cả `full` (216 ô) và `fast` (108 ô).
- Xây dựng `plan()`, `save_plan()` (write-once), `resolve_adaptive_profiles()` (tìm `--profile-dir` → `~/.flexsds/profiles` → `src/flexsds/profiles`, fail-closed mặc định), `ensure_adaptive_profiles()` (hỗ trợ `--auto-calibrate`), `build_cell_serving_config()`, `drive()`, `validate_cell()`, `validate_campaign()`, `aggregate()` (phân nhóm theo `gpu` trước khi tính `speedup_vs_vanilla`), `render_markdown_report()`, và CLI `main()`.

## Task 3: Modal benchmark app độc lập (`experiments/modal_benchmark_runner.py`)

**Tệp tạo mới / cập nhật:**
- `experiments/modal_benchmark_runner.py`
- `tests/test_benchmark_campaign.py`

**Yêu cầu chính:**
- Modal app riêng `flexsds-benchmark` (không đụng `ptit-vllm` hay `flexsds-serving`), cấu hình truyền qua `FLEXSDS_BENCH_CONFIG_B64`.
- Image dựng từ `serving_image()` + bản vá `experiments/vllm_runtime_patch.py` (SHA-256 gated) + mã nguồn `src/` và `experiments/`.
- `@modal.web_server` chạy `LocalServingRunner.build_command()` + `setup_environment()`, ghi `server_env.json` (gồm `instance_id`, `gpu`, `dtype`, `quantization`, `runtime_variant`, `vllm_outputs_sha256`, `vllm_args`, `profile_sha256`) và mẫu NVML `vram.jsonl`.
- Xác thực runtime GPU và `validate_measured_profile()` trong container cho làn `adaptive`.
- Hàm detached `@app.function` `bench(...)` lấy URL qua `get_web_url()` (không khóa cứng workspace hay `ptit-proxy-token`), chạy `experiments.benchmark_harness` với `--environment-profile modal-<gpu>-v1 --warmup-policy m3-fixed3`, ghép VRAM qua `modal_vram_join.py`, và cung cấp `bench_submit`, `bench_status`, `bench_collect`, `bench_recover`, `stop_app`.

## Task 4: Tích hợp `flexsds benchmark` và nhánh `Measure & available profile GPU -> Measure now`

**Tệp tạo mới / sửa đổi:**
- `src/flexsds/cli/benchmark.py` (tạo mới)
- `src/flexsds/cli/main.py`
- `src/flexsds/cli/wizard.py`
- `tests/test_cli_menu_help.py`
- `tests/test_benchmark_campaign.py`
- `.agents/skills/flexsds-assistant/references/flexsds-cli.md`

**Yêu cầu chính:**
- Lệnh con `flexsds benchmark` hỗ trợ đầy đủ `plan`, `run`, `status`, `validate`, `aggregate`, `stop` cùng các cờ `--mode`, `--gpus`, `--workloads`, `--lanes`, `--target`, `--draft`, `--draft-method`, `--static-k`, `--match-adaptive-runtime`, `--profile-dir`, `--auto-calibrate`, `--calibrate-mode`, `--campaign-id`, `--execution-sha`, `--retry-failed`, `--dry-run`.
- Giữ nguyên 5 mục ở menu chính `wizard.py` (không thêm mục ngoài). Bên trong `Measure & available profile GPU` → `Measure now`, thêm lựa chọn `Measurement type:` gồm `GPU cost profile (calibration)` (mặc định, tương thích ngược 100%) và `Multi-GPU benchmark (3 nodes × 4 GPUs)`.
- Cập nhật `CHOICE_HELP` và kiểm thử `flexsds benchmark --help` không đụng phần cứng hay mạng.

## Trọng tâm rà soát (Review Focus)
1. **Cô lập SKU GPU:** `L40S` không khớp `L4`; `A100` ánh xạ `A100-80GB` và từ chối profile/server 40 GB; `T4` dùng `half`/`awq` cả khi serve, calibrate lẫn benchmark.
2. **An toàn tổng hợp đa GPU:** `aggregate()` bắt buộc tách theo `gpu` trước khi tính `speedup_vs_vanilla` — không bao giờ so làn của GPU này với `vanilla` của GPU khác.
3. **Không phá M3/M5 và menu CLI cũ:** `run_m3_campaign.py`, `run_m5_campaign.py`, `modal_lanes.LANES`, `validate_m3.py`, và chỉ số 0..4 của menu chính trong `wizard.py` giữ nguyên hành vi cũ.
4. **Tính nhất quán `full` vs `fast`:** Cả hai chế độ cùng 4 GPU × 3 làn × 3 node (`W1_V3`, `W2`, `W3`) × 3 warm-up × 5 repeats × 32 request; chỉ khác concurrency `(1, 2, 4, 8, 16, 32)` (216 ô) so với `(1, 8, 32)` (108 ô).
