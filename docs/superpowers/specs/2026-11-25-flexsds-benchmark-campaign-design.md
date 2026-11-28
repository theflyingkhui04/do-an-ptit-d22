# Thiết kế luồng benchmark đa phần cứng (3 node × 4 GPU × 3 làn) cho FlexSDS CLI

**Ngày:** 2026-11-25  
**Nhánh làm việc:** `feat/modal-gpu-catalog-and-config-export` (giữ nguyên nhánh hiện tại từ `develop`)  
**Phạm vi:** Phía CLI (`src/flexsds/`) và bộ điều phối thực nghiệm trên Modal (`experiments/`). **Không chỉnh sửa** mã nguồn web demo (`web/`, `src/api/`) và **không làm vỡ** các kịch bản/bằng chứng M3-05, M5-01 lịch sử.

---

## 1. Mục tiêu và yêu cầu đã chốt

1. **Ma trận thực nghiệm chuẩn (Full benchmark):**
   - **3 node (3 task):**
     - Node 1: `W1_V3` (`experiments/configs/w1_v3/node1_chat_requests.jsonl`, ngữ cảnh `8192`, timeout `600 s`, vLLM `0.30.0` + bản vá `#55708`).
     - Node 2: `W2` (`experiments/configs/w2_v2/chat_requests.jsonl`, ngữ cảnh `4096`).
     - Node 3: `W3` (`experiments/configs/w3_v2/chat_requests.jsonl`, ngữ cảnh `4096`).
   - **4 phần cứng Modal:**
     - `T4` (Low, 16 GB VRAM, `sm_75`, tự động chọn `dtype="half"` + `quantization="awq"` vì Turing không hỗ trợ BF16/Marlin).
     - `A10` (Medium, 24 GB VRAM, `sm_86`, `dtype="bfloat16"` + `quantization="awq_marlin"`).
     - `L40S` (High, 48 GB VRAM, `sm_89`, `dtype="bfloat16"` + `quantization="awq_marlin"`).
     - `A100` (High, 80 GB VRAM, `sm_80`, ánh xạ Modal SDK `gpu="A100-80GB"`, `dtype="bfloat16"` + `quantization="awq_marlin"`).
   - **3 làn (lanes):**
     - `vanilla`: chỉ chạy 1 mô hình target (không bật speculative decoding).
     - `static`: bật speculative decoding trên cùng cặp target + draft/speculator với `K` cố định (mặc định `K=3`, cho phép đổi qua `--static-k`).
     - `adaptive`: giống hệt `static` về cặp mô hình nhưng bật lớp Hardware-Adaptive (`AdaptiveAsyncScheduler` + `LineageUniProcExecutor` + cost profile đo thật theo đúng GPU, `ADAPTIVE_MAX_K=4`).
   - **2 chế độ concurrency:**
     - `full`: `(1, 2, 4, 8, 16, 32)` → 4 GPU × 3 làn × 3 node × 6 concurrency = **216 ô**.
     - `fast`: `(1, 8, 32)` → giữ nguyên 4 GPU × 3 làn × 3 node và giữ nguyên số lượng đo mỗi ô (`3 warm-up + 5 repeats × 32 request = 160 request đo/ô`) → **108 ô**.
2. **Mô hình mặc định và khả năng tùy biến:**
   - Mặc định: `target = Qwen/Qwen2.5-7B-Instruct-AWQ`, `draft = duckhoi/qwen2.5-7b-eagle3.1-speculator` (`eagle3`), `static_k = 3`, `max_k = 4`.
   - Hỗ trợ đổi `--target`, `--draft`, `--draft-method`, `--static-k`, `--match-adaptive-runtime`, cũng như lọc tập con `--gpus`, `--workloads`, `--lanes` khi cần.
3. **Kiểm soát cost profile cho làn `adaptive` (fail-closed):**
   - Với mỗi GPU trong danh sách chạy, tìm profile `flexsds-component-v2` khớp chặt SKU (`L40S` không khớp `L4`; `A100` 80 GB từ chối profile 40 GB; `A10` đọc được profile đóng gói `profile_sm_86_remote_a10g_fcda5486b12d.json`) theo thứ tự `--profile-dir` → `~/.flexsds/profiles` → `src/flexsds/profiles`.
   - Nếu thiếu profile cho một hoặc nhiều GPU:
     - Mặc định **fail-closed** trước khi deploy bất kỳ container nào, liệt kê GPU còn thiếu và in lệnh `flexsds profile --benchmark --backend modal --gpu <GPU>`.
     - Nếu truyền `--auto-calibrate`, tự động chạy `run_hardware_micro_benchmark(..., backend="modal", measurement_mode=...)` cho từng GPU còn thiếu rồi xác thực lại trước khi bắt đầu benchmark.
   - Chế độ `plan` và `--dry-run` chỉ đọc và in bảng sẵn sàng profile (không chặn, không gọi Modal).
4. **Điểm vào (entrypoint):**
   - Bộ điều phối campaign mới: `experiments/run_benchmark_campaign.py` + Modal app tách biệt `experiments/modal_benchmark_runner.py` (`app = "flexsds-benchmark"`).
   - Tích hợp vào FlexSDS CLI qua lệnh `flexsds benchmark` (`src/flexsds/cli/benchmark.py`, `src/flexsds/cli/main.py`). Trong wizard tương tác (`src/flexsds/cli/wizard.py`), **không thêm mục mới vào menu chính** mà đặt trong nhánh sẵn có `Measure & available profile GPU` → `Measure now` (chọn giữa `GPU cost profile (calibration)` mặc định và `Multi-GPU benchmark (3 nodes × 4 GPUs)`).

---

## 2. Thiết kế kiến trúc chi tiết

### 2.1. Mở rộng nền tảng GPU và sửa lỗi AWQ trên `T4` (`sm_75`)

1. **`experiments/modal_lanes.py`:**
   - Mở rộng `SUPPORTED_GPUS` thành `("L4", "T4", "A10", "L40S", "A100")` (giữ `DEFAULT_GPU = "L4"` và `LANES` cũ để M3/M5 không thay đổi).
   - Thêm các hàm thuần Python:
     - `modal_gpu_resource(gpu)` (`A100` → `A100-80GB`).
     - `default_dtype_profile(gpu)` (`T4` → `fp16-fallback`; `A10`/`L40S`/`A100`/`L4` → `l4-bf16`).
     - `environment_profile_for_gpu(gpu)` (`L4` → `m3-l4-modal-v1`, `T4` → `modal-t4-v1`, `A10` → `modal-a10-v1`, `L40S` → `modal-l40s-v1`, `A100` → `modal-a100-v1`).
   - Cho phép truyền `environment_profile` tùy chọn vào `bench_harness_args()` (mặc định vẫn là `L4_ENVIRONMENT_PROFILE`).
2. **`experiments/benchmark_harness.py`:**
   - Thêm 4 server profile vào `ENVIRONMENT_PROFILES`: `modal-t4-v1`, `modal-a10-v1`, `modal-l40s-v1`, `modal-a100-v1` (cùng phiên bản ghim vLLM `0.30.0`, PyTorch `2.13.0+cu130`, CUDA `13.0`, `mode: "server"`).
   - Cập nhật `validate_environment()` dùng bộ khớp nhãn SKU chặt cho 4 profile mới (tái dùng quy tắc `matches_gpu_label`: `L40S` từ chối `L4`, `A100-80GB` từ chối `40GB`, `A10` chấp nhận `NVIDIA A10`/`NVIDIA A10G`, `T4` chấp nhận `Tesla T4`/`NVIDIA T4`).
3. **`src/flexsds/server/local_runner.py` và `src/flexsds/core/calibration.py`:**
   - Khi mô hình target là AWQ và chạy trên `T4` (`target_gpu == "T4"` hoặc `sm_architecture == "sm_75"`), tự động dùng `quantization="awq"` và `dtype="half"` (trong `LocalServingRunner.build_command`) / `dtype="float16"` (trong `calibration.py`). Các GPU Ampere/Ada (`A10`, `L40S`, `A100`, `L4`) giữ nguyên `awq_marlin` + `bfloat16`.
4. **`experiments/modal_runner.py`:**
   - Dùng `lanes.modal_gpu_resource(GPU)` khi truyền `gpu=` cho Modal decorator để nếu có gọi với `PTIT_GPU=A100` thì cấp đúng 80 GB.

### 2.2. Bộ điều phối chiến dịch đa GPU (`experiments/run_benchmark_campaign.py`)

- **Hằng số chuẩn:**
  - `BENCHMARK_GPUS = ("T4", "A10", "L40S", "A100")`
  - `BENCHMARK_WORKLOADS = ("W1_V3", "W2", "W3")`
  - `BENCHMARK_LANES = ("vanilla", "static", "adaptive")`
  - `CONCURRENCY_MODES = {"full": (1, 2, 4, 8, 16, 32), "fast": (1, 8, 32)}`
  - Giao thức thống kê mỗi ô (cả `full` và `fast`): `warmup = 3` (`m3-fixed3`), `repeats = 5`, `num_requests = 32`, `sample_policy = "cycle-rotate"`, `lifecycle = "fresh-server-per-cell"`.
- **Khóa định danh ô (cell key):**
  - `f"{gpu}__{lane}__{workload}__c{c}"` (ví dụ `A10__adaptive__W1_V3__c8`).
  - Mỗi lần thử (attempt) lưu riêng tại `experiments/runs/flexsds_bench/<campaign_id>/attempts/<attempt_id>/` với `submission.json`, `spawn_attempt.json`, `bench.json`, `server_env.json`, `raw.jsonl`, `warmup.jsonl`, `raw.vram.jsonl`, `validation.json`.
  - Bản kế hoạch đóng băng ghi 1 lần duy nhất (write-once, không ghi đè khi sai khác) tại `experiments/runs/flexsds_bench/<campaign_id>/plan.json`.
- **Xác thực (`validate_cell` / `validate_campaign`):**
  - Kiểm tra khớp `submission.json` ↔ `bench.json` ↔ `server_env.json` ↔ `plan.json` (đúng `gpu`, `lane`, `workload`, `concurrency`, `commit`, `target_model`, `draft_model`, `static_k`).
  - Kiểm tra đúng GPU thực tế trên server (`matches_gpu_label(modal_gpu_resource(gpu), server_env["gpu"])`), đúng `dtype`/`quantization` của GPU đó (`T4` phải dùng `half`/`awq`), đúng `vllm_outputs_sha256` bản vá `#55708`, đúng `max_model_len` của workload, và mỗi ô dùng một `server_instance_id` riêng biệt (không tái sử dụng container giữa các ô).
  - Kiểm tra đủ 160 request đo (5 × 32), đúng 3 warm-up hoàn tất trước khi đo, đủ `server_metrics` cho làn `static` và `adaptive`, và đủ `raw.vram.jsonl`.
  - Với làn `adaptive`, kiểm tra `server_env["profile_sha256"]` trùng khớp với profile đã xác thực.
- **Tổng hợp (`aggregate`):**
  - Phân nhóm theo **`gpu`** trước khi gọi `aggregate_results.apply_speedup_vs_vanilla()` để đảm bảo `speedup_vs_vanilla` của từng GPU luôn so với chính làn `vanilla` trên **cùng GPU, cùng workload, cùng concurrency**.
  - Xuất `experiments/results/flexsds_bench_<campaign_id>_summary.json` (kèm nhóm theo GPU/workload) và bảng Markdown `experiments/results/flexsds_bench_<campaign_id>_report.md`.

### 2.3. Modal benchmark app tách biệt (`experiments/modal_benchmark_runner.py`)

- App riêng `flexsds-benchmark` (tùy biến qua `FLEXSDS_BENCH_APP_NAME`), dùng chung Volume `ptit-model-cache`, `ptit-vllm-cache`, `ptit-run-artifacts`.
- Đóng gói `serving_image()` + áp dụng bản vá `experiments/vllm_runtime_patch.py` (SHA-256 gated) + mã nguồn `src/` và `experiments/`.
- Nhận cấu hình 1 ô qua biến môi trường base64 `FLEXSDS_BENCH_CONFIG_B64` (tránh lỗi escape JSON trên Windows), dựng lệnh vLLM qua `LocalServingRunner.build_command()` + `setup_environment()`.
- Không khóa cứng workspace `nguyenduonghb940` hay secret `ptit-proxy-token`: server chạy qua `@modal.web_server(PORT, startup_timeout=900)` với `max_containers=1`, `scaledown_window=60`, và hàm `bench(...)` tự lấy URL endpoint qua `modal.Function.from_name(APP_NAME, "serve").get_web_url()` nên chạy được trên mọi workspace Modal đã đăng nhập qua `flexsds auth login`.
- Với làn `adaptive`, gắn `hardware_profile.json` vào `/root/project/hardware_profile.json` và xác thực lại GPU thực tế (`torch.cuda.get_device_name(0)`) + `validate_measured_profile()` bên trong container trước khi khởi chạy vLLM.
- Chạy `experiments.benchmark_harness` từ hàm `@app.function` `bench(...)` cùng vùng Modal (detached `FunctionCall`, có `bench_submit`, `bench_status`, `bench_collect`, `bench_recover`), ghép VRAM server từ NVML (`modal_vram_join.py`).

### 2.4. Tích hợp vào FlexSDS CLI (`src/flexsds/cli/benchmark.py`, `main.py`, `wizard.py`)

- **Lệnh con `flexsds benchmark`:**
  - Hỗ trợ hành động: `plan` (mặc định khi truyền `--dry-run`), `run`, `status`, `validate`, `aggregate`, `stop`.
  - Cờ chính:
    - `--mode {full,fast}` (mặc định `full`).
    - `--gpus` (mặc định `all` → `T4,A10,L40S,A100`).
    - `--workloads` (mặc định `all` → `W1_V3,W2,W3`).
    - `--lanes` (mặc định `all` → `vanilla,static,adaptive`).
    - `--target`, `--draft`, `--draft-method`, `--static-k` (mặc định `3`), `--match-adaptive-runtime`.
    - `--profile-dir`, `--auto-calibrate`, `--calibrate-mode {fast,full}`.
    - `--campaign-id`, `--execution-sha`, `--retry-failed`, `--dry-run`.
  - Đảm bảo `flexsds benchmark --help` và `flexsds benchmark plan` / `--dry-run` hoạt động hoàn toàn offline, không đụng GPU hay credentials.
- **Wizard tương tác (`wizard.py`):**
  - Giữ nguyên 5 mục ở menu chính (`Select action:`), không thêm mục mới ở cấp ngoài.
  - Trong nhánh `Measure & available profile GPU` → `Measure now`, thêm bước chọn `Measurement type:` với 2 lựa chọn:
    1. `GPU cost profile (calibration)` (mặc định — giữ nguyên hoàn toàn luồng calibration và tương thích ngược với bộ test cũ).
    2. `Multi-GPU benchmark (3 nodes × 4 GPUs)` — cho chọn chế độ (`Full benchmark` 216 ô vs `Fast benchmark` 108 ô), phạm vi phần cứng (`All 4 Modal GPUs` hoặc GPU lẻ), mô hình, hiển thị bảng sẵn sàng profile của cả 4 GPU, rồi chọn `Preview plan (dry-run)`, `Run benchmark now`, `Auto-calibrate missing GPUs & run`, hoặc `Back`.

---

## 3. Chiến lược kiểm thử (TDD, chạy offline không tốn GPU)

1. **`tests/test_benchmark_campaign.py` (mới):**
   - Kiểm tra ma trận `full` (216 ô = 4 GPU × 3 làn × 3 node × 6 concurrency) và `fast` (108 ô = 4 GPU × 3 làn × 3 node × 3 concurrency), cùng `warmup=3`, `repeats=5`, `num_requests=32` ở cả hai chế độ.
   - Kiểm tra `T4` tự động dùng `half`/`awq` còn `A10`/`L40S`/`A100` dùng `bfloat16`/`awq_marlin` trong cả `LocalServingRunner`, `calibration.py` và `modal_lanes.py`.
   - Kiểm tra `benchmark_harness.validate_environment()` cho cả 4 profile `modal-{t4,a10,l40s,a100}-v1` (gồm chặn nhầm `L4` ↔ `L40S` và `A100-40GB` ↔ `A100-80GB`).
   - Kiểm tra fail-closed khi thiếu profile adaptive cho `T4`/`L40S`/`A100`, và kiểm tra nhánh `--auto-calibrate` gọi đúng `run_hardware_micro_benchmark` cho từng GPU thiếu.
   - Kiểm tra `validate_cell` / `validate_campaign` từ chối tái sử dụng server, sai GPU, sai bản vá vLLM, thiếu request, và kiểm tra `aggregate` tính `speedup_vs_vanilla` **độc lập theo từng GPU**.
   - Kiểm tra `flexsds benchmark --help`, `flexsds benchmark plan --mode fast`, `flexsds benchmark --mode full --dry-run`, và luồng chọn `Run benchmark` trong wizard.
2. **Hồi quy:** Chạy lại toàn bộ suite offline hiện có (`test_m3_preflight.py`, `test_m5_campaign.py`, `test_benchmark_profiles.py`, `test_modal_gpu_catalog.py`, `test_modal_session_ux.py`, `test_cli_menu_help.py`, `test_local_runner.py`, v.v.) đảm bảo 0 lỗi vỡ.
