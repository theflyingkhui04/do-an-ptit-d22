# Danh mục tổng hợp toàn bộ các lệnh trong dự án FlexSDS (D22 PTIT)

- **Ngày tổng hợp:** 2026-11-22
- **Mốc mã nguồn đối chiếu:** nhánh `fix/r6-hardware-cost-driven-budget` (`flexsds v0.3.4`, vLLM `0.30.0`)
- **Nguyên tắc:** Mọi lệnh và cờ dưới đây đều được đối chiếu trực tiếp từ mã nguồn thực tế (`src/flexsds/cli/main.py`, `src/flexsds/cli/wizard.py`, `src/api/README.md`, `web/package.json`, `scripts/`, `experiments/`).

---

## 1. FlexSDS CLI (`flexsds`)

> **Lưu ý cài đặt:** Cài ở chế độ editable từ thư mục gốc `do-an-ptit-d22/`:
> ```bash
> pip install -e .
> # Hoặc chạy trực tiếp không cần cài:
> PYTHONPATH=src:. python3 -m flexsds.cli.main <lệnh>
> ```

### 1.1. Tùy chọn chung và chế độ tương tác (wizard)

| Lệnh | Mục đích | Ghi chú |
|---|---|---|
| `flexsds` | Mở giao diện wizard tương tác | Menu gồm 5 mục: *Launch serving*, *Detect hardware*, *Measure & available profile GPU*, *Delete Modal token*, *Running endpoint*. Mặc định luôn bật Speculative Decoding (không có tùy chọn Vanilla trong wizard). |
| `flexsds --help` / `flexsds -h` | Xem trợ giúp tổng quát | Thoát mã `0`. |
| `flexsds --version` / `flexsds -v` | Hiển thị phiên bản gói | Hiện tại `0.3.4`. |
| `flexsds <subcommand> --help` | Xem đầy đủ cờ của từng lệnh con | Ví dụ: `flexsds serve --help`. |

### 1.2. Phát hiện phần cứng, kiểm tra model và đo profile GPU

| Lệnh | Các cờ hỗ trợ | Mục đích |
|---|---|---|
| `flexsds detect` | *(không có cờ)* | Phát hiện phần cứng máy hiện tại (GPU CUDA, kiến trúc `sm_*`, VRAM trống; fallback về CPU/RAM nếu không có CUDA). |
| `flexsds inspect [model]` | `--eagle-version {1,2,3,3.1}`<br>`--method <policy>` | Nhận diện phương pháp đầu cơ từ model Hugging Face / thư mục cục bộ (vd: `flexsds inspect duckhoi/qwen2.5-7b-eagle3.1-speculator`) hoặc kiểm tra một policy tường minh (`ngram`, `jacobi`). |
| `flexsds profile` | `--benchmark`<br>`--target <model>` (mặc định `Qwen/Qwen2.5-7B-Instruct-AWQ`)<br>`--draft <model>` (mặc định `duckhoi/qwen2.5-7b-eagle3.1-speculator`)<br>`--model <override>`<br>`--measure-mode {fast,full}` (mặc định `full`: 50 lặp; `fast`: 5 lặp)<br>`--backend {local,modal}` (mặc định `local`)<br>`--gpu {A10G,L4,T4}` (mặc định `A10G`) | Không có `--benchmark`: tra cứu và hiển thị hardware cost profile hiện có cho GPU.<br>Có `--benchmark`: chạy đo thực nghiệm ma trận độ trễ `cost_matrix_ms` theo `(K, B)` và lưu file JSON vào `~/.flexsds/profiles/`. |

Ví dụ đo profile thực nghiệm trên Modal A10G:
```bash
flexsds profile --benchmark --backend modal --gpu A10G --measure-mode full
```

### 1.3. Quản lý xác thực và tài khoản dịch vụ

| Lệnh | Các cờ hỗ trợ | Mục đích |
|---|---|---|
| `flexsds auth status` | *(không có cờ)* | Xem trạng thái (đã che ký tự) các token đã lưu ngoài repo (`~/.flexsds/credentials`). |
| `flexsds auth login` | `--provider {modal,huggingface,cloudflare}` (mặc định `modal`)<br>`--key <key_name>`<br>`--token <value>` *(nên bỏ trống để nhập ẩn qua prompt)* | Lưu khóa/token dịch vụ một cách an toàn. |
| `flexsds auth logout` | `--provider <name>` *(bỏ trống để xóa tất cả)* | Xóa thông tin đăng nhập đã lưu cục bộ. |
| `flexsds clear modal` | Tham số bắt buộc `modal` | Ngắt kết nối tài khoản Modal cục bộ (xóa cả token FlexSDS lẫn profile Modal hiện hành, yêu cầu đăng nhập lại). |

### 1.4. Khởi chạy serving (`flexsds serve`)

> **Quy tắc mặc định:** `flexsds serve` mặc định bật **Speculative Decoding + Hardware-Adaptive (R6 cost-driven)** trên backend `modal`. Các cờ đối chứng (`--match-adaptive-runtime`, `--vanilla`, `--no-prefix-caching`, `--static-k`) là cờ opt-in ở dòng lệnh phục vụ kiểm chứng khoa học và benchmark.

| Cờ của `flexsds serve` | Kiểu / Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `--target <model>` | `str` | ID Hugging Face hoặc đường dẫn mô hình đích. |
| `--draft <model>` | `str` | ID Hugging Face hoặc đường dẫn mô hình nháp / đầu EAGLE. |
| `--backend` | `{modal,local,colab}` (mặc định `modal`) | Nền tảng triển khai (`colab` chỉ in hướng dẫn cấu hình notebook). |
| `--gpu` | `{L4,T4,A10G}` (mặc định `L4`) | Loại GPU cấp phát trên Modal và ngân sách VRAM ước lượng. |
| `--port` | `int` (`1..65535`, mặc định `8000`) | Cổng máy chủ OpenAI-compatible HTTP. |
| `--max-num-seqs` | `int > 0` (mặc định `32`) | Số request chạy đồng thời tối đa. |
| `--max-model-len` | `int > 0` (mặc định `4096`) | Độ dài ngữ cảnh tối đa mỗi sequence. |
| `--eagle-version` | `{1,2,3,3.1}` | Chỉ định rõ phiên bản EAGLE khi checkpoint dùng chung metadata cũ. |
| `--background` | cờ bật | Chạy máy chủ local dưới nền và ghi PID vào `.flexsds.pid` (chỉ POSIX/Linux). |
| `--deploy` | cờ bật | Triển khai persistent trên Modal Cloud (`modal deploy` thay vì `modal serve`). |
| `--allow-oom-risk` | cờ bật | Bỏ qua cảnh báo ước lượng vượt VRAM (các lỗi tương thích khác vẫn chặn). |
| `--no-adaptive` | cờ bật | Tắt bộ điều khiển Hardware-Adaptive, chạy Static Speculative Decoding. |
| `--match-adaptive-runtime` | cờ bật *(yêu cầu `--no-adaptive` hoặc `--vanilla`)* | Ép Static/Vanilla chạy trên cùng hợp đồng V1 eager (`--enforce-eager`, `--compilation-config '{"mode": 0}'`, `--async-scheduling`, `VLLM_USE_V2_MODEL_RUNNER=0`) với Adaptive ON để so sánh công bằng. |
| `--vanilla` | cờ bật *(ngụ ý `--no-adaptive`, cấm kèm `--draft`)* | Chạy chế độ đối chứng Target-only (tắt hoàn toàn Speculative Decoding). |
| `--no-prefix-caching` | cờ bật | Thêm `--no-enable-prefix-caching` cho vLLM (khuyến nghị khi chạy lossless/repeatability). |
| `--static-k <K>` | `int` (`1..8`, yêu cầu `--no-adaptive`, cấm kèm `--vanilla`) | Đặt số token nháp tĩnh `num_speculative_tokens` (mặc định `4`). |
| `--dry-run` | cờ bật | Chỉ in lệnh thực thi và nhãn `Execution contract:` mà không khởi chạy server. |
| `--allow-fallback` | cờ bật *(deprecated)* | Không còn được phép bỏ qua yêu cầu profile thực nghiệm. |

#### Bộ 4 lệnh chuẩn dùng cho kiểm chứng lossless và benchmark đối chứng trên Modal A10G

```bash
# 1. Adaptive EAGLE3 (R6 — chọn K động theo EmpiricalCostProfile A10G)
flexsds serve --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-prefix-caching

# 2. Static EAGLE3 matched (K=4 cố định, cùng V1 eager với Adaptive ON)
flexsds serve --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-adaptive --match-adaptive-runtime --static-k 4 --no-prefix-caching

# 3. Vanilla target-only matched (tắt đầu cơ, cùng V1 eager)
flexsds serve --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --vanilla --match-adaptive-runtime --no-prefix-caching

# 4. Static EAGLE3 mặc định của vLLM (V2 + CUDA Graphs + torch.compile — cột tham chiếu end-to-end)
flexsds serve --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-adaptive
```

### 1.5. Tương tác, giám sát và dừng server local

| Lệnh | Các cờ hỗ trợ | Mục đích |
|---|---|---|
| `flexsds chat` | `--endpoint <url>` (mặc định `http://localhost:8000/v1`)<br>`--model <name>`<br>`--temperature <float>` (mặc định `0.0`) | Mở phiên chat streaming tương tác với endpoint đang phục vụ. |
| `flexsds monitor` | `--trace-file <path>` (mặc định `telemetry/serving_trace.jsonl`)<br>`--refresh-rate <sec>` (mặc định `1.0`) | Theo dõi trực tiếp file log JSONL telemetry. |
| `flexsds stop` | *(không có cờ)* | Dừng tiến trình local đang chạy nền theo `.flexsds.pid` (POSIX/Linux). |

---

## 2. Web demo (Backend FastAPI & Frontend Next.js)

### 2.1. Backend FastAPI (`src/api/`)

```bash
# Khởi chạy API demo cục bộ
DEMO_KEY=... DEMO_DATABASE_URL=... DEMO_READER_DATABASE_URL=... \
  MODAL_PROXY_TOKEN_ID=... MODAL_PROXY_TOKEN_SECRET=... \
  uvicorn src.api.main:build_app --factory --port 8000

# Deploy dịch vụ OCR thư viện (CPU) trên Modal
modal deploy -m experiments.modal_ocr_lib

# Nạp dữ liệu mồi 25 hóa đơn W1_v3 vào Supabase Postgres
DEMO_DATABASE_URL=... SUPABASE_URL=... SUPABASE_SERVICE_KEY=... \
  python3 scripts/seed_demo_db.py
```

### 2.2. Frontend Next.js (`web/`)

Chạy trong thư mục `do-an-ptit-d22/web/`:

| Lệnh | Mục đích |
|---|---|
| `NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev` | Chạy máy chủ phát triển Next.js |
| `npm run build` | Build production |
| `npm run start` | Khởi chạy bản build production |
| `npm run lint` | Kiểm tra ESLint |
| `npm run test` | Chạy unit test Vitest (`vitest run`) |
| `npm run check:bundle` | Kiểm tra kích thước bundle (`node scripts/check-bundle.mjs`) |

---

## 3. Pipeline hóa đơn VAT (Node 1 → Node 2 → Node 3) và dữ liệu (`scripts/`)

| Lệnh | Mục đích |
|---|---|
| `python3 scripts/prepare_mcocr_devset.py` | Chuẩn bị tập dev MC-OCR |
| `python3 scripts/build_w1_v3_workload.py` | Sinh bộ workload đóng băng W1 (v3) |
| `python3 scripts/build_w2_w3_workloads_v2.py` | Sinh bộ workload đóng băng W2 (Text-to-SQL) và W3 (Báo cáo) |
| `python3 scripts/check_w1_dataset.py` | Kiểm tra toàn vẹn bộ dữ liệu W1 |
| `python3 scripts/build_node1_requests.py` | Dựng request đầu vào cho Node 1 |
| `python3 scripts/run_node1_extractor.py` | Chạy trích xuất JSON hóa đơn VAT (Node 1) |
| `python3 scripts/evaluate_node1_lanes.py` | Đánh giá độ chính xác trích xuất Node 1 giữa các lane |
| `python3 scripts/validate_invoice_schema.py` | Kiểm tra JSON trích xuất theo schema hóa đơn VAT |
| `python3 scripts/check_invoice_contract.py` | Kiểm tra hợp đồng dữ liệu hóa đơn |
| `python3 scripts/init_invoice_sqlite.py` | Khởi tạo CSDL SQLite mẫu cho hóa đơn |
| `python3 scripts/run_pipeline_e2e.py` | Chạy luồng end-to-end Node 1 → Node 2 → Node 3 |
| `python3 scripts/eval_demo_questions.py [--reference-only]` | Chấm điểm bộ câu hỏi demo (`--reference-only` chỉ kiểm tra DB) |
| `python3 scripts/build_full_srs_excel.py` / `build_sdd_excel.py` | Xuất tài liệu SRS / SDD sang Excel |

---

## 4. Thực nghiệm, benchmark, lossless và script chạy lane (`experiments/` & `scripts/`)

### 4.1. Khởi chạy server theo từng lane trên Linux/GPU (legacy M1–M3)
- `bash scripts/launch_lane1_vanilla.sh` — Lane 1 (Vanilla)
- `bash scripts/launch_lane2_static.sh` — Lane 2 (Static speculative)
- `bash scripts/launch_lane3_adaptive.sh` — Lane 3 (Adaptive scheduler)
- `bash scripts/launch_lane4_eagle.sh` — Lane 4 (EAGLE)
- `bash scripts/switch_lane.sh` — Chuyển đổi sạch giữa các lane (giải phóng GPU)

### 4.2. Harness benchmark, kiểm chứng và phân tích (`experiments/`)
- `python3 experiments/benchmark_harness.py` — Chạy quét concurrency/batch theo config YAML trong `experiments/configs/`
- `python3 experiments/aggregate_results.py` — Tổng hợp kết quả TPOT/TTFT/throughput ra CSV
- `python3 experiments/m4_readiness.py <evidence_dir> --source-commit <40-hex-sha>` — Bộ chấm readiness fail-closed cho chiến dịch M4
- `python3 experiments/m4_lossless.py` / `experiments/lossless_verifier.py` — So khớp token ID greedy lossless
- `python3 experiments/m4_cost_profiler.py` / `scripts/build_m4_cost_profiles.py` — Công cụ dựng hardware cost profile M4
- `python3 experiments/async_lineage_runtime_probe.py` — Kiểm tra nhanh lineage bất đồng bộ trên runtime
- `python3 experiments/plot_f1.py` / `experiments/plot_m2.py` — Vẽ biểu đồ phân tích kết quả
- `python3 scripts/smoke_lane3_adaptive.py` / `scripts/stress_test_extreme.py` / `scripts/test_vram_stability.py` — Smoke test và stress test VRAM

---

## 5. Kiểm thử tự động (`pytest`)

Chạy từ thư mục `do-an-ptit-d22/`:

```bash
# 1. Bộ test mới cho R6 (cost-driven K) và Matched execution contract
PYTHONPATH=src:. pytest \
  tests/test_hardware_cost_driven_controller.py \
  tests/test_matched_execution_contract.py -v

# 2. Bộ test hồi quy offline toàn diện (không cần torch, modal SDK hay mạng)
PYTHONPATH=src:. pytest \
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

# 3. Bộ test đóng gói Modal (cần cài gói modal trong cùng Python env)
PYTHONPATH=src:. pytest tests/test_modal_packaging_contract.py -v

# 4. Bộ test vòng đời VRAM (cần cài torch)
PYTHONPATH=src:. pytest tests/test_vram_memory_lifecycle.py -v

# 5. Bộ test Backend API demo (cần cài fastapi, httpx, pydantic)
PYTHONPATH=src:. pytest \
  tests/test_api_*.py tests/test_demo_*.py \
  tests/test_eval_demo_questions.py tests/test_invoice_rows.py \
  tests/test_node1_request.py -q
```
