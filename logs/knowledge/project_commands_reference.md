# Danh mục tổng hợp toàn bộ các lệnh trong dự án FlexSDS (D22 PTIT)

- **Ngày cập nhật:** 2026-11-25
- **Mốc mã nguồn đối chiếu:** nhánh `feat/modal-gpu-catalog-and-config-export` (`flexsds v0.3.4`, vLLM `0.30.0`)
- **Nguyên tắc:** Mọi lệnh, cờ và hành vi dưới đây đều được đối chiếu trực tiếp từ mã nguồn thực tế ([`src/flexsds/cli/main.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/cli/main.py), [`src/flexsds/cli/wizard.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/cli/wizard.py), [`src/flexsds/cli/benchmark.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/cli/benchmark.py), [`src/flexsds/cli/serving_session.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/cli/serving_session.py), [`src/api/README.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/api/README.md), [`web/package.json`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/web/package.json), `scripts/`, `experiments/`).

---

## 1. FlexSDS CLI (`flexsds`)

> **Lưu ý cài đặt:** Cài ở chế độ editable từ thư mục gốc `do-an-ptit-d22/`:
> ```bash
> pip install -e .
> # Hoặc chạy trực tiếp không cần cài:
> PYTHONPATH=src:. python3 -m flexsds.cli.main <lệnh>
> ```

### 1.1. Danh mục 4 cấu hình GPU Modal chuẩn hóa trong CLI

| Tầng (Tier) | Tên CLI (`--gpu`) | Tài nguyên Modal SDK | VRAM | Kiến trúc | Giá tham chiếu (`modal.com/pricing`) | Dtype / Quant AWQ |
|---|---|---|---:|---|---|---|
| Low | `T4` | `T4` | 16 GB | `sm_75` | `$0.000164/s` (`~$0.59/h`) | `half` / `awq` (tự động FP16) |
| Medium *(mặc định)* | `A10` | `A10` | 24 GB | `sm_86` | `$0.000306/s` (`~$1.10/h`) | `bfloat16` / `awq_marlin` |
| High | `L40S` | `L40S` | 48 GB | `sm_89` | `$0.000542/s` (`~$1.95/h`) | `bfloat16` / `awq_marlin` |
| High | `A100` | `A100-80GB` | 80 GB | `sm_80` | `$0.000694/s` (`~$2.50/h`) | `bfloat16` / `awq_marlin` |

> **Lưu ý:** `L4` và `A100-40GB` đã được bỏ khỏi danh mục Modal CLI; toàn bộ tham chiếu hoạt động đã chuẩn hóa từ `A10G` về `A10`. Khi chọn GPU trong wizard Modal, dòng `Info:` ở footer tự động hiển thị tầng, dung lượng VRAM, kiến trúc SM và đơn giá. Ở chế độ Google Colab, việc chọn GPU chỉ là gợi ý ước lượng VRAM cho notebook tải lên, không tự cấp phát GPU.

### 1.2. Tùy chọn chung và chế độ tương tác (wizard)

| Lệnh | Mục đích | Ghi chú |
|---|---|---|
| `flexsds` | Mở giao diện wizard tương tác | Menu chính gồm đúng **5 mục**: *Launch serving*, *Detect hardware*, *Measure & available profile GPU*, *Delete Modal token*, *Running endpoint*. Trong *Measure & available profile GPU* → *Measure now* cho chọn *GPU cost profile (calibration)* hoặc *Multi-GPU benchmark (3 nodes × 4 GPUs)*. |
| `flexsds --help` / `flexsds -h` | Xem trợ giúp tổng quát | Thoát mã `0`, không chạm phần cứng. |
| `flexsds --version` / `flexsds -v` | Hiển thị phiên bản gói | Hiện tại `0.3.4`. |
| `flexsds <subcommand> --help` | Xem đầy đủ cờ của từng lệnh con | Ví dụ: `flexsds serve --help`, `flexsds benchmark --help`. |

### 1.3. Phát hiện phần cứng, kiểm tra model và đo profile GPU

| Lệnh | Các cờ hỗ trợ | Mục đích |
|---|---|---|
| `flexsds detect` | *(không có cờ)* | Phát hiện phần cứng máy hiện tại (GPU CUDA, kiến trúc `sm_*`, VRAM trống; fallback về CPU/RAM nếu không có CUDA). |
| `flexsds inspect [model]` | `--eagle-version {1,2,3,3.1}`<br>`--method <policy>` | Nhận diện phương pháp suy đoán (EAGLE-1/2/3/3.1, draft model) từ Hugging Face / thư mục cục bộ hoặc kiểm tra policy tường minh (`ngram`, `jacobi`). |
| `flexsds profile` | `--benchmark`<br>`--target <model>` (mặc định `Qwen/Qwen2.5-7B-Instruct-AWQ`)<br>`--draft <model>` (mặc định `duckhoi/qwen2.5-7b-eagle3.1-speculator`)<br>`--model <override>`<br>`--measure-mode {fast,full}` (mặc định `full`: 50 lặp; `fast`: 5 lặp)<br>`--backend {local,modal}` (mặc định `local`)<br>`--gpu {T4,A10,L40S,A100}` (mặc định `A10`) | Không có `--benchmark`: tra cứu profile hiện có.<br>Có `--benchmark`: đo ma trận độ trễ thành phần `flexsds-component-v2` theo `K ∈ [0..4] × B ∈ [1, 2, 4, 8, 16, 32]` và lưu JSON vào `~/.flexsds/profiles/`. |

Ví dụ đo profile thực nghiệm trên Modal:
```bash
flexsds profile --benchmark --backend modal --gpu A10 --measure-mode full
flexsds profile --benchmark --backend modal --gpu L40S --measure-mode fast
flexsds profile --benchmark --backend modal --gpu A100 --measure-mode fast
flexsds profile --benchmark --backend modal --gpu T4 --measure-mode fast
```

### 1.4. Quản lý xác thực và tài khoản dịch vụ

| Lệnh | Các cờ hỗ trợ | Mục đích |
|---|---|---|
| `flexsds auth status` | *(không có cờ)* | Xem trạng thái (đã che ký tự) các token lưu ngoài repo (`~/.flexsds/credentials`). |
| `flexsds auth login` | `--provider {modal,huggingface,cloudflare}` (mặc định `modal`)<br>`--key <key_name>`<br>`--token <value>` *(bỏ trống để nhập ẩn)* | Lưu token dịch vụ an toàn. |
| `flexsds auth logout` | `--provider <name>` *(bỏ trống để xóa hết)* | Xóa thông tin đăng nhập đã lưu cục bộ. |
| `flexsds clear modal` | Tham số bắt buộc `modal` | Ngắt kết nối tài khoản Modal cục bộ (xóa token FlexSDS và profile Modal hiện hành). |

### 1.5. Khởi chạy serving (`flexsds serve`) và menu sau khi deploy

> **Quy tắc mặc định:** `flexsds serve` mặc định bật **Speculative Decoding + Hardware-Adaptive (R6 geometric MLE + cost-driven K)** trên backend `modal`, GPU mặc định `A10`. Sau khi triển khai Modal sẵn sàng, menu phiên làm việc cung cấp **3 lựa chọn**:
> 1. `Watch log` — xem log server trực tiếp.
> 2. `Realtime chat with deployed LLM` — trò chuyện streaming ngay trên CLI (hiển thị đúng phương pháp suy đoán thực tế, không hardcode EAGLE-3.1).
> 3. `Export API config JSON (for web demo)` — xuất tệp JSON không chứa secret (gồm `endpoint`, `served_model`, `target_model`, `draft_model`, `draft_method`, `eagle_version`, `enable_adaptive`, `target_gpu`, `hardware`) tương thích trực tiếp với nút *Nhập từ `serving_config.json`* trên web demo.

| Cờ của `flexsds serve` | Kiểu / Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `--target <model>` | `str` | ID Hugging Face hoặc đường dẫn mô hình đích. |
| `--draft <model>` | `str` | ID Hugging Face hoặc đường dẫn mô hình nháp / đầu EAGLE. |
| `--backend` | `{modal,local,colab}` (mặc định `modal`) | Nền tảng triển khai (`colab` chỉ sinh/hướng dẫn notebook tải lên). |
| `--gpu` | `{T4,A10,L40S,A100}` (mặc định `A10`) | Loại GPU cấp phát trên Modal (`A100` ánh xạ `A100-80GB`) và ngân sách VRAM ước lượng. |
| `--port` | `int` (`1..65535`, mặc định `8000`) | Cổng máy chủ OpenAI-compatible HTTP. |
| `--max-num-seqs` | `int > 0` (mặc định `32`) | Số request chạy đồng thời tối đa. |
| `--max-model-len` | `int > 0` (mặc định `4096`) | Độ dài ngữ cảnh tối đa mỗi sequence. |
| `--eagle-version` | `{1,2,3,3.1}` | Chỉ định rõ phiên bản EAGLE khi checkpoint cũ dùng chung metadata. |
| `--background` | cờ bật | Chạy máy chủ local dưới nền và ghi PID vào `.flexsds.pid` (POSIX/Linux). |
| `--deploy` | cờ bật | Triển khai persistent trên Modal Cloud (`modal deploy` thay vì `modal serve`). |
| `--allow-oom-risk` | cờ bật | Bỏ qua cảnh báo ước lượng vượt 95% VRAM (các lỗi tương thích khác vẫn chặn). |
| `--no-adaptive` | cờ bật | Tắt Hardware-Adaptive, chạy Static Speculative Decoding. |
| `--match-adaptive-runtime` | cờ bật *(cần `--no-adaptive` hoặc `--vanilla`)* | Ép Static/Vanilla chạy cùng hợp đồng V1 eager (`--enforce-eager`, `--compilation-config '{"mode": 0}'`, `--async-scheduling`, `VLLM_USE_V2_MODEL_RUNNER=0`) để so sánh công bằng với Adaptive ON. |
| `--vanilla` | cờ bật *(ngụ ý `--no-adaptive`, cấm kèm `--draft`)* | Chạy đối chứng Target-only (tắt hoàn toàn Speculative Decoding). |
| `--no-prefix-caching` | cờ bật | Thêm `--no-enable-prefix-caching` cho vLLM. |
| `--static-k <K>` | `int` (`1..8`, cần `--no-adaptive`, cấm kèm `--vanilla`) | Đặt ngân sách nháp tĩnh `num_speculative_tokens`. |
| `--no-step-telemetry` | cờ bật | Tắt ghi log từng bước (`ADAPTIVE_STEP_TELEMETRY_PATH`, mặc định bật với batched flush = 64). |
| `--dry-run` | cờ bật | Chỉ in lệnh thực thi và nhãn `Execution contract:` mà không khởi chạy server. |
| `--allow-fallback` | cờ bật *(deprecated)* | Không được phép bỏ qua yêu cầu profile thực nghiệm. |

#### Bộ 4 lệnh chuẩn dùng cho kiểm chứng lossless và đối chứng trên Modal A10

```bash
# 1. Adaptive EAGLE-3.1 (R6 — chọn K động theo profile A10)
flexsds serve --backend modal --gpu A10 \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-prefix-caching

# 2. Static EAGLE-3.1 matched (K=4 cố định, cùng V1 eager)
flexsds serve --backend modal --gpu A10 \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-adaptive --match-adaptive-runtime --static-k 4 --no-prefix-caching

# 3. Vanilla target-only matched (tắt suy đoán, cùng V1 eager)
flexsds serve --backend modal --gpu A10 \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --vanilla --match-adaptive-runtime --no-prefix-caching

# 4. Static EAGLE-3.1 mặc định của vLLM (V2 + CUDA Graphs + torch.compile)
flexsds serve --backend modal --gpu A10 \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-adaptive
```

### 1.6. Chiến dịch benchmark đa phần cứng (`flexsds benchmark`)

> **Ma trận chuẩn:** 4 GPU (`T4`, `A10`, `L40S`, `A100`) × 3 node (`W1_V3`, `W2`, `W3`) × 3 làn (`vanilla`, `static`, `adaptive`), mỗi ô gồm `3 warm-up + 5 repeats × 32 request` (= 160 request đo/ô) với server tươi trên app Modal riêng `flexsds-benchmark`:
> - `--mode full`: concurrency `(1, 2, 4, 8, 16, 32)` → **216 ô** (34,560 request đo).
> - `--mode fast`: concurrency `(1, 8, 32)` → **108 ô** (17,280 request đo).

| Cờ của `flexsds benchmark` | Kiểu / Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `action` (vị trí 1) | `{plan,run,status,validate,aggregate,stop}` (mặc định `plan`) | Hành động trong vòng đời chiến dịch (`status`/`validate`/`aggregate` bắt buộc `--campaign-id`). |
| `--mode` | `{full,fast}` (mặc định `full`) | Chọn lưới concurrency đầy đủ (216 ô) hay nhanh (108 ô). |
| `--campaign-id <id>` | `str` | Mã định danh chiến dịch đóng băng tại `experiments/runs/flexsds_bench/<id>/plan.json`. |
| `--execution-sha <sha>` | 40-hex Git SHA | Commit nguồn đóng băng (mặc định lấy `HEAD`). |
| `--gpus` | `all` hoặc danh sách `T4,A10,L40S,A100` | Chọn toàn bộ 4 GPU hoặc tập con. |
| `--workloads` | `all` hoặc danh sách `W1_V3,W2,W3` | Chọn toàn bộ 3 node hoặc tập con. |
| `--lanes` | `all` hoặc danh sách `vanilla,static,adaptive` | Chọn toàn bộ 3 làn hoặc tập con. |
| `--target`, `--draft`, `--draft-method` | Mặc định `Qwen2.5-7B-Instruct-AWQ` + `eagle3.1` (`eagle3`) | Ghi đè cặp mô hình và phương pháp (`eagle`, `eagle3`, `draft_model`). |
| `--static-k`, `--adaptive-max-k` | Mặc định `3` và `4` (`1..8`) | Ngân sách `K` cố định của làn `static` và trần `K` của làn `adaptive`. |
| `--match-adaptive-runtime` | cờ bật | Bật V1 eager parity cho `vanilla` và `static`. |
| `--profile-dir <dir>` | `path` | Thư mục bổ sung tìm profile `flexsds-component-v2` (ưu tiên profile `full` trước `*_fast`). |
| `--auto-calibrate`, `--calibrate-mode` | cờ bật, `{fast,full}` (mặc định `fast`) | Tự động chạy calibration cho các GPU còn thiếu profile trước khi chạy làn `adaptive` (mặc định fail-closed). |
| `--retry-failed` | cờ bật | Tạo attempt mới cho ô bị lỗi trước đó. |
| `--dry-run`, `--json` | cờ bật | Xem trước ma trận và trạng thái sẵn sàng profile của 4 GPU (hoặc JSON thuần khi thêm `--json`), không gọi Modal hay CUDA cục bộ. |

```bash
# Xem trước ma trận full (216 ô) và fast (108 ô)
flexsds benchmark --mode full --dry-run
flexsds benchmark --mode fast --dry-run --json

# Chạy chiến dịch fast mode (tự calibrate các GPU còn thiếu như T4, L40S, A100)
flexsds benchmark run --mode fast --campaign-id flexsds-fast-v1 --auto-calibrate

# Theo dõi, xác thực và tổng hợp (tính speedup_vs_vanilla độc lập theo từng GPU)
flexsds benchmark status --campaign-id flexsds-fast-v1
flexsds benchmark validate --campaign-id flexsds-fast-v1
flexsds benchmark aggregate --campaign-id flexsds-fast-v1
flexsds benchmark stop
```

### 1.7. Tương tác, giám sát và dừng server local

| Lệnh | Các cờ hỗ trợ | Mục đích |
|---|---|---|
| `flexsds chat` | `--endpoint <url>` (mặc định `http://localhost:8000/v1`)<br>`--model <name>`<br>`--temperature <float>` (mặc định `0.0`) | Mở phiên chat streaming tương tác với endpoint đang phục vụ. |
| `flexsds monitor` | `--trace-file <path>` (mặc định `telemetry/serving_trace.jsonl`)<br>`--refresh-rate <sec>` (mặc định `1.0`) | Theo dõi trực tiếp file log JSONL telemetry. |
| `flexsds stop` | *(không có cờ)* | Dừng tiến trình local đang chạy nền theo `.flexsds.pid` (POSIX/Linux). |

---

## 2. Web demo (Backend FastAPI và Frontend Next.js)

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
| `npm test -- --run` | Chạy unit test Vitest (`38/38` bài kiểm thử, bao gồm parse `eagle_version` và GPU `T4`/`A10`/`L40S`/`A100`) |
| `npm run check:bundle` | Kiểm tra kích thước bundle (`node scripts/check-bundle.mjs`) |

---

## 3. Pipeline hóa đơn VAT (Node 1 → Node 2 → Node 3) và dữ liệu (`scripts/`)

| Lệnh | Mục đích |
|---|---|
| `python3 scripts/prepare_mcocr_devset.py` | Chuẩn bị tập dev MC-OCR |
| `python3 scripts/build_w1_v3_workload.py` | Sinh bộ workload đóng băng `W1_V3` |
| `python3 scripts/build_w2_w3_workloads_v2.py` | Sinh bộ workload đóng băng `W2` (Text-to-SQL) và `W3` (Báo cáo) |
| `python3 scripts/check_w1_dataset.py` | Kiểm tra toàn vẹn bộ dữ liệu W1 |
| `python3 scripts/build_node1_requests.py` | Dựng request đầu vào cho Node 1 |
| `python3 scripts/run_node1_extractor.py` | Chạy trích xuất JSON hóa đơn VAT (Node 1) |
| `python3 scripts/evaluate_node1_lanes.py` | Đánh giá độ chính xác trích xuất Node 1 giữa các làn |
| `python3 scripts/validate_invoice_schema.py` | Kiểm tra JSON trích xuất theo schema hóa đơn VAT |
| `python3 scripts/check_invoice_contract.py` | Kiểm tra hợp đồng dữ liệu hóa đơn |
| `python3 scripts/init_invoice_sqlite.py` | Khởi tạo CSDL SQLite mẫu cho hóa đơn |
| `python3 scripts/run_pipeline_e2e.py` | Chạy luồng end-to-end Node 1 → Node 2 → Node 3 |
| `python3 scripts/eval_demo_questions.py [--reference-only]` | Chấm điểm bộ câu hỏi demo (`--reference-only` chỉ kiểm tra DB) |
| `python3 scripts/build_full_srs_excel.py` / `build_sdd_excel.py` | Xuất tài liệu SRS / SDD sang Excel |

---

## 4. Thực nghiệm, chiến dịch benchmark và script chạy làn (`experiments/` và `scripts/`)

### 4.1. Chiến dịch benchmark đa GPU hiện hành (`experiments/`)
- `python3 -m experiments.run_benchmark_campaign plan|run|status|validate|aggregate|stop --campaign-id <id> --mode full|fast` — Điều phối trực tiếp chiến dịch 3 node (`W1_V3`, `W2`, `W3`) × 4 GPU (`T4`, `A10`, `L40S`, `A100`) × 3 làn (`vanilla`, `static`, `adaptive`) trên Modal app `flexsds-benchmark` ([`experiments/modal_benchmark_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/modal_benchmark_runner.py)).
- `python3 -m experiments.benchmark_harness --workload-file <jsonl> --environment-profile <profile> ...` — Harness bắn tải SSE bất đồng bộ, hỗ trợ profile `m1-t4-cu130-v1`, `m3-l4-modal-v1`, `m4-a100-colab-v1` và 4 server profile mới `modal-{t4,a10,l40s,a100}-v1`.
- `python3 -m experiments.aggregate_results` — Tổng hợp chỉ số TTFT/TPOT/E2E/throughput/$\alpha$/$\tau$/VRAM và `speedup_vs_vanilla`.

### 4.2. Chiến dịch lịch sử M1–M5 và công cụ kiểm chứng (giữ nguyên tính tái lập)
- `python3 -m experiments.run_m3_campaign plan|run|status|validate|aggregate|figures|stop` — Chiến dịch M3-05 (L4, 4 làn × 3 workload W1/W2/W3 × 6 concurrency = 72 ô).
- `python3 -m experiments.run_m5_campaign plan|run|status|validate|aggregate|compare|figures|stop` — Chiến dịch M5-01 (L4, 4 làn × W1_V3 ngữ cảnh 8192 = 24 ô).
- `python3 experiments/m4_readiness.py <evidence_dir> --source-commit <40-hex-sha>` — Bộ chấm readiness fail-closed M4.
- `python3 experiments/m4_lossless.py` / `experiments/lossless_verifier.py` — Kiểm chứng greedy token-level lossless.
- `python3 experiments/m4_cost_profiler.py` / `experiments/async_lineage_runtime_probe.py` — Profiler thành phần M4 và probe lineage bất đồng bộ.
- `bash scripts/launch_lane{1_vanilla,2_static,3_adaptive,4_eagle}.sh` và `bash scripts/switch_lane.sh` — Script chạy từng làn cục bộ trên Linux/T4 (legacy M1–M3).

---

## 5. Kiểm thử tự động (`pytest` và `vitest`)

Chạy từ thư mục `do-an-ptit-d22/`:

```bash
# 1. Bộ kiểm thử danh mục 4 GPU Modal, xuất JSON web demo, EAGLE-1/2/3/3.1 và benchmark đa GPU
PYTHONPATH=src:. pytest \
  tests/test_modal_gpu_catalog.py \
  tests/test_modal_session_ux.py \
  tests/test_benchmark_campaign.py \
  tests/test_cli_menu_help.py \
  tests/test_web_cli_sample.py -v

# 2. Bộ kiểm thử hồi quy offline toàn diện (FlexSDS CLI, R6 MLE, M3/M4/M5, EAGLE, Colab)
PYTHONPATH=src:. pytest \
  tests/test_benchmark_campaign.py \
  tests/test_benchmark_profiles.py \
  tests/test_m3_preflight.py \
  tests/test_m5_campaign.py \
  tests/test_modal_gpu_catalog.py \
  tests/test_modal_session_ux.py \
  tests/test_modal_runner_cli.py \
  tests/test_modal_app_container_paths.py \
  tests/test_modal_image.py \
  tests/test_modal_onboarding.py \
  tests/test_matched_execution_contract.py \
  tests/test_r6_geometric_mle_and_telemetry_io.py \
  tests/test_eagle_inspection.py \
  tests/test_speculative_method.py \
  tests/test_hardware_profiler_benchmark.py \
  tests/test_hardware_cost_driven_controller.py \
  tests/test_serving_profile_guard.py \
  tests/test_m4_a100_profile.py \
  tests/test_local_runner.py \
  tests/test_cost_model_solver.py \
  tests/test_profile_import.py -q

# 3. Kiểm thử môi trường có đủ SDK (chạy trong conda env doan-d22)
PYTHONPATH=src:. pytest \
  tests/test_modal_packaging_contract.py \
  tests/test_modal_bench_jobs.py \
  tests/test_modal_runner_config.py \
  tests/test_vram_memory_lifecycle.py \
  tests/test_api_*.py -q

# 4. Kiểm thử Frontend Web Demo
cd web && npm test -- --run
```
