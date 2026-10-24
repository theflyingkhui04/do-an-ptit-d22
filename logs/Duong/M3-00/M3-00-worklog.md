# [M3-00] Worklog Dương — Serverless vLLM Runner trên Modal.com (GPU L4)

> **Task:** [M3-00] Xây dựng Hạ tầng Serverless vLLM Runner trên Modal.com (Trello `d6tuJNZZ`)<br>
> **Vai trò:** Dương — Lead 100%<br>
> **Nhánh:** `feat/m3-00-modal-runner` (tách từ `origin/develop` @ `0b13ae8`)<br>
> **Thiết kế:** OpenSpec change `m3-00-modal-runner` (proposal, spec `serverless-inference-runner`, design D1–D11, tasks) — duyệt 24/10/2026<br>
> **Ngày cập nhật:** 24/10/2026<br>
> **Trạng thái:** Code + chạy thật trên Modal L4 xong (3 làn `check` OK, đo cold/warm xong). **Chưa commit / chưa PR**; chờ đoạn văn quyển và duyệt git.

---

## 0. Phạm vi theo Trello

| Checklist | Hạn | Trạng thái |
|---|---|---|
| [Dương] Khởi tạo Modal app, cấu hình container Image (CUDA 13.0, PyTorch 2.13.0+cu130, vLLM 0.30.0) | 22/10 | Đang làm |
| [Dương] Cấu hình modal.Volume (ptit-model-cache) và script sync weights Qwen2.5-7B-AWQ & 0.5B | 23/10 | Chưa |
| [Dương] Lập trình experiments/modal_runner.py với GPU L4, FlashAttention-2 và native BF16 | 24/10 | Chưa |
| [Dương] Kiểm thử cold start (< 15s) và warm start (< 2s) | 24/10 | Chưa |

Card cập nhật 24/10: dòng cấu hình đổi từ "CUDA 12.4, PyTorch 2.5, vLLM 0.6+" sang "CUDA 13.0, PyTorch 2.13.0+cu130, vLLM 0.30.0" (lý do: vLLM 0.6–0.7 không có `vllm.v1.core.sched.async_scheduler.AsyncScheduler`, Làn 3 không chạy được — kiểm trên tag v0.6.4, v0.6.6, v0.7.3 của repo vLLM).

---

## 1. Tiền đề (task 1.2)

| Kiểm tra | Kết quả 24/10/2026 |
|---|---|
| `modal profile current` | `nguyenduonghb940` |
| `git check-ignore .env` | `.env` (proxy token chỉ nằm ở `.env`, không commit) |
| Gói / credit còn lại (Dashboard → Usage) | Credits **$30.00 left / $0.00 used**; Workspace usage limit **$30.00** (chặn chi vượt credit); kỳ 01–31/10/2026. `modal billing report` chỉ trả header (chưa phát sinh chi phí) |
| Môi trường dev | conda `doan-d22` (Python 3.11, `modal` 1.6.2, torch 2.11.0+cu128) — full test 804 pass, 3 fail Windows đã biết |

---

## 2. Nhật ký triển khai

### 24/10 — Nhóm 1–2: branch + cấu hình làn (không dùng GPU)

- Tạo nhánh `feat/m3-00-modal-runner` từ `origin/develop` @ `0b13ae8` (không track `develop`).
- TDD: viết `tests/test_modal_runner_config.py` trước (fail: module chưa có) → viết `experiments/modal_lanes.py` → **19/19 pass**.
- Test so trực tiếp với `scripts/launch_lane*.sh` (đọc file `.sh`): model, `served-model-name`, `generation-config`, `gpu-memory-utilization 0.85`, `max-model-len 4096`, `max-num-seqs 32`, `SPEC_CONFIG` Làn 2/3, `--scheduler-cls`, giá trị mặc định `ADAPTIVE_*`. Chỉ `--dtype`/`--quantization` khác theo profile: `l4-bf16` (`bfloat16` + `awq_marlin`, mặc định) và `fp16-fallback` (`half` + `awq`, y hệt T4).
- Điều chỉnh so với design: cấu hình làn tách thành `experiments/modal_lanes.py` (không import `modal`) để test chạy được ở mọi môi trường; design đã cập nhật.
- Full suite (`conda run -n doan-d22 python -m pytest tests -q`): **823 passed, 3 failed** — 3 lỗi Windows đã biết (thiếu `dirname` trong bash, CRLF), giống hệt trước thay đổi.

### 24/10 — Nhóm 3: image, weights (Modal, GPU L4)

- `experiments/modal_runner.py`: App `ptit-vllm`, image `nvidia/cuda:13.0.3-devel-ubuntu22.04` + Python 3.11 + `vllm==0.30.0` (uv), mount `src/` → `/root/project/src` và `experiments/modal_lanes.py`; 3 Volume `ptit-model-cache`, `ptit-vllm-cache`, `ptit-run-artifacts`. Chạy dạng module: `modal run|deploy -m experiments.modal_runner` (Modal 1.x không tự mount module được import).
- **3.1 image check** (`modal run -m experiments.modal_runner::image_check`, run-id `20261024-m300-setup`): `vllm 0.30.0`, `torch 2.13.0+cu130`, CUDA `13.0`, GPU **`NVIDIA L4`**, Python 3.11.5 → `experiments/runs/m3_00/20261024-m300-setup/image.json`. Build image (cài vLLM) ~30 s. *Ghi chú:* trường `commit` = `0b13ae8` vì code M3-00 chưa commit lúc chạy.
- **3.2 weights** (`::weights`): `Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e9394b818fdfca67ab2a00ecc7dd641` (5 582 400 357 B), `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775` (999 604 126 B). Revision đã ghim vào `experiments/modal_lanes.py`. Chạy lại (run-id `20261024-m300-weights-rerun`): cả hai **"already present"** → không tải lại. Server đặt `HF_HUB_OFFLINE=1` nên chỉ đọc từ Volume.
- Test cấu hình + probe: **36/36 pass** (test so `--speculative-config` bỏ qua khoá `revision` — launcher T4 cũng thêm khoá này khi đặt `PROTOCOL_B_DRAFT_REVISION`).
- **3.3 Làn 1** (`PTIT_LANES=vanilla`, scaledown 120 s): `fp16-fallback` rồi `l4-bf16` đều `check` OK (health 200, không token → **401**, `/v1/models` = `qwen2.5-7b-vanilla`, chat 200). Cấu hình thực dùng đọc từ log vLLM (`deploy.json`):

  | Profile | dtype thực | quantization | Linear kernel | Attention | init engine | compile |
  |---|---|---|---|---|---|---|
  | fp16-fallback | `torch.float16` | `auto_awq` | `MarlinLinearKernel` | `FLASH_ATTN` | 111.6 s | 22.4 s |
  | l4-bf16 | `torch.bfloat16` | `auto_awq` | `MarlinLinearKernel` | `FLASH_ATTN` | 114.6 s | 24.1 s |

  **Phát hiện:** vLLM 0.30.0 trên L4 ánh xạ cả `--quantization awq` lẫn `awq_marlin` sang `auto_awq` + **kernel Marlin** → profile fp16 trên L4 **không** còn giống kernel T4. Attention backend `FLASH_ATTN` (FlashAttention-2) — đáp ứng checklist card. Chốt profile **`l4-bf16`** cho 3 làn. *(Sync S1: cần báo Khôi cho M3-02.)*

### 24/10 — Sự cố: Làn 2/3 không khởi động (đã sửa)

- **Triệu chứng:** deploy 3 làn → Làn 2, 3 crash-loop: `OSError: We couldn't connect to 'https://huggingface.co' ... couldn't find them in the cached files` tại `vllm/v1/spec_decode/draft_model.py:45` (`get_tokenizer(spec.draft_model_config.model, ...)`). Làn 1 không bị.
- **Nguyên nhân gốc (đã xác minh):** khi `use_heterogeneous_vocab`, vLLM 0.30.0 nạp tokenizer model nháp **theo repo id, không truyền revision** → transformers dùng `"main"`. Server chạy `HF_HUB_OFFLINE=1`; `download_weights` tải theo commit hash nên cache **không có `refs/main`** (kiểm `modal volume ls` → không có thư mục `refs`) → không resolve được offline. Trên T4 không gặp vì có mạng.
- **Sửa:** `modal_lanes.pin_main_ref()` ghi `refs/main = <sha đã ghim>`; `download_weights` gọi sau mỗi model. Test tái hiện lỗi offline (`try_to_load_from_cache(..., revision="main")` thất bại khi thiếu ref) + test sửa (resolve đúng snapshot ghim) + test thiếu snapshot → 39/39 pass. Chạy lại `::weights` (run-id `20261024-m300-pin-main-ref`) → volume có `refs/main` cho cả 2 model.
- **Phương án loại:** bỏ `HF_HUB_OFFLINE` (tokenizer có thể lấy `main` mới hơn bản ghim → lệch phiên bản); đổi draft sang đường dẫn local (lệch cấu hình launcher).
- **Chi phí sự cố:** crash-loop chạy ~20 phút trước khi phát hiện → đã `modal app stop`. **Biện pháp:** theo dõi log liên tục (lọc lỗi + mốc, tự `modal app stop` khi engine fail ≥ 2 lần; khử trùng dòng vì `modal app logs` phát lại lịch sử khi kết nối lại) — hiện là script tạm, đề xuất đưa vào repo (`modal_probe watch`).

### 24/10 — Nhóm 4: Làn 2, Làn 3, check 3 làn

- **4.1 Làn 2** (run-id `20261024-m300-lane2-bf16`): `check` OK; request stream như harness (`stream_options.include_usage`) → chunk usage có `metrics.speculative_decoding` với `num_spec_tokens = 3`, `draft_acceptance_rate` 0.537, `mean_acceptance_length` 2.61 (`spec_metrics.json`).
- **4.2 Làn 3** (run-id `20261024-m300-lane3-bf16`): `check` OK, scheduler `AdaptiveAsyncScheduler` khởi động không lỗi. Sau khi container về 0, `lane3/<run-id>/telemetry.jsonl` vẫn đọc được từ volume `ptit-run-artifacts` (2 dòng, có `request_id`, `tree_budgets`); request thứ 2 α_ema 0.53 → `tree_budgets` hạ 3→2→0 (controller phản ứng đúng). Lưu `lane3_telemetry.jsonl`.
- **4.3 3 làn cùng lúc** (run-id `20261024-m300-3lanes-check`): cả 3 `ok: true`.
- **Thời gian khởi tạo engine (log vLLM):**

  | Làn | Lần đầu (init / compile) | Có cache biên dịch (init / compile) | KV cache (lần 2) |
  |---|---|---|---|
  | vanilla | 114.6 s / 24.1 s | 126.3 s / 2.6 s | 242 192 token |
  | static | 369.5 s / 57.6 s | 365.5 s / 8.9 s | 179 040 token |
  | adaptive | 377.3 s / 59.4 s | 460.8 s / 16.8 s | 178 368 token |

  → Cache biên dịch giảm compile 5–10× nhưng **không giảm tổng**: nút thắt là profile/CUDA graph/warmup (nhất là có model nháp). Mục tiêu cold < 15 s của card **khó đạt** với cấu hình hiện tại; hướng tiếp theo nếu cần: Modal memory snapshot hoặc giữ ấm khi benchmark.
- **Chi phí đến 12:54 UTC:** 1.27 USD (gồm crash-loop) — `modal billing report --start 2026-10-24 --end 2026-10-25`.

### 24/10 — Nhóm 5: đo cold/warm start (run-id `20261024-m300-startup`)

- Lệnh: với từng làn, `python -m experiments.modal_probe startup --lanes <làn> --kind cold --repeats 3 --idle-wait 200` (chờ > scaledown 120 s để container về 0) rồi `--kind warm --repeats 3`. Profile `l4-bf16`, app deploy run-id `20261024-m300-3lanes-check`. 12:53–14:02 UTC.
- **18/18 bản ghi `status: ok`**; mọi mẫu cold có `observed_cold = true` (lần gọi `/health` đầu không phải 200), mọi mẫu warm `false`. Đồng hồ phía client (`time.monotonic`), gồm cả thời gian mạng VN → `us-east`.
- `first_response_seconds` = từ lúc bắt đầu tới khi nhận phản hồi chat 1 token:

  | Làn | Cold — median (min–max) | Warm — median (min–max) | Mục tiêu card (đề xuất – cần xác nhận) |
  |---|---|---|---|
  | vanilla | **200.9 s** (200.6–226.1) | **1.84 s** (1.73–2.78) | cold < 15 s ❌ · warm < 2 s ✅ |
  | static | **321.7 s** (288.4–406.7) | **2.17 s** (1.80–3.36) | cold ❌ · warm ❌ (vượt 0.17 s) |
  | adaptive | **221.9 s** (207.0–242.4) | **1.77 s** (1.64–1.78) | cold ❌ · warm ✅ |

- **Diễn giải:** cold start bị chi phối bởi khởi tạo engine vLLM (log: init engine 128–150 s cho vanilla; 135–325 s cho làn có draft) + cấp container/mount volume; cache biên dịch đã có nhưng không đủ. Ngưỡng 15 s của card không đạt với kiến trúc "scale-to-zero" hiện tại. **Đề xuất:** khi benchmark (M3-05) deploy với `PTIT_KEEP_WARM=1` để luôn ở chế độ warm (~1.8 s), tắt lại sau khi đo; tối ưu cold start (Modal memory snapshot) để ngoài phạm vi M3-00 nếu nhóm thấy cần.
- Evidence: `experiments/runs/m3_00/20261024-m300-startup/startup.json` (records + summary), log 3 làn + telemetry Làn 3 trong `.../20261024-m300-3lanes-check/`.
- Quét secret: không có chuỗi token `wk-…`/`ws-…` trong `experiments/runs/m3_00`, code M3-00, test, worklog.
- **Chi phí ngày 24/10 (tổng cộng):** ≈ **2.08 USD** (6 app run; lớn nhất 1.43 USD là app deploy chính gồm 18 mẫu đo) → credit còn ≈ 27.9 / 30 USD.

---

## 3. Đề xuất QĐ-14 (chờ Khôi đưa vào `docs/03`, không tự sửa)

> **QĐ-14 — Modal.com GPU L4 là nền tảng thực nghiệm từ Sprint 4.** Runtime giữ vLLM 0.30.0 / torch 2.13.0+cu130 / CUDA 13.0 (Làn 3 cần `AsyncScheduler`). Profile `l4-bf16`: dtype bfloat16, vLLM tự dùng kernel AWQ Marlin + FlashAttention. Số liệu L4 là **bộ mới**, không trộn với số T4 trong cùng bảng speedup; T4 (M1/M2) giữ làm mốc "GPU phổ thông thế hệ cũ". Bổ sung QĐ-02/QĐ-09, không thay thế.

## 4. Còn lại / bàn giao

| Việc | Ai | Ghi chú |
|---|---|---|
| Báo Khôi cấu hình thực dùng trên L4 (S1) và gửi URL 3 làn + cách dùng token (S2) | Dương | Không gửi token qua kênh công khai |
| Mời Khôi/Đại vào workspace `nguyenduonghb940` (Starter 3 seat) | Dương + nhóm | Cần cho M3-02/M3-05 và volume EAGLE (C28) |
| Harness ghi cứng `EXPECTED_GPU = "Tesla T4"` (`experiments/benchmark_harness.py:58`) | M3-01 / M3-05 | Phải thêm profile L4 trước khi đo trên Modal |
| Đưa bộ theo dõi log + tự dừng crash-loop vào repo (`modal_probe watch`) | Dương (đề xuất) | Hiện là script tạm |
| Xoay proxy token sau sprint | Dương | Token cũ đã xuất hiện trong lịch sử chat |
| 1–2 đoạn quyển (hạ tầng L4/Modal + bảng cold/warm) | Dương | Chờ chốt vị trí file trong `docs/` |
