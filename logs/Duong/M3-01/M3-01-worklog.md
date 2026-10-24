# [M3-01] Worklog Dương — Kết nối Client, Harness và Pipeline với Server Modal L4

> **Task:** [M3-01] Kết nối Client Code & End-to-End Pipeline với Server Modal.com (Trello `oTLgMBr4`)<br>
> **Vai trò:** Dương — Lead 100%<br>
> **Nhánh:** `feat/m3-01-modal-client` (tách từ `origin/develop` @ `194327e`)<br>
> **Thiết kế:** OpenSpec change `m3-01-modal-client` (spec `modal-lane-client`, `remote-benchmark-execution`; design D1–D9) — duyệt 24/10/2026<br>
> **Ngày cập nhật:** 25/10/2026<br>
> **Trạng thái:** Code + chạy thật trên Modal L4 xong; **chưa commit / chưa PR**.

---

## 0. Phạm vi theo Trello

| Checklist | Kết quả |
|---|---|
| Mở endpoint OpenAI-compatible (`/v1/completions`, `/v1/chat/completions`) | Có từ M3-00 (bắt proxy token) |
| `src/client/modal_client.py` stream + logit margin | ✅ Xong, kiểm thật 3 làn |
| "Chuyển launch_lane1/2/3.sh sang gọi Modal" | Hiểu lại (C23): `.sh` là script **khởi động server** cho T4 → giữ nguyên; trên Modal việc này do `modal deploy`. Thêm bảng đối chiếu T4 ⇔ Modal trong runbook §12 |
| Smoke 5 prompt | ✅ 15/15 (5 prompt × 3 làn) |

Mở rộng đã thống nhất (24/10): phục vụ cả benchmark (Đại, M3-05) lẫn pipeline/demo (M3-06) → harness chạy được với server ở xa, benchmark chạy **trong** Modal.

## 1. Vấn đề phát hiện khi khảo sát

`experiments/benchmark_harness.py` giả định client và server cùng máy: `environment_info()` đọc GPU/torch/vLLM **local** và đo VRAM bằng NVML **local**; `EXPECTED_GPU = "Tesla T4"`. Chạy từ laptop tới Modal sẽ ghi môi trường "RTX 5060 Laptop", VRAM của laptop, và TTFT lẫn độ trễ mạng VN → `us-east`.

## 2. Thay đổi

| File | Nội dung |
|---|---|
| `src/client/modal_client.py` (mới) | `ModalLaneClient(lane)`: URL theo làn (không cần tài khoản Modal), token từ env/`.env`, `chat`, `stream` (SSE, `include_usage`), `margins` (`logprobs` + `top_logprobs=2` → top-1 − top-2), tự chờ 503; `load_token()` dùng chung (`experiments/modal_probe.py` gọi lại) |
| `experiments/benchmark_harness.py` | `ENVIRONMENT_PROFILES` (T4 mặc định không đổi + `m3-l4-modal-v1`), `--environment-profile`, `--server-env-json`; chế độ server: môi trường từ `server_env.json`, không đo VRAM local; mọi repeat có `run_started_at_utc`/`run_finished_at_utc`; `PTIT_GIT_COMMIT` cho chạy không có `.git` |
| `experiments/modal_lanes.py` | `build_server_env`, `server_artifact_dir`, `bench_harness_args` (đối số canonical như M2-04), `BENCH_WORKLOADS` |
| `experiments/modal_runner.py` | Container server ghi `server_env.json` + sampler NVML 0.25 s → `vram.jsonl` (commit 10 s); hàm `bench` (CPU, secret `ptit-proxy-token`) + `bench_cell` |
| `experiments/modal_vram_join.py` (mới) | Ghép VRAM theo cửa sổ UTC ± 0.5 s; thiếu mẫu → lỗi; không ghi đè raw; chặn đường dẫn ngoài thư mục làm việc |
| `experiments/modal_smoke.py` (mới) | `client-check`, `smoke` (5 prompt frozen, payload qua `workload_adapter.build_request_payload`) |
| Tests | `test_modal_client.py` (24), `test_benchmark_profiles.py` (15), `test_modal_vram_join.py` (8), `test_modal_smoke.py` (9), bổ sung `test_modal_runner_config.py`; sửa chuỗi escape `r"x\y"` trong `test_modal_probe.py` |
| Hạ tầng | Modal Secret `ptit-proxy-token` (tạo từ `.env` qua file JSON tạm, đã xoá; không in giá trị) |

## 3. Kết quả chạy thật (Modal L4)

### 3.1 Benchmark 1 cell trong Modal — `experiments/runs/m3_01/20261024-m301-vanilla-W1-c1/`
Làn vanilla, W1, c = 1, 1 repeat × 8 request (**non-canonical**, chỉ kiểm đường chạy). Server deploy run-id `20261024-m301-server`.
- 8/8 `success`, `metrics_complete`; `ENVIRONMENT CONTRACT PASS`, profile `m3-l4-modal-v1`, `environment_source: server`.
- Môi trường từ server: `NVIDIA L4`, driver 610.57.04, vLLM 0.30.0, torch 2.13.0+cu130, bf16, `awq_marlin`, revision target ghim.
- `run_vram_peak_mb` = **19 850.62 MB** ghép từ **347** mẫu NVML của server (`vram_source: server_nvml`).
- Workload W1 SHA256 `0de2387c…` (trùng M2-04); commit ghi `194327e` (qua `PTIT_GIT_COMMIT`).
- Server sẵn sàng sau 203.6 s (cold start). TPOT ≈ 19.3 ms (1 cell non-canonical — **không** dùng so với T4).

### 3.2 Client check — `experiments/runs/m3_01/20261025-m301-client/client_check.json`
- Cả 3 làn: chat OK, stream có `usage`; static/adaptive có `metrics.speculative_decoding`.
- **Logit margin chạy được trên cả 3 làn** (kể cả khi bật speculative decoding) — rủi ro lớn nhất của thiết kế không xảy ra. Margin cùng câu gần như trùng giữa các làn (lệch ≤ 0.25 logit).

### 3.3 Smoke — `experiments/runs/m3_01/20261025-m301-smoke/smoke.json`
- **15/15 ok**; W1 JSON hợp lệ 6/6. Request đầu mỗi làn gồm cold start (217–401 s); sau đó TTFT ≈ 0.8–1.2 s (đo từ laptop, gồm mạng — tham khảo).
- **Quan sát (chưa kết luận):** cùng prompt, `temperature=0`, số token sinh khác giữa làn (W1 `m0_04_001`: vanilla 635, static 614, adaptive 635; W3: 505/483/508) → dấu hiệu output không hoàn toàn lossless giữa làn trên L4/bf16, giống hiện tượng M2-03 (45/54 trên T4). Ghi lại làm dữ liệu cho M2-03 trên Modal; logit margin là công cụ để điều tra.

### 3.4 Node 1 qua Modal — `experiments/runs/m3_01/20261025-m301-node1/`
- 5 hoá đơn đầu devset, làn vanilla: `fully_validated_count = 5/5`, `schema_valid_rate = 1.0`, `strict_json_rate = 1.0`, `source_binding_rate = 1.0`, acceptance ≥ 0.95 PASS.
- **Sự cố:** lần chạy đầu fail ở preflight: `HTTP 503 {"error":"no upstreams available"}` — làn đã scale-to-zero và client pipeline (`src/pipeline/openai_compatible.py`) **không chờ 503**. Lần chạy đạt dùng cách làm ấm làn bằng `modal_probe check` trước. **Đã sửa (Dương duyệt 25/10):** `OpenAICompatibleClient` thêm `ready_timeout_seconds` (mặc định 0 = hành vi cũ), `run_node1_extractor.py` thêm `--ready-timeout`; test `tests/test_openai_compatible_ready_wait.py` (5). Runbook §11 dùng `--ready-timeout 1200`.

## 4. Kiểm thử

- Full suite (`conda run -n doan-d22 python -m pytest tests -q`, 25/10): **916 passed, 1 failed** — chỉ `test_t4_rowwise_attention.py::test_review_diff_is_exact_generated_transform` (CRLF trên Windows, đã biết từ trước). Test mới M3-01: 56 test + bổ sung config.
- Phát hiện có sẵn trên `develop`: `python -m experiments.benchmark_harness --help` lỗi `TypeError: not enough arguments for format string` (chuỗi help có `%`) — kiểm trên bản `origin/develop`, không do M3-01; chưa sửa (ngoài phạm vi).

## 5. Vận hành & chi phí

- Chạy lần lượt nhiều làn với `PTIT_SCALEDOWN_SECONDS=180` làm làn trước tắt rồi cold start lại → lần sau dùng ≥ 600 s.
- Warning `Cubin file saved by TritonBundler not found` khi nhiều làn dùng chung `ptit-vllm-cache` — không fail; có thể tách thư mục cache theo làn sau này.
- Chi phí Modal 24–25/10 tổng 2.94 USD (M3-00 ≈ 2.12) → **M3-01 ≈ 0.82 USD**.

## 6. Bàn giao

| Việc | Ai |
|---|---|
| S1: cách chạy `bench_cell` (runbook §10) — dùng `raw.vram.jsonl` cho aggregate | Đại (M3-05) |
| S2: client chung + smoke (runbook §9) | Cả nhóm (M3-06, demo) |
| Client pipeline tự chờ 503 | ✅ Đã làm (`--ready-timeout`) |
| Quan sát lossless giữa làn trên L4 | M2-03 (Khôi/Đại) |
