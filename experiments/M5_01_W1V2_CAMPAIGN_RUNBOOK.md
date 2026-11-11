# [M5-01] Runbook — campaign benchmark W1_V3 (4 làn × 6 mức đồng thời, Modal L4)

> **Cập nhật 08/11 — luồng chuẩn:** campaign chạy trên **W1_V3** (`experiments/configs/w1_v3/`, `package_sha256` `5e22c3cb…`): chữ OCR từ thư viện (PaddleOCR tách dòng + VietOCR), đã bỏ dấu. Đây là mặc định của `run_m5_campaign`. W1_V2 (chữ Qwen-VL) chỉ chạy khi cần đối chiếu: thêm `--workload W1_V2`. Mọi mô tả W1_V2 bên dưới áp dụng nguyên cho W1_V3 (cùng ngữ cảnh 8192, timeout 600 s, vLLM vá #55708). Số đo W1_V3 (Node 1 vanilla A10G, `experiments/runs/m5_01/m5_01_v3_vanilla_r1/`): prompt 1.837–2.155 token, đầu ra trung bình ~1.431 token, dài nhất 2.263 token; 25/30 mẫu ra JSON hợp lệ ở mọi làn (5 mẫu lỗi `SOURCE_BINDING_ERROR` giống nhau ở cả 4 làn — vẫn là request đo hợp lệ của benchmark).

Người chạy: **Đại** (luồng B). Bộ công cụ: `experiments/run_m5_campaign.py`. File này mô tả hợp đồng chạy; nó **không** khẳng định campaign đã chạy.

## 1. Đo cái gì

- Workload **W1_V2** (`experiments/configs/w1_v2/`, `package_sha256` `cd048f71…`): Node 1 trích xuất JSON từ văn bản OCR toàn tờ hoá đơn, `max_tokens` 3072, prompt ~1,85–2,16 nghìn token. Đầu ra trung bình ~1.408 token, dài nhất 2.129 token (đo ở lần chạy Node 1 vanilla A10G `experiments/runs/m5_01/m5_01_vanilla_r1/`).
- Lưới: vanilla / static / adaptive / eagle × c = 1, 2, 4, 8, 16, 32 = **24 ô**. Mỗi ô: đúng 3 warm-up + 5 lần × 32 request đo = 160 request; tổng 3.840 request đo.
- Giao thức giống M3-05 (`experiments/M3_05_CAMPAIGN_RUNBOOK.md`): server mới cho mỗi ô (`KEEP_WARM=0`, scaledown 60 s), không chạy lại ô đã hợp lệ để cải thiện số, giữ nguyên ô có tăng tốc < 1, không ghép attempt dở.

## 2. Khác M3-05 ở đâu

| | M3-05 | M5-01 (campaign này) |
|---|---|---|
| Nơi chạy | Google Colab L4 | Modal L4 (`experiments.modal_runner`) |
| Workload | W1, W2, W3 | W1_V2 |
| `--max-model-len` | 4096 | **8192** |
| Timeout mỗi request của harness | 120 s (mặc định) | **600 s** (đầu ra ~2,1 nghìn token ở c=32 sát 120 s) |
| vLLM | 0.30.0 + backport #55708 | **giống hệt**: `vllm/outputs.py` SHA-256 `8f586842…` |

Bằng chứng bản vá trùng byte với Colab: `experiments/results/m5_01_vllm_backport_check.json`.

Các chốt chặn tự động:
- Image Modal tự vá khi deploy với `PTIT_VLLM_RUNTIME=backport-55708` (`experiments/vllm_runtime_patch.py`). Build **thất bại** nếu `outputs.py` trước hoặc sau khi vá không đúng SHA mong đợi.
- `server_env.json` ghi `runtime_variant` và `vllm_outputs_sha256` của file thật trong container.
- `bench_submit` từ chối gửi W1_V2 lên server ngữ cảnh 4096 hoặc vLLM chưa vá.
- `validate_m3.validate_cell` từ chối ô W1_V2 có tham số server khác `build_vllm_args(lane, max_model_len=8192)` hoặc runtime khác bản vá.
- `run_m5_campaign validate` từ chối khi thiếu ô, khi hai ô dùng chung một server, hoặc khi có ô không chạy trên `NVIDIA L4`.

## 3. Chuẩn bị

1. Env conda `doan-d22`. Token Modal của workspace `nguyenduonghb940`: hỏi Dương, đặt bằng `modal token set`. Không ghi token vào file hay log.
2. Checkout **đúng SHA đầy đủ đã review**. `src/` và `experiments/` không được có thay đổi chưa commit (runner tự kiểm và từ chối).
3. Không chạy deploy `ptit-vllm` nào khác cùng lúc: controller dừng app `ptit-vllm` trước mỗi ô.

## 4. Lệnh (PowerShell, từ thư mục gốc repo)

```powershell
$sha = '<SHA_DA_REVIEW>'
python -m pytest tests/test_m5_campaign.py tests/test_m3_preflight.py tests/test_modal_bench_jobs.py tests/test_modal_runner_config.py tests/test_m2_harness_contract.py tests/test_benchmark_profiles.py tests/test_benchmark_workload_adapter.py tests/test_modal_vram_join.py tests/test_aggregate_results.py -q
```

### 4.1. Pilot — 1 ô eagle × c=32 (~10–15 phút), không bao giờ gộp vào kết quả

Ô này khó nhất: đầu ra dài nhất ở tải cao nhất, kiểm tra cùng lúc timeout, KV cache và số liệu speculative.

```powershell
$pilot = 'm501-w1v3-pilot-v1'
python -m experiments.run_m5_campaign plan     --campaign-id $pilot --execution-sha $sha --pilot
python -m experiments.run_m5_campaign run      --campaign-id $pilot --execution-sha $sha --pilot
python -m experiments.run_m5_campaign validate --campaign-id $pilot --execution-sha $sha --pilot
```

Pilot `VALID` thì mới chạy campaign. Pilot hỏng: xem mục 6, sửa xong thì commit và dùng SHA mới **cho cả pilot lẫn campaign**.

### 4.2. Campaign 24 ô

```powershell
$campaign = 'm501-w1v3-confirmatory-v1'
python -m experiments.run_m5_campaign plan   --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m5_campaign run    --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m5_campaign status --campaign-id $campaign --execution-sha $sha
```

`run` đi tuần tự từng ô: dừng app, deploy đúng 1 làn với `PTIT_RUN_ID` riêng, gửi job, theo dõi, lấy kết quả, kiểm tra, dừng app. Máy bị ngắt thì job đang chạy trên Modal vẫn chạy tiếp; chạy lại **đúng lệnh `run`** để tiếp tục: ô đã xong bị bỏ qua, ô đang chạy được theo dõi tiếp, không bao giờ gửi lại.

### 4.3. Sau khi đủ 24 ô

```powershell
python -m experiments.run_m5_campaign validate  --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m5_campaign aggregate --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m5_campaign compare   --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m5_campaign figures   --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m5_campaign stop      --campaign-id $campaign --execution-sha $sha
python -m modal container list --json
```

### Bằng chứng nằm ở đâu

| Nội dung | Đường dẫn |
|---|---|
| Plan (đóng băng, ghi một lần) | `experiments/runs/m5_01/<campaign>/plan.json` |
| Mỗi attempt | `experiments/runs/m3_01/<campaign>-<lane>-W1_V3-c<k>-a<NNN>/` (cùng chỗ với attempt M3) |
| Tổng hợp | `experiments/results/m5_01_w1_v3_sweep_summary.json` |
| So với W1 v1 của M3-05 | `experiments/results/m5_01_w1_v3_vs_m3_05_w1.md` |
| Biểu đồ | `experiments/results/figures/m5_01/F1_…png`, `F2_…png` |

Commit bằng chứng theo quy ước nhóm. Không commit `.env`, token hay file `.sqlite`.

## 5. Thời gian và chi phí (ước tính, cần xác nhận)

- Riêng ô vanilla × c=1: 160 request × ~28–30 s ≈ 80 phút (TPOT vanilla L4 ~19–20 ms ở M3-05 × ~1.400 token).
- Cả campaign khoảng **10–11 giờ** chạy liền, khoảng **9–12 USD** GPU L4 trên workspace Modal của Dương (~0,8 USD/giờ). Pilot khoảng 0,2–0,3 USD.

## 6. Khi gặp lỗi

| Hiện tượng | Làm gì |
|---|---|
| Build image báo `unexpected sha256` | Bản vLLM không còn đúng 0.30.0. Dừng, báo Dương; **không** tắt kiểm tra |
| Ô `FAILED` | Xem `harness.log`/`vllm.log` trong Volume `ptit-run-artifacts` (`bench/<attempt>/`, `<lane>/<run_id>/`), rồi `run ... --retry-failed` (attempt mới, server mới, đủ 160 request) |
| `Spawn outcome unknown` | Làm theo mục *Receipt recovery* của runbook M3-05 (`bench_recover`); không gửi lại khi chưa rõ |
| Request timeout ở c=32 | Không đổi timeout giữa campaign. Báo lại; đổi timeout là campaign mới với ID mới |
| Thiếu `server_metrics` ở làn SD dù đã vá | Giữ nguyên bằng chứng, báo lại; không lọc request |
| `Deployed max_model_len 4096 != 8192` | Đang deploy sai; chỉ chạy qua `run_m5_campaign` (tự đặt `PTIT_MAX_MODEL_LEN=8192`, `PTIT_VLLM_RUNTIME=backport-55708`, `PTIT_GPU=L4`) |

## 7. Diễn giải kết quả

- Chỉ so **tỉ lệ tăng tốc so với vanilla trong từng campaign**. TPOT và throughput tuyệt đối của W1_V2 khác W1 v1 vì ngữ cảnh dài hơn, đầu ra dài hơn và nơi chạy khác (Colab → Modal).
- Không có ngưỡng đạt/không đạt (`engineering_gate = NOT_DEFINED`). Tăng tốc < 1 là kết quả hợp lệ, phải giữ.
- Ghi nơi chạy khác M3-05 như một sai khác giao thức đã biết (cùng loại GPU, cùng phiên bản và bản vá vLLM, cùng giao thức).
