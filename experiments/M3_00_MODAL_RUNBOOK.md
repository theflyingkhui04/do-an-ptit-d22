# M3-00 Runbook — vLLM 3 làn trên Modal.com (GPU L4)

Hướng dẫn chạy lại hạ tầng [M3-00]. Mọi lệnh đã chạy thật ngày 24/10/2026 (evidence: `experiments/runs/m3_00/`, nhật ký: `logs/Duong/M3-00/M3-00-worklog.md`).

| Thành phần | Giá trị |
|---|---|
| Workspace Modal | `nguyenduonghb940` |
| App | `ptit-vllm` — 3 server `vanilla`, `static`, `adaptive` |
| URL | `https://nguyenduonghb940--ptit-vllm-<làn>.us-east.modal.direct` |
| Runtime | vLLM 0.30.0 · torch 2.13.0+cu130 · CUDA 13.0 · GPU NVIDIA L4 |
| Model | `Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e…` · draft `Qwen/Qwen2.5-0.5B-Instruct@7ae557604a…` |
| Code | `experiments/modal_lanes.py` (cấu hình làn) · `experiments/modal_runner.py` (Modal) · `experiments/modal_probe.py` (client) |

Mọi lệnh chạy **từ gốc repo**. Ví dụ dùng PowerShell; trong Git Bash thay `$env:X="y";` bằng `X=y`.

---

## 0. Chuẩn bị (một lần mỗi máy)

```powershell
conda activate doan-d22                 # env có modal 1.6.2, pytest, torch
pip install modal                       # nếu env chưa có
modal setup                             # mở trình duyệt, đăng nhập Modal
modal profile current                   # phải in: nguyenduonghb940
```

Tạo file `.env` ở gốc repo (đã gitignore — **không commit**, không gửi qua chat):

```
MODAL_PROXY_TOKEN_ID=wk-...
MODAL_PROXY_TOKEN_SECRET=ws-...
```

Token tạo ở Dashboard → Settings → Proxy Tokens (hoặc `modal workspace proxy-tokens create --name <tên>`). Secret chỉ hiện một lần.

Kiểm tra code trước khi tốn GPU:

```powershell
python -m pytest tests/test_modal_runner_config.py tests/test_modal_probe.py -q
```

## 1. Image và weights (chỉ khi làm mới workspace / đổi phiên bản)

```powershell
modal run -m experiments.modal_runner::image_check   # in vllm/torch/GPU, ghi image.json
modal run -m experiments.modal_runner::weights       # tải 2 model vào volume ptit-model-cache
```

- `weights` chạy trên CPU (rẻ); lần sau in `already present` = không tải lại.
- Nó cũng ghi `refs/main` trỏ đúng revision đã ghim — **bắt buộc**, vì server chạy offline và vLLM nạp tokenizer model nháp theo tên repo (xem mục 7).
- Kết quả: `experiments/runs/m3_00/<run-id>/image.json`, `weights_manifest.json`.

## 2. Deploy server

```powershell
# Cả 3 làn, BF16 (mặc định), tự tắt sau 120 s rảnh
$env:PTIT_RUN_ID="20261025-ten-lan-chay"; $env:PTIT_SCALEDOWN_SECONDS="120"
modal deploy -m experiments.modal_runner
```

Biến môi trường lúc deploy:

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `PTIT_RUN_ID` | chuỗi | Thư mục log/telemetry trong volume `ptit-run-artifacts` |
| `PTIT_DTYPE_PROFILE` | `l4-bf16` (mặc định) / `fp16-fallback` | Kiểu dữ liệu |
| `PTIT_LANES` | `vanilla,static,adaptive` (mặc định) | Chỉ deploy một số làn (ví dụ `static`) — làn không liệt kê sẽ bị gỡ |
| `PTIT_SCALEDOWN_SECONDS` | `300` mặc định | Rảnh bao lâu thì tắt container |
| `PTIT_KEEP_WARM` | `1` | Luôn giữ 1 container/làn bật — **chỉ khi benchmark**, tốn tiền liên tục |

> Xoá biến sau khi dùng (`Remove-Item Env:PTIT_KEEP_WARM`) để lần deploy sau không vô tình giữ ấm.

## 3. Kiểm tra server

```powershell
python -m experiments.modal_probe check --run-id 20261025-ten-lan-chay
python -m experiments.modal_probe check --lanes static --run-id ...   # một làn
```

- Lần gọi đầu kích hoạt cold start (Làn 1 ~3.5 phút, Làn 2/3 ~3.5–7 phút); lệnh tự chờ qua `503`.
- Đạt khi mỗi làn có `"ok": true` (health 200, không token → 401, đúng tên model, chat 200). Ghi `check.json`.

Gọi tay bằng client OpenAI:

```python
from openai import OpenAI
from experiments.modal_probe import load_token
client = OpenAI(base_url="https://nguyenduonghb940--ptit-vllm-vanilla.us-east.modal.direct/v1",
                api_key=load_token())
client.chat.completions.create(model="qwen2.5-7b-vanilla",
                               messages=[{"role": "user", "content": "Xin chào"}], max_tokens=16)
```

Tên model: `qwen2.5-7b-vanilla`, `qwen2.5-7b-static-draft`, `qwen2.5-7b-adaptive`.

## 4. Đo cold / warm start

```powershell
# cold: chờ 200 s (> scaledown 120 s) trước mỗi mẫu để container về 0
python -m experiments.modal_probe startup --lanes vanilla --kind cold --repeats 3 --idle-wait 200 --run-id <id>
# warm: chạy ngay sau đó, khi container còn sống
python -m experiments.modal_probe startup --lanes vanilla --kind warm --repeats 3 --run-id <id>
```

Kết quả nối thêm vào `experiments/runs/m3_00/<id>/startup.json` (records + median). Kiểm `observed_cold` = `true` cho mẫu cold.

## 5. Benchmark (giữ ấm) rồi tắt

```powershell
$env:PTIT_KEEP_WARM="1"; $env:PTIT_SCALEDOWN_SECONDS="600"; $env:PTIT_RUN_ID="<id>"
modal deploy -m experiments.modal_runner
python -m experiments.modal_probe check --run-id <id>     # làm ấm cả 3 làn
# ... chạy benchmark trỏ tới URL các làn ...
Remove-Item Env:PTIT_KEEP_WARM; $env:PTIT_SCALEDOWN_SECONDS="120"
modal deploy -m experiments.modal_runner                  # về chế độ tự tắt
```

## 6. Log, telemetry, chi phí

```powershell
modal app logs ptit-vllm                       # log trực tiếp của mọi làn
modal container list                           # container đang chạy (rỗng = không tốn tiền)
modal volume ls ptit-run-artifacts <làn>       # các run-id đã có
modal volume get --force ptit-run-artifacts vanilla/<run-id>/vllm.log experiments/runs/m3_00/<run-id>/vllm_vanilla.log
modal volume get --force ptit-run-artifacts lane3/<run-id>/telemetry.jsonl experiments/runs/m3_00/<run-id>/lane3_telemetry.jsonl
modal billing report --start 2026-10-01 --end 2026-11-01
```

- Log/telemetry chỉ ghi xong vào volume khi container **tắt** — chờ `modal container list` rỗng rồi mới `volume get`.
- Trong **Git Bash**, đường dẫn bắt đầu bằng `/` bị đổi thành đường dẫn Windows: thêm `MSYS_NO_PATHCONV=1` trước lệnh `modal volume ...`.

## 7. Dừng & xử lý sự cố

```powershell
modal app stop ptit-vllm -y        # dừng tất cả (URL ngừng hoạt động tới lần deploy sau)
```

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| Container khởi động lại liên tục, log có `EngineCore failed to start` | **Dừng app ngay** (tốn tiền mỗi lần khởi động), đọc log trong volume rồi mới sửa |
| `OSError: We couldn't connect to 'https://huggingface.co' ... cached files` ở `draft_model.py` | Thiếu `refs/main` trong volume → chạy lại `modal run -m experiments.modal_runner::weights` |
| `503` khi gọi | Đang cold start — chờ, `modal_probe` tự thử lại |
| `401` | Thiếu/sai token trong `.env` |
| Deploy chỉ còn 1 làn | Lần deploy trước đặt `PTIT_LANES` — xoá biến rồi deploy lại |
| `modal_probe` báo không có URL | App chưa deploy hoặc đã `app stop` |
| Benchmark harness từ chối GPU | `experiments/benchmark_harness.py` ghi cứng `EXPECTED_GPU = "Tesla T4"` — cần thêm profile L4 (M3-01/M3-05) |

## 8. Thông số tham khảo (24/10/2026, profile `l4-bf16`)

| Làn | Cold (median, tới phản hồi đầu) | Warm (median) | KV cache |
|---|---|---|---|
| vanilla | 200.9 s | 1.84 s | 242 192 token |
| static | 321.7 s | 2.17 s | 179 040 token |
| adaptive | 221.9 s | 1.77 s | 178 368 token |

L4 ≈ 0.8 USD/giờ; Starter 30 USD credit/tháng (giới hạn chi tiêu workspace 30 USD). Toàn bộ M3-00 ngày 24/10 tốn ≈ 2.08 USD.
