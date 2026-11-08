# Backend web demo [M5-01] — FastAPI trên Render

Luồng: ảnh hoá đơn → OCR (Modal `ptit-ocr-serve`) → Node 1 (Modal `ptit-vllm-demo`, làn theo bản đồ) → Postgres Supabase (schema `demo`) → câu hỏi tiếng Việt → Node 2 SQL (đã kiểm tra) → câu trả lời ngắn không bịa số.
Thiết kế: `openspec/changes/m5-01-web-demo/design.md` (bản local). Frontend: `web/README.md`.

| Module | Việc |
|---|---|
| `main.py` | Nối dịch vụ thật (Postgres, Storage, Modal), `uvicorn src.api.main:build_app --factory` |
| `app.py` | Route, mật khẩu demo `X-Demo-Key`, giới hạn 10 ảnh / 30 câu hỏi / 10 phút / IP, CORS |
| `jobs.py` | Job tải ảnh: OCR → Node 1 (prompt `node1_extractor.v2`, `max_tokens` 3072) → so sánh làn → lưu DB |
| `questions.py` | Prompt Node 2 có giá trị thật của DB, kiểm tra SQL bằng `sqlglot`, sửa 1 lần, kiểm tra số trong câu trả lời |
| `llm.py` | Gọi model qua streaming, đo TTFT/TPOT/token/s |
| `grounding.py` | Liệt kê trường chữ/chuỗi tiền không có trong văn bản OCR |
| `store.py`, `storage.py` | Postgres (`demo_writer` / `demo_reader`) và Supabase Storage (bucket riêng tư, signed URL 5 phút) |
| `warmup.py` | Làm nóng + giữ nóng tối đa 60 phút (ping mỗi 4 phút) |

## Các bước deploy (Dương làm; bí mật chỉ nhập trên dashboard, không ghi vào repo)

1. **Supabase** — tạo project (region Singapore). SQL Editor: chạy `src/pipeline/postgres/schema.sql` rồi `src/pipeline/postgres/roles.sql`, sau đó đặt mật khẩu 2 role:
   `ALTER ROLE demo_writer PASSWORD '...'; ALTER ROLE demo_reader PASSWORD '...';`
   Chuỗi kết nối lấy ở *Connect → Session pooler*, đổi user thành `demo_writer.<project-ref>` / `demo_reader.<project-ref>`.
2. **Nạp dữ liệu mồi** (máy local, biến môi trường chỉ đặt trong terminal):
   `DEMO_DATABASE_URL=<chuỗi demo_writer> SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python scripts/seed_demo_db.py`
   → `{"rows": 29, "written": 29, ...}`; chạy lại không nhân đôi.
3. **Modal** (chi phí GPU, xin phép trước):
   - OCR: `modal deploy -m experiments.modal_ocr_serve`
   - Làn: `PTIT_APP_NAME=ptit-vllm-demo PTIT_GPU=A10G PTIT_MAX_MODEL_LEN=8192 modal deploy -m experiments.modal_runner`
4. **Render** — *New → Blueprint* từ repo, dùng `render.yaml`; nhập các biến `sync: false`: `DEMO_KEY`, `DEMO_FRONTEND_ORIGINS` (URL Vercel), `DEMO_DATABASE_URL`, `DEMO_READER_DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET`.
5. **Vercel** — import repo, Root Directory `web`, biến `NEXT_PUBLIC_API_URL` = URL Render.
6. **Trước buổi demo** — mở web, bấm *Làm nóng hệ thống*, chờ OCR và các làn `sẵn sàng` (khởi động lạnh vài phút). Giữ nóng ≈ 4,4 USD/giờ với OCR + 3 làn A10G (ước tính, cần xác nhận). Supabase miễn phí tạm dừng sau 1 tuần không dùng: kiểm tra trước buổi bảo vệ.

## Chạy ở máy

```bash
# backend (env conda doan-d22); unset SSL_CERT_FILE nếu nó trỏ tới file không tồn tại
DEMO_KEY=... DEMO_DATABASE_URL=... DEMO_READER_DATABASE_URL=... MODAL_PROXY_TOKEN_ID=... MODAL_PROXY_TOKEN_SECRET=... \
  uvicorn src.api.main:build_app --factory --port 8000
# frontend
cd web && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Test

- `pytest tests/test_api_*.py tests/test_demo_*.py tests/test_eval_demo_questions.py tests/test_invoice_rows.py tests/test_node1_request.py -q`
- Test cần Postgres chỉ chạy khi có `DEMO_TEST_DATABASE_URL` (Postgres dùng một lần) và `DEMO_TEST_ALLOW_RESET=1` — chúng xoá và tạo lại schema `demo`. Không bao giờ trỏ vào Supabase.
- Chấm bộ câu hỏi: `python scripts/eval_demo_questions.py --reference-only` (chỉ DB), hoặc không cờ để chạy cả model → `experiments/results/m5_01_demo_questions_eval.json`.
