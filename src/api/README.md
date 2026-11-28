# Backend web demo [M5-01] — FastAPI trên Render

Luồng: ảnh hoá đơn → OCR thư viện (Modal `ptit-ocr-lib`, CPU: PaddleOCR tách dòng + VietOCR, chữ đã bỏ dấu) → Node 1 → Postgres Supabase (schema `demo`); câu hỏi tiếng Việt → Node 2 SQL (đã kiểm tra); yêu cầu báo cáo → Node 3 (báo cáo + dashboard từ số liệu lấy bằng truy vấn cố định). Mọi node chạy trên **endpoint người dùng dán vào** (đầu ra của `flexsds serve`: Modal, Colab hoặc máy cá nhân qua đường hầm cloudflared); backend không dựng sẵn endpoint nào và không bật/tắt GPU.
Thiết kế: `openspec/changes/m5-01-web-console/design.md` (bản local). Frontend: `web/README.md`.

| Module | Việc |
|---|---|
| `main.py` | Nối dịch vụ thật (Postgres, Storage, Modal), `uvicorn src.api.main:build_app --factory` |
| `app.py` | Route, mật khẩu demo `X-Demo-Key`, giới hạn 10 ảnh / 30 câu hỏi / 10 phút / IP, CORS |
| `endpoints.py` | Sổ endpoint: chuẩn hoá địa chỉ, nhận nền tảng, chặn địa chỉ nội bộ (kiểm tra sau DNS), hỏi `/v1/models` lấy id model, báo địa chỉ đã có; sửa địa chỉ (kiểm tra lại trước khi lưu), khoá và ghi chú cấu hình; khoá API chỉ lưu ở DB |
| `jobs.py` | Job tải ảnh: OCR thư viện (`library_ocr_client.py`) → bỏ dấu → Node 1 (prompt `node1_extractor.v2`, `max_tokens` 3072) song song trên các endpoint đã chọn → so với endpoint mốc → lưu DB |
| `questions.py` | Prompt Node 2 có giá trị thật của DB, kiểm tra SQL bằng `sqlglot`, sửa 1 lần, kiểm tra số trong câu trả lời |
| `report_data.py` | Số liệu đầu vào của Node 3: truy vấn cố định (không qua Node 2) theo phạm vi N hoá đơn có ngày lập gần nhất hoặc tất cả |
| `reports.py` | Node 3 độc lập: model chọn 2–4 biểu đồ từ các bảng (JSON, prompt `node3_dashboard_demo.v1`, số luôn lấy từ bảng) rồi viết báo cáo (system prompt W3 v2); kiểm tra số trong báo cáo. Chế độ chuỗi dùng Q1–Q6 qua Node 2 như M3-06 |
| `bench.py` | Đo trực tiếp: (A) request của gói đóng băng W1 `w1_v3` / W2 `w2_v2` / W3 `w3_v2` trên nhiều endpoint cùng lúc, tăng tốc = TPOT p50 mốc / TPOT p50 endpoint; (B) chuỗi Node 1 → 2 → 3 trên từng endpoint. Một lần đo tại một thời điểm |
| `llm.py` | Gọi model qua streaming, đo TTFT/TPOT/token/s |
| `grounding.py` | Liệt kê trường chữ/chuỗi tiền không có trong văn bản OCR |
| `store.py`, `storage.py` | Postgres (`demo_writer` / `demo_reader`) và Supabase Storage (bucket riêng tư, signed URL 5 phút) |

## Các bước deploy (Dương làm; bí mật chỉ nhập trên dashboard, không ghi vào repo)

1. **Supabase** — tạo project (region Singapore). SQL Editor: chạy `src/pipeline/postgres/schema.sql` rồi `src/pipeline/postgres/roles.sql` (chạy lại cả hai mỗi khi schema có bảng mới, ví dụ `endpoints`, `bench_runs`), sau đó đặt mật khẩu 2 role:
   `ALTER ROLE demo_writer PASSWORD '...'; ALTER ROLE demo_reader PASSWORD '...';`
   Chuỗi kết nối lấy ở *Connect → Session pooler*, đổi user thành `demo_writer.<project-ref>` / `demo_reader.<project-ref>`.
2. **Nạp dữ liệu mồi** (máy local, biến môi trường chỉ đặt trong terminal):
   `DEMO_DATABASE_URL=<chuỗi demo_writer> SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python scripts/seed_demo_db.py`
   → `{"rows": 25, "written": 25, ...}` (25 hoá đơn W1_v3); chạy lại không nhân đôi.
3. **Modal** — chỉ OCR (CPU, đã deploy 11/11): `modal deploy -m experiments.modal_ocr_lib`. Model không deploy từ đây: người chạy `flexsds serve` tự bật GPU và dán Endpoint URL vào trang Endpoint của web.
4. **Render** — *New → Blueprint* từ repo, dùng `render.yaml`; nhập các biến `sync: false`: `DEMO_KEY`, `DEMO_FRONTEND_ORIGINS` (URL Vercel), `DEMO_DATABASE_URL`, `DEMO_READER_DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET` (chỉ dùng gọi OCR).
5. **Vercel** — import repo, Root Directory `web`, biến `NEXT_PUBLIC_API_URL` = URL Render.
6. **Trước buổi demo** — chạy `flexsds serve` cho các cấu hình cần so (ví dụ vanilla và adaptive), dán các Endpoint URL vào trang Endpoint, chọn endpoint mốc, bấm *Kiểm tra lại*. Web không thăm dò ngầm để khỏi đánh thức GPU. Khi đo ở trang *Đo và so sánh*, giữ trang mở (Render free ngủ sau 15 phút không có request). Supabase miễn phí tạm dừng sau 1 tuần không dùng: kiểm tra trước buổi bảo vệ.

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
