# 3.7. Sản phẩm Minh hoạ: Web Console FlexSDS Benchmark (M5-01)

> **Trạng thái:** bản nháp ngày 23/11/2026, viết theo mã nguồn trên `develop` tại merge commit `229b3e4` (PR #46 ngày 15/11 và PR #51 ngày 23/11).
> **Người viết:** Dương. Mục này bổ sung cho Chương 3 ([dàn ý](08-outline-quyen.md)); hai quyết định thiết kế đi kèm là **QĐ-18** và **QĐ-19** trong [03](03-quyet-dinh-thiet-ke.md), hiện ở trạng thái *đề xuất, chờ Khôi duyệt*.
> **Quy ước bằng chứng:** **[Đã xác minh]** có file, test hoặc commit kèm theo; **[Quan sát]** thấy được khi chạy thật nhưng log chưa lưu trong repo; **[Đề xuất]** chưa được nhóm phê duyệt.

---

## 3.7.1. Mục đích và Phạm vi

Theo [QĐ-08](03-quyet-dinh-thiet-ke.md), pipeline hoá đơn ba node là **dụng cụ đo**, không phải domain chính của đề tài. Web console FlexSDS Benchmark là hình thức trình bày dụng cụ đo đó cho hội đồng: cùng một ảnh hoá đơn, cùng một câu hỏi hay cùng một yêu cầu báo cáo được gửi đồng thời tới nhiều endpoint phục vụ khác nhau (ví dụ vanilla và adaptive), và kết quả của mỗi endpoint được đặt cạnh **endpoint mốc** để so cả chất lượng lẫn tốc độ.

Web console **làm**:

- Chạy Node 1 (trích xuất), Node 2 (truy vấn SQL) và Node 3 (báo cáo, dashboard) trên các endpoint người dùng chọn, luôn kèm endpoint mốc.
- Đo tốc độ trực tiếp trên các gói workload đã đóng băng và hiển thị lại số chính thức của campaign M3-05.
- Quản lý danh sách endpoint do CLI `flexsds serve` tạo ra.

Web console **không làm** (có chủ đích, xem QĐ-19):

- Không bật, tắt hay giữ GPU; không có endpoint dựng sẵn.
- Không thay thế số liệu chính thức: số đo trên web là đo trực tiếp, có nhiễu, và trang "Đo và so sánh" ghi rõ điều này.
- Không phải sản phẩm nghiệp vụ kế toán: dữ liệu là 25 hoá đơn của bộ devset W1_v3 cùng các ảnh người dùng tải lên.

## 3.7.2. Kiến trúc Triển khai

Hệ thống tách thành bốn tầng, mỗi tầng chạy trên nền tảng phù hợp nhất với nó **[Đã xác minh]**:

```
Trình duyệt ──HTTPS──► Next.js 16 (Vercel)                     web/
                          │  fetch JSON + header X-Demo-Key
                          ▼
                     FastAPI (Render)                           src/api/
          ┌───────────────┼──────────────────────┐
          ▼               ▼                      ▼
  Postgres + Storage   OCR thư viện (Modal,   Endpoint LLM tương thích OpenAI
  (Supabase,           CPU, ptit-ocr-lib)     do `flexsds serve` tạo
  schema `demo`)                              (Modal, Colab hoặc máy riêng)
```

| Tầng | Nền tảng | Vai trò | Mã nguồn |
|---|---|---|---|
| Giao diện | Vercel (Next.js 16.4.6) | Năm trang: Node 1, Node 2, Node 3, Đo và so sánh, Endpoint | `web/` |
| Điều phối | Render (FastAPI) | Gọi OCR và các endpoint, kiểm tra kết quả, ghi CSDL, đo tốc độ | `src/api/`, `render.yaml` |
| Dữ liệu | Supabase (Postgres, Storage, vùng Singapore) | Hoá đơn, job, endpoint, lần đo; ảnh trong bucket riêng tư | `src/pipeline/postgres/schema.sql`, `roles.sql` |
| Tính toán | Modal | OCR thư viện (PaddleOCR tách dòng + VietOCR) và các endpoint vLLM | `experiments/modal_ocr_lib.py`, `experiments/modal_runner.py` |

Lý do chọn kiến trúc này thay cho Streamlit/Gradio như kế hoạch ban đầu trong [04](04-quy-trinh-va-ke-hoach.md) được trình bày ở QĐ-18. Điểm cốt lõi: hội đồng chỉ cần một đường dẫn trình duyệt (`https://flexsds-benchmark.vercel.app`), còn GPU chỉ được cấp phát ở tầng tính toán khi có endpoint chạy.

Các route API chính **[Đã xác minh, `src/api/app.py`]**: `POST /api/invoices` và `GET /api/jobs/{id}` (Node 1), `POST /api/questions` (Node 2), `POST /api/reports` (Node 3), `POST /api/bench` và `GET /api/bench/{id}` (đo), cùng nhóm `/api/endpoints` (thêm, phát hiện model, kiểm tra, sửa, xoá).

## 3.7.3. Luồng Xử lý của Từng Trang

**Node 1 — Trích xuất hoá đơn.** Người dùng chọn ảnh (JPEG/PNG, tối đa 10 MB) hoặc một ảnh mẫu, chọn các LLM rồi bấm "Bắt đầu". Backend gọi OCR thư viện, bỏ dấu, rồi chạy Node 1 (prompt `node1_extractor.v2`, `max_tokens` 3072) song song trên các endpoint đã chọn và endpoint mốc. Kết quả hiện thành bảng: mỗi cột là một endpoint, ô nào khác mốc được tô màu, có kết luận "giống mốc x/y trường" và hệ số tốc độ theo TPOT. Mỗi job có log theo từng phần (OCR, từng endpoint), chỉ mở khi người dùng bấm **[Đã xác minh, `src/api/jobs.py`, `src/api/job_log.py`, `web/src/components/Node1Page.tsx`]**.

**Node 2 — Truy vấn SQL.** Câu hỏi tiếng Việt được chuyển thành SQL. SQL được kiểm tra bằng `sqlglot` trước khi chạy (chỉ đọc, chỉ trên bốn view được phép), sửa tối đa một lần, và chạy qua kết nối của role chỉ đọc `demo_reader` (`PostgresStore.run_readonly`). Câu trả lời hiện nguyên văn kèm các cờ: số không có trong kết quả truy vấn, chữ không phải tiếng Việt **[Đã xác minh, `src/api/questions.py`, `roles.sql`]**.

**Node 3 — Báo cáo và dashboard.** Số liệu đầu vào lấy bằng các truy vấn cố định, không qua Node 2. Model chọn 2–4 biểu đồ từ các bảng (số luôn lấy từ bảng, không do model viết) rồi viết báo cáo; mọi con số trong báo cáo được đối chiếu với bảng số liệu, số không khớp được đánh dấu ngay trong câu **[Đã xác minh, `src/api/report_data.py`, `src/api/reports.py`]**.

**Đo và so sánh.** Hai chế độ **[Đã xác minh, `src/api/bench.py`, `web/src/components/ResultPage.tsx`]**:

- *Đo tốc độ* trên request của gói đóng băng W1 (`w1_v3`), W2 (`w2_v2`) hoặc W3 (`w3_v2`). Mức "Nhanh" chạy 1 và 8 người dùng đồng thời, 8 request mỗi mức; mức "Đầy đủ" chạy 1, 2, 4, 8, 16, 32 người dùng, 32 request mỗi mức, lặp 3 lần. Hệ số tăng tốc = TPOT p50 của mốc / TPOT p50 của endpoint.
- *Chạy cả chuỗi* Node 1 → 2 → 3 trên từng endpoint và so từng node với mốc.

Bên cạnh số đo trực tiếp, trang hiển thị lại số chính thức của campaign M3-05 (GPU L4, 6 mức người dùng, 32 request mỗi mức, lặp 5 lần, trung vị p50) để người xem phân biệt số chính thức với số đo trên web.

**Endpoint.** Xem mục 3.7.4.

## 3.7.4. Quản lý Endpoint và Vấn đề Khởi động Nguội

**Thêm endpoint.** Người dùng dán địa chỉ API mà `flexsds serve` in ra sau dòng "Endpoint LLM created" (có thể dán nguyên dòng `- API: https://…/v1`, web tự lọc lấy địa chỉ). Backend chuẩn hoá địa chỉ, **chặn địa chỉ nội bộ** (kiểm tra sau khi phân giải DNS), gọi `GET /v1/models` để lấy tên model và độ dài ngữ cảnh, báo nếu địa chỉ đã có. Tối đa 12 endpoint **[Đã xác minh, `src/api/endpoints.py`, `tests/test_api_endpoint_registry.py`]**.

**Endpoint mốc.** Mỗi lần chỉ có một mốc, chọn bằng cột radio "Mốc" ở đầu mỗi dòng trên trang Endpoint, hoặc bỏ chọn bằng nút "Bỏ chọn mốc". Khi không có mốc, các trang chỉ hiện số tuyệt đối (thời gian, TTFT, TPOT), không hiện hệ số tăng tốc **[Đã xác minh, commit `c647774`]**.

**Khởi động nguội.** Endpoint trên Modal tự tắt GPU khi không được dùng (5 phút với các lane `ptit-vllm-demo`). Lần gọi đầu tiên sau đó phải chờ Modal cấp GPU và vLLM nạp mô hình; trong thời gian này Modal trả HTTP 503. Nếu coi 503 là "đã dừng" thì người xem sẽ hiểu sai. Vì vậy hệ thống có ba trạng thái **[Đã xác minh, commit `94019d7`, `4eaa0b8`]**:

| Trạng thái | Điều kiện | Web hiển thị |
|---|---|---|
| `ok` | `/v1/models` trả lời | "Đang chạy" |
| `starting` | HTTP 503, hết thời gian đọc, hoặc mất kết nối giữa chừng | "Đang khởi động GPU", tự kiểm tra lại mỗi 20 s, tối đa 5 phút |
| `down` | Không kết nối được | "Đã dừng" (chấm đỏ ở mục Lựa chọn LLM) |

Web **không thăm dò ngầm**: việc kiểm tra lại chỉ chạy khi người dùng đã bấm "Kiểm tra" và đang mở trang, để không tự đánh thức GPU (tốn tiền) khi không ai xem.

Thời gian khởi động đo được trên trình duyệt **[Quan sát, 19/11/2026]**: lane vanilla lên sau khoảng 142 s, lane adaptive sau khoảng 256 s. Bộ theo dõi container của Modal cho thấy phần lớn thời gian là chờ cấp GPU; sau khi có GPU, mỗi lane chỉ mất thêm khoảng 20–30 s. Con số 256 s khá sát giới hạn 5 phút, nên giới hạn này có thể cần nâng **[Đề xuất]**.

**Endpoint do CLI tạo.** Ngày 23/11/2026, CLI `flexsds` trên `develop` (`ec8ae65`) deploy cấu hình adaptive (Qwen2.5-7B-Instruct-AWQ + EAGLE-3.1, Modal A10G); endpoint sẵn sàng sau 280 s. Dán dòng `- API: …/v1` vào web thì web nhận đúng model `qwen2.5-7b-adaptive` và báo "Đang chạy" **[Quan sát]**. Endpoint thử nghiệm đã được xoá và app trên Modal đã dừng sau khi kiểm tra.

## 3.7.5. Bảo mật và Quyền Truy cập

Các lớp bảo vệ hiện có **[Đã xác minh, `src/api/app.py`, `src/api/settings.py`, `roles.sql`, `tests/test_api_security.py`]**:

- **Mật khẩu demo** (`X-Demo-Key`) cho mọi route trừ `GET /api/status`; giới hạn 10 ảnh, 30 câu hỏi trên mỗi IP trong 10 phút.
- **CORS** chỉ cho các tên miền khai báo trong `DEMO_FRONTEND_ORIGINS`.
- **Tách quyền CSDL:** API ghi bằng role `demo_writer`; SQL do Node 2 sinh chạy bằng role chỉ đọc `demo_reader`.
- **Ảnh** nằm trong bucket riêng tư, chỉ xem được qua signed URL hết hạn sau 5 phút.
- **Không có bí mật trong frontend:** test `no secret in the frontend` và lệnh `npm run check:bundle` chặn tên các khoá backend (Modal, Supabase, chuỗi kết nối) trong mã nguồn và bundle.

**Quyết định cho giai đoạn bảo vệ (23/11/2026):** để hội đồng mở đường dẫn là vào thẳng màn chính, mật khẩu demo được đặt sẵn lúc build qua biến `NEXT_PUBLIC_ACCESS_KEY` trên Vercel (không có trong repo). Hệ quả đã được chủ sở hữu chấp nhận: mật khẩu nằm công khai trong bundle, ai mở trang cũng dùng được backend. Nếu backend từ chối mật khẩu này, màn nhập mật khẩu tự hiện lại **[Đã xác minh, commit `563ace8`]**. Sau buổi bảo vệ cần đổi `DEMO_KEY` trên Render **[Đề xuất]**.

## 3.7.6. Kiểm chứng

**Kiểm thử tự động** (tại thời điểm merge PR #51, ngày 23/11/2026) **[Đã xác minh]**:

| Phạm vi | Kết quả |
|---|---|
| Test Python của API và kho dữ liệu, chạy với Postgres thật (`tests/test_api_*.py`, `tests/test_demo_store.py`) | 171 passed |
| Test giao diện (Vitest + Testing Library, `web/src/__tests__/`) | 80 passed |
| `npm run lint`, `npm run build`, `npm run check:bundle` | Đạt |
| SonarCloud quality gate cho PR #51 | Đạt (lần chấm đầu không đạt ở độ tin cậy, đã sửa ở commit `053005e`) |

**Chất lượng các node** đo bằng các script đánh giá trên cùng mã nguồn backend, ngày 14/11/2026 **[Đã xác minh]**:

| Node | Kết quả | File |
|---|---|---|
| Node 1 trên W1_v3 (30 hoá đơn) | JSON hợp lệ 30/30 ở cả vanilla và adaptive (bản trước 25/30); người bán 26/30, tổng tiền 29/30 (vanilla) và 28/30 (adaptive) | `experiments/results/m5_01_v3_restore_entity_match.md` |
| Node 2, bộ 15 câu hỏi khoá | 15/15 đúng, 0 câu bịa số, 0 câu lẫn chữ lạ, 0 lệnh ghi | `experiments/results/m5_01_canonical_questions_eval_v3.json` |
| Node 3, 3 yêu cầu báo cáo | 198 con số trong báo cáo, 0 số không có trong bảng số liệu | `experiments/results/m5_01_node3_request_remeasure.json` |

**Chạy thật trên hệ thống đã deploy** **[Quan sát]**: tự kiểm tra lại khi GPU khởi động (19/11); chọn và bỏ mốc qua API thật ở các độ rộng màn hình 1440, 900 và 400 px, không tràn ngang (23/11); thêm endpoint do CLI tạo (23/11); mở trang từ trình duyệt mới vào thẳng màn chính và tải được danh sách endpoint (23/11).

## 3.7.7. Hạn chế

1. **Thời gian khởi động nguội** 2–5 phút trên Modal; người trình bày cần bấm "Kiểm tra" vài phút trước khi demo. Render bản miễn phí cũng ngủ khi không có request (khoảng 50 s để thức dậy theo thông báo của Render).
2. **Số đo trên web có nhiễu**: một vùng mạng, một lần chạy, chung hạ tầng với các tác vụ khác; chỉ để minh hoạ, không thay số chính thức M3-05.
3. **Mật khẩu demo công khai** trong giai đoạn bảo vệ (mục 3.7.5).
4. **Endpoint do CLI tạo trên Modal không có xác thực**: ai có địa chỉ cũng gọi được GPU.
5. **Phiên bản CLI:** bản PyPI 0.3.4 deploy cấu hình adaptive lên Modal bị lỗi thiếu module `src.server` trong container; bản trên `develop` đã sửa nhưng chưa có tag phát hành **[Quan sát, 21/11 và 23/11]**.

## 3.7.8. Tóm tắt

Web console biến pipeline hoá đơn ba node — dụng cụ đo của đề tài — thành một công cụ so sánh trực quan: mọi kết quả đều đặt cạnh endpoint mốc, mọi con số do model viết đều được đối chiếu với dữ liệu, và các endpoint là chính những endpoint do CLI `flexsds serve` tạo ra. Kiến trúc tách giao diện, điều phối, dữ liệu và tính toán giúp hội đồng chỉ cần một đường dẫn, trong khi GPU chỉ chạy khi thực sự được dùng. Cái giá phải trả là thời gian khởi động nguội và độ nhiễu của số đo trực tiếp, cả hai đều được ghi rõ ngay trên giao diện.
