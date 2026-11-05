# Worklog — [M3-06] Pipeline Node 2/3 VAT + E2E qua Modal

- Trello: [M3-06] `DPsomkjN` — Dương (Lead 70%) / Đại (Partner 30%).
- Branch: `feat/m3-06-node2-node3-pipeline` từ `origin/develop` @ `90158a6` (03/11/2026).

## 1. Mục tiêu

Theo DoD của card: Node 2 Text-to-SQL và Node 3 Báo cáo VAT chạy với backend Modal; pipeline E2E OCR → SQL → VAT Report; bản thảo quyển ≥ 20 trang (bước riêng).

## 2. Quyết định đã chốt (03/11)

- Theo card Trello: Node 3 nằm trong M3-06 ở bản tối thiểu, không có web demo. Theo `docs/04` §5 và QĐ-08, Node 3 thuộc tầng COULD, nên đã đề xuất QĐ mới để Khôi xác nhận.
- Node 2 dùng bộ câu hỏi cố định Q1–Q6 trong `src/pipeline/README.md`.
- Nghiệm thu chạy trên làn vanilla; script vẫn hỗ trợ `--lane`.
- Node 2 và Node 3 dùng nguyên văn system prompt W2 v2 / W3 v2. Hệ quả: Node 2 không trả `explanation`; điểm lệch này đã ghi vào README.
- Node 2 `max_tokens=256`, trong khi W2 dùng 96, để tránh cắt cụt các câu có GROUP BY.
- Mỗi lần chạy có một SQLite riêng. Bộ kiểm tra SQL tự viết, DB mở `mode=ro` và `query_only`.
- Node 1 được gọi bằng subprocess CLI có sẵn với `cwd=REPO_ROOT`; token truyền qua biến môi trường.
- Không commit file `.sqlite`.

## 3. Kiểm tra giả định

- 28/30 hoá đơn W1 có ngày tháng 08/2020 trong `experiments/configs/w1/samples.jsonl`, nên Q1 có dữ liệu.

## 4. Test (conda `doan-d22`, không GPU)

| Lệnh | Kết quả |
|---|---|
| Baseline: `pytest tests/test_run_node1_extractor.py tests/test_invoice_models.py tests/test_openai_compatible_ready_wait.py tests/test_modal_client.py -q` | 35 passed |
| `pytest tests/test_sql_guard.py -q` | 9 passed, 11 subtests |
| `pytest tests/test_node2_runner.py -q` | 13 passed |
| `pytest tests/test_node3_report.py -q` | 9 passed |
| `pytest tests/test_run_pipeline_e2e.py -q` | 8 passed |
| Hồi quy 8 file (4 mới + 4 cũ) | 74 passed, 11 subtests |
| `python scripts/check_invoice_contract.py` | OK (`sqlite_user_version: 1`) |
| Toàn bộ suite `pytest tests -q` (trước khi sửa theo review) | 999 passed, 195 subtests |
| Hồi quy 8 file sau khi sửa theo review | 78 passed, 13 subtests |

## 4b. Review cuối và các chỗ đã sửa

Một reviewer độc lập (không phải người viết code) đã soát toàn bộ thay đổi. Kết quả: không có lỗi Critical, có 2 lỗi Important. Đã sửa cả hai, mỗi chỗ có test fail trước rồi mới pass:

- **Dữ liệu cũ bị dùng lại:** chạy lại cùng `run_id` mà Node 1 hỏng thì Node 2/3 sẽ đọc DB cũ.
  - Cách sửa: xoá các output Node 1 cũ trước khi chạy. Nếu Node 1 không ghi được summary thì `invoice_count=0` và bỏ qua Node 2/3.
  - Test: `test_stale_artifacts_not_reused_when_node1_crashes`.
- **Lỗi mạng làm chết cả pipeline:** timeout hoặc mất kết nối thô (`OSError`, `HTTPException`) làm crash, không ghi được summary.
  - Cách sửa: ghi nhận thành `transport_error`.
  - Test: `test_raw_network_errors_are_transport_errors`, `test_raw_network_error_is_transport_error`.

Thêm một chỗ tôi tự nâng mức để sửa luôn:

- **Báo cáo rỗng vẫn được tính là `ok`:** Node 3 trả nội dung trống thì giờ có trạng thái `empty_response`, và lệnh E2E trả exit 2.
  - Test: `test_empty_content_is_not_ok`.

Các điểm Minor để lại làm hạn chế đã biết, ghi trong PR.

## 5. Nghiệm thu Modal

Chưa chạy, đang chờ duyệt deploy làn vanilla vì tốn credit.

Ngưỡng đề xuất, cần nhóm xác nhận:
- Node 1 `schema_valid_rate ≥ 0.95`.
- Node 2 có ≥ 4/6 câu ở trạng thái `ok|empty`.
- Node 3 ở trạng thái `ok`, không có con số bịa.

## 6. Lần chạy Modal 1 (03/11, run-id `20261103-m306-e2e`): không có GPU, chưa đạt

- **Deploy:** chỉ làn vanilla (`PTIT_LANES=vanilla`, `PTIT_RUN_ID=20261103-m306`, scaledown 120 s). Deploy thành công lúc 23:47.
- **Hiện tượng:** Node 1 chờ hết `--ready-timeout` 1200 s mà server vẫn trả `HTTP 503 {"error":"no upstreams available"}`.
- **Nguyên nhân:** theo `modal app logs ptit-vllm`, function nằm chờ L4 suốt thời gian đó: "Function 'vanilla' … is waiting to be scheduled on a GPU_L4 worker. We are actively working on acquiring more capacity". Container chưa bao giờ khởi động, và volume không có `vanilla/20261103-m306/`.
- **Hành vi của E2E:** đúng thiết kế. Ghi `e2e_summary.json` với `invoice_count=0` và `node1.returncode=1`; Node 2/3 bị bỏ qua; exit 2. Bằng chứng: `experiments/runs/m3_06/20261103-m306-e2e-vanilla/`.
- **Dọn dẹp:** `modal app stop ptit-vllm -y`. `modal container list` rỗng.
- **Kết luận:** nghiệm thu Modal chưa đạt vì thiếu hạ tầng, không phải do code. Cần chạy lại với run-id mới.

## 7. Lần chạy Modal 2 (04/11, run-id `20261104-m306-e2e`): vẫn không có GPU, chưa đạt

- **Deploy:** chỉ làn vanilla (`PTIT_LANES=vanilla`, `PTIT_RUN_ID=20261104-m306`, scaledown 120 s). Deploy thành công.
- **Hiện tượng:** giống lần 1. `modal app logs ptit-vllm` lặp lại "Function 'vanilla' … is waiting to be scheduled on a GPU_L4 worker". Không container nào khởi động trong 1200 s. Node 1 preflight nhận `HTTP 503 no upstreams available`.
- **Hành vi của E2E:** đúng thiết kế, exit 2, `invoice_count=0`, Node 2/3 bị bỏ qua (`wall_ms ≈ 1210550`). Bằng chứng: `experiments/runs/m3_06/20261104-m306-e2e-vanilla/`.
- **Dọn dẹp:** `modal app stop ptit-vllm -y`; `modal container list` rỗng.
- **Kết luận:** sau 2 lần, L4 trên Modal liên tục hết capacity. Đây là rủi ro hạ tầng; cần nhóm quyết định hướng tiếp theo.
- **Kiểm tra lại 21:42 (04/11):** `modal run -m experiments.modal_runner::versions` (hàm GPU nhẹ nhất) chờ 300 s vẫn "waiting to be scheduled on a GPU_L4 worker", không container nào được tạo. Đã `modal app stop` app ephemeral `ap-OTT73KVW5M1H4xuQLDabWO`; workspace không còn app/container chạy.
- **Thử các loại GPU khác 21:58 (04/11)** bằng script `nvidia-smi` tạm (ngoài repo, chờ tối đa 180 s mỗi loại): T4 OK (3,5 s), A10G OK, L40S OK, A100-40GB OK, H100 OK; **chỉ L4 không được cấp**. Các app thử đều đã stopped, không còn container.

## 8. Lần chạy Modal 3 trên A10G (04/11, run-id `20261104-m306-e2e-a10g`): ĐẠT

- **Lý do đổi GPU:** L4 hết capacity cả ngày 04/11 (mục 7). Dương duyệt chạy A10G (24 GB, BF16) để nghiệm thu chức năng.
- **Thay đổi code:** `experiments/modal_lanes.py` thêm `DEFAULT_GPU="L4"`, `SUPPORTED_GPUS=("L4","A10G","T4")`, `resolve_gpu(environ)`; `experiments/modal_runner.py` dùng `GPU = lanes.resolve_gpu(os.environ)` (biến `PTIT_GPU`, mặc định L4 → benchmark cũ không đổi). Test `tests/test_modal_runner_config.py` (+5 test, fail trước rồi pass); `test_modal_runner_config.py` + `test_modal_client.py`: 61 passed. **Cần báo Khôi** vì là file dùng chung.
- **Deploy:** `PTIT_GPU=A10G PTIT_LANES=vanilla PTIT_RUN_ID=20261104-m306-a10g PTIT_SCALEDOWN_SECONDS=120`. Container có sau ~20 s.
- **Kết quả** (`experiments/runs/m3_06/20261104-m306-e2e-a10g-vanilla/`): exit 0; E2E wall ≈ 418,9 s (gồm cold start).
  - Node 1: 30/30 request, `schema_valid_rate = 1.0` (≥ 0,95 ✓); latency trung vị ≈ 7,96 s.
  - Node 2: 6/6 `ok|empty` (ok 4, empty 2) (≥ 4/6 ✓); p50 ≈ 1,13 s.
  - Node 3: `ok`, `finish_reason=stop`, 341 completion tokens (✓).
- **Kiểm tra tay `report.md`:** mọi con số trong báo cáo đều có trong phụ lục Q1–Q6 — 5.155.094 (Q1) = tổng các ngày 08/2020 ở Q5; không có số bịa.
- **Phát hiện chất lượng (không phải lỗi code):**
  - Q3 sai nghĩa: SQL Node 2 lấy hoá đơn có `MAX(total_amount_value)` thay vì `SUM` theo cửa hàng → trả "CỬA HÀNG NĂM OÁNH" (1.161.500), trong khi đúng là "MINIMART ANAN" (3.918.594). Node 3 trung thành với dữ liệu nhận được. Đây là giới hạn chất lượng Text-to-SQL của model 7B; theo thiết kế không sửa lại SQL.
  - Q2/Q6 rỗng và Q4 = 0 là đúng thực tế dữ liệu: `raw_text` W1 (MC-OCR `anno_texts`) chỉ có tên cửa hàng, địa chỉ, ngày, tổng tiền; 0/30 mẫu nhắc VAT/thuế; DB có 0 dòng item, 0 dòng thuế.
  - Báo cáo ghi Q4 là "KHÔNG CÓ DỮ LIỆU" thay vì "0 hoá đơn", và suy đoán nguyên nhân ("bảng chưa cập nhật") — diễn đạt chưa chính xác nhưng không bịa số.
- **Lệch so với kế hoạch:** GPU A10G thay vì L4 → latency không so sánh được với số liệu L4 của M3-02/M3-05. Nếu cần số trên L4, chạy lại lệnh với `PTIT_GPU` mặc định khi L4 có lại.
- **Dọn dẹp:** `modal app stop ptit-vllm -y`; `modal container list` rỗng.

## 9. Kiểm tra chức năng làn `static` trên A10G (04/11, run-id `20261104-m306-e2e-a10g`)

Mục đích: xác nhận pipeline vẫn chạy khi bật speculative decoding. **Không phải số đo hiệu năng** (1 lần, A10G, gồm cold start, telemetry phía client).

- Deploy `PTIT_GPU=A10G PTIT_LANES=static PTIT_RUN_ID=20261104-m306-a10g-static`. Kết quả `experiments/runs/m3_06/20261104-m306-e2e-a10g-static/`: exit 0; Node 1 30/30 hợp lệ; Node 2 ok 4 / empty 2; Node 3 `ok`.
- So với vanilla (cùng A10G):
  - Trường nghiệp vụ Node 1 (tên cửa hàng, ngày, tổng tiền) **giống 30/30**; toàn bộ payload JSON giống 14/30 (khác ở evidence/chuẩn hoá).
  - Node 2: Q2–Q6 giống hệt SQL và kết quả; **Q1 khác** — static sinh `WHERE issued_date LIKE '2020-08-%' AND extraction_status = 'Success'`, giá trị `'Success'` là tự bịa nên trả `NULL`; báo cáo Node 3 ghi "NULL do không có dữ liệu phù hợp".
  - Node 3: văn bản khác nhau (do đầu vào Q1 khác).
- Diễn giải: output không giống bit-to-bit giữa hai làn. Chưa đủ để kết luận vi phạm lossless — memory/M2-03 đã ghi nhận ngay cả vanilla-vs-vanilla cũng chỉ giống 45/54 do near-tie số học BF16; muốn kết luận cần đối chứng vanilla-vs-vanilla trên cùng A10G. Ghi vào hạn chế, chuyển cho Khôi (Luồng A, lossless QĐ-11).
- Tham khảo (không dùng làm số liệu): Node 1 latency trung vị 6,17 s (static) vs 7,96 s (vanilla); Node 3 3,83 s/289 token vs 4,72 s/341 token.
- Dọn dẹp: `modal app stop ptit-vllm -y`; `modal container list` rỗng.

## 10. Rà lỗi kết quả E2E (vanilla + static, A10G, 04/11)

Phạm vi: `experiments/runs/m3_06/20261104-m306-e2e-a10g-{vanilla,static}/`. Cách kiểm: đọc `node2_results.jsonl`, `node3_result.json`, `report.md`; đối chiếu `node1/validated_outputs.jsonl` với `raw_text` trong `experiments/configs/w1/samples.jsonl`; truy vấn trực tiếp `invoices.sqlite`.

### 10.1. Phần không có lỗi (đã xác minh)

- Node 1: cả hai làn 30/30 hợp lệ, `persistence_error` = 0. Tên cửa hàng, ngày, tổng tiền của 30/30 hoá đơn đều xuất hiện trong `raw_text` gốc (kiểm tự động bằng so khớp chuỗi/chữ số) — không thấy trường bịa.
- Node 2/3: mọi request `finish_reason=stop` (không bị cắt bởi `max_tokens`); không câu nào bị guard chặn, không `sql_error`/`timeout`.
- Q1 (vanilla) = 5.155.094 đúng: bằng tổng 7 ngày tháng 08/2020 ở Q5; 28/30 hoá đơn thuộc 08/2020.
- Q5 có 9 dòng (< 20) nên Node 3 nhận đủ dữ liệu.

### 10.2. Các lỗi phát hiện

| # | Mức | Loại | Hiện tượng | Nguyên nhân | Đề xuất |
|---|---|---|---|---|---|
| L1 | Trung bình | **Code** (`src/pipeline/text_to_sql.py:141`) | Static Q1 trả `[[NULL]]` nhưng status `ok`; Node 3 viết "tổng chi tiêu… là NULL". | Status chỉ xét `rows` rỗng. Hàm gộp (`SUM`, `MAX`…) trên 0 dòng khớp vẫn trả **1 dòng toàn NULL**, nên không bao giờ thành `empty`. | Coi kết quả mà mọi ô đều NULL là `empty` (thêm test RED→GREEN). Cần duyệt trước khi sửa. |
| L2 | Cao (chất lượng) | Model – SQL sai nghĩa | Q3 cả hai làn: `WHERE total_amount_value = (SELECT MAX(...))` → "CỬA HÀNG NĂM OÁNH" (1 hoá đơn 1.161.500). Đúng phải là `SUM` theo `seller_name` → "MINIMART ANAN" 3.918.594. | Giới hạn Text-to-SQL của Qwen2.5-7B 4-bit; prompt W2 frozen không có ví dụ; theo thiết kế không sửa lại SQL. | Ghi vào hạn chế. Đề xuất (ngoài M3-06): bộ đáp án chuẩn Q1–Q6 để đo *execution accuracy*. |
| L3 | Trung bình (chất lượng) | Model – bịa giá trị | Static Q1 thêm `extraction_status = 'Success'`; DB chỉ có `'partial'` (30/30) → kết quả NULL. | Model không biết miền giá trị cột; prompt chỉ có tên cột. | Ghi hạn chế; L1 giúp lỗi này hiện thành `empty` thay vì "NULL". |
| L4 | Thấp | Node 3 – diễn đạt chưa trung thành | Vanilla: Q4 = 0 nhưng báo cáo ghi "KHÔNG CÓ DỮ LIỆU" (đúng phải là "0 hoá đơn có VAT"); cả hai làn tự suy đoán nguyên nhân ("bảng chưa cập nhật", "lỗi truy vấn") không có trong dữ liệu. Không có con số bịa. | Model 7B; template user Node 3 chưa phân biệt "kết quả = 0" với "không có dòng". | Có thể thêm 1 câu vào `node3_report_user.md`: "giá trị 0 là kết quả hợp lệ, không phải thiếu dữ liệu; không suy đoán nguyên nhân". Cần duyệt. |
| L5 | Thấp | Bằng chứng – truy vết | `e2e_summary.json` không ghi loại GPU; chỉ biết A10G qua run-id và worklog. | Script chạy phía client, không biết GPU của server. | Thêm trường tuỳ chọn (ví dụ `--hardware-label`) hoặc đọc `server_env.json` trong volume. Cần duyệt. |
| L6 | Hạn chế dữ liệu | Workload | Q2, Q6 rỗng; Q4 = 0 ở cả hai làn. | `raw_text` W1 (MC-OCR `anno_texts`) chỉ có cửa hàng, địa chỉ, ngày, tổng tiền: 0/30 mẫu có mặt hàng hay VAT; DB: 0 item, 0 dòng thuế, 30/30 `partial`. | Ghi rõ trong quyển: các câu VAT/mặt hàng mới chỉ kiểm được nhánh "không có dữ liệu". |
| L7 | Mở (Luồng A) | Lossless | Static vs vanilla: payload Node 1 giống 14/30 (trường nghiệp vụ giống 30/30); SQL Q1 khác. | Chưa xác định: có thể là near-tie số học BF16 (M2-03: vanilla-vs-vanilla chỉ giống 45/54) hoặc khác biệt thật. | Cần đối chứng vanilla-vs-vanilla trên cùng A10G trước khi kết luận; chuyển Khôi (QĐ-11). |

### 10.3. Ghi chú đo lường

- `e2e_wall_ms` gồm cold start: static 504 s > vanilla 419 s vì nạp thêm draft model — **không** dùng để so tốc độ. Số latency từng node chỉ để tham khảo (1 lần, A10G).

### 10.4. Kết luận

Không có lỗi chặn nghiệm thu M3-06: pipeline chạy thông, xử lý lỗi đúng thiết kế, không có số bịa. Có 1 lỗi code nhỏ (L1) và 2 đề xuất cải tiến (L4, L5), tất cả chờ duyệt. Các lỗi chất lượng model (L2, L3) và hạn chế dữ liệu (L6) đưa vào phần hạn chế của PR và quyển; L7 chuyển Khôi.

## 11. Sửa L1/L4/L5 và chạy lại (04/11, run-id `20261104-m306-e2e-a10g-r2`, A10G, vanilla)

Sửa theo TDD (test fail trước rồi pass):
- **L1** `src/pipeline/text_to_sql.py`: kết quả chỉ gồm giá trị NULL → `empty`. Test: `test_aggregate_over_no_rows_is_empty` (RED→GREEN); thêm `test_zero_count_stays_ok`, `test_row_with_some_nulls_stays_ok` để giữ hành vi đúng (0 vẫn là `ok`).
- **L4** `src/pipeline/prompts/node3_report_user.md`: thêm "Giá trị 0 là kết quả hợp lệ…; không suy đoán nguyên nhân". Test: `test_user_message_treats_zero_as_result_and_forbids_speculation` (RED→GREEN). Hash `node3_user_template` trong summary đổi theo.
- **L5** `scripts/run_pipeline_e2e.py`: tham số `--hardware` → trường `hardware` trong `e2e_summary.json`. Test: `test_hardware_label_recorded`, `test_hardware_label_defaults_to_none` (RED→GREEN).
- README cập nhật (`empty` gồm kết quả toàn NULL; `--hardware`). Hồi quy 9 file liên quan: **124 passed, 13 subtests**.

Kết quả chạy lại (`experiments/runs/m3_06/20261104-m306-e2e-a10g-r2-vanilla/`): exit 0; `hardware = modal-A10G`; Node 1 30/30 (`schema_valid_rate 1.0`); Node 2 ok 4 / empty 2, SQL giống hệt lần chạy đầu; Node 3 `ok`, 596 completion tokens, `stop`.

Đối chiếu `report.md`:
- L4 đã khắc phục: Q4 được viết "Có 0 hóa đơn có phát sinh VAT"; không còn câu suy đoán nguyên nhân kiểu "bảng chưa cập nhật" (chỉ còn đề xuất kiểm tra lại dữ liệu).
- Mọi số trong báo cáo khớp phụ lục (Q1 = 5.155.094; 9 dòng theo ngày khớp Q5).
- Điểm mới (chất lượng model, không sửa): báo cáo thêm mục (5)(6) ngoài cấu trúc 4 phần yêu cầu, và bỏ qua Q3 (cửa hàng). L2 (Q3 sai nghĩa) vẫn còn vì SQL do model sinh không đổi.
- Dọn dẹp: `modal app stop ptit-vllm -y`; không còn app/container chạy.

Bộ bằng chứng chính cho nghiệm thu M3-06: `20261104-m306-e2e-a10g-r2-vanilla` (code cuối cùng). Các thư mục khác giữ làm lịch sử (L4 hết capacity, lần A10G đầu, kiểm tra static).
