# Worklog — [M4-03] Tích hợp cơ chế hoán đổi 4 làn vào pipeline hoá đơn

- Trello: [M4-03] `8mLOu2jW` — Dương (Lead 70%) / Khôi (Partner 30%), hạn 09/11/2026.
- Branch: `feat/m4-03-four-lane-node1` từ `origin/develop` @ `f7b15fd` (07/11/2026).

## 1. Mục tiêu

Theo DoD của card: `run_node1_extractor.py --lane <lane>` chạy trên cả 4 làn; dữ liệu nạp vào `data/processed/invoices.db`; bảng so sánh chất lượng trích xuất giữa 4 làn.

## 2. Quyết định đã chốt (07/11)

- SQLite nâng lên **schema v2, chỉ thêm**: lane `eagle`, `invoices.latency_ms`, `extraction_runs.latency_ms_median`, `user_version = 2`. Lý do: `CHECK` nằm trong DDL từng file nên DB v1 vẫn phải tạo lại; đúng quy tắc versioning của README.
- Latency lưu ở **cả hai cấp**: từng hoá đơn (giữ số liệu) và trung vị của lần chạy (đúng yêu cầu card).
- Đo trên **Modal A10G** vì L4 chưa có; có lỗi thì báo.
- `--backend` mặc định `modal`; `--base-url`/`--model` truyền tay vẫn ưu tiên.
- Chấm chất lượng với nhãn MC-OCR `key_text_pairs`: bỏ tiền tố địa chỉ, giữ dấu tiếng Việt, báo khớp tuyệt đối + token-F1 + số bỏ trống; không sửa nhãn.
- Đối chứng bằng 2 lần vanilla vì M2-03 cho thấy hai lần vanilla không giống hệt nhau.

## 3. Phát hiện khi audit và khi làm

- Làn `eagle` không ghi được vào DB v1 (`CHECK constraint failed`).
- Mặc định `--output-dir`, `--report` trỏ vào bằng chứng M1: chạy đúng lệnh của card sẽ ghi đè. Đã đổi mặc định sang `experiments/runs/m4_03/<run_id>/`.
- `check_invoice_contract.py` ghi đè `experiments/results/m1_03_invoice_contract_summary.json` mỗi lần chạy; với v2 sẽ làm sai bằng chứng M1. Đã khôi phục file M1 và đổi sang `m4_03_invoice_contract_summary.json`.
- DB sai hoặc `run_id` đã tồn tại giờ dừng **trước** khi gọi server, tránh tốn cold start.
- `modal deploy` từ M3-05 yêu cầu `src/`, `experiments/` đã commit (`_committed_bench_source`), nên phải commit trước khi chạy nhóm 5.

## 4. Test (conda `doan-d22`, không GPU)

| Lệnh | Kết quả |
|---|---|
| Baseline: `pytest tests/test_run_node1_extractor.py tests/test_invoice_models.py tests/test_node2_runner.py tests/test_run_pipeline_e2e.py -q` | 39 passed, 9 subtests |
| `pytest tests/test_invoice_sqlite_v2.py -q` | RED 8 failed → GREEN 11 passed |
| `pytest tests/test_run_node1_lanes.py -q` | RED 22 failed → GREEN 22 passed |
| `pytest tests/test_entity_eval.py -q` | RED (thiếu module) → GREEN 27 passed |
| `pytest tests/test_evaluate_node1_lanes.py -q` | 10 passed |
| `python scripts/check_invoice_contract.py` | PASS, `sqlite_user_version: 2` |
| Toàn bộ `pytest tests -q` | 1186 passed, 204 subtests |

Ghi chú TDD: `scripts/evaluate_node1_lanes.py` được viết cùng lúc với test, không thấy RED. Đã bù bằng kiểm tra đột biến: 6 lỗi cố ý (bỏ kiểm dao động, bỏ đếm bỏ trống, bỏ danh sách mẫu thiếu, tính cả baseline vào so sánh, bỏ điều kiện đủ 2 lần vanilla, bỏ danh sách mẫu lệch) đều bị test bắt.

Chạy thử không GPU trên 3 lần chạy M3-06: SELLER 29/30, ADDRESS 27/30, TIMESTAMPS 30/30, TOTAL_COST 30/30 ở cả ba; khớp số đã xem tay.

## 5. Chạy thật trên Modal A10G (07/11, commit `c19118c`)

- **Deploy:** cả 4 làn, `PTIT_GPU=A10G`, `PTIT_RUN_ID=20261107-m403`, `PTIT_SCALEDOWN_SECONDS=120`. Không gặp lỗi GPU hay làn nào.
- **Chạy tuần tự** với `--hardware modal-A10G --min-schema-valid-rate 1.0`, tất cả exit 0, 13:46–14:24:

| Lần chạy | Làn | Hợp lệ | Lỗi ghi DB | Hoá đơn trong DB | Trung vị latency (ms) |
|---|---|---|---|---|---|
| `m4_03_vanilla_r1` | vanilla | 30/30 | 0 | 30 | 7866,2 |
| `m4_03_vanilla_r2` | vanilla | 30/30 | 0 | 30 | 7813,8 |
| `m4_03_static` | static | 30/30 | 0 | 30 | 6118,8 |
| `m4_03_adaptive` | adaptive | 30/30 | 0 | 30 | 6159,2 |
| `m4_03_eagle` | eagle | 30/30 | 0 | 30 | 6047,7 |

Latency đo phía client, một lần, trên A10G: **chỉ tham khảo, không phải số đo hiệu năng.**

- **DB:** `data/processed/invoices.db`, `user_version = 2`, 150 hoá đơn, không dòng nào thiếu `latency_ms`.
- **Dọn dẹp:** `modal app stop ptit-vllm -y`; không còn app/container chạy.

## 6. Kết quả chất lượng

Bảng đầy đủ: `experiments/results/m4_03_entity_match.{json,md}`.

| Lần chạy | SELLER | ADDRESS | TIMESTAMPS | TOTAL_COST |
|---|---|---|---|---|
| vanilla_r1 | 29/30 | 27/30 | 30/30 | 30/30 |
| vanilla_r2 | 29/30 | 28/30 | 30/30 | 30/30 |
| static, adaptive, eagle | 29/30 | 27/30 | 30/30 | 30/30 |

- **Tiêu chí đề xuất đạt:** chênh lệch của cả 3 làn so với vanilla_r1 bằng 0 ở mọi trường, nằm trong dao động vanilla (0 hoặc 3,3 điểm ở ADDRESS).
- **Đối chiếu tay:** các mẫu lệch trùng nhau giữa các làn. 021: thêm ", Cẩm Phả"; 027: nhãn gán 2 dòng SELLER, model đẩy dòng 2 sang địa chỉ; 029: `raw_text` có địa chỉ nhưng model bỏ trống. Mẫu 021 đúng ở vanilla_r2, là nguồn duy nhất của dao động ADDRESS.

## 7. Phát hiện cần báo: đầu ra không giống hệt nhau

| Cặp | Toàn văn giống | Payload giống |
|---|---|---|
| vanilla_r1 – vanilla_r2 | 25/30 | 25/30 |
| vanilla_r1 – static | 13/30 | 14/30 |
| vanilla_r1 – adaptive | 14/30 | 15/30 |
| vanilla_r1 – eagle | 14/30 | 14/30 |
| vanilla_r2 – static / adaptive / eagle | 13 / 15 / 14 | 14 / 16 / 14 |
| static – adaptive | 17/30 | 19/30 |
| static – eagle | 22/30 | 24/30 |
| adaptive – eagle | 19/30 | 20/30 |

- Khác biệt giữa vanilla và làn suy đoán **lớn hơn** dao động vanilla–vanilla, nên không giải thích được chỉ bằng dao động giữa các lần chạy. Đây cũng là đối chứng vanilla–vanilla trên A10G còn nợ từ M3-06 (L7).
- Vị trí khác biệt (vanilla_r1 so với static): `extraction.field_evidence` 88 giá trị, `extraction.missing_fields` 72, `invoice.issued_time` 1 (`12:03:00` / `12:03`, cùng giá trị), `totals.subtotal` 1 mẫu; còn lại là khoảng trắng JSON.
- Năm trường nghiệp vụ (cửa hàng, địa chỉ, ngày, giờ, tổng tiền) giống vanilla_r1 ở 29/30 mẫu; mẫu còn lại chỉ khác cách viết giờ.
- **Kết luận được phép nói:** không thấy suy giảm độ chính xác ở 4 trường trên 30 mẫu. **Không** được nói đầu ra khớp từng ký tự. Nguyên nhân chưa xác lập; chuyển Khôi (QĐ-11).
- Đã sửa câu trong `docs/chuong-4-thuc-nghiem.md` §4.2.4 từng quy cho "số học BF16": M2-03 chạy trên T4 (không có BF16) và `logs/problems/2026-10-16-...` ghi nguyên nhân chưa xác lập.

## 8. Sonar trên PR #32

Lần đầu quality gate fail, PR chưa merge. Đã sửa:

- **Path traversal (`pythonsecurity:S8707`):** `scripts/evaluate_node1_lanes.py` chỉ đọc/ghi trong thư mục repo (`ALLOWED_ROOT`). Test `test_input_outside_allowed_root_is_refused`, `test_output_outside_allowed_root_is_refused` (RED → GREEN).
- **Bug `python:S1244`:** `check_invoice_contract.py` so số thực bằng `==`, đổi sang sai số nhỏ.
- **Code smell:** tách `summarize_run`, `render_markdown` (`S3776`); đơn giản hoá `_AMOUNT_RE` (`S5843`); `\D` (`S6353`); `startswith` (`S6659`).
- Sau khi sửa: tái tạo `m4_03_entity_match.{json,md}`, **không có thay đổi**; `pytest tests -q`: 1188 passed, 204 subtests.
- Chưa sửa: `S3776` ở `main()` của `run_node1_extractor.py` (độ phức tạp 36). Hàm này dài từ M1; tách ra là refactor rủi ro, ngoài phạm vi M4-03. Là code smell, không phải bug.
