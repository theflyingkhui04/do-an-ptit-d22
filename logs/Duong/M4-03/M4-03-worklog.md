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
