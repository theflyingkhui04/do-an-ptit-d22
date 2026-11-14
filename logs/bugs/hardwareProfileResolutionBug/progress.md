# SDD ledger — plan: logs/bugs/hardwareProfileResolutionBug/plan.md

**Mục tiêu**: Thay thế cơ chế tự động fallback phân tích bằng cơ chế chặn lỗi tương tác (fail-closed prompt) với 3 phương án lựa chọn khi gặp bất kỳ GPU nào chưa có hồ sơ thực nghiệm (A10G `sm_86`, H100 `sm_90`, v.v.).

## Trạng thái các tác vụ (Task Status)
- [x] Tác vụ 1: Triển khai chặn lỗi tương tác và menu 3 lựa chọn trong `wizard.py` (Commit: `a22ca24`)
- [x] Tác vụ 2: Chuẩn hóa kiểm tra nghiêm ngặt trong CLI `main.py` cho GPU từ xa (Commit: `e723d08`)
- [x] Tác vụ 3: Cập nhật `cost_model.py` hỗ trợ đo đạc Ampere A10G (`sm_86`) (Commit: `f812064`)
- [x] Tác vụ 4: Kiểm thử hồi quy toàn diện và nghiệm thu dry-run

## Nhật ký thực thi (Execution Log)
- 2026-11-14: Cập nhật đặc tả kỹ thuật và kế hoạch thực thi theo yêu cầu của người dùng (tuyệt đối không dùng fallback phân tích, bắt buộc hỏi ý kiến người dùng khi gặp GPU chưa đo đạc).
- 2026-11-14: Hoàn thành Tác vụ 1 (Commit `a22ca24`) - Triển khai cơ chế chặn lỗi tương tác với menu 3 lựa chọn (`calibrate`, `switch_static`, `abort`) khi phát hiện GPU chưa có profile đo đạc trong `wizard.py`; bổ sung bài test xác thực 3 nhánh, 9/9 tests wizard flow pass.
- 2026-11-14: Hoàn thành Tác vụ 2 (Commit `e723d08`) - Chuẩn hóa kiểm tra nghiêm ngặt profile cho GPU từ xa (Modal, Google Colab) trong CLI `main.py`; chặn chạy với mã lỗi 1 khi thiếu profile và không có cờ `--no-adaptive` hoặc `--allow-fallback`; 16/16 tests pass.
- 2026-11-14: Hoàn thành Tác vụ 3 (Commit `f812064`) - Bổ sung baseline kiến trúc Ampere (`sm_86`) vào `cost_model.py` phục vụ việc hiệu chuẩn micro-benchmark phần cứng A10G; 17/17 tests pass.
- 2026-11-14: Hoàn thành Tác vụ 4 - Kiểm thử hồi quy toàn diện và nghiệm thu dry-run:
  + Chạy toàn bộ 71 tệp test hoạt động trong kho mã nguồn: 1031 passed, 125 skipped, 54 subtests passed, 0 failed.
  + Chạy bộ test trọng tâm (`test_cli_wizard_flow.py`, `test_serving_profile_guard.py`, `test_modal_app_container_paths.py`, `test_modal_runner_cli.py`, `test_local_runner.py`, `test_flexsds_integration.py`, `test_flexsds_baseline.py`, `test_flexsds_advisor.py`): 95/95 passed trong 1.39s.
  + Nghiệm thu dry-run CLI:
    * `serve --backend modal --gpu A10G --dry-run`: Chặn lỗi đỏ `✖ [Hardware Profile Error] No measured latency profile for GPU 'Remote A10G (sm_86)'`, mã thoát 1.
    * `serve --backend modal --gpu A10G --no-adaptive --dry-run`: Chạy thành công không tra cứu profile, mã thoát 0.
    * `serve --backend modal --gpu L4 --dry-run`: Nạp thành công `profile_sm89_l4.json`, mã thoát 0.
    * `serve --backend modal --gpu T4 --dry-run`: Nạp thành công `profile_sm75_t4.json`, mã thoát 0.
