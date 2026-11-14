# Task 5 Brief: End-to-End Regression & Dry-Run Verification

**Mục tiêu**: Kiểm thử toàn diện tất cả các test suite liên quan, chạy thử nghiệm các lệnh dry-run của FlexSDS CLI (cả Modal và Local) để xác thực không còn lỗi đường dẫn hay cờ tham số xung đột, và cập nhật trạng thái trong tệp đặc tả.

## Chi tiết các bước thực hiện:
1. Chạy toàn bộ các bài kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_modal_app_container_paths.py tests/test_modal_runner_cli.py tests/test_local_runner.py tests/test_cli_wizard_flow.py tests/test_flexsds_integration.py -v`
   Xác nhận 100% test pass.
2. Kiểm tra lệnh dry-run cho Modal serving:
   `PYTHONPATH=src:. python3 -m flexsds.cli.main serve --backend modal --gpu A10G --dry-run`
   Xác nhận sinh đúng cấu hình `A10G` và đường dẫn `modal_app.py`.
3. Kiểm tra lệnh dry-run cho Local serving:
   `PYTHONPATH=src:. python3 -m flexsds.cli.main serve --backend local --dry-run`
   Xác nhận không chứa `--scheduler-cls` khi dùng `eagle3`.
4. Cập nhật trạng thái trong file spec `logs/bugs/startModalBug/2026-11-14-modal-runner-cross-platform-container-hydration.md`:
   Thay `- **Trạng thái**: Đang chờ phê duyệt (Pending approval)` thành:
   `- **Trạng thái**: Đã phê duyệt và hoàn thành (Approved and implemented)`
5. Commit thay đổi:
   ```bash
   git add logs/bugs/startModalBug/2026-11-14-modal-runner-cross-platform-container-hydration.md
   git commit -m "docs(spec): mark modal hydration and cross-platform fix spec as implemented"
   ```
6. Báo cáo lại kết quả kiểm thử và trạng thái cuối cùng.
