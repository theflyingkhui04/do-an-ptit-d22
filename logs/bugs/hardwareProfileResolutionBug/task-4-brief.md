# Task 4 Brief: Kiểm thử hồi quy toàn diện và nghiệm thu Dry-run

**Mục tiêu**: Chạy kiểm thử toàn diện tất cả các bộ test suite liên quan đến phân giải profile phần cứng, wizard flow, modal runner và local runner; kiểm tra lệnh dry-run của FlexSDS CLI; và cập nhật trạng thái trong tệp đặc tả kỹ thuật.

## Chi tiết các bước thực hiện:
1. Chạy toàn bộ các bài kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py tests/test_serving_profile_guard.py tests/test_modal_app_container_paths.py tests/test_modal_runner_cli.py tests/test_local_runner.py tests/test_flexsds_integration.py -v`
   Xác nhận 100% test pass.
2. Kiểm tra lệnh dry-run cho Modal serving (A10G):
   `PYTHONPATH=src:. python3 -m flexsds.cli.main serve --backend modal --gpu A10G --dry-run`
   Xác nhận in ra đúng cấu hình `A10G`.
3. Cập nhật trạng thái trong file spec `logs/bugs/hardwareProfileResolutionBug/2026-11-14-wizard-hardware-profile-resolution.md`:
   Thay `- **Trạng thái**: Đang chờ phê duyệt (Pending approval)` thành:
   `- **Trạng thái**: Đã phê duyệt và hoàn thành (Approved and implemented)`
4. Commit thay đổi:
   ```bash
   git add logs/bugs/hardwareProfileResolutionBug/2026-11-14-wizard-hardware-profile-resolution.md
   git commit -m "docs(spec): mark wizard hardware profile resolution as implemented"
   ```
5. Báo cáo lại kết quả kiểm thử và trạng thái cuối cùng.
