# SDD ledger — plan: logs/bugs/hardwareProfileResolutionBug/plan.md

**Mục tiêu**: Thực thi kế hoạch chuẩn hóa phân giải hồ sơ phần cứng và trạng thái HardwareAdaptive trong CLI wizard bằng Subagent-Driven Development.

## Trạng thái các tác vụ (Task Status)
- [x] Tác vụ 1: Tách biệt trạng thái tắt HardwareAdaptive trong `wizard.py` (Commit: `ff952c50efa3c8db11070103a067ef957ae0c77b`)
- [x] Tác vụ 2: Phân giải phần cứng GPU đích cho nền tảng Modal và Colab trong `wizard.py` (Commit: `dc8b1f431e823ef9d410b3469e3f3943260d887b`)
- [x] Tác vụ 3: Chuẩn hóa cơ chế tra cứu CPU và fallback trong `cost_model.py` (Commit: `96743d850cef7aee1f3933a3e9b84d507be46338`)
- [x] Tác vụ 4: Kiểm thử hồi quy toàn diện và nghiệm thu Dry-run

## Nhật ký thực thi (Execution Log)
- 2026-11-14: Khởi tạo ledger SDD cho bug phân giải hồ sơ phần cứng.
- 2026-11-14: Hoàn thành Tác vụ 1 (Commit `ff952c50efa3c8db11070103a067ef957ae0c77b`) - Tách biệt trạng thái tắt HardwareAdaptive trong `wizard.py`, 5/5 tests pass.
- 2026-11-14: Hoàn thành Tác vụ 2 (Commit `dc8b1f431e823ef9d410b3469e3f3943260d887b`) - Phân giải phần cứng GPU đích từ xa (Modal A10G `sm_86`, L4 `sm_89`, T4 `sm_75`), cảnh báo fallback minh bạch, 6/6 tests pass.
- 2026-11-14: Hoàn thành Tác vụ 3 (Commit `96743d850cef7aee1f3933a3e9b84d507be46338`) - Chuẩn hóa tra cứu CPU, không còn gán ngầm CPU về L4 profile ở chế độ nghiêm ngặt, 5/5 tests pass.
- 2026-11-14: Hoàn thành Tác vụ 4 - Kiểm thử hồi quy toàn diện (57/57 tests pass), nghiệm thu lệnh dry-run A10G trên Modal thành công, cập nhật tài liệu đặc tả.
