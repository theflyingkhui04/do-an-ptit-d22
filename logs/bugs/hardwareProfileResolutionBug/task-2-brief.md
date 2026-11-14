# Task 2 Brief: Phân giải phần cứng GPU đích cho nền tảng Modal và Colab trong `wizard.py`

**Mục tiêu**: Khi bật `HardwareAdaptive` trên backend đám mây (`modal` hoặc `colab`), wizard phải tra cứu hồ sơ phần cứng dựa trên GPU đích (`target_gpu`: A10G `sm_86`, L4 `sm_89`, T4 `sm_75`, A100 `sm_80`) thay vì dùng thông số CPU của laptop người dùng. Nếu chưa có profile đo đạc (như A10G), hiển thị cảnh báo đang dùng `conservative_analytical_fallback` và gợi ý lệnh hiệu chuẩn.

## Các tệp liên quan:
- Modify: `src/flexsds/cli/wizard.py`
- Modify: `tests/test_cli_wizard_flow.py`

## Chi tiết các bước thực hiện:
1. Thêm vào `tests/test_cli_wizard_flow.py`:
   - Bài test `test_wizard_remote_serving_hardware_resolution`: kiểm tra logic phân giải đúng kiến trúc GPU đích (`A10G` -> `sm_86`, `L4` -> `sm_89`, `T4` -> `sm_75`) cho `backend in ("modal", "colab")`.
2. Cập nhật `src/flexsds/cli/wizard.py`:
   - Tại Bước 6:
     ```python
     if not enable_adaptive:
         print("\033[32m✔ HardwareAdaptive engine: Disabled (Static speculation mode)\033[0m")
     else:
         if backend in ("modal", "colab"):
             remote_sm_map = {
                 "A10G": "sm_86",
                 "L4": "sm_89",
                 "T4": "sm_75",
                 "A100": "sm_80",
             }
             target_sm = remote_sm_map.get(target_gpu, "sm_89")
             budgets = {"T4": 15000.0, "L4": 23000.0, "A10G": 22000.0, "A100": 40000.0}
             target_vram = budgets.get(target_gpu, 22000.0)
             serving_hw = HardwareInfo(
                 device_name=f"Remote {target_gpu}",
                 sm_architecture=target_sm,
                 total_vram_mb=target_vram,
                 free_vram_mb=target_vram,
                 cuda_version="12.4",
                 is_cuda_available=True,
                 device_count=1,
             )
         else:
             serving_hw = hw_info

         cost_model, profile_name = load_profile_for_hardware(serving_hw, strict=False)
         if profile_name == "conservative_analytical_fallback":
             print(
                 f"\033[33m⚠ [Hardware Profile Warning] No measured profile for GPU '{target_gpu or serving_hw.device_name}' "
                 f"({serving_hw.sm_architecture}). Using conservative analytical fallback.\033[0m"
             )
             print("  Tip: Calibrate exact latency using: flexsds profile --benchmark (or disable adaptive to use static speculation)")
         else:
             print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")
     ```
   - Cập nhật trường `hardware` trong `config` sử dụng `serving_hw`.
3. Chạy kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -v`
   Đảm bảo toàn bộ test pass.
4. Commit:
   ```bash
   git add src/flexsds/cli/wizard.py tests/test_cli_wizard_flow.py
   git commit -m "fix(wizard): resolve remote target GPU hardware profile correctly"
   ```
5. Báo cáo lại kết quả thực thi và commit hash.
