# Task 1 Brief: Tách biệt trạng thái tắt HardwareAdaptive trong `wizard.py`

**Mục tiêu**: Khi người dùng chọn tắt `HardwareAdaptive` (`enable_adaptive = False`), wizard không được gọi nạp hoặc liên kết bất kỳ profile phần cứng nào, in ra thông báo engine đã tắt ở chế độ đầu cơ tĩnh, và lưu `profile_applied: null` trong cấu hình.

## Các tệp liên quan:
- Modify: `src/flexsds/cli/wizard.py`
- Modify: `tests/test_cli_wizard_flow.py`

## Chi tiết các bước thực hiện:
1. Thêm vào `tests/test_cli_wizard_flow.py`:
   - Bài test `test_wizard_disabled_adaptive_skips_profile_mapping`: kiểm tra cấu trúc cấu hình khi `enable_adaptive = False` thì `profile_applied` là `None`.
2. Cập nhật `src/flexsds/cli/wizard.py`:
   - Tại Bước 5 & Bước 6 (khoảng dòng 423-435):
     ```python
     # Step 5: HardwareAdaptive Optimization Engine
     enable_adaptive = prompt_bool(
         "Enable HardwareAdaptive engine?",
         default_val=True,
     )

     profile_name = None
     if not enable_adaptive:
         print("\033[32m✔ HardwareAdaptive engine: Disabled (Static speculation mode)\033[0m")
     else:
         cost_model, profile_name = load_profile_for_hardware(hw_info)
         print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")
     ```
   - Trong `config` lưu `"profile_applied": profile_name`.
3. Chạy kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -v`
   Đảm bảo toàn bộ test pass.
4. Commit:
   ```bash
   git add src/flexsds/cli/wizard.py tests/test_cli_wizard_flow.py
   git commit -m "fix(wizard): skip profile mapping when HardwareAdaptive is disabled"
   ```
5. Báo cáo lại kết quả thực thi và commit hash.
