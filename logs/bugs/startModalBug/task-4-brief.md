# Task 4 Brief: Update Modal GPU Options & Recommendation in `wizard.py`

**Mục tiêu**: Đưa `A10G (Khuyến nghị - 24 GB VRAM)` lên làm tùy chọn mặc định trong wizard Modal GPU để tránh thiếu hụt L4, đồng thời trích xuất chuẩn xác mã GPU.

## Các tệp liên quan:
- Modify: `src/flexsds/cli/wizard.py`
- Modify: `tests/test_cli_wizard_flow.py`

## Chi tiết các bước thực hiện:
1. Cập nhật `tests/test_cli_wizard_flow.py`:
   - Viết test `test_modal_gpu_selection_extraction` kiểm tra trích xuất đúng `"A10G"`, `"L4"`, `"T4"` từ danh sách lựa chọn có kèm mô tả.
2. Cập nhật `src/flexsds/cli/wizard.py`:
   - Tại vị trí xử lý `backend == "modal"` (dòng 309-315):
     ```python
     target_gpu = "A10G"
     if backend == "modal":
         modal_gpu_choices = [
             "A10G (Recommended - 24 GB VRAM)",
             "L4 (24 GB VRAM - Capacity limited)",
             "T4 (16 GB VRAM)",
         ]
         chosen = prompt_choice("Modal GPU:", modal_gpu_choices, default_idx=0)
         target_gpu = "A10G" if "A10G" in chosen else ("L4" if "L4" in chosen else "T4")
     ```
3. Chạy kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -v`
   Đảm bảo toàn bộ test pass.
4. Commit thay đổi:
   ```bash
   git add src/flexsds/cli/wizard.py tests/test_cli_wizard_flow.py
   git commit -m "feat(wizard): recommend A10G GPU for Modal serving to avoid L4 shortage"
   ```
5. Báo cáo lại kết quả thực thi và diff.
