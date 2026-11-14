# Task 1 Brief: Safe Runtime Container Hydration & Path Resolution in `modal_app.py`

**Mục tiêu**: Xử lý an toàn đường dẫn nạp container trong `modal_app.py` để không còn bị sập với `IndexError: 2` hoặc `IndexError: 3` khi chạy trong container Modal (`/root/modal_app.py`).

## Các tệp liên quan:
- Modify: `src/flexsds/server/modal_app.py`
- Create: `tests/test_modal_app_container_paths.py`

## Chi tiết các bước thực hiện:
1. Viết bài test `test_modal_app_container_paths.py`:
   - Mô phỏng đường dẫn nông `/root/modal_app.py` có `len(parents) == 2` để đảm bảo logic phân giải đường dẫn trả về:
     `pkg_root == "/root/project/src"` và `project_root == Path("/root/project")`.
   - Viết bài test `test_modal_app_host_import` kiểm tra khi import trực tiếp `flexsds.server.modal_app` trên host thì `_pkg_root` và `PROJECT_ROOT` được gán chính xác mà không gặp lỗi.
2. Cập nhật `src/flexsds/server/modal_app.py`:
   - Thay thế việc truy xuất cứng `Path(__file__).resolve().parents[2]` và `parents[3]` bằng:
     ```python
     resolved = Path(__file__).resolve()
     if len(resolved.parents) > 2 and (resolved.parents[2] / "flexsds").exists():
         _pkg_root = str(resolved.parents[2])
         PROJECT_ROOT = resolved.parents[3]
     else:
         _pkg_root = "/root/project/src"
         PROJECT_ROOT = Path("/root/project")

     if _pkg_root not in sys.path:
         sys.path.insert(0, _pkg_root)
     ```
   - Cập nhật dòng gán `flexsds_dir` (khoảng dòng 51):
     ```python
     flexsds_dir = resolved.parents[1] if len(resolved.parents) > 1 else Path("/root/project/src/flexsds")
     ```
3. Chạy kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_modal_app_container_paths.py tests/test_modal_runner_cli.py -v`
   Đảm bảo toàn bộ test pass.
4. Commit thay đổi:
   ```bash
   git add tests/test_modal_app_container_paths.py src/flexsds/server/modal_app.py
   git commit -m "fix(modal): resolve container hydration path index errors safely"
   ```
5. Báo cáo lại kết quả thực thi và diff.
