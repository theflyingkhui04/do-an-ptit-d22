# Task 3 Brief: Cross-Platform CLI Binary Resolution & Environment in `modal_runner.py`

**Mục tiêu**: Sử dụng `shutil.which("modal")` để tìm kiếm file thực thi (hỗ trợ `modal.cmd`, `modal.exe` trên Windows và script trên Linux) và chuẩn hóa `PYTHONPATH` bằng `os.pathsep`.

## Các tệp liên quan:
- Modify: `src/flexsds/server/modal_runner.py`
- Modify: `tests/test_modal_runner_cli.py`

## Chi tiết các bước thực hiện:
1. Cập nhật `tests/test_modal_runner_cli.py`:
   - Viết test `test_modal_runner_resolves_modal_binary` sử dụng monkeypatch giả lập `shutil.which` trả về `C:\Python\Scripts\modal.cmd` và kiểm tra `build_modal_command` trả về path đã resolve.
   - Viết test `test_modal_runner_setup_environment_uses_os_pathsep` kiểm tra `PYTHONPATH` được ghép bằng `os.pathsep`.
2. Cập nhật `src/flexsds/server/modal_runner.py`:
   - Bổ sung `import shutil`.
   - Cập nhật `build_modal_command`:
     ```python
     def build_modal_command(self, config: Dict[str, Any], deploy: bool = False) -> List[str]:
         """Tạo danh sách đối số dòng lệnh gọi công cụ CLI modal tới modal_app.py."""
         modal_bin = shutil.which("modal") or "modal"
         app_path = str(Path(__file__).parent / "modal_app.py")
         return [modal_bin, "deploy" if deploy else "serve", app_path]
     ```
   - Cập nhật an toàn trong `setup_environment`:
     ```python
     resolved = Path(__file__).resolve()
     pkg_root = str(resolved.parents[2]) if len(resolved.parents) > 2 else "/root/project/src"
     ```
3. Chạy kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_modal_runner_cli.py -v`
   Đảm bảo toàn bộ test pass.
4. Commit thay đổi:
   ```bash
   git add src/flexsds/server/modal_runner.py tests/test_modal_runner_cli.py
   git commit -m "fix(modal_runner): resolve modal binary via shutil.which for cross-platform support"
   ```
5. Báo cáo lại kết quả thực thi và diff.
