# Đặc tả kỹ thuật: Khắc phục lỗi khởi tạo container Modal và tương thích đa nền tảng Linux - Windows

- **Mã tài liệu**: SPEC-MODAL-20261114
- **Ngày tạo**: 2026-11-14
- **Tác giả**: Đội ngũ phát triển FlexSDS (D22 PTIT)
- **Trạng thái**: Đã phê duyệt và hoàn thành (Approved and implemented)
- **Phạm vi tác động**: `src/flexsds/server/modal_app.py`, `src/flexsds/server/modal_runner.py`, `src/flexsds/server/local_runner.py`, `src/flexsds/cli/wizard.py`

---

## 1. Bối cảnh và mục đích

Khi người dùng chạy lệnh phục vụ trên đám mây Modal (`flexsds serve --backend modal` hoặc qua wizard cấu hình tương tác), hệ thống gặp sự cố sập container ngay trong giai đoạn nạp mã nguồn (container hydration). Container bị khởi động lại 18 lần liên tục rồi dừng hẳn với lỗi `APP_STATE_STOPPED` và `ConflictError`.

Mục đích của tài liệu đặc tả này là xác định nguyên nhân gốc rễ và đề xuất giải pháp xử lý dứt điểm các lỗi phát sinh, đảm bảo engine FlexSDS chạy mượt mà, ổn định trên cả máy chủ chạy hệ điều hành Linux và máy trạm Windows, đồng thời tối ưu hóa cho các loại GPU khả dụng trên hạ tầng Modal Cloud (L4, A10G, T4).

---

## 2. Mục tiêu và giới hạn phạm vi

### 2.1. Mục tiêu (Goals)
- Đảm bảo `modal_app.py` nạp thành công cả trên máy host cục bộ (Linux/Windows) và bên trong container đám mây Modal (Debian Linux).
- Loại bỏ hoàn toàn ngoại lệ `IndexError: 2` và `IndexError: 3` do phụ thuộc vào cấp độ cha của đường dẫn tệp (`Path.parents`).
- Tương thích đường dẫn đa nền tảng: Xử lý đúng ký tự phân cách thư mục (`/` so với `\`) và phân cách danh sách đường dẫn (`:` so với `;` qua `os.pathsep`).
- Đảm bảo CLI `modal` được kích hoạt chính xác trên Windows (`modal.exe`, `modal.cmd`) lẫn Linux.
- Khắc phục xung đột cờ tham số của vLLM đối với phương pháp đầu cơ EAGLE-3.1: Chỉ kích hoạt `--scheduler-cls AdaptiveAsyncScheduler` khi sử dụng phương pháp `draft_model`.
- Cung cấp cơ chế cảnh báo hoặc chuyển đổi GPU sang A10G khi GPU L4 trên Modal bị cạn kiệt tài nguyên.

### 2.2. Phi mục tiêu (Non-goals)
- Không can thiệp vào thuật toán lõi của Adaptive Async Scheduler trong `src/server/adaptive_scheduler.py`.
- Không thay đổi phiên bản vLLM đã ghim (`vllm==0.30.0` trên Modal và `vllm==0.33.0` trên Colab).
- Không yêu cầu người dùng phải tự thiết lập thủ công các biến môi trường phức tạp.

---

## 3. Phân tích nguyên nhân gốc rễ

### 3.1. Sự khác biệt cấu trúc tệp giữa máy trạm (Host) và container Modal
- **Trên máy trạm (Linux/Windows)**: Tệp `modal_app.py` nằm sâu trong cấu trúc gói mã nguồn: `<project_root>/src/flexsds/server/modal_app.py`. Do đó:
  - `Path(__file__).resolve().parents[2]` trỏ tới `<project_root>/src`.
  - `Path(__file__).resolve().parents[3]` trỏ tới `<project_root>`.
- **Trong container Modal**: Khi lệnh `modal serve` hoặc `modal deploy` đóng gói tệp nạp vào container, Modal đặt tệp trực tiếp tại `/root/modal_app.py`.
  - Đối với đường dẫn `/root/modal_app.py`, danh sách `parents` chỉ gồm: `[Path('/root'), Path('/')]` (độ dài chỉ bằng 2).
  - Lệnh truy xuất `parents[2]` (dòng 14) gây lỗi `IndexError: 2` ngay khi container vừa import module `modal_app`.
  - Tiếp theo, lệnh truy xuất `parents[3]` (dòng 32) cũng sẽ gây lỗi `IndexError: 3`.

### 3.2. Vấn đề tương thích đường dẫn giữa Windows và Linux
- Trên Windows, đường dẫn cục bộ có dạng `C:\Users\...\src`, sử dụng dấu gạch chéo ngược (`\`) và phân cách đường dẫn bằng dấu chấm phẩy (`;`).
- Trong container Modal (Debian Linux), toàn bộ đường dẫn từ xa bắt buộc phải theo chuẩn POSIX (`/root/project/src`).
- Khi khởi chạy tiến trình con `modal` trên Windows bằng `subprocess.Popen(["modal", ...])`, nếu hệ thống không tìm thấy phần mở rộng thực thi `.exe` hoặc `.cmd` trong biến `PATH`, lệnh sẽ phát sinh lỗi `FileNotFoundError`.

### 3.3. Xung đột cờ tham số đầu cơ EAGLE-3.1 trong `LocalServingRunner`
- Trong file `src/flexsds/server/local_runner.py`, phương thức `build_command` tự động bổ sung `--scheduler-cls src.server.adaptive_scheduler.AdaptiveAsyncScheduler` và tham số `num_speculative_tokens_per_batch_size` mỗi khi `enable_adaptive=True`.
- Với EAGLE-3.1 (`draft_method: "eagle3"`), vLLM xử lý cấu trúc cây đầu cơ và phân bổ token bằng engine nội bộ. Việc chèn thêm scheduler tùy biến bên ngoài gây xung đột tham số khiến vLLM crash ngay khi khởi động tiến trình.

### 3.4. Thiếu hụt dung lượng GPU L4 trên hạ tầng Modal
- Vùng đám mây của Modal thường xuyên quá tải GPU L4 do nhu cầu cao. Trong khi đó, tài nguyên GPU A10G (24 GB VRAM) và T4 (16 GB VRAM) luôn sẵn sàng.

---

## 4. Các hướng tiếp cận kỹ thuật

### Hướng tiếp cận 1: Nhận diện môi trường thực thi động (Runtime context detection) & chuẩn hóa đường dẫn (Đề xuất)
- **Cơ chế**:
  - Kiểm tra môi trường thực thi thông qua độ sâu đường dẫn `len(resolved.parents)` kết hợp sự tồn tại của thư mục container `/root/project`:
    - Nếu chạy trong container Modal: Cố định `PROJECT_ROOT = Path("/root/project")` và gói mã nguồn tại `/root/project/src`.
    - Nếu chạy trên máy trạm (Linux/Windows): Xác định `PROJECT_ROOT` từ `Path(__file__).resolve().parents[3]`, chuẩn hóa toàn bộ đường dẫn cục bộ bằng `.resolve()`.
  - Sử dụng `shutil.which("modal")` trong `modal_runner.py` để tự động định vị tệp thực thi `modal` trên cả Linux và Windows (hỗ trợ `modal.cmd`, `modal.exe`).
  - Trong `local_runner.py`, chỉ gắn `--scheduler-cls` và `num_speculative_tokens_per_batch_size` khi `draft_method == "draft_model"`. Khi dùng `eagle3`, chỉ truyền cấu hình đầu cơ tiêu chuẩn.
- **Ưu điểm**: Đơn giản, tin cậy tuyệt đối, không phát sinh chi phí I/O quét tệp, tương thích 100% với môi trường container của Modal.
- **Nhược điểm**: Cần cập nhật đồng bộ tại cả `modal_app.py`, `modal_runner.py` và `local_runner.py`.

### Hướng tiếp cận 2: Quét ngược tìm tệp neo dự án (Project root anchor discovery)
- **Cơ chế**: Duyệt ngược cây thư mục từ `__file__` lên trên cho đến khi tìm thấy tệp `pyproject.toml` hoặc thư mục `.git`. Nếu không tìm thấy (như trong container), gán về `/root/project`.
- **Ưu điểm**: Không phụ thuộc vào số cấp thư mục nếu cấu trúc mã nguồn nội bộ thay đổi.
- **Nhược điểm**: Tốn thêm thao tác đọc đĩa (I/O check), vẫn phải giữ nhánh fallback đặc thù cho container Modal vì trong container không có `.git`.

**Lựa chọn**: Hướng tiếp cận 1 là tối ưu nhất, xử lý triệt để nguyên nhân gốc rễ và phù hợp với tiêu chuẩn phát triển của Modal.

---

## 5. Thiết kế giải pháp chi tiết

### 5.1. Cập nhật `src/flexsds/server/modal_app.py`
Xử lý an toàn đường dẫn nạp gói và thư mục gốc:
```python
resolved = Path(__file__).resolve()
# Kiểm tra môi trường thực thi: Trong container Modal (/root/modal_app.py) hay trên máy host
if len(resolved.parents) > 2 and (resolved.parents[2] / "flexsds").exists():
    _pkg_root = str(resolved.parents[2])
    PROJECT_ROOT = resolved.parents[3]
else:
    # Môi trường container Modal đã mount mã nguồn vào /root/project
    _pkg_root = "/root/project/src"
    PROJECT_ROOT = Path("/root/project")

if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
```
Chuẩn hóa ánh xạ thư mục sang POSIX khi nạp vào Modal Image:
```python
src_dir = PROJECT_ROOT / "src"
if src_dir.exists():
    image = image.add_local_dir(str(src_dir), remote_path="/root/project/src")
```

### 5.2. Cập nhật `src/flexsds/server/modal_runner.py`
- Bổ sung `shutil.which` để tìm kiếm lệnh `modal`:
  ```python
  modal_bin = shutil.which("modal") or "modal"
  cmd = [modal_bin, "deploy" if deploy else "serve", app_path]
  ```
- Chuẩn hóa biến môi trường `PYTHONPATH` bằng `os.pathsep` để tương thích Windows (`;`) và Linux (`:`).

### 5.3. Cập nhật `src/flexsds/server/local_runner.py`
Phân định rõ ràng giữa đầu cơ mô hình độc lập (`draft_model`) và đầu cơ tích hợp EAGLE-3.1 (`eagle3`):
```python
is_eagle = (draft_method == "eagle3")

speculative_config: Dict[str, Any] = {
    "method": draft_method,
    "model": draft_model,
    "num_speculative_tokens": 4,
    "draft_sample_method": "greedy",
}
if not is_eagle:
    speculative_config["num_speculative_tokens_per_batch_size"] = [[1, 32, 4]]

if use_hetero:
    speculative_config["use_heterogeneous_vocab"] = True

# Chỉ gắn scheduler tùy biến cho phương pháp draft_model truyền thống
if serving_cfg.get("enable_adaptive", True) and not is_eagle:
    cmd.extend([
        "--speculative-config", json.dumps(speculative_config),
        "--scheduler-cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
        "--distributed-executor-backend", "src.server.lineage_executor.LineageUniProcExecutor",
        "--per-request-spec-decode-metrics", "summary",
    ])
else:
    cmd.extend([
        "--speculative-config", json.dumps(speculative_config),
        "--per-request-spec-decode-metrics", "summary",
    ])
```

### 5.4. Cập nhật `src/flexsds/cli/wizard.py`
- Bổ sung ghi chú khuyến nghị GPU A10G khi chọn hạ tầng Modal, giúp người dùng chủ động tránh tình trạng nghẽn hàng đợi L4:
  - Lựa chọn: `A10G (Khuyến nghị - 24 GB VRAM)`, `L4 (24 GB VRAM - Thường quá tải)`, `T4 (16 GB VRAM)`.

---

## 6. Kế hoạch kiểm thử và nghiệm thu (Verification plan)

1. **Kiểm thử đơn vị (Unit tests)**:
   - Chạy toàn bộ bộ test runner và wizard:
     `PYTHONPATH=src:. pytest tests/test_modal_runner_cli.py tests/test_local_runner.py tests/test_cli_wizard_flow.py`
   - Đảm bảo 100% test case pass.
2. **Kiểm thử mô phỏng đa nền tảng (Cross-platform simulation)**:
   - Mô phỏng đường dẫn kiểu container (`/root/modal_app.py`) để khẳng định không còn ngoại lệ `IndexError`.
   - Mô phỏng đường dẫn Windows (sử dụng dấu gạch chéo ngược) để đảm bảo `modal_runner` tạo đúng câu lệnh và ánh xạ `remote_path`.
3. **Kiểm thử lệnh thực thi (Dry-run command inspection)**:
   - Chạy `flexsds serve --backend modal --dry-run` để kiểm tra toàn bộ cấu trúc lệnh và biến môi trường được tạo ra.
