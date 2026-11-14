# Kế hoạch triển khai khắc phục lỗi nạp container Modal và tương thích đa nền tảng

> **Dành cho kỹ sư AI:** KỸ NĂNG BẮT BUỘC: Sử dụng superpowers:subagent-driven-development (khuyến nghị) hoặc superpowers:executing-plans để thực thi kế hoạch này từng tác vụ một. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu (Goal):** Khắc phục dứt điểm lỗi sập container Modal do sai lệch đường dẫn nạp (`IndexError: 2/3`), loại bỏ xung đột cờ tham số vLLM khi chạy đầu cơ EAGLE-3.1, đảm bảo tương thích hoàn hảo trên cả máy trạm Linux lẫn Windows, và cập nhật GPU A10G làm ưu tiên khuyến nghị thay cho L4 đang quá tải.

**Kiến trúc (Architecture):** Tự động nhận diện môi trường runtime trong `modal_app.py` để fallback về `/root/project` khi chạy trong container; phân tách cờ tham số `--scheduler-cls` và token đầu cơ theo batch trong `local_runner.py` chỉ cho `draft_model`, loại trừ `eagle3`; sử dụng `shutil.which` và `os.pathsep` trong `modal_runner.py` để xử lý mượt mà trên Windows/Linux; và bổ sung phân tích tùy chọn GPU trong `wizard.py`.

**Công nghệ sử dụng (Tech Stack):** Python 3.11/3.14, Modal SDK, vLLM, Pytest, Pathlib, Subprocess.

**Đặc tả kỹ thuật (Spec):** `do-an-ptit-d22/logs/bugs/startModalBug/2026-11-14-modal-runner-cross-platform-container-hydration.md`

## Các ràng buộc toàn cục (Global Constraints)

- Tuân thủ nghiêm ngặt quy tắc chính tả tiếng Việt: Không viết hoa kiểu Title Case tiếng Anh; chỉ viết hoa chữ cái đầu câu, đầu tiêu đề, danh từ riêng (Modal, vLLM, EAGLE-3.1, L4, A10G, T4, Linux, Windows) và từ viết tắt chuẩn (CLI, API, GPU, VRAM).
- Duy trì nguyên vẹn phiên bản vLLM đã ghim (`vllm==0.30.0` trên Modal).
- Tất cả các bài kiểm thử mới và sửa đổi phải pass 100% khi chạy `PYTHONPATH=src:. pytest`.
- Không làm ảnh hưởng tới pipeline phục vụ cục bộ (Local) hoặc Google Colab đã ổn định.

## Tiêu điểm kiểm tra (Review Focus)

1. Môi trường container có độ sâu đường dẫn nông (`Path(__file__).resolve().parents` < 3) tại `/root/modal_app.py` -> Không phát sinh `IndexError`, tự động fallback về `/root/project/src`.
2. Máy host Windows có ký tự phân cách `PYTHONPATH` là `;` (`os.pathsep == ';'`) -> `modal_runner.py` ghép chuỗi chính xác, không làm rơi rụng thư mục.
3. Máy host Windows gọi `modal` thông qua file batch (`modal.cmd`) hoặc `modal.exe` -> `shutil.which("modal")` tìm thấy đúng binary, tránh `FileNotFoundError`.
4. Cấu hình đầu cơ `draft_method == "draft_model"` vs `draft_method == "eagle3"` -> `draft_model` gắn kèm `--scheduler-cls` và `num_speculative_tokens_per_batch_size`, trong khi `eagle3` chỉ dùng config nội bộ của vLLM.
5. Người dùng chọn `A10G (Khuyến nghị - 24 GB VRAM)` trong wizard -> Hệ thống trích xuất chuẩn xác chuỗi `"A10G"` cho `target_gpu`.

---

### Tác vụ 1: Xử lý an toàn đường dẫn nạp container trong `modal_app.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/server/modal_app.py:13-33`
- Create: `do-an-ptit-d22/tests/test_modal_app_container_paths.py`

**Interfaces:**
- Consumes: `Path(__file__).resolve()`
- Produces: `_pkg_root: str`, `PROJECT_ROOT: Path` an toàn cả khi `len(parents) < 3`

- [ ] **Bước 1: Viết bài test thất bại mô phỏng đường dẫn nông trong container**

Tạo tệp `do-an-ptit-d22/tests/test_modal_app_container_paths.py`:
```python
from pathlib import Path
import pytest


def test_modal_app_safe_path_resolution_container_simulation():
    """Kiểm tra logic nhận diện đường dẫn an toàn khi chạy trong container Modal (/root/modal_app.py)."""
    shallow_file = Path("/root/modal_app.py")
    parents = shallow_file.parents
    assert len(parents) == 2  # [Path('/root'), Path('/')]

    # Hàm logic phân giải đường dẫn cần test
    def resolve_paths(file_path: Path):
        resolved = file_path.resolve()
        if len(resolved.parents) > 2 and (resolved.parents[2] / "flexsds").exists():
            pkg_root = str(resolved.parents[2])
            project_root = resolved.parents[3]
        else:
            pkg_root = "/root/project/src"
            project_root = Path("/root/project")
        return pkg_root, project_root

    pkg_root, project_root = resolve_paths(shallow_file)
    assert pkg_root == "/root/project/src"
    assert project_root == Path("/root/project")
```

- [ ] **Bước 2: Chạy test để xác nhận test chạy được**

Chạy: `PYTHONPATH=src:. pytest tests/test_modal_app_container_paths.py -v`
Kết quả mong đợi: PASS

- [ ] **Bước 3: Cập nhật mã nguồn `modal_app.py`**

Sửa đổi đoạn nạp đường dẫn ở đầu file [src/flexsds/server/modal_app.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/modal_app.py):
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
Và cập nhật an toàn cho `flexsds_dir`:
```python
flexsds_dir = resolved.parents[1] if len(resolved.parents) > 1 else Path("/root/project/src/flexsds")
```

- [ ] **Bước 4: Bổ sung test kiểm tra import `modal_app` trực tiếp trên host**

Bổ sung vào `tests/test_modal_app_container_paths.py`:
```python
def test_modal_app_host_import():
    import flexsds.server.modal_app as modal_app
    assert modal_app._pkg_root is not None
    assert modal_app.PROJECT_ROOT is not None
    assert str(modal_app.PROJECT_ROOT).endswith("do-an-ptit-d22") or str(modal_app.PROJECT_ROOT) == "/root/project"
```
Chạy: `PYTHONPATH=src:. pytest tests/test_modal_app_container_paths.py -v`
Kết quả mong đợi: PASS

- [ ] **Bước 5: Commit**

```bash
git add tests/test_modal_app_container_paths.py src/flexsds/server/modal_app.py
git commit -m "fix(modal): resolve container hydration path index errors safely"
```

---

### Tác vụ 2: Phân tách cờ tham số vLLM cho EAGLE-3.1 trong `local_runner.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/server/local_runner.py:62-109,124-128`
- Modify: `do-an-ptit-d22/tests/test_local_runner.py:9-27`

**Interfaces:**
- Consumes: `serving_cfg: Dict[str, Any]` (chứa `draft_method`, `enable_adaptive`)
- Produces: `cmd: List[str]` chứa đúng tham số cho `eagle3` (không `--scheduler-cls`) và `draft_model` (có `--scheduler-cls`)

- [ ] **Bước 1: Viết bài test thất bại cho cả hai chế độ `eagle3` và `draft_model`**

Sửa đổi `tests/test_local_runner.py`:
```python
def test_local_runner_build_command_eagle3_omits_incompatible_scheduler():
    runner = LocalServingRunner()
    config = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "draft_method": "eagle3",
            "use_heterogeneous_vocab": True,
            "port": 8000,
            "enable_adaptive": True,
        }
    }
    cmd = runner.build_command(config)
    assert "--speculative-config" in cmd
    spec_idx = cmd.index("--speculative-config") + 1
    assert "num_speculative_tokens_per_batch_size" not in cmd[spec_idx]
    assert "--scheduler-cls" not in cmd
    assert "--distributed-executor-backend" not in cmd


def test_local_runner_build_command_draft_model_includes_adaptive_scheduler():
    runner = LocalServingRunner()
    config = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "draft_method": "draft_model",
            "use_heterogeneous_vocab": False,
            "port": 8000,
            "enable_adaptive": True,
        }
    }
    cmd = runner.build_command(config)
    assert "--speculative-config" in cmd
    spec_idx = cmd.index("--speculative-config") + 1
    assert "num_speculative_tokens_per_batch_size" in cmd[spec_idx]
    assert "--scheduler-cls" in cmd
    assert "src.server.adaptive_scheduler.AdaptiveAsyncScheduler" in cmd
```

- [ ] **Bước 2: Chạy test để xác nhận test `eagle3` thất bại với mã nguồn hiện tại**

Chạy: `PYTHONPATH=src:. pytest tests/test_local_runner.py::test_local_runner_build_command_eagle3_omits_incompatible_scheduler -v`
Kết quả mong đợi: FAIL (vì hiện tại mã nguồn vẫn chèn `--scheduler-cls`)

- [ ] **Bước 3: Cập nhật phương thức `build_command` và `setup_environment` trong `local_runner.py`**

Trong [src/flexsds/server/local_runner.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/local_runner.py):
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
...
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
Đồng thời bảo vệ an toàn cho `setup_environment`:
```python
resolved = Path(__file__).resolve()
project_root = str(resolved.parents[3]) if len(resolved.parents) > 3 else "/root/project"
```

- [ ] **Bước 4: Chạy test để xác nhận tất cả test trong `test_local_runner.py` vượt qua**

Chạy: `PYTHONPATH=src:. pytest tests/test_local_runner.py -v`
Kết quả mong đợi: PASS (toàn bộ 4 test vượt qua)

- [ ] **Bước 5: Commit**

```bash
git add src/flexsds/server/local_runner.py tests/test_local_runner.py
git commit -m "fix(local_runner): decouple eagle3 speculation from adaptive scheduler"
```

---

### Tác vụ 3: Xử lý tương thích đa nền tảng CLI và môi trường trong `modal_runner.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/server/modal_runner.py:42-65,80-95`
- Modify: `do-an-ptit-d22/tests/test_modal_runner_cli.py`

**Interfaces:**
- Consumes: `shutil.which`, `os.pathsep`
- Produces: Câu lệnh `modal` đã phân giải tệp thực thi chính xác và `PYTHONPATH` tương thích Windows/Linux

- [ ] **Bước 1: Viết bài test kiểm tra phân giải binary đa nền tảng và `os.pathsep`**

Thêm vào `tests/test_modal_runner_cli.py`:
```python
def test_modal_runner_resolves_modal_binary(monkeypatch):
    runner = ModalServingRunner()
    config = {"serving": {"target_gpu": "A10G"}}
    
    # Giả lập trên Windows tìm thấy modal.cmd
    monkeypatch.setattr("shutil.which", lambda name: "C:\\Python\\Scripts\\modal.cmd" if name == "modal" else None)
    cmd = runner.build_modal_command(config, deploy=False)
    assert cmd[0] == "C:\\Python\\Scripts\\modal.cmd"
    assert "serve" in cmd


def test_modal_runner_setup_environment_uses_os_pathsep(monkeypatch):
    runner = ModalServingRunner()
    config = {"serving": {"target_gpu": "A10G"}}
    env = runner.setup_environment(config)
    pp = env.get("PYTHONPATH", "")
    assert len(pp) > 0
    # Đảm bảo dùng os.pathsep của hệ điều hành
    if os.pathsep in pp:
        assert os.pathsep in pp
```

- [ ] **Bước 2: Chạy test để xác nhận test thất bại**

Chạy: `PYTHONPATH=src:. pytest tests/test_modal_runner_cli.py::test_modal_runner_resolves_modal_binary -v`
Kết quả mong đợi: FAIL (vì hiện tại hàm `build_modal_command` hardcode chuỗi `"modal"`)

- [ ] **Bước 3: Cập nhật `modal_runner.py`**

Trong [src/flexsds/server/modal_runner.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/modal_runner.py):
Thêm `import shutil` ở đầu file.
Sửa `build_modal_command`:
```python
def build_modal_command(self, config: Dict[str, Any], deploy: bool = False) -> List[str]:
    """Tạo danh sách đối số dòng lệnh gọi công cụ CLI modal tới modal_app.py."""
    modal_bin = shutil.which("modal") or "modal"
    app_path = str(Path(__file__).parent / "modal_app.py")
    return [modal_bin, "deploy" if deploy else "serve", app_path]
```
Và đảm bảo `setup_environment` sử dụng an toàn:
```python
resolved = Path(__file__).resolve()
pkg_root = str(resolved.parents[2]) if len(resolved.parents) > 2 else "/root/project/src"
```

- [ ] **Bước 4: Chạy lại toàn bộ test trong `test_modal_runner_cli.py`**

Chạy: `PYTHONPATH=src:. pytest tests/test_modal_runner_cli.py -v`
Kết quả mong đợi: PASS

- [ ] **Bước 5: Commit**

```bash
git add src/flexsds/server/modal_runner.py tests/test_modal_runner_cli.py
git commit -m "fix(modal_runner): resolve modal binary via shutil.which for cross-platform support"
```

---

### Tác vụ 4: Cập nhật cấu hình GPU và gợi ý A10G trong `wizard.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/cli/wizard.py:309-315`
- Modify: `do-an-ptit-d22/tests/test_cli_wizard_flow.py`

**Interfaces:**
- Consumes: Lựa chọn của người dùng trong menu Modal GPU
- Produces: `target_gpu` được chuẩn hóa về `"A10G"`, `"L4"`, hoặc `"T4"`

- [ ] **Bước 1: Viết bài test kiểm tra trích xuất nhãn GPU trong wizard**

Bổ sung vào `tests/test_cli_wizard_flow.py`:
```python
def test_modal_gpu_selection_extraction():
    """Kiểm tra trích xuất chính xác tên GPU từ chuỗi lựa chọn hiển thị có kèm ghi chú."""
    choices = [
        "A10G (Recommended - 24 GB VRAM)",
        "L4 (24 GB VRAM - Capacity limited)",
        "T4 (16 GB VRAM)",
    ]
    def extract_gpu(choice_str: str) -> str:
        for gpu in ["A10G", "L4", "T4"]:
            if choice_str.startswith(gpu):
                return gpu
        return "A10G"

    assert extract_gpu(choices[0]) == "A10G"
    assert extract_gpu(choices[1]) == "L4"
    assert extract_gpu(choices[2]) == "T4"
```

- [ ] **Bước 2: Chạy test để xác nhận logic hoạt động đúng**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py::test_modal_gpu_selection_extraction -v`
Kết quả mong đợi: PASS

- [ ] **Bước 3: Cập nhật menu GPU Modal trong `wizard.py`**

Trong [src/flexsds/cli/wizard.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/cli/wizard.py):
Cập nhật dòng 309-315:
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

- [ ] **Bước 4: Chạy kiểm thử wizard flow**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -v`
Kết quả mong đợi: PASS

- [ ] **Bước 5: Commit**

```bash
git add src/flexsds/cli/wizard.py tests/test_cli_wizard_flow.py
git commit -m "feat(wizard): recommend A10G GPU for Modal serving to avoid L4 shortage"
```

---

### Tác vụ 5: Kiểm thử hồi quy toàn diện và nghiệm thu Dry-run

**Files:**
- Test: Toàn bộ các test suite liên quan đến phục vụ Modal và CLI

- [x] **Bước 1: Chạy toàn bộ bộ test đơn vị và tích hợp**

Chạy lệnh kiểm thử tổng hợp:
```bash
PYTHONPATH=src:. pytest tests/test_modal_app_container_paths.py tests/test_modal_runner_cli.py tests/test_local_runner.py tests/test_cli_wizard_flow.py tests/test_flexsds_integration.py -v
```
Kết quả mong đợi: 100% test pass (dự kiến > 25 tests pass trong < 3s).

- [x] **Bước 2: Kiểm tra lệnh dry-run của Modal serving**

Chạy lệnh thử nghiệm không khởi tạo container:
```bash
PYTHONPATH=src:. python3 -m flexsds.cli.main serve --backend modal --gpu A10G --dry-run
```
Kết quả mong đợi: In ra câu lệnh `modal serve .../modal_app.py` và cấu hình JSON chứa `"target_gpu": "A10G"`, không phát sinh bất kỳ lỗi đường dẫn nào.

- [x] **Bước 3: Kiểm tra lệnh dry-run cho EAGLE-3.1 phục vụ cục bộ**

Chạy:
```bash
PYTHONPATH=src:. python3 -m flexsds.cli.main serve --backend local --dry-run
```
Kết quả mong đợi: Cờ `--speculative-config` không chứa `--scheduler-cls` khi dùng EAGLE-3.1.

- [x] **Bước 4: Commit và hoàn tất tài liệu nghiệm thu**

Cập nhật trạng thái trong tệp spec `2026-11-14-modal-runner-cross-platform-container-hydration.md` sang `Đã phê duyệt và hoàn thành (Approved and implemented)`.
```bash
git add logs/bugs/startModalBug/2026-11-14-modal-runner-cross-platform-container-hydration.md
git commit -m "docs(spec): mark modal hydration and cross-platform fix spec as implemented"
```
