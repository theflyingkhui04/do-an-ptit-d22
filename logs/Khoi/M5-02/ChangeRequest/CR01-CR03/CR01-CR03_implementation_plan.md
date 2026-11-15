# Kế hoạch thực hiện yêu cầu thay đổi CR01 -> CR03 (Phân hệ Phần cứng & Tương thích)

> **Dành cho kỹ sư/agent thực thi:** BẮT BUỘC SỬ DỤNG KỸ NĂNG: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi từng nhiệm vụ trong kế hoạch này. Các bước sử dụng cú pháp ô kiểm (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Nâng cấp hệ thống FlexSDS đáp ứng trọn vẹn 3 phiếu yêu cầu thay đổi (CR01 $\to$ CR03): tách độc lập lệnh benchmark phần cứng với cơ chế thất bại sớm (fail-fast) khi phục vụ (`REQ-HW-03`), nạp và lưu bộ nhớ đệm `config.json` ngoại tuyến cùng khóa cứng kiến trúc EAGLE (`REQ-CMP-01`), và phát hiện từ vựng dị thể qua kích thước `vocab_size` thực tế (`REQ-CMP-02`).

**Kiến trúc:** Tách bạch hoàn toàn vòng đời đo đạc hiệu chuẩn (`flexsds profile --benchmark`) khỏi vòng đời phục vụ (`flexsds serve`). Quản lý bộ nhớ đệm metadata cục bộ tại thư mục `~/.flexsds/cache/models/` để cho phép kiểm tra tương thích ngoại tuyến mà không phụ thuộc kết nối mạng liên tục; trích xuất `vocab_size` và `architectures` trực tiếp từ tệp cấu hình để xác thực mô hình nháp EAGLE.

**Ngăn xếp công nghệ:** Python 3.10+, PyTorch CUDA, urllib.request chuẩn (không phụ thuộc thư viện mạng ngoài), dataclasses, argparse, pytest.

**Tài liệu đặc tả (Spec):** [logs/Khoi/M5-02/ChangeRequest/CR01->CR03/CR01-CR03_specification.md](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/M5-02/ChangeRequest/CR01->CR03/CR01-CR03_specification.md)

---

## Ràng buộc toàn cục (Global Constraints)

- **Quy tắc chính tả tiếng Việt:** Tuyệt đối không viết hoa chữ cái đầu của tất cả các từ theo kiểu Title Case của tiếng Anh. Chỉ viết hoa chữ cái đầu câu, đầu đoạn, đầu tiêu đề, tên riêng và từ viết tắt chuẩn (GPU, VRAM, OOM, REQ, CLI, EAGLE, TPOT, PTIT, L4, T4, CR, SRS, SDD, JSON, API).
- **Không phụ thuộc thư viện ngoài:** Toàn bộ việc tải metadata và lưu cache chỉ sử dụng thư viện chuẩn của Python (`urllib.request`, `json`, `pathlib`, `os`).
- **Bảo toàn khả năng tương thích:** Không làm phá vỡ các trường hợp phục vụ chuẩn đã có trên GPU NVIDIA L4 (`profile_sm89_l4.json`) và T4 (`profile_sm75_t4.json`).
- **Đường dẫn tệp định dạng chuẩn:** Luôn tạo liên kết có thể nhấp chuột theo định dạng `file:///absolute/path`.

---

## Tiêu điểm rà soát (Review Focus)

Năm trường hợp biên và lỗi tiềm ẩn cần được kiểm chứng chặt chẽ bằng bài kiểm thử tự động:
1. **Mất kết nối mạng khi chưa có cache:** Người dùng chạy kiểm tra mô hình khi offline và chưa từng tải `config.json` trước đó $\to$ phải ném lỗi mạng tường minh `NetworkOfflineError` hoặc trả về thông báo lỗi chi tiết, không để traceback lọt ra ngoài.
2. **Mô hình nháp không thuộc kiến trúc EAGLE:** Người dùng chỉ định mô hình nháp tự hồi quy chuẩn như `Qwen/Qwen2.5-0.5B-Instruct` $\to$ hệ thống phải từ chối phục vụ (`is_compatible = False`) với thông báo chỉ hỗ trợ EAGLE.
3. **Phần cứng GPU chưa có hồ sơ và không có cờ `--allow-fallback`:** Người dùng gõ `flexsds serve` $\to$ hệ thống phải dừng ngay lập tức (exit code 1) và in dòng hướng dẫn chạy `flexsds profile --benchmark`.
4. **Trường hợp `--allow-fallback` được cấp phép:** Người dùng gõ `flexsds serve --allow-fallback` trên GPU lạ $\to$ hệ thống được phép tiếp tục phục vụ với hồ sơ an toàn dự phòng ($k \le 3$).
5. **Đọc tệp `config.json` từ thư mục ổ đĩa cục bộ:** Người dùng truyền đường dẫn cục bộ (ví dụ `/local/checkpoints/qwen2.5-eagle/`) $\to$ hệ thống phải đọc trực tiếp từ đĩa mà không gửi bất kỳ yêu cầu mạng nào.

---

## Kế hoạch công việc chi tiết (Task Decomposition)

### Task 1: Bộ nạp và quản lý bộ nhớ đệm metadata mô hình (CR02 - `REQ-CMP-01`)

**Tệp tin:**
- Sửa đổi: `src/flexsds/core/compatibility.py:48-90`
- Kiểm thử: `tests/test_model_metadata_cache.py`

**Giao diện:**
- Cung cấp:
  - `get_model_config(model_id_or_path: str, cache_dir: Optional[Path] = None, timeout_sec: float = 5.0) -> Dict[str, Any]`
  - Ngoại lệ: `ModelMetadataError(Exception)`, `NetworkOfflineError(ModelMetadataError)`

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho việc tải và lưu cache `config.json`**

Tạo tệp `tests/test_model_metadata_cache.py`:
```python
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from flexsds.core.compatibility import (
    get_model_config,
    NetworkOfflineError,
    MODEL_CACHE_DIR,
)

def test_get_model_config_downloads_and_caches(tmp_path):
    mock_json = b'{"architectures": ["EagleModel"], "vocab_size": 16000}'
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_json
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        cfg = get_model_config("test-org/test-eagle-model", cache_dir=tmp_path)
        assert cfg.get("vocab_size") == 16000
        assert "EagleModel" in cfg.get("architectures", [])

        # Kiểm tra file đã được lưu vào cache
        cached_file = tmp_path / "test-org--test-eagle-model" / "config.json"
        assert cached_file.exists()

def test_get_model_config_reads_from_cache_without_network(tmp_path):
    # Tạo sẵn cache giả lập
    cached_dir = tmp_path / "test-org--test-eagle-model"
    cached_dir.mkdir(parents=True)
    cached_file = cached_dir / "config.json"
    cached_file.write_text(json.dumps({"architectures": ["EagleModel"], "vocab_size": 16000}))

    # Không gọi mạng, đọc từ cache
    with patch("urllib.request.urlopen", side_effect=Exception("Should not call network")):
        cfg = get_model_config("test-org/test-eagle-model", cache_dir=tmp_path)
        assert cfg.get("vocab_size") == 16000

def test_get_model_config_offline_no_cache_raises_network_error(tmp_path):
    with patch("urllib.request.urlopen", side_effect=Exception("Network unreachable")):
        with pytest.raises(NetworkOfflineError):
            get_model_config("non-cached/model-id", cache_dir=tmp_path)
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận lỗi**

Chạy: `PYTHONPATH=src:. pytest tests/test_model_metadata_cache.py -v`  
Kỳ vọng: Thất bại với lỗi `ImportError: cannot import name 'get_model_config'`.

- [ ] **Bước 3: Hiện thực hóa `get_model_config` và quản lý cache trong `src/flexsds/core/compatibility.py`**

Hiện thực hóa:
- Khai báo hằng số `MODEL_CACHE_DIR = Path.home() / ".flexsds" / "cache" / "models"`.
- Định nghĩa lớp ngoại lệ `ModelMetadataError` và `NetworkOfflineError`.
- Triển khai `get_model_config()`: kiểm tra đường dẫn cục bộ $\to$ kiểm tra cache $\to$ nếu không có, tải từ Hugging Face Hub `raw/main/config.json` $\to$ lưu tệp vào cache $\to$ bắt lỗi mạng ném `NetworkOfflineError`.

- [ ] **Bước 4: Chạy kiểm thử để xác nhận thành công**

Chạy: `PYTHONPATH=src:. pytest tests/test_model_metadata_cache.py -v`  
Kỳ vọng: 3 passed in 0.05s.

- [ ] **Bước 5: Commit mã nguồn**

```bash
git add tests/test_model_metadata_cache.py src/flexsds/core/compatibility.py
git commit -m "feat(compat): add model config cache and offline network error handling (CR02)"
```

---

### Task 2: Khóa phạm vi kiến trúc EAGLE và phát hiện từ vựng dị thể qua `vocab_size` (CR02 & CR03)

**Tệp tin:**
- Sửa đổi: `src/flexsds/core/compatibility.py:180-250`
- Sửa đổi/Kiểm thử: `tests/test_compatibility_guard.py`

**Giao diện:**
- Tiêu thụ: `get_model_config` từ Nhiệm vụ 1.
- Cung cấp:
  - `check_compatibility(target_model: str, draft_model: str, free_vram_mb: float = 23000.0, ...) -> CompatibilityReport`
  - Đảm bảo `CompatibilityReport.draft_method == "eagle3"` khi là EAGLE, hoặc từ chối mô hình không phải EAGLE.
  - Đảm bảo `CompatibilityReport.use_heterogeneous_vocab == True` khi `target_vocab != draft_vocab` hoặc là EAGLE.

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho việc khóa EAGLE và so sánh `vocab_size`**

Cập nhật `tests/test_compatibility_guard.py`:
```python
def test_reject_non_eagle_draft_architecture(tmp_path):
    # Mock config target là Qwen2.5 và draft là Qwen2.5-0.5B (không phải EAGLE)
    target_cfg = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064}
    draft_cfg = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064}

    with patch("flexsds.core.compatibility.get_model_config", side_effect=[target_cfg, draft_cfg]):
        report = check_compatibility(
            target_model="Qwen/Qwen2.5-7B-Instruct",
            draft_model="Qwen/Qwen2.5-0.5B-Instruct",
        )
        assert report.is_compatible is False
        assert any("chỉ hỗ trợ kiến trúc đầu nháp EAGLE" in w for w in report.warnings)

def test_accept_eagle_draft_architecture_and_detect_hetero_vocab(tmp_path):
    # Mock config target 152k và draft EAGLE 16k
    target_cfg = {"architectures": ["Qwen2ForCausalLM"], "vocab_size": 152064}
    draft_cfg = {"architectures": ["EagleModel"], "vocab_size": 16000}

    with patch("flexsds.core.compatibility.get_model_config", side_effect=[target_cfg, draft_cfg]):
        report = check_compatibility(
            target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        )
        assert report.is_compatible is True
        assert report.draft_method == "eagle3"
        assert report.use_heterogeneous_vocab is True

def test_shared_vocab_detection_when_same_vocab_size():
    # Giả lập 2 mô hình LLaMA cùng vocab_size và kiến trúc EAGLE-LLaMA
    target_cfg = {"architectures": ["LlamaForCausalLM"], "vocab_size": 128256}
    draft_cfg = {"architectures": ["EagleForCausalLM"], "vocab_size": 128256}

    with patch("flexsds.core.compatibility.get_model_config", side_effect=[target_cfg, draft_cfg]):
        report = check_compatibility(
            target_model="meta-llama/Meta-Llama-3-8B",
            draft_model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        )
        assert report.is_compatible is True
        assert report.draft_method == "eagle3"
        # EAGLE head luôn bật dị thể do tính chất đầu chiếu
        assert report.use_heterogeneous_vocab is True
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận lỗi**

Chạy: `PYTHONPATH=src:. pytest tests/test_compatibility_guard.py::test_reject_non_eagle_draft_architecture -v`  
Kỳ vọng: FAIL vì mã hiện tại vẫn chấp nhận mô hình nháp không phải EAGLE.

- [ ] **Bước 3: Nâng cấp `check_compatibility` trong `src/flexsds/core/compatibility.py`**

Cập nhật `check_compatibility`:
- Gọi `get_model_config()` nạp metadata của `target_model` và `draft_model`.
- Kiểm tra trường `architectures` và `model_type` của `draft_model`:
  - Khai báo tập kiến trúc EAGLE hợp lệ: `EAGLE_ARCHITECTURES = {"EagleModel", "EagleForCausalLM", "Eagle3Model"}`.
  - Nếu không khớp tập trên $\to$ gán `is_compatible = False`, thêm thông báo lỗi: *"Phiên bản FlexSDS hiện tại chỉ hỗ trợ kiến trúc đầu nháp EAGLE. Mô hình nháp '{draft_model}' không được hỗ trợ."*
- Đọc `target_vocab = target_cfg.get("vocab_size", 0)` và `draft_vocab = draft_cfg.get("vocab_size", 0)`.
- Thiết lập `use_hetero_vocab = (target_vocab != draft_vocab) or is_eagle`.

- [ ] **Bước 4: Chạy kiểm thử để xác nhận tất cả đều đạt**

Chạy: `PYTHONPATH=src:. pytest tests/test_compatibility_guard.py -v`  
Kỳ vọng: PASS 100%.

- [ ] **Bước 5: Commit mã nguồn**

```bash
git add tests/test_compatibility_guard.py src/flexsds/core/compatibility.py
git commit -m "feat(compat): enforce EAGLE-only architecture and dynamic vocab_size comparison (CR02, CR03)"
```

---

### Task 3: Tách rời module đo đạc benchmark phần cứng độc lập (CR01 - `REQ-HW-03`)

**Tệp tin:**
- Tạo/Sửa đổi: `src/flexsds/core/cost_model.py:400-470`
- Sửa đổi: `src/flexsds/cli/main.py:165-175`
- Kiểm thử: `tests/test_hardware_profiler_benchmark.py`

**Giao diện:**
- Tiêu thụ: `HardwareInfo`, `save_hardware_profile` từ `cost_model.py`.
- Cung cấp:
  - `run_hardware_micro_benchmark(hw_info: HardwareInfo, target_model: str, draft_model: str, batch_sizes: Optional[List[int]] = None) -> Tuple[FlexCostModel, Path]`
  - Cờ dòng lệnh: `flexsds profile --benchmark [--model <tên>]`

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho `run_hardware_micro_benchmark` và cờ `--benchmark`**

Tạo tệp `tests/test_hardware_profiler_benchmark.py`:
```python
import pytest
from flexsds.core.hardware import HardwareInfo
from flexsds.core.cost_model import (
    run_hardware_micro_benchmark,
    FlexCostModel,
    USER_PROFILES_DIR,
)
from flexsds.cli.main import main

def test_run_hardware_micro_benchmark_creates_profile(tmp_path, monkeypatch):
    import flexsds.core.cost_model as cm
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)

    mock_hw = HardwareInfo(
        device_name="NVIDIA RTX 4090",
        sm_architecture="sm_89",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.4",
        is_cuda_available=True,
    )
    model, saved_path = run_hardware_micro_benchmark(
        hw_info=mock_hw,
        target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        batch_sizes=[1, 2, 4],
        target_dir=tmp_path,
    )
    assert isinstance(model, FlexCostModel)
    assert saved_path.exists()
    assert "sm_89" in saved_path.name
    assert model.forward_target[1] > 0
    assert model.forward_draft[1] > 0

def test_cli_profile_with_benchmark_flag(capsys, monkeypatch, tmp_path):
    import flexsds.core.cost_model as cm
    monkeypatch.setattr(cm, "USER_PROFILES_DIR", tmp_path)

    ret = main(["profile", "--benchmark"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Calibrated & saved new hardware profile" in captured.out
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận lỗi**

Chạy: `PYTHONPATH=src:. pytest tests/test_hardware_profiler_benchmark.py -v`  
Kỳ vọng: FAIL vì chưa có hàm `run_hardware_micro_benchmark` và chưa có cờ `--benchmark`.

- [ ] **Bước 3: Hiện thực hóa `run_hardware_micro_benchmark` và tích hợp CLI**

Trong `src/flexsds/core/cost_model.py`:
- Viết hàm `run_hardware_micro_benchmark(...)`: thực hiện vòng lặp đo đạc giả lập hoặc đo đạc forward pass an toàn, tính toán trung vị $c_t, c_d$, tạo payload JSON chuẩn và gọi `save_hardware_profile()`.

Trong `src/flexsds/cli/main.py`:
- Thêm cờ `--benchmark` vào `profile_parser.add_argument("--benchmark", action="store_true", help="Run micro-benchmark to measure and save profile")`.
- Khi `args.command == "profile"` và `args.benchmark`:
  - Thực thi `run_hardware_micro_benchmark(hw_info, ...)`.
  - In thông báo: `✔ Calibrated & saved new hardware profile: <đường_dẫn_tệp>`.

- [ ] **Bước 4: Chạy kiểm thử để xác nhận thành công**

Chạy: `PYTHONPATH=src:. pytest tests/test_hardware_profiler_benchmark.py -v`  
Kỳ vọng: 2 passed in 0.20s.

- [ ] **Bước 5: Commit mã nguồn**

```bash
git add tests/test_hardware_profiler_benchmark.py src/flexsds/core/cost_model.py src/flexsds/cli/main.py
git commit -m "feat(profiler): implement standalone hardware micro-benchmark command (CR01)"
```

---

### Task 4: Cơ chế thất bại sớm (Fail-Fast) khi phục vụ thiếu hồ sơ phần cứng (CR01 - `REQ-HW-03`)

**Tệp tin:**
- Sửa đổi: `src/flexsds/core/cost_model.py:load_profile_for_hardware`
- Sửa đổi: `src/flexsds/cli/main.py:serve` parser và action
- Kiểm thử: `tests/test_serving_profile_guard.py`

**Giao diện:**
- Tiêu thụ: `load_profile_for_hardware(hw_info, strict=True)`.
- Cung cấp:
  - Cờ CLI `--allow-fallback` trong lệnh `flexsds serve`.
  - Khả năng dừng tiến trình (exit code 1) khi phát hiện phần cứng chưa được benchmark.

- [ ] **Bước 1: Viết bài kiểm thử thất bại cho cơ chế fail-fast khi `serve`**

Tạo tệp `tests/test_serving_profile_guard.py`:
```python
import pytest
from unittest.mock import patch
from flexsds.core.hardware import HardwareInfo
from flexsds.cli.main import main

def test_serve_fails_fast_when_missing_profile_without_fallback_flag(capsys):
    unknown_hw = HardwareInfo(
        device_name="NVIDIA Custom GPU",
        sm_architecture="sm_99",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.6",
        is_cuda_available=True,
    )
    with patch("flexsds.cli.main.detect_hardware", return_value=unknown_hw):
        # Chạy lệnh serve không truyền --allow-fallback
        ret = main(["serve", "--backend", "local", "--dry-run"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Chưa có hồ sơ độ trễ cho GPU" in captured.out
        assert "flexsds profile --benchmark" in captured.out

def test_serve_proceeds_when_allow_fallback_flag_provided(capsys):
    unknown_hw = HardwareInfo(
        device_name="NVIDIA Custom GPU",
        sm_architecture="sm_99",
        total_vram_mb=24576.0,
        free_vram_mb=20000.0,
        cuda_version="12.6",
        is_cuda_available=True,
    )
    with patch("flexsds.cli.main.detect_hardware", return_value=unknown_hw):
        # Chạy lệnh serve có truyền cờ --allow-fallback
        ret = main(["serve", "--backend", "local", "--dry-run", "--allow-fallback"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Conservative Fallback" in captured.out or "conservative_analytical_fallback" in captured.out
```

- [ ] **Bước 2: Chạy kiểm thử để xác nhận lỗi**

Chạy: `PYTHONPATH=src:. pytest tests/test_serving_profile_guard.py -v`  
Kỳ vọng: FAIL vì lệnh `serve` hiện tại chưa kiểm tra cờ `--allow-fallback` và không trả về mã thoát `1`.

- [ ] **Bước 3: Cập nhật hàm `load_profile_for_hardware` và lệnh `serve`**

Trong `src/flexsds/core/cost_model.py`:
- Thêm tham số `strict: bool = False` cho `load_profile_for_hardware(hw_info, custom_profile_path=None, strict=False)`.
- Nếu `strict=True` mà không tìm thấy hồ sơ trong tệp chỉ định, thư mục cá nhân `~/.flexsds/profiles/` hay kho chuẩn $\to$ trả về `(None, "")`.

Trong `src/flexsds/cli/main.py`:
- Thêm cờ `--allow-fallback` vào parser của lệnh `serve`:
  `serve_parser.add_argument("--allow-fallback", action="store_true", help="Allow running with conservative analytical fallback if exact GPU profile is missing")`.
- Trong hàm xử lý `serve`:
  - Gọi `model, profile_name = load_profile_for_hardware(hw_info, strict=not args.allow_fallback)`.
  - Nếu `model is None`:
    - In thông báo lỗi màu đỏ:
      `✖ [Lỗi phần cứng] Chưa có hồ sơ độ trễ cho GPU '{hw_info.device_name} ({hw_info.sm_architecture})'. Vui lòng chạy lệnh đo đạc: flexsds profile --benchmark (hoặc thêm cờ '--allow-fallback').`
    - Trả về mã thoát `1`.

- [ ] **Bước 4: Chạy kiểm thử để xác nhận thành công**

Chạy: `PYTHONPATH=src:. pytest tests/test_serving_profile_guard.py -v`  
Kỳ vọng: 2 passed in 0.15s.

- [ ] **Bước 5: Commit mã nguồn**

```bash
git add tests/test_serving_profile_guard.py src/flexsds/core/cost_model.py src/flexsds/cli/main.py
git commit -m "feat(cli): enforce fail-fast guard when serving with uncalibrated hardware (CR01)"
```

---

### Task 5: Kiểm thử toàn diện và đồng bộ hóa tài liệu SRS & SDD

**Tệp tin:**
- Chạy: Toàn bộ bài kiểm thử `tests/`
- Cập nhật: `scripts/build_full_srs_excel.py`, `scripts/build_sdd_excel.py`
- Tái tạo: Các tệp SRS/SDD `.xlsx`, `.csv`, `.md` trong `logs/Khoi/spec&plan/`

**Giao diện:**
- Đảm bảo 100% test cases (75+ tests) đạt PASS.
- Đảm bảo các tài liệu SRS và SDD phản ánh đúng trạng thái của CR01, CR02, CR03.

- [ ] **Bước 1: Chạy toàn bộ bộ kiểm thử FlexSDS**

Chạy:
```bash
PYTHONPATH=src:. pytest tests/test_flexsds*.py tests/test_cli_*.py tests/test_task_runner.py tests/test_credentials_manager.py tests/test_compatibility_guard.py tests/test_end_to_end_flow.py tests/test_cost_model_*.py tests/test_model_metadata_cache.py tests/test_hardware_profiler_benchmark.py tests/test_serving_profile_guard.py -v
```
Kỳ vọng: Tất cả các bài kiểm thử đạt PASS 100%.

- [ ] **Bước 2: Cập nhật kịch bản sinh dữ liệu SRS và SDD**

Chạy:
```bash
python3 scripts/build_full_srs_excel.py && python3 scripts/build_sdd_excel.py
```
Kỳ vọng: Tái tạo thành công toàn bộ các tệp `.xlsx`, `.csv`, `.md` của SRS và SDD.

- [ ] **Bước 3: Kiểm tra trạng thái git**

Chạy: `git status`  
Kỳ vọng: Cây làm việc sạch sẽ, các tệp sửa đổi nằm trong danh mục dự kiến.

- [ ] **Bước 4: Commit và hoàn tất**

```bash
git add scripts/build_full_srs_excel.py scripts/build_sdd_excel.py logs/Khoi/spec&plan/
git commit -m "docs(spec): synchronize SRS and SDD documents with CR01-CR03 implementation"
```
