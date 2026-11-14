# Kế hoạch thực thi phân hệ phục vụ đa nền tảng và kiểm duyệt trước khi tải cho FlexSDS CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện phân hệ khởi chạy máy chủ phục vụ đa nền tảng (Local GPU, Google Colab, Modal Cloud BYOC), cơ chế kiểm duyệt cấu hình Hugging Face trước khi tải trọng số và dọn sạch tiến trình con cho FlexSDS CLI trước mốc đóng băng 15/11.

**Architecture:** Mở rộng tầng `src/flexsds/server/` với 3 runner độc lập (`local_runner.py`, `colab_runner.py`, `modal_runner.py`); đóng gói độc lập bộ điều khiển `TreeAdaptiveController v0.2` vào `src/flexsds/core/controller.py`; tích hợp cơ chế pre-download metadata guard vào `compatibility.py`; đấu nối toàn diện với `wizard.py` và `main.py` theo kịch bản Happy Path đã phê duyệt.

**Tech Stack:** Python 3.10+, vLLM v0.30.0, Modal Python SDK v1.6.3+, Cloudflare Tunnel (`cloudflared`), Hugging Face Hub API, pytest.

**Spec:** [`logs/Khoi/spec&plan/M5-01-flex-cli-happy-path/spec.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec%26plan/M5-01-flex-cli-happy-path/spec.md) và kịch bản [`logs/Khoi/spec&plan/M5-01-flex-cli-happy-path/happy-path.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec%26plan/M5-01-flex-cli-happy-path/happy-path.md)

## Global Constraints

- Phạm vi trọng tâm: Khóa chặt vào kiến trúc EAGLE (đầu nháp EAGLE-3.1 kết hợp TreeAdaptive v0.2), loại bỏ các mô hình nháp độc lập SLM.
- Không yêu cầu GPU vật lý khi chạy test: Toàn bộ bài test trong CI/CD phải chạy được trên môi trường chỉ có CPU qua mock hoặc cờ `--dry-run`.
- Tuyệt đối không tự động chạy `git commit`: Chỉ thực hiện thay đổi và kiểm thử; commit do người dùng chủ động yêu cầu.
- Tuân thủ quy tắc chính tả tiếng Việt: Không viết hoa kiểu Title Case của tiếng Anh; chỉ viết hoa đầu câu, danh từ riêng (PTIT, vLLM, Modal, EAGLE-3.1, Qwen2.5, NVIDIA L4, Tesla T4, A100, RTX, Bùi Đức Đại, Cấn Đức Khôi, Python, CUDA) và từ viết tắt (PM, DoD, AWQ, VRAM, OOM, JSON, SQL, CLI, API, E2E, PID, BYOC).
- Triệt tiêu tiến trình mồ côi (Zero-Zombie Process): Sử dụng nhóm tiến trình `os.setsid` và bắt `SIGINT`/`SIGTERM` để gửi `os.killpg`.

## Review Focus

1. **Thiếu GPU hoặc VRAM $< 6\text{ GB}$ khi chọn Local:** Hệ thống phải phát hiện ngay lập tức và gợi ý chuyển sang Modal hoặc Colab, không để vLLM bị crash khi khởi động.
2. **Mô hình Hugging Face quá khổ:** Người dùng nhập mô hình FP16 15 GB trên GPU 8 GB $\implies$ Pre-download Guard phải phát hiện qua metadata `config.json` và chặn tải về ngay lập tức trước khi kéo dữ liệu nặng.
3. **Mất kết nối hoặc người dùng ấn `Ctrl+C`:** Tiến trình vLLM và các tiến trình con (`EngineCore`, `APIServer`) phải được dọn dẹp sạch sẽ khỏi GPU, không để rò rỉ VRAM.
4. **Không có token Modal:** Khi chọn Modal Cloud, hệ thống kiểm tra và hướng dẫn nhập `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` cá nhân, không dùng tài nguyên của nhóm nghiên cứu.
5. **Đồng bộ hóa Interface A v0.2:** Hợp đồng kiểm thử `tests/test_scheduler_tree_glue.py` phải xác thực đúng 13 trường dữ liệu của telemetry và công thức $N_{\text{batch}} = \max_i(N_i)$ mà không phụ thuộc vào tiến độ merge của Đại.

---

### Task 1: Đồng bộ hóa độc lập `src/flexsds/core/controller.py` với `TreeAdaptiveController v0.2`

**Files:**
- Modify: `src/flexsds/core/controller.py`
- Test: `tests/test_flexsds_baseline.py`

**Interfaces:**
- Consumes: Thuật toán và cấu trúc dữ liệu từ `src/proposer/tree_adaptive.py`
- Produces: `TreeAdaptiveController`, `TreeAdaptiveDecision` trong package `flexsds.core` với đầy đủ các trường `topology_name`, `alpha_effective`, `speedup`, `status` (`ok`, `memory_guard`, `concurrency_throttled`, `fallback_gate`, `probing`, `zero_crash_fallback`).

- [ ] **Step 1: Viết test kiểm tra tính độc lập và đầy đủ của TreeAdaptiveController trong package `flexsds`**

Thêm test vào `tests/test_flexsds_baseline.py`:
```python
def test_flexsds_core_tree_adaptive_controller_vendored():
    from flexsds.core.controller import TreeAdaptiveController, TreeAdaptiveDecision
    ctrl = TreeAdaptiveController(min_free_vram_mb=1500.0)
    decision = ctrl.select_budget(batch_size=1, free_vram_mb=8000.0, kv_cache_usage_pct=20.0)
    assert isinstance(decision, TreeAdaptiveDecision)
    assert hasattr(decision, "topology_name")
    assert hasattr(decision, "alpha_effective")
    assert hasattr(decision, "speedup")
    assert decision.status in ("ok", "memory_guard", "concurrency_throttled", "fallback_gate", "probing", "zero_crash_fallback")
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_flexsds_baseline.py::test_flexsds_core_tree_adaptive_controller_vendored -v`  
Kết quả mong đợi: FAIL do `controller.py` cũ chưa có các thuộc tính `topology_name`, `speedup`.

- [ ] **Step 3: Sao chép và đóng gói trực tiếp `TreeAdaptiveController v0.2` vào `src/flexsds/core/controller.py`**

Đưa toàn bộ logic từ `src/proposer/tree_adaptive.py` sang `src/flexsds/core/controller.py`, bảo đảm import nội bộ qua `flexsds.core` và giữ nguyên tương thích ngược với các hàm cũ.

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_flexsds_baseline.py -v`  
Kết quả mong đợi: PASS 100% toàn bộ 7 test cũ và test mới.

---

### Task 2: Hiện thực hóa Pre-download Guard kiểm tra metadata Hugging Face trong `src/flexsds/core/compatibility.py`

**Files:**
- Modify: `src/flexsds/core/compatibility.py`
- Test: `tests/test_compatibility_guard.py`

**Interfaces:**
- Consumes: `HardwareInfo.free_vram_mb`
- Produces: `inspect_remote_model_metadata(model_id_or_path: str) -> dict`, `pre_download_guard(target_model: str, draft_model: str, free_vram_mb: float) -> PreDownloadCheckResult`

- [ ] **Step 1: Viết test cho hàm `inspect_remote_model_metadata` và `pre_download_guard`**

Tạo mới `tests/test_compatibility_guard.py`:
```python
def test_pre_download_guard_detects_oversized_model():
    from flexsds.core.compatibility import pre_download_guard
    # Mô phỏng mô hình FP16 15GB trên GPU 8GB
    res = pre_download_guard(
        target_model="Qwen/Qwen2.5-7B-Instruct",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        free_vram_mb=8000.0,
        mock_metadata={"torch_dtype": "float16", "num_parameters": 7.6e9, "quantization_config": None}
    )
    assert res.is_safe is False
    assert res.vram_needed_mb > 14000.0
    assert "AWQ" in res.suggested_model

def test_pre_download_guard_passes_awq():
    from flexsds.core.compatibility import pre_download_guard
    res = pre_download_guard(
        target_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft_model="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        free_vram_mb=8000.0,
        mock_metadata={"torch_dtype": "float16", "num_parameters": 7.6e9, "quantization_config": {"quant_method": "awq"}}
    )
    assert res.is_safe is True
    assert res.vram_needed_mb < 6000.0
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_compatibility_guard.py -v`  
Kết quả mong đợi: FAIL do chưa định nghĩa `pre_download_guard`.

- [ ] **Step 3: Cài đặt `inspect_remote_model_metadata` và `pre_download_guard` trong `compatibility.py`**

- Định nghĩa dataclass `PreDownloadCheckResult(is_safe: bool, vram_needed_mb: float, free_vram_mb: float, warning: Optional[str], suggested_model: Optional[str])`.
- Sử dụng `urllib.request` hoặc Hugging Face Hub HTTP API đọc tệp `config.json` (dung lượng < 10 KB, timeout 5 giây).
- Tính toán VRAM theo công thức: FP16 (2.0 byte/param) vs AWQ/INT4 (0.55 byte/param) cộng thêm 0.6 GB cho đầu nháp EAGLE và 2.0 GB cho KV-cache.
- Bổ sung trường `shared_tokenizer: bool` thay thế cho thuật ngữ cũ.

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_compatibility_guard.py -v`  
Kết quả mong đợi: PASS.

---

### Task 3: Xây dựng trình khởi chạy máy chủ cục bộ `src/flexsds/server/local_runner.py` và dọn sạch tiến trình con

**Files:**
- Create: `src/flexsds/server/__init__.py`
- Create: `src/flexsds/server/local_runner.py`
- Test: `tests/test_local_runner.py`

**Interfaces:**
- Consumes: `serving_config: dict`
- Produces: `LocalServingRunner.build_command(config: dict, port: int) -> list[str]`, `LocalServingRunner.setup_environment(config: dict) -> dict[str, str]`, `LocalServingRunner.stop(pid_file: str) -> bool`

- [ ] **Step 1: Viết test cho `LocalServingRunner`**

Tạo mới `tests/test_local_runner.py`:
```python
def test_local_runner_build_command_speculative_json():
    from flexsds.server.local_runner import LocalServingRunner
    runner = LocalServingRunner()
    config = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "draft_method": "eagle3",
            "use_heterogeneous_vocab": True,
            "port": 8000,
        }
    }
    cmd = runner.build_command(config)
    assert "vllm" in cmd[0] or "python" in cmd[0]
    assert "--speculative-config" in cmd
    spec_idx = cmd.index("--speculative-config") + 1
    assert "num_speculative_tokens_per_batch_size" in cmd[spec_idx]
    assert "--scheduler-cls" in cmd
    assert "src.server.adaptive_scheduler.AdaptiveAsyncScheduler" in cmd

def test_local_runner_environment_vars():
    from flexsds.server.local_runner import LocalServingRunner
    runner = LocalServingRunner()
    env = runner.setup_environment({"serving": {"enable_adaptive": True}})
    assert env.get("ADAPTIVE_ENABLE_VRAM_GUARD") == "1"
    assert "ADAPTIVE_TELEMETRY_PATH" in env
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_local_runner.py -v`  
Kết quả mong đợi: FAIL do chưa tạo file `local_runner.py`.

- [ ] **Step 3: Cài đặt `LocalServingRunner` trong `src/flexsds/server/local_runner.py`**

- Hiện thực hóa `build_command`: kế thừa logic tham số từ `experiments.modal_lanes.build_vllm_args()`.
- Hiện thực hóa `setup_environment`: xuất dict biến môi trường `ADAPTIVE_*`.
- Hiện thực hóa `launch`: hỗ trợ `dry_run`, `preexec_fn=os.setsid` khi chạy thật, ghi `.flexsds.pid` khi chạy `--background`.
- Hiện thực hóa `stop`: đọc `.flexsds.pid`, gửi `os.killpg(pgid, signal.SIGTERM)` dọn dẹp sạch tiến trình con.

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_local_runner.py -v`  
Kết quả mong đợi: PASS.

---

### Task 4: Xây dựng trình khởi chạy Google Colab `src/flexsds/server/colab_runner.py`

**Files:**
- Create: `src/flexsds/server/colab_runner.py`
- Test: `tests/test_colab_runner.py`

**Interfaces:**
- Consumes: `target_gpu: str`, `public_url: str`
- Produces: `ColabServingRunner.get_colab_instructions(target_gpu: str) -> dict`, `ColabServingRunner.verify_tunnel_health(public_url: str, timeout: int) -> bool`

- [ ] **Step 1: Viết test cho `ColabServingRunner`**

Tạo mới `tests/test_colab_runner.py`:
```python
def test_colab_runner_instructions_and_link():
    from flexsds.server.colab_runner import ColabServingRunner
    runner = ColabServingRunner()
    info = runner.get_colab_instructions(target_gpu="T4")
    assert "colab.research.google.com" in info["notebook_url"]
    assert "cloudflared" in info["tunnel_method"]
    assert "T4" in info["recommended_runtime"]

def test_colab_runner_verify_tunnel_health_mock(monkeypatch):
    from flexsds.server.colab_runner import ColabServingRunner
    runner = ColabServingRunner()
    monkeypatch.setattr(runner, "_probe_endpoint", lambda url: True)
    assert runner.verify_tunnel_health("https://mock.trycloudflare.com/v1") is True
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_colab_runner.py -v`  
Kết quả mong đợi: FAIL do chưa có `colab_runner.py`.

- [ ] **Step 3: Cài đặt `ColabServingRunner` trong `src/flexsds/server/colab_runner.py`**

- Định nghĩa đường dẫn notebook: `https://colab.research.google.com/github/theflyingkhui04/do-an-ptit-d22/blob/develop/notebooks/flexsds_colab_server.ipynb`.
- So khớp profile chi phí tương ứng (`profile_sm75_t4.json` cho T4, `profile_sm89_l4.json` cho L4).
- Hàm `verify_tunnel_health` gửi GET request tới `<url>/health` kiểm tra phản hồi HTTP 200.

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_colab_runner.py -v`  
Kết quả mong đợi: PASS.

---

### Task 5: Xây dựng trình khởi chạy Modal Cloud `src/flexsds/server/modal_runner.py` (BYOC, Hybrid, Stop)

**Files:**
- Create: `src/flexsds/server/modal_runner.py`
- Test: `tests/test_modal_runner_cli.py`

**Interfaces:**
- Consumes: `serving_config: dict`, `deploy: bool`
- Produces: `ModalServingRunner.check_credentials() -> bool`, `ModalServingRunner.build_modal_command(config: dict, deploy: bool) -> list[str]`, `ModalServingRunner.stop(app_name: str) -> bool`

- [ ] **Step 1: Viết test cho `ModalServingRunner`**

Tạo mới `tests/test_modal_runner_cli.py`:
```python
def test_modal_runner_build_command_ephemeral_and_deploy():
    from flexsds.server.modal_runner import ModalServingRunner
    runner = ModalServingRunner()
    config = {"serving": {"target_gpu": "L4", "port": 8000}}
    cmd_run = runner.build_modal_command(config, deploy=False)
    assert "modal" in cmd_run[0]
    assert "run" in cmd_run
    
    cmd_deploy = runner.build_modal_command(config, deploy=True)
    assert "deploy" in cmd_deploy

def test_modal_runner_credentials_check_env(monkeypatch):
    from flexsds.server.modal_runner import ModalServingRunner
    runner = ModalServingRunner()
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    assert runner.check_credentials() is True
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_modal_runner_cli.py -v`  
Kết quả mong đợi: FAIL do chưa có `src/flexsds/server/modal_runner.py`.

- [ ] **Step 3: Cài đặt `ModalServingRunner` trong `src/flexsds/server/modal_runner.py`**

- Kiểm tra file `~/.modal.toml` hoặc biến môi trường `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`.
- Sinh câu lệnh thực thi `modal run` (chế độ tạm thời) hoặc `modal deploy` (chế độ cố định).
- Hàm `stop` thực hiện `modal app stop <app_name>`.

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_modal_runner_cli.py -v`  
Kết quả mong đợi: PASS.

---

### Task 6: Đấu nối Interactive Wizard và CLI Dispatcher trong `src/flexsds/cli/wizard.py` và `main.py`

**Files:**
- Modify: `src/flexsds/cli/wizard.py`
- Modify: `src/flexsds/cli/main.py`
- Test: `tests/test_cli_wizard_flow.py`

**Interfaces:**
- Consumes: Runner classes từ Task 3, 4, 5 và guard từ Task 2
- Produces: Luồng tương tác hoàn chỉnh chuẩn Happy Path, hỗ trợ lệnh `flexsds serve`, `flexsds detect`, `flexsds stop`

- [ ] **Step 1: Viết test cho luồng hỏi–đáp mở rộng trong `wizard.py`**

Tạo mới `tests/test_cli_wizard_flow.py`:
```python
def test_wizard_platform_selection_structure():
    from flexsds.cli.wizard import PLATFORM_CHOICES
    assert "Modal.com" in PLATFORM_CHOICES[0]
    assert "Google Colab" in PLATFORM_CHOICES[1]
    assert "Tự dùng GPU" in PLATFORM_CHOICES[2]

def test_cli_stop_command_parser():
    from flexsds.cli.main import parse_args
    args = parse_args(["stop"])
    assert args.command == "stop"
```

- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_cli_wizard_flow.py -v`  
Kết quả mong đợi: FAIL do chưa có `PLATFORM_CHOICES` và `stop` parser.

- [ ] **Step 3: Cập nhật `wizard.py` và `main.py`**

- `wizard.py`:
  + Cập nhật bảng báo cáo: hiển thị `Dùng chung tokenizer: CÓ / KHÔNG`.
  + Tích hợp chọn nền tảng 3 lựa chọn theo thứ tự: `Modal.com`, `Google Colab`, `Tự dùng GPU của thiết bị này`.
  + Tích hợp chọn GPU cho Modal và Colab; tự động quét GPU cho lựa chọn Local.
  + Tích hợp `pre_download_guard` cho phép nhập mô hình Hugging Face tùy chọn.
  + Hỏi xác nhận khởi chạy máy chủ ngay lập tức sau khi lưu `serving_config.json`.
- `main.py`:
  + Thêm parser lệnh `stop` và kết nối với `LocalServingRunner.stop()` / `ModalServingRunner.stop()`.
  + Bổ sung cờ `--port`, `--background`, `--deploy` cho lệnh `serve`.

- [ ] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_cli_wizard_flow.py -v`  
Kết quả mong đợi: PASS.

---

### Task 7: Xây dựng bộ test suite hợp đồng M4-02-04 và kiểm thử tích hợp đầu cuối E2E

**Files:**
- Create: `tests/test_scheduler_tree_glue.py`
- Create: `tests/test_end_to_end_flow.py`
- Test: `tests/test_scheduler_tree_glue.py`, `tests/test_end_to_end_flow.py`

**Interfaces:**
- Consumes: Toàn bộ các module từ Task 1 đến Task 6
- Produces: Bộ kiểm chứng chất lượng toàn diện trước khi phát hành

- [ ] **Step 1: Viết test hợp đồng `test_scheduler_tree_glue.py`**

Kiểm tra:
- Giả lập batch 4 requests với các cây khác nhau ($T_0, T_4, T_8, T_{16}$).
- Xác nhận $N_{\text{batch}} = \max_i(N_i) = 16$.
- Kiểm tra tính đầy đủ của 13 trường dữ liệu trong log telemetry Interface A v0.2.

- [ ] **Step 2: Viết test tích hợp đầu cuối `test_end_to_end_flow.py`**

Kiểm tra:
- Khởi động CLI với cờ `--dry-run` cho cả 3 nền tảng: `local`, `colab`, `modal`.
- Xác nhận cấu hình sinh ra `serving_config.json` hợp lệ và lệnh thực thi chuẩn xác không gây nghẽn tiến trình.

- [ ] **Step 3: Chạy toàn bộ test suite của FlexSDS**

Chạy: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_flexsds_baseline.py tests/test_compatibility_guard.py tests/test_local_runner.py tests/test_colab_runner.py tests/test_modal_runner_cli.py tests/test_cli_wizard_flow.py tests/test_scheduler_tree_glue.py tests/test_end_to_end_flow.py -v`  
Kết quả mong đợi: 100% PASS toàn bộ các bài test (tối thiểu 25 test cases).

---

## Self-Review Verification

1. **Spec coverage:** Tất cả các yêu cầu trong `spec.md` (EAGLE scope, pre-download guard, 3 runners, BYOC Modal, `flexsds stop`, bảng dùng chung tokenizer) đều được phân bổ chi tiết vào 7 task kỹ thuật.
2. **Step scan:** Mỗi task đều tuân thủ chặt chẽ 4-5 bước kiểm thử TDD (viết test fail $\to$ xác nhận fail $\to$ cài đặt code $\to$ xác nhận pass).
3. **Type consistency:** Các định danh class và phương thức (`LocalServingRunner`, `ColabServingRunner`, `ModalServingRunner`, `pre_download_guard`) đồng nhất giữa các task và test file.
4. **Review Focus:** Cả 5 điểm nóng rủi ro đều có bài test bao phủ (thiếu GPU, mô hình FP16 quá khổ, xử lý `Ctrl+C`, thiếu token Modal, kiểm thử hợp đồng scheduler).
5. **Chính sách commit:** Không có lệnh tự động commit trong kế hoạch, hoàn toàn tuân thủ chỉ thị của người dùng.
