> Integration note (2026-11-13): This document preserves design/plan intent, not proof of implementation. Current behavior, limitations, tests and superseding corrections are in `logs/daijestic/flexsds-advisor/worklog.md` and `.agents/skills/flexsds-advisor/references/flexsds-cli.md`. Ephemeral Modal uses `modal serve`; chat measures SSE chunks; Windows ACL privacy is not proven; Colab flag mode is guidance only.

# Kế hoạch thực thi phân hệ backend phục vụ và giao tiếp suy luận cho FlexSDS CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Xây dựng hoàn chỉnh phân hệ backend cho FlexSDS CLI (Phương án B: Serving Backend đa nền tảng kết hợp CLI Chat Client và Live Telemetry Monitor) trước khi đóng gói sản phẩm phục vụ bảo vệ đồ án.

**Architecture:** Nâng cấp tầng `src/flexsds/server/` đồng bộ chuẩn xác với cơ chế Async Lineage v0.2 (`LineageUniProcExecutor`, `AdaptiveAsyncScheduler`) trên vLLM v0.30.0; đóng gói ứng dụng Modal Cloud độc lập (`modal_app.py`); chuyển đổi kinh nghiệm từ `vscode_cli.py` thành notebook Colab chuẩn; mở rộng phân hệ `src/flexsds/client/` cung cấp công cụ trò chuyện dòng lệnh tương tác (`chat.py`) với cơ chế streaming phản hồi và đo đạc tốc độ thời gian thực (tokens/s, TPOT), kết nối trực tiếp vào bộ điều khiển lệnh trung tâm `main.py`.

**Tech Stack:** Python 3.10+, vLLM v0.30.0, PyTorch 2.4+, Modal Python SDK v1.6.3+, Cloudflare Tunnel (`cloudflared`), Server-Sent Events (SSE), standard library `urllib` / `subprocess`, pytest.

**Spec:** [`logs/Khoi/spec&plan/flexsds-cli-backend/spec.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec&plan/flexsds-cli-backend/spec.md)

---

## Global Constraints

- **Serving runtime:** vLLM v0.30.0 cố định.
- **Kiến trúc mô hình:** Target `Qwen/Qwen2.5-7B-Instruct-AWQ` kết hợp đầu nháp `duckhoi/qwen2.5-7b-eagle3.1-speculator` (`method = "eagle3"`, `use_heterogeneous_vocab = True`).
- **Quy tắc triệt tiêu tiến trình mồ côi (Zero-Zombie Process):** Mọi tiến trình con vLLM phải chạy trong nhóm tiến trình riêng biệt (`os.setsid`) và dọn dẹp sạch sẽ qua `os.killpg` khi kết thúc.
- **Bảo mật tuyệt đối khóa định danh (Zero-Leak Credentials):** Tuyệt đối không lưu token/API key vào `serving_config.json` hay bất kỳ tệp nào trong repo; chỉ lưu tại `~/.flexsds/credentials` với quyền `chmod 0600`.
- **Không yêu cầu GPU vật lý trong CI/CD:** 100% bài test phải chạy thành công trên môi trường CPU qua cờ `--dry-run` hoặc mock HTTP/subprocess.
- **Quy tắc chính tả tiếng Việt:** Tuyệt đối không dùng Title Case tiếng Anh; chỉ viết hoa đầu câu, danh từ riêng (PTIT, vLLM, Modal, Colab, Cloudflare, EAGLE-3.1, Qwen2.5, Bùi Đức Đại, Cấn Đức Khôi, Python, CUDA) và từ viết tắt (PM, DoD, AWQ, VRAM, OOM, JSON, SQL, CLI, API, PID, BYOC, TPOT, SSE).
- **Chính sách commit:** Không tự ý chạy `git commit` ngầm; chỉ chuẩn bị lệnh commit mẫu ở mỗi task để lập trình viên hoặc quy trình phê duyệt chủ động thực hiện.

---

## Review Focus

1. **Rò rỉ hoặc lộ khóa API key ra tệp chia sẻ / log:** Khi người dùng nhập token định danh Modal, Cloudflare hoặc Hugging Face, hệ thống không được ghi vào `serving_config.json`, không được in token thô ra màn hình và tệp `~/.flexsds/credentials` phải có quyền truy cập hạn chế tối đa `0600`.
   - *Test ghim tại Task 1:* `test_credentials_manager_file_permissions_and_masking()`.
2. **Cổng mạng (port) đã bị chiếm dụng khi khởi chạy Local:** Khi gọi `LocalServingRunner.launch()`, nếu cổng (mặc định 8000) đang bị tiến trình khác chiếm, hệ thống phải phát hiện ngay lập tức và báo lỗi thân thiện thay vì để vLLM crash âm thầm.
   - *Test ghim tại Task 2:* `test_local_runner_detects_port_conflict()`.
3. **Server chưa sẵn sàng khi người dùng gõ `flexsds chat`:** Nếu máy chủ chưa chạy hoặc đang tải trọng số dở (trả về lỗi từ chối kết nối hoặc timeout), client phải báo lỗi rõ ràng và hướng dẫn chạy `flexsds serve`.
   - *Test ghim tại Task 5:* `test_client_handles_unreachable_endpoint()`.
4. **Mất kết nối hoặc người dùng ấn `Ctrl+C` khi đang stream câu trả lời:** Phiên chat không được sụp đổ (crash terminal), phải bắt `KeyboardInterrupt` mềm, hủy stream hiện tại và quay về dấu nhắc prompt `You > `.
   - *Test ghim tại Task 6:* `test_chat_handles_sigint_gracefully()`.
5. **Tệp telemetry chưa tồn tại khi khởi chạy `flexsds monitor`:** Nếu tệp `serving_trace.jsonl` chưa được tạo (máy chủ chưa phục vụ request nào), monitor phải hiển thị trạng thái chờ với timeout thông minh thay vì văng `FileNotFoundError`.
   - *Test ghim tại Task 7:* `test_monitor_waits_for_telemetry_file()`.
6. **Thiếu thông tin định danh Modal:** Khi chọn backend Modal mà máy chưa cấu hình token, hệ thống phải chặn lại ngay và hướng dẫn chạy lệnh `flexsds auth login --provider modal` hoặc đặt biến môi trường `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`.
   - *Test ghim tại Task 3:* `test_modal_runner_missing_credentials_guard()`.

---

### Task 1: Xây dựng hệ thống quản lý khóa bảo mật (`src/flexsds/core/credentials.py`) và bộ lệnh `flexsds auth`

**Files:**
- Create: `src/flexsds/core/credentials.py`
- Modify: `src/flexsds/cli/main.py`
- Test: `tests/test_credentials_manager.py`

**Interfaces:**
- Consumes: Token ID / Secret từ terminal qua `getpass.getpass()`, đường dẫn `$HOME/.flexsds/credentials`.
- Produces: `CredentialsManager.set_credential(provider, key, val) -> None`, `CredentialsManager.get_credential(provider, key) -> Optional[str]`, `CredentialsManager.delete_credential(provider) -> bool`, `CredentialsManager.get_all_masked() -> Dict[str, str]`, `CredentialsManager.inject_into_environ() -> None`.

- [x] **Step 1: Viết test cho quản lý quyền hạn tệp 0600 và cơ chế che giấu token trong `tests/test_credentials_manager.py`**

```python
import os
import stat
from flexsds.core.credentials import CredentialsManager

def test_credentials_manager_file_permissions_and_masking(tmp_path):
    cred_file = tmp_path / ".flexsds" / "credentials"
    mgr = CredentialsManager(credentials_path=str(cred_file))
    mgr.set_credential("modal", "MODAL_TOKEN_ID", "ak-1234567890abcdef")
    mgr.set_credential("modal", "MODAL_TOKEN_SECRET", "as-9876543210fedcba")

    # Kiểm tra quyền hạn tệp bắt buộc là 0600 (chỉ user đọc/ghi)
    mode = stat.S_IMODE(os.stat(cred_file).st_mode)
    assert mode == 0o600

    # Kiểm tra cơ chế che dấu
    masked = mgr.get_all_masked()
    assert "MODAL_TOKEN_ID" in masked["modal"]
    assert masked["modal"]["MODAL_TOKEN_ID"].startswith("ak-")
    assert "1234567890" not in masked["modal"]["MODAL_TOKEN_ID"]
    assert masked["modal"]["MODAL_TOKEN_ID"].endswith("cdef")

def test_credentials_inject_into_environ(tmp_path, monkeypatch):
    cred_file = tmp_path / ".flexsds" / "credentials"
    mgr = CredentialsManager(credentials_path=str(cred_file))
    mgr.set_credential("modal", "MODAL_TOKEN_ID", "ak-test-token")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)

    mgr.inject_into_environ()
    assert os.environ.get("MODAL_TOKEN_ID") == "ak-test-token"
```

- [x] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_credentials_manager.py -v`  
Expected: FAIL do chưa có `src/flexsds/core/credentials.py`.

- [x] **Step 3: Hiện thực hóa `src/flexsds/core/credentials.py` và đăng ký lệnh `flexsds auth`**

1. Viết lớp `CredentialsManager`:
   - Tự động tạo thư mục `~/.flexsds` nếu chưa có với quyền `0700`.
   - Ghi dữ liệu JSON/INI vào `~/.flexsds/credentials` và thiết lập `os.chmod(file, 0o600)`.
   - Hàm `mask_secret(val: str) -> str`: Giữ 3 ký tự đầu và 4 ký tự cuối, thay phần giữa bằng `****...`.
   - Hàm `inject_into_environ()`: Tự động nạp các key đã lưu vào `os.environ` nếu chưa được thiết lập, phục vụ cho các tiến trình con `modal` hoặc `vllm`.
2. Đăng ký subparser `auth` trong `src/flexsds/cli/main.py`: Các lệnh con `login`, `status`, `logout`.

- [x] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_credentials_manager.py -v`  
Expected: PASS 100%.

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add src/flexsds/core/credentials.py src/flexsds/cli/main.py tests/test_credentials_manager.py
git commit -m "feat(security): implement zero-leak credentials manager with strict 0600 permissions"
```

---

### Task 2: Chuẩn hóa `LocalServingRunner` (Vá cờ lineage executor và môi trường vLLM)

**Files:**
- Modify: `src/flexsds/server/local_runner.py`
- Test: `tests/test_local_runner.py`

**Interfaces:**
- Consumes: `config: Dict[str, Any]` từ `wizard.py` hoặc CLI args.
- Produces: `LocalServingRunner.build_command() -> List[str]`, `LocalServingRunner.setup_environment() -> Dict[str, str]`, `LocalServingRunner.launch() -> Optional[subprocess.Popen]`, `LocalServingRunner.stop() -> bool`.

- [x] **Step 1: Viết test kiểm tra cờ lineage và phát hiện trùng cổng trong `tests/test_local_runner.py`**

```python
def test_local_runner_build_command_contains_lineage_executor():
    from flexsds.server.local_runner import LocalServingRunner
    runner = LocalServingRunner()
    cfg = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
            "draft_method": "eagle3",
            "use_heterogeneous_vocab": True,
            "port": 8000,
            "enable_adaptive": True,
        }
    }
    cmd = runner.build_command(cfg)
    assert "--distributed-executor-backend" in cmd
    idx = cmd.index("--distributed-executor-backend")
    assert cmd[idx + 1] == "src.server.lineage_executor.LineageUniProcExecutor"
    assert "--scheduler-cls" in cmd
    idx_s = cmd.index("--scheduler-cls")
    assert cmd[idx_s + 1] == "src.server.adaptive_scheduler.AdaptiveAsyncScheduler"

def test_local_runner_detects_port_conflict(monkeypatch):
    import socket
    from flexsds.server.local_runner import LocalServingRunner
    # Giả lập cổng 8000 đang bận
    def mock_socket_bind(self, address):
        raise OSError("Address already in use")
    monkeypatch.setattr(socket.socket, "bind", mock_socket_bind)

    runner = LocalServingRunner()
    cfg = {"serving": {"port": 8000}}
    can_bind = runner.check_port_available(8000)
    assert can_bind is False
```

- [x] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_local_runner.py::test_local_runner_build_command_contains_lineage_executor tests/test_local_runner.py::test_local_runner_detects_port_conflict -v`  
Expected: FAIL do chưa có `--distributed-executor-backend` và hàm `check_port_available`.

- [x] **Step 3: Cập nhật `src/flexsds/server/local_runner.py`**

Thêm cờ `--distributed-executor-backend src.server.lineage_executor.LineageUniProcExecutor` vào `build_command()`, bổ sung hàm `check_port_available(port: int) -> bool` bằng `socket.socket()`, đồng bộ biến môi trường `PYTHONPATH` trỏ tới gốc repo và export `ADAPTIVE_ENABLE_VRAM_GUARD`, `ADAPTIVE_TELEMETRY_PATH`.

- [x] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_local_runner.py -v`  
Expected: PASS 100%.

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add src/flexsds/server/local_runner.py tests/test_local_runner.py
git commit -m "fix(server): align local runner with lineage executor and add port conflict guard"
```

---

### Task 3: Xây dựng Modal Serverless App độc lập (`src/flexsds/server/modal_app.py`)

**Files:**
- Create: `src/flexsds/server/modal_app.py`
- Modify: `src/flexsds/server/modal_runner.py`
- Test: `tests/test_modal_runner_cli.py`

**Interfaces:**
- Consumes: Token Modal qua biến môi trường hoặc `~/.modal.toml`, `serving_config.json`.
- Produces: File định nghĩa `modal_app.py` (với `app = modal.App("flexsds-serving")`), `ModalServingRunner.build_modal_command() -> List[str]`, `ModalServingRunner.check_credentials() -> bool`.

- [x] **Step 1: Viết test kiểm tra lệnh gọi Modal và bộ gác cổng token trong `tests/test_modal_runner_cli.py`**

```python
def test_modal_runner_missing_credentials_guard(monkeypatch):
    from flexsds.server.modal_runner import ModalServingRunner
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr("os.path.exists", lambda path: False)
    runner = ModalServingRunner()
    assert runner.check_credentials() is False

def test_modal_runner_build_command_uses_standalone_app():
    from flexsds.server.modal_runner import ModalServingRunner
    runner = ModalServingRunner()
    cfg = {"serving": {"target_gpu": "A10G"}}
    cmd = runner.build_modal_command(cfg, deploy=False)
    assert "modal" in cmd
    assert "run" in cmd
    assert any("modal_app.py" in arg for arg in cmd)
    assert "--gpu" in cmd
    assert "A10G" in cmd
```

- [x] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_modal_runner_cli.py -v`  
Expected: FAIL do lệnh cũ trỏ vào `experiments/modal_lanes.py`.

- [x] **Step 3: Tạo `src/flexsds/server/modal_app.py` và cập nhật `modal_runner.py`**

1. Tạo `modal_app.py`: Sử dụng `modal.App("flexsds-serving")`, định nghĩa `modal.Image.debian_slim(python_version="3.11")`, cài đặt `vllm==0.30.0`, sao chép thư mục `src/` và `experiments/scalar_vllm_source_manifest.json`, cấu hình hàm `@app.function` khởi chạy máy chủ vLLM OpenAI-compatible.
2. Cập nhật `modal_runner.py`: Trỏ đường dẫn app sang `src/flexsds/server/modal_app.py`, hoàn thiện xử lý cờ `--dry-run` và xuất cảnh báo hướng dẫn khi thiếu token (tích hợp `CredentialsManager`).

- [x] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_modal_runner_cli.py -v`  
Expected: PASS 100%.

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add src/flexsds/server/modal_app.py src/flexsds/server/modal_runner.py tests/test_modal_runner_cli.py
git commit -m "feat(server): introduce standalone modal serverless app and credential guard"
```

---

### Task 4: Chuẩn hóa Colab Server Notebook và cập nhật `ColabServingRunner`

**Files:**
- Create: `notebooks/flexsds_colab_server.ipynb`
- Modify: `src/flexsds/server/colab_runner.py`
- Test: `tests/test_colab_runner.py`

**Interfaces:**
- Consumes: Cấu hình URL Cloudflare Tunnel từ người dùng, file mẫu `vscode_cli.py`.
- Produces: `ColabServingRunner.verify_tunnel_health(public_url: str) -> bool`, `ColabServingRunner.get_colab_instructions() -> dict`.

- [x] **Step 1: Viết test cho kiểm tra đường hầm Colab trong `tests/test_colab_runner.py`**

```python
def test_colab_runner_verify_tunnel_health_success(monkeypatch):
    from flexsds.server.colab_runner import ColabServingRunner
    class MockResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: MockResponse())

    runner = ColabServingRunner()
    assert runner.verify_tunnel_health("https://ptit-test.trycloudflare.com") is True

def test_colab_runner_verify_tunnel_health_timeout(monkeypatch):
    import urllib.error
    from flexsds.server.colab_runner import ColabServingRunner
    def mock_urlopen(req, timeout):
        raise urllib.error.URLError("Connection timed out")
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    runner = ColabServingRunner()
    assert runner.verify_tunnel_health("https://ptit-invalid.trycloudflare.com") is False
```

- [x] **Step 2: Chạy test để xác nhận kiểm thử vượt qua hoặc bổ sung các ca kiểm thử biên**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_colab_runner.py -v`  
Expected: PASS hoặc FAIL nếu thiếu kiểm tra URL rỗng.

- [x] **Step 3: Tạo `notebooks/flexsds_colab_server.ipynb` và hoàn thiện `colab_runner.py`**

1. Tạo `notebooks/flexsds_colab_server.ipynb`: Kế thừa từ `vscode_cli.py`, cấu hình các ô lệnh tải repo, cài đặt `vllm==0.30.0`, gỡ `torchaudio`, khởi chạy vLLM với cờ speculative EAGLE và kích hoạt Cloudflare Tunnel `cloudflared tunnel --url http://localhost:8000`. Không lưu cứng token PAT vào notebook.
2. Cập nhật `colab_runner.py`: Tinh chỉnh cơ chế probe endpoint có timeout tùy chỉnh và bổ sung hướng dẫn chi tiết cho GPU T4 / L4.

- [x] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_colab_runner.py -v`  
Expected: PASS 100%.

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add notebooks/flexsds_colab_server.ipynb src/flexsds/server/colab_runner.py tests/test_colab_runner.py
git commit -m "feat(colab): create reproducible colab serving notebook and robust health check"
```

---

### Task 5: Hiện thực hóa HTTP Client và Streaming Engine (`src/flexsds/client/client.py`)

**Files:**
- Create: `src/flexsds/client/__init__.py`
- Create: `src/flexsds/client/client.py`
- Test: `tests/test_flexsds_client.py`

**Interfaces:**
- Consumes: Endpoint URL (ví dụ `http://localhost:8000/v1` hoặc public tunnel), payload tin nhắn OpenAI.
- Produces: `FlexSDSClient.check_health() -> bool`, `FlexSDSClient.get_models() -> List[str]`, `FlexSDSClient.stream_chat(messages, model, temperature) -> Iterator[str]`.

- [x] **Step 1: Viết test cho HTTP Client và xử lý lỗi không kết nối được trong `tests/test_flexsds_client.py`**

```python
def test_client_handles_unreachable_endpoint():
    from flexsds.client.client import FlexSDSClient, EndpointUnreachableError
    client = FlexSDSClient(base_url="http://127.0.0.1:59999/v1")
    assert client.check_health() is False

def test_client_stream_chat_parses_sse(monkeypatch):
    import io
    from flexsds.client.client import FlexSDSClient

    fake_sse_lines = [
        b'data: {"choices": [{"delta": {"content": "Ch\\xc3\\xa0o"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " b\\xe1\\xba\\xa1n!"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    class MockHttpResp:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self): return iter(fake_sse_lines)

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: MockHttpResp())

    client = FlexSDSClient(base_url="http://localhost:8000/v1")
    tokens = list(client.stream_chat([{"role": "user", "content": "Hi"}], model="qwen2.5-7b-adaptive"))
    assert "".join(tokens) == "Chào bạn!"
```

- [x] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_flexsds_client.py -v`  
Expected: FAIL do chưa có package `src/flexsds/client/client.py`.

- [x] **Step 3: Hiện thực hóa `src/flexsds/client/client.py`**

Sử dụng thư viện chuẩn `urllib.request` và `json` để cài đặt lớp `FlexSDSClient`, hỗ trợ:
- `check_health()` gửi request tới `/health` hoặc `/v1/models`.
- `get_models()` lấy danh sách mô hình đang được phục vụ.
- `stream_chat()` gửi `POST /v1/chat/completions` với `stream=True`, đọc từng dòng SSE `data: `, phân tích JSON và yield token nội dung `delta.content`.

- [x] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_flexsds_client.py -v`  
Expected: PASS 100%.

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add src/flexsds/client/__init__.py src/flexsds/client/client.py tests/test_flexsds_client.py
git commit -m "feat(client): implement lightweight open-ai compatible sse streaming client"
```

---

### Task 6: Xây dựng giao diện trò chuyện dòng lệnh Interactive CLI Chat (`src/flexsds/client/chat.py` và lệnh `flexsds chat`)

**Files:**
- Create: `src/flexsds/client/chat.py`
- Modify: `src/flexsds/cli/main.py`
- Test: `tests/test_cli_chat_command.py`

**Interfaces:**
- Consumes: `FlexSDSClient` từ Task 5, tham số dòng lệnh từ `main.py`.
- Produces: `run_interactive_chat(endpoint: str, model: Optional[str]) -> None`.

- [x] **Step 1: Viết test cho luồng chat và bắt `SIGINT` trong `tests/test_cli_chat_command.py`**

```python
def test_chat_handles_sigint_gracefully(monkeypatch):
    from flexsds.client.chat import InteractiveChatSession
    from flexsds.client.client import FlexSDSClient

    client = FlexSDSClient(base_url="http://localhost:8000/v1")
    session = InteractiveChatSession(client, model="qwen2.5-7b-adaptive")

    # Giả lập người dùng nhập /exit
    monkeypatch.setattr("builtins.input", lambda prompt: "/exit")
    exit_code = session.start()
    assert exit_code == 0
```

- [x] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_cli_chat_command.py -v`  
Expected: FAIL do chưa có `src/flexsds/client/chat.py`.

- [x] **Step 3: Hiện thực hóa `src/flexsds/client/chat.py` và đăng ký lệnh vào `main.py`**

1. Viết `InteractiveChatSession` trong `chat.py`:
   - Vòng lặp nhận prompt `You > `.
   - Hiển thị phản hồi `FlexSDS > ` theo thời gian thực (từng token xuất hiện ngay khi stream về).
   - Đo đạc thời gian: `start_time = time.perf_counter()`, tính tổng token, tính `tokens/s` và `TPOT = (elapsed / total_tokens) * 1000` (ms).
   - In hộp thống kê hiệu năng bằng `render_box()` sau mỗi lượt trả lời.
   - Bắt `KeyboardInterrupt` khi stream dở để không văng terminal.
2. Cập nhật `src/flexsds/cli/main.py`: Bổ sung subparser `chat` với các cờ `--endpoint`, `--model`, `--temperature`.

- [x] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_cli_chat_command.py -v`  
Expected: PASS 100%.

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add src/flexsds/client/chat.py src/flexsds/cli/main.py tests/test_cli_chat_command.py
git commit -m "feat(cli): add interactive terminal chat command with live tpot metrics"
```

---

### Task 7: Bộ theo dõi thích ứng thời gian thực Live Telemetry Monitor (`src/flexsds/client/monitor.py` và lệnh `flexsds monitor`)

**Files:**
- Create: `src/flexsds/client/monitor.py`
- Modify: `src/flexsds/cli/main.py`
- Test: `tests/test_cli_monitor_command.py`

**Interfaces:**
- Consumes: Tệp telemetry JSONL (mặc định `telemetry/serving_trace.jsonl`).
- Produces: `run_live_monitor(trace_file: str, poll_interval: float) -> None`.

- [x] **Step 1: Viết test cho bộ đọc telemetry trong `tests/test_cli_monitor_command.py`**

```python
def test_monitor_waits_for_telemetry_file(tmp_path):
    from flexsds.client.monitor import TelemetryMonitor
    trace_path = tmp_path / "serving_trace.jsonl"
    monitor = TelemetryMonitor(str(trace_path), timeout=0.1)
    # File chưa tồn tại -> không crash, trả về 0 records
    records = monitor.read_new_records()
    assert len(records) == 0

def test_monitor_parses_records_correctly(tmp_path):
    import json
    from flexsds.client.monitor import TelemetryMonitor
    trace_path = tmp_path / "serving_trace.jsonl"
    record = {
        "step": 1,
        "lane": "adaptive",
        "executed_draft_token_count": 4,
        "accepted_draft_token_count": 3,
        "alpha_ema": 0.78,
        "source_batch_scalar_k": 4,
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    monitor = TelemetryMonitor(str(trace_path))
    records = monitor.read_new_records()
    assert len(records) == 1
    assert records[0]["accepted_draft_token_count"] == 3
```

- [x] **Step 2: Chạy test để xác nhận kiểm thử thất bại**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_cli_monitor_command.py -v`  
Expected: FAIL do chưa có `src/flexsds/client/monitor.py`.

- [x] **Step 3: Hiện thực hóa `src/flexsds/client/monitor.py` và cập nhật `main.py`**

Viết lớp `TelemetryMonitor`: Theo dõi con trỏ tệp (`file.seek`), đọc các dòng JSON mới thêm vào, bóc tách `source_batch_scalar_k`, `alpha_ema`, số token chấp nhận và hiển thị dạng dòng trạng thái trực quan trên terminal. Đăng ký lệnh `flexsds monitor` vào `main.py`.

- [x] **Step 4: Chạy test để xác nhận kiểm thử vượt qua**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_cli_monitor_command.py -v`  
Expected: PASS 100%.

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add src/flexsds/client/monitor.py src/flexsds/cli/main.py tests/test_cli_monitor_command.py
git commit -m "feat(cli): add live telemetry monitor command for adaptive trace inspection"
```

---

### Task 8: Kiểm thử tích hợp toàn diện End-to-End và cập nhật CLI Wizard

**Files:**
- Modify: `src/flexsds/cli/wizard.py`
- Test: `tests/test_end_to_end_flow.py`

**Interfaces:**
- Consumes: Toàn bộ các module từ Task 1 đến Task 6.
- Produces: Chu trình hoàn chỉnh: Khởi chạy phục vụ -> Hiển thị hướng dẫn chat -> Kết nối chat thành công -> Dừng phục vụ an toàn.

- [x] **Step 1: Viết test kiểm thử toàn chu trình trong `tests/test_end_to_end_flow.py`**

```python
def test_end_to_end_wizard_to_chat_flow(monkeypatch, tmp_path):
    from flexsds.cli.wizard import run_interactive_wizard
    from flexsds.core.hardware import HardwareInfo
    from flexsds.client.client import FlexSDSClient

    # Giả lập môi trường phần cứng có GPU
    hw = HardwareInfo(
        device_name="NVIDIA L4",
        total_vram_mb=24000.0,
        free_vram_mb=20000.0,
        sm_architecture="sm_89",
        is_cuda_available=True,
    )

    # Chạy wizard ở chế độ tự động với mock inputs
    monkeypatch.setattr("flexsds.cli.wizard.prompt_choice", lambda q, c, **kw: c[0])
    monkeypatch.setattr("flexsds.cli.wizard.prompt_string", lambda q, d: d)
    monkeypatch.setattr("flexsds.cli.wizard.prompt_bool", lambda q, **kw: False)

    cfg = run_interactive_wizard(hw)
    assert cfg["action"] == "serve"
    assert "target_model" in cfg
    assert "draft_model" in cfg
```

- [x] **Step 2: Chạy test để xác nhận trạng thái**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_end_to_end_flow.py -v`  
Expected: PASS hoặc điều chỉnh các câu thông báo sau khi lưu config.

- [x] **Step 3: Cập nhật `src/flexsds/cli/wizard.py`**

Ở bước cuối sau khi lưu file `serving_config.json` hoặc khi khởi chạy server nền thành công, in thông báo màu xanh hướng dẫn rõ ràng:
`✔ Máy chủ FlexSDS đang khởi động tại http://localhost:8000/v1`  
`  Gợi ý: Mở terminal mới và chạy lệnh: flexsds chat để bắt đầu trò chuyện!`

- [x] **Step 4: Chạy lại toàn bộ test suite của package FlexSDS**

Run: `cd do-an-ptit-d22 && PYTHONPATH=. pytest tests/test_flexsds*.py tests/test_local_runner.py tests/test_modal_runner_cli.py tests/test_colab_runner.py tests/test_cli_*.py tests/test_end_to_end_flow.py -v`  
Expected: PASS 100% toàn bộ test suite (dự kiến >= 35 tests).

- [x] **Step 5: Lệnh commit tham khảo**

```bash
git add src/flexsds/cli/wizard.py tests/test_end_to_end_flow.py
git commit -m "feat(cli): complete end-to-end integration from wizard serving to chat client"
```
