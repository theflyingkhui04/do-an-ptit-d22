# Đặc tả kỹ thuật phân hệ backend phục vụ và giao tiếp suy luận cho FlexSDS CLI

> **Tài liệu:** `spec.md`  
> **Mã định danh task:** `M5-02-FLEX-BACKEND`  
> **Dự án:** Xây dựng phân hệ phục vụ đa nền tảng và giao tiếp suy luận (CLI Chat Client) cho FlexSDS  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead 70%), Bùi Đức Đại (Reviewer 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/flexsds-cli-backend/spec.md`  
> **Ngày lập:** 12/11/2026  
> **Trạng thái:** Bản phê duyệt triển khai (Phương án B: Serving Backend + CLI Chat Client)  

---

## 1. Mục đích và bối cảnh kỹ thuật (Intent & Why)

Sau khi hoàn thiện giao diện tương tác dòng lệnh (Interactive Wizard) ở mốc M5-01, công cụ FlexSDS CLI đã có khả năng hướng dẫn người dùng cấu hình hệ thống, kiểm tra phần cứng và lưu tệp `serving_config.json`. Tuy nhiên, để công cụ thực sự trở thành một sản phẩm hoàn chỉnh phục vụ bảo vệ đồ án và cho phép người dùng trải nghiệm thực tế, phân hệ backend cần được thiết kế và triển khai trọn vẹn:

1. **Khép kín chu trình trải nghiệm (Serving to Chat Loop):**
   - Hiện tại, lệnh `flexsds serve` chỉ dừng ở việc chuẩn bị tham số và bật tiến trình nền hoặc xuất file cấu hình. Người dùng chưa có công cụ để trực tiếp giao tiếp với mô hình.
   - Nhóm quyết định lựa chọn **Phương án B: Serving Backend kết hợp CLI Chat Client**. Sau khi máy chủ khởi động thành công trên bất kỳ nền tảng nào (Local GPU, Google Colab hay Modal Cloud), người dùng có thể dùng lệnh `flexsds chat` để trò chuyện trực tiếp trên terminal với cơ chế streaming token và hiển thị chỉ số tốc độ thời gian thực (tokens/s, TPOT).
2. **Đồng bộ hóa với thành tựu của mốc M4-02-04 (Async Lineage v0.2):**
   - Đợt merge mới nhất trên nhánh `develop` đã hoàn tất cơ chế kiểm soát nguồn gốc bất đồng bộ (`LineageUniProcExecutor` và `LineageStartup`).
   - Phân hệ [`local_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/local_runner.py) hiện tại đang bị thiếu cờ `--distributed-executor-backend` và các biến môi trường lineage, dẫn đến việc nếu chạy thực tế vLLM sẽ crash ngay lập tức. Phân hệ backend cần vá dứt điểm sự thiếu sót này.
3. **Hiện thực hóa 3 nền tảng phục vụ thực tế:**
   - **Local GPU:** Quản lý vòng đời tiến trình vLLM cục bộ theo nhóm tiến trình (`os.setsid`), triệt tiêu hoàn toàn tiến trình mồ côi (zero-zombie process).
   - **Modal Cloud BYOC:** Xây dựng ứng dụng Modal chuyên dụng (`modal_app.py`) nhận tham số từ CLI thay vì gọi vào script đo đạc benchmark nội bộ `modal_lanes.py`.
   - **Google Colab:** Tích hợp kinh nghiệm từ [`vscode_cli.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/vscode_cli.py) và [`vscode_cli.ipynb`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/vscode_cli.ipynb) (đã có cơ chế mount Google Drive, clone repo tự động và mở đường hầm Cloudflare Tunnel / VS Code CLI Tunnel).

---

## 2. Mục tiêu, phi mục tiêu và phản mục tiêu (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục tiêu (Goals)
1. **Chuẩn hóa bộ khởi chạy máy chủ cục bộ (`LocalServingRunner`):**
   - Cập nhật hàm `build_command()` bổ sung đầy đủ các cờ cần thiết cho vLLM v0.30.0:
     + `--scheduler-cls src.server.adaptive_scheduler.AdaptiveAsyncScheduler`
     + `--distributed-executor-backend src.server.lineage_executor.LineageUniProcExecutor`
     + `--per-request-spec-decode-metrics summary`
   - Thiết lập chuẩn xác các biến môi trường: `PYTHONPATH`, `ADAPTIVE_ENABLE_VRAM_GUARD`, `ADAPTIVE_TELEMETRY_PATH`, `ADAPTIVE_MAX_K`, `ADAPTIVE_PER_REQUEST`.
   - Quản lý PID qua `.flexsds.pid` và dọn dẹp sạch sẽ khi nhận `SIGINT` / lệnh `flexsds stop`.
2. **Xây dựng ứng dụng Modal độc lập cho FlexSDS (`src/flexsds/server/modal_app.py`):**
   - Định nghĩa `modal.App` độc lập mang tên `flexsds-serving`.
   - Mount đúng các thư mục mã nguồn cần thiết (`src/`, `experiments/scalar_vllm_source_manifest.json`) và ánh xạ cổng phục vụ ra internet qua Web Endpoint của Modal.
   - Hỗ trợ cả 2 chế độ: Ephemeral run (tự tắt khi dừng lệnh) và Persistent deploy (phục vụ liên tục).
3. **Chuẩn hóa phân hệ Google Colab Server:**
   - Đóng gói file notebook chuẩn `notebooks/flexsds_colab_server.ipynb` dựa trên mẫu đã chạy thực tế tại [`vscode_cli.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/vscode_cli.py).
   - Tự động hóa việc cài đặt `requirements-server.txt`, khởi chạy máy chủ vLLM và kích hoạt Cloudflare Tunnel phát ra URL public `.trycloudflare.com`.
   - [`colab_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/colab_runner.py) kiểm tra tình trạng kết nối (health check) của đường hầm trước khi cho phép client kết nối.
4. **Hiện thực hóa phân hệ giao tiếp dòng lệnh (`flexsds chat`):**
   - Xây dựng module client [`src/flexsds/client/chat.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/client/) kết nối tới endpoint tương thích chuẩn OpenAI (`/v1/chat/completions`).
   - Hỗ trợ streaming phản hồi theo thời gian thực (Server-Sent Events - SSE).
   - Hiển thị thanh trạng thái thống kê sau mỗi lượt trả lời: Số lượng token sinh ra, tổng thời gian, tốc độ trung bình (tokens/giây) và độ trễ trên từng token (Time Per Output Token - TPOT).
5. **Bộ theo dõi trạng thái thích ứng (Live Telemetry Monitor):**
   - Cung cấp module đọc tệp `telemetry/serving_trace.jsonl` để hiển thị trạng thái thích ứng đang diễn ra: Ngân sách $K$ hiện tại, tỷ lệ chấp nhận hiệu dụng $\alpha_{effective}$, và số token draft được chấp nhận.
6. **Hệ thống quản lý thông tin xác thực bảo mật (Secure Credentials Manager):**
   - Xây dựng module [`src/flexsds/core/credentials.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/credentials.py) để lưu trữ an toàn các token/API key nhạy cảm (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`, `CLOUDFLARE_TUNNEL_TOKEN`).
   - Tuyệt đối không lưu token vào tệp cấu hình chia sẻ `serving_config.json` hay bất kỳ tệp nào trong kho lưu trữ Git.
   - Vị trí lưu trữ độc quyền tại thư mục người dùng cá nhân: `~/.flexsds/credentials` với quyền truy cập nghiêm ngặt `0600` (chỉ tài khoản người dùng hiện tại có quyền đọc/ghi).
   - Cung cấp nhóm lệnh `flexsds auth login`, `flexsds auth status` (che giấu ký tự dạng `mod-****...cdef`), và `flexsds auth logout` (xóa sạch dữ liệu bí mật).
   - Khi cần khởi chạy ứng dụng (Modal, Colab, Hugging Face), hệ thống nạp trực tiếp vào bộ nhớ để chuyển giao cho tiến trình con mà không để lộ ra ngoài.
7. **Kiểm thử tự động toàn diện:**
   - Đảm bảo 100% test chạy được trên môi trường CPU CI/CD qua mock subprocess và mock HTTP stream. Nâng tổng số test của riêng phân hệ FlexSDS lên trên 40 tests.

### 2.2. Phi mục tiêu (Non-Goals)
- Không can thiệp hoặc viết lại thuật toán lõi của `LineageUniProcExecutor` và `AdaptiveAsyncScheduler` đã được Đại hoàn thiện và đóng gói ở M4-02-04.
- Không xây dựng giao diện web UI trong task này (được phân bổ riêng cho Sprint 6 / M5-03).
- Không tự huấn luyện lại trọng số mô hình hoặc tải thêm mô hình mới ngoài danh mục đã phê duyệt.

### 2.3. Phản mục tiêu (Anti-Goals)
- Tuyệt đối không để xảy ra tiến trình vLLM chạy ngầm chiếm dụng bộ nhớ GPU sau khi người dùng dừng lệnh (zero-zombie guarantee).
- Tuyệt đối không thêm các thư viện bên ngoài quá nặng nề cho phần client (chỉ sử dụng thư viện chuẩn của Python hoặc thư viện HTTP nhẹ, không cài thêm các SDK cồng kềnh).
- Tuyệt đối không nhúng cứng (hardcode) GitHub PAT token hay Modal secret cá nhân vào mã nguồn repo (rút kinh nghiệm từ file mẫu `vscode_cli.py`).

---

## 3. Ràng buộc kỹ thuật (Constraints)

1. **Môi trường thực thi:** Python 3.10+, PyTorch 2.4+ (nếu chạy local GPU), CUDA 12.1 đến 12.8.
2. **Serving runtime:** vLLM v0.30.0 cố định.
3. **Mô hình phục vụ cốt lõi:**
   - Target model: `Qwen/Qwen2.5-7B-Instruct-AWQ`
   - Draft head: `duckhoi/qwen2.5-7b-eagle3.1-speculator`
   - Phương pháp giải mã đầu cơ: EAGLE (`method = "eagle3"`, `use_heterogeneous_vocab = True`).
4. **Tiêu chuẩn giao tiếp API:** Chuẩn OpenAI RESTful API (`POST /v1/chat/completions`, `GET /health`, `GET /v1/models`).
5. **Quy tắc chính tả học thuật:** Tuân thủ nghiêm ngặt quy tắc viết hoa tiếng Việt (không dùng Title Case tiếng Anh; chỉ viết hoa đầu câu, danh từ riêng và từ viết tắt).
6. **Bảo mật mã nguồn:** Mọi token xác thực (Hugging Face, Modal, GitHub) phải được truyền qua biến môi trường hoặc nhập trực tiếp từ prompt bảo mật, không lưu vào tệp văn bản theo dõi của Git.

---

## 4. Thiết kế kiến trúc và phân rã module

Hệ thống phân hệ backend của FlexSDS CLI được tổ chức theo sơ đồ kiến trúc sau:

```
                          ┌───────────────────────────────┐
                          │         flexsds CLI           │
                          │ (main.py, wizard.py, banner)  │
                          └───────────────┬───────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│     src/flexsds/server/       │                       │     src/flexsds/client/       │
├───────────────────────────────┤                       ├───────────────────────────────┤
│ • local_runner.py (Subprocess)│                       │ • client.py (HTTP / SSE API)  │
│ • modal_app.py (Serverless)   │                       │ • chat.py (Interactive TUI)   │
│ • modal_runner.py (BYOC)      │                       │ • monitor.py (Telemetry tail) │
│ • colab_runner.py (Tunnel)    │                       └───────────────┬───────────────┘
└───────────────┬───────────────┘                                       │
                │ Khởi chạy và quản lý vòng đời                         │ Giao tiếp suy luận HTTP
                ▼                                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           vLLM Serving Engine (Port 8000)                             │
│ • OpenAI-compatible API Server (/v1/chat/completions, /health)                        │
│ • EngineCore: AdaptiveAsyncScheduler + LineageUniProcExecutor                          │
│ • Target: Qwen2.5-7B-AWQ | Speculator: EAGLE-3.1 Head (k: 0..4 động)                  │
│ • Telemetry Sink: telemetry/serving_trace.jsonl                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1. Chi tiết phân hệ khởi chạy (`src/flexsds/server/`)

#### 1. Module `local_runner.py`:
- **Nhiệm vụ:** Tạo câu lệnh thực thi `vllm serve` và quản lý tiến trình nền trên máy cục bộ.
- **Cập nhật cốt lõi:**
  ```python
  cmd.extend([
      "--speculative-config", json.dumps(speculative_config),
      "--scheduler-cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
      "--distributed-executor-backend", "src.server.lineage_executor.LineageUniProcExecutor",
      "--per-request-spec-decode-metrics", "summary",
  ])
  ```
- **Quản lý tiến trình (Zero-Zombie):**
  - Khởi chạy bằng `subprocess.Popen(..., preexec_fn=os.setsid)`.
  - Bắt tín hiệu ngắt và gửi `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)` để triệt tiêu toàn bộ tiến trình con vLLM.
  - Lưu PID vào tệp `.flexsds.pid` để hỗ trợ lệnh `flexsds stop`.

#### 2. Module `modal_app.py` và `modal_runner.py`:
- **Nhiệm vụ:** Cung cấp ứng dụng serverless chuyên dụng chạy trên Modal Cloud.
- **Kiến trúc Modal App:**
  - Định nghĩa môi trường qua `modal.Image.debian_slim(python_version="3.11")`:
    + Cài đặt `vllm==0.30.0`, `torch==2.13.0+cu130`.
    + Sao chép mã nguồn nhóm (`src/`, `experiments/scalar_vllm_source_manifest.json`).
  - Gắn volume lưu trữ trọng số mô hình (`ptit-model-cache`).
  - Sử dụng `@app.function(gpu="L4", timeout=3600)` để khởi chạy vLLM và mở cổng `@modal.asgi_app()` hoặc web endpoint cho phép client từ xa kết nối.
- **Tích hợp `modal_runner.py`:** Đọc cấu hình `serving_config.json`, tự động kiểm tra token định danh Modal qua `check_credentials()` và kích hoạt `modal run` hoặc `modal deploy`.

#### 3. Module `colab_runner.py` và notebook phục vụ:
- **Nhiệm vụ:** Hỗ trợ khởi chạy máy chủ trên Google Colab miễn phí.
- **Kế thừa từ `vscode_cli.py`:**
  - Cung cấp notebook hoàn chỉnh `notebooks/flexsds_colab_server.ipynb`: Tự động mount Google Drive, tải mã nguồn repo, cài đặt dependencies và bật vLLM.
  - Tích hợp Cloudflare Tunnel tự động sinh URL public dạng `https://xxx.trycloudflare.com`.
  - Bổ sung tùy chọn kích hoạt VS Code CLI Tunnel nếu người dùng muốn can thiệp debug trực tiếp trên máy chủ Colab.
- **Hàm `verify_tunnel_health(url)`:** Gửi request thăm dò `/health` định kỳ (tối đa 30 giây) để báo hiệu cho người dùng ngay khi mô hình đã nạp xong vào VRAM và sẵn sàng phục vụ.

---

### 4.2. Chi tiết phân hệ giao tiếp dòng lệnh (`src/flexsds/client/`)

#### 1. Module `client.py` (HTTP Client):
- **Nhiệm vụ:** Đóng gói các hàm gọi API chuẩn OpenAI tới backend.
- **Phương thức chính:**
  - `check_health(base_url: str) -> bool`: Kiểm tra tình trạng sẵn sàng của máy chủ.
  - `get_model_info(base_url: str) -> dict`: Lấy thông tin mô hình đang được phục vụ.
  - `stream_chat(base_url: str, messages: list[dict], model: str, temperature: float = 0.0) -> Iterator[str]`: Gửi yêu cầu giải mã và nhận về từng token văn bản theo định dạng SSE (`data: {"choices": [{"delta": {"content": "..."}}]}`).

#### 2. Module `chat.py` (Interactive Terminal Chat):
- **Nhiệm vụ:** Cung cấp giao diện trò chuyện dòng lệnh trực quan và mượt mà.
- **Trải nghiệm người dùng (UX):**
  - Hiển thị khung giới thiệu phiên chat (tên mô hình, endpoint đang kết nối, cơ chế đầu cơ đang áp dụng).
  - Vòng lặp nhận prompt từ người dùng (`You > `).
  - Xuất phản hồi của trợ lý ảo (`FlexSDS > `) theo thời gian thực (từng token xuất hiện ngay khi sinh ra).
  - **Khung thống kê hiệu năng (Performance Box) sau mỗi câu trả lời:**
    ```text
    ┌─── Thống kê suy luận ──────────────────────────┐
    │ Số token sinh ra : 142 tokens                   │
    │ Thời gian xử lý  : 1.85 giây                    │
    │ Tốc độ trung bình: 76.8 tokens/s                │
    │ Độ trễ mỗi token : 13.02 ms (TPOT)              │
    └─────────────────────────────────────────────────┘
    ```
  - Hỗ trợ các lệnh điều khiển nhanh trong phiên chat: `/exit` (thoát chat), `/clear` (xóa lịch sử ngữ cảnh), `/stats` (xem thống kê tổng phiên).

#### 3. Module `monitor.py` (Live Telemetry Monitor):
- **Nhiệm vụ:** Theo dõi trực tiếp tệp `telemetry/serving_trace.jsonl` được sinh ra từ `AdaptiveAsyncScheduler`.
- **Thông số hiển thị:**
  - Bước suy luận (step count).
  - Ngân sách $K$ được chọn trong bước.
  - Số lượng token đề xuất và số lượng token được chấp nhận thực tế.
  - Tỷ lệ chấp nhận trung bình trượt $\alpha_{EMA}$.

---

## 5. Cập nhật các lệnh dòng lệnh (CLI Commands Specification)

Phân hệ backend sẽ bổ sung và hoàn thiện các lệnh chính trong [`src/flexsds/cli/main.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/cli/main.py):

### 5.1. Lệnh `flexsds serve` (Khởi chạy máy chủ)
- **Cú pháp:**
  ```bash
  flexsds serve [--target TARGET] [--draft DRAFT] [--backend {local,modal,colab}]
                [--port PORT] [--background] [--deploy] [--no-adaptive] [--dry-run]
  ```
- **Hành vi:**
  - Nếu không truyền cờ: Kích hoạt Interactive Wizard như hiện tại.
  - Khi hoàn tất bước cuối: Tự động khởi chạy backend tương ứng và hiển thị gợi ý lệnh:
    `✔ Server đã sẵn sàng. Bạn có thể mở terminal khác và chạy: flexsds chat`

### 5.2. Lệnh `flexsds chat` (Trò chuyện trực tiếp)
- **Cú pháp:**
  ```bash
  flexsds chat [--endpoint ENDPOINT] [--model MODEL] [--temperature TEMP]
  ```
- **Hành vi:**
  - Mặc định kết nối tới `http://localhost:8000/v1` (hoặc đọc URL từ `serving_config.json` nếu có).
  - Kiểm tra kết nối trước khi vào giao diện chat. Nếu server chưa bật, thông báo lỗi rõ ràng và hướng dẫn chạy `flexsds serve`.
  - Mở giao diện chat tương tác có đo lường tốc độ.

### 5.3. Lệnh `flexsds monitor` (Theo dõi hoạt động thích ứng)
- **Cú pháp:**
  ```bash
  flexsds monitor [--trace-file PATH]
  ```
- **Hành vi:**
  - Đọc đuôi tệp telemetry và hiển thị biểu đồ hoặc bảng cập nhật liên tục các quyết định thay đổi ngân sách $K$ của thuật toán giải mã đầu cơ.

### 5.4. Lệnh `flexsds stop` (Dừng máy chủ)
- **Cú pháp:**
  ```bash
  flexsds stop
  ```
- **Hành vi:**
  - Đọc PID từ `.flexsds.pid`, gửi tín hiệu dừng tới toàn bộ nhóm tiến trình và xóa tệp PID.

### 5.5. Nhóm lệnh `flexsds auth` (Quản lý khóa định danh bảo mật)
- **Cú pháp:**
  ```bash
  flexsds auth login [--provider {modal,huggingface,cloudflare}]
  flexsds auth status
  flexsds auth logout [--provider {modal,huggingface,cloudflare,all}]
  ```
- **Hành vi:**
  - `login`: Yêu cầu nhập token bằng chế độ ẩn ký tự (`getpass`), lưu trữ an toàn vào `~/.flexsds/credentials` với quyền `chmod 0600`.
  - `status`: Kiểm tra tình trạng đăng nhập của từng dịch vụ, hiển thị dưới dạng che giấu (masking) như `mod-key-****...4a1b`, tuyệt đối không in chuỗi token thô.
  - `logout`: Xóa bỏ khóa định danh đã lưu khỏi hệ thống máy tính một cách an toàn.

---

## 6. Tiêu chí nghiệm thu kỹ thuật (Definition of Done - DoD)

Mốc công việc `M5-02-FLEX-BACKEND` được xem là hoàn thành khi đạt đủ các tiêu chí sau:

1. **Khởi chạy Local chính xác (Local Serving Alignment):**
   - Lệnh `LocalServingRunner.build_command()` sinh ra đầy đủ cờ `--distributed-executor-backend src.server.lineage_executor.LineageUniProcExecutor` và cấu hình speculative EAGLE.
   - Quản lý tiến trình đạt chuẩn Zero-Zombie, dừng sạch sẽ khi ấn `Ctrl+C` hoặc gọi `flexsds stop`.
2. **Modal Cloud Runner sẵn sàng:**
   - Ứng dụng `modal_app.py` được định nghĩa độc lập, có thể deploy hoặc run bằng lệnh của Modal SDK mà không phụ thuộc vào các script benchmark nội bộ M3.
3. **Colab Notebook hoàn thiện:**
   - File `notebooks/flexsds_colab_server.ipynb` hiện diện trong repository, chứa đầy đủ quy trình cài đặt và kết nối đường hầm Cloudflare Tunnel.
4. **Trải nghiệm Chat dòng lệnh mượt mà (`flexsds chat`):**
   - Lệnh `flexsds chat` hoạt động ổn định, nhận phản hồi streaming không giật lag.
   - Thống kê chính xác số lượng token, tốc độ tokens/s và thời gian phản hồi sau mỗi lượt prompt.
5. **Bảo mật tuyệt đối khóa định danh (`flexsds auth`):**
   - Token người dùng được bảo vệ tại `~/.flexsds/credentials` với quyền hạn `0600`, hoàn toàn nằm ngoài kho lưu trữ Git và không xuất hiện trong `serving_config.json`.
   - Cơ chế nạp khóa trực tiếp vào biến môi trường trong bộ nhớ khi kích hoạt ứng dụng Modal/Colab.
6. **Độ bao phủ kiểm thử tự động (CPU Unit Tests):**
   - Bổ sung bộ test cho `LocalServingRunner`, `ModalServingRunner`, `ColabServingRunner`, `ChatClient` và `CredentialsManager`.
   - Toàn bộ bài test chạy vượt qua 100% trong môi trường CI/CD thuần CPU (sử dụng cờ `--dry-run` hoặc mock HTTP).
7. **Tuân thủ quy tắc văn bản:**
   - Tài liệu đặc tả và toàn bộ chuỗi hiển thị terminal tuân thủ nghiêm ngặt quy tắc viết hoa chuẩn tiếng Việt, không phạm lỗi Title Case tiếng Anh.