"""Builds comprehensive System Design Document (SDD) in Excel (.xlsx), CSV, and Markdown."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from export_spec_excel_core import create_xlsx


def generate_sdd_data():
    # ---------------------------------------------------------
    # SHEET 1: TỔNG QUAN KIẾN TRÚC HỆ THỐNG
    # ---------------------------------------------------------
    sheet1_rows = [
        ["TÀI LIỆU THIẾT KẾ HỆ THỐNG FLEXSDS CLI (SYSTEM DESIGN DOCUMENT - SDD)"],
        ["PHÂN HỆ GIAO DIỆN DÒNG LỆNH VÀ BỘ ĐIỀU PHỐI PHỤC VỤ ĐA NỀN TẢNG"],
        [""],
        ["1. THÔNG TIN CHUNG VỀ TÀI LIỆU"],
        ["Tên hệ thống", "FlexSDS: Flexible Speculative Decoding Serving Engine for LLMs"],
        ["Phân hệ thiết kế", "FlexSDS Command-Line Interface (CLI) & Serving Backend"],
        ["Đơn vị thực hiện", "Nhóm đồ án D22 - Học viện Công nghệ Bưu chính Viễn thông (PTIT)"],
        ["Sinh viên thực hiện", "Cấn Đức Khôi (Chủ nhiệm phân hệ CLI & Client), Bùi Đức Đại (Kiến trúc động cơ & Lineage)"],
        ["Phiên bản thiết kế", "Phiên bản 2.0 (Chuẩn hóa HardwareAdaptive và bảo mật zero-leak)"],
        ["Mục tiêu tài liệu", "Đặc tả chi tiết kiến trúc phân lớp, thiết kế thành phần, cấu trúc tệp dữ liệu, giao thức API và cơ chế xử lý sự cố."],
        [""],
        ["2. KIẾN TRÚC PHÂN LỚP (LAYERED ARCHITECTURE)"],
        ["Mã tầng", "Tên tầng kiến trúc", "Các module thành phần", "Trách nhiệm chính trong hệ thống"],
        ["L1 - Presentation", "Tầng giao diện người dùng", "main.py, wizard.py, banner.py, task_runner.py, chat.py, monitor.py", "Tương tác TTY, bắt phím điều hướng, hiển thị hộp căn viền, stream token chat, phím tắt Ctrl+O."],
        ["L2 - Core Engine", "Tầng logic cốt lõi & an toàn", "hardware.py, cost_model.py, compatibility.py, credentials.py, controller.py", "Nhận diện phần cứng, nạp hồ sơ chi phí GPU, kiểm tra tương thích từ vựng dị thể, bảo mật khóa."],
        ["L3 - Serving Backend", "Tầng thực thi phục vụ", "local_runner.py, modal_runner.py, modal_app.py, colab_runner.py, client.py", "Điều phối tiến trình vLLM cục bộ, container Modal Serverless, đường hầm Colab Tunnel, HTTP client."],
        ["L4 - Infrastructure", "Tầng hạ tầng & tệp dữ liệu", "serving_config.json, ~/.flexsds/credentials, serving_trace.jsonl, vLLM runtime", "Tệp lưu trữ cấu hình, biến môi trường, khóa phân quyền 0600, tiến trình daemon vLLM."],
        [""],
        ["3. CÁC NGUYÊN TẮC THIẾT KẾ CỐT LÕI (DESIGN PRINCIPLES)"],
        ["Nguyên tắc", "Tên tiếng Anh", "Ý nghĩa và cách áp dụng trong FlexSDS"],
        ["Thích ứng phần cứng", "Hardware-Centric Adaptation", "Tối ưu hóa số token nháp k dựa trên hồ sơ độ trễ phần cứng GPU cụ thể (sm_89, sm_75)."],
        ["Bảo mật không rò rỉ", "Zero-Leak Security", "Lưu token bảo mật tại ~/.flexsds/credentials (0600), không bao giờ ghi vào repo hay file chia sẻ."],
        ["Khách hàng độc lập", "Zero Heavy Dependency", "Client CLI chỉ dùng thư viện chuẩn Python (urllib), khởi động tức thì < 0.2s, không cần requests/openai."],
        ["Phân tách trách nhiệm", "Separation of Concerns", "Tách bạch tuyệt đối giữa logic giao diện dòng lệnh, bộ kiểm tra an toàn và tiến trình server thực thi."],
        ["Chịu lỗi êm dịu", "Graceful Degradation", "Khi thiếu GPU hoặc không có TTY, hệ thống tự động fallback an toàn mà không làm sập tiến trình."],
    ]

    sheet1_widths = {0: 22.0, 1: 30.0, 2: 45.0, 3: 55.0}

    # ---------------------------------------------------------
    # SHEET 2: THIẾT KẾ CHI TIẾT MODULE VÀ LỚP (MODULE & CLASS DESIGN)
    # ---------------------------------------------------------
    sheet2_headers = [
        "Mã thành phần",
        "Tên tệp mã nguồn",
        "Tên lớp / Hàm chính",
        "Tầng kiến trúc",
        "Trách nhiệm thiết kế",
        "Thuộc tính chính",
        "Phương thức chính / API",
        "Phụ thuộc (Dependencies)",
    ]

    sheet2_data = [
        [
            "MOD-01", "src/flexsds/cli/main.py", "main(), parse_args()", "Presentation",
            "Điểm vào phân phối toàn bộ lệnh CLI (detect, profile, auth, serve, chat, monitor, stop).",
            "parser: ArgumentParser, subparsers",
            "parse_args(argv), main(argv)",
            "argparse, wizard, hardware, credentials, local_runner, modal_runner",
        ],
        [
            "MOD-02", "src/flexsds/cli/banner.py", "print_banner(), render_banner()", "Presentation",
            "Kết xuất logo trường PTIT và thông tin hệ thống HardwareAdaptive v0.2.",
            "PTIT_LOGO_ANSI: tuple, info_lines: list",
            "render_banner(hw_info), print_banner(hw_info)",
            "hardware.HardwareInfo, mã màu ANSI",
        ],
        [
            "MOD-03", "src/flexsds/cli/wizard.py", "run_interactive_wizard(), render_box()", "Presentation",
            "Trình hướng dẫn cấu hình tương tác từng bước bằng phím mũi tên và hộp viền căn chỉnh.",
            "PLATFORM_CHOICES: list, HAS_TERMIOS: bool",
            "prompt_choice(), prompt_bool(), prompt_string(), render_box(), strip_ansi()",
            "termios, tty, hardware, compatibility, cost_model, local_runner",
        ],
        [
            "MOD-04", "src/flexsds/cli/task_runner.py", "TaskRunner, Task, TaskStatus", "Presentation",
            "Quản lý danh sách tác vụ thu gọn, bắt phím tắt Ctrl+O để mở rộng/thu gọn nhật ký thời gian thực.",
            "tasks: list[Task], show_details: bool, _listener_thread",
            "add_task(), toggle_details(), handle_key(), render_view(), run_task()",
            "threading, select, termios, tty, time",
        ],
        [
            "MOD-05", "src/flexsds/core/hardware.py", "HardwareInfo, detect_hardware()", "Core Engine",
            "Phát hiện thông số phần cứng GPU NVIDIA và RAM khả dụng không nạp context nặng.",
            "device_name, sm_architecture, total_vram_mb, free_vram_mb, is_cuda_available",
            "free_vram_gb, total_vram_gb, summary(), detect_hardware()",
            "torch (nếu có), os, dataclasses",
        ],
        [
            "MOD-06", "src/flexsds/core/cost_model.py", "FlexCostModel, load_profile_for_hardware()", "Core Engine",
            "Mô hình chi phí tính toán GPU và bộ nạp hồ sơ độ trễ phần cứng tự động.",
            "c_draft_ms, c_target_ms, alpha_decay, profile_path",
            "predict_latency(), load_profile_for_hardware()",
            "json, pathlib.Path, hardware.HardwareInfo",
        ],
        [
            "MOD-07", "src/flexsds/core/compatibility.py", "CompatibilityReport, check_compatibility()", "Core Engine",
            "Phân tích tương thích cặp mô hình, nhận diện từ vựng dị thể và cảnh báo tràn bộ nhớ VRAM.",
            "is_compatible, draft_method, use_heterogeneous_vocab, warnings",
            "check_compatibility(), pre_download_guard(), summary()",
            "hardware.HardwareInfo, re",
        ],
        [
            "MOD-08", "src/flexsds/core/credentials.py", "CredentialsManager, mask_secret()", "Core Engine",
            "Quản lý khóa bí mật an toàn ngoài repo tại ~/.flexsds/credentials với quyền phân quyền 0600.",
            "DEFAULT_PATH: Path, path: Path",
            "get_credential(), set_credential(), delete_credential(), inject_into_environ(), get_all_masked()",
            "os, json, stat, pathlib.Path",
        ],
        [
            "MOD-09", "src/flexsds/core/controller.py", "HardwareAdaptiveController, HardwareAdaptiveDecision", "Core Engine",
            "Bộ điều phối ngân sách đầu cơ thích ứng tối ưu hóa số token nháp k dựa trên chi phí GPU.",
            "cost_model, initial_alpha, ema_beta, min_free_vram_mb",
            "make_decision(), update_alpha(), _make_decision()",
            "cost_model.FlexCostModel, dataclasses, logging",
        ],
        [
            "MOD-10", "src/flexsds/server/local_runner.py", "LocalServingRunner", "Serving Backend",
            "Điều phối khởi chạy máy chủ vLLM cục bộ với Lineage Executor, phát hiện trùng cổng và daemon.",
            "PID_FILE: str, LOG_FILE: str",
            "build_command(), check_port_available(), setup_environment(), launch(), stop()",
            "subprocess, socket, signal, os, pathlib.Path",
        ],
        [
            "MOD-11", "src/flexsds/server/modal_runner.py", "ModalServingRunner", "Serving Backend",
            "Điều phối thực thi đám mây Modal Serverless, kiểm tra định danh và giám sát tiến trình.",
            "DEFAULT_APP_NAME: str, cred_mgr: CredentialsManager",
            "check_credentials(), build_modal_command(), launch(), stop(), check_process_status()",
            "subprocess, credentials.CredentialsManager, pathlib.Path",
        ],
        [
            "MOD-12", "src/flexsds/server/modal_app.py", "app, flexsds_image, serve()", "Serving Backend",
            "Ứng dụng Modal serverless độc lập đóng gói vLLM v0.30.0, duy trì vòng lặp phục vụ trên GPU L4.",
            "APP_NAME: str, flexsds_image: modal.Image, model_volume: modal.Volume",
            "serve(gpu, target, draft)",
            "modal, vllm, subprocess, os, json",
        ],
        [
            "MOD-13", "src/flexsds/server/colab_runner.py", "ColabServingRunner", "Serving Backend",
            "Quản lý chỉ dẫn Google Colab và kiểm tra sức khỏe đường hầm Cloudflare Tunnel.",
            "COLAB_NOTEBOOK_URL: str",
            "get_colab_instructions(), verify_tunnel_health()",
            "urllib.request, urllib.error",
        ],
        [
            "MOD-14", "src/flexsds/client/client.py", "FlexSDSClient, EndpointUnreachableError", "Serving Backend",
            "Khách hàng HTTP thuần chuẩn OpenAI hỗ trợ streaming SSE không phụ thuộc thư viện mạng ngoài.",
            "base_url: str, api_key: str",
            "check_health(), get_models(), stream_chat(), _make_request()",
            "urllib.request, urllib.error, json",
        ],
        [
            "MOD-15", "src/flexsds/client/chat.py", "InteractiveChatSession, PerformanceMetrics", "Presentation",
            "Phiên trò chuyện tương tác dòng lệnh hai chiều, đo đạc TPOT và tốc độ sinh token.",
            "client: FlexSDSClient, model: str, temperature: float, history: list",
            "start(), tokens_per_second, tpot_ms, run_interactive_chat()",
            "client.FlexSDSClient, wizard.render_box, time, sys",
        ],
        [
            "MOD-16", "src/flexsds/client/monitor.py", "TelemetryMonitor, run_live_monitor()", "Presentation",
            "Bộ theo dõi thích ứng thời gian thực đọc tail tệp JSONL telemetry và định dạng màu sắc.",
            "trace_path: str, _file_pos: int",
            "read_new_records(), format_record(), run_live_monitor()",
            "json, os, time, sys",
        ],
    ]

    sheet2_rows = [sheet2_headers] + sheet2_data
    sheet2_widths = {0: 12.0, 1: 30.0, 2: 35.0, 3: 16.0, 4: 45.0, 5: 35.0, 6: 40.0, 7: 35.0}

    # ---------------------------------------------------------
    # SHEET 3: THIẾT KẾ CẤU TRÚC TỆP VÀ LƯỢC ĐỒ DỮ LIỆU (DATA & SCHEMA)
    # ---------------------------------------------------------
    sheet3_headers = [
        "Mã thiết kế",
        "Tên tệp / Mục dữ liệu",
        "Loại lưu trữ",
        "Đường dẫn lưu trữ",
        "Phân quyền bảo mật",
        "Mục đích sử dụng",
        "Lược đồ cấu trúc dữ liệu (Field Schema)",
    ]

    sheet3_data = [
        [
            "FILE-01", "serving_config.json", "JSON", "./serving_config.json", "0644 (Đọc/Ghi)",
            "Lưu vết cấu hình phục vụ do người dùng chọn qua wizard hoặc cờ CLI.",
            "flexsds_version (str), action (str), backend (local|modal|colab), target_model (str), draft_model (str), draft_method (str), use_heterogeneous_vocab (bool), enable_adaptive (bool), profile_applied (str), target_gpu (str), colab_tunnel_url (str), modal_deploy (bool), hardware (dict).",
        ],
        [
            "FILE-02", "credentials", "JSON", "~/.flexsds/credentials", "0600 (Chỉ User)",
            "Lưu trữ khóa API và token truy cập các dịch vụ đám mây an toàn ngoài kho mã nguồn.",
            "modal: {MODAL_TOKEN_ID: str, MODAL_TOKEN_SECRET: str}, huggingface: {HF_TOKEN: str}, custom_providers: dict.",
        ],
        [
            "FILE-03", ".flexsds.pid", "Text", "./.flexsds.pid", "0644 (Đọc/Ghi)",
            "Lưu mã tiến trình (PID) của máy chủ vLLM chạy ngầm để phục vụ lệnh dừng flexsds stop.",
            "Chuỗi số nguyên biểu diễn PID của tiến trình cha (ví dụ: '12485').",
        ],
        [
            "FILE-04", "flexsds_server.log", "Text Log", "./flexsds_server.log", "0644 (Nối thêm)",
            "Chuyển hướng toàn bộ stdout và stderr của máy chủ vLLM khi chạy ở chế độ background daemon.",
            "Các dòng nhật ký thời gian thực do vLLM engine, scheduler và LineageUniProcExecutor xuất ra.",
        ],
        [
            "FILE-05", "serving_trace.jsonl", "JSONL", "telemetry/serving_trace.jsonl", "0644 (Nối thêm)",
            "Ghi nhận chi tiết từng bước giải mã đầu cơ động phục vụ bộ theo dõi telemetry monitor.",
            "step (int), lane (str), executed_draft_token_count (int), accepted_draft_token_count (int), alpha_ema (float), source_batch_scalar_k (int).",
        ],
        [
            "FILE-06", "profile_sm89_l4.json", "JSON", "src/flexsds/profiles/profile_sm89_l4.json", "0644 (Chỉ đọc)",
            "Hồ sơ độ trễ phần cứng chuẩn đo đạc cho GPU NVIDIA L4 (Ada Lovelace, 24GB).",
            "gpu_name (str), sm_architecture ('sm_89'), target_latency_ms (float), draft_latency_ms (float), alpha_decay (float), optimal_k_curve (dict).",
        ],
        [
            "FILE-07", "profile_sm75_t4.json", "JSON", "src/flexsds/profiles/profile_sm75_t4.json", "0644 (Chỉ đọc)",
            "Hồ sơ độ trễ phần cứng chuẩn đo đạc cho GPU NVIDIA T4 (Turing, 16GB).",
            "gpu_name (str), sm_architecture ('sm_75'), target_latency_ms (float), draft_latency_ms (float), alpha_decay (float), optimal_k_curve (dict).",
        ],
        [
            "FILE-08", "flexsds_colab_server.ipynb", "Jupyter Notebook", "notebooks/flexsds_colab_server.ipynb", "0644 (Mã nguồn)",
            "Sổ tay Google Colab chuẩn hóa để người dùng khởi chạy máy chủ GPU T4/L4 miễn phí và mở đường hầm.",
            "Cấu trúc cells: Clone repo (không lộ PAT) -> Cài đặt vllm v0.30.0 -> Cài đặt cloudflared -> Khởi chạy vLLM server và Cloudflare Tunnel.",
        ],
    ]

    sheet3_rows = [sheet3_headers] + sheet3_data
    sheet3_widths = {0: 12.0, 1: 28.0, 2: 15.0, 3: 35.0, 4: 18.0, 5: 35.0, 6: 60.0}

    # ---------------------------------------------------------
    # SHEET 4: THIẾT KẾ GIAO DIỆN LẬP TRÌNH VÀ GIAO THỨC (API & PROTOCOL)
    # ---------------------------------------------------------
    sheet4_headers = [
        "Mã giao diện",
        "Tên giao thức / Lệnh",
        "Phương thức",
        "Nguồn phát -> Nơi nhận",
        "Định dạng truyền nhận",
        "Tham số đầu vào (Request Parameters)",
        "Đầu ra kỳ vọng (Response / Behavior)",
        "Xử lý lỗi & Trạng thái",
    ]

    sheet4_data = [
        [
            "API-01", "CLI Detect", "CLI Subcommand", "Người dùng -> MainDispatcher", "Dòng lệnh",
            "Không tham số: 'flexsds detect'",
            "In logo PTIT, banner HardwareAdaptive và thông số tóm tắt phần cứng GPU.",
            "Mã thoát exit code 0.",
        ],
        [
            "API-02", "CLI Profile", "CLI Subcommand", "Người dùng -> MainDispatcher", "Dòng lệnh",
            "Không tham số: 'flexsds profile'",
            "Nạp hồ sơ chi phí GPU và in tên file hồ sơ tương ứng (ví dụ profile_sm89_l4.json).",
            "Mã thoát exit code 0.",
        ],
        [
            "API-03", "CLI Auth Login", "CLI Subcommand", "Người dùng -> CredentialsManager", "Dòng lệnh / Prompt ẩn",
            "--provider [modal|huggingface|...], --key [tên_khóa], --token [giá_trị_nếu_có]",
            "Lưu bảo mật vào ~/.flexsds/credentials quyền 0600; in xác nhận thành công.",
            "Ẩn nhập mật khẩu bằng getpass nếu không truyền token.",
        ],
        [
            "API-04", "CLI Auth Status", "CLI Subcommand", "Người dùng -> CredentialsManager", "Dòng lệnh",
            "Không tham số: 'flexsds auth status'",
            "In danh sách dịch vụ và token đã lưu ở dạng che giấu (mask_secret).",
            "Mã thoát exit code 0, không lộ key gốc.",
        ],
        [
            "API-05", "CLI Auth Logout", "CLI Subcommand", "Người dùng -> CredentialsManager", "Dòng lệnh",
            "Tùy chọn: '--provider <tên_dịch_vụ>' (hoặc không truyền để xóa hết)",
            "Xóa khóa trong file và dọn sạch biến môi trường trong bộ nhớ.",
            "Mã thoát exit code 0.",
        ],
        [
            "API-06", "CLI Serve Flags", "CLI Subcommand", "Người dùng -> TaskRunner / Runner", "Dòng lệnh",
            "--target <tên>, --draft <tên>, --backend [local|modal|colab], --port, --background, --dry-run",
            "Điều phối khởi chạy qua TaskRunner; kiểm tra cổng; chạy server.",
            "Phát hiện cổng trùng, bắt lỗi định danh modal.",
        ],
        [
            "API-07", "CLI Stop", "CLI Subcommand", "Người dùng -> LocalServingRunner", "Dòng lệnh",
            "Không tham số: 'flexsds stop'",
            "Gửi SIGTERM dọn sạch nhóm tiến trình vLLM; xóa file .flexsds.pid.",
            "Báo êm dịu nếu không có tiến trình chạy.",
        ],
        [
            "API-08", "CLI Chat", "CLI Subcommand", "Người dùng -> InteractiveChatSession", "Dòng lệnh",
            "--endpoint <url_api>, --model <tên_mô_hình>, --temperature <nhiệt_độ>",
            "Mở phiên chat terminal; stream từng token qua SSE; in bảng đo TPOT sau mỗi lượt.",
            "Báo lỗi kết nối nếu server chưa bật (exit code 1).",
        ],
        [
            "API-09", "CLI Monitor", "CLI Subcommand", "Người dùng -> TelemetryMonitor", "Dòng lệnh",
            "--trace-file <đường_dẫn_jsonl>, --refresh-rate <giây>",
            "Lắng nghe và in liên tục các bước suy luận: Step, k, chấp nhận, alpha, làn.",
            "Bỏ qua dòng JSON ghi dở, dừng êm khi ấn Ctrl+C.",
        ],
        [
            "API-10", "HTTP Health Check", "GET /health hoặc /v1/models", "FlexSDSClient -> vLLM Server", "HTTP Request",
            "Headers: Authorization Bearer (nếu có), timeout 3s",
            "Mã trạng thái HTTP 200 OK nếu máy chủ đã nạp xong mô hình và sẵn sàng.",
            "Trả về False nếu kết nối bị từ chối hoặc timeout.",
        ],
        [
            "API-11", "HTTP Chat Streaming", "POST /v1/chat/completions", "FlexSDSClient -> vLLM Server", "HTTP / SSE",
            "JSON Body: {model, messages: list, stream: true, temperature: float}",
            "Luồng Server-Sent Events (SSE) từng chunk 'data: {\"choices\": [{\"delta\": ...}]}'.",
            "Ném EndpointUnreachableError nếu mất mạng.",
        ],
    ]

    sheet4_rows = [sheet4_headers] + sheet4_data
    sheet4_widths = {0: 12.0, 1: 22.0, 2: 15.0, 3: 32.0, 4: 20.0, 5: 35.0, 6: 45.0, 7: 30.0}

    # ---------------------------------------------------------
    # SHEET 5: THIẾT KẾ XỬ LÝ SỰ CỐ VÀ CHỊU LỖI (FAULT TOLERANCE)
    # ---------------------------------------------------------
    sheet5_headers = [
        "Mã sự cố",
        "Tình huống sự cố",
        "Phân hệ phát hiện",
        "Nguyên nhân gốc rễ (Root Cause)",
        "Chiến lược xử lý & Khắc phục (Mitigation)",
        "Trạng thái sau khắc phục",
        "Thông báo hiển thị cho người dùng",
    ]

    sheet5_data = [
        [
            "ERR-01", "Xung đột cổng mạng (Port Conflict)", "LocalServingRunner",
            "Cổng mạng (mặc định 8000) đang bị chiếm bởi tiến trình vLLM cũ hoặc ứng dụng khác.",
            "Dùng socket thăm dò trước khi gọi subprocess; nếu bị chiếm thì từ chối khởi chạy.",
            "Hệ thống không khởi chạy đè tiến trình hỏng, giữ nguyên trạng thái an toàn.",
            "✔ '[Lỗi] Cổng 8000 đang bị chiếm dụng. Gợi ý: Dùng cờ --port hoặc chạy: flexsds stop'.",
        ],
        [
            "ERR-02", "Thiếu thông tin xác thực Modal", "ModalServingRunner",
            "Chưa cấu hình MODAL_TOKEN_ID/SECRET trong biến môi trường, credentials file hoặc ~/.modal.toml.",
            "Bộ gác cổng check_credentials() kiểm tra trước; nếu thiếu thì chặn gọi subprocess modal.",
            "Chặn sớm trước khi gọi lệnh hệ thống, không tốn tài nguyên vô ích.",
            "✔ '[Lỗi định danh] Chưa cấu hình thông tin xác thực Modal. Vui lòng chạy: flexsds auth login'.",
        ],
        [
            "ERR-03", "Ứng dụng Modal thoát sớm (Container Stop)", "ModalServingRunner / modal_app.py",
            "Hàm serve() trong modal_app.py kết thúc ngay làm Modal hiểu lầm tác vụ hoàn thành và tắt container.",
            "Bổ sung vòng lặp proc.wait() duy trì tiến trình vLLM; hàm check_process_status() bắt mã thoát.",
            "Container giữ trạng thái hoạt động; nếu chết thì CLI phát hiện ngay lập tức.",
            "✔ Ném RuntimeError kèm mã lỗi thoát của container thay vì treo terminal.",
        ],
        [
            "ERR-04", "Đường hầm Cloudflare Colab chưa mở", "ColabServingRunner",
            "Máy chủ Colab đang tải checkpoint mô hình vào VRAM, chưa kịp lắng nghe port.",
            "verify_tunnel_health gửi HTTP GET trong timeout 5s; bắt lỗi URLError êm dịu.",
            "Trả về False, không làm gián đoạn chương trình wizard.",
            "✔ '⚠ Endpoint not responding yet (model weights may still be loading)'.",
        ],
        [
            "ERR-05", "Mất kết nối máy chủ khi đang trò chuyện", "FlexSDSClient / InteractiveChatSession",
            "Máy chủ vLLM bị tắt đột ngột hoặc rớt mạng khi đang stream token.",
            "Bắt lỗi URLError, đóng luồng socket an toàn, chuyển thành EndpointUnreachableError.",
            "Phiên chat bắt ngoại lệ, không bị văng crash, cho phép người dùng thử lại.",
            "✔ '[Lỗi đường truyền] Mất kết nối tới server: <chi_tiết_lỗi>'.",
        ],
        [
            "ERR-06", "Người dùng nhấn Ctrl+C khi đang stream", "InteractiveChatSession",
            "Người dùng muốn ngắt câu trả lời dài của mô hình giữa chừng.",
            "Bắt KeyboardInterrupt trong vòng lặp đọc SSE; dừng ghi stdout; gửi dòng ngắt nhẹ nhàng.",
            "Lịch sử hội thoại được bảo toàn; phiên chat tiếp tục nhận prompt mới.",
            "✔ '(Hủy lượt trả lời hiện tại)' và hiển thị lại 'You > '.",
        ],
        [
            "ERR-07", "Tệp telemetry JSONL chưa tồn tại", "TelemetryMonitor",
            "Người dùng chạy lệnh monitor khi máy chủ chưa thực hiện lượt suy luận nào.",
            "Kiểm tra os.path.exists(); nếu chưa có thì in thông báo chờ đợi và ngủ chu kỳ refresh.",
            "Bộ theo dõi tiếp tục lắng nghe, tự động kết nối ngay khi tệp xuất hiện.",
            "✔ '(Đang chờ máy chủ tạo tệp trace tại telemetry/serving_trace.jsonl...)'.",
        ],
        [
            "ERR-08", "Dòng bản ghi JSONL bị ghi dở", "TelemetryMonitor",
            "Tiến trình server đang ghi buffer dở thì bộ monitor đọc phải.",
            "Bọc json.loads trong khối try...except json.JSONDecodeError; bỏ qua dòng dở.",
            "Bỏ qua dòng lỗi, tiếp tục đọc dòng hoàn chỉnh ở chu kỳ quét tiếp theo.",
            "Không in thông báo rác, giữ màn hình sạch sẽ.",
        ],
    ]

    sheet5_rows = [sheet5_headers] + sheet5_data
    sheet5_widths = {0: 12.0, 1: 30.0, 2: 25.0, 3: 45.0, 4: 45.0, 5: 35.0, 6: 45.0}

    sheets = {
        "Tong_quan_Kien_truc": sheet1_rows,
        "Thiet_ke_Module_Class": sheet2_rows,
        "Thiet_ke_Du_lieu_va_Tep": sheet3_rows,
        "Thiet_ke_Giao_thuc_API": sheet4_rows,
        "Thiet_ke_Xu_ly_Su_co": sheet5_rows,
    }

    sheet_widths = {
        "Tong_quan_Kien_truc": sheet1_widths,
        "Thiet_ke_Module_Class": sheet2_widths,
        "Thiet_ke_Du_lieu_va_Tep": sheet3_widths,
        "Thiet_ke_Giao_thuc_API": sheet4_widths,
        "Thiet_ke_Xu_ly_Su_co": sheet5_widths,
    }

    return sheets, sheet_widths, sheet2_rows


def export_csv_sdd(sheet2_rows, csv_path: str):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for row in sheet2_rows:
            writer.writerow(row)
    print(f"✔ Đã tạo tệp CSV thiết kế thành phần: {csv_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "logs" / "Khoi" / "spec&plan"
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = str(output_dir / "flexsds_cli_system_design_document.xlsx")
    csv_path = str(output_dir / "flexsds_cli_system_design_document.csv")
    md_target_path = output_dir / "flexsds_cli_system_design_document.md"
    md_template_path = output_dir / "flexsds_cli_system_design_document_template.md"

    # Sao chép / đồng bộ file markdown đúng tên flexsds_cli_system_design_document.md
    if md_template_path.exists() and not md_target_path.exists():
        shutil.copyfile(md_template_path, md_target_path)
        print(f"✔ Đã đồng bộ tệp Markdown SDD: {md_target_path}")

    sheets, widths, sheet2_rows = generate_sdd_data()
    create_xlsx(sheets, widths, xlsx_path)
    export_csv_sdd(sheet2_rows, csv_path)


if __name__ == "__main__":
    main()
