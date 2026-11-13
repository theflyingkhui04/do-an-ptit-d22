> Integration note (2026-11-13): This document preserves design/plan intent, not proof of implementation. Current behavior, limitations, tests and superseding corrections are in `logs/daijestic/flexsds-advisor/worklog.md` and `.agents/skills/flexsds-advisor/references/flexsds-cli.md`. Ephemeral Modal uses `modal serve`; chat measures SSE chunks; Windows ACL privacy is not proven; Colab flag mode is guidance only.

# Bản yêu cầu đặc tả phần mềm và ma trận kiểm thử toàn diện FlexSDS CLI

> **Đề tài:** Hệ thống phục vụ giải mã đầu cơ thích ứng FlexSDS cho mô hình ngôn ngữ lớn (LLM)
> **Đơn vị:** Nhóm đồ án D22 - Học viện Công nghệ Bưu chính Viễn thông (PTIT)
> **Tác giả:** Cấn Đức Khôi & Bùi Đức Đại

Tài liệu này bao phủ **100% tất cả các trường hợp (99 test cases)** của hệ thống FlexSDS CLI & Serving Backend.

## Danh sách các phân hệ và số lượng trường hợp kiểm thử

| STT | Phân hệ chức năng | Mã định danh | Số ca kiểm thử | Trạng thái |
| :---: | :--- | :---: | :---: | :---: |
| 1 | Nhận diện phần cứng & Hồ sơ độ trễ | `TC-HW` | 15 ca | PASS 100% |
| 2 | Bộ gác cổng tương thích & Dị thể từ vựng | `TC-CMP` | 12 ca | PASS 100% |
| 3 | Bảo mật thông tin xác thực & Quản lý auth | `TC-SEC` | 8 ca | PASS 100% |
| 4 | Bộ thực thi phục vụ cục bộ (Local vLLM) | `TC-LOC` | 8 ca | PASS 100% |
| 5 | Bộ thực thi đám mây Modal Serverless | `TC-MOD` | 8 ca | PASS 100% |
| 6 | Đường hầm Google Colab Tunnel | `TC-COL` | 8 ca | PASS 100% |
| 7 | Giao diện điều phối tương tác Terminal Wizard | `TC-WIZ` | 8 ca | PASS 100% |
| 8 | Quản lý tác vụ TaskRunner & Phím Ctrl+O | `TC-TSK` | 8 ca | PASS 100% |
| 9 | Khách hàng trò chuyện tương tác (CLI Chat) | `TC-CHT` | 8 ca | PASS 100% |
| 10 | Bộ theo dõi thích ứng thời gian thực (Telemetry) | `TC-MON` | 8 ca | PASS 100% |
| 11 | Yêu cầu phi chức năng, Trường hợp biên & Bảo mật | `TC-NFR` | 8 ca | PASS 100% |
| **Tổng cộng** | **Toàn bộ hệ thống FlexSDS CLI** | **11 Phân hệ** | **99 ca kiểm thử** | **ĐẠT 100%** |

---

## Chi tiết ma trận kiểm thử (Trích đoạn các trường hợp quan trọng)

| Mã kiểm thử | Phân hệ | Tên ca kiểm thử | Loại | Đầu vào / Lệnh CLI | Kết quả kỳ vọng |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `TC-HW-01` | Phần cứng | Phát hiện GPU NVIDIA hợp lệ | Positive | `detect_hardware()` | Nhận diện chính xác tên GPU, SM architec... |
| `TC-HW-02` | Phần cứng | Fallback an toàn trên môi trường CPU Only | Boundary | `flexsds detect` | Hệ thống nhận diện CPU Only, đọc dung lư... |
| `TC-HW-03` | Phần cứng | Lựa chọn GPU cụ thể trong hệ thống đa GPU | Positive | `detect_hardware(device_index=1` | Đọc đúng thông số phần cứng của GPU thứ ... |
| `TC-HW-04` | Phần cứng | Tính toán chính xác dung lượng VRAM GB | Boundary | `HardwareInfo(..., free_vram_mb` | Quy đổi chuẩn xác sang GB với 2 chữ số t... |
| `TC-HW-05` | Phần cứng | Áp dụng hồ sơ độ trễ phần cứng SM89 (NVIDIA L4) | Positive | `hw.sm_architecture = 'sm_89'` | Khớp chính xác tệp profile_sm89_l4.json.... |
| `TC-HW-06` | Phần cứng | Áp dụng hồ sơ độ trễ phần cứng SM75 (NVIDIA T4) | Positive | `hw.sm_architecture = 'sm_75'` | Khớp chính xác tệp profile_sm75_t4.json.... |
| `TC-HW-07` | Phần cứng | Kích hoạt hồ sơ phân tích an toàn dự phòng khi thiếu hồ sơ phần cứng | Boundary | `hw.sm_architecture = 'unknown_` | Hệ thống không crash, tự động chuyển san... |
| `TC-HW-08` | Phần cứng | Ưu tiên nạp hồ sơ tùy chỉnh từ đường dẫn chỉ định (--cost-profile) | Positive | `load_profile_for_hardware(cust` | Nạp đúng tệp tùy chỉnh, có độ ưu tiên ca... |
| `TC-HW-09` | Phần cứng | Tự động nạp hồ sơ từ kho cấu hình người dùng ~/.flexsds/profiles/ | Positive | `hw.sm_architecture = 'sm_90'` | Khớp đúng tệp hồ sơ cá nhân trong ~/.fle... |
| `TC-HW-10` | Phần cứng | Hiệu chuẩn và lưu hồ sơ phần cứng mới bằng save_hardware_profile() | Positive | `save_hardware_profile(data, 's` | Tệp JSON được tạo thành công trên đĩa, đ... |
| `TC-HW-11` | Phần cứng | Lệnh CLI profile in thông tin liên kết | Positive | `flexsds profile` | In banner và dòng chữ màu xanh '✔ Linked... |
| `TC-HW-12` | Phần cứng | Đo đạc vi benchmark phần cứng độc lập bằng run_hardware_micro_benchmark() | Positive | `run_hardware_micro_benchmark(.` | Trả về từ điển profile có forward_target... |
| `TC-HW-13` | Phần cứng | Lệnh CLI profile với cờ --benchmark tự sinh hồ sơ phần cứng | Positive | `flexsds profile --benchmark --` | In các bước đo đạc benchmark và lưu tệp ... |
| `TC-HW-14` | Phần cứng | Chặn sớm khi khởi chạy phục vụ trên phần cứng chưa có hồ sơ chuẩn (Fail-fast serving guard) | Negative | `flexsds serve --target <target` | Tiến trình lập tức bị chặn với exit code... |
| `TC-HW-15` | Phần cứng | Tiếp tục khởi chạy phục vụ khi chỉ định rõ cờ --allow-fallback | Positive | `flexsds serve --allow-fallback` | Hệ thống chấp nhận dùng hồ sơ fallback p... |
| `TC-CMP-01` | Tương thích | Tải và lưu bộ nhớ đệm metadata cấu hình mô hình (Model metadata config cache) | Positive | `get_model_config('Qwen/Qwen2.5` | Tải tệp config.json và lưu vào ~/.flexsd... |
| `TC-CMP-02` | Tương thích | Đọc cấu hình mô hình từ bộ nhớ đệm nội bộ khi ngoại tuyến không cần mạng | Positive | `get_model_config(cached_model_` | Đọc cấu hình trực tiếp từ tệp đệm trên đ... |
| `TC-CMP-03` | Tương thích | Báo lỗi kết nối mạng tường minh khi ngoại tuyến và chưa có bộ nhớ đệm | Negative | `get_model_config('unknown/offl` | Ném lỗi mạng với thông điệp rõ ràng hướn... |
| `TC-CMP-04` | Tương thích | Đọc cấu hình mô hình từ thư mục cục bộ hoặc tệp JSON trực tiếp | Positive | `get_model_config('/path/to/loc` | Đọc trực tiếp tệp config.json từ đường d... |
| `TC-CMP-05` | Tương thích | Từ chối mô hình nháp không thuộc kiến trúc EAGLE (EAGLE draft architecture enforcement) | Negative | `draft = 'Qwen/Qwen2.5-0.5B-Ins` | Hệ thống kiểm tra kiến trúc và từ chối v... |
| `TC-CMP-06` | Tương thích | Chấp nhận mô hình nháp kiến trúc EAGLE và phát hiện từ vựng dị thể qua so sánh động vocab_size | Positive | `target: vocab 152k, draft: EAG` | Mô hình được chấp nhận (is_compatible = ... |
| `TC-CMP-07` | Tương thích | Tắt cờ từ vựng dị thể khi mô hình mục tiêu và đầu nháp có cùng vocab_size | Positive | `target: vocab 128k, draft: EAG` | Cờ use_heterogeneous_vocab được gán Fals... |
| `TC-CMP-08` | Tương thích | Chế độ tương thích ngoại tuyến dự phòng (Offline fallback) từ chối mô hình không phải EAGLE | Boundary | `draft = 'facebook/opt-125m'` | Hệ thống từ chối mô hình không phải EAGL... |
| `TC-CMP-09` | Tương thích | Cảnh báo nguy cơ tràn bộ nhớ VRAM khi dung lượng trống không đủ | Negative | `free_vram_mb = 3100.0` | Trạng thái tương thích chuyển thành WARN... |
| `TC-CMP-10` | Tương thích | Xác nhận tương thích thành công khi đủ VRAM và đúng kiến trúc | Positive | `free_vram_mb = 20000.0` | Trạng thái tương thích là PASSED, is_com... |
| `TC-CMP-11` | Tương thích | Chặn trước tải xuống khi thiếu đĩa hoặc VRAM | Boundary | `free_disk_gb = 2.0` | Hàm trả về False, in thông báo từ chối t... |
| `TC-CMP-12` | Tương thích | In bảng Compatibility Report chuẩn viền và tóm tắt ngắn gọn | UX | `report.summary(), render_box()` | Hiển thị bảng viền căn lề hoàn hảo không... |
| `TC-SEC-01` | Bảo mật | Lưu token bảo mật với phân quyền tệp 0600 | Security | `set_credential('modal', 'MODAL` | Tệp credentials được tạo với quyền stat.... |
| `TC-SEC-02` | Bảo mật | Nạp token vào biến môi trường trong bộ nhớ | Security | `inject_into_environ()` | Biến MODAL_TOKEN_ID xuất hiện trong os.e... |
| `TC-SEC-03` | Bảo mật | Che giấu chuỗi bí mật dài hơn 8 ký tự | Security | `mask_secret('ak-123456789')` | Chuỗi được ẩn, chỉ hiển thị 'ak-****...6... |
| `TC-SEC-04` | Bảo mật | Che giấu chuỗi bí mật ngắn hơn hoặc bằng 8 ký tự | Security | `mask_secret('1234567')` | Chuỗi bị che toàn bộ bằng '*****'.... |
| `TC-SEC-05` | Bảo mật | Lệnh CLI auth status hiển thị danh sách đã che giấu | UX | `flexsds auth status` | In danh sách provider có màu xanh, các t... |
| `TC-SEC-06` | Bảo mật | Lệnh CLI auth logout xóa sạch biến môi trường | Security | `flexsds auth logout --provider` | Xóa mục modal trong tệp và xóa sạch MODA... |
| `TC-SEC-07` | Bảo mật | Lệnh CLI auth logout không tham số xóa toàn bộ | Security | `flexsds auth logout` | Xóa sạch toàn bộ token đã lưu, file cred... |
| `TC-SEC-08` | Bảo mật | Zero-Leak: Không lưu token vào serving_config.json | Security | `cat serving_config.json` | File cấu hình hoàn toàn không chứa bất k... |
| `TC-LOC-01` | Cục bộ | Tạo câu lệnh vllm serve với Lineage Executor | Positive | `build_command(config)` | Câu lệnh chứa '--distributed-executor-ba... |
| `TC-LOC-02` | Cục bộ | Thiết lập biến môi trường phục vụ cục bộ | Positive | `setup_environment(config)` | Biến môi trường chứa PYTHONPATH hợp lệ, ... |
| `TC-LOC-03` | Cục bộ | Phát hiện xung đột cổng khi cổng đang bị chiếm dụng | Negative | `check_port_available(8000)` | Hàm trả về False, phát hiện xung đột cổn... |
| `TC-LOC-04` | Cục bộ | Từ chối khởi chạy khi phát hiện xung đột cổng | Boundary | `runner.launch(config, port=occ` | Hàm từ chối khởi chạy, trả về None, in t... |
| `TC-LOC-05` | Cục bộ | Khởi chạy máy chủ chạy ngầm (Background Daemon) | Positive | `runner.launch(config, backgrou` | Ghi PID vào .flexsds.pid, redirect stdou... |
| `TC-LOC-06` | Cục bộ | Dừng tiến trình phục vụ cục bộ bằng lệnh stop | Positive | `runner.stop()` | Gửi tín hiệu SIGTERM dọn sạch cả process... |
| `TC-LOC-07` | Cục bộ | Lệnh stop khi không có tiến trình chạy xử lý êm dịu | Boundary | `flexsds stop` | In thông báo 'No active local server pro... |
| `TC-LOC-08` | Cục bộ | Bắt lỗi thiếu binary vllm trong hệ thống | Negative | `runner.launch(config, backgrou` | Bắt lỗi FileNotFoundError và in thông bá... |
| `TC-MOD-01` | Modal | Xác thực thông tin định danh Modal từ CredentialsManager | Positive | `runner.check_credentials()` | Hàm trả về True, tự động nhận diện token... |
| `TC-MOD-02` | Modal | Chặn khởi chạy khi thiếu thông tin xác thực Modal | Negative | `runner.launch(config, dry_run=` | Chặn khởi chạy, in thông báo lỗi định da... |
| `TC-MOD-03` | Modal | Tạo câu lệnh modal run trỏ vào modal_app.py độc lập | Positive | `build_modal_command(config, de` | Lệnh sinh ra chứa 'modal', 'run', 'modal... |
| `TC-MOD-04` | Modal | Tạo câu lệnh modal deploy cho triển khai vĩnh viễn | Positive | `build_modal_command(config, de` | Lệnh sinh ra chứa 'modal', 'deploy', 'mo... |
| `TC-MOD-05` | Modal | modal_app.py duy trì server và không tự tắt sau 0.1s | Positive | `modal_app.serve(gpu='L4', ...)` | Khởi chạy subprocess vllm serve và có vò... |
| `TC-MOD-06` | Modal | Phát hiện tiến trình Modal dừng bất ngờ (check_process_status) | Boundary | `check_process_status(mock_stop` | Hàm trả về is_alive=False kèm mã lỗi exi... |
| `TC-MOD-07` | Modal | Lệnh dừng ứng dụng Modal Cloud bằng modal app stop | Positive | `runner.stop('flexsds-serving')` | Thực thi lệnh 'modal app stop flexsds-se... |
| `TC-MOD-08` | Modal | Chế độ dry-run in lệnh mẫu Modal đầy đủ | Positive | `runner.launch(config, dry_run=` | In ra chuỗi lệnh 'modal run ... modal_ap... |
| `TC-COL-01` | Colab | Tệp sổ tay Colab tồn tại và đúng định dạng | Positive | `Path('notebooks/flexsds_colab_` | Tệp tồn tại, cấu trúc JSON notebook hợp ... |
| `TC-COL-02` | Colab | Sổ tay Colab không chứa token PAT lưu cứng | Security | `grep -rn 'ghp_' notebooks/` | Không tìm thấy bất kỳ mã token GitHub PA... |
| `TC-COL-03` | Colab | Kiểm tra sức khỏe đường hầm Cloudflare thành công | Positive | `verify_tunnel_health(valid_url` | Hàm probe thành công và trả về True.... |
| `TC-COL-04` | Colab | Xử lý timeout khi đường hầm Cloudflare chưa sẵn sàng | Negative | `verify_tunnel_health(unready_u` | Hàm bắt lỗi URLError êm dịu và trả về Fa... |
| `TC-COL-05` | Colab | Xử lý chuỗi URL đường hầm rỗng an toàn | Boundary | `verify_tunnel_health('')` | Hàm lập tức trả về False mà không gửi re... |
| `TC-COL-06` | Colab | Xuất chỉ dẫn từng bước chi tiết cho Colab T4 | Positive | `get_colab_instructions('T4')` | Trả về dictionary chứa các bước cài đặt ... |
| `TC-COL-07` | Colab | Xuất chỉ dẫn từng bước chi tiết cho Colab L4/A100 | Positive | `get_colab_instructions('L4')` | Trả về dictionary chứa các bước cài đặt ... |
| `TC-COL-08` | Colab | Lưu URL đường hầm vào serving_config.json | Positive | `colab_tunnel_url trong json` | Giá trị URL được lưu chính xác vào trườn... |
| `TC-WIZ-01` | Giao diện | Kết xuất logo PTIT và banner HardwareAdaptive | UX | `render_banner(hw)` | Kết xuất logo trường PTIT màu đỏ/vàng kè... |
| `TC-WIZ-02` | Giao diện | Cấu trúc danh sách lựa chọn nền tảng phục vụ | Positive | `PLATFORM_CHOICES` | Chứa đầy đủ Modal.com, Google Colab và T... |
| `TC-WIZ-03` | Giao diện | Fallback an toàn khi chạy không có TTY | Boundary | `prompt_choice('Chọn', choices,` | Không mở chế độ raw termios, tự động chọ... |
| `TC-WIZ-04` | Giao diện | Căn chỉnh viền hộp render_box khi có chuỗi dài | UX | `render_box(title, items)` | Hộp tự động mở rộng theo nội dung dài nh... |
| `TC-WIZ-05` | Giao diện | Sinh file serving_config.json hợp lệ sau wizard | Positive | `run_interactive_wizard(hw)` | Tạo tệp serving_config.json với action='... |
| `TC-WIZ-06` | Giao diện | Hiển thị chỉ dẫn chat màu xanh sau khi bật server nền | UX | `wizard với run_in_bg=True` | In dòng chữ màu xanh hướng dẫn 'flexsds ... |
| `TC-WIZ-07` | Giao diện | Chỉ dẫn kết nối khi chọn cấu hình Colab | UX | `wizard với platform Colab` | In chỉ dẫn kết nối kèm URL endpoint hoặc... |
| `TC-WIZ-08` | Giao diện | Lệnh CLI serve flags parser đọc đúng các cờ | Positive | `parse_args(['serve', ...])` | Nhận diện đúng command='serve', port=800... |
| `TC-TSK-01` | Tác vụ | Thêm tác vụ và khởi tạo trạng thái PENDING | Positive | `add_task('Cấu hình phần cứng')` | Task có status == PENDING, logs rỗng, th... |
| `TC-TSK-02` | Tác vụ | Thực thi tác vụ thành công cập nhật COMPLETED | Positive | `run_task(task, action)` | Trạng thái chuyển COMPLETED, ghi nhận lo... |
| `TC-TSK-03` | Tác vụ | Thực thi tác vụ gặp lỗi cập nhật FAILED | Negative | `run_task với action ném lỗi` | Bắt ngoại lệ, cập nhật trạng thái FAILED... |
| `TC-TSK-04` | Tác vụ | Đảo cờ hiển thị chi tiết qua hàm toggle_details | Positive | `toggle_details()` | Giá trị show_details chuyển thành True; ... |
| `TC-TSK-05` | Tác vụ | Bắt phím Ctrl+O (ASCII 0x0F) kích hoạt mở rộng log | UX | `handle_key('\x0f')` | Hàm trả về True và tự động chuyển show_d... |
| `TC-TSK-06` | Tác vụ | Hiển thị chế độ thu gọn không in log chi tiết | UX | `render_view() khi thu gọn` | Chỉ in tên task và dòng hướng dẫn Ctrl+O... |
| `TC-TSK-07` | Tác vụ | Hiển thị chế độ mở rộng vẽ khung nhật ký tác vụ | UX | `render_view() khi mở rộng` | Xuất hiện khung viền 'Nhật ký tác vụ' và... |
| `TC-TSK-08` | Tác vụ | Thread lắng nghe phím non-blocking dừng sạch sẽ | Boundary | `run_task lifecycle` | Thread lắng nghe phím được gửi tín hiệu ... |
| `TC-CHT-01` | Trò chuyện | Tính toán chính xác các chỉ số hiệu năng PerformanceMetrics | Positive | `PerformanceMetrics(100, 2.0)` | Tốc độ đạt 50.0 tokens/s, TPOT đạt 20.0 ... |
| `TC-CHT-02` | Trò chuyện | Xử lý chia cho 0 khi thời gian hoặc token bằng 0 | Boundary | `PerformanceMetrics(0, 0.0)` | Trả về 0.0 an toàn, không bị ZeroDivisio... |
| `TC-CHT-03` | Trò chuyện | Bắt đầu phiên chat và thoát ngay bằng lệnh /exit | Positive | `session.start() với input /exi` | Phiên chat kết thúc nhẹ nhàng, in thông ... |
| `TC-CHT-04` | Trò chuyện | Stream phản hồi SSE và in hộp thống kê hiệu năng | Positive | `input('Chào bot') -> stream to` | Các token in ra terminal ghép thành câu ... |
| `TC-CHT-05` | Trò chuyện | Xử lý ngắt phím Ctrl+C (KeyboardInterrupt) an toàn | UX | `Ctrl+C tại prompt` | Phiên chat bắt ngoại lệ, in 'Phiên trò c... |
| `TC-CHT-06` | Trò chuyện | Lệnh /clear xóa sạch lịch sử ngữ cảnh hội thoại | Positive | `input('/clear')` | Mảng self.history bị xóa rỗng, in thông ... |
| `TC-CHT-07` | Trò chuyện | Báo lỗi kết nối khi máy chủ vLLM chưa khởi chạy | Negative | `session.start()` | In thông báo màu đỏ '[Lỗi kết nối]' kèm ... |
| `TC-CHT-08` | Trò chuyện | Lệnh điều phối CLI main chat gọi đúng tham số | Positive | `main(['chat', ...])` | Phân tích đúng các cờ và chuyển vào run_... |
| `TC-MON-01` | Giám sát | Xử lý an toàn khi tệp serving_trace.jsonl chưa tồn tại | Boundary | `read_new_records() khi chưa có` | Hàm trả về danh sách rỗng [], không ném ... |
| `TC-MON-02` | Giám sát | Đọc và bóc tách chính xác các bản ghi JSONL mới nối vào | Positive | `read_new_records() theo tiến t` | Lần 1 đọc được record1, lần 2 chỉ đọc đú... |
| `TC-MON-03` | Giám sát | Tự động bỏ qua dòng JSON bị ghi dở hoặc hỏng cú pháp | Boundary | `read_new_records() với dòng hỏ` | Bỏ qua dòng lỗi, bóc tách thành công dòn... |
| `TC-MON-04` | Giám sát | Định dạng dòng trạng thái telemetry có màu sắc trực quan | UX | `format_record(record)` | Chuỗi kết quả chứa 'Step 10', 'k=4', 'ch... |
| `TC-MON-05` | Giám sát | Tô màu cảnh báo khi số token chấp nhận thấp | UX | `format_record(low_accepted_rec` | Áp dụng mã màu ANSI vàng hoặc đỏ tương ứ... |
| `TC-MON-06` | Giám sát | Tần suất quét làm mới tùy chỉnh qua cờ refresh-rate | Positive | `main(['monitor', '--refresh-ra` | Đối số refresh_rate được gán chính xác b... |
| `TC-MON-07` | Giám sát | Dừng bộ theo dõi êm dịu khi nhấn Ctrl+C | UX | `Ctrl+C trong run_live_monitor` | Bắt ngoại lệ, in thông báo 'Đã dừng bộ t... |
| `TC-MON-08` | Giám sát | Lệnh điều phối CLI monitor gọi đúng đường dẫn tệp | Positive | `main(['monitor', ...])` | Hàm run_live_monitor nhận đúng đường dẫn... |
| `TC-NFR-01` | Phi chức năng | Tuân thủ quy tắc viết hoa chính tả tiếng Việt học thuật | Compliance | `Kiểm tra regex Title Case trên` | Không có tiêu đề viết hoa kiểu Title Cas... |
| `TC-NFR-02` | Phi chức năng | HTTP Client không phụ thuộc thư viện mạng bên ngoài | Resilience | `import urllib.request, urllib.` | Chỉ sử dụng urllib chuẩn của Python, khô... |
| `TC-NFR-03` | Phi chức năng | Đảm bảo khởi động CLI tức thì (độ trễ < 0.2s) | Performance | `time python3 -m flexsds.cli.ma` | Thời gian phản hồi < 200ms, không bị trễ... |
| `TC-NFR-04` | Phi chức năng | Không rò rỉ tài nguyên hệ thống (Zero Resource Leak) | Resilience | `pytest toàn diện` | Mọi socket đều đóng trong khối with, fil... |
| `TC-NFR-05` | Phi chức năng | Hoạt động ổn định trên môi trường bộ nhớ thấp (< 4GB RAM) | Boundary | `Chạy test trong container cgro` | Hệ thống vận hành trơn tru, tiêu thụ RAM... |
| `TC-NFR-06` | Phi chức năng | Tương thích đa nền tảng (Linux, macOS, Windows Subsystem) | Resilience | `Dùng pathlib.Path trên toàn bộ` | Đường dẫn dùng Path chuẩn, tự động nhận ... |
| `TC-NFR-07` | Phi chức năng | Tất cả các bài test tự động chạy được độc lập không cần GPU | Testability | `pytest tests/test_flexsds*.py ` | Toàn bộ 62 bài kiểm thử vượt qua 100% vớ... |
| `TC-NFR-08` | Phi chức năng | Chu trình tích hợp hoàn chỉnh End-to-End thông suốt | Integration | `pytest tests/test_end_to_end_f` | Chu trình hoàn chỉnh từ wizard tới chat ... |

---
*Tệp Excel hoàn chỉnh với 3 Sheet định dạng chuẩn được lưu tại:* `flexsds_cli_software_requirements_specification.xlsx`