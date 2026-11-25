# Future work: tự động hóa môi trường Google Colab qua Colab MCP và Colab API

Ngày ghi nhận: 2026-11-25.

Trạng thái: Đề xuất cho tương lai, chưa triển khai kết nối Colab MCP hoặc Colab API.

## Mục tiêu

Loại bỏ bước tải thủ công file `notebooks/flexsds_colab_server.ipynb` lên Google Colab, chọn runtime bằng tay và sao chép ngược đường dẫn Cloudflare tunnel về CLI.

Mục tiêu dài hạn là cho phép `flexsds serve --backend colab` hoặc wizard tương tác tự động khởi tạo phiên runtime trên Google Colab, cài đặt môi trường phục vụ `vLLM`/FlexSDS, khởi chạy endpoint OpenAI-compatible và trả về URL công khai ngay trong terminal.

## Phạm vi hiện tại

- Backend Google Colab trong [colab_runner.py](../../src/flexsds/server/colab_runner.py) và [wizard.py](../../src/flexsds/cli/wizard.py) hiện chỉ ghi đè cấu hình (`TARGET_MODEL`, `DRAFT_MODEL`, `DRAFT_METHOD`, `QUANTIZATION`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `ENABLE_ADAPTIVE`) vào file notebook cục bộ `notebooks/flexsds_colab_server.ipynb`.
- Người dùng phải tự mở Google Colab, upload file `.ipynb`, tự chọn loại GPU trong `Runtime -> Change runtime type`, chạy toàn bộ cell và dán URL `*.trycloudflare.com` quay lại CLI.
- Vì CLI không cấp phát phần cứng thật trên Colab, lựa chọn GPU cho Colab trong CLI hiện chỉ có tác dụng ước lượng ngân sách VRAM tiền kiểm tra và in hướng dẫn chọn runtime; nó không ảnh hưởng đến GPU thực tế mà Google Colab cấp cho phiên chạy.
- Chế độ cờ `flexsds serve --backend colab` chỉ in hướng dẫn, không tạo máy chủ hay endpoint.

## Thiết kế dự kiến

1. **Xác thực tài khoản Google Colab an toàn:** quản lý thông tin xác thực ngoài repository (`~/.flexsds/credentials` hoặc luồng OAuth cục bộ), tuyệt đối không lưu token/cookie trong file cấu hình JSON hay mã nguồn.
2. **Tầng kết nối Colab MCP / Colab API:** bổ sung adapter phía `src/flexsds/server/` gọi qua Colab MCP server hoặc Colab API để:
   - Kiểm tra danh sách runtime GPU khả dụng trên tài khoản người dùng (T4 miễn phí, L4 / A100 với gói trả phí).
   - Khởi tạo hoặc gắn vào một phiên runtime phù hợp và xác nhận GPU thực tế (`nvidia-smi` / CUDA capability) trước khi tải mô hình.
3. **Dựng môi trường không cần upload file `.ipynb`:** đẩy trực tiếp cấu hình phục vụ (target/draft model, revision, phương pháp EAGLE-1/2/3/3.1, chế độ adaptive/static, giới hạn batch/context) và chạy kịch bản thiết lập `vLLM 0.30.0` + FlexSDS ngay trên phiên Colab.
4. **Tự động nhận diện endpoint và tích hợp hậu triển khai:** thu thập URL công khai (Cloudflare tunnel hoặc proxy hợp lệ), chạy bước kiểm tra sẵn sàng `/health` -> `/v1/models` -> `/v1/completions` tương đương luồng Modal, sau đó đưa người dùng thẳng vào menu hành động (`Watch log`, `Realtime chat with deployed LLM`, `Export API config JSON`).
5. **Giữ đường dự phòng thủ công:** khi Colab MCP/API không khả dụng, bị giới hạn hạn mức hoặc người dùng không muốn cấp quyền tự động, CLI tự động rơi về quy trình sinh file notebook và hướng dẫn upload như hiện tại.

## Điều kiện kỹ thuật và giới hạn vận hành

- **Chính sách nền tảng và độ ổn định API:** cần bám sát chính sách sử dụng (Terms of Service) và giao diện chính thức/được hỗ trợ của Google Colab (hoặc Colab MCP server chính thống), tránh phụ thuộc vào các endpoint nội bộ không ổn định.
- **Vòng đời runtime và ngắt kết nối do nhàn rỗi:** phiên Colab có giới hạn thời gian chạy liên tục và có thể bị thu hồi GPU đột ngột; CLI cần phát hiện phiên chết sớm và báo rõ trạng thái thay vì treo vô hạn.
- **Kiểm chứng phần cứng thực tế:** không tin tưởng nhãn GPU người dùng chọn trước khi kết nối; với chế độ Hardware-Adaptive, vẫn phải đối chiếu tên GPU thực tế, kiến trúc SM và profile đo thực nghiệm đúng môi trường Colab (xem thêm tài liệu registry chia sẻ GPU profile).
- **An toàn thông tin đăng nhập:** không ghi token Colab, `HF_TOKEN` hoặc khóa tunnel vào log terminal, telemetry hay file `serving_config.json` xuất cho web demo.

## Lộ trình đề xuất

1. Nghiên cứu và chốt giao thức tích hợp ổn định (Colab MCP server hoặc Colab API chính thức) cùng cơ chế xác thực OAuth an toàn.
2. Xây dựng prototype tạo phiên runtime, chạy lệnh kiểm tra GPU (`detect`) và đồng bộ mã nguồn không cần upload `.ipynb`.
3. Nối luồng khởi chạy `vLLM` + kiểm tra sẵn sàng endpoint (`/v1/completions`) và tái sử dụng menu phiên làm việc (`serving_session.py`) để chat và xuất JSON cho web demo.
4. Bổ sung kiểm thử mô phỏng (offline mock) cho các tình huống hết GPU, mất kết nối runtime, hết hạn xác thực và fallback sang notebook thủ công.

## Tiêu chí hoàn thành

- Người dùng có thể chạy `flexsds serve --backend colab` hoặc wizard mà không cần tự tải file `.ipynb` lên trình duyệt.
- CLI phát hiện đúng loại GPU được cấp phát thực tế trên phiên Colab và chặn sớm nếu cấu hình mô hình không tương thích.
- Khi endpoint sẵn sàng, CLI hiển thị đủ URL công khai, hỗ trợ chat trực tiếp và xuất file cấu hình JSON tương thích web demo.
- Lỗi mạng, hết hạn mức Colab hoặc từ chối xác thực đều rơi về thông báo rõ ràng và giữ nguyên tùy chọn xuất notebook thủ công.
