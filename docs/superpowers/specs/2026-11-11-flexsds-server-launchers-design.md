# Thiết kế kỹ thuật phân hệ khởi chạy máy chủ phục vụ FlexSDS (Local & Modal Runners)

> **Tài liệu:** Thiết kế kỹ thuật chi tiết phân hệ phục vụ máy chủ FlexSDS (Task 6 Baseline CLI)  
> **Dự án:** Đồ án tốt nghiệp đại học — Ngành Công nghệ Thông tin, PTIT  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead), Bùi Đức Đại (Reviewer)  
> **Vị trí lưu trữ:** `docs/superpowers/specs/2026-11-11-flexsds-server-launchers-design.md`  
> **Ngày lập:** 11/11/2026  
> **Trạng thái:** Chờ phê duyệt  

---

## 1. Mục đích và bối cảnh (Intent & Why)

Sau khi hoàn tất 5 task đầu tiên của FlexSDS Baseline CLI (PR #28) và các phân hệ toán học cốt lõi của TreeAdaptive v0.2 (PR #33, PR #34, PR #35), FlexSDS đã có:
- Bộ nhận diện thương hiệu logo PTIT ANSI TrueColor và bộ quét GPU siêu tốc (`hardware.py`, `banner.py`).
- Bộ gác cổng an toàn tiền phục vụ (`compatibility.py`).
- Ma trận chi phí thực nghiệm L4/T4 và bộ giải nghiệm hòa vốn động $\alpha^*(T, B)$ (`cost_model.py`, `tree_topology.py`).
- Bộ trợ lý tương tác hỏi–đáp từng bước (`wizard.py`) xuất tệp cấu hình chuẩn `serving_config.json`.

Tuy nhiên, lệnh `flexsds serve` hiện tại mới chỉ dừng lại ở bước kiểm tra tương thích và in ra cờ `--dry-run`. Phân hệ thực thi máy chủ (`src/flexsds/server/`) vẫn chưa được lập trình. Trong khi đó, Bùi Đức Đại đang tập trung hoàn thiện module điều phối vLLM (`AsyncScheduler`) ở task M4-02-04 và kiểm chứng đối chứng 72 cấu hình ở task M4-02-06.

**Mục đích:** Xây dựng hoàn chỉnh phân hệ khởi chạy máy chủ phục vụ (`src/flexsds/server/`) hỗ trợ cả GPU cục bộ (`local_runner.py`) lẫn đám mây không máy chủ (`modal_runner.py`), đồng thời chuẩn hóa hợp đồng dữ liệu với M4-02-04. Nhờ đó, ngay khi Đại bàn giao xong scheduler, toàn bộ hệ thống FlexSDS CLI có thể khởi động và demo phục vụ tức thì.

---

## 2. Mục tiêu, phi mục tiêu và phản mục tiêu (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục tiêu (Goals)
1. **Thu hẹp phạm vi trọng tâm vào kiến trúc EAGLE:** Định vị rõ ràng FlexSDS là bộ công cụ tối ưu hóa suy luận phỏng đoán thích ứng phần cứng chuyên biệt cho **kiến trúc EAGLE** (cụ thể là đầu nháp EAGLE-3.1 với thuật toán TreeAdaptive v0.2). Loại bỏ việc dàn trải sang các mô hình nháp độc lập (như Qwen2.5-0.5B) vì dữ liệu thực nghiệm M1-M3 đã chỉ ra chi phí forward và độ trễ xác thực của mô hình độc lập là không tối ưu.
2. **Đồng bộ hóa lớp điều khiển CLI:** Nâng cấp [`src/flexsds/core/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/controller.py) sang cấu trúc cây 2D đa tầng của `TreeAdaptiveController v0.2`, đảm bảo đồng nhất với runtime [`src/proposer/tree_adaptive.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/tree_adaptive.py).
3. **Bộ gác cổng tiền tải về từ Hugging Face (Pre-download Guard):** Hỗ trợ người dùng nhập ID/link mô hình bất kỳ từ Hugging Face Hub, nhưng hệ thống sẽ đọc nhẹ metadata `config.json` (< 10 KB) để tính toán dung lượng VRAM trước. Nếu phát hiện không vừa bộ nhớ GPU hiện thời, lập tức phát cảnh báo và chặn tải để tránh lãng phí băng thông và nguy cơ OOM.
4. **Khởi chạy GPU cục bộ (`local_runner.py`):** Đọc `serving_config.json` và sinh câu lệnh `vllm serve` chuẩn xác với đầy đủ cờ suy đoán phỏng đoán EAGLE, cờ chỉ định scheduler `--scheduler-cls src.server.adaptive_scheduler.AdaptiveAsyncScheduler`, cùng các biến môi trường cấu hình `ADAPTIVE_*`. Hỗ trợ chạy nền hoặc chạy trực tiếp có theo dõi trạng thái sẵn sàng (`/health`).
5. **Khởi chạy qua Google Colab miễn phí (`colab_runner.py`):** Cung cấp đường dẫn mở sẵn tệp Colab Notebook của đề tài (kết hợp Cloudflare Tunnel / `cloudflared` miễn phí không cần tạo tài khoản). Hướng dẫn người dùng khởi chạy server trên GPU Colab T4/L4 và nhận URL tunnel công khai để kết nối vào CLI.
6. **Khởi chạy đám mây không máy chủ (`modal_runner.py`):** Tích hợp với hạ tầng Modal theo cơ chế tự cấp tự dùng (BYOC), cho phép người dùng tự nhập token cá nhân hoặc chạy phiên tạm thời (ephemeral session) với cơ chế dừng an toàn (`flexsds stop`).
7. **Đấu nối giao diện dòng lệnh (`main.py` & `wizard.py`):** Sau khi hoàn tất wizard hoặc khi gọi `flexsds serve` không có cờ `--dry-run`, hỏi xác nhận người dùng và kích hoạt runner tương ứng.
8. **Kiểm thử hợp đồng và luồng tích hợp đầu cuối:** Viết bài kiểm thử hợp đồng mô phỏng scheduler `tests/test_scheduler_tree_glue.py` và kiểm thử luồng CLI `tests/test_end_to_end_flow.py`, nâng tổng số test của FlexSDS CLI lên trên 25 tests, đạt 100% tỷ lệ pass.

### 2.2. Phi mục tiêu (Non-Goals)
- Không can thiệp hoặc sửa đổi mã nguồn nội bộ thuật toán `AsyncScheduler` đang thuộc quyền sở hữu của Đại ở M4-02-04; chỉ định nghĩa biến môi trường và hợp đồng giao tiếp mà scheduler đọc.
- Không tải trọng số mới hoặc huấn luyện lại checkpoint mô hình trong phạm vi phân hệ này.
- Không xây dựng giao diện đồ họa web UI (hạng mục này dành riêng cho Sprint 6 / M5-01).

### 2.3. Phản mục tiêu (Anti-Goals)
- Không ghim cứng (hardcode) đường dẫn tuyệt đối hoặc cổng phục vụ cố định; phải cho phép tùy biến linh hoạt qua tham số dòng lệnh hoặc biến môi trường.
- Tuyệt đối không làm sập (crash) tiến trình dòng lệnh nếu máy không có GPU hoặc chưa cài đặt `modal`/chưa đăng nhập tài khoản; phải đưa ra thông báo lỗi phòng vệ rõ ràng và hướng dẫn khắc phục.
- Không để xảy ra tiến trình vLLM chạy ngầm mất kiểm soát (zombie process) khi người dùng hủy lệnh bằng tổ hợp phím `Ctrl+C`.

---

## 3. Ràng buộc kỹ thuật (Constraints)

- **Môi trường thực thi:** Python 3.10+, tương thích chuẩn package Python định nghĩa trong `pyproject.toml`.
- **Engine phục vụ:** vLLM v0.30.0 với kiến trúc suy đoán phỏng đoán (speculative decoding) và kernel xác thực của EAGLE-3.1.
- **Nền tảng đám mây:** Modal Python SDK v1.6.3+, tận dụng Modal Image và Modal Volume đã cấu hình sẵn trong dự án.
- **Quy chuẩn mã nguồn:** Không yêu cầu GPU vật lý khi chạy bộ unit test tự động (tất cả các bước kích hoạt GPU thực tế trong bài test phải được cô lập bằng mock hoặc cơ chế `--dry-run`).
- **Quy tắc chính tả:** Tuân thủ nghiêm ngặt quy tắc viết hoa học thuật tiếng Việt, không dùng Title Case kiểu tiếng Anh trong văn bản và tài liệu.

---

## 4. Các phương án kiến trúc đã xem xét (Approaches)

### Phương án 1: Chỉ hoàn thiện `local_runner.py` và sinh lệnh Modal qua chuỗi shell
- **Mô tả:** Tập trung viết module sinh lệnh vLLM cục bộ; đối với Modal chỉ đơn thuần in ra dòng lệnh mẫu `modal run ...` để người dùng tự sao chép và dán vào terminal.
- **Ưu điểm:** Triển khai rất nhanh, không cần tương tác trực tiếp với API của Modal SDK.
- **Đánh đổi:** Trải nghiệm người dùng chưa trọn vẹn, chưa đạt đúng tiêu chuẩn nghiệm thu của Task 6 đã cam kết trong `plan.md`.

### Phương án 2: Hoàn thiện đồng bộ cả `local_runner.py` và `modal_runner.py` qua SDK (Được lựa chọn)
- **Mô tả:** Xây dựng đầy đủ cả hai module thực thi trong package `src/flexsds/server/`:
  + `local_runner.py` quản lý tiến trình vLLM cục bộ (khởi tạo, kiểm tra cổng mạng, bắt tín hiệu thoát an toàn).
  + `modal_runner.py` kết nối trực tiếp với Modal App của dự án, hỗ trợ chế độ dry-run xuất mã triển khai và chế độ kích hoạt hàm phục vụ từ xa.
- **Lý do chọn:** Đáp ứng trọn vẹn đặc tả sản phẩm `09-dac-ta-san-pham-flexsds.md`, đem lại sản phẩm đóng gói hoàn thiện sẵn sàng cho cả bảo vệ đồ án lẫn phát hành mã nguồn mở.

---

## 5. Thiết kế chi tiết các module thành phần

### 5.1. Đồng bộ và đóng gói độc lập `src/flexsds/core/controller.py`
Để đảm bảo package `flexsds` hoàn toàn độc lập, không bị lỗi `ModuleNotFoundError` khi cài đặt riêng biệt:
- Sao chép và đóng gói trực tiếp (vendor) toàn bộ logic của `TreeAdaptiveController v0.2` từ `src.proposer.tree_adaptive` vào [`src/flexsds/core/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/controller.py).
- Tích hợp đầy đủ: bộ lọc sentinel VRAM, máy trạng thái phục hồi theo nguồn kích hoạt, tách biệt số lần quan sát $\alpha$, cơ chế kẹp trần luồng (concurrency throttling), dynamic fallback gate và zero-crash fallback.
- Duy trì bí danh tương thích ngược 100% với các bài test baseline cũ.

### 5.2. Module `src/flexsds/server/local_runner.py`
Chịu trách nhiệm quản lý vòng đời khởi chạy engine vLLM trên GPU cục bộ:
- **Chuẩn hóa cú pháp lệnh CLI:** Kế thừa hàm chuẩn `experiments.modal_lanes.build_vllm_args()`. Sử dụng cờ `--speculative-config` dạng chuỗi JSON `{"method": ..., "num_speculative_tokens_per_batch_size": ...}` theo đúng chuẩn vLLM v0.30.0 để kích hoạt bảng tra cứu `dynamic_sd_lookup` của `AdaptiveAsyncScheduler`.
- **Tự động suy diễn profile phần cứng:** Ánh xạ từ `sm_architecture` của phần cứng quét được sang profile dữ liệu tương ứng (ví dụ L4 sử dụng `bfloat16`/`awq_marlin`, T4 sử dụng `float16`/`awq`) với dung lượng VRAM cấp phát an toàn.
- **Quản lý tiến trình và triệt tiêu tiến trình mồ côi (Zero-Zombie Process):**
  + Ở chế độ mặc định (tiền cảnh / foreground): Khởi tạo tiến trình bằng `subprocess.Popen(..., preexec_fn=os.setsid)` để tạo nhóm tiến trình (process group) riêng biệt. Bắt tín hiệu `SIGINT` (`Ctrl+C`) và `SIGTERM` để gửi `os.killpg(pgid, signal.SIGTERM)`, đảm bảo toàn bộ tiến trình con của vLLM (`EngineCore`, `APIServer`) đều được giải phóng hoàn toàn khỏi VRAM.
  + Ở chế độ chạy ngầm (`--background`): Ghi log vào `logs/flexsds_server.log`, ghi mã định danh tiến trình vào `.flexsds.pid` để người dùng có thể tra cứu hoặc dừng lại.

### 5.3. Module `src/flexsds/server/colab_runner.py`
Chịu trách nhiệm cung cấp giải pháp phục vụ miễn phí qua Google Colab:
- **Đường dẫn Colab Notebook chính thức:** Xuất link "Open in Colab" trỏ tới notebook mẫu `notebooks/flexsds_colab_server.ipynb` trong repository.
- **Cơ chế đường hầm công khai (Cloudflare Tunnel):** Trong notebook Colab, tích hợp tiện ích mở tunnel qua `cloudflared` (hoàn toàn miễn phí, không yêu cầu người dùng phải tạo tài khoản ngrok) để forward cổng vLLM `8000` ra internet.
- **Kết nối vào CLI:** Wizard tại máy cục bộ in ra liên kết mở Colab, sau đó nhận URL tunnel (ví dụ: `https://xxxx.trycloudflare.com/v1`) từ người dùng, thực hiện healthcheck và ghi nhận vào `serving_config.json`.
- **Dừng máy chủ:** Người dùng chỉ cần dừng cell thực thi hoặc ngắt kết nối runtime (Disconnect runtime) trên giao diện web của Colab.

### 5.4. Module `src/flexsds/server/modal_runner.py`
Chịu trách nhiệm giao tiếp với nền tảng Modal serverless GPU theo cơ chế tự cấp tự dùng (Bring Your Own Cloud / BYOC) và kiến trúc lai (Hybrid):
- **Cơ chế xác thực tài khoản cá nhân (BYOC Token Guard):**
  + Tuyệt đối không chia sẻ hoặc ghim cứng token nội bộ của nhóm nghiên cứu.
  + Kiểm tra trạng thái đăng nhập của người dùng qua cấu hình `~/.modal.toml` hoặc biến môi trường `MODAL_TOKEN_ID` và `MODAL_TOKEN_SECRET`.
  + Nếu chưa có thông tin xác thực, wizard chủ động hỏi người dùng nhập token Modal cá nhân (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) hoặc hướng dẫn chạy `modal setup`. Toàn bộ chi phí tính toán GPU được hạch toán trực tiếp vào tài khoản Modal của chính người dùng.
- **Kế thừa hạ tầng:** Tận dụng cấu hình Modal Image chuẩn vLLM v0.30.0 và cơ chế gắn ổ đĩa cache mô hình độc lập trên tài khoản của người dùng.
- **Chế độ mặc định (Phiên tạm thời / Ephemeral Session):** Kích hoạt ứng dụng qua `modal run` hoặc `modal serve`. Terminal stream log trực tiếp từ container; khi người dùng bấm `Ctrl+C` hoặc ngắt kết nối, container GPU L4 tự động được giải phóng ngay, không phát sinh chi phí ngầm.
- **Chế độ triển khai cố định (Persistent Deployment):** Kích hoạt khi người dùng truyền cờ `--deploy` hoặc lựa chọn trong wizard (`modal deploy`). Trả về URL OpenAI-compatible API công khai cố định.
- **Cơ chế dừng/thu hồi an toàn (`flexsds stop`):** Bổ sung chức năng dừng máy chủ cho cả hai backend:
  + Cục bộ: Đọc `.flexsds.pid` và kết thúc nhóm tiến trình qua tín hiệu `SIGTERM`.
  + Đám mây: Gọi lệnh hủy ứng dụng Modal (`modal app stop <app-name>` hoặc Modal SDK API tương ứng), đảm bảo người dùng luôn có thể tắt server dễ dàng, tuyệt đối không bị kẹt container chạy nền gây tốn tài nguyên.

### 5.5. Đấu nối điều phối CLI trong `src/flexsds/cli/main.py` và `wizard.py`

#### A. Cấu trúc menu lựa chọn nền tảng phục vụ (Serving Platform Selection)
Menu wizard tổ chức 3 nền tảng theo thứ tự chuẩn hóa:
```text
? Chọn nền tảng phục vụ (Serving Platform):
  > [1] Modal.com (Serverless GPU Cloud) (Default)
    [2] Google Colab (Notebook GPU miễn phí)
    [3] Tự dùng GPU của thiết bị này (Local / Container GPU)
```

#### B. Cơ chế định danh và chọn GPU cho từng nền tảng
1. **Khi chọn `[1] Modal.com` hoặc `[2] Google Colab`:**
   - Hệ thống hỏi người dùng muốn cấp phát/sử dụng loại GPU nào:
     `? Chọn loại GPU mục tiêu: [1] NVIDIA L4 (Default)  [2] Tesla T4  [3] NVIDIA A100 (40GB)  [4] NVIDIA A10G`
   - Tự động liên kết với hồ sơ chi phí tương ứng:
     + Nếu chọn `NVIDIA L4` $\to$ nạp `profile_sm89_l4.json`.
     + Nếu chọn `Tesla T4` $\to$ nạp `profile_sm75_t4.json`.
     + Nếu chọn kiến trúc khác $\to$ áp dụng cơ chế giải tích (Analytical Fallback).
   - Với Modal: Cấp phát container đúng loại GPU đã chọn.
   - Với Colab: Nhắc nhở người dùng chọn đúng Runtime tương ứng trên giao diện Colab.

2. **Khi chọn `[3] Tự dùng GPU của thiết bị này`:**
   - Hệ thống tự động kích hoạt bộ quét phần cứng `detect_hardware()` kiểm tra chip đang gắn vào máy hoặc container:
     + **Nếu máy có GPU (ví dụ chạy trực tiếp trên máy trạm có card rời, hoặc chạy CLI ngay bên trong container của Modal / máy ảo Colab):** Tự động nhận diện chính xác tên GPU (`NVIDIA L4`, `RTX 4090`, `Tesla T4`), kiến trúc `sm_xx`, tổng VRAM và VRAM khả dụng.
     + **Nếu máy không có GPU (chỉ có CPU) hoặc VRAM khả dụng < 6 GB:** Lập tức phát cảnh báo đỏ:
       `! Cảnh báo: Thiết bị này chỉ có CPU hoặc không đủ VRAM để chạy trực tiếp mô hình lớn.`
       Đồng thời khuyến nghị người dùng chuyển sang lựa chọn `[1] Modal.com` hoặc `[2] Google Colab` để tận dụng GPU đám mây.

#### C. Chuẩn hóa thông điệp kiểm tra tương thích và điều phối
- **Chuẩn hóa thông điệp kiểm tra tương thích:** Trong bảng báo cáo tương thích, thay thế thuật ngữ kỹ thuật khó hiểu bằng câu hỏi tự nhiên:
  `│ Dùng chung tokenizer  : CÓ` (nếu đồng nhất) hoặc `KHÔNG (Tự động kích hoạt --use-heterogeneous-vocab)` (nếu lệch từ vựng).
- Bổ sung tham số tùy chọn: `--port` (mặc định 8000), `--background` (chạy ngầm).
- Trong `wizard.py`: Sau khi lưu tệp `serving_config.json`, hỏi thêm câu hỏi tương tác:
  `? Bạn có muốn khởi chạy máy chủ phục vụ ngay bây giờ không? [Y/n]`
  Nếu người dùng chọn có: gọi trực tiếp runner tương ứng (`modal_runner`, `colab_runner` hoặc `local_runner`).
- Trong chế độ dòng lệnh tĩnh (`flexsds serve --target ... --draft ...`):
  Nếu có cờ `--dry-run`: in ra câu lệnh vLLM và biến môi trường cấu hình hoàn chỉnh;
  Nếu không có `--dry-run`: tự động kích hoạt máy chủ phục vụ theo backend đã chọn.

### 5.6. Hợp đồng kiểm thử giả lập M4-02-04 và luồng E2E
1. **`tests/test_scheduler_tree_glue.py`:**
   - Được thiết kế dưới dạng kiểm thử hợp đồng độc lập (Contract Test Double Harness) mà không sửa đổi trực tiếp file `src/server/` của Đại:
     + Định nghĩa lớp giả lập harness kiểm tra giao diện truyền nhận giữa `AdaptiveProposerAdapter` và hợp đồng lập lịch cây $N_{\text{batch}} = \max_i(N_i)$.
     + Xác nhận đầy đủ 13 trường dữ liệu của log Telemetry theo đúng đặc tả Interface A v0.2.
     + Ngay khi Đại merge M4-02-04, harness này sẽ được trỏ thẳng vào `src.server.adaptive_scheduler` thật.
2. **`tests/test_end_to_end_flow.py`:**
   - Kiểm thử toàn bộ luồng gọi CLI từ khởi động banner $\to$ quét phần cứng $\to$ kiểm tra tương thích $\to$ xuất `serving_config.json` $\to$ sinh lệnh khởi chạy.
   - Mô phỏng cả hai backend `local` và `modal` ở chế độ `--dry-run` mà không yêu cầu GPU vật lý và không gây nghẽn tiến trình.

---

## 6. Những phần dành cho người thực thi quyết định (What's Left to the Builder)

- Lựa chọn cổng mạng mặc định (port): đặt mặc định là 8000, tự động cảnh báo nếu cổng đã bị chiếm dụng bởi tiến trình khác.
- Cơ chế ghi log của tiến trình cục bộ: lưu nhật ký stdout/stderr của server vLLM vào tệp `logs/flexsds_server.log` khi chạy chế độ nền (background mode).
- Cơ chế kiểm tra sức khỏe máy chủ (Healthcheck): thăm dò endpoint `http://127.0.0.1:{port}/health` với chu kỳ 1 giây, thời gian chờ tối đa 120 giây trước khi báo lỗi timeout.
