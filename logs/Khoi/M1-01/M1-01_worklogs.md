# Worklog Kỹ Thuật: [M1-01] Đóng Gói Server vLLM 2 Làn & Kiểm Chứng Chịu Tải Cực Hạn trên Colab T4

- **Mã công việc:** `[M1-01]` (Cột mốc M1 · Baseline Benchmark)
- **Người thực hiện:** Cấn Đức Khôi (Dev + PM — AI Systems & Quản lý dự án)
- **Thời gian thực hiện:** 27/09/2026 – 28/09/2026
- **Tài nguyên kiểm chứng:**
  - Sổ tay cấu hình tự động: [`vscode_cli.ipynb`](./vscode_cli.ipynb) (chạy trên Google Colab GPU T4 16GB)
  - Bộ điều phối làn: [`scripts/switch_lane.sh`](../../../scripts/switch_lane.sh)
  - Launcher Làn 1 (Vanilla): [`scripts/launch_lane1_vanilla.sh`](../../../scripts/launch_lane1_vanilla.sh)
  - Launcher Làn 2 (Static Draft): [`scripts/launch_lane2_static.sh`](../../../scripts/launch_lane2_static.sh)
  - Script kiểm thử cực hạn: [`scripts/stress_test_extreme.py`](../../../scripts/stress_test_extreme.py)
- **Trạng thái nghiệm thu:** Hoàn thành 100% (Đạt toàn bộ tiêu chí đóng gói server, tunnel public và kiểm thử chịu tải trần 32 luồng)

---

## 1. Mục tiêu Công việc & Đóng góp Kỹ thuật

Mục tiêu cốt lõi của task M1-01 là hoàn thiện hạ tầng phục vụ suy luận hai làn độc lập cho đồ án tốt nghiệp:
1. **Đóng gói quy trình khởi tạo máy chủ vLLM trên Google Colab:** Thiết lập kênh SSH/Tunnel thông qua VS Code CLI Remote Tunnels, cho phép các thành viên trong nhóm gọi API trực tiếp từ xa mà không cần chạy GPU cục bộ.
2. **Hiện thực hóa bộ chuyển làn vLLM (Lane Switcher):** Cung cấp cơ chế chuyển đổi sạch sẽ giữa Làn 1 (Vanilla Baseline trên cổng 8001) và Làn 2 (Static Draft Baseline trên cổng 8002), tự động thu hồi và giải phóng 100% VRAM giữa các phiên.
3. **Chuẩn hóa API tương thích OpenAI (OpenAI-compatible Endpoints):** Phơi bày endpoint công khai ra Internet thông qua Dev Tunnels với đầy đủ các chuẩn `/v1/models`, `/v1/chat/completions`, và `/health`.
4. **Kiểm thử cực hạn (Extreme Stress Test):** Đo đạc khả năng chịu tải trần (32 luồng đồng thời) để chứng minh tính ổn định tuyệt đối của hệ thống trước khi bước vào chiến dịch đo đạc M1-02.

---

## 2. Thông số Môi trường Máy chủ Thực tế

Thông số trích xuất trực tiếp từ log khởi động của `APIServer` và `EngineCore` trên Colab:

| Thành phần | Chi tiết cấu hình |
| :--- | :--- |
| **GPU phần cứng** | NVIDIA Tesla T4 (Bộ nhớ: 14.56 GiB thực tế khả dụng) |
| **Kiến trúc vi xử lý** | Turing — Compute Capability **sm_75 (7.5)** |
| **Hệ thống phần mềm** | Ubuntu Linux (Colab Runtime), Python 3.13, PyTorch 2.13.0 |
| **Phiên bản vLLM** | `vllm == 0.30.0` (Engine Core V1) |
| **Mô hình mục tiêu (Làn 1)** | `Qwen/Qwen2.5-7B-Instruct-AWQ` (W4A16 AutoAWQ Marlin) |
| **Dung lượng checkpoint** | 5.19 GiB safetensors |
| **Thời gian nạp trọng số** | Tải từ Hugging Face: 90.22 giây \| Nạp vào VRAM: 18.01 giây |
| **Tổng VRAM nạp mô hình** | 5.29 GiB VRAM |
| **Dung lượng KV Cache** | 5.79 GiB VRAM (108,496 tokens, Concurrency tối đa ở 4K context: 26.49x) |
| **Cấu hình phục vụ** | `--host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 32` |
| **Kênh kết nối công khai** | Microsoft Dev Tunnels (`https://tstrtz2l-8001.usw2.devtunnels.ms/v1`) |

---

## 3. Nhật ký Xử lý Sự cố Kỹ thuật (Systematic Debugging)

Trong quá trình đóng gói và vận hành trên Colab, nhóm đã ghi nhận và giải quyết triệt để 4 rào cản kỹ thuật:

### 3.1. Xung đột CUDA giữa PyTorch và TorchAudio
- **Hiện tượng:** Khởi chạy server thất bại với lỗi:
  ```text
  RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. 
  PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA version 12.8.
  ```
- **Nguyên nhân gốc rễ:** `vllm 0.30.0` nâng cấp PyTorch lên bản 2.13 (CUDA 13.0), xung đột với gói `torchaudio` cài sẵn của Colab. `transformers` cố tình nạp module audio dù mô hình chỉ xử lý văn bản.
- **Biện pháp xử lý:** Tự động gỡ bỏ `torchaudio` ngay trong kịch bản cài đặt:
  ```bash
  pip uninstall -y torchaudio
  ```

### 3.2. Tham số không hợp lệ trong vLLM 0.30 CLI
- **Hiện tượng:** `vllm: error: unrecognized arguments: --disable-log-requests`.
- **Nguyên nhân gốc rễ:** Cú pháp `vllm serve` trong các bản cập nhật mới đã loại bỏ cờ `--disable-log-requests`.
- **Biện pháp xử lý:** Chỉnh sửa file launcher `launch_lane1_vanilla.sh` và `launch_lane2_static.sh`, loại bỏ cờ lỗi thời, đồng thời tăng timeout `wait_for_health` từ 120s lên 300s để kịp tải trọng số 4.5GB ở lần chạy đầu tiên.

### 3.3. Lỗi ký tự xuống dòng Windows (CRLF vs LF)
- **Hiện tượng:** Shell báo lỗi khi thực thi script:
  ```text
  /usr/bin/env: ‘bash\r’: No such file or directory
  ```
- **Nguyên nhân gốc rễ:** Script được commit hoặc chỉnh sửa trên Windows mang ký tự `\r` (CRLF) khiến shebang Linux không nhận dạng được lệnh `bash`.
- **Biện pháp xử lý:**
  - Sửa nhanh: `sed -i 's/\r//g' scripts/*.sh`
  - Khắc phục vĩnh viễn: Khởi tạo file `.gitattributes` cấu hình `*.sh text eol=lf` để Git luôn tự động ép định dạng LF chuẩn cho mọi hệ điều hành.

### 3.4. Tối ưu hóa chu kỳ khởi động và lưu trữ Drive
- **Hiện tượng:** Thử nghiệm lưu `pip_cache` lên Google Drive gây nghẽn I/O nghiêm trọng (kẹt ở bước giải ma trận phụ thuộc `pip is looking at multiple versions...`).
- **Kết luận kiến trúc:** Thư mục thư viện chứa hàng vạn file nhỏ gây nghẽn FUSE trên Google Drive. Biện pháp tối ưu nhất là cài đặt trực tiếp vào ổ SSD NVMe của Colab (chỉ mất ~60s nhờ băng thông trung tâm dữ liệu Google ~50MB/s), đồng thời liên kết symlink thư mục xác thực `.vscode-cli` lên Drive để nhớ thông tin đăng nhập GitHub vĩnh viễn mà không cần xác thực lại.

---

## 4. Bằng chứng Khởi động Máy chủ vLLM (Làn 1 Online)

Trích đoạn log khởi động thành công sau 291 giây từ file `logs/vllm_lane1.log`:

```text
==> Đang khởi chạy Làn 1: Vanilla Baseline (Cổng 8001)...
    Server trên cổng 8001 đã sẵn sàng hoạt động (HTTP 200) sau 291s!

(APIServer pid=8391) INFO:     version 0.30.0
(APIServer pid=8391) INFO:     model   Qwen/Qwen2.5-7B-Instruct-AWQ
(APIServer pid=8391) INFO:     non-default args: {'host': '0.0.0.0', 'port': 8001, 'dtype': 'half', 'max_model_len': 4096, 'quantization': 'awq', 'served_model_name': ['qwen2.5-7b-vanilla'], 'gpu_memory_utilization': 0.85, 'max_num_seqs': 32}
(EngineCore pid=9270) INFO:    Using TRITON_ATTN attention backend (Turing CC 7.5 fallback)
(EngineCore pid=9270) INFO:    Time spent downloading weights: 90.22 seconds
(EngineCore pid=9270) INFO:    Loading weights took 18.01 seconds (5.29 GiB memory)
(EngineCore pid=9270) INFO:    torch.compile took 34.10 s in total
(EngineCore pid=9270) INFO:    Graph capturing finished in 7 secs, took 0.08 GiB
(EngineCore pid=9270) INFO:    Available KV cache memory: 5.79 GiB (108,496 tokens)
(APIServer pid=8391) INFO:     Starting vLLM server on http://0.0.0.0:8001
(APIServer pid=8391) INFO:     Application startup complete.
(APIServer pid=8391) INFO:     127.0.0.1:48434 - "GET /health HTTP/1.1" 200 OK
```

---

## 5. Kết quả Thử nghiệm Kiểm chứng Cực hạn (Extreme Stress Test)

Để xác nhận hệ thống có đạt tiêu chuẩn phục vụ thực tế (Production-ready) hay không, nhóm đã xây dựng script kiểm thử chịu tải kịch trần `scripts/stress_test_extreme.py` bắn từ máy cá nhân qua Dev Tunnels:

- **Quy mô kiểm thử:** **32 luồng đồng thời (Concurrency = 32)** bắn vào cùng một mili-giây.
- **Nội dung yêu cầu:** Bài luận giải thích cơ chế PagedAttention & Speculative Decoding (~384 tokens mỗi request).
- **Tổng lượng tải:** 32 x 384 = **12,288 tokens sinh ra cùng lúc**.

### Bảng Kết quả Thu được:

| Chỉ số Đo lường | Giá trị Ghi nhận Thực tế |
| :--- | :--- |
| **Tổng số request gửi đi** | 32 requests |
| **Số request thành công** | **32 / 32 (100.0%)** |
| **Số request thất bại / OOM / Timeout** | **0 (0.0%)** |
| **Tổng số token sinh ra** | **12,288 tokens** |
| **Tổng thời gian hoàn tất đợt nã tải** | **24.54 giây** |
| **Độ trễ trung bình mỗi request** | **24.42 giây** |
| **Biên độ trễ (Nhanh nhất / Chậm nhất)** | **24.25 giây / 24.50 giây** (Độ lệch cực nhỏ: 0.25s) |
| **Thông lượng toàn hệ thống (End-to-End Throughput)** | **500.70 tokens / giây** |
| **Đỉnh thông lượng sinh (Engine Log Peak)** | **633.5 tokens / giây** |
| **Tỉ lệ trúng bộ nhớ đệm tiền tố (Prefix Cache Hit)** | **95.9%** |
| **Đỉnh chiếm dụng GPU KV Cache** | **8.6%** (Dư địa dồi dào, hoàn toàn không bị tràn) |

### Đánh giá Kỹ thuật:
1. **vLLM PagedAttention hoạt động xuất sắc:** 32 request dài sinh đồng thời 12,288 tokens chỉ tiêu tốn 8.6% KV Cache, chứng minh khả năng quản lý phân mảnh bộ nhớ của PagedAttention trên Turing T4 là cực kỳ vững chắc.
2. **Thông lượng đạt đỉnh cao:** Đạt **500.70 tok/s** xuyên suốt từ máy trạm Windows qua Dev Tunnels đến máy chủ Colab, với độ trễ phân bổ đồng đều tuyệt đối (24.25s đến 24.50s cho tất cả 32 luồng).
3. **Độ ổn định tuyệt đối:** Không xảy ra bất kỳ lỗi treo cổng, tràn VRAM (OOM), hay timeout nào.

---

## 6. Kết luận & Bàn giao Cột mốc M1-02

Task `[M1-01]` đã hoàn thành 100% mục tiêu đề ra:
- [x] Đóng gói thành công máy chủ vLLM phục vụ 2 làn trên Colab T4.
- [x] Xuất bản endpoint chuẩn OpenAI (`https://<tunnel-id>.devtunnels.ms/v1`) cho toàn nhóm kiểm thử.
- [x] Giải quyết sạch sẽ các lỗi môi trường (CUDA mismatch, vLLM CLI, Windows CRLF).
- [x] Vượt qua bài kiểm tra chịu tải cực hạn 32 luồng với thông lượng 500.7 tok/s.

**Hạ tầng đã sẵn sàng bàn giao cho Task M1-02:** Đồng chí Đại và Dương có thể sử dụng trực tiếp các launcher này để tiến hành quét benchmark tự động (sweep benchmark matrix 60 ô) theo đúng kế hoạch thực nghiệm đã phê duyệt.
