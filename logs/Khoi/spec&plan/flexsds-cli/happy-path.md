# Kịch bản sử dụng chuẩn của công cụ dòng lệnh FlexSDS CLI (Happy Path Runbook)

> **Tài liệu:** Hướng dẫn kịch bản người dùng chuẩn (Happy Path) cho FlexSDS Baseline CLI  
> **Dự án:** Đồ án tốt nghiệp đại học — Ngành Công nghệ Thông tin, PTIT  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead), Bùi Đức Đại (Reviewer)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/flexsds-cli/happy-path.md`  
> **Ngày lập:** 11/11/2026  
> **Trạng thái:** Tài liệu vận hành chính thức  

---

## 1. Mục đích tài liệu

Tài liệu này mô tả chi tiết từng bước trải nghiệm người dùng lý tưởng (Happy Path) khi tương tác với công cụ dòng lệnh `flexsds`. Kịch bản bao quát toàn bộ luồng từ khi gõ lệnh, quét phần cứng, kiểm tra an toàn tương thích, lựa chọn 1 trong 3 nền tảng phục vụ (`Modal.com`, `Google Colab`, `Tự dùng GPU của thiết bị này`), gửi truy vấn thử nghiệm cho đến khi dừng máy chủ an toàn.

---

## 2. Kịch bản Happy Path từng bước

### Bước 0: Khởi động dòng lệnh và quét phần cứng

Người dùng mở terminal và gõ:

```bash
flexsds
```

Trong vòng chưa đầy **0.3 giây**, terminal hiển thị logo PTIT chuẩn màu ANSI TrueColor 24-bit cùng bảng trạng thái tài nguyên phần cứng hiện thời:

```text
$ flexsds

             ▄▄▀▀▀▀▀▀▀▀▄     
          ▄▀▀▀▀      ▀▀▀▄      ███████╗██╗     ███████╗██╗  ██╗███████╗██████╗ ███████╗
       ▄▀▀▀           ▀▀▀      ██╔════╝██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗██╔════╝
      ▀▀               ▀       █████╗  ██║     █████╗   ╚███╔╝ ███████╗██║  ██║███████╗
    ▄▀▀             ▄ ▀▀       ██╔══╝  ██║     ██╔══╝   ██╔██╗ ╚════██║██║  ██║╚════██║
   ▄▀▀ ▄▄▄▄ ▄▄▄▄▄▄▄▀▀▄▀▄▄▄▄    ██║     ███████╗███████╗██╔╝ ██╗███████║██████╔╝███████║
  ▄▀▀ ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀ ▀▀▀▀▀    FlexSDS: Flexible Speculative Decoding Serving Engine · v0.2.0-alpha
  ▀▀▀ ▀▀▀▀▀  ▀▀ ▄▀▀▀   ▀▀      Powered by TreeAdaptive & Hardware-Aware Profiling
  ▀▀▀ ▀▀     ▀▀▀▀▀▀▀   ▀▀      Học viện Công nghệ Bưu chính Viễn thông (PTIT)
  ▀▀▀▀▄  ▄▄▄▀▀▀▀             
    ▀▀▀▀▀▀▀▀▀                  ✔ Hardware   : NVIDIA L4 (sm_89) · VRAM: 22.4 GB / 23.0 GB
 ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀    ✔ Speculative: TreeAdaptive v0.2 (Dynamic alpha* & 2D Tree Topology)
                               ✔ Backends   : Modal.com / Google Colab / Local GPU

--- TRỢ LÝ THIẾT LẬP PHỤC VỤ (FLEXSDS SERVING WIZARD) ---
```

---

### Bước 1: Chọn chế độ hoạt động

```text
? Chọn chế độ hoạt động:
  > [1] Khởi chạy phục vụ (Launch Serving) (Default)
    [2] Chỉ quét phần cứng (Hardware Detection Only)
    [3] Đo đạc hồ sơ GPU (Quick Micro-calibration)
  Nhập lựa chọn [1-3] (Enter để chọn mặc định): [Enter]
```

Người dùng ấn `Enter` để chọn mặc định là khởi chạy phục vụ.

---

### Bước 2: Chọn mô hình đích và đầu nháp EAGLE

> **Phạm vi kỹ thuật chuẩn hóa:** FlexSDS tập trung chuyên sâu vào giải mã suy đoán thích ứng phần cứng cho **kiến trúc EAGLE** (đầu nháp chuyên biệt), loại bỏ mô hình nháp độc lập (như Qwen2.5-0.5B) vì dữ liệu thực nghiệm M1-M3 đã chứng minh chi phí forward của mô hình độc lập là không tối ưu.

Người dùng có thể chọn dùng ngay cặp mô hình chuẩn của đồ án hoặc nhập liên kết / ID từ Hugging Face:

```text
? Chọn cấu hình mô hình phục vụ:
  > [1] Cấu hình chuẩn đồ án: Qwen2.5-7B-Instruct-AWQ + EAGLE-3.1 Speculator (Khuyên dùng) (Default)
    [2] Nhập tùy chọn liên kết / ID mô hình từ Hugging Face Hub
  Nhập lựa chọn [1-2] (Enter để chọn mặc định): [Enter]
```

#### Trường hợp 2.1: Sử dụng cấu hình mặc định (Nhấn Enter)
Hệ thống tự động liên kết:
- **Target Model:** `Qwen/Qwen2.5-7B-Instruct-AWQ`
- **Draft Speculator:** `duckhoi/qwen2.5-7b-eagle3.1-speculator` (EAGLE-3.1 Head)

#### Trường hợp 2.2: Nhập liên kết / ID từ Hugging Face (Tính năng kiểm tra trước khi tải)
Nếu người dùng chọn `[2]`, họ có thể nhập ID mô hình tùy ý:
```text
? Nhập Hugging Face ID hoặc liên kết Target Model: Qwen/Qwen2.5-7B-Instruct
? Nhập Hugging Face ID hoặc liên kết EAGLE Speculator Head: duckhoi/qwen2.5-7b-eagle3.1-speculator

Đang kiểm tra thông số kỹ thuật mô hình từ Hugging Face Hub (metadata config.json < 10 KB)...
```

**Cơ chế gác cổng tiền tải về (Pre-download Guard):**
- Hệ thống chỉ tải tệp `config.json` cực nhẹ (< 10 KB) để đọc số tham số (`num_parameters`), kiểu dữ liệu (`torch_dtype`) và lượng tử hóa (`quantization_config`).
- **Nếu mô hình quá lớn so với GPU hiện tại (ví dụ bản FP16 15 GB trên GPU 8 GB):** Hệ thống lập tức chặn lại **trước khi tải hàng chục gigabyte dữ liệu**:
  ```text
  ! Cảnh báo đỏ: Mô hình Target 'Qwen/Qwen2.5-7B-Instruct' (FP16) cần tối thiểu ~14.8 GB VRAM.
  ! GPU hiện tại chỉ có 8.0 GB VRAM khả dụng -> Nguy cơ chắc chắn OOM!
  ! ĐÃ CHẶN TẢI VỀ để tiết kiệm băng thông và thời gian của bạn.

  ℹ Gợi ý giải pháp:
    - Sử dụng phiên bản lượng tử hóa 4-bit: 'Qwen/Qwen2.5-7B-Instruct-AWQ' (~4.2 GB VRAM).
    - Hoặc chọn triển khai trên [1] Modal.com / [2] Google Colab với GPU cấu hình cao hơn.

  ? Bạn có muốn đổi sang phiên bản AWQ không? [Y/n]: Y
  ✔ Đã tự động chuyển sang: Qwen/Qwen2.5-7B-Instruct-AWQ
  ```
- **Nếu thông số vừa vặn với GPU:** Hệ thống cho phép tiến hành tải về và phục vụ.

---

### Bước 3: Báo cáo an toàn tiền phục vụ (Pre-flight Guard)

Hệ thống tự động phân tích cấu hình mô hình, kiểm tra VRAM và so khớp từ vựng trong 0.05 giây:

```text
Đang kiểm tra an toàn VRAM & tương thích bộ tách từ...

┌──────────────── BÁO CÁO KIỂM TRA TƯƠNG THÍCH ────────────────┐
│ Trạng thái VRAM       : AN TOÀN (Cần: ~7.6GB / Trống: 22.4GB)
│ Phương thức suy đoán  : eagle3 (Custom EAGLE-3.1 Speculator Head)
│ Dùng chung tokenizer  : KHÔNG (Tự động kích hoạt --use-heterogeneous-vocab)
│ Khuyến nghị           : Cấu hình an toàn, sẵn sàng phục vụ không lo OOM
└───────────────────────────────────────────────────────────────┘
```

> **Ghi chú hiển thị:** Trường `Dùng chung tokenizer` sẽ hiển thị `CÓ` nếu hai mô hình hoàn toàn đồng nhất từ vựng, hoặc `KHÔNG (Tự động kích hoạt --use-heterogeneous-vocab)` nếu có sự chênh lệch để người dùng yên tâm rằng FlexSDS đã xử lý tự động.

---

### Bước 4: Lựa chọn nền tảng phục vụ (Serving Platform)

Người dùng được chọn 1 trong 3 nền tảng theo thứ tự chuẩn hóa:

```text
? Chọn nền tảng phục vụ (Serving Platform):
  > [1] Modal.com (Serverless GPU Cloud) (Default)
    [2] Google Colab (Notebook GPU miễn phí)
    [3] Tự dùng GPU của thiết bị này (Local / Container GPU)
  Nhập lựa chọn [1-3] (Enter để chọn mặc định): 
```

Dưới đây là chi tiết kịch bản cho từng lựa chọn:

---

#### Nhánh 4.1: Khi người dùng chọn `[1] Modal.com`

1. **Chọn loại GPU mục tiêu trên đám mây:**
   ```text
   ? Chọn loại GPU mục tiêu:
     > [1] NVIDIA L4 (Ada Lovelace - 24GB VRAM) (Default)
       [2] Tesla T4 (Turing - 16GB VRAM)
       [3] NVIDIA A100 (Ampere - 40GB VRAM)
       [4] NVIDIA A10G (24GB VRAM)
     Nhập lựa chọn [1-4]: 1

   ✔ Đã tự động gắn hồ sơ chi phí phần cứng: profile_sm89_l4.json
   ```

2. **Kiểm tra thông tin xác thực tài khoản cá nhân (Tự cấp tự dùng / BYOC):**
   - Nếu máy đã cấu hình sẵn (`~/.modal.toml` hoặc biến môi trường): Bỏ qua bước hỏi và hiển thị:
     `✔ Đã nhận diện thông tin xác thực Modal từ môi trường.`
   - Nếu chưa có cấu hình: Wizard chủ động yêu cầu người dùng cung cấp thông tin tài khoản Modal cá nhân:
     ```text
     ℹ Đang kiểm tra thông tin tài khoản Modal...
     ! Chưa tìm thấy thông tin xác thực Modal trên thiết bị này.
       (FlexSDS vận hành theo mô hình tự cấp tài nguyên, chi phí tính trực tiếp vào tài khoản Modal của bạn).

     ? Nhập MODAL_TOKEN_ID cá nhân của bạn: ak-xxx...
     ? Nhập MODAL_TOKEN_SECRET cá nhân của bạn: as-yyy...
     ✔ Xác thực tài khoản Modal thành công!
     ```

3. **Chọn mô hình phiên làm việc:**
   ```text
   ? Chọn chế độ vận hành:
     > [1] Phiên tạm thời (Ephemeral Session - Stream log, tự tắt khi ngắt lệnh) (Default)
       [2] Triển khai cố định (Persistent Deployment - Chạy nền dài hạn)
     Nhập lựa chọn [1-2]: 1
   ```

---

#### Nhánh 4.2: Khi người dùng chọn `[2] Google Colab`

1. **Chọn loại GPU chạy trên Colab:**
   ```text
   ? Chọn loại GPU mục tiêu trên Colab:
     > [1] Tesla T4 (Miễn phí trên Colab Free) (Default)
       [2] NVIDIA L4 (Colab Pro)
       [3] NVIDIA A100 (Colab Pro+)
     Nhập lựa chọn [1-3]: 1

   ✔ Đã tự động gắn hồ sơ chi phí phần cứng: profile_sm75_t4.json
   ```

2. **Cung cấp liên kết mở Colab Notebook:**
   ```text
   🔗 Nhấp vào liên kết để mở Colab Notebook:
      https://colab.research.google.com/github/theflyingkhui04/do-an-ptit-d22/blob/develop/notebooks/flexsds_colab_server.ipynb

   Hướng dẫn thao tác trên Colab:
   1. Chọn Runtime -> Change runtime type -> T4 GPU.
   2. Chạy lần lượt các cell trong notebook để khởi động FlexSDS server.
   3. Sao chép đường link public tunnel (dạng https://xxxx.trycloudflare.com/v1) hiển thị ở cell cuối.

   ? Dán đường dẫn public URL từ Colab của bạn vào đây: https://swift-orchid-ptit.trycloudflare.com/v1

   Đang kiểm tra kết nối tới máy chủ Colab...
   ✔ Kết nối thành công! Đã lưu endpoint vào serving_config.json.
   ```

---

#### Nhánh 4.3: Khi người dùng chọn `[3] Tự dùng GPU của thiết bị này`

Hệ thống kích hoạt ngay module quét phần cứng `detect_hardware()` để kiểm tra chip hiện thời:

- **Trường hợp thiết bị có GPU hợp lệ:**
  *(Ví dụ: chạy trực tiếp trên máy trạm có card rời, hoặc chạy lệnh này ngay bên trong container Modal / máy ảo Colab)*:
  ```text
  ✔ Đã nhận diện GPU: NVIDIA L4 (sm_89) · VRAM khả dụng: 22.4 GB / 23.0 GB
  ✔ Phần cứng tương thích hoàn toàn.
  ✔ Đã tự động gắn hồ sơ chi phí phần cứng: profile_sm89_l4.json
  ```

- **Trường hợp thiết bị không có GPU (chỉ có CPU) hoặc VRAM < 6 GB:**
  ```text
  ! Cảnh báo: Thiết bị này chỉ có CPU (hoặc VRAM < 6 GB), không khả thi để phục vụ trực tiếp mô hình lớn.
  ℹ Khuyến nghị: Bạn vui lòng chọn [1] Modal.com hoặc [2] Google Colab để chạy trên GPU đám mây!

  ? Bạn có muốn quay lại chọn nền tảng đám mây không? [Y/n]: Y
  (Hệ thống tự động quay lại Bước 4 để người dùng chọn Modal hoặc Colab).
  ```

---

### Bước 5: Kích hoạt TreeAdaptive v0.2 và xuất tệp cấu hình

```text
? Kích hoạt lớp thích ứng phần cứng TreeAdaptive (Hardware-Aware Dynamic k*)? [Y/n]
  Chọn: Y

✔ Đã lưu cấu hình phục vụ chuẩn tại: /home/duckhoi04/.../serving_config.json
```

---

### Bước 6: Khởi chạy máy chủ phục vụ (Launch Serving)

```text
? Bạn có muốn khởi chạy máy chủ phục vụ ngay bây giờ không? [Y/n]
  Chọn: Y

Đang khởi tạo máy chủ phục vụ FlexSDS...
  • Nạp mô hình Qwen2.5-7B-Instruct-AWQ và đầu nháp EAGLE-3.1...
  • Kích hoạt AdaptiveAsyncScheduler với ma trận cây TreeAdaptive v0.2...
  • Kiểm tra trạng thái sẵn sàng qua endpoint /health...

✔ Máy chủ FlexSDS đã sẵn sàng phục vụ!
  Endpoint URL  : https://canduckhoi--flexsds-serve.modal.run/v1 (hoặc http://127.0.0.1:8000/v1)
  Chuẩn giao tiếp: OpenAI Compatible (/v1/chat/completions)

ℹ Hướng dẫn vận hành:
  - Bấm Ctrl+C trên terminal này để ngắt phiên phục vụ và giải phóng GPU ngay lập tức.
  - Hoặc mở tab terminal khác và gõ: flexsds stop
```

---

### Bước 7: Người dùng gửi truy vấn thử nghiệm

Người dùng có thể gửi câu hỏi qua lệnh `curl`:

```bash
curl https://canduckhoi--flexsds-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [
      {"role": "user", "content": "Trích xuất thông tin mã số thuế và tổng tiền từ hóa đơn sau..."}
    ],
    "temperature": 0
  }'
```

Mô hình phản hồi tức thì với tốc độ sinh từ vượt trội nhờ cơ chế suy đoán thích ứng phần cứng.

---

### Bước 8: Dừng máy chủ an toàn (Teardown & Stop)

Khi hoàn tất công việc, người dùng có 2 cách dừng máy chủ:
- **Cách 1 (Ngắt trực tiếp):** Bấm `Ctrl+C` ngay trên terminal đang chạy `flexsds` $\to$ toàn bộ tiến trình con vLLM hoặc container Modal được dọn dẹp sạch sẽ trong 1 giây.
- **Cách 2 (Lệnh dừng chuyên dụng):** Gõ lệnh:
  ```bash
  flexsds stop
  ```
  Terminal in ra:
  ```text
  ✔ Đã dừng máy chủ FlexSDS an toàn. Toàn bộ tiến trình và GPU container đã được giải phóng.
  ```

---

## 3. Phụ lục: Chế độ dòng lệnh tĩnh (Headless CLI Flags)

Người dùng nâng cao hoặc các kịch bản kiểm thử tự động CI/CD có thể bỏ qua wizard bằng cách truyền cờ trực tiếp:

```bash
# Kiểm tra cấu hình và xuất lệnh mà không tốn GPU:
flexsds serve --target Qwen/Qwen2.5-7B-Instruct-AWQ --draft duckhoi/qwen2.5-7b-eagle3.1-speculator --backend modal --dry-run

# Khởi chạy thật trên Modal:
flexsds serve --target Qwen/Qwen2.5-7B-Instruct-AWQ --draft duckhoi/qwen2.5-7b-eagle3.1-speculator --backend modal

# Dừng máy chủ:
flexsds stop
```
