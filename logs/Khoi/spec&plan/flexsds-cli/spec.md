# Đặc tả kỹ thuật FlexSDS Baseline CLI

> **Tài liệu:** `spec.md`  
> **Dự án:** Xây dựng Baseline CLI cho công cụ phục vụ suy luận phỏng đoán FlexSDS  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead), Bùi Đức Đại (Reviewer)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/spec.md`  
> **Trạng thái:** Bản phê duyệt triển khai  

---

## 1. Mục tiêu kỹ thuật của bản Baseline

Bản baseline của FlexSDS (phiên bản `0.2.0-alpha`) tập trung giải quyết các mục tiêu kỹ thuật cốt lõi sau:
1. **Đóng gói chuẩn Python package:** Độc lập, có thể cài đặt thông qua `pip install -e .` (chế độ phát triển) hoặc `pip install flexsds` (khi phát hành) và cung cấp lệnh thực thi toàn cục `flexsds`.
2. **Nhận diện thương hiệu đồ án PTIT:** Khởi động dòng lệnh hiển thị logo Học viện Công nghệ Bưu chính Viễn thông (PTIT) dạng ký tự khối Unicode kết hợp bảng màu 24-bit ANSI TrueColor.
3. **Bộ quét phần cứng không gây nghẽn (Zero-delay hardware detection):** Đọc thông số GPU (tên card, kiến trúc `sm_xx`, tổng VRAM, VRAM khả dụng) trong thời gian dưới 0.3 giây mà không cần khởi tạo nặng nề context CUDA.
4. **Bộ gác cổng kiểm tra tương thích (Compatibility guard):** Tính toán VRAM an toàn và phát hiện xung đột bảng từ vựng (vocabulary mismatch) trước khi khởi động tiến trình phục vụ.
5. **Giao diện tương tác từng bước (Interactive Wizard):** Menu hỏi–đáp thân thiện giúp người dùng thiết lập cấu hình serving mà không phải gõ dòng lệnh phức tạp.
6. **Bộ khung nạp hồ sơ phần cứng & ngưỡng động:** Kết nối với hồ sơ chi phí có sẵn của L4 (`sm_89`) và T4 (`sm_75`).

### 1.1. Cơ chế phân phối và hai chế độ cài đặt
* **Chế độ phát triển nội bộ (Developer Mode — `pip install -e .`):**
  * Áp dụng trong toàn bộ quá trình phát triển của nhóm từ Sprint 5 đến khi bảo vệ.
  * Cờ `-e` (`--editable`) liên kết trực tiếp môi trường Python tới thư mục mã nguồn `src/flexsds`. Mọi chỉnh sửa, thêm tính năng hoặc sửa lỗi sẽ phản ánh ngay lập tức vào lệnh `flexsds` trên terminal mà không cần cài đặt lại package.
* **Chế độ người dùng cuối (End-user Mode — `pip install flexsds`):**
  * Áp dụng sau mốc đóng băng mã nguồn khi thư viện được build thành wheel và phát hành lên PyPI.
  * Người dùng bên ngoài chỉ cần chạy `pip install flexsds` là tải bản phân phối hoàn chỉnh về máy.

---

## 2. Kiến trúc package và cấu trúc mã nguồn

Mã nguồn được phân tách module hóa rõ ràng, kế thừa và tích hợp các thành phần cốt lõi đã kiểm chứng từ Sprint 3 và Sprint 4:

```text
flexsds/
├── pyproject.toml                     # Định nghĩa package, metadata và CLI entrypoint flexsds
├── README.md                          # Hướng dẫn cài đặt và sử dụng nhanh
├── src/
│   └── flexsds/
│       ├── __init__.py                # Khai báo phiên bản __version__ = "0.2.0-alpha"
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py                # Điểm vào phân phối lệnh (dispatch action)
│       │   ├── banner.py              # Render logo PTIT ANSI TrueColor & tiêu đề hệ thống
│       │   └── wizard.py              # Giao diện menu tương tác hỏi-đáp từng bước
│       ├── core/
│       │   ├── __init__.py
│       │   ├── hardware.py            # Nhận diện GPU, VRAM, kiến trúc compute capability
│       │   ├── compatibility.py       # Kiểm tra tương thích Target-Draft, bộ nhớ & từ vựng
│       │   ├── cost_model.py          # Tra cứu ma trận chi phí C(k, B) và giải ngưỡng hòa vốn
│       │   └── controller.py          # Bộ điều khiển thích ứng (kế thừa AdaptiveDraftController)
│       ├── profiles/                  # Danh mục tệp hồ sơ phần cứng đo sẵn (JSON)
│       │   ├── profile_sm89_l4.json   # Hồ sơ GPU NVIDIA L4 (Ada Lovelace)
│       │   └── profile_sm75_t4.json   # Hồ sơ GPU Tesla T4 (Turing)
│       └── server/
│           ├── __init__.py
│           ├── modal_runner.py        # Module chuẩn bị cấu hình và triển khai lên Modal
│           └── local_runner.py        # Module khởi chạy lệnh vLLM trên GPU cục bộ
└── tests/
    ├── test_cli_banner.py             # Kiểm thử hiển thị banner và logo
    ├── test_hardware_detector.py      # Kiểm thử quét phần cứng và fallback an toàn
    ├── test_compatibility_guard.py    # Kiểm thử logic phát hiện OOM và lệch từ vựng
    └── test_cost_profile_loader.py    # Kiểm thử nạp hồ sơ phần cứng và giải ngưỡng động
```

---

## 3. Đặc tả chi tiết các module thành phần

### 3.1. Module `flexsds.cli.banner`
* **Trách nhiệm:** Dựng hình ảnh logo PTIT và bảng thông tin thương hiệu FlexSDS trên màn hình terminal.
* **Đặc tả hiển thị:**
  * Cột trái (Logo PTIT): Dựng từ 26 cột $\times$ 12 dòng ký tự khối `▀` và `▄`. 
  * Sử dụng dải màu RGB TrueColor: Màu đỏ PTIT `(237, 28, 36)`, màu vàng ngôi sao `(255, 242, 0)`.
  * Cột phải: Khối chữ ASCII `FlexSDS`, dòng phụ đề giải thích, đơn vị bảo trợ (Học viện Công nghệ Bưu chính Viễn thông) và các dòng trạng thái GPU được tô màu sinh động.
* **Giao diện hàm:**
  ```python
  def get_ptit_logo_lines() -> list[str]: ...
  def render_full_banner(gpu_info: dict) -> str: ...
  def print_banner(gpu_info: dict) -> None: ...
  ```

### 3.2. Module `flexsds.core.hardware`
* **Trách nhiệm:** Thu thập trạng thái vật lý của phần cứng máy chủ.
* **Đặc tả dữ liệu trả về (`HardwareInfo` dataclass):**
  * `device_name: str`: Tên card GPU (ví dụ: `"NVIDIA L4"`, `"NVIDIA GeForce RTX 4090"` hoặc `"CPU"`).
  * `sm_architecture: str`: Định danh Compute Capability (ví dụ: `"sm_89"`, `"sm_75"`, `"sm_80"`).
  * `total_vram_mb: float`: Tổng dung lượng VRAM vật lý.
  * `free_vram_mb: float`: Dung lượng VRAM thực tế đang rảnh.
  * `cuda_version: str`: Phiên bản CUDA runtime.
  * `is_cuda_available: bool`: Cờ kiểm tra tính khả dụng của GPU.
* **Yêu cầu an toàn:** Tuyệt đối không gây sập ứng dụng (crash) nếu máy không có card NVIDIA; tự động fallback về chế độ CPU một cách nhẹ nhàng.

### 3.3. Module `flexsds.core.compatibility`
* **Trách nhiệm:** Đóng vai trò lớp phòng vệ (Pre-flight Guard) ngăn chặn các lỗi cấu hình phổ biến trước khi khởi chạy engine phục vụ.
* **Các quy tắc kiểm tra:**
  1. *Quy tắc an toàn bộ nhớ (VRAM Guard):*
     * Ước tính kích thước mô hình theo trọng số (FP16: 2 byte/param, AWQ/INT4: 0.55 byte/param).
     * Ước lượng KV-cache tối thiểu cho 32 luồng đồng thời: tối thiểu 2.5 GB.
     * Cảnh báo nếu $\text{VRAM}_{\text{cần}} > \text{VRAM}_{\text{free}} \times 0.95$.
  2. *Quy tắc tương thích từ vựng (Vocab Alignment Guard):*
     * Đọc metadata của mô hình từ Hugging Face cache hoặc cấu hình tệp `config.json`.
     * So khớp `vocab_size`. Nếu kích thước từ vựng khác nhau $\to$ đánh dấu kích hoạt cờ `use_heterogeneous_vocab`.
  3. *Quy tắc loại mô hình nháp (Draft Classification):*
     * Nếu tên mô hình chứa `eagle` $\to$ phân loại `method = "eagle3"`.
     * Nếu là mô hình độc lập (như `Qwen2.5-0.5B`) $\to$ phân loại `method = "draft_model"`.
* **Giao diện hàm:**
  ```python
  def check_compatibility(
      target_model: str,
      draft_model: str,
      free_vram_mb: float,
      target_quant: str = "awq_marlin"
  ) -> CompatibilityReport: ...
  ```

### 3.4. Module `flexsds.core.cost_model`
* **Trách nhiệm:** Tự động tra cứu hồ sơ phần cứng tương ứng với GPU hiện tại và tính toán ngưỡng hòa vốn $\alpha^*(k, B)$.
* **Cơ chế tải hồ sơ:**
  * Ánh xạ mã `sm_architecture` tới tệp hồ sơ lưu sẵn trong package `flexsds/profiles/`.
  * Trả về thực thể `HardwareCostModel` sẵn sàng phục vụ bộ điều khiển thích ứng.

### 3.5. Module `flexsds.cli.wizard` & `main`
* **Trách nhiệm:** Điều phối luồng làm việc tương tác với người dùng.
* **Hỗ trợ 2 chế độ vận hành:**
  1. *Chế độ tương tác (Interactive Wizard - Mặc định):* Đặt câu hỏi lần lượt, hiển thị gợi ý (defaults), kiểm tra đầu vào trực tiếp.
  2. *Chế độ dòng lệnh tĩnh (Headless/CLI Flags):* Cho phép truyền trực tiếp các cờ `--target`, `--draft`, `--backend`, `--adaptive` phục vụ tự động hóa và CI/CD.

---

## 4. Hợp đồng dữ liệu cấu hình (`serving_config.json`)

Khi người dùng hoàn tất wizard, hệ thống sẽ lưu cấu hình đã kiểm tra vào tệp `serving_config.json` để có thể tái lập hoặc truyền cho các tiến trình nền:

```json
{
  "flexsds_version": "0.2.0-alpha",
  "generated_at": "2026-11-05T12:00:00Z",
  "hardware": {
    "device_name": "NVIDIA L4",
    "sm_architecture": "sm_89",
    "free_vram_mb": 22450.0
  },
  "serving": {
    "backend": "modal",
    "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
    "draft_method": "eagle3",
    "quantization": "awq_marlin",
    "max_model_len": 4096,
    "max_num_seqs": 32,
    "adaptive_enabled": true,
    "profile_applied": "profile_sm89_l4.json"
  }
}
```

---

## 5. Yêu cầu phi chức năng (Non-functional Requirements)

1. **Thời gian khởi động cực nhanh:** Lệnh `flexsds` từ lúc gõ tới lúc hiện xong logo PTIT và bảng thông số GPU phải dưới **500 mili-giây**.
2. **Độ tương thích terminal rộng:** Hiển thị tốt trên cả terminal Linux/macOS và Windows Terminal, tự động hạ cấp xuống ký tự ASCII đơn giản nếu terminal không hỗ trợ màu TrueColor hoặc UTF-8.
3. **Độ tin cậy và tự phục hồi:** Tuyệt đối không dừng đột ngột (crash) nếu thiếu kết nối mạng hoặc thiếu GPU; luôn đưa ra thông báo hướng dẫn dễ hiểu.
4. **Phạm vi kiểm thử:** Đạt tối thiểu 90% độ phủ mã nguồn (code coverage) trên toàn bộ các module core của CLI.
