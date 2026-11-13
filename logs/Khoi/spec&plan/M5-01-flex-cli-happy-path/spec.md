# Đặc tả kỹ thuật phân hệ phục vụ đa nền tảng và kiểm duyệt trước khi tải cho FlexSDS CLI

> **Tài liệu:** `spec.md`  
> **Mã định danh task:** `M5-01-FLEX-CLI`  
> **Dự án:** Xây dựng Baseline CLI và phân hệ phục vụ đa nền tảng cho FlexSDS  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead 70%), Bùi Đức Đại (Reviewer 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/M5-01-flex-cli-happy-path/spec.md`  
> **Ngày lập:** 11/11/2026  
> **Trạng thái:** Bản phê duyệt triển khai  

---

## 1. Mục đích và bối cảnh kỹ thuật (Intent & Why)

Sau khi hoàn tất việc kiểm chứng các phân hệ toán học và cấu trúc cây nháp ở Sprint 5 (M4-02-01 đến M4-02-05), công cụ dòng lệnh FlexSDS cần được hoàn thiện trọn vẹn để chuyển sang giai đoạn đóng gói sản phẩm phục vụ bảo vệ đồ án:
1. **Thu hẹp phạm vi trọng tâm vào kiến trúc EAGLE:** Dữ liệu thực nghiệm ở các mốc M1-M3 đã chứng minh rằng việc sử dụng mô hình nháp độc lập (như Qwen2.5-0.5B) dẫn đến chi phí forward cao, chi phí xác thực bùng nổ và hiệu quả tăng tốc rất thấp (thậm chí âm tốc độ). Do đó, FlexSDS chính thức khóa phạm vi cốt lõi là **giải mã suy đoán thích ứng phần cứng cho kiến trúc EAGLE** (đặc biệt là đầu nháp EAGLE-3.1 kết hợp thuật toán TreeAdaptive v0.2).
2. **Nối liền từ cấu hình đến thực thi phục vụ thực tế:** Hiện tại, lệnh `flexsds serve` mới dừng ở việc xuất cấu hình `serving_config.json`. Phân hệ thực thi máy chủ (`src/flexsds/server/`) cần được xây dựng hoàn chỉnh để có thể khởi chạy server thực sự trên 3 nền tảng:
   - `[1] Modal.com` (Đám mây GPU serverless tự cấp token cá nhân).
   - `[2] Google Colab` (Môi trường GPU miễn phí qua notebook tích hợp Cloudflare Tunnel).
   - `[3] Tự dùng GPU của thiết bị này` (Khởi chạy cục bộ trên máy trạm hoặc bên trong container có GPU).
3. **Bảo vệ tài nguyên người dùng (Pre-download Guard):** Khi người dùng nhập mô hình tùy ý từ Hugging Face Hub, hệ thống bắt buộc phải kiểm tra thông số VRAM qua metadata trước khi cho phép tải trọng số nặng về máy, tránh lãng phí băng thông và nguy cơ chắc chắn bị tràn bộ nhớ (OOM).

---

## 2. Mục tiêu, phi mục tiêu và phản mục tiêu (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục tiêu (Goals)
1. **Cơ chế gác cổng tiền tải về (Pre-download Guard):** Cho phép người dùng nhập Hugging Face ID / liên kết bất kỳ. Hệ thống tải tệp metadata `config.json` (< 10 KB) từ Hugging Face Hub trong 0.2 giây để đọc số lượng tham số, kiểu dữ liệu (`torch_dtype`) và lượng tử hóa (`quantization_config`). Nếu VRAM yêu cầu vượt quá VRAM khả dụng của GPU hiện tại (ví dụ bản FP16 15 GB trên card RTX 5060 Laptop 8 GB), hệ thống lập tức phát cảnh báo đỏ và chặn tải về, đồng thời gợi ý chuyển sang bản lượng tử AWQ 4-bit tương ứng.
2. **Đồng bộ hóa độc lập bộ điều khiển cây:** Sao chép và đóng gói trực tiếp (vendor) mã nguồn `TreeAdaptiveController v0.2` từ `src.proposer.tree_adaptive` vào [`src/flexsds/core/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/controller.py), đảm bảo package `flexsds` tự chứa hoàn toàn, không phụ thuộc import ngoài.
3. **Phân hệ khởi chạy máy chủ phục vụ (`src/flexsds/server/`):**
   - [`local_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/local_runner.py): Kế thừa cú pháp chuẩn `build_vllm_args()`, sinh cờ `--speculative-config` dạng chuỗi JSON và `--scheduler-cls` theo đúng chuẩn vLLM v0.30.0. Quản lý tiến trình qua nhóm tiến trình riêng biệt (`os.setsid`), triệt tiêu hoàn toàn tiến trình mồ côi (zero-zombie) khi bấm `Ctrl+C`.
   - [`colab_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/colab_runner.py): Xuất link mở Colab notebook 1-click, hướng dẫn chạy và nhận URL public tunnel từ Cloudflare Tunnel (`cloudflared`) để kết nối vào CLI.
   - [`modal_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/server/modal_runner.py): Vận hành theo mô hình tự cấp tài nguyên (BYOC) và mô hình lai (Hybrid): mặc định chạy phiên tạm thời (ephemeral) tự tắt khi ngắt lệnh; hỗ trợ cờ `--deploy` cho triển khai cố định.
4. **Lệnh dừng an toàn (`flexsds stop`):** Cung cấp cơ chế dừng máy chủ đồng nhất cho cả 3 nền tảng, dọn dẹp sạch tiến trình nền cục bộ và giải phóng container Modal.
5. **Giao diện thân thiện và trực quan:** Trong wizard, hiển thị câu hỏi tự nhiên `Dùng chung tokenizer: CÓ / KHÔNG (Tự động kích hoạt --use-heterogeneous-vocab)`.
6. **Bộ kiểm thử tự động toàn diện:** Viết bài kiểm thử hợp đồng `tests/test_scheduler_tree_glue.py` và kiểm thử luồng CLI `tests/test_end_to_end_flow.py`, nâng tổng số test của riêng phân hệ FlexSDS CLI lên trên 25 tests, đạt 100% tỷ lệ pass.

### 2.2. Phi mục tiêu (Non-Goals)
- Không can thiệp hoặc sửa đổi logic nội bộ của `AdaptiveAsyncScheduler` đang thuộc quyền sở hữu của Đại ở M4-02-04; chỉ cung cấp giao diện tham số và hợp đồng telemetry.
- Không huấn luyện lại mô hình hoặc tải thêm checkpoint mới ngoài cấu hình đã duyệt.
- Không làm giao diện web UI trong task này (dành riêng cho Sprint 6 / M5-01 sau).

### 2.3. Phản mục tiêu (Anti-Goals)
- Tuyệt đối không ghim cứng (hardcode) token Modal nội bộ của nhóm nghiên cứu vào mã nguồn sản phẩm.
- Không cho phép tải hàng chục gigabyte dữ liệu mô hình nếu chưa qua bước kiểm duyệt kích thước bộ nhớ VRAM.
- Tuyệt đối không để xảy ra tiến trình vLLM chạy ngầm chiếm dụng GPU (zombie process) khi người dùng thoát CLI.

---

## 3. Ràng buộc kỹ thuật (Constraints)

- **Môi trường thực thi:** Python 3.10+, PyTorch 2.4+, CUDA 12.1 đến 12.8.
- **Engine phục vụ:** vLLM v0.30.0 với kiến trúc giải mã phỏng đoán EAGLE.
- **Tương thích phần cứng:**
  + Tự động nhận diện các kiến trúc Compute Capability: `sm_75` (Tesla T4), `sm_89` (NVIDIA L4 / RTX 4000 series), `sm_100` / `sm_120` (Blackwell / RTX 5000 series).
  + Tự động cảnh báo và đề xuất chuyển sang đám mây nếu thiết bị chỉ có CPU hoặc VRAM khả dụng $< 6\text{ GB}$.
- **Chính sách không yêu cầu GPU vật lý trong CI/CD:** Toàn bộ test suite của CLI phải có khả năng chạy thành công 100% trên môi trường chỉ có CPU nhờ cơ chế mock và cờ `--dry-run`.
- **Quy tắc chính tả:** Tuân thủ nghiêm ngặt quy tắc viết hoa học thuật tiếng Việt, không dùng Title Case kiểu tiếng Anh trong văn bản và tài liệu.

---

## 4. Thiết kế chi tiết các phân hệ chức năng

### 4.1. Phân hệ kiểm duyệt trước khi tải từ Hugging Face (`src/flexsds/core/compatibility.py`)

Module `compatibility.py` được bổ sung hàm `inspect_remote_model_metadata()`:

```python
def inspect_remote_model_metadata(model_id_or_path: str) -> dict:
    """Tải tệp metadata config.json nhẹ (< 10 KB) từ Hugging Face Hub hoặc đường dẫn cục bộ.
    
    Trích xuất:
    - num_parameters: Ước lượng số lượng tham số
    - torch_dtype: float16, bfloat16, float32
    - quantization_config: awq, gptq, marlin, hoặc none
    - vocab_size: Kích thước từ vựng
    """
    ...

def pre_download_guard(target_model: str, draft_model: str, free_vram_mb: float) -> PreDownloadCheckResult:
    """Đánh giá tính khả thi về bộ nhớ TRƯỚC KHI TẢI trọng số nặng.
    
    Nếu dung lượng ước tính > 0.95 * free_vram_mb:
    - Trả về is_safe = False
    - Gợi ý mã định danh mô hình lượng tử tương ứng (ví dụ: chuyển từ FP16 sang AWQ)
    """
    ...
```

### 4.2. Phân hệ đồng bộ bộ điều khiển độc lập (`src/flexsds/core/controller.py`)

- Sao chép toàn bộ logic của `TreeAdaptiveController v0.2` từ `src.proposer.tree_adaptive`.
- Tích hợp đầy đủ:
  + Bộ lọc sentinel VRAM (`SENTINEL_VRAM_THRESHOLD_MB = 1.0e8`).
  + Máy trạng thái an toàn có truy vết nguồn kích hoạt (`_guard_trigger_source`).
  + Tách biệt bộ đếm quan sát thực nghiệm $\alpha$ và tự phục hồi khi gặp dữ liệu lỗi `NaN`/`Inf`.
  + Cơ chế kẹp trần kích thước cây theo luồng đồng thời (concurrency throttling).
  + Cơ chế dynamic fallback gate theo hàm bisection solver $\alpha^*(T, B)$ của `cost_model.py`.
  + Tầng bảo vệ bọc thép zero-crash fallback về $T_0$.

### 4.3. Phân hệ khởi chạy máy chủ cục bộ (`src/flexsds/server/local_runner.py`)

```python
class LocalServingRunner:
    """Quản lý tiến trình vLLM trên GPU cục bộ của thiết bị hoặc container."""

    def build_command(self, config: dict, port: int = 8000) -> list[str]:
        """Tạo danh sách tham số vllm serve chuẩn xác.
        
        Kế thừa từ experiments.modal_lanes.build_vllm_args:
        - Sử dụng --speculative-config dạng chuỗi JSON chứa num_speculative_tokens_per_batch_size
        - Bổ sung --scheduler-cls src.server.adaptive_scheduler.AdaptiveAsyncScheduler
        - Tự động gán --gpu-memory-utilization an toàn theo thông số phần cứng quét được
        """
        ...

    def launch(self, config: dict, port: int = 8000, background: bool = False, wait_ready: bool = True) -> int:
        """Kích hoạt tiến trình vllm với nhóm tiến trình riêng biệt (preexec_fn=os.setsid).
        
        - Nếu background=False: Stream log trực tiếp, bắt SIGINT để dọn dẹp sạch tiến trình con.
        - Nếu background=True: Ghi log vào logs/flexsds_server.log và lưu PID vào .flexsds.pid.
        """
        ...
```

### 4.4. Phân hệ khởi chạy qua Google Colab (`src/flexsds/server/colab_runner.py`)

```python
class ColabServingRunner:
    """Điều phối phục vụ miễn phí qua Google Colab Notebook."""

    COLAB_NOTEBOOK_URL = "https://colab.research.google.com/github/theflyingkhui04/do-an-ptit-d22/blob/develop/notebooks/flexsds_colab_server.ipynb"

    def print_colab_instructions(self, target_gpu: str = "T4") -> None:
        """In đường dẫn mở notebook và hướng dẫn 3 bước thao tác trên Colab."""
        ...

    def verify_tunnel_url(self, public_url: str, timeout_sec: int = 30) -> bool:
        """Kiểm tra phản hồi /health từ Cloudflare Tunnel public URL do người dùng cung cấp."""
        ...
```

### 4.5. Phân hệ khởi chạy Modal Cloud (`src/flexsds/server/modal_runner.py`)

```python
class ModalServingRunner:
    """Điều phối phục vụ trên Modal Cloud GPU theo cơ chế tự cấp tài nguyên (BYOC)."""

    def check_or_prompt_credentials(self) -> dict:
        """Kiểm tra ~/.modal.toml hoặc yêu cầu người dùng nhập MODAL_TOKEN_ID và MODAL_TOKEN_SECRET."""
        ...

    def launch(self, config: dict, target_gpu: str = "L4", deploy: bool = False) -> str:
        """Khởi chạy phiên tạm thời (modal run) hoặc triển khai dài hạn (modal deploy)."""
        ...

    def stop_app(self, app_name: str = "flexsds-serve") -> bool:
        """Hủy ứng dụng Modal đang chạy để giải phóng GPU và dừng tính cước."""
        ...
```

### 4.6. Lệnh dừng an toàn (`flexsds stop`)

Trong `src/flexsds/cli/main.py`:
- Bổ sung lệnh phụ: `subparsers.add_parser("stop", help="Dừng máy chủ FlexSDS đang phục vụ")`.
- Khi gõ `flexsds stop`:
  + Kiểm tra tệp `.flexsds.pid`: Nếu có tiến trình cục bộ, gửi `SIGTERM` tới nhóm tiến trình và xóa tệp PID.
  + Kiểm tra kết nối Modal: Nếu có ứng dụng Modal đang triển khai, gọi lệnh dừng ứng dụng.
  + In thông báo xác nhận: `✔ Đã dừng máy chủ FlexSDS an toàn. Toàn bộ tiến trình và GPU container đã được giải phóng.`

---

## 5. Hợp đồng dữ liệu cấu hình (`serving_config.json`)

Tệp cấu hình chuẩn xuất ra từ wizard phản ánh đầy đủ lựa chọn nền tảng và thông số kiểm duyệt:

```json
{
  "flexsds_version": "0.2.0-alpha",
  "generated_at": "2026-11-11T23:30:00Z",
  "serving": {
    "platform": "modal",
    "target_gpu": "NVIDIA L4",
    "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "draft_model": "duckhoi/qwen2.5-7b-eagle3.1-speculator",
    "draft_method": "eagle3",
    "use_heterogeneous_vocab": true,
    "shared_tokenizer": false,
    "enable_adaptive": true,
    "profile_applied": "profile_sm89_l4.json",
    "quantization": "awq_marlin",
    "max_model_len": 4096,
    "port": 8000
  },
  "hardware_detected": {
    "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "sm_architecture": "sm_89",
    "total_vram_mb": 8151.0,
    "free_vram_mb": 7650.0
  }
}
```

---

## 6. Kế hoạch kiểm thử và tiêu chuẩn nghiệm thu (Definition of Done - DoD)

| Hạng mục kiểm thử | Tệp kiểm thử | Tiêu chuẩn đạt yêu cầu (DoD) |
|---|---|---|
| **Pre-download Guard** | `tests/test_compatibility_guard.py` | Phát hiện chính xác mô hình FP16 quá khổ, kích hoạt cảnh báo đỏ và gợi ý bản AWQ mà không tải trọng số nặng |
| **TreeAdaptive Core** | `tests/test_flexsds_baseline.py` | 100% Pass các ca kiểm thử nạp profile, giải nghiệm $\alpha^*(T, B)$ và phòng vệ sentinel VRAM |
| **Scheduler Contract Mock** | `tests/test_scheduler_tree_glue.py` | Kiểm thử thành công batching đa luồng với $N_{\text{batch}} = \max_i(N_i)$ và đủ 13 trường của Telemetry Interface A v0.2 |
| **Local Runner Command** | `tests/test_local_runner.py` | Sinh lệnh `vllm serve` chuẩn xác với `--speculative-config` dạng JSON và `--scheduler-cls` |
| **End-to-End CLI Flow** | `tests/test_end_to_end_flow.py` | Kiểm thử trọn vẹn luồng gọi CLI từ khởi động banner $\to$ wizard $\to$ kiểm duyệt $\to$ sinh lệnh khởi chạy ở cả 3 nền tảng (`--dry-run`) |
| **Quy tắc cam kết mã nguồn** | Toàn bộ dự án | Tuyệt đối không tự ý chạy `git commit` khi chưa có yêu cầu từ người dùng |
