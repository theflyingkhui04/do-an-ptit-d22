# Thiết kế Hệ thống (System Architecture)

> **Cập nhật:** 14/09/2026.  
> **Lưu ý quan trọng:** Hai giao diện đóng băng ở **Mục 3** là bản hợp đồng kỹ thuật giúp 3 luồng công việc chạy song song mà không dẫm chân lên nhau ([xem docs/04](04-quy-trinh-va-ke-hoach.md)). Bắt buộc phải giữ nguyên vẹn 2 giao diện này khi lập trình.

---

## 1. Kiến trúc Tổng thể

Hệ thống được thiết kế theo mô hình kiến trúc phân lớp, tách biệt rõ ràng giữa tầng phục vụ suy luận (Serving Engine), bộ điều khiển thích ứng (Adaptive Proposer) và các ứng dụng tiêu thụ (Client Applications):

```text
┌──────────────────────────────────────────────────────────────────┐
│  CLIENT ỨNG DỤNG (Dify / LangGraph / Script Benchmark / Pipeline) │
└───────────────────────────┬──────────────────────────────────────┘
                            │  Giao thức HTTP chuẩn OpenAI
                            │  POST /v1/chat/completions
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  MÁY PHỤC VỤ vLLM (OpenAI-compatible Serving Engine)              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Vòng lặp giải mã (Decoding Loop)                          │  │
│  │     ├─ Mô hình đích (Target Model): Qwen2.5-7B 4-bit       │  │
│  │     └─ Bộ đề xuất nháp (Proposer) ◄── Điểm tích hợp module │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                            ▲
        ┌───────────────────┴────────────────────┐
        │  ADAPTIVE PROPOSER   (src/proposer/)   │
        │                                        │
        │  ┌─ HardwareProfiler ──────────────┐   │  Chạy 1 lần duy nhất lúc khởi động
        │  │  Đo t_draft(N), t_verify(N)     │   │
        │  │  → Bảng chi phí C(N, batch)     │   │
        │  └─────────────────────────────────┘   │
        │  ┌─ TelemetryCollector ────────────┐   │  Thu thập liên tục mỗi bước sinh
        │  │  Rolling α · VRAM trống · Batch │   │
        │  └─────────────────────────────────┘   │
        │  ┌─ BudgetController ──────────────┐   │  Ra quyết định ngân sách mỗi bước
        │  │  N* = argmax  E[τ(N)] / C(N,B)  │   │
        │  │  N* = 0  ⇒  TẮT suy đoán        │   │
        │  └─────────────────────────────────┘   │
        └────────────────────────────────────────┘
```

**Ba chế độ phục vụ (Serving Lanes):** Cùng chạy trên một máy chủ duy nhất, được lựa chọn qua cờ cấu hình lúc khởi động:
* `vanilla`: Giải mã tuần tự truyền thống (mô hình gốc tự sinh từng token).
* `static`: Giải mã suy đoán với cấu hình cây EAGLE-2 cố định.
* `adaptive`: Giải mã suy đoán với bộ điều khiển thích ứng phần cứng của nhóm.

---

## 2. Thuật toán Bộ điều khiển Thích ứng (Adaptive Controller)

### 2.1. Giai đoạn khởi động — Đo lường năng lực phần cứng (Hardware Profiling)
Trước khi mở cổng phục vụ, hệ thống chạy một lượt đo thực tế để lập bảng chi phí trên đúng chiếc GPU đang chạy.  
Với mỗi ngân sách cây $N \in \{0, 8, 16, 32, 60\}$ và mỗi mức batch $B \in \{1, 2, 4, 8, 16, 32\}$, hệ thống đo độ trễ trung vị của một bước giải mã:
$$C(N, B) = \text{Thời gian thực thi trung vị một bước (ms)}$$

* **Đặc điểm then chốt:** $N = 0$ tương ứng với việc **tắt hoàn toàn suy đoán** (quay về vanilla). Đây chính là **cơ chế sàn an toàn (safety fallback)** đảm bảo hệ thống không bao giờ chạy chậm hơn chế độ mặc định.
* **Rời rạc hóa ngân sách:** Tập giá trị của $N$ được cố định thành các mốc rời rạc để tensor kích thước ổn định, tránh việc biên dịch lại CUDA Graph liên tục làm giảm hiệu năng.

### 2.2. Giai đoạn phục vụ — Ra quyết định ngân sách theo thời gian thực
Tại mỗi bước sinh token của request:
```python
# 1. Thu thập thông số thời gian thực
alpha_hat = get_rolling_acceptance_rate(window=W)  # Tỉ lệ chấp nhận trượt
B = get_current_running_requests()                 # Số request đồng thời
free_vram = get_available_vram()                   # VRAM còn trống (MB)

# 2. Xác định trần ngân sách an toàn theo VRAM
N_max = calculate_max_safe_budget(free_vram)

# 3. Tối ưu hóa tỉ số lợi ích / chi phí đo thực
N_star = argmax(
    N <= N_max,
    expected_gain(N, alpha_hat) / C(N, B)
)

# 4. Fallback an toàn nếu không có lợi
if N_star == 0:
    skip_speculative_for_this_step()
```

* **Tử số:** Lợi ích kỳ vọng (số token dự kiến được chấp nhận $\mathbb{E}[\tau]$) tính theo tỉ lệ trúng $\hat{\alpha}$.
* **Mẫu số:** Chi phí thời gian $C(N, B)$ lấy trực tiếp từ bảng đo thực nghiệm phần cứng, không dùng hằng số lý thuyết.

---

## 3. Hai Giao diện Đóng băng (Frozen Interfaces)

> Đây là hợp đồng kỹ thuật chuẩn hóa giữa 3 luồng công việc. Nhờ 2 giao diện này, các thành viên khi thực hiện cơ chế **Ghép cặp Luân phiên 70/30** có thể tự tin tham gia code chéo, viết harness benchmark hoặc kết nối pipeline mà không lo xung đột cấu trúc.

### 3.1. Giao diện A — Giữa Luồng Core-Engine (Khôi chủ trì) và Luồng Benchmark (Đại chủ trì)
Máy chủ phơi dữ liệu telemetry của từng request qua endpoint `/metrics/request/{request_id}` hoặc đính kèm trong response payload:

```jsonc
{
  "request_id":      "req_abc123",
  "lane":            "adaptive",        // "vanilla" | "static" | "adaptive"
  "ttft_ms":         142.5,             // Thời gian tới token đầu tiên
  "tpot_ms":         18.3,              // Thời gian trung bình mỗi output token
  "itl_ms":          [19.1, 17.8, ...], // Mảng độ trễ giữa từng token (để tính p50/p95)
  "output_tokens":   256,
  "acceptance_rate": 0.71,              // Tỉ lệ chấp nhận alpha
  "mean_accept_len": 3.42,              // Chiều dài chấp nhận trung bình tau
  "tree_budgets":    [60, 60, 32, 0],   // Ngân sách N* qua từng bước (bằng chứng thích ứng)
  "vram_peak_mb":    13820,             // Đỉnh VRAM sử dụng
  "batch_size_avg":  8.0
}
```
* **Ý nghĩa:** Luồng Benchmark (Đại lead, Khôi/Dương hỗ trợ) chỉ cần đọc cấu trúc JSON này để phân tích và vẽ biểu đồ, hoàn toàn độc lập với việc can thiệp sâu vào nhân vLLM.

### 3.2. Giao diện B — Giữa Luồng Core-Engine (Khôi chủ trì) và Luồng Pipeline Ứng dụng (Dương chủ trì)
Pipeline hóa đơn giao tiếp với máy chủ thuần túy qua giao thức HTTP chuẩn OpenAI:

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0,
  "max_tokens": 1024
}
```
* **Ý nghĩa:** Chế độ phục vụ (`vanilla`, `static`, `adaptive`) được cấu hình ở cấp độ khởi động server, không nằm trong body request. Nhờ đó, code pipeline của Dương (cùng Đại/Khôi hỗ trợ) giữ nguyên vẹn 100% khi chạy thử nghiệm trên cả 3 chế độ.

**Ghi chú về Node 1 Extractor:** Node 1 nhận văn bản do tầng dữ liệu/OCR cung cấp và sinh JSON; nó không thực hiện nhận dạng ảnh. Đầu ra nội bộ được cố định bằng JSON Schema `invoice_extraction.v1` tại [`src/pipeline/schemas/invoice_extraction.schema.json`](../src/pipeline/schemas/invoice_extraction.schema.json). Schema này chỉ chuẩn hóa dữ liệu để nạp SQLite và đo workload W1 (JSON entropy thấp); nó không thêm tham số nào vào request OpenAI-compatible ở Giao diện B, và không đưa logic nghiệp vụ hóa đơn vào engine.

**Ghi chú về Node 2 Text-to-SQL:** CSDL trung gian dùng SQLite `user_version=1` theo lược đồ tại [`src/pipeline/sqlite/schema.sql`](../src/pipeline/sqlite/schema.sql). Node 2 chỉ sinh truy vấn đọc và ưu tiên ba view `v_invoice_facts`, `v_item_facts`, `v_tax_facts` để giữ SQL ngắn, ổn định, và đủ đại diện cho workload entropy trung bình. Hợp đồng liên tầng chính thức được chốt tại [`src/pipeline/README.md`](../src/pipeline/README.md).

**Backend kiểm thử cục bộ:** Node 1 và Node 2 đều gọi LLM qua cùng giao diện OpenAI-compatible. Có thể dùng Ollama + `qwen2.5:7b-instruct` để smoke-test chức năng trên máy cá nhân. Kết quả Ollama không được dùng làm số liệu hiệu năng của đồ án vì backend này không chạy ba serving lane, Adaptive Proposer hoặc telemetry đã đóng băng; số liệu chính thức vẫn phải chạy trên vLLM với mô hình đích Qwen2.5-7B-Instruct 4-bit.

---

## 4. Cấu trúc Module Mã nguồn dự kiến

```text
src/
├── server/               # Khởi tạo vLLM, quản lý chế độ phục vụ, phơi API & /metrics
├── proposer/             # Module điều khiển cây nháp thích ứng (Adaptive Proposer)
│   ├── profiler.py       # HardwareProfiler: Đo phần cứng lúc bật máy → Bảng C(N, B)
│   ├── telemetry.py      # TelemetryCollector: Theo dõi α trượt, VRAM, batch
│   ├── controller.py     # BudgetController: Giải thuật chọn ngân sách N*
│   └── adapter.py        # Lớp adapter tích hợp vào custom_class của vLLM
└── pipeline/             # Pipeline xử lý hóa đơn tiếng Việt MC-OCR (3 nodes)
```

---

## 5. Các Câu hỏi Kỹ thuật Chặn (Phải giải quyết dứt điểm ở mốc M0)

| Mã | Câu hỏi kỹ thuật cần kiểm chứng | Phương án dự phòng nếu không khả thi |
|---|---|---|
| **M-03** | Cơ chế `custom_class` của vLLM có cho phép thay đổi kích thước cây nháp $N$ linh hoạt theo từng bước sinh không? | Fork nhẹ vLLM để mở API chỉnh cây, hoặc chuyển về vòng lặp suy luận độc lập (chậm hơn nhưng kiểm soát hoàn toàn). |
| **M-02** | Checkpoint EAGLE-2 công bố sẵn cho Qwen2.5-7B có tải và chạy tương thích với bản lượng tử hóa 4-bit (AWQ/GPTQ) không? | Đổi mô hình đích sang phiên bản có checkpoint sẵn — kiên quyết **không tự train lại** ([QĐ-10](03-quyet-dinh-thiet-ke.md)). |
| **HW-1** | Tổ hợp vLLM + AWQ + EAGLE có chạy trơn tru trên GPU kiến trúc Turing T4 (sm_75) không (khi thiếu bf16 và FlashAttention-2)? | Chuyển sang backend attention tương thích (XFormers) hoặc đổi sang môi trường GPU L4 trên Colab. |
| **HW-2** | Dung lượng 16GB của T4 có đủ chứa đồng thời mô hình 4-bit + KV Cache + Cây nháp ở mức concurrency 32 không? | Đo đạc dung lượng thực tế, chủ động hạ trần concurrency tối đa nếu VRAM không cho phép. |
