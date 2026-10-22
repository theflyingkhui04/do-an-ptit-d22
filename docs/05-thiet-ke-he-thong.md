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

### 2.2. Giai đoạn phục vụ — Ra quyết định ngân sách theo thời gian thực (Adaptive Controller v0)
Tại mỗi bước sinh token của request, bộ điều khiển `AdaptiveDraftController` quyết định độ sâu nháp k trong tập {0, 2, 3, 4}:

```python
# 1. Thu thập thông số thời gian thực từ telemetry
alpha_ema = get_rolling_acceptance_rate(window=10)   # EMA alpha kết hợp warm-up unbiased
batch_size = get_current_running_requests()          # Số request đồng thời
free_vram_mb = get_available_vram()                  # VRAM vật lý còn trống (MB)
kv_cache_usage_pct = get_kv_cache_usage_percentage() # vllm:kv_cache_usage_perc (0.0 .. 100.0)

# 2. Memory Pressure Rule (Ngắt nhị phân chống OOM)
# Giám sát 2 điều kiện: KV-cache > 90.0% HOẶC free VRAM < 1500 MB (kèm cờ enable_vram_guard).
# Áp dụng máy trạng thái Cooldown & Hysteresis:
# - Kích hoạt: Ép k = 0 ngay lập tức, đếm bước cooldown.
# - Cooldown: Giữ tối thiểu 3 bước liên tiếp (memory_cooldown_steps = 3).
# - Hysteresis: Sau cooldown, chỉ khôi phục khi KV-cache giảm sâu dưới 80.0% (kv_cache_recover_pct)
#   nhằm triệt tiêu hiện tượng ping-pong dao động quanh ngưỡng.
if is_memory_pressure_active(kv_cache_usage_pct, free_vram_mb):
    return 0, "Memory pressure fallback (cooldown/hysteresis active)"

# 3. Safety Fallback Gate & Periodic Probing (Chống tăng tốc âm & giải phóng Deadlock)
# Khi chất lượng nháp kém (alpha < 0.65), chuyển về Vanilla (k = 0) để không lãng phí GPU.
# Để tránh bẫy "đóng băng alpha" (vì k = 0 thì không có token nháp để verify -> alpha không cập nhật),
# cơ chế Periodic Probing định kỳ phát động 1 bước thăm dò với k = 2 sau mỗi probe_interval bước:
if alpha_ema < 0.65:
    consecutive_fallback_steps += 1
    if consecutive_fallback_steps >= 20:
        # Periodic Probing (probe_interval = 20, probe_k = 2):
        # Thiết lập probe_interval = 20 (thay vì 5) giúp giảm tần suất thăm dò trên workload W3 khó đoán
        # từ ~33% xuống dưới 5% số bước giải mã, triệt tiêu độ trễ dư thừa mà vẫn đảm bảo thoát Deadlock.
        consecutive_fallback_steps = 0
        return 2, "Periodic probe (k=2) after 20 fallback steps"
    return 0, "Low acceptance fallback (alpha < 0.65)"

consecutive_fallback_steps = 0  # Reset bộ đếm khi alpha phục hồi >= 0.65

# 4. Tối ưu hóa tỉ số lợi ích / chi phí đo thực (Cost-Model Argmax) kết hợp Heuristic Guards
# - alpha_ema >= 0.80: duy trì k = 3 (hoặc mở rộng k = 4 nếu alpha_ema >= 0.90)
# - 0.65 <= alpha_ema < 0.80: hạ về k = 2 để giảm tải GPU
k = select_budget_argmax(alpha_ema, batch_size, cost_profile)
```

* **Tử số:** Lợi ích kỳ vọng (số token dự kiến được sinh E[tau]) tính theo tỉ lệ trúng: `E[tau] = 1 + sum_(i=1..k) alpha^i`.
* **Mẫu số:** Chi phí thời gian `C(k, B)` lấy trực tiếp từ bảng đo thực nghiệm phần cứng, không dùng hằng số lý thuyết.
* **Tích hợp tầng Serving (vLLM v0.30.0 V1 Engine):**
  Lớp `AdaptiveAsyncScheduler` (kế thừa `vllm.v1.core.sched.async_scheduler.AsyncScheduler`) được nạp trực tiếp qua cờ `--scheduler-cls`. Tại mỗi chu kỳ scheduling:
  - `AdaptiveBudgetPlanner` chạy `select_budget` cho từng request để xác định mảng `k_i`.
  - Batch K được gán bằng `max(k_i)` qua bảng tra cứu `dynamic_sd_lookup` của vLLM.
  - Từng request được cắt placeholder nháp (`spec_token_ids`) khớp đúng ngân sách `k_i` riêng biệt trước khi chuyển sang runner.

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
  "effective_alpha": 0.71,              // Tỉ lệ chấp nhận sau hiệu chỉnh warm-up
  "mean_accept_len": 3.42,              // Chiều dài chấp nhận trung bình thực tế tau (số token nháp Target chấp nhận)
  "mean_k":          3.75,              // Chiều sâu ngân sách nháp trung bình được đề xuất (mean của tree_budgets)
  "tree_budgets":    [3, 3, 4, 2, 0],   // Ngân sách k* qua từng bước (bằng chứng thích ứng)
  "vram_peak_mb":    13820,             // Đỉnh VRAM sử dụng
  "batch_size_avg":  8.0,
  "is_warmed_up":    true
}
```
* **Lưu ý quan trọng cho Luồng Benchmark:** `mean_accept_len` phản ánh số token nháp thực tế được Target Model chấp nhận (`tau`), dùng cho các công thức tính toán tốc độ chấp nhận. `mean_k` là giá trị trung bình của ngân sách đề xuất (`k`). Hai trường này đã được phân định độc lập từ bản vá M2-02.
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

## 4. Cấu trúc Module Mã nguồn Hiện thực (Cập nhật sau M2)

```text
src/
├── server/                       # Tích hợp vLLM v0.30.0 V1 Engine
│   ├── adaptive_scheduler.py     # AdaptiveAsyncScheduler: Lớp scheduler kế thừa AsyncScheduler
│   └── budget_planner.py         # AdaptiveBudgetPlanner: Chuyển quyết định per-request sang batch plan
├── proposer/                     # Module điều khiển cây nháp thích ứng (Adaptive Proposer)
│   ├── controller.py             # BudgetController & AdaptiveDraftController (Budget Rule + Memory Pressure)
│   └── adapter.py                # AdaptiveProposerAdapter: Quản lý session per-request & Giao diện A telemetry
├── pipeline/                     # Pipeline xử lý hóa đơn tiếng Việt MC-OCR (3 nodes)
│   ├── extractor.py              # Node 1: Schema normalization
│   └── validation.py             # Data contract validation
scripts/
├── launch_lane1_vanilla.sh       # Khởi động Làn 1 (cổng 8001)
├── launch_lane2_static.sh        # Khởi động Làn 2 (cổng 8002, K=4 cố định)
├── launch_lane3_adaptive.sh      # Khởi động Làn 3 (cổng 8003, Adaptive Scheduler)
├── switch_lane.sh                # Chuyển đổi an toàn giữa 3 làn trên 1 GPU T4 duy nhất
└── smoke_lane3_adaptive.py       # Kịch bản kiểm chứng 10 requests tự động so sánh output
```

---

## 5. Các Câu hỏi Kỹ thuật Chặn (Phải giải quyết dứt điểm ở mốc M0)

| Mã | Câu hỏi kỹ thuật cần kiểm chứng | Phương án dự phòng nếu không khả thi |
|---|---|---|
| **M-03** | Cơ chế `custom_class` của vLLM có cho phép thay đổi kích thước cây nháp $N$ linh hoạt theo từng bước sinh không? | Fork nhẹ vLLM để mở API chỉnh cây, hoặc chuyển về vòng lặp suy luận độc lập (chậm hơn nhưng kiểm soát hoàn toàn). |
| **M-02** | Checkpoint EAGLE-2 công bố sẵn cho Qwen2.5-7B có tải và chạy tương thích với bản lượng tử hóa 4-bit (AWQ/GPTQ) không? | Đổi mô hình đích sang phiên bản có checkpoint sẵn — kiên quyết **không tự train lại** ([QĐ-10](03-quyet-dinh-thiet-ke.md)). |
| **HW-1** | Tổ hợp vLLM + AWQ + EAGLE có chạy trơn tru trên GPU kiến trúc Turing T4 (sm_75) không (khi thiếu bf16 và FlashAttention-2)? | Chuyển sang backend attention tương thích (XFormers) hoặc đổi sang môi trường GPU L4 trên Colab. |
| **HW-2** | Dung lượng 16GB của T4 có đủ chứa đồng thời mô hình 4-bit + KV Cache + Cây nháp ở mức concurrency 32 không? | Đo đạc dung lượng thực tế, chủ động hạ trần concurrency tối đa nếu VRAM không cho phép. |
