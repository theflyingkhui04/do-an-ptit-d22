# Thiết kế hệ thống

> 🟡 **Khung sườn — phải chốt trong Tuần 1 (hạn 20/09).**
> Hai giao diện ở §3 là thứ cho phép ba luồng chạy song song mà không chờ nhau ([04 §3](04-quy-trinh-va-ke-hoach.md)). **Đóng băng chúng trước khi viết dòng code thứ hai.**
> Ký hiệu: [07-thuat-ngu.md](07-thuat-ngu.md).

## 1. Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────────────┐
│  KHÁCH (Dify / LangGraph / script benchmark / pipeline hoá đơn)  │
└───────────────────────────┬──────────────────────────────────────┘
                            │  HTTP  ·  /v1/chat/completions
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  MÁY PHỤC VỤ vLLM  (OpenAI-compatible)                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Vòng lặp giải mã                                          │  │
│  │     ├─ Mô hình đích: Qwen2.5-7B-Instruct 4-bit             │  │
│  │     └─ Bộ đề xuất nháp  ◄── điểm cắm của đồ án             │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                            ▲
        ┌───────────────────┴────────────────────┐
        │  ADAPTIVE PROPOSER   (src/proposer/)   │
        │                                        │
        │  ┌─ HardwareProfiler ──────────────┐   │  chạy 1 lần lúc khởi động
        │  │  đo t_draft(N), t_verify(N)     │   │
        │  │  → bảng chi phí C(N, batch)     │   │
        │  └─────────────────────────────────┘   │
        │  ┌─ Telemetry ─────────────────────┐   │  cập nhật mỗi bước
        │  │  α trượt · VRAM trống · batch   │   │
        │  └─────────────────────────────────┘   │
        │  ┌─ BudgetController ──────────────┐   │  quyết định mỗi bước
        │  │  N* = argmax  E[τ(N)] / C(N,B)  │   │
        │  │  N* = 0  ⇒  TẮT suy đoán        │   │
        │  └─────────────────────────────────┘   │
        └────────────────────────────────────────┘
```

**Ba làn phục vụ** dùng chung một máy chủ, chọn bằng cấu hình lúc khởi động: `vanilla` · `static` · `adaptive`.

## 2. Thuật toán bộ điều khiển (bản nháp — chờ M2/M3 hiệu chỉnh)

### 2.1. Lúc khởi động — đo hồ sơ

Với mỗi ngân sách cây `N ∈ {0, 8, 16, 32, 60}` và mỗi mức batch `B ∈ {1, 2, 4, 8, 16, 32}`, đo độ trễ một bước giải mã đầy đủ:
```
C(N, B)  =  thời gian trung vị của một bước  (ms)
```
`N = 0` nghĩa là **tắt suy đoán** — đây chính là **đường sàn an toàn**, và là điều kiện đủ để đảm bảo RQ2 (không bao giờ chậm hơn vanilla).

Tập `N` được **rời rạc hoá có chủ ý** để giữ kích thước tensor ổn định và không phá CUDA graph — bài học lấy từ [D-cut §3.3](paper_summary/06-dcut-liu2026.md).

### 2.2. Lúc chạy — chọn ngân sách

```
mỗi bước giải mã:
    α̂    ← trung bình trượt tỉ lệ chấp nhận (cửa sổ W bước gần nhất)
    B    ← số request đang chạy
    free ← VRAM còn trống

    N_max ← ngân sách lớn nhất còn vừa VRAM (có biên an toàn)

    N* = argmax [ N ≤ N_max ]   E[τ(N, α̂)] / C(N, B)

    nếu  N* = 0  →  bỏ qua suy đoán ở bước này
```

- **Tử số** = lợi ích kỳ vọng, ước lượng từ `α̂` đo được.
- **Mẫu số** = chi phí **đã đo trên đúng GPU này**, không phải hằng số lý thuyết.

Ba điểm cần quyết trong Tuần 1 (đánh dấu ❓ vì chưa chốt):
- ❓ Dạng hàm `E[τ(N, α̂)]` — dùng công thức giải tích cho chuỗi tuyến tính rồi hiệu chỉnh, hay dựng bảng tra từ đo thực?
- ❓ Cửa sổ trượt `W` — ngắn thì nhiễu, dài thì phản ứng chậm khi chuyển node trong pipeline.
- ❓ Chống dao động: có cần độ trễ trễ (hysteresis) để tránh `N*` nhảy qua lại mỗi bước không?

## 3. Hai giao diện đóng băng

> Đây là **hợp đồng** giữa ba luồng làm việc. Đổi thì phải báo cả nhóm và ghi vào [03](03-quyet-dinh-thiet-ke.md).

### Giao diện A — Luồng Engine ↔ Luồng Đo lường

Máy chủ phơi telemetry của mỗi request. Luồng Đo lường **chỉ đọc cấu trúc này**, không đọc nội tại của engine.

```jsonc
// GET /metrics/request/{request_id}   hoặc trường phụ trong response
{
  "request_id":      "abc123",
  "lane":            "adaptive",        // vanilla | static | adaptive
  "ttft_ms":         142.5,
  "tpot_ms":         18.3,
  "itl_ms":          [19.1, 17.8, ...], // chuỗi đầy đủ, để tính phân vị
  "output_tokens":   256,
  "acceptance_rate": 0.71,              // α
  "mean_accept_len": 3.42,              // τ
  "tree_budgets":    [60, 60, 32, 0],   // N* theo từng bước — bằng chứng thích ứng
  "vram_peak_mb":    13820,
  "batch_size_avg":  8.0
}
```

Trường `tree_budgets` là **bằng chứng trực quan** rằng bộ điều khiển thực sự thích ứng — nó sẽ thành một biểu đồ trong quyển.

### Giao diện B — Luồng Engine ↔ Luồng Pipeline

Pipeline hoá đơn **chỉ nói chuyện qua HTTP chuẩn OpenAI**, không import gì từ `src/proposer/`:

```
POST /v1/chat/completions
{ "model": "...", "messages": [...], "temperature": 0, "max_tokens": ... }
```

Chọn làn bằng **cờ khởi động máy chủ**, không bằng tham số request → pipeline chạy y hệt nhau ở cả ba làn, loại bỏ một nguồn sai lệch khi so sánh.

**Hệ quả:** luồng Pipeline có thể phát triển **từ Tuần 1** dựa trên bất kỳ máy chủ OpenAI-compat nào (kể cả Ollama), không phải chờ Adaptive Proposer.

## 4. Bản đồ module (dự kiến)

```
src/
├── server/          # dựng vLLM, chọn làn, phơi /metrics   [Giao diện A]
├── proposer/
│   ├── profiler.py       # HardwareProfiler → bảng C(N,B)
│   ├── telemetry.py      # α trượt, VRAM, batch
│   ├── controller.py     # BudgetController — thuật toán §2.2
│   └── adapter.py        # lớp nối vào custom_class của vLLM
└── pipeline/        # 3 node hoá đơn MC-OCR                [Giao diện B]
```

## 5. Câu hỏi kỹ thuật chặn (phải trả lời ở M0)

| # | Câu hỏi | Nếu câu trả lời là "không" thì sao |
|---|---|---|
| **M-03** | vLLM `custom_class` proposer có cho đổi cấu hình cây **theo từng bước** không, hay chỉ đặt được lúc khởi động? | Fork vLLM, hoặc lùi về vòng lặp suy luận tự viết (chậm hơn nhưng kiểm soát được) |
| **M-02** | Có checkpoint EAGLE-2 cho Qwen2.5-7B, chạy được với bản 4-bit không? | Đổi mô hình đích cho khớp checkpoint — **không** chuyển sang tự train ([QĐ-10](03-quyet-dinh-thiet-ke.md)) |
| **HW-1** | vLLM + AWQ/GPTQ + EAGLE có chạy trên **T4 (sm_75)** không? T4 không có bf16, không chạy được FlashAttention-2, một số nhân lượng tử hoá đòi sm_80+ | Đổi sang GPU khác (Colab L4/A100), hoặc mô hình nhỏ hơn ở FP16 |
| **HW-2** | Capacity thật: model 4-bit + KV cache + cây nháp ở batch 32 có vừa 16GB không? | Hạ trần concurrency và ghi rõ trong phạm vi |

Bốn câu này là **toàn bộ nội dung của mốc M0**. Chưa trả lời xong thì không được viết code điều khiển.
