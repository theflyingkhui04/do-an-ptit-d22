# Kế hoạch thực nghiệm

> 🟡 **Khung sườn — chốt trước M1 (04/10).** Chủ trì: TV2.
> Ký hiệu: [07-thuat-ngu.md](07-thuat-ngu.md). Câu hỏi nghiên cứu: [01 §1.3](01-gioi-thieu.md).

## 1. Nguyên tắc

1. **Ma trận kết quả được định nghĩa trước khi chạy thí nghiệm đầu tiên.** Biết trước bảng cuối cùng trông ra sao thì mới biết đang thiếu ô nào — và tránh việc chạy xong mới phát hiện thiếu một biến.
2. **Một con số speedup là vô nghĩa.** Mọi con số phải kèm: làn, workload, mức concurrency, phần cứng, số lần lặp.
3. **Số liệu thô được commit.** Biểu đồ sinh lại từ số liệu bằng script, không chỉnh tay. Quyển báo cáo phải tái lập được.

## 2. Ma trận kết quả

```
3 làn  ×  4 workload  ×  5 mức concurrency  =  60 ô  ×  ≥5 lần chạy
```

| Trục | Giá trị |
|---|---|
| **Làn** | `vanilla` · `static` (EAGLE-2 cố định) · `adaptive` |
| **Workload** | W1 Extractor (JSON, entropy thấp) · W2 Text-to-SQL (trung bình) · W3 Phân tích tự do (entropy cao) · W4 Pipeline E2E (W1→W2→W3) |
| **Concurrency** | 1 · 2 · 4 · 8 · 16 · 32 *(hạ trần nếu VRAM không cho)* |
| **Đo mỗi ô** | TTFT · TPOT · ITL(p50/p95) · E2E · α · τ · VRAM đỉnh · throughput |

Tiến độ dự án đo bằng **số ô đã có số liệu** ([04 §8](04-quy-trinh-va-ke-hoach.md)).

### Ô nào trả lời câu hỏi nào

| Câu hỏi | Đọc ở đâu |
|---|---|
| **RQ1** — adaptive có nhanh hơn static không | `adaptive` vs `static`, cột TPOT, ở concurrency thấp |
| **RQ2** — có loại được tăng tốc âm / OOM không | **Ô nào có speedup < 1.0×**, và ô nào `static` OOM còn `adaptive` sống |
| **RQ3** — workload nào hưởng lợi | So W1 / W2 / W3 → dựng bản đồ use-case |

**Ô quan trọng nhất của cả đồ án:** `static` ở **W3, concurrency cao**. Đây là nơi lý thuyết dự đoán speculative decoding **thua vanilla**. Nếu tái lập được hiện tượng đó rồi cho thấy `adaptive` không bị, thì RQ2 được trả lời — **kể cả khi RQ1 hoà**.

## 3. Kỷ luật đo

Wall-clock trên Colab/Kaggle rất nhiễu (GPU chia sẻ, throttling nhiệt, tần số biến động). Bắt buộc:

| Quy tắc | Chi tiết |
|---|---|
| **Warm-up** | Bỏ ≥3 request đầu mỗi cấu hình (nạp trọng số, dựng CUDA graph, JIT) |
| **Lặp lại** | ≥5 lần/ô. Báo cáo **trung vị + min–max**, không báo trung bình trần |
| **Cảnh báo phương sai** | Nếu (max−min)/trung vị > 15% → đánh dấu ô đó **không tin cậy**, chạy lại |
| **Ghi ngữ cảnh** | Mỗi lần chạy ghi: mã GPU, driver, phiên bản vLLM, commit hash, nhiệt độ/tần số nếu lấy được |
| **Đan xen làn** | Chạy ba làn **xen kẽ** thay vì chạy hết làn này rồi sang làn khác → trạng thái máy trôi sẽ ảnh hưởng đều cả ba |
| **Cố định sinh mẫu** | `temperature = 0`, seed cố định, cùng tập prompt cho cả ba làn |
| **Không tin số trên máy lạ** | Colab đổi GPU giữa chừng → mọi số trước đó vứt. Kiểm tra tên GPU **mỗi lần chạy** |

## 4. Giao thức xác minh lossless

Bắt buộc, dù đúng theo lý thuyết ([QĐ-11](03-quyet-dinh-thiet-ke.md)).

```
Với ≥50 prompt trải đều W1/W2/W3:
    out_vanilla  = gọi làn vanilla  (temperature = 0, seed cố định)
    out_adaptive = gọi làn adaptive (cùng prompt, cùng seed)
    khẳng định  out_vanilla == out_adaptive   (khớp từng ký tự)
```

- Chạy như một **test tự động**, không phải một lần thủ công.
- Ghi vào quyển dưới dạng một bảng: *N prompt, N khớp, 0 lệch*.
- **Nếu có lệch** thì đó là bug nghiêm trọng ở tiêu chí chấp nhận, **không phải** "sai số chấp nhận được". Dừng mọi việc khác cho tới khi sửa xong.
- Lưu ý: ở `temperature > 0` việc so khớp từng ký tự không áp dụng được (ngẫu nhiên). Khi đó chỉ kiểm tra ở `temperature = 0`, và nêu rõ giới hạn này trong quyển.

## 5. Cấu trúc lưu kết quả

```
experiments/
├── configs/              # định nghĩa workload, tập prompt (được commit)
├── runs/                 # JSONL thô, một dòng một request  (gitignore)
├── results/
│   ├── summary.csv       # bảng tổng hợp — NGUỒN SỰ THẬT DUY NHẤT (được commit)
│   └── figures/          # biểu đồ sinh từ summary.csv     (được commit)
└── analyze.py            # runs/ → summary.csv → figures/
```

Lược đồ một dòng `summary.csv`:
```
lane, workload, concurrency, gpu, n_runs, ttft_p50, tpot_p50, tpot_p95,
e2e_p50, alpha, tau, vram_peak_mb, throughput, speedup_vs_vanilla,
variance_flag, commit, timestamp
```

`.gitignore` hiện đã chặn `results/` — **phải thêm ngoại lệ** cho `experiments/results/summary.csv` và `figures/`, nếu không bảng số của đồ án sẽ không bao giờ được commit.

## 6. Biểu đồ sẽ đưa vào quyển (quyết định trước, vẽ sau)

| # | Biểu đồ | Trả lời |
|---|---|---|
| F1 | TPOT theo concurrency, 3 đường (3 làn), mỗi workload một khung | RQ1 + RQ2 |
| F2 | **Speedup vs vanilla**, có **kẻ đường 1.0×** làm ngưỡng | RQ2 — thấy ngay ô nào tăng tốc âm |
| F3 | `tree_budgets` theo thời gian trong một request pipeline | Bằng chứng bộ điều khiển thực sự thích ứng |
| F4 | VRAM đỉnh theo concurrency, 3 làn | An toàn bộ nhớ |
| F5 | α và τ theo workload | Giải thích *vì sao* W3 khó |
| F6 | Bản đồ use-case: nên/không nên bật | RQ3 — hình chốt của buổi bảo vệ |

F2 là hình quan trọng nhất. **Đường 1.0× phải luôn được vẽ** — nó biến "chúng tôi nhanh hơn" thành "chúng tôi không bao giờ chậm hơn", vốn là luận điểm mạnh hơn.

## 7. Mối đe doạ tới tính hợp lệ (viết vào quyển)

| Mối đe doạ | Xử lý |
|---|---|
| GPU chia sẻ, wall-clock nhiễu | §3; báo cáo phương sai công khai |
| Một tập dữ liệu duy nhất (MC-OCR) | Nêu rõ là giới hạn; 3 chế độ entropy giảm nhẹ |
| Một họ mô hình duy nhất | Nêu rõ; thêm mô hình thứ hai thuộc tầng COULD |
| Checkpoint EAGLE-2 không tối ưu cho tiếng Việt → α thấp | Ảnh hưởng **cả ba làn như nhau** → không làm hỏng phép so sánh. Phải nói rõ điều này |
| Điều chỉnh siêu tham số trên chính tập đánh giá | Chốt siêu tham số bộ điều khiển **trước** khi chạy sweep cuối; không tinh chỉnh sau khi nhìn kết quả |
