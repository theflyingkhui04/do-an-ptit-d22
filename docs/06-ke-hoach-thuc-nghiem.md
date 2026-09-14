# Kế hoạch thực nghiệm
# Kế hoạch Thực nghiệm (Experimental Methodology)

> 🟡 **Khung sườn — chốt trước M1 (04/10).** Chủ trì: TV2.
> Ký hiệu: [07-thuat-ngu.md](07-thuat-ngu.md). Câu hỏi nghiên cứu: [01 §1.3](01-gioi-thieu.md).
> **Cập nhật:** 14/09/2026.  
> **Chủ trì thực hiện:** Đại (Dev + IT — MLOps & Performance Engineer).  
> **Cột mốc áp dụng:** Hoàn thiện bộ công cụ đo trước mốc **M1 · Baseline Benchmark (04/10)**.

## 1. Nguyên tắc
---

1. **Ma trận kết quả được định nghĩa trước khi chạy thí nghiệm đầu tiên.** Biết trước bảng cuối cùng trông ra sao thì mới biết đang thiếu ô nào — và tránh việc chạy xong mới phát hiện thiếu một biến.
2. **Một con số speedup là vô nghĩa.** Mọi con số phải kèm: làn, workload, mức concurrency, phần cứng, số lần lặp.
3. **Số liệu thô được commit.** Biểu đồ sinh lại từ số liệu bằng script, không chỉnh tay. Quyển báo cáo phải tái lập được.
## 1. Nguyên tắc Đo lường

## 2. Ma trận kết quả
1. **Ma trận kết quả phải được định nghĩa trước:** Bảng số liệu tổng hợp được thiết kế cấu trúc từ trước khi bấm nút chạy thử nghiệm đầu tiên. Biết trước bảng số liệu cần gì giúp tránh tình trạng chạy xong mới nhận ra thiếu biến.
2. **Con số tăng tốc (Speedup) phải đi kèm ngữ cảnh:** Mọi số liệu báo cáo bắt buộc phải kèm theo: chế độ phục vụ (lane), tác vụ kiểm thử (workload), mức tải (concurrency), thông số GPU và số lần lặp lại.
3. **Số liệu thô được lưu trữ minh bạch:** File tổng hợp `summary.csv` là nguồn sự thật duy nhất. Mọi biểu đồ đưa vào quyển báo cáo đều được vẽ tự động bằng script từ file này, tuyệt đối không vẽ hoặc chỉnh sửa thủ công.

```
3 làn  ×  4 workload  ×  5 mức concurrency  =  60 ô  ×  ≥5 lần chạy
```
---

| Trục | Giá trị |
|---|---|
| **Làn** | `vanilla` · `static` (EAGLE-2 cố định) · `adaptive` |
| **Workload** | W1 Extractor (JSON, entropy thấp) · W2 Text-to-SQL (trung bình) · W3 Phân tích tự do (entropy cao) · W4 Pipeline E2E (W1→W2→W3) |
| **Concurrency** | 1 · 2 · 4 · 8 · 16 · 32 *(hạ trần nếu VRAM không cho)* |
| **Đo mỗi ô** | TTFT · TPOT · ITL(p50/p95) · E2E · α · τ · VRAM đỉnh · throughput |
## 2. Ma trận Kết quả Thí nghiệm

Tiến độ dự án đo bằng **số ô đã có số liệu** ([04 §8](04-quy-trinh-va-ke-hoach.md)).
Ma trận đo lường toàn diện gồm **60 ô cấu hình**:

### Ô nào trả lời câu hỏi nào
$$\text{Ma trận} = 3\text{ Chế độ phục vụ} \times 4\text{ Tác vụ kiểm thử} \times 5\text{ Mức Concurrency} = \mathbf{60\text{ ô thí nghiệm}} \times (\ge 5\text{ lần lặp})$$

| Câu hỏi | Đọc ở đâu |
| Trục thí nghiệm | Các giá trị cụ thể |
|---|---|
| **RQ1** — adaptive có nhanh hơn static không | `adaptive` vs `static`, cột TPOT, ở concurrency thấp |
| **RQ2** — có loại được tăng tốc âm / OOM không | **Ô nào có speedup < 1.0×**, và ô nào `static` OOM còn `adaptive` sống |
| **RQ3** — workload nào hưởng lợi | So W1 / W2 / W3 → dựng bản đồ use-case |
| **Chế độ phục vụ (Lanes)** | • `vanilla`: Giải mã tuần tự truyền thống.<br>• `static`: EAGLE-2 với cấu hình cây nháp tĩnh mặc định.<br>• `adaptive`: Bộ điều khiển thích ứng phần cứng của nhóm. |
| **Tác vụ kiểm thử (Workloads)** | • **W1 (JSON Extractor):** Trích xuất thông tin hóa đơn (entropy thấp, dự đoán dễ).<br>• **W2 (Text-to-SQL):** Sinh truy vấn SQL từ câu hỏi tiếng Việt (entropy trung bình).<br>• **W3 (Phân tích tự do):** Sinh báo cáo văn xuôi (entropy cao, dự đoán khó).<br>• **W4 (Pipeline E2E):** Chuỗi xử lý tuần tự toàn diện ($W1 \rightarrow W2 \rightarrow W3$). |
| **Mức tải (Concurrency)** | 1 · 2 · 4 · 8 · 16 · 32 request đồng thời *(sẽ hạ trần nếu VRAM thực tế bị giới hạn)*. |
| **Chỉ số thu thập mỗi ô** | TTFT (ms), TPOT (ms), ITL p50/p95 (ms), Độ trễ E2E (ms), Tỉ lệ chấp nhận $\alpha$, Chiều dài nháp $\tau$, Đỉnh VRAM (MB), Thông lượng (tokens/s). |

**Ô quan trọng nhất của cả đồ án:** `static` ở **W3, concurrency cao**. Đây là nơi lý thuyết dự đoán speculative decoding **thua vanilla**. Nếu tái lập được hiện tượng đó rồi cho thấy `adaptive` không bị, thì RQ2 được trả lời — **kể cả khi RQ1 hoà**.
### Ô thí nghiệm quan trọng nhất của cả đề tài
* **Điểm rơi lý thuyết:** Cấu hình `static` ở tác vụ **W3 (entropy cao)** với **mức concurrency cao (16–32)**.
* **Ý nghĩa:** Đây là nơi giải mã suy đoán tĩnh dễ bị **tăng tốc âm (chạy chậm hơn cả vanilla, speedup < 1.0×)** do chi phí xác thực cây nháp lớn vượt quá lợi ích.
* **Đóng góp của nhóm:** Tái lập được hiện tượng tăng tốc âm này trên `static`, sau đó chứng minh chế độ `adaptive` tự động co/tắt nháp để **không bao giờ chạy chậm hơn vanilla (speedup luôn $\ge 1.0\times$)**. Đây là bằng chứng đanh thép nhất để bảo vệ đồ án!

## 3. Kỷ luật đo
---

Wall-clock trên Colab/Kaggle rất nhiễu (GPU chia sẻ, throttling nhiệt, tần số biến động). Bắt buộc:
## 3. Kỷ luật Đo lường Thực nghiệm

| Quy tắc | Chi tiết |
Môi trường máy chủ đám mây chia sẻ (như Google Colab, Kaggle) thường có độ trôi lớn do điều tiết nhiệt (thermal throttling) và xung nhịp GPU biến động. Nhóm bắt buộc áp dụng các quy tắc sau:

| Quy tắc | Chi tiết thực hiện |
|---|---|
| **Warm-up** | Bỏ ≥3 request đầu mỗi cấu hình (nạp trọng số, dựng CUDA graph, JIT) |
| **Lặp lại** | ≥5 lần/ô. Báo cáo **trung vị + min–max**, không báo trung bình trần |
| **Cảnh báo phương sai** | Nếu (max−min)/trung vị > 15% → đánh dấu ô đó **không tin cậy**, chạy lại |
| **Ghi ngữ cảnh** | Mỗi lần chạy ghi: mã GPU, driver, phiên bản vLLM, commit hash, nhiệt độ/tần số nếu lấy được |
| **Đan xen làn** | Chạy ba làn **xen kẽ** thay vì chạy hết làn này rồi sang làn khác → trạng thái máy trôi sẽ ảnh hưởng đều cả ba |
| **Cố định sinh mẫu** | `temperature = 0`, seed cố định, cùng tập prompt cho cả ba làn |
| **Không tin số trên máy lạ** | Colab đổi GPU giữa chừng → mọi số trước đó vứt. Kiểm tra tên GPU **mỗi lần chạy** |
| **Chạy khởi động (Warm-up)** | Bỏ qua tối thiểu 3 request đầu tiên của mỗi cấu hình (để nạp trọng số, dựng CUDA Graph và biên dịch JIT). |
| **Lặp lại tối thiểu 5 lần** | Mỗi ô thí nghiệm chạy tối thiểu 5 lần. Báo cáo bằng **giá trị trung vị (Median)** kèm khoảng biến thiên (min–max), không dùng giá trị trung bình trần. |
| **Kiểm soát độ lệch phương sai** | Nếu $\frac{\text{Max} - \text{Min}}{\text{Median}} > 15\%$, đánh dấu kết quả ô đó là **bị nhiễu** và phải lên lịch chạy lại. |
| **Ghi nhận ngữ cảnh máy chạy** | Mỗi lần chạy ghi lại: Tên GPU, phiên bản driver CUDA, phiên bản vLLM, commit hash của mã nguồn. |
| **Chạy đan xen các chế độ** | Chạy luân phiên `vanilla` $\rightarrow$ `static` $\rightarrow$ `adaptive` thay vì chạy dồn hết một chế độ; giúp triệt tiêu sai số nếu trạng thái máy chủ bị trôi nhiệt. |
| **Cố định tham số sinh mẫu** | Đặt cứng `temperature = 0`, cố định `seed`, dùng chung tập prompt cho cả 3 chế độ. |

## 4. Giao thức xác minh lossless
---

Bắt buộc, dù đúng theo lý thuyết ([QĐ-11](03-quyet-dinh-thiet-ke.md)).
## 4. Giao thức Kiểm chứng Không giảm chất lượng (Lossless Verification)

Mặc dù giải mã suy đoán đã được chứng minh về mặt toán học là bảo toàn phân phối xác suất, nhóm vẫn xây dựng một script tự động kiểm chứng thực nghiệm để trả lời phản biện trước hội đồng:

```python
# Script kiểm tra tự động trên tập kiểm thử:
for prompt in test_prompts_dataset:  # >= 50 prompt trải đều W1, W2, W3
    out_vanilla  = call_api(lane="vanilla",  prompt=prompt, temperature=0, seed=42)
    out_adaptive = call_api(lane="adaptive", prompt=prompt, temperature=0, seed=42)
    
    assert out_vanilla == out_adaptive, f"Lệch đầu ra tại prompt: {prompt}"
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
* **Kết quả đưa vào báo cáo:** Một bảng đối chứng: *Tổng số prompt thử nghiệm: 50, Số prompt khớp 100% từng ký tự: 50, Số prompt sai lệch: 0*.
* Bảng này giúp dập tắt hoàn toàn câu hỏi: *"Tăng tốc độ sinh chữ thì chất lượng câu văn có bị giảm sút không?"*.

## 5. Cấu trúc lưu kết quả
---

```
## 5. Cấu trúc Lưu trữ Kết quả

```text
experiments/
├── configs/              # định nghĩa workload, tập prompt (được commit)
├── runs/                 # JSONL thô, một dòng một request  (gitignore)
├── configs/              # Cấu hình tập prompt, workload kiểm thử (được commit git)
├── runs/                 # File log JSONL thô của từng request (được gitignore để tránh nặng repo)
├── results/
│   ├── summary.csv       # bảng tổng hợp — NGUỒN SỰ THẬT DUY NHẤT (được commit)
│   └── figures/          # biểu đồ sinh từ summary.csv     (được commit)
└── analyze.py            # runs/ → summary.csv → figures/
│   ├── summary.csv       # File tổng hợp toàn bộ ma trận số liệu — NGUỒN SỰ THẬT DUY NHẤT (commit git)
│   └── figures/          # Toàn bộ biểu đồ sinh ra từ summary.csv (commit git)
└── analyze.py            # Script đọc runs/ → trích xuất summary.csv → vẽ biểu đồ trong figures/
```

Lược đồ một dòng `summary.csv`:
### Schema của file `summary.csv`
```csv
lane,workload,concurrency,gpu,n_runs,ttft_p50,tpot_p50,tpot_p95,e2e_p50,alpha,tau,vram_peak_mb,throughput,speedup_vs_vanilla,variance_flag,commit,timestamp
```
lane, workload, concurrency, gpu, n_runs, ttft_p50, tpot_p50, tpot_p95,
e2e_p50, alpha, tau, vram_peak_mb, throughput, speedup_vs_vanilla,
variance_flag, commit, timestamp
```

`.gitignore` hiện đã chặn `results/` — **phải thêm ngoại lệ** cho `experiments/results/summary.csv` và `figures/`, nếu không bảng số của đồ án sẽ không bao giờ được commit.
---

## 6. Biểu đồ sẽ đưa vào quyển (quyết định trước, vẽ sau)
## 6. Hệ thống Biểu đồ Đưa vào Báo cáo (Figures F1 → F6)

| # | Biểu đồ | Trả lời |
| Mã | Tên biểu đồ | Mục đích chứng minh |
|---|---|---|
| F1 | TPOT theo concurrency, 3 đường (3 làn), mỗi workload một khung | RQ1 + RQ2 |
| F2 | **Speedup vs vanilla**, có **kẻ đường 1.0×** làm ngưỡng | RQ2 — thấy ngay ô nào tăng tốc âm |
| F3 | `tree_budgets` theo thời gian trong một request pipeline | Bằng chứng bộ điều khiển thực sự thích ứng |
| F4 | VRAM đỉnh theo concurrency, 3 làn | An toàn bộ nhớ |
| F5 | α và τ theo workload | Giải thích *vì sao* W3 khó |
| F6 | Bản đồ use-case: nên/không nên bật | RQ3 — hình chốt của buổi bảo vệ |
| **F1** | **TPOT theo mức Concurrency** (3 đường tương ứng 3 chế độ, vẽ cho từng workload) | Trả lời câu hỏi: Adaptive có giảm độ trễ sinh so với Static và Vanilla không? |
| **F2** | **Hệ số tăng tốc (Speedup vs. Vanilla)** (Có kẻ sẵn **vạch chuẩn 1.0×**) | **Biểu đồ quan trọng nhất:** Chỉ ra các điểm Static bị rớt xuống dưới vạch 1.0× (tăng tốc âm) và chứng minh Adaptive luôn nằm trên vạch 1.0×. |
| **F3** | **Biến thiên ngân sách cây $N^*$ theo thời gian** trong 1 request pipeline | Bằng chứng trực quan chứng minh thuật toán Adaptive thực sự co/giãn cây linh hoạt theo từng bước sinh. |
| **F4** | **Đỉnh VRAM theo mức Concurrency** | Chứng minh an toàn bộ nhớ: Adaptive chủ động hạ cây nháp khi VRAM căng thẳng, không bị OOM ở concurrency cao. |
| **F5** | **Tỉ lệ chấp nhận $\alpha$ và chiều dài nháp $\tau$ theo Workload** | Giải thích cơ chế vì sao tác vụ W1 (JSON) đạt speedup cao nhất còn W3 (văn tự do) là tác vụ khó nhất. |
| **F6** | **Bản đồ trường hợp sử dụng (Use-case Map)** | Tổng kết hướng dẫn thực tế: Loại workload nào và cấu hình nào thì nên bật hoặc nên tắt giải mã suy đoán. |

F2 là hình quan trọng nhất. **Đường 1.0× phải luôn được vẽ** — nó biến "chúng tôi nhanh hơn" thành "chúng tôi không bao giờ chậm hơn", vốn là luận điểm mạnh hơn.
---

## 7. Mối đe doạ tới tính hợp lệ (viết vào quyển)
## 7. Các Yếu tố Đe dọa Tính hợp lệ (Threats to Validity)

| Mối đe doạ | Xử lý |
|---|---|
| GPU chia sẻ, wall-clock nhiễu | §3; báo cáo phương sai công khai |
| Một tập dữ liệu duy nhất (MC-OCR) | Nêu rõ là giới hạn; 3 chế độ entropy giảm nhẹ |
| Một họ mô hình duy nhất | Nêu rõ; thêm mô hình thứ hai thuộc tầng COULD |
| Checkpoint EAGLE-2 không tối ưu cho tiếng Việt → α thấp | Ảnh hưởng **cả ba làn như nhau** → không làm hỏng phép so sánh. Phải nói rõ điều này |
| Điều chỉnh siêu tham số trên chính tập đánh giá | Chốt siêu tham số bộ điều khiển **trước** khi chạy sweep cuối; không tinh chỉnh sau khi nhìn kết quả |
Trong quyển báo cáo, nhóm chủ động phân tích các hạn chế thực nghiệm để tăng tính khách quan khoa học:
1. **GPU đám mây bị chia sẻ tài nguyên:** Triệt tiêu bằng giao thức warm-up, chạy lặp lại 5 lần lấy trung vị và đan xen các lần chạy giữa các chế độ.
2. **Bộ dữ liệu thử nghiệm (MC-OCR):** Mặc dù dùng tập hóa đơn, nhưng đã chủ động thiết kế thành 3 mức entropy tương phản (JSON lặp khuôn $\rightarrow$ SQL ràng buộc $\rightarrow$ Văn bản tự do) để đại diện cho nhiều dạng tác vụ khác nhau.
3. **Checkpoint EAGLE-2 chưa tối ưu riêng cho tiếng Việt:** Tỉ lệ đoán trúng $\alpha$ có thể thấp hơn so với tiếng Anh, nhưng **ảnh hưởng đồng đều lên cả 3 chế độ phục vụ**, do đó không làm sai lệch bản chất so sánh tương quan giữa các phương pháp.
