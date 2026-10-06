# Biên Bản Nghiệm Thu Cổng Kiểm Soát M1: Đánh Giá Hệ Thống Suy Luận 2 Làn & Đồng Bộ Báo Cáo Thực Nghiệm

- **Mã công việc:** `[M1-04]` (Cổng kiểm soát kết thúc Sprint 2 · M1 Baseline Benchmark)
- **Thời gian họp nghiệm thu:** 20h30 – 22h00, Thứ Bảy ngày 03/10/2026 (Chốt dữ liệu thực nghiệm ngày 05/10/2026)
- **Địa điểm / Hình thức:** Google Meet trực tuyến & Phiên đối soát dữ liệu trên GitHub / Trello
- **Chủ trì cuộc họp:** Cấn Đức Khôi (Team Lead / PM / AI Systems)
- **Thành viên tham dự đầy đủ:**
  1. **Cấn Đức Khôi** (Lead 40% — Điều phối, Kiến trúc Server, Đối soát Kỷ luật Đo lường, Trực quan hóa Biểu đồ)
  2. **Bùi Đại** (Member 30% — MLOps, Nâng cấp Benchmark Harness, Quét Concurrency Sweep trên T4)
  3. **Dương Nguyễn Thế** (Member 30% — Core Engine, MC-OCR Pipeline Node 1, Chuẩn hóa Telemetry & Soạn thảo Báo cáo)
- **Tài nguyên thẩm định:**
  - Kho mã nguồn: `https://github.com/theflyingkhui04/do-an-ptit-d22`
  - Bảng số liệu chuẩn: [`experiments/results/summary.csv`](../../../experiments/results/summary.csv)
  - Biểu đồ khoa học: [`experiments/results/figures/F1_tpot_throughput_concurrency.png`](../../../experiments/results/figures/F1_tpot_throughput_concurrency.png) & [`F2_speedup_vs_concurrency.png`](../../../experiments/results/figures/F2_speedup_vs_concurrency.png)
  - Bản thảo Chương 4: [`docs/chuong-4-thuc-nghiem.md`](../../../docs/chuong-4-thuc-nghiem.md)
  - Bảng theo dõi Sprint 2 trên Trello: Board Đồ Án PTIT D22

---

## 1. Mục Đích & Chương Trình Cuộc Họp

Cuộc họp Cổng Kiểm Soát M1 (M1 Checkpoint Review Gate) là điểm dừng bắt buộc theo quy trình quản trị kỹ thuật quy định trong [`docs/04-quy-trinh-va-ke-hoach.md`](../../../docs/04-quy-trinh-va-ke-hoach.md). Mục tiêu:
1. **Nghiệm thu thực tế 3 luồng kỹ thuật độc lập** đã hoàn thành trong Sprint 2 (`M1-01`, `M1-02`, `M1-03`).
2. **Thẩm định bảng số liệu thực nghiệm 36 ô (summary.csv)** và phê duyệt các phát hiện khoa học qua biểu đồ F1, F2.
3. **Phân tích chuyên sâu hiện tượng "Tăng tốc âm" (Negative Speedup)** trên Workload W3 ở dải tải cao (Concurrency 16–32), tạo cơ sở luận chứng vững chắc cho cột mốc M2 (Adaptive Speculative Decoding).
4. **Đóng băng Giao diện A (Engine ↔ Đo lường) và Giao diện B (Engine ↔ Ứng dụng MC-OCR)**.
5. **Ký duyệt hoàn thành Sprint 2**, chuyển toàn bộ thẻ sang trạng thái **Done**, và công bố **Kế hoạch phân rã chi tiết Sprint 3 (M2)**.

---

## 2. Báo Cáo & Demo Nghiệm Thu 3 Luồng Hoạt Động Độc Lập

### 2.1. Luồng 1 (Task M1-01): Đóng gói Server vLLM 2 Làn & Chuẩn hóa Telemetry
- **Đại diện báo cáo:** Cấn Đức Khôi & Dương Nguyễn Thế.
- **Kết quả nghiệm thu:**
  - Đã đóng gói thành công 2 launcher độc lập: [`scripts/launch_lane1_vanilla.sh`](../../../scripts/launch_lane1_vanilla.sh) (port 8001) và [`scripts/launch_lane2_static.sh`](../../../scripts/launch_lane2_static.sh) (port 8002).
  - Bộ chuyển làn tự động [`scripts/switch_lane.sh`](../../../scripts/switch_lane.sh) giải phóng và thu hồi 100% VRAM giữa các phiên chạy, chống rò rỉ bộ nhớ triệt để.
  - Chuẩn hóa cấu trúc telemetry trả về qua OpenAI streaming chunks: `alpha`, `mean_accept_len` ($\tau$), `tpot_ms`, `vram_peak_mb`.
  - Smoke test 10 requests đạt 10/10 PASS; Stress test kịch trần 32 luồng đồng thời đạt thông lượng **500.70 tok/s** với 100% tỷ lệ thành công.
- **Trạng thái PR:** PR #3 (`feat/m1-01-vllm-dual-lane-server`) đã merge thành công vào `main` tại commit `09058bd`.

### 2.2. Luồng 2 (Task M1-02): Benchmark Harness Tự Động & Quét Ma Trận Concurrency 1→32
- **Đại diện báo cáo:** Bùi Đức Đại & Cấn Đức Khôi.
- **Kết quả nghiệm thu:**
  - Nâng cấp [`experiments/benchmark_harness.py`](../../../experiments/benchmark_harness.py) hỗ trợ tự động sweep qua 6 mức concurrency: `[1, 2, 4, 8, 16, 32]`.
  - Khóa cứng hợp đồng môi trường `m1-t4-cu130-v1` (Tesla T4, CUDA 13.0, PyTorch 2.13.0, vLLM 0.29.0/0.30.0).
  - Thực thi nghiêm ngặt **3 kỷ luật đo lường**: Warm-up Option B (concurrency-matched), Lặp lại $\ge 5$ lần độc lập mỗi ô (`r001` → `r005`), tính toán Median (p50), Min-Max và gắn cờ cảnh báo phương sai `variance_flag` khi độ lệch $> 15\%$.
  - Module [`experiments/aggregate_results.py`](../../../experiments/aggregate_results.py) tự động trích xuất toàn bộ dữ liệu thô JSONL thành bảng số liệu chuẩn 36 ô tại [`experiments/results/summary.csv`](../../../experiments/results/summary.csv).
  - Xuất bản bộ biểu đồ khoa học F1 và F2 chất lượng cao (300 DPI) tại [`experiments/results/figures/`](../../../experiments/results/figures/).
- **Trạng thái PR:** PR #4 (`feat/m1-02-concurrency-sweep-baseline`) và PR #5 (`fix/m1-02-throughput-aggregation`) đã merge vào `main`.

### 2.3. Luồng 3 (Task M1-03): Pipeline MC-OCR Node 1 & Đóng Gói Bộ Workload W1, W2, W3
- **Đại diện báo cáo:** Dương Nguyễn Thế & Bùi Thiên Đại.
- **Kết quả nghiệm thu:**
  - Hoàn thiện Node 1 (VAT JSON Extractor) theo kiến trúc đường ống xử lý hóa đơn MC-OCR tiếng Việt: trích xuất chuẩn 7 trường thông tin pháp lý từ văn bản OCR thô.
  - Đóng gói 3 tập prompt kiểm thử chuẩn hóa lưu trữ dưới dạng JSONL:
    - `experiments/configs/workloads/w1_vat_extractor.jsonl` (Workload W1 — cấu trúc JSON chặt chẽ, max-tokens 256).
    - `experiments/configs/workloads/w2_text_to_sql.jsonl` (Workload W2 — sinh câu truy vấn SQL bán cấu trúc, max-tokens 96).
    - `experiments/configs/workloads/w3_free_analysis.jsonl` (Workload W3 — phân tích tài chính mở, văn phong tự do, max-tokens 192).
  - Kiểm chứng toàn bộ 30 mẫu devset MC-OCR đạt độ chính xác cú pháp 100% trên vLLM OpenAI-compatible endpoint.
- **Trạng thái PR:** PR #6 (`feat/m1-03-mc-ocr-node1-extractor`) đã merge vào `main` tại commit `f6d2c3e`.

---

## 3. Rà Soát Bảng Số Liệu 36 Ô & Phân Tích Điểm Rơi Lý Thuyết (W3)

Hội đồng đã kiểm tra toàn bộ 36 ô đo lường trong file `summary.csv` và thống nhất xác nhận các phát hiện khoa học quan trọng:

```
                      TÓM TẮT MA TRẬN TĂNG TỐC (SPEEDUP) M1 BASELINE
+---------------+----------+----------+----------+----------+----------+----------+
| Workload      |   c = 1  |   c = 2  |   c = 4  |   c = 8  |  c = 16  |  c = 32  |
+---------------+----------+----------+----------+----------+----------+----------+
| W1 (VAT JSON) |  1.34×   |  1.29×   |  1.33×   |  1.26×   |  1.22×   |  1.16×   | -> Tăng tốc tốt (alpha ~91.6%)
| W2 (SQL)      |  1.34×   |  1.32×   |  1.33×   |  1.23×   |  1.17×   |  1.04×   | -> Tăng tốc vừa (alpha ~80.5%)
| W3 (Phân tích)|  1.00×   |  0.98×   |  1.00×   |  0.93×   |  0.90×   |  0.81×   | -> TĂNG TỐC ÂM (alpha ~54.5%)
+---------------+----------+----------+----------+----------+----------+----------+
```

### 3.1. Phân Tích Chuyên Sâu Hiện Tượng Tăng Tốc Âm (Negative Speedup) trên W3
Đồng chí Bùi Thiên Đại và Cấn Đức Khôi đã trình bày báo cáo phân tích cơ chế suy hao tính toán trên Tesla T4:

1. **Bản chất của Speculative Decoding:**
   Độ trễ trung bình của một bước giải mã suy đoán được biểu diễn qua công thức lý thuyết:
   $$\mathbb{E}[T_{\text{step}}] = T_{\text{draft}}(k) + T_{\text{target\_verify}}(k+1)$$
   Và số lượng token được sinh ra trung bình sau mỗi chu kỳ là:
   $$\tau = 1 + \sum_{i=1}^k \alpha^i$$
   Trong đó $k=3$ là số token nháp cố định trong Static Draft, $\alpha$ là xác suất chấp nhận token.

2. **Tại sao W1 và W2 tăng tốc tốt?**
   - Trên W1 (JSON) và W2 (SQL), cú pháp có tính lặp lại cực cao. Tỉ lệ chấp nhận token của draft model 0.5B đạt $\alpha \approx 91.6\%$ (W1) và $\alpha \approx 80.5\%$ (W2).
   - Số token chấp nhận trung bình đạt $\tau \approx 3.75$ (W1) và $\tau \approx 3.42$ (W2). Mỗi lần kiểm chứng, mô hình 7B sinh được gần trọn vẹn 3-4 token, giúp giảm hơn 70% số lượt forward của mô hình lớn, bù đắp vượt trội chi phí chạy mô hình nhỏ.

3. **Cơ chế gây "Tăng tốc âm" (-19%) trên W3:**
   - Khi văn bản là ngôn ngữ tự do, không gian từ vựng mở rộng, mô hình nháp 0.5B liên tục đoán trượt: $\alpha$ sụt giảm xuống chỉ còn $\approx 54.5\%$, khiến $\tau$ tụt xuống $\approx 2.62$.
   - Ở mức Concurrency cao ($c=16, 32$), GPU Turing T4 bị nghẽn năng lực tính toán (Compute-bound) do phải phục vụ song song 32 luồng:
     - Khi đoán trượt, GPU vẫn phải tốn tài nguyên chạy mô hình nháp $T_{\text{draft}}(3)$.
     - Mô hình Target 7B vẫn phải chạy xác thực và thực hiện rollback token sai.
     - Áp lực tính toán kép này khiến độ trễ sinh mỗi token (TPOT) của Static Draft bị đẩy từ **54.19 ms** (Vanilla) lên tới **67.22 ms** (Static Draft) ở $c=32$.
     - Throughput toàn hệ thống bị sụt giảm từ **483.10 tok/s** xuống **394.99 tok/s** (chạy chậm hơn Vanilla tới **18.24%**).

4. **Ý nghĩa Khoa học Sống còn:**
   Cả nhóm thống nhất kết luận: **Số liệu thực nghiệm này chính là "bằng chứng vàng" (Hard Scientific Evidence) để bảo vệ luận án tốt nghiệp**. Nó chứng minh rõ ràng rằng phương pháp Static Speculative Decoding truyền thống không an toàn cho môi trường phục vụ đa tác vụ, từ đó mở ra sự bắt buộc phải có của **Hardware-Aware Adaptive Speculative Decoding (Mốc M2)**: Bộ điều khiển sẽ tự động nhận diện $\alpha$ thấp để ngắt speculative decoding và chuyển về Vanilla, triệt tiêu 100% hiện tượng tăng tốc âm!

---

## 4. Đóng Băng Giao Diện (Frozen Interfaces)

Hội đồng chính thức ký duyệt đóng băng 2 giao diện phục vụ các giai đoạn tiếp theo:
1. **Giao diện A (Engine ↔ Đo lường & Giám sát):**
   - Chuẩn hóa payload trích xuất telemetry streaming: `alpha` (float), `mean_accept_len` (float), `tpot_ms` (float), `vram_peak_mb` (float), `tokens_generated` (int).
   - Khóa cứng cấu trúc 17 cột của file tổng hợp [`experiments/results/summary.csv`](../../../experiments/results/summary.csv).
2. **Giao diện B (Engine ↔ Ứng dụng MC-OCR & Client):**
   - Chuẩn hóa Endpoint OpenAI REST: `/v1/chat/completions` và `/health`.
   - Chuẩn hóa schema đầu ra JSON trích xuất hóa đơn của Node 1 tuân thủ nghiêm ngặt `invoice_extraction.v1`.

---

## 5. Kết Luận Nghiệm Thu & Phê Duyệt Kế Hoạch Sprint 3

### 5.1. Nghiệm thu Sprint 2 (M1): ĐẠT 100% TIÊU CHÍ (PASS)
- [x] Code/kết quả đã được test không lỗi trên môi trường mục tiêu (Tesla T4).
- [x] Đã có log/kết quả lưu đầy đủ tại `experiments/` và commit lên Git.
- [x] Đã cập nhật số liệu và viết bản thảo quyển báo cáo tại [`docs/chuong-4-thuc-nghiem.md`](../../../docs/chuong-4-thuc-nghiem.md).
- [x] Toàn bộ 4 thẻ task của Sprint 2 (`M1-01`, `M1-02`, `M1-03`, `M1-04`) đủ điều kiện chuyển sang cột **Done**.

### 5.2. Chuyển giao sang Sprint 3 (Mốc M2):
- **Thời gian Sprint 3:** Bắt đầu từ 05/10/2026 đến 18/10/2026.
- **Trọng tâm Sprint 3:** Xây dựng bộ điều khiển thích ứng (Adaptive Controller v0), kiểm chứng tính nguyên vẹn không suy giảm chất lượng văn bản (Lossless Verification 100% Match), và đo đạc 3 làn đối chứng.
- Tài liệu phân rã chi tiết đã được phê duyệt tại: [`docs/sprint_3_plan.md`](../../../docs/sprint_3_plan.md).

---

### Chữ Ký Xác Nhận Của Các Thành Viên

| Thành viên | Vai trò | Chữ ký xác nhận |
| :--- | :--- | :---: |
| **Cấn Đức Khôi** | Team Lead / AI Systems / PM | *Khôi (Đã ký)* |
| **Bùi Thiên Đại** | MLOps / Benchmark Harness Lead | *Đại (Đã ký)* |
| **Dương Nguyễn Thế** | Core Engine / BA / Data Lead | *Dương (Đã ký)* |
