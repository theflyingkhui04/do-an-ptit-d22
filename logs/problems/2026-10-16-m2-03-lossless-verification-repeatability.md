# Báo cáo kỹ thuật và quyết định PM: [M2-03] Hiện tượng bất định số học và giao thức kiểm chứng lossless

- **Mã vấn đề:** `PROB-M2-03-LOSSLESS-REPEATABILITY`
- **Cột mốc liên quan:** `M2 (Adaptive v0 & Lossless Verification)`
- **Ngày ghi nhận:** 16/10/2026
- **Người phát hiện / Raise:** Bùi Đức Đại (Dev + IT / MLOps & Benchmark)
- **Người xem xét & Phê duyệt:** Cấn Đức Khôi (Dev + PM / AI Systems)
- **Trạng thái:** **PM approved — Phê duyệt phương án C (qualify baseline và paired test)**

---

## 1. Bối cảnh và mục tiêu ban đầu (Gate M2-03)

Tiêu chí nghiệm thu cốt lõi (Definition of Done - DoD) của mốc M2-03:
- Ở chế độ giải mã suy luận chuẩn xác (`temperature = 0.0`, greedy search), thuật toán tăng tốc thích ứng (**Adaptive**) bắt buộc phải sinh ra chuỗi token **khớp chính xác 100% từng token ID** so với chế độ giải mã tuần tự truyền thống (**Vanilla**).
- Tỷ lệ sai lệch chấp nhận: `Zero token discrepancy` trên tập canonical >= 50 prompts (hiện tại sử dụng bộ 54 prompts: W1 = 30, W2 = 12, W3 = 12).
- Cơ sở lý thuyết: Theo chuẩn mực Speculative Decoding (Leviathan et al., 2023), mô hình mục tiêu (Target model) luôn là người đưa ra phán quyết xác thực cuối cùng (Rejection Sampler). Về mặt toán học ở `temperature = 0`, phân phối đầu ra của Speculative Decoding phải hoàn toàn đồng nhất với giải mã chuẩn (lossless).

---

## 2. Bằng chứng thực nghiệm và dữ liệu phát hiện lỗi

### 2.1. Kết quả kiểm thử thực tế trên GPU Tesla T4 (vLLM 0.30.0)
Trong chiến dịch kiểm thử canonical `m2-03-20261013-r03`:
- **Tổng số prompt kiểm thử:** 54 prompts (W1 = 30, W2 = 12, W3 = 12).
- **Kết quả:** **44/54 prompt khớp chính xác (81.48%) -> Không đạt tiêu chuẩn nghiệm thu.**
- **Chi tiết sai lệch:** 10 prompt bị lệch token (mismatches) và 7 cặp bị cắt cụt độ dài (truncated pairs).

### 2.2. Đi sâu phân tích ca lỗi điển hình (reproducible case)
- **Prompt:** `W3 / w3_business_v2_004` (tác vụ phân tích báo cáo tài chính mở).
- **Quan sát đầu ra:**
  - `Vanilla (reference)`: Sinh 284 tokens. Token thứ 200 (index 200) là `128371`.
  - `Adaptive (attempt 1)`: Sinh 284 tokens. Token thứ 200 là `128371` -> **Match** (khớp hoàn toàn Vanilla).
  - `Adaptive (attempt 2)`: Sinh 285 tokens. Token thứ 200 lại là `128585` -> **Lệch** (bắt đầu khác từ index 200 và kéo theo toàn bộ chuỗi token phía sau bị thay đổi).

### 2.3. Manh mối logit tại index 200
Đại đã capture thành công giá trị Target Logits của Vanilla tại vị trí index 200:
- Token top 1 (`128371`): Logit = `12.500`.
- Token top 2 (`128585`): Logit = `12.500`.
- **Hiện tượng hòa điểm (tie):** Khoảng cách chênh lệch giữa 2 token đứng đầu là tuyệt đối bằng không (`top2 margin = 0.0`).
- Kết quả chọn:
  - Vanilla chọn: `128371`.
  - Adaptive (ở lần chạy lỗi) chọn: `128585`.

---

## 3. Phát hiện bước ngoặt: Tính bất định của chính baseline (Vanilla non-repeatability)

Khi tiến hành kiểm tra chéo tính lặp lại độc lập giữa các phiên chạy (cross-run repeatability), Đại phát hiện bằng chứng thực nghiệm quan trọng:

| Cặp so sánh đối chứng | Tỷ lệ khớp chính xác | Trạng thái |
| :--- | :---: | :---: |
| **Vanilla Run A vs Vanilla Run B** | **45 / 54 (83.33%)** | **Chính Vanilla tự chạy lại cũng lệch 9 prompt** |
| **Adaptive Run A vs Adaptive Run B** | **45 / 54 (83.33%)** | **Adaptive tự chạy lại cũng lệch** |
| **Vanilla vs Adaptive (Run r03)** | **44 / 54 (81.48%)** | Không khớp hoàn toàn |
| **Vanilla vs Adaptive (Run prodq3)** | **44 / 54 (81.48%)** | Không khớp hoàn toàn |

### Kết luận khoa học từ bằng chứng:
1. Bản thân phép đo cơ sở (**Vanilla Baseline**) chưa đạt tính lặp lại 100% giữa các phiên chạy độc lập (cross-run numerical non-determinism).
2. Khi hai token bị hòa điểm logit (`margin = 0.0`), các sai số làm tròn số thực (floating-point precision), thứ tự tính tổng trong phép nhân ma trận song song của CUDA kernel (reduction order), hoặc cơ chế phân bổ trang bộ nhớ PagedAttention sẽ làm đảo chiều token được chọn giữa các lần chạy.
3. **Do đó, không thể dùng một bản chạy Vanilla cũ ở môi trường hoặc thời điểm khác làm chuẩn tuyệt đối (ground truth) để quy kết rằng "Adaptive gây ra lỗi".**

---

## 4. Phân tích ba phương án giải quyết

### Phương án A: Giữ nguyên gate 54/54 cross-run ban đầu
- **Cách làm:** Bắt buộc chứng minh Vanilla tự khớp 54/54 trước, sau đó Adaptive phải khớp Vanilla 54/54.
- **Ưu điểm:** Giữ nguyên cam kết ban đầu, không thay đổi tiêu chí nghiệm thu.
- **Nhược điểm:** Thực nghiệm cho thấy Vanilla chỉ đạt 45/54 cross-run. Rất dễ rơi vào bẫy "săn bóng ma", đốt tài nguyên GPU vô ích để giải quyết một bài toán non-deterministic phần cứng cố hữu.
- **Điểm đánh giá khoa học:** 4.5 / 10.

### Phương án B: Chuyển sang so sánh cặp có kiểm soát (paired controlled comparison)
- **Cách làm:** Chạy đồng thời Vanilla và Adaptive theo từng cặp trong cùng một phiên thực thi (cùng GPU vật lý, driver, CUDA, revision mô hình, seed, cache state). Đánh giá xem: *"Mức độ sai lệch của Adaptive có vượt quá phương sai tự nhiên (drift) của chính Vanilla hay không?"*.
- **Ưu điểm:** Phản ánh đúng bản chất vật lý của hệ thống suy luận phân tán, dễ bảo vệ luận điểm khoa học.
- **Nhược điểm:** Phải hạ mức cam kết từ "100% lossless tuyệt đối" xuống "lossless trong biên độ sai số phần cứng", làm giảm sức nặng của kết luận.
- **Điểm đánh giá khoa học:** 8.5 / 10.

### Phương án C: Giữ 100% token match nhưng chuẩn hóa điều kiện baseline (qualify baseline first)
- **Cách làm:** Tách quy trình thành 2 giai đoạn (2-phase protocol):
  - **Phase 1 (Baseline qualification):** Khóa cứng môi trường thực thi (freeze execution regime), kích hoạt các cờ tối đa hóa tính xác định (deterministic mode). Chạy kiểm tra `Vanilla A vs Vanilla B vs Vanilla C`. Nếu Phase 1 đạt 54/54, chứng minh môi trường đo lường đã đủ tin cậy.
  - **Phase 2 (Paired lossless test):** Trong chính môi trường đã chuẩn hóa đó, chạy `Adaptive vs Vanilla` và tiếp tục áp dụng tiêu chí nghiêm ngặt **100% token ID match (54/54)**.
- **Ưu điểm:**
  - Vẫn bảo toàn 100% tính nguyên bản của kết luận "lossless tuyệt đối".
  - Tách biệt rành mạch giữa sai số hạ tầng đo (measurement drift) và lỗi thuật toán (adaptive bug).
  - Phương pháp luận nghiên cứu đạt chuẩn NIST và PyTorch Reproducibility Guidelines.
- **Điểm đánh giá khoa học:** **9.5 / 10 (được Đại và Khôi nhất trí đề xuất).**

---

## 5. Quyết định chính thức của PM (Cấn Đức Khôi)

1. **Chấp thuận phương án C:** Duyệt đề xuất của Đại, áp dụng giao thức 2 giai đoạn cho task M2-03.
2. **Hướng dẫn kỹ thuật cho Phase 1 (Baseline qualification):**
   Để triệt tiêu tính bất định số học trên GPU Tesla T4 (kiến trúc Turing CC 7.5), Đại áp dụng 3 quy tắc sau trong script chạy:
   - **Ghim cờ cuBLAS:** `export CUBLAS_WORKSPACE_CONFIG=:4096:8` (hoặc `:16:8`) trước khi khởi chạy tiến trình Python.
   - **Ép giải thuật xác định:** Bổ sung `torch.use_deterministic_algorithms(True)` trong runner.
   - **Khử nhiễu batching:** Chạy kiểm chứng lossless ở chế độ đơn luồng tuần tự (`concurrency = 1`, batch size = 1) để tránh việc gộp batch động làm thay đổi thứ tự cộng dồn dấu phẩy động giữa các token bị tie.
3. **Điều khoản chốt chặn (Fallback gate ngày 23/10):**
   - Nếu Phase 1 đạt 54/54 -> Tiến hành Phase 2 để chốt kết luận lossless 100%.
   - Nếu sau khi bật toàn bộ cờ deterministic mà kernel lượng tử hóa AWQ Marlin trên T4 vẫn cố hữu lệch ở các token tie điểm (như logit 12.500 vs 12.500) khiến Vanilla chỉ đạt 45-50/54 -> Nhóm sẽ tự tin kích hoạt **phương án B (paired comparison)** và đưa chính bản phân tích này vào Chương 4 và Chương 5 của quyển đồ án như một đóng góp thực nghiệm sâu sắc về giới hạn vật lý của phần cứng GPU phổ thông.
