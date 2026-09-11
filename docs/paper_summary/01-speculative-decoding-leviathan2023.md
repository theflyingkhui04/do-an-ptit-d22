# Tóm tắt Paper 1: Fast Inference from Transformers via Speculative Decoding

* **Tác giả:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
* **Hội nghị:** ICML 2023
* **Link PDF:** https://arxiv.org/pdf/2211.17192.pdf
* **Link arXiv:** https://arxiv.org/abs/2211.17192
* **Mã nguồn tham khảo:** Đã được tích hợp rộng rãi trong Hugging Face Transformers, vLLM, TensorRT-LLM.

---

## 1. Bối cảnh & Vấn đề cốt lõi (Memory-Bandwidth Bound)

Khi thực hiện suy luận trên các mô hình ngôn ngữ lớn (LLM), quá trình sinh văn bản diễn ra theo kiểu **tự hồi quy (Autoregressive)**:
```
Input  ──>  Token 1  ──>  Token 2  ──>  Token 3  ──>  ...  ──>  Token N
```

Để sinh ra **duy nhất 1 token**:
* GPU bắt buộc phải nạp (load) toàn bộ hàng tỷ tham số trọng số (weights) từ bộ nhớ đồ họa (VRAM/HBM) vào bộ nhớ nội bộ của chip (SRAM / Register).
* Ở kịch bản phục vụ thông thường (đặc biệt khi `batch size = 1` hoặc batch size nhỏ), lượng tính toán cho 1 token là quá nhỏ so với lượng dữ liệu khổng lồ phải nạp từ bộ nhớ.
* Hiện tượng này gọi là **Memory-Bandwidth Bound** (nghẽn băng thông bộ nhớ). Tỉ lệ tính toán trên mỗi byte truyền tải (FLOPs/byte - Arithmetic Intensity) rất thấp, khiến các nhân tính toán cực mạnh của GPU (Tensor Cores) phần lớn thời gian phải "ngồi chơi" chờ nạp dữ liệu.

---

## 2. Ý tưởng của Giải mã suy đoán (Speculative Decoding)

Speculative Decoding (SpD) giải quyết nút thắt trên bằng cách phối hợp hai mô hình:
1. **Target Model (M_p):** Mô hình mục tiêu lớn, sinh văn bản chất lượng cao nhất, có phân phối xác suất p(x). Mô hình này chạy chậm và tốn tài nguyên.
2. **Draft Model (M_q):** Mô hình nháp nhỏ hơn rất nhiều (thường nhỏ hơn 10 - 20 lần), có phân phối xác suất q(x). Mô hình này chạy cực nhanh và nhẹ.

### Quy trình 3 bước trong mỗi vòng lặp (Iteration):
1. **Bước 1 (Soạn nháp - Drafting):**
   Mô hình nhỏ M_q chạy tự hồi quy K bước liên tiếp để đề xuất trước một chuỗi gồm K tokens nháp:
   `x_nháp_1,  x_nháp_2,  ...,  x_nháp_K`
   Vì M_q rất nhỏ, việc sinh K token này tốn rất ít thời gian.
2. **Bước 2 (Xác thực song song - Verification):**
   Đưa toàn bộ ngữ cảnh cùng K tokens nháp vào mô hình lớn M_p **trong duy nhất 1 lần chạy (single forward pass)**.
   Nhờ khả năng xử lý ma trận song song của GPU, thời gian M_p xử lý cùng lúc (K + 1) tokens gần như bằng với thời gian nó chỉ chạy 1 token đơn lẻ.
3. **Bước 3 (Duyệt & Lấy mẫu lại - Acceptance & Resampling):**
   Duyệt qua từng token nháp bằng thuật toán **Acceptance Criterion**. Chấp nhận các token hợp lệ và dừng lại ngay khi gặp token đầu tiên sai lệch phân phối để hiệu chỉnh.

---

## 3. Thuật toán Tiêu chí chấp nhận (Acceptance Criterion)

Thuật toán sử dụng cơ chế **Modified Rejection Sampling** để đảm bảo phân phối đầu ra hoàn toàn giữ nguyên:

Tại vị trí thứ i:
* Mô hình nháp đề xuất token `x_nháp` dựa trên phân phối `q(x)`.
* Mô hình lớn tính toán phân phối chuẩn `p(x)`.

### Quy tắc chấp nhận:
1. Tính xác suất chấp nhận α:
   ```
   α = min(1,  p(x_nháp) / q(x_nháp))
   ```
2. Bốc ngẫu nhiên một số thực r trong khoảng đều (0, 1):
   * **Nếu r < α:** **Chấp nhận** `x_nháp`. Tiếp tục xét token nháp thứ (i + 1).
   * **Nếu r ≥ α:** **Bác bỏ** `x_nháp`. Toàn bộ các token nháp đằng sau từ (i + 1) đến K lập tức bị hủy bỏ.

### Quy tắc lấy mẫu lại khi bị bác bỏ (Resample):
Khi `x_nháp` bị bác bỏ, lấy mẫu một token thay thế `x_mới` từ **phân phối hiệu chỉnh dư (Residual Distribution) p'(x)**:
```
            max(0,  p(x) - q(x))
p'(x) = ─────────────────────────────
         ∑ [y] max(0,  p(y) - q(y))
```
Token `x_mới ~ p'(x)` được chọn làm kết quả chính thức và vòng lặp dừng lại.

### Phần thưởng Token miễn phí (Bonus Token):
Nếu **cả K tokens nháp đều được chấp nhận**, mô hình lớn M_p từ lần forward pass song song đã có sẵn logit tại vị trí (K + 1). Ta chỉ việc lấy mẫu thêm một token:
```
x_(K+1) ~ p_(K+1)(x)
```
Token này hoàn toàn không tốn thêm chi phí tính toán nào.

> Như vậy, trong mỗi lần gọi mô hình lớn, ta thu được từ **1 đến (K + 1) tokens**.

---

## 4. Chứng minh toán học bảo toàn phân phối (Exact Distribution Preservation)

Cần chứng minh rằng xác suất token x bất kỳ được chọn cuối cùng, P(X = x), đúng bằng p(x) của mô hình lớn:

Một token x có thể được chọn qua 2 trường hợp:
* **Trường hợp 1:** Nháp chọn x và được chấp nhận.
* **Trường hợp 2:** Nháp bị bác bỏ và quá trình lấy mẫu lại bốc trúng x.

Công thức xác suất toàn phần:
```
P(X = x) = P(Trường hợp 1) + P(Bị bác bỏ) × P(Lấy mẫu lại ra x)
```

1. **Xác suất Trường hợp 1:**
   ```
   P(Trường hợp 1) = q(x) × min(1,  p(x) / q(x)) = min(q(x),  p(x))
   ```

2. **Xác suất Bị bác bỏ:**
   ```
   P(Bị bác bỏ) = 1 - ∑ [y] min(q(y),  p(y))
   ```
   Vì với mọi số a, b ta có: a - min(a, b) = max(0, a - b), và tổng xác suất ∑ [y] p(y) = 1, ta biến đổi được:
   ```
   1 - ∑ [y] min(q(y),  p(y)) = ∑ [y] p(y) - ∑ [y] min(q(y),  p(y))
                              = ∑ [y] max(0,  p(y) - q(y))
   ```

3. **Xác suất Trường hợp 2:**
   ```
   P(Trường hợp 2) = [ ∑ [y] max(0, p(y) - q(y)) ] × [ max(0, p(x) - q(x)) / ∑ [y] max(0, p(y) - q(y)) ]
                   = max(0,  p(x) - q(x))
   ```

4. **Tổng hợp 2 trường hợp:**
   ```
   P(X = x) = min(q(x),  p(x)) + max(0,  p(x) - q(x)) = p(x)
   ```

**Kết luận:** Phân phối đầu ra của Speculative Decoding trùng khớp 100% với mô hình lớn. Hoàn toàn không mất mát chất lượng (Lossless).

---

## 5. Đánh giá tốc độ (Speedup Analysis)

Gọi:
* `T_target`: Thời gian chạy 1 forward pass của mô hình lớn.
* `T_draft`: Thời gian chạy 1 forward pass của mô hình nhỏ (`T_draft << T_target`).
* `K`: Số lượng tokens nháp trong 1 vòng.
* `α`: Tỷ lệ chấp nhận trung bình của các token nháp.

1. **Thời gian 1 vòng lặp:**
   ```
   T_vòng ≈ K × T_draft + T_target
   ```
2. **Số token trung bình sinh ra sau 1 vòng:**
   ```
   E[tokens] = 1 + α + α² + ... + α^K = (1 - α^(K+1)) / (1 - α)
   ```
3. **Hệ số tăng tốc (Speedup):**
   ```
   Speedup = [ (1 - α^(K+1)) / (1 - α) × T_target ] / [ K × T_draft + T_target ]
   ```

*Ví dụ:* Nếu `T_draft = 0.05 × T_target`, `K = 5`, tỷ lệ chấp nhận `α = 0.75` (75%):
* Số token trung bình sinh ra: `E[tokens] ≈ 3.29` tokens.
* Thời gian tốn: `5 × 0.05 + 1.0 = 1.25 × T_target`.
* Tốc độ: `Speedup = 3.29 / 1.25 ≈ 2.63x` (Tăng tốc gấp 2.6 lần).

---

## 6. Hạn chế & Hướng phát triển trong các nghiên cứu tiếp theo

| Hạn chế của Paper 1 | Vấn đề cụ thể | Hướng khắc phục ở các bài báo sau |
|---|---|---|
| **Cần 1 mô hình nháp riêng biệt** | Tốn VRAM chứa 2 mô hình; khó đồng bộ tokenizer và căn chỉnh phân phối | **Paper 2 (EAGLE):** Bỏ mô hình nháp, dùng 1 Transformer Head nhẹ dự đoán ở mức đặc trưng (Feature). |
| **Chuỗi nháp tuyến tính (Sequential)** | Token nháp đầu tiên sai dẫn đến hủy toàn bộ chuỗi phía sau | **Paper 3 (EAGLE-2) & Paper 4 (Talon):** Cấu trúc cây nháp (Draft Tree) mở rộng nhiều nhánh đồng thời. |
| **Độ dài nháp K cố định (Static Length)** | Câu dễ sinh ít, câu khó sinh thừa gây lãng phí tính toán | **Paper 5 (PEARL):** Tự động điều chỉnh độ dài chuỗi nháp theo trạng thái. |
| **Hiệu năng giảm khi chạy Batched Serving** | Khi batch size tăng, GPU chuyển sang Compute-Bound, mất ưu thế song song | **Paper 6 (D-cut):** Tự động cắt tỉa độ sâu xác thực (depth pruning) khi GPU chịu tải cao. |
