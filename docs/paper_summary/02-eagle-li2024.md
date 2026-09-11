# Tóm tắt Paper 2: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

* **Tác giả:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (Peking University, MSRA)
* **Hội nghị:** ICML 2024
* **Link PDF:** https://arxiv.org/pdf/2401.15077.pdf
* **Link arXiv:** https://arxiv.org/abs/2401.15077
* **Mã nguồn:** https://github.com/SafeAILab/EAGLE

---

## 1. Vấn đề cốt lõi: Độ bất định ở tầng Token (Token Uncertainty)

Các phương pháp Speculative Decoding thế hệ đầu (như Paper 1 của Leviathan et al. hoặc Medusa) đều thực hiện dự đoán ở **tầng Token (Token-level)**:
* Mô hình nháp cố gắng dự đoán chuỗi các token rời rạc tiếp theo: `x_(t+1), x_(t+2), ...`
* **Điểm nghẽn:** Ngôn ngữ tự nhiên có độ bất định từ ngữ rất cao (High Token Uncertainty). Cùng một ý nghĩa ngữ nghĩa nhưng có thể diễn đạt bằng nhiều từ đồng nghĩa khác nhau (ví dụ: "chú mèo", "con mèo", "mèo con"). Phân phối xác suất rời rạc trên không gian từ điển khổng lồ (32.000 - 128.000 tokens) có độ hỗn loạn (entropy) rất lớn.
* **Lan truyền lỗi (Error Propagation):** Khi mô hình nháp đoán sai chỉ 1 token ở bước đầu, toàn bộ các token phía sau dù có hợp lý cũng bị Target Model bác bỏ do ngữ cảnh đã lệch.

### Khám phá then chốt của EAGLE:
Trong kiến trúc Transformer, các vector trạng thái ẩn ở tầng trên cùng (top-layer hidden states / feature vectors) chứa thông tin ngữ nghĩa trừu tượng, cô đọng và có **độ bất định thấp hơn nhiều (Low Feature Uncertainty)** so với phân phối token rời rạc ở đầu ra của LM Head.

> **Tư tưởng của EAGLE:** Thay vì đoán xem token tiếp theo là chữ gì, hãy đoán xem **vector đặc trưng ngữ nghĩa tiếp theo (Feature Vector) trông như thế nào**.

---

## 2. Kiến trúc & Cơ chế hoạt động của EAGLE

EAGLE loại bỏ hoàn toàn việc phải duy trì một mô hình nháp độc lập (vốn tốn VRAM và khó đồng bộ). Thay vào đó, EAGLE bổ sung một **Auto-regression Head (Draft Head)** siêu nhẹ được gắn trực tiếp vào mô hình lớn (Target Model).

### Thiết kế của Draft Head:
* Bản chất là **chỉ 1 tầng Transformer Decoder** (gồm Self-Attention và Feed-Forward Network).
* Kích thước tham số cực nhỏ (chỉ chiếm khoảng 1% - 3% kích thước mô hình lớn).
* Sử dụng chung không gian embedding và đầu ra LM Head với mô hình lớn.

### Quy trình sinh nháp theo cơ chế hồi quy Feature (Feature-level Drafting):

Tại thời điểm t, mô hình lớn đã tạo ra vector đặc trưng ẩn ở tầng trên cùng là `f_t`:

1. **Bước 1:** Đưa `f_t` qua LM Head gốc của mô hình lớn để thu được phân phối xác suất và lấy mẫu token `x_t`:
   ```
   x_t ~ Softmax( LM_Head(f_t) )
   ```
2. **Bước 2:** Lấy vector nhúng (embedding) của token `x_t` là `e(x_t)`. Kết hợp `f_t` và `e(x_t)` rồi đưa vào tầng Draft Head nhẹ để ngoại suy ra vector đặc trưng tiếp theo:
   ```
   f_(t+1) = Draft_Head( f_t + e(x_t) )
   ```
3. **Bước 3:** Lặp lại quá trình trên: Đưa `f_(t+1)` qua LM Head gốc để sinh token `x_(t+1)`, rồi tiếp tục đưa `f_(t+1)` cùng `e(x_(t+1))` vào Draft Head để dự đoán `f_(t+2)`...

```
   [f_t]  ────────────────────────>  [f_(t+1)]  ────────────────────>  [f_(t+2)]
     │                                   │                                  │
  LM Head                             LM Head                            LM Head
     ▼                                   ▼                                  ▼
   (x_t) ──> Embedding ──[ + ] ───────> (x_(t+1)) ──> Embedding ──[ + ] ──> (x_(t+2))
```

**Tại sao cơ chế này vượt trội?**
* Draft Head luôn được neo (anchor) vào biểu diễn ngữ nghĩa chất lượng cao của mô hình lớn.
* Dự đoán trong không gian liên tục (continuous feature space) mượt mà và ổn định hơn rất nhiều so với không gian từ rời rạc.

---

## 3. Cấu trúc Cây nháp & Tree Attention (Draft Tree & Tree Mask)

Thay vì chỉ sinh một chuỗi nháp đơn tuyến tính (nếu sai từ đầu là bỏ hết), EAGLE mở rộng các ứng viên nháp thành một **Cây nháp (Draft Tree)**:
* Tại mỗi bước, Draft Head giữ lại top-k tokens có xác suất cao nhất để rẽ nhánh.
* Ví dụ: từ 1 node gốc tạo ra 3 nhánh, mỗi nhánh lại sinh tiếp 2 nhánh con -> tạo thành một cây gồm nhiều đường đi tiềm năng.

### Cơ chế Tree Mask (Xác thực song song toàn bộ cây):
Làm sao để mô hình lớn xác thực toàn bộ các nhánh trên cây trong **duy nhất 1 lần forward pass** mà các nhánh không nhìn thấy nhau gây loạn ngữ cảnh?

EAGLE sử dụng một ma trận **Tree Attention Mask** tùy biến:
* Một token ở nhánh A chỉ được phép attend (chú ý) đến chính nó, các token tổ tiên (ancestors) trên cùng nhánh của nó, và chuỗi prompt gốc.
* Một token ở nhánh A **bị chặn hoàn toàn (mask = -vô cùng)** không được nhìn thấy các token ở nhánh anh em B.

Nhờ Tree Mask, mô hình lớn tính toán song song tất cả các nhánh trong cây cùng lúc. Nhánh nào dài nhất và vượt qua được Acceptance Criterion sẽ được chọn làm kết quả chính thức.

---

## 4. Huấn luyện Draft Head cực kỳ nhẹ nhàng

* **Mô hình lớn (Target Model):** Đóng băng hoàn toàn (Freeze 100%), không can thiệp vào trọng số.
* **Mô hình nháp:** Chỉ huấn luyện duy nhất 1 tầng Transformer Decoder của Draft Head.
* **Dữ liệu huấn luyện:** Trích xuất các cặp `(f_t, x_t, f_(t+1))` từ quá trình chạy thông thường của mô hình lớn trên tập văn bản (ví dụ tập ShareGPT).
* **Hàm mất mát (Loss function):** Kết hợp giữa:
  * Regression Loss (Smooth L1 Loss) giữa feature dự đoán và feature thật.
  * Cross-Entropy Loss của token dự đoán so với token thật.
* **Thời gian huấn luyện:** Chỉ mất 1 - 2 ngày trên 1 GPU phổ thông (như RTX 3090 / A100) cho mô hình 7B - 13B.

---

## 5. Kết quả thực nghiệm chính

* **Tốc độ tăng tốc (Speedup):**
  * Tăng tốc từ **2.0x đến 3.0x** so với Vanilla Autoregressive Decoding trên các mô hình LLaMA-2-Chat (7B, 13B, 70B).
  * Nhanh hơn đáng kể so với các phương pháp cùng thời như Lookahead Decoding (1.5x) hay Medusa (1.9x - 2.2x).
* **Tỷ lệ chấp nhận (Acceptance Rate):** Trung bình mỗi lần gọi mô hình lớn sinh được từ 2.5 đến 3.5 tokens.
* **Bảo toàn chất lượng:** Vẫn giữ nguyên 100% phân phối gốc (Lossless) nhờ tuân thủ chặt chẽ thuật toán Acceptance Criterion.

---

## 6. Điểm hạn chế của EAGLE (Tiền đề cho EAGLE-2)

Mặc dù EAGLE-1 rất mạnh, nó vẫn tồn tại một nhược điểm thiết kế:
* **Cấu trúc cây nháp là tĩnh (Static Draft Tree):**
  * Hình dạng của cây nháp (độ sâu, số lượng nhánh ở mỗi tầng) được cấu hình cố định từ trước cho mọi ngữ cảnh.
  * *Hậu quả:* Khi gặp đoạn văn bản dễ đoán (độ tự tin cao), cây tĩnh không đủ sâu để sinh nhiều token hơn. Ngược lại, khi gặp đoạn văn bản khó (độ tự tin thấp), cây tĩnh vẫn cố sinh nhiều nhánh, dẫn đến hầu hết các nhánh đều bị Target Model bác bỏ, làm lãng phí năng lực tính toán của GPU.

*→ Điểm hạn chế này chính là động lực ra đời của **Paper 3 (EAGLE-2)**: Thay thế cây tĩnh bằng **Cây nháp động (Dynamic Draft Tree)** tự co giãn theo độ bất định.*
