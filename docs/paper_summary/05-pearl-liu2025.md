# Tóm tắt Paper 5: PEARL: Parallel Speculative Decoding with Adaptive Draft Length

* **Tác giả:** Tianyu Liu, Yun Li, Qitan Lv, Kai Liu, Jianchen Zhu, Winston Hu, Xiao Sun (USTC, Tencent, Shanghai AI Laboratory)
* **Hội nghị:** ICLR 2025
* **Link PDF:** https://arxiv.org/pdf/2408.11850
* **Link arXiv:** https://arxiv.org/abs/2408.11850
* **Trang dự án / mã nguồn:** https://pearl-code.github.io

---

## 1. Vấn đề cốt lõi: Bài toán chờ nhau (Mutual Waiting Problem)

Mọi paper trước (1 → 4) đều tối ưu **nội dung bản nháp** (nháp cái gì, nháp bao nhiêu, xếp thành hình gì). PEARL tấn công một trục hoàn toàn khác: **lịch chạy (scheduling)**.

Trong Speculative Decoding chuẩn, hai pha chạy **nối tiếp tuyệt đối**:

```
Thời gian ───────────────────────────────────────────────────────>

Draft  M_q : [nháp γ token]                 [nháp γ token]
Target M_p :                 [xác thực]                    [xác thực]
             └─ M_p NGỒI KHÔNG ─┘           └─ M_q NGỒI KHÔNG ─┘
```

Nguyên nhân là ràng buộc dữ liệu hai chiều:

1. Muốn nháp thì phải có **tiền tố đã được xác thực**.
2. Muốn xác thực thì phải có **bản nháp đã sinh xong**.

Số đo thực tế trong bài báo (cặp Codellama): mô hình nháp nằm không **0.07 s** trong mỗi pha xác thực, còn mô hình lớn nằm không **0.14 s** trong mỗi pha nháp. Với hai mô hình nằm trên hai GPU khác nhau, đây là **một nửa tài nguyên bị lãng phí theo chu kỳ**.

Cộng thêm: độ dài nháp `γ` cố định làm thời lượng mỗi pha cứng nhắc, không co giãn theo độ khó của từng bước.

---

## 2. Ý tưởng lõi: chồng lấn hai pha bằng hai thủ thuật xác thực

PEARL cho **cả hai mô hình chạy đồng thời**, bằng cách nhét thêm việc hữu ích vào đúng khoảng thời gian mà mỗi mô hình đang rảnh. Hai chiến lược đối xứng nhau:

### 2.1. Pre-verify — "xác thực trước token đầu tiên"

**Khoảng rảnh được tận dụng:** mô hình lớn `M_p` đang ngồi không trong lúc bộ nháp sinh `γ` token.

**Làm gì:** ngay khi bắt đầu pha nháp, cho `M_p` chạy song song trên **chính tiền tố hiện tại `x`** để lấy logits tại vị trí 1. Việc này **không cần chờ bản nháp** — chỉ cần tiền tố, thứ đã có sẵn.

Khi bộ nháp trả về `x_1, ..., x_γ`, ta đã có sẵn `p_1` để duyệt ngay token đầu theo tiêu chí chuẩn:

```
α_i  =  1                    nếu  p_i[x_i]  ≥  q_i[x_i]
        p_i[x_i] / q_i[x_i]  nếu ngược lại
```

* **Nếu `x_1` bị bác bỏ:** **bỏ qua luôn cả pha xác thực**. Lấy mẫu lại `y_1` từ phân phối dư, nối thành `x + [y_1]`, quay lại pha nháp.
  → Lợi ích: khi bản nháp đã sai ngay từ token đầu, `x_2...x_γ` chắc chắn vô giá trị; PEARL **không tốn một lượt forward pass nào của mô hình lớn cho rác**. Đây chính là hành vi "nháp ngắn lại khi khó".
* **Nếu `x_1` được chấp nhận:** đi tiếp sang post-verify.

### 2.2. Post-verify — "nháp trước phần tiếp theo"

**Khoảng rảnh được tận dụng:** bộ nháp `M_q` đang ngồi không trong lúc mô hình lớn xác thực.

**Làm gì:** **giả định lạc quan rằng cả `γ` token nháp đều sẽ được chấp nhận**, cho bộ nháp chạy tiếp sinh `x_(γ+1), ..., x_(2γ)` ngay trong lúc `M_p` còn đang xác thực lô đầu.

* **Nếu cả `γ` token đều được chấp nhận:** canh bạc thắng. Tiền tố nhảy thẳng thành `x + [y_1, x_1, ..., x_(2γ)]` và đi **thẳng vào pha xác thực kế tiếp** mà không phải chờ nháp lần nữa. Thêm nữa, logits `M_p(x + [x_1..x_γ])` từ lượt vừa rồi đóng luôn vai trò "pre-verify" cho lô sau → miễn phí.
  → Đây là hành vi "nháp dài ra khi dễ": chuỗi chấp nhận liên tiếp làm độ dài nháp hiệu dụng **tăng không giới hạn**.
* **Nếu bị bác ở vị trí `n`:** chấp nhận `x_1...x_n`, lấy mẫu lại tại `n+1`, phần nháp sinh thừa bị bỏ, quay về chế độ pre-verify.
  → Phần nháp bỏ đi **không tốn gì thêm** vì nó chạy trên tài nguyên vốn dĩ đang rảnh.

```
PEARL:
Draft  M_q : [nháp 1..γ    ][nháp γ+1..2γ ][nháp 2γ+1..3γ]
Target M_p : [pre-verify x ][xác thực 1..γ][xác thực ...  ]
             └── không còn khoảng trắng ở cả hai làn ──┘
```

### 2.3. Độ dài nháp thích ứng "tự mọc ra" (Adaptive Draft Length)

Điểm đẹp về mặt thiết kế: PEARL **không có thuật toán nào dự đoán độ dài nháp cả**. Tính thích ứng là **hệ quả** của hai thủ thuật trên:

* Nháp sai sớm → pre-verify cắt ngay → độ dài hiệu dụng ngắn.
* Nháp đúng liên tục → post-verify nối chuỗi → độ dài hiệu dụng dài.

Không cần bộ dự đoán, không cần ngưỡng, không cần huấn luyện. Trích nguyên văn tinh thần bài báo: *"PEARL sinh ít token nháp hơn nếu chúng sẽ bị bác bỏ, và sinh nhiều token nháp hơn nếu chúng sẽ được chấp nhận."*

---

## 3. Phân tích lý thuyết

**Định lý 1 (kích thước cửa sổ tối ưu):** kích thước cửa sổ tối ưu `γ*` bằng đúng tỷ số thời gian chạy hai mô hình:

```
γ*  =  c  =  T(M_p) / T(M_q)
```

Ý nghĩa thực dụng: `γ` không còn là siêu tham số phải dò tay — nó **đo được** bằng cách bấm giờ hai mô hình một lần. Đây là điểm khác biệt lớn so với SD gốc (Paper 1), nơi `K` phải tune thủ công cho từng cặp model và từng tác vụ.

**Định lý 2 (kỳ vọng số token chấp nhận):** với tỷ lệ chấp nhận `α` i.i.d.:

```
E[ số token chấp nhận ]  =  1 / (1 - α)  +  1
```

So với SD thuần:

```
E_vanilla  =  ( 1 - α^(γ+1) ) / (1 - α)
```

Vì `1/(1-α)` là giới hạn của `E_vanilla` khi `γ → ∞`, PEARL **luôn ≥ SD thuần với mọi `γ` hữu hạn**. Trực giác: SD thuần bị chặn trên bởi `γ` (không bao giờ ăn quá `γ+1` token một lượt), còn PEARL nhờ post-verify nối chuỗi nên **không có trần**.

---

## 4. Kết quả thực nghiệm

* **Phần cứng:** NVIDIA H800 (1 GPU cho model 7B, 2 GPU cho 34B, 3 GPU cho 70B), bfloat16.
* **Cặp mô hình:** Codellama-7B→34B/70B, Deepseek-1.3B/6.7B→33B, Llama-2-7B→70B.
* **Bộ dữ liệu:** HumanEval (164 bài), GSM8K (100 bài), MT-bench (80 bài).
* **Đối thủ:** tự hồi quy, SD thuần, Ouroboros, Lookahead Decoding, DistillSpec.

| Cặp mô hình      | Tăng tốc của PEARL |
| ------------------- | :-------------------: |
| Codellama 7B & 70B  |   **3.79×**   |
| Deepseek 1.3B & 33B |        3.48×        |
| Llama-2 7B & 70B    |        3.01×        |
| Deepseek 7B & 33B   |        2.75×        |
| Codellama 7B & 34B  |        2.35×        |

* Cao nhất **4.43×** so với tự hồi quy, và **1.50×** so với SD thuần.
* Trên GSM8K: 2.87× (SD thuần chỉ 1.99×). Trên MT-bench: 2.48× (SD thuần 1.63×).
* **Ablation:** post-verify đóng góp nhiều hơn pre-verify khi cặp mô hình khớp nhau tốt (vì `α` cao → chuỗi chấp nhận dài → post-verify thắng liên tục).
* **Lossless:** giữ nguyên phân phối, vì cả hai thủ thuật chỉ đổi *thời điểm* tính toán, không đổi *tiêu chí* chấp nhận.

---

## 5. Hạn chế — và đây là phần quan trọng nhất với đồ án

| Hạn chế                                                       | Chi tiết                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Đòi tài nguyên GPU gấp đôi (về mặt logic)**    | Muốn hai mô hình chạy*thật sự* đồng thời thì phải có hai luồng tính toán không tranh nhau — thực tế trong bài là **2–3 GPU H800**. Bài báo tự thừa nhận: *"vẫn là thách thức để lập lịch tài nguyên GPU sao cho tránh tranh chấp."* |
| **Trên 1 GPU, lợi ích co lại đáng kể**             | Nếu nhồi cả hai mô hình vào một GPU, chúng tranh SM và băng thông → "song song" biến thành chia sẻ thời gian, và phần lớn lợi ích bốc hơi.                                                                                                                    |
| **Chi phí bộ nhớ phụ**                                | Phải cache logits từ pha xác thực trước, tỷ lệ với độ dài nháp tối đa.                                                                                                                                                                                                |
| **Nháp phí khi đoán sai**                             | Post-verify sinh`γ` token có thể bị vứt. Vô hại khi GPU rảnh, nhưng **tốn thật** nếu tài nguyên đang căng (tức là chế độ batched).                                                                                                                      |
| **Là mô hình nháp rời (không phải feature-level)** | PEARL đặt trên kiến trúc SD cổ điển hai mô hình,**trực giao** với dòng EAGLE. Trên nguyên tắc có thể ghép PEARL với EAGLE, nhưng bài báo không làm.                                                                                                   |

> **Liên hệ đồ án — cảnh báo phần cứng:** ràng buộc của nhóm là **T4 16GB + mượn A100 một lần** (xem [02-tong-quan-tai-lieu.md](../02-tong-quan-tai-lieu.md)). PEARL giả định nhiều GPU H800 chạy song song. **Tái lập PEARL đúng nghĩa là không khả thi với phần cứng của nhóm.** Giá trị của paper này với đồ án là **ý tưởng** (trục lịch chạy còn bỏ ngỏ, và tính thích ứng có thể là *hệ quả* của thiết kế chứ không cần bộ dự đoán), chứ không phải là thứ để chạy lại.
