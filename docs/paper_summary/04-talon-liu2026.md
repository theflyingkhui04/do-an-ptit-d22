# Tóm tắt Paper 4: TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees

* **Tác giả:** Tianyu Liu, Qitan Lv, Yuhao Shen, Xiao Sun, Xiaoyan Sun (USTC, Shanghai AI Laboratory, Zhejiang University)
* **Nơi công bố:** arXiv, 12/01/2026 — `arXiv:2601.07353` (đang bình duyệt, có trang OpenReview)
* **Link PDF:** https://arxiv.org/pdf/2601.07353
* **Link arXiv:** https://arxiv.org/abs/2601.07353

> **Lưu ý khi trích dẫn trong đồ án:** đây là preprint 2026, **chưa qua peer-review**. Nên xếp vào nhóm "state-of-the-art mới nhất" nhưng khi bảo vệ phải nói rõ trạng thái này, và ưu tiên EAGLE-2/EAGLE-3 làm anchor chính thức.

---

## 1. Vấn đề: "động" của EAGLE-2 vẫn chưa đủ động

EAGLE-2 (Paper 3) đã cho cây nháp thích ứng theo ngữ cảnh, nhưng Talon chỉ ra rằng cây của EAGLE-2/EAGLE-3 **vẫn bị đóng khung bởi hai con số cứng**:
* Độ sâu cố định (`depth = 6`).
* Độ rộng cố định ở mỗi tầng (`top-k = 10` node được mở rộng).

Hệ quả: cấu hình cây bị **ràng buộc chéo** — muốn đi sâu hơn thì phải trả giá bằng chiều rộng và ngược lại, nhưng tỷ lệ đánh đổi này bị ấn định trước cho **mọi token trong mọi câu**.

Điều Talon muốn mà cây fixed-width/fixed-depth không làm được:
* **Dừng sớm (early stop)** khi gặp token khó — đừng cố đâm xuống tầng 5, 6 trong khi tầng 2 đã hết hy vọng.
* **Kéo dài (extend)** khi gặp đoạn dễ — đáng lẽ đi sâu 12 tầng được thì lại bị chặn ở 6.

---

## 2. Ý tưởng lõi: đổi trục điều khiển từ "hình dạng" sang "ngân sách"

Đây là dịch chuyển tư duy chính của bài báo:

> Đừng khai báo cây trông như thế nào (rộng bao nhiêu, sâu bao nhiêu). Chỉ khai báo **một ngân sách token duy nhất `N`**, rồi để độ tự tin quyết định ngân sách đó được tiêu vào chiều sâu hay chiều rộng.

Lý do ngân sách `N` mới là đại lượng đáng quan tâm: **chi phí xác thực của mô hình lớn tỷ lệ với tổng số token đưa vào**, chứ không quan tâm chúng xếp thành hình gì. Vậy nên cố định `N` là cố định đúng thứ ta thực sự phải trả tiền, còn hình dạng cây thì thả tự do.

* **Ngữ cảnh xác định (deterministic):** ngân sách dồn theo chiều dọc → cây **sâu và hẹp (deep-and-narrow)**.
* **Ngữ cảnh bất định (uncertain):** ngân sách trải theo chiều ngang → cây **nông và rộng (shallow-and-wide)**.

Và tất cả đều **training-free**, cắm thẳng vào EAGLE-2/EAGLE-3 như một plug-in thay thế phần xây cây.

---

## 3. Cơ chế chi tiết

### 3.1. Điểm tự tin tích lũy của một node

Giống EAGLE-2, Talon dùng **xác suất tích lũy dọc đường đi** làm điểm số của node `u` có cha là `v`:

```
p(u)  =  p(v)  ×  P_Md( w | x_≤v )
```

trong đó `P_Md(·)` là phân phối của mô hình nháp, `w` là token tại node `u`. Node gốc có `p = 1`.

### 3.2. Chiến lược mở rộng lai (Hybrid Expansion)

Điểm tinh tế nhất của Talon: **tầng gốc và các tầng sâu được đối xử khác nhau**.

**Tầng 0 (gốc) — khởi tạo an toàn bằng top-K cứng:**
```
P_1  =  { (x_t, v)  |  v ∈ Top-K( P_Md( · | x_t ) ) }
```
Ở tầng đầu tiên, Talon **cố tình không lọc theo độ tự tin**, mà lấy thẳng top-K (mặc định `K = 10`, giữ nguyên như EAGLE).

*Vì sao?* Vì bộ nháp hay bị **miscalibrated** ngay tại bước đầu tiên: nó có thể tự tin sai lè ở token đầu. Nếu áp ngưỡng lọc ngay từ tầng 0, ta có nguy cơ vứt bỏ token đúng ngay từ đầu và **hỏng toàn bộ lượt**. Đây là bước "bảo hiểm" chống lại chính điểm yếu mà EAGLE-2 để lộ.

**Tầng d ≥ 1 — mở rộng có cổng chặn theo độ tự tin (confidence-gated):**

Gọi `S_d` là tập ứng viên ở tầng `d`. Lấy mức tự tin neo (anchor) là node mạnh nhất tầng đó:
```
m_d  =  max [ u ∈ S_d ]  p(u)
```

Chỉ giữ lại những ứng viên đủ mạnh **tương đối so với node mạnh nhất cùng tầng**:
```
P_(d+1)  =  { u ∈ S_d  |  p(u)  ≥  μ · m_d }
```
với ngưỡng tương đối `μ = 0.03` (cố định cho mọi thí nghiệm).

**Vì sao dùng ngưỡng *tương đối* mà không dùng ngưỡng tuyệt đối?** Vì xác suất tích lũy `p(u)` tụt rất nhanh theo độ sâu (tích của các số < 1). Một ngưỡng tuyệt đối kiểu `p ≥ 0.01` sẽ tự động giết sạch mọi node từ tầng 4 trở xuống, làm cây không bao giờ sâu được. Ngưỡng tương đối `μ · m_d` tự chuẩn hóa lại theo từng tầng → **cây được phép sâu tùy ý miễn là các nhánh còn cạnh tranh sát nhau**. Chính công thức này là thứ tạo ra hành vi "early stop khi khó, extend khi dễ":
* Nếu ở tầng `d` chỉ có 1 node vượt trội hẳn → tập lọc còn đúng 1 node → cây **thu hẹp và đi sâu**.
* Nếu nhiều node ngang ngửa nhau → nhiều node cùng vượt ngưỡng → cây **bung rộng**.
* Nếu mọi node đều đã rất yếu và phân tán → nhánh cạn kiệt, dừng lại tự nhiên.

### 3.3. Vòng lặp theo ngân sách và cắt cuối

Cây được xây **lặp theo tầng cho tới khi tiêu hết ngân sách**, không có giới hạn độ sâu khai báo trước:
```
while  |T| < N  do
    xây tầng kế tiếp theo quy tắc 3.2
```

Khi tầng kế tiếp làm tràn ngân sách, cắt bớt bằng chính điểm tích lũy:
```
if  |T| + |P_(d+1)|  >  N  then
    P_(d+1)  ←  Top-( N - |T| ) ( P_(d+1)  theo  p(u) )
```

**Siêu tham số:** `N = 60` (ngân sách token, chọn để so sánh công bằng với EAGLE), `μ = 0.03`, `K = 10`.
Đáng chú ý: chỉ có **một** siêu tham số thật sự mới (`μ`), và nó được giữ **cố định trên toàn bộ 5 mô hình × 6 bộ dữ liệu** — một luận cứ mạnh cho tính bền vững, không phải tune riêng từng ca.

```
Ngữ cảnh DỄ (p tập trung)          Ngữ cảnh KHÓ (p trải đều)
        ●                                    ●
        │                          ┌───┬───┼───┬───┐
        ●                          ●   ●   ●   ●   ●
        │                        ┌─┴─┐   ┌─┴─┐
        ●                        ●   ●   ●   ●
        │
        ●          deep & narrow            shallow & wide
        │
        ●          ── cùng tiêu đúng N = 60 token ──
```

---

## 4. Kết quả thực nghiệm

* **Thiết lập:** 5 mô hình × 6 bộ dữ liệu (Alpaca, GSM8K, HumanEval, MT-Bench, QA, CNN/DM).
* **Đối thủ chính:** **EAGLE-3** (bản mạnh nhất của dòng EAGLE tính tới thời điểm đó).

| Mô hình | Bộ dữ liệu | Talon | EAGLE-3 | MAT của Talon |
|---|---|:--:|:--:|:--:|
| Vicuna-13B | HumanEval | **5.16×** | 4.77× | 9.48 |
| Llama3-8B | Alpaca | **4.04×** | 3.92× | 6.51 |
| Qwen3-8B | GSM8K | **2.67×** | 2.37× | 3.85 |

*(MAT = Mean Accepted Tokens, số token trung bình chấp nhận được mỗi lượt gọi mô hình lớn.)*

* Tăng tốc cao nhất **5.16×** so với giải mã tự hồi quy.
* Vượt EAGLE-3 **một cách nhất quán** trên toàn bộ cấu hình được báo cáo.
* Vẫn **lossless** (không đụng vào tiêu chí chấp nhận).

**Đọc bảng một cách tỉnh táo:** mức cải thiện so với EAGLE-3 là **khoảng 3% – 13%**, không phải bước nhảy vọt. Giá trị thật của Talon nằm ở chỗ *lấy được phần cải thiện đó mà không tốn một đồng huấn luyện nào và chỉ thêm đúng một siêu tham số* — chứ không phải ở con số 5.16× (con số đó chủ yếu là công của EAGLE-3 bên dưới).

---

## 5. Hạn chế & liên hệ với đồ án

* **Chưa peer-review** (preprint 01/2026) → dùng làm tham chiếu, không làm nền móng.
* **Vẫn là thế giới batch = 1.** Toàn bộ lập luận "ngân sách N là thứ ta phải trả" chỉ đúng khi GPU còn memory-bound. Khi phục vụ nhiều request đồng thời, `N = 60` cho **mỗi** request lập tức trở thành chi phí compute nặng → đúng chỗ **Paper 6 (D-cut)** nhảy vào.
* **Ngân sách `N` vẫn do người dùng đặt tay.** Talon giải phóng *hình dạng* cây nhưng không giải phóng *kích thước* cây; `N` tối ưu vẫn phụ thuộc phần cứng và độ dài câu.
* **Chi phí điều khiển (control overhead):** việc xây cây theo vòng lặp có điều kiện, số node mỗi tầng thay đổi mỗi bước → kích thước tensor không cố định, **khó bắt CUDA graph**, dễ mất một phần lợi ích khi triển khai thật. Bài báo đo bằng speedup end-to-end nên chi phí này đã nằm trong số liệu, nhưng nó cảnh báo rằng hiệu quả có thể khác trên engine khác (vLLM/TensorRT-LLM).

> **Liên hệ đồ án:** Talon là ví dụ mẫu mực cho kiểu đóng góp mà nhóm nên nhắm tới — một **plug-in training-free, ít siêu tham số, cắm lên anchor có sẵn, đo bằng speedup + MAT**. Đồng thời nó cũng **chiếm mất** ý tưởng "cây động theo confidence có ngân sách" → biến tấu của đồ án phải đi xa hơn điểm này (xem [03-quyet-dinh-thiet-ke.md](../03-quyet-dinh-thiet-ke.md)).
