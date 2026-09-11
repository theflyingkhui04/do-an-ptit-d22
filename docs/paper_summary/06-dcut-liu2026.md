# Tóm tắt Paper 6: D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding

* **Tác giả:** Tianyu Liu, Yuhao Shen, Rui Cen, Junhan Shi, Jiebin Zhang, Guangshuo Qin, Hong Liu, Song Liu, Guanghua Yu, Jianchen Zhu (Tencent Hunyuan)
* **Nơi công bố:** arXiv, 16/07/2026 — `arXiv:2607.14647`
* **Link PDF:** https://arxiv.org/pdf/2607.14647
* **Link arXiv:** https://arxiv.org/abs/2607.14647
* **Mã nguồn:** đã gửi upstream vào vLLM (`github.com/vllm-project/vllm` PR #47131)

> **Lưu ý khi trích dẫn:** preprint 07/2026, **chưa peer-review**. Nhưng đây là paper "đến từ nhà máy" — làm bởi đội Tencent Hunyuan, đo trên vLLM thật, phần cứng thật, và đã có PR upstream. Về mặt độ tin cậy kỹ thuật thì đáng giá hơn nhiều so với một preprint học thuật thuần.

---

## 1. Vấn đề: Speculative Decoding có thể **chậm hơn** giải mã thường

Đây là paper duy nhất trong 6 bài dám nói thẳng cái sự thật khó chịu:

> Ở chế độ phục vụ nhiều request đồng thời (high concurrency), speculative decoding **thường xuyên tụt xuống dưới 1.0×** — tức là chậm hơn cả giải mã tự hồi quy thường.

### Vì sao cơ chế đúng ở batch = 1 lại hỏng ở batch lớn?

Toàn bộ lý lẽ của Paper 1 dựa trên: GPU đang **memory-bound**, phần tính toán rảnh rỗi, nên xác thực thêm `K` token gần như **miễn phí**.

Giả định đó **chết ngay khi batch tăng lên**. Với batch `B`, GPU đã có đủ việc để làm → chuyển sang **compute-bound**. Lúc này mỗi token nháp đưa vào xác thực là một chi phí tính toán **thật**, cộng dồn qua cả batch:
```
Tổng token cần xác thực mỗi bước  =  B × (D + 1)
```
Với `B = 32` và độ dài nháp `D = 15`, đó là **512 token mỗi bước** — bằng cả một prefill nhỏ, cho mỗi bước decode.

Tệ hơn: phần lớn số đó là **rác sẽ bị bác bỏ**. Số liệu trong bài: baseline DFlash(16) chấp nhận trung bình ~**5.81** token, tức là **hơn 10 trong 16 vị trí bị vứt**, nhưng vẫn bị tính tiền đầy đủ. Kết quả đo được ở concurrency 32:
* Llama-3.1-8B: speedup chỉ **0.67×** (chậm hơn AR 33%).
* Qwen3-8B: **0.80×**.

---

## 2. Ý tưởng lõi: cắt tỉa độ sâu xác thực **theo cả batch**, chứ không theo từng request

Mọi phương pháp trước đó (EAGLE-2, Talon) quyết định nháp **độc lập cho từng request**. D-cut chỉ ra rằng ở chế độ batched, đó là góc nhìn sai:

> Ngân sách xác thực là **tài nguyên dùng chung của cả batch**. Nếu request A đang ở đoạn văn dễ (nháp tự tin) còn request B đang bí, thì nên **lấy ngân sách của B đưa cho A**, thay vì cho mỗi bên một suất bằng nhau.

Hai thành phần:
1. **Cross-request pruning:** xếp hạng token nháp **xuyên suốt toàn batch**, giữ lại những vị trí có triển vọng nhất bất kể chúng thuộc request nào.
2. **Environment-adaptive verification:** dùng **mô hình chi phí đo tại runtime** để quyết định nên cắt mạnh tới đâu — vì câu trả lời tối ưu phụ thuộc GPU, kiến trúc model và cấu hình tensor-parallel.

---

## 3. Cơ chế chi tiết

### 3.1. Điểm sống sót của tiền tố (Prefix Survival Score)

Bộ nháp trả về điểm tự tin `c_(i,t) ∈ [0,1]` cho token nháp ở vị trí `t` của request `i`. Xác suất để **cả tiền tố dài `k`** sống sót là tích dồn:
```
s_(i,k)  =  ∏ [ t = 1 → k ]  c_(i,t)        ,   s_(i,0) = 1
```
(`s_(i,0) = 1` ứng với **bonus token** — token luôn được đảm bảo, nên luôn đáng giữ.)

Kỳ vọng số token tiến được nếu giữ độ sâu `n_i` cho request `i`:
```
E[ A_i(n_i) ]  =  ∑ [ k = 0 → n_i ]  s_(i,k)
```
Cách đọc quan trọng: `s_(i,k)` chính là **lợi ích biên** của việc xác thực thêm vị trí thứ `k`. Nó cho ta một đơn vị so sánh chung giữa các request.

### 3.2. Xếp hạng toàn cục & tính chất tiền tố liên tục

D-cut trải phẳng toàn bộ `B × (D+1)` vị trí ứng viên của cả batch vào **một danh sách chung**, sắp xếp theo `s_(i,k)` giảm dần, giữ top-K.

**Vì sao phép cắt này luôn hợp lệ?** Vì trong mỗi request, `s_(i,k)` **giảm đơn điệu theo `k`** (nhân thêm số ≤ 1). Do đó nếu vị trí `k` được chọn thì mọi vị trí `< k` của cùng request cũng được chọn → tập top-K **tự động tạo thành các tiền tố liên tục**, không bao giờ có "lỗ hổng" giữa chuỗi. (Cùng một mẹo toán học mà EAGLE-2 dùng để đảm bảo cây liên thông — ở đây áp lên trục batch.)

Độ sâu giữ lại của từng request rơi ra tự nhiên:
```
n_i  =  max( { k : (i,k) ∈ S_K }  ∪  {0} )
```

### 3.3. Mô hình chi phí runtime — phần "hardware-aware" thật sự

Đây là phần khiến D-cut khác hẳn các paper thuần thuật toán.

**Lúc khởi động**, D-cut đo thực nghiệm độ trễ một bước speculative `C(B, ρ)` cho từng cặp (kích thước batch, tỷ lệ giữ lại). Tỷ lệ bị giới hạn vào tập rời rạc:
```
R  =  { 0.25,  0.50,  0.75,  1.00 }
```
*Vì sao chỉ 4 mức mà không phải liên tục?* Để **giữ tương thích CUDA-graph**. Nếu kích thước tensor thay đổi tùy tiện mỗi bước, engine phải re-capture graph và chi phí đó ăn hết phần lợi. Đây là một đánh đổi kỹ thuật rất "đời" mà các paper học thuật hay bỏ qua.

Sức chứa ngân sách ứng với tỷ lệ `ρ` (luôn đảm bảo mỗi request giữ ít nhất 1 token):
```
K_ρ(B)  =  max( B,  ⌈ ρ · B · (D+1) ⌉ )
```

**Quyết định cắt tỉa** = tối đa hóa tỷ số lợi ích/chi phí:
```
                ∑ [ (i,k) ∈ S_(K_ρ(B)) ]  s_(i,k)
ρ*  =  arg max  ─────────────────────────────────
       ρ ∈ R              C(B, ρ)
```
* **Tử số** = lợi ích thuật toán (kỳ vọng token ăn được).
* **Mẫu số** = chi phí phần cứng thật (đã đo, không phải ước lượng lý thuyết).

Đây chính là điểm mấu chốt: D-cut **không tối ưu acceptance rate, nó tối ưu throughput**. Và nó tự điều chỉnh theo môi trường mà không cần người dùng tune.

### 3.4. Thuật toán một bước giải mã

```
1.  Nhận token nháp + điểm tự tin từ bộ nháp
2.  Tính điểm tiền tố  s_(i,k)  cho mọi vị trí trong batch
3.  Với mỗi  ρ ∈ {0.25, 0.5, 0.75, 1.0}:  tính tỷ số lợi ích / chi phí
4.  Chọn  ρ*  có tỷ số cao nhất
5.  Suy ra độ sâu giữ lại  n_i  cho từng request
6.  Đóng gói các tiền tố còn lại  →  đưa mô hình lớn xác thực
```

**Siêu tham số:** độ dài nháp `D = 15` (khối xác thực 16 vị trí), batch tối đa 64, tối đa 32.768 token/bước.
**Training-free** hoàn toàn — chỉ là một tầng lập lịch đặt giữa bộ nháp và bộ xác thực.

---

## 4. Kết quả thực nghiệm

* **Phần cứng:** 1 node × 8 GPU **NVIDIA H20** (kết quả chính); 8 × **H800** cho nghiên cứu chéo phần cứng.
* **Engine:** **vLLM** + FlashAttention + piecewise CUDA-graph, tensor-parallel TP=1/4/8 theo cỡ model.
* **Mô hình đích:** dense (Llama-3.1-8B, Qwen3-4B/8B, Qwen3.5-27B, Qwen3.5-35B-A3B) và MoE (Hy3-295B-A21B).
* **Bộ dữ liệu:** GSM8K, Math500, HumanEval, MBPP, MT-Bench (256 prompt, ≤2048 token ra, greedy).
* **Concurrency:** 4, 8, 16, 32, 64 — kết quả chính báo cáo ở **32**.

### Kết quả chính (concurrency 32)

| Cấu hình | Baseline DFlash(16) | D-cut(16) |
|---|:--:|:--:|
| Trung bình toàn bộ | 1.26× | **1.65×** |
| Llama-3.1-8B | 0.67× (chậm hơn AR) | **1.12×** |
| Qwen3-8B | 0.80× (chậm hơn AR) | **1.21×** |
| MoE Hy3-295B-A21B | — | tới **3.0×** |

**Điểm đáng chú ý nhất:** phần cải thiện lớn nhất rơi đúng vào những chỗ baseline **đang thua giải mã thường**. D-cut không phải là thứ làm nhanh-hơn-nữa; nó là thứ **cứu speculative decoding khỏi việc trở thành gánh nặng**.

### Tính thích ứng theo phần cứng (Bảng 2, Qwen3-8B)

Không có một tỷ lệ cắt cố định nào tốt trên mọi phần cứng:
* Trên **H20** (nghèo compute): mức cắt mạnh `ρ = 0.25` là tốt nhất → 2.564 token/s (MT-Bench, bs=64).
* Trên **H800** (dồi dào compute): mức `ρ = 0.50` thường thắng → 10.824 token/s (GSM8K, bs=64).
* **D-cut (auto)** bám sát hoặc vượt mức tốt nhất cố định: **11.194 token/s** trên H800-GSM8K-bs64 (vượt cả mức fixed tốt nhất).
* Lợi ích trung bình so với DFlash: **1.58×** trên H20, **1.25×** trên H800 — không cần chọn tỷ lệ tay cho từng deployment.

→ Đây là **luận cứ thực nghiệm cho tính "hardware-aware"**: cùng một thuật toán, cùng một model, đổi GPU thì cấu hình tối ưu đổi theo. Bất kỳ phương pháp nào ấn định tham số cứng đều sai ở một nửa số môi trường.

### Độ bền với sampling

Ở nhiệt độ `temperature = 1` (thay vì greedy), baseline DFlash sụp xuống **0.68×** trung bình. D-cut giữ được **1.15×** — cải thiện tương đối **1.68×** — nhờ cắt mạnh hơn khi phần đuôi nháp trở nên ít giá trị.

---

## 5. Hạn chế & liên hệ với đồ án

| Hạn chế | Chi tiết |
|---|---|
| **Phụ thuộc độ chính xác của bảng chi phí** | Nếu profile lúc khởi động không khớp tải thực tế (request dài ngắn lệch nhau, batch biến động mạnh), lựa chọn `ρ*` sẽ lệch. |
| **Chỉ 4 mức cắt rời rạc** | Đánh đổi để giữ CUDA-graph; ở batch nhỏ, bước nhảy giữa 0.25 và 0.50 khá thô. |
| **Gắn với bộ nháp cho ra điểm tự tin theo khối** | Thí nghiệm chính dùng drafter block-diffusion DFlash; ghép với cây nháp kiểu EAGLE/Talon cần thêm việc (cây thì "độ sâu" không phải khái niệm một chiều). |
| **Chưa peer-review** | preprint 07/2026. |
| **Cần phần cứng cỡ lớn để tái lập** | 8× H20 / 8× H800. |

> **Liên hệ đồ án — đây là paper quan trọng nhất trong 6 bài đối với phần "twist hardware-aware":**
> 1. Nó **hợp pháp hóa hướng đi của nhóm**: chứng minh bằng số rằng "chọn tham số theo phần cứng đo được" là bài toán thật, chưa được giải xong, và cho cải thiện lớn (0.67× → 1.12×).
> 2. Nó cho **khuôn mẫu phương pháp có thể bắt chước ở quy mô nhỏ**: profile chi phí lúc khởi động → tối đa hóa tỷ số (lợi ích thuật toán / độ trễ đo thật) → chọn cấu hình. Khuôn mẫu này áp được lên **T4 16GB** y hệt như lên H800, chỉ khác con số.
> 3. Nó **cảnh báo về việc đo đạc**: nếu nhóm chỉ đo ở batch = 1 thì đang đo trong điều kiện dễ nhất và kết luận sẽ không đại diện cho phục vụ thật. Nên báo cáo speedup theo **thang concurrency**, không phải một con số.
> 4. Nó cũng **thu hẹp khoảng trống đóng góp**: ý tưởng "cắt độ sâu xác thực theo tải" đã bị chiếm. Biến tấu của nhóm cần khác về trục (ví dụ: chọn *ngân sách cây* theo capacity VRAM/băng thông đo được trên GPU phổ thông, ở chế độ batch nhỏ mà D-cut không nhắm tới).
