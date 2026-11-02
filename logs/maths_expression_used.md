# Tổng hợp các công thức toán học và biểu thức kỹ thuật trong đồ án D22

> **Đề tài:** Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference on Consumer GPUs  
> **Kho lưu trữ:** `do-an-ptit-d22`  
> **Mục đích:** Bảng tra cứu toàn diện tất cả các công thức toán học, mô hình chi phí, chỉ số phục vụ và hàm mất mát được sử dụng xuyên suốt từ cột mốc M0 đến M3.

---

## 1. Đo lường hiệu năng suy luận và chỉ số phục vụ (serving metrics)

### 1.1. Thời gian đến token đầu tiên (Time To First Token - TTFT)
* **Công thức:**
  $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_sent}} \approx T_{\text{prefill}}(L_{\text{prompt}})$$
* **Ý nghĩa:**
  * $t_{\text{request\_sent}}$: Thời điểm client gửi yêu cầu đến server suy luận.
  * $t_{\text{first\_token}}$: Thời điểm client nhận được token đầu tiên từ server.
  * $T_{\text{prefill}}$: Thời gian mô hình nạp và xử lý toàn bộ prompt có độ dài $L_{\text{prompt}}$ qua cơ chế self-attention.
* **Ngữ cảnh trong đồ án:** Áp dụng tại M1-02 và M2-04. Đây là chỉ số ràng buộc (constraint metric) — thuật toán nháp không được phép làm xấu đi TTFT.

---

### 1.2. Thời gian trung bình mỗi token sinh ra (Time Per Output Token - TPOT)
* **Công thức:**
  $$\text{TPOT} = \frac{T_{\text{end}} - T_{\text{first\_token}}}{N_{\text{gen}} - 1}$$
* **Ý nghĩa:**
  * $T_{\text{end}}$: Thời điểm nhận token kết thúc (hoặc token cuối cùng).
  * $N_{\text{gen}}$: Tổng số token sinh ra trong pha giải mã (decode phase).
* **Ngữ cảnh trong đồ án:** Là **mục tiêu tối ưu hóa chính** của toàn bộ đề tài. Speculative decoding hướng tới việc giảm thiểu TPOT ở pha giải mã.

---

### 1.3. Khoảng cách thời gian giữa hai token liên tiếp (Inter-Token Latency - ITL)
* **Công thức:**
  $$\text{ITL}_i = t_i - t_{i-1} \quad (\text{với } i = 2, \dots, N_{\text{gen}})$$
* **Ý nghĩa:** Đo lường độ trễ của từng token riêng lẻ để tính phân vị $p50$ và $p95$, phản ánh độ mượt mà của trải nghiệm người dùng thực tế.

---

### 1.4. Độ trễ tổng thể từ đầu đến cuối (End-to-End Latency - E2E)
* **Công thức:**
  $$\text{E2E} \approx \text{TTFT} + \text{TPOT} \times (N_{\text{gen}} - 1)$$
* **Ý nghĩa:** Tổng thời gian hoàn tất một yêu cầu suy luận đối với người dùng cuối.

---

### 1.5. Thông lượng hệ thống (System Throughput)
* **Công thức:**
  $$\text{Throughput} = \frac{\sum_{j=1}^{B} N_{\text{gen}, j}}{T_{\text{wall}}} \quad (\text{đơn vị: tokens/giây})$$
* **Ý nghĩa:**
  * $B$: Mức tải đồng thời (concurrency / batch size).
  * $N_{\text{gen}, j}$: Số token sinh ra của request thứ $j$.
  * $T_{\text{wall}}$: Tổng thời gian đồng hồ đo được từ khi bắt đầu request đầu tiên đến khi request cuối cùng kết thúc.

---

### 1.6. Tỉ lệ tăng tốc (Speedup vs. Vanilla)
* **Công thức:**
  $$\text{Speedup} = \frac{\text{Throughput}_{\text{Speculative}}}{\text{Throughput}_{\text{Vanilla}}} = \frac{\text{TPOT}_{\text{Vanilla}}}{\text{TPOT}_{\text{Speculative}}}$$
* **Ý nghĩa:** Tỉ lệ đo lường mức độ cải thiện tốc độ so với giải mã tự hồi quy truyền thống (Vanilla). Speedup $> 1.0\times$ là tăng tốc dương; Speedup $< 1.0\times$ là hiện tượng tăng tốc âm (negative acceleration).

---

## 2. Suy đoán thích ứng và bộ điều khiển ngân sách nháp (adaptive controller)

### 2.1. Tỉ lệ chấp nhận tức thời của từng bước nháp
* **Công thức:**
  $$\alpha_{\text{step}} = \frac{N_{\text{accepted}}}{N_{\text{drafted}}}$$
* **Ý nghĩa:**
  * $N_{\text{drafted}}$: Số token mà mô hình nháp đề xuất trong bước hiện tại ($N_{\text{drafted}} = k$).
  * $N_{\text{accepted}}$: Số token nháp được Target model chấp nhận ($0 \le N_{\text{accepted}} \le k$).
  * $\alpha_{\text{step}} \in [0.0, 1.0]$: Tỉ lệ sống sót của token nháp tại bước đó.

---

### 2.2. Tỉ lệ chấp nhận trung bình trượt mũ (Exponential Moving Average - EMA)
* **Công thức:**
  $$\alpha_{\text{EMA}}^{(t)} = \beta \cdot \alpha_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \alpha_{\text{step}}^{(t)}$$
* **Ý nghĩa:**
  * $\beta = 0.8$: Hệ số suy giảm (decay factor), gán trọng số 80% cho lịch sử và 20% cho bước tức thời.
  * $t$: Chỉ số bước suy luận trong một phiên.
* **Ngữ cảnh trong đồ án:** Được cài đặt trong lớp `BudgetController` (`src/proposer/controller.py`) để theo dõi diễn biến độ khó của văn bản.

---

### 2.3. Hiệu chỉnh khử lệch bước khởi đầu (Unbiased Warm-up Correction)
* **Công thức:**
  $$\alpha_{\text{unbiased}}^{(t)} = \frac{\alpha_{\text{EMA}}^{(t)} - \alpha_0 \cdot \beta^t}{1 - \beta^t}$$
* **Ý nghĩa:**
  * $\alpha_0 = 0.8$: Giá trị khởi tạo giả định ban đầu.
  * $t$: Số bước telemetry đã thu thập.
  * Giống như cơ chế bias correction trong thuật toán tối ưu hóa Adam, công thức này triệt tiêu ảnh hưởng của $\alpha_0$ khi $t < 10$, giúp phản ánh trung thực tỉ lệ chấp nhận ngay từ các bước giải mã đầu tiên.

---

### 2.4. Kỳ vọng số token sinh ra trong một bước suy đoán (Expected Accepted Tokens)
* **Công thức tổng đóng hình học:**
  $$\mathbb{E}[\tau(k)] = 1 + \sum_{i=1}^{k} \alpha^i = 1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha} \quad (\text{khi } \alpha \ne 1)$$
* **Ý nghĩa:**
  * $k$: Ngân sách token nháp đề xuất ($k \in \{0, 2, 3, 4\}$).
  * Số $1$ đại diện cho token chắc chắn thu được từ Target model.
  * Mỗi token thứ $i$ chỉ được chấp nhận nếu toàn bộ $i-1$ token trước đó đều được chấp nhận, do đó xác suất sống sót giảm theo quy luật lũy thừa $\alpha^i$.
  * Khi $\alpha = 1.0$: $\mathbb{E}[\tau] = 1 + k$.
  * Khi $\alpha = 0.0$: $\mathbb{E}[\tau] = 1.0$ (tương đương Vanilla).

---

### 2.5. Quy tắc chọn ngân sách cực đại hóa hiệu năng (Argmax Efficiency Rule)
* **Công thức:**
  $$k^* = \arg\max_{k \in \{0, 2, 3, 4\}} \frac{\mathbb{E}[\tau(k, \alpha)]}{C(k, B)}$$
* **Ý nghĩa:**
  * $C(k, B)$: Chi phí thời gian (ms) của một bước suy đoán với ngân sách $k$ ở batch size $B$.
  * Đại lượng $\frac{\mathbb{E}[\tau(k, \alpha)]}{C(k, B)}$ biểu thị **thông lượng token sinh ra trên mỗi mili-giây**. Bộ điều khiển sẽ chọn $k^*$ mang lại hiệu suất cao nhất.

---

### 2.6. Cơ chế thăm dò định kỳ chống bẫy đóng băng (Deadlock Fallback & Periodic Probing)
* **Quy tắc:**
  $$\text{Nếu } k = 0 \quad \text{và} \quad N_{\text{fallback\_steps}} \ge M_{\text{probe}} (20 \text{ bước}) \implies \text{Ép thử nghiệm } k = 2$$
* **Ý nghĩa:** Khi $k=0$, hệ thống không nháp token nào nên không có dữ liệu để cập nhật $\alpha$. Sau 20 bước, bộ điều khiển chủ động phát lệnh thăm dò $k=2$ để cập nhật lại $\alpha$. Nếu $\alpha$ đã vượt ngưỡng an toàn ($\ge 0.65$), hệ thống sẽ tự động thoát khỏi trạng thái fallback.

---

## 3. Mô hình chi phí phần cứng và phân tích hòa vốn (cost model & break-even)

### 3.1. Tổng thời gian một chu kỳ suy đoán Speculative Decoding
* **Công thức:**
  $$C(k, B) = k \times C_D(B) + C_V(k, B)$$
* **Ý nghĩa:**
  * $C_D(B)$: Thời gian forward của Draft model sinh 1 token ở batch size $B$.
  * $C_V(k, B)$: Thời gian xác thực song song $k$ token của Target model ở batch size $B$.
  * Khi $k=0$ (Vanilla): $C(0, B) = C_T(B)$ *(với runner vLLM có gắn model nháp, bước đồng bộ KV tốn $C(0, B) \approx C_T + C_D$)*.

---

### 3.2. Mô hình chi phí xác thực có hệ số suy biến Attention (Verification Overhead)
* **Công thức:**
  $$C_V(k, B) = C_T(B) \times \left(1 + \gamma \cdot (k - 1)\right)$$
* **Ý nghĩa:**
  * $C_T(B)$: Thời gian forward 1 token cơ sở của Target model.
  * $\gamma$: Hệ số gia tăng chi phí do tính toán attention song song trên chuỗi $k$ token. Trên GPU NVIDIA L4, thực nghiệm xác định $\gamma = 0.008$ (tương đương tăng $0.8\%$ mỗi token nháp bổ sung).

---

### 3.3. Điều kiện cần và đủ để đạt tăng tốc dương (Speedup > 1.0x)
* **Bất đẳng thức điều kiện:**
  $$\text{Speedup} = \frac{\mathbb{E}[\tau(k, \alpha)] \times C_{\text{vanilla}}(B)}{k \cdot C_D(B) + C_V(k, B)} > 1.0$$
  Biến đổi tương đương:
  $$\frac{C_D(B)}{C_T(B)} < \frac{\mathbb{E}[\tau(k, \alpha)] - 1}{k} = \bar{\alpha}$$
* **Ý nghĩa cốt lõi:** Tỉ số chi phí phần cứng giữa Draft model và Target model ($C_D / C_T$) bắt buộc phải nhỏ hơn tỉ lệ chấp nhận trung bình $\bar{\alpha}$ trên mỗi token nháp thì hệ thống mới có lời về mặt tốc độ.

---

### 3.4. Phương trình giải nghiệm điểm hòa vốn (Break-Even Acceptance Rate $\alpha^*$)
* **Phương trình:**
  $$\mathbb{E}[\tau(k, \alpha^*)] = 1 + \sum_{i=1}^{k} (\alpha^*)^i = \frac{C(k, B)}{C_{\text{vanilla}}(B)}$$
* **Thuật toán giải:** Do hàm số $f(\alpha) = 1 + \sum_{i=1}^k \alpha^i$ đồng biến ngặt trên đoạn $[0, 1]$, nghiệm $\alpha^*(k, B)$ được tìm bằng phương pháp chia đôi (Bisection root solver) trong module `src/proposer/cost_model.py`.
* **Số liệu thực nghiệm trên GPU NVIDIA L4 (tại $B=1$):**
  * $k = 2: \alpha^* \approx 0.3773$ (chỉ cần $\alpha \ge 37.7\%$ là bắt đầu có tăng tốc).
  * $k = 3: \alpha^* \approx 0.4657$ (chỉ cần $\alpha \ge 46.6\%$).
  * $k = 4: \alpha^* \approx 0.5325$ (chỉ cần $\alpha \ge 53.3\%$).

---

## 4. Tính bảo toàn lossless và kiểm chứng tính xác định

### 4.1. Lọc loại trừ có hiệu chỉnh (Modified Rejection Sampling khi $T > 0$)
* **Xác suất chấp nhận token nháp $x$:**
  $$P_{\text{accept}}(x) = \min\left(1, \frac{q(x)}{p(x)}\right)$$
* **Phân phối thặng dư khi token nháp bị từ chối:**
  $$q'(x) = \frac{\max\left(0, q(x) - p(x)\right)}{\sum_{x'} \max\left(0, q(x') - p(x')\right)}$$
* **Hệ quả bảo toàn phân phối:**
  $$P_{\text{final}}(x) = p(x) \cdot P_{\text{accept}}(x) + \left(1 - \sum_{x'} p(x') P_{\text{accept}}(x')\right) \cdot q'(x) \equiv q(x)$$
  Đảm bảo phân phối đầu ra của Speculative Decoding hoàn toàn trùng khớp với phân phối của Target model.

---

### 4.2. Bảo toàn Lossless ở chế độ giải mã tham lam (Greedy Decoding khi $T = 0.0$)
* **Điều kiện chấp nhận từng token:**
  $$\tilde{x}_i == \arg\max_{x} \text{logits}_{\text{target}}(x \mid x_{<i}, \tilde{x}_{<i})$$
* **Công thức kiểm chứng tỉ lệ trùng khớp token (Token Match Rate):**
  $$\text{TokenMatchRate} = \frac{\sum_{i=1}^{\max(L_V, L_A)} \mathbb{I}(V_i == A_i)}{\max(L_V, L_A)}$$
* **Tiêu chuẩn nghiệm thu:** $\text{TokenMatchRate} = 1.0 \ (100\%)$ và số lượng sai lệch $\text{DiscrepancyCount} = 0$.

---

## 5. Mô hình bộ nhớ phần cứng VRAM và KV-Cache

### 5.1. Cân bằng ngân sách bộ nhớ GPU (VRAM Budget Model)
* **Công thức:**
  $$VRAM_{\text{total}} = VRAM_{\text{weights}} + VRAM_{\text{kv\_cache}} + VRAM_{\text{activation}} + VRAM_{\text{cuda\_ctx}}$$
* **Dữ liệu thực nghiệm trên Tesla T4 (15,360 MiB):**
  * $VRAM_{\text{weights}}$: Qwen2.5-7B AWQ Marlin chiếm $\approx 5.29\text{ GiB}$ ($5,420\text{ MiB}$).
  * Draft Qwen2.5-0.5B chiếm $\approx 0.98\text{ GiB}$ ($1,004\text{ MiB}$).
  * Đỉnh an toàn: $\le 13,820\text{ MiB}$ (dưới trần an toàn $15,000\text{ MiB}$, chống OOM).

---

### 5.2. Kích thước bộ nhớ đệm KV-Cache cho kiến trúc GQA (Grouped-Query Attention)
* **Công thức dung lượng trên mỗi token:**
  $$\text{KV\_Bytes}_{\text{per\_token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
* **Tính toán cụ thể cho Qwen2.5-7B:**
  * $n_{\text{layers}} = 28$
  * $n_{\text{kv\_heads}} = 4$
  * $d_{\text{head}} = 128$
  * $\text{bytes\_per\_elem} = 2$ (định dạng FP16 / BF16)
  $$\text{KV\_Bytes}_{\text{per\_token}} = 2 \times 28 \times 4 \times 128 \times 2 = 57,344\text{ bytes} \approx 56\text{ KB/token}$$

---

### 5.3. Quy tắc áp lực bộ nhớ kép và trễ phục hồi (Hysteresis Memory Guard)
* **Điều kiện kích hoạt ngắt nháp về $k=0$:**
  $$KV_{\text{usage}} > 90.0\% \quad \text{HOẶC} \quad VRAM_{\text{free}} < 1500\text{ MB}$$
* **Điều kiện trễ cho phép bật lại nháp (Hysteresis Recovery):**
  $$KV_{\text{usage}} < 65.0\% \quad \text{và duy trì trạng thái an toàn tối thiểu } N_{\text{cooldown}} = 3\text{ bước}$$

---

## 6. Phương pháp luận thực nghiệm và kiểm soát nhiễu

### 6.1. Triệt tiêu thời gian Prefill để đo độ trễ Decode thuần
* **Công thức:**
  $$step\_ms = \frac{T(1 + N) - T(1)}{N} \quad (\text{với } N = 64\text{ tokens})$$
* **Ý nghĩa:**
  * $T(1)$: Wall-clock time sinh 1 token (gồm prefill + 1 decode step).
  * $T(1+N)$: Wall-clock time sinh $1+N$ tokens (gồm prefill + $1+N$ decode steps).
  * Hiệu số triệt tiêu hoàn toàn thành phần $T_{\text{prefill}}$, cho ra độ trễ thực tế của từng bước decode phần cứng.

---

### 6.2. Kiểm soát độ phân tán dữ liệu thực nghiệm (Variance Ratio - Protocol Option B)
* **Công thức:**
  $$\text{Variance Ratio} = \frac{\text{TPOT}_{\max} - \text{TPOT}_{\min}}{\text{TPOT}_{p50}}$$
* **Tiêu chuẩn loại bỏ nhiễu:**
  $$\text{Variance Ratio} \le 15.0\% \quad (0.15)$$
* Nếu $\text{Variance Ratio} > 15\%$, hệ thống tự động cắm cờ `Variance Flag = True` và yêu cầu đo lặp lại để loại bỏ ảnh hưởng của nghẽn nhiệt GPU hoặc tác vụ nền hệ thống.

---

## 7. Hàm mất mát huấn luyện mô hình nháp EAGLE-3.1 Speculator

### 7.1. Hàm mất mát kết hợp (Dual Loss Function)
* **Công thức:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{SmoothL1}} \quad (\text{với } \lambda = 0.5)$$

---

### 7.2. Thành phần Cross-Entropy phân loại token (Token Classification Loss)
* **Công thức:**
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{K} \sum_{t=1}^{K} \log P_{\text{draft}}\left(x_{t+1} \mid h_t, x_{\le t}\right)$$
* **Ý nghĩa:** Đánh giá độ chính xác trong việc dự đoán nhãn token tiếp theo của đầu nháp so với ground truth của mô hình gốc.

---

### 7.3. Thành phần hồi quy biểu diễn ẩn (Hidden State Regression Loss)
* **Công thức Smooth L1 Loss:**
  $$\mathcal{L}_{\text{SmoothL1}}\left(h_{\text{pred}}, h_{\text{target}}\right) = \frac{1}{d} \sum_{j=1}^{d} z_j$$
  Trong đó:
  $$z_j = \begin{cases} 0.5 \left(h_{\text{pred}, j} - h_{\text{target}, j}\right)^2 & \text{nếu } \left|h_{\text{pred}, j} - h_{\text{target}, j}\right| < 1.0 \\ \left|h_{\text{pred}, j} - h_{\text{target}, j}\right| - 0.5 & \text{ngược lại} \end{cases}$$
* **Ý nghĩa:** Ép vector đặc trưng ẩn của đầu nháp (EAGLE-3.1 Head) bám sát theo vector ẩn của tầng trên cùng mô hình Target (7B), giúp đầu nháp tận dụng được tri thức biểu diễn ngữ nghĩa của mô hình lớn mà không bị bùng nổ gradient.
