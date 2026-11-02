# Tổng hợp các công thức toán học và biểu thức kỹ thuật trong đồ án D22

> **Đề tài:** Hardware-Aware Speculative Decoding for Self-Hosted LLM Inference on Consumer GPUs  
> **Kho lưu trữ:** `do-an-ptit-d22`  
> **Mục đích:** Bảng tra cứu toàn diện tất cả các công thức toán học, mô hình chi phí, chỉ số phục vụ và hàm mất mát kèm **tham chiếu task Trello chi tiết** xuyên suốt từ cột mốc M0 đến M3.

---

## 1. Đo lường hiệu năng suy luận và chỉ số phục vụ (serving metrics)

### 1.1. Thời gian đến token đầu tiên (Time To First Token - TTFT)
* **Công thức:**
  $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_sent}} \approx T_{\text{prefill}}(L_{\text{prompt}})$$
* **Ý nghĩa:**
  * $t_{\text{request\_sent}}$: Thời điểm client gửi yêu cầu đến server suy luận.
  * $t_{\text{first\_token}}$: Thời điểm client nhận được token đầu tiên từ server.
  * $T_{\text{prefill}}$: Thời gian mô hình nạp và xử lý toàn bộ prompt có độ dài $L_{\text{prompt}}$ qua cơ chế self-attention.
* **Tham chiếu task:**
  * **Task Trello:** `Task #3 [M1: Baseline Benchmark]` (Đại Lead, Khôi Partner) và `Task #26 [M2-04: Mở rộng Benchmark Harness 3 Làn]` (Đại Lead 60%, Dương 40%).
  * **File triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py).
  * **Vai trò:** Chỉ số ràng buộc (constraint metric) — speculative decoding chỉ tối ưu pha decode, tuyệt đối không được làm xấu đi TTFT của pha prefill.

---

### 1.2. Thời gian trung bình mỗi token sinh ra (Time Per Output Token - TPOT)
* **Công thức:**
  $$\text{TPOT} = \frac{T_{\text{end}} - T_{\text{first\_token}}}{N_{\text{gen}} - 1}$$
* **Ý nghĩa:**
  * $T_{\text{end}}$: Thời điểm nhận token kết thúc (hoặc token cuối cùng).
  * $N_{\text{gen}}$: Tổng số token sinh ra trong pha giải mã (decode phase).
* **Tham chiếu task:**
  * **Task Trello:** `Task #3 [M1]`, `Task #26 [M2-04]`, `Task #35 [M3-05: Đo kiểm 4 Làn]` (Đại Lead 70%, Khôi 30%).
  * **File triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py).
  * **Vai trò:** **Mục tiêu tối ưu hóa chính** của toàn bộ đề tài. Giảm TPOT đồng nghĩa với việc tăng tốc độ phản hồi văn bản của LLM.

---

### 1.3. Khoảng cách thời gian giữa hai token liên tiếp (Inter-Token Latency - ITL)
* **Công thức:**
  $$\text{ITL}_i = t_i - t_{i-1} \quad (\text{với } i = 2, \dots, N_{\text{gen}})$$
* **Ý nghĩa:** Đo lường độ trễ của từng token riêng lẻ để tính phân vị $p50$ và $p95$, phản ánh độ mượt mà của trải nghiệm người dùng thực tế.
* **Tham chiếu task:**
  * **Task Trello:** `Task #3 [M1]`, `Task #26 [M2-04]`.
  * **File triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py).
  * **Nhật ký:** [`logs/daijestic/M1-02/README.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/daijestic/M1-02/README.md).

---

### 1.4. Độ trễ tổng thể từ đầu đến cuối (End-to-End Latency - E2E)
* **Công thức:**
  $$\text{E2E} \approx \text{TTFT} + \text{TPOT} \times (N_{\text{gen}} - 1)$$
* **Ý nghĩa:** Tổng thời gian hoàn tất một yêu cầu suy luận đối với ứng dụng người dùng cuối.
* **Tham chiếu task:**
  * **Task Trello:** `Task #3 [M1]`, `Task #31 [M3-01: Client Code Pipeline]` (Dương Lead 100%).
  * **File triển khai:** [`src/client/client.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/client/client.py).

---

### 1.5. Thông lượng hệ thống (System Throughput)
* **Công thức:**
  $$\text{Throughput} = \frac{\sum_{j=1}^{B} N_{\text{gen}, j}}{T_{\text{wall}}} \quad (\text{đơn vị: tokens/giây})$$
* **Ý nghĩa:**
  * $B$: Mức tải đồng thời (concurrency / batch size).
  * $N_{\text{gen}, j}$: Số token sinh ra của request thứ $j$.
  * $T_{\text{wall}}$: Tổng thời gian đồng hồ đo được từ khi bắt đầu request đầu tiên đến khi request cuối cùng kết thúc.
* **Tham chiếu task:**
  * **Task Trello:** `Task #3 [M1]`, `Task #26 [M2-04]`, `Task #35 [M3-05]`.
  * **File triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py).

---

### 1.6. Tỉ lệ tăng tốc (Speedup vs. Vanilla)
* **Công thức:**
  $$\text{Speedup} = \frac{\text{Throughput}_{\text{Speculative}}}{\text{Throughput}_{\text{Vanilla}}} = \frac{\text{TPOT}_{\text{Vanilla}}}{\text{TPOT}_{\text{Speculative}}}$$
* **Ý nghĩa:** Tỉ lệ đo lường mức độ cải thiện tốc độ so với giải mã tự hồi quy truyền thống (Vanilla). Speedup $> 1.0\times$ là tăng tốc dương; Speedup $< 1.0\times$ là tăng tốc âm.
* **Tham chiếu task:**
  * **Task Trello:** `Task #3 [M1]`, `Task #26 [M2-04]`, `Task #32 [M3-02]`, `Task #35 [M3-05]`.
  * **File tài liệu:** [`docs/chuong-4-thuc-nghiem.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/docs/chuong-4-thuc-nghiem.md) (§4.1.2).

---

## 2. Suy đoán thích ứng và bộ điều khiển ngân sách nháp (adaptive controller)

### 2.1. Tỉ lệ chấp nhận tức thời của từng bước nháp ($\alpha_{\text{step}}$)
* **Công thức:**
  $$\alpha_{\text{step}} = \frac{N_{\text{accepted}}}{N_{\text{drafted}}}$$
* **Ý nghĩa:**
  * $N_{\text{drafted}}$: Số token mà mô hình nháp đề xuất trong bước hiện tại ($N_{\text{drafted}} = k$).
  * $N_{\text{accepted}}$: Số token nháp được Target model chấp nhận ($0 \le N_{\text{accepted}} \le k$).
* **Tham chiếu task:**
  * **Task Trello:** `Task #20 [M2-01: Module Hardware Profiler & Mô hình Chi phí T4]` (Khôi Lead 70%, Đại 30%) và `Task #25 [M2-02: Bộ điều khiển Adaptive Controller v0]` (Khôi Lead 60%, Dương 40%).
  * **File mã nguồn:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (phương thức `update()`).

---

### 2.2. Tỉ lệ chấp nhận trung bình trượt mũ ($\alpha_{\text{EMA}}$)
* **Công thức:**
  $$\alpha_{\text{EMA}}^{(t)} = \beta \cdot \alpha_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \alpha_{\text{step}}^{(t)}$$
* **Ý nghĩa:**
  * $\beta = 0.8$: Hệ số suy giảm (decay factor), gán trọng số 80% cho lịch sử và 20% cho bước tức thời.
  * $t$: Chỉ số bước suy luận trong một phiên.
* **Tham chiếu task:**
  * **Task Trello:** `Task #20 [M2-01]`, `Task #25 [M2-02]`.
  * **File mã nguồn:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py).
  * **Nhật ký:** [`logs/Khoi/M2-01/M2-01_worklogs.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/M2-01/M2-01_worklogs.md).

---

### 2.3. Hiệu chỉnh khử lệch bước khởi đầu (Unbiased Warm-up Correction)
* **Công thức:**
  $$\alpha_{\text{unbiased}}^{(t)} = \frac{\alpha_{\text{EMA}}^{(t)} - \alpha_0 \cdot \beta^t}{1 - \beta^t}$$
* **Ý nghĩa:**
  * $\alpha_0 = 0.8$: Giá trị khởi tạo giả định ban đầu.
  * $t$: Số bước telemetry đã thu thập.
  * Khử độ thiên vị của $\alpha_0$ khi $t < 10$, giúp phản ánh trung thực tỉ lệ chấp nhận ngay từ các bước giải mã đầu tiên.
* **Tham chiếu task:**
  * **Task Trello:** `Task #25 [M2-02: Bộ điều khiển Adaptive Controller v0]` (Cấn Đức Khôi).
  * **File mã nguồn:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (phương thức `_get_effective_alpha()`).
  * **Nhật ký:** [`logs/Khoi/M2-02/M2-02_worklogs.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/M2-02/M2-02_worklogs.md).

---

### 2.4. Kỳ vọng số token sinh ra trong một bước suy đoán ($\mathbb{E}[\tau(k)]$)
* **Công thức tổng đóng hình học:**
  $$\mathbb{E}[\tau(k)] = 1 + \sum_{i=1}^{k} \alpha^i = 1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha} \quad (\text{khi } \alpha \ne 1)$$
* **Ý nghĩa:**
  * $k$: Ngân sách token nháp đề xuất ($k \in \{0, 2, 3, 4\}$).
  * Số $1$ đại diện cho token chắc chắn thu được từ Target model.
  * Mỗi token thứ $i$ chỉ được chấp nhận nếu toàn bộ $i-1$ token trước đó đều được chấp nhận, do đó xác suất sống sót giảm theo lũy thừa $\alpha^i$.
* **Tham chiếu task:**
  * **Task Trello:** `Task #20 [M2-01]`, `Task #25 [M2-02]`, `Task #32 [M3-02: Hardware Profiler L4]`.
  * **File mã nguồn:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) và [`src/proposer/cost_model.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/cost_model.py) (hàm `estimate_expected_tokens`).

---

### 2.5. Quy tắc chọn ngân sách cực đại hóa hiệu năng (Argmax Efficiency Rule)
* **Công thức:**
  $$k^* = \arg\max_{k \in \{0, 2, 3, 4\}} \frac{\mathbb{E}[\tau(k, \alpha)]}{C(k, B)}$$
* **Ý nghĩa:** Bộ điều khiển chọn ngân sách $k$ sao cho số token sinh được trên mỗi mili-giây (thông lượng) là lớn nhất tại concurrency $B$ hiện tại.
* **Tham chiếu task:**
  * **Task Trello:** `Task #20 [M2-01]`, `Task #25 [M2-02]`, `Task #32 [M3-02]`.
  * **File thiết kế:** [`docs/05-thiet-ke-he-thong.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/docs/05-thiet-ke-he-thong.md) (§2.2).
  * **File mã nguồn:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (phương thức `select_budget()`).

---

### 2.6. Cơ chế thăm dò định kỳ chống bẫy đóng băng (Deadlock Fallback & Periodic Probing)
* **Quy tắc:**
  $$\text{Nếu } k = 0 \quad \text{và} \quad N_{\text{fallback\_steps}} \ge M_{\text{probe}} (20 \text{ bước}) \implies \text{Ép thử nghiệm } k = 2$$
* **Ý nghĩa:** Tránh bẫy "đóng băng alpha" khi hệ thống rơi vào trạng thái fallback $k=0$. Sau 20 bước, bộ điều khiển chủ động phát lệnh thăm dò $k=2$ để cập nhật lại $\alpha$.
* **Tham chiếu task:**
  * **Task Trello:** `Task #25 [M2-02]` (Khôi Lead, Dương Partner).
  * **File mã nguồn:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) và [`src/proposer/adapter.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/adapter.py).
  * **File kiểm thử:** [`tests/test_budget_fallback_recovery.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/tests/test_budget_fallback_recovery.py).

---

## 3. Mô hình chi phí phần cứng và phân tích hòa vốn (cost model & break-even)

### 3.1. Tổng thời gian một chu kỳ suy đoán Speculative Decoding
* **Công thức:**
  $$C(k, B) = k \times C_D(B) + C_V(k, B)$$
* **Ý nghĩa:**
  * $C_D(B)$: Thời gian forward của Draft model sinh 1 token ở batch size $B$.
  * $C_V(k, B)$: Thời gian xác thực song song $k$ token của Target model ở batch size $B$.
  * Khi $k=0$ (Vanilla): $C(0, B) = C_T(B)$ *(với runner vLLM có gắn model nháp, bước đồng bộ KV tốn $C(0, B) \approx C_T + C_D$)*.
* **Tham chiếu task:**
  * **Task Trello:** `Task #20 [M2-01]` (trên GPU Tesla T4) và `Task #32 [M3-02]` (trên GPU NVIDIA L4).
  * **File mã nguồn:** [`src/proposer/cost_model.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/cost_model.py).

---

### 3.2. Mô hình chi phí xác thực có hệ số suy biến Attention (Verification Overhead)
* **Công thức:**
  $$C_V(k, B) = C_T(B) \times \left(1 + \gamma \cdot (k - 1)\right)$$
* **Ý nghĩa:**
  * $C_T(B)$: Thời gian forward 1 token cơ sở của Target model.
  * $\gamma$: Hệ số gia tăng chi phí do tính toán attention song song trên chuỗi $k$ token. Trên GPU NVIDIA L4, thực nghiệm xác định $\gamma = 0.008$ (tương đương tăng $0.8\%$ mỗi token nháp bổ sung).
* **Tham chiếu task:**
  * **Task Trello:** `Task #32 [M3-02: Hardware Profiler L4]` (Khôi Lead 70%, Dương 30%).
  * **File triển khai:** [`scripts/generate_l4_hardware_profile.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/scripts/generate_l4_hardware_profile.py).
  * **Dữ liệu xuất:** [`experiments/results/hardware_profile_l4.json`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/results/hardware_profile_l4.json).

---

### 3.3. Điều kiện cần và đủ để đạt tăng tốc dương (Speedup > 1.0x)
* **Bất đẳng thức điều kiện:**
  $$\frac{C_D(B)}{C_T(B)} < \frac{\mathbb{E}[\tau(k, \alpha)] - 1}{k} = \bar{\alpha}$$
* **Ý nghĩa cốt lõi:** Tỉ số chi phí phần cứng giữa Draft model và Target model ($C_D / C_T$) bắt buộc phải nhỏ hơn tỉ lệ chấp nhận trung bình $\bar{\alpha}$ trên mỗi token nháp thì hệ thống mới có lời về mặt tốc độ.
* **Tham chiếu task:**
  * **Task Trello:** `Task #20 [M2-01]`, `Task #32 [M3-02]`.
  * **Nhật ký:** [`logs/Khoi/M3-02/M3-02_worklogs.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/M3-02/M3-02_worklogs.md).

---

### 3.4. Phương trình giải nghiệm điểm hòa vốn (Break-Even Acceptance Rate $\alpha^*$)
* **Phương trình:**
  $$\mathbb{E}[\tau(k, \alpha^*)] = 1 + \sum_{i=1}^{k} (\alpha^*)^i = \frac{C(k, B)}{C_{\text{vanilla}}(B)}$$
* **Thuật toán giải:** Giải nghiệm đa thức $\alpha^*(k, B) \in (0, 1)$ bằng phương pháp chia đôi (Bisection root solver).
* **Tham chiếu task:**
  * **Task Trello:** `Task #32 [M3-02: Hardware Profiler L4]` (Khôi Lead 70%).
  * **File mã nguồn:** [`src/proposer/cost_model.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/cost_model.py) (phương thức `calculate_alpha_star()`).
  * **File kiểm thử:** [`tests/test_cost_model_l4.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/tests/test_cost_model_l4.py).

---

## 4. Tính bảo toàn lossless và kiểm chứng tính xác định

### 4.1. Lọc loại trừ có hiệu chỉnh (Modified Rejection Sampling khi $T > 0$)
* **Xác suất chấp nhận:**
  $$P_{\text{accept}}(x) = \min\left(1, \frac{q(x)}{p(x)}\right)$$
* **Phân phối thặng dư khi bị từ chối:**
  $$q'(x) = \frac{\max\left(0, q(x) - p(x)\right)}{\sum_{x'} \max\left(0, q(x') - p(x')\right)}$$
* **Hệ quả bảo toàn phân phối:** $P_{\text{final}}(x) \equiv q(x)$.
* **Tham chiếu task:**
  * **Task Trello:** `Task #2 [M0: PoC Nền tảng]` (Khôi Lead) và `Task #4 [M2: Adaptive v0 & Lossless]` (Khôi, Đại, Dương).
  * **Tài liệu lý thuyết:** [`docs/07-thuat-ngu.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/docs/07-thuat-ngu.md) (§Thuật ngữ Acceptance Criterion).

---

### 4.2. Bảo toàn Lossless ở chế độ giải mã tham lam (Greedy Decoding khi $T = 0.0$)
* **Điều kiện chấp nhận từng token:**
  $$\tilde{x}_i == \arg\max_{x} \text{logits}_{\text{target}}(x \mid x_{<i}, \tilde{x}_{<i})$$
* **Công thức kiểm chứng tỉ lệ trùng khớp token (Token Match Rate):**
  $$\text{TokenMatchRate} = \frac{\sum_{i=1}^{\max(L_V, L_A)} \mathbb{I}(V_i == A_i)}{\max(L_V, L_A)}$$
* **Tham chiếu task:**
  * **Task Trello:** `Task #21 [M2-03: Kịch bản Kiểm chứng Lossless Verification]` (Đại Lead 70%, Khôi 30%).
  * **File mã nguồn:** [`experiments/lossless_verifier.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/lossless_verifier.py) và [`src/benchmark/lossless.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/benchmark/lossless.py).
  * **Báo cáo kỹ thuật:** [`logs/problems/2026-10-16-m2-03-lossless-verification-repeatability.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/problems/2026-10-16-m2-03-lossless-verification-repeatability.md).

---

## 5. Mô hình bộ nhớ phần cứng VRAM và KV-Cache

### 5.1. Cân bằng ngân sách bộ nhớ GPU (VRAM Budget Model)
* **Công thức:**
  $$VRAM_{\text{total}} = VRAM_{\text{weights}} + VRAM_{\text{kv\_cache}} + VRAM_{\text{activation}} + VRAM_{\text{cuda\_ctx}}$$
* **Tham chiếu task:**
  * **Task Trello:** `Task #2 [M0: PoC Nền tảng]` (Khôi Lead 100%) và `Task #20 [M2-01]` (Khôi Lead 70%).
  * **File kiểm thử:** [`scripts/test_vram_stability.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/scripts/test_vram_stability.py).
  * **Nhật ký:** [`logs/Khoi/M0-01/M0-01_worklogs.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/M0-01/M0-01_worklogs.md).

---

### 5.2. Kích thước bộ nhớ đệm KV-Cache cho kiến trúc GQA (Grouped-Query Attention)
* **Công thức dung lượng trên mỗi token:**
  $$\text{KV\_Bytes}_{\text{per\_token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
* **Áp dụng cho Qwen2.5-7B:**
  $$\text{KV\_Bytes}_{\text{per\_token}} = 2 \times 28 \times 4 \times 128 \times 2 = 57,344\text{ bytes} \approx 56\text{ KB/token}$$
* **Tham chiếu task:**
  * **Task Trello:** `Task #2 [M0: PoC]` (Khôi) và `Task #30 [M3-00: Hạ tầng Serverless Modal.com]` (Dương Lead 100%).
  * **File cấu hình:** [`experiments/modal_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/modal_runner.py).

---

### 5.3. Quy tắc áp lực bộ nhớ kép và trễ phục hồi (Hysteresis Memory Guard)
* **Điều kiện ngắt nháp về $k=0$:**
  $$KV_{\text{usage}} > 90.0\% \quad \text{HOẶC} \quad VRAM_{\text{free}} < 1500\text{ MB}$$
* **Điều kiện trễ cho phép bật lại nháp (Hysteresis Recovery):**
  $$KV_{\text{usage}} < 65.0\% \quad \text{và duy trì trạng thái an toàn tối thiểu } N_{\text{cooldown}} = 3\text{ bước}$$
* **Tham chiếu task:**
  * **Task Trello:** `Task #25 [M2-02]` (Dương đóng góp rule KV-cache, Khôi tích hợp adapter).
  * **File mã nguồn:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py).
  * **Nhật ký:** [`logs/Duong/M2-02/M2-02-worklog.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Duong/M2-02/M2-02-worklog.md).

---

## 6. Phương pháp luận thực nghiệm và kiểm soát nhiễu

### 6.1. Triệt tiêu thời gian Prefill để đo độ trễ Decode thuần
* **Công thức:**
  $$step\_ms = \frac{T(1 + N) - T(1)}{N} \quad (\text{với } N = 64\text{ tokens})$$
* **Ý nghĩa:** Lấy hiệu số $T(1+N) - T(1)$ để triệt tiêu thời gian xử lý prompt ban đầu ($T_{\text{prefill}}$), thu được thời gian thực thi thuần túy của hardware ở pha decode.
* **Tham chiếu task:**
  * **Task Trello:** `Task #32 [M3-02: Hardware Profiler L4]` (Dương thực thi đo đạc, Khôi xử lý dữ liệu).
  * **File mã nguồn:** [`experiments/modal_profiler.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/modal_profiler.py) và [`experiments/modal_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/modal_runner.py).
  * **Dữ liệu thực nghiệm:** [`experiments/results/m3_02_forward_costs_l4.json`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/results/m3_02_forward_costs_l4.json).

---

### 6.2. Kiểm soát độ phân tán dữ liệu thực nghiệm (Variance Ratio - Protocol Option B)
* **Công thức:**
  $$\text{Variance Ratio} = \frac{\text{TPOT}_{\max} - \text{TPOT}_{\min}}{\text{TPOT}_{p50}} \le 15.0\%$$
* **Ý nghĩa:** Loại bỏ các lần chạy bị nhiễu do GPU throttling nhiệt độ hoặc CPU background load.
* **Tham chiếu task:**
  * **Task Trello:** `Task #3 [M1]`, `Task #26 [M2-04]`, `Task #32 [M3-02]`.
  * **File triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py) và [`experiments/modal_profiler.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/modal_profiler.py).
  * **Quy chuẩn thực nghiệm:** [`docs/chuong-4-thuc-nghiem.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/docs/chuong-4-thuc-nghiem.md) (§4.1.2).

---

## 7. Hàm mất mát huấn luyện mô hình nháp EAGLE-3.1 Speculator

### 7.1. Hàm mất mát kết hợp (Dual Loss Function)
* **Công thức:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{SmoothL1}} \quad (\text{với } \lambda = 0.5)$$
* **Ý nghĩa:** Cân bằng giữa việc dự đoán đúng token tiếp theo ($\mathcal{L}_{\text{CE}}$) và việc khớp biểu diễn không gian ẩn với mô hình lớn ($\mathcal{L}_{\text{SmoothL1}}$).
* **Tham chiếu task:**
  * **Task Trello:** `Task #33 [M3-03: Training Pipeline EAGLE-3.1]` (Khôi Lead 100%) và `Task #34 [M3-04: Canh Train & Quản trị Checkpoint]` (Đại Lead 100%).
  * **File đặc tả:** [`qwen2.5-7b-eagle3.1-speculator/2026-10-10-qwen2.5-eagle3.1-training-spec.md`](file:///home/duckhoi04/Documents/Repo/Do-an/qwen2.5-7b-eagle3.1-speculator/2026-10-10-qwen2.5-eagle3.1-training-spec.md).

---

### 7.2. Thành phần Cross-Entropy phân loại token (Token Classification Loss)
* **Công thức:**
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{K} \sum_{t=1}^{K} \log P_{\text{draft}}\left(x_{t+1} \mid h_t, x_{\le t}\right)$$
* **Ý nghĩa:** Đo lường độ sai lệch phân phối xác suất dự đoán của draft head so với token ground truth của Target model.
* **Tham chiếu task:**
  * **Task Trello:** `Task #33 [M3-03]`, `Task #34 [M3-04]`.
  * **File mã nguồn:** Repo `qwen2.5-7b-eagle3.1-speculator` (module loss/training).

---

### 7.3. Thành phần hồi quy biểu diễn ẩn (Hidden State Regression Loss)
* **Công thức Smooth L1 Loss:**
  $$\mathcal{L}_{\text{SmoothL1}}\left(h_{\text{pred}}, h_{\text{target}}\right) = \frac{1}{d} \sum_{j=1}^{d} z_j$$
  Trong đó:
  $$z_j = \begin{cases} 0.5 \left(h_{\text{pred}, j} - h_{\text{target}, j}\right)^2 & \text{nếu } \left|h_{\text{pred}, j} - h_{\text{target}, j}\right| < 1.0 \\ \left|h_{\text{pred}, j} - h_{\text{target}, j}\right| - 0.5 & \text{ngược lại} \end{cases}$$
* **Ý nghĩa:** Tránh hiện tượng bùng nổ gradient khi sai số lớn ($L_1$), đồng thời duy trì độ hội tụ mượt mà khi sai số nhỏ ($L_2$), giúp Draft head học được ngữ nghĩa tầng sâu của Target model.
* **Tham chiếu task:**
  * **Task Trello:** `Task #33 [M3-03]`, `Task #34 [M3-04]`.
  * **File đặc tả:** [`qwen2.5-7b-eagle3.1-speculator/2026-10-10-qwen2.5-eagle3.1-training-spec.md`](file:///home/duckhoi04/Documents/Repo/Do-an/qwen2.5-7b-eagle3.1-speculator/2026-10-10-qwen2.5-eagle3.1-training-spec.md).
