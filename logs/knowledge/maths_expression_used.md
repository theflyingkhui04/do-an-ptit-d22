# Tổng hợp các công thức toán học và biểu thức kỹ thuật trong đồ án D22

> **Đề tài:** FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs  
> **Kho lưu trữ:** `do-an-ptit-d22` (`flexsds v0.3.4`, vLLM `0.30.0`, cập nhật 2026-11-25)  
> **Mục đích:** Bảng tra cứu toàn diện các công thức toán học, mô hình chi phí thực nghiệm, bộ ước lượng xác suất chấp nhận (bao gồm R6 geometric MLE), chỉ số phục vụ, quy tắc bảo vệ bộ nhớ và hàm mất mát huấn luyện đối chiếu trực tiếp với mã nguồn hiện hành (từ M0 đến M5 / R6).

---

## 1. Đo lường hiệu năng suy luận và chỉ số phục vụ (serving metrics)

### 1.1. Thời gian đến token đầu tiên (Time To First Token — TTFT)
* **Công thức:**
  $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_sent}} \approx T_{\text{prefill}}(L_{\text{prompt}})$$
* **Ý nghĩa:**
  * $t_{\text{request\_sent}}$: Thời điểm client gửi yêu cầu đến máy chủ suy luận.
  * $t_{\text{first\_token}}$: Thời điểm client nhận được token đầu tiên qua luồng SSE.
  * $T_{\text{prefill}}$: Thời gian mô hình nạp và xử lý toàn bộ prompt độ dài $L_{\text{prompt}}$ qua self-attention.
* **Mã nguồn triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py), [`src/api/bench.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/api/bench.py).
* **Vai trò:** Chỉ số ràng buộc (constraint metric) — speculative decoding tối ưu pha decode và không được làm xấu TTFT ở pha prefill.

---

### 1.2. Thời gian trung bình mỗi token sinh ra (Time Per Output Token — TPOT)
* **Công thức:**
  $$\text{TPOT} = \frac{T_{\text{end}} - T_{\text{first\_token}}}{N_{\text{gen}} - 1} \quad (\text{khi } N_{\text{gen}} > 1)$$
* **Ý nghĩa:**
  * $T_{\text{end}}$: Thời điểm nhận token cuối cùng của phản hồi.
  * $N_{\text{gen}}$: Tổng số token sinh ra trong pha decode (`completion_tokens`).
* **Mã nguồn triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py), [`experiments/aggregate_results.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/aggregate_results.py), [`src/api/bench.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/api/bench.py).
* **Vai trò:** **Mục tiêu tối ưu hóa chính** của toàn bộ đề tài.

---

### 1.3. Khoảng cách thời gian giữa hai token liên tiếp (Inter-Token Latency — ITL)
* **Công thức:**
  $$\text{ITL}_i = t_i - t_{i-1} \quad (\text{với } i = 2, \dots, N_{\text{gen}})$$
* **Ý nghĩa:** Đo độ trễ từng token riêng lẻ để tính phân vị $p50$ và $p95$, phản ánh độ mượt của trải nghiệm streaming thực tế.
* **Mã nguồn triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py).

---

### 1.4. Độ trễ tổng thể từ đầu đến cuối (End-to-End Latency — E2E)
* **Công thức:**
  $$\text{E2E} \approx \text{TTFT} + \text{TPOT} \times (N_{\text{gen}} - 1)$$
* **Ý nghĩa:** Tổng thời gian hoàn tất một yêu cầu suy luận đối với ứng dụng người dùng cuối.
* **Mã nguồn triển khai:** [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py), [`src/flexsds/client/client.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/client/client.py).

---

### 1.5. Thông lượng hệ thống (System Throughput)
* **Công thức:**
  $$\text{Throughput} = \frac{\sum_{j=1}^{M} N_{\text{gen}, j}}{T_{\text{wall}}} \quad (\text{tokens/giây})$$
* **Ý nghĩa:**
  * $M$: Tổng số request trong một lần lặp đo (chuẩn thực nghiệm $M = 32$ request mỗi repeat).
  * $N_{\text{gen}, j}$: Số token sinh ra của request thứ $j$.
  * $T_{\text{wall}}$: Thời gian đồng hồ thực của cả lần lặp đo (`run_duration_s`).
* **Mã nguồn triển khai:** [`experiments/aggregate_results.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/aggregate_results.py) (hàm `summarize_repeat()`).

---

### 1.6. Tỉ lệ tăng tốc (Speedup vs. Vanilla)
* **Công thức:**
  $$\text{Speedup}(g, w, c) = \frac{\text{TPOT}_{p50,\text{vanilla}}(g, w, c)}{\text{TPOT}_{p50,\text{lane}}(g, w, c)}$$
* **Quy tắc cô lập khoa học:** Speedup luôn được tính trên **cùng GPU $g$, cùng workload $w$ và cùng mức concurrency $c$**, có kiểm tra trùng khớp `commit`, `dataset_version`, `workload_file_sha256` và `_experiment_id`.
  * $\text{Speedup} > 1.0\times$: tăng tốc dương.
  * $\text{Speedup} < 1.0\times$: tăng tốc âm (vẫn được giữ nguyên làm bằng chứng thực nghiệm).
* **Mã nguồn triển khai:** [`experiments/aggregate_results.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/aggregate_results.py) (`apply_speedup_vs_vanilla()`), [`experiments/run_benchmark_campaign.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/run_benchmark_campaign.py) (`aggregate()` tách riêng từng GPU trong 4 GPU `T4`, `A10`, `L40S`, `A100`).

---

## 2. Suy đoán thích ứng và bộ điều khiển ngân sách nháp (Adaptive Controller — R6)

### 2.1. Tỉ lệ chấp nhận tổng hợp từ bộ đếm gốc của vLLM ($\alpha, \tau$)
* **Công thức (cho một ô benchmark gồm nhiều request):**
  $$\alpha = \frac{\sum N_{\text{accepted\_draft\_tokens}}}{\sum N_{\text{draft\_tokens}}}, \qquad \tau = 1 + \frac{\sum N_{\text{accepted\_draft\_tokens}}}{\sum N_{\text{spec\_steps}}}$$
* **Ghi chú biên độ $K = 0$:** Ở làn `adaptive`, nếu toàn bộ ô chọn $K = 0$ (tắt nháp hợp lệ khi tải cao hoặc câu ngắn) dẫn đến $\sum N_{\text{draft\_tokens}} = 0$, hệ thống giữ $\alpha = \text{N/A}, \tau = \text{N/A}$ thay vì chia cho 0; ở làn `static` thì $\sum N_{\text{draft\_tokens}} = 0$ bị từ chối fail-closed.
* **Mã nguồn triển khai:** [`experiments/aggregate_results.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/aggregate_results.py) (`aggregate_static_acceptance()`), [`experiments/run_benchmark_campaign.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/run_benchmark_campaign.py) (`summarize_speculative_acceptance()`).

---

### 2.2. Bộ ước lượng R6 Discounted-Count Geometric MLE ($\hat{\alpha}_{\text{MLE}}$ — mặc định khi có cost matrix)
* **Vấn đề của tỉ số thô $a_t / k_t$:** Trong speculative decoding tuần tự, một khi vị trí nháp đầu tiên bị từ chối thì mọi token nháp phía sau đều bị loại bỏ theo chuỗi hình học bị chặn phải (right-censored geometric trial). Do đó tỉ số $a_t / k_t$ đánh giá thấp $\alpha$ khi $k_t$ lớn.
* **Số vị trí thực sự được kiểm chứng ở bước $t$:**
  $$n_t = a_t + \mathbb{I}(a_t < k_t)$$
  *(với $a_t = \text{num\_accepted}$, $k_t = \text{num\_drafted}$; nếu toàn bộ $k_t$ token đều được chấp nhận thì kiểm chứng đúng $k_t$ vị trí, còn nếu bị từ chối sớm ở vị trí $a_t + 1$ thì kiểm chứng đúng $a_t + 1$ vị trí)*.
* **Bộ tích lũy có suy giảm và tiên nghiệm Beta yếu:**
  $$A_0 = w_0 \cdot \alpha_0, \qquad N_0 = w_0 \qquad (\text{mặc định } w_0 = 2.0,\; \alpha_0 = 0.8,\; \beta_{\text{mle}} = 0.90)$$
  $$A_t = \beta_{\text{mle}} \cdot A_{t-1} + a_t, \qquad N_t = \beta_{\text{mle}} \cdot N_{t-1} + n_t$$
  $$\hat{\alpha}_{\text{MLE}}^{(t)} = \text{clip}\left(\frac{A_t}{N_t},\; 0,\; 1\right)$$
* **Mã nguồn triển khai:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (`AdaptiveDraftController.update()`, `get_geometric_alpha()`, `_resolved_alpha_estimator()`).
* **Chế độ chọn tự động (`alpha_estimator="auto"`):** Dùng `geometric_mle` khi bộ điều khiển có `cost_profile.matrix` (mọi phiên adaptive có hardware profile thực nghiệm), ngược lại dùng `legacy_ratio` (EMA khử lệch).

---

### 2.3. Bộ ước lượng EMA truyền thống và hiệu chỉnh khử lệch bước đầu (legacy fallback)
* **Trung bình trượt mũ:**
  $$\alpha_{\text{step}}^{(t)} = \frac{a_t}{k_t}, \qquad \alpha_{\text{EMA}}^{(t)} = \beta \cdot \alpha_{\text{EMA}}^{(t-1)} + (1 - \beta) \cdot \alpha_{\text{step}}^{(t)} \quad (\beta = 0.8)$$
* **Hiệu chỉnh khử lệch bước khởi đầu (unbiased warm-up correction):**
  $$\alpha_{\text{unbiased}}^{(t)} = \text{clip}\left(\frac{\alpha_{\text{EMA}}^{(t)} - \alpha_0 \cdot \beta^t}{1 - \beta^t},\; 0,\; 1\right) \quad (\alpha_0 = 0.8)$$
* **Mã nguồn triển khai:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (`get_effective_alpha()`), [`src/flexsds/core/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/controller.py).

---

### 2.4. Kỳ vọng số token sinh ra trong một bước suy đoán ($\mathbb{E}[\tau(k, \alpha)]$)
* **Công thức tổng đóng hình học:**
  $$\mathbb{E}[\tau(k, \alpha)] = 1 + \sum_{i=1}^{k} \alpha^i = \begin{cases} 1 & \text{nếu } k = 0 \\ 1 + k & \text{nếu } \alpha = 1 \\ \dfrac{1 - \alpha^{k+1}}{1 - \alpha} & \text{nếu } k > 0,\; \alpha \ne 1 \end{cases}$$
* **Ý nghĩa:** Số $1$ là token chắc chắn thu được từ mô hình đích; mỗi token nháp thứ $i$ chỉ sống sót nếu $i - 1$ token trước đó đều được chấp nhận (xác suất $\alpha^i$).
* **Mã nguồn triển khai:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (`estimate_expected_tokens()`), [`src/proposer/cost_model.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/cost_model.py).

---

### 2.5. Quy tắc chọn ngân sách dựa trên chi phí phần cứng R6 (Hardware Cost-Driven Rule)
* **Chuẩn hóa cận dưới chi phí theo batch $B$:**
  $$C_{\text{ref}}(B) = \min_{k \in \{0\} \cup \mathcal{K}_+} C(k, B)$$
  *(Xử lý an toàn các ma trận đo thực nghiệm không đơn điệu khi $C(k, B) < C(0, B)$ do biến động đo lường hoặc tối ưu kernel).*
* **Hệ số tăng tốc dự đoán của từng ứng viên $k \in \mathcal{K}_+$:**
  $$\widehat{\text{Speedup}}(k, B, \alpha) = \mathbb{E}[\tau(k, \alpha)] \cdot \frac{C_{\text{ref}}(B)}{C(k, B)}$$
* **Quy tắc chọn $k^*$ với biên an toàn $\rho_{\min}$ (`min_speedup_ratio = 1.05`):**
  $$k^* = \begin{cases} \arg\max\limits_{k \in \mathcal{K}_+} \widehat{\text{Speedup}}(k, B, \alpha) & \text{nếu } \max\limits_{k \in \mathcal{K}_+} \widehat{\text{Speedup}}(k, B, \alpha) \ge \rho_{\min} \\ 0 & \text{ngược lại (fallback về giải mã thường)} \end{cases}$$
* **Phương trình điểm hòa vốn $\alpha^*(k, B)$ (giải bằng chia đôi 40 vòng lặp trên $[0, 1]$):**
  $$\mathbb{E}[\tau(k, \alpha^*)] = \rho_{\min} \cdot \frac{C(k, B)}{C_{\text{ref}}(B)}$$
* **Mã nguồn triển khai:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (`apply_cost_driven_budget_rule()`, `solve_break_even_alpha()`).

---

### 2.6. Cơ chế thăm dò định kỳ chống đóng băng (Periodic Probing) và quy hoạch theo lô (Batch Planner)
* **Thăm dò định kỳ khi fallback:**
  $$\text{Nếu } k^* = 0 \text{ liên tiếp } N_{\text{fallback}} \ge M_{\text{probe}} \;(M_{\text{probe}} = 20) \implies \text{thử nghiệm } k = k_{\text{probe}} \;(k_{\text{probe}} = 2)$$
  *(Trong `TreeAdaptiveController` v0.2, chu kỳ thăm dò thích ứng theo tải: $T_{\text{probe}}(B) = 15$ nếu $B \le 4$, $20$ nếu $B \le 16$, và $30$ nếu $B > 16$).*
* **Quy hoạch lô trong `AdaptiveBudgetPlanner`:**
  $$K_{\text{batch}} = \max\left(K_{\min},\; \min\left(K_{\max},\; \max_{i \in \text{decode}} k_i\right)\right)$$
  Sau bước lập lịch, từng request có $k_i < K_{\text{batch}}$ được cắt tỉa placeholder về đúng $k_i$ (`_update_after_schedule()`).
* **Mã nguồn triển khai:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py), [`src/server/budget_planner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/budget_planner.py), [`src/server/adaptive_scheduler.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/adaptive_scheduler.py).

---

## 3. Mô hình chi phí phần cứng và hiệu chuẩn thực nghiệm (`flexsds-component-v2`)

### 3.1. Ma trận chi phí thành phần đo thật trên worker CUDA (`flexsds-component-v2`)
* **Công thức tổng hợp mỗi ô $(k, B)$:**
  $$C_{\text{measured}}(k, B) = \operatorname{median}_{r \in \text{measured}}\Big(T_{\text{target}}(k, B, r) + T_{\text{draft}}(k, B, r)\Big)$$
* **Lưới đo chuẩn:** $k \in \{0, 1, 2, 3, 4\}$, $B \in \{1, 2, 4, 8, 16, 32\}$, ngữ cảnh cố định $L = 1024$ token, đo bằng `torch.cuda.synchronize()` + `perf_counter` bên trong worker thực (`CalibrationWorker`), tách tiến trình riêng cho từng giá trị $k$ để tránh nhiễu trạng thái.
  * Chế độ `fast`: 2 warm-up + 5 lặp đo mỗi ô $(k, B)$.
  * Chế độ `full`: 5 warm-up + 50 lặp đo mỗi ô $(k, B)$.
* **Tra cứu `EmpiricalCostProfile`:** Chọn batch đo gần nhất $\arg\min_{b \in \mathcal{B}_k} (|b - B|, b)$ và **không bao giờ tự bịa chi phí** nếu thiếu $k$.
* **Mã nguồn triển khai:** [`src/flexsds/core/calibration.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/calibration.py), [`src/flexsds/core/calibration_worker.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/calibration_worker.py), [`src/flexsds/core/cost_model.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/cost_model.py), [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (`EmpiricalCostProfile`).

---

### 3.2. Mô hình phân tích thành phần lịch sử (M2/M3 analytical cost model)
* **Tổng thời gian một bước suy đoán:**
  $$C(k, B) = k \cdot C_D(B) + C_V(k, B), \qquad C_V(k, B) = C_T(B) \cdot \big(1 + \gamma (k - 1)\big)$$
  *(Trên L4 M3-02 xác định $\gamma = 0.008$; khi $k = 0$ trong vLLM có gắn đầu nháp, vẫn phát sinh đồng bộ KV của drafter nên chi phí đo thật ở $k = 0$ là $C_{\text{measured}}(0, B)$).*
* **Điều kiện cần để tăng tốc dương:**
  $$\frac{C_D(B)}{C_T(B)} < \frac{\mathbb{E}[\tau(k, \alpha)] - 1}{k}$$
* **Mã nguồn tham chiếu:** [`src/proposer/cost_model.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/cost_model.py), [`scripts/generate_l4_hardware_profile.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/scripts/generate_l4_hardware_profile.py), [`experiments/results/hardware_profile_l4.json`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/results/hardware_profile_l4.json).

---

## 4. Tính bảo toàn lossless và kiểm chứng tính xác định

### 4.1. Lấy mẫu loại trừ có hiệu chỉnh (Modified Rejection Sampling khi $T > 0$)
* **Xác suất chấp nhận:**
  $$P_{\text{accept}}(x) = \min\left(1, \frac{q(x)}{p(x)}\right)$$
* **Phân phối thặng dư khi bị từ chối:**
  $$q'(x) = \frac{\max\left(0, q(x) - p(x)\right)}{\sum_{x'} \max\left(0, q(x') - p(x')\right)}$$
* **Hệ quả bảo toàn phân phối:** $P_{\text{final}}(x) \equiv q(x)$ (phân phối đích không đổi).

---

### 4.2. Bảo toàn lossless ở chế độ tham lam (Greedy Decoding khi $T = 0.0$)
* **Điều kiện chấp nhận từng token:**
  $$\tilde{x}_i = \arg\max_{x} \text{logits}_{\text{target}}(x \mid x_{<i}, \tilde{x}_{<i})$$
* **Tỉ lệ trùng khớp token (Token Match Rate):**
  $$\text{TokenMatchRate} = \frac{\sum_{i=1}^{\max(L_V, L_A)} \mathbb{I}(V_i = A_i)}{\max(L_V, L_A)}$$
* **Mã nguồn kiểm chứng:** [`experiments/lossless_verifier.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/lossless_verifier.py), [`experiments/m4_lossless.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/m4_lossless.py), [`src/benchmark/lossless.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/benchmark/lossless.py).

---

## 5. Mô hình bộ nhớ phần cứng VRAM, KV cache và rào chắn chống OOM

### 5.1. Cân bằng ngân sách bộ nhớ GPU và ngưỡng preflight CLI
* **Tổng VRAM chiếm dụng:**
  $$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{kv\_cache}} + \text{VRAM}_{\text{activation}} + \text{VRAM}_{\text{cuda\_ctx}}$$
* **Ngưỡng cảnh báo preflight trong CLI:** Ước lượng dung lượng yêu cầu không vượt quá $95\%$ ngân sách VRAM khả dụng (`T4`: 15,000 MB / 16 GB; `A10`: 22,000 MB / 24 GB; `L40S`: 46,000 MB / 48 GB; `A100`: 78,000 MB / 80 GB).
* **Mã nguồn triển khai:** [`src/flexsds/core/hardware.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/hardware.py), [`src/flexsds/core/compatibility.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/compatibility.py), [`scripts/test_vram_stability.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/scripts/test_vram_stability.py).

---

### 5.2. Kích thước bộ nhớ đệm KV cache cho kiến trúc GQA
* **Dung lượng trên mỗi token:**
  $$\text{KV\_Bytes}_{\text{per\_token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
* **Áp dụng cho Qwen2.5-7B** ($n_{\text{layers}} = 28,\; n_{\text{kv\_heads}} = 4,\; d_{\text{head}} = 128,\; \text{FP16/BF16} = 2\text{ bytes}$):
  $$\text{KV\_Bytes}_{\text{per\_token}} = 2 \times 28 \times 4 \times 128 \times 2 = 57,344\text{ bytes} \approx 56\text{ KB/token}$$
* **Ước lượng dung lượng KV cache tối thiểu cho cấu hình `(max_num_seqs, max_model_len)` trong preflight (`_kv_cache_mb()`):**
  $$\text{KV\_MB}_{\min} = \frac{2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot 2 \cdot B_{\max} \cdot L_{\max}}{1024^2}$$
* **Mã nguồn triển khai:** [`src/flexsds/core/compatibility.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/compatibility.py) (`_kv_cache_mb()`).

---

### 5.3. Quy tắc áp lực bộ nhớ kép và trễ phục hồi (Hysteresis Memory Guard)
* **Điều kiện kích hoạt ngắt nháp về $k = 0$:**
  $$\text{KV}_{\text{usage}} > 90.0\% \quad \text{hoặc} \quad \text{VRAM}_{\text{free}} < 1500\text{ MB}$$
* **Điều kiện trễ phục hồi (hysteresis recovery):**
  * Giữ $k = 0$ tối thiểu $N_{\text{cooldown}} = 3$ bước giải mã liên tiếp (`memory_cooldown_steps = 3`).
  * Chỉ nhả khóa khi đồng thời đạt $\text{KV}_{\text{usage}} < 80.0\%$ (`kv_cache_recover_pct = 80.0`) và $\text{VRAM}_{\text{free}} \ge 1500\text{ MB}$.
  * *(Riêng trong `TreeAdaptiveController` v0.2, nếu thiếu tín hiệu `VRAM_free` hợp lệ, ngưỡng KV tự động siết chặt thành kích hoạt $> 75.0\%$ và phục hồi $< 65.0\%$).*
* **Mã nguồn triển khai:** [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py) (`apply_memory_pressure_rule()`), [`src/flexsds/core/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/core/controller.py) (`select_budget()`).

---

## 6. Phương pháp luận thực nghiệm và kiểm soát nhiễu

### 6.1. Triệt tiêu thời gian prefill để đo độ trễ decode thuần (M3-02 forward profiler)
* **Công thức:**
  $$\text{step\_ms} = \frac{T(1 + N) - T(1)}{N} \quad (N = 64\text{ tokens})$$
* **Mã nguồn triển khai:** [`experiments/modal_profiler.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/modal_profiler.py), [`experiments/results/m3_02_forward_costs_l4.json`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/results/m3_02_forward_costs_l4.json).

---

### 6.2. Chính sách xoay mẫu tất định (`cycle-rotate`) và kiểm soát phương sai
* **Ánh xạ mẫu theo repeat $r \in \{1, \dots, 5\}$ và chỉ số request $i \in \{0, \dots, 31\}$:**
  $$\text{offset}(r) = \big((r - 1) \cdot (M \bmod S)\big) \bmod S, \qquad \text{sample\_index}(r, i) = \big(i + \text{offset}(r)\big) \bmod S$$
  *(với $M = 32$ request/repeat và $S$ là số mẫu chuẩn của gói workload — đảm bảo 3 làn `vanilla`, `static`, `adaptive` nhận cùng dãy prompt và cùng thứ tự).*
* **Tỉ lệ biến thiên (Variance Ratio):**
  $$\text{Variance Ratio} = \frac{\text{TPOT}_{\max} - \text{TPOT}_{\min}}{\text{TPOT}_{p50}} \le 15.0\%$$
* **Mã nguồn triển khai:** [`experiments/workload_adapter.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/workload_adapter.py), [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py), [`experiments/aggregate_results.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/aggregate_results.py).

---

## 7. Hàm mất mát huấn luyện mô hình nháp EAGLE-3.1 Speculator

### 7.1. Hàm mất mát kết hợp (Dual Loss Function)
* **Công thức:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{SmoothL1}} \quad (\lambda = 0.5)$$
* **Đặc tả tham chiếu:** [`qwen2.5-7b-eagle3.1-speculator/2026-10-10-qwen2.5-eagle3.1-training-spec.md`](file:///home/duckhoi04/Documents/Repo/Do-an/qwen2.5-7b-eagle3.1-speculator/2026-10-10-qwen2.5-eagle3.1-training-spec.md).

### 7.2. Thành phần Cross-Entropy phân loại token
* **Công thức:**
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{K} \sum_{t=1}^{K} \log P_{\text{draft}}\left(x_{t+1} \mid h_t, x_{\le t}\right)$$

### 7.3. Thành phần hồi quy biểu diễn ẩn (Smooth L1 Loss)
* **Công thức:**
  $$\mathcal{L}_{\text{SmoothL1}}\left(h_{\text{pred}}, h_{\text{target}}\right) = \frac{1}{d} \sum_{j=1}^{d} z_j, \qquad z_j = \begin{cases} 0.5 \left(h_{\text{pred}, j} - h_{\text{target}, j}\right)^2 & \text{nếu } \left|h_{\text{pred}, j} - h_{\text{target}, j}\right| < 1.0 \\ \left|h_{\text{pred}, j} - h_{\text{target}, j}\right| - 0.5 & \text{ngược lại} \end{cases}$$
* **Kiểm tra đặc trưng EAGLE-3.1 lúc nạp checkpoint (`flexsds inspect` / `compatibility.py`):** Xác thực bộ cờ `fc_norm`, `norm_output`, `use_post_norm_feedback` kèm các tensor `fc_norm.*.weight` và ánh xạ từ vựng rút gọn `d2t`/`t2d`.
