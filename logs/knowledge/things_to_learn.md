# Danh mục kiến thức trọng tâm cần nghiên cứu: Cẩm nang toàn diện cho đồ án tốt nghiệp

> **Cập nhật:** 2026-11-25 — đối chiếu đồng bộ với mã nguồn hiện hành (`flexsds v0.3.4`, vLLM `0.30.0`, R6 geometric MLE + cost-driven controller, EAGLE-3.1, danh mục 4 GPU Modal `T4`/`A10`/`L40S`/`A100` và luồng benchmark 3 node × 4 GPU × 3 làn).

Tài liệu này tổng hợp toàn bộ các mảng kiến thức lý thuyết, kỹ thuật hệ thống, cơ chế tối ưu phần cứng và từ khóa tra cứu phục vụ đồ án tốt nghiệp: **"FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs"** (PTIT D22).

Nội dung được cấu trúc thành 6 phần:
1. **Bộ nhớ KV cache:** Bản chất toán học, kiến trúc chú ý GQA và cơ chế quản lý bộ nhớ trong vLLM.
2. **Giải mã suy đoán thích ứng (Adaptive Speculative Decoding):** Nguyên lý lossless, tiến hóa EAGLE-1 → EAGLE-3.1, bộ điều khiển R6 (geometric MLE + cost-driven K), `LineageUniProcExecutor` và rào chắn OOM có hysteresis.
3. **Cơ chế tối ưu tính toán chú ý (PagedAttention và FlashAttention):** Phân trang bộ nhớ, tối ưu IO-aware và giới hạn trên Turing `sm_75`.
4. **Lượng tử hóa mô hình (Quantization):** PTQ (AWQ, GPTQ, RTN), khác biệt kernel `awq`/`half` (T4) với `awq_marlin`/`bfloat16` (A10, L40S, A100).
5. **Đo lường hiệu năng và quy trình benchmark của đồ án:** TTFT, TPOT, ITL, throughput, $\alpha$, $\tau$, quy trình 3 node (`W1_V3`, `W2`, `W3`) × 4 GPU (`T4`, `A10`, `L40S`, `A100`) × 3 làn (`vanilla`, `static`, `adaptive`), chế độ `full` (216 ô) và `fast` (108 ô).
6. **Ma trận câu hỏi phản biện bảo vệ đồ án:** Gợi ý trả lời chuẩn xác theo hệ thống thực tế.

---

## Phần 1: Cơ chế và tối ưu hóa bộ nhớ KV cache trong mô hình ngôn ngữ lớn

### 1. Bản chất và kiến trúc KV cache trong mô hình Transformer
- **Tại sao cần KV cache:** Quá trình sinh tự hồi quy chia làm hai pha rõ rệt:
  + *Prefill phase:* Tính toán biểu diễn cho toàn bộ prompt đầu vào trong một lượt song song; pha này bị giới hạn bởi năng lực tính toán (compute-bound).
  + *Decode phase:* Sinh từng token kế tiếp tuần tự; nếu không lưu lại vector Key và Value của các token trước, mô hình phải tính lại từ đầu với độ phức tạp $O(N^2)$. KV cache đưa chi phí mỗi bước về $O(N)$.
- **Nghẽn băng thông bộ nhớ (Memory-bandwidth bound):** Trong pha decode ở batch nhỏ, cường độ số học (arithmetic intensity — số phép tính FLOPS trên mỗi byte đọc từ HBM/VRAM) rất thấp. Tốc độ sinh token bị giới hạn bởi băng thông nạp trọng số và KV cache từ VRAM.
- **Kiến trúc chú ý và kích thước KV cache:**
  + *MHA (Multi-Head Attention):* $n_{\text{kv}} = n_{\text{q}}$, tốn VRAM nhất.
  + *MQA (Multi-Query Attention):* $n_{\text{kv}} = 1$.
  + *GQA (Grouped-Query Attention):* Chia $n_{\text{q}}$ Query heads thành các nhóm dùng chung $n_{\text{kv}}$ Key-Value heads.
  + *Mô hình của đồ án (`Qwen/Qwen2.5-7B-Instruct-AWQ`):* Sử dụng GQA với $n_{\text{layers}} = 28$, $n_{\text{q}} = 28$, $n_{\text{kv}} = 4$, $d_{\text{head}} = 128$, giúp giảm dung lượng KV cache xuống $4/28 = 1/7$ so với MHA.
- **Công thức chính xác tính dung lượng KV cache mỗi token:**
  $$\text{KV\_Bytes}_{\text{per\_token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_element}$$
  *(Số $2$ đại diện cho cặp tensor Key và Value; với FP16/BF16 thì $\text{bytes\_per\_element} = 2$. Áp dụng cho Qwen2.5-7B: $2 \times 28 \times 4 \times 128 \times 2 = 57,344\text{ bytes} \approx 56\text{ KB/token}$).*

### 2. Quản lý bộ nhớ KV cache trong vLLM (`0.30.0`)
- **PagedAttention:** Chia KV cache thành các block cố định (thường 16 token), quản lý qua bảng trang (*block table*), triệt tiêu phân mảnh ngoài và giữ phân mảnh nội bộ dưới $4\%$.
- **Continuous batching:** Cấp phát và thu hồi block động theo từng bước lập lịch (iteration-level scheduling), ghép xen kẽ request mới vào lô đang giải mã.
- **Giám sát dung lượng:** `gpu_memory_utilization` (mặc định `0.85`) ấn định phần VRAM cho trọng số và KV cache; `scheduler.get_kv_cache_usage()` trả về tỷ lệ chiếm dụng KV cache theo thời gian thực để cấp cho bộ điều khiển FlexSDS.

### 3. KV cache trong giải mã suy đoán (EAGLE-1 / EAGLE-2 / EAGLE-3 / EAGLE-3.1)
- **Cấu trúc KV cache kép:** Cả target model và draft head (EAGLE head 1 lớp hoạt động ở tầng đặc trưng) đều duy trì KV cache riêng.
- **Xác thực song song nhiều token:** Thay vì truy vấn $q = 1$, target model xác thực song song $q = k + 1$ vị trí trong một lượt forward.
- **Chi phí đồng bộ khi $k = 0$:** Khi bộ điều khiển hạ $k = 0$ trong vLLM có gắn drafter, đầu nháp vẫn phải chạy bước đồng bộ trạng thái/KV cache tối thiểu; vì vậy đồ án đo trực tiếp $C_{\text{measured}}(0, B)$ trên đúng worker thật thay vì giả định $C(0, B) = C_{\text{vanilla}}(B)$.
- **Hoàn tác và ghi nhận (Rollback & Commit):** Chỉ các token được chấp nhận mới ghi chính thức vào block table; các token nháp bị từ chối bị loại bỏ để tránh rò rỉ bộ nhớ.

### 4. Áp lực bộ nhớ, an toàn VRAM và rào chắn chống OOM có hysteresis
- **Tranh chấp bộ nhớ theo tầng GPU:** Trên `T4` (16 GB), trọng số AWQ 4-bit chiếm ~4.5–5.0 GB, phần còn lại chia cho KV cache (đặc biệt khi `W1_V3` có `max_model_len = 8192` và `concurrency = 32`), activations và drafter; trên `A10` (24 GB), `L40S` (48 GB) và `A100` (80 GB), biên độ KV cache rộng hơn rõ rệt nhưng vẫn cần giám sát khi ngữ cảnh dài.
- **Luật bảo vệ bộ nhớ có hysteresis (Memory Pressure Rule — [`src/proposer/controller.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/controller.py)):**
  + **Ngưỡng kích hoạt ngắt nháp ($k = 0$):** $\text{KV}_{\text{usage}} > 90.0\%$ hoặc $\text{VRAM}_{\text{free}} < 1500\text{ MB}$.
  + **Trễ phục hồi (hysteresis):** Giữ $k = 0$ tối thiểu $3$ bước (`memory_cooldown_steps = 3`), chỉ nhả khóa khi $\text{KV}_{\text{usage}} < 80.0\%$ và $\text{VRAM}_{\text{free}} \ge 1500\text{ MB}$.
  + *(Ở `TreeAdaptiveController` v0.2, nếu thiếu tín hiệu VRAM hợp lệ thì ngưỡng KV tự động siết xuống kích hoạt $> 75.0\%$ và phục hồi $< 65.0\%$).*

### 5. Kỹ thuật nén và loại bỏ KV cache nâng cao (đối chiếu học thuật)
- **Eviction & Pruning:** H2O (Heavy-Hitter Oracle), StreamingLLM (Attention Sinks giữ 4 token đầu + cửa sổ trượt cục bộ), SnapKV / Ada-KV.
- **Lượng tử hóa KV cache:** Nén tensor K/V sang FP8 (E4M3/E5M2) hoặc INT8/INT4 trên các GPU kiến trúc mới (Ada/Hopper).

### 6. Bảng từ khóa tra cứu về KV cache
| Chủ đề | Từ khóa tìm kiếm học thuật |
| :--- | :--- |
| **Lý thuyết nền** | `Grouped-Query Attention (GQA)`, `LLM decode memory bandwidth bound`, `Arithmetic intensity of Transformer` |
| **Hạ tầng vLLM** | `PagedAttention (Woosuk Kwon et al., SOSP 2023)`, `Block table memory allocation vLLM`, `Continuous batching` |
| **KV cache trong suy đoán** | `Multi-token verification KV cache`, `EAGLE-3 feature-level speculative decoding`, `vLLM dynamic speculative tokens` |
| **An toàn bộ nhớ** | `KV cache capacity planning GPU VRAM`, `Speculative decoding memory overhead concurrency`, `Hysteresis OOM guard` |
| **Nén & Eviction** | `Attention Sinks StreamingLLM`, `H2O Heavy Hitter Oracle`, `SnapKV`, `FP8 KV Cache` |

---

## Phần 2: Giải mã suy đoán và cơ chế điều khiển thích ứng phần cứng (FlexSDS)

### 1. Cơ sở lý thuyết và chứng minh bảo toàn phân phối (lossless)
- **Luồng 3 bước:** (1) Nháp $k$ token bằng đầu nháp nhẹ → (2) Target model xác thực song song cả $k + 1$ vị trí trong một lượt forward → (3) Rejection sampling (khi $T > 0$) hoặc so khớp argmax (khi $T = 0$).
- **Bảo toàn phân phối (Leviathan et al.):** Với phân phối đích $q(x)$ và phân phối nháp $p(x)$:
  $$P_{\text{accept}}(x) = \min\left(1, \frac{q(x)}{p(x)}\right), \qquad q'(x) = \frac{\max(0, q(x) - p(x))}{\sum_{x'} \max(0, q(x') - p(x'))}$$
  Đảm bảo phân phối đầu ra trùng khớp $100\%$ với target model gốc; ở greedy decoding ($T = 0.0$), token sinh ra khớp tuyệt đối (loại trừ sai lệch dấu phẩy động không xác định của phần cứng nếu tắt prefix caching và giữ cùng hợp đồng thực thi).

### 2. Tiến hóa dòng họ EAGLE và phạm vi hỗ trợ trong FlexSDS
- **EAGLE-1 (Li et al., ICML 2024):** Dự đoán ở tầng đặc trưng (feature level) thay vì chỉ tầng token, giải quyết bất định biểu diễn.
- **EAGLE-2 (Li et al., EMNLP 2024):** Bổ sung cây nháp động theo độ tự tin (confidence-aware dynamic tree).
- **EAGLE-3 (Li et al., 2025):** Hợp nhất đặc trưng đa tầng (multi-layer feature fusion) và huấn luyện kiểu training-time test, giảm kích thước từ vựng đầu nháp bằng bảng ánh xạ `d2t`/`t2d`.
- **EAGLE-3.1 (mô hình nháp chính của đồ án — `duckhoi/qwen2.5-7b-eagle3.1-speculator`):** Mở rộng EAGLE-3 với chuẩn hóa tầng nối đặc trưng (`fc_norm`, `norm_output`, `use_post_norm_feedback`) được nhóm tự huấn luyện cho `Qwen/Qwen2.5-7B-Instruct-AWQ`.
- **Hỗ trợ trong FlexSDS CLI:** `flexsds inspect` và `flexsds serve` kiểm tra checkpoint thực (safetensors header + `d2t`/`t2d` + cờ kiến trúc) và hỗ trợ đầy đủ **EAGLE-1, EAGLE-2** (`method="eagle"`) cũng như **EAGLE-3, EAGLE-3.1** (`method="eagle3"`) trên đường chạy tuần tự của vLLM `0.30.0`.

### 3. Động lực đề tài: Sụt tốc (negative acceleration) và nghẽn tải đồng thời
- **Sụt tốc theo entropy tác vụ:** Ở tác vụ cấu trúc cao như trích xuất hóa đơn JSON (`W1_V3`), $\alpha$ cao giúp speculative decoding tăng tốc rõ; ở tác vụ viết báo cáo tự do (`W3`), entropy cao làm $\alpha$ giảm mạnh, khiến nháp tĩnh trở thành gánh nặng (speedup $< 1.0\times$).
- **Nghẽn compute khi tăng concurrency ($B = 1 \to 32$):** Ở $B = 1$, GPU dư năng lực tính toán nên xác thực $k$ token gần như miễn phí; khi $B = 16 \text{ hoặc } 32$, GPU bão hòa compute, chi phí $C(k, B)$ tăng vọt khiến nháp tĩnh tụt xuống dưới $1.0\times$ (đặc biệt trên GPU phổ thông như T4/L4/A10).

### 4. Đóng góp cốt lõi hiện hành: Bộ điều khiển R6 và kiến trúc thực thi
- **Hiệu chuẩn phần cứng thực nghiệm (`flexsds-component-v2`):** Đo trực tiếp $C(k, B)$ trên đúng worker CUDA (`src/flexsds/core/calibration.py`) cho từng GPU (`T4`, `A10`, `L40S`, `A100`), lưu kèm SHA-256 bằng chứng đo và kiểm tra khớp chặt SKU trước khi chạy adaptive.
- **Ước lượng $\hat{\alpha}_{\text{MLE}}$ theo Geometric MLE có suy giảm (R6):** Đếm đúng số vị trí kiểm chứng $n_t = a_t + \mathbb{I}(a_t < k_t)$ với $\beta_{\text{mle}} = 0.90$ và tiên nghiệm Beta yếu ($w_0 = 2.0$), loại bỏ sai lệch đánh giá thấp của tỉ số thô $a_t / k_t$.
- **Quyết định $k^*$ hoàn toàn dựa trên chi phí phần cứng (`apply_cost_driven_budget_rule`):**
  $$C_{\text{ref}}(B) = \min_k C(k, B), \qquad \widehat{\text{Speedup}}(k, B, \alpha) = \mathbb{E}[\tau(k, \alpha)] \cdot \frac{C_{\text{ref}}(B)}{C(k, B)}$$
  Chỉ chọn $k^* > 0$ khi $\widehat{\text{Speedup}} \ge 1.05\times$ (`min_speedup_ratio = 1.05`), ngược lại hạ $k = 0$ kèm thăm dò định kỳ mỗi 20 bước (`probe_k = 2`).
- **Tích hợp vLLM bất đồng bộ an toàn (`AdaptiveAsyncScheduler` + `LineageUniProcExecutor`):**
  + [`src/server/adaptive_scheduler.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/adaptive_scheduler.py): Ghi $K_{\text{batch}}$ vào `dynamic_sd_lookup` và cắt tỉa từng request về đúng $k_i$.
  + [`src/server/lineage_executor.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/lineage_executor.py): Theo dõi chính xác phả hệ (lineage) của phản hồi bất đồng bộ trong vLLM V1 eager (`VLLM_USE_V2_MODEL_RUNNER=0`) để cập nhật đúng `(drafted, accepted)` về bộ điều khiển.
  + Ghi telemetry theo lô (`step_telemetry_flush_every = 64`, tùy chọn tắt qua `--no-step-telemetry`) nhằm loại bỏ nghẽn I/O đĩa trên mỗi bước giải mã.

### 5. Đặc thù phần cứng theo 4 tầng GPU Modal
- **T4 (Low — 16 GB, Turing `sm_75`):** Không hỗ trợ native BF16 và `awq_marlin`; FlexSDS tự động chuyển AWQ sang `half`/`float16` + `awq` khi serve, calibrate và benchmark. Khi chạy bản vá lịch sử trên T4, đồ án từng triển khai [`src/server/t4_rowwise_attention.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/t4_rowwise_attention.py) để xử lý giới hạn FlashAttention-2 trên `sm_75`.
- **A10 (Medium — 24 GB, Ampere `sm_86`, mặc định CLI):** Hỗ trợ BF16 + `awq_marlin`, đã có sẵn profile đo thực nghiệm đóng gói [`profile_sm_86_remote_a10g_fcda5486b12d.json`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json).
- **L40S (High — 48 GB, Ada Lovelace `sm_89`):** Hỗ trợ BF16 + `awq_marlin`, cùng kiến trúc `sm_89` với L4 cũ nhưng được cô lập SKU nghiêm ngặt (`matches_gpu_label` cấm dùng nhầm profile L4).
- **A100 (High — 80 GB, Ampere `sm_80`):** Hiển thị `A100` trên CLI nhưng luôn ánh xạ sang `A100-80GB` trên Modal SDK và từ chối mọi profile 40 GB ngay từ phía host lẫn trong container.

### 6. Bảng từ khóa tra cứu về giải mã suy đoán thích ứng
| Chủ đề | Từ khóa tìm kiếm học thuật |
| :--- | :--- |
| **Cơ sở toán học** | `Speculative decoding Leviathan ICML 2023`, `Rejection sampling distribution matching`, `Right-censored geometric MLE` |
| **Kiến trúc EAGLE** | `EAGLE feature-level speculative sampling`, `EAGLE-2 dynamic draft tree`, `EAGLE-3 training-time test` |
| **Thích ứng phần cứng** | `Hardware-aware speculative decoding`, `D-cut high-concurrency speculative decoding`, `Empirical step-cost profiling` |
| **Hạ tầng vLLM** | `vLLM AsyncScheduler dynamic_sd_lookup`, `vLLM PR #55708 spec_decode_metrics`, `Turing sm_75 AWQ vs Marlin` |

---

## Phần 3: Cơ chế tối ưu hóa tính toán chú ý: PagedAttention và FlashAttention

### 1. Cơ chế PagedAttention
- Chia KV cache thành **Physical Blocks** cố định và ánh xạ từ **Logical Blocks** thông qua **Block Table**, không yêu cầu vùng nhớ vật lý liên tục, giảm phân mảnh nội bộ từ $60\text{--}80\%$ xuống dưới $4\%$.
- Hỗ trợ **Copy-on-Write (COW)** cho các nhánh dùng chung tiền tố prompt (parallel sampling, beam search, prefix caching).

### 2. FlashAttention (IO-aware exact attention)
- Phép tính chú ý chuẩn $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})V$ tạo ma trận trung gian $N \times N$ gây nghẽn đọc/ghi HBM.
- **Tiling + Online Softmax:** Chia $Q, K, V$ vừa khít SRAM trên chip (SM), tính softmax tích lũy theo từng khối mà không bao giờ ghi ma trận $N \times N$ ra HBM, giảm bộ nhớ từ $O(N^2)$ xuống $O(N)$.
- **Pha hưởng lợi:** FlashAttention tăng tốc mạnh ở **prefill** (GEMM, $N$ lớn); trong khi **decode** ($q = 1$, GEMV) chủ yếu hưởng lợi từ **PagedAttention** và từ **speculative decoding** (biến $q = 1$ thành $q = k + 1$).

### 3. So sánh PagedAttention và FlashAttention
| Tiêu chí | PagedAttention | FlashAttention |
| :--- | :--- | :--- |
| **Mục tiêu chính** | Quản lý bộ nhớ VRAM (chống phân mảnh, chia sẻ block). | Giảm IO đọc/ghi giữa SRAM và HBM. |
| **Cơ chế cốt lõi** | Phân trang logic qua Block Table. | Tiling + Online Softmax ngay trong SRAM. |
| **Pha hưởng lợi chính** | **Decode phase** (phục vụ đồng thời nhiều request). | **Prefill phase** (prompt dài, ma trận $N \times N$). |
| **Kết hợp trong vLLM** | PagedAttention tổ chức bố cục bộ nhớ; kernel FlashAttention / Triton tính chú ý trực tiếp trên từng trang. |

---

## Phần 4: Lượng tử hóa mô hình (Quantization)

### 1. Nguyên lý và phân loại
- **PTQ (Post-Training Quantization):** Nén mô hình đã huấn luyện bằng tập hiệu chuẩn nhỏ, không cần huấn luyện lại (phương pháp dùng trong đồ án).
- **W4A16 (Weight-only 4-bit):** Lưu trọng số 4-bit trên VRAM và giải lượng tử nhanh trên SRAM thành FP16/BF16 khi tính toán — đánh trúng nút thắt băng thông nạp trọng số của pha decode.

### 2. So sánh RTN, GPTQ và AWQ
- **RTN (Round-to-Nearest):** Nhanh nhưng làm suy giảm nghiêm trọng chất lượng ở 4-bit.
- **GPTQ:** Dùng ma trận Hessian bậc hai bù sai số làm tròn theo cột, chất lượng tốt nhưng hiệu chuẩn chậm hơn.
- **AWQ (Activation-aware Weight Quantization — dùng trong đồ án):** Quan sát rằng chỉ ~1% kênh trọng số trọng yếu (salient weights, nhận diện qua biên độ activation) quyết định chất lượng mô hình; nhân hệ số bảo vệ theo kênh trước khi lượng tử hóa giúp giữ chất lượng sát FP16.

### 3. Vai trò thực tế trên 4 tầng GPU của đồ án
- Qwen2.5-7B FP16 chiếm ~14.5 GB VRAM (gần như lập tức OOM trên T4 16 GB ở $B > 4$). Bản **AWQ 4-bit** rút còn ~4.5–5.0 GB VRAM, mở đủ không gian KV cache cho `concurrency = 32` trên cả `T4`, `A10`, `L40S` và `A100`.
- **Khác biệt kernel bắt buộc:** Trên `T4` (`sm_75`), vLLM dùng `--quantization awq --dtype half` (`fp16-fallback`); trên `A10`, `L40S` và `A100` (`sm_80`/`sm_86`/`sm_89`), vLLM dùng `--quantization awq_marlin --dtype bfloat16` cho thông lượng giải lượng tử vượt trội.

---

## Phần 5: Chỉ số hiệu năng và quy trình benchmark thực tế của đồ án

### 1. Nhóm chỉ số cốt lõi
- **TTFT (ms):** Độ trễ pha prefill đến token đầu tiên.
- **TPOT (ms/token):** $\text{TPOT} = \dfrac{T_{\text{end}} - T_{\text{first\_token}}}{N_{\text{gen}} - 1}$ — **chỉ số mục tiêu số một**.
- **ITL (ms) & E2E Latency (ms):** Độ mượt streaming ($p50$, $p95$) và độ trễ toàn trình.
- **Output Throughput (tok/s) & Speedup vs. Vanilla:** Luôn so trong **cùng GPU, cùng workload, cùng concurrency**.
- **Chỉ số suy đoán & an toàn:** Tỷ lệ chấp nhận $\alpha$, số token sinh trung bình mỗi bước $\tau$, đỉnh VRAM (đo bằng NVML trên server và ghép qua `modal_vram_join.py`), và `Variance Ratio <= 15%`.

### 2. Công cụ và quy trình benchmark hiện hành (không dùng `vllm benchmark_serving` hay Langfuse)
- **Công cụ do nhóm tự xây dựng:**
  + [`experiments/benchmark_harness.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/benchmark_harness.py): Harness bắn tải bất đồng bộ dùng gói request đóng băng (SHA-256 verified), xoay mẫu `cycle-rotate`, tách riêng `warmup.jsonl` (3 warm-up) và `raw.jsonl` (5 repeats × 32 request = 160 request/ô).
  + [`experiments/run_benchmark_campaign.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/run_benchmark_campaign.py) + [`experiments/modal_benchmark_runner.py`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/experiments/modal_benchmark_runner.py) + `flexsds benchmark`: Tự động điều phối toàn bộ ma trận trên Modal app riêng `flexsds-benchmark` với vLLM `0.30.0` + bản vá backport `#55708` (chống rơi `spec_decode_metrics` khi stream), mỗi ô một container tươi (`fresh-server-per-cell`).
- **Ma trận benchmark 3 node × 4 GPU × 3 làn:**
  + **3 node (tác vụ pipeline hóa đơn VAT):** `W1_V3` (Node 1 trích xuất JSON hóa đơn toàn tờ, ngữ cảnh 8192, timeout 600 s), `W2` (Node 2 sinh SQL `w2_v2`, ngữ cảnh 4096), `W3` (Node 3 sinh báo cáo phân tích `w3_v2`, ngữ cảnh 4096).
  + **4 phần cứng Modal:** `T4` (16 GB), `A10` (24 GB), `L40S` (48 GB), `A100` (80 GB → `A100-80GB`).
  + **3 làn so sánh trên cùng cặp mô hình:**
    1. `vanilla`: chỉ chạy target model (tắt suy đoán).
    2. `static`: bật speculative decoding với `K` cố định (mặc định `K = 3`).
    3. `adaptive`: cùng cặp mô hình như `static`, bật thêm lớp Hardware-Adaptive (R6 geometric MLE + cost-driven K + `LineageUniProcExecutor`).
  + **2 chế độ concurrency:**
    - `full`: `(1, 2, 4, 8, 16, 32)` → **216 ô** (34,560 request đo).
    - `fast`: `(1, 8, 32)` → giữ nguyên 3 warm-up + 5 × 32 request/ô → **108 ô** (17,280 request đo).
- **Web demo trực quan (`FastAPI` + `Next.js`):** Kết nối trực tiếp các endpoint triển khai từ `flexsds serve` (nhập nhanh bằng tệp JSON xuất từ menu `Export API config JSON (for web demo)`), chạy thực tế pipeline 3 node và đo so sánh trực tiếp trên trình duyệt.

---

## Phần 6: Ma trận đối chiếu câu hỏi phản biện bảo vệ đồ án

| Câu hỏi tiềm năng của hội đồng phản biện | Kiến thức cốt lõi và bằng chứng trong hệ thống |
| :--- | :--- |
| *1. "Tại sao giải mã suy đoán nhanh hơn mà không làm giảm độ chính xác của mô hình gốc?"* | Trình bày **Modified Rejection Sampling** (Phần 2, Mục 1) đảm bảo $P_{\text{final}}(x) \equiv q(x)$, và chế độ kiểm chứng greedy lossless trong `experiments/lossless_verifier.py` / `m4_lossless.py` với `--no-prefix-caching` và `--match-adaptive-runtime`. |
| *2. "Vì sao không dùng speculative decoding tĩnh mặc định của vLLM mà phải xây dựng lớp Hardware-Adaptive?"* | Nháp tĩnh gặp 2 lỗi lớn: **sụt tốc (negative acceleration)** trên tác vụ entropy cao (`W3`) hoặc tải lớn (`c = 16, 32`), và nguy cơ **tràn VRAM (OOM)**. FlexSDS dùng R6 **geometric MLE + cost-driven K** theo profile đo thật `flexsds-component-v2` và rào chắn hysteresis memory guard để cắt về $K = 0$ đúng lúc. |
| *3. "Tại sao dùng Geometric MLE thay cho tỉ số trung bình `accepted / drafted`?"* | Do suy đoán tuần tự là phép thử hình học bị chặn phải (right-censored): sau token sai đầu tiên, mọi token nháp sau đều bị vứt dù chưa được đánh giá độc lập. Công thức $n_t = a_t + \mathbb{I}(a_t < k_t)$ của R6 khử sai lệch đánh giá thấp $\alpha$ khi $K$ lớn. |
| *4. "Hệ thống xử lý tính đa dạng phần cứng từ T4 đến A100 như thế nào?"* | Danh mục chuẩn 4 GPU (`T4` 16 GB, `A10` 24 GB, `L40S` 48 GB, `A100` 80 GB), tự động chuyển `half`/`awq` trên `T4` (`sm_75`) và `bfloat16`/`awq_marlin` trên Ampere/Ada, khóa khớp chặt SKU (không trộn `L4`/`L40S` hay `A100-40GB`/`80GB`), và hiệu chuẩn chi phí thực nghiệm riêng cho từng GPU. |
| *5. "Làm thế nào đảm bảo benchmark công bằng giữa `vanilla`, `static` và `adaptive`?"* | Đóng băng SHA nguồn và gói workload (`W1_V3`, `W2`, `W3`), dùng `cycle-rotate` đồng bộ mẫu, 3 warm-up + 5 × 32 request/ô, server container tươi mỗi ô (`server_instance_id` duy nhất), bản vá vLLM `#55708` chống rơi metrics, tùy chọn `--match-adaptive-runtime`, và tính `speedup_vs_vanilla` **độc lập theo từng GPU**. |
| *6. "Cơ chế PagedAttention và FlashAttention đóng vai trò gì trong vLLM?"* | PagedAttention phân trang KV cache thành block cố định chống phân mảnh (lợi nhất ở decode); FlashAttention dùng tiling + online softmax trong SRAM giảm IO HBM (lợi nhất ở prefill). |
| *7. "Tại sao phải lượng tử hóa AWQ 4-bit thay vì chạy FP16 hay RTN?"* | Qwen2.5-7B FP16 chiếm ~14.5 GB gây OOM trên T4 16 GB ở $B > 4$; RTN 4-bit hỏng chất lượng; AWQ bảo vệ 1% kênh trọng số trọng yếu theo phân phối activation nên rút còn ~5 GB mà giữ chất lượng sát FP16. |
| *8. "Làm sao đảm bảo hệ thống không bị OOM ở concurrency 32?"* | Áp dụng **Memory Pressure Rule có hysteresis**: liên tục theo dõi `kv_cache_usage` và `free_vram_mb`, cưỡng chế $k = 0$ khi KV $> 90\%$ hoặc VRAM $< 1500$ MB, giữ tối thiểu 3 bước cooldown và chỉ mở lại khi KV $< 80\%$. |
