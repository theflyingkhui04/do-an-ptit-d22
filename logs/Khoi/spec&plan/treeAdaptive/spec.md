# Đặc tả kỹ thuật nâng cấp TreeAdaptive từ phiên bản v0.1 lên v0.2

> **Tài liệu:** `spec.md`  
> **Dự án:** Nâng cấp bộ điều khiển thích ứng TreeAdaptive (Sprint 5 / M4)  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead 70%), Bùi Đức Đại (Reviewer 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptive/spec.md`  
> **Trạng thái:** Bản phê duyệt kỹ thuật chính thức  

---

## 1. Bối cảnh và lý do nâng cấp

Trong các đợt chạy thực nghiệm của Sprint 3 và Sprint 4 (mốc M2-05 và M3-05), nhóm nghiên cứu đã hiện thực hóa thành công bộ điều khiển thích ứng cơ sở **TreeAdaptive v0.1** (được triển khai qua lớp `AdaptiveDraftController`). Phiên bản v0.1 đã đạt được mục tiêu cơ bản nhất: **triệt tiêu hiện tượng tăng tốc âm (negative speedup)** trên tập văn bản tự do (Workload W3) và đảm bảo an toàn bộ nhớ VRAM không bị tràn (OOM).

Tuy nhiên, khi tiến hành đo đạc mở rộng trên hạ tầng đám mây Modal GPU NVIDIA L4 và chuẩn bị cho đợt đóng băng toàn diện (15/11 Freeze), phiên bản v0.1 bộc lộ 4 giới hạn kỹ thuật lớn:

1. **Ngưỡng ra quyết định bị cứng hóa (Hardcoded static thresholds):**
   * v0.1 sử dụng các ngưỡng heuristic cố định: ngưỡng ngắt an toàn `fallback_threshold = 0.65` và ngưỡng mở rộng `high_threshold = 0.80`.
   * Các ngưỡng này được ước lượng dựa trên thực nghiệm ở mức tải đơn luồng ($B=1$). Khi mức độ phục vụ đồng thời tăng cao ($B \in \{8, 16, 32\}$), thời gian forward của mô hình đích $C_T(B)$ và mô hình nháp $C_D(B)$ thay đổi phi tuyến tính do nghẽn băng thông bộ nhớ. Hệ quả là ngưỡng $0.65$ không còn phản ánh đúng điểm hòa vốn thực tế, khiến hệ thống phản ứng chậm hoặc bỏ lỡ cơ hội bứt phá tốc độ.
2. **Bó hẹp trong chuỗi nháp tuyến tính (Linear chain draft limitation):**
   * v0.1 chỉ điều chỉnh độ sâu nháp $k \in \{0, 2, 3, 4\}$ theo một chuỗi thẳng duy nhất.
   * Do xác suất chấp nhận giảm theo cấp số nhân $\alpha^k$, nếu token nháp thứ nhất hoặc thứ hai bị từ chối, toàn bộ các token phía sau bị loại bỏ vô ích. Phiên bản v0.1 chưa tận dụng được cấu trúc cây phân nhánh (Tree drafting topology) để mở rộng không gian tìm kiếm ứng viên của đầu nháp chuyên biệt EAGLE-3.1.
3. **Mô hình chi phí phần cứng còn đơn giản:**
   * Hàm tính toán chi phí trong v0.1 giả định mối quan hệ tuyến tính thô sơ ($20\text{ ms} + k \times 3\text{ ms}$), chưa tích hợp đầy đủ ma trận chi phí thực nghiệm $C(N, B)$ được đo lường chính xác từ vi kiến trúc NVIDIA Ada Lovelace (sm_89) và Turing (sm_75).
4. **Kiểm soát tải cao chưa nhận biết áp lực luồng đồng thời:**
   * Cơ chế phòng vệ của v0.1 chỉ dựa vào phần trăm bộ nhớ đệm KV và dung lượng VRAM trống, chưa chủ động kẹp trần kích thước nháp khi số lượng request đồng thời trong batch vượt ngưỡng bão hòa tính toán của GPU.

Phiên bản **TreeAdaptive v0.2** được thiết kế nhằm giải quyết triệt để các hạn chế trên, nâng cấp từ một bộ điều khiển heuristic tuyến tính thành **hệ thống điều khiển cấu trúc cây nháp thích ứng động nhận biết phần cứng (Hardware-Aware Dynamic Tree Speculative Controller)**.

---

## 2. Bảng đối chiếu tổng quan giữa phiên bản v0.1 và v0.2

| Tiêu chí kỹ thuật | TreeAdaptive v0.1 (Sprint 3–4) | TreeAdaptive v0.2 (Sprint 5 — Đóng băng M4) |
| :--- | :--- | :--- |
| **Không gian nháp (Search space)** | Chuỗi tuyến tính 1D: $k \in \{0, 2, 3, 4\}$ tokens. | Cấu trúc cây 2D: Cặp tham số $(d, b)$ và tổng số node cây $N \in \{0, 4, 8, 16, 32\}$. |
| **Cơ chế xác định ngưỡng ngắt** | Ngưỡng tĩnh heuristic ghim cứng (`0.65` và `0.80`). | Giải nghiệm hòa vốn động theo thời gian thực $\alpha^*(N, B)$ từ ma trận chi phí GPU. |
| **Mô hình chi phí phần cứng** | Ước lượng tuyến tính giả định (`base_t + k * 3ms`). | Tra cứu ma trận thực nghiệm $C(N, B)$ và mô hình hóa chi phí xác thực song song $C_V(N, B)$. |
| **Độ nhạy với mức tải (Batch size $B$)** | Không nhận biết rõ rệt sự trôi dạt chi phí theo batch size. | Tự động thích ứng ngưỡng $\alpha^*$ và kẹp trần kích thước cây theo $B \in \{1, 2, 4, 8, 16, 32\}$. |
| **Cơ chế thăm dò (Probing)** | Chu kỳ cố định 20 bước giải mã ($k=2$). | Chu kỳ thăm dò động theo mức tải (15 bước khi tải nhẹ, 30 bước khi tải nặng). |
| **Cơ chế bảo vệ bộ nhớ** | Hysteresis 1 tầng trên bộ đệm KV ($90\% \to 80\%$). | Hysteresis 2 tầng kết hợp giữa VRAM vật lý, bộ đệm KV và áp lực luồng đang chờ. |
| **Khả năng tương thích đầu nháp** | Chủ yếu phục vụ mô hình nháp độc lập (SLM 0.5B). | Tối ưu hóa song song cho cả SLM 0.5B và đầu nháp chuyên biệt EAGLE-3.1 tự huấn luyện. |
| **Chuẩn dữ liệu Telemetry** | Interface A v0.1 cơ bản (`k`, `alpha`). | Interface A v0.2 mở rộng (`tree_topology`, $\alpha^*$, `theoretical_speedup`, `reason`). |

---

## 3. Kiến trúc kỹ thuật chi tiết của TreeAdaptive v0.2

Kiến trúc v0.2 được tổ chức thành 5 phân hệ chức năng độc lập nhưng phối hợp đồng bộ:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   KIẾN TRÚC TREEADAPTIVE CONTROLLER V0.2               │
├────────────────────────────────────────────────────────────────────────┤
│ 1. BỘ PHÂN TÍCH VÀ CẬP NHẬT TÍN HIỆU (TELEMETRY & SIGNAL SMOOTHING)     │
│    • Thu nhận (accepted_tokens, drafted_tokens) sau mỗi bước forward   │
│    • Hiệu chỉnh sai lệch khởi động: Unbiased EMA với beta = 0.8         │
│    • Trích xuất thông số runtime: Batch size B, Free VRAM, KV Cache %  │
├────────────────────────────────────────────────────────────────────────┤
│ 2. BỘ PHÒNG VỆ HAI TẦNG VÀ KẸP TẢI CAO (TWO-TIER GUARD & THROTTLING)   │
│    • Memory Guard: Ngắt về N=0 nếu KV > 90% hoặc Free VRAM < 1500 MB   │
│    • Hysteresis Cooldown: Giữ tối thiểu 3 bước, khôi phục khi KV < 80% │
│    • Concurrency Throttling: Kẹp trần N <= 8 khi B >= 16               │
├────────────────────────────────────────────────────────────────────────┤
│ 3. BỘ GIẢI NGƯỠNG HÒA VỐN ĐỘNG (DYNAMIC BREAK-EVEN SOLVER)             │
│    • Nạp ma trận chi phí thực nghiệm C(N, B) từ profile GPU            │
│    • Giải phương trình hòa vốn alpha*(N_min, B) qua tìm kiếm nhị phân  │
│    • Dynamic Fallback Gate: Nếu alpha_effective < alpha*, chọn N = 0   │
│    • Adaptive Probing: Thăm dò N=4 sau mỗi T_probe(B) bước ngắt        │
├────────────────────────────────────────────────────────────────────────┤
│ 4. BỘ TỐI ƯU HÓA CẤU TRÚC CÂY (TREE TOPOLOGY OPTIMIZER)                │
│    • Tính hàm kỳ vọng token chấp nhận trên cây: E[tau(Tree, alpha)]    │
│    • Argmax tỉ số tốc độ: N* = argmax (E[tau] * C(0, B) / C(N, B))     │
│    • Chọn bộ ba cấu trúc cây tối ưu: (Độ sâu d, Số nhánh b, Tổng node N│
├────────────────────────────────────────────────────────────────────────┤
│ 5. GIAO DIỆN TÍCH HỢP SCHEDULER & TELEMETRY (GLUE & CONTRACT)          │
│    • Xuất cấu hình cây sang vLLM AsyncScheduler qua dynamic_sd_lookup  │
│    • Sinh Tree Attention Mask và Tree Position IDs cho pha Verify      │
│    • Ghi nhận đầy đủ log thực nghiệm chuẩn hóa theo Interface A v0.2   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Đặc tả chi tiết các phân hệ chức năng nâng cấp

### 4.1. Bộ giải ngưỡng hòa vốn động nhận biết phần cứng (Hardware-Aware Dynamic Break-Even Solver)

* **Bản chất toán học:**
  Điểm hòa vốn xuất hiện khi tốc độ sinh từ của chế độ suy đoán bằng đúng tốc độ sinh từ của chế độ tuần tự cơ sở (Vanilla).
  Tại mức tải $B$, với một cấu hình cây gồm $N$ node nháp, hệ số tăng tốc lý thuyết được định nghĩa:
  $$\text{Speedup}(N, B, \alpha) = \frac{\mathbb{E}[\tau(N, \alpha)] \cdot C(0, B)}{C(N, B)}$$
  Trong đó:
  * $C(0, B)$ là thời gian forward một bước của mô hình đích ở chế độ tuần tự ($N=0$).
  * $C(N, B)$ là tổng thời gian thực thi của bước suy đoán gồm forward nháp và xác thực song song.
  * $\mathbb{E}[\tau(N, \alpha)]$ là số lượng token kỳ vọng được mô hình lớn chấp thuận.

* **Thuật toán giải nghiệm $\alpha^*(N, B)$:**
  Ngưỡng chấp nhận tối thiểu để không bị giảm tốc độ là nghiệm của phương trình $\text{Speedup}(N, B, \alpha^*) = 1.0$:
  $$\mathbb{E}[\tau(N, \alpha^*)] = \frac{C(N, B)}{C(0, B)}$$
  Hệ thống sử dụng thuật toán tìm kiếm nhị phân (Bisection search) trên đoạn $[0.0, 1.0]$ với số vòng lặp tối đa 50 bước. Do hàm $\mathbb{E}[\tau(N, \alpha)]$ đơn điệu tăng theo $\alpha$, nghiệm $\alpha^*$ hội tụ chính xác trong thời gian dưới $0.01\text{ ms}$, không gây trễ bước lập lịch.

* **Quy tắc chuyển làn tự động (Dynamic Fallback Rule):**
  * Gọi $N_{\text{min}}$ là cấu hình nháp nhỏ nhất khác 0 trong danh mục (ví dụ $N=4$).
  * Tính $\alpha^*(N_{\text{min}}, B)$.
  * Nếu $\alpha_{\text{effective}} < \alpha^*(N_{\text{min}}, B)$: Bộ điều khiển lập tức hạ ngân sách về $N=0$ (ngắt hoàn toàn suy đoán, đưa hệ thống về chế độ an toàn).
  * Điều này đảm bảo triệt tiêu hoàn toàn vùng tăng tốc âm trên mọi kiến trúc phần cứng mà không cần hiệu chỉnh tay bất kỳ hệ số nào.

---

### 4.2. Bộ biểu diễn và tối ưu hóa cấu trúc cây nháp (Tree Topology Optimizer)

Trong phiên bản v0.1, ngân sách chỉ là một số nguyên $k$. Trong v0.2, cấu trúc nháp được biểu diễn bởi đối tượng `TreeTopology`:

* **Định nghĩa cấu trúc cây:**
  Mỗi cấu hình cây được xác định bởi bộ ba thông số:
  1. `depth` ($d$): Độ sâu tối đa của cây (số bước suy đoán lớn nhất có thể đạt được).
  2. `branching_factor` ($b$): Hệ số phân nhánh tối đa tại mỗi tầng.
  3. `total_nodes` ($N$): Tổng số token ứng viên nằm trên toàn bộ các nhánh của cây.

* **Danh mục các cấu hình cây rời rạc (Discrete Tree Catalog):**
  Để tránh việc phải biên dịch lại đồ thị CUDA (CUDA Graph) liên tục trong vLLM, v0.2 chuẩn hóa 5 mốc topo cây cố định:
  1. **$T_0$ — Fallback/Tuần tự:** $N = 0, d = 0, b = 0$ (tắt suy đoán).
  2. **$T_4$ — Chuỗi hẹp sâu:** $N = 4, d = 4, b = 1$ (tương đương chuỗi tuyến tính $k=4$, tối ưu cho tác vụ cú pháp cao như JSON).
  3. **$T_8$ — Cây cân bằng nhỏ:** $N = 8, d = 3, b = 2$ (cấu trúc phân nhánh nhẹ, phù hợp cho Text-to-SQL).
  4. **$T_{16}$ — Cây cân bằng rộng:** $N = 16, d = 4, b = [3, 2, 2, 1]$ (tối ưu hóa cho đầu nháp EAGLE-3.1).
  5. **$T_{32}$ — Cây mở rộng tối đa:** $N = 32, d = 5, b = [4, 3, 2, 2, 1]$ (chỉ kích hoạt ở mức tải thấp $B \le 2$ và $\alpha > 0.90$).

* **Tính toán kỳ vọng chấp nhận trên cây $\mathbb{E}[\tau(T, \alpha)]$:**
  Trên cấu trúc cây, nhiều nhánh có thể được chấp nhận song song một phần. Kỳ vọng token chấp nhận bằng tổng xác suất chấp nhận trên đường đi dài nhất hợp lệ từ gốc đến lá:
  $$\mathbb{E}[\tau(T, \alpha)] = 1 + \sum_{l=1}^{d} \left( 1 - \prod_{p \in \text{Paths}(l)} (1 - \alpha^l) \right)$$
  Hàm tính toán này được tiền lập trình trong bảng tra cứu (lookup table) giúp bộ điều khiển tính toán tức thì trong chu kỳ ra quyết định.

---

### 4.3. Mô hình chi phí thực nghiệm đa chiều (Multi-Dimensional Cost Model)

Phiên bản v0.2 thay thế mô hình chi phí ước lượng của v0.1 bằng ma trận đo kiểm thực nghiệm thu được từ chiến dịch M3-02:

* **Công thức tổng quát của bước giải mã suy đoán:**
  $$C(N, B) = t_{\text{draft}}(N, B) + C_V(N, B)$$
  Trong đó:
  * $t_{\text{draft}}(N, B)$ là thời gian sinh nháp của mô hình phụ. Đối với đầu nháp EAGLE-3.1 (1 layer transformer), chi phí này chỉ bằng $\sim 20-25\%$ so với mô hình nháp SLM 0.5B.
  * $C_V(N, B)$ là thời gian mô hình đích thực hiện bước kiểm tra xác thực đồng thời toàn bộ $N$ token qua mặt nạ chú ý cây (Tree Attention Mask):
    $$C_V(N, B) = C_T(B) \cdot \left( 1 + \gamma_{\text{attn}} \cdot \frac{N}{64} \right)$$
    Hệ số phạt chú ý $\gamma_{\text{attn}}$ được đo thực nghiệm: $\gamma_{\text{attn}} \approx 0.08$ trên GPU NVIDIA L4 và $\approx 0.15$ trên Tesla T4.

* **Cơ chế nạp tự động theo kiến trúc vi xử lý:**
  Hệ thống đọc mã kiến trúc từ `HardwareInfo` (ví dụ `sm_89` cho L4 hoặc `sm_75` cho T4) và tự động ánh xạ tới tệp hồ sơ JSON tương ứng trong thư mục `profiles/`. Nếu gặp dòng GPU chưa được đo sẵn, hệ thống kích hoạt chế độ phân tích giải tích (Analytical Fallback) bảo đảm tính hoạt động liên tục.

---

### 4.4. Cơ chế phòng vệ quá tải và phân tầng bộ nhớ (Concurrency Throttling & Memory Guard)

1. **Bộ kẹp trần kích thước cây theo luồng đồng thời (Concurrency Throttling):**
   * Khi số lượng request đang giải mã trong batch $B$ tăng cao, năng lực tính toán của Tensor Core và băng thông bộ nhớ bị chia sẻ. Nếu vẫn duy trì cây lớn ($N=16$ hoặc $N=32$), chi phí xác thực $C_V(N, B)$ sẽ bùng nổ, kéo tụt hiệu năng chung.
   * v0.2 áp dụng quy tắc kẹp trần chủ động:
     * Nếu $B \le 4$: Ngân sách cây tối đa cho phép là $N_{\text{max}} = 32$.
     * Nếu $4 < B \le 12$: Kẹp trần ngân sách cây $N_{\text{max}} = 16$.
     * Nếu $12 < B \le 24$: Kẹp trần ngân sách cây $N_{\text{max}} = 8$.
     * Nếu $B > 24$: Kẹp trần ngân sách cây $N_{\text{max}} = 4$, và tự động ngắt về $N=0$ nếu $\alpha_{\text{effective}} < 0.85$.

2. **Cơ chế bảo vệ bộ nhớ hai tầng (Dual-Tier Memory Guard):**
   * Giám sát đồng thời hai chỉ số: dung lượng VRAM thực tế còn rảnh (`free_vram_mb`) và tỷ lệ sử dụng bảng trang bộ nhớ đệm KV (`kv_cache_usage_pct`).
   * **Ngưỡng kích hoạt ngắt:** `kv_cache_usage_pct > 90.0%` HOẶC `free_vram_mb < 1500 MB`. Khi chạm ngưỡng, hệ thống lập tức ép $N = 0$.
   * **Cơ chế hồi phục trễ (Hysteresis Recovery):** Sau khi kích hoạt ngắt bộ nhớ, hệ thống bắt buộc duy trì chế độ an toàn $N=0$ trong tối thiểu 3 bước liên tiếp (`memory_cooldown_steps = 3`), và chỉ cho phép phục hồi suy đoán khi tỷ lệ sử dụng bộ đệm KV hạ xuống dưới ngưỡng an toàn `80.0%`. Điều này triệt tiêu hoàn toàn hiện tượng dao động bật/tắt liên tục (ping-pong effect).

3. **Cơ chế thăm dò chu kỳ thích ứng tải (Load-Aware Adaptive Probing):**
   * Khi hệ thống đang ở trạng thái ngắt an toàn ($N=0$), mô hình đích tự giải mã tuần tự nên không có token nháp nào được đưa vào xác thực, dẫn đến giá trị $\alpha$ bị đóng băng.
   * Để giải phóng hệ thống khỏi bẫy đóng băng (Deadlock), v0.2 duy trì cơ chế phát động định kỳ một bước thăm dò với $N=4$.
   * Tần suất thăm dò được điều chỉnh theo tải:
     $$T_{\text{probe}}(B) = \begin{cases} 15 \text{ bước}, & \text{khi } B \le 4 \\ 20 \text{ bước}, & \text{khi } 4 < B \le 16 \\ 30 \text{ bước}, & \text{khi } B > 16 \end{cases}$$
     Nhờ đó, ở tải cao hệ thống không bị lãng phí tài nguyên cho các bước thăm dò không cần thiết.

---

### 4.5. Chuẩn hóa giao diện dữ liệu Telemetry (Interface A v0.2)

Dữ liệu telemetry của từng request được trả về qua endpoint hoặc ghi vào log phục vụ phân tích đối chứng đạt chuẩn:

```jsonc
{
  "request_id": "req_adaptive_v02_001",
  "engine_version": "treeadaptive_v0.2",
  "hardware_profile": "sm_89_l4",
  "concurrency_batch_b": 8,
  "acceptance_rate_ema": 0.842,
  "effective_alpha": 0.835,
  "dynamic_alpha_star": 0.685,
  "selected_topology": {
    "name": "T8_balanced",
    "total_nodes_N": 8,
    "depth_d": 3,
    "branching_b": 2
  },
  "mean_accepted_tokens_tau": 3.65,
  "theoretical_speedup": 1.28,
  "actual_step_latency_ms": 22.4,
  "memory_status": {
    "free_vram_mb": 14250,
    "kv_cache_usage_pct": 64.2,
    "guard_triggered": false
  },
  "decision_reason": "Optimal budget N=8 (Speedup 1.28x, alpha 0.835 >= alpha* 0.685 at B=8)"
}
```

---

## 5. Yêu cầu phi chức năng và tiêu chí an toàn

1. **Độ trễ tính toán của bộ điều khiển (Decision Overhead):**
   * Toàn bộ logic tính toán (tính EMA, giải nghiệm hòa vốn $\alpha^*$, tra cứu chi phí và chọn cấu hình cây tối ưu) phải hoàn tất trong thời gian $\le 0.08\text{ ms}$ cho mỗi request.
   * Tuyệt đối không được trở thành điểm nghẽn (bottleneck) làm tăng độ trễ của scheduler.
2. **Nguyên tắc an toàn không sập ứng dụng (Zero-Crash Guarantee):**
   * Nếu có bất kỳ lỗi ngoại lệ nào xảy ra trong quá trình tính toán cây hoặc đọc thông số phần cứng, bộ điều khiển phải bắt lỗi và tự động fallback về chế độ tuần tự cơ sở ($N=0$) mà không làm gián đoạn luồng sinh token của vLLM.
3. **Bảo đảm không suy giảm chất lượng (Lossless Guarantee 100%):**
   * Việc co giãn cấu trúc cây và số lượng node nháp chỉ ảnh hưởng đến tốc độ thực thi, không được làm thay đổi phân phối xác suất đầu ra của mô hình đích. Kết quả văn bản sinh ra trên mọi tác vụ (JSON, SQL, Chat) phải khớp chính xác với giải mã tham lam của mô hình gốc Qwen2.5-7B.
4. **Tính tương thích ngược (Backward Compatibility):**
   * Module `TreeAdaptiveController` v0.2 phải kế thừa hoặc cung cấp giao diện tương thích hoàn toàn với các unit test và launcher hiện có của Làn 3 và Làn 4.
