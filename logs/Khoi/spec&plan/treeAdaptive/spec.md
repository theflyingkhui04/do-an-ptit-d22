# Đặc tả kỹ thuật nâng cấp TreeAdaptive từ phiên bản v0.1 lên v0.2

> **Tài liệu:** `spec.md`  
> **Dự án:** Nâng cấp bộ điều khiển thích ứng TreeAdaptive (Sprint 5 / M4)  
> **Cơ sở lý thuyết:** Kế thừa từ công trình OPT-Tree (*Wang et al., 2025, TACL*) và Speculative Decoding (*Leviathan et al., 2023*)  
> **Thành viên phụ trách:** Cấn Đức Khôi (Lead 70%), Bùi Đức Đại (Reviewer & Verification 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptive/spec.md`  
> **Trạng thái:** Bản phê duyệt kỹ thuật sau thẩm định (Revision 2 — Post Technical Review)  

---

## 1. Bối cảnh và lý do nâng cấp

Trong chiến dịch đo kiểm của Sprint 4 (mốc M3-05), nhóm nghiên cứu đã triển khai đánh giá 4 làn phục vụ trên hạ tầng GPU NVIDIA L4. Kết quả thực nghiệm cho thấy:
* **Hiện tượng tăng tốc âm trên văn bản tự do (Workload W3):** Cả chế độ nháp tĩnh ($k=3$) và bộ điều khiển cơ sở v0.1 (`AdaptiveDraftController`) đều gặp hiện tượng suy giảm hiệu năng trên tập Chat tự do. Cụ thể, hệ số tăng tốc TPOT của Adaptive trên W3 tại mọi mức concurrency $c \in \{1, 2, 4, 8, 16, 32\}$ đều nhỏ hơn 1.0 (dao động từ $0.745\times$ xuống $0.474\times$). Giả thuyết loại bỏ hoàn toàn tăng tốc âm ở phiên bản v0.1 trên W3 đã bị bác bỏ bởi dữ liệu đo đạc thực tế.
* **Nguyên nhân cốt lõi:**
  1. **Chuỗi nháp tuyến tính 1D ($k \in \{0, 2, 3, 4\}$):** Xác suất chấp nhận giảm nhanh theo cấp số nhân $\alpha^k$. Nếu token đầu tiên hoặc thứ hai bị từ chối, toàn bộ các token phía sau bị loại bỏ vô ích, gây lãng phí chi phí sinh nháp và xác thực.
  2. **Ngưỡng ra quyết định heuristic ghim cứng (`0.65` và `0.80`):** Không phản ánh đúng sự biến thiên phi tuyến tính của chi phí forward trên GPU khi batch size tăng cao.
  3. **Thiếu cơ chế kẹp tải đồng thời:** Khi nhiều luồng cùng chạy ($B \ge 16$), việc duy trì số lượng token nháp lớn làm bùng nổ chi phí xác thực, gây nghẽn băng thông bộ nhớ.

Phiên bản **TreeAdaptive v0.2** được thiết kế nhằm giải quyết triệt để các hạn chế trên, kế thừa nền tảng lý thuyết từ bài báo **OPT-Tree** (*Wang et al., 2025: Speculative Decoding with Adaptive Draft Tree Structure*) để tối ưu hóa kỳ vọng số token được chấp thuận trên cấu trúc cây phân nhánh, kết hợp mô hình chi phí phần cứng và cơ chế ngắt an toàn hai tầng.

---

## 2. Bảng đối chiếu tổng quan giữa phiên bản v0.1 và v0.2

| Tiêu chí kỹ thuật | TreeAdaptive v0.1 (Sprint 3–4) | TreeAdaptive v0.2 (Sprint 5 — Chuẩn OPT-Tree) |
| :--- | :--- | :--- |
| **Không gian nháp (Search space)** | Chuỗi tuyến tính 1D: $k \in \{0, 2, 3, 4\}$ tokens. | Cấu trúc cây 2D: Cặp $(d, b)$ với topology catalog xác định $N \in \{0, 4, 8, 16, 32\}$. |
| **Hàm mục tiêu kỳ vọng token** | Chuỗi thẳng cơ bản: $1 + \sum_{\ell=1}^k \alpha^\ell$. | Chuẩn xác suất OPT-Tree: $\mathbb{E}[\tau(T)] = 1 + \sum_{v \in T \setminus \{r\}} q(v)$. |
| **Cơ chế xác định ngưỡng ngắt** | Ngưỡng tĩnh heuristic ghim cứng (`0.65` và `0.80`). | Giải nghiệm hòa vốn động theo thời gian thực $\alpha^*(T, B)$ từ mô hình chi phí GPU. |
| **Mô hình chi phí phần cứng** | Ước lượng tuyến tính giả định (`base_t + k * 3ms`). | Tra cứu ma trận thực nghiệm $C(T, B)$ kết hợp mô hình hóa chi phí xác thực $C_V(N, B)$. |
| **Độ nhạy với mức tải (Batch size $B$)** | Không nhận biết rõ rệt sự trôi dạt chi phí theo batch size. | Tự động thích ứng ngưỡng $\alpha^*$ và kẹp trần kích thước cây theo $B \in \{1, 2, 4, 8, 16, 32\}$. |
| **Cơ chế thăm dò (Probing)** | Chu kỳ cố định 20 bước giải mã ($k=2$). | Chu kỳ thăm dò động theo mức tải (15 bước khi $B \le 4$, 20 bước khi $4 < B \le 16$, 30 bước khi $B > 16$). |
| **Cơ chế bảo vệ bộ nhớ** | Hysteresis 1 tầng trên bộ đệm KV ($90\% \to 80\%$). | Hysteresis 2 tầng kết hợp giữa VRAM vật lý ($< 1500\text{ MB}$), bộ đệm KV và cooldown 3 bước. |
| **Bảo toàn chất lượng (Lossless)** | Khẳng định chung chung. | Phân định rõ: Bảo toàn phân phối ($P_{\text{spec}} = P_{\text{target}}$) và khớp token tuyệt đối khi greedy. |
| **Đường dẫn serving chính thức** | `src/proposer/controller.py` | `src/proposer/tree_adaptive.py` (tích hợp trực tiếp qua `src/proposer/adapter.py`). |

---

## 3. Kiến trúc kỹ thuật chi tiết của TreeAdaptive v0.2

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
│    • Memory Guard: Ngắt về N=0 nếu KV > 90% HOẶC Free VRAM < 1500 MB   │
│    • Hysteresis Cooldown: Giữ tối thiểu 3 bước, khôi phục khi KV < 80% │
│      và VRAM >= 1500 MB; không thăm dò vượt rào khi bộ nhớ nguy cấp    │
│    • Concurrency Throttling: Kẹp trần N <= 8 khi B >= 16; N <= 4 khi   │
│      B >= 24; tự động ngắt N=0 nếu alpha_effective < 0.85              │
├────────────────────────────────────────────────────────────────────────┤
│ 3. BỘ TÍNH TOÁN KỲ VỌNG VÀ GIẢI NGƯỠNG HÒA VỐN ĐỘNG (OPT-TREE SOLVER)   │
│    • Kế thừa công thức OPT-Tree: E[tau(T)] = 1 + sum q(v)              │
│    • Nạp ma trận chi phí thực nghiệm C(T, B) từ profile GPU            │
│    • Giải phương trình hòa vốn alpha*(T_min, B) qua tìm kiếm nhị phân  │
│    • Dynamic Fallback Gate: Nếu alpha_effective < alpha*, chọn N = 0   │
│    • Adaptive Probing: Thăm dò T4 sau mỗi T_probe(B) bước ngắt an toàn │
├────────────────────────────────────────────────────────────────────────┤
│ 4. BỘ ĐẶC TẢ TOPO CÂY VÀ SINH MẶT NẠ (TREE TOPOLOGY & ATTENTION MASK)  │
│    • Định nghĩa 5 cấu hình chuẩn: parent_indices, position_ids duy nhất│
│    • Sinh Tree Attention Mask 2D mở rộng bảo đảm tính causal           │
│    • Argmax tỉ số tốc độ: T* = argmax (E[tau(T)] * C(0, B) / C(T, B))  │
├────────────────────────────────────────────────────────────────────────┤
│ 5. GIAO DIỆN TÍCH HỢP SCHEDULER & TELEMETRY (CANONICAL SERVING GLUE)   │
│    • Đấu nối trực tiếp vào src/proposer/adapter.py và AsyncScheduler   │
│    • Cắt tỉa placeholder nháp per-request trimming và gán padding mask │
│    • Xuất log telemetry per-step đạt chuẩn Interface A v0.2            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Đặc tả chi tiết các phân hệ chức năng nâng cấp

### 4.1. Cơ sở lý thuyết kỳ vọng chấp thuận trên cây nháp (Kế thừa từ OPT-Tree)

Theo công trình nghiên cứu OPT-Tree (*Wang et al., 2025*):
* **Định nghĩa không gian cây:**
  Cho một cây nháp ứng viên $\mathcal{T} = (V, \mathcal{E})$ với node gốc $r$ (đại diện cho token ngữ cảnh hiện tại). Mỗi node $v \in V \setminus \{r\}$ biểu diễn một token ứng viên được sinh ra bởi đầu nháp.
  Độ sâu $\text{depth}(v)$ là khoảng cách từ gốc $r$ tới $v$.
  Chuỗi tiền tố dẫn tới node $v$ được ký hiệu là $\text{prefix}(v) = (u_1, u_2, \dots, u_\ell = v)$.

* **Quy tắc xác thực tham lam (Greedy verification):**
  Trong quá trình xác thực song song của mô hình đích:
  - Một node $v$ được chấp nhận khi và chỉ khi token tại $v$ trùng khớp với đầu ra của mô hình đích VÀ toàn bộ các node tổ tiên trong $\text{prefix}(v)$ đều đã được chấp nhận.
  - Tại mỗi độ sâu $\ell$, mô hình đích chỉ chọn duy nhất một token theo giải mã tham lam. Do đó, các sự kiện $\{\text{prefix}(v) \text{ được chấp nhận}\}$ đối với các node khác nhau ở cùng độ sâu $\ell$ là **các biến cố xung khắc (mutually exclusive)**.
  - Tập hợp các token nháp được chấp nhận luôn tạo thành một đường đi đơn duy nhất bắt đầu từ gốc $r$ (accepted prefix path).

* **Công thức kỳ vọng độ dài chấp nhận chính xác:**
  Gọi $L(\mathcal{T})$ là biến ngẫu nhiên biểu diễn số lượng token nháp được chấp nhận ($0 \le L(\mathcal{T}) \le d$).
  Kỳ vọng của biến ngẫu nhiên nguyên không âm được tính bằng tổng đuôi xác suất:
  $$\mathbb{E}[L(\mathcal{T})] = \sum_{\ell=1}^d \mathbb{P}(L(\mathcal{T}) \ge \ell)$$
  Do các nhánh cạnh tranh ở cùng độ sâu $\ell$ là xung khắc, xác suất có ít nhất một đường đi độ dài $\ell$ được chấp thuận chính bằng tổng xác suất của từng node ở độ sâu đó:
  $$\mathbb{P}(L(\mathcal{T}) \ge \ell) = \sum_{v \in \text{Nodes}(\mathcal{T}, \ell)} \mathbb{P}(\text{prefix}(v) \text{ accepted})$$
  Do đó:
  $$\mathbb{E}[L(\mathcal{T})] = \sum_{\ell=1}^d \sum_{v \in \text{Nodes}(\mathcal{T}, \ell)} q(v) = \sum_{v \in V \setminus \{r\}} q(v)$$
  Trong đó $q(v) \triangleq \mathbb{P}(\text{prefix}(v) \text{ accepted})$.

* **Tổng số token phát sinh trong một bước suy đoán ($\tau$):**
  Khi kết thúc bước xác thực, mô hình đích luôn phát sinh thêm một token mới (bonus token):
  $$\tau(\mathcal{T}) = 1 + L(\mathcal{T})$$
  $$\mathbb{E}[\tau(\mathcal{T})] = 1 + \sum_{v \in V \setminus \{r\}} q(v)$$

* **Xấp xỉ qua xác suất phân phối của mô hình nháp và hợp đồng toán học với scalar $\alpha$:**
  Trong thực tế serving runtime, controller nhận tín hiệu chất lượng chấp nhận trung bình $\alpha \in [0, 1]$ (EMA acceptance rate). Để liên kết scalar $\alpha$ với vector xác suất node $q(v)$ trên cây, hệ thống thiết lập hợp đồng toán học rõ ràng (stated approximation contract):
  - Tại mỗi độ sâu $\ell \in [1, d]$, có $n_\ell$ ứng viên cạnh tranh (`layer_distribution[l]`).
  - Dưới giải mã tham lam (greedy decoding), mô hình đích chỉ sinh duy nhất 1 token đúng tại mỗi vị trí giải mã; các nhánh cùng tầng là xung khắc. Do đó, xác suất bước (step acceptance probability) có ít nhất một ứng viên khớp tại tầng $\ell$ khi node cha đã được chấp nhận được mô hình hóa theo độ bao phủ nhánh:
    $$P_{\text{step}}(\ell, \alpha) = 1 - (1 - \alpha)^{n_\ell}$$
  - Xác suất chuỗi tiền tố được chấp nhận liên tục tới độ sâu $\ell$:
    $$P_{\text{path}}(\ell, \alpha) = \prod_{k=1}^\ell P_{\text{step}}(k, \alpha)$$
  - Kỳ vọng số token phát sinh $\mathbb{E}[\tau(\mathcal{T}, \alpha)]$ được định nghĩa:
    $$\mathbb{E}[\tau(\mathcal{T}, \alpha)] = 1 + \sum_{\ell=1}^d P_{\text{path}}(\ell, \alpha)$$
  - **Các tính chất bất biến được bảo đảm:**
    1. Đơn điệu tăng theo $\alpha \in [0.0, 1.0]$.
    2. Chặn dưới chặt: $\mathbb{E}[\tau(\mathcal{T}, 0.0)] = 1.0$ (chỉ token gốc được phát sinh).
    3. Chặn trên chặt: $\mathbb{E}[\tau(\mathcal{T}, 1.0)] = 1 + d$ (đường dẫn dài nhất tới độ sâu tối đa $d$).
    4. Không vượt quá $1 + d$ với mọi $\alpha \in [0, 1]$.

* **Trường hợp suy biến về chuỗi tuyến tính (Linear chain):**
  Khi cây $\mathcal{T}$ là chuỗi thẳng độ dài $k$ ($b=1$, $n_\ell = 1$ với mọi $\ell$), ta có $P_{\text{step}}(\ell, \alpha) = \alpha$, dẫn tới $P_{\text{path}}(\ell, \alpha) = \alpha^\ell$. Khi đó công thức suy biến chính xác về:
  $$\mathbb{E}[\tau(k, \alpha)] = 1 + \sum_{\ell=1}^k \alpha^\ell$$
  Hoàn toàn nhất quán với lý thuyết suy đoán tuyến tính cổ điển.

---

### 4.2. Danh mục cấu trúc cây nháp chuẩn hóa (Discrete Tree Catalog Specification)

Nhằm đảm bảo tương thích với cơ chế tối ưu hóa đồ thị CUDA (CUDA Graph) của vLLM và tránh cấp phát động tensor mặt nạ trong vòng lặp giải mã, hệ thống chuẩn hóa 5 cấu hình cây cố định với cấu trúc hình học được định nghĩa duy nhất:

#### 1. Cấu hình $T_0$ — Fallback / Chế độ tuần tự cơ sở
* **Tổng số node:** $N = 0$, độ sâu $d = 0$.
* **Ý nghĩa:** Tắt hoàn toàn suy đoán, mô hình đích forward 1 token tuần tự.

#### 2. Cấu hình $T_4$ — Chuỗi nháp sâu
* **Tổng số node:** $N = 4$, độ sâu $d = 4$, hệ số phân nhánh $b = 1$.
* **Mảng quan hệ cha-con (`parent_indices`):** `[-1, 0, 1, 2, 3]` (với -1 là gốc).
* **Mảng vị trí (`position_ids`):** `[0, 1, 2, 3, 4]`.
* **Mặt nạ chú ý cây:** Ma trận tam giác dưới thông thường kích thước $5 \times 5$.

#### 3. Cấu hình $T_8$ — Cây phân nhánh cân bằng nhỏ
* **Tổng số node:** $N = 8$, độ sâu $d = 3$, cấu trúc phân nhánh:
  * Tầng 1 (depth 1): 2 node con (node 1, 2).
  * Tầng 2 (depth 2): node 1 có 2 con (node 3, 4); node 2 có 2 con (node 5, 6).
  * Tầng 3 (depth 3): node 3 có 1 con (node 7); node 5 có 1 con (node 8).
* **Mảng quan hệ cha-con (`parent_indices`):** `[-1, 0, 0, 1, 1, 2, 2, 3, 5]`.
* **Mảng vị trí (`position_ids`):** `[0, 1, 1, 2, 2, 2, 2, 3, 3]`.

#### 4. Cấu hình $T_{16}$ — Cây cân bằng rộng
* **Tổng số node:** $N = 16$, độ sâu $d = 4$.
* **Phân bổ node theo tầng:** $[1, 3, 6, 4, 3]$ (gồm gốc và 16 node nháp).
* **Mảng quan hệ cha-con (`parent_indices`):**  
  `[-1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 6, 8, 10, 12, 13]`.  
  *(Đã sửa lỗi S04: node 16 có cha là node 13 ở độ sâu 3, bảo đảm độ sâu của node 16 là đúng 4).*
* **Mảng vị trí (`position_ids`):**  
  `[0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]`.
* **Đặc tính:** Tối ưu hóa cho đầu nháp EAGLE-3.1 khi tỷ lệ chấp thuận trung bình $\alpha \in [0.75, 0.88]$.

#### 5. Cấu hình $T_{32}$ — Cây mở rộng tối đa
* **Tổng số node:** $N = 32$, độ sâu $d = 5$.
* **Phân bổ node theo tầng:** $[1, 4, 8, 8, 6, 6]$ (gồm gốc và 32 node nháp).
* **Mảng quan hệ cha-con duy nhất (`parent_indices` — 33 phần tử):**  
  `[-1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]`.  
  *(Đã xác định duy nhất và kiểm chứng 100% qua validator code).*
* **Mảng vị trí (`position_ids` — 33 phần tử):**  
  `[0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]`.
* **Điều kiện kích hoạt:** Chỉ cho phép kích hoạt khi batch size thấp ($B \le 2$) và $\alpha_{\text{effective}} \ge 0.90$.

* **Quy tắc sinh Tree Attention Mask:**
  Cho một topology $\mathcal{T}$ gồm $N+1$ token (tính cả token gốc). Ma trận mặt nạ chú ý $M \in \{0, 1\}^{(N+1) \times (N+1)}$ được sinh theo nguyên tắc nhân quả trên cây:
  $$M_{i, j} = \begin{cases} 1, & \text{nếu node } j \text{ là tổ tiên của node } i \text{ hoặc } j = i \\ 0, & \text{ngược lại (hai nhánh khác nhau không thể nhìn thấy nhau)} \end{cases}$$
  Toàn bộ 5 ma trận mặt nạ này được tiền tính toán (pre-compute) thành các tensor hằng số nạp sẵn vào VRAM khi khởi động engine.

---

### 4.3. Bộ giải ngưỡng hòa vốn động nhận biết phần cứng (Dynamic Break-Even Solver)

* **Hệ số tăng tốc dự báo (Predicted Speedup):**
  Tại mức tải đồng thời $B$, đối với cấu hình cây $\mathcal{T}$, hệ số tăng tốc lý thuyết được định nghĩa:
  $$S(\mathcal{T}, B, \alpha) = \frac{\mathbb{E}[\tau(\mathcal{T}, \alpha)] \cdot C(0, B)}{C(\mathcal{T}, B)}$$
  Trong đó:
  * $C(0, B)$ là thời gian thực thi một bước forward của mô hình đích ở chế độ tuần tự cơ sở.
  * $C(\mathcal{T}, B)$ là thời gian thực thi bước suy đoán gồm forward nháp và xác thực song song mô hình đích qua Tree Attention Mask.
  * Lưu ý khoa học: $S(\mathcal{T}, B, \alpha)$ là tốc độ dự báo theo mô hình toán nhằm phục vụ thuật toán ra quyết định; giá trị này độc lập và cần được đối chứng với tốc độ đo thực tế $S_{\text{measured}}$.

* **Phương trình điểm hòa vốn và thuật toán giải:**
  Điểm hòa vốn xuất hiện khi tốc độ dự báo bằng đúng chế độ cơ sở: $S(\mathcal{T}, B, \alpha^*) = 1.0$:
  $$\mathbb{E}[\tau(\mathcal{T}, \alpha^*)] = \frac{C(\mathcal{T}, B)}{C(0, B)}$$
  Hệ thống sử dụng thuật toán tìm kiếm nhị phân (Bisection search) trên đoạn $[0.0, 1.0]$ với 50 vòng lặp (thời gian tính toán $< 0.01\text{ ms}$).
  * Trường hợp biên và xử lý vô nghiệm (Infeasible & No-Root Handling):  
    Do độ dài chuỗi token được chấp nhận không thể vượt quá độ sâu tối đa $d$ của cây, ta có chặn trên lý thuyết:
    $$\mathbb{E}[\tau(\mathcal{T}, \alpha)] \le 1 + d$$
    Nếu tỉ số chi phí $\frac{C(\mathcal{T}, B)}{C(0, B)} > 1 + d$, phương trình hòa vốn chắc chắn vô nghiệm trên $[0, 1]$ (không tồn tại bất kỳ $\alpha \in [0, 1]$ nào để đạt tốc độ bằng $1.0$). Khi đó, bộ giải trả về trạng thái rõ ràng `infeasible = True` (hoặc `status: "no_root"`), tuyệt đối không gán $\alpha^* = 1.0$ giả định để tránh làm sai lệch tín hiệu điều khiển.

* **Quy tắc chuyển làn an toàn (Dynamic Fallback Gate):**
  * Tại mỗi bước suy luận, bộ điều khiển đánh giá hệ số tăng tốc dự báo $S(\mathcal{T}, B, \alpha_{\text{effective}})$ trên toàn bộ danh mục cây ứng viên $\mathcal{T} \in \{T_4, T_8, T_{16}, T_{32}\}$.
  * Nếu $\max_{\mathcal{T} \in \text{Catalog}} S(\mathcal{T}, B, \alpha_{\text{effective}}) \le 1.0$: Hệ thống chuyển về chế độ cơ sở $T_0$ ($N=0$), giúp giảm thiểu tối đa nguy cơ suy giảm tốc độ (negative speedup) khi chất lượng đề xuất nháp suy giảm hoặc GPU bị quá tải.

---

### 4.4. Mô hình chi phí thực nghiệm đa chiều (Multi-Dimensional Cost Model)

Tổng chi phí thực thi bước suy đoán:
$$C(\mathcal{T}, B) = t_{\text{draft}}(\mathcal{T}, B) + C_V(\mathcal{T}, B)$$
* **Chi phí sinh nháp $t_{\text{draft}}(\mathcal{T}, B)$:** Đo trực tiếp từ đầu nháp EAGLE-3.1 (khoảng 3–5 ms trên NVIDIA L4).
* **Chi phí xác thực song song $C_V(\mathcal{T}, B)$:**
  Thời gian forward mô hình đích trên tập $N$ token ứng viên kèm Tree Attention Mask được mô hình hóa qua hệ số phạt chú ý $\gamma_{\text{attn}}$:
  $$C_V(\mathcal{T}, B) = C_T(B) \cdot \left( 1 + \gamma_{\text{attn}} \cdot \frac{N}{64} \right)$$
  Các hệ số này được hiệu chuẩn từ ma trận đo thực nghiệm lưu trong `src/flexsds/profiles/sm_89_l4.json` và `sm_75_t4.json`.

---

### 4.5. Cơ chế phòng vệ hai tầng và điều tiết tải cao (Dual-Tier Guard & Throttling)

1. **Bộ phòng vệ bộ nhớ hai tầng (Dual-Tier Memory Guard):**
   * **Điều kiện ngắt an toàn:** Kích hoạt ngắt ngay về $T_0$ ($N=0$) khi:
     $$\text{kv\_cache\_usage\_pct} > 90.0\% \quad \text{HOẶC} \quad \text{free\_vram\_mb} < 1500\text{ MB}$$
   * **Cơ chế hồi phục trễ (Hysteresis Recovery):** Bắt buộc duy trì tối thiểu 3 bước cooldown ở $T_0$. Sau 3 bước, hệ thống CHỈ cho phép phục hồi suy đoán khi:
     $$\text{kv\_cache\_usage\_pct} \le 80.0\% \quad \text{VÀ} \quad \text{free\_vram\_mb} \ge 1500\text{ MB}$$
     Nếu dung lượng VRAM vẫn dưới 1500 MB, tuyệt đối không tự động phục hồi dù KV-cache đã hạ.

2. **Cơ chế thăm dò thích ứng mức tải (Load-Aware Adaptive Probing):**
   * Khi đang ngắt an toàn ($T_0$), định kỳ phát một bước thăm dò với $T_4$ để làm mới tín hiệu $\alpha$.
   * Chu kỳ thăm dò $T_{\text{probe}}(B)$:
     $$T_{\text{probe}}(B) = \begin{cases} 15 \text{ bước}, & B \le 4 \\ 20 \text{ bước}, & 4 < B \le 16 \\ 30 \text{ bước}, & B > 16 \end{cases}$$
   * **Quy tắc bất biến:** Tuyệt đối không phát bước thăm dò nếu Memory Guard đang bị kích hoạt do thiếu hụt VRAM vật lý.

3. **Kẹp trần kích thước cây theo luồng đồng thời (Concurrency Throttling):**
   * Nếu $B \le 4$: Cho phép cấu hình tối đa $T_{32}$ ($N \le 32$).
   * Nếu $4 < B \le 12$: Kẹp trần tối đa $T_{16}$ ($N \le 16$).
   * Nếu $12 < B \le 24$: Kẹp trần tối đa $T_8$ ($N \le 8$).
   * Nếu $B > 24$: Kẹp trần tối đa $T_4$ ($N \le 4$), đồng thời ngắt về $T_0$ nếu $\alpha_{\text{effective}} < 0.85$.

---

### 4.6. Chuẩn hóa giao diện dữ liệu Telemetry (Interface A v0.2)

Dữ liệu telemetry per-step xuất ra theo đúng định dạng JSON chuẩn phục vụ phân tích:

```jsonc
{
  "request_id": "req_tree_v02_001",
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
    "parent_indices": [-1, 0, 0, 1, 1, 2, 2, 3, 5]
  },
  "mean_accepted_tokens_tau": 3.65,
  "theoretical_speedup": 1.28,
  "actual_step_latency_ms": 22.4,
  "memory_status": {
    "free_vram_mb": 14250,
    "kv_cache_usage_pct": 64.2,
    "guard_triggered": false
  },
  "decision_reason": "Optimal budget T8 (Speedup 1.28x, alpha 0.835 >= alpha* 0.685 at B=8)"
}
```

---

## 5. Yêu cầu phi chức năng và tính đúng đắn khoa học

1. **Phân định rõ ràng tính chất bảo toàn chất lượng (Lossless Guarantee):**
   * **Bảo toàn phân phối (Distribution preservation):** Thuật toán tuân thủ nguyên lý của *Leviathan et al. (2023)*, đảm bảo phân phối xác suất đầu ra không bị lệch lạc: $P_{\text{spec}}(Y) = P_{\text{target}}(Y)$.
   * **Khớp chuỗi token tham lam (Greedy Token Equality 100%):** Khi giải mã ở nhiệt độ `temperature = 0`, chuỗi token sinh ra bởi TreeAdaptive v0.2 bắt buộc phải khớp chính xác 100% từng token so với giải mã tuần tự cơ sở (Vanilla) trên 50/50 test prompt.
2. **Nguyên tắc an toàn không sập ứng dụng (Zero-Crash Guarantee):**
   * Mọi ngoại lệ trong quá trình tính toán cây hoặc đọc thông số phần cứng phải được bắt lại và chuyển an toàn về chế độ tuần tự cơ sở ($T_0$), bảo đảm không gián đoạn luồng serving.
3. **Độ trễ quyết định (Overhead ceiling):**
   * Toàn bộ pipeline tính toán (EMA, tra cứu kỳ vọng, giải nghiệm $\alpha^*$, chọn cây) phải hoàn tất trong thời gian $\le 0.05\text{ ms}$ cho mỗi lượt gọi.
4. **Vị trí tích hợp runtime chính thức:**
   * Mã nguồn phục vụ chính thức nằm tại `src/proposer/tree_adaptive.py`, được nạp qua `src/proposer/adapter.py` để bảo đảm phục vụ trực tiếp inference runtime của dự án.
