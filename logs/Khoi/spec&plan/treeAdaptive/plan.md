# Kế hoạch triển khai nâng cấp TreeAdaptive từ phiên bản v0.1 lên v0.2

> **Tài liệu:** `plan.md`  
> **Dự án:** Kế hoạch thực thi nâng cấp bộ điều khiển thích ứng TreeAdaptive (Sprint 5 / M4)  
> **Cơ sở đối chiếu:** Báo cáo thẩm định kỹ thuật `tree.pdf` (06/11/2026) và lý thuyết OPT-Tree (*Wang et al., 2025*)  
> **Thành viên:** Cấn Đức Khôi (Lead 70%), Bùi Đức Đại (Reviewer & Benchmark Lead 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/treeAdaptive/plan.md`  
> **Hạn chốt hoàn thành:** 12/11/2026 (trước mốc đóng băng toàn diện 15/11/2026)  
> **Trạng thái:** Bản phê duyệt kế hoạch sau thẩm định (Revision 2 — Post Technical Review)  

---

## 1. Mục tiêu và lộ trình thực thi song song

Kế hoạch này phân rã toàn bộ công việc nâng cấp **TreeAdaptive v0.2** thành **6 task tương ứng 6 thẻ Trello (`M4-02-01` đến `M4-02-06`)**. Nhằm triệt tiêu hoàn toàn điểm nghẽn (bottleneck) giữa hai thành viên, các công việc được ghép cặp theo mô hình **3 chặng song song độc lập**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│             LỘ TRÌNH THỰC THI SONG SONG TREEADAPTIVE V0.2              │
├────────────────────────────────────────────────────────────────────────┤
│ CHẶNG 1: THIẾT KẾ CÂY NHÁP & ĐO PROFILE PHẦN CỨNG (05/11 – 07/11)      │
│  ├─ [M4-02-01] (Khôi Lead): Module cấu trúc topo cây nháp (CPU/repo)    │
│  └─ [M4-02-02] (Đại Lead): Đo ma trận chi phí C(T, B) trên Modal L4/T4 │
│                                                                        │
│ CHẶNG 2: BỘ GIẢI HÒA VỐN ĐỘNG & TÍCH HỢP SCHEDULER (07/11 – 09/11)     │
│  ├─ [M4-02-03] (Khôi Lead): FlexCostModel & Bisection Solver alpha*    │
│  └─ [M4-02-04] (Đại Lead): AsyncScheduler glue & Telemetry v0.2 log    │
│                                                                        │
│ CHẶNG 3: HOÀN THIỆN ĐIỀU KHIỂN & ĐO KIỂM ĐÓNG BĂNG (09/11 – 12/11)     │
│  ├─ [M4-02-05] (Khôi Lead): Tái cấu trúc TreeAdaptiveController v0.2   │
│  └─ [M4-02-06] (Đại Lead): Test lossless 100%, sweep 4 làn & Báo cáo   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết phân rã công việc (Task breakdown)

### [M4-02-01] Hiện thực hóa module cấu trúc topo cây nháp (`tree_topology.py`)
* **Mục tiêu:**
  * Xây dựng lớp dữ liệu `TreeTopology` và danh mục 5 topo chuẩn: $T_0$ (Vanilla, $N=0$), $T_4$ (Chuỗi sâu, $N=4$), $T_8$ (Cây cân bằng nhỏ, $N=8$), $T_{16}$ (Cây cân bằng rộng, $N=16$), $T_{32}$ (Cây mở rộng, $N=32$).
  * Định nghĩa duy nhất cho từng cấu hình: `parent_indices`, `position_ids`, số lượng node con ở mỗi nhánh.
  * Lập trình hàm sinh ma trận Tree Attention Mask 2D mở rộng bảo đảm tính nhân quả (causal mask): các nhánh song song không nhìn thấy nhau.
  * Cài đặt bảng tra cứu hàm kỳ vọng chấp thuận theo đúng chuẩn công thức OPT-Tree:
    $$\mathbb{E}[\tau(T)] = 1 + \sum_{v \in T \setminus \{r\}} q(v)$$
* **Các tệp tạo/sửa đổi:**
  * `src/flexsds/core/tree_topology.py`
  * `tests/test_tree_topology.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Khởi tạo chính xác 5 cấu hình cây với tổng số node $N$, độ sâu $d$ và mảng `parent_indices` khớp đặc tả.
  - Ma trận Tree Attention Mask sinh ra đúng chuẩn causal mask, vượt qua bài kiểm tra tính độc lập giữa các nhánh (branch exclusivity test).
  - Hàm tính kỳ vọng $\mathbb{E}[\tau]$ tăng đơn điệu theo $\alpha$, có thời gian tra cứu tức thì ($< 0.005\text{ ms}$).
  - 100% Pass bộ unit test kiểm tra thuộc tính đồ thị cây.
* **Người thực hiện:** Khôi (Lead 70%), Đại (Reviewer 30%) — **Thời gian:** 05/11 – 07/11/2026.

---

### [M4-02-02] Đo ma trận chi phí thực nghiệm $C(T, B)$ và sinh profile GPU trên Modal L4/T4
* **Mục tiêu:**
  * Viết script micro-benchmark đo lường độc lập trên Modal GPU để xác định thời gian forward thực tế của mô hình đích Qwen2.5-7B và đầu nháp EAGLE-3.1.
  * Đo quét toàn diện ma trận chi phí trên 30 ô cấu hình cho mỗi GPU: 5 topology ($T_0, T_4, T_8, T_{16}, T_{32}$) $\times$ 6 mức batch size ($B \in \{1, 2, 4, 8, 16, 32\}$).
  * Hiệu chuẩn hệ số phạt xác thực song song qua attention mask $\gamma_{\text{attn}}$ trên GPU NVIDIA L4 (sm_89) và Tesla T4 (sm_75).
  * Đóng gói kết quả thành các tệp profile chuẩn hóa: `src/flexsds/profiles/sm_89_l4.json` và `src/flexsds/profiles/sm_75_t4.json`.
* **Các tệp tạo/sửa đổi:**
  * `scripts/benchmark_tree_cost_matrix.py`
  * `src/flexsds/profiles/sm_89_l4.json`
  * `src/flexsds/profiles/sm_75_t4.json`
  * `experiments/runs/m4_cost_profiling/`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Thu thập đủ dữ liệu thực nghiệm cho 60 ô đo lường (30 ô/GPU), có ghi nhận metadata phần cứng, số lần lặp (repetitions $\ge 50$) và loại bỏ warm-up.
  - Các tệp JSON profile chứa đầy đủ các trường: `target_c_t`, `draft_c_d`, `cost_matrix_ms`, `gamma_attn`.
  - Dữ liệu đo được kiểm tra không có nhiễu đột biến (outliers) do khởi động lạnh container.
* **Người thực hiện:** Đại (Lead 70%), Khôi (Partner 30%) — **Thời gian:** 05/11 – 07/11/2026.

---

### [M4-02-03] Nâng cấp FlexCostModel và bộ giải nghiệm hòa vốn động $\alpha^*(T, B)$
* **Mục tiêu:**
  * Nâng cấp lớp `FlexCostModel` trong `src/flexsds/core/cost_model.py` để nạp ma trận thực nghiệm $C(T, B)$ từ tệp profile JSON do Đại cung cấp ở Task 2.
  * Cài đặt thuật toán tìm kiếm nhị phân (Bisection search) 50 bước giải nhanh phương trình điểm hòa vốn:
    $$\mathbb{E}[\tau(T, \alpha^*)] = \frac{C(T, B)}{C(0, B)}$$
  * Xử lý chặt chẽ các trường hợp biên: nghiệm không tồn tại khi tỉ số chi phí $> 1 + N$ (gán $\alpha^* = 1.0$), hoặc chi phí suy đoán rẻ hơn tuần tự (gán $\alpha^* = 0.0$).
  * Tích hợp chế độ phân tích giải tích (Analytical Fallback) tự động ước lượng khi chạy trên dòng GPU mới chưa có tệp profile.
* **Các tệp tạo/sửa đổi:**
  * `src/flexsds/core/cost_model.py`
  * `tests/test_cost_model_solver.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Thời gian giải $\alpha^*$ cho một cấu hình $< 0.01\text{ ms}$.
  - Sai số số học (numerical residual) $|S(T, B, \alpha^*) - 1.0| < 10^{-4}$ trên các trường hợp có nghiệm.
  - 100% Pass các ca unit test nhân tạo có nghiệm giải tích biết trước và kiểm thử các điều kiện biên.
* **Người thực hiện:** Khôi (Lead 70%), Đại (Partner 30%) — **Thời gian:** 07/11 – 09/11/2026.

---

### [M4-02-04] Tích hợp vLLM AsyncScheduler, bộ đệm lập lịch và chuẩn hóa Telemetry Interface A v0.2
* **Mục tiêu:**
  * Nâng cấp `AdaptiveBudgetPlanner` trong `src/server/budget_planner.py` nhận diện cấu hình cây $T_i$ cho từng request.
  * Cập nhật `AdaptiveAsyncScheduler` trong `src/server/adaptive_scheduler.py` đồng bộ kích thước cây lớn nhất trong batch $N_{\text{batch}} = \max(N_i)$ vào bảng tra cứu `dynamic_sd_lookup`.
  * Hiện thực hóa cơ chế cắt tỉa placeholder nháp (per-request trimming) và gán padding mask cho các request có $N_i < N_{\text{batch}}$ tại hook `_update_after_schedule`.
  * Chuẩn hóa định dạng log telemetry theo đúng chuẩn Interface A v0.2 (đầy đủ các trường `dynamic_alpha_star`, `selected_topology`, `theoretical_speedup`, `memory_status`, `decision_reason`).
* **Các tệp tạo/sửa đổi:**
  * `src/server/budget_planner.py`
  * `src/server/adaptive_scheduler.py`
  * `src/proposer/adapter.py`
  * `tests/test_scheduler_tree_glue.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - `AdaptiveBudgetPlanner` chạy độc lập trên CPU host, vượt qua 100% test mô phỏng multi-request batching.
  - Không gây xung đột luồng hoặc race condition trong async scheduling loop.
  - Tệp log telemetry ghi nhận đầy đủ 100% các trường dữ liệu chuẩn Interface A v0.2.
* **Người thực hiện:** Đại (Lead 70%), Khôi (Partner 30%) — **Thời gian:** 07/11 – 09/11/2026.

---

### [M4-02-05] Tái cấu trúc bộ điều khiển TreeAdaptiveController v0.2 đa tầng
* **Mục tiêu:**
  * Hiện thực hóa lớp `TreeAdaptiveController` tại đường dẫn chuẩn runtime `src/proposer/tree_adaptive.py` (kế thừa `BudgetController`) và đấu nối trực tiếp vào `src/proposer/adapter.py`.
  * Tích hợp Dual-Tier Memory Guard: ngắt về $T_0$ ($N=0$) khi VRAM trống $< 1500\text{ MB}$ HOẶC KV-cache $> 90\%$. Duy trì trễ hysteresis cooldown 3 bước và chỉ phục hồi khi KV-cache $\le 80\%$ VÀ VRAM trống $\ge 1500\text{ MB}$.
  * Cài đặt Concurrency Throttling kẹp trần ngân sách cây theo luồng đồng thời: $B \le 4 \implies N \le 32$; $4 < B \le 12 \implies N \le 16$; $12 < B \le 24 \implies N \le 8$; $B > 24 \implies N \le 4$ (và ngắt $T_0$ nếu $\alpha_{\text{effective}} < 0.85$).
  * Tích hợp Dynamic Fallback Gate qua nghiệm $\alpha^*(T_4, B)$ và cơ chế thăm dò thích ứng chu kỳ $T_{\text{probe}}(B)$ (15, 20, 30 bước).
  * Bảo đảm nguyên tắc Zero-Crash: bắt mọi ngoại lệ tính toán và tự động fallback về $T_0$ mà không làm gián đoạn luồng serving.
* **Các tệp tạo/sửa đổi:**
  * `src/proposer/tree_adaptive.py`
  * `src/proposer/adapter.py`
  * `tests/test_tree_adaptive_controller.py`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Thời gian ra quyết định chọn topo cây tối ưu $T^*$ hoàn tất trong $< 0.05\text{ ms}$ cho mỗi lượt gọi.
  - Khi mô phỏng áp lực bộ nhớ hoặc VRAM $< 1500\text{ MB}$, bộ điều khiển lập tức trả về $T_0$ và không phát lệnh thăm dò vượt rào.
  - Khi $\alpha < \alpha^*(T_4, B)$, hệ thống chuyển về $T_0$ và kích hoạt thăm dò đúng chu kỳ $T_{\text{probe}}(B)$.
  - Đạt 100% Pass các kịch bản kiểm thử chuyển trạng thái (State Transition Scenarios).
* **Người thực hiện:** Khôi (Lead 70%), Đại (Partner 30%) — **Thời gian:** 09/11 – 11/11/2026.

---

### [M4-02-06] Xây dựng bộ kiểm thử hồi quy lossless và đo kiểm sweep đối chứng đóng băng M4
* **Mục tiêu:**
  * Xây dựng bài kiểm tra Lossless Guarantee (`tests/test_lossless_tree_verification.py`): so sánh khớp token 100% giữa TreeAdaptive v0.2 và Vanilla baseline khi giải mã tham lam (`temperature = 0`) trên 50 prompt chuẩn (Workload W1, W2, W3).
  * Thiết kế bài test stress-test 1.000 bước kiểm tra độ ổn định và rò rỉ bộ nhớ (memory leak) khi chuyển làn liên tục Vanilla — Static — TreeAdaptive.
  * Khởi chạy chiến dịch đo sweep đối chứng 4 làn trên Modal GPU NVIDIA L4 24GB trên 3 workloads $\times$ 6 mức concurrency ($c \in \{1, 2, 4, 8, 16, 32\}$) = 72 cấu hình (hoặc 18 cặp so sánh trực tiếp v0.1 vs v0.2).
  * Thu thập đầy đủ dữ liệu TPOT, Throughput, TTFT, tỉ lệ chấp thuận $\alpha$ và log telemetry JSONL.
  * Phân tích đối chứng khoa học, cập nhật bảng số liệu và biểu đồ vào bản thảo Chương 4 quyển đồ án (`docs/chuong-4-thuc-nghiem.md`).
  * Gắn tag phiên bản `v0.4.0-m4-freeze` đóng băng toàn bộ mã nguồn trước hạn 15/11.
* **Các tệp tạo/sửa đổi:**
  * `tests/test_lossless_tree_verification.py`
  * `experiments/runs/m4_treeadaptive/`
  * `experiments/results/summary_treeadaptive_v02.csv`
  * `docs/chuong-4-thuc-nghiem.md`
* **Tiêu chuẩn nghiệm thu (DoD):**
  - TokenMatchRate = 100% (so khớp chính xác tuyệt đối 50/50 prompts so với greedy decoding gốc).
  - Hoàn tất 100% các lượt chạy sweep theo đúng protocol, có lưu đầy đủ log thực nghiệm JSONL và file kết quả CSV.
  - Phân tích trung thực kết quả thực nghiệm: ghi nhận đầy đủ dù giả thuyết tăng tốc có được hỗ trợ hay không (tách bạch giữa tính hợp lệ của thí nghiệm và kết quả speedup).
  - Viết tối thiểu 2 trang phân tích kết quả vào bản thảo Chương 4 quyển báo cáo tốt nghiệp.
* **Người thực hiện:** Đại (Lead 70%), Khôi (Partner 30%) — **Thời gian:** 09/11 – 12/11/2026.

---

## 3. Ma trận trách nhiệm (RACI Matrix)

| Thẻ công việc Trello | Cấn Đức Khôi | Bùi Đức Đại | Vai trò phối hợp chi tiết |
| :--- | :---: | :---: | :--- |
| **[M4-02-01] Module cấu trúc topo cây** | **A / R (70%)** | C / I (30%) | Khôi code cấu trúc dữ liệu cây; Đại review tính nhân quả của mặt nạ |
| **[M4-02-02] Đo ma trận chi phí thực nghiệm** | C / I (30%) | **A / R (70%)** | Đại chạy micro-benchmark trên Modal; Khôi review schema JSON |
| **[M4-02-03] Nâng cấp FlexCostModel & Solver** | **A / R (70%)** | C / I (30%) | Khôi code bisection solver; Đại thẩm định tính hội tụ nghiệm |
| **[M4-02-04] Tích hợp Scheduler & Telemetry** | C / I (30%) | **A / R (70%)** | Đại tích hợp scheduler glue; Khôi review per-request trimming |
| **[M4-02-05] Tái cấu trúc TreeAdaptiveController**| **A / R (70%)** | C / I (30%) | Khôi lập trình bộ điều khiển đa tầng; Đại thiết kế test chuyển trạng thái |
| **[M4-02-06] Kiểm thử lossless & Sweep Modal** | C / I (30%) | **A / R (70%)** | Đại điều phối sweep trên L4; Khôi trực xử lý lỗi và phân tích log |

---

## 4. Quản trị rủi ro kỹ thuật (Risk mitigation)

1. **Rủi ro 1: vLLM v0.30.0 dynamic_sd_lookup chỉ ánh xạ số nguyên $K$ chứ không nhận cây nháp tùy ý:**
   * *Giải pháp:* Tại Sprint 5, toàn bộ cấu trúc cây nháp chuẩn hóa ($T_0, T_4, T_8, T_{16}, T_{32}$) được quản lý bởi đầu nháp EAGLE-3.1 thông qua module `TreeProposer` và kernel Tree Attention Mask của EAGLE. `AdaptiveAsyncScheduler` chỉ cần truyền kích thước node trần $N_{\text{batch}} = \max(N_i)$ cho bước cấp phát bộ nhớ đệm, sau đó áp dụng per-request trimming để gán mặt nạ che (padding mask) cho các luồng con.
2. **Rủi ro 2: Sai lệch mô hình chi phí khi chưa có profile GPU thực tế:**
   * *Giải pháp:* Task [M4-02-02] của Đại chạy độc lập ngay từ chặng đầu tiên trên GPU NVIDIA L4 để tạo profile JSON thực tế trước khi Khôi hoàn tất bộ giải ở Task [M4-02-03], tránh việc solver phải dựa trên số liệu giả định.
3. **Rủi ro 3: Bị tụt tốc độ do overhead ra quyết định của bộ điều khiển:**
   * *Giải pháp:* Tiền tính toán (pre-compute) toàn bộ ma trận chú ý và bảng tra cứu kỳ vọng $\mathbb{E}[\tau(T, \alpha)]$ vào mảng hằng số tĩnh, đảm bảo bước chọn cây hoàn tất trong $< 0.05\text{ ms}$.
