# Báo cáo kỹ thuật và quyết định PM: [M4-02-04] Hiện tượng ép phẳng cấu trúc cây trên vLLM 0.30.0, chiến lược phân tách phạm vi và kiến trúc async lineage v0.2

- **Mã vấn đề:** `PROB-M4-02-04-TREE-FLATTENING-LINEAGE-V02`
- **Cột mốc liên quan:** `M4 (Adaptive Speculative Decoding & Tree Scheduling)`
- **Ngày ghi nhận & đánh giá:** 10/11/2026 – 12/11/2026
- **Người phát hiện & triển khai kỹ thuật:** Bùi Đức Đại (Dev + IT / MLOps & Benchmark)
- **Người xem xét & phê duyệt quyết định:** Cấn Đức Khôi (Dev + PM / AI Systems)
- **Tài liệu căn cứ PM:**
  - [Pm decision m4 02 04.pdf](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec&plan/treeAdaptiveController/Pm%20decision%20m4%2002%2004.pdf)
  - [Pm decision m4 02 04_2.pdf](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec&plan/treeAdaptiveController/Pm%20decision%20m4%2002%2004_2.pdf)
- **Trạng thái:** **RESOLVED VIA SCOPE SPLIT (Đã giải quyết theo Option B: Hoãn Tree-Adaptive sang nhánh `future/tree-adaptive-v02`, chuẩn hóa canonical serving về linear scalar adaptive và hoàn thiện async lineage v0.2)**

---

## 1. Bối cảnh và mục tiêu ban đầu

Trong khuôn khổ mốc công việc `[M4-02-04] Scheduler Telemetry / Adaptive Scalar Budget Integration`, mục tiêu ban đầu của nhóm là đưa bộ điều khiển thích ứng dạng cây (`TreeAdaptiveController`) vào serving runtime của vLLM v0.30.0 để thực thi thuật toán giải mã đầu cơ thích ứng cấu trúc cây (Adaptive Tree Speculative Decoding).

Kỳ vọng ban đầu:
- Bộ điều khiển tính toán và lựa chọn topology cây phù hợp cho từng request dựa trên tỷ lệ chấp nhận $\alpha_{EMA}$ và điều kiện phần cứng (ví dụ: $T_0, T_4, T_8, T_{16}$).
- Bộ lập lịch (`AdaptiveAsyncScheduler`) nhận diện số node của cây và đưa vào pipeline xử lý của vLLM.
- vLLM và mô hình nháp sinh cây token, áp dụng mặt nạ chú ý cây (tree attention mask) và mô hình mục tiêu xác thực toàn bộ cây theo đường dẫn phân nhánh hợp lệ (valid root-to-leaf paths).

Tuy nhiên, trong quá trình tích hợp và kiểm thử thực tế trên vLLM v0.30.0, đội ngũ kỹ thuật đã phát hiện một rào cản nền tảng trong kiến trúc runtime của vLLM, đe dọa trực tiếp đến tính đúng đắn khoa học của đồ án.

---

## 2. Bản chất kỹ thuật chi tiết của lỗi tree flattening trên vLLM 0.30.0

### 2.1. Giới hạn bản chất của vLLM v0.30.0 Dynamic Speculative Decoding
Theo mã nguồn và tài liệu kiến trúc của vLLM v0.30.0:
- Cơ chế Dynamic Speculative Decoding của vLLM chỉ hỗ trợ một đại lượng vô hướng duy nhất là số lượng speculative tokens $K$ dựa trên kích thước batch $B$:
  $$B \longrightarrow K \quad (K \in \mathbb{N})$$
- vLLM v0.30.0 **hoàn toàn không hỗ trợ biểu diễn hay truyền tải cấu trúc cây tùy ý (arbitrary tree topology)**. Một cấu trúc cây toán học đầy đủ bắt buộc phải chứa bộ tham số:
  $$\mathcal{T}_i = (N_i, d_i, P_i, M_i, \Pi_i)$$
  trong đó:
  - $N_i$: Số lượng draft nodes (kích thước ngân sách cây).
  - $d_i$: Độ sâu của cây (tree depth).
  - $P_i$: Mảng liên kết cha-con (`parent_indices`).
  - $M_i$: Ma trận mặt nạ chú ý cây (`tree_attention_mask`).
  - $\Pi_i$: Thông tin vị trí (`position_ids`).

### 2.2. Cơ chế ép phẳng cây (tree flattening) và mất mát quan hệ cha-con
Khi `TreeAdaptiveController` đưa ra quyết định chọn topology $T_8$ ($N=8$) hoặc $T_{16}$ ($N=16$):
1. Hệ thống chỉ có thể truyền một giá trị số nguyên vào `dynamic_sd_lookup` thông qua $N_{batch} = \max_i(N_i)$.
2. vLLM tiếp nhận $N_{batch} = 8$ và hiểu đơn thuần rằng đây là **một chuỗi giải mã đầu cơ tuyến tính gồm 8 token liên tiếp ($K=8$)**.
3. **Mất quan hệ cha-con:** Trong một cây $T_8$, 8 node có thể được tổ chức thành nhiều nhánh song song (ví dụ: 1 gốc rẽ thành 2 nhánh, mỗi nhánh sâu 4). Nhưng khi bị vLLM ép phẳng thành chuỗi tuyến tính, token ở nhánh A bị đặt liền kề sau token ở nhánh B. Mô hình mục tiêu xác thực chuỗi này như một câu văn tuần tự thay vì kiểm tra các đường dẫn song song độc lập.
4. Hệ quả toán học:
   $$N(\mathcal{T}_1) = N(\mathcal{T}_2) \not\Longrightarrow \mathcal{T}_1 = \mathcal{T}_2$$
   Hai cấu trúc cây hoàn toàn khác biệt về độ sâu và ma trận chú ý nếu có cùng 8 node thì trên vLLM 0.30.0 đều bị biến thành cùng một chuỗi linear $K=8$.

```
[Kỳ vọng Tree T8]                      [Thực tế vLLM 0.30.0 ép phẳng]
         (Root)                                (Root)
        /      \                                 ↓
      (A1)    (B1)                              (A1)
      /  \    /  \                               ↓
    (A2) (A3)(B2)(B3)                           (A2)
     ↓                                           ↓
    (A4)                                        (A3)  <- Ép thành chuỗi tuyến tính!
                                                 ↓
                                                (B1)  <- Mất hoàn toàn ngữ nghĩa cha-con
                                                 ↓
                                                (B2)
                                                 ↓
                                                ...
```

### 2.3. Nguy cơ vi phạm tính đúng đắn khoa học (scientific validity)
Nếu hệ thống ghi nhận trong telemetry:
$$\text{selected\_topology} = T_8$$
nhưng dưới tầng runtime vLLM thực tế chỉ thực thi:
$$\text{linear } K = 8$$
thì phát sinh lỗi nghiêm trọng:
$$\text{Planned Treatment} \neq \text{Executed Treatment}$$

- Máy chủ vẫn phục vụ bình thường, không crash, vẫn có throughput và latency đo được.
- Tuy nhiên, dữ liệu thu được thực chất là của **Linear Speculative Decoding $K=8$**, không phản ánh hiệu năng hay thuật toán của TreeAdaptive.
- Nếu lấy dữ liệu này để khẳng định thuật toán TreeAdaptive tăng tốc hoặc cải thiện tỷ lệ chấp nhận, đồ án sẽ phạm lỗi ngụy tạo bằng chứng khoa học (scientific fallacy).

---

## 3. Vấn đề phân bổ phản hồi bất đồng bộ (async feedback lineage)

Bên cạnh lỗi tree flattening, kiến trúc bất đồng bộ `AsyncScheduler` của vLLM V1 còn chứa một lỗ hổng phân bổ phản hồi (feedback attribution):
- Trong `AsyncScheduler`, việc lập lịch bước kế tiếp diễn ra song song và gối đầu (pipelined) với việc xác thực bước trước.
- **Hiện tượng lệch nhịp:** Bước $t$ sinh draft với ngân sách $K=2$ (kế hoạch $P_{10}$). Trong khi GPU đang xác thực $P_{10}$, scheduler bước $t+1$ đã chạy và chọn kế hoạch mới $P_{11}$ với $K=4$.
- Khi kết quả xác thực của $P_{10}$ quay về: Nếu scheduler gán số token được chấp nhận cho kế hoạch hiện tại nhất ($P_{11}$), bộ điều khiển thích ứng sẽ nhận nhầm số liệu của $K=2$ gán cho $K=4$.
- Trường hợp yêu cầu bị ngắt (preemption) hoặc hoàn tất, dữ liệu phản hồi cũ (stale feedback) có thể làm ô nhiễm bộ nhớ EMA của các request khác tái sử dụng bộ nhớ.

Invariant bắt buộc phải bảo đảm:
$$\text{Feedback Origin} \equiv \text{Origin of the Actual Consumed Draft Slots}$$

---

## 4. Quyết định của PM và chiến lược rollback phân tách phạm vi

### 4.1. Phê duyệt của PM: Chọn phương án B (Split Scope)
Dựa trên báo cáo [Pm decision m4 02 04.pdf](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec&plan/treeAdaptiveController/Pm%20decision%20m4%2002%2004.pdf), PM Cấn Đức Khôi đã chính thức phê duyệt **Phương án B (Split Scope + Evidence-Preserving Integration)** với điểm đánh giá kỹ thuật 98/100:
1. Đóng mốc `M4-02-04` ở phạm vi **Scalar Adaptive Budget Scheduling** đã kiểm chứng đầy đủ trên vLLM v0.30.0.
2. Tách toàn bộ việc nghiên cứu và thực thi Arbitrary Tree Runtime ($T_4, T_8, T_{16}, T_{32}$) thành **hướng nghiên cứu tương lai (future work)**.
3. Không chuyển đổi runtime sang SGLang trong giai đoạn M4 để bảo toàn tính ổn định của hệ thống benchmark.

### 4.2. Cách thức Đại thực hiện rollback và cô lập mã nguồn
Để vừa giữ nguyên thành quả nghiên cứu cây của nhóm, vừa đảm bảo runtime chính thức sạch sẽ, Đại đã thực hiện chiến lược cô lập như sau:

1. **Lưu trữ toàn bộ lịch sử tree vào nhánh tương lai:**
   - Tạo nhánh [future/tree-adaptive-v02](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22) tại commit `53b813a`.
   - Nhánh này lưu giữ đầy đủ mã nguồn `TreeAdaptiveController` (commit `a2510f2` của Khôi), toàn bộ tài liệu đặc tả, kế hoạch thiết kế và các bài kiểm thử liên quan.
   - Nhánh `future/tree-adaptive-v02` độc lập với luồng phát triển chính và không bị ghi đè.

2. **Dọn dẹp canonical serving path trên nhánh `develop`:**
   - **[src/proposer/\_\_init\_\_.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/__init__.py):** Gỡ bỏ eager import của `TreeAdaptiveController` và `TreeAdaptiveDecision`. Thay vào đó, thiết lập cơ chế lazy import thông qua hàm `__getattr__` để phục vụ các bên ngoài muốn nghiên cứu độc lập mà không nạp vào serving runtime:
     ```python
     def __getattr__(name):
         if name in {"TreeAdaptiveController", "TreeAdaptiveDecision"}:
             from src.proposer import tree_adaptive
             return getattr(tree_adaptive, name)
         raise AttributeError(name)
     ```
   - **[src/proposer/adapter.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/proposer/adapter.py):** Xóa bỏ hoàn toàn việc import `tree_adaptive`. Mặc định `AdaptiveProposerAdapter` hoạt động với `AdaptiveDraftController` chuẩn tuyến tính.
   - **[src/server/budget_planner.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/budget_planner.py):** Xóa bỏ `from src.proposer.tree_adaptive import get_tree_topology`. Không suy diễn cấu trúc topology cây từ số lượng node. Trường `selected_topology` được đặt tường minh là `null` trong canonical serving, và cờ `tree_execution_verified` được đặt cố định là `False`.

---

## 5. Toàn bộ thay đổi và cải tiến trong phiên bản 0.2 (v0.2)

Phiên bản 0.2 (`feat/m4-02-04-async-lineage-v02`, merge vào `develop` qua PR #38 và PR #39) tập trung giải quyết triệt để tính toàn vẹn của hệ thống serving tuyến tính và kiểm định thực nghiệm:

### 5.1. Kiến trúc theo dõi nguồn gốc bất đồng bộ (async feedback lineage)
- **`src/server/lineage.py`:**
  - Định nghĩa cấu trúc bất biến `DraftOrigin` lưu vết: `request_id`, `incarnation`, `plan_sequence`, `schedule_sequence`, `request_k`, `batch_k`.
  - Triển khai `LineageLedger` để quản lý vòng đời của các execution, đảm bảo mỗi lượt sinh draft được gán đúng định danh duy nhất và giải phóng bộ nhớ khi request hoàn tất.
- **`src/server/lineage_executor.py`:**
  - Xây dựng `LineageUniProcExecutor` kế thừa `UniProcExecutor` của vLLM.
  - Tích hợp hàm `verify_sources()`: Đối soát mã băm SHA-256 của 15 file nguồn vLLM 0.30.0 theo danh mục `experiments/scalar_vllm_source_manifest.json` nhằm bảo đảm môi trường chạy khớp 100% bản chuẩn.
  - Hook vào hàm `_copy_draft_token_ids_to_cpu` thông qua `observe_birth`: Theo dõi chính xác thời điểm draft ra đời từ GPU buffer, phân biệt rõ ràng giữa zero nhân tạo (synthetic zeros) và draft thật.
- **`src/server/lineage_startup.py`:**
  - Khởi tạo cơ chế ghép cặp bắt buộc (`pairing`) giữa `AdaptiveAsyncScheduler` và `LineageUniProcExecutor`. Ràng buộc tính sẵn sàng của lineage theo `engine_generation`, ngăn ngừa việc scheduler chạy độc lập mà không có executor giám sát.

### 5.2. Nâng cấp bộ lập lịch và quản lý ngân sách
- **[src/server/adaptive_scheduler.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/adaptive_scheduler.py):**
  - Đăng ký mọi bước lập lịch vào ledger qua `_lineage.register`.
  - Trong `update_from_output`: Đối chiếu chính xác draft được tiêu thụ (`consumed draft slots`) với hàng kết quả của model runner.
  - Bỏ qua các thống kê bị stale sau preemption. Nếu phát hiện lineage bị mất hoặc không khớp (`stale-or-mismatched-lineage`), hệ thống kích hoạt `record_lineage_failure` và bỏ qua lượt cập nhật feedback thay vì làm sai lệch hệ số $\alpha_{EMA}$.
- **[src/server/budget_planner.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/server/budget_planner.py):**
  - Bổ sung phương thức `record_lineage_failure(req_id, reason)` và `record_verification(..., source=...)`.
  - Thêm các trường truy vết nguồn gốc vào bản ghi telemetry: `source_plan_sequence`, `source_schedule_sequence`, `source_request_scalar_k`, `source_batch_scalar_k`, `lineage_status`.

### 5.3. Chuẩn hóa Telemetry Interface A v0.2
- Phân biệt rõ ràng 3 tầng trạng thái trong từng bước suy luận:
  1. `planned_k`: Ngân sách do bộ điều khiển dự tính cho từng request ($K_i$).
  2. `scheduled_k`: Ngân sách thực tế scheduler cấp phát cho toàn batch ($K = \max_i K_i$).
  3. `measured_k` / `accepted_tokens`: Kết quả xác thực thực tế đo được từ mô hình mục tiêu.
- Bảo đảm cách ly dữ liệu giữa các request (`request isolation`), dọn dẹp triệt để trạng thái khi request kết thúc (`_free_request`), không để rò rỉ bộ nhớ giữa các phiên serving.

### 5.4. Hồ sơ kiểm định và bằng chứng thực nghiệm trên GPU
Đại đã thiết lập quy trình kiểm định khép kín và lưu trữ bằng chứng đầy đủ trong Git:
1. **Kiểm chứng trên Google Colab (GPU NVIDIA L4):**
   - Commit bằng chứng: `6906985eeb0b53bfce312847dbcf104d6254b1ed`.
   - Thư mục bằng chứng: `experiments/runs/m4_02_04/20261111T132628Z-scalar-observer-l4-b45a0afc/`.
   - Xác nhận chuyển đổi runtime $K \in \{0, 2\}$ thành công mà không gây lỗi serving.
2. **Kiểm chứng trên Modal (GPU NVIDIA A10):**
   - Commit bằng chứng: `97bf088c889bd84703d8835bbb2b32ef6b7ea6db`.
   - Thư mục bằng chứng: `experiments/runs/m4_02_04/20261112T014441Z-async-lineage-pinned-runtime/`.
   - Môi trường: Ubuntu 22.04, NVIDIA A10 (23028 MiB), CUDA 13.0, PyTorch 2.13.0+cu130, vLLM 0.30.0.
   - Thu thập đầy đủ telemetry VRAM (`vram.jsonl` với 1889 bản ghi), log vLLM, và thông số tiến trình server.
3. **Bộ kiểm toán ngoại tuyến (Offline Evidence Validator):**
   - `experiments/validate_async_lineage_runtime.py` với 13 cổng audit nghiêm ngặt kiểm tra tính toàn vẹn SHA-256, số lượng token, tính liên tục của thời gian và trạng thái server.
   - Tài liệu nghiệm thu chính thức: `CLOSURE.md` và `CLOSURE-VALIDATION.json`.

---

## 6. Ranh giới công bố khoa học (claim boundary)

Để bảo đảm tính trung thực học thuật tuyệt đối cho đồ án, nhóm thống nhất phân định rõ 4 mức tuyên bố:

| Cấp độ | Định nghĩa tuyên bố | Trạng thái trong đồ án hiện tại |
| :--- | :--- | :--- |
| **L1** | Bộ điều khiển có thuật toán tính toán và lựa chọn ngân sách/topology. | **ĐÃ XÁC MINH (VERIFIED)** |
| **L2** | Quyết định có tích hợp thông tin phần cứng và tín hiệu runtime online ($\alpha_{EMA}$, VRAM). | **ĐÃ XÁC MINH (VERIFIED)** |
| **L3** | Runtime thực tế thực thi đúng ngữ nghĩa và xác thực đúng ngân sách đã lập lịch. | **ĐÃ XÁC MINH VỚI SCALAR LINEAR $K$**<br>*(Chưa xác minh cho Arbitrary Tree)* |
| **L4** | Thuật toán tạo ra gia tốc tốc độ (measured speedup) thực tế trong benchmark có kiểm soát. | **ĐANG ĐO ĐẠC Ở CÁC MỐC TIẾP THEO** |

### Tuyên bố được phép đưa vào đồ án:
- "Trên phiên bản vLLM v0.30.0 cố định, đồ án đã hiện thực hóa thành công bộ lập lịch thích ứng bất đồng bộ (`AdaptiveAsyncScheduler`) cho giải mã đầu cơ tuyến tính (Scalar Speculative Decoding), hỗ trợ thay đổi ngân sách $K$ động theo từng request và từng bước giải mã."
- "Đồ án đã giải quyết bài toán lệch nhịp bất đồng bộ bằng cơ chế truy vết nguồn gốc draft (`Async Feedback Lineage`), đảm bảo dữ liệu phản hồi được gắn kết chính xác 100% với đúng lượt sinh của nó."

### Tuyên bố tuyệt đối KHÔNG được đưa vào đồ án ở giai đoạn này:
- *Không tuyên bố:* "vLLM đã thực thi giải mã đầu cơ thích ứng cấu trúc cây $T_4, T_8, T_{16}$." (Vì vLLM 0.30.0 đã ép phẳng cây thành linear $K$).
- *Không tuyên bố:* "Thuật toán TreeAdaptive trên vLLM đã được chứng minh mang lại tốc độ vượt trội so với linear." (Vì planned treatment khác executed treatment).

---

## 7. Kết luận và định hướng tiếp theo

1. **Về mặt kỹ thuật mốc M4-02-04:**
   - Việc tách scope theo Option B và hoàn thiện async lineage v0.2 là quyết định chính xác, giúp cứu đồ án khỏi nguy cơ sai lệch dữ liệu khoa học nghiêm trọng.
   - Toàn bộ serving path hiện tại trên `develop` là ổn định, an toàn, có bằng chứng thực nghiệm đầy đủ trên GPU L4/A10 và đã sẵn sàng cho pha đo đạc chi phí phần cứng.
2. **Về hướng phát triển Tree-Adaptive tương lai (`future/tree-adaptive-v02`):**
   - Khi có nhu cầu chứng minh giải thuật cây ở các mốc sau hoặc sau khi hoàn thành đồ án, nhóm có thể:
     - Xây dựng một reference runtime riêng biệt (ngoài vLLM, ví dụ dựa trên mã nguồn EAGLE-2 hoặc OPT-Tree) để kiểm chứng cấu trúc cây độc lập.
     - Hoặc can thiệp sâu vào kernel xác thực của vLLM nếu các phiên bản vLLM tương lai hỗ trợ cấu trúc cây chính thức.
