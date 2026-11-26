# Đặc tả mục tiêu sản phẩm FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs

> **Dự án:** Đồ án tốt nghiệp đại học — Ngành Công nghệ Thông tin, PTIT  
> **Tên sản phẩm:** FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs
> **Phiên bản tài liệu:** 1.0 (Sprint 5 / M4)  
> **Người thực hiện chính:** Cấn Đức Khôi, Bùi Đức Đại  
> **Trạng thái:** Bản thảo kỹ thuật chính thức  

---

## Trạng thái đối chiếu ngày 14/11/2026

Đề tài tập trung vào EAGLE kết hợp điều khiển thích ứng theo phần cứng. Các yêu cầu bên dưới là đặc tả mục tiêu, không phải danh sách tính năng đã nghiệm thu. Phạm vi hiện tại:

- CLI đã nối **Hardware-Adaptive EAGLE scalar K** theo đường M4, với profile v2, async scheduler và paired lineage executor. CPU/source contract và functional GPU smoke ở batch 1 đã kiểm tra; full-grid và so sánh lossless/speedup còn chờ nghiệm thu. Static `k=4` là chế độ đối chứng có thể chọn.
- Calibration mới đã đo component trên Modal A10 ở batch 1 và K=0..4, giữ 275 raw records; full grid tới batch 32 còn chờ kiểm chứng.
- Các lane baseline `draft_model` giữ nguyên ý nghĩa đối chứng. Đường adaptive EAGLE dùng cơ chế scalar chung đã được nhánh M4 mở rộng, có kiểm tra source EAGLE riêng.
- Tree-adaptive được hoãn theo [quyết định phạm vi M4](../logs/problems/2026-11-12-m4-02-04-tree-adaptive-vllm-flattening-and-lineage-v02.md). Không diễn giải scalar K thành cấu trúc cây đã thực thi.
- Lossless nói về tính đúng đắn của đầu ra; không đồng nghĩa bảo đảm speedup ≥1 trong mọi workload.

Hành vi CLI hiện hành được mô tả tại [runbook](runbooks/flexsds-cli-modal-calibration.md). Các mô tả cây, web demo và bảo đảm hiệu năng trong phần đặc tả gốc dưới đây cần được nghiệm thu riêng trước khi công bố là khả năng sản phẩm.

## 1. Bối cảnh và vấn đề giải quyết

### 1.1. Bối cảnh thực tiễn
Suy luận phỏng đoán (speculative decoding) là kỹ thuật hàng đầu giúp giảm độ trễ sinh từ (TPOT) của các mô hình ngôn ngữ lớn (LLM) mà không làm suy giảm chất lượng đầu ra (guaranteed lossless). Tuy nhiên, việc đưa kỹ thuật này vào môi trường phục vụ thực tế (serving production) trên các engine hiện đại như vLLM gặp phải những rào cản rất lớn:
* **Cấu hình phức tạp và rời rạc:** Người dùng phải tự tìm mô hình nháp (draft model), tự kiểm tra độ lệch từ vựng (heterogeneous vocabulary), tự tính toán cờ bộ nhớ và cấu hình serving bằng các câu lệnh dòng lệnh dài dòng, dễ sai sót.
* **Nguy cơ tràn bộ nhớ (OOM):** Khi cùng lúc nạp cả mô hình đích, mô hình nháp và bảng phân bổ bộ nhớ đệm KV (KV-cache), nguy cơ tràn VRAM rất cao nếu không tính toán kỹ trước khi khởi chạy.
* **Hiện tượng suy giảm tốc độ ở tải cao:** Dữ liệu thực nghiệm của nhóm ở chiến dịch M3-05 đã chỉ ra rằng: nếu áp dụng suy luận phỏng đoán cố định ($k=3$), khi mức độ phục vụ đồng thời tăng cao ($c \ge 16$), hệ thống sẽ bị suy giảm tốc độ nghiêm trọng (chỉ còn $0.67\times - 0.93\times$ so với suy luận tuần tự) do nghẽn băng thông bộ nhớ và chi phí xác thực bùng nổ.

### 1.2. Định vị sản phẩm FlexSDS
`FlexSDS` ra đời như một **bộ công cụ phục vụ suy luận phỏng đoán thích ứng thông minh (smart adaptive speculative serving toolkit)**, đóng gói toàn bộ các đột phá nghiên cứu của đề tài thành một sản phẩm mã nguồn mở hoàn chỉnh:
* **Thân thiện tối đa (User-friendly CLI):** Cung cấp giao diện terminal tương tác từng bước (Interactive Wizard), nhận diện thương hiệu Học viện Công nghệ Bưu chính Viễn thông (PTIT).
* **An toàn tuyệt đối (Pre-flight Guard):** Tự động quét phần cứng, tính toán sức chứa VRAM và kiểm tra tương thích giữa mô hình đích và mô hình nháp trước khi khởi chạy.
* **Thích ứng cấu trúc cây nhận biết phần cứng (Hardware-Aware TreeAdaptive):** Tự động nhận diện GPU, nạp ma trận chi phí thực nghiệm và giải ngưỡng hòa vốn động $\alpha^*(T, B)$ theo thời gian thực để co giãn cây nháp hoặc ngắt về suy luận cơ sở, bảo đảm không bao giờ bị giảm tốc độ ở tải cao.
* **Đa nền tảng phục vụ:** Hỗ trợ khởi chạy cục bộ (Local GPU) hoặc đóng gói triển khai tức thì lên nền tảng đám mây không máy chủ (Modal Cloud GPU) với chuẩn giao tiếp OpenAI API.

---

## 2. Kiến trúc tổng thể hệ thống

Hệ thống FlexSDS được tổ chức thành 4 phân tầng độc lập:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        KIẾN TRÚC TỔNG THỂ FLEXSDS                      │
├────────────────────────────────────────────────────────────────────────┤
│ 1. TẦNG GIAO DIỆN TƯƠNG TÁC (PRESENTATION & CLI LAYER)                 │
│    • Banner & Branding: Logo PTIT ANSI TrueColor + bảng trạng thái GPU │
│    • Interactive Wizard: Menu hỏi-đáp từng bước cấu hình serving       │
│    • Pre-flight Guard: Bộ kiểm tra an toàn VRAM, Vocab, Tokenizer      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. TẦNG ĐIỀU KHIỂN THÍCH ỨNG (TREEADAPTIVE CORE ENGINE)                │
│    • Hardware Profiler: Nhận diện GPU (sm_xx) & nạp ma trận chi phí    │
│    • Dynamic Thresholds Solver: Giải phương trình hòa vốn alpha*(T, B) │
│    • Tree Controller: Ra quyết định bộ ba tham số Cây nháp (b, d, γ)   │
├────────────────────────────────────────────────────────────────────────┤
│ 3. TẦNG KẾT NỐI VLLM & SCHEDULER (ENGINE INTEGRATION GLUE)            │
│    • AdaptiveAsyncScheduler: Ghi đè bộ điều phối của vLLM 0.30.0       │
│    • Dynamic SD Lookup Table: Điều chỉnh K_batch theo từng bước decode │
│    • Per-request Trimming: Cắt tỉa placeholder nháp theo từng luồng    │
├────────────────────────────────────────────────────────────────────────┤
│ 4. TẦNG NỀN TẢNG THỰC THI (SERVING RUNTIMES & INFRASTRUCTURE)          │
│    • Modal Cloud Serverless: Triển khai container GPU L4 tự co giãn    │
│    • Local Dedicated Runner: Khởi chạy server vLLM trên máy cục bộ     │
│    • OpenAI Compatible Gateway: Endpoint /v1/chat/completions          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Đặc tả chi tiết các phân hệ chức năng

### 3.1. Phân hệ nhận diện phần cứng và thương hiệu (Branding & Hardware Detection)
Khi người dùng gõ lệnh `flexsds` trên dòng lệnh:
1. **Hiển thị nhận diện thương hiệu:**
   * Sử dụng thuật toán dựng điểm ảnh dạng nửa khối Unicode (`▀` và `▄`) kết hợp mã màu ANSI TrueColor 24-bit.
   * Render chuẩn xác logo PTIT với dải oval đỏ (`#ED1C24`), chữ PTIT và cuốn sách mở, ngôi sao vàng (`#FFF200`) tại vị trí chữ I.
   * Cột bên phải hiển thị tên thương hiệu `FlexSDS`, khẩu hiệu, phiên bản và đơn vị bảo trợ đề tài (PTIT).
2. **Quét phần cứng tự động trong 0.2 giây:**
   * Tên vi xử lý đồ họa (ví dụ: `NVIDIA L4`, `RTX 4090`, `Tesla T4`).
   * Mã kiến trúc tính toán Compute Capability (ví dụ: `sm_89`, `sm_75`, `sm_80`).
   * Trạng thái bộ nhớ: Dung lượng VRAM đang trống / Tổng dung lượng VRAM thực tế.
   * Môi trường thực thi: Phiên bản CUDA, PyTorch, vLLM.

### 3.2. Phân hệ trợ lý thiết lập tương tác (Interactive Wizard & Pre-flight Guard)
Thay vì buộc người dùng phải ghi nhớ các cờ lệnh dài, FlexSDS hướng dẫn người dùng qua menu hỏi–đáp:
* **Bước 1 — Chọn tác vụ:** Khởi chạy phục vụ (`serve`), quét phần cứng (`detect`), đo hồ sơ GPU (`profile`), hoặc kiểm tra endpoint (`test`).
* **Bước 2 — Nhập mô hình đích (Target Model):** Nhập tên Hugging Face ID hoặc đường dẫn cục bộ (ví dụ: `Qwen/Qwen2.5-7B-Instruct-AWQ`).
* **Bước 3 — Nhập mô hình nháp (Draft Model):** Nhập mô hình nháp độc lập (ví dụ `Qwen/Qwen2.5-0.5B-Instruct`) hoặc đầu nháp chuyên biệt (ví dụ `duckhoi/qwen2.5-7b-eagle3.1-speculator`).
* **Bước 4 — Kiểm tra an toàn tương thích (Compatibility Guard):**
  * *Kiểm tra VRAM:* Tính toán dung lượng tối thiểu $M_{\text{req}} = M_{\text{target}} + M_{\text{draft}} + M_{\text{kv\_min}}$. Nếu $M_{\text{req}} > M_{\text{free}}$, hiển thị cảnh báo đỏ và đề xuất giải pháp (như dùng bản lượng tử AWQ 4-bit).
  * *Kiểm tra từ vựng:* So sánh kích thước và tokenizer của hai mô hình. Nếu phát hiện lệch bảng từ vựng, tự động kích hoạt cờ `use_heterogeneous_vocab = True`.
  * *Phân loại kiến trúc nháp:* Tự động xác định phương pháp suy đoán là `draft_model` (mô hình độc lập) hay `eagle3` (đầu nháp chuyên biệt).
* **Bước 5 — Lựa chọn lớp thích ứng:** Hỏi người dùng bật lớp TreeAdaptive (mặc định: `Có`).
* **Bước 6 — Lựa chọn nền tảng:** Triển khai lên Modal Cloud hoặc chạy vLLM cục bộ.

### 3.3. Phân hệ hồ sơ phần cứng và ngưỡng động (Hardware Profiling & Dynamic Thresholds)
Cơ chế thích ứng phần cứng tuân theo chiến lược 3 tầng:
1. **Tầng 1 (Hồ sơ đo sẵn - Pre-profiled Catalog):**
   * Tra cứu tệp JSON trong thư mục `profiles/` theo mã kiến trúc GPU (ví dụ `profile_sm89_l4.json` hoặc `profile_sm75_t4.json`).
   * Tệp chứa ma trận thời gian thực nghiệm: $C_T(B)$ (thời gian forward mô hình đích), $C_D(B)$ (thời gian forward mô hình nháp), và $C_V(k, B)$ (thời gian kiểm tra xác thực song song).
2. **Tầng 2 (Hiệu chuẩn nhanh tại chỗ - Quick Micro-calibration):**
   * Nếu chạy trên GPU mới chưa có hồ sơ (ví dụ RTX 4090), hệ thống kích hoạt chế độ đo nhanh 15 giây trong pha khởi động.
   * Chạy thử 5 bước giải mã mẫu ở các mức tải $B \in \{1, 8, 32\}$ để xác định tỷ số chi phí $r = \frac{C_D(B)}{C_T(B)}$ và lưu tệp hồ sơ mới vào máy.
3. **Tầng 3 (Giải ngưỡng động $\alpha^*$):**
   * Với mỗi bước lập lịch có batch size $B$, bộ điều khiển giải phương trình cân bằng chi phí để tìm ngưỡng chấp thuận tối thiểu:
     $$\alpha^*(T, B) = \arg_{\alpha} \left( \mathbb{E}[\tau(T, \alpha)] \cdot C_T(B) = C(T, B) \right)$$
   * Nếu tỷ lệ chấp nhận thực tế $\alpha < \alpha^*(T, B)$, hệ thống tự động co nhỏ cây hoặc ngắt về Vanilla ($N=0$), bảo đảm không bị âm tốc độ.

### 3.4. Phân hệ xuất điểm cuối phục vụ (Serving Endpoint Deployment)
* **Triển khai Modal Cloud (Mặc định):**
  * Tự động sinh file cấu hình và nạp trọng số từ cache đám mây.
  * Xuất địa chỉ API công khai: `https://<workspace>--flexsds-serve.modal.run/v1`
* **Triển khai cục bộ (Local vLLM):**
  * Khởi chạy server trên cổng `http://localhost:8000/v1` kèm cơ chế giám sát tiến trình nền.
* **Chuẩn hóa đầu ra:**
  * Tương thích 100% với đặc tả OpenAI API (`/v1/chat/completions`, `/v1/models`).
  * Lưu trữ tệp cấu hình phục vụ `serving_config.json` để hỗ trợ chạy tự động (CI/CD hoặc headless mode).

---

## 4. Đặc tả giao diện dòng lệnh (CLI Reference)

Cú pháp tổng quát:
```bash
flexsds [HÀNH_ĐỘNG] [TÙY_CHỌN]
```

### Các lệnh chính:
* `flexsds`: Mở màn hình chào PTIT và khởi động Interactive Wizard.
* `flexsds serve`: Khởi chạy quy trình serving (hỗ trợ cờ `--target`, `--draft`, `--backend`, `--adaptive`).
* `flexsds detect`: Chỉ quét và in báo cáo chi tiết về phần cứng GPU, VRAM và CUDA.
* `flexsds profile`: Chạy tiến trình micro-calibration để đo đạc ma trận chi phí của GPU hiện tại.
* `flexsds test --endpoint <URL>`: Gửi request thử nghiệm để kiểm tra độ trễ TTFT, TPOT và trạng thái hoạt động của endpoint.

---

## 5. FlexSDS Assistant

Nhằm mở rộng tính ứng dụng của FlexSDS vào các công cụ lập trình AI thế hệ mới (Antigravity CLI, Claude Code), nhóm quy hoạch skill **FlexSDS Assistant** (`flexsds-assistant`):
* **Kênh phân phối:** Chia sẻ bundle qua repository GitHub. Codex và Antigravity dùng thư mục skill trong repo. Claude có thể dùng bản ZIP hoặc thư mục project của Claude Code.
* **Năng lực của Skill:**
  1. *Tư vấn tương thích trước triển khai:* Đánh giá cấu hình phần cứng của người dùng và đề xuất cặp mô hình Target-Draft tối ưu nhất.
  2. *Tư vấn theo tác vụ:* Hướng dẫn người dùng khi nào NÊN bật suy luận phỏng đoán (các tác vụ có độ trễ quan trọng như agent loop, OCR, coding, trích xuất hóa đơn) và khi nào KHÔNG NÊN bật (tác vụ batch throughput quá lớn hoặc input dài mà output ngắn).
  3. *Tự động tích hợp mã nguồn:* Quét mã nguồn Python của dự án người dùng, tự động refactor các lời gọi `openai.OpenAI()` để trỏ về endpoint của FlexSDS.
  4. *Chẩn đoán log hiệu năng:* Đọc dữ liệu telemetry của request để giải thích nguyên nhân tăng tốc hoặc sụt giảm tốc độ.

---

## 6. Kết luận và giá trị đóng góp

`FlexSDS` là kết tinh từ các kết quả nghiên cứu lý thuyết, thực nghiệm đo đạc 72 cell của chiến dịch M3-05 và kỹ thuật phần mềm hệ thống hiện đại. Sản phẩm không chỉ bảo đảm tính thuyết phục cao nhất trước hội đồng bảo vệ đồ án tốt nghiệp tại PTIT mà còn có khả năng đóng góp thiết thực cho cộng đồng mã nguồn mở trong việc tối ưu hóa chi phí và tốc độ suy luận mô hình ngôn ngữ lớn.
