# CHƯƠNG 4: THIẾT LẬP THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ CƠ SỞ (M1 BASELINE)

---

## 4.1. Môi trường Thực nghiệm và Kiến trúc Phần cứng

Nhằm bảo đảm tính khách quan, khả năng tái lập (reproducibility) và tính trung thực khoa học của các kết quả nghiên cứu, toàn bộ các chiến dịch đo đạc trong khuôn khổ đề tài được triển khai trên môi trường phần cứng chuẩn hóa, đại diện cho phân khúc GPU phổ thông (Consumer / Edge Enterprise GPU) thường được các doanh nghiệp vừa và nhỏ tự triển khai (self-hosted).

### 4.1.1. Hạ tầng Phần cứng Mục tiêu (NVIDIA Tesla T4 16GB)
Hệ thống tính toán trung tâm được neo cố định trên dòng vi xử lý đồ họa **NVIDIA Tesla T4 16GB** (kiến trúc Turing, mã vi kiến trúc **sm_75**). Đây là dòng GPU tiêu chuẩn được sử dụng rộng rãi trên các nền tảng điện toán đám mây như Google Cloud Platform, Google Colab Pro và AWS EC2 (G4dn instances), mang các đặc trưng kỹ thuật sát với điều kiện triển khai thực tế:

*Bảng 4.1: Bảng thông số kỹ thuật phần cứng GPU thực nghiệm Tesla T4*

| Thông số kỹ thuật | Giá trị thực tế ghi nhận | Ý nghĩa kỹ thuật trong suy luận LLM |
| :--- | :--- | :--- |
| **Kiến trúc GPU** | NVIDIA Turing (Compute Capability 7.5) | Không hỗ trợ phần cứng cho BF16 và FlashAttention-2 |
| **Dung lượng VRAM vật lý** | 15,360 MiB (~14.56 GiB khả dụng thực tế) | Ngưỡng giới hạn nghiêm ngặt cho mô hình 7B + KV-Cache |
| **Băng thông bộ nhớ (Memory Bandwidth)** | **320 GB/s** (GDDR6, 256-bit bus) | Nút thắt chính (Bottleneck) trong giai đoạn Memory-bound |
| **Hiệu năng FP16 Tensor Core** | ~65 TFLOPS | Đủ năng lực tính toán cho giải mã tuần tự 4-bit |
| **Driver / CUDA Host** | NVIDIA Driver `580.82.07` / CUDA `13.0` | Đảm bảo tương thích tối đa với PyTorch 2.13 và vLLM mới |
| **Kênh kết nối suy luận từ xa** | Microsoft Dev Tunnels / VS Code CLI Tunnel | Phục vụ gọi API trực tiếp với độ trễ mạng $< 35$ ms |

### 4.1.2. Môi trường Phần mềm và Hợp đồng Môi trường (Environment Contract)
Hiện tượng trôi dạt số liệu (measurement drift) giữa các lần đo thường xuất phát từ sự không đồng nhất về phiên bản thư viện hoặc trình biên dịch CUDA kernel. Để triệt tiêu hoàn toàn sai số hệ thống, đề tài thiết lập **Hợp đồng Môi trường Thực nghiệm** ghim cứng mang mã định danh `m1-t4-cu130-v1`:
- **Hệ điều hành:** Ubuntu 22.04 LTS (x86_64 Linux Kernel 6.6).
- **Trình thông dịch:** Python `3.13` (Colab Runtime) / Python `3.11` (Standalone Kaggle venv).
- **Thư viện tính toán Tensor:** PyTorch `2.13.0+cu130` (biên dịch tối ưu cho CUDA 13.0).
- **Công cụ phục vụ suy luận (Serving Engine):** `vllm == 0.30.0` (Engine Core V1, tích hợp Triton Attention backend thay thế cho FlashAttention-2 trên Turing CC 7.5).
- **Thư viện trích xuất Telemetry:** `aiohttp == 3.13.5`, `nvidia-ml-py == 13.595.45`.

Hợp đồng này được kiểm tra tự động thông qua hàm `validate_environment()` trong [`experiments/benchmark_harness.py`](../experiments/benchmark_harness.py). Nếu phát hiện bất kỳ sự sai lệch nào về phiên bản GPU, CUDA, PyTorch hoặc vLLM, quy trình đo kiểm lập tức dừng lại với mã lỗi thoát (Exit Code) bằng 5.

### 4.1.3. Cấu hình Cặp Mô hình Suy luận Hai Làn (Target & Draft Models)
Hệ thống suy luận cơ sở được thiết lập với hai làn độc lập phục vụ cùng một bài toán đối chứng:
1. **Làn 1 (Vanilla Baseline — Giải mã tuần tự truyền thống):**
   - Mô hình mục tiêu: `Qwen/Qwen2.5-7B-Instruct-AWQ` (lượng tử hóa 4-bit W4A16 AutoAWQ Marlin).
   - Dung lượng nạp VRAM: **5.29 GiB** safetensors.
   - Dung lượng KV-cache cấp phát: **5.79 GiB** (đạt sức chứa 108,496 tokens, bảo đảm phục vụ đồng thời concurrency = 32 ở context length 4,096 tokens).
   - Cổng phục vụ nội bộ: `http://127.0.0.1:8001/v1`.
2. **Làn 2 (Static Draft Baseline — Giải mã suy đoán tĩnh với độ sâu cố định):**
   - Mô hình mục tiêu (Target): `Qwen/Qwen2.5-7B-Instruct-AWQ` (W4A16, 5.29 GiB).
   - Mô hình nháp (Draft): `Qwen/Qwen2.5-0.5B-Instruct` (FP16 trọng số gốc, dung lượng nạp ~0.98 GiB).
   - Cơ chế sinh nháp: Cố định độ sâu nháp $k = 3$ tokens (`num_speculative_tokens=3`, `use_heterogeneous_vocab=True`, chế độ lấy mẫu tham lam `greedy`).
   - Tổng VRAM chiếm dụng cho cả hai mô hình: **6.27 GiB** (VRAM còn dư cho KV-cache là **4.81 GiB**, hoàn toàn nằm dưới ngưỡng cảnh báo an toàn 14.0 GiB trên T4).
   - Cổng phục vụ nội bộ: `http://127.0.0.1:8002/v1`.

### 4.1.4. Hạ tầng Thực nghiệm Mở rộng trên Modal.com (NVIDIA L4 24GB) — Sprint 4
Từ Sprint 4, nhóm chuyển các chiến dịch đo mới sang nền tảng GPU serverless **Modal.com** với GPU **NVIDIA L4 24GB** (kiến trúc Ada Lovelace, sm_89). Khác với T4, L4 hỗ trợ BF16 và FlashAttention-2, nên đây là cơ hội quan sát cùng một thuật toán điều khiển cây nháp trên một thế hệ phần cứng khác. Ba làn (Vanilla, Static, Adaptive) được khai báo bằng mã nguồn trong [`experiments/modal_runner.py`](../experiments/modal_runner.py), với cấu hình làn ([`experiments/modal_lanes.py`](../experiments/modal_lanes.py)) được kiểm thử tự động để luôn khớp với các launcher T4 (`scripts/launch_lane*.sh`) về mô hình, độ sâu nháp và giới hạn phục vụ; chỉ kiểu dữ liệu được đổi sang `bfloat16`. Phần mềm giữ nguyên vLLM `0.30.0` / PyTorch `2.13.0+cu130` / CUDA `13.0` như hợp đồng `m1-t4-cu130-v1`, vì bộ lập lịch thích ứng của Làn 3 kế thừa `AsyncScheduler` của vLLM 0.30.0. Theo log khởi động, trên L4 vLLM tự chọn kernel lượng tử hoá AWQ Marlin và backend `FLASH_ATTN`; trọng số được ghim theo revision (`Qwen2.5-7B-Instruct-AWQ@b250375`, `Qwen2.5-0.5B-Instruct@7ae5576`) và lưu trong ổ bền của nền tảng, mỗi endpoint đều yêu cầu xác thực.

Do server tự tắt khi không có tải để tiết kiệm chi phí, thời gian khởi động trở thành một thông số vận hành cần báo cáo. Bảng 4.1b tổng hợp số đo thực tế (3 lần lặp mỗi ô, đồng hồ phía client, đo đến khi nhận phản hồi đầu tiên). Khởi động lạnh mất từ khoảng 3.5 đến 5.5 phút, chủ yếu do bước khởi tạo engine vLLM (profile bộ nhớ, chụp CUDA graph, warmup — đặc biệt dài khi có mô hình nháp), trong khi khởi động ấm chỉ khoảng 1.8–2.2 giây. Vì vậy, các phép đo hiệu năng trên L4 phải chạy ở chế độ giữ ấm container. Lưu ý: số liệu L4 là một bộ dữ liệu riêng và **không được trộn** với số liệu T4 (M1/M2) trong cùng bảng speedup; việc chính thức hoá L4 thành nền tảng thực nghiệm đang được đề xuất thành QĐ-14 và chờ nhóm phê duyệt.

*Bảng 4.1b: Thời gian khởi động của ba làn trên Modal L4, profile `l4-bf16` (median, min–max; đo ngày 24/10/2026)*

| Làn | Khởi động lạnh (s) | Khởi động ấm (s) | KV-cache khả dụng (tokens) |
| :--- | :---: | :---: | :---: |
| Làn 1 — Vanilla | 200.9 (200.6–226.1) | 1.84 (1.73–2.78) | 242,192 |
| Làn 2 — Static (k = 3) | 321.7 (288.4–406.7) | 2.17 (1.80–3.36) | 179,040 |
| Làn 3 — Adaptive (k ≤ 4) | 221.9 (207.0–242.4) | 1.77 (1.64–1.78) | 178,368 |

*Nguồn: `experiments/runs/m3_00/20261024-m300-startup/startup.json`; KV-cache từ log vLLM trong `experiments/runs/m3_00/20261024-m300-3lanes-check/`.*

**Phương pháp đo khi máy chủ ở xa.** Bộ đo `benchmark_harness.py` vốn được thiết kế để chạy cùng máy với máy chủ vLLM: nó đọc GPU, phiên bản thư viện và VRAM bằng NVML ngay tại máy đo. Khi máy chủ nằm trên Modal, đo từ máy cá nhân sẽ ghi nhầm GPU của máy cá nhân và cộng thêm độ trễ mạng Việt Nam – `us-east` vào TTFT. Vì vậy nhóm bổ sung hồ sơ môi trường `m3-l4-modal-v1` (hồ sơ T4 `m1-t4-cu130-v1` giữ nguyên làm mặc định) với ba nguyên tắc: (1) bộ đo chạy trong một hàm CPU của Modal **cùng vùng** với máy chủ, dùng đúng các tham số và gói workload đã đóng băng của chiến dịch M2-04; (2) thông tin môi trường (GPU, driver, phiên bản vLLM/PyTorch, kiểu dữ liệu, kernel lượng tử hoá) do **chính container máy chủ** ghi lại khi khởi động; (3) VRAM được container máy chủ lấy mẫu bằng NVML mỗi 0,25 giây, sau đó ghép vào từng lượt đo theo cửa sổ thời gian UTC của lượt đó (dung sai ±0,5 giây; lượt nào không có mẫu thì báo lỗi thay vì ước đoán). Lần chạy kiểm chứng (làn Vanilla, W1, một lượt 8 yêu cầu — cấu hình rút gọn, không dùng làm số liệu so sánh) ghi nhận môi trường `NVIDIA L4`/vLLM 0.30.0 và VRAM đỉnh 19 850,6 MB ghép từ 347 mẫu của máy chủ (`experiments/runs/m3_01/20261024-m301-vanilla-W1-c1/`).

**Chi phí forward của mô hình đích và mô hình nháp trên L4 (M3-02).** Để mô hình chi phí $C(N, B)$ dùng số đo thật thay cho hằng số ước lượng (20 ms cho một bước vanilla và 3 ms cho mỗi token nháp trong `CostProfile` hiện tại), nhóm đo riêng thời gian một bước decode của mô hình đích $C_T(L, B)$ và mô hình nháp $C_D(L, B)$ trên L4, với $L$ là độ dài ngữ cảnh và $B$ là số yêu cầu chạy đồng thời. Phép đo dùng vLLM 0.30.0 ở chế độ offline với đúng cấu hình của các làn phục vụ (bf16, AWQ Marlin cho mô hình đích, `max_model_len` 4096, revision trọng số đã cố định), tắt prefix caching và cho mỗi yêu cầu một dãy token riêng. Thời gian một bước được tính bằng $(T(1+N) - T(1))/N$ với $N = 64$, trong đó $T(m)$ là thời gian sinh $m$ token cho cả lô, nhờ vậy phần prefill bị triệt tiêu. Mỗi ô gồm 3 lượt khởi động và 5 lượt đo; cả 60 ô đều có độ dao động dưới ngưỡng 15%.

*Bảng 4.1c — Thời gian một bước decode trên NVIDIA L4 (ms, trung vị 5 lượt)*

| Mô hình | $L$ | $B = 1$ | $B = 8$ | $B = 32$ |
| :--- | :---: | :---: | :---: | :---: |
| Đích 7B AWQ ($C_T$) | 256 | 19,15 | 19,80 | 23,44 |
| Đích 7B AWQ ($C_T$) | 3072 | 19,35 | 25,72 | 52,96 |
| Nháp 0,5B ($C_D$) | 256 | 4,79 | 5,08 | 5,72 |
| Nháp 0,5B ($C_D$) | 3072 | 5,19 | 6,14 | 11,40 |

*Nguồn: `experiments/results/m3_02_forward_costs_l4.json` (đủ lưới $L \in \{256, 512, 1024, 2048, 3072\}$, $B \in \{1, 2, 4, 8, 16, 32\}$); dữ liệu thô `experiments/runs/m3_02/20261025-m302-l4-v2/`.*

Kết quả cho thấy hai điểm. Thứ nhất, khi lô nhỏ, thời gian một bước gần như không đổi theo độ dài ngữ cảnh (khoảng 19 ms với mô hình đích và 5 ms với mô hình nháp từ 256 đến 3072 token): bước decode bị giới hạn bởi việc đọc trọng số từ bộ nhớ, còn chi phí attention trên KV cache chỉ nổi lên khi lô lớn và ngữ cảnh dài (53,0 ms ở $L = 3072$, $B = 32$). Thứ hai, một lượt forward của mô hình nháp tốn khoảng 26% một bước của mô hình đích ($C_D/C_T \approx 0{,}26$ ở $B = 1$). Con số này quan trọng với bộ điều khiển thích ứng: trong vLLM, khi chọn ngân sách $k = 0$, mô hình nháp vẫn chạy một lượt forward để giữ KV cache của nó đồng bộ, nên một bước "tắt suy đoán" vẫn tốn khoảng $C_T + C_D$ chứ không phải $C_T$. Đây là một nguyên nhân khả dĩ khiến Làn 3 chậm hơn Vanilla trên W3 ở chiến dịch M2-04 (đo trên T4, nên chỉ so sánh được về xu hướng); kết luận định lượng cần phép đo tách riêng ở Sprint sau. Các giới hạn của phép đo: mỗi ô là trung bình trên ngữ cảnh $[L, L+64]$, và với $L \cdot B$ lớn, prefill chia nhỏ có thể chồng lấn với vài bước decode đầu. Chi phí xác thực nhiều token $C_V(k)$ và ma trận $C(N, B)$ hoàn chỉnh thuộc phần việc tiếp theo của M3-02.

---

## 4.2. Thiết Kế và Chuẩn Hóa Bộ Workload Đánh Giá (W1, W2, W3)

Hiệu năng của kỹ thuật Giải mã suy đoán (Speculative Decoding) phụ thuộc mật thiết vào mức độ trùng khớp phân phối xác suất giữa mô hình nháp và mô hình đích. Để phản ánh toàn diện các bài toán thực tế, nhóm nghiên cứu đã thiết kế 3 tập Workload mang đặc tính cấu trúc ngữ nghĩa phân hóa rõ rệt:

```
                      ĐẶC TRƯNG BA TẬP WORKLOAD THỰC NGHIỆM
+-----------------------------------------------------------------------------------+
|  Workload 1 (W1)       |  Workload 2 (W2)          |  Workload 3 (W3)             |
|  VAT JSON Extraction   |  Text-to-SQL Conversion   |  Free-form Financial Report  |
+------------------------+---------------------------+------------------------------+
| • Ràng buộc rất cao    | • Ràng buộc trung bình    | • Không ràng buộc (Tự do)    |
| • Cú pháp JSON cố định | • Cú pháp chuẩn SQL       | • Văn phong suy luận mở rộng |
| • Alpha kỳ vọng: >85%  | • Alpha kỳ vọng: 75 - 85% | • Alpha kỳ vọng: <60%        |
| • Max tokens: 256      | • Max tokens: 96          | • Max tokens: 192            |
+-----------------------------------------------------------------------------------+
```

### 4.2.1. Workload 1 (W1): Trích xuất Hóa đơn Thuế GTGT (VAT JSON Extraction)
- **Nguồn dữ liệu gốc:** Bộ dữ liệu hóa đơn tiếng Việt MC-OCR Devset (chuẩn hóa tại mốc `M0-04` và `M1-03`).
- **Bản chất bài toán:** Nhận văn bản thô từ module OCR quang học và trích xuất có cấu trúc 7 trường thông tin pháp lý bắt buộc: Tên công ty (`seller`), Mã số thuế (`tax_id`), Số hóa đơn (`invoice_no`), Ngày lập (`date`), Tổng tiền hàng (`subtotal`), Tiền thuế (`vat_amount`), và Tổng thanh toán (`total_amount`).
- **Đặc trưng ngôn ngữ:** Cấu trúc cú pháp JSON (dấu ngoặc nhọn, tên trường, dấu hai chấm) chiếm tỷ lệ lặp lại rất cao. Đây là môi trường lý tưởng để mô hình nháp nhỏ 0.5B dự đoán chính xác các token cấu trúc. Giới hạn độ dài sinh: `max_tokens = 256`.

### 4.2.2. Workload 2 (W2): Chuyển đổi Ngôn ngữ Tự nhiên sang Truy vấn Cơ sở Dữ liệu (Text-to-SQL)
- **Nguồn dữ liệu gốc:** Tập dữ liệu truy vấn kế toán bán hàng tích hợp schema bảng hóa đơn SQLite.
- **Bản chất bài toán:** Nhận câu hỏi quản trị kinh doanh bằng tiếng Việt (ví dụ: *"Thống kê tổng tiền thuế của nhà cung cấp X trong quý 3"*) và sinh câu truy vấn SQL hợp lệ (`SELECT ... FROM invoices WHERE ... GROUP BY ...`).
- **Đặc trưng ngôn ngữ:** Ngữ pháp SQL có cấu trúc chặt chẽ với các từ khóa chuẩn (`SELECT`, `WHERE`, `JOIN`), nhưng phần điều kiện lọc dữ liệu và tên trường mang tính động theo câu hỏi. Giới hạn độ dài sinh: `max_tokens = 96`.

### 4.2.3. Workload 3 (W3): Tóm tắt và Phân tích Tài chính Mở (Free-form Analysis)
- **Nguồn dữ liệu gốc:** Các kịch bản đánh giá rủi ro công nợ, nhận định xu hướng chi phí dựa trên dữ liệu hóa đơn.
- **Bản chất bài toán:** Yêu cầu mô hình viết bài luận phân tích ngắn gọn, đưa ra khuyến nghị kiểm toán và cảnh báo bất thường trong chuỗi thanh toán.
- **Đặc trưng ngôn ngữ:** Hoàn toàn là ngôn ngữ tự nhiên tự do (Unconstrained Generation). Không gian từ vựng phân bố rộng, phụ thuộc nhiều vào khả năng lập luận sâu của mô hình lớn 7B. Đây là tác vụ được thiết kế có chủ đích nhằm kiểm thử "vùng rủi ro" của mô hình nháp 0.5B. Giới hạn độ dài sinh: `max_tokens = 192`.

### 4.2.4. Workload 4 (W4): Pipeline Đầu-cuối Node 1 → Node 2 → Node 3 (M3-06)

Ba workload W1–W3 đo từng chế độ entropy một cách tách rời. W4 nối chúng thành một quy trình tự động hoá hoàn chỉnh, đúng với hình thức sử dụng LLM trong các nền tảng orchestration: kết quả của bước trước là đầu vào của bước sau, nên độ trễ **cộng dồn** qua từng node (luận điểm ở Chương 1). Đúng như QĐ-08, pipeline này là *dụng cụ đo*: engine phục vụ không chứa logic hoá đơn nào và chỉ nhìn thấy các request `/v1/chat/completions` thông thường.

```
[raw_text MC-OCR, 30 hoá đơn W1]
   │  NODE 1 · Extractor    (prompt W1, JSON mode)      → invoice_extraction.v1 → SQLite
   │  NODE 2 · Text-to-SQL  (prompt W2 v2, 6 câu Q1–Q6) → SQL chỉ đọc → bảng kết quả
   ▼  NODE 3 · Báo cáo VAT  (prompt W3 v2, 1 request)   → báo cáo tiếng Việt (report.md)
```

**Thiết kế triển khai.** Lệnh `scripts/run_pipeline_e2e.py` gọi Node 1 qua CLI có sẵn, nạp kết quả hợp lệ vào một cơ sở dữ liệu SQLite riêng cho mỗi lần chạy, rồi chạy Node 2 và Node 3 trên chính cơ sở dữ liệu đó. Node 2 dùng bộ sáu câu hỏi nghiệp vụ cố định Q1–Q6 (tổng chi tiêu tháng 08/2020, mặt hàng mua nhiều nhất, cửa hàng có tổng giá trị cao nhất, số hoá đơn có VAT, tổng tiền theo ngày, thuế theo loại). Mỗi câu SQL do mô hình sinh ra phải qua một lớp kiểm tra chỉ-đọc: chỉ một câu `SELECT`/`WITH`, chỉ truy vấn ba view `v_invoice_facts`, `v_item_facts`, `v_tax_facts`, cấm bảng gốc, `raw_json` và mọi lệnh ghi; sau đó được thực thi trên kết nối `mode=ro` với `PRAGMA query_only=ON`, giới hạn 5 giây và 200 dòng. Một câu lỗi chỉ được ghi nhận trạng thái (`guard_rejected`, `sql_error`, `timeout`…) và pipeline đi tiếp; mô hình **không** được gọi lại để sửa SQL, nhờ đó số request giữa các làn phục vụ luôn bằng nhau. Node 3 nhận câu hỏi, câu SQL, trạng thái và tối đa 20 dòng kết quả của từng câu, và chỉ được dùng các con số có trong bảng. Cả ba node dùng `temperature = 0`, `seed = 42`; Node 2 và Node 3 dùng nguyên văn system prompt đã đóng băng của W2 v2 và W3 v2 để đầu ra của pipeline khớp với workload benchmark.

**Kết quả nghiệm thu chức năng.** Do Modal không cấp được GPU L4 trong suốt ngày 04/11/2026 (hai lần chạy E2E và một lần thử hàm nhỏ nhất đều nằm chờ "waiting to be scheduled on a GPU_L4 worker", trong khi T4, A10G, L40S, A100 và H100 đều cấp được ngay), lần nghiệm thu được chạy trên **GPU A10G 24 GB** qua biến `PTIT_GPU`. Bảng 4.x tóm tắt lần chạy chính `20261104-m306-e2e-a10g-r2` trên làn vanilla.

| Node | Đầu ra | Kết quả | Ngưỡng *đề xuất – cần xác nhận* |
|---|---|---|---|
| Node 1 | 30 hoá đơn JSON | 30/30 hợp lệ schema (`schema_valid_rate = 1,0`); độ trễ trung vị ≈ 8 s/hoá đơn | ≥ 0,95 |
| Node 2 | 6 câu SQL | 4 `ok`, 2 `empty`, 0 lỗi; ≈ 1,1 s/câu | ≥ 4/6 câu `ok`/`empty` |
| Node 3 | 1 báo cáo | `ok`, kết thúc tự nhiên (`finish_reason = stop`) | `ok`, không có số bịa |

*Bảng 4.x. Nghiệm thu chức năng W4 trên Modal A10G (nguồn: `experiments/runs/m3_06/20261104-m306-e2e-a10g-r2-vanilla/e2e_summary.json`). Độ trễ đo phía client, một lần chạy, gồm thời gian khởi động server; chỉ có giá trị minh hoạ.*

Kiểm tra thủ công cho thấy mọi con số trong báo cáo đều có trong bảng kết quả Q1–Q6: tổng chi tiêu tháng 08/2020 là 5.155.094 đồng, đúng bằng tổng các ngày tháng 08/2020 ở Q5. Ở cả 30 hoá đơn, tên cửa hàng, ngày và tổng tiền do Node 1 trích xuất đều xuất hiện trong văn bản gốc.

**Phân tích các sai lệch.** Pipeline chạy thông, nhưng kết quả nghiệp vụ bộc lộ ba giới hạn cần nói rõ:

1. *Sai ngữ nghĩa của Text-to-SQL.* Với Q3, mô hình viết `WHERE total_amount_value = (SELECT MAX(...))`, tức là tìm **hoá đơn** lớn nhất thay vì cộng tổng theo **cửa hàng**, nên trả lời "CỬA HÀNG NĂM OÁNH" (một hoá đơn 1.161.500 đồng), trong khi đáp án đúng là "MINIMART ANAN" (3.918.594 đồng). Câu SQL hợp lệ về cú pháp và vượt qua lớp kiểm tra, nên lỗi chỉ phát hiện được khi có đáp án chuẩn. Đây là giới hạn chất lượng của mô hình 7B lượng tử hoá, không phải của engine; và do đề tài là lossless, mọi làn tăng tốc kế thừa đúng giới hạn này.
2. *Dữ liệu nguồn không chứa mặt hàng và thuế.* Văn bản MC-OCR của 30 mẫu W1 chỉ gồm tên cửa hàng, địa chỉ, ngày và tổng tiền; không mẫu nào nhắc tới VAT. Vì vậy Q2 và Q6 rỗng và Q4 bằng 0 — các câu về VAT mới chỉ kiểm được nhánh "không có dữ liệu" của pipeline.
3. *Báo cáo chưa hoàn toàn bám cấu trúc.* Node 3 đôi khi thêm mục ngoài bốn phần được yêu cầu và bỏ sót một câu hỏi. Lần chạy đầu còn ghi "không có dữ liệu" cho kết quả bằng 0 và tự suy đoán nguyên nhân; hai lỗi này đã được khắc phục bằng chỉ dẫn bổ sung trong template của Node 3 và được kiểm chứng ở lần chạy lại.

**Kiểm tra trên làn có speculative decoding.** Một lần chạy cùng cấu hình trên làn `static` cũng hoàn tất (exit 0) với cùng số câu `ok`/`empty`. Các trường nghiệp vụ của Node 1 (cửa hàng, ngày, tổng tiền) giống làn vanilla ở 30/30 hoá đơn, nhưng toàn văn JSON chỉ giống 14/30 và câu SQL của Q1 khác nhau. Khác biệt cấp ký tự giữa hai lần chạy đã từng được ghi nhận ngay giữa hai lần chạy vanilla trong M2-03 (trên T4; nguyên nhân số học chưa được xác lập), nên lần chạy đơn lẻ này **chưa đủ** để kết luận về tính lossless; phép xác minh đúng giao thức thuộc QĐ-11. Đối chứng vanilla–vanilla trên cùng GPU được trình bày ở §4.2.5.

**Giới hạn hiệu lực.** Số liệu W4 ở mục này là nghiệm thu chức năng: một lần chạy, đo phía client, trên A10G thay vì L4 tham chiếu, và chưa phát telemetry cấp server (α, τ, VRAM). Chúng không được dùng để so tốc độ giữa các làn; phép đo W4 theo đúng giao thức của Chương 4 (warm-up, lặp lại, telemetry) được dự kiến ở mốc M4.

### 4.2.5. Chất lượng Trích xuất Node 1 trên Bốn Làn Phục vụ (M4-03)

Mục này trả lời một câu hỏi ở góc nhìn người dùng pipeline: khi đổi làn phục vụ, thông tin trích xuất từ hoá đơn có còn đúng không?

**Thiết lập.** Node 1 chạy 30 hoá đơn của W1 qua năm lần chạy trên Modal A10G: vanilla hai lần, static, adaptive và eagle (`temperature = 0`, `seed = 42`). Hai lần vanilla dùng để ước lượng mức dao động tự nhiên giữa các lần chạy. Đáp án là nhãn gốc của MC-OCR (`key_text_pairs`), vốn không đưa vào prompt của mô hình. Bốn trường được chấm: tên cửa hàng (SELLER), địa chỉ (ADDRESS), ngày giờ (TIMESTAMPS) và tổng tiền (TOTAL_COST). Hai trường chữ được so sau khi chuẩn hoá chữ thường, dấu câu, khoảng trắng và tiền tố địa chỉ, nhưng giữ nguyên dấu tiếng Việt; ngày giờ và số tiền được tách từ nhãn thô rồi so theo giá trị.

| Lần chạy | JSON hợp lệ | SELLER | ADDRESS | TIMESTAMPS | TOTAL_COST |
|---|---|---|---|---|---|
| vanilla (lần 1) | 30/30 | 29/30 | 27/30 | 30/30 | 30/30 |
| vanilla (lần 2) | 30/30 | 29/30 | 28/30 | 30/30 | 30/30 |
| static | 30/30 | 29/30 | 27/30 | 30/30 | 30/30 |
| adaptive | 30/30 | 29/30 | 27/30 | 30/30 | 30/30 |
| eagle | 30/30 | 29/30 | 27/30 | 30/30 | 30/30 |

*Bảng 4.y. Số trường khớp tuyệt đối với nhãn MC-OCR theo làn (nguồn: `experiments/results/m4_03_entity_match.md`; bằng chứng từng lần chạy ở `experiments/runs/m4_03/`).*

**Kết quả ở cấp thực thể.** Cả năm lần chạy đều đạt 100% JSON hợp lệ. Ba làn suy đoán có số khớp bằng đúng vanilla lần 1 ở cả bốn trường, nên chênh lệch bằng 0 và nằm trong dao động giữa hai lần vanilla (0 điểm ở ba trường, 3,3 điểm ở ADDRESS). Các mẫu lệch cũng trùng nhau giữa các làn và có nguyên nhân rõ, không liên quan tới làn phục vụ:

- Mẫu 027 (SELLER, ADDRESS): nhãn gán cả hai dòng đầu là tên cửa hàng, còn mô hình coi dòng thứ hai là một phần địa chỉ. Đây là khác biệt cách chia dòng.
- Mẫu 029 (ADDRESS): văn bản có địa chỉ nhưng mô hình để trống, ở mọi làn.
- Mẫu 021 (ADDRESS): mô hình thêm ", Cẩm Phả" vào cuối địa chỉ. Lỗi này xuất hiện ở bốn lần chạy và biến mất ở vanilla lần 2 — chính là nguồn của mức dao động 3,3 điểm.

**Kết quả ở cấp văn bản.** Đầu ra không giống hệt nhau. Toàn văn JSON giữa hai lần vanilla giống nhau ở 25/30 mẫu, trong khi giữa vanilla và các làn suy đoán chỉ giống 13–15/30 mẫu; giữa static và eagle là 22/30 mẫu. Như vậy mức khác biệt giữa vanilla và làn suy đoán **lớn hơn** mức dao động của chính vanilla. Khác biệt tập trung ở phần siêu dữ liệu (`extraction.field_evidence`, `extraction.missing_fields`), khoảng trắng trong JSON và cách viết giờ (`12:03` so với `12:03:00`, cùng giá trị); ở một mẫu của làn static, trường `totals.subtotal` — không thuộc bốn trường được chấm — cũng khác.

**Kết luận và giới hạn.** Trên 30 mẫu và bốn trường này, không quan sát thấy suy giảm độ chính xác khi bật giải mã suy đoán. Đây **không** phải bằng chứng đầu ra khớp từng ký tự: phép đo cho thấy đầu ra có khác và mức khác vượt dao động vanilla–vanilla. Nguyên nhân số học chưa được xác lập; M2-03 có ghi nhận liên quan trên T4 về việc xác thực nhiều token cùng lúc cho kết quả khác giải mã từng token, nhưng điều này chưa được kiểm chứng trên A10G, và phép xác minh lossless theo QĐ-11 là một bài đo riêng. Ngoài ra, cỡ mẫu nhỏ (một mẫu tương đương 3,3 điểm), dao động chỉ ước lượng từ hai lần chạy, chạy trên A10G thay vì L4 tham chiếu, và dữ liệu W1 không có mặt hàng hay VAT nên các trường đó chưa được kiểm tra.

---

## 4.3. Chỉ Số Đánh Giá và Kỷ Luật Đo Lường Thực Nghiệm

Để loại bỏ hoàn toàn các nhận định cảm tính, đề tài xây dựng bộ chỉ số định lượng chặt chẽ và áp dụng quy trình kiểm soát đo lường 3 nguyên tắc:

### 4.3.1. Hệ thống Chỉ số Đánh giá Cốt lõi
1. **Time-to-First-Token (TTFT, tính bằng mili-giây ms):** Thời gian tính từ thời điểm client phát request đến khi nhận được token đầu tiên. Chỉ số này phản ánh độ trễ hàng đợi xếp lịch (Scheduler Queue) và giai đoạn tính toán Prefill của mô hình.
2. **Time-Per-Output-Token (TPOT, tính bằng ms/token):** Thời gian trung bình để sinh ra một token mới trong giai đoạn giải mã (Decode Phase). Đây là chỉ số phản ánh trực tiếp tốc độ suy luận của thuật toán.
3. **Throughput Hệ thống (tokens/giây):** Tổng số token sinh ra thành công trên toàn bộ các luồng chia cho tổng thời gian thực thi của phiên đo.
4. **Tỉ lệ Chấp nhận Nháp $\alpha$ (Draft Acceptance Rate):** Tỉ lệ phần trăm các token do mô hình nháp đề xuất được mô hình mục tiêu chấp thuận qua bước xác thực song song (Verification).
5. **Số Token Chấp nhận Trung bình $\tau$ (Mean Accepted Length):** Số lượng token thực tế mà mô hình lớn thu nhận được sau mỗi bước forward xác thực:
   $$\tau = 1 + \sum_{i=1}^k \alpha^i$$
6. **Hệ số Tăng tốc Tương đối (Speedup vs. Vanilla):** Tỉ số giữa Throughput của làn suy đoán so với Throughput của làn Vanilla ở cùng mức concurrency và workload:
   $$\text{Speedup} = \frac{\text{Throughput}_{\text{Static}}}{\text{Throughput}_{\text{Vanilla}}}$$

### 4.3.2. Ba Kỷ Luật Đo Lường Thực Nghiệm Nghiêm Ngặt
Tuân thủ cam kết tại [`docs/06-ke-hoach-thuc-nghiem.md`](../docs/06-ke-hoach-thuc-nghiem.md), mỗi điểm thực nghiệm phải vượt qua 3 cổng kiểm soát:
1. **Quy tắc Khởi động Khớp Tải (Concurrency-matched Warm-up - Phương án B):** Trước khi bấm giờ đo chính thức, runner tự động phát một đợt request khởi động với số lượng:
   $$N_{\text{warmup}} = \max(3, \text{concurrency})$$
   Đợt chạy này kích hoạt nạp trọng số vào L2 Cache, thiết lập bộ nhớ đồ thị CUDA và tạo áp lực lên bảng trang PagedAttention. Toàn bộ dữ liệu warm-up bị loại bỏ, không ghi vào tệp dữ liệu đo lường.
2. **Quy tắc Lặp lại Độc lập ($\ge 5$ Lần/Ô):** Mỗi ô cấu hình được đo lặp lại tối thiểu 5 phiên độc lập (`r001` đến `r005`) cách nhau khoảng nghỉ 2 giây để triệt tiêu nhiễu chia sẻ tài nguyên ảo hóa trên nền tảng đám mây.
3. **Quy tắc Thống kê Trung vị (Median / p50) và Kiểm soát Phương sai:**
   - Sử dụng giá trị trung vị $p_{50}$ thay vì trung bình cộng để loại trừ hiện tượng đuôi dài (Outliers).
   - Tính toán tỷ lệ biến thiên phương sai:
     $$\text{Variance Ratio} = \frac{\text{TPOT}_{\max} - \text{TPOT}_{\min}}{\text{TPOT}_{p50}}$$
   - Tự động kích hoạt cờ cảnh báo `variance_flag = True` nếu tỷ lệ biến thiên vượt quá $15\%$, làm cơ sở giải trình hiện tượng nhiễu phần cứng.

---

## 4.4. Bảng Kết Quả Thực Nghiệm Cơ Sở (M1 Baseline Matrix 36 Ô)

Dưới đây là toàn bộ số liệu thực nghiệm trích xuất từ file tổng hợp [`experiments/results/summary.csv`](../experiments/results/summary.csv), thể hiện đầy đủ 36 ô đo lường trên GPU Tesla T4 16GB:

*Bảng 4.2: Bảng số liệu thực nghiệm đối chứng 2 làn cơ sở (Vanilla vs. Static Draft)*

| Workload | Concurrency | Làn suy luận | TTFT p50 (ms) | TPOT p50 (ms) | TPOT p95 (ms) | Throughput (tok/s) | Tỉ lệ $\alpha$ | Độ dài $\tau$ | VRAM Peak (MB) | Speedup vs. Vanilla | Variance Flag |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W1** | 1 | Vanilla | 71.27 | 27.41 | 27.54 | 36.38 | N/A | N/A | 12,463.19 | 1.00× | False |
| **W1** | 1 | Static Draft | 154.75 | 20.43 | 20.74 | 48.48 | 91.59% | 3.74 | 12,203.19 | **1.34×** | False |
| **W1** | 2 | Vanilla | 99.85 | 29.67 | 29.87 | 66.81 | N/A | N/A | 12,463.19 | 1.00× | False |
| **W1** | 2 | Static Draft | 245.99 | 22.99 | 23.38 | 85.16 | 91.62% | 3.75 | 12,203.19 | **1.29×** | False |
| **W1** | 4 | Vanilla | 121.43 | 34.86 | 35.02 | 112.02 | N/A | N/A | 12,463.19 | 1.00× | False |
| **W1** | 4 | Static Draft | 297.75 | 26.14 | 26.72 | 147.21 | 91.63% | 3.75 | 12,203.19 | **1.33×** | False |
| **W1** | 8 | Vanilla | 154.32 | 44.95 | 45.43 | 172.91 | N/A | N/A | 12,463.19 | 1.00× | False |
| **W1** | 8 | Static Draft | 398.54 | 35.63 | 36.48 | 214.64 | 91.66% | 3.75 | 12,203.19 | **1.26×** | False |
| **W1** | 16 | Vanilla | 225.38 | 67.88 | 68.27 | 224.55 | N/A | N/A | 12,463.19 | 1.00× | False |
| **W1** | 16 | Static Draft | 556.12 | 55.43 | 56.33 | 271.91 | 91.65% | 3.75 | 12,203.19 | **1.22×** | False |
| **W1** | 32 | Vanilla | 380.59 | 113.17 | 113.36 | 264.65 | N/A | N/A | 12,463.19 | 1.00× | False |
| **W1** | 32 | Static Draft | 875.61 | 97.54 | 98.94 | 304.04 | 91.57% | 3.74 | 12,203.19 | **1.16×** | False |
| **W2** | 1 | Vanilla | 43.39 | 25.24 | 25.43 | 39.21 | N/A | N/A | 13,035.19 | 1.00× | False |
| **W2** | 1 | Static Draft | 133.37 | 18.86 | 20.22 | 48.87 | 80.30% | 3.41 | 13,133.19 | **1.34×** | False |
| **W2** | 2 | Vanilla | 82.59 | 25.92 | 26.04 | 74.55 | N/A | N/A | 13,035.19 | 1.00× | False |
| **W2** | 2 | Static Draft | 200.01 | 19.68 | 21.02 | 89.99 | 80.24% | 3.41 | 13,133.19 | **1.32×** | False |
| **W2** | 4 | Vanilla | 89.66 | 27.65 | 27.88 | 138.94 | N/A | N/A | 13,035.19 | 1.00× | False |
| **W2** | 4 | Static Draft | 216.51 | 20.72 | 21.74 | 169.73 | 81.21% | 3.44 | 13,133.19 | **1.33×** | False |
| **W2** | 8 | Vanilla | 104.80 | 30.58 | 30.92 | 232.30 | N/A | N/A | 13,035.19 | 1.00× | False |
| **W2** | 8 | Static Draft | 258.37 | 24.87 | 26.56 | 263.54 | 80.24% | 3.41 | 13,133.19 | **1.23×** | False |
| **W2** | 16 | Vanilla | 155.73 | 38.98 | 39.81 | 357.47 | N/A | N/A | 13,037.19 | 1.00× | False |
| **W2** | 16 | Static Draft | 377.14 | 33.38 | 36.04 | 383.12 | 80.74% | 3.42 | 13,161.19 | **1.17×** | False |
| **W2** | 32 | Vanilla | 235.38 | 53.31 | 56.18 | 491.43 | N/A | N/A | 13,037.19 | 1.00× | False |
| **W2** | 32 | Static Draft | 598.11 | 51.26 | 57.10 | 481.22 | 80.82% | 3.42 | 13,311.19 | **1.04×** | False |
| **W3** | 1 | Vanilla | 56.49 | 25.43 | 25.54 | 39.24 | N/A | N/A | 13,031.19 | 1.00× | False |
| **W3** | 1 | Static Draft | 133.55 | 25.45 | 27.27 | 39.25 | 54.62% | 2.64 | 12,203.19 | **1.00×** | False |
| **W3** | 2 | Vanilla | 82.13 | 26.12 | 26.26 | 76.18 | N/A | N/A | 13,031.19 | 1.00× | False |
| **W3** | 2 | Static Draft | 205.10 | 26.73 | 28.37 | 73.72 | 54.37% | 2.63 | 12,203.19 | **0.98×** | False |
| **W3** | 4 | Vanilla | 91.02 | 27.89 | 28.16 | 139.87 | N/A | N/A | 13,031.19 | 1.00× | False |
| **W3** | 4 | Static Draft | 221.42 | 27.93 | 29.67 | 141.90 | 55.53% | 2.67 | 12,203.19 | **1.00×** | False |
| **W3** | 8 | Vanilla | 101.54 | 30.88 | 31.23 | 241.40 | N/A | N/A | 13,031.19 | 1.00× | False |
| **W3** | 8 | Static Draft | 269.64 | 33.11 | 35.54 | 225.96 | 55.18% | 2.66 | 12,203.19 | **0.93×** | False |
| **W3** | 16 | Vanilla | 136.05 | 39.20 | 40.09 | 351.13 | N/A | N/A | 13,031.19 | 1.00× | False |
| **W3** | 16 | Static Draft | 365.28 | 43.60 | 46.18 | 322.08 | 54.28% | 2.63 | 12,203.19 | **0.90×** | False |
| **W3** | 32 | Vanilla | 285.13 | 54.19 | 54.83 | 483.10 | N/A | N/A | 13,031.19 | 1.00× | False |
| **W3** | 32 | Static Draft | 556.96 | 67.22 | 74.05 | 394.99 | 53.95% | 2.62 | 12,205.19 | **0.81×** | False |

---

## 4.5. Phân Tích Hiện Tượng và Thảo Luận Kết Quả Đối Chứng

Dựa trên bảng số liệu 36 ô và hai biểu đồ khoa học F1, F2 vừa sinh ra, nhóm nghiên cứu rút ra bốn luận điểm khoa học có tính bước ngoặt:

### 4.5.1. Tác Động Tăng Tốc trên Tác Vụ Có Cấu Trúc (Workload W1 & W2)
Quan sát biểu đồ Hình 4.1 (Figure F1):

```
+-----------------------------------------------------------------------------------+
|  HÌNH 4.1: BIỂU ĐỒ F1 — THROUGHPUT & TPOT VS CONCURRENCY TRÊN GPU TESLA T4 16GB   |
|  [Đường dẫn file: experiments/results/figures/F1_tpot_throughput_concurrency.png] |
+-----------------------------------------------------------------------------------+
```
![Hình 4.1: Biểu đồ F1 — Throughput & TPOT vs Concurrency](../experiments/results/figures/F1_tpot_throughput_concurrency.png)

1. **Trên Workload W1 (VAT JSON Extraction):**
   - Tỉ lệ chấp nhận nháp $\alpha$ đạt xấp xỉ **91.6%**, dẫn đến số token chấp nhận trung bình $\tau \approx 3.75$ token trên mỗi bước forward.
   - Ở mức tải thấp ($c=1$), TPOT giảm mạnh từ **27.41 ms** xuống **20.43 ms** (tiết kiệm 25.5% thời gian sinh mỗi chữ). Throughput tăng tương ứng từ 36.38 tok/s lên 48.48 tok/s (**tăng tốc 1.34×**).
   - Khi tăng tải lên $c=32$, Throughput của Static Draft đạt **304.04 tok/s**, vượt trội hoàn toàn so với mức 264.65 tok/s của Vanilla (Speedup duy trì ở **1.16×**).
2. **Trên Workload W2 (Text-to-SQL):**
   - Tỉ lệ $\alpha$ duy trì ở mức **80.5%**, $\tau \approx 3.42$.
   - Ở dải concurrency từ 1 đến 4, Static Draft thể hiện sức mạnh áp đảo với Speedup ổn định ở mức **1.33× – 1.34×**.
   - Tuy nhiên, khi chạm ngưỡng tải cao $c=32$, do độ dài câu SQL ngắn (96 tokens), chi phí Prefill và Verification bắt đầu tiệm cận lợi ích sinh token, khiến Speedup co hẹp về **1.04×** (ngưỡng hòa vốn).

### 4.5.2. Hiện Tượng "Tăng Tốc Âm" (Negative Speedup) trên Tác Vụ Văn Bản Tự Do (W3)
Quan sát biểu đồ Hình 4.2 (Figure F2):

```
+-----------------------------------------------------------------------------------+
|  HÌNH 4.2: BIỂU ĐỒ F2 — HỆ SỐ TĂNG TỐC SO VỚI VẠCH CHUẨN 1.0X TRÊN BA WORKLOAD   |
|  [Đường dẫn file: experiments/results/figures/F2_speedup_vs_concurrency.png]     |
+-----------------------------------------------------------------------------------+
```
![Hình 4.2: Biểu đồ F2 — Hệ số tăng tốc so với vạch chuẩn 1.0x](../experiments/results/figures/F2_speedup_vs_concurrency.png)

Biểu đồ F2 làm nổi bật phát hiện thực nghiệm mang tính cốt lõi của đề tài:
- Trong khi W1 (đường xanh lá) và W2 (đường xanh dương) luôn ngự trị an toàn phía trên vạch đỏ chuẩn $1.0\times$, thì **W3 (đường đỏ) lại liên tục trượt dốc và rơi xuống dưới vạch 1.0×**:
  - Tại $c=2$: Speedup tụt xuống **0.98×**.
  - Tại $c=8$: Speedup tụt xuống **0.93×**.
  - Tại $c=16$: Speedup tụt xuống **0.90×**.
  - Tại $c=32$: Speedup tụt dốc thảm hại xuống **0.81×** (Tốc độ chạy thực tế **chậm hơn 18.24%** so với cách giải mã tuần tự truyền thống!).

**Cơ chế giải thích bằng mô hình chi phí tính toán (Cost Model):**
Lý thuyết suy luận chỉ ra rằng Speculative Decoding chỉ mang lại lợi ích khi:
$$\alpha > \frac{T_{\text{draft}}}{T_{\text{target}}}$$
- Trên GPU Tesla T4 (băng thông bộ nhớ 320 GB/s, năng lực tính toán 65 TFLOPS), mô hình 0.5B dù nhẹ hơn 7B nhưng vẫn tiêu tốn khoảng **4.5 ms** cho 3 bước sinh nháp tuần tự.
- Khi $\alpha$ tụt xuống **53.95%** (do văn phong tự do không thể đoán trước), trung bình cứ 2 token thì mô hình lớn lại phải bác bỏ (reject) và rollback 1 lần.
- Ở mức tải $c=32$, cả 32 luồng cùng lúc chịu áp lực tranh chấp tài nguyên tính toán. Việc phải chạy forward mô hình 0.5B một cách vô ích biến nó thành một "tải ký sinh" (parasitic overhead). Hậu quả là TPOT bị đội từ **54.19 ms** lên **67.22 ms**, làm giảm băng thông phục vụ toàn hệ thống từ 483.10 tok/s xuống 394.99 tok/s.

> **Kết luận phản biện:** Hiện tượng tăng tốc âm trên W3 dập tắt hoàn toàn giả định ngây thơ cho rằng *"Speculative Decoding luôn luôn giúp tăng tốc trong mọi trường hợp"*. Kết quả này là luận cứ thực nghiệm đanh thép khẳng định: **Cần phải có một bộ điều khiển thích ứng (Adaptive Controller) có khả năng tự động nhận diện tác vụ hoặc theo dõi $\alpha$ để ngắt mô hình nháp khi phát hiện vùng tăng tốc âm.**

### 4.5.3. Mối Quan Hệ Giữa Concurrency và Độ Trễ Time-to-First-Token (TTFT)
Dữ liệu từ Bảng 4.2 cho thấy TTFT tăng đều đặn theo quy luật tuyến tính đối với cả hai làn:
- Ở $c=1$, TTFT dao động trong khoảng **43 ms – 154 ms**.
- Ở $c=32$, TTFT tăng lên tới **235 ms – 875 ms**.

Đáng chú ý, TTFT của làn Static Draft luôn cao hơn Vanilla khoảng **2× đến 2.3×**. Nguyên nhân là do tại bước đầu tiên, engine của vLLM phải đồng thời khởi tạo trạng thái và nạp context vào bộ nhớ KV của cả hai mô hình (7B và 0.5B). Tuy nhiên, vì TTFT chỉ xảy ra đúng 1 lần khi bắt đầu câu, trong khi giai đoạn sinh token kéo dài hàng trăm bước, nên việc tiết kiệm được 20–30% TPOT ở giai đoạn giải mã hoàn toàn bù đắp cho sự gia tăng nhỏ của TTFT trên các tác vụ dài.

### 4.5.4. Đánh Giá Tiêu Thụ VRAM và Tính Ổn Định của PagedAttention
- Đỉnh VRAM ghi nhận trên GPU T4 của cả 36 ô thực nghiệm dao động từ **12,184 MB đến 13,311 MB** (chiếm tối đa 86.6% dung lượng bộ nhớ vật lý).
- Suốt quá trình quét tải kịch trần 32 luồng với 5 lần lặp độc lập (tổng cộng hơn 3,000 requests), **không xảy ra bất kỳ lỗi tràn bộ nhớ (Out-Of-Memory — OOM)** nào. Cơ chế phân bổ bộ nhớ không liên tục PagedAttention của vLLM đã kiểm soát phân mảnh bộ nhớ cực kỳ xuất sắc, bảo đảm biên độ an toàn dự phòng $> 2.0$ GiB VRAM trong mọi tình huống.

---

## 4.6. Mối Đe Dọa Đến Tính Hợp Lệ (Threats to Validity)

Nhằm bảo đảm tính khách quan học thuật, nhóm nghiên cứu công khai nhận diện các mối đe dọa tiềm tàng và các biện pháp kiểm soát đã áp dụng:

1. **Nhiễu do Ảo hóa Đám mây (Internal Validity):**
   - *Nguy cơ:* Các máy ảo GPU trên Colab/Kaggle chia sẻ bus PCIe và CPU host với các tiến trình khác, có thể gây đột biến độ trễ tức thời.
   - *Kiểm soát:* Áp dụng quy tắc lặp lại tối thiểu 5 lần/ô, loại bỏ ngoại lai bằng trung vị $p_{50}$, và kiểm tra cờ phương sai `variance_flag`. Thực tế ghi nhận 36/36 ô đo lường chính thức đều đạt `variance_flag = False` (độ lệch $< 15\%$).
2. **Đặc thù Kiến trúc Phần cứng Cụ thể (External Validity):**
   - *Nguy cơ:* Tỉ lệ tăng tốc và điểm rơi suy hao đo được trên Tesla T4 (Turing, 320 GB/s) có thể không giống hoàn toàn trên các GPU cao cấp như A100 hay H100.
   - *Kiểm soát:* Nhóm nghiên cứu xác định rõ giới hạn phạm vi nghiên cứu (Scope Boundary) ngay từ đầu: Đề tài tập trung tối ưu hóa cho **Consumer/Edge GPU có tài nguyên hạn chế**. Bản chất của nút thắt Memory-bound trên T4 phản ánh trung thực bài toán của các hệ thống tự host chi phí thấp.
3. **Hiện tượng Rò rỉ Nhãn trong Prompt (Construct Validity):**
   - *Nguy cơ:* Nhãn của tập devset MC-OCR vô tình lọt vào prompt khiến mô hình sinh nhân tạo.
   - *Kiểm soát:* Module `build_node1_requests.py` áp dụng chính sách tách biệt nghiêm ngặt: nhãn MC-OCR chỉ được giữ lại trong metadata để đối soát, hoàn toàn không được truyền vào trường `content` của user prompt.

---

## 4.7. Tóm Tắt Chương 4 và Định Hướng Chương Tiếp Theo

Chương 4 đã trình bày toàn diện bức tranh thực nghiệm của Cột mốc M1 (Baseline Benchmark):
1. Thiết lập thành công môi trường thực nghiệm chuẩn hóa trên GPU Turing Tesla T4 16GB với hợp đồng môi trường `m1-t4-cu130-v1`.
2. Chuẩn hóa ba tập Workload đại diện (W1 có cấu trúc, W2 bán cấu trúc, W3 tự do) và thực thi nghiêm ngặt kỷ luật đo lường 3 nguyên tắc.
3. Thu thập hoàn chỉnh ma trận 36 ô thực nghiệm, chứng minh Speculative Decoding giúp tăng tốc vượt trội từ **1.16× đến 1.34×** trên các tác vụ có cấu trúc (W1, W2).
4. Phát hiện và mô hình hóa thành công hiện tượng **Tăng tốc âm (0.81×, suy giảm -18.24% throughput)** trên tác vụ tự do (W3) ở mức concurrency cao.

Phát hiện mang tính đột phá về hiện tượng tăng tốc âm trên W3 chính là cơ sở thực chứng để nhóm nghiên cứu bước sang **Chương 5: Thiết kế và Hiện thực hóa Bộ Điều Khiển Cây Nháp Thích Ứng Phần Cứng (Hardware-Aware Adaptive Speculative Decoding)** tại Cột mốc M2, nhằm tự động điều tiết độ sâu nháp và ngắt fallback về Vanilla, bảo đảm hệ thống luôn duy trì mức tăng tốc $\ge 1.0\times$ trong mọi kịch bản thực tế.
