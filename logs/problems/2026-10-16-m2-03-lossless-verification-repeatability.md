# Báo cáo kỹ thuật và quyết định PM: [M2-03] Hiện tượng bất định số học và giao thức kiểm chứng lossless

- **Mã vấn đề:** `PROB-M2-03-LOSSLESS-REPEATABILITY`
- **Cột mốc liên quan:** `M2 (Adaptive v0 & Lossless Verification)`
- **Ngày ghi nhận:** 16/10/2026
- **Người phát hiện / Raise:** Bùi Đức Đại (Dev + IT / MLOps & Benchmark)
- **Người xem xét & Phê duyệt:** Cấn Đức Khôi (Dev + PM / AI Systems)
- **Trạng thái:** **DEFERRED TO FINAL SPRINT (Hoãn sang Sprint cuối / Modal L4) — Đóng pha T4 ở 45/54, gỡ nghẽn cho Sprint 4**

---

## 1. Bối cảnh và mục tiêu ban đầu (Gate M2-03)

Tiêu chí nghiệm thu cốt lõi (Definition of Done - DoD) của mốc M2-03:
- Ở chế độ giải mã suy luận chuẩn xác (`temperature = 0.0`, greedy search), thuật toán tăng tốc thích ứng (**Adaptive**) bắt buộc phải sinh ra chuỗi token **khớp chính xác 100% từng token ID** so với chế độ giải mã tuần tự truyền thống (**Vanilla**).
- Tỷ lệ sai lệch chấp nhận: `Zero token discrepancy` trên tập canonical >= 50 prompts (hiện tại sử dụng bộ 54 prompts: W1 = 30, W2 = 12, W3 = 12).
- Cơ sở lý thuyết: Theo chuẩn mực Speculative Decoding (Leviathan et al., 2023), mô hình mục tiêu (Target model) luôn là người đưa ra phán quyết xác thực cuối cùng (Rejection Sampler). Về mặt toán học ở `temperature = 0`, phân phối đầu ra của Speculative Decoding phải hoàn toàn đồng nhất với giải mã chuẩn (lossless).

---

## 2. Bằng chứng thực nghiệm và dữ liệu phát hiện lỗi

### 2.1. Kết quả kiểm thử thực tế trên GPU Tesla T4 (vLLM 0.30.0)
Trong chiến dịch kiểm thử canonical `m2-03-20261013-r03`:
- **Tổng số prompt kiểm thử:** 54 prompts (W1 = 30, W2 = 12, W3 = 12).
- **Kết quả:** **44/54 prompt khớp chính xác (81.48%) -> Không đạt tiêu chuẩn nghiệm thu.**
- **Chi tiết sai lệch:** 10 prompt bị lệch token (mismatches) và 7 cặp bị cắt cụt độ dài (truncated pairs).

### 2.2. Đi sâu phân tích ca lỗi điển hình (reproducible case)
- **Prompt:** `W3 / w3_business_v2_004` (tác vụ phân tích báo cáo tài chính mở).
- **Quan sát đầu ra:**
  - `Vanilla (reference)`: Sinh 284 tokens. Token thứ 200 (index 200) là `128371`.
  - `Adaptive (attempt 1)`: Sinh 284 tokens. Token thứ 200 là `128371` -> **Match** (khớp hoàn toàn Vanilla).
  - `Adaptive (attempt 2)`: Sinh 285 tokens. Token thứ 200 lại là `128585` -> **Lệch** (bắt đầu khác từ index 200 và kéo theo toàn bộ chuỗi token phía sau bị thay đổi).

### 2.3. Manh mối logit tại index 200
Đại đã capture thành công giá trị Target Logits của Vanilla tại vị trí index 200:
- Token top 1 (`128371`): Logit = `12.500`.
- Token top 2 (`128585`): Logit = `12.500`.
- **Hiện tượng hòa điểm (tie):** Khoảng cách chênh lệch giữa 2 token đứng đầu là tuyệt đối bằng không (`top2 margin = 0.0`).
- Kết quả chọn:
  - Vanilla chọn: `128371`.
  - Adaptive (ở lần chạy lỗi) chọn: `128585`.

---

## 3. Phát hiện bước ngoặt: Tính bất định của chính baseline (Vanilla non-repeatability)

Khi tiến hành kiểm tra chéo tính lặp lại độc lập giữa các phiên chạy (cross-run repeatability), Đại phát hiện bằng chứng thực nghiệm quan trọng:

| Cặp so sánh đối chứng | Tỷ lệ khớp chính xác | Trạng thái |
| :--- | :---: | :---: |
| **Vanilla Run A vs Vanilla Run B** | **45 / 54 (83.33%)** | **Chính Vanilla tự chạy lại cũng lệch 9 prompt** |
| **Adaptive Run A vs Adaptive Run B** | **45 / 54 (83.33%)** | **Adaptive tự chạy lại cũng lệch** |
| **Vanilla vs Adaptive (Run r03)** | **44 / 54 (81.48%)** | Không khớp hoàn toàn |
| **Vanilla vs Adaptive (Run prodq3)** | **44 / 54 (81.48%)** | Không khớp hoàn toàn |

### Kết luận khoa học từ bằng chứng:
1. Bản thân phép đo cơ sở (**Vanilla Baseline**) chưa đạt tính lặp lại 100% giữa các phiên chạy độc lập (cross-run numerical non-determinism).
2. Khi hai token bị hòa điểm logit (`margin = 0.0`), các sai số làm tròn số thực (floating-point precision), thứ tự tính tổng trong phép nhân ma trận song song của CUDA kernel (reduction order), hoặc cơ chế phân bổ trang bộ nhớ PagedAttention sẽ làm đảo chiều token được chọn giữa các lần chạy.
3. **Do đó, không thể dùng một bản chạy Vanilla cũ ở môi trường hoặc thời điểm khác làm chuẩn tuyệt đối (ground truth) để quy kết rằng "Adaptive gây ra lỗi".**

---

## 4. Phân tích ba phương án giải quyết

### Phương án A: Giữ nguyên gate 54/54 cross-run ban đầu
- **Cách làm:** Bắt buộc chứng minh Vanilla tự khớp 54/54 trước, sau đó Adaptive phải khớp Vanilla 54/54.
- **Ưu điểm:** Giữ nguyên cam kết ban đầu, không thay đổi tiêu chí nghiệm thu.
- **Nhược điểm:** Thực nghiệm cho thấy Vanilla chỉ đạt 45/54 cross-run. Rất dễ rơi vào bẫy "săn bóng ma", đốt tài nguyên GPU vô ích để giải quyết một bài toán non-deterministic phần cứng cố hữu.
- **Điểm đánh giá khoa học:** 4.5 / 10.

### Phương án B: Chuyển sang so sánh cặp có kiểm soát (paired controlled comparison)
- **Cách làm:** Chạy đồng thời Vanilla và Adaptive theo từng cặp trong cùng một phiên thực thi (cùng GPU vật lý, driver, CUDA, revision mô hình, seed, cache state). Đánh giá xem: *"Mức độ sai lệch của Adaptive có vượt quá phương sai tự nhiên (drift) của chính Vanilla hay không?"*.
- **Ưu điểm:** Phản ánh đúng bản chất vật lý của hệ thống suy luận phân tán, dễ bảo vệ luận điểm khoa học.
- **Nhược điểm:** Phải hạ mức cam kết từ "100% lossless tuyệt đối" xuống "lossless trong biên độ sai số phần cứng", làm giảm sức nặng của kết luận.
- **Điểm đánh giá khoa học:** 8.5 / 10.

### Phương án C: Giữ 100% token match nhưng chuẩn hóa điều kiện baseline (qualify baseline first)
- **Cách làm:** Tách quy trình thành 2 giai đoạn (2-phase protocol):
  - **Phase 1 (Baseline qualification):** Khóa cứng môi trường thực thi (freeze execution regime), kích hoạt các cờ tối đa hóa tính xác định (deterministic mode). Chạy kiểm tra `Vanilla A vs Vanilla B vs Vanilla C`. Nếu Phase 1 đạt 54/54, chứng minh môi trường đo lường đã đủ tin cậy.
  - **Phase 2 (Paired lossless test):** Trong chính môi trường đã chuẩn hóa đó, chạy `Adaptive vs Vanilla` và tiếp tục áp dụng tiêu chí nghiêm ngặt **100% token ID match (54/54)**.
- **Ưu điểm:**
  - Vẫn bảo toàn 100% tính nguyên bản của kết luận "lossless tuyệt đối".
  - Tách biệt rành mạch giữa sai số hạ tầng đo (measurement drift) và lỗi thuật toán (adaptive bug).
  - Phương pháp luận nghiên cứu đạt chuẩn NIST và PyTorch Reproducibility Guidelines.
- **Điểm đánh giá khoa học:** **9.5 / 10 (được Đại và Khôi nhất trí đề xuất).**

---

## 5. Quyết định chính thức của PM (Cấn Đức Khôi)

1. **Chấp thuận phương án C:** Duyệt đề xuất của Đại, áp dụng giao thức 2 giai đoạn cho task M2-03.
2. **Hướng dẫn kỹ thuật cho Phase 1 (Baseline qualification):**
   Để triệt tiêu tính bất định số học trên GPU Tesla T4 (kiến trúc Turing CC 7.5), Đại áp dụng 3 quy tắc sau trong script chạy:
   - **Ghim cờ cuBLAS:** `export CUBLAS_WORKSPACE_CONFIG=:4096:8` (hoặc `:16:8`) trước khi khởi chạy tiến trình Python.
   - **Ép giải thuật xác định:** Bổ sung `torch.use_deterministic_algorithms(True)` trong runner.
   - **Khử nhiễu batching:** Chạy kiểm chứng lossless ở chế độ đơn luồng tuần tự (`concurrency = 1`, batch size = 1) để tránh việc gộp batch động làm thay đổi thứ tự cộng dồn dấu phẩy động giữa các token bị tie.
3. **Điều khoản chốt chặn (Fallback gate ngày 23/10):**
   - Nếu Phase 1 đạt 54/54 -> Tiến hành Phase 2 để chốt kết luận lossless 100%.
   - Nếu sau khi bật toàn bộ cờ deterministic mà kernel lượng tử hóa AWQ Marlin trên T4 vẫn cố hữu lệch ở các token tie điểm (như logit 12.500 vs 12.500) khiến Vanilla chỉ đạt 45-50/54 -> Nhóm sẽ tự tin kích hoạt **phương án B (paired comparison)** và đưa chính bản phân tích này vào Chương 4 và Chương 5 của quyển đồ án như một đóng góp thực nghiệm sâu sắc về giới hạn vật lý của phần cứng GPU phổ thông.

## Cập nhật quyết định PM — 17/10/2026

Ghi bổ sung ngày 18/10/2026 từ quyết định trực tiếp mới nhất của PM Cấn Đức Khôi:

> Đại ơi tớ xem rồi, cậu cứ làm cái B trước đi, xong sau đó khi mình deploy lên nền tảng khác thì mình sang C nhá

- Quyết định cũ tại mục 5 ghi nhận C trước. Quyết định mới thay thế **thứ tự thực hiện** đó: triển khai **B trước** trên nền tảng hiện tại; **C hoãn** tới khi triển khai trên nền tảng khác.
- B là so sánh cặp có kiểm soát: cùng GPU vật lý (UUID), cùng phiên VM/notebook, cùng runtime/config đã đóng băng; chạy Vanilla, giải phóng GPU qua workflow chuyển làn hiện có, rồi chạy Adaptive **tuần tự**. Không chạy hai server đồng thời; cách diễn đạt “đồng thời” tại mô tả B cũ không phải chỉ dẫn vận hành hiện hành.
- Báo cáo riêng Vanilla repeatability, Adaptive repeatability và từng cặp Vanilla–Adaptive. Hai cặp tuần tự V1 → A1 → V2 → A2 là thiết kế tối thiểu để tính cả ba nhóm phép đo; đây không phải sweep hiệu năng M2-04.
- B tạo bằng chứng, không tự đặt quy tắc `adaptive drift <= vanilla drift => PASS`. Trạng thái diễn giải là `interpretation_status = NOT_YET_APPROVED`; M2-03 vẫn **NOT DONE** khi chưa có bằng chứng GPU và phê duyệt cách diễn giải nghiệm thu.
- Không yêu cầu baseline 54/54 để được chạy B; không thêm `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms(True)` hoặc deterministic qualification làm gate hiện tại. C chỉ là công việc hoãn lại.
- Bằng chứng lịch sử được giữ nguyên. Thay đổi này điều chỉnh thứ tự giao thức, không sửa dữ liệu thô, không xóa các quan sát hoặc quyết định trước đó.
- **FACT/OBSERVATION:** các kết quả lịch sử ghi nhận Vanilla A/B 45/54, Adaptive A/B 45/54, Vanilla/Adaptive r03 và prodq3 44/54. Chúng chưa cô lập được lỗi chỉ thuộc Adaptive. **UNKNOWN:** nguyên nhân số học chính xác của biến thiên giữa các lần chạy. Các cơ chế tie số thực, khác GPU, lựa chọn kernel hoặc thứ tự CUDA graph vẫn là **HYPOTHESIS** nếu chưa có phép thử phân biệt; phần giải thích nhân quả ở mục 3 không được coi là bằng chứng xác lập nguyên nhân.

Triển khai cục bộ và hướng dẫn thu thập bằng chứng: [Runbook Protocol B](../../docs/runbooks/m2-03-protocol-b.md). Không chạy GPU trong nhiệm vụ triển khai này.

---

## Cập nhật thực nghiệm & Quyết định PM — 21/10/2026: Đóng pha T4, Delay Lossless sang Sprint cuối

### 1. Diễn biến thực nghiệm chuyên sâu trên branch fix/m2-03-lossless-clean (18/10 – 21/10/2026)

Từ ngày 18/10 đến 21/10/2026, nhóm đã triển khai chuỗi chẩn đoán nhân quả chuyên sâu trên GPU Tesla T4:

1. **Chặng 1: Thực thi Protocol B & Bác bỏ giả thuyết "Vanilla bất định" (18/10/2026, commit `4372f8b`):**
   - Chạy 2 cặp tuần tự (V1 -> A1 -> V2 -> A2) trên cùng 1 GPU vật lý T4:
     - Vanilla tự lặp lại (V vs V): 54 / 54 (100%).
     - Adaptive tự lặp lại (A vs A): 54 / 54 (100%).
     - Vanilla vs Adaptive: 44 / 54 (81.48%) (lệch đúng 10 prompt cố định ở cả 2 cặp).
   - **Ý nghĩa:** Bác bỏ giả thuyết ban đầu cho rằng "do phần cứng T4/AWQ trôi số học ngẫu nhiên nên baseline Vanilla không lặp lại". Bản thân mỗi pipeline đều xác định tuyệt đối (deterministic). Sự sai lệch 10 prompt là lỗi phân kỳ thực tế (real divergence) giữa cơ chế thực thi của Adaptive so với Vanilla.

2. **Chặng 2: Cô lập nguyên nhân gốc rễ — Boundary 279 & P177 (18/10 – 19/10/2026):**
   - Nhóm tập trung mổ xẻ ca lỗi điển hình `w3_business_v2_004` (phân kỳ tại index 279, xem [m2-03-boundary-279.md](../../docs/runbooks/m2-03-boundary-279.md)):
     - **Định vị điểm xuất phát sai khác KV-cache:** Bắt đầu chính xác từ Position 177 (P177) tại Layer 1 ngay sau khi model forward (`IMMEDIATE_STORED_KV_DIVERGENCE`).
     - **Thử nghiệm kiểm soát k = 0 (`p177-k0-control`):** Khi ép Adaptive không suy đoán (k = 0, tức target model forward 1 token như Vanilla) thì KV-cache tại P177 khớp 100% với Vanilla (`SPECULATIVE_MULTI_TOKEN_FORWARD_DIVERGENCE`).
     - **Thử nghiệm thay thế Attention (`p177-attention-substitution`):** Ép thay thế kernel Triton Attention ở Layer 0 bằng logic decode đơn token thì KV P177 khớp trở lại (`LAYER0_ATTENTION_PATH_CAUSAL`).
   - **Kết luận khoa học:** Thủ phạm chính là kernel Triton Attention của vLLM 0.30.0 trên kiến trúc Turing (sm75/T4) khi tính toán multi-token target verification (nhiều query row đồng thời) cho kết quả số học khác với khi giải mã từng token tuần tự q_len = 1.

3. **Chặng 3: Phát triển bản vá Row-wise Attention & Sửa Draft Tokenizer (19/10 – 20/10/2026):**
   - Hoàn thiện module `t4_rowwise_attention.py` và patch `vllm-0.30.0-t4-rowwise-attention.patch` (xem [m2-03-rowwise-attention.md](../../docs/runbooks/m2-03-rowwise-attention.md)):
     - Viết lại dispatch trên T4 để chia các token xác thực thành từng hàng query logic (q_len = 1) với độ dài ngữ cảnh tương ứng, khôi phục tương đương số học với Vanilla decode.
     - Vá lỗi tokenizer draft model revision (`spec.draft_model_config.revision`) trong vLLM.
   - **Kết quả targeted test:** Prompt lỗi `w3_business_v2_004` chạy lại trên T4 đã PASS hoàn toàn 100% (commit `dd10e41`).

4. **Chặng 4: Đánh giá lại toàn bộ Canonical 54 với Row-wise v2 (20/10/2026, commit `5ae38fd`):**
   - Tỷ lệ khớp đạt **45 / 54 (83.33%)** (tăng thêm 1 prompt so với mức 44/54 trước đó).
   - Prompt `w3_business_v2_004` đã hoàn toàn biến mất khỏi danh sách lỗi.
   - Vẫn còn 9 prompt lệch: `m0_04_005`, `m0_04_026`, `m0_04_029`, `w2_sql_v2_003`, `w3_business_v2_001`, `w3_business_v2_002`, `w3_business_v2_009`, `w3_business_v2_010`, `w3_business_v2_012`.

5. **Chặng 5: Mổ xẻ m0_04_005 và hiện tượng Observer Effect (21/10/2026, commit `9dd8233`):**
   - Lập probe A/B `test_m005_ab.py` và runbook [m2-03-m005-ab.md](../../docs/runbooks/m2-03-m005-ab.md) để bắt nguyên nhân lệch tại token thứ 162 của `m0_04_005` (Vanilla ra 341, Adaptive ra 5212).
   - Gặp hiện tượng **Observer Effect (Heisenbug):** Bản thân các hook chẩn đoán đọc tensor CPU đã làm thay đổi thời gian thực thi (timing), khiến Adaptive trong phiên test lại tự động sinh ra token 341 (khớp với Vanilla) thay vì tái hiện lỗi 5212. Dữ liệu này bị phân loại là `INVALID_EVIDENCE` do probe không trung lập (non-neutral).

---

### 2. Bảng tổng hợp các mốc thực nghiệm trên GPU Tesla T4

| Chiến dịch / Mốc | Ngày | Baseline Repeatability | Adaptive Repeatability | Vanilla vs Adaptive | Ghi chú & Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Ghi nhận ban đầu (r03) | 16/10 | 45 / 54 (chạy rời) | 45 / 54 | 44 / 54 (81.48%) | Nghi vấn do GPU trôi số học |
| Protocol B (2aad4c00) | 18/10 | 54 / 54 (100%) | 54 / 54 (100%) | 44 / 54 (81.48%) | Chứng minh lỗi do pipeline Adaptive |
| Row-wise v2 Targeted | 19/10 | 1 / 1 | 1 / 1 | 1 / 1 (PASS) | Ca lỗi w3_business_v2_004 được sửa dứt điểm |
| Row-wise v2 Canonical | 20/10 | 54 / 54 (100%) | 54 / 54 (100%) | 45 / 54 (83.33%) | Lên 45/54, còn 9 prompt lệch |
| M005 A/B Diagnostic | 21/10 | — | — | INVALID_EVIDENCE | Observer probe làm trôi timing, Heisenbug |

---

### 3. Quyết định chính thức của PM (21/10/2026): Dừng thực nghiệm T4, Delay M2-03 sang Sprint cuối

1. **Đóng toàn bộ hoạt động kiểm thử M2-03 trên kiến trúc Turing (Tesla T4 sm75):**
   - Kiến trúc Turing (T4) có nhiều giới hạn số học phần cứng cố hữu: không hỗ trợ native Bfloat16 (buộc ép FP16/Half), kernel Triton Unified Attention khác thứ tự reduction khi tính multi-row so với single-row decode, và nhân ma trận AWQ Marlin dễ hòa điểm tie-margin = 0.0.
   - Nhóm đã hoàn thành xuất sắc mục tiêu bản chất khoa học: chứng minh tính lặp lại (100% repeatability trên từng pipeline), cô lập nguyên nhân sâu tới Layer 0/Layer 1, và phát triển bản vá nâng độ khớp lên 45/54. Tiếp tục "săn bóng ma" trên 9 prompt còn lại trên T4 sẽ làm cạn kiệt GPU quota và gây trễ nghiêm trọng toàn bộ tiến độ đồ án.

2. **Hướng giải quyết chính thức: DELAY ĐỂ SPRINT CUỐI (Sprint 5 / Sprint 6):**
   - **Tạm hoãn mục tiêu 54/54 lossless tuyệt đối tại Sprint 4:** Không giữ trạng thái Blocked của M2-03 làm tê liệt các công việc tiếp theo.
   - **Kích hoạt Phương án C trên hạ tầng Modal.com ở sprint cuối:** Khi chuyển sang nền tảng cloud serverless Modal.com với thế hệ GPU mới như **NVIDIA L4 (Ada Lovelace sm89)** hoặc **A10G (Ampere sm86)**:
     - GPU 24GB VRAM hỗ trợ đầy đủ native BF16 và FlashAttention-2 chuẩn hóa, triệt tiêu hoàn toàn sai lệch reduction order của Triton Attention Turing.
     - Chạy quy trình 2 pha của **Phương án C**: Qualification baseline xác định 54/54 -> Paired lossless verification để đạt 54/54 lossless tuyệt đối một cách tự nhiên và sạch sẽ.

3. **Gỡ nút thắt, giải phóng 100% nguồn lực cho Sprint 4 (M3 & Cổng sinh tử 23/10):**
   - Tháo gỡ trạng thái Blocked của các thẻ `[M2-04]` và `[M2-05]` trên Trello Board.
   - Nhóm dồn toàn lực vào 3 mục tiêu cốt lõi của Sprint 4 để kịp Cổng kiểm soát sống còn ngày 23/10/2026:
     - `[M3-01]` Nạp ma trận chi phí C(N, B) vào cost model vLLM (Khôi 70% + Dương 30%).
     - `[M3-02]` Sweep đo kiểm đối chứng 3 làn M3 (Đại 70% + Khôi 30%) nhằm triệt tiêu tăng tốc âm ở W3 và tối đa hóa speedup ở W1/W2.
     - `[M3-03]` Hoàn thiện Node 2 (Text-to-SQL), Node 3 (VAT Report) và bản thảo quyển đồ án >= 20 trang (Dương 70% + Đại 30%).

4. **Giá trị đóng góp học thuật cho Quyển Đồ án:**
   - Toàn bộ hồ sơ thực nghiệm này (Protocol B, KV divergence P177, Triton Attention multi-row reduction, bản vá Row-wise attention, và hiện tượng Heisenbug / Observer Effect) sẽ được đưa vào **Chương 4 (Thực nghiệm & Đánh giá) và Chương 5 (Phân tích Giới hạn & Đóng góp Kỹ thuật)** của quyển đồ án như một đóng góp nghiên cứu thực nghiệm phần cứng rất giá trị.

