# Quyết định: chọn K theo batch cho R6 (24–25/11/2026)

Đặc tả: [2026-11-24-r6-batch-k-selection-spec.md](./2026-11-24-r6-batch-k-selection-spec.md) · Kết quả đo: [progress.md](./progress.md)

Trạng thái dùng trong bảng: **Chốt** (người có quyền đã quyết), **Thực hiện** (đã làm trong nhánh, chờ xác nhận), **Đề xuất** (chưa ai quyết).

| # | Quyết định | Trạng thái | Người quyết | Ngày |
|---|---|---|---|---|
| D-01 | Chọn một K cho cả batch, mặc định `ADAPTIVE_K_SELECTION=batch` | Chốt | Khôi (Dương báo lại) | 25/11 |
| D-02 | Thăm dò ở mức batch, mỗi 20 bước, chỉ khi có K với α* ≤ 0,80 | Chốt theo spec | Khôi (Dương báo lại) | 25/11 |
| D-03 | Mốc K = 0 tính theo `C_ref(B)` | **Thực hiện, chưa báo Khôi** | Dương | 25/11 |
| D-04 | Giữ đường cũ bằng `max_k`, giữ mọi API và artifact profile | Thực hiện | theo spec | 25/11 |
| D-05 | Chưa cộng chi phí engine vào profile (cách A) | Chốt, để sau | Khôi và Dương | 24/11 |
| D-06 | Mọi phép đo chạy trên NVIDIA A10, mọi nhánh trong một container | Chốt | Dương | 25/11 |
| D-07 | Không dùng số quy đổi A10G sang A10 | Thực hiện | Dương | 25/11 |
| D-08 | Chưa merge vào `develop` | Chốt | Dương | 25/11 |
| D-09 | Phần kết quả trong quyển chỉ nêu phương pháp cuối, rút gọn Adaptive v0 | Đề xuất | Dương; cần Khôi và Đại | 25/11 |
| D-10 | Đo chính thức sau khi merge | Đề xuất | Dương | 25/11 |
| D-11 | Chấp nhận code mới dựa trên kết quả đo kiểm chứng | **Đề xuất**, chờ Khôi và Dương | – | 26/11 |

## D-01. Chọn một K cho cả batch
- **Bối cảnh:** R6 chọn K cho từng request, nhưng `K_batch = max(k_i)` và mọi request trả chi phí của batch. Ở c = 8, 28% request chọn K = 1 nằm trong batch K = 3; ở c = 32, thăm dò của từng request kéo khoảng 40% số bước vào K > 0.
- **Quyết định:** `K = argmax_K Σ_i E[τ_i(K)] / C(K, B)`, chỉ chấp nhận K > 0 khi đạt `min_speedup_ratio` = 1,05 so với K = 0; mọi request dùng chung K.
- **Hệ quả:** server adaptive đổi cách chọn K theo mặc định. Quay lại cách cũ bằng `k_selection: max_k`.
- **Bằng chứng:** `src/server/budget_planner.py` (`_select_batch_k`, `plan_step`), `tests/test_r6_batch_k_selection.py`.

## D-02. Thăm dò có điều kiện ở mức batch
- **Quyết định:** tắt thăm dò theo request khi ở chế độ batch. Thăm dò một bước sau 20 bước liên tiếp có K = 0, với K khả thi có α* nhỏ nhất và α* ≤ 0,80; không khả thi thì không thăm dò.
- **Lý do:** ở B = 32 không có K nào hoà vốn (α* từ 0,89), nên mọi thăm dò chỉ tốn thời gian.
- **Còn mở:** 0,80 là giá trị đề xuất, chưa có phép đo độ nhạy; đổi được bằng `ADAPTIVE_BATCH_PROBE_ALPHA_CAP`.

## D-03. Mốc K = 0 theo C_ref
- **Bối cảnh:** profile A10 có 5 ô nhiễu `C(k, B) < C(0, B)` ở B = 1, 2, 4. Luật per-request dùng `C_ref(B) = min_K C(K, B)`; bản đầu của đường batch dùng `C(0, B)`.
- **Ảnh hưởng:** chỉ đổi quyết định khi α ≤ 0,04 và B ≤ 6 (K = 1 thành K = 0). α đo được khoảng 0,5–0,7.
- **Phát hiện:** khi dựng lệnh đo ngày 25/11, trước lần đo GPU; các lần đo `batchk-2511*` dùng bản đã sửa.
- **Cần làm:** báo Khôi trong PR.

## D-04. Tương thích ngược
Đường `max_k` giữ nguyên; artifact profile và `raw_sha256` không đổi; profile không có ma trận chi phí hoặc controller không có `decision_alpha` tự quay về đường cũ.

## D-05. Chưa cộng thêm chi phí engine vào profile
- **Quyết định:** ghi ở [logs/future-work/2026-11-24-r6-engine-step-cost-correction.md](../../future-work/2026-11-24-r6-engine-step-cost-correction.md), không làm trong nhánh này.
- **Lý do:** khi K còn chọn theo từng request, cộng chi phí vào B = 32 khiến K = 4 được chọn cho khoảng 11% quyết định và kéo cả batch. Sau D-01, có thể xét lại, vì bản batch-K vẫn kém static ở c = 8 (xem progress.md).

## D-06. Luôn đo trên NVIDIA A10
- **Bối cảnh:** Modal bán một lớp A10 (cũng nhận tên A10G). `nvidia-smi` trong container báo hai card: NVIDIA A10 (máy OCI, 150 W) hoặc NVIDIA A10G (máy AWS, 300 W). Xin `"A10"` hay `"A10G"` đều không ghim được; `cloud="oci"` không được hỗ trợ.
- **Bằng chứng:** đã gặp cả hai card ở các lần chạy tối 25/11; `r6-dai-2311` (23/11) có nhánh R6 trên A10G còn vanilla và static trên A10, nên bảng R6 của quyển so sánh khác card.
- **Quyết định:** kiểm tra tên card ở đầu container, sai thì dừng trước khi nạp model rồi xin lại (`gpu_guard.py`, ngoài repo); mọi nhánh của một phép so sánh chạy chung một container.
- **Hệ quả:** chậm xếp hàng hơn khi Modal thiếu máy A10. Số liệu ghi kèm tên card và driver. Cùng cấu hình R6, A10G chậm hơn A10 17–29% theo mức tải (chưa tách được card và driver).

## D-07. Không quy đổi A10G sang A10
- **Lý do:** tỉ lệ chậm thay đổi theo tải (17–29%), nên không có hệ số chung; bản ghi gốc có tên card. Số quy đổi ghi như số đo là sai sự thật.

## D-08. Chưa merge
Chờ Dương cho phép. Commit, push, PR và merge là các bước riêng.

## D-09. Cách trình bày trong quyển (đề xuất)
Phần kết quả R6 chỉ dùng bản cuối; rút gọn Adaptive v0 còn khoảng nửa trang và chuyển bảng sang phụ lục; giữ vanilla, static, EAGLE K = 3 vì là mốc so sánh; vẫn ghi các ô bản cuối thua (c = 8 so với static, c = 32 so với vanilla). Cần Khôi và Đại (Lead chương thực nghiệm theo `docs/08`) xác nhận.

## D-10. Đo chính thức sau khi merge (đề xuất)
Giao thức M3-05 (5 lần lặp, 6 mức tải, W1–W3), kèm lossless và chất lượng Node 1, trên đúng commit đã merge, mọi nhánh chung một container A10. Ước tính ban đầu 5 USD, chưa tính phần đo thêm.

## D-11. Chấp nhận code mới (đề xuất)
- **Căn cứ (progress.md mục 1 và 2):** không kém R6 hiện tại ở 7/7 ô; ở c = 32 nhanh hơn static 15–34% và giảm mức chậm hơn vanilla ở cả ba node; B ≥ 25 không còn bước nháp. Không tiêu chí nào thất bại rõ ràng; W1 c = 32 đạt 0,86× so với mức đề xuất 0,88×, nhỏ hơn độ lệch giữa các container.
- **Đánh đổi:** ở c = 1 và c = 8, code mới vẫn kém static 5–8% ở Node 1 và Node 3.
- **Nếu chấp nhận:** sửa tiêu chí W1 c = 32 trong đặc tả cho khớp số đo, hoặc ghi là chưa đạt và giải thích (bước K = 0 của engine đắt hơn vanilla khoảng 19%).

## Đề xuất đưa vào `docs/03` (chưa ghi vào file đó)
- **QĐ-20 (đề xuất, chờ Khôi duyệt)** — R6 chọn một K cho cả batch và chỉ thăm dò khi có thể hoà vốn. Nội dung: D-01, D-02, D-03.
- **QĐ-21 (đề xuất, chờ Khôi duyệt)** — Số đo GPU của đồ án chạy trên NVIDIA A10, kiểm tra bằng `nvidia-smi`, mọi nhánh một container. Nội dung: D-06. Liên quan M-07 (GPU tham chiếu) và M-06 (số chính thức).
