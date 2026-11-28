# Future work: Bảng chi phí R6 theo độ dài bước engine thật

Ngày ghi nhận: 2026-11-24.

Trạng thái: Đề xuất cho tương lai, **chưa triển khai**. Không áp dụng trước khi sửa cách chọn K ở mức batch và cách thăm dò (xem mục "Điều kiện áp dụng").

## Mục tiêu

Profile phần cứng của R6 đo thời gian GPU của hai phần: forward của model đích và
forward của model nháp. Một bước giải mã thật trong engine còn gồm lấy mẫu, kiểm
chứng, lập lịch và xử lý đầu ra. Vì vậy số trong profile nhỏ hơn bước thật, làm
R6 ước lượng sai lợi ích của việc nháp. Mục này ghi lại hai cách bổ sung và điều
kiện để dùng.

## Phạm vi hiện tại

- `C(K, B)` lấy từ `cost_matrix_ms` trong profile (`EmpiricalCostProfile`,
  `src/proposer/controller.py`); R6 chọn K theo E[τ] × C_ref / C_k.
- Profile hiện tại: `profile_sm_86_remote_a10g_fcda5486b12d.json`, đo trên
  NVIDIA A10 (`flexsds-component-v2`).
- Cách đo: engine vLLM một K đồng nhất cho cả batch, ngữ cảnh 1.024 token,
  `async_scheduling=False`; chỉ đo hai mốc target và draft
  (`src/flexsds/core/calibration_worker.py`, `calibration.py`). Khi phục vụ thật
  server chạy `--async-scheduling` và có thêm các phần nêu trên.

## Bằng chứng

Đo trên một container Modal NVIDIA A10 (driver 610.57.04), vLLM 0.30.0, W1
(Node 1, `--no-prefix-caching`), từ telemetry `bench-r6fix-2411/r6_new`
(`scheduled_budget`, độ dài bước = chênh `recorded_at` giữa hai bước liên tiếp
của cùng một request):

| B | K | Profile (ms) | Bước thật (ms) | Chênh |
|---:|---:|---:|---:|---:|
| 1 | 1 | 15,2 | 17,9 | +2,7 |
| 1 | 3 | 18,5 | 20,2 | +1,7 |
| 1 | 4 | 19,7 | 21,2 | +1,5 |
| 8 | 1 | 18,6 | 25,5 | +7,0 |
| 8 | 3 | 23,2 | 30,1 | +6,9 |
| 32 | 0 | 18,7 | 24,5 | +5,8 |

Hai dòng K > 0 ở B = 32 không dùng để so (các bước đó phần lớn chỉ có vài request
nháp, không phải K đồng nhất): K=2 đo được 58,8 ms (profile 55,8), K=4 đo được
58,7 ms (profile 71,3).

## Cách A: cộng thêm δ(B) vào chi phí của profile

`C_dùng(K, B) = C_profile(K, B) + δ(B)`

- `δ(B)` đo một lần hoặc tính từ telemetry, lưu riêng, **không sửa artifact profile**
  (giữ `raw_sha256`).
- Ưu: không tốn GPU, không chạm luồng đo profile.
- Nhược: coi phần thiếu là hằng số theo B; ở B = 32 mới chỉ có giá trị đo ở K = 0.

Mô phỏng trên telemetry đã ghi (không phải chạy thật): áp dụng phần cộng thêm cho
các quyết định của R6 hiện tại, K được chọn thay đổi như sau.

| | K khi dùng riêng profile | K khi cộng thêm |
|---|---|---|
| B = 1 | K1 19%, K3 39%, K4 42% | K1 15%, K3 38%, K4 47% |
| B = 8 | K1 27%, K3 73% | K1 13%, K3 85% |
| B = 32, δ = 3 ms | K0 99,7%, K4 0,3% | K0 97,5%, K4 2,5% |
| B = 32, δ = 5,8 ms | K0 99,7%, K4 0,3% | K0 89%, **K4 11%** |

Rủi ro ở B = 32: K của batch = max(k_i), nên chỉ một request chọn K = 4 cũng khiến
cả batch nháp. Nếu 11% request chọn K = 4, gần như mọi bước (khoảng 98%) sẽ có ít
nhất một request như vậy; một bước K = 4 ở B = 32 mất 58–71 ms so với 24,5 ms của
bước K = 0. Vì vậy **không áp dụng cách A khi chưa có chọn K ở mức batch**.

## Cách B: đo trọn bước engine khi calibration

- Sửa hook đo để bao trọn một bước engine (gồm lấy mẫu, kiểm chứng, lập lịch),
  `async_scheduling=True` như khi phục vụ.
- Sửa `calibration_worker.py`, `calibration.py` (schema, `cost_semantics`),
  `validate_measured_profile` trong `cost_model.py`, test; đo lại profile và thêm
  artifact mới, giữ artifact cũ.
- Ưu: là nguồn duy nhất và đúng hơn. Khi có, **không cần cách A** (cộng cả hai sẽ
  tính phần engine hai lần).
- Nhược: tốn GPU và phải sửa code đo.

## Điều kiện áp dụng (thứ tự)

1. Sửa trước: chọn K ở mức batch và thăm dò có điều kiện ở mức batch
   (`DE-XUAT-SUA-R6.md`, mục 3.1 và 3.2).
2. Đo lại trên cùng một container A10 (c = 1 / 8 / 32), xem phần chi phí lệch còn
   làm R6 chọn sai K bao nhiêu.
3. Nếu còn lệch đáng kể: chọn A (nhanh, có kiểm tra ở B = 32) hoặc B (bài bản, khi
   cần số chính xác để viết quyển). Không làm cả hai.

## Ngoài phạm vi

- Chi phí bước K = 0 trong engine có speculative (vẫn cao hơn vanilla sau khi bỏ
  thăm dò ở c = 32): giới hạn của vLLM, ghi trong quyển.
- Đổi sang vanilla ở tải cao bằng điều hướng request ở lớp ngoài.

## Tiêu chí hoàn thành

- Chọn K ở mức batch và thăm dò có điều kiện đã có, test CPU đạt.
- Cách A hoặc B được chọn có số đo đối chứng trên A10, không làm B = 32 chậm hơn
  so với chưa cộng thêm.
- Nếu là cách A: δ(B) có nguồn gốc, số mẫu và ngày đo; nếu là cách B: profile mới
  qua được `validate_measured_profile`.

## Liên quan

- Đề xuất sửa R6: `DE-XUAT-SUA-R6.md` (máy Dương, ngoài repo).
- Phép thử nguyên nhân: `SPIKE-R6.md`, `bench-r6fix-2411`, `spike-r6-2411`
  (cùng thư mục).
