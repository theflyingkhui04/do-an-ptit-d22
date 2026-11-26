# Báo cáo tiến độ: R6 chọn một K cho cả batch, thăm dò có điều kiện

- **Đặc tả**: [2026-11-24-r6-batch-k-selection-spec.md](./2026-11-24-r6-batch-k-selection-spec.md) · **Quyết định**: [decisions.md](./decisions.md) · **Log số đo đầy đủ**: [results-log.md](./results-log.md)
- **Nhánh Git**: `fix/r6-batch-k-selection` (tách từ `develop@e25530a`). **Chưa commit, chưa push, chưa merge.**
- **Ngày đo**: 25/11/2026 21:21 đến 26/11/2026 00:20, trên workspace Modal của Đại.
- **Trạng thái**: code và test CPU xong; đo kiểm chứng GPU xong (chưa phải số chính thức M3-05).

## 1. Kết quả chốt

Hệ số tốc độ so với vanilla (TPOT), cùng container NVIDIA A10 (driver 610.57.04), vLLM 0.30.0, Qwen2.5-7B-Instruct-AWQ + EAGLE-3.1, `--no-prefix-caching`, 32 request, 2 lần lặp, mọi ô 64/64 thành công:

| | Static K=3 | R6 hiện tại (develop) | **Code mới** |
|---|---:|---:|---:|
| Node 1 (W1), c=1 | 1,80× | 1,70× | **1,71×** |
| Node 1, c=8 | 1,26× | 1,11× | **1,16×** |
| Node 1, c=32 | 0,65× | 0,66× | **0,86×** |
| Node 2 (W2), c=8 | 1,26× | 1,17× | **1,25×** |
| Node 2, c=32 | 0,77× | 0,83× | **0,89×** |
| Node 3 (W3), c=8 | 1,46× | 1,24× | **1,35×** |
| Node 3, c=32 | 0,71× | 0,82× | **0,83×** |

Nguồn: W1 c=8 và c=32 từ `batchk-2511`; năm hàng còn lại từ `batchk-2511-table`. Mỗi hàng lấy từ một container, cả bốn nhánh chạy trong container đó.

So trực tiếp trong cùng container (ít nhạy với độ lệch nền, TPOT, > 1 là code mới nhanh hơn):

| Hàng | so với R6 hiện tại | so với Static K=3 |
|---|---:|---:|
| W1 c=1 | 1,00× | 0,95× |
| W1 c=8 | 1,05× | 0,92× |
| W1 c=32 | **1,31×** | **1,34×** |
| W2 c=8 | 1,07× | 0,99× |
| W2 c=32 | 1,08× | 1,15× |
| W3 c=8 | 1,09× | 0,92× |
| W3 c=32 | 1,02× | 1,17× |

**Kết luận đo**
1. Code mới không kém R6 hiện tại ở cả bảy ô; cải thiện lớn nhất ở W1 c=32 (+31%), còn lại +0 đến +9%.
2. Ở c=32, code mới nhanh hơn static 15–34%. Mức chậm hơn vanilla giảm: W1 35% → 14%, W2 23% → 11%, W3 29% → 17%.
3. Ở c=1 và c=8, code mới vẫn kém static 5–8% ở W1 và W3; W2 c=8 ngang static.
4. Ở c=32, cả ba node vẫn chậm hơn vanilla. Lúc này K = 0 ở mọi bước (B ≥ 25), phần chậm còn lại đến từ bước K = 0 của engine có nháp (W1: 33,2 ms so với 27,8 ms của vanilla, +19%).
5. Telemetry đúng thiết kế: 100% quyết định của code mới ở chế độ batch; ở B ≥ 25 không có bước K > 0 (0/4.549 lượt lập lịch: 1.816 + 1.402 ở `batchk-2511`, 1.331 ở `batchk-2511-table`) và không có thăm dò.

## 2. Đối chiếu tiêu chí của đặc tả (đề xuất, chưa chốt)

| Tiêu chí | Kết quả | Nhận xét |
|---|---|---|
| W1 c=1 ≥ 1,55× | 1,71× (`table`), 1,59× (`batchk-2511`) | Đạt ở cả hai container |
| W1 c=8 ≥ 1,15× | 1,16× | Đạt, sát mức |
| W1 c=32 ≥ 0,88× | 0,86× | Thiếu 0,02, nhỏ hơn độ lệch giữa các container (mục 3) |
| W2, W3 c=32 ≥ static + 0,05 | +0,12 (W2), +0,12 (W3) | Đạt |
| Không kém R6 hiện tại | 7/7 ô | Đạt |
| B ≥ 25: tỉ lệ bước K > 0 < 5% | 0% | Đạt |

## 3. Độ tin cậy và giới hạn

- **Độ lệch giữa hai container A10.** Cùng cấu hình, vanilla ở `batchk-2511-table` chậm hơn `batchk-2511` 6–9%, còn code mới chỉ lệch −1 đến +3%. Nguyên nhân chưa xác định. Vì vậy hệ số "so với vanilla" có thể lệch khoảng ±0,05–0,1 giữa các lần đo; nên ưu tiên đọc các cột so trực tiếp trong cùng container.
- Kiểm tra thứ tự chạy: nhánh đầu container không chậm hơn có hệ thống (static chậm hơn 4% khi chạy đầu ở `bench-r6fix-2411`, nhưng R6 chạy đầu ở `spike-r6-2411` thì không).
- **Đo bù `batchk-2511-f2`** (container A10 khác, 3 lần lặp, 96/96): code mới so với R6 hiện tại: W2 c=8 1,08×; W2 c=32 1,03×; W3 c=32 0,99×. So với bảng trên (1,07; 1,08; 1,02), tức lệch tới 0,05. Coi mức 2–8% ở Node 2 và Node 3 c=32 là "bằng hoặc hơn chút".
- Mỗi ô 2 lần lặp, 32 request (một "sóng" ở c=32), chỉ c = 1, 8, 32; chưa có lossless và chất lượng hoá đơn trên code mới. Bản chẩn đoán và kiểm chứng đều dùng W1–W3.
- Cần chạy lại toàn bộ test trước khi merge (lần chạy 24/11 có thể chưa phải bản cuối).

## 4. Kiểm thử CPU

- `tests/test_r6_batch_k_selection.py`: **27 pass** (20 test theo AC-01 đến AC-08, 7 tham số hoá thêm cho mốc `C_ref`).
- Test liên quan (planner, controller, R6, runner, adaptive scheduler, lineage): **479 pass, 1 skip** (POSIX process group không có trên Windows).
- Toàn bộ `tests/` (24/11): 2.532 pass, 17 fail, 37 skip. Cả 17 fail đều có trên develop; lần chạy này bắt đầu trước lần sửa cuối của `budget_planner.py`.
- `plan_step` ở 32 request: 2,83 → 2,97 ms (+0,14 ms; NFR-01 ≤ 1 ms).

## 5. Sự cố trong lần đo và cách xử lý

| Sự cố | Xử lý |
|---|---|
| Một request của R6 hiện tại ở W3 c=8 hỏng ở client (`ClientOSError: Can not write request body`), server không lỗi, 31/32 request còn lại trả đủ. Harness dừng cả ô nên mất các ô sau | Đo lại nguyên các hàng thiếu trong container mới, harness thử lại một lần |
| Modal cấp NVIDIA A10G thay vì A10 ở nhiều lần thử (chốt chặn dừng trong vài giây đến một phút) | `gpu_guard.py`, xin lại đến khi trúng A10 |
| Lời gọi đã huỷ vẫn được Modal cấp container và giữ khoá | Khoá chỉ nhả khi bị huỷ trong 60 giây đầu; không dùng huỷ nữa |
| Modal báo thiếu máy lớp A10 | Xếp hàng 8 lời gọi trên Modal, lời gọi đầu trúng A10 chạy, còn lại thoát |

Chi phí GPU ước tính khoảng 2,5 USD (A10 khoảng 1,10 USD/giờ, 121 phút chạy chính cộng các lần thử), chưa đối chiếu hoá đơn.

## 6. Việc còn lại

1. Báo Khôi D-03 (mốc `C_ref`) và cho Khôi xác nhận kết quả mục 1.
2. Chạy lại toàn bộ test, rồi commit, push, PR vào `develop` (chờ Dương cho phép từng bước).
3. Đo chính thức M3-05 sau khi merge, kèm lossless và chất lượng Node 1, mọi nhánh một container A10.
4. Sửa Chương 2 và Chương 4 của quyển theo bản cuối.

## 7. Nơi lưu dữ liệu (ngoài repo)

`D:/DoAn-D22/r6-remeasure-2311/`: `results/batchk-2511`, `results/batchk-2511-table`, `results/batchk-2511-f2` (kèm raw JSONL, telemetry, log server), `results/FINAL-BATCHK.md`, script `tools/measure_batchk.py`, `gpu_guard.py`, `final_table.py`, `analyze_batchk.py`, `batchk.patch` (code đo so với develop).
