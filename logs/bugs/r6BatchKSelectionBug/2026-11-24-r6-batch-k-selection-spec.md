# Đặc tả kỹ thuật: Chọn K ở mức batch và thăm dò có điều kiện cho R6 Hardware-Adaptive

- **Mã tài liệu**: SPEC-R6-BATCHK-20261124
- **Ngày tạo**: 2026-11-24
- **Người soạn**: Dương (kèm bằng chứng đo); **chủ sở hữu mã nguồn và người duyệt**: Khôi (Core-Engine)
- **Trạng thái**: Khôi đã duyệt (Dương báo lại 25/11/2026). Đã triển khai trên nhánh `fix/r6-batch-k-selection`, đã đo GPU ngày 25/11/2026 (xem [progress.md](./progress.md)), **chưa merge**. Quyết định: [decisions.md](./decisions.md).
- **Cập nhật 25/11/2026**: FR-01.2 và FR-01.3 tính mốc K = 0 theo `C_ref(B) = min_K C(K, B)` thay vì `C(0, B)` (chi tiết ở FR-01).
- **Mốc mã nguồn**: `develop@e25530a` (R6 có Geometric MLE alpha và telemetry theo lô, PR #53)
- **Phụ thuộc tài liệu**: `DE-XUAT-SUA-R6.md` (vấn đề và số đo), `SPIKE-R6.md` (phép thử kiểm chứng nguyên nhân)
- **Phạm vi tác động**:
  - `src/server/budget_planner.py` (`AdaptiveBudgetPlanner.plan_step`, `__init__`, `from_env`)
  - `src/server/adaptive_scheduler.py` (`_update_after_schedule`)
  - `src/proposer/controller.py` (tái dùng `estimate_expected_tokens`, `solve_break_even_alpha`; bỏ thăm dò theo request khi planner bật chế độ batch)
  - `src/flexsds/server/local_runner.py`, `modal_runner.py` (chuyển tiếp 2 khóa cấu hình)
  - `tests/test_r6_batch_k_selection.py` (mới)
- **Không đụng tới**: Geometric MLE alpha, telemetry theo lô, artifact profile, `min_speedup_ratio` (giữ 1,05), profile calibration.

---

## 1. Bối cảnh và bằng chứng

Đo trong một container Modal NVIDIA A10 (driver 610.57.04), vLLM 0.30.0, Qwen2.5-7B-Instruct-AWQ + EAGLE-3.1, W1, `--no-prefix-caching`, 32 request, 2 lần lặp, 64/64 thành công. Chi tiết trong `SPIKE-R6.md`.

| Cấu hình | c=8 | c=32 |
|---|---:|---:|
| Static EAGLE K=3 (so với vanilla) | 1,28× | 0,65× |
| R6 hiện tại (develop) | 1,14× | 0,68× |
| R6, tắt thăm dò (chỉ để chẩn đoán) | 1,14× | 0,91× |
| R6, bỏ cắt K theo request | 1,20× | 0,74× |
| R6, cả hai | 1,18× | 0,92× |

Nguyên nhân: R6 chọn K cho từng request, nhưng **K của batch = max(k_i)** và mọi request trả chi phí của batch.
- **Thăm dò**: mỗi request tự thăm dò (probe) sau `probe_interval = 20` bước ở K=0. Ở B = 32, khoảng 40% số bước có K batch > 0, trong đó 59% chỉ do một request. Một bước K=2 ở B=32 tốn 58 ms so với 24,5 ms (K=0).
- **Cắt K theo request**: ở B=8, request chọn K=1 (28% quyết định) vẫn nằm trong batch K=3 (96% số bước); trả giá K=3 mà chỉ sinh khoảng 1,5 token. Bỏ cắt: τ tăng 1,96 → 2,08.

### 1.1. Điều kiện hoà vốn (α*) theo mức tải (profile A10G, `min_speedup_ratio = 1,05`)

| B | K=1 | K=2 | K=3 | K=4 |
|---:|---:|---:|---:|---:|
| 1 | 0,05 | 0,19 | 0,22 | 0,27 |
| 8 | 0,07 | 0,21 | 0,26 | 0,39 |
| 16 | 0,45 | 0,49 | 0,55 | 0,67 |
| 32 | không thể (> 1) | không thể (> 1) | 0,97 | 0,89 |

α đo được ở B=32 trung vị 0,68, nên ở B=32 không có K nào có thể hoà vốn trong thực tế, và `probe_k = 2` mặc định không bao giờ hoà vốn.

### 1.2. Số tham khảo W2, W3 (lần đo 23/11, chỉ dùng static và vanilla; R6 lần đó không dùng được vì khác container)

| c=32 | W2 (Text-to-SQL) | W3 (phân tích kinh doanh) |
|---|---:|---:|
| Static K=3 so với vanilla | 0,68× | 0,67× |

---

## 2. Yêu cầu chức năng

### FR-01: Chọn K ở mức batch
Trong `AdaptiveBudgetPlanner.plan_step(decode_req_ids, batch_size, ...)`:

1. Với mỗi request đang decode, lấy α_i bằng `alpha_signals` như hiện tại (MLE khi có, effective alpha cũ khi không), **không gọi quyết định K riêng**.
2. Với mọi ứng viên `K` trong `candidate_budgets` (≤ `max_k`):
   ```
   T(K)  = Σ_i E[τ_i(K)]            với E[τ_i(0)] = 1,  E[τ_i(K)] = 1 + Σ_{j=1..K} α_i^j
   C(K)  = cost_profile.get_cost(K, batch_size)      # như hiện tại, batch_size = len(self.running)
   C_ref = min_K C(K)                                # mốc K = 0, như luật per-request hiện tại
   Điểm(K) = T(K) / C(K) với K > 0;   Điểm(0) = n / C_ref      (n = số request)
   ```
   (Cập nhật 25/11/2026: bản trước dùng `C(0, B)`; profile A10 có 5 ô `C(k, B) < C(0, B)` (B = 1, 2, 4), nên ở α ≤ 0,04 và B ≤ 6 đường batch chọn K = 1 trong khi luật cũ chọn K = 0. Dùng `C_ref` để trùng luật cũ theo FR-01.6; có 7 test tham số hoá.)
3. `K_best = argmax_K Điểm(K)`. Chỉ chấp nhận K > 0 khi `Điểm(K_best) ≥ min_speedup_ratio × Điểm(0)` (giữ 1,05). Nếu hoà nhau (±1e-6), lấy K nhỏ hơn.
4. Áp dụng `K_best` cho mọi request: `per_request_k = {req: K_best}`, `batch_k = K_best`. **Không cắt** (trim) K theo từng request.
5. **Memory guard giữ nguyên và ưu tiên cao nhất.** Nếu `select_budget` hoặc memory rule áp K nhỏ hơn cho một request (hay cho cả batch), K cuối của request đó là `min(K_best, K_mem_i)`; chỉ trong trường hợp này mới trim.
6. Với B = 1, kết quả phải trùng quyết định hiện tại (xem AC-03).

### FR-02: Thăm dò (probe) ở mức batch và có điều kiện
1. Planner giữ một bộ đếm `self._batch_fallback_steps`: số bước liên tiếp có K batch = 0 vì lý do chi phí (không tính memory guard). Reset khi chọn K > 0.
2. Khi đếm đạt `probe_interval` (mặc định 20, có thể đổi) **và** tồn tại ít nhất một K khả thi, chạy **một** bước thăm dò với K_probe, rồi reset bộ đếm.
3. Khả thi: `α*(K, B) ≤ α_probe_cap` với `α_probe_cap` = **0,80** **[cần chốt]**. Nếu không có K nào khả thi, bỏ qua thăm dò, không reset bộ đếm ở mức lãng phí; dùng giới hạn trên để bộ đếm không tràn.
4. `K_probe` = K khả thi có `α*` nhỏ nhất (nếu bằng nhau lấy K nhỏ hơn). **Không** dùng `probe_k = 2` cố định.
5. Khi planner bật chế độ batch, per-request probing phải tắt hẳn: planner đặt `probe_interval = 0` cho controller. Hành vi thăm dò chuẩn cũ chỉ còn khi chạy chế độ cũ (xem FR-04).
6. Kỳ vọng: ở B ≤ 16 thăm dò được phép, tốn tối đa 1 bước trong mỗi 20 bước (theo số liệu α* ở B=16: 0,45–0,67). Ở B=32 (α* từ 0,89 trở lên) không thăm dò.

### FR-03: Telemetry
1. Bản ghi `planned_decision` thêm (cộng thêm, không đổi trường cũ): `batch_k_selected`, `batch_score`, `batch_score_k0`, `batch_probe` (bool).
2. `effective_alpha`, `geometric_alpha_mle`, `decision_reason`, `theoretical_speedup` giữ nguyên ý nghĩa hiện tại (theo từng request); bổ sung ghi chú trong `decision_reason` khi K do batch quyết định.
3. Không có `Periodic probe` mức request ở chế độ batch.

### FR-04: Cờ bật/tắt và tương thích ngược
1. Biến môi trường `ADAPTIVE_K_SELECTION` = `batch` | `max_k` (mặc định **`batch`** sau khi AC đạt; `max_k` = hành vi hiện tại, dùng cho đối chứng) **[cần chốt: mặc định]**. `ADAPTIVE_PER_REQUEST` cũ giữ nguyên ý nghĩa khi `max_k`.
2. `LocalServingRunner.setup_environment()` đọc `serving_cfg["k_selection"]` (tuỳ chọn) và chuyển thành biến môi trường; `modal_runner` thêm `k_selection` vào danh sách khóa chuyển tiếp. Không thêm cờ CLI trong đợt này.
3. Xử lý lỗi: giá trị không hợp lệ → `ValueError` rõ ràng khi khởi động (không mặc định âm thầm).

---

## 3. Yêu cầu phi chức năng
- **NFR-01 (hiệu năng):** chi phí chọn K trên CPU trong một bước không tăng quá 1 ms ở B=32 (đo bằng thời gian `plan_step`).
- **NFR-02 (tương thích):** toàn bộ test hiện hành phải pass, trừ các test khẳng định hành vi cắt K theo request ở mức batch (liệt kê và thay bằng test mới).
- **NFR-03 (không đụng artifact profile):** `raw_sha256` của profile giữ nguyên.

---

## 4. Tiêu chí nghiệm thu (AC)

### 4.1. CPU (không cần GPU)
- **AC-01** (FR-01): B=8, α trộn (4 request 0,8; 4 request 0,3), profile A10G: chọn một K duy nhất cho cả batch; `per_request_k` mọi request bằng nhau; `batch_k` bằng giá trị đó.
- **AC-02** (FR-01, khả thi theo profile): B=32, α_i ≈ 0,68 (đo được): `batch_k = 0`. B=1, α=0,6: `batch_k ∈ {3, 4}` và trùng quyết định bản hiện tại.
- **AC-03** (không hồi quy B=1): với α ∈ {0,2; 0,4; 0,6; 0,8} tại B=1, quyết định của bản mới trùng bản cũ (`max_k`) ± chọn K trùng.
- **AC-04** (FR-02): B=32 giả lập 200 bước với α=0,68: số lần thăm dò = 0. B=8, α thấp 0,1 (K=0), 200 bước: số bước thăm dò ≤ 200/`probe_interval`, và `K_probe` khả thi (α* ≤ 0,80).
- **AC-05** (FR-02): chế độ batch, không có `Periodic probe` trong `decision_reason` của từng request.
- **AC-06** (FR-01.5): memory guard giữ K_i ≤ K_mem; request bị guard trim; các request khác không bị trim.
- **AC-07** (FR-03): có các trường mới; các trường MLE và telemetry lô (`flush=64`) không đổi.
- **AC-08** (FR-04): `max_k` cho đúng hành vi hiện tại (so với quyết định ghi trong telemetry `r6_new`).

### 4.2. GPU (cùng một container NVIDIA A10; tắt sớm sau bước đầu)
Cấu hình: vanilla matched, static K=3 matched, R6 mới (batch); `--no-prefix-caching`; c = 1 / 8 / 32; W1, **W2, W3**; 3 lần lặp (đợt này vẫn là kiểm chứng, số chính thức M3-05 đo sau). Ước tính 2–3 USD credit, khoảng 2 giờ; có nấc dừng sớm sau W1 c=32.

| Tiêu chí (đề xuất, **[cần chốt]**) | Ô áp dụng |
|---|---|
| Không hồi quy tải thấp: R6 ≥ 1,55× vanilla ở c=1 (W1) | W1 |
| Trung bình: R6 ≥ 1,15× vanilla ở c=8 (W1) | W1 |
| **Cắt lỗ ở c=32** (R6 so với static và R6 hiện tại; **không** đòi đạt 0,90× ở cả ba node, xem `KET-QUA-3-NODE.md`): W1 R6 ≥ 0,88×; W2, W3 R6 ≥ static + 0,05 và không kém bản R6 hiện tại | W1, W2, W3 ở c=32 (static 0,65 / 0,76 / 0,65×) |
| Không kém bản R6 hiện tại ở cả 3 workload | W1, W2, W3 |
| Telemetry: ở B = 32 tỉ lệ bước có K batch > 0 < 5%; không có thăm dò mức request | mọi workload |

Chú thích "cắt lỗ": ở c=32, static K=3 chậm hơn vanilla 35% (W1), 24% (W2), 35% (W3). Đo 24–25/11 (`KET-QUA-3-NODE.md`) cho thấy R6 hiện tại đã cắt một phần (W2 còn 15%, W3 còn 22%), và bản giả lập sửa (hai công tắc) đưa W1 về 8%, nhưng W2 và W3 không cải thiện thêm vì chi phí bước K=0 trong engine speculative chiếm phần lớn khoản lỗ còn lại (khoảng 20% ở W3). Tiêu chí vì thế theo từng node, không yêu cầu 0,90× ở mọi node.

---

## 5. Giới hạn đã biết (không sửa trong đợt này)
- Bước K = 0 trong engine có speculative vẫn đắt hơn vanilla (kết quả spike: 0,92×, không phải 1,0×). Ghi trong quyển.
- Chi phí đường dynamic-K của vLLM và stack adaptive ở c = 8 (khoảng +3 ms mỗi bước) giữ nguyên.
- Profile đo thiếu phần engine; hướng bổ sung ghi ở `logs/future-work/2026-11-24-r6-engine-step-cost-correction.md`. **Không làm trước FR-01/FR-02** vì khi còn chọn K theo từng request, cộng thêm chi phí vào B = 32 khiến một phần request chọn K = 4 và kéo cả batch.
- Đây là kết quả trên W1; W2 và W3 phải đo theo mục 4.2.

## 6. Kế hoạch triển khai (chia việc đề xuất)
1. Khôi duyệt spec, chốt: mặc định `ADAPTIVE_K_SELECTION`, `α_probe_cap = 0,80`, tiêu chí 4.2.
2. Khôi/Đại/Dương: viết test CPU trước (AC-01 → AC-08) rồi mới sửa mã (TDD).
3. Dương chạy đo GPU kiểm chứng (mục 4.2) sau khi test pass, trên workspace Modal (đã thống nhất với Đại), dừng sớm nếu W1 c=32 không cải thiện.
4. Ghi kết quả vào `progress.md` của thư mục bug tương ứng trong `logs/bugs/`; nếu đạt, đo chính thức M3-05 cho chương 5.

## 7. Phân loại
- **Đã xác minh**: cơ chế max(k_i); thăm dò kéo batch; cắt K mất token; mốc α* ở các mức tải; hai công tắc chẩn đoán cải thiện W1.
- **Giả định cần xác nhận**: kết quả R6 sau khi sửa (dự đoán c=8 khoảng 1,18–1,2×; c=32 khoảng 0,9×); kết quả W2, W3 chưa đo với R6 mới.
- **Quyết định còn mở**: các dòng **[cần chốt]**; phạm vi B dùng cho `C(K, B)` khi có request đang prefill (hiện dùng `len(self.running)`, giữ nguyên).
- **Rủi ro**: thay đổi hành vi chọn K ở mọi request; cần test hồi quy kỹ trước khi bật mặc định.
