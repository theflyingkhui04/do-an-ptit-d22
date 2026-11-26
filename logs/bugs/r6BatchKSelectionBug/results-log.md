# Log kết quả đầy đủ: code mới (chọn K theo batch) so với vanilla, static và R6 hiện tại

> Sinh tự động bằng `final_table.py` từ dữ liệu đo; không sửa tay. Bảng tóm tắt và diễn giải: [progress.md](./progress.md).

- `batchk-2511`: GPU `NVIDIA A10, 610.57.04, 23028 MiB, GPU-5608d3ad-a598-66a7-08ec-6684d00483f9`; code `{'cur': 'develop e25530a', 'new': 'develop e25530a + batchk.patch (budget_planner sha256 af2c622a, controller f31fe7a5)'}`; hoàn tất: None; dừng: RuntimeError: harness W3 exit 2
- `batchk-2511-table`: GPU `NVIDIA A10, 610.57.04, 23028 MiB, GPU-eb959eb1-8128-65fb-80b1-6e6d15d3c846`; code `{'cur': 'develop e25530a', 'new': 'develop e25530a + batchk.patch (budget_planner sha256 af2c622a, controller f31fe7a5)'}`; hoàn tất: True; dừng: None
- `batchk-2511-f2`: GPU `NVIDIA A10, 610.57.04, 23028 MiB, GPU-55836b89-a89b-4611-476d-e50e3d691dc5`; code `{'cur': 'develop e25530a', 'new': 'develop e25530a + batchk.patch (budget_planner sha256 af2c622a, controller f31fe7a5)'}`; hoàn tất: True; dừng: None

## 1. Hệ số tốc độ so với vanilla (TPOT, cùng container)

| Hàng | c | Static K=3 | R6 hiện tại | Code mới | Nguồn |
|---|---:|---:|---:|---:|---|
| Node 1 (W1) | 1 | 1.80× | 1.70× | 1.71× | `batchk-2511-table` |
| Node 1 (W1) | 8 | 1.26× | 1.11× | 1.16× | `batchk-2511` |
| Node 1 (W1) | 32 | 0.65× | 0.66× | 0.86× | `batchk-2511` |
| Node 2 (W2) | 8 | 1.26× | 1.17× | 1.25× | `batchk-2511-table` |
| Node 2 (W2) | 32 | 0.77× | 0.83× | 0.89× | `batchk-2511-table` |
| Node 3 (W3) | 8 | 1.46× | 1.24× | 1.35× | `batchk-2511-table` |
| Node 3 (W3) | 32 | 0.71× | 0.82× | 0.83× | `batchk-2511-table` |

### 1b. So trực tiếp trong cùng container (TPOT, > 1 là code mới nhanh hơn)

| Hàng | c | Code mới so với R6 hiện tại | Code mới so với Static K=3 | Vanilla (ms) |
|---|---:|---:|---:|---:|
| Node 1 (W1) | 1 | 1.00× | 0.95× | 17.50 |
| Node 1 (W1) | 8 | 1.05× | 0.92× | 20.11 |
| Node 1 (W1) | 32 | 1.31× | 1.34× | 27.76 |
| Node 2 (W2) | 8 | 1.07× | 0.99× | 21.84 |
| Node 2 (W2) | 32 | 1.08× | 1.15× | 30.46 |
| Node 3 (W3) | 8 | 1.09× | 0.92× | 17.63 |
| Node 3 (W3) | 32 | 1.02× | 1.17× | 18.85 |

### 1c. Vanilla giữa hai container (độ lệch nền)

| Ô | batchk-2511 (ms) | batchk-2511-table (ms) | lệch |
|---|---:|---:|---:|
| W1 c=1 | 15.98 | 17.50 | +9.4% |
| W2 c=8 | 20.53 | 21.84 | +6.4% |
| W2 c=32 | 28.73 | 30.46 | +6.0% |
| W3 c=8 | 16.56 | 17.63 | +6.5% |
| W3 c=32 | 17.61 | 18.85 | +7.0% |

| Ô | code mới batchk-2511 (ms) | code mới batchk-2511-table (ms) | lệch |
|---|---:|---:|---:|
| W1 c=1 | 10.04 | 10.23 | +1.9% |
| W2 c=8 | 17.15 | 17.53 | +2.2% |
| W2 c=32 | 33.21 | 34.24 | +3.1% |
| W3 c=8 | 12.69 | 13.02 | +2.6% |
| W3 c=32 | 22.80 | 22.60 | -0.9% |

## 2. Số đo chi tiết

| Hàng | c | Nhánh | TPOT p50 (ms) | theo lượt lặp | TTFT p50 | e2e p50 (s) | τ | K/bước | α=nhận/nháp | thành công | GPU |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Node 1 (W1) | 1 | vanilla | 17.50 | 17.29 / 18.26 | 360 | 11.79 | 1.00 | 0 | – | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 1 | Static K=3 | 9.70 | 9.59 / 9.81 | 382 | 6.62 | 2.10 | 2.71 | 0.40 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 1 | R6 hiện tại | 10.28 | 10.27 / 10.31 | 379 | 7.02 | 2.05 | 2.72 | 0.39 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 1 | Code mới | 10.23 | 10.21 / 10.26 | 379 | 6.93 | 2.05 | 2.72 | 0.39 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 8 | vanilla | 20.11 | 20.2 / 19.93 | 371 | 13.48 | 1.00 | 0 | – | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 8 | Static K=3 | 15.94 | 15.91 / 16.05 | 427 | 11.03 | 2.10 | 2.71 | 0.40 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 8 | R6 hiện tại | 18.08 | 18.08 / 18.14 | 750 | 12.50 | 1.96 | 2.24 | 0.43 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 8 | Code mới | 17.29 | 17.21 / 17.38 | 604 | 12.09 | 2.02 | 2.44 | 0.42 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 32 | vanilla | 27.76 | 28.12 / 27.43 | 5885 | 23.98 | 1.00 | 0 | – | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 32 | Static K=3 | 43.02 | 43.02 / 42.94 | 6432 | 34.43 | 2.10 | 2.71 | 0.40 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 32 | R6 hiện tại | 42.10 | 42.62 / 41.36 | 6931 | 34.22 | 2.07 | 1.99 | 0.54 | 64/64 | NVIDIA A10 |
| Node 1 (W1) | 32 | Code mới | 32.11 | 31.82 / 32.45 | 6890 | 27.55 | 2.06 | 2.07 | 0.51 | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 8 | vanilla | 21.84 | 21.88 / 21.84 | 197 | 2.26 | 1.00 | 0 | – | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 8 | Static K=3 | 17.40 | 17.85 / 17.1 | 151 | 1.69 | 1.80 | 3.00 | 0.27 | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 8 | R6 hiện tại | 18.74 | 19.11 / 18.18 | 180 | 1.85 | 1.63 | 1.86 | 0.34 | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 8 | Code mới | 17.53 | 18.29 / 17.32 | 207 | 1.72 | 1.68 | 2.08 | 0.32 | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 32 | vanilla | 30.46 | 30.34 / 30.46 | 1458 | 4.24 | 1.00 | 0 | – | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 32 | Static K=3 | 39.37 | 39.14 / 39.37 | 1533 | 4.98 | 1.79 | 3.00 | 0.26 | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 32 | R6 hiện tại | 36.86 | 36.52 / 36.86 | 1529 | 4.69 | 1.59 | 1.95 | 0.30 | 64/64 | NVIDIA A10 |
| Node 2 (W2) | 32 | Code mới | 34.24 | 34.24 / 34.53 | 1531 | 4.68 | 1.59 | 1.68 | 0.35 | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 8 | vanilla | 17.63 | 17.8 / 17.59 | 66 | 8.80 | 1.00 | 0 | – | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 8 | Static K=3 | 12.03 | 11.96 / 12.04 | 97 | 6.12 | 1.79 | 3.00 | 0.26 | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 8 | R6 hiện tại | 14.23 | 14.11 / 14.36 | 87 | 6.99 | 1.60 | 1.72 | 0.35 | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 8 | Code mới | 13.02 | 13.04 / 13.01 | 85 | 6.35 | 1.59 | 1.74 | 0.34 | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 32 | vanilla | 18.85 | 18.84 / 18.88 | 966 | 10.33 | 1.00 | 0 | – | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 32 | Static K=3 | 26.43 | 26.44 / 26.34 | 1010 | 14.36 | 1.79 | 3.00 | 0.26 | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 32 | R6 hiện tại | 23.11 | 23.51 / 23.01 | 1018 | 12.28 | 1.64 | 1.79 | 0.36 | 64/64 | NVIDIA A10 |
| Node 3 (W3) | 32 | Code mới | 22.60 | 22.63 / 22.53 | 1014 | 12.10 | 1.60 | 1.49 | 0.40 | 64/64 | NVIDIA A10 |

## 3. Telemetry theo kích thước batch (một dòng = một lượt lập lịch)

| Run | Nhánh | B | lượt | K>0 | thăm dò | K lẫn lộn | phân bố K |
|---|---|---|---:|---:|---:|---:|---|
| `batchk-2511` (A) | R6 hiện tại | B=1 | 215 | 100.0% | 0 | 0.0% | {1: 21, 3: 97, 4: 97} |
| `batchk-2511` (A) | R6 hiện tại | B2–6 | 252 | 100.0% | 0 | 65.9% | {3: 168, 4: 84} |
| `batchk-2511` (A) | R6 hiện tại | B7–12 | 2959 | 100.0% | 0 | 58.4% | {1: 111, 3: 2848} |
| `batchk-2511` (A) | R6 hiện tại | B13–24 | 73 | 100.0% | 0 | 100.0% | {2: 21, 3: 52} |
| `batchk-2511` (A) | R6 hiện tại | B25–32 | 1727 | 40.2% | 0 | 40.2% | {0: 1033, 2: 655, 4: 39} |
| `batchk-2511` (A) | Code mới | B=1 | 222 | 100.0% | 0 | 0.0% | {1: 48, 3: 77, 4: 97} |
| `batchk-2511` (A) | Code mới | B2–6 | 190 | 100.0% | 0 | 0.0% | {3: 172, 4: 18} |
| `batchk-2511` (A) | Code mới | B7–12 | 2890 | 100.0% | 0 | 0.0% | {1: 477, 3: 2413} |
| `batchk-2511` (A) | Code mới | B13–24 | 80 | 100.0% | 0 | 0.0% | {2: 80} |
| `batchk-2511` (A) | Code mới | B25–32 | 1816 | 0.0% | 0 | 0.0% | {0: 1816} |
| `batchk-2511` (B) | Code mới | B=1 | 21383 | 100.0% | 0 | 0.0% | {1: 4099, 3: 8308, 4: 8976} |
| `batchk-2511` (B) | Code mới | B2–6 | 750 | 100.0% | 0 | 0.0% | {1: 289, 2: 5, 3: 456} |
| `batchk-2511` (B) | Code mới | B7–12 | 3212 | 100.0% | 0 | 0.0% | {1: 2004, 3: 1208} |
| `batchk-2511` (B) | Code mới | B13–24 | 304 | 80.9% | 2 | 0.0% | {0: 58, 1: 174, 2: 64, 3: 8} |
| `batchk-2511` (B) | Code mới | B25–32 | 1402 | 0.0% | 0 | 0.0% | {0: 1402} |
| `batchk-2511-table` (T) | R6 hiện tại | B=1 | 21482 | 100.0% | 0 | 0.0% | {1: 4173, 3: 8331, 4: 8978} |
| `batchk-2511-table` (T) | R6 hiện tại | B2–6 | 825 | 100.0% | 0 | 78.4% | {1: 81, 2: 12, 3: 726, 4: 6} |
| `batchk-2511-table` (T) | R6 hiện tại | B7–12 | 3128 | 100.0% | 0 | 95.7% | {1: 107, 3: 3021} |
| `batchk-2511-table` (T) | R6 hiện tại | B13–24 | 311 | 98.1% | 0 | 95.2% | {0: 6, 1: 23, 2: 237, 3: 45} |
| `batchk-2511-table` (T) | R6 hiện tại | B25–32 | 1391 | 15.3% | 0 | 15.3% | {0: 1178, 2: 192, 4: 21} |
| `batchk-2511-table` (T) | Code mới | B=1 | 21410 | 100.0% | 0 | 0.0% | {1: 4137, 3: 8296, 4: 8977} |
| `batchk-2511-table` (T) | Code mới | B2–6 | 784 | 100.0% | 0 | 0.0% | {1: 316, 2: 5, 3: 463} |
| `batchk-2511-table` (T) | Code mới | B7–12 | 3146 | 100.0% | 0 | 0.0% | {1: 1927, 3: 1219} |
| `batchk-2511-table` (T) | Code mới | B13–24 | 351 | 70.4% | 4 | 0.0% | {0: 104, 1: 181, 2: 57, 3: 9} |
| `batchk-2511-table` (T) | Code mới | B25–32 | 1331 | 0.0% | 0 | 0.0% | {0: 1331} |
