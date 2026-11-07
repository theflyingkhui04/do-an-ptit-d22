# M4-03 — So khớp thực thể Node 1 giữa các làn

Nhãn: `data/processed/m0_04_mcocr_devset/samples.jsonl` · 30 mẫu. Ô bảng: số khớp/số mẫu chấm (tỉ lệ).

## Kết quả theo lần chạy

| Lần chạy | JSON hợp lệ | SELLER | ADDRESS | TIMESTAMPS | TOTAL_COST |
|---|---|---|---|---|---|
| vanilla_r1 | 100.0% | 29/30 (96.7%), F1 0.982 | 27/30 (90.0%), F1 0.960 | 30/30 (100.0%) | 30/30 (100.0%) |
| vanilla_r2 | 100.0% | 29/30 (96.7%), F1 0.982 | 28/30 (93.3%), F1 0.964 | 30/30 (100.0%) | 30/30 (100.0%) |
| static | 100.0% | 29/30 (96.7%), F1 0.982 | 27/30 (90.0%), F1 0.960 | 30/30 (100.0%) | 30/30 (100.0%) |
| adaptive | 100.0% | 29/30 (96.7%), F1 0.982 | 27/30 (90.0%), F1 0.960 | 30/30 (100.0%) | 30/30 (100.0%) |
| eagle | 100.0% | 29/30 (96.7%), F1 0.982 | 27/30 (90.0%), F1 0.960 | 30/30 (100.0%) | 30/30 (100.0%) |

## So với vanilla_r1

Dao động là |vanilla_r2 − vanilla_r1| của từng trường.

| Lần chạy | Trường | Chênh lệch | Dao động vanilla | Trong dao động vanilla |
|---|---|---|---|---|
| static | SELLER | +0.0% | 0.0% | đúng |
| static | ADDRESS | +0.0% | 3.3% | đúng |
| static | TIMESTAMPS | +0.0% | 0.0% | đúng |
| static | TOTAL_COST | +0.0% | 0.0% | đúng |
| adaptive | SELLER | +0.0% | 0.0% | đúng |
| adaptive | ADDRESS | +0.0% | 3.3% | đúng |
| adaptive | TIMESTAMPS | +0.0% | 0.0% | đúng |
| adaptive | TOTAL_COST | +0.0% | 0.0% | đúng |
| eagle | SELLER | +0.0% | 0.0% | đúng |
| eagle | ADDRESS | +0.0% | 3.3% | đúng |
| eagle | TIMESTAMPS | +0.0% | 0.0% | đúng |
| eagle | TOTAL_COST | +0.0% | 0.0% | đúng |

## Mẫu lệch

### vanilla_r1

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_021 | ADDRESS | mismatch | ĐC: 212 Đường Trần Phú-Cẩm Phả | 212 Đường Trần Phú-Cẩm Phả, Cẩm Phả |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi |
| m0_04_029 | ADDRESS | blank | Chợ Sủi Phú Thị Gia Lâm | None |

### vanilla_r2

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi |
| m0_04_029 | ADDRESS | blank | Chợ Sủi Phú Thị Gia Lâm | None |

### static

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_021 | ADDRESS | mismatch | ĐC: 212 Đường Trần Phú-Cẩm Phả | 212 Đường Trần Phú-Cẩm Phả, Cẩm Phả |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi |
| m0_04_029 | ADDRESS | blank | Chợ Sủi Phú Thị Gia Lâm | None |

### adaptive

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_021 | ADDRESS | mismatch | ĐC: 212 Đường Trần Phú-Cẩm Phả | 212 Đường Trần Phú-Cẩm Phả, Cẩm Phả |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi |
| m0_04_029 | ADDRESS | blank | Chợ Sủi Phú Thị Gia Lâm | None |

### eagle

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_021 | ADDRESS | mismatch | ĐC: 212 Đường Trần Phú-Cẩm Phả | 212 Đường Trần Phú-Cẩm Phả, Cẩm Phả |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi |
| m0_04_029 | ADDRESS | blank | Chợ Sủi Phú Thị Gia Lâm | None |
