# [M5-01] Node 1 W1_v3 sau khi hệ thống điền lại khối nguồn (14/11)

Nhãn: `data/processed/m0_04_mcocr_devset/samples.jsonl` · 30 mẫu. Ô bảng: số khớp/số mẫu chấm (tỉ lệ).

## Kết quả theo lần chạy

| Lần chạy | JSON hợp lệ | SELLER | ADDRESS | TIMESTAMPS | TOTAL_COST |
|---|---|---|---|---|---|
| vanilla_r1 | 83.3% | 22/30 (73.3%), F1 0.771 | 18/30 (60.0%), F1 0.775 | 20/30 (66.7%) | 24/30 (80.0%) |
| restore_vanilla | 100.0% | 26/30 (86.7%), F1 0.904 | 21/30 (70.0%), F1 0.889 | 25/30 (83.3%) | 29/30 (96.7%) |
| restore_adaptive | 100.0% | 26/30 (86.7%), F1 0.904 | 21/30 (70.0%), F1 0.889 | 25/30 (83.3%) | 28/30 (93.3%) |

## Mặt hàng (kiểm tra số học)

Không có nhãn mặt hàng: chỉ kiểm SL × đơn giá = thành tiền (dung sai max(1 VND, 0,5%)) và tổng thành tiền = tổng tiền / tạm tính / tổng tiền + giảm giá (±1 VND). Tên hàng chưa được chấm.

| Lần chạy | Hoá đơn có mặt hàng | Dòng mặt hàng | Dòng khớp SL×ĐG | Hoá đơn khớp tổng |
|---|---|---|---|---|
| vanilla_r1 | 25 | 71 | 65/71 | 23/25 |
| restore_vanilla | 30 | 85 | 77/85 | 27/30 |
| restore_adaptive | 30 | 85 | 77/85 | 27/30 |

## Mẫu thiếu

- vanilla_r1: m0_04_004, m0_04_008, m0_04_020, m0_04_022, m0_04_029

## Mẫu lệch

### vanilla_r1

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_002 | TIMESTAMPS | mismatch | Ngay: 12/08/2020 - 10:46 | 2020-08-12 10:45:00 |
| m0_04_004 | SELLER | blank | MINIMART ANAN | None |
| m0_04_004 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_004 | TIMESTAMPS | blank | Ngay: 12/08/2020-12:03 | None |
| m0_04_004 | TOTAL_COST | blank | Tong tien 47,000 | None |
| m0_04_005 | TIMESTAMPS | mismatch | Ngay: 16/08/2020 21:18 | 2020-08-16 None |
| m0_04_006 | SELLER | mismatch | MINIMART ANAN | MINIMARTANAN |
| m0_04_008 | SELLER | blank | CUA HANG NAM OANH | None |
| m0_04_008 | ADDRESS | blank | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | None |
| m0_04_008 | TIMESTAMPS | blank | Ngay: 11-08-2020 | None |
| m0_04_008 | TOTAL_COST | blank | Cong tien hang 840,000 | None |
| m0_04_009 | ADDRESS | mismatch | Cho Sui Phu Thi Gia Lam | Trong Cho Sui Phu Thi Gia Lam |
| m0_04_010 | SELLER | mismatch | MINIMART ANAN | MINIMART |
| m0_04_010 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_010 | TIMESTAMPS | mismatch | Ngay: 15/08/2020 - 18:24 | 2020-08-15 None |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tong cong | 350000 |
| m0_04_018 | TIMESTAMPS | mismatch | Ngay: 21.04.2019 Gio vao: 10.30 Gio ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Cho Sui Phu Thi Gia Lam | Va Cho Sui Phu Thi Gia Lam Do |
| m0_04_020 | SELLER | blank | MINIMART ANAN | None |
| m0_04_020 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_020 | TIMESTAMPS | blank | Ngay: 09/08/2020 - 09:26 | None |
| m0_04_020 | TOTAL_COST | blank | Tong cong cong 295,000 | None |
| m0_04_021 | ADDRESS | mismatch | DC: 212 Duong Tran Phu-Cam Pha | 212 Duong Tran Phu-Cam Pha |
| m0_04_022 | SELLER | blank | CUA HANG NAM OANH | None |
| m0_04_022 | ADDRESS | blank | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | None |
| m0_04_022 | TIMESTAMPS | blank | Ngay: 11-08-2020 | None |
| m0_04_022 | TOTAL_COST | blank | Cong tien hang 150,000 | None |
| m0_04_023 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Thon Phu Thuy, Xa Phu Thi, Huyen Gan Lam, TP Ha Noi |
| m0_04_025 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Thon Phu Thong, Xa Phu Thi Nam Oan Lam, TP Ha Noi |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK, Tang 1,SH42 - Toa K3,Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Xhu do thi moi Van Phus P. Phu La, 0. Ha Dong, Tp Ha Noi |
| m0_04_028 | TIMESTAMPS | blank | Ngay: 14/08/2020 14:39 | None |
| m0_04_029 | SELLER | blank | MINIMART ANAN | None |
| m0_04_029 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_029 | TIMESTAMPS | blank | Ngay: 16/08/2020 - 17:03 | None |
| m0_04_029 | TOTAL_COST | blank | 69,000 Tong tien | None |

### restore_vanilla

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_002 | TIMESTAMPS | mismatch | Ngay: 12/08/2020 - 10:46 | 2020-08-12 10:45:00 |
| m0_04_004 | SELLER | mismatch | MINIMART ANAN | MINIMARTANAN Va Cho Sui Phu Thi Gia Lam |
| m0_04_004 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_005 | TIMESTAMPS | mismatch | Ngay: 16/08/2020 21:18 | 2020-08-16 None |
| m0_04_006 | SELLER | mismatch | MINIMART ANAN | MINIMARTANAN |
| m0_04_008 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Than Phu Thung Va Ha Tri, Thuan Cua Lam, TP Ha Noi |
| m0_04_009 | ADDRESS | mismatch | Cho Sui Phu Thi Gia Lam | Trong Cho Sui Phu Thi Gia Lam |
| m0_04_010 | SELLER | mismatch | MINIMART ANAN | MINIMART |
| m0_04_010 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_010 | TIMESTAMPS | mismatch | Ngay: 15/08/2020 - 18:24 | 2020-08-15 None |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tong cong | 350000 |
| m0_04_018 | TIMESTAMPS | mismatch | Ngay: 21.04.2019 Gio vao: 10.30 Gio ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Cho Sui Phu Thi Gia Lam | Va Cho Sui Phu Thi Gia Lam Do |
| m0_04_021 | ADDRESS | mismatch | DC: 212 Duong Tran Phu-Cam Pha | 212 Duong Tran Phu-Cam Pha |
| m0_04_023 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Thon Phu Thuy, Xa Phu Thi, Huyen Gan Lam, TP Ha Noi |
| m0_04_025 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Thon Phu Thong, Xa Phu Thi Nam Oan Lam, TP Ha Noi |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK, Tang 1,SH42 - Toa K3,Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Xhu do thi moi Van Phus P. Phu La, 0. Ha Dong, Tp Ha Noi |
| m0_04_028 | TIMESTAMPS | blank | Ngay: 14/08/2020 14:39 | None |

### restore_adaptive

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_002 | TIMESTAMPS | mismatch | Ngay: 12/08/2020 - 10:46 | 2020-08-12 10:45:00 |
| m0_04_004 | SELLER | mismatch | MINIMART ANAN | MINIMARTANAN Va Cho Sui Phu Thi Gia Lam |
| m0_04_004 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_005 | TIMESTAMPS | mismatch | Ngay: 16/08/2020 21:18 | 2020-08-16 None |
| m0_04_006 | SELLER | mismatch | MINIMART ANAN | MINIMARTANAN |
| m0_04_008 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Than Phu Thung Va Ha Tri, Thuan Cua Lam, TP Ha Noi |
| m0_04_009 | ADDRESS | mismatch | Cho Sui Phu Thi Gia Lam | Trong Cho Sui Phu Thi Gia Lam |
| m0_04_010 | SELLER | mismatch | MINIMART ANAN | MINIMART |
| m0_04_010 | ADDRESS | blank | Cho Sui Phu Thi Gia Lam | None |
| m0_04_010 | TIMESTAMPS | mismatch | Ngay: 15/08/2020 - 18:24 | 2020-08-15 None |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tong cong | 350000 |
| m0_04_018 | TIMESTAMPS | mismatch | Ngay: 21.04.2019 Gio vao: 10.30 Gio ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Cho Sui Phu Thi Gia Lam | Va Cho Sui Phu Thi Gia Lam Do |
| m0_04_021 | ADDRESS | mismatch | DC: 212 Duong Tran Phu-Cam Pha | 212 Duong Tran Phu-Cam Pha |
| m0_04_022 | TOTAL_COST | blank | Cong tien hang 150,000 | None |
| m0_04_023 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Thon Phu Thuy, Xa Phu Thi, Huyen Gan Lam, TP Ha Noi |
| m0_04_025 | ADDRESS | mismatch | Thon Phu Thuy, Xa Phu Thi, Huyen Gia Lam, TP Ha Noi | Thon Phu Thong, Xa Phu Thi Nam Oan Lam, TP Ha Noi |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK, Tang 1,SH42 - Toa K3,Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Xhu do thi moi Van Phus P. Phu La, 0. Ha Dong, Tp Ha Noi |
| m0_04_028 | TIMESTAMPS | blank | Ngay: 14/08/2020 14:39 | None |
