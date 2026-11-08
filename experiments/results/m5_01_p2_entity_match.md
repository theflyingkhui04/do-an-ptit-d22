# M5-01 — So khớp thực thể Node 1 trên W1_v2 (OCR cả tờ, prompt lượt 2) giữa các làn

Nhãn: `data/processed/m0_04_mcocr_devset/samples.jsonl` · 30 mẫu. Ô bảng: số khớp/số mẫu chấm (tỉ lệ).

## Kết quả theo lần chạy

| Lần chạy | JSON hợp lệ | SELLER | ADDRESS | TIMESTAMPS | TOTAL_COST |
|---|---|---|---|---|---|
| vanilla_r1 | 96.7% | 23/30 (76.7%), F1 0.907 | 20/30 (66.7%), F1 0.906 | 27/30 (90.0%) | 25/30 (83.3%) |
| vanilla_r2 | 96.7% | 23/30 (76.7%), F1 0.919 | 21/30 (70.0%), F1 0.908 | 27/30 (90.0%) | 25/30 (83.3%) |
| static | 96.7% | 23/30 (76.7%), F1 0.907 | 20/30 (66.7%), F1 0.906 | 27/30 (90.0%) | 25/30 (83.3%) |
| adaptive | 96.7% | 23/30 (76.7%), F1 0.919 | 21/30 (70.0%), F1 0.908 | 27/30 (90.0%) | 26/30 (86.7%) |
| eagle | 96.7% | 23/30 (76.7%), F1 0.919 | 21/30 (70.0%), F1 0.908 | 27/30 (90.0%) | 26/30 (86.7%) |

## So với vanilla_r1

Dao động là |vanilla_r2 − vanilla_r1| của từng trường.

| Lần chạy | Trường | Chênh lệch | Dao động vanilla | Trong dao động vanilla |
|---|---|---|---|---|
| static | SELLER | +0.0% | 0.0% | đúng |
| static | ADDRESS | +0.0% | 3.3% | đúng |
| static | TIMESTAMPS | +0.0% | 0.0% | đúng |
| static | TOTAL_COST | +0.0% | 0.0% | đúng |
| adaptive | SELLER | +0.0% | 0.0% | đúng |
| adaptive | ADDRESS | +3.3% | 3.3% | đúng |
| adaptive | TIMESTAMPS | +0.0% | 0.0% | đúng |
| adaptive | TOTAL_COST | +3.3% | 0.0% | sai |
| eagle | SELLER | +0.0% | 0.0% | đúng |
| eagle | ADDRESS | +3.3% | 3.3% | đúng |
| eagle | TIMESTAMPS | +0.0% | 0.0% | đúng |
| eagle | TOTAL_COST | +3.3% | 0.0% | sai |

## Mặt hàng (kiểm tra số học)

Không có nhãn mặt hàng: chỉ kiểm SL × đơn giá = thành tiền (dung sai max(1 VND, 0,5%)) và tổng thành tiền = tổng tiền / tạm tính / tổng tiền + giảm giá (±1 VND). Tên hàng chưa được chấm.

| Lần chạy | Hoá đơn có mặt hàng | Dòng mặt hàng | Dòng khớp SL×ĐG | Hoá đơn khớp tổng |
|---|---|---|---|---|
| vanilla_r1 | 29 | 84 | 83/84 | 28/29 |
| vanilla_r2 | 29 | 84 | 83/84 | 28/29 |
| static | 29 | 84 | 83/84 | 28/29 |
| adaptive | 29 | 84 | 84/84 | 28/29 |
| eagle | 29 | 84 | 83/84 | 28/29 |

## Mẫu thiếu

- vanilla_r1: m0_04_021
- vanilla_r2: m0_04_021
- static: m0_04_021
- adaptive: m0_04_021
- eagle: m0_04_021

## Mẫu lệch

### vanilla_r1

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_001 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Suli Phu Thi Gia Lam |
| m0_04_002 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_006 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |
| m0_04_008 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_008 | TOTAL_COST | blank | Cộng tiền hàng 840,000 | None |
| m0_04_013 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_013 | TOTAL_COST | blank | Cộng tiền hàng 92,000 | None |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tổng cộng | 350000 |
| m0_04_017 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_018 | TIMESTAMPS | mismatch | Ngày: 21.04.2019 Giờ vào: 10.30 Giờ ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_020 | TIMESTAMPS | mismatch | Ngày: 09/08/2020 - 09:26 | 2020-08-09 09:28 |
| m0_04_021 | SELLER | blank | NHÀ SÁCH GD-TC CẨM PHẢ | None |
| m0_04_021 | ADDRESS | blank | ĐC: 212 Đường Trần Phú-Cẩm Phả | None |
| m0_04_021 | TIMESTAMPS | blank | Thời gian: 21:43:39 - 14/08/2020 | None |
| m0_04_021 | TOTAL_COST | blank | Tổng số thanh toán: 75.000đ | None |
| m0_04_022 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_022 | TOTAL_COST | blank | Cộng tiền hàng 150,000 | None |
| m0_04_023 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_024 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phù Thị Gia Lâm |
| m0_04_025 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_025 | ADDRESS | mismatch | Thôn Phú Thuỵ, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội | Thôn Phú Thụy, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK, Tang 1,SH42 - Toa K3,Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi |
| m0_04_030 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |

### vanilla_r2

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_001 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Suli Phu Thi Gia Lam |
| m0_04_002 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_006 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |
| m0_04_008 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_008 | TOTAL_COST | blank | Cộng tiền hàng 840,000 | None |
| m0_04_013 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_013 | TOTAL_COST | blank | Cộng tiền hàng 92,000 | None |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tổng cộng | 350000 |
| m0_04_017 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_018 | TIMESTAMPS | mismatch | Ngày: 21.04.2019 Giờ vào: 10.30 Giờ ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_020 | TIMESTAMPS | mismatch | Ngày: 09/08/2020 - 09:26 | 2020-08-09 09:28 |
| m0_04_021 | SELLER | blank | NHÀ SÁCH GD-TC CẨM PHẢ | None |
| m0_04_021 | ADDRESS | blank | ĐC: 212 Đường Trần Phú-Cẩm Phả | None |
| m0_04_021 | TIMESTAMPS | blank | Thời gian: 21:43:39 - 14/08/2020 | None |
| m0_04_021 | TOTAL_COST | blank | Tổng số thanh toán: 75.000đ | None |
| m0_04_022 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_022 | TOTAL_COST | blank | Cộng tiền hàng 150,000 | None |
| m0_04_023 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_024 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phù Thị Gia Lâm |
| m0_04_025 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_025 | ADDRESS | mismatch | Thôn Phú Thuỵ, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội | Thôn Phú Thụy, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Co.op Food HN THE K-PARK |
| m0_04_030 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |

### static

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_001 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Suli Phu Thi Gia Lam |
| m0_04_002 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_006 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |
| m0_04_008 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_008 | TOTAL_COST | blank | Cộng tiền hàng 840,000 | None |
| m0_04_013 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_013 | TOTAL_COST | blank | Cộng tiền hàng 92,000 | None |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tổng cộng | 350000 |
| m0_04_017 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_018 | TIMESTAMPS | mismatch | Ngày: 21.04.2019 Giờ vào: 10.30 Giờ ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_020 | TIMESTAMPS | mismatch | Ngày: 09/08/2020 - 09:26 | 2020-08-09 09:28 |
| m0_04_021 | SELLER | blank | NHÀ SÁCH GD-TC CẨM PHẢ | None |
| m0_04_021 | ADDRESS | blank | ĐC: 212 Đường Trần Phú-Cẩm Phả | None |
| m0_04_021 | TIMESTAMPS | blank | Thời gian: 21:43:39 - 14/08/2020 | None |
| m0_04_021 | TOTAL_COST | blank | Tổng số thanh toán: 75.000đ | None |
| m0_04_022 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_022 | TOTAL_COST | blank | Cộng tiền hàng 150,000 | None |
| m0_04_023 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_024 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phù Thị Gia Lâm |
| m0_04_025 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_025 | ADDRESS | mismatch | Thôn Phú Thuỵ, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội | Thôn Phú Thụy, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Saigon Co.op |
| m0_04_027 | ADDRESS | mismatch | Tang 1,SH42 - Toa K3, Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi | Co.op Food HN THE K-PARK, Tang 1,SH42 - Toa K3,Cong trinh chung cu ket hop dich vu tren o dat H-CT2 thuoc Du an Dau tu xay dung Khu nha o Hi Brand tai Khu do thi moi Van Phu, P. Phu La, Q. Ha Dong, Tp Ha Noi |
| m0_04_030 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |

### adaptive

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_001 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Suli Phu Thi Gia Lam |
| m0_04_002 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_006 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |
| m0_04_008 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_008 | TOTAL_COST | blank | Cộng tiền hàng 840,000 | None |
| m0_04_013 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tổng cộng | 350000 |
| m0_04_017 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_018 | TIMESTAMPS | mismatch | Ngày: 21.04.2019 Giờ vào: 10.30 Giờ ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_020 | TIMESTAMPS | mismatch | Ngày: 09/08/2020 - 09:26 | 2020-08-09 09:28 |
| m0_04_021 | SELLER | blank | NHÀ SÁCH GD-TC CẨM PHẢ | None |
| m0_04_021 | ADDRESS | blank | ĐC: 212 Đường Trần Phú-Cẩm Phả | None |
| m0_04_021 | TIMESTAMPS | blank | Thời gian: 21:43:39 - 14/08/2020 | None |
| m0_04_021 | TOTAL_COST | blank | Tổng số thanh toán: 75.000đ | None |
| m0_04_022 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_022 | TOTAL_COST | blank | Cộng tiền hàng 150,000 | None |
| m0_04_023 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_024 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phù Thị Gia Lâm |
| m0_04_025 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_025 | ADDRESS | mismatch | Thôn Phú Thuỵ, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội | Thôn Phú Thụy, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Co.op Food HN THE K-PARK |
| m0_04_030 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |

### eagle

| Mẫu | Trường | Trạng thái | Nhãn | Model |
|---|---|---|---|---|
| m0_04_001 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Suli Phu Thi Gia Lam |
| m0_04_002 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_006 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |
| m0_04_008 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_008 | TOTAL_COST | blank | Cộng tiền hàng 840,000 | None |
| m0_04_013 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_015 | TOTAL_COST | mismatch | 335,500 Tổng cộng | 350000 |
| m0_04_017 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_018 | TIMESTAMPS | mismatch | Ngày: 21.04.2019 Giờ vào: 10.30 Giờ ra: 10.31 | 2019-04-21 None |
| m0_04_019 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sùi Phú Thị Gia Lâm |
| m0_04_020 | TIMESTAMPS | mismatch | Ngày: 09/08/2020 - 09:26 | 2020-08-09 09:28 |
| m0_04_021 | SELLER | blank | NHÀ SÁCH GD-TC CẨM PHẢ | None |
| m0_04_021 | ADDRESS | blank | ĐC: 212 Đường Trần Phú-Cẩm Phả | None |
| m0_04_021 | TIMESTAMPS | blank | Thời gian: 21:43:39 - 14/08/2020 | None |
| m0_04_021 | TOTAL_COST | blank | Tổng số thanh toán: 75.000đ | None |
| m0_04_022 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_022 | TOTAL_COST | blank | Cộng tiền hàng 150,000 | None |
| m0_04_023 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_024 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phù Thị Gia Lâm |
| m0_04_025 | SELLER | mismatch | CỬA HÀNG NĂM OÁNH | CỬA HÀNG NAM OÁNH |
| m0_04_025 | ADDRESS | mismatch | Thôn Phú Thuỵ, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội | Thôn Phú Thụy, Xã Phú Thị, Huyện Gia Lâm, TP Hà Nội |
| m0_04_027 | SELLER | mismatch | Saigon Co.op Co.op Food HN THE K-PARK | Co.op Food HN THE K-PARK |
| m0_04_030 | ADDRESS | mismatch | Chợ Sủi Phú Thị Gia Lâm | Chợ Sủi Phu Thị Gia Lâm |
