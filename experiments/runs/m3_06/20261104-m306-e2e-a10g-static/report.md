# Báo cáo tổng hợp VAT (Node 3)

- run_id: `20261104-m306-e2e-a10g` · lane: `static` · model: `qwen2.5-7b-static-draft`
- Node 3 status: `ok` · finish_reason: `stop`

### Báo cáo tổng hợp tình hình chi tiêu và thuế VAT cho tập 30 hóa đơn bán lẻ đã trích xuất

#### 1. Tổng quan
- Tổng chi tiêu trong tháng 08/2020 là NULL do không có dữ liệu phù hợp.
- Cửa hàng CỬA HÀNG NĂM OÁNH có tổng giá trị hóa đơn cao nhất trong tập dữ liệu.
- Không có hóa đơn nào phát sinh VAT.

#### 2. Mặt hàng và cửa hàng nổi bật
- Không có mặt hàng nào được mua nhiều nhất trong tập dữ liệu.
- Cửa hàng CỬA HÀNG NĂM OÁNH có tổng giá trị hóa đơn cao nhất.

#### 3. Tình hình VAT
- Không có hóa đơn nào phát sinh VAT.

#### 4. Hạn chế của dữ liệu và đề xuất kiểm tra tiếp
- Tổng chi tiêu trong tháng 08/2020 là NULL do chưa có dữ liệu phù hợp.
- Không có mặt hàng nào được mua nhiều nhất, cần kiểm tra lại dữ liệu.
- Không có thông tin về mức thuế suất và tổng tiền thuế theo từng loại thuế, cần kiểm tra lại cấu trúc dữ liệu.

Tổng kết, cần tiếp tục kiểm tra và xác nhận dữ liệu để có được thông tin chính xác về chi tiêu và thuế VAT.

## Phụ lục: kết quả Q1–Q6

### Q1. Tổng chi tiêu trong tháng 08/2020 là bao nhiêu?
```sql
SELECT SUM(total_amount_value) AS total_spending FROM v_invoice_facts WHERE issued_date LIKE '2020-08-%' AND extraction_status = 'Success'
```
Trạng thái: ok
| total_spending |
|---|
| NULL |

### Q2. Mặt hàng nào được mua nhiều nhất?
```sql
SELECT description FROM v_item_facts GROUP BY description ORDER BY SUM(quantity_value) DESC LIMIT 1
```
Trạng thái: empty
KHÔNG CÓ DỮ LIỆU

### Q3. Cửa hàng nào có tổng giá trị hóa đơn cao nhất?
```sql
SELECT seller_name FROM v_invoice_facts WHERE total_amount_value = (SELECT MAX(total_amount_value) FROM v_invoice_facts)
```
Trạng thái: ok
| seller_name |
|---|
| CỬA HÀNG NĂM OÁNH |

### Q4. Có bao nhiêu hóa đơn có phát sinh VAT?
```sql
SELECT COUNT(DISTINCT invoice_id) FROM v_tax_facts WHERE tax_amount_value > 0
```
Trạng thái: ok
| COUNT(DISTINCT invoice_id) |
|---|
| 0 |

### Q5. Tổng tiền theo từng ngày lập hóa đơn là bao nhiêu?
```sql
SELECT issued_date, SUM(total_amount_value) AS total_amount_value FROM v_invoice_facts GROUP BY issued_date
```
Trạng thái: ok
| issued_date | total_amount_value |
|---|---|
| 2019-04-21 | 52000.0 |
| 2019-09-23 | 27200.0 |
| 2020-08-09 | 529016.0 |
| 2020-08-10 | 366000.0 |
| 2020-08-11 | 1006500.0 |
| 2020-08-12 | 934764.0 |
| 2020-08-14 | 396016.0 |
| 2020-08-15 | 722750.0 |
| 2020-08-16 | 1200048.0 |

### Q6. Mức thuế suất và tổng tiền thuế theo từng loại thuế là bao nhiêu?
```sql
SELECT tax_name, SUM(tax_amount_value) AS total_tax_amount_value, MAX(tax_rate) AS max_tax_rate 
FROM v_tax_facts 
GROUP BY tax_name
```
Trạng thái: empty
KHÔNG CÓ DỮ LIỆU
