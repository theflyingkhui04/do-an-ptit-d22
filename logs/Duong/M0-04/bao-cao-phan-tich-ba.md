# BÁO CÁO PHÂN TÍCH NGHIỆP VỤ VÀ MÔ HÌNH HÓA DỮ LIỆU HÓA ĐƠN

> **Mốc:** M0-04  
> **Vai trò:** BA - Phân tích nghiệp vụ và mô hình hóa dữ liệu  
> **Phạm vi:** Hóa đơn bán lẻ/hóa đơn VAT tiếng Việt trong workload MC-OCR  
> **Schema áp dụng:** `invoice_extraction.v1`  
> **SQLite version:** `user_version=1`

## 1. Mục đích và phạm vi phân tích

Tài liệu này mô tả cách chuyển thông tin trên hóa đơn bán lẻ hoặc hóa đơn giá trị gia tăng (VAT) thành dữ liệu có cấu trúc để phục vụ ba node của pipeline MC-OCR. Mục tiêu của phần BA không phải là thay thế hệ thống kế toán hay đưa ra kết luận pháp lý về tính hợp lệ của hóa đơn. Mục tiêu là xác định các trường nghiệp vụ cần giữ lại, quan hệ giữa các trường, quy tắc biểu diễn dữ liệu thiếu và các câu hỏi mà người dùng có thể đặt trên dữ liệu sau khi trích xuất.

Dữ liệu đầu vào của Node 1 là văn bản OCR tiếng Việt. Văn bản có thể bị thiếu dòng, sai ký tự, sai mã hóa tiếng Việt, không có đầy đủ mã số thuế hoặc không chứa danh mục hàng hóa. Vì vậy, mô hình dữ liệu phải phân biệt rõ ba trạng thái: thông tin có bằng chứng trong OCR, thông tin không xuất hiện và thông tin không thể chuẩn hóa chắc chắn. Trường scalar thiếu được biểu diễn bằng `null`; danh sách thiếu được biểu diễn bằng `[]`; không được tự suy đoán giá trị nghiệp vụ.

Pipeline được phân tách như sau:

```text
OCR text
  -> Node 1: text-to-JSON
  -> invoice_extraction.v1
  -> SQLite: invoices/invoice_items/invoice_taxes
  -> Node 2: Vietnamese question-to-SQL
```

Node 1 chịu trách nhiệm chuẩn hóa đầu ra. SQLite chịu trách nhiệm lưu trữ quan hệ, provenance và khả năng truy vấn. Node 2 chỉ đọc các view đã được thiết kế để giảm độ phức tạp của câu SQL và tránh cho LLM truy cập trực tiếp vào JSON thô.

### Ranh giới trạng thái của M0-04

Các nội dung đã **chốt và có artefact** trong mốc này gồm JSON Schema
`invoice_extraction.v1`, SQLite DDL `user_version=1`, ba view đọc dành cho
Node 2, prompt Node 1, dev-set 30 mẫu và runner Node 1 qua giao diện
OpenAI-compatible. Node 1 đã được smoke-test và chạy đủ dev-set bằng Ollama.

Node 2 trong M0-04 đã chốt contract dữ liệu, catalog câu hỏi và nguyên tắc
chỉ đọc; runner gọi LLM để sinh SQL chưa phải artefact hoàn thiện của mốc này.
Node 3 SQL-to-BI vẫn là tầng tùy chọn trong kiến trúc, chưa được trình bày như
một phần đã triển khai. Benchmark hiệu năng chính thức vẫn thuộc luồng vLLM
tiếp theo, không được suy ra từ kết quả functional PoC Ollama.

## 2. Phân tích cấu trúc nghiệp vụ hóa đơn

### 2.1. Phân loại tài liệu

Trong dev-set MC-OCR có thể xuất hiện cả hóa đơn bán lẻ và chứng từ có đặc điểm của hóa đơn VAT. Trường `document_type` sử dụng ba giá trị:

- `retail_receipt`: biên lai/hóa đơn bán lẻ, thường có tên cửa hàng và tổng tiền nhưng có thể không có MST, số hóa đơn hoặc dòng VAT.
- `vat_invoice`: chỉ dùng khi OCR có bằng chứng rõ ràng về hóa đơn VAT, mã số thuế, thuế suất, tiền VAT hoặc các dấu hiệu tương đương.
- `unknown`: dùng khi không đủ bằng chứng để phân loại.

Việc phân loại không được dựa trên tên file hoặc suy đoán từ một trường đơn lẻ. Nếu văn bản chỉ có tên cửa hàng, ngày và tổng tiền thì Node 1 nên chọn `retail_receipt` hoặc `unknown`, không tự gắn nhãn `vat_invoice`.

### 2.2. Thông tin bên bán

Bên bán là chủ thể phát hành hóa đơn hoặc cung cấp hàng hóa. Các trường thiết yếu gồm:

| Trường | Ý nghĩa nghiệp vụ | Cách xử lý |
|---|---|---|
| `seller.name` | Tên cửa hàng/doanh nghiệp/đơn vị bán | Giữ nội dung OCR có bằng chứng; không tự sửa tên thương hiệu. |
| `seller.address` | Địa chỉ điểm bán hoặc trụ sở thể hiện trên chứng từ | Có thể giữ nguyên chuỗi OCR; không tự bổ sung địa chỉ từ nguồn bên ngoài. |
| `seller.tax_code` | Mã số thuế người bán | Chỉ điền khi OCR thể hiện đủ căn cứ; nếu thiếu dùng `null`. |
| `seller.phone` | Số điện thoại liên hệ | Không bắt buộc cho mọi loại hóa đơn; không suy ra từ mã khác. |

Mã số thuế là trường quan trọng để nhóm hóa đơn theo đơn vị bán, nhưng chất lượng OCR có thể làm sai một hoặc nhiều chữ số. Vì vậy, raw text và `field_evidence` phải được giữ lại để audit. Schema không coi MST là khóa chính và không cho phép dùng MST thiếu bằng chứng để tạo dữ liệu mới.

### 2.3. Thông tin bên mua

Hóa đơn VAT thường có thông tin bên mua đầy đủ hơn hóa đơn bán lẻ. Mô hình vẫn giữ cùng nhóm trường với bên bán để hỗ trợ cả hai loại chứng từ:

- `buyer.name`: tên cá nhân hoặc tổ chức mua hàng.
- `buyer.address`: địa chỉ người mua.
- `buyer.tax_code`: mã số thuế người mua nếu có.
- `buyer.phone`: số điện thoại nếu có.

Toàn bộ trường bên mua đều nullable. Đây là lựa chọn nghiệp vụ có chủ ý vì nhiều hóa đơn bán lẻ không yêu cầu hoặc không hiển thị thông tin người mua. Việc bắt buộc model điền chuỗi rỗng hoặc giá trị giả sẽ làm giảm chất lượng dữ liệu và gây khó khăn cho các truy vấn đếm/tổng hợp.

### 2.4. Thông tin hóa đơn và thời gian

Nhóm `invoice` mô tả định danh và thời điểm phát hành:

| Trường | Ý nghĩa |
|---|---|
| `invoice.invoice_number` | Số hóa đơn hoặc số chứng từ nếu có. |
| `invoice.invoice_series` | Ký hiệu/series hóa đơn nếu có. |
| `invoice.issued_date` | Ngày lập, chuẩn hóa về `YYYY-MM-DD`. |
| `invoice.issued_time` | Giờ lập, chuẩn hóa về `HH:MM[:SS]`. |
| `invoice.issued_at` | Timestamp nội bộ khi có cả ngày và giờ, dạng `YYYY-MM-DDTHH:MM:SS`. |
| `invoice.currency` | Mã tiền tệ ISO 4217, thông thường là `VND`. |
| `invoice.payment_method` | Tiền mặt, chuyển khoản, thẻ hoặc phương thức thể hiện trên OCR. |

Không suy ra timezone từ MC-OCR. Nếu chỉ có ngày thì `issued_date` được điền còn `issued_at` để `null`. Nếu thời gian không có giây, Node 1 có thể chuẩn hóa về giây bằng `00` khi có căn cứ rằng đó là cùng timestamp; quy tắc này được mô tả trong prompt và schema.

### 2.5. Danh mục hàng hóa

Mỗi dòng hàng là một thực thể riêng trong `items` và sau đó được lưu ở bảng `invoice_items`. Các trường cần thiết gồm:

- `line_number`: số thứ tự dòng, bắt đầu từ 1 và không trùng trong cùng hóa đơn.
- `description`: tên/mô tả hàng hóa.
- `sku`: mã hàng nếu có.
- `quantity`: số lượng, đơn vị và chuỗi raw.
- `unit_price`: đơn giá, chuỗi raw, giá trị số và tiền tệ.
- `line_total`: thành tiền dòng, chuỗi raw, giá trị số và tiền tệ.
- `vat_rate`: thuế suất gắn với dòng hàng nếu thể hiện.

Không tạo dòng hàng chỉ vì nhìn thấy một con số đứng cạnh tổng tiền. Một item chỉ được tạo khi OCR có đủ ngữ cảnh hàng hóa hoặc dòng danh mục. Với receipt chỉ có tổng tiền, `items` có thể là `[]` trong khi `totals.total_amount` vẫn được điền.

### 2.6. Tổng tiền và thuế VAT

Nhóm `totals` sử dụng money object gồm ba thuộc tính:

```json
{"raw": "42,000", "value": 42000, "currency": "VND"}
```

Các loại tổng chính:

- `subtotal`: tổng tiền hàng trước giảm giá/thuế nếu có bằng chứng.
- `discount`: tổng mức giảm giá.
- `vat_amount`: tiền thuế VAT.
- `total_amount`: tổng tiền phải thanh toán.

Nhóm `taxes` lưu các dòng thuế có thể xuất hiện theo loại hoặc theo mức thuế suất. Mỗi dòng có `tax_name`, `tax_rate`, `taxable_amount` và `tax_amount`. Nếu OCR không có thông tin thuế, danh sách là `[]` và không được suy ra VAT từ một tổng tiền duy nhất.

Mô hình giữ cả `raw` và `value` vì raw text cần cho audit còn value cần cho SQL aggregation. Giá trị số tiền được chuẩn hóa bằng cách loại bỏ dấu phân tách hàng nghìn theo quy ước của input; cần thận trọng với dấu thập phân và không được coi mọi dấu chấm là phân tách hàng nghìn nếu ngữ cảnh không rõ.

## 3. Từ điển dữ liệu và quy tắc chất lượng

### 3.1. Quy tắc nguồn dữ liệu

Các trường `source.sample_id`, `source.img_id`, `source.image_path` và `source.raw_text` thuộc quyền sở hữu của pipeline. Node 1 phải sao chép đúng các giá trị này từ request, không được sửa lỗi chính tả hay tự thay đổi đường dẫn. Runner so sánh `source` trong output với `expected_source` trước khi insert SQLite. Nếu không khớp, output bị loại dù JSON có hợp lệ về mặt cú pháp.

Nhãn annotation của MC-OCR không đưa vào prompt model. Nhãn chỉ phục vụ audit/evaluation ở tầng ngoài. Quy tắc này tránh leakage và giúp phép đo phản ánh năng lực trích xuất từ OCR text thay vì khả năng sao chép đáp án.

### 3.2. Quy tắc trạng thái trích xuất

`extraction.status` có ba giá trị:

- `ok`: dữ liệu hữu ích đã trích xuất và `missing_fields` rỗng.
- `partial`: có dữ liệu nghiệp vụ hữu ích nhưng còn ít nhất một trường quan trọng bị thiếu.
- `failed`: không trích xuất được dữ liệu nghiệp vụ đáng tin cậy.

`confidence` là audit hint, không phải ground truth và không được dùng để kết luận chất lượng benchmark. `missing_fields` và `warnings` phải là mảng chuỗi không trùng lặp. `field_evidence` phải chỉ ra đoạn OCR ngắn làm căn cứ cho trường đã trích xuất.

Các quy tắc này được kiểm bằng JSON Schema tại [invoice_extraction.schema.json](../../../src/pipeline/schemas/invoice_extraction.schema.json), prompt tại [node1_extractor_system.md](../../../src/pipeline/prompts/node1_extractor_system.md) và runner tại `scripts/run_node1_extractor.py`.

## 4. Thiết kế JSON Schema cho Node 1

### 4.1. Mục tiêu thiết kế

JSON Schema `invoice_extraction.v1` ép LLM trả về một object ổn định với các nhóm key ở cấp cao nhất: `schema_version`, `document_type`, `source`, `seller`, `buyer`, `invoice`, `totals`, `taxes`, `items` và `extraction`. Schema đặt `additionalProperties=false` ở các object chính để phát hiện key ngoài hợp đồng thay vì âm thầm lưu dữ liệu không được định nghĩa.

Các object lặp lại được tách thành definition dùng lại:

- `party` cho seller/buyer.
- `money` cho subtotal, discount, VAT amount, total và các trường tiền trong item/tax.
- `quantity` cho số lượng hàng hóa.
- `line_item` cho từng dòng hàng.
- `tax_line` cho từng dòng thuế.
- `field_evidence` cho bằng chứng OCR.

Cách thiết kế này làm giảm sự khác biệt giữa các output của LLM, thuận tiện cho validator và giúp SQLite mapping có quy tắc cố định.

### 4.2. Nullable và collection

Các trường có thể không xuất hiện trên hóa đơn được khai báo kiểu `string | null`, `number | null` hoặc `currency | null`. Các nhóm lặp lại luôn là array. Do đó, model phải trả về `null` cho `seller.tax_code` khi thiếu MST, chứ không được bỏ key; phải trả về `items: []` khi không có dòng hàng, chứ không được trả về chuỗi mô tả.

Schema cũng ràng buộc miền giá trị: tiền và số lượng không âm, thuế suất trong khoảng 0 đến 100, confidence trong khoảng 0 đến 1, mã tiền tệ gồm ba chữ cái in hoa, và line number bắt đầu từ 1. Các kiểm tra ngữ nghĩa như source binding, ngày hợp lệ và line number duy nhất được thực hiện thêm trong runner/SQLite loader.

### 4.3. Ví dụ input/output

Ví dụ input chuẩn nằm tại [node1_extractor_input_m0_04_001.txt](../../../src/pipeline/examples/node1_extractor_input_m0_04_001.txt). Input chứa metadata nguồn và OCR text:

```text
SOURCE_METADATA:
{"sample_id": "m0_04_001", "img_id": "mcocr_public_145013tlipr.jpg", "image_path": "data/processed/m0_04_mcocr_devset/images/mcocr_public_145013tlipr.jpg"}

OCR_TEXT:
MINIMART ANAN
Chợ Sủi Phú Thị Gia Lâm
Ngày: 10/08/2020 - 17:12
Tổng tiền
42,000
```

Output tương ứng nằm tại [node1_extractor_output_m0_04_001.json](../../../src/pipeline/examples/node1_extractor_output_m0_04_001.json). Output giữ source verbatim, chuẩn hóa ngày giờ, ghi `total_amount.value=42000` và để các trường không có bằng chứng ở `null`/`[]`. Đây là ví dụ cho nguyên tắc: thiếu dữ liệu không đồng nghĩa với lỗi schema.

## 5. Thiết kế mô hình quan hệ SQLite

### 5.1. Bảng `extraction_runs`

Mỗi lần chạy Node 1 có một `run_id`, model, prompt version, schema version, lane và thời gian. Bảng này tạo provenance để phân biệt kết quả Ollama PoC với các lần benchmark vLLM sau này. Một invoice có thể xuất hiện ở nhiều run nhưng khóa ghép `(run_id, sample_id)` không được trùng trong cùng run.

### 5.2. Bảng `invoices`

`invoices` lưu thông tin cấp hóa đơn và là bảng trung tâm. Các nhóm cột gồm:

- Định danh run: `invoice_id`, `run_id`, `schema_version`, `sample_id`.
- Nguồn audit: `img_id`, `image_path`, `raw_text`, `raw_json`.
- Seller/buyer: tên, địa chỉ, MST, phone.
- Metadata hóa đơn: số, series, ngày, giờ, currency, payment method.
- Tổng tiền: raw/value/currency cho subtotal, discount, VAT và total.
- Trạng thái: extraction status, confidence, missing fields, warnings.

`invoice_id` được tạo xác định theo `<run_id>:<sample_id>`. `raw_json` vẫn được giữ để audit nhưng Node 2 không truy vấn trường này trong luồng nghiệp vụ bình thường. Foreign key và cascade delete bảo đảm dữ liệu con bị xóa đúng khi một run bị xóa.

### 5.3. Bảng `invoice_items`

`invoice_items` có một dòng cho mỗi item của một invoice. Khóa ngoại `invoice_id` liên kết về `invoices`; ràng buộc `(invoice_id, line_number)` ngăn dòng hàng trùng thứ tự. Các cột quantity, unit price và line total đều giữ raw/value/currency để vừa audit vừa tổng hợp.

### 5.4. Bảng `invoice_taxes` và `field_evidence`

`invoice_taxes` tránh việc nhét nhiều mức thuế vào một cột của `invoices`. `field_evidence` lưu đường dẫn trường, đoạn OCR và source label nullable, cho phép truy ngược lý do Node 1 điền một giá trị. Hai bảng này giúp mô hình quan hệ giữ được dữ liệu lặp lại mà không phá vỡ cấu trúc chuẩn hóa.

### 5.5. Các view cho Node 2

Node 2 ưu tiên ba view:

- `v_invoice_facts`: một dòng trên mỗi invoice, phù hợp với tổng tiền, ngày, seller/buyer và số lượng item/tax.
- `v_item_facts`: một dòng trên mỗi item, phù hợp với câu hỏi về mặt hàng, quantity, unit price và line total.
- `v_tax_facts`: một dòng trên mỗi dòng thuế, phù hợp với thuế suất và tiền VAT.

Các view che giấu chi tiết join và tên cột audit không cần thiết. Nhờ đó prompt Text-to-SQL ngắn hơn, SQL ổn định hơn và dễ kiểm tra quyền chỉ đọc.

DDL chính thức nằm tại [schema.sql](../../../src/pipeline/sqlite/schema.sql), với `PRAGMA user_version=1`. Kiểm tra tương thích giữa JSON Schema và SQLite bằng:

```powershell
python scripts\check_invoice_contract.py
```

## 6. Danh mục câu hỏi nghiệp vụ cho Text-to-SQL

Node 2 nhận câu hỏi tiếng Việt, xác định view phù hợp và sinh một câu lệnh `SELECT`. Các câu hỏi mẫu dưới đây là hợp đồng workload ban đầu:

| ID | Câu hỏi | View ưu tiên | Ý nghĩa dữ liệu |
|---|---|---|---|
| Q1 | Tổng chi tiêu trong tháng 08/2020 là bao nhiêu? | `v_invoice_facts` | `SUM(total_amount_value)` theo khoảng `issued_date`. |
| Q2 | Mặt hàng nào được mua nhiều nhất? | `v_item_facts` | `SUM(quantity_value)` theo `description`. |
| Q3 | Cửa hàng nào có tổng giá trị hóa đơn cao nhất? | `v_invoice_facts` | `SUM(total_amount_value)` theo `seller_name`. |
| Q4 | Có bao nhiêu hóa đơn có phát sinh VAT? | `v_invoice_facts`/`v_tax_facts` | Đếm invoice có VAT hoặc dòng thuế. |
| Q5 | Tổng tiền theo từng ngày lập hóa đơn là bao nhiêu? | `v_invoice_facts` | `SUM(total_amount_value)` theo `issued_date`. |
| Q6 | Mức thuế suất và tổng tiền thuế theo từng loại thuế là bao nhiêu? | `v_tax_facts` | `SUM(tax_amount_value)` theo `tax_name`, `tax_rate`. |

Một số quy tắc an toàn của Node 2:

1. Chỉ chấp nhận câu lệnh bắt đầu bằng `SELECT` hoặc `WITH` kết thúc bằng truy vấn đọc.
2. Chỉ cho phép truy cập ba view nghiệp vụ trong danh mục, trừ khi audit được phê duyệt riêng.
3. Từ chối `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `ATTACH` và nhiều statement trong một request.
4. Trả về SQL, view đã chọn và giải thích ngắn để kết quả có thể audit.
5. Khi dữ liệu thiếu, trả về `null` hoặc thông báo không đủ dữ liệu, không bịa thêm item/MST/VAT.

Catalog chính thức được duy trì trong [src/pipeline/README.md](../../../src/pipeline/README.md) để Node 2, benchmark harness và tài liệu không dùng các bộ câu hỏi khác nhau.

## 7. Kết luận BA

Phần BA của M0-04 đã xác định cấu trúc nghiệp vụ hóa đơn bán lẻ/VAT, từ điển dữ liệu, quy tắc xử lý thiếu dữ liệu và các quan hệ cần thiết cho phân tích. JSON Schema `invoice_extraction.v1` tạo hợp đồng ổn định giữa LLM và lớp dữ liệu. SQLite `user_version=1` chuyển output JSON thành các bảng quan hệ có provenance, audit và view đọc phù hợp cho Text-to-SQL. Sáu câu hỏi nghiệp vụ cung cấp workload ban đầu cho Node 2 và bao phủ các nhóm tổng tiền, mặt hàng, cửa hàng, VAT, thời gian và thuế.

Các file nguồn liên quan:

- [Pipeline contract](../../../src/pipeline/README.md)
- [JSON Schema](../../../src/pipeline/schemas/invoice_extraction.schema.json)
- [SQLite DDL](../../../src/pipeline/sqlite/schema.sql)
- [SQLite notes](../../../src/pipeline/sqlite/README.md)
- [Node 1 prompt](../../../src/pipeline/prompts/node1_extractor_system.md)
