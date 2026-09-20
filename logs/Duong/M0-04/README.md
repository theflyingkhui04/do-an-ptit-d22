# M0-04 - Hồ sơ nghiệm thu

## Phạm vi

[M0-04] Phân tích nghiệp vụ hóa đơn VAT, thiết kế Schema và lập trình xử lý
MC-OCR. Hồ sơ này đối chiếu trực tiếp với ba tiêu chí Definition of Done của
thẻ Trello.

## Bằng chứng hoàn thành

### 1. Mô tả nghiệp vụ và JSON Schema

- Mô tả nghiệp vụ và contract pipeline: [src/pipeline/README.md](../../../src/pipeline/README.md).
- JSON Schema Node 1: [invoice_extraction.schema.json](../../../src/pipeline/schemas/invoice_extraction.schema.json).
- Ví dụ input: [node1_extractor_input_m0_04_001.txt](../../../src/pipeline/examples/node1_extractor_input_m0_04_001.txt).
- Ví dụ output: [node1_extractor_output_m0_04_001.json](../../../src/pipeline/examples/node1_extractor_output_m0_04_001.json).
- SQLite contract: [schema.sql](../../../src/pipeline/sqlite/schema.sql).

Kiểm tra đã chạy:

```powershell
python scripts\check_invoice_contract.py
python scripts\validate_invoice_schema.py src\pipeline\examples\node1_extractor_output_m0_04_001.json
```

### 2. Dev-set MC-OCR

Dev-set đã làm sạch gồm 30 mẫu, vượt yêu cầu tối thiểu 20 mẫu. Dữ liệu nằm
trong `data/processed/m0_04_mcocr_devset/`, gồm manifest, JSONL input, OCR text
và ảnh audit. Nhãn annotation không được đưa vào prompt Node 1.

### 3. Bản thảo Chương 1

Bản thảo hoàn chỉnh nằm tại
[chuong-1-gioi-thieu.md](chuong-1-gioi-thieu.md). Nội dung gồm 3.888 từ,
11 mục chính, và phần quy cách dàn trang A4 theo yêu cầu bản thảo PTIT. Bản
Word chỉnh sửa được nằm tại [chuong-1-gioi-thieu.docx](chuong-1-gioi-thieu.docx).
Khi
đưa vào template chính thức của khoa, giữ nguyên nội dung và áp dụng lề,
font, giãn dòng, đánh số chương/mục và số trang theo template đó.

## Trạng thái nghiệm thu

- [x] Bản mô tả nghiệp vụ, JSON Schema và ví dụ input/output.
- [x] Dev-set tối thiểu 20 hóa đơn sạch.
- [x] Bản nháp Chương 1 đủ nội dung 5–7 trang sau dàn trang.

Các kết quả Ollama local là functional PoC; benchmark hiệu năng chính thức
vẫn phải chạy trên vLLM theo kế hoạch thực nghiệm.