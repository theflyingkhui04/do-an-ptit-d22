# src/ — Mã nguồn

Nơi chứa mã nguồn hệ thống (sẽ hiện thực sau giai đoạn thiết kế). Dự kiến:

- **Máy phục vụ tối ưu** — endpoint tương thích OpenAI (`/v1/chat/completions`), hai làn *vanilla* / *optimized*.
- **Module nháp thích ứng phần cứng** — điều chỉnh chiến lược cây nháp theo hồ sơ GPU đích (xem twist ở [../docs/03-quyet-dinh-thiet-ke.md](../docs/03-quyet-dinh-thiet-ke.md)).
- **Tiện ích profiling** — đo băng thông/độ trễ GPU để nạp vào cost-model chọn cây.
- **Pipeline MC-OCR** — workload đo end-to-end gồm Node 1 Extractor, Node 2 Text-to-SQL và Node 3 SQL-to-BI. Prompt/schema chuẩn của Node 1 nằm ở [pipeline/prompts/node1_extractor_system.md](pipeline/prompts/node1_extractor_system.md) và [pipeline/schemas/invoice_extraction.schema.json](pipeline/schemas/invoice_extraction.schema.json); lược đồ SQLite cho Node 2 nằm ở [pipeline/sqlite/schema.sql](pipeline/sqlite/schema.sql).

*Phần pipeline đang được hiện thực theo hợp đồng v1 tại [pipeline/README.md](pipeline/README.md); các module serving/proposer tiếp tục theo kế hoạch chung.*
