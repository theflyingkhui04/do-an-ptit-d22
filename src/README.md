# src/ — Mã nguồn

Nơi chứa mã nguồn hệ thống (sẽ hiện thực sau giai đoạn thiết kế). Dự kiến:

- **Máy phục vụ tối ưu** — endpoint tương thích OpenAI (`/v1/chat/completions`), hai làn *vanilla* / *optimized*.
- **Module nháp thích ứng phần cứng** — điều chỉnh chiến lược cây nháp theo hồ sơ GPU đích (xem twist ở [../docs/03-quyet-dinh-thiet-ke.md](../docs/03-quyet-dinh-thiet-ke.md)).
- **Tiện ích profiling** — đo băng thông/độ trễ GPU để nạp vào cost-model chọn cây.

*Chưa hiện thực — đang ở giai đoạn thiết kế/đề cương.*
