# Future work: Registry chia sẻ GPU profile

Ngày ghi nhận: 2026-11-16.

Trạng thái: Đề xuất cho tương lai, chưa triển khai registry hoặc tự động upload.

## Mục tiêu

Cho phép profile do một người dùng đo được chia sẻ và tái sử dụng bởi người dùng
khác có cấu hình tương thích, giảm thời gian và chi phí đo lặp lại.

Profile mới đo không tự trở thành profile mặc định cho tất cả mọi người.
Việc phân phối cần có bước xác thực và kiểm tra tương thích.

## Phạm vi hiện tại

- Profile tự đo được lưu local tại `~/.flexsds/profiles/`.
- Có thể chia sẻ file JSON thủ công; lệnh export/import chuyên dụng là hướng
  bổ sung, chưa được coi là tính năng đã triển khai.
- Profile đã kiểm tra có thể được đưa vào `src/flexsds/profiles/` và phân phối
  cùng bản phát hành package, như profile A10G hiện tại.
- Chưa cần vận hành backend registry riêng trong phạm vi hiện tại của đồ án.

## Thiết kế dự kiến

1. Người dùng chạy Fast hoặc Full measure; kết quả vẫn lưu local trước.
2. Người dùng chủ động chọn gửi profile lên registry, kèm thông tin nguồn gốc
   và các mẫu đo cần thiết để xác thực. Không tự động upload mặc định.
3. Registry kiểm tra schema, dữ liệu đo, tính toàn vẹn và cấu hình; profile chờ
   duyệt được tách khỏi profile đã xác thực.
4. Client tra cứu profile đã xác thực, kiểm tra tương thích rồi mới tải và dùng.
5. Profile có phiên bản để cập nhật, thu hồi hoặc quay lại bản trước khi cần.

## Điều kiện tương thích

Không dùng tên GPU làm khóa duy nhất. Cần xét ít nhất:

- Model GPU, kiến trúc SM, số GPU và cấu hình bộ nhớ liên quan.
- Môi trường đo: Modal, Colab hoặc local; cấu hình CPU/RAM khi có ảnh hưởng.
- Target/draft model và revision, speculative method.
- Quantization, dtype, engine và phiên bản runtime liên quan.
- Cấu hình engine, phạm vi batch/K/context thực sự đã đo.
- Chế độ Fast/Full, số lần warmup/lặp và chất lượng kết quả đo.

T4 Modal và T4 Colab thuộc cùng nhóm phần cứng nhưng nên có các biến thể
profile theo môi trường. Không mặc định coi hai kết quả đo là thay thế được
cho nhau; chỉ tái sử dụng khi có cơ sở kiểm tra tương thích hoặc đo xác nhận.

## Xác thực và vận hành

- Giữ provenance, thời điểm đo và bằng chứng đo để tái kiểm tra kết quả.
- Phát hiện dữ liệu không đầy đủ, sai cấu hình, nhiễu hoặc bất thường;
  kiểm tra tính toàn vẹn file không đủ để chứng minh phép đo đáng tin cậy.
- Có quy trình duyệt và quyền publish; giới hạn upload để tránh spam.
- Kiểm tra dữ liệu gửi lên, tránh đưa token, đường dẫn riêng hoặc thông tin
  nhạy cảm vào artifact công khai.
- Không dùng profile Fast để âm thầm ghi đè profile Full đã xác thực.
- Khi registry không khả dụng hoặc không có profile phù hợp, tiếp tục cho
  phép dùng profile local hoặc đo mới.

## Lộ trình đề xuất

1. Bổ sung export/import JSON với kiểm tra tương thích và bằng chứng đo.
2. Duy trì bộ profile được kiểm tra thủ công, phân phối qua package release.
3. Xây registry có upload chủ động, duyệt profile và download theo cấu hình.
4. Đánh giá kiểm chứng tự động bằng nhiều phép đo độc lập trước khi mở rộng
   phân phối profile cộng đồng.

## Tiêu chí hoàn thành registry

- Người dùng khác tìm và dùng được profile đã duyệt khi cấu hình phù hợp.
- Profile không tương thích hoặc không đạt kiểm tra không được tự áp dụng.
- Có thể truy nguyên, thu hồi và quay lại phiên bản profile trước.
- Upload cần lựa chọn chủ động; lỗi mạng không làm mất profile local.
