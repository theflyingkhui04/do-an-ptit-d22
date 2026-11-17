# Kịch bản sử dụng chuẩn của công cụ dòng lệnh FlexSDS CLI (Happy Path Runbook)

> **Tài liệu tham chiếu:** Chi tiết kịch bản Happy Path đầy đủ được lưu tại [`logs/Khoi/spec&plan/flexsds-cli/happy-path.md`](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/logs/Khoi/spec%26plan/flexsds-cli/happy-path.md).  
> **Dự án:** Đồ án tốt nghiệp đại học — Ngành Công nghệ Thông tin, PTIT  
> **Ngày lập:** 11/11/2026  

Vui lòng xem chi tiết tài liệu kịch bản tại đường dẫn trên.

## Modal onboarding

- Menu đầu có `Delete Modal token`: xóa credentials Modal local và token trong
  active native profile; không thu hồi token trên website Modal.
- Khi đăng nhập hoặc chạy Modal mà chưa cài SDK/CLI, FlexSDS tự chạy
  `python -m pip install modal` bằng Python đang chạy FlexSDS. Nếu launcher
  chưa nằm trong PATH, FlexSDS dùng `python -m modal`.
- Wizard và `flexsds auth login --provider modal` in hướng dẫn tiếng Anh từng
  bước: đăng nhập modal.com, chọn workspace, vào Settings > API Tokens, tạo
  token rồi copy Token ID và Token Secret vào CLI.
- CLI kiểm tra bằng `modal token info` với chính cặp token vừa nhập, hiển thị
  `Modal workspace: <name> (<id>)`, rồi mới lưu credentials. Token không hợp lệ
  hoặc lỗi kết nối không ghi đè credentials cũ; thử lại sau khi kiểm tra key.

Tham khảo: [Modal token CLI](https://frontend.modal.com/docs/cli/latest/token).

## Running endpoint

Chọn `Running endpoint` từ menu chính để lấy danh sách web endpoint đã deploy
trong workspace/environment Modal hiện tại. Deployment có 0 active tasks vẫn
được hiển thị; app development, app đã dừng và function batch không nằm trong
danh sách này. `Refresh list` tải lại dữ liệu, `Back to main menu` quay lại menu.

Chọn một endpoint để xem URL, App ID, Function ID, GPU, trạng thái và thời điểm
deploy. `Chat with endpoint` mở chat và tự lấy model qua API. Endpoint có xác
thực riêng cần cấu hình quyền truy cập phù hợp. `Stop deployment` yêu cầu xác
nhận app cụ thể, rồi dừng bằng App ID; thao tác này dừng toàn bộ endpoint trong
cùng app. Dừng xong quay lại danh sách cập nhật. Chỉ xem thông tin không gửi
request inference và không chủ động đánh thức GPU.

Tham khảo: [Modal app CLI](https://frontend.modal.com/docs/cli/latest/app).

## Thứ tự cấu hình Colab

Wizard hỏi bật/tắt HardwareAdaptive và xử lý profile trước khi cập nhật notebook.
Nếu thiếu profile và chọn chuyển sang static, notebook cũng nhận
`ENABLE_ADAPTIVE = False`. Sau khi chốt cấu hình, CLI đồng bộ notebook local,
rồi mới hiện hướng dẫn upload, mở thư mục và hỏi tunnel URL. Khi không ghi được
notebook, wizard báo lỗi và dừng để tránh upload file cấu hình cũ.

Upload file ở đường dẫn local CLI vừa in. Link Colab từ repository chỉ mở bản
template, không mang theo các lựa chọn vừa nhập trong wizard.

## Measure now và Import hardware profile

Trong `Measure & available profile GPU`, danh sách profile hiện có được giữ
nguyên; chọn một trong hai chức năng:

- `Measure now`: chọn nền tảng/GPU, model và Fast/Full để đo như trước.
- `Import hardware profile`: chọn dán JSON hoặc nhập đường dẫn file/link JSON
  trực tiếp qua HTTP(S). Với màn hình dán, Ctrl+X gửi nội dung để kiểm tra và lưu;
  Esc hoặc Ctrl+C hủy. File/link phải trả nội dung JSON, không phải trang xem file.

Import thành công lưu artifact JSON trong `~/.flexsds/profiles/` và metadata
vào SQLite `~/.flexsds/profiles/profiles.sqlite3`, rồi quay lại menu chính.
Nội dung trùng nhau chỉ có một artifact/bản ghi. Serving tiếp tục đọc artifact
JSON theo cơ chế hiện có nên không cần chuyển đổi dữ liệu.

Import kiểm tra JSON, metadata GPU/model và độ hợp lệ của chi phí. Profile
`flexsds-component-v2` phải vượt kiểm tra worker evidence. File legacy được lưu
và hiển thị là legacy; import không bỏ qua kiểm tra adaptive hoặc tự đổi nhãn
legacy thành measured. Nội dung JSON lỗi không được lưu.
