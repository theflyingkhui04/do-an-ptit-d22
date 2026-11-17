# Windows Modal serving: JSONDecodeError khi khởi động

## Bằng chứng

- `modal-serving-_yib0uly.log`: container import `/root/modal_app.py`, đọc
  `FLEXSDS_SERVING_CONFIG`, rồi lỗi `Invalid \escape: line 1 column 401`.
  Lỗi xảy ra trước khi load profile, PyTorch hoặc model.
- `modal-serving-68l8i8ws.log`: dừng ở `App deployed`; không có log container
  production. Không thể khẳng định nguyên nhân runtime chỉ từ file này.
  Image của lần deploy cũng chứa cấu hình có đường dẫn Windows nên có cùng
  nguy cơ lỗi escape như lần development.

## Nguyên nhân và bản sửa

`setup_environment()` dùng `json.dumps()` để truyền cấu hình vào subprocess,
đúng ở bước local. Nhưng `modal_app.py` tiếp tục đưa JSON này vào `Image.env()`.
Modal SDK 1.6.3 tạo Dockerfile `ENV` từ giá trị đó; đường dẫn Windows có nhiều
dấu backslash có thể mất một lớp escape khi build image. `json.loads()` trong
container sau đó gặp các chuỗi như `\U` không phải escape JSON hợp lệ.

Bản sửa truyền JSON vào image bằng base64 trong `FLEXSDS_SERVING_CONFIG_B64`,
giải mã trước khi parse JSON. Đường dẫn profile trong cấu hình image được đổi
thành `/root/project/hardware_profile.json`. Đường dẫn local vẫn được giữ để
upload đúng file profile. CLI cũng nhận diện `JSONDecodeError` trong bộ thu lỗi.

Test hồi quy thực thi module ở cả hai bước dựng app local và import container
bằng Modal giả lập; kiểm tra đường dẫn Windows, Unicode và upload profile.
Test này không chứng minh inference GPU thành công.

## Cài bản sửa trên Windows

Trong thư mục repo `do-an-ptit-d22` chứa bản sửa:

```powershell
py -m pip install --force-reinstall --no-deps .
flexsds
```

Phải dùng đúng Python đã cài launcher `flexsds`; kiểm tra `where.exe flexsds`
nếu có nhiều Python. Chỉ nâng cấp bản PyPI hiện tại không lấy được bản sửa
chưa phát hành này. Chạy lại wizard để deploy/serve image mới.

`App deployed` chỉ xác nhận publish thành công. Container web server chạy
on-demand khi có HTTP request; CLI tiếp tục kiểm tra `/health`, `/v1/models`
và completion thực tế, với timeout tổng 900 giây. Nếu vẫn lỗi ở production,
lấy log runtime qua `modal app logs flexsds-serving` hoặc dashboard Modal.

Tham khảo: https://modal.com/docs/guide/webhooks#non-asgi-web-servers
