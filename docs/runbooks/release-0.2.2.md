# Phát hành FlexSDS 0.2.2

Version đã được cập nhật trong `pyproject.toml` và `src/flexsds/__init__.py`.
Bản này gồm sửa UTF-8 cho log Modal trên Windows, nhận diện speculative method,
kiểm tra checkpoint/vocabulary và chọn riêng nhãn EAGLE-1/2 khi weights dùng chung.
Nhận diện checkpoint không chứng minh đầy đủ thuật toán cây động EAGLE-2 hay
inference GPU thành công; xem `model-compatibility.md` trong cùng thư mục.

Các lệnh dưới đây chạy trong thư mục repo `do-an-ptit-d22` trên Linux.

## 1. Chuẩn bị công cụ

```bash
python3 -m venv .venv-release
.venv-release/bin/python -m pip install --upgrade build twine
```

Nếu `.venv-release` đã tồn tại thì dùng lại môi trường đó.

## 2. Kiểm tra và đóng gói

```bash
PYTHONPATH=src:. python3 -m pytest tests/test_flexsds_baseline.py::test_flexsds_version -q
.venv-release/bin/python -m build --outdir dist/0.2.2
.venv-release/bin/python -m twine check dist/0.2.2/*
```

Hai file cần phát hành:

- `dist/0.2.2/flexsds-0.2.2-py3-none-any.whl`
- `dist/0.2.2/flexsds-0.2.2.tar.gz`

Chỉ dùng thư mục `dist/0.2.2`, tránh upload nhầm các artifact version cũ.

## 3. Cài thử wheel trước khi upload

```bash
python3 -m venv /tmp/flexsds-022-check
/tmp/flexsds-022-check/bin/python -m pip install dist/0.2.2/flexsds-0.2.2-py3-none-any.whl
/tmp/flexsds-022-check/bin/flexsds --version
/tmp/flexsds-022-check/bin/flexsds inspect --method ngram
```

Version phải là `flexsds 0.2.2`. N-gram được nhận diện nhưng chưa được FlexSDS
serving hỗ trợ. Lệnh inspect này không cấp GPU.

## 4. Upload lên PyPI

Dùng API token của project `flexsds` tại PyPI. Không lưu token vào repo hoặc
đưa token trực tiếp vào câu lệnh có lịch sử shell.

```bash
.venv-release/bin/python -m twine upload --username __token__ dist/0.2.2/*
```

Nhập API token khi Twine hỏi password, hoặc dùng credential đã lưu trong keyring.
Bước upload là bước xuất bản công khai. Gói hiện mới được chuẩn bị tại máy;
Codex chưa upload lên PyPI. PyPI không cho ghi đè cùng tên file đã phát hành;
nếu version 0.2.2 đã tồn tại, kiểm tra bản đó trước khi quyết định version tiếp theo.

## 5. Nâng cấp sau khi đã upload

Linux:

```bash
python3 -m pip install --upgrade "flexsds==0.2.2"
flexsds --version
```

Windows CMD hoặc PowerShell:

```powershell
py -m pip install --upgrade "flexsds==0.2.2"
flexsds --version
```

Nếu đang dùng pipx:

```bash
pipx install --force "flexsds==0.2.2"
flexsds --version
```

Khi CLI vẫn báo version cũ, dùng `which flexsds` trên Linux hoặc `where flexsds`
trong Windows CMD (`where.exe flexsds` trong PowerShell) để kiểm tra launcher
đang chạy thuộc môi trường Python nào.

Nguồn: https://packaging.python.org/en/latest/tutorials/packaging-projects/
