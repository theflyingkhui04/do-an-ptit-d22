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


## GPU profiles và chế độ đo

Menu 3 là `Measure & available profile GPU`: hiển thị profile đóng gói và profile
trong `~/.flexsds/profiles`, chọn nền tảng/GPU và model target/draft, sau đó dùng
profile phù hợp đã có hoặc đo lại. Khi serving thiếu profile, wizard cũng cho
chọn `Fast measure` hoặc `Full measure` trước khi đo.

- Fast: 2 lần warmup + 5 lần đo cho mỗi ô batch/K.
- Full: 5 lần warmup + 50 lần đo cho mỗi ô batch/K (hành vi CLI mặc định).
- Cả hai giữ đủ batch `[1, 2, 4, 8, 16, 32]`, K `[0, 1, 2, 3, 4]`, context 1024;
  xuất JSON kèm mẫu đo gốc và chế độ đo. Tên file riêng cho Fast/Full, tránh
  ghi đè Full khi chạy Fast. Fast vẫn cần tải model cho từng K.

```bash
flexsds profile --benchmark --backend modal --gpu A10G --measure-mode fast
flexsds profile --benchmark --backend local --measure-mode full
```

Profile A10G đã đo được đóng gói ở
`src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`, cùng wheel
nhờ cấu hình package-data hiện có. Profile này dành cho Qwen2.5-7B-Instruct-AWQ
và `duckhoi/qwen2.5-7b-eagle3.1-speculator`; không tự áp dụng cho model khác.
T4/L4 cũ vẫn được liệt kê nhưng chưa đạt kiểm tra worker evidence cho adaptive
serving. Profile tự đo lưu tại máy chạy CLI, không tự đồng bộ sang máy khác.


## Cảnh báo VRAM và giới hạn serving

Wizard hỏi `Maximum concurrent sequences (batch)` và `Maximum context length`
trước compatibility check. Phép tính KV cache dùng chính hai giá trị này;
chúng được lưu trong `serving_config.json`, truyền sang vLLM local/Modal và
đồng bộ vào `MAX_NUM_SEQS`/`MAX_MODEL_LEN` trong notebook Colab được tạo.

Khi ước lượng vượt 95% budget VRAM, wizard cho chọn giảm batch/context rồi
kiểm tra lại, tiếp tục với cấu hình hiện tại hoặc hủy. Đây là ước lượng cho
phạm vi cấu hình, không chứng minh workload nhỏ sẽ OOM hoặc chắc chắn chạy được.
Không dùng mốc 100% làm bảo đảm an toàn. Lỗi runtime thực tế vẫn được báo qua
luồng startup/readiness hiện có.

CLI yêu cầu xác nhận rõ bằng `--allow-oom-risk` nếu muốn tiếp tục khi ước lượng
VRAM vượt ngưỡng (kể cả preview). Ví dụ:

```bash
flexsds serve --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --backend modal --gpu T4 --max-num-seqs 4 --max-model-len 2048 \
  --no-adaptive --allow-oom-risk --dry-run
```

Cờ này chỉ bỏ chặn do ước lượng VRAM. Pairing/checkpoint/tokenizer không hợp lệ,
method chưa hỗ trợ và thiếu CUDA khi chạy local vẫn chặn, kể cả có cờ này.
CLI Colab in các giá trị cần đặt trong notebook; wizard đồng bộ file notebook
local, cần upload đúng file đã đồng bộ thay vì dùng bản trên URL repository.
