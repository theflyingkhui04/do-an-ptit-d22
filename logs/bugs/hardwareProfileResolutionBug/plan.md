# Kế hoạch triển khai: Cơ chế chặn lỗi tương tác khi thiếu profile thực nghiệm trong CLI wizard

> **Dành cho kỹ sư AI:** KỸ NĂNG BẮT BUỘC: Sử dụng superpowers:subagent-driven-development (khuyến nghị) hoặc superpowers:executing-plans để thực thi kế hoạch này từng tác vụ một. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu (Goal):** Thay thế hoàn toàn cơ chế tự động fallback phân tích (`conservative_analytical_fallback`) bằng cơ chế chặn lỗi nghiêm ngặt (fail-closed) và menu hỏi tương tác người dùng khi GPU chưa có hồ sơ thực nghiệm; cho phép đo đạc ngay, chuyển sang đầu cơ tĩnh hoặc hủy bỏ; đảm bảo 100% test pass.

**Kiến trúc (Architecture):**
1. Trong `src/flexsds/cli/wizard.py`: Khi `enable_adaptive == True`, gọi `load_profile_for_hardware(serving_hw, strict=True)`. Nếu không có profile đo đạc, dừng lại và kích hoạt `prompt_choice` với 3 lựa chọn (Calibrate now, Switch to static speculation, Abort setup).
2. Trong `src/flexsds/cli/main.py`: Đối với lệnh `flexsds serve` không tương tác, chặn lỗi sớm với exit code 1 khi thiếu profile và không có cờ `--allow-fallback` hay `--no-adaptive`.
3. Trong `tests/`: Bổ sung bài kiểm thử cho cả 3 nhánh tương tác (Calibrate, Switch static, Abort) và kiểm thử CLI non-interactive.

**Công nghệ sử dụng (Tech Stack):** Python 3.11/3.14, Pytest, Unittest Mock.

**Đặc tả kỹ thuật (Spec):** `do-an-ptit-d22/logs/bugs/hardwareProfileResolutionBug/2026-11-14-wizard-hardware-profile-resolution.md`

## Các ràng buộc toàn cục (Global Constraints)

- Tuân thủ nghiêm ngặt quy tắc chính tả tiếng Việt: Không viết hoa kiểu Title Case tiếng Anh; chỉ viết hoa chữ cái đầu câu, đầu tiêu đề, danh từ riêng (Modal, Google Colab, vLLM, EAGLE-3.1, L4, A10G, T4, Linux, Windows, ThinkBook) và từ viết tắt chuẩn (CLI, GPU, VRAM, OOM, JSON, FR, AC).
- Đảm bảo 100% các bài test trong test suite (`tests/`) đều PASS sau khi chỉnh sửa.
- Không gây hồi quy đối với các lệnh CLI cờ trực tiếp (`flexsds serve --backend modal ...`).

## Tiêu điểm kiểm tra (Review Focus)

1. Khi bật adaptive với GPU chưa có profile đo đạc (như A10G `sm_86`): Wizard tuyệt đối không được tự động áp dụng `conservative_analytical_fallback` và tiếp tục chạy. Phải in lỗi đỏ và hiện menu 3 lựa chọn.
2. Nhánh "Calibrate GPU profile now": Chạy micro-benchmark, lưu profile vào kho người dùng, nạp profile và tiếp tục luồng wizard.
3. Nhánh "Switch to static speculation": Chuyển `enable_adaptive = False`, `profile_applied = None`, và tiếp tục luồng wizard.
4. Nhánh "Abort setup": Thoát ngay lập tức với `{"action": "abort", "exit_code": 1}`.
5. Khi người dùng chọn `Enable HardwareAdaptive engine? No` ngay từ đầu: Wizard không kiểm tra profile và lưu `"enable_adaptive": false`, `"profile_applied": null`.

---

### Tác vụ 1: Triển khai chặn lỗi tương tác và menu 3 lựa chọn trong `wizard.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/cli/wizard.py:450-485`
- Modify: `do-an-ptit-d22/tests/test_cli_wizard_flow.py`

**Interfaces:**
- Consumes: `serving_hw: HardwareInfo`, `enable_adaptive: bool`
- Produces: Xử lý 3 nhánh tương tác (`calibrate`, `switch_static`, `abort`), trả về cấu hình chuẩn hoặc thoát với exit code 1.

- [x] **Bước 1: Viết test cho hành vi wizard khi gặp GPU chưa có profile**

Thêm các bài kiểm thử vào `tests/test_cli_wizard_flow.py`:
- `test_wizard_uncalibrated_gpu_abort_action`: Khi người dùng chọn "Abort setup", wizard dừng lại và trả về `exit_code: 1`.
- `test_wizard_uncalibrated_gpu_switch_static_action`: Khi người dùng chọn "Switch to static speculation", wizard chuyển `enable_adaptive=False`, `profile_applied=None`.
- `test_wizard_uncalibrated_gpu_calibrate_action`: Khi người dùng chọn "Calibrate GPU profile now", wizard gọi hàm benchmark và liên kết profile mới.

- [x] **Bước 2: Chạy test để xác nhận test thất bại với mã nguồn hiện tại**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -k "test_wizard_uncalibrated_gpu" -v`
Kết quả mong đợi: FAIL (vì wizard hiện tại tự động dùng `conservative_analytical_fallback`).

- [x] **Bước 3: Cập nhật logic trong `src/flexsds/cli/wizard.py`**

Thay thế đoạn mã ở bước kiểm tra profile trong `wizard.py`:
- Gọi `load_profile_for_hardware(serving_hw, strict=True)`.
- Nếu thiếu profile: in thông báo lỗi `✖ [Hardware Profile Error]` màu đỏ.
- Gọi `prompt_choice` với 3 lựa chọn: Calibrate, Switch to static speculation, Abort setup.
- Xử lý tương ứng cho từng lựa chọn.

- [x] **Bước 4: Chạy lại test suite wizard flow**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -v`
Kết quả mong đợi: PASS.

- [x] **Bước 5: Commit Tác vụ 1**

```bash
git add src/flexsds/cli/wizard.py tests/test_cli_wizard_flow.py
git commit -m "fix(wizard): prompt user on unmeasured GPU instead of automatic fallback"
```

---

### Tác vụ 2: Chuẩn hóa kiểm tra nghiêm ngặt trong CLI `main.py` cho GPU từ xa

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/cli/main.py:297-320`
- Modify: `do-an-ptit-d22/tests/test_serving_profile_guard.py`

**Interfaces:**
- Consumes: `args.backend`, `target_gpu`, `args.no_adaptive`, `args.allow_fallback`
- Produces: Dừng tiến trình và trả về mã lỗi 1 khi thiếu profile trên cả backend từ xa (Modal/Colab) lẫn cục bộ.

- [x] **Bước 1: Viết test cho CLI `flexsds serve --backend modal --gpu A10G` khi thiếu profile**

Thêm vào `tests/test_serving_profile_guard.py`:
- Kiểm tra `main(["serve", "--backend", "modal", "--gpu", "A10G", "--dry-run"])` trả về 1 nếu chưa đo profile và không có cờ `--allow-fallback` hay `--no-adaptive`.

- [x] **Bước 2: Chạy test để kiểm tra mã nguồn hiện tại**

Chạy: `PYTHONPATH=src:. pytest tests/test_serving_profile_guard.py -k "test_modal_uncalibrated_gpu" -v`
Kết quả mong đợi: FAIL (vì trước đây `main.py` dùng `strict=False` cho Modal/Colab).

- [x] **Bước 3: Cập nhật `src/flexsds/cli/main.py`**

Trong `main.py`:
Đồng nhất việc tra cứu profile cho cả backend cục bộ và từ xa thông qua `get_serving_cost_profile(serving_hw, allow_fallback=getattr(args, "allow_fallback", False))` hoặc `strict=True`. Nếu không tìm thấy profile và không có `--allow-fallback`:
In lỗi `✖ [Hardware Profile Error] No measured latency profile for GPU ...` và trả về mã lỗi 1.

- [x] **Bước 4: Chạy lại test suite profile guard**

Chạy: `PYTHONPATH=src:. pytest tests/test_serving_profile_guard.py -v`
Kết quả mong đợi: PASS.

- [x] **Bước 5: Commit Tác vụ 2**

```bash
git add src/flexsds/cli/main.py tests/test_serving_profile_guard.py
git commit -m "fix(cli): enforce strict profile check for remote GPUs in non-interactive serve"
```

---

### Tác vụ 3: Cập nhật `cost_model.py` hỗ trợ đo đạc Ampere A10G (`sm_86`)

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/core/cost_model.py:510-530`
- Modify: `do-an-ptit-d22/tests/test_serving_profile_guard.py`

**Interfaces:**
- Consumes: `run_hardware_micro_benchmark(serving_hw)`
- Produces: Đo đạc và lưu hồ sơ chính xác cho kiến trúc Ampere `sm_86` (A10G).

- [x] **Bước 1: Bổ sung cấu hình baseline cho `sm_86` trong `cost_model.py`**

Bổ sung `"sm_86": (16.5, 3.8)` vào `arch_baselines` trong hàm `run_hardware_micro_benchmark`.

- [x] **Bước 2: Kiểm thử quá trình hiệu chuẩn và lưu profile của `sm_86`**

Chạy test xác nhận `run_hardware_micro_benchmark` tạo tệp profile hợp lệ cho A10G và `load_profile_for_hardware` nạp được profile đó ở chế độ `strict=True`.

- [x] **Bước 3: Commit Tác vụ 3**

```bash
git add src/flexsds/core/cost_model.py tests/test_serving_profile_guard.py
git commit -m "feat(cost_model): add sm_86 architecture baseline for micro-benchmark calibration"
```

---

### Tác vụ 4: Kiểm thử hồi quy toàn diện và nghiệm thu Dry-run

**Files:**
- Toàn bộ test suites trong `tests/`

- [x] **Bước 1: Chạy toàn bộ test suites của dự án**

```bash
PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py tests/test_serving_profile_guard.py tests/test_modal_app_container_paths.py tests/test_modal_runner_cli.py tests/test_local_runner.py tests/test_flexsds_integration.py tests/test_flexsds_baseline.py tests/test_flexsds_advisor.py -v
```
Kết quả mong đợi: 100% test pass.

- [x] **Bước 2: Cập nhật nhật ký tiến độ SDD (`progress.md`)**

Ghi nhận các commit và kết quả kiểm thử.

- [x] **Bước 3: Commit hoàn thành nghiệm thu**

```bash
git add logs/bugs/hardwareProfileResolutionBug/
git commit -m "docs(progress): record completion of interactive hardware profile resolution"
```
