# Đặc tả kỹ thuật: Chuẩn hóa phân giải hồ sơ phần cứng và cơ chế chặn lỗi tương tác khi thiếu profile thực nghiệm trong CLI wizard

- **Mã tài liệu**: SPEC-PROFILE-20261114-REV2
- **Ngày cập nhật**: 2026-11-14
- **Tác giả**: Đội ngũ phát triển FlexSDS (D22 PTIT)
- **Trạng thái**: Đã phê duyệt và hoàn thành (Approved and implemented)
- **Phạm vi tác động**: `src/flexsds/cli/wizard.py`, `src/flexsds/core/cost_model.py`, `src/flexsds/cli/main.py`, `tests/test_cli_wizard_flow.py`, `tests/test_serving_profile_guard.py`

---

## 1. Bối cảnh và mục đích

Trong phiên làm việc với wizard cấu hình phục vụ (`flexsds serve` hoặc wizard tương tác), khi người dùng lựa chọn nền tảng đám mây Modal với GPU **A10G (24 GB VRAM)**, hệ thống trước đây từng gặp hai vấn đề nghiêm trọng:
1. Lấy nhầm hồ sơ của GPU L4 (`profile_sm89_l4.json`) do đọc nhầm cấu hình CPU của máy laptop thay vì GPU đích từ xa.
2. Khi khắc phục việc lấy nhầm thông số, hệ thống lại tự động áp dụng hồ sơ phân tích dè dặt (`conservative_analytical_fallback`) và tiếp tục khởi chạy server mà không có sự đồng ý rõ ràng của người dùng.

Việc tự ý áp đặt thông số độ trễ giả lập hoặc mượn thông số của phần cứng khác là phi khoa học, làm sai lệch kết quả đánh giá thực nghiệm của đề tài tốt nghiệp. Bản chất của engine HardwareAdaptive là dựa trên hồ sơ đo đạc thực nghiệm chính xác để tối ưu hóa kích thước bước đầu cơ $k$. Khi phần cứng chưa từng được đo kiểm độ trễ, việc cố tình chạy giải thuật thích ứng với số liệu ước đoán sẽ làm mất tính tối ưu và thiếu trung thực về mặt dữ liệu.

Mục đích của tài liệu đặc tả này là hoàn thiện cơ chế phân giải hồ sơ phần cứng theo nguyên tắc **chặn lỗi nghiêm ngặt (fail-closed)** và **hỏi ý kiến người dùng tương tác**:
1. Phân giải đúng GPU mục tiêu (phân biệt giữa GPU máy trạm cục bộ và GPU đám mây Modal/Colab).
2. Tách biệt hoàn toàn trạng thái bật/tắt của engine HardwareAdaptive: Khi tắt adaptive, không tra cứu, không nạp hay liên kết bất kỳ profile phần cứng nào (`profile_applied = None`).
3. Áp dụng chính sách chặn lỗi phổ quát cho **mọi phần cứng chưa có hồ sơ thực nghiệm** (không chỉ riêng A10G `sm_86`, mà cho mọi GPU mới/tùy biến): Tuyệt đối không tự động dùng fallback phân tích. Dừng ngay quy trình và hiển thị menu tương tác để người dùng quyết định:
   - Phương án 1: Đo kiểm và hiệu chuẩn profile ngay lập tức (`flexsds profile --benchmark` / `run_hardware_micro_benchmark`), sau đó nạp profile vừa đo để tiếp tục phục vụ.
   - Phương án 2: Chuyển sang chế độ đầu cơ tĩnh (Static speculation mode, $k=4$, tắt HardwareAdaptive) để phục vụ ngay mà không cần đo profile.
   - Phương án 3: Hủy bỏ quy trình thiết lập phục vụ (Abort) và thoát an toàn.

---

## 2. Phân tích nguyên nhân gốc rễ

### 2.1. Tra cứu nhầm phần cứng máy trạm thay vì GPU đích
- Biến `hw_info` ban đầu đại diện cho máy trạm nơi gõ lệnh CLI (ví dụ laptop ThinkBook chỉ có CPU, kiến trúc `sm_architecture = "cpu"`).
- Khi chọn phục vụ trên Modal với GPU A10G (`target_gpu = "A10G"`), nếu không tạo đối tượng phần cứng đại diện cho GPU đám mây, hệ thống sẽ tra cứu thông số của CPU laptop.

### 2.2. Cơ chế tự động fallback phân tích gây che giấu lỗi
- Trong `load_profile_for_hardware`, khi tham số `strict=False`, nếu không tìm thấy tệp profile thì hàm tự động trả về `(fallback_model, "conservative_analytical_fallback")`.
- Khi wizard gọi `load_profile_for_hardware(serving_hw, strict=False)`, hệ thống chỉ in một dòng cảnh báo màu vàng và tiếp tục tạo file cấu hình rồi hỏi khởi chạy server, biến một trạng thái thiếu dữ liệu thực nghiệm nghiêm trọng thành một bước chạy ngầm với dữ liệu giả lập.

### 2.3. Thiếu cơ chế hỏi ý kiến người dùng khi thiếu profile
- Người dùng không có cơ hội lựa chọn thực thi việc đo đạc ngay tại thời điểm cấu hình hoặc chuyển sang chế độ đầu cơ tĩnh.
- Nếu không bật HardwareAdaptive, hệ thống hoàn toàn không cần tới profile độ trễ. Do đó người dùng cần có lựa chọn tắt adaptive ngay tại chỗ để tiếp tục phục vụ nếu không muốn đo profile.

---

## 3. Yêu cầu chức năng (Functional Requirements)

### 3.1. FR-01: Tách biệt hoàn toàn khi tắt HardwareAdaptive
- Khi người dùng chọn `Enable HardwareAdaptive engine? No`:
  - Bỏ qua hoàn toàn bước tra cứu và nạp profile phần cứng.
  - In thông báo xác nhận:
    `✔ HardwareAdaptive engine: Disabled (Static speculation mode)`
  - Trong tệp `serving_config.json`:
    `"enable_adaptive": false`, `"profile_applied": null`.

### 3.2. FR-02: Phân giải phần cứng GPU đích cho nền tảng từ xa
- Khi `backend in ("modal", "colab")` và `enable_adaptive == True`:
  - Khởi tạo đối tượng `serving_hw` chuẩn xác theo `target_gpu`:
    - `A10G` -> `sm_86` (NVIDIA Ampere, 22-24 GB VRAM).
    - `L4` -> `sm_89` (NVIDIA Ada Lovelace, 23-24 GB VRAM).
    - `T4` -> `sm_75` (NVIDIA Turing, 15-16 GB VRAM).
    - `A100` -> `sm_80` (NVIDIA Ampere, 40/80 GB VRAM).
  - Đối tượng `serving_hw` này được dùng cho việc kiểm tra và liên kết hồ sơ phần cứng.

### 3.3. FR-03: Dừng quy trình và hỏi tương tác khi thiếu profile thực nghiệm
- Áp dụng phổ quát cho **bất kỳ phần cứng nào chưa có profile thực nghiệm** khi `enable_adaptive == True`:
  - Gọi `load_profile_for_hardware(serving_hw, strict=True)`.
  - Nếu không tìm thấy profile đo đạc (`cost_model is None` hoặc `profile_name == ""` hoặc `profile_name == "conservative_analytical_fallback"`):
    - Dừng tiến trình cấu hình ngay lập tức (không lưu file cấu hình, không hỏi khởi chạy server).
    - Hiển thị thông báo lỗi rõ ràng:
      ```text
      ✖ [Hardware Profile Error] No measured latency profile found for GPU '<gpu_name>' (<sm_architecture>).
      HardwareAdaptive requires an empirical calibration profile to dynamically optimize speculative step size k.
      ```
    - Hiển thị menu lựa chọn tương tác bằng `prompt_choice`:
      ```text
      ? Hardware profile is missing. What would you like to do?
        1. Calibrate GPU profile now (run micro-benchmark)
        2. Switch to static speculation (serve without HardwareAdaptive)
        3. Abort setup
      ```
    - **Xử lý lựa chọn 1 (Calibrate GPU profile now)**:
      - Tiến hành chạy đo đạc và hiệu chuẩn: `saved_path = run_hardware_micro_benchmark(serving_hw, target_model, draft_model)`.
      - In thông báo: `✔ Calibrated & saved new hardware profile: {saved_path}`.
      - Nạp lại profile vừa lưu với `strict=True`.
      - In thông báo: `✔ Linked hardware profile: {profile_name}`.
      - Tiếp tục quy trình lưu cấu hình và hỏi khởi chạy server.
    - **Xử lý lựa chọn 2 (Switch to static speculation)**:
      - Thiết lập `enable_adaptive = False`, `profile_name = None`.
      - In thông báo: `✔ Switched to static speculation mode (HardwareAdaptive disabled)`.
      - Tiếp tục quy trình lưu cấu hình và hỏi khởi chạy server.
    - **Xử lý lựa chọn 3 (Abort setup)**:
      - In thông báo: `✖ Serving setup aborted: Cannot enable HardwareAdaptive without a calibrated profile.`
      - Trả về `{"action": "abort", "exit_code": 1}` để thoát chương trình với mã lỗi 1.

### 3.4. FR-04: Đồng bộ cơ chế chặn lỗi cho lệnh CLI không tương tác (`flexsds serve`)
- Trong `src/flexsds/cli/main.py`:
  - Khi cờ `args.no_adaptive` không được truyền (tức adaptive đang bật):
  - Tra cứu profile phần cứng ở chế độ nghiêm ngặt (`strict=True`).
  - Nếu không tìm thấy profile thực nghiệm:
    - Báo lỗi và dừng chương trình (exit code 1):
      ```text
      ✖ [Hardware Profile Error] No measured latency profile for GPU '<gpu_name>' (<sm_architecture>).
      Please calibrate using: flexsds profile --benchmark (or run with --no-adaptive for static speculation).
      ```
    - Không cho phép tự ý áp dụng fallback trừ khi người dùng chỉ định rõ ràng cờ `--allow-fallback`.

---

## 4. Thiết kế mã nguồn chi tiết

### 4.1. Logic bước kiểm tra hồ sơ phần cứng trong `src/flexsds/cli/wizard.py`

```python
    profile_name = None
    if not enable_adaptive:
        print("\033[32m✔ HardwareAdaptive engine: Disabled (Static speculation mode)\033[0m")
    else:
        cost_model, profile_name = load_profile_for_hardware(serving_hw, strict=True)
        if cost_model is None or not profile_name or profile_name == "conservative_analytical_fallback":
            print(
                f"\n\033[1;31m✖ [Hardware Profile Error] No measured latency profile found for GPU "
                f"'{target_gpu or serving_hw.device_name}' ({serving_hw.sm_architecture}).\033[0m"
            )
            print("  HardwareAdaptive requires an empirical calibration profile to dynamically optimize speculative step size k.")

            missing_action = prompt_choice(
                "Hardware profile is missing. What would you like to do?",
                [
                    "Calibrate GPU profile now (run micro-benchmark)",
                    "Switch to static speculation (serve without HardwareAdaptive)",
                    "Abort setup",
                ],
                default_idx=0,
            )

            if "Calibrate" in missing_action:
                print(f"\nCalibrating hardware cost profile for {serving_hw.device_name} ({serving_hw.sm_architecture})...")
                saved_path = run_hardware_micro_benchmark(
                    serving_hw,
                    target_model=target_model,
                    draft_model=draft_model,
                )
                print(f"\033[1;32m✔ Calibrated & saved new hardware profile: {saved_path}\033[0m")
                cost_model, profile_name = load_profile_for_hardware(serving_hw, strict=True)
                print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")
            elif "Switch to static" in missing_action:
                enable_adaptive = False
                profile_name = None
                print("\033[32m✔ Switched to static speculation mode (HardwareAdaptive disabled)\033[0m")
            else:
                print("\n\033[1;31m✖ Serving setup aborted: Cannot enable HardwareAdaptive without a calibrated profile.\033[0m")
                return {"action": "abort", "exit_code": 1}
        else:
            print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")
```

---

## 5. Tiêu chuẩn nghiệm thu (Acceptance Criteria)

- **AC-01**: Khi người dùng chọn `Enable HardwareAdaptive engine? No`:
  - Wizard không in dòng `Linked hardware profile`.
  - Tệp `serving_config.json` lưu `"enable_adaptive": false` và `"profile_applied": null`.
- **AC-02**: Khi người dùng chọn GPU chưa có hồ sơ thực nghiệm (ví dụ A10G `sm_86`) với `enable_adaptive == True`:
  - Wizard không tự ý tiếp tục với `conservative_analytical_fallback`.
  - Wizard dừng lại và hiển thị cảnh báo lỗi màu đỏ `✖ [Hardware Profile Error]`.
  - Wizard hiển thị menu hỏi người dùng với 3 phương án: Calibrate now, Switch to static speculation, Abort setup.
- **AC-03**: Khi người dùng chọn phương án `Calibrate GPU profile now`:
  - Hệ thống chạy benchmark hiệu chuẩn, lưu tệp hồ sơ phần cứng, liên kết hồ sơ vừa tạo và cho phép tiếp tục luồng phục vụ.
- **AC-04**: Khi người dùng chọn phương án `Switch to static speculation`:
  - Hệ thống chuyển `enable_adaptive = False`, `profile_applied = None` và tiếp tục luồng phục vụ.
- **AC-05**: Khi người dùng chọn phương án `Abort setup`:
  - Hệ thống dừng lại ngay lập tức, không lưu tệp cấu hình và trả về mã thoát `exit_code: 1`.
- **AC-06**: Lệnh CLI `flexsds serve --backend modal --gpu A10G` ở chế độ non-interactive khi bật adaptive sẽ dừng lại và trả về mã lỗi 1 nếu chưa có profile.
- **AC-07**: 100% các bài kiểm thử unit test trong toàn bộ repository đều PASS.
