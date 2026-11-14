# Đặc tả kỹ thuật: Chuẩn hóa phân giải hồ sơ phần cứng và trạng thái HardwareAdaptive trong CLI wizard

- **Mã tài liệu**: SPEC-PROFILE-20261114
- **Ngày tạo**: 2026-11-14
- **Tác giả**: Đội ngũ phát triển FlexSDS (D22 PTIT)
- **Trạng thái**: Đã phê duyệt và hoàn thành (Approved and implemented)
- **Phạm vi tác động**: `src/flexsds/cli/wizard.py`, `src/flexsds/core/cost_model.py`, `src/flexsds/cli/main.py`

---

## 1. Bối cảnh và mục đích

Trong phiên làm việc với wizard cấu hình phục vụ (`flexsds serve` hoặc wizard tương tác), khi người dùng lựa chọn nền tảng đám mây Modal với GPU **A10G (24 GB VRAM)**, hệ thống vẫn liên kết và in ra hồ sơ phần cứng là `profile_sm89_l4.json` (thuộc GPU L4).

Đồng thời, khi người dùng lựa chọn tắt engine tối ưu hóa thích ứng (`Enable HardwareAdaptive engine? No`), wizard vẫn thực hiện tra cứu và liên kết hồ sơ phần cứng, gây nhầm lẫn rằng hệ thống vẫn bắt buộc phải có tệp profile độ trễ thì mới phục vụ được mô hình.

Mục đích của tài liệu đặc tả này là định nghĩa giải pháp chuẩn hóa quy trình phân giải hồ sơ phần cứng trong wizard:
1. Xác định đúng GPU mục tiêu (phân biệt giữa GPU máy trạm cục bộ và GPU đám mây Modal/Colab).
2. Tách biệt hoàn toàn trạng thái bật/tắt của engine HardwareAdaptive: Khi tắt adaptive, không nạp hay liên kết bất kỳ profile phần cứng nào.
3. Xử lý trường hợp GPU đích chưa có hồ sơ thực nghiệm (như A10G `sm_86`) theo tiêu chuẩn chặn lỗi sớm (fail-fast) hoặc cho phép chuyển đổi sang đầu cơ tĩnh / fallback có kiểm soát.

---

## 2. Phân tích nguyên nhân gốc rễ

### 2.1. Tra cứu nhầm phần cứng máy trạm thay vì GPU đích
- Trong file `src/flexsds/cli/wizard.py`, hàm `load_profile_for_hardware(hw_info)` được gọi bằng biến `hw_info`.
- Biến `hw_info` này đại diện cho máy tính cá nhân nơi người dùng gõ lệnh CLI (ví dụ laptop ThinkBook không có GPU NVIDIA CUDA, kiến trúc `sm_architecture = "cpu"`).
- Khi người dùng chọn phục vụ trên Modal với GPU A10G (`target_gpu = "A10G"`), wizard không tạo hoặc truyền thông tin của GPU đích vào hàm tra cứu profile, dẫn đến việc tiếp tục dùng thông số CPU của máy laptop.

### 2.2. Cơ chế gán ngầm profile L4 cho thiết bị CPU trong `cost_model.py`
- Trong file `src/flexsds/core/cost_model.py`, từ điển định tuyến hồ sơ mặc định được khai báo:
  ```python
  arch_map = {
      "sm_89": "profile_sm89_l4.json",
      "sm_75": "profile_sm75_t4.json",
      "cpu": "profile_sm89_l4.json",
  }
  ```
- Khi thiết bị máy trạm là CPU, hệ thống tự động gán `profile_sm89_l4.json`. Điều này dẫn đến hiện tượng dù chọn GPU A10G trên Modal, wizard vẫn in ra `✔ Linked hardware profile: profile_sm89_l4.json`.

### 2.3. Thực thi bước Profile Mapping vô điều kiện
- Bước 6 trong `wizard.py` được thực thi mà không kiểm tra giá trị của biến `enable_adaptive` vừa nhập ở bước 5.
- Trên thực tế, khi `enable_adaptive = False`:
  - vLLM sử dụng cơ chế đầu cơ tĩnh (static speculative decoding với số token cố định $k=4$) hoặc suy luận thông thường.
  - vLLM không khởi tạo `AdaptiveAsyncScheduler`, không dùng `LineageUniProcExecutor` và hoàn toàn không đọc ma trận chi phí độ trễ trong tệp `profile_*.json`.
  - Việc tiếp tục nạp và in profile khi adaptive bị tắt là dư thừa và làm sai lệch bản chất hoạt động của engine.

### 2.4. Thiếu hồ sơ thực nghiệm đóng gói sẵn cho GPU A10G (`sm_86`)
- GPU NVIDIA A10G thuộc kiến trúc Ampere (`sm_86`).
- Thư mục `src/flexsds/profiles/` hiện tại mới chỉ có sẵn 2 hồ sơ đo đạc: `profile_sm75_t4.json` (T4) và `profile_sm89_l4.json` (L4).
- Chưa có tệp `profile_sm86_a10g.json`. Khi người dùng muốn chạy HardwareAdaptive trên A10G, hệ thống phải thông báo rõ sự thiếu hụt này và đưa ra phương án xử lý thay vì âm thầm lấy nhầm profile của L4.

---

## 3. Mục tiêu và yêu cầu chức năng (Functional Requirements)

### 3.1. FR-01: Tách biệt hoàn toàn khi tắt HardwareAdaptive
- Nếu người dùng chọn `Enable HardwareAdaptive engine? No`:
  - Bỏ qua hoàn toàn bước tra cứu và liên kết profile phần cứng.
  - In thông báo rõ ràng ra giao diện:
    `✔ HardwareAdaptive engine: Disabled (Static speculation mode)`
  - Thiết lập trong cấu hình `serving_config.json`:
    `"enable_adaptive": false`, `"profile_applied": null`.

### 3.2. FR-02: Phân giải phần cứng dựa trên GPU đích khi phục vụ từ xa
- Khi `backend == "modal"` hoặc `backend == "colab"` và `enable_adaptive == True`:
  - Xác định cấu trúc GPU đích dựa trên `target_gpu`:
    - `A10G` -> Kiến trúc `sm_86` (NVIDIA Ampere, 24 GB VRAM).
    - `L4` -> Kiến trúc `sm_89` (NVIDIA Ada Lovelace, 24 GB VRAM).
    - `T4` -> Kiến trúc `sm_75` (NVIDIA Turing, 16 GB VRAM).
    - `A100` -> Kiến trúc `sm_80` (NVIDIA Ampere, 40/80 GB VRAM).
  - Sử dụng thông tin kiến trúc của GPU đích để tra cứu hồ sơ phần cứng thay cho máy trạm cục bộ.

### 3.3. FR-03: Xử lý trường hợp thiếu hồ sơ đo đạc (Áp dụng cho A10G)
- Khi bật HardwareAdaptive trên GPU chưa có hồ sơ thực nghiệm (như A10G `sm_86`):
  - Hiển thị thông báo cảnh báo rõ ràng:
    ```text
    ⚠ [Hardware Profile Warning] No measured latency profile found for GPU 'A10G (sm_86)'.
    ```
  - Cung cấp các hướng dẫn cụ thể:
    1. **Phương án 1 (Khuyến nghị để chạy ngay)**: Cho phép dùng mô hình suy diễn dè dặt phân tích (`conservative_analytical_fallback`) với cờ `--allow-fallback`.
    2. **Phương án 2 (Đầu cơ tĩnh)**: Tắt HardwareAdaptive để chạy suy luận đầu cơ tĩnh với $k=4$ cố định mà không cần đo profile.
    3. **Phương án 3 (Đo đạc thực nghiệm)**: Chạy lệnh đo kiểm trên máy có GPU đó để sinh hồ sơ chuẩn: `flexsds profile --benchmark`.

### 3.4. FR-04: Chuẩn hóa từ điển `arch_map` trong `cost_model.py`
- Loại bỏ việc gán ngầm `"cpu": "profile_sm89_l4.json"` khi tra cứu ở chế độ nghiêm ngặt (`strict=True`).
- Thiết bị CPU khi tra cứu profile phục vụ suy luận phải trả về thông báo lỗi không hỗ trợ thay vì giả lập là GPU L4.

---

## 4. Thiết kế giải pháp chi tiết

### 4.1. Cập nhật `src/flexsds/cli/wizard.py`

Thay thế đoạn mã ở bước 5 và bước 6 (dòng 423-445):
```python
    # Step 5: HardwareAdaptive Optimization Engine
    enable_adaptive = prompt_bool(
        "Enable HardwareAdaptive engine?",
        default_val=True,
    )

    profile_name = None
    if not enable_adaptive:
        print("\033[32m✔ HardwareAdaptive engine: Disabled (Static speculation mode)\033[0m")
    else:
        # Xác định đối tượng phần cứng phục vụ cho việc tra cứu profile
        if backend in ("modal", "colab"):
            remote_sm_map = {
                "A10G": "sm_86",
                "L4": "sm_89",
                "T4": "sm_75",
                "A100": "sm_80",
            }
            target_sm = remote_sm_map.get(target_gpu, "sm_89")
            serving_hw = HardwareInfo(
                device_name=f"Remote {target_gpu}",
                sm_architecture=target_sm,
                total_vram_mb=budgets.get(target_gpu, 22000.0),
                free_vram_mb=budgets.get(target_gpu, 22000.0),
                cuda_version="12.4",
                is_cuda_available=True,
                device_count=1,
            )
        else:
            serving_hw = hw_info

        cost_model, profile_name = load_profile_for_hardware(serving_hw, strict=False)
        if profile_name == "conservative_analytical_fallback":
            print(f"\033[33m⚠ [Hardware Profile] No measured profile for '{serving_hw.device_name} ({serving_hw.sm_architecture})'. "
                  f"Using conservative analytical fallback.\033[0m")
            print("  Tip: Calibrate exact latency using: flexsds profile --benchmark (or disable adaptive to use static speculation)")
        else:
            print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")
```

### 4.2. Cập nhật `src/flexsds/core/cost_model.py`

Bổ sung kiến trúc `sm_86` vào `arch_map` với cơ chế nhận diện rõ ràng:
```python
    arch_map = {
        "sm_89": "profile_sm89_l4.json",
        "sm_75": "profile_sm75_t4.json",
    }
```
Không gán cứng CPU thành L4 khi `strict=True`.

---

## 5. Tiêu chuẩn nghiệm thu (Acceptance Criteria)

- **AC-01**: Khi người dùng chọn `Enable HardwareAdaptive engine? No` trong wizard:
  - Wizard không in dòng `✔ Linked hardware profile`.
  - Tệp `serving_config.json` lưu `"enable_adaptive": false` và `"profile_applied": null`.
- **AC-02**: Khi người dùng chọn `backend == "modal"` và `target_gpu == "A10G"` khi bật adaptive:
  - Hệ thống không được in ra `profile_sm89_l4.json`.
  - Hệ thống nhận diện đúng kiến trúc `sm_86` của A10G và cảnh báo chưa có profile đo đạc, sau đó sử dụng `conservative_analytical_fallback` một cách minh bạch hoặc hướng dẫn người dùng tắt adaptive.
- **AC-03**: Khi người dùng chọn `backend == "modal"` và `target_gpu == "L4"` khi bật adaptive:
  - Hệ thống liên kết chính xác `profile_sm89_l4.json`.
- **AC-04**: 100% các bài kiểm thử unit test trong `tests/test_cli_wizard_flow.py` và `tests/test_serving_profile_guard.py` đều PASS.
