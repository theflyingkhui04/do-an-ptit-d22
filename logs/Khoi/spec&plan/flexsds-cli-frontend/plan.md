# Kế hoạch triển khai FlexSDS Baseline CLI

> **Tài liệu:** `plan.md`  
> **Dự án:** Kế hoạch thực thi xây dựng Baseline CLI cho FlexSDS (Sprint 5 / M4)  
> **Thành viên:** Cấn Đức Khôi (Lead 70%), Bùi Đức Đại (Reviewer 30%)  
> **Vị trí lưu trữ:** `logs/Khoi/spec&plan/plan.md`  
> **Hạn chốt hoàn thành:** 08/11/2026 (trước mốc đóng băng toàn diện 15/11)  

---

## 1. Mục tiêu và lộ trình tổng thể

Kế hoạch này vạch ra các bước kỹ thuật cụ thể để hiện thực hóa bản **Baseline CLI của FlexSDS**, chuyển hóa toàn bộ thuật toán thích ứng và dữ liệu thực nghiệm đã có thành một công cụ dòng lệnh hoàn chỉnh, thân thiện và sẵn sàng demo.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     LỘ TRÌNH THỰC HIỆN CÁC TASK                        │
├────────────────────────────────────────────────────────────────────────┤
│ [Task 1] Khởi tạo khung package & cấu hình entrypoint pyproject.toml   │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 2] Xây dựng bộ quét phần cứng hardware.py & banner PTIT ANSI     │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 3] Xây dựng bộ gác cổng an toàn tương thích compatibility.py     │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 4] Đóng gói danh mục hồ sơ phần cứng (profiles/) & bộ tra cứu    │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 5] Xây dựng Interactive Wizard & bộ phân phối lệnh CLI (main.py) │
│     │                                                                  │
│     ▼                                                                  │
│ [Task 6] Tích hợp launcher (Modal / Local) & hoàn tất bộ test suite    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết phân rã công việc (Task breakdown)

### Task 1: Khởi tạo khung package Python và cấu hình điểm vào (Entry point)
* **Mục tiêu:** 
  * Tạo cấu trúc thư mục package `src/flexsds`, thiết lập `pyproject.toml` định nghĩa điểm vào CLI.
  * Phân định rõ 2 chế độ: Trong giai đoạn phát triển, nhóm chạy `pip install -e .` (chế độ editable) để mọi cập nhật mã nguồn ăn ngay vào lệnh `flexsds`; khi xuất bản lên PyPI người dùng chỉ cần chạy `pip install flexsds`.
* **Các tệp cần tạo/sửa đổi:**
  * `pyproject.toml` (bổ sung khối `[project.scripts] flexsds = "flexsds.cli.main:cli_entry"`).
  * `src/flexsds/__init__.py`.
  * `src/flexsds/cli/__init__.py`.
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Chạy `pip install -e .` thành công không có cảnh báo lỗi.
  - Gõ `flexsds --version` trên terminal in ra đúng: `flexsds 0.2.0-alpha`.
* **Người thực hiện:** Khôi (Lead) — **Thời gian:** 1.5 giờ.

---

### Task 2: Hiện thực hóa bộ quét phần cứng `hardware.py` và Banner logo PTIT `banner.py`
* **Mục tiêu:** 
  * Xây dựng module nhận diện phần cứng nhanh, không gây nghẽn, đọc được tên card GPU, compute capability (`sm_xx`), tổng VRAM và VRAM còn trống; có fallback CPU an toàn.
  * Render logo PTIT chuẩn nhận diện thương hiệu dạng ký tự khối ANSI TrueColor kết hợp bảng hiển thị thông số GPU bên cạnh.
* **Các tệp cần tạo:**
  * `src/flexsds/core/hardware.py`.
  * `src/flexsds/cli/banner.py`.
  * `tests/test_cli_banner.py`.
  * `tests/test_hardware_detector.py`.
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Thời gian thực thi in banner và quét phần cứng $< 0.3$ giây.
  - Logo hiển thị đúng màu đỏ (`#ED1C24`) và ngôi sao vàng (`#FFF200`), không bị vỡ bố cục trên màn hình terminal 80 cột.
  - 100% Pass các unit tests kiểm tra cả trường hợp có GPU và trường hợp máy chỉ có CPU.
* **Người thực hiện:** Khôi (Lead) — **Thời gian:** 3 giờ.

---

### Task 3: Hiện thực hóa bộ gác cổng kiểm tra tương thích `compatibility.py`
* **Mục tiêu:** Xây dựng logic phòng vệ kiểm tra tính an toàn của cặp mô hình Target-Draft trước khi cho phép serving.
* **Nội dung kiểm tra:**
  1. *Tính toán an toàn VRAM:* So khớp tổng dung lượng mô hình cần thiết với lượng VRAM trống thực tế; cảnh báo đỏ nếu vượt ngưỡng 95% VRAM.
  2. *So khớp bảng từ vựng:* Tự động phát hiện mô hình lệch từ vựng để kích hoạt cờ `use_heterogeneous_vocab`.
  3. *Phân loại kiến trúc nháp:* Tự động gán phương thức suy đoán (`eagle3` cho đầu nháp EAGLE hoặc `draft_model` cho mô hình SLM độc lập).
* **Các tệp cần tạo:**
  * `src/flexsds/core/compatibility.py`.
  * `tests/test_compatibility_guard.py`.
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Phát hiện chính xác nguy cơ OOM với mô hình lớn trên GPU có VRAM nhỏ.
  - Phân loại đúng cấu hình cho cả hai cặp: Qwen2.5-7B + Qwen2.5-0.5B và Qwen2.5-7B + EAGLE-3.1 Head.
  - Đạt 100% Pass unit tests với dữ liệu mô phỏng cấu hình mô hình thật.
* **Người thực hiện:** Khôi (Lead), Đại (Review logic tính VRAM) — **Thời gian:** 2.5 giờ.

---

### Task 4: Đóng gói danh mục hồ sơ phần cứng (`profiles/`) và tích hợp mô hình chi phí
* **Mục tiêu:** Tích hợp dữ liệu thực nghiệm đã đo được từ M3-02 vào thư mục đóng gói của package, cho phép tự động nạp hồ sơ phần cứng tương ứng với GPU hiện tại.
* **Các tệp cần tạo/di chuyển:**
  * `src/flexsds/profiles/profile_sm89_l4.json` (sao chép và chuẩn hóa từ `experiments/results/hardware_profile_l4.json`).
  * `src/flexsds/profiles/profile_sm75_t4.json`.
  * `src/flexsds/core/cost_model.py` (tích hợp và mở rộng lớp `HardwareCostModel`).
  * `tests/test_cost_profile_loader.py`.
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Khi phát hiện GPU có `sm_89`, hệ thống tự động tìm và nạp đúng tệp `profile_sm89_l4.json` trong thời gian $< 5$ mili-giây.
  - Giải đúng ngưỡng hòa vốn $\alpha^*(k, B)$ cho các mức concurrency $B \in \{1, 4, 8, 16, 32\}$.
* **Người thực hiện:** Đại (Cung cấp & kiểm chứng ma trận chi phí), Khôi (Tích hợp module) — **Thời gian:** 2 giờ.

---

### Task 5: Xây dựng giao diện tương tác Interactive Wizard (`wizard.py`) và bộ điều phối (`main.py`)
* **Mục tiêu:** Xây dựng trải nghiệm hỏi–đáp thân thiện khi gõ `flexsds`, kết nối các module đã xây dựng ở các task trước thành một luồng hoàn chỉnh.
* **Nội dung hiển thị:**
  * Hỏi lựa chọn hành động: `Serve` / `Detect` / `Profile` / `Test`.
  * Hỏi lựa chọn hoặc nhập Target Model, Draft Model.
  * Chạy compatibility check và hiển thị báo cáo dạng bảng màu trực quan.
  * Hỏi lựa chọn Backend phục vụ (Modal Cloud / Local vLLM).
  * Hỏi bật lớp thích ứng TreeAdaptive.
  * Xuất tệp cấu hình phục vụ chuẩn `serving_config.json`.
* **Các tệp cần tạo:**
  * `src/flexsds/cli/wizard.py`.
  * `src/flexsds/cli/main.py`.
  * `tests/test_cli_wizard.py`.
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Chạy thử nghiệm trơn tru từ đầu đến cuối luồng wizard không bị gián đoạn.
  - Hỗ trợ cả chế độ không tương tác (Headless mode) qua các tham số `--target`, `--draft`, `--backend`.
  - Sinh đúng tệp `serving_config.json` hợp lệ.
* **Người thực hiện:** Khôi (Lead) — **Thời gian:** 3.5 giờ.

---

### Task 6: Tích hợp Launcher phục vụ và nghiệm thu toàn diện (Verification)
* **Mục tiêu:** Nối tệp `serving_config.json` vừa sinh với script khởi chạy server thực tế (kế thừa `modal_runner.py` và `launch_lane4_eagle.sh`).
* **Các tệp cần tạo:**
  * `src/flexsds/server/modal_runner.py`.
  * `src/flexsds/server/local_runner.py`.
  * `tests/test_end_to_end_flow.py`.
* **Tiêu chuẩn nghiệm thu (DoD):**
  - Lệnh `flexsds serve --dry-run` xuất ra đầy đủ và chính xác lệnh CLI của vLLM kèm các cờ speculative decoding và scheduler thích ứng.
  - Toàn bộ test suite của FlexSDS (tối thiểu 25 unit tests mới) đạt kết quả **100% Pass**.
  - Commit mã nguồn sạch sẽ vào nhánh `feature/flexsds-cli-builder`.
* **Người thực hiện:** Khôi & Đại — **Thời gian:** 3 giờ.

---

## 3. Lịch trình và điểm kiểm soát sống còn (Timeline & Milestones)

| Ngày | Hạng mục công việc | Người thực hiện | Cổng kiểm soát (Gate) |
|---|---|---|---|
| **05/11** | Viết xong tài liệu đặc tả `09-dac-ta-san-pham-flexsds.md`, `spec.md`, `plan.md` | Khôi, Đại | ✅ Cổng tài liệu đạt yêu cầu |
| **06/11** | Hoàn thành Task 1, Task 2 (Khung package, banner PTIT ANSI và hardware detector) | Khôi | Lệnh `flexsds` chạy mượt trên terminal |
| **07/11** | Hoàn thành Task 3, Task 4 (Compatibility guard và profile cost loader) | Khôi, Đại | Kiểm tra VRAM và nạp profile L4 tự động |
| **08/11** | Hoàn thành Task 5, Task 6 (Wizard tương tác, launcher và toàn bộ test suite) | Khôi, Đại | 100% Pass unit tests, sẵn sàng demo |
| **09/11–14/11**| Ghép thuật toán TreeAdaptive nâng cao & sweep kiểm chứng trên Modal | Cả nhóm | Đo đạc đối chứng hoàn thiện |
| **15/11** | **ĐÓNG BĂNG TOÀN DIỆN MÃ NGUỒN VÀ SỐ LIỆU (15/11 FREEZE)** | Cả nhóm | Khóa git commit chuẩn bị nộp quyển |
