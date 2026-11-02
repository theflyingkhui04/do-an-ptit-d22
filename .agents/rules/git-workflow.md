---
description: Quy chuẩn Git workflow, đặt tên nhánh, commit convention theo cấu trúc Why-What-How và quy trình PR cho dự án Đồ án D22
trigger: always_on
---

# Quy chuẩn Git cho dự án Đồ án D22 (Hardware-Aware Speculative Decoding)

Quy định này áp dụng bắt buộc cho tất cả thành viên (Khôi, Đại, Dương) và trợ lý AI khi thao tác với Git trong kho lưu trữ này.

---

## 1. Phân cấp nhánh (branching model)

* **`main`**: Nhánh ổn định, chứa các mốc nghiệm thu đã chốt (Milestones M0 → M5). Tuyệt đối không commit trực tiếp vào `main`.
* **`develop`**: Nhánh tích hợp chính của các sprint. Tất cả các nhánh tính năng đều tách ra từ `develop` và tạo Pull Request (PR) về `develop`.
* **Nhánh tính năng / sửa lỗi (feature / fix branches)**:
  * Tính năng mới: `feat/<task-id>-<short-description>` (ví dụ: `feat/m3-02-cost-model-l4`, `feat/m3-04-eagle-head-pipeline`).
  * Sửa lỗi: `fix/<task-id>-<short-description>` (ví dụ: `fix/m2-03-lossless-clean`, `fix/m3-01-harness-help-patch-eol`).
  * Thực nghiệm / bằng chứng: `evidence/<task-id>-<short-hash>` (ví dụ: `evidence/m2-04-6424c6a`).

---

## 2. Quy chuẩn thông điệp commit (commit convention: Why + What + How)

Mọi commit có nội dung kỹ thuật quan trọng **bắt buộc phải có đầy đủ 3 phần: Why (Tại sao), What (Làm gì) và How (Như thế nào)**.

### Cấu trúc thông điệp chuẩn:

```text
<type>(<scope>): <tóm tắt ngắn gọn thay đổi>

Why:
- Nêu rõ lý do, bối cảnh vấn đề kỹ thuật hoặc yêu cầu từ task Trello.
- Giải thích tại sao thay đổi này là cần thiết (ví dụ: tránh tăng tốc âm, khắc phục OOM, phục vụ mốc M3).

What:
- Liệt kê các thay đổi cụ thể trong mã nguồn hoặc tài liệu.
- Các file, hàm, class hoặc cấu trúc dữ liệu mới được thêm/sửa/xóa.

How:
- Mô tả phương pháp, thuật toán, công thức toán học hoặc cơ chế thực thi.
- Cách tiếp cận giải quyết vấn đề (ví dụ: dùng giải thuật bisection, ánh xạ ma trận chi phí, adapter hook).

[Ref: #task-id | Tests: <trạng thái kiểm thử>]
```

---

### Tiền tố phân loại (Types):
* **`feat`**: Tính năng mới hoặc mở rộng chức năng.
* **`fix`**: Sửa lỗi logic, sửa bug hoặc xử lý lỗi runtime / kiểm thử.
* **`docs`**: Thêm hoặc cập nhật tài liệu, worklog, chương quyển đồ án.
* **`test`**: Bổ sung hoặc chỉnh sửa bộ kiểm thử tự động.
* **`chore`**: Tác vụ bảo trì, cấu hình repo, thêm artifact dữ liệu thô.
* **`refactor`**: Tái cấu trúc mã nguồn mà không thay đổi logic bên ngoài.

### Phạm vi (Scope):
* Mã task trên Trello: `m3-02`, `m3-04`, `m2-01`,...
* Hoặc tên module trong `src/`: `proposer`, `pipeline`, `client`, `server`, `repo`, `rules`.

---

### Ví dụ mẫu commit chuẩn mực:

```text
feat(proposer): add L4 hardware cost model and profile matrix for M3-02

Why:
- Hoàn thành tiêu chí Definition of Done cho task M3-02 trên Trello.
- Cần mô hình chi phí động trên GPU L4 để tự động tính điểm hòa vốn alpha*, thay thế các ngưỡng tĩnh cứng nhắc cũ của M2 và triệt tiêu tăng tốc âm.

What:
- Tạo module src/proposer/cost_model.py với lớp HardwareCostModel.
- Xuất file ma trận chi phí experiments/results/hardware_profile_l4.json.
- Thêm script scripts/generate_l4_hardware_profile.py để tái tạo dữ liệu từ forward costs thô.
- Viết bộ unit test tests/test_cost_model_l4.py (8 test cases).

How:
- Tiếp nhận 60 ô đo thời gian C_T và C_D từ experiments/results/m3_02_forward_costs_l4.json.
- Mô hình hóa chi phí xác thực song song C_V(k) với hệ số chú ý 0.8% mỗi token nháp.
- Cài đặt thuật toán chia đôi (Bisection) giải nghiệm đa thức E[tau] = C(k, B) / C_vanilla để tìm nghiệm hòa vốn alpha*.
- Cung cấp hàm to_cost_profile() để tương thích ngược 100% với BudgetController.

Ref: #32 | Tests: 41/41 PASS
```

---

### Lưu ý bắt buộc:
* Dòng tiêu đề không vượt quá 72 ký tự, không có dấu chấm câu ở cuối.
* Cách một dòng trống giữa dòng tiêu đề và phần thân (Why / What / How).
* Khi viết bằng tiếng Việt (trong commit message hay tài liệu), **nghiêm cấm dùng Title Case** của tiếng Anh (chỉ viết hoa đầu câu, đầu đoạn và danh từ riêng/từ viết tắt).

---

## 3. Quy trình làm việc và nghiệm thu (PR và DoD)

1. **Trước khi commit:**
   * Chạy kiểm thử cục bộ: `PYTHONPATH=. pytest tests/ -v` (đặc biệt các test liên quan đến module sửa đổi) để đảm bảo 100% PASS.
   * Đảm bảo không commit file nhạy cảm (`.env`, token cá nhân, khóa API Trello/Modal/GitHub).
   * Không commit file nhị phân checkpoint nặng (`.pt`, `.safetensors`, `.bin`) vào Git; lưu trữ trên HuggingFace hoặc Modal Volume.

2. **Ghi chép nhật ký (Worklog):**
   * Mỗi task kỹ thuật phải có file nhật ký cập nhật tại `logs/<Assignee>/<Task-ID>/<Task-ID>_worklogs.md`.
   * Ghi nhận rõ ràng: Mục tiêu, phương pháp, kết quả số liệu thực nghiệm, và các artifact đã tạo.

3. **Mở Pull Request (PR):**
   * Tiêu đề PR rõ ràng, gắn mã task tương ứng trên Trello.
   * Ghi chú ngắn gọn các thay đổi và kết quả test đã kiểm chứng theo cấu trúc Why + What + How.
