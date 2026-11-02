---
description: Quy chuẩn Git workflow, đặt tên nhánh, commit convention và quy trình PR cho dự án Đồ án D22
trigger: always_on
---

# Quy chuẩn Git cho dự án Đồ án D22 (Hardware-Aware Speculative Decoding)

Quy định này áp dụng bắt buộc cho tất cả thành viên (Khôi, Đại, Dương) và trợ lý AI khi thao tác với Git trong kho lưu trữ này.

---

## 1. Phân cấp nhánh (Branching model)

* **`main`**: Nhánh ổn định, chứa các mốc nghiệm thu đã chốt (Milestones M0 → M5). Tuyệt đối không commit trực tiếp vào `main`.
* **`develop`**: Nhánh tích hợp chính của các sprint. Tất cả các nhánh tính năng đều tách ra từ `develop` và tạo Pull Request (PR) về `develop`.
* **Nhánh tính năng / sửa lỗi (Feature / Fix branches)**:
  * Tính năng mới: `feat/<task-id>-<short-description>` (ví dụ: `feat/m3-02-cost-model-l4`, `feat/m3-04-eagle-head-pipeline`).
  * Sửa lỗi: `fix/<task-id>-<short-description>` (ví dụ: `fix/m2-03-lossless-clean`, `fix/m3-01-harness-help-patch-eol`).
  * Thực nghiệm / Bằng chứng: `evidence/<task-id>-<short-hash>` (ví dụ: `evidence/m2-04-6424c6a`).

---

## 2. Quy chuẩn thông điệp commit (Commit convention)

Tuân thủ định dạng **Conventional Commits**:
```text
<type>(<scope>): <mô tả ngắn bằng tiếng Anh hoặc tiếng Việt chuẩn>
```

### Các tiền tố (Types):
* **`feat`**: Tính năng mới hoặc mở rộng chức năng (ví dụ: `feat(proposer): add L4 hardware cost model`).
* **`fix`**: Sửa lỗi logic, sửa bug hoặc xử lý cảnh báo kiểm thử (ví dụ: `fix(m3-02): address SonarCloud findings in profiler`).
* **`docs`**: Thêm hoặc cập nhật tài liệu, worklog, chương quyển đồ án (ví dụ: `docs(m3-02): update worklog formatting`).
* **`test`**: Bổ sung hoặc chỉnh sửa bộ kiểm thử (ví dụ: `test(proposer): add unit tests for cost model L4`).
* **`chore`**: Tác vụ bảo trì, cấu hình repo, thêm artifact evidence (ví dụ: `chore(m3-02): add L4 forward-cost rerun evidence`).
* **`refactor`**: Tái cấu trúc mã nguồn mà không thay đổi logic bên ngoài.

### Phạm vi (Scope):
* Mã task trên Trello: `m3-02`, `m3-04`, `m2-01`,...
* Hoặc module trong `src/`: `proposer`, `pipeline`, `client`, `server`, `repo`.

### Lưu ý quan trọng:
* Mô tả ngắn gọn, viết chữ thường, không chấm câu ở cuối tiêu đề.
* Nếu viết commit bằng tiếng Việt, **bắt buộc tuân thủ quy tắc chính tả tiếng Việt** (không dùng English Title Case).

---

## 3. Quy trình làm việc & nghiệm thu (PR & DoD)

1. **Trước khi commit:**
   * Chạy kiểm thử cục bộ: `PYTHONPATH=. pytest tests/ -v` (đặc biệt các test liên quan đến module sửa đổi) để đảm bảo 100% PASS.
   * Đảm bảo không commit file nhạy cảm (`.env`, token cá nhân, khóa API Trello/Modal/GitHub).
   * Không commit file nhị phân checkpoint nặng (`.pt`, `.safetensors`, `.bin`) vào Git; lưu trữ trên HuggingFace hoặc Modal Volume.

2. **Ghi chép nhật ký (Worklog):**
   * Mỗi task kỹ thuật phải có file nhật ký cập nhật tại `logs/<Assignee>/<Task-ID>/<Task-ID>_worklogs.md`.
   * Ghi nhận rõ ràng: Mục tiêu, phương pháp, kết quả số liệu thực nghiệm, và các artifact đã tạo.

3. **Mở Pull Request (PR):**
   * Tiêu đề PR rõ ràng, gắn mã task tương ứng trên Trello.
   * Ghi chú ngắn gọn các thay đổi và kết quả test đã kiểm chứng.
