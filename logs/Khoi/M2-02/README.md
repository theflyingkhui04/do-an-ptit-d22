# [M2-02] Hiện Thực Hóa Bộ Điều Khiển Cây Nháp Thích Ứng (Adaptive Controller v0)

Tài liệu nhật ký kỹ thuật chi tiết của task M2-02 được lưu tại:
👉 **[M2-02_worklogs.md](./M2-02_worklogs.md)**

### File Mã Nguồn & Triển Khai:
- Module điều khiển cây nháp thích ứng: [`src/proposer/controller.py`](../../../src/proposer/controller.py)
- Lớp adapter tích hợp vLLM engine: [`src/proposer/adapter.py`](../../../src/proposer/adapter.py)
- Bộ kiểm thử tự động: [`tests/test_budget_controller.py`](../../../tests/test_budget_controller.py) & [`tests/test_proposer_adapter.py`](../../../tests/test_proposer_adapter.py)
- Launcher Làn 3 (Adaptive Speculative): [`scripts/launch_lane3_adaptive.sh`](../../../scripts/launch_lane3_adaptive.sh)
