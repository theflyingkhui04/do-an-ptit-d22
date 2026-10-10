# [M2-01] Xây Dựng Module Hardware Profiler & Mô Hình Chi Phí Suy Luận trên T4

Tài liệu nhật ký kỹ thuật chi tiết của task M2-01 được lưu tại:
👉 **[M2-01_worklogs.md](./M2-01_worklogs.md)**

### File Mã Nguồn & Thực Nghiệm:
- Module điều khiển cây nháp thích ứng v0: [`src/proposer/controller.py`](../../../src/proposer/controller.py)
- Lớp adapter tích hợp vòng lặp vLLM: [`src/proposer/adapter.py`](../../../src/proposer/adapter.py)
- Package export: [`src/proposer/__init__.py`](../../../src/proposer/__init__.py)
- Bộ kiểm thử đơn vị tự động (Unit Tests):
  - Kiểm thử BudgetController: [`tests/test_budget_controller.py`](../../../tests/test_budget_controller.py)
  - Kiểm thử ProposerAdapter: [`tests/test_proposer_adapter.py`](../../../tests/test_proposer_adapter.py)
- Kế hoạch triển khai kỹ thuật:
  - BudgetController v0: [`docs/superpowers/plans/2026-10-10-budget-controller-v0.md`](../../../docs/superpowers/plans/2026-10-10-budget-controller-v0.md)
  - AdaptiveProposerAdapter: [`docs/superpowers/plans/2026-10-10-adaptive-proposer-adapter.md`](../../../docs/superpowers/plans/2026-10-10-adaptive-proposer-adapter.md)

