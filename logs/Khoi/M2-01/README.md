# [M2-01] Xây Dựng Module Hardware Profiler & Mô Hình Chi Phí Suy Luận trên T4

- **Trạng thái:** ✅ **100% HOÀN THÀNH (DONE)** — Đã nghiệm thu 5/5 checkItems trên Trello.
- **Người thực hiện:** Cấn Đức Khôi (Lead 70%) & Bùi Đức Đại (Partner 30%)
- **Nhánh phát triển:** `feat/m2-01-budget-controller-v0` (Đã merge PR #10 vào `develop`)

Tài liệu nhật ký kỹ thuật chi tiết của task M2-01 được lưu tại:
👉 **[M2-01_worklogs.md](./M2-01_worklogs.md)** (Nhật ký kỹ thuật của Khôi)  
👉 **[logs/daijestic/M2-01/M2-01_worklogs.md](../../daijestic/M2-01/M2-01_worklogs.md)** (Nhật ký kỹ thuật của Đại)

### File Mã Nguồn & Thực Nghiệm:
- Module điều khiển cây nháp thích ứng v0: [`src/proposer/controller.py`](../../../src/proposer/controller.py)
- Lớp adapter tích hợp vòng lặp vLLM: [`src/proposer/adapter.py`](../../../src/proposer/adapter.py)
- Package export: [`src/proposer/__init__.py`](../../../src/proposer/__init__.py)
- Script benchmark đo ổn định VRAM trên T4: [`scripts/test_vram_stability.py`](../../../scripts/test_vram_stability.py)
- Artifact kết quả đo VRAM: [`logs/Khoi/M2-01/vram_stability_t4.json`](./vram_stability_t4.json)

### Bộ Kiểm Thử Tự Động (44/44 Tests PASS 100%):
- Kiểm thử BudgetController: [`tests/test_budget_controller.py`](../../../tests/test_budget_controller.py) (19 tests)
- Kiểm thử ProposerAdapter: [`tests/test_proposer_adapter.py`](../../../tests/test_proposer_adapter.py) (8 tests)
- Kiểm thử kịch bản chuyển đổi ngân sách (Đại): [`tests/test_budget_transition_scenarios.py`](../../../tests/test_budget_transition_scenarios.py) (5 tests)
- Kiểm thử hồi phục fallback & probing (Đại & Khôi): [`tests/test_budget_fallback_recovery.py`](../../../tests/test_budget_fallback_recovery.py) (8 tests)
- Kiểm thử vòng đời bộ nhớ & Zero Leak (Khôi): [`tests/test_vram_memory_lifecycle.py`](../../../tests/test_vram_memory_lifecycle.py) (4 tests)

### Kế Hoạch Triển Khai Kỹ Thuật:
- BudgetController v0: [`docs/superpowers/plans/2026-10-10-budget-controller-v0.md`](../../../docs/superpowers/plans/2026-10-10-budget-controller-v0.md)
- AdaptiveProposerAdapter: [`docs/superpowers/plans/2026-10-10-adaptive-proposer-adapter.md`](../../../docs/superpowers/plans/2026-10-10-adaptive-proposer-adapter.md)


