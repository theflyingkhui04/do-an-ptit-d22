# Nhật ký tiến độ triển khai: Hợp đồng thực thi tương đồng (matched execution contract) cho Static EAGLE và Vanilla target-only

- **Nhánh thực thi**: `fix/r6-hardware-cost-driven-budget` (tiếp nối sau khi đóng xong R6 trên CPU, giữ nguyên nhánh audit `audit/flexsds-v023-hardwareadaptive-20261120`)
- **Ngày hoàn thành pha CPU/unit**: 2026-11-22
- **Đặc tả tham chiếu**: `logs/bugs/matchedExecutionContractBug/2026-11-22-matched-execution-contract-spec.md` (`SPEC-MATCHED-EXEC-20261122`)
- **Kế hoạch tham chiếu**: `logs/bugs/matchedExecutionContractBug/plan.md`
- **Trạng thái**: Hoàn thành toàn bộ 4 tác vụ trên CPU (sẵn sàng chạy đối chứng 3 cánh tay trên Modal A10G)

---

## 1. Danh sách commit của phần matched execution contract

| Thứ tự | Commit SHA | Nội dung |
|---:|---|---|
| 1 | `9eab22b` | `docs(spec): add matched execution contract spec and plan` |
| 2 | `c05eadb` | `feat(runner): add matched execution contract, vanilla mode, and env isolation in LocalServingRunner` (Tác vụ 1 — FR-01, FR-02) |
| 3 | `aee19f2` | `feat(modal_runner): forward and pre-validate matched execution contract options` (Tác vụ 2 — FR-03) |
| 4 | `0a080a9` | `feat(cli): expose --match-adaptive-runtime, --vanilla, --no-prefix-caching, and --static-k on serve` (Tác vụ 3 — FR-04 + 2 ca rà soát đối nghịch: đồng bộ `num_speculative_tokens_per_batch_size` theo `static_k` cho `draft_model`, và chặn `--vanilla` không rơi vào wizard) |

---

## 2. Kết quả kiểm thử hồi quy offline

- **Bộ test mới (`tests/test_matched_execution_contract.py`):** **14 passed / 0 failed** (phủ AC-01 → AC-05, 5 ca Review Focus và 2 ca rà soát đối nghịch).
- **Bộ test hồi quy toàn diện (17 nhóm file offline gồm cả 15 test R6 + 14 test matched contract + planner/controller/runner/modal/lineage):**
  - Kết quả: **236 passed, 88 skipped, 0 failed** (trong 8.80 giây).
- **Toàn vẹn profile A10G:** `git diff` rỗng và `validate_measured_profile()` trả `A10G_PROFILE_VALID_OK`.

---

## 3. Ba lệnh CLI đối chứng chuẩn trên cùng hợp đồng thực thi V1 eager (Modal A10G)

Khi chạy kiểm chứng lossless sạch và benchmark batch sweep (`B = 1, 4, 8, 16, 32`) trên Modal A10G (có thể thêm `--deploy` nếu muốn giữ endpoint cố định hoặc `--dry-run` để kiểm tra trước):

1. **Cánh tay 1 — Vanilla target-only matched (kèm cache-off cho lossless):**
   ```bash
   flexsds serve --backend modal --gpu A10G \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ \
     --vanilla --match-adaptive-runtime --no-prefix-caching
   ```
2. **Cánh tay 2 — Static EAGLE3 matched (`K = 4`, cùng V1 eager với Adaptive ON):**
   ```bash
   flexsds serve --backend modal --gpu A10G \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ \
     --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
     --no-adaptive --match-adaptive-runtime --static-k 4 --no-prefix-caching
   ```
3. **Cánh tay 3 — Adaptive EAGLE3 (R6 cost-driven theo profile A10G):**
   ```bash
   flexsds serve --backend modal --gpu A10G \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ \
     --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
     --no-prefix-caching
   ```
4. **Cột tham chiếu bổ sung (as-shipped vLLM default — có CUDA Graphs và `torch.compile`):**
   ```bash
   flexsds serve --backend modal --gpu A10G \
     --target Qwen/Qwen2.5-7B-Instruct-AWQ \
     --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
     --no-adaptive
   ```
