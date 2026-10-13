# M2-03 Lossless Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hiện thực hóa hàm kiểm tra tương đương tuyệt đối token ID và phát hiện sai lệch token (Lossless Verifier) cho subtask 02 của M2-03, kèm cơ chế tổng hợp xuất file nghiệm thu `experiments/results/m2_lossless_verification_summary.json`.

**Architecture:** Xây dựng module `experiments/lossless_verifier.py` (và re-export qua `src/benchmark/lossless.py`) với các dataclass `TokenDiscrepancy`, `LosslessVerificationResult`, `LosslessBatchSummary` và các hàm `verify_token_sequence`, `verify_text_output`, `aggregate_lossless_results`, `export_lossless_summary`.

**Tech Stack:** Python 3.10+, Standard Library (dataclasses, json, typing, pathlib), pytest.

**Spec:** [docs/sprint_3_plan.md](../../sprint_3_plan.md) (Task 3 M2-03) & [docs/06-ke-hoach-thuc-nghiem.md](../../06-ke-hoach-thuc-nghiem.md) (Mục 4).

## Global Constraints
- Khung chat và tài liệu không render LaTeX: Dùng chữ thường và ký tự Unicode trực quan.
- Tỷ lệ khớp yêu cầu tuyệt đối: TokenMatchRate == 100.0%, zero discrepancy ở temperature = 0.0.
- Định dạng JSON xuất chuẩn: experiments/results/m2_lossless_verification_summary.json.

---

### Task 1: Định nghĩa Data Structures và Core Verification Functions
**Files:**
- Create: `experiments/lossless_verifier.py`
- Create: `src/benchmark/__init__.py`
- Create: `src/benchmark/lossless.py`
- Test: `tests/test_lossless_verifier.py`

- [ ] **Step 1: Viết test failing kiểm tra token matching và discrepancy**
- [ ] **Step 2: Chạy test để xác nhận lỗi module chưa tồn tại**
- [ ] **Step 3: Viết mã nguồn cho TokenDiscrepancy, LosslessVerificationResult, verify_token_sequence, verify_text_output**
- [ ] **Step 4: Chạy test để xác nhận pass**

---

### Task 2: Định nghĩa Batch Aggregation và Export Summary
**Files:**
- Modify: `experiments/lossless_verifier.py`
- Modify: `src/benchmark/lossless.py`
- Test: `tests/test_lossless_verifier.py`

- [ ] **Step 1: Viết test failing cho aggregate_lossless_results và export_lossless_summary**
- [ ] **Step 2: Chạy test để xác nhận fail**
- [ ] **Step 3: Hiện thực LosslessBatchSummary, aggregate_lossless_results, export_lossless_summary**
- [ ] **Step 4: Chạy test để xác nhận pass toàn bộ test suite**
