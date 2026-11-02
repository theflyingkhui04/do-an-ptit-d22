# [M3-02] Worklog Khôi — Xây dựng Hardware Profiler & Mô hình Chi phí C(N, B) trên GPU L4

> **Task:** `[M3-02] Hardware Profiler & Mô hình Chi phí C(N, B) trên GPU L4 (24/10 – 26/10)`  
> **Người thực hiện:** Cấn Đức Khôi (Lead 70%) & Dương Nguyễn Thế (Partner 30%)  
> **Nhánh phát triển:** `feat/m3-02-cost-model-l4`  
> **Trạng thái:** ✅ **HOÀN THÀNH (100% PASS 41/41 unit tests)**

---

## 1. Mục tiêu hoàn thành
1. Tiếp nhận kết quả đo thực nghiệm forward $C_T$ và $C_D$ trên GPU NVIDIA L4 từ Dương (`experiments/results/m3_02_forward_costs_l4.json`).
2. Mô hình hóa chi phí xác thực đa token $C_V(k, B)$ trên kernel Attention của GPU L4 (sm_89).
3. Lập ma trận chi phí 2 chiều $C(k, B)$ và xuất file chuẩn hóa `experiments/results/hardware_profile_l4.json`.
4. Xây dựng module `src/proposer/cost_model.py` với lớp `HardwareCostModel` tự động hóa tính điểm hòa vốn $\alpha^*(k, B)$ bằng thuật toán giải nghiệm đơn điệu (Bisection root solver).
5. Tích hợp trực tiếp với `BudgetController` và đảm bảo triệt tiêu hoàn toàn hiện tượng tăng tốc âm.

---

## 2. Kết quả Toán học & Thực nghiệm trên GPU L4

### A. Chi phí Forward cơ sở (Median across context lengths):
* **Target Model ($C_T$) — Qwen2.5-7B AWQ:**
  * $B=1: 19.095\text{ ms}$
  * $B=2: 19.606\text{ ms}$
  * $B=4: 20.082\text{ ms}$
  * $B=8: 20.809\text{ ms}$
  * $B=16: 23.273\text{ ms}$
  * $B=32: 30.043\text{ ms}$
* **Draft Model ($C_D$) — Qwen2.5-0.5B:**
  * $B=1: 4.885\text{ ms}$
  * $B=2: 5.230\text{ ms}$
  * $B=4: 5.266\text{ ms}$
  * $B=8: 5.415\text{ ms}$
  * $B=16: 6.270\text{ ms}$
  * $B=32: 6.697\text{ ms}$
* **Tỷ số $C_D / C_T$:** Dao động từ $0.2228$ đến $0.2694$ (trung bình $\approx 0.25$).

### B. Chi phí Xác thực $C_V(k, B)$ & Ma trận $C(k, B)$:
* Công thức xác thực song song:
  $$C_V(k, B) = C_T(B) \times (1 + 0.008 \times (k - 1))$$
* Tổng chi phí chu kỳ nháp $k$:
  $$C(k, B) = k \times C_D(B) + C_V(k, B)$$

### C. Điểm hòa vốn lý thuyết $\alpha^*(k, B)$ (Break-Even Acceptance Rate):
Giải phương trình:
$$\mathbb{E}[\tau] = 1 + \sum_{i=1}^k \alpha^i = \frac{C(k, B)}{C_{vanilla}(B)}$$

Tại $B=1$:
* $k=2: \alpha^* \approx 0.3773$ (chỉ cần $\alpha \ge 37.7\%$ là có tăng tốc dương).
* $k=3: \alpha^* \approx 0.4657$ (chỉ cần $\alpha \ge 46.6\%$).
* $k=4: \alpha^* \approx 0.5325$ (chỉ cần $\alpha \ge 53.3\%$).

So với ngưỡng tĩnh cũ của M2 ($0.65$ và $0.80$), mô hình chi phí động phần cứng cho thấy GPU L4 có thể bắt đầu tăng tốc ngay từ mức $\alpha \approx 0.38 - 0.53$, giúp mở rộng vùng hoạt động sinh lời cho Speculative Decoding.

---

## 3. Các artifact đã tạo
1. Script sinh ma trận profile: `scripts/generate_l4_hardware_profile.py`
2. File ma trận chi phí phần cứng: `experiments/results/hardware_profile_l4.json`
3. Module mô hình chi phí: `src/proposer/cost_model.py`
4. Cập nhật export: `src/proposer/__init__.py`
5. Test suite kiểm thử: `tests/test_cost_model_l4.py` (8 tests mới, 100% PASS)
