# Task 3 Brief: Chuẩn hóa cơ chế tra cứu CPU và fallback trong `cost_model.py`

**Mục tiêu**: Loại bỏ cơ chế tự động gán thiết bị CPU về `profile_sm89_l4.json` trong `cost_model.py`. Khi `strict=True` (không có cờ `--allow-fallback`), thiết bị CPU hoặc kiến trúc không có profile đo đạc phải trả về `(None, "")` để kích hoạt fail-fast. Khi `strict=False`, cho phép trả về `(fallback_model, "conservative_analytical_fallback")`.

## Các tệp liên quan:
- Modify: `src/flexsds/core/cost_model.py`
- Modify: `tests/test_serving_profile_guard.py`

## Chi tiết các bước thực hiện:
1. Thêm vào `tests/test_serving_profile_guard.py`:
   - Bài test `test_get_serving_cost_profile_cpu_fails_in_strict_mode`: kiểm tra thiết bị CPU với `allow_fallback=False` (`strict=True`) trả về `(None, "")`.
2. Cập nhật `src/flexsds/core/cost_model.py`:
   - Xóa bỏ `"cpu": "profile_sm89_l4.json"` khỏi `arch_map`:
     ```python
     # Tier 3: built-in vendor profiles for non-A100 hardware.
     arch_map = {
         "sm_89": "profile_sm89_l4.json",
         "sm_75": "profile_sm75_t4.json",
     }
     ```
3. Chạy kiểm thử:
   `PYTHONPATH=src:. pytest tests/test_serving_profile_guard.py -v`
   Đảm bảo toàn bộ test pass.
4. Commit:
   ```bash
   git add src/flexsds/core/cost_model.py tests/test_serving_profile_guard.py
   git commit -m "fix(cost_model): do not map CPU to L4 profile in strict mode"
   ```
5. Báo cáo lại kết quả thực thi và commit hash.
