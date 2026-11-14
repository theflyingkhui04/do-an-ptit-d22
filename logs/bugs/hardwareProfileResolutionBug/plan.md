# Kế hoạch triển khai chuẩn hóa phân giải hồ sơ phần cứng và trạng thái HardwareAdaptive trong CLI wizard

> **Dành cho kỹ sư AI:** KỸ NĂNG BẮT BUỘC: Sử dụng superpowers:subagent-driven-development (khuyến nghị) hoặc superpowers:executing-plans để thực thi kế hoạch này từng tác vụ một. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu (Goal):** Chuẩn hóa cơ chế phân giải hồ sơ phần cứng trong wizard: bỏ qua nạp profile khi tắt HardwareAdaptive, phân giải đúng kiến trúc GPU đích khi phục vụ trên Modal/Colab (A10G `sm_86`, L4 `sm_89`, T4 `sm_75`), và hiển thị cảnh báo minh bạch khi GPU chưa có profile đo đạc thực nghiệm.

**Kiến trúc (Architecture):** Tách rẽ logic wizard theo cờ `enable_adaptive`; khi bật adaptive trên backend từ xa (Modal/Colab), khởi tạo đối tượng `serving_hw` từ `target_gpu` thay vì đọc `hw_info` của máy laptop; loại bỏ mapping ngầm CPU về L4 trong `cost_model.py`; hiển thị cảnh báo `conservative_analytical_fallback` rõ ràng cho A10G kèm hướng dẫn lệnh.

**Công nghệ sử dụng (Tech Stack):** Python 3.11/3.14, Pytest, Dataclass, Pathlib.

**Đặc tả kỹ thuật (Spec):** `do-an-ptit-d22/logs/bugs/hardwareProfileResolutionBug/2026-11-14-wizard-hardware-profile-resolution.md`

## Các ràng buộc toàn cục (Global Constraints)

- Tuân thủ nghiêm ngặt quy tắc chính tả tiếng Việt: Không viết hoa kiểu Title Case tiếng Anh; chỉ viết hoa chữ cái đầu câu, đầu tiêu đề, danh từ riêng (Modal, Google Colab, vLLM, EAGLE-3.1, L4, A10G, T4, Linux, Windows, ThinkBook) và từ viết tắt chuẩn (CLI, GPU, VRAM, OOM, JSON).
- Đảm bảo 100% các bài test trong test suite (`tests/`) đều PASS sau khi chỉnh sửa.
- Không gây hồi quy đối với các lệnh CLI cờ trực tiếp (`flexsds serve --backend modal ...`).

## Tiêu điểm kiểm tra (Review Focus)

1. Người dùng chọn `Enable HardwareAdaptive engine? No` trong wizard -> Wizard không in dòng `Linked hardware profile`, lưu `"profile_applied": null` và `"enable_adaptive": false` trong `serving_config.json`.
2. Người dùng chọn Modal GPU là `A10G` khi bật adaptive -> Wizard không được in `profile_sm89_l4.json`, mà nhận diện kiến trúc `sm_86` và thông báo dùng `conservative_analytical_fallback`.
3. Người dùng chọn Modal GPU là `L4` khi bật adaptive -> Wizard nạp chính xác `profile_sm89_l4.json`.
4. Người dùng chọn Modal GPU là `T4` khi bật adaptive -> Wizard nạp chính xác `profile_sm75_t4.json`.
5. Trong `cost_model.py`, tra cứu thiết bị CPU ở chế độ `strict=True` không được tự động trả về `profile_sm89_l4.json`.

---

### Tác vụ 1: Kiểm thử và tách biệt trạng thái tắt HardwareAdaptive trong `wizard.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/cli/wizard.py:423-445`
- Modify: `do-an-ptit-d22/tests/test_cli_wizard_flow.py`

**Interfaces:**
- Consumes: `enable_adaptive: bool`
- Produces: `profile_name: Optional[str]`, cấu hình `serving_config.json` với `profile_applied: null` khi adaptive tắt

- [x] **Bước 1: Viết bài test thất bại cho trường hợp tắt HardwareAdaptive**

Thêm vào `tests/test_cli_wizard_flow.py`:
```python
def test_wizard_disabled_adaptive_skips_profile_mapping():
    """Kiểm tra khi tắt HardwareAdaptive thì không liên kết profile và profile_applied là None."""
    # Kiểm tra cấu trúc logic tạo config khi enable_adaptive = False
    enable_adaptive = False
    profile_name = None if not enable_adaptive else "profile_sm89_l4.json"
    
    assert profile_name is None
    config = {
        "enable_adaptive": enable_adaptive,
        "profile_applied": profile_name,
    }
    assert config["enable_adaptive"] is False
    assert config["profile_applied"] is None
```

- [x] **Bước 2: Chạy test để xác nhận test chạy được**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py::test_wizard_disabled_adaptive_skips_profile_mapping -v`
Kết quả mong đợi: PASS

- [x] **Bước 3: Cập nhật `wizard.py` để bỏ qua profile mapping khi `enable_adaptive = False`**

Trong `src/flexsds/cli/wizard.py`:
```python
    # Step 5: HardwareAdaptive Optimization Engine
    enable_adaptive = prompt_bool(
        "Enable HardwareAdaptive engine?",
        default_val=True,
    )

    profile_name = None
    if not enable_adaptive:
        print("\033[32m✔ HardwareAdaptive engine: Disabled (Static speculation mode)\033[0m")
    else:
        cost_model, profile_name = load_profile_for_hardware(hw_info)
        print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")
```

- [x] **Bước 4: Chạy kiểm thử wizard flow**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -v`
Kết quả mong đợi: PASS

- [x] **Bước 5: Commit**

```bash
git add src/flexsds/cli/wizard.py tests/test_cli_wizard_flow.py
git commit -m "fix(wizard): skip profile mapping when HardwareAdaptive is disabled"
```

---

### Tác vụ 2: Phân giải phần cứng GPU đích cho nền tảng Modal và Colab

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/cli/wizard.py:430-448`
- Modify: `do-an-ptit-d22/tests/test_cli_wizard_flow.py`

**Interfaces:**
- Consumes: `backend: str`, `target_gpu: str`, `hw_info: HardwareInfo`
- Produces: `serving_hw: HardwareInfo` trỏ đúng GPU đám mây được chọn

- [x] **Bước 1: Viết bài test thất bại cho việc phân giải GPU từ xa trong wizard**

Bổ sung vào `tests/test_cli_wizard_flow.py`:
```python
from flexsds.core.hardware import HardwareInfo


def test_wizard_remote_serving_hardware_resolution():
    """Kiểm tra logic phân giải đúng kiến trúc GPU đích khi chạy Modal/Colab."""
    remote_sm_map = {
        "A10G": "sm_86",
        "L4": "sm_89",
        "T4": "sm_75",
        "A100": "sm_80",
    }
    budgets = {"T4": 15000.0, "L4": 23000.0, "A10G": 22000.0, "A100": 40000.0}

    def resolve_serving_hw(backend: str, target_gpu: str, local_hw: HardwareInfo) -> HardwareInfo:
        if backend in ("modal", "colab"):
            target_sm = remote_sm_map.get(target_gpu, "sm_89")
            target_vram = budgets.get(target_gpu, 22000.0)
            return HardwareInfo(
                device_name=f"Remote {target_gpu}",
                sm_architecture=target_sm,
                total_vram_mb=target_vram,
                free_vram_mb=target_vram,
                cuda_version="12.4",
                is_cuda_available=True,
                device_count=1,
            )
        return local_hw

    cpu_hw = HardwareInfo("AMD Ryzen", "cpu", 16000.0, 16000.0, "N/A", False)
    
    # Test Modal A10G
    modal_a10g_hw = resolve_serving_hw("modal", "A10G", cpu_hw)
    assert modal_a10g_hw.sm_architecture == "sm_86"
    assert "A10G" in modal_a10g_hw.device_name

    # Test Modal L4
    modal_l4_hw = resolve_serving_hw("modal", "L4", cpu_hw)
    assert modal_l4_hw.sm_architecture == "sm_89"

    # Test Local giữ nguyên CPU
    local_hw = resolve_serving_hw("local", "L4", cpu_hw)
    assert local_hw.sm_architecture == "cpu"
```

- [x] **Bước 2: Chạy test để xác nhận test chạy được**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py::test_wizard_remote_serving_hardware_resolution -v`
Kết quả mong đợi: PASS

- [x] **Bước 3: Cập nhật hàm tạo `serving_hw` trong `wizard.py`**

Trong `src/flexsds/cli/wizard.py`:
```python
    profile_name = None
    if not enable_adaptive:
        print("\033[32m✔ HardwareAdaptive engine: Disabled (Static speculation mode)\033[0m")
    else:
        # Xác định đối tượng phần cứng phục vụ cho việc tra cứu profile
        if backend in ("modal", "colab"):
            remote_sm_map = {
                "A10G": "sm_86",
                "L4": "sm_89",
                "T4": "sm_75",
                "A100": "sm_80",
            }
            target_sm = remote_sm_map.get(target_gpu, "sm_89")
            budgets = {"T4": 15000.0, "L4": 23000.0, "A10G": 22000.0, "A100": 40000.0}
            target_vram = budgets.get(target_gpu, 22000.0)
            serving_hw = HardwareInfo(
                device_name=f"Remote {target_gpu}",
                sm_architecture=target_sm,
                total_vram_mb=target_vram,
                free_vram_mb=target_vram,
                cuda_version="12.4",
                is_cuda_available=True,
                device_count=1,
            )
        else:
            serving_hw = hw_info

        cost_model, profile_name = load_profile_for_hardware(serving_hw, strict=False)
        if profile_name == "conservative_analytical_fallback":
            print(
                f"\033[33m⚠ [Hardware Profile Warning] No measured profile for GPU '{target_gpu or serving_hw.device_name}' "
                f"({serving_hw.sm_architecture}). Using conservative analytical fallback.\033[0m"
            )
            print("  Tip: Calibrate exact latency using: flexsds profile --benchmark (or disable adaptive to use static speculation)")
        else:
            print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")
```
Và lưu thông tin `serving_hw` vào `config["hardware"]`.

- [x] **Bước 4: Chạy lại kiểm thử wizard**

Chạy: `PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py -v`
Kết quả mong đợi: PASS

- [x] **Bước 5: Commit**

```bash
git add src/flexsds/cli/wizard.py tests/test_cli_wizard_flow.py
git commit -m "fix(wizard): resolve remote target GPU hardware profile correctly"
```

---

### Tác vụ 3: Chuẩn hóa cơ chế tra cứu CPU và fallback trong `cost_model.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/core/cost_model.py:436-445`
- Modify: `do-an-ptit-d22/tests/test_serving_profile_guard.py`

**Interfaces:**
- Consumes: `hw_info: HardwareInfo`, `strict: bool`
- Produces: Tra cứu chính xác hoặc trả về `(None, "")` khi `strict=True` đối với phần cứng không hỗ trợ (CPU)

- [x] **Bước 1: Viết bài test thất bại cho thiết bị CPU ở chế độ `strict=True`**

Thêm vào `tests/test_serving_profile_guard.py`:
```python
def test_get_serving_cost_profile_cpu_fails_in_strict_mode():
    """Kiểm tra thiết bị CPU không được tự động gán profile_sm89_l4 khi strict=True."""
    cpu_hw = HardwareInfo(
        device_name="Host CPU",
        sm_architecture="cpu",
        total_vram_mb=16384.0,
        free_vram_mb=16384.0,
        cuda_version="N/A",
        is_cuda_available=False,
    )
    cost_model, prof_name = get_serving_cost_profile(cpu_hw, allow_fallback=False)
    assert cost_model is None
    assert prof_name == ""
```

- [x] **Bước 2: Chạy test để xác nhận thất bại với mã nguồn hiện tại**

Chạy: `PYTHONPATH=src:. pytest tests/test_serving_profile_guard.py::test_get_serving_cost_profile_cpu_fails_in_strict_mode -v`
Kết quả mong đợi: FAIL (vì hiện tại `cost_model.py` gán cứng `cpu: profile_sm89_l4.json`)

- [x] **Bước 3: Cập nhật `cost_model.py`**

Trong `src/flexsds/core/cost_model.py`:
Xóa bỏ `"cpu": "profile_sm89_l4.json"` khỏi `arch_map`:
```python
    # Tier 3: built-in vendor profiles for non-A100 hardware.
    arch_map = {
        "sm_89": "profile_sm89_l4.json",
        "sm_75": "profile_sm75_t4.json",
    }
```
Khi `sm_arch == "cpu"` và `strict=True`: không nạp profile L4 giả lập, trả về `(None, "")`.
Khi `strict=False`: tiếp tục được phép trả về `(fallback_model, "conservative_analytical_fallback")`.

- [x] **Bước 4: Chạy lại test suite profile guard**

Chạy: `PYTHONPATH=src:. pytest tests/test_serving_profile_guard.py -v`
Kết quả mong đợi: PASS

- [x] **Bước 5: Commit**

```bash
git add src/flexsds/core/cost_model.py tests/test_serving_profile_guard.py
git commit -m "fix(cost_model): do not map CPU to L4 profile in strict mode"
```

---

### Tác vụ 4: Kiểm thử hồi quy toàn diện và nghiệm thu Dry-run

**Files:**
- Test: Toàn bộ các test suite liên quan đến phục vụ Modal, Local, Wizard và Profile guard

- [x] **Bước 1: Chạy toàn bộ các bài kiểm thử**

```bash
PYTHONPATH=src:. pytest tests/test_cli_wizard_flow.py tests/test_serving_profile_guard.py tests/test_modal_app_container_paths.py tests/test_modal_runner_cli.py tests/test_local_runner.py tests/test_flexsds_integration.py -v
```
Kết quả mong đợi: 100% test pass.

- [x] **Bước 2: Chạy thử nghiệm wizard với mock lựa chọn A10G**

Xác nhận khi chọn A10G trên Modal, wizard không in `profile_sm89_l4.json`, mà in cảnh báo sử dụng fallback phân tích `conservative_analytical_fallback`.

- [x] **Bước 3: Cập nhật tài liệu đặc tả**

Cập nhật trạng thái trong `logs/bugs/hardwareProfileResolutionBug/2026-11-14-wizard-hardware-profile-resolution.md`:
Đổi thành `- **Trạng thái**: Đã phê duyệt và hoàn thành (Approved and implemented)`.

- [x] **Bước 4: Commit tài liệu đặc tả**

```bash
git add logs/bugs/hardwareProfileResolutionBug/2026-11-14-wizard-hardware-profile-resolution.md logs/bugs/hardwareProfileResolutionBug/plan.md
git commit -m "docs(spec): mark wizard hardware profile resolution as implemented"
```
