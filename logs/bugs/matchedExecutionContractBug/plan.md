# Kế hoạch triển khai: Hợp đồng thực thi tương đồng (matched execution contract) cho Static EAGLE và Vanilla target-only

> **Dành cho kỹ sư AI:** KỸ NĂNG BẮT BUỘC: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi kế hoạch này từng tác vụ một. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu (Goal):** Bổ sung chế độ opt-in `match_adaptive_runtime`, `disable_speculative` (`--vanilla`), `disable_prefix_caching` (`--no-prefix-caching`) và `static_k` (`--static-k`) đồng bộ từ CLI xuống `LocalServingRunner` và `ModalServingRunner`, giúp chạy 3 cánh tay đối chứng `Vanilla` / `Static EAGLE` / `Adaptive EAGLE` trên cùng hợp đồng thực thi V1 eager mà không phá vỡ hành vi `--no-adaptive` mặc định.

**Kiến trúc (Architecture):**
1. Trong `src/flexsds/server/local_runner.py`: tách hàm thuần túy `_validate_execution_options()` và `describe_execution_contract()`, mở rộng `build_command()` và `setup_environment()` xử lý 4 trường mới đồng thời dọn sạch biến môi trường rò rỉ từ shell cha ở nhánh `enable_adaptive=False`.
2. Trong `src/flexsds/server/modal_runner.py`: mở rộng whitelist khóa truyền vào `FLEXSDS_SERVING_CONFIG`, kiểm tra hợp đồng sớm trước khi gọi Modal, và in `Execution contract:` ở `--dry-run`.
3. Trong `src/flexsds/cli/main.py`: thêm 4 cờ CLI cho `flexsds serve`, chặn tổ hợp cờ sai (exit code `2`) ngay trước khi gọi preflight mạng, hỗ trợ luồng `--vanilla` không cần `--draft`, và gắn đủ cấu hình vào `cfg["serving"]`.

**Công nghệ sử dụng (Tech Stack):** Python 3.11+, Argparse, Pytest, JSON.

**Đặc tả kỹ thuật (Spec):** `do-an-ptit-d22/logs/bugs/matchedExecutionContractBug/2026-11-22-matched-execution-contract-spec.md`

## Các ràng buộc toàn cục (Global Constraints)

- **NFR-01 (Cân bằng tuyệt đối):** Khi dùng cùng target/draft/revision và giới hạn batch/context, lệnh Adaptive ON và Static matched (`enable_adaptive=False, match_adaptive_runtime=True`) phải giống nhau 100% ở `--enforce-eager`, `--compilation-config '{"mode": 0}'`, `--async-scheduling`, `VLLM_USE_V2_MODEL_RUNNER="0"`, và chỉ khác đúng phần của lớp Adaptive (`--scheduler-cls`, `--distributed-executor-backend`, `num_speculative_tokens_per_batch_size`, `FLEXSDS_PROFILE_PATH`, `ADAPTIVE_STRICT`).
- **FR-05 (Tương thích ngược tuyệt đối):** Không thay đổi hành vi mặc định khi không truyền 4 cờ mới (`--no-adaptive` mặc định giữ nguyên vLLM default, wizard tương tác không đổi).
- **NFR-02 (Nhánh Git):** Không chỉnh sửa nhánh audit đóng băng `audit/flexsds-v023-hardwareadaptive-20261120`. Thực thi trên nhánh phát triển hiện hành (`fix/r6-hardware-cost-driven-budget` hoặc nhánh mới tách từ nó).
- **NFR-03 (Kiểm thử offline):** 100% unit test mới chạy độc lập trên CPU không cần mạng, `torch`, hay `modal`.
- **Chính tả tiếng Việt:** Không viết hoa kiểu Title Case tiếng Anh trong tài liệu/comment tiếng Việt; chỉ viết hoa chữ cái đầu câu/tiêu đề, danh từ riêng (FlexSDS, EAGLE, vLLM, Modal, CUDA, NVIDIA A10, A10G, Hardware-Adaptive) và từ viết tắt chuẩn (CLI, GPU, VRAM, JSON, FR, NFR, AC).

## Tiêu điểm kiểm tra (Review Focus)

1. **Rò rỉ biến môi trường từ shell cha (`VLLM_USE_V2_MODEL_RUNNER=0`, `ADAPTIVE_STRICT=1`, `FLEXSDS_PROFILE_PATH`):** Khi `enable_adaptive=False, match_adaptive_runtime=False`, `setup_environment()` phải xóa sạch cả 3 biến trên (và `ADAPTIVE_STEP_TELEMETRY_PATH`, `ADAPTIVE_STARTUP_TELEMETRY_PATH`, `ADAPTIVE_CONTROLLER_KWARGS`); khi `match_adaptive_runtime=True`, giữ `VLLM_USE_V2_MODEL_RUNNER="0"` nhưng vẫn xóa các biến Adaptive (kiểm tra trong Tác vụ 1).
2. **`static_k` là `bool` (`True`/`False`), ngoài `1..8`, hoặc kết hợp với `enable_adaptive=True` / `disable_speculative=True`:** Phải ném `ValueError` ở cả `build_command()` lẫn `setup_environment()` (kiểm tra trong Tác vụ 1).
3. **`disable_speculative=True` không truyền `draft_model`/`draft_method` (hoặc `draft_method="none"`):** `build_command()` không được báo `Unsupported draft method` và phải bỏ hoàn toàn `--speculative-config` cùng `--per-request-spec-decode-metrics` (kiểm tra trong Tác vụ 1).
4. **Cấu hình đối chứng mâu thuẫn đi qua `ModalServingRunner.launch(..., dry_run=True)`:** Phải chặn sớm trên host bằng `ValueError` trước khi gọi Modal (kiểm tra trong Tác vụ 2).
5. **Tổ hợp cờ CLI sai & luồng `--vanilla --dry-run` không cần mạng/không cần `--draft`:** `cli.main()` trả exit code `2` cho mọi tổ hợp sai ngay trước khi gọi preflight mạng, và trả `0` khi chạy `--vanilla --match-adaptive-runtime --no-prefix-caching --target ... --backend modal --gpu A10G --dry-run` (kiểm tra trong Tác vụ 3).

---

### Tác vụ 1: Hỗ trợ matched execution contract và dọn sạch env trong `LocalServingRunner`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/server/local_runner.py:64-250`
- Create: `do-an-ptit-d22/tests/test_matched_execution_contract.py`

**Interfaces:**
- Consumes: `serving_cfg` với 4 khóa opt-in mới (`match_adaptive_runtime: bool = False`, `disable_speculative: bool = False`, `disable_prefix_caching: bool = False`, `static_k: Optional[int] = None`).
- Produces:
  - `LocalServingRunner.describe_execution_contract(self, config: Dict[str, Any]) -> str` trả nhãn chuẩn (`"adaptive-eager-v1"`, `"static-matched-eager-v1"`, `"static-default"`, `"vanilla-matched-eager-v1"`, `"vanilla-default"`, kèm `"+no-prefix-cache"` nếu `disable_prefix_caching=True`).
  - `LocalServingRunner.build_command(config, port=None) -> List[str]` và `LocalServingRunner.setup_environment(config) -> Dict[str, str]` áp dụng đúng FR-01 và FR-02.
  - `LocalServingRunner.launch(..., dry_run=True)` in thêm dòng `Execution contract: <label>`.

- [x] **Bước 1: Viết các bài test thất bại cho AC-01, AC-02, AC-03 và Review Focus #1–#3**

Tạo `do-an-ptit-d22/tests/test_matched_execution_contract.py`:
```python
"""Offline tests for matched execution contract across Vanilla, Static EAGLE, and Adaptive."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flexsds.server.local_runner import LocalServingRunner

A10G_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flexsds"
    / "profiles"
    / "profile_sm_86_remote_a10g_fcda5486b12d.json"
)


def _base_pair_cfg(**overrides):
    data = json.loads(A10G_PROFILE_PATH.read_text(encoding="utf-8"))
    cfg = {
        "target_model": data["models"]["target"],
        "draft_model": data["models"]["draft"],
        "draft_method": data["draft_method"],
        "target_revision": data["models"]["target_revision"],
        "draft_revision": data["models"]["draft_revision"],
        "port": 8000,
        "max_model_len": 4096,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.85,
    }
    cfg.update(overrides)
    return {"serving": cfg}


def test_adaptive_on_and_static_matched_share_identical_v1_eager_contract():
    """AC-01: Adaptive ON và Static matched dùng chung 100% cờ V1 eager và env runner."""
    runner = LocalServingRunner()
    on_cfg = _base_pair_cfg(enable_adaptive=True, profile_path=str(A10G_PROFILE_PATH))
    static_matched_cfg = _base_pair_cfg(
        enable_adaptive=False,
        match_adaptive_runtime=True,
    )

    on_cmd = runner.build_command(on_cfg)
    st_cmd = runner.build_command(static_matched_cfg)
    on_env = runner.setup_environment(on_cfg)
    st_env = runner.setup_environment(static_matched_cfg)

    for flag in ("--enforce-eager", "--compilation-config", "--async-scheduling"):
        assert flag in on_cmd
        assert flag in st_cmd
    assert st_cmd[st_cmd.index("--compilation-config") + 1] == on_cmd[on_cmd.index("--compilation-config") + 1]
    assert on_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert st_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"

    assert "--scheduler-cls" not in st_cmd
    assert "--distributed-executor-backend" not in st_cmd
    st_spec = json.loads(st_cmd[st_cmd.index("--speculative-config") + 1])
    assert "num_speculative_tokens_per_batch_size" not in st_spec
    assert st_spec["num_speculative_tokens"] == 4
    assert "FLEXSDS_PROFILE_PATH" not in st_env
    assert "ADAPTIVE_STRICT" not in st_env
    assert runner.describe_execution_contract(on_cfg) == "adaptive-eager-v1"
    assert runner.describe_execution_contract(static_matched_cfg) == "static-matched-eager-v1"


def test_vanilla_matched_omits_speculative_and_supports_no_prefix_caching():
    """AC-02 & Review Focus #3: Vanilla matched bỏ hẳn speculative và hỗ trợ cache-off."""
    runner = LocalServingRunner()
    cfg = {
        "serving": {
            "target_model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "target_revision": "b25037543e9394b818fdfca67ab2a00ecc7dd641",
            "enable_adaptive": False,
            "disable_speculative": True,
            "match_adaptive_runtime": True,
            "disable_prefix_caching": True,
            "draft_method": "none",
            "port": 8000,
        }
    }
    cmd = runner.build_command(cfg)
    env = runner.setup_environment(cfg)

    assert "--enforce-eager" in cmd
    assert "--compilation-config" in cmd
    assert "--async-scheduling" in cmd
    assert "--no-enable-prefix-caching" in cmd
    assert "--speculative-config" not in cmd
    assert "--per-request-spec-decode-metrics" not in cmd
    assert "--scheduler-cls" not in cmd
    assert "--distributed-executor-backend" not in cmd
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert runner.describe_execution_contract(cfg) == "vanilla-matched-eager-v1+no-prefix-cache"


def test_setup_environment_strips_leaked_shell_vars_for_off_modes(monkeypatch):
    """AC-03 & Review Focus #1: Dọn sạch biến rò từ shell cha ở cả default OFF và matched OFF."""
    runner = LocalServingRunner()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("ADAPTIVE_STRICT", "1")
    monkeypatch.setenv("FLEXSDS_PROFILE_PATH", "/tmp/leaked.json")
    monkeypatch.setenv("ADAPTIVE_STEP_TELEMETRY_PATH", "/tmp/steps.jsonl")
    monkeypatch.setenv("ADAPTIVE_STARTUP_TELEMETRY_PATH", "/tmp/lineage.jsonl")
    monkeypatch.setenv("ADAPTIVE_CONTROLLER_KWARGS", '{"initial_alpha": 0.5}')

    default_env = runner.setup_environment({"serving": {"enable_adaptive": False}})
    for var in (
        "VLLM_USE_V2_MODEL_RUNNER",
        "ADAPTIVE_STRICT",
        "FLEXSDS_PROFILE_PATH",
        "ADAPTIVE_STEP_TELEMETRY_PATH",
        "ADAPTIVE_STARTUP_TELEMETRY_PATH",
        "ADAPTIVE_CONTROLLER_KWARGS",
    ):
        assert var not in default_env

    matched_env = runner.setup_environment(
        {"serving": {"enable_adaptive": False, "match_adaptive_runtime": True}}
    )
    assert matched_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    for var in (
        "ADAPTIVE_STRICT",
        "FLEXSDS_PROFILE_PATH",
        "ADAPTIVE_STEP_TELEMETRY_PATH",
        "ADAPTIVE_STARTUP_TELEMETRY_PATH",
        "ADAPTIVE_CONTROLLER_KWARGS",
    ):
        assert var not in matched_env


def test_static_k_customization_and_invalid_option_rejection():
    """Review Focus #2: static_k hợp lệ ghi vào speculative_config; bool/mâu thuẫn bị từ chối."""
    runner = LocalServingRunner()
    ok_cfg = _base_pair_cfg(enable_adaptive=False, match_adaptive_runtime=True, static_k=3)
    cmd = runner.build_command(ok_cfg)
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec["num_speculative_tokens"] == 3

    bad_configs = [
        _base_pair_cfg(enable_adaptive=True, disable_speculative=True),
        _base_pair_cfg(enable_adaptive=True, static_k=3),
        _base_pair_cfg(enable_adaptive=False, disable_speculative=True, static_k=3),
        _base_pair_cfg(enable_adaptive=False, static_k=True),
        _base_pair_cfg(enable_adaptive=False, static_k=0),
        _base_pair_cfg(enable_adaptive=False, static_k=9),
    ]
    for bad in bad_configs:
        with pytest.raises(ValueError):
            runner.build_command(bad)
        with pytest.raises(ValueError):
            runner.setup_environment(bad)
```

- [x] **Bước 2: Chạy test để xác nhận thất bại**

Chạy từ `do-an-ptit-d22/`: `PYTHONPATH=src:. pytest tests/test_matched_execution_contract.py -v`
Kết quả mong đợi: `FAIL` (do `LocalServingRunner` chưa có `describe_execution_contract` và chưa xử lý 4 trường mới).

- [x] **Bước 3: Triển khai FR-01 và FR-02 trong `src/flexsds/server/local_runner.py`**

1. Thêm method nội bộ `_validate_execution_options(self, serving_cfg: Dict[str, Any]) -> None` kiểm tra các điều kiện mâu thuẫn (`disable_speculative` + `enable_adaptive`, `static_k` với `enable_adaptive`/`disable_speculative`, `static_k` không phải `int` không-`bool` trong `1..8`).
2. Thêm method `describe_execution_contract(self, config: Dict[str, Any]) -> str` trả đúng chuỗi nhãn như khối **Interfaces**.
3. Cập nhật `build_command()`:
   - Gọi `self._validate_execution_options(serving_cfg)`.
   - Tách `enable_adaptive`, `match_adaptive_runtime`, `disable_speculative`, `disable_prefix_caching`, `static_k`.
   - `use_matched_eager = bool(enable_adaptive or match_adaptive_runtime)`.
   - Khi `use_matched_eager=True`: thêm `--enforce-eager`, `--compilation-config {"mode": 0}` và `--async-scheduling` (chỉ thêm `--async-scheduling` 1 lần duy nhất, không bị trùng ở nhánh `enable_adaptive`).
   - Khi `disable_prefix_caching=True`: thêm `--no-enable-prefix-caching`.
   - Khi `disable_speculative=True`: bỏ qua kiểm tra `draft_method` và không thêm `--speculative-config` / `--per-request-spec-decode-metrics` / `--scheduler-cls` / `--distributed-executor-backend`.
   - Khi `disable_speculative=False`: dùng `num_speculative_tokens = static_k if (not enable_adaptive and static_k is not None) else 4`.
4. Cập nhật `setup_environment()`:
   - Gọi `self._validate_execution_options(serving_cfg)`.
   - Ở nhánh `else` (`enable_adaptive=False`): xóa `FLEXSDS_PROFILE_PATH`, `ADAPTIVE_STRICT`, `ADAPTIVE_STEP_TELEMETRY_PATH`, `ADAPTIVE_STARTUP_TELEMETRY_PATH`, `ADAPTIVE_CONTROLLER_KWARGS`; nếu `match_adaptive_runtime=True` đặt `env["VLLM_USE_V2_MODEL_RUNNER"] = "0"`, ngược lại `env.pop("VLLM_USE_V2_MODEL_RUNNER", None)`.
5. Trong `launch(..., dry_run=True)`: in thêm dòng `Execution contract: {self.describe_execution_contract(config)}`.

- [x] **Bước 4: Chạy test mới và `test_local_runner.py` xác nhận pass**

Chạy: `PYTHONPATH=src:. pytest tests/test_matched_execution_contract.py tests/test_local_runner.py -v`
Kết quả mong đợi: 100% PASS.

- [x] **Bước 5: Commit Tác vụ 1**

```bash
git add src/flexsds/server/local_runner.py tests/test_matched_execution_contract.py
git commit -m "feat(runner): add matched execution contract, vanilla mode, and env isolation in LocalServingRunner"
```

---

### Tác vụ 2: Chuyển tiếp và xác thực hợp đồng thực thi trong `ModalServingRunner`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/server/modal_runner.py:73-126`
- Modify: `do-an-ptit-d22/tests/test_matched_execution_contract.py`

**Interfaces:**
- Consumes: `LocalServingRunner.build_command()`, `LocalServingRunner.setup_environment()`, `LocalServingRunner.describe_execution_contract()` từ Tác vụ 1.
- Produces:
  - `ModalServingRunner.setup_environment()` chuyển tiếp thêm 4 khóa `"match_adaptive_runtime"`, `"disable_speculative"`, `"disable_prefix_caching"`, `"static_k"` vào `FLEXSDS_SERVING_CONFIG`.
  - `ModalServingRunner.launch()` xác thực cấu hình sớm trên host khi có bất kỳ cờ đối chứng nào và in `Execution contract:` khi `dry_run=True`.

- [x] **Bước 1: Viết bài test thất bại cho AC-04 và Review Focus #4**

Bổ sung vào `do-an-ptit-d22/tests/test_matched_execution_contract.py`:
```python
def test_modal_runner_forwards_matched_contract_keys_and_validates_early(capsys):
    """AC-04 & Review Focus #4: ModalServingRunner chuyển tiếp đủ 4 khóa và chặn cấu hình sai sớm."""
    from flexsds.server.modal_runner import ModalServingRunner

    runner = ModalServingRunner()
    cfg = _base_pair_cfg(
        enable_adaptive=False,
        match_adaptive_runtime=True,
        disable_prefix_caching=True,
        static_k=3,
        target_gpu="A10G",
    )
    env = runner.setup_environment(cfg)
    payload = json.loads(env["FLEXSDS_SERVING_CONFIG"])
    assert payload["match_adaptive_runtime"] is True
    assert payload["disable_prefix_caching"] is True
    assert payload["static_k"] == 3

    runner.launch(cfg, deploy=False, dry_run=True)
    out = capsys.readouterr().out
    assert "Execution contract: static-matched-eager-v1+no-prefix-cache" in out

    invalid_cfg = _base_pair_cfg(
        enable_adaptive=False,
        disable_speculative=True,
        static_k=3,
        target_gpu="A10G",
    )
    with pytest.raises(ValueError):
        runner.launch(invalid_cfg, deploy=False, dry_run=True)
```

- [x] **Bước 2: Chạy test để xác nhận thất bại**

Chạy: `PYTHONPATH=src:. pytest tests/test_matched_execution_contract.py::test_modal_runner_forwards_matched_contract_keys_and_validates_early -v`
Kết quả mong đợi: FAIL (`KeyError: 'match_adaptive_runtime'`).

- [x] **Bước 3: Cập nhật `src/flexsds/server/modal_runner.py`**

1. Trong `setup_environment()` (dòng 76–77), thêm `"match_adaptive_runtime"`, `"disable_speculative"`, `"disable_prefix_caching"`, `"static_k"` vào tuple `keys`.
2. Trong `launch()` (dòng 111–125):
   - Khi `cfg.get("enable_adaptive", False)` hoặc bất kỳ khóa nào trong `("match_adaptive_runtime", "disable_speculative", "disable_prefix_caching")` bật hoặc `cfg.get("static_k") is not None`: gọi `LocalServingRunner().build_command(config)` và `LocalServingRunner().setup_environment(config)` để fail-fast ngay trên máy host.
   - Ở nhánh `if dry_run:`, in thêm `print(f"Execution contract: {LocalServingRunner().describe_execution_contract(config)}")`.

- [x] **Bước 4: Chạy test mới và bộ test Modal offline**

Chạy: `PYTHONPATH=src:. pytest tests/test_matched_execution_contract.py tests/test_modal_runner_cli.py tests/test_modal_app_container_paths.py -v`
Kết quả mong đợi: 100% PASS.

- [x] **Bước 5: Commit Tác vụ 2**

```bash
git add src/flexsds/server/modal_runner.py tests/test_matched_execution_contract.py
git commit -m "feat(modal_runner): forward and pre-validate matched execution contract options"
```

---

### Tác vụ 3: Bổ sung 4 cờ CLI tường minh và luồng `--vanilla` trong `src/flexsds/cli/main.py`

**Files:**
- Modify: `do-an-ptit-d22/src/flexsds/cli/main.py:181-193,376-486`
- Modify: `do-an-ptit-d22/tests/test_matched_execution_contract.py`

**Interfaces:**
- Consumes: `LocalServingRunner.describe_execution_contract()` và các trường mới trong `cfg["serving"]` từ Tác vụ 1 & Tác vụ 2.
- Produces:
  - 4 cờ CLI cho `flexsds serve`: `--match-adaptive-runtime`, `--vanilla`, `--no-prefix-caching`, `--static-k <int>`.
  - Validation fail-fast trả exit code `2` ngay trước khi gọi preflight mạng.
  - Luồng `--vanilla` khởi chạy trực tiếp chỉ cần `--target` (hoặc `--dry-run` với `--backend`/`--gpu`), bỏ qua kiểm tra draft model và hardware profile.

- [x] **Bước 1: Viết bài test thất bại cho AC-05 và Review Focus #5**

Bổ sung vào `do-an-ptit-d22/tests/test_matched_execution_contract.py`:
```python
@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--match-adaptive-runtime", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--static-k", "3", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--vanilla", "--static-k", "3", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--vanilla", "--dry-run"],
        ["serve", "--target", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--draft", "duckhoi/qwen2.5-7b-eagle3.1-speculator", "--no-adaptive", "--static-k", "9", "--dry-run"],
    ],
)
def test_cli_rejects_invalid_matched_contract_flag_combinations(argv, capsys):
    """AC-05 & Review Focus #5: Chặn tổ hợp cờ CLI sai với exit code 2 trước preflight mạng."""
    from flexsds.cli.main import main

    rc = main(argv)
    assert rc == 2
    out = capsys.readouterr().out.lower()
    assert "invalid" in out


def test_cli_vanilla_matched_modal_dry_run_succeeds_offline_without_draft(capsys):
    """AC-05 & Review Focus #5: --vanilla --match-adaptive-runtime --dry-run chạy offline không cần --draft."""
    from flexsds.cli.main import main

    rc = main(
        [
            "serve",
            "--backend",
            "modal",
            "--gpu",
            "A10G",
            "--target",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--vanilla",
            "--match-adaptive-runtime",
            "--no-prefix-caching",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Execution contract: vanilla-matched-eager-v1+no-prefix-cache" in out
    assert '"disable_speculative": true' in out
    assert '"match_adaptive_runtime": true' in out
    assert '"disable_prefix_caching": true' in out
```

- [x] **Bước 2: Chạy test để xác nhận thất bại**

Chạy: `PYTHONPATH=src:. pytest tests/test_matched_execution_contract.py -k "test_cli_" -v`
Kết quả mong đợi: FAIL (`SystemExit: 2` do argparse chưa biết các cờ mới).

- [x] **Bước 3: Cập nhật `src/flexsds/cli/main.py`**

1. Trong `parse_args()` (dòng 181–192), thêm vào `serve_parser`:
   - `--match-adaptive-runtime` (`action="store_true"`)
   - `--vanilla` (`action="store_true"`)
   - `--no-prefix-caching` (`action="store_true"`)
   - `--static-k` (`type=positive_int`, `default=None`)
2. Ngay đầu nhánh `if args.command == "serve":` (dòng 376):
   - Kiểm tra 4 quy tắc tổ hợp cờ sai theo FR-04 (trả `2` kèm thông báo bắt đầu bằng `"Invalid ..."`).
   - Cập nhật điều kiện vào nhánh non-interactive: thêm `or (args.vanilla and args.target)` hoặc `or (args.vanilla and args.dry_run)`.
3. Trong nhánh non-interactive:
   - Nếu `args.vanilla`:
     - Đặt `enable_adaptive = False`, `draft_model = None`, `draft_method = "none"`, `profile_name = None`, bỏ qua `evaluate_serving_preflight(...)` (chỉ kiểm tra nếu `args.backend == "local"` và không phải `args.dry_run` và `not hw_info.is_cuda_available` thì chặn với thông báo CUDA unavailable, exit code `2`).
     - Gắn vào `cfg["serving"]`: `"disable_speculative": True`, `"match_adaptive_runtime": args.match_adaptive_runtime`, `"disable_prefix_caching": args.no_prefix_caching`, `"static_k": None`.
   - Nếu không phải `args.vanilla`: giữ nguyên luồng preflight/profile hiện có, và gắn thêm vào `cfg["serving"]`:
     `"match_adaptive_runtime": args.match_adaptive_runtime`, `"disable_speculative": False`, `"disable_prefix_caching": args.no_prefix_caching`, `"static_k": args.static_k`.
4. Ở nhánh `if args.dry_run:`, `LocalServingRunner().launch(..., dry_run=True)` và `ModalServingRunner().launch(..., dry_run=True)` đã tự in `Execution contract: ...` (không in trùng lặp 2 lần).

- [x] **Bước 4: Chạy toàn bộ `test_matched_execution_contract.py` để xác nhận pass**

Chạy: `PYTHONPATH=src:. pytest tests/test_matched_execution_contract.py -v`
Kết quả mong đợi: 100% PASS.

- [x] **Bước 5: Commit Tác vụ 3**

```bash
git add src/flexsds/cli/main.py tests/test_matched_execution_contract.py
git commit -m "feat(cli): expose --match-adaptive-runtime, --vanilla, --no-prefix-caching, and --static-k on serve"
```

---

### Tác vụ 4: Kiểm thử hồi quy toàn diện và ghi nhận tiến độ

**Files:**
- Create: `do-an-ptit-d22/logs/bugs/matchedExecutionContractBug/progress.md`
- Modify: `do-an-ptit-d22/logs/bugs/matchedExecutionContractBug/plan.md` & `2026-11-22-matched-execution-contract-spec.md`

- [x] **Bước 1: Chạy toàn bộ bộ test hồi quy offline (AC-06)**

Chạy từ `do-an-ptit-d22/`:
```bash
PYTHONPATH=src:. pytest \
  tests/test_matched_execution_contract.py \
  tests/test_hardware_cost_driven_controller.py \
  tests/test_budget_controller.py \
  tests/test_budget_planner.py \
  tests/test_memory_pressure_rule.py \
  tests/test_proposer_adapter.py \
  tests/test_tree_adaptive_controller.py \
  tests/test_modal_native_interpreter.py \
  tests/test_modal_app_container_paths.py \
  tests/test_modal_runner_cli.py \
  tests/test_local_runner.py \
  tests/test_m4_readiness.py \
  tests/test_cost_model*.py \
  tests/test_lineage*.py \
  tests/test_adaptive*.py \
  tests/test_hardware_profiler_benchmark.py \
  tests/test_profile_import.py -q
```
Kết quả mong đợi: 100% PASS (0 failed).

- [x] **Bước 2: Ghi `logs/bugs/matchedExecutionContractBug/progress.md` và cập nhật trạng thái spec/plan**

Ghi lại danh sách commit SHA, kết quả kiểm thử, và 3 lệnh CLI mẫu sẵn sàng chạy đối chứng trực tiếp trên Modal A10G (Vanilla matched / Static matched / Adaptive ON).

- [x] **Bước 3: Commit tài liệu tiến độ**

```bash
git add logs/bugs/matchedExecutionContractBug/
git commit -m "docs(progress): record completion of matched execution contract"
```
