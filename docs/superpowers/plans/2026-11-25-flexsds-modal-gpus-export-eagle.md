# FlexSDS Modal GPU Catalog, API JSON Export, EAGLE-1/2/3/3.1 Sync & Colab Future Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize FlexSDS Modal GPU selection to 4 tiered GPUs (`T4` Low, `A10` Medium, `L40S` High, `A100` High mapping to Modal `A100-80GB`) with VRAM & pricing footers, normalize active repo references from `A10G` to `A10`, add post-deploy API JSON config export for the web demo, fix EAGLE-1/2/3/3.1 consistency across CLI and web, and document Colab MCP/API automation in `logs/future-work/`.

**Architecture:** Introduce a single source of truth (`MODAL_GPU_SPECS` and helpers) in `src/flexsds/core/hardware.py` consumed by `compatibility.py`, `cost_model.py`, `calibration.py`, `wizard.py`, `main.py`, and Modal server modules. Extend `serving_session.py` with a pure `build_api_config_payload()` helper and interactive export action compatible with `web/src/lib/endpoints.ts`'s `notesFromServingConfig()`, updated to honor `eagle_version`.

**Tech Stack:** Python 3.10+, pytest, Modal SDK 1.6.3 string GPU specifiers, TypeScript / Next.js / Vitest (`web/`).

**Spec:** `docs/superpowers/specs/2026-11-25-flexsds-modal-gpus-export-eagle-design.md`

## Global Constraints

- Modal GPU catalog is strictly 4 GPUs: `T4` (Low, 16 GB, `sm_75`, `$0.000164/s` `~$0.59/h`, Modal `"T4"`), `A10` (Medium, default, 24 GB, `sm_86`, `$0.000306/s` `~$1.10/h`, Modal `"A10"`), `L40S` (High, 48 GB, `sm_89`, `$0.000542/s` `~$1.95/h`, Modal `"L40S"`), `A100` (High, 80 GB, `sm_80`, `$0.000694/s` `~$2.50/h`, Modal `"A100-80GB"`).
- CLI label/flag displays `A100` (never `A100-80GB`), while footer `Info:` displays tier, VRAM (GB), SM, and pricing from `https://modal.com/pricing`.
- Never mutate byte contents of historical SHA-256 measured evidence (`src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`, `logs/`, `experiments/runs/`, `experiments/results/`, `docs/chuong-4-thuc-nghiem.md`, `docs/runbooks/release-0.2.2.md`).
- Never write credentials/secrets (`MODAL_TOKEN_*`, `HF_TOKEN`, `api_key`) into exported API config JSON files.
- Vietnamese prose in docs and commits must follow Vietnamese capitalization rules (no English Title Case); no Trello task ID required.
- All unit tests must run offline without physical GPUs or live network calls.

## Review Focus

1. **A100 40 GB vs 80 GB cross-matching:** A 40 GB `NVIDIA A100-SXM4-40GB` profile in `~/.flexsds/profiles/` must be rejected on the host during `load_profile_for_hardware(..., strict=True)` and `calibration.py:build_profile()` when Modal `A100` (80 GB) is selected, while `NVIDIA A100-SXM4-80GB` matches and M4 `strict=False` on `sm_80` still raises `FileNotFoundError` when `profile_sm80.json` is absent. (Pinned in Task 1.)
2. **L4 vs L40S `sm_89` collision:** Both `L4` and `L40S` share `sm_89`; `matches_gpu_label("L40S", "NVIDIA L4")` and `load_profile_for_hardware(L40S, strict=True)` must never accept an `L4` profile. (Pinned in Task 1.)
3. **Exported JSON path & vanilla/EAGLE-1/2/3/3.1 fidelity:** Exporting config JSON from `present_session()` when `config` comes from flag mode (`--vanilla`, `--eagle-version 1/2`, EAGLE-3, EAGLE-3.1) or contains extra/secret keys must never crash, must strip secrets, must not terminate the serving process, and must round-trip cleanly into `web/src/lib/endpoints.ts:notesFromServingConfig()`. (Pinned in Task 3 & Task 4.)
4. **Hardcoded chat banner regression:** `InteractiveChatSession.start()` must never claim `EAGLE-3.1 (HardwareAdaptive)` when serving EAGLE-1, EAGLE-2, EAGLE-3, vanilla, or an unknown external endpoint. (Pinned in Task 3.)
5. **CLI–Web contract drift:** `tests/test_web_cli_sample.py` must pass after normalizing `SERVE_SAMPLE` in `web/src/components/EndpointsPage.tsx` from `--gpu A10G` to `--gpu A10`, and `serve --gpu L4` / `--gpu A10G` must be cleanly rejected by the parser for Modal. (Pinned in Task 2 & Task 4.)

---

### Task 1: Centralized Modal GPU Catalog & Hardware/Profile Matching

**Files:**
- Modify: `src/flexsds/core/hardware.py:8-92`
- Modify: `src/flexsds/core/compatibility.py:460-489`
- Modify: `src/flexsds/core/cost_model.py:420-570, 630-655`
- Modify: `src/flexsds/core/calibration.py:44-50, 110-120`
- Create: `tests/test_modal_gpu_catalog.py`
- Modify: `tests/test_hardware_profiler_benchmark.py`, `tests/test_serving_profile_guard.py`

**Interfaces:**
- Produces (in `flexsds.core.hardware`):
  - `ModalGpuSpec(name: str, tier: str, vram_gb: int, vram_budget_mb: float, sm_architecture: str, modal_gpu: str, price_per_sec: float, price_per_hour: float, summary: str)`
  - `MODAL_GPU_SPECS: tuple[ModalGpuSpec, ...]`
  - `MODAL_GPU_NAMES: tuple[str, ...]` (`("T4", "A10", "L40S", "A100")`)
  - `DEFAULT_MODAL_GPU: str` (`"A10"`)
  - `MODAL_GPU_CHOICES: list[str]`
  - `get_modal_gpu_spec(gpu: str) -> ModalGpuSpec`
  - `modal_gpu_resource(gpu: str) -> str`
  - `modal_gpu_choice_label(spec: ModalGpuSpec) -> str`
  - `modal_gpu_footer(spec: ModalGpuSpec) -> str`
  - `remote_gpu_budget_mb(backend: str, target_gpu: str) -> float`
  - `build_remote_hardware_info(backend: str, target_gpu: str) -> HardwareInfo`
  - `HardwareInfo(..., gpu_match_label: Optional[str] = None)`
  - `matches_gpu_label(label: str, device_name: str) -> bool`

- [ ] **Step 1: Write failing tests in `tests/test_modal_gpu_catalog.py`**

Create `tests/test_modal_gpu_catalog.py` asserting:
- `MODAL_GPU_NAMES == ("T4", "A10", "L40S", "A100")` and `DEFAULT_MODAL_GPU == "A10"`.
- Tiers (`Low`, `Medium`, `High`, `High`), VRAM (`16`, `24`, `48`, `80`), SM (`sm_75`, `sm_86`, `sm_89`, `sm_80`), Modal resources (`"T4"`, `"A10"`, `"L40S"`, `"A100-80GB"`), and exact `modal.com/pricing` rates (`0.000164`, `0.000306`, `0.000542`, `0.000694`).
- `modal_gpu_choice_label` produces `"T4 [Low]"`, `"A10 [Medium] (Recommended)"`, `"L40S [High]"`, `"A100 [High]"` (display name is `"A100"`, not `"A100-80GB"`).
- `modal_gpu_footer` contains tier, `"{vram_gb} GB VRAM"`, `sm_*`, and pricing (`$.../s` and `$.../h`).
- `get_modal_gpu_spec` extracts `"A10"` from `"A10 [Medium] (Recommended)"` and `"A100"` from `"A100 [High]"`, and rejects `"L4"` / `"A100-40GB"` / `"A10G"` with `ValueError`.
- `matches_gpu_label`:
  - `matches_gpu_label("A10", "NVIDIA A10")` and `matches_gpu_label("A10G", "NVIDIA A10")` are `True`; `matches_gpu_label("A10", "NVIDIA RTX 3090")` is `False`.
  - `matches_gpu_label("L40S", "NVIDIA L40S")` is `True`; `matches_gpu_label("L40S", "NVIDIA L4")` is `False`.
  - `matches_gpu_label("A100-80GB", "NVIDIA A100-SXM4-80GB")` and `matches_gpu_label("A100-80GB", "NVIDIA A100 80GB PCIe")` are `True`; `matches_gpu_label("A100-80GB", "NVIDIA A100-SXM4-40GB")` is `False`.
- **Review Focus #1:** `build_remote_hardware_info("modal", "A100")` has `device_name == "Remote A100"`, `sm_architecture == "sm_80"`, `gpu_match_label == "A100-80GB"`; `load_profile_for_hardware(a100_hw, strict=True)` rejects a 40 GB `flexsds-component-v2` profile in `USER_PROFILES_DIR` and returns `(None, "")` (does not raise `FileNotFoundError`), while accepting an 80 GB `NVIDIA A100-SXM4-80GB` profile; `calibration.build_profile()` with `target_gpu="A100"` rejects `NVIDIA A100-SXM4-40GB` and accepts `NVIDIA A100-SXM4-80GB`.
- **Review Focus #2:** `load_profile_for_hardware(build_remote_hardware_info("modal", "L40S"), strict=True)` returns `(None, "")` even when `profile_sm89_l4.json` or a measured `NVIDIA L4` profile exists.
- Bundled `profile_sm_86_remote_a10g_fcda5486b12d.json` loads and validates cleanly with `build_remote_hardware_info("modal", "A10")`.

- [ ] **Step 2: Run test to verify RED**

Run: `PYTHONPATH=src:. pytest tests/test_modal_gpu_catalog.py -v`
Expected: FAIL (`ImportError` for `MODAL_GPU_SPECS` / `build_remote_hardware_info`).

- [ ] **Step 3: Implement catalog & matching updates in `hardware.py`, `compatibility.py`, `cost_model.py`, `calibration.py`**

- In `src/flexsds/core/hardware.py`: add `gpu_match_label: Optional[str] = None` to `HardwareInfo`, add `ModalGpuSpec`, `MODAL_GPU_SPECS`, `MODAL_GPU_NAMES`, `DEFAULT_MODAL_GPU`, `MODAL_GPU_CHOICES`, `get_modal_gpu_spec`, `modal_gpu_resource`, `modal_gpu_choice_label`, `modal_gpu_footer`, `COLAB_GPU_SPECS` (`T4`, `L4`, `A100`), `remote_gpu_budget_mb`, `build_remote_hardware_info`, and harden `matches_gpu_label` (`NVIDIA`/`TESLA`/`REMOTE` stripping, `A10`/`A10G`, `L40S` exact, `A100-80GB` requiring `A100` + `80GB`, `A100-40GB` requiring `A100` + `40GB`, bare `A100` matching `A100*`).
- In `src/flexsds/core/compatibility.py`: replace hardcoded `budgets` dict in `evaluate_serving_preflight` with `remote_gpu_budget_mb(backend, target_gpu)`.
- In `src/flexsds/core/cost_model.py`:
  - In `validate_measured_profile`: use `expected_label = hardware.gpu_match_label or hardware.device_name.removeprefix("Remote ")`.
  - In `load_profile_for_hardware`: for `sm_arch == "sm_80"` when `strict=True`, check packaged `PROFILES_DIR.glob("profile_sm*80*.json")` via `load_checked` (catching `ValueError`) and return `(None, "")` if none matches; preserve `FileNotFoundError` when `strict=False`.
  - Update error hint in `run_hardware_micro_benchmark` to `--backend modal --gpu A10`.
- In `src/flexsds/core/calibration.py`: in `measure_cell_group` and `build_profile`, resolve expected GPU label via `modal_gpu_resource(expected)` when `expected in MODAL_GPU_NAMES` (so `"A100"` enforces `"A100-80GB"` while legacy `"A10G"` passes through to `matches_gpu_label`).
- Update `tests/test_hardware_profiler_benchmark.py` and `tests/test_serving_profile_guard.py` for `A10` canonical naming (keeping narrow legacy `A10G` alias test in `test_hardware_profiler_benchmark.py` and passing `--gpu T4` in `test_eagle_adaptive_still_requires_empirical_profile`).

- [ ] **Step 4: Run targeted tests to verify GREEN**

Run: `PYTHONPATH=src:. pytest tests/test_modal_gpu_catalog.py tests/test_hardware_profiler_benchmark.py tests/test_serving_profile_guard.py tests/test_m4_a100_profile.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit Task 1**

---

### Task 2: Modal Runners, CLI Flags, and Interactive Wizard UI

**Files:**
- Modify: `src/flexsds/server/modal_app.py:44-47, 97-116`
- Modify: `src/flexsds/server/modal_calibration.py:11-18`
- Modify: `src/flexsds/server/modal_calibration_app.py:15-31`
- Modify: `src/flexsds/server/modal_runner.py:73-83`
- Modify: `src/flexsds/cli/wizard.py:58-70, 172-218, 444-455, 530-544, 662-684`
- Modify: `src/flexsds/cli/main.py:79-92, 137-140, 169-175, 362-375, 429-480, 529-547`
- Modify: `tests/test_cli_wizard_flow.py`, `tests/test_modal_runner_cli.py`, `tests/test_modal_app_container_paths.py`, `tests/test_cli_serving_readiness.py`, `tests/test_matched_execution_contract.py`, `tests/test_r6_geometric_mle_and_telemetry_io.py`, `tests/test_eagle_inspection.py`, `tests/test_end_to_end_flow.py`, `tests/test_speculative_method.py`

**Interfaces:**
- Consumes: `MODAL_GPU_SPECS`, `MODAL_GPU_NAMES`, `DEFAULT_MODAL_GPU`, `MODAL_GPU_CHOICES`, `get_modal_gpu_spec`, `modal_gpu_resource`, `modal_gpu_footer`, `build_remote_hardware_info` from `flexsds.core.hardware`.
- Produces:
  - `modal_app.py`: `GPU` (canonical FlexSDS GPU name, default `"A10"`), `MODAL_GPU = modal_gpu_resource(GPU)` passed to `@app.function(gpu=MODAL_GPU, ...)`.
  - `modal_runner.py`: forwards `"eagle_version"` inside `FLEXSDS_SERVING_CONFIG`.
  - `wizard.py`: `MODAL_GPU_CHOICES` with `choice_description()` returning `modal_gpu_footer(spec)` for every Modal GPU choice, and Colab footer noting GPU selection does not provision Colab hardware.
  - `main.py`: `serve --gpu` and `profile --gpu` accepting `["T4", "A10", "L40S", "A100"]` (default `"A10"`), and `cfg["serving"]` carrying `backend`, `hardware`, and `eagle_version`.

- [ ] **Step 1: Write/update failing tests for Modal runners, wizard menu/footers, and CLI flags**

Add tests in `tests/test_modal_gpu_catalog.py` (or `tests/test_cli_wizard_flow.py` & `tests/test_modal_app_container_paths.py`):
- `modal_app.py` with `target_gpu="A100"` calls `app.function(gpu="A100-80GB", ...)`, with `target_gpu="L40S"` calls `gpu="L40S"`, with `target_gpu="A10"` calls `gpu="A10"`, with `target_gpu="L4"` or `"A10G"` raises `ValueError`.
- `modal_calibration.py` accepts `T4`, `A10`, `L40S`, `A100` and rejects `L4` / `A100-40GB`.
- `wizard.choice_description("Modal GPU:", choice)` includes tier (`Low`/`Medium`/`High`), VRAM (`16 GB`/`24 GB`/`48 GB`/`80 GB`), and `/s` + `/h` pricing for every choice in `MODAL_GPU_CHOICES`.
- `main.parse_args(["serve", "--gpu", gpu])` and `main.parse_args(["profile", "--gpu", gpu])` accept `T4`, `A10`, `L40S`, `A100` (default `"A10"`) and reject `L4` and `A10G` with exit code 2 (**Review Focus #5**).
- `ModalServingRunner().setup_environment({"target_gpu": "A10", "eagle_version": "3.1"})` preserves `"eagle_version": "3.1"` in `FLEXSDS_SERVING_CONFIG`.

- [ ] **Step 2: Run targeted tests to verify RED**

Run: `PYTHONPATH=src:. pytest tests/test_modal_gpu_catalog.py tests/test_cli_wizard_flow.py tests/test_modal_app_container_paths.py -v`
Expected: FAIL on new assertions.

- [ ] **Step 3: Implement Modal runner, wizard, and CLI updates + normalize `A10G` → `A10` in affected tests**

- Update `modal_app.py`, `modal_calibration.py`, `modal_calibration_app.py`, `modal_runner.py`.
- Update `wizard.py` (`MODAL_GPU_CHOICES`, `CHOICE_HELP`, `choice_description`, `run_profile_measurement`, `run_interactive_wizard` using `default_idx=1` for `A10` and `build_remote_hardware_info`).
- Update `main.py` (`examples`, `profile_parser`, `serve_parser`, `build_remote_hardware_info`, `cfg["serving"]` including `backend`, `hardware`, `eagle_version`).
- Update existing CLI/runner tests referencing `--gpu A10G` / `target_gpu='A10G'` / `A10G (Recommended - 24 GB VRAM)` to `A10`.

- [ ] **Step 4: Run targeted tests to verify GREEN**

Run: `PYTHONPATH=src:. pytest tests/test_modal_gpu_catalog.py tests/test_cli_wizard_flow.py tests/test_modal_runner_cli.py tests/test_modal_app_container_paths.py tests/test_cli_serving_readiness.py tests/test_matched_execution_contract.py tests/test_r6_geometric_mle_and_telemetry_io.py tests/test_eagle_inspection.py tests/test_end_to_end_flow.py tests/test_speculative_method.py tests/test_colab_wizard_config.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit Task 2**

---

### Task 3: Post-Setup API Config JSON Export & EAGLE-1/2/3/3.1 Chat/Session Sync

**Files:**
- Modify: `src/flexsds/cli/serving_session.py:1-65`
- Modify: `src/flexsds/cli/wizard.py:172-195`
- Modify: `src/flexsds/client/chat.py:69-98, 198-221`
- Modify: `tests/test_modal_session_ux.py`

**Interfaces:**
- Consumes: `get_modal_gpu_spec`, `MODAL_GPU_NAMES` from `flexsds.core.hardware`; `__version__` from `flexsds`.
- Produces:
  - `build_api_config_payload(endpoint: str, model: str, config: dict, deploy: bool = False) -> dict` in `flexsds.cli.serving_session`
  - `format_speculation_label(config: dict) -> Optional[str]` in `flexsds.cli.serving_session`
  - `InteractiveChatSession(..., speculation_label: Optional[str] = None)` and `run_interactive_chat(..., speculation_label: Optional[str] = None)` in `flexsds.client.chat`

- [ ] **Step 1: Write failing tests in `tests/test_modal_session_ux.py`**

Add tests asserting:
- **Review Focus #3:** `build_api_config_payload()` produces valid web-demo-compatible JSON for:
  - EAGLE-1 (`draft_method="eagle"`, `eagle_version="1"`, static & adaptive)
  - EAGLE-2 (`draft_method="eagle"`, `eagle_version="2"`, static & adaptive)
  - EAGLE-3 (`draft_method="eagle3"`, `eagle_version="3"`, static & adaptive)
  - EAGLE-3.1 (`draft_method="eagle3"`, `eagle_version="3.1"`, adaptive)
  - Vanilla (`disable_speculative=True` / `draft_model=None` → `draft_model=None`, `draft_method="none"`, `eagle_version=None`, `enable_adaptive=False`)
  - Contains `endpoint`, `served_model`, `target_gpu` (`A10`/`L40S`/`A100`/`T4`), clean `hardware.device_name` (`"NVIDIA A10"`, `"NVIDIA A100"`, etc.), `hardware.vram_gb`, `hardware.tier`.
  - Strips any accidental secret keys (`api_key`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`, `token`) from `config`.
- Interactive `present_session()` offers 3 choices including `'Export API config JSON (for web demo)'`; choosing it writes the JSON to the prompted path (default `serving_config.json`), prints the path and JSON summary, returns to the menu loop without calling `proc.terminate()`, and allows subsequent chat/watch-log actions.
- **Review Focus #4:** `InteractiveChatSession.start()` with `speculation_label="EAGLE-2 (Static)"` prints `EAGLE-2 (Static)` (not `EAGLE-3.1`), and with `speculation_label=None` does NOT print `EAGLE-3.1 (HardwareAdaptive)`.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src:. pytest tests/test_modal_session_ux.py -v`
Expected: FAIL (`AttributeError` / missing menu option / hardcoded banner).

- [ ] **Step 3: Implement export action in `serving_session.py`, help text in `wizard.py`, and dynamic label in `chat.py`**

- In `src/flexsds/cli/serving_session.py`: implement `format_speculation_label()`, `build_api_config_payload()`, add `'Export API config JSON (for web demo)'` to `present_session()`, write file UTF-8 with `indent=2`, print absolute path + JSON + web demo import guidance, handle `OSError` gracefully without crashing or terminating `proc`, and pass `speculation_label=format_speculation_label(config)` into `run_interactive_chat(...)`.
- In `src/flexsds/cli/wizard.py`: add `'Export API config JSON (for web demo)'` to `CHOICE_HELP`.
- In `src/flexsds/client/chat.py`: accept `speculation_label: Optional[str] = None` in `InteractiveChatSession.__init__` and `run_interactive_chat`; infer from local `serving_config.json` only when `endpoint` matches saved `endpoint`; print `| Speculation: ...` only when `speculation_label` is non-empty.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `PYTHONPATH=src:. pytest tests/test_modal_session_ux.py tests/test_end_to_end_flow.py tests/test_cli_output_encoding.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit Task 3**

---

### Task 4: Web Demo Config Parser (`eagle_version`) & GPU Normalization

**Files:**
- Modify: `web/src/lib/endpoints.ts:65, 119-140`
- Modify: `web/src/components/EndpointsPage.tsx:31-35`
- Modify: `web/src/__tests__/endpoints.test.ts`
- Modify: `web/src/__tests__/app.test.tsx`

**Interfaces:**
- Consumes: CLI exported JSON schema produced by Task 3 (`target_gpu`, `hardware.device_name`, `target_model`, `draft_model`, `draft_method`, `eagle_version`, `enable_adaptive`).
- Produces: updated `notesFromServingConfig()` and `GPU_SUGGESTIONS` in `web/src/lib/endpoints.ts`, and `SERVE_SAMPLE` in `web/src/components/EndpointsPage.tsx`.

- [ ] **Step 1: Write failing Vitest tests in `web/src/__tests__/endpoints.test.ts`**

Add tests asserting (**Review Focus #3**):
- `notesFromServingConfig()` maps CLI JSON with `draft_method: "eagle"` + `eagle_version: "1"` → `eagle1_static` / `eagle1_adaptive`.
- `notesFromServingConfig()` maps CLI JSON with `draft_method: "eagle"` + `eagle_version: "2"` → `eagle2_static` / `eagle2_adaptive`.
- `notesFromServingConfig()` maps CLI JSON with `draft_method: "eagle3"` + `eagle_version: "3"` → `eagle3_static` / `eagle3_adaptive`.
- `notesFromServingConfig()` maps CLI JSON with `draft_method: "eagle3"` + `eagle_version: "3.1"` (and `"EAGLE-3.1"`) → `eagle31_static` / `eagle31_adaptive`.
- `notesFromServingConfig()` cleans legacy `"Remote A10"` → `"NVIDIA A10"` and preserves `"NVIDIA L40S"`, `"NVIDIA A100"`, `"NVIDIA T4"`.
- `GPU_SUGGESTIONS` contains `"NVIDIA A10"`, `"NVIDIA L40S"`, `"NVIDIA A100"`, `"NVIDIA T4"` (and not `"NVIDIA A10G"`).

- [ ] **Step 2: Run Vitest to verify RED**

Run: `cd web && npm test -- --run src/__tests__/endpoints.test.ts`
Expected: FAIL (`eagle3_adaptive` returned instead of `eagle31_adaptive` / `eagle1_static` / `eagle2_adaptive`).

- [ ] **Step 3: Implement web parser and UI updates**

- Update `notesFromServingConfig()` in `web/src/lib/endpoints.ts` to derive the versioned method from `eagle_version` when present (normalizing `"1"`/`"2"`/`"3"`/`"3.1"` or `"EAGLE-*"` to `"eagle1"`/`"eagle2"`/`"eagle3"`/`"eagle3.1"`), fall back to `draft_method`, and strip `"Remote "` prefix into `"NVIDIA <gpu>"`.
- Update `GPU_SUGGESTIONS` in `web/src/lib/endpoints.ts` to `["NVIDIA A10", "NVIDIA L40S", "NVIDIA A100", "NVIDIA T4", "NVIDIA H100"]`.
- Update `SERVE_SAMPLE` in `web/src/components/EndpointsPage.tsx` from `--gpu A10G` to `--gpu A10`.
- Update `web/src/__tests__/app.test.tsx` (`NVIDIA A10G` → `NVIDIA A10`) and `web/src/__tests__/endpoints.test.ts`.

- [ ] **Step 4: Run web tests + CLI sample contract test to verify GREEN**

Run: `cd web && npm test -- --run` and `PYTHONPATH=src:. pytest tests/test_web_cli_sample.py -v`
Expected: All PASS (**Review Focus #5**).

- [ ] **Step 5: Commit Task 4**

---

### Task 5: Colab Future Work Document & Repo-Wide `A10G` → `A10` Normalization

**Files:**
- Create: `logs/future-work/2026-11-25-colab-mcp-api-automation.md`
- Modify: `serving_config.json`, `README.md`, `.agents/skills/flexsds-assistant/references/flexsds-cli.md`, `docs/runbooks/flexsds-cli-modal-calibration.md`, `docs/09-dac-ta-san-pham-flexsds.md`, `src/pipeline/README.md`, `scripts/run_node1_extractor.py`, `scripts/run_pipeline_e2e.py`, `scripts/node1_marks_experiment.py`, `experiments/modal_lanes.py`, `experiments/modal_ocr.py`, `experiments/modal_ocr_serve.py`, `experiments/modal_runner.py`, `tests/test_modal_runner_config.py`

- [ ] **Step 1: Create `logs/future-work/2026-11-25-colab-mcp-api-automation.md`**

Follow the exact structure of `logs/future-work/2026-11-16-shared-gpu-profile-registry.md` and Vietnamese capitalization rules:
- Ngày ghi nhận: `2026-11-25`, Trạng thái: Đề xuất cho tương lai.
- Mục tiêu, Phạm vi hiện tại (upload file `.ipynb`, lựa chọn GPU trong CLI không ảnh hưởng cấp phát thật), Thiết kế dự kiến với Colab MCP / Colab API, Điều kiện kỹ thuật & bảo mật/chính sách, Lộ trình đề xuất, Tiêu chí hoàn thành.

- [ ] **Step 2: Normalize active repo files from `A10G` to `A10`**

- Update `serving_config.json`, `README.md`, `.agents/skills/flexsds-assistant/references/flexsds-cli.md`, `docs/runbooks/flexsds-cli-modal-calibration.md`, `docs/09-dac-ta-san-pham-flexsds.md`, `src/pipeline/README.md`, `scripts/run_node1_extractor.py`, `scripts/run_pipeline_e2e.py`, `scripts/node1_marks_experiment.py`, `experiments/modal_ocr.py`, `experiments/modal_ocr_serve.py`, `experiments/modal_runner.py`, `experiments/modal_lanes.py` (update `SUPPORTED_GPUS` to include `"A10"` as standard and update `tests/test_modal_runner_config.py`).
- Preserve immutable measured evidence (`src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`, `logs/`, `experiments/runs/`, `experiments/results/`, `docs/chuong-4-thuc-nghiem.md`, `docs/runbooks/release-0.2.2.md`).

- [ ] **Step 3: Full-suite verification (Verification Before Completion)**

Run complete project suites:
- `PYTHONPATH=src:. pytest tests/ -q`
- `cd web && npm test -- --run`
- `git status` and `git diff --stat develop`

Expected: 0 failures across both Python and web suites; clean working tree after committing.

- [ ] **Step 4: Commit Task 5**
