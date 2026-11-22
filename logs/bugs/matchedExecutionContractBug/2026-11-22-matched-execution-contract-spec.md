# Đặc tả kỹ thuật: Chuẩn hóa hợp đồng thực thi tương đồng (matched execution contract) cho Static EAGLE và Vanilla target-only khi so sánh với Hardware-Adaptive

- **Mã tài liệu**: SPEC-MATCHED-EXEC-20261122
- **Ngày tạo**: 2026-11-22
- **Tác giả**: Đội ngũ phát triển FlexSDS (D22 PTIT)
- **Trạng thái**: Đã triển khai và kiểm thử xong trên CPU (Implemented & CPU-verified — chờ kiểm chứng GPU trên Modal A10G)
- **Liên kết kiểm toán**: Problem 1 / Mục 16 (`Adaptive ON/OFF không policy-isolated`) và Problem 6 (`Correctness / lossless chưa đóng`) trên nhánh `audit/flexsds-v023-hardwareadaptive-20261120`
- **Phạm vi tác động**:
  - `src/flexsds/server/local_runner.py` (`LocalServingRunner.build_command`, `LocalServingRunner.setup_environment`, `LocalServingRunner.launch`)
  - `src/flexsds/server/modal_runner.py` (`ModalServingRunner.setup_environment`, `ModalServingRunner.launch`)
  - `src/flexsds/server/modal_app.py` (xác thực truyền cấu hình và điều kiện kiểm tra profile)
  - `src/flexsds/cli/main.py` (`parse_args`, nhánh `serve`, in hợp đồng thực thi ở `--dry-run`)
  - `tests/test_matched_execution_contract.py` (tạo mới), `tests/test_local_runner.py`, `tests/test_modal_runner_cli.py`

---

## 1. Bối cảnh và mục đích

Trong báo cáo kiểm toán FlexSDS v0.2.3 (Mục 16 — `ON/OFF execution contract`), Đại đã chỉ rõ rằng hai nhánh chạy sẵn hiện tại (`enable_adaptive=True` và `--no-adaptive`) **không thể dùng để kết luận khoa học về hiệu quả của chính sách Hardware-Adaptive**, vì hai nhánh khác nhau đồng thời nhiều yếu tố hạ tầng thực thi của vLLM:

| Tham số thực thi | Adaptive ON hiện tại | Adaptive OFF (`--no-adaptive`) hiện tại | Tác động gây nhiễu |
|---|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `"0"` (V1 legacy runner) | Không thiết lập (mặc định V2 runner) | Hai kiến trúc model runner khác nhau hoàn toàn |
| `--enforce-eager` | Bật (tắt CUDA Graphs) | Tắt (bật CUDA Graphs) | CUDA Graphs ảnh hưởng cực mạnh tới độ trễ ở batch nhỏ (`B = 1..8`) |
| `--compilation-config` | `{"mode": 0}` (tắt `torch.compile`) | Không truyền (bật `torch.compile`) | Chênh lệch lớn về tốc độ thực thi kernel |
| `--async-scheduling` | Bật | Không truyền | Khác cơ chế chồng lấp CPU scheduling với GPU |
| Prefix caching | Mặc định của vLLM (bật) | Mặc định của vLLM (bật) | Gây nhiễu tính lặp lại khi chạy kiểm chứng lossless |
| Scheduler / Executor | `AdaptiveAsyncScheduler` + `LineageUniProcExecutor` | Native vLLM scheduler / executor | Phần cần tách riêng để đo lường |

Hệ quả trực tiếp:
1. **Không cô lập được thuật toán (no policy-isolation):** So sánh hiện tại chỉ là so sánh "cả gói Adaptive ON (eager/V1)" với "cả gói vLLM mặc định (compiled/CUDA Graphs/V2)". Ngay cả khi thuật toán chọn `K` theo `EmpiricalCostProfile` (R6) hoạt động đúng, Adaptive ON vẫn tự chịu bất lợi nặng do bị tắt CUDA Graphs và `torch.compile`.
2. **Không có đường CLI sạch để chạy bộ đối chứng 3 cánh tay (`Vanilla` vs `Static EAGLE` vs `Adaptive EAGLE`) trên cùng hợp đồng thực thi** cho kiểm chứng lossless (cache-off) và benchmark quét batch (`B = 1, 4, 8, 16, 32`).

Mục đích của đặc tả này là bổ sung chế độ **hợp đồng thực thi tương đồng (matched execution contract)** hoàn toàn opt-in cho `LocalServingRunner`, `ModalServingRunner` và CLI `flexsds serve` để:
1. Cho phép chạy **Static EAGLE** và **Vanilla (target-only)** trên **cùng chính xác nền tảng V1 + eager + `compilation_config={"mode": 0}` + `--async-scheduling`** (và tùy chọn `--no-prefix-caching`) như Adaptive ON.
2. Bảo toàn 100% hành vi mặc định của `--no-adaptive` (vẫn là vLLM default với CUDA Graphs/`torch.compile` cho người dùng thông thường và cho cột so sánh end-to-end trong báo cáo).
3. Chặn rò rỉ biến môi trường từ shell cha (như `VLLM_USE_V2_MODEL_RUNNER`, `ADAPTIVE_STRICT`, `FLEXSDS_PROFILE_PATH`) giữa các lần chạy đối chứng.

---

## 2. Phân tích nguyên nhân gốc rễ

### 2.1. Gom cứng cấu hình hạ tầng vào cờ `enable_adaptive` trong `local_runner.py`
- Tại `src/flexsds/server/local_runner.py:130-131`, hai cờ `--enforce-eager` và `--compilation-config {"mode": 0}` chỉ được thêm khi `enable_adaptive=True`.
- Tại `src/flexsds/server/local_runner.py:152-164`, `--async-scheduling` chỉ được thêm trong nhánh `enable_adaptive=True`, còn nhánh `else` không truyền.
- Tại `src/flexsds/server/local_runner.py:96-103`, luôn luôn tạo `--speculative-config` với `num_speculative_tokens = 4` cố định; chưa có đường chạy target-only (`disable_speculative=True`) hay đặt `static_k` tường minh (`3` hoặc `4`).

### 2.2. Dọn dẹp biến môi trường chưa triệt để ở nhánh `enable_adaptive=False`
- Tại `src/flexsds/server/local_runner.py:207-210`, khi `enable_adaptive=False`, hàm chỉ xóa `FLEXSDS_PROFILE_PATH`, nhưng **không xóa** `ADAPTIVE_STRICT`, `ADAPTIVE_STEP_TELEMETRY_PATH`, `ADAPTIVE_STARTUP_TELEMETRY_PATH`, `ADAPTIVE_CONTROLLER_KWARGS`, và không kiểm soát tường minh `VLLM_USE_V2_MODEL_RUNNER`.
- Nếu shell của người chạy từng export `VLLM_USE_V2_MODEL_RUNNER=0` từ phiên trước, phiên `--no-adaptive` mặc định sẽ vô tình chạy V1 runner mà không hề hay biết.

### 2.3. Thiếu chuyển tiếp cấu hình từ CLI xuống Modal container
- Tại `src/flexsds/server/modal_runner.py:76-77`, danh sách khóa `keys` được truyền qua `FLEXSDS_SERVING_CONFIG` chưa có các trường điều khiển hợp đồng thực thi (`match_adaptive_runtime`, `disable_speculative`, `disable_prefix_caching`, `static_k`).
- Tại `src/flexsds/cli/main.py:162-192` và `379-473`, lệnh `flexsds serve` chưa có cờ để chọn matched runtime, tắt đầu cơ (`--vanilla`), tắt prefix caching (`--no-prefix-caching`), hoặc chọn `static_k`, và `--dry-run` chưa in rõ nhãn hợp đồng thực thi.

---

## 3. Ba lớp so sánh khoa học được hỗ trợ sau khi triển khai

| Lớp so sánh | Cấu hình các cánh tay | Kết luận khoa học hợp lệ trong báo cáo |
|---|---|---|
| **Lớp 1 — Matched runtime (trọng tâm spec)** | `Vanilla (--vanilla --match-adaptive-runtime)` vs `Static EAGLE (--no-adaptive --match-adaptive-runtime)` vs `Adaptive EAGLE (R6)` — tất cả cùng V1 runner, eager, `compilation_config={"mode": 0}`, `--async-scheduling`, cùng revision | Cô lập ảnh hưởng của cơ chế **Adaptive K theo profile phần cứng** so với Static EAGLE và Vanilla trên cùng engine thực thi; dùng cho cả **kiểm chứng lossless** (kèm `--no-prefix-caching`) lẫn **benchmark batch sweep (`B = 1..32`)**. |
| **Lớp 2 — As-shipped end-to-end (giữ nguyên mặc định)** | `Adaptive EAGLE` vs `Static EAGLE (--no-adaptive mặc định)` vs `Vanilla (--vanilla mặc định)` | Đo chênh lệch thực tế so với vLLM default (có CUDA Graphs + `torch.compile` + V2 runner) để ghi nhận trung thực chi phí tích hợp hiện tại và định hướng tối ưu tương lai. |
| **Lớp 3 — Cache-off repeatability** | Thêm `--no-prefix-caching` đồng nhất cho mọi cánh tay trong Lớp 1 | Loại trừ hoàn toàn sai lệch do prefix cache khi kiểm tra `Vanilla A vs Vanilla B` và `Vanilla vs Static vs Adaptive`. |

---

## 4. Yêu cầu chức năng (Functional Requirements)

### 4.1. FR-01: Mở rộng `LocalServingRunner.build_command()` với 4 trường opt-in
Trong `src/flexsds/server/local_runner.py`, đọc từ `serving_cfg` 4 khóa tùy chọn mới (mặc định đều tắt/None để bảo toàn 100% tương thích ngược):
- `match_adaptive_runtime: bool = False`
- `disable_speculative: bool = False`
- `disable_prefix_caching: bool = False`
- `static_k: Optional[int] = None`

Quy tắc xác thực (fail-fast với `ValueError`):
1. Nếu `disable_speculative=True` mà `enable_adaptive=True` → `raise ValueError("Cannot enable adaptive scheduling when speculative decoding is disabled")`.
2. Nếu `static_k is not None`:
   - Nếu `enable_adaptive=True` hoặc `disable_speculative=True` → `raise ValueError("static_k is only valid for static speculative mode (enable_adaptive=False, disable_speculative=False)")`.
   - Kiểm tra `not isinstance(static_k, bool) and isinstance(static_k, int) and 1 <= static_k <= 8`, sai thì `raise ValueError("static_k must be an integer between 1 and 8")`.

Quy tắc sinh lệnh `vllm serve`:
1. **Nhóm cờ hợp đồng thực thi V1/eager:** `use_matched_eager = bool(enable_adaptive or match_adaptive_runtime)`.
   - Khi `use_matched_eager=True`: thêm `"--enforce-eager"`, `"--compilation-config"`, `json.dumps({"mode": 0})`, và `"--async-scheduling"`.
   - Khi `use_matched_eager=False`: **không** thêm các cờ này (giữ nguyên hành vi mặc định cũ).
2. **Prefix caching:**
   - Khi `disable_prefix_caching=True`: thêm `"--no-enable-prefix-caching"` vào lệnh cho mọi chế độ (Vanilla, Static, Adaptive).
3. **Chế độ đầu cơ:**
   - Khi `disable_speculative=True` (Vanilla target-only):
     - Không kiểm tra `draft_method`, không thêm `--speculative-config`, không thêm `--per-request-spec-decode-metrics`, không thêm `--scheduler-cls` và `--distributed-executor-backend`.
     - Vẫn giữ nguyên `--revision` của target model nếu `target_revision` hợp lệ (khác `"local"`).
   - Khi `disable_speculative=False` và `enable_adaptive=False` (Static EAGLE / Static draft):
     - Đặt `speculative_config["num_speculative_tokens"] = static_k if static_k is not None else 4`.
     - Không gắn `--scheduler-cls`, `--distributed-executor-backend`, không gắn `num_speculative_tokens_per_batch_size` cho EAGLE.
   - Khi `enable_adaptive=True` (Adaptive ON):
     - Giữ nguyên toàn bộ lệnh hiện tại (`--speculative-config` kèm `num_speculative_tokens_per_batch_size`, `--async-scheduling`, `--scheduler-cls`, `--distributed-executor-backend`, `--per-request-spec-decode-metrics`, ghim revision theo profile).

### 4.2. FR-02: Cách ly và đồng bộ biến môi trường trong `LocalServingRunner.setup_environment()`
1. Xác thực lại `disable_speculative` và `static_k` giống FR-01 (để không gọi nhầm `setup_environment()` trực tiếp với cấu hình mâu thuẫn).
2. Khi `enable_adaptive=True`:
   - Giữ nguyên hành vi hiện tại (`FLEXSDS_PROFILE_PATH`, `ADAPTIVE_STRICT="1"`, `VLLM_USE_V2_MODEL_RUNNER="0"`, các đường dẫn telemetry).
3. Khi `enable_adaptive=False`:
   - Xóa sạch mọi biến môi trường có thể rò từ shell cha: `FLEXSDS_PROFILE_PATH`, `ADAPTIVE_STRICT`, `ADAPTIVE_STEP_TELEMETRY_PATH`, `ADAPTIVE_STARTUP_TELEMETRY_PATH`, `ADAPTIVE_CONTROLLER_KWARGS`.
   - Nếu `match_adaptive_runtime=True`: đặt tường minh `env["VLLM_USE_V2_MODEL_RUNNER"] = "0"`.
   - Nếu `match_adaptive_runtime=False`: xóa `env.pop("VLLM_USE_V2_MODEL_RUNNER", None)` để đảm bảo baseline mặc định thực sự chạy đúng default của vLLM.
4. Bổ sung helper thuần túy `describe_execution_contract(config: Dict[str, Any]) -> str` trên `LocalServingRunner` (hoặc hàm module) trả nhãn rõ ràng:
   - `"adaptive-eager-v1"` (khi `enable_adaptive=True`)
   - `"vanilla-matched-eager-v1"` / `"vanilla-default"` (khi `disable_speculative=True`)
   - `"static-matched-eager-v1"` / `"static-default"` (khi `enable_adaptive=False` và `disable_speculative=False`)
   - Kèm hậu tố `"+no-prefix-cache"` nếu `disable_prefix_caching=True`.

### 4.3. FR-03: Chuyển tiếp đầy đủ qua `ModalServingRunner` và `modal_app.py`
1. Trong `src/flexsds/server/modal_runner.py:76-77`, bổ sung 4 khóa mới vào tuple `keys`:
   `"match_adaptive_runtime"`, `"disable_speculative"`, `"disable_prefix_caching"`, `"static_k"`.
2. Trong `ModalServingRunner.launch()`:
   - Gọi kiểm tra `LocalServingRunner().build_command(config)` và `LocalServingRunner().setup_environment(config)` sớm trước khi khởi chạy Modal không chỉ cho `enable_adaptive=True` mà cả khi có bất kỳ cờ đối chứng mới nào (`match_adaptive_runtime`, `disable_speculative`, `static_k is not None`), đảm bảo phát hiện lỗi cấu hình ngay trên máy trạm trước khi tốn thời gian build/deploy Modal.
   - Ở chế độ `dry_run=True`, in thêm dòng `Execution contract: <describe_execution_contract(config)>` và lệnh vLLM bên trong container (hoặc tóm tắt cấu hình đã chuyển tiếp).
3. Trong `src/flexsds/server/modal_app.py`:
   - Khi `enable_adaptive=False` (kể cả khi `match_adaptive_runtime=True` hoặc `disable_speculative=True`), không bắt buộc nạp `hardware_profile.json` và không bắt buộc kiểm tra manifest trong `experiments/`.
   - Container vẫn gọi `LocalServingRunner().build_command(runtime_config)` và `setup_environment(runtime_config)` nên tự động nhận đủ các cờ matched execution và `VLLM_USE_V2_MODEL_RUNNER="0"`.

### 4.4. FR-04: Bổ sung cờ CLI tường minh trong `src/flexsds/cli/main.py`
Thêm 4 cờ tùy chọn cho subcommand `flexsds serve`:
1. `--match-adaptive-runtime` (`action="store_true"`): Chạy Static EAGLE hoặc Vanilla trên cùng hợp đồng thực thi V1/eager với Adaptive ON để so sánh công bằng.
2. `--vanilla` (`action="store_true"`): Chạy chế độ Target-only (tắt speculative decoding, tự động ngụ ý `--no-adaptive`).
3. `--no-prefix-caching` (`action="store_true"`): Tắt prefix caching của vLLM cho các phép đo repeatability và lossless.
4. `--static-k` (`type=positive_int`, `default=None`): Đặt số token nháp tĩnh cho chế độ Static (`1..8`, mặc định `4`).

Quy tắc kiểm tra cờ CLI (trả exit code `2` kèm thông báo lỗi rõ ràng, không chạy preflight mạng vô ích):
1. Nếu `--match-adaptive-runtime` được truyền mà không có `--no-adaptive` và không có `--vanilla`:
   → In `"Invalid flag combination: --match-adaptive-runtime requires --no-adaptive or --vanilla."` và trả `2`.
2. Nếu `--static-k` được truyền cùng `--vanilla`, hoặc truyền mà không có `--no-adaptive`:
   → In `"Invalid flag combination: --static-k requires --no-adaptive and cannot be combined with --vanilla."` và trả `2`.
3. Nếu `--static-k` nằm ngoài `1..8`:
   → In `"Invalid --static-k: expected integer in 1..8."` và trả `2`.
4. Nếu `--vanilla` được truyền cùng `--draft`:
   → In `"Invalid flag combination: --vanilla cannot be combined with --draft."` và trả `2`.
5. Xử lý luồng `--vanilla`:
   - Chấp nhận khởi chạy trực tiếp khi có `--target` (hoặc `--dry-run` với `--backend`/`--gpu`) mà **không yêu cầu `--draft`**.
   - Bỏ qua bước kiểm tra tương thích draft model và bước nạp hardware profile; đặt `enable_adaptive = False`, `disable_speculative = True`, `draft_model = None`, `draft_method = "none"`, `profile_path = None`.
   - Nếu chạy `--backend local` mà không có CUDA GPU (và không phải `--dry-run`), vẫn giữ thông báo chặn hợp lý hoặc để preflight target báo rõ.
6. Khi `--dry-run`:
   - In thêm dòng `Execution contract: <describe_execution_contract(cfg)>` để người dùng và kiểm toán viên đối chiếu nhanh bằng mắt.

### 4.5. FR-05: Tương thích ngược tuyệt đối
- Mọi lệnh CLI cũ không truyền 4 cờ mới (`flexsds serve --target ... --draft ...`, `flexsds serve ... --no-adaptive`, wizard tương tác) giữ nguyên 100% lệnh vLLM sinh ra, biến môi trường và mã thoát.

---

## 5. Yêu cầu phi chức năng (Non-Functional Requirements)

- **NFR-01 (Minh bạch đối chứng):** Khi cùng truyền 1 target model, 1 draft model, cùng `target_revision` / `draft_revision`, `max_model_len`, `max_num_seqs`, `gpu_memory_utilization`, sự khác biệt duy nhất giữa lệnh Adaptive ON và lệnh Static matched (`--no-adaptive --match-adaptive-runtime`) chỉ được phép nằm ở đúng các cờ của lớp Adaptive (`--scheduler-cls`, `--distributed-executor-backend`, `num_speculative_tokens_per_batch_size`, và các biến môi trường `FLEXSDS_PROFILE_PATH` / `ADAPTIVE_*`).
- **NFR-02 (Không sửa nhánh audit cũ):** Triển khai trên nhánh phát triển mới (hoặc tiếp nối `fix/r6-hardware-cost-driven-budget`), tuyệt đối không sửa nhánh `audit/flexsds-v023-hardwareadaptive-20261120`.
- **NFR-03 (Kiểm thử không phụ thuộc mạng/GPU):** Toàn bộ unit test mới cho `LocalServingRunner`, `ModalServingRunner` và validation cờ CLI phải chạy độc lập 100% offline trên CPU.

---

## 6. Tiêu chí nghiệm thu (Acceptance Criteria)

1. **AC-01 — Cân bằng tuyệt đối giữa Adaptive ON và Static matched:** Unit test đối chiếu `build_command()` và `setup_environment()` của Adaptive ON (với profile A10G thật) và Static matched (`enable_adaptive=False, match_adaptive_runtime=True`, cùng revision):
   - Cả hai đều có `--enforce-eager`, `--compilation-config '{"mode": 0}'`, `--async-scheduling`, `VLLM_USE_V2_MODEL_RUNNER="0"`.
   - Static matched **không** có `--scheduler-cls`, `--distributed-executor-backend`, `num_speculative_tokens_per_batch_size`, `FLEXSDS_PROFILE_PATH`, `ADAPTIVE_STRICT`.
2. **AC-02 — Vanilla matched target-only:** Với `enable_adaptive=False, disable_speculative=True, match_adaptive_runtime=True, disable_prefix_caching=True`:
   - Lệnh có `--enforce-eager`, `--compilation-config '{"mode": 0}'`, `--async-scheduling`, `--no-enable-prefix-caching`, và `VLLM_USE_V2_MODEL_RUNNER="0"`.
   - Lệnh **không** có `--speculative-config`, `--per-request-spec-decode-metrics`, `--scheduler-cls`, `--distributed-executor-backend`.
3. **AC-03 — Cách ly biến môi trường shell cha:** Khi `os.environ` có sẵn `VLLM_USE_V2_MODEL_RUNNER="0"`, `ADAPTIVE_STRICT="1"`, `FLEXSDS_PROFILE_PATH="/tmp/x.json"`, gọi `setup_environment({"enable_adaptive": False})` (default OFF) phải xóa sạch cả 3 biến trên; còn khi `match_adaptive_runtime=True` thì giữ `VLLM_USE_V2_MODEL_RUNNER="0"` nhưng vẫn xóa `ADAPTIVE_STRICT` và `FLEXSDS_PROFILE_PATH`.
4. **AC-04 — Truyền thông suốt qua `ModalServingRunner`:** `FLEXSDS_SERVING_CONFIG` trong `ModalServingRunner.setup_environment()` chứa đủ `match_adaptive_runtime`, `disable_speculative`, `disable_prefix_caching`, `static_k` khi được cấu hình.
5. **AC-05 — Chặn tổ hợp cờ CLI sai:** Kiểm thử `cli.main()` trả exit code `2` cho 4 tổ hợp sai (`--match-adaptive-runtime` thiếu `--no-adaptive`/`--vanilla`; `--static-k` thiếu `--no-adaptive`; `--static-k` kèm `--vanilla`; `--vanilla` kèm `--draft`; `--static-k 9`), và trả `0` với `--vanilla --match-adaptive-runtime --no-prefix-caching --target ... --backend modal --gpu A10G --dry-run`.
6. **AC-06 — Không hồi quy:** Toàn bộ bộ test offline hiện hữu (bao gồm 15 test R6 trong `test_hardware_cost_driven_controller.py`, `test_local_runner.py`, `test_modal_runner_cli.py`, `test_modal_app_container_paths.py`) pass 100%.
