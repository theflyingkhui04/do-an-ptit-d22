# Thiết kế kỹ thuật: chuẩn hóa danh mục GPU Modal, xuất cấu hình JSON cho web demo, củng cố EAGLE-1/2/3/3.1 và future work Colab

> **Tài liệu:** Thiết kế kỹ thuật chi tiết  
> **Nhánh làm việc:** `feat/modal-gpu-catalog-and-config-export` (tách từ `develop`)  
> **Ngày lập:** 2026-11-25  
> **Trạng thái:** Chờ phê duyệt

---

## 1. Mục đích và bối cảnh

Phiên bản FlexSDS hiện tại (`0.3.4`) cần được hoàn thiện cho giai đoạn production demo ở bốn điểm chính:

1. **Danh mục phần cứng Modal còn thiếu và chưa đồng nhất:** CLI hiện chỉ hardcode 3 GPU (`A10G`, `L4`, `T4`) rải rác ở 7 file nguồn. Trên thực tế, Modal không còn cấp phát `L4` ổn định, đã chuẩn hóa tên định danh `A10G` thành `A10`, và bản demo cần các lựa chọn theo 3 phân khúc phần cứng (**Low**, **Medium**, **High**) có kèm thông số VRAM và giá tham chiếu từ `https://modal.com/pricing` ngay tại footer chọn lựa của CLI.
2. **Thiếu luồng xuất cấu hình JSON sau khi dựng endpoint:** Màn hình tương tác sau khi khởi chạy xong endpoint ([serving_session.py](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/src/flexsds/cli/serving_session.py)) mới chỉ có 2 hành động (*Watch log* và *Realtime chat with deployed LLM*). Trong khi đó, web demo đã có nút *"Nhập từ serving_config.json"* ([EndpointEditDrawer.tsx](file:///home/duckhoi04/Documents/Repo/Do-an/do-an-ptit-d22/web/src/components/EndpointEditDrawer.tsx)), nhưng CLI chưa có bước xuất JSON sau khi có URL API thật, còn parser phía web lại bỏ qua trường `eagle_version`.
3. **Cần kiểm tra và đồng bộ hỗ trợ `EAGLE-1`, `EAGLE-2`, `EAGLE-3`, `EAGLE-3.1`:** Phần nhận diện và thực thi cốt lõi đã hỗ trợ cả 4 phiên bản, nhưng vẫn còn các điểm lệch ở tầng tích hợp (`modal_runner.py` không forward `eagle_version`, `chat.py` hardcode nhãn `EAGLE-3.1 (HardwareAdaptive)`, web parser map sai `EAGLE-1`/`EAGLE-2`/`EAGLE-3.1` thành `EAGLE-3`).
4. **Google Colab và ghi nhận hướng phát triển tương lai:** Backend Colab hiện chỉ đồng bộ file `notebooks/flexsds_colab_server.ipynb` để upload thủ công, nên việc chọn GPU trong CLI không tác động đến phần cứng thực tế. Cần ghi nhận rõ hướng tự động hóa qua Colab MCP / Colab API vào `logs/future-work/`.

---

## 2. Mục tiêu, phi mục tiêu và phản mục tiêu

### 2.1. Mục tiêu
1. Xây dựng một nguồn chân lý duy nhất (single source of truth) cho danh mục GPU Modal trong `src/flexsds/core/hardware.py` gồm đúng **4 cấu hình** theo 3 phân khúc:
   - **Low:** `T4` — 16 GB VRAM (`sm_75`), `$0.000164/s` (`~$0.59/h`), Modal resource `"T4"`.
   - **Medium:** `A10` — 24 GB VRAM (`sm_86`), `$0.000306/s` (`~$1.10/h`), Modal resource `"A10"` (mặc định / khuyến nghị).
   - **High:** `L40S` — 48 GB VRAM (`sm_89`), `$0.000542/s` (`~$1.95/h`), Modal resource `"L40S"`.
   - **High:** `A100` — 80 GB VRAM (`sm_80`), `$0.000694/s` (`~$2.50/h`), Modal resource `"A100-80GB"`.
2. Hiển thị đầy đủ 4 cấu hình này trong CLI (cả luồng *Launch serving* lẫn *Measure & available profile GPU*) với phân loại `Low` / `Medium` / `High`, tên `A100` (không ghi `A100-80GB` ở nhãn hiển thị), và **footer `Info:`** thể hiện rõ phân khúc, dung lượng VRAM và mức giá từ `https://modal.com/pricing`.
3. Chuẩn hóa toàn bộ thông tin đang hoạt động trong repository từ `A10G` sang `A10` (mã nguồn CLI/server, cờ `--gpu`, mặc định, gợi ý và ví dụ trên web demo, README, runbook hiện hành, `serving_config.json`, skill reference và unit test).
4. Sửa các lỗi tương thích GPU đi kèm khi mở rộng sang `L40S` và `A100`:
   - `matches_gpu_label()` khớp chính xác `A100-80GB` / `A100` với tên CUDA thật của NVIDIA A100 80GB, khớp `L40S` tách biệt hoàn toàn với `L4`, và vẫn giữ khả năng đọc profile đo thực nghiệm cũ `NVIDIA A10`.
   - `load_profile_for_hardware(..., strict=True)` cho `sm_80` trả `(None, "")` khi chưa có profile đo hợp lệ thay vì ném `FileNotFoundError` làm sập wizard, đồng thời vẫn giữ nguyên ràng buộc fail-closed cho chế độ `strict=False` (M4 invariant).
5. Bổ sung hành động thứ 3 trong màn hình sau khi setup xong endpoint: **xuất file config JSON của API vừa tạo** để nhập trực tiếp vào nút *"Nhập từ serving_config.json"* của web demo; đồng thời sửa `web/src/lib/endpoints.ts` để đọc đúng `eagle_version` (`EAGLE-1`, `EAGLE-2`, `EAGLE-3`, `EAGLE-3.1`) và cập nhật danh sách gợi ý GPU + lệnh mẫu CLI.
6. Đồng bộ chuỗi xử lý `EAGLE-1`, `EAGLE-2`, `EAGLE-3`, `EAGLE-3.1` giữa `modal_runner.py`, `main.py`, `serving_session.py`, `chat.py` và web demo; giữ nguyên nguyên tắc chỉ công nhận loader đã kiểm chứng trên vLLM `0.30.0`.
7. Tạo tài liệu future work tại `logs/future-work/2026-11-25-colab-mcp-api-automation.md` về hướng dùng Colab MCP / Colab API để tự động dựng môi trường phục vụ mà không cần upload notebook thủ công, đồng thời làm rõ ghi chú phần cứng Colab trên CLI.

### 2.2. Phi mục tiêu
- Không thêm hỗ trợ `A100-40GB` hay đưa `L4` vào danh sách GPU Modal.
- Không thêm nút push HTTP trực tiếp từ CLI sang backend web demo (đã chốt Phương án A: xuất file JSON tương thích nút import hiện có).
- Không sửa nội dung byte bên trong các bằng chứng đo lịch sử đã băm SHA-256 (`src/flexsds/profiles/profile_sm_86_remote_a10g_fcda5486b12d.json`, `logs/`, `experiments/runs/`, `experiments/results/`, báo cáo thực nghiệm chương 4 và ghi chú phát hành `release-0.2.2.md`) để bảo toàn tính toàn vẹn khoa học.
- Không giả định hoặc bật thêm loader EAGLE chưa có trong vLLM `0.30.0` (như Qwen2 legacy EAGLE-1/2 hay confidence-adaptive tree của EAGLE-2).
- Không triển khai tích hợp Colab MCP / Colab API ngay trong nhánh này (chỉ ghi tài liệu future work).

### 2.3. Phản mục tiêu
- Không để việc chọn `A100` trên FlexSDS vô tình cấp phát nhầm `A100-40GB` trên Modal (Modal quy định `gpu="A100"` là bản 40 GB, phải truyền `"A100-80GB"` khi tạo function Modal).
- Không để profile đo của `L4` (`sm_89`) bị áp nhầm sang `L40S` (cùng `sm_89`) hoặc profile 40 GB bị áp nhầm sang `A100` 80 GB (cùng `sm_80`).
- Không ghi bất kỳ token/secret nào (`MODAL_TOKEN_*`, `HF_TOKEN`, `api_key`) vào file config JSON xuất ra cho web demo.
- Không làm gãy bộ kiểm thử hiện tại (`pytest` và `vitest`) hoặc tạo tiến trình chạy ngầm khi người dùng chọn xuất JSON.

---

## 3. Ràng buộc kỹ thuật

- **Python / CLI:** Tương thích Python 3.10+, không phụ thuộc GPU vật lý hay gọi mạng thật trong toàn bộ unit test.
- **Modal SDK:** Phiên bản `1.6.3` hiện dùng; các chuỗi GPU hợp lệ phía Modal là `"T4"`, `"A10"`, `"L40S"`, `"A100-80GB"`.
- **vLLM runtime:** Ghim ở `vllm==0.30.0` (`src/flexsds/server/modal_image.py`).
- **Web demo:** Next.js + TypeScript + Vitest tại `web/`, ràng buộc đồng bộ CLI–web được kiểm tra bởi `tests/test_web_cli_sample.py` và `web/src/__tests__/endpoints.test.ts`.
- **Chính tả và quy chuẩn:** Tuân thủ quy tắc viết hoa tiếng Việt (không dùng English Title Case); không dùng mã task Trello cho nhánh và commit theo xác nhận mới nhất của người dùng.

---

## 4. Các phương án đã xem xét

### 4.1. Luồng đẩy cấu hình API từ CLI lên web demo
- **Phương án A (Được chọn):** Thêm hành động thứ 3 trong `present_session()` để xuất file JSON cấu hình API (mặc định `serving_config.json`, tùy chọn đường dẫn), tương thích trực tiếp với nút *"Nhập từ serving_config.json"* sẵn có trên `EndpointEditDrawer.tsx`, đồng thời sửa `notesFromServingConfig()` trên web để nhận đủ `eagle_version` (EAGLE-1/2/3/3.1) và GPU mới.
  - *Lý do chọn:* Khớp chính xác luồng nút nhập sẵn có của web demo, không cần cấu hình URL hay khóa truy cập web demo bên trong CLI, dùng ổn định cả khi demo offline/local lẫn online.
- **Phương án B (Loại):** Gọi trực tiếp HTTP API (`POST /api/endpoints`, `PATCH /api/endpoints/{id}`) từ CLI lên backend web demo.
  - *Đánh đổi:* Phát sinh cấu hình URL + access key web demo trong CLI và lỗi mạng/auth không cần thiết cho mục tiêu demo hiện tại.

### 4.2. Chuẩn hóa `A10` vs `A10G` trên Modal
- **Phương án 1 (Được chọn):** Chuẩn hóa toàn bộ danh mục, CLI, web và tài liệu hiện hành sang `A10` (một lựa chọn duy nhất ở nhóm Medium), ánh xạ `A100` → Modal `"A100-80GB"`, đồng thời vẫn để `matches_gpu_label` đọc được bằng chứng đo cũ (`NVIDIA A10` / `A10G`) mà không làm hỏng `raw_sha256`.
  - *Lý do chọn:* Phản ánh đúng bảng giá và định danh chính thức của Modal, tránh nhầm lẫn 2 gói giống hệt nhau lúc demo.
- **Phương án 2 (Loại):** Hiển thị cả `A10` và `A10G` thành 2 dòng riêng trong menu.
  - *Đánh đổi:* Dư thừa và dễ gây hiểu nhầm cho người dùng vì Modal cấp cùng một pool GPU với cùng đơn giá.

---

## 5. Thiết kế chi tiết các phần thay đổi

### 5.1. Danh mục GPU Modal tập trung và bộ khớp phần cứng (`src/flexsds/core/hardware.py`, `compatibility.py`, `cost_model.py`)

Định nghĩa một nguồn chân lý duy nhất trong `src/flexsds/core/hardware.py`:

| Tên chuẩn (CLI) | Phân loại | VRAM | Ngân sách preflight (MiB) | Kiến trúc SM | Chuỗi Modal `gpu=` | Giá Modal (`modal.com/pricing`) |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `T4` | `Low` | 16 GB | `15000.0` | `sm_75` | `T4` | `$0.000164/s` (`~$0.59/h`) |
| `A10` | `Medium` | 24 GB | `22000.0` | `sm_86` | `A10` | `$0.000306/s` (`~$1.10/h`) |
| `L40S` | `High` | 48 GB | `46000.0` | `sm_89` | `L40S` | `$0.000542/s` (`~$1.95/h`) |
| `A100` | `High` | 80 GB | `78000.0` | `sm_80` | `A100-80GB` | `$0.000694/s` (`~$2.50/h`) |

- **Các hàm dùng chung trong `hardware.py`:**
  - `ModalGpuSpec` và `MODAL_GPU_SPECS` (tuple có thứ tự `T4`, `A10`, `L40S`, `A100`), `DEFAULT_MODAL_GPU = "A10"`.
  - `MODAL_GPU_NAMES = ("T4", "A10", "L40S", "A100")`.
  - `get_modal_gpu_spec(gpu: str) -> ModalGpuSpec`: tra cứu theo tên chuẩn (không phân biệt hoa thường; tách tên từ nhãn menu dạng `"A10 [Medium] (Recommended)"`).
  - `modal_gpu_resource(gpu: str) -> str`: trả chuỗi cấp phát cho Modal (`"A100"` → `"A100-80GB"`).
  - `modal_gpu_choice_label(spec: ModalGpuSpec) -> str`:
    - `T4 [Low]`
    - `A10 [Medium] (Recommended)`
    - `L40S [High]`
    - `A100 [High]`
  - `modal_gpu_footer(spec: ModalGpuSpec) -> str`: hiển thị phân loại, **số VRAM**, SM và giá từ `https://modal.com/pricing` trong 1 dòng gọn cho footer `Info:`:
    - `T4`: `[Low] 16 GB VRAM (sm_75) · $0.000164/s (~$0.59/h) · Entry GPU; check dtype/VRAM.`
    - `A10`: `[Medium] 24 GB VRAM (sm_86) · $0.000306/s (~$1.10/h) · Default for 7B AWQ + EAGLE.`
    - `L40S`: `[High] 48 GB VRAM (sm_89) · $0.000542/s (~$1.95/h) · Balanced 48 GB LLM serving.`
    - `A100`: `[High] 80 GB VRAM (sm_80) · $0.000694/s (~$2.50/h) · Max VRAM (Modal A100-80GB).`
  - `remote_gpu_budget_mb(backend: str, target_gpu: str) -> float` và `build_remote_hardware_info(backend: str, target_gpu: str) -> HardwareInfo`: hỗ trợ 4 GPU Modal ở trên và thêm `L4` (`23000.0` MiB, `sm_89`) riêng cho backend `colab`.
  - *(Quyết định kỹ thuật đề xuất của người thiết kế)* Bổ sung trường tùy chọn `gpu_match_label: Optional[str] = None` (mặc định `None`, tương thích ngược 100% với mọi lời gọi `HardwareInfo(...)` hiện có). Khi dựng `HardwareInfo` cho Modal qua `build_remote_hardware_info("modal", target_gpu)`, giữ `device_name = f"Remote {spec.name}"` (ví dụ `"Remote A100"`, hiển thị nhất quán `"A100"` ở CLI/file config) nhưng gán `gpu_match_label = spec.modal_gpu` (tức `"A100-80GB"` cho `A100`).
  - Cập nhật `matches_gpu_label(label: str, device_name: str) -> bool`:
    - Chuẩn hóa tiền tố `NVIDIA `, `TESLA `, `REMOTE `.
    - `A10` / `A10G` chỉ khớp `A10` hoặc `A10G` (giữ tương thích bằng chứng cũ `profile_sm_86_remote_a10g_fcda5486b12d.json`).
    - `L40S` chỉ khớp `L40S` (không khớp `L4`).
    - `A100-80GB` chỉ khớp thiết bị có `A100` và `80GB` (ví dụ `NVIDIA A100-SXM4-80GB`, `NVIDIA A100 80GB PCIe`), **không** khớp `A100-SXM4-40GB`.
    - `A100` (không hậu tố, dùng ngoài bối cảnh Modal) khớp `A100` hoặc `A100-...` / `A100 ...`.
  - Đảm bảo chặn profile 40 GB nhất quán ở **cả host lẫn container**:
    - Trong `cost_model.py:validate_measured_profile()`, lấy nhãn đối chiếu `expected_label = hardware.gpu_match_label or hardware.device_name.removeprefix("Remote ")` → host sẽ từ chối ngay profile `NVIDIA A100-SXM4-40GB` trước khi khởi chạy Modal.
    - Trong `calibration.py` (`measure_cell_group` và `build_profile`), khi `config.get("target_gpu")` là tên Modal chuẩn (như `"A100"`), ánh xạ sang nhãn kiểm chứng thực tế qua `modal_gpu_resource(expected)` (`"A100-80GB"`), còn `"A10G"` trong bằng chứng cũ vẫn khớp hợp lệ qua `matches_gpu_label`.
    - Trong `modal_app.py`, container kiểm tra cả `matches_gpu_label(modal_gpu_resource(GPU), torch.cuda.get_device_name(0))` lẫn `matches_gpu_label(data["metadata"]["device"], torch.cuda.get_device_name(0))`.
- **Sửa `cost_model.py` (`load_profile_for_hardware`):**
  - Với `sm_80`: ở chế độ `strict=True`, duyệt các profile hợp lệ trong `USER_PROFILES_DIR` và `PROFILES_DIR` (bỏ qua file lỗi bằng `ValueError`) và trả `(None, "")` nếu không tìm thấy — không ném `FileNotFoundError` làm sập wizard. Ở chế độ `strict=False`, giữ nguyên hành vi ném `FileNotFoundError` như hợp đồng M4 (`tests/test_m4_a100_profile.py`).
  - Cập nhật thông báo lỗi trong `run_hardware_micro_benchmark` từ `--gpu A10G` sang `--gpu A10`.
- **Cập nhật các runner Modal (`modal_app.py`, `modal_calibration.py`, `modal_calibration_app.py`):**
  - Chấp nhận `MODAL_GPU_NAMES` (`T4`, `A10`, `L40S`, `A100`), mặc định `"A10"`.
  - Khi gọi `@app.function(gpu=...)`, luôn dùng `modal_gpu_resource(GPU)` (để `"A100"` gửi `"A100-80GB"` lên Modal).
  - Trong `modal_app.py` và `calibration.py`, kiểm tra `matches_gpu_label(modal_gpu_resource(GPU), torch.cuda.get_device_name(0))` và `matches_gpu_label(data["metadata"]["device"], torch.cuda.get_device_name(0))`.
- **Cập nhật CLI (`wizard.py`, `main.py`):**
  - `wizard.py`: cả `run_interactive_wizard()` và `run_profile_measurement()` dùng chung danh sách `MODAL_GPU_CHOICES` (default `default_idx=1`, tức `A10 [Medium] (Recommended)`), hiển thị đúng footer `modal_gpu_footer()` cho từng lựa chọn.
  - `main.py`:
    - `profile --gpu`: `choices=["T4", "A10", "L40S", "A100"]`, `default="A10"`.
    - `serve --gpu`: `choices=["T4", "A10", "L40S", "A100", "L4"]`? Không — vì `L4` chỉ còn dùng cho Colab hướng dẫn, ta đặt `serve --gpu` `choices=["T4", "A10", "L40S", "A100"]`, `default="A10"` (và chấp nhận `L4` nếu cần cho `--backend colab` hoặc từ chối rõ ràng khi `--backend modal --gpu L4`? Quyết định cho builder: để `serve --gpu` nhận `["T4", "A10", "L40S", "A100"]`, mặc định `"A10"`; nếu có test cũ dùng `--backend colab` thì Colab không cần `--gpu L4` vì bản thân Colab không cấp phát phần cứng từ CLI; tuy nhiên kiểm tra lại toàn bộ `tests/` trước khi chốt để không làm vỡ test Colab hợp lệ).
    - Chuẩn hóa các chuỗi ví dụ trong `main.py` sang `--gpu A10`.
    - Bảo đảm `cfg["serving"]` trong flag mode `serve` có đủ `backend`, `hardware`, `eagle_version` để bước xuất JSON ở `present_session()` có đủ dữ liệu giống wizard.

### 5.2. Xuất cấu hình JSON sau khi tạo endpoint và đồng bộ web demo

- **CLI (`src/flexsds/cli/serving_session.py`, `src/flexsds/cli/wizard.py`):**
  - Mở rộng menu trong `present_session()` thành 3 lựa chọn:
    1. `Watch log`
    2. `Realtime chat with deployed LLM`
    3. `Export API config JSON (for web demo)`
  - Bổ sung hàm thuần `build_api_config_payload(endpoint: str, model: str, config: dict, deploy: bool) -> dict` tạo JSON tương thích ngược hoàn toàn với `notesFromServingConfig()` của web demo và không chứa bất kỳ secret nào:
    - `flexsds_version`, `action: "serve"`, `backend`, `endpoint` (URL `/v1`), `served_model`, `target_model`, `draft_model`, `draft_method`, `eagle_version`, `enable_adaptive`, `max_num_seqs`, `max_model_len`, `target_gpu`, `modal_deploy`, `hardware` (`device_name`, `sm_architecture`, `vram_gb`, `tier`, `free_vram_mb`).
    - Khi chạy `--vanilla` (`disable_speculative=True` hoặc `draft_model=None`): `draft_model=None`, `draft_method="none"`, `eagle_version=None`, `enable_adaptive=False`.
    - `hardware.device_name` dùng tên sạch theo GPU đã chọn (ví dụ `"NVIDIA A10"`, `"NVIDIA L40S"`, `"NVIDIA A100"`, `"NVIDIA T4"`).
  - Khi chọn `Export API config JSON (for web demo)`:
    - Hỏi đường dẫn lưu (mặc định `"serving_config.json"`), ghi file JSON UTF-8 (`indent=2`).
    - In đường dẫn tuyệt đối của file đã xuất, hiển thị nội dung JSON và hướng dẫn ngắn gọn nhập vào web demo (*Add endpoint bằng URL → Edit → Nhập từ serving_config.json*).
    - Quay lại vòng lặp menu mà **không dừng server**.
- **Web demo (`web/src/lib/endpoints.ts`, `web/src/components/EndpointsPage.tsx`):**
  - Sửa `notesFromServingConfig()` trong `web/src/lib/endpoints.ts`:
    - Nếu có `draft_model`: ưu tiên xác định phiên bản từ `eagle_version` khi trường này tồn tại (chấp nhận `"1"`, `"2"`, `"3"`, `"3.1"`, `"EAGLE-1"`, `"EAGLE-2"`, `"EAGLE-3"`, `"EAGLE-3.1"` → ánh xạ thành `"eagle1"`, `"eagle2"`, `"eagle3"`, `"eagle3.1"` trước khi gọi `decodingOf`).
    - Nếu không có `eagle_version`: dùng `draft_method` như trước; riêng `"eagle1"`, `"eagle2"`, `"eagle3"`, `"eagle3.1"` hoạt động chính xác cho cả chế độ `static` và `adaptive`.
    - Làm sạch tiền tố `"Remote "` ở `hardware.device_name` nếu đọc file config phiên bản cũ (ví dụ `"Remote A10"` → `"NVIDIA A10"` hoặc giữ chuỗi người dùng ghi nếu đã có `"NVIDIA "` / tên khác).
  - Cập nhật `GPU_SUGGESTIONS` trong `web/src/lib/endpoints.ts`: `["NVIDIA A10", "NVIDIA L40S", "NVIDIA A100", "NVIDIA T4", "NVIDIA H100"]`.
  - Cập nhật `SERVE_SAMPLE` trong `web/src/components/EndpointsPage.tsx`: đổi `--gpu A10G` thành `--gpu A10`.

### 5.3. Củng cố nhất quán `EAGLE-1`, `EAGLE-2`, `EAGLE-3`, `EAGLE-3.1`

- `src/flexsds/server/modal_runner.py`: bổ sung `"eagle_version"` vào tuple `keys` trong `setup_environment()`.
- `src/flexsds/client/chat.py`: bỏ chuỗi hardcode `EAGLE-3.1 (HardwareAdaptive)` ở dòng 96; cho phép truyền tham số tùy chọn `speculation_label: Optional[str] = None` (hoặc suy ra từ `serving_config.json` khi URL trùng khớp), chỉ in nhãn suy đoán khi biết chính xác, không khẳng định sai nếu không có thông tin.
- `src/flexsds/cli/serving_session.py`: truyền `speculation_label` chính xác (EAGLE-1 / EAGLE-2 / EAGLE-3 / EAGLE-3.1 + Adaptive/Static, hoặc Vanilla) vào `run_interactive_chat(...)`.
- `src/flexsds/core/speculative_method.py`: giữ nguyên cơ chế xác thực dựa trên bằng chứng checkpoint (`2H` cho EAGLE-1/2, `3H` cho EAGLE-3, `3H` + 3 cờ + 3 tensor `fc_norm.{0,1,2}.weight` cho EAGLE-3.1) và giữ nguyên danh sách loader đã kiểm chứng với vLLM `0.30.0` (không thêm architecture chưa được xác minh trong vLLM `0.30.0`). Bổ sung bài test xuyên suốt cho đủ cả 4 phiên bản EAGLE-1, EAGLE-2, EAGLE-3, EAGLE-3.1 từ kiểm tra tương thích → sinh lệnh/env → xuất config JSON → web parser.

### 5.4. Tài liệu future work Colab và chuẩn hóa repo `A10G` → `A10`

- Tạo file `logs/future-work/2026-11-25-colab-mcp-api-automation.md` theo đúng phong cách của `logs/future-work/2026-11-16-shared-gpu-profile-registry.md`, trình bày rõ hiện trạng upload notebook thủ công, kiến trúc dự kiến dùng Colab MCP / Colab API, xác thực, xử lý vòng đời runtime/tunnel, rủi ro chính sách và tiêu chí hoàn thành.
- Trong `wizard.py` (`CHOICE_HELP`), ghi rõ ở footer khi chọn Colab rằng GPU đã chọn chỉ phục vụ ước lượng VRAM và hướng dẫn chọn runtime trên Colab, không tự cấp phát GPU.
- Chuẩn hóa các tham chiếu `A10G` đang dùng sang `A10` tại: `README.md`, `serving_config.json`, `.agents/skills/flexsds-assistant/references/flexsds-cli.md`, `docs/runbooks/flexsds-cli-modal-calibration.md`, `src/pipeline/README.md`, `experiments/modal_lanes.py` (thêm `"A10"`, cập nhật mặc định/comment; kiểm tra test liên quan để giữ tương thích script thí nghiệm), `experiments/modal_ocr.py`, `experiments/modal_ocr_serve.py`, `scripts/run_node1_extractor.py`, `scripts/run_pipeline_e2e.py`, `scripts/node1_marks_experiment.py`.

---

## 6. Những điểm để cho bước lập kế hoạch / triển khai quyết định

- Cấu trúc chính xác của hàm/module phụ trợ test và cách tách helper trong `hardware.py` / `serving_session.py`.
- Xử lý tương thích cho các bài test hiện có đang kiểm tra `experiments/modal_lanes.py` (ví dụ `test_modal_runner_config.py` vẫn test `SUPPORTED_GPUS` cho bộ runner thí nghiệm M3).
- Kiểm tra toàn bộ bộ test Python (`pytest`) và Web (`npm test` / `vitest`) sau khi sửa để đảm bảo 100% pass.
