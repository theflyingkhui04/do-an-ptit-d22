# FlexSDS CLI: calibration giả lập và khoảng trống tích hợp Modal serving

- **Mã vấn đề:** `PROB-CLI-MODAL-CALIBRATION-SERVING`
- **Ngày ghi nhận:** 14/11/2026
- **Trạng thái:** IMPLEMENTED / GPU SMOKE VERIFIED (B=1) — calibration, public endpoint và adaptive K/feedback đã kiểm chứng trên Modal A10G; grid mặc định B≤32 và đánh giá lossless/speedup còn ngoài lượt kiểm tra này.
- **Phạm vi:** Wizard/CLI → hardware profile → Modal → vLLM serving runtime.
- **Kịch bản tái hiện:** Laptop CPU-only, chọn Qwen2.5-7B-Instruct-AWQ + EAGLE-3.1, Modal A10G, development (`modal serve`), bật HardwareAdaptive, GPU chưa có profile, chọn calibrate rồi launch.

> Các mục 1–8 ghi lại lịch sử phát hiện và sửa lỗi. Kết quả hiện hành nằm ở **mục 9** và thư mục evidence đi kèm.

## 1. Kết quả xác minh

### P1 — Micro-benchmark sinh latency bằng công thức, không đo GPU/model thật

**Mức độ:** Critical — sai nguồn gốc dữ liệu chi phí dùng cho quyết định adaptive.

**Căn cứ:**
- `src/flexsds/cli/wizard.py`: nhánh thiếu profile gọi trực tiếp `run_hardware_micro_benchmark(serving_hw, ...)` trong tiến trình CLI trên laptop; không dispatch benchmark sang Modal.
- `src/flexsds/core/cost_model.py`, hàm `run_hardware_micro_benchmark`: bảng `arch_baselines` đặt sẵn A10G/sm_86 là target 16.5 ms, draft 3.8 ms.
- `ct = base_t * (1 + 0.025 * (b - 1))`, `cd = base_d * (1 + 0.020 * (b - 1))`; chi phí verify dùng hệ số cố định 0.008.
- CUDA probe chỉ chạy vài phép matmul cục bộ nếu có CUDA; không timing, không dùng kết quả để tính profile; exception bị bỏ qua. Không load target/draft model.
- File người dùng `~/.flexsds/profiles/profile_sm_86_remote_a10g.json` có đúng các giá trị trên, nhưng metadata ghi `micro_benchmark_calibrated` và các trường mang tên `median_ms`.

**Hệ quả:** Calibration hoàn tất gần như tức thì; profile analytical bị trình bày thành số đo thực nghiệm. Tên model trong JSON chỉ là nhãn, không chứng minh model được chạy.

**Sẽ sửa:**
- Loại bỏ việc báo calibrated/measured cho dữ liệu công thức; không cho synthetic profile qua gate empirical.
- Tách rõ local calibration và remote calibration. Modal phải chạy phép đo trên GPU được chọn, kiểm tra GPU thực tế, load đúng model/method và lưu kết quả về máy người dùng.
- Đánh giá tái sử dụng `experiments/modal_profiler.py`, `experiments/m4_cost_profiler.py` và schema hiện có; chưa mặc định chúng hỗ trợ đầy đủ A10G/EAGLE.
- Lưu nguồn đo, model/revision, GPU thực tế, batch/K/context, warmup/repeats, đơn vị và raw measurements hoặc tham chiếu bằng chứng. Ghi rõ thành phần nào đo, thành phần nào suy diễn.
- Lỗi cấp GPU/load model/đo phải được báo và dừng calibration; cho phép người dùng chọn static hoặc abort thay vì tạo profile thay thế.

### P2 — Strict profile gate chưa xác thực empirical cho A10G

**Mức độ:** High.

**Căn cứ:** `load_profile_for_hardware()` trong `src/flexsds/core/cost_model.py` có validation measured riêng cho sm_80; với sm_86, profile tìm thấy vẫn được load dù sinh từ hàm trên. Lookup dựa trên kiến trúc và thứ tự tên file chưa đủ xác nhận đúng model/GPU/workload.

**Sẽ sửa:**
- Áp dụng validation nguồn đo và giá trị latency cho các GPU được hỗ trợ; xác thực identity và phạm vi áp dụng của profile.
- Nhận diện profile synthetic cũ, báo lý do không hợp lệ và yêu cầu đo lại hoặc chuyển static. Không tự xóa profile của người dùng.
- Giữ các invariant A100/M4 hiện có; không mở rộng gate bằng cách làm yếu validation.

### P3 — Bật adaptive với EAGLE nhưng command vẫn dùng speculative budget cố định

**Mức độ:** High — giao diện và runtime không khớp.

**Căn cứ:** `LocalServingRunner.build_command()` chỉ gắn `AdaptiveAsyncScheduler` và lineage executor khi `enable_adaptive and not is_eagle`. Với cấu hình tái hiện, command có `num_speculative_tokens: 4` và không có `--scheduler-cls`, dù config lưu `enable_adaptive: true`.

**Sẽ sửa:**
- Làm rõ capability theo draft method và runtime, thống nhất lựa chọn wizard với command thực thi.
- Nếu EAGLE adaptive chưa được hỗ trợ/kiểm chứng, báo rõ và yêu cầu chọn static hoặc phương pháp được hỗ trợ; không âm thầm chạy static dưới nhãn adaptive.
- Chỉ bật EAGLE adaptive sau khi xác minh budget plumbing, feedback và lineage tương ứng. Không chỉ bỏ điều kiện `not is_eagle` để ép dùng scheduler hiện tại.
- Tôn trọng quyết định phạm vi scalar/tree trong báo cáo vấn đề ngày 12/11; không hứa hỗ trợ tree adaptive trong thay đổi này.

### P4 — Profile được “linked” ở CLI nhưng chưa được truyền sang container/runtime

**Mức độ:** High.

**Căn cứ:** `ModalServingRunner.setup_environment()` lọc cấu hình nhưng không truyền `profile_applied`; `modal_app.py` upload source và một manifest, không upload user profile. `LocalServingRunner.setup_environment()` chưa nối selected profile vào runtime consumer.

**Sẽ sửa:**
- Truyền artifact profile và metadata cần thiết sang container; dùng đường dẫn container hợp lệ thay cho đường dẫn laptop.
- Nối profile vào đúng cost-model consumer của adaptive runtime; kiểm tra runtime thực sự load artifact được chọn.
- Fail trước readiness khi bật adaptive mà profile thiếu/sai/không tương thích. Log identity/hash của profile thực dùng để đối chiếu.
- Kiểm tra các module phụ thuộc của scheduler/executor có được đóng gói đầy đủ trong image; hiện scheduler có import từ `experiments`, trong khi app chỉ upload một file manifest của thư mục này. Đây là rủi ro cần kiểm chứng riêng cho nhánh adaptive, chưa phải lỗi runtime đã tái hiện.

### P5 — Launch chưa chứng minh endpoint inference sẵn sàng

**Mức độ:** High đối với xác nhận thành công của luồng người dùng.

**Căn cứ:** `finish_launch()` chỉ thông báo process started rồi đợi tiến trình. `modal_app.py` dùng `@modal.web_server(..., startup_timeout=600)` và `Popen(vllm ...)`, nhưng CLI chưa kiểm tra HTTP health/model/completion. Log được cung cấp chỉ có created objects/URL, chưa có response inference. Cuối phiên có thông báo disconnect; chưa xác định nguyên nhân từ log này.

**Sẽ sửa:**
- Phân biệt trạng thái launch process, endpoint assigned, server ready và inference verified.
- Thêm kiểm tra endpoint có timeout, HTTP health/model và smoke completion phù hợp; lấy model phục vụ từ cấu hình hoặc `/v1/models`.
- Báo rõ startup failure/timeout và log liên quan; kiểm tra hành vi khi tiến trình vLLM thoát sớm.
- Làm rõ vòng đời development so với persistent deploy và shutdown khi ngắt CLI; không suy diễn nguyên nhân disconnect khi chưa có bằng chứng.

## 2. Thứ tự triển khai dự kiến

1. Chặn synthetic profile được nhận là empirical; sửa thông báo và gate capability EAGLE/static.
2. Nối remote calibration thực, schema/provenance và validation profile.
3. Đóng gói/truyền profile và nối consumer runtime; kiểm tra dependency container.
4. Bổ sung readiness/smoke inference và xử lý vòng đời process.
5. Cập nhật tài liệu CLI và regression tests theo hành vi cuối cùng.

## 3. Tiêu chí nghiệm thu

- Máy CPU-only không thể tự sinh profile được coi là measured của A10G.
- Calibration Modal có bằng chứng chạy trên GPU/model được chọn và có số đo thời gian thực; lỗi không tạo profile thành công.
- Profile synthetic cũ bị từ chối cho empirical adaptive; static/abort vẫn có thông báo rõ ràng.
- Config yêu cầu adaptive phải thực thi đúng capability hoặc báo không hỗ trợ; không im lặng chạy EAGLE k=4 cố định dưới nhãn adaptive.
- Runtime load đúng profile đã chọn, có log đối chiếu; thiếu/sai profile dừng trước readiness.
- Chỉ báo inference verified sau khi endpoint trả completion hợp lệ; tạo URL không đồng nghĩa model đã sẵn sàng.
- Test có ý nghĩa cho provenance, calibration dispatch/failure, artifact propagation, capability và readiness; mock test không được coi là bằng chứng GPU benchmark.

## 4. Phương pháp xác minh và giới hạn hiện tại

- Đã đọc README, skill Modal (khớp SDK 1.6.3), các file CLI/core/server liên quan, profile A10G thực tế và một số test/profiler.
- Đã dựng command và environment offline bằng Python từ `serving_config.json`: `adaptive requested=True`, `adaptive scheduler attached=False`, `profile forwarded=False`; EAGLE budget bằng 4.
- Chưa chạy benchmark GPU, chưa deploy app, chưa gọi endpoint live; chưa kết luận model serving hiện tại load thành công hay thất bại.
- Chưa sửa implementation hoặc profile người dùng. Các mục “Sẽ sửa” là kế hoạch, không phải kết quả đã hoàn thành.


## 5. Implementation ngày 14/11/2026

- Thay hàm sinh latency analytical bằng calibration vLLM worker-local: đồng bộ CUDA trước/sau timing, chỉ nhận đúng B/K/context, warmup 5 + repeats 50, median từ raw samples. Thiếu cell, GPU mismatch, lỗi model/worker hoặc latency không hữu hạn đều dừng, không có synthetic replacement.
- Thêm Modal calibration app tạm và dispatch qua Modal CLI để hỗ trợ hai môi trường pipx riêng. Mỗi K chạy engine subprocess riêng; resolve/pin model revisions; giữ evidence/hash trong JSON và ghi file atomic.
- Strict gate từ chối synthetic/legacy không có raw evidence tái tạo được (giữ validation A100/M4 riêng). Profile mới có identity suffix; không xóa profile cũ. `--allow-fallback` không vượt empirical serving gate.
- Wizard hỏi static/abort khi bật EAGLE adaptive; direct CLI và runner từ chối cấu hình EAGLE adaptive thay vì bỏ scheduler âm thầm.
- Truyền selected profile sang Modal, rewrite đường dẫn trong container, kiểm tra actual GPU và model identity, nối matrix vào scalar planner, log hash. Consumer không dùng generic heuristic cho K thiếu; B trung gian chọn điểm đo gần nhất. Adaptive component profile dùng eager/compilation mode 0; đây không phải bảo đảm speedup end-to-end.
- Upload module phụ thuộc trong `experiments` cho scheduler/executor; giữ dữ liệu runs/results ngoài mount. Giới hạn serving app một container để readiness retry không mở rộng nhiều GPU đồng thời.
- Launch forward log Modal, lấy URL, gọi health/model discovery/completion trước khi báo `Inference verified`; timeout/exit/error trả failure và dừng tiến trình CLI con. Development thông báo rõ cần giữ terminal mở.
- Hướng dẫn sử dụng và giới hạn: [runbook CLI/Modal](../../docs/runbooks/flexsds-cli-modal-calibration.md).

### Giới hạn còn lại cần giữ rõ

- Compatibility gate hiện tại chỉ chấp nhận EAGLE, còn scheduler adaptive dành cho `draft_model`. Thay đổi này không mở rộng gate hoặc giả định EAGLE tương thích scheduler đó. Chưa có tổ hợp adaptive được CLI công bố là luồng serving đã kiểm chứng; EAGLE static là luồng hiện có.
- Calibration/profile propagation/HTTP readiness đã được kiểm tra bằng CPU regression và fixtures. Chưa chạy GPU benchmark hoặc live Modal inference, nên chưa đóng vấn đề dưới nhãn end-to-end verified. Raw samples trong unit tests là fixtures, không phải bằng chứng đo GPU.
- Model revision/GPU được kiểm tra, nhưng ma trận chỉ đo context=1024; các context khác chưa có grid riêng. Chi phí loại trừ target logits/sampling và scheduling overhead.

### Validation

- Bộ regression liên quan: 261 tests pass. Sau khi bổ sung compilation mode 0 và hoàn thiện kiểm tra runtime, chạy lại 115 tests trực tiếp bị ảnh hưởng: tất cả pass. Calibration cấp đủ token capacity cho toàn bộ prefill batch (tối đa B×1024), tránh tách batch làm mất shape đo.
- Modal SDK 1.6.3: `modal run src/flexsds/server/modal_calibration_app.py --help` load entrypoint thành công, không chạy GPU.
- `git diff --check` sạch. Các thay đổi có sẵn của người dùng trong workbook và `serving_config.json` được giữ nguyên.


## 6. Đồng bộ định hướng sản phẩm

Tên đề tài được xác nhận: **FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs**. EAGLE static chỉ là đường kiểm tra serving tạm thời; capability gate vừa sửa không giải quyết xong mục tiêu adaptive EAGLE. Phần còn thiếu là tích hợp quyết định K, EAGLE proposer, verification feedback và lineage trên cùng runtime, rồi kiểm chứng GPU. Các lane `draft_model` là đối chứng/nghiên cứu, không thay thế mục tiêu này. README, tên package/CLI và các tài liệu giới thiệu đã được đồng bộ; không công bố tree execution hoặc bảo đảm speedup tuyệt đối khi chưa có bằng chứng.


## 7. Tích hợp adaptive EAGLE vào CLI — cập nhật tiếp theo ngày 14/11/2026

Phần này thay thế nhận định “scheduler chỉ hỗ trợ draft_model” và “EAGLE adaptive chưa được nối vào CLI” ở các lần inspect trước. Đọc thêm nhánh M4 cho thấy repo đã có `eagle_adaptive` trong `experiments/m4_treatments.py`, contract sequential dynamic K và kiểm tra source EAGLE trong lineage executor. CLI được nối vào chính đường này, không xây lại feedback engine và không bỏ kiểm tra source.

### Thay đổi

- Runner bật dynamic speculative lookup `[[1,32,4]]`, async scheduling, `AdaptiveAsyncScheduler` và `LineageUniProcExecutor` cho adaptive EAGLE. Static vẫn giữ K=4 và không gắn scheduler tùy biến.
- Bỏ chặn EAGLE adaptive trong wizard/direct CLI. Thiếu profile vẫn phải calibrate, chọn static hoặc abort; synthetic profile không vượt strict gate.
- K được controller chọn từ measured matrix, scheduler chuyển cho proposer; acceptance feedback chỉ cập nhật qua lineage hợp lệ. Cơ chế giữ producer origin của M4 được tái sử dụng, không gắn feedback vào quyết định hiện tại một cách suy đoán.
- Bật step/startup telemetry và tạo thư mục trước launch. Giới hạn max_num_seqs ≤32, một GPU/worker, V1 runner và sequential padded EAGLE theo contract M4. Source không khớp hash phải fail trước readiness.
- Calibration chuyển schema **flexsds-component-v2**: luôn load drafter, kể cả K=0, dùng dynamic lookup cho K=0..4 và cộng latency cache-sync của EAGLE vào cost K=0. Source vLLM cho thấy EAGLE chạy forward để đồng bộ KV trước khi trả tensor nháp rỗng; không thể dùng vanilla không drafter làm chi phí K=0 của đường này. Profile v1 phải đo lại.
- README/runbook/đặc tả được cập nhật theo trạng thái tích hợp mới. Tree execution, lossless và speedup GPU không được suy ra từ CPU tests.

### Kiểm chứng

- Tải source vLLM v0.30.0 vào cache `/tmp`, kiểm tra khớp cả 19 hash của scalar/EAGLE manifests; không sửa hoặc ghi đè evidence nghiên cứu trong repo.
- Source contract kiểm tra EAGLE dùng shared proposer, scheduler truyền dynamic K và K=0 giữ cache-sync forward.
- Đã chạy 142 tests pass, 1 skip (thiếu CPU torch cho row-copy tensor test), gồm CLI/profile, scheduler transitions, stale/delayed feedback, startup binding và source contracts. Các test scheduler đã chạy với source cache qua `M4_VLLM_SOURCE_ROOT` thay vì bị skip vì thiếu capture cục bộ.
- Chưa chạy calibration hoặc adaptive inference trên GPU Modal thực. `Inference verified` vẫn chỉ chứng minh completion, không chứng minh dynamic K hay hiệu năng; cần đối chiếu step/lineage telemetry và kiểm tra lossless trong lần nghiệm thu GPU.


### Kết quả regression cuối

**345 tests pass, 1 skip** trong bộ CLI/profile/calibration, M4 treatments, exact-source contract, scheduler/lineage startup, planner/controller và serving runners. Test bị skip cần CPU torch để kiểm tra tensor row-copy; không bị thay bằng mock để tuyên bố GPU coverage. `git diff --check` sạch. Preflight giờ chặn cấu hình OOM/không tương thích trước khi yêu cầu profile. Benchmark subprocess nhận package path tuyệt đối để hoạt động khi thư mục làm việc thay đổi. Chưa commit các thay đổi tích hợp này và chưa xác minh trên GPU Modal thực.


## 8. Kiểm tra Modal thực sau lần tiếp tục

- Đã kết nối Modal và khởi chạy calibration thật; không dùng profile synthetic để vượt gate.
- Lượt đầu build layer pip riêng dừng ở tải wheel vLLM. Đã hủy app kiểm tra đó và đồng bộ recipe để dùng cache của serving; không giữ GPU song song.
- Lượt GPU tiếp theo phát hiện resource `A10G` báo `NVIDIA A10` qua CUDA. Đã thêm alias hẹp A10/A10G trong calibration/profile validation, giữ kiến trúc sm_86 và từ chối RTX 3090 cùng SM. Có regression cho identity và profile roundtrip với tên CUDA thực.
- Sau sửa alias, target và EAGLE head đã tới bước load trên A10; engine thất bại trong FlashInfer JIT vì image thiếu `nvcc` (`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`).
- Đã thay base `debian_slim` bằng shared CUDA development recipe tương ứng nhánh M4, pin vLLM 0.30.0; calibration và serving dùng chung helper `modal_image.py`. Có regression chống mất compiler base.
- Đang kiểm tra trước với batch 1 và đủ K=0..4, giữ nguyên warmup/repeats. Profile kiểm tra nằm trong `/tmp`, không ghi đè profile người dùng. Consumer chỉ cho phép profile này khi max_num_seqs=1; cấu hình mặc định 32 sequences vẫn yêu cầu đầy đủ các batch chuẩn.
- Chưa có kết luận calibration/serving thành công cho tới khi lượt GPU sau sửa image hoàn tất. Các app lỗi được dừng, không deploy persistent endpoint.


## 9. Kết quả kiểm chứng GPU thực — 15/11/2026

**Kết luận:** Đã kiểm chứng calibration thật và đường serving Hardware-Adaptive EAGLE trên Modal A10G ở **batch 1, max_num_seqs=1**. Không đóng phạm vi full-grid/performance chỉ từ smoke test này.

### Calibration

- Profile schema v2 đo đủ K=0..4, context=1024, warmup 5 + measured repeats 50 mỗi cell: **275 raw records**.
- JSON vượt `validate_measured_profile`; giữ đúng model revisions và raw SHA256. K=0 bao gồm cache-sync của drafter.
- Evidence: [measured_batch1_profile.json](evidence/2026-11-15-modal-a10g/measured_batch1_profile.json). Đây là profile kiểm tra riêng; không ghi đè profile synthetic cũ trong thư mục người dùng.

### Serving và adaptive feedback

- Endpoint Modal công khai trả thành công `/health`, `/v1/models`, `/v1/completions` và ba request `/v1/chat/completions`, mỗi request sinh đủ **128 output tokens**.
- Job GPU riêng dùng cùng `LocalServingRunner`, scheduler/executor và profile để trả trace trực tiếp trước khi đóng container. Cách này tránh việc Modal CLI non-PTY không trả stdout của lệnh thu file trong lượt kiểm tra hiện tại.
- **360 scheduled-budget records:** K=0 có 336 lần, K=2 có 15 lần, K=3 có 9 lần. EAGLE không chạy K=4 cố định.
- **24 actual draft-birth records:** K=2 có 15 lần, K=3 có 9 lần, khớp các nonzero scheduled K.
- **24 verification feedback records**, toàn bộ `lineage_status=attributed`. Tổng **57 drafted tokens / 25 accepted draft tokens** khớp giữa feedback, actual draft rows và response metrics.
- Startup trace xác nhận executor publication → scheduler consumption → scheduler initialization trong cùng binding; giữ nguyên source gate vLLM/EAGLE.
- Evidence: [public_serving_responses.json](evidence/2026-11-15-modal-a10g/public_serving_responses.json), [native_runtime_verification.json](evidence/2026-11-15-modal-a10g/native_runtime_verification.json), [verification_summary.json](evidence/2026-11-15-modal-a10g/verification_summary.json).

### Các sửa bổ sung từ test thực

- Dùng alias A10/A10G ở cả calibration, profile loader và runtime identity check; vẫn kiểm tra SM và từ chối GPU khác như RTX 3090.
- Dùng chung CUDA devel image có nvcc cho calibration/serving, thay base thiếu compiler khiến FlashInfer JIT thất bại.
- Chỉ chuyển source roots qua PYTHONPATH của Modal subprocess. Không chuyển sys.path/third-party packages của Python hệ thống sang pipx Modal; đã xác minh import watcher dùng typing_extensions trong đúng Modal venv.
- Profile batch nhỏ chỉ dùng với serving capacity được grid đo bao phủ. Không dùng profile B=1 để phục vụ cấu hình mặc định 32 sequences.

### Validation và phạm vi còn lại

- Regression cuối: **354 pass, 1 skip** (test tensor row-copy cần CPU torch). `git diff --check` sạch.
- Các app kiểm tra của lượt này đã dừng, không có persistent deployment được tạo. Model cache/image cache được giữ để tái sử dụng.
- Chưa chạy full grid B=1,2,4,8,16,32 trên GPU trong lượt này; chưa so token đầu ra với vanilla, chưa đo end-to-end speedup hoặc đưa ra bảo đảm lossless/không giảm tốc. Kết quả là functional GPU verification có phạm vi rõ ràng.
- Chưa commit các thay đổi tích hợp.
