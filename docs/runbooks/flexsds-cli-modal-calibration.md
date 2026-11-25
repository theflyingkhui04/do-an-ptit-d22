# FlexSDS CLI: kiểm tra calibration và Modal serving

**Đề tài:** FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs

CLI đã nối Hardware-Adaptive EAGLE theo đường scalar M4, có regression CPU/source contract. Đã có functional GPU verification ở batch 1 trên Modal A10; full-grid, lossless và speedup chưa được nghiệm thu trong lượt này.

## Hành vi sau sửa ngày 14/11/2026

Calibration và serving dùng chung CUDA development image `nvidia/cuda:13.0.3-devel-ubuntu22.04`, pin vLLM 0.30.0. FlashInfer JIT cần `nvcc`; chỉ cài torch/vLLM lên `debian_slim` chưa đủ cho đường chạy này. Xem [CUDA stack trên Modal](https://modal.com/docs/guide/cuda).

Modal resource `A10` (alias cũ `A10G`) báo tên CUDA `NVIDIA A10`. Kiểm tra `sm_86` và nhãn GPU vẫn giữ nguyên; RTX 3090 cùng kiến trúc không được coi là A10. Danh mục Modal chuẩn hiện có 4 cấu hình: `T4` (Low, 16 GB), `A10` (Medium, 24 GB), `L40S` (High, 48 GB) và `A100` (High, 80 GB — ánh xạ sang Modal `A100-80GB`).


- EAGLE-3 hỗ trợ cấu hình **adaptive scalar K=0..4**, sequential drafting, async scheduler và paired lineage executor. `--no-adaptive` chọn static K=4. Không hỗ trợ parallel drafting hoặc tree topology qua đường này.
- Calibration thật là một tác vụ riêng: chạy vLLM 0.30.0 trên GPU, warmup 5 lần và đo 50 lần cho mỗi ô B/K, có raw samples và observed shapes. Không còn tạo empirical profile từ hằng số.
- Grid mặc định: B=1,2,4,8,16,32; K=0,1,2,3,4; context=1024. Mỗi K dùng một tiến trình engine mới. Tác vụ có thể mất nhiều phút do cấp GPU, build image, tải model và khởi tạo engine.
- Chi phí là **target forward + draft proposal ở eager mode**, không gồm target logits/sampling hoặc scheduler overhead. Không dùng profile này để tuyên bố bảo đảm speedup end-to-end. Batch nằm giữa các điểm đo dùng batch đo gần nhất; context khác 1024 chưa có phép đo riêng.
- Model revision được resolve thành commit SHA khi đo và pin khi phục vụ adaptive. Profile khác model/GPU, raw evidence không đầy đủ hoặc matrix không tái tạo được từ raw đều bị từ chối.
- Profile analytical cũ, kể cả `profile_sm_86_remote_a10g.json` có nhãn `micro_benchmark_calibrated`, không đạt empirical gate. File cũ không bị tự xóa. Profile mới có hậu tố identity của cặp model/revision để tránh ghi đè profile của cặp model khác.
- CLI legacy `--allow-fallback` không cho analytical profile vượt empirical gate.

## Test EAGLE serving trên Modal A10

```bash
flexsds serve --backend modal --gpu A10 \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-adaptive
```

Hoặc chạy wizard, chọn Modal/A10 và tắt HardwareAdaptive để dùng static. Static không yêu cầu hardware profile.

Luồng thành công cần có `Inference verified: .../v1 (model=...)`. CLI gọi `/health`, `/v1/models`, rồi `/v1/completions` với một output token. URL được tạo hoặc process còn sống chưa đủ để báo ready. Lần kiểm tra này kích hoạt cold start thực của endpoint.

Development dùng `modal serve`: giữ terminal mở; Ctrl-C dừng phiên. `--deploy` dùng `modal deploy` cho deployment tồn tại sau khi CLI kết thúc. Tham khảo [Modal web server](https://modal.com/docs/sdk/py/latest/web_server) và [development lifecycle](https://modal.com/docs/guide/developing-debugging).

## Đo profile thực trên GPU Modal

```bash
flexsds profile --benchmark --backend modal --gpu A10 \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator
```

Lệnh dùng Modal CLI nên hỗ trợ trường hợp `modal` và `flexsds` ở hai môi trường pipx khác nhau. Cần Modal CLI và credentials đã cấu hình. Calibration dùng app tạm `flexsds-calibration`, không tạo serving endpoint. Vẫn sử dụng GPU thật và chi phí Modal trong thời gian chạy.

Khi thất bại, xem lỗi GPU/model/worker trong log; không có profile thành công thay thế. Khi thành công, JSON dưới `~/.flexsds/profiles` gồm metadata `schema=flexsds-component-v2`, raw evidence, SHA256, model revisions và matrix tái tạo từ median của các mẫu không warmup.

## Chạy adaptive EAGLE

Sau khi đo profile v2, chạy lại lệnh serving bên trên và bỏ `--no-adaptive`, hoặc bật HardwareAdaptive trong wizard. Thiếu profile sẽ được hỏi calibrate/static/abort. Profile phải khớp GPU, target, EAGLE head và model revisions; không có analytical fallback.

CLI dùng đường `eagle_adaptive` đã có trong `experiments/m4_treatments.py`: dynamic speculative lookup, `--async-scheduling`, `AdaptiveAsyncScheduler` và `LineageUniProcExecutor`. Executor kiểm tra hash source vLLM/EAGLE trước khi bật observation. Phản hồi acceptance chỉ được đưa về controller khi có lineage hợp lệ; callback xác thực hiện tại không bị gắn nhầm với quyết định K mới nhất.

Profile v2 đo cả EAGLE cache-sync forward tại K=0, dùng cùng drafter đã load và dynamic lookup. Profile v1 dùng vanilla K=0 chưa phản ánh chi phí này và phải đo lại. K=0 không sinh token nháp nhưng vẫn giữ cache drafter đồng bộ, không tương đương tắt toàn bộ EAGLE.

Telemetry mặc định gồm `telemetry/serving_trace.jsonl`, `serving_steps.jsonl`, `serving_lineage.jsonl` trong môi trường chạy server. Trên Modal, các file này nằm trong container; chưa được xuất thành artifact lâu dài. Kiểm tra scheduled K, draft producer và feedback lineage khi nghiệm thu. `Inference verified` chỉ xác nhận endpoint trả completion; không chứng minh chuyển K, lossless hoặc speedup.

Các giới hạn source/runtime là vLLM 0.30.0 đúng hash, V1 GPUModelRunner, một worker/GPU, sequential padded drafting và max_num_seqs ≤32. Profile phải chứa đầy đủ các điểm batch chuẩn tới cận phục vụ; profile chỉ đo batch 1 chỉ dùng với max_num_seqs=1, không thay thế grid mặc định 32 sequences. Tree-adaptive vẫn ngoài phạm vi này. Các profile A100/M4 legacy và bằng chứng nghiên cứu đã freeze không bị viết lại.

## Kiểm chứng hiện có

Regression tests kiểm tra CPU không tạo profile measured, remote dispatch/failure, raw evidence validation, artifact propagation, cost consumer, capability gate và HTTP readiness. Test CPU/mocks không phải bằng chứng benchmark GPU. Cần chạy lệnh calibration và serving thực ở trên để xác nhận image/model/worker trên tài khoản Modal và GPU mục tiêu.


## Bằng chứng kiểm tra ngày 15/11/2026

[Verification summary](../../logs/problems/evidence/2026-11-15-modal-a10g/verification_summary.json) lưu kết quả calibration B=1/K=0..4, 275 raw records, endpoint HTTP thành công, scheduled K=0/2/3 và 24 feedback có lineage. Tổng 57 draft / 25 accepted khớp giữa worker birth records, feedback và response metrics. Các artifact này không thay thế full-grid B≤32 hoặc benchmark lossless/speedup.


## Log Modal thu gọn

Calibration và serving mặc định hiện một dòng `[x%] tác vụ đang chạy (Ctrl+O for details)`, kèm app URL. Phần trăm theo các mốc khởi động và số cell đã đo; không tăng giả theo thời gian chờ GPU/warmup.

`Ctrl+O` mở khung log trong alternate screen; bấm lại để đóng và khôi phục màn hình chính, không để log vừa xem trên màn hình chính. Dùng Up/Down hoặc j/k để cuộn, G để về log mới nhất. File đầy đủ nằm ở `~/.flexsds/logs/modal-*.log` và được in đường dẫn khi bắt đầu. Khi output không phải terminal, CLI chỉ in các mốc tiến độ và đường dẫn log. Ctrl+C hủy tác vụ và trả terminal về chế độ bình thường.

### Ngắt tài khoản và dùng endpoint

`flexsds clear modal` xóa token Modal trong FlexSDS và profile Modal hiện hành ở máy này; giữ các profile khác. FlexSDS bỏ qua token cũ còn được shell truyền vào sau khi clear. Đăng nhập lại trong wizard, bằng `flexsds auth login --provider modal`, hoặc `modal setup`. Lệnh không thu hồi token trên cloud và không dừng app đang chạy.

Sau khi kiểm tra inference thành công, CLI hiển thị `Endpoint LLM created:` cùng API, model, GPU, mode và đường dẫn log, rồi cho chọn **Watch log**, **Realtime chat with deployed LLM**, hoặc **Export API config JSON (for web demo)** để xuất file JSON nhập vào web demo. Lệnh `flexsds chat --endpoint ... --model ...` được in sẵn để mở ở terminal khác. `/exit` rời chat về menu. Development serving cần giữ terminal hiện tại mở; Ctrl+C kết thúc phiên.
