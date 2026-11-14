# FlexSDS CLI: kiểm tra calibration và Modal serving

**Đề tài:** FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs

Hardware-Adaptive EAGLE là mục tiêu sản phẩm. EAGLE static trong luồng hiện tại là bước kiểm tra serving tạm thời; tích hợp adaptive EAGLE vẫn là công việc còn thiếu.

## Hành vi sau sửa ngày 14/11/2026

- EAGLE-3 trong CLI hiện hỗ trợ **static k=4**. Bật adaptive sẽ được wizard hỏi chuyển static hoặc abort; direct CLI yêu cầu `--no-adaptive`. Calibration không tự bổ sung khả năng adaptive cho EAGLE.
- Calibration thật là một tác vụ riêng: chạy vLLM 0.30.0 trên GPU, warmup 5 lần và đo 50 lần cho mỗi ô B/K, có raw samples và observed shapes. Không còn tạo empirical profile từ hằng số.
- Grid mặc định: B=1,2,4,8,16,32; K=0,1,2,3,4; context=1024. Mỗi K dùng một tiến trình engine mới. Tác vụ có thể mất nhiều phút do cấp GPU, build image, tải model và khởi tạo engine.
- Chi phí là **target forward + draft proposal ở eager mode**, không gồm target logits/sampling hoặc scheduler overhead. Không dùng profile này để tuyên bố bảo đảm speedup end-to-end. Batch nằm giữa các điểm đo dùng batch đo gần nhất; context khác 1024 chưa có phép đo riêng.
- Model revision được resolve thành commit SHA khi đo và pin khi phục vụ adaptive. Profile khác model/GPU, raw evidence không đầy đủ hoặc matrix không tái tạo được từ raw đều bị từ chối.
- Profile analytical cũ, kể cả `profile_sm_86_remote_a10g.json` có nhãn `micro_benchmark_calibrated`, không đạt empirical gate. File cũ không bị tự xóa. Profile mới có hậu tố identity của cặp model/revision để tránh ghi đè profile của cặp model khác.
- CLI legacy `--allow-fallback` không cho analytical profile vượt empirical gate.

## Test EAGLE serving trên Modal A10G

```bash
flexsds serve --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator \
  --no-adaptive
```

Hoặc chạy wizard, chọn Modal/A10G rồi chọn static EAGLE khi được hỏi. Static không yêu cầu hardware profile.

Luồng thành công cần có `Inference verified: .../v1 (model=...)`. CLI gọi `/health`, `/v1/models`, rồi `/v1/completions` với một output token. URL được tạo hoặc process còn sống chưa đủ để báo ready. Lần kiểm tra này kích hoạt cold start thực của endpoint.

Development dùng `modal serve`: giữ terminal mở; Ctrl-C dừng phiên. `--deploy` dùng `modal deploy` cho deployment tồn tại sau khi CLI kết thúc. Tham khảo [Modal web server](https://modal.com/docs/sdk/py/latest/web_server) và [development lifecycle](https://modal.com/docs/guide/developing-debugging).

## Đo profile thực trên GPU Modal

```bash
flexsds profile --benchmark --backend modal --gpu A10G \
  --target Qwen/Qwen2.5-7B-Instruct-AWQ \
  --draft duckhoi/qwen2.5-7b-eagle3.1-speculator
```

Lệnh dùng Modal CLI nên hỗ trợ trường hợp `modal` và `flexsds` ở hai môi trường pipx khác nhau. Cần Modal CLI và credentials đã cấu hình. Calibration dùng app tạm `flexsds-calibration`, không tạo serving endpoint. Vẫn sử dụng GPU thật và chi phí Modal trong thời gian chạy.

Khi thất bại, xem lỗi GPU/model/worker trong log; không có profile thành công thay thế. Khi thành công, JSON dưới `~/.flexsds/profiles` gồm metadata `schema=flexsds-component-v1`, raw evidence, SHA256, model revisions và matrix tái tạo từ median của các mẫu không warmup.

## Phạm vi adaptive

Runner đã nối profile vào cost consumer của scalar `draft_model` scheduler và đóng gói các module `experiments` mà scheduler/executor import. Điều này không mở rộng compatibility gate của bản phát hành: gate hiện tại vẫn chỉ chấp nhận kiến trúc EAGLE, còn EAGLE adaptive chưa được hỗ trợ. Vì vậy chưa có tổ hợp adaptive được CLI phát hành như một luồng serving đã kiểm chứng. Không ép bỏ gate hoặc gắn scheduler `draft_model` vào EAGLE.

Các profile A100/M4 legacy và luồng nghiên cứu riêng được giữ nguyên; empirical profile mới của CLI không thay thế bằng chứng nghiên cứu đã freeze.

## Kiểm chứng hiện có

Regression tests kiểm tra CPU không tạo profile measured, remote dispatch/failure, raw evidence validation, artifact propagation, cost consumer, capability gate và HTTP readiness. Test CPU/mocks không phải bằng chứng benchmark GPU. Cần chạy lệnh calibration và serving thực ở trên để xác nhận image/model/worker trên tài khoản Modal và GPU mục tiêu.
