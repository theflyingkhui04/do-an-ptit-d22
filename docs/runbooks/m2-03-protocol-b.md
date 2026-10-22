# Runbook M2-03 Protocol B trên Kaggle/T4

Trạng thái chuẩn bị: **IMPLEMENTATION READY FOR GPU VERIFICATION**. Chưa chạy GPU.
Quyết định áp dụng: [PM ngày 17/10/2026](../../logs/problems/2026-10-16-m2-03-lossless-verification-repeatability.md#cập-nhật-quyết-định-pm--17102026).
B trước; C hoãn tới nền tảng khác. Đây là thu thập bằng chứng, không phải nghiệm thu M2-03.

## Điều kiện trước khi chạy

Chỉ thực hiện các lệnh dưới đây sau khi người dùng phê duyệt GPU execution và việc
chuyển mã đã review tới Kaggle. Việc commit/push/publish mã cần phê duyệt riêng;
nhiệm vụ cục bộ này không thực hiện các thao tác đó. Kaggle phải có checkout Linux
chứa implementation đã review tại một execution SHA, không có tracked edits.
Không dùng archive branch để chạy, không tự cài lại môi trường hoặc tải model.

Runtime có sẵn: Tesla T4, vLLM `0.30.0`, PyTorch `2.13.0+cu130`, torch CUDA `13.0`.
Hai model phải được cache đầy đủ. Trước khi chạy, chọn revision hiện có trong cache;
launcher truyền SHA rõ qua `--revision` và draft speculative config. Collector chỉ
chấp nhận revision được chứng minh trong argv của server, không suy ra từ refs.
Bật offline để không tải thêm model. Không chỉnh cache flags, seed, sampling,
max_tokens hoặc workload để tìm kết quả khớp. Không bật deterministic gating của C.

Chọn một T4 vật lý bằng UUID, kể cả khi host có hai T4. Chỉ một GPU được CUDA nhìn
thấy. Chỉ một server chạy tại một thời điểm. Bắt đầu khi không có server đang dùng
cho công việc khác: `switch_lane.sh` dừng các server hiện có. Không restart server
khỏe đang phục vụ mục đích khác chỉ để chuẩn bị thí nghiệm.

## Chuẩn bị phiên và lưu provenance

Chạy từ repository root trong **một Bash session** của Kaggle. Không chạy các cell
song song. Nếu lệnh bất kỳ lỗi, dừng, giữ toàn bộ thư mục kết quả; không retry ngầm,
không tiếp tục cell sau, không sửa kết quả hoặc xóa run thất bại.

```bash
set -euo pipefail
set -o noclobber
test "$(git branch --show-current)" = fix/m2-03-lossless-clean
test -z "$(git status --porcelain --untracked-files=no)"
git ls-files --error-unmatch experiments/protocol_b.py experiments/protocol_b_capture.py
EXECUTION_SHA=$(git rev-parse HEAD)
GPU_UUID=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
test "$(nvidia-smi -i "$GPU_UUID" --query-gpu=name --format=csv,noheader)" = 'Tesla T4'
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=$(python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)')
export PROTOCOL_B_TARGET_REVISION=$(python -c 'from experiments.protocol_b_capture import revision,TARGET; print(revision(TARGET))')
export PROTOCOL_B_DRAFT_REVISION=$(python -c 'from experiments.protocol_b_capture import revision,DRAFT; print(revision(DRAFT))')
[[ "$PROTOCOL_B_TARGET_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ "$PROTOCOL_B_DRAFT_REVISION" =~ ^[0-9a-f]{40}$ ]]
export VLLM_USE_V1=1
export ADAPTIVE_ENABLE_VRAM_GUARD=1
export ADAPTIVE_MAX_K=4
export ADAPTIVE_MIN_BATCH_K=0
export ADAPTIVE_PER_REQUEST=1
export ADAPTIVE_CONTROLLER_KWARGS=''
test -z "${CUBLAS_WORKSPACE_CONFIG:-}"
SESSION_ID=$(cat /proc/sys/kernel/random/uuid)
RUN_ROOT="/kaggle/working/m2-03-protocol-b-$(date -u +%Y%m%dT%H%M%SZ)-$SESSION_ID"
mkdir "$RUN_ROOT"
export ADAPTIVE_TELEMETRY_PATH="$RUN_ROOT/adaptive_telemetry.jsonl"
printf '%s\n' "$EXECUTION_SHA" > "$RUN_ROOT/execution_sha.txt"
printf '%s\n' "$SESSION_ID" > "$RUN_ROOT/session_id.txt"
printf '%s\n' "$GPU_UUID" > "$RUN_ROOT/gpu_uuid.txt"
nvidia-smi -q > "$RUN_ROOT/nvidia_before.txt"
python -m pip freeze > "$RUN_ROOT/packages.txt"
git status --short > "$RUN_ROOT/git_status.txt"
git ls-files src scripts tests 'experiments/*.py' experiments/configs > "$RUN_ROOT/source_files.txt"
tar -czf "$RUN_ROOT/execution_sources.tar.gz" -T "$RUN_ROOT/source_files.txt"
python -c 'from experiments.protocol_b import load_plan,write_new; import sys; write_new(sys.argv[1],load_plan())' "$RUN_ROOT/frozen_plan.json"
```

`load_plan` phải xác minh đủ 54 prompt: W1=30, W2=12, W3=12; SHA request gốc,
dataset/prompt version, `temperature=0`, `seed=42`, caps/response_format gốc.
Các file frozen trong Git dùng LF. Checkout Windows với CRLF bị hash guard từ chối;
không normalize byte trong verifier, không sửa manifest hoặc dữ liệu đóng băng.
Checkout Linux LF trên Kaggle phải vượt qua kiểm tra byte gốc này.

Không có request warm-up riêng trong phép đo lossless này (giống runner canonical
đã lưu ở archive): mỗi run replay đúng 54 request tuần tự, concurrency=1. Quy tắc
warm-up ≥3 và ≥5 repeats/ô tại kế hoạch đo hiệu năng không phải sweep đang thực hiện.
Không thay đổi cache/runtime giữa các lần chạy; fresh server mỗi lần chuyển làn là
trạng thái bắt đầu cố định, không phải chứng minh KV/cache nội bộ bằng nhau.

## Chạy hai cặp tuần tự sau phê duyệt

Hàm bên dưới chỉ bọc workflow `switch_lane.sh` hiện có và collector; không có retry.
Log server được sao chép trước khi lần chuyển làn tiếp theo có thể ghi đè log chung.
Mỗi label được reserve bằng atomic mkdir trước mọi side effect; label trùng dừng
trước cả việc mở log. `noclobber` và capture claim độc quyền bảo vệ artifact đã có.
Collector fsync attempt trước HTTP và result sau HTTP (cả khi parse lỗi); timeout=600 giây,
giữ HTTP status/body lỗi, partial run và `complete=false`. Cặp thất bại không bị bỏ
ra khỏi mẫu để báo cáo kết quả có vẻ tốt hơn.

```bash
assert_gpu() {
  test "$(nvidia-smi -i "$GPU_UUID" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$GPU_UUID"
  test "$(git rev-parse HEAD)" = "$EXECUTION_SHA"
  test -z "$(git status --porcelain --untracked-files=no)"
}

run_lane() {
  local pair="$1" lane="$2" selector="$3" log_number="$4"
  local label="${pair}-${lane}"
  # Reserve before any switch, restart, log open/copy or request.
  python -m experiments.protocol_b_capture --reserve-only \
    --output "$RUN_ROOT/$label" --pair-id "$pair" --run-id "$label" \
    --lane "$lane" --session-id "$SESSION_ID" || return $?
  assert_gpu
  nvidia-smi -i "$GPU_UUID" -q > "$RUN_ROOT/$label/gpu-before.txt"
  local rc=0
  bash scripts/switch_lane.sh "$selector" > "$RUN_ROOT/$label/switch.log" 2>&1 || rc=$?
  if grep -Fq '[CẢNH BÁO] GPU vẫn còn' "$RUN_ROOT/$label/switch.log"; then
    rc=1
  fi
  if [ "$rc" -eq 0 ]; then
    python -m experiments.protocol_b_capture --reserved \
      --output "$RUN_ROOT/$label" --pair-id "$pair" --run-id "$label" \
      --lane "$lane" --session-id "$SESSION_ID" \
      > "$RUN_ROOT/$label/capture.log" 2>&1 || rc=$?
  fi
  if [ -f "logs/vllm_lane${log_number}.log" ]; then
    cat "logs/vllm_lane${log_number}.log" > "$RUN_ROOT/$label/server.log"
  fi
  printf '%s\n' "$rc" > "$RUN_ROOT/$label/exit-code.txt"
  nvidia-smi -i "$GPU_UUID" -q > "$RUN_ROOT/$label/gpu-after.txt"
  assert_gpu
  return "$rc"
}

run_lane pair1 vanilla lane1 1
run_lane pair1 adaptive lane3 3
run_lane pair2 vanilla lane1 1
run_lane pair2 adaptive lane3 3
bash scripts/switch_lane.sh stop > "$RUN_ROOT/final-stop.log" 2>&1
nvidia-smi -q > "$RUN_ROOT/nvidia_after.txt"
```

Mỗi lần chuyển làn phải giải phóng GPU trước khi server mới bắt đầu. Workflow cũ
cảnh báo nhưng vẫn tiếp tục nếu GPU chưa nhả VRAM sau timeout; nếu log có cảnh báo
đó, giữ bằng chứng và dừng, không coi run là cặp kiểm soát hợp lệ. Collector cũng
từ chối nếu có nhiều server vLLM hoặc process args khác cấu hình launcher.

## Tổng hợp bằng chứng

Chỉ chạy đoạn này nếu cả bốn run hoàn tất. Nó tạo file mới, không ghi đè summary cũ.
Nếu bất kỳ run thất bại: giữ raw, logs và exit code; báo `INVALID_EVIDENCE`, không
điền kết quả còn thiếu bằng run lịch sử và không kết luận PASS.

```bash
cat > "$RUN_ROOT/experiment.json" <<'JSON'
{"pairs":[{"pair_id":"pair1","vanilla":"pair1-vanilla/run.json","adaptive":"pair1-adaptive/run.json"},{"pair_id":"pair2","vanilla":"pair2-vanilla/run.json","adaptive":"pair2-adaptive/run.json"}]}
JSON
python -m experiments.protocol_b "$RUN_ROOT/experiment.json" "$RUN_ROOT/summary.json" \
  --generated-at "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"
```

`summary.json` báo riêng V1/V2, A1/A2, V1/A1, V2/A2. Mỗi metric gồm valid prompt
pairs, exact matches, match rate, workload/sample ID lệch, vị trí/token ID lệch đầu,
hai độ dài completion, finish reasons và truncation. Output token exact match
không có nghĩa là full completion nếu có `finish_reason=length`.

Provenance gồm pair/run/lane, timestamp, request SHA, Git SHA, hash source và
launcher, config/environment, session/boot ID, GPU name/UUID/driver,
CUDA/torch/vLLM, target/draft model revision. Environment phải khớp cả bốn run;
phải theo thứ tự V1 → A1 → V2 → A2. Report kiểm tra hash raw journal và identity,
không tự điền provenance còn thiếu. `generated_at` được truyền rõ; cùng inputs và
timestamp cho cùng serialization. Exit 0 chỉ có nghĩa báo cáo evidence hợp lệ,
không có nghĩa M2-03 PASS.

Binding live server: `fuser` xác định PID sở hữu đúng cổng; `/proc/PID/cmdline`,
`environ`, `exe` và start ticks chứng minh argv, environment và lifetime của process.
Chỉ chấp nhận tập argv của launcher với target `--revision` và draft `revision`
đã đóng băng ([trường revision của vLLM 0.30.0](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/speculative.py#L396)).
Vanilla không nạp draft; draft SHA ở làn này chỉ là cấu hình đóng băng của phiên.
Cache policy là defaults của vLLM 0.30.0 với argv và live environment không có cache
override; không suy diễn trạng thái KV bên trong. Không chứng minh được binding thì
dừng. Version probe dùng executable và environment của server. Raw environment đầy
đủ chỉ dùng trong bộ nhớ để probe, không ghi credentials vào evidence.

Luôn giữ `interpretation_status=NOT_YET_APPROVED`, `task_done=false`. Không dùng
“Adaptive drift không vượt Vanilla drift” làm ngưỡng PASS khi PM chưa duyệt.
Không suy ra nguyên nhân số học hoặc khẳng định Adaptive không có bug từ một đợt đo.

Sau khi lưu/download nguyên thư mục evidence, dừng để PM xem xét cách diễn giải.
Mọi lần thu thập mới phải có session/run root mới; không sửa JSONL/tar hoặc summary
lịch sử, không ghi vào `experiments/runs/m2_03/**` hay `m2_03_targeted/**`.
