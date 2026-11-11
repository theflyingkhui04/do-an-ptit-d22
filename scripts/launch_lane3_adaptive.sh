#!/usr/bin/env bash
set -e

echo "KHỞI CHẠY vLLM LÀN 3: ADAPTIVE DRAFT (Target: Qwen2.5-7B-AWQ + Draft: Qwen2.5-0.5B-FP16, k động 0..4)"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Cấu hình biến môi trường
export HF_HUB_ENABLE_HF_TRANSFER=0
# Scheduler của nhóm (src.server.adaptive_scheduler) phải import được trong tiến trình EngineCore.
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 2. Cấu hình AdaptiveAsyncScheduler (có thể ghi đè trước khi gọi script)
#    ADAPTIVE_ENABLE_VRAM_GUARD=0 nếu free VRAM lúc idle trên T4 < 1500 MB (quyết định B8-4).
export ADAPTIVE_TELEMETRY_PATH="${ADAPTIVE_TELEMETRY_PATH:-$PROJECT_ROOT/experiments/runs/m2_02/lane3_telemetry.jsonl}"
export ADAPTIVE_ENABLE_VRAM_GUARD="${ADAPTIVE_ENABLE_VRAM_GUARD:-1}"
export ADAPTIVE_MAX_K="${ADAPTIVE_MAX_K:-4}"
export ADAPTIVE_MIN_BATCH_K="${ADAPTIVE_MIN_BATCH_K:-0}"
export ADAPTIVE_PER_REQUEST="${ADAPTIVE_PER_REQUEST:-1}"
export ADAPTIVE_CONTROLLER_KWARGS="${ADAPTIVE_CONTROLLER_KWARGS:-}"
mkdir -p "$(dirname "$ADAPTIVE_TELEMETRY_PATH")"

echo "    telemetry=$ADAPTIVE_TELEMETRY_PATH vram_guard=$ADAPTIVE_ENABLE_VRAM_GUARD max_k=$ADAPTIVE_MAX_K min_batch_k=$ADAPTIVE_MIN_BATCH_K per_request=$ADAPTIVE_PER_REQUEST"

# 3. Speculative config: giống Làn 2 nhưng num_speculative_tokens=4 (trần k) và bật
#    dynamic SD để scheduler đổi K mỗi bước qua dynamic_sd_lookup.
SPEC_CONFIG='{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 4, "num_speculative_tokens_per_batch_size": [[1, 32, 4]], "use_heterogeneous_vocab": true, "draft_sample_method": "greedy"}'
if [ -n "${PROTOCOL_B_DRAFT_REVISION:-}" ]; then
  SPEC_CONFIG=$(python3 -c 'import json,os,sys; c=json.loads(sys.argv[1]); c["revision"]=os.environ["PROTOCOL_B_DRAFT_REVISION"]; print(json.dumps(c))' "$SPEC_CONFIG")
fi
REVISION_ARGS=()
if [ -n "${PROTOCOL_B_TARGET_REVISION:-}" ]; then
  REVISION_ARGS=(--revision "$PROTOCOL_B_TARGET_REVISION")
fi

# 4. Xác định lệnh vLLM (hỗ trợ cả CLI và Python module)
if command -v vllm &> /dev/null; then
  VLLM_CMD="vllm serve"
else
  VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server"
fi

# 5. Khởi chạy máy chủ vLLM OpenAI-compatible trên cổng 8003.
# Cùng generation defaults, quantization, VRAM và max-num-seqs như Làn 1/2;
# khác biệt thực nghiệm duy nhất là scheduler điều khiển k thích ứng.
$VLLM_CMD Qwen/Qwen2.5-7B-Instruct-AWQ \
  "${REVISION_ARGS[@]}" \
  --host 0.0.0.0 \
  --port 8003 \
  --served-model-name qwen2.5-7b-adaptive \
  --generation-config vllm \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --speculative-config "$SPEC_CONFIG" \
  --scheduler-cls src.server.adaptive_scheduler.AdaptiveAsyncScheduler \
  --distributed-executor-backend src.server.lineage_executor.LineageUniProcExecutor \
  --per-request-spec-decode-metrics summary
