#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Khởi chạy vLLM Làn 4: Custom EAGLE-3.1 Speculator (Target: Qwen2.5-7B-AWQ + Draft: duckhoi/qwen2.5-7b-eagle3.1-speculator, k=3)"

# 1. Cấu hình biến môi trường tối ưu bộ nhớ
export HF_HUB_ENABLE_HF_TRANSFER=0

# 2. One serving contract, including immutable target/draft revisions.
# Keep the local FP16/AWQ profile used by the other local launchers.
# Assignment fails before vLLM starts if Python cannot build the arguments.
VLLM_ARGS_TEXT=$(python3 - <<'PY'
from experiments.modal_lanes import build_vllm_args
import sys
args = build_vllm_args("eagle", profile="fp16-fallback", port=8004)
sys.stdout.buffer.write(("\n".join(args) + "\n").encode("utf-8"))
PY
)
mapfile -t VLLM_ARGS <<< "$VLLM_ARGS_TEXT"

# 3. Xác định lệnh vLLM (hỗ trợ cả CLI và Python module)
if command -v vllm &> /dev/null; then
  VLLM_CMD="vllm serve"
else
  VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server"
fi

# 4. Khởi chạy máy chủ vLLM OpenAI-compatible trên cổng 8004.
$VLLM_CMD "${VLLM_ARGS[@]}"
