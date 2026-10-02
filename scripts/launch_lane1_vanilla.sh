#!/usr/bin/env bash
set -e

echo "  KHỞI CHẠY vLLM LÀN 1: VANILLA BASELINE (Qwen2.5-7B-Instruct AWQ)"

# 1. Cấu hình biến môi trường tối ưu bộ nhớ
export HF_HUB_ENABLE_HF_TRANSFER=0

# 2. Xác định lệnh vLLM (hỗ trợ cả CLI và Python module nếu chưa cấu hình PATH)
if command -v vllm &> /dev/null; then
  VLLM_CMD="vllm serve"
else
  VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server"
fi

# 3. Khởi chạy máy chủ vLLM OpenAI-compatible
$VLLM_CMD Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name qwen2.5-7b-vanilla \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 32
