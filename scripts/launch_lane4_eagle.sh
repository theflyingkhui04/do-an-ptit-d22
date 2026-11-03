#!/usr/bin/env bash
set -e

echo "Khởi chạy vLLM Làn 4: Custom EAGLE-3.1 Speculator (Target: Qwen2.5-7B-AWQ + Draft: duckhoi/qwen2.5-7b-eagle3.1-speculator, k=3)"

# 1. Cấu hình biến môi trường tối ưu bộ nhớ
export HF_HUB_ENABLE_HF_TRANSFER=0

# 2. Cấu hình Speculative Decoding JSON
SPEC_CONFIG='{"method": "draft_model", "model": "duckhoi/qwen2.5-7b-eagle3.1-speculator", "num_speculative_tokens": 3, "use_heterogeneous_vocab": true, "draft_sample_method": "greedy"}'

# 3. Xác định lệnh vLLM (hỗ trợ cả CLI và Python module)
if command -v vllm &> /dev/null; then
  VLLM_CMD="vllm serve"
else
  VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server"
fi

# 4. Khởi chạy máy chủ vLLM OpenAI-compatible trên cổng 8004.
$VLLM_CMD Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8004 \
  --served-model-name qwen2.5-7b-eagle3-draft \
  --generation-config vllm \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --speculative-config "$SPEC_CONFIG" \
  --per-request-spec-decode-metrics summary
