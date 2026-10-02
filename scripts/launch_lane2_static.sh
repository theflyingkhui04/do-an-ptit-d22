#!/usr/bin/env bash
set -e

echo "KHỞI CHẠY vLLM LÀN 2: STATIC DRAFT BASELINE (Target: Qwen2.5-7B-AWQ + Draft: Qwen2.5-0.5B-FP16, k=3)"

# 1. Cấu hình biến môi trường tối ưu bộ nhớ
export HF_HUB_ENABLE_HF_TRANSFER=0

# 2. Cấu hình Speculative Decoding JSON
SPEC_CONFIG='{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 3, "use_heterogeneous_vocab": true, "draft_sample_method": "greedy"}'

# 3. Xác định lệnh vLLM (hỗ trợ cả CLI và Python module)
if command -v vllm &> /dev/null; then
  VLLM_CMD="vllm serve"
else
  VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server"
fi

# 4. Khởi chạy máy chủ vLLM OpenAI-compatible trên cổng 8002
$VLLM_CMD Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8002 \
  --served-model-name qwen2.5-7b-static-draft \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --speculative-config "$SPEC_CONFIG" \
  --per-request-spec-decode-metrics summary

