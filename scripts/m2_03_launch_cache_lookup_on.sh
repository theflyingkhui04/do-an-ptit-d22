#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Preflight
test "$(git branch --show-current)" = \
  "feat/m2-03-lossless-verification"

test -x .venv/bin/vllm
test -f src/server/m2_03_cache_lookup_diag.py

if ps -eo args | grep -Eq \
  '[v]llm serve|[V]LLM::EngineCore'; then
  echo "STOP: Existing vLLM process detected"
  exit 2
fi

.venv/bin/python - <<'PY'
import socket
with socket.socket() as s:
    assert s.connect_ex(("127.0.0.1", 8003)) != 0, \
        "STOP: Port 8003 occupied"
PY

D="$(mktemp -d \
  "$ROOT/experiments/runs/m2_03/diagnostics/cache_lookup_on_$(date -u +%Y%m%dT%H%M%SZ)_XXXX")"

printf '%s\n' "$D" > /tmp/m203_cache_lookup_on_dir

git rev-parse HEAD > "$D/git_sha.txt"
nvidia-smi > "$D/gpu_before.txt"

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export M203_CACHE_LOOKUP_JSONL="$D/cache_lookup.jsonl"
export ADAPTIVE_TELEMETRY_PATH="$D/telemetry.jsonl"

# Fail early if output cannot be created.
: > "$M203_CACHE_LOOKUP_JSONL"

SPEC_CONFIG='{"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4,"num_speculative_tokens_per_batch_size":[[1,32,4]],"use_heterogeneous_vocab":true,"draft_sample_method":"greedy"}'

echo "=== CACHE LOOKUP ON ==="
echo "EVIDENCE_DIR=$D"
echo "CACHE_LOOKUP_JSONL=$M203_CACHE_LOOKUP_JSONL"

.venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8003 \
  --served-model-name qwen2.5-7b-adaptive \
  --generation-config vllm \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --speculative-config "$SPEC_CONFIG" \
  --scheduler-cls \
    src.server.m2_03_cache_lookup_diag.M203CacheLookupScheduler \
  --per-request-spec-decode-metrics summary \
  > >(tee "$D/server.log") 2>&1
