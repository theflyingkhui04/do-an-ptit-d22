#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if curl -fsS --max-time 2 http://127.0.0.1:8003/health >/dev/null 2>&1; then
  echo "ERROR: port 8003 already serves /health. Stop the old Adaptive server first." >&2
  exit 2
fi

if ! grep -q 'UNCONDITIONAL_HOOK_COUNT: 1' <(
  .venv/bin/python scripts/m2_03_install_a3_hook_v6.py status 2>&1
); then
  echo "ERROR: unconditional A3 hook is not installed. Run installer first." >&2
  .venv/bin/python scripts/m2_03_install_a3_hook_v6.py status || true
  exit 2
fi

D=$(mktemp -d \
  "$ROOT/experiments/runs/m2_03/diagnostics/a3_adaptive_$(date -u +%Y%m%dT%H%M%SZ)_XXXX")
echo "$D" > /tmp/m203_a3_adaptive_dir

GIT_SHA="$(git rev-parse HEAD)"
.venv/bin/python - "$D" "$GIT_SHA" <<'PY'
import json
from pathlib import Path
import sys
out, sha = sys.argv[1], sys.argv[2]
config = {
    "lane": "adaptive",
    "target": 200,
    "git_sha": sha,
    "output_dir": str(Path(out) / "probe"),
}
Path("/tmp/m203_a3_probe_config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
PY

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ADAPTIVE_TELEMETRY_PATH="$D/lane3_telemetry.jsonl"

SPEC_CONFIG='{"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4,"num_speculative_tokens_per_batch_size":[[1,32,4]],"use_heterogeneous_vocab":true,"draft_sample_method":"greedy"}'

echo "EVIDENCE_DIR=$D"
echo "A3_CONFIG=/tmp/m203_a3_probe_config.json"
echo "Starting Adaptive diagnostic server on 127.0.0.1:8003"

exec .venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 127.0.0.1 \
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
  --per-request-spec-decode-metrics summary \
  2>&1 | tee "$D/server.log"
