#!/usr/bin/env bash
# M2-03 diagnostic only. Does NOT run the canonical acceptance sweep.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python || ! -x .venv/bin/vllm ]]; then
  echo "ERROR: pinned vLLM .venv missing" >&2; exit 2
fi

# Refuse to launch if a server is already listening (including /health not ready).
if .venv/bin/python - <<'PY'
import socket
s = socket.socket()
s.settimeout(1.0)
try:
    raise SystemExit(0 if s.connect_ex(("127.0.0.1", 8003)) == 0 else 1)
finally:
    s.close()
PY
then
  echo "ERROR: port 8003 occupied. Stop/inspect the existing server manually." >&2
  exit 2
fi

STATUS="$(.venv/bin/python scripts/m2_03_install_a3_v8.py status)"
echo "$STATUS"
if ! grep -q '^A3_V8_HOOK_STATUS: V8_INSTALLED$' <<<"$STATUS"; then
  echo "ERROR: V1 A3 v8 hook missing. Verify/install before launch." >&2
  exit 2
fi

D="$(mktemp -d "$ROOT/experiments/runs/m2_03/diagnostics/a3_v1_v8_adaptive_$(date -u +%Y%m%dT%H%M%SZ)_XXXX")"
echo "$D" > /tmp/m203_a3_adaptive_dir
GIT_SHA="$(git rev-parse HEAD)"
.venv/bin/python - "$D" "$GIT_SHA" <<'PY'
from pathlib import Path
import json, sys
out, git_sha = sys.argv[1:]
config = {"lane": "adaptive", "target": 200, "git_sha": git_sha,
          "output_dir": str(Path(out) / "probe")}
Path("/tmp/m203_a3_probe_config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8")
print("A3_V8_CONFIG:", json.dumps(config, sort_keys=True))
PY
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ADAPTIVE_TELEMETRY_PATH="$D/lane3_telemetry.jsonl"
SPEC_CONFIG='{"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":4,"num_speculative_tokens_per_batch_size":[[1,32,4]],"use_heterogeneous_vocab":true,"draft_sample_method":"greedy"}'
echo "EVIDENCE_DIR=$D"
echo "Starting Adaptive V1 (vLLM 0.30.0) on 127.0.0.1:8003"
# Do not force V2: draft_model is unsupported by V2 in this pinned release.
exec .venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 127.0.0.1 --port 8003 \
  --served-model-name qwen2.5-7b-adaptive \
  --generation-config vllm \
  --quantization awq --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --max-num-seqs 32 \
  --speculative-config "$SPEC_CONFIG" \
  --scheduler-cls src.server.adaptive_scheduler.AdaptiveAsyncScheduler \
  --per-request-spec-decode-metrics summary \
  2>&1 | tee "$D/server.log"
