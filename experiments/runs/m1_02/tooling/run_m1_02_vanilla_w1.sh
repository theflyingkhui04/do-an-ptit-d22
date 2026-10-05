#!/usr/bin/env bash
set -euo pipefail

BENCH_REPO=/kaggle/working/do-an-ptit-d22
EVID_REPO=/kaggle/working/do-an-ptit-d22-evidence

FROZEN_SHA=f307e33f9aea4bbd45499f548480cb7d6f90a4b9
EVID_BRANCH=evidence/m1-02-f307e33

CAMPAIGN_FILE="$EVID_REPO/experiments/runs/m1_02/CAMPAIGN.json"

PREFLIGHT=/kaggle/working/run_m1_02_kaggle.sh
VALIDATOR=/kaggle/working/validate_m1_02_sweep.py
CHECKPOINT=/kaggle/working/checkpoint_m1_02_sweep.sh
RESUME_CHECK=/kaggle/working/check_m1_02_resume.py

LAUNCHER="$BENCH_REPO/scripts/launch_lane1_vanilla.sh"
HARNESS="$BENCH_REPO/experiments/benchmark_harness.py"
WORKLOAD_FILE="$BENCH_REPO/experiments/configs/w1/node1_chat_requests.jsonl"

LANE=vanilla
WORKLOAD=W1
PORT=8001
MODEL=qwen2.5-7b-vanilla

STARTUP_TIMEOUT_S=900
GPU_RELEASE_TIMEOUT_S=120
MIN_GPU_FREE_MIB=13000

SERVER_PID=""
SERVER_PGID=""

fail() {
    echo
    echo "VANILLA/W1 CANONICAL FAIL: $*" >&2
    exit 1
}

cleanup_server() {
    if [ -n "${SERVER_PGID:-}" ]; then
        if ps -eo pgid= |
            awk '{print $1}' |
            grep -qx "$SERVER_PGID"
        then
            echo "Stopping server process group $SERVER_PGID ..."
            kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true

            for _ in $(seq 1 30); do
                if ! ps -eo pgid= |
                    awk '{print $1}' |
                    grep -qx "$SERVER_PGID"
                then
                    break
                fi
                sleep 2
            done

            if ps -eo pgid= |
                awk '{print $1}' |
                grep -qx "$SERVER_PGID"
            then
                echo "Graceful shutdown timed out; killing PGID $SERVER_PGID"
                kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
            fi
        fi
    fi
}

trap cleanup_server EXIT INT TERM

echo "================================================"
echo "M1-02 CANONICAL ONE-SWEEP RUNNER: VANILLA / W1"
echo "================================================"

echo
echo "=== 1. PREFLIGHT ==="

"$PREFLIGHT"

echo
echo "=== 2. LOAD DURABLE CAMPAIGN ID ==="

test -f "$CAMPAIGN_FILE" \
    || fail "CAMPAIGN.json missing"

EXPERIMENT_ID="$(
    python3 - "$CAMPAIGN_FILE" "$FROZEN_SHA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_sha = sys.argv[2]

data = json.loads(path.read_text(encoding="utf-8"))

if data.get("schema") != "m1_02_campaign.v1":
    raise SystemExit("unexpected campaign schema")

if data.get("benchmark_commit") != expected_sha:
    raise SystemExit("campaign benchmark_commit mismatch")

experiment_id = data.get("experiment_id")

if not isinstance(experiment_id, str) or not experiment_id:
    raise SystemExit("invalid campaign experiment_id")

print(experiment_id)
PY
)"

echo "experiment_id = $EXPERIMENT_ID"

LOCAL_POINTER=/kaggle/working/m1_02_campaign_id

if [ -f "$LOCAL_POINTER" ]; then
    POINTER_ID="$(cat "$LOCAL_POINTER")"

    test "$POINTER_ID" = "$EXPERIMENT_ID" \
        || fail "local campaign pointer disagrees with durable CAMPAIGN.json"
fi

RUN_DIR="/kaggle/working/m1_02_campaign/$EXPERIMENT_ID"
RAW_TMP="$RUN_DIR/vanilla_w1.jsonl"
SUMMARY_TMP="$RUN_DIR/vanilla_w1_smry_not_canonical.csv"
SERVER_LOG="$RUN_DIR/vanilla_w1.server.log"

mkdir -p "$RUN_DIR"

echo
echo "=== 3. RESUME CHECK ==="

set +e
python3 "$RESUME_CHECK" \
    --evidence-repo "$EVID_REPO" \
    --lane "$LANE" \
    --workload "$WORKLOAD" \
    --experiment-id "$EXPERIMENT_ID"
RESUME_RC=$?
set -e

case "$RESUME_RC" in
    0)
        echo
        echo "VANILLA/W1 ALREADY COMPLETE_VALID."
        echo "Nothing will be rerun."
        exit 0
        ;;

    10)
        echo
        echo "VANILLA/W1 state = NOT_STARTED"
        echo "Proceeding with whole canonical sweep."
        ;;

    *)
        fail "resume checker returned rc=$RESUME_RC; refusing to run"
        ;;
esac

echo
echo "=== 4. PRE-LAUNCH CLEANLINESS ==="

test ! -e "$RAW_TMP" \
    || fail "temporary raw already exists: $RAW_TMP"

if curl -fsS \
    --max-time 1 \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null 2>&1
then
    fail "port $PORT already serves /health"
fi

if ps -eo cmd= |
    grep -E '[v]llm|[E]ngineCore|[a]pi_server' \
    >/dev/null
then
    ps -eo pid,ppid,pgid,stat,etime,cmd |
        grep -E '[v]llm|[E]ngineCore|[a]pi_server' || true

    fail "existing vLLM/EngineCore process"
fi

FREE_MIB="$(
    nvidia-smi -i 0 \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits |
    head -n1 |
    xargs
)"

echo "free MiB before server: $FREE_MIB"

test "$FREE_MIB" -ge "$MIN_GPU_FREE_MIB" \
    || fail "GPU not sufficiently free"

echo
echo "=== 5. START VANILLA SERVER ==="

export CUDA_VISIBLE_DEVICES=0

: > "$SERVER_LOG"

setsid bash "$LAUNCHER" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

sleep 1

SERVER_PGID="$(
    ps -o pgid= -p "$SERVER_PID" |
    xargs
)"

test -n "$SERVER_PGID" \
    || fail "could not resolve server PGID"

echo "launcher PID : $SERVER_PID"
echo "server PGID  : $SERVER_PGID"
echo "server log   : $SERVER_LOG"

echo
echo "=== 6. WAIT FOR HEALTH ==="

START_TS="$(date +%s)"

while true; do
    if curl -fsS \
        --max-time 3 \
        "http://127.0.0.1:${PORT}/health" \
        >/dev/null 2>&1
    then
        NOW_TS="$(date +%s)"
        echo "/health PASS after $((NOW_TS - START_TS))s"
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "===== SERVER LOG TAIL ====="
        tail -n 160 "$SERVER_LOG" || true
        fail "server exited during startup"
    fi

    NOW_TS="$(date +%s)"
    ELAPSED=$((NOW_TS - START_TS))

    if [ "$ELAPSED" -ge "$STARTUP_TIMEOUT_S" ]; then
        echo "===== SERVER LOG TAIL ====="
        tail -n 160 "$SERVER_LOG" || true
        fail "server startup timeout"
    fi

    echo "waiting for health... ${ELAPSED}s"
    sleep 3
done

echo
echo "=== 7. VERIFY MODEL IDENTITY ==="

curl -fsS \
    --max-time 10 \
    "http://127.0.0.1:${PORT}/v1/models" |
python3 -c '
import json
import sys

expected = "qwen2.5-7b-vanilla"

payload = json.load(sys.stdin)
models = [
    item.get("id")
    for item in payload.get("data", [])
    if isinstance(item, dict)
]

print("models =", models)

if expected not in models:
    raise SystemExit(
        f"expected {expected!r}, got {models!r}"
    )

print("MODEL IDENTITY: PASS")
'

echo
echo "=== 8. VERIFY LIVE VANILLA FLAGS ==="

GROUP_PS="$(
    ps -eo pid=,ppid=,pgid=,args= |
    awk -v pg="$SERVER_PGID" '$3 == pg'
)"

printf '%s\n' "$GROUP_PS"

for token in \
    "Qwen/Qwen2.5-7B-Instruct-AWQ" \
    "--port 8001" \
    "--served-model-name qwen2.5-7b-vanilla" \
    "--generation-config vllm" \
    "--quantization awq" \
    "--dtype half" \
    "--gpu-memory-utilization 0.85" \
    "--max-model-len 4096" \
    "--max-num-seqs 32"
do
    printf '%s\n' "$GROUP_PS" |
        grep -F -- "$token" >/dev/null \
        || fail "live command missing: $token"
done

echo "VANILLA LIVE CONTRACT: PASS"

echo
echo "=== 9. RUN CANONICAL VANILLA/W1 SWEEP ==="

cd "$BENCH_REPO"

python3 "$HARNESS" \
    --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --model "$MODEL" \
    --workload-file "$WORKLOAD_FILE" \
    --sample-policy cycle-rotate \
    --num-requests 32 \
    --concurrency-sweep 1 2 4 8 16 32 \
    --repeats 5 \
    --warmup-requests 3 \
    --lane vanilla \
    --workload W1 \
    --experiment-id "$EXPERIMENT_ID" \
    --raw-out "$RAW_TMP" \
    --summary-csv "$SUMMARY_TMP" \
    --overwrite

echo
echo "=== 10. VALIDATE COMPLETE RAW ==="

python3 "$VALIDATOR" \
    --raw "$RAW_TMP" \
    --lane vanilla \
    --workload W1 \
    --experiment-id "$EXPERIMENT_ID"

RAW_SHA="$(
    sha256sum "$RAW_TMP" |
    awk '{print $1}'
)"

echo "validated raw SHA256 = $RAW_SHA"

echo
echo "=== 11. SHUT DOWN SERVER ==="

cleanup_server
SERVER_PGID=""
SERVER_PID=""

echo
echo "=== 12. VERIFY GPU RELEASE ==="

GPU_RELEASED=0
FREE_MIB=0

for _ in $(seq 1 "$GPU_RELEASE_TIMEOUT_S"); do
    FREE_MIB="$(
        nvidia-smi -i 0 \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits |
        head -n1 |
        xargs
    )"

    if [ "$FREE_MIB" -ge "$MIN_GPU_FREE_MIB" ]; then
        GPU_RELEASED=1
        break
    fi

    sleep 1
done

echo "free MiB after shutdown: $FREE_MIB"

test "$GPU_RELEASED" -eq 1 \
    || fail "GPU memory did not recover"

if ps -eo cmd= |
    grep -E '[v]llm|[E]ngineCore|[a]pi_server' \
    >/dev/null
then
    fail "orphan vLLM/EngineCore remains"
fi

echo "SERVER CLEANUP: PASS"

echo
echo "=== 13. CREATE LOCAL EVIDENCE CHECKPOINT ==="

"$CHECKPOINT" \
    "$EVID_REPO" \
    "$RAW_TMP" \
    vanilla \
    W1 \
    "$EXPERIMENT_ID"

echo
echo "=== 14. ATTACH SERVER LOG TO SAME CHECKPOINT COMMIT ==="

WORKLOAD_LOWER=w1
REL_DIR="experiments/runs/m1_02/$EXPERIMENT_ID"

DONE_REL="$REL_DIR/vanilla_w1.done.json"
LOG_REL="$REL_DIR/vanilla_w1.server.log"

DONE_FILE="$EVID_REPO/$DONE_REL"
LOG_DST="$EVID_REPO/$LOG_REL"

cp "$SERVER_LOG" "$LOG_DST"

LOG_SHA="$(
    sha256sum "$LOG_DST" |
    awk '{print $1}'
)"

python3 - \
    "$DONE_FILE" \
    "$LOG_REL" \
    "$LOG_SHA" <<'PY'
import json
import sys
from pathlib import Path

done_path = Path(sys.argv[1])
server_log_path = sys.argv[2]
server_log_sha = sys.argv[3]

data = json.loads(
    done_path.read_text(encoding="utf-8")
)

data["server_log_path"] = server_log_path
data["server_log_sha256"] = server_log_sha

done_path.write_text(
    json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

git -C "$EVID_REPO" add \
    "$DONE_REL" \
    "$LOG_REL"

git -C "$EVID_REPO" diff --cached --check

git -C "$EVID_REPO" commit \
    --amend \
    --no-edit

echo
echo "=== 15. REVERIFY CHECKPOINT AFTER AMEND ==="

python3 "$VALIDATOR" \
    --raw "$EVID_REPO/$REL_DIR/vanilla_w1.jsonl" \
    --lane vanilla \
    --workload W1 \
    --experiment-id "$EXPERIMENT_ID"

test "$(
    sha256sum "$EVID_REPO/$REL_DIR/vanilla_w1.jsonl" |
    awk '{print $1}'
)" = "$RAW_SHA" \
    || fail "evidence raw SHA changed"

echo
echo "=== 16. PUSH DURABLE EVIDENCE ==="

git -C "$EVID_REPO" push \
    origin "$EVID_BRANCH"

echo
echo "=== 17. VERIFY REMOTE REF ==="

LOCAL_SHA="$(
    git -C "$EVID_REPO" rev-parse HEAD
)"

REMOTE_SHA="$(
    git -C "$EVID_REPO" ls-remote origin \
        "refs/heads/$EVID_BRANCH" |
    awk '{print $1}'
)"

echo "local evidence HEAD : $LOCAL_SHA"
echo "remote evidence HEAD: $REMOTE_SHA"

test "$LOCAL_SHA" = "$REMOTE_SHA" \
    || fail "remote ref does not match local HEAD"

echo
echo "=== 18. POST-PUSH RESUME VERIFICATION ==="

python3 "$RESUME_CHECK" \
    --evidence-repo "$EVID_REPO" \
    --lane vanilla \
    --workload W1 \
    --experiment-id "$EXPERIMENT_ID"

echo
echo "=== 19. FINAL CLEANLINESS ==="

test -z "$(git -C "$BENCH_REPO" status --porcelain)" \
    || fail "benchmark checkout became dirty"

test -z "$(git -C "$EVID_REPO" status --porcelain)" \
    || fail "evidence checkout became dirty"

echo
echo "================================================"
echo "VANILLA/W1 CANONICAL SWEEP: DURABLE PASS"
echo "experiment_id=$EXPERIMENT_ID"
echo "raw_sha256=$RAW_SHA"
echo "================================================"
