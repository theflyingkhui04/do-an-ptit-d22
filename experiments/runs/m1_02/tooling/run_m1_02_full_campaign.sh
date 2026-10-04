#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# M1-02 FULL CANONICAL BASELINE CAMPAIGN
#
# Order:
#   Vanilla/W1
#   Static/W1
#   Vanilla/W2
#   Static/W2
#   Vanilla/W3
#   Static/W3
#
# Each sweep:
#   6 concurrency cells
#   5 measured repeats/cell
#   32 measured requests/repeat
#   = 960 measured requests
#
# Campaign:
#   6 sweeps
#   36 cells
#   180 measured repeats
#   5760 measured requests
# ============================================================

BENCH=/kaggle/working/do-an-ptit-d22
EVID=/kaggle/working/do-an-ptit-d22-evidence

FROZEN_SHA=f307e33f9aea4bbd45499f548480cb7d6f90a4b9
EVID_BRANCH=evidence/m1-02-f307e33
EXPECTED_EXPERIMENT_ID=m1-02-f307e33-20261004T102725Z

CAMPAIGN_FILE="$EVID/experiments/runs/m1_02/CAMPAIGN.json"

PREFLIGHT=/kaggle/working/run_m1_02_kaggle.sh
VALIDATOR=/kaggle/working/validate_m1_02_sweep.py
RESUME_CHECK=/kaggle/working/check_m1_02_resume.py
CHECKPOINT=/kaggle/working/checkpoint_m1_02_sweep.sh

HARNESS="$BENCH/experiments/benchmark_harness.py"
AGGREGATOR="$BENCH/experiments/aggregate_results.py"

STARTUP_TIMEOUT_S=900
GPU_RELEASE_TIMEOUT_S=120
MIN_GPU_FREE_MIB=13000

SERVER_PID=""
SERVER_PGID=""

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


fail() {
    echo
    echo "M1-02 FULL CAMPAIGN FAIL: $*" >&2
    exit 1
}


cleanup_server() {
    if [ -z "${SERVER_PGID:-}" ]; then
        return 0
    fi

    if ps -eo pgid= |
        awk '{print $1}' |
        grep -qx "$SERVER_PGID"
    then
        echo "Stopping server PGID $SERVER_PGID ..."

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
            echo "Graceful shutdown timed out; KILL PGID $SERVER_PGID"
            kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
        fi
    fi
}


trap cleanup_server EXIT INT TERM


wait_gpu_release() {
    local released=0
    local free_mib=0

    for _ in $(seq 1 "$GPU_RELEASE_TIMEOUT_S"); do

        free_mib="$(
            nvidia-smi -i 0 \
                --query-gpu=memory.free \
                --format=csv,noheader,nounits |
            head -n1 |
            xargs
        )"

        if [ "$free_mib" -ge "$MIN_GPU_FREE_MIB" ]; then
            released=1
            break
        fi

        sleep 1
    done

    echo "free MiB after shutdown: $free_mib"

    test "$released" -eq 1 \
        || fail "GPU memory did not recover"

    if ps -eo cmd= |
        grep -E '[v]llm|[E]ngineCore|[a]pi_server' \
        >/dev/null
    then
        ps -eo pid,ppid,pgid,stat,etime,args |
            grep -E '[v]llm|[E]ngineCore|[a]pi_server' || true

        fail "vLLM/EngineCore orphan remains"
    fi

    echo "GPU/PROCESS RELEASE: PASS"
}


ensure_prelaunch_clean() {
    local free_mib

    if ps -eo cmd= |
        grep -E '[v]llm|[E]ngineCore|[a]pi_server' \
        >/dev/null
    then
        ps -eo pid,ppid,pgid,stat,etime,args |
            grep -E '[v]llm|[E]ngineCore|[a]pi_server' || true

        fail "existing vLLM/EngineCore process before sweep"
    fi

    free_mib="$(
        nvidia-smi -i 0 \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits |
        head -n1 |
        xargs
    )"

    echo "free MiB before sweep: $free_mib"

    test "$free_mib" -ge "$MIN_GPU_FREE_MIB" \
        || fail "GPU not sufficiently free before sweep"
}


wait_health() {
    local port="$1"
    local start_ts
    local now_ts
    local elapsed

    start_ts="$(date +%s)"

    while true; do

        if curl -fsS \
            --max-time 3 \
            "http://127.0.0.1:${port}/health" \
            >/dev/null 2>&1
        then
            now_ts="$(date +%s)"
            elapsed=$((now_ts - start_ts))

            echo "/health PASS after ${elapsed}s"
            return 0
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return 11
        fi

        now_ts="$(date +%s)"
        elapsed=$((now_ts - start_ts))

        if [ "$elapsed" -ge "$STARTUP_TIMEOUT_S" ]; then
            return 12
        fi

        echo "waiting for health... ${elapsed}s"
        sleep 3
    done
}


verify_model() {
    local port="$1"
    local expected="$2"

    curl -fsS \
        --max-time 10 \
        "http://127.0.0.1:${port}/v1/models" |
    EXPECTED_MODEL="$expected" python3 -c '
import json
import os
import sys

expected = os.environ["EXPECTED_MODEL"]

payload = json.load(sys.stdin)

models = [
    item.get("id")
    for item in payload.get("data", [])
    if isinstance(item, dict)
]

print("models =", models)

if expected not in models:
    raise SystemExit(
        f"expected served model={expected!r}, got={models!r}"
    )

print("MODEL IDENTITY: PASS")
'
}


verify_live_command() {
    local lane="$1"
    local port="$2"

    SERVER_PGID="$SERVER_PGID" \
    LANE="$lane" \
    PORT="$port" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

pgid = int(os.environ["SERVER_PGID"])
lane = os.environ["LANE"]
port = os.environ["PORT"]

target_model = "Qwen/Qwen2.5-7B-Instruct-AWQ"

served = {
    "vanilla": "qwen2.5-7b-vanilla",
    "static": "qwen2.5-7b-static-draft",
}[lane]

candidates = []

for proc in Path("/proc").iterdir():

    if not proc.name.isdigit():
        continue

    pid = int(proc.name)

    try:
        if os.getpgid(pid) != pgid:
            continue

        raw = (proc / "cmdline").read_bytes()

        args = [
            x.decode("utf-8", errors="replace")
            for x in raw.split(b"\0")
            if x
        ]

    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
    ):
        continue

    if (
        target_model in args
        and "--port" in args
        and port in args
    ):
        candidates.append((pid, args))


if len(candidates) != 1:
    raise SystemExit(
        f"expected exactly one vLLM server process, "
        f"found {len(candidates)}: {candidates!r}"
    )

pid, args = candidates[0]

print("vLLM PID =", pid)


def value(flag):
    try:
        idx = args.index(flag)
    except ValueError:
        raise SystemExit(f"missing flag {flag}")

    if idx + 1 >= len(args):
        raise SystemExit(f"flag has no value {flag}")

    return args[idx + 1]


expected = {
    "--host": "0.0.0.0",
    "--port": port,
    "--served-model-name": served,
    "--generation-config": "vllm",
    "--quantization": "awq",
    "--dtype": "half",
    "--gpu-memory-utilization": "0.85",
    "--max-model-len": "4096",
    "--max-num-seqs": "32",
}

for flag, expected_value in expected.items():

    actual = value(flag)

    if actual != expected_value:
        raise SystemExit(
            f"{flag}: expected={expected_value!r}, "
            f"actual={actual!r}"
        )


if lane == "vanilla":

    if "--speculative-config" in args:
        raise SystemExit(
            "Vanilla unexpectedly has --speculative-config"
        )

    if "--per-request-spec-decode-metrics" in args:
        raise SystemExit(
            "Vanilla unexpectedly enables spec-decode metrics"
        )

else:

    if value("--per-request-spec-decode-metrics") != "summary":
        raise SystemExit(
            "Static must use "
            "--per-request-spec-decode-metrics summary"
        )

    raw_spec = value("--speculative-config")

    spec = json.loads(raw_spec)

    expected_spec = {
        "method": "draft_model",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_speculative_tokens": 3,
        "use_heterogeneous_vocab": True,
        "draft_sample_method": "greedy",
    }

    if spec != expected_spec:
        raise SystemExit(
            f"static speculative-config mismatch: {spec!r}"
        )


print(f"{lane.upper()} LIVE SERVER CONTRACT: PASS")
PY
}


load_experiment_id() {
    python3 - "$CAMPAIGN_FILE" "$FROZEN_SHA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_sha = sys.argv[2]

data = json.loads(
    path.read_text(encoding="utf-8")
)

if data.get("schema") != "m1_02_campaign.v1":
    raise SystemExit("unexpected campaign schema")

if data.get("benchmark_commit") != expected_sha:
    raise SystemExit("benchmark_commit mismatch")

experiment_id = data.get("experiment_id")

if not isinstance(experiment_id, str) or not experiment_id:
    raise SystemExit("invalid experiment_id")

print(experiment_id)
PY
}


resume_rc() {
    local lane="$1"
    local workload="$2"

    set +e

    python3 "$RESUME_CHECK" \
        --evidence-repo "$EVID" \
        --lane "$lane" \
        --workload "$workload" \
        --experiment-id "$EXPERIMENT_ID"

    local rc=$?

    set -e

    return "$rc"
}


# ============================================================
# Campaign identity
# ============================================================

EXPERIMENT_ID="$(load_experiment_id)"

test "$EXPERIMENT_ID" = "$EXPECTED_EXPERIMENT_ID" \
    || fail "unexpected experiment ID: $EXPERIMENT_ID"

REL_DIR="experiments/runs/m1_02/$EXPERIMENT_ID"
WORK_DIR="/kaggle/working/m1_02_campaign/$EXPERIMENT_ID"

mkdir -p "$WORK_DIR"


# ============================================================
# Preflight
# ============================================================

echo "============================================================"
echo "M1-02 FULL CANONICAL CAMPAIGN"
echo "experiment_id=$EXPERIMENT_ID"
echo "============================================================"

PREFLIGHT_LOG="$WORK_DIR/preflight.log"

"$PREFLIGHT" 2>&1 | tee "$PREFLIGHT_LOG"

git -C "$EVID" push \
    --dry-run \
    origin "$EVID_BRANCH"

echo "PUSH AUTH: PASS"


# ============================================================
# Freeze this exact full-campaign runner + six run configs
# BEFORE any measurement.
# ============================================================

prepare_provenance() {
    local runner_source
    local runner_sha
    local runner_rel
    local runner_dst
    local durable_preflight

    runner_source="$(readlink -f "$0")"

    runner_sha="$(
        sha256sum "$runner_source" |
        awk '{print $1}'
    )"

    runner_rel="experiments/runs/m1_02/tooling/run_m1_02_full_campaign.sh"
    runner_dst="$EVID/$runner_rel"

    durable_preflight="$EVID/$REL_DIR/preflight.log"

    echo
    echo "=== FREEZE FULL CAMPAIGN PROVENANCE ==="
    echo "runner SHA256: $runner_sha"

    if [ -f "$runner_dst" ]; then

        existing_sha="$(
            sha256sum "$runner_dst" |
            awk '{print $1}'
        )"

        if [ "$existing_sha" != "$runner_sha" ]; then

            if find "$EVID/$REL_DIR" \
                -maxdepth 1 \
                -name '*.done.json' \
                -print -quit 2>/dev/null |
                grep -q .
            then
                fail \
                    "durable full runner differs after measurements exist"
            fi

            cp "$runner_source" "$runner_dst"
            chmod +x "$runner_dst"
        fi

    else
        cp "$runner_source" "$runner_dst"
        chmod +x "$runner_dst"
    fi


    if [ ! -f "$durable_preflight" ]; then
        mkdir -p "$(dirname "$durable_preflight")"
        cp "$PREFLIGHT_LOG" "$durable_preflight"
    fi


    RUNNER_SHA="$runner_sha" \
    RUNNER_REL="$runner_rel" \
    BENCH="$BENCH" \
    EVID="$EVID" \
    EXPERIMENT_ID="$EXPERIMENT_ID" \
    FROZEN_SHA="$FROZEN_SHA" \
    REL_DIR="$REL_DIR" \
    python3 - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

bench = Path(os.environ["BENCH"])
evid = Path(os.environ["EVID"])

experiment_id = os.environ["EXPERIMENT_ID"]
frozen_sha = os.environ["FROZEN_SHA"]
rel_dir = Path(os.environ["REL_DIR"])

runner_sha = os.environ["RUNNER_SHA"]
runner_rel = os.environ["RUNNER_REL"]


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


specs = [
    (
        "vanilla",
        "W1",
        "scripts/launch_lane1_vanilla.sh",
        "experiments/configs/w1/node1_chat_requests.jsonl",
        8001,
        "qwen2.5-7b-vanilla",
    ),
    (
        "static",
        "W1",
        "scripts/launch_lane2_static.sh",
        "experiments/configs/w1/node1_chat_requests.jsonl",
        8002,
        "qwen2.5-7b-static-draft",
    ),
    (
        "vanilla",
        "W2",
        "scripts/launch_lane1_vanilla.sh",
        "experiments/configs/w2_v2/chat_requests.jsonl",
        8001,
        "qwen2.5-7b-vanilla",
    ),
    (
        "static",
        "W2",
        "scripts/launch_lane2_static.sh",
        "experiments/configs/w2_v2/chat_requests.jsonl",
        8002,
        "qwen2.5-7b-static-draft",
    ),
    (
        "vanilla",
        "W3",
        "scripts/launch_lane1_vanilla.sh",
        "experiments/configs/w3_v2/chat_requests.jsonl",
        8001,
        "qwen2.5-7b-vanilla",
    ),
    (
        "static",
        "W3",
        "scripts/launch_lane2_static.sh",
        "experiments/configs/w3_v2/chat_requests.jsonl",
        8002,
        "qwen2.5-7b-static-draft",
    ),
]

harness = bench / "experiments/benchmark_harness.py"

for (
    lane,
    workload,
    launcher_rel,
    workload_rel,
    port,
    model,
) in specs:

    name = f"{lane}_{workload.lower()}"

    config_path = (
        evid
        / rel_dir
        / f"{name}.run_config.json"
    )

    done_path = (
        evid
        / rel_dir
        / f"{name}.done.json"
    )

    old = None

    if config_path.exists():
        old = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        old_runner_sha = (
            old.get("runner", {}).get("sha256")
        )

        if (
            done_path.exists()
            and old_runner_sha != runner_sha
        ):
            raise SystemExit(
                f"{name}: completed sweep has different "
                "runner provenance"
            )

        if old_runner_sha == runner_sha:
            continue

    launcher = bench / launcher_rel
    workload_file = bench / workload_rel

    config = {
        "schema": "m1_02_sweep_run_config.v2",
        "experiment_id": experiment_id,
        "lane": lane,
        "workload": workload,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "benchmark_commit": frozen_sha,
        "runner": {
            "path": runner_rel,
            "sha256": runner_sha,
        },
        "benchmark_harness": {
            "path": "experiments/benchmark_harness.py",
            "sha256": sha256(harness),
        },
        "launcher": {
            "path": launcher_rel,
            "sha256": sha256(launcher),
        },
        "workload_artifact": {
            "path": workload_rel,
            "sha256": sha256(workload_file),
        },
        "server": {
            "port": port,
            "endpoint": (
                f"http://127.0.0.1:{port}"
                "/v1/chat/completions"
            ),
            "served_model": model,
            "fresh_server_per_sweep": True,
            "restart_between_cells": False,
            "restart_between_repeats": False,
            "prefix_cache_reset_between_cells": False,
            "prefix_cache_reset_between_repeats": False,
        },
        "measurement": {
            "sample_policy": "cycle-rotate",
            "concurrency_order": [1, 2, 4, 8, 16, 32],
            "repeats_per_cell": 5,
            "measured_requests_per_repeat": 32,
            "requested_warmup_requests": 3,
            "warmup_policy": (
                "max(3, requested_warmup, concurrency)"
            ),
            "warmup_once_per_cell": True,
            "timeout_s": 120.0,
            "gpu_index": 0,
            "vram_sample_interval_s": 0.05,
            "expected_cells": 6,
            "expected_measured_repeats": 30,
            "expected_measured_requests": 960,
        },
    }

    if lane == "static":
        config["speculative_decoding"] = {
            "method": "draft_model",
            "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "num_speculative_tokens": 3,
            "use_heterogeneous_vocab": True,
            "draft_sample_method": "greedy",
            "per_request_spec_decode_metrics": "summary",
        }

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
PY


    (
        cd "$EVID/experiments/runs/m1_02/tooling"

        sha256sum \
            validate_m1_02_sweep.py \
            check_m1_02_resume.py \
            checkpoint_m1_02_sweep.sh \
            run_m1_02_preflight.sh \
            run_m1_02_vanilla_w1.sh \
            run_m1_02_full_campaign.sh \
            > SHA256SUMS
    )


    git -C "$EVID" add \
        "experiments/runs/m1_02/tooling/run_m1_02_full_campaign.sh" \
        "experiments/runs/m1_02/tooling/SHA256SUMS" \
        "$REL_DIR"/*.run_config.json \
        "$REL_DIR/preflight.log"


    if ! git -C "$EVID" diff --cached --quiet; then

        git -C "$EVID" diff --cached --check

        git -C "$EVID" commit \
            -m "evidence(m1-02): freeze full baseline campaign provenance"

        git -C "$EVID" push \
            origin "$EVID_BRANCH"
    fi


    local_sha="$(
        git -C "$EVID" rev-parse HEAD
    )"

    remote_sha="$(
        git -C "$EVID" ls-remote origin \
            "refs/heads/$EVID_BRANCH" |
        awk '{print $1}'
    )"

    echo "local evidence HEAD : $local_sha"
    echo "remote evidence HEAD: $remote_sha"

    test "$local_sha" = "$remote_sha" \
        || fail "provenance remote durability failed"

    test -z "$(
        git -C "$EVID" status --porcelain
    )" || fail "evidence checkout dirty after provenance freeze"

    echo "FULL CAMPAIGN PROVENANCE: DURABLE PASS"
}


prepare_provenance


# ============================================================
# One canonical lane/workload sweep
# ============================================================

run_sweep() {
    local lane="$1"
    local workload="$2"

    local launcher
    local port
    local model
    local workload_file

    local key
    local run_dir
    local raw_tmp
    local harness_log
    local server_log

    local done_rel
    local raw_rel
    local harness_log_rel
    local server_log_rel

    key="${lane}_${workload,,}"

    echo
    echo "============================================================"
    echo "SWEEP: $lane / $workload"
    echo "============================================================"

    set +e

    python3 "$RESUME_CHECK" \
        --evidence-repo "$EVID" \
        --lane "$lane" \
        --workload "$workload" \
        --experiment-id "$EXPERIMENT_ID"

    resume=$?

    set -e


    case "$resume" in
        0)
            echo "$lane/$workload COMPLETE_VALID -> SKIP"
            return 0
            ;;

        10)
            echo "$lane/$workload NOT_STARTED -> RUN WHOLE SWEEP"
            ;;

        *)
            fail \
                "$lane/$workload resume state INVALID rc=$resume"
            ;;
    esac


    case "$lane" in
        vanilla)
            launcher="$BENCH/scripts/launch_lane1_vanilla.sh"
            port=8001
            model=qwen2.5-7b-vanilla
            ;;

        static)
            launcher="$BENCH/scripts/launch_lane2_static.sh"
            port=8002
            model=qwen2.5-7b-static-draft
            ;;

        *)
            fail "unknown lane=$lane"
            ;;
    esac


    case "$workload" in
        W1)
            workload_file="$BENCH/experiments/configs/w1/node1_chat_requests.jsonl"
            ;;

        W2)
            workload_file="$BENCH/experiments/configs/w2_v2/chat_requests.jsonl"
            ;;

        W3)
            workload_file="$BENCH/experiments/configs/w3_v2/chat_requests.jsonl"
            ;;

        *)
            fail "unknown workload=$workload"
            ;;
    esac


    run_dir="$WORK_DIR/$key"

    mkdir -p "$run_dir"

    raw_tmp="$run_dir/${key}.jsonl"
    harness_log="$run_dir/${key}.harness.log"
    server_log="$run_dir/${key}.server.log"


    if [ -e "$raw_tmp" ]; then
        fail \
            "$lane/$workload partial local raw exists: $raw_tmp; " \
            "do not append/delete blindly"
    fi


    ensure_prelaunch_clean


    echo
    echo "=== START SERVER: $lane / $workload ==="

    : > "$server_log"

    setsid bash "$launcher" \
        >"$server_log" 2>&1 &

    SERVER_PID=$!

    sleep 1

    SERVER_PGID="$(
        ps -o pgid= -p "$SERVER_PID" |
        xargs
    )"

    test -n "$SERVER_PGID" \
        || fail "could not resolve server PGID"

    echo "launcher PID: $SERVER_PID"
    echo "server PGID : $SERVER_PGID"
    echo "server log  : $server_log"


    echo
    echo "=== WAIT HEALTH ==="

    set +e
    wait_health "$port"
    health_rc=$?
    set -e

    if [ "$health_rc" -ne 0 ]; then
        echo "===== SERVER LOG TAIL ====="
        tail -n 200 "$server_log" || true
        fail \
            "$lane/$workload server readiness failed rc=$health_rc"
    fi


    verify_model "$port" "$model"

    verify_live_command "$lane" "$port"


    echo
    echo "=== RUN CANONICAL HARNESS ==="

    cd "$BENCH"

    python3 "$HARNESS" \
        --endpoint "http://127.0.0.1:${port}/v1/chat/completions" \
        --model "$model" \
        --workload-file "$workload_file" \
        --sample-policy cycle-rotate \
        --num-requests 32 \
        --concurrency-sweep 1 2 4 8 16 32 \
        --repeats 5 \
        --warmup-requests 3 \
        --lane "$lane" \
        --workload "$workload" \
        --experiment-id "$EXPERIMENT_ID" \
        --raw-out "$raw_tmp" \
        --summary-csv "$run_dir/noncanonical_harness_summary.csv" \
        --overwrite \
        2>&1 |
    tee "$harness_log"


    echo
    echo "=== VALIDATE 960-RECORD SWEEP ==="

    python3 "$VALIDATOR" \
        --raw "$raw_tmp" \
        --lane "$lane" \
        --workload "$workload" \
        --experiment-id "$EXPERIMENT_ID"


    raw_sha="$(
        sha256sum "$raw_tmp" |
        awk '{print $1}'
    )"

    echo "validated raw SHA256=$raw_sha"


    echo
    echo "=== SHUT DOWN SERVER ==="

    cleanup_server

    SERVER_PID=""
    SERVER_PGID=""

    wait_gpu_release


    echo
    echo "=== CHECKPOINT RAW → EVIDENCE ==="

    "$CHECKPOINT" \
        "$EVID" \
        "$raw_tmp" \
        "$lane" \
        "$workload" \
        "$EXPERIMENT_ID"


    done_rel="$REL_DIR/${key}.done.json"
    raw_rel="$REL_DIR/${key}.jsonl"
    harness_log_rel="$REL_DIR/${key}.harness.log"
    server_log_rel="$REL_DIR/${key}.server.log"

    cp \
        "$harness_log" \
        "$EVID/$harness_log_rel"

    cp \
        "$server_log" \
        "$EVID/$server_log_rel"


    HARNESS_LOG_SHA="$(
        sha256sum "$EVID/$harness_log_rel" |
        awk '{print $1}'
    )"

    SERVER_LOG_SHA="$(
        sha256sum "$EVID/$server_log_rel" |
        awk '{print $1}'
    )"


    DONE="$EVID/$done_rel" \
    HARNESS_LOG_REL="$harness_log_rel" \
    SERVER_LOG_REL="$server_log_rel" \
    HARNESS_LOG_SHA="$HARNESS_LOG_SHA" \
    SERVER_LOG_SHA="$SERVER_LOG_SHA" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DONE"])

data = json.loads(
    path.read_text(encoding="utf-8")
)

data["harness_log_path"] = os.environ["HARNESS_LOG_REL"]
data["harness_log_sha256"] = os.environ["HARNESS_LOG_SHA"]

data["server_log_path"] = os.environ["SERVER_LOG_REL"]
data["server_log_sha256"] = os.environ["SERVER_LOG_SHA"]

path.write_text(
    json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY


    git -C "$EVID" add \
        "$done_rel" \
        "$harness_log_rel" \
        "$server_log_rel"

    git -C "$EVID" diff --cached --check

    git -C "$EVID" commit \
        --amend \
        --no-edit


    echo
    echo "=== REVERIFY COPIED RAW ==="

    python3 "$VALIDATOR" \
        --raw "$EVID/$raw_rel" \
        --lane "$lane" \
        --workload "$workload" \
        --experiment-id "$EXPERIMENT_ID"


    copied_sha="$(
        sha256sum "$EVID/$raw_rel" |
        awk '{print $1}'
    )"

    test "$copied_sha" = "$raw_sha" \
        || fail "$lane/$workload raw SHA changed after copy"


    echo
    echo "=== PUSH SWEEP EVIDENCE ==="

    git -C "$EVID" push \
        origin "$EVID_BRANCH"


    local_sha="$(
        git -C "$EVID" rev-parse HEAD
    )"

    remote_sha="$(
        git -C "$EVID" ls-remote origin \
            "refs/heads/$EVID_BRANCH" |
        awk '{print $1}'
    )"

    echo "local evidence HEAD : $local_sha"
    echo "remote evidence HEAD: $remote_sha"

    test "$local_sha" = "$remote_sha" \
        || fail "$lane/$workload remote durability mismatch"


    echo
    echo "=== POST-PUSH RESUME CHECK ==="

    python3 "$RESUME_CHECK" \
        --evidence-repo "$EVID" \
        --lane "$lane" \
        --workload "$workload" \
        --experiment-id "$EXPERIMENT_ID"


    test -z "$(
        git -C "$EVID" status --porcelain
    )" || fail "evidence checkout dirty after $lane/$workload"


    echo
    echo "============================================================"
    echo "$lane/$workload: DURABLE PASS"
    echo "raw_sha256=$raw_sha"
    echo "============================================================"
}


# ============================================================
# Required PM-approved interleaved order
# ============================================================

run_sweep vanilla W1
run_sweep static  W1

run_sweep vanilla W2
run_sweep static  W2

run_sweep vanilla W3
run_sweep static  W3


# ============================================================
# Verify all 6 are durable before aggregation
# ============================================================

echo
echo "============================================================"
echo "VERIFY ALL SIX SWEEPS"
echo "============================================================"

for spec in \
    "vanilla W1" \
    "static W1" \
    "vanilla W2" \
    "static W2" \
    "vanilla W3" \
    "static W3"
do
    set -- $spec

    python3 "$RESUME_CHECK" \
        --evidence-repo "$EVID" \
        --lane "$1" \
        --workload "$2" \
        --experiment-id "$EXPERIMENT_ID"
done


# ============================================================
# If final campaign marker already exists, verify and exit.
# ============================================================

RUN_COMPLETE="$EVID/$REL_DIR/RUN_COMPLETE.json"
FINAL_SUMMARY="$EVID/experiments/results/summary.csv"

if [ -f "$RUN_COMPLETE" ]; then

    echo
    echo "RUN_COMPLETE.json already exists; verifying..."

    RUN_COMPLETE="$RUN_COMPLETE" \
    FINAL_SUMMARY="$FINAL_SUMMARY" \
    EVID="$EVID" \
    python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

marker_path = Path(os.environ["RUN_COMPLETE"])
summary_path = Path(os.environ["FINAL_SUMMARY"])
evid = Path(os.environ["EVID"])

marker = json.loads(
    marker_path.read_text(encoding="utf-8")
)

if marker.get("status") != "COMPLETE_VALID":
    raise SystemExit("RUN_COMPLETE status is not COMPLETE_VALID")

summary_sha = hashlib.sha256(
    summary_path.read_bytes()
).hexdigest()

if summary_sha != marker["summary_sha256"]:
    raise SystemExit("summary SHA mismatch")

for item in marker["raw_files"]:
    path = evid / item["path"]

    actual = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if actual != item["sha256"]:
        raise SystemExit(
            f"raw SHA mismatch: {item['path']}"
        )

print("EXISTING RUN_COMPLETE: VERIFIED")
PY

    echo
    echo "M1-02 FULL CAMPAIGN: ALREADY DURABLE COMPLETE"
    exit 0
fi


# ============================================================
# Aggregate the six validated raw files -> summary.csv
# ============================================================

echo
echo "============================================================"
echo "AGGREGATE RAW → experiments/results/summary.csv"
echo "============================================================"

RAW_FILES=(
    "$EVID/$REL_DIR/vanilla_w1.jsonl"
    "$EVID/$REL_DIR/static_w1.jsonl"
    "$EVID/$REL_DIR/vanilla_w2.jsonl"
    "$EVID/$REL_DIR/static_w2.jsonl"
    "$EVID/$REL_DIR/vanilla_w3.jsonl"
    "$EVID/$REL_DIR/static_w3.jsonl"
)

SUMMARY_TMP="$WORK_DIR/summary.csv"
AGG_LOG_TMP="$WORK_DIR/aggregate_results.log"

python3 "$AGGREGATOR" \
    --raw "${RAW_FILES[@]}" \
    --output-csv "$SUMMARY_TMP" \
    --overwrite \
    2>&1 |
tee "$AGG_LOG_TMP"


# ============================================================
# Validate canonical summary invariants
# ============================================================

SUMMARY_TMP="$SUMMARY_TMP" \
FROZEN_SHA="$FROZEN_SHA" \
python3 - <<'PY'
import csv
import math
import os
from pathlib import Path

path = Path(os.environ["SUMMARY_TMP"])
frozen_sha = os.environ["FROZEN_SHA"]

with path.open(
    encoding="utf-8",
    newline="",
) as f:
    rows = list(csv.DictReader(f))

print("summary rows =", len(rows))

if len(rows) != 36:
    raise SystemExit(
        f"expected exactly 36 summary rows, got {len(rows)}"
    )

expected = {
    (lane, workload, concurrency)
    for lane in ("vanilla", "static")
    for workload in ("W1", "W2", "W3")
    for concurrency in (1, 2, 4, 8, 16, 32)
}

actual = {
    (
        row["lane"],
        row["workload"],
        int(row["concurrency"]),
    )
    for row in rows
}

if actual != expected:
    raise SystemExit(
        f"summary cell coverage mismatch; "
        f"missing={sorted(expected-actual)}, "
        f"extra={sorted(actual-expected)}"
    )

for row in rows:

    cell = (
        row["lane"],
        row["workload"],
        row["concurrency"],
    )

    if int(row["n_runs"]) != 5:
        raise SystemExit(
            f"{cell}: n_runs != 5"
        )

    if row["commit"] != frozen_sha:
        raise SystemExit(
            f"{cell}: wrong benchmark commit"
        )

    speedup = float(
        row["speedup_vs_vanilla"]
    )

    if not math.isfinite(speedup) or speedup <= 0:
        raise SystemExit(
            f"{cell}: invalid speedup={speedup}"
        )

    if row["lane"] == "vanilla":

        if not math.isclose(
            speedup,
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                f"{cell}: vanilla speedup != 1"
            )

        if row["alpha"] != "N/A":
            raise SystemExit(
                f"{cell}: vanilla alpha should be N/A"
            )

        if row["tau"] != "N/A":
            raise SystemExit(
                f"{cell}: vanilla tau should be N/A"
            )

    else:

        alpha = float(row["alpha"])
        tau = float(row["tau"])

        if not (0.0 <= alpha <= 1.0):
            raise SystemExit(
                f"{cell}: alpha out of range"
            )

        if not (1.0 <= tau <= 4.0):
            raise SystemExit(
                f"{cell}: tau out of range"
            )

print("SUMMARY CANONICAL VALIDATION: PASS")
PY


# ============================================================
# Copy canonical summary + aggregation log
# ============================================================

mkdir -p "$EVID/experiments/results"

cp \
    "$SUMMARY_TMP" \
    "$EVID/experiments/results/summary.csv"

cp \
    "$AGG_LOG_TMP" \
    "$EVID/$REL_DIR/aggregate_results.log"


# ============================================================
# Build RUN_COMPLETE marker
# ============================================================

EVID="$EVID" \
REL_DIR="$REL_DIR" \
EXPERIMENT_ID="$EXPERIMENT_ID" \
FROZEN_SHA="$FROZEN_SHA" \
python3 - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

evid = Path(os.environ["EVID"])
rel_dir = Path(os.environ["REL_DIR"])

experiment_id = os.environ["EXPERIMENT_ID"]
frozen_sha = os.environ["FROZEN_SHA"]

names = [
    "vanilla_w1.jsonl",
    "static_w1.jsonl",
    "vanilla_w2.jsonl",
    "static_w2.jsonl",
    "vanilla_w3.jsonl",
    "static_w3.jsonl",
]


def sha256(path: Path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


raw_files = []

total_lines = 0

for name in names:

    rel = rel_dir / name
    path = evid / rel

    lines = sum(
        1
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )

    if lines != 960:
        raise SystemExit(
            f"{name}: expected 960 measured rows, got {lines}"
        )

    total_lines += lines

    raw_files.append({
        "path": rel.as_posix(),
        "sha256": sha256(path),
        "measured_records": lines,
    })


if total_lines != 5760:
    raise SystemExit(
        f"expected total=5760, got={total_lines}"
    )


summary = evid / "experiments/results/summary.csv"

summary_rows = (
    len(
        summary.read_text(
            encoding="utf-8"
        ).splitlines()
    )
    - 1
)

if summary_rows != 36:
    raise SystemExit(
        f"expected 36 summary rows, got {summary_rows}"
    )


marker = {
    "schema": "m1_02_campaign_complete.v1",
    "status": "COMPLETE_VALID",
    "experiment_id": experiment_id,
    "benchmark_commit": frozen_sha,
    "completed_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "sweeps": 6,
    "cells": 36,
    "measured_repeats": 180,
    "measured_requests": total_lines,
    "raw_files": raw_files,
    "summary_path": "experiments/results/summary.csv",
    "summary_rows": summary_rows,
    "summary_sha256": sha256(summary),
}

output = evid / rel_dir / "RUN_COMPLETE.json"

output.write_text(
    json.dumps(
        marker,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print(json.dumps(marker, indent=2))
PY


# ============================================================
# Final evidence commit + remote verification
# ============================================================

git -C "$EVID" add \
    "experiments/results/summary.csv" \
    "$REL_DIR/aggregate_results.log" \
    "$REL_DIR/RUN_COMPLETE.json"

git -C "$EVID" diff --cached --check

git -C "$EVID" commit \
    -m "evidence(m1-02): finalize baseline raw telemetry and summary"

git -C "$EVID" push \
    origin "$EVID_BRANCH"


LOCAL_SHA="$(
    git -C "$EVID" rev-parse HEAD
)"

REMOTE_SHA="$(
    git -C "$EVID" ls-remote origin \
        "refs/heads/$EVID_BRANCH" |
    awk '{print $1}'
)"

echo "local evidence HEAD : $LOCAL_SHA"
echo "remote evidence HEAD: $REMOTE_SHA"

test "$LOCAL_SHA" = "$REMOTE_SHA" \
    || fail "final remote durability mismatch"


test -z "$(
    git -C "$EVID" status --porcelain
)" || fail "evidence checkout dirty after finalization"


echo
echo "============================================================"
echo "M1-02 FULL CAMPAIGN: DURABLE PASS"
echo "experiment_id=$EXPERIMENT_ID"
echo "raw files        = 6"
echo "measured requests= 5760"
echo "summary rows     = 36"
echo "summary path     = experiments/results/summary.csv"
echo "============================================================"
