#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "usage:" >&2
    echo "  $0 <evidence_repo> <raw> <lane> <workload> <experiment_id>" >&2
    exit 2
fi

EVID_REPO="$1"
RAW="$2"
LANE="$3"
WORKLOAD="$4"
EXPERIMENT_ID="$5"

BENCH_REPO=/kaggle/working/do-an-ptit-d22
VALIDATOR=/kaggle/working/validate_m1_02_sweep.py
FROZEN_SHA=f307e33f9aea4bbd45499f548480cb7d6f90a4b9

fail() {
    echo "CHECKPOINT FAIL: $*" >&2
    exit 1
}

case "$LANE" in
    vanilla|static) ;;
    *) fail "invalid lane: $LANE" ;;
esac

case "$WORKLOAD" in
    W1|W2|W3) ;;
    *) fail "invalid workload: $WORKLOAD" ;;
esac

test -d "$EVID_REPO/.git" \
    || fail "evidence repo missing"

test -f "$RAW" \
    || fail "raw file missing"

test -x "$VALIDATOR" \
    || fail "validator missing/not executable"

test "$(git -C "$BENCH_REPO" rev-parse HEAD)" = "$FROZEN_SHA" \
    || fail "benchmark source not frozen"

test -z "$(git -C "$BENCH_REPO" status --porcelain)" \
    || fail "benchmark checkout dirty"

test -z "$(git -C "$EVID_REPO" status --porcelain)" \
    || fail "evidence checkout dirty before checkpoint"

echo "=== 1. VALIDATE RAW ==="

python3 "$VALIDATOR" \
    --raw "$RAW" \
    --lane "$LANE" \
    --workload "$WORKLOAD" \
    --experiment-id "$EXPERIMENT_ID"

echo
echo "=== 2. HASH RAW ==="

RAW_SHA="$(
    sha256sum "$RAW" |
    awk '{print $1}'
)"

RAW_RECORDS="$(
    wc -l < "$RAW" |
    xargs
)"

echo "raw SHA256 : $RAW_SHA"
echo "raw records: $RAW_RECORDS"

test "$RAW_RECORDS" -eq 960 \
    || fail "validated file unexpectedly not 960 lines"

echo
echo "=== 3. PREPARE CANONICAL PATH ==="

WORKLOAD_LOWER="$(
    printf '%s' "$WORKLOAD" |
    tr '[:upper:]' '[:lower:]'
)"

REL_DIR="experiments/runs/m1_02/$EXPERIMENT_ID"
RAW_REL="$REL_DIR/${LANE}_${WORKLOAD_LOWER}.jsonl"
DONE_REL="$REL_DIR/${LANE}_${WORKLOAD_LOWER}.done.json"

RAW_DST="$EVID_REPO/$RAW_REL"
DONE_DST="$EVID_REPO/$DONE_REL"

echo "raw destination : $RAW_REL"
echo "done marker     : $DONE_REL"

if [ -e "$RAW_DST" ] || [ -e "$DONE_DST" ]; then
    fail "checkpoint destination already exists"
fi

mkdir -p "$EVID_REPO/$REL_DIR"

TMP_RAW="${RAW_DST}.tmp.$$"
cp "$RAW" "$TMP_RAW"

COPIED_SHA="$(
    sha256sum "$TMP_RAW" |
    awk '{print $1}'
)"

test "$COPIED_SHA" = "$RAW_SHA" \
    || fail "copied raw SHA mismatch"

mv "$TMP_RAW" "$RAW_DST"

echo
echo "=== 4. CREATE DONE MARKER ==="

CREATED_UTC="$(
    date -u +"%Y-%m-%dT%H:%M:%SZ"
)"

python3 - \
    "$DONE_DST" \
    "$EXPERIMENT_ID" \
    "$LANE" \
    "$WORKLOAD" \
    "$RAW_REL" \
    "$RAW_SHA" \
    "$RAW_RECORDS" \
    "$FROZEN_SHA" \
    "$CREATED_UTC" <<'PY'
import json
import sys
from pathlib import Path

(
    done_path,
    experiment_id,
    lane,
    workload,
    raw_rel,
    raw_sha,
    raw_records,
    frozen_sha,
    created_utc,
) = sys.argv[1:]

marker = {
    "schema": "m1_02_sweep_done.v1",
    "experiment_id": experiment_id,
    "lane": lane,
    "workload": workload,
    "benchmark_commit": frozen_sha,
    "raw_path": raw_rel,
    "raw_sha256": raw_sha,
    "measured_records": int(raw_records),
    "expected_measured_records": 960,
    "validator": "validate_m1_02_sweep.py",
    "validation_status": "PASS",
    "created_utc": created_utc,
}

Path(done_path).write_text(
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

echo
echo "=== 5. REVERIFY CHECKPOINT ==="

test "$(
    sha256sum "$RAW_DST" |
    awk '{print $1}'
)" = "$RAW_SHA" \
    || fail "final raw SHA mismatch"

python3 "$VALIDATOR" \
    --raw "$RAW_DST" \
    --lane "$LANE" \
    --workload "$WORKLOAD" \
    --experiment-id "$EXPERIMENT_ID"

echo
echo "=== 6. STAGE ONLY THIS SWEEP ==="

git -C "$EVID_REPO" add \
    "$RAW_REL" \
    "$DONE_REL"

git -C "$EVID_REPO" diff --cached --check

git -C "$EVID_REPO" diff --cached --name-status

echo
echo "=== 7. LOCAL CHECKPOINT COMMIT ==="

git -C "$EVID_REPO" commit \
    -m "evidence(m1-02): checkpoint ${LANE} ${WORKLOAD} ${EXPERIMENT_ID}"

echo
echo "========================================"
echo "LOCAL SWEEP CHECKPOINT: PASS"
echo "No push was performed by this test helper."
echo "========================================"
