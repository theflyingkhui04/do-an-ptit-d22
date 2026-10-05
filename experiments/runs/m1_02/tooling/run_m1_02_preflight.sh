#!/usr/bin/env bash
set -euo pipefail

BENCH_REPO=/kaggle/working/do-an-ptit-d22
EVID_REPO=/kaggle/working/do-an-ptit-d22-evidence

FROZEN_SHA=f307e33f9aea4bbd45499f548480cb7d6f90a4b9
BENCH_BRANCH=feat/m1-02-concurrency-sweep-baseline
EVID_BRANCH=evidence/m1-02-f307e33

EXPECTED_GPU="Tesla T4"
EXPECTED_DRIVER="580.178.04"
MIN_GPU_FREE_MIB=13000

W1_REL=experiments/configs/w1/node1_chat_requests.jsonl
W2_REL=experiments/configs/w2_v2/chat_requests.jsonl
W3_REL=experiments/configs/w3_v2/chat_requests.jsonl

W1_SHA=0de2387c4e6a8c39523bf869788a55ee7487c6a32318a949ce22e1f8f7b17360
W2_SHA=bc26be01a4f5a813a4f6e843e2533e2f56d2e663918be2bd0e7b231e6385685c
W3_SHA=57341b0fc8c2ebbdfcfc579a6343aea66b08ca11fe2bfed5bc53a2bd49f5d83b

fail() {
    echo "PREFLIGHT FAIL: $*" >&2
    exit 1
}

echo "=== 1. REQUIRED TOOLS ==="
for cmd in git python3 curl nvidia-smi sha256sum awk; do
    command -v "$cmd" >/dev/null 2>&1 || fail "missing command: $cmd"
done
echo "TOOLS: PASS"

echo
echo "=== 2. CHECKOUTS ==="
test -d "$BENCH_REPO/.git" || fail "benchmark checkout missing"
test -d "$EVID_REPO/.git" || fail "evidence checkout missing"

BENCH_HEAD="$(git -C "$BENCH_REPO" rev-parse HEAD)"
BENCH_CURRENT_BRANCH="$(git -C "$BENCH_REPO" branch --show-current)"
BENCH_STATUS="$(git -C "$BENCH_REPO" status --porcelain)"

echo "benchmark branch : $BENCH_CURRENT_BRANCH"
echo "benchmark HEAD   : $BENCH_HEAD"

test "$BENCH_HEAD" = "$FROZEN_SHA" \
    || fail "benchmark HEAD is not frozen SHA"

test "$BENCH_CURRENT_BRANCH" = "$BENCH_BRANCH" \
    || fail "unexpected benchmark branch"

test -z "$BENCH_STATUS" \
    || fail "benchmark checkout is dirty"

echo "BENCHMARK CHECKOUT: PASS"

echo
echo "=== 3. EVIDENCE CHECKOUT ==="
EVID_CURRENT_BRANCH="$(git -C "$EVID_REPO" branch --show-current)"
EVID_STATUS="$(git -C "$EVID_REPO" status --porcelain)"
EVID_LOCAL_SHA="$(git -C "$EVID_REPO" rev-parse HEAD)"
EVID_REMOTE_SHA="$(
    git -C "$EVID_REPO" ls-remote origin \
        "refs/heads/$EVID_BRANCH" |
    awk '{print $1}'
)"

echo "evidence branch     : $EVID_CURRENT_BRANCH"
echo "evidence local HEAD : $EVID_LOCAL_SHA"
echo "evidence remote HEAD: $EVID_REMOTE_SHA"

test "$EVID_CURRENT_BRANCH" = "$EVID_BRANCH" \
    || fail "wrong evidence branch"

test -z "$EVID_STATUS" \
    || fail "evidence checkout is dirty"

test -n "$EVID_REMOTE_SHA" \
    || fail "remote evidence branch missing"

test "$EVID_LOCAL_SHA" = "$EVID_REMOTE_SHA" \
    || fail "local evidence branch is not synced to remote"

echo "EVIDENCE CHECKOUT: PASS"

echo
echo "=== 4. CUDA VISIBILITY ==="
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo
echo "=== 5. PYTHON RUNTIME CONTRACT ==="
python3 - <<'PY'
import importlib.metadata as md
import os
import torch

expected = {
    "torch": "2.13.0+cu130",
    "torch_cuda": "13.0",
    "vllm": "0.30.0",
    "aiohttp": "3.13.5",
    "nvidia-ml-py": "13.595.45",
}

actual = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": md.version("vllm"),
    "aiohttp": md.version("aiohttp"),
    "nvidia-ml-py": md.version("nvidia-ml-py"),
}

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch devices        =", torch.cuda.device_count())

if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"expected exactly 1 visible CUDA device, got {torch.cuda.device_count()}"
    )

gpu = torch.cuda.get_device_name(0)
print("torch GPU            =", gpu)

if gpu != "Tesla T4":
    raise SystemExit(f"expected Tesla T4, got {gpu!r}")

for key, expected_value in expected.items():
    actual_value = actual[key]
    print(f"{key:20s} = {actual_value}")
    if actual_value != expected_value:
        raise SystemExit(
            f"{key} mismatch: expected {expected_value!r}, "
            f"got {actual_value!r}"
        )

print("PYTHON RUNTIME: PASS")
PY

echo
echo "=== 6. NVIDIA DRIVER / GPU FREE MEMORY ==="

GPU_NAME="$(
    nvidia-smi -i 0 \
        --query-gpu=name \
        --format=csv,noheader |
    head -n 1 |
    xargs
)"

DRIVER="$(
    nvidia-smi -i 0 \
        --query-gpu=driver_version \
        --format=csv,noheader |
    head -n 1 |
    xargs
)"

GPU_FREE="$(
    nvidia-smi -i 0 \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits |
    head -n 1 |
    xargs
)"

echo "GPU name : $GPU_NAME"
echo "driver   : $DRIVER"
echo "free MiB : $GPU_FREE"

test "$GPU_NAME" = "$EXPECTED_GPU" \
    || fail "unexpected physical GPU"

test "$DRIVER" = "$EXPECTED_DRIVER" \
    || fail "driver mismatch"

test "$GPU_FREE" -ge "$MIN_GPU_FREE_MIB" \
    || fail "GPU free memory below ${MIN_GPU_FREE_MIB} MiB"

echo "GPU CONTRACT: PASS"

echo
echo "=== 7. WORKLOAD ARTIFACTS ==="

verify_sha() {
    local rel="$1"
    local expected="$2"
    local path="$BENCH_REPO/$rel"

    test -f "$path" || fail "missing workload: $rel"

    local actual
    actual="$(sha256sum "$path" | awk '{print $1}')"

    echo "$rel"
    echo "  expected: $expected"
    echo "  actual  : $actual"

    test "$actual" = "$expected" \
        || fail "workload SHA mismatch: $rel"
}

verify_sha "$W1_REL" "$W1_SHA"
verify_sha "$W2_REL" "$W2_SHA"
verify_sha "$W3_REL" "$W3_SHA"

echo "WORKLOAD ARTIFACTS: PASS"

echo
echo "=== 8. BENCHMARK ENTRYPOINTS ==="
for rel in \
    experiments/benchmark_harness.py \
    experiments/aggregate_results.py \
    scripts/launch_lane1_vanilla.sh \
    scripts/launch_lane2_static.sh
do
    test -f "$BENCH_REPO/$rel" || fail "missing: $rel"
    echo "FOUND: $rel"
done

python3 -m py_compile \
    "$BENCH_REPO/experiments/benchmark_harness.py" \
    "$BENCH_REPO/experiments/aggregate_results.py" \
    "$BENCH_REPO/experiments/workload_adapter.py"

bash -n "$BENCH_REPO/scripts/launch_lane1_vanilla.sh"
bash -n "$BENCH_REPO/scripts/launch_lane2_static.sh"

echo "ENTRYPOINTS: PASS"

echo
echo "========================================"
echo "M1-02 PRE-FLIGHT: PASS"
echo "No server was started."
echo "No benchmark request was sent."
echo "No campaign ID was created."
echo "========================================"
