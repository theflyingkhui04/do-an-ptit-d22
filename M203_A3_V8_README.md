# M2-03 A3 v8 — bounded async-safe diagnostic hotfix

**Pinned runtime:** vLLM 0.30.0, PyTorch 2.13.0+cu130, Tesla T4.  
**Git checkpoint:** `ffee9247d1978d25f41d935a486aa93564b17859`.  
**Canonical r03 remains 44/54 FAIL.** Do not rerun or replace canonical evidence to force PASS.

## Why v7 crashed

The v7 probe raises `ProbeError("A3 v7 excludes async scheduling")` inside the V1 EngineCore's `sample_tokens()` during the first request; EngineCore returns HTTP 500. In upstream vLLM 0.30.0 async speculative decoding, the V1 request cache may contain optimistic `-1` placeholders. Therefore deleting the async guard while using `len(req.output_token_ids)` to identify generated index 200 is **incorrect**.

## What changes in v8

- V1 site-packages patched **v7 hook** remains at the same sampling point, but its two imports are switched from `m2_03_v1_target_logits_probe` to `m2_03_v1_target_logits_probe_v8`. The installer verifies the exact current v7 SHA `06e685...`, makes a new backup, and fails closed on drift.
- v8 derives generated-token index from **actual model positions and `logits_indices`**, not from optimistic request output length. It checks sampler selection and accepted intra-step token inputs. It never modifies logits or sampling results.
- Exceptions inside diagnostic probe become `[M2-03 A3 V8 SKIP]` messages, so observer failure does not abort inference; **a SKIP is missing evidence, not PASS**.
- v8 records a preliminary sampler-validated row with `PENDING_API_FULL_PREFIX_VERIFICATION`, NOT a committed prefix. Offline comparator validates HTTP response `id`, emitted token IDs, entire prompt+200-token prefix hash and all active model-forward input positions. Only then can Vanilla-V2/Adaptive-V1 raw logits be compared.
- All run output is under new `experiments/runs/m2_03/diagnostics/a3_v1_v8_adaptive_*/`; V7 failed logs, canonical `r03`, and Trello are not modified. The diagnostic adds CPU/GPU synchronization, so it is **NOT a performance benchmark**.

## Deploy on Kaggle (terminal A)

Copy `m203_a3_v8.zip` into the **repository root** in VS Code, then:

```bash
cd /kaggle/working/do-an-ptit-d22
set -e
unzip -t m203_a3_v8.zip
for f in \
  experiments/m2_03_v1_target_logits_probe_v8.py \
  scripts/m2_03_install_a3_v8.py \
  scripts/m2_03_launch_adaptive_a3_v8.sh \
  scripts/m2_03_compare_a3_v8.py \
  scripts/m2_03_finish_a3_v8.py \
  tests/test_m2_03_v1_target_logits_v8.py; do
  test ! -e "$f" || { echo "ALREADY EXISTS: $f; stop"; exit 2; }
done
unzip -n m203_a3_v8.zip -d .
.venv/bin/python -m pytest -q \
  tests/test_m2_03_v1_target_logits_v7.py \
  tests/test_m2_03_v1_target_logits_v8.py
.venv/bin/python scripts/m2_03_install_a3_v8.py status
.venv/bin/python scripts/m2_03_install_a3_v8.py install
.venv/bin/python scripts/m2_03_install_a3_v8.py status
```

Required status: `A3_V8_HOOK_STATUS: V8_INSTALLED`. If any command fails, stop and preserve output. Do not override SHA guards. Offline tests do not prove GPU integration.

## One focused GPU diagnostic

In **terminal A**, launch foreground and leave running:

```bash
cd /kaggle/working/do-an-ptit-d22
bash scripts/m2_03_launch_adaptive_a3_v8.sh
```

In **terminal B**, once model startup completes (may require CUDA compilation):

```bash
cd /kaggle/working/do-an-ptit-d22
set -e
for i in $(seq 1 120); do
  if curl -fsS --max-time 5 http://127.0.0.1:8003/health >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS --max-time 5 http://127.0.0.1:8003/health >/dev/null
V="$PWD/experiments/runs/m2_03/diagnostics/a3_vanilla_20261013T125113Z_7nwv"
test -f "$V/vanilla_response.json"
test -f "$V/probe/records.jsonl"
printf '%s\n' "$V" > /tmp/m203_a3_smoke_dir
.venv/bin/python scripts/m2_03_run_adaptive_a3_two_requests_v6.py
.venv/bin/python scripts/m2_03_finish_a3_v8.py
D="$(cat /tmp/m203_a3_adaptive_dir)"
printf 'EVIDENCE_DIR=%s\n' "$D"
cat "$D/a3_v8_diagnostic_summary.json"
grep -E 'M2-03 A3 V8 (PRE|RECORD|SKIP|REJECTED_EARLY)' "$D/server.log" | tail -n 40 || true
```

`m2_03_run_adaptive_a3_two_requests_v6.py` is reused only as the frozen W3 request driver. **Do not use the v6 hook or launcher.** Two requests may fail to reproduce the original index-200 mismatch; that negative observation must be retained. The offline `finish` script automatically selects only responses with first mismatch at 200 and refuses to compare unverified/missing records. A `TARGET_200_EVIDENCE_VALIDATED` record still does **not** establish KV-cache equivalence, root cause, or canonical lossless PASS.

## Undo v8 without touching original source backup

Only after inspecting current status:

```bash
.venv/bin/python scripts/m2_03_install_a3_v8.py status
.venv/bin/python scripts/m2_03_install_a3_v8.py restore
```

This switches the v8 import back to the exact v7 hook, without deleting either original v7 or v8 backup. The old v7 installer can restore the original uninstrumented V1 runner **only after v8 restore**.

## Sources (pinned)

- https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/worker/gpu_model_runner/
- https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/worker/gpu_input_batch/
- https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/sample/rejection_sampler/

GPU integration has **not** been executed by the assistant. Proven offline: v8 unit tests plus source/shell syntax. Scientific interpretation remains evidence-dependent.
