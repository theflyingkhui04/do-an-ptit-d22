# Code Plan M1-01 — Dương Telemetry & Smoke-Test Workstream

> **Task:** M1-01  
> **Branch:** `feat/m1-01-vllm-dual-lane-server`  
> **Focus:** Dương Partner 30%  
> **Base reviewed:** branch head `737f2e8b3dc3a61ba1614d29b428dc91567a9627`  
> **Status:** Plan only — chưa sửa Git.

---

# 0. Goal

Hoàn tất hai checklist item của Dương:

```text
[Dương] Chuẩn hóa telemetry JSON theo Interface A
[Dương] Smoke test 10 requests, VRAM stability, handoff Đại
```

Không mở rộng sang M1-02 benchmark sweep.

---

# 1. Planned File Map

## CREATE

```text
experiments/configs/m1_request_telemetry.schema.json
experiments/configs/m1_request_telemetry.example.json

scripts/check_m1_telemetry.py
scripts/smoke_m1_01_telemetry.py

tests/test_m1_telemetry_contract.py
tests/test_m1_01_smoke_helpers.py

experiments/results/m1_01_telemetry_contract_summary.json
experiments/results/m1_01_smoke_summary.json
```

## UPDATE / SHARED

```text
requirements-server.txt
```

Only if needed to pin runtime/add `nvidia-ml-py`; Khôi should review because this is server environment ownership.

## KHÔI-OWNED POSSIBLE CHANGE

Server/response integration needed to expose:

```text
data["metrics"]
```

Dương specifies/tests; Khôi implements engine hook.

## DOCUMENTATION

```text
logs/Khoi/M1-01/openspec.md
logs/Khoi/M1-01/proposal.md
logs/Khoi/M1-01/plan.md
logs/Khoi/M1-01/M1-01_worklogs.md
```

---

# 1A. Branch workflow

M1-01 branch đã do Khôi tạo:

```text
feat/m1-01-vllm-dual-lane-server
```

Dương tiếp tục trên **chính branch này**.

Before edit:

```bash
git fetch origin
git switch feat/m1-01-vllm-dual-lane-server
git pull --ff-only origin feat/m1-01-vllm-dual-lane-server
git rev-parse HEAD
```

Không tạo branch M1-01 thứ hai. Không force-push.

---

# 2. Phase 0 — Sync & Baseline

- [ ] Fetch latest `feat/m1-01-vllm-dual-lane-server`.
- [ ] Record `git rev-parse HEAD`.
- [ ] Verify current head or note newer commit.
- [ ] `python -m py_compile scripts/test_lane1.py scripts/test_lane2.py`.
- [ ] `bash -n scripts/launch_lane1_vanilla.sh`.
- [ ] `bash -n scripts/launch_lane2_static.sh`.
- [ ] `bash -n scripts/switch_lane.sh`.
- [ ] `git diff --check`.
- [ ] Record current `requirements-server.txt`.
- [ ] Confirm M1 environment with Khôi:
  - Tesla T4;
  - vLLM **0.29.0 exactly**;
  - exact Torch/CUDA.
- [ ] Record `vllm --version`; fail live acceptance on drift.
- [ ] Confirm served model names remain:
  - `qwen2.5-7b-vanilla`;
  - `qwen2.5-7b-static-draft`.
- [ ] Confirm ports remain 8001/8002.

### Stop

If ports/models/spec config changed materially, update spec before code.

---

# 2A. Execution split — VSCode local first, T4 second

## Local VSCode

Hoàn thành trước cloud run:

- [ ] schema;
- [ ] normalizer;
- [ ] checker;
- [ ] SSE parser;
- [ ] fixture tests;
- [ ] summary generation;
- [ ] docs.

## T4 live acceptance

Linux + Tesla T4 16GB (Kaggle/Colab), ưu tiên cùng environment M1-02.

Chỉ ở đây chạy:

- [ ] vLLM;
- [ ] native acceptance metrics;
- [ ] Vanilla 3+10;
- [ ] Static 3+10;
- [ ] NVML peak/stability;
- [ ] OOM acceptance.

## Sau live run

Quay lại VSCode để chốt compact summaries/worklog.

---

# 3. Phase 1 — Native telemetry contract + normalized schema

Native vLLM 0.29.0 response là nguồn sự thật.

Expected Static native object:

```json
{
  "metrics": {
    "speculative_decoding": {
      "mean_acceptance_length": 2.5,
      "draft_acceptance_rate": 0.5,
      "acceptance_histogram": [1, 1, 1, 1],
      "num_spec_steps": 4,
      "num_accepted_draft_tokens": 6,
      "num_draft_tokens": 12,
      "num_spec_tokens": 3
    }
  }
}
```

Dương tạo normalized schema:

```text
experiments/configs/m1_request_telemetry.schema.json
```

Tasks:

- [ ] Draft 2020-12 schema.
- [ ] Preserve native summary counters.
- [ ] Static `num_spec_tokens == 3`.
- [ ] Static alpha `[0,1]`.
- [ ] Static tau `[1,4]`.
- [ ] alpha == `draft_acceptance_rate`.
- [ ] tau == `mean_acceptance_length`.
- [ ] Vanilla spec/alpha/tau/tree = null.
- [ ] Add `client_metrics` with nonnegative `ttft_ms/tpot_ms/e2e_ms/output_tokens`.
- [ ] Add `gpu_metrics.vram_peak_mb >= 0`.
- [ ] Document tau includes bonus token.
- [ ] Add fixture.

---

# 4. Phase 2 — Contract Checker

Create:

```text
scripts/check_m1_telemetry.py
```

Functions:

```python
extract_native_spec_metrics(final_usage_chunk)
normalize_record(...)
validate_normalized_record(...)
recompute_alpha(...)
recompute_tau(...)
```

Checks:

- [ ] native Static object present;
- [ ] `num_spec_tokens == 3`;
- [ ] histogram length == 4;
- [ ] histogram sum == `num_spec_steps`;
- [ ] accepted tokens recompute from histogram;
- [ ] alpha recompute from counters;
- [ ] tau recompute as `1 + accepted/spec_steps`;
- [ ] values match within float tolerance;
- [ ] Vanilla null normalization;
- [ ] client timing fields nonnegative;
- [ ] VRAM field nonnegative;
- [ ] no NaN/Infinity;
- [ ] nonzero exit on failure.

Tests:

```text
valid vanilla
valid static
missing speculative_decoding
bad alpha/tau
wrong k
bad histogram
counter mismatch
derived alpha mismatch
derived tau mismatch
```

---

# 5. Phase 3 — Enable native vLLM telemetry

Default path là config, không custom instrumentation.

Static launcher:

```text
--per-request-spec-decode-metrics summary
```

Tasks:

- [ ] Verify `vllm --version` == `0.29.0`.
- [ ] Add/review native flag on Static launcher.
- [ ] Start Static on T4.
- [ ] Send `n=1`, `stream=true`, `include_usage=true`.
- [ ] Verify final usage chunk has `metrics.speculative_decoding`.
- [ ] Verify `num_spec_tokens == 3`.
- [ ] Save a real runtime fixture.

### HARD STOP

Nếu pinned vLLM 0.29.0 + native flag vẫn không trả metrics:

```text
STOP
return blocker -> Khôi
```

Chỉ khi đó mới xem custom instrumentation như thay đổi riêng.


# 6. Phase 4 — Build Smoke Client

Create:

```text
scripts/smoke_m1_01_telemetry.py
```

Dependencies:

```text
requests hoặc aiohttp
nvidia-ml-py
```

CLI proposal:

```bash
python scripts/smoke_m1_01_telemetry.py   --lane vanilla   --endpoint http://127.0.0.1:8001/v1/chat/completions   --model qwen2.5-7b-vanilla   --warmup 3   --requests 10   --raw-out experiments/runs/m1_01_smoke/vanilla.jsonl
```

Static analogous at 8002.

Request:

- [ ] same prompt for both lanes;
- [ ] deterministic prompt long enough for speculative steps;
- [ ] `temperature=0`;
- [ ] `seed=42` REQUIRED; fail preflight nếu pinned vLLM 0.29.0 reject;
- [ ] `max_tokens>=64`;
- [ ] `stream=true`;
- [ ] `stream_options.include_usage=true`;
- [ ] concurrency=1;
- [ ] capture response id;
- [ ] capture usage;
- [ ] capture `data["metrics"]`.

Timing:

```text
ttft_ms
itl_ms
e2e_ms
tpot_ms
```

TPOT formula matches M0-03/M1-02:

```text
(E2E - TTFT) / (output_tokens - 1)
```

when output tokens >1.

---

# 7. Phase 5 — NVML / VRAM Stability

Per lane:

1. server health ready;
2. capture cold idle;
3. warm-up 3;
4. cooldown 2–5 sec;
5. capture post-warmup idle baseline;
6. run 10 measured;
7. sample VRAM approx every 50ms;
8. cooldown >=5 sec;
9. capture final idle.

Fields:

```text
vram_idle_cold_mb
vram_idle_after_warmup_mb
vram_peak_measured_mb
vram_idle_after_smoke_mb
idle_delta_mb
```

Gate:

- [ ] 0 OOM.
- [ ] 10/10 success.
- [ ] peak <=14336 MiB.
- [ ] final idle delta <=256 MiB (project stability heuristic).

`<=14336 MiB` is the operational interpretation of the Trello `<=14GB` gate (14 GiB, because NVML reports MiB).

If idle delta >256 MiB:

- [ ] run second 10-request batch;
- [ ] compare baseline/batch1/batch2;
- [ ] monotonic growth -> `VRAM_STABILITY_FAIL`;
- [ ] return to Khôi.

Do not patch allocator in Dương scope.

---

# 8. Phase 6 — Vanilla Live Smoke

Start:

```bash
bash scripts/switch_lane.sh lane1
```

Preflight:

```text
GET /health -> 200
GET /v1/models -> qwen2.5-7b-vanilla
```

Expected normalized result:

```text
10/10 success
speculative_decoding = null
alpha = null
tau = null
tree_budgets = null
```

Raw Vanilla response MAY omit `metrics` entirely.

Checks:

- [ ] normal OpenAI Chat Completions payload;
- [ ] no lane-specific request fields;
- [ ] raw JSONL written;
- [ ] VRAM gate pass;
- [ ] telemetry checker pass.

Then stop lane.

---

# 9. Phase 7 — Static Live Smoke

Start:

```bash
bash scripts/switch_lane.sh lane2
```

Preflight:

```text
GET /health -> 200
GET /v1/models -> qwen2.5-7b-static-draft
```

Expected:

```text
10/10 success
`metrics.speculative_decoding` present
0 <= draft_acceptance_rate <=1
1 <= mean_acceptance_length <=4
num_spec_tokens == 3
server response IDs retained
```

Checks:

- [ ] native speculative summary fields non-null;
- [ ] no fabricated counters;
- [ ] raw JSONL written;
- [ ] peak VRAM gate pass;
- [ ] stability gate pass.

If text generation succeeds but server metrics absent -> Interface A FAIL.

---

# 10. Phase 8 — Compact Evidence

Create:

```text
experiments/results/m1_01_telemetry_contract_summary.json
```

Fields:

```text
schema_version
telemetry_transport
field_ownership
vanilla_semantics
static_semantics
alpha/tau mapping
checker_status
```

Create:

```text
experiments/results/m1_01_smoke_summary.json
```

Fields:

```text
git_commit
gpu
driver
torch
cuda
vllm
served_models
ports
warmup_count
measured_count

vanilla:
  success_count
  ttft/tpot descriptive values
  vram_peak
  idle_delta
  contract pass

static:
  success_count
  ttft/tpot descriptive values
  draft_acceptance_rate descriptive values
  mean_acceptance_length descriptive values
  aggregate native counters
  vram_peak
  idle_delta
  contract pass

handoff_ready
```

No speedup conclusion.

---

# 11. Phase 9 — M1-02 Handoff Check

Re-read current M1-02 branch before handoff.

Current expected consumer path:

```python
record["server_metrics"] = data.get("metrics")
```

Dương checks:

- [ ] final usage event parser retains metrics;
- [ ] `server_request_id` retained;
- [ ] per-request alpha maps from `server_metrics.speculative_decoding.draft_acceptance_rate`;
- [ ] per-request tau maps from `server_metrics.speculative_decoding.mean_acceptance_length`;
- [ ] cell alpha formula uses summed accepted/draft counters;
- [ ] cell tau formula uses `1 + summed accepted / summed spec steps`;
- [ ] Vanilla cell alpha/tau remain null;
- [ ] TPOT remains harness-owned;
- [ ] VRAM remains NVML-owned;
- [ ] raw JSONL retains native counters/histogram;
- [ ] Static telemetry mode remains `summary` for every measured cell.

Dương writes handoff snippet into worklog.

Do not modify M1-02 scheduler in M1-01.

---

# 12. Phase 10 — Documentation / Completion

Update:

```text
logs/Khoi/M1-01/M1-01_worklogs.md
```

Dương section must include:

```text
telemetry schema version
metric definitions
chosen transport
exact branch commit
exact runtime versions
Vanilla result 10/10
Static result 10/10
Static alpha/tau
VRAM peak
VRAM idle delta
known limitations
handoff status Đại
```

Run final checks:

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/check_m1_telemetry.py
python -m py_compile scripts/smoke_m1_01_telemetry.py
bash -n scripts/launch_lane1_vanilla.sh
bash -n scripts/launch_lane2_static.sh
bash -n scripts/switch_lane.sh
git diff --check
```

---

# 13. Dương Completion Gate

Dương marks both Trello items ready only if:

```text
[ ] telemetry schema exists
[ ] telemetry checker tests pass
[ ] native vLLM Static metrics enabled and verified
[ ] Vanilla 10/10
[ ] Static 10/10
[ ] Static native counters + alpha/tau valid
[ ] peak VRAM <=14GiB each lane
[ ] VRAM stability pass
[ ] runtime versions recorded
[ ] compact summary exists
[ ] M1-02 handoff verified
```

Then Dương stops M1-01.

---

# 14. Khôi Follow-up

Khôi remains owner for:

```text
server-side metric instrumentation fixes
runtime requirements pinning
final task worklog integration
PR
```

Nếu Dương smoke phát hiện defect server, task quay lại Khôi, sau đó Dương rerun acceptance.

---

# 15. Proposed Commit Split

## D1

```text
feat(m1-01): define request telemetry contract
```

Includes:

```text
schema
fixture
checker
tests
```

## K/shared

```text
feat(m1-01): enable native speculative request telemetry
```

Shared launcher config change; Khôi reviews launcher.

## D2

```text
test(m1-01): add dual-lane telemetry smoke validation
```

Includes:

```text
smoke script
smoke helper tests
```

## D3

```text
docs(m1-01): record telemetry and smoke handoff evidence
```

Includes compact summaries + worklog.

Do not commit raw `experiments/runs/`.

---

# 16. Test Matrix

## Contract

```text
valid vanilla
valid static
invalid alpha
invalid tau
unknown lane
unknown field
null static alpha
non-null vanilla alpha
request-id mismatch
```

## Smoke helpers

```text
SSE parsing
usage event detection
metrics extraction
TTFT formula
TPOT formula
one-output-token behavior
VRAM delta
summary generation
```

## Live

```text
Vanilla health/models
Vanilla 3+10
Static health/models
Static 3+10
VRAM monitor
M1-02 parser handoff
```

---

# 17. Explicit Stop Line

Dương's work ends at:

```text
Telemetry contract PASS
+
Dual-lane smoke PASS
+
VRAM stability PASS
+
M1-02 handoff READY
```

Dương does **not** continue into:

```text
concurrency sweep
repeat matrix
speedup analysis
summary.csv final aggregation
```

Those are Đại/M1-02 responsibilities.
