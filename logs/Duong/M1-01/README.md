# M1-01 Worklog & Handoff - Duong Telemetry Track

## 2026-09-27 - Local Contract/Tooling Ready

### Scope completed by Duong

- Added the normalized Interface A telemetry schema:
  - `experiments/configs/m1_request_telemetry.schema.json`
  - `experiments/configs/m1_request_telemetry.example.json`
- Added telemetry validator and normalization helpers:
  - `scripts/check_m1_telemetry.py`
- Added live smoke client for T4/vLLM:
  - `scripts/smoke_m1_01_telemetry.py`
- Added unit tests:
  - `tests/test_m1_telemetry_contract.py`
  - `tests/test_m1_01_smoke_helpers.py`
- Enabled native vLLM per-request speculative metrics on the Static launcher:
  - `scripts/launch_lane2_static.sh`
  - added `--per-request-spec-decode-metrics summary`
- Added tracked local evidence:
  - `experiments/results/m1_01_telemetry_contract_summary.json`
  - `experiments/results/m1_01_smoke_summary.json`

### Contract identity

- Schema version: `m1_request_telemetry.v1`
- Telemetry transport: `vllm_stream_usage_metrics.summary`
- Static alpha source: `metrics.speculative_decoding.draft_acceptance_rate`
- Static tau source: `metrics.speculative_decoding.mean_acceptance_length`
- Static draft budget: `num_spec_tokens = 3`
- Static tree budgets: `null` in native summary mode
- Vanilla speculative fields: `null`
- VRAM peak gate: `<= 14336 MiB`
- VRAM idle delta gate: `<= 256 MiB`

### Local checks run

```powershell
python scripts\check_m1_telemetry.py experiments\configs\m1_request_telemetry.example.json
python -m unittest tests.test_m1_telemetry_contract tests.test_m1_01_smoke_helpers
python -m compileall scripts tests
```

Result: all local checks passed.

### Live T4 smoke status

Live T4 smoke was not run in this local workspace because no live vLLM T4 endpoints are available here. The tracked summary is intentionally marked:

```text
PENDING_T4_LIVE_RUN
```

Do not mark Dương's second Trello item complete until the live smoke commands below pass on the target runtime.

### Commands for Khôi/Dương on T4

Start Vanilla:

```bash
bash scripts/switch_lane.sh lane1
```

Run Vanilla smoke:

```bash
python scripts/smoke_m1_01_telemetry.py \
  --lane vanilla \
  --endpoint http://127.0.0.1:8001/v1/chat/completions \
  --model qwen2.5-7b-vanilla \
  --requests 10 \
  --warmup 3 \
  --raw-out experiments/runs/m1_01_smoke/vanilla.jsonl \
  --summary experiments/results/m1_01_smoke_summary.json
```

Start Static:

```bash
bash scripts/switch_lane.sh lane2
```

Run Static smoke:

```bash
python scripts/smoke_m1_01_telemetry.py \
  --lane static \
  --endpoint http://127.0.0.1:8002/v1/chat/completions \
  --model qwen2.5-7b-static-draft \
  --requests 10 \
  --warmup 3 \
  --raw-out experiments/runs/m1_01_smoke/static.jsonl \
  --summary experiments/results/m1_01_smoke_summary.json
```

Validate generated records:

```bash
python scripts/check_m1_telemetry.py experiments/runs/m1_01_smoke/vanilla.jsonl experiments/runs/m1_01_smoke/static.jsonl
```

### Handoff for Dai / M1-02

The M1-02 harness should consume normalized telemetry records using this mapping:

- `alpha = record.alpha`
- `tau = record.tau`
- `acceptance_rate = record.alpha`
- `mean_accept_len = record.tau`
- `tree_budgets = null` for Static summary mode
- `tpot_ms = record.client_metrics.tpot_ms`
- `vram_peak_mb = record.gpu_metrics.vram_peak_mb`

Cell-level Static alpha should be recomputed from counters when aggregating:

```text
alpha_cell = sum(num_accepted_draft_tokens) / sum(num_draft_tokens)
```

Vanilla cells must keep `alpha = null` and `tau = null`.

### Stop conditions

- If Static final usage chunks do not include `metrics.speculative_decoding`, return to Khôi: launcher/runtime telemetry is not ready.
- If `num_spec_tokens != 3`, stop and fix Static lane identity before benchmarking.
- If VRAM peak exceeds `14336 MiB` or idle delta exceeds `256 MiB`, return to Khôi for runtime/server tuning.
- If M1-02 needs a different alpha/tau definition, stop and update the interface contract before running sweeps.
