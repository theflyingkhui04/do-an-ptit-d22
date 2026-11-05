# M3-05 frozen manual campaign

PM decision lock supplied by Đại in this chat on 2026-11-03 overrides older
Trello/worklog wording. This file records the executable contract; it does not
claim the final campaign has run.

## Protocol

Exactly vanilla/static/adaptive/eagle × W1/W2/W3 × c=1,2,4,8,16,32:
72 cells, 5 repeats × 32 measured requests = 160/cell, 11,520 total.
Exactly 3 warm-ups per cell, separately saved as `warmup.jsonl`, all successful
before measurement. Use `--warmup-policy m3-fixed3`; historical M1/M2 defaults
remain concurrency-matched. The synchronous `bench_cell` is deprecated for
scientific M3 execution: use the deployed-function submission path below.

Every cell gets a new server/container. Stop the previous app, deploy only the
active lane with a new attempt-specific PTIT_RUN_ID, KEEP_WARM=0 and scaledown=60.
Immutable downloaded weights and compilation files may persist; live prefix/KV
state may not cross cells. The server emits a boot UUID and admits one benchmark
claim per run. Raw records bind to that instance; a restart during the cell fails
validation. Startup is operational time, excluded from request latency.

Frozen runtime: vLLM 0.30.0; target Qwen/Qwen2.5-7B-Instruct-AWQ at
b25037543e9394b818fdfca67ab2a00ecc7dd641; static/adaptive draft
Qwen/Qwen2.5-0.5B-Instruct at 7ae557604adf67be50417f59c2c2f167def9a775;
eagle draft duckhoi/qwen2.5-7b-eagle3.1-speculator at
c3adca7e211b471d97a519a625df7ffd37b8b461 (epoch3/step15000).
Eagle: method=eagle3, k=3, greedy, heterogeneous-vocab omitted.
Other lane arguments remain those in `modal_lanes.py`.

No implicit performance threshold. Negative speedup is valid evidence. Preserve
noise, failures and partial attempts. Never rerun a complete valid cell to improve
performance. Incomplete/failed retries use a new attempt ID, fresh server, three
warm-ups and the entire 160-request cell. No appending to partial attempts.

## Source and SDK

Commit executable/config changes before deployment/submission. Source/config is
packaged from exact Git blobs with `cat-file --batch` (no archive/checkout CRLF
filters); manifest hashes are unchanged. Historical run/results are excluded.
The deployed CPU identity handshake and remote bench check the same SHA/run/lane.
Do not redeploy an active attempt. Preserve this checkout/commit for collection.

Installed SDK audited: Modal 1.6.2 `_functions.py` from_name/spawn/from_id/get,
`_object.py` object_id, and `app.py` run. `App.run` starts an ephemeral app and
hydrates its resources before entering; it is unnecessary for deployed-function
submission and can spend time building/hydrating before spawn. This does not
prove which operation stalled the historical interrupted submission.

Use `modal.Function.from_name("ptit-vllm", "bench").spawn(...)`.
Persist/print FunctionCall.object_id, exit, then recover via
`modal.FunctionCall.from_id(call_id).get()`. No submit process stays alive.
Official cross-checks: [job processing](https://modal.com/docs/guide/job-queue),
[Function](https://modal.com/docs/sdk/py/latest/Function),
[FunctionCall](https://modal.com/docs/sdk/py/latest/FunctionCall).

## Receipt recovery

`call_id=null` means NOT_SUBMITTED, never RUNNING. Retain the old directory and
use a new attempt identity after verifying remote state. An additional
`spawn_attempt.json` marker means the spawn RPC may have reached Modal: do not
blindly retry even if the receipt has no ID. Recover the ID from submit stdout or
Modal call history, wait for completion, then use `bench_recover` (validates the
completed return's identity before attaching it). No automatic resubmit in this
ambiguous state. Receipts are atomically replaced; orphan temporary files remain
audit evidence and do not block a later receipt update.

Collection never spawns/cancels a call and refuses conflicting local evidence.
Identical repeated collection is idempotent. Collect promptly: Modal FunctionCall
outputs have limited retention. Full return bundles also persist in the artifacts
Volume at `bench/<attempt>/result.json` for explicit archival recovery.

## Manual full campaign — Đại only

The following commands must not be executed by Codex during pre-flight.
Use the reviewed pre-flight SHA, not an unreviewed later HEAD. A campaign ID is
reserved for this confirmatory study; pilot directories are not in its grid.

```powershell
$sha = '<REVIEWED_PREFLIGHT_SHA>'
$campaign = 'm305-confirmatory-v1'
python -m pytest tests/test_m3_preflight.py tests/test_modal_bench_jobs.py tests/test_modal_runner_config.py tests/test_m2_harness_contract.py tests/test_benchmark_profiles.py tests/test_benchmark_workload_adapter.py tests/test_modal_vram_join.py tests/test_aggregate_results.py -q
python -m experiments.run_m3_campaign plan --campaign-id $campaign --execution-sha $sha
# This is the full scientific campaign launch; Đại runs it manually.
python -m experiments.run_m3_campaign run --campaign-id $campaign --execution-sha $sha
```

`run` sequentially deploys/submits/polls/collects/validates/stops one cell at a
time, moving through the frozen lane order. It prints FunctionCall IDs. If the
laptop disconnects, the submitted remote cell continues. Rerun the identical
`run` command to collect its completion and resume. Valid collected cells skip;
pending calls are polled, never resubmitted. A network observation failure exits
without stopping/cancelling the live job. A known terminal failed cell stops;
inspect its logs/Volume artifacts before explicitly authorizing a new attempt:

```powershell
python -m experiments.run_m3_campaign status --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m3_campaign run --campaign-id $campaign --execution-sha $sha --retry-failed
```

For an individual ID (fresh Python process):

```powershell
python -m experiments.modal_runner bench_collect --call-id fc-REPLACE
python -m experiments.validate_m3 --cell-dir experiments/runs/m3_01/REPLACE_ATTEMPT
# Only for a missing receipt ID with an identified completed remote call:
python -m experiments.modal_runner bench_recover --bench-run-id REPLACE_ATTEMPT --call-id fc-REPLACE
```

After all cells are complete, Đại validates, aggregates and produces figures:

```powershell
python -m experiments.run_m3_campaign validate --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m3_campaign aggregate --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m3_campaign figures --campaign-id $campaign --execution-sha $sha
python -m experiments.run_m3_campaign stop --campaign-id $campaign --execution-sha $sha
python -m modal container list --json
```

Aggregation uses existing request→repeat→median formulas and summed native
speculative counters. Speedup pairs the validated same-source campaign/workload/
concurrency against Vanilla. Original attempt IDs remain in raw and the selected
attempt index; no raw records are rewritten. Summary includes every attempt ID
for audit and exactly one valid complete attempt per scientific cell. F1 plots
TPOT and output throughput; F2 retains below-parity observations. Engineering
gate remains NOT_DEFINED. Never aggregate the pilot into these results.
