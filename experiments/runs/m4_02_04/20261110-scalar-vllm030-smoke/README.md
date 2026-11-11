# M4-02-04 real vLLM 0.30.0 scalar integration smoke

Result: **L3 SCALAR PATH VERIFIED**, limited to this three-request smoke.
Run date: 2026-11-10, Asia/Bangkok (worker timestamps are UTC).
This is L3 functional evidence, not a benchmark or task-level DONE decision.

## Identity and environment

The checkout was clean before evidence creation. Branch:
`feat/m4-02-04-scheduler-telemetry-v02`; HEAD:
`a6cd6ad5cee5374738f890c270ada4ca01f0946b`.
Preceding commits: `94edea218aa2b48b4035dea4a87b7a7a4d6ceee6`,
`c52cd0ca2a4044fb78376e9100f3d718c38302f0`.
Git's sandbox ownership check required per-command `-c safe.directory=...`;
global Git configuration was not changed.

Local Windows/PyTorch 2.6.0+cu124/no-vLLM environment failed the gate and was
not used to serve. Modal profile: `nguyenduonghb940`. The existing L4 versions
function remained queued with backlog 1 and zero runners. Only that call received
cancellation requests (`l4_probe_cleanup.json`); final metadata still showed one pending input and zero runners. The waiting local caller was stopped; remote cancellation is UNCONFIRMED. The deployed app was not stopped or changed.

The isolated supported A10G functional fallback allocated an NVIDIA A10.
Worker: Ubuntu 22.04.5, driver 610.57.04, CUDA UMD 13.3, PyTorch CUDA 13.0,
Python 3.11.5, PyTorch 2.13.0+cu130, **vLLM 0.30.0**. See
`attempt2/environment.json` for raw uname, os-release, pip and nvidia-smi.
The project image reused its existing pinned package layer; no runtime version
upgrade or dependency change was made.

`experiments.bench_source.snapshot` packages Git blobs from the exact HEAD,
independently of Windows checkout line endings. The harness mounts committed
`src` at `/root/project/src` and committed `experiments` at `/root/experiments`
to make the scheduler's committed observer imports available. This is a new
isolated app, not a redeployment of the existing authenticated endpoints.
All 63 worker Python source files hash-match the committed blobs. The harness
does not patch scheduler/controller/runtime methods.

## Deployment and commands

Canonical configuration: `experiments.modal_lanes.build_vllm_args('adaptive',
'l4-bf16')`, port 8000. Exact argv and launch environment are saved in
`attempt2/deployment.json`.

- Target: `Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e9394b818fdfca67ab2a00ecc7dd641`.
- Draft: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- dtype `bfloat16`, requested quantization `awq_marlin` (effective engine log
  reports `auto_awq`), `draft_model`, greedy draft sampling, heterogeneous vocab.
- Speculative ceiling 4, dynamic lookup ranges `[[1, 32, 4]]`.
- Scheduler `src.server.adaptive_scheduler.AdaptiveAsyncScheduler`.
- Model length 4096, max sequences 32, GPU memory utilization 0.85,
  generation config `vllm`, per-request speculative metrics `summary`.
- `ADAPTIVE_ENABLE_VRAM_GUARD=1`, `ADAPTIVE_MAX_K=4`,
  `ADAPTIVE_MIN_BATCH_K=0`, `ADAPTIVE_PER_REQUEST=1`,
  `ADAPTIVE_CONTROLLER_KWARGS=''`.
- `ADAPTIVE_TELEMETRY_PATH=/tmp/m40204-smoke/telemetry.jsonl`;
  `ADAPTIVE_STEP_TELEMETRY_PATH=/tmp/m40204-smoke/steps.jsonl`.
- `HF_HUB_OFFLINE=1`; cached pinned weights used. Default scalar
  `AdaptiveDraftController` identity was checked; tree controller not activated.

Executed from repository root in PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PTIT_GPU='A10G'
$env:PTIT_DTYPE_PROFILE='l4-bf16'
$env:PTIT_LANES='adaptive'
python -m modal run experiments/runs/m4_02_04/20261110-scalar-vllm030-smoke/smoke_runner.py::main
python experiments/runs/m4_02_04/20261110-scalar-vllm030-smoke/validate_evidence.py
```

Each invocation is bounded, uses no retries, and terminates its own server
process group after collection. The isolated Modal app completed. No existing
server process was killed. Run/call IDs are in each `call.json`.

## Observations

Phase A PASS: real package import/subclass and actual hook signatures verified;
model/draft initialized; custom scheduler selected; health returned HTTP 200.
Installed AsyncScheduler/Scheduler source snapshots were collected from the
worker, not upstream/latest. The installed Scheduler builds `dynamic_sd_lookup`,
selects `num_spec_tokens_to_schedule` from it and passes it into SchedulerOutput
before invoking `_update_after_schedule`.

Phase B PASS: a single deterministic request (`temperature=0`, `seed=42`,
`max_tokens=48`) returned HTTP 200 and nonempty text; server remained alive.
The count prompt reached the token limit; finishing the full count was not the
smoke acceptance criterion.

Phase C PASS: 36 planned records match their actual scheduled-budget records;
39 scheduled records include initial prefill K=0. Scheduled values observed:
0, 2, 3, 4. No thresholds, controller kwargs or feedback were forced.

For the single request, natural high-acceptance planning at K=3 changed to
aggressive K=4. Measured callback totals: 37 drafted, 37 accepted over 10
feedback events. Across the three requests: 83 drafted, 72 accepted over 25
events. These are functional counters, not performance results.

The chain established is committed planner decision -> real SchedulerOutput K
observed in the real hook -> successful GPU inference -> real verification
callback counts. An individual feedback event cannot be attributed to a specific
earlier K/plan/schedule because source identifiers are absent.

Phase D PASS for minimal smoke: two concurrent clients completed successfully,
with distinct request state and natural budget/acceptance histories. Client
concurrency was 2; independently observed scheduler running/decode batch counts
were 1 and 2 (`concurrency_batch_b`), not a worker tensor-shape measurement.
At schedule 21, one request had shared scheduled K=2 and placeholder budget=0,
demonstrating the per-request trim path. All three completion aggregates were
emitted. In the exact committed source, those records are written after
`adapter.end_request` deletes session/history and planner dictionaries are
cleared. No records for a request occur after its completion aggregate. This
is cleanup-hook evidence; live heap state was not separately inspected.

## Telemetry semantics and retained boundaries

- Planned: topology field retained as null for this scalar controller; planned
  draft nodes, batch nodes and scalar budget recorded separately.
- Scheduled: `linear_dynamic_k`, scalar K, placeholder budget, plan/schedule
  sequence; 36 non-null plan references verified.
- Measured: verification drafted/accepted counters, request completion.
- Unverified: per-feedback `source_plan_sequence` and
  `source_schedule_sequence` are null; `executed_runtime_mode=unverified`.
- Every record retains `tree_execution_verified=false`. Aggregate executed draft
  count is null; step feedback is the measured counter source.
- Engine/hardware metadata fields in planner telemetry are null; exact runtime
  identity is supplied by the separate worker environment artifact.

`ASYNC_FEEDBACK_SOURCE_LINEAGE = UNVERIFIED`.
**TEAM TECHNICAL ALIGNMENT REQUIRED**. Never infer feedback lineage from latest
plan. Tree-controller calibration remains **TECHNICAL REVIEW REQUIRED**.
Dynamic scalar K does not establish arbitrary-tree execution. No T8/T16/T32,
arbitrary masks/parents/positions/accepted-path reconstruction or speedup claim
is supported. No frozen benchmark protocol was changed.

## Failures and anomalies retained

1. Local environment mismatch; no local engine attempt.
2. Sandbox Modal connection stalled; elevated network access succeeded.
3. L4 probe queued without a runner; cancellation requested twice after A10G fallback, including terminate_containers=True. Final API counters still show pending=1, cancelled=0, runners=0. No active GPU containers remain; cancellation confirmation is an unresolved Modal control-plane anomaly.
4. First Modal CLI invocation failed on Windows console U+2713 encoding before
   execution; fixed only the caller's `PYTHONIOENCODING`.
5. First worker attempt imported successfully but stopped on an erroneous harness
   assertion expecting `BudgetController` instead of committed default
   `AdaptiveDraftController`. Root `error.log`/`result.json` preserve this
   HARNESS BUG; no project bug or tree-configuration drift was demonstrated.
6. Windows Modal live-log diagnostics failed with CLI argument/stream/PTY and SDK
   interface errors. They did not alter server behavior; final raw logs came from
   the harness return value.
7. vLLM logged two offline safetensors discovery ERRORs for the pinned draft,
   then loaded cached weights and served all requests. Recovered discovery
   anomaly, not a fatal model/configuration failure.
8. Expected automatic V1 fallback for draft_model, PIECEWISE CUDA-graph override,
   and custom-interface stability warnings are in server.log.
9. During deliberate teardown only: engine process-manager force-kill warning and
   one leaked-semaphore resource_tracker warning. Their cause is UNKNOWN; no
   scheduler/request exception appeared during serving. This does not establish
   clean long-running service shutdown behavior.

No pinned-vLLM API mismatch or scalar project bug was demonstrated. No source
compatibility fix, commit, push, PR, merge, Trello update or benchmark performed.
Evidence and worklog remain uncommitted for review; task is not marked DONE.

Next technical action: team technical alignment on exact async feedback source
lineage identifiers and their callback contract.

## Raw artifact index

`manifest.json` lists every generated evidence path and SHA-256 (except itself).
Root artifacts preserve the first failed worker attempt; `attempt2/` contains
the successful real-engine run. `validation.json` summarizes checks over the
saved raw evidence, not simulated engine output.
