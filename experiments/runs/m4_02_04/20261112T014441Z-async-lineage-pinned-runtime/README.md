# Async lineage pinned-runtime integration — VERIFIED

Integration validation only. No profitability, latency, throughput or TreeAdaptive
claim. Historical L4 scalar observer evidence was neither changed nor reused.

## Source and environment

Reviewed baseline: `f889865f82bb91681d436e80570815490b179da5` on
`feat/m4-02-04-async-lineage-v02`; merge base and origin/develop were both
`53b813a0b6ce8fef530c4286fbd8bf4da5b4a149`. Tracked baseline was clean.
Three pre-existing untracked evidence directories remain untouched.

The actual successful runtime belongs exclusively to
`fb4544166c428e73b164756a2ef0e11ad6b97d6f`. Changes since the reviewed baseline
are opt-in UUID/PID startup diagnostics, before-update alpha telemetry, CPU tests,
and a harness reusing the canonical Modal image and `_start_vllm` / `_stop_vllm`.
No dependency pins, upstream package sources, policy, models or workload matrix
changed. Later validator/evidence/documentation commits do not relabel the runtime
as having executed their SHA.

Provider: Modal, hostname `modal`, NVIDIA A10 (documented A10G fallback),
23028 MiB, UUID `GPU-c32ea946-0a2b-0c71-9a82-5dc7edd2871f`.
Driver 610.57.04; CUDA 13.0; Python 3.11.5; PyTorch 2.13.0+cu130;
vLLM 0.30.0. All 15 installed source hashes match the unchanged pinned manifest,
including EngineCore, UniProcExecutor, scheduler and AsyncScheduler.

Target: Qwen/Qwen2.5-7B-Instruct-AWQ @
`b25037543e9394b818fdfca67ab2a00ecc7dd641`.
Draft: Qwen/Qwen2.5-0.5B-Instruct @
`7ae557604adf67be50417f59c2c2f167def9a775`.

`runtime-04/launch.json` records the command and all effective ADAPTIVE_* values.
`server_env.json` records the actual environment. `vllm.log` records resolved V1,
world_size/DP/PP/TP=1, BF16, internal auto_awq with MarlinLinearKernel, and the
canonical dynamic scalar K configuration. The normal pinned runtime automatically
selects V1 GPUModelRunner for draft_model; no V2 override was introduced.
The named profile remains `l4-bf16` on the documented A10G fallback.

## Mandatory gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Pinned source gate | PASS | runtime-04/source-gate.json, all 15 hashes |
| Server startup | PASS | runtime-04/readiness.json and vllm.log |
| Child hook bootstrap | PASS | runtime-04/startup.jsonl, child PID 32 |
| EngineCore wrapper | PASS | PID 32, before attempt/publication |
| Executor publication | PASS | LineageUniProcExecutor, exact attempt |
| Scheduler consumption | PASS | AdaptiveAsyncScheduler, same active attempt |
| Startup-attempt ownership | PASS | same attempt/generation/ledger UUIDs; consumed |
| Constructor cleanup | PASS | context reset, consumed registration retained |
| Real speculative feedback | PASS | 13 attributed events, 49 drafted / 49 accepted |
| Exact plan/schedule | MATCH | source plan 1..13 / schedule 2..14 |
| Controller update | PASS | alpha 0.8 -> 0.98900488372224, EMA checked per event |
| Offline validator | PASS | runtime-04/validation.json |

## Process and exact lineage

API server PID 10; EngineCore child PID 32, parent PID 10. The child independently
logs hook installation before entering the wrapper. At wrapper entry, publication
and consumption, the process reports `start_method=spawn`. During the earlier
child module-import phase, the hook-install observation reports `fork`; preserve
that observation rather than silently normalizing it. The actual constructor
phase is spawn, and the custom executor module is imported in the child before
the wrapper. This validates only this concrete process topology, not general
distributed/multiprocess ownership.

Attempt `999c9934abe7414584845896aadd7385`, generation
`55b13c4336134c20927d83873b96bc7d`, ledger
`90617d84beb44dfcbd668bf6559ca5a1` are consistent from publication through
consumption and successful cleanup. These are diagnostic UUIDs, not raw addresses.

Response `chatcmpl-863a80d9a3f74fb9`; internal request
`chatcmpl-863a80d9a3f74fb9-b0204581`. One greedy request, concurrency 1,
64 output tokens. The first feedback references plan 1 / schedule 2 / K=3:
drafted=3, accepted=3, alpha 0.8 -> 0.8400000000000001. The last references
plan 13 / schedule 14 / K=4: drafted=4, accepted=4,
alpha 0.9862561046528 -> 0.98900488372224. All 13 feedbacks join exact source
records and match response totals and final aggregate state. No missing-lineage
event occurs. Runtime semantics remain linear_dynamic_k, no selected topology,
and tree_execution_verified=false. Conservative production fields named
executed_runtime_mode remain `unverified`; they were not rewritten to manufacture
runtime evidence. The separate validator establishes the scoped lineage result.

## Offline reproduction

From repository root:

```powershell
python -B -m experiments.validate_async_lineage_runtime experiments/runs/m4_02_04/20261112T014441Z-async-lineage-pinned-runtime/runtime-04 --expected-sha fb4544166c428e73b164756a2ef0e11ad6b97d6f
```

Result: all seven validator gates PASS, exit 0. It checks SHA and installed-source
gate, hook/wrapper order in the same actual child, exact startup ownership,
registration retention, source plan/schedule/K, counts, every EMA update, response
totals, aggregate controller state, no skipped lineage, and scope limits.
CPU tests: 143 before instrumentation; 180 after instrumentation; 16 final
validator tests including adversarial evidence mutations. Synthetic fixtures are
tests only and are not placed in the runtime evidence directory.

## Failed attempts and collection repair

1. CPU preflight 01: Windows charmap could not encode Modal's checkmark. Preserved
   `modal-preflight.log`; retry used UTF-8 and passed (`modal-preflight-02.log`).
2. Integration 01 / aeada583: client automount tried to upload its changing log.
   No function/model execution. Preserved `modal-integration-01.log`.
3. Integration 02 / c7eda977: Modal rejects image.env after local source mounts.
   No function/model execution. Preserved `modal-integration-02.log`. Moved import
   path to function env; CPU preflight 03 at fb454416 passed with automount off.
4. Integration 03 / fb454416 / L4: backlog=1, containers=0, running_inputs=0;
   stopped before A10G fallback. Allocation snapshot and full log retained.
   Exact provider cause unknown; this is allocation/infrastructure evidence, not
   a feature defect. First stop CLI aborted at its prompt; --yes succeeded.
5. Integration 04 / fb454416 / A10G: successful canonical runtime. Offline-HF
   safetensors discovery errors were followed by successful loading of the exact
   cached revision; no download/config change was needed. Shutdown logged one
   multiprocessing resource-tracker semaphore warning; all task apps are stopped
   with zero containers (`apps-final.json`).
6. Initial offline validation failed because the harness collected `adaptive/`
   but canonical aggregate telemetry lives in `lane3/`. The unchanged generated
   file was downloaded from ptit-run-artifacts at
   `lane3/20261112T014441Z-async-lineage-pinned-runtime-fb454416-a10g/telemetry.jsonl`
   into runtime-04. No GPU rerun or synthetic replacement. Initial failed result
   retained as `validation-initial-collection-gap.json`; final validator PASS.

Other local operational checks: Git ownership required command-scoped safe.directory;
linked-worktree Git metadata and network access required sandbox escalation;
local installed-only source check failed because vLLM is absent; WSL enumeration
was sandbox-denied; `modal token list` was an unsupported CLI command. None is a
real runtime attempt. No credentials were printed. Failed attempts were retained.

## Optional checks and claim ceiling

REAL INVALID-PAIR CHECK NOT RUN: actual EngineCore initializes/loads the executor
before constructing the scheduler, so another real plain-executor bootstrap would
require a second model load. CPU/source-contract evidence retained.

REAL FAILED-STARTUP RECOVERY: UNVERIFIED AT RUNTIME; CPU/source-contract verified.
No safe existing production injection was found; no OOM/failure was invented.
K=0 runtime sanity NOT RUN: no free per-request runtime configuration switch.

Async feedback lineage, startup pairing, startup-attempt ownership and consumed
generation binding are verified on this tested pinned vLLM 0.30.0 V1 / UniProc /
single-worker NVIDIA A10 configuration. Exclusive reuse rejection and
generation-specific revocation retain their CPU/source-contract evidence; no
additional real failed-startup recovery/revocation claim is made.

Other GPUs/models, arbitrary vLLM versions, distributed/general multiprocess
ownership, deployment compatibility beyond this ephemeral canonical launcher,
TreeAdaptive runtime, profitability and hardware-derived policy remain unverified.

No push, merge, PR, Trello change, develop modification or preservation-ref change.
