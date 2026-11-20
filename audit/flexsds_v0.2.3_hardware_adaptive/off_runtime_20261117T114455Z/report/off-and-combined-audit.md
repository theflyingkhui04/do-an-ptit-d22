# OUTCOME

**OFF RUNTIME AUDIT FAILED.** Exactly one OFF development launch was attempted. The endpoint did not pass the existing 900-second readiness window. Modal's system log reported waiting for GPU_L4 worker capacity; allocation polls showed no serving container. No saved request was sent, no second OFF attempt occurred, and ON was not rerun. The failure is infrastructure/readiness evidence, not evidence that static speculation or the adaptive algorithm produced incorrect tokens.

## SOURCE

- Branch develop; HEAD 8006205a6f19790a776d67155a377def053d2736.
- Checkout D:/Document/Four-year/do_an/do-an-ptit-d22-flexsds-audit; local Python .venv/Scripts/python.exe, version 3.11.0; FlexSDS 0.2.3 imported from checkout src/flexsds/__init__.py.
- Modal CLI 1.6.4, native profile/authentication verified.
- Preserved ON serving versions: vLLM 0.30.0, PyTorch 2.13.0+cu130, CUDA 13.0. The unchanged image definition still pins vLLM 0.30.0 and uses CUDA 13.0.3 development image/Python 3.11.
- **Actual OFF serving vLLM/PyTorch/CUDA/Python versions: UNKNOWN / NOT OBSERVED.** No serving container was available for inspection. Do not treat intended image versions as observed runtime values.
- All pre-launch source hashes match the ON source identity. The complete ON artifact hash manifest was checked before launch and after cleanup and remains unchanged. No dependency/source/configuration restoration or modification was performed.

## GPU SESSION

- OFF App ID ap-l4FtcaDTylSEmfiSoEY0Eg; image im-7l3yRxsEFNxpEz3ZvaXTth; function fu-szlDiahq8zJHIDgH6QEQje.
- Requested GPU L4. Actual GPU identity/VRAM: UNKNOWN / NOT OBSERVED. No GPU serving container was observed; image-build task IDs in app logs are not claimed as GPU-serving identities.
- Endpoint https://daibui1208--flexsds-serving-serve-dev.us-east.modal.run was registered but never passed readiness. CLI exited 1 after 900 seconds with `Endpoint readiness timed out: The read operation timed out`. TLS handshake timeouts also occurred during the same readiness polling loop.
- Modal logged: Function serve was waiting to be scheduled on a GPU_L4 worker and more capacity was being acquired. This establishes a capacity wait, not an observed model/import failure; the exact eventual allocation outcome beyond the bounded attempt is unknown because the app was stopped.
- App stopped at 2026-11-17 19:03:22 Asia/Bangkok. Verified state=stopped, tasks=0, audit containers=[]. No matching local development watcher remained. No unrelated app was stopped.

## TREATMENT VERIFICATION

The actual pre-Popen launch configuration was checked against the preserved frozen OFF configuration before the one Modal serve process was allowed to start. It matched exactly: the same target/draft IDs and pinned revisions, eagle3, L4, max_num_seqs=8, max_model_len=2048, enable_adaptive=false and profile_path=null. The target revision was b25037543e9394b818fdfca67ab2a00ecc7dd641; draft revision d9016c3860f42cc3f33a87fa5d346fee768f6290.

The unchanged builder produces static speculative maximum K=4, greedy draft sampling, bfloat16/AWQ-Marlin, summary speculative metrics, and no custom Adaptive scheduler/executor or dynamic batch-K lookup. OFF is configured as static EAGLE, not target-only serving. The image's encoded configuration also contains adaptive=false and profile=null; no profile artifact was uploaded.

**Actual OFF scheduler/executor behavior and absence/bypass of Adaptive decision hooks are UNKNOWN / NOT OBSERVED.** Configuration and uploaded source are not execution proof. No serving startup, counters, controller records or request results were captured, so treatment separation cannot be closed beyond the preserved ON runtime and OFF launch configuration.

The known builder differences beyond adaptive control remain: ON forces eager execution/compilation mode 0 and explicit async/custom executor settings; OFF leaves corresponding defaults unresolved. No setting was changed to obtain a control result. Even a later successful comparison must disclose actual execution modes before causal claims.

## CORRECTNESS COMPARISON

The two ON requests were loaded read-only in their original client submission order (sequence, rainbow). Their exact payloads, temperature=0, seed=42, max_tokens=64, return_token_ids=true, stop/default fields and client concurrency=2 were retained in requests/replay-plan.json. Canonical payload hashes are:

- sequence: ddc5ee6c901821110f21afedf23b67178bd8ffc1fd2cc591bd543e8ed491b934.
- rainbow: b6427e4b09f9ba7cef18e6ac8a2c70b0e56af1c01623e39c4035cdffb0fda366.

Replay was gated on serving readiness/runtime identity. That gate never passed; the prepared replay/snapshot scripts were not executed. There is no protocol substitution or new prompt.

- Planned pairs: 2.
- Comparable pairs: 0.
- TOKEN-EQUAL: 0.
- TOKEN-DIVERGENT: 0.
- NOT COMPARABLE: 0.
- MISSING EVIDENCE: 2 (both lack OFF token outputs).

ON retained 57 sequence token IDs (finish_reason=stop) and 64 rainbow token IDs (finish_reason=length); OFF token IDs/counts/finish reasons are missing. No decoded-text comparison was substituted. A missing OFF output is not classified as divergence. Request identity, ON token hashes and conditions are preserved in analysis/paired-comparison.json.

**ADAPTIVE ON VS STATIC OFF: INSUFFICIENT EVIDENCE.**

**LOSSLESS VS TARGET-ONLY: INSUFFICIENT EVIDENCE.** Existing matching audit exchanges were inspected; none supplies a valid target-only treatment for these requests. Static EAGLE would not establish target-only equivalence even if its outputs had matched. No stochastic distribution-preservation claim is supported.

## RUNTIME EVIDENCE

OFF drafted/accepted/speculative-step counters: UNKNOWN / NOT OBSERVED, not zero. There is no native OFF metric payload, so its internal counter consistency cannot be evaluated. The preserved ON result remains 26 drafts, 3 accepts and 10 speculative steps; it is not reused as an OFF result or performance comparison.

Important OFF evidence, relative to this session root:

- environment/identity.json: authoritative import, SHA/status/version/source hashes and pre-launch checks.
- startup/frozen-config.json; startup/actual-launch.json: preserved OFF builder and checked actual launch settings/session count.
- startup/cli-launch-01.txt; startup/cli-exit.json: real readiness timeout and exit 1.
- modal_logs/app-logs-before-stop.stdout: timestamped image creation and GPU_L4 capacity wait.
- modal_logs/local-modal-final.log: raw local Modal output.
- modal_logs/app-stop.stdout; modal_logs/containers-after-stop.stdout; modal_logs/apps-after-stop.stdout; analysis/cleanup.json: successful stop and empty container verification.
- requests/replay-plan.json: exact saved ON requests, order, hashes and source paths; there are no OFF response files.
- analysis/paired-comparison.json; analysis/runtime-summary.json; analysis/combined-requirements.json: derived comparison and audit decisions.
- analysis/artifact-sha256.json: evidence hashes.

Preserved ON evidence remains in ../on_runtime_20261117T111844Z, especially report/on-runtime-audit.md, analysis/joined-lineage.csv, telemetry/*.jsonl and requests/*-response-body.json. These files were neither rewritten nor normalized.

## COMBINED 8-REQUIREMENT AUDIT TABLE

| Requirement | Evidence | Verdict | Remaining gap |
|---|---|---|---|
| 1. CLI → runner → scheduler → planner → controller | SOURCE/UNIT: traced construction and focused tests. GPU-ON: actual scheduler, executor, controller and downstream proposal/verification records. OFF: frozen launch matched, readiness failed. | Đạt một phần | OFF runtime/bypass remains unobserved. |
| 2. Verification feedback / EMA / warmup | GPU-ON: 10 attributed updates and native summary agreement; independent checking of 119 decision states. Source/unit logic preserved. | Đạt một phần | Ten-update warmup completion and stale/preempted feedback cases unobserved; OFF unavailable. |
| 3. Dynamic K propagation | GPU-ON: K=3/0/2; 10 producer/scheduler/birth/verification joins agree. | Đạt một phần | Heterogeneous-batch trimming and static OFF execution not established. |
| 4. Memory guard | GPU-ON: actual VRAM/KV readings. Source/unit tests cover thresholds/cooldown. | Đạt một phần | No GPU pressure trigger, cooldown or recovery observed. |
| 5. Fallback / probing / recovery | GPU-ON: fallback and four K=2 probes with verifier feedback. | Đạt một phần | Recovery not observed. |
| 6. Hardware profile consumption | GPU-ON identifies the exact active threshold controller; matching deployed source does not query measured costs. Profile-loader path/identity established in ON. | Chưa đạt | Measured latency values do not drive active K selection. |
| 7. Batch / request behavior | GPU-ON: request initialization and cleanup; client concurrency 2, effective batch 1. | Đạt một phần | Multi-row batching, mixed K, ID reuse and concurrent-row isolation unobserved. |
| 8. Telemetry lineage | GPU-ON: 10 stable producer joins, native-counter agreement and cleanup records. | Đạt một phần | Consuming schedule/incarnation fields incomplete; no OFF serving evidence. |

These verdicts preserve the distinction between inspected source, reused unit/integration results, ON GPU observations, and missing OFF GPU observations. None is upgraded merely from uploaded code or configuration.

## UNRESOLVED FINDINGS

- **PROFILE VALUES NOT CONSUMED BY ACTIVE DECISION PATH** remains unchanged. OFF supplied no direct contrary evidence.
- ON warmup completion, memory-trigger/cooldown/recovery, fallback recovery, K=4, stale/preempted feedback and request-ID reuse were not observed.
- Client concurrency 2 did not establish a multi-row runtime batch: ON observed batch=1, OFF no batch observations. Heterogeneous K/row trimming and simultaneous request-state isolation remain unverified.
- The ON trace joins producing plan/schedule identifiers; consuming verifier schedule IDs and uniform incarnation fields are incomplete. Native per-step API arrays are null; project telemetry supplied ON step counts. No OFF trace exists to assess instrumentation there.
- No comparable ON/OFF token pair and no matching target-only control exist. Greedy correctness and stochastic preservation remain unproven.
- OFF actual hardware, serving versions, model loading, effective static K and Adaptive-hook absence remain unknown. The failure does not demonstrate an implementation defect. No source or evaluator was patched to force progress.

## PERFORMANCE READINESS

**NOT READY FOR PERFORMANCE BENCHMARK.** Exact blocking evidence: OFF endpoint timed out, zero OFF outputs/counters were captured, both planned token comparisons are MISSING EVIDENCE, and target-only lossless correctness is unsupported. The previous measured-cost consumption finding and execution-mode differences also remain explicit. No throughput, latency, speedup, profitability or benchmark-validity conclusion is made.

## CODE CHANGES

**None in this phase.** All implementation hashes still match the ON baseline. Audit scripts/artifacts were added only under audit/. Earlier uncommitted fixes/tests remain unchanged. No commit, push, merge, PR, Trello update, recalibration, benchmark, ON rerun, A10G substitution or additional OFF session occurred.

## EXACTLY ONE NEXT ACTION

Review the captured Modal L4 scheduling-capacity diagnostics for App ap-l4FtcaDTylSEmfiSoEY0Eg before considering authorization of any replacement OFF attempt.
