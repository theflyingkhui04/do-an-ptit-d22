# OUTCOME

**ON RUNTIME AUDIT PARTIAL.** Exactly one L4 development session launched. Serving, actual adaptive integration, dynamic drafting, attributed verification feedback, fallback/probes and request cleanup were observed. No OFF session, calibration, retry, implementation change or performance benchmark occurred. This is not full audit completion.

## SOURCE

- Branch: develop. HEAD: 8006205a6f19790a776d67155a377def053d2736.
- Checkout: D:/Document/Four-year/do_an/do-an-ptit-d22-flexsds-audit.
- Authoritative local Python: checkout .venv/Scripts/python.exe, Python 3.11.0. FlexSDS import resolves to checkout src/flexsds/__init__.py; FlexSDS 0.2.3.
- External Modal CLI: D:/ProgramFiles/Python3_11/Scripts/modal.EXE, version 1.6.4. Native profile inspection and authentication passed before launch.
- Serving container: Python 3.11.17; vLLM 0.30.0; PyTorch 2.13.0+cu130; compiled CUDA runtime 13.0; NVIDIA driver 610.57.04.
- Pre-existing uncommitted packaging/auth fixes and their tests are captured in environment/identity.json. All audited source hashes are unchanged since pre-launch. The five runtime adaptive source hashes obtained from the container match the checkout.

## GPU SESSION

- App: ap-jaaoDmS5BdYqbji7qTNJdA.
- Container: ta-01M7MM26DEGHAJ3R4SF5HWQE6R.
- Actual GPU: NVIDIA L4, GPU-c3c2b4fc-8ffe-201b-d87f-723f4772c116, nvidia-smi total 23034 MiB.
- Endpoint: https://daibui1208--flexsds-serving-serve-dev.us-east.modal.run . Health/model discovery and CLI readiness inference succeeded; both audit smoke requests returned HTTP 200.
- Frozen models/revisions: Qwen/Qwen2.5-7B-Instruct-AWQ @ b25037543e9394b818fdfca67ab2a00ecc7dd641; duckhoi/qwen2.5-7b-eagle3.1-speculator @ d9016c3860f42cc3f33a87fa5d346fee768f6290; eagle3; L4; max_num_seqs=8; max_model_len=2048; bfloat16; requested AWQ-Marlin. Native engine normalizes quantization to auto_awq. Eager execution, compilation mode 0, custom scheduler/executor, async scheduling and summary speculative metrics appear in actual startup arguments.
- Cleanup: app stopped at 2026-11-17 18:34:02 Asia/Bangkok; state=stopped, tasks=0, container list empty. The local development watcher was subsequently closed after verifying its PID/parent/command. No unrelated app was stopped.

## EVIDENCE

### Actual serving path

Actual CLI -> ModalServingRunner -> uploaded corrected Modal app/src/experiments/profile -> vLLM APIServer -> EngineCore PID 43 -> LineageUniProcExecutor -> AdaptiveAsyncScheduler -> planner -> request-local AdaptiveDraftController -> native EAGLE proposer -> verifier counts -> controller updates.

This is supported by actual startup arguments, source hashes, EngineCore startup telemetry (hook_installed, enginecore_wrapper_entered, executor_published, scheduler_consumed, scheduler_initialized), per-decision controller_class, proposal births, verifier feedback and cleanup records. The scheduler/executor identities are runtime observations, not merely builder configuration. Startup used exactly one attempt/generation/ledger; all three request cleanup events reported active_sessions=0 and active_lineage=0 (CLI readiness request plus two smoke requests).

### Requests and native speculative metrics

Two deterministic chat requests were sent once each with client concurrency 2, temperature=0, seed=42, max_tokens=64, stream=false, return_token_ids=true. Both raw request/response exchanges and response token IDs are preserved. Response IDs map uniquely to runtime request IDs by their client-supplied request-ID prefix, not timing guesses.

- Rainbow: response chatcmpl-flexsds-audit-rainbow; runtime request chatcmpl-flexsds-audit-rainbow-ac325be3; output 64 token IDs, finish_reason=length. Native summary: 13 drafted, 2 accepted, 5 speculative steps.
- Sequence: response chatcmpl-flexsds-audit-sequence; runtime request chatcmpl-flexsds-audit-sequence-ae84f146; output 57 token IDs, finish_reason=stop. Native summary: 13 drafted, 1 accepted, 5 speculative steps.

Totals: 26 drafted, 3 accepted, 10 native speculative steps. These counts come from native metrics and project verifier feedback, not completion-token counts. Native num_spec_tokens=4 is the configured maximum, not an observation that every generation used K=4. Native per_step_accepted/per_step_drafted fields are null; project step telemetry supplies the ten individual feedback records.

### K and feedback lineage

Preserved telemetry contains 119 planned decisions, 120 scheduled-budget records, 10 draft-birth events and 10 attributed verification-feedback records. There are no verification_feedback_skipped records. Each feedback joins a request/producing plan/producing schedule to its selected K, scheduled K, observed proposal-buffer width and native consumed draft count. All ten checked joins agree. K_effective here is the observed birth-buffer width; no token tensor was read by the auditor. Runtime batches were 1, so observed batch width also equals the intended request K for these nonzero generations.

Examples, using request suffixes for readability:

```text
request   producing_plan/schedule  selected/scheduled/birth  drafted accepted  EMA before -> after  later K
rainbow   2/5                      3/3/3                     3       0         .800000 -> .640000   0
rainbow   24/27                    2/2/2                     2       0         .409600 -> .327680   0
rainbow   44/47                    2/2/2                     2       2         .327680 -> .462144   0
sequence  106/111                  2/2/2                     2       1         .327680 -> .362144   0
```

Full rows and exact request IDs are in analysis/joined-lineage.csv. Producing identifiers are explicit; the consuming verifier schedule ID is not exposed. Later decisions are located by same-request JSONL order and their plan IDs, with independently checked controller state; no wall-clock timing join is used. Some initial K=3 drafts were already in flight when low acceptance drove later planning to K=0; asynchronous feedback must not be attached to the latest plan.

### EMA / warmup / fallback

All ten raw updates match m_t=0.8*m_(t-1)+0.2*(accepted/drafted). Every one of 119 logged decision EMA/effective-alpha values matches independently maintained per-request state and correction (m_t−0.8*0.8^t)/(1−0.8^t), clamped to [0,1], with initial alpha 0.8 at t=0.

Each smoke request had five feedback updates and is_warmed_up=false. Zero-draft fallback decisions do not increment that update count. Final EMA was .462144 for rainbow and .362144 for sequence. Sequential request reinitialization to .8 and cleanup were observed; simultaneous multi-row isolation was not exercised.

Observed K values: 3, 0, 2. Low acceptance caused fallback. Four periodic probes at K=2 were observed with proposal births and verifier feedback: rainbow plans 24/44 and sequence plans 86/106. Within each request the probe plan gap is 20 decisions, consistent with the policy cadence. Some probes accepted tokens, but corrected acceptance remained below recovery threshold; recovery was NOT OBSERVED. There was no policy adjustment to force these states.

### Memory / batch

Logged free VRAM: 3500.625–3502.625 MiB. Logical KV occupancy: 0.0078284–0.0547988 percent. These are different signals/units. No memory-guard decision, cooldown, memory-pressure retrigger or recovery was observed. guard_triggered is null in the scalar telemetry; it is not interpreted as an explicit false-state assertion.

Client concurrency=2; configured max_num_seqs=8; every recorded effective runtime planning batch=1. Heterogeneous simultaneous batch-K behavior and cross-row trimming were not exercised. K=0 did not unload EAGLE, and absence of a draft-birth event is not proof of zero drafter/cache/executor overhead.

### Hardware profile

The serving profile is /root/project/hardware_profile.json, transported from the specified local Full L4 artifact. Container SHA256 matches local SHA256 33f02c2b0c6eba1c131e96fbd8341970b73b8d10b56109044ec57304f09833ba. Corrected Modal app validation against actual GPU precedes vLLM startup; the EngineCore environment contains FLEXSDS_PROFILE_PATH and strict mode. Scheduler construction completed with the exact profiled planner source and no fallback. The original evidence/raw SHA256 is 853d4306822798f0bf9f4691760ef6905ec320f1524904a6d500278408271268.

Classification: **PROFILE VALUES NOT CONSUMED BY ACTIVE DECISION PATH.** Runtime records identify src.proposer.controller.AdaptiveDraftController and show acceptance/probe threshold reasons, null dynamic_alpha_star/theoretical_speedup and the actual K outcomes. The matching deployed select_budget source calls memory/budget rules and contains no measured-cost lookup. This is a source-confirmed conclusion about the observed active path, not a newly instrumented count of cost calls. No cost-query counter exists; no evidence resolves the previous non-consumption finding. The profile-loader path ran, but measured latency magnitudes did not enter this policy equation.

### Lossless

**LOSSLESS: INSUFFICIENT EVIDENCE.** Token IDs were captured, but there is no matching control within this ON-only phase. No OFF/target-only process was launched to manufacture a comparison. Greedy output capture is not distribution-preservation evidence.

## FAILURES / UNVERIFIED

- Cold-start health reads timed out before readiness; they later resolved within the same session. No L4 allocation error or serving/model/import failure was observed. The queued/cold-start interval is not labeled as proven capacity failure.
- First read-only container inspection was rejected locally because the Modal CLI required -- before Python's -c. Corrected inspection succeeded on the same container; no server restart occurred.
- Supplemental app-log export failed on Windows cp1252 encoding. The complete local Modal capture and all three telemetry files were preserved independently. Failed export output is retained.
- Initial stop commands aborted at Modal's noninteractive confirmation requirement; documented --yes succeeded. Final app/container state is verified stopped. The local watcher was then deliberately terminated; CLI_EXIT=-1/4294967295 is a post-cleanup local termination result, not an inference failure or server crash.
- No runtime batch>1, K=4, heterogeneous batch trimming, request-ID reuse, preemption/stale feedback or concurrent-row isolation was observed.
- Consuming verifier schedule IDs and consistent request-incarnation fields are not exposed in every record; ten producing-lineage chains are checked, not a claim of universal lineage correctness.
- Ten-update warmup completion, memory guard/cooldown/recovery and fallback recovery are NOT OBSERVED. Native per-step API arrays are null, although the project step trace exposes counts.
- No direct cost-call counter exists; the observed active source/decision path does not use measured costs.
- Lossless correctness, stochastic distribution preservation, OFF behavior, production readiness and all performance claims remain unverified. Smoke timings are not used for performance conclusions. Rainbow output reached its 64-token cap; that is preserved, not rerun.

## CODE CHANGES

**None during this runtime audit.** Audited implementation hashes are unchanged since the pre-launch gate. Local scripts read existing telemetry/process metadata, sent the two requests, and managed lifecycle; they did not patch serving code or add runtime instrumentation. Earlier uncommitted fixes remain unchanged.

## READINESS

Evidence is sufficient to proceed to a **separate Adaptive OFF correctness/comparison phase** on L4 using the saved requests, revisions and limits. This statement does not establish OFF equivalence, lossless correctness, performance benchmarking validity, hardware-aware speedup, or full audit completion. Existing ON/OFF execution-mode differences must remain explicit in that phase.

## EXACTLY ONE NEXT ACTION

Run the separately authorized Adaptive OFF L4 audit using these exact saved smoke requests and frozen model/runtime configuration.

## Artifact index

Paths are relative to this session directory:

- environment/identity.json: pre-launch identity, branch/SHA/status/source hashes.
- startup/frozen-config.json; startup/session.json: frozen configuration and App/endpoint identity.
- startup/container-snapshot-final.stdout: actual runtime versions, GPU, process environment, hashes and complete telemetry bundle.
- telemetry/serving_lineage.jsonl; telemetry/serving_steps.jsonl; telemetry/serving_trace.jsonl: preserved raw container files, hashes checked against snapshot.
- requests/*-exchange.json; requests/*-response-body.json: exact requests, responses, native metrics and token IDs.
- analysis/joined-lineage.csv; analysis/ema-and-warmup-checks.json; analysis/runtime-summary.json: derived joins/checks with raw-record identifiers.
- modal_logs/local-modal-final.log: full local Modal capture, including startup arguments and native metric logging.
- modal_logs/apps-after-stop-03.json; modal_logs/containers-after-stop-03.json: stopped/zero-container evidence.
- analysis/artifact-sha256.json: final evidence hashes.
