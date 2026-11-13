# M4-02-06 methodology and pinned runtime review

PM's 2026-11-13 change request supersedes the previous four-lane comparison.
The experiment is Vanilla, trained EAGLE fixed K=3, and the same trained EAGLE
with the project's scalar Adaptive controller (ceiling K=4). All three use
`Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e9394b818fdfca67ab2a00ecc7dd641`.
Both speculative treatments use
`duckhoi/qwen2.5-7b-eagle3.1-speculator@c3adca7e211b471d97a519a625df7ffd37b8b461`.
The source definition is `experiments/m4_treatments.py:definition`.
Historical `modal_lanes.LANES` definitions remain unchanged.

## What executes

Historical fixed EAGLE is `modal_lanes.LANES['eagle']`: `eagle3`, K=3,
native per-request summary metrics, default upstream scheduler/executor.
Historical Adaptive is `draft_model` with Qwen2.5-0.5B, dynamic-K ceiling 4,
`AdaptiveAsyncScheduler` and `LineageUniProcExecutor`.
CURRENT ADAPTIVE IS NOT THE PM-REQUESTED EAGLE_ADAPTIVE TREATMENT.

New M4 Adaptive executes
`AdaptiveAsyncScheduler -> AdaptiveBudgetPlanner -> AdaptiveProposerAdapter ->
AdaptiveDraftController`. Fixed uses upstream async scheduling; Adaptive uses
the project async subclass. Both retain the same target, speculator, BF16/AWQ
Marlin kernel selection and serving limits. Only Adaptive enables the project
executor observations/controller, per-request trimming and dynamic budget.
The executor is plumbing required for attributable Adaptive feedback.
There is no arbitrary-tree runner, topology mask or tree-attention claim.

## vLLM 0.30.0 contract

Official source reference:
[v0.30.0 source](https://github.com/vllm-project/vllm/tree/v0.30.0).
The 15 existing installed-source hashes in `scalar_vllm_source_manifest.json`
match official v0.30.0 bytes. Four additional hashes are retained in
`m4_eagle_source_manifest.json`. Full downloaded source remains in temporary
local inspection storage; no dependency code was silently patched or upgraded.

| Pinned source / symbol | Project dependency and finding | Local evidence / remaining boundary |
| --- | --- | --- |
| `vllm/config/speculative.py:SpeculativeConfig` | `eagle3`, `revision`, greedy draft sampling and `num_speculative_tokens_per_batch_size: list[tuple[int,int,int]]` exist. Dynamic SD uses inclusive batch ranges. | Hash/AST contract; actual model configuration acceptance requires A100. |
| `vllm/v1/core/sched/scheduler.py:Scheduler.schedule` | Mutable `dynamic_sd_lookup` selects `SchedulerOutput.num_spec_tokens_to_schedule` by scheduled batch size. Project `schedule` changes the lookup before calling upstream. | Existing captured-source schedule tests and new pinned audit. This is project-controlled plumbing, not the built-in policy substituted for Adaptive. |
| `vllm/v1/core/sched/async_scheduler.py:AsyncScheduler._update_after_schedule` | Upstream replaces next-proposal placeholders with current output K. Project trims request placeholders afterward, without changing hidden states. | CPU glue tests exercise K=0, heterogeneity, changes and old-versus-new scheduling decisions. EAGLE GPU shape/cache safety remains runtime-required. |
| `vllm/v1/spec_decode/eagle.py:EagleProposer` | Inherits `SpecDecodeBaseProposer`, passing hidden states. | Exact source hashes and AST inheritance test. |
| `vllm/v1/spec_decode/llm_base_proposer.py:SpecDecodeBaseProposer.propose` | Receives mutable K each call. Combines EAGLE3 auxiliary hidden states. K=0 returns `[B,0]` **after** a synchronization forward that maintains draft KV. | Source contract verified. K=0 does not mean zero drafter compute. Actual transition/cache behavior requires A100. |
| `vllm/v1/worker/gpu_model_runner.py:GPUModelRunner.propose_draft_token_ids` | Forwards output K to proposer; retains auxiliary hidden-state construction and padded preparation. Project does not edit these tensors/metadata. | Exact runner hash matches previous installed-source inventory. Real EAGLE execution is unverified. |
| `gpu_model_runner.py:_copy_draft_token_ids_to_cpu`, `_prepare_input_ids`, `sample_tokens` | Executor observes a real proposal-buffer birth and binds later consumption through prior-row indices/width, not callback order. | Existing executor/captured-method CPU tests; added EAGLE source/config gate. Single V1/UniProc worker, async, padded sequential EAGLE only. No V2, parallel drafting or multiworker generalization. |
| `scheduler.py:Scheduler.update_from_output`, `make_spec_decoding_stats` | Per-request accepted/drafted feedback is from a prior proposal. Invalid slots are subtracted. Stale/incarnation mismatches cannot update a new session. | Existing delayed-feedback, startup ownership, failure and cleanup tests; future raw birth/schedule/feedback joins required. |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | Request `return_token_ids`, response-choice `token_ids`, and per-request `metrics` exist. | Hash/AST check; diagnostic preserves original payload, actual emitted IDs and native counters. No decoded-string retokenization. |

The executor formerly rejected every method except `draft_model`. The new
`validate_speculator_contract` explicitly admits only sequential dynamic-K
`eagle3` after verifying extra sources. Historical draft-model behavior remains.
M4 strict mode rejects missing budget plumbing/planning failures rather than
continuing with a fixed budget. Other telemetry/lineage failures invalidate
diagnostics/cells, including native counters without attributable feedback.

## Lossless and metrics

The diagnostic defaults to two frozen prompts per workload, with two Vanilla
captures to qualify repeatability. `--full-lossless` expands to all 54 corpus
prompts, comparing both speculative treatments against Vanilla. Input payloads,
response format, seed 42, temperature 0, output caps and default EOS semantics
are shared. `m4_lossless.compare_captures` rejects incomplete/duplicate captures,
prompt identity drift and every mismatch, including trailing-token differences.
This verifies greedy equality only, not stochastic distribution preservation.

The 54 cells use 5 repeats of 32 measured requests, 3 warmups with the existing
`m3-fixed3` policy, and cycle-rotate sampling. The preparation JSON enumerates
every cell, request cap, sample index and committed workload hash. Its null
source field is deliberately non-executable; `freeze` fills the reviewed SHA
and the runtime validator compares the entire manifest to that frozen source.
Windows working-copy CRLF hashes differ from Git blobs for all three workloads;
the corpus is unchanged. Linux clone with `core.autocrlf=false` must pass the
working-byte gate before measurement.

Harness metrics preserve existing definitions: TTFT is time to first choice
event, TPOT is `(E2E-TTFT)/(completion_tokens-1)`, ITL is event spacing, and
throughput is emitted output tokens per measured repeat wall-second. A stream
event is not a token. Cell latencies/throughput are medians of repeat summaries;
TPOT range/median greater than 15% is flagged. Speedup denominator is paired
Vanilla TPOT p50. Native acceptance is summed accepted/summed drafted, not
speedup. Undefined acceptance in a genuine zero-draft Adaptive cell stays N/A;
zero drafting in fixed EAGLE invalidates the cell. NVML samples include the
whole GPU. Client concurrency is not effective batch size; Adaptive step
telemetry separately records runtime planning batch size.

## A100 profile and claim boundary

Canonical architecture mapping is `sm_80 -> src/flexsds/profiles/profile_sm80.json`.
No such measured file is supplied locally. Missing sm80 profiles cannot silently
load an L4 or analytical profile. The future profile is first written under
the immutable profile run directory and referenced/copied into diagnostic Git
evidence. Keep RUN_REPO unchanged; a later reviewed repository profile copy is
made in EVIDENCE_REPO only after validation.

The new profile uses the existing `metadata/models/grid/cost_matrix_ms` loader
schema, in milliseconds, with measured scalar K=0..4 and B=1,2,4,8,16,32 at
context 1024. Every calibration has 5 warmups and 50 measured repeats; raw
shape observations and paired forward/proposer timings must be complete.
Costs are measured target verification forward plus full EAGLE propose call,
excluding target logits/sampling. They are component cost inputs, not total
end-to-end cycle latency. No bandwidth, tree costs or verification factor is
invented. Scalar-only profiles reject unsupported tree/batch analytical fallback.
The builder reproduces costs from raw paired samples before accepting a profile.

PROFILE CONSUMPTION GAP — TECHNICAL REVIEW REQUIRED for **cost-profile-aware**
claims. The executing `AdaptiveDraftController.select_budget` consumes empirical
acceptance, free VRAM and KV usage thresholds; it does not consume
`profile_sm80.json` costs. Its inherited cost-profile field does not change this
fact. PROFILE MEASURED BUT NOT CONSUMED BY EXECUTING POLICY is the appropriate
future statement if that execution remains unchanged. This change request
keeps the specified project controller; it does not replace it with the dormant
tree cost solver. Memory-signal adaptation and running on A100 do not establish
cost-model-aware scheduling or profitable speedup.

## Persistence, validation and failure handling

RUN_REPO executes a clean detached SOURCE_COMMIT. EVIDENCE_REPO tracks the task
branch; its later commits are EVIDENCE_COMMIT, never execution source. Exact-path
Git publication is append-only and uses temporary askpass plus Colab Secrets.
Diagnostic failure evidence is pushed before readiness is evaluated. Campaign
launch recomputes raw readiness and checks that diagnostic files are committed
and reachable on the remote task branch. No boolean override enables the sweep.

Each cell attempt preserves immutable `started.json` and later `finished.json`,
raw requests, process logs, telemetry and errors. Initial and terminal cell
snapshots are published; completed cells are revalidated before skipping.
RUNNING is not completion. Failed/OOM/timeout/invalid/abandoned attempts require
an explicit assessment naming all previous attempts, hypothesis, offline fix
evidence and expected pass/fail signal. Partial local files are retained when
available. A disconnect can lose unpushed in-flight bytes in ephemeral storage;
committed completed cells and start markers survive. No automatic rerun occurs.

Canonical aggregation requires all 54 exact cells to validate. It writes a new
analysis directory and `summary_eagle_adaptive_m4.csv`; historical TreeAdaptive
filenames are not used to imply unsupported semantics. No results or Chapter 4
performance conclusions are generated locally. Methodology is ready; results
remain pending actual Colab A100 evidence. Khôi owns final review/freeze/tag.
