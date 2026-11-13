# M4-02-06 — local lossless regression and comparative sweep preparation

Date: 2026-11-13 (Asia/Bangkok). Owner: Bùi Đức Đại. PM: Cấn Đức Khôi.
Scope: local implementation, CPU/static/source-contract verification and user-run
Colab handoff. Final technical freeze/tag remains Khôi's responsibility.

## PM change and baseline

The approved comparison is exactly Vanilla / trained EAGLE3.1 fixed / the same
trained EAGLE3.1 plus project Adaptive. A100/sm80 is future canonical hardware.
Static, old Qwen0.5B Adaptive and arbitrary-tree execution are excluded.

Refreshed baseline `origin/develop` and task HEAD were both
`e7db7ccf17e48814489b067a0307c8fe2dae441e`, ahead/behind 0/0, branch
`feat/m4-02-06-lossless-sweep-freeze`. Tracked tree was clean. Three unrelated
untracked M4-02-04 evidence directories were preserved (exact paths in
`experiments/runs/m4_02_06/m4-02-06-local-readiness-20261113/baseline.json`).
Git fetch required access to the parent worktree's metadata and a per-command
safe.directory exception; no global Git configuration was changed.

Before this replacement local-only request, the superseded task had advanced
the clean task branch from 48ae3a1 to refreshed develop and opened only the
Colab homepage. No notebook, runtime, GPU allocation, Drive mount or inference
was started. Under this local-only request no Colab UI access or cloud compute
occurred. No credentials were requested/read, no Modal resource was used, and
no remote branch/PR/tag/Trello write occurred.

The optional exact ref `origin/feature/M5-02` is absent. The differently named
`origin/feat/m5-02-flexsds-cli-backend` is visible (7695751) and its backend work
is already in refreshed develop. No WIP branch was modified, merged or copied.

## Audit and identities

Inspected README, requirements/pyproject, ignores, runtime/source/tests,
experiment definitions, profile loaders/profilers, frozen workloads, M3-04
checkpoint export, M3-05 evidence, M4-02-04 lineage and M4-02-05 controller
worklogs. The requested `experiments/modal_bench_jobs.py` does not exist:
benchmark job functions are in `modal_runner.py`, with CPU tests in
`test_modal_bench_jobs.py`.

Current fixed EAGLE (`modal_lanes.LANES['eagle']`) is method eagle3, trained
checkpoint, revision c3adca7e..., fixed K=3, default upstream scheduling/executor,
and native speculative summary metrics. Current historical Adaptive uses
Qwen2.5-0.5B, method draft_model, custom async scheduler/planner/adapter/scalar
controller and lineage executor. CURRENT ADAPTIVE IS NOT THE PM-REQUESTED
EAGLE_ADAPTIVE TREATMENT.

Target remains Qwen/Qwen2.5-7B-Instruct-AWQ at
`b25037543e9394b818fdfca67ab2a00ecc7dd641`.
Both new EAGLE treatments use duckhoi/qwen2.5-7b-eagle3.1-speculator at
`c3adca7e211b471d97a519a625df7ffd37b8b461`.
The old draft revision remains historical
`7ae557604adf67be50417f59c2c2f167def9a775`.
Pinned runtime remains vLLM 0.30.0 / torch 2.13.0+cu130 / CUDA runtime 13.0.

Detailed exact-source mapping and limitations are in
`docs/runbooks/m4-02-06-local-methodology.md`. Nineteen exact pinned source hashes
were checked on CPU, including the 15 previously captured installed sources.
The EAGLE proposer inherits the shared base, accepts current-output K, preserves
auxiliary hidden states and has a K=0 cache-synchronizing path. The executor's
old draft_model-only rejection was extended narrowly for sequential, padded,
single-worker V1 async EAGLE3, with additional exact-source verification.
No latest-version API was substituted. GPU compatibility remains unverified.

## Implementation

Separate M4 treatment definitions preserve historical lane meaning. Fixed K=3
and Adaptive ceiling 4 share all model/revision identities, BF16/AWQ Marlin and
serving limits. Adaptive alone enables the project scheduler/controller,
lineage executor, strict budget plumbing, controller histories bounded to 4096,
and planned/produced/feedback/cleanup observations. This is scalar-K adaptation.

Strict token-ID infrastructure requires exact prompt contracts, complete
coverage, two repeatable Vanilla captures and both EAGLE comparisons; one
mismatch fails. Default future diagnostic covers 6 prompts; the supplied
handoff requests `--full-lossless` for all 54 frozen corpus prompts.

The CPU lifecycle regression executes 50 request lifecycles x 20 steps = 1,000
decisions with feedback, K changes, bounded histories, reused external request
IDs, cleanup and lineage release. Historical failure/startup/async suites remain
green. These are CPU resource-lifecycle results, not GPU memory-leak evidence.

A100 profile support maps sm80 explicitly and refuses L4/analytical substitution.
The future profiler uses existing exact-shape worker observations with fresh
processes, observed driver memory release and paired scalar component timings.
Schema validation reproduces costs from raw data; measured labels alone are
insufficient. Scalar-only profiles reject unmeasured tree/batch fallback.
The notebook/orchestrator queries CUDA in short-lived subprocesses so it does
not keep its own CUDA context resident beside workers.

PROFILE CONSUMPTION GAP — TECHNICAL REVIEW REQUIRED for cost-aware claims.
The executing scalar AdaptiveDraftController uses acceptance, KV and free-VRAM
signals, not the measured cost matrix. Generating a profile does not establish
cost-model-aware A100 scheduling. This limitation is documented, without
replacing the PM-requested project algorithm with the dormant tree solver.

Campaign implementation enumerates 54 exact cells, committed workload hashes,
per-prompt caps and repeat sampling indices. W1/W2/W3 counts are 30/12/12.
Windows working bytes have CRLF differences; no corpus was changed. Linux
execution requires exact committed LF bytes. The preparation protocol's null
source field must be frozen by the driver before runtime.

The raw readiness evaluator joins planned K, real proposal shape, verification
counts, native counters, prior-decision lineage and later controller state. It
requires multiple meaningful K states, greedy equality and cleanup. Campaign
launch recomputes readiness and requires the diagnostic already committed and
reachable remotely. No full-sweep override exists.

Append-only Git persistence separates detached RUN_REPO / SOURCE_COMMIT from
branch-tracking EVIDENCE_REPO / EVIDENCE_COMMIT. Temporary askpass uses
user-owned secrets, exact-path staging and ff-only synchronization. Failed
diagnostics are publishable. Each cell start and terminal snapshot is pushed;
all completed cells are revalidated before skipping. Failed/abandoned attempts
require recorded root-cause assessment, never automatic retry. Partial in-flight
bytes not yet pushed can be lost on an ephemeral disconnect; start markers and
committed completed evidence remain. No raw failure may be edited away.

Canonical aggregation requires all 54 exact cells, preserves repeat variance,
native acceptance semantics and paired Vanilla TPOT denominator, and writes a
new `summary_eagle_adaptive_m4.csv` analysis artifact. No measured summary exists
locally. Chapter 4 methodology is prepared; results await actual A100 evidence.

## Verification and local freeze

Evidence root:
`experiments/runs/m4_02_06/m4-02-06-local-readiness-20261113/`.

- Initial focused historical regression: 242 passed, 0 failed.
- Broad relevant CPU regression: **488 passed, 0 failed, 0 skipped**, 54.21 s.
- Final focused verification after review fixes: **88 passed, 0 failed, 0 skipped**.
- Pinned source contract: **19/19 file hashes**, source/AST contract PASS.
- 1,000-step lifecycle and strict lossless comparator: PASS.
- Treatment identity, manifest, profile schema/raw reconstruction, readiness,
  resume/corruption and aggregation synthetic-fixture tests: PASS.
- Eight user-run handoff cells parsed with `ast.parse`; none executed.
- New CLI help entry points run on CPU; no GPU/cloud actions performed.
- `git diff --check`: PASS.

Exact test commands and outputs are in `cpu-tests.json/.log` and
`final-focused-tests.json/.log`. The broad command is
`python -B -m pytest -p no:cacheprovider -q` with the 22 explicitly recorded
files; the final command covers all changed gate/profile/campaign contracts and
the affected cost loaders. Tests use synthetic measurements only in test
fixtures. No scientific measured profile or benchmark result was fabricated.

Frozen executable **SOURCE_COMMIT**:
`187cfff5f8c2a72b9bfb227d97e6221458ddc283`.
The local source commit includes implementation, tests, methodology and local
verification evidence. Later handoff/worklog commits do not change its runtime
identity. `source-freeze.json` verifies committed source against all 28 tested
source-file hashes, allowing only Git's CRLF/LF checkout conversion.

All local failures: none in the final selections. The earlier connection/socket
restriction while downloading pinned source was resolved through an approved
read-only official-source download. No runtime failures were manufactured or
discarded. The earlier local source-download process ended on interruption;
its complete temporary files were revalidated by hash before reuse.

## Handoff and status

User-run instructions:
`logs/daijestic/M4-02/M4-02-06_colab_handoff.md`.
They define repository/branch/exact source, temporary secret authentication, two
clones, base/installed environments, A100 and model gates, CPU tests, profile
measurement, three-treatment full-corpus diagnostic, diagnostic-first Git push,
readiness STOP, full campaign, resume, incremental publishing and aggregation.

LOCAL_READY_FOR_COLAB = **PASS**.
This means local preparation is ready for a gated user-run diagnostic, not that
the experiment has already passed its runtime gates.

profile_sm80.json = **NOT MEASURED — USER MUST GENERATE ON COLAB A100**.
ADAPTIVE_RUNTIME = **NOT RUNTIME-VERIFIED — COLAB A100 DIAGNOSTIC REQUIRED**.
ADAPTIVE_READY runtime result = **NOT EVALUATED**.
FULL_SWEEP = **NOT RUN — 54 CELLS PREPARED FOR USER EXECUTION**.
CANONICAL SUMMARY = **NOT GENERATED — NO A100 RAW DATA**.

Remaining runtime evidence: actual A100/sm80 and VRAM/driver/runtime identity;
model loading at both frozen revisions; measured validated profile; real EAGLE
proposal budgets and accepted/drafted counters; multi-K control, feedback/state
updates and lineage; full token-ID equality/Vanilla repeatability; native
metrics; request and GPU process-memory cleanup; absence of fallback/crashes.
All must pass before the 54-cell sweep. No performance/VRAM-leak/tree-runtime
or cost-profile-aware scheduling conclusion is asserted locally.

The branch remains local for review/publication by the user. No push, merge,
PR, release tag or Trello mutation was performed. Khôi owns final monitoring,
review and freeze/tag. The user first publishes the reviewed task branch, then
manually creates the notebook and follows the handoff.
