# Async lineage pinned-runtime integration

This is integration validation, not a profitability benchmark.

Reviewed baseline: `f889865f82bb91681d436e80570815490b179da5`.
Runtime diagnostics/harness revision: `aeada583e6f9688f2efe9ea2fa2dcc72c0ee43bf`.
Runtime rerun revision: `c7eda977e5c2a568a0355eb211c719b39325fba9`.
Runtime attempt 03 revision: `fb4544166c428e73b164756a2ef0e11ad6b97d6f`.
Difference: harness disables Modal automatic source mounts and explicitly imports
from the canonical committed snapshot via PYTHONPATH. Production code unchanged.
Branch: `feat/m4-02-04-async-lineage-v02`.
Merge base / origin/develop: `53b813a0b6ce8fef530c4286fbd8bf4da5b4a149`.
Tracked baseline clean; three pre-existing untracked evidence directories preserved.

HYPOTHESIS: The real pinned process imports the custom executor, installs the
startup-attempt hook before EngineCore.__init__, constructs LineageUniProcExecutor
before AdaptiveAsyncScheduler, and attributes real speculative feedback to the
exact execution.

WHY CPU IS INSUFFICIENT: CPU/source-contract tests cannot prove child-process
import/bootstrap or actual vLLM execution integration.

PASS OUTPUT: matching installed vLLM 0.30.0 and all 15 manifest hashes; same-PID
hook installation and EngineCore wrapper entry; ordered attempt creation,
executor publication, scheduler consumption, successful context cleanup with
consumed registration retained; real drafted > 0 feedback joined to exact planned
and scheduled IDs; controller before/after telemetry; offline validator PASS.

FAIL OUTPUT: missing hook, startup rejection, source mismatch, missing lineage,
wrong source identity, absent controller feedback, or offline validator failure.

Workload: canonical Modal Lane 3 `_start_vllm` / `_stop_vllm`, L4 default profile,
one greedy request, concurrency 1, max_tokens 64, canonical dynamic K ceiling 4.
No policy, dependency pin, model revision, upstream package, or scientific scope
change. No benchmark matrix, performance comparison, or tree execution claim.

CPU-only Modal preflight at reviewed baseline: all 15 installed hashes PASS,
vLLM 0.30.0, PyTorch 2.13.0, Python 3.11.5. Local CPU source-contract checks:
143 passed before instrumentation; 180 passed after instrumentation.

Local host: daijestic, Windows, Python 3.11.0, PyTorch 2.6.0+cu124,
vLLM absent; RTX 3050 Ti Laptop 4096 MiB; driver 592.27, driver CUDA 13.1.
Local installed-only source gate stopped with FileNotFoundError (vLLM absent).
WSL inventory denied by sandbox. No local model execution attempted.

Failed CPU preflight 01: Windows console encoding could not encode a checkmark;
no GPU allocated. Rerun hypothesis: PYTHONUTF8=1 and PYTHONIOENCODING=utf-8
resolve console output only. Preflight 02 passed; both logs retained.
Network sandbox denied Modal socket access; read-only access succeeded outside
the sandbox. Git metadata writes similarly required escalation because this
checkout is a linked worktree; only exact-path local staging/commit was used.

No safe pre-existing real-process failure injection was identified. Failed-startup
recovery and K=0 remain optional/unverified unless separately evidenced.

Integration launch 01 failed before function execution: Modal source automount
attempted to upload a changing local integration log. Classification:
CHILD-PROCESS IMPORT/PACKAGING FAILURE (client-side packaging). No model execution
or hook evidence. Not canonical. Rerun hypothesis: include_source=False uses only
the committed snapshot already packaged by the canonical image. Offline checks:
AST syntax PASS; reviewed SDK include_source implementation; diff check PASS.
Distinguishing result: remote function imports successfully, source gate passes,
then real startup either proves the gates or produces a classified failure.

Integration launch 02 failed before function execution: Modal rejects image.env
after source mounts. Same client-side packaging category; no model loaded.
Fix: move PYTHONPATH to function env (canonical GPU function already has it).
CPU-only preflight 03 on fb454416 successfully imported from the committed image
without automount; all 15 source hashes and exact vLLM version PASS.
Attempt 03 tests the original GPU hypothesis with this verified packaging.
Validator CPU tests: 15 passed, including wrong SHA/source, missing hook PID,
attempt/generation mismatch, revoked registration, bad lineage/counts/EMA,
tree/performance claims and missing evidence. Synthetic fixtures are tests only.

Attempt 03 (L4): Modal allocation snapshot at 01:59:58Z proves backlog=1,
containers=0, running_inputs=0. No EngineCore/model execution occurred. Stopped
the queued app before trying another allocation. Classification: UNRELATED
INFRASTRUCTURE FAILURE / GPU allocation pending, exact provider cause unknown.
The initial stop command aborted at a noninteractive CLI prompt; `--yes` then
stopped this task's app successfully. No other apps were modified.

Attempt 04 hypothesis: the project's documented PTIT_GPU=A10G fallback may be
available when L4 is unallocated. This cannot be tested offline; pinned bytes,
model revisions, lane args, BF16/AWQ-Marlin profile and policy stay identical.
Expected distinction: a running container and startup logs, or another pending
allocation/infrastructure failure. No simultaneous GPU jobs or second model load.
Source revision remains fb454416; new run ID distinguishes this attempt.
