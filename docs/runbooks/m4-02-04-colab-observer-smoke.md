# Git-based Colab L4 scalar observer handoff

Historical profile: reproduce the original observer smoke from immutable source
`16528241468e2624ccfe07a69a67fcc123bfe4a3`. Its plain `UniProcExecutor` configuration
predates Item 8 and is **not current async-lineage validation**. The current
adaptive scheduler requires a registered `LineageUniProcExecutor` at startup;
this historical runner provides no bypass. Its current CLI refuses live stages
from other source SHAs before hardware checks; offline `validate` remains usable.
Existing L4 raw evidence is unchanged and does not establish current lineage or
pairing runtime correctness. The preparation status below records the original
handoff, not a new GPU run.

Status: PREPARED / LIVE OBSERVER NOT YET VERIFIED. No model, GPU smoke, Modal
execution, profitability benchmark or full grid is run by this preparation.
Production serving source and fixed thresholds remain unchanged.

## Notebook and immutable source

Committed template: `experiments/colab_m4_02_04_scalar_observer_git.ipynb`.
Repository: `https://github.com/theflyingkhui04/do-an-ptit-d22.git`.
Branch: `feat/m4-02-04-scheduler-telemetry-v02`.

The user approved the template/handoff split to avoid self-referencing a commit
SHA inside that commit. The template requires an explicit 40-character CODE_SHA;
it never resolves branch/HEAD as a fallback. After checkpoint commits are pushed
and local HEAD equals the fetched remote branch, render an uncommitted handoff
copy with that exact SHA. It clones source from Git; no ZIP upload is needed.

The structural reference was read at
`D:/Document/Four-year/do_an/colab/m4_02_02_l4_v4_git.ipynb`.
Only its two-checkout/auth/source/evidence patterns are reused. Its grid and
component benchmark cells are not copied; the reference is not a dependency.

## Sequential execution

1. Open the SHA-filled handoff copy in Colab. Select an L4 runtime and configure
   `GITHUB_TOKEN` in Colab Secrets with read/write access to the private repo.
   The first cell checks Linux, exactly one NVIDIA L4 using nvidia-smi and torch,
   compute capability8.9 and a driver compatible with pinned CUDA13.
2. Clone twice using a temporary GIT_ASKPASS script and a token passed only in
   the subprocess environment. URLs stay credential-free. Tokens are not printed,
   embedded in the notebook, staged, or passed to model processes.
3. `RUN_REPO=/content/do-an-ptit-d22-run` checks out `--detach CODE_SHA`.
   `EVIDENCE_REPO=/content/do-an-ptit-d22-evidence` stays on the writable branch.
   Both HEADs must equal CODE_SHA at initial clone time. A mismatch stops without
   reset, merge or rebase. Never execute model code from EVIDENCE_REPO.
4. Create an isolated Python3.11 venv outside both checkouts. An existing
   `python3.11` executable must be available, or set PYTHON311 explicitly to its
   existing path. Install `experiments/requirements-scalar-smoke.txt` under
   `scalar_smoke_constraints.txt`, then pip-check. There is no unpinned fallback,
   silent Python/vLLM substitution or kernel environment replacement. A Colab
   image unable to reproduce the exact stack fails closed.
5. Run environment/source preflight, the six required measurement test files
   (>=230 passes, zero skips), and `test_scalar_colab_smoke.py` plus
   `test_scalar_colab_git.py`. Preserve exact commands, results and log hashes.
6. Check detached HEAD == CODE_SHA and source cleanliness immediately before
   model execution. The only extra untracked source-gate exclusion is the exact
   current `experiments/runs/m4_02_04/<RUN_ID>/` directory. Git-ignored caches
   retain their existing ignore rules. Notebook edits never alter source files.
7. Run only K=0/B1, then K=2/B1 if K0 passes, in fresh isolated subprocesses.
   Reuse preflight evidence only after verifying the same SHA, interpreter,
   commands and log hashes; do not repeat an unchanged test gate unnecessarily.
8. Run offline validation, then copy the complete run directory into the matching
   path in EVIDENCE_REPO. Structural failures are retained too. No failure is
   labelled verified and no automatic retry is performed.
9. Fetch origin and require evidence HEAD == origin/BRANCH before staging. Stage
   only the explicit new run path; inspect cached names/stat/check and reject
   unrelated paths, binary/capture files or credential-like content. Only the
   evidence checkout configures an evidence author and creates a commit.
10. Fetch again before pushing and require the remote still equals the commit's
    parent. Push normally to the same branch and verify the fetched remote SHA.
    Any divergence refuses automatic push; keep both checkouts and local evidence
    commits for manual reconciliation. The detached run checkout is not mutated.

Git is the durable evidence transport. Drive is optional and never required.
Early setup failures stop before a smoke attempt; their setup logs stay under the
run directory and can be preserved using the evidence helper after review.
Smoke/validator failures flow to the final evidence cell without an intervening
success assertion. A push failure leaves all artifacts and any evidence commit
local and prints BLOCKED, never a false successful publication.

## Exact canonical configuration

- NVIDIA L4/sm89; Linux; Python3.11; PyTorch `2.13.0+cu130`, CUDA13.0;
  vLLM `0.30.0`, Transformers `5.22.2`.
- Target `Qwen/Qwen2.5-7B-Instruct-AWQ` at
  `b25037543e9394b818fdfca67ab2a00ecc7dd641`.
- Draft `Qwen/Qwen2.5-0.5B-Instruct` at
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- BF16, requested/effective AWQ Marlin; V1 in-process EngineCore; UniProc;
  TP1/PP1; async scheduling; draft_model; padded absolute positions; ceiling4.
  Canonical graph/cache defaults, memory utilization0.85, model length4096,
  sequence capacity32 are retained. These are capacities, not proof of actual B.
- One explicit counting fixture, greedy seed42, ignore_eos=true, max_tokens48.
  Prompt text, token IDs, hashes and sampling configuration are saved in each arm.

All mismatches fail closed with CANONICAL ENVIRONMENT MISMATCH. The pure
`modal_lanes.py` module supplies canonical constants only; no Modal SDK, volumes
or remote filesystem assumptions enter the measurement contract.

## Observer and tiny window

The measurement-only fixed treatment wraps the isolated planner instance and
replaces its StepPlan for the fixture. Natural controller memory-pressure state
prevents forcing K. Both arms keep the draft model loaded, including K0. Serving
source, thresholds and controller policy are unchanged.

`scalar_live_observer.py` wraps actual executor submission, canonical future
returns, runner preparation/padding, proposer and scheduler retirement. Class
call wrappers count dispatches only for the actual target/draft objects and keep
model/graph-wrapper identity. Counts are Python forward dispatches, including
CUDA graph replay, not CUDA-kernel counts or GPU cache-content validation.

Strong S/future references and process-local observation IDs associate S/O and
retirement; they are not persistent runtime or proposal-lineage identifiers.
Immutable CPU H/P/D/query/row snapshots plus raw V after canonical output
materialization reproduce the verified recurrence. CPU forecasts are never
promoted to exact context. Useful output uses final EngineCoreOutput.new_token_ids
separately from raw V. The primary clock remains perf_counter_ns immediately
after successful scheduler update return.

Four eligible decode retirements settle a queue of at most two entries. One
anchor plus three intervals follows: eight eligible retirements total. Prefill,
no-op, mixed input, unsafe lifecycle and terminal rows cannot enter the primary
window. Capture stops after that tiny window; canonical execution drains with
capture disabled. Already-submitted metadata outside the window is retained as
excluded drain, and final full output-token lists remain diagnostic evidence.

Actual B=1 is checked at worker preparation and proposer, not inferred from client
concurrency. Planned K/K_i, scheduled K, proposer K, assigned K, draft width,
post-trim placeholders and current verification/invalid slots stay separate.
K0 requires an observed draft dispatch and zero-width drafts; it is not vanilla.

No observer CUDA synchronization, tensor readback, early future materialization,
JSON serialization or per-retirement I/O is added. Partial CPU hook/record costs
are retained separately. Disabled/enabled fixture output equality is a limited
structural check, not proof of equal scheduling or zero GPU perturbation.

## Sources and evidence

`scalar_pinned_sources.py` verifies bytes against committed hash-only metadata.
Local historical captures remain local. Clean Colab clones read the exact
installed vLLM distribution after pinned installation; mismatching bytes fail,
missing sources cannot pass the zero-skips gate. No full upstream source capture
or old ZIP is required or committed by this workflow.

`RUN_REPO/experiments/runs/m4_02_04/<RUN_ID>/` contains setup/hardware/install/test
logs, source-gate snapshots, orchestration logs, and child `smoke/` artifacts:
manifest/config, source hashes, environment/pip freeze, fixture, raw records,
S/future/O facts, reconstructed contexts, K traces, timestamps, failures, hashes
and validation report. The whole path is copied byte-for-byte to EVIDENCE_REPO.
Post-publication receipts are outside both checkouts so copied runs stay immutable.

The tiny offline validator reuses the existing identity/window checks, replays
context recurrence and verifies pairing, shapes, K, yield, eligibility and hashes.
It calls exact reference_match; divergent context trajectories remain
UNMATCHED_REFERENCE. A transport smoke can pass with unmatched cost references.
The full-profile validator retains its >=5/50/8 requirements unchanged.

Final output includes CODE_SHA, GPU/stack, source gate, preflight, each K structural
result, observer verdict, offline validator, evidence commit SHA and push result.
Timings are INSTRUMENTATION DIAGNOSTIC ONLY. Profitability is NOT EVALUATED;
full grid is NOT RUN. Current live status remains LIVE OBSERVER NOT VERIFIED.
