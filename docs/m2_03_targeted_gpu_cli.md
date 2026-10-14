# M2-03 targeted terminal diagnostic — review package

**NOT AUTHORIZED TO EXECUTE YET.** This runbook describes future commands after
code, policy, commit and GPU-execution approval. No agent runs on Kaggle.
Historical empirical gate remains **47/54 FAIL**. This package cannot pass it.

## Provenance and preparation gates

Workflow: Windows Codex -> reviewed code commit -> Kaggle terminal -> raw bundle
-> reviewed GitHub evidence checkpoint -> Windows offline analysis.

Historical evidence checkpoint is `bdb26dd89259dcc6b4085041c783bc9536e1287c`;
historical GPU execution is `4b830a8b7530b9bb3218fb9628477ffd164705ba`.
The new diagnostic execution SHA must be a different **full commit SHA**. A later
evidence checkpoint is another commit; never substitute it for execution SHA.

Target only W3 `w3_business_v2_003`, generated index **60**, window 56..64.
Request stays seed=42, temperature=0, max_tokens=768. One workload POST per lane;
no automatic retry, hidden workload warm-up, 54-prompt sweep or Cache ON/OFF cycle.
Internal engine compilation/dummy warm-up is retained and is not an HTTP workload.

Windows has no installed vLLM. Source pinning uses five exact official upstream
`v0.30.0` files; SHA-256/provenance is in
`experiments/m2_03_targeted/upstream.json`. The gzip fixture is source data, not
executable dependency. An already patched v7/v8 installation is rejected.

Before GPU execution, prepare an approved Linux environment with vLLM 0.30.0.
This package does **not** install/upgrade dependencies or download models.
If dependencies/source hashes differ, STOP and return the inventory for review;
do not patch anchors to accommodate the environment.

After the diagnostic code is reviewed and committed, these read-only Kaggle
commands can collect preparation evidence, without model inference:
`inspect-runtime` reads CUDA device properties in a short-lived subprocess;
it does not run a model or send workload requests. Missing UUID support fails
closed rather than assuming which GPU was selected.

```bash
python -B scripts/m2_03_gpu_diag.py inspect-runtime
python -B scripts/m2_03_gpu_diag.py inspect-model --role target --snapshot /ABSOLUTE/HF/snapshots/FULL_MODEL_SHA
python -B scripts/m2_03_gpu_diag.py inspect-model --role draft --snapshot /ABSOLUTE/HF/snapshots/FULL_DRAFT_SHA
```

Use these inventories to fill a COPY of `runtime_policy.example.json` on Windows.
Review and commit the actual policy in the repository before init. It must name
exact Python/torch/CUDA/driver values, T4, vLLM 0.30.0, inherited runtime controls,
both immutable model revisions, absolute snapshot paths and all relevant file
SHA-256 values. Model hashes are streamed, not read as whole weight files.

`ADAPTIVE_ENABLE_VRAM_GUARD` and `ADAPTIVE_CONTROLLER_KWARGS` must be explicitly
reviewed; the example deliberately does not invent historical effective values.
The example is not runnable policy. This preparation loop may require a second
reviewed commit; execute that final SHA, not the first code-only commit.
Init/preflight reject `REVIEW_REQUIRED` anywhere before accepting a template as
runtime policy. Python/torch/CUDA/driver and runtime controls have no assumed
values; vLLM 0.30.0 and Tesla T4 remain protocol constraints.

Read-only Kaggle inventory must collect: exact Python, torch build and CUDA
runtime, driver; CUDA-selected device UUID/capability and physical inventory;
the five installed source hashes; both model IDs, resolved revisions, snapshot
paths and complete config/tokenizer/weight SHA-256 inventories; inherited
VLLM/CUDA/torch controls and the five ADAPTIVE controls printed by inspect-runtime.
An unset ADAPTIVE variable is recorded as null, NOT substituted with a guessed
effective historical value. Historical effective guard/controller settings may
remain UNKNOWN and need a reviewed runtime choice before execution.

No Kaggle Codex, ChatGPT API or AI agent is needed. No GPU inference on Windows.

## Future Kaggle commands (only after explicit execution approval)

Use a clean checkout of the final reviewed commit. Obtain it through your normal
authenticated Git setup; never place credentials in the run directory or scripts.
Set the actual full SHA and committed policy path; examples are placeholders.

```bash
git fetch origin
git checkout --detach FULL_REVIEWED_DIAGNOSTIC_COMMIT
export DIAG_SHA=FULL_REVIEWED_DIAGNOSTIC_COMMIT
export RUN_DIR=/kaggle/working/m203-targeted/UNIQUE_RUN_ID
export POLICY=experiments/m2_03_targeted/APPROVED_POLICY.json

python -B scripts/m2_03_gpu_diag.py init --run "$RUN_DIR" --execution-sha "$DIAG_SHA" --policy "$POLICY"
python -B scripts/m2_03_gpu_diag.py preflight --run "$RUN_DIR"
python -B scripts/m2_03_gpu_diag.py checkpoint --run "$RUN_DIR" --label preflight
python -B scripts/m2_03_gpu_diag.py install-observer --run "$RUN_DIR"
```

Preflight verifies clean committed source, frozen Git-blob request/hash, full
execution SHA, model inventories, exact runtime/source hashes, both free ports,
reviewed launch arguments, policy and output isolation. Init refuses an existing
directory and any directory inside the Git checkout. No fallback to old /tmp
state. No server auto-starts from preflight/install.

### Vanilla

Terminal A (blocks until server exits):

```bash
python -B scripts/m2_03_gpu_diag.py serve --run "$RUN_DIR" --lane vanilla
```

Terminal B: after the readiness log, send one request. A health/model GET failure
does not send workload; inspect first, do not blindly repeat commands.

```bash
python -B scripts/m2_03_gpu_diag.py request --run "$RUN_DIR" --lane vanilla
python -B scripts/m2_03_gpu_diag.py stop --run "$RUN_DIR" --lane vanilla
```

Wait for Terminal A to exit successfully and save `vanilla/server_exit.json`.
Then in Terminal B:

```bash
python -B scripts/m2_03_gpu_diag.py validate-lane --run "$RUN_DIR" --lane vanilla
python -B scripts/m2_03_gpu_diag.py checkpoint --run "$RUN_DIR" --label vanilla
```

The checkpoint command prints an immutable snapshot path. Export/transfer this
checkpoint off the Kaggle VM and verify its descriptor before continuing if VM
loss protection is required. `/kaggle/working` alone is not durable backup.

### Adaptive

Terminal A:

```bash
python -B scripts/m2_03_gpu_diag.py serve --run "$RUN_DIR" --lane adaptive
```

Terminal B:

```bash
python -B scripts/m2_03_gpu_diag.py request --run "$RUN_DIR" --lane adaptive
python -B scripts/m2_03_gpu_diag.py stop --run "$RUN_DIR" --lane adaptive
```

Wait for Terminal A to exit successfully, then:

```bash
python -B scripts/m2_03_gpu_diag.py validate-lane --run "$RUN_DIR" --lane adaptive
python -B scripts/m2_03_gpu_diag.py checkpoint --run "$RUN_DIR" --label complete
export SNAPSHOT=ABSOLUTE_SNAPSHOT_PATH_PRINTED_ABOVE
python -B scripts/m2_03_gpu_diag.py verify-checkpoint --snapshot "$SNAPSHOT"
python -B scripts/m2_03_gpu_diag.py compare --snapshot "$SNAPSHOT"
python -B scripts/m2_03_gpu_diag.py export --snapshot "$SNAPSHOT"
python -B scripts/m2_03_gpu_diag.py restore-observer --run "$RUN_DIR"
```

`stop` only signals the owned process group after verifying PID/start ticks.
PID identity includes Linux boot ID, and zombie/exited processes cannot receive
a retroactive valid stop intent. Expected stop accepts raw child code **0 or -15
only**, never other nonzero codes, and only with all of these preserved proofs:
server-start record bound to the launch/commit, completed HTTP response and its
checksums, durable stop intent, successful SIGTERM dispatch acknowledgement and
ordered monotonic timestamps. A raw `server_exit.json` code alone is insufficient.
The wait process may observe exit before the acknowledgement file is published;
it waits at most 3 seconds for that metadata, without resending any signal/request.
Missing acknowledgement, wrong identity/order, response drift, spontaneous exit 0,
SIGKILL/OOM or wrapper interruption are invalid/UNKNOWN, not expected shutdown.
`serve`, `validate-lane`, the Adaptive gate and comparator share the same checker.
Adaptive launch refuses to proceed unless Vanilla provenance, scores and
accounting were validated; it revalidates that evidence before launch.
No port scanning kills and no broad pkill. Servers bind loopback only, sequentially
on GPU 0, with V1 and prefix caching OFF. Physical two-T4 inventory is recorded;
only GPU 0 is selected. No tunnel or public endpoint is needed.

## Crash and stop rules

- A durable `request.claim` spends the budget **before** the POST. Any claim,
  including an empty one after crash, prohibits another workload POST.
- Timeout/crash after intent means UNKNOWN, even if the request may not have
  reached the server. Do not resend or reuse the run ID.
  Normal `stop` refuses to create a valid stop intent without preserved response.
  If that owned server is still alive, explicit abort can clean it up:

```bash
python -B scripts/m2_03_gpu_diag.py abort --run "$RUN_DIR" --lane vanilla
python -B scripts/m2_03_gpu_diag.py checkpoint --run "$RUN_DIR" --label partial
```

  Select the actual lane. Abort never authorizes resend or a successful diagnostic;
  it permanently invalidates expected shutdown, even if child later returns 0.
- Wrong commit/source/version/model/policy, occupied ports, launch mismatch,
  output collision, OOM, server error or capture error: STOP. No fallback or retry.
- If the server crashes, save a partial checkpoint; it is not a completed result:

```bash
python -B scripts/m2_03_gpu_diag.py checkpoint --run "$RUN_DIR" --label partial
```

- Partial install: do not serve. `restore-observer` can restore only recognized
  pristine/our-patched sources with verified backups, with both ports free.
  Unknown source means STOP, never overwrite it.
  Existing backups are checked even for still-pristine installed files. Restore
  validates ALL sources/backups before its first replacement; mixed pristine/our-
  patched source after an interrupted install can be restored, never auto-resumed.
  Orphan non-Python pending files are retained, not silently deleted.
- Multi-file restore is **NOT atomic**. Each file replacement is separate. Before
  the first replacement, `restore_intent.json` permanently blocks install, serve
  and request for this run. Serve/request also recheck all five installed source
  hashes; mixed files are rejected even if the intent marker was lost. Stop/abort
  remain available to clean up an owned process. Do not run concurrent install,
  restore, server or external source-edit operations in this environment.
- If restore fails or is interrupted, first inspect and preserve its state:

```bash
python -B scripts/m2_03_gpu_diag.py restoration-status --run "$RUN_DIR"
python -B scripts/m2_03_gpu_diag.py checkpoint --run "$RUN_DIR" --label partial
```

  Status reports every installed file as PRISTINE/PATCHED/UNCHANGED/UNKNOWN and
  the overall state. `INCOMPLETE`, `UNKNOWN`, and `UNTRACKED_MIXED` exit 2. Missing
  completion remains INCOMPLETE even if the last source replacement succeeded.
  Do not infer success from an apparently pristine source tree alone.
  With both ports free and no active worker, explicit recovery is:

```bash
python -B scripts/m2_03_gpu_diag.py restore-observer --run "$RUN_DIR"
python -B scripts/m2_03_gpu_diag.py restoration-status --run "$RUN_DIR"
```

  Recovery verifies every current source and backup before any further write,
  accepts only exact pristine/our-patched bytes, and rechecks each replacement.
  Unknown source or changed/missing required backup means STOP for manual review;
  do not delete markers, copy backups blindly, or adapt hashes to current files.
  A reviewed clean-environment rebuild is preferable if integrity is unknown.
  `restore_attempts/` preserves per-attempt progress; `restore_complete.json`
  binds successful completion to the plan and all five pristine source hashes.
  RESTORED means source recovery only: inference remains prohibited for that run.
  This journal handles process interruption, not guaranteed disk/VM-loss recovery.
- Snapshot copies live bytes into a new directory; later log growth cannot
  invalidate that frozen copy. Partial writes and errors are retained. A snapshot
  exceeding 64 MiB fails rather than truncating evidence.
- SIGKILL/VM loss can destroy unflushed data; no trap guarantees recovery. Export
  off-VM after each lane when possible. A failed capture never authorizes retry.

## Evidence checkpoint and Windows analysis

Export creates `.tar.gz` and `.tar.gz.sha256.json` beside the snapshot. Preserve
both exact byte streams. Transfer them using an available approved artifact route
(for example Kaggle saved outputs/download), without exposing credentials.
The transfer method must be confirmed BEFORE inference; this package does not
assume Kaggle output persistence or authentication availability.

After separate review/approval, copy only those two artifacts to a NEW evidence
path such as `experiments/runs/m2_03_targeted/UNIQUE_RUN_ID/`. Stage explicit paths
and create an evidence checkpoint using the approved Git workflow. Never change
canonical/historical JSONL or summaries. This runbook does not authorize commit,
push or merge, and scripts do none of those actions.

On Windows, after that evidence checkpoint is available:

```powershell
git fetch origin
python -B scripts/m2_03_gpu_diag.py import-checkpoint --evidence-sha FULL_EVIDENCE_COMMIT --bundle-path experiments/runs/m2_03_targeted/UNIQUE_RUN_ID/BUNDLE.tar.gz --descriptor-path experiments/runs/m2_03_targeted/UNIQUE_RUN_ID/BUNDLE.tar.gz.sha256.json --output "$env:TEMP/m203-import-UNIQUE_RUN_ID"
python -B scripts/m2_03_gpu_diag.py compare --snapshot "$env:TEMP/m203-import-UNIQUE_RUN_ID"
```

Import reads exact Git blobs, checks bundle SHA/size, rejects traversal/symlinks/
duplicates, extracts outside checkout into a new directory, then verifies every
snapshot file. No newline normalization and no overwrite.

## Interpreting results and limits

- `REPRODUCED`: valid capture, first Vanilla/Adaptive divergence is index 60.
  `historical_token_pair_reproduced` separately states whether IDs are 51/79759.
- `NOT_REPRODUCED`: valid boundary capture and exact full observed outputs.
- `BOUNDARY_NOT_COVERED`: new divergence elsewhere or termination before index 60.
  No causal logits comparison is permitted. No automatic window expansion/replay.
- `INVALID_CAPTURE`: missing/ambiguous/incorrect provenance, row, scores,
  request linkage, accounting, checksum, observer errors or abnormal server exit.

CLI exits 2 on STOP/INVALID_CAPTURE; other diagnostic states exit 0. Exit 0 never
means canonical gate PASS. Full completion/truncation stays separate from equality.

Observer calls are standalone read-only additions. CPU tests strip them and
recover the upstream AST, and check tensor values unchanged. Observer copies add
CPU work, D2H synchronization and filesystem writes, potentially changing GPU
scheduling/timing. These tests cannot prove empirical noninterference or GPU
integration. No capture proves KV/internal-state equality across two lanes.

Mapping review uses the hash-verified real `_get_cumsum_and_arange` and
`_calc_spec_decode_metadata` method bodies from the source fixture, replacing only
the H2D transfer with a CPU copy. At c1 this constructs verification rows 0..K-1
and bonus row K. Generated index is actual position minus prompt length plus 1;
row 60 is emitted only if its offset survives the sampler's accepted/recovered
prefix, and sampler IDs must agree with HTTP/scheduler IDs. Rejected candidate
rows are not evidence of emitted output. The non-speculative row is ordinary.
**UNKNOWN:** synthetic tensors/extracted methods do not exercise real async V1
buffer reuse, stream ordering, Triton acceptance or Linux process integration.
An explicitly approved targeted Linux/T4 run is required to validate the hook
bindings; uncovered roles remain UNKNOWN, with no automatic extra requests.

### Pinned source hook and emitted-index audit

| Owner | Hook position | Required mapping proof |
| --- | --- | --- |
| `GPUModelRunner.sample_tokens` | begin before grammar; end after `_sample`, before runner state update | Actual query positions and `logits_indices`; explicit ordinary/speculative path; single request, no grammar/discard/parallel ambiguity |
| `Sampler.sample` | Before greedy selection, after upstream forward processors | Ordinary row 0, or separately indexed speculative bonus row K |
| `RejectionSampler.forward` | After processors/constraints, before `rejection_sample` | Verification rows 0..K-1; processed matrix source row recorded separately from raw row |
| `Scheduler.update_from_output` | Immediately around `_update_request_with_output` | Complete selected GPU prefix equals scheduler input; emitted IDs equal HTTP slice; stale deliveries rejected |
| `AsyncScheduler._update_request_with_output` | Unmodified, source hash pinned | Non-stale placeholder decrease equals actually emitted count, including stop trimming; optimistic cached output length is never the generated-index source |

At the boundary, scheduler committed start must equal the forward's first
generated index, and `60 - start` must equal the selected source row offset.
Repeated token IDs cannot substitute for that index proof. Ordinary decoding,
accepted draft, recovery token and all-accepted bonus have distinct roles.
Rejected/unselected candidate rows never prove emitted output. Invalid speculative
rows, stale delivery, missing/duplicate scores and mismatched capture points fail
closed. The exact upstream `parse_output` filters placeholders/out-of-vocabulary
tokens and discarded requests; any resulting unproven shift/filtering is
INVALID_CAPTURE, never repaired by guessing indices.

CPU regressions run the exact pinned metadata, parse-output and async-accounting
method bodies with fake CPU dependencies and synthetic events. They also check
hook owner/order and inference AST preservation. They do not execute the actual
GPU runner, rejection Triton kernels, CUDA stream/buffer reuse or Linux lifecycle.
Synthetic score vectors are test data, not scientific GPU logits or proof of
greedy/acceptance correctness. Capture I/O and D2H synchronization can perturb
timing; GPU noninterference and equal KV/internal state remain UNKNOWN.

If observer_error cannot be written, the fallback marker in merged server.log
invalidates validation even when boundary capture files already look complete.
This is regression-tested; missing logs/partial writes also fail validation.

Processed scores are copied immediately before the pinned greedy sampling /
rejection call. Bonus and verification rows are mapped separately; rejected
candidates are not treated as emitted tokens. Dtype, top-1/top-2 and margin are
recorded with full float32 little-endian vector files. Masked -inf is allowed;
NaN/+inf and all-masked rows invalidate capture. Scheduler records are observations
around the actual output update, never replacement accounting.

Numerical sensitivity remains a hypothesis. Equal processed argmax with different
emitted tokens at a prefix/row-verified boundary would contradict a pure argmax-
flip explanation. Different scores do not alone distinguish numerical effects
from different internal state. No target-logits requirement is declared PASS here.

## Windows offline validation before commit

```powershell
python -B -m pytest -q tests/test_lossless_verifier.py tests/test_verify_lossless.py tests/test_verify_lossless_evidence_guard.py tests/test_m2_03_evidence_analyzer.py tests/test_m2_03_targeted_gpu_diag.py -p no:cacheprovider
python -B scripts/m2_03_gpu_diag.py --help
```

Do not install vLLM on Windows or call GPU stages locally. Existing 88 baseline
regressions must remain green; source-transform, mapping, budget, corruption,
crash and checkpoint tests are CPU/synthetic, not T4 inference evidence.
