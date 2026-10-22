# M2-03: targeted boundary 279 on Kaggle/T4

## Early Adaptive timeline extension (2026-10-19; CPU implementation only)

New preparations enable `early_timeline` in the manifest and Adaptive config
only. Keep the existing execution commands, 45 conditioning requests per lane,
sample 004, complete response capture, boundary fingerprint, and all K/V and
layer-1 localization gates. No launcher, metrics mode, runner, speculative budget,
model, cache, graph, sampling or request-order override is added.

The scoped result is `boundary_comparison.json.early_timeline`, schema
`m203.boundary.early.v1`. Raw `early_*` events reuse the existing observer
directory/envelope. They retain the prefill producer, attempts intersecting
positions 176..180 (whole speculative attempts, including rows beyond 180),
associated production/accounting, and the first forward beyond the window.
Rejected attempts and later rewrites remain separate events.

Pinned vLLM's `Scheduler.current_step` is monotonic but not carried in
`SchedulerOutput`; `sched_step_seq` is a conditional deferred-free fence.
The observer preserves `current_step` against the exact output object and
records its process-local identity through worker execution and accounting.
This requires the unchanged same-process executor; lost identity fails closed.
No step identifier is inserted into engine objects. Native counter gaps are
allowed; observer `event_seq` must be complete and unique.
The initial actual prefill suffix is reconciled with the existing cache-lookup
event (including a naturally cached prefix); no full-prompt recomputation is
assumed or requested. Scheduling, rejection rollback and commit counters are
replayed in event order, allowing a later schedule before an earlier commit.

One metadata-only hook immediately after actual draft proposal binds the
producer step, tensor identity, row and produced count. Consumption must match
that tensor and the previous request row, the earlier plan/placeholder origin,
actual verification draft IDs, actual input IDs, and scheduler commits. Future
plan(T) is reported separately from the plan that produced drafts consumed by T.
An initial prefill without a per-request planner decision retains a null budget
and its explicit zero batch budget; no per-request decision is invented.

Actual inputs use bounded existing observer-style GPU-to-CPU reads; the V1
pre-forward positions read is reused at verification. Production copies no
tensor values, selected IDs come from normal CPU accounting, and early events
copy no logits or K/V. No explicit CUDA synchronization, model forward, cache
operation or policy change is added. Blocking small reads and serialization
change timing: `runtime_probe_changes_timing=true` is mandatory.

`diagnostic_generated_index = position - 176 + 1` labels logits rows; input
positions/token ordinals and committed output ordinals remain separate.
Scheduled `-1` slots require count and producer provenance, not literal equality
with real draft IDs. Real scheduled IDs must agree. Ambiguous/missing events,
wrong ownership, counter/commit disagreements, observer errors, a missing unique
first coverage of 177, or failed boundary/KV/localization gates yield
`INVALID_EVIDENCE`. Valid evidence yields `INTERPRETABLE`, never a scientific
PASS: `task_done=false`, `interpretation_status=NOT_YET_APPROVED`, and
`temporal_kv_onset=UNKNOWN`. First coverage does not date K/V divergence or prove
corruption. Historical evidence and prior decisions remain unchanged.

GPU verification of this extension: **NOT RUN**. Stop after local CPU checks;
execution and publication require separate authorization.

## Correction to draft tokenizer revision fix (2026-10-19)

This update supersedes the revision expression in the original entry below.
**FACT (reported EngineArgs reproduction):** the draft model revision is
`7ae557604adf67be50417f59c2c2f167def9a775`, but its `tokenizer_revision` inherits
the target revision `b25037543e9394b818fdfca67ab2a00ecc7dd641`.
The draft `get_tokenizer()` call must therefore use exactly
`revision=spec.draft_model_config.revision`. The target call is unchanged.
The authenticated transformer and AST validator now require this expression;
pristine hashes, observer guarantees, `runtime_fixes` metadata and the
install/authenticate/restore lifecycle remain unchanged. Historical evidence
is preserved. **UNKNOWN:** corrected EngineCore startup on GPU; GPU NOT RUN
for this correction.

## Diagnostic-time draft tokenizer revision fix (2026-10-19)

The authenticated installer now applies one explicit runtime fix alongside the
observation hooks. In pristine vLLM 0.30.0 `DraftModelProposer`, only the draft
`get_tokenizer()` call gains
`revision=spec.draft_model_config.tokenizer_revision`. The target call and model
configuration remain unchanged. This fixes the offline heterogeneous-vocabulary
draft tokenizer lookup; it is not an observation-only transformation.

`instrumentation.json.runtime_fixes` declares
`v1/spec_decode/draft_model.py: draft-tokenizer-revision-v1`. The pristine hash
and backup remain required; installed hashes include the fixed source.
Authentication recomputes the approved transformation and rejects any mismatch.
Restore accepts only pristine or exactly transformed bytes and restores the
original file. No manual installed-source edit or cache/ref workaround is needed.

The AST gate strips observer hooks only in observer-instrumented files.
For the draft source alone, it validates and removes exactly the approved
revision keyword before requiring equality with pristine AST; any different
expression or unrelated AST change is rejected. New evidence must record this
runtime fix explicitly. Historical artifacts are not rewritten.

CPU source/lifecycle verification only; runtime initialization and GPU inference
with this fix: **NOT RUN**.

## Layer-1 historical-position localization

New preparations enable `layer1_localization` in the manifest and both lane
configs. The existing all-target-layer prefix hashes and gates remain required.
Only `model.layers.1.self_attn.attn` adds per-position content, nested as
`layers[layer_name].localization` in its existing `kv_history` event; all
snapshot, request, source, binding and block/write-slot linkage is inherited.

Schema `m203.boundary.layer1.positions.v1` records coverage `[0,454]`
(exclusive end), explicit positions 0..453, dimensions, byte accounting,
and separate K/V prefix digests plus 454 position digests each. The same
canonical FP16 CPU bytes supply both prefix and position hashes, with exact
span reconstruction checked before hashing. No additional device transfer,
CUDA synchronization, model hook or raw value dump is introduced.

The analyzer emits `historical_kv.layer1_localization` with `P_K`, `P_V`,
`P`, `K_differs_at_P`, and `V_differs_at_P`. `LOCALIZED` requires exact
boundary reproduction, valid whole-model `STATE_DIFFERENT`, both expected
layer-1 prefix differences, complete scoped artifacts, and matching prefix
linkage. Missing, malformed or contradictory evidence is `INVALID_EVIDENCE`.
Scoped equality never grants whole-model equivalence. Earliest stored position
does not identify the time or cause of divergence.

GPU verification of this extension: **NOT RUN**. CPU hashing changes timing;
`runtime_probe_changes_timing=true` remains mandatory.

## Historical K/V probe implementation update (2026-10-18)

New preparations enable the narrowly scoped `kv_probe` for sample
`w3_business_v2_004`, query `[(454, 128367)]` only. No additional request,
cache reset, mode override, or prelude change is introduced.
GPU verification of this extension: **NOT RUN**.

Installation authenticates the bounded V2/Triton source set and preserves its
pristine bytes alongside the existing boundary sources. Only instrumented
files are rewritten. Any unsupported source/backend/layout fails closed.

The observer records final FULL capture bindings outside capture, excluding
warm-up and throwaway memory-profiling captures. Immediately before FULL replay
it verifies those persistent storage identities; PIECEWISE and Adaptive V1
observe their prepared runtime bindings before forward. The supported target
representation is one full-attention cache group, dense FP16, unit scales,
single-device execution; other representations are invalid evidence.

`kv_history` events contain per-target-layer K/V SHA-256 for all historical
positions 0..453, canonical position/head/dimension ordering, source linkage,
coverage, addresses/strides, actual dispatch, and current-write-slot safety.
Only small indices/metadata and one layer/component are staged at a time.
Copies introduce synchronization/timing changes; exact reproduction remains
mandatory. Capture identities copy no tensor values.

The analyzer adds `historical_kv.status`: `STATE_DIFFERENT`,
`STATE_EQUIVALENT`, or `INVALID_EVIDENCE`. Only complete authenticated equality
sets `historical_kv_state_equivalent=true`. The broader
`same_kv_cache_state_proven` remains false: this probe does not establish equal
current-position activations or all internal execution state.
Changed boundary fingerprints are invalid K/V evidence. Existing historical
bundles without the probe remain score evidence only.

Local implementation and CPU verification only. GPU execution: **NOT RUN**.
The user must invoke this runbook on the approved, otherwise idle T4 setup.
Publishing a reviewed execution checkpoint requires separate commit/push approval.
Do not run from a dirty checkout or substitute archive code as the execution tree.

## Question and validity

Observe `w3_business_v2_004`, request **46**: W1 001–030, W2 001–012,
W3 001–003, then 004. Both Protocol B pairs have this exact order. No evidence
justifies removing preceding requests; replay all 45 as conditioning, once per
lane. No extra warm-up, retry, cache reset, graph-mode override, deterministic
flags, k0/k4 sweep, or remaining eight prompts. Two sequential lanes, not servers
running together. These 92 requests are one targeted diagnostic, not a new
Protocol B acceptance run.

Reference: evidence checkpoint `4372f8b81998e8a50ee2b87ea6d7c4a9c4b523c5`,
original execution `a63baa1c42fd75f542f876ab512765149ddd5e20`.
The observer necessarily has a **new** execution SHA; manifest and source bundle
record it. Model revisions, frozen requests and launcher settings stay unchanged.

Interpret only an exact reproduction: first mismatch **279**, **79126 → 128273**,
lengths **284/282**, finish **stop/stop**. Another boundary, including historical
200, yields `NON_REPRODUCTION`. Missing/ambiguous alignment, missing scores,
observer failures or incomplete capture yield `INVALID_EVIDENCE`.
`INTERPRETABLE` only qualifies this diagnostic evidence; it is not M2-03 PASS.
Always retain `interpretation_status=NOT_YET_APPROVED`, `task_done=false`.

## Reused implementation and limitations

Ported only observation calls, source-hash/AST checks, `map_rows`, processed-score
hooks and scheduler accounting from archive `a7b52e7`'s
`experiments/m2_03_targeted/{observer,sources}.py`. A3/V8 supplies the rule to use
actual GPU positions/input IDs rather than optimistic output lengths/placeholders.
The cache lookup observation records the scheduler's actual local lookup result
and external count. It does **not** prove KV tensor equality or cache causality.
No R14–R16 controls, attention patches, KV recomputation or graph changes are ported.

The boundary observer copies scores to CPU and changes timing. Reproduction is
therefore checked before interpreting those scores. An ordinary row predicts
index 279 at model position `176 + 279 - 1 = 454`; speculative forwards must
prove the selected row from actual `logits_indices`, positions, returned sampled
IDs and scheduler commits. Rejected candidate rows never qualify.

Raw/processed vectors and top ten candidates plus both fingerprint token scores
are retained. Scheduler before/after events retain committed prefixes, emitted
IDs, scheduled drafts and async counters. The comparison verifies their linkage
to the HTTP conditioning prefix. A boundary sampled/committed/emitted token
disagreement is retained for H3 rather than rejected as invalid evidence.
Runtime metadata includes models, revisions, GPU/driver,
CUDA/PyTorch/vLLM, session/boot, source hashes and process command/lifetime.
Baseline/Adaptive effective graph modes are recorded rather than equalized.

## Exact execution

The 2026-10-18 diagnostic used Vanilla Model Runner V2 and Adaptive V1
(`draft_model` fallback). The observer now also pins and patches the executed
V2 sampler from evidence checkpoint `31d31f242af55faecb262114e738aa0340c07429`.
V2 raw capture runs after model graph replay, processed capture runs after
top-k/top-p and before Gumbel sampling, and output capture retains sampled IDs
and sampled/rejected counts. Only single-request ordinary Vanilla decoding
without structured output is supported. A selected row without pre-Gumbel
scores (including an uninstrumented FlashInfer path) is invalid evidence.
No runner-selection or graph-mode override is added; V1 Adaptive hooks remain.

Use **one Bash session**, from a clean reviewed checkout on
`fix/m2-03-lossless-clean`. This checkout must contain the evidence Git object
and the new diagnostic at an explicitly reviewed checkpoint. Set
`APPROVED_DIAGNOSTIC_SHA` to that checkpoint before pasting this block.
The environment must already contain vLLM 0.30.0, PyTorch 2.13.0+cu130 / torch
CUDA 13.0 and both cached model snapshots. No installs or model downloads.
Do not invoke while another workload/server is using the GPU: lane switching
stops existing servers. Source instrumentation refuses any unknown vLLM hash.

```bash
set -euo pipefail
set -o noclobber
: "${APPROVED_DIAGNOSTIC_SHA:?Set the reviewed diagnostic execution SHA}"
test "$(git branch --show-current)" = fix/m2-03-lossless-clean
test "$(git rev-parse HEAD)" = "$APPROVED_DIAGNOSTIC_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
git cat-file -e 4372f8b81998e8a50ee2b87ea6d7c4a9c4b523c5^{commit}
git ls-files --error-unmatch experiments/protocol_b_boundary.py \
  experiments/protocol_b_boundary_observer.py experiments/protocol_b_boundary_sources.py

# Bind imports to this reviewed repo root in every server/EngineCore child.
test "$PWD" = "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import experiments.protocol_b_boundary_observer'

export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
test "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader)" = 'Tesla T4'
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=$(python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)')
export PROTOCOL_B_TARGET_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export PROTOCOL_B_DRAFT_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
export VLLM_USE_V1=1
export ADAPTIVE_ENABLE_VRAM_GUARD=1
export ADAPTIVE_MAX_K=4
export ADAPTIVE_MIN_BATCH_K=0
export ADAPTIVE_PER_REQUEST=1
export ADAPTIVE_CONTROLLER_KWARGS=''
test -z "${CUBLAS_WORKSPACE_CONFIG:-}"
SESSION_ID=$(cat /proc/sys/kernel/random/uuid)
RUN_ROOT="/kaggle/working/m2-03-boundary279-$(date -u +%Y%m%dT%H%M%SZ)-$SESSION_ID"
export ADAPTIVE_TELEMETRY_PATH="$RUN_ROOT/adaptive_telemetry.jsonl"
python -m experiments.protocol_b_boundary prepare "$RUN_ROOT" --session "$SESSION_ID"

finish() {
  local prior=$?
  trap - EXIT
  set +e
  bash scripts/switch_lane.sh stop > "$RUN_ROOT/final-stop.log" 2>&1
  local stopped=$?
  if [ "$stopped" -eq 0 ] && ! grep -Fq 'GPU vẫn còn' "$RUN_ROOT/final-stop.log" \
     && [ -f "$RUN_ROOT/instrumentation.json" ]; then
    python -m experiments.protocol_b_boundary restore "$RUN_ROOT" > "$RUN_ROOT/restore.log" 2>&1
    local restored=$?
    if [ "$restored" -ne 0 ]; then prior=1; fi
  elif [ -f "$RUN_ROOT/instrumentation.json" ]; then
    prior=1
  fi
  # Also run on partial/error evidence. The analyzer keeps INVALID_EVIDENCE.
  python -m experiments.protocol_b_boundary analyze "$RUN_ROOT" > "$RUN_ROOT/analysis.log" 2>&1
  local analyzed=$?
  if [ "$analyzed" -ne 0 ]; then prior=1; fi
  nvidia-smi -q > "$RUN_ROOT/nvidia-after.txt"
  printf '%s\n' "$prior" > "$RUN_ROOT/workflow-exit-code.txt"
  python -m experiments.protocol_b_boundary seal "$RUN_ROOT"
  local sealed=$?
  if [ "$sealed" -ne 0 ]; then prior=1; fi
  exit "$prior"
}
trap finish EXIT

nvidia-smi -q > "$RUN_ROOT/nvidia-before.txt"
python -m pip freeze > "$RUN_ROOT/packages.txt"
git status --short > "$RUN_ROOT/git-status.txt"
git ls-files src scripts tests 'experiments/*.py' experiments/configs > "$RUN_ROOT/source-files.txt"
tar -czf "$RUN_ROOT/execution-sources.tar.gz" -T "$RUN_ROOT/source-files.txt"

# Invoke on an idle session only. Never install while a server is importing code.
bash scripts/switch_lane.sh stop > "$RUN_ROOT/initial-stop.log" 2>&1
if grep -Fq 'GPU vẫn còn' "$RUN_ROOT/initial-stop.log"; then exit 1; fi
python -m experiments.protocol_b_boundary install "$RUN_ROOT"

assert_identity() {
  test "$(git rev-parse HEAD)" = "$APPROVED_DIAGNOSTIC_SHA"
  test -z "$(git status --porcelain --untracked-files=no)"
  test "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" = "$CUDA_VISIBLE_DEVICES"
}

lane() {
  local name="$1" selector="$2" number="$3" rc=0
  export PROTOCOL_B_BOUNDARY_CONFIG="$RUN_ROOT/$name/boundary_config.json"
  assert_identity
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -q > "$RUN_ROOT/$name/gpu-before.txt"
  bash scripts/switch_lane.sh "$selector" > "$RUN_ROOT/$name/switch.log" 2>&1 || rc=$?
  if grep -Fq 'GPU vẫn còn' "$RUN_ROOT/$name/switch.log"; then rc=1; fi
  if [ "$rc" -eq 0 ]; then
    python -m experiments.protocol_b_boundary capture "$RUN_ROOT" --lane "$name" \
      > "$RUN_ROOT/$name/capture.log" 2>&1 || rc=$?
  fi
  # Copy before the next switch can overwrite the shared launcher log.
  if [ -f "logs/vllm_lane${number}.log" ]; then
    cat "logs/vllm_lane${number}.log" > "$RUN_ROOT/$name/server.log"
  fi
  printf '%s\n' "$rc" > "$RUN_ROOT/$name/exit-code.txt"
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -q > "$RUN_ROOT/$name/gpu-after.txt"
  assert_identity
  return "$rc"
}
lane vanilla lane1 1
lane adaptive lane3 3
```

`finish` retains partial data, stops servers, restores known installed sources,
analyzes once and seals once, including failure paths. It never retries a request
or starts another experiment. If stopping/restoring fails, preserve that failure
and do not run further inference in that environment. If installation fails,
backups and instrumentation metadata remain under the fresh directory; stop and
restore from them before inference. Do not delete or reuse the directory.

## Reading the sealed directory

Verify `SHA256SUMS` from inside the directory with `sha256sum -c SHA256SUMS`.
Read `workflow-exit-code.txt`, lane exit codes, stop/restore logs and
`boundary_comparison.json`; a shell exit alone is not a scientific result.

- `manifest.json`: source evidence, 46-request prefix, expected results and identity.
- `{vanilla,adaptive}/run.json`, `responses.jsonl`: raw API evidence, errors and provenance.
- `{vanilla,adaptive}/observer/`: raw/processed vectors, selected-row and accounting/cache events.
- Lane server/switch/capture logs and GPU snapshots.
- `instrumentation.json`, `vllm-originals/`, source bundle, package inventory and checksums.

Import-path provenance is already retained in each lane's `run.json` and
`environment.json`: `config.environment.execution_root` and `base_pythonpath`.
`run.json` also retains `live_server.environment.PYTHONPATH` read from the
live server. The exported value is inherited through `switch_lane.sh`, the
launcher and vLLM children; the Adaptive launcher additionally prepends the
same repo root. Existing capture validation checks both lane-specific values.

For an interpretable reproduction, unequal raw vectors localize a difference
before sampling but do not isolate numerical execution from KV/prefill history.
Equal raw but unequal processed scores point to processing; equal processed
scores with different sampled/emitted IDs point further downstream. Review
row roles, masks and committed prefixes before drawing conclusions. H2 remains
unresolved unless state is directly discriminated; no automatic root-cause label
or acceptance decision is emitted.

## Position-177 stored K/V discriminator — 2026-10-19

New preparations enable `postwrite_position177` in the manifest and both lane
configs. Historical directories without this flag retain their original analysis
contract. The request order, model revisions, sampling, cache policy, graph modes,
runner selection and existing boundary/historical observations remain unchanged.

This probe observes only `model.layers.1.self_attn.attn` at logical position 177.
Vanilla must be ordinary V2 FULL decode with actual positions `[177]`; Adaptive
must be the first authenticated V1 PIECEWISE speculative target verification with
actual positions `[177,178,179,180]`. Current early-timeline identities establish
the first verification; historical schedule numbers are never selection keys.
No layer-0 hidden states or pre-store Q/K/V intermediates are retained.

The existing `kv_v2_full` observation stays immediately before graph replay.
A separate `postwrite_v2_full` hook runs immediately after
`self.graphs[desc].replay()`, before the manager returns. The existing V1 historical
hook also stays before the target forward. A separate `postwrite_v1` hook runs
after the complete `model_output = self._model_forward(...)` assignment, inside
the same surrounding context and before postprocessing, sampling or draft work.
These are post-target-forward reads of persistent cache storage, not Python
module hooks inside captured computation and not observations at the store
instruction itself. Source SHA authentication, exact hook placement checks,
strip-to-pristine AST validation, and install/authenticate/restore remain required.
The approved draft-tokenizer runtime fix stays distinct from observer insertions.

Preparation retains metadata and tensor references in dedicated post-write state;
it never copies K/V values and never overwrites historical `_local.kv_pending`.
The post hook consumes the prepared state once, checks the same manager/descriptor
or scheduler-output/runner identity, and revalidates bindings and execution stream.
FULL requires the exact final capture binding; warm-up/capture cannot qualify.
Later Adaptive rewrites do not replace the first capture. Duplicate emitted
captures or missing/stale evidence invalidate the result.

Each lane independently resolves the cache's actual kernel block size `B`:
`block_table[batch_row, 177 // B]`, offset `177 % B`, and
`slot = physical_block * B + offset`. The unique actual position-177 row must
write that slot. Bounds, actual-row aliases, padding aliases over the complete
consumed slot mapping, target/draft ownership and overlapping cache storage are
checked. Physical block IDs are not compared between lanes. The existing
`history_slots()` all-454-position contract is unchanged.

For the supported dense FP16 `[block, kv_head, token_in_block, 2*head_dim]`
cache, K selects the first `head_dim` values and V the second half. Each component
uses `detach -> one bounded blocking CPU transfer -> contiguous -> uint8 bytes ->
SHA-256`, in `[kv_head, head_dimension]` order. Original bits, including signed
zero, are preserved. Shape, dtype, element count and byte count are recorded.
Added K/V payload is `4 * heads * head_dim` bytes per lane: 2048 bytes for
4 heads of dimension 128. Metadata reads are additional; the byte total describes
K/V payload only. Existing later per-position hashes are reused without another
later value transfer.

The blocking reads wait for current-stream work and perturb timing. The probe
records `runtime_probe_changes_timing=true`, `blocking_device_to_cpu=true`, and
`device_wide_synchronize=false`. It adds no explicit `torch.cuda.synchronize()`,
stream switch, graph operation, model input change, or graph-mode override.

`boundary_comparison.json.postwrite_position177` is interpretable only after the
exact mismatch fingerprint (279, 79126/128273, 284/282, stop/stop), authenticated
historical `STATE_DIFFERENT`, layer-1 `P_K=P_V=P=177`, valid speculative-first early
timeline, and all existing source/ownership/no-observer-error gates pass.

- If immediate K or V differs: `IMMEDIATE_STORED_KV_DIVERGENCE`. This does not
  distinguish upstream numerical differences from cache writing.
- If both immediate components match and later position-177 digests differ:
  `LATER_STATE_CHANGE`. Per-lane/component `*_same_as_later` fields identify the
  changed stored values, without assigning an overwrite/rollback mechanism.
- Missing, duplicate, incompatible or unauthenticated evidence:
  `INVALID_EVIDENCE`.

The scoped result always keeps `task_done=false`,
`historical_kv_state_equivalent=false`, `same_kv_cache_state_proven=false`,
`interpretation_status=NOT_YET_APPROVED`, and `temporal_kv_onset=UNKNOWN`.
CPU byte/metadata/analyzer regressions do not establish CUDA execution or empirical
losslessness. GPU verification of this new probe has **NOT RUN** during local
implementation; any future run needs its own authorization and fresh evidence path.

## One-shot P177 producer-k0 control — non-canonical

**NON-CANONICAL DIAGNOSTIC CONTROL. NOT VALID FOR 54/54 ACCEPTANCE BY ITSELF.**
This section is separate from the canonical exact-execution block above. Do not
execute that block to prepare this control: it would rerun both reference arms.
Local implementation stops after CPU tests. The following commands are a future,
separately authorized GPU procedure, not authorization to run GPU now.

Reuse the immutable archive at evidence commit
`9c0f15e8ebcef4d3fff3bd4aa07b6f41d3288ff2`:
`experiments/runs/m2_03/diagnostics/postwrite-position177-c306aef-20261019T041028Z.tar.gz`.
Its execution baseline is `c306aef3c8ffea97aa2e045507c2245ee77df220`;
its Git blob is pinned as `b5590f982b97c0aa6a05ea0217eb6413f7bb2510`.
Preparation checks that exact blob, every internal SHA256SUMS entry, and the
canonical boundary/KV/localization/early/post-write validators in a temporary
verification copy. The archive itself is never rewritten. Its SHA-256 is retained
in the new manifest and config. No fresh Vanilla or normal Adaptive arm is run.

Keep the baseline physical GPU/boot, models, packages, graph configuration,
sampling/seed, prefix-cache policy and frozen workload/prelude. Use the same
execution-root/import-path environment. Only fresh run/session identifiers,
diagnostic source changes, and the telemetry output path may differ. Preparation
and analysis reject incompatible provenance; no new environment or model rebuild
is part of this procedure.

The sole mutation is an explicitly authenticated call immediately before native
`super()._update_after_schedule(scheduler_output)`. For the unique P176 producer,
it sets only `scheduler_output.num_spec_tokens_to_schedule=0`. The normal planner
decision and lookup table remain untouched. Native AsyncScheduler must see zero
and create zero next-step placeholders; the producer must execute `[176]`, produce
zero drafts, and lead to the first ordinary consumer `[177]`. Exact live output
objects and early-timeline origins bind the chain; schedule numbers are not
selection keys. A second override, wrong query, missing native proof, or ambiguous
request invalidates the control. The native AsyncScheduler source is an additional
control-only pinned dependency; the canonical 19-file source set is unchanged.

The existing P177 post-write hook retains its original FP16 K/V byte contract.
Only this profile accepts V1 `[177]` with zero scheduled drafts. It also hashes
layer-1 logical positions 0..176 at that same observation and compares every K/V
position digest with sealed normal localization. Prefix slots must be unique and
disjoint from all current/padded writes. Added prefix payload is
`177 * heads * head_dim * 2 components * 2 bytes` (362496 bytes for 4x128 FP16),
using two bounded blocking D2H transfers. This perturbs timing after the target
forward and proves scoped layer-1 prefix equality only. No all-layer probe, graph
override, stream change, or explicit device-wide synchronization is added.

After review and separate GPU authorization, use an idle baseline environment:

```bash
set -euo pipefail
set -o noclobber
: "${APPROVED_DIAGNOSTIC_SHA:?Set the reviewed control implementation SHA}"
test "$(git rev-parse HEAD)" = "$APPROVED_DIAGNOSTIC_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test "$(git branch --show-current)" = fix/m2-03-lossless-clean
test "$PWD" = "$(git rev-parse --show-toplevel)"
# Retain the exact frozen exports/PYTHONPATH from the sealed session.
# Do NOT invoke the earlier canonical two-lane execution block.
python -c 'import experiments.protocol_b_boundary_observer'
export M203_P177_PRODUCER_K0=1
BASELINE="$PWD/experiments/runs/m2_03/diagnostics/postwrite-position177-c306aef-20261019T041028Z.tar.gz"
test -f "$BASELINE"
SESSION_ID=$(cat /proc/sys/kernel/random/uuid)
RUN_ROOT="/kaggle/working/m2-03-p177-k0-$(date -u +%Y%m%dT%H%M%SZ)-$SESSION_ID"
export ADAPTIVE_TELEMETRY_PATH="$RUN_ROOT/adaptive_telemetry.jsonl"
python -m experiments.protocol_b_boundary prepare-control "$RUN_ROOT" \
  --session "$SESSION_ID" --baseline "$BASELINE"
export PROTOCOL_B_BOUNDARY_CONFIG="$RUN_ROOT/adaptive/boundary_config.json"

finish_control() {
  local prior=$?
  trap - EXIT
  set +e
  bash scripts/switch_lane.sh stop > "$RUN_ROOT/final-stop.log" 2>&1
  local stopped=$?
  if [ "$stopped" -eq 0 ] && ! grep -Fq 'GPU vẫn còn' "$RUN_ROOT/final-stop.log" \
      && [ -f "$RUN_ROOT/instrumentation.json" ]; then
    python -m experiments.protocol_b_boundary restore "$RUN_ROOT" > "$RUN_ROOT/restore.log" 2>&1
    if [ "$?" -ne 0 ]; then prior=1; fi
  else
    prior=1
  fi
  python -m experiments.protocol_b_boundary analyze-control "$RUN_ROOT" > "$RUN_ROOT/analysis.log" 2>&1
  if [ "$?" -ne 0 ]; then prior=1; fi
  printf '%s\n' "$prior" > "$RUN_ROOT/workflow-exit-code.txt"
  python -m experiments.protocol_b_boundary seal "$RUN_ROOT"
  if [ "$?" -ne 0 ]; then prior=1; fi
  unset M203_P177_PRODUCER_K0
  exit "$prior"
}
trap finish_control EXIT
git ls-files src scripts tests 'experiments/*.py' experiments/configs > "$RUN_ROOT/source-files.txt"
tar -czf "$RUN_ROOT/execution-sources.tar.gz" -T "$RUN_ROOT/source-files.txt"
python -m pip freeze > "$RUN_ROOT/packages.txt"
nvidia-smi -q > "$RUN_ROOT/nvidia-before.txt"
bash scripts/switch_lane.sh stop > "$RUN_ROOT/initial-stop.log" 2>&1
if grep -Fq 'GPU vẫn còn' "$RUN_ROOT/initial-stop.log"; then exit 1; fi
python -m experiments.protocol_b_boundary install "$RUN_ROOT"
bash scripts/switch_lane.sh lane3 > "$RUN_ROOT/adaptive/switch.log" 2>&1
python -m experiments.protocol_b_boundary capture-control "$RUN_ROOT" > "$RUN_ROOT/adaptive/capture.log" 2>&1
cat logs/vllm_lane3.log > "$RUN_ROOT/adaptive/server.log"
nvidia-smi -q > "$RUN_ROOT/nvidia-after.txt"
```

Only Adaptive is launched; the existing collector sends the 45 conditioning
requests followed by sample 004, never all 54. Failure paths retain partial data,
stop, restore known instrumentation, analyze once, and seal once. No retries.
Never resume inference after a stop/restore failure. The environment gate must be
unset before returning to canonical operation.

Read `p177_k0_control.json`, raw events, journals, source/provenance bindings and
exit logs. Control K/V equal to sealed Vanilla supports
`SPECULATIVE_MULTI_TOKEN_FORWARD_DIVERGENCE`; unequal control K or V yields
`RUNNER_OR_EXECUTION_PATH_STILL_DIFFERENT`, only after all gates pass. A missing
or mismatched prefix/lineage/provenance/source record yields `INVALID_EVIDENCE`.
The control need not retain the later mismatch-279 fingerprint. That fingerprint
remains mandatory for the sealed canonical reference and ordinary canonical
analysis; this separate result never grants canonical acceptance.

Always retain `non_canonical=true`, `task_done=false`,
`same_kv_cache_state_proven=false`, `historical_kv_state_equivalent=false`,
`temporal_kv_onset=UNKNOWN`, and `interpretation_status=NOT_YET_APPROVED`.
The classification does not isolate projection arithmetic from cache writing,
establish all-layer equality, or prove general speculative-decoding incorrectness.

## 2026-10-19: one-shot layer0 attention substitution (non-canonical)

This is a separate causal diagnostic, not Candidate A's production fix. CPU
verification does not authorize a GPU run. Keep the frozen environment, physical
GPU/boot, model revisions, sampling, prefix cache and PIECEWISE graph setting.
Do not launch Vanilla or the full 54. Retain the existing 45 conditioning requests
and then `w3_business_v2_004` through the existing collector.

The normal P176 producer keeps k=3. The actual target query remains
`[177,178,179,180]`, with three drafts. The hook runs only in target layer0's
Triton attention, after native attention returns its result and before returning
to Qwen's output projection. It computes one shadow using the same live Q[0:1]
and K/V views, query offsets `[0,1]`, sequence length `[178]`, the same block table
and scale, and native segmented-3D attention with 16 segments. Output and scratch
are private allocations; live attention metadata and cache are never rewritten.
The sole intervention copies the shadow into output[0:1]. Rows 1:4 remain native.
There is no second model forward, RoPE, cache-write call or budget override.

Use the preceding control workflow's preparation, provenance collection,
switch/stop/restore trap and seal lifecycle, with **all** substitutions below.
Use a fresh run root and review the resulting commands before any GPU execution:

```bash
# Run from the reviewed clean execution checkout with the frozen environment
# already exported, after existing switch_lane.sh stop has succeeded.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset M203_P177_PRODUCER_K0
export M203_P177_ATTENTION_SUBSTITUTION=1
SESSION_ID=$(cat /proc/sys/kernel/random/uuid)
RUN_ROOT="/kaggle/working/m2-03-p177-attention-$(date -u +%Y%m%dT%H%M%SZ)-$SESSION_ID"
export ADAPTIVE_TELEMETRY_PATH="$RUN_ROOT/adaptive_telemetry.jsonl"
BASELINE="$PWD/experiments/runs/m2_03/diagnostics/postwrite-position177-c306aef-20261019T041028Z.tar.gz"
CONTROL="$PWD/experiments/runs/m2_03/diagnostics/p177-k0-control-692f21c-20261019T062959Z.tar.gz"
python -m experiments.protocol_b_boundary prepare-attention "$RUN_ROOT" \
  --session "$SESSION_ID" --baseline "$BASELINE" --control-reference "$CONTROL"
export PROTOCOL_B_BOUNDARY_CONFIG="$RUN_ROOT/adaptive/boundary_config.json"
```

The preparation command authenticates both sealed archives, including control
SHA256 `2403a8c2f786b6f6f1cdfff3f846535fa43c494b970139b3ac4c28fbec8c7340`.
Keep the preceding workflow's package freeze, source tar, nvidia snapshots,
instrumentation install, Adaptive launch, stop-before-restore and failure logs.
Replace its `capture-control` command with `capture-attention`, and its trap's
`analyze-control` with `analyze-attention`. Do not export the old k=0 gate anywhere
in that workflow. Unset `M203_P177_ATTENTION_SUBSTITUTION` after restoration.
Never continue after stop, source authentication, capture or restoration fails.
Do not retry or reuse a run root. Analyze once and seal once, preserving failures.

Read `p177_layer0_attention_substitution.json` and its raw events. Native/shadow
FP16 bytes (hex and SHA256), live Q/cache identities, original M=4 input binding,
producer/draft lineage, unchanged tail bytes and layer1 postwrite hashes are
required. The analyzer also requires the authenticated layer1 prefix 0..176.

* `LAYER0_ATTENTION_PATH_CAUSAL`: native/shadow differ and substituted P177 K/V
  both match sealed Vanilla/control while normal Adaptive differs.
* `ATTENTION_SUBSTITUTION_INSUFFICIENT`: all gates pass but substituted P177 K/V
  do not both match Vanilla. Do not promote Candidate A as sufficient.
* `INVALID_EVIDENCE`: any missing, duplicate, inconsistent or unauthenticated gate;
  this includes equal native/shadow with an apparently restored result.

All results remain non-canonical, `task_done=false`,
`same_kv_cache_state_proven=false`, `historical_kv_state_equivalent=false`,
`interpretation_status=NOT_YET_APPROVED`, `temporal_kv_onset=UNKNOWN`.
