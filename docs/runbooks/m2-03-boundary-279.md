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
