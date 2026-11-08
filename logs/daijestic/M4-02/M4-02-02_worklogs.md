# M4-02-02 GPU cost profiling

## Scope

Preserve the M4 measurement pipeline using the existing Modal app, pinned image
and model/cache/artifact volumes. Following the user's reconciliation instruction,
only a one-cell smoke is authorized. Pilot/full campaigns and canonical profile
generation are disabled pending technical alignment. No Trello or M3 evidence changes.

## Source commit

Audited base: `f7253125361635d6e5184b76c39ea248f2323fc0`.
Branch: `feat/m4-02-02-gpu-cost-profiler`. Initial checkout was clean.
Each submission requires a clean committed checkout and records its immutable
snapshot SHA. Source commits for actual runs are recorded in submission/env files.

## Research question

How do controlled target-forward and EAGLE proposal costs vary with runtime batch
size on L4/T4? What part of canonical tree costs can be supported by these data?
This experiment measures costs; it does not establish application speedup.

## Theory used

- Leviathan's linear i.i.d. model: expected yield is `sum(alpha**i, i=0..K)`.
- OPT-Tree Eq. (2): approximate expected acceptance is the sum of root-to-node
  products of draft probabilities, including root contribution 1. A scalar alpha
  and layer widths do not in general identify these probabilities.
- Approved TreeAdaptive scalar-alpha approximation contract:
  `P_step(l,a)=1-(1-a)**n_l`, `P_path(l,a)=product_{j<=l} P_step(j,a)`,
  `E[tau(T,a)]=1+sum_l P_path(l,a)`. This project contract is intentionally
  distinguished from the exact node-probability identity and remains unchanged.
- Verification formula conflict: TreeAdaptive uses `CT(B)*(1+gamma*N/64)`;
  FlexCostModel uses `CT(B)*(1+factor*max(0,N-1))`. No canonical gamma is fitted.

Primary sources:
[OPT-Tree](https://arxiv.org/html/2406.17276v3#S3),
[linear speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html).
No acceptance formula is used to manufacture timing evidence.

## Runtime audit

Repository searches of Python sources found `tree_mask`, `parent_indices`, and
`nodes_at_depth` in the topology representation/validator, with canonical lookup
in the cost model. No serving/proposer code consumes these canonical arrays.
T4 is a represented linear chain of four speculative nodes; T8/T16/T32 branch.
The EAGLE lane executes ordinary linear `method=eagle3`, K=3, greedy sampling.
Calibration deliberately varies K=1..5 and records `linear_eagle_kK` labels.

Pinned vLLM 0.30.0 source was retrieved and inspected:

- [EagleProposer](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/eagle.py)
  derives from SpecDecodeBaseProposer.
- [proposer](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/llm_base_proposer.py):
  initial forward, then `range(num_speculative_tokens-1)` additional forwards.
- [GPU runner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_model_runner.py):
  target forward, sampling, and proposer operations have distinct boundaries.
- [scheduler output](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/output.py):
  per-request scheduled tokens identify actual query workload. Preemption field
  may be None in the v1 runner; exact shapes/context remain mandatory.
- [LLM collective_rpc](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/llm.py)
  accepts a callable whose first argument is the worker.

No arbitrary-tree engine was added. `runtime_tree_available=false` is mandatory.
M3 used `(T(1+64)-T(1))/64`, five repeats, three discarded warmups and a separate
Qwen 0.5B draft. Its cost generator modeled verification with an inherited gamma
and multiplied draft cost by K. Those draft measurements are not EAGLE evidence.

## Measurement protocol

Controlled context: 1024 tokens, a member of the previous M3 context grid.
Use M3's deterministic distinct prompts. Prefix caching and chunked prefill off.
Target max length 4096, max sequences 32, token budget 4096, GPU memory fraction .85.
Eager execution, compilation mode 0, synchronous scheduling, V1 model runner
selected explicitly with `VLLM_USE_V2_MODEL_RUNNER=0`. This deliberately
controlled protocol differs from graph-enabled production serving; do not mix it
with historical timings. Runtime source file hashes are captured by the worker.

Each repeat prefills a fresh batch, then observes the first decode cycle at
exactly context L. Target-forward samples require B actual requests, each with
K+1 scheduled query positions (1 at baseline). Record all observed shapes;
missing exact shapes fail rather than substituting queued/partial batches.
This also rejects cells that fail to admit the whole batch due to KV capacity.
Warmups initialize the same operation at each B/K: at least 5 retained warmup
repeats, followed by at least 50 independent post-warmup repeats in full runs.

Timing occurs **inside the CUDA worker**, with synchronize before/after the
operation and perf_counter wall time. Target boundary is model forward, including
attention but excluding logits and sampling. EAGLE boundary is the real `propose`
call including input preparation and draft sampling. Target and EAGLE inputs are
generated by the pinned runtime, not standalone synthetic hidden tensors.
These are instrumented component costs, not complete serving-cycle wall times.
No application speedup conclusion is supported by these boundaries.

Authorized smoke: one B/K cell, 5 warmups + 2 repeats; not canonical. K1 loads
both target and real EAGLE checkpoint. Submission rejects pilot/full stages,
multiple B/K cells, and more than five post-warmup repeats. Full-evidence parsing
still validates >=50 repeats for future data inspection; it does not authorize
running a campaign. Use single-use containers. No automatic retry of failed CUDA work.
NVML name and CUDA architecture must match requested GPU. Failures remain in
raw records and/or run-level environment metadata; partial volume evidence is
retained even when the caller disconnects. Submission stores the Modal call ID.

## Modal L4 environment

Required: NVIDIA L4 / sm_89, bfloat16, awq_marlin, vLLM 0.30.0.
Actual driver/CUDA/PyTorch/VRAM identities belong in each run's env.json.
No hardware measurement claimed before successful smoke/campaign evidence.

## Modal T4 environment

Required: Tesla T4 / sm_75, half, awq, vLLM 0.30.0.
No silent hardware/dtype fallback. Any OOM/timeout or invalid batch is a failed
cell and prevents publication of a canonical profile.

## Raw artifact locations

Local: `experiments/runs/m4_cost_profiling/<unique-run-id>/`.
Remote: `ptit-run-artifacts:/m4_cost_profiling/<unique-run-id>/kK/`.
Each K retains env.json and raw.jsonl. Successful collection retains merged
env.json, raw.jsonl, summary.json and submission/call IDs. Never overwrite runs.

## Aggregation method

Per-cell median of all successful post-warmup repeats at one fixed context.
Keep min/max/p90, range/median variance ratio and warning above .15.
No post-hoc outlier removal. Any failure or missing repeat blocks a complete
component report. No aggregation across contexts. Preserve optional OLS
intercept/slope/RMSE for linear EAGLE proposal K calibration when at least two K
values exist. Do not map this regression to topology depth/width or create tree
cost estimates. One-cell smoke cannot estimate slope. Gamma fitting is withheld
until its normalization, formula and measurement boundary are aligned.

## Profile schema

Current output is `m4-component-calibration-v1`, `artifact_kind=component_report`,
`not_consumer_profile=true`. It has measured component summaries, provenance and
optional linear proposal diagnostics. It emits neither `cost_matrix_ms` nor
canonical `target_c_t`/`draft_c_d`/gamma aliases. K1 proposer call is not renamed
to a standalone draft forward. The old `build_profile` API fails closed.
The CLI retains its filename but refuses canonical profile destinations/names.

M4-02-03 consumer remains byte-for-byte unchanged from the base:
`from_json()` marks loaded profiles measured, so derived arbitrary-tree matrices
cannot be published through it without an explicitly approved contract extension.
Canonical filenames `profile_sm89_l4.json` / `profile_sm75_t4.json` and their
historical contents remain unchanged. Exact matrix lookup is tested with an
in-memory synthetic compatibility fixture, never written as GPU evidence.

## Validation

Initial local suite: 120 passed (M4, tree topology, cost solver, M3 profiler,
Modal runner configuration). Further observer/remote validation recorded below.
Synthetic fixtures in tests are never written as experiment evidence.

## Known limitations

There are no newly generated canonical topology cells. T8/T16/T32 cannot be
direct runtime measurements without an executable arbitrary-tree path. Component
boundaries omit target sampling and other cycle/scheduler work; they cannot be
represented as total arbitrary-tree latency or used to establish measured speedup.

## Technical review findings

Correction to the initial audit: the approved TreeAdaptive specification §4.1
explicitly defines `branch_aware` as the **project scalar-alpha approximation
contract**. The initial worklog overstated this as an unapproved mismatch. It is
not the exact OPT-Tree node-probability identity, but is an approved controller
approximation. No acceptance, solver, infeasibility or fallback behavior is changed.

**TEAM TECHNICAL ALIGNMENT REQUIRED — verification coefficient normalization.**

| Evidence | Formula / contract |
| --- | --- |
| TreeAdaptive spec §4.4 | `CV=CT*(1+gamma*N/64)` |
| FlexCostModel spec §3.3 and plan step 1 | `CV=CT*(1+factor*max(0,N-1))` |
| Current fallback and M3 generator | Same per-token formula as FlexCostModel |
| Existing JSON profiles | `attn_verify_factor=.008` L4 / `.012` T4 |
| M4-02-03 worklog | JSON profiles load with provenance `measured`; exact matrix lookup |

Git history: TreeAdaptive math update `f7b15fd` (2026-11-06), topology `fe3a34b`
(2026-11-07), FlexCostModel implementation/spec `8125b2c` (2026-11-07), checklist
`2aaacd7` (2026-11-07). The newer FlexCostModel documents explicitly specify the
existing fallback and its per-token coefficient. They do not explicitly amend
TreeAdaptive §4.4 or resolve cross-spec canonical calibration/provenance semantics.
M4-02-03 worklog calls its specification Revision 1.1 while the spec header still
says Revision 1.0; use actual formula text rather than infer precedence from labels.

For the same coefficient .008: N=4 gives A=1.0005*CT versus B=1.024*CT;
N=32 gives A=1.004*CT versus B=1.248*CT. Equivalence requires
`gamma_A=64*factor_B*(N-1)/N` for N>0, which depends on N. There is no single
constant alias across all N (N=1 also has different zero/intercept behavior).
Consequences: fitted gamma values are not interchangeable; fallback predictions
and ratios change, generally reducing predicted speedup and increasing alpha*
under B for the same positive coefficient, possibly crossing the infeasibility
bound. Exact measured matrix lookups bypass fallback, but generated modeled
cells and fallback consistency still require agreement. Do not silently change
either shared fallback or coefficient normalization.

Required alignment: choose/document the verification formula and N convention,
measurement boundary, acceptable calibration domain, and consumer handling of
measured versus derived costs. Until then, no gamma fit or canonical profile.

## Reconciliation of the initial seven-file diff

Audited HEAD `55e219c454e1f5b7f93bd72ac859220a95a43e5e` against required base,
including the pending five-line runner fix and all current artifact files.

| File / major hunks | Initial +/− | Disposition |
| --- | --- | --- |
| m4_cost_profiler: config, timing, summaries, evidence validation | +240/0 file total | KEEP; validate declared grid, retain >=50 validation for full evidence |
| m4_cost_profiler: gamma fit, depth proxy, topology matrix | included above | REVERT automatic modeling; MODIFY to component-only report and fail-closed canonical builder |
| m4_modal_runner: inherited image/volumes, source gate, env/raw/call IDs | +170/0 | KEEP; MODIFY to named worker RPC and one-cell-only authorization gate |
| m4_worker_observer: hooks, shape checks, synchronizations | +101/0 | KEEP; MODIFY with explicit V1 check and named extension methods |
| build_m4_cost_profiles CLI: exclusive writes and input parsing | +31/0 | KEEP; MODIFY output to component reports; reject canonical destinations |
| test_m4_cost_profiler: evidence/timing/hardware/observer tests | +182/0 | KEEP; MODIFY modeled-profile tests to component reports and publication/submission guards |
| worklog: provenance, protocol, raw paths, reproduction | +214/0 | KEEP structure; MODIFY acceptance/cost-contract wording, limits, commands and actual failures |
| cost_model.py: metadata-controlled loader provenance | +1/−1 | REVERT to approved consumer; no remaining diff against base |

No sqrt(width) model, arbitrary-tree runtime, fake tree measurement or solver
theory modification was introduced. Prior depth proxy was explicitly modeled,
but is removed because it cannot satisfy the approved consumer contract.

Modal 1.6.2 installed signatures and [official Image documentation](https://modal.com/docs/reference/modal.Image)
confirm `copy=False` mounts at container startup and require `copy=True` if later
image build steps are necessary. Keeping runtime variables in function `env`
avoids the need for any additional image layer and preserves timing semantics.

## Reproduction commands

From repository root, PowerShell (unique run IDs are mandatory):

```powershell
python -m pytest -q tests/test_m4_cost_profiler.py tests/test_tree_topology.py tests/test_cost_model_solver.py tests/test_modal_profiler.py tests/test_modal_runner_config.py
$env:PTIT_GPU='L4'
$env:PTIT_DTYPE_PROFILE='l4-bf16'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
modal run -m experiments.m4_modal_runner::main --run-id <unique-l4-one-cell-smoke> --stage smoke --batches 1 --ks 1 --warmup 5 --repeats 2
# Component inspection only; this does not build a consumer profile:
python -m scripts.build_m4_cost_profiles experiments/runs/m4_cost_profiling/<smoke-run-id> --output experiments/runs/m4_cost_profiling/<smoke-run-id>/component_report.json
modal volume get ptit-run-artifacts /m4_cost_profiling/<run-id> <new-recovery-directory>
```

## Results

Local validation after observer shape tests: 125 tests (28 M4 tests).

Initial submission attempts (source `dcd0382da06d20fe1c8c77e8a0c7f91db981856c`):

- `20261108-m40202-l4-smoke-01`: Windows cp1252 console failed to encode U+2713;
  no evidence folder and no benchmark submission. Set PYTHONIOENCODING=utf-8 and
  PYTHONUTF8=1 for subsequent commands.
- `20261108-m40202-l4-smoke-02`: Modal app
  `ap-gjvxWR2lV316siRxi0nFca`; image build failed before main/measurement:
  `InvalidError: An image tried to run a build step after using image.add_local_*`.
  Move GPU/PYTHONPATH overrides to the function environment, keep inherited image
  mounts last, use Modal's `single_use_containers=True` replacement for max_inputs.

No timing measurements or replacement profiles resulted from either attempt.
These attempts produced **NO MEASUREMENT** and are excluded from canonical results.

`20261108-m40202-l4-smoke-03`, source `55e219c454e1f5b7f93bd72ac859220a95a43e5e`,
app `ap-Bpvu9zBfTS4WdMDA3ibmnx`, call `fc-01M6WH93QX678ESR9SM4N88GJS`:
NVML confirmed NVIDIA L4/sm_89, 23034 MB, driver 610.57.04, CUDA 13.0,
torch 2.13.0+cu130, vLLM 0.30.0. Target loaded successfully, but observer RPC
installation failed: `TypeError: Object of type <class 'function'> is not
serializable ... VLLM_ALLOW_INSECURE_SERIALIZATION=1 ...`. No enabling of insecure
serialization was made. Use the supported named worker extension instead.
Local/volume `k0/env.json` retains the error; `k0/raw.jsonl` is zero bytes.
**NO MEASUREMENT**. K1 was not submitted. Default runner selected V2; the observer
now explicitly selects V1 to match the audited source. Initialization also
warned that disabling chunked prefill is not officially supported for the model;
retain this diagnostic and do not interpret this failed startup as validation
of output correctness or timing.

Reconciliation local validation: **147 passed**, including unchanged tree,
solver, M3 profiler, Modal configuration, baseline and L4 cost tests.
Canonical GPU profile files remain unchanged. Full campaigns remain prohibited.
