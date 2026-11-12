# Offline closure of independent runtime evidence audit

Closure: 2026-11-12 12:26 +07:00 (Asia/Bangkok).
Technical status: **READY FOR PM HANDOFF**. R1, R2 and R3 are resolved within
this offline closure scope. No new GPU run or runtime implementation change.

This successor note corrects/supplements the sealed README and historical worklog.
The original README, original validation.json and all other sealed bytes remain
unchanged. The old README's validator invocation omits the now-required
`--no-write`; use the safe command below. Original seven-gate results describe
the historical validator; the hardened validator has 13 gates.

## Revision and acquisition provenance (R2)

| Role | Revision |
| --- | --- |
| Reviewed pre-runtime implementation | f889865f82bb91681d436e80570815490b179da5 |
| Actual runtime-tested source | fb4544166c428e73b164756a2ef0e11ad6b97d6f |
| Independent evidence-audit / starting HEAD | d023e3f79c896f8e0044ef37bdbeabe2edc5968a |
| Original validator introduced | 6400bcd1be0a4a2f699dd6aabad295bf42d6ac1d |
| Hardened validator / closure implementation HEAD | cb4d28282fbe40ff1f3cfe71449638013aa1dd29 |
| Final closure HEAD | The documentation commit containing this note; resolve as below |

Final closure HEAD is recorded by Git, avoiding a self-referential commit hash
inside its own content. At handoff `git rev-parse HEAD` resolves it; the stable
historical locator is:

```text
git log -1 --format=%H -- experiments/runs/m4_02_04/20261112T014441Z-async-lineage-pinned-runtime/CLOSURE.md
```

The final handoff report supplies that literal full SHA. This final documentation
commit is a runtime-source-equivalent descendant, not another runtime-tested SHA.
The hardened validator source SHA-256 is
`ef7c10c37e9986825f42e5b113c4b5bb7cfd819b6d2c52306530a4cfc85e461f`.

The preserved manifest records aggregate retrieval via `modal volume get` from
Modal volume `ptit-run-artifacts`, remote path
`lane3/20261112T014441Z-async-lineage-pinned-runtime-fb454416-a10g/telemetry.jsonl`,
to `runtime-04/telemetry.jsonl`. It retrieved an already-generated artifact after
the initial collection gap; it did not rerun the runtime.

- Exact retrieval timestamp: **NOT RETAINED**.
- Retrieval time: **APPROXIMATE / BOUNDED BY** aggregate `finished_at`
  2026-11-12T02:10:43.251384+00:00 and first evidence commit
  `97bf088c889bd84703d8835bbb2b32ef6b7ea6db`, whose recorded committer time is
  2026-11-12T09:17:29+07:00 (02:17:29 UTC). This is a record-based bound,
  not an independently authenticated transfer timestamp.
- Source-side digest: **NOT RETAINED**.
- Transfer receipt / download console transcript: **NOT RETAINED**.
- Local acquired aggregate SHA-256:
  `e9b9661c1c5b08576f6150de8105287723a9c510c4125cfcf3c6ed6c85784dff`.
  This is the sealed local digest, not a claimed source-side digest.
- Initial collection-gap failure remains preserved in
  `runtime-04/validation-initial-collection-gap.json`.

Requested Modal resource class: **A10G**.
Runtime-observed GPU device string: **NVIDIA A10**, 23028 MiB.
Driver 610.57.04, CUDA 13.0, Python 3.11.5, PyTorch 2.13.0+cu130,
vLLM 0.30.0. The runtime did not report an NVIDIA A10G device string.
Target Qwen2.5-7B-Instruct-AWQ revision
`b25037543e9394b818fdfca67ab2a00ecc7dd641`; draft Qwen2.5-0.5B-Instruct
revision `7ae557604adf67be50417f59c2c2f167def9a775`.

Historical captured source: **392 files matched the historical capture manifest**.
Actual runtime source gate: **15 production/version-sensitive files checked**.
These are different scopes. Some bootstrap dependencies are outside the 15-file
runtime hash gate; it does not establish full environment reproducibility.

## Validator closure (R1)

The validator consumes raw JSON/JSONL, runtime logs, source verification records
and the sealed hash manifest. README/worklog conclusions are not correctness
inputs. README bytes are hashed only for immutable evidence integrity.

The committed seal index is itself pinned to SHA-256
`8f41e7606244436e2ee6f6aa4b4c9ba4bd1fd484871ea8a180b652117d33bb60`.
All 37 entries must match, including parent logs and metadata; replacing both a
file and the index cannot pass. Original working bytes and committed blobs were
independently checked against the index before/after offline execution.

Semantic checks independently reject duplicate plan/schedule identities, absent
producer records, cross-request joins, incompatible budgets, impossible row/time
ordering and compatible latest-plan redirection. This validator is specifically
for the sealed, single-request, non-preempted positive trace: positive schedules
form an ordered producer stream, and feedback consumes its prefix. It does not
infer a general `schedule = plan + 1` rule or claim to reconstruct GPU buffer
identities from timestamps. Trailing producers without feedback at request finish
are permitted. Exact source IDs remain the join keys, backed by the immutable
capture. It is not a general validator for other execution topologies.

The actual child's hook/wrapper/attempt/publication/consumption timestamps must be
strictly ordered; attempt/generation/ledger identities must match. Raw feedback
counts must be integers excluding booleans, drafted > 0, and
0 <= accepted <= drafted. Response metrics and final aggregate request/step/alpha
must agree with raw feedback. EMA uses the source expression
`0.8 * before + (1.0 - 0.8) * (accepted / drafted)` with absolute tolerance
1e-12 and no relative tolerance. Tested SHA is pinned to the actual runtime SHA.
The unchanged 15-file installed-source gate remains required.

Runtime-only log/telemetry scanning rejects missing-lineage, stale lineage,
duplicate lineage, lineage mismatch, feedback skipped, provenance failure and
the discovered production failure strings. Explanatory documentation is excluded.
Scope limits remain enforced. CLI `--no-write` is read-only; writing
`validation.json` under a seal index is explicitly refused.

All A1-A20 required mutation classes are covered by temporary-mirror tests.
Semantic tests assert the specific failing gate, so seal failure cannot mask a
missing semantic check. The suite also covers re-sealing, negative/zero/boolean
counts, strict EMA tolerance, scope claims, retained registration, parent/class
identity, skipped feedback and CLI byte preservation.

## Reproduction and independent results

Run from the repository root (safe directly on canonical evidence because this
command does not write):

```text
python -B -m experiments.validate_async_lineage_runtime experiments/runs/m4_02_04/20261112T014441Z-async-lineage-pinned-runtime/runtime-04 --expected-sha fb4544166c428e73b164756a2ef0e11ad6b97d6f --no-write
```

The command was run on an isolated complete temporary mirror: **13/13 gates
PASS**, offline_validator=PASS, exit 0. `CLOSURE-VALIDATION.json` preserves the
new offline result and independent recomputation; it is successor metadata and
is outside the original seal, not newly generated runtime evidence.

A separate inline Python recomputation, without importing the validator, parsed
raw steps, joined plan/schedule/request IDs, summed counts and applied the
controller expression: **13 events, 49 drafted, 49 accepted**, alpha
**0.8 -> 0.98900488372224**, maximum per-event difference **0**.
Plan/schedule pairs are (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8),
(8,9), (9,10), (10,11), (11,12), (12,13), (13,14). First three counts are
3/3; remaining ten are 4/4. An initial independent calculation using literal
0.2 and a different operation order differed by 3.33e-16; using the actual source
expression resolved it exactly. No evidence bytes were changed.

Child PID 32 bootstrap: hook 02:01:58.254591, wrapper 02:01:58.269039,
attempt 02:01:58.269494, executor publication 02:02:14.638870,
scheduler consumption 02:10:38.585270 (UTC, 2026-11-12).
Classes: LineageUniProcExecutor / AdaptiveAsyncScheduler.
Attempt `999c9934abe7414584845896aadd7385`, generation
`55b13c4336134c20927d83873b96bc7d`, ledger
`90617d84beb44dfcbd668bf6559ca5a1` match.

## Shutdown correction (R3)

`runtime-04/vllm.log` records API shutdown `mode=abort timeout=0s`, then
**force killing remaining process EngineCore pid 32**. EngineCore was force-killed
during abort-mode shutdown. A semaphore/resource-tracker warning was observed.
This was not clean/graceful shutdown. All task apps subsequently stopped with
zero containers, as captured in apps-final.json.

Positive startup and feedback evidence: **VALID**.
Graceful runtime shutdown: **RUNTIME UNVERIFIED / NOT VERIFIED**.
Runtime revocation during graceful shutdown: **RUNTIME UNVERIFIED / NOT VERIFIED**.
Generation-specific revocation: **CPU/source-contract verified** separately.
Force termination does not contradict the completed positive request.

## CPU regressions and review

All commands used `python -B -m pytest -p no:cacheprovider -q --tb=short`.
Every selection below has 0 skipped and 0 deselected.

| Selection | Passed | Failed |
| --- | ---: | ---: |
| tests/test_async_lineage_runtime_validator.py | 51 | 0 |
| Targeted six-file async-lineage selection | 202 | 0 |
| Reconstructed historical broad selection | 441 | 6 |

Targeted selection:

```text
tests/test_adaptive_scheduler_glue.py tests/test_lineage_startup.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_modal_runner_config.py tests/test_scalar_colab_smoke.py
```

Broader selection (same as latest historical worklog):

```text
tests/test_adaptive_scheduler_glue.py tests/test_lineage_startup.py tests/test_budget_planner.py tests/test_budget_transition_scenarios.py tests/test_budget_controller.py tests/test_budget_fallback_recovery.py tests/test_proposer_adapter.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_cost_model_l4.py tests/test_cost_model_solver.py tests/test_modal_runner_config.py tests/test_modal_smoke.py tests/test_scalar_context.py tests/test_scalar_colab_smoke.py tests/test_scalar_colab_git.py tests/test_scalar_cost_capture.py tests/test_scalar_cost_evidence.py
```

Broader regression is **NOT green**. Exactly the six known W1 CRLF/workload-hash
failures remain in test_modal_smoke.py: smoke_samples_are_fixed_and_come_from_frozen_workloads,
smoke_runs_5_prompts_per_lane_with_canonical_params, smoke_runs_eagle_lane,
smoke_records_failures_instead_of_crashing, json_valid_only_for_w1,
main_writes_evidence (all have `test_` prefix).
Actual W1 hash `19808e79c743443bf9c8b989b797becc23c9f0483282b93e2c967f416b493386`
versus expected `0de2387c4e6a8c39523bf869788a55ee7487c6a32318a949ce22e1f8f7b17360`.
Tree-controller timing passed this run; its historical sensitivity remains.
No new failure category, workload normalization or threshold change.

The initial adversarial run against the old validator had 21 failed / 18 passed,
confirming blind spots and absent new gates/CLI protection. Final full validator
runs are green; these intermediate test failures are not runtime failures.
Self-review covered complete validator/test diff, seal trust, source joins,
producer order scope, logs, CLI safety, documentation and all changed paths.
`git diff --check` passes. Sealed originals remain **37/37 unchanged**.

## Exact claim and technical handoff ceiling

At tested SHA `fb4544166c428e73b164756a2ef0e11ad6b97d6f`, async feedback lineage
and positive startup ownership were verified on the tested pinned vLLM 0.30.0
V1 / UniProc / single-worker NVIDIA A10 runtime configuration. Canonical request
`chatcmpl-863a80d9a3f74fb9` produced 13 verification events, 49 drafted speculative
tokens and 49 accepted speculative tokens, with exact producer source-plan/source-
schedule attribution, zero observed missing-lineage events, and controller alpha
from 0.8 to 0.98900488372224. These counts do not describe emitted tokens or a
general acceptance rate.

Invalid scheduler/executor pair, failed-startup recovery and K=0 sanity remain
**RUNTIME UNVERIFIED**; existing CPU/source-contract evidence stays separate.
Graceful shutdown and graceful-shutdown revocation remain **RUNTIME UNVERIFIED**.

NO profitability claim; NO adaptive-vs-vanilla speedup claim; NO generalized 100%
acceptance claim; NO distributed/multiworker lineage claim; NO arbitrary
TreeAdaptive runtime claim; NO hardware-aware measured-cost policy claim;
NO graceful shutdown runtime verification; NO invalid-pair runtime verification;
NO failed-startup recovery runtime verification; NO K=0 runtime verification.

All paths changed since tested SHA are validator, validator tests, evidence
metadata/bytes, documentation or worklog. No runtime correctness source changed;
final HEAD inherits runtime-source equivalence, not literal execution at its SHA.
Branch remains feat/m4-02-04-async-lineage-v02. Local closure commits only;
develop and future/tree-adaptive-v02 untouched. Three pre-existing untracked
evidence directories preserved. Nothing pushed/merged; no PR or Trello mutation.
PM owns review and task state. Next action: PM/reviewer reviews this closure.
