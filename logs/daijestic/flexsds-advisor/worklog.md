# FlexSDS Advisor technical handoff

Date: 2026-11-13 (Asia/Bangkok). Status: READY FOR REVIEW.
PM request: review Agent Skills and integrate a small working skill with the
actual FlexSDS CLI. PM: Can Duc Khoi; technical owner: Bui Duc Dai / daijestic.
No Trello task ID supplied; no governance ID invented. This follows the existing
logs/daijestic technical-worklog convention without attaching unrelated M4 work.

## Baseline and isolation

- Starting worktree: D:/Document/Four-year/do_an/do-an-ptit-d22-flexsds-advisor.
- Starting branch: feat/flexsds-advisor-skill, already clean and isolated.
- Starting HEAD, fetched origin/develop and merge-base:
  `48ae3a166ad9e42fdb81d3d401e496efef3b71bd`.
- `git fetch origin --prune` succeeded after command-scoped safe.directory and
  elevated access to shared Git metadata. No global Git configuration changed.
- Existing feature branch was inspected and reused. M4-02-06 remains in its own
  do-an-ptit-d22-m4-02-02 worktree at the baseline SHA; no operations targeted it.
  No unrelated untracked files were present in this checkout; other evidence was
  not touched. No reset/clean/rebase, push, merge, PR or Trello mutation occurred.

## Sources and audit

Read pyproject.toml, __init__.py, CLI main/wizard/banner, hardware, compatibility,
cost_model/controller, local/modal runners, baseline/CLI/runner/guard/cost tests,
README, logs/Khoi/spec&plan/flexsds-cli/{spec,plan}.md and product specification
docs/09-dac-ta-san-pham-flexsds.md. Repository-wide keyword/path discovery also
found docs/runbooks/flexsds-agent-skill-happy-path.md, which exceeds actual CLI.
No AGENTS.md was found by repository file discovery.

The pre-implementation [contract table](cli-contract.md) records inputs, outputs,
failure modes and execution classes. Actual commands are detect/profile/stop/serve;
test/check-compat/--json/--adaptive and calibration are documented-only.
Compatibility is heuristic, warning-only and not the wizard's metadata guard.
Unknown SM/CPU maps to L4. --no-adaptive is parsed but ignored by local runner.
Modal drops selected model/port settings. Colab flag path is a placeholder.
These defects do not prevent safe inspection/preview; the skill blocks real
actions whose requested settings cannot be honored. No product code was changed.

Official conventions were checked against
[OpenAI skill documentation](https://learn.chatgpt.com/docs/build-skills) and the
installed C:/Users/daibu/.codex/skills/.system/skill-creator/SKILL.md plus its
openai_yaml reference. The local initializer and validator were used.

## Implementation and boundary

Agent -> skill -> existing CLI -> hardware/config/runner; no core algorithms
duplicated. No wrapper is necessary: the module executes the existing main
entrypoint and supports deterministic subprocess argv invocation.

Changed source files:

- .agents/skills/flexsds-advisor/SKILL.md: routing for FlexSDS hardware/profile,
  setup/command advice and CLI diagnosis; explicit generic-topic exclusions;
  default dry-run, complete inputs, failure capture and authorized action gates.
- agents/openai.yaml inside that skill: generated display metadata, implicit
  discovery retained by default; no unused assets or branding.
- references/flexsds-cli.md: implemented command contract and runner limitations.
- references/claim-boundaries.md: evidence labels and scientific claim ceiling.
- tests/test_flexsds_advisor.py: 21 CLI behavior/contract tests using real safe
  subprocesses and isolated doubles for unsafe/error paths.
- This evidence directory: contract/evals/worklog, captured CLI JSON, regression,
  collection-error and focused latency logs. No temporary staging text retained.

The skill entrypoint is 62 lines; frontmatter has only name and description. References
are read conditionally. All four packaged files directly support execution.

## Eval-first and verification

[evals.md](evals.md) was written before the skill. ROUTING EVAL: MANUAL/STATIC:
5 positive and 4 negative cases assessed, plus behavioral expectations E10–E18.
All 18 are represented in the final instructions; no automatic routing or model
rollout accuracy is claimed. BEHAVIOR TEST: EXECUTED against the real CLI contract;
these tests do not establish autonomous agent compliance.

Python: D:/ProgramFiles/Python3_11/python.exe (3.11). Test environment:
PYTHONPATH='src;.'; subprocess capture uses PYTHONIOENCODING=utf-8, stdin closed,
45-second timeouts and temporary working directories.

Before implementation:

- Required seven-suite baseline attempt: collection error in
  test_cli_wizard_flow.py (missing PLATFORM_CHOICES).
- Remaining six suites: 50 passed in 7.01s (baseline, Modal CLI, compatibility,
  cost solver/L4 and benchmark profiles).

After implementation:

- Same six suites plus new contract tests: 71 passed in 26.81s.
- Additional local/Colab/controller/topology/budget suites: 56 passed, 1 failed
  in 0.76s (POSIX /tmp test).
- Final combined verification: 126 passed, 2 failed, 0 skipped in 31.82s;
  separate wizard suite: 1 collection error, 0.23s. See regression.txt and
  wizard-collection.txt. All 21 new tests pass; 105 existing tests pass in that
  combined run. Do not present the overall regression as fully green.
- Focused latency recheck: 1 passed in 0.31s; see latency-recheck.txt.

Failure classification:

| Failure | Classification/evidence |
|---|---|
| Missing PLATFORM_CHOICES | PRE-EXISTING: reproduced before and after; wizard/test unchanged |
| /tmp/test_flexsds.pid PermissionError | ENVIRONMENT-SPECIFIC / OUT OF SCOPE: POSIX path and process-group test on Windows restricted sandbox; runner/test unchanged |
| Decision latency 62.897 us >= 50 us | ENVIRONMENT-SPECIFIC timing variability: passed standalone suite, failed combined run, passed focused recheck; controller/test unchanged |

`git diff BASE -- src tests/test_local_runner.py tests/test_tree_adaptive_controller.py
tests/test_cli_wizard_flow.py` is empty. No new regression identified. No production
fix was made merely to satisfy unrelated existing tests.

## Actual CLI demo

[cli-demo.json](cli-demo.json) records full exact argv, exit code and raw stdout/
stderr for module-entrypoint equivalents of these commands:

```text
python -m flexsds.cli.main --version
python -m flexsds.cli.main detect
python -m flexsds.cli.main profile
python -m flexsds.cli.main serve --target Qwen/Qwen2.5-7B-Instruct-AWQ --draft Qwen/Qwen2.5-0.5B-Instruct --backend local --dry-run
python -m flexsds.cli.main test
```

First four exit 0; final controlled unsupported command exits 2. Detection observed
RTX 3050 Ti Laptop GPU, sm_86, 4 GB total / ~3.21 GB free. Profile selected
profile_sm89_l4.json, demonstrating unknown-SM fallback. Dry-run warned ~9.0 GB
needed / 3.2 GB free and still exited 0. No launch occurred. Temporary demo CWD
remained empty. The flexsds executable was not on PATH; known-source module fallback
was used. CPU fallback and missing executable are tested with controlled fixtures.

## Failed attempts and validation

- Initial sandbox Git ownership check failed; command-scoped safe.directory fixed
  it. Fetch then hit shared-metadata permissions; elevated fetch succeeded.
- Initial skill initializer was denied under protected .agents; elevated retry
  succeeded. No automatic approval rejection occurred.
- Demo capture succeeded, but its first console summary hit cp1252 Unicode output;
  re-reading the saved records with UTF-8 output succeeded. CLI evidence itself
  was captured with UTF-8 from the start.
- Installed quick_validate.py: Skill is valid! before and after self-review.
  Installed skill-creator has no packaging script. A standard-library ZIP was
  generated under ignored dist/flexsds-advisor.zip; testzip passed and exactly four
  files were verified. SHA256:
  `8fb2aac072d14b98962b1a5207b0f497aaab58853f07e3526c0134ef80d7c224`.
  Archive is a local handoff artifact, not committed or externally published.
- Self-review checked complete-ID wizard avoidance, zero-exit warnings, ignored
  runner settings, PID ownership, credential handling, no implicit cloud launch,
  positive/negative routing scope and claim boundaries. Staged exact paths and
  full staged diff were reviewed. Captured pytest blank-line whitespace was
  trimmed for Git hygiene; failure messages/results are unchanged. Final
  diff --check and cached --check passed.

## Claim ceiling and commits

Verified: safe CLI inspection and command previews, controlled failure contracts,
skill format and package integrity. Recommendations are instruction-level advice;
model routing/compliance is only statically assessed. No serving readiness or
performance inference follows from a successful preview.

Deferred/not claimed: TreeAdaptive runtime, arbitrary trees, hardware-optimal policy,
profitability, no slowdown, adaptive-vs-vanilla speedup, telemetry diagnosis,
client refactoring, real GPU/cloud deployment, publication and cross-agent support.
Historical M4 acceptance evidence remains configuration-specific.

Local implementation commit: `2f7695d242e82b6da9f1fe0875dd362a9efe0d5c`
(feat: skill plus contract tests). This worklog and evidence are committed in the
following docs commit; obtain its exact self-referential SHA with
`git log -1 --format=%H -- logs/daijestic/flexsds-advisor/worklog.md`.

Final gate: READY FOR REVIEW for advisory v0.1, with existing regression exceptions
disclosed. Review is local only; no deployment, push, merge, PR or Trello action.

PM handoff: Khôi, tớ đã hoàn thành skill flexsds-advisor v0.1 và commit local.
Skill dùng CLI thật để kiểm tra phần cứng/profile, dry-run cấu hình và chẩn đoán lỗi.
21 test mới pass; tớ đã ghi rõ lỗi test cũ, giới hạn runner và các claim chưa được
kiểm chứng. Nhờ Khôi review branch feat/flexsds-advisor-skill.

## 2026-11-13: installed Windows console blocker and remediation

The human review exposed a cp1252 UnicodeEncodeError not covered by the original
forced-UTF-8 module demos. The historical report above is preserved; its readiness
gate did not establish the default installed console boundary. The subsequent
[Windows audit](windows-remediation.md) records the original failure, baseline
attribution, explicit product requirement, presentation-only fix, test-first
evidence and actual console-script revalidation.

Current result: READY FOR REVIEW for this narrow integration correction. The new
18-test encoding suite passes; focused baseline/advisor/encoding selection has
47 passes. Expanded regression reports 144 passes / 2 failures, plus one separate
collection error. All previous failures are individually audited against exported
baseline sources; they remain disclosed. No real serving/cloud action or claim
expansion occurred. Fix/test commit: c4212e9b6ffa035dcd9c4b64593aa8e199db7f0c.

Wizard presentation follow-up commit: c70814c8fd52dc1ac61af58212a385474fd4f15e.
