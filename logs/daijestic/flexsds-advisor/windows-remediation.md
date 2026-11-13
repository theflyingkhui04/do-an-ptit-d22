# Windows installed-console audit and remediation

Date: 2026-11-13 (Asia/Bangkok). Final status: READY FOR REVIEW.
Scope: CLI presentation/output compatibility and installed-console integration;
no serving algorithm, public command, scheduler, cost model or protocol changes.

## Trigger, baseline and read-only findings

The human reviewer installed the editable package and found that `flexsds --version`
worked but detect and local serve --dry-run failed at Unicode banner output. This
audit supersedes the sufficiency of the first READY FOR REVIEW gate: the earlier
work used module invocation with forced UTF-8 and did not prove default console
output. Historical worklog, CLI captures and failed test evidence are retained.

Fetched origin/develop: `48ae3a166ad9e42fdb81d3d401e496efef3b71bd`.
Starting HEAD: `375143acef4afc9878f3070db6ebf3e420f4c57d`.
Branch: feat/flexsds-advisor-skill; starting tracked/untracked state was clean.
The existing isolated advisor worktree was reused. Worktree inspection confirmed
M4-02-06 remains separately checked out; no writes/staging targeted it.

Before editing, inspected pyproject, main/banner/wizard, hardware/compatibility/
cost-profile paths, tests, skill files, prior evidence and requested specification,
plan/product/README material. Git diffs for product sources and pyproject were
empty at the start; skill/test changes were confined to the prior skill commits.

| Audit question | Finding before edits |
|---|---|
| Q1 banner modified by skill branch? | No; baseline and starting HEAD blob both 5350b991d9ad9c216ac74d01affa9192f43a3ca2 |
| Q2 main modified by skill branch? | No; both blobs 65c933c2623e68fa8b2ac17ed09dc23a8e86a448 |
| Q3 baseline-identical code reproduces crash? | Yes; installed entrypoint points to this editable checkout and unchanged sources reproduce U+2584 failure |
| Q4 explicit Windows/ASCII requirement? | Yes; spec.md section 5, item 2, line 164 |
| Q5 product, skill or environment? | PRE-EXISTING PRODUCT BUG exposed by cp1252 pipe output; skill tests had an integration coverage gap |
| Q6 installed-path claim in skill? | No: SKILL.md describes invocation, not prior execution evidence; old worklog explicitly says module fallback was used. Old CLI reference's UTF-8 workaround is now replaced |
| Q7 existing fallback coverage? | None found: existing banner test only checks render_banner text; advisor subprocess tests force UTF-8 |

`windows-environment.json` preserves blob attribution, platform and environment.

## Exact requirement

Source: logs/Khoi/spec&plan/flexsds-cli/spec.md:164:

> **Độ tương thích terminal rộng:** Hiển thị tốt trên cả terminal Linux/macOS và Windows Terminal, tự động hạ cấp xuống ký tự ASCII đơn giản nếu terminal không hỗ trợ màu TrueColor hoặc UTF-8.

The unconditional Unicode print violated that requirement. Automatic fallback is
already specified; this patch adds no command/flag or platform support commitment.

## Reproduction and causal evidence

Host: Windows-10-10.0.26300-SP0; PowerShell 7.6.9; Python 3.11.0.
stdout.encoding=cp1252, stdout.isatty=False; PYTHONUTF8/PYTHONIOENCODING unset;
TERM=dumb, COLORTERM empty, NO_COLOR=1, WT_SESSION unset.
This run used PowerShell, not the reviewer's Git Bash; the same failing codec and
call path were independently reproduced. No live interactive wizard was run.

Resolved public executable: D:/ProgramFiles/Python3_11/Scripts/flexsds.EXE.
`python -m pip show flexsds` confirms an editable installation at this worktree.
No reinstall was needed: only imported Python sources changed; pyproject and
console entrypoint definition were unchanged.

Exact public command matrix (the JSON records include absolute argv):

```text
flexsds --version
flexsds detect
flexsds profile
flexsds serve --target Qwen/Qwen2.5-7B-Instruct-AWQ --draft Qwen/Qwen2.5-0.5B-Instruct --backend local --dry-run
```

| Output mode | Before: version/detect/profile/dry-run | After |
|---|---|---|
| No encoding override | 0 / 1 / 1 / 1 | 0 / 0 / 0 / 0 |
| PYTHONUTF8=1 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| PYTHONIOENCODING=utf-8 (diagnostic only) | 0 / 0 / 0 / 0 | Not repeated; no production dependency |
| PYTHONUTF8=0, PYTHONIOENCODING=cp1252 | 0 / 1 / 1 / 1 | 0 / 0 / 0 / 0 |

Environment injection used subprocess env mappings, the PowerShell/platform-safe
equivalent of a shell `PYTHONUTF8=1 flexsds ...` prefix. No persistent env setting
was changed. Each command closed stdin, captured stdout/stderr and exit code and
had a 45-second timeout.

Root cause location: banner.py print_banner -> print(render_banner(hw_info)).
The banner contains U+2584, which cp1252 cannot encode; print fails before command
dispatch. Follow-on status checkmarks are also unencodable, so banner-only changes
would leave detect/profile/serve broken. `windows-console-before.json` captures the
actual tracebacks. `windows-minimal-repro.json` proves render_banner(...).encode(
'cp1252') fails on U+2584 both with and without ANSI sequences: escape codes are
not the encoding cause. Tests cover TTY-like and non-TTY streams separately.

## Narrow patch

- src/flexsds/cli/output.py: presentation helper, conservative codec/capability
  inspection, ANSI removal when color is unsupported, ASCII status replacements
  and escaping of remaining Unicode. Only UnicodeEncodeError triggers a write
  retry; unrelated I/O errors propagate. No stdout.reconfigure/global mutation.
- banner.py: retains render_banner unchanged; adds compact ASCII PTIT/FlexSDS,
  hardware, engine-label and backend information. Rich Unicode/TrueColor banner
  remains byte-for-byte identical on a capable stream. Unknown capabilities,
  pipes and NO_COLOR use plain ASCII branding. The inherited TreeAdaptive label
  remains presentation, not new runtime verification evidence.
- main.py and local/modal_runner.py: use the helper for existing status/command
  printing, including Unicode model/path output. Parser, configs, command building,
  process launch/stop and dry-run branches are otherwise unchanged.
- wizard.py: routes status and prompt writes through the same helper, preserving
  explicit end="" cursor/newline behavior, inputs, choices and actions.
- tests/test_cli_output_encoding.py: 18 deterministic presentation tests.
- Skill references/flexsds-cli.md: narrowly replaces the old UTF-8 workaround with
  observed automatic fallback and explicit scope limits. SKILL.md, openai.yaml
  and claim-boundaries.md remain unchanged; no routing/scope expansion.

Self-review found the same unsafe direct writes in wizard prompts, so a second
local fix applies the shared helper there too. Three additional tests exercise
choice confirmation, a text prompt and detection-only output without serving.
They fail on exported baseline code (windows-wizard-red.json: 3 failed) and pass
with the patch. Full end-to-end interactive serving/terminal navigation was not
run; the skill still forbids automating the wizard. Real local/background/stop
and cloud deployment limitations remain documented; no such action was run.
AST comparison against baseline confirms main/wizard/runner code is unchanged
except presentation imports/write calls, and render_banner is unchanged. See
windows-presentation-scope.json. The first comparison attempt retained an empty
helper-import try block after normalization and failed; removing that purely
import scaffold made the scope comparison pass, without changing product code.

## Test-first evidence and current regression status

Before the patch, the new tests yielded 13 failed / 2 passed. Failures include
cp1252, missing/unknown codecs, no-TrueColor fallback and all three command paths.
See windows-encoding-red.txt (failed attempts retained).

The initial focused run passed 44 checks (15 new plus baseline/advisor). After
the wizard presentation extension, the final focused declared suite is fully green:

```text
python -m pytest -q tests/test_cli_output_encoding.py tests/test_flexsds_baseline.py tests/test_flexsds_advisor.py
47 passed
```

Coverage: UTF-8 rich-banner equality, ASCII/cp1252 fallback, missing/None/unknown
encoding metadata, incorrect UTF-8 metadata over an ASCII sink, unknown streams,
pipes, NO_COLOR/TERM=dumb/no-TrueColor, detect/profile/dry-run completion and
BrokenPipeError propagation, plus the three isolated wizard output cases. A failing Popen stand-in proves no server can start
in the encoding dry-run tests; existing advisor tests additionally check both
local and Modal dry-run non-launch behavior.

The broader declared selection is NOT green:

```text
python -m pytest -q tests/test_cli_output_encoding.py tests/test_flexsds_advisor.py tests/test_flexsds_baseline.py tests/test_modal_runner_cli.py tests/test_compatibility_guard.py tests/test_cost_model_solver.py tests/test_cost_model_l4.py tests/test_benchmark_profiles.py tests/test_local_runner.py tests/test_colab_runner.py tests/test_tree_adaptive_controller.py tests/test_tree_topology.py tests/test_budget_controller.py
144 passed, 2 failed, 0 skipped
```

Separate `python -m pytest -q tests/test_cli_wizard_flow.py`: 1 collection error.
See windows-regression-final.txt and windows-wizard-collection.txt. The earlier
141-pass/2-failure broad run is preserved in windows-regression.txt. The 18 new tests
and all 21 advisor checks pass. No tests were deleted, weakened or skipped.

## Individually audited previous failures

| Exact item | Classification (one per item) | Evidence |
|---|---|---|
| tests/test_cli_wizard_flow.py import PLATFORM_CHOICES | PRE-EXISTING | ImportError in feature and isolated origin/develop export; test blob unchanged; wizard changes only presentation, not the missing symbol |
| tests/test_local_runner.py::test_local_runner_stop_process_group | ENVIRONMENT-SPECIFIC | Feature fails at open('/tmp/test_flexsds.pid') with PermissionError on drive D; exported baseline on drive C fails at the same open with FileNotFoundError. The POSIX-style path resolves against the current Windows drive root, outside the temporary/workspace directory; Windows also lacks os.killpg used later by the test |
| tests/test_tree_adaptive_controller.py::test_decision_latency_benchmark | FLAKY / NON-DETERMINISTIC | Current focused run passes; broad runs fail at 61.770 and 72.306 us against 50 us. Isolated baseline passes 3 runs; historical unchanged-code combined run failed at 62.897 us. No latency claim/fix follows |

Baseline reproduction used `git archive origin/develop src` and the three exact
test files, extracted to an isolated temporary directory, with PYTHONPATH pointing
at its src/root. Baseline tests run in separate Python processes, independently
of the editable package's sources. Their full results are in
windows-baseline-failures.json. Current individual results are in
windows-previous-failures.txt. No unrelated production/test fix was attempted.

## Console revalidation, failed harness attempt and skill validation

windows-console-after.json contains all 12 successful public-executable records.
After the wizard output extension, all four commands were repeated once more
without encoding overrides: windows-console-final.json, four exit-0 records.
Observed GPU remains RTX 3050 Ti/sm_86, ~4 GB total. Unknown-SM L4 profile fallback
and the ~9 GB-needed / ~3.2 GB-free dry-run warning remain visible. Zero exit is
not a hardware readiness claim. The real command matrix's temporary CWD remained
empty (no config, PID or serving logs). No GPU workload or cloud allocation ran.

One post-patch capture harness initially decoded every mode as ASCII and raised
UnicodeDecodeError on valid UTF-8 status text after default commands had passed.
This was a harness error, not a CLI failure. It was corrected to decode each
mode's actual encoding, save each record incrementally and repeat the complete
matrix. The failed attempt is disclosed here; before/after CLI failures remain
unmodified in their JSON capture files.

Installed quick_validate.py: Skill is valid! Reference links, frontmatter limited
to name/description, metadata agreement and exactly four skill files were checked.
The updated ZIP is dist/flexsds-advisor-windows.zip, not committed; original ZIP
is preserved. CRC and archived-byte equality checks pass. SHA256:
`b7435dd18c18a80c9064f116f6cea8968d07217203f75fc45b8809ecb6263d99`.
See windows-skill-validation.json. Captured pytest text trailing whitespace is
trimmed for Git hygiene; tracebacks/results are preserved.

## Completion audit and review findings

| Gate | Authoritative evidence |
|---|---|
| Fresh baseline/feature isolation/source attribution | Fetch/status/log/worktree inspection; windows-environment.json blob identities |
| Original bug, root cause, requirement | Before matrix, minimal encoding repro, exact spec:164 quote |
| Narrow patch, UTF-8 preserved, ASCII fallback | Source diff; test-first failures then 47 focused passes |
| Real entrypoint version/detect/profile/dry-run | 12 post-patch executable records, all exit 0 |
| No dry-run launch | Popen prohibition tests; unchanged dry-run branches; empty actual-matrix CWD |
| Skill validation/package integrity | Validator success; windows-skill-validation.json |
| Previous failures audited | Isolated baseline and current individual/broad results; exact classifications above |
| No scope/claim expansion | SKILL.md/metadata/claim-reference unchanged; no core algorithms/config/entrypoint changes |
| Evidence/history and local commits | Historical files retained; append-only worklog pointer; fix commit below and subsequent docs commit |
| Git hygiene/external boundaries | Exact staging, staged diff review, whitespace checks and final clean status; no push/merge/PR/Trello/cloud action |

BLOCKER: none for this installed-console advisory integration.
MAJOR: existing wizard test collection error and runner limitations remain separate
follow-up concerns; the skill continues to avoid unsupported actions.
MINOR: existing environment-dependent POSIX test and unstable timing assertion.
These do not establish a fully green broad suite or cross-platform serving.

Verified claim: installed FlexSDS console inspection/profile/local preview works
under observed cp1252 output and deterministic fallback tests. Not verified or
claimed: full wizard compatibility, serving readiness, TreeAdaptive runtime,
hardware-optimal policy, profitability, adaptive speedup/no slowdown, real cloud
deployment or publication. Those remain deferred.

Fix/test commits: `c4212e9b6ffa035dcd9c4b64593aa8e199db7f0c` (banner/status)
and `c70814c8fd52dc1ac61af58212a385474fd4f15e` (wizard presentation).
The following docs commit includes this audit, the skill reference and evidence;
its exact hash is available with `git log -1 --format=%H --` this audit file.

PM handoff: Khôi, tớ đã tái hiện lỗi console cp1252 từ code giống develop và sửa
fallback ASCII ở lớp hiển thị. Bốn lệnh console thật đều pass, không cần biến môi
trường UTF-8; 18 test mới pass. Tớ giữ nguyên bằng chứng lỗi cũ và ghi rõ hai lỗi
regression cùng collection error chưa xử lý. Nhờ Khôi review bản sửa local.
