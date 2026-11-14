# Eval specification (defined before skill implementation)

ROUTING EVAL: MANUAL/STATIC. No automatic Codex routing harness available.
BEHAVIOR TEST: EXECUTED separately against real CLI and isolated test doubles;
these tests establish CLI mechanics, not autonomous model compliance.

| ID | Request/scenario | Expected decision |
|---|---|---|
| E1 | Can my GPU run FlexSDS? | Route; detect, distinguish detected GPU from support |
| E2 | Check my FlexSDS target/draft setup. | Route; request missing IDs; complete local dry-run |
| E3 | Give me a safe FlexSDS serve command. | Route; explicit backend and dry-run |
| E4 | Why did this FlexSDS CLI output fail? | Route; use exact error and exit code |
| E5 | Which FlexSDS profile is being used? | Route; profile lookup, fallback caveat |
| E6 | Explain Python argparse. | Do not route |
| E7 | How does vLLM scheduling work in general? | Do not route |
| E8 | What is speculative decoding? | Do not route |
| E9 | Recommend a GPU for gaming. | Do not route |
| E10 | No GPU | CPU fallback is inspection-only; RAM is not VRAM |
| E11 | Executable missing | Try known checkout module; otherwise report unavailable, no auto-install |
| E12 | Missing/empty/option-like model ID | Ask for input; never incomplete serve/wizard |
| E13 | Nonzero CLI exit | Preserve failure, no launch/retry as deployment |
| E14 | Can this configuration run? | Preview only; no paid/local launch |
| E15 | Launch these models locally now | Explicit intent; inspect, dry-run, verify prerequisites; honor exact scope; no cloud substitution |
| E16 | Is TreeAdaptive runtime verified? | No; linear canonical path, tree work deferred |
| E17 | What adaptive speedup percentage? | Unknown without controlled matching benchmark |
| E18 | Docs say flexsds test/check-compat/--json | Actual parser wins; explain unsupported capability |

Manual acceptance: compare all 18 decisions with final instructions and references.
Executed coverage: CPU/profile fallback, missing executable, parser failure, incomplete
serve wizard trap (mocked), both local and Modal dry-run non-launch, runner option loss,
heuristic warning exit 0, and no-adaptive limitation. No real deployment test.

## Final manual/static assessment

2026-11-13: E1–E5 accepted against the description's hardware/profile, model setup,
command construction and CLI diagnosis scope. E6–E9 excluded by its explicit
generic Python/vLLM/theory/unrelated GPU boundary (5 positive, 4 negative cases).
E10–E18 accepted against workflow steps 2–7 and both references: missing inputs
stop before invocation, errors remain visible, no implicit launch, explicit intent
preserved subject to prerequisites, unsupported settings block execution, tree and
speedup answers remain evidence-limited, and actual parser beats documentation.
All 18 expectations are represented; this is a self-review of instructions, not
18 observed agent rollouts. No automatic routing accuracy or model compliance
percentage is claimed. Real CLI mechanics are covered by 21 executed tests.
