---
name: flexsds-assistant
description: Inspect FlexSDS hardware/profile selection, check target/draft serving setup, construct safe FlexSDS commands, and diagnose FlexSDS CLI errors from model IDs, environment facts or command output. Return observed results, caveats and exact next actions. Use for FlexSDS setup/advisory requests; exclude generic Python, vLLM scheduling, speculative-decoding theory and unrelated GPU advice.
---

# FlexSDS Assistant

Use the actual CLI for execution; keep reasoning and recommendations here. Support
hardware/profile inspection, serving configuration previews and CLI diagnosis.

1. Identify inspection, command construction, diagnosis, or explicitly requested
   real serving/stop action. For invocation details or unexpected output, read
   [the CLI contract](references/flexsds-cli.md). For performance, tree, support or
   optimality questions, read [claim boundaries](references/claim-boundaries.md).
2. Locate `flexsds`; inspect `--version` and `--help` when the installed contract is
   unknown. In a known source checkout, `python -m flexsds.cli.main` with that
   checkout's `src` on PYTHONPATH is the same entrypoint. If neither works, report
   the exact missing executable/import and suggest installation into the user's
   chosen environment; do not install or change credentials automatically.
   `pip show flexsds` must show the checkout as editable location; an old editable
   install (e.g. 0.2.1) makes `flexsds benchmark` fail with "No module named 'src'"
   until `pip install -e .` is rerun. PyPI 0.3.4 (yanked; it is tag v0.2.4 under a
   mistyped number) is older code (no `benchmark`, `--gpu {L4,T4,A10G}`); check
   `--help` before assuming develop flags.
3. For hardware use `flexsds detect`; for profile lookup use `flexsds profile`.
   CPU fallback means CUDA was not detected by this Python environment, not proof
   that no physical GPU exists. CPU memory is RAM, never serving VRAM. A GPU with
   no measured profile gets `conservative_analytical_fallback`, which cannot serve
   the adaptive lane; suggest `profile --benchmark` or `--no-adaptive`.
4. For configuration obtain both nonempty target and draft IDs plus backend.
   Ask for missing inputs. Reject option-like IDs and control characters; pass
   IDs as individual argv values with shell-appropriate quoting, never shell
   interpolation. Validate port 1–65535 if supplied. Do not replace user models.
   For advice with unspecified backend, explicitly label a local preview as such:

   ```text
   flexsds serve --target TARGET_ID --draft DRAFT_ID --backend local --dry-run
   ```

   Always include both IDs: bare `flexsds` or incomplete `serve`, even with
   `--dry-run`, enters the interactive wizard. Never automate that wizard.
   Capture command, exit code, stdout/stderr with a bounded timeout and closed
   stdin. Inspect warning/error text even on exit 0; do not treat it as readiness.
   Previews blocked only by the VRAM estimate or missing local CUDA print
   "Launch blocked", show the command and return 0; an unsupported or
   incompatible model pair returns 2 even with --dry-run; real unsafe serving
   returns 2 before dispatch. Compatibility is heuristic, not proof of model fit.
   Modal uses an estimated remote GPU budget, not local free memory.
5. Default to inspection/explanation/dry-run. A question such as "can this run?"
   does not authorize a launch. Real `serve`, background start, `stop`, paid GPU
   allocation (including `benchmark run` and `profile --benchmark --backend modal`),
   deployment and credential changes require explicit user intent. `benchmark stop
   --campaign-id ID` cancels only that campaign's GPU calls; without an ID it stops
   the shared Modal app and every campaign on it.
   Existing explicit intent suffices; do not ask again for routine steps. Before
   an authorized action, inspect backend prerequisites and dry-run the exact
   configuration. Stop at missing GPU/runtime, OOM warnings, unsupported hardware
   or an option the runner cannot honor. Report the narrow blocker and next fix;
   never substitute cloud or broaden the action. Read backend limitations in
   the reference before launch. Colab flag mode provides guidance only.
   `--no-adaptive` retains static speculation with the default vLLM scheduler.
   Local serving binds 0.0.0.0; verify that exposure matches the user's intent.
   For stop verify CWD, PID ownership and platform support first; do not stop an
   unrelated process or infer that local stop terminates Modal. Never collect
   secrets in chat. Check credential presence only, without printing values.
6. For `auth`, `chat` or `monitor`, read the CLI reference first. Login/logout
   changes credentials; never put secrets in argv or chat. Chat performs network
   inference; its displayed rates count SSE chunks, not tokens. Monitor tails a
   local trace until interrupted; bound observation and do not imply endpoint health.
7. On unavailable CLI, nonzero exit, timeout, or printed runner error, preserve
   the failure and explain a specific repair. Do not retry as a real launch or
   silently drop flags. Use supplied command/error text for diagnosis; reproduce
   only safe inspection/preview commands. Actual parser/runtime wins over docs;
   do not invent `test`, `check-compat`, `detect --json` or other unsupported
   options (`--json` exists only for `benchmark`).
8. Return the result, issue/limitation and exact next action. Label evidence as
   FACT (code contract), OBSERVATION (captured output), INFERENCE, RECOMMENDATION
   or UNKNOWN where relevant. A printed command is not a running endpoint.
   Never promise TreeAdaptive runtime, profitability, optimal hardware policy,
   no slowdown or adaptive speedup without matching controlled runtime evidence.
