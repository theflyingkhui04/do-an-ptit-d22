# FlexSDS Advisor v0.1 contract (before skill implementation)

Baseline: origin/develop `48ae3a166ad9e42fdb81d3d401e496efef3b71bd`, fetched 2026-11-13.
PM: Can Duc Khoi. Technical owner: Bui Duc Dai / daijestic. No task ID assigned.

| Intent | Command/API boundary | Input | Observable output | Failure mode | Execution class |
|---|---|---|---|---|---|
| Hardware | `flexsds detect` | Current process environment | Human-readable hardware summary | Torch failures silently become CPU; RAM may default to 16000 MB | Headless, no credentials/GPU required |
| Profile | `flexsds profile` | Detected SM | Profile filename | CPU/unknown SM selects L4; missing file gives analytical_fallback | Headless, read-only; not calibration |
| Command preview | `flexsds serve --target ID --draft ID --backend local --dry-run` | Both nonempty model IDs, explicit backend/port | Compatibility summary and vLLM argv display | Estimates only; warnings return 0; display is not shell-escaped | Headless, no GPU/credentials required, no launch |
| Compatibility/preflight | Same complete dry-run; `check_compatibility` internally | Names and local free memory | Heuristic method/vocabulary/VRAM summary | Not remote metadata validation; CPU RAM treated as VRAM; no blocking gate | Safe preview, not proof of runtime support |
| Stop | `flexsds stop` | CWD `.flexsds.pid` | Stopped/no active server | POSIX process groups; stale PID risk; 0 not proof of VRAM release | Mutating, explicit intent and PID ownership check; not cloud stop |
| Wizard | Bare `flexsds`, or incomplete `serve` even with `--dry-run` | TTY menus | Prompts, config, optional launch | Can write serving_config.json, request credentials, network metadata, launch | Interactive-only; never automated headlessly |
| Cloud preview | Complete `serve --backend modal --dry-run` | Target/draft parsed but NOT forwarded | `modal run experiments/modal_lanes.py --gpu L4` | Does not honor model/port/adaptive choices | Read-only preview; real cloud action deferred until corrected/verified |
| Colab | Complete `serve --backend colab --dry-run` | Both models | Placeholder message | No actual notebook URL or deployment in flag path | Preview only |

Implemented: detect, profile, stop, serve; --version/--help. Serve flags:
--target, --draft, --backend {modal,local,colab} (default modal), --port (8000),
--background, --deploy, --no-adaptive, --dry-run.

Documented/planned but absent: test, check-compat, --json, --adaptive,
--gpu-memory-utilization at the FlexSDS parser, profile micro-calibration.
README/product/runbook performance promises are UNVERIFIED, not executable contracts.
Tree topology/controller code exists; canonical v0.2 runtime is linear adaptive;
TreeAdaptive runtime and optimal hardware policy remain FUTURE WORK. Prior M4 evidence
is HISTORICAL and configuration-specific. No telemetry workflow in this v0.1.

Boundary decision: agent reasons and reports; CLI executes inspection and previews.
No wrapper: existing module entrypoint provides a deterministic argv interface;
tests use subprocess argv with stdin closed, timeout and captured output. No core duplication.
Real launch needs explicit user intent plus verified backend prerequisites; known runner
limitations must be resolved rather than silently ignoring requested settings.
