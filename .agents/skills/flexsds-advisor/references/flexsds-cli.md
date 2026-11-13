# Actual CLI contract

Audited at develop 48ae3a166ad9e42fdb81d3d401e496efef3b71bd (2026-11-13).
Authority: pyproject.toml; src/flexsds/cli/{main,wizard,banner}.py;
core/{hardware,compatibility,cost_model}.py; server/{local,modal}_runner.py.
Recheck installed help/code if versions differ. No JSON output contract.

| Command | Observable behavior | Agent boundary |
|---|---|---|
| `--help`, `--version` | Parser exits before detection; version 0.2.0-alpha | Headless, read-only |
| `detect` | ANSI/Unicode banner and hardware summary | Headless, GPU optional; Torch exception silently becomes CPU/RAM fallback |
| `profile` | Filename lookup, not micro-calibration | sm_89 -> L4; sm_75 -> T4; everything else -> L4; missing file -> analytical_fallback |
| Complete `serve ... --backend local --dry-run` | Heuristic compatibility plus displayed vLLM argv and partial environment | Headless; no server, credentials or weights downloaded; display is not shell-escaped |
| Complete `serve ... --backend modal --dry-run` | Prints Modal command only, no credential check | Headless, no allocation; backend default is modal, so always specify it |
| `stop` | Reads CWD .flexsds.pid; sends POSIX process-group SIGTERM; removes PID file | Explicit stop intent plus verified ownership/platform; local only |
| Bare command or incomplete `serve` | Interactive wizard | Never headless, including with --dry-run; may fetch metadata, set credentials, write serving_config.json, launch |

Serve options actually parsed: --target, --draft, --backend {modal,local,colab}
(default modal), --port (integer, default 8000), --background, --deploy,
--no-adaptive, --dry-run. Both nonempty IDs are required to avoid wizard;
parser does not validate model existence, port range or hardware capability.
Use a subprocess argv list; never execute the printed joined command as shell code.

## Important execution limits

- Flag-mode check_compatibility uses model-name heuristics, defaults to AWQ-like
  target estimates and uses local free memory even for remote backends. It does
  not call the metadata-based pre_download_guard used by the wizard. Warning/OOM
  does not prevent runner invocation; exit 0 is possible despite warnings.
- CPU fallback fills memory fields from available system RAM (or 16000 MB default).
  A COMPATIBLE summary on CPU is not a local GPU serving approval. Unknown GPUs
  and same-SM non-L4 GPUs are not validated by selecting the L4 file.
- Local command uses vllm serve, binds 0.0.0.0, sets speculative tokens to 4 and
  AdaptiveAsyncScheduler. `enable_adaptive` from --no-adaptive is ignored by the
  runner: it does not disable adaptation or produce a vanilla baseline.
  vLLM/GPU/runtime and repository src.server imports are separate prerequisites;
  installing this lightweight package does not establish them. Background/stop
  use os.setsid/os.killpg and are not Windows-compatible. Foreground Popen returns
  before server readiness and does not propagate child exit status.
- Modal builder emits `modal run experiments/modal_lanes.py --gpu L4` (or deploy).
  Selected target/draft/port/adaptive settings are not forwarded. Real use requires
  Modal CLI, authenticated account and paid-resource intent; the launch path does
  not check credentials. Do not launch arbitrary requested models through this
  path until forwarding and underlying script arguments are verified/corrected.
- Complete Colab flag mode prints only a placeholder on dry-run and does nothing
  otherwise. Do not claim a notebook URL or endpoint was created.
- Local launch can create flexsds_server.log/.flexsds.pid; downstream server may
  download weights/write telemetry. Modal launch can allocate billable cloud GPU.
  stop is not a readiness/cleanup guarantee; never trust a stale PID blindly.

## Failures and documentation discrepancies

Argparse invalid command/option -> exit 2. Imports/runtime may raise nonzero.
Missing vllm/modal binary is caught and printed by runner, yet dispatcher returns
0. Inspect text and verify readiness separately after an authorized real launch.
Windows console-script revalidation (2026-11-13): detect, profile and complete
local serve --dry-run use automatic ASCII presentation when stdout cannot encode
Unicode. Rich branding remains on UTF-8/TrueColor terminals; pipes, NO_COLOR and
unknown terminal capabilities use an ASCII banner. No PYTHONUTF8 or
PYTHONIOENCODING setup is required. Only presentation changed; interactive wizard
execution and real serving remain outside this validation. Capture raw stdout/
stderr using the actual stream encoding; a successful dry-run is not readiness.

Product docs mention `test --endpoint`, `check-compat`, `detect --json`, --adaptive,
--gpu-memory-utilization and profile calibration: these are absent from the
current CLI. They are planned/documented-only, not commands to recommend.
The agent happy-path runbook is product vision and includes invented forecasts
and broader publication claims; it is not evidence of current capability.
