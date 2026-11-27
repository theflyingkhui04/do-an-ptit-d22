# Current CLI contract

Reconciled 2026-11-27 against develop e53b737 plus fix/m5-01-cli-final-fixes
(every command run on Windows from outside the checkout). Actual parser/source wins
over documentation. Version 0.3.4 from a `pip install -e .` checkout; record the
execution Git SHA too. PyPI 0.3.4 is older code (tag v0.2.4): 9 commands, no
`benchmark`, `--gpu {L4,T4,A10G}` defaulting to L4. Only `benchmark --json` emits JSON.

| Command | Arguments / behavior | Effects and exits |
|---|---|---|
| --help, --version | Parser exits before detection | Read-only, 0; parser errors 2 |
| detect | Torch detection, exceptions fall back to CPU/RAM | Inspection, 0; RAM is not VRAM |
| profile | Lookup: bundled A10 profile, user profiles in ~/.flexsds/profiles (T4, L40S, A100); a GPU with no measured profile -> conservative_analytical_fallback (adaptive refuses it). `--benchmark` measures and saves a profile (local CUDA, or `--backend modal --gpu G`), `--measure-mode {fast,full}`, `--batches` (a prefix of 1,2,4,8,16,32; T4 uses 1,2,4,8) | Lookup is not validated support; `--benchmark --backend modal` spends GPU |
| inspect | `inspect MODEL [--eagle-version] [--method]` | Reads checkpoint metadata; reports speculative method (e.g. EAGLE-3.1 -> eagle3) |
| clear | `clear modal` | Removes FlexSDS Modal tokens and the active native Modal profile; explicit intent |
| benchmark | [plan\|run\|status\|validate\|aggregate\|stop], --campaign-id, --mode {full,fast}, --gpus, --workloads, --lanes, --auto-calibrate, --retry-failed, --yes, --allow-uncommitted, --dry-run, --json. Needs a checkout installed with `pip install -e .` (experiments package + frozen workloads); `modal` client >= 1.6.4 | Multi-GPU 3-node (W1_V3/W2/W3) x 4 GPUs (T4/A10/L40S/A100) x 3 lanes (vanilla/static/adaptive); full=216 cells, fast=108 cells. One Modal container per GPU (exact card: A10 = NVIDIA A10, never A10G), vLLM on localhost so concurrency is real (validated from vLLM /metrics); all lanes eager + async without prefix caching. `run` prints a rough cost per GPU and asks before spending unless --yes; the run continues on Modal if the laptop sleeps (`status` collects finished cells). `stop --campaign-id ID` cancels only that campaign's GPU calls; `stop` without an ID stops the shared `flexsds-benchmark` app and every campaign on it. A profile measured only up to B<max concurrency is accepted; `plan` prints "(measured B<=N; larger batches use B=N costs)". `aggregate` leaves out of α/τ only requests showing the known vLLM 0.30 JSON-grammar post-EOS over-count (rule C1–C6, ≤ 5 % of a cell) and lists them in the summary/report; any other counter anomaly keeps the cell INVALID |
| auth status | Masked stored credentials | Reads ~/.flexsds/credentials; no provider validation or environment-only token report |
| auth login | --provider {modal,huggingface,cloudflare}, --key; getpass prompt | Writes credentials; never recommend --token with a secret in argv |
| auth logout | Optional --provider, otherwise all stored providers | Deletes stored keys and matching process environment; explicit intent |
| Complete serve --dry-run | Shared preflight plus generated command | 0 with the command, also when blocked only by the VRAM estimate or missing CUDA ("Launch blocked" printed); 2 for an unsupported/unverifiable model pair; no process, weights, credential access or port probe |
| Real local serve | Preflight, port probe, vLLM subprocess | Unsafe 2 before runner; runner None -> 1; foreground waits and propagates status |
| Real Modal serve | Shared remote budget, credentials, standalone HTTP app | modal serve (ephemeral) or modal deploy; subprocess status propagated; cloud effects |
| serve --backend colab | Repository notebook guidance | 0 means guidance; no endpoint or model forwarding |
| chat | --endpoint http://localhost:8000/v1, --model, --temperature 0 | Health HTTP then interactive SSE inference POSTs; initially unreachable 1; normal exit 0 |
| monitor | --trace-file telemetry/serving_trace.jsonl, --refresh-rate 1 (positive) | Tails file, waits if absent, skips malformed JSON; Ctrl+C 0; not endpoint health. serving_trace.jsonl rows: one per finished request (steps, mean_k, accept_len, alpha); serving_steps.jsonl (adaptive with step telemetry): per-step k and accepted/drafted, skipped feedback shown, planning records hidden |
| stop | CWD .flexsds.pid, local only | POSIX process-group SIGTERM and PID removal; verify ownership. Windows unsupported |
| Bare/incomplete serve | Interactive wizard, even incomplete --dry-run | Never automate: config writes/launch possible; unsafe guard -> CLI 2 before writing/launching |

Serve flags: --target, --draft, --backend {modal,local,colab} (default modal),
--gpu {T4,A10,L40S,A100} (default A10; remote budget/Modal allocation; A100 maps to Modal A100-80GB; L4 is
rejected), --port (1..65535, default 8000), --max-num-seqs (default 32), --max-model-len (default 4096; Node 1
needs 8192), --allow-oom-risk (skip only the VRAM-estimate block), --background, --deploy, --no-adaptive
(static), --static-k N (1..8, requires --no-adaptive), --vanilla (target only, no --draft),
--match-adaptive-runtime (static/vanilla run eager + async like adaptive), --no-prefix-caching,
--no-step-telemetry, --eagle-version {1,2,3,3.1}, --allow-fallback (deprecated, no effect), --dry-run,
--scaledown-window (Modal: seconds the GPU stays up after the last request, 2..3600, default 300; a longer cold
start than Modal's 150 s web limit reaches clients as HTTP 303).
Invalid combinations (--vanilla with --draft, --static-k without --no-adaptive) and a bad port return 2.
Pass both exact nonempty model IDs, explicitly select backend, use argv arrays.
Printed joined commands are not shell-escaped.

## Guard and readiness

Flag mode and wizard consume evaluate_serving_preflight(), which combines
check_compatibility() and pre_download_guard(). It supplies mock_metadata={} for
offline name/dtype estimates, so no metadata HTTP or weights are fetched.
Both results must be safe. OOM, incompatible report, unsafe pre-download result,
or missing local CUDA blocks real dispatch. Known Qwen/Llama cross-family pairs
are rejected. This is heuristic validation, not tokenizer/kernel support.

Threshold: estimates must fit 95% of budget. Local budget is detected free VRAM;
remote budgets are T4 15000 MB, A10 22000 MB, L40S 46000 MB, A100 78000 MB (Colab wizard also maps L4 23000 MB; CLI `--gpu` accepts T4/A10/L40S/A100). Those are conservative
assumptions, not measured container memory. Laptop RAM/VRAM does not decide remote
GPU compatibility. A dry-run blocked only by the VRAM estimate or missing local
CUDA prints "Launch blocked", shows the preview and returns 0; an unsupported or
incompatible model pair returns 2 even with --dry-run (no valid command exists).
Unsafe real serving returns 2 with no runner call. Adaptive also needs a measured
profile whose batch grid is a prefix of 1,2,4,8,16,32 (missing profile -> exit 1); a
grid shorter than --max-num-seqs is accepted with "Profile measured up to B=N" (the
planner uses the nearest measured batch). Launch failure
returns 1; foreground child exits propagate. Started process/background PID/
successful deployment is not endpoint readiness or performance evidence.
ADAPTIVE_ENABLE_VRAM_GUARD controls runtime speculative budgets separately.

## Backend details

Local binds 0.0.0.0. Runner/wizard defaults are Qwen/Qwen2.5-7B-Instruct-AWQ and
duckhoi/qwen2.5-7b-eagle3.1-speculator. Never substitute these for user inputs.
Qwen/Qwen2.5-0.5B-Instruct selects draft_model, not eagle3. Adaptive commands add
AdaptiveAsyncScheduler, LineageUniProcExecutor and per-request metrics.
--no-adaptive keeps static speculation and omits the custom scheduler/executor
and dynamic batch lookup; it does not make a vanilla baseline.
Port conflict stops Popen, but probing is not atomic reservation.
Background/stop are POSIX-only. Never stop an unrelated/stale PID.

Modal needs the `modal` client >= 1.6.4 on PATH (older clients are refused by the
workspace image builder 2026.10). It forwards exact non-secret model/draft/method/port/GPU/adaptive settings in
FLEXSDS_SERVING_CONFIG via an allowlist (FLEXSDS_SERVING_CONFIG_B64 inside the container); modal_app declares GPU resources and
web_server port and reuses the local command builder. Dry-run never accesses
credentials. Real launch accepts environment/stored Modal ID+secret or existence
of ~/.modal.toml; presence is not verified authentication. Configure both keys
securely: default auth login saves only the ID. modal serve exposes a temporary
public development URL; deploy persists. Live cloud serving is not established
by mocked tests. Official references:
https://modal.com/docs/guide/webhooks
https://modal.com/docs/reference/modal.web_server

Colab flag mode gives guidance only. Wizard can show instructions and probe a
supplied tunnel URL. The notebook is a fixed EAGLE example with installation and
tunnel effects; configure exact models on that machine. Its develop URL is not
available until the notebook is published there. No local command opens a
notebook, creates a tunnel or launches a Colab server.

vLLM, CUDA, quantization kernels and repository src.server/src.proposer imports
are separate prerequisites. pyproject installs only psutil; a lightweight
installation does not establish the full serving runtime.

## Client, credentials and platform limits

Chat uses /models then root /health and POST /chat/completions. Chunk count/rate
and ms/chunk measure SSE content chunks, not tokenizer tokens, token throughput,
TPOT or speedup. Streaming has no explicit read timeout.
Monitor does not repair partial-line loss or file rotation.

Credentials are JSON outside the repo; POSIX 0700/0600 modes are attempted.
Windows ACL confidentiality is not established by chmod/stat. Never claim
universal zero-leak security or place tokens in URLs, argv, logs or chat.
Imported helper artifacts now use public Git URLs; historical removal is not
revocation: CREDENTIAL ROTATION STILL REQUIRED BY OWNER.

Windows output preserves Unicode when supported with ASCII fallback for cp1252/
unknown streams, including auth/chat/monitor/TaskRunner/Modal. No global stream
reconfiguration required.

Unsupported doc examples: test --endpoint, check-compat, detect --json,
--adaptive, serve --gpu-memory-utilization.
Plan checkboxes and historical worklogs are not current runtime proof.
