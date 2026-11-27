# Current CLI contract

Reconciled 2026-11-13 against backend 76957514f9a52e11d9a0a2322469d1a45cba9273
plus the Skill integration changes. Actual parser/source wins over documentation.
Version remains 0.2.0-alpha: record execution Git SHA too. No JSON output contract.

| Command | Arguments / behavior | Effects and exits |
|---|---|---|
| --help, --version | Parser exits before detection | Read-only, 0; parser errors 2 |
| detect | Torch detection, exceptions fall back to CPU/RAM | Inspection, 0; RAM is not VRAM |
| profile | sm_89 -> L4; sm_75 -> T4; other -> L4; missing file -> analytical_fallback | Lookup, not calibration or validated hardware support |
| benchmark | [plan\|run\|status\|validate\|aggregate\|stop], --mode {full,fast}, --gpus, --workloads, --lanes, --auto-calibrate, --yes, --allow-uncommitted, --dry-run | Multi-GPU 3-node (W1_V3/W2/W3) x 4 GPUs (T4/A10/L40S/A100) x 3 lanes (vanilla/static/adaptive); full=216 cells, fast=108 cells. One Modal container per GPU (exact card: A10 = NVIDIA A10, never A10G), vLLM on localhost so concurrency is real (validated from vLLM /metrics); all lanes eager + async without prefix caching. `run` prints a rough cost per GPU and asks before spending unless --yes; the run continues on Modal if the laptop sleeps (`status` collects finished cells). `aggregate` leaves out of α/τ only requests showing the known vLLM 0.30 JSON-grammar post-EOS over-count (rule C1–C6, ≤ 5 % of a cell) and lists them in the summary/report; any other counter anomaly keeps the cell INVALID |
| auth status | Masked stored credentials | Reads ~/.flexsds/credentials; no provider validation or environment-only token report |
| auth login | --provider {modal,huggingface,cloudflare}, --key; getpass prompt | Writes credentials; never recommend --token with a secret in argv |
| auth logout | Optional --provider, otherwise all stored providers | Deletes stored keys and matching process environment; explicit intent |
| Complete serve --dry-run | Shared preflight plus generated command | Preview 0 even when blocked; no process, weights, credential access or port probe |
| Real local serve | Preflight, port probe, vLLM subprocess | Unsafe 2 before runner; runner None -> 1; foreground waits and propagates status |
| Real Modal serve | Shared remote budget, credentials, standalone HTTP app | modal serve (ephemeral) or modal deploy; subprocess status propagated; cloud effects |
| serve --backend colab | Repository notebook guidance | 0 means guidance; no endpoint or model forwarding |
| chat | --endpoint http://localhost:8000/v1, --model, --temperature 0 | Health HTTP then interactive SSE inference POSTs; initially unreachable 1; normal exit 0 |
| monitor | --trace-file telemetry/serving_trace.jsonl, --refresh-rate 1 (positive) | Tails file, waits if absent, skips malformed JSON; Ctrl+C 0; not endpoint health |
| stop | CWD .flexsds.pid, local only | POSIX process-group SIGTERM and PID removal; verify ownership. Windows unsupported |
| Bare/incomplete serve | Interactive wizard, even incomplete --dry-run | Never automate: config writes/launch possible; unsafe guard -> CLI 2 before writing/launching |

Serve flags: --target, --draft, --backend {modal,local,colab} (default modal),
--gpu {T4,A10,L40S,A100} (default A10; remote budget/Modal allocation; A100 maps to Modal A100-80GB), --port (1..65535,
default 8000), --background, --deploy, --no-adaptive, --dry-run, --scaledown-window (Modal: seconds the GPU
stays up after the last request, 2..3600, default 300; a longer cold start than Modal's 150 s web limit reaches
clients as HTTP 303).
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
Unsafe real serving returns 2 with no runner call. Launch failure
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

Modal forwards exact non-secret model/draft/method/port/GPU/adaptive settings in
FLEXSDS_SERVING_CONFIG via an allowlist; modal_app declares GPU resources and
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
--adaptive, serve --gpu-memory-utilization, profile calibration.
Plan checkboxes and historical worklogs are not current runtime proof.
