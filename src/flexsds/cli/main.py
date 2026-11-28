"""FlexSDS CLI main dispatcher module."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

try:
    from flexsds.cli.output import print_output as print
    from flexsds import __version__
    from flexsds.cli.banner import print_banner
    from flexsds.cli.wizard import run_interactive_wizard
    from flexsds.core.hardware import (
        DEFAULT_MODAL_GPU,
        canonical_modal_gpu_name,
        MODAL_GPU_NAMES,
        HardwareInfo,
        build_remote_hardware_info,
        detect_hardware,
    )
    from flexsds.core.compatibility import (
        check_compatibility,
        evaluate_serving_preflight,
        ServingBlockedError,
    )
    from flexsds.core.cost_model import (
        FlexCostModel,
        load_profile_for_hardware,
        resolve_profile_path,
        run_hardware_micro_benchmark,
    )
    from flexsds.core.credentials import CredentialsManager
    from flexsds.client.chat import run_interactive_chat
    from flexsds.client.monitor import run_live_monitor
    from flexsds.cli.task_runner import TaskRunner
    from flexsds.server.local_runner import LocalServingRunner, finish_launch
    from flexsds.server.modal_runner import ModalServingRunner
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print
    from src.flexsds import __version__
    from src.flexsds.cli.banner import print_banner
    from src.flexsds.cli.wizard import run_interactive_wizard
    from src.flexsds.core.hardware import (
        DEFAULT_MODAL_GPU,
        canonical_modal_gpu_name,
        MODAL_GPU_NAMES,
        HardwareInfo,
        build_remote_hardware_info,
        detect_hardware,
    )
    from src.flexsds.core.compatibility import (
        check_compatibility,
        evaluate_serving_preflight,
        ServingBlockedError,
    )
    from src.flexsds.core.cost_model import (
        FlexCostModel,
        load_profile_for_hardware,
        resolve_profile_path,
        run_hardware_micro_benchmark,
    )
    from src.flexsds.core.credentials import CredentialsManager
    from src.flexsds.client.chat import run_interactive_chat
    from src.flexsds.client.monitor import run_live_monitor
    from src.flexsds.cli.task_runner import TaskRunner
    from src.flexsds.server.local_runner import LocalServingRunner, finish_launch
    from src.flexsds.server.modal_runner import ModalServingRunner


def positive_int(value):
    import argparse
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def scaledown_seconds(value):
    """argparse type for --scaledown-window: seconds a Modal GPU stays up after the last request (2..3600)."""
    try:
        seconds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--scaledown-window must be an integer number of seconds") from exc
    if not 2 <= seconds <= 3600:
        raise argparse.ArgumentTypeError("--scaledown-window must be between 2 and 3600 seconds")
    return seconds


def _is_console(stream):
    """True for an interactive console. On Windows the NUL device reports isatty() True as well."""
    import os

    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt

        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(stream.fileno()), ctypes.byref(mode)))
    except (AttributeError, OSError, ValueError):
        return True


def read_secret(name, context=""):
    """Prompt for a secret on a terminal, or read one line from stdin when there is none.

    On Windows getpass reads the console, not stdin, so a non-interactive login used to hang.
    """
    import getpass

    stdin = sys.stdin
    if stdin is not None and stdin.isatty() and _is_console(stdin):
        return getpass.getpass(f"Enter {name}{context}: ")
    line = stdin.readline() if stdin is not None else ""
    if not line.strip():
        raise RuntimeError(f"No terminal to type {name}: pipe it on stdin "
                           "(e.g. `printf '%s\\n' \"$VALUE\" | flexsds auth login ...`) or run in a terminal.")
    return line.strip()


def calibration_memory_share(value):
    try:
        share = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number within 0.50..0.98") from None
    if not 0.5 <= share <= 0.98:
        raise argparse.ArgumentTypeError("expected a number within 0.50..0.98")
    return share


def calibration_batches(value):
    """argparse type for --batches: a subset of the calibration grid 1,2,4,8,16,32."""
    try:
        batches = sorted({int(part) for part in str(value).split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batches must be integers such as 1,2,4,8") from exc
    if not batches or any(b not in (1, 2, 4, 8, 16, 32) for b in batches):
        raise argparse.ArgumentTypeError("--batches must be chosen from 1,2,4,8,16,32")
    return batches


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flexsds",
        description="FlexSDS: Hardware-Adaptive EAGLE Speculative Decoding Serving Engine for LLMs",
        epilog="Run flexsds for the wizard, or flexsds COMMAND --help for command details.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"flexsds {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    examples = {
        "clear": "flexsds clear modal",
        "detect": "flexsds detect",
        "inspect": "flexsds inspect duckhoi/qwen2.5-7b-eagle3.1-speculator",
        "profile": "flexsds profile --benchmark --backend modal --gpu A10",
        "benchmark": "flexsds benchmark --mode fast --dry-run",
        "stop": "flexsds stop",
        "auth": "flexsds auth status",
        "login": "flexsds auth login --provider modal",
        "status": "flexsds auth status",
        "logout": "flexsds auth logout --provider modal",
        "serve": "flexsds serve --backend modal --gpu A10 --dry-run",
        "chat": "flexsds chat --endpoint http://localhost:8000/v1",
        "monitor": "flexsds monitor --trace-file telemetry/serving_trace.jsonl",
    }

    def add_command(group, name, *, help):
        return group.add_parser(name, help=help, description=help,
                                epilog=f"Example: {examples[name]}",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    clear_parser = add_command(subparsers, "clear", help="Disconnect a local service account and require login again")
    clear_parser.add_argument("provider", choices=["modal"], help="Clear FlexSDS tokens and the active native Modal profile")

    # Command: detect
    add_command(subparsers, "detect", help="Detect and display hardware / GPU specs")

    # Method inspection is independent of serving capability and GPU hardware.
    inspect_parser = add_command(subparsers, "inspect", help="Identify a speculative method from model artifacts")
    inspect_parser.add_argument("model", nargs="?", help="Hugging Face model ID or local model directory")
    inspect_parser.add_argument("--eagle-version", choices=["1", "2", "3", "3.1"], help="Resolve the version of a shared EAGLE checkpoint")
    inspect_parser.add_argument("--method", help="Inspect an explicitly selected inference policy, e.g. ngram or jacobi")

    # Command: profile
    profile_parser = add_command(subparsers, "profile", help="Load hardware cost profile for current GPU")
    profile_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure real model costs on CUDA and save a profile (may take minutes)",
    )
    profile_parser.add_argument(
        "--target",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="Target model identifier",
    )
    profile_parser.add_argument(
        "--draft",
        type=str,
        default="duckhoi/qwen2.5-7b-eagle3.1-speculator",
        help="Draft model or speculator head identifier",
    )
    profile_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier override for benchmarking",
    )

    profile_parser.add_argument("--measure-mode", choices=["fast", "full"], default="full", help="Measurement repetitions: fast (5) or full (50)")
    profile_parser.add_argument("--batches", type=calibration_batches, default=None,
                                help="Comma-separated batch sizes from 1,2,4,8,16,32 (default: all); a shorter prefix such as 1,2,4,8 when larger batches do not fit")
    profile_parser.add_argument("--gpu-memory-utilization", type=calibration_memory_share, default=None,
                                help="vLLM memory share while calibrating (0.50..0.98, default 0.85); more room for the one-step B x 1024 prefill (on T4, 0.95 still ran out of memory at B=16 on 27/11)")
    profile_parser.add_argument("--backend", choices=["local", "modal"], default="local", help="Where to run calibration")
    profile_parser.add_argument("--gpu", type=canonical_modal_gpu_name, choices=list(MODAL_GPU_NAMES), default=DEFAULT_MODAL_GPU, help="GPU to allocate for Modal calibration")

    # Command: benchmark
    bench_parser = add_command(
        subparsers,
        "benchmark",
        help="Plan, run, validate, or aggregate the multi-GPU 3-node 3-lane Modal benchmark",
    )
    bench_parser.add_argument(
        "bench_action",
        nargs="?",
        choices=["plan", "run", "status", "validate", "aggregate", "stop"],
        default="plan",
        help="Campaign lifecycle action",
    )
    bench_parser.add_argument(
        "--mode",
        choices=["full", "fast"],
        default="full",
        help="Concurrency grid: full (1,2,4,8,16,32 -> 216 cells) or fast (1,8,32 -> 108 cells)",
    )
    bench_parser.add_argument("--campaign-id", type=str, default=None, help="Explicit benchmark campaign identifier")
    bench_parser.add_argument("--execution-sha", type=str, default=None, help="Committed 40-char Git SHA (defaults to HEAD)")
    bench_parser.add_argument("--gpus", type=str, default="all", help="Comma-separated GPUs (T4,A10,L40S,A100) or 'all'")
    bench_parser.add_argument("--workloads", type=str, default="all", help="Comma-separated workloads (W1_V3,W2,W3) or 'all'")
    bench_parser.add_argument("--lanes", type=str, default="all", help="Comma-separated lanes (vanilla,static,adaptive) or 'all'")
    bench_parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-7B-Instruct-AWQ", help="Target model identifier")
    bench_parser.add_argument("--draft", type=str, default="duckhoi/qwen2.5-7b-eagle3.1-speculator", help="Draft model or speculator identifier")
    bench_parser.add_argument("--draft-method", choices=["eagle", "eagle3", "draft_model"], default="eagle3", help="Speculative decoding method")
    bench_parser.add_argument("--static-k", type=int, default=3, help="Fixed speculative draft budget K for static lane (1..8)")
    bench_parser.add_argument("--adaptive-max-k", type=int, default=4, help="Maximum K ceiling for adaptive lane (1..8)")
    bench_parser.add_argument("--profile-dir", type=str, default=None, help="Optional extra directory for verified GPU cost profiles")
    bench_parser.add_argument("--auto-calibrate", action="store_true", help="Calibrate missing Modal GPUs automatically before running adaptive lane")
    bench_parser.add_argument("--calibrate-mode", choices=["fast", "full"], default="fast", help="Calibration mode when --auto-calibrate is enabled")
    bench_parser.add_argument("--retry-failed", action="store_true", help="Submit a new attempt for a failed cell")
    bench_parser.add_argument("--yes", action="store_true", help="Do not ask before spending GPU time on Modal")
    bench_parser.add_argument("--allow-uncommitted", action="store_true",
                              help="Measure the working tree; record a source digest instead of a clean commit")
    bench_parser.add_argument("--dry-run", action="store_true", help="Preview benchmark matrix and GPU profile readiness without Modal calls")
    bench_parser.add_argument("--json", dest="json_output", action="store_true", help="Output raw plan JSON instead of summary text")

    # Command: stop
    add_command(subparsers, "stop", help="Stop running local FlexSDS server")

    # Command: auth
    auth_parser = add_command(subparsers, "auth", help="Manage secure credentials outside repository")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_action", help="Auth action to perform")
    
    # auth login
    auth_login_parser = add_command(auth_subparsers, "login", help="Save API key / token securely")
    auth_login_parser.add_argument("--provider", choices=["modal", "huggingface", "cloudflare"], default="modal", help="Target service provider")
    auth_login_parser.add_argument("--key", type=str, default=None, help="Specific credential key name")
    auth_login_parser.add_argument("--token", type=str, default=None, help="Token value (prompted securely if omitted)")

    # auth status
    add_command(auth_subparsers, "status", help="Show masked status of saved credentials")

    # auth logout
    auth_logout_parser = add_command(auth_subparsers, "logout", help="Delete saved credentials")
    auth_logout_parser.add_argument("--provider", type=str, default=None, help="Provider to remove; omit to remove all saved providers")

    # Command: serve
    serve_parser = add_command(subparsers, "serve", help="Launch speculative serving server")
    serve_parser.add_argument("--target", type=str, help="Target model identifier")
    serve_parser.add_argument("--draft", type=str, help="Draft model or speculator head identifier")
    serve_parser.add_argument(
        "--backend", choices=["modal", "local", "colab"], default="modal", help="Serving backend platform"
    )
    serve_parser.add_argument("--allow-oom-risk", action="store_true", help="Continue despite estimated VRAM risk; other compatibility failures still block")
    serve_parser.add_argument("--max-num-seqs", type=positive_int, default=32, help="Maximum concurrent sequences")
    serve_parser.add_argument("--max-model-len", type=positive_int, default=4096, help="Maximum context length per sequence")
    serve_parser.add_argument("--gpu", type=canonical_modal_gpu_name, choices=list(MODAL_GPU_NAMES), default=DEFAULT_MODAL_GPU, help="Remote GPU memory budget and Modal allocation")
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="API server port"
    )
    serve_parser.add_argument(
        "--scaledown-window", type=scaledown_seconds, default=300,
        help="Modal: seconds the GPU stays up after the last request (default 300; a cold start can exceed Modal's 150 s web limit)",
    )
    serve_parser.add_argument(
        "--background", action="store_true", help="Run local server as background daemon"
    )
    serve_parser.add_argument(
        "--deploy", action="store_true", help="Deploy persistently on Modal Cloud"
    )
    serve_parser.add_argument(
        "--no-adaptive", action="store_true", help="Disable HardwareAdaptive optimization engine"
    )
    serve_parser.add_argument(
        "--match-adaptive-runtime",
        action="store_true",
        help="Match Adaptive ON's V1 eager execution contract when serving static or vanilla baselines",
    )
    serve_parser.add_argument(
        "--vanilla",
        action="store_true",
        help="Serve target model only without speculative decoding (implies --no-adaptive)",
    )
    serve_parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="Disable vLLM automatic prefix caching (recommended for lossless repeatability checks)",
    )
    serve_parser.add_argument(
        "--static-k",
        type=positive_int,
        default=None,
        help="Explicit static draft token budget K in 1..8 (requires --no-adaptive without --vanilla)",
    )
    serve_parser.add_argument(
        "--no-step-telemetry",
        action="store_true",
        help="Disable per-step adaptive telemetry file I/O (ADAPTIVE_STEP_TELEMETRY_PATH) for peak throughput",
    )
    serve_parser.add_argument(
        "--allow-fallback", action="store_true", help="Deprecated: analytical fallback cannot satisfy empirical serving requirements"
    )
    serve_parser.add_argument(
        "--dry-run", action="store_true", help="Print execution command without launching"
    )

    serve_parser.add_argument("--eagle-version", choices=["1", "2", "3", "3.1"], help="Explicit EAGLE version; required for shared legacy checkpoints")

    # Command: chat
    chat_parser = add_command(subparsers, "chat", help="Start interactive streaming chat with served model")
    chat_parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1", help="API base URL")
    chat_parser.add_argument("--model", type=str, default=None, help="Served model name override")
    chat_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    # Command: monitor
    monitor_parser = add_command(subparsers, "monitor", help="Live telemetry monitor for speculative decoding traces")
    monitor_parser.add_argument("--trace-file", type=str, default="telemetry/serving_trace.jsonl", help="Path to JSONL telemetry trace")
    monitor_parser.add_argument("--refresh-rate", type=float, default=1.0, help="Telemetry poll interval in seconds")

    return parser.parse_args(argv)


def get_serving_cost_profile(
    hw_info: HardwareInfo,
    allow_fallback: bool = False,
    custom_profile_path: Optional[str] = None,
) -> tuple[Optional[FlexCostModel], str]:
    """Load an empirical serving profile; legacy allow_fallback cannot bypass it."""
    return load_profile_for_hardware(
        hw_info,
        custom_profile_path=custom_profile_path,
        strict=True,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for FlexSDS command-line interface."""
    args = parse_args(argv)
    if args.command == "inspect":
        from flexsds.core.speculative_method import detect_method, inspect_method, method_support
        if bool(args.model) == bool(args.method):
            print("Provide a model ID/path or --method, exactly one.")
            return 2
        try:
            if args.model:
                detection, inspected_config = inspect_method(args.model, with_config=True, eagle_version=args.eagle_version)
            else:
                detection, inspected_config = detect_method({"speculative_method": args.method}), {}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Method identification failed: {exc}")
            return 2
        if detection.method == 'eagle':
            print("EAGLE version selection required: use --eagle-version 1 or --eagle-version 2")
            return 2
        print(f"Architecture: {detection.family_label}")
        print(f"Speculative method: {detection.label}")
        print(f"Evidence: {detection.evidence}")
        if args.model:
            supported, support_reason = method_support(detection, inspected_config)
        else:
            supported, support_reason = False, "Method policy only; no checkpoint inspected"
        print("FlexSDS checkpoint support: " + ("Supported (pairing/VRAM checks required)" if supported else "Not supported yet / unverified"))
        if supported:
            print(f"Backend method: {detection.backend_method} (sequential drafting)")
        if support_reason:
            print(support_reason)
        return 2 if detection.issue or detection.method == 'unknown' or detection.family == 'unknown' else 0
    if args.command == "serve" and not 1 <= args.port <= 65535:
        print("Invalid port: expected 1..65535")
        return 2
    if args.command == "monitor" and args.refresh_rate <= 0:
        print("Refresh rate must be positive")
        return 2
    if args.command == "clear":
        from flexsds.server.modal_auth import clear_modal
        try:
            clear_modal(CredentialsManager())
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Modal disconnect failed: {exc}")
            return 1
        print("Modal disconnected locally. Log in again when selecting Modal, or run flexsds auth login --provider modal.")
        return 0
    if args.command == "benchmark":
        try:
            from flexsds.cli.benchmark import run_benchmark_command
        except ModuleNotFoundError:
            from src.flexsds.cli.benchmark import run_benchmark_command
        return run_benchmark_command(args)
    hw_info = detect_hardware()

    if args.command == "stop":
        local_runner = LocalServingRunner()
        local_stopped = local_runner.stop()
        if local_stopped:
            print("\033[32m✔ Local vLLM server process stopped successfully.\033[0m")
        else:
            print("\033[90mNo active local server process found.\033[0m")
        return 0

    if args.command == "auth":
        cred_mgr = CredentialsManager()
        if args.auth_action == "status":
            masked = cred_mgr.get_all_masked()
            print("\n\033[1;37mFlexSDS Secure Credentials (~/.flexsds/credentials):\033[0m")
            if not masked:
                print("  \033[90m(No credentials saved. Run 'flexsds auth login' to add.)\033[0m")
            for prov, keys in masked.items():
                print(f"  \033[32m✔ {prov}:\033[0m")
                for k, v in keys.items():
                    print(f"      {k}: {v}")
            return 0
        elif args.auth_action == "logout":
            target_prov = getattr(args, "provider", None)
            if target_prov:
                deleted = cred_mgr.delete_credential(target_prov)
                if deleted:
                    print(f"\033[32m✔ Removed credentials for provider: {target_prov}\033[0m")
                else:
                    print(f"\033[90mNo credentials found for provider: {target_prov}\033[0m")
            else:
                for prov in list(cred_mgr.get_all_masked().keys()):
                    cred_mgr.delete_credential(prov)
                print("\033[32m✔ Removed all stored credentials.\033[0m")
            return 0
        elif args.auth_action == "login":
            prov = args.provider
            if prov == "modal":
                from flexsds.server.modal_auth import ensure_modal, print_token_instructions, save_verified_modal_token
                try:
                    ensure_modal()
                    print_token_instructions()
                    if args.key not in (None, 'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET'):
                        print('Modal requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.')
                        return 1
                    token_id = (args.token if args.key != 'MODAL_TOKEN_SECRET' else None) or read_secret("MODAL_TOKEN_ID")
                    secret = (args.token if args.key == 'MODAL_TOKEN_SECRET' else None) or read_secret("MODAL_TOKEN_SECRET")
                    save_verified_modal_token(cred_mgr, token_id, secret)
                except (OSError, RuntimeError) as exc:
                    print(str(exc))
                    return 1
                return 0
            key_name = args.key or ("HF_TOKEN" if prov == "huggingface" else ("MODAL_TOKEN_ID" if prov == "modal" else "TOKEN"))
            try:
                val = args.token or read_secret(key_name, f" for {prov}")
            except RuntimeError as exc:
                print(str(exc))
                return 1
            if val:
                cred_mgr.set_credential(prov, key_name, val.strip())
                print(f"\033[32m✔ Securely saved {key_name} for {prov} (POSIX mode 0600; Windows uses filesystem ACLs).\033[0m")
            return 0
        return 0

    # Display banner on interactive startup or regular commands
    print_banner(hw_info)

    if args.command == "detect":
        print(f"\n\033[1;32m✔ Hardware: {hw_info.summary()}\033[0m")
        return 0

    if args.command == "profile":
        if getattr(args, "benchmark", False):
            target_model = args.model or args.target
            draft_model = args.draft
            if args.backend == "modal":
                hw_info = build_remote_hardware_info("modal", args.gpu)
            try:
                cost_model, saved_path = run_hardware_micro_benchmark(
                    hw_info=hw_info, target_model=target_model, draft_model=draft_model, backend=args.backend,
                    measurement_mode=args.measure_mode, batch_sizes=args.batches,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
            except KeyboardInterrupt:
                print("Calibration cancelled.")
                return 130
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"Calibration failed: {exc}")
                return 1
            print(f"\n\033[1;32m✔ Calibrated & saved new hardware profile: {saved_path}\033[0m")
            return 0

        _, profile_name = load_profile_for_hardware(hw_info)
        print(f"\n\033[1;32m✔ Linked hardware profile: {profile_name}\033[0m")
        return 0

    if args.command == "chat":
        return run_interactive_chat(
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
        )

    if args.command == "monitor":
        return run_live_monitor(
            trace_file=args.trace_file,
            refresh_rate=args.refresh_rate,
        )

    if args.command == "serve":
        if args.match_adaptive_runtime and not (args.no_adaptive or args.vanilla):
            print("Invalid flag combination: --match-adaptive-runtime requires --no-adaptive or --vanilla.")
            return 2
        if args.vanilla and args.draft:
            print("Invalid flag combination: --vanilla cannot be combined with --draft.")
            return 2
        if args.static_k is not None:
            if args.vanilla or not args.no_adaptive:
                print("Invalid flag combination: --static-k requires --no-adaptive and cannot be combined with --vanilla.")
                return 2
            if not 1 <= args.static_k <= 8:
                print("Invalid --static-k: expected integer in 1..8.")
                return 2

        raw_args = argv if argv is not None else sys.argv[1:]
        has_explicit_platform = any(a.startswith("--backend") or a.startswith("--gpu") for a in raw_args)
        if (
            (args.target and args.draft)
            or args.vanilla
            or (
                args.dry_run
                and has_explicit_platform
                and not (args.target and not args.draft)
                and not (args.draft and not args.target)
            )
        ):
            target_gpu = getattr(args, "gpu", DEFAULT_MODAL_GPU)
            if args.backend in ("modal", "colab"):
                serving_hw = build_remote_hardware_info(args.backend, target_gpu)
            else:
                serving_hw = hw_info

            target_model = args.target or "Qwen/Qwen2.5-7B-Instruct-AWQ"
            draft_model = None if args.vanilla else (args.draft or "duckhoi/qwen2.5-7b-eagle3.1-speculator")

            if args.vanilla:
                if args.backend == "local" and not args.dry_run and not hw_info.is_cuda_available:
                    print("Launch blocked: CUDA GPU unavailable; system RAM is not serving VRAM")
                    return 2
                cfg = {"serving": {
                    "backend": args.backend,
                    "target_model": target_model,
                    "draft_model": None,
                    "draft_method": "none",
                    "eagle_version": None,
                    "target_revision": None,
                    "draft_revision": None,
                    "use_heterogeneous_vocab": False,
                    "port": args.port,
                    "target_gpu": target_gpu,
                    "modal_deploy": bool(args.deploy),
                    "enable_adaptive": False,
                    "match_adaptive_runtime": args.match_adaptive_runtime,
                    "disable_speculative": True,
                    "disable_prefix_caching": args.no_prefix_caching,
                    "static_k": None,
                    "max_num_seqs": args.max_num_seqs,
                    "max_model_len": args.max_model_len,
                    "scaledown_window": args.scaledown_window,
                    "allow_oom_risk": args.allow_oom_risk,
                    "profile_path": None,
                    "hardware": {
                        "device_name": serving_hw.device_name,
                        "sm_architecture": serving_hw.sm_architecture,
                        "free_vram_mb": serving_hw.free_vram_mb,
                    },
                }}
            else:
                from flexsds.cli.spinner import status_spinner
                with status_spinner("Checking models compatibility"):
                    preflight = evaluate_serving_preflight(target_model, draft_model, hw_info,
                                                       args.backend, target_gpu,
                                                       max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len)
                if args.eagle_version:
                    from flexsds.core.compatibility import select_report_eagle_version
                    try:
                        preflight.report = select_report_eagle_version(preflight.report, args.eagle_version)
                    except ValueError as exc:
                        print(str(exc))
                        return 2
                elif preflight.report.identified_method == "eagle":
                    print("EAGLE version selection required: use --eagle-version 1 or --eagle-version 2")
                    return 2
                print(preflight.report.summary())
                for detail in preflight.report.warnings + preflight.report.recommendations:
                    print(detail)
                reasons = preflight.launch_reasons(args.allow_oom_risk)
                if preflight.report.oom_risk:
                    print(f"VRAM warning at batch={args.max_num_seqs}, context={args.max_model_len}: "
                          "smaller workloads may fit; runtime memory fit is unverified.")
                    if not args.allow_oom_risk:
                        print("Reduce --max-num-seqs/--max-model-len or pass --allow-oom-risk to continue.")
                if reasons:
                    print("Launch blocked: " + "; ".join(reasons))
                if args.backend != "local":
                    print(f"Remote {target_gpu} memory budget is estimated, not measured free VRAM.")
                # CLI contract: a preview blocked only by this machine's limits (VRAM estimate, no CUDA)
                # prints "Launch blocked", still shows the command and returns 0. A real launch, or a
                # model pair that is unsupported/incompatible, is refused with 2 before any runner call.
                if reasons and not (args.dry_run and preflight.preview_allowed(args.allow_oom_risk)):
                    return 2

                if not args.no_adaptive:
                    cost_model, profile_name = load_profile_for_hardware(
                        serving_hw, strict=True, target_model=target_model, draft_model=draft_model,
                    )
                    if cost_model is None or not profile_name:
                        print(
                            f"\n\033[1;31m✖ [Hardware Profile Error] No measured latency profile for GPU '{serving_hw.device_name} ({serving_hw.sm_architecture})'. "
                            f"Please calibrate using: flexsds profile --benchmark (or run with --no-adaptive for static speculation).\033[0m"
                        )
                        return 1

                    if profile_name == "conservative_analytical_fallback":
                        print(
                            f"  \033[1;33m⚠ [Hardware Warning] Using conservative fallback profile ({profile_name} / Conservative Fallback).\033[0m"
                        )
                    else:
                        print(f"\033[32m✔ Linked hardware profile: {profile_name}\033[0m")

                cfg = {"serving": {
                    "backend": args.backend,
                    "target_model": target_model, "draft_model": draft_model,
                    "draft_method": preflight.report.draft_method,
                    "eagle_version": preflight.report.eagle_version.removeprefix("EAGLE-"),
                    "target_revision": preflight.report.target_revision,
                    "draft_revision": preflight.report.draft_revision,
                    "use_heterogeneous_vocab": preflight.report.use_heterogeneous_vocab,
                    "port": args.port, "target_gpu": target_gpu,
                    "modal_deploy": bool(args.deploy),
                    "enable_adaptive": not args.no_adaptive,
                    "enable_step_telemetry": not args.no_step_telemetry,
                    "match_adaptive_runtime": args.match_adaptive_runtime,
                    "disable_speculative": False,
                    "disable_prefix_caching": args.no_prefix_caching,
                    "static_k": args.static_k,
                    "max_num_seqs": args.max_num_seqs, "max_model_len": args.max_model_len,
                    "scaledown_window": args.scaledown_window,
                    "allow_oom_risk": args.allow_oom_risk,
                    "profile_path": str(resolve_profile_path(profile_name)) if not args.no_adaptive else None,
                    "hardware": {
                        "device_name": serving_hw.device_name,
                        "sm_architecture": serving_hw.sm_architecture,
                        "free_vram_mb": serving_hw.free_vram_mb,
                    },
                }}
            if args.backend == "colab":
                print("Colab guidance only: no server or endpoint was created; supplied models are not forwarded to the notebook.")
                print("Open notebooks/flexsds_colab_server.ipynb and configure the exact models on that runtime.")
                print(f"Set MAX_NUM_SEQS={args.max_num_seqs} and MAX_MODEL_LEN={args.max_model_len} in the notebook.")
                return 0
            if args.dry_run:
                print("Preview only; not a readiness check.")
                if args.backend == "local":
                    LocalServingRunner().launch(cfg, port=args.port, background=args.background, dry_run=True)
                else:
                    ModalServingRunner().launch(cfg, deploy=args.deploy, dry_run=True)
                return 0
            tasks = TaskRunner()
            task = tasks.add_task(f"Start {args.backend} serving process")
            def launch(t):
                if args.backend == "local":
                    return LocalServingRunner().launch(cfg, port=args.port, background=args.background)
                return ModalServingRunner().launch(cfg, deploy=args.deploy)
            try:
                proc = launch(task) if args.backend == "modal" else tasks.run_task(task, launch)
            except (ValueError, RuntimeError, OSError) as exc:
                print(f"Serving launch failed: {exc}")
                return 1
            return finish_launch(proc, background=args.backend == "local" and args.background)

    try:
        result = run_interactive_wizard(hw_info)
        return result.get("exit_code", 0) if isinstance(result, dict) else 0
    except (ServingBlockedError, ValueError, RuntimeError, OSError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
