"""Local serving runner and zero-zombie process management."""
from __future__ import annotations

import json
import re
import os
import signal
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from flexsds.cli.output import print_output as print
except ModuleNotFoundError:
    from src.flexsds.cli.output import print_output as print


def finish_launch(proc, background: bool = False) -> int:
    """Propagate startup/foreground failure without equating a process with readiness."""
    if proc is None:
        print("Launch failed; no serving readiness established.")
        return 1
    output = getattr(proc, '__dict__', {}).get('_flexsds_output')
    if output is None:
        print("Serving process started; verifying endpoint readiness...")
    readiness = getattr(proc, '__dict__', {}).get('_flexsds_readiness')
    try:
        if readiness is not None:
            readiness()
        else:
            print("Endpoint readiness is not verified for this process.")
        session = getattr(proc, '__dict__', {}).get('_flexsds_session')
        if session and output:
            from flexsds.cli.serving_session import present_session
            present_session(proc, output, **session)
        code = proc.poll() if background else proc.wait()
        return code if isinstance(code, int) else 0
    except (RuntimeError, OSError, ValueError) as exc:
        if output:
            output.update('Serving failed')
            output.close()
            print(f"Full details: {output.log_path}")
        print(f"Serving readiness failed: {exc}")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 1
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 130
    finally:
        if output:
            output.close()


K_SELECTION_MODES = frozenset({"batch", "max_k"})


DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"


def default_served_model_name(serving_cfg: Dict[str, Any]) -> str:
    """Name the endpoint after its target and lane: <target>-adaptive, -static or -vanilla.

    The default Qwen2.5-7B target keeps the historical "qwen2.5-7b" prefix.
    """
    if serving_cfg.get("disable_speculative"):
        lane = "vanilla"
    elif serving_cfg.get("enable_adaptive", True):
        lane = "adaptive"
    else:
        lane = "static"
    target = str(serving_cfg.get("target_model") or DEFAULT_TARGET_MODEL)
    if target.startswith(("Qwen/Qwen2.5-7B-Instruct", "qwen/qwen2.5-7b-instruct")):
        base = "qwen2.5-7b"
    else:
        base = re.sub(r"[^a-z0-9.]+", "-", target.rstrip("/").split("/")[-1].lower()).strip("-") or "model"
    return f"{base}-{lane}"


class LocalServingRunner:
    """Khởi chạy và quản lý vòng đời máy chủ vLLM cục bộ với FlexSDS."""

    @staticmethod
    def _validate_execution_options(serving_cfg: Dict[str, Any]) -> None:
        enable_adaptive = bool(serving_cfg.get("enable_adaptive", True))
        disable_speculative = bool(serving_cfg.get("disable_speculative", False))
        static_k = serving_cfg.get("static_k")
        flush_every = serving_cfg.get("step_telemetry_flush_every")

        if disable_speculative and enable_adaptive:
            raise ValueError("Cannot enable adaptive scheduling when speculative decoding is disabled")
        if static_k is not None:
            if enable_adaptive or disable_speculative:
                raise ValueError(
                    "static_k is only valid for static speculative mode "
                    "(enable_adaptive=False, disable_speculative=False)"
                )
            if isinstance(static_k, bool) or not isinstance(static_k, int) or not 1 <= static_k <= 8:
                raise ValueError("static_k must be an integer between 1 and 8")
        if flush_every is not None:
            if isinstance(flush_every, bool) or not isinstance(flush_every, int) or flush_every < 1:
                raise ValueError("step_telemetry_flush_every must be a positive integer")
        k_selection = serving_cfg.get("k_selection")
        if k_selection is not None and k_selection not in K_SELECTION_MODES:
            raise ValueError(f"k_selection must be one of {sorted(K_SELECTION_MODES)}: {k_selection!r}")

    def describe_execution_contract(self, config: Dict[str, Any]) -> str:
        """Trả về nhãn hợp đồng thực thi chuẩn cho log đối chứng và dry-run."""
        serving_cfg = config.get("serving", config)
        self._validate_execution_options(serving_cfg)
        enable_adaptive = bool(serving_cfg.get("enable_adaptive", True))
        match_adaptive_runtime = bool(serving_cfg.get("match_adaptive_runtime", False))
        disable_speculative = bool(serving_cfg.get("disable_speculative", False))
        disable_prefix_caching = bool(serving_cfg.get("disable_prefix_caching", False))

        if enable_adaptive:
            label = "adaptive-eager-v1"
        elif disable_speculative:
            label = "vanilla-matched-eager-v1" if match_adaptive_runtime else "vanilla-default"
        else:
            label = "static-matched-eager-v1" if match_adaptive_runtime else "static-default"

        if disable_prefix_caching:
            label += "+no-prefix-cache"
        return label

    def build_command(self, config: Dict[str, Any], port: Optional[int] = None) -> List[str]:
        """Tạo danh sách đối số dòng lệnh chạy vLLM phục vụ mô hình."""
        serving_cfg = config.get("serving", config)
        self._validate_execution_options(serving_cfg)

        target_model = serving_cfg.get("target_model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        draft_model = serving_cfg.get("draft_model", "duckhoi/qwen2.5-7b-eagle3.1-speculator")
        draft_method = serving_cfg.get("draft_method", "eagle3")
        use_hetero = serving_cfg.get("use_heterogeneous_vocab", False)
        enable_adaptive = bool(serving_cfg.get("enable_adaptive", True))
        match_adaptive_runtime = bool(serving_cfg.get("match_adaptive_runtime", False))
        disable_speculative = bool(serving_cfg.get("disable_speculative", False))
        disable_prefix_caching = bool(serving_cfg.get("disable_prefix_caching", False))
        static_k = serving_cfg.get("static_k")

        effective_port = port if port is not None else serving_cfg.get("port", 8000)
        target_lower = target_model.lower()

        # Xác định quantization và dtype (T4 / sm_75 không hỗ trợ BF16 và awq_marlin)
        target_gpu = str(serving_cfg.get("target_gpu") or "").strip().upper()
        hw_cfg = serving_cfg.get("hardware") if isinstance(serving_cfg.get("hardware"), dict) else {}
        sm_arch = str(hw_cfg.get("sm_architecture") or "").strip().lower()
        is_turing_t4 = target_gpu == "T4" or sm_arch == "sm_75"
        explicit_dtype = serving_cfg.get("dtype")
        explicit_quant = serving_cfg.get("quantization")

        if "awq" in target_lower:
            quantization = explicit_quant or ("awq" if is_turing_t4 else "awq_marlin")
            dtype = explicit_dtype or ("half" if is_turing_t4 else "bfloat16")
        elif "gptq" in target_lower:
            quantization = explicit_quant or "gptq"
            dtype = explicit_dtype or "float16"
        else:
            quantization = explicit_quant or "none"
            dtype = explicit_dtype or ("half" if is_turing_t4 else "bfloat16")

        profile_data = None
        if serving_cfg.get('profile_path'):
            profile_data = json.loads(Path(serving_cfg['profile_path']).read_text())

        speculative_config: Optional[Dict[str, Any]] = None
        if not disable_speculative:
            if draft_method not in {'eagle', 'eagle3', 'draft_model'}:
                raise ValueError('Unsupported draft method')
            is_eagle = draft_method in {"eagle", "eagle3"}
            num_spec_tokens = static_k if (not enable_adaptive and static_k is not None) else 4
            speculative_config = {
                "method": draft_method,
                "model": draft_model,
                "num_speculative_tokens": num_spec_tokens,
                "draft_sample_method": "greedy",
            }
            if enable_adaptive or not is_eagle:
                speculative_config["num_speculative_tokens_per_batch_size"] = [[1, 32, num_spec_tokens]]

            if use_hetero:
                speculative_config["use_heterogeneous_vocab"] = True

        cmd: List[str] = [
            "vllm",
            "serve",
            target_model,
            "--host",
            "0.0.0.0",
            "--port",
            str(effective_port),
            "--served-model-name",
            serving_cfg.get("served_model_name") or default_served_model_name(serving_cfg),
            "--dtype",
            dtype,
            "--generation-config",
            "vllm",
            "--gpu-memory-utilization",
            str(serving_cfg.get("gpu_memory_utilization", 0.85)),
            "--max-model-len",
            str(serving_cfg.get("max_model_len", 4096)),
            "--max-num-seqs",
            str(serving_cfg.get("max_num_seqs", 32)),
        ]

        use_matched_eager = bool(enable_adaptive or match_adaptive_runtime)
        if use_matched_eager:
            cmd.extend([
                "--enforce-eager",
                "--compilation-config",
                json.dumps({"mode": 0}),
                "--async-scheduling",
            ])

        if disable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")

        if quantization != "none":
            cmd.extend(["--quantization", quantization])

        target_revision = serving_cfg.get('target_revision')
        draft_revision = serving_cfg.get('draft_revision')
        if target_revision and target_revision != 'local':
            cmd.extend(['--revision', target_revision])
        if speculative_config is not None and draft_revision and draft_revision != 'local':
            speculative_config['revision'] = draft_revision
        if profile_data and enable_adaptive:
            if target_revision and target_revision != 'local':
                if target_revision != profile_data['models']['target_revision']:
                    raise ValueError('Inspected target revision does not match the measured profile; recalibrate')
            else:
                cmd.extend(['--revision', profile_data['models']['target_revision']])
            if draft_revision and draft_revision != 'local' and draft_revision != profile_data['models']['draft_revision']:
                raise ValueError('Inspected draft revision does not match the measured profile; recalibrate')
            if speculative_config is not None:
                speculative_config['revision'] = profile_data['models']['draft_revision']

        # EAGLE uses the same pinned sequential dynamic-K path as the M4 treatment.
        if enable_adaptive and speculative_config is not None:
            cmd.extend([
                "--speculative-config", json.dumps(speculative_config),
                "--scheduler-cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
                "--distributed-executor-backend", "src.server.lineage_executor.LineageUniProcExecutor",
                "--per-request-spec-decode-metrics", "summary",
            ])
        elif speculative_config is not None:
            cmd.extend([
                "--speculative-config", json.dumps(speculative_config),
                "--per-request-spec-decode-metrics", "summary",
            ])

        return cmd

    def check_port_available(self, port: int) -> bool:
        """Kiểm tra xem cổng port có đang khả dụng không."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False

    def setup_environment(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Thiết lập các biến môi trường cho FlexSDS Adaptive Scheduler."""
        serving_cfg = config.get("serving", config)
        self._validate_execution_options(serving_cfg)
        enable_adaptive = bool(serving_cfg.get("enable_adaptive", True))
        match_adaptive_runtime = bool(serving_cfg.get("match_adaptive_runtime", False))
        env = dict(os.environ)
        resolved = Path(__file__).resolve()
        project_root = str(resolved.parents[3]) if len(resolved.parents) > 3 else "/root/project"
        existing_pythonpath = env.get("PYTHONPATH", "")
        if project_root not in existing_pythonpath:
            env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else project_root
        # Never inherit a different profile/controller from the launcher's shell.
        if enable_adaptive:
            from flexsds.core.cost_model import validate_measured_profile
            path = Path(serving_cfg.get("profile_path") or "")
            if not path.is_file():
                raise ValueError("Adaptive requires a measured profile artifact")
            data = json.loads(path.read_text())
            validate_measured_profile(data, target_model=serving_cfg.get("target_model"),
                                      draft_model=serving_cfg.get("draft_model"))
            if data.get('draft_method') != serving_cfg.get('draft_method', 'eagle3'):
                raise ValueError('Profile draft method mismatch')
            from flexsds.core.calibration import profile_batch_coverage
            max_seqs = int(serving_cfg.get('max_num_seqs', 32))
            top = profile_batch_coverage(data['grid']['batches'], max_seqs)
            if top is not None:
                print(f"Profile measured up to B={top}; batches above use the B={top} costs "
                      f"(max_num_seqs={max_seqs}).")
            env["FLEXSDS_PROFILE_PATH"] = str(path.resolve())
            env["ADAPTIVE_STRICT"] = "1"
            env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        else:
            for leaked in (
                "FLEXSDS_PROFILE_PATH",
                "ADAPTIVE_STRICT",
                "ADAPTIVE_STEP_TELEMETRY_PATH",
                "ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY",
                "ADAPTIVE_K_SELECTION",
                "ADAPTIVE_STARTUP_TELEMETRY_PATH",
                "ADAPTIVE_CONTROLLER_KWARGS",
            ):
                env.pop(leaked, None)
            if match_adaptive_runtime:
                env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            else:
                env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        env["ADAPTIVE_ENABLE_VRAM_GUARD"] = "1"
        env["ADAPTIVE_TELEMETRY_PATH"] = os.path.abspath(
            serving_cfg.get("telemetry_path", "telemetry/serving_trace.jsonl")
        )
        if enable_adaptive:
            trace_dir = Path(env['ADAPTIVE_TELEMETRY_PATH']).parent
            trace_dir.mkdir(parents=True, exist_ok=True)
            if bool(serving_cfg.get("enable_step_telemetry", True)):
                flush_every = serving_cfg.get("step_telemetry_flush_every")
                if flush_every is None:
                    flush_every = 64
                env['ADAPTIVE_STEP_TELEMETRY_PATH'] = str(trace_dir / 'serving_steps.jsonl')
                env['ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY'] = str(flush_every)
            else:
                env.pop('ADAPTIVE_STEP_TELEMETRY_PATH', None)
                env.pop('ADAPTIVE_STEP_TELEMETRY_FLUSH_EVERY', None)
            env['ADAPTIVE_STARTUP_TELEMETRY_PATH'] = str(trace_dir / 'serving_lineage.jsonl')
        env["ADAPTIVE_MAX_K"] = str(serving_cfg.get("max_k", 4))
        env["ADAPTIVE_MIN_BATCH_K"] = "0"
        env["ADAPTIVE_PER_REQUEST"] = "1"
        if enable_adaptive:
            env["ADAPTIVE_K_SELECTION"] = serving_cfg.get("k_selection") or "batch"
        return env

    def launch(
        self,
        config: Dict[str, Any],
        port: Optional[int] = None,
        background: bool = False,
        pid_file: str = ".flexsds.pid",
        dry_run: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Khởi chạy tiến trình máy chủ cục bộ."""
        effective_port = port if port is not None else config.get("serving", config).get("port", 8000)
        if not dry_run and not self.check_port_available(effective_port):
            print(f"[Error] Port {effective_port} is already in use; choose another port or run flexsds stop.")
            return None
        cmd = self.build_command(config, port=effective_port)
        env = self.setup_environment(config)

        if dry_run:
            print("\n[FlexSDS] Dry-run execution command:")
            print(" ".join(cmd))
            print(f"Execution contract: {self.describe_execution_contract(config)}")
            print(f"Environment: ADAPTIVE_ENABLE_VRAM_GUARD={env.get('ADAPTIVE_ENABLE_VRAM_GUARD')}")
            return None

        if config.get('serving', config).get('enable_adaptive', True):
            from flexsds.core.hardware import detect_hardware, matches_gpu_label
            actual = detect_hardware()
            data = json.loads(Path(config.get('serving', config)['profile_path']).read_text())
            if not actual.is_cuda_available or (actual.sm_architecture != data['metadata']['sm_architecture']
                                               or not matches_gpu_label(data['metadata']['device'], actual.device_name)):
                raise ValueError('Profile differs from actual serving GPU')

        if background and os.name == "nt":
            print("Background process-group serving is unsupported on Windows; use foreground serving.")
            return None

        try:
            if background:
                log_path = "flexsds_server.log"
                log_file = open(log_path, "a", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    preexec_fn=os.setsid,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                with open(pid_file, "w", encoding="utf-8") as f:
                    f.write(f"{proc.pid}\n")
                print(f"[FlexSDS] Server started in background (PID: {proc.pid}, Log: {log_path})")
                from flexsds.server.readiness import verify_endpoint
                def ready():
                    model = verify_endpoint(f"http://127.0.0.1:{effective_port}", process=proc)
                    print(f"Inference verified: http://127.0.0.1:{effective_port}/v1 (model={model})")
                proc._flexsds_readiness = ready
                return proc
            else:
                proc = subprocess.Popen(cmd, env=env)
                from flexsds.server.readiness import verify_endpoint
                def ready():
                    model = verify_endpoint(f"http://127.0.0.1:{effective_port}", process=proc)
                    print(f"Inference verified: http://127.0.0.1:{effective_port}/v1 (model={model})")
                proc._flexsds_readiness = ready
                return proc
        except FileNotFoundError:
            print("\n\033[31m[Error]\033[0m Server binary not found in PATH.")
            return None

    def stop(self, pid_file: str = ".flexsds.pid") -> bool:
        """Dừng nhóm tiến trình máy chủ vLLM và dọn dẹp sạch tiến trình con."""
        if os.name == "nt":
            print("Local process-group stop is unsupported on Windows; stop the foreground session.")
            return False
        if not os.path.exists(pid_file):
            return False

        try:
            with open(pid_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return False
                pid = int(content)

            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        finally:
            if os.path.exists(pid_file):
                os.remove(pid_file)

        return True
