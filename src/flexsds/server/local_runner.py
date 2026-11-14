"""Local serving runner and zero-zombie process management."""
from __future__ import annotations

import json
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
    print("Serving process started; endpoint readiness is not verified.")
    try:
        code = proc.poll() if background else proc.wait()
        return code if isinstance(code, int) else 0
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 130


class LocalServingRunner:
    """Khởi chạy và quản lý vòng đời máy chủ vLLM cục bộ với FlexSDS."""

    def build_command(self, config: Dict[str, Any], port: Optional[int] = None) -> List[str]:
        """Tạo danh sách đối số dòng lệnh chạy vLLM phục vụ mô hình."""
        serving_cfg = config.get("serving", config)
        target_model = serving_cfg.get("target_model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        draft_model = serving_cfg.get("draft_model", "duckhoi/qwen2.5-7b-eagle3.1-speculator")
        draft_method = serving_cfg.get("draft_method", "eagle3")
        use_hetero = serving_cfg.get("use_heterogeneous_vocab", True)

        effective_port = port if port is not None else serving_cfg.get("port", 8000)
        target_lower = target_model.lower()

        # Xác định quantization và dtype
        if "awq" in target_lower:
            quantization = "awq_marlin"
            dtype = "bfloat16"
        elif "gptq" in target_lower:
            quantization = "gptq"
            dtype = "float16"
        else:
            quantization = serving_cfg.get("quantization", "none")
            dtype = serving_cfg.get("dtype", "bfloat16")

        speculative_config: Dict[str, Any] = {
            "method": draft_method,
            "model": draft_model,
            "num_speculative_tokens": 4,
            "num_speculative_tokens_per_batch_size": [[1, 32, 4]],
            "draft_sample_method": "greedy",
        }
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
            serving_cfg.get("served_model_name", "qwen2.5-7b-adaptive"),
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

        if quantization != "none":
            cmd.extend(["--quantization", quantization])

        if serving_cfg.get("enable_adaptive", True):
            cmd.extend([
                "--speculative-config", json.dumps(speculative_config),
                "--scheduler-cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
                "--distributed-executor-backend", "src.server.lineage_executor.LineageUniProcExecutor",
                "--per-request-spec-decode-metrics", "summary",
            ])
        else:
            # Static speculation uses vLLM's default scheduler/executor, not the adaptive pair.
            speculative_config.pop("num_speculative_tokens_per_batch_size", None)
            cmd.extend(["--speculative-config", json.dumps(speculative_config)])

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
        env = dict(os.environ)
        project_root = str(Path(__file__).resolve().parents[3])
        existing_pythonpath = env.get("PYTHONPATH", "")
        if project_root not in existing_pythonpath:
            env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else project_root
        env["ADAPTIVE_ENABLE_VRAM_GUARD"] = "1"
        env["ADAPTIVE_TELEMETRY_PATH"] = os.path.abspath(
            serving_cfg.get("telemetry_path", "telemetry/serving_trace.jsonl")
        )
        env["ADAPTIVE_MAX_K"] = str(serving_cfg.get("max_k", 4))
        env["ADAPTIVE_MIN_BATCH_K"] = "0"
        env["ADAPTIVE_PER_REQUEST"] = "1"
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
        cmd = self.build_command(config, port=effective_port)
        env = self.setup_environment(config)

        if dry_run:
            print("\n[FlexSDS] Dry-run execution command:")
            print(" ".join(cmd))
            print(f"Environment: ADAPTIVE_ENABLE_VRAM_GUARD={env.get('ADAPTIVE_ENABLE_VRAM_GUARD')}")
            return None

        if not self.check_port_available(effective_port):
            print(f"\n\033[31m[Error]\033[0m Port {effective_port} is already in use by another process.")
            print(f"  Tip: Use '--port <other_port>' or run: flexsds stop")
            return None

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
                return proc
            else:
                return subprocess.Popen(cmd, env=env)
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
