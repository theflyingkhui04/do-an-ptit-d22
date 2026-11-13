"""Standalone Modal serverless serving application for FlexSDS."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

APP_NAME = "flexsds-serving"

if modal is not None:
    app = modal.App(APP_NAME)

    # Đóng gói image phục vụ vLLM với đầy đủ mã nguồn và dependencies
    flexsds_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "vllm==0.30.0",
            "torch==2.13.0+cu130",
            "fastapi",
            "uvicorn",
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
        .add_local_dir("src", remote_path="/root/project/src")
        .add_local_file(
            "experiments/scalar_vllm_source_manifest.json",
            remote_path="/root/project/experiments/scalar_vllm_source_manifest.json",
        )
    )

    model_volume = modal.Volume.from_name("ptit-model-cache", create_if_missing=True)

    @app.function(
        image=flexsds_image,
        volumes={"/root/.cache/huggingface": model_volume},
        timeout=3600,
    )
    def serve(
        gpu: str = "L4",
        target: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        draft: str = "duckhoi/qwen2.5-7b-eagle3.1-speculator",
    ):
        """Khởi chạy vLLM server trên GPU của Modal Cloud và duy trì phục vụ."""
        print(f"[FlexSDS Modal] Khởi chạy máy chủ trên GPU {gpu}: Target={target}, Draft={draft}")

        env = dict(os.environ)
        env["PYTHONPATH"] = f"/root/project:{env.get('PYTHONPATH', '')}"
        env["ADAPTIVE_ENABLE_VRAM_GUARD"] = "1"

        speculative_cfg = {
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": 4,
            "use_heterogeneous_vocab": True,
            "draft_sample_method": "greedy",
        }

        cmd = [
            "vllm", "serve", target,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--served-model-name", "qwen2.5-7b-adaptive",
            "--quantization", "awq",
            "--dtype", "half",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "4096",
            "--speculative-config", json.dumps(speculative_cfg),
            "--scheduler-cls", "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
            "--distributed-executor-backend", "src.server.lineage_executor.LineageUniProcExecutor",
            "--per-request-spec-decode-metrics", "summary",
        ]

        print(f"[FlexSDS Modal] Thực thi tiến trình vLLM: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env)

        # Giữ tiến trình phục vụ sống và giám sát lỗi
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        raise RuntimeError(f"Máy chủ vLLM trên Modal đã dừng bất ngờ với mã lỗi: {ret}")
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[FlexSDS Modal] Nhận tín hiệu dừng, đang tắt máy chủ vLLM...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

        return {"status": "stopped", "code": proc.returncode}
else:
    app = None
