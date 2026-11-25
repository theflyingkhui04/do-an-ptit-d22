"""Modal HTTP app; configuration is passed by the CLI, never embedded credentials.

Development: modal serve. Persistent service: modal deploy.
The web_server decorator owns startup readiness and endpoint exposure.
"""
from __future__ import annotations
import json
import base64
import os
import sys
import subprocess
from pathlib import Path

# Ensure flexsds package root is in sys.path when executed directly by modal CLI
resolved = Path(__file__).resolve()
if len(resolved.parents) > 2 and (resolved.parents[2] / "flexsds").exists():
    _pkg_root = str(resolved.parents[2])
    PROJECT_ROOT = resolved.parents[3]
else:
    _pkg_root = "/root/project/src"
    PROJECT_ROOT = Path("/root/project")

if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

try:
    from flexsds.core.hardware import DEFAULT_MODAL_GPU, MODAL_GPU_NAMES, modal_gpu_resource
    from flexsds.server.local_runner import LocalServingRunner
except ModuleNotFoundError:
    try:
        from src.flexsds.core.hardware import DEFAULT_MODAL_GPU, MODAL_GPU_NAMES, modal_gpu_resource
        from src.flexsds.server.local_runner import LocalServingRunner
    except ModuleNotFoundError:
        from local_runner import LocalServingRunner
        DEFAULT_MODAL_GPU = "A10"
        MODAL_GPU_NAMES = ("T4", "A10", "L40S", "A100")
        def modal_gpu_resource(gpu: str) -> str:
            return {"T4": "T4", "A10": "A10", "L40S": "L40S", "A100": "A100-80GB"}[gpu]

APP_NAME = "flexsds-serving"
# Image.env becomes a Dockerfile ENV instruction. Raw JSON containing Windows
# paths can lose a layer of backslash escaping there. The local CLI still sends
# raw JSON; only the image transport uses base64 to survive Dockerfile parsing.
_encoded_config = os.environ.get("FLEXSDS_SERVING_CONFIG_B64")
SERVING_CONFIG = json.loads(
    base64.b64decode(_encoded_config, validate=True).decode("utf-8")
    if _encoded_config is not None else os.environ.get("FLEXSDS_SERVING_CONFIG", "{}")
)
PORT = int(SERVING_CONFIG.get("port", 8000))
GPU = SERVING_CONFIG.get("target_gpu", DEFAULT_MODAL_GPU)
if GPU not in MODAL_GPU_NAMES or not 1 <= PORT <= 65535:
    raise ValueError("Unsupported Modal GPU or serving port")
MODAL_GPU = modal_gpu_resource(GPU)

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    from flexsds.server.modal_image import serving_image
    app = modal.App(APP_NAME)
    container_config = dict(SERVING_CONFIG)
    if container_config.get("profile_path"):
        container_config["profile_path"] = "/root/project/hardware_profile.json"
    image = (
        serving_image()
        .env({"PYTHONPATH": "/root/project:/root/project/src",
              "FLEXSDS_SERVING_CONFIG_B64": base64.b64encode(
                  json.dumps(container_config).encode("utf-8")).decode("ascii")})
    )
    if modal.is_local():
        src_dir = PROJECT_ROOT / "src"
        if src_dir.exists():
            image = image.add_local_dir(str(src_dir), remote_path="/root/project/src")
        else:
            flexsds_dir = resolved.parents[1] if len(resolved.parents) > 1 else Path("/root/project/src/flexsds")
            image = image.add_local_dir(str(flexsds_dir), remote_path="/root/project/src/flexsds")
            # Wheels install these as siblings, but the runtime imports src.*.
            for package in ("server", "proposer"):
                image = image.add_local_dir(str(flexsds_dir.parent / package),
                                           remote_path=f"/root/project/src/{package}")

        experiments_dir = PROJECT_ROOT / "experiments"
        if not experiments_dir.exists():
            experiments_dir = Path(_pkg_root) / "experiments"
        if SERVING_CONFIG.get("enable_adaptive", True):
            for name in ("protocol_b_boundary_observer.py", "scalar_vllm_source_manifest.json",
                         "m4_eagle_source_manifest.json", "eagle_legacy_source_manifest.json"):
                if not (experiments_dir / name).is_file():
                    raise ValueError(f"Required adaptive runtime support file is missing: {name}")
        if experiments_dir.exists():
            image = image.add_local_dir(str(experiments_dir), remote_path="/root/project/experiments",
                                       ignore=["runs/**", "results/**", "__pycache__/**", "*.ipynb"])
        selected_profile = SERVING_CONFIG.get("profile_path")
        if selected_profile:
            profile = Path(selected_profile)
            if not profile.is_file():
                raise ValueError("Selected hardware profile artifact is missing")
            image = image.add_local_file(str(profile), remote_path="/root/project/hardware_profile.json")

    volume = modal.Volume.from_name("ptit-model-cache", create_if_missing=True)

    @app.function(image=image, gpu=MODAL_GPU, max_containers=1,
                  volumes={"/root/.cache/huggingface": volume}, timeout=3600)
    @modal.web_server(PORT, startup_timeout=600)
    def serve():
        runtime_config = dict(SERVING_CONFIG)
        if runtime_config.get("profile_path"):
            runtime_config["profile_path"] = "/root/project/hardware_profile.json"
        if runtime_config.get("enable_adaptive", True):
            import torch
            from flexsds.core.hardware import HardwareInfo, matches_gpu_label
            from flexsds.core.cost_model import validate_measured_profile
            sm = 'sm_%d%d' % torch.cuda.get_device_capability(0)
            actual_device = torch.cuda.get_device_name(0)
            data = json.loads(Path(runtime_config['profile_path']).read_text())
            if (not matches_gpu_label(MODAL_GPU, actual_device)
                    or not matches_gpu_label(data['metadata']['device'], actual_device)):
                raise ValueError('Profile differs from actual Modal GPU')
            hardware = HardwareInfo(device_name=f"Remote {GPU}", sm_architecture=sm,
                                    total_vram_mb=0, free_vram_mb=0, cuda_version=torch.version.cuda,
                                    is_cuda_available=True, gpu_match_label=MODAL_GPU)
            validate_measured_profile(data, hardware, runtime_config.get('target_model'), runtime_config.get('draft_model'))
        runner = LocalServingRunner()
        # Do not use the laptop's preflight here: the CLI evaluated the remote budget.
        # vLLM and Modal's web-server startup check remain the runtime authorities.
        subprocess.Popen(runner.build_command(runtime_config),
                         env=runner.setup_environment(runtime_config))
else:
    app = None
