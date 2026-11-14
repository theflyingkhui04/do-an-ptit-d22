"""Hardware detection module for FlexSDS."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """Hardware specification details for adaptive serving."""
    device_name: str
    sm_architecture: str
    total_vram_mb: float
    free_vram_mb: float
    cuda_version: str
    is_cuda_available: bool
    device_count: int = 0

    @property
    def free_vram_gb(self) -> float:
        return round(self.free_vram_mb / 1024.0, 2)

    @property
    def total_vram_gb(self) -> float:
        return round(self.total_vram_mb / 1024.0, 2)

    def summary(self) -> str:
        if not self.is_cuda_available:
            return f"CPU Only · RAM Free: {self.free_vram_gb} GB"
        return f"{self.device_name} ({self.sm_architecture}) · VRAM: {self.free_vram_gb}/{self.total_vram_gb} GB Free"


def detect_hardware(device_index: int = 0) -> HardwareInfo:
    """Detects active GPU hardware details without initializing heavy CUDA context."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            dev_idx = min(device_index, torch.cuda.device_count() - 1)
            name = torch.cuda.get_device_name(dev_idx)
            major, minor = torch.cuda.get_device_capability(dev_idx)
            sm_arch = f"sm_{major}{minor}"

            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
            free_mb = free_bytes / (1024.0 * 1024.0)
            total_mb = total_bytes / (1024.0 * 1024.0)
            cuda_ver = torch.version.cuda or "N/A"

            return HardwareInfo(
                device_name=name,
                sm_architecture=sm_arch,
                total_vram_mb=round(total_mb, 1),
                free_vram_mb=round(free_mb, 1),
                cuda_version=cuda_ver,
                is_cuda_available=True,
                device_count=torch.cuda.device_count(),
            )
    except Exception:
        pass

    # CPU Fallback
    ram_mb = 16000.0
    try:
        import psutil
        ram_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
    except Exception:
        pass

    return HardwareInfo(
        device_name="CPU",
        sm_architecture="cpu",
        total_vram_mb=round(ram_mb, 1),
        free_vram_mb=round(ram_mb, 1),
        cuda_version="N/A",
        is_cuda_available=False,
        device_count=0,
    )


def matches_gpu_label(label: str, device_name: str) -> bool:
    """Match platform labels to CUDA names without treating equal SM as identity.

    Modal's A10G resource reports NVIDIA A10 through CUDA. RTX 3090 shares
    sm_86 but must never match that label.
    """
    actual = device_name.upper().strip().removeprefix('NVIDIA ').removeprefix('TESLA ')
    label = label.upper().strip().removeprefix('NVIDIA ').removeprefix('TESLA ')
    if label in {'A10', 'A10G'}:
        return actual in {'A10', 'A10G'}
    if label == 'A100':
        return actual == 'A100' or actual.startswith(('A100-', 'A100 '))
    return actual == label
