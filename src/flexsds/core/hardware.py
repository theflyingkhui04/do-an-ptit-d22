"""Hardware detection module for FlexSDS."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModalGpuSpec:
    """Authoritative specification and pricing for one Modal GPU option."""
    name: str
    tier: str
    vram_gb: int
    vram_budget_mb: float
    sm_architecture: str
    modal_gpu: str
    price_per_sec: float
    price_per_hour: float
    summary: str


# Pricing reference: https://modal.com/pricing
MODAL_GPU_SPECS: tuple[ModalGpuSpec, ...] = (
    ModalGpuSpec(
        name="T4",
        tier="Low",
        vram_gb=16,
        vram_budget_mb=15000.0,
        sm_architecture="sm_75",
        modal_gpu="T4",
        price_per_sec=0.000164,
        price_per_hour=0.59,
        summary="Entry GPU; check dtype and memory fit.",
    ),
    ModalGpuSpec(
        name="A10",
        tier="Medium",
        vram_gb=24,
        vram_budget_mb=22000.0,
        sm_architecture="sm_86",
        modal_gpu="A10",
        price_per_sec=0.000306,
        price_per_hour=1.10,
        summary="Recommended for 7B AWQ + EAGLE.",
    ),
    ModalGpuSpec(
        name="L40S",
        tier="High",
        vram_gb=48,
        vram_budget_mb=46000.0,
        sm_architecture="sm_89",
        modal_gpu="L40S",
        price_per_sec=0.000542,
        price_per_hour=1.95,
        summary="Balanced 48 GB GPU for LLM serving.",
    ),
    ModalGpuSpec(
        name="A100",
        tier="High",
        vram_gb=80,
        vram_budget_mb=78000.0,
        sm_architecture="sm_80",
        modal_gpu="A100-80GB",
        price_per_sec=0.000694,
        price_per_hour=2.50,
        summary="Max VRAM (Modal A100-80GB).",
    ),
)

MODAL_GPU_NAMES: tuple[str, ...] = tuple(spec.name for spec in MODAL_GPU_SPECS)
DEFAULT_MODAL_GPU: str = "A10"
DEFAULT_MODAL_GPU_INDEX: int = MODAL_GPU_NAMES.index(DEFAULT_MODAL_GPU)
_MODAL_GPU_BY_NAME: dict[str, ModalGpuSpec] = {spec.name: spec for spec in MODAL_GPU_SPECS}

COLAB_GPU_MAP: dict[str, tuple[str, float]] = {
    "T4": ("sm_75", 15000.0),
    "A10": ("sm_86", 22000.0),
    "L4": ("sm_89", 23000.0),
    "L40S": ("sm_89", 46000.0),
    "A100": ("sm_80", 78000.0),
}


def modal_gpu_choice_label(spec: ModalGpuSpec) -> str:
    """Format one Modal GPU option for interactive CLI menus."""
    suffix = " (Recommended)" if spec.name == DEFAULT_MODAL_GPU else ""
    return f"{spec.name} [{spec.tier}]{suffix}"


MODAL_GPU_CHOICES: list[str] = [modal_gpu_choice_label(spec) for spec in MODAL_GPU_SPECS]


def modal_gpu_footer(spec: ModalGpuSpec) -> str:
    """Format single-line CLI footer with tier, VRAM, SM, and Modal pricing."""
    return (
        f"[{spec.tier}] {spec.vram_gb} GB VRAM ({spec.sm_architecture}) · "
        f"${spec.price_per_sec:.6f}/s (~${spec.price_per_hour:.2f}/h) · {spec.summary}"
    )


# Legacy names accepted for old configs and scripts (Modal now sells the A10G class as "A10").
_MODAL_GPU_ALIASES: dict[str, str] = {"A10G": "A10"}


def canonical_modal_gpu_name(gpu: str) -> str:
    """Upper-case catalog name for a Modal GPU, mapping legacy aliases (A10G -> A10)."""
    token = str(gpu or "").strip().split()[0].upper() if str(gpu or "").strip() else ""
    return _MODAL_GPU_ALIASES.get(token, token)


def get_modal_gpu_spec(gpu: str) -> ModalGpuSpec:
    """Resolve a canonical Modal GPU name or wizard menu choice to its spec."""
    token = canonical_modal_gpu_name(gpu)
    spec = _MODAL_GPU_BY_NAME.get(token)
    if spec is None:
        supported = ", ".join(MODAL_GPU_NAMES)
        if token == "L4":
            raise ValueError(
                f"Unsupported Modal GPU: 'L4'. L4 is not available on Modal for FlexSDS; supported: {supported}"
            )
        raise ValueError(f"Unsupported Modal GPU: {gpu!r}. Supported: {supported}")
    return spec


def device_matches_exactly(gpu: str, device_name: str) -> bool:
    """Exact CUDA device name accepted when measuring or calibrating on a Modal GPU.

    Unlike matches_gpu_label (lenient, for serving), A10 accepts only "NVIDIA A10": Modal's A10
    class also hands out "NVIDIA A10G" (AWS, 300 W), which is measurably slower.
    """
    name = str(device_name or "").strip()
    token = canonical_modal_gpu_name(gpu)
    if token == "T4":
        return name in {"Tesla T4", "NVIDIA T4"}
    if token in {"A10", "L40S"}:
        return name == f"NVIDIA {token}"
    if token == "A100":
        return name.startswith("NVIDIA A100") and "80GB" in name.upper()
    return False


def modal_gpu_resource(gpu: str) -> str:
    """Return the exact Modal SDK gpu= identifier (e.g. A100 -> A100-80GB)."""
    return get_modal_gpu_spec(gpu).modal_gpu


def remote_gpu_budget_mb(backend: str, target_gpu: str) -> float:
    """Return conservative remote VRAM budget in MiB for preflight checks."""
    key = str(target_gpu or "").strip().split()[0].upper() if str(target_gpu or "").strip() else ""
    if backend == "modal":
        spec = _MODAL_GPU_BY_NAME.get(key)
        return spec.vram_budget_mb if spec is not None else 0.0
    if backend == "colab":
        entry = COLAB_GPU_MAP.get(key)
        return entry[1] if entry is not None else 0.0
    return 0.0


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
    gpu_match_label: Optional[str] = None

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


def build_remote_hardware_info(backend: str, target_gpu: str) -> HardwareInfo:
    """Build synthetic remote HardwareInfo for Modal or Colab profile lookup."""
    if backend == "modal":
        spec = get_modal_gpu_spec(target_gpu)
        return HardwareInfo(
            device_name=f"Remote {spec.name}",
            sm_architecture=spec.sm_architecture,
            total_vram_mb=spec.vram_budget_mb,
            free_vram_mb=spec.vram_budget_mb,
            cuda_version="12.4",
            is_cuda_available=True,
            device_count=1,
            gpu_match_label=spec.modal_gpu,
        )
    key = str(target_gpu or "").strip().upper()
    target_sm, target_vram = COLAB_GPU_MAP.get(key, ("sm_89", 22000.0))
    return HardwareInfo(
        device_name=f"Remote {target_gpu}",
        sm_architecture=target_sm,
        total_vram_mb=target_vram,
        free_vram_mb=target_vram,
        cuda_version="12.4",
        is_cuda_available=True,
        device_count=1,
    )


def matches_gpu_label(label: str, device_name: str) -> bool:
    """Match platform labels to CUDA names without treating equal SM as identity.

    Modal's A10 resource (and legacy A10G profiles) report NVIDIA A10 through CUDA.
    RTX 3090 shares sm_86, L4 shares sm_89 with L40S, and A100-40GB shares sm_80
    with A100-80GB — none may cross-match.
    """
    actual = (
        device_name.upper()
        .strip()
        .removeprefix("REMOTE ")
        .removeprefix("NVIDIA ")
        .removeprefix("TESLA ")
        .strip()
    )
    label = (
        label.upper()
        .strip()
        .removeprefix("REMOTE ")
        .removeprefix("NVIDIA ")
        .removeprefix("TESLA ")
        .strip()
    )
    if label in {"A10", "A10G"}:
        return actual in {"A10", "A10G"}
    if label == "A100-80GB":
        return (actual == "A100" or actual.startswith(("A100-", "A100 "))) and "80GB" in actual
    if label == "A100-40GB":
        return (actual == "A100" or actual.startswith(("A100-", "A100 "))) and "40GB" in actual
    if label == "A100":
        return actual == "A100" or actual.startswith(("A100-", "A100 "))
    return actual == label
