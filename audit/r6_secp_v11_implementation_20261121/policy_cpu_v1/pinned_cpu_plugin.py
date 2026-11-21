"""Point existing CPU source tests at already verified local pinned bytes."""
import os
from pathlib import Path


def pytest_configure(config):
    from experiments import scalar_pinned_sources
    scalar_pinned_sources.CAPTURE=Path(os.environ["M4_VLLM_SOURCE_ROOT"])
