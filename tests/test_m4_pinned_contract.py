import os
from pathlib import Path
import tempfile

import pytest

from experiments.m4_pinned_contract import verify


def test_pinned_eagle_shared_dynamic_k_contract():
    root = Path(os.environ.get("M4_VLLM_SOURCE_ROOT", str(Path(tempfile.gettempdir()) / "m4-02-06-vllm-0.30.0")))
    if not root.exists():
        pytest.skip("Set M4_VLLM_SOURCE_ROOT to exact vLLM 0.30.0 package parent; required before LOCAL_READY")
    result = verify(root)
    assert result["source_contract"] == "PASS" and result["verified_files"] == 19
