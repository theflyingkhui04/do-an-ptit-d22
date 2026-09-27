"""Unit tests for M1-01 smoke helper behavior that does not need a live server."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from smoke_m1_01_telemetry import (  # noqa: E402
    DEFAULT_PROMPTS,
    VRAM_IDLE_DELTA_LIMIT_MB,
    VRAM_PEAK_LIMIT_MB,
    NvmlSampler,
    prompt_for_index,
    runtime_info,
)


class M101SmokeHelpersTests(unittest.TestCase):
    def test_prompt_selection_cycles_through_ten_smoke_prompts(self) -> None:
        self.assertEqual(len(DEFAULT_PROMPTS), 10)
        self.assertEqual(prompt_for_index(0), DEFAULT_PROMPTS[0])
        self.assertEqual(prompt_for_index(10), DEFAULT_PROMPTS[0])

    def test_vram_thresholds_match_operational_gate(self) -> None:
        self.assertEqual(VRAM_PEAK_LIMIT_MB, 14336.0)
        self.assertEqual(VRAM_IDLE_DELTA_LIMIT_MB, 256.0)

    def test_nvml_sampler_degrades_without_gpu(self) -> None:
        sampler = NvmlSampler(gpu_index=0)

        self.assertIsNone(sampler.used_mb())
        self.assertFalse(sampler.available)

    def test_runtime_info_has_required_keys(self) -> None:
        sampler = NvmlSampler(gpu_index=0)
        info = runtime_info(sampler)

        self.assertIn("vllm_version", info)
        self.assertIn("gpu_name", info)
        self.assertIn("git_commit", info)


if __name__ == "__main__":
    unittest.main()
