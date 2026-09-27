"""Unit tests for M1 Interface A telemetry normalization and validation."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_m1_telemetry import (  # noqa: E402
    DEFAULT_SCHEMA,
    normalize_record,
    recompute_alpha,
    recompute_tau,
    validate_normalized_record,
)


def client_metrics() -> dict:
    return {"ttft_ms": 100.0, "tpot_ms": 20.0, "e2e_ms": 260.0, "output_tokens": 9}


def gpu_metrics() -> dict:
    return {
        "gpu_index": 0,
        "vram_peak_mb": 13820.0,
        "vram_idle_before_warmup_mb": 12480.0,
        "vram_idle_after_warmup_mb": 13200.0,
        "vram_idle_after_smoke_mb": 13240.0,
        "vram_idle_delta_mb": 40.0,
        "vram_peak_limit_mb": 14336.0,
        "vram_peak_pass": True,
        "vram_stability_pass": True,
    }


def runtime() -> dict:
    return {
        "vllm_version": "0.29.0",
        "torch_version": "2.x",
        "driver_version": "unknown",
        "gpu_name": "Tesla T4",
    }


def static_spec() -> dict:
    return {
        "mean_acceptance_length": 2.5,
        "draft_acceptance_rate": 0.5,
        "acceptance_histogram": [1, 1, 1, 1],
        "num_spec_steps": 4,
        "num_accepted_draft_tokens": 6,
        "num_draft_tokens": 12,
        "num_spec_tokens": 3,
    }


class M1TelemetryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(schema)

    def test_valid_static_record(self) -> None:
        record = normalize_record(
            client_request_id="static-1",
            server_request_id="chatcmpl-static",
            lane="static",
            model="qwen2.5-7b-static-draft",
            server_metrics={"metrics": {"speculative_decoding": static_spec()}},
            client_metrics=client_metrics(),
            gpu_metrics=gpu_metrics(),
            runtime=runtime(),
        )

        self.assertEqual(validate_normalized_record(record, self.validator), [])
        self.assertEqual(record["alpha"], 0.5)
        self.assertEqual(record["tau"], 2.5)

    def test_valid_vanilla_record(self) -> None:
        record = normalize_record(
            client_request_id="vanilla-1",
            server_request_id="chatcmpl-vanilla",
            lane="vanilla",
            model="qwen2.5-7b-vanilla",
            server_metrics=None,
            client_metrics=client_metrics(),
            gpu_metrics=gpu_metrics(),
            runtime=runtime(),
        )

        self.assertEqual(validate_normalized_record(record, self.validator), [])
        self.assertIsNone(record["alpha"])
        self.assertIsNone(record["tau"])
        self.assertIsNone(record["speculative_decoding"])

    def test_recomputes_alpha_and_tau_from_counters(self) -> None:
        spec = static_spec()

        self.assertEqual(recompute_alpha(spec), 0.5)
        self.assertEqual(recompute_tau(spec), 2.5)

    def test_rejects_wrong_static_k(self) -> None:
        spec = static_spec()
        spec["num_spec_tokens"] = 4

        with self.assertRaises(ValueError):
            normalize_record(
                client_request_id="bad-k",
                server_request_id="chatcmpl",
                lane="static",
                model="qwen2.5-7b-static-draft",
                server_metrics={"metrics": {"speculative_decoding": spec}},
                client_metrics=client_metrics(),
                gpu_metrics=gpu_metrics(),
                runtime=runtime(),
            )

    def test_rejects_histogram_counter_mismatch(self) -> None:
        spec = static_spec()
        spec["acceptance_histogram"] = [1, 1, 1, 2]

        with self.assertRaises(ValueError):
            normalize_record(
                client_request_id="bad-hist",
                server_request_id="chatcmpl",
                lane="static",
                model="qwen2.5-7b-static-draft",
                server_metrics={"metrics": {"speculative_decoding": spec}},
                client_metrics=client_metrics(),
                gpu_metrics=gpu_metrics(),
                runtime=runtime(),
            )

    def test_validator_rejects_static_alpha_mismatch(self) -> None:
        record = normalize_record(
            client_request_id="static-2",
            server_request_id="chatcmpl-static",
            lane="static",
            model="qwen2.5-7b-static-draft",
            server_metrics={"metrics": {"speculative_decoding": static_spec()}},
            client_metrics=client_metrics(),
            gpu_metrics=gpu_metrics(),
            runtime=runtime(),
        )
        bad = copy.deepcopy(record)
        bad["alpha"] = 0.25

        errors = validate_normalized_record(bad, self.validator)

        self.assertTrue(any("alpha" in error for error in errors))

    def test_validator_rejects_non_null_vanilla_alpha(self) -> None:
        record = normalize_record(
            client_request_id="vanilla-2",
            server_request_id="chatcmpl-vanilla",
            lane="vanilla",
            model="qwen2.5-7b-vanilla",
            server_metrics=None,
            client_metrics=client_metrics(),
            gpu_metrics=gpu_metrics(),
            runtime=runtime(),
        )
        bad = copy.deepcopy(record)
        bad["alpha"] = 0.0

        errors = validate_normalized_record(bad, self.validator)

        self.assertTrue(any("alpha" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
