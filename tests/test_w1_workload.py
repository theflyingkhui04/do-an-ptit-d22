"""Tests for frozen W1 workload artifacts and helper views."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.pipeline.w1_workload import load_jsonl, request_semantic_view


W1_DIR = Path("experiments/configs/w1")


class W1WorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        if not (W1_DIR / "samples.jsonl").exists():
            self.skipTest("W1 artifacts have not been generated yet")

    def test_samples_are_model_facing_and_label_free(self) -> None:
        samples = load_jsonl(W1_DIR / "samples.jsonl")

        self.assertGreaterEqual(len(samples), 30)
        self.assertEqual(len({sample["sample_id"] for sample in samples}), len(samples))
        for sample in samples:
            self.assertNotIn("labels", sample)
            self.assertNotIn("key_text_pairs", sample)
            self.assertTrue(sample["raw_text"])

    def test_requests_match_samples_and_do_not_leak_audit_labels(self) -> None:
        samples = load_jsonl(W1_DIR / "samples.jsonl")
        requests = load_jsonl(W1_DIR / "node1_chat_requests.jsonl")

        self.assertEqual([row["sample_id"] for row in requests], [row["sample_id"] for row in samples])
        samples_by_id = {sample["sample_id"]: sample for sample in samples}
        for row in requests:
            sample = samples_by_id[row["sample_id"]]
            self.assertEqual(row["expected_source"]["raw_text"], sample["raw_text"])
            rendered_messages = "\n".join(
                message["content"] for message in row["request"]["messages"]
            )
            self.assertNotIn("key_text_pairs", rendered_messages)
            self.assertNotIn('"labels"', rendered_messages)
            self.assertNotIn("source_label", row["request"]["messages"][1]["content"])

    def test_request_semantic_view_is_stable_for_harness(self) -> None:
        request = load_jsonl(W1_DIR / "node1_chat_requests.jsonl")[0]
        view = request_semantic_view(request)

        self.assertEqual(view["schema_version"], "invoice_extraction.v1")
        self.assertEqual(view["prompt_version"], "node1_extractor.v1")
        self.assertEqual(view["temperature"], 0.0)
        self.assertEqual(view["seed"], 42)
        self.assertEqual(view["response_format"], {"type": "json_object"})
        self.assertEqual(len(view["system_prompt_sha256"]), 64)
        self.assertEqual(len(view["user_content_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
