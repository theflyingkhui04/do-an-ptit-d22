"""Offline unit tests for M2-03; no GPU or running server required."""
from __future__ import annotations

import copy
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_lossless as v  # noqa: E402
from experiments import lossless_verifier as shared_verifier  # noqa: E402


def result(lane="vanilla", ids=None, prompt_ids=None, *, content="same", ok=True):
    return {
        "workload": "W1", "sample_id": "sample-1", "dataset_version": "v1",
        "prompt_version": "p1", "request_sha256": "f" * 64,
        "lane": lane, "ok": ok,
        "token_ids": [101, 202, 303] if ids is None else ids,
        "prompt_token_ids": [11, 22] if prompt_ids is None else prompt_ids,
        "finish_reason": "stop", "content": content,
    }


class ComparatorTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(v.compare_pair(result(), result("adaptive"))["passed"])

    def test_same_text_different_token_ids_fails(self):
        got = v.compare_pair(result(), result("adaptive", ids=[101, 404, 303]))
        self.assertFalse(got["passed"])
        self.assertIn("generated_token_ids_mismatch", got["failures"])
        self.assertEqual(got["first_mismatch"]["position_zero_based"], 1)
        self.assertEqual(got["first_mismatch"]["vanilla_token_id"], 202)
        self.assertEqual(got["first_mismatch"]["adaptive_token_id"], 404)

    def test_compare_pair_delegates_to_shared_verifier(self):
        """The harness must delegate token comparisons without changing JSON contract."""
        vanilla = result("vanilla", ids=[101, 202, 303])
        adaptive = result("adaptive", ids=[101, 404, 303])
        original = shared_verifier.verify_token_sequence

        # Patch the module attribute; the implementation must look up the
        # function as shared_verifier.verify_token_sequence(...).
        with patch.object(shared_verifier, "verify_token_sequence", wraps=original) as spy:
            compared = v.compare_pair(vanilla, adaptive)

        # Before integration, compare_pair used its own comparison loop.
        # This assertion guards against reintroducing that duplication.
        spy.assert_called_once()

        # Accept both positional and keyword arguments for the two sequences.
        bound = inspect.signature(original).bind_partial(
            *spy.call_args.args, **spy.call_args.kwargs
        )
        self.assertEqual(bound.arguments["vanilla_tokens"], vanilla["token_ids"])
        self.assertEqual(bound.arguments["adaptive_tokens"], adaptive["token_ids"])

        # The public canonical summary shape must not drift.
        self.assertFalse(compared["passed"])
        self.assertIn("generated_token_ids_mismatch", compared["failures"])
        self.assertEqual(compared["first_mismatch"], {
            "position_zero_based": 1,
            "vanilla_token_id": 202,
            "adaptive_token_id": 404,
        })

    def test_length_mismatch_fails(self):
        got = v.compare_pair(result(), result("adaptive", ids=[101, 202]))
        self.assertFalse(got["passed"])
        self.assertEqual(got["first_mismatch"]["position_zero_based"], 2)
        self.assertIsNone(got["first_mismatch"]["adaptive_token_id"])

    def test_missing_ids_fails(self):
        got = v.compare_pair(result(), result("adaptive", ids=[]))
        self.assertFalse(got["passed"])

    def test_boolean_ids_rejected(self):
        with self.assertRaises(v.VerificationError):
            v.validate_token_ids([True, 2], "test")

    def test_string_ids_rejected(self):
        with self.assertRaises(v.VerificationError):
            v.validate_token_ids(["1", 2], "test")

    def test_prompt_id_mismatch_fails(self):
        got = v.compare_pair(result(), result("adaptive", prompt_ids=[11, 33]))
        self.assertIn("prompt_token_ids_mismatch", got["failures"])

    def test_request_failure_fails(self):
        got = v.compare_pair(result(), result("adaptive", ok=False))
        self.assertIn("request_or_response_failed", got["failures"])

    def test_input_hash_mismatch_fails(self):
        ada = result("adaptive")
        ada["request_sha256"] = "e" * 64
        self.assertIn("identity_mismatch:request_sha256",
                      v.compare_pair(result(), ada)["failures"])

    def test_duplicate_and_missing_records_fail(self):
        record = result()
        keys = {("W1", "sample-1"), ("W1", "sample-2")}
        with self.assertRaisesRegex(v.VerificationError, "Missing"):
            v.index_records([record], keys)
        with self.assertRaisesRegex(v.VerificationError, "Duplicate"):
            v.index_records([record, copy.deepcopy(record)], {("W1", "sample-1")})

    def test_truncated_prefix_not_claimed_full_completion(self):
        ref = result()
        ada = result("adaptive")
        ref["finish_reason"] = "length"
        ada["finish_reason"] = "length"
        got = v.compare_pair(ref, ada)
        self.assertTrue(got["passed"])
        self.assertTrue(got["truncated"])

    def test_mismatched_finish_reason_fails(self):
        ada = result("adaptive")
        ada["finish_reason"] = "length"
        self.assertIn("finish_reason_mismatch",
                      v.compare_pair(result(), ada)["failures"])

    def test_payload_keeps_frozen_generation_settings(self):
        prompt = {"canonical_request": {
            "model": "qwen2.5:7b-instruct", "temperature": 0.0,
            "seed": 42, "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        }}
        before = copy.deepcopy(prompt)
        payload = v.make_payload(prompt, "adaptive")
        self.assertEqual(payload["model"], "qwen2.5-7b-adaptive")
        self.assertTrue(payload["return_token_ids"])
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(prompt, before)


class ResponseContractTests(unittest.TestCase):
    def body(self):
        return {
            "id": "chatcmpl-1", "model": "qwen2.5-7b-vanilla",
            "prompt_token_ids": [1, 2], "usage": {"completion_tokens": 2},
            "choices": [{"index": 0, "token_ids": [3, 4],
                         "finish_reason": "stop", "message": {"content": "hi"}}],
        }

    def test_parse_valid_response(self):
        parsed = v.parse_completion(self.body(), "vanilla")
        self.assertEqual(parsed["token_ids"], [3, 4])
        self.assertEqual(parsed["prompt_token_ids"], [1, 2])

    def test_missing_token_ids_rejected(self):
        b = self.body()
        del b["choices"][0]["token_ids"]
        with self.assertRaises(v.VerificationError):
            v.parse_completion(b, "vanilla")

    def test_completion_usage_length_mismatch_rejected(self):
        b = self.body()
        b["usage"]["completion_tokens"] = 3
        with self.assertRaises(v.VerificationError):
            v.parse_completion(b, "vanilla")

    def test_wrong_model_rejected(self):
        b = self.body()
        b["model"] = "unexpected-model"
        with self.assertRaises(v.VerificationError):
            v.parse_completion(b, "vanilla")

    def test_missing_prompt_token_ids_rejected(self):
        b = self.body()
        del b["prompt_token_ids"]
        with self.assertRaises(v.VerificationError):
            v.parse_completion(b, "vanilla")

    def test_request_one_records_http_failure(self):
        sess = MagicMock()
        sess.post.side_effect = v.requests.Timeout("timeout")
        prompt = {"workload": "W1", "sample_id": "s", "dataset_version": "d",
                  "prompt_version": "p", "request_sha256": "abc",
                  "canonical_request": {"model": "x", "temperature": 0.0,
                                        "seed": 42, "max_tokens": 2,
                                        "messages": [{"role": "user", "content": "hi"}]}}
        got = v.request_one(sess, prompt, "adaptive")
        self.assertFalse(got["ok"])
        self.assertIn("Timeout", got["error"])

    def test_provenance_change_rejected(self):
        runtime = {name: "x" for name in (
            "git_sha", "verification_script_sha256", "workload_adapter_sha256",
            "launcher_sha256", "implementation_sha256", "vllm_python_package_version",
            "torch_python_package_version", "torch_runtime_version",
            "torch_cuda_version", "gpu_name", "driver_version")}
        old = {"plan_sha256": "abc", "runtime": copy.deepcopy(runtime)}
        v.ensure_same_provenance(old, runtime, "abc")
        with self.assertRaisesRegex(v.VerificationError, "git_sha"):
            v.ensure_same_provenance(old, {**runtime, "git_sha": "y"}, "abc")


class EndToEndContractTests(unittest.TestCase):
    """Mock the HTTP/GPU boundary; exercise both stages and real on-disk JSONL."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.patcher = patch.multiple(v, RUN_ROOT=root / "runs",
                                      SUMMARY_PATH=root / "results/summary.json")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.runtime = {name: "x" for name in (
            "git_sha", "verification_script_sha256", "workload_adapter_sha256",
            "launcher_sha256", "implementation_sha256", "vllm_python_package_version",
            "torch_python_package_version", "torch_runtime_version",
            "torch_cuda_version", "gpu_name", "driver_version")}
        self.prompts = []
        for workload, n in (("W1", 30), ("W2", 12), ("W3", 12)):
            for i in range(n):
                sample_id = f"{workload.lower()}-{i:03d}"
                self.prompts.append({
                    "workload": workload, "sample_id": sample_id,
                    "dataset_version": "version", "prompt_version": "prompt",
                    "request_sha256": f"{workload}-{i}",
                    "canonical_request": {"model": "target", "seed": 42,
                                          "temperature": 0.0, "max_tokens": 12,
                                          "messages": [{"role": "user", "content": sample_id}]},
                })
        self.plan = {"prompts": self.prompts, "sources": {}, "plan_sha256": "fixed-hash"}

    def fake_request(self, prompt, lane, mismatch=False):
        out = {k: prompt[k] for k in ("workload", "sample_id", "dataset_version",
                                       "prompt_version", "request_sha256")}
        out.update({"lane": lane, "ok": True, "response_id": "abc",
                    "token_ids": [12, 23], "prompt_token_ids": [7, 8],
                    "finish_reason": "stop", "content": "hello"})
        if mismatch and lane == "adaptive" and prompt["sample_id"] == "w3-011":
            out["token_ids"] = [12, 24]
        return out

    def test_fifty_four_pairs_pass_then_one_mismatch_fails(self):
        with (patch.object(v, "load_plan", return_value=self.plan),
              patch.object(v, "collect_runtime", return_value=self.runtime),
              patch.object(v, "probe_server", return_value={"model_root": "target"}),
              patch.object(v, "request_one", side_effect=lambda sess, p, lane:
                           self.fake_request(p, lane))):
            self.assertEqual(v.reference_stage("mock-run"), 0)
            self.assertEqual(v.verify_stage("mock-run"), 0)
        saved = v.load_json(v.SUMMARY_PATH)
        self.assertTrue(saved["token_id_gate_passed"])
        self.assertEqual(saved["exact_sequence_matches"], 54)
        self.assertEqual(saved["valid_pairs"], 54)
        self.assertFalse(saved["task_done"])

        # New run ID: never reuse/adapt results or reference from previous run.
        with (patch.object(v, "load_plan", return_value=self.plan),
              patch.object(v, "collect_runtime", return_value=self.runtime),
              patch.object(v, "probe_server", return_value={"model_root": "target"}),
              patch.object(v, "request_one", side_effect=lambda sess, p, lane:
                           self.fake_request(p, lane, mismatch=True))):
            self.assertEqual(v.reference_stage("mock-mismatch"), 0)
            self.assertEqual(v.verify_stage("mock-mismatch"), 1)
        failed = v.load_json(v.SUMMARY_PATH)
        self.assertFalse(failed["token_id_gate_passed"])
        self.assertEqual(failed["exact_sequence_matches"], 53)
        self.assertEqual(failed["comparisons"][-1]["first_mismatch"]["position_zero_based"], 1)

    def test_reference_tamper_fails_before_adaptive(self):
        with (patch.object(v, "load_plan", return_value=self.plan),
              patch.object(v, "collect_runtime", return_value=self.runtime),
              patch.object(v, "probe_server", return_value={"model_root": "target"}),
              patch.object(v, "request_one", side_effect=lambda sess, p, lane:
                           self.fake_request(p, lane)) as sender):
            self.assertEqual(v.reference_stage("mock-tamper"), 0)
            journal = v.RUN_ROOT / "mock-tamper/vanilla.jsonl"
            journal.write_text(journal.read_text() + "{}\n", encoding="utf-8")
            self.assertEqual(v.verify_stage("mock-tamper"), 1)
            self.assertEqual(sender.call_count, 54)  # no adaptive calls
        failed = v.load_json(v.SUMMARY_PATH)
        self.assertFalse(failed["token_id_gate_passed"])
        self.assertIn("hash mismatch", failed["fatal_error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
