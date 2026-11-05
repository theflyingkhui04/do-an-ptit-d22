from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from benchmark_harness import run_requests  # noqa: E402
from workload_adapter import (  # noqa: E402
    WorkloadContractError,
    build_request_payload,
    build_workload_metadata,
    load_workload_rows,
    resolve_repeat_offset,
    resolve_sample_index,
    select_workload_row,
    validate_manifest_consistency,
    validate_workload_label,
    verify_workload_integrity,
)


class BenchmarkWorkloadAdapterTests(unittest.TestCase):
    def _write_jsonl(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            delete=False,
        )
        path = Path(tmp.name)
        with tmp:
            for row in rows:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.addCleanup(path.unlink, missing_ok=True)
        return path


    def _write_package(
        self,
        rows: list[dict],
        *,
        manifest_overrides: dict | None = None,
        expected_hash_override: str | None = None,
    ) -> tuple[Path, Path]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        package_dir = Path(tmpdir.name)
        workload_path = package_dir / "chat_requests.jsonl"
        workload_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

        with workload_path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()

        manifest = {
            "workload_id": rows[0]["workload_id"],
            "dataset_version": rows[0]["dataset_version"],
            "prompt_version": rows[0]["prompt_version"],
            "sample_count": len(rows),
            "model": rows[0]["request"]["model"],
            "temperature": rows[0]["request"]["temperature"],
            "max_tokens": rows[0]["request"]["max_tokens"],
            "seed": rows[0]["request"]["seed"],
            "package_sha256": "a" * 64,
            "artifacts": {
                "chat_requests_jsonl": "chat_requests.jsonl",
            },
            "file_hashes": {
                "chat_requests_jsonl": expected_hash_override or digest,
            },
        }

        if "response_format" in rows[0]["request"]:
            manifest["response_format"] = copy.deepcopy(
                rows[0]["request"]["response_format"]
            )

        if manifest_overrides:
            manifest.update(copy.deepcopy(manifest_overrides))

        manifest_path = package_dir / "package_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return workload_path, manifest_path

    @staticmethod
    def _row(sample_id: str, *, response_format: bool = False) -> dict:
        request = {
            "model": "qwen2.5:7b-instruct",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"prompt-{sample_id}"},
            ],
            "temperature": 0.0,
            "max_tokens": 96,
            "seed": 42,
        }
        if response_format:
            request["response_format"] = {"type": "json_object"}

        return {
            "workload_id": "w2_text_to_sql",
            "dataset_version": "w2.m1_03.v1",
            "sample_id": sample_id,
            "prompt_version": "w2_text_to_sql.v1",
            "request": request,
        }

    def test_load_valid_package(self) -> None:
        path = self._write_jsonl(
            [self._row("sample-001"), self._row("sample-002")]
        )
        rows = load_workload_rows(path)
        self.assertEqual([row["sample_id"] for row in rows], ["sample-001", "sample-002"])

    def test_duplicate_sample_id_fails(self) -> None:
        path = self._write_jsonl(
            [self._row("sample-001"), self._row("sample-001")]
        )
        with self.assertRaises(WorkloadContractError):
            load_workload_rows(path)

    def test_unknown_request_semantics_fail_fast(self) -> None:
        row = self._row("sample-001")
        row["request"]["top_p"] = 0.9
        path = self._write_jsonl([row])

        with self.assertRaises(WorkloadContractError):
            load_workload_rows(path)

    def test_workload_label_matches_package(self) -> None:
        self.assertEqual(
            validate_workload_label("W2", "w2_text_to_sql"),
            "W2",
        )
        self.assertEqual(
            validate_workload_label("w2", "w2_text_to_sql"),
            "W2",
        )

    def test_workload_label_package_mismatch_fails(self) -> None:
        with self.assertRaises(WorkloadContractError):
            validate_workload_label("W3", "w2_text_to_sql")

    def test_cycle_resolves_explicit_sample_index(self) -> None:
        self.assertEqual(
            resolve_sample_index(
                12,
                31,
                sample_policy="cycle",
            ),
            7,
        )

    def test_cycle_rotate_offsets_balance_w2_across_five_repeats(self) -> None:
        offsets = [
            resolve_repeat_offset(
                12,
                32,
                repeat_index,
                sample_policy="cycle-rotate",
            )
            for repeat_index in range(1, 6)
        ]
        self.assertEqual(offsets, [0, 8, 4, 0, 8])

        counts = [0] * 12
        for repeat_index in range(1, 6):
            sample_offset = resolve_repeat_offset(
                12,
                32,
                repeat_index,
                sample_policy="cycle-rotate",
            )
            for request_index in range(32):
                sample_index = resolve_sample_index(
                    12,
                    request_index,
                    sample_policy="cycle-rotate",
                    sample_offset=sample_offset,
                )
                counts[sample_index] += 1

        self.assertEqual(sum(counts), 160)
        self.assertEqual(sorted(counts), [13] * 8 + [14] * 4)

    def test_cycle_rotate_uses_explicit_offset(self) -> None:
        self.assertEqual(
            resolve_sample_index(
                12,
                5,
                sample_policy="cycle-rotate",
                sample_offset=4,
            ),
            9,
        )

    def test_strict_sample_index_is_identity_mapping(self) -> None:
        self.assertEqual(
            resolve_sample_index(
                12,
                7,
                sample_policy="strict",
            ),
            7,
        )

    def test_cycle_mapping_is_deterministic(self) -> None:
        rows = [self._row(f"sample-{index:03d}") for index in range(12)]

        sample_index = resolve_sample_index(
            len(rows),
            31,
            sample_policy="cycle",
        )
        selected = select_workload_row(
            rows,
            31,
            sample_policy="cycle",
        )

        self.assertEqual(sample_index, 7)
        self.assertEqual(selected["sample_id"], "sample-007")

    def test_strict_policy_refuses_implicit_reuse(self) -> None:
        rows = [self._row(f"sample-{index:03d}") for index in range(12)]
        with self.assertRaises(WorkloadContractError):
            select_workload_row(
                rows,
                12,
                sample_policy="strict",
            )

    def test_payload_preserves_semantics_and_overrides_only_model(self) -> None:
        row = self._row("sample-001", response_format=True)
        original_request = copy.deepcopy(row["request"])

        payload = build_request_payload(
            row,
            runtime_model="qwen2.5-7b-static-draft",
        )

        self.assertEqual(payload["model"], "qwen2.5-7b-static-draft")
        self.assertEqual(payload["messages"], original_request["messages"])
        self.assertEqual(payload["temperature"], original_request["temperature"])
        self.assertEqual(payload["seed"], original_request["seed"])
        self.assertEqual(
            payload["max_completion_tokens"],
            original_request["max_tokens"],
        )
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["n"], 1)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})

        # Adapter must not mutate the frozen package row.
        self.assertEqual(row["request"], original_request)

    def test_no_response_format_is_invented(self) -> None:
        row = self._row("sample-001")
        payload = build_request_payload(
            row,
            runtime_model="qwen2.5-7b-vanilla",
        )
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["max_completion_tokens"], 96)

    def test_metadata_preserves_optional_audit_fields(self) -> None:
        row = self._row("sample-001")
        row["schema_version"] = "invoice_extraction.v1"
        row["img_id"] = "invoice.jpg"
        row["expected_source"] = {
            "sample_id": "sample-001",
            "raw_text": "OCR text",
        }

        metadata = build_workload_metadata(row)

        self.assertEqual(metadata["sample_id"], "sample-001")
        self.assertEqual(metadata["schema_version"], "invoice_extraction.v1")
        self.assertEqual(metadata["img_id"], "invoice.jpg")
        self.assertEqual(
            metadata["expected_source"]["raw_text"],
            "OCR text",
        )
        self.assertEqual(metadata["package_model"], "qwen2.5:7b-instruct")

    def test_workload_integrity_valid_hash_passes(self) -> None:
        workload_path, _ = self._write_package(
            [self._row("sample-001"), self._row("sample-002")]
        )

        integrity = verify_workload_integrity(workload_path)

        self.assertEqual(integrity.artifact_key, "chat_requests_jsonl")
        self.assertEqual(integrity.actual_sha256, integrity.expected_sha256)
        self.assertEqual(
            integrity.manifest_package_sha256,
            "a" * 64,
        )

        metadata = build_workload_metadata(
            self._row("sample-001"),
            integrity=integrity,
        )
        self.assertEqual(
            metadata["manifest_package_sha256"],
            "a" * 64,
        )
        self.assertNotIn("package_sha256", metadata)

    def test_workload_integrity_tamper_fails(self) -> None:
        workload_path, _ = self._write_package(
            [self._row("sample-001")]
        )
        workload_path.write_text(
            workload_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            WorkloadContractError,
            "SHA-256 mismatch",
        ):
            verify_workload_integrity(workload_path)

    def test_workload_integrity_requires_exactly_one_artifact_match(self) -> None:
        workload_path, manifest_path = self._write_package(
            [self._row("sample-001")]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"] = {
            "first": "chat_requests.jsonl",
            "second": "chat_requests.jsonl",
        }
        manifest["file_hashes"] = {
            "first": manifest["file_hashes"]["chat_requests_jsonl"],
            "second": manifest["file_hashes"]["chat_requests_jsonl"],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            WorkloadContractError,
            "exactly one manifest artifact",
        ):
            verify_workload_integrity(workload_path)

    def test_manifest_semantic_consistency_passes(self) -> None:
        workload_path, _ = self._write_package(
            [self._row("sample-001"), self._row("sample-002")]
        )
        integrity = verify_workload_integrity(workload_path)
        rows = load_workload_rows(workload_path)

        validate_manifest_consistency(integrity, rows)

    def test_manifest_sample_count_mismatch_fails(self) -> None:
        workload_path, _ = self._write_package(
            [self._row("sample-001")],
            manifest_overrides={"sample_count": 2},
        )
        integrity = verify_workload_integrity(workload_path)
        rows = load_workload_rows(workload_path)

        with self.assertRaisesRegex(
            WorkloadContractError,
            "sample_count mismatch",
        ):
            validate_manifest_consistency(integrity, rows)

    def test_manifest_generation_mismatch_fails(self) -> None:
        workload_path, _ = self._write_package(
            [self._row("sample-001")],
            manifest_overrides={"max_tokens": 97},
        )
        integrity = verify_workload_integrity(workload_path)
        rows = load_workload_rows(workload_path)

        with self.assertRaisesRegex(
            WorkloadContractError,
            "generation mismatch",
        ):
            validate_manifest_consistency(integrity, rows)

    def test_real_frozen_packages_if_present(self) -> None:
        # Validate canonical committed bytes; checkout CRLF is not a dataset change.
        from experiments.bench_source import snapshot
        packed, _ = snapshot(REPO_ROOT)
        self.addCleanup(packed.cleanup)
        canonical_root = Path(packed.name)
        w1_path = (
            canonical_root
            / "experiments"
            / "configs"
            / "w1"
            / "node1_chat_requests.jsonl"
        )
        w2_path = (
            canonical_root
            / "experiments"
            / "configs"
            / "w2"
            / "chat_requests.jsonl"
        )
        w3_path = (
            canonical_root
            / "experiments"
            / "configs"
            / "w3"
            / "chat_requests.jsonl"
        )

        if not (w1_path.exists() and w2_path.exists() and w3_path.exists()):
            self.skipTest("run this test after copying the files into the repo")

        w1_integrity = verify_workload_integrity(w1_path)
        w2_integrity = verify_workload_integrity(w2_path)
        w3_integrity = verify_workload_integrity(w3_path)

        w1 = load_workload_rows(w1_path)
        w2 = load_workload_rows(w2_path)
        w3 = load_workload_rows(w3_path)

        validate_manifest_consistency(w1_integrity, w1)
        validate_manifest_consistency(w2_integrity, w2)
        validate_manifest_consistency(w3_integrity, w3)

        self.assertEqual(len(w1), 30)
        self.assertEqual(len(w2), 12)
        self.assertEqual(len(w3), 12)

        self.assertEqual(w1[0]["request"]["max_tokens"], 1024)
        self.assertEqual(w2[0]["request"]["max_tokens"], 96)
        self.assertEqual(w3[0]["request"]["max_tokens"], 192)
        self.assertEqual(
            w1_integrity.actual_sha256,
            "0de2387c4e6a8c39523bf869788a55ee7487c6a32318a949ce22e1f8f7b17360",
        )
        self.assertEqual(
            w2_integrity.actual_sha256,
            "9acc38d38f410cc2261fac963d74648180c8ec3fdef249143952ca1dee8da2c2",
        )
        self.assertEqual(
            w3_integrity.actual_sha256,
            "40ef62c105803365215f115c6aeacb85b5fe14ea2f8e48ce151aabe86b89d648",
        )
        self.assertEqual(
            w1[0]["request"]["response_format"],
            {"type": "json_object"},
        )



    def test_launchers_pin_vllm_generation_config(self) -> None:
        vanilla = REPO_ROOT / "scripts" / "launch_lane1_vanilla.sh"
        static = REPO_ROOT / "scripts" / "launch_lane2_static.sh"

        if not (vanilla.exists() and static.exists()):
            self.skipTest("run this test after copying the files into the repo")

        vanilla_text = vanilla.read_text(encoding="utf-8")
        static_text = static.read_text(encoding="utf-8")

        self.assertIn("--generation-config vllm", vanilla_text)
        self.assertIn("--generation-config vllm", static_text)
        self.assertIn("--served-model-name qwen2.5-7b-vanilla", vanilla_text)
        self.assertIn(
            "--served-model-name qwen2.5-7b-static-draft",
            static_text,
        )



class BenchmarkHarnessOrchestrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_run_requests_resolves_workload_before_execution(
        self,
    ) -> None:
        rows = [
            BenchmarkWorkloadAdapterTests._row(
                f"sample-{index:03d}"
            )
            for index in range(3)
        ]

        args = argparse.Namespace(
            model="qwen2.5-7b-vanilla",
            lane="vanilla",
        )

        observed: list[dict] = []

        async def fake_run_one(
            *,
            session,
            semaphore,
            args,
            env,
            identity,
            concurrency,
            payload,
            workload_metadata,
        ) -> dict:
            # Force completion order to differ from request order.
            await asyncio.sleep(
                (5 - identity["request_index"]) * 0.001
            )

            observed.append(
                {
                    "request_index": identity["request_index"],
                    "sample_index": workload_metadata[
                        "sample_index"
                    ],
                    "sample_offset": workload_metadata[
                        "sample_offset"
                    ],
                    "sample_id": workload_metadata[
                        "sample_id"
                    ],
                    "canonical_sample_count": workload_metadata[
                        "canonical_sample_count"
                    ],
                    "sample_policy": workload_metadata[
                        "sample_policy"
                    ],
                    "messages": payload["messages"],
                    "model": payload["model"],
                }
            )

            return {
                "request_index": identity["request_index"],
                "sample_index": workload_metadata[
                    "sample_index"
                ],
                "sample_id": workload_metadata["sample_id"],
            }

        with patch(
            "benchmark_harness.run_one",
            new=fake_run_one,
        ):
            results = await run_requests(
                session=object(),
                args=args,
                env={},
                workload_rows=rows,
                workload_integrity=None,
                sample_policy="cycle-rotate",
                count=5,
                concurrency=3,
                experiment_id="exp-test",
                cell_id="vanilla__W2__c3",
                repeat_id="r001",
                phase="measured",
                request_index_start=0,
                sample_offset=2,
            )

        # gather() preserves input-awaitable result order even when
        # fake_run_one completes in a different order.
        self.assertEqual(
            results,
            [
                {
                    "request_index": 0,
                    "sample_index": 2,
                    "sample_id": "sample-002",
                },
                {
                    "request_index": 1,
                    "sample_index": 0,
                    "sample_id": "sample-000",
                },
                {
                    "request_index": 2,
                    "sample_index": 1,
                    "sample_id": "sample-001",
                },
                {
                    "request_index": 3,
                    "sample_index": 2,
                    "sample_id": "sample-002",
                },
                {
                    "request_index": 4,
                    "sample_index": 0,
                    "sample_id": "sample-000",
                },
            ],
        )

        by_request_index = {
            item["request_index"]: item
            for item in observed
        }

        self.assertEqual(
            by_request_index[3]["sample_index"],
            2,
        )
        self.assertEqual(
            by_request_index[3]["sample_id"],
            "sample-002",
        )
        self.assertEqual(
            by_request_index[4]["sample_index"],
            0,
        )
        self.assertEqual(
            by_request_index[4]["sample_id"],
            "sample-000",
        )

        for request_index in range(5):
            item = by_request_index[request_index]

            self.assertEqual(
                item["canonical_sample_count"],
                3,
            )
            self.assertEqual(
                item["sample_policy"],
                "cycle-rotate",
            )
            self.assertEqual(item["sample_offset"], 2)
            self.assertEqual(
                item["model"],
                "qwen2.5-7b-vanilla",
            )

            expected_sample_index = (request_index + 2) % 3

            self.assertEqual(
                item["messages"],
                rows[expected_sample_index][
                    "request"
                ]["messages"],
            )


if __name__ == "__main__":
    unittest.main()
