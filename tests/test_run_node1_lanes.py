"""Node 1 runner: lane selection, endpoint resolution and SQLite provenance."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_node1_extractor as runner  # noqa: E402

from src.pipeline.extractor import (  # noqa: E402
    ExtractorStage,
    FailureCode,
    Node1Failure,
    Node1Success,
)

EXAMPLE_JSON = REPO_ROOT / "src/pipeline/examples/node1_extractor_output_m0_04_001.json"
REQUESTS = REPO_ROOT / "experiments/configs/w1/node1_chat_requests.jsonl"
MANIFEST = REPO_ROOT / "experiments/configs/w1/package_manifest.json"
SCHEMA_SQL = REPO_ROOT / "src/pipeline/sqlite/schema.sql"
JSON_SCHEMA = REPO_ROOT / "src/pipeline/schemas/invoice_extraction.schema.json"
V1_DB = REPO_ROOT / "data/processed/m0_04_mcocr_devset/invoices_v1.sqlite"
SECRET = "secret-token-xyz"
ENV_KEYS = ("PIPELINE_LLM_BASE_URL", "PIPELINE_LLM_MODEL", "PIPELINE_LLM_API_KEY")


def clean_env(**extra: str):
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env.update(extra)
    return patch.dict(os.environ, env, clear=True)


class ResolveArgsTests(unittest.TestCase):
    def parse(self, *argv: str, **env: str):
        with clean_env(**env), patch.object(runner, "load_token", return_value="modal-token") as token:
            args = runner.parse_args(list(argv))
        self.token = token
        return args

    def test_unknown_lane_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse("--lane", "turbo")

    def test_lane_choices_are_the_four_lanes(self) -> None:
        for lane in ("vanilla", "static", "adaptive", "eagle"):
            self.assertEqual(self.parse("--lane", lane).lane, lane)

    def test_modal_is_default_backend(self) -> None:
        args = self.parse("--lane", "static")
        self.assertEqual(args.backend, "modal")
        self.assertIn("ptit-vllm-static", args.base_url)
        self.assertEqual(args.model, "qwen2.5-7b-static-draft")
        self.assertEqual(args.api_key, "modal-token")
        self.assertEqual(args.ready_timeout, 1200.0)

    def test_local_backend_uses_launcher_ports(self) -> None:
        launchers = {
            "vanilla": "launch_lane1_vanilla.sh",
            "static": "launch_lane2_static.sh",
            "adaptive": "launch_lane3_adaptive.sh",
            "eagle": "launch_lane4_eagle.sh",
        }
        for lane, script in launchers.items():
            text = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
            port = re.search(r"(?:--port |port=)(\d+)", text).group(1)
            args = self.parse("--lane", lane, "--backend", "local")
            self.assertEqual(args.base_url, f"http://127.0.0.1:{port}/v1", lane)
            self.assertEqual(args.ready_timeout, 0.0)
        self.token.assert_not_called()

    def test_explicit_url_and_model_win(self) -> None:
        args = self.parse(
            "--lane", "vanilla", "--base-url", "http://localhost:11434/v1", "--model", "qwen2.5:7b-instruct"
        )
        self.assertEqual(args.base_url, "http://localhost:11434/v1")
        self.assertEqual(args.model, "qwen2.5:7b-instruct")
        self.assertEqual(args.lane, "vanilla")
        self.assertEqual(args.api_key, "")
        self.assertEqual(args.ready_timeout, 0.0)
        self.token.assert_not_called()

    def test_environment_url_and_model_win_over_lane(self) -> None:
        args = self.parse(
            "--lane", "eagle", PIPELINE_LLM_BASE_URL="http://x/v1", PIPELINE_LLM_MODEL="m"
        )
        self.assertEqual((args.base_url, args.model), ("http://x/v1", "m"))
        self.token.assert_not_called()

    def test_environment_api_key_wins_over_modal_token(self) -> None:
        args = self.parse("--lane", "vanilla", PIPELINE_LLM_API_KEY=SECRET)
        self.assertEqual(args.api_key, SECRET)
        self.token.assert_not_called()

    def test_explicit_ready_timeout_kept(self) -> None:
        self.assertEqual(self.parse("--lane", "vanilla", "--ready-timeout", "5").ready_timeout, 5.0)

    def test_default_outputs_follow_lane_and_avoid_m1_evidence(self) -> None:
        args = self.parse("--lane", "eagle")
        self.assertEqual(args.run_id, "m4_03_eagle")
        self.assertEqual(args.output_dir, Path("experiments/runs/m4_03/m4_03_eagle"))
        self.assertEqual(args.report, Path("experiments/runs/m4_03/m4_03_eagle/run_summary.json"))
        for value in (args.run_id, args.output_dir.as_posix(), args.report.as_posix()):
            self.assertNotIn("m1_03", value)

    def test_run_id_drives_default_output_dir(self) -> None:
        args = self.parse("--lane", "vanilla", "--run-id", "m4_03_vanilla_r2")
        self.assertEqual(args.output_dir, Path("experiments/runs/m4_03/m4_03_vanilla_r2"))

    def test_explicit_outputs_kept(self) -> None:
        args = self.parse("--lane", "eagle", "--run-id", "r", "--output-dir", "o", "--report", "o/s.json")
        self.assertEqual((args.run_id, args.output_dir, args.report), ("r", Path("o"), Path("o/s.json")))

    def test_default_database_is_ignored_by_git(self) -> None:
        args = self.parse("--lane", "vanilla")
        self.assertEqual(args.db, Path("data/processed/invoices.db"))
        result = subprocess.run(
            ["git", "check-ignore", "-q", args.db.as_posix()], cwd=REPO_ROOT, check=False
        )
        self.assertEqual(result.returncode, 0)


class FakeExtractor:
    """Returns a queued latency per request; None means an invalid JSON sample."""

    latencies: list[float | None] = []
    preflight_calls = 0

    def __init__(self, client, model, json_schema_path, source_policy="restore") -> None:  # noqa: ANN001
        self.source_policy = source_policy
        type(self).last_policy = source_policy
        self.queue = list(type(self).latencies)
        self.example = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))

    def preflight(self) -> None:
        type(self).preflight_calls += 1

    def run_request(self, row: dict):
        latency = self.queue.pop(0)
        if latency is None:
            return Node1Failure(
                sample_id=row["sample_id"],
                error_code=FailureCode.JSON_PARSE_ERROR,
                stage=ExtractorStage.PARSE,
                message="bad json",
                latency_ms=10.0,
            )
        payload = copy.deepcopy(self.example)
        payload["source"]["sample_id"] = row["sample_id"]
        payload["source"]["img_id"] = row["img_id"]
        return Node1Success(
            sample_id=row["sample_id"],
            latency_ms=latency,
            raw_response={"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
            assistant_content="{}",
            canonical_payload=payload,
            normalizations=(),
        )


class RunnerMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / "invoices.db"
        FakeExtractor.preflight_calls = 0
        self._patches = [
            clean_env(PIPELINE_LLM_API_KEY=SECRET),
            patch.object(runner, "Node1Extractor", FakeExtractor),
            patch.object(runner, "OpenAICompatibleClient", MagicMock()),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self._patches):
            item.stop()
        self._tmp.cleanup()

    def run_main(self, latencies: list[float | None], *extra: str) -> int:
        FakeExtractor.latencies = latencies
        return runner.main(
            [
                "--lane", "eagle",
                "--run-id", "r1",
                "--requests", str(REQUESTS),
                "--package-manifest", str(MANIFEST),
                "--json-schema", str(JSON_SCHEMA),
                "--sql-schema", str(SCHEMA_SQL),
                "--output-dir", str(self.root / "out"),
                "--db", str(self.db),
                "--limit", str(len(latencies)),
                *extra,
            ]
        )

    def query(self, sql: str) -> list[tuple]:
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    def summary(self) -> dict:
        return json.loads((self.root / "out" / "run_summary.json").read_text(encoding="utf-8"))

    def test_lane_model_latency_and_median_stored(self) -> None:
        self.assertEqual(self.run_main([100.0, 900.1234, 200.0]), 0)
        self.assertEqual(
            self.query("SELECT lane, model_name, latency_ms_median FROM extraction_runs"),
            [("eagle", "qwen2.5-7b-eagle3-draft", 200.0)],
        )
        self.assertEqual(
            sorted(row[0] for row in self.query("SELECT latency_ms FROM invoices")), [100.0, 200.0, 900.123]
        )

    def test_latency_matches_validated_outputs(self) -> None:
        self.run_main([123.4567])
        line = (self.root / "out" / "validated_outputs.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(self.query("SELECT latency_ms FROM invoices")[0][0], json.loads(line)["latency_ms"])

    def test_invalid_sample_not_stored(self) -> None:
        self.run_main([100.0, None])
        self.assertEqual(self.query("SELECT COUNT(*) FROM invoices"), [(1,)])
        self.assertEqual(len((self.root / "out" / "failures.jsonl").read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(self.summary()["rates"]["schema_valid_rate"], 0.5)

    def test_median_is_null_without_stored_invoice(self) -> None:
        self.run_main([None])
        self.assertEqual(self.query("SELECT latency_ms_median FROM extraction_runs"), [(None,)])

    def test_hardware_label_in_summary_and_run_notes(self) -> None:
        self.run_main([100.0], "--hardware", "modal-A10G")
        self.assertEqual(self.summary()["hardware"], "modal-A10G")
        self.assertIn("modal-A10G", self.query("SELECT notes FROM extraction_runs")[0][0])

    def test_source_policy_defaults_to_restore_and_is_recorded(self) -> None:
        self.run_main([100.0])
        self.assertEqual(self.summary()["source_policy"], "restore")
        self.run_main([100.0], "--source-policy", "strict", "--replace-run")
        self.assertEqual(self.summary()["source_policy"], "strict")
        self.assertEqual(FakeExtractor.last_policy, "strict")

    def test_hardware_defaults_to_null(self) -> None:
        self.run_main([100.0])
        self.assertIsNone(self.summary()["hardware"])

    def test_rerun_without_replace_stops_before_any_request(self) -> None:
        self.run_main([100.0])
        calls = FakeExtractor.preflight_calls
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.run_main([500.0])
        self.assertEqual(FakeExtractor.preflight_calls, calls)
        self.assertEqual(self.query("SELECT latency_ms FROM invoices"), [(100.0,)])

    def test_rerun_with_replace_overwrites(self) -> None:
        self.run_main([100.0])
        self.assertEqual(self.run_main([500.0], "--replace-run"), 0)
        self.assertEqual(self.query("SELECT latency_ms FROM invoices"), [(500.0,)])

    def test_v1_database_refused_before_any_request(self) -> None:
        shutil.copy(V1_DB, self.db)
        with self.assertRaisesRegex(RuntimeError, "user_version 1"):
            self.run_main([100.0])
        self.assertEqual(FakeExtractor.preflight_calls, 0)

    def test_token_not_written(self) -> None:
        self.run_main([100.0, None])
        for path in (self.root / "out").iterdir():
            self.assertNotIn(SECRET, path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
