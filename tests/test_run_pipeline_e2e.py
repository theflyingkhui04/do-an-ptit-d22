"""Tests for scripts/run_pipeline_e2e.py with fake Node 1 and fake LLM clients."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_pipeline_e2e as e2e  # noqa: E402

from src.client.modal_client import lane_base_url  # noqa: E402
from tests.test_node2_runner import build_example_db  # noqa: E402

SECRET = "secret-token-xyz"


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_completions(self, payload: dict) -> dict:
        self.calls += 1
        system = payload["messages"][0]["content"]
        if system.startswith("You are Node 2"):
            content = "SELECT seller_name, total_amount_value FROM v_invoice_facts"
        else:
            content = "Tổng quan: MINIMART ANAN có tổng 42000."
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }


class FakeClientFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> FakeClient:
        self.calls.append((args, kwargs))
        return FakeClient()


def fake_node1_runner(args, run_dir: Path, base_url: str, model: str, api_key: str) -> dict:
    build_example_db(run_dir / "invoices.sqlite", run_id=args.run_id)
    summary = {
        "request_count": 1,
        "fully_validated_count": 1,
        "rates": {"schema_valid_rate": 1.0},
        "latency_ms": {"p50": 10.0},
        "returncode": 0,
    }
    (run_dir / "node1_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def failing_node1_runner(args, run_dir, base_url, model, api_key) -> dict:
    return {"returncode": 1, "error": "node1_summary_missing"}


class RunPipelineE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"PIPELINE_LLM_API_KEY": SECRET})
        self._env.start()
        self._sha = patch.object(e2e, "git_sha", return_value="abc123")
        self._sha.start()

    def tearDown(self) -> None:
        self._sha.stop()
        self._env.stop()
        self._tmp.cleanup()

    def argv(self, *extra: str) -> list[str]:
        return [
            "--run-id", "t1",
            "--base-url", "http://x/v1",
            "--model", "m",
            "--output-root", str(self.root),
            *extra,
        ]

    def test_e2e_writes_all_outputs(self) -> None:
        factory = FakeClientFactory()
        rc = e2e.main(self.argv(), client_factory=factory, node1_runner=fake_node1_runner)
        self.assertEqual(rc, 0)
        run_dir = self.root / "t1-vanilla"
        for name in (
            "node1_summary.json",
            "invoices.sqlite",
            "node2_results.jsonl",
            "node3_result.json",
            "report.md",
            "e2e_summary.json",
        ):
            self.assertTrue((run_dir / name).exists(), name)
        lines = (run_dir / "node2_results.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 6)
        summary = json.loads((run_dir / "e2e_summary.json").read_text(encoding="utf-8"))
        for key in (
            "task", "run_id", "lane", "model", "base_url", "git_sha", "started_at",
            "finished_at", "e2e_wall_ms", "invoice_count", "node1", "node2", "node3", "prompt_hashes",
        ):
            self.assertIn(key, summary)
        self.assertEqual(summary["task"], "M3-06")
        self.assertEqual(summary["invoice_count"], 1)
        self.assertEqual(summary["git_sha"], "abc123")
        self.assertEqual(summary["node2"]["requests"], 6)
        self.assertEqual(summary["node2"]["status_counts"], {"ok": 6})
        self.assertEqual(summary["node3"]["status"], "ok")
        self.assertEqual(summary["node1"]["returncode"], 0)
        self.assertEqual(
            set(summary["prompt_hashes"]), {"w2_system_prompt", "w3_system_prompt", "node3_user_template"}
        )
        self.assertIn("MINIMART ANAN", (run_dir / "report.md").read_text(encoding="utf-8"))

    def test_hardware_label_recorded(self) -> None:
        # M3-06 run L5: the summary must say which GPU served the run.
        rc = e2e.main(self.argv("--hardware", "modal-A10G"), client_factory=FakeClientFactory(),
                      node1_runner=fake_node1_runner)
        self.assertEqual(rc, 0)
        summary = json.loads((self.root / "t1-vanilla" / "e2e_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["hardware"], "modal-A10G")

    def test_summary_records_node1_package(self) -> None:
        rc = e2e.main(self.argv(), client_factory=FakeClientFactory(), node1_runner=fake_node1_runner)
        self.assertEqual(rc, 0)
        summary = json.loads((self.root / "t1-vanilla" / "e2e_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["node1_package_dir"], "experiments/configs/w1")

    def test_hardware_label_defaults_to_none(self) -> None:
        e2e.main(self.argv(), client_factory=FakeClientFactory(), node1_runner=fake_node1_runner)
        summary = json.loads((self.root / "t1-vanilla" / "e2e_summary.json").read_text(encoding="utf-8"))
        self.assertIsNone(summary["hardware"])

    def test_secret_not_written(self) -> None:
        e2e.main(self.argv(), client_factory=FakeClientFactory(), node1_runner=fake_node1_runner)
        for path in (self.root / "t1-vanilla").iterdir():
            if path.suffix != ".sqlite":
                self.assertNotIn(SECRET, path.read_text(encoding="utf-8"), path.name)

    def test_ready_timeout_passed_to_client(self) -> None:
        factory = FakeClientFactory()
        e2e.main(self.argv(), client_factory=factory, node1_runner=fake_node1_runner)
        args, kwargs = factory.calls[0]
        self.assertEqual(args, ("http://x/v1", SECRET))
        self.assertEqual(kwargs["ready_timeout_seconds"], 1200.0)
        self.assertEqual(kwargs["timeout_seconds"], 900.0)

    def test_node1_failure_skips_downstream(self) -> None:
        factory = FakeClientFactory()
        rc = e2e.main(self.argv(), client_factory=factory, node1_runner=failing_node1_runner)
        self.assertEqual(rc, 2)
        run_dir = self.root / "t1-vanilla"
        summary = json.loads((run_dir / "e2e_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["invoice_count"], 0)
        self.assertEqual(summary["node3"]["status"], "skipped_no_data")
        self.assertEqual(summary["node1"]["returncode"], 1)
        self.assertFalse((run_dir / "report.md").exists())
        self.assertEqual(factory.calls, [])

    def test_stale_artifacts_not_reused_when_node1_crashes(self) -> None:
        run_dir = self.root / "t1-vanilla"
        run_dir.mkdir(parents=True)
        build_example_db(run_dir / "invoices.sqlite", run_id="t1")
        (run_dir / "node1_summary.json").write_text("{}", encoding="utf-8")
        factory = FakeClientFactory()
        rc = e2e.main(self.argv(), client_factory=factory, node1_runner=failing_node1_runner)
        self.assertEqual(rc, 2)
        summary = json.loads((run_dir / "e2e_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["invoice_count"], 0)
        self.assertEqual(factory.calls, [])
        self.assertFalse((run_dir / "invoices.sqlite").exists())

    def test_skip_node1_uses_given_db(self) -> None:
        db = self.root / "given.sqlite"
        build_example_db(db)

        def must_not_run(*_a, **_k):
            raise AssertionError("node1 runner must not be called")

        rc = e2e.main(
            self.argv("--skip-node1", "--db", str(db)),
            client_factory=FakeClientFactory(),
            node1_runner=must_not_run,
        )
        self.assertEqual(rc, 0)

    def test_skip_node1_requires_db(self) -> None:
        argv = self.argv("--skip-node1")
        with self.assertRaises(SystemExit):
            e2e.parse_args(argv)

    def test_unsafe_run_id_rejected(self) -> None:
        # Sonar pythonsecurity:S8707: run_id must not escape --output-root.
        for bad in ("../x", "a/b", "a\\b","..", "", " x", "x" * 101):
            with self.subTest(run_id=bad), self.assertRaises(SystemExit):
                e2e.parse_args(["--run-id", bad, "--base-url", "http://x/v1"])

    def test_safe_run_id_accepted(self) -> None:
        args = e2e.parse_args(["--run-id", "20261104-m306-e2e.a10g_r2", "--base-url", "http://x/v1"])
        self.assertEqual(args.run_id, "20261104-m306-e2e.a10g_r2")

    def test_resolve_endpoint_lane_default(self) -> None:
        args = e2e.parse_args(["--run-id", "r"])
        self.assertEqual(e2e.resolve_endpoint(args), (lane_base_url("vanilla"), "qwen2.5-7b-vanilla"))

    def test_node1_command_runs_from_repo_root(self) -> None:
        args = e2e.parse_args(self.argv("--limit", "5"))
        run_dir = self.root / "t1-vanilla"
        run_dir.mkdir(parents=True)
        with patch.object(e2e.subprocess, "run") as run:
            run.return_value.returncode = 3
            out = e2e.run_node1(args, run_dir, "http://x/v1", "m", SECRET)
        cmd = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(Path(kwargs["cwd"]), e2e.REPO_ROOT)
        self.assertEqual(kwargs["env"]["PIPELINE_LLM_API_KEY"], SECRET)
        self.assertNotIn(SECRET, " ".join(cmd))
        self.assertIn("experiments/configs/w1/node1_chat_requests.jsonl", cmd)
        self.assertIn("--limit", cmd)
        self.assertEqual(out, {"returncode": 3, "error": "node1_summary_missing"})

    def test_node1_package_dir_selects_w1_v2(self) -> None:
        args = e2e.parse_args(self.argv("--node1-package-dir", "experiments/configs/w1_v2"))
        run_dir = self.root / "t1-vanilla"
        run_dir.mkdir(parents=True)
        with patch.object(e2e.subprocess, "run") as run:
            run.return_value.returncode = 0
            e2e.run_node1(args, run_dir, "http://x/v1", "m", SECRET)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--requests") + 1], "experiments/configs/w1_v2/node1_chat_requests.jsonl")
        self.assertEqual(cmd[cmd.index("--package-manifest") + 1], "experiments/configs/w1_v2/package_manifest.json")

    def test_node1_package_dir_must_hold_a_package(self) -> None:
        with self.assertRaises(SystemExit):
            e2e.parse_args(self.argv("--node1-package-dir", "experiments/configs/missing"))


if __name__ == "__main__":
    unittest.main()
