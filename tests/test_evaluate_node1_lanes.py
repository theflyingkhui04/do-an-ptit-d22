"""Lane comparison report built from Node 1 outputs and MC-OCR labels."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_node1_lanes as report  # noqa: E402

SAMPLES = ("s1", "s2", "s3")


def label_row(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "key_text_pairs": [
            {"label": "SELLER", "text": f"Shop {sample_id}"},
            {"label": "ADDRESS", "text": "ĐC: Chợ Sủi"},
            {"label": "TIMESTAMPS", "text": "Ngày: 10/08/2020 - 17:12"},
            {"label": "TOTAL_COST", "text": "Tổng tiền"},
            {"label": "TOTAL_COST", "text": "42,000"},
        ],
    }


def output_row(sample_id: str, seller: str | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "payload": {
            "seller": {"name": seller or f"Shop {sample_id}", "address": "Chợ Sủi"},
            "invoice": {"issued_date": "2020-08-10", "issued_time": "17:12:00"},
            "totals": {"total_amount": {"value": 42000}},
        },
    }


class EvaluateLanesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.labels = self.root / "labels.jsonl"
        self.write(self.labels, [label_row(s) for s in SAMPLES])
        self.prefix = self.root / "result"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def run_file(self, name: str, wrong: int = 0, skip: tuple[str, ...] = ()) -> str:
        rows = [
            output_row(s, seller="Sai" if i < wrong else None)
            for i, s in enumerate(SAMPLES)
            if s not in skip
        ]
        path = self.root / f"{name}.jsonl"
        self.write(path, rows)
        return f"{name}={path}"

    def evaluate(self, *runs: str) -> dict:
        argv = ["--labels", str(self.labels), "--output-prefix", str(self.prefix)]
        for run in runs:
            argv += ["--run", run]
        self.assertEqual(report.main(argv), 0)
        return json.loads(self.prefix.with_suffix(".json").read_text(encoding="utf-8"))

    def five_runs(self) -> dict:
        return self.evaluate(
            self.run_file("vanilla_r1"),
            self.run_file("vanilla_r2", wrong=1),
            self.run_file("static"),
            self.run_file("adaptive", wrong=1),
            self.run_file("eagle", wrong=2),
        )

    def test_field_counts_per_run(self) -> None:
        field = self.five_runs()["runs"]["eagle"]["fields"]["SELLER"]
        self.assertEqual((field["scored"], field["matched"], field["blank"]), (3, 1, 0))
        self.assertAlmostEqual(field["match_rate"], 1 / 3, places=4)

    def test_address_prefix_applies_in_report(self) -> None:
        self.assertEqual(self.five_runs()["runs"]["static"]["fields"]["ADDRESS"]["matched"], 3)

    def test_noise_comes_from_the_two_vanilla_runs(self) -> None:
        comparison = self.five_runs()["comparison"]
        self.assertAlmostEqual(comparison["noise"]["SELLER"], 1 / 3, places=4)
        self.assertEqual(comparison["noise"]["TOTAL_COST"], 0.0)

    def test_within_noise_flags(self) -> None:
        lanes = self.five_runs()["comparison"]["lanes"]
        self.assertEqual(set(lanes), {"static", "adaptive", "eagle"})
        self.assertTrue(lanes["static"]["SELLER"]["within_noise"])
        self.assertTrue(lanes["adaptive"]["SELLER"]["within_noise"])
        self.assertFalse(lanes["eagle"]["SELLER"]["within_noise"])
        self.assertAlmostEqual(lanes["eagle"]["SELLER"]["delta"], -2 / 3, places=4)

    def test_markdown_has_run_rows_and_comparison(self) -> None:
        self.five_runs()
        text = self.prefix.with_suffix(".md").read_text(encoding="utf-8")
        for label in ("vanilla_r1", "vanilla_r2", "static", "adaptive", "eagle"):
            self.assertIn(f"| {label} |", text)
        self.assertIn("Trong dao động vanilla", text)
        self.assertIn("1/3 (33.3%)", text)

    def test_mismatches_are_listed(self) -> None:
        mismatches = self.five_runs()["runs"]["eagle"]["mismatches"]
        self.assertEqual(len(mismatches), 2)
        self.assertEqual(mismatches[0]["field"], "SELLER")
        self.assertEqual(mismatches[0]["predicted"], "Sai")

    def test_missing_sample_counts_as_wrong_and_is_listed(self) -> None:
        run = self.evaluate(self.run_file("static", skip=("s2",)))["runs"]["static"]
        self.assertEqual(run["missing_samples"], ["s2"])
        self.assertAlmostEqual(run["valid_json_rate"], 2 / 3, places=4)
        field = run["fields"]["TOTAL_COST"]
        self.assertEqual((field["scored"], field["matched"], field["blank"]), (3, 2, 1))
        self.assertIn("s2", self.prefix.with_suffix(".md").read_text(encoding="utf-8"))

    def test_no_comparison_without_both_vanilla_runs(self) -> None:
        result = self.evaluate(self.run_file("vanilla_r1"), self.run_file("static"))
        self.assertIsNone(result["comparison"])
        self.assertNotIn("Trong dao động vanilla", self.prefix.with_suffix(".md").read_text(encoding="utf-8"))

    def test_unreadable_input_stops_without_output(self) -> None:
        good = self.run_file("vanilla_r1")
        argv = [
            "--labels", str(self.labels), "--output-prefix", str(self.prefix),
            "--run", good, "--run", f"static={self.root / 'missing.jsonl'}",
        ]
        with self.assertRaises(SystemExit) as caught:
            report.main(argv)
        self.assertIn("missing.jsonl", str(caught.exception.code))
        self.assertFalse(self.prefix.with_suffix(".json").exists())
        self.assertFalse(self.prefix.with_suffix(".md").exists())

    def test_run_argument_needs_label_and_path(self) -> None:
        with self.assertRaises(SystemExit):
            report.main(["--labels", str(self.labels), "--run", "no-separator"])


if __name__ == "__main__":
    unittest.main()
