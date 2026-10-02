from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class W2W3WorkloadPackageTests(unittest.TestCase):
    def test_w2_w3_packages_have_replayable_chat_requests(self) -> None:
        cases = {
            "w2": ("w2_text_to_sql", "w2.m1_03.v1", 96),
            "w3": ("w3_free_analysis", "w3.m1_03.v1", 192),
        }

        for folder, (workload_id, dataset_version, max_tokens) in cases.items():
            with self.subTest(folder=folder):
                package_dir = REPO_ROOT / "experiments" / "configs" / folder
                manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
                samples = read_jsonl(package_dir / "samples.jsonl")
                requests = read_jsonl(package_dir / "chat_requests.jsonl")
                provenance = read_jsonl(package_dir / "provenance.jsonl")

                self.assertEqual(manifest["workload_id"], workload_id)
                self.assertEqual(manifest["dataset_version"], dataset_version)
                self.assertEqual(manifest["sample_count"], 12)
                self.assertEqual(len(samples), 12)
                self.assertEqual(len(requests), 12)
                self.assertEqual(len(provenance), 12)

                sample_ids = [row["sample_id"] for row in samples]
                self.assertEqual(len(sample_ids), len(set(sample_ids)))
                self.assertEqual(sample_ids, [row["sample_id"] for row in requests])
                self.assertEqual(sample_ids, [row["sample_id"] for row in provenance])

                for row in requests:
                    request = row["request"]
                    self.assertEqual(row["workload_id"], workload_id)
                    self.assertEqual(row["dataset_version"], dataset_version)
                    self.assertEqual(request["temperature"], 0.0)
                    self.assertEqual(request["max_tokens"], max_tokens)
                    self.assertEqual(request["seed"], 42)
                    self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
                    self.assertTrue(request["messages"][0]["content"].strip())
                    self.assertTrue(request["messages"][1]["content"].strip())


if __name__ == "__main__":
    unittest.main()
