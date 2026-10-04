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


class W2W3V2WorkloadPackageTests(unittest.TestCase):
    def _package(self, folder: str) -> tuple[dict, list[dict], list[dict], list[dict]]:
        package_dir = REPO_ROOT / "experiments" / "configs" / folder
        manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
        samples = read_jsonl(package_dir / "samples.jsonl")
        requests = read_jsonl(package_dir / "chat_requests.jsonl")
        provenance = read_jsonl(package_dir / "provenance.jsonl")
        return manifest, samples, requests, provenance

    def test_v2_packages_are_replayable_and_versioned(self) -> None:
        cases = {
            "w2_v2": ("w2_text_to_sql", "w2.m1_03.v2", "w2_text_to_sql.v2", 96),
            "w3_v2": ("w3_free_analysis", "w3.m1_03.v2", "w3_business_analysis.v2", 768),
        }

        for folder, (workload_id, dataset_version, prompt_version, max_tokens) in cases.items():
            with self.subTest(folder=folder):
                manifest, samples, requests, provenance = self._package(folder)
                self.assertEqual(manifest["workload_id"], workload_id)
                self.assertEqual(manifest["dataset_version"], dataset_version)
                self.assertEqual(manifest["prompt_version"], prompt_version)
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
                    self.assertEqual(row["prompt_version"], prompt_version)
                    self.assertEqual(request["temperature"], 0.0)
                    self.assertEqual(request["max_tokens"], max_tokens)
                    self.assertEqual(request["seed"], 42)
                    self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
                    self.assertTrue(request["messages"][0]["content"].strip())
                    self.assertTrue(request["messages"][1]["content"].strip())

    def test_w2_v2_is_pipeline_schema_aligned(self) -> None:
        manifest, samples, requests, _ = self._package("w2_v2")
        self.assertEqual(manifest["entropy_profile"], "medium")

        system_prompt = requests[0]["request"]["messages"][0]["content"]
        self.assertIn("SQLite", system_prompt)
        self.assertIn("v_invoice_facts", system_prompt)
        self.assertIn("v_item_facts", system_prompt)
        self.assertIn("v_tax_facts", system_prompt)
        self.assertIn("SQL text only", system_prompt)
        self.assertIn("Do not query base tables", system_prompt)

        user_text = "\n".join(
            row["request"]["messages"][1]["content"] for row in requests
        ).lower()
        self.assertNotIn("postgresql", user_text)
        for legacy_table in ("sellers", "buyers", "invoices", "invoice_items", "invoice_taxes"):
            self.assertNotIn(legacy_table, user_text)

        covered_views = {view for row in samples for view in row["target_views"]}
        self.assertEqual(
            covered_views,
            {"v_invoice_facts", "v_item_facts", "v_tax_facts"},
        )

        covered_features = {feature for row in samples for feature in row["sql_features"]}
        for feature in (
            "filter",
            "aggregate",
            "group-by",
            "order-by",
            "top-k",
            "having",
            "join",
            "cte",
            "count-distinct",
            "ratio",
            "multi-condition",
        ):
            self.assertIn(feature, covered_features)

    def test_w3_v2_is_business_analysis_only_and_diverse(self) -> None:
        manifest, samples, requests, _ = self._package("w3_v2")
        self.assertEqual(manifest["entropy_profile"], "high")
        self.assertEqual(manifest["max_tokens"], 768)

        categories = {row["scenario_category"] for row in samples}
        required = {
            "trend-summary",
            "concentration",
            "product-mix",
            "anomaly-risk",
            "compliance-risk",
            "data-quality",
            "reconciliation",
            "audit-prioritization",
            "dashboard-decision",
            "decision-recommendation",
        }
        self.assertTrue(required.issubset(categories))

        text = "\n".join(
            row["request"]["messages"][1]["content"] for row in requests
        ).lower()
        for banned in (
            "speculative decoding",
            "vanilla",
            "static draft",
            "benchmark",
            "luận văn",
            "đồ án",
        ):
            self.assertNotIn(banned, text)

        system_prompt = requests[0]["request"]["messages"][0]["content"]
        self.assertIn("Node 3 business-analysis writer", system_prompt)
        self.assertIn("grounded, cautious reasoning", system_prompt)




if __name__ == "__main__":
    unittest.main()
