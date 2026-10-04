from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments import aggregate_results as agg


def make_record(
    *,
    lane: str = "vanilla",
    workload: str = "W1",
    concurrency: int = 1,
    repeat_id: str = "r001",
    run_id: str | None = None,
    tpot_ms: float = 20.0,
    ttft_ms: float = 50.0,
    e2e_ms: float = 250.0,
    success: bool = True,
    metrics_complete: bool = True,
    output_tokens: int = 10,
    run_duration_s: float = 1.0,
    server_metrics: dict | None = None,
) -> dict:
    if run_id is None:
        run_id = f"run-{repeat_id}"

    return {
        "phase": "measured",
        "success": success,
        "metrics_complete": metrics_complete,
        "lane": lane,
        "workload": workload,
        "concurrency": concurrency,
        "repeat_id": repeat_id,
        "run_id": run_id,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "run_vram_peak_mb": 12000.0,
        "run_duration_s": run_duration_s,
        "output_tokens": output_tokens,
        "gpu": "Tesla T4",
        "commit": "test-sha",
        "server_metrics": server_metrics,
    }


def spec_metrics(
    *,
    accepted: int,
    drafted: int,
    steps: int,
) -> dict:
    return {
        "speculative_decoding": {
            "num_accepted_draft_tokens": accepted,
            "num_draft_tokens": drafted,
            "num_spec_steps": steps,
            "num_spec_tokens": 3,
        }
    }


class AggregateResultsContractTests(unittest.TestCase):
    def write_raw(self, records: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            delete=False,
        )

        with tmp:
            for record in records:
                tmp.write(json.dumps(record) + "\n")

        self.addCleanup(
            lambda: Path(tmp.name).unlink(missing_ok=True)
        )

        return Path(tmp.name)

    def test_cell_fails_if_any_measured_request_failed(self) -> None:
        records = [
            make_record(
                repeat_id="r001",
                run_id="run-r001",
            ),
            make_record(
                repeat_id="r001",
                run_id="run-r001",
                success=False,
                metrics_complete=False,
            ),
        ]

        raw = self.write_raw(records)

        with self.assertRaises(ValueError):
            agg.aggregate_raw_jsonl(raw)

    def test_cell_fails_if_any_measured_request_metrics_incomplete(
        self,
    ) -> None:
        records = [
            make_record(
                repeat_id="r001",
                run_id="run-r001",
            ),
            make_record(
                repeat_id="r001",
                run_id="run-r001",
                success=True,
                metrics_complete=False,
            ),
        ]

        raw = self.write_raw(records)

        with self.assertRaises(ValueError):
            agg.aggregate_raw_jsonl(raw)

    def test_cell_tpot_is_median_of_repeat_medians_not_pooled_requests(
        self,
    ) -> None:
        # Repeat-level TPOT medians:
        #
        # r001 = 10
        # r002 = 10
        # r003 = 20
        # r004 = 30
        # r005 = 40
        #
        # Cell median = 20.
        #
        # If all 15 requests were incorrectly pooled together,
        # the median would instead be 30.
        repeat_values = {
            "r001": [10.0, 10.0, 100.0],
            "r002": [10.0, 10.0, 100.0],
            "r003": [20.0, 20.0, 20.0],
            "r004": [30.0, 30.0, 30.0],
            "r005": [40.0, 40.0, 40.0],
        }

        records = []

        for repeat_id, values in repeat_values.items():
            for value in values:
                records.append(
                    make_record(
                        repeat_id=repeat_id,
                        run_id=f"run-{repeat_id}",
                        tpot_ms=value,
                    )
                )

        raw = self.write_raw(records)
        summaries = agg.aggregate_raw_jsonl(raw)

        self.assertEqual(len(summaries), 1)

        cell = summaries[0]

        self.assertEqual(cell["n_runs"], 5)
        self.assertEqual(cell["tpot_p50"], 20.0)
        self.assertEqual(cell["_tpot_min"], 10.0)
        self.assertEqual(cell["_tpot_max"], 40.0)

    def test_static_alpha_uses_summed_counters(self) -> None:
        records = [
            make_record(
                lane="static",
                repeat_id="r001",
                run_id="run-r001",
                server_metrics=spec_metrics(
                    accepted=1,
                    drafted=2,
                    steps=1,
                ),
            ),
            make_record(
                lane="static",
                repeat_id="r001",
                run_id="run-r001",
                server_metrics=spec_metrics(
                    accepted=9,
                    drafted=10,
                    steps=3,
                ),
            ),
        ]

        raw = self.write_raw(records)
        summaries = agg.aggregate_raw_jsonl(raw)

        alpha = summaries[0]["alpha"]

        self.assertIsInstance(alpha, float)
        self.assertAlmostEqual(
            alpha,
            10 / 12,
            places=9,
        )

    def test_static_tau_uses_summed_counters(self) -> None:
        records = [
            make_record(
                lane="static",
                repeat_id="r001",
                run_id="run-r001",
                server_metrics=spec_metrics(
                    accepted=1,
                    drafted=2,
                    steps=1,
                ),
            ),
            make_record(
                lane="static",
                repeat_id="r001",
                run_id="run-r001",
                server_metrics=spec_metrics(
                    accepted=9,
                    drafted=10,
                    steps=3,
                ),
            ),
        ]

        raw = self.write_raw(records)
        summaries = agg.aggregate_raw_jsonl(raw)

        tau = summaries[0]["tau"]

        self.assertIsInstance(tau, float)
        self.assertAlmostEqual(
            tau,
            1 + 10 / 4,
            places=9,
        )

    def test_static_speedup_pairs_same_workload_and_concurrency(
        self,
    ) -> None:
        rows = [
            {
                "lane": "vanilla",
                "workload": "W2",
                "concurrency": 4,
                "tpot_p50": 5.0,
                "speedup_vs_vanilla": None,
            },
            {
                "lane": "vanilla",
                "workload": "W2",
                "concurrency": 8,
                "tpot_p50": 25.0,
                "speedup_vs_vanilla": None,
            },
            {
                "lane": "vanilla",
                "workload": "W3",
                "concurrency": 8,
                "tpot_p50": 100.0,
                "speedup_vs_vanilla": None,
            },
            {
                "lane": "static",
                "workload": "W2",
                "concurrency": 8,
                "tpot_p50": 20.0,
                "speedup_vs_vanilla": None,
            },
        ]

        result = agg.apply_speedup_vs_vanilla(
            copy.deepcopy(rows)
        )

        by_key = {
            (
                row["lane"],
                row["workload"],
                row["concurrency"],
            ): row
            for row in result
        }

        self.assertEqual(
            by_key[
                ("vanilla", "W2", 8)
            ]["speedup_vs_vanilla"],
            1.0,
        )

        self.assertAlmostEqual(
            by_key[
                ("static", "W2", 8)
            ]["speedup_vs_vanilla"],
            1.25,
            places=9,
        )
        
    def test_cell_summary_preserves_pairing_provenance(self) -> None:
        record = make_record(
            lane="vanilla",
            workload="W2",
            concurrency=8,
        )
        record.update(
            {
                "experiment_id": "exp-001",
                "dataset_version": "w2.m1_03.v2",
                "workload_file_sha256": "sha-w2",
                "commit": "commit-001",
            }
        )

        raw = self.write_raw([record])
        summaries = agg.aggregate_raw_jsonl(raw)

        self.assertEqual(len(summaries), 1)
        cell = summaries[0]

        self.assertEqual(cell["_experiment_id"], "exp-001")
        self.assertEqual(cell["_dataset_version"], "w2.m1_03.v2")
        self.assertEqual(cell["_workload_file_sha256"], "sha-w2")
        self.assertEqual(cell["commit"], "commit-001")


    def test_speedup_rejects_mismatched_experiment_id(self) -> None:
        rows = [
            {
                "lane": "vanilla",
                "workload": "W2",
                "concurrency": 8,
                "tpot_p50": 25.0,
                "commit": "commit-001",
                "_experiment_id": "exp-A",
                "_dataset_version": "w2.m1_03.v2",
                "_workload_file_sha256": "sha-w2",
            },
            {
                "lane": "static",
                "workload": "W2",
                "concurrency": 8,
                "tpot_p50": 20.0,
                "commit": "commit-001",
                "_experiment_id": "exp-B",
                "_dataset_version": "w2.m1_03.v2",
                "_workload_file_sha256": "sha-w2",
            },
        ]

        with self.assertRaises(ValueError):
            agg.apply_speedup_vs_vanilla(rows)


    def test_speedup_rejects_mismatched_commit(self) -> None:
        rows = [
            {
                "lane": "vanilla",
                "workload": "W2",
                "concurrency": 8,
                "tpot_p50": 25.0,
                "commit": "commit-A",
                "_experiment_id": "exp-001",
                "_dataset_version": "w2.m1_03.v2",
                "_workload_file_sha256": "sha-w2",
            },
            {
                "lane": "static",
                "workload": "W2",
                "concurrency": 8,
                "tpot_p50": 20.0,
                "commit": "commit-B",
                "_experiment_id": "exp-001",
                "_dataset_version": "w2.m1_03.v2",
                "_workload_file_sha256": "sha-w2",
            },
        ]

        with self.assertRaises(ValueError):
            agg.apply_speedup_vs_vanilla(rows)


    def test_speedup_rejects_mismatched_workload_provenance(self) -> None:
        vanilla = {
            "lane": "vanilla",
            "workload": "W2",
            "concurrency": 8,
            "tpot_p50": 25.0,
            "commit": "commit-001",
            "_experiment_id": "exp-001",
            "_dataset_version": "w2.m1_03.v2",
            "_workload_file_sha256": "sha-A",
        }

        static = {
            "lane": "static",
            "workload": "W2",
            "concurrency": 8,
            "tpot_p50": 20.0,
            "commit": "commit-001",
            "_experiment_id": "exp-001",
            "_dataset_version": "w2.m1_03.v2",
            "_workload_file_sha256": "sha-B",
        }

        with self.assertRaises(ValueError):
            agg.apply_speedup_vs_vanilla(
                [vanilla, static]
            )

        static["_workload_file_sha256"] = "sha-A"
        static["_dataset_version"] = "w2.WRONG"

        with self.assertRaises(ValueError):
            agg.apply_speedup_vs_vanilla(
                [vanilla, static]
            )


if __name__ == "__main__":
    unittest.main()
