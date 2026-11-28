from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pytest

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


# Known vLLM 0.30 artifact: a grammar-terminated draft after EOS is verified unmasked and counted
# as accepted, while the per-request drafted count excludes it. Counters below are the real record
# req-05a67192 of flexsds-fast-a10-2611b, cell A10__adaptive__W1_V3__c32.
ARTIFACT_REQUEST_ID = "req-05a67192-4f84-44a0-b9d7-ad4f062add0f"


def artifact_record(**overrides) -> dict:
    spec = {
        "num_accepted_draft_tokens": 4,
        "num_draft_tokens": 3,
        "num_spec_steps": 4,
        "num_spec_tokens": 4,
        "acceptance_histogram": [1, 2, 1, 0, 0],
    }
    spec.update(overrides.pop("spec", {}))
    record = make_record(lane=overrides.pop("lane", "adaptive"), output_tokens=1073,
                         server_metrics={"speculative_decoding": spec})
    record.update({"request_id": ARTIFACT_REQUEST_ID, "sample_id": "m0_04_006",
                   "response_format": {"type": "json_object"}, "max_completion_tokens": 3072})
    record.update(overrides)
    return record


def normal_record(index: int = 0, *, lane: str = "adaptive") -> dict:
    record = make_record(lane=lane, server_metrics={"speculative_decoding": {
        "num_accepted_draft_tokens": 9, "num_draft_tokens": 15, "num_spec_steps": 5,
        "num_spec_tokens": 3, "acceptance_histogram": [1, 1, 1, 2]}})
    record.update({"request_id": f"req-normal-{index}", "sample_id": "m0_04_001",
                   "response_format": {"type": "json_object"}, "max_completion_tokens": 3072})
    return record


def test_grammar_post_stop_overcount_is_classified():
    assert agg.classify_spec_counter_record(artifact_record()) == agg.GRAMMAR_POST_STOP_OVERCOUNT
    assert agg.GRAMMAR_POST_STOP_OVERCOUNT == "grammar_post_stop_overcount"


def test_normal_counters_are_not_classified():
    assert agg.classify_spec_counter_record(normal_record()) is None


@pytest.mark.parametrize("response_format", [None, {"type": "text"}])
def test_overcount_without_grammar_is_rejected(response_format):
    with pytest.raises(ValueError, match="C3") as info:
        agg.classify_spec_counter_record(artifact_record(response_format=response_format))
    assert ARTIFACT_REQUEST_ID in str(info.value)


def test_overcount_beyond_k_minus_one_is_rejected():
    record = artifact_record(spec={"num_accepted_draft_tokens": 7, "num_spec_steps": 2,
                                   "acceptance_histogram": [0, 0, 0, 1, 1]})
    with pytest.raises(ValueError, match="C4"):
        agg.classify_spec_counter_record(record)


def test_overcount_at_length_limit_is_rejected():
    with pytest.raises(ValueError, match="C5"):
        agg.classify_spec_counter_record(artifact_record(output_tokens=3072))


@pytest.mark.parametrize("spec", [{"acceptance_histogram": [1, 2, 1, 0]},
                                  {"acceptance_histogram": [1, 1, 2, 0, 0]},
                                  {"num_spec_steps": 5}])
def test_inconsistent_histogram_is_rejected(spec):
    with pytest.raises(ValueError, match="C6"):
        agg.classify_spec_counter_record(artifact_record(spec=spec))


def test_overcount_without_num_spec_tokens_is_rejected():
    record = artifact_record()
    del record["server_metrics"]["speculative_decoding"]["num_spec_tokens"]
    with pytest.raises(ValueError, match="C1"):
        agg.classify_spec_counter_record(record)


def test_exclusion_share_boundary():
    def cell(n_artifacts: int) -> list[dict]:
        return [artifact_record(request_id=f"req-a{i}") for i in range(n_artifacts)] + [
            normal_record(i) for i in range(160 - n_artifacts)]

    kept, excluded = agg.partition_spec_counter_records(cell(8))
    assert len(kept) == 152 and len(excluded) == 8
    with pytest.raises(ValueError, match="exclusion share"):
        agg.partition_spec_counter_records(cell(9))


def test_partition_keeps_normal_records_in_order():
    # 1 excluded record in 20 is exactly the 5 % limit.
    records = [normal_record(0), artifact_record(repeat_id="r004")] + [normal_record(i) for i in range(1, 19)]
    kept, excluded = agg.partition_spec_counter_records(records)
    assert [r["request_id"] for r in kept] == [f"req-normal-{i}" for i in range(19)]
    assert excluded == [{
        "request_id": ARTIFACT_REQUEST_ID, "repeat_id": "r004", "sample_id": "m0_04_006",
        "reason": "grammar_post_stop_overcount", "num_accepted_draft_tokens": 4,
        "num_draft_tokens": 3, "num_spec_tokens": 4,
    }]


def test_static_lane_tolerates_artifact_only_when_opted_in():
    records = [artifact_record(lane="static")] + [normal_record(i, lane="static") for i in range(19)]
    row = agg._aggregate_measured_records(records, tolerate_known_counter_artifacts=True)[0]
    assert row["alpha"] == pytest.approx(9 / 15)
    assert row["tau"] == pytest.approx(1 + 9 / 5)
    with pytest.raises(ValueError, match="cannot exceed"):
        agg._aggregate_measured_records(records)


if __name__ == "__main__":
    unittest.main()
