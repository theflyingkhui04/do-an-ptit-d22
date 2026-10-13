"""Offline, read-only join and step #13 inspection for M2-03 traces."""
from pathlib import Path
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "experiments/runs/m2_03/diagnostics/k4_steptrace_coldstart_w3_004"


def load(name):
    path = DIAG / name
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    probes = load("probes.jsonl")
    traces = load("step_trace.jsonl")
    telemetry = load("telemetry.jsonl")
    assert len(probes) == 2, f"expected 2 probes, got {len(probes)}"
    assert traces, "missing trace rows"

    for probe in probes:
        rid = probe["response_id"]
        joined_ids = [t["request_id"] for t in traces if t["request_id"] == rid or t["request_id"].startswith(rid + "-")]
        engine_ids = set(joined_ids)
        assert len(engine_ids) == 1, f"trace join for {rid}: {engine_ids}"
        engine_id = next(iter(engine_ids))
        rows = sorted([t for t in traces if t["request_id"] == engine_id], key=lambda t: t["spec_step_index"])
        keys = [t["spec_step_index"] for t in rows]
        assert len(set(keys)) == len(keys), "duplicate step indices"
        assert keys == list(range(1, len(keys) + 1)), "missing/gapped step indices"
        spec = probe["speculative_metrics"]
        assert len(rows) == spec["num_spec_steps"], "step count != response metrics"
        assert [t["num_accepted"] for t in rows] == spec["per_step_accepted"], "accepted mismatch"
        assert [t["num_drafted_effective"] for t in rows] == spec["per_step_drafted"], "drafted mismatch"
        assert sum(t["num_accepted"] for t in rows) == spec["num_accepted_draft_tokens"]

        telemetry_matches = [t for t in telemetry if t.get("request_id") == engine_id]
        assert len(telemetry_matches) == 1, f"telemetry join for {rid} not unique"

        print("\n=== RUN", probe["attempt"], "===")
        print("ENGINE_ID:", engine_id)
        print("TRACE_STEPS:", len(rows))
        print("MATCH_VANILLA:", probe["match_vanilla"])
        print("FIRST_MISMATCH:", probe["first_mismatch_vanilla"])
        for row in rows:
            if row["spec_step_index"] in (12, 13, 14):
                print("STEP:", row["spec_step_index"])
                print("OUTPUT_RANGE:", (row["output_start_index"], row["output_end_index_exclusive"]))
                print("SCHEDULED_DRAFT_IDS:", row["scheduled_draft_token_ids"])
                print("RAW_GENERATED_IDS:", row["generated_token_ids_raw"])
                print("EMITTED_IDS:", row["emitted_token_ids"])
                print("ACCEPTED:", row["num_accepted"])
                print("DRAFTED:", row["num_drafted_effective"])
        target = rows[12]
        first = probe["first_mismatch_vanilla"]
        print("STEP13_COVERS_MISMATCH:",
              first is not None and target["output_start_index"] <= first < target["output_end_index_exclusive"])
        print("STEP13_IDS:", target["scheduled_draft_token_ids"], target["generated_token_ids_raw"])

    print("\nTRACE_JOIN_AND_METRICS: PASS")


if __name__ == "__main__":
    main()
