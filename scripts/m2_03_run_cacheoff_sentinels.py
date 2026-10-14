#!/usr/bin/env python3
"""M2-03 isolated Cache-OFF V1 sentinels; diagnostic only."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tests/verify_lossless_cacheoff.py"
OUTPUT = ROOT / "experiments/runs/m2_03_cacheoff/sentinels"

WANTED = {
    ("W1", "m0_04_007"),
    ("W3", "w3_business_v2_004"),
}


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "m203_isolated_verifier", VERIFIER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", choices=("vanilla", "adaptive"))
    args = parser.parse_args()

    v = load_verifier()
    plan = v.load_plan()

    selected = [
        p for p in plan["prompts"]
        if (p["workload"], p["sample_id"]) in WANTED
    ]

    assert len(plan["prompts"]) == 54
    assert len(selected) == 2

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if args.lane == "adaptive":
        vanilla_path = OUTPUT / "vanilla.json"
        if not vanilla_path.is_file():
            raise SystemExit(
                "STOP: Vanilla evidence missing; no GPU requests sent"
            )
        vanilla = json.loads(vanilla_path.read_text())
        if (
            vanilla.get("plan_sha256") != plan["plan_sha256"]
            or len(vanilla.get("records", [])) != 2
            or not all(r.get("ok") is True
                       for r in vanilla["records"])
        ):
            raise SystemExit("STOP: Invalid Vanilla evidence")

    path = OUTPUT / f"{args.lane}.json"
    journal = OUTPUT / f"{args.lane}.partial.jsonl"

    if journal.exists():
        raise SystemExit("STOP: Existing partial journal")

    if args.lane == "adaptive":
        v.ensure_same_provenance(
            {
                "runtime": vanilla["runtime"],
                "plan_sha256": vanilla["plan_sha256"],
            },
            v.collect_runtime(),
            plan["plan_sha256"],
        )

    if path.exists():
        raise SystemExit(
            f"STOP: Existing evidence must not be overwritten: {path}"
        )

    with requests.Session() as session:
        server = v.probe_server(session, args.lane)

        records = []
        with journal.open("x", encoding="utf-8") as handle:
            for prompt in selected:
                result = v.request_one(session, prompt, args.lane)
                records.append(result)
                handle.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"SAVED_REQUEST: {prompt['workload']}/"
                    f"{prompt['sample_id']} ok={result['ok']}",
                    flush=True,
                )
                if result["ok"] is not True:
                    raise SystemExit(
                        "STOP: Request failed; journal preserved"
                    )

        document = {
            "result_kind": "TARGETED_DIAGNOSTIC_NOT_CANONICAL",
            "lane": args.lane,
            "plan_sha256": plan["plan_sha256"],
            "runtime": v.collect_runtime(),
            "server": server,
            "records": records,
        }

    # Persist even failed requests for later diagnosis.
    with path.open("x", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("SAVED:", path)

    if not all(r["ok"] is True for r in records):
        raise SystemExit("STOP: Request failed; evidence preserved")

    if args.lane == "adaptive":
        vanilla_path = OUTPUT / "vanilla.json"
        if not vanilla_path.is_file():
            raise SystemExit("STOP: Vanilla reference missing")

        vanilla = json.loads(vanilla_path.read_text())

        assert (
            vanilla["plan_sha256"] == document["plan_sha256"]
        )

        v.ensure_same_provenance(
            {
                "runtime": vanilla["runtime"],
                "plan_sha256": vanilla["plan_sha256"],
            },
            document["runtime"],
            document["plan_sha256"],
        )

        comparisons = [
            v.compare_pair(a, b)
            for a, b in zip(
                vanilla["records"],
                document["records"],
            )
        ]

        # Identity must match independently of order.
        for a, b in zip(
            vanilla["records"], document["records"]
        ):
            assert (a["workload"], a["sample_id"]) == (
                b["workload"], b["sample_id"]
            )

        report = {
            "result_kind": "TARGETED_DIAGNOSTIC_NOT_CANONICAL",
            "comparisons": comparisons,
            "all_passed": all(c["passed"] for c in comparisons),
        }

        report_path = OUTPUT / "comparison.json"
        with report_path.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(json.dumps(report, ensure_ascii=False, indent=2))

        if not report["all_passed"]:
            raise SystemExit("SENTINEL_GATE: FAIL")

        print("SENTINEL_GATE: PASS")
    else:
        print("VANILLA_SENTINELS: PASS")


if __name__ == "__main__":
    main()
