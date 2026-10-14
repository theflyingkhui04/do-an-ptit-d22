#!/usr/bin/env python3
"""Choose and validate only an actually mismatching A3 v8 Adaptive response.

Runs OFFLINE after v6 two-request driver saved responses. Never re-sends requests
or changes frozen evidence. Refuses to claim a cause if row/prefix is not proven.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.m2_03_compare_a3_v8 import InvalidEvidence, compare

VANILLA = ROOT / "experiments/runs/m2_03/diagnostics/a3_vanilla_20261013T125113Z_7nwv"


def mismatch(a: list[int], b: list[int]) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def main() -> int:
    d = Path("/tmp/m203_a3_adaptive_dir").read_text(encoding="utf-8").strip()
    root = Path(d).resolve()
    expected_parent = (ROOT / "experiments/runs/m2_03/diagnostics").resolve()
    if root.parent != expected_parent or not root.name.startswith("a3_v1_v8_adaptive_"):
        raise InvalidEvidence(f"Unexpected v8 evidence directory: {root}")
    reference = json.loads((VANILLA / "vanilla_response.json").read_text(encoding="utf-8"))
    va = reference["choices"][0]["token_ids"]
    if not isinstance(va, list) or not va:
        raise InvalidEvidence("Vanilla reference token IDs missing")
    rows = []
    for i in (1, 2):
        file = root / f"adaptive_attempt{i}.json"
        if not file.is_file():
            rows.append({"attempt": i, "status": "NO_SAVED_RESPONSE"})
            continue
        response = json.loads(file.read_text(encoding="utf-8"))
        ad = response["choices"][0]["token_ids"]
        index = mismatch(va, ad)
        report = {
            "attempt": i, "request_id": response.get("id"),
            "output_tokens": len(ad), "first_mismatch": index,
            "token_200": ad[200] if len(ad) > 200 else None,
            "status": "EXACT_MATCH" if index is None else "DIFFERENT_MISMATCH",
        }
        if index == 200:
            try:
                detailed = compare(VANILLA / "probe", root / "probe", file)
                report["status"] = "TARGET_200_EVIDENCE_VALIDATED"
                path = root / f"a3_v8_comparison_attempt{i}.json"
                with path.open("x", encoding="utf-8") as out:
                    json.dump(detailed, out, indent=2, ensure_ascii=False)
                    out.write("\n")
                report["comparison_file"] = path.name
                report["raw_argmax_equal"] = detailed["raw_argmax_equal"]
                report["max_abs_logit_diff"] = detailed["max_abs_logit_diff"]
            except (InvalidEvidence, OSError, TypeError, KeyError, ValueError) as exc:
                report["status"] = "TARGET_200_EVIDENCE_INVALID"
                report["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(report)
    summary = {
        "task": "M2-03",
        "result_kind": "DIAGNOSTIC_NOT_CANONICAL_GATE",
        "canonical_r03_unchanged": True,
        "canonical_exact_matches_original": "44/54 FAIL",
        "attempts": rows,
    }
    path = root / "a3_v8_diagnostic_summary.json"
    with path.open("x", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if all(r["status"] not in ("NO_SAVED_RESPONSE", "TARGET_200_EVIDENCE_INVALID") for r in rows) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InvalidEvidence, KeyError, OSError, ValueError, TypeError) as exc:
        print(f"M203_A3_V8_FINISH_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
