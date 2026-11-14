"""Write a component report only; canonical profile generation is disabled.

The filename is retained for continuity. No cost_matrix_ms or gamma coefficient
is emitted until the team aligns verification and provenance contracts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.m4_cost_profiler import build_component_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = json.loads((args.run_dir / "env.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (args.run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    report = build_component_report(env, records)
    canonical_root = Path(__file__).resolve().parents[1] / "src" / "flexsds" / "profiles"
    if (args.output.resolve().is_relative_to(canonical_root) or
            args.output.name.startswith(("profile_", "sm_"))):
        parser.error("Component reports cannot be written as canonical GPU profiles")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote component report (not a consumer profile): {args.output}")


if __name__ == "__main__":
    main()
