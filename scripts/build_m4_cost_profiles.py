"""Build one compatible GPU profile from a complete M4 campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.m4_cost_profiler import build_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-ops-prior-ms", type=float, required=True,
                        help="Explicit unmeasured bookkeeping prior; 0 omits this unknown cost")
    args = parser.parse_args()
    env = json.loads((args.run_dir / "env.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (args.run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    profile = build_profile(env, records, tree_ops_prior_ms=args.tree_ops_prior_ms)
    # Existing canonical profiles may be replaced deliberately only after the
    # caller reviews this artifact; never overwrite evidence or profile files.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Validated 30 canonical cells: {args.output}")


if __name__ == "__main__":
    main()
