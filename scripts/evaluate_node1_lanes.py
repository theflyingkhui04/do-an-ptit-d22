"""Compare Node 1 extraction quality across serving lanes using MC-OCR labels.

Each run is given as ``label=path/to/validated_outputs.jsonl``:

    python scripts/evaluate_node1_lanes.py \
      --run vanilla_r1=experiments/runs/m4_03/m4_03_vanilla_r1/validated_outputs.jsonl \
      --run vanilla_r2=experiments/runs/m4_03/m4_03_vanilla_r2/validated_outputs.jsonl \
      --run static=experiments/runs/m4_03/m4_03_static/validated_outputs.jsonl

With both the baseline and the control run, every other lane is compared with
the baseline, and the difference between the two vanilla runs is the noise level.
A sample with no valid JSON in a run counts as blank (wrong) on labelled fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.entity_eval import (  # noqa: E402
    BLANK,
    FIELDS,
    MATCH,
    MISMATCH,
    NO_LABEL,
    UNSCORED,
    score_sample,
)

DEFAULT_LABELS = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
DEFAULT_OUTPUT_PREFIX = Path("experiments/results/m4_03_entity_match")
TEXT_FIELDS = ("SELLER", "ADDRESS")
RATE_DIGITS = 6
FLOAT_EPSILON = 1e-9


def run_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {value!r}")
    return label, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", dest="runs", action="append", type=run_argument, required=True)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--baseline", default="vanilla_r1")
    parser.add_argument("--control", default="vanilla_r2")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def summarize_run(labels: dict[str, list[dict]], outputs: dict[str, dict]) -> dict[str, Any]:
    counts = {field: {"scored": 0, "matched": 0, "blank": 0, "unscored": 0, "f1_sum": 0.0} for field in FIELDS}
    mismatches: list[dict[str, Any]] = []
    for sample_id, pairs in labels.items():
        scores = score_sample(outputs.get(sample_id), pairs)
        for field, score in scores.items():
            count = counts[field]
            if score.status == NO_LABEL:
                continue
            if score.status == UNSCORED:
                count["unscored"] += 1
                continue
            count["scored"] += 1
            count["matched"] += score.status == MATCH
            count["blank"] += score.status == BLANK
            count["f1_sum"] += score.f1 or 0.0
            if score.status in (MISMATCH, BLANK):
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        "status": score.status,
                        "gold": score.gold,
                        "predicted": score.predicted,
                        "f1": score.f1,
                    }
                )

    fields: dict[str, dict[str, Any]] = {}
    for field, count in counts.items():
        scored = count["scored"]
        fields[field] = {
            "scored": scored,
            "matched": count["matched"],
            "match_rate": round(count["matched"] / scored, RATE_DIGITS) if scored else None,
            "blank": count["blank"],
            "unscored": count["unscored"],
        }
        if field in TEXT_FIELDS:
            fields[field]["mean_f1"] = round(count["f1_sum"] / scored, RATE_DIGITS) if scored else None

    missing = sorted(set(labels) - set(outputs))
    return {
        "valid_json_rate": round((len(labels) - len(missing)) / len(labels), RATE_DIGITS),
        "missing_samples": missing,
        "fields": fields,
        "mismatches": mismatches,
    }


def compare_runs(runs: dict[str, dict], baseline: str, control: str) -> dict[str, Any] | None:
    if baseline not in runs or control not in runs:
        return None

    def rate(run: str, field: str) -> float:
        return runs[run]["fields"][field]["match_rate"] or 0.0

    noise = {field: round(abs(rate(control, field) - rate(baseline, field)), RATE_DIGITS) for field in FIELDS}
    lanes = {}
    for label in runs:
        if label in (baseline, control):
            continue
        lanes[label] = {}
        for field in FIELDS:
            delta = round(rate(label, field) - rate(baseline, field), RATE_DIGITS)
            lanes[label][field] = {
                "delta": delta,
                "within_noise": abs(delta) <= noise[field] + FLOAT_EPSILON,
            }
    return {"baseline": baseline, "control": control, "noise": noise, "lanes": lanes}


def _cell(field: str, summary: dict[str, Any]) -> str:
    if not summary["scored"]:
        return "—"
    text = f"{summary['matched']}/{summary['scored']} ({summary['match_rate']:.1%})"
    if field in TEXT_FIELDS:
        text += f", F1 {summary['mean_f1']:.3f}"
    return text


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# M4-03 — So khớp thực thể Node 1 giữa các làn",
        "",
        f"Nhãn: `{result['labels']}` · {result['sample_count']} mẫu. Ô bảng: số khớp/số mẫu chấm (tỉ lệ).",
        "",
        "## Kết quả theo lần chạy",
        "",
        "| Lần chạy | JSON hợp lệ | " + " | ".join(FIELDS) + " |",
        "|---|---|" + "---|" * len(FIELDS),
    ]
    for label, run in result["runs"].items():
        cells = [_cell(field, run["fields"][field]) for field in FIELDS]
        lines.append(f"| {label} | {run['valid_json_rate']:.1%} | " + " | ".join(cells) + " |")

    comparison = result["comparison"]
    if comparison:
        lines += [
            "",
            f"## So với {comparison['baseline']}",
            "",
            f"Dao động là |{comparison['control']} − {comparison['baseline']}| của từng trường.",
            "",
            "| Lần chạy | Trường | Chênh lệch | Dao động vanilla | Trong dao động vanilla |",
            "|---|---|---|---|---|",
        ]
        for label, fields in comparison["lanes"].items():
            for field, item in fields.items():
                verdict = "đúng" if item["within_noise"] else "sai"
                lines.append(
                    f"| {label} | {field} | {item['delta']:+.1%} | {comparison['noise'][field]:.1%} | {verdict} |"
                )

    missing = {label: run["missing_samples"] for label, run in result["runs"].items() if run["missing_samples"]}
    if missing:
        lines += ["", "## Mẫu thiếu", ""]
        lines += [f"- {label}: {', '.join(samples)}" for label, samples in missing.items()]

    lines += ["", "## Mẫu lệch", ""]
    for label, run in result["runs"].items():
        lines += [f"### {label}", ""]
        if not run["mismatches"]:
            lines += ["Không có.", ""]
            continue
        lines += ["| Mẫu | Trường | Trạng thái | Nhãn | Model |", "|---|---|---|---|---|"]
        for item in run["mismatches"]:
            gold = str(item["gold"]).replace("|", "\\|").replace("\n", " ")
            predicted = str(item["predicted"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {item['sample_id']} | {item['field']} | {item['status']} | {gold} | {predicted} |")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = {row["sample_id"]: row["key_text_pairs"] for row in read_jsonl(args.labels)}
    # Read every run before writing, so a bad path leaves no partial report.
    outputs = {
        label: {row["sample_id"]: row["payload"] for row in read_jsonl(path)}
        for label, path in args.runs
    }
    runs = {label: summarize_run(labels, rows) for label, rows in outputs.items()}
    result = {
        "task": "M4-03",
        "labels": args.labels.as_posix(),
        "sample_count": len(labels),
        "inputs": {label: path.as_posix() for label, path in args.runs},
        "runs": runs,
        "comparison": compare_runs(runs, args.baseline, args.control),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    print(f"[M4-03] wrote {args.output_prefix}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
