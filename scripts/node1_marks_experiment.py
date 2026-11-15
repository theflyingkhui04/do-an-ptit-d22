"""[M5-01 Node 1 sửa dấu, 14/11] Build the request arms and score them (design §8 of m5-01-canonical-flow).

    python scripts/node1_marks_experiment.py build --run-id 20261114-m501-node1-marks
    # then, per arm and set (lane vanilla, no DB):
    python scripts/run_node1_extractor.py --lane vanilla --backend modal --source-policy restore --no-db \
        --requests <run>/<set>/requests_<arm>.jsonl --package-manifest <run>/<set>/manifest_<arm>.json \
        --run-id <set>_<arm> --output-dir <run>/<set>/out_<arm> --hardware modal-A10G
    python scripts/node1_marks_experiment.py score --run-id 20261114-m501-node1-marks --set select
    python scripts/node1_marks_experiment.py score --run-id 20261114-m501-node1-marks --set devset

Sets: ``select`` = the 60 images of the OCR preprocessing choice (OCR text of P0, labels rebuilt from MC-OCR and
checked against the image ids of that report); ``devset`` = the 30 M0-04 images (OCR text of the 08/11 run, engine A).
The rule runs on ``select``; ``devset`` runs once afterwards for B0 and the chosen arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import node1_marks as marks

RUNS = Path("experiments/runs/m5_01")
REPORT = Path("experiments/results/m5_01_node1_marks_choice.json")
OCR_CHOICE = Path("experiments/results/m5_01_ocr_preprocess_choice.json")
SELECT_OCR = RUNS / "20261114-m501-ocr-preprocess/select/P0.jsonl"
DEVSET_OCR = RUNS / "20261108-m501-ocrlib-choice/ocr_lib_A.jsonl"
DEVSET = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
MODEL = "qwen2.5-7b-vanilla"  # recorded in the requests; the runner sends the lane's served model


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_labels() -> list[dict[str, Any]]:
    from experiments import modal_ocr_preprocess as ocr

    devset_ids = {r["img_id"] for r in read_jsonl(DEVSET)}
    selection = ocr.selection_set(ocr._train_rows(), devset_ids)
    frozen = json.loads((REPO_ROOT / OCR_CHOICE).read_text(encoding="utf-8"))["selection"]["img_ids"]
    if [s["img_id"] for s in selection] != frozen:
        raise SystemExit("selection set differs from the OCR preprocessing report")
    return [{"sample_id": s["sample_id"], "img_id": s["img_id"], "key_text_pairs": s["key_text_pairs"]}
            for s in selection]


def build(run_id: str) -> None:
    run_dir = REPO_ROOT / RUNS / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; use a new --run-id")
    sets = {
        "select": (read_jsonl(SELECT_OCR), select_labels()),
        "devset": (read_jsonl(DEVSET_OCR), [{k: r[k] for k in ("sample_id", "img_id", "key_text_pairs")}
                                             for r in read_jsonl(DEVSET)]),
    }
    for name, (ocr_rows, labels) in sets.items():
        if [r["sample_id"] for r in ocr_rows] != [r["sample_id"] for r in labels]:
            raise SystemExit(f"{name}: OCR rows and labels are not the same samples")
        out = run_dir / name
        out.mkdir(parents=True)
        write_jsonl(out / "labels.jsonl", labels)
        for arm in marks.ARMS:
            sha = write_jsonl(out / f"requests_{arm}.jsonl", marks.arm_requests(ocr_rows, arm, MODEL))
            manifest = {"workload_id": "m5_01_node1_marks", "dataset_version": f"node1_marks.{name}.{arm}",
                        "package_sha256": sha, "file_hashes": {f"requests_{arm}.jsonl": sha}}
            (out / f"manifest_{arm}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(labels)} samples, arms {', '.join(marks.ARMS)} -> {out.relative_to(REPO_ROOT)}")


def arm_score(set_dir: Path, arm: str, labels: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    out_dir = set_dir / f"out_{arm}"
    outputs = {row["sample_id"]: row["payload"] for row in read_jsonl((out_dir / "validated_outputs.jsonl")
                                                                       .relative_to(REPO_ROOT))}
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    return {**marks.mark_counts(labels, outputs), "prompt_version": summary["prompt_version"],
            "median_latency_ms": summary["latency_ms"]["median"], "prompt_tokens": summary["usage"]["prompt_tokens"],
            "completion_tokens": summary["usage"]["completion_tokens"]}


def score(run_id: str, set_name: str) -> None:
    set_dir = REPO_ROOT / RUNS / run_id / set_name
    labels = {r["sample_id"]: r["key_text_pairs"] for r in read_jsonl((set_dir / "labels.jsonl").relative_to(REPO_ROOT))}
    report_path = REPO_ROOT / REPORT
    provenance = {"source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                                  text=True, check=False).stdout.strip(),
                  "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if set_name == "select":
        if report_path.exists():
            raise SystemExit(f"{report_path} exists: the choice is made once")
        scores = {arm: arm_score(set_dir, arm, labels) for arm in marks.ARMS}
        report = {"task": "M5-01", "step": "node1-marks-choice", "run_id": run_id,
                  "design": "openspec m5-01-canonical-flow §8 (local), approved by Dương 14/11",
                  "rule": "B2 only if: B2 exact > B1 exact; B2 stripped >= B0 stripped - 1; "
                          "B2 valid JSON >= B0 valid JSON - 1; B2 right-letters-wrong-marks <= 5% of B2 stripped",
                  "fields": list(marks.TEXT_FIELDS), "lane": "vanilla", "scores_selection": scores,
                  "decision": marks.choose_arm(scores), **provenance}
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "scores_devset" in report:
            raise SystemExit("the devset check already ran")
        arms = sorted({"B0", report["decision"]["chosen"]})
        report["scores_devset"] = {arm: arm_score(set_dir, arm, labels) for arm in arms}
        report["devset_provenance"] = provenance
        scores = report["scores_devset"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for arm, s in scores.items():
        print(arm, {k: s[k] for k in ("valid_json", "labelled_fields", "exact", "stripped",
                                      "right_letters_wrong_marks", "median_latency_ms", "prompt_tokens")})
    if set_name == "select":
        print(json.dumps(report["decision"], ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "score"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--set", choices=("select", "devset"), default="select")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.run_id)
    else:
        score(args.run_id, args.set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
