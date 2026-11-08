"""[M5-01 canonical flow] Build Workload W1_v3: Node 1 requests over library OCR text without marks.

Same 30 M0-04 samples, prompt node1_extractor.v2, max_tokens 3072 and request shape as
W1_v2, but ``raw_text`` comes from the library OCR engine chosen once in
experiments/results/m5_01_ocr_lib_choice.json (PaddleOCR detection + VietOCR recognition)
and passes through ``strip_vietnamese_marks``. W1 v1 and W1_v2 stay frozen.

    python scripts/build_w1_v3_workload.py --ocr-jsonl experiments/runs/m5_01/<run>/ocr_lib_A.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_node1_requests import SCHEMA_VERSION, build_chat_request, build_node1_input_text, load_jsonl, normalize_text  # noqa: I001
from build_w1_workload import make_provenance_row, write_manifest
from src.pipeline.text_normalize import strip_vietnamese_marks
from src.pipeline.w1_workload import (
    HASH_POLICY_LF,
    W1_V2_PROMPT_VERSION,
    W1_V3_DATASET_VERSION,
    W1_WORKLOAD_ID,
    artifact_sha256,
    compose_node1_prompt_v2,
    compute_package_sha256,
    repo_relative,
    sha256_text,
    write_jsonl,
)

DEFAULT_SOURCE_SAMPLES = Path("data/processed/m0_04_mcocr_devset/samples.jsonl")
DEFAULT_SOURCE_MANIFEST = Path("experiments/results/m0_04_mcocr_devset_manifest.csv")
DEFAULT_V1_PROMPT = Path("src/pipeline/prompts/node1_extractor_system.md")
DEFAULT_RULES = Path("src/pipeline/prompts/node1_full_receipt_rules.md")
DEFAULT_CHOICE = Path("experiments/results/m5_01_ocr_lib_choice.json")
DEFAULT_OUTPUT_DIR = Path("experiments/configs/w1_v3")
DEFAULT_REPORT_DIR = Path("experiments/results")
FROZEN_DIRS = ("w1", "w1_v2", "w2", "w2_v2", "w3", "w3_v2")
DEFAULT_MODEL = "qwen2.5:7b-instruct"
MAX_TOKENS = 3072
REQUIRES_MAX_MODEL_LEN = 8192
MARK_STRIPPING = "strip_vietnamese_marks (src/pipeline/text_normalize.py): NFD, drop combining marks, đ->d, Đ->D, NFC"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-jsonl", type=Path, required=True)
    parser.add_argument("--choice-report", type=Path, default=DEFAULT_CHOICE)
    parser.add_argument("--source-samples", type=Path, default=DEFAULT_SOURCE_SAMPLES)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--v1-prompt", type=Path, default=DEFAULT_V1_PROMPT)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def ocr_by_sample(ocr_rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    known = {s["sample_id"] for s in samples}
    by_id: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for row in ocr_rows:
        sample_id = row.get("sample_id")
        if sample_id not in known:
            problems.append(f"{sample_id}: not in the M0-04 devset")
        elif sample_id in by_id:
            problems.append(f"{sample_id}: duplicate OCR row")
        else:
            by_id[sample_id] = row
    for sample in samples:
        row = by_id.get(sample["sample_id"])
        if row is None:
            problems.append(f"{sample['sample_id']}: missing OCR row")
        elif row.get("error"):
            problems.append(f"{sample['sample_id']}: OCR error {row['error']!r}")
        elif not normalize_text(row.get("text")):
            problems.append(f"{sample['sample_id']}: empty OCR text")
        elif row.get("img_id") != sample.get("img_id"):
            problems.append(f"{sample['sample_id']}: img_id {row.get('img_id')!r} != {sample.get('img_id')!r}")
    if problems:
        raise SystemExit("W1_v3 build refused:\n- " + "\n- ".join(problems))
    return by_id


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve(args.output_dir).resolve()
    if output_dir in {(REPO_ROOT / "experiments/configs" / n).resolve() for n in FROZEN_DIRS}:
        raise SystemExit(f"Refusing to write into frozen workload package {output_dir}")
    samples = load_jsonl(resolve(args.source_samples))
    ocr_path = resolve(args.ocr_jsonl)
    by_id = ocr_by_sample(load_jsonl(ocr_path), samples)
    choice_path = resolve(args.choice_report)
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    engine = choice["chosen_engine"]

    system_prompt = compose_node1_prompt_v2(resolve(args.v1_prompt).read_text(encoding="utf-8"),
                                            resolve(args.rules).read_text(encoding="utf-8"))
    sample_rows = []
    for sample in samples:
        sample_rows.append({
            "workload_id": W1_WORKLOAD_ID, "dataset_version": W1_V3_DATASET_VERSION,
            "sample_id": sample["sample_id"], "source_row_idx": sample.get("source_row_idx"),
            "img_id": sample.get("img_id"), "image_path": sample.get("image_path"),
            "source_image_path": sample.get("source_image_path"), "source_csv": sample.get("source_csv"),
            "anno_image_quality": sample.get("anno_image_quality"),
            "raw_text": normalize_text(strip_vietnamese_marks(by_id[sample["sample_id"]]["text"])),
        })
    provenance_rows = []
    for sample in samples:
        row = make_provenance_row(sample, args.source_samples)
        ocr_row = by_id[sample["sample_id"]]
        row.update(dataset_version=W1_V3_DATASET_VERSION, upstream_task="M0-04 + M5-01 library OCR")
        row["ocr"] = {"engine": engine, **{k: ocr_row.get(k) for k in ("image_sha256", "boxes", "latency_ms")}}
        provenance_rows.append(row)
    input_rows = [{"workload_id": W1_WORKLOAD_ID, "dataset_version": W1_V3_DATASET_VERSION, "sample_id": r["sample_id"],
                   "img_id": r["img_id"], "image_path": r["image_path"], "node1_input_text": build_node1_input_text(r)}
                  for r in sample_rows]
    request_rows = []
    for row in sample_rows:
        request = build_chat_request(row, system_prompt, args.model, MAX_TOKENS, 0.0, 42)
        request["prompt_version"] = W1_V2_PROMPT_VERSION
        request_rows.append({"workload_id": W1_WORKLOAD_ID, "dataset_version": W1_V3_DATASET_VERSION, **request})

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = resolve(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "samples_jsonl": output_dir / "samples.jsonl", "provenance_jsonl": output_dir / "provenance.jsonl",
        "manifest_csv": output_dir / "manifest.csv", "node1_inputs_jsonl": output_dir / "node1_inputs.jsonl",
        "node1_chat_requests_jsonl": output_dir / "node1_chat_requests.jsonl",
        "system_prompt": output_dir / "system_prompt.md",
        "json_schema": REPO_ROOT / "src/pipeline/schemas/invoice_extraction.schema.json",
    }
    paths["system_prompt"].write_text(system_prompt + "\n", encoding="utf-8", newline="\n")
    write_jsonl(paths["samples_jsonl"], sample_rows)
    write_jsonl(paths["provenance_jsonl"], provenance_rows)
    write_manifest(paths["manifest_csv"], sample_rows, provenance_rows)
    write_jsonl(paths["node1_inputs_jsonl"], input_rows)
    write_jsonl(paths["node1_chat_requests_jsonl"], request_rows)

    label_totals: Counter[str] = Counter()
    for row in provenance_rows:
        label_totals.update(row["annotation_label_counts"])
    file_hashes = {name: artifact_sha256(name, path, HASH_POLICY_LF) for name, path in paths.items()}
    metadata = {
        "workload_id": W1_WORKLOAD_ID, "dataset_version": W1_V3_DATASET_VERSION, "schema_version": SCHEMA_VERSION,
        "prompt_version": W1_V2_PROMPT_VERSION, "sample_count": len(sample_rows), "model": args.model,
        "temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": 42, "response_format": {"type": "json_object"},
        "hash_policy": HASH_POLICY_LF, "requires_max_model_len": REQUIRES_MAX_MODEL_LEN,
        "mark_stripping": MARK_STRIPPING,
        "ocr": {"kind": "library", "engine": engine, "engine_description": choice["engines"][engine],
                "versions": choice["versions"], "choice_report": repo_relative(choice_path, REPO_ROOT),
                "choice_report_sha256": artifact_sha256("choice", choice_path, HASH_POLICY_LF),
                "ocr_jsonl": repo_relative(ocr_path, REPO_ROOT),
                "ocr_jsonl_sha256": artifact_sha256("ocr_jsonl", ocr_path, HASH_POLICY_LF)},
        "label_totals_audit_only": dict(sorted(label_totals.items())),
        "no_label_leakage_policy": ("samples.jsonl and node1_chat_requests.jsonl exclude labels/key_text_pairs; "
                                    "annotation labels are kept only in provenance.jsonl for audit."),
        "freeze_decision": "W1_v3 (M5-01 canonical flow): library OCR text without marks; W1 v1 and W1_v2 stay frozen",
    }
    package_sha256 = compute_package_sha256(file_hashes, metadata)
    manifest = {**metadata, "package_sha256": package_sha256,
                "artifacts": {name: repo_relative(path, REPO_ROOT) for name, path in paths.items()},
                "file_hashes": file_hashes,
                "source_artifacts": {"source_samples": repo_relative(resolve(args.source_samples), REPO_ROOT),
                                     "source_manifest": repo_relative(resolve(args.source_manifest), REPO_ROOT),
                                     "source_system_prompt": repo_relative(resolve(args.v1_prompt), REPO_ROOT),
                                     "source_full_receipt_rules": repo_relative(resolve(args.rules), REPO_ROOT),
                                     "ocr_jsonl": metadata["ocr"]["ocr_jsonl"]}}
    (output_dir / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8", newline="\n")
    (output_dir / "README.md").write_text("\n".join([
        "# Workload W1_v3 - Node 1 on library OCR text without Vietnamese marks", "",
        f"- Workload ID: `{W1_WORKLOAD_ID}` (label `W1_V3`)", f"- Dataset version: `{W1_V3_DATASET_VERSION}`",
        f"- Sample count: `{len(sample_rows)}`", f"- Package SHA-256: `{package_sha256}`",
        f"- Prompt: `{W1_V2_PROMPT_VERSION}`, SHA-256 `{file_hashes['system_prompt']}`",
        f"- OCR: engine {engine} ({choice['engines'][engine]}), source `{metadata['ocr']['ocr_jsonl']}`",
        f"- Marks: {MARK_STRIPPING}", "",
        f"Requests need a server with `--max-model-len` >= {REQUIRES_MAX_MODEL_LEN}.",
        "Check: `python scripts/check_w1_dataset.py --w1-dir experiments/configs/w1_v3`.", ""]),
        encoding="utf-8", newline="\n")
    summary = {"task": "M5-01", "step": "build-w1-v3-workload", "dataset_version": W1_V3_DATASET_VERSION,
               "sample_count": len(sample_rows), "package_sha256": package_sha256, "ocr_engine": engine,
               "raw_text_sha256": {r["sample_id"]: sha256_text(r["raw_text"]) for r in sample_rows}}
    (report_dir / "m5_01_w1_v3_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                                encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("dataset_version", "sample_count", "package_sha256", "ocr_engine")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
