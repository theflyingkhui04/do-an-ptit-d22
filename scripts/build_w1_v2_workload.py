"""[M5-01] Build Workload W1_v2: Node 1 requests over full-receipt OCR text.

W1 v1 (experiments/configs/w1) stays frozen. W1_v2 keeps the same 30 M0-04
samples and request shape, but ``raw_text`` is the OCR text of the whole receipt
(experiments/modal_ocr.py), the prompt is node1_extractor.v2 and max_tokens 3072.

    python scripts/build_w1_v2_workload.py --ocr-jsonl experiments/runs/m5_01/<run>/ocr_texts.jsonl
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

from build_node1_requests import (  # noqa: E402
    SCHEMA_VERSION,
    build_chat_request,
    build_node1_input_text,
    load_jsonl,
    normalize_text,
)
from build_w1_workload import make_provenance_row, write_manifest  # noqa: E402
from src.pipeline.safe_paths import contained_path  # noqa: E402
from src.pipeline.w1_workload import (  # noqa: E402
    HASH_POLICY_LF,
    W1_V2_DATASET_VERSION,
    W1_V2_PROMPT_VERSION,
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
DEFAULT_OUTPUT_DIR = Path("experiments/configs/w1_v2")
DEFAULT_REPORT_DIR = Path("experiments/results")
FROZEN_DIRS = ("w1", "w2", "w2_v2", "w3", "w3_v2")
DEFAULT_MODEL = "qwen2.5:7b-instruct"
MAX_TOKENS = 3072
REQUIRES_MAX_MODEL_LEN = 8192
OCR_KEYS = ("ocr_model", "ocr_revision", "ocr_prompt_version", "ocr_prompt_sha256")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr-jsonl", type=Path, required=True)
    parser.add_argument("--source-samples", type=Path, default=DEFAULT_SOURCE_SAMPLES)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--v1-prompt", type=Path, default=DEFAULT_V1_PROMPT)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    """Command-line path inside the repository or the temp directory (relative = repository root)."""
    try:
        return contained_path(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def ocr_by_sample(ocr_rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index OCR rows by sample_id; refuse anything a benchmark cannot use."""
    known = {sample["sample_id"] for sample in samples}
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
        sample_id = sample["sample_id"]
        row = by_id.get(sample_id)
        if row is None:
            problems.append(f"{sample_id}: missing OCR row")
        elif row.get("error"):
            problems.append(f"{sample_id}: OCR error {row['error']!r}")
        elif not normalize_text(row.get("text")):
            problems.append(f"{sample_id}: empty OCR text")
        elif row.get("finish_reason") != "stop":
            problems.append(f"{sample_id}: finish_reason={row.get('finish_reason')!r} (truncated OCR)")
        elif row.get("img_id") != sample.get("img_id"):
            problems.append(f"{sample_id}: img_id {row.get('img_id')!r} != {sample.get('img_id')!r}")
    identities = {tuple(row.get(key) for key in OCR_KEYS) for row in by_id.values()}
    if len(identities) > 1:
        problems.append(f"OCR rows come from more than one model/prompt: {sorted(identities)}")
    if problems:
        raise SystemExit("W1_v2 build refused:\n- " + "\n- ".join(problems))
    return by_id


def make_sample_row(sample: dict[str, Any], ocr_text: str) -> dict[str, Any]:
    return {
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_V2_DATASET_VERSION,
        "sample_id": sample["sample_id"],
        "source_row_idx": sample.get("source_row_idx"),
        "img_id": sample.get("img_id"),
        "image_path": sample.get("image_path"),
        "source_image_path": sample.get("source_image_path"),
        "source_csv": sample.get("source_csv"),
        "anno_image_quality": sample.get("anno_image_quality"),
        "raw_text": normalize_text(ocr_text),
    }


def make_v2_provenance_row(sample: dict[str, Any], source_samples: Path, ocr_row: dict[str, Any]) -> dict[str, Any]:
    row = make_provenance_row(sample, source_samples)
    row["dataset_version"] = W1_V2_DATASET_VERSION
    row["upstream_task"] = "M0-04 + M5-01 OCR"
    row["ocr"] = {
        key: ocr_row.get(key)
        for key in (*OCR_KEYS, "image_sha256", "scaled_sha256", "scale", "scaled_size",
                    "finish_reason", "prompt_tokens", "completion_tokens", "latency_ms")
    }
    return row


def readme_text(package_sha256: str, prompt_sha256: str, sample_count: int, ocr_jsonl: str) -> str:
    return "\n".join([
        "# Workload W1_v2 - Node 1 Invoice Extraction on full-receipt OCR text",
        "",
        f"- Workload ID: `{W1_WORKLOAD_ID}` (label `W1_V2`)",
        f"- Dataset version: `{W1_V2_DATASET_VERSION}`",
        f"- Sample count: `{sample_count}`",
        f"- Package SHA-256: `{package_sha256}`",
        f"- Prompt: `{W1_V2_PROMPT_VERSION}`, SHA-256 `{prompt_sha256}`",
        f"- OCR source: `{ocr_jsonl}`",
        "",
        "Same 30 M0-04 samples and request shape as W1 v1, but `raw_text` is the OCR text of",
        "the whole receipt (Qwen2.5-VL, `src/pipeline/ocr.py`) and `max_tokens` is 3072.",
        "",
        f"Requests need a server with `--max-model-len` >= {REQUIRES_MAX_MODEL_LEN}:",
        f"deploy the Modal lanes with `PTIT_MAX_MODEL_LEN={REQUIRES_MAX_MODEL_LEN}`.",
        "",
        "Check: `python scripts/check_w1_dataset.py --w1-dir experiments/configs/w1_v2`.",
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve(args.output_dir).resolve()
    frozen = {(REPO_ROOT / "experiments/configs" / name).resolve() for name in FROZEN_DIRS}
    if output_dir in frozen:
        raise SystemExit(f"Refusing to write into frozen workload package {output_dir}")

    samples = load_jsonl(resolve(args.source_samples))
    ocr_path = resolve(args.ocr_jsonl)
    by_id = ocr_by_sample(load_jsonl(ocr_path), samples)

    system_prompt = compose_node1_prompt_v2(
        resolve(args.v1_prompt).read_text(encoding="utf-8"),
        resolve(args.rules).read_text(encoding="utf-8"),
    )
    sample_rows = [make_sample_row(sample, by_id[sample["sample_id"]]["text"]) for sample in samples]
    provenance_rows = [make_v2_provenance_row(s, args.source_samples, by_id[s["sample_id"]]) for s in samples]
    input_rows = [
        {"workload_id": W1_WORKLOAD_ID, "dataset_version": W1_V2_DATASET_VERSION, "sample_id": row["sample_id"],
         "img_id": row["img_id"], "image_path": row["image_path"], "node1_input_text": build_node1_input_text(row)}
        for row in sample_rows
    ]
    request_rows = []
    for row in sample_rows:
        request = build_chat_request(row, system_prompt, args.model, MAX_TOKENS, 0.0, 42)
        request["prompt_version"] = W1_V2_PROMPT_VERSION
        request_rows.append({"workload_id": W1_WORKLOAD_ID, "dataset_version": W1_V2_DATASET_VERSION, **request})

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = resolve(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "samples_jsonl": output_dir / "samples.jsonl",
        "provenance_jsonl": output_dir / "provenance.jsonl",
        "manifest_csv": output_dir / "manifest.csv",
        "node1_inputs_jsonl": output_dir / "node1_inputs.jsonl",
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
    first_ocr = next(iter(by_id.values()))
    file_hashes = {name: artifact_sha256(name, path, HASH_POLICY_LF) for name, path in paths.items()}
    metadata = {
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_V2_DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": W1_V2_PROMPT_VERSION,
        "sample_count": len(sample_rows),
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "seed": 42,
        "response_format": {"type": "json_object"},
        "hash_policy": HASH_POLICY_LF,
        "requires_max_model_len": REQUIRES_MAX_MODEL_LEN,
        "ocr": {
            "model": first_ocr["ocr_model"],
            "revision": first_ocr["ocr_revision"],
            "prompt_version": first_ocr["ocr_prompt_version"],
            "prompt_sha256": first_ocr["ocr_prompt_sha256"],
            "ocr_jsonl": repo_relative(ocr_path, REPO_ROOT),
            "ocr_jsonl_sha256": artifact_sha256("ocr_jsonl", ocr_path, HASH_POLICY_LF),
        },
        "label_totals_audit_only": dict(sorted(label_totals.items())),
        "no_label_leakage_policy": (
            "samples.jsonl and node1_chat_requests.jsonl exclude labels/key_text_pairs; "
            "annotation labels are kept only in provenance.jsonl for audit."
        ),
        "freeze_decision": "W1_v2 (M5-01): the 30 M0-04 samples with full-receipt OCR text; W1 v1 stays frozen",
        "coverage_note": (
            "MC-OCR labels cover seller/address/timestamp/total only; items, buyer and VAT fields are "
            "extracted from OCR text but have no labels and are checked arithmetically only."
        ),
    }
    package_sha256 = compute_package_sha256(file_hashes, metadata)
    manifest = {
        **metadata,
        "package_sha256": package_sha256,
        "artifacts": {name: repo_relative(path, REPO_ROOT) for name, path in paths.items()},
        "file_hashes": file_hashes,
        "source_artifacts": {
            "source_samples": repo_relative(resolve(args.source_samples), REPO_ROOT),
            "source_manifest": repo_relative(resolve(args.source_manifest), REPO_ROOT),
            "source_system_prompt": repo_relative(resolve(args.v1_prompt), REPO_ROOT),
            "source_full_receipt_rules": repo_relative(resolve(args.rules), REPO_ROOT),
            "ocr_jsonl": metadata["ocr"]["ocr_jsonl"],
        },
    }
    (output_dir / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "README.md").write_text(
        readme_text(package_sha256, file_hashes["system_prompt"], len(sample_rows), metadata["ocr"]["ocr_jsonl"]),
        encoding="utf-8", newline="\n",
    )
    summary = {
        "task": "M5-01",
        "step": "build-w1-v2-workload",
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": W1_V2_DATASET_VERSION,
        "prompt_version": W1_V2_PROMPT_VERSION,
        "sample_count": len(sample_rows),
        "max_tokens": MAX_TOKENS,
        "package_sha256": package_sha256,
        "prompt_sha256": file_hashes["system_prompt"],
        "raw_text_sha256": {row["sample_id"]: sha256_text(row["raw_text"]) for row in sample_rows},
        "artifacts": manifest["artifacts"],
    }
    (report_dir / "m5_01_w1_v2_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: summary[k] for k in ("dataset_version", "sample_count", "package_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
