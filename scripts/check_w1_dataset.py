"""Check frozen Workload W1 artifacts before handing them to the benchmark harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.safe_paths import contained_path  # noqa: E402
from src.pipeline.w1_workload import (  # noqa: E402
    W1_DATASET_VERSION,
    W1_DATASET_VERSIONS,
    W1_V2_DATASET_VERSION,
    W1_V3_DATASET_VERSION,
    W1_WORKLOAD_ID,
    artifact_sha256,
    compute_package_sha256,
    load_jsonl,
    request_semantic_view,
)


DEFAULT_W1_DIR = Path("experiments/configs/w1")
DEFAULT_REPORT = Path("experiments/results/m1_03_w1_check_summary.json")
DEFAULT_REPORTS = {
    W1_DATASET_VERSION: DEFAULT_REPORT,
    W1_V2_DATASET_VERSION: Path("experiments/results/m5_01_w1_v2_check_summary.json"),
    W1_V3_DATASET_VERSION: Path("experiments/results/m5_01_w1_v3_check_summary.json"),
}


FORBIDDEN_MODEL_FACING_KEYS = {"labels", "key_text_pairs"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def assert_no_label_leakage(samples: list[dict[str, Any]], requests: list[dict[str, Any]]) -> None:
    for sample in samples:
        leaked = sorted(FORBIDDEN_MODEL_FACING_KEYS.intersection(sample))
        if leaked:
            raise AssertionError(f"{sample.get('sample_id')} leaks audit labels in samples.jsonl: {leaked}")

    forbidden_fragments = ("key_text_pairs", '"labels"')
    for row in requests:
        request = row.get("request")
        messages = request.get("messages") if isinstance(request, dict) else None
        if not isinstance(messages, list):
            raise AssertionError(f"{row.get('sample_id')} request has no messages")
        rendered = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
        for fragment in forbidden_fragments:
            if fragment in rendered:
                raise AssertionError(f"{row.get('sample_id')} request leaks forbidden fragment: {fragment}")
        user_content = str(messages[1].get("content", "")) if len(messages) > 1 and isinstance(messages[1], dict) else ""
        if "source_label" in user_content:
            raise AssertionError(f"{row.get('sample_id')} user content leaks source_label")


def assert_requests_match_samples(
    samples: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    max_tokens: object = None,
) -> None:
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    request_ids = [row.get("sample_id") for row in requests]
    if request_ids != [sample["sample_id"] for sample in samples]:
        raise AssertionError("request order/sample IDs must match samples.jsonl")
    for row in requests:
        sample = samples_by_id[row["sample_id"]]
        expected_source = row.get("expected_source")
        expected = {
            "sample_id": sample["sample_id"],
            "img_id": sample["img_id"],
            "image_path": sample["image_path"],
            "raw_text": sample["raw_text"],
        }
        if expected_source != expected:
            raise AssertionError(f"{sample['sample_id']} expected_source does not match sample")
        semantic_view = request_semantic_view(row)
        if semantic_view["temperature"] != 0.0:
            raise AssertionError(f"{sample['sample_id']} temperature must be 0.0")
        if semantic_view["seed"] != 42:
            raise AssertionError(f"{sample['sample_id']} seed must be 42")
        if semantic_view["response_format"] != {"type": "json_object"}:
            raise AssertionError(f"{sample['sample_id']} response_format must be json_object")
        if max_tokens is not None and semantic_view["max_tokens"] != max_tokens:
            raise AssertionError(
                f"{sample['sample_id']} max_tokens {semantic_view['max_tokens']} != manifest {max_tokens}"
            )


def assert_marks_stripped(samples: list[dict[str, Any]], package_manifest: dict[str, Any]) -> None:
    """[M5-01 canonical flow] A package that declares mark stripping must not contain Vietnamese marks."""
    if not package_manifest.get("mark_stripping"):
        return
    from src.pipeline.text_normalize import has_vietnamese_marks

    marked = [s.get("sample_id") for s in samples if has_vietnamese_marks(str(s.get("raw_text", "")))]
    if marked:
        raise AssertionError(f"Vietnamese marks left in a mark-stripped package: {marked}")


def assert_manifest_hashes(w1_dir: Path, package_manifest: dict[str, Any]) -> dict[str, str]:
    artifacts = package_manifest.get("artifacts")
    expected_hashes = package_manifest.get("file_hashes")
    if not isinstance(artifacts, dict) or not isinstance(expected_hashes, dict):
        raise AssertionError("package_manifest.json must contain artifacts and file_hashes")
    hash_policy = package_manifest.get("hash_policy")
    actual_hashes = {}
    for name, rel_path in artifacts.items():
        path = REPO_ROOT / rel_path
        if not path.exists():
            raise AssertionError(f"missing artifact {name}: {rel_path}")
        actual_hashes[name] = artifact_sha256(name, path, hash_policy)
    for name, digest in actual_hashes.items():
        if expected_hashes.get(name) != digest:
            raise AssertionError(f"hash mismatch for {name}")

    metadata = {
        key: value
        for key, value in package_manifest.items()
        if key not in {"artifacts", "file_hashes", "source_artifacts", "package_sha256"}
    }
    recomputed = compute_package_sha256(actual_hashes, metadata)
    if package_manifest.get("package_sha256") != recomputed:
        raise AssertionError("package_sha256 mismatch")
    return actual_hashes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w1-dir", type=Path, default=DEFAULT_W1_DIR)
    parser.add_argument("--report", type=Path, help="Default: per dataset version (W1 v1 or W1_v2).")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:  # both paths come from the command line: repository or temp directory only
        w1_dir = contained_path(args.w1_dir)
        report_arg = None if args.report is None else contained_path(args.report)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    samples = load_jsonl(w1_dir / "samples.jsonl")
    provenance = load_jsonl(w1_dir / "provenance.jsonl")
    requests = load_jsonl(w1_dir / "node1_chat_requests.jsonl")
    package_manifest = load_json(w1_dir / "package_manifest.json")

    if len(samples) < args.min_samples:
        raise AssertionError(f"expected at least {args.min_samples} samples, found {len(samples)}")
    sample_ids = [sample.get("sample_id") for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError("sample_id values must be unique")
    if len(provenance) != len(samples):
        raise AssertionError("provenance row count must match samples")
    if package_manifest.get("workload_id") != W1_WORKLOAD_ID:
        raise AssertionError("unexpected workload_id")
    dataset_version = package_manifest.get("dataset_version")
    if dataset_version not in W1_DATASET_VERSIONS:
        raise AssertionError("unexpected dataset_version")
    for sample in samples:
        if sample.get("dataset_version") != dataset_version:
            raise AssertionError(f"{sample.get('sample_id')} dataset_version does not match the manifest")

    assert_no_label_leakage(samples, requests)
    assert_marks_stripped(samples, package_manifest)
    assert_requests_match_samples(samples, requests, package_manifest.get("max_tokens"))
    actual_hashes = assert_manifest_hashes(w1_dir, package_manifest)

    summary = {
        "task": "M1-03" if dataset_version == W1_DATASET_VERSION else "M5-01",
        "step": "check-w1-workload",
        "status": "PASS",
        "workload_id": W1_WORKLOAD_ID,
        "dataset_version": dataset_version,
        "w1_dir": args.w1_dir.as_posix(),
        "sample_count": len(samples),
        "request_count": len(requests),
        "package_sha256": package_manifest["package_sha256"],
        "file_hashes": actual_hashes,
        "checks": {
            "min_samples": True,
            "unique_sample_ids": True,
            "provenance_complete": True,
            "no_label_leakage": True,
            "request_source_binding": True,
            "request_max_tokens": True,
            "package_hash": True,
        },
    }
    if not args.no_write_report:
        # The default is one of the fixed paths, chosen by comparison: nothing read from the
        # package becomes part of the path (dataset_version was checked against the known set above).
        if dataset_version == W1_DATASET_VERSION:
            default_report = DEFAULT_REPORTS[W1_DATASET_VERSION]
        elif dataset_version == W1_V2_DATASET_VERSION:
            default_report = DEFAULT_REPORTS[W1_V2_DATASET_VERSION]
        else:
            default_report = DEFAULT_REPORTS[W1_V3_DATASET_VERSION]
        report = report_arg or REPO_ROOT / default_report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
