"""Prepare a small clean MC-OCR dev set for task M0-04.

The script expects the Kaggle dataset to be downloaded under
data/raw/mc_ocr_2021. It finds mcocr_train_df.csv and train_images
recursively, selects receipt rows with good image quality and useful key
labels, copies selected images to data/processed (gitignored), and writes
a compact manifest/summary to experiments/results (tracked).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_ROOT = Path("data/raw/mc_ocr_2021")
DEFAULT_PROCESSED_ROOT = Path("data/processed/m0_04_mcocr_devset")
DEFAULT_REPORT_DIR = Path("experiments/results")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_label(label: str) -> str:
    label = label.strip().upper()
    aliases = {
        "TIMESTAMP": "TIMESTAMPS",
        "TIME_STAMP": "TIMESTAMPS",
        "TIME_STAMPS": "TIMESTAMPS",
        "TOTAL": "TOTAL_COST",
        "TOTAL_PRICE": "TOTAL_COST",
        "TOTAL_AMOUNT": "TOTAL_COST",
    }
    return aliases.get(label, label)


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake in the Kaggle CSV."""
    mojibake_markers = ("Ã", "Â", "Ä", "Æ", "Ð", "áº", "á»")
    has_c1_control = any("\u0080" <= char <= "\u009f" for char in text)
    if not has_c1_control and not any(marker in text for marker in mojibake_markers):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    return repaired


def split_mcocr_field(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [repair_mojibake(part.strip()) for part in text.split("|||") if part.strip()]


def find_one(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename!r} under {root}")
    return matches[0]


def find_image_index(root: Path) -> tuple[dict[str, Path], list[Path]]:
    train_image_dirs = sorted(
        path for path in root.rglob("*") if path.is_dir() and path.name.lower() == "train_images"
    )
    search_roots = train_image_dirs or [root]

    image_index: dict[str, Path] = {}
    for image_root in search_roots:
        for image_path in image_root.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                image_index.setdefault(image_path.name, image_path)
    return image_index, search_roots


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_key_text_pairs(labels: list[str], texts: list[str]) -> list[dict[str, str]]:
    pairs = []
    for label, text in zip(labels, texts):
        pairs.append({"label": normalize_label(label), "text": text})
    return pairs


def build_text_preview(texts: list[str], max_chars: int = 300) -> str:
    preview = " | ".join(line.replace("\n", " ").strip() for line in texts if line.strip())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."


def select_devset(
    df: pd.DataFrame,
    image_index: dict[str, Path],
    sample_size: int,
    quality_threshold: float,
    required_labels: Iterable[str],
    preferred_labels: Iterable[str],
) -> pd.DataFrame:
    required = {normalize_label(label) for label in required_labels}
    preferred = {normalize_label(label) for label in preferred_labels}

    records = []
    for row_idx, row in df.iterrows():
        img_id = str(row.get("img_id", "")).strip()
        labels = [normalize_label(label) for label in split_mcocr_field(row.get("anno_labels"))]
        texts = split_mcocr_field(row.get("anno_texts"))
        label_set = set(labels)
        quality = safe_float(row.get("anno_image_quality"))
        image_path = image_index.get(img_id)

        has_required = required.issubset(label_set)
        if image_path is None or quality < quality_threshold or not has_required:
            continue

        preferred_count = len(label_set & preferred)
        records.append(
            {
                "row_idx": row_idx,
                "img_id": img_id,
                "image_path": image_path,
                "quality": quality,
                "anno_num": safe_int(row.get("anno_num")),
                "text_line_count": len(texts),
                "labels": labels,
                "label_set": label_set,
                "texts": texts,
                "preferred_label_count": preferred_count,
                "score": quality + preferred_count * 0.05 + min(len(texts), 80) * 0.0005,
            }
        )

    selected = pd.DataFrame(records)
    if selected.empty:
        return selected
    return selected.sort_values(
        by=["score", "quality", "preferred_label_count", "text_line_count", "img_id"],
        ascending=[False, False, False, False, True],
    ).head(sample_size)


def write_outputs(
    selected: pd.DataFrame,
    source_csv: Path,
    processed_root: Path,
    report_dir: Path,
    repo_root: Path,
    copy_images: bool,
    quality_threshold: float,
    required_labels: list[str],
    preferred_labels: list[str],
) -> dict[str, object]:
    processed_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = processed_root / "images"
    if copy_images:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    jsonl_rows = []
    label_counter: Counter[str] = Counter()

    for rank, record in enumerate(selected.to_dict("records"), start=1):
        sample_id = f"m0_04_{rank:03d}"
        source_image = Path(record["image_path"])
        dev_image = image_output_dir / source_image.name if copy_images else source_image
        if copy_images:
            shutil.copy2(source_image, dev_image)

        labels = list(record["labels"])
        texts = list(record["texts"])
        label_set = set(record["label_set"])
        label_counter.update(label_set)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source_row_idx": int(record["row_idx"]),
                "img_id": record["img_id"],
                "anno_image_quality": round(float(record["quality"]), 4),
                "anno_num": int(record["anno_num"]),
                "text_line_count": int(record["text_line_count"]),
                "has_seller": "SELLER" in label_set,
                "has_address": "ADDRESS" in label_set,
                "has_timestamps": "TIMESTAMPS" in label_set,
                "has_total_cost": "TOTAL_COST" in label_set,
                "labels": "|".join(sorted(label_set)),
                "text_preview": build_text_preview(texts),
                "source_csv": repo_relative(source_csv, repo_root),
                "source_image_path": repo_relative(source_image, repo_root),
                "dev_image_path": repo_relative(dev_image, repo_root),
                "reason_selected": (
                    f"quality>={quality_threshold}; "
                    f"required={','.join(required_labels)}; image_exists=True"
                ),
            }
        )

        jsonl_rows.append(
            {
                "sample_id": sample_id,
                "source_row_idx": int(record["row_idx"]),
                "img_id": record["img_id"],
                "image_path": repo_relative(dev_image, repo_root),
                "source_image_path": repo_relative(source_image, repo_root),
                "source_csv": repo_relative(source_csv, repo_root),
                "anno_image_quality": round(float(record["quality"]), 4),
                "anno_num": int(record["anno_num"]),
                "labels": labels,
                "raw_text": "\n".join(texts),
                "key_text_pairs": build_key_text_pairs(labels, texts),
            }
        )

    processed_manifest = processed_root / "manifest.csv"
    tracked_manifest = report_dir / "m0_04_mcocr_devset_manifest.csv"
    jsonl_path = processed_root / "samples.jsonl"
    summary_path = report_dir / "m0_04_mcocr_devset_summary.json"

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(processed_manifest, index=False, encoding="utf-8")
    manifest_df.to_csv(tracked_manifest, index=False, encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "task": "M0-04",
        "source_csv": repo_relative(source_csv, repo_root),
        "processed_root": repo_relative(processed_root, repo_root),
        "sample_count": len(manifest_rows),
        "quality_threshold": quality_threshold,
        "required_labels": required_labels,
        "preferred_labels": preferred_labels,
        "copied_images": copy_images,
        "tracked_manifest": repo_relative(tracked_manifest, repo_root),
        "processed_manifest": repo_relative(processed_manifest, repo_root),
        "processed_jsonl": repo_relative(jsonl_path, repo_root),
        "label_counts_in_selected_samples": dict(sorted(label_counter.items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument(
        "--required-labels",
        nargs="+",
        default=["SELLER", "ADDRESS", "TIMESTAMPS"],
        help="Labels that must be present in every selected receipt.",
    )
    parser.add_argument(
        "--preferred-labels",
        nargs="+",
        default=["TOTAL_COST"],
        help="Labels that improve ranking but are not mandatory.",
    )
    parser.add_argument("--no-copy-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(
            f"{data_root} does not exist. Download the Kaggle dataset before running this script."
        )

    source_csv = find_one(data_root, "mcocr_train_df.csv")
    image_index, image_roots = find_image_index(data_root)
    df = pd.read_csv(source_csv)
    selected = select_devset(
        df=df,
        image_index=image_index,
        sample_size=args.sample_size,
        quality_threshold=args.quality_threshold,
        required_labels=args.required_labels,
        preferred_labels=args.preferred_labels,
    )

    if selected.empty:
        raise RuntimeError(
            "No receipts matched the current filters. Try lowering --quality-threshold "
            "or reducing --required-labels."
        )

    summary = write_outputs(
        selected=selected,
        source_csv=source_csv,
        processed_root=args.processed_root,
        report_dir=args.report_dir,
        repo_root=repo_root,
        copy_images=not args.no_copy_images,
        quality_threshold=args.quality_threshold,
        required_labels=[normalize_label(label) for label in args.required_labels],
        preferred_labels=[normalize_label(label) for label in args.preferred_labels],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Image roots searched:")
    for image_root in image_roots:
        print(f"- {repo_relative(image_root, repo_root)}")


if __name__ == "__main__":
    main()
