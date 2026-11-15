"""[M5-01 web demo] Seed the demo Postgres with the W1_v3 adaptive devset extractions.

Canonical flow: writes the 25 valid Node 1 extractions of lane adaptive on W1_v3
(library OCR text without Vietnamese marks;
experiments/runs/m5_01/m5_01_v3_adaptive/validated_outputs.jsonl) as ``source = 'devset'``
and uploads each invoice's devset image to the private bucket. Re-running changes nothing
(existing invoices are skipped). The W1_v2 seed (29 extractions, commit b668709) stays
available with ``--validated`` for the v1 question set whose reference rows were frozen on it.

    python scripts/seed_demo_db.py --apply-schema          # DEMO_DATABASE_URL = owner connection
    python scripts/seed_demo_db.py --no-images             # DB only

Environment: DEMO_DATABASE_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY, DEMO_IMAGE_BUCKET
(default invoice-images). Credentials are read from the environment only.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.postgres import writer

DEFAULT_VALIDATED = Path("experiments/runs/m5_01/m5_01_v3_adaptive/validated_outputs.jsonl")
SEED_NOTES = "seed: experiments/runs/m5_01/m5_01_v3_adaptive (W1_v3, library OCR, marks stripped)"
W1_V2_VALIDATED = Path("experiments/runs/m5_01/m5_01_adaptive/validated_outputs.jsonl")
W1_V2_SEED_NOTES = "seed: experiments/runs/m5_01/m5_01_adaptive (W1_v2, commit b668709)"


class ImageStore(Protocol):
    def upload(self, path: str, data: bytes, content_type: str) -> None: ...


def storage_path_for(image_path: str) -> str:
    return f"devset/{Path(image_path).name}"


def load_validated(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(r.get("payload") is None for r in rows):
        raise SystemExit(f"{path}: every row needs a validated payload")
    return rows


def seed_notes(validated: Path) -> str:
    if validated == DEFAULT_VALIDATED:
        return SEED_NOTES
    if validated == W1_V2_VALIDATED:
        return W1_V2_SEED_NOTES
    return f"seed: {validated.parent.as_posix()}"


def seed(conn: Any, rows: list[dict[str, Any]], *, storage: ImageStore | None = None,
         images_root: Path = REPO_ROOT, notes: str = SEED_NOTES) -> dict[str, int]:
    written = skipped = uploaded = 0
    for row in rows:
        payload = row["payload"]
        image_path = payload["source"]["image_path"]
        run = writer.RunInfo(run_id=row["run_id"], lane=row["lane"], model_name=row["model"],
                             prompt_version="node1_extractor.v2", notes=notes)
        if storage is not None:
            image = images_root / image_path
            storage.upload(storage_path_for(image_path), image.read_bytes(),
                           mimetypes.guess_type(image.name)[0] or "image/jpeg")
            uploaded += 1
        _, was_written = writer.insert_extraction(
            conn, payload, run, source="devset", storage_path=storage_path_for(image_path),
            latency_ms=row.get("latency_ms"), if_exists="skip")
        written += was_written
        skipped += not was_written
    return {"rows": len(rows), "written": written, "skipped": skipped, "images_uploaded": uploaded}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--apply-schema", action="store_true", help="run schema.sql and roles.sql first (owner)")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)

    import psycopg

    url = os.environ.get("DEMO_DATABASE_URL")
    if not url:
        raise SystemExit("Set DEMO_DATABASE_URL (owner or demo_writer connection)")
    storage = None
    if not args.no_images:
        from src.api.storage import SupabaseStorage

        storage = SupabaseStorage(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"],
                                  os.environ.get("DEMO_IMAGE_BUCKET", "invoice-images"))
        storage.ensure_bucket()
    rows = load_validated(REPO_ROOT / args.validated)
    with psycopg.connect(url, autocommit=True) as conn:
        if args.apply_schema:
            writer.apply_schema(conn, roles=True)
        conn.execute("SET search_path = demo")
        summary = seed(conn, rows, storage=storage, notes=seed_notes(args.validated))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
