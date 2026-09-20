"""Validate Node 1 invoice extraction JSON files against the canonical schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from init_invoice_sqlite import semantic_validation_errors


DEFAULT_SCHEMA_PATH = Path("src/pipeline/schemas/invoice_extraction.schema.json")


def format_path(path_parts: object) -> str:
    parts = list(path_parts)
    if not parts:
        return "$"
    rendered = "$"
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_json(args.schema)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    failed = False
    for json_file in args.json_files:
        instance = load_json(json_file)
        errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
        semantic_errors = semantic_validation_errors(instance) if isinstance(instance, dict) else []
        if not errors and not semantic_errors:
            print(f"OK {json_file}")
            continue

        failed = True
        print(f"FAIL {json_file}")
        for error in errors:
            print(f"  {format_path(error.path)}: {error.message}")
        for error in semantic_errors:
            print(f"  {error}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
