"""Verify the curated evidence bytes without modifying evidence or launching vLLM."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Evidence path escapes root: {entry['path']}")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Evidence integrity failure: {entry['path']}")
    print(f"PASS: {len(manifest['files'])} curated evidence files match SHA-256 and size")


if __name__ == "__main__":
    main()
