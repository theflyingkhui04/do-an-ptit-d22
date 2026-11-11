"""Read exact installed/captured vLLM bytes against committed hash metadata.

Never imports vLLM/torch, downloads a package, copies upstream sources into Git,
or accepts a version/source mismatch. Local historical captures remain usable;
the Colab Git checkout uses the exact installed distribution instead.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/scalar_vllm_source_manifest.json"
CAPTURE = ROOT / "experiments/runs/m4_02_04/20261110-async-lineage-design/installed"


def manifest():
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if value["vllm"] != "0.30.0":
        raise ValueError("CANONICAL ENVIRONMENT MISMATCH: pinned source manifest version")
    return value


def read_source(name, *, installed_only=False):
    entries = {row["path"]: row for row in manifest()["files"]}
    if name not in entries:
        raise ValueError("Unreviewed pinned source: " + name)
    path = CAPTURE / name
    if installed_only or not path.is_file():
        try:
            distribution = importlib.metadata.distribution("vllm")
        except importlib.metadata.PackageNotFoundError as error:
            raise FileNotFoundError("Pinned source unavailable; install vllm==0.30.0: " + name) from error
        if distribution.version != "0.30.0":
            raise ValueError("CANONICAL ENVIRONMENT MISMATCH: vllm version")
        path = Path(distribution.locate_file(name))
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entries[name]["sha256"]:
        raise ValueError("CANONICAL ENVIRONMENT MISMATCH: pinned source hash " + name)
    return content
