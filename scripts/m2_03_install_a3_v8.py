#!/usr/bin/env python3
"""Switch the already-installed V1 v7 hook to v8 module without removing backup.

Touches ONLY pinned vLLM site-packages GPUModelRunner imports. Refuses drifted
sources. The v7 script/probe and canonical evidence remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

EXPECTED_VLLM = "0.30.0"
EXPECTED_V7_SHA256 = "06e685b75346bb63542ff29fe7883317253f0980684f9946e7ca89a53b7cafd8"
BACKUP_SUFFIX = ".m203_before_v1_a3_v8"
OLD_IMPORT = "from experiments.m2_03_v1_target_logits_probe import "
NEW_IMPORT = "from experiments.m2_03_v1_target_logits_probe_v8 import "
ENTRYPOINTS = ("prepare_v1_capture", "finalize_v1_capture")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runner_path() -> Path:
    import vllm
    if vllm.__version__ != EXPECTED_VLLM:
        raise RuntimeError(f"Expected pinned vLLM {EXPECTED_VLLM}, got {vllm.__version__}")
    return Path(vllm.__file__).resolve().parent / "v1/worker/gpu_model_runner.py"


def transform(v7: str) -> str:
    if digest(v7.encode("utf-8")) != EXPECTED_V7_SHA256:
        raise RuntimeError("Original installed source is not the checked V1 v7 SHA")
    for name in ENTRYPOINTS:
        marker = OLD_IMPORT + name
        if v7.count(marker) != 1:
            raise RuntimeError(f"Missing/duplicated v7 hook import: {marker}")
    candidate = v7.replace(OLD_IMPORT, NEW_IMPORT)
    for name in ENTRYPOINTS:
        if candidate.count(NEW_IMPORT + name) != 1:
            raise RuntimeError(f"New v8 hook import not unique: {name}")
    compile(candidate, "gpu_model_runner.py", "exec")
    return candidate


def _status(path: Path) -> str:
    source = path.read_bytes()
    if digest(source) == EXPECTED_V7_SHA256:
        status = "V7_INSTALLED"
    else:
        original = source.decode("utf-8").replace(NEW_IMPORT, OLD_IMPORT)
        status = "V8_INSTALLED" if (
            original != source.decode("utf-8")
            and digest(original.encode("utf-8")) == EXPECTED_V7_SHA256
            and source.decode("utf-8") == transform(original)
        ) else "UNEXPECTED_SOURCE"
    print("V1_MODEL_RUNNER:", path)
    print("V1_SOURCE_SHA256:", digest(source))
    print("A3_V8_HOOK_STATUS:", status)
    return status


def _write_atomically(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="m203_a3_v8_", delete=False) as f:
        temp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        temp.chmod(path.stat().st_mode)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def install(path: Path, repo: Path) -> None:
    v8 = repo / "experiments/m2_03_v1_target_logits_probe_v8.py"
    if not v8.is_file():
        raise RuntimeError(f"Missing v8 probe source: {v8}")
    text = v8.read_text(encoding="utf-8")
    for name in ENTRYPOINTS:
        if f"def {name}(" not in text:
            raise RuntimeError(f"v8 module missing required method {name}")
    compile(text, str(v8), "exec")
    status = _status(path)
    if status == "V8_INSTALLED":
        return
    if status != "V7_INSTALLED":
        raise RuntimeError("Unknown site-packages source. Refusing patch")
    v7 = path.read_bytes()
    patched = transform(v7.decode("utf-8")).encode("utf-8")
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        if backup.read_bytes() != v7:
            raise RuntimeError("Existing v8 backup is not exact v7 runner")
    else:
        with backup.open("xb") as f:
            f.write(v7)
    _write_atomically(path, patched)
    if path.read_bytes() != patched:
        shutil.copy2(backup, path)
        raise RuntimeError("Installed runner bytes not as expected; restored v7")
    print("A3_V8_SWITCH: PASS")
    print("V8_BACKUP:", backup)
    assert _status(path) == "V8_INSTALLED"


def restore(path: Path) -> None:
    if _status(path) != "V8_INSTALLED":
        raise RuntimeError("Installed runner not exact v8; refusing restore")
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.is_file() or digest(backup.read_bytes()) != EXPECTED_V7_SHA256:
        raise RuntimeError("Missing/invalid v7 backup")
    _write_atomically(path, backup.read_bytes())
    assert _status(path) == "V7_INSTALLED"
    print("A3_V8_RESTORE_V7: PASS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("status", "install", "restore"))
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    path = _runner_path()
    if args.action == "install":
        install(path, args.repo_root.resolve())
    elif args.action == "restore":
        restore(path)
    else:
        _status(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"A3_V8_INSTALL_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
