#!/usr/bin/env python3
"""Install/verify/restore the M2-03 A3 diagnostic hook in vLLM 0.30.0.

This script intentionally modifies ONLY the installed vLLM copy inside the
active Python environment. It does not modify canonical M2-03 evidence.

The v6 change removes the *outer* M203_A3_OUTPUT_DIR gate from model_runner.py.
GPUModelRunner.sample() always calls capture_target_logits(); the probe itself
then decides whether diagnostics are enabled via environment variables or the
process-independent /tmp/m203_a3_probe_config.json fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

EXPECTED_VLLM_VERSION = "0.30.0"

GUARDED_HOOK = """        # M2-03 A3: diagnostic-only target-logits observer.\n        import os\n        if os.environ.get('M203_A3_OUTPUT_DIR'):\n            from experiments.m2_03_target_logits_probe import capture_target_logits\n            capture_target_logits(self, input_batch, logits)\n"""

UNCONDITIONAL_HOOK = """        # M2-03 A3: diagnostic-only target-logits observer.\n        from experiments.m2_03_target_logits_probe import capture_target_logits\n        capture_target_logits(self, input_batch, logits)\n"""

COMPUTE_MARKER = "            logits = self.model.compute_logits(sample_hidden_states)\n"
GRAMMAR_MARKER = "        if grammar_output is not None:\n"
BACKUP_SUFFIX = ".m203_before_unconditional_v6"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_text(text: str) -> tuple[str, str]:
    """Return (candidate, state). Fail closed on unknown/ambiguous layouts."""
    guarded_count = text.count(GUARDED_HOOK)
    unconditional_count = text.count(UNCONDITIONAL_HOOK)

    if unconditional_count == 1 and guarded_count == 0:
        candidate = text
        state = "ALREADY_INSTALLED"
    elif guarded_count == 1 and unconditional_count == 0:
        candidate = text.replace(GUARDED_HOOK, UNCONDITIONAL_HOOK, 1)
        state = "PATCH_REQUIRED"
    else:
        raise RuntimeError(
            "Unexpected A3 hook layout: "
            f"guarded_count={guarded_count}, unconditional_count={unconditional_count}. "
            "Inspect model_runner.py manually; refusing to patch."
        )

    # Prove the hook remains after raw target-logit creation and before grammar.
    compute_pos = candidate.find(COMPUTE_MARKER)
    hook_pos = candidate.find(UNCONDITIONAL_HOOK)
    grammar_pos = candidate.find(GRAMMAR_MARKER, hook_pos)
    if min(compute_pos, hook_pos, grammar_pos) < 0:
        raise RuntimeError("Cannot prove compute_logits -> A3 hook -> grammar ordering")
    if not (compute_pos < hook_pos < grammar_pos):
        raise RuntimeError("Unsafe A3 hook ordering; refusing to patch")

    compile(candidate, "model_runner.py", "exec")
    return candidate, state


def locate_model_runner() -> tuple[Path, str]:
    try:
        import vllm  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Cannot import vLLM from active Python: {exc}") from exc

    version = getattr(vllm, "__version__", "UNKNOWN")
    if version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"Expected vLLM {EXPECTED_VLLM_VERSION}, got {version}; refusing to patch"
        )
    path = Path(vllm.__file__).resolve().parent / "v1/worker/gpu/model_runner.py"
    if not path.is_file():
        raise RuntimeError(f"Installed model_runner.py not found: {path}")
    return path, version


def check_probe(repo_root: Path) -> Path:
    probe = repo_root / "experiments/m2_03_target_logits_probe.py"
    if not probe.is_file():
        raise RuntimeError(f"Probe not found: {probe}")
    text = probe.read_text(encoding="utf-8")
    required = (
        'CONFIG_PATH = Path("/tmp/m203_a3_probe_config.json")',
        "[M2-03 A3 CONFIG]",
        "[M2-03 A3 MAP]",
        "def capture_target_logits(",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"Probe is not v5+ config-fallback build; missing: {missing}")
    compile(text, str(probe), "exec")
    return probe


def status(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    print("VLLM_MODEL_RUNNER:", path)
    print("SHA256:", sha256_bytes(path.read_bytes()))
    print("GUARDED_HOOK_COUNT:", text.count(GUARDED_HOOK))
    print("UNCONDITIONAL_HOOK_COUNT:", text.count(UNCONDITIONAL_HOOK))
    try:
        _, state = patch_text(text)
        print("HOOK_STATUS:", state)
    except Exception as exc:
        print("HOOK_STATUS: INVALID")
        print("DETAIL:", exc)
        raise


def install(path: Path, repo_root: Path) -> None:
    probe = check_probe(repo_root)
    original_bytes = path.read_bytes()
    original_text = original_bytes.decode("utf-8")
    candidate, state = patch_text(original_text)

    print("VLLM_MODEL_RUNNER:", path)
    print("PROBE:", probe)
    print("BEFORE_SHA256:", sha256_bytes(original_bytes))
    print("HOOK_STATE_BEFORE:", state)

    if state == "ALREADY_INSTALLED":
        print("PATCH_APPLIED: NO (already installed)")
        print("HOOK_INSTALLATION: PASS")
        return

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        if backup.read_bytes() != original_bytes:
            raise RuntimeError(
                f"Backup already exists and differs from current source: {backup}. "
                "Refusing to overwrite it."
            )
    else:
        # Exclusive creation prevents silently overwriting provenance.
        with backup.open("xb") as handle:
            handle.write(original_bytes)

    candidate_bytes = candidate.encode("utf-8")
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix="model_runner.py.m203_v6_", delete=False
    ) as handle:
        tmp = Path(handle.name)
        handle.write(candidate_bytes)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

    installed = path.read_text(encoding="utf-8")
    if installed.count(UNCONDITIONAL_HOOK) != 1 or installed.count(GUARDED_HOOK) != 0:
        shutil.copy2(backup, path)
        raise RuntimeError("Post-write verification failed; restored backup")
    patch_text(installed)

    print("BACKUP:", backup)
    print("BACKUP_SHA256:", sha256_bytes(backup.read_bytes()))
    print("AFTER_SHA256:", sha256_bytes(path.read_bytes()))
    print("PATCH_APPLIED: YES")
    print("HOOK_INSTALLATION: PASS")


def restore(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.is_file():
        raise RuntimeError(f"Backup not found: {backup}")
    current_sha = sha256_bytes(path.read_bytes())
    backup_sha = sha256_bytes(backup.read_bytes())
    shutil.copy2(backup, path)
    print("RESTORE_FROM:", backup)
    print("CURRENT_BEFORE_SHA256:", current_sha)
    print("RESTORED_SHA256:", backup_sha)
    print("RESTORE: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "status", "restore"), nargs="?", default="install")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing experiments/m2_03_target_logits_probe.py",
    )
    args = parser.parse_args()

    path, version = locate_model_runner()
    print("VLLM_VERSION:", version)
    if args.action == "status":
        status(path)
    elif args.action == "install":
        install(path, args.repo_root.resolve())
    else:
        restore(path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"A3_HOOK_INSTALLER_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
