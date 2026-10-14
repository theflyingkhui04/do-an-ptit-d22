"""Standard-library storage, Git and one-request guards."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = "bdb26dd89259dcc6b4085041c783bc9536e1287c"
HISTORICAL_EXECUTION = "4b830a8b7530b9bb3218fb9628477ffd164705ba"
CASE = "w3_business_v2_003"
TARGET = 60
WINDOW = range(56, 65)
LANES = ("vanilla", "adaptive")
REQUEST_FILE = "experiments/configs/w3_v2/chat_requests.jsonl"
REQUEST_HASH = "0c1ed2e2e80a61792e7f5f01f03ddc4c9ba0bf6ff29ed1a0a2b98c4a5b294ebb"
WORKLOAD_HASH = "57341b0fc8c2ebbdfcfc579a6343aea66b08ca11fe2bfed5bc53a2bd49f5d83b"


class Invalid(RuntimeError):
    pass


def need(condition, reason):
    if not condition:
        raise Invalid(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("ascii")


def fingerprint(value):
    # Same canonical serialization as frozen request/verifier fingerprint.
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        need(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def decode(raw):
    def constant(_):
        raise Invalid("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=constant)


def load(path):
    return decode(Path(path).read_bytes())


def reviewed_policy(value):
    """Templates are discovery aids, never executable runtime assumptions."""
    if isinstance(value, str):
        need("REVIEW_REQUIRED" not in value, "runtime policy template is not executable")
    elif isinstance(value, dict):
        for item in value.values():
            reviewed_policy(item)
    elif isinstance(value, list):
        for item in value:
            reviewed_policy(item)


def git(*args, repo=ROOT):
    try:
        return subprocess.run(["git", "-c", f"safe.directory={Path(repo).as_posix()}",
                               "-C", str(repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Invalid("Git/provenance unavailable; no working-tree fallback") from exc


def blob(path, commit=HISTORICAL):
    return git("cat-file", "blob", f"{commit}:{path}")


def frozen_request():
    raw = blob(REQUEST_FILE)
    need(sha(raw) == WORKLOAD_HASH, "frozen workload checksum mismatch")
    rows = [decode(line) for line in raw.splitlines()]
    matches = [r for r in rows if r["sample_id"] == CASE]
    need(len(matches) == 1, "frozen sample missing/ambiguous")
    request = matches[0]["request"]
    need(fingerprint(request) == REQUEST_HASH, "frozen request checksum mismatch")
    need(request["seed"] == 42 and request["temperature"] == 0
         and request["max_tokens"] == 768, "frozen generation drift")
    return request


def write_new(path, raw):
    """Publish a complete file without ever replacing an existing destination.

    A killed writer can leave a .pending file. It is preserved, not resumed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with tmp.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)  # Atomic no-clobber publication on the same filesystem.
    finally:
        tmp.unlink(missing_ok=True)


def save(path, value):
    write_new(path, encoded(value))


def event(root, kind, **fields):
    save(Path(root) / "events" / (uuid.uuid4().hex + ".json"), {"kind": kind, **fields})


def claim_request(lane_dir):
    """Persistent budget spent BEFORE HTTP. Existing/empty claim forbids resend."""
    p = Path(lane_dir) / "request.claim"
    with p.open("xb") as f:
        f.write(encoded({"outcome": "UNKNOWN", "budget_spent": True}))
        f.flush()
        os.fsync(f.fileno())


def isolated(path):
    path = Path(path).resolve()
    need(path.is_absolute() and not path.is_relative_to(ROOT),
         "run output must be outside Git checkout")
    return path


def safe_files(root):
    root = Path(root)
    for p in sorted(root.rglob("*")):
        need(not p.is_symlink(), "symlink in evidence directory")
        if p.is_file() and "checkpoints" not in p.relative_to(root).parts:
            yield p


def checkpoint(root, label):
    """Independent byte snapshot, including incomplete files, never a live hash list."""
    root = Path(root)
    need(label.replace("-", "").isalnum(), "invalid checkpoint label")
    dst = root / "checkpoints" / (label + "-" + uuid.uuid4().hex)
    dst.mkdir(parents=True, exist_ok=False)
    files = {}
    total = 0
    for p in safe_files(root):
        rel = p.relative_to(root).as_posix()
        with p.open("rb") as f:
            raw = f.read(64 * 1024 * 1024 + 1)
        total += len(raw)
        need(total <= 64 * 1024 * 1024, "snapshot exceeds 64 MiB budget; no truncation")
        write_new(dst / "data" / rel, raw)
        files[rel] = {"sha256": sha(raw), "size": len(raw)}
    save(dst / "manifest.json", {"schema": "m203.targeted.snapshot.v1", "files": files,
                                 "partial": label != "complete"})
    verify_checkpoint(dst)
    return dst


def verify_checkpoint(dst):
    dst = Path(dst)
    manifest = load(dst / "manifest.json")
    need(manifest["schema"] == "m203.targeted.snapshot.v1", "unknown snapshot schema")
    paths = list((dst / "data").rglob("*"))
    need(not any(p.is_symlink() for p in paths), "snapshot symlink")
    actual = {p.relative_to(dst / "data").as_posix() for p in paths if p.is_file()}
    need(actual == set(manifest["files"]), "snapshot missing/extra files")
    for rel, info in manifest["files"].items():
        path = dst / "data" / rel
        need(path.resolve().is_relative_to((dst / "data").resolve()) and not path.is_symlink(), "unsafe snapshot path")
        raw = path.read_bytes()
        need(sha(raw) == info["sha256"] and len(raw) == info["size"], "snapshot integrity failure")
    return manifest
