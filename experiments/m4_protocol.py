"""Frozen M4 manifest and committed-byte workload identities (CPU only)."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path

from experiments.m4_treatments import TREATMENTS, VLLM_VERSION, definition

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "W1": "experiments/configs/w1/node1_chat_requests.jsonl",
    "W2": "experiments/configs/w2_v2/chat_requests.jsonl",
    "W3": "experiments/configs/w3_v2/chat_requests.jsonl",
}
CONCURRENCIES = (1, 2, 4, 8, 16, 32)


def git(*args, root=ROOT) -> str:
    return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                    *args], cwd=root, text=True).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")


def source_gate(source_commit: str, root=ROOT) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or git("rev-parse", "HEAD", root=root) != source_commit:
        raise ValueError("SOURCE_GATE FAIL: exact frozen SHA required")
    # Runtime artifacts MUST live outside this detached source checkout.
    if git("status", "--porcelain", "--untracked-files=all", root=root):
        raise ValueError("SOURCE_GATE FAIL: run checkout must be clean")


def workload_identity(workload: str, commit: str = "HEAD", root=ROOT) -> dict:
    path = WORKLOADS[workload]
    data = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                    "show", f"{commit}:{path}"], cwd=root)
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    ids = [r["sample_id"] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Missing/duplicate workload sample identity")
    settings = [{key: r["request"].get(key) for key in ("temperature", "seed", "max_tokens")}
                for r in rows]
    if any(s["temperature"] != 0 or s["seed"] != 42 or not isinstance(s["max_tokens"], int)
           or s["max_tokens"] <= 0 for s in settings):
        raise ValueError("Unfrozen request settings")
    return dict(path=path, sha256=sha256(data), blob=git("rev-parse", f"{commit}:{path}", root=root),
                prompt_ids=ids, count=len(rows), request_settings=settings,
                prompt_hashes=[sha256(json.dumps(r["request"], sort_keys=True,
                                                ensure_ascii=False).encode()) for r in rows])


def build_manifest(source_commit: str | None = None, root=ROOT) -> dict:
    # null is a preparation template, never executable. Runtime freeze fills HEAD.
    workloads = {w: workload_identity(w, source_commit or "HEAD", root) for w in WORKLOADS}
    cells = []
    for treatment, workload, concurrency in itertools.product(TREATMENTS, WORKLOADS, CONCURRENCIES):
        cells.append(dict(cell_id=f"{treatment}__{workload}__c{concurrency}",
                          **definition(treatment), workload=workload,
                          workload_sha256=workloads[workload]["sha256"], concurrency=concurrency,
                          sample_policy="cycle-rotate", repeats=5, measured_requests=32,
                          warmup=3, warmup_policy="m3-fixed3", temperature=0, seed=42,
                          sample_indices_by_repeat=[[(repeat * (32 % workloads[workload]["count"]) + index)
                                                    % workloads[workload]["count"] for index in range(32)]
                                                    for repeat in range(5)],
                          output_caps=[s["max_tokens"] for s in workloads[workload]["request_settings"]]))
    return dict(schema_version="m4-02-06-v1", source_commit=source_commit,
                expected_gpu="NVIDIA A100", expected_capability=[8, 0],
                vllm_version=VLLM_VERSION, profile_requirement="profile_sm80.json",
                workloads=workloads, cells=cells)


def validate_manifest(manifest: dict, source_commit: str, root=ROOT) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Manifest requires full SOURCE_COMMIT")
    if manifest != build_manifest(source_commit, root):
        raise ValueError("Manifest differs from frozen source/protocol; missing, duplicate or altered cells")


def validate_working_workloads(manifest: dict, root=ROOT) -> None:
    for identity in manifest["workloads"].values():
        if sha256((root / identity["path"]).read_bytes()) != identity["sha256"]:
            raise ValueError("Working workload bytes differ from committed blob; do not normalize the corpus")
