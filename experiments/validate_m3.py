"""Fail-closed M3-05 cell validation; pilot evidence is never a campaign input by default."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from experiments import aggregate_results as agg, modal_lanes as lanes
from experiments import vllm_runtime_patch as runtime_patch
from experiments.bench_source import snapshot
from experiments.workload_adapter import load_workload_rows, verify_workload_integrity

ROOT = Path(__file__).resolve().parents[1]


def validate_cell(out: Path) -> dict:
    receipt = json.loads((out / "submission.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    server = json.loads((out / "server_env.json").read_text(encoding="utf-8"))
    for key in ("lane", "server_run_id", "bench_run_id", "cell", "commit", "call_id"):
        if meta.get(key) != receipt.get(key) or receipt.get(key) is None:
            raise ValueError(f"Receipt/collected identity mismatch: {key}")
    lane, cell = receipt["lane"], receipt["cell"]
    # [M5-01] W1_V2 is accepted with its own serving contract (lanes.workload_serving).
    if (lane not in lanes.LANES or cell["workload"] not in lanes.ALL_WORKLOADS
            or cell["concurrency"] not in (1, 2, 4, 8, 16, 32)
            or cell["repeats"] != 5 or cell["num_requests"] != 32):
        raise ValueError("Noncanonical M3 cell")
    if meta.get("returncode") != 0 or meta.get("server_instance_unchanged") is not True:
        raise ValueError("Failed harness or server restarted during cell")
    draft = None if lane == "vanilla" else (
        lanes.EAGLE_DRAFT_MODEL if lane == "eagle" else lanes.DRAFT_MODEL)
    revision = None if lane == "vanilla" else (
        lanes.EAGLE_REVISION if lane == "eagle" else lanes.DRAFT_REVISION)
    expected = {"lane": lane, "run_id": receipt["server_run_id"], "source_commit": receipt["commit"],
                "target_model": lanes.TARGET_MODEL, "target_revision": lanes.TARGET_REVISION,
                "draft_model": draft, "draft_revision": revision,
                "served_model_name": lanes.LANES[lane].served_model_name, "vllm_version": "0.30.0"}
    if any(server.get(k) != v for k, v in expected.items()) or not server.get("instance_id"):
        raise ValueError("Server source/model/instance identity mismatch")
    serving = lanes.workload_serving(cell["workload"])
    if server.get("vllm_args") != lanes.build_vllm_args(
            lane, profile="l4-bf16", max_model_len=serving["max_model_len"]):
        raise ValueError("Frozen server arguments mismatch")
    if serving["vllm_runtime"] == "backport-55708" and (
            server.get("runtime_variant") != runtime_patch.RUNTIME_LABEL
            or server.get("vllm_outputs_sha256") != runtime_patch.PATCHED_OUTPUTS_SHA256):
        raise ValueError("Patched vLLM runtime (backport #55708) required for this workload")
    if meta.get("endpoint_model_ids") != [lanes.LANES[lane].served_model_name]:
        raise ValueError("Missing endpoint model verification")
    raw = agg._load_jsonl(out / "raw.jsonl")
    warm = agg._load_jsonl(out / "warmup.jsonl")
    if len(raw) != 160 or Counter(r.get("repeat_id") for r in raw) != {
            f"r{i:03}": 32 for i in range(1, 6)}:
        raise ValueError("Expected exactly 160 measured requests: 5 repeats x 32")
    if len(warm) != 3 or any(r.get("phase") != "warmup" for r in warm):
        raise ValueError("Expected exactly 3 separate warm-ups")
    if len({r.get("request_id") for r in raw + warm}) != 163:
        raise ValueError("Missing or duplicate request identities")
    packed, sha = snapshot(ROOT, receipt["commit"])
    try:
        path = Path(packed.name) / lanes.workload_file(cell["workload"])
        integrity = verify_workload_integrity(path)
        corpus = load_workload_rows(path)
    finally:
        packed.cleanup()
    if sha != receipt["commit"]:
        raise ValueError("Invalid submitted source SHA")
    for records, phase in ((raw, "measured"), (warm, "warmup")):
        seen = set()
        for row in records:
            identity = (row.get("repeat_id"), row.get("request_index"))
            if identity in seen:
                raise ValueError("Duplicate request index within repeat")
            seen.add(identity)
            index = row["request_index"]
            if type(index) is not int or index not in range(32 if phase == "measured" else 3):
                raise ValueError("Invalid request index")
            repeat = int(row["repeat_id"][1:]) if phase == "measured" else 1
            offset = ((repeat - 1) * (32 % len(corpus))) % len(corpus) if phase == "measured" else 0
            sample_index = (index + offset) % len(corpus)
            exp = {"experiment_id": receipt["bench_run_id"], "lane": lane,
                   "workload": cell["workload"], "concurrency": cell["concurrency"], "phase": phase,
                   "commit": sha, "server_run_id": receipt["server_run_id"],
                   "server_instance_id": server["instance_id"], "sample_policy": "cycle-rotate",
                   "sample_id": corpus[sample_index]["sample_id"], "sample_index": sample_index,
                   "sample_offset": offset, "canonical_sample_count": len(corpus),
                   "workload_file_sha256": integrity.actual_sha256, "success": True,
                   "metrics_complete": True, "http_status": 200, "target_revision": lanes.TARGET_REVISION,
                   "draft_revision": revision, "model": lanes.LANES[lane].served_model_name}
            request = corpus[sample_index]["request"]
            exp.update({k: request.get(k) for k in (
                "temperature", "seed", "response_format")})
            exp["max_completion_tokens"] = request["max_tokens"]
            if any(row.get(k) != v for k, v in exp.items()) or row.get("error_type") or row.get("error_message"):
                raise ValueError(f"Request protocol/identity mismatch: {identity}")
    warm_finished = max(datetime.fromisoformat(r["timestamp_utc"]) +
                        timedelta(milliseconds=r["e2e_ms"]) for r in warm)
    if warm_finished > min(datetime.fromisoformat(r["timestamp_utc"]) for r in raw):
        raise ValueError("Measured execution began before warm-up completed")
    agg.validate_measured_records(raw)
    summary = None  # request metrics are aggregated only with measured server VRAM
    acceptance = None
    if lane != "vanilla":
        acceptance = agg.aggregate_static_acceptance(raw)
    if (out / "raw.vram.jsonl").exists():
        joined = agg._load_jsonl(out / "raw.vram.jsonl")
        if len(joined) != 160 or any({k: v for k, v in a.items() if k != "run_vram_peak_mb"} !=
                                   {k: v for k, v in b.items() if k not in (
                                       "run_vram_peak_mb", "vram_join_samples", "vram_source")}
                                   for a, b in zip(raw, joined)):
            raise ValueError("VRAM join changed raw evidence")
        if any(r.get("run_vram_peak_mb") is None for r in joined):
            raise ValueError("Missing VRAM samples")
        summary = agg._aggregate_measured_records(joined)[0]
    duration = sum({r["repeat_id"]: r["run_duration_s"] for r in raw}.values())
    return {"protocol_status": "VALID", "commit": sha, "measured_requests": 160,
            "repeats": 5, "requests_per_repeat": 32, "warmup_attempted": 3,
            "warmup_successful": 3, "warmup_excluded": True, "request_failures": 0,
            "workload_hash": integrity.actual_sha256, "server_instance_id": server["instance_id"],
            "target_revision": lanes.TARGET_REVISION, "draft_revision": revision,
            "speculative_alpha_tau": acceptance, "measured_duration_s": duration,
            "measured_requests_per_s": 160 / duration,
            "output_tokens_per_s": sum(r["output_tokens"] for r in raw) / duration,
            "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_cell(args.cell_dir)
    content = json.dumps(result, indent=2) + "\n"
    path = args.cell_dir / "validation.json"
    if path.exists() and path.read_bytes() != content.encode("utf-8"):
        raise FileExistsError(f"Conflicting validation: {path}")
    if not path.exists():
        path.write_bytes(content.encode("utf-8"))
    print(content)


if __name__ == "__main__":
    main()
