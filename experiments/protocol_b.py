"""M2-03 Protocol B: evidence only, never a scientific PASS decision.

CPU-only validation/reporting. Run with ``python -m experiments.protocol_b``.
Run paths in the experiment manifest are relative to that manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from experiments.lossless_verifier import verify_token_sequence
from experiments.workload_adapter import (
    load_workload_rows, validate_manifest_consistency, validate_workload_label,
    verify_workload_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
# Frozen identities from the archive's tests/verify_lossless.py, not its gates.
WORKLOADS = {
    "W1": ("w1/node1_chat_requests.jsonl", "w1.m1_03.v1", 30,
           "0de2387c4e6a8c39523bf869788a55ee7487c6a32318a949ce22e1f8f7b17360"),
    "W2": ("w2_v2/chat_requests.jsonl", "w2.m1_03.v2", 12,
           "bc26be01a4f5a813a4f6e843e2533e2f56d2e663918be2bd0e7b231e6385685c"),
    "W3": ("w3_v2/chat_requests.jsonl", "w3.m1_03.v2", 12,
           "57341b0fc8c2ebbdfcfc579a6343aea66b08ca11fe2bfed5bc53a2bd49f5d83b"),
}
ENV_FIELDS = (
    "session_id", "boot_id", "git_sha", "gpu_name", "gpu_uuid", "driver_version",
    "cuda_version", "torch_version", "vllm_version", "target_model",
    "target_revision", "draft_model", "draft_revision", "launcher_hashes",
    "config", "source_hashes", "plan_sha256",
)
CONFIG_FIELDS = ("environment", "concurrency", "warmup_requests", "request_timeout_seconds",
                 "sampling", "cache_policy", "quantization", "dtype", "generation_config",
                 "gpu_memory_utilization", "max_model_len", "max_num_seqs")
CONFIG_ENV = ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_HUB_CACHE",
              "VLLM_USE_V1", "ADAPTIVE_ENABLE_VRAM_GUARD", "ADAPTIVE_MAX_K",
              "ADAPTIVE_MIN_BATCH_K", "ADAPTIVE_PER_REQUEST",
              "ADAPTIVE_CONTROLLER_KWARGS", "ADAPTIVE_TELEMETRY_PATH",
              "PROTOCOL_B_TARGET_REVISION", "PROTOCOL_B_DRAFT_REVISION")
LAUNCHERS = ("scripts/launch_lane1_vanilla.sh", "scripts/launch_lane3_adaptive.sh",
             "scripts/switch_lane.sh")
REQUIRED_SOURCES = ("experiments/protocol_b.py", "experiments/protocol_b_capture.py",
                    "experiments/lossless_verifier.py", "experiments/workload_adapter.py",
                    "src/server/adaptive_scheduler.py", "src/server/budget_planner.py",
                    "src/proposer/adapter.py", "src/proposer/controller.py")
FROZEN_CONFIG = {
    "concurrency": 1, "warmup_requests": 0, "request_timeout_seconds": 600,
    "sampling": "frozen requests: temperature=0, seed=42, original caps/format",
    "cache_policy": "vllm-0.30.0 defaults; no live cache overrides; fresh server per lane",
    "quantization": "awq", "dtype": "half", "generation_config": "vllm",
    "gpu_memory_utilization": 0.85, "max_model_len": 4096, "max_num_seqs": 32,
}
FROZEN_ENV = {"HF_HUB_OFFLINE": "1", "VLLM_USE_V1": "1", "ADAPTIVE_ENABLE_VRAM_GUARD": "1",
              "ADAPTIVE_MAX_K": "4", "ADAPTIVE_MIN_BATCH_K": "0", "ADAPTIVE_PER_REQUEST": "1",
              "ADAPTIVE_CONTROLLER_KWARGS": ""}


class EvidenceError(ValueError):
    """Incomplete or ambiguous evidence cannot be interpreted."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value)


def hex_digest(value, width):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % width, value)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    require(isinstance(value, str), "Missing timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("Invalid timestamp") from exc
    require(parsed.tzinfo is not None, "Timestamp needs timezone")
    return parsed


def load_json(path):
    def unique_keys(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_keys)


def write_new(path, value):
    """Exclusive creation: never replace a previous result."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")


def load_plan():
    """Minimum port of archive load_plan; reuse develop's workload validators."""
    prompts, sources, seen_messages = [], {}, set()
    for workload, (relative, version, count, digest) in WORKLOADS.items():
        path = ROOT / "experiments/configs" / relative
        integrity = verify_workload_integrity(path)
        rows = load_workload_rows(path)
        validate_manifest_consistency(integrity, rows)
        validate_workload_label(workload, integrity.manifest["workload_id"])
        require(integrity.actual_sha256 == digest, f"{workload}: frozen SHA changed")
        require(integrity.manifest["dataset_version"] == version and len(rows) == count,
                f"{workload}: frozen package changed")
        sources[workload] = {"file": relative, "sha256": digest,
                             "manifest_sha256": file_sha(integrity.manifest_path)}
        for row in rows:
            request = row["request"]
            require(request["temperature"] == 0 and request["seed"] == 42,
                    "Frozen sampling changed")
            message_hash = fingerprint(request["messages"])
            require(message_hash not in seen_messages, "Duplicate frozen messages")
            seen_messages.add(message_hash)
            prompts.append({"workload": workload, "sample_id": row["sample_id"],
                            "dataset_version": row["dataset_version"],
                            "prompt_version": row["prompt_version"],
                            "request_sha256": fingerprint(request), "request": request})
    index_prompts(prompts)
    return {"prompts": prompts, "sources": sources,
            "plan_sha256": fingerprint({"prompts": prompts, "sources": sources})}


def index_prompts(prompts):
    require(isinstance(prompts, list) and bool(prompts), "Missing expected prompts")
    result = {}
    for prompt in prompts:
        require(isinstance(prompt, dict), "Invalid prompt")
        for field in ("workload", "sample_id", "dataset_version", "prompt_version"):
            require(isinstance(prompt.get(field), str) and bool(prompt[field]),
                    f"Missing prompt identity: {field}")
        require(re.fullmatch(r"[0-9a-f]{64}", prompt.get("request_sha256", "")),
                "Invalid request SHA")
        key = (prompt["workload"], prompt["sample_id"])
        require(key not in result, "Duplicate expected prompt")
        result[key] = prompt
    return result


def token_ids(value):
    # Same validation contract as archive validate_token_ids, with bool rejected.
    require(isinstance(value, list) and bool(value)
            and all(type(t) is int and t >= 0 for t in value), "Invalid token IDs")
    return value


def validate_run(run, expected, pair_id, lane, environment):
    require(isinstance(run, dict), "Invalid run")
    require(run.get("pair_id") == pair_id and run.get("lane") == lane,
            "Pair/lane identity mismatch")
    require(identifier(run.get("run_id")) and identifier(pair_id), "Invalid run/pair ID")
    require(run.get("complete") is True and not run.get("error"), "Incomplete/failed run")
    env = run.get("environment")
    require(isinstance(env, dict), "Missing environment")
    for field in ENV_FIELDS:
        require(field in env and env[field] not in (None, "", {}, []),
                f"Missing provenance: {field}")
        require(field in environment and env[field] == environment[field],
                f"Inconsistent provenance: {field}")
    for field in set(ENV_FIELDS) - {"config", "launcher_hashes", "source_hashes"}:
        require(nonempty(env[field]), f"Invalid provenance type/value: {field}")
    require(identifier(env["session_id"]) and nonempty(env["boot_id"]), "Invalid session identity")
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", env["gpu_uuid"]),
            "Invalid GPU UUID")
    for field in ("driver_version", "cuda_version", "torch_version", "vllm_version"):
        require(re.fullmatch(r"\d+\.\d+[A-Za-z0-9.+_-]*", env[field]), f"Invalid version: {field}")
    for field, width in (("git_sha", 40), ("target_revision", 40),
                         ("draft_revision", 40), ("plan_sha256", 64)):
        require(isinstance(env[field], str)
                and re.fullmatch(r"[0-9a-f]{%d}" % width, env[field]),
                f"Invalid provenance: {field}")
    require(isinstance(env["config"], dict)
            and all(field in env["config"] and env["config"][field] is not None
                    for field in CONFIG_FIELDS), "Incomplete runtime config")
    require(env["config"]["concurrency"] == 1, "Require sequential requests")
    for field in ("sampling", "cache_policy", "quantization", "dtype", "generation_config"):
        require(nonempty(env["config"][field]), f"Empty config: {field}")
    for field in ("concurrency", "warmup_requests", "request_timeout_seconds", "max_model_len", "max_num_seqs"):
        require(type(env["config"][field]) is int
                and env["config"][field] >= (0 if field == "warmup_requests" else 1),
                f"Invalid config type/value: {field}")
    require(type(env["config"]["gpu_memory_utilization"]) in (int, float)
            and 0 < env["config"]["gpu_memory_utilization"] <= 1, "Invalid GPU memory config")
    require(all(env["config"].get(k) == v for k, v in FROZEN_CONFIG.items()),
            "Runtime config differs from frozen Protocol B")
    settings = env["config"]["environment"]
    require(isinstance(settings, dict)
            and all(k in settings and isinstance(settings[k], str) for k in CONFIG_ENV),
            "Incomplete runtime environment")
    for key in CONFIG_ENV:
        require(key == "ADAPTIVE_CONTROLLER_KWARGS" or nonempty(settings[key]),
                f"Empty runtime environment: {key}")
    require(all(settings.get(k) == v for k, v in FROZEN_ENV.items()), "Unapproved runtime environment")
    require(settings["CUDA_VISIBLE_DEVICES"] == env["gpu_uuid"]
            and settings["HF_HUB_OFFLINE"] == "1", "GPU/cache selection inconsistent")
    for field in ("launcher_hashes", "source_hashes"):
        require(isinstance(env[field], dict) and bool(env[field]), f"Missing {field}")
        require(all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                    for v in env[field].values()), f"Invalid {field}")
    require(set(LAUNCHERS) == set(env["launcher_hashes"]), "Missing/unexpected launcher hash keys")
    require(set(REQUIRED_SOURCES) == set(env["source_hashes"]), "Missing/unrelated source hash keys")
    require(settings["PROTOCOL_B_TARGET_REVISION"] == env["target_revision"]
            and settings["PROTOCOL_B_DRAFT_REVISION"] == env["draft_revision"],
            "Model revision/environment mismatch")
    from experiments.protocol_b_capture import validate_live_server, TARGET, DRAFT
    require(env["gpu_name"] == "Tesla T4" and env["torch_version"] == "2.13.0+cu130"
            and env["cuda_version"] == "13.0" and env["vllm_version"] == "0.30.0",
            "Noncanonical runtime")
    require(env["target_model"] == TARGET and env["draft_model"] == DRAFT, "Model identity mismatch")
    validate_live_server(run.get("live_server"), lane, env)
    start, end = timestamp(run.get("started_at")), timestamp(run.get("finished_at"))
    require(start <= end, "Invalid run time interval")
    records = run.get("records")
    require(isinstance(records, list), "Missing records")
    index = {}
    for record in records:
        require(isinstance(record, dict), "Invalid record")
        key = (record.get("workload"), record.get("sample_id"))
        require(key in expected and key not in index, "Unexpected/duplicate prompt")
        for field in ("dataset_version", "prompt_version", "request_sha256"):
            require(record.get(field) == expected[key][field], f"Identity mismatch: {field}")
        for field in ("pair_id", "run_id", "lane"):
            require(record.get(field) == run[field], f"Record {field} mismatch")
        require(start <= timestamp(record.get("timestamp")) <= end, "Record time outside run")
        require(record.get("ok") is True and not record.get("error"),
                "Failed request retained; experiment invalid")
        token_ids(record.get("token_ids"))
        token_ids(record.get("prompt_token_ids"))
        require(record.get("finish_reason") in ("stop", "length"), "Invalid finish reason")
        require(type(record.get("completion_tokens")) is int
                and record["completion_tokens"] == len(record["token_ids"]),
                "Completion length disagrees with token IDs")
        index[key] = record
    require(index.keys() == expected.keys(), "Missing expected prompt")
    return index


def compare_runs(left, right):
    details = []
    for key in sorted(left):
        a, b = left[key], right[key]
        require(a["prompt_token_ids"] == b["prompt_token_ids"], "Prompt token IDs changed")
        result = verify_token_sequence(a["token_ids"], b["token_ids"], key[1], key[0])
        details.append({"workload": key[0], "sample_id": key[1],
                        "request_sha256": a["request_sha256"],
                        "left_run_id": a["run_id"], "right_run_id": b["run_id"],
                        "exact_match": result.is_match,
                        "first_mismatch": (result.first_discrepancy.to_dict()
                                           if result.first_discrepancy else None),
                        "completion_lengths": [len(a["token_ids"]), len(b["token_ids"])],
                        "finish_reasons": [a["finish_reason"], b["finish_reason"]],
                        "truncated": "length" in (a["finish_reason"], b["finish_reason"])})
    matched = sum(item["exact_match"] for item in details)
    return {"valid_prompt_pairs": len(details), "exact_matches": matched,
            "match_rate": matched / len(details),
            "mismatched_sample_ids": [{"workload": d["workload"], "sample_id": d["sample_id"]}
                                      for d in details if not d["exact_match"]],
            "truncated_pairs": sum(d["truncated"] for d in details),
            "full_completion_verified": not any(d["truncated"] for d in details),
            "details": details}


def build_summary(prompts, pairs, *, generated_at):
    """Two sequential V/A pairs, the minimum needed for both repeatabilities.

    No baseline qualification and no comparison of drift against a PASS rule.
    All runs must share the frozen environment, including physical GPU/session.
    """
    timestamp(generated_at)
    expected = index_prompts(prompts)
    require(isinstance(pairs, list) and len(pairs) == 2, "Exactly two controlled pairs required")
    environment = pairs[0].get("vanilla", {}).get("environment", {})
    indices, runs, seen_pairs, seen_runs = [], [], set(), set()
    previous_end = None
    for pair in pairs:
        pair_id = pair.get("pair_id")
        require(isinstance(pair_id, str) and bool(pair_id) and pair_id not in seen_pairs,
                "Missing/duplicate pair ID")
        seen_pairs.add(pair_id)
        lanes = {}
        for lane in ("vanilla", "adaptive"):
            run = pair.get(lane)
            lanes[lane] = validate_run(run, expected, pair_id, lane, environment)
            require(run["run_id"] not in seen_runs, "Reused run ID")
            seen_runs.add(run["run_id"])
            start = timestamp(run["started_at"])
            require(previous_end is None or previous_end <= start,
                    "Runs must be sequential Vanilla/Adaptive pairs")
            previous_end = timestamp(run["finished_at"])
            runs.append({key: value for key, value in run.items() if key != "records"})
        indices.append(lanes)
    require(timestamp(generated_at) >= previous_end, "Summary precedes run completion")
    return {"protocol": "B", "protocol_version": "1",
            "execution_sha": environment["git_sha"], "generated_at": generated_at,
            "environment": environment, "runs": runs,
            "prompts_expected": len(expected), "prompts_valid": len(expected),
            "vanilla_repeatability": compare_runs(indices[0]["vanilla"], indices[1]["vanilla"]),
            "adaptive_repeatability": compare_runs(indices[0]["adaptive"], indices[1]["adaptive"]),
            "paired_comparisons": [{"pair_id": pair["pair_id"],
                                    **compare_runs(index["vanilla"], index["adaptive"])}
                                   for pair, index in zip(pairs, indices)],
            "interpretation_status": "NOT_YET_APPROVED", "task_done": False}


def report(manifest_path, output_path, *, generated_at):
    from experiments.protocol_b_capture import LANES, parse_completion

    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    plan = load_plan()
    prompts = index_prompts(plan["prompts"])
    pairs, evidence = [], []
    for pair in manifest["pairs"]:
        loaded = {"pair_id": pair["pair_id"]}
        for lane in ("vanilla", "adaptive"):
            path = manifest_path.parent / pair[lane]
            loaded[lane] = load_json(path)
            require(loaded[lane]["environment"]["plan_sha256"] == plan["plan_sha256"],
                    "Frozen plan changed")
            journal = path.parent / "responses.jsonl"
            require(file_sha(journal) == loaded[lane].get("responses_sha256"),
                    "Raw response journal hash mismatch")
            events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            require(len(events) % 2 == 0, "Unfinished journal attempt")
            for attempt, result in zip(events[::2], events[1::2]):
                require(attempt.get("event") == "attempt" and result.get("event") == "result"
                        and attempt.get("request") == result.get("request")
                        and attempt.get("identity") == result.get("identity"), "Journal event mismatch")
            raw = events[1::2]
            require([item["record"] for item in raw] == loaded[lane]["records"],
                    "Run records differ from raw journal")
            for item in raw:
                record = item["record"]
                prompt = prompts[(record["workload"], record["sample_id"])]
                payload = dict(prompt["request"], model=LANES[lane][1], return_token_ids=True)
                require(item.get("request") == payload, "Journal request differs from frozen request")
                require(item.get("http_status") == 200 and not item.get("transport_error")
                        and not item.get("decoding_error"),
                        "Failed journal response")
                body = json.loads(item["raw_response_body"])
                require(body == item.get("response"), "Raw body/parsed payload mismatch")
                parsed = parse_completion(body, LANES[lane][1])
                require(all(record.get(key) == value for key, value in parsed.items()),
                        "Journal response differs from compared tokens")
            evidence.append({"path": pair[lane], "sha256": file_sha(path)})
        pairs.append(loaded)
    summary = build_summary(plan["prompts"], pairs, generated_at=generated_at)
    summary["input_evidence"] = evidence
    summary["manifest_sha256"] = file_sha(manifest_path)
    write_new(output_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()
    try:
        report(args.manifest, args.output, generated_at=args.generated_at)
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"INVALID_EVIDENCE: {exc}\n")


if __name__ == "__main__":
    main()
