"""Read-only M2-03 forensic audit of a fixed Git checkpoint (stdlib only).

No network, model execution, working-tree evidence fallback, or file output.
Exit 0: valid audit; 1: valid audit + --require-gate-pass + failed token gate;
2: invalid integrity/schema/provenance. Telemetry has its own linkage status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

CHECKPOINT = "bdb26dd89259dcc6b4085041c783bc9536e1287c"
RUNS = {
    "cacheoff": ("experiments/runs/m2_03_cacheoff/m2-03-cacheoff-full-20261014-r01",
                 "4b830a8b7530b9bb3218fb9628477ffd164705ba"),
    "canonical": ("experiments/runs/m2_03/m2-03-20261013-r03",
                  "6c5adda05a562eebfc7182ead3e7d6377737c380"),
}
IDENTITY = ("workload", "sample_id", "dataset_version", "prompt_version", "request_sha256")
COUNTS = {"W1": 30, "W2": 12, "W3": 12}
LABELS = {"W1": "w1_node1_invoice_extraction", "W2": "w2_text_to_sql", "W3": "w3_free_analysis"}
OLD_VERIFIER = ("experiments/runs/m2_03/diagnostics/code_backups/"
                "verify_lossless.before_guard.27b5570839b571c491422124333ee91d59f9502cde0ae49570f35c9da17280fa.py.bak")


class AuditError(ValueError):
    """Evidence cannot support a valid analysis."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8"))


def checked_bytes(raw, expected, label):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"invalid SHA-256: {label}")
    require(digest(raw) == expected, f"checksum mismatch: {label}")
    return raw


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise AuditError(f"non-finite JSON constant: {value}")


def parse_json(raw):
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                           parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("malformed UTF-8 JSON") from exc
    require(isinstance(value, dict), "expected JSON object")
    return value


def parse_jsonl(raw):
    # One optional final newline; blank records are never silently discarded.
    lines = raw.splitlines()
    require(bool(lines) and all(line.strip() for line in lines), "empty/blank JSONL record")
    return [parse_json(line) for line in lines]


class GitBlobs:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.hashes = {}
        self.cache = {}
        require(self.git("rev-parse", "--show-toplevel").decode().strip().replace("\\", "/").casefold()
                == self.repo.as_posix().casefold(), "repository root mismatch")
        require(self.git("rev-parse", "--verify", CHECKPOINT + "^{commit}").decode().strip()
                == CHECKPOINT, "checkpoint commit unavailable")

    def git(self, *args):
        try:
            return subprocess.run(
                ["git", "-c", f"safe.directory={self.repo.as_posix()}", "-C", str(self.repo), *args],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AuditError("Git blob/provenance unavailable; no working-tree fallback") from exc

    def read(self, path, sha=CHECKPOINT):
        require(isinstance(path, str) and path and "\\" not in path and ":" not in path
                and not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts,
                "invalid blob path")
        require(re.fullmatch(r"[0-9a-f]{40}", sha) is not None, "full Git SHA required")
        spec = f"{sha}:{path}"
        if spec in self.cache:
            return self.cache[spec]
        require(self.git("cat-file", "-t", spec).strip() == b"blob", "expected Git blob")
        raw = self.git("cat-file", "blob", spec)
        self.hashes[spec] = digest(raw)
        self.cache[spec] = raw
        return raw


def index_records(rows, expected=None):
    result = {}
    for row in rows:
        key = (row.get("workload"), row.get("sample_id"))
        require(all(isinstance(x, str) and x for x in key), "invalid composite identity")
        require(key not in result, f"duplicate record: {key}")
        result[key] = row
    if expected is not None:
        require(set(result) == set(expected), "missing/extra records")
    return result


def token_ids(value):
    require(isinstance(value, list) and bool(value)
            and all(type(x) is int and x >= 0 for x in value), "invalid token IDs")
    return value


def sequence_comparison(left, right):
    prefix = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y),
                  min(len(left), len(right)))
    exact = left == right
    return {
        "exact_token_match": exact, "equal_prefix_length": prefix,
        "vanilla_token_count": len(left), "adaptive_token_count": len(right),
        "first_mismatch": None if exact else {
            "position_zero_based": prefix,
            "vanilla_token_id": left[prefix] if prefix < len(left) else None,
            "adaptive_token_id": right[prefix] if prefix < len(right) else None,
        },
        "boundary_window": None if exact else {
            "start": max(0, prefix - 4), "end_exclusive": prefix + 5,
            "vanilla": left[max(0, prefix - 4):prefix + 5],
            "adaptive": right[max(0, prefix - 4):prefix + 5],
        },
    }


def compare_pair(vanilla, adaptive, frozen):
    for lane, row in (("vanilla", vanilla), ("adaptive", adaptive)):
        require(all(row.get(f) == frozen[f] for f in IDENTITY), f"identity mismatch: {lane}")
        require(row.get("lane") == lane and row.get("ok") is True
                and row.get("http_status") == 200, f"unsuccessful/wrong lane: {lane}")
        ids = token_ids(row.get("token_ids"))
        token_ids(row.get("prompt_token_ids"))
        require(type(row.get("completion_tokens_reported")) is int
                and row["completion_tokens_reported"] == len(ids), "reported/raw count mismatch")
        require(row.get("finish_reason") in ("stop", "length"), "invalid finish reason")
        require(isinstance(row.get("response_id"), str) and bool(row["response_id"]), "missing response ID")
        require(isinstance(row.get("content"), str), "invalid content")
    require(vanilla["prompt_token_ids"] == adaptive["prompt_token_ids"], "prompt token mismatch")
    result = {f: frozen[f] for f in IDENTITY}
    result.update(sequence_comparison(vanilla["token_ids"], adaptive["token_ids"]))
    result.update({
        "prompt_token_count": len(vanilla["prompt_token_ids"]),
        "prompt_token_ids_sha256": fingerprint(vanilla["prompt_token_ids"]),
        "finish_reason_vanilla": vanilla["finish_reason"],
        "finish_reason_adaptive": adaptive["finish_reason"],
        "truncated": "length" in (vanilla["finish_reason"], adaptive["finish_reason"]),
        "passed": result["exact_token_match"] and vanilla["finish_reason"] == adaptive["finish_reason"],
        "content_equal_diagnostic_only": vanilla["content"] == adaptive["content"],
    })
    return result


def link_telemetry(adaptive, telemetry):
    """Join sample identity -> response ID -> unique internal ID, never row index."""
    links, used = {}, set()
    ids = [r.get("request_id") for r in telemetry]
    if not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(ids):
        return {"status": "FAIL", "reason": "invalid/duplicate telemetry request IDs", "links": []}
    responses = [r["response_id"] for r in adaptive.values()]
    if len(set(responses)) != len(responses):
        return {"status": "FAIL", "reason": "duplicate response IDs", "links": []}
    for key, row in sorted(adaptive.items()):
        hits = [i for i, r in enumerate(telemetry)
                if re.fullmatch(re.escape(row["response_id"]) + r"-[0-9a-f]{8}", r["request_id"])]
        if len(hits) != 1 or hits[0] in used:
            return {"status": "FAIL", "reason": "missing/ambiguous telemetry linkage", "links": []}
        i = hits[0]
        if telemetry[i].get("lane") != "adaptive":
            return {"status": "FAIL", "reason": "wrong telemetry lane", "links": []}
        used.add(i)
        links[key] = {"workload": key[0], "sample_id": key[1],
                      "response_id": row["response_id"], "request_id": telemetry[i]["request_id"]}
    if len(used) != len(telemetry):
        return {"status": "FAIL", "reason": "unmatched telemetry records", "links": []}
    return {"status": "PASS", "reason": None, "links": list(links.values())}


def frozen_plan(reader, meta, execution):
    sources = meta["sources"]
    require(set(sources) == set(COUNTS), "workload set mismatch")
    prompts, messages = [], set()
    for workload, count in COUNTS.items():
        src = sources[workload]
        raw = checked_bytes(reader.read(src["file"]), src["file_sha256"], src["file"])
        require(reader.read(src["file"], execution) == raw, "execution/checkpoint workload drift")
        manifest = parse_json(reader.read(src["manifest_file"]))
        require(manifest["package_sha256"] == src["manifest_package_sha256_claim"], "package claim mismatch")
        require(manifest["workload_id"] == LABELS[workload], "workload label mismatch")
        require(src["sample_count"] == manifest["sample_count"] == count, "workload count mismatch")
        selected = [key for key, path in manifest["artifacts"].items() if path == src["file"]]
        require(len(selected) == 1, "ambiguous/missing request artifact in manifest")
        checked_bytes(raw, manifest["file_hashes"][selected[0]], "manifest request checksum")
        rows = parse_jsonl(raw)
        require(len(rows) == count, "frozen row count mismatch")
        for row in rows:
            for f in ("workload_id", "dataset_version", "prompt_version"):
                require(row[f] == manifest[f], f"manifest identity mismatch: {f}")
            require(row["dataset_version"] == src["dataset_version"], "source version mismatch")
            req = row["request"]
            require(req["seed"] == 42 and type(req["seed"]) is int
                    and type(req["temperature"]) in (int, float) and req["temperature"] == 0,
                    "frozen sampling mismatch")
            require(type(req["max_tokens"]) is int and req["max_tokens"] > 0, "invalid max_tokens")
            for f in ("model", "temperature", "max_tokens", "seed", "response_format"):
                require(req.get(f) == manifest.get(f), f"generation setting mismatch: {f}")
            require(isinstance(req["messages"], list) and bool(req["messages"]), "invalid messages")
            message_hash = fingerprint(req["messages"])
            require(message_hash not in messages, "duplicate frozen messages")
            messages.add(message_hash)
            prompts.append({"workload": workload, "sample_id": row["sample_id"],
                            "dataset_version": row["dataset_version"], "prompt_version": row["prompt_version"],
                            "request_sha256": fingerprint(req)})
    index_records(prompts)
    require(len(prompts) == 54, "denominator must remain 54")
    require(fingerprint({"sources": sources, "prompts": prompts}) == meta["plan_sha256"], "plan checksum mismatch")
    return prompts


def audit_run(reader, name):
    path, execution = RUNS[name]
    summary = parse_json(reader.read(path + "/summary.json"))
    meta = parse_json(reader.read(path + "/reference_meta.json"))
    require(summary["run_id"] == meta["run_id"] == path.rsplit("/", 1)[1], "run ID mismatch")
    require(meta["complete"] is True and meta["stage"] == "vanilla_reference", "incomplete reference")
    require(meta["failed_requests"] == 0, "reference request failure")
    for doc in (summary, meta):
        require(doc["runtime"]["git_sha"] == execution, "execution SHA mismatch")
        require(doc["runtime"]["vllm_python_package_version"] == "0.30.0", "vLLM provenance mismatch")
        require(doc["expected_prompts"] == 54, "denominator mismatch")
    for f in ("sources", "plan_sha256"):
        require(summary[f] == meta[f], f"reference/summary mismatch: {f}")
    runtime = meta["runtime"]
    require({k: v for k, v in runtime.items() if k != "timestamp_utc"}
            == {k: v for k, v in summary["runtime"].items() if k != "timestamp_utc"}, "runtime drift between lanes")
    for group in ("implementation_sha256", "launcher_sha256"):
        for file, sha256 in runtime[group].items():
            checked_bytes(reader.read(file, execution), sha256, file)
    checked_bytes(reader.read("experiments/workload_adapter.py", execution), runtime["workload_adapter_sha256"], "adapter")
    verifier = "tests/verify_lossless_cacheoff.py" if name == "cacheoff" else OLD_VERIFIER
    checked_bytes(reader.read(verifier, execution if name == "cacheoff" else CHECKPOINT),
                  runtime["verification_script_sha256"], "verifier source")
    prompts = frozen_plan(reader, meta, execution)
    frozen = index_records(prompts)
    lanes = {}
    for lane in ("vanilla", "adaptive"):
        file = path + "/" + lane + ".jsonl"
        raw = checked_bytes(reader.read(file), summary[lane + "_jsonl_sha256"], file)
        if lane == "vanilla":
            checked_bytes(raw, meta["reference_jsonl_sha256"], "reference JSONL")
        lanes[lane] = index_records(parse_jsonl(raw), frozen)
    pairs = [compare_pair(lanes["vanilla"][k], lanes["adaptive"][k], frozen[k]) for k in sorted(frozen)]
    # Historical summary counts pair passes (token equality AND matching finish
    # reasons). Keep raw token equality as a separate diagnostic, not a new gate.
    matched = sum(p["passed"] for p in pairs)
    exact_tokens = sum(p["exact_token_match"] for p in pairs)
    truncated = sum(p["truncated"] for p in pairs)
    gate = all(p["passed"] for p in pairs)
    expected = {"valid_pairs": 54, "exact_sequence_matches": matched,
                "truncated_pairs": truncated, "token_id_gate_passed": gate,
                "full_completion_verified": truncated == 0,
                "exact_sequence_match_rate": matched / 54}
    for field, value in expected.items():
        require(type(summary[field]) is type(value) and summary[field] == value,
                f"summary consistency mismatch: {field}")
    saved = index_records(summary["comparisons"], frozen)
    for p in pairs:
        row = saved[p["workload"], p["sample_id"]]
        for f in ("passed", "first_mismatch", "vanilla_token_count", "adaptive_token_count",
                  "finish_reason_vanilla", "finish_reason_adaptive", "truncated", "content_equal_diagnostic_only"):
            require(type(row[f]) is type(p[f]) and row[f] == p[f], f"summary pair mismatch: {f}")
        failures = ([] if p["exact_token_match"] else ["generated_token_ids_mismatch"])
        if p["finish_reason_vanilla"] != p["finish_reason_adaptive"]:
            failures.append("finish_reason_mismatch")
        require(row["failures"] == failures, "summary failure list mismatch")
    telemetry = ({"status": "NOT_AVAILABLE", "reason": "not preserved for canonical run", "links": []}
                 if name == "canonical" else link_telemetry(lanes["adaptive"], parse_jsonl(reader.read(path + "/adaptive_telemetry.jsonl"))))
    result = {"run_id": summary["run_id"], "checkpoint_sha": CHECKPOINT, "execution_sha": execution,
              "verifier_source_kind": "execution_blob" if name == "cacheoff" else "checkpoint_archive_hash_matched",
              "denominator": 54, "valid_pairs": 54, "exact_sequence_matches": matched,
              "exact_sequence_match_rate": matched / 54,
              "exact_token_sequence_matches": exact_tokens,
              "mismatches": 54 - matched, "truncated_pairs": truncated,
              "token_id_gate_passed": gate, "full_completion_verified": truncated == 0,
              "pairs": pairs, "telemetry_linkage": telemetry}
    return result, lanes


def cross_run(old, off):
    results = []
    for key in (("W1", "m0_04_021"), ("W3", "w3_business_v2_001")):
        rows = [run[lane][key] for run in (old, off) for lane in ("vanilla", "adaptive")]
        require(all(all(r[f] == rows[0][f] for f in IDENTITY + ("prompt_token_ids",)) for r in rows),
                "cross-run identity/prompt mismatch")
        comparisons = {}
        for label, left, right in (("vanilla_across_runs", rows[0], rows[2]),
                                   ("adaptive_across_runs", rows[1], rows[3]),
                                   ("canonical_adaptive_vs_cacheoff_vanilla", rows[1], rows[2]),
                                   ("canonical_vanilla_vs_cacheoff_adaptive", rows[0], rows[3])):
            comparisons[label] = sequence_comparison(left["token_ids"], right["token_ids"])
        results.append({"workload": key[0], "sample_id": key[1], "identity_and_prompt_equal": True,
                        "comparisons": comparisons})
    return results


def analyze(repo):
    reader = GitBlobs(repo)
    off, off_raw = audit_run(reader, "cacheoff")
    old, old_raw = audit_run(reader, "canonical")
    linkage = off["telemetry_linkage"]
    valid = linkage["status"] == "PASS"
    return {"schema_version": "m2_03_forensic.v1", "analysis_valid": valid,
            "token_id_gate_passed": off["token_id_gate_passed"],
            "telemetry_linkage": {"status": linkage["status"], "reason": linkage["reason"]},
            "errors": [] if valid else ["cacheoff telemetry linkage failed: " + linkage["reason"]],
            "checkpoint_sha": CHECKPOINT, "runs": {"cacheoff": off, "canonical": old},
            "cross_run": cross_run(old_raw, off_raw),
            "blob_sha256": dict(sorted(reader.hashes.items())),
            "integrity_scope": "Consumed request-plan, raw-output and provenance blobs; not entire workload packages.",
            "limitations": ["Artifact audit only; no proof of effective GPU runtime or unpatched installed vLLM.",
                            "No target-logits, KV-state equality or causal inference.",
                            "Truncated token equality does not establish full completion.",
                            "Canonical verifier is hash-matched archive, not source in its execution commit.",
                            "Ancillary manifest artifacts and package_sha256 claims are not full-package integrity verification."]}


def exit_status(report, require_gate_pass=False):
    if not report["analysis_valid"]:
        return 2
    return int(require_gate_pass and not report["token_id_gate_passed"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-gate-pass", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.repo)
    except (AuditError, KeyError, TypeError, ValueError, AttributeError, IndexError, OSError) as exc:
        report = {"schema_version": "m2_03_forensic.v1", "analysis_valid": False,
                  "token_id_gate_passed": None, "checkpoint_sha": CHECKPOINT,
                  "telemetry_linkage": {"status": "NOT_EVALUATED"},
                  "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))
    return exit_status(report, args.require_gate_pass)


if __name__ == "__main__":
    sys.exit(main())
