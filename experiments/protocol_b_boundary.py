"""Approved sample-004/index-279 diagnostic: preparation, existing capture, analysis.

No command launches a GPU server. See docs/runbooks/m2-03-boundary-279.md.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from experiments.protocol_b import (ROOT, EvidenceError, canonical_json, file_sha, fingerprint,
                                    load_json, load_plan, require, utc_now, write_new)
from experiments.protocol_b_capture import capture, command, reserve_run, server_processes
from experiments.protocol_b_capture import control_gpu_selection, validate_gpu_selection
from experiments.protocol_b_boundary_observer import (map_rows, selected_tokens,
    history_slots, validate_full_binding, LAYER1, LOCALIZATION_SCHEMA, EARLY_SCHEMA,
    POSTWRITE_SCHEMA, position177_slot, postwrite_ownership)
from experiments.protocol_b_boundary_observer import CONTROL_PROFILE

CHECKPOINT = "4372f8b81998e8a50ee2b87ea6d7c4a9c4b523c5"
BASE_EXECUTION = "a63baa1c42fd75f542f876ab512765149ddd5e20"
EVIDENCE = "experiments/runs/m2_03/protocol_b_20261018T085118Z_2aad4c00"
SAMPLE = "w3_business_v2_004"
TARGET = 279
SCHEMA = "m203.boundary.comparison.v1"
CONTROL_CHECKPOINT = "9c0f15e8ebcef4d3fff3bd4aa07b6f41d3288ff2"
CONTROL_BASELINE_NAME = "postwrite-position177-c306aef-20261019T041028Z.tar.gz"
CONTROL_BASELINE_BLOB = "b5590f982b97c0aa6a05ea0217eb6413f7bb2510"
CONTROL_BASE_EXECUTION = "c306aef3c8ffea97aa2e045507c2245ee77df220"
CONTROL_SOURCE_PATHS = ("experiments/protocol_b_boundary.py", "experiments/protocol_b_boundary_observer.py",
    "experiments/protocol_b_boundary_sources.py", "src/server/adaptive_scheduler.py",
    "experiments/protocol_b.py", "experiments/protocol_b_capture.py")

# Approved orchestration-only fix:
# switch_lane.sh must invoke tracked launchers via `bash` instead of chmod +x,
# preventing executable-bit drift during the diagnostic.
APPROVED_SWITCH_LANE_SHA256 = (
    "0dd97d5eb0ca74ce3457b9e6fee8d9260e4bcde5e5ca1447ea814a0f9c46c3df"
)


def reference():
    def read(path):
        return json.loads(subprocess.check_output(
            ["git", "show", CHECKPOINT + ":" + EVIDENCE + "/" + path], cwd=ROOT))
    plan = read("frozen_plan.json")
    require(plan == load_plan(), "Frozen plan differs from Protocol B checkpoint")
    runs = [read(pair + "-" + lane + "/run.json")
            for pair in ("pair1", "pair2") for lane in ("vanilla", "adaptive")]
    ids = [p["sample_id"] for p in plan["prompts"]]
    require(len(ids) == len(set(ids)) == 54 and ids.index(SAMPLE) == 45, "Prelude order changed")
    for run in runs:
        require(run["complete"] and run["environment"]["git_sha"] == BASE_EXECUTION,
                "Unexpected reference execution")
        require([r["sample_id"] for r in run["records"]] == ids, "Reference order differs")
        require(all(r["ok"] and r["request_sha256"] == p["request_sha256"]
                    for r, p in zip(run["records"], plan["prompts"])), "Reference request changed")
    for i in (0, 1):
        require(all(a["token_ids"] == b["token_ids"] and
                    a["prompt_token_ids"] == b["prompt_token_ids"]
                    for a, b in zip(runs[i]["records"], runs[i + 2]["records"])),
                "Reference repeatability differs")
    require(is_fingerprint(runs[0]["records"][45], runs[1]["records"][45]),
            "Reference fingerprint differs")
    return plan, runs[:2]


def first_difference(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)) if len(left) != len(right) else None)


def is_fingerprint(vanilla, adaptive):
    v, a = vanilla["token_ids"], adaptive["token_ids"]
    return (vanilla["sample_id"] == adaptive["sample_id"] == SAMPLE
            and first_difference(v, a) == TARGET
            and len(v) == 284 and len(a) == 282
            and v[TARGET] == 79126 and a[TARGET] == 128273
            and vanilla["finish_reason"] == adaptive["finish_reason"] == "stop")


def prepare(directory, session):
    plan, runs = reference()
    require(not command(["git", "status", "--porcelain", "--untracked-files=no"]),
            "Require reviewed clean execution checkout")
    command(["git", "ls-files", "--error-unmatch", "experiments/protocol_b_boundary.py",
             "experiments/protocol_b_boundary_observer.py", "experiments/protocol_b_boundary_sources.py"])
    execution = command(["git", "rev-parse", "HEAD"])
    require(command(["git", "branch", "--show-current"]) == "fix/m2-03-lossless-clean",
            "Unexpected execution branch")
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    require(gpu.startswith("GPU-"), "Pin physical GPU UUID")
    for path, digest in runs[0]["environment"]["source_hashes"].items():
        if path != "experiments/protocol_b_capture.py":
            if path == "src/server/adaptive_scheduler.py":
                from experiments.protocol_b_boundary_sources import validate_scheduler_observation
                raw = subprocess.check_output(["git", "show", CHECKPOINT + ":" + path], cwd=ROOT)
                import hashlib
                require(hashlib.sha256(raw).hexdigest() == digest, "Frozen scheduler source drift")
                validate_scheduler_observation(raw, (ROOT / path).read_bytes())
            else:
                require(file_sha(ROOT / path) == digest, "Baseline inference/plan source changed: " + path)
    for path, digest in runs[0]["environment"]["launcher_hashes"].items():
        current = file_sha(ROOT / path)
        if path == "scripts/switch_lane.sh":
            require(
                current == APPROVED_SWITCH_LANE_SHA256,
                "Unapproved switch_lane.sh change",
            )
        else:
            require(current == digest, "Launcher changed: " + path)
    directory.mkdir(exist_ok=False)
    manifest = dict(schema="m203.boundary.manifest.v1", created_at=utc_now(),
                    evidence_checkpoint=CHECKPOINT, base_execution=BASE_EXECUTION,
                    execution_sha=execution, sample_id=SAMPLE, target=TARGET,
                    session_id=session, gpu_uuid=gpu,
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    diagnostic_source_hashes={p.name: file_sha(p) for p in
                        sorted((ROOT / "experiments").glob("protocol_b_boundary*.py"))},
                    request_position_one_based=46, prelude_count=45,
                    plan=plan, reference_records={r["lane"]: r["records"][:46] for r in runs},
                    kv_probe=True, layer1_localization=True, early_timeline=True, postwrite_position177=True,
                    interpretation_status="NOT_YET_APPROVED", task_done=False)
    for key in ("PROTOCOL_B_TARGET_REVISION", "PROTOCOL_B_DRAFT_REVISION"):
        field = "target_revision" if "TARGET" in key else "draft_revision"
        require(os.environ[key] == runs[0]["environment"][field], "Model revision changed")
        manifest[field] = os.environ[key]
    write_new(directory / "manifest.json", manifest)
    for lane in ("vanilla", "adaptive"):
        output = directory / lane
        reserve_run(output, "boundary279", lane, lane, session)
        (output / "observer").mkdir()
        write_new(output / "boundary_config.json", dict(
            directory=str(output.resolve()), lane=lane, sample_id=SAMPLE, target=TARGET, kv_probe=True,
            layer1_localization=True,
            early_timeline=lane == "adaptive",
            postwrite_position177=True,
            execution_sha=execution, session_id=session, gpu_uuid=gpu,
            target_revision=manifest["target_revision"], draft_revision=manifest["draft_revision"],
            prompt_token_ids=runs[0]["records"][45]["prompt_token_ids"]))


def verify_instrumentation(directory):
    from experiments.protocol_b_boundary_sources import authenticate
    authenticate(directory, installed=True)


def capture_lane(directory, lane):
    manifest = load_json(directory / "manifest.json")
    config = directory / lane / "boundary_config.json"
    require(Path(os.environ["PROTOCOL_B_BOUNDARY_CONFIG"]).resolve() == config.resolve(),
            "Observer config differs from lane")
    require(os.environ["CUDA_VISIBLE_DEVICES"] == manifest["gpu_uuid"]
            and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == manifest["boot_id"],
            "Physical GPU/session changed before capture")
    live = server_processes(lane)
    require(live["_process_environment"].get("PROTOCOL_B_BOUNDARY_CONFIG") == str(config.resolve()),
            "Server did not inherit observer config")
    settings = load_json(config)
    require(all(settings[key] == manifest[key] for key in
                ("execution_sha", "session_id", "gpu_uuid", "target_revision", "draft_revision"))
            and settings["lane"] == lane and settings["sample_id"] == SAMPLE
            and settings["target"] == TARGET
            and settings.get("kv_probe") is True and manifest.get("kv_probe") is True
            and settings.get("layer1_localization") == manifest.get("layer1_localization")
            and settings.get("early_timeline", False) == (lane == "adaptive" and manifest.get("early_timeline", False))
            and settings.get("postwrite_position177", False) == manifest.get("postwrite_position177", False)
            and settings["prompt_token_ids"] == manifest["reference_records"][lane][-1]["prompt_token_ids"]
            and Path(settings["directory"]).resolve() == config.parent,
            "Observer config identity changed")
    for name, digest in manifest["diagnostic_source_hashes"].items():
        require(file_sha(ROOT / "experiments" / name) == digest, "Diagnostic source changed")
    fingerprint_config = file_sha(config)
    write_new(directory / lane / "observer_binding.json", {
        "config_sha256": fingerprint_config, "config_fingerprint": fingerprint(settings),
        "server_pid": live["pid"], "process_start_ticks": live["process_start_ticks"]})
    verify_instrumentation(directory)
    require(command(["git", "rev-parse", "HEAD"]) == manifest["execution_sha"], "Execution changed")
    if lane == "adaptive":
        previous = load_json(directory / "vanilla" / "run.json")
        require(previous["complete"], "Vanilla must complete first")
    try:
        run = capture(directory / lane, "boundary279", lane, lane, manifest["session_id"],
                      reserved=True, diagnostic_through=SAMPLE)
        require(file_sha(config) == fingerprint_config, "Observer config changed during capture")
        return run
    finally:
        verify_instrumentation(directory)


def same_request(engine, api):
    return isinstance(engine, str) and isinstance(api, str) and bool(
        re.fullmatch(re.escape(api) + r"(?:-[0-9a-f]{8})?", engine))


def validate_boundary(record, events, output):
    """CPU proof of selected row -> sampled ID -> scheduler commit -> HTTP ID."""
    require(not any(e["kind"] == "observer_error" for e in events), "Observer errors retained")
    forwards = [e for e in events if e["kind"] == "forward"
                and same_request(e["request_id"], record["response_id"])]
    require(len(forwards) == 1, "Missing/ambiguous selected boundary forward")
    f = forwards[0]
    rows = f["rows"]
    n = f["draft_count"]
    query = f["active_query"]
    require(f["prompt_token_ids"] == record["prompt_token_ids"], "Prompt differs")
    require(f["query_bounds"] == [0, len(query)], "Query bounds differ")
    require(len({p for p, _ in query}) == len(query), "Duplicate input positions")
    mapped = map_rows(len(record["prompt_token_ids"]), [p for p, _ in query],
                      f["logits_indices"], n, list(range(n)), [n])
    require(rows == mapped, "Recorded row mapping differs")
    targets = [r for r in rows if r["generated_index"] == TARGET]
    require(len(targets) == 1 and f["target_row"] == targets[0], "Missing/ambiguous target row")
    row = targets[0]
    j = row["row"]
    selected = selected_tokens(f["selected"])
    require(0 < len(selected) <= n + 1 and j < len(selected), "Boundary candidate was rejected")
    if f.get("model_runner") == "v2":
        require(n == 0 and f["req_ids"] == [f["request_id"]]
                and len(f["idx_mapping"]) == 1
                and all(type(i) is int and i >= 0 for i in f["idx_mapping"])
                and f["expanded_idx_mapping"] == f["idx_mapping"] and f["cu_num_logits"] == [0, 1],
                "V2 request/row ownership differs")
        require(f["sampled_token_ids"] == [selected] and f["num_sampled"] == [1]
                and f["num_rejected"] == [0]
                and f["processed_capture_point"] == "v2_pre_gumbel_after_top_k_top_p",
                "V2 sampled output/processed path differs")
    base = rows[0]["generated_index"]
    require(base >= 0 and record["token_ids"][base:base + j] == selected[:j],
            "Earlier accepted tokens differ from HTTP conditioning prefix")
    prefix = record["prompt_token_ids"] + record["token_ids"][:TARGET]
    require(all(0 <= p < len(prefix) and prefix[p] == t
                for p, t in query if p <= row["position"]), "Actual model inputs differ from HTTP prefix")
    require(dict(query).get(row["position"]) == prefix[-1], "Missing conditioning token")
    require(f["accepted_drafts"] == len(selected) - 1
            and f["rejected_drafts"] == n - len(selected) + 1, "Draft accounting differs")
    drafts = f.get("draft_token_ids", [])
    require(len(drafts) == n and all(type(t) is int and t >= 0 for t in drafts),
            "Missing/invalid actual draft IDs")
    commits = [e for e in events if e["kind"] == "account_after"
               and e["request_id"] == f["request_id"] and e["stale"] is False
               and e["output_count"] > TARGET]
    require(commits and all(e["committed_ids"][:TARGET] == record["token_ids"][:TARGET]
                            and len(e["committed_ids"]) > TARGET for e in commits),
            "Missing/inconsistent conditioning prefix in scheduler commit")
    # Boundary disagreement is the H3 observation we are seeking, not a reason
    # to discard correctly aligned scores. Prefix disagreement invalidates mapping.
    f = dict(f, sampled_token_id_at_target=selected[j], emitted_token_id_at_target=record["token_ids"][TARGET],
             committed_token_ids_at_target=sorted({e["committed_ids"][TARGET] for e in commits}),
             accepted_draft_token_ids=drafts[:len(selected) - 1],
             rejected_draft_candidate_ids=drafts[len(selected) - 1:],
             sampling_emission_agree=selected[j] == record["token_ids"][TARGET])
    cache = [e for e in events if e["kind"] == "cache_lookup" and e["request_id"] == f["request_id"]]
    require(cache and all(type(e["cached_prefix_tokens"]) is int
                         and 0 <= e["cached_prefix_tokens"] <= len(record["prompt_token_ids"])
                         and e["external_cached_tokens"] == 0 for e in cache), "Missing/invalid cache lookup")
    import numpy as np
    vectors = {}
    for stage in ("raw", "processed"):
        scores = f[stage]
        name = scores["file"]
        require(Path(name).name == name and name.endswith(".npy"), "Unsafe vector path")
        path = output / "observer" / name
        require(file_sha(path) == scores["sha256"], "Score hash mismatch")
        vector = np.load(path, allow_pickle=False)
        require(vector.dtype == np.float32 and vector.ndim == 1 and len(vector) > 128273
                and list(vector.shape) == scores["shape"] and not np.isnan(vector).any()
                and not np.isposinf(vector).any() and np.isfinite(vector).any(), "Invalid score vector")
        require(int(np.argmax(vector)) == scores["argmax"], "Argmax differs")
        require({79126, 128273} <= {c["token_id"] for c in scores["candidates"]},
                "Missing fingerprint candidates")
        vectors[stage] = vector
    return f, vectors


def compare_kv_histories(snapshots, captures, source_events, authenticated, boundaries):
    """Offline gate. Pointer equality is local binding evidence, never cross-lane content."""
    result = dict(status="INVALID_EVIDENCE", historical_kv_state_equivalent=False)
    try:
        for lane in ("vanilla", "adaptive"):
            s, f = snapshots[lane], boundaries[lane]
            require(s["history_schema"] == "m203.boundary.kv.v1"
                    and s["phase"] == "pre_forward" and s["batch_row"] == 0
                    and s["request_id"] == f["request_id"] and s["pid"] == f["pid"]
                    and s["active_query"] == f["active_query"] == [[454, 128367]]
                    and s["position"] == 454 and s["generated_index"] == 279
                    and f["draft_count"] == 0 and f["target_row"]["position"] == 454
                    and s["timestamp"] <= f["timestamp"], "KV stale/ambiguous boundary snapshot")
            if f.get("model_runner") == "v2":
                require(f["idx_mapping"] == [s["request_state_index"]], "KV request-state index differs")
            else:
                require(s["request_state_index"] == 0, "KV V1 request-state index differs")
            auth = [e for e in source_events[lane] if e["auth_id"] == s["auth_id"]
                    and e["pid"] == s["pid"]]
            require(len(auth) == 1 and all(auth[0][k] == v for k, v in authenticated.items())
                    and auth[0]["timestamp"] <= s["timestamp"], "KV source authentication differs")
            require(s["runtime_probe_changes_timing"] is True
                    and s["copies"]["device_wide_synchronize"] is False,
                    "KV missing perturbation record")
            require(s["visibility"] == dict(query_start_loc=[0, 1], seq_len=455,
                                           historical_positions=[0, 454]),
                    "KV incomplete effective visibility")
            binding = s["runtime_binding"]
            require(set(binding["inputs"]) == {"input_ids", "positions"},
                    "KV incomplete input identities")
            names = s["expected_layers"]
            require(names == sorted(set(names)) and len(names) == s["model_layer_count"] > 0
                    and names == binding["expected_layers"]
                    and binding["model_layer_count"] == len(names)
                    and set(s["layers"]) == set(binding["layers"]) == set(names),
                    "KV missing/ambiguous target layer coverage")
            mode = s["dispatched_graph_mode"]
            require(mode in ("FULL", "PIECEWISE")
                    and (lane != "adaptive" or mode == "PIECEWISE"), "KV unsupported dispatch")
            if mode == "FULL":
                found = [c for c in captures[lane] if c["capture_id"] == s["capture_id"]]
                require(len(found) == 1 and found[0]["pid"] == s["pid"]
                        and found[0]["auth_id"] == s["auth_id"]
                        and found[0]["descriptor"] == s["descriptor"]
                        and found[0]["timestamp"] <= s["timestamp"], "KV missing FULL capture")
                validate_full_binding(found[0]["bindings"], binding)
            else:
                require(s["capture_id"] is None and s["descriptor"] is None, "KV wrong PIECEWISE binding")
            for name in names:
                x, b = s["layers"][name], binding["layers"][name]
                require(set(b["metadata"]) == {"query_start_loc", "seq_lens", "block_table", "slot_mapping",
                                                "softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"}
                        and set(b["scales"]) == {"_q_scale", "_k_scale", "_v_scale"},
                        "KV incomplete metadata/scale identities")
                require(x["coverage"] == [0, 454]
                        and x["order"] == ["logical_position", "kv_head", "head_dimension"]
                        and x["heads"] == b["heads"] > 0 and x["head_dim"] == b["head_dim"] > 0
                        and x["elements_per_component"] == 454 * x["heads"] * x["head_dim"]
                        and x["bytes_per_component"] == 2 * x["elements_per_component"]
                        and x["dtype"] == "torch.float16" and x["byte_order"] == "little",
                        "KV incomplete or unsupported content coverage")
                require(x["cache_identity"] == b["cache"] and x["group"] == b["group"]
                        and x["spec_block_size"] == b["spec_block_size"]
                        and x["spec_block_size"] % x["kernel_block_size"] == 0
                        and x["cache_identity"]["shape"][1:] ==
                        [x["heads"], x["kernel_block_size"], 2 * x["head_dim"]]
                        and x["cache_identity"]["device"].startswith("cuda"),
                        "KV runtime cache binding/layout differs")
                _, current, blocks = history_slots([x["physical_blocks"]], x["kernel_block_size"])
                require(current == x["write_slot"] and blocks == x["physical_blocks"]
                        and x["no_write_alias"] is True
                        and max(blocks) < x["cache_identity"]["shape"][0],
                        "KV write-slot safety/physical coverage differs")
                require(set(x["digests"]) == {"K", "V"} and all(
                    re.fullmatch("[0-9a-f]{64}", digest) for digest in x["digests"].values()),
                    "KV missing component fingerprints")
                interpretation = dict(x["interpretation"])
                scales = interpretation.pop("scales")
                require(interpretation == b["interpretation"]
                        and interpretation["backend"] == "TRITON_ATTN"
                        and interpretation["quantization"] == "NONE"
                        and set(scales) == {"_q_scale", "_k_scale", "_v_scale"}
                        and all(v == [1.0] for v in scales.values()),
                        "KV unsupported interpretation/scales")
        v, a = snapshots["vanilla"], snapshots["adaptive"]
        require(v["expected_layers"] == a["expected_layers"], "KV target layer sets differ")
        differences = []
        if v["visibility"] != a["visibility"]:
            differences.append("effective_visibility")
        for name in v["expected_layers"]:
            x, y = v["layers"][name], a["layers"][name]
            require((x["heads"], x["head_dim"]) == (y["heads"], y["head_dim"]),
                    "KV cross-lane architecture differs")
            if x["interpretation"] != y["interpretation"]:
                differences.append(name + ":interpretation")
            for component in ("K", "V"):
                if x["digests"][component] != y["digests"][component]:
                    differences.append(name + ":" + component)
        result.update(status="STATE_DIFFERENT" if differences else "STATE_EQUIVALENT",
                      historical_kv_state_equivalent=not differences, differences=differences,
                      snapshot_ids={lane: s["snapshot_id"] for lane, s in snapshots.items()})
    except (EvidenceError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def first_position_differences(vanilla, adaptive):
    """Numeric logical positions, K/V independently; this alone grants no evidence status."""
    first, at_p = {}, {}
    for component in ("K", "V"):
        left, right = vanilla[component]["position_sha256"], adaptive[component]["position_sha256"]
        require(isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == 454
                and all(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest)
                        for digest in left + right), "Layer-1 missing/malformed position digests")
        first[component] = next((p for p in range(454) if left[p] != right[p]), None)
    earliest = min((p for p in first.values() if p is not None), default=None)
    for component in ("K", "V"):
        at_p[component] = (vanilla[component]["position_sha256"][earliest] !=
                           adaptive[component]["position_sha256"][earliest]) if earliest is not None else None
    return dict(P_K=first["K"], P_V=first["V"], P=earliest,
                K_differs_at_P=at_p["K"], V_differs_at_P=at_p["V"])


def compare_layer1_localization(snapshots, whole):
    """Accept only as an extension of the same fully validated whole-model comparison."""
    result = dict(schema=LOCALIZATION_SCHEMA, target_layer=LAYER1, status="INVALID_EVIDENCE",
                  P_K=None, P_V=None, P=None, K_differs_at_P=None, V_differs_at_P=None)
    try:
        require(whole["status"] == "STATE_DIFFERENT"
                and whole["historical_kv_state_equivalent"] is False
                and whole["snapshot_ids"] == {lane: snapshots[lane]["snapshot_id"]
                                              for lane in ("vanilla", "adaptive")},
                "Layer-1 localization requires valid whole-model STATE_DIFFERENT linkage")
        localizations = {}
        for lane in ("vanilla", "adaptive"):
            layers = snapshots[lane]["layers"]
            require([name for name, layer in layers.items() if "localization" in layer] == [LAYER1],
                    "Layer-1 missing/ambiguous localization scope")
            layer = layers[LAYER1]
            loc = layer["localization"]
            require(loc["schema"] == LOCALIZATION_SCHEMA and loc["target_layer"] == LAYER1
                    and loc["coverage"] == [0, 454] and loc["position_count"] == 454
                    and type(loc["position_count"]) is int
                    and loc["logical_positions"] == list(range(454))
                    and all(type(p) is int for p in loc["logical_positions"])
                    and loc["order"] == layer["order"]
                    == ["logical_position", "kv_head", "head_dimension"],
                    "Layer-1 logical coverage/order differs")
            require(type(loc["heads"]) is int and type(loc["head_dim"]) is int
                    and loc["heads"] == layer["heads"] > 0
                    and loc["head_dim"] == layer["head_dim"] > 0
                    and loc["dtype"] == layer["dtype"] == "torch.float16"
                    and loc["byte_order"] == layer["byte_order"] == "little"
                    and type(loc["bytes_per_position"]) is int
                    and loc["bytes_per_position"] == loc["heads"] * loc["head_dim"] * 2
                    and 454 * loc["bytes_per_position"] == layer["bytes_per_component"],
                    "Layer-1 dimensional/byte accounting differs")
            for component in ("K", "V"):
                require(set(loc[component]) == {"prefix_sha256", "position_sha256"}
                        and loc[component]["prefix_sha256"] == layer["digests"][component]
                        and isinstance(loc[component]["prefix_sha256"], str)
                        and re.fullmatch("[0-9a-f]{64}", loc[component]["prefix_sha256"]),
                        "Layer-1 prefix digest differs from whole-prefix evidence")
            localizations[lane] = loc
        positions = first_position_differences(localizations["vanilla"], localizations["adaptive"])
        for component in ("K", "V"):
            prefix_differs = (localizations["vanilla"][component]["prefix_sha256"] !=
                              localizations["adaptive"][component]["prefix_sha256"])
            require(prefix_differs == (positions["P_" + component] is not None),
                    "Layer-1 prefix/position fingerprints contradict each other")
            require(prefix_differs and LAYER1 + ":" + component in whole["differences"],
                    "Layer-1 whole-prefix K/V difference not reproduced")
        result.update(positions, status="LOCALIZED", snapshot_ids=whole["snapshot_ids"])
    except (EvidenceError, KeyError, TypeError, ValueError, IndexError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def analyze_kv(directory, events, boundaries):
    from experiments.protocol_b_boundary_sources import authenticate
    authenticated = authenticate(directory)
    manifest = load_json(directory / "manifest.json")
    require(authenticated["observer_sha256"] ==
            manifest["diagnostic_source_hashes"]["protocol_b_boundary_observer.py"],
            "KV observer differs from execution manifest")
    snapshots, captures, sources = {}, {}, {}
    for lane in ("vanilla", "adaptive"):
        require(not any(e["kind"] == "observer_error" for e in events[lane]), "KV observer errors")
        selected = [e for e in events[lane] if e["kind"] == "kv_history"
                    and e["request_id"] == boundaries[lane]["request_id"]]
        require(len(selected) == 1, "KV missing/duplicate history snapshot")
        snapshots[lane] = selected[0]
        captures[lane] = [e for e in events[lane] if e["kind"] == "kv_capture"]
        sources[lane] = [e for e in events[lane] if e["kind"] == "kv_sources"]
    whole = compare_kv_histories(snapshots, captures, sources, authenticated, boundaries)
    if manifest.get("layer1_localization") is True:
        require(all(load_json(directory / lane / "boundary_config.json").get("layer1_localization") is True
                    for lane in ("vanilla", "adaptive")), "Layer-1 lane configuration differs")
        whole["layer1_localization"] = compare_layer1_localization(snapshots, whole)
    return whole


def analyze_early(events, record, boundary, historical_kv):
    """Validate the scoped timeline; callers must first pass canonical boundary gates."""
    result = dict(schema=EARLY_SCHEMA, status="INVALID_EVIDENCE", task_done=False,
                  interpretation_status="NOT_YET_APPROVED", temporal_kv_onset="UNKNOWN")
    try:
        require(historical_kv["status"] == "STATE_DIFFERENT"
                and historical_kv["layer1_localization"]["status"] == "LOCALIZED"
                and all(historical_kv["layer1_localization"][k] == 177 for k in ("P_K", "P_V", "P")),
                "Early timeline requires reproduced layer-1 position 177")
        require(record["sample_id"] == SAMPLE and len(record["prompt_token_ids"]) == 176
                and same_request(boundary["request_id"], record["response_id"]), "Early request identity differs")
        require(not any(e["kind"] == "observer_error" for e in events), "Early observer errors")
        cache = [e for e in events if e["kind"] == "cache_lookup" and e["request_id"] == boundary["request_id"]]
        require(len(cache) == 1 and cache[0]["num_computed_tokens"] == 0
                and all(type(cache[0][k]) is int and cache[0][k] >= 0 for k in
                        ("cached_prefix_tokens", "external_cached_tokens")), "Early cache/prefill origin ambiguous")
        prefill_start = cache[0]["cached_prefix_tokens"] + cache[0]["external_cached_tokens"]
        require(prefill_start < 176, "Early missing prefill row")
        early = [e for e in events if e["kind"].startswith("early_")]
        require(early and all(e["early_schema"] == EARLY_SCHEMA and e["lane"] == "adaptive"
                and e["request_id"] == boundary["request_id"] and e["pid"] == boundary["pid"]
                and e["runtime_probe_changes_timing"] is True for e in early), "Early event identity differs")
        require(all(type(e["event_seq"]) is int for e in early), "Early order is not integral")
        early.sort(key=lambda e: e["event_seq"])
        require([e["event_seq"] for e in early] == list(range(1, len(early) + 1)),
                "Early event order missing/duplicated")
        kinds = {"early_schedule", "early_verify", "early_produced", "early_account_before",
                 "early_account_after", "early_complete"}
        require(all(e["kind"] in kinds and type(e["schedule_seq"]) is int
                    and e["schedule_seq"] > 0 for e in early), "Early unknown event/step")

        def unique(kind):
            rows = [e for e in early if e["kind"] == kind]
            require(len({e["schedule_seq"] for e in rows}) == len(rows), "Early duplicate " + kind)
            return {e["schedule_seq"]: e for e in rows}

        schedules, verifies, produced, before, after, complete = [unique(k) for k in
            ("early_schedule", "early_verify", "early_produced", "early_account_before",
             "early_account_after", "early_complete")]
        schedule_order = list(schedules)
        require(schedule_order == sorted(schedule_order)
                and len({s["scheduler_output_id"] for s in schedules.values()}) == len(schedules)
                and all(type(s["scheduler_output_id"]) is int and s["scheduler_output_id"] > 0
                        for s in schedules.values()), "Early schedule order/identity differs")
        require(all(e["schedule_seq"] in schedules and e["scheduler_output_id"]
                    == schedules[e["schedule_seq"]]["scheduler_output_id"] for e in early),
                "Early output object linkage differs")
        require(len(complete) == 1, "Early window not closed uniquely")
        stop, close = next(iter(complete.items()))
        require(stop in schedules and schedules[stop]["event_seq"] < close["event_seq"]
                and close["first_outside_position"] > 180, "Early close linkage differs")
        require(set(verifies) == {s for s in schedules if s < stop}
                and list(verifies) == sorted(verifies)
                and set(before) == set(after) == set(verifies)
                and set(produced) <= set(verifies), "Early incomplete step lifecycle")
        # Replay actual observation order, including scheduling ahead of prior
        # results. A simple per-forward counter comparison would be wrong here.
        counters = dict(num_computed_tokens=prefill_start, num_in_flight_tokens=0,
                        num_output_placeholders=0)
        for e in early:
            kind = e["kind"]
            if kind == "early_schedule":
                count = e["scheduled_tokens"]
                require(type(count) is int and count > 0, "Early scheduled count malformed")
                counters["num_computed_tokens"] += count
                counters["num_in_flight_tokens"] += count
                counters["num_output_placeholders"] += 1 + len(e["scheduled_spec_decode_tokens"])
            elif kind == "early_account_before":
                counters["num_in_flight_tokens"] -= schedules[e["schedule_seq"]]["scheduled_tokens"]
                rejected = len(e["scheduled_spec_decode_tokens"]) - len(e["selected_token_ids"]) + 1
                counters["num_computed_tokens"] -= rejected
                counters["num_output_placeholders"] -= rejected
            elif kind == "early_account_after":
                counters["num_output_placeholders"] -= len(e["output_token_ids"])
            else:
                continue
            require(all(type(e[k]) is int and e[k] == value and value >= 0
                        for k, value in counters.items()), "Early scheduling/rejection counter ledger differs")
        timeline, covers, prior_count, prior_after = [], [], 0, 0
        for index, seq in enumerate(sorted(verifies)):
            s, v, b, a = (table[seq] for table in (schedules, verifies, before, after))
            require(s["identity_source"] == "Scheduler.current_step+same-output-object"
                    and s["event_seq"] < v["event_seq"] < b["event_seq"] < a["event_seq"]
                    and b["event_seq"] > prior_after, "Early malformed lifecycle ordering")
            require(b["stale"] is False and a["stale"] is False, "Early stale output")
            query, drafts = v["active_query"], v["actual_draft_token_ids"]
            slots, invalid, n = (v[k] for k in
                                ("verification_draft_slots", "num_invalid", "effective_num_drafted"))
            require(all(type(x) is int for x in (slots, invalid, n))
                    and 0 <= invalid <= slots <= 4 and n == slots - invalid
                    and len(drafts) == slots, "Early drafted/invalid count mismatch")
            require(all(type(t) is int and t >= 0 for t in drafts[:n])
                    and all(t == -1 for t in drafts[n:]), "Early invalid draft padding")
            require(query and all(len(row) == 2 and all(type(x) is int and x >= 0 for x in row)
                                  for row in query), "Early actual inputs malformed")
            positions = [p for p, _ in query]
            require(v["decoding_path"] == ("speculative" if slots else "ordinary"), "Early path differs")
            if index == 0:
                require(v["actual_position_span"] == [prefill_start, 176]
                        and query == [[175, record["prompt_token_ids"][-1]]]
                        and slots == 0 and s["scheduled_tokens"] == 176 - prefill_start, "Early prefill origin missing")
                expected_rows = map_rows(176, list(range(prefill_start, 176)), [175 - prefill_start], 0, [], [0])
            else:
                require(positions == list(range(175 + prior_count, 176 + prior_count + slots))
                        and v["actual_position_span"] == [positions[0], positions[-1] + 1]
                        and positions[0] <= 180 and positions[-1] >= 176
                        and s["scheduled_tokens"] == slots + 1
                        and [t for _, t in query[1:]] == drafts,
                        "Early actual position/draft mapping differs")
                require(query[0][1] == record["token_ids"][prior_count - 1], "Early base input differs from committed prefix")
                expected_rows = map_rows(176, positions, list(range(slots + 1)), slots,
                                         list(range(slots)), [slots])
            require(v["rows"] == expected_rows, "Early logits/generated mapping differs")
            require(s["placeholder_origin_schedule_seq"] == (None if index == 0 else schedule_order[index - 1]),
                    "Early placeholder producer differs")
            k, batch, budget = (s[x] for x in ("planned_k", "batch_k", "num_spec_tokens_to_schedule"))
            require(all(type(x) is int and 0 <= x <= 4 for x in (batch, budget)) and budget == batch
                    and (k is None and index == 0 and batch == 0 or type(k) is int and 0 <= k <= batch)
                    and s["planner_request_id"] == (None if k is None else s["request_id"])
                    and s["next_placeholders"] == [-1] * (batch if k is None else k), "Early plan/placeholder mismatch")
            if seq in produced:
                p = produced[seq]
                require(v["event_seq"] < p["event_seq"] < b["event_seq"]
                        and type(p["producer_tensor_id"]) is int and p["producer_tensor_id"] > 0
                        and p["producer_row"] == 0
                        and type(p["produced_count"]) is int and p["produced_count"] == budget,
                        "Early produced draft budget/order differs")
            else:
                require(budget == 0, "Early missing draft production")
            origin = s["placeholder_origin_schedule_seq"]
            producer = v["producer_schedule_seq"]
            if slots:
                require(producer == origin and producer in produced and producer < seq
                        and produced[producer]["event_seq"] < v["event_seq"]
                        and produced[producer]["producer_tensor_id"] == v["producer_tensor_id"]
                        and v["previous_request_row"] == produced[producer]["producer_row"]
                        and slots <= len(schedules[origin]["next_placeholders"])
                        and slots <= produced[producer]["produced_count"],
                        "Early planner/producer/verification linkage differs")
            elif producer is not None:
                require(producer == origin and producer in produced
                        and produced[producer]["event_seq"] < v["event_seq"], "Early ordinary producer differs")
            if index and not slots:
                require(not schedules[origin]["next_placeholders"], "Early lost scheduled drafts")
            scheduled = s["scheduled_spec_decode_tokens"]
            require(len(scheduled) == slots, "Early scheduled draft count differs")
            selected = b["selected_token_ids"]
            accepted = len(selected) - 1
            require(selected and all(type(t) is int and t >= 0 for t in selected)
                    and 0 <= accepted <= n and selected[:accepted] == drafts[:accepted],
                    "Early acceptance mismatch")
            for accounting in (b, a):
                ids = accounting["scheduled_spec_decode_tokens"]
                require(len(ids) == slots and all(type(t) is int and (t == -1 or t == drafts[i])
                            for i, t in enumerate(ids)) and accounting["num_invalid"] == invalid
                        and accounting["selected_token_ids"] == selected
                        and accounting["num_drafted"] == n
                        and accounting["num_accepted"] == accepted
                        and accounting["num_rejected"] == n - accepted
                        and accounting["scheduler_rejected_slots"] == slots - accepted
                        and accounting["output_token_ids"] == selected
                        and accounting["output_count"] == len(accounting["committed_ids"]),
                        "Early scheduled IDs/output accounting differs")
                require(all(type(accounting[k]) is int and accounting[k] >= 0 for k in
                            ("output_count", "num_computed_tokens", "num_in_flight_tokens", "num_output_placeholders")),
                        "Early accounting counters malformed")
            require(all(type(t) is int and (t == -1 or t == drafts[i]) for i, t in enumerate(scheduled)),
                    "Early real scheduled IDs differ")
            require(b["output_count"] == prior_count and a["output_count"] == prior_count + len(selected)
                    and b["committed_ids"] == record["token_ids"][:prior_count]
                    and a["committed_ids"] == b["committed_ids"] + selected
                    and a["committed_ids"] == record["token_ids"][:a["output_count"]]
                    and a["num_computed_tokens"] == b["num_computed_tokens"]
                    and a["num_in_flight_tokens"] == b["num_in_flight_tokens"]
                    and a["num_output_placeholders"] == b["num_output_placeholders"] - len(selected),
                    "Early scheduler commit/counter linkage differs")
            covers_177 = 177 in positions
            if covers_177:
                covers.append(seq)
            timeline.append(dict(request_id=s["request_id"], schedule_seq=seq, planner_producer_step=origin,
                draft_producer_step=producer, planned_k=None if origin is None else schedules[origin]["planned_k"],
                producing_plan_k=k, producing_batch_k=batch,
                batch_k=None if origin is None else schedules[origin]["batch_k"], active_query=query, rows=v["rows"],
                scheduled_spec_decode_tokens=scheduled, draft_token_ids=drafts,
                effective_num_drafted=n, num_invalid=invalid, num_accepted=accepted,
                num_rejected=n - accepted, scheduler_rejected_slots=slots - accepted,
                selected_token_ids=selected, committed_token_ids=selected,
                committed_token_ordinals_zero_based=list(range(prior_count, a["output_count"])),
                input_token_ordinals_zero_based=[p - 176 for p in positions],
                output_count=a["output_count"], decoding_path=v["decoding_path"], covers_position_177=covers_177,
                accounting_before=b, accounting_after=a))
            prior_count, prior_after = a["output_count"], a["event_seq"]
        require(covers and prior_count >= 6 and close["first_outside_position"] == 175 + prior_count
                and close["event_seq"] > max(v["event_seq"] for v in verifies.values()),
                "Early position 177/window coverage missing")
        result.update(status="INTERPRETABLE", first_coverage_step=covers[0],
                      first_coverage_path=verifies[covers[0]]["decoding_path"], timeline=timeline)
    except (EvidenceError, KeyError, TypeError, ValueError, IndexError) as exc:
        result.update(error=f"{type(exc).__name__}: {exc}")
    return result


def analyze_postwrite(events, records, boundaries, historical, early):
    """Scoped inference only after the parent's canonical evidence gates pass."""
    result = dict(schema=POSTWRITE_SCHEMA, status="INVALID_EVIDENCE", classification="INVALID_EVIDENCE",
                  task_done=False, historical_kv_state_equivalent=False, same_kv_cache_state_proven=False,
                  interpretation_status="NOT_YET_APPROVED", temporal_kv_onset="UNKNOWN")
    try:
        require(historical["status"] == "STATE_DIFFERENT"
                and historical["layer1_localization"]["status"] == "LOCALIZED"
                and all(historical["layer1_localization"][k] == 177 for k in ("P", "P_K", "P_V"))
                and early["status"] == "INTERPRETABLE" and early["first_coverage_path"] == "speculative",
                "Post-write prerequisite gates failed")
        selected, later_linkage = {}, {}
        for lane in ("vanilla", "adaptive"):
            require(not any(e["kind"] == "observer_error" for e in events[lane]), "Post-write observer errors")
            captures = [e for e in events[lane] if e["kind"] == "postwrite_position177"]
            require(len(captures) == 1, "Post-write missing/duplicate " + lane + " capture")
            e = captures[0]
            histories = [s for s in events[lane] if s["kind"] == "kv_history"
                         and s["request_id"] == boundaries[lane]["request_id"]]
            require(len(histories) == 1, "Post-write missing/duplicate later snapshot")
            later = histories[0]
            require(e["postwrite_schema"] == POSTWRITE_SCHEMA and e["lane"] == lane
                    and e["phase"] == "post_target_forward" and e["logical_position"] == 177
                    and e["target_layer"] == LAYER1 and e["request_id"] == boundaries[lane]["request_id"]
                    and e["pid"] == boundaries[lane]["pid"] and e["auth_id"] == later["auth_id"]
                    and e["batch_row"] == 0 and e["request_state_index"] == later["request_state_index"]
                    and isinstance(e["forward_id"], str) and bool(e["forward_id"])
                    and e["timestamp"] <= later["timestamp"]
                    and e["runtime_probe_changes_timing"] is True and e["blocking_device_to_cpu"] is True
                    and e["device_wide_synchronize"] is False and e["capture_warmup"] is False
                    and same_request(e["request_id"], records[lane]["response_id"]), "Post-write event ownership/phase differs")
            require(later["snapshot_id"] == historical["snapshot_ids"][lane]
                    == historical["layer1_localization"]["snapshot_ids"][lane], "Post-write later snapshot linkage differs")
            expected_positions = [177] if lane == "vanilla" else [177, 178, 179, 180]
            require(e["actual_positions"] == expected_positions
                    and [p for p, _ in e["active_query"]] == expected_positions
                    and e["input_token_id"] == e["active_query"][0][1] == records[lane]["token_ids"][1],
                    "Post-write actual row/input differs")
            binding = e["runtime_binding"]
            require(binding["expected_layers"] == later["runtime_binding"]["expected_layers"]
                    and binding["layers"][LAYER1]["cache"] == e["cache_identity"]
                    == later["layers"][LAYER1]["cache_identity"], "Post-write cache/later binding differs")
            postwrite_ownership(binding, e["cache_peers"])
            require(e["cache_identity"]["device"].startswith("cuda")
                    and e["execution_stream"]["device"] == e["cache_identity"]["device"]
                    and type(e["execution_stream"]["cuda_stream"]) is int
                    and e["execution_stream"]["cuda_stream"] >= 0, "Post-write execution stream differs")
            shape = e["cache_identity"]["shape"]
            location = position177_slot(e["block_table"], e["write_slots"], e["actual_positions"], shape[2], shape[0])
            require(all(e[k] == v for k, v in location.items())
                    and len(e["write_slots"]) == binding["inputs"]["positions"]["shape"][0]
                    and len(e["write_slots"]) == binding["layers"][LAYER1]["metadata"]["slot_mapping"]["shape"][0],
                    "Post-write physical slot/write extent differs")
            h, d = shape[1], shape[3] // 2
            require(shape[3] == 2 * d and e["dtype"] == "torch.float16" and e["shape"] == [h, d]
                    and e["canonical_order"] == ["kv_head", "head_dimension"]
                    and all(type(e[k]) is int and e[k] == v for k, v in
                            dict(heads=h, head_dim=d, element_count=h*d, byte_count=h*d*2, total_value_bytes=h*d*4).items()),
                    "Post-write dtype/shape/byte count differs")
            loc = later["layers"][LAYER1]["localization"]
            require(loc["target_layer"] == LAYER1 and loc["logical_positions"] == list(range(454))
                    and loc["dtype"] == e["dtype"] and loc["heads"] == h and loc["head_dim"] == d
                    and loc["bytes_per_position"] == e["byte_count"], "Post-write later digest representation differs")
            later_linkage[lane] = {}
            for component in ("K", "V"):
                digest = e[component]["sha256"]
                after = loc[component]["position_sha256"][177]
                require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                        and isinstance(after, str) and re.fullmatch(r"[0-9a-f]{64}", after)
                        and loc[component]["prefix_sha256"] == later["layers"][LAYER1]["digests"][component],
                        "Post-write immediate/later digest linkage differs")
                later_linkage[lane][component + "_same_as_later"] = digest == after
            if lane == "vanilla":
                require(e["graph_mode"] == "FULL" and e["hook"] == "postwrite_v2_full"
                        and e["schedule_seq"] is None and e["scheduler_output_id"] is None,
                        "Post-write Vanilla execution differs")
                full = [c for c in events[lane] if c["kind"] == "kv_capture" and c["capture_id"] == e["capture_id"]]
                require(len(full) == 1 and full[0]["descriptor"] == e["descriptor"]
                        and full[0]["pid"] == e["pid"] and full[0]["auth_id"] == e["auth_id"]
                        and full[0]["timestamp"] <= e["timestamp"],
                        "Post-write FULL capture identity differs")
                validate_full_binding(full[0]["bindings"], binding)
            else:
                steps = [t for t in early["timeline"] if t["schedule_seq"] == early["first_coverage_step"]]
                require(len(steps) == 1 and e["graph_mode"] == "PIECEWISE" and e["hook"] == "postwrite_v1"
                        and e["capture_id"] is None and e["descriptor"] is None
                        and e["schedule_seq"] == steps[0]["schedule_seq"]
                        and e["scheduler_output_id"] == steps[0]["accounting_before"]["scheduler_output_id"]
                        and e["active_query"] == steps[0]["active_query"]
                        and steps[0]["decoding_path"] == "speculative", "Post-write wrong first Adaptive verification")
            selected[lane] = e
        require(all(selected["vanilla"][k] == selected["adaptive"][k] for k in
                    ("dtype", "shape", "heads", "head_dim", "byte_count", "canonical_order")),
                "Post-write lane representations differ")
        immediate = {c + "_equal": selected["vanilla"][c]["sha256"] == selected["adaptive"][c]["sha256"]
                     for c in ("K", "V")}
        if all(immediate.values()):
            require(any(not same for lane in later_linkage.values() for same in lane.values()),
                    "Post-write equality contradicts later divergence")
        result.update(status="INTERPRETABLE", immediate=immediate, later_linkage=later_linkage,
            classification="LATER_STATE_CHANGE" if all(immediate.values()) else "IMMEDIATE_STORED_KV_DIVERGENCE",
            forward_ids={lane: e["forward_id"] for lane, e in selected.items()})
    except (EvidenceError, KeyError, TypeError, ValueError, IndexError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def analyze(directory):
    m = load_json(directory / "manifest.json")
    result = dict(schema=SCHEMA, generated_at=utc_now(), sample_id=SAMPLE,
                  interpretation_status="NOT_YET_APPROVED", task_done=False,
                  temporal_kv_onset="UNKNOWN",
                  status="INVALID_EVIDENCE", same_kv_cache_state_proven=False,
                  historical_kv_state_equivalent=False,
                  historical_kv=dict(status="INVALID_EVIDENCE", historical_kv_state_equivalent=False))
    if m.get("postwrite_position177") is True:
        result["postwrite_position177"] = dict(schema=POSTWRITE_SCHEMA, status="INVALID_EVIDENCE",
            classification="INVALID_EVIDENCE", task_done=False, historical_kv_state_equivalent=False,
            same_kv_cache_state_proven=False, interpretation_status="NOT_YET_APPROVED", temporal_kv_onset="UNKNOWN")
    try:
        prompts = m["plan"]["prompts"]
        require(len(prompts) == 54 and len({p["sample_id"] for p in prompts}) == 54
                and prompts[45]["sample_id"] == SAMPLE, "Missing/ambiguous frozen request order")
        runs, records, event_sets = {}, {}, {}
        for lane in ("vanilla", "adaptive"):
            output = directory / lane
            run = load_json(output / "run.json")
            require(run["complete"] and run.get("diagnostic_only") is True
                    and run.get("diagnostic_through") == SAMPLE and len(run["records"]) == 46,
                    "Incomplete/wrong diagnostic replay")
            require(file_sha(output / "responses.jsonl") == run["responses_sha256"], "Journal hash differs")
            journal = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            require(len(journal) == 92, "Incomplete journal")
            for i, (record, prompt) in enumerate(zip(run["records"], m["plan"]["prompts"][:46])):
                require(record["ok"] and record["sample_id"] == prompt["sample_id"]
                        and record["request_sha256"] == prompt["request_sha256"] == fingerprint(prompt["request"]),
                        "Replay order/request differs")
                attempt, response = journal[2*i:2*i+2]
                require(attempt["event"] == "attempt" and response["event"] == "result"
                        and response["record"] == record and response["http_status"] == 200,
                        "Journal disagrees with records")
                payload = dict(prompt["request"], model=attempt["request"]["model"], return_token_ids=True)
                require(attempt["request"] == response["request"] == payload, "Actual request differs")
                from experiments.protocol_b_capture import parse_completion
                parsed = parse_completion(json.loads(response["raw_response_body"]), payload["model"])
                require(all(record[k] == v for k, v in parsed.items()), "Raw response differs")
                require(record["prompt_token_ids"] == m["reference_records"][lane][i]["prompt_token_ids"],
                        "Runtime prompt IDs changed")
            env = run["environment"]
            for field, key in (("git_sha", "execution_sha"), ("session_id", "session_id"),
                               ("gpu_uuid", "gpu_uuid"), ("boot_id", "boot_id"),
                               ("target_revision", "target_revision"), ("draft_revision", "draft_revision")):
                require(env[field] == m[key], "Provenance mismatch: " + field)
            runs[lane], records[lane] = run, run["records"][-1]
            event_sets[lane] = [load_json(p) for p in sorted((output / "observer").glob("*.json"))]
            binding = load_json(output / "observer_binding.json")
            require(file_sha(output / "boundary_config.json") == binding["config_sha256"],
                    "Observer config hash differs")
            require(fingerprint(load_json(output / "boundary_config.json")) == binding["config_fingerprint"],
                    "Observer config fingerprint differs")
            require(load_json(output / "boundary_config.json").get("early_timeline", False)
                    == (lane == "adaptive" and m.get("early_timeline", False)), "Early configuration differs")
            require(load_json(output / "boundary_config.json").get("postwrite_position177", False)
                    == m.get("postwrite_position177", False), "Post-write configuration differs")
            require(all(e["schema"] == "m203.boundary.event.v1" and e["lane"] == lane
                        and e["execution_sha"] == m["execution_sha"]
                        and e["config_fingerprint"] == binding["config_fingerprint"] for e in event_sets[lane]),
                    "Observer identity mismatch")
        require(runs["vanilla"]["environment"] == runs["adaptive"]["environment"], "Lane runtime differs")
        require(runs["vanilla"]["finished_at"] <= runs["adaptive"]["started_at"], "Overlapping lanes")
        v, a = records["vanilla"], records["adaptive"]
        result.update(first_mismatch=first_difference(v["token_ids"], a["token_ids"]),
                      completion_lengths=[len(v["token_ids"]), len(a["token_ids"])],
                      finish_reasons=[v["finish_reason"], a["finish_reason"]],
                      emitted_windows={lane: r["token_ids"][275:284] for lane, r in records.items()},
                      prelude_matches_reference={lane: all(r["token_ids"] == old["token_ids"]
                          for r, old in zip(runs[lane]["records"][:45], m["reference_records"][lane][:45]))
                          for lane in records})
        if not is_fingerprint(v, a):
            result["status"] = "NON_REPRODUCTION"
            if m.get("kv_probe") is True:
                result.update(status="INVALID_EVIDENCE", boundary_status="NON_REPRODUCTION",
                              error="KV boundary fingerprint not reproduced")
        else:
            captures = {lane: validate_boundary(records[lane], event_sets[lane], directory / lane)
                        for lane in records}
            result.update(status="INTERPRETABLE", boundary={lane: f for lane, (f, _) in captures.items()})
            import numpy as np
            result["score_comparison"] = {}
            for stage in ("raw", "processed"):
                x, y = (captures[lane][1][stage] for lane in ("vanilla", "adaptive"))
                require(x.shape == y.shape, "Vocabulary shape changed")
                finite = np.isfinite(x) & np.isfinite(y)
                result["score_comparison"][stage] = dict(equal=bool(np.array_equal(x, y)),
                    argmax=[int(np.argmax(x)), int(np.argmax(y))],
                    finite_max_abs_diff=float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else None,
                    finite_mask_equal=bool(np.array_equal(np.isfinite(x), np.isfinite(y))))
            if m.get("kv_probe") is True:
                require(all(load_json(directory / lane / "boundary_config.json").get("kv_probe") is True
                            for lane in records), "KV lane configuration differs")
                result["historical_kv"] = analyze_kv(directory, event_sets, result["boundary"])
                require(result["historical_kv"]["status"] != "INVALID_EVIDENCE",
                        result["historical_kv"].get("error", "KV invalid evidence"))
                if m.get("postwrite_position177") is True:
                    require(result["historical_kv"]["status"] == "STATE_DIFFERENT",
                            "Post-write requires historical STATE_DIFFERENT")
                result["historical_kv_state_equivalent"] = result["historical_kv"]["historical_kv_state_equivalent"]
                if m.get("layer1_localization") is True:
                    require(result["historical_kv"]["layer1_localization"]["status"] == "LOCALIZED",
                            result["historical_kv"]["layer1_localization"].get("error", "Layer-1 invalid evidence"))
            if m.get("early_timeline") is True:
                require(m.get("kv_probe") is True and m.get("layer1_localization") is True,
                        "Early timeline cannot bypass KV/localization gates")
                result["early_timeline"] = analyze_early(event_sets["adaptive"], a,
                    result["boundary"]["adaptive"], result["historical_kv"])
                require(result["early_timeline"]["status"] == "INTERPRETABLE",
                        result["early_timeline"].get("error", "Early invalid evidence"))
            if m.get("postwrite_position177") is True:
                require(m.get("early_timeline") is True, "Post-write cannot bypass early timeline")
                result["postwrite_position177"] = analyze_postwrite(event_sets, records, result["boundary"],
                    result["historical_kv"], result["early_timeline"])
                require(result["postwrite_position177"]["status"] == "INTERPRETABLE",
                        result["postwrite_position177"].get("error", "Post-write invalid evidence"))
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result.update(status="INVALID_EVIDENCE", error=f"{type(exc).__name__}: {exc}")
        result.pop("boundary", None)
        result.pop("score_comparison", None)
    write_new(directory / "boundary_comparison.json", result)
    return result


def seal(directory):
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    with (directory / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        for path in paths:
            handle.write(f"{file_sha(path)}  {path.relative_to(directory).as_posix()}\n")


def control_packages(text):
    return sorted(line.strip() for line in text.splitlines() if line.strip())


def control_baseline(archive):
    """Read the pinned sealed artifact; replay CPU validators in a disposable copy only."""
    import hashlib
    import io
    import tarfile
    import tempfile
    from pathlib import PurePosixPath
    archive = Path(archive)
    raw = archive.read_bytes()
    require(archive.name == CONTROL_BASELINE_NAME and hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == CONTROL_BASELINE_BLOB,
        "Control baseline archive identity differs")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in member.name,
                    "Unsafe baseline archive path")
            if member.isdir():
                continue
            require(member.isfile() and str(path) not in files, "Unsupported/duplicate archive member")
            files[str(path)] = tar.extractfile(member).read()
    manifests = [p for p in files if p.endswith("/manifest.json") or p == "manifest.json"]
    require(len(manifests) == 1, "Ambiguous baseline manifest")
    prefix = manifests[0][:-len("manifest.json")]
    require(all(p.startswith(prefix) for p in files), "Baseline archive has unrelated roots")
    files = {p[len(prefix):]: value for p, value in files.items()}
    seals = {}
    for line in files["SHA256SUMS"].decode().splitlines():
        digest, name = line.split("  ", 1)
        require(name not in seals and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid baseline seal")
        seals[name] = digest
    require(set(seals) == set(files) - {"SHA256SUMS"}
            and all(hashlib.sha256(files[p]).hexdigest() == digest for p, digest in seals.items()),
            "Baseline seal mismatch")
    with tempfile.TemporaryDirectory(prefix="m203-baseline-read-") as tmp:
        root = Path(tmp)
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        original_report = load_json(root / "boundary_comparison.json")
        (root / "boundary_comparison.json").unlink()  # Derived verification copy, never historical evidence.
        report = analyze(root)
        require(original_report["status"] == report["status"] == "INTERPRETABLE"
                and report["postwrite_position177"]["classification"] == "IMMEDIATE_STORED_KV_DIVERGENCE"
                and report["postwrite_position177"]["immediate"] == dict(K_equal=False, V_equal=False)
                and all(all(link.values()) for link in report["postwrite_position177"]["later_linkage"].values()),
                "Sealed baseline no longer validates")
        manifest = load_json(root / "manifest.json")
        require(manifest["execution_sha"] == CONTROL_BASE_EXECUTION, "Baseline execution differs")
        runs = {lane: load_json(root / lane / "run.json") for lane in ("vanilla", "adaptive")}
        events = {lane: [load_json(p) for p in (root / lane / "observer").glob("*.json")] for lane in runs}
        posts = {lane: next(e for e in events[lane] if e["kind"] == "postwrite_position177") for lane in runs}
        history = next(e for e in events["adaptive"] if e["kind"] == "kv_history")
        step = report["early_timeline"]["first_coverage_step"]
        consumer = next(e for e in events["adaptive"] if e["kind"] == "early_verify" and e["schedule_seq"] == step)
        producer = next(e for e in events["adaptive"] if e["kind"] == "early_schedule"
                        and e["schedule_seq"] == consumer["producer_schedule_seq"])
        require(producer["num_spec_tokens_to_schedule"] == 3, "Baseline producer budget differs")
        with tarfile.open(fileobj=io.BytesIO(files["execution-sources.tar.gz"]), mode="r:gz") as tar:
            scheduler = tar.extractfile("src/server/adaptive_scheduler.py").read()
        require(hashlib.sha256(scheduler).hexdigest() == runs["adaptive"]["environment"]["source_hashes"]
                ["src/server/adaptive_scheduler.py"], "Sealed scheduler source differs")
        return dict(identity=dict(evidence_checkpoint=CONTROL_CHECKPOINT, archive_name=archive.name,
                    archive_sha256=hashlib.sha256(raw).hexdigest(), git_blob=CONTROL_BASELINE_BLOB,
                    execution_sha=CONTROL_BASE_EXECUTION), manifest=manifest, runs=runs, postwrites=posts,
                    prefix={c: history["layers"][LAYER1]["localization"][c]["position_sha256"][:177] for c in ("K", "V")},
                    normal_producer_budget=producer["num_spec_tokens_to_schedule"], scheduler_source=scheduler.decode(),
                    packages=control_packages(files["packages.txt"].decode()))


def control_scheduler_auth(baseline):
    import ast
    from experiments.protocol_b_boundary_sources import StripObserver, validate_scheduler_observation
    raw = ast.unparse(StripObserver().visit(ast.parse(baseline["scheduler_source"]))).encode()
    path = ROOT / "src/server/adaptive_scheduler.py"
    validate_scheduler_observation(raw, path.read_bytes())
    return dict(profile=CONTROL_PROFILE, scheduler_sha256=file_sha(path),
                mutation="one P176 producer SchedulerOutput.num_spec_tokens_to_schedule=0")


def prepare_control(directory, session, archive):
    from experiments.protocol_b import CONFIG_ENV
    require(os.environ.get("M203_P177_PRODUCER_K0") == "1", "Explicit control env gate required")
    require(not command(["git", "status", "--porcelain", "--untracked-files=no"]), "Require reviewed clean checkout")
    baseline = control_baseline(archive)
    source_auth = control_scheduler_auth(baseline)
    original = baseline["runs"]["adaptive"]
    require([i for i, r in enumerate(original["records"][:46])
             if r["prompt_token_ids"] == original["records"][45]["prompt_token_ids"]] == [45],
            "Control prompt identity is ambiguous in the conditioning prefix")
    env = original["environment"]
    require(load_plan() == baseline["manifest"]["plan"], "Control frozen plan differs")
    gpu_selection = control_gpu_selection(env["gpu_uuid"], env["boot_id"])
    for key in CONFIG_ENV:
        if key not in ("ADAPTIVE_TELEMETRY_PATH", "CUDA_VISIBLE_DEVICES"):
            require(os.environ.get(key) == env["config"]["environment"][key], "Control environment differs: " + key)
    for path, digest in env["source_hashes"].items():
        if path not in ("src/server/adaptive_scheduler.py", "experiments/protocol_b.py", "experiments/protocol_b_capture.py"):
            require(file_sha(ROOT / path) == digest, "Frozen inference source differs: " + path)
    require(all(file_sha(ROOT / path) == digest for path, digest in env["launcher_hashes"].items()), "Control launcher changed")
    directory.mkdir(exist_ok=False)
    manifest = dict(schema="m203.p177-k0.manifest.v1", non_canonical=True, control_profile=CONTROL_PROFILE,
        created_at=utc_now(), execution_sha=command(["git", "rev-parse", "HEAD"]), session_id=session,
        baseline_archive=str(Path(archive).resolve()), baseline_identity=baseline["identity"],
        source_auth=source_auth, plan=baseline["manifest"]["plan"], gpu_selection=gpu_selection,
        source_hashes={p: file_sha(ROOT / p) for p in CONTROL_SOURCE_PATHS}, task_done=False)
    write_new(directory / "manifest.json", manifest)
    output = directory / "adaptive"
    reserve_run(output, "boundary279", "adaptive", "adaptive", session)
    (output / "observer").mkdir()
    prompt = manifest["plan"]["prompts"][45]
    write_new(output / "boundary_config.json", dict(directory=str(output.resolve()), lane="adaptive", sample_id=SAMPLE,
        target=TARGET, non_canonical=True, early_timeline=True, postwrite_position177=True, kv_probe=True,
        layer1_localization=False, execution_sha=manifest["execution_sha"],
        prompt_token_ids=original["records"][45]["prompt_token_ids"], p177_k0_control=dict(profile=CONTROL_PROFILE,
            request=prompt["request"], request_sha256=prompt["request_sha256"],
            baseline_sha256=baseline["identity"]["archive_sha256"], normal_producer_budget=baseline["normal_producer_budget"],
            token_prefix=original["records"][45]["token_ids"][:2])))


def capture_control(directory):
    require(os.environ.get("M203_P177_PRODUCER_K0") == "1", "Control env gate required")
    manifest = load_json(directory / "manifest.json")
    require(manifest["control_profile"] == CONTROL_PROFILE and manifest["non_canonical"] is True,
            "Not a control run")
    config = directory / "adaptive/boundary_config.json"
    require(Path(os.environ["PROTOCOL_B_BOUNDARY_CONFIG"]).resolve() == config.resolve(), "Control config path differs")
    require(all(file_sha(ROOT / p) == sha for p, sha in manifest["source_hashes"].items()), "Control execution source drift")
    baseline = control_baseline(manifest["baseline_archive"])
    require(baseline["identity"] == manifest["baseline_identity"]
            and control_scheduler_auth(baseline) == manifest["source_auth"], "Control baseline/source drift")
    env = baseline["runs"]["adaptive"]["environment"]
    require(control_gpu_selection(env["gpu_uuid"], env["boot_id"]) == manifest["gpu_selection"],
            "Control physical session changed")
    live = server_processes("adaptive")
    require(live["_process_environment"].get("M203_P177_PRODUCER_K0") == "1"
            and live["_process_environment"].get("PROTOCOL_B_BOUNDARY_CONFIG") == str(config.resolve()),
            "Server did not inherit control env/config")
    packages = command([live["executable"], "-m", "pip", "freeze"], env=live["_process_environment"])
    require(control_packages(packages) == baseline["packages"], "Control serving packages differ from sealed baseline")
    with (directory / "server-packages.txt").open("x", encoding="utf-8") as handle:
        handle.write(packages + "\n")
    write_new(directory / "adaptive/observer_binding.json", dict(config_sha256=file_sha(config),
        config_fingerprint=fingerprint(load_json(config)), server_pid=live["pid"],
        process_start_ticks=live["process_start_ticks"], control_env="1"))
    verify_instrumentation(directory)
    try:
        return capture(directory / "adaptive", "boundary279", "adaptive", "adaptive", manifest["session_id"],
                       reserved=True, diagnostic_through=SAMPLE, gpu_selection=manifest["gpu_selection"])
    finally:
        verify_instrumentation(directory)


def control_result():
    return dict(status="INVALID_EVIDENCE", classification="INVALID_EVIDENCE", non_canonical=True,
        task_done=False, same_kv_cache_state_proven=False, historical_kv_state_equivalent=False,
        temporal_kv_onset="UNKNOWN", interpretation_status="NOT_YET_APPROVED")


def compare_control(events, record, settings, baseline, authenticated):
    """Separate diagnostic contract: never relax the canonical fingerprint/first-speculative gates."""
    result = control_result()
    try:
        require(settings["non_canonical"] is True and settings["lane"] == "adaptive"
                and settings["p177_k0_control"]["profile"] == CONTROL_PROFILE
                and settings["p177_k0_control"]["baseline_sha256"] == baseline["identity"]["archive_sha256"],
                "Control profile/baseline differs")
        require(record["sample_id"] == SAMPLE and record["ok"] is True
                and record["prompt_token_ids"] == settings["prompt_token_ids"]
                == baseline["runs"]["adaptive"]["records"][45]["prompt_token_ids"]
                and len(record["prompt_token_ids"]) == 176
                and record["request_sha256"] == settings["p177_k0_control"]["request_sha256"]
                == fingerprint(settings["p177_k0_control"]["request"])
                and record["token_ids"][:2] == baseline["runs"]["adaptive"]["records"][45]["token_ids"][:2],
                "Control request/token prefix differs")
        require(not any(e["kind"] == "observer_error" for e in events), "Control observer errors")
        require(all(e["schema"] == "m203.boundary.event.v1" and e["lane"] == "adaptive"
                    and e["execution_sha"] == settings["execution_sha"]
                    and e["config_fingerprint"] == fingerprint(settings) for e in events), "Control event config identity differs")
        def one(kind):
            selected = [e for e in events if e["kind"] == kind]
            require(len(selected) == 1, "Missing/duplicate " + kind)
            return selected[0]
        override, native, producer, drafts, consumer = [one("p177_control_" + k)
                                                       for k in ("override", "native", "producer", "drafts", "consumer")]
        post = one("postwrite_position177")
        selected = (override, native, producer, drafts, consumer, post)
        require(all(e["non_canonical"] is True and e["control_profile"] == CONTROL_PROFILE
                    and e["control_id"] == override["control_id"] and e["pid"] == override["pid"]
                    and e["request_id"] == override["request_id"] for e in selected)
                and same_request(post["request_id"], record["response_id"]), "Control lineage identity differs")
        require(all(isinstance(e["timestamp"], str) and e["timestamp"] for e in selected)
                and [e["timestamp"] for e in selected] == sorted(e["timestamp"] for e in selected),
                "Control runtime event order differs")
        seq = override["producer_seq"]
        require(type(override["override_count"]) is int and override["override_count"] == 1
                and type(override["original_budget"]) is int
                and override["original_budget"] == baseline["normal_producer_budget"] > 0
                and override["effective_budget"] == native["effective_budget"] == 0
                and override["original_plan"]["batch_k"] == override["original_budget"]
                and override["original_plan"]["per_request_k"][post["request_id"]] == override["original_budget"]
                and override["request_sha256"] == record["request_sha256"]
                and override["baseline_sha256"] == baseline["identity"]["archive_sha256"]
                and native["next_placeholders"] == [] and drafts["produced_count"] == 0,
                "Control zero-budget/native draft proof failed")
        require(all(e["producer_seq"] == seq for e in selected[:-1])
                and all(e["scheduler_output_id"] == override["scheduler_output_id"] for e in (native, producer, drafts))
                and consumer["scheduler_output_id"] != override["scheduler_output_id"]
                and consumer["scheduler_output_id"] == post["scheduler_output_id"]
                and consumer["schedule_seq"] == post["schedule_seq"] > seq,
                "Control producer/consumer object linkage differs")
        require(producer["actual_positions"] == [176] and consumer["actual_positions"] == post["actual_positions"] == [177]
                and producer["active_query"] == [[176, record["token_ids"][0]]]
                and consumer["active_query"] == post["active_query"] == [[177, record["token_ids"][1]]]
                and post["input_token_id"] == record["token_ids"][1]
                and consumer["scheduled_spec_tokens"] == [], "Control actual queries differ")
        schedules = [e for e in events if e["kind"] == "early_schedule" and e["request_id"] == post["request_id"]]
        ps = [e for e in schedules if e["schedule_seq"] == seq]
        cs = [e for e in schedules if e["schedule_seq"] == post["schedule_seq"]]
        require(len(ps) == len(cs) == 1 and ps[0]["scheduler_output_id"] == override["scheduler_output_id"]
                and ps[0]["num_spec_tokens_to_schedule"] == 0 and ps[0]["next_placeholders"] == []
                and ps[0]["planned_k"] == ps[0]["batch_k"] == override["original_budget"]
                and cs[0]["placeholder_origin_schedule_seq"] == seq
                and cs[0]["scheduled_spec_decode_tokens"] == [], "Control native early timeline differs")
        verifies = sorted([e for e in events if e["kind"] == "early_verify" and e["request_id"] == post["request_id"]],
                          key=lambda e: e["schedule_seq"])
        covering = [e for e in verifies if any(p == 177 for p, _ in e["active_query"])]
        require(covering and covering[0]["schedule_seq"] == post["schedule_seq"]
                and covering[0]["scheduler_output_id"] == post["scheduler_output_id"]
                and covering[0]["decoding_path"] == "ordinary" and covering[0]["verification_draft_slots"] == 0
                and covering[0]["active_query"] == [[177, record["token_ids"][1]]]
                and len({e["schedule_seq"] for e in covering}) == len(covering), "First ordinary coverage not proven")
        auth = [e for e in events if e["kind"] == "kv_sources" and e["auth_id"] == post["auth_id"] and e["pid"] == post["pid"]]
        require(len(auth) == 1 and all(auth[0][k] == v for k, v in authenticated.items()), "Control source auth differs")
        require(post["postwrite_schema"] == POSTWRITE_SCHEMA and post["target_layer"] == LAYER1
                and post["phase"] == "post_target_forward" and post["logical_position"] == 177
                and post["hook"] == "postwrite_v1" and post["graph_mode"] == "PIECEWISE"
                and post["batch_row"] == post["request_state_index"] == post["selected_row"] == 0
                and post["capture_id"] is None and post["descriptor"] is None
                and post["capture_warmup"] is False and post["device_wide_synchronize"] is False
                and post["runtime_probe_changes_timing"] is True and post["blocking_device_to_cpu"] is True,
                "Control post-write phase/binding differs")
        binding, identity = post["runtime_binding"], post["cache_identity"]
        require(binding["layers"][LAYER1]["cache"] == identity and identity["device"].startswith("cuda")
                and post["execution_stream"]["device"] == identity["device"]
                and type(post["execution_stream"]["cuda_stream"]) is int
                and post["execution_stream"]["cuda_stream"] >= 0, "Control cache/stream differs")
        postwrite_ownership(binding, post["cache_peers"])
        shape = identity["shape"]
        location = position177_slot(post["block_table"], post["write_slots"], [177], shape[2], shape[0])
        require(all(post[k] == v for k, v in location.items())
                and len(post["write_slots"]) == binding["inputs"]["positions"]["shape"][0]
                == binding["layers"][LAYER1]["metadata"]["slot_mapping"]["shape"][0], "Control slot extent differs")
        h, d = shape[1], shape[3] // 2
        require(shape[3] == 2*d and post["dtype"] == "torch.float16" and post["shape"] == [h, d]
                and post["heads"] == h and post["head_dim"] == d
                and post["element_count"] == h*d and post["byte_count"] == h*d*2
                and post["total_value_bytes"] == h*d*4
                and post["canonical_order"] == ["kv_head", "head_dimension"], "Control byte representation differs")
        prefix = post["prefix_0_176"]
        require(prefix["logical_positions"] == list(range(177)) and prefix["dtype"] == post["dtype"]
                and prefix["shape"] == [177, h, d] and prefix["bytes_per_position"] == post["byte_count"]
                and prefix["canonical_order"] == post["canonical_order"]
                and all(prefix[c] == baseline["prefix"][c] for c in ("K", "V")), "Control prefix K/V mismatch")
        prefix_slots = [post["block_table"][p // shape[2]] * shape[2] + p % shape[2] for p in range(177)]
        require(len(set(prefix_slots)) == 177 and all(0 <= s < shape[0]*shape[2] for s in prefix_slots)
                and not set(prefix_slots).intersection(s for s in post["write_slots"] if s >= 0), "Control prefix slot alias")
        for lane, reference in baseline["postwrites"].items():
            require(all(post[k] == reference[k] for k in ("dtype", "shape", "heads", "head_dim", "byte_count", "canonical_order")),
                    "Control/baseline digest representation differs")
            require(all(re.fullmatch(r"[0-9a-f]{64}", reference[c]["sha256"])
                        and re.fullmatch(r"[0-9a-f]{64}", post[c]["sha256"]) for c in ("K", "V")), "Malformed control digest")
        require(any(baseline["postwrites"]["vanilla"][c] != baseline["postwrites"]["adaptive"][c] for c in ("K", "V")),
                "Baseline normal does not differ from Vanilla")
        comparisons = {lane: {c + "_equal": post[c]["sha256"] == reference[c]["sha256"] for c in ("K", "V")}
                       for lane, reference in baseline["postwrites"].items()}
        result.update(status="INTERPRETABLE", override_count=1,
            producer=dict(actual_positions=[176], original_budget=override["original_budget"], effective_budget=0),
            consumer=dict(actual_positions=[177], scheduled_spec_tokens=[]),
            prefix_0_176=dict(K_all_equal=True, V_all_equal=True), control_vs_vanilla=comparisons["vanilla"],
            control_vs_adaptive_normal=comparisons["adaptive"], classification="SPECULATIVE_MULTI_TOKEN_FORWARD_DIVERGENCE"
                if all(comparisons["vanilla"].values()) else "RUNNER_OR_EXECUTION_PATH_STILL_DIFFERENT")
    except (EvidenceError, KeyError, TypeError, ValueError, IndexError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def analyze_control(directory):
    import tarfile
    result = control_result()
    try:
        from experiments.protocol_b import validate_run
        from experiments.protocol_b_capture import parse_completion
        from experiments.protocol_b_boundary_sources import authenticate
        manifest = load_json(directory / "manifest.json")
        require(manifest["control_profile"] == CONTROL_PROFILE and manifest["non_canonical"] is True,
                "Not a control manifest")
        require(set(manifest["source_hashes"]) == set(CONTROL_SOURCE_PATHS), "Incomplete control execution sources")
        baseline = control_baseline(manifest["baseline_archive"])
        require(baseline["identity"] == manifest["baseline_identity"], "Control archive binding differs")
        require(control_packages((directory / "packages.txt").read_text())
                == control_packages((directory / "server-packages.txt").read_text()) == baseline["packages"],
                "Control package identity differs from sealed baseline")
        output = directory / "adaptive"
        run, config = load_json(output / "run.json"), load_json(output / "boundary_config.json")
        require(config["execution_sha"] == run["environment"]["git_sha"] == manifest["execution_sha"], "Control execution differs")
        require(run["session_id"] == manifest["session_id"] if "session_id" in run else
                run["environment"]["session_id"] == manifest["session_id"], "Control session differs")
        expected = {(p["workload"], p["sample_id"]): p for p in manifest["plan"]["prompts"][:46]}
        require(manifest["plan"] == baseline["manifest"]["plan"], "Control plan differs")
        old = baseline["runs"]["adaptive"]["environment"]
        selection = manifest["gpu_selection"]
        validate_gpu_selection(selection, run["environment"]["config"]["environment"]["CUDA_VISIBLE_DEVICES"], old["gpu_uuid"])
        require(selection["boot_id"] == old["boot_id"] == run["environment"]["boot_id"], "Control physical session differs")
        validate_run(run, expected, "boundary279", "adaptive", run["environment"], gpu_selection=selection)
        require(run["diagnostic_only"] is True and run["diagnostic_through"] == SAMPLE
                and [r["sample_id"] for r in run["records"]] == [p["sample_id"] for p in manifest["plan"]["prompts"][:46]],
                "Control replay order/count differs")
        old, new = baseline["runs"]["adaptive"]["environment"], run["environment"]
        require({k: v for k, v in new.items() if k not in ("git_sha", "session_id", "source_hashes", "config")}
                == {k: v for k, v in old.items() if k not in ("git_sha", "session_id", "source_hashes", "config")},
                "Control runtime/package/model identity differs")
        def comparable_config(value):
            import copy
            value = copy.deepcopy(value)
            value["environment"].pop("ADAPTIVE_TELEMETRY_PATH", None)
            value["environment"]["CUDA_VISIBLE_DEVICES"] = old["gpu_uuid"]
            return value
        require(comparable_config(new["config"]) == comparable_config(old["config"]), "Control sampling/cache/graph config differs")
        require(set(new["source_hashes"]) == set(old["source_hashes"])
                and all(digest == old["source_hashes"][p] for p, digest in new["source_hashes"].items()
                        if p not in ("src/server/adaptive_scheduler.py", "experiments/protocol_b.py", "experiments/protocol_b_capture.py"))
                and all(new["source_hashes"][p] == manifest["source_hashes"][p]
                        for p in ("experiments/protocol_b.py", "experiments/protocol_b_capture.py"))
                and new["source_hashes"]["src/server/adaptive_scheduler.py"] == manifest["source_auth"]["scheduler_sha256"],
                "Control frozen source hashes differ")
        require(file_sha(output / "responses.jsonl") == run["responses_sha256"], "Control journal hash differs")
        journal = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
        require(len(journal) == 92, "Incomplete control journal")
        for i, (r, p) in enumerate(zip(run["records"], manifest["plan"]["prompts"][:46])):
            attempt, response = journal[i*2:i*2+2]
            payload = dict(p["request"], model=attempt["request"]["model"], return_token_ids=True)
            require(attempt["event"] == "attempt" and response["event"] == "result"
                    and response["http_status"] == 200 and response["record"] == r
                    and attempt["request"] == response["request"] == payload
                    and all(r[k] == v for k, v in parse_completion(json.loads(response["raw_response_body"]), payload["model"]).items()),
                    "Control raw request/response differs")
        binding = load_json(output / "observer_binding.json")
        require(binding["config_sha256"] == file_sha(output / "boundary_config.json")
                and binding["config_fingerprint"] == fingerprint(config) and binding["control_env"] == "1"
                and binding["server_pid"] == run["live_server"]["pid"]
                and binding["process_start_ticks"] == run["live_server"]["process_start_ticks"], "Control live/config binding differs")
        require(manifest["source_auth"]["profile"] == CONTROL_PROFILE, "Missing scheduler control auth")
        import ast
        import hashlib
        import tarfile
        from experiments.protocol_b_boundary_sources import StripObserver, validate_scheduler_observation
        with tarfile.open(directory / "execution-sources.tar.gz", mode="r:gz") as tar:
            code = {name: tar.extractfile(name).read() for name in manifest["source_hashes"]}
        require(all(hashlib.sha256(code[name]).hexdigest() == digest for name, digest in manifest["source_hashes"].items()),
                "Control execution source bundle differs")
        original_scheduler = ast.unparse(StripObserver().visit(ast.parse(baseline["scheduler_source"]))).encode()
        validate_scheduler_observation(original_scheduler, code["src/server/adaptive_scheduler.py"])
        require(hashlib.sha256(code["src/server/adaptive_scheduler.py"]).hexdigest()
                == manifest["source_auth"]["scheduler_sha256"]
                and hashlib.sha256(code["experiments/protocol_b_boundary_observer.py"]).hexdigest()
                == authenticate(directory)["observer_sha256"], "Control scheduler/helper source auth differs")
        events = [load_json(p) for p in (output / "observer").glob("*.json")]
        result = compare_control(events, run["records"][-1], config, baseline, authenticate(directory))
    except (EvidenceError, OSError, KeyError, TypeError, ValueError, IndexError, tarfile.TarError, EOFError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_new(directory / "p177_k0_control.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "capture", "analyze", "restore", "seal",
                                          "prepare-control", "capture-control", "analyze-control"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--lane", choices=("vanilla", "adaptive"))
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == "prepare-control":
        require(args.session and args.baseline, "Control session/baseline required")
        prepare_control(directory, args.session, args.baseline)
    elif args.action == "capture-control":
        capture_control(directory)
    elif args.action == "analyze-control":
        result = analyze_control(directory)
        print(canonical_json(result))
        if result["status"] != "INTERPRETABLE":
            raise SystemExit(1)
    elif args.action == "prepare":
        require(args.session, "Session required")
        prepare(directory, args.session)
    elif args.action in ("install", "restore"):
        from experiments import protocol_b_boundary_sources as sources
        getattr(sources, args.action)(directory)
    elif args.action == "capture":
        require(args.lane, "Lane required")
        capture_lane(directory, args.lane)
    elif args.action == "analyze":
        result = analyze(directory)
        print(canonical_json(result))
        if result["status"] == "INVALID_EVIDENCE":
            raise SystemExit(1)
    else:
        seal(directory)


if __name__ == "__main__":
    main()
