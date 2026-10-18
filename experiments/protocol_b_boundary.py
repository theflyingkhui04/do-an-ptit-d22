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
from experiments.protocol_b_boundary_observer import (map_rows, selected_tokens,
    history_slots, validate_full_binding, LAYER1, LOCALIZATION_SCHEMA)

CHECKPOINT = "4372f8b81998e8a50ee2b87ea6d7c4a9c4b523c5"
BASE_EXECUTION = "a63baa1c42fd75f542f876ab512765149ddd5e20"
EVIDENCE = "experiments/runs/m2_03/protocol_b_20261018T085118Z_2aad4c00"
SAMPLE = "w3_business_v2_004"
TARGET = 279
SCHEMA = "m203.boundary.comparison.v1"


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
            require(file_sha(ROOT / path) == digest, "Baseline inference/plan source changed: " + path)
    for path, digest in runs[0]["environment"]["launcher_hashes"].items():
        require(file_sha(ROOT / path) == digest, "Launcher changed: " + path)
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
                    kv_probe=True, layer1_localization=True,
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


def analyze(directory):
    m = load_json(directory / "manifest.json")
    result = dict(schema=SCHEMA, generated_at=utc_now(), sample_id=SAMPLE,
                  interpretation_status="NOT_YET_APPROVED", task_done=False,
                  status="INVALID_EVIDENCE", same_kv_cache_state_proven=False,
                  historical_kv=dict(status="INVALID_EVIDENCE", historical_kv_state_equivalent=False))
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
                result["historical_kv_state_equivalent"] = result["historical_kv"]["historical_kv_state_equivalent"]
                if m.get("layer1_localization") is True:
                    require(result["historical_kv"]["layer1_localization"]["status"] == "LOCALIZED",
                            result["historical_kv"]["layer1_localization"].get("error", "Layer-1 invalid evidence"))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "capture", "analyze", "restore", "seal"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--lane", choices=("vanilla", "adaptive"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == "prepare":
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
