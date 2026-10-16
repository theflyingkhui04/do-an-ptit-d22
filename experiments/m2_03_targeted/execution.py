"""Explicit terminal stages. No request retries, tunnel or background agent."""
from __future__ import annotations

import argparse
import gzip
import http.client
import importlib.metadata
import importlib.util
import io
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import tarfile
import time
import uuid

from . import sources
from .analysis import compare, response, validate_lane
from .lifecycle import response_receipt, started_process, verify_shutdown
from .common import (CASE, HISTORICAL, HISTORICAL_EXECUTION, LANES, REQUEST_HASH, ROOT,
                     RECONSTRUCTION_TARGET, OBSERVATION_TARGET,
                     Invalid, blob, checkpoint, claim_request, decode, encoded, event,
                     file_sha, frozen_request, git, isolated, load, need, save, sha,
                     verify_checkpoint, write_new, reviewed_policy)

PORTS = {"vanilla": 8001, "adaptive": 8003}
MODELS = {"target": "Qwen/Qwen2.5-7B-Instruct-AWQ", "draft": "Qwen/Qwen2.5-0.5B-Instruct"}
PRODUCTION = ("src/server/adaptive_scheduler.py", "src/server/budget_planner.py",
              "src/proposer/controller.py", "src/proposer/adapter.py")


def verify_execution(execution):
    need(re.fullmatch(r"[0-9a-f]{40}", execution or ""), "full execution SHA required")
    need(git("rev-parse", "HEAD").decode().strip() == execution, "wrong execution checkout")
    need(not git("status", "--porcelain", "--untracked-files=no").strip(), "tracked checkout dirty")
    # All code actually imported/served must be committed with exact bytes.
    paths = list((ROOT / "experiments/m2_03_targeted").glob("*"))
    paths += [ROOT / "scripts/m2_03_gpu_diag.py"] + [ROOT / p for p in PRODUCTION]
    for p in paths:
        if p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            need(blob(rel, execution) == p.read_bytes(), "uncommitted/drifted execution input: " + rel)
    need(blob("experiments/configs/w3_v2/chat_requests.jsonl", execution) == blob("experiments/configs/w3_v2/chat_requests.jsonl"), "execution frozen input drift")


def model_inventory(policy):
    result = {}
    for role, identity in MODELS.items():
        spec = policy["models"][role]
        need(spec["id"] == identity and re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), "model ID/revision unreviewed")
        root = Path(spec["snapshot"]).resolve()
        need(root.is_absolute() and root.name == spec["revision"], "model snapshot revision/path mismatch")
        names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                 and (p.suffix in (".json", ".safetensors", ".model", ".txt", ".jinja") or p.name in ("merges.txt", "vocab.json"))}
        need(names and names == set(spec["file_sha256"]), "incomplete model/tokenizer inventory")
        need(any(n.endswith(".safetensors") for n in names) and "config.json" in names
             and "tokenizer_config.json" in names, "model weights/tokenizer missing")
        for name in sorted(names):
            path = root / name
            # HF snapshot symlinks to immutable blob storage are expected.
            need(file_sha(path) == spec["file_sha256"][name], "model file hash mismatch: " + role + "/" + name)
        result[role] = spec
    return result


def inspect_model(role, snapshot):
    root = Path(snapshot).resolve()
    need(root.is_dir() and re.fullmatch(r"[0-9a-f]{40}", root.name), "snapshot must resolve to immutable revision directory")
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*")) if p.is_file()
             and (p.suffix in (".json", ".safetensors", ".model", ".txt", ".jinja") or p.name in ("merges.txt", "vocab.json"))}
    return {"id": MODELS[role], "revision": root.name, "snapshot": str(root), "file_sha256": files}


def selected_gpu_probe():
    # Properties only: no model, tensor forward, sampling or workload request.
    code = ("import json,torch; p=torch.cuda.get_device_properties(0); "
            "print(json.dumps(dict(count=torch.cuda.device_count(),name=p.name,"
            "uuid=str(getattr(p,'uuid','UNAVAILABLE')),capability=list(torch.cuda.get_device_capability(0)))))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    result = subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, env=env, timeout=60)
    return decode(result.stdout)


def runtime():
    need(platform.system() == "Linux", "GPU stages require Linux Kaggle runtime")
    import torch
    spec = importlib.util.find_spec("vllm")
    need(spec is not None and spec.origin, "vLLM not installed")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
                                         "--format=csv,noheader"], text=True).strip().splitlines()
    selected = [x.strip() for x in inventory[0].split(",")]
    need(selected[0] == "0", "unexpected GPU inventory")
    probe = selected_gpu_probe()
    need(probe["count"] == 1 and probe["name"] == "Tesla T4" and probe["capability"] == [7, 5]
         and probe["uuid"].removeprefix("GPU-").lower() == selected[2].removeprefix("GPU-").lower(),
         "selected CUDA device/UUID not proven equal to physical GPU 0")
    controls = {k: v for k, v in os.environ.items()
                if k.startswith(("VLLM_", "CUDA_", "TORCH_", "PYTORCH_", "NCCL_"))}
    need(not any(k.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
                 for k in controls), "credential-bearing runtime control present; values not recorded")
    return {"python": platform.python_version(), "vllm": importlib.metadata.version("vllm"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": selected[1], "gpu_uuid": selected[2], "driver": selected[3],
            "physical_inventory": inventory, "vllm_root": str(Path(spec.origin).resolve().parent),
            "environment": controls, "selected_cuda_device": probe}


def free_port(port):
    with socket.socket() as s:
        need(s.connect_ex(("127.0.0.1", port)) != 0, "port occupied: " + str(port))


def init_run(root, execution, policy_path, *, r14_single_token_boundary=False,
             r15_attention_dispatch=False, r16_force_attention_3d=False,
             production_attention_observation=False):
    verify_execution(execution)
    root = isolated(root)
    policy_raw = blob(policy_path, execution)
    need((ROOT / policy_path).read_bytes() == policy_raw, "runtime policy not committed/exact")
    policy = decode(policy_raw)
    reviewed_policy(policy)
    need(policy.get('diagnostic_control') == {
        'vanilla_cudagraph_mode': 'PIECEWISE', 'adaptive_cudagraph_mode': 'NONE',
        'vanilla_kv_recompute_output_count': RECONSTRUCTION_TARGET,
        'adaptive_kv_recompute_output_count': RECONSTRUCTION_TARGET}, 'r13 requires unchanged r12 controls')
    frozen_request()
    need(not (r14_single_token_boundary and r16_force_attention_3d),
         'R14 q=1 control and R16 q=3 control are mutually exclusive')
    production = ({'production_attention_observation': True,
                   'production_patch_sha': sources.PRODUCTION_COMMIT}
                  if production_attention_observation else {})
    sources.profile_options(dict(production, r14_single_token_boundary=r14_single_token_boundary,
        r15_attention_dispatch=r15_attention_dispatch, r16_force_attention_3d=r16_force_attention_3d))
    r15_attention_dispatch = r15_attention_dispatch or r16_force_attention_3d
    root.mkdir(parents=True, exist_ok=False)
    r14 = {'r14_single_token_boundary': OBSERVATION_TARGET} if r14_single_token_boundary else {}
    r15 = {'r15_attention_dispatch': True} if r15_attention_dispatch else {}
    r16 = {'r16_force_attention_3d': True} if r16_force_attention_3d else {}
    save(root / "run.json", {"schema": "m203.targeted.run.v1", "case": CASE,
                            "historical_checkpoint_sha": HISTORICAL,
                            "historical_execution_sha": HISTORICAL_EXECUTION,
                            "diagnostic_execution_sha": execution, "policy": policy,
                            "policy_path": policy_path, "policy_sha256": sha(policy_raw),
                            "request_sha256": REQUEST_HASH, "target": OBSERVATION_TARGET,
                            "reconstruction_target": RECONSTRUCTION_TARGET,
                            "observation_target": OBSERVATION_TARGET, **r14, **r15, **r16, **production})
    for lane in LANES:
        directory = root / lane
        (directory / "capture").mkdir(parents=True)
        save(directory / "observer.json", {"lane": lane, "directory": str(directory / "capture"),
                                            "execution_sha": execution,
                                            "reconstruction_target": RECONSTRUCTION_TARGET,
                                            "observation_target": OBSERVATION_TARGET,
                                            **(r14 if lane == 'adaptive' else {}),
                                            **(r15 if lane == 'adaptive' else {}),
                                            **(r16 if lane == 'adaptive' else {}),
                                            **(production if lane == 'adaptive' else {}),
                                            "diagnostic_control": policy.get("diagnostic_control", {})})
    return root


def preflight(root):
    config = load(root / "run.json")
    verify_execution(config["diagnostic_execution_sha"])
    need(sha(blob(config["policy_path"], config["diagnostic_execution_sha"])) == config["policy_sha256"], "policy checksum drift")
    need(decode(blob(config["policy_path"], config["diagnostic_execution_sha"])) == config["policy"], "run policy drift")
    need(config["case"] == CASE and config["request_sha256"] == REQUEST_HASH, "run request drift")
    reviewed_policy(config["policy"])
    frozen_request()
    actual = runtime()
    expected = config["policy"]["runtime"]
    need(set(expected) == {"python", "vllm", "torch", "cuda", "gpu", "driver"}, "runtime policy incomplete")
    need(expected["vllm"] == "0.30.0" and expected["gpu"] == "Tesla T4", "unpinned runtime policy")
    need(all(actual[k] == v for k, v in expected.items()), "runtime differs from approved policy")
    need(actual["environment"] == config["policy"]["runtime_environment"], "unreviewed runtime environment")
    for port in PORTS.values():
        free_port(port)
    models = model_inventory(config["policy"])
    hashes = sources.verify_installed(actual["vllm_root"], **sources.profile_options(config))
    launch = {lane: launch_args(root, lane)[0] for lane in LANES}
    save(root / "preflight.json", {"runtime": actual, "models": models,
                                  "source_base_sha256": hashes,
                                  "launcher_argv": launch,
                                  "execution_sha": config["diagnostic_execution_sha"]})


def install(root):
    need(not (root / 'restore_intent.json').exists(), 'restore started; installation/inference cannot resume this run')
    config = load(root / "run.json")
    verify_execution(config["diagnostic_execution_sha"])
    approved = load(root / "preflight.json")
    actual = runtime()
    need(actual == approved["runtime"], "runtime drift before install")
    target = Path(actual["vllm_root"])
    profile = sources.profile_options(config)
    sources.verify_installed(target, **profile)
    if profile:
        need(approved.get('source_base_sha256') == sources.base_hashes(**profile),
             'production source profile differs from preflight')
    # A durable one-shot installation intent prevents partially installed resume.
    write_new(root / "install.intent", encoded({"state": "UNKNOWN_UNTIL_INSTALL_COMPLETE"}))
    changed = {}
    for name in sources.INSERTIONS:
        path = target / name
        raw = path.read_bytes()
        patched = sources.transform(name, raw)
        write_new(root / "original_sources" / name, raw)
        # Explicit install stage only. Never invoked automatically by serve.
        tmp = path.with_name(path.name + ".m203-targeted-" + uuid.uuid4().hex + ".pending")
        with tmp.open("xb") as f:
            f.write(patched)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        changed[name] = sha(patched)
    sources.verify_installed(target, patched=True, **profile)
    save(root / "installed.json", {"patched_sha256": changed, "runtime": actual})


def restore_source_options(root):
    # Legacy recovery also works without run.json. It still accepts only the
    # pristine/diagnostic pair, so missing metadata can never downgrade production.
    config = root / 'run.json'
    return sources.profile_options(load(config)) if config.exists() else {}


def restore_plan(root):
    base = str(Path(load(root / 'preflight.json')['runtime']['vllm_root']).resolve())
    profile = restore_source_options(root)
    hashes = sources.base_hashes(**profile) if profile else sources.manifest()['sha256']
    return {'vllm_root': base, 'pristine_sha256': hashes,
            'multi_file_atomic': False, 'inference_restart_authorized': False}


def restoration_status(root):
    plan = restore_plan(root)
    profile = sources.installed_state(plan['vllm_root'],
                                      **restore_source_options(root))
    intent = root / 'restore_intent.json'
    completed = root / 'restore_complete.json'
    state = 'NOT_STARTED'
    if profile['mode'] == 'UNKNOWN' or (intent.exists() and load(intent) != plan):
        state = 'UNKNOWN'
    elif completed.exists():
        state = ('RESTORED' if intent.exists() and profile['mode'] == 'PRISTINE'
                 and load(completed) == {'plan_sha256': sha(encoded(plan)), 'sha256': plan['pristine_sha256']}
                 else 'UNKNOWN')
    elif intent.exists():
        state = 'INCOMPLETE'
    elif profile['mode'] == 'MIXED':
        state = 'UNTRACKED_MIXED'
    return {'state': state, 'source_state': profile, 'multi_file_atomic': False,
            'inference_allowed': not intent.exists() and profile['mode'] == 'PATCHED'
                                 and (root / 'installed.json').is_file(),
            'scope': 'source gate only, not GPU-execution authorization'}


def restore(root):
    """NON-ATOMIC multi-file operation. Explicit recovery of known bytes only."""
    for port in PORTS.values():
        free_port(port)
    plan = restore_plan(root)
    base = plan['vllm_root']
    profile = restore_source_options(root)
    replacements = []
    for name, raw in sources.base_sources(**profile).items():
        patched = sources.transform(name, raw)
        path = Path(base) / name
        current = path.read_bytes()
        need(current in (raw, patched), "unknown source during restore; refusing mutation")
        backup = root / "original_sources" / name
        if backup.exists():
            need(backup.read_bytes() == raw, "restore backup mismatch")
        if current != raw:
            need(backup.is_file(), "restore backup missing")
            replacements.append((name, path, raw, current))
    intent = root / 'restore_intent.json'
    if intent.exists():
        need(load(intent) == plan, 'restore intent/target drift; manual review required')
    else:
        save(intent, plan)  # Sticky inference block, durable BEFORE any replacement.
    if (root / 'restore_complete.json').exists():
        need(restoration_status(root)['state'] == 'RESTORED', 'source drift after completed restore')
        return
    attempt = root / 'restore_attempts' / uuid.uuid4().hex
    save(attempt / 'before.json', {'source_state': sources.installed_state(base, **profile), 'multi_file_atomic': False})
    try:
        for i, (name, path, raw, before) in enumerate(replacements):
            need(path.read_bytes() == before, 'source changed during restore; refusing overwrite')
            need((root / 'original_sources' / name).read_bytes() == raw, 'restore backup changed')
            tmp = path.with_name(path.name + '.m203-restore-' + uuid.uuid4().hex + '.pending')
            write_new(tmp, raw)
            os.replace(tmp, path)  # Atomic per-file only; the sequence is NOT atomic.
            save(attempt / f'{i:02d}-applied.json', {'file': name, 'sha256': sha(raw)})
        hashes = sources.verify_installed(base, **profile)
        event(root, 'observer_restored', files_restored=len(replacements), inference_restart_authorized=False)
        save(root / 'restore_complete.json', {'plan_sha256': sha(encoded(plan)), 'sha256': hashes})
    except BaseException as exc:
        try:
            save(attempt / 'incomplete.json', {'state': 'INCOMPLETE', 'error_type': type(exc).__name__, 'multi_file_atomic': False})
        except OSError:
            pass  # Missing completion plus sticky intent remains fail-closed after I/O loss/SIGKILL.
        raise


def inference_sources(root):
    need(not (root / 'restore_intent.json').exists(), 'restore started/incomplete/completed; no inference may resume this run')
    installed = load(root / 'installed.json')
    sources.verify_installed(installed['runtime']['vllm_root'], patched=True,
                             **sources.profile_options(load(root / 'run.json')))
    return installed


def process_identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    tail = raw[raw.rfind(")") + 2:].split()
    need(tail[0] not in ("Z", "X"), "server process exited/zombie; no retroactive stop intent")
    return {"pid": pid, "start_ticks": tail[19], "process_group": int(tail[2]),
            "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def launch_args(root, lane):
    config = load(root / "run.json")
    models = config["policy"]["models"]
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", models["target"]["snapshot"],
            "--host", "127.0.0.1", "--port", str(PORTS[lane]), "--served-model-name", "qwen2.5-7b-" + lane,
            "--generation-config", "vllm", "--quantization", "awq", "--dtype", "half",
            "--gpu-memory-utilization", "0.85", "--max-model-len", "4096", "--max-num-seqs", "32",
            "--no-enable-prefix-caching"]
    control = config["policy"].get("diagnostic_control", {})
    need(set(control) <= {"vanilla_cudagraph_mode", "adaptive_cudagraph_mode",
                          "vanilla_kv_recompute_output_count", "adaptive_kv_recompute_output_count"}
         and control.get("vanilla_cudagraph_mode") in (None, "PIECEWISE")
         and control.get("adaptive_cudagraph_mode") in (None, "NONE")
         and control.get("vanilla_kv_recompute_output_count") in (None, 60)
         and control.get("adaptive_kv_recompute_output_count") in (None, 60),
         "diagnostic control unreviewed")
    cudagraph_mode = control.get(f"{lane}_cudagraph_mode")
    if cudagraph_mode:
        argv += ["--compilation-config", encoded({
            "cudagraph_mode": cudagraph_mode,
        }).decode().strip()]
    if lane == "adaptive":
        spec = {"method": "draft_model", "model": models["draft"]["snapshot"], "num_speculative_tokens": 4,
                "num_speculative_tokens_per_batch_size": [[1, 32, 4]], "use_heterogeneous_vocab": True,
                "draft_sample_method": "greedy"}
        argv += ["--speculative-config", encoded(spec).decode().strip(), "--scheduler-cls",
                 "src.server.adaptive_scheduler.AdaptiveAsyncScheduler", "--per-request-spec-decode-metrics", "summary"]
    env = os.environ.copy()
    env.pop('M203_R14_SINGLE_TOKEN_BOUNDARY', None)
    # Remove stale old diagnostic controls; never consult old /tmp state.
    for k in list(env):
        if k.startswith(("M203_A3_", "ADAPTIVE_")):
            del env[k]
    env.update(VLLM_USE_V2_MODEL_RUNNER="0", CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               M203_TARGETED_CONFIG=str(root / lane / "observer.json"))
    adaptive = config["policy"]["adaptive_env"]
    need(set(adaptive) == {"ADAPTIVE_MAX_K", "ADAPTIVE_MIN_BATCH_K", "ADAPTIVE_PER_REQUEST",
                          "ADAPTIVE_ENABLE_VRAM_GUARD", "ADAPTIVE_CONTROLLER_KWARGS"}, "adaptive policy incomplete")
    need(adaptive["ADAPTIVE_MAX_K"] == "4" and adaptive["ADAPTIVE_MIN_BATCH_K"] == "0"
         and adaptive["ADAPTIVE_PER_REQUEST"] == "1"
         and adaptive["ADAPTIVE_ENABLE_VRAM_GUARD"] in ("0", "1"), "adaptive settings unreviewed")
    if adaptive["ADAPTIVE_CONTROLLER_KWARGS"]:
        need(isinstance(decode(adaptive["ADAPTIVE_CONTROLLER_KWARGS"]), dict), "controller settings unreviewed")
    env.update(adaptive)
    r14 = config.get('r14_single_token_boundary')
    need(r14 is None or type(r14) is int and r14 == OBSERVATION_TARGET, 'unreviewed R14 boundary')
    if r14 is not None and lane == 'adaptive':
        env['M203_R14_SINGLE_TOKEN_BOUNDARY'] = str(r14)
    env["ADAPTIVE_TELEMETRY_PATH"] = str(root / lane / "adaptive_telemetry.jsonl")
    return argv, env


def require_previous_lane(root):
    need(load(root / "vanilla/validated.json") == validate_lane(root, "vanilla"),
         "Vanilla evidence not validated/changed")


def serve(root, lane):
    config = load(root / "run.json")
    verify_execution(config["diagnostic_execution_sha"])
    installed = inference_sources(root)
    need(runtime() == installed["runtime"], "runtime drift before launch")
    model_inventory(config["policy"])
    for port in PORTS.values():
        free_port(port)
    if lane == "adaptive":
        require_previous_lane(root)
    argv, env = launch_args(root, lane)
    need(argv == load(root / "preflight.json")["launcher_argv"][lane], "launcher drift")
    directory = root / lane
    save(directory / "launch.json", {"argv": argv, "execution_sha": config["diagnostic_execution_sha"],
                                    "environment": {k: env[k] for k in env if k.startswith(("ADAPTIVE_", "M203_TARGETED_", "M203_R14_"))
                                                    or k in ("VLLM_USE_V2_MODEL_RUNNER", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}})
    with (directory / "server.log").open("xb") as log:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(child.pid)
        save(directory / "process.json", identity)
        save(directory / "server_started.json", {"schema": "m203.targeted.lifecycle.v1", "process": identity,
             "lane": lane, "execution_sha": config["diagnostic_execution_sha"],
             "launch_sha256": file_sha(directory / "launch.json"), "started_ns": time.monotonic_ns()})
        try:
            code = child.wait()
            observed_ns = time.monotonic_ns()
        except BaseException:
            event(directory, "server_wrapper_interrupted", outcome="UNKNOWN")
            # Cleanup is never an expected-stop proof and must not signal a
            # recycled PID/process group after wait() was interrupted.
            try:
                if process_identity(child.pid) == identity:
                    os.killpg(child.pid, signal.SIGTERM)
            except (OSError, Invalid):
                event(directory, "interrupted_cleanup_identity_unavailable", outcome="UNKNOWN")
            raise
    save(directory / "server_exit.json", {"code": code, "process": identity, "observed_ns": observed_ns})
    # Metadata publication race only: never resend a signal or workload request.
    deadline = time.monotonic() + 3
    while (directory / "stop_intent.json").exists() and not (directory / "stop_sent.json").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    verify_shutdown(directory)


def stop(root, lane):
    directory = root / lane
    verify_execution(load(root / "run.json")["diagnostic_execution_sha"])
    identity, started = started_process(directory)
    need(not (directory / "abort_intent.json").exists(), "aborted run cannot acquire valid stop intent")
    need(not (directory / "server_exit.json").exists(), "server already exited; no stop signal")
    need(process_identity(identity["pid"]) == identity and identity["process_group"] == identity["pid"], "process ownership/PID reuse mismatch")
    response(directory, lane)  # Preserved HTTP body must satisfy the frozen response contract.
    receipt = response_receipt(directory)
    need(started["started_ns"] < receipt["saved_ns"], "response predates server start")
    save(directory / "stop_intent.json", {"process": identity, "signal": 15,
         "started_sha256": file_sha(directory / "server_started.json"), "response_receipt": receipt,
         "intent_ns": time.monotonic_ns()})
    need(process_identity(identity["pid"]) == identity, "PID identity changed before signal")
    begin_ns = time.monotonic_ns()
    os.killpg(identity["pid"], signal.SIGTERM)
    save(directory / "stop_sent.json", {"process": identity, "signal": 15,
         "intent_sha256": file_sha(directory / "stop_intent.json"),
         "send_begin_ns": begin_ns, "send_completed_ns": time.monotonic_ns()})


def abort(root, lane):
    """Explicit owned-process cleanup for UNKNOWN evidence, never a valid stop."""
    verify_execution(load(root / "run.json")["diagnostic_execution_sha"])
    directory = root / lane
    identity, _ = started_process(directory)
    need(not (directory / "server_exit.json").exists(), "server already exited; no abort signal")
    need(process_identity(identity["pid"]) == identity, "process identity mismatch before abort")
    save(directory / "abort_intent.json", {"process": identity, "signal": 15,
         "reason": "EVIDENCE_INCOMPLETE_OR_UNKNOWN", "intent_ns": time.monotonic_ns()})
    need(process_identity(identity["pid"]) == identity, "process identity changed before abort")
    os.killpg(identity["pid"], signal.SIGTERM)


def http_exchange(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def request(root, lane):
    directory = root / lane
    config = load(root / "run.json")
    verify_execution(config["diagnostic_execution_sha"])
    identity, _ = started_process(directory)
    need(process_identity(identity["pid"]) == identity, "server session not alive/owned")
    need(not (directory / "server_exit.json").exists(), "server already exited")
    need(not (directory / "stop_intent.json").exists(), "stop already requested; no workload allowed")
    need(not (directory / "abort_intent.json").exists(), "run aborted; no workload allowed")
    need(not (directory / "request.claim").exists(), "request budget spent; outcome may be UNKNOWN, no resend")
    inference_sources(root)  # Recheck on disk before ANY HTTP, including an already-running server.
    status, raw = http_exchange(PORTS[lane], "GET", "/health")
    need(status == 200, "server not ready; no workload request sent")
    status, raw = http_exchange(PORTS[lane], "GET", "/v1/models")
    need(status == 200 and [x["id"] for x in decode(raw)["data"]] == ["qwen2.5-7b-" + lane], "served model/session mismatch")
    payload = frozen_request()
    payload.update(model="qwen2.5-7b-" + lane, return_token_ids=True)
    save(directory / "request.json", payload)
    claim_request(directory)  # MUST precede the only workload POST, including exceptions.
    try:
        status, raw = http_exchange(PORTS[lane], "POST", "/v1/chat/completions", encoded(payload))
        write_new(directory / "response.json", raw)
        save(directory / "response_meta.json", {"status": status})
        need(status == 200, "HTTP workload error; no retry")
        need(isinstance(decode(raw), dict), "malformed HTTP response; no retry")
        save(directory / "response_saved.json", {"sha256": sha(raw), "saved_ns": time.monotonic_ns()})
    except BaseException:
        event(directory, "request_outcome_unknown_or_failed", budget_spent=True)
        raise


def export_snapshot(snapshot):
    verify_checkpoint(snapshot)
    out = Path(snapshot).with_suffix(".tar.gz")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in sorted(Path(snapshot).rglob("*")):
            need(not p.is_symlink(), "snapshot symlink")
            if p.is_file():
                info = tarfile.TarInfo(p.relative_to(snapshot).as_posix())
                raw = p.read_bytes()
                info.size, info.mtime = len(raw), 0
                tar.addfile(info, io.BytesIO(raw))
    raw = gzip.compress(buf.getvalue(), mtime=0)
    write_new(out, raw)
    save(out.with_suffix(out.suffix + ".sha256.json"), {"file": out.name, "sha256": sha(raw), "size": len(raw)})
    return out


def import_checkpoint(execution, bundle_path, descriptor_path, output):
    """Read Git blob bytes and safely unpack an evidence snapshot on Windows."""
    need(re.fullmatch(r"[0-9a-f]{40}", execution), "full evidence checkpoint SHA required")
    raw = blob(bundle_path, execution)
    descriptor = decode(blob(descriptor_path, execution))
    need(sha(raw) == descriptor["sha256"] and len(raw) == descriptor["size"], "bundle checksum mismatch")
    output = isolated(output)
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        total, names = 0, set()
        for member in tar:
            path = output / member.name
            need(member.isfile() and not Path(member.name).is_absolute()
                 and "\\" not in member.name and ":" not in member.name
                 and path.resolve().is_relative_to(output) and member.name not in names,
                 "unsafe/duplicate archive member")
            names.add(member.name)
            total += member.size
            need(0 <= member.size and total <= 64 * 1024 * 1024, "archive exceeds 64 MiB budget")
            write_new(path, tar.extractfile(member).read())
    verify_checkpoint(output)
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("inspect-runtime")
    p = sub.add_parser("inspect-model")
    p.add_argument("--role", choices=tuple(MODELS), required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p = sub.add_parser("import-checkpoint")
    p.add_argument("--evidence-sha", required=True)
    p.add_argument("--bundle-path", required=True)
    p.add_argument("--descriptor-path", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("init")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--execution-sha", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument('--r14-single-token-boundary', action='store_true',
                   help='diagnostic only: suppress Adaptive current drafts at proven generated index 368')
    p.add_argument('--r15-attention-dispatch', action='store_true',
                   help='diagnostic only: record boundary attention launch metadata')
    p.add_argument('--r16-force-attention-3d', action='store_true',
                   help='diagnostic only: force 3D attention for the Adaptive q=3 index-368 forward')
    p.add_argument('--production-attention-observation', action='store_true',
                   help='observe exact production-patched attention; no diagnostic attention control')
    for stage in ("preflight", "install-observer", "restore-observer", "restoration-status", "serve", "request", "stop", "abort", "validate-lane", "checkpoint"):
        p = sub.add_parser(stage)
        p.add_argument("--run", type=Path, required=True)
        if stage in ("serve", "request", "stop", "abort", "validate-lane"):
            p.add_argument("--lane", choices=LANES, required=True)
        if stage == "checkpoint":
            p.add_argument("--label", required=True)
    for stage in ("verify-checkpoint", "export", "compare"):
        p = sub.add_parser(stage)
        p.add_argument("--snapshot", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.stage == "inspect-runtime":
            rt = runtime()
            result = {"runtime": rt,
                      "inherited_adaptive_controls": {key: os.environ.get(key) for key in (
                          "ADAPTIVE_MAX_K", "ADAPTIVE_MIN_BATCH_K", "ADAPTIVE_PER_REQUEST",
                          "ADAPTIVE_ENABLE_VRAM_GUARD", "ADAPTIVE_CONTROLLER_KWARGS")},
                      "installed_source_sha256": {
                name: file_sha(Path(rt["vllm_root"]) / name) for name in sources.manifest()["sha256"]}}
        elif args.stage == "inspect-model":
            result = inspect_model(args.role, args.snapshot)
        elif args.stage == "import-checkpoint":
            result = {"snapshot": str(import_checkpoint(args.evidence_sha, args.bundle_path, args.descriptor_path, args.output))}
        elif args.stage == "init":
            result = {"run": str(init_run(args.run, args.execution_sha, args.policy,
                                        r14_single_token_boundary=args.r14_single_token_boundary,
                                        r15_attention_dispatch=args.r15_attention_dispatch,
                                        r16_force_attention_3d=args.r16_force_attention_3d,
                                        production_attention_observation=args.production_attention_observation))}
        elif args.stage in ("verify-checkpoint", "export", "compare"):
            checked = verify_checkpoint(args.snapshot)
            if args.stage == "export":
                result = {"bundle": str(export_snapshot(args.snapshot))}
            elif args.stage == "compare":
                result = ({"state": "INVALID_CAPTURE", "reason": "partial snapshot is not a completed diagnostic"}
                          if checked["partial"] else compare(args.snapshot / "data"))
            else:
                result = {"checkpoint_valid": True}
        else:
            root = isolated(args.run)
            if args.stage == "checkpoint":
                result = {"snapshot": str(checkpoint(root, args.label))}
            elif args.stage == 'restoration-status':
                result = restoration_status(root)
            elif args.stage == "validate-lane":
                result = validate_lane(root, args.lane)
                save(root / args.lane / "validated.json", result)
            else:
                fn = {"preflight": preflight, "install-observer": install, "restore-observer": restore,
                      "serve": serve, "request": request, "stop": stop, "abort": abort}[args.stage]
                fn(root, args.lane) if hasattr(args, "lane") else fn(root)
                result = {"stage": args.stage, "status": "COMPLETE"}
        print(encoded(result).decode(), end="")
        return 2 if result.get("state") in ("INVALID_CAPTURE", "INCOMPLETE", "UNKNOWN", "UNTRACKED_MIXED") else 0
    except (Invalid, OSError, KeyError, TypeError, ValueError, AttributeError, IndexError, subprocess.SubprocessError) as exc:
        print(encoded({"status": "STOP", "error": f"{type(exc).__name__}: {exc}", "automatic_retry": False}).decode(), end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
