"""Capture one already-running Protocol-B lane; no launching or automatic retries.

Linux/T4 only for capture. Importing this module does not import torch or vLLM.
Use the runbook in docs/runbooks/m2-03-protocol-b.md, after GPU approval.
"""
from __future__ import annotations

import argparse
import base64
import copy
import http.client
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from experiments.protocol_b import (
    ROOT, CONFIG_ENV, LAUNCHERS, REQUIRED_SOURCES, FROZEN_CONFIG, EvidenceError, canonical_json, file_sha, load_plan, require,
    token_ids, utc_now, write_new, load_json, identifier, hex_digest,
)

LANES = {"vanilla": (8001, "qwen2.5-7b-vanilla"),
         "adaptive": (8003, "qwen2.5-7b-adaptive")}
TARGET = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"


def command(argv, **kwargs):
    return subprocess.check_output(argv, cwd=ROOT, text=True, timeout=30, **kwargs).strip()


def revision(model):
    cache = Path(os.environ["HF_HUB_CACHE"]) / ("models--" + model.replace("/", "--"))
    value = (cache / "refs/main").read_text(encoding="utf-8").strip()
    require((cache / "snapshots" / value / "config.json").is_file(),
            f"Model must already be cached: {model}")
    return value


def resolve_gpu_selector(selector, inventory):
    """Resolve one selector against an authenticated nvidia-smi index/UUID list."""
    require(isinstance(selector, str) and bool(selector) and selector == selector.strip()
            and "," not in selector, "Require exactly one GPU selector")
    require(isinstance(inventory, str), "Invalid nvidia-smi GPU inventory")
    devices = {}
    for line in inventory.splitlines():
        fields = [s.strip() for s in line.split(",")]
        require(len(fields) == 2 and re.fullmatch(r"[0-9]+", fields[0])
                and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", fields[1]),
                "Invalid nvidia-smi GPU inventory")
        index = int(fields[0])
        require(index not in devices and fields[1] not in devices.values(), "Ambiguous GPU inventory")
        devices[index] = fields[1]
    require(bool(devices) and set(devices) == set(range(len(devices))), "Incomplete GPU inventory")
    if re.fullmatch(r"[0-9]+", selector):
        require(int(selector) in devices, "GPU ordinal out of range")
        return devices[int(selector)]
    # This diagnostic deliberately accepts full UUIDs only, never UUID prefixes.
    require(selector in devices.values(), "Unknown or non-full GPU UUID selector")
    return selector


def validate_gpu_selection(selection, selector, expected):
    require(isinstance(selection, dict)
            and selection.get("raw_selector") == selector
            and selection.get("expected_gpu_uuid") == expected
            and resolve_gpu_selector(selector, selection.get("inventory", ""))
                == selection.get("resolved_gpu_uuid") == expected, "Control physical GPU differs")


def control_gpu_selection(expected, boot_id):
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot_id,
            "Control physical session differs")
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        inventory = command(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("Cannot resolve control physical GPU: nvidia-smi failed") from exc
    selection = dict(raw_selector=selector, inventory=inventory, boot_id=boot_id,
                     resolved_gpu_uuid=resolve_gpu_selector(selector, inventory), expected_gpu_uuid=expected)
    validate_gpu_selection(selection, selector, expected)
    return selection


def runtime(plan, session_id, live, *, gpu_selection=None):
    if live.get('_process_environment', {}).get('M203_M005_CONFIG'):
        from experiments.m005_ab import validate_execution_checkout
        validate_execution_checkout(live['_process_environment']['M203_M005_CONFIG'])
    else:
        require(not command(["git", "status", "--porcelain", "--untracked-files=no"]),
                "Execution checkout must have no tracked edits")
    settings = live["environment"]
    require(all(settings[k] == os.environ.get(k) for k in CONFIG_ENV),
            "Live process differs from frozen session environment")
    require(settings.get("HF_HUB_OFFLINE") == "1", "Require offline frozen model cache")
    gpu = settings.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_selection is not None:
        validate_gpu_selection(gpu_selection, gpu, gpu_selection["expected_gpu_uuid"])
        require(control_gpu_selection(gpu_selection["expected_gpu_uuid"], gpu_selection["boot_id"])
                == gpu_selection, "Control GPU selection changed during capture")
        gpu = gpu_selection["resolved_gpu_uuid"]
    else:
        require(gpu.startswith("GPU-"), "Pin the physical GPU UUID")
    info = command(["nvidia-smi", "-i", gpu,
                    "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]).split(",")
    require(len(info) == 3, "Ambiguous GPU selection")
    name, uuid, driver = [item.strip() for item in info]
    require(name == "Tesla T4" and uuid == gpu, "Physical T4 UUID changed")
    # Inspect the serving process's interpreter/environment, not the collector's
    # torch install. This subprocess reads versions only; it serves no requests.
    versions = json.loads(command([live["executable"], "-c",
        "import json,torch,importlib.metadata as m; print(json.dumps(dict("
        "torch=torch.__version__,cuda=torch.version.cuda,vllm=m.version('vllm'))))"],
        env=live.pop("_process_environment")))
    require(versions["torch"] == "2.13.0+cu130" and versions["cuda"] == "13.0",
            "Require canonical torch 2.13.0+cu130 / CUDA 13.0")
    require(versions["vllm"] == "0.30.0", "Require vLLM 0.30.0")
    live["versions"] = versions
    shared_settings = {key: settings[key] for key in CONFIG_ENV}
    shared_settings["other_overrides"] = settings["other_overrides"]
    shared_settings.update(execution_root=str(ROOT), base_pythonpath=os.environ.get("PYTHONPATH", ""))
    return {
        "session_id": session_id,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "git_sha": command(["git", "rev-parse", "HEAD"]),
        "gpu_name": name, "gpu_uuid": uuid, "driver_version": driver,
        "cuda_version": versions["cuda"], "torch_version": versions["torch"],
        "vllm_version": versions["vllm"],
        "target_model": TARGET, "target_revision": settings["PROTOCOL_B_TARGET_REVISION"],
        "draft_model": DRAFT, "draft_revision": settings["PROTOCOL_B_DRAFT_REVISION"],
        "launcher_hashes": {p: file_sha(ROOT / p) for p in LAUNCHERS},
        "source_hashes": {p: file_sha(ROOT / p) for p in REQUIRED_SOURCES},
        "plan_sha256": plan["plan_sha256"],
        "config": {**FROZEN_CONFIG, "environment": shared_settings},
    }


def http_json(url, payload=None):
    data = None if payload is None else canonical_json(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # urllib has no retry adapter; timeout/error is journaled by capture.
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def request_raw(url, payload):
    """Return all available bytes/status even on HTTP error or interrupted read."""
    result = {"http_status": None, "raw_response_body": ""}
    req = urllib.request.Request(url, data=canonical_json(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        try:
            response = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            result["http_status"] = response.code
            try:
                data = response.read()
            except http.client.IncompleteRead as exc:
                data = exc.partial
                result["transport_error"] = f"IncompleteRead: {exc}"
            result["raw_response_base64"] = base64.b64encode(data).decode("ascii")
            try:
                result["raw_response_body"] = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                result["raw_response_body"] = data.decode("utf-8", errors="replace")
                result["decoding_error"] = f"UnicodeDecodeError: {exc}"
    except Exception as exc:
        result["transport_error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_completion(body, model):
    """Minimal archive response contract; never use text as token evidence."""
    require(isinstance(body, dict) and body.get("model") == model, "Response model mismatch")
    choices = body.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "Expected one choice")
    choice = choices[0]
    require(isinstance(choice, dict), "Completion choice must be an object")
    require(choice.get("finish_reason") in ("stop", "length"), "Invalid finish reason")
    ids = token_ids(choice.get("token_ids"))
    prompts = token_ids(body.get("prompt_token_ids"))
    require(isinstance(body.get("usage"), dict), "Missing usage object")
    count = body["usage"].get("completion_tokens")
    require(type(count) is int and count == len(ids), "Reported completion length changed")
    require(isinstance(body.get("id"), str) and bool(body["id"]), "Missing response ID")
    return {"token_ids": ids, "prompt_token_ids": prompts, "completion_tokens": count,
            "finish_reason": choice["finish_reason"], "response_id": body["id"]}


def validate_server_command(args, lane, target_revision=None, draft_revision=None):
    """Check actual process arguments against the existing launcher configuration."""
    port, model = LANES[lane]
    if "serve" in args:
        position = args.index("serve") + 1
    elif "vllm.entrypoints.openai.api_server" in args:
        position = args.index("vllm.entrypoints.openai.api_server") + 1
    else:
        raise EvidenceError("Unsupported server entry point")
    require(position < len(args) and args[position] == TARGET, "Server target changed")
    options = args[position + 1:]
    require(len(options) % 2 == 0, "Unsupported server flag")
    flags = options[::2]
    require(len(set(flags)) == len(flags), "Duplicate server flags")
    actual = dict(zip(flags, options[1::2]))
    require(hex_digest(target_revision, 40) and hex_digest(draft_revision, 40),
            "Explicit frozen revisions required")
    expected = {"--port": str(port), "--served-model-name": model,
                "--host": "0.0.0.0", "--revision": target_revision,
                "--generation-config": "vllm", "--quantization": "awq",
                "--dtype": "half", "--gpu-memory-utilization": "0.85",
                "--max-model-len": "4096", "--max-num-seqs": "32"}
    if lane == "adaptive":
        require("--speculative-config" in actual and "--scheduler-cls" in actual,
                "Missing Adaptive configuration")
        spec = json.loads(actual["--speculative-config"])
        require(spec == {"method": "draft_model", "model": DRAFT, "num_speculative_tokens": 4,
                         "num_speculative_tokens_per_batch_size": [[1, 32, 4]],
                         "use_heterogeneous_vocab": True, "draft_sample_method": "greedy",
                         "revision": draft_revision},
                "Adaptive speculative config changed")
        expected.update({"--speculative-config": actual["--speculative-config"],
                         "--scheduler-cls": "src.server.adaptive_scheduler.AdaptiveAsyncScheduler",
                         "--per-request-spec-decode-metrics": "summary"})
    require(actual == expected, "Unsupported/unapproved live server config")


def server_interpreter(argv0, environment, executable):
    # /proc/PID/exe dereferences venv symlinks. Invoke argv[0] via the live PATH
    # to preserve its venv, while checking that it names the same binary.
    invoked = shutil.which(argv0, path=environment.get("PATH", ""))
    require(invoked and Path(invoked).resolve() == executable.resolve(),
            "Cannot prove serving interpreter")
    return str(Path(invoked).absolute())


def server_processes(lane):
    port = LANES[lane][0]
    owners = command(["fuser", f"{port}/tcp"]).split()
    require(len(owners) == 1 and owners[0].isdigit(), "Require one serving-port owner")
    pid = int(owners[0])
    for other_port in (8001, 8002, 8003):
        if other_port != port:
            probe = subprocess.run(["fuser", f"{other_port}/tcp"], capture_output=True, text=True)
            require(probe.returncode == 1 and not probe.stdout.strip() and not probe.stderr.strip(),
                    "Another lane is active/unknown")
    proc = Path(f"/proc/{pid}")
    args = [a.decode("utf-8") for a in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    require(not any(a.split("=", 1)[0] in ("--api-key", "--token", "--password") for a in args),
            "Refuse credential-bearing process arguments")
    environment = dict(item.decode("utf-8").split("=", 1)
                       for item in (proc / "environ").read_bytes().split(b"\0") if item)
    selected = {key: environment[key] for key in CONFIG_ENV}
    selected["PYTHONPATH"] = environment.get("PYTHONPATH", "")
    require((proc / "cwd").resolve(strict=True) == ROOT, "Server uses another execution checkout")
    selected["other_overrides"] = {
        key: value for key, value in environment.items()
        if key.startswith(("VLLM_", "CUDA_", "TORCH_", "PYTORCH_", "CUBLAS_"))
        and key not in CONFIG_ENV
        and not key.endswith(("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET", "_CREDENTIAL"))}
    require(not any(k.startswith("VLLM_") for k in selected["other_overrides"]),
            "Unsupported vLLM environment override")
    validate_server_command(args, lane, selected["PROTOCOL_B_TARGET_REVISION"],
                            selected["PROTOCOL_B_DRAFT_REVISION"])
    return {"pid": pid, "port": port, "argv": args, "environment": selected,
            "executable": server_interpreter(args[0], environment, proc / "exe"),
            "working_directory": str(ROOT),
            "process_start_ticks": (proc / "stat").read_text().rsplit(")", 1)[1].split()[19],
            "sources": {"argv": f"/proc/{pid}/cmdline", "environment": f"/proc/{pid}/environ",
                        "port": f"fuser {port}/tcp", "versions": f"/proc/{pid}/exe + live environment"},
            "_process_environment": environment}


def validate_live_server(live, lane, env):
    require(isinstance(live, dict) and type(live.get("pid")) is int and live["pid"] > 0,
            "Missing live server binding")
    require(live.get("port") == LANES[lane][0] and isinstance(live.get("argv"), list),
            "Live lane/port mismatch")
    validate_server_command(live["argv"], lane, env["target_revision"], env["draft_revision"])
    settings = live.get("environment", {})
    extras = settings.get("other_overrides")
    require(isinstance(extras, dict)
            and all(isinstance(k, str) and isinstance(v, str) and not k.startswith("VLLM_")
                    for k, v in extras.items()), "Missing/unsupported live environment overrides")
    require(all(settings.get(k) == env["config"]["environment"].get(k)
                for k in (*CONFIG_ENV, "other_overrides")), "Live environment mismatch")
    shared = env["config"]["environment"]
    root, base_path = shared.get("execution_root"), shared.get("base_pythonpath")
    require(isinstance(root, str) and root.startswith("/") and isinstance(base_path, str),
            "Missing execution-root/PYTHONPATH provenance")
    expected_path = (root + (":" + base_path if base_path else "")) if lane == "adaptive" else base_path
    require(live.get("working_directory") == root and settings.get("PYTHONPATH") == expected_path,
            "Live checkout/import path mismatch")
    require(live.get("sources") == {
        "argv": f"/proc/{live['pid']}/cmdline", "environment": f"/proc/{live['pid']}/environ",
        "port": f"fuser {LANES[lane][0]}/tcp",
        "versions": f"/proc/{live['pid']}/exe + live environment"}, "Missing live evidence sources")
    require(live.get("versions") == {"torch": env["torch_version"], "cuda": env["cuda_version"],
                                     "vllm": env["vllm_version"]}, "Live version evidence mismatch")
    require(isinstance(live.get("executable"), str) and live["executable"].startswith("/")
            and isinstance(live.get("process_start_ticks"), str)
            and live["process_start_ticks"].isdigit(), "Missing process lifetime/interpreter")


def reserve_run(output, pair_id, run_id, lane, session_id):
    require(identifier(pair_id) and identifier(run_id) and identifier(session_id) and lane in LANES,
            "Invalid run reservation identity")
    Path(output).mkdir(parents=False, exist_ok=False)
    write_new(Path(output) / "reservation.json", dict(pair_id=pair_id, run_id=run_id,
                                                    lane=lane, session_id=session_id))


def capture(output, pair_id, run_id, lane, session_id, *, reserved=False,
            diagnostic_through=None, gpu_selection=None):
    """Append raw responses, then seal one new run. On failure keep partial evidence."""
    require(gpu_selection is None or (lane == "adaptive" and diagnostic_through == "w3_business_v2_004"),
            "GPU selector resolution is scoped to the Adaptive control diagnostic")
    output = Path(output)
    if not reserved:
        reserve_run(output, pair_id, run_id, lane, session_id)
    require(load_json(output / "reservation.json") == dict(
        pair_id=pair_id, run_id=run_id, lane=lane, session_id=session_id), "Reservation mismatch")
    # Exclusive claim occurs before touching existing artifacts or any server.
    write_new(output / "capture.claim", {"timestamp": utc_now()})
    started = utc_now()
    identity = {"pair_id": pair_id, "run_id": run_id, "lane": lane}
    write_new(output / "started.json", {**identity, "timestamp": started})
    run = {**identity, "started_at": started, "complete": False, "records": []}
    try:
        plan = load_plan()
        prompts = plan["prompts"]
        if diagnostic_through is not None:
            if diagnostic_through == "m0_04_005":
                from experiments.m005_ab import conditioned_prompts
                prompts = conditioned_prompts(plan)
            else:
                require(diagnostic_through == "w3_business_v2_004"
                    and len(prompts) == 54 and prompts[45]["sample_id"] == diagnostic_through,
                    "Only the approved 46-request boundary diagnostic is supported")
                prompts = prompts[:46]
            run.update(diagnostic_only=True, diagnostic_through=diagnostic_through)
        live = server_processes(lane)
        require(not live.get('_process_environment', {}).get('M203_M005_CONFIG')
                or diagnostic_through == 'm0_04_005', 'M005 observer cannot capture another scope')
        if diagnostic_through == "m0_04_005":
            from experiments.m005_ab import validate_capture_scope
            validate_capture_scope(output, session_id, lane, live)
        env = runtime(plan, session_id, live, **({"gpu_selection": gpu_selection} if gpu_selection is not None else {}))
        validate_live_server(live, lane, env)
        run["live_server"] = live
        run["environment"] = env
        write_new(output / "environment.json", env)
        port, model = LANES[lane]
        url = f"http://127.0.0.1:{port}"
        # Preserve model identity and actual server command lines as provenance.
        models = http_json(url + "/v1/models")
        require(sum(m.get("id") == model for m in models["data"]) == 1,
                "Unexpected served model")
        run["server_models"] = models
        with (output / "responses.jsonl").open("x", encoding="utf-8") as journal:
            for prompt in prompts:
                record = {key: prompt[key] for key in (
                    "workload", "sample_id", "dataset_version", "prompt_version", "request_sha256")}
                record.update(identity, timestamp=utc_now(), ok=False)
                payload = copy.deepcopy(prompt["request"])
                payload.update(model=model, return_token_ids=True)
                raw = {"request": payload, "identity": dict(record)}
                journal.write(canonical_json({"event": "attempt", **raw}) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                began = time.monotonic()
                try:
                    raw.update(request_raw(url + "/v1/chat/completions", payload))
                    require(not raw.get("transport_error"), raw.get("transport_error"))
                    require(not raw.get("decoding_error"), raw.get("decoding_error"))
                    require(raw.get("http_status") == 200, "HTTP request failed")
                    body = json.loads(raw["raw_response_body"])
                    raw["response"] = body
                    record.update(parse_completion(body, model), ok=True)
                except BaseException as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    raw["error"] = record["error"]
                    if not isinstance(exc, Exception):
                        raise
                finally:
                    record["latency_seconds"] = time.monotonic() - began
                    record["completion_status"] = "success" if record["ok"] else "failed"
                    journal.write(canonical_json({"event": "result", **raw, "record": record}) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    run["records"].append(record)
                if not record["ok"]:
                    raise EvidenceError("Request failed; retained without retry")
        final_live = server_processes(lane)
        require(runtime(plan, session_id, final_live, **({"gpu_selection": gpu_selection} if gpu_selection is not None else {})) == env, "Runtime changed during run")
        require(final_live == live, "Server restarted/config changed during run")
        run["complete"] = True
    except Exception as exc:
        run["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        run["finished_at"] = utc_now()
        journal_path = output / "responses.jsonl"
        if journal_path.exists():
            run["responses_sha256"] = file_sha(journal_path)
        write_new(output / "run.json", run)
    require(run["complete"], run.get("error", "Incomplete capture"))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--lane", choices=LANES, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reserve-only", action="store_true")
    mode.add_argument("--reserved", action="store_true")
    args = parser.parse_args()
    try:
        if args.reserve_only:
            reserve_run(args.output, args.pair_id, args.run_id, args.lane, args.session_id)
        else:
            capture(args.output, args.pair_id, args.run_id, args.lane, args.session_id,
                    reserved=args.reserved)
    except (EvidenceError, OSError, ValueError) as exc:
        parser.exit(1, f"CAPTURE_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
