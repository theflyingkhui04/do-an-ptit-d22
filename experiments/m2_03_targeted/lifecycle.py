"""One offline-verifiable shutdown contract for execution AND validation.

No blanket acceptance of nonzero exits. Even exit 0 requires preserved response,
verified process/start identity, durable stop intent and successful SIGTERM send.
Times use one host's monotonic clock; boot identity prevents cross-boot reuse.
"""
from pathlib import Path

from .common import file_sha, load, need

RESPONSE_FILES = ("request.claim", "request.json", "response.json", "response_meta.json", "response_saved.json")


def response_receipt(directory):
    directory = Path(directory)
    need(load(directory / "request.claim") == {"outcome": "UNKNOWN", "budget_spent": True}, "request budget proof missing")
    need(load(directory / "response_meta.json")["status"] == 200, "response not successful/preserved")
    saved = load(directory / "response_saved.json")
    need(type(saved["saved_ns"]) is int and saved["saved_ns"] > 0, "response completion time missing")
    need(saved["sha256"] == file_sha(directory / "response.json"), "response checksum mismatch before shutdown")
    return {"files_sha256": {name: file_sha(directory / name) for name in RESPONSE_FILES},
            "saved_ns": saved["saved_ns"]}


def started_process(directory):
    directory = Path(directory)
    identity = load(directory / "process.json")
    need(set(identity) == {"pid", "process_group", "start_ticks", "boot_id"}, "incomplete process identity")
    need(type(identity["pid"]) is int and identity["pid"] > 0
         and type(identity["process_group"]) is int and identity["process_group"] == identity["pid"]
         and isinstance(identity["start_ticks"], str) and identity["start_ticks"].isdigit()
         and isinstance(identity["boot_id"], str) and bool(identity["boot_id"]), "invalid owned process identity")
    started = load(directory / "server_started.json")
    launch = load(directory / "launch.json")
    execution = load(directory.parent / "run.json")["diagnostic_execution_sha"]
    need(started["schema"] == "m203.targeted.lifecycle.v1" and started["process"] == identity
         and started["lane"] == directory.name and started["execution_sha"] == launch["execution_sha"] == execution
         and started["launch_sha256"] == file_sha(directory / "launch.json"), "server start/launch identity mismatch")
    need(type(started["started_ns"]) is int and started["started_ns"] > 0, "server start time invalid")
    return identity, started


def verify_shutdown(directory):
    directory = Path(directory)
    need(not (directory / "abort_intent.json").exists(), "aborted run cannot be expected shutdown")
    identity, started = started_process(directory)
    receipt = response_receipt(directory)
    intent = load(directory / "stop_intent.json")
    sent = load(directory / "stop_sent.json")
    exited = load(directory / "server_exit.json")
    need(type(exited["code"]) is int and exited["code"] in (0, -15), "unexpected crash/exit code")
    need(intent["process"] == sent["process"] == exited["process"] == identity,
         "shutdown process identity mismatch")
    need(intent["signal"] == sent["signal"] == 15, "shutdown signal mismatch")
    need(intent["started_sha256"] == file_sha(directory / "server_started.json")
         and intent["response_receipt"] == receipt
         and sent["intent_sha256"] == file_sha(directory / "stop_intent.json"), "stop intent/response proof mismatch")
    times = [started["started_ns"], receipt["saved_ns"], intent["intent_ns"], sent["send_begin_ns"],
             sent["send_completed_ns"], exited["observed_ns"]]
    need(all(type(t) is int and t > 0 for t in times), "lifecycle timestamps invalid")
    start, saved, asked, send_begin, send_end, observed = times
    # Parent wait() may observe exit before stop's acknowledgement file is saved.
    need(start < saved <= asked <= send_begin <= send_end and send_begin <= observed,
         "unexpected lifecycle ordering")
    events = [load(p) for p in (directory / "events").glob("*.json")]
    need(not any(e["kind"] in ("server_wrapper_interrupted", "request_outcome_unknown_or_failed") for e in events),
         "interrupted/unknown lifecycle")
    return {"classification": "EXPECTED_STOP", "exit_code": exited["code"],
            "process": identity, "stop_intent_sha256": file_sha(directory / "stop_intent.json"),
            "response_sha256": receipt["files_sha256"]["response.json"]}
