"""Offline synthetic tests; never start vLLM or access GPU."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m2_03_cache_replication.py"
SPEC = importlib.util.spec_from_file_location("m2replication_tested", str(SCRIPT))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def make_probe_script(cond: str):
    name = ("k4_all_output_coldstart_w3_004" if cond == "on" else
            "k4_no_prefix_coldstart_w3_004")
    return '''from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "experiments/runs/m2_03/diagnostics/''' + name + '''"
PROBES = DIAG / "probes.jsonl"
TRACE = DIAG / "step_trace.jsonl"
TELEMETRY = DIAG / "telemetry.jsonl"
for name, rows in {
"probes.jsonl": [
 {"attempt": 1,"http_status": 200,"response_id":"chatcmpl-abc","token_ids":[10,11],"match_vanilla":True,"first_mismatch_vanilla":None,"speculative_metrics":{"per_step_accepted":[0]}},
 {"attempt": 2,"http_status": 200,"response_id":"chatcmpl-def","token_ids":[10,11],"match_vanilla":True,"first_mismatch_vanilla":None,"speculative_metrics":{"per_step_accepted":[0]}}
],
"all_output_trace.jsonl": [
 {"request_id": "chatcmpl-abc-aa","output_start_index":0,"output_end_index_exclusive":2,"emitted_token_ids":[10,11],"spec_step_index":1},
 {"request_id": "chatcmpl-def-bb","output_start_index":0,"output_end_index_exclusive":2,"emitted_token_ids":[10,11],"spec_step_index":1}
],
"step_trace.jsonl": [
 {"request_id":"chatcmpl-abc-aa","spec_step_index":1,"num_accepted":0},
 {"request_id":"chatcmpl-def-bb","spec_step_index":1,"num_accepted":0}
],
"telemetry.jsonl": [{"request_id":"chatcmpl-abc-aa"},{"request_id":"chatcmpl-def-bb"}],
}.items():
 with (DIAG / name).open("x",encoding="utf-8") as f:
  for row in rows: f.write(json.dumps(row)+"\\n")
print("FAKE_PROBES_COMPLETE")
'''


class TestCacheReplicationOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        (root / "scripts").mkdir()
        (root / "logs").mkdir()
        (root / "logs" / "vllm_lane3.log").write_text("OLD LOG\n")
        (root / "experiments/configs/w3_v2").mkdir(parents=True)
        (root / "experiments/runs/m2_03/m2-03-20261013-r03").mkdir(parents=True)
        (root / "src/server").mkdir(parents=True)
        for name in ("adaptive_scheduler.py", "adaptive_scheduler_trace_diag.py", "adaptive_scheduler_all_output_trace_diag.py"):
            (root / "src/server" / name).write_text("# fake, no GPU\n")
        self.workload = root / "experiments/configs/w3_v2/chat_requests.jsonl"
        self.workload.write_text(json.dumps({"sample_id": "w3_business_v2_004", "request":
                                             {"max_tokens": 768, "seed": 42, "temperature": 0.0}}) + "\n")
        self.vanilla = root / "experiments/runs/m2_03/m2-03-20261013-r03/vanilla.jsonl"
        self.vanilla.write_text(json.dumps({"workload": "W3", "sample_id": "w3_business_v2_004",
                                            "ok": True, "token_ids": [10,11]}) + "\n")
        self.orig = {}
        for attr in ("ROOT", "SCRIPTS", "RUNS", "DIAGNOSTICS", "LOG_FILE", "WORKLOAD", "VANILLA",
                     "EXPECTED_WORKLOAD_SHA", "EXPECTED_VANILLA_SHA", "R1_DIRS"):
            self.orig[attr] = getattr(M, attr)
        M.ROOT = root
        M.SCRIPTS = root / "scripts"
        M.RUNS = root / "experiments/runs/m2_03"
        M.DIAGNOSTICS = M.RUNS / "diagnostics"
        M.R1_DIRS = {"on": M.DIAGNOSTICS / "k4_all_output_coldstart_w3_004",
                     "off": M.DIAGNOSTICS / "k4_no_prefix_coldstart_w3_004"}
        M.LOG_FILE = root / "logs/vllm_lane3.log"
        M.WORKLOAD = self.workload
        M.VANILLA = self.vanilla
        M.EXPECTED_WORKLOAD_SHA = M.digest(self.workload)
        M.EXPECTED_VANILLA_SHA = M.digest(self.vanilla)
        for cond in ("on", "off"):
            launch, switch, probe = M.source_names(cond)
            launch.write_text("#!/bin/bash\n"
                              "vllm serve model \\\n"
                              "  --scheduler-cls src.server.adaptive_scheduler_all_output_trace_diag.TraceAllOutputScheduler \\\n"
                              "  --per-request-spec-decode-metrics detailed" +
                              (" \\\n  --no-enable-prefix-caching" if cond == "off" else "") + "\n")
            switch.write_text("#!/bin/bash\n"
                              f"echo {launch.name}\n"
                              f"echo {launch.name}\n")
            probe.write_text(make_probe_script(cond))
        self.preexisting_contents = {p: p.read_bytes() for cond in ("on", "off") for p in M.source_names(cond)}

    def tearDown(self):
        for attr, val in self.orig.items():
            setattr(M, attr, val)
        self.tmp.cleanup()

    def test_prepare_no_overwrite_and_exact_drifts(self):
        with patch.object(M, "cmd_output", return_value=""):
            with contextlib.redirect_stdout(io.StringIO()):
                M.prepare()
            for cond in ("on", "off"):
                for rep in ("R2", "R3"):
                    M.validate_prepared(rep, cond)
            for path, value in self.preexisting_contents.items():
                self.assertEqual(path.read_bytes(), value)
            self.assertEqual(len([p for p in M.SCRIPTS.iterdir() if "r2_cache_" in p.name]), 6)
            self.assertEqual(len([p for p in M.SCRIPTS.iterdir() if "r3_cache_" in p.name]), 6)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                M.prepare()
            modified = M.generated_names("R2", "on")[0]
            modified.write_text(modified.read_text() + "# drift\n")
            with self.assertRaisesRegex(RuntimeError, "generated file drifted"):
                M.validate_prepared("R2", "on")

    def test_run_cell_and_trace_reconstruction_no_gpu(self):
        with patch.object(M, "cmd_output", return_value=""):
            with contextlib.redirect_stdout(io.StringIO()):
                M.prepare()
        fake_provenance = {"kind": "MOCK_NO_GPU"}
        with patch.object(M, "snapshot_provenance", return_value=fake_provenance), \
             patch.object(M, "verify_live_config", return_value={"effective_prefix_cache_enabled": True}), \
             contextlib.redirect_stdout(io.StringIO()):
            M.run_cell("R2", "on", True)
        out = M.diag_dir("R2", "on")
        state = json.loads((out / "run_state.json").read_text())
        self.assertEqual(state["status"], "VALIDATED")
        result = json.loads((out / "validation.json").read_text())
        self.assertEqual(result["trace_integrity"], "PASS")
        self.assertEqual([v["match_vanilla"] for v in result["outcomes"]], [True,True])
        with patch.object(M, "cmd_output", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                M.run_cell("R2", "on", True)
        report_out = io.StringIO()
        with contextlib.redirect_stdout(report_out):
            M.summarize(False)
        self.assertIn("R2 CACHE_ON", report_out.getvalue())
        self.assertIn("INCOMPLETE", report_out.getvalue())

    def test_requires_consent_and_frozen_hash(self):
        with self.assertRaisesRegex(RuntimeError, "pass --allow-stop-vllm"):
            M.run_cell("R2", "on", False)
        self.workload.write_text(self.workload.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "workload SHA changed"):
            M.ensure_frozen_inputs()


if __name__ == "__main__":
    unittest.main()
