"""Mocked callback test; DOES NOT verify actual vLLM GPU behavior."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

TARGET = Path(__file__).resolve().parents[1] / "src/server/adaptive_scheduler_trace_diag.py"


class RequestStub:
    def __init__(self, name: str):
        self.request_id = name
        self._output_token_ids = [90, 91]


class BaseSchedulerStub:
    def __init__(self, *args, **kwargs):
        self.requests = {"engine-A": RequestStub("engine-A"), "engine-B": RequestStub("engine-B")}
        self.num_sampled_tokens_per_step = 1
        self.callback_count = 0

    def update_from_output(self, so, mro):
        # Same per-request hook/call order as the relevant vLLM logic.
        for rid, drafts in so.scheduled_spec_decode_tokens.items():
            self.make_spec_decoding_stats(None, len(drafts), 1, None, rid)
            ids = list(mro.sampled_token_ids[mro.req_id_to_index[rid]])
            self.requests[rid]._output_token_ids.extend(ids)
        return {"original_return": True}

    def make_spec_decoding_stats(self, stats, drafted, accepted, invalid, rid):
        self.callback_count += 1
        return "canonical-stats-result"

    def _free_request(self, request, *args, **kwargs):
        return "free-original"


class TestTrace(unittest.TestCase):
    def test_two_requests_same_batch_are_not_cross_joined(self):
        vllm = types.ModuleType("vllm")
        logger_module = types.ModuleType("vllm.logger")
        logger_module.init_logger = lambda _: types.SimpleNamespace(info=lambda *args: None, warning=lambda *args: None, exception=lambda *args: None)
        src = types.ModuleType("src")
        src.__path__ = []
        server = types.ModuleType("src.server")
        server.__path__ = []
        canonical = types.ModuleType("src.server.adaptive_scheduler")
        canonical.AdaptiveAsyncScheduler = BaseSchedulerStub
        env = {
            "vllm": vllm,
            "vllm.logger": logger_module,
            "src": src,
            "src.server": server,
            "src.server.adaptive_scheduler": canonical,
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, env):
            spec = importlib.util.spec_from_file_location("trace_module_under_test", TARGET)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            with patch.dict("os.environ", {"ADAPTIVE_TRACE_PATH": str(Path(temporary) / "trace.jsonl")}):
                scheduler = module.TraceAdaptiveScheduler()
            so = types.SimpleNamespace(scheduled_spec_decode_tokens={"engine-A": [1, 2], "engine-B": [7, 8]})
            mro = types.SimpleNamespace(req_id_to_index={"engine-B": 0, "engine-A": 1}, sampled_token_ids=[[300, 301], [200, 201]])
            result = scheduler.update_from_output(so, mro)
            self.assertEqual(result, {"original_return": True})
            self.assertEqual(scheduler.callback_count, 2)
            rows = [json.loads(line) for line in (Path(temporary) / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            by_id = {row["request_id"]: row for row in rows}
            self.assertEqual(by_id["engine-A"]["emitted_token_ids"], [200, 201])
            self.assertEqual(by_id["engine-B"]["emitted_token_ids"], [300, 301])
            self.assertEqual(by_id["engine-A"]["output_start_index"], 2)
            self.assertEqual(by_id["engine-B"]["output_end_index_exclusive"], 4)
            self.assertEqual(by_id["engine-A"]["spec_step_index"], 1)
            self.assertEqual(by_id["engine-B"]["spec_step_index"], 1)


if __name__ == "__main__":
    unittest.main()
