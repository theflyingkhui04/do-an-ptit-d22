"""CPU tensor/serialization contracts only; not a CUDA replay simulator."""
import ast
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from experiments import protocol_b_boundary_observer as observer
from experiments.protocol_b import EvidenceError
from experiments.protocol_b_boundary import validate_boundary
from experiments.protocol_b_boundary_sources import transform


class V2ObserverTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "observer").mkdir()
        self.config = dict(directory=str(self.root), lane="vanilla", execution_sha="a" * 40,
                           prompt_token_ids=[8] * 176)
        self.path = self.root / "boundary_config.json"
        self.path.write_text(json.dumps(self.config))
        self.environment = patch.dict(observer.os.environ, PROTOCOL_B_BOUNDARY_CONFIG=str(self.path))
        self.environment.start()
        self.addCleanup(self.environment.stop)
        observer._local.context = None
        observer._local.v2_context = None
        self.rid = "chatcmpl-test-1234abcd"
        self.batch = NS(
            req_ids=[self.rid], num_reqs=1, num_tokens=3, num_draft_tokens=0,
            has_structured_output_reqs=False, is_prefilling_np=np.array([False]),
            idx_mapping_np=np.array([3]), idx_mapping=torch.tensor([3]),
            expanded_idx_mapping=torch.tensor([3]),
            cu_num_logits_np=np.array([0, 1]), cu_num_logits=torch.tensor([0, 1]),
            query_start_loc_np=np.array([0, 3]), query_start_loc=torch.tensor([0, 3]),
            positions=torch.tensor([452, 453, 454, 9999]),
            input_ids=torch.tensor([7, 7, 7, 9999]), logits_indices=torch.tensor([2]))
        states = NS(req_id_to_index={self.rid: 3}, prompt_len=NS(np=np.array([0, 0, 0, 176])),
                    all_token_ids=NS(gpu=torch.full((4, 176), 8, dtype=torch.int64)))
        self.sampler = NS(req_states=states, sampling_states=NS(temperature=NS(np=np.zeros(4))))
        self.logits = torch.zeros((1, 152064), dtype=torch.float16)
        self.logits[0, 79126] = 2
        self.logits[0, 128273] = 1
        self.output = NS(sampled_token_ids=torch.tensor([[79126]]),
                         num_sampled=torch.tensor([1]), num_rejected=torch.tensor([0]))

    def start(self):
        observer.observe("v2_begin", self.sampler, self.batch, self.logits)

    def processed(self):
        observer.observe("v2_processed", self.sampler, self.logits,
                         self.batch.expanded_idx_mapping, self.batch.idx_mapping_np,
                         self.batch.positions[self.batch.logits_indices])

    def finish(self):
        observer.observe("v2_end", self.output)

    def events(self, kind):
        return [e for p in (self.root / "observer").glob("*.json")
                if (e := json.loads(p.read_text()))["kind"] == kind]

    def test_actual_position_nonzero_flat_row_and_request_slot_bind_to_279(self):
        self.start()
        self.processed()
        self.finish()
        event, = self.events("forward")
        self.assertEqual(event["target_row"], dict(row=0, flat_index=2, position=454,
                                                generated_index=279, role="bonus"))
        self.assertEqual(event["request_id"], self.rid)
        self.assertEqual(event["idx_mapping"], [3])
        self.assertEqual(event["logits_indices"], [2])
        self.assertEqual(event["active_query"], [[452, 7], [453, 7], [454, 7]])
        self.assertEqual(event["sampled_token_ids"], [[79126]])
        self.assertEqual((event["num_sampled"], event["num_rejected"]), ([1], [0]))
        self.assertEqual(event["processed_capture_point"], "v2_pre_gumbel_after_top_k_top_p")
        record = dict(response_id="chatcmpl-test", prompt_token_ids=[8] * 176,
                      token_ids=[7] * 279 + [79126] + [6] * 4)
        events = [event, dict(kind="account_after", request_id=self.rid, stale=False,
                             output_count=280, committed_ids=record["token_ids"][:280]),
                  dict(kind="cache_lookup", request_id=self.rid,
                       cached_prefix_tokens=96, external_cached_tokens=0)]
        result, _ = validate_boundary(record, events, self.root)
        self.assertTrue(result["sampling_emission_agree"])
        for changed in (events[1:], events + [event]):
            with self.assertRaises(EvidenceError):
                validate_boundary(record, changed, self.root)
        for key, bad in (("expanded_idx_mapping", [2]), ("num_sampled", [0]),
                         ("sampled_token_ids", [[128273]])):
            changed = copy.deepcopy(events)
            changed[0][key] = bad
            with self.subTest(key=key), self.assertRaises(EvidenceError):
                validate_boundary(record, changed, self.root)

    def test_raw_and_processed_snapshots_own_data_after_inputs_mutate(self):
        self.start()
        _, record = observer._local.v2_context
        # A context must be JSON-serializable: no retained tensors/views.
        json.dumps(observer._local.v2_context)
        self.logits.fill_(0)
        self.logits[0, 128273] = 3
        self.processed()
        self.logits.fill_(99)
        self.batch.input_ids.fill_(42)
        self.batch.positions.fill_(42)
        self.finish()
        event, = self.events("forward")
        raw = np.load(self.root / "observer" / record["raw"]["file"], allow_pickle=False)
        processed = np.load(self.root / "observer" / record["processed"]["file"], allow_pickle=False)
        self.assertEqual((raw[79126], raw[128273]), (2, 1))
        self.assertEqual((processed[79126], processed[128273]), (0, 3))
        self.assertEqual(event["target_row"]["position"], 454)
        self.assertEqual(event["active_query"][-1], [454, 7])
        self.assertEqual({79126, 128273} - {x["token_id"] for x in event["raw"]["candidates"]}, set())

    def test_nonboundary_position_does_not_snapshot(self):
        self.batch.positions[:3] -= 1
        self.start()
        self.processed()
        self.finish()
        self.assertFalse(list((self.root / "observer").iterdir()))

    def test_dummy_sampler_before_request_registration_is_ignored(self):
        self.sampler.req_states.req_id_to_index.clear()
        self.batch.req_ids = None
        self.start()
        self.finish()
        self.assertFalse(list((self.root / "observer").iterdir()))

    def test_unsupported_flashinfer_path_without_processed_hook_fails_closed(self):
        self.start()
        self.finish()
        self.assertFalse(self.events("forward"))
        error, = self.events("observer_error")
        self.assertIn("unsupported sampling path", error["error"])
        self.assertIsNone(observer._local.v2_context)

    def test_observer_failure_preserves_sampling_inputs(self):
        before = self.logits.clone()
        with patch.object(observer, "score", side_effect=OSError("injected snapshot failure")):
            self.start()
        self.assertTrue(self.torch.equal(before, self.logits))
        self.assertEqual(len(self.events("observer_error")), 1)
        self.assertFalse(self.events("forward"))
        self.assertIsNone(observer._local.v2_context)

    def test_ambiguous_or_incorrect_bindings_fail_closed(self):
        cases = [
            lambda: setattr(self.batch, "idx_mapping_np", np.array([2])),
            lambda: setattr(self.batch, "idx_mapping", self.torch.tensor([2])),
            lambda: setattr(self.batch, "expanded_idx_mapping", self.torch.tensor([2])),
            lambda: setattr(self.batch, "logits_indices", self.torch.tensor([2, 2])),
            lambda: setattr(self.batch, "logits_indices", self.torch.tensor([3])),
            lambda: setattr(self.batch, "has_structured_output_reqs", True),
            lambda: setattr(self.batch, "num_draft_tokens", 1),
            lambda: setattr(self.batch, "cu_num_logits_np", np.array([0, 2])),
        ]
        baseline = copy.deepcopy(self.batch)
        for change in cases:
            self.batch = copy.deepcopy(baseline)
            before = len(self.events("observer_error"))
            change()
            self.start()
            self.assertEqual(len(self.events("observer_error")), before + 1)
            self.assertIsNone(observer._local.v2_context)
        self.assertFalse(self.events("forward"))

    def test_processed_binding_or_duplicate_capture_is_invalid(self):
        for bad in ("position", "duplicate"):
            with self.subTest(bad=bad):
                self.start()
                if bad == "position":
                    self.batch.positions[2] = 453
                else:
                    self.processed()
                self.processed()
                self.finish()
                self.assertIsNone(observer._local.v2_context)
                self.batch.positions[2] = 454
        self.assertEqual(len(self.events("observer_error")), 2)
        self.assertFalse(self.events("forward"))

    def test_invalid_sampled_counts_never_produce_forward(self):
        for sampled, rejected in ((0, 0), (1, 1), (2, 0)):
            self.start()
            self.processed()
            self.output.num_sampled[0], self.output.num_rejected[0] = sampled, rejected
            self.finish()
        self.assertEqual(len(self.events("observer_error")), 3)
        self.assertFalse(self.events("forward"))

    def test_v1_and_v2_contexts_coexist_without_metadata_adaptation(self):
        config = dict(self.config, lane="adaptive")
        runner = NS(input_batch=NS(req_ids=[self.rid], num_reqs=1),
                    requests={self.rid: NS(prompt_token_ids=[8] * 176)},
                    uses_mrope=False, enable_prompt_embeds=False,
                    vllm_config=NS(parallel_config=NS(), compilation_config=NS(cudagraph_mode="PIECEWISE")),
                    query_start_loc=NS(np=np.array([0, 3])), positions=self.batch.positions,
                    input_ids=NS(gpu=self.batch.input_ids), use_async_scheduling=True,
                    cache_config=NS(enable_prefix_caching=True))
        observer.begin(config, runner, NS(total_num_scheduled_tokens=3), self.logits, None, None)
        v1_context = observer._local.context
        self.start()
        self.processed()
        self.finish()
        self.assertIs(observer._local.context, v1_context)
        observer.processed("sample_scores", self.logits, NS(all_greedy=True))
        observer.end(config, runner, self.output)
        self.assertEqual({e["lane"] for e in self.events("forward")}, {"vanilla", "adaptive"})
        self.assertIsNone(observer._local.context)


class V2SourceHookTests(unittest.TestCase):
    def test_pinned_hooks_are_in_uncaptured_sampler_and_fallback_before_gumbel(self):
        fixture = Path(__file__).parent / "fixtures/protocol_b_boundary_v2_v030.json.gz"
        name, text = next(iter(json.loads(gzip.decompress(fixture.read_bytes())).items()))
        patched = transform(name, text.encode()).decode()
        tree = ast.parse(patched)
        sampler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sampler")
        methods = {n.name: n for n in sampler.body if isinstance(n, ast.FunctionDef)}
        call = ast.get_source_segment(patched, methods["__call__"])
        fallback = ast.get_source_segment(patched, methods["_sample_random"])
        self.assertLess(call.index("input_ids = input_batch.input_ids["), call.index('"v2_begin"'))
        self.assertLess(call.index('"v2_begin"'), call.index("self.sample("))
        self.assertLess(call.index("sampler_output = SamplerOutput("), call.index('"v2_end"'))
        self.assertLess(call.index('"v2_end"'), call.index("return sampler_output"))
        self.assertLess(fallback.index("apply_top_k_top_p("), fallback.index('"v2_processed"'))
        self.assertLess(fallback.index('"v2_processed"'), fallback.index("gumbel_sample("))
        branch = methods["_sample_random"].body[0]
        self.assertIsInstance(branch, ast.If)
        self.assertNotIn("_pb279_observe", ast.unparse(ast.Module(body=branch.body, type_ignores=[])))
        self.assertIn("_pb279_observe", ast.unparse(ast.Module(body=branch.orelse, type_ignores=[])))


if __name__ == "__main__":
    unittest.main()
