"""CPU observation/serialized-evidence contracts, not a simulation of GPU execution."""
import ast
import copy
import json
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

import test_protocol_b_boundary as boundary_tests
import test_protocol_b_boundary_layer1 as layer1_tests
from experiments.protocol_b import EvidenceError, fingerprint, file_sha
from experiments import protocol_b_boundary as boundary
from experiments import protocol_b_boundary_observer as observer
from experiments import protocol_b_boundary_sources as sources


RID = "chatcmpl-early-12345678"


def fixture(mode="ordinary", real_ids=False, constant_output=False, cached_prefix=96):
    """Exercise real emitter helpers with CPU tensor metadata, then serialize."""
    observer._early.clear()
    config = dict(early_timeline=True, lane="adaptive", sample_id=boundary.SAMPLE,
                  prompt_token_ids=[8] * 176, execution_sha="a" * 40)
    request = NS(request_id=RID, prompt_token_ids=config["prompt_token_ids"],
                 spec_token_ids=[], _output_token_ids=[], num_computed_tokens=cached_prefix,
                 num_in_flight_tokens=0, num_output_placeholders=0)
    scheduler = NS(current_step=0, requests={RID: request})
    runner = NS(input_batch=NS(req_ids=[RID], num_reqs=1, prev_req_id_to_index={RID: 0}),
                _draft_token_ids=None, requests={RID: request}, uses_mrope=False,
                enable_prompt_embeds=False, vllm_config=NS(parallel_config=NS()))
    # budget for future drafts, selected output for this attempt, new proposal.
    if mode == "rewrite":
        steps = [(3, [101], [11, 12, 13]), (0, [102], [])]
        steps += [(0, [t], []) for t in (103, 104, 105, 106)]
    elif mode == "multi":
        steps = [(0, [101], []), (3, [102], [11, 12, 13]),
                 (3, [11, 103], [21, 22, 23]), (0, [104], []), (0, [105], [])]
    else:
        steps = [(0, [t], []) for t in range(101, 107)]
    if constant_output:
        steps = [(k, [7] * len(selected), drafts) for k, selected, drafts in steps]
    events, previous_drafts = [], []

    def emit(cfg, kind, **fields):
        events.append(dict(kind=kind, schema="m203.boundary.event.v1", lane=cfg["lane"],
                           execution_sha=cfg["execution_sha"], config_fingerprint=fingerprint(cfg), **fields))

    with patch.object(observer, "emit", side_effect=emit), \
         patch.object(observer, "score", side_effect=AssertionError("early logits copy")):
        for index, (budget, selected, new_drafts) in enumerate(steps):
            scheduler.current_step += 1
            slots = len(previous_drafts)
            scheduled_ids = previous_drafts[:] if real_ids else [-1] * slots
            scheduled = NS(num_scheduled_tokens={RID: 176 - cached_prefix if index == 0 else slots + 1},
                scheduled_spec_decode_tokens={RID: scheduled_ids} if slots else {},
                num_spec_tokens_to_schedule=budget, num_invalid_spec_tokens={})
            scheduled.total_num_scheduled_tokens = sum(scheduled.num_scheduled_tokens.values())
            request.spec_token_ids = [-1] * budget
            scheduler._plan = NS(batch_k=budget, per_request_k={RID: budget})
            request.num_computed_tokens += scheduled.total_num_scheduled_tokens
            request.num_in_flight_tokens += scheduled.total_num_scheduled_tokens
            request.num_output_placeholders += 1 + slots
            observer.early_schedule(config, scheduler, scheduled)
            if index == 0:
                positions, inputs = list(range(cached_prefix, 176)), [8] * (176 - cached_prefix)
            else:
                start = 175 + len(request._output_token_ids)
                positions = list(range(start, start + slots + 1))
                inputs = [request._output_token_ids[-1]] + previous_drafts
            runner.positions = torch.tensor(positions)
            runner.input_ids = NS(gpu=torch.tensor(inputs))
            runner.query_start_loc = NS(np=[0, len(positions)])
            observer.kv_v1(config, runner, scheduled, runner.input_ids.gpu, runner.positions, None, None, None)
            spec = NS(num_draft_tokens=[slots], draft_token_ids=torch.tensor(previous_drafts),
                      logits_indices=torch.arange(slots + 1), target_logits_indices=torch.arange(slots),
                      bonus_logits_indices=torch.tensor([slots]), cu_num_sampled_tokens=torch.tensor([slots + 1])) if slots else None
            observer.begin(config, runner, scheduled, torch.empty(slots + 1, 1), spec, None)
            runner._draft_token_ids = torch.tensor([new_drafts], dtype=torch.int64)
            observer.early_produced(config, runner, scheduled)
            request.num_computed_tokens -= slots - len(selected) + 1
            request.num_in_flight_tokens -= scheduled.total_num_scheduled_tokens
            request.num_output_placeholders -= slots - len(selected) + 1
            observer.accounting(config, "account_before", request, selected, scheduled, False, selected)
            request._output_token_ids.extend(selected)
            request.num_output_placeholders -= len(selected)
            observer.accounting(config, "account_after", request, selected, scheduled, False, selected)
            previous_drafts = new_drafts
        scheduler.current_step += 1
        scheduler._plan = NS(batch_k=0, per_request_k={RID: 0})
        scheduled = NS(num_scheduled_tokens={RID: 1}, total_num_scheduled_tokens=1, scheduled_spec_decode_tokens={},
                       num_spec_tokens_to_schedule=0, num_invalid_spec_tokens={})
        request.spec_token_ids = []
        request.num_computed_tokens += 1
        request.num_in_flight_tokens += 1
        request.num_output_placeholders += 1
        observer.early_schedule(config, scheduler, scheduled)
        observer.early_prepare(config, runner, scheduled, torch.tensor([request._output_token_ids[-1]]), [181])
    record = dict(sample_id=boundary.SAMPLE, prompt_token_ids=[8] * 176,
                  response_id="chatcmpl-early", token_ids=request._output_token_ids)
    historical = dict(status="STATE_DIFFERENT", layer1_localization=dict(
                     status="LOCALIZED", P_K=177, P_V=177, P=177))
    events.append(dict(kind="cache_lookup", request_id=RID, cached_prefix_tokens=cached_prefix,
                       external_cached_tokens=0, num_computed_tokens=0))
    return json.loads(json.dumps(events)), record, dict(request_id=RID, pid=os.getpid()), historical


class TimelineContracts(unittest.TestCase):
    def report(self, data):
        return boundary.analyze_early(*data)

    def valid(self, data):
        report = self.report(data)
        self.assertEqual(report["status"], "INTERPRETABLE", report)
        return report

    def invalid(self, data):
        report = self.report(data)
        self.assertEqual(report["status"], "INVALID_EVIDENCE", report)
        self.assertFalse(report["task_done"])
        return report

    def test_ordinary_mapping_and_separate_input_output_ordinals(self):
        report = self.valid(fixture())
        row = report["timeline"][2]
        self.assertEqual((row["active_query"], row["rows"][0]["generated_index"]), ([[177, 102]], 2))
        self.assertEqual(row["input_token_ordinals_zero_based"], [1])
        self.assertEqual(row["committed_token_ordinals_zero_based"], [2])
        self.assertEqual(row["committed_token_ids"], [103])
        self.assertEqual(report["first_coverage_path"], "ordinary")

    def test_speculative_whole_event_beyond_window_and_distinct_producer(self):
        report = self.valid(fixture("multi"))
        row = report["timeline"][3]
        self.assertEqual([p for p, _ in row["active_query"]], [179, 180, 181, 182])
        self.assertEqual((row["draft_producer_step"], row["schedule_seq"]), (3, 4))
        self.assertEqual((row["planned_k"], row["producing_plan_k"]), (3, 0))
        self.assertEqual(report["first_coverage_path"], "speculative")

    def test_first_coverage_unique_but_ordered_rejection_rewrite_retained(self):
        report = self.valid(fixture("rewrite"))
        covering = [row for row in report["timeline"] if row["covers_position_177"]]
        self.assertEqual([row["schedule_seq"] for row in covering], [2, 3])
        self.assertEqual(report["first_coverage_step"], 2)
        self.assertEqual((covering[0]["num_accepted"], covering[0]["num_rejected"]), (0, 3))
        self.assertEqual(covering[0]["active_query"][1], [177, 11])
        self.assertEqual(covering[1]["active_query"], [[177, 102]])

    def test_duplicate_first_coverage_is_invalid(self):
        data = fixture("rewrite")
        event = copy.deepcopy(next(e for e in data[0] if e["kind"] == "early_verify" and e["schedule_seq"] == 2))
        event["event_seq"] = max(e.get("event_seq", 0) for e in data[0]) + 1
        data[0].append(event)
        self.invalid(data)

    def test_wrong_request_or_wrong_producer_invalid(self):
        for kind, field, value in (("early_schedule", "request_id", "wrong"),
                                   ("early_verify", "producer_schedule_seq", 99)):
            data = fixture("rewrite")
            next(e for e in data[0] if e["kind"] == kind and e["schedule_seq"] == 2)[field] = value
            self.invalid(data)

    def test_draft_accept_reject_invalid_count_inconsistencies(self):
        for field in ("num_drafted", "num_accepted", "num_rejected", "num_invalid", "scheduler_rejected_slots"):
            with self.subTest(field=field):
                data = fixture("rewrite")
                event = next(e for e in data[0] if e["kind"] == "early_account_before" and e["schedule_seq"] == 2)
                event[field] += 1
                self.invalid(data)

    def test_real_scheduled_ids_and_placeholder_provenance(self):
        self.valid(fixture("rewrite", real_ids=True))
        self.valid(fixture("rewrite"))
        for kind in ("early_schedule", "early_account_before", "early_account_after"):
            data = fixture("rewrite", real_ids=True)
            next(e for e in data[0] if e["kind"] == kind and e["schedule_seq"] == 2)["scheduled_spec_decode_tokens"][0] = 999
            self.invalid(data)

    def test_produced_ids_or_placeholders_do_not_match(self):
        for kind, field, value in (("early_produced", "produced_count", 2),
                                   ("early_schedule", "next_placeholders", [])):
            data = fixture("rewrite")
            next(e for e in data[0] if e["kind"] == kind and e["schedule_seq"] == 1)[field] = value
            self.invalid(data)

    def test_tensor_ownership_row_and_output_identity_mismatch(self):
        for field, value in (("producer_tensor_id", -1), ("previous_request_row", 9),
                             ("scheduler_output_id", -1)):
            data = fixture("rewrite")
            next(e for e in data[0] if e["kind"] == "early_verify" and e["schedule_seq"] == 2)[field] = value
            self.invalid(data)

    def test_commit_linkage_mismatch(self):
        data = fixture()
        next(e for e in data[0] if e["kind"] == "early_account_after")["committed_ids"] = [999]
        self.invalid(data)

    def test_event_order_malformed_missing_observer_error_and_unclosed(self):
        for mode in ("duplicate", "reversed", "missing", "error", "unclosed"):
            data = fixture()
            if mode == "duplicate":
                data[0][1]["event_seq"] = data[0][0]["event_seq"]
            elif mode == "reversed":
                data[0][0]["event_seq"], data[0][1]["event_seq"] = data[0][1]["event_seq"], data[0][0]["event_seq"]
            elif mode == "error":
                data[0].append(dict(kind="observer_error"))
            else:
                if mode == "unclosed":
                    data[0].remove(next(e for e in data[0] if e["kind"] == "early_complete"))
                else:
                    data[0].pop(3)
                for i, e in enumerate((e for e in data[0] if e["kind"].startswith("early_")), 1):
                    e["event_seq"] = i
            self.invalid(data)

    def test_flags_and_existing_kv_localization_required(self):
        report = self.valid(fixture())
        self.assertFalse(report["task_done"])
        self.assertEqual(report["interpretation_status"], "NOT_YET_APPROVED")
        self.assertEqual(report["temporal_kv_onset"], "UNKNOWN")
        self.assertNotIn("PASS", json.dumps(report))
        for field, value in (("status", "INVALID_EVIDENCE"), ("layer1_localization", {})):
            data = fixture()
            data[3][field] = value
            self.invalid(data)

    def test_file_order_is_not_event_order(self):
        data = fixture("multi")
        data[0].reverse()
        self.valid(data)

    def test_cached_and_uncached_prefill_use_actual_suffix(self):
        for count in (0, 96, 160):
            data = fixture(cached_prefix=count)
            report = self.valid(data)
            self.assertEqual(report["timeline"][0]["rows"][0]["flat_index"], 175 - count)
        data = fixture()
        next(e for e in data[0] if e["kind"] == "cache_lookup")["cached_prefix_tokens"] = 0
        self.invalid(data)

    def test_schedule_ahead_of_prior_commit_reconciles_counter_ledger(self):
        data = fixture()
        def event(kind, seq):
            return next(e for e in data[0] if e["kind"] == kind and e["schedule_seq"] == seq)
        ahead = event("early_schedule", 2)
        data[0].remove(ahead)
        data[0].insert(data[0].index(event("early_account_before", 1)), ahead)
        ahead["num_in_flight_tokens"] += 80
        ahead["num_output_placeholders"] += 1
        for kind in ("early_account_before", "early_account_after"):
            for field in ("num_computed_tokens", "num_in_flight_tokens", "num_output_placeholders"):
                event(kind, 1)[field] += 1
        for i, e in enumerate((e for e in data[0] if e["kind"].startswith("early_")), 1):
            e["event_seq"] = i
        self.valid(data)

    def test_consistent_but_wrong_before_after_counters_still_invalid(self):
        data = fixture("rewrite")
        for e in data[0]:
            if e["kind"] in ("early_account_before", "early_account_after") and e["schedule_seq"] == 2:
                e["num_computed_tokens"] += 1
        self.invalid(data)


class ObservationAndSourceContracts(unittest.TestCase):
    def tearDown(self):
        observer._early.clear()

    def test_disabled_scope_does_not_read_runtime_or_emit(self):
        with patch.object(observer, "ints", side_effect=AssertionError("unexpected copy")), patch.object(observer, "emit") as emit:
            observer.early_schedule({}, None, None)
            observer.early_prepare({}, None, None, None, None)
            observer.early_produced({}, None, None)
        emit.assert_not_called()
        with self.assertRaises(EvidenceError):
            observer.early_state(dict(early_timeline=True, lane="vanilla"))

    def test_same_output_object_required_and_native_step_monotonic(self):
        config = dict(early_timeline=True, lane="adaptive", sample_id=boundary.SAMPLE, prompt_token_ids=[8] * 176)
        req = NS(prompt_token_ids=[8] * 176, request_id=RID, spec_token_ids=[],
                 num_computed_tokens=176, num_in_flight_tokens=1, num_output_placeholders=1)
        scheduler = NS(requests={RID: req}, current_step=41, _plan=NS(batch_k=0, per_request_k={RID: 0}))
        output = NS(num_scheduled_tokens={RID: 1}, scheduled_spec_decode_tokens={}, num_spec_tokens_to_schedule=0)
        with patch.object(observer, "emit"):
            observer.early_schedule(config, scheduler, output)
            state = observer.early_state(config)
            self.assertEqual(observer.early_step(state, output)["schedule_seq"], 41)
            with self.assertRaises(EvidenceError):
                observer.early_step(state, copy.copy(output))
            with self.assertRaises(EvidenceError):
                observer.early_schedule(config, scheduler, copy.copy(output))

    def test_production_observation_is_metadata_only(self):
        observer._early.clear()
        # The fixture exercises actual production helpers; only actual-input and
        # consumed-draft reads may use ints(). Production must never copy values.
        original = observer.early_produced
        def checked(*args):
            with patch.object(observer, "ints", side_effect=AssertionError("production D2H")):
                return original(*args)
        with patch.object(observer, "early_produced", side_effect=checked):
            fixture("rewrite")

    def test_pinned_sources_compile_strip_and_production_hook_is_after_proposal(self):
        raw = boundary_tests.pinned_sources()["v1/worker/gpu_model_runner.py"].encode()
        patched = sources.transform("v1/worker/gpu_model_runner.py", raw)
        text = patched.decode()
        start = text.index("def propose_draft_token_ids(sampled_token_ids):")
        proposal = text.index("self._draft_token_ids = self.propose_draft_token_ids(", start)
        hook = text.index('_pb279_observe("early_produced", self, scheduler_output)', proposal)
        self.assertLess(hook, text.index("self._copy_draft_token_ids_to_cpu(scheduler_output)", start))
        self.assertEqual(ast.dump(sources.normalized_transform_ast("v1/worker/gpu_model_runner.py", patched)),
                         ast.dump(ast.parse(raw)))

    def test_frozen_scheduler_only_observer_calls_approved(self):
        root = Path(__file__).resolve().parents[1]
        raw = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "show",
                                       boundary.CHECKPOINT + ":src/server/adaptive_scheduler.py"], cwd=root)
        patched = (root / "src/server/adaptive_scheduler.py").read_bytes()
        sources.validate_scheduler_observation(raw, patched)
        for bad in (patched.replace(b"self._plan.batch_k", b"self.num_spec_tokens"),
                    patched.replace(b'"early_schedule"', b'"different_event"')):
            with self.assertRaises(EvidenceError):
                sources.validate_scheduler_observation(raw, bad)


class EarlyAnalyzerIntegrationTests(unittest.TestCase):
    def artifact(self):
        helper = layer1_tests.LocalizationAnalyzerIntegrationTests()
        self.addCleanup(helper.doCleanups)
        evidence = layer1_tests.evidence
        with patch.object(layer1_tests, "evidence", side_effect=lambda: evidence(
                changes=(("K", 177, -123), ("V", 177, -124)))):
            root = helper.artifact()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["early_timeline"] = True
        manifest_path.write_text(json.dumps(manifest))
        for lane in ("vanilla", "adaptive"):
            output = root / lane
            config_path = output / "boundary_config.json"
            config = json.loads(config_path.read_text())
            config["early_timeline"] = lane == "adaptive"
            config_path.write_text(json.dumps(config))
            binding_path = output / "observer_binding.json"
            binding = json.loads(binding_path.read_text())
            binding.update(config_sha256=file_sha(config_path), config_fingerprint=fingerprint(config))
            binding_path.write_text(json.dumps(binding))
            existing = []
            for path in (output / "observer").glob("*.json"):
                event = json.loads(path.read_text())
                event["config_fingerprint"] = fingerprint(config)
                if event["kind"] == "cache_lookup":
                    event["num_computed_tokens"] = 0
                path.write_text(json.dumps(event))
                existing.append(event)
            if lane == "adaptive":
                forward = next(e for e in existing if e["kind"] == "forward")
                for i, event in enumerate(fixture(constant_output=True)[0]):
                    if not event["kind"].startswith("early_"):
                        continue
                    event.update(request_id=forward["request_id"], pid=forward["pid"],
                                 execution_sha=manifest["execution_sha"], config_fingerprint=fingerprint(config))
                    if event["kind"] == "early_schedule" and event["planner_request_id"] is not None:
                        event["planner_request_id"] = forward["request_id"]
                    (output / "observer" / f"early-{i}.json").write_text(json.dumps(event))
        return root

    def test_full_analyzer_requires_boundary_and_kv_and_scoped_timeline(self):
        result = boundary.analyze(self.artifact())
        self.assertEqual(result["status"], "INTERPRETABLE", result)
        self.assertEqual(result["early_timeline"]["first_coverage_path"], "ordinary")
        self.assertFalse(result["task_done"])
        self.assertEqual(result["temporal_kv_onset"], "UNKNOWN")

    def test_missing_early_commit_invalidates_parent(self):
        root = self.artifact()
        for path in (root / "adaptive/observer").glob("early-*.json"):
            if json.loads(path.read_text())["kind"] == "early_account_after":
                path.unlink()
                break
        result = boundary.analyze(root)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertEqual(result["early_timeline"]["status"], "INVALID_EVIDENCE")

    def test_scoped_events_cannot_bypass_missing_boundary_or_kv(self):
        root = self.artifact()
        path = next(p for p in (root / "adaptive/observer").glob("*.json")
                    if json.loads(p.read_text())["kind"] == "forward")
        path.unlink()
        result = boundary.analyze(root)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertNotIn("early_timeline", result)

    def test_enabled_timeline_cannot_disable_existing_kv_gate(self):
        root = self.artifact()
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["kv_probe"] = False
        path.write_text(json.dumps(manifest))
        result = boundary.analyze(root)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertFalse(result["task_done"])


if __name__ == "__main__":
    unittest.main()
