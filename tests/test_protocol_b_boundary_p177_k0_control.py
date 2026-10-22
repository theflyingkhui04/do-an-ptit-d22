"""CPU scheduler/bytes/evidence contracts; no CUDA execution is emulated."""
import ast
import copy
import functools
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

import test_protocol_b_boundary_kv as kv_tests
import test_protocol_b_boundary_postwrite as post_tests
from experiments import protocol_b_boundary as boundary
from experiments import protocol_b_boundary_observer as observer
from experiments import protocol_b_boundary_sources as sources
from experiments.protocol_b import EvidenceError, fingerprint


# vLLM v0.30.0 Apache-2.0 source; pinned read-only native control dependency.
ASYNC_SCHEDULER_SOURCE = "# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n\nfrom vllm.logger import init_logger\nfrom vllm.v1.core.sched.output import SchedulerOutput\nfrom vllm.v1.core.sched.scheduler import Scheduler\nfrom vllm.v1.request import Request, RequestStatus\n\nlogger = init_logger(__name__)\n\n\nclass AsyncScheduler(Scheduler):\n    def __init__(self, *args, **kwargs) -> None:\n        super().__init__(*args, **kwargs)\n        # reusable read-only placeholder list for speculative decoding.\n        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens\n        self.pp_size = self.parallel_config.pipeline_parallel_size\n\n    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:\n        super()._update_after_schedule(scheduler_output)\n        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens\n        # Use the latest num of scheduled draft tokens in next step as placeholder.\n        self._spec_token_placeholders = [\n            -1\n        ] * scheduler_output.num_spec_tokens_to_schedule\n        for req_id in scheduler_output.num_scheduled_tokens:\n            request = self.requests[req_id]\n            if request.is_prefill_chunk:\n                continue\n\n            scheduler_output.pending_structured_output_tokens |= (\n                request.use_structured_output and request.num_output_placeholders > 0\n            )\n            # The request will generate num_sampled_tokens_per_step new tokens\n            # plus num_spec_tokens in this scheduling step. Diffusion has no AR\n            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas\n            # (spec) tokens.\n            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))\n            request.num_output_placeholders += (\n                self.num_sampled_tokens_per_step + cur_num_spec_tokens\n            )\n            # Add placeholders for the new draft/spec tokens.\n            # We will update the actual spec token ids in the worker process.\n            request.spec_token_ids = self._spec_token_placeholders\n\n            if self.use_v2_model_runner:\n                # Set the next step index in which this request is eligible to be\n                # scheduled for decode (for PP microbatching).\n                request.next_decode_eligible_step = self.current_step + self.pp_size\n\n    def _update_request_with_output(\n        self, request: Request, new_token_ids: list[int], is_stale: bool = False\n    ) -> tuple[list[int], bool]:\n        status_before_update = request.status\n        new_token_ids, stopped = super()._update_request_with_output(\n            request, new_token_ids\n        )\n\n        # Placeholders were zeroed at preemption; a stale delivery must not\n        # decrement them (it would underflow).\n        if not is_stale:\n            request.num_output_placeholders -= len(new_token_ids)\n            assert request.num_output_placeholders >= 0\n\n        # Cache the new tokens. Preempted requests should be skipped.\n        if status_before_update == RequestStatus.RUNNING:\n            self.kv_cache_manager.cache_blocks(\n                request, request.num_computed_tokens - request.num_output_placeholders\n            )\n        return new_token_ids, stopped\n"


def config():
    request = dict(messages=[dict(role="user", content="fixture")], seed=42, temperature=0)
    return dict(directory="fixture", sample_id=boundary.SAMPLE, lane="adaptive", non_canonical=True,
        early_timeline=True, postwrite_position177=True, prompt_token_ids=[8]*176, execution_sha="a"*40,
        p177_k0_control=dict(profile=observer.CONTROL_PROFILE, request=request, request_sha256=fingerprint(request),
            baseline_sha256="b"*64, normal_producer_budget=3, token_prefix=[7, 7]))


class ProducerTests(unittest.TestCase):
    def setUp(self):
        observer._controls.clear()
        observer._early.clear()
        self.addCleanup(observer._controls.clear)
        self.addCleanup(observer._early.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"
        self.cfg = config()
        self.write()
        env = patch.dict(os.environ, M203_P177_PRODUCER_K0="1", PROTOCOL_B_BOUNDARY_CONFIG=str(self.path))
        env.start()
        self.addCleanup(env.stop)
        self.events = []
        emitter = patch.object(observer, "emit", side_effect=lambda cfg, kind, **fields:
                               self.events.append(dict(kind=kind, **fields)))
        emitter.start()
        self.addCleanup(emitter.stop)
        self.request = NS(request_id="rid", prompt_token_ids=[8]*176, is_prefill_chunk=False,
                          num_computed_tokens=176, spec_token_ids=[-1]*3,
                          num_in_flight_tokens=0, num_output_placeholders=0)
        self.scheduler = NS(requests={"rid": self.request}, current_step=17,
            _plan=NS(batch_k=3, per_request_k={"rid": 3}), _planner=NS(per_request=False))
        self.output = NS(num_scheduled_tokens={"rid": 1}, total_num_scheduled_tokens=1,
                         scheduled_spec_decode_tokens={}, num_spec_tokens_to_schedule=3)
        observer.early_state(self.cfg)["request_id"] = "rid"

    def write(self):
        self.path.write_text(json.dumps(self.cfg))

    def force(self):
        observer.control_producer(self.scheduler, self.output)

    def native(self):
        self.request.spec_token_ids = [-1] * self.output.num_spec_tokens_to_schedule
        self.request.num_computed_tokens += 1
        observer.early_schedule(self.cfg, self.scheduler, self.output)

    def worker(self, positions, output=None):
        output = output or self.output
        state = observer.early_state(self.cfg)
        if id(output) not in state["steps"]:
            self.scheduler.current_step += 1
            observer.early_schedule(self.cfg, self.scheduler, output)
        entry = observer.early_step(state, output)
        entry.update(positions=positions, active_query=[[p, 7] for p in positions])
        runner = NS(_draft_token_ids=torch.empty(1, 0, dtype=torch.int64))
        observer.control_query(self.cfg, runner, output, positions)
        return runner

    def test_env_off_has_no_reads_or_mutations(self):
        with patch.dict(os.environ, M203_P177_PRODUCER_K0="0"), \
             patch.object(observer.Path, "read_text", side_effect=AssertionError("disabled read")):
            self.force()
        self.assertEqual(self.output.num_spec_tokens_to_schedule, 3)
        self.assertEqual(self.events, [])

    def test_wrong_sample_request_position_and_prefill_do_not_override(self):
        for mode in ("sample", "request", "position", "prefill"):
            with self.subTest(mode=mode):
                if mode == "sample":
                    self.cfg["sample_id"] = "other"
                    self.write()
                elif mode == "request":
                    self.request.request_id = "other"
                elif mode == "position":
                    self.request.num_computed_tokens = 175
                else:
                    self.request.is_prefill_chunk = True
                self.force()
                self.assertEqual(self.output.num_spec_tokens_to_schedule, 3)
                self.cfg["sample_id"] = boundary.SAMPLE
                self.request.request_id, self.request.num_computed_tokens, self.request.is_prefill_chunk = "rid", 176, False
                self.write()

    def test_ambiguous_batch_fails_without_mutation(self):
        self.output.num_scheduled_tokens["other"] = 1
        with self.assertRaises(EvidenceError):
            self.force()
        self.assertEqual(self.output.num_spec_tokens_to_schedule, 3)

    def test_only_budget_changes_original_plan_preserved_and_second_override_fails(self):
        before = copy.deepcopy(vars(self.output))
        plan = copy.deepcopy(vars(self.scheduler._plan))
        self.force()
        before["num_spec_tokens_to_schedule"] = 0
        self.assertEqual(vars(self.output), before)
        self.assertEqual(vars(self.scheduler._plan), plan)
        self.assertEqual((self.events[0]["original_budget"], self.events[0]["effective_budget"]), (3, 0))
        self.output.num_spec_tokens_to_schedule = 3  # A fresh-looking candidate cannot bypass the one-shot ledger.
        with self.assertRaisesRegex(EvidenceError, "Second"):
            self.force()

    def test_exact_real_method_passes_zero_to_native_super(self):
        tree = ast.parse((boundary.ROOT / "src/server/adaptive_scheduler.py").read_bytes())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_update_after_schedule")
        seen = []
        def native(output):
            seen.append(output.num_spec_tokens_to_schedule)
            self.request.spec_token_ids = [-1] * output.num_spec_tokens_to_schedule
            self.request.num_computed_tokens += 1
        namespace = dict(_pb177_control=observer.control_producer,
            _pb279_observe=lambda stage, scheduler, output: observer.early_schedule(self.cfg, scheduler, output),
            super=lambda: NS(_update_after_schedule=native))
        exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_scheduler_method", "exec"), namespace)
        namespace["_update_after_schedule"](self.scheduler, self.output)
        self.assertEqual(seen, [0])
        self.assertEqual(self.request.spec_token_ids, [])
        self.assertEqual(self.events[-1]["kind"], "p177_control_native")

    def test_producer_consumer_and_zero_draft_identity(self):
        self.force()
        self.native()
        runner = self.worker([176])
        observer.control_drafts(self.cfg, runner, self.output)
        consumer = NS(num_scheduled_tokens={"rid": 1}, total_num_scheduled_tokens=1,
                      scheduled_spec_decode_tokens={}, num_spec_tokens_to_schedule=3)
        self.worker([177], consumer)
        self.assertEqual(self.events[-1]["kind"], "p177_control_consumer")
        self.assertEqual(consumer.num_spec_tokens_to_schedule, 3)
        with self.assertRaises(EvidenceError):
            self.worker([177], consumer)

    def test_wrong_actual_producer_and_nonzero_drafts_fail(self):
        self.force()
        self.native()
        with self.assertRaises(EvidenceError):
            self.worker([175, 176])
        runner = self.worker([176])
        runner._draft_token_ids = torch.empty(1, 3, dtype=torch.int64)
        with self.assertRaises(EvidenceError):
            observer.control_drafts(self.cfg, runner, self.output)

    def test_env_requires_explicit_profile_and_request_hash(self):
        self.cfg.pop("p177_k0_control")
        self.write()
        with self.assertRaises(EvidenceError):
            self.force()

    def test_profile_hash_and_canonical_flag_are_mandatory(self):
        for mode in ("hash", "canonical", "profile"):
            self.cfg = config()
            if mode == "hash":
                self.cfg["p177_k0_control"]["request_sha256"] = "0"*64
            elif mode == "canonical":
                self.cfg["non_canonical"] = False
            else:
                self.cfg["p177_k0_control"]["profile"] = "wrong"
            self.write()
            with self.subTest(mode=mode), self.assertRaises(EvidenceError):
                self.force()
            self.assertEqual(self.output.num_spec_tokens_to_schedule, 3)

    def test_consumer_wrong_positions_scheduled_drafts_and_later_rewrite_fail(self):
        self.force()
        self.native()
        runner = self.worker([176])
        observer.control_drafts(self.cfg, runner, self.output)
        consumer = NS(num_scheduled_tokens={"rid": 4}, total_num_scheduled_tokens=4,
                      scheduled_spec_decode_tokens={"rid": [1,2,3]}, num_spec_tokens_to_schedule=3)
        with self.assertRaises(EvidenceError):
            self.worker([177,178,179,180], consumer)

    def test_lookup_table_is_never_mutated(self):
        self.scheduler.dynamic_sd_lookup = [4]*33
        self.force()
        self.assertEqual(self.scheduler.dynamic_sd_lookup, [4]*33)

    def test_ambiguous_reference_prompt_is_rejected_before_run_creation(self):
        with patch.object(boundary, "command", return_value=""), \
             patch.object(boundary, "control_scheduler_auth", return_value={}), \
             patch.object(boundary, "control_baseline", return_value={"runs": {"adaptive": {
                 "records": [dict(prompt_token_ids=[8]*176)]*46}}}), self.assertRaisesRegex(EvidenceError, "ambiguous"):
            boundary.prepare_control(Path(self.tmp.name) / "new-run", "session", Path("fixture"))


class PrefixTests(unittest.TestCase):
    def test_exact_original_fp16_prefix_and_only_two_bounded_transfers(self):
        cache, table, expected, _ = kv_tests.cache_fixture(3, noncontiguous=True)
        calls, original = [], torch.Tensor.cpu
        def cpu(tensor, *args, **kwargs):
            calls.append(tensor.numel())
            return original(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, "cpu", cpu):
            result = observer.hash_control_prefix(cache, table[0].tolist(), [225, -1])
        self.assertEqual(calls, [177*2*3]*2)
        for component, values in (("K", expected), ("V", -expected)):
            self.assertEqual(result[component], [hashlib.sha256(
                values[p].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for p in range(177)])

    def test_duplicate_history_current_and_padding_aliases_fail(self):
        cache, table, _, _ = kv_tests.cache_fixture()
        for mapping, writes in (([0]*12, [177]), (table[0].tolist(), [176]), (table[0].tolist(), [177, 0])):
            with self.subTest(writes=writes), self.assertRaises(EvidenceError):
                observer.hash_control_prefix(cache, mapping, writes)

    def test_prefix_signed_zero_bits_preserved(self):
        cache, table, _, _ = kv_tests.cache_fixture()
        before = observer.hash_control_prefix(cache, table[0].tolist(), [177])
        cache[0, 0, 0, 0] = -0.0
        self.assertNotEqual(before["K"], observer.hash_control_prefix(cache, table[0].tolist(), [177])["K"])


@functools.lru_cache(maxsize=1)
def template():
    args = post_tests.evidence((("K", -123), ("V", -124)))
    cfg = config()
    events = args[0]
    baseline = dict(identity=dict(archive_sha256="b"*64), normal_producer_budget=3,
        postwrites={lane: copy.deepcopy(events[lane][1]) for lane in ("vanilla", "adaptive")},
        runs=dict(adaptive=dict(records=[{}]*45 + [dict(prompt_token_ids=[8]*176, token_ids=[7]*284)])))
    post = copy.deepcopy(events["adaptive"][1])
    post.update(actual_positions=[177], active_query=[[177, 7]], input_token_id=7,
                schedule_seq=18, scheduler_output_id=200, non_canonical=True,
                control_profile=observer.CONTROL_PROFILE, control_id="control")
    post["write_slots"] = post["write_slots"][:1]
    for identity in post["runtime_binding"]["inputs"].values():
        identity["shape"] = [1]
    for layer in post["runtime_binding"]["layers"].values():
        layer["metadata"]["slot_mapping"]["shape"] = [1]
    cache, table, _, _ = kv_tests.cache_fixture(3)
    post["prefix_0_176"] = observer.hash_control_prefix(cache, table[0].tolist(), post["write_slots"])
    baseline["prefix"] = {c: copy.deepcopy(post["prefix_0_176"][c]) for c in ("K", "V")}
    common = dict(non_canonical=True, control_profile=observer.CONTROL_PROFILE, control_id="control",
                  request_id="adaptive", pid=post["pid"], producer_seq=17, scheduler_output_id=100)
    dedicated = [dict(common, kind="p177_control_override", override_count=1, original_budget=3, effective_budget=0,
                     original_plan=dict(batch_k=3, per_request_k=dict(adaptive=3)),
                     request_sha256=cfg["p177_k0_control"]["request_sha256"], baseline_sha256="b"*64),
                 dict(common, kind="p177_control_native", effective_budget=0, next_placeholders=[]),
                 dict(common, kind="p177_control_producer", actual_positions=[176], active_query=[[176, 7]]),
                 dict(common, kind="p177_control_drafts", produced_count=0),
                 dict(common, kind="p177_control_consumer", scheduler_output_id=200, schedule_seq=18,
                      actual_positions=[177], active_query=[[177, 7]], scheduled_spec_tokens=[])]
    early = [dict(kind="early_schedule", request_id="adaptive", schedule_seq=17, scheduler_output_id=100,
                  num_spec_tokens_to_schedule=0, next_placeholders=[], planned_k=3, batch_k=3),
             dict(kind="early_schedule", request_id="adaptive", schedule_seq=18, scheduler_output_id=200,
                  placeholder_origin_schedule_seq=17, scheduled_spec_decode_tokens=[]),
             dict(kind="early_verify", request_id="adaptive", schedule_seq=18, scheduler_output_id=200,
                  active_query=[[177, 7]], decoding_path="ordinary", verification_draft_slots=0)]
    authenticated = dict(fixture_auth="authenticated")
    auth = dict(kind="kv_sources", auth_id=post["auth_id"], pid=post["pid"], **authenticated)
    all_events = dedicated + early + [auth, post]
    for i, e in enumerate(all_events):
        e.update(schema="m203.boundary.event.v1", lane="adaptive", execution_sha=cfg["execution_sha"],
                 config_fingerprint=fingerprint(cfg), timestamp=f"2026-10-18T00:01:{i:02d}+00:00")
    record = dict(sample_id=boundary.SAMPLE, ok=True, response_id="adaptive", prompt_token_ids=[8]*176,
                  token_ids=[7]*284, request_sha256=cfg["p177_k0_control"]["request_sha256"])
    return all_events, record, cfg, baseline, authenticated


def evidence():
    return copy.deepcopy(template())


class ComparisonTests(unittest.TestCase):
    def report(self, data):
        return boundary.compare_control(*data)

    def invalid(self, data):
        result = self.report(data)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertEqual(result["classification"], "INVALID_EVIDENCE")
        self.assertFalse(result["task_done"])

    def test_control_equal_vanilla_and_different_from_vanilla_classifications(self):
        for match in (False, True):
            data = evidence()
            if match:
                for c in ("K", "V"):
                    data[0][-1][c] = copy.deepcopy(data[3]["postwrites"]["vanilla"][c])
            result = self.report(data)
            self.assertEqual(result["status"], "INTERPRETABLE", result)
            self.assertEqual(result["classification"], "SPECULATIVE_MULTI_TOKEN_FORWARD_DIVERGENCE" if match
                             else "RUNNER_OR_EXECUTION_PATH_STILL_DIFFERENT")
            for field in ("task_done", "same_kv_cache_state_proven", "historical_kv_state_equivalent"):
                self.assertIs(result[field], False)
            self.assertIs(result["non_canonical"], True)
            self.assertEqual(result["temporal_kv_onset"], "UNKNOWN")

    def test_prefix_k_and_v_mismatch_invalid(self):
        for component in ("K", "V"):
            data = evidence()
            data[0][-1]["prefix_0_176"][component][176] = "0"*64
            self.invalid(data)

    def test_missing_duplicate_override_and_wrong_producer_consumer(self):
        changes = [(0, "override_count", 2), (0, "original_budget", 0), (0, "effective_budget", 1),
                   (1, "next_placeholders", [-1]), (2, "actual_positions", [175]),
                   (3, "produced_count", 3), (4, "actual_positions", [177,178,179,180]),
                   (4, "scheduled_spec_tokens", [1]), (4, "scheduler_output_id", 999)]
        for index, field, value in changes:
            data = evidence()
            data[0][index][field] = value
            with self.subTest(field=field):
                self.invalid(data)
        for mode in ("missing", "duplicate"):
            data = evidence()
            if mode == "missing":
                data[0].pop(0)
            else:
                data[0].append(copy.deepcopy(data[0][0]))
            self.invalid(data)

    def test_speculative_or_earlier_coverage_cannot_qualify_as_control(self):
        for mode in ("speculative", "earlier", "duplicate"):
            data = evidence()
            verify = next(e for e in data[0] if e["kind"] == "early_verify")
            if mode == "speculative":
                verify.update(decoding_path="speculative", verification_draft_slots=3)
            else:
                previous = copy.deepcopy(verify)
                if mode == "earlier":
                    previous["schedule_seq"] -= 1
                data[0].append(previous)
            self.invalid(data)

    def test_baseline_must_differ_and_provenance_must_match(self):
        for mode in ("baseline_equal", "request", "config", "auth", "archive", "row"):
            data = evidence()
            if mode == "baseline_equal":
                data[3]["postwrites"]["adaptive"] = copy.deepcopy(data[3]["postwrites"]["vanilla"])
            elif mode == "request":
                data[1]["request_sha256"] = "0"*64
            elif mode == "config":
                data[0][-1]["config_fingerprint"] = "0"*64
            elif mode == "auth":
                data[4]["fixture_auth"] = "wrong"
            elif mode == "archive":
                data[3]["identity"]["archive_sha256"] = "0"*64
            else:
                data[0][-1]["selected_row"] = 1
            with self.subTest(mode=mode):
                self.invalid(data)

    def test_canonical_postwrite_cannot_accept_ordinary_control(self):
        data = post_tests.evidence()
        data[0]["adaptive"][1]["actual_positions"] = [177]
        self.assertEqual(boundary.analyze_postwrite(*data)["status"], "INVALID_EVIDENCE")


class SourceTests(unittest.TestCase):
    def test_exact_hook_is_before_super_and_all_other_scheduler_ast_is_pristine(self):
        root = boundary.ROOT
        raw = subprocess.check_output(["git", "-c", "safe.directory="+root.as_posix(), "show",
                                       boundary.CHECKPOINT+":src/server/adaptive_scheduler.py"], cwd=root)
        patched = (root / "src/server/adaptive_scheduler.py").read_bytes()
        sources.validate_scheduler_observation(raw, patched)
        hook = b"        _pb177_control(self, scheduler_output)\n"
        native = b"        super()._update_after_schedule(scheduler_output)\n"
        self.assertIn(hook+native, patched.replace(b"\r\n", b"\n"))
        for bad in (patched.replace(b"_pb177_control(self, scheduler_output)", b"_pb177_control(self, None)"),
                    patched.replace(hook+native, native+hook),
                    patched.replace(hook, hook*2),
                    patched.replace(b"self._plan.batch_k", b"self.num_spec_tokens")):
            with self.assertRaises(EvidenceError):
                sources.validate_scheduler_observation(raw, bad)

    def test_planner_controller_and_adapter_unchanged_from_base(self):
        for name in ("src/server/budget_planner.py", "src/proposer/controller.py", "src/proposer/adapter.py"):
            raw = subprocess.check_output(["git", "-c", "safe.directory="+boundary.ROOT.as_posix(), "show",
                                           boundary.CONTROL_BASE_EXECUTION+":"+name], cwd=boundary.ROOT)
            self.assertEqual(raw, (boundary.ROOT / name).read_bytes())

    def test_pinned_vllm_transforms_remain_supported(self):
        # Existing lifecycle tests exercise temporary packages, not installed /usr/local vLLM.
        data = {name: raw.encode() for name, raw in post_tests.base_tests.pinned_sources().items()}
        self.assertEqual(set(data), set(sources.HASHES))
        for name, raw in data.items():
            sources.validate_transform(name, raw, sources.transform(name, raw))

    def test_control_install_authenticate_restore_preserves_pristine_vllm(self):
        helper = post_tests.base_tests.AtomicSourceWriteTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        auth = dict(profile=observer.CONTROL_PROFILE, scheduler_sha256=hashlib.sha256(
            (boundary.ROOT / "src/server/adaptive_scheduler.py").read_bytes()).hexdigest())
        (helper.run / "manifest.json").write_text(json.dumps(dict(control_profile=observer.CONTROL_PROFILE,
            non_canonical=True, source_auth=auth)))
        native = helper.package / "v1/core/sched/async_scheduler.py"
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_bytes(ASYNC_SCHEDULER_SOURCE.encode())
        sources.install(helper.run)
        sources.authenticate(helper.run, installed=True)
        metadata = json.loads((helper.run / "instrumentation.json").read_text())
        self.assertEqual(metadata["scheduler_control"], auth)
        sources.restore(helper.run)
        for name, raw in helper.originals.items():
            self.assertEqual((helper.package / name).read_bytes(), raw)


class BaselineArchiveTests(unittest.TestCase):
    def archive(self):
        helper = post_tests.AnalyzerIntegrationTests()
        self.addCleanup(helper.doCleanups)
        root = helper.artifact()
        for path in (root / "adaptive/observer").glob("*.json"):
            event = json.loads(path.read_text())
            if event["kind"] == "postwrite_position177":
                history = next(json.loads(p.read_text()) for p in (root / "adaptive/observer").glob("*.json")
                               if json.loads(p.read_text())["kind"] == "kv_history")
                for c in ("K", "V"):
                    event[c]["sha256"] = history["layers"][observer.LAYER1]["localization"][c]["position_sha256"][177]
                path.write_text(json.dumps(event))
        scheduler = subprocess.check_output(["git", "-c", "safe.directory="+boundary.ROOT.as_posix(), "show",
            boundary.CONTROL_BASE_EXECUTION+":src/server/adaptive_scheduler.py"], cwd=boundary.ROOT)
        with tarfile.open(root / "execution-sources.tar.gz", "w:gz") as tar:
            info = tarfile.TarInfo("src/server/adaptive_scheduler.py")
            info.size = len(scheduler)
            tar.addfile(info, io.BytesIO(scheduler))
        for lane in ("vanilla", "adaptive"):
            path = root / lane / "run.json"
            run = json.loads(path.read_text())
            run["environment"]["source_hashes"] = {"src/server/adaptive_scheduler.py": hashlib.sha256(scheduler).hexdigest()}
            path.write_text(json.dumps(run))
        report = boundary.analyze(root)
        self.assertEqual(report["status"], "INTERPRETABLE", report)
        (root / "packages.txt").write_text("fixture==1\n")
        boundary.seal(root)
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as tar:
            tar.add(root, arcname="sealed")
        blob = raw.getvalue()
        path = root.parent / boundary.CONTROL_BASELINE_NAME
        path.write_bytes(blob)
        self.addCleanup(path.unlink, missing_ok=True)
        return path, hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest()

    def test_pinned_archive_seal_and_full_canonical_baseline_revalidation(self):
        path, blob = self.archive()
        before = path.read_bytes()
        with patch.object(boundary, "CONTROL_BASELINE_BLOB", blob), \
             patch.object(boundary, "CONTROL_BASE_EXECUTION", "a"*40):
            baseline = boundary.control_baseline(path)
        self.assertEqual(baseline["normal_producer_budget"], 3)
        self.assertEqual(len(baseline["prefix"]["K"]), 177)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(EvidenceError):
            boundary.control_baseline(path)

    def test_unknown_archive_rejected_before_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / boundary.CONTROL_BASELINE_NAME
            path.write_bytes(b"unknown")
            with self.assertRaises(EvidenceError):
                boundary.control_baseline(path)


class ControlArtifactTests(unittest.TestCase):
    def artifact(self):
        import test_protocol_b as protocol_tests
        from experiments import protocol_b as pb
        from experiments.protocol_b import file_sha
        _, pairs = protocol_tests.evidence.__wrapped__()
        env = copy.deepcopy(pairs[0]["adaptive"]["environment"])
        data = evidence()
        events, record, cfg, baseline, _ = data
        lifecycle = post_tests.base_tests.AtomicSourceWriteTests()
        lifecycle.setUp()
        self.addCleanup(lifecycle.doCleanups)
        root = lifecycle.run
        output = root / "adaptive"
        (output / "observer").mkdir(parents=True)
        scheduler = subprocess.check_output(["git", "-c", "safe.directory="+boundary.ROOT.as_posix(), "show",
            boundary.CONTROL_BASE_EXECUTION+":src/server/adaptive_scheduler.py"], cwd=boundary.ROOT)
        baseline["scheduler_source"] = scheduler.decode()
        source_auth = boundary.control_scheduler_auth(baseline)
        env["source_hashes"]["src/server/adaptive_scheduler.py"] = source_auth["scheduler_sha256"]
        for name in ("experiments/protocol_b.py", "experiments/protocol_b_capture.py"):
            env["source_hashes"][name] = file_sha(boundary.ROOT / name)
        old_env = copy.deepcopy(env)
        old_env["git_sha"] = boundary.CONTROL_BASE_EXECUTION
        old_env["source_hashes"]["src/server/adaptive_scheduler.py"] = hashlib.sha256(scheduler).hexdigest()
        baseline["runs"]["adaptive"]["environment"] = old_env
        baseline["packages"] = ["fixture==1"]
        prompts = []
        for i in range(54):
            request = cfg["p177_k0_control"]["request"] if i == 45 else dict(messages=[dict(role="user", content=str(i))],
                                                                        seed=42, temperature=0)
            prompts.append(dict(workload="W3", sample_id=boundary.SAMPLE if i == 45 else f"fixture-{i}",
                dataset_version="fixture.v1", prompt_version="fixture.v1", request=request,
                request_sha256=fingerprint(request)))
        plan = dict(prompts=prompts)
        baseline["manifest"] = dict(plan=plan)
        code_paths = boundary.CONTROL_SOURCE_PATHS
        gpu_selection = dict(raw_selector=env["gpu_uuid"], resolved_gpu_uuid=env["gpu_uuid"],
            expected_gpu_uuid=env["gpu_uuid"], inventory="0, " + env["gpu_uuid"], boot_id=env["boot_id"])
        manifest = dict(control_profile=observer.CONTROL_PROFILE, non_canonical=True, baseline_archive="fixture.tar.gz",
            baseline_identity=baseline["identity"], source_auth=source_auth, execution_sha=env["git_sha"],
            session_id=env["session_id"], plan=plan, gpu_selection=gpu_selection,
            source_hashes={name: file_sha(boundary.ROOT / name) for name in code_paths})
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "packages.txt").write_text("fixture==1\n")
        (root / "server-packages.txt").write_text("fixture==1\n")
        native = lifecycle.package / "v1/core/sched/async_scheduler.py"
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_bytes(ASYNC_SCHEDULER_SOURCE.encode())
        sources.install(root)
        auth = sources.authenticate(root)
        with tarfile.open(root / "execution-sources.tar.gz", "w:gz") as tar:
            for name in code_paths:
                tar.add(boundary.ROOT / name, arcname=name)
        records, journal = [], []
        model = protocol_tests.capture.LANES["adaptive"][1]
        for i, prompt in enumerate(prompts[:46]):
            r = dict(prompt, pair_id="boundary279", run_id="adaptive", lane="adaptive",
                     timestamp="2026-10-18T00:01:10+00:00", ok=True, token_ids=[7]*284,
                     prompt_token_ids=[8]*176, completion_tokens=284, finish_reason="stop",
                     response_id="adaptive" if i == 45 else f"response-{i}")
            payload = dict(prompt["request"], model=model, return_token_ids=True)
            body = dict(id=r["response_id"], model=model, prompt_token_ids=r["prompt_token_ids"],
                        choices=[dict(token_ids=r["token_ids"], finish_reason="stop")], usage=dict(completion_tokens=284))
            journal += [dict(event="attempt", request=payload), dict(event="result", request=payload, record=r,
                       http_status=200, raw_response_body=json.dumps(body))]
            records.append(r)
        (output / "responses.jsonl").write_text("\n".join(json.dumps(e) for e in journal)+"\n")
        live = protocol_tests.live_fixture("adaptive", env)
        run = dict(complete=True, diagnostic_only=True, diagnostic_through=boundary.SAMPLE, pair_id="boundary279",
            run_id="adaptive", lane="adaptive", records=records, environment=env, live_server=live,
            started_at="2026-10-18T00:01:00+00:00", finished_at="2026-10-18T00:01:30+00:00",
            responses_sha256=file_sha(output / "responses.jsonl"))
        (output / "run.json").write_text(json.dumps(run))
        (output / "boundary_config.json").write_text(json.dumps(cfg))
        (output / "observer_binding.json").write_text(json.dumps(dict(control_env="1", server_pid=live["pid"],
            process_start_ticks=live["process_start_ticks"], config_sha256=file_sha(output / "boundary_config.json"),
            config_fingerprint=fingerprint(cfg))))
        for i, e in enumerate(events):
            if e["kind"] == "kv_sources":
                e.update(auth)
            if "pid" in e:
                e["pid"] = live["pid"]
            (output / "observer" / f"event-{i}.json").write_text(json.dumps(e))
        return root, baseline

    def test_complete_control_artifact_validates_without_canonical_fingerprint(self):
        root, baseline = self.artifact()
        with patch.object(boundary, "control_baseline", return_value=baseline):
            result = boundary.analyze_control(root)
        self.assertEqual(result["status"], "INTERPRETABLE", result)
        self.assertFalse(result["task_done"])
        self.assertEqual(result["classification"], "RUNNER_OR_EXECUTION_PATH_STILL_DIFFERENT")
        self.assertFalse((root / "vanilla").exists())

    def test_ordinal_control_artifact_validates_and_keeps_raw_selector(self):
        root, baseline = self.artifact()
        manifest_path, run_path = root / "manifest.json", root / "adaptive/run.json"
        manifest, run = json.loads(manifest_path.read_text()), json.loads(run_path.read_text())
        manifest["gpu_selection"]["raw_selector"] = "0"
        run["environment"]["config"]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
        run["live_server"]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
        manifest_path.write_text(json.dumps(manifest))
        run_path.write_text(json.dumps(run))
        from experiments.protocol_b import validate_run
        expected = {(p["workload"], p["sample_id"]): p for p in manifest["plan"]["prompts"][:46]}
        with self.assertRaisesRegex(EvidenceError, "GPU/cache selection inconsistent"):
            validate_run(run, expected, "boundary279", "adaptive", run["environment"])
        with patch.object(boundary, "control_baseline", return_value=baseline):
            result = boundary.analyze_control(root)
        self.assertEqual(result["status"], "INTERPRETABLE", result)
        self.assertEqual(json.loads(run_path.read_text())["environment"]["config"]["environment"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_gpu_resolution_manifest_tampering_invalidates_control(self):
        root, baseline = self.artifact()
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["gpu_selection"]["resolved_gpu_uuid"] = "GPU-b306be9c-78ed-b3de-9c67-33130c066b7e"
        path.write_text(json.dumps(manifest))
        with patch.object(boundary, "control_baseline", return_value=baseline):
            result = boundary.analyze_control(root)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)

    def test_runtime_provenance_mismatch_invalidates_control(self):
        root, baseline = self.artifact()
        path = root / "adaptive/run.json"
        run = json.loads(path.read_text())
        run["environment"]["driver_version"] = "999.0"
        path.write_text(json.dumps(run))
        with patch.object(boundary, "control_baseline", return_value=baseline):
            result = boundary.analyze_control(root)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)

    def test_sealed_baseline_binding_mismatch_invalidates_control(self):
        root, baseline = self.artifact()
        baseline["identity"]["archive_sha256"] = "0"*64
        with patch.object(boundary, "control_baseline", return_value=baseline):
            result = boundary.analyze_control(root)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)

    def test_package_mismatch_invalidates_control(self):
        root, baseline = self.artifact()
        (root / "server-packages.txt").write_text("fixture==2\n")
        with patch.object(boundary, "control_baseline", return_value=baseline):
            result = boundary.analyze_control(root)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)


if __name__ == "__main__":
    unittest.main()
