"""CPU harness: full hash-pinned native retirement plus production pending pipeline.

Fixture mechanics retained from approved G1-E; no audit recording observer is used.
"""
import __future__
import ast
from collections import defaultdict, deque
import enum
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
import pytest
from experiments.m4_pinned_contract import verify
from test_adaptive_scheduler_glue import FakeAsyncScheduler, add_running, build, scheduler_module
from scalar_eagle_fixtures import identity, planner
from flexsds.core.eagle_consumer_profile import EagleConsumerProfile
PIN=Path(os.environ["M4_VLLM_SOURCE_ROOT"])/"vllm"


def native_nodes(relative, class_name, names):
    tree = ast.parse((PIN / relative).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]

def native_classes():
    assert verify(PIN.parent)["verified_files"] == 19
    request_tree = ast.parse((PIN / "v1/request.py").read_text(encoding="utf-8"))
    status = next(n for n in request_tree.body if isinstance(n, ast.ClassDef) and n.name == "RequestStatus")
    utils = ast.parse((PIN / "v1/core/sched/utils.py").read_text(encoding="utf-8"))
    stop = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == "check_stop")
    namespace = dict(enum=enum, defaultdict=defaultdict, time=time, logger=logging.getLogger(__name__),
                     EngineCoreOutput=SimpleNamespace, EngineCoreOutputs=SimpleNamespace,
                     remove_all=lambda rows, removed: [r for r in rows if r not in removed])
    methods = native_nodes("v1/core/sched/scheduler.py", "Scheduler", {
        "update_from_output", "_update_request_with_output", "_handle_stopped_request",
        "_preempt_request", "_update_after_schedule",
    })
    base = ast.ClassDef(name="NativeBase", bases=[], keywords=[], body=methods, decorator_list=[])
    async_cls = ast.ClassDef(name="NativeAsync", bases=[ast.Name(id="NativeBase", ctx=ast.Load())],
        keywords=[], body=native_nodes("v1/core/sched/async_scheduler.py", "AsyncScheduler",
                                     {"_update_request_with_output", "_update_after_schedule"}), decorator_list=[])
    for node, filename in [(status, "v1/request.py"), (stop, "v1/core/sched/utils.py"),
                           (base, "v1/core/sched/scheduler.py"), (async_cls, "v1/core/sched/async_scheduler.py")]:
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(PIN / filename), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
    return namespace["NativeBase"], namespace["NativeAsync"], namespace["RequestStatus"]

class CpuRequest:
    """Only native Request data; no model execution or synthetic stop algorithm."""
    def __init__(self, rid, status):
        self.request_id = rid
        self.status = status.RUNNING
        self.status_type = status
        self.is_prefill_chunk = False
        self.spec_token_ids = []
        self.num_computed_tokens = 127
        self.num_in_flight_tokens = 0
        self.num_stale_output_tokens = 0
        self.num_output_placeholders = 0
        self.num_preemptions = 0
        self.num_prompt_tokens = 127
        self.drop_stale_output = False
        self.has_encoder_inputs = False
        self.spec_decode_metrics = None
        self.pooling_params = None
        self.client_index = 0
        self.stop_reason = None
        self.num_nans_in_logits = 0
        self.trace_headers = None
        self.resumable = False
        self.streaming_queue = deque()
        self._output_token_ids = []
        self.output_token_ids = self._output_token_ids
        self.max_tokens = 100
        self.use_structured_output = False
        self.sampling_params = SimpleNamespace(eos_token_id=2, stop_token_ids=[3],
                                               min_tokens=0, repetition_detection=None,
                                               num_logprobs=None)

    @property
    def num_tokens(self):
        return self.num_prompt_tokens + len(self._output_token_ids)

    @property
    def num_output_tokens(self):
        return len(self._output_token_ids)

    def append_output_token_ids(self, token):
        self._output_token_ids.append(token)

    def is_finished(self):
        return self.status_type.is_finished(self.status)

    def get_finished_reason(self):
        return self.status.name if self.is_finished() else None

    def take_events(self):
        return None

    def take_prefill_stats(self):
        return None

def retire(s,output,tokens=None):
    ids=list(output.num_scheduled_tokens)
    result=SimpleNamespace(req_ids=ids,req_id_to_index={r:i for i,r in enumerate(ids)},
        sampled_token_ids=[list(tokens or [10,11,12]) for _ in ids],logprobs=None,
        prompt_logprobs_dict={},pooler_output=None,num_nans_in_logits=None,
        kv_connector_output=None,ec_connector_output=None,cudagraph_stats=None,
        routed_experts=None,sampling_masks=None)
    output.num_invalid_spec_tokens=None
    output.total_num_scheduled_tokens=sum(output.num_scheduled_tokens.values())
    return s.update_from_output(output,result)


@pytest.fixture
def native_runtime(scheduler_module, monkeypatch, tmp_path):
    base, native_async, status = native_classes()
    # Keep the actual project scheduler/lineage. Native async super() resolves
    # through this test subclass; no upstream file or installed module changes.
    project = scheduler_module.AdaptiveAsyncScheduler
    class FeasibilityScheduler(project, native_async):
        pass
    monkeypatch.setattr(scheduler_module, "AdaptiveAsyncScheduler", FeasibilityScheduler)
    monkeypatch.setattr(FakeAsyncScheduler, "update_from_output", base.update_from_output)
    monkeypatch.setattr(FakeAsyncScheduler, "_update_request_with_output", native_async._update_request_with_output,
                        raising=False)
    monkeypatch.setattr(FakeAsyncScheduler, "_handle_stopped_request", base._handle_stopped_request, raising=False)
    def native_schedule_accounting(scheduler, output):
        # The shared fake schedule supplies the admitted cohort. Execute the
        # actual pinned accounting/async placeholders for these query widths.
        output.num_scheduled_tokens = {
            rid: 1 + len(output.scheduled_spec_decode_tokens.get(rid, ()))
            for rid in output.num_scheduled_tokens}
        output.has_structured_output_requests = False
        output.pending_structured_output_tokens = False
        native_async._update_after_schedule(scheduler, output)
    monkeypatch.setattr(FakeAsyncScheduler, "_update_after_schedule", native_schedule_accounting)
    # Cover exact host-only old widths reached by this CPU fixture.
    p, snap=planner(base={"r":4},ready=False)
    from dataclasses import replace
    cells=[]
    for width in (0,2,3,4):
        state=replace(snap,old_verification_width_by_request=(("r",width),))
        for key,cell in __import__("scalar_eagle_fixtures").profile_for(
                p,p._candidate_decision_states(state,{"r":4},()),(1.,1.8,2.3,3.2)).cells:
            cells.append((key,cell))
    profile=EagleConsumerProfile("synthetic-native-CPU",identity(),tuple(cells),synthetic=True)
    path=tmp_path/"synthetic-profile.json"
    path.write_text(json.dumps(profile.to_data()))
    monkeypatch.setattr(scheduler_module,"observe_runtime_identity",lambda *a,**kw:identity())
    s = build(scheduler_module, monkeypatch, tmp_path, ADAPTIVE_REPLACEMENT_POLICY="1",
              ADAPTIVE_COST_POLICY_MODE="strict",ADAPTIVE_CONSUMER_PROFILE_PATH=str(path),
              ADAPTIVE_STEP_TELEMETRY_PATH=str(tmp_path/"steps.jsonl"),
              ADAPTIVE_CONTROLLER_KWARGS=json.dumps({"initial_alpha": .95,"min_warmup_samples":50}))
    s.defer_block_free = False
    s.perf_metrics = None
    s.enable_return_routed_experts = False
    s.num_sampled_tokens_per_step = 1
    s.use_v2_model_runner = False
    s.is_mm_encoder_only = False
    s.return_sampling_mask = False
    s.spec_decode_metrics_level = "summary"
    s.grammar_compile_error_reqs = set()
    s.recompute_kv_load_failures = True
    s.ec_connector = None
    s.connector = None
    s.kv_cache_manager = SimpleNamespace(take_events=lambda: None, cache_blocks=lambda *args: None)
    s.finished_req_ids_dict = None
    s.structured_output_manager = SimpleNamespace(accept_tokens=lambda *args: True)
    s.make_stats = lambda *args: None
    s.max_model_len = 4096
    s.num_waiting_for_streaming_input = 0
    class Waiting(list):
        def prepend_request(self, request):
            self.insert(0, request)
    s.waiting = Waiting()
    s._enqueue_waiting_request = s.waiting.append
    s._free_request_blocks = lambda r: None
    s.encoder_cache_manager = SimpleNamespace(free=lambda r: None)
    s._inflight_prefills = set()
    s.log_stats = False
    s.reset_preempted_req_ids = set()
    r = CpuRequest("r", status)
    add_running(s, r)
    producer=s.schedule()
    retire(s,producer,tokens=[7])
    return s,r,base,status
