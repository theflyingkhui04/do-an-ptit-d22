"""Opt-in worker-local observer for pinned vLLM; no runtime source edits.

Installed only after engine initialization. Eager execution keeps the outer
target module hooks observable. RPCs transfer control and small timing records.
"""
from __future__ import annotations

import hashlib
import inspect
import time
import json
import traceback

from experiments.m4_cost_profiler import timed


def install(worker):
    import torch
    import vllm
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    if vllm.__version__ != "0.30.0":
        raise RuntimeError("Observer requires vLLM 0.30.0")
    runner = worker.model_runner
    if getattr(worker, "use_v2_model_runner", False):
        raise RuntimeError("M4 observer requires V1 model runner; set VLLM_USE_V2_MODEL_RUNNER=0")
    if hasattr(runner, "_m4_observer"):
        raise RuntimeError("Observer already installed")
    state = {"active": False, "samples": {}, "observed_shapes": []}
    runner._m4_observer = state
    execute = runner.execute_model

    def execute_observed(scheduler_output, *args, **kwargs):
        state["scheduled"] = dict(scheduler_output.num_scheduled_tokens)
        state["preempted"] = list(scheduler_output.preempted_req_ids or ())
        try:
            return execute(scheduler_output, *args, **kwargs)
        except Exception as exc:
            # Diagnostic only, outside successful timing. Never attempt another
            # forward on this context. Caller persists the original exception.
            print(json.dumps(dict(m4_worker_failure=str(exc),phase="execute_model",
                                  memory=memory_snapshot(),traceback=traceback.format_exc())),flush=True)
            raise

    runner.execute_model = execute_observed

    def before_target(module, args):
        state["eligible"] = False
        if not state["active"]:
            return
        batch = runner.input_batch
        ids = list(batch.req_ids)
        counts = [state["scheduled"].get(r, 0) for r in ids]
        contexts = [int(v) for v in batch.num_computed_tokens_cpu[:batch.num_reqs]]
        shape = dict(actual_batch_size=batch.num_reqs, query_tokens_per_request=counts,
                     context_lengths=contexts, preempted_req_ids=state["preempted"])
        state["observed_shapes"].append(shape)
        state["eligible"] = (batch.num_reqs == state["batch_size"] and
                             counts == [state["K"]+1]*state["batch_size"] and
                             contexts == [state["context_len"]]*state["batch_size"] and
                             not state["preempted"])
        if state["eligible"] and "target" not in state["samples"]:
            torch.cuda.synchronize()
            state["start"] = time.perf_counter()

    def after_target(module, args, output):
        if state["active"] and state["eligible"] and "target" not in state["samples"]:
            torch.cuda.synchronize()
            state["samples"]["target"] = (time.perf_counter()-state["start"])*1000

    runner.model.register_forward_pre_hook(before_target)
    runner.model.register_forward_hook(after_target)
    sources = {"runner": inspect.getfile(type(runner)),
               "eagle": inspect.getfile(EagleProposer),
               "base_proposer": inspect.getfile(SpecDecodeBaseProposer)}
    if runner.speculative_config is not None:
        if not isinstance(runner.drafter, EagleProposer) or runner.drafter.parallel_drafting:
            raise RuntimeError("Only ordinary non-parallel EAGLE supported")
        propose = runner.drafter.propose
        signature = inspect.signature(propose)

        def propose_observed(*args, **kwargs):
            eligible = state["active"] and state.get("eligible") and "propose" not in state["samples"]
            if not eligible:
                return propose(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            metadata = bound["common_attn_metadata"]
            if (bound["num_speculative_tokens"] != state["K"] or
                    metadata.batch_size() != state["batch_size"]):
                raise RuntimeError("Proposer shape differs from requested calibration")
            output, ms = timed(lambda: propose(*args, **kwargs), torch.cuda.synchronize, time.perf_counter)
            if tuple(output.shape) != (state["batch_size"], state["K"]):
                raise RuntimeError("Unexpected proposal output shape")
            state["samples"]["propose"] = ms
            return output

        runner.drafter.propose = propose_observed
    return {name: hashlib.sha256(open(path, "rb").read()).hexdigest() for name, path in sources.items()}


def begin(worker, batch_size, k, context_len):
    state = worker.model_runner._m4_observer
    state.update(active=True, batch_size=batch_size, K=k, context_len=context_len,
                 samples={}, observed_shapes=[], eligible=False)


def finish(worker):
    state = worker.model_runner._m4_observer
    state["active"] = False
    return dict(samples=state["samples"],observed_shapes=state["observed_shapes"],
                memory=memory_snapshot())


class M4ObserverExtension:
    """Named RPCs avoid enabling arbitrary pickle/callable deserialization."""

    def m4_install(self):
        return install(self)

    def m4_begin(self, batch_size, k, context_len):
        return begin(self, batch_size, k, context_len)

    def m4_finish(self):
        return finish(self)

    def m4_memory(self):
        return memory_snapshot()


def memory_snapshot():
    """Allocator and driver evidence in the CUDA worker, never a frontend guess."""
    import torch
    result = {}
    try:
        free, total = torch.cuda.mem_get_info()
        result.update(free_bytes=free,total_bytes=total,
                      allocated_bytes=torch.cuda.memory_allocated(),
                      reserved_bytes=torch.cuda.memory_reserved(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      allocator_stats=dict(torch.cuda.memory_stats()))
    except Exception as exc:
        result["snapshot_error"] = f"{type(exc).__name__}: {exc}"
    return result
