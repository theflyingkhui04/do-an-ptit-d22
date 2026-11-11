"""Exact-source-gated observation of vLLM 0.30.0 UniProc draft rows.

No worker/result schema changes and no added tensor reads or synchronization.
Unknown source/configuration is rejected before enabling this executor.
"""
import hashlib
import json
import logging
from pathlib import Path

import vllm
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from src.server.lineage import ledger_for, release_ledger

logger = logging.getLogger(__name__)


def verify_sources():
    if vllm.__version__ != "0.30.0":
        raise RuntimeError("Async lineage requires vLLM 0.30.0")
    manifest = Path(__file__).resolve().parents[2] / "experiments/scalar_vllm_source_manifest.json"
    root = Path(vllm.__file__).resolve().parent.parent
    records = {r["path"]: r for r in json.loads(manifest.read_text())["files"]}
    for path in records:
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != records[path]["sha256"]:
            raise RuntimeError(f"Unsupported async lineage source: {path}")


class LineageUniProcExecutor(UniProcExecutor):
    def _init_executor(self):
        verify_sources()
        config = self.vllm_config
        parallel = config.parallel_config
        spec = config.speculative_config
        if (parallel.world_size != 1 or parallel.data_parallel_size != 1
                or not config.scheduler_config.async_scheduling
                or spec is None or spec.method != "draft_model"
                or spec.disable_padded_drafter_batch):
            raise RuntimeError("Async lineage supports only single-worker async padded draft_model")
        super()._init_executor()
        runner = self.driver_worker.worker.model_runner
        if (type(runner).__module__ != "vllm.v1.worker.gpu_model_runner"
                or type(runner).__name__ != "GPUModelRunner" or not runner.use_async_scheduling):
            raise RuntimeError("Async lineage requires the pinned V1 GPUModelRunner")
        self._lineage_runner = runner
        self._lineage = ledger_for(config)
        self._lineage_pending = None
        self._lineage_sampling = None
        original = runner._copy_draft_token_ids_to_cpu

        def observe_birth(output, zeros_only=False):
            result = original(output, zeros_only=zeros_only)
            # Called by sample_tokens only after actual proposal or synthetic
            # zeros. This distinguishes a real generation from a shape guess.
            try:
                if output is self._lineage_sampling:
                    self._lineage.produced(output, runner, zeros_only)
                else:
                    self._lineage.generation = None
                    logger.warning("Draft birth outside a paired sampling call; provenance unavailable")
            except Exception:
                self._lineage.clear()
                logger.exception("Draft provenance observation failed; adaptive feedback disabled")
            return result

        runner._copy_draft_token_ids_to_cpu = observe_birth

    def execute_model(self, scheduler_output, non_block=False):
        runner = self._lineage_runner
        generation = self._lineage.generation
        try:
            result = super().execute_model(scheduler_output, non_block=non_block)
            state = runner.execute_model_state
            if state is not None and state[0] is scheduler_output:
                try:
                    self._lineage.prepared(scheduler_output, runner, generation)
                except Exception:
                    self._lineage.clear()
                    logger.exception("Draft consumption observation failed; adaptive feedback disabled")
                self._lineage_pending = scheduler_output
            else:
                self._lineage_pending = None
            return self._watch(result)
        except BaseException:
            self._lineage.clear()
            self._lineage_pending = None
            raise

    def sample_tokens(self, grammar_output, non_block=False):
        output = self._lineage_pending
        self._lineage_pending = None
        state = self._lineage_runner.execute_model_state
        self._lineage_sampling = output if state is not None and state[0] is output else None
        # Even a K=0/no-proposal sample replaces the preceding generation.
        self._lineage.generation = None
        try:
            result = super().sample_tokens(grammar_output, non_block=non_block)
            return self._watch(result)
        except BaseException:
            self._lineage.clear()
            raise
        finally:
            self._lineage_sampling = None

    def _watch(self, result):
        # AsyncOutputFuture materializes on .result(); its completion callback
        # also handles exceptions/cancellation without waiting for feedback.
        if hasattr(result, "add_done_callback"):
            def terminal(future):
                if future.cancelled() or future.exception() is not None:
                    self._lineage.clear()
            result.add_done_callback(terminal)
        return result

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            release_ledger(self.vllm_config)
