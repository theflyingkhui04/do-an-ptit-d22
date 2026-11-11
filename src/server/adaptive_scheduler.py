"""Lane 3 adaptive scheduler for vLLM v0.30.0 (loaded via ``--scheduler-cls``).

Thin glue between vLLM's ``AsyncScheduler`` and ``AdaptiveBudgetPlanner``:

- ``schedule``                : plan k_i for decoding requests, write K = max(k_i)
                                into ``dynamic_sd_lookup`` for this output's proposer pass
                                (its drafts are verified in a later scheduling step).
- ``_update_after_schedule``  : trim each request's draft placeholders to k_i
                                (vLLM trims real drafts to the placeholder length).
- ``make_spec_decoding_stats``: per-request (drafted, accepted) feedback to the controller.
- ``_free_request``           : finish the request session and write its telemetry line.

Lanes 1/2 run with async scheduling (auto-enabled for ``draft_model``), so this
class extends ``AsyncScheduler`` to keep the comparison fair. Requires the
speculative config to set ``num_speculative_tokens_per_batch_size`` so that
``dynamic_sd_lookup`` exists. See logs/Duong/M2-02/M2-02-openspec.md Part C.
"""
from __future__ import annotations

import os
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from src.server.budget_planner import AdaptiveBudgetPlanner, StepPlan, _env_flag
from src.server.lineage import ledger_for, release_ledger
from experiments.protocol_b_boundary_observer import observe as _pb279_observe
from experiments.protocol_b_boundary_observer import control_producer as _pb177_control

logger = init_logger(__name__)


class AdaptiveAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._planner = AdaptiveBudgetPlanner.from_env()
        self._planner.max_k = min(self._planner.max_k, self.num_spec_tokens)
        self._read_vram = _env_flag(os.environ.get("ADAPTIVE_ENABLE_VRAM_GUARD", "1"))
        self._plan = StepPlan(batch_k=0, per_request_k={})
        self._schedule_sequence = 0
        self._lineage_config = kwargs.get("vllm_config", args[0] if args else None)
        self._lineage = ledger_for(self._lineage_config)
        self._retiring = None

        if self.dynamic_sd_lookup is None:
            logger.warning(
                "AdaptiveAsyncScheduler: num_speculative_tokens_per_batch_size is not set, "
                "batch K stays fixed at %d; only per-request trimming is active.",
                self.num_spec_tokens,
            )
        logger.info(
            "AdaptiveAsyncScheduler ready: max_k=%d min_batch_k=%d per_request=%s "
            "vram_guard=%s telemetry=%s",
            self._planner.max_k,
            self._planner.min_batch_k,
            self._planner.per_request,
            self._read_vram,
            self._planner.telemetry_path,
        )

    def schedule(self, *args, **kwargs):
        self._schedule_sequence += 1
        try:
            decode_ids = [r.request_id for r in self.running if not r.is_prefill_chunk]
            self._plan = self._planner.plan_step(
                decode_ids,
                batch_size=len(self.running),
                kv_usage=self.get_kv_cache_usage(),
                free_vram_mb=self._free_vram_mb(),
            )
            lookup = self.dynamic_sd_lookup
            if lookup is not None:
                for batch_size in range(1, len(lookup)):
                    lookup[batch_size] = self._plan.batch_k
        except Exception:
            # Never take the engine down over a planning error: keep the previous K.
            logger.exception("AdaptiveAsyncScheduler: planning failed, keeping previous K")
            self._plan = StepPlan(batch_k=self._plan.batch_k, per_request_k={})
        return super().schedule(*args, **kwargs)

    def _update_after_schedule(self, scheduler_output) -> None:
        _pb177_control(self, scheduler_output)
        super()._update_after_schedule(scheduler_output)
        scheduled = scheduler_output.num_scheduled_tokens
        if self._planner.per_request:
            for req_id, k in self._plan.per_request_k.items():
                request = self.requests.get(req_id)
                if request is None or req_id not in scheduled or request.is_prefill_chunk:
                    continue
                # Only shrink: these placeholders reserve drafts for later verification.
                if k < len(request.spec_token_ids):
                    request.spec_token_ids = [-1] * k
        try:
            for req_id in scheduled:
                request = self.requests.get(req_id)
                if request is not None and not request.is_prefill_chunk:
                    self._planner.record_scheduled_budget(
                        req_id, scheduler_output.num_spec_tokens_to_schedule,
                        len(request.spec_token_ids), dynamic=self.dynamic_sd_lookup is not None,
                        schedule_sequence=self._schedule_sequence,
                        plan_sequence=(self._plan.plan_sequence
                                       if req_id in self._plan.per_request_k else None),
                    )
        except Exception:
            logger.exception("AdaptiveAsyncScheduler: scheduled-budget telemetry failed")
        try:
            self._lineage.register(scheduler_output, self._plan, self._schedule_sequence, self.requests)
        except Exception:
            self._lineage.clear()
            logger.exception("AdaptiveAsyncScheduler: lineage registration failed")
        _pb279_observe("early_schedule", self, scheduler_output)

    def update_from_output(self, scheduler_output, model_runner_output):
        previous = self._retiring
        # Upstream may deliver stale statistics after preemption. Preserve them
        # upstream, but do not adapt from a discarded execution incarnation.
        stale = {rid for rid in scheduler_output.num_scheduled_tokens
                 if getattr(self.requests.get(rid), "num_stale_output_tokens", 0) > 0}
        self._retiring = (self._lineage.get(scheduler_output), model_runner_output, stale)
        try:
            return super().update_from_output(scheduler_output, model_runner_output)
        finally:
            self._retiring = previous
            self._lineage.retire(scheduler_output)

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens,
        request_id: str,
    ):
        try:
            drafted = num_draft_tokens
            if num_invalid_spec_tokens:
                drafted -= num_invalid_spec_tokens.get(request_id, 0)
            binding = None
            reason = "missing-lineage"
            if self._retiring is not None:
                entry, result, stale = self._retiring
                if entry is not None:
                    binding = entry.consumed.pop(request_id, None)
                if binding is not None:
                    origin, expected = binding
                    current = self._lineage.active.get(request_id)
                    index = result.req_id_to_index.get(request_id)
                    valid = (current is not None and self.requests.get(request_id) is current[0]
                             and current[1] == origin.incarnation
                             and origin.request_id == request_id
                             and index is not None and 0 <= index < len(result.req_ids)
                             and result.req_ids[index] == request_id
                             and expected == num_draft_tokens
                             and 0 <= num_accepted_tokens <= drafted <= expected
                             and request_id not in stale
                             and not getattr(current[0], "is_finished", lambda: False)())
                    if not valid:
                        binding = None
                        reason = "stale-or-mismatched-lineage"
            if binding is None:
                self._planner.record_lineage_failure(request_id, reason)
            elif drafted > 0:
                self._planner.record_verification(
                    request_id, drafted, num_accepted_tokens, source=binding[0])
        except Exception:
            logger.exception("AdaptiveAsyncScheduler: feedback failed for %s", request_id)
        return super().make_spec_decoding_stats(
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            num_invalid_spec_tokens,
            request_id,
        )

    def _free_request(self, request, *args, **kwargs):
        try:
            self._planner.finish_request(request.request_id)
        except Exception:
            logger.exception("AdaptiveAsyncScheduler: telemetry failed for %s", request.request_id)
        finally:
            self._lineage.finish_request(request)
            # Drop request-owned snapshots even when aggregate I/O fails.
            self._plan.per_request_k.pop(request.request_id, None)
            self._plan.per_request_node_count.pop(request.request_id, None)
            self._plan.per_request_decisions.pop(request.request_id, None)
        return super()._free_request(request, *args, **kwargs)

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            self._lineage.clear()
            release_ledger(self._lineage_config)

    def _free_vram_mb(self) -> Optional[float]:
        if not self._read_vram:
            return None
        try:
            import torch

            # With the single-GPU "uni" executor the worker shares this process, so
            # CUDA is already initialized. Never create a new context just to read VRAM.
            if not torch.cuda.is_initialized():
                return None
            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes / (1024 * 1024)
        except Exception:
            return None
