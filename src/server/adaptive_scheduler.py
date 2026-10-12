"""Lane 3 adaptive scheduler for vLLM v0.30.0 (loaded via ``--scheduler-cls``).

Thin glue between vLLM's ``AsyncScheduler`` and ``AdaptiveBudgetPlanner``:

- ``schedule``                : plan k_i for decoding requests, write K = max(k_i)
                                into ``dynamic_sd_lookup`` so the draft model drafts K tokens.
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

logger = init_logger(__name__)


class AdaptiveAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._planner = AdaptiveBudgetPlanner.from_env()
        self._planner.max_k = min(self._planner.max_k, self.num_spec_tokens)
        self._read_vram = _env_flag(os.environ.get("ADAPTIVE_ENABLE_VRAM_GUARD", "1"))
        self._plan = StepPlan(batch_k=0, per_request_k={})

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
        super()._update_after_schedule(scheduler_output)
        if not self._planner.per_request:
            return
        scheduled = scheduler_output.num_scheduled_tokens
        for req_id, k in self._plan.per_request_k.items():
            request = self.requests.get(req_id)
            if request is None or req_id not in scheduled or request.is_prefill_chunk:
                continue
            # Only shrink: placeholders were sized to the batch K.
            if k < len(request.spec_token_ids):
                request.spec_token_ids = [-1] * k

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
            self._planner.record_verification(request_id, drafted, num_accepted_tokens)
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
        return super()._free_request(request, *args, **kwargs)

    def _free_vram_mb(self) -> Optional[float]:
        if not self._read_vram:
            return None
        try:
            import torch

            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes / (1024 * 1024)
        except Exception:
            return None
