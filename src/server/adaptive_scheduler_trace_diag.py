"""M2-03 diagnostic-only token/verification trace for pinned vLLM v0.30.0.

This is NOT a replacement for the canonical adaptive scheduler. Only use
it for controlled c1, single-request-at-a-time diagnostics; logging alters
runtime overhead and can affect timing/nondeterminism.

Trace records are emitted after the original vLLM Scheduler.update_from_output
returns, pairing by request_id, never by batch list position alone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from src.server.adaptive_scheduler import AdaptiveAsyncScheduler

logger = init_logger(__name__)


class TraceAdaptiveScheduler(AdaptiveAsyncScheduler):
    """Observational tracing around vLLM's existing implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._trace_path = os.environ.get("ADAPTIVE_TRACE_PATH", "").strip()
        self._trace_pending: dict[str, dict[str, Any]] = {}
        self._trace_step_index: dict[str, int] = {}
        if not self._trace_path:
            logger.warning("TraceAdaptiveScheduler: ADAPTIVE_TRACE_PATH unset: trace disabled")
        else:
            logger.info("TraceAdaptiveScheduler: trace_path=%s", self._trace_path)

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any):
        """Snapshot verification inputs and actual appended output positions.

        In vLLM 0.30.0, make_spec_decoding_stats() is called *inside*
        Scheduler.update_from_output(), before tokens are appended. Its
        callback populates the matching request's pending record. After the
        parent returns, the same Request object reveals the tokens appended.
        """
        if not self._trace_path:
            return super().update_from_output(scheduler_output, model_runner_output)

        pending: dict[str, dict[str, Any]] = {}
        try:
            scheduled_drafts = scheduler_output.scheduled_spec_decode_tokens
            index_by_req = model_runner_output.req_id_to_index
            sampled = model_runner_output.sampled_token_ids
            for request_id, draft_ids in scheduled_drafts.items():
                if not draft_ids:
                    continue
                request = self.requests.get(request_id)
                if request is None or request_id not in index_by_req:
                    continue
                before_count = len(request._output_token_ids)
                req_index = index_by_req[request_id]
                raw_generated = sampled[req_index] if sampled else []
                pending[request_id] = {
                    "_request": request,  # stable object ref if request finishes
                    "request_id": request_id,
                    "output_start_index": before_count,
                    # IDs presented to the target verification (post scheduling,
                    # potentially grammar-processed); -1 denotes an invalid slot.
                    "scheduled_draft_token_ids": [int(x) for x in draft_ids],
                    # Copy BEFORE calling super: it can mutate generated lists
                    # during EOS / stop truncation.
                    "generated_token_ids_raw": [int(x) for x in raw_generated],
                    "num_sampled_tokens_per_step": int(self.num_sampled_tokens_per_step),
                }
        except Exception:
            # A trace failure must never skip the canonical scheduler update.
            logger.exception("M2-03 trace: failed to snapshot update_from_output")
            pending.clear()

        # The core scheduler has a synchronous update_from_output implementation.
        # Do not share this pending map across invocations.
        self._trace_pending = pending
        completed = False
        try:
            result = super().update_from_output(scheduler_output, model_runner_output)
            completed = True
            return result
        finally:
            try:
                if completed:
                    for record in pending.values():
                        # Only output a step when the vLLM speculative callback
                        # actually ran; avoids falsely aligning skipped/stale steps.
                        if "spec_step_index" not in record:
                            continue
                        request = record.pop("_request")
                        start = record["output_start_index"]
                        current_output_ids = request._output_token_ids
                        end = len(current_output_ids)
                        record["output_end_index_exclusive"] = end
                        record["emitted_token_ids"] = [
                            int(x) for x in current_output_ids[start:end]
                        ]
                        record["actual_output_delta"] = end - start
                        record["trace_version"] = "m2-03.steptrace.v1"
                        self._write_trace(record)
            except Exception:
                logger.exception("M2-03 trace: failed to finalize output snapshot")
            finally:
                self._trace_pending = {}

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Any,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: Any,
        request_id: str,
    ):
        # Execute the canonical feedback/stats logic EXACTLY once first.
        result = super().make_spec_decoding_stats(
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            num_invalid_spec_tokens,
            request_id,
        )

        if not self._trace_path:
            return result
        try:
            next_index = self._trace_step_index.get(request_id, 0) + 1
            self._trace_step_index[request_id] = next_index
            pending = self._trace_pending.get(request_id)
            if pending is not None:
                invalid = (
                    num_invalid_spec_tokens.get(request_id, 0)
                    if num_invalid_spec_tokens
                    else 0
                )
                pending["spec_step_index"] = next_index  # 1-based
                pending["num_drafted_raw"] = int(num_draft_tokens)
                pending["num_invalid_draft_tokens"] = int(invalid)
                pending["num_drafted_effective"] = int(num_draft_tokens) - int(invalid)
                pending["num_accepted"] = int(num_accepted_tokens)
        except Exception:
            logger.exception("M2-03 trace: stats callback annotation failed")
        return result

    def _free_request(self, request: Any, *args: Any, **kwargs: Any):
        # Keep cleanup secondary to the original vLLM/adapter behavior.
        try:
            return super()._free_request(request, *args, **kwargs)
        finally:
            self._trace_step_index.pop(request.request_id, None)

    def _write_trace(self, record: dict[str, Any]) -> None:
        try:
            path = Path(self._trace_path)
            # Directory must already exist; do not create hidden output roots.
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("M2-03 trace: could not append JSONL")
