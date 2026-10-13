"""M2-03 diagnostic extension: record ALL output updates, including K=0.

Based on the existing v0.30.0 TraceAdaptiveScheduler. Never use for
performance benchmarking or replace the canonical Lane 3 scheduler.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from src.server.adaptive_scheduler_trace_diag import TraceAdaptiveScheduler

logger = init_logger(__name__)


class TraceAllOutputScheduler(TraceAdaptiveScheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._all_output_trace_path = os.environ.get(
            "ADAPTIVE_ALL_OUTPUT_TRACE_PATH", ""
        ).strip()
        if not self._all_output_trace_path:
            logger.warning("M2-03 all-output trace disabled: no output path")

    def _update_request_with_output(
        self, request: Any, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # Snapshot before delegating: vLLM can truncate new_token_ids in-place.
        before = len(request._output_token_ids)
        proposed = [int(x) for x in new_token_ids]

        # The canonical update executes exactly once and exceptions propagate.
        result = super()._update_request_with_output(
            request, new_token_ids, is_stale=is_stale
        )

        if self._all_output_trace_path:
            try:
                after = len(request._output_token_ids)
                pending = self._trace_pending.get(request.request_id) or {}
                event = {
                    "trace_version": "m2-03.all-output.v1",
                    "event_type": "output_update",
                    "request_id": request.request_id,
                    "output_start_index": before,
                    "output_end_index_exclusive": after,
                    "generated_token_ids_raw": proposed,
                    "emitted_token_ids": [
                        int(x) for x in request._output_token_ids[before:after]
                    ],
                    "is_stale": bool(is_stale),
                    "spec_step_index": pending.get("spec_step_index"),
                    "has_scheduled_spec_snapshot": bool(pending),
                }
                # Preserve diagnostics in a dedicated file, without touching
                # the existing step_trace.jsonl or canonical evidence.
                with Path(self._all_output_trace_path).open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("M2-03 all-output trace write failed")
        return result
