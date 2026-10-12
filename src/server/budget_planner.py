"""vLLM-free budget planning logic behind the Lane 3 adaptive scheduler.

The vLLM scheduler subclass (``src.server.adaptive_scheduler``) only forwards
runtime signals to this planner, so every decision rule here can be unit-tested
without a GPU or a vLLM install.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Set

from src.proposer.adapter import AdaptiveProposerAdapter

# Free VRAM value passed to the controller when no reading is available,
# so the VRAM guard never fires on a missing signal.
NO_VRAM_READING_MB = 1.0e9


@dataclass
class StepPlan:
    """Draft budget for one scheduling step."""
    batch_k: int
    per_request_k: Dict[str, int] = field(default_factory=dict)


class AdaptiveBudgetPlanner:
    """Turns per-request AdaptiveDraftController decisions into a batch plan."""

    def __init__(
        self,
        adapter: Optional[AdaptiveProposerAdapter] = None,
        *,
        max_k: int = 4,
        min_batch_k: int = 0,
        per_request: bool = True,
        telemetry_path: Optional[str] = None,
    ):
        self.adapter = adapter or AdaptiveProposerAdapter()
        self.max_k = max_k
        self.min_batch_k = min_batch_k
        self.per_request = per_request
        self.telemetry_path = telemetry_path
        self._known_requests: Set[str] = set()

    def plan_step(
        self,
        decode_req_ids: Iterable[str],
        batch_size: int,
        kv_usage: float,
        free_vram_mb: Optional[float] = None,
    ) -> StepPlan:
        """Selects k_i for every decoding request and the batch-wide K = max(k_i).

        kv_usage is the vLLM fraction in [0, 1] (Scheduler.get_kv_cache_usage()).
        """
        kv_pct = 100.0 * kv_usage
        vram = NO_VRAM_READING_MB if free_vram_mb is None else free_vram_mb

        per_request_k: Dict[str, int] = {}
        for req_id in decode_req_ids:
            self._known_requests.add(req_id)
            controller = self.adapter.get_controller(req_id)
            decision = controller.select_budget(
                batch_size=batch_size,
                free_vram_mb=vram,
                kv_cache_usage_pct=kv_pct,
            )
            per_request_k[req_id] = min(decision.k, self.max_k)

        if not per_request_k:
            return StepPlan(batch_k=0, per_request_k={})

        batch_k = max(max(per_request_k.values()), self.min_batch_k)
        batch_k = min(batch_k, self.max_k)
        if not self.per_request:
            per_request_k = {req_id: batch_k for req_id in per_request_k}
        return StepPlan(batch_k=batch_k, per_request_k=per_request_k)

    def record_verification(self, req_id: str, num_drafted: int, num_accepted: int) -> None:
        """Feeds one verification result back to the request's controller.

        A request that just finished prefill may be verified before it is planned,
        so an unseen request id is registered here instead of being dropped.
        """
        if num_drafted <= 0:
            return
        self._known_requests.add(req_id)
        self.adapter.on_verification_completed(
            request_id=req_id,
            num_accepted=num_accepted,
            num_drafted=num_drafted,
        )

    def finish_request(self, req_id: str) -> Optional[dict]:
        """Releases the request session and appends its telemetry as one JSONL line."""
        if req_id not in self._known_requests:
            return None
        self._known_requests.discard(req_id)
        telemetry = self.adapter.end_request(req_id)
        if self.telemetry_path:
            record = dict(telemetry)
            record["finished_at"] = time.time()
            with open(self.telemetry_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        return telemetry

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "AdaptiveBudgetPlanner":
        """Builds a planner from ADAPTIVE_* environment variables (read once at server start)."""
        env = os.environ if environ is None else environ

        raw_kwargs = env.get("ADAPTIVE_CONTROLLER_KWARGS", "").strip()
        try:
            controller_kwargs = json.loads(raw_kwargs) if raw_kwargs else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"ADAPTIVE_CONTROLLER_KWARGS is not valid JSON: {exc}") from exc
        if not isinstance(controller_kwargs, dict):
            raise ValueError("ADAPTIVE_CONTROLLER_KWARGS must be a JSON object")

        if "ADAPTIVE_ENABLE_VRAM_GUARD" in env:
            controller_kwargs["enable_vram_guard"] = _env_flag(env["ADAPTIVE_ENABLE_VRAM_GUARD"])

        return cls(
            adapter=AdaptiveProposerAdapter(controller_kwargs=controller_kwargs),
            max_k=int(env.get("ADAPTIVE_MAX_K", "4")),
            min_batch_k=int(env.get("ADAPTIVE_MIN_BATCH_K", "0")),
            per_request=_env_flag(env.get("ADAPTIVE_PER_REQUEST", "1")),
            telemetry_path=env.get("ADAPTIVE_TELEMETRY_PATH") or None,
        )


def _env_flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
