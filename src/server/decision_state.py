"""Immutable, host-only transport for the SECP source-feasibility gates.

These values do not select K or certify a consumer profile. Unknown runtime
identity and worker-only information stay unknown; no CUDA context is created.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import sys
from typing import Optional


CONTEXT_SEMANTICS = "scheduler_num_computed_tokens_post_schedule_v1"
RUNTIME_SCHEMA = "runtime_identity_snapshot_v1"
STATE_SCHEMA = "decision_state_snapshot_v1"
CONTEXT_BUCKETS = ((0, 256), (256, 1024), (1024, 2048), (2048, 4096))
OBSERVED_FIELDS = (
    "gpu_name", "sm_architecture", "vram_total_mb", "vllm_version",
    "torch_version", "cuda_version", "driver_version", "effective_dtype",
    "effective_quantization",
)
CONFIGURED_FIELDS = (
    "target_model", "target_revision", "draft_model", "draft_revision",
    "draft_method", "scheduler_class", "executor_class", "tensor_parallel_size",
    "async_scheduling", "enforce_eager", "compilation_mode", "prefix_caching",
    "chunked_prefill", "max_model_len", "max_num_seqs", "gpu_memory_utilization",
    "cudagraph_mode", "cache_dtype", "cache_block_size", "speculative_ceiling",
    "disable_padded_drafter_batch", "max_num_batched_tokens",
)
MODE_FIELDS = (
    "scheduler_class", "executor_class", "tensor_parallel_size",
    "async_scheduling", "enforce_eager", "compilation_mode", "cudagraph_mode",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _pairs(value):
    """Detach caller-owned containers and accept only immutable JSON scalars."""
    pairs = tuple(tuple(pair) for pair in value)
    if any(len(pair) != 2 for pair in pairs) or len(dict(pairs)) != len(pairs):
        raise ValueError("Snapshot pairs require unique keys")
    normalized = []
    for key, item in pairs:
        if isinstance(item, str):
            item = str(item)
        if not isinstance(key, str) or type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("Snapshot values must be JSON scalars")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Snapshot values must be finite")
        normalized.append((key, item))
    return tuple(normalized)


@dataclass(frozen=True)
class RuntimeIdentitySnapshotV1:
    observed: tuple = ()
    configured: tuple = ()
    schema_version: str = RUNTIME_SCHEMA

    def __post_init__(self):
        object.__setattr__(self, "observed", _pairs(self.observed))
        object.__setattr__(self, "configured", _pairs(self.configured))

    @property
    def unknown_fields(self):
        observed, configured = dict(self.observed), dict(self.configured)
        return tuple(
            ["observed." + k for k in OBSERVED_FIELDS if observed.get(k) is None]
            + ["configured." + k for k in CONFIGURED_FIELDS if configured.get(k) is None]
        )

    @property
    def status(self):
        return "SUPPORTED" if self.schema_version == RUNTIME_SCHEMA and not self.unknown_fields else "UNSUPPORTED"

    @property
    def identity_key(self):
        return _digest(asdict(self)) if self.status == "SUPPORTED" else None

    @property
    def runtime_mode_id(self):
        configured = dict(self.configured)
        if any(configured.get(k) is None for k in MODE_FIELDS):
            return None
        return "runtime_mode_v1:" + _digest({k: configured[k] for k in MODE_FIELDS})

    def to_record(self):
        return dict(asdict(self), status=self.status, unknown_fields=self.unknown_fields,
                    identity_key=self.identity_key, runtime_mode_id=self.runtime_mode_id)


def _revision(model):
    resolved = getattr(getattr(model, "hf_config", None), "_commit_hash", None)
    configured = getattr(model, "revision", None)
    for value in (resolved, configured):
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value):
            return value
    return None


def observe_runtime_identity(config, *, runner=None, scheduler_class=None, executor_class=None,
                             torch_module=None, vllm_module=None):
    """Read already-loaded runtime/config metadata, never import/init CUDA.

    Configured quantization is not proof of a loaded quantizer, and no current
    driver-version observation is installed. Those observed fields stay null.
    """
    model = getattr(config, "model_config", None)
    spec = getattr(config, "speculative_config", None)
    draft = getattr(spec, "draft_model_config", None)
    scheduler = getattr(config, "scheduler_config", None)
    parallel = getattr(config, "parallel_config", None)
    cache = getattr(config, "cache_config", None)
    compilation = getattr(config, "compilation_config", None)
    configured = dict(
        target_model=getattr(model, "model", None), target_revision=_revision(model),
        draft_model=getattr(draft, "model", None), draft_revision=_revision(draft),
        draft_method=getattr(spec, "method", None), scheduler_class=scheduler_class,
        executor_class=executor_class, tensor_parallel_size=getattr(parallel, "tensor_parallel_size", None),
        async_scheduling=getattr(scheduler, "async_scheduling", None),
        enforce_eager=getattr(model, "enforce_eager", None),
        compilation_mode=getattr(compilation, "mode", None),
        prefix_caching=getattr(cache, "enable_prefix_caching", None),
        chunked_prefill=getattr(scheduler, "enable_chunked_prefill", None),
        max_model_len=getattr(model, "max_model_len", None),
        max_num_seqs=getattr(scheduler, "max_num_seqs", None),
        gpu_memory_utilization=getattr(cache, "gpu_memory_utilization", None),
        cudagraph_mode=(str(compilation.cudagraph_mode)
                        if getattr(compilation, "cudagraph_mode", None) is not None else None),
        cache_dtype=getattr(cache, "cache_dtype", None),
        cache_block_size=getattr(cache, "block_size", None),
        speculative_ceiling=getattr(spec, "num_speculative_tokens", None),
        disable_padded_drafter_batch=getattr(spec, "disable_padded_drafter_batch", None),
        max_num_batched_tokens=getattr(scheduler, "max_num_batched_tokens", None),
        configured_quantization=getattr(model, "quantization", None),
        configured_dtype=str(model.dtype) if getattr(model, "dtype", None) is not None else None,
    )
    # Enum values in the pinned compilation config are integer modes.
    if configured["compilation_mode"] is not None:
        try:
            configured["compilation_mode"] = int(configured["compilation_mode"])
        except (TypeError, ValueError):
            configured["compilation_mode"] = None
    torch_module = torch_module if torch_module is not None else sys.modules.get("torch")
    vllm_module = vllm_module if vllm_module is not None else sys.modules.get("vllm")
    observed = {k: None for k in OBSERVED_FIELDS}
    observed.update(vllm_version=getattr(vllm_module, "__version__", None),
                    torch_version=getattr(torch_module, "__version__", None),
                    cuda_version=getattr(getattr(torch_module, "version", None), "cuda", None))
    runner_model = getattr(runner, "model_config", None)
    if getattr(runner_model, "dtype", None) is not None:
        observed["effective_dtype"] = str(runner_model.dtype)
    cuda = getattr(torch_module, "cuda", None)
    try:
        if cuda is not None and cuda.is_initialized():
            device = cuda.current_device()
            major, minor = cuda.get_device_capability(device)
            observed.update(gpu_name=cuda.get_device_name(device),
                            sm_architecture=f"sm_{major}{minor}",
                            vram_total_mb=cuda.get_device_properties(device).total_memory / 2**20)
    except Exception:
        # Observation failure is an unsupported identity, not startup failure.
        observed.update(gpu_name=None, sm_architecture=None, vram_total_mb=None)
    return RuntimeIdentitySnapshotV1(tuple(observed.items()), tuple(configured.items()))


@dataclass(frozen=True)
class DecisionStateSnapshotV1:
    scheduler_sequence: int
    request_incarnations: tuple
    full_scheduled_request_ids: tuple
    eligible_decode_request_ids: tuple
    pure_decode: bool
    has_scheduled_prefill: bool
    old_verification_width_by_request: tuple
    host_context_tokens_after_schedule_by_request: tuple
    runtime_mode_id: Optional[str]
    schema_version: str = STATE_SCHEMA
    context_snapshot_semantics: str = CONTEXT_SEMANTICS
    unsupported_reasons: tuple = ()

    def __post_init__(self):
        for field in ("request_incarnations", "old_verification_width_by_request",
                      "host_context_tokens_after_schedule_by_request"):
            object.__setattr__(self, field, _pairs(getattr(self, field)))
        for field in ("full_scheduled_request_ids", "eligible_decode_request_ids", "unsupported_reasons"):
            object.__setattr__(self, field, tuple(getattr(self, field)))

    @property
    def status(self):
        full = self.full_scheduled_request_ids
        rows = (dict(self.request_incarnations), dict(self.old_verification_width_by_request),
                dict(self.host_context_tokens_after_schedule_by_request))
        incomplete = any(set(values) != set(full) for values in rows)
        contexts = rows[2]
        invalid = (self.schema_version != STATE_SCHEMA or self.context_snapshot_semantics != CONTEXT_SEMANTICS
                   or type(self.scheduler_sequence) is not int or self.scheduler_sequence <= 0
                   or len(set(full)) != len(full)
                   or not set(self.eligible_decode_request_ids) <= set(full)
                   or type(self.pure_decode) is not bool or type(self.has_scheduled_prefill) is not bool
                   or self.pure_decode == self.has_scheduled_prefill
                   or self.runtime_mode_id is None or incomplete
                   or any(type(contexts.get(rid)) is not int or contexts[rid] < 0 for rid in full))
        return "UNSUPPORTED" if self.unsupported_reasons or invalid else "SUPPORTED"

    def to_record(self):
        return dict(asdict(self), status=self.status)


def capture_decision_state(sequence, requests, eligible_ids, old_widths, incarnations, runtime_mode_id):
    """Called after native scheduling accounting, before new placeholders/K."""
    full = tuple(r.request_id for r in requests)
    has_prefill = any(r.is_prefill_chunk for r in requests)
    contexts, reasons = [], []
    for request in requests:
        value = getattr(request, "num_computed_tokens", None)
        if type(value) is not int or value < 0:
            value = None
            reasons.append("missing_host_context:" + request.request_id)
        contexts.append((request.request_id, value))
    if runtime_mode_id is None:
        reasons.append("unknown_runtime_mode")
    return DecisionStateSnapshotV1(
        sequence, tuple((rid, incarnations[rid]) for rid in full), full, tuple(eligible_ids),
        not has_prefill, has_prefill, tuple((rid, old_widths[rid]) for rid in full),
        tuple(contexts), runtime_mode_id, unsupported_reasons=tuple(reasons),
    )


def context_bucket(tokens):
    if type(tokens) is not int:
        return None
    for low, high in CONTEXT_BUCKETS:
        if low <= tokens < high:
            return f"[{low},{high})"
    return None


@dataclass(frozen=True)
class CandidateDecisionStateV1:
    runtime_identity_key: Optional[str]
    pure_decode_state: bool
    eligible_decode_rows: int
    old_verification_width_by_request: tuple
    candidate_future_width_by_request: tuple
    candidate_shared_worker_k: int
    context_bucket_by_request: tuple
    runtime_mode_id: Optional[str]
    unsupported_reasons: tuple = ()

    @property
    def status(self):
        return "UNSUPPORTED" if self.unsupported_reasons else "SUPPORTED"
