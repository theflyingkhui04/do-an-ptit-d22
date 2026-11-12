"""Process-local provenance for pinned V1/UniProc scalar draft consumption.

The verifier output is an identity anchor, not the producer of its drafts.
Only the executor's observed row-copy binding can supply a producer origin.
No token values, GPU tensors or mutable Requests are stored in origins.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from weakref import ref


@dataclass(frozen=True)
class DraftOrigin:
    request_id: str
    incarnation: int
    plan_sequence: int | None
    schedule_sequence: int
    request_k: int
    batch_k: int


@dataclass
class Execution:
    output: object  # Strong identity witness; released at retirement/error.
    producers: object
    consumed: dict = field(default_factory=dict)


class LineageLedger:
    def __init__(self):
        self.executions = {}
        self.active = {}
        self._incarnation = 0
        self.generation = None
        self._executor = None

    def request_key(self, request):
        current = self.active.get(request.request_id)
        if current is None or current[0] is not request:
            self._incarnation += 1
            current = (request, self._incarnation)
            self.active[request.request_id] = current
        return current[1]

    def register(self, output, plan, sequence, requests):
        producers = {}
        for rid in output.num_scheduled_tokens:
            req = requests.get(rid)
            if req is not None and not req.is_prefill_chunk:
                producers[rid] = DraftOrigin(
                    rid, self.request_key(req),
                    plan.plan_sequence if rid in plan.per_request_k else None,
                    sequence, len(req.spec_token_ids), output.num_spec_tokens_to_schedule)
        self.executions[id(output)] = Execution(output, MappingProxyType(producers))

    def get(self, output):
        entry = self.executions.get(id(output))
        return entry if entry is not None and entry.output is output else None

    def produced(self, output, runner, zeros_only=False):
        self.generation = None
        entry = self.get(output)
        buffer = runner._draft_token_ids
        shape = getattr(buffer, "shape", ())
        rows = tuple(runner.input_batch.req_ids)
        if zeros_only or entry is None or len(shape) != 2 or shape[1] <= 0:
            return
        if shape[0] != len(rows) or shape[1] != output.num_spec_tokens_to_schedule:
            return
        origins = tuple(entry.producers.get(rid) for rid in rows)
        # Buffer id is only an immediate witness; each call stamps a NEW birth,
        # even if the allocator/proposer reuses the same tensor object.
        self.generation = (id(buffer), int(shape[1]), rows, origins)

    def prepared(self, output, runner, generation):
        entry = self.get(output)
        if entry is None:
            return
        entry.consumed.clear()
        if generation is None or runner.input_batch.prev_sampled_token_ids is None:
            return
        witness, width, rows, origins = generation
        if id(runner._draft_token_ids) != witness or runner.prev_num_spec_tokens != width:
            return
        for index, rid in enumerate(runner.input_batch.req_ids):
            count = len(output.scheduled_spec_decode_tokens.get(rid, ()))
            old = int(runner.prev_positions.np[index])
            if count <= 0 or count > width or not 0 <= old < len(rows):
                continue
            origin = origins[old]
            if (rows[old] == rid and origin is not None
                    and count <= origin.request_k):
                entry.consumed[rid] = (origin, count)

    def finish_request(self, request):
        current = self.active.get(request.request_id)
        if current is not None and current[0] is request:
            self.active.pop(request.request_id)
        # Remove all references for this request without affecting other rows.
        for entry in self.executions.values():
            entry.consumed.pop(request.request_id, None)
            entry.producers = MappingProxyType({rid: origin for rid, origin in entry.producers.items()
                                               if rid != request.request_id})
        if self.generation is not None:
            witness, width, rows, origins = self.generation
            remaining = tuple(None if rid == request.request_id else origin
                              for rid, origin in zip(rows, origins))
            self.generation = (witness, width, rows, remaining) if any(remaining) else None

    def retire(self, output):
        if self.get(output) is not None:
            del self.executions[id(output)]

    def clear(self):
        self.executions.clear()
        self.active.clear()
        self.generation = None


# Config identity scopes the same-process executor/scheduler rendezvous.
# Explicit shutdown deregisters it; there is no global "latest engine".
_ledgers = {}


def ledger_for(config):
    if config is None:  # isolated tests / no supported executor: fail closed
        return LineageLedger()
    key = id(config)
    if key not in _ledgers:
        _ledgers[key] = (config, LineageLedger())
    assert _ledgers[key][0] is config
    return _ledgers[key][1]


def register_lineage_executor(config, executor):
    """Publish readiness after the executor installs its validated observer.

    Pinned EngineCore constructs the executor before the scheduler, using the
    identical config object. Registration is engine-local, not a class-name or
    environment-variable promise. A weak owner prevents stale readiness after
    executor destruction; shutdown removes the registration with the ledger.
    """
    if config is None or executor.vllm_config is not config:
        raise RuntimeError("Lineage executor registration requires its exact VllmConfig")
    ledger = ledger_for(config)
    owner = ledger._executor() if ledger._executor is not None else None
    if owner is not None and owner is not executor:
        raise RuntimeError("A lineage executor is already registered for this VllmConfig")
    if executor._lineage is not ledger:
        raise RuntimeError("Lineage executor registration requires its exact ledger")
    ledger._executor = ref(executor)


def require_lineage_executor(config):
    """Reject unpaired adaptive startup without creating a permissive ledger."""
    pair = _ledgers.get(id(config)) if config is not None else None
    ledger = pair[1] if pair is not None and pair[0] is config else None
    owner = ledger._executor() if ledger is not None and ledger._executor is not None else None
    if (owner is None or owner.vllm_config is not config
            or owner._lineage is not ledger):
        backend = getattr(getattr(config, "parallel_config", None),
                          "distributed_executor_backend", "<missing VllmConfig/backend>")
        raise RuntimeError(
            "Adaptive async lineage requires a lineage-capable executor initialized "
            "with this exact VllmConfig; no ready executor registration was found. "
            f"Observed distributed_executor_backend={backend!r}. Configure "
            "src.server.lineage_executor.LineageUniProcExecutor.")
    return ledger


def release_ledger(config):
    pair = _ledgers.pop(id(config), None)
    if pair is not None:
        pair[1]._executor = None
        pair[1].clear()
