"""Process-local provenance for pinned V1/UniProc scalar draft consumption.

The verifier output is an identity anchor, not the producer of its drafts.
Only the executor's observed row-copy binding can supply a producer origin.
No token values, GPU tensors or mutable Requests are stored in origins.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from weakref import WeakSet, WeakValueDictionary, ref
from threading import RLock, current_thread
from contextlib import contextmanager
from contextvars import ContextVar
from src.server.lineage_diagnostics import observe


@dataclass(frozen=True)
class DraftOrigin:
    request_id: str
    incarnation: int
    plan_sequence: int | None
    schedule_sequence: int
    request_k: int
    batch_k: int
    physical_draft_width: int | None = field(default=None, compare=False)
    source_base_k: int | None = None
    source_decision_mode: str | None = None

    @property
    def planned_worker_k(self) -> int:
        return self.batch_k


@dataclass
class Execution:
    output: object  # Strong identity witness; released at retirement/error.
    producers: object
    consumed: dict = field(default_factory=dict)
    schedule_sequence: int | None = None


class LineageLedger:
    def __init__(self):
        self.executions = {}
        self.active = {}
        self._incarnation = 0
        self.generation = None
        self.on_policy_draft = None
        self._issued_origins = WeakValueDictionary()
        self._counted_origins = WeakSet()

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
            if self.on_policy_draft is not None and rid not in plan.per_request_k:
                continue
            if req is not None and not req.is_prefill_chunk:
                producers[rid] = DraftOrigin(
                    rid, self.request_key(req),
                    plan.plan_sequence if rid in plan.per_request_k else None,
                    sequence, len(req.spec_token_ids), output.num_spec_tokens_to_schedule,
                    source_base_k=getattr(plan, "per_request_base_k", {}).get(rid),
                    source_decision_mode=getattr(plan, "per_request_source_modes", {}).get(rid))
        self.executions[id(output)] = Execution(output, MappingProxyType(producers), schedule_sequence=sequence)

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
        if shape[0] != len(rows):
            return
        origins = tuple(entry.producers.get(rid) for rid in rows)
        for origin in origins:
            if origin is not None:
                # Measurement is attached to the same immutable identity witness.
                # Excluding it from equality/hash preserves weak-set membership.
                object.__setattr__(origin, "physical_draft_width", int(shape[1]))
                if self.on_policy_draft is not None:
                    self.on_policy_draft(origin, int(shape[1]))
        if shape[1] != output.num_spec_tokens_to_schedule:
            return  # Preserve mismatch telemetry without accepting its feedback.
        # Buffer id is only an immediate witness; each call stamps a NEW birth,
        # even if the allocator/proposer reuses the same tensor object.
        self.generation = (id(buffer), int(shape[1]), rows, origins)
        if self.on_policy_draft is not None:
            for origin in origins:
                if origin is not None:
                    self._issued_origins[(origin.incarnation, origin.schedule_sequence)] = origin
        observe("draft_produced", batch_k=int(shape[1]),
                rows=[dict(request_id=rid, plan_sequence=origin.plan_sequence,
                           schedule_sequence=origin.schedule_sequence,
                           request_k=origin.request_k, batch_k=origin.batch_k)
                      for rid, origin in zip(rows, origins) if origin is not None])

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

    def claim_policy_feedback(self, origin: DraftOrigin) -> str | None:
        """Accept a real birth once while any in-flight output still retains it."""
        key = (origin.incarnation, origin.schedule_sequence)
        if self._issued_origins.get(key) is not origin:
            return "unknown-producer-origin"
        if origin in self._counted_origins:
            return "duplicate-origin"
        self._counted_origins.add(origin)
        return None

    def retire(self, output):
        if self.get(output) is not None:
            del self.executions[id(output)]

    def clear(self):
        self.executions.clear()
        self.active.clear()
        self.generation = None
        self._issued_origins.clear()
        self._counted_origins.clear()


# Process-local rendezvous only: config identity indexes a slot, never ownership.
# Pinned EngineCore constructs exactly one scheduler after its executor on the
# same initialization thread and inside one project-owned startup boundary.
# A publication can be consumed exactly once by that attempt. Live same-config
# overlap is rejected; failure revokes the orphan and success requires release.
_registry = {}
_registry_lock = RLock()
_startup = ContextVar("lineage_engine_startup", default=None)


@dataclass(frozen=True, eq=False)
class StartupAttempt:
    config: object
    publisher_thread: object
    token: object


@dataclass
class _StartupState:
    attempt: StartupAttempt
    active: bool = True
    registration: object = None


@contextmanager
def _engine_startup(config):
    """One EngineCore.__init__ invocation, never a config/thread-scoped lease.

    The pinned constructor wrapper owns this boundary. Failed construction and
    successful construction without consumption both revoke their exact handle.
    A copied context cannot reactivate a completed attempt. Nested constructors
    get fresh identities and restore the outer context in finally.
    """
    with _registry_lock:
        state = _StartupState(StartupAttempt(config, current_thread(), object()))
        context_token = _startup.set(state)
    succeeded = False
    observe("attempt_created", attempt=state.attempt)
    try:
        yield state.attempt
        succeeded = True
    finally:
        try:
            with _registry_lock:
                state.active = False
                registration = state.registration
                current = _registry.get(id(config))
                if registration is not None and (
                        not succeeded or (current is not None
                        and current.registration is registration and not current.consumed)):
                    release_ledger(registration)
        finally:
            _startup.reset(context_token)
            observe("attempt_closed", attempt=state.attempt, registration=state.registration,
                    succeeded=succeeded, context_reset=_startup.get() is not state,
                    registration_retained=(state.registration is not None
                        and (current := _registry.get(id(config))) is not None
                        and current.registration is state.registration),
                    consumed=(current.consumed if state.registration is not None
                              and (current := _registry.get(id(config))) is not None
                              and current.registration is state.registration else False))


@dataclass(frozen=True, eq=False)
class LineageRegistration:
    config: object
    ledger: LineageLedger
    owner: object  # weak reference; never retain the executor through the registry
    publisher_thread: object
    token: object
    attempt: StartupAttempt


@dataclass
class _RegistrationState:
    registration: LineageRegistration
    consumed: bool = False


def _expire_registration(key, token):
    # Callback closes over only an integer key and an opaque generation token.
    # Neither the config, owner nor registration participates in a strong cycle.
    with _registry_lock:
        state = _registry.get(key)
        if state is not None and state.registration.token is token:
            del _registry[key]
            state.registration.ledger.clear()


def register_lineage_executor(config, executor):
    """Publish one successfully initialized executor generation; return its handle.

    Publication allocates a fresh ledger; the executor retains it from the handle.
    Only that handle authorizes teardown. A config/backend name is not evidence
    of ownership, and an active registration cannot be replaced implicitly.
    """
    if config is None or executor.vllm_config is not config:
        raise RuntimeError("Lineage executor registration requires its exact VllmConfig")
    with _registry_lock:
        startup = _startup.get()
        if (startup is None or not startup.active
                or startup.attempt.config is not config
                or startup.attempt.publisher_thread is not current_thread()):
            raise RuntimeError("Lineage publication requires an active EngineCore startup attempt")
        if startup.registration is not None:
            raise RuntimeError("A lineage generation was already published by this startup attempt")
        if id(config) in _registry:
            raise RuntimeError("A lineage executor generation is already registered for this VllmConfig; release it first")
        ledger = LineageLedger()
        token = object()
        id_key = id(config)
        owner = ref(executor, lambda unused: _expire_registration(id_key, token))
        registration = LineageRegistration(config, ledger, owner, current_thread(), token,
                                           startup.attempt)
        _registry[id_key] = _RegistrationState(registration)
        startup.registration = registration
        observe("executor_published", registration=registration,
                executor_class=f"{type(executor).__module__}.{type(executor).__name__}")
        return registration


def require_lineage_executor(config):
    """Atomically consume readiness for one pinned executor-before-scheduler init.

    No ledger is created on failure. Keeping a bound scheduler/owner alive cannot
    make its consumed readiness reusable by another engine sharing the config.
    """
    with _registry_lock:
        state = _registry.get(id(config)) if config is not None else None
        registration = state.registration if state is not None else None
        owner = registration.owner() if registration is not None else None
        reason = "no ready executor registration was found"
        if (owner is not None and registration.config is config
                and owner.vllm_config is config
                and getattr(owner, "_lineage", None) is registration.ledger):
            if state.consumed:
                reason = "registration already consumed by another scheduler initialization"
            elif registration.publisher_thread is not current_thread():
                reason = "registration belongs to another initialization thread"
            elif ((startup := _startup.get()) is None or not startup.active
                    or startup.attempt is not registration.attempt):
                reason = "registration belongs to another EngineCore startup attempt"
            else:
                state.consumed = True
                observe("scheduler_consumed", registration=registration, consumed=state.consumed,
                        same_active_attempt=startup.attempt is registration.attempt)
                return registration
        backend = getattr(getattr(config, "parallel_config", None),
                          "distributed_executor_backend", "<missing VllmConfig/backend>")
        raise RuntimeError(
            "Adaptive async lineage requires a lineage-capable executor initialized "
            f"with this exact VllmConfig; {reason}. "
            f"Observed distributed_executor_backend={backend!r}. Configure "
            "src.server.lineage_executor.LineageUniProcExecutor.")


def release_ledger(registration):
    """Revoke/clear only the supplied generation, including on repeated teardown."""
    if registration is None:  # executor initialization failed before publication
        return
    if not isinstance(registration, LineageRegistration):
        raise TypeError("Lineage release requires a registration handle, not a config")
    with _registry_lock:
        state = _registry.get(id(registration.config))
        if state is not None and state.registration is registration:
            del _registry[id(registration.config)]
        # A stale handle can clear only its own ledger, never a replacement's.
        registration.ledger.clear()
