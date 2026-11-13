"""Explicit diagnostic-only hooks for hash-pinned vLLM 0.30.0.

Never selected by canonical serving. No file/model/tensor changes on import.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from weakref import WeakValueDictionary

from experiments.m4_cache_probe import MEASUREMENT, token_hash


# UniProcExecutor shares the EngineCore's config object with its scheduler and
# worker. Weak, identity-scoped ownership never patches a vLLM class/global.
_schedulers = WeakValueDictionary()


class _InstanceHooks:
    def __init__(self):
        self.originals = []

    def replace(self,owner,name,replacement):
        self.originals.append((owner,name,name in vars(owner),vars(owner).get(name)))
        setattr(owner,name,replacement)

    def restore(self):
        failed = []
        for owner,name,present,original in reversed(self.originals):
            try:
                if present:
                    setattr(owner,name,original)
                elif name in vars(owner):
                    delattr(owner,name)
            except BaseException as exc:
                failed.append((owner,name,present,original,exc))
        self.originals = [entry[:4] for entry in reversed(failed)]
        if failed:
            raise RuntimeError('Diagnostic instance-hook restoration failed') from failed[0][4]


def verify_runtime_sources() -> dict:
    import vllm
    if vllm.__version__ != '0.30.0':
        raise RuntimeError('Diagnostic requires exact vLLM 0.30.0')
    root = Path(vllm.__file__).resolve().parent.parent
    return verify_source_tree(root)


def verify_source_tree(root: Path,manifest=None) -> dict:
    """Byte checks are CPU-only and can be exercised without importing vLLM."""
    if manifest is None:
        manifest = json.loads(Path(__file__).with_name('m4_cache_probe_sources.json').read_text(encoding='utf8'))
    for record in manifest['files']:
        if hashlib.sha256((root/record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f"Unreviewed pinned source: {record['path']}")
    return manifest


def configuration() -> dict:
    path = os.environ.get('M4_CACHE_PROBE_CONFIG')
    if not path:
        raise RuntimeError('Explicit diagnostic configuration required')
    config = json.loads(Path(path).read_text(encoding='utf8'))
    if token_hash(config['forced_input_token_ids']) != config['input_token_ids_sha256']:
        raise RuntimeError('Forced input identity mismatch')
    return config


def observe(config: dict, event: dict) -> None:
    with Path(config['cache_trace']).open('a',encoding='utf8') as fh:
        fh.write(json.dumps(event,allow_nan=False)+'\n')


class CacheProbeSchedulerMixin:
    """Delegate every scheduling/cache operation; seed only the forced grammar."""
    def __init__(self,*args,**kwargs):
        verify_runtime_sources()
        self._m4_config = configuration()
        self._m4_seeded = set()
        super().__init__(*args,**kwargs)
        if self.vllm_config.speculative_config is not None or not self.cache_config.enable_prefix_caching:
            raise RuntimeError('Vanilla with enabled prefix caching required')
        if id(self.vllm_config) in _schedulers:
            raise RuntimeError('Diagnostic scheduler already registered for this engine')
        self._m4_cache_hooks = _InstanceHooks()
        self._m4_active = True
        original = self.kv_cache_manager.get_computed_blocks

        def measured_lookup(request):
            result = original(request)
            observe(self._m4_config,dict(event='cache_lookup',request_id=request.request_id,
                input_token_count=len(request.prompt_token_ids),
                input_token_ids_sha256=token_hash(request.prompt_token_ids),
                prefix_cache_enabled=self.kv_cache_manager.enable_caching,
                skip_reading_prefix_cache=request.skip_reading_prefix_cache,
                cached_tokens=result[1],shared_prefix_boundary=result[2],
                block_size=self.block_size,
                block_ids=result[0].get_block_ids()))
            return result
        try:
            self._m4_cache_hooks.replace(self.kv_cache_manager,'get_computed_blocks',measured_lookup)
            _schedulers[id(self.vllm_config)] = self
        except BaseException:
            self.m4_probe_cache_uninstall()
            raise

    def m4_probe_cache_uninstall(self):
        self._m4_active = False
        self._m4_seeded.clear()
        self._m4_cache_hooks.restore()
        self._m4_config = None
        if _schedulers.get(id(self.vllm_config)) is self:
            del _schedulers[id(self.vllm_config)]

    def shutdown(self):
        try:
            self.m4_probe_cache_uninstall()
        finally:
            super().shutdown()

    def _update_after_schedule(self,scheduler_output):
        super()._update_after_schedule(scheduler_output)
        if not self._m4_active:
            return
        for rid in scheduler_output.num_scheduled_tokens:
            request = self.requests[rid]
            if (rid in self._m4_seeded or request.is_prefill_chunk
                    or request.prompt_token_ids != self._m4_config['forced_input_token_ids']):
                continue
            prefix = self._m4_config['generated_prefix_token_ids']
            if self._m4_config['historical_request'].get('response_format') == {'type':'json_object'}:
                structured = request.structured_output_request
                grammar = structured.grammar if structured else None
                if grammar is None or isinstance(grammar,Exception):
                    raise RuntimeError('Compiled JSON grammar unavailable at forced boundary')
                if not grammar.accept_tokens(rid,prefix):
                    raise RuntimeError('Preserved generated prefix rejected by JSON grammar')
                structured.reasoning_ended = True
            self._m4_seeded.add(rid)
            observe(self._m4_config,dict(event='forced_boundary_state',request_id=rid,
                generated_prefix_count=len(prefix),input_token_ids_sha256=token_hash(request.prompt_token_ids),
                grammar_seeded=bool(request.structured_output_request)))

    def _free_request(self,request,*args,**kwargs):
        self._m4_seeded.discard(request.request_id)
        return super()._free_request(request,*args,**kwargs)


def logits_summary(vector, processed=None, tracked_tokens=(4164,3752)) -> tuple:
    """Called after request quiescence; never synchronize a live forward."""
    import torch
    dtype = str(vector.dtype)
    if dtype not in ('torch.float16','torch.bfloat16','torch.float32'):
        raise RuntimeError('Unexpected raw-logit dtype; avoid lossy comparison conversion')
    cpu = vector.detach().to(device='cpu',dtype=torch.float32)
    finite = torch.isfinite(cpu)
    nan_count = int(torch.isnan(cpu).sum())
    inf_count = int(torch.isinf(cpu).sum())
    if not bool(finite.all()):
        raise RuntimeError(f'Nonfinite raw logits: nan={nan_count}, inf={inf_count}')
    values,indices = torch.topk(cpu,20)
    top = [dict(token_id=int(i),logit=float(v)) for i,v in zip(indices,values)]
    argmax = int(cpu.argmax())
    # torch.topk may order exact ties differently from torch.argmax. Report
    # the actual greedy winner first without changing any model tensor.
    top = ([dict(token_id=argmax,logit=float(cpu[argmax]))]
           + [entry for entry in top if entry['token_id']!=argmax])[:20]
    result = dict(measurement_point=MEASUREMENT,dtype=dtype,
        argmax_token_id=argmax,top_tokens=top,
        logit_token_4164=float(cpu[4164]),logit_token_3752=float(cpu[3752]),
        historical_competing_logits={str(t):float(cpu[t]) for t in tracked_tokens},
        difference_4164_minus_3752=float(cpu[4164])-float(cpu[3752]),
        top1_minus_top2=float(values[0])-float(values[1]),nan_count=nan_count,inf_count=inf_count)
    if processed is not None:
        p = processed.detach().to(device='cpu',dtype=torch.float32)
        result['grammar_masked_argmax_token_id'] = int(p.argmax())
        result['grammar_masked_measurement_point'] = 'GPUModelRunner._sample entry; after grammar, before sampler'
    return result,cpu


class CacheProbeWorkerExtension:
    """Only explicit collective_rpc calls install/arm raw-logit observation."""
    def m4_probe_install(self):
        verify_runtime_sources()
        config = configuration()
        runner = self.model_runner
        if (type(runner).__module__ != 'vllm.v1.worker.gpu_model_runner'
                or type(runner).__name__ != 'GPUModelRunner'
                or not runner.use_async_scheduling
                or self.vllm_config.speculative_config is not None):
            raise RuntimeError('Pinned Vanilla V1 async single-worker runner required')
        if hasattr(self,'_m4_probe'):
            raise RuntimeError('Probe already installed')
        scheduler = _schedulers.get(id(self.vllm_config))
        if scheduler is None or scheduler.vllm_config is not self.vllm_config:
            raise RuntimeError('Owning diagnostic UniProc scheduler unavailable')
        projection = runner.model.logits_processor
        if projection.scale != 1.0 or projection.soft_cap is not None or projection.logits_as_input:
            raise RuntimeError('Unexpected model-logit transform; measurement review required')
        self._m4_probe = dict(armed=None,pending=None,previous=None,output=None,repeats={},sampling_output=None,
            tracked_tokens=(config['reference_next_token_id'],config['candidate_next_token_id']))
        compute = runner.model.compute_logits
        execute = runner.execute_model
        sample = runner._sample
        sample_tokens = runner.sample_tokens

        def execution(output,*args,**kwargs):
            self._m4_probe['output'] = output
            try:
                return execute(output,*args,**kwargs)
            finally:
                self._m4_probe['output'] = None

        def raw_logits(*args,**kwargs):
            tensor = compute(*args,**kwargs)
            state = self._m4_probe
            armed,output = state['armed'],state['output']
            if armed is None or output is None or state['pending'] is not None:
                return tensor
            ids = runner.input_batch.req_ids
            if len(ids) != 1:
                raise RuntimeError('Diagnostic batching must contain exactly one request')
            rid = ids[0]
            request = runner.requests[rid]
            scheduled = output.num_scheduled_tokens.get(rid,0)
            computed = int(runner.input_batch.num_computed_tokens_cpu[0])
            if computed+scheduled != len(armed['tokens']):
                return tensor  # Intermediate chunk or a later decode position.
            if request.prompt_token_ids != armed['tokens'] or tensor.shape[0] != 1:
                raise RuntimeError('Raw-logit row is not the exact forced boundary')
            state['pending'] = dict(name=armed['name'],request_id=rid,output=output,
                raw=tensor[0].detach().clone(),processed=None,
                query_ids=runner.input_ids.gpu[:scheduled].detach().clone(),
                expected_query_ids=armed['tokens'][computed:computed+scheduled],
                computed_tokens_before=computed,scheduled_query_tokens=scheduled,
                input_token_ids_sha256=token_hash(request.prompt_token_ids))
            # Return the exact original tensor. Clone is read-only GPU storage;
            # no .item(), .cpu(), synchronize, sampling or cache mutation here.
            return tensor

        def before_sampling(logits,*args,**kwargs):
            pending = self._m4_probe['pending']
            if (pending is not None and pending['processed'] is None
                    and self._m4_probe['sampling_output'] is pending['output']):
                pending['processed'] = logits[0].detach().clone()
            return sample(logits,*args,**kwargs)

        def sampling(*args,**kwargs):
            # Pinned sample_tokens clears execute_model_state BEFORE _sample.
            # Preserve its scheduler-output identity across that exact call.
            state = runner.execute_model_state
            previous = self._m4_probe['sampling_output']
            self._m4_probe['sampling_output'] = state[0] if state is not None else None
            try:
                return sample_tokens(*args,**kwargs)
            finally:
                self._m4_probe['sampling_output'] = previous

        self._m4_hooks = _InstanceHooks()
        try:
            self._m4_hooks.replace(runner,'execute_model',execution)
            self._m4_hooks.replace(runner.model,'compute_logits',raw_logits)
            self._m4_hooks.replace(runner,'_sample',before_sampling)
            self._m4_hooks.replace(runner,'sample_tokens',sampling)
        except BaseException:
            self.m4_probe_uninstall()
            raise
        return dict(installed=True,measurement_point=MEASUREMENT,
            head_scale=projection.scale,head_soft_cap=projection.soft_cap,
            model_class=f'{type(runner.model).__module__}.{type(runner.model).__name__}',
            synchronization='Three boundary GPU clones (raw/masked logits and query IDs); CPU transfer/reductions after abort')

    def m4_probe_uninstall(self):
        # Always attempt both owners. Retain failed restoration records so a
        # repeated teardown can retry, but release every captured tensor now.
        try:
            try:
                if hasattr(self,'_m4_hooks'):
                    self._m4_hooks.restore()
            finally:
                if hasattr(self,'_m4_probe'):
                    self._m4_probe.clear()
                    del self._m4_probe
        finally:
            scheduler = _schedulers.get(id(self.vllm_config))
            if scheduler is not None and scheduler.vllm_config is self.vllm_config:
                scheduler.m4_probe_cache_uninstall()
        return dict(restored=True,captures_released=True,cache_observer_restored=True)

    def m4_probe_arm(self,name,tokens):
        if self._m4_probe['pending'] is not None:
            raise RuntimeError('Previous capture not collected')
        self._m4_probe['armed'] = dict(name=name,tokens=tokens)
        return True

    def m4_probe_collect(self):
        state = self._m4_probe
        pending = state['pending']
        if pending is None or pending['processed'] is None:
            raise RuntimeError('Missing paired raw/grammar-masked boundary measurement')
        query = pending['query_ids'].to(device='cpu').tolist()
        if query != pending['expected_query_ids']:
            raise RuntimeError('Actual GPU query tokens differ from preserved forced suffix')
        summary,cpu = logits_summary(pending['raw'],pending['processed'],state['tracked_tokens'])
        previous = state['previous']
        difference = None
        def compare(a,b):
            if a.shape != b.shape:
                raise RuntimeError('Vocabulary shape changed')
            delta = (a-b).abs()
            return dict(vocabulary_equal=True,max_absolute_difference=float(delta.max()),
                        mean_absolute_difference=float(delta.mean()),vocabulary_size=len(b))
        if previous is not None:
            difference = compare(previous,cpu)
        same_condition = state['repeats'].get(pending['name'])
        repeated = compare(same_condition,cpu) if same_condition is not None else None
        state['repeats'][pending['name']] = cpu
        state['previous'] = cpu
        state['pending'] = None
        state['armed'] = None
        return dict(logits=summary,vector_difference=difference,repeat_vector_difference=repeated,
            request_id=pending['request_id'],input_token_ids_sha256=pending['input_token_ids_sha256'],
            query_matches_forced_suffix=True,query_token_ids_sha256=token_hash(query),
            computed_tokens_before=pending['computed_tokens_before'],
            scheduled_query_tokens=pending['scheduled_query_tokens'])


# Resolving this class is an explicit engine CLI/config opt-in. Canonical
# vLLM/default scheduler classes and project serving files are untouched.
def __getattr__(name):
    if name != 'CacheProbeScheduler':
        raise AttributeError(name)
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    class CacheProbeScheduler(CacheProbeSchedulerMixin,AsyncScheduler):
        pass
    return CacheProbeScheduler
