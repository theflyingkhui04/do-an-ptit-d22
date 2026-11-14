"""Opt-in execution-history diagnostic; CPU contracts and boundary-only evidence.

No vLLM/torch import or hooks on import. This module never changes sampled IDs.
The device latch freezes the FIRST mismatching decision (or the requested final
boundary). Host output delivery may lag async execution; later in-flight forwards
cannot replace that evidence. Full vectors transfer only after request abort.
"""
from __future__ import annotations

import hashlib
import base64
from functools import lru_cache
import json
from pathlib import Path
import re
import struct
import sys
import time

from experiments.m4_cache_probe import MEASUREMENT, token_hash, cache_evidence, load_input
from experiments.m4_cache_probe_hooks import (
    CacheProbeWorkerExtension, CacheProbeSchedulerMixin, _InstanceHooks, _schedulers,
    configuration, verify_runtime_sources, logits_summary,
)
from experiments.m4_protocol import write_new
from experiments.m4_treatments import TARGET, TARGET_REVISION
from experiments.m4_head_input_probe import (HEAD_STAGE,head_contract,clone_head_input,
    serialize_head_input,analyze_head_input)

SCHEMA = 'm4.decode_trajectory.v1'
VOCABULARY_SIZE = 152064  # Frozen Qwen target; matches preserved probe vectors.
SEQUENTIAL = 'SEQUENTIAL_DECODE_COLD'
FORCED = 'FORCED_PREFILL_COLD'


@lru_cache(maxsize=1)
def _frozen_reference():
    # Private cache: callers never receive or mutate this dictionary.
    return load_input('W1','m0_04_001',549)


def validate_prepared_case(case, expected_source_commit=None):
    expected = _frozen_reference()
    if (case.get('diagnostic_mode') != 'decode_trajectory'
            or case.get('localization_stage') not in (None,HEAD_STAGE)
            or any(case.get(key) != value for key,value in expected.items())):
        raise ValueError('Prepared case differs from frozen historical trajectory')
    source = case.get('source_commit')
    if (not isinstance(source,str) or re.fullmatch('[0-9a-f]{40}',source) is None
            or case.get('instrumentation_commit') != source
            or not isinstance(case.get('run_id'),str) or not case['run_id']
            or (expected_source_commit is not None and source != expected_source_commit)):
        raise ValueError('Prepared source differs from executed/manifest SHA')


class TrajectorySchedulerMixin(CacheProbeSchedulerMixin):
    def __init__(self, *args, **kwargs):
        self._m4_grammar_meta = {}
        super().__init__(*args, **kwargs)

    def get_grammar_bitmask(self, output):
        result = super().get_grammar_bitmask(output)
        if self._m4_active and result is not None:
            meta = {}
            for rid in result.structured_output_request_ids:
                request = self.requests[rid]
                grammar = request.structured_output_request.grammar
                backend = self.structured_output_manager.backend
                meta[rid] = dict(backend_class=f'{type(backend).__module__}.{type(backend).__name__}',
                    grammar_class=f'{type(grammar).__module__}.{type(grammar).__name__}',
                    num_processed_tokens=getattr(grammar, 'num_processed_tokens', None),
                    reasoner_class=(None if self.structured_output_manager.reasoner_cls is None
                                    else str(self.structured_output_manager.reasoner_cls)),
                    reasoning_ended=request.structured_output_request.reasoning_ended)
            self._m4_grammar_meta[id(output)] = meta
        return result

    def m4_probe_cache_uninstall(self):
        try:
            super().m4_probe_cache_uninstall()
        finally:
            self._m4_grammar_meta.clear()


def __getattr__(name):
    if name != 'TrajectoryScheduler':
        raise AttributeError(name)
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    class TrajectoryScheduler(TrajectorySchedulerMixin, AsyncScheduler):
        pass
    return TrajectoryScheduler


def prefix_identity(prompt, generated):
    # The empty generation prefix is valid at generation index zero.
    return dict(prompt_token_ids_sha256=token_hash(prompt),
        generated_prefix_sha256=hashlib.sha256(
            json.dumps(generated, separators=(',', ':')).encode('ascii')).hexdigest(),
        logical_input_sha256=token_hash(prompt + generated),
        generation_index=len(generated))


def first_boundary(observed, reference, target):
    """Inspect emitted integer IDs; never decode text or force a continuation."""
    if type(target) is not int or not 0 <= target < len(reference):
        raise ValueError('Invalid trajectory target')
    if any(type(t) is not int or t < 0 for t in observed):
        raise ValueError('Invalid emitted trajectory tokens')
    for i, token in enumerate(observed[:target+1]):
        if token != reference[i] or i == target:
            return i
    return None


class FirstMismatchCapture:
    """Constant-size GPU logit storage; no per-step device-to-host reduction.

    Reference comparison uses the ACTUAL sampler return. A raw argmax can differ
    from it because structured-output masking happens before sampling.
    """
    def __init__(self, reference, target):
        if type(target) is not int or not 0 <= target < len(reference):
            raise ValueError('Invalid trajectory target')
        self.reference = reference
        self.target = target
        self.next_index = 0
        self.selected = None

    def record(self, index, raw, masked, sampled, head_input=None):
        import torch
        if index != self.next_index or index > self.target:
            raise RuntimeError('Non-contiguous sequential decision indices')
        if raw.ndim != 1 or raw.shape != masked.shape or sampled.numel() != 1:
            raise RuntimeError('Single Vanilla logit row/token required')
        self.next_index += 1
        if self.selected is None:
            self.selected = dict(raw=torch.zeros_like(raw), masked=torch.zeros_like(masked),
                index=torch.full((), -1, dtype=torch.int64, device=raw.device),
                token=torch.full((), -1, dtype=torch.int64, device=raw.device))
            if head_input is not None:
                self.selected['head_input'] = torch.zeros_like(head_input)
        s = self.selected
        if (('head_input' in s) != (head_input is not None)
                or (head_input is not None and (head_input.shape != s['head_input'].shape
                    or head_input.dtype != s['head_input'].dtype))):
            raise RuntimeError('LM-head input missing/shape/dtype changed within trajectory')
        token = sampled.reshape(())
        take = (s['index'] < 0) & ((token != self.reference[index]) | (index == self.target))
        for key, value in (('raw',raw), ('masked',masked), ('index',index), ('token',token)):
            s[key].copy_(torch.where(take, value, s[key]))
        if head_input is not None:
            s['head_input'].copy_(torch.where(take,head_input,s['head_input']))


class TrajectoryWorkerExtension(CacheProbeWorkerExtension):
    """Uses the existing instance-hook registry and exception-safe teardown."""
    def m4_trajectory_install(self):
        verify_runtime_sources()
        config = configuration()
        runner = self.model_runner
        scheduler = _schedulers.get(id(self.vllm_config))
        if (config.get('diagnostic_mode') != 'decode_trajectory'
                or type(runner).__module__ != 'vllm.v1.worker.gpu_model_runner'
                or type(runner).__name__ != 'GPUModelRunner'
                or not runner.use_async_scheduling
                or self.vllm_config.speculative_config is not None
                or scheduler is None or scheduler.vllm_config is not self.vllm_config):
            raise RuntimeError('Explicit pinned single-worker Vanilla trajectory engine required')
        projection = runner.model.logits_processor
        if projection.scale != 1.0 or projection.soft_cap is not None or projection.logits_as_input:
            raise RuntimeError('Unexpected model-logit transform')
        if hasattr(self, '_m4_probe'):
            raise RuntimeError('Diagnostic already installed')
        stage = config.get('localization_stage')
        if stage not in (None,HEAD_STAGE):
            raise RuntimeError('Unsupported localization stage')
        self._m4_probe = dict(armed=None, output=None, candidate=None, shape=None,
            sampling_output=None, metadata={}, queries={},localization_stage=stage,
            head_contract=head_contract(runner) if stage == HEAD_STAGE else None)
        self._m4_hooks = _InstanceHooks()
        execute, compute = runner.execute_model, runner.model.compute_logits
        sample_tokens, sample = runner.sample_tokens, runner._sample
        determine = runner._determine_batch_execution_and_padding

        def execution(output, *args, **kwargs):
            self._m4_probe['output'] = output
            self._m4_probe['candidate'] = None
            self._m4_probe['shape'] = None
            try:
                return execute(output, *args, **kwargs)
            finally:
                self._m4_probe['output'] = None

        def shape(*args, **kwargs):
            result = determine(*args, **kwargs)
            mode, desc, ubatch, _, _ = result
            self._m4_probe['shape'] = dict(cudagraph_mode=mode.name,
                padded_tokens=desc.num_tokens, padded_requests=desc.num_reqs,
                uniform_decode=desc.uniform, microbatching=ubatch)
            return result

        def raw_logits(*args, **kwargs):
            head_input, layout = None, None
            s = self._m4_probe
            armed, output = s['armed'], s['output']
            if s['localization_stage'] == HEAD_STAGE and armed is not None and output is not None:
                ids = runner.input_batch.req_ids
                if len(ids) != 1 or runner.requests[ids[0]].prompt_token_ids != armed['prompt']:
                    raise RuntimeError('LM-head boundary request isolation mismatch')
                computed = int(runner.input_batch.num_computed_tokens_cpu[0])
                end = computed + output.num_scheduled_tokens[ids[0]]
                index = end-len(armed['prompt']) if armed['name']==SEQUENTIAL else armed['target']
                if (0 <= index <= armed['target'] and
                        (armed['name']==SEQUENTIAL or end==len(armed['prompt']))):
                    # Before the original head call; never use a Python layer hook
                    # inside a compiled/CUDA-graph-replayed model forward.
                    head_input, layout = clone_head_input(args,kwargs,s['head_contract']['hidden_size'])
            tensor = compute(*args, **kwargs)
            s = self._m4_probe
            armed, output = s['armed'], s['output']
            if armed is None or output is None:
                return tensor
            if len(runner.input_batch.req_ids) != 1 or tensor.shape[0] != 1:
                raise RuntimeError('Trajectory diagnostic requires a single request/logit row')
            rid = runner.input_batch.req_ids[0]
            req = runner.requests[rid]
            if req.prompt_token_ids != armed['prompt']:
                raise RuntimeError('Trajectory prompt ownership mismatch')
            computed = int(runner.input_batch.num_computed_tokens_cpu[0])
            query = output.num_scheduled_tokens[rid]
            end = computed + query
            index = end - len(armed['prompt']) if armed['name'] == SEQUENTIAL else armed['target']
            if index > armed['target']:
                return tensor  # Async work after the requested boundary is not evidence.
            if computed in s['queries']:
                raise RuntimeError('Recomputed/preempted trajectory; measurement review required')
            s['queries'][computed] = runner.input_ids.gpu[:query].detach().clone()
            if armed['name'] == FORCED and end != len(armed['prompt']):
                return tensor
            if index < 0:
                return tensor  # Incomplete prompt chunk: no generated decision yet.
            if s['shape'] is None:
                raise RuntimeError('Missing actual execution-mode observation')
            s['candidate'] = dict(index=index, raw=tensor[0].detach().clone(), output=output)
            s['metadata'][index] = dict(request_id=rid, computed_tokens_before=computed,
                scheduled_query_tokens=query, execution=dict(s['shape']),
                block_ids=[list(ids) for ids in req.block_ids])
            if s['localization_stage'] == HEAD_STAGE:
                if head_input is None:
                    raise RuntimeError('Missing paired LM-head input measurement')
                s['candidate']['head_input'] = head_input
                s['metadata'][index]['head_input_layout'] = layout
            return tensor

        def sampling(grammar_output):
            s = self._m4_probe
            state = runner.execute_model_state
            s['sampling_output'] = state[0] if state is not None else None
            candidate = s['candidate']
            if candidate is not None and s['sampling_output'] is candidate['output']:
                rid = s['metadata'][candidate['index']]['request_id']
                if grammar_output is None or grammar_output.structured_output_request_ids != [rid]:
                    raise RuntimeError('Missing request-owned JSON grammar mask')
                # This is already a CPU ndarray, not a synchronization of logits.
                s['metadata'][candidate['index']]['grammar_bitmask'] = grammar_output.grammar_bitmask.copy()
                grammar_meta = scheduler._m4_grammar_meta.pop(id(s['sampling_output']), {})
                if rid not in grammar_meta:
                    raise RuntimeError('Missing scheduler-owned grammar provenance')
                s['metadata'][candidate['index']]['grammar_state'] = grammar_meta[rid]
            try:
                return sample_tokens(grammar_output)
            finally:
                s['sampling_output'] = None

        def before_sampling(logits, *args, **kwargs):
            s = self._m4_probe
            candidate = s['candidate']
            paired = candidate is not None and s['sampling_output'] is candidate['output']
            masked = logits[0].detach().clone() if paired else None
            result = sample(logits, *args, **kwargs)
            if paired:
                armed = s['armed']
                if armed['name'] == SEQUENTIAL:
                    armed['capture'].record(candidate['index'], candidate['raw'], masked,
                                            result.sampled_token_ids,head_input=candidate.get('head_input'))
                else:
                    if armed.get('selected') is not None:
                        raise RuntimeError('Forced boundary sampled twice')
                    armed['selected'] = dict(raw=candidate['raw'], masked=masked,
                        index=candidate['index'], token=result.sampled_token_ids.detach().clone())
                    if 'head_input' in candidate:
                        armed['selected']['head_input'] = candidate['head_input']
                s['candidate'] = None
            return result

        try:
            for owner, name, fn in ((runner,'execute_model',execution),
                    (runner,'_determine_batch_execution_and_padding',shape),
                    (runner.model,'compute_logits',raw_logits),
                    (runner,'sample_tokens',sampling), (runner,'_sample',before_sampling)):
                self._m4_hooks.replace(owner, name, fn)
        except BaseException:
            self.m4_probe_uninstall()
            raise
        return dict(installed=True, measurement_point=MEASUREMENT,
            synchronization='GPU first-mismatch latch; full-vector host transfer only after abort')

    def m4_trajectory_arm(self, name, prompt, reference, target):
        s = self._m4_probe
        if s['armed'] is not None or name not in (SEQUENTIAL, FORCED):
            raise RuntimeError('Invalid/uncollected trajectory arm')
        config = configuration()
        original = config['original_prompt_token_ids']
        expected = original if name == SEQUENTIAL else original + reference[:target]
        if (prompt != expected or reference != config['generated_prefix_token_ids'] +
                [config['reference_next_token_id']] or not 0 <= target <= config['generation_index']):
            raise RuntimeError('Trajectory reference/prompt differs from frozen input')
        s.update(metadata={}, queries={}, candidate=None)
        s['armed'] = dict(name=name, prompt=list(prompt), reference=list(reference), target=target)
        if name == SEQUENTIAL:
            s['armed']['capture'] = FirstMismatchCapture(reference, target)
        # Only forced prefill seeds grammar. Sequential requests never match this
        # scheduler's forced prompt and use vLLM's natural grammar advancement.
        scheduler = _schedulers[id(self.vllm_config)]
        scheduler._m4_config = dict(config,
            forced_input_token_ids=prompt if name == FORCED else [],
            generated_prefix_token_ids=reference[:target])
        return True

    def m4_trajectory_collect(self):
        try:
            return self._m4_trajectory_collect()
        except Exception as exc:
            # Collection is after abort. Preserve a retained decision even when
            # prefix/finite/sample validation fails before a normal RPC return.
            state = self._m4_probe
            armed = state['armed']
            selected = ((armed['capture'].selected if armed['name'] == SEQUENTIAL
                         else armed.get('selected')) if armed else None)
            if selected is not None and int(selected['index']) >= 0:
                import torch
                config = configuration()
                def vector_record(vector):
                    cpu = vector.detach().to(device='cpu',dtype=torch.float32).contiguous()
                    return dict(captured_dtype=str(vector.dtype),storage_dtype='float32',
                        shape=list(cpu.shape),byte_order=sys.byteorder,
                        values_base64=base64.b64encode(cpu.view(torch.uint8).numpy().tobytes()).decode('ascii'))
                write_new(Path(config['cache_trace']).parent/f"failed-{armed['name'].lower()}-vectors.json",
                    dict(schema_version=SCHEMA+'.failure_vectors',condition=armed['name'],
                        source_commit=config['source_commit'],generation_index=int(selected['index']),
                        request_id=state['metadata'].get(int(selected['index']),{}).get('request_id'),
                        observed_token=int(selected['token']),collection_error=f'{type(exc).__name__}: {exc}',
                        raw_measurement_point=MEASUREMENT,
                        masked_measurement_point='GPUModelRunner._sample entry; after grammar, before sampler',
                        classification='INCONCLUSIVE',raw=vector_record(selected['raw']),
                        masked=vector_record(selected['masked']),
                        head_input=(vector_record(selected['head_input']) if 'head_input' in selected else None),
                        query_chunks=[dict(offset=offset,token_ids=query.to(device='cpu').tolist())
                            for offset,query in sorted(state['queries'].items())]))
            raise

    def _m4_trajectory_collect(self):
        import torch
        s = self._m4_probe
        armed = s['armed']
        if armed is None:
            raise RuntimeError('Trajectory not armed')
        selected = (armed['capture'].selected if armed['name'] == SEQUENTIAL
                    else armed.get('selected'))
        if selected is None:
            raise RuntimeError('Missing trajectory boundary capture')
        index = int(selected['index'])
        if index < 0 or index not in s['metadata']:
            raise RuntimeError('No first mismatch/target boundary reached')
        observed = int(selected['token'])
        logical = []
        end = len(configuration()['original_prompt_token_ids']) + index
        for offset, gpu_query in sorted(s['queries'].items()):
            if offset >= end:
                continue
            if offset != len(logical):
                raise RuntimeError('Non-contiguous actual GPU input trajectory')
            logical.extend(gpu_query.to(device='cpu').tolist())
        if len(logical) != end:
            raise RuntimeError('Actual GPU logical prefix length mismatch')
        original = configuration()['original_prompt_token_ids']
        if logical != original + armed['reference'][:index]:
            raise RuntimeError('Captured actual GPU prefix differs before first mismatch')
        summary, raw = logits_summary(selected['raw'], selected['masked'],
            tuple(dict.fromkeys((4164,3752,armed['reference'][index],observed))))
        if summary['grammar_masked_argmax_token_id'] != observed:
            raise RuntimeError('Actual greedy sample differs from grammar-masked argmax')
        meta = dict(s['metadata'][index])
        bitmask = meta.pop('grammar_bitmask')
        meta['grammar_bitmask'] = dict(shape=list(bitmask.shape), dtype=str(bitmask.dtype),
            sha256=hashlib.sha256(bitmask.tobytes()).hexdigest(), values=bitmask.tolist())
        result = dict(schema_version=SCHEMA, condition=armed['name'],
            identity=prefix_identity(original, logical[len(original):]),
            original_prompt_token_ids=original, generated_prefix_token_ids=logical[len(original):],
            full_logical_input_token_ids=logical, historical_token=armed['reference'][index],
            observed_token=observed, logits=summary, state=meta,
            raw_full_logits=raw.tolist(),
            grammar_masked_full_logits=selected['masked'].to(device='cpu',dtype=torch.float32).tolist())
        # JSON represents masked-out values as null, with explicit meaning.
        result['grammar_masked_full_logits'] = [None if v == -float('inf') else v
            for v in result['grammar_masked_full_logits']]
        if s['localization_stage'] == HEAD_STAGE:
            if 'head_input' not in selected:
                raise RuntimeError('LM-head input hook did not retain selected decision')
            result['head_input'] = serialize_head_input(selected['head_input'],
                meta.pop('head_input_layout'),s['head_contract'],result)
        s.update(armed=None, candidate=None, metadata={}, queries={})
        return result


def sequential_request(llm, prompt, params, reference, target, label):
    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError('Unrelated request traffic')
    rid = engine.add_request(label, {'prompt_token_ids':prompt}, params)
    deadline = time.monotonic() + 300
    try:
        while engine.has_unfinished_requests():
            if time.monotonic() > deadline:
                raise TimeoutError('Sequential trajectory timed out')
            for output in engine.step():
                # add_request returns the internal ID; RequestOutput carries
                # the caller's external label (pinned output_processor.py:384).
                if output.request_id != label:
                    raise RuntimeError('Unexpected sequential output owner')
                if len(output.outputs) != 1:
                    raise RuntimeError('Expected one Vanilla output')
                tokens = list(output.outputs[0].token_ids)
                index = first_boundary(tokens, reference, target)
                if index is not None:
                    return dict(request_id=rid, generation_index=index,
                        observed_tokens=tokens[:index+1],
                        delivered_tokens_after_boundary=tokens[index+1:])
        raise RuntimeError('Historical trajectory stopped before target or mismatch')
    finally:
        engine.abort_request([rid], internal=True)


def validate_capture(result, case):
    import math
    if result['schema_version'] != SCHEMA or result['condition'] not in (SEQUENTIAL, FORCED):
        raise ValueError('Unknown trajectory schema/condition')
    index = result['identity']['generation_index']
    reference = case['generated_prefix_token_ids'] + [case['reference_next_token_id']]
    if type(index) is not int or not 0 <= index <= case['generation_index']:
        raise ValueError('Invalid captured boundary')
    prompt, generated = result['original_prompt_token_ids'], result['generated_prefix_token_ids']
    if (prompt != case['original_prompt_token_ids'] or generated != reference[:index]
            or result['full_logical_input_token_ids'] != prompt + generated
            or result['identity'] != prefix_identity(prompt, generated)
            or result['historical_token'] != reference[index]):
        raise ValueError('Logical prefix/reference identity mismatch')
    if (result['condition'] == SEQUENTIAL and index < case['generation_index']
            and result['observed_token'] == reference[index]):
        raise ValueError('Early boundary is not a first mismatch')
    raw, masked = result['raw_full_logits'], result['grammar_masked_full_logits']
    if (len(raw) != VOCABULARY_SIZE or len(raw) != len(masked)
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in raw)
            or any(v is not None and (type(v) not in (int,float) or not math.isfinite(v)) for v in masked)):
        raise ValueError('Incomplete/nonfinite full vectors')
    argmax = max(range(len(raw)), key=raw.__getitem__)
    allowed = [i for i, v in enumerate(masked) if v is not None]
    if not allowed:
        raise ValueError('Grammar masks entire vocabulary')
    masked_argmax = max(allowed, key=masked.__getitem__)
    logits = result['logits']
    if (logits['measurement_point'] != MEASUREMENT or logits['argmax_token_id'] != argmax
            or logits['dtype'] not in ('torch.bfloat16','torch.float32','torch.float16')
            or logits['grammar_masked_argmax_token_id'] != masked_argmax
            or result['observed_token'] != masked_argmax
            or logits['logit_token_4164'] != raw[4164] or logits['logit_token_3752'] != raw[3752]):
        raise ValueError('Raw/masked/sample measurement mismatch')
    top = logits['top_tokens']
    ranked = sorted(raw, reverse=True)
    if (len(top) != 20 or len({t['token_id'] for t in top}) != 20
            or any(t['logit'] != raw[t['token_id']] for t in top)
            or [t['logit'] for t in top] != ranked[:20]
            or top[0]['token_id'] != argmax
            or logits['top1_minus_top2'] != ranked[0]-ranked[1]
            or logits['difference_4164_minus_3752'] != raw[4164]-raw[3752]
            or logits['nan_count'] != 0 or logits['inf_count'] != 0
            or any(v is not None and v != raw[i] for i,v in enumerate(masked))):
        raise ValueError('Full vectors and logit summaries disagree')
    state = result['state']
    if (type(state['computed_tokens_before']) is not int or state['computed_tokens_before'] < 0
            or type(state['scheduled_query_tokens']) is not int
            or state['computed_tokens_before'] + state['scheduled_query_tokens'] != len(prompt)+index
            or state['scheduled_query_tokens'] <= 0 or not state['execution']
            or not state['block_ids'] or not state['grammar_bitmask']):
        raise ValueError('Missing execution/query/cache/grammar state')
    execution = state['execution']
    if (execution['cudagraph_mode'] not in ('NONE','PIECEWISE','FULL')
            or type(execution['padded_tokens']) is not int
            or execution['padded_tokens'] < state['scheduled_query_tokens']
            or execution['padded_requests'] not in (None,1)
            or type(execution['uniform_decode']) is not bool
            or execution['microbatching'] is not False):
        raise ValueError('Missing/unsupported single-request execution geometry')
    grammar = state['grammar_state']
    if (not grammar['backend_class'] or not grammar['grammar_class']
            or grammar['num_processed_tokens'] != index or grammar['reasoner_class'] is not None
            or (index > 0 and grammar['reasoning_ended'] is not True)):
        raise ValueError('Unverified natural/seeded grammar prefix state')
    mask = state['grammar_bitmask']
    words = [v for row in mask['values'] for v in row]
    if (mask['shape'] != [1,(len(raw)+31)//32] or mask['dtype'] != 'int32'
            or len(words) != mask['shape'][1]
            or hashlib.sha256(struct.pack('<'+'i'*len(words), *words)).hexdigest() != mask['sha256']
            or any(((words[i//32] >> (i%32)) & 1) != (v is not None)
                   for i,v in enumerate(masked))):
        raise ValueError('Grammar mask bytes disagree with masked vector')
    if result['cache']['cached_tokens'] != 0 or result['reset_verified'] is not True:
        raise ValueError('Cold trajectory reused prefix cache')
    lookup_prompt = prompt if result['condition'] == SEQUENTIAL else prompt + generated
    checked = cache_evidence(result['cache']['lookup_records'],state['request_id'],'COLD',
        len(prompt),len(lookup_prompt),token_hash(lookup_prompt))
    if checked != result['cache']:
        raise ValueError('Cold request-local lookup summary differs')
    env = result['environment']
    if (env['vllm'] != '0.30.0' or env['pytorch'] != '2.13.0+cu130'
            or env['cuda'] != '13.0' or env['capability'] != [8,0]
            or 'A100' not in env['gpu_name'] or not env['driver'] or not env['python']
            or set(env['runtime_overrides']) != {'VLLM_USE_V2_MODEL_RUNNER',
                'VLLM_WORKER_MULTIPROC_METHOD','VLLM_ATTENTION_BACKEND','VLLM_BATCH_INVARIANT',
                'VLLM_USE_FLASHINFER_SAMPLER','NVIDIA_TF32_OVERRIDE','CUDA_VISIBLE_DEVICES',
                'CUBLAS_WORKSPACE_CONFIG'}):
        raise ValueError('Pinned environment mismatch')
    if (not re.fullmatch('[0-9a-f]{40}',result['source_commit'])
            or result['instrumentation_commit'] != result['source_commit']
            or result['source_commit'] != case['source_commit']):
        raise ValueError('Instrumentation/source SHA mismatch')
    from experiments.run_m4_cache_probe import engine_config, sampling_config, expected_effective_config
    if (result['engine_config'] != engine_config('decode_trajectory')
            or result['effective_config'] != expected_effective_config()
            or result['sampling_config'] != sampling_config(case['historical_request'])):
        raise ValueError('Historical engine/sampling contract differs')
    if result['model'] != dict(target=TARGET,revision=TARGET_REVISION,
            tokenizer_revision=TARGET_REVISION,dtype='bfloat16',quantization='awq_marlin'):
        raise ValueError('Model/revision identity differs')
    if (result['run_id'] != case['run_id'] or not isinstance(result['timestamp_utc'],str)
            or not result['timestamp_utc']):
        raise ValueError('Run identity/timestamp missing or differs')
    if logits['grammar_masked_measurement_point'] != 'GPUModelRunner._sample entry; after grammar, before sampler':
        raise ValueError('Grammar-masked measurement stage differs')
    observed = result['observed']
    if observed['request_id'] != state['request_id']:
        raise ValueError('Delivered output/capture request ownership differs')
    if result['condition'] == SEQUENTIAL:
        tokens = observed['observed_tokens']
        if (observed['generation_index'] != index
                or tokens != reference[:index] + [result['observed_token']]
                or first_boundary(tokens,reference,case['generation_index']) != index):
            raise ValueError('Delivered trajectory and captured first decision differ')
    elif (observed['first_output_token_id'] != result['observed_token']
            or not observed['observed_output_tokens']
            or observed['observed_output_tokens'][0] != result['observed_token']):
        raise ValueError('Delivered forced decision differs from captured sample')


def analyze_trajectory(sequential, forced, case, expected_source_commit=None):
    """Cases are ordered: invalid, earlier divergence, equal vectors, A, B."""
    base = dict(root_cause='ROOT_CAUSE_NOT_PROVEN', full_sweep='FULL_SWEEP_BLOCKED')
    try:
        validate_prepared_case(case,expected_source_commit)
        validate_capture(sequential, case)
        validate_capture(forced, case)
        if sequential['condition'] != SEQUENTIAL or forced['condition'] != FORCED:
            raise ValueError('Wrong trajectory condition order')
        if (sequential['identity'] != forced['identity'] or
                sequential['full_logical_input_token_ids'] != forced['full_logical_input_token_ids']):
            raise ValueError('A/B logical prefixes differ')
        for key in ('environment','engine_config','effective_config','sampling_config','source_commit','model','run_id'):
            if not sequential.get(key) or sequential[key] != forced.get(key):
                raise ValueError(f'Provenance {key} missing/differs')
        if sequential['logits']['dtype'] != forced['logits']['dtype']:
            raise ValueError('Raw-logit dtype differs')
        if (sequential['state']['grammar_bitmask'] != forced['state']['grammar_bitmask']
                or any(sequential['state']['grammar_state'][k] != forced['state']['grammar_state'][k]
                    for k in ('backend_class','grammar_class','num_processed_tokens','reasoner_class'))):
            raise ValueError('A/B effective grammar state masks differ')
        a, b = sequential['raw_full_logits'], forced['raw_full_logits']
        if len(a) != len(b):
            raise ValueError('A/B vocabularies differ')
        delta = [abs(x-y) for x,y in zip(a,b)]
        differences = dict(max_absolute_difference=max(delta),
            mean_absolute_difference=sum(delta)/len(delta), vocabulary_size=len(delta))
    except (KeyError,ValueError,TypeError,IndexError,struct.error) as exc:
        return dict(base, classification='INCONCLUSIVE', reasons=[str(exc)])
    index = sequential['identity']['generation_index']
    s, f = sequential['logits']['argmax_token_id'], forced['logits']['argmax_token_id']
    if index < case['generation_index']:
        classification = 'EARLIER_DIVERGENCE_OBSERVED'
    elif not max(delta) and s == f:
        classification = 'H_TRAJECTORY_NOT_SUPPORTED'
    elif ((s,f) == (4164,3752) and sequential['observed_token'] == 4164
            and forced['observed_token'] == 3752):
        classification = 'H_TRAJECTORY_SUPPORTED'
    elif s == 3752 and sequential['observed_token'] == 3752:
        classification = 'HISTORICAL_TRAJECTORY_NOT_REPRODUCED'
    else:
        classification = 'INCONCLUSIVE'
    return dict(base, classification=classification, vector_difference=differences,
        trajectory_status=('HISTORICAL_TRAJECTORY_NOT_REPRODUCED'
            if index < case['generation_index'] or sequential['observed_token'] != 4164
            else 'HISTORICAL_PREFIX_REPRODUCED'), generation_index=index)


def run_conditions(llm, directory, case, params, common):
    try:
        return _run_conditions(llm,directory,case,params,common)
    except Exception as exc:
        if not (directory/'analysis.json').exists():
            write_new(directory/'analysis.json',dict(classification='INCONCLUSIVE',
                reasons=[f'{type(exc).__name__}: {exc}'],root_cause='ROOT_CAUSE_NOT_PROVEN',
                full_sweep='FULL_SWEEP_BLOCKED'))
        raise


def _run_conditions(llm, directory, case, params, common):
    """Called only inside the existing pinned GPU/environment/lifecycle gates."""
    from experiments.run_m4_cache_probe import one_request
    reference = case['generated_prefix_token_ids'] + [case['reference_next_token_id']]
    installed = llm.collective_rpc('m4_trajectory_install')
    if len(installed) != 1 or installed[0].get('installed') is not True:
        raise RuntimeError('Expected one installed trajectory worker')
    write_new(directory/'hook-installation.json', installed)
    results = []
    target = case['generation_index']
    for name in (SEQUENTIAL, FORCED):
        if llm.reset_prefix_cache() is not True:
            raise RuntimeError('Cold prefix-cache reset failed')
        write_new(directory/f'reset-{name.lower()}.json', dict(reset_succeeded=True, condition=name))
        prompt = case['original_prompt_token_ids'] + (reference[:target] if name == FORCED else [])
        llm.collective_rpc('m4_trajectory_arm', args=(name,prompt,reference,target))
        if name == SEQUENTIAL:
            observed = sequential_request(llm,prompt,params(),reference,target,'trajectory-sequential')
            target = observed['generation_index']
        else:
            observed = one_request(llm,prompt,params(),'trajectory-forced')
        captured = llm.collective_rpc('m4_trajectory_collect')
        if len(captured) != 1:
            raise RuntimeError('Expected one trajectory capture')
        result = captured[0]
        # Preserve the unvalidated worker vectors before ownership/cache/schema
        # guards. Failure cannot erase a returned boundary tensor snapshot.
        write_new(directory/f'worker-{name.lower()}.json',result)
        if result['state']['request_id'] != observed['request_id']:
            raise RuntimeError('Trajectory output/capture request ownership differs')
        if name == SEQUENTIAL:
            if (result['identity']['generation_index'] != target or
                    observed['observed_tokens'] != reference[:target] + [result['observed_token']]):
                raise RuntimeError('Device first-mismatch and delivered output disagree')
        elif result['observed_token'] != observed['first_output_token_id']:
            raise RuntimeError('Forced sampled token differs from capture')
        events = [json.loads(line) for line in Path(case['cache_trace']).read_text().splitlines()]
        result['cache'] = cache_evidence(events, observed['request_id'], 'COLD',
            len(case['original_prompt_token_ids']),len(prompt),token_hash(prompt))
        result.update(common, observed=observed, reset_verified=True,
            grammar_masked_null_means='negative infinity')
        validate_capture(result, case)
        # Full raw float32 JSON is a lossless expansion of captured BF16. Never
        # substitute top-N/reductions for the independently auditable vector.
        write_new(directory/f'{name.lower()}.json', result)
        results.append(result)
    analyze = analyze_head_input if case.get('localization_stage')==HEAD_STAGE else analyze_trajectory
    analysis = analyze(*results, case,expected_source_commit=common['source_commit'])
    write_new(directory/'analysis.json', analysis)
    if analysis['classification'] == 'INCONCLUSIVE':
        raise RuntimeError('Trajectory result INCONCLUSIVE; preserve artifacts')
    return analysis
