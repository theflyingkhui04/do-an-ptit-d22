"""Explicit user-run A100 Vanilla-only controlled replay; imports are CPU-safe."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

from experiments.m4_cache_probe import (SCHEMA,FROZEN_SOURCE,EVIDENCE_COMMIT,
    load_input,token_hash,cache_evidence,validate_result,analyze_replays,verify_run_manifest)
from experiments.m4_protocol import ROOT,write_new
from experiments.m4_security import artifact_path,validated_source_commit
from experiments.m4_treatments import TARGET,TARGET_REVISION,engine_kwargs


def now():
    return datetime.now(timezone.utc).isoformat()


def engine_config(diagnostic_mode='cache_probe',localization_stage=None):
    config = engine_kwargs('vanilla')
    # Pinned EngineArgs uses 8192 for LLM_CLASS but 2048 for the historical
    # A100 API server. Preserve the server's resolved budget explicitly.
    config.update(tokenizer_revision=TARGET_REVISION,enable_prefix_caching=True,disable_log_stats=False,
        enable_chunked_prefill=True,max_num_batched_tokens=2048,served_model_name='m4-vanilla',
        scheduler_cls='experiments.m4_cache_probe_hooks.CacheProbeScheduler',
        worker_extension_cls='experiments.m4_cache_probe_hooks.CacheProbeWorkerExtension',
        distributed_executor_backend='uni')
    if diagnostic_mode == 'decode_trajectory':
        config.update(scheduler_cls='experiments.m4_decode_trajectory.TrajectoryScheduler',
            worker_extension_cls='experiments.m4_decode_trajectory.TrajectoryWorkerExtension')
    elif diagnostic_mode != 'cache_probe':
        raise ValueError('Unknown diagnostic mode')
    if localization_stage in ('layer_boundaries','layer0_internal'):
        if diagnostic_mode != 'decode_trajectory':
            raise ValueError('Layer localization requires explicit trajectory diagnostic mode')
        config['worker_cls'] = 'experiments.m4_layer_probe_runtime.LayerBoundaryWorker'
    return config


def sampling_config(request):
    if request['temperature'] != 0 or request['seed'] != 42:
        raise ValueError('Greedy historical request required')
    result = dict(temperature=request['temperature'],seed=request['seed'],n=1,
                  max_tokens=request['max_tokens'])
    if request.get('response_format'):
        if request['response_format'] != {'type':'json_object'}:
            raise ValueError('Only historical JSON-object grammar is supported')
        result['structured_outputs'] = {'json_object':True}
    return result


def expected_effective_config():
    """Exact reviewed historical flags, shared by live and offline validation."""
    return dict(max_num_batched_tokens=2048,max_num_seqs=32,
        enable_chunked_prefill=True,async_scheduling=True,enable_prefix_caching=True,
        dtype='torch.bfloat16',enforce_eager=False,max_model_len=4096,
        compilation_mode='VLLM_COMPILE',cudagraph_mode='FULL_AND_PIECEWISE',
        cudagraph_capture_sizes=[1,2,4,8,16,24,32,40,48,56,64])


def prepare(directory,source_commit,workload,prompt_id,index,diagnostic_mode='cache_probe',localization_stage=None):
    engine_config(diagnostic_mode,localization_stage)  # Reject unknown modes before writing artifacts.
    if localization_stage is not None:
        from experiments.m4_head_input_probe import HEAD_STAGE
        from experiments.m4_layer0_probe import INTERNAL_STAGE
        from experiments.m4_layer_probe import LAYER_STAGE
        if diagnostic_mode != 'decode_trajectory' or localization_stage not in (HEAD_STAGE,LAYER_STAGE,INTERNAL_STAGE):
            raise ValueError('Localization requires explicit trajectory diagnostic mode')
    if diagnostic_mode == 'decode_trajectory' and (workload,prompt_id,index) != ('W1','m0_04_001',549):
        raise ValueError('Trajectory diagnostic is scoped to W1/m0_04_001/index 549')
    case = load_input(workload,prompt_id,index)
    case.update(schema_version=SCHEMA,source_commit=source_commit,historical_source_commit=FROZEN_SOURCE,
        evidence_commit=EVIDENCE_COMMIT,instrumentation_commit=source_commit,
        cache_trace=str((directory/'cache.jsonl').resolve()),
        construction='preserved original prompt IDs + vanilla repeat-0 generated IDs[:generation_index]',
        cache_conditioning='original prompt only; never full generated prefix',
        replay_limitation='Forced continuation is prefilled, not the historical sequence of decode forwards.')
    if diagnostic_mode == 'decode_trajectory':
        case.update(diagnostic_mode=diagnostic_mode,
            run_id=directory.name,
            construction='sequential: original prompt only; forced: original prompt + reference prefix at selected boundary',
            cache_conditioning='none; verified cold reset before each condition',
            replay_limitation='Sequential execution is natural; forced prefill compares only the identical selected logical boundary.')
        if localization_stage is not None:
            case['localization_stage'] = localization_stage
    write_new(directory/'input.json',case)
    return case


def one_request(llm,tokens,params,label):
    """Preserve max_tokens/EOS settings; abort after the first observed output.

    The engine may have async work in flight. That work is retained in worker
    logs; cache-hit extent is checked rather than assumed from this abort.
    """
    engine = llm.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError('Unrelated request traffic in diagnostic engine')
    rid = engine.add_request(label,{'prompt_token_ids':tokens},params)
    deadline = time.monotonic()+180
    seen = None
    try:
        while engine.has_unfinished_requests():
            if time.monotonic()>deadline:
                raise TimeoutError('Controlled request timed out')
            for output in engine.step():
                if output.outputs and output.outputs[0].token_ids:
                    seen = dict(request_id=rid,first_output_token_id=output.outputs[0].token_ids[0],
                                observed_output_tokens=list(output.outputs[0].token_ids))
                    return seen
        raise RuntimeError('No target output at boundary')
    finally:
        engine.abort_request([rid],internal=True)


@contextmanager
def diagnostic_engine(llm,directory):
    """Own all post-construction paths, including validation/install failures."""
    cleanup = dict(hooks_restored=False,engine_shutdown=False)
    try:
        yield llm
    finally:
        try:
            restored = llm.collective_rpc('m4_probe_uninstall')
            cleanup['worker_results'] = restored
            if len(restored)!=1 or restored[0] != dict(
                    restored=True,captures_released=True,cache_observer_restored=True):
                raise RuntimeError('Diagnostic teardown not confirmed')
            cleanup['hooks_restored'] = True
        finally:
            try:
                # Exact pinned CoreClient API; invoked even when cleanup RPC fails.
                llm.llm_engine.engine_core.shutdown()
                cleanup['engine_shutdown'] = True
            finally:
                write_new(directory/'hook-cleanup.json',cleanup)


def worker(directory,source_commit):
    """Runs only when explicitly invoked by `run`; never called by CPU tests."""
    from experiments.m4_cache_probe_hooks import verify_runtime_sources
    from experiments.m4_cost_runner import observe_hardware
    from huggingface_hub import HfApi,snapshot_download
    manifest = verify_runtime_sources()
    case = json.loads((directory/'input.json').read_text(encoding='utf8'))
    mode = case.get('diagnostic_mode','cache_probe')
    config = engine_config(mode,case.get('localization_stage'))
    if case.get('localization_stage') in ('layer_boundaries','layer0_internal'):
        from experiments.m4_layer_probe_runtime import verify_layer_sources
        manifest = verify_layer_sources()
        if case.get('localization_stage') == 'layer0_internal':
            from experiments.m4_layer0_sources import verify_internal_sources
            manifest = verify_internal_sources()
    if mode == 'decode_trajectory' and (case['workload'],case['prompt_id'],case['generation_index']) != ('W1','m0_04_001',549):
        raise ValueError('Trajectory diagnostic is scoped to W1/m0_04_001/index 549')
    if (case['source_commit'] != source_commit or case['instrumentation_commit'] != source_commit
            or token_hash(case['forced_input_token_ids']) != case['input_token_ids_sha256']):
        raise ValueError('Prepared input provenance differs')
    expected = load_input(case['workload'],case['prompt_id'],case['generation_index'])
    if any(case[k] != expected[k] for k in expected):
        raise ValueError('Prepared input differs from authoritative frozen evidence')
    actual = observe_hardware()
    if (not actual.get('cuda_available') or 'A100' not in (actual.get('gpu') or '')
            or actual.get('gpu_compute_capability') != [8,0]
            or actual.get('vllm_version') != '0.30.0'
            or actual.get('torch_version') != '2.13.0+cu130'
            or actual.get('cuda_version') != '13.0' or not actual.get('driver_version')):
        write_new(directory/'environment-rejected.json',actual)
        raise RuntimeError('Pinned A100/sm80/runtime dependency gate failed')
    env = dict(gpu_name=actual['gpu'],capability=actual['gpu_compute_capability'],
        driver=actual['driver_version'],cuda=actual['cuda_version'],python=platform.python_version(),
        pytorch=actual['torch_version'],vllm=actual['vllm_version'])
    env['runtime_overrides'] = {k:os.environ.get(k) for k in (
        'VLLM_USE_V2_MODEL_RUNNER','VLLM_WORKER_MULTIPROC_METHOD','VLLM_ATTENTION_BACKEND',
        'VLLM_BATCH_INVARIANT','VLLM_USE_FLASHINFER_SAMPLER','NVIDIA_TF32_OVERRIDE',
        'CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')}
    historical = dict(gpu_name='NVIDIA A100-SXM4-40GB',capability=[8,0],driver='580.82.07',
        cuda='13.0',python='3.13.16',pytorch='2.13.0+cu130',vllm='0.30.0')
    drift = {k:dict(historical=v,actual=env[k]) for k,v in historical.items() if env[k]!=v}
    write_new(directory/'environment.json',dict(actual=env,historical=historical,drift=drift,
        managed_drift_interpretation='Recorded, never silently treated as historical equivalence',hardware=actual))
    resolved = HfApi().model_info(TARGET,revision=TARGET_REVISION).sha
    if resolved != TARGET_REVISION:
        raise RuntimeError('Target revision resolution differs')
    snapshot_download(TARGET,revision=TARGET_REVISION)
    write_new(directory/'runtime-sources.json',manifest)
    write_new(directory/'pip-freeze.txt.json',dict(packages=subprocess.check_output(
        [sys.executable,'-m','pip','freeze'],text=True).splitlines()))
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    with diagnostic_engine(LLM(**config),directory) as llm:
        effective = llm.llm_engine.vllm_config
        flags = dict(max_num_batched_tokens=effective.scheduler_config.max_num_batched_tokens,
            max_num_seqs=effective.scheduler_config.max_num_seqs,
            enable_chunked_prefill=effective.scheduler_config.enable_chunked_prefill,
            async_scheduling=effective.scheduler_config.async_scheduling,
            enable_prefix_caching=effective.cache_config.enable_prefix_caching,
            dtype=str(effective.model_config.dtype),enforce_eager=effective.model_config.enforce_eager,
            max_model_len=effective.model_config.max_model_len,
            compilation_mode=effective.compilation_config.mode.name,
            cudagraph_mode=effective.compilation_config.cudagraph_mode.name,
            cudagraph_capture_sizes=effective.compilation_config.cudagraph_capture_sizes)
        write_new(directory/'effective-config.json',flags)
        expected_flags = expected_effective_config()
        if flags != expected_flags:
            raise RuntimeError('Resolved engine/compilation flags differ from historical Vanilla; review required')
        settings = sampling_config(case['historical_request'])
        def params():
            kwargs = dict(settings)
            if 'structured_outputs' in kwargs:
                kwargs['structured_outputs'] = StructuredOutputsParams(**kwargs['structured_outputs'])
            return SamplingParams(**kwargs)
        if mode == 'decode_trajectory':
            from experiments.m4_decode_trajectory import run_conditions
            run_conditions(llm,directory,case,params,dict(environment=env,
                engine_config=config,effective_config=flags,sampling_config=settings,
                source_commit=source_commit,instrumentation_commit=source_commit,
                run_id=directory.name,timestamp_utc=now(),
                model=dict(target=TARGET,revision=TARGET_REVISION,tokenizer_revision=TARGET_REVISION,
                    dtype='bfloat16',quantization='awq_marlin'),historical_environment_drift=drift))
            return
        install = llm.collective_rpc('m4_probe_install')
        if len(install)!=1 or not install[0]['installed']:
            raise RuntimeError('Single diagnostic worker required')
        write_new(directory/'hook-installation.json',install)
        results,differences = [],[]
        for sequence,condition in enumerate(('COLD','CACHE_HIT')*2):
            repetition = sequence//2+1
            suffix = '' if repetition==1 else '_replay_2'
            if llm.reset_prefix_cache() is not True:
                raise RuntimeError('Fresh logical prefix-cache state not established')
            write_new(directory/f'reset-{condition.lower()}{suffix}.json',
                      dict(condition=condition,repetition=repetition,reset_succeeded=True,timestamp_utc=now()))
            conditioning = None
            if condition=='CACHE_HIT':
                conditioning = one_request(llm,case['original_prompt_token_ids'],params(),f'conditioning-{repetition}')
                # FIFO worker RPC is a barrier following abort; it does not reset or
                # warm any cache. The subsequent request-local lookup proves reuse.
            llm.collective_rpc('m4_probe_arm',args=(condition,case['forced_input_token_ids']))
            observed = one_request(llm,case['forced_input_token_ids'],params(),f'{condition}-{repetition}')
            captures = llm.collective_rpc('m4_probe_collect')
            if len(captures)!=1:
                raise RuntimeError('Expected exactly one worker capture')
            captured = captures[0]
            if captured['input_token_ids_sha256'] != case['input_token_ids_sha256']:
                raise RuntimeError('Worker input hash differs')
            if captured['request_id'] != observed['request_id']:
                raise RuntimeError('Observed output and logit request ownership differs')
            if observed['first_output_token_id'] != captured['logits']['grammar_masked_argmax_token_id']:
                raise RuntimeError('Observed greedy token differs from measured grammar-masked argmax')
            events = [json.loads(s) for s in Path(case['cache_trace']).read_text(encoding='utf8').splitlines()]
            if conditioning:
                conditioning['cache_evidence'] = cache_evidence(events,conditioning['request_id'],'COLD',
                    len(case['original_prompt_token_ids']),len(case['original_prompt_token_ids']),
                    case['original_prompt_token_ids_sha256'])
            cache = cache_evidence(events,captured['request_id'],condition,
                                   len(case['original_prompt_token_ids']),case['input_token_count'],
                                   case['input_token_ids_sha256'])
            seeded = [e for e in events if e['event']=='forced_boundary_state'
                      and e['request_id']==captured['request_id']]
            if len(seeded)!=1 or seeded[0]['generated_prefix_count']!=case['generation_index']:
                raise RuntimeError('Forced grammar/boundary state not established')
            result = dict(schema_version=SCHEMA,diagnostic_name='vanilla_cold_cache_hit',
                source_commit=source_commit,historical_source_commit=FROZEN_SOURCE,instrumentation_commit=source_commit,
                run_id=directory.name,timestamp_utc=now(),environment=env,repetition=repetition,
                model=dict(target=TARGET,revision=TARGET_REVISION,tokenizer_revision=TARGET_REVISION,
                           quantization='awq_marlin',dtype='bfloat16'),
                request={k:case[k] for k in ('workload','prompt_id','prompt_hash','generation_index',
                         'input_token_count','input_token_ids_sha256')},
                condition=dict(name=condition,prefix_cache_enabled=True,cache_state=cache['cache_state'],
                    reset_verified=True,cache_hit_verified=cache['cache_hit_verified'],cache_evidence=cache),
                logits=captured['logits'],provenance=dict(command=sys.argv,engine_config=config,effective_config=flags,
                    sampling_config=settings,evidence_commit=EVIDENCE_COMMIT,
                    historical_environment_drift=drift,conditioning=conditioning,observed=observed,
                    repeat_vector_difference=captured['repeat_vector_difference'],
                    forced_grammar_seed=seeded[0],computed_tokens_before=captured['computed_tokens_before'],
                    scheduled_query_tokens=captured['scheduled_query_tokens'],
                    query_matches_forced_suffix=captured['query_matches_forced_suffix'],
                    query_token_ids_sha256=captured['query_token_ids_sha256'],
                    relevant_files=['input.json','cache.jsonl','runtime-sources.json','environment.json'],
                    synchronization=install[0]['synchronization']))
            result['request']['original_prompt_token_count'] = len(case['original_prompt_token_ids'])
            validate_result(result)
            write_new(directory/f'{condition.lower()}{suffix}.json',result)
            results.append(result)
            if condition=='CACHE_HIT':
                differences.append(captured['vector_difference'])
                write_new(directory/f'vector-difference{suffix}.json',captured['vector_difference'])
        analysis = analyze_replays(results,differences)
        write_new(directory/'analysis.json',analysis)
        print(json.dumps(dict(classification=analysis['classification'],
            within_condition_exact_repeatability=analysis['within_condition_exact_repeatability'],
            argmaxes=[r['logits']['argmax_token_id'] for r in results]),indent=2))
        if analysis['classification']=='INCONCLUSIVE':
            raise RuntimeError('Diagnostic result INCONCLUSIVE; preserve evidence for review')


def run(directory,source_commit):
    from experiments.m4_runtime import gpu_snapshot,require_gpu_release
    before = gpu_snapshot()
    if before['processes']:
        raise RuntimeError('Diagnostic requires a clean single GPU; existing jobs left untouched')
    command = [sys.executable,'-B','-m','experiments.run_m4_cache_probe','worker',
               '--out',str(directory),'--source-commit',source_commit]
    prepared = json.loads((directory/'input.json').read_text(encoding='utf8'))
    mode = prepared.get('diagnostic_mode','cache_probe')
    write_new(directory/'launch.json',dict(command=command,engine_config=engine_config(mode,prepared.get('localization_stage')),
        source_commit=source_commit,historical_source_commit=FROZEN_SOURCE,instrumentation_commit=source_commit,timestamp_utc=now()))
    status = 'FAIL'
    try:
        overrides = dict(VLLM_USE_V2_MODEL_RUNNER='0',M4_CACHE_PROBE_CONFIG=str(directory/'input.json'),
                         VLLM_WORKER_MULTIPROC_METHOD='spawn',
                         PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
        with (directory/'worker.log').open('x',encoding='utf8') as log:
            proc = subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,**overrides),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                proc.wait(timeout=1800)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait(timeout=30)
        if proc.returncode:
            raise RuntimeError(f'Diagnostic worker failed ({proc.returncode}); see worker.log')
        status = 'COMPLETE'
        print((directory/'analysis.json').read_text(encoding='utf8'))
    finally:
        after = gpu_snapshot()
        deadline = time.monotonic()+30
        while after['processes'] != before['processes'] and time.monotonic()<deadline:
            time.sleep(1)
            after = gpu_snapshot()
        try:
            require_gpu_release(before,after)
        except Exception:
            status = 'FAIL'
            raise
        finally:
            write_new(directory/'lifecycle.json',dict(status=status,gpu_before=before,gpu_after=after,timestamp_utc=now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker','analyze'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--workload',choices=list(('W1','W2','W3')),default='W1')
    parser.add_argument('--prompt-id',default='m0_04_001')
    parser.add_argument('--generation-index',type=int,default=549)
    parser.add_argument('--diagnostic-mode',choices=['cache_probe','decode_trajectory'],default='cache_probe')
    parser.add_argument('--localization-stage',choices=['lm_head_input','layer_boundaries','layer0_internal'],default=None)
    args = parser.parse_args()
    source_commit = validated_source_commit(args.source_commit)
    directory = artifact_path(args.out,kind='new_directory' if args.action=='prepare' else 'directory')
    if args.action=='prepare':
        directory.mkdir()
        case = prepare(directory,source_commit,args.workload,args.prompt_id,args.generation_index,
            args.diagnostic_mode,args.localization_stage)
        print(json.dumps({k:case[k] for k in ('prompt_id','input_token_count','input_token_ids_sha256')}))
        return
    if args.action=='run' and {q.name for q in directory.iterdir()} != {'input.json'}:
        raise ValueError('Run directory already used or contains unrelated files; preserve it and prepare a new run')
    try:
        if args.action=='run': run(directory,source_commit)
        elif args.action=='worker': worker(directory,source_commit)
        else:
            verify_run_manifest(directory,source_commit)
            read = lambda name: json.loads((directory/name).read_text(encoding='utf8'))
            if read('input.json').get('diagnostic_mode') == 'decode_trajectory':
                from experiments.m4_decode_trajectory import analyze_trajectory
                from experiments.m4_head_input_probe import HEAD_STAGE,analyze_head_input,require_completed_lifecycle
                from experiments.m4_layer_probe import LAYER_STAGE,analyze_layers
                from experiments.m4_layer0_probe import INTERNAL_STAGE,analyze_internal
                if read('input.json').get('localization_stage') in (HEAD_STAGE,LAYER_STAGE,INTERNAL_STAGE):
                    require_completed_lifecycle(read('hook-cleanup.json'),read('lifecycle.json'))
                analyze = (analyze_internal if read('input.json').get('localization_stage')==INTERNAL_STAGE else
                           analyze_layers if read('input.json').get('localization_stage')==LAYER_STAGE else
                           analyze_head_input if read('input.json').get('localization_stage')==HEAD_STAGE
                           else analyze_trajectory)
                result = analyze(read('sequential_decode_cold.json'),
                    read('forced_prefill_cold.json'),read('input.json'),expected_source_commit=source_commit)
                print(json.dumps(result,indent=2))
                if result['classification']=='INCONCLUSIVE': raise SystemExit(1)
                return
            result = analyze_replays([read(name+'.json') for name in
                ('cold','cache_hit','cold_replay_2','cache_hit_replay_2')],
                [read('vector-difference'+suffix+'.json') for suffix in ('','_replay_2')])
            print(json.dumps(result,indent=2))
            if result['classification']=='INCONCLUSIVE': raise SystemExit(1)
    except Exception as exc:
        failure = directory/f'{args.action}-failure.json'
        if args.action!='analyze' and not failure.exists():
            write_new(failure,dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise
    finally:
        if args.action=='run':
            # Append-only manifest; no canonical evidence files are rewritten.
            write_new(directory/'manifest.json',dict(schema_version=SCHEMA,
                instrumentation_commit=source_commit,source_commit=source_commit,historical_source_commit=FROZEN_SOURCE,
                files={q.name:hashlib.sha256(q.read_bytes()).hexdigest()
                       for q in sorted(directory.iterdir()) if q.is_file()}))


if __name__=='__main__':
    main()
