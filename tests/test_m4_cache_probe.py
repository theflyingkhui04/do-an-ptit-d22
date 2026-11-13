"""CPU contracts only; no model load, vLLM import, CUDA or network calls."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from experiments import m4_cache_probe as probe
from experiments import m4_cache_probe_hooks as hooks
from experiments import run_m4_cache_probe as runner


@pytest.fixture(scope='module')
def case():
    return probe.load_input()


def test_exact_preserved_input(case):
    assert case['prompt_id']=='m0_04_001' and case['workload']=='W1'
    assert case['generation_index']==549
    assert len(case['original_prompt_token_ids'])==1522
    assert len(case['generated_prefix_token_ids'])==549
    assert case['forced_input_token_ids']==case['original_prompt_token_ids']+case['generated_prefix_token_ids']
    assert case['input_token_count']==2071
    # SHA256 over the compact ASCII JSON integer array, no trailing newline.
    assert case['input_token_ids_sha256']=='3f52b8b8ee17eca82386c6e5ab6fa0510ee089e71dbffe60cf0efcabdd5fc75a'
    assert (case['reference_next_token_id'],case['candidate_next_token_id'])==(4164,3752)
    assert case['historical_request']['response_format']=={'type':'json_object'}
    assert runner.sampling_config(case['historical_request'])==dict(
        temperature=0.0,seed=42,n=1,max_tokens=1024,structured_outputs={'json_object':True})


def test_no_text_roundtrip_or_metadata_tensor_changes():
    cap = dict(workload='W1',prompt_id='p',treatment='vanilla',repeat=0,
        source_commit=probe.FROZEN_SOURCE,target_model=probe.TARGET,target_revision=probe.TARGET_REVISION,
        tokenizer_revision=probe.TARGET_REVISION,temperature=0,seed=42,complete=True,
        request_id='r0',token_ids=[10,11,12],prompt_hash='h',max_tokens=1024,stopping='EOS')
    repeat = dict(cap,repeat=1,request_id='r1',token_ids=[10,11,99])
    # Deliberately unusable decoded text must never be accessed/tokenized.
    raw = [dict(id='r0',prompt_token_ids=[1,2],choices=[dict(token_ids=[10,11,12],message=None)])]
    a = probe.construct_input([cap,repeat],raw,'W1','p',2)
    cap.update(max_tokens=7,stopping='different metadata')
    b = probe.construct_input([cap,repeat],raw,'W1','p',2)
    assert a['forced_input_token_ids']==b['forced_input_token_ids']==[1,2,10,11]
    assert a['input_token_ids_sha256']==b['input_token_ids_sha256']
    with pytest.raises(ValueError,match='Raw emitted'):
        probe.construct_input([cap,repeat],[dict(raw[0],choices=[dict(token_ids=[9])])],'W1','p',2)
    with pytest.raises(ValueError,match='Missing/ambiguous'):
        probe.construct_input([cap,cap,repeat],raw,'W1','p',2)
    with pytest.raises(ValueError,match='outside'):
        probe.construct_input([cap,repeat],raw,'W1','p',3)
    repeat['token_ids']=[0,11,99]
    with pytest.raises(ValueError,match='share requested prefix'):
        probe.construct_input([cap,repeat],raw,'W1','p',2)


@pytest.mark.parametrize('tokens',[[],[-1],[True],[1.0],['1']])
def test_hash_rejects_non_emitted_ids(tokens):
    with pytest.raises(ValueError): probe.token_hash(tokens)


def event(cached=0,**kwargs):
    return dict(event='cache_lookup',request_id='r',input_token_count=2071,
        input_token_ids_sha256='hash',prefix_cache_enabled=True,skip_reading_prefix_cache=False,
        cached_tokens=cached,**kwargs)


@pytest.mark.parametrize('condition,cached,valid',[
    ('COLD',0,True),('COLD',1520,False),('CACHE_HIT',1520,True),
    ('CACHE_HIT',0,False),('CACHE_HIT',2048,False),('unknown',0,False)])
def test_condition_requires_runtime_lookup(condition,cached,valid):
    call=lambda:probe.cache_evidence([event(cached)],'r',condition,1522,2071,'hash')
    if valid: assert call()['cached_tokens']==cached
    else:
        with pytest.raises(ValueError): call()


def test_cache_identity_not_inferred_from_intent():
    for events in ([],[dict(event(1520),request_id='other')],
                   [dict(event(1520),input_token_ids_sha256='different')],
                   [dict(event(1520),skip_reading_prefix_cache=True)]):
        with pytest.raises(ValueError):
            probe.cache_evidence(events,'r','CACHE_HIT',1522,2071,'hash')


def result(condition='COLD',argmax=4164):
    top=[dict(token_id=argmax,logit=10.)]+[dict(token_id=i,logit=9.-i) for i in range(9)]
    return dict(schema_version=probe.SCHEMA,diagnostic_name='vanilla_cold_cache_hit',
        source_commit='a'*40,instrumentation_commit='a'*40,run_id='test',timestamp_utc='test',
        environment=dict(vllm='0.30.0',gpu_name='NVIDIA A100',capability=[8,0],driver='driver',
                         cuda='13.0',python='3.13.16',pytorch='2.13.0+cu130'),
        model=dict(target=probe.TARGET,revision=probe.TARGET_REVISION,tokenizer_revision=probe.TARGET_REVISION,
                   dtype='bfloat16',quantization='awq_marlin'),
        request=dict(input_token_ids_sha256='f'*64,input_token_count=2071,generation_index=549,
                     prompt_hash='1'*64,original_prompt_token_count=1522,workload='W1',prompt_id='m0_04_001'),
        condition=dict(name=condition,prefix_cache_enabled=True,
            reset_verified=True,
            cache_hit_verified=condition=='CACHE_HIT',cache_evidence=dict(cached_tokens=0 if condition=='COLD' else 1520,
                lookup_records=[dict(event(0 if condition=='COLD' else 1520),input_token_ids_sha256='f'*64)])),
        logits=dict(measurement_point=probe.MEASUREMENT,dtype='torch.float32',nan_count=0,inf_count=0,
            argmax_token_id=argmax,top_tokens=top,logit_token_4164=10.,logit_token_3752=9.,
            difference_4164_minus_3752=1.,top1_minus_top2=1.),
        provenance=dict(engine_config={'same':True},observed={'request_id':'r'},
            query_matches_forced_suffix=True,query_token_ids_sha256='e'*64,
            computed_tokens_before=2048 if condition=='COLD' else 1520,
            scheduled_query_tokens=23 if condition=='COLD' else 551,sampling_config=dict(
            temperature=0,seed=42,n=1,max_tokens=1024,structured_outputs={'json_object':True})))


@pytest.mark.parametrize('argmax,delta,expected',[
    (3752,0.1,'H_CACHE_SUPPORTED'),(4164,0.,'H_CACHE_NOT_REPRODUCED'),
    (4164,0.01,'INCONCLUSIVE')])
def test_analysis_decision_matrix(argmax,delta,expected):
    cold,hit=result(),result('CACHE_HIT',argmax)
    difference=dict(vocabulary_equal=True,max_absolute_difference=delta,mean_absolute_difference=delta/2)
    assert probe.analyze(cold,hit,difference)['classification']==expected
    assert probe.analyze(cold,hit,difference)['classification'] != 'ROOT_CAUSE_PROVEN'


@pytest.mark.parametrize('damage',['input','environment','hit','nan','stage','flags','margin','source','sampling','owner'])
def test_analysis_fails_closed(damage):
    cold,hit=result(),result('CACHE_HIT',3752)
    if damage=='input': hit['request']['input_token_ids_sha256']='changed'
    elif damage=='environment': hit['environment']['driver']='different'
    elif damage=='hit': hit['condition']['cache_hit_verified']=False
    elif damage=='nan': hit['logits']['nan_count']=1
    elif damage=='stage': hit['logits']['measurement_point']='API logprobs'
    elif damage=='flags': hit['provenance']['engine_config']={}
    elif damage=='margin': hit['logits']['top1_minus_top2']=99
    elif damage=='sampling': hit['provenance']['sampling_config']['seed']=1
    elif damage=='owner': hit['provenance']['observed']['request_id']='other'
    else: hit['source_commit']='b'*40
    assert probe.analyze(cold,hit,dict(vocabulary_equal=True,max_absolute_difference=1))['classification']=='INCONCLUSIVE'


def test_schema_serialization():
    obj=json.loads(json.dumps(result(),allow_nan=False))
    probe.validate_result(obj)
    del obj['request']
    with pytest.raises(ValueError): probe.validate_result(obj)


def test_replay_controls_reject_order_and_within_condition_instability():
    results=[result(),result('CACHE_HIT',3752),result(),result('CACHE_HIT',3752)]
    stable=dict(vocabulary_equal=True,max_absolute_difference=0.,mean_absolute_difference=0.)
    difference=dict(vocabulary_equal=True,max_absolute_difference=1.,mean_absolute_difference=.1)
    for r in results[2:]: r['provenance']['repeat_vector_difference']=stable
    assert probe.analyze_replays(results,[difference,difference])['classification']=='H_CACHE_SUPPORTED'
    results[2]['provenance']['repeat_vector_difference']=dict(stable,max_absolute_difference=.01)
    assert probe.analyze_replays(results,[difference,difference])['classification']=='INCONCLUSIVE'
    assert probe.analyze_replays(results[:2],[difference])['classification']=='INCONCLUSIVE'


def test_nonfinite_vector_statistics_fail_closed():
    for value in (-1.,float('nan'),float('inf')):
        d=dict(vocabulary_equal=True,max_absolute_difference=value,mean_absolute_difference=0.)
        assert probe.analyze(result(),result('CACHE_HIT'),d)['classification']=='INCONCLUSIVE'


def test_hooks_inactive_without_explicit_configuration(monkeypatch):
    monkeypatch.delenv('M4_CACHE_PROBE_CONFIG',raising=False)
    with pytest.raises(RuntimeError,match='Explicit diagnostic'): hooks.configuration()
    assert 'speculative_config' not in runner.engine_config()
    assert runner.engine_config()['async_scheduling'] is True
    assert runner.engine_config()['max_num_seqs']==32
    assert runner.engine_config()['max_num_batched_tokens']==2048


def test_pinned_source_hash_fails_closed(tmp_path):
    import hashlib
    path=tmp_path/'pinned.py';path.write_bytes(b'exact pinned bytes\n')
    manifest=dict(files=[dict(path='pinned.py',sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    assert hooks.verify_source_tree(tmp_path,manifest) is manifest
    path.write_bytes(b'changed runtime\n')
    with pytest.raises(RuntimeError,match='Unreviewed pinned source'):
        hooks.verify_source_tree(tmp_path,manifest)


def test_prepared_artifact_is_self_contained_and_append_only(case,tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'load_input',lambda *args:deepcopy(case))
    prepared=runner.prepare(tmp_path,'a'*40,'W1','m0_04_001',549)
    stored=json.loads((tmp_path/'input.json').read_text(encoding='utf8'))
    assert stored['source_commit']==stored['instrumentation_commit']=='a'*40
    assert stored['historical_source_commit']==probe.FROZEN_SOURCE
    assert stored['forced_input_token_ids']==case['forced_input_token_ids']
    assert stored==prepared
    with pytest.raises(FileExistsError): runner.prepare(tmp_path,'a'*40,'W1','m0_04_001',549)


def test_completed_run_cannot_be_modified_by_rerun(tmp_path,monkeypatch):
    (tmp_path/'input.json').write_text('{}')
    (tmp_path/'manifest.json').write_text('preserved')
    monkeypatch.setattr(runner,'validated_source_commit',lambda sha:sha)
    monkeypatch.setattr(runner,'artifact_path',lambda *args,**kwargs:tmp_path)
    monkeypatch.setattr(runner,'run',lambda *args:pytest.fail('Existing run reached GPU entrypoint'))
    monkeypatch.setattr('sys.argv',['probe','run','--out',str(tmp_path),'--source-commit','a'*40])
    with pytest.raises(ValueError,match='already used'): runner.main()
    assert sorted(q.name for q in tmp_path.iterdir())==['input.json','manifest.json']
    assert (tmp_path/'manifest.json').read_text()=='preserved'


def test_run_manifest_detects_mutation_and_extra_artifacts(tmp_path):
    import hashlib
    (tmp_path/'input.json').write_text('preserved')
    manifest=dict(source_commit='a'*40,files={'input.json':hashlib.sha256(b'preserved').hexdigest()})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    probe.verify_run_manifest(tmp_path,'a'*40)
    (tmp_path/'extra.json').write_text('{}')
    with pytest.raises(ValueError,match='inventory'):probe.verify_run_manifest(tmp_path,'a'*40)


def test_scheduler_observes_identity_and_seeds_exact_grammar(monkeypatch,case,tmp_path):
    config=dict(case,cache_trace=str(tmp_path/'trace.jsonl'))
    monkeypatch.setattr(hooks,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(hooks,'configuration',lambda:config)
    accepted=[]
    req=NS(request_id='r',prompt_token_ids=case['forced_input_token_ids'],is_prefill_chunk=False,
           skip_reading_prefix_cache=False,structured_output_request=NS(
               grammar=NS(accept_tokens=lambda rid,tokens:accepted.append((rid,tokens)) or True)))
    original_tuple=(NS(get_block_ids=lambda:([1,2],)),32,0)
    class Base:
        def __init__(self):
            self.vllm_config=NS(speculative_config=None)
            self.cache_config=NS(enable_prefix_caching=True)
            self.kv_cache_manager=NS(get_computed_blocks=lambda request:original_tuple,enable_caching=True)
            self.block_size=16;self.requests={'r':req};self.calls=0
        def _update_after_schedule(self,output): self.calls+=1
        def _free_request(self,request): return 'base freed'
    class Scheduler(hooks.CacheProbeSchedulerMixin,Base): pass
    scheduler=Scheduler()
    assert scheduler.kv_cache_manager.get_computed_blocks(req) is original_tuple
    output=NS(num_scheduled_tokens={'r':2071})
    scheduler._update_after_schedule(output);scheduler._update_after_schedule(output)
    assert accepted==[('r',case['generated_prefix_token_ids'])]
    assert scheduler.calls==2 and req.structured_output_request.reasoning_ended
    assert scheduler._free_request(req)=='base freed' and not scheduler._m4_seeded
    assert json.loads((tmp_path/'trace.jsonl').read_text().splitlines()[0])['cached_tokens']==32
    scheduler.m4_probe_cache_uninstall()
    assert scheduler.kv_cache_manager.get_computed_blocks(req) is original_tuple
    before=(tmp_path/'trace.jsonl').read_bytes()
    scheduler._update_after_schedule(output)
    assert (tmp_path/'trace.jsonl').read_bytes()==before
    scheduler.m4_probe_cache_uninstall()
    assert id(scheduler.vllm_config) not in hooks._schedulers


def test_worker_capture_returns_original_tensor_and_defers_transfer(monkeypatch,case):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(hooks,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(hooks,'configuration',lambda:case)
    tokens=case['forced_input_token_ids'];original=torch.zeros((1,5000))
    original[0,4164]=12;original[0,3752]=11
    model=NS(compute_logits=lambda *args,**kwargs:original,
             logits_processor=NS(scale=1.0,soft_cap=None,logits_as_input=False))
    output=NS(num_scheduled_tokens={'r':len(tokens)})
    runner_object=type('GPUModelRunner',(),{})()
    type(runner_object).__module__='vllm.v1.worker.gpu_model_runner'
    runner_object.model=model;runner_object.use_async_scheduling=True
    runner_object.input_batch=NS(req_ids=['r'],num_computed_tokens_cpu=[0])
    runner_object.input_ids=NS(gpu=torch.tensor(tokens,dtype=torch.int32))
    runner_object.requests={'r':NS(prompt_token_ids=tokens)}
    runner_object._sample=lambda logits:logits
    def sample_tokens(grammar):
        # The actual pinned method clears this before invoking _sample.
        runner_object.execute_model_state=None
        return runner_object._sample(original)
    runner_object.sample_tokens=sample_tokens
    def execute(out):
        logits=model.compute_logits(None)
        runner_object.execute_model_state=(out,)
        return logits
    runner_object.execute_model=execute
    worker=type('Worker',(hooks.CacheProbeWorkerExtension,),{})()
    worker.model_runner=runner_object;worker.vllm_config=NS(speculative_config=None)
    owner=NS(vllm_config=worker.vllm_config,m4_probe_cache_uninstall=lambda:None)
    # Weak ownership registry requires a weak-referenceable scheduler instance.
    scheduler=type('Scheduler',(),{})()
    scheduler.__dict__.update(vars(owner))
    monkeypatch.setitem(hooks._schedulers,id(worker.vllm_config),scheduler)
    originals=[runner_object.execute_model,model.compute_logits,runner_object._sample,runner_object.sample_tokens]
    worker.m4_probe_install();worker.m4_probe_arm('COLD',tokens)
    returned=runner_object.execute_model(output)
    assert returned is original
    # Simulate grammar masking after raw capture. Raw snapshot must survive.
    original[0,4164]=-float('inf')
    assert runner_object.sample_tokens(None) is original
    captured=worker.m4_probe_collect()
    assert captured['logits']['argmax_token_id']==4164
    assert captured['logits']['grammar_masked_argmax_token_id']==3752
    assert captured['input_token_ids_sha256']==case['input_token_ids_sha256']
    assert captured['query_matches_forced_suffix'] is True
    assert worker._m4_probe['pending'] is None
    # A failed collect retains a snapshot until the owning finally tears down.
    original[0,4164]=12
    worker.m4_probe_arm('CACHE_HIT',tokens)
    runner_object.input_ids.gpu[0]=-1
    runner_object.execute_model(output);runner_object.sample_tokens(None)
    state=worker._m4_probe
    with pytest.raises(RuntimeError,match='Actual GPU query tokens'):
        worker.m4_probe_collect()
    worker.m4_probe_uninstall()
    assert state=={}
    assert [runner_object.execute_model,model.compute_logits,runner_object._sample,runner_object.sample_tokens]==originals
    assert not hasattr(worker,'_m4_probe')


def test_abort_after_output_preserves_params_and_ownership():
    calls=[]
    class Engine:
        def has_unfinished_requests(self): return bool(calls)
        def add_request(self,label,prompt,params):
            calls.append(('add',label,prompt,params));return 'internal'
        def step(self): return [NS(outputs=[NS(token_ids=[4164])])]
        def abort_request(self,ids,internal): calls.append(('abort',ids,internal))
    params=object()
    observed=runner.one_request(NS(llm_engine=Engine()),[1,2],params,'COLD')
    assert calls[0][3] is params
    assert calls[-1]==('abort',['internal'],True)
    assert observed['first_output_token_id']==4164


@pytest.mark.parametrize('failure',[None,'capture','install','cleanup','shutdown','unconfirmed'])
def test_engine_teardown_always_attempts_shutdown(tmp_path,failure):
    calls=[]
    class LLM:
        llm_engine=NS(engine_core=NS(shutdown=lambda:self_shutdown()))
        def collective_rpc(self,method):
            calls.append(method)
            if failure=='cleanup': raise RuntimeError('cleanup')
            return [dict(restored=failure!='unconfirmed',captures_released=True,cache_observer_restored=True)]
    def self_shutdown():
        calls.append('shutdown')
        if failure=='shutdown': raise RuntimeError('shutdown')
    def run():
        with runner.diagnostic_engine(LLM(),tmp_path):
            if failure in ('capture','install'): raise ValueError(failure)
    if failure:
        with pytest.raises((ValueError,RuntimeError)): run()
    else: run()
    assert calls==['m4_probe_uninstall','shutdown']
    cleanup=json.loads((tmp_path/'hook-cleanup.json').read_text())
    assert cleanup['hooks_restored']==(failure not in ('cleanup','unconfirmed'))
    assert cleanup['engine_shutdown']==(failure!='shutdown')


@pytest.mark.parametrize('partial',[False,True])
def test_worker_teardown_restores_descriptors_overrides_and_releases_state(monkeypatch,case,partial):
    monkeypatch.setattr(hooks,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(hooks,'configuration',lambda:case)
    class Model:
        logits_processor=NS(scale=1.,soft_cap=None,logits_as_input=False)
        def compute_logits(self,*args): return 'original logits'
    class GPUModelRunner:
        use_async_scheduling=True
        def execute_model(self,*args): raise ValueError('capture')
        def _sample(self,*args): return 'original sample'
        def sample_tokens(self,*args): return 'original sampling'
    GPUModelRunner.__module__='vllm.v1.worker.gpu_model_runner'
    obj=GPUModelRunner();obj.model=Model()
    obj._sample=lambda *args:'preexisting override'
    original_sample=obj._sample
    worker=type('Worker',(hooks.CacheProbeWorkerExtension,),{})()
    worker.model_runner=obj;worker.vllm_config=NS(speculative_config=None)
    calls=[]
    scheduler=type('Scheduler',(),{})()
    scheduler.vllm_config=worker.vllm_config
    scheduler.m4_probe_cache_uninstall=lambda:calls.append('cache restored')
    monkeypatch.setitem(hooks._schedulers,id(worker.vllm_config),scheduler)
    if partial:
        replace=hooks._InstanceHooks.replace
        def interrupted(self,owner,name,value):
            replace(self,owner,name,value)
            if name=='_sample': raise ValueError('partial install')
        monkeypatch.setattr(hooks._InstanceHooks,'replace',interrupted)
        with pytest.raises(ValueError,match='partial install'): worker.m4_probe_install()
    else:
        worker.m4_probe_install()
        state=worker._m4_probe
        state['pending']={'raw':object()};state['previous']=object();state['repeats']={'COLD':object()}
        with pytest.raises(ValueError,match='capture'): obj.execute_model(NS())
        assert state['output'] is None
        worker.m4_probe_uninstall()
        assert state=={}
    assert not hasattr(worker,'_m4_probe')
    assert 'execute_model' not in vars(obj) and 'sample_tokens' not in vars(obj)
    assert 'compute_logits' not in vars(obj.model)
    assert obj._sample is original_sample
    assert obj.model.compute_logits()=='original logits'
    worker.m4_probe_uninstall()
    assert len(calls)==2


def test_hook_restore_attempts_other_methods_and_retries():
    class Owner:
        reject=False
        def __setattr__(self,name,value):
            if name=='second' and self.reject: raise RuntimeError('restore failure')
            super().__setattr__(name,value)
    obj=Owner();obj.first='first';obj.second='second'
    registry=hooks._InstanceHooks()
    registry.replace(obj,'first','hook1');registry.replace(obj,'second','hook2')
    obj.reject=True
    with pytest.raises(RuntimeError,match='restoration failed'): registry.restore()
    assert obj.first=='first' and obj.second=='hook2'
    obj.reject=False;registry.restore();registry.restore()
    assert obj.second=='second'


@pytest.mark.parametrize('shutdown_error',[False,True])
def test_scheduler_shutdown_restores_observer_before_base_error(monkeypatch,case,shutdown_error):
    monkeypatch.setattr(hooks,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(hooks,'configuration',lambda:case)
    original=lambda request:('unchanged',0,0)
    class Base:
        def __init__(self):
            self.vllm_config=NS(speculative_config=None)
            self.cache_config=NS(enable_prefix_caching=True)
            self.kv_cache_manager=NS(get_computed_blocks=original)
        def shutdown(self):
            assert self.kv_cache_manager.get_computed_blocks is original
            if shutdown_error: raise ValueError('base shutdown')
    class Scheduler(hooks.CacheProbeSchedulerMixin,Base): pass
    scheduler=Scheduler()
    if shutdown_error:
        with pytest.raises(ValueError,match='base shutdown'): scheduler.shutdown()
    else: scheduler.shutdown()
    assert scheduler._m4_config is None and not scheduler._m4_active
    assert id(scheduler.vllm_config) not in hooks._schedulers


def test_worker_cleanup_releases_snapshots_and_observer_when_restore_fails(monkeypatch):
    calls=[]
    worker=type('Worker',(hooks.CacheProbeWorkerExtension,),{})()
    worker.vllm_config=object()
    scheduler=type('Scheduler',(),{})()
    scheduler.vllm_config=worker.vllm_config
    scheduler.m4_probe_cache_uninstall=lambda:calls.append('cache restored')
    monkeypatch.setitem(hooks._schedulers,id(worker.vllm_config),scheduler)
    def fail(): raise RuntimeError('restoration failed')
    worker._m4_hooks=NS(restore=fail)
    state=worker._m4_probe={'pending':object(),'repeats':{'COLD':object()}}
    with pytest.raises(RuntimeError,match='restoration failed'): worker.m4_probe_uninstall()
    assert state=={} and not hasattr(worker,'_m4_probe')
    assert calls==['cache restored']
