"""CPU contracts only: these tests make no claim about GPU numerics."""
from copy import deepcopy
import hashlib
import json
import struct
from types import SimpleNamespace as NS

import pytest

from experiments import m4_cache_probe as probe
from experiments import m4_cache_probe_hooks as hooks
from experiments import m4_decode_trajectory as traj
from experiments import run_m4_cache_probe as runner
from experiments import m4_head_input_probe as head


@pytest.fixture(scope='module')
def case():
    return dict(probe.load_input(),diagnostic_mode='decode_trajectory',
        source_commit='a'*40,instrumentation_commit='a'*40,run_id='cpu-test')


def test_exact_reference_and_explicit_prepare(case,tmp_path):
    p=runner.prepare(tmp_path,'a'*40,'W1','m0_04_001',549,'decode_trajectory')
    assert p['diagnostic_mode']=='decode_trajectory'
    assert p['original_prompt_token_ids']==case['original_prompt_token_ids']
    assert probe.token_hash(p['generated_prefix_token_ids'])=='ba579161a70b4aa19b19b4262ceb73c1c540dbf51ce1bce4c0dfdf75e752622b'
    assert traj.prefix_identity(p['original_prompt_token_ids'],p['generated_prefix_token_ids'])['logical_input_sha256']==case['input_token_ids_sha256']
    with pytest.raises(ValueError,match='scoped'):
        runner.prepare(tmp_path,'a'*40,'W2','m0_04_001',549,'decode_trajectory')


def test_default_serving_and_existing_probe_unchanged(monkeypatch):
    monkeypatch.delenv('M4_CACHE_PROBE_CONFIG',raising=False)
    with pytest.raises(RuntimeError,match='Explicit diagnostic'): hooks.configuration()
    normal=runner.engine_config(); explicit=runner.engine_config('decode_trajectory')
    assert normal['worker_extension_cls'].endswith('CacheProbeWorkerExtension')
    assert {k for k in normal if normal[k]!=explicit[k]}=={'scheduler_cls','worker_extension_cls'}
    with pytest.raises(ValueError): runner.engine_config('typo')


def test_import_has_no_runtime_or_tensor_side_effects():
    import subprocess
    import sys
    subprocess.run([sys.executable,'-B','-c',
        'import sys; import experiments.m4_decode_trajectory; '
        'assert "vllm" not in sys.modules; assert "torch" not in sys.modules'],
        cwd=probe.ROOT,check=True)


@pytest.mark.parametrize('tokens,index',[([],None),([10],None),([10,11],None),
    ([99],0),([10,99,12],1),([10,11,12],2),([10,11,99,42],2)])
def test_first_boundary_tracking(tokens,index):
    assert traj.first_boundary(tokens,[10,11,12],2)==index


@pytest.mark.parametrize('target',[-1,3,True])
def test_invalid_target_fails_closed(target):
    with pytest.raises(ValueError): traj.first_boundary([],[10,11,12],target)


def test_gpu_style_latch_keeps_first_sample_mismatch_and_raw_masked_separate():
    import torch
    raw=torch.tensor([9.,8.,7.]); masked=torch.tensor([-float('inf'),8.,7.])
    capture=traj.FirstMismatchCapture([1,0,2],2)
    capture.record(0,raw,masked,torch.tensor([[1]]))
    assert int(capture.selected['index'])==-1  # raw argmax != sampler is not a mismatch.
    capture.record(1,raw,masked,torch.tensor([[1]]))
    before={k:v.clone() for k,v in capture.selected.items()}
    capture.record(2,raw+20,masked+20,torch.tensor([[2]]))
    assert all(torch.equal(v,capture.selected[k]) for k,v in before.items())
    assert int(capture.selected['index'])==1 and int(capture.selected['token'])==1
    assert int(capture.selected['raw'].argmax())==0
    assert int(capture.selected['masked'].argmax())==1
    with pytest.raises(RuntimeError,match='Non-contiguous'): capture.record(2,raw,masked,torch.tensor([[2]]))


def test_latch_selects_final_boundary_without_mismatch():
    import torch
    capture=traj.FirstMismatchCapture([1,1],1)
    for i in range(2): capture.record(i,torch.tensor([0.,1.]),torch.tensor([0.,1.]),torch.tensor([[1]]))
    assert int(capture.selected['index'])==1


def test_driver_aborts_on_first_delivered_mismatch_without_forcing():
    calls=[]; params=object()
    class Engine:
        def has_unfinished_requests(self):return bool(calls)
        def add_request(self,label,prompt,settings):
            assert prompt=={'prompt_token_ids':[1,2]} and settings is params
            calls.append('add');return 'internal-id'
        def step(self):
            calls.append('step')
            # Cumulative output can include in-flight work after the first mismatch.
            return [NS(request_id='external-label',outputs=[NS(token_ids=[10,99,42])])]
        def abort_request(self,ids,internal):calls.append(('abort',ids,internal))
    observed=traj.sequential_request(NS(llm_engine=Engine()),[1,2],params,[10,11,12],2,'external-label')
    assert observed['generation_index']==1 and observed['observed_tokens']==[10,99]
    assert observed['delivered_tokens_after_boundary']==[42]
    assert calls==['add','step',('abort',['internal-id'],True)]


@pytest.mark.parametrize('exception',[False,True])
def test_driver_abort_on_early_eos_and_exception(exception):
    calls=[]
    class Engine:
        def has_unfinished_requests(self):return len(calls)==1
        def add_request(self,*args):calls.append('add');return 'id'
        def step(self):
            calls.append('step')
            if exception:raise ValueError('step failed')
            return [NS(request_id='label',outputs=[NS(token_ids=[10])])]
        def abort_request(self,*args,**kwargs):calls.append('abort')
    with pytest.raises((ValueError,RuntimeError)):
        traj.sequential_request(NS(llm_engine=Engine()),[1],object(),[10,11],1,'label')
    assert calls[-1]=='abort'


def capture(case,condition,argmax=4164,index=549,delta=0.):
    import torch
    original=case['original_prompt_token_ids'];generated=case['generated_prefix_token_ids'][:index]
    raw=torch.arange(traj.VOCABULARY_SIZE,dtype=torch.float32)*.00001+delta
    raw[4164]=9.;raw[3752]=8.
    if argmax==3752:raw[3752]=10.
    summary,cpu=hooks.logits_summary(raw,raw)
    words=[-1]*((len(raw)+31)//32)
    bitmask=dict(shape=[1,len(words)],dtype='int32',values=[words],
        sha256=hashlib.sha256(struct.pack('<'+'i'*len(words),*words)).hexdigest())
    prompt=original if condition==traj.SEQUENTIAL else original+generated
    event=dict(event='cache_lookup',request_id='rid',input_token_count=len(prompt),
        input_token_ids_sha256=probe.token_hash(prompt),prefix_cache_enabled=True,
        skip_reading_prefix_cache=False,cached_tokens=0)
    reference=case['generated_prefix_token_ids']+[4164]
    return dict(schema_version=traj.SCHEMA,condition=condition,
        identity=traj.prefix_identity(original,generated),original_prompt_token_ids=original,
        generated_prefix_token_ids=generated,full_logical_input_token_ids=original+generated,
        historical_token=reference[index],observed_token=argmax,logits=summary,
        raw_full_logits=cpu.tolist(),grammar_masked_full_logits=cpu.tolist(),
        state=dict(request_id='rid',computed_tokens_before=len(original)+index-1,
            scheduled_query_tokens=1,execution=dict(cudagraph_mode='FULL',padded_tokens=1,
                padded_requests=1,uniform_decode=True,microbatching=False),
            block_ids=[[1]],grammar_bitmask=bitmask,grammar_state=dict(backend_class='xgrammar',
                grammar_class='XGrammarGrammar',reasoner_class=None,num_processed_tokens=index,reasoning_ended=True)),
        cache=probe.cache_evidence([event],'rid','COLD',len(original),len(prompt),probe.token_hash(prompt)),
        reset_verified=True,environment=dict(vllm='0.30.0',pytorch='2.13.0+cu130',cuda='13.0',
            capability=[8,0],gpu_name='NVIDIA A100',driver='580.82.07',python='3.13.16',
            runtime_overrides={k:None for k in ('VLLM_USE_V2_MODEL_RUNNER','VLLM_WORKER_MULTIPROC_METHOD',
                'VLLM_ATTENTION_BACKEND','VLLM_BATCH_INVARIANT','VLLM_USE_FLASHINFER_SAMPLER',
                'NVIDIA_TF32_OVERRIDE','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')}),
        engine_config=runner.engine_config('decode_trajectory'),effective_config=runner.expected_effective_config(),
        sampling_config=runner.sampling_config(case['historical_request']),source_commit='a'*40,
        model=dict(target=probe.TARGET,revision=probe.TARGET_REVISION,tokenizer_revision=probe.TARGET_REVISION,
            dtype='bfloat16',quantization='awq_marlin'),
        instrumentation_commit='a'*40,
        run_id=case['run_id'],timestamp_utc='2026-11-14T00:00:00+00:00',
        observed=(dict(request_id='rid',generation_index=index,observed_tokens=generated+[argmax],
                       delivered_tokens_after_boundary=[]) if condition==traj.SEQUENTIAL else
                  dict(request_id='rid',first_output_token_id=argmax,observed_output_tokens=[argmax])))


@pytest.mark.parametrize('s,f,delta,index,expected',[
    (4164,3752,0.,549,'H_TRAJECTORY_SUPPORTED'),
    (3752,3752,.01,549,'HISTORICAL_TRAJECTORY_NOT_REPRODUCED'),
    (3752,3752,0.,549,'H_TRAJECTORY_NOT_SUPPORTED'),
    (3752,4164,.01,5,'EARLIER_DIVERGENCE_OBSERVED'),
    (4164,4164,.01,549,'INCONCLUSIVE')])
def test_decision_matrix(case,s,f,delta,index,expected):
    a,b=capture(case,traj.SEQUENTIAL,s,index),capture(case,traj.FORCED,f,index,delta)
    if index<549 and a['historical_token']==s:
        pytest.fail('Fixture must actually diverge')
    result=traj.analyze_trajectory(a,b,case)
    assert result['classification']==expected,result
    assert result['root_cause']=='ROOT_CAUSE_NOT_PROVEN' and result['full_sweep']=='FULL_SWEEP_BLOCKED'


@pytest.mark.parametrize('damage',['hash','prefix','raw','masked','sample','source','env','grammar',
    'bitmask','cache','reset','query','raw_stage','top','flags','model'])
def test_schema_and_analysis_fail_closed(case,damage):
    a,b=capture(case,traj.SEQUENTIAL),capture(case,traj.FORCED,3752)
    if damage=='hash':b['identity']['logical_input_sha256']='f'*64
    elif damage=='prefix':b['full_logical_input_token_ids'][0]=-1
    elif damage=='raw':b['raw_full_logits']=[]
    elif damage=='masked':b['grammar_masked_full_logits'][4164]=None
    elif damage=='sample':b['observed_token']=4164
    elif damage=='source':b['source_commit']='b'*40
    elif damage=='env':b['environment']['vllm']='latest'
    elif damage=='grammar':b['state']['grammar_state']['num_processed_tokens']=0
    elif damage=='bitmask':b['state']['grammar_bitmask']['sha256']='0'*64
    elif damage=='cache':b['cache']['cached_tokens']=16
    elif damage=='reset':b['reset_verified']=False
    elif damage=='query':b['state']['scheduled_query_tokens']=3
    elif damage=='raw_stage':b['logits']['measurement_point']='API logprobs'
    elif damage=='top':b['logits']['top_tokens'][0]['logit']=99.
    elif damage=='model':b['model']['revision']='b'*40
    else:b['engine_config']['async_scheduling']=False
    assert traj.analyze_trajectory(a,b,case)['classification']=='INCONCLUSIVE'


def test_full_vector_json_roundtrip_and_independent_reduction(case,tmp_path):
    a,b=capture(case,traj.SEQUENTIAL),capture(case,traj.FORCED,3752)
    path=tmp_path/'capture.json';path.write_text(json.dumps(a,allow_nan=False))
    restored=json.loads(path.read_text());traj.validate_capture(restored,case)
    result=traj.analyze_trajectory(restored,b,case)
    d=[abs(x-y) for x,y in zip(restored['raw_full_logits'],b['raw_full_logits'])]
    assert result['vector_difference']==dict(max_absolute_difference=max(d),
        mean_absolute_difference=sum(d)/len(d),vocabulary_size=len(d))


@pytest.mark.parametrize('boundary_index,fail_sampling,invalid_capture,localize',[
    (i,fail,bad,enabled) for enabled in (False,True) for i,fail,bad in
    [(0,False,None),(1,False,None),(548,False,None),(549,False,None),
     (1,True,None),(1,False,'prefix'),(1,False,'nonfinite')]]+[(1,False,'head_nonfinite',True)])
def test_hook_lifecycle_and_first_mismatch_with_async_later_forward(monkeypatch,tmp_path,boundary_index,fail_sampling,invalid_capture,localize):
    import numpy as np
    import torch
    config=dict(diagnostic_mode='decode_trajectory',original_prompt_token_ids=[1,2],
        generated_prefix_token_ids=[10]*549,reference_next_token_id=4164,generation_index=549,
        source_commit='a'*40,cache_trace=str(tmp_path/'cache.jsonl'))
    if localize:config['localization_stage']=head.HEAD_STAGE
    monkeypatch.setattr(traj,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(traj,'configuration',lambda:config)
    reference=[10]*549+[4164]
    output_box={};raw_box={};calls=[]
    class GPUModelRunner:
        use_async_scheduling=True
        def execute_model(self,output):
            self._determine_batch_execution_and_padding()
            logits=self.model.compute_logits(raw_box['head_input'])
            self.execute_model_state=(output,)
            return logits
        def _determine_batch_execution_and_padding(self):
            return NS(name='FULL'),NS(num_tokens=1,num_reqs=1,uniform=True),False,None,None
        def _sample(self,logits,*args):
            if fail_sampling:raise ValueError('sampler failed')
            return NS(sampled_token_ids=logits.argmax(dim=-1).reshape(1,1))
        def sample_tokens(self,grammar):
            self.execute_model_state=None
            logits=raw_box['raw']
            logits[0,4164]=-float('inf')
            return self._sample(logits,None)
    GPUModelRunner.__module__='vllm.v1.worker.gpu_model_runner'
    obj=GPUModelRunner()
    obj.model=NS(logits_processor=NS(scale=1.,soft_cap=None,logits_as_input=False,head_dtype=None),
        config=NS(hidden_size=8),lm_head=NS(tp_size=1,quant_method=object()),
        compute_logits=lambda *args:raw_box['raw'])
    obj.input_batch=NS(req_ids=['rid'],num_computed_tokens_cpu=[0])
    obj.requests={'rid':NS(prompt_token_ids=[1,2],block_ids=[[1]])}
    worker=traj.TrajectoryWorkerExtension();worker.model_runner=obj;worker.vllm_config=NS(speculative_config=None)
    scheduler=type('Scheduler',(),{})();scheduler.vllm_config=worker.vllm_config
    scheduler._m4_grammar_meta={};scheduler.m4_probe_cache_uninstall=lambda:calls.append('restored')
    monkeypatch.setitem(hooks._schedulers,id(worker.vllm_config),scheduler)
    worker.m4_trajectory_install()
    worker.m4_trajectory_arm(traj.SEQUENTIAL,[1,2],reference,549)
    sampled_token=None
    for i in range(boundary_index+2):
        query=[1,2] if i==0 else [sampled_token]
        output=NS(num_scheduled_tokens={'rid':len(query)})
        obj.input_batch.num_computed_tokens_cpu=[0 if i==0 else i+1]
        obj.input_ids=NS(gpu=torch.tensor(query))
        raw=torch.zeros((1,5000));raw[0,4164]=12;raw[0,3752]=11
        if i<boundary_index:raw[0,10]=13
        raw_box['raw']=raw
        padded_head=torch.full((len(query)+1,8),-999.,dtype=torch.bfloat16)
        padded_head[len(query)-1].fill_(float(i%64))
        raw_box['head_input']=padded_head[len(query)-1:len(query)]
        assert obj.execute_model(output) is raw
        words=np.full((1,157),-1,dtype=np.int32);words[0,4164//32]&=~(1<<(4164%32))
        grammar=NS(structured_output_request_ids=['rid'],grammar_bitmask=words)
        scheduler._m4_grammar_meta[id(output)]={'rid':dict(backend_class='xgrammar',grammar_class='XGrammarGrammar',
            num_processed_tokens=i,reasoner_class=None,reasoning_ended=True)}
        if fail_sampling:
            state=worker._m4_probe
            with pytest.raises(ValueError,match='sampler failed'):obj.sample_tokens(grammar)
            assert state['sampling_output'] is None
            worker.m4_probe_uninstall()
            assert state=={} and calls==['restored']
            assert not any(k in vars(obj) for k in ('execute_model','_sample','sample_tokens',
                '_determine_batch_execution_and_padding'))
            return
        sampled_token=int(obj.sample_tokens(grammar).sampled_token_ids)
        # Input storage and original logits may be reused by later forwards.
        obj.input_ids.gpu.fill_(-1);raw.fill_(99.);padded_head.fill_(99.)
    if invalid_capture:
        import base64
        if invalid_capture=='prefix':worker._m4_probe['queries'][0][0]=999
        elif invalid_capture=='nonfinite':worker._m4_probe['armed']['capture'].selected['raw'][3752]=float('nan')
        else:worker._m4_probe['armed']['capture'].selected['head_input'][0]=float('nan')
        match={'prefix':'prefix differs','nonfinite':'Nonfinite raw logits','head_nonfinite':'Nonfinite LM-head input'}[invalid_capture]
        with pytest.raises(RuntimeError,match=match):
            worker.m4_trajectory_collect()
        preserved=json.loads((tmp_path/'failed-sequential_decode_cold-vectors.json').read_text())
        assert preserved['classification']=='INCONCLUSIVE' and preserved['generation_index']==boundary_index
        raw=np.frombuffer(base64.b64decode(preserved['raw']['values_base64']),dtype=np.float32)
        masked=np.frombuffer(base64.b64decode(preserved['masked']['values_base64']),dtype=np.float32)
        assert len(raw)==5000 and raw[4164]==12. and masked[3752]==11. and np.isneginf(masked[4164])
        if invalid_capture=='nonfinite':assert np.isnan(raw[3752])
        if localize:
            values=np.frombuffer(base64.b64decode(preserved['head_input']['values_base64']),dtype=np.float32)
            assert len(values)==8
            if invalid_capture=='head_nonfinite':assert np.isnan(values[0])
        state=worker._m4_probe;worker.m4_probe_uninstall()
        assert state=={} and calls==['restored']
        return
    result=worker.m4_trajectory_collect()
    assert result['identity']['generation_index']==boundary_index
    assert result['full_logical_input_token_ids']==[1,2]+[10]*boundary_index
    assert result['observed_token']==3752 and result['historical_token']==reference[boundary_index]
    assert result['logits']['argmax_token_id']==4164
    assert result['logits']['grammar_masked_argmax_token_id']==3752
    assert result['raw_full_logits'][4164]==12. and result['grammar_masked_full_logits'][3752]==11.
    if localize:
        record=result['head_input']
        assert record['values']==[float(boundary_index%64)]*8
        assert record['source_query_row']==(1 if boundary_index==0 else 0)
        head.validate_head_input(record,result)
    else:assert 'head_input' not in result
    assert len(result['raw_full_logits'])==5000
    worker.m4_probe_uninstall()
    assert calls==['restored'] and not hasattr(worker,'_m4_probe')
    assert 'execute_model' not in vars(obj) and 'compute_logits' in vars(obj.model)
    assert '_determine_batch_execution_and_padding' not in vars(obj)


@pytest.mark.parametrize('failure',['execute_model','_sample','sample_tokens'])
def test_partial_trajectory_install_restores_all_owners(monkeypatch,failure):
    monkeypatch.setattr(traj,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(traj,'configuration',lambda:{'diagnostic_mode':'decode_trajectory'})
    class GPUModelRunner:
        use_async_scheduling=True
        def execute_model(self,*args):pass
        def _sample(self,*args):pass
        def sample_tokens(self,*args):pass
        def _determine_batch_execution_and_padding(self,*args):pass
    GPUModelRunner.__module__='vllm.v1.worker.gpu_model_runner'
    obj=GPUModelRunner();obj.model=NS(logits_processor=NS(scale=1.,soft_cap=None,logits_as_input=False),compute_logits=lambda:None)
    worker=traj.TrajectoryWorkerExtension();worker.model_runner=obj;worker.vllm_config=NS(speculative_config=None)
    scheduler=type('Scheduler',(),{})();scheduler.vllm_config=worker.vllm_config
    calls=[];scheduler.m4_probe_cache_uninstall=lambda:calls.append('restored')
    monkeypatch.setitem(hooks._schedulers,id(worker.vllm_config),scheduler)
    original=hooks._InstanceHooks.replace
    def replace(self,owner,name,value):
        original(self,owner,name,value)
        if name==failure:raise RuntimeError('partial install')
    monkeypatch.setattr(hooks._InstanceHooks,'replace',replace)
    with pytest.raises(RuntimeError,match='partial install'):worker.m4_trajectory_install()
    assert calls==['restored'] and not hasattr(worker,'_m4_probe')
    assert not any(k in vars(obj) for k in ('execute_model','_sample','sample_tokens','_determine_batch_execution_and_padding'))


def test_scheduler_natural_sequential_grammar_and_cleanup(monkeypatch,tmp_path):
    config=dict(diagnostic_mode='decode_trajectory',forced_input_token_ids=[1,2,10],
        generated_prefix_token_ids=[10],historical_request={'response_format':{'type':'json_object'}},
        cache_trace=str(tmp_path/'trace.jsonl'))
    monkeypatch.setattr(hooks,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(hooks,'configuration',lambda:config)
    accepted=[]
    grammar=NS(accept_tokens=lambda rid,ids:accepted.append(ids) or True,num_processed_tokens=0)
    request=NS(prompt_token_ids=[1,2],is_prefill_chunk=False,
        structured_output_request=NS(grammar=grammar,reasoning_ended=None))
    import numpy as np
    mask=NS(structured_output_request_ids=['rid'],grammar_bitmask=np.full((1,1),-1,dtype=np.int32))
    class Base:
        def __init__(self):
            self.vllm_config=NS(speculative_config=None);self.cache_config=NS(enable_prefix_caching=True)
            self.kv_cache_manager=NS(get_computed_blocks=lambda r:None)
            self.requests={'rid':request};self.structured_output_manager=NS(backend=object(),reasoner_cls=None)
        def _update_after_schedule(self,out):pass
        def get_grammar_bitmask(self,out):return mask
    class Scheduler(traj.TrajectorySchedulerMixin,Base):pass
    scheduler=Scheduler();output=NS(num_scheduled_tokens={'rid':2})
    scheduler._update_after_schedule(output)
    assert accepted==[] and request.structured_output_request.reasoning_ended is None
    assert scheduler.get_grammar_bitmask(output) is mask
    assert scheduler._m4_grammar_meta[id(output)]['rid']['num_processed_tokens']==0
    scheduler.m4_probe_cache_uninstall()
    assert scheduler._m4_grammar_meta=={} and id(scheduler.vllm_config) not in hooks._schedulers


def test_forced_prefill_comparison_moves_to_first_divergence(case,tmp_path,monkeypatch):
    local=deepcopy(case);local['cache_trace']=str(tmp_path/'cache.jsonl')
    a,b=capture(local,traj.SEQUENTIAL,3752,5),capture(local,traj.FORCED,4164,5)
    events=a['cache']['lookup_records']+b['cache']['lookup_records']
    # Each request has its own internal owner.
    b['state']['request_id']='forced-rid';events[1]['request_id']='forced-rid'
    (tmp_path/'cache.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    calls=[]
    class LLM:
        def reset_prefix_cache(self):calls.append('reset');return True
        def collective_rpc(self,method,args=()):
            calls.append((method,args))
            if method=='m4_trajectory_install':return [dict(installed=True)]
            if method=='m4_trajectory_collect':return [deepcopy(a if sum(c[0]=='m4_trajectory_collect' for c in calls if isinstance(c,tuple))==1 else b)]
            return [True]
    monkeypatch.setattr(traj,'sequential_request',lambda *args:dict(request_id='rid',generation_index=5,
        observed_tokens=local['generated_prefix_token_ids'][:5]+[3752],delivered_tokens_after_boundary=[]))
    monkeypatch.setattr(runner,'one_request',lambda *args:dict(request_id='forced-rid',
        first_output_token_id=4164,observed_output_tokens=[4164]))
    common={k:a[k] for k in ('environment','engine_config','effective_config','sampling_config',
        'source_commit','instrumentation_commit','model','run_id','timestamp_utc')}
    result=traj.run_conditions(LLM(),tmp_path,local,lambda:object(),common)
    assert result['classification']=='EARLIER_DIVERGENCE_OBSERVED'
    arms=[c[1] for c in calls if isinstance(c,tuple) and c[0]=='m4_trajectory_arm']
    assert arms[0][1]==local['original_prompt_token_ids']
    assert arms[1][1]==local['original_prompt_token_ids']+local['generated_prefix_token_ids'][:5]
    assert arms[1][3]==5
    assert (tmp_path/'sequential_decode_cold.json').exists() and (tmp_path/'forced_prefill_cold.json').exists()


def test_schema_rejects_truncated_vocabulary_and_changed_dtype(case):
    a,b=capture(case,traj.SEQUENTIAL),capture(case,traj.FORCED,3752)
    b['raw_full_logits'].pop();b['grammar_masked_full_logits'].pop()
    assert traj.analyze_trajectory(a,b,case)['classification']=='INCONCLUSIVE'
    b=capture(case,traj.FORCED,3752);b['logits']['dtype']='torch.bfloat16'
    assert traj.analyze_trajectory(a,b,case)['classification']=='INCONCLUSIVE'


def test_raw_argmax_flip_without_masked_decision_flip_is_inconclusive(case):
    a,b=capture(case,traj.SEQUENTIAL),capture(case,traj.FORCED,3752)
    for result in (a,b):
        result['grammar_masked_full_logits'][3752]=None
        result['logits']['grammar_masked_argmax_token_id']=4164
        result['observed_token']=4164
        if result['condition']==traj.SEQUENTIAL:
            result['observed']['observed_tokens'][-1]=4164
        else:
            result['observed'].update(first_output_token_id=4164,observed_output_tokens=[4164])
        words=result['state']['grammar_bitmask']['values'][0]
        words[3752//32]&=~(1<<(3752%32))
        result['state']['grammar_bitmask']['sha256']=hashlib.sha256(
            struct.pack('<'+'i'*len(words),*words)).hexdigest()
    assert traj.analyze_trajectory(a,b,case)['classification']=='INCONCLUSIVE'


def test_failure_retains_returned_worker_vectors_and_inconclusive_analysis(case,tmp_path,monkeypatch):
    a=capture(case,traj.SEQUENTIAL)
    a['state']['request_id']='wrong-owner'
    class LLM:
        def reset_prefix_cache(self):return True
        def collective_rpc(self,method,args=()):
            if method=='m4_trajectory_install':return [dict(installed=True)]
            if method=='m4_trajectory_collect':return [a]
            return [True]
    monkeypatch.setattr(traj,'sequential_request',lambda *args:dict(request_id='rid',generation_index=549,
        observed_tokens=case['generated_prefix_token_ids']+[4164]))
    with pytest.raises(RuntimeError,match='ownership'):
        traj.run_conditions(LLM(),tmp_path,case,lambda:object(),{})
    stored=json.loads((tmp_path/'worker-sequential_decode_cold.json').read_text())
    assert len(stored['raw_full_logits'])==traj.VOCABULARY_SIZE
    assert json.loads((tmp_path/'analysis.json').read_text())['classification']=='INCONCLUSIVE'


@pytest.mark.parametrize('damage,reason',[
    ('source','Instrumentation/source SHA'),('effective','Historical engine/sampling'),
    ('observed_missing','observed'),('observed_owner','request ownership'),
    ('observed_prefix','first decision'),('masked_stage','measurement stage'),
    ('run','Run identity'),('geometry','uniform_decode'),('python','Pinned environment')])
def test_review_provenance_findings_fail_closed(case,damage,reason):
    a,b=capture(case,traj.SEQUENTIAL),capture(case,traj.FORCED,3752)
    if damage=='source':
        for r in (a,b):r.update(source_commit='b'*40,instrumentation_commit='b'*40)
    elif damage=='effective':
        for r in (a,b):r['effective_config']['async_scheduling']=False
    elif damage=='observed_missing':del a['observed']
    elif damage=='observed_owner':a['observed']['request_id']='different'
    elif damage=='observed_prefix':a['observed']['observed_tokens'][0]=-1
    elif damage=='masked_stage':a['logits']['grammar_masked_measurement_point']='after penalties'
    elif damage=='run':a['run_id']='other-run'
    elif damage=='geometry':a['state']['execution'].pop('uniform_decode')
    else:a['environment']['python']=''
    result=traj.analyze_trajectory(a,b,case)
    assert result['classification']=='INCONCLUSIVE' and reason in result['reasons'][0],result


def test_review_prepared_input_is_rechecked_against_frozen_reference(case):
    altered=deepcopy(case);altered['generated_prefix_token_ids'][1]=777
    a,b=capture(altered,traj.SEQUENTIAL),capture(altered,traj.FORCED,3752)
    result=traj.analyze_trajectory(a,b,altered)
    assert result['classification']=='INCONCLUSIVE'
    assert 'frozen historical trajectory' in result['reasons'][0]


def test_review_manifest_source_is_bound_to_prepared_source(case):
    a,b=capture(case,traj.SEQUENTIAL),capture(case,traj.FORCED,3752)
    result=traj.analyze_trajectory(a,b,case,expected_source_commit='b'*40)
    assert result['classification']=='INCONCLUSIVE'
    assert 'manifest SHA' in result['reasons'][0]
