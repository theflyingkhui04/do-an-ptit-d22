"""CPU-only LM-head boundary contracts; no GPU numerical claim."""
from copy import deepcopy
import json

import pytest
import torch

from experiments import m4_head_input_probe as head
from experiments import m4_decode_trajectory as traj
from experiments import run_m4_cache_probe as runner
from test_m4_decode_trajectory import case, capture


def add_head(result, values):
    tensor=torch.tensor([values],dtype=torch.bfloat16)
    vector,layout=head.clone_head_input((tensor,),{},len(values))
    contract=dict(hidden_size=len(values),tensor_parallel_size=1,quant_method_class='test.HeadMethod',head_dtype='None')
    result['head_input']=head.serialize_head_input(vector,layout,contract,result)
    return result


def pair(case,a_values=(1.,2.),b_values=(1.,2.),raw_different=True):
    prepared=dict(deepcopy(case),localization_stage=head.HEAD_STAGE)
    a=add_head(capture(prepared,traj.SEQUENTIAL,4164),a_values)
    b=capture(prepared,traj.FORCED,3752 if raw_different else 4164)
    b['state'].update(computed_tokens_before=2048,scheduled_query_tokens=23,
        execution=dict(cudagraph_mode='PIECEWISE',padded_tokens=24,padded_requests=None,
                       uniform_decode=False,microbatching=False))
    b=add_head(b,b_values)
    return prepared,a,b


def test_opt_in_prepare_and_default_inactive(case,tmp_path):
    normal=runner.prepare(tmp_path/'default','a'*40,'W1','m0_04_001',549,'decode_trajectory')
    assert 'localization_stage' not in normal
    enabled=runner.prepare(tmp_path/'head','a'*40,'W1','m0_04_001',549,'decode_trajectory',head.HEAD_STAGE)
    assert enabled['localization_stage']==head.HEAD_STAGE
    assert normal['historical_request']==enabled['historical_request']==case['historical_request']
    with pytest.raises(ValueError,match='explicit trajectory'):
        runner.prepare(tmp_path/'wrong','a'*40,'W1','m0_04_001',549,'cache_probe',head.HEAD_STAGE)


@pytest.mark.parametrize('a,b,raw_different,expected',[
    ((1.,2.),(1.,3.),True,'UPSTREAM_HIDDEN_STATE_DIVERGENCE'),
    ((1.,2.),(1.,2.),True,'LM_HEAD_OUTPUT_DIVERGENCE_WITH_IDENTICAL_INPUT'),
    ((1.,2.),(1.,2.),False,'NO_BOUNDARY_DIVERGENCE_REPRODUCED'),
    ((0.,2.),(-0.,2.),True,'HEAD_INPUT_BYTES_DIFFER_WITH_EQUAL_VALUES')])
def test_binary_split(case,a,b,raw_different,expected):
    prepared,sequential,forced=pair(case,a,b,raw_different)
    result=head.analyze_head_input(sequential,forced,prepared)
    assert result['classification']==expected,result
    assert result['root_cause']=='ROOT_CAUSE_NOT_PROVEN'
    assert result['full_sweep']=='FULL_SWEEP_BLOCKED'
    difference=result['head_input_difference']
    assert difference['max_absolute_difference']==max(abs(x-y) for x,y in zip(a,b))
    assert difference['mean_absolute_difference']==sum(abs(x-y) for x,y in zip(a,b))/len(a)
    numerical=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    assert difference['numerically_different_elements']==len(numerical)
    assert difference['first_numerically_different_index']==(numerical[0] if numerical else None)
    if expected=='HEAD_INPUT_BYTES_DIFFER_WITH_EQUAL_VALUES':
        assert difference['byte_different_elements']==1 and difference['first_byte_different_index']==0


@pytest.mark.parametrize('input_differs',[False,True])
def test_unreproduced_historical_flip_requires_provenance_runtime_review(case,input_differs):
    prepared,a,b=pair(case,b_values=(1.,3.) if input_differs else (1.,2.),raw_different=False)
    result=head.analyze_head_input(a,b,prepared)
    assert result['historical_boundary_flip_reproduced'] is False
    assert result['provenance_runtime_review_required'] is True
    assert 'does not falsify' in result['interpretation']
    assert result['classification']==('UPSTREAM_HIDDEN_STATE_DIVERGENCE' if input_differs
        else 'NO_BOUNDARY_DIVERGENCE_REPRODUCED')


@pytest.mark.parametrize('damage', ['owner','stage','row','position','prefix','shape','dtype','layout',
    'stride','bytes','values','nonfinite','contract','missing','index','runtime','source'])
def test_localization_fails_closed(case,damage):
    prepared,a,b=pair(case)
    r=b['head_input']
    if damage=='owner':r['request_id']='other'
    elif damage=='stage':r['measurement_point']='layer output'
    elif damage=='row':r['source_query_row']+=1
    elif damage=='position':r['logical_token_position']+=1
    elif damage=='prefix':r['logical_input_sha256']='0'*64
    elif damage=='shape':r['shape']=[2,1]
    elif damage=='dtype':r['dtype']='torch.float32'
    elif damage=='layout':r['layout']='torch.sparse_coo'
    elif damage=='stride':r['stride']=[1,2]
    elif damage=='bytes':r['bytes_base64']='!invalid'
    elif damage=='values':r['values'][0]=99.
    elif damage=='nonfinite':r['values'][0]=float('nan')
    elif damage=='contract':r['head_contract']['quant_method_class']='other.Method'
    elif damage=='missing':del b['head_input']
    elif damage=='index':r['generation_index']=548
    elif damage=='runtime':b['effective_config']['enable_chunked_prefill']=False
    else:b['source_commit']='b'*40
    assert head.analyze_head_input(a,b,prepared)['classification']=='INCONCLUSIVE'


def test_earlier_boundary_is_not_a_generation_549_localization(case):
    prepared=dict(case,localization_stage=head.HEAD_STAGE)
    a=add_head(capture(prepared,traj.SEQUENTIAL,3752,5),[1.,2.])
    b=add_head(capture(prepared,traj.FORCED,4164,5),[1.,3.])
    result=head.analyze_head_input(a,b,prepared)
    assert result['classification']=='INCONCLUSIVE'
    assert result['trajectory_classification']=='EARLIER_DIVERGENCE_OBSERVED'


def test_query_row_is_gathered_before_hook_and_not_padding(case):
    prepared,a,b=pair(case)
    # The forced query has 23 useful rows plus one padding row. The runner
    # supplies row 22 as [1, hidden], not the last padded row (23).
    padded=torch.full((24,2),99.,dtype=torch.bfloat16);padded[22]=torch.tensor([1.,2.])
    vector,layout=head.clone_head_input((padded[22:23],),{},2)
    vector_record=head.serialize_head_input(vector,layout,b['head_input']['head_contract'],b)
    padded.fill_(55.)
    assert vector_record['values']==[1.,2.] and vector_record['storage_offset']==44
    assert vector_record['source_query_row']==22 and vector_record['logits_row']==0
    head.validate_head_input(vector_record,b)
    with pytest.raises(RuntimeError,match='one gathered'):
        head.clone_head_input((padded,),{},2)


@pytest.mark.parametrize('tensor',[torch.ones((1,2)),torch.ones((2,2),dtype=torch.bfloat16),
    torch.ones((1,4),dtype=torch.bfloat16)[:,::2]])
def test_clone_rejects_dtype_shape_and_non_dense_row(tensor):
    with pytest.raises(RuntimeError,match='shape/dtype/layout'):head.clone_head_input((tensor,),{},2)


def test_clone_before_head_mutation_and_serialization_roundtrip(case):
    prepared,a,b=pair(case)
    tensor=torch.tensor([[1.,2.]],dtype=torch.bfloat16)
    vector,layout=head.clone_head_input((),{'hidden_states':tensor},2)
    tensor.fill_(100.)
    record=head.serialize_head_input(vector,layout,a['head_input']['head_contract'],a)
    restored=json.loads(json.dumps(record,allow_nan=False))
    values,payload=head.validate_head_input(restored,a)
    assert values==[1.,2.] and len(payload)==4
    vector[0]=float('nan')
    with pytest.raises(RuntimeError,match='Nonfinite LM-head'):head.serialize_head_input(vector,layout,record['head_contract'],a)


def test_missing_or_multiple_hook_arguments_rejected():
    tensor=torch.ones((1,2),dtype=torch.bfloat16)
    for args,kwargs in (((),{}),((tensor,tensor),{}),((tensor,),{'hidden_states':tensor})):
        with pytest.raises(RuntimeError,match='signature'):head.clone_head_input(args,kwargs,2)


@pytest.mark.parametrize('damage',[None,'hooks','shutdown','worker','status','process','memory'])
def test_offline_lifecycle_fails_closed(damage):
    cleanup=dict(hooks_restored=True,engine_shutdown=True,worker_results=[dict(restored=True,
        captures_released=True,cache_observer_restored=True)])
    lifecycle=dict(status='COMPLETE',gpu_before=dict(processes='',memory='6,40960'),
        gpu_after=dict(processes='',memory='6,40960'))
    if damage=='hooks':cleanup['hooks_restored']=False
    elif damage=='shutdown':cleanup['engine_shutdown']=False
    elif damage=='worker':cleanup['worker_results']=[]
    elif damage=='status':lifecycle['status']='FAIL'
    elif damage=='process':lifecycle['gpu_after']['processes']='123'
    elif damage=='memory':lifecycle['gpu_after']['memory']='200,40960'
    if damage:
        with pytest.raises(ValueError,match='cleanup/release'):head.require_completed_lifecycle(cleanup,lifecycle)
    else:head.require_completed_lifecycle(cleanup,lifecycle)


def test_forced_chunk_hook_retains_row_22_and_skips_partial_chunk(case,tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    from experiments import m4_cache_probe_hooks as hooks
    config=dict(case,localization_stage=head.HEAD_STAGE,cache_trace=str(tmp_path/'cache.jsonl'))
    monkeypatch.setattr(traj,'configuration',lambda:config)
    monkeypatch.setattr(traj,'verify_runtime_sources',lambda:{})
    clones=[];original_clone=traj.clone_head_input
    def measured_clone(args,kwargs,width):
        clones.append(list(args[0].shape))
        return original_clone(args,kwargs,width)
    monkeypatch.setattr(traj,'clone_head_input',measured_clone)
    class GPUModelRunner:
        use_async_scheduling=True
        def _determine_batch_execution_and_padding(self):
            return NS(name='PIECEWISE'),NS(num_tokens=len(self.hidden),num_reqs=None,uniform=False),False,None,None
        def execute_model(self,output):
            self._determine_batch_execution_and_padding()
            q=output.num_scheduled_tokens['rid']
            self.model.compute_logits(self.hidden[q-1:q])
            self.execute_model_state=(output,)
        def _sample(self,logits,*args):
            return NS(sampled_token_ids=logits.argmax(-1).reshape(1,1))
        def sample_tokens(self,grammar):
            self.execute_model_state=None
            return self._sample(self.live_logits,None)
    GPUModelRunner.__module__='vllm.v1.worker.gpu_model_runner'
    obj=GPUModelRunner()
    def compute(hidden_states):
        assert hidden_states.shape==(1,8)
        return obj.live_logits
    obj.model=NS(config=NS(hidden_size=8),lm_head=NS(tp_size=1,quant_method=object()),
        logits_processor=NS(scale=1.,soft_cap=None,logits_as_input=False,head_dtype=None),compute_logits=compute)
    tokens=case['forced_input_token_ids']
    obj.input_batch=NS(req_ids=['rid'],num_computed_tokens_cpu=[0])
    obj.requests={'rid':NS(prompt_token_ids=tokens,block_ids=[[1]])}
    worker=traj.TrajectoryWorkerExtension();worker.model_runner=obj;worker.vllm_config=NS(speculative_config=None)
    scheduler=type('Scheduler',(),{})();scheduler.vllm_config=worker.vllm_config
    scheduler._m4_grammar_meta={};restored=[]
    scheduler.m4_probe_cache_uninstall=lambda:restored.append(True)
    monkeypatch.setitem(hooks._schedulers,id(worker.vllm_config),scheduler)
    worker.m4_trajectory_install()
    try:
        worker.m4_trajectory_arm(traj.FORCED,tokens,case['generated_prefix_token_ids']+[4164],549)
        for computed,q,padded in ((0,2048,2048),(2048,23,24)):
            obj.input_batch.num_computed_tokens_cpu=[computed]
            obj.input_ids=NS(gpu=torch.tensor(tokens[computed:computed+q]))
            obj.hidden=torch.full((padded,8),99.,dtype=torch.bfloat16);obj.hidden[q-1].fill_(float(q-1))
            obj.live_logits=torch.zeros((1,5000));obj.live_logits[0,3752]=10.
            output=NS(num_scheduled_tokens={'rid':q})
            obj.execute_model(output)
            grammar=None
            if computed:
                grammar=NS(structured_output_request_ids=['rid'],grammar_bitmask=np.full((1,157),-1,dtype=np.int32))
                scheduler._m4_grammar_meta[id(output)]={'rid':dict(backend_class='xgrammar',grammar_class='XgrammarGrammar',
                    num_processed_tokens=549,reasoner_class=None,reasoning_ended=True)}
            obj.sample_tokens(grammar)
            obj.hidden.fill_(77.);obj.input_ids.gpu.fill_(-1)
        result=worker.m4_trajectory_collect();record=result['head_input']
        assert clones==[[1,8]]
        assert record['values']==[22.]*8 and record['source_query_row']==22 and record['logits_row']==0
        assert record['logical_token_position']==2070 and result['identity']['generation_index']==549
        assert result['full_logical_input_token_ids']==tokens
        head.validate_head_input(record,result)
    finally:
        worker.m4_probe_uninstall()
    assert restored==[True] and not hasattr(worker,'_m4_probe')


def test_offline_entrypoint_checks_cleanup_before_numerical_classification(tmp_path,monkeypatch):
    (tmp_path/'input.json').write_text(json.dumps(dict(diagnostic_mode='decode_trajectory',
        localization_stage=head.HEAD_STAGE)))
    (tmp_path/'hook-cleanup.json').write_text(json.dumps(dict(hooks_restored=False,engine_shutdown=False)))
    (tmp_path/'lifecycle.json').write_text(json.dumps(dict(status='FAIL')))
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(runner,'validated_source_commit',lambda source:source)
    monkeypatch.setattr(runner,'artifact_path',lambda *args,**kwargs:tmp_path)
    monkeypatch.setattr(runner,'verify_run_manifest',lambda *args:None)
    monkeypatch.setattr(head,'analyze_head_input',lambda *args,**kwargs:pytest.fail('Failed cleanup reached classifier'))
    monkeypatch.setattr('sys.argv',['probe','analyze','--source-commit','a'*40,'--out',str(tmp_path)])
    with pytest.raises(ValueError,match='cleanup/release'):runner.main()
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before
