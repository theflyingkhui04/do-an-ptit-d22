"""CPU contracts and graph-surface simulation. Real CUDA validation is pending."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from experiments import m4_layer_probe as layer
from experiments import m4_layer_probe_runtime as runtime
from experiments import m4_decode_trajectory as traj
from experiments import run_m4_cache_probe as runner
from test_m4_decode_trajectory import case, capture


def contract(layers=3, selected=None):
    return dict(hidden_size=2,num_hidden_layers=layers,selected_layers=list(range(layers)) if selected is None else selected,
        tensor_parallel_size=1,pipeline_parallel_size=1,
        installation='after_kv_allocation_before_runtime_capture',graph_modes=['FULL','PIECEWISE'])


def attach(result, changed=None, selected=None):
    c = contract(selected=selected)
    plan = layer.stage_plan(c['num_hidden_layers'],c['selected_layers'])
    table = torch.ones((len(plan),2),dtype=torch.bfloat16)
    if changed is not None:
        table[next(i for i,p in enumerate(plan) if p[0]==changed),0] = 2
    identity = layer.row_identity(result)
    stamp = torch.tensor([identity['source_query_row'],identity['logical_token_position'],
        result['full_logical_input_token_ids'][-1],result['state']['execution']['padded_tokens']],dtype=torch.int64)
    result['layer_boundaries'] = layer.serialize_layers(table,stamp,c,result)
    return result


def pair(case, changed='layer_1', selected=None):
    prepared = dict(deepcopy(case),localization_stage=layer.LAYER_STAGE)
    a = capture(prepared,traj.SEQUENTIAL,4164)
    b = capture(prepared,traj.FORCED,3752)
    for r in (a,b):
        r['engine_config'] = runner.engine_config('decode_trajectory',layer.LAYER_STAGE)
    b['state'].update(computed_tokens_before=2048,scheduled_query_tokens=23,
        execution=dict(cudagraph_mode='PIECEWISE',padded_tokens=24,padded_requests=None,
                       uniform_decode=False,microbatching=False))
    return prepared,attach(a,selected=selected),attach(b,changed,selected)


def test_opt_in_and_inactive_by_default(case,tmp_path):
    assert 'worker_cls' not in runner.engine_config('decode_trajectory')
    assert 'worker_cls' not in runner.engine_config()
    assert runner.engine_config('decode_trajectory',layer.LAYER_STAGE)['worker_cls'].endswith('LayerBoundaryWorker')
    prepared = runner.prepare(tmp_path/'prepared','a'*40,'W1','m0_04_001',549,
                              'decode_trajectory',layer.LAYER_STAGE)
    traj.validate_prepared_case(prepared,'a'*40)
    with pytest.raises(ValueError,match='explicit trajectory'):
        runner.prepare(tmp_path/'wrong','a'*40,'W1','m0_04_001',549,'cache_probe',layer.LAYER_STAGE)


@pytest.mark.parametrize('changed,expected',[('embedding','EMBEDDING_STATE_DIVERGENCE'),
    ('layer_0','FIRST_DIVERGENT_LAYER_IDENTIFIED'),('layer_1','FIRST_DIVERGENT_LAYER_IDENTIFIED'),
    ('layer_2','FIRST_DIVERGENT_LAYER_IDENTIFIED'),('final_norm','FINAL_NORM_STATE_DIVERGENCE'),
    (None,'NO_LAYER_BOUNDARY_DIVERGENCE_REPRODUCED')])
def test_exact_first_observed_state(case,changed,expected):
    prepared,a,b = pair(case,changed)
    out = layer.analyze_layers(a,b,prepared,'a'*40)
    assert out['classification']==expected,out
    assert out['first_different_stage']==changed
    assert out['EARLIEST_OBSERVED_DIVERGENCE_STAGE']==changed
    assert out['root_cause']=='ROOT_CAUSE_NOT_PROVEN' and out['full_sweep']=='FULL_SWEEP_BLOCKED'
    first = next((i for i,r in enumerate(out['stage_comparisons']) if not r['bytes_equal']),None)
    assert first is None or all(r['bytes_equal'] for r in out['stage_comparisons'][:first])
    if first is not None:
        assert out['stage_comparisons'][first]['first_different_index']==0


def test_sparse_precision_does_not_claim_exact_layer(case):
    prepared,a,b = pair(case,'layer_2',[0,2])
    out = layer.analyze_layers(a,b,prepared)
    assert out['classification']=='DIVERGENCE_INTERVAL_IDENTIFIED',out
    assert out['DIVERGENCE_INTERVAL']==['layer_0','layer_2']
    # An adjacent last checkpoint does not fill an earlier measurement gap.
    prepared,a,b = pair(case,'layer_2',[1,2])
    assert layer.analyze_layers(a,b,prepared)['classification']=='DIVERGENCE_INTERVAL_IDENTIFIED'


def test_residual_component_and_signed_zero_count_as_boundary_difference(case):
    prepared,a,b = pair(case,None)
    records = b['layer_boundaries']['records']
    result = b
    c = contract()
    table = torch.ones((len(records),2),dtype=torch.bfloat16)
    residual_index = next(i for i,r in enumerate(records) if r['stage']=='layer_1' and r['component']=='residual')
    table[residual_index,0] = 2
    b['layer_boundaries'] = layer.serialize_layers(table,torch.tensor(b['layer_boundaries']['graph_stamp']),c,result)
    out = layer.analyze_layers(a,b,prepared)
    assert out['first_different_stage']=='layer_1'
    assert not out['stage_comparisons'][residual_index]['bytes_equal']
    # Exact byte equality, even when numerical summaries report zero.
    for target,value in ((a,0.),(b,-0.)):
        table.fill_(1); table[0,0]=value
        target['layer_boundaries']=layer.serialize_layers(table,
            torch.tensor(target['layer_boundaries']['graph_stamp']),c,target)
    out=layer.analyze_layers(a,b,prepared)
    assert out['classification']=='EMBEDDING_STATE_DIVERGENCE'
    assert out['stage_comparisons'][0]['max_abs_diff']==0


@pytest.mark.parametrize('damage',['missing','duplicate','order','layer','component','owner','prefix','row',
    'position','index','shape','dtype','stride','layout','bytes','sha','values','nonfinite','contract',
    'stamp','graph','model','source','early','unreproduced'])
def test_fail_closed(case,damage):
    prepared,a,b = pair(case)
    cap=b['layer_boundaries'];r=cap['records'][1]
    if damage=='missing':cap['records'].pop()
    elif damage=='duplicate':cap['records'].append(deepcopy(r))
    elif damage=='order':cap['records'][1:3]=list(reversed(cap['records'][1:3]))
    elif damage=='layer':r['layer_index']=2
    elif damage=='component':r['component']='merged'
    elif damage=='owner':r['request_id']='other'
    elif damage=='prefix':r['logical_input_sha256']='0'*64
    elif damage=='row':r['source_query_row']=23
    elif damage=='position':r['logical_token_position']-=1
    elif damage=='index':r['generation_index']=548
    elif damage=='shape':r['shape']=[2,1]
    elif damage=='dtype':r['dtype']='torch.float16'
    elif damage=='stride':r['stride']=[1,2]
    elif damage=='layout':r['layout']='torch.sparse_coo'
    elif damage=='bytes':r['bytes_base64']='!'
    elif damage=='sha':r['bytes_sha256']='0'*64
    elif damage=='values':r['values'][0]=99
    elif damage=='nonfinite':r['values'][0]=float('nan')
    elif damage=='contract':cap['contract']['num_hidden_layers']=4
    elif damage=='stamp':cap['graph_stamp'][0]=23
    elif damage=='graph':b['state']['execution']['cudagraph_mode']='FULL'
    elif damage=='model':b['model']['revision']='b'*40
    elif damage=='source':b['source_commit']='b'*40
    elif damage=='early':a['identity']['generation_index']=548
    else:
        b=capture(prepared,traj.FORCED,4164)
        b['engine_config']=runner.engine_config('decode_trajectory',layer.LAYER_STAGE)
        attach(b)
    assert layer.analyze_layers(a,b,prepared)['classification']=='INCONCLUSIVE'


@pytest.mark.parametrize('indices',[[1,0],[0,0],[-1],[3],[True],[]])
def test_layer_index_order_rejected(indices):
    with pytest.raises(ValueError):layer.stage_plan(3,indices)


def test_serialization_independent_bf16_bytes_and_alias_protection(case):
    prepared,a,b=pair(case)
    encoded=json.loads(json.dumps(a['layer_boundaries'],allow_nan=False))
    decoded=layer.validate_layers(encoded,a)
    assert all(values==[1.,1.] and payload==b'\x80\x3f\x80\x3f' for values,payload in decoded)
    c=contract();table=torch.ones((len(layer.stage_plan(3)),2),dtype=torch.bfloat16)
    stamp=torch.tensor(a['layer_boundaries']['graph_stamp'])
    retained=layer.serialize_layers(table,stamp,c,a)
    table.fill_(99);stamp.fill_(-1)
    assert layer.validate_layers(retained,a)==decoded
    with pytest.raises(RuntimeError,match='stamp'):
        layer.serialize_layers(table,stamp,c,a)


def test_same_latch_freezes_layers_and_ignores_later_async_work():
    latch=traj.FirstMismatchCapture([10,10,10],2)
    raw=torch.ones(4); table=torch.ones((8,2),dtype=torch.bfloat16);stamp=torch.tensor([0,2,10,1])
    for i,token in enumerate((10,11,12)):
        table.fill_(i);stamp[1]=i
        latch.record(i,raw,raw,torch.tensor([token]),layer_table=table,layer_stamp=stamp)
        table.fill_(99);stamp.fill_(-1)
    assert int(latch.selected['index'])==1
    assert latch.selected['layer_table'].unique().tolist()==[1.]
    assert latch.selected['layer_stamp'][1]==1


def fake_runtime(monkeypatch, fail_reset=False, live_graph=False):
    """An eager CPU runner tests lifecycle/ownership, NOT CUDA replay behavior."""
    compilation=NS(cache_dir='original',local_cache_dir='local',traced_files={'original.py'})
    vconfig=NS(compilation_config=compilation)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config=NS(hidden_size=2,num_hidden_layers=2)
            self.layers=[0,1];self.start_layer=0;self.end_layer=2
            self.vllm_config=vconfig;self.compiled=True;self.aot_compiled_fn=object()
        def _maybe_add_hidden_state(self,aux,index,hidden,residual):
            return aux
        def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
            hidden=input_ids[:,None].expand(-1,2).to(torch.bfloat16).contiguous()
            residual=None;aux=[]
            self._maybe_add_hidden_state(aux,0,hidden,residual)
            for i in range(2):
                residual=hidden.clone() if residual is None else residual.add_(hidden)
                hidden=hidden+1
                self._maybe_add_hidden_state(aux,i+1,hidden,residual)
            # Later fused operations mutate original sources, as in add/norm.
            residual.add_(hidden)
            return (hidden,aux) if aux else hidden
    model=Model()
    runner_obj=NS(device='cpu',use_aux_hidden_state_outputs=False,kv_caches=[torch.zeros(2)],
        input_batch=NS(req_ids=['rid']),requests={'rid':NS(prompt_token_ids=[1])})
    runner_obj._model_forward=lambda **kwargs:model(**kwargs)
    worker=NS(model_runner=runner_obj)
    wrapper=NS(vllm_config=vconfig,concrete_cudagraph_entries={'existing':1} if live_graph else {})
    wrapper.clear_graphs=lambda:wrapper.concrete_cudagraph_entries.clear()
    reset_calls=[]
    def reset(target):
        reset_calls.append(type(target))
        target.compiled=False;target.aot_compiled_fn=None
        if fail_reset and len(reset_calls)==1:raise RuntimeError('reset failure')
    return worker,model,wrapper,reset,reset_calls


def test_runtime_row_mapping_output_copy_cleanup_and_reinstallation(monkeypatch):
    worker,model,wrapper,reset,calls=fake_runtime(monkeypatch)
    original=type(model);original_forward=original.forward
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],original_forward)
    assert dict(model.named_buffers())['_m4_layer_row'] is observer.row
    assert '_m4_layer_row' not in model.state_dict()  # Observational, not a weight.
    worker._m4_probe=dict(armed=dict(prompt=[1]),output=NS(num_scheduled_tokens={'rid':23}))
    inputs=dict(input_ids=torch.arange(24),positions=torch.arange(2048,2072))
    output=worker.model_runner._model_forward(**inputs)
    table,stamp=observer.snapshot(output[0][22].clone())
    assert stamp.tolist()==[22,2070,22,24]
    assert table[0].tolist()==[22.,22.]  # Embedding not later mutated residual.
    preserved=table.clone()
    inputs['input_ids'].fill_(99)
    worker.model_runner._model_forward(**inputs)
    assert torch.equal(table,preserved)
    assert type(model) is not original and type(model).forward.__code__ is not original_forward.__code__
    wrapper.concrete_cudagraph_entries['runtime']=object()
    observer.close();observer.close()
    assert type(model) is original and model.forward.__func__ is original_forward
    assert not hasattr(model,'_m4_layer_row') and not worker.model_runner.use_aux_hidden_state_outputs
    assert wrapper.concrete_cudagraph_entries=={} and observer.row is None
    assert model.vllm_config.compilation_config.cache_dir=='original'
    assert len(calls)==2


def test_exported_cpu_graph_keeps_dynamic_buffer_rows_without_python_callbacks(monkeypatch):
    worker,model,wrapper,reset,calls=fake_runtime(monkeypatch)
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],type(model).forward)
    ids=torch.arange(24);positions=torch.arange(2048,2072)
    exported=torch.export.export(model,(ids,positions),strict=False).module()
    # Executing exported tensor work is a CPU graph-surface check, not a
    # simulation claimed to validate vLLM or CUDA graph capture/replay.
    exported._m4_layer_row.fill_(22)
    output=exported(ids,positions)
    assert output[1][0].tolist()==[[22.,22.]]
    assert output[1][-1].tolist()==[22,2070,22,24]
    exported._m4_layer_row.fill_(0)
    output=exported(ids+10,positions+100)
    assert output[1][0].tolist()==[[10.,10.]]
    assert output[1][-1].tolist()==[0,2148,10,24]
    observer.close()


def test_cleanup_exception_still_restores_model_and_allows_reported_failure(monkeypatch):
    worker,model,wrapper,reset,calls=fake_runtime(monkeypatch)
    original=type(model)
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],original.forward)
    wrapper.clear_graphs=lambda:(_ for _ in ()).throw(RuntimeError('graph cleanup error'))
    with pytest.raises(RuntimeError,match='cleanup failed'):observer.close()
    assert type(model) is original and not hasattr(model,'_m4_layer_row')
    assert not worker.model_runner.use_aux_hidden_state_outputs and observer.row is None
    wrapper.clear_graphs=lambda:wrapper.concrete_cudagraph_entries.clear()
    observer.close()
    assert observer.cleanup_complete


@pytest.mark.parametrize('bad_return',[None,torch.zeros((1,2)),(torch.zeros((1,2)),[]),
    (torch.zeros((1,2)),[torch.zeros((1,2))]*6)])
def test_unsupported_graph_return_type_or_dtype_fails_closed(monkeypatch,bad_return):
    worker,model,wrapper,reset,calls=fake_runtime(monkeypatch)
    worker.model_runner._model_forward=lambda **kwargs:bad_return
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],type(model).forward)
    with pytest.raises(RuntimeError,match='return'):
        worker.model_runner._model_forward(input_ids=torch.tensor([1]),positions=torch.tensor([2070]))
    assert observer.current is None
    observer.close()


def test_partial_install_after_compiler_reset_restores_every_owner(monkeypatch):
    worker,model,wrapper,reset,calls=fake_runtime(monkeypatch)
    original_class=type(model);original=runtime._InstanceHooks.replace
    def failing_replace(hooks,owner,name,value):
        if name=='_model_forward':raise RuntimeError('wrapper installation failed')
        return original(hooks,owner,name,value)
    monkeypatch.setattr(runtime._InstanceHooks,'replace',failing_replace)
    with pytest.raises(RuntimeError,match='installation failed'):
        runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],original_class.forward)
    assert len(calls)==2 and type(model) is original_class
    assert not hasattr(model,'_m4_layer_row') and '_maybe_add_hidden_state' not in vars(model)
    assert not worker.model_runner.use_aux_hidden_state_outputs


def test_trajectory_cleanup_exception_still_closes_layer_runtime(monkeypatch):
    from experiments.m4_cache_probe_hooks import CacheProbeWorkerExtension
    worker=traj.TrajectoryWorkerExtension()
    closed=[]
    worker._m4_layer_runtime=NS(close=lambda:closed.append(True))
    def failure(self):raise RuntimeError('trajectory restoration failed')
    monkeypatch.setattr(CacheProbeWorkerExtension,'m4_probe_uninstall',failure)
    with pytest.raises(RuntimeError,match='restoration failed'):worker.m4_probe_uninstall()
    assert closed==[True]


def test_extended_source_manifest_preserves_original_hashes():
    from pathlib import Path
    location=Path(runtime.__file__).parent
    old=json.loads((location/'m4_cache_probe_sources.json').read_text(encoding='utf8'))
    new=json.loads((location/'m4_layer_probe_sources.json').read_text(encoding='utf8'))
    assert new['vllm']==old['vllm']=='0.30.0'
    assert new['files'][:len(old['files'])]==old['files']
    paths=[r['path'] for r in new['files']]
    assert len(paths)==len(set(paths))==36
    assert {'vllm/compilation/decorators.py','vllm/compilation/wrapper.py',
        'vllm/model_executor/models/interfaces.py'} <= set(paths)


@pytest.mark.parametrize('damage',[None,'optin','compile_mode','graph_mode','eager','async','v2',
    'rank','lora','nested_wrapper','existing_graph','missing_kv'])
def test_pre_warmup_install_gates_and_runtime_context(monkeypatch,damage):
    import sys
    import types
    from contextlib import nullcontext
    worker,model,_,reset,calls=fake_runtime(monkeypatch)
    vconfig=model.vllm_config;worker.vllm_config=vconfig
    vconfig.speculative_config=None
    vconfig.parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,use_ubatching=False)
    vconfig.model_config=NS(enforce_eager=False)
    vconfig.compilation_config.mode=NS(name='VLLM_COMPILE')
    vconfig.compilation_config.cudagraph_mode=NS(name='FULL_AND_PIECEWISE')
    worker.model_runner.use_async_scheduling=True;worker.model_runner.lora_config=None
    cleaned=[];model.cleanup=lambda:cleaned.append(True)
    original=type(model)
    class Causal:
        pass
    causal=Causal();causal.model=model
    class FullGraph:
        def __init__(self):
            self.runtime_mode=NS(name='FULL');self.runnable=causal;self.vllm_config=vconfig
            self.concrete_cudagraph_entries={}
        def clear_graphs(self):self.concrete_cudagraph_entries.clear()
    wrapper=FullGraph();FullGraph._all_instances=[wrapper];worker.model_runner.model=wrapper
    envs=NS(VLLM_USE_V2_MODEL_RUNNER=False)
    values={'vllm':dict(envs=envs),'vllm.config':dict(set_current_vllm_config=lambda _:nullcontext()),
        'vllm.compilation.cuda_graph':dict(CUDAGraphWrapper=FullGraph),
        'vllm.compilation.wrapper':dict(reset_compile_wrapper=reset),
        'vllm.model_executor.models.qwen2':dict(Qwen2Model=original,Qwen2ForCausalLM=Causal)}
    for name,attrs in values.items():
        module=types.ModuleType(name);vars(module).update(attrs);monkeypatch.setitem(sys.modules,name,module)
    config=dict(diagnostic_mode='decode_trajectory',localization_stage=layer.LAYER_STAGE)
    monkeypatch.setattr(runtime,'configuration',lambda:config)
    monkeypatch.setattr(runtime,'verify_layer_sources',lambda:{})
    if damage=='optin':config['localization_stage']=None
    elif damage=='compile_mode':vconfig.compilation_config.mode.name='DYNAMO_ONCE'
    elif damage=='graph_mode':vconfig.compilation_config.cudagraph_mode.name='NONE'
    elif damage=='eager':vconfig.model_config.enforce_eager=True
    elif damage=='async':worker.model_runner.use_async_scheduling=False
    elif damage=='v2':envs.VLLM_USE_V2_MODEL_RUNNER=True
    elif damage=='rank':vconfig.parallel_config.tensor_parallel_size=2
    elif damage=='lora':worker.model_runner.lora_config=object()
    elif damage=='nested_wrapper':wrapper.runnable=NS(model=model)
    elif damage=='existing_graph':wrapper.concrete_cudagraph_entries['existing']=object()
    elif damage=='missing_kv':worker.model_runner.kv_caches=[]
    if damage:
        with pytest.raises(RuntimeError):runtime.install_before_warmup(worker)
        assert not cleaned and not calls and type(model) is original
        assert not hasattr(model,'_m4_layer_row')
    else:
        observer=runtime.install_before_warmup(worker)
        observer.close()
        assert len(cleaned)==len(calls)==2 and type(model) is original


@pytest.mark.parametrize('live_graph,reset_failure',[(True,False),(False,True)])
def test_partial_runtime_installation_cleanup(monkeypatch,live_graph,reset_failure):
    worker,model,wrapper,reset,calls=fake_runtime(monkeypatch,reset_failure,live_graph)
    original=type(model)
    with pytest.raises(RuntimeError):
        runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],original.forward)
    assert type(model) is original and not hasattr(model,'_m4_layer_row')
    assert worker.model_runner.use_aux_hidden_state_outputs is False
    if live_graph:
        assert wrapper.concrete_cudagraph_entries=={'existing':1} and not calls


def test_runtime_missing_graph_surface_wrong_owner_kv_and_cleanup_fail_closed(monkeypatch):
    worker,model,wrapper,reset,calls=fake_runtime(monkeypatch)
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],type(model).forward)
    with pytest.raises(RuntimeError,match='Missing current'):observer.snapshot(torch.ones(2,dtype=torch.bfloat16))
    worker._m4_probe=dict(armed=dict(prompt=[999]),output=NS(num_scheduled_tokens={'rid':1}))
    with pytest.raises(RuntimeError,match='wrong request'):
        worker.model_runner._model_forward(input_ids=torch.tensor([1]),positions=torch.tensor([2070]))
    worker.model_runner.kv_caches=[torch.zeros(3)]
    with pytest.raises(RuntimeError,match='KV allocation'):observer.check_kv_unchanged()
    with pytest.raises(RuntimeError,match='cleanup failed'):observer.close()
    assert not hasattr(model,'_m4_layer_row') and observer.row is None


def test_no_runtime_imports_or_graph_mode_shortcuts():
    import ast
    from pathlib import Path
    for module in (layer,runtime):
        tree=ast.parse(Path(module.__file__).read_text(encoding='utf8'))
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        assert not any((getattr(n,'module','') or '').startswith(('torch','vllm')) or
            any(a.name.startswith(('torch','vllm')) for a in getattr(n,'names',[])) for n in imports)


def test_worker_pairs_both_arms_with_same_latch_and_graph_stamp(monkeypatch,tmp_path):
    import numpy as np
    from experiments import m4_cache_probe_hooks as hooks
    config=dict(diagnostic_mode='decode_trajectory',localization_stage=layer.LAYER_STAGE,
        original_prompt_token_ids=[1,2],generated_prefix_token_ids=[10]*549,
        reference_next_token_id=4164,generation_index=549,source_commit='a'*40,
        cache_trace=str(tmp_path/'cache.jsonl'))
    monkeypatch.setattr(traj,'configuration',lambda:config)
    monkeypatch.setattr(traj,'verify_runtime_sources',lambda:{})
    _,model,wrapper,reset,_=fake_runtime(monkeypatch)
    box={}
    class GPUModelRunner:
        use_async_scheduling=True
        def execute_model(self,output):
            self._determine_batch_execution_and_padding()
            result=self._model_forward(input_ids=self.input_ids.gpu,positions=box['positions'])
            logits=self.model.compute_logits(result[0][box['query']-1:box['query']])
            self.execute_model_state=(output,)
            return logits
        def _determine_batch_execution_and_padding(self):
            return (NS(name=box['mode']),NS(num_tokens=len(self.input_ids.gpu),
                num_reqs=1 if box['mode']=='FULL' else None,uniform=box['mode']=='FULL'),False,None,None)
        def _sample(self,logits,*args):
            return NS(sampled_token_ids=logits.argmax(dim=-1).reshape(1,1))
        def sample_tokens(self,grammar):
            self.execute_model_state=None
            return self._sample(box['raw'])
    GPUModelRunner.__module__='vllm.v1.worker.gpu_model_runner'
    obj=GPUModelRunner()
    obj.device='cpu';obj.kv_caches=[torch.zeros(2)];obj.use_aux_hidden_state_outputs=False
    obj._model_forward=lambda **kwargs:model(**kwargs)
    obj.model=NS(config=model.config,lm_head=NS(tp_size=1,quant_method=object()),
        logits_processor=NS(scale=1.,soft_cap=None,logits_as_input=False,head_dtype=None),
        compute_logits=lambda *args:box['raw'])
    obj.input_batch=NS(req_ids=['rid'],num_computed_tokens_cpu=[0])
    obj.requests={'rid':NS(prompt_token_ids=[1,2],block_ids=[[1]])}
    worker=traj.TrajectoryWorkerExtension();worker.model_runner=obj
    worker.vllm_config=model.vllm_config;worker.vllm_config.speculative_config=None
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],type(model).forward)
    worker._m4_layer_runtime=observer
    scheduler=type('Scheduler',(),{})()
    scheduler.vllm_config=worker.vllm_config;scheduler._m4_grammar_meta={}
    scheduler.m4_probe_cache_uninstall=lambda:None
    monkeypatch.setitem(hooks._schedulers,id(worker.vllm_config),scheduler)
    worker.m4_trajectory_install()
    reference=[10]*549+[4164]
    def step(computed,query_ids,token,mode,padding=0,sample=True,index=0):
        box.update(query=len(query_ids),mode=mode)
        obj.input_batch.num_computed_tokens_cpu=[computed]
        obj.input_ids=NS(gpu=torch.tensor(query_ids+[0]*padding))
        box['positions']=torch.arange(computed,computed+len(query_ids)+padding)
        box['raw']=torch.zeros((1,5000));box['raw'][0,token]=10
        output=NS(num_scheduled_tokens={'rid':len(query_ids)})
        obj.execute_model(output)
        if sample:
            scheduler._m4_grammar_meta[id(output)]={'rid':dict(backend_class='test',grammar_class='Test',
                num_processed_tokens=index,reasoner_class=None,reasoning_ended=True)}
            obj.sample_tokens(NS(structured_output_request_ids=['rid'],
                grammar_bitmask=np.full((1,157),-1,dtype=np.int32)))
        obj.input_ids.gpu.fill_(-1)
    worker.m4_trajectory_arm(traj.SEQUENTIAL,[1,2],reference,549)
    for i in range(550):
        step(0 if i==0 else i+1,[1,2] if i==0 else [10],10 if i<549 else 4164,'FULL',index=i)
    frozen=worker._m4_probe['armed']['capture'].selected['layer_table'].clone()
    step(551,[4164],3752,'FULL',index=550)  # Later async execution is not evidence.
    assert torch.equal(frozen,worker._m4_probe['armed']['capture'].selected['layer_table'])
    sequential=worker.m4_trajectory_collect()
    assert sequential['identity']['generation_index']==549
    assert sequential['layer_boundaries']['graph_stamp']==[0,550,10,1]
    assert sequential['layer_boundaries']['records'][0]['values']==[10.,10.]
    forced_prompt=[1,2]+reference[:549]
    obj.requests['rid'].prompt_token_ids=forced_prompt
    worker.m4_trajectory_arm(traj.FORCED,forced_prompt,reference,549)
    step(0,forced_prompt[:528],10,'PIECEWISE',sample=False)
    step(528,forced_prompt[528:],3752,'PIECEWISE',padding=1,index=549)
    forced=worker.m4_trajectory_collect()
    assert forced['identity']==sequential['identity']
    assert forced['layer_boundaries']['graph_stamp']==[22,550,10,24]
    assert forced['layer_boundaries']['records'][0]['values']==[10.,10.]
    assert worker.m4_probe_uninstall()==dict(restored=True,captures_released=True,cache_observer_restored=True)
    assert not hasattr(worker,'_m4_probe') and observer.row is None
