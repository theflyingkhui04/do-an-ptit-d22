"""CPU contracts and exported tensor surface only. GPU validation pending."""
from copy import deepcopy
import base64
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
import torch

from experiments import m4_layer0_probe as internal
from experiments import m4_layer_probe_runtime as runtime
from experiments import m4_decode_trajectory as trajectory
from experiments import run_m4_cache_probe as runner
from test_m4_decode_trajectory import case, capture
from test_m4_layer_probe import fake_runtime


def contract():
    return dict(hidden_size=4,q_size=4,kv_size=2,intermediate_size=6,selected_layers=[0],
        tensor_parallel_size=1,pipeline_parallel_size=1,
        installation='after_kv_allocation_before_runtime_capture',graph_modes=['FULL','PIECEWISE'])


def attach(result,changed=None):
    sizes=internal.widths(contract())
    table=torch.ones((1,sum(sizes)),dtype=torch.bfloat16)
    if changed is not None:
        table[0,sum(sizes[:internal.STAGES.index(changed)])]=2
    owner=internal.row_identity(result)
    stamp=torch.tensor([owner['source_query_row'],2070,result['full_logical_input_token_ids'][-1],
                        result['state']['execution']['padded_tokens'],123])
    result['layer0_internal']=internal.serialize_internal(table,stamp,contract(),result)
    return result


def pair(case,changed='qkv_projection'):
    case=dict(deepcopy(case),localization_stage=internal.INTERNAL_STAGE)
    a=capture(case,trajectory.SEQUENTIAL,4164); b=capture(case,trajectory.FORCED,3752)
    for r in (a,b):
        r['engine_config']=runner.engine_config('decode_trajectory',internal.INTERNAL_STAGE)
    b['state'].update(computed_tokens_before=2048,scheduled_query_tokens=23,
        execution=dict(cudagraph_mode='PIECEWISE',padded_tokens=24,padded_requests=None,
                       uniform_decode=False,microbatching=False))
    return case,attach(a),attach(b,changed)


def test_explicit_opt_in(case,tmp_path):
    assert 'worker_cls' not in runner.engine_config('decode_trajectory')
    assert 'worker_cls' not in runner.engine_config()
    assert runner.engine_config('decode_trajectory',internal.INTERNAL_STAGE)['worker_cls'].endswith('LayerBoundaryWorker')
    prepared=runner.prepare(tmp_path/'yes','a'*40,'W1','m0_04_001',549,'decode_trajectory',internal.INTERNAL_STAGE)
    trajectory.validate_prepared_case(prepared,'a'*40)
    with pytest.raises(ValueError):
        runner.prepare(tmp_path/'no','a'*40,'W1','m0_04_001',549,'cache_probe',internal.INTERNAL_STAGE)


@pytest.mark.parametrize('stage,operation',[
    ('input_norm','input_norm'),('qkv_projection','qkv_projection'),('q_post_rope','rope'),
    ('k_post_rope','rope'),('attention_output','attention_output'),('o_proj','o_proj'),
    ('post_attention_norm','post_attention_norm'),('post_attention_residual','post_attention_norm'),
    ('gate_up','mlp_gate_up'),('activation','mlp_activation'),('down_proj','mlp_down_proj')])
def test_first_boundary(case,stage,operation):
    prepared,a,b=pair(case,stage)
    result=internal.analyze_internal(a,b,prepared,'a'*40)
    assert result['FIRST_INTERNAL_DIVERGENCE']==operation,result
    assert result['root_cause']=='ROOT_CAUSE_NOT_PROVEN'
    assert result['full_sweep']=='FULL_SWEEP_BLOCKED'
    diff=next(r for r in result['stage_comparisons'] if not r['bytes_equal'])
    assert diff['stage']==stage and diff['byte_different_elements']==1
    assert diff['max_abs_diff']==1 and diff['numerically_different_elements']==1


@pytest.mark.parametrize('stage',['input','v','return_hidden','return_residual'])
def test_interval_not_overclaimed(case,stage):
    prepared,a,b=pair(case,stage)
    result=internal.analyze_internal(a,b,prepared)
    assert result['INTERNAL_DIVERGENCE_INTERVAL'][-1]==stage
    assert 'FIRST_INTERNAL_DIVERGENCE' not in result


def test_no_difference_and_unreproduced_flip(case):
    prepared,a,b=pair(case,None)
    assert internal.analyze_internal(a,b,prepared)['classification']=='NO_INTERNAL_DIVERGENCE_REPRODUCED'
    b['observed_token']=4164
    assert internal.analyze_internal(a,b,prepared)['classification']=='INCONCLUSIVE'


@pytest.mark.parametrize('damage',['duplicate','missing','order','shape','dtype','owner','row','index',
    'position','hash','nan','inf','slot','stamp','layer','mode','boolean_shape'])
def test_invalid_capture_rejected(case,damage):
    prepared,a,b=pair(case)
    cap=b['layer0_internal']; r=cap['records'][2]
    if damage=='duplicate':cap['records'].append(deepcopy(r))
    elif damage=='missing':cap['records'].pop()
    elif damage=='order':cap['records'][0],cap['records'][1]=cap['records'][1],cap['records'][0]
    elif damage=='shape':r['shape']=[1,9]
    elif damage=='boolean_shape':r['shape']=[True,8]
    elif damage=='dtype':r['dtype']='torch.float32'
    elif damage=='owner':r['request_id']='other'
    elif damage=='row':r['source_query_row']=0
    elif damage=='index':r['generation_index']=548
    elif damage=='position':r['logical_token_position']=2069
    elif damage=='hash':r['bytes_sha256']='0'*64
    elif damage in ('nan','inf'):
        values=torch.ones(8,dtype=torch.bfloat16);values[0]=float(damage)
        payload=values.view(torch.uint8).numpy().tobytes()
        r['bytes_base64']=base64.b64encode(payload).decode();r['bytes_sha256']=hashlib.sha256(payload).hexdigest()
    elif damage=='slot':cap['selected_kv_slot']=-1
    elif damage=='stamp':cap['graph_stamp'][1]=2069
    elif damage=='layer':r['layer_index']=1
    elif damage=='mode':b['state']['execution']['cudagraph_mode']='FULL'
    assert internal.analyze_internal(a,b,prepared)['classification']=='INCONCLUSIVE'


class Projection(torch.nn.Module):
    def __init__(self,size):super().__init__();self.size=size
    def forward(self,x):return x.mean(-1,keepdim=True).expand(-1,self.size).contiguous(),None


class Norm(torch.nn.Module):
    def forward(self,x,residual=None):
        if residual is None:return x+1
        residual.add_(x)
        return residual+1,residual


class Rope(torch.nn.Module):
    def forward(self,positions,q,k):q.add_(10);k.add_(20);return q,k


class Attention(torch.nn.Module):
    layer_name='model.layers.0.self_attn.attn'
    def forward(self,q,k,v):return q+k.repeat(1,2)+v.repeat(1,2)


class Activation(torch.nn.Module):
    def forward(self,x):return x[:,:6]*x[:,6:]


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm=Norm();self.post_attention_layernorm=Norm()
        self.self_attn=torch.nn.Module()
        self.self_attn.q_size=4;self.self_attn.kv_size=2
        self.self_attn.qk_norm=False;self.self_attn.dual_chunk_attention_config=None
        self.self_attn.qkv_proj=Projection(8);self.self_attn.rotary_emb=Rope()
        self.self_attn.attn=Attention();self.self_attn.o_proj=Projection(4)
        self.mlp=torch.nn.Module();self.mlp.gate_up_proj=Projection(12)
        self.mlp.act_fn=Activation();self.mlp.down_proj=Projection(4)
    def forward(self,positions,hidden,residual):
        residual.add_(hidden)
        return hidden+1,residual


def toy_runtime(monkeypatch,fail_reset=False):
    worker,old,wrapper,reset,calls=fake_runtime(monkeypatch,fail_reset=fail_reset)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.vllm_config=old.vllm_config
            self.config=NS(hidden_size=4,num_hidden_layers=2,intermediate_size=6)
            self.layers=torch.nn.ModuleList([Layer(),Layer()]);self.norm=Norm()
            self.start_layer=0;self.end_layer=2;self.compiled=True;self.aot_compiled_fn=object()
        def embed_input_ids(self,ids):return ids[:,None].expand(-1,4).to(torch.bfloat16).contiguous()
        def forward(self,*args,**kwargs):raise AssertionError('baseline wrapper reused')
    model=Model();worker.model_runner._model_forward=lambda **kw:model(**kw)
    from experiments import m4_layer0_sources
    monkeypatch.setattr(m4_layer0_sources,'selected_implementations',lambda m:{'test':'CPU only'})
    import sys,types
    context=types.ModuleType('vllm.forward_context')
    slots=torch.arange(24)+100
    context.get_forward_context=lambda:NS(slot_mapping={Attention.layer_name:slots})
    monkeypatch.setitem(sys.modules,'vllm.forward_context',context)
    return worker,model,wrapper,reset,calls,slots


def test_packed_qkv_alias_row_ownership_export_and_cleanup(monkeypatch):
    worker,model,wrapper,reset,calls,slots=toy_runtime(monkeypatch)
    original=type(model)
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],original.forward,internal=True)
    worker._m4_probe=dict(armed=dict(prompt=[1]),output=NS(num_scheduled_tokens={'rid':23}))
    ids=torch.arange(24);positions=torch.arange(2048,2072)
    output=worker.model_runner._model_forward(input_ids=ids,positions=positions)
    rows=output[1][:-1]
    assert [r.shape[-1] for r in rows]==list(internal.widths(contract()))
    assert rows[2].tolist()==[[23.]*8]  # pre-RoPE independent packed snapshot
    assert rows[3].tolist()==[[33.]*4] and rows[4].tolist()==[[43.]*2]
    assert rows[5].tolist()==[[23.]*2]  # V is not RoPE-mutated
    assert rows[6].tolist()==[[99.]*4]  # selected attention query row
    assert rows[9].tolist()==[[121.]*4]  # before layer 1 in-place merge
    assert rows[-1].tolist()==[[121.]*4]
    table,stamp=observer.snapshot(output[0][22])
    assert stamp.tolist()==[22,2070,22,24,122]
    frozen=table.clone();slots.fill_(999)
    worker.model_runner._model_forward(input_ids=ids+1,positions=positions)
    assert torch.equal(table,frozen)
    exported=torch.export.export(model,(ids,positions),strict=False).module()
    exported._m4_layer_row.fill_(0)
    assert exported(ids,positions)[1][0].tolist()==[[0.]*4]
    exported._m4_layer_row.fill_(22)
    assert exported(ids,positions)[1][2].tolist()==[[23.]*8]
    observer.close();observer.close()
    assert type(model) is original and not hasattr(model,'_m4_layer_row')
    assert not worker.model_runner.use_aux_hidden_state_outputs and observer.current_slot is None


def test_internal_partial_install_cleanup(monkeypatch):
    worker,model,wrapper,reset,calls,slots=toy_runtime(monkeypatch,fail_reset=True)
    original=type(model)
    with pytest.raises(RuntimeError,match='reset failure'):
        runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],original.forward,internal=True)
    assert type(model) is original and not hasattr(model,'_m4_layer_row')
    assert not worker.model_runner.use_aux_hidden_state_outputs


def test_variable_width_latch_protects_later_async_work():
    latch=trajectory.FirstMismatchCapture([0,0],1)
    table=torch.arange(sum(internal.widths(contract())),dtype=torch.bfloat16).reshape(1,-1)
    stamp=torch.tensor([22,2070,10,24,122]);raw=torch.tensor([1.,2.])
    latch.record(0,raw,raw,torch.tensor([[1]]),layer_table=table,layer_stamp=stamp)
    frozen=latch.selected['layer_table'].clone()
    table.zero_();stamp.zero_()
    latch.record(1,raw,raw,torch.tensor([[0]]),layer_table=table,layer_stamp=stamp)
    assert torch.equal(latch.selected['layer_table'],frozen)
    assert latch.selected['layer_stamp'].tolist()==[22,2070,10,24,122]


def test_exact_bytes_and_signed_zero(case):
    prepared,a,b=pair(case,None)
    for result,zero in ((a,b'\x00\x00'),(b,b'\x00\x80')):
        r=result['layer0_internal']['records'][0]
        payload=zero+base64.b64decode(r['bytes_base64'])[2:]
        r['bytes_base64']=base64.b64encode(payload).decode();r['bytes_sha256']=hashlib.sha256(payload).hexdigest()
    decoded=internal.validate_internal(json.loads(json.dumps(b['layer0_internal'])),b)
    assert decoded[0][1][:2]==b'\x00\x80'
    comparison=internal.analyze_internal(a,b,prepared)['stage_comparisons'][0]
    assert comparison['byte_different_elements']==1 and comparison['numerically_different_elements']==0
    assert comparison['max_abs_diff']==0 and not comparison['bytes_equal']


def test_pre_rope_qkv_byte_slicing_and_semantics(case):
    prepared,a,b=pair(case,None)
    cap=b['layer0_internal'];components=cap['pre_rope_qkv']
    assert [r['component'] for r in components]==['Q','K','V']
    assert [r['shape'] for r in components]==[[1,4],[1,2],[1,2]]
    assert [r['offset_elements'] for r in components]==[0,4,6]
    assert b''.join(base64.b64decode(r['bytes_base64']) for r in components)==base64.b64decode(cap['records'][2]['bytes_base64'])
    components[0]['shape']=[1,2]
    assert internal.analyze_internal(a,b,prepared)['classification']=='INCONCLUSIVE'


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_serialization_rejected(case,value):
    prepared,a,b=pair(case,None)
    table=torch.ones((1,sum(internal.widths(contract()))),dtype=torch.bfloat16)
    table[0,0]=value
    with pytest.raises(ValueError,match='Nonfinite'):
        internal.serialize_internal(table,torch.tensor(a['layer0_internal']['graph_stamp']),contract(),a)


def test_manifest_preserves_old_hashes():
    from pathlib import Path
    root=Path(internal.__file__).parent
    old=json.loads((root/'m4_layer_probe_sources.json').read_text())
    new=json.loads((root/'m4_layer0_probe_sources.json').read_text())
    assert new['files'][:len(old['files'])]==old['files']
    assert len(new['files'])==len({r['path'] for r in new['files']})==40


def test_cleanup_after_internal_forward_exception(monkeypatch):
    worker,model,wrapper,reset,calls,slots=toy_runtime(monkeypatch)
    original=type(model)
    observer=runtime.GraphLayerRuntime(worker,model,reset,lambda:[wrapper],original.forward,internal=True)
    worker._m4_probe=dict(armed=dict(prompt=[999]),output=NS(num_scheduled_tokens={'rid':1}))
    try:
        with pytest.raises(RuntimeError,match='wrong request'):
            worker.model_runner._model_forward(input_ids=torch.tensor([1]),positions=torch.tensor([2070]))
    finally:
        observer.close()
    assert type(model) is original and not hasattr(model,'_m4_layer_row')
    assert observer.cleanup_complete and observer.current_slot is None


def test_unreviewed_runtime_selection_rejected():
    from experiments.m4_layer0_sources import selected_implementations
    model=NS(layers=[Layer()])
    model.layers[0].self_attn.qkv_proj.quant_method=object()
    with pytest.raises(RuntimeError,match='Unreviewed'):
        selected_implementations(model)


def test_internal_module_imports_without_runtime():
    import subprocess,sys
    subprocess.run([sys.executable,'-B','-c',
        'import sys; import experiments.m4_layer0_probe; import experiments.m4_layer0_sources; '
        'assert "vllm" not in sys.modules; assert "torch" not in sys.modules'],check=True)
