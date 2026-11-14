"""Opt-in layer-0 graph return surface and exact BF16 analysis; no runtime imports."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import sys

from experiments.m4_layer_probe import row_identity

INTERNAL_STAGE = 'layer0_internal'
SCHEMA = 'm4.layer0_internal.v1'
STAGES = ('input', 'input_norm', 'qkv_projection', 'q_post_rope', 'k_post_rope',
          'v', 'attention_output', 'o_proj', 'post_attention_norm',
          'post_attention_residual', 'gate_up', 'activation', 'down_proj',
          'return_hidden', 'return_residual')


def widths(contract):
    h, q, kv, m = (contract[k] for k in ('hidden_size','q_size','kv_size','intermediate_size'))
    if any(type(x) is not int or x <= 0 for x in (h,q,kv,m)) or q != h:
        raise ValueError('Invalid layer-0 widths')
    return (h,h,q+2*kv,q,kv,kv,q,h,h,h,2*m,m,h,h,h)


def attention_context(result):
    state = result['state']
    blocks = state['block_ids']
    return dict(computed_tokens_before=state['computed_tokens_before'],
        scheduled_query_tokens=state['scheduled_query_tokens'],execution=dict(state['execution']),
        request_block_ids=blocks,
        request_block_ids_sha256=hashlib.sha256(json.dumps(blocks,separators=(',',':')).encode()).hexdigest())


def packed_components(record,contract):
    """Independent immutable byte slices of the already frozen PRE-RoPE row."""
    packed = base64.b64decode(record['bytes_base64'],validate=True)
    q,kv = contract['q_size'],contract['kv_size']
    components = []
    offset = 0
    for name,size in (('Q',q),('K',kv),('V',kv)):
        payload = packed[2*offset:2*(offset+size)]
        components.append(dict(component=name,shape=[1,size],offset_elements=offset,
            dtype='torch.bfloat16',bytes_base64=base64.b64encode(payload).decode('ascii'),
            bytes_sha256=hashlib.sha256(payload).hexdigest()))
        offset += size
    return components


def observed_forward(model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    """Pinned Qwen2 single-rank path, original operators, only L0 observed.

    This function is traced BEFORE runtime graph capture, never a replay hook.
    Independent gathers precede every possible in-place RoPE/add-norm mutation.
    """
    if intermediate_tensors is not None or inputs_embeds is not None or input_ids is None:
        raise RuntimeError('Layer-0 diagnostic requires single-rank token-ID input')
    rows = []

    def save(value):
        rows.append(value.index_select(0,model._m4_layer_row).clone())

    hidden = model.embed_input_ids(input_ids)
    save(hidden)
    layer = model.layers[0]
    residual = hidden
    hidden = layer.input_layernorm(hidden)
    save(hidden)
    attention = layer.self_attn
    qkv, _ = attention.qkv_proj(hidden)
    save(qkv)  # MUST precede in-place rotary mutation of its split views.
    q,k,v = qkv.split([attention.q_size,attention.kv_size,attention.kv_size],dim=-1)
    q,k = attention.rotary_emb(positions,q,k)
    save(q); save(k); save(v)
    hidden = attention.attn(q,k,v)
    save(hidden)
    hidden,_ = attention.o_proj(hidden)
    save(hidden)
    hidden,residual = layer.post_attention_layernorm(hidden,residual)
    save(hidden); save(residual)
    gate_up,_ = layer.mlp.gate_up_proj(hidden)
    save(gate_up)
    hidden = layer.mlp.act_fn(gate_up)
    save(hidden)
    hidden,_ = layer.mlp.down_proj(hidden)
    save(hidden); save(hidden); save(residual)
    for index in range(1,model.end_layer):
        hidden,residual = model.layers[index](positions,hidden,residual)
    hidden,_ = model.norm(hidden,residual)
    return hidden,rows


def capture_contract(model):
    attention = model.layers[0].self_attn
    if (attention.qk_norm or attention.dual_chunk_attention_config is not None
            or getattr(model.config,'is_causal',True) is not True):
        raise RuntimeError('Unsupported layer-0 attention variant')
    contract = dict(hidden_size=model.config.hidden_size,q_size=attention.q_size,
        kv_size=attention.kv_size,intermediate_size=model.config.intermediate_size,
        selected_layers=[0],tensor_parallel_size=1,pipeline_parallel_size=1,
        installation='after_kv_allocation_before_runtime_capture',graph_modes=['FULL','PIECEWISE'])
    widths(contract)
    return contract


def serialize_internal(table, stamp, contract, result):
    import torch
    sizes = widths(contract)
    if table.dtype != torch.bfloat16 or list(table.shape) != [1,sum(sizes)]:
        raise RuntimeError('Layer-0 table shape/dtype mismatch')
    if stamp.dtype != torch.int64 or list(stamp.shape) != [5]:
        raise RuntimeError('Layer-0 stamp shape/dtype mismatch')
    identity = row_identity(result)
    cpu = table.detach().to(device='cpu').contiguous()
    records = []
    for stage,row in zip(STAGES,cpu.split(sizes,dim=-1)):
        row = row.reshape(-1).clone().reshape(1,-1)
        payload = row.view(torch.uint8).numpy().tobytes()
        records.append(dict(**identity,stage=stage,layer_index=0,shape=list(row.shape),
            dtype='torch.bfloat16',stride=list(row.stride()),layout='torch.strided',
            byte_order=sys.byteorder,bytes_base64=base64.b64encode(payload).decode('ascii'),
            bytes_sha256=hashlib.sha256(payload).hexdigest(),nan_count=0,inf_count=0))
    capture = dict(schema_version=SCHEMA,contract=contract,records=records,
                   graph_stamp=stamp.to(device='cpu').tolist(),
                   selected_kv_slot=int(stamp[-1].to(device='cpu').item()),
                   attention_context=attention_context(result))
    capture['pre_rope_qkv'] = packed_components(records[2],contract)
    validate_internal(capture,result)
    return capture


def validate_internal(capture,result):
    contract = capture['contract']
    sizes = widths(contract)
    identity = row_identity(result)
    execution = result['state']['execution']
    if (capture['schema_version'] != SCHEMA or contract['selected_layers'] != [0]
            or any(type(v) is not int for v in contract['selected_layers'])
            or type(contract['tensor_parallel_size']) is not int or contract['tensor_parallel_size'] != 1
            or type(contract['pipeline_parallel_size']) is not int or contract['pipeline_parallel_size'] != 1
            or contract['installation'] != 'after_kv_allocation_before_runtime_capture'
            or contract['graph_modes'] != ['FULL','PIECEWISE']
            or identity['generation_index'] != 549 or identity['logical_token_position'] != 2070
            or result['condition'] not in ('SEQUENTIAL_DECODE_COLD','FORCED_PREFILL_COLD')
            or execution['cudagraph_mode'] != ('FULL' if result['condition']=='SEQUENTIAL_DECODE_COLD' else 'PIECEWISE')
            or capture['graph_stamp'][:4] != [identity['source_query_row'],2070,
                result['full_logical_input_token_ids'][-1],execution['padded_tokens']]
            or len(capture['graph_stamp']) != 5
            or any(type(v) is not int for v in capture['graph_stamp'])
            or type(capture['selected_kv_slot']) is not int or capture['selected_kv_slot'] < 0
            or capture['selected_kv_slot'] != capture['graph_stamp'][4]):
        raise ValueError('Invalid layer-0 contract/owner/graph stamp')
    if capture['attention_context'] != attention_context(result):
        raise ValueError('Layer-0 attention context differs from request')
    if len(capture['records']) != len(STAGES):
        raise ValueError('Missing/duplicate layer-0 stage')
    if capture['pre_rope_qkv'] != packed_components(capture['records'][2],contract):
        raise ValueError('Pre-RoPE packed Q/K/V slices differ')
    decoded = []
    for record,stage,size in zip(capture['records'],STAGES,sizes):
        if (record['stage'] != stage or type(record['layer_index']) is not int or record['layer_index'] != 0
                or any(record[k] != v for k,v in identity.items())
                or any(type(record[k]) is not int for k in ('generation_index','logical_token_position','source_query_row'))
                or record['shape'] != [1,size] or record['dtype'] != 'torch.bfloat16'
                or record['stride'] != [size,1] or record['layout'] != 'torch.strided'
                or any(type(v) is not int for v in record['shape']+record['stride'])
                or record['byte_order'] not in ('little','big')
                or record['nan_count'] != 0 or record['inf_count'] != 0):
            raise ValueError('Wrong layer-0 stage/order/owner/shape/dtype/layout')
        payload = base64.b64decode(record['bytes_base64'],validate=True)
        if len(payload) != size*2 or hashlib.sha256(payload).hexdigest() != record['bytes_sha256']:
            raise ValueError('Invalid layer-0 bytes/hash')
        words = struct.unpack(('<' if record['byte_order']=='little' else '>')+'H'*size,payload)
        values = [struct.unpack('>f',struct.pack('>I',w << 16))[0] for w in words]
        if any(not math.isfinite(v) for v in values):
            raise ValueError('Nonfinite layer-0 state')
        decoded.append((values,payload))
    return decoded


def classify(comparisons):
    """Only claim an operation when all its immediate current-row inputs match."""
    first = next((i for i,r in enumerate(comparisons) if not r['bytes_equal']),None)
    if first is None:
        return dict(classification='NO_INTERNAL_DIVERGENCE_REPRODUCED')
    stage = comparisons[first]['stage']
    operation = {'input_norm':'input_norm','qkv_projection':'qkv_projection',
        'q_post_rope':'rope','k_post_rope':'rope','attention_output':'attention_output',
        'o_proj':'o_proj','post_attention_norm':'post_attention_norm',
        'post_attention_residual':'post_attention_norm','gate_up':'mlp_gate_up',
        'activation':'mlp_activation','down_proj':'mlp_down_proj'}.get(stage)
    if operation:
        return dict(classification='FIRST_INTERNAL_DIVERGENCE_IDENTIFIED',FIRST_INTERNAL_DIVERGENCE=operation)
    return dict(classification='INTERNAL_DIVERGENCE_INTERVAL_IDENTIFIED',
        INTERNAL_DIVERGENCE_INTERVAL=[('qkv_projection' if stage=='v' else comparisons[first-1]['stage']) if first else 'incoming_state',stage])


def analyze_internal(sequential,forced,case,expected_source_commit=None):
    from experiments.m4_decode_trajectory import analyze_trajectory
    base = analyze_trajectory(sequential,forced,case,expected_source_commit)
    out = dict(root_cause='ROOT_CAUSE_NOT_PROVEN',full_sweep='FULL_SWEEP_BLOCKED',
               trajectory_classification=base['classification'])
    try:
        if (case.get('localization_stage') != INTERNAL_STAGE or base.get('reasons')
                or base['classification'] != 'H_TRAJECTORY_SUPPORTED' or base.get('generation_index') != 549):
            raise ValueError('Historical generation-549 flip not reproduced with valid provenance')
        a,b = sequential['layer0_internal'],forced['layer0_internal']
        if a['contract'] != b['contract']:
            raise ValueError('Layer-0 contracts differ')
        pairs = zip(a['records'],b['records'],validate_internal(a,sequential),validate_internal(b,forced))
        comparisons = []
        for ar,br,(av,ab),(bv,bb) in pairs:
            if ar['byte_order'] != br['byte_order']:
                raise ValueError('Layer-0 byte orders differ')
            different = [i for i in range(len(av)) if ab[2*i:2*i+2] != bb[2*i:2*i+2]]
            delta = [abs(x-y) for x,y in zip(av,bv)]
            comparisons.append(dict(stage=ar['stage'],bytes_equal=ab==bb,
                byte_different_elements=len(different),numerically_different_elements=sum(d!=0 for d in delta),
                first_different_index=different[0] if different else None,
                max_abs_diff=max(delta),mean_abs_diff=sum(delta)/len(delta),nan_count=0,inf_count=0))
        return dict(out,**classify(comparisons),stage_comparisons=comparisons,
            interpretation='Earliest observed boundary; numerical inequality alone does not establish a defect.',
            disposition='TECHNICAL_DISPOSITION_REQUIRED_AFTER_GPU_VALIDATION')
    except (KeyError,ValueError,TypeError,IndexError,struct.error) as exc:
        return dict(out,classification='INCONCLUSIVE',reasons=[str(exc)])
