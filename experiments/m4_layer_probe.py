"""Exact, row-only Qwen2 boundary records. No runtime imports on import.

Layer boundaries mean the PAIR (MLP output, post-attention residual merge),
before the next fused add/norm. Neither component alone is a residual stream.
The graph adapter is in m4_layer_probe_runtime; the trajectory owns the latch.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import math
import struct
import sys

LAYER_STAGE = 'layer_boundaries'
SCHEMA = 'm4.layer_boundaries.v1'
POINT = 'embedding; decoder return pair before next residual merge; gathered final norm'


def stage_plan(layers, selected=None):
    if type(layers) is not int or layers <= 0:
        raise ValueError('Invalid model layer count')
    selected = list(range(layers)) if selected is None else selected
    if (not isinstance(selected,list) or not selected or
            any(type(i) is not int or not 0 <= i < layers for i in selected) or
            selected != sorted(set(selected))):
        raise ValueError('Invalid layer indices/order')
    plan = [('embedding',None,'embedding')]
    for i in selected:
        plan.extend([(f'layer_{i}',i,'hidden_states'),(f'layer_{i}',i,'residual')])
    return plan + [('final_norm',None,'normalized_hidden_states')]


def row_identity(result):
    state = result['state']
    query = state['scheduled_query_tokens']
    position = len(result['full_logical_input_token_ids'])-1
    if (type(query) is not int or query <= 0 or
            state['computed_tokens_before'] + query-1 != position):
        raise ValueError('Layer query row does not map to logical position')
    return dict(condition=result['condition'],request_id=state['request_id'],
        generation_index=result['identity']['generation_index'],logical_token_position=position,
        logical_input_sha256=result['identity']['logical_input_sha256'],source_query_row=query-1)


def serialize_layers(table, stamp, contract, result):
    """Only after abort/barrier. The table was frozen by the sampler's latch."""
    import torch
    plan = stage_plan(contract['num_hidden_layers'],contract['selected_layers'])
    width = contract['hidden_size']
    if (table.dtype != torch.bfloat16 or list(table.shape) != [len(plan),width]
            or stamp.dtype != torch.int64 or list(stamp.shape) != [4]):
        raise RuntimeError('Layer table/stamp shape/dtype mismatch')
    identity = row_identity(result)
    expected_stamp = [identity['source_query_row'],identity['logical_token_position'],
        result['full_logical_input_token_ids'][-1],result['state']['execution']['padded_tokens']]
    if stamp.to(device='cpu').tolist() != expected_stamp:
        raise RuntimeError('Graph layer stamp row/position/input/geometry mismatch')
    cpu = table.detach().to(device='cpu').contiguous()
    records = []
    for i,(stage,layer,component) in enumerate(plan):
        row = cpu[i:i+1]
        values = row.to(dtype=torch.float32).reshape(-1).tolist()
        if any(not math.isfinite(v) for v in values):
            raise RuntimeError('Nonfinite layer boundary')
        payload = row.view(torch.uint8).numpy().tobytes()
        records.append(dict(stage=stage,layer_index=layer,component=component,**identity,
            shape=[1,width],dtype='torch.bfloat16',stride=[width,1],layout='torch.strided',
            byte_order=sys.byteorder,bytes_base64=base64.b64encode(payload).decode('ascii'),
            bytes_sha256=hashlib.sha256(payload).hexdigest(),values=values,nan_count=0,inf_count=0))
    return dict(schema_version=SCHEMA,measurement_point=POINT,contract=contract,
        graph_stamp=expected_stamp,records=records)


def validate_layers(capture, result):
    contract = capture['contract']
    width = contract['hidden_size']
    plan = stage_plan(contract['num_hidden_layers'],contract['selected_layers'])
    if (capture['schema_version'] != SCHEMA or capture['measurement_point'] != POINT
            or type(width) is not int or width <= 0
            or type(contract['tensor_parallel_size']) is not int or contract['tensor_parallel_size'] != 1
            or type(contract['pipeline_parallel_size']) is not int or contract['pipeline_parallel_size'] != 1
            or contract['installation'] != 'after_kv_allocation_before_runtime_capture'
            or contract['graph_modes'] != ['FULL','PIECEWISE']):
        raise ValueError('Unsupported layer capture contract')
    identity = row_identity(result)
    if identity['generation_index'] != 549:
        raise ValueError('Layer localization requires boundary 549')
    execution = result['state']['execution']
    expected_mode = 'FULL' if result['condition']=='SEQUENTIAL_DECODE_COLD' else 'PIECEWISE'
    if execution['cudagraph_mode'] != expected_mode:
        raise ValueError('Unsupported layer graph path')
    if (not isinstance(capture['graph_stamp'],list) or len(capture['graph_stamp']) != 4
            or any(type(v) is not int for v in capture['graph_stamp'])
            or capture['graph_stamp'] != [identity['source_query_row'],identity['logical_token_position'],
            result['full_logical_input_token_ids'][-1],execution['padded_tokens']]):
        raise ValueError('Layer graph stamp mismatch')
    records = capture['records']
    if len(records) != len(plan):
        raise ValueError('Missing/duplicate layer stage')
    decoded = []
    for r,(stage,layer,component) in zip(records,plan):
        if ((r['stage'],r['layer_index'],r['component']) != (stage,layer,component)
                or (layer is not None and type(r['layer_index']) is not int)):
            raise ValueError('Wrong layer stage/order/component')
        if (any(r[k] != v for k,v in identity.items()) or
                any(type(r[k]) is not int for k in ('generation_index','logical_token_position','source_query_row'))):
            raise ValueError('Layer owner/prefix/row/boundary mismatch')
        if (r['shape'] != [1,width] or any(type(v) is not int for v in r['shape'])
                or r['dtype'] != 'torch.bfloat16'
                or r['stride'] != [width,1] or r['layout'] != 'torch.strided'
                or r['byte_order'] not in ('little','big') or r['nan_count'] != 0 or r['inf_count'] != 0):
            raise ValueError('Layer shape/dtype/layout/nonfinite mismatch')
        payload = base64.b64decode(r['bytes_base64'],validate=True)
        if len(payload) != width*2 or hashlib.sha256(payload).hexdigest() != r['bytes_sha256']:
            raise ValueError('Layer byte serialization mismatch')
        words = struct.unpack(('<' if r['byte_order']=='little' else '>')+'H'*width,payload)
        values = [struct.unpack('>f',struct.pack('>I',w << 16))[0] for w in words]
        if (len(r['values']) != width or any(not math.isfinite(v) for v in values)
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in r['values'])
                or any(x != y or math.copysign(1,x) != math.copysign(1,y)
                       for x,y in zip(values,r['values']))):
            raise ValueError('Layer values do not match finite BF16 bytes')
        decoded.append((values,payload))
    return decoded


def analyze_layers(sequential, forced, case, expected_source_commit=None):
    from experiments.m4_decode_trajectory import analyze_trajectory
    base = analyze_trajectory(sequential,forced,case,expected_source_commit)
    result = dict(root_cause='ROOT_CAUSE_NOT_PROVEN',full_sweep='FULL_SWEEP_BLOCKED',
        trajectory_classification=base['classification'],measurement_point=POINT)
    try:
        if case.get('localization_stage') != LAYER_STAGE or base.get('reasons'):
            raise ValueError('Invalid layer localization trajectory/provenance')
        if base.get('generation_index') != 549 or base['classification'] != 'H_TRAJECTORY_SUPPORTED':
            raise ValueError('Historical generation-549 flip not reproduced')
        a = validate_layers(sequential['layer_boundaries'],sequential)
        b = validate_layers(forced['layer_boundaries'],forced)
        contract = sequential['layer_boundaries']['contract']
        if contract != forced['layer_boundaries']['contract']:
            raise ValueError('Layer model/capture contracts differ')
        stages = []
        for r,(av,ab),(bv,bb),other in zip(sequential['layer_boundaries']['records'],a,b,
                                         forced['layer_boundaries']['records']):
            if r['byte_order'] != other['byte_order']:
                raise ValueError('Layer byte orders differ')
            delta = [abs(x-y) for x,y in zip(av,bv)]
            numerical = [i for i,d in enumerate(delta) if d]
            byte_indices = [i for i in range(len(av)) if ab[2*i:2*i+2] != bb[2*i:2*i+2]]
            stages.append(dict(stage=r['stage'],layer_index=r['layer_index'],component=r['component'],
                bytes_equal=ab==bb,numerically_different_elements=len(numerical),
                first_different_index=byte_indices[0] if byte_indices else None,
                first_numerically_different_index=numerical[0] if numerical else None,
                byte_different_elements=len(byte_indices),max_abs_diff=max(delta),mean_abs_diff=sum(delta)/len(delta)))
        groups = []
        for r in stages:
            if not groups or groups[-1]['stage'] != r['stage']:
                groups.append(dict(stage=r['stage'],layer_index=r['layer_index'],bytes_equal=True))
            groups[-1]['bytes_equal'] &= r['bytes_equal']
        first = next((i for i,g in enumerate(groups) if not g['bytes_equal']),None)
        out = dict(result,generation_index=549,historical_boundary_flip_reproduced=True,
            raw_logit_difference=base['vector_difference'],stage_comparisons=stages,
            first_equal_stage=groups[0]['stage'] if groups[0]['bytes_equal'] else None,
            last_equal_stage_before_divergence=groups[first-1]['stage'] if first else None,
            first_different_stage=groups[first]['stage'] if first is not None else None,
            EARLIEST_OBSERVED_DIVERGENCE_STAGE=groups[first]['stage'] if first is not None else None)
        if first is None:
            return dict(out,classification='NO_LAYER_BOUNDARY_DIVERGENCE_REPRODUCED',
                last_equal_stage_before_divergence='final_norm')
        current = groups[first]
        if current['stage']=='embedding':
            classification = 'EMBEDDING_STATE_DIVERGENCE'
        elif current['stage']=='final_norm':
            classification = ('FINAL_NORM_STATE_DIVERGENCE' if
                              contract['selected_layers']==list(range(contract['num_hidden_layers']))
                              else 'DIVERGENCE_INTERVAL_IDENTIFIED')
        else:
            classification = ('FIRST_DIVERGENT_LAYER_IDENTIFIED' if
                contract['selected_layers'][:first]==list(range(current['layer_index']+1))
                else 'DIVERGENCE_INTERVAL_IDENTIFIED')
        if classification=='DIVERGENCE_INTERVAL_IDENTIFIED':
            out['DIVERGENCE_INTERVAL'] = [groups[first-1]['stage'],current['stage']]
        return dict(out,classification=classification,
            interpretation='First observed captured state boundary only; no internal operation or root cause proven.')
    except (KeyError,ValueError,TypeError,IndexError,binascii.Error,struct.error) as exc:
        return dict(result,classification='INCONCLUSIVE',reasons=[str(exc)])
