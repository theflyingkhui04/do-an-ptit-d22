"""Small, opt-in split at the graph-external LM-head input boundary.

No layer hooks, KV reads, graph changes or tensor/runtime imports on import.
The parent trajectory diagnostic owns request, prefix, cache and lifecycle proof.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import math
import struct
import sys

HEAD_STAGE = 'lm_head_input'
HEAD_SCHEMA = 'm4.head_input.v1'
HEAD_POINT = 'model.compute_logits entry; gathered post-final-norm row; before head projection'


def head_contract(runner):
    width = runner.model.config.hidden_size
    head = runner.model.lm_head
    if type(width) is not int or width <= 0 or head.tp_size != 1:
        raise RuntimeError('Single-rank head with observed hidden width required')
    method = head.quant_method
    if method is None:
        raise RuntimeError('Missing observed LM-head quantization method')
    return dict(hidden_size=width,tensor_parallel_size=head.tp_size,
        quant_method_class=f'{type(method).__module__}.{type(method).__name__}',
        head_dtype=str(runner.model.logits_processor.head_dtype))


def clone_head_input(args, kwargs, width):
    """Called before projection, after runner gathers the valid last-query row.

    Accept precisely the pinned signature and dense BF16 [1, hidden] layout.
    Reading shape/stride/layout is host metadata, not a device synchronization.
    """
    import torch
    if len(args) == 1 and not kwargs:
        tensor = args[0]
    elif not args and set(kwargs) == {'hidden_states'}:
        tensor = kwargs['hidden_states']
    else:
        raise RuntimeError('Unsupported compute_logits argument ownership/signature')
    if (not isinstance(tensor,torch.Tensor) or tensor.layout != torch.strided
            or list(tensor.shape) != [1,width] or tensor.dtype != torch.bfloat16
            or tensor.stride() != (width,1)):
        raise RuntimeError('Unsupported LM-head input shape/dtype/layout; one gathered BF16 row required')
    return tensor[0].detach().clone(), dict(shape=list(tensor.shape),
        stride=list(tensor.stride()),storage_offset=tensor.storage_offset(),layout=str(tensor.layout))


def serialize_head_input(vector, layout, contract, result):
    """After abort: serialize all small hidden-state values and exact BF16 bytes."""
    import torch
    cpu = vector.detach().to(device='cpu').contiguous()
    values = cpu.to(dtype=torch.float32).tolist()
    nan_count = sum(math.isnan(v) for v in values)
    inf_count = sum(math.isinf(v) for v in values)
    if nan_count or inf_count:
        raise RuntimeError('Nonfinite LM-head input; preserve failure vectors')
    payload = cpu.view(torch.uint8).numpy().tobytes()
    state, identity = result['state'], result['identity']
    return dict(schema_version=HEAD_SCHEMA,measurement_point=HEAD_POINT,**layout,
        dtype=str(vector.dtype),byte_order=sys.byteorder,values=values,
        bytes_base64=base64.b64encode(payload).decode('ascii'),bytes_sha256=hashlib.sha256(payload).hexdigest(),
        nan_count=nan_count,inf_count=inf_count,head_contract=contract,
        request_id=state['request_id'],generation_index=identity['generation_index'],
        logical_input_sha256=identity['logical_input_sha256'],
        logical_token_position=len(result['full_logical_input_token_ids'])-1,
        source_query_row=state['scheduled_query_tokens']-1,logits_row=0)


def validate_head_input(record, result):
    contract = record['head_contract']
    width = contract['hidden_size']
    if (record['schema_version'] != HEAD_SCHEMA or record['measurement_point'] != HEAD_POINT
            or type(width) is not int or width <= 0 or record['shape'] != [1,width]
            or record['dtype'] != 'torch.bfloat16' or record['layout'] != 'torch.strided'
            or record['stride'] != [width,1] or type(record['storage_offset']) is not int
            or record['storage_offset'] < 0 or contract['tensor_parallel_size'] != 1
            or not contract['quant_method_class'] or not isinstance(contract['head_dtype'],str)):
        raise ValueError('Unsupported/missing LM-head stage/shape/dtype/layout contract')
    identity, state = result['identity'], result['state']
    if (record['request_id'] != state['request_id'] or record['generation_index'] != identity['generation_index']
            or record['logical_input_sha256'] != identity['logical_input_sha256']
            or record['logical_token_position'] != len(result['full_logical_input_token_ids'])-1
            or record['source_query_row'] != state['scheduled_query_tokens']-1 or record['logits_row'] != 0):
        raise ValueError('LM-head row/owner/prefix/boundary identity mismatch')
    values = record['values']
    if (len(values) != width or record['nan_count'] != 0 or record['inf_count'] != 0
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)):
        raise ValueError('Incomplete/nonfinite LM-head input')
    payload = base64.b64decode(record['bytes_base64'],validate=True)
    if (record['byte_order'] not in ('little','big') or len(payload) != width*2
            or hashlib.sha256(payload).hexdigest() != record['bytes_sha256']):
        raise ValueError('Invalid exact BF16 serialization')
    order = '<' if record['byte_order'] == 'little' else '>'
    words = struct.unpack(order+'H'*width,payload)
    decoded = [struct.unpack('>f',struct.pack('>I',word << 16))[0] for word in words]
    if any(x != y or math.copysign(1,x) != math.copysign(1,y) for x,y in zip(decoded,values)):
        raise ValueError('LM-head values differ from preserved BF16 bytes')
    return values,payload


def analyze_head_input(sequential, forced, case, expected_source_commit=None):
    from experiments.m4_decode_trajectory import analyze_trajectory
    base = analyze_trajectory(sequential,forced,case,expected_source_commit)
    result = dict(root_cause='ROOT_CAUSE_NOT_PROVEN',full_sweep='FULL_SWEEP_BLOCKED',
        trajectory_classification=base['classification'],measurement_point=HEAD_POINT)
    try:
        if base.get('reasons'):
            raise ValueError('; '.join(base['reasons']))
        if case.get('localization_stage') != HEAD_STAGE or base['generation_index'] != 549:
            raise ValueError('Requested generation-549 localization boundary not reproduced')
        a, a_bytes = validate_head_input(sequential['head_input'],sequential)
        b, b_bytes = validate_head_input(forced['head_input'],forced)
        if (sequential['head_input']['head_contract'] != forced['head_input']['head_contract']
                or len(a) != len(b) or sequential['head_input']['byte_order'] != forced['head_input']['byte_order']):
            raise ValueError('A/B head execution/shape/serialization contracts differ')
        delta = [abs(x-y) for x,y in zip(a,b)]
        numerical_indices = [i for i,v in enumerate(delta) if v]
        byte_indices = [i for i in range(len(a)) if a_bytes[2*i:2*i+2] != b_bytes[2*i:2*i+2]]
        indices = sorted({0,len(a)-1,max(range(len(a)),key=delta.__getitem__)})
        maximum = max(delta)
        if maximum:
            classification = 'UPSTREAM_HIDDEN_STATE_DIVERGENCE'
        elif a_bytes != b_bytes:
            classification = 'HEAD_INPUT_BYTES_DIFFER_WITH_EQUAL_VALUES'
        elif base['vector_difference']['max_absolute_difference']:
            classification = 'LM_HEAD_OUTPUT_DIVERGENCE_WITH_IDENTICAL_INPUT'
        else:
            classification = 'NO_BOUNDARY_DIVERGENCE_REPRODUCED'
        reproduced = base['classification'] == 'H_TRAJECTORY_SUPPORTED'
        return dict(result,classification=classification,generation_index=549,
            historical_boundary_flip_reproduced=reproduced,
            provenance_runtime_review_required=not reproduced,
            raw_logit_difference=base['vector_difference'],head_input_difference=dict(
                shape=[1,len(a)],dtype='torch.bfloat16',bytes_equal=a_bytes==b_bytes,
                max_absolute_difference=maximum,mean_absolute_difference=sum(delta)/len(delta),
                numerically_different_elements=len(numerical_indices),
                first_numerically_different_index=numerical_indices[0] if numerical_indices else None,
                byte_different_elements=len(byte_indices),first_byte_different_index=byte_indices[0] if byte_indices else None,
                nan_count=0,inf_count=0,selected_elements=[dict(index=i,sequential=a[i],
                    forced=b[i],absolute_difference=delta[i]) for i in indices]),
            interpretation=('Boundary localization only; no first layer, KV or kernel cause proven.' if reproduced else
                'Historical boundary flip not reproduced; review provenance/runtime before historical attribution. '
                'This result does not falsify the preserved trajectory observation.'))
    except (KeyError,ValueError,TypeError,IndexError,binascii.Error,struct.error) as exc:
        return dict(result,classification='INCONCLUSIVE',reasons=[str(exc)])


def require_completed_lifecycle(cleanup, lifecycle):
    """Offline localization cannot turn a failed/unfinished run into success."""
    if (cleanup['hooks_restored'] is not True or cleanup['engine_shutdown'] is not True
            or cleanup['worker_results'] != [dict(restored=True,captures_released=True,
                cache_observer_restored=True)] or lifecycle['status'] != 'COMPLETE'
            or lifecycle['gpu_before']['processes'] or lifecycle['gpu_after']['processes']):
        raise ValueError('Localization run cleanup/release incomplete or failed')
    # Reuse the frozen release policy; this adds no new memory tolerance.
    from experiments.m4_runtime import require_gpu_release
    try:
        require_gpu_release(lifecycle['gpu_before'],lifecycle['gpu_after'])
    except (RuntimeError,ValueError) as exc:
        raise ValueError('Localization run cleanup/release incomplete or failed') from exc
