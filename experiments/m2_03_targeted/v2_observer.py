"""Index-81 V2 observation. No inference inputs, settings or outputs are written.

GPU inputs/logits are cloned on the existing stream. Host reads happen only
after sampling. FULL replay uses decisions observed while that exact graph was
captured; it never substitutes a fresh attention call for the graph's kernels.
"""
import os
import threading
import uuid
import weakref
from pathlib import Path

from . import canonical
from .common import (canonical_observation, observation_spec, need, load, save,
                     encoded, decode, sha, event, write_new)

_state = threading.local()


def descriptor(desc):
    return dict(cg_mode=desc.cg_mode.name, **{k: getattr(desc, k) for k in (
        'num_tokens', 'num_reqs', 'uniform_token_count', 'max_query_len',
        'num_active_loras', 'num_ubatches')})


def capture(key):
    return dict(key=key, force_3d=False, expected_calls=28, calls=[])


def buffer_signature(model_inputs, metadata):
    """CPU metadata only: prove which persistent buffers the graph reads."""
    def signature(tensor):
        need(tensor is not None, 'V2 embedded/absent model input unsupported')
        storage = tensor.untyped_storage()
        return dict(pointer=tensor.data_ptr(), shape=list(tensor.shape),
                    stride=list(tensor.stride()), dtype=str(tensor.dtype), device=str(tensor.device),
                    storage_pointer=storage.data_ptr(), storage_bytes=storage.nbytes())
    return dict(inputs={k: signature(model_inputs[k]) for k in ('input_ids', 'positions')},
        attention={name: {k: signature(getattr(m, k)) for k in (
            'query_start_loc', 'seq_lens', 'block_table', 'slot_mapping')}
            for name, m in metadata.items()})


def attention_shape(metadata):
    return {name: dict(max_query_len=int(m.max_query_len), num_actual_tokens=int(m.num_actual_tokens),
        num_seqs=int(m.query_start_loc.shape[0]) - 1) for name, m in metadata.items()}


def same_buffers(captured, live):
    if captured['inputs'] != live['inputs'] or set(captured['attention']) != set(live['attention']):
        return False
    for layer, fields in captured['attention'].items():
        if set(fields) != set(live['attention'][layer]):
            return False
        for key, cap in fields.items():
            actual = live['attention'][layer][key]
            # Metadata can expose a shorter prefix view at runtime. Kernels in
            # the graph still read the same persistent allocation and address.
            if ({k: v for k, v in cap.items() if k != 'shape'}
                    != {k: v for k, v in actual.items() if k != 'shape'}):
                return False
            if len(cap['shape']) != len(actual['shape']) or not cap['shape'] or (
                    cap['shape'][0] < actual['shape'][0] or cap['shape'][1:] != actual['shape'][1:]):
                return False
    return True


def graph_inputs(config, manager, desc, model_inputs, metadata, warmup):
    if warmup or desc.cg_mode.name != 'FULL' or not 1 <= desc.num_tokens <= 8:
        return
    if manager.vllm_config.model_config.hf_config.num_hidden_layers != 28:
        return
    signatures = getattr(manager, '_m203_v2_inputs', None)
    if signatures is None:
        signatures = manager._m203_v2_inputs = {}
    signatures[desc] = dict(buffers=buffer_signature(model_inputs, metadata),
                            attention_metadata=attention_shape(metadata))


def initialized(config, runner):
    if runner.vllm_config.model_config.hf_config.num_hidden_layers != 28:
        return  # A drafter's runner is not the pinned target runner.
    identity = canonical.runner_identity(runner)
    need(identity['use_v2_model_runner'] is True, 'V2 initialization from unsupported runner')
    save(Path(config['directory']) / 'runtime_identity.json', dict(identity,
        execution_sha=config['execution_sha'], lane=config['lane']))
    _state.frames = {}
    _state.sample = None


def graph_begin(config, manager, desc, graph):
    from .observer import _local
    _state.graph = None
    d = descriptor(desc)
    if (manager.vllm_config.model_config.hf_config.num_hidden_layers != 28
            or d['cg_mode'] != 'FULL' or not 1 <= d['num_tokens'] <= 8
            or d['num_active_loras'] != 0 or d['num_ubatches'] != 1):
        return
    need(_local.production_attention_capture is None, 'overlapping V2 graph/forward observation')
    item = dict(key=uuid.uuid4().hex, descriptor=d)
    binding = getattr(manager, '_m203_v2_inputs', {}).get(desc)
    need(binding is not None, 'V2 capture input-buffer linkage missing')
    item.update(buffer_signature=binding['buffers'], attention_metadata=binding['attention_metadata'])
    item['dispatch'] = capture(item['key'])
    _state.graph = (manager, desc, graph, item)
    _local.production_attention_capture = item['dispatch']


def graph_end(config, manager, desc, graph):
    from .observer import _local
    pending = getattr(_state, 'graph', None)
    if pending is None:
        return
    _state.graph = None
    _local.production_attention_capture = None
    owner, original_desc, original_graph, item = pending
    need(owner is manager and desc == original_desc and graph is original_graph
         and manager.graphs.get(desc) is graph, 'ambiguous V2 graph capture identity')
    need(len(item['dispatch']['calls']) == 28, 'incomplete V2 graph production dispatch')
    trace = dict(schema='m203.targeted.v2.graph.v1', execution_sha=config['execution_sha'],
                 lane=config['lane'], **item)
    rel = 'graphs/' + item['key'] + '.json'
    raw = encoded(trace)
    write_new(Path(config['directory']) / rel, raw)
    cache = getattr(manager, '_m203_v2_observations', None)
    if cache is None:
        cache = manager._m203_v2_observations = {}
    # Never extend a CUDA graph's lifetime (profiling explicitly frees graphs).
    cache[desc] = (weakref.ref(graph), dict(file=rel, sha256=sha(raw)), trace)


def graph_replay(config, manager, desc, graph):
    frame = getattr(_state, 'forward', None)
    if frame is None:
        return
    need(frame['_manager'] is manager and frame['descriptor'] == descriptor(desc)
         and frame['descriptor']['cg_mode'] == 'FULL', 'V2 replay/forward descriptor mismatch')
    cached = getattr(manager, '_m203_v2_observations', {}).get(desc)
    need(cached is not None and cached[0]() is graph and manager.graphs.get(desc) is graph,
         'missing/ambiguous exact V2 replay graph observation')
    _, reference, trace = cached
    need(trace['execution_sha'] == config['execution_sha'] and trace['lane'] == config['lane'],
         'V2 replay graph run mismatch')
    need(same_buffers(trace['buffer_signature'], frame['buffer_signature']),
         'V2 graph replay uses different input/KV buffers')
    need('graph_reference' not in frame, 'duplicate V2 graph replay')
    frame['graph_reference'] = dict(reference)
    frame['dispatch'] = decode(encoded(trace['dispatch']))


def prepared(config, runner, scheduled, batch, desc, tables, slots, metadata, dummy):
    from .observer import _local, device_snapshot
    _state.forward = None
    if dummy:
        return
    need(batch.num_reqs == 1 and len(batch.req_ids) == 1, 'V2 observation requires one real request')
    rid = batch.req_ids[0]
    state_index = int(batch.idx_mapping_np[0])
    need(runner.req_states.req_id_to_index.get(rid) == state_index, 'V2 request/state-slot ambiguity')
    prompt_len = int(runner.req_states.prompt_len.np[state_index])
    optimistic = int(batch.num_computed_tokens_np[0]) - prompt_len + 1
    if not 81 - 16 <= optimistic <= 81 + 8:
        return
    need(not batch.has_prefill and not batch.has_structured_output_reqs,
         'unsupported V2 prefill/grammar at observation boundary')
    need(runner.batch_sharder is None and runner.pcp_manager is None
         and runner.adaptive_verification is None and runner.ubatch_runner is None,
         'unsupported V2 partitioned/adaptive-verification linkage')
    pc = runner.vllm_config.parallel_config
    need(all(getattr(pc, k, 1) == 1 for k in ('tensor_parallel_size', 'pipeline_parallel_size',
        'data_parallel_size', 'decode_context_parallel_size')), 'V2 parallel linkage unsupported')
    need(runner.req_states.num_reqs == 1, 'multiple V2 resident requests')
    dispatch = getattr(scheduled, '_m203_observation_provenance', None)
    count = scheduled.num_scheduled_tokens.get(rid)
    drafts = len(scheduled.scheduled_spec_decode_tokens.get(rid, []))
    need(isinstance(dispatch, dict) and dispatch.get('state') == 'OBSERVATION_ONLY'
         and type(dispatch.get('scheduler_step')) is int, 'V2 scheduler linkage missing')
    need(count == scheduled.total_num_scheduled_tokens == batch.num_tokens
         and count == int(batch.num_scheduled_tokens[0]) and 1 <= count <= 5
         and drafts == batch.num_draft_tokens and drafts in range(5), 'V2 scheduled query/draft mismatch')
    need(isinstance(metadata, dict) and len(metadata) == 28 and tables is not None and slots is not None,
         'V2 target attention metadata incomplete')
    key = uuid.uuid4().hex
    frame = dict(schema='m203.targeted.v2.forward.v1', key=key, lane=config['lane'],
        execution_sha=config['execution_sha'], request_id=rid, generated_index=81,
        observation_spec=observation_spec(config), scheduler_step=dispatch['scheduler_step'],
        observation_scheduler_provenance=dict(dispatch), state_index=state_index,
        prompt_len=prompt_len, scheduled_tokens=count, draft_count=drafts,
        runtime_identity=canonical.runner_identity(runner),
        prefix_cache_enabled=bool(runner.cache_config.enable_prefix_caching),
        resolved_cudagraph_mode=runner.vllm_config.compilation_config.cudagraph_mode.name,
        descriptor=descriptor(desc), padded_token_count=batch.num_tokens_after_padding,
        buffer_signature=buffer_signature(dict(input_ids=batch.input_ids, positions=batch.positions), metadata),
        attention_metadata=attention_shape(metadata),
        async_scheduling=bool(runner.vllm_config.scheduler_config.async_scheduling),
        same_internal_state_proven=False, runtime_probe_changes_timing=True,
        dispatch=capture(key), _batch=batch, _manager=runner.cudagraph_manager,
        _tensors={name: device_snapshot(value) for name, value in dict(
            input_ids=batch.input_ids[:count], positions=batch.positions[:count],
            query_start_loc=batch.query_start_loc, seq_lens=batch.seq_lens,
            logits_indices=batch.logits_indices, idx_mapping=batch.idx_mapping,
            expanded_idx_mapping=batch.expanded_idx_mapping,
            cu_num_logits=batch.cu_num_logits,
            prompt_token_ids=runner.req_states.all_token_ids.gpu[state_index, :prompt_len]).items()})
    groups = []
    for gid, table in enumerate(tables):
        width = int(runner.block_tables.num_blocks.np[gid, state_index])
        size = runner.block_tables.kernel_block_sizes[gid]
        need(runner.block_tables.blocks_per_kv_block[gid] == 1, 'V2 split KV blocks unsupported')
        groups.append(dict(kv_cache_block_size=size, block_table=device_snapshot(table[0, :width]),
                           slot_mapping=device_snapshot(slots[gid, :count])))
    frame['_groups'] = groups
    frames = getattr(_state, 'frames', None)
    need(isinstance(frames, dict) and id(batch) not in frames and len(frames) < 2,
         'V2 pending forward ambiguity/startup hook missing')
    frames[id(batch)] = frame
    _state.forward = frame
    _local.production_attention_capture = frame['dispatch']


def forward_done(config, runner, batch, dummy):
    from .observer import _local
    _local.production_attention_capture = None
    frame = getattr(_state, 'forward', None)
    _state.forward = None
    if frame is not None:
        need(frame['_batch'] is batch and not dummy, 'V2 forward completion mismatch')
        need(len(frame['dispatch']['calls']) == 28, 'V2 forward production routing unavailable')
        need(('graph_reference' in frame) is (frame['descriptor']['cg_mode'] == 'FULL'),
             'V2 full graph lacks exact captured dispatch')
        frame['forward_complete'] = True


def raw(config, runner, batch, logits, grammar):
    from .observer import device_snapshot
    frame = getattr(_state, 'frames', {}).get(id(batch))
    _state.sample = frame
    if frame is None:
        return
    need(frame['_batch'] is batch and frame.get('forward_complete') is True
         and grammar is None and logits.ndim == 2 and logits.shape[0] == frame['draft_count'] + 1,
         'V2 raw logits/forward linkage mismatch')
    need(float(runner.sampler.sampling_states.temperature.np[frame['state_index']]) == 0,
         'V2 non-greedy sampling unsupported')
    need('raw' not in frame['_tensors'], 'duplicate V2 raw logits')
    frame['_tensors']['raw'] = device_snapshot(logits)


def processed(config, logits, mapping, positions):
    from .observer import device_snapshot
    frame = getattr(_state, 'sample', None)
    if frame is None:
        return
    need('processed' not in frame['_tensors'] and logits.ndim == 2
         and logits.shape[0] == frame['draft_count'] + 1, 'V2 processed logits ambiguous/chunked')
    frame['_tensors'].update(processed=device_snapshot(logits),
        processed_mapping=device_snapshot(mapping), processed_positions=device_snapshot(positions))


def selected(config, runner, batch, output):
    from .observer import ints, snapshot
    frame = getattr(_state, 'frames', {}).pop(id(batch), None)
    _state.sample = None
    if frame is None:
        return
    need(frame['_batch'] is batch and 'processed' in frame['_tensors'], 'V2 sampling provenance incomplete')
    sampled_count = ints(output.num_sampled)
    rejected = ints(output.num_rejected)
    need(len(sampled_count) == len(rejected) == 1 and 0 < sampled_count[0] <= frame['draft_count'] + 1,
         'V2 selected span ambiguous')
    tokens = ints(output.sampled_token_ids)
    need(len(tokens) == 1, 'V2 sampler output ambiguous')
    frame['selected'] = tokens[0][:sampled_count[0]]
    need(all(type(x) is int and x >= 0 for x in frame['selected'])
         and rejected[0] == frame['draft_count'] + 1 - sampled_count[0], 'V2 rejection accounting drift')
    tensors = frame.pop('_tensors')
    vectors = {name: ints(value) for name, value in tensors.items() if name not in ('raw', 'processed')}
    frame.update(vectors)
    positions = frame['positions']
    indices = frame['logits_indices']
    need(len(indices) == frame['draft_count'] + 1 and len(set(indices)) == len(indices)
         and all(type(i) is int and 0 <= i < len(positions) for i in indices), 'V2 row indices invalid')
    generated = [positions[i] - frame['prompt_len'] + 1 for i in indices]
    need(all(b == a + 1 for a, b in zip(generated, generated[1:])), 'V2 noncontiguous generated rows')
    need(frame['processed_positions'] == [positions[i] for i in indices]
         and frame['processed_mapping'] == frame['expanded_idx_mapping'], 'V2 processed row linkage mismatch')
    frame['rows'] = [dict(row=j, flat_index=i, position=positions[i], generated_index=generated[j],
        role=('ordinary' if frame['draft_count'] == 0 else 'bonus' if j == len(indices) - 1 else 'draft_verification'))
        for j, i in enumerate(indices)]
    frame['groups'] = [dict(g, block_table=ints(g['block_table']), slot_mapping=ints(g['slot_mapping']))
                       for g in frame.pop('_groups')]
    frame.pop('_batch')
    frame.pop('_manager')
    frame['active_query'] = list(map(list, zip(positions, frame['input_ids'])))
    boundary = [r for r in frame['rows'][:sampled_count[0]] if r['generated_index'] == 81]
    if not boundary:
        return
    need(not list(Path(config['directory']).glob('*.v2-forward.json')), 'duplicate emitted V2 index-81 boundary')
    row = boundary[0]['row']
    pending = {}
    frame['scores'] = {str(row): {name: snapshot(tensors[name][row], config['directory'],
        f'{frame["key"]}-r{row}-{name}.f32', pending) for name in ('raw', 'processed')}}
    for name, value in frame['scores'][str(row)].items():
        value.update(capture_point='v2_pre_grammar' if name == 'raw' else 'v2_pre_greedy_or_rejection', source_row=row)
    for name, data in pending.items():
        write_new(Path(config['directory']) / name, data)
    save(Path(config['directory']) / (frame['key'] + '.v2-forward.json'), frame)
    event(config['directory'], 'v2_production_attention_dispatch', key=frame['key'],
          request_id=frame['request_id'], scheduler_step=frame['scheduler_step'], generated_index=81,
          execution_sha=frame['execution_sha'], lane=frame['lane'], dispatch=frame['dispatch'],
          graph_reference=frame.get('graph_reference'))


HANDLERS = {name: globals()[name] for name in ('initialized', 'graph_inputs', 'graph_begin', 'graph_end', 'graph_replay',
                                             'prepared', 'forward_done', 'raw', 'processed', 'selected')}


def observe(stage, *args):
    config_path = os.environ.get('M203_TARGETED_CONFIG')
    if not config_path:
        return
    try:
        config = load(config_path)
        need(canonical_observation(config), 'V2 hook outside reviewed W3/006@81 diagnostic')
        observation_spec(config)
        HANDLERS[stage](config, *args)
    except Exception as exc:
        from .observer import _local
        _local.production_attention_capture = None
        _state.forward = _state.sample = _state.graph = None
        _state.frames = {}
        try:
            config = load(config_path)
            event(config['directory'], 'observer_error', stage='v2_' + stage,
                  error=f'{type(exc).__name__}: {exc}')
        except Exception:
            import sys
            print('M203_TARGETED_OBSERVER_ERROR_UNSAVED', file=sys.stderr, flush=True)
