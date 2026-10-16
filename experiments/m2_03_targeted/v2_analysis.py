"""CPU validation of the single reviewed V2 boundary and exact graph linkage."""
import re
from pathlib import Path
from . import canonical
from .common import load, need, sha, fingerprint, observation_spec, v2_observation_profile


def graph_trace(root, reference):
    need(isinstance(reference, dict) and set(reference) == {'file', 'sha256'}, 'V2 graph reference missing')
    rel = reference['file']
    need(re.fullmatch(r'graphs/[0-9a-f]{32}\.json', rel) is not None, 'unsafe V2 graph reference')
    raw = (root / rel).read_bytes()
    need(sha(raw) == reference['sha256'], 'V2 graph trace checksum drift')
    return load(root / rel)


def dispatch_proof(root, frame, events):
    from .observer import PRODUCTION_DISPATCH_FIELDS
    from .v2_observer import same_buffers
    dispatch = frame['dispatch']
    need(dispatch['force_3d'] is False and dispatch['expected_calls'] == len(dispatch['calls']) == 28,
         'V2 production dispatch incomplete/intervened')
    matches = [e for e in events if e['kind'] == 'v2_production_attention_dispatch']
    need(len(matches) == 1, 'V2 dispatch event missing/duplicate')
    event = matches[0]
    for name in ('key', 'request_id', 'scheduler_step', 'execution_sha', 'lane', 'generated_index', 'dispatch'):
        need(event[name] == frame[name], 'V2 attention/request/forward event linkage mismatch')
    desc = frame['descriptor']
    mode = desc['cg_mode']
    need(mode in ('FULL', 'PIECEWISE', 'NONE') and desc['num_active_loras'] == 0
         and desc['num_ubatches'] == 1, 'V2 unsupported execution descriptor')
    reference = frame.get('graph_reference')
    need(event['graph_reference'] == reference, 'V2 event graph reference drift')
    if mode == 'FULL':
        trace = graph_trace(root, reference)
        need(trace['schema'] == 'm203.targeted.v2.graph.v1'
             and trace['execution_sha'] == frame['execution_sha'] and trace['lane'] == frame['lane']
             and trace['descriptor'] == desc and trace['dispatch'] == dispatch
             and same_buffers(trace['buffer_signature'], frame['buffer_signature'])
             and trace['key'] == dispatch['key'], 'V2 replay/captured production graph linkage mismatch')
    else:
        need(reference is None and dispatch['key'] == frame['key'], 'V2 live dispatch mixed with graph evidence')
    calls = dispatch['calls']
    booleans = {'softmax_segm_output_present', 'softmax_segm_max_present', 'softmax_segm_expsum_present',
                'is_batch_invariant', 'use_3d', 'short_query_3d', 'guards_active', 'force_3d'}
    metadata = trace['attention_metadata'] if mode == 'FULL' else frame['attention_metadata']
    need(isinstance(metadata, dict) and len(metadata) == 28, 'V2 attention layer metadata missing')
    observed_shapes = {(m['max_query_len'], m['num_seqs'], m['num_actual_tokens']) for m in metadata.values()}
    need(len(observed_shapes) == 1, 'V2 target attention shapes ambiguous')
    attention_q, attention_reqs, attention_tokens = next(iter(observed_shapes))
    need(0 < attention_tokens <= frame['padded_token_count'], 'V2 actual attention tensor extent invalid')
    modes = []
    for ordinal, call in enumerate(calls):
        need(set(call) == set(PRODUCTION_DISPATCH_FIELDS) | {'key', 'ordinal'}
             and call['key'] == dispatch['key'] and call['ordinal'] == ordinal,
             'V2 attention call identity/order drift')
        for name in PRODUCTION_DISPATCH_FIELDS:
            value = call[name]
            need((name in booleans and type(value) is bool)
                 or (name == 'grid' and isinstance(value, list) and len(value) in (2, 3)
                     and all(type(x) is int and x > 0 for x in value))
                 or (name not in booleans | {'grid'} and type(value) is int and value > 0)
                 or (name in {'seq_threshold_3D', 'num_par_softmax_segments'} and value is None),
                 'V2 invalid attention metadata: ' + name)
        q, n = call['max_seqlen_q'], call['num_seqs']
        need((q, n) == (attention_q, attention_reqs), 'V2 captured/live attention shape mismatch')
        need(call['force_3d'] is False, 'V2 attention intervention forbidden')
        if mode == 'FULL':
            bound = desc['uniform_token_count'] or desc['max_query_len']
            need(n == desc['num_reqs'] and (bound is None or q == bound)
                 and q >= frame['scheduled_tokens'], 'V2 captured query/descriptor mismatch')
        else:
            need(q >= frame['scheduled_tokens'] and attention_tokens == frame['scheduled_tokens'],
                 'V2 live metadata/attention query mismatch')
        available = (call['seq_threshold_3D'] is not None and n <= call['seq_threshold_3D']
            and call['num_par_softmax_segments'] is not None
            and all(call[k] for k in ('softmax_segm_output_present', 'softmax_segm_max_present',
                                      'softmax_segm_expsum_present')) and not call['is_batch_invariant'])
        short = call['short_query_3d']
        need(not short or (q == 3 and n == 1 and attention_tokens == 3 and available),
             'V2 natural short-query admission mismatch')
        need(call['guards_active'] is short and call['use_3d'] is ((q == 1 and available) or short),
             'V2 natural production routing/guard mismatch')
        segments = call['num_par_softmax_segments'] if call['use_3d'] else 1
        need(call['num_segments'] == segments and call['block_size'] in {
            g['kv_cache_block_size'] for g in frame['groups']}
             and call['grid'][0] == attention_tokens // call['BLOCK_Q'] + n
             and len(call['grid']) == (3 if call['use_3d'] else 2)
             and (not call['use_3d'] or call['grid'][2] == segments), 'V2 production grid/slot mismatch')
        modes.append('3D' if call['use_3d'] else '2D')
    return dict(event=event, mode=modes[0] if len(set(modes)) == 1 else 'MIXED', force_3d=False,
                graph_reference=reference, logical_runtime_q=frame['scheduled_tokens'],
                captured_graph_q_shape=attention_tokens if mode == 'FULL' else None,
                max_seqlen_q=[c['max_seqlen_q'] for c in calls])


def capture_v2(directory, lane, execution, request_id, prompt, ids):
    from .analysis import vector
    config = load(directory / 'observer.json')
    need(v2_observation_profile(config), 'V2 analyzer outside reviewed observer profile')
    run = load(directory.parent / 'run.json')
    need(v2_observation_profile(run) and observation_spec(run) == observation_spec(config),
         'V2 analyzer outside reviewed run profile')
    root = directory / 'capture'
    identity = load(root / 'runtime_identity.json')
    canonical.verify_runner(identity, config)
    need(identity['model_runner'] == canonical.V2_RUNNER and identity['use_v2_model_runner'] is True,
         'V2 capture with unsupported runtime identity')
    events = [load(p) for p in (root / 'events').glob('*.json')]
    need(not any(e['kind'] == 'observer_error' or e['kind'].startswith(('kv_recompute', 'r14_')) for e in events)
         and not list(root.rglob('*.pending-*'))
         and 'M203_TARGETED_OBSERVER_ERROR_UNSAVED' not in (directory / 'server.log').read_text(errors='replace'),
         'V2 observer error/intervention/incomplete capture')
    paths = list(root.glob('*.v2-forward.json'))
    need(len(paths) == 1 and not list(root.glob('*.forward.json')), 'V2 boundary missing/duplicate/mixed with V1')
    frame = load(paths[0])
    need(frame['schema'] == 'm203.targeted.v2.forward.v1' and frame['execution_sha'] == execution
         and frame['lane'] == lane and frame['generated_index'] == 81
         and frame['observation_spec'] == observation_spec(config)
         and dict(frame['runtime_identity'], execution_sha=execution, lane=lane) == identity
         and frame['prefix_cache_enabled'] is True and frame['async_scheduling'] is True
         and frame['resolved_cudagraph_mode'] == 'FULL_AND_PIECEWISE', 'V2 canonical runtime/capture drift')
    need(re.fullmatch(re.escape(request_id) + r'(?:-[0-9a-f]{8})?', frame['request_id']) is not None,
         'V2 request linkage mismatch')
    need(frame['observation_scheduler_provenance'] == dict(state='OBSERVATION_ONLY',
         scheduler_step=frame['scheduler_step']), 'V2 scheduler/forward linkage missing')
    count, drafts = frame['scheduled_tokens'], frame['draft_count']
    need(type(count) is int and count == drafts + 1 and 1 <= count <= 5
         and frame['padded_token_count'] == frame['descriptor']['num_tokens'] >= count,
         'V2 active/padded query or draft mismatch')
    need(frame['prompt_len'] == len(prompt) and frame['prompt_token_ids'] == prompt, 'V2 prompt linkage drift')
    positions = frame['positions']
    need(len(positions) == count and all(type(p) is int and p >= 0 for p in positions)
         and positions == list(range(positions[0], positions[0] + count))
         and frame['active_query'] == list(map(list, zip(positions, frame['input_ids'])))
         and len(frame['input_ids']) == count, 'V2 active query incomplete')
    qsl, seq = frame['query_start_loc'], frame['seq_lens']
    padded_reqs = frame['descriptor']['num_reqs'] or 1
    need(qsl == [0] + [count] * padded_reqs and len(seq) == padded_reqs
         and seq[0] == positions[0] + count
         and frame['idx_mapping'] == [frame['state_index']]
         and frame['expanded_idx_mapping'] == [frame['state_index']] * (drafts + 1)
         and frame['processed_mapping'] == frame['expanded_idx_mapping']
         and frame['cu_num_logits'] == [0, drafts + 1]
         and frame['logits_indices'] == list(range(count))
         and frame['processed_positions'] == positions, 'V2 GPU row/query/state-slot linkage mismatch')
    rows = frame['rows']
    need(len(rows) == count and all(r == dict(row=j, flat_index=j, position=positions[j],
        generated_index=positions[j] - len(prompt) + 1,
        role='ordinary' if drafts == 0 else 'bonus' if j == count - 1 else 'draft_verification')
        for j, r in enumerate(rows)), 'V2 logits generated-index mapping invalid')
    selected = frame['selected']
    start = rows[0]['generated_index']
    need(0 < len(selected) <= count and start >= 0 and selected == ids[start:start + len(selected)]
         and start <= 81 < start + len(selected), 'V2 index 81 not emitted exactly')
    prefix = prompt + ids[:start + len(selected) - 1]
    last_position = rows[len(selected) - 1]['position']
    need(all(p < len(prefix) and prefix[p] == token for p, token in frame['active_query'] if p <= last_position),
         'V2 active query/API conditioning mismatch')
    need(bool(frame['groups']), 'V2 KV slot equivalent missing')
    for g in frame['groups']:
        size, table = g['kv_cache_block_size'], g['block_table']
        need(type(size) is int and size > 0 and all(type(b) is int and b >= 0 for b in table)
             and all(p // size < len(table) for p in positions)
             and g['slot_mapping'] == [table[p // size] * size + p % size for p in positions],
             'V2 active query/KV slot linkage mismatch')
    matching = [e for e in events if e.get('request_id') == frame['request_id']
                and e.get('scheduler_step') == frame['scheduler_step']]
    before = [e for e in matching if e['kind'] == 'account_before']
    after = [e for e in matching if e['kind'] == 'account_after']
    need(len(before) == len(after) == 1, 'V2 scheduler accounting ambiguous')
    b, a = before[0], after[0]
    need(b['output_count'] == start and a['output_count'] == start + len(selected)
         and a['num_output_placeholders'] == b['num_output_placeholders'] - len(selected),
         'V2 scheduler emitted span mismatch')
    for ev in (b, a):
        need(ev['output_token_ids'] == selected and ev['stale'] is False and ev['invalid_spec_tokens'] == 0
             and ev['scheduled_tokens'] == count and len(ev['scheduled_spec_token_ids']) == drafts,
             'V2 scheduler/forward/API linkage mismatch')
    row = rows[81 - start]
    scores = frame['scores'][str(row['row'])]
    need(set(frame['scores']) == {str(row['row'])} and set(scores) == {'raw', 'processed'},
         'V2 raw/processed logits missing')
    for name, point in [('raw', 'v2_pre_grammar'), ('processed', 'v2_pre_greedy_or_rejection')]:
        need(scores[name]['capture_point'] == point and scores[name]['source_row'] == row['row'],
             'V2 logits hook/row provenance drift')
    need(len(vector(root, scores['raw'])) == len(vector(root, scores['processed'])), 'V2 logits shape drift')
    dispatch = dispatch_proof(root, frame, events)
    return dict(record=frame['key'], predecessor=None, row=row, scores=scores,
        selected_token=selected[row['row']], prefix_sha256=fingerprint(prompt + ids[:81]),
        decoding_path='ordinary' if drafts == 0 else 'speculative',
        active_query=frame['active_query'], active_query_token_count=count,
        logical_runtime_q=count, captured_graph_q_shape=dispatch['captured_graph_q_shape'],
        max_seqlen_q=dispatch['max_seqlen_q'],
        preparation={frame['key']: frame}, accounting=dict(before=b, after=a),
        kv_recompute_ready=None, kv_recompute_refill_skipped=None, attention_dispatch=None,
        production_attention_dispatch=dispatch)
