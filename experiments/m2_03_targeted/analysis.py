"""CPU validation of targeted capture. Invalid data never becomes a causal claim."""
import array
import re
import sys
from pathlib import Path

from .common import CASE, TARGET, RECONSTRUCTION_TARGET, OBSERVATION_TARGET, LANES, REQUEST_HASH, HISTORICAL, HISTORICAL_EXECUTION, Invalid, decode, fingerprint, load, need, sha
from .observer import (map_rows, score_stats, valid_selected, observation_target,
                       DISPATCH_ARGUMENT_FIELDS, DISPATCH_FIELDS, PRODUCTION_DISPATCH_FIELDS)
from . import sources, canonical as canonical_contract
from .common import (observation_spec, transition_observation, observation_controls,
                     canonical_observation, canonical_observation_profile, v2_observation_profile)
from .lifecycle import verify_shutdown

STATES = ("REPRODUCED", "NOT_REPRODUCED", "BOUNDARY_NOT_COVERED", "INVALID_CAPTURE")


def configured_attention_proof(events, current, target):
    """Validate observed production decisions, including q=1 and 2D fallback."""
    matches = [e for e in events if e['kind'] == 'production_attention_dispatch']
    need(len(matches) == 1, 'configured attention event missing/duplicate')
    event = matches[0]
    prep = current['preparation']
    dispatch = prep['production_attention_dispatch']
    need(event['generated_index'] == target and event['expected_calls'] == dispatch['expected_calls'] == 28
         and len(event['calls']) == 28 and event['calls'] == dispatch['calls']
         and dispatch['force_3d'] is False and dispatch['key'] == current['key'],
         'configured attention completeness/boundary mismatch')
    for key in ('execution_sha', 'lane', 'request_id', 'key', 'scheduler_step', 'forward_sequence'):
        need(event[key] == current[key], 'configured attention event linkage')
    flags = {'softmax_segm_output_present', 'softmax_segm_max_present', 'softmax_segm_expsum_present',
             'is_batch_invariant', 'use_3d', 'short_query_3d', 'guards_active', 'force_3d'}
    for ordinal, call in enumerate(event['calls']):
        need(set(call) == set(PRODUCTION_DISPATCH_FIELDS) | {'key', 'ordinal'}, 'configured dispatch fields')
        for name in PRODUCTION_DISPATCH_FIELDS:
            value = call[name]
            need((name in flags and type(value) is bool)
                 or (name == 'grid' and type(value) is list and all(type(x) is int and x > 0 for x in value))
                 or (name not in flags | {'grid'} and type(value) is int and value > 0)
                 or (name in {'seq_threshold_3D', 'num_par_softmax_segments'} and value is None),
                 'configured attention metadata: ' + name)
        count = prep['scheduled_tokens']
        need(type(call['ordinal']) is int and call['ordinal'] == ordinal and call['key'] == current['key']
             and call['max_seqlen_q'] == count and call['num_seqs'] == 1 and call['force_3d'] is False,
             'configured attention query/intervention mismatch')
        available = (call['seq_threshold_3D'] is not None and call['seq_threshold_3D'] >= 1
                     and call['num_par_softmax_segments'] is not None
                     and all(call[k] for k in ('softmax_segm_output_present', 'softmax_segm_max_present',
                                               'softmax_segm_expsum_present')) and not call['is_batch_invariant'])
        short = call['short_query_3d']
        need(not short or (count == 3 and available), 'configured short-query admission mismatch')
        need(call['guards_active'] is short and call['use_3d'] is ((available and count == 1) or short),
             'configured production routing/guards mismatch')
        segments = call['num_par_softmax_segments'] if call['use_3d'] else 1
        need(call['num_segments'] == segments and len(call['grid']) == (3 if call['use_3d'] else 2)
             and (not call['use_3d'] or (call['grid'][2] == segments and segments & (segments - 1) == 0))
             and call['grid'][0] == count // call['BLOCK_Q'] + 1
             and call['block_size'] in {g['kv_cache_block_size'] for g in prep['groups']},
             'configured attention grid/segment mismatch')
    modes = ['3D' if call['use_3d'] else '2D' for call in event['calls']]
    return dict(event=event, mode=modes[0] if len(set(modes)) == 1 else 'MIXED', force_3d=False,
                production_patch_sha=sources.PRODUCTION_COMMIT)


def production_attention_proof(events, current, enabled):
    matches = [ev for ev in events if ev['kind'] == 'production_attention_dispatch']
    if not enabled:
        need(not matches and 'production_attention_dispatch' not in current['preparation'],
             'unexpected production attention observation')
        return None
    need(len(matches) == 1, 'production dispatch event missing/duplicate')
    event = matches[0]
    for name in ('execution_sha', 'lane', 'request_id', 'key', 'scheduler_step', 'forward_sequence'):
        need(event[name] == current[name], 'production dispatch linkage: ' + name)
    prep = current['preparation']
    dispatch = prep['production_attention_dispatch']
    need(event['generated_index'] == OBSERVATION_TARGET and current['lane'] == 'adaptive'
         and current['draft_count'] == 2 and current['decoding_path'] == 'speculative'
         and len(current['active_query']) == prep['scheduled_tokens'] == 3
         and prep['scheduled_draft_count'] == 2 and prep['query_start_loc'] == [0, 3],
         'production q=3 verification boundary not established')
    need(set(dispatch) == {'key', 'expected_calls', 'force_3d', 'calls'}
         and dispatch['key'] == current['key'] and dispatch['force_3d'] is False
         and type(event['expected_calls']) is int
         and event['expected_calls'] == dispatch['expected_calls'] == 28
         and event['calls'] == dispatch['calls'] and len(event['calls']) == 28,
         'production attention completeness/intervention mismatch')
    flags = {'softmax_segm_output_present', 'softmax_segm_max_present', 'softmax_segm_expsum_present',
             'is_batch_invariant', 'use_3d', 'short_query_3d', 'guards_active', 'force_3d'}
    for ordinal, call in enumerate(event['calls']):
        need(set(call) == set(PRODUCTION_DISPATCH_FIELDS) | {'key', 'ordinal'}, 'production dispatch fields')
        for name in PRODUCTION_DISPATCH_FIELDS:
            value = call[name]
            need((name in flags and type(value) is bool)
                 or (name == 'grid' and type(value) is list
                     and all(type(x) is int and x > 0 for x in value))
                 or (name not in flags | {'grid'} and type(value) is int and value > 0),
                 'production dispatch metadata: ' + name)
        need(type(call['ordinal']) is int and call['ordinal'] == ordinal and call['key'] == current['key']
             and call['max_seqlen_q'] == 3 and call['num_seqs'] == 1
             and call['short_query_3d'] is call['use_3d'] is call['guards_active'] is True
             and call['force_3d'] is call['is_batch_invariant'] is False
             and all(call[k] for k in ('softmax_segm_output_present', 'softmax_segm_max_present',
                                      'softmax_segm_expsum_present')),
             'production routing/guards/intervention mismatch')
        segments = call['num_par_softmax_segments']
        need(segments & (segments - 1) == 0 and call['num_segments'] == segments
             and len(call['grid']) == 3 and call['grid'][2] == segments
             and call['grid'][0] == 3 // call['BLOCK_Q'] + 1
             and call['block_size'] in {g['kv_cache_block_size'] for g in prep['groups']},
             'production grid/segment mismatch')
    return {'event': event, 'mode': '3D', 'force_3d': False, 'guards_active': True,
            'production_patch_sha': sources.PRODUCTION_COMMIT}


def attention_dispatch_proof(events, current, enabled, forced=False):
    """Validate actual launches, retaining the full ordered, possibly mixed pattern."""
    matches = [e for e in events if e['kind'] == 'attention_dispatch']
    if not enabled:
        need(not matches, 'unexpected R15 dispatch event')
        return None  # Preserved R13/R14 evidence is unknown, not invalid.
    need(len(matches) == 1, 'R15 dispatch event missing/duplicate')
    event = matches[0]
    need(event['generated_index'] == OBSERVATION_TARGET, 'R15 boundary drift')
    for name in ('execution_sha', 'lane', 'request_id', 'key', 'scheduler_step', 'forward_sequence'):
        need(event[name] == current[name], 'R15 dispatch linkage mismatch: ' + name)
    prep = current['preparation']
    dispatch = prep['attention_dispatch']
    legacy = not forced and 'force_3d' not in dispatch
    need(dispatch['key'] == current['key'] and event['calls'] == dispatch['calls']
         and event['expected_calls'] == dispatch['expected_calls']
         and (legacy or dispatch['force_3d'] is forced), 'R15/R16 preparation linkage mismatch')
    calls, expected = event['calls'], event['expected_calls']
    need(type(expected) is int and expected > 0 and type(calls) is list
         and len(calls) == expected, 'R15 incomplete ordered attention calls')
    if not legacy:
        need(type(dispatch['controls']) is list and len(dispatch['controls']) == len(calls),
             'R15/R16 control/launch count mismatch')
    flags = {'softmax_segm_output_present', 'softmax_segm_max_present',
             'softmax_segm_expsum_present', 'is_batch_invariant', 'use_3d',
             'natural_use_3d', 'force_3d', 'q3_correctness_guards'}
    nullable = {'seq_threshold_3D', 'num_par_softmax_segments'}
    for ordinal, call in enumerate(calls):
        legacy_call = legacy and set(call) == set(DISPATCH_ARGUMENT_FIELDS) | {'key', 'ordinal'}
        old_r15 = not forced and set(call) == (set(DISPATCH_FIELDS) - {'q3_correctness_guards'}) | {'key', 'ordinal'}
        need(legacy_call or old_r15 or set(call) == set(DISPATCH_FIELDS) | {'key', 'ordinal'},
             'R15 dispatch fields')
        need(type(call['ordinal']) is int and call['ordinal'] == ordinal
             and call['key'] == current['key'], 'R15 attention call order/key')
        checked = (dict(call, natural_use_3d=call['use_3d'], force_3d=False, q3_correctness_guards=False)
                   if legacy_call else call)
        if old_r15:
            checked = dict(call, q3_correctness_guards=False)
        if not legacy_call:
            need(dispatch['controls'][ordinal] == {
                'ordinal': ordinal, 'natural_use_3d': checked['natural_use_3d'],
                'force_3d': checked['force_3d']}, 'R15/R16 control/launch linkage')
        for name in DISPATCH_FIELDS:
            value = checked[name]
            if name in flags:
                need(type(value) is bool, 'R15 dispatch boolean: ' + name)
            elif name == 'grid':
                need(type(value) is list and all(type(x) is int and x > 0 for x in value), 'R15 grid')
            else:
                need((value is None and name in nullable)
                     or (type(value) is int and value == 0 and name == 'seq_threshold_3D')
                     or (type(value) is int and value > 0), 'R15 dispatch integer: ' + name)
        natural_use_3d = not (
            checked['seq_threshold_3D'] is None or checked['num_par_softmax_segments'] is None
            or not checked['softmax_segm_output_present'] or not checked['softmax_segm_max_present']
            or not checked['softmax_segm_expsum_present'] or checked['max_seqlen_q'] > 1
            or checked['num_seqs'] > checked['seq_threshold_3D'] or checked['is_batch_invariant'])
        need(checked['natural_use_3d'] == natural_use_3d
             and checked['force_3d'] is forced
             and checked['q3_correctness_guards'] is forced
             and checked['use_3d'] == (natural_use_3d or forced),
             'R15/R16 contradictory dispatch predicate/control')
        use_3d = checked['use_3d']
        segments = checked['num_par_softmax_segments'] if use_3d else 1
        grid = checked['grid']
        need(checked['num_seqs'] == 1 and checked['max_seqlen_q'] == prep['scheduled_tokens'],
             'R15 query metadata mismatch')
        need(len(grid) == (3 if use_3d else 2) and checked['num_segments'] == segments
             and (not use_3d or grid[2] == segments)
             and grid[0] == prep['scheduled_tokens'] // checked['BLOCK_Q'] + checked['num_seqs'],
             'R15 launch grid/segments mismatch')
        need(checked['BLOCK_M'] >= 16 and checked['BLOCK_M'] & (checked['BLOCK_M'] - 1) == 0
             and checked['BLOCK_Q'] <= checked['BLOCK_M']
             and checked['tile_size'] & (checked['tile_size'] - 1) == 0
             and checked['block_size'] in {g['kv_cache_block_size'] for g in prep['groups']},
             'R15 launch block/tile mismatch')
    pattern = ['3D' if call['use_3d'] else '2D' for call in calls]
    if forced:
        need(expected == len(calls) == 28 and all(not call['natural_use_3d']
             and call['force_3d'] and call['use_3d'] and call['q3_correctness_guards'] for call in calls),
             'R16 did not force all 28 naturally-2D target attention calls')
    return dict(event=event, modes=pattern, mode=pattern[0] if len(set(pattern)) == 1 else 'MIXED')


def r16_boundary_state(current, enabled):
    if not enabled:
        return
    expected_query = [[556, 47742], [557, 129831], [558, 128319]]
    prep = current['preparation']
    need(current['draft_count'] == 2 and current['decoding_path'] == 'speculative'
         and current['active_query'] == expected_query
         and prep['scheduled_tokens'] == 3 and prep['scheduled_draft_count'] == 2
         and prep['query_start_loc'] == [0, 3] and prep['seq_lens'] == [559]
         and prep['cpu_num_computed_tokens'] == prep['gpu_num_computed_tokens_after'] == [556]
         and all(group['slot_mapping'] == [572, 573, 574] for group in prep['groups']),
         'R16 did not preserve the proven R13 q=3 boundary state')


def r16_causal_result(capture):
    token = capture['selected_token']
    outcome = ('BOUNDARY_ARGMAX_CHANGED_TO_9765' if token == 9765 else
               'BOUNDARY_ARGMAX_REMAINS_128281' if token == 128281 else
               'UNEXPECTED_BOUNDARY_TOKEN')
    return {'intervention': 'natural_2D_to_forced_3D',
            'adaptive_q_len': capture['active_query_token_count'],
            'selected_token': token, 'outcome': outcome,
            'causal_conclusion': 'PENDING_R15_BASELINE_STATE_COMPARISON',
            'same_r15_prefix_verified': False,
            'same_r15_kv_values_verified': False,
            'q3_correctness_guards_enabled': all(call['q3_correctness_guards']
                for call in capture['attention_dispatch']['event']['calls']),
            'all_28_calls_forced_3D': capture['attention_dispatch']['mode'] == '3D'}


def validate_transition(previous, current, prompt, ids, target=OBSERVATION_TARGET):
    """Validate owned snapshots; never infer equality from optimistic CPU state."""
    need(current['previous_key'] == previous['key']
         and current['forward_sequence'] == previous['forward_sequence'] + 1
         and current['scheduler_step'] > previous['scheduler_step']
         and current['request_id'] == previous['request_id'], 'ambiguous predecessor linkage')

    def integer_list(value, size=None, minimum=0):
        need(isinstance(value, list) and bool(value)
             and all(type(x) is int and x >= minimum for x in value)
             and (size is None or len(value) == size), 'invalid preparation integer vector')
        return value

    for record in (previous, current):
        prep = record['preparation']
        need(prep['key'] == record['key'] and prep['request_id'] == record['request_id'],
             'preparation/forward linkage mismatch')
        query = record['active_query']
        need(prep['active_query'] == list(map(list, query)), 'prepared model input linkage mismatch')
        need(bool(query) and all(isinstance(x, (list, tuple)) and len(x) == 2
                                 and all(type(v) is int and v >= 0 for v in x) for x in query),
             'invalid active query')
        count = prep['scheduled_tokens']
        need(type(count) is int and count == len(query)
             and prep['scheduled_draft_count'] == record['draft_count'], 'scheduled span mismatch')
        after = integer_list(prep['gpu_num_computed_tokens_after'], 1)[0]
        cpu = integer_list(prep['cpu_num_computed_tokens'], 1)[0]
        before = integer_list(prep['gpu_num_computed_tokens_before'])
        prev = integer_list(prep['prev_positions'], 1, -1)[0]
        drafts = integer_list(prep['prev_num_draft_tokens'])
        mapping = prep['prev_req_id_to_index']
        need(isinstance(mapping, dict) and prev == mapping.get(record['request_id'], -1),
             'previous request-row mapping mismatch')
        valid = prep['valid_sampled_token_count_gpu']
        if record['lane'] == 'adaptive':
            integer_list(valid)
        need(type(prep['correction_enabled']) is bool, 'unknown correction branch')
        if prep['correction_enabled']:
            integer_list(valid)
            j = max(prev, 0)
            need(j < len(before) and j < len(drafts) and j < len(valid), 'correction gather out of bounds')
            expected = before[j] + valid[j] if prev >= 0 and drafts[j] > 0 else cpu
        else:
            need(record['lane'] == 'vanilla', 'adaptive correction inputs/branch missing')
            expected = cpu
        need(after == expected, 'incorrect GPU correction accounting')
        need([p for p, _ in query] == list(range(after, after + count))
             and integer_list(prep['query_start_loc'], 2) == [0, count]
             and integer_list(prep['seq_lens'], 1) == [after + count],
             'position/seq_len/query span mismatch')
        rows = record['rows']
        expected_rows = map_rows(len(prompt), [p for p, _ in query],
                                 [r['flat_index'] for r in rows], record['draft_count'],
                                 record['spec_target_rows'], record['spec_bonus_rows'])
        need(record['decoding_path'] in ('ordinary', 'speculative'), 'unknown decoding path')
        if record['decoding_path'] == 'ordinary':
            need(record['draft_count'] == 0 and len(rows) == 1, 'ordinary row mismatch')
            expected_rows[0]['role'] = 'ordinary'
        need(rows == expected_rows, 'logits-row mapping mismatch')
        selected = valid_selected(record['selected'])
        need(0 < len(selected) <= len(rows)
             and record['accepted_drafts'] == len(selected) - 1
             and record['rejected_drafts'] == record['draft_count'] - len(selected) + 1,
             'selected verification accounting mismatch')
        start = rows[0]['generated_index']
        need(start >= 0 and selected == ids[start:start + len(selected)], 'selected/API linkage mismatch')
        prefix = prompt + ids[:start + len(selected) - 1]
        last = rows[len(selected) - 1]['position']
        need(all(p < len(prefix) and prefix[p] == token for p, token in query if p <= last),
             'model query/API conditioning mismatch')
        groups = prep['groups']
        need(isinstance(groups, list) and bool(groups), 'missing active block/slot data')
        for group in groups:
            need(group['request_id'] == record['request_id'] and group['row_index'] == 0,
                 'block/slot active request linkage mismatch')
            size, kv_size, split = group['block_size'], group['kv_cache_block_size'], group['blocks_per_kv_block']
            need(all(type(x) is int and x > 0 for x in (size, kv_size, split))
                 and split == 1 and kv_size == size
                 and group['dcp_world_size'] == group['pcp_world_size'] == 1,
                 'unsupported block size/parallel metadata')
            table = integer_list(group['block_table'])
            slots = integer_list(group['slot_mapping'], count)
            need(all(p // size < len(table) for p, _ in query), 'active block missing')
            need(slots == [table[p // size] * size + p % size for p, _ in query],
                 'incorrect slot mapping linkage')

    prep = current['preparation']
    previous_start = previous['rows'][0]['generated_index']
    current_start = current['rows'][0]['generated_index']
    need(current_start == previous_start + len(previous['selected']), 'predecessor emitted span discontinuity')
    need(prep['prev_positions'] == [0] and prep['prev_req_id_to_index'] == {current['request_id']: 0},
         'predecessor request-row linkage mismatch')
    need(prep['gpu_num_computed_tokens_before'] == previous['preparation']['gpu_num_computed_tokens_after'],
         'previous GPU count snapshot mismatch')
    if current['lane'] == 'adaptive':
        need(prep['prev_num_draft_tokens'] == [previous['draft_count']]
             and prep['valid_sampled_token_count_gpu'] == [len(previous['selected'])],
             'previous selected/count correction linkage mismatch')
    need(current_start <= target < current_start + len(current['selected']),
         'current forward does not cover observation target')


def capture_transition(directory, lane, execution, request_id, prompt, ids):
    config = load(directory / 'observer.json')
    target = observation_target(config)
    root = directory / 'capture'
    events = [load(p) for p in (root / 'events').glob('*.json')]
    need(not any(e['kind'] == 'observer_error' for e in events)
         and 'M203_TARGETED_OBSERVER_ERROR_UNSAVED' not in (directory / 'server.log').read_text(errors='replace')
         and not list(root.rglob('*.pending-*')), 'observer error/incomplete capture')
    refill = refill_proof(directory, lane, request_id, len(prompt))
    ready = ready_proof(directory, lane, request_id, len(prompt))
    records = [load(p) for p in root.glob('*.forward.json')]
    need(len(records) == 2 and len({r['key'] for r in records}) == 2, 'expected exactly predecessor/current forwards')
    records.sort(key=lambda r: r['forward_sequence'])
    previous, current = records
    accounting = {}
    for record in records:
        need(record['schema'] == 'm203.targeted.transition.v1' and record['lane'] == lane
             and record['execution_sha'] == execution and record['prompt_token_ids'] == prompt
             and type(record['prefix_cache_enabled']) is bool
             and (canonical_observation(config) or record['prefix_cache_enabled'] is False)
             and record['async_scheduling'] is True,
             'transition provenance drift')
        need(re.fullmatch(re.escape(request_id) + r'(?:-[0-9a-f]{8})?', record['request_id']),
             'transition request linkage mismatch')
        need(record['preparation']['scheduler_step'] == record['scheduler_step']
             and record['preparation']['forward_sequence'] == record['forward_sequence'],
             'preparation sequence linkage mismatch')
        if config.get('observation_only'):
            need(record.get('observation_spec') == observation_spec(config)
                 and record.get('observation_scheduler_provenance') == dict(
                state='OBSERVATION_ONLY', scheduler_step=record['scheduler_step']), 'observation scheduler linkage')
        if canonical_observation_profile(config):
            runtime_identity = load(root / 'runtime_identity.json')
            canonical_contract.verify_runner(runtime_identity, config)
            need(dict(record['preparation'].get('runtime_identity', {}),
                      execution_sha=config['execution_sha'], lane=lane) == runtime_identity,
                 'resolved runner/capture identity mismatch')
            need(record['preparation'].get('resolved_cudagraph_mode') in (
                'NONE', 'PIECEWISE', 'FULL', 'FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE'),
                 'resolved runtime CUDA-graph mode missing')
        if ready is not None:
            need(scheduler_provenance(record, ready) == 'POST_READY', 'transition precedes reconstruction')
        matching = [ev for ev in events if ev.get('request_id') == record['request_id']
                    and ev.get('scheduler_step') == record['scheduler_step']]
        before = [ev for ev in matching if ev['kind'] == 'account_before']
        after = [ev for ev in matching if ev['kind'] == 'account_after']
        need(len(before) == len(after) == 1, 'ambiguous scheduler/forward linkage')
        b, a = before[0], after[0]
        start = record['rows'][0]['generated_index']
        selected = record['selected']
        for ev in (b, a):
            need(ev['stale'] is False and ev['invalid_spec_tokens'] == 0
                 and ev['scheduled_tokens'] == record['preparation']['scheduled_tokens']
                 and len(ev['scheduled_spec_token_ids']) == record['draft_count'],
                 'scheduler/current forward mismatch')
        need(b['output_count'] == start and b['output_token_ids'] == selected
             and a['output_count'] == start + len(selected) and a['output_token_ids'] == selected
             and selected == ids[start:a['output_count']], 'scheduler/API emitted span mismatch')
        need(a['num_output_placeholders'] == b['num_output_placeholders'] - len(selected),
             'scheduler placeholder accounting mismatch')
        accounting[record['key']] = dict(before=b, after=a, emitted_span=[start, a['output_count']])
    validate_transition(previous, current, prompt, ids, target)
    config = load(directory / 'observer.json')
    production = sources.profile_options(config)
    r15 = config.get('r15_attention_dispatch', False)
    need(type(r15) is bool and (not r15 or lane == 'adaptive'), 'R15 observer gate drift')
    r16 = config.get('r16_force_attention_3d', False)
    need(type(r16) is bool and (not r16 or r15 and lane == 'adaptive'), 'R16 observer gate drift')
    dispatch_proof = attention_dispatch_proof(events, current, r15, r16)
    production_proof = (configured_attention_proof(events, current, target) if config.get('observation_only')
                        else production_attention_proof(events, current, bool(production)))
    r16_boundary_state(current, r16)
    if config.get('r14_single_token_boundary') is not None:
        need(config['r14_single_token_boundary'] == OBSERVATION_TARGET and lane == 'adaptive',
             'R14 observer gate mismatch')
        enabled = [ev for ev in events if ev['kind'] == 'r14_control_enabled']
        fired = [ev for ev in events if ev['kind'] == 'r14_current_drafts_suppressed']
        need(len(enabled) == len(fired) == 1, 'R14 control missing/repeated')
        proof = fired[0]
        need(all(ev['control_enabled'] is True and ev['execution_sha'] == execution
                 and ev['request_id'] == current['request_id']
                 and ev['target_generated_index'] == OBSERVATION_TARGET for ev in enabled + fired),
             'R14 control provenance mismatch')
        need(proof['fire_count'] == 1 and proof['pre_control_draft_count'] in range(1, 5)
             and proof['post_control_draft_count'] == 0
             and proof['scheduler_step'] == current['scheduler_step']
             and proof['num_output_placeholders'] == proof['num_in_flight_tokens'] in (0, 1)
             and proof['num_output_tokens'] + proof['num_output_placeholders'] == OBSERVATION_TARGET
             and proof['num_prompt_tokens'] == len(prompt)
             and proof['num_computed_tokens'] == len(prompt) + OBSERVATION_TARGET - 1
             and proof['num_tokens'] == len(prompt) + proof['num_output_tokens']
             and proof['num_tokens_with_spec'] == proof['num_tokens'] + proof['pre_control_draft_count']
             and proof['num_stale_output_tokens'] == 0
             and current['preparation']['cpu_num_computed_tokens'] == [proof['num_computed_tokens']]
             and current['draft_count'] == 0 and current['decoding_path'] == 'ordinary'
             and current['preparation']['scheduled_tokens'] == 1, 'R14 ordinary boundary/control mismatch')
    j = target - current['rows'][0]['generated_index']
    row = current['rows'][j]
    scores = current['scores'][str(j)]
    need(set(current['scores']) == {str(j)} and not previous['scores']
         and set(scores) == {'raw', 'processed'}, 'unexpected/missing boundary scores')
    need(scores['raw']['capture_point'] == 'runner_pre_grammar' and scores['raw']['source_row'] == j,
         'raw logits-row linkage mismatch')
    verify = row['role'] == 'draft_verification'
    need(scores['processed']['capture_point'] == ('rejection_pre_acceptance' if verify else 'sampler_pre_greedy')
         and scores['processed']['source_row'] == (j if verify else 0), 'processed logits-row linkage mismatch')
    need(len(vector(root, scores['raw'])) == len(vector(root, scores['processed'])), 'vocabulary shape drift')
    return dict(record=current['key'], predecessor=previous['key'], row=row, scores=scores,
                selected_token=current['selected'][j], prefix_sha256=fingerprint(prompt + ids[:target]),
                decoding_path=current['decoding_path'], active_query=current['active_query'],
                active_query_token_count=len(current['active_query']), accounting=accounting,
                preparation={r['key']: r['preparation'] for r in records},
                kv_recompute_ready=ready, kv_recompute_refill_skipped=refill,
                attention_dispatch=dispatch_proof,
                **({'production_attention_dispatch': production_proof} if production else {}))


def provenance(root, lanes=LANES):
    config = load(root / "run.json")
    spec = observation_spec(config)
    execution = config["diagnostic_execution_sha"]
    need(re.fullmatch(r"[0-9a-f]{40}", execution) and execution != HISTORICAL_EXECUTION, "execution SHA invalid/conflated")
    need(config["schema"] == "m203.targeted.run.v1" and config["case"] == spec['case']
         and config["target"] == observation_target(config)
         and config["historical_checkpoint_sha"] == HISTORICAL
         and config["historical_execution_sha"] == HISTORICAL_EXECUTION
         and config["request_sha256"] == spec['request_sha256'], "run provenance drift")
    if 'observation_target' in config:
        need(config['observation_target'] == (spec['target'] if config.get('observation_only') else OBSERVATION_TARGET)
             and config['reconstruction_target'] == RECONSTRUCTION_TARGET,
             'observation/reconstruction target drift')
    preflight, installed = load(root / "preflight.json"), load(root / "installed.json")
    if config.get('observation_only'):
        need(preflight.get('observation_spec') == spec, 'observation preflight selection drift')
    need(preflight["execution_sha"] == execution and preflight["runtime"] == installed["runtime"], "runtime provenance drift")
    r15 = config.get('r15_attention_dispatch', False)
    need(type(r15) is bool, 'R15 run gate drift')
    r16 = config.get('r16_force_attention_3d', False)
    need(type(r16) is bool and (not r16 or r15), 'R16 run gate drift')
    production = sources.profile_options(config)
    base = sources.base_hashes(**production) if production else sources.manifest()['sha256']
    expected = {k: sha(sources.transform(k, v, canonical=production.get('canonical', False),
                                       v2=production.get('v2', False)))
                for k, v in sources.base_sources(**production).items()
                if k in sources.patch_names(v2=production.get('v2', False))}
    profiles = [(base, expected)]
    if r15 and not r16:
        historical_r15 = dict(expected)
        historical_r15[sources.ATTENTION_SOURCE] = sources.R15_ATTENTION_PATCHED_SHA256
        profiles.append((base, historical_r15))
    if not r15 and not production:
        # Only the exact pre-R15 profile is accepted, never an arbitrary subset.
        profiles.append(({k: v for k, v in base.items() if k != sources.ATTENTION_SOURCE},
                         {k: v for k, v in expected.items() if k != sources.ATTENTION_SOURCE}))
    need(any(preflight['source_base_sha256'] == b and installed['patched_sha256'] == p
             for b, p in profiles), 'base/observer source provenance drift')
    policy = config["policy"]
    control = policy.get("diagnostic_control", {})
    r14 = config.get('r14_single_token_boundary')
    need(r14 is None or type(r14) is int and r14 == OBSERVATION_TARGET, 'R14 run gate drift')
    need(not (r14 is not None and r16), 'R14/R16 control conflict')
    if 'observation_target' in config:
        need(control == (observation_controls(config['case'])
                         if config.get('observation_only') else {
                         'vanilla_cudagraph_mode': 'PIECEWISE', 'adaptive_cudagraph_mode': 'NONE',
                         'vanilla_kv_recompute_output_count': RECONSTRUCTION_TARGET,
                         'adaptive_kv_recompute_output_count': RECONSTRUCTION_TARGET}), 'r12 controls changed')
    need(set(control) <= {"vanilla_cudagraph_mode", "adaptive_cudagraph_mode",
                          "vanilla_kv_recompute_output_count", "adaptive_kv_recompute_output_count"}
         and control.get("vanilla_cudagraph_mode") in (None, "PIECEWISE")
         and control.get("adaptive_cudagraph_mode") in (None, "NONE")
         and control.get("vanilla_kv_recompute_output_count") in (None, TARGET)
         and control.get("adaptive_kv_recompute_output_count") in (None, TARGET),
         "diagnostic control unreviewed")
    need(preflight["models"] == policy["models"] and all(preflight["runtime"][k] == v for k, v in policy["runtime"].items()), "model/runtime policy mismatch")
    need(preflight["runtime"]["vllm"] == "0.30.0" and preflight["runtime"]["gpu"] == "Tesla T4", "wrong pinned runtime")
    for lane in lanes:
        launch = load(root / lane / "launch.json")
        need(launch["execution_sha"] == execution and launch["argv"] == preflight["launcher_argv"][lane], "launch provenance drift")
        argv, env = launch["argv"], launch["environment"]
        for flag, value in {"--model": policy["models"]["target"]["snapshot"],
                            "--host": "0.0.0.0" if canonical_observation(config) else "127.0.0.1",
                            "--port": "8001" if lane == "vanilla" else "8003", "--dtype": "half",
                            "--quantization": "awq", "--generation-config": "vllm",
                            "--gpu-memory-utilization": "0.85", "--max-model-len": "4096", "--max-num-seqs": "32",
                            "--served-model-name": "qwen2.5-7b-" + lane}.items():
            need(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, "launcher setting drift: " + flag)
        need(("--no-enable-prefix-caching" in argv) is (not canonical_observation(config))
             and (canonical_observation(config) or env.get("VLLM_USE_V2_MODEL_RUNNER") == "0")
             and env.get("CUDA_VISIBLE_DEVICES") == "0", "cache/runner/GPU policy drift")
        if canonical_observation(config):
            canonical_contract.verify_launch(lane, argv, env, launch['canonical_contract'],
                policy['models'], load(root / lane / 'observer.json'))
            inherited = launch['canonical_contract']['inherited_environment']
            runtime_env = {k: v for k, v in inherited.items() if k.startswith(
                ('VLLM_', 'CUDA_', 'TORCH_', 'PYTORCH_', 'NCCL_'))}
            need(runtime_env == preflight['runtime']['environment'] == policy['runtime_environment'],
                 'canonical inherited runtime environment drift')
            need(preflight['canonical_launch'][lane] == dict(
                contract=launch['canonical_contract'], environment=env), 'canonical preflight launch drift')
        expected_mode = control.get(f"{lane}_cudagraph_mode")
        need(argv.count("--compilation-config") == int(expected_mode is not None),
             "cudagraph control presence drift")
        if expected_mode is not None:
            need(decode(argv[argv.index("--compilation-config") + 1]) == {
                "cudagraph_mode": expected_mode,
            }, "cudagraph control value drift")
        if lane == 'adaptive' or not canonical_observation(config):
            need(all(env[k] == v for k, v in policy["adaptive_env"].items()), "adaptive environment drift")
        observer = load(root / lane / "observer.json")
        need(sources.profile_options(observer) == (production if lane == 'adaptive' or config.get('observation_only') else {}),
             'production run/observer profile mismatch')
        if config.get('observation_only'):
            need(observer.get('observation_only') is True and observation_spec(observer) == spec,
                 'observation run/observer case/index drift')
        need(observer.get('r15_attention_dispatch', False) is (r15 and lane == 'adaptive'),
             'R15 run/observer gate drift')
        need(observer.get('r16_force_attention_3d', False) is (r16 and lane == 'adaptive'),
             'R16 run/observer gate drift')
        expected_r14 = r14 if lane == 'adaptive' else None
        need(observer.get('r14_single_token_boundary') == expected_r14
             and env.get('M203_R14_SINGLE_TOKEN_BOUNDARY') == (None if expected_r14 is None else str(expected_r14)),
             'R14 launch/observer gate drift')
        if 'observation_target' in config:
            need(observer.get('observation_target') == spec['target']
                 and observer.get('reconstruction_target') == RECONSTRUCTION_TARGET,
                 'observer target drift')
        need(observer.get("diagnostic_control", {}) == control,
             "observer diagnostic control drift")
        if lane == "adaptive":
            spec = decode(argv[argv.index("--speculative-config") + 1])
            need(spec == {"method": "draft_model", "model": policy["models"]["draft"]["snapshot"],
                          "num_speculative_tokens": 4, "num_speculative_tokens_per_batch_size": [[1, 32, 4]],
                          "use_heterogeneous_vocab": True, "draft_sample_method": "greedy"}, "speculative policy drift")
    return execution


def response(directory, lane):
    raw = load(directory / "response.json")
    http = load(directory / "response_meta.json")
    need(load(directory / "request.claim") == {"outcome": "UNKNOWN", "budget_spent": True}, "missing/invalid one-shot request claim")
    need(http["status"] == 200, "HTTP request failed")
    request = load(directory / "request.json")
    need(request["model"] == "qwen2.5-7b-" + lane and request.pop("return_token_ids") is True, "payload audit flags")
    request["model"] = "qwen2.5:7b-instruct"
    expected = (observation_spec(load(directory / 'observer.json'))['request_sha256']
                if (directory / 'observer.json').exists() else REQUEST_HASH)
    need(fingerprint(request) == expected, "request semantics changed")
    need(len(raw["choices"]) == 1, "response choices ambiguous")
    choice = raw["choices"][0]
    ids, prompt = choice["token_ids"], raw["prompt_token_ids"]
    for tokens in (ids, prompt):
        need(isinstance(tokens, list) and bool(tokens) and all(type(x) is int and x >= 0 for x in tokens), "invalid API token IDs")
    need(choice["finish_reason"] in ("stop", "length") and len(ids) <= 768, "invalid finish/count")
    need(raw["usage"]["completion_tokens"] == len(ids), "reported/raw token count mismatch")
    need(load(directory / "response_saved.json")["sha256"] == sha((directory / "response.json").read_bytes()), "response incomplete/checksum mismatch")
    return raw["id"], prompt, ids, choice["finish_reason"]


def vector(root, descriptor):
    name = descriptor["file"]
    need(Path(name).name == name, "unsafe vector path")
    raw = (root / name).read_bytes()
    need(sha(raw) == descriptor["sha256"] and len(raw) == 4 * descriptor["vocabulary_size"], "score checksum/size mismatch")
    need(descriptor["storage_dtype"] == "float32-le", "unsupported score storage")
    stats = score_stats(raw)
    need(all(descriptor[k] == v for k, v in stats.items()), "score stats mismatch")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def kv_recompute_target(control, lane):
    need(lane in LANES, "unknown diagnostic lane")
    return control.get(f"{lane}_kv_recompute_output_count")


def refill_proof(directory, lane, request_id, prompt_len):
    """Validate the durable proof for the one controlled discarded refill."""
    root = directory / "capture"
    paths = sorted((root / "events").glob("*.json"))
    events = [(p, load(p)) for p in paths]
    starts = [(p, e) for p, e in events if e["kind"] == "kv_recompute_started"]
    skipped = [e for _, e in events if e["kind"] == "kv_recompute_refill_skipped"]
    observer_path = directory / "observer.json"
    control = (load(observer_path).get("diagnostic_control", {})
               if observer_path.is_file() else {})
    expected = kv_recompute_target(control, lane) == TARGET
    need(len(starts) == len(skipped) == int(expected),
         "controlled discarded refill proof missing/ambiguous")
    if not expected:
        return None
    started_path, started = starts[0]
    proof = skipped[0]
    need(re.fullmatch(re.escape(request_id) + r"(?:-[0-9a-f]{8})?",
                      started["request_id"])
         and proof["request_id"] == started["request_id"],
         "discarded refill request linkage mismatch")
    need(started["output_count"] == proof["output_count"] == TARGET
         and started["total_tokens"] == proof["total_tokens"] == prompt_len + TARGET
         and started["recomputed_tokens"] == proof["recomputed_tokens"]
         == started["total_tokens"] - 1
         and proof["scheduled_tokens"] == proof["recomputed_tokens"]
         and proof["active_query_token_count"] == proof["recomputed_tokens"]
         and proof["query_start_position"] == 0
         and proof["query_end_position"] == proof["recomputed_tokens"] - 1
         and proof["row_generated_index"] == TARGET - 1
         and proof["decoding_path"] == "ordinary"
         and proof["started_event_sha256"] == sha(started_path.read_bytes()),
         "controlled discarded refill proof invalid")
    return proof


def ready_proof(directory, lane, request_id, prompt_len):
    """Return the one content-addressed ready event required by KV control."""
    root = directory / "capture"
    paths = sorted((root / "events").glob("*.json"))
    events = [(p, load(p)) for p in paths]
    ready = [(p, e) for p, e in events if e["kind"] == "kv_recompute_ready"]
    observer_path = directory / "observer.json"
    control = (load(observer_path).get("diagnostic_control", {})
               if observer_path.is_file() else {})
    expected = kv_recompute_target(control, lane) == TARGET
    need(len(ready) == int(expected), "KV recompute ready event missing/ambiguous")
    if not expected:
        return None
    path, proof = ready[0]
    need(re.fullmatch(re.escape(request_id) + r"(?:-[0-9a-f]{8})?",
                      proof["request_id"])
         and proof["output_count"] == TARGET
         and proof["total_tokens"] == prompt_len + TARGET
         and proof["recomputed_tokens"] == proof["total_tokens"] - 1
         and proof["num_computed_tokens"] == proof["recomputed_tokens"]
         and proof["budget_restored"] is True,
         "KV recompute ready event invalid")
    return {
        "kv_recompute_ready_event_sha256": sha(path.read_bytes()),
        "request_id": proof["request_id"],
        "output_count": proof["output_count"],
        "total_tokens": proof["total_tokens"],
        "recomputed_tokens": proof["recomputed_tokens"],
        "num_computed_tokens": proof["num_computed_tokens"],
        "budget_restored": True,
    }


def scheduler_provenance(record, ready):
    """Validate the scheduler-time epoch carried by this exact forward."""
    proof = record.get("kv_recompute_scheduler_provenance")
    need(isinstance(proof, dict), "boundary scheduler provenance missing")
    need(type(proof.get("scheduler_step")) is int
         and proof["scheduler_step"] >= 0,
         "boundary scheduler step invalid")
    state = proof.get("state")
    need(state in ("PRE_READY", "POST_READY"),
         "boundary scheduler provenance state invalid")
    if state == "PRE_READY":
        need(set(proof) == {"state", "scheduler_step", "control_epoch"}
             and proof["control_epoch"] in (None, 1),
             "pre-ready scheduler provenance invalid")
        return state
    need(ready is not None and proof.get("control_epoch") == 1
         and {key: proof.get(key) for key in ready} == ready
         and set(proof) == {"state", "scheduler_step", "control_epoch", *ready},
         "ready-event scheduler provenance mismatch")
    return state


def capture(directory, lane, execution, request_id, prompt, ids):
    observer_path = directory / 'observer.json'
    identity_path = directory / 'capture/runtime_identity.json'
    if observer_path.is_file() and identity_path.is_file():
        identity = load(identity_path)
        if identity.get('model_runner') == canonical_contract.V2_RUNNER:
            observer = load(observer_path)
            need(v2_observation_profile(observer), 'V2 analyzer outside reviewed observer profile')
            run = load(directory.parent / 'run.json')
            need(v2_observation_profile(run) and observation_spec(run) == observation_spec(observer),
                 'V2 analyzer outside reviewed run profile')
            from .v2_analysis import capture_v2
            return capture_v2(directory, lane, execution, request_id, prompt, ids)
    if (directory / 'observer.json').is_file() and transition_observation(load(directory / 'observer.json')):
        return capture_transition(directory, lane, execution, request_id, prompt, ids)
    root = directory / "capture"
    errors = [load(p) for p in sorted((root / "events").glob("*.json"))]
    need(not any(e["kind"] == "observer_error" for e in errors), "observer error present")
    need("M203_TARGETED_OBSERVER_ERROR_UNSAVED" not in (directory / "server.log").read_text(errors="replace"), "observer error unsaved")
    need(not list(root.rglob("*.pending-*")), "incomplete capture write")
    refill = refill_proof(directory, lane, request_id, len(prompt))
    ready = ready_proof(directory, lane, request_id, len(prompt))
    records = [load(p) for p in sorted(root.glob("*.forward.json"))]
    hits = []
    for rec in records:
        need(rec["schema"] == "m203.targeted.forward.v1" and rec["lane"] == lane
             and rec["execution_sha"] == execution, "capture provenance mismatch")
        need(re.fullmatch(re.escape(request_id) + r"(?:-[0-9a-f]{8})?", rec["request_id"]), "request linkage mismatch")
        need(rec["prompt_token_ids"] == prompt and rec["prefix_cache_enabled"] is False, "prompt/cache drift")
        rows = rec["rows"]
        expected_rows = map_rows(len(prompt), [p for p, _ in rec["active_query"]],
                                 [r["flat_index"] for r in rows], rec["draft_count"],
                                 rec["spec_target_rows"], rec["spec_bonus_rows"])
        need(rec['decoding_path'] in ('ordinary', 'speculative'), 'unknown decoding path')
        if rec['decoding_path'] == 'ordinary':
            need(rec['draft_count'] == 0 and len(rows) == 1, 'ordinary path has speculative rows')
            expected_rows[0]["role"] = "ordinary"
        need(rows == expected_rows, "row-role/flattened position proof mismatch")
        selected = valid_selected(rec["selected"])
        need(0 < len(selected) <= len(rows), 'invalid selected prefix length')
        need(rec["accepted_drafts"] == len(selected) - 1
             and rec["rejected_drafts"] == rec["draft_count"] - len(selected) + 1, "verification accounting mismatch")
        for j, row in enumerate(rows):
            need(row["row"] == j and row["generated_index"] == row["position"] - len(prompt) + 1,
                 "capture position mapping mismatch")
            if row["generated_index"] != TARGET or j >= len(selected):
                continue  # A rejected speculative candidate is not an emitted row.
            dispatch_state = (scheduler_provenance(rec, ready)
                              if ready is not None else None)
            base = rows[0]["generated_index"]
            need(base >= 0, "negative selected boundary base")
            if dispatch_state != "PRE_READY":
                need(selected[j] == ids[TARGET], "sampler/API token mismatch")
                need(selected[:j] == ids[base:TARGET], "earlier accepted prefix mismatch")
            prefix = prompt + ids[:TARGET]
            pairs = [(p, t) for p, t in rec["active_query"] if p <= row["position"]]
            need(pairs and len({p for p, _ in pairs}) == len(pairs), "ambiguous active query")
            need(all(type(p) is int and 0 <= p < len(prefix) and prefix[p] == t for p, t in pairs)
                 and row["position"] in dict(pairs), "active query differs from API conditioning prefix")
            scores = rec["scores"][str(j)]
            need(set(scores) == {"raw", "processed"}, "missing raw/processed scores")
            need(scores['raw']['capture_point'] == 'runner_pre_grammar' and scores['raw']['source_row'] == j,
                 'raw source row/capture point mismatch')
            expected_point = 'rejection_pre_acceptance' if row['role'] == 'draft_verification' else 'sampler_pre_greedy'
            expected_source = j if row['role'] == 'draft_verification' else 0
            need(scores['processed']['capture_point'] == expected_point
                 and scores['processed']['source_row'] == expected_source, 'processed source row/capture point mismatch')
            raw = vector(root, scores["raw"])
            processed = vector(root, scores["processed"])
            need(len(raw) == len(processed), "vocabulary shape drift")
            hits.append({"record": rec["key"], "row": row, "scores": scores,
                         "selected_token": selected[j], "prefix_sha256": fingerprint(prefix),
                         "decoding_path": rec["decoding_path"],
                         "active_query": rec["active_query"],
                         "active_query_token_count": len(rec["active_query"]),
                         "scheduler_provenance_state": dispatch_state})
    if ready is None:
        need(len(hits) == 1, "missing/ambiguous selected boundary capture")
        selected_hit = hits[0]
    else:
        post_ready = [hit for hit in hits
                      if hit["scheduler_provenance_state"] == "POST_READY"]
        need(len(post_ready) == 1,
             "missing/ambiguous post-recompute selected boundary capture")
        selected_hit = post_ready[0]
        selected_hit["kv_recompute_ready"] = ready
        selected_hit["excluded_pre_ready_boundary_records"] = [
            hit["record"] for hit in hits
            if hit["scheduler_provenance_state"] == "PRE_READY"]
    hits = [selected_hit]
    matching = [e for e in errors if e.get("request_id") == next(r["request_id"] for r in records if r["key"] == hits[0]["record"])]
    boundary_accounting = [e for e in matching if e["kind"] in ("account_before", "account_after")
                           and (e["output_count"] <= TARGET < e["output_count"] + len(e["output_token_ids"])
                                if e["kind"] == "account_before" else
                                e["output_count"] - len(e["output_token_ids"]) <= TARGET < e["output_count"])]
    need(all(type(e.get("stale")) is bool for e in boundary_accounting),
         "invalid accounting stale flag")
    before = [e for e in boundary_accounting
              if e["kind"] == "account_before" and not e["stale"]]
    after = [e for e in boundary_accounting
             if e["kind"] == "account_after" and not e["stale"]]
    need(len(before) == len(after) == 1, "missing/ambiguous verification accounting")
    b, a = before[0], after[0]
    need(not b["stale"] and not a["stale"], "stale output at boundary")
    for e in (b, a):
        need(all(type(e[k]) is int and e[k] >= 0 for k in
                 ("num_computed_tokens", "num_in_flight_tokens", "num_output_placeholders")), "invalid scheduler counters")
    start = b["output_count"]
    chosen = next(r for r in records if r['key'] == hits[0]['record'])
    emitted_offset = TARGET - start
    need(start == chosen['rows'][0]['generated_index'] and emitted_offset == hits[0]['row']['row'],
         'source row does not prove emitted output index')
    need(b['output_token_ids'] == chosen['selected'], 'scheduler selected sequence differs from GPU step')
    need(len(b['scheduled_spec_token_ids']) == chosen['draft_count'], 'scheduled draft count/forward mismatch')
    need(type(b['scheduled_tokens']) is int and b['scheduled_tokens'] >= chosen['draft_count'] + 1,
         'scheduled forward span unproven')
    need(b['invalid_spec_tokens'] == a['invalid_spec_tokens'] == 0, 'invalid speculative rows unsupported')
    need(a['output_token_ids'] == b['output_token_ids'][:len(a['output_token_ids'])]
         and 0 <= emitted_offset < len(a['output_token_ids'])
         and a['output_token_ids'][emitted_offset] == hits[0]['selected_token'], 'emitted token offset/stop trimming mismatch')
    need(a["output_count"] == start + len(a["output_token_ids"])
         and a["output_token_ids"] == ids[start:a["output_count"]], "scheduler/API accounting mismatch")
    need(type(chosen['async_scheduling']) is bool, 'unknown async scheduling mode')
    consumed_placeholders = len(a['output_token_ids']) if chosen['async_scheduling'] else 0
    need(a['num_output_placeholders'] == b['num_output_placeholders'] - consumed_placeholders,
         'async placeholder delta/emitted count mismatch')
    hits[0]["accounting"] = {"before": b, "after": a}
    j = hits[0]['row']['row']
    hits[0]['selection_role'] = ('ordinary' if chosen['decoding_path'] == 'ordinary' else 'bonus'
                                if j == chosen['draft_count'] else 'accepted_draft'
                                if j < len(chosen['selected']) - 1 else 'recovered')
    if refill is not None:
        hits[0]["kv_recompute_refill_skipped"] = refill
    return hits[0]


def validate_lane(root, lane):
    """Stop before the second workload if the first lane lacks valid evidence."""
    root = Path(root)
    execution = provenance(root, lanes=(lane,))
    directory = root / lane
    shutdown = verify_shutdown(directory)
    request_id, prompt, ids, finish = response(directory, lane)
    target = observation_target(load(directory / 'observer.json'))
    need(len(ids) > target, "lane terminated before target; stop, do not send next lane")
    proof = capture(directory, lane, execution, request_id, prompt, ids)
    return {"lane_valid": True, "execution_sha": execution, "request_id": request_id,
            "response_sha256": sha((directory / "response.json").read_bytes()),
            "capture_record": proof["record"], "finish_reason": finish, "shutdown": shutdown}


def compare(root):
    root = Path(root)
    config = load(root / 'run.json')
    target = config.get('observation_target', 60)
    report = {"result_kind": "DIAGNOSTIC_NOT_CANONICAL_GATE", "case": config.get('case', CASE), "target": target,
              "diagnostic_control": load(root / "run.json")["policy"].get("diagnostic_control", {}),
              "same_internal_state_proven": False, "same_kv_state_proven": False,
              "runtime_probe_changes_timing": True,
              "historical_gate": "44/54 FAIL" if canonical_observation(config) else "47/54 FAIL"}
    if transition_observation(config):
        report.update(observation_target=target, reconstruction_target=RECONSTRUCTION_TARGET)
    try:
        config = load(root / "run.json")
        execution = provenance(root)
        rows, recompute_proofs = {}, {}
        for lane in LANES:
            directory = root / lane
            verify_shutdown(directory)
            events = [load(p) for p in (directory / "capture/events").glob("*.json")]
            need(not any(e["kind"] == "observer_error" for e in events)
                 and "M203_TARGETED_OBSERVER_ERROR_UNSAVED" not in (directory / "server.log").read_text(errors="replace")
                 and not list((directory / "capture").rglob("*.pending-*")), "observer error/incomplete write")
            recomputes = [e for e in events if e["kind"] == "kv_recompute_ready"]
            expected_recomputes = int(kv_recompute_target(
                config["policy"].get("diagnostic_control", {}), lane) == TARGET)
            need(len(recomputes) == expected_recomputes, "KV recompute control missing/ambiguous")
            if recomputes:
                proof = recomputes[0]
                need(proof["output_count"] == TARGET
                     and proof["recomputed_tokens"] == proof["total_tokens"] - 1
                     and proof["num_computed_tokens"] == proof["recomputed_tokens"]
                     and proof["budget_restored"] is True,
                     "KV recompute control proof invalid")
                recompute_proofs[lane] = proof
            request_id, prompt, ids, finish = response(directory, lane)
            rows[lane] = {"request_id": request_id, "prompt": prompt, "ids": ids, "finish_reason": finish}
        v, a = rows["vanilla"], rows["adaptive"]
        need(v["prompt"] == a["prompt"], "lane prompt IDs differ")
        d = None if v["ids"] == a["ids"] else next(
            (i for i, (x, y) in enumerate(zip(v["ids"], a["ids"])) if x != y),
            min(len(v["ids"]), len(a["ids"])))
        report.update(first_divergence=d, execution_sha=execution,
                      finish_reasons={k: r["finish_reason"] for k, r in rows.items()},
                      full_completion_verified=all(r["finish_reason"] == "stop" for r in rows.values()),
                      truncated_lanes=[k for k, r in rows.items() if r["finish_reason"] == "length"],
                      exact_output_token_ids_equal=d is None,
                      finish_reasons_equal=v["finish_reason"] == a["finish_reason"])
        captures = ({k: capture(root / k, k, execution, r['request_id'], r['prompt'], r['ids'])
                     for k, r in rows.items()} if transition_observation(config) else None)
        r16 = config.get('r16_force_attention_3d', False)
        outside = d is not None and d != target and not (r16 and d > target)
        if min(len(v["ids"]), len(a["ids"])) <= target or outside:
            report.update(state="BOUNDARY_NOT_COVERED", reason="new divergence/termination outside target boundary")
            return report
        if captures is None:
            captures = {k: capture(root / k, k, execution, r["request_id"], r["prompt"], r["ids"]) for k, r in rows.items()}
        need(captures["vanilla"]["prefix_sha256"] == captures["adaptive"]["prefix_sha256"], "lane conditioning prefixes differ")
        for lane, row in rows.items():
            if kv_recompute_target(config["policy"].get(
                    "diagnostic_control", {}), lane) == TARGET and target == TARGET:
                proof = recompute_proofs[lane]
                need(proof["total_tokens"] == len(row["prompt"]) + TARGET
                     and proof["request_id"].startswith(row["request_id"])
                     and captures[lane]["decoding_path"] == "ordinary"
                     and captures[lane]["active_query_token_count"] == 1
                     and captures[lane]["active_query"] == [[
                         len(row["prompt"]) + TARGET - 1,
                         row["ids"][TARGET - 1],
                     ]],
                     "KV recompute did not preserve one-token boundary forward")
        if recompute_proofs:
            report["kv_recompute_proofs"] = {
                lane: captures[lane]["kv_recompute_ready"]
                for lane in recompute_proofs
            }
        report.update(state="NOT_REPRODUCED" if d is None else "REPRODUCED", captures=captures,
                      historical_token_pair_reproduced=(not config.get('observation_only')
                          and (v["ids"][TARGET], a["ids"][TARGET]) == (51, 79759)))
        geometry = {lane: {name: proof[name] for name in (
            'logical_runtime_q', 'captured_graph_q_shape', 'max_seqlen_q')}
            for lane, proof in captures.items() if 'logical_runtime_q' in proof}
        if geometry:
            report['v2_query_geometry'] = geometry
        report["processed_argmax_equal"] = captures["vanilla"]["scores"]["processed"]["top1_id"] == captures["adaptive"]["scores"]["processed"]["top1_id"]
        report["selected_equals_processed_argmax"] = {k: x["selected_token"] == x["scores"]["processed"]["top1_id"] for k, x in captures.items()}
        if config.get('r16_force_attention_3d', False):
            report['attention_dispatch_causal_test'] = r16_causal_result(captures['adaptive'])
        differences = {}
        for kind in ("raw", "processed"):
            values = [vector(root / lane / "capture", captures[lane]["scores"][kind]) for lane in LANES]
            need(len(values[0]) == len(values[1]), "cross-lane vocabulary shape drift")
            finite = [abs(x - y) for x, y in zip(*values) if x != float('-inf') and y != float('-inf')]
            differences[kind] = {"max_abs_diff_on_common_finite_support": max(finite) if finite else None,
                                 "masked_support_differences": sum((x == float('-inf')) != (y == float('-inf')) for x, y in zip(*values))}
        report["score_differences"] = differences
    except (Invalid, OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        report.update(state="INVALID_CAPTURE", reason=f"{type(exc).__name__}: {exc}")
    return report
