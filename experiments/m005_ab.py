"""Explicit prepare/capture/seal/analyze for one conditioned M005 A/B diagnostic.

No command starts/stops a server or runs a canonical sweep. Source installation
is separate and requires the server to have been stopped by its owner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.protocol_b import (ROOT, require, load_json, load_plan, fingerprint,
                                    file_sha, write_new, EvidenceError, ENV_FIELDS, validate_run, index_prompts)
from experiments.m005_ab_sources import PROFILES, HASHES, PROJECT_FILES, sha, transform, authenticate
from experiments.m005_ab_observer import SAMPLE, PROMPT, POSITION, INDEX

PLAN = '6d9af3ced74d37dc73928b2fee4ede9e3e95048d8d6445cabb8ecedc38e17389'
REQUEST = '82cae35e6e648b01a76864382d0a875bbaa83cda784e8491c6a70249e1bdd16e'
IDS = ['m0_04_%03d' % n for n in range(1, 6)]
REFERENCE = ROOT / 'experiments/runs/m2_03/protocol_b_20261018T085118Z_2aad4c00'


def validate_execution_checkout(config_path):
    from experiments.protocol_b_capture import command
    config = load_json(config_path)
    authenticate(config['source_directory'])
    status = command(['git', 'status', '--porcelain', '--untracked-files=no'])
    require(status.strip() == 'M src/server/adaptive_scheduler.py', 'Unapproved diagnostic checkout edits')
    command(['git', 'ls-files', '--error-unmatch', 'experiments/m005_ab.py',
             'experiments/m005_ab_sources.py', 'experiments/m005_ab_observer.py'])


def conditioned_prompts(plan):
    require(plan['plan_sha256'] == PLAN and len(plan['prompts']) == 54
            and fingerprint({'prompts': plan['prompts'], 'sources': plan['sources']}) == PLAN
            and [p['sample_id'] for p in plan['prompts'][:5]] == IDS
            and plan['prompts'][4]['request_sha256'] == REQUEST, 'Frozen M005 prefix changed')
    return plan['prompts'][:5]


def prepare(directory, session, gpu, boot):
    from experiments.protocol_b_capture import reserve_run
    plan = load_plan()
    conditioned_prompts(plan)
    reference = load_json(REFERENCE / 'pair1-vanilla/run.json')['records'][4]
    require(len(reference['prompt_token_ids']) == PROMPT and reference['request_sha256'] == REQUEST,
            'M005 prompt reference mismatch')
    require(session and boot and gpu.startswith('GPU-'), 'Pin session/boot/full GPU UUID')
    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=False)
    write_new(directory / 'experiment.json', dict(sample=SAMPLE, index=INDEX, plan=plan,
              session_id=session, gpu_uuid=gpu, boot_id=boot, non_canonical=True, task_done=False))
    for condition in PROFILES:
        arm = directory / condition
        reserve_run(arm, 'm005-' + condition, 'm005-' + condition, 'adaptive', session)
        (arm / 'observer').mkdir()
        config = dict(profile=PROFILES[condition], condition=condition, sample_id=SAMPLE, index=INDEX,
                      prompt_token_ids=reference['prompt_token_ids'], request_sha256=REQUEST,
                      directory=str(arm), source_directory=str(arm / 'sources'), session_id=session,
                      gpu_uuid=gpu, boot_id=boot, runtime_probe_changes_timing=True)
        write_new(arm / 'observer_config.json', config)


def validate_capture_scope(output, session, lane, live):
    config = load_json(Path(output) / 'observer_config.json')
    require(lane == 'adaptive' and config['session_id'] == session
            and config['sample_id'] == SAMPLE and config['index'] == INDEX, 'Wrong M005 capture scope')
    require(live['_process_environment'].get('M203_M005_CONFIG') == str(Path(output).resolve() / 'observer_config.json'),
            'Server is not bound to selected observer config')
    require(live['_process_environment'].get('PROTOCOL_B_BOUNDARY_CONFIG') is None,
            'Historical observer enabled')
    source = authenticate(config['source_directory'])
    require(source['profile'] == config['profile'], 'Capture profile mismatch')


def capture_arm(directory, condition):
    from experiments.protocol_b_capture import capture
    arm = Path(directory).resolve() / condition
    config = load_json(arm / 'observer_config.json')
    capture(arm, 'm005-' + condition, 'm005-' + condition, 'adaptive', config['session_id'],
            reserved=True, diagnostic_through=SAMPLE)
    authenticate(config['source_directory'])


def seal_arm(directory, condition):
    arm = Path(directory).resolve() / condition
    config = load_json(arm / 'observer_config.json')
    authenticate(config['source_directory'])
    require((arm / 'run.json').is_file(), 'Missing captured run')
    files = {p.relative_to(arm).as_posix(): file_sha(p) for p in arm.rglob('*') if p.is_file()}
    require('seal.json' not in files, 'Already sealed')
    write_new(arm / 'seal.json', dict(profile=PROFILES[condition], files=files))


def validate_bits(record, *, bytes_required=False):
    require(record['byteorder'] == 'little' and record['byte_count'] > 0
            and len(record['sha256']) == 64 and record['shape'] and record['dtype'], 'Invalid bitwise record')
    if bytes_required:
        raw = bytes.fromhex(record['hex'])
        require(sha(raw) == record['sha256'] and len(raw) == record['byte_count'], 'Tensor bytes/hash mismatch')
    return record['sha256'], record['shape'], record['dtype'], record.get('strides')


def validate_arm(arm, condition):
    """Offline validation of sealed raw evidence; never trust a classification flag."""
    arm = Path(arm)
    seal = load_json(arm / 'seal.json')
    require(seal['profile'] == PROFILES[condition], 'Wrong seal profile')
    for relative, digest in seal['files'].items():
        path = (arm / relative).resolve()
        require(path.is_relative_to(arm.resolve()) and file_sha(path) == digest, 'Sealed artifact drift')
    required = ['observer_config.json', 'run.json', 'responses.jsonl', 'sources/sources.json']
    require(all(p in seal['files'] for p in required), 'Incomplete evidence seal')
    config = load_json(arm / 'observer_config.json')
    require(config['profile'] == PROFILES[condition] and config['sample_id'] == SAMPLE
            and config['index'] == INDEX and config['request_sha256'] == REQUEST
            and len(config['prompt_token_ids']) == PROMPT, 'Wrong sealed scope')
    source = load_json(arm / 'sources/sources.json')
    require(source['profile'] == PROFILES[condition] and source['originals'] == HASHES
            and source['condition'] == condition
            and set(source['project_hashes']) == set(PROJECT_FILES), 'Wrong composed source declaration')
    for name in HASHES:
        relative = 'sources/originals/' + name.replace('project:', 'project/')
        require(relative in seal['files'], 'Unsealed source backup')
        raw = (arm / relative).read_bytes()
        require(sha(transform(name, raw, condition)) == source['composed'][name], 'Composed hash drift')
    for name, digest in source['project_hashes'].items():
        relative = 'sources/project-sources/' + name
        require(relative in seal['files'] and file_sha(arm / relative) == digest, 'Unsealed/drifted project source')
    run = load_json(arm / 'run.json')
    require(run['complete'] is True and run.get('diagnostic_only') is True
            and run['diagnostic_through'] == SAMPLE and run['lane'] == 'adaptive'
            and [r['sample_id'] for r in run['records']] == IDS, 'Incomplete/wrong run scope')
    require(file_sha(arm / 'responses.jsonl') == run['responses_sha256'], 'Response journal drift')
    plan = load_json(arm.parent / 'experiment.json')['plan']
    prompts = conditioned_prompts(plan)
    validate_run(run, index_prompts(prompts), 'm005-' + condition, 'adaptive', run['environment'])
    require(run['environment']['plan_sha256'] == PLAN
            and run['environment']['session_id'] == config['session_id']
            and run['environment']['boot_id'] == config['boot_id']
            and run['environment']['gpu_uuid'] == config['gpu_uuid'], 'Run/config identity drift')
    from experiments.m005_ab_sources import SCHEDULER
    expected_project = dict(source['project_hashes'])
    expected_project['src/server/adaptive_scheduler.py'] = source['composed'][SCHEDULER]
    for field in ('source_hashes', 'launcher_hashes'):
        require(all(expected_project.get(n) == digest for n, digest in run['environment'][field].items()),
                'Runtime/project source manifest differs')
    journal = [json.loads(l) for l in (arm / 'responses.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(journal) == 10, 'Missing/extra HTTP attempts')
    for i, (record, prompt) in enumerate(zip(run['records'], prompts)):
        require(record['ok'] is True and record['request_sha256'] == prompt['request_sha256'], 'Request failed/drifted')
        attempt, result = journal[2*i:2*i+2]
        require(attempt['event'] == 'attempt' and result['event'] == 'result'
                and result['record'] == record and result['http_status'] == 200, 'Malformed raw HTTP linkage')
        expected_payload = dict(prompt['request'], model='qwen2.5-7b-adaptive', return_token_ids=True)
        require(attempt['request'] == expected_payload, 'Frozen payload drift')
        body = json.loads(result['raw_response_body'])
        from experiments.protocol_b_capture import parse_completion
        parsed = parse_completion(body, 'qwen2.5-7b-adaptive')
        require(body == result['response'] and all(record[k] == v for k, v in parsed.items()),
                'Raw response/record drift')
    events = [load_json(p) for p in sorted((arm / 'observer').glob('*.json'))]
    require(events and all('observer/' + ('%06d.json' % e['event_seq']) in seal['files'] for e in events),
            'Unsealed/missing observer events')
    require([e['event_seq'] for e in events] == list(range(1, len(events)+1))
            and all(e['config_sha256'] == fingerprint(config) and e['profile'] == config['profile'] for e in events),
            'Observer sequence/config drift')
    require(events[0]['kind'] == 'bound' and not any(e['kind'] == 'observer_error' for e in events), 'Observer failed/unbound')
    serving = run['live_server']
    parents = [p for p in events[0]['process_chain'] if p['pid'] == serving['pid']]
    require(len(parents) == 1 and parents[0]['start_ticks'] == serving['process_start_ticks']
            and all(e['pid'] == events[0]['pid'] for e in events), 'Observer not bound to serving process ancestry')
    commits = [e for e in events if e['kind'] == 'attempt' and e['committing'] is True]
    require(len(commits) == 1, 'Missing/ambiguous committing attempt')
    entry = commits[0]
    record = run['records'][4]
    require(record['prompt_token_ids'] == config['prompt_token_ids'], 'Tokenized prompt/config differs')
    require(entry['request_id'].startswith(record['response_id'] + '-')
            and entry['M'] in range(1, 6) and POSITION in entry['positions']
            and entry['positions'] == list(range(entry['positions'][0], entry['positions'][-1]+1)), 'Wrong boundary identity')
    n = entry['draft_count']
    require(type(n) is int and 0 <= n <= 4 and len(entry['rows']) == n+1, 'Invalid consumed draft rows')
    from experiments.protocol_b_boundary_observer import map_rows
    require(entry['rows'] == map_rows(PROMPT, entry['positions'], [r['flat_index'] for r in entry['rows']],
                                    n, list(range(n)), [n]), 'Actual logits-row mapping differs')
    selected = [r for r in entry['rows'] if r['generated_index'] == INDEX]
    require(len(selected) == 1 and selected[0] == entry['target_row']
            and selected[0]['position'] == POSITION, 'Ambiguous selected row')
    row = selected[0]['row']
    require(not entry['stale'] and entry['before_count'] + row == INDEX
            and row < len(entry['selected']) and entry['committed'][INDEX] == entry['selected'][row]
            and entry['committed'] == record['token_ids'][:len(entry['committed'])], 'Commit linkage failed')
    require(entry['pre_count'] == entry['post_count'] == entry['dispatch_count'] == 1, 'Incomplete attention hooks')
    pre = entry['pre']
    require(pre['query_starts'] == [0, entry['M']] and len(pre['seq_lens']) == 1
            and pre['Lr'] == [pre['seq_lens'][0]-entry['M']+r+1 for r in range(entry['M'])]
            and pre['Lr'][entry['positions'].index(POSITION)] == POSITION+1, 'Invalid effective row geometry')
    for tensor in (pre['Q'], entry['post']): validate_bits(tensor, bytes_required=True)
    for tensor in (pre['KV']['K'], pre['KV']['V'], entry['raw']['vector'], entry['processed']['vector']):
        validate_bits(tensor)
    if n:
        producer = entry['producer']
        require(producer is not None and producer['schedule_seq'] == entry['producer_schedule_seq']
                and producer['planned_k'] is not None and producer['produced_count'] >= n
                and 0 <= entry['effective_drafted'] <= n, 'Broken producer/consumer lineage')
    require(type(entry['dispatch']['rowwise']) is bool
            and (condition != 'B' or not entry['dispatch']['rowwise'])
            and (entry['M'] != 1 or not entry['dispatch']['rowwise']), 'Impossible rowwise treatment')
    metrics = journal[-1]['response'].get('metrics', {}).get('speculative_decoding', {})
    require(metrics.get('num_draft_tokens', 0) > 0 and metrics.get('num_spec_steps', 0) > 0,
            'No authenticated real speculation in selected request')
    caches = [e for e in events if e['kind'] == 'cache_lookup']
    require(len(caches) == 1, 'Missing/ambiguous cache lookup')
    return dict(run=run, entry=entry, config=config, source=source, cache=caches[0])


def compare_validated(a, b):
    """Classify only independently validated arms. No numeric PASS tolerance."""
    ar, br, x, y = a['run'], b['run'], a['entry'], b['entry']
    gates = {}
    for name in ENV_FIELDS:
        if name == 'config':
            # Artifact destination is condition-specific, never a runtime policy.
            import copy
            left, right = copy.deepcopy(ar['environment'][name]), copy.deepcopy(br['environment'][name])
            for value in (left, right): value['environment'].pop('ADAPTIVE_TELEMETRY_PATH', None)
            gates[name] = left == right
        else: gates[name] = ar['environment'][name] == br['environment'][name]
    gates['physical_T4'] = ar['environment']['gpu_name'] == br['environment']['gpu_name'] == 'Tesla T4'
    gates['conditioned_outputs'] = all((l['token_ids'], l['finish_reason'], l['prompt_token_ids']) ==
        (r['token_ids'], r['finish_reason'], r['prompt_token_ids']) for l, r in zip(ar['records'][:4], br['records'][:4]))
    av, bv = ar['records'][4]['token_ids'], br['records'][4]['token_ids']
    gates['prefix'] = av[:INDEX] == bv[:INDEX]
    for field in ('controller', 'past_budget_sha256', 'planned_k', 'batch_k', 'future_k', 'M',
                  'positions', 'input_ids', 'draft_count', 'draft_ids', 'effective_drafted', 'graph_mode'):
        gates[field] = x[field] == y[field]
    # Normalize process-local schedule/tensor ownership; compare producer semantics.
    for field in ('planned_k', 'batch_k', 'future_k', 'produced_count', 'controller'):
        gates['producer_' + field] = (x['producer'] or {}).get(field) == (y['producer'] or {}).get(field)
    gates['cache_lookup'] = (a['cache']['local'], a['cache']['external']) == (b['cache']['local'], b['cache']['external'])
    gates['pre_attention'] = x['pre'] == y['pre']
    gates['treatment_at_boundary'] = x['dispatch']['rowwise'] and not y['dispatch']['rowwise']
    semantics = []
    for entry in (x, y):
        operands = dict(entry['dispatch'].get('operands', {}))
        operands.pop('rowwise_decode_max_q', None)  # The declared policy metadata difference.
        semantics.append(operands)
    gates['attention_semantics'] = semantics[0] == semantics[1]
    gates['observer_identical'] = a['source']['project_hashes'] == b['source']['project_hashes']
    localization = ('INHERITED_DIVERGENCE_OPERATOR_UNKNOWN' if not gates['pre_attention'] else
                    'LAYER0_ATTENTION_DIFFERENCE_LOCALIZED' if x['post'] != y['post'] else
                    'OUTSIDE_OBSERVED_LAYER0_ATTENTION_OUTPUT' if x['raw'] != y['raw'] else 'NO_LOCALIZED_DIFFERENCE')
    if not all(gates.values()) or av[INDEX] != 5212:
        classification = 'INCONCLUSIVE'
    elif bv[INDEX] == 341:
        classification = 'ROWWISE_POLICY_CAUSAL'
    elif bv[INDEX] == 5212:
        classification = 'ROWWISE_POLICY_NOT_SUFFICIENT'
    else: classification = 'INCONCLUSIVE'
    return dict(classification=classification, gates=gates, localization=localization,
                tokens=[av[INDEX], bv[INDEX]], kernel_root_cause='UNKNOWN', task_done=False,
                interpretation_status='NOT_YET_APPROVED')


def analyze(directory):
    try:
        a, b = [validate_arm(Path(directory) / c, c) for c in PROFILES]
        return compare_validated(a, b)
    except (EvidenceError, KeyError, ValueError, OSError, TypeError, IndexError) as exc:
        return dict(classification='INVALID_EVIDENCE', error=str(exc), task_done=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'capture', 'seal', 'analyze', 'install', 'verify', 'restore', 'diff'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--condition', choices=PROFILES)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--session-id'); parser.add_argument('--gpu-uuid'); parser.add_argument('--boot-id')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.directory, args.session_id, args.gpu_uuid, args.boot_id)
    elif args.command == 'analyze': print(json.dumps(analyze(args.directory), indent=2))
    else:
        require(args.condition in PROFILES, 'Condition required')
        if args.command == 'capture': capture_arm(args.directory, args.condition)
        elif args.command == 'seal': seal_arm(args.directory, args.condition)
        else:
            from experiments import m005_ab_sources as sources
            directory = args.directory / args.condition / 'sources'
            if args.command in ('install', 'diff'):
                require(args.source_root is not None, 'Explicit pristine vLLM package path required')
                if args.command == 'diff': print(sources.diff(args.source_root, args.project_root, args.condition), end='')
                else:
                    require_stopped()
                    sources.install(args.source_root, args.project_root, directory, args.condition)
            else:
                if args.command == 'restore': require_stopped()
                sources.authenticate(directory, restore=args.command == 'restore')


def require_stopped():
    """Read-only Linux port-owner gate; never stop or restart anything."""
    import subprocess
    for port in (8001, 8002, 8003):
        probe = subprocess.run(['fuser', str(port) + '/tcp'], capture_output=True, text=True)
        require(probe.returncode == 1 and not probe.stdout.strip() and not probe.stderr.strip(),
                'All lanes must be stopped before source install/restore')


if __name__ == '__main__': main()
