"""Focused CPU-only observation tests. All runtime/score captures are synthetic."""
from pathlib import Path
from types import SimpleNamespace as NS
import re
import shlex
import ast
import os

import numpy as np
import pytest
import torch

from experiments.m2_03_targeted import common as c, execution as e, analysis as a, observer as o, sources as s
from experiments.m2_03_targeted import canonical as cc
import test_m2_03_targeted_gpu_diag as legacy
import test_m2_03_production_observer as production_observer
import test_triton_spec_decode_patch as production_tests

CASE = 'w3_business_v2_006'
POLICY = 'experiments/m2_03_targeted/runtime_policy.kaggle_t4_index60.json'


def init_case(tmp_path, monkeypatch, **options):
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'blob', lambda path, _: (c.ROOT / path).read_bytes())
    for key, value in c.load(c.ROOT / POLICY)['runtime_environment'].items():
        monkeypatch.setenv(key, value)
    return e.init_run(tmp_path, 'd' * 40, POLICY, **options)


def test_default_is_unchanged_and_explicit_case_uses_frozen_request(tmp_path, monkeypatch):
    default = init_case(tmp_path / 'default', monkeypatch)
    original = c.load(default / 'run.json')
    assert original['case'] == c.CASE and original['observation_target'] == original['target'] == 368
    assert 'observation_only' not in original
    assert original['request_sha256'] == c.REQUEST_HASH
    assert original['policy'] == c.load(c.ROOT / POLICY)
    root = init_case(tmp_path / 'explicit', monkeypatch, observation_case=CASE, observation_index=81)
    run = c.load(root / 'run.json')
    assert c.observation_spec(run) == dict(case=CASE, target=81, request_sha256=c.OBSERVATION_CASES[CASE])
    assert c.fingerprint(c.frozen_request(CASE)) == run['request_sha256']
    assert c.sha(c.blob(c.REQUEST_FILE)) == c.WORKLOAD_HASH
    assert run['policy']['diagnostic_control'] == {}
    for lane in c.LANES:
        config = c.load(root / lane / 'observer.json')
        assert c.observation_spec(config) == c.observation_spec(run)
        assert s.profile_options(config) == {'production': True, 'canonical': True, 'v2': True}
        assert o.kv_recompute_target(config) is None
        assert not any(config.get(k) for k in ('r14_single_token_boundary', 'r15_attention_dispatch', 'r16_force_attention_3d'))
        argv, env = e.launch_args(root, lane)
        assert '--compilation-config' not in argv
        assert '--no-enable-prefix-caching' not in argv
        assert env['HF_HUB_ENABLE_HF_TRANSFER'] == '0'
        assert '--compilation-config' in e.launch_args(default, lane)[0]


@pytest.mark.parametrize('options', [
    dict(observation_case='unknown'), dict(observation_index=0), dict(observation_index=-1),
    dict(observation_index=768), dict(observation_index=True),
    dict(observation_case=CASE), dict(observation_case=CASE, observation_index=368),
    dict(observation_case=c.CASE, observation_index=81),
    dict(observation_case=CASE, observation_index=80),
    dict(observation_case=CASE, observation_index=82), dict(observation_index=367),
    dict(observation_case=CASE, observation_index=81, r14_single_token_boundary=True),
    dict(observation_case=CASE, observation_index=81, r16_force_attention_3d=True),
    dict(observation_case=CASE, observation_index=81, r15_attention_dispatch=True),
])
def test_invalid_selection_or_intervention_is_refused_before_creating_run(tmp_path, monkeypatch, options):
    with pytest.raises(c.Invalid): init_case(tmp_path / 'run', monkeypatch, **options)
    assert not (tmp_path / 'run').exists()


def prepare_runtime(root, monkeypatch):
    package = root.parent / 'vllm'
    profile = s.profile_options(c.load(root / 'run.json'))
    for name, raw in s.base_sources(**profile).items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    policy = c.load(root / 'run.json')['policy']
    rt = dict(policy['runtime'], environment=policy['runtime_environment'], vllm_root=str(package))
    monkeypatch.setattr(e, 'runtime', lambda: rt)
    monkeypatch.setattr(e, 'free_port', lambda _: None)
    monkeypatch.setattr(e, 'model_inventory', lambda _: policy['models'])
    e.preflight(root)
    e.install(root)
    return package


def observe_lane(directory, lane, q_len=1, offset=0):
    config = c.load(directory / 'observer.json')
    target = config['observation_target']
    prompt, ids, rid = [0] * 5, [1] * (target + 2), 'api-1234abcd'
    table = NS(num_blocks_per_row=np.array([32]), block_size=16, kv_cache_block_size=16,
               blocks_per_kv_block=1, dcp_world_size=1, pcp_world_size=1,
               get_device_tensor=lambda _: torch.tensor([[2] * 32]))
    batch = NS(num_reqs=1, req_ids=[rid], has_structured_output_reqs=False,
               prev_req_id_to_index={rid: 0}, block_table=NS(block_tables=[table]))
    runner_class = type('GPUModelRunner', (NS,), {'__module__': 'vllm.v1.worker.gpu_model_runner'})
    runner = runner_class(input_batch=batch, uses_mrope=False, enable_prompt_embeds=False,
                model_config=NS(hf_config=NS(num_hidden_layers=28)),
                cache_config=NS(enable_prefix_caching=c.canonical_observation(config)),
                vllm_config=NS(use_v2_model_runner=False,
                    parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                    data_parallel_size=1, decode_context_parallel_size=1),
                    compilation_config=NS(cudagraph_mode=NS(name=config['diagnostic_control'].get(
                        f'{lane}_cudagraph_mode', 'PIECEWISE')))),
                requests={rid: NS(prompt_token_ids=prompt)}, use_async_scheduling=True,
                use_async_spec_decode=lane == 'adaptive',
                prev_positions=NS(np=np.array([0]), gpu=torch.tensor([0])),
                prev_num_draft_tokens=NS(np=np.array([0]), gpu=torch.tensor([0])),
                valid_sampled_token_count_gpu=torch.tensor([1]) if lane == 'adaptive' else None,
                discard_request_mask=NS(np=[False]))
    scheduler = NS(running=[rid], max_num_scheduled_tokens=4096)
    o.observe('runner_initialized', runner)
    fn, _, _ = production_observer.observed_wrapper()
    for step, index in enumerate(range(target - offset - 2, target - offset + 1), 1):
        q = q_len if step == 3 else 1
        drafts = q - 1
        position = len(prompt) + index - 1
        batch.num_computed_tokens_cpu = np.array([position])
        runner.num_computed_tokens = torch.tensor([position - 1])
        runner.query_start_loc = NS(np=np.array([0, q]), gpu=torch.tensor([0, q]))
        scheduled = NS(total_num_scheduled_tokens=q, num_scheduled_tokens={rid: q},
                       scheduled_spec_decode_tokens={rid: [2] * drafts})
        before = (list(scheduler.running), scheduler.max_num_scheduled_tokens)
        o.control('before_schedule', scheduler)
        o.control('after_schedule', scheduler, scheduled)
        assert before == (scheduler.running, scheduler.max_num_scheduled_tokens)
        assert not hasattr(scheduler, '_m203_kv_recompute')
        o.observe('correction_before', runner, scheduled)
        runner.num_computed_tokens[0] = position
        runner.positions = torch.arange(position, position + q)
        tokens = [1] * (offset + 1) + [2] * max(0, q - offset - 1) if step == 3 else [1]
        runner.input_ids = NS(gpu=torch.tensor(tokens))
        runner.seq_lens = torch.tensor([position + q])
        o.observe('prepared', runner, scheduled, runner.input_ids.gpu, runner.positions,
                  {0: torch.tensor([32 + p % 16 for p in range(position, position + q)])})
        for _ in range(28): fn(**production_tests.arguments(q))
        raw = torch.tensor([[0., 4., 2., 1.]] * q)
        spec = None if q == 1 else NS(num_draft_tokens=[drafts], logits_indices=torch.arange(q),
            target_logits_indices=torch.arange(drafts), bonus_logits_indices=torch.tensor([drafts]),
            cu_num_sampled_tokens=torch.tensor([q]))
        o.observe('begin', runner, scheduled, raw, spec, None)
        if drafts:
            o.observe('verify_scores', raw[:drafts], NS(all_greedy=True), spec)
            o.observe('sample_scores', raw[drafts:], NS(all_greedy=True))
        else:
            o.observe('sample_scores', raw, NS(all_greedy=True))
        selected = [1] * (offset + 1) if step == 3 else [1]
        o.observe('end', runner, NS(sampled_token_ids=torch.tensor([selected])))
        request = NS(request_id=rid, _output_token_ids=ids[:index], num_computed_tokens=position + q,
                     num_in_flight_tokens=1, num_output_placeholders=4)
        o.observe('account_before', request, selected, scheduled, False)
        request._output_token_ids.extend(selected)
        request.num_output_placeholders -= len(selected)
        o.observe('account_after', request, selected, scheduled, False)
    (directory / 'server.log').write_text('SYNTHETIC CPU CAPTURE\n')
    return prompt, ids


def complete_run(tmp_path, monkeypatch, q_len=3, offset=0):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    prepare_runtime(root, monkeypatch)
    def intervention(*_): pytest.fail('observation-only run invoked an intervention')
    for name in ('force_attention_3d', 'r14_before_schedule',
                 'kv_recompute_before_schedule', 'kv_recompute_after_schedule'):
        monkeypatch.setattr(o, name, intervention)
    identity = dict(pid=171)
    monkeypatch.setattr(e, 'started_process', lambda _: (identity, {}))
    monkeypatch.setattr(e, 'process_identity', lambda _: identity)
    for lane in c.LANES:
        directory = root / lane
        monkeypatch.delenv('M203_TARGETED_CONFIG', raising=False)
        argv, env = e.launch_args(root, lane)
        c.save(directory / 'launch.json', dict(execution_sha='d' * 40, argv=argv,
            environment=cc.environment_view(env),
            canonical_contract=cc.launch_evidence(lane, os.environ, env, str(c.ROOT))))
        monkeypatch.setenv('M203_TARGETED_CONFIG', str(directory / 'observer.json'))
        prompt, ids = observe_lane(directory, lane, q_len if lane == 'adaptive' else 1,
                                   offset if lane == 'adaptive' else 0)
        body = dict(id='api', prompt_token_ids=prompt, usage=dict(completion_tokens=len(ids)),
                    choices=[dict(token_ids=ids, finish_reason='stop')])
        sent = []
        def exchange(port, method, path, raw=None):
            if method == 'POST':
                sent.append(c.decode(raw))
                return 200, c.encoded(body)
            return 200, c.encoded({'data': [{'id': 'qwen2.5-7b-' + lane}]})
        monkeypatch.setattr(e, 'http_exchange', exchange)
        e.request(root, lane)
        expected = c.frozen_request(CASE)
        expected.update(model='qwen2.5-7b-' + lane, return_token_ids=True)
        assert sent == [expected]
        legacy.synthetic_shutdown(directory)
    return root


@pytest.mark.parametrize('q_len,offset', [(1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (5, 0)])
def test_exact_boundary_capture_and_frozen_payload(tmp_path, monkeypatch, q_len, offset):
    root = complete_run(tmp_path, monkeypatch, q_len, offset)
    report = a.compare(root)
    assert report['state'] == 'NOT_REPRODUCED', report
    assert report['case'] == CASE and report['target'] == report['observation_target'] == 81
    assert report['historical_gate'] == '44/54 FAIL'
    for lane in c.LANES:
        proof = report['captures'][lane]
        assert proof['row']['generated_index'] == 81 and proof['selected_token'] == 1
        assert set(proof['scores']) == {'raw', 'processed'}
        assert a.validate_lane(root, lane)['lane_valid'] is True
        current = next(c.load(p) for p in (root / lane / 'capture').glob('*.forward.json')
                       if c.load(p)['scores'])
        prep = current['preparation']
        count = q_len if lane == 'adaptive' else 1
        assert prep['scheduled_tokens'] == count and prep['scheduled_draft_count'] == count - 1
        assert prep['active_query'] == current['active_query']
        assert prep['resolved_cudagraph_mode'] == 'PIECEWISE'
        assert current['prefix_cache_enabled'] is True
        assert len(prep['groups'][0]['slot_mapping']) == count
        calls = proof['production_attention_dispatch']['event']['calls']
        assert len(calls) == 28
        assert all(call['max_seqlen_q'] == count and call['force_3d'] is False for call in calls)
        assert all(call['short_query_3d'] is call['guards_active'] is (count == 3) for call in calls)
        assert all(call['use_3d'] is (count in (1, 3)) for call in calls)
    snapshot = c.checkpoint(root, 'complete')
    bundle = e.export_snapshot(snapshot)
    assert bundle.is_file() and c.verify_checkpoint(snapshot)['partial'] is False
    assert a.compare(snapshot / 'data')['state'] == 'NOT_REPRODUCED'
    e.restore(root)
    assert e.restoration_status(root)['state'] == 'RESTORED'
    package = Path(c.load(root / 'preflight.json')['runtime']['vllm_root'])
    assert c.file_sha(package / s.ATTENTION_SOURCE) == s.PRODUCTION_ATTENTION_SHA256


@pytest.mark.parametrize('problem', ['target', 'case', 'raw', 'processed', 'event', 'slot', 'termination',
                                     'index_event', 'missing_forward', 'frame_case', 'runtime_mode',
                                     'scheduler', 'request', 'force'])
def test_case_index_or_missing_boundary_evidence_fails_closed(tmp_path, monkeypatch, problem):
    root = complete_run(tmp_path, monkeypatch)
    directory = root / 'adaptive'
    if problem in ('target', 'case'):
        path = directory / 'observer.json'
        config = c.load(path)
        if problem == 'target': config['observation_target'] = 82
        else: config['case'] = c.CASE
        path.write_bytes(c.encoded(config))
    elif problem == 'event':
        path = next(p for p in (directory / 'capture/events').glob('*.json')
                    if c.load(p)['kind'] == 'production_attention_dispatch')
        path.unlink()
    elif problem == 'index_event':
        path = next(p for p in (directory / 'capture/events').glob('*.json')
                    if c.load(p)['kind'] == 'production_attention_dispatch')
        event = c.load(path)
        event['generated_index'] = 82
        path.write_bytes(c.encoded(event))
    elif problem == 'missing_forward':
        next(p for p in (directory / 'capture').glob('*.forward.json') if c.load(p)['scores']).unlink()
    elif problem == 'termination':
        path = directory / 'response.json'
        body = c.load(path)
        body['choices'][0]['token_ids'] = body['choices'][0]['token_ids'][:81]
        body['usage']['completion_tokens'] = 81
        path.write_bytes(c.encoded(body))
        legacy.synthetic_shutdown(directory)
    else:
        path = next(p for p in (directory / 'capture').glob('*.forward.json') if c.load(p)['scores'])
        record = c.load(path)
        if problem == 'slot': record['preparation']['groups'][0]['slot_mapping'][0] += 1
        elif problem == 'frame_case': record['observation_spec']['case'] = c.CASE
        elif problem == 'runtime_mode': del record['preparation']['resolved_cudagraph_mode']
        elif problem == 'scheduler': record['scheduler_step'] += 1
        elif problem == 'request': record['request_id'] = 'foreign-request'
        elif problem == 'force': record['preparation']['production_attention_dispatch']['force_3d'] = True
        else: del record['scores']['0'][problem]
        path.write_bytes(c.encoded(record))
    assert a.compare(root)['state'] != 'NOT_REPRODUCED'
    with pytest.raises((c.Invalid, KeyError, IndexError)):
        a.validate_lane(root, 'adaptive')


@pytest.mark.parametrize('location', ['run', 'observer'])
def test_selection_drift_is_rejected_before_http_or_budget_spend(tmp_path, monkeypatch, location):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    prepare_runtime(root, monkeypatch)
    path = root / ('run.json' if location == 'run' else 'adaptive/observer.json')
    config = c.load(path)
    config['observation_target'] = 82
    path.write_bytes(c.encoded(config))
    identity = dict(pid=171)
    monkeypatch.setattr(e, 'started_process', lambda _: (identity, {}))
    monkeypatch.setattr(e, 'process_identity', lambda _: identity)
    monkeypatch.setattr(e, 'http_exchange', lambda *args: pytest.fail('HTTP reached after selection drift'))
    with pytest.raises(c.Invalid): e.request(root, 'adaptive')
    assert not (root / 'adaptive/request.claim').exists()


@pytest.mark.parametrize('case,target', sorted(c.OBSERVATION_PAIRS))
def test_only_reviewed_pairs_and_legacy_explicit_policy(tmp_path, monkeypatch, case, target):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=case, observation_index=target)
    config = c.load(root / 'run.json')
    assert c.observation_spec(config)['target'] == target
    expected = {} if case == CASE else dict(vanilla_cudagraph_mode='PIECEWISE', adaptive_cudagraph_mode='NONE')
    assert config['policy']['diagnostic_control'] == expected
    for lane in c.LANES:
        argv, env = e.launch_args(root, lane)
        assert ('--compilation-config' in argv) is (case == c.CASE)
        if case == c.CASE:
            assert env['VLLM_USE_V2_MODEL_RUNNER'] == '0'
        assert o.kv_recompute_target(c.load(root / lane / 'observer.json')) is None


@pytest.mark.parametrize('lane,script', [('vanilla', 'launch_lane1_vanilla.sh'),
                                         ('adaptive', 'launch_lane3_adaptive.sh')])
def test_new_target_matches_canonical_launcher_semantics(tmp_path, monkeypatch, lane, script):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    policy = c.load(root / 'run.json')['policy']
    text = (c.ROOT / 'scripts' / script).read_text(encoding='utf-8')
    command = text[text.index('$VLLM_CMD Qwen/'):].replace('\\\n', '')
    if lane == 'adaptive':
        spec = re.search(r"SPEC_CONFIG='([^']+)'", text).group(1)
        command = command.replace('"$SPEC_CONFIG"', shlex.quote(spec))
    canonical = shlex.split(command)[2:]
    argv, env = e.launch_args(root, lane)
    actual = argv[5:]
    if lane == 'adaptive':
        actual = list(actual)
        i = actual.index('--speculative-config') + 1
        spec = c.decode(actual[i])
        spec['model'] = policy['models']['draft']['id']
        actual[i] = c.encoded(spec).decode().strip()
        canonical[canonical.index('--speculative-config') + 1] = actual[i]
    assert actual == canonical
    assert 'VLLM_USE_V2_MODEL_RUNNER' not in env
    if lane == 'adaptive':
        for key, value in policy['adaptive_env'].items():
            assert env[key] == value


@pytest.mark.parametrize('problem', ['request_hash', 'frozen_request', 'frozen_workload',
                                     'reconstruction', 'cuda_override'])
def test_frozen_hash_and_control_drift_fail_closed(tmp_path, monkeypatch, problem):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    config = c.load(root / 'adaptive/observer.json')
    if problem == 'request_hash':
        config['request_sha256'] = '0' * 64
    elif problem in ('reconstruction', 'cuda_override'):
        config['diagnostic_control'] = ({'adaptive_kv_recompute_output_count': 60}
                                        if problem == 'reconstruction' else {'adaptive_cudagraph_mode': 'NONE'})
    elif problem == 'frozen_workload':
        monkeypatch.setattr(c, 'blob', lambda _: b'{}\n')
    else:
        monkeypatch.setitem(c.OBSERVATION_CASES, CASE, '0' * 64)
    with pytest.raises(c.Invalid):
        c.observation_spec(config)
        c.frozen_request(CASE)


@pytest.mark.parametrize('lane', c.LANES)
def test_canonical_environment_only_changes_enumerated_instrumentation(tmp_path, monkeypatch, lane):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    monkeypatch.setenv('PYTHONPATH', '/existing/modules:/other/modules')
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '0')
    monkeypatch.setenv('UNRELATED_SETTING', 'preserved')
    inherited = dict(os.environ)
    argv, env = e.launch_args(root, lane)
    assert 'VLLM_USE_V2_MODEL_RUNNER' not in env
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '0'
    assert env['UNRELATED_SETTING'] == 'preserved'
    assert env['PYTHONPATH'] == str(c.ROOT) + ':/existing/modules:/other/modules'
    baseline = cc.script_environment(lane, inherited, str(c.ROOT))
    changes = {k for k in env if env[k] != baseline.get(k)}
    assert changes == ({'M203_TARGETED_CONFIG', 'PYTHONPATH'} if lane == 'vanilla'
                       else {'M203_TARGETED_CONFIG', 'ADAPTIVE_TELEMETRY_PATH'})
    assert set(cc.INSTRUMENTATION_ENV) == {'M203_TARGETED_CONFIG', 'PYTHONPATH', 'ADAPTIVE_TELEMETRY_PATH'}
    assert set(baseline) <= set(env)
    for key in set(env) - set(cc.INSTRUMENTATION_ENV):
        assert env[key] == baseline[key]
    proof = cc.launch_evidence(lane, inherited, env, str(c.ROOT))
    cc.verify_launch(lane, argv, cc.environment_view(env), proof,
                    c.load(root / 'run.json')['policy']['models'], c.load(root / lane / 'observer.json'))


@pytest.mark.parametrize('value', ['0', '1', ''])
@pytest.mark.parametrize('lane', c.LANES)
def test_inherited_runner_override_is_rejected_not_rewritten(tmp_path, monkeypatch, lane, value):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', value)
    with pytest.raises(c.Invalid, match='must remain unset'): e.launch_args(root, lane)
    assert os.environ['VLLM_USE_V2_MODEL_RUNNER'] == value


@pytest.mark.parametrize('lane', c.LANES)
def test_canonical_observation_does_not_add_offline_or_device_settings(tmp_path, monkeypatch, lane):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        monkeypatch.delenv(key, raising=False)
    _, env = e.launch_args(root, lane)
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'VLLM_USE_V2_MODEL_RUNNER'):
        assert key not in env


@pytest.mark.parametrize('problem', ['argv', 'runner', 'offline', 'device', 'extra_diagnostic'])
def test_source_contract_rejects_launch_drift_independently_of_builder(tmp_path, monkeypatch, problem):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    lane = 'adaptive'
    argv, env = e.launch_args(root, lane)
    proof = cc.launch_evidence(lane, os.environ, env, str(c.ROOT))
    env = cc.environment_view(env)
    if problem == 'argv': argv.append('--enforce-eager')
    elif problem == 'runner': env['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    elif problem == 'offline': env['HF_HUB_OFFLINE'] = '1'
    elif problem == 'device': env['CUDA_VISIBLE_DEVICES'] = '1'
    else: env['M203_R16_FORCE_3D'] = '1'
    # A self-consistent builder cannot authorize a departure from the shell source.
    monkeypatch.setattr(e, 'launch_args', lambda *_: (argv, env))
    with pytest.raises(c.Invalid):
        cc.verify_launch(lane, argv, env, proof, c.load(root / 'run.json')['policy']['models'],
                         c.load(root / lane / 'observer.json'))


def test_analyzer_rejects_coordinated_preflight_and_builder_argv_drift(tmp_path, monkeypatch):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    prepare_runtime(root, monkeypatch)
    preflight = c.load(root / 'preflight.json')
    launches = {}
    for lane in c.LANES:
        argv, env = e.launch_args(root, lane)
        argv.append('--enforce-eager')
        launches[lane] = argv, env
        preflight['launcher_argv'][lane] = argv
        c.save(root / lane / 'launch.json', dict(execution_sha='d' * 40, argv=argv,
            environment=cc.environment_view(env),
            canonical_contract=cc.launch_evidence(lane, os.environ, env, str(c.ROOT))))
    (root / 'preflight.json').write_bytes(c.encoded(preflight))
    monkeypatch.setattr(e, 'launch_args', lambda _, lane: launches[lane])
    with pytest.raises(c.Invalid, match='source contract'): a.provenance(root)


@pytest.mark.parametrize('lane', c.LANES)
def test_changed_canonical_launcher_source_fails_closed(tmp_path, monkeypatch, lane):
    source = c.ROOT / cc.SCRIPTS[lane][0]
    target = tmp_path / cc.SCRIPTS[lane][0]
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes() + b'\nexport VLLM_USE_V2_MODEL_RUNNER=0\n')
    monkeypatch.setattr(cc, 'ROOT', tmp_path)
    with pytest.raises(c.Invalid, match='source drift'): cc.contract(lane)


def test_canonical_startup_hook_is_observation_only_and_legacy_bytes_unchanged():
    name = 'v1/worker/gpu_model_runner.py'
    raw = s.base_sources(production=True)[name]
    legacy_bytes = s.transform(name, raw)
    observed = s.transform(name, raw, canonical=True)
    assert b'runner_initialized' not in legacy_bytes
    assert observed.count(b'_m203_observe("runner_initialized", self)') == 1
    tree = ast.parse(observed)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPUModelRunner')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assert ast.unparse(init.body[0]) == 'self.vllm_config = vllm_config'
    assert ast.unparse(init.body[1]) == "_m203_observe('runner_initialized', self)"
    assert ast.dump(s.StripObserver().visit(tree)) == ast.dump(ast.parse(raw))
    assert b'VLLM_USE_V2_MODEL_RUNNER' not in observed
    compile(observed, name, 'exec')


@pytest.mark.parametrize('problem', ['v2', 'unknown', 'missing_hook', 'missing_resolution'])
def test_unsupported_canonical_runtime_stops_before_http_or_claim(tmp_path, monkeypatch, problem):
    root = init_case(tmp_path / 'run', monkeypatch, observation_case=CASE, observation_index=81)
    prepare_runtime(root, monkeypatch)
    lane = 'vanilla'
    directory = root / lane
    monkeypatch.setenv('M203_TARGETED_CONFIG', str(directory / 'observer.json'))
    runner_class = type('GPUModelRunner', (NS,), {'__module__': 'vllm.v1.worker.gpu_model_runner'})
    if problem == 'unknown': runner_class = NS
    config = NS() if problem == 'missing_resolution' else NS(use_v2_model_runner=problem == 'v2')
    runner = runner_class(vllm_config=config)
    if problem != 'missing_hook': o.observe('runner_initialized', runner)
    assert not (directory / 'capture/runtime_identity.json').exists()
    identity = dict(pid=171)
    monkeypatch.setattr(e, 'started_process', lambda _: (identity, {}))
    monkeypatch.setattr(e, 'process_identity', lambda _: identity)
    monkeypatch.setattr(e, 'http_exchange', lambda *_: pytest.fail('HTTP after unsupported runtime'))
    with pytest.raises((c.Invalid, FileNotFoundError)): e.request(root, lane)
    assert not (directory / 'request.claim').exists()
    assert not hasattr(config, 'use_v2_model_runner') or config.use_v2_model_runner is (problem == 'v2')
    assert 'VLLM_USE_V2_MODEL_RUNNER' not in os.environ


@pytest.mark.parametrize('resolved', [True, 0, None, 'false'])
def test_runtime_evidence_requires_explicit_supported_boolean(resolved):
    config = dict(execution_sha='d' * 40, lane='vanilla')
    proof = dict(config, model_runner=cc.RUNNER, use_v2_model_runner=resolved)
    with pytest.raises(c.Invalid): cc.verify_runner(proof, config)
