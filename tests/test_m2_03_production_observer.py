"""CPU-only production observer compatibility; synthetic captures are not GPU evidence."""
import ast
from pathlib import Path

import pytest

from experiments.m2_03_targeted import analysis as a, common as c, execution as e, observer as o, sources as s
from src.server import triton_spec_decode_patch as production
import test_m2_03_targeted_gpu_diag as legacy
import test_triton_spec_decode_patch as production_tests


PROFILE = dict(production_attention_observation=True, production_patch_sha=s.PRODUCTION_COMMIT)


def observed_wrapper():
    fn, main, reducer = production_tests.wrapper()
    ns = fn.__globals__
    ns['_m203_observe'] = o.observe
    tree = ast.parse(s.transform(s.ATTENTION_SOURCE, s.production_attention_source()))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('unified_attention', '_get_tile_size', '_is_gemma3_attention')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<production-observer>', 'exec'), ns)
    return ns['unified_attention'], main, reducer


def test_exact_production_revision_and_observation_only_ast():
    assert c.sha(Path(production.__file__).read_bytes()) == s.PRODUCTION_MODULE_SHA256
    raw = production.transform(s.fixture_sources()[s.ATTENTION_SOURCE].encode())
    assert raw == s.production_attention_source()
    assert c.sha(raw) == s.PRODUCTION_ATTENTION_SHA256
    observed = s.transform(s.ATTENTION_SOURCE, raw)
    assert b'force_attention_3d' not in observed and b'_m203_control' not in observed
    assert b'_M203_Q3_GUARDS' not in observed and b'M203_R16' not in observed
    assert observed.count(b'_m203_observe(') == 1
    assert ast.dump(s.StripObserver().visit(ast.parse(observed))) == ast.dump(ast.parse(raw))
    compile(observed, s.ATTENTION_SOURCE, 'exec')


@pytest.mark.parametrize('mutation', [
    lambda raw: raw + b'\n', lambda raw: raw[:-1],
    lambda raw: raw.replace(b'max_seqlen_q == 3', b'max_seqlen_q <= 3'),
    lambda raw: raw.replace(b'_SHORT_QUERY_3D_GUARDS=short_query_3d', b'_SHORT_QUERY_3D_GUARDS=False'),
    lambda raw: s.transform(s.ATTENTION_SOURCE, raw),
], ids=['foreign', 'partial', 'routing-drift', 'arithmetic-drift', 'already-observed'])
def test_foreign_partial_or_already_instrumented_production_source_refused(mutation):
    raw = mutation(s.production_attention_source())
    with pytest.raises(c.Invalid): s.transform(s.ATTENTION_SOURCE, raw)


def test_production_module_drift_refused(monkeypatch):
    monkeypatch.setattr(s, 'PRODUCTION_MODULE_SHA256', '0' * 64)
    with pytest.raises(c.Invalid, match='production patch module'):
        s.production_attention_source()


@pytest.mark.parametrize('enabled', [False, True], ids=['pristine-diagnostics', 'production'])
def test_install_restore_preserves_exact_run_baseline(tmp_path, monkeypatch, enabled):
    run, package = tmp_path / 'run', tmp_path / 'vllm'
    options = {'production': True} if enabled else {}
    originals = s.base_sources(**options)
    for name, raw in originals.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    rt = dict(vllm_root=str(package))
    c.save(run / 'run.json', dict(diagnostic_execution_sha='d' * 40, **(PROFILE if enabled else {})))
    c.save(run / 'preflight.json', dict(runtime=rt, source_base_sha256=s.base_hashes(**options)))
    monkeypatch.setattr(e, 'runtime', lambda: rt)
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'free_port', lambda _: None)
    e.install(run)
    assert s.installed_state(package, **options)['mode'] == 'PATCHED'
    assert c.load(run / 'installed.json')['patched_sha256'] == {
        name: c.sha(s.transform(name, raw)) for name, raw in originals.items() if name in s.INSERTIONS}
    assert e.inference_sources(run)['runtime'] == rt
    assert (run / 'original_sources' / s.ATTENTION_SOURCE).read_bytes() == originals[s.ATTENTION_SOURCE]
    observed = (package / s.ATTENTION_SOURCE).read_bytes()
    assert (b'force_attention_3d' in observed) is (not enabled)
    if enabled:
        run_raw = (run / 'run.json').read_bytes()
        before = {name: (package / name).read_bytes() for name in originals}
        (run / 'run.json').unlink()
        with pytest.raises(c.Invalid, match='unknown source'): e.restore(run)
        assert not (run / 'restore_intent.json').exists()
        assert {name: (package / name).read_bytes() for name in originals} == before
        (run / 'run.json').write_bytes(run_raw)
        replace = e.os.replace
        calls = []
        def interrupted(src, dst):
            calls.append(dst)
            if len(calls) == 2: raise OSError('synthetic interrupted restoration')
            return replace(src, dst)
        with monkeypatch.context() as m:
            m.setattr(e.os, 'replace', interrupted)
            with pytest.raises(OSError, match='interrupted'): e.restore(run)
        assert e.restoration_status(run)['state'] == 'INCOMPLETE'
        assert not e.restoration_status(run)['inference_allowed']
        with pytest.raises(c.Invalid): e.inference_sources(run)
    e.restore(run)
    e.restore(run)
    assert e.restoration_status(run)['state'] == 'RESTORED'
    assert not e.restoration_status(run)['inference_allowed']
    assert {name: (package / name).read_bytes() for name in originals} == originals
    expected_sha = s.PRODUCTION_ATTENTION_SHA256 if enabled else s.manifest()['sha256'][s.ATTENTION_SOURCE]
    assert c.sha((package / s.ATTENTION_SOURCE).read_bytes()) == expected_sha
    assert c.load(run / 'restore_complete.json')['sha256'][s.ATTENTION_SOURCE] == expected_sha
    with pytest.raises(c.Invalid): e.inference_sources(run)


@pytest.mark.parametrize('problem', ['foreign', 'partial', 'wrong-profile', 'diagnostic'])
def test_install_refuses_unknown_or_wrong_profile_before_writes(tmp_path, monkeypatch, problem):
    run, package = tmp_path / 'run', tmp_path / 'vllm'
    raw_files = s.base_sources(production=True)
    if problem == 'foreign': raw_files[s.ATTENTION_SOURCE] += b'\n'
    if problem == 'partial': raw_files[s.ATTENTION_SOURCE] = raw_files[s.ATTENTION_SOURCE][:-20]
    if problem == 'wrong-profile': raw_files[s.ATTENTION_SOURCE] = s.fixture_sources()[s.ATTENTION_SOURCE].encode()
    if problem == 'diagnostic':
        raw_files[s.ATTENTION_SOURCE] = s.transform(s.ATTENTION_SOURCE, s.fixture_sources()[s.ATTENTION_SOURCE].encode())
    for name, raw in raw_files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    rt = dict(vllm_root=str(package))
    c.save(run / 'run.json', dict(diagnostic_execution_sha='d' * 40, **PROFILE))
    c.save(run / 'preflight.json', dict(runtime=rt))
    monkeypatch.setattr(e, 'runtime', lambda: rt)
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    with pytest.raises(c.Invalid): e.install(run)
    assert not (run / 'install.intent').exists()
    assert {name: (package / name).read_bytes() for name in raw_files} == raw_files


def test_init_selects_explicit_production_profile_and_preserves_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'blob', lambda path, _: (c.ROOT / path).read_bytes())
    policy = 'experiments/m2_03_targeted/runtime_policy.kaggle_t4_index60.json'
    old = e.init_run(tmp_path / 'old', 'd' * 40, policy)
    new = e.init_run(tmp_path / 'new', 'd' * 40, policy, production_attention_observation=True)
    assert s.profile_options(c.load(new / 'run.json')) == {'production': True}
    for lane in c.LANES:
        config = c.load(new / lane / 'observer.json')
        assert s.profile_options(config) == ({'production': True} if lane == 'adaptive' else {})
        assert not any(config.get(k) for k in ('r14_single_token_boundary', 'r15_attention_dispatch', 'r16_force_attention_3d'))
        assert e.launch_args(old, lane)[0] == e.launch_args(new, lane)[0]
    package = tmp_path / 'vllm'
    for name, raw in s.base_sources(production=True).items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    policy = c.load(new / 'run.json')['policy']
    rt = dict(policy['runtime'], environment=policy['runtime_environment'], vllm_root=str(package))
    monkeypatch.setattr(e, 'runtime', lambda: rt)
    monkeypatch.setattr(e, 'free_port', lambda _: None)
    monkeypatch.setattr(e, 'model_inventory', lambda _: policy['models'])
    e.preflight(new)
    assert c.load(new / 'preflight.json')['source_base_sha256'] == s.base_hashes(production=True)


@pytest.mark.parametrize('conflict', ['r14_single_token_boundary', 'r15_attention_dispatch', 'r16_force_attention_3d'])
def test_production_profile_refuses_diagnostic_controls(tmp_path, monkeypatch, conflict):
    monkeypatch.setattr(e, 'verify_execution', lambda _: None)
    monkeypatch.setattr(e, 'blob', lambda path, _: (c.ROOT / path).read_bytes())
    with pytest.raises(c.Invalid, match='diagnostic attention controls'):
        e.init_run(tmp_path / 'conflict', 'd' * 40,
                   'experiments/m2_03_targeted/runtime_policy.kaggle_t4_index60.json',
                   production_attention_observation=True, **{conflict: True})
    assert not (tmp_path / 'conflict').exists()


@pytest.mark.parametrize('q_len', [1, 2, 3])
def test_observation_captures_actual_production_routing_without_force(tmp_path, monkeypatch, q_len):
    monkeypatch.setenv('M203_TARGETED_CONFIG', str(tmp_path / 'must-not-be-read'))
    monkeypatch.setattr(o, 'load', lambda _: pytest.fail('pre-launch configuration I/O'))
    monkeypatch.setattr(o, 'control', lambda *args: pytest.fail('diagnostic control invoked'))
    capture = dict(key='boundary', expected_calls=1, force_3d=False, calls=[])
    o._local.production_attention_capture = capture
    fn, main, reducer = observed_wrapper()
    base, base_main, base_reducer = production_tests.wrapper()
    args = production_tests.arguments(q_len)
    try:
        fn(**args)
        base(**args)
        assert main.calls == base_main.calls and reducer.calls == base_reducer.calls
        record = capture['calls'][0]
        assert record['max_seqlen_q'] == q_len and record['num_seqs'] == 1
        assert record['short_query_3d'] is record['guards_active'] is (q_len == 3)
        assert record['force_3d'] is False and record['use_3d'] is (q_len != 2)
        assert record['grid'] == list(main.calls[0][0])
        assert record['num_segments'] == main.calls[0][1]['NUM_SEGMENTS_PER_SEQ']
    finally:
        o._local.production_attention_capture = None


def production_proof_fixture(monkeypatch):
    _, current, _, _ = legacy.transition_fixture()
    current.update(draft_count=2, decoding_path='speculative',
                   active_query=[[556, 47742], [557, 129831], [558, 128319]])
    prep = current['preparation']
    prep.update(scheduled_tokens=3, scheduled_draft_count=2, query_start_loc=[0, 3])
    capture = dict(key=current['key'], expected_calls=28, force_3d=False, calls=[])
    prep['production_attention_dispatch'] = capture
    monkeypatch.setenv('M203_TARGETED_CONFIG', 'not-read')
    fn, _, _ = observed_wrapper()
    o._local.production_attention_capture = capture
    try:
        for _ in range(28): fn(**production_tests.arguments())
    finally:
        o._local.production_attention_capture = None
    event = {k: current[k] for k in ('execution_sha', 'lane', 'request_id', 'key', 'scheduler_step', 'forward_sequence')}
    event.update(kind='production_attention_dispatch', generated_index=368,
                 expected_calls=28, calls=capture['calls'])
    return current, event


def test_q3_analysis_proves_all_28_natural_guarded_calls(monkeypatch):
    current, event = production_proof_fixture(monkeypatch)
    proof = a.production_attention_proof([event], current, True)
    assert proof['mode'] == '3D' and proof['force_3d'] is False and proof['guards_active'] is True
    assert proof['production_patch_sha'] == s.PRODUCTION_COMMIT


@pytest.mark.parametrize('field', ['force_3d', 'short_query_3d', 'use_3d', 'guards_active'])
def test_q3_analysis_refuses_intervention_or_routing_drift(monkeypatch, field):
    current, event = production_proof_fixture(monkeypatch)
    event['calls'][0][field] ^= True
    with pytest.raises(c.Invalid): a.production_attention_proof([event], current, True)


def test_production_analysis_requires_event_and_all_calls(monkeypatch):
    current, event = production_proof_fixture(monkeypatch)
    with pytest.raises(c.Invalid): a.production_attention_proof([], current, True)
    with pytest.raises(c.Invalid): a.production_attention_proof([event, event], current, True)
    event['calls'].pop()
    with pytest.raises(c.Invalid): a.production_attention_proof([event], current, True)


def test_prepared_and_begin_scope_production_capture(monkeypatch):
    import torch
    from types import SimpleNamespace as NS
    prep = dict(key='p', request_id='api', scheduled_tokens=3, correction_enabled=False)
    o._local.preparation = prep
    runner = NS(input_batch=NS(req_ids=['api'], block_table=NS(block_tables=[])),
                num_computed_tokens=torch.tensor([556]), query_start_loc=NS(gpu=torch.tensor([0, 3])),
                seq_lens=torch.tensor([559]), model_config=NS(hf_config=NS(num_hidden_layers=28)))
    config = dict(PROFILE, lane='adaptive')
    o.prepared(config, runner, NS(total_num_scheduled_tokens=3),
               torch.tensor([47742, 129831, 128319]), torch.tensor([556, 557, 558]), {})
    assert o._local.production_attention_capture is prep['production_attention_dispatch']
    assert o._local.production_attention_capture['force_3d'] is False
    # A warmup begin still clears capture before sampling/drafter work.
    runner.input_batch.num_reqs = 1
    runner.input_batch.req_ids = ['_warmup_1']
    o.begin({}, runner, None, None, None, None)
    assert o._local.production_attention_capture is None
    o._local.preparation = None


def test_finish_transition_publishes_linked_production_event(tmp_path, monkeypatch):
    current, event = production_proof_fixture(monkeypatch)
    config = dict(PROFILE, lane='adaptive', execution_sha=current['execution_sha'], directory=str(tmp_path))
    o._local.transition_previous = dict(key='previous', scores={})
    o._local.transition_published = False
    o._local.pending_vectors = {}
    o.finish_transition(config, current)
    events = [c.load(p) for p in (tmp_path / 'events').glob('*.json')]
    assert len(events) == 1 and events[0]['kind'] == 'production_attention_dispatch'
    assert events[0]['calls'] == event['calls']
    restored = c.load(tmp_path / (current['key'] + '.forward.json'))
    assert a.production_attention_proof(events, restored, True)['force_3d'] is False
    assert 'attention_dispatch' not in restored['preparation']
    o._local.transition_previous = None
    o._local.transition_published = False


def test_production_profile_validates_complete_synthetic_two_lane_capture(tmp_path, monkeypatch):
    """Re-label a private R16 copy with observed CPU launches; never GPU evidence."""
    import tarfile
    archive = c.ROOT / 'experiments/runs/m2_03/diagnostics/complete-051f50ab905f4b36ad0054ec30017cf1.tar.gz'
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.startswith('data/'): continue
            relative = Path(member.name).relative_to('data')
            assert not relative.is_absolute() and '..' not in relative.parts
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    for path in (tmp_path / 'run.json', tmp_path / 'adaptive/observer.json'):
        config = c.load(path)
        config.pop('r15_attention_dispatch')
        config.pop('r16_force_attention_3d')
        path.write_bytes(c.encoded(dict(config, **PROFILE)))
    preflight = c.load(tmp_path / 'preflight.json')
    preflight['source_base_sha256'] = s.base_hashes(production=True)
    (tmp_path / 'preflight.json').write_bytes(c.encoded(preflight))
    installed = c.load(tmp_path / 'installed.json')
    installed['patched_sha256'][s.ATTENTION_SOURCE] = c.sha(
        s.transform(s.ATTENTION_SOURCE, s.production_attention_source()))
    (tmp_path / 'installed.json').write_bytes(c.encoded(installed))
    path = next(p for p in (tmp_path / 'adaptive/capture').glob('*.forward.json')
                if 'attention_dispatch' in c.load(p)['preparation'])
    current = c.load(path)
    current['preparation'].pop('attention_dispatch')
    capture = dict(key=current['key'], expected_calls=28, force_3d=False, calls=[])
    current['preparation']['production_attention_dispatch'] = capture
    monkeypatch.setenv('M203_TARGETED_CONFIG', 'not-read')
    fn, _, _ = observed_wrapper()
    o._local.production_attention_capture = capture
    try:
        for _ in range(28): fn(**production_tests.arguments())
    finally:
        o._local.production_attention_capture = None
    path.write_bytes(c.encoded(current))
    event_path = next(p for p in (tmp_path / 'adaptive/capture/events').glob('*.json')
                      if c.load(p)['kind'] == 'attention_dispatch')
    event = c.load(event_path)
    event.update(kind='production_attention_dispatch', calls=capture['calls'])
    event_path.write_bytes(c.encoded(event))
    report = a.compare(tmp_path)
    assert report['state'] == 'NOT_REPRODUCED', report
    proof = report['captures']['adaptive']['production_attention_dispatch']
    assert proof['mode'] == '3D' and proof['force_3d'] is False
    assert a.validate_lane(tmp_path, 'adaptive')['lane_valid'] is True
    assert 'attention_dispatch_causal_test' not in report
    assert report['historical_gate'] == '47/54 FAIL'
    # A production run cannot borrow the legacy pristine-source profile.
    preflight['source_base_sha256'] = s.manifest()['sha256']
    (tmp_path / 'preflight.json').write_bytes(c.encoded(preflight))
    assert a.compare(tmp_path)['state'] == 'INVALID_CAPTURE'
