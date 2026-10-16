"""Pinned-source and CPU-only V2 tests. No vLLM import, server, or GPU work."""
import ast
import os
import gc
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from experiments.m2_03_targeted import (common as c, canonical as cc, sources as s,
    execution as e, analysis as a, observer as o, v2_sources as vs, v2_observer as v,
    v2_analysis as va)
import test_m2_03_configurable_observation as focused
import test_m2_03_production_observer as production
import test_triton_spec_decode_patch as kernels
import test_m2_03_targeted_gpu_diag as legacy


def config_at(root, lane='adaptive'):
    config = dict(lane=lane, execution_sha='d' * 40, directory=str(root / 'capture'),
        observation_only=True, case=focused.CASE, observation_target=81,
        request_sha256=c.OBSERVATION_CASES[focused.CASE], diagnostic_control={},
        production_attention_observation=True, production_patch_sha=s.PRODUCTION_COMMIT)
    c.save(root / 'observer.json', config)
    run_path = root.parent / 'run.json'
    if not run_path.exists():
        c.save(run_path, dict(config))
    return config


def desc(q=3, mode='FULL', padded=None, max_q=None, requests=1):
    return NS(cg_mode=NS(name=mode), num_tokens=padded or q, num_reqs=requests if mode == 'FULL' else None,
        uniform_token_count=None, max_query_len=max_q or q, num_active_loras=0, num_ubatches=1)


class Manager:
    def __init__(self, config):
        self.vllm_config = config
        self.graphs = {}


class Graph:
    pass


class Desc(NS):
    def __hash__(self): return hash(tuple(sorted(v.descriptor(self).items())))


def cpu_runner(config, q=3, mode='FULL', padded=None, max_q=None, requests=1, offset=0):
    descriptor = Desc(**vars(desc(q, mode, padded, max_q, requests)))
    padded = descriptor.num_tokens
    prompt = [0] * 5
    start = 81 - offset
    position = len(prompt) + start - 1
    state_slot, rid = 7, 'api-1234abcd'
    pc = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
            decode_context_parallel_size=1)
    cfg = NS(use_v2_model_runner=True, parallel_config=pc,
        model_config=NS(hf_config=NS(num_hidden_layers=28)),
        compilation_config=NS(cudagraph_mode=NS(name='FULL_AND_PIECEWISE')),
        scheduler_config=NS(async_scheduling=True))
    lengths = np.zeros(8, dtype=int)
    lengths[state_slot] = len(prompt)
    tokens = torch.zeros((8, 512), dtype=torch.int64)
    num_blocks = np.zeros((1, 8), dtype=int)
    num_blocks[0, state_slot] = 32
    cls = type('GPUModelRunner', (NS,), {'__module__': 'vllm.v1.worker.gpu.model_runner'})
    runner = cls(vllm_config=cfg, req_states=NS(req_id_to_index={rid: state_slot}, num_reqs=1,
        prompt_len=NS(np=lengths), all_token_ids=NS(gpu=tokens)),
        batch_sharder=None, pcp_manager=None, adaptive_verification=None, ubatch_runner=None,
        cache_config=NS(enable_prefix_caching=True),
        block_tables=NS(num_blocks=NS(np=num_blocks), kernel_block_sizes=[16], blocks_per_kv_block=[1]),
        sampler=NS(sampling_states=NS(temperature=NS(np=np.zeros(8)))))
    runner.cudagraph_manager = Manager(cfg)
    batch = NS(num_reqs=1, req_ids=[rid], idx_mapping_np=np.array([state_slot]),
        num_computed_tokens_np=np.array([position]), has_prefill=False, has_structured_output_reqs=False,
        num_tokens=q, num_scheduled_tokens=np.array([q]), num_draft_tokens=q - 1,
        num_tokens_after_padding=padded, input_ids=torch.tensor([1] * q + [0] * (padded - q)),
        positions=torch.arange(position, position + padded),
        query_start_loc=torch.tensor([0] + [q] * requests),
        seq_lens=torch.tensor([position + q] + [0] * (requests - 1)),
        logits_indices=torch.arange(q), idx_mapping=torch.tensor([state_slot]),
        expanded_idx_mapping=torch.tensor([state_slot] * q), cu_num_logits=torch.tensor([0, q]))
    scheduled = NS(total_num_scheduled_tokens=q, num_scheduled_tokens={rid: q},
        scheduled_spec_decode_tokens={rid: [1] * (q - 1)},
        _m203_observation_provenance=dict(state='OBSERVATION_ONLY', scheduler_step=17))
    tables = [torch.tensor([[2] * 32] * requests)]
    slots = torch.tensor([[32 + p % 16 for p in range(position, position + q)] + [-1] * (padded - q)])
    metadata = {f'layer-{i}': NS(max_query_len=max_q or q,
        num_actual_tokens=q,
        query_start_loc=batch.query_start_loc, seq_lens=batch.seq_lens,
        block_table=tables[0], slot_mapping=slots[0]) for i in range(28)}
    return runner, batch, scheduled, descriptor, tables, slots, metadata, prompt


def attention_calls(fn, query, max_query=None, requests=1):
    args = kernels.arguments(query)
    args['max_seqlen_q'] = max_query or query
    if requests != 1:
        # Metadata-only tensors; the production wrapper runs, kernels are fakes.
        for name, shape in [('cu_seqlens_q', (requests + 1,)), ('seqused_k', (requests,)),
                            ('block_table', (requests, 40))]:
            old = args[name]
            args[name] = kernels.Metadata(shape, old.dtype, old.device)
    for _ in range(28): fn(**args)


def run_capture(root, monkeypatch, q=3, mode='FULL', padded=None, max_q=None, requests=1, offset=0):
    config_path = root / 'observer.json'
    config = c.load(config_path) if config_path.exists() else config_at(root)
    monkeypatch.setenv('M203_TARGETED_CONFIG', str(config_path))
    o._local.production_attention_capture = None
    data = cpu_runner(config, q, mode, padded, max_q, requests, offset)
    runner, batch, scheduled, descriptor, tables, slots, metadata, prompt = data
    v.observe('initialized', runner)
    fn, _, _ = production.observed_wrapper()
    manager, graph = runner.cudagraph_manager, Graph()
    if mode == 'FULL':
        captured_metadata = {name: NS(**dict(vars(m), num_actual_tokens=descriptor.num_tokens))
                             for name, m in metadata.items()}
        v.observe('graph_inputs', manager, descriptor,
                  dict(input_ids=batch.input_ids, positions=batch.positions), captured_metadata, False)
        v.observe('graph_begin', manager, descriptor, graph)
        attention_calls(fn, descriptor.num_tokens, descriptor.max_query_len, requests)
        manager.graphs[descriptor] = graph
        v.observe('graph_end', manager, descriptor, graph)
    v.observe('prepared', runner, scheduled, batch, descriptor, tables, slots, metadata, False)
    if mode == 'FULL': v.observe('graph_replay', manager, descriptor, graph)
    else: attention_calls(fn, batch.num_tokens, descriptor.max_query_len, requests)
    v.observe('forward_done', runner, batch, False)
    raw = torch.tensor([[0., 4., 2., 1.]] * q)
    v.observe('raw', runner, batch, raw, None)
    v.observe('processed', raw, batch.expanded_idx_mapping, batch.positions[:q])
    selected_count = offset + 1
    output = NS(sampled_token_ids=torch.tensor([[1] * selected_count + [-1] * (q - selected_count)]),
        num_sampled=torch.tensor([selected_count]), num_rejected=torch.tensor([q - selected_count]))
    v.observe('selected', runner, batch, output)
    start = 81 - offset
    ids = [1] * 90
    request = NS(request_id=batch.req_ids[0], _output_token_ids=ids[:start],
        num_computed_tokens=positions_end(batch), num_in_flight_tokens=1, num_output_placeholders=5)
    o.accounting(config, 'account_before', request, [1] * selected_count, scheduled, False)
    request._output_token_ids.extend([1] * selected_count)
    request.num_output_placeholders -= selected_count
    o.accounting(config, 'account_after', request, [1] * selected_count, scheduled, False)
    (root / 'server.log').write_text('SYNTHETIC CPU V2 CAPTURE\n')
    return data, ids


def positions_end(batch): return int(batch.positions[0]) + batch.num_tokens


@pytest.mark.parametrize('name', list(vs.INSERTIONS))
def test_pinned_v2_hooks_are_unique_observation_only_and_compile(name):
    raw = vs.references()[name]
    observed = vs.transform(name, raw)
    assert ast.dump(vs.Strip().visit(ast.parse(observed))) == ast.dump(ast.parse(raw))
    assert observed.count(b'_m203_v2(') == len(vs.INSERTIONS[name])
    for forbidden in (b'VLLM_USE_V2_MODEL_RUNNER', b'M203_R14', b'force_attention_3d'):
        assert forbidden not in observed
    compile(observed, name, 'exec')
    with pytest.raises(c.Invalid): vs.transform(name, raw + b'\n')
    with pytest.raises(c.Invalid): vs.transform(name, observed)


def test_graph_hooks_bracket_actual_capture_and_bind_exact_replay():
    text = vs.transform(vs.GRAPHS, vs.references()[vs.GRAPHS]).decode()
    assert text.index('_m203_v2("graph_begin"') < text.index('with torch.cuda.graph(')
    assert text.index('self.graphs[desc] = graph') < text.index('_m203_v2("graph_end"')
    assert text.index('_m203_v2("graph_replay"') < text.index('self.graphs[desc].replay()')
    assert s.transform(s.ATTENTION_SOURCE, s.production_attention_source(), canonical=True) == s.transform(
        s.ATTENTION_SOURCE, s.production_attention_source())


@pytest.mark.parametrize('q,mode,padded,max_q,requests,offset', [
    (1, 'FULL', 1, 1, 1, 0), (2, 'FULL', 2, 2, 1, 0),
    (1, 'FULL', 1, 5, 1, 0), (4, 'FULL', 4, 5, 4, 0), (5, 'FULL', 5, 5, 5, 0),
    (3, 'FULL', 3, 3, 1, 0), (3, 'FULL', 3, 3, 1, 1),
    (3, 'FULL', 4, 5, 4, 0), (5, 'FULL', 5, 5, 1, 0),
    (3, 'PIECEWISE', 3, 3, 1, 0), (3, 'PIECEWISE', 4, 3, 1, 0), (3, 'NONE', 3, 3, 1, 0),
])
def test_exact_v2_q_length_routing_and_gpu_row_linkage(tmp_path, monkeypatch, q, mode, padded, max_q, requests, offset):
    data, ids = run_capture(tmp_path, monkeypatch, q, mode, padded, max_q, requests, offset)
    runner, batch, *_ = data
    proof = va.capture_v2(tmp_path, 'adaptive', 'd' * 40, batch.req_ids[0], data[-1], ids)
    assert proof['row']['generated_index'] == 81 and proof['selected_token'] == 1
    assert proof['active_query_token_count'] == q
    assert proof['logical_runtime_q'] == q
    assert proof['captured_graph_q_shape'] == (padded if mode == 'FULL' else None)
    assert proof['max_seqlen_q'] == [max_q] * 28
    attention = proof['production_attention_dispatch']
    short = max_q == 3 and requests == 1 and (padded == 3 or mode != 'FULL')
    assert attention['mode'] == ('3D' if max_q == 1 or short else '2D')
    assert attention['max_seqlen_q'] == [max_q] * 28
    assert attention['force_3d'] is False
    assert set(proof['scores']) == {'raw', 'processed'}
    assert runner.vllm_config.use_v2_model_runner is True
    assert runner.cache_config.enable_prefix_caching is True
    assert runner.vllm_config.compilation_config.cudagraph_mode.name == 'FULL_AND_PIECEWISE'
    assert all(call['force_3d'] is False for call in attention['event']['dispatch']['calls'])


@pytest.mark.parametrize('problem', ['slot', 'index', 'scheduler', 'request', 'logits', 'event',
    'graph_hash', 'graph_descriptor', 'graph_calls', 'force', 'missing_frame', 'identity', 'prefix_cache', 'buffer'])
def test_ambiguous_or_incomplete_v2_evidence_fails_closed(tmp_path, monkeypatch, problem):
    data, ids = run_capture(tmp_path, monkeypatch)
    path = next((tmp_path / 'capture').glob('*.v2-forward.json'))
    frame = c.load(path)
    if problem == 'slot': frame['groups'][0]['slot_mapping'][0] += 1
    elif problem == 'index': frame['rows'][0]['generated_index'] = 82
    elif problem == 'scheduler': frame['scheduler_step'] += 1
    elif problem == 'request': frame['request_id'] = 'other-request'
    elif problem == 'logits': del frame['scores']['0']['processed']
    elif problem == 'event':
        next(p for p in (tmp_path / 'capture/events').glob('*.json')
             if c.load(p)['kind'] == 'v2_production_attention_dispatch').unlink()
    elif problem == 'graph_hash': frame['graph_reference']['sha256'] = '0' * 64
    elif problem == 'graph_descriptor': frame['descriptor']['num_tokens'] += 1
    elif problem == 'graph_calls': frame['dispatch']['calls'].pop()
    elif problem == 'force': frame['dispatch']['calls'][0]['force_3d'] = True
    elif problem == 'identity': frame['runtime_identity']['use_v2_model_runner'] = False
    elif problem == 'prefix_cache': frame['prefix_cache_enabled'] = False
    elif problem == 'buffer': frame['buffer_signature']['inputs']['input_ids']['pointer'] += 1
    if problem == 'missing_frame': path.unlink()
    else: path.write_bytes(c.encoded(frame))
    with pytest.raises((c.Invalid, KeyError, IndexError)):
        va.capture_v2(tmp_path, 'adaptive', 'd' * 40, data[1].req_ids[0], data[-1], ids)


def test_v2_fullgraph_rejects_replaced_graph_object(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    monkeypatch.setenv('M203_TARGETED_CONFIG', str(tmp_path / 'observer.json'))
    data = cpu_runner(config)
    runner, batch, scheduled, descriptor, tables, slots, metadata, _ = data
    v.observe('initialized', runner)
    manager, graph = runner.cudagraph_manager, Graph()
    o._local.production_attention_capture = None
    v.observe('graph_inputs', manager, descriptor,
              dict(input_ids=batch.input_ids, positions=batch.positions), metadata, False)
    v.observe('graph_begin', manager, descriptor, graph)
    fn, _, _ = production.observed_wrapper()
    attention_calls(fn, 3)
    manager.graphs[descriptor] = graph
    v.observe('graph_end', manager, descriptor, graph)
    v.observe('prepared', runner, scheduled, batch, descriptor, tables, slots, metadata, False)
    replacement = Graph()
    manager.graphs[descriptor] = replacement
    v.observe('graph_replay', manager, descriptor, replacement)
    assert any(c.load(p)['kind'] == 'observer_error' for p in (tmp_path / 'capture/events').glob('*.json'))
    assert not list((tmp_path / 'capture').glob('*.v2-forward.json'))


def test_graph_observation_does_not_keep_released_graphs_alive(tmp_path, monkeypatch):
    data, _ = run_capture(tmp_path, monkeypatch)
    manager = data[0].cudagraph_manager
    descriptor = data[3]
    ref = weakref.ref(manager.graphs[descriptor])
    manager.graphs.clear()
    gc.collect()
    assert ref() is None
    assert manager._m203_v2_observations[descriptor][0]() is None


def test_same_descriptor_with_reallocated_buffers_is_rejected(tmp_path, monkeypatch):
    data, _ = run_capture(tmp_path, monkeypatch)
    runner, batch, scheduled, descriptor, tables, slots, metadata, _ = data
    # The descriptor still matches; the graph reads the old allocation.
    batch.input_ids = batch.input_ids.clone()
    v.observe('prepared', runner, scheduled, batch, descriptor, tables, slots, metadata, False)
    manager = runner.cudagraph_manager
    v.observe('graph_replay', manager, descriptor, manager.graphs[descriptor])
    errors = [c.load(p) for p in (tmp_path / 'capture/events').glob('*.json')
              if c.load(p)['kind'] == 'observer_error']
    assert any('different input/KV buffers' in e['error'] for e in errors)


def test_gpu_index_not_reached_does_not_publish_a_boundary(tmp_path, monkeypatch):
    original = cpu_runner
    def earlier_position(*args, **kwargs):
        data = original(*args, **kwargs)
        data[1].positions.sub_(1)
        data[1].seq_lens.sub_(1)
        return data
    monkeypatch.setattr(__import__(__name__, fromlist=['cpu_runner']), 'cpu_runner', earlier_position)
    data, ids = run_capture(tmp_path, monkeypatch)
    assert not list((tmp_path / 'capture').glob('*.v2-forward.json'))
    with pytest.raises(c.Invalid, match='boundary missing'):
        va.capture_v2(tmp_path, 'adaptive', 'd' * 40, data[1].req_ids[0], data[-1], ids)


def test_v2_profile_is_confined_to_the_canonical_pair():
    assert vs.RUNNER not in s.base_sources(production=True)
    assert vs.RUNNER not in s.base_sources(production=True, canonical=True)
    config = dict(observation_only=True, case=focused.CASE, observation_target=81,
        request_sha256=c.OBSERVATION_CASES[focused.CASE], production_attention_observation=True,
        production_patch_sha=s.PRODUCTION_COMMIT, diagnostic_control={})
    assert c.v2_observation_profile(config) is True
    profile = s.profile_options(config)
    assert profile == {'production': True, 'canonical': True, 'v2': True}
    required_v2 = {'v1/worker/gpu/model_runner.py', 'v1/worker/gpu/cudagraph_utils.py',
                   'v1/worker/gpu/sample/sampler.py', 'v1/worker/gpu/spec_decode/rejection_sampler.py'}
    assert required_v2 <= set(s.base_sources(**profile))
    assert required_v2 <= set(s.patch_names(v2=profile.get('v2', False)))
    legacy_config = dict(observation_only=True, case=c.CASE, observation_target=368,
        request_sha256=c.REQUEST_HASH, production_attention_observation=True,
        production_patch_sha=s.PRODUCTION_COMMIT, diagnostic_control=c.observation_controls(c.CASE))
    assert s.profile_options(legacy_config) == {'production': True, 'canonical': True}
    assert vs.RUNNER not in s.patch_names()
    assert s.patch_names(canonical=True) == s.patch_names()


# Golden source profile from committed pre-V2 sources.py at
# 60045f38ec6404d6cc3ab592d17ab403ad54a767. These are independent of current
# inventory/transform implementations. Patches use canonical=True, including
# the pre-existing V1 runner_initialized hook; V2 inventory is excluded.
LEGACY_BASE = {
    'v1/attention/ops/triton_unified_attention.py': '1dbfc2c1efb4012cc9bb801a89c2dbe355e4539a1c54599edc6e8a9a7db73087',
    'v1/core/sched/async_scheduler.py': 'e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508',
    'v1/core/sched/scheduler.py': 'c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884',
    'v1/sample/rejection_sampler.py': '0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750',
    'v1/sample/sampler.py': 'f95805b2c6d06218c724826aa5f881a606e8d0a534a566aee7c85948520dd216',
    'v1/worker/gpu_model_runner.py': '87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51',
}
LEGACY_PATCHES = {
    'v1/attention/ops/triton_unified_attention.py': '9ce4a8a8f8f06db062aec23d59078c7f19eaddf0996b33bd9953f97a044eabc0',
    'v1/worker/gpu_model_runner.py': 'c7e69067b5a0de0b3cd4bd5c69618f48d00d1121c3b69bc554bd2781747da1a0',
    'v1/sample/sampler.py': '88d06be9a235cec090bbb2dcf112aaec343f2d067a672f291f3f47b47eb79a8d',
    'v1/sample/rejection_sampler.py': '1f50cee6ef58b34c1d2d0d3e82104dbdb7a6f96de16642a444bceb4e4daebf61',
    'v1/core/sched/scheduler.py': '59ba2d1425257819fdac297c235e2c4403a1b7880182a97260d219c37a3ad145',
}


def test_legacy_003_profile_install_restore_matches_pre_v2_bytes(tmp_path, monkeypatch):
    root = focused.init_case(tmp_path / 'run', monkeypatch,
                             observation_case=c.CASE, observation_index=368)
    config = c.load(root / 'run.json')
    assert c.v2_observation_profile(config) is False
    profile = s.profile_options(config)
    assert profile == {'production': True, 'canonical': True}
    base = s.base_sources(**profile)
    assert {name: c.sha(raw) for name, raw in base.items()} == LEGACY_BASE
    assert s.base_hashes(**profile) == LEGACY_BASE
    assert s.patch_names() == list(LEGACY_PATCHES)
    assert s.patch_names(canonical=True) == list(LEGACY_PATCHES)
    assert {name: c.sha(s.transform(name, base[name], canonical=True))
            for name in s.patch_names()} == LEGACY_PATCHES
    runner_name = 'v1/worker/gpu_model_runner.py'
    canonical_runner = s.transform(runner_name, base[runner_name], canonical=True)
    assert canonical_runner.count(b'_m203_observe("runner_initialized", self)') == 1
    assert b'_m203_v2' not in canonical_runner
    package = focused.prepare_runtime(root, monkeypatch)
    assert c.load(root / 'preflight.json')['source_base_sha256'] == LEGACY_BASE
    assert c.load(root / 'installed.json')['patched_sha256'] == LEGACY_PATCHES
    status = s.installed_state(package, **profile)
    assert status['mode'] == 'PATCHED' and set(status['files']) == set(LEGACY_BASE)
    assert set(p.relative_to(package).as_posix() for p in package.rglob('*.py')) == set(LEGACY_BASE)
    assert (package / runner_name).read_bytes() == canonical_runner
    for lane in c.LANES:
        argv, env = e.launch_args(root, lane)
        assert env['VLLM_USE_V2_MODEL_RUNNER'] == '0'
        assert c.decode(argv[argv.index('--compilation-config') + 1]) == {
            'cudagraph_mode': 'PIECEWISE' if lane == 'vanilla' else 'NONE'}
        c.save(root / lane / 'launch.json', dict(execution_sha='d' * 40, argv=argv,
            environment=cc.environment_view(env)))
    assert a.provenance(root) == 'd' * 40
    for lane in c.LANES:
        monkeypatch.setenv('M203_TARGETED_CONFIG', str(root / lane / 'observer.json'))
        prompt, ids = focused.observe_lane(root / lane, lane, q_len=1 if lane == 'vanilla' else 3)
        identity = c.load(root / lane / 'capture/runtime_identity.json')
        assert identity == dict(model_runner=cc.RUNNER, use_v2_model_runner=False,
                               execution_sha='d' * 40, lane=lane)
        proof = a.capture_transition(root / lane, lane, 'd' * 40, 'api-1234abcd', prompt, ids)
        current = proof['preparation'][proof['record']]
        assert current['runtime_identity'] == dict(model_runner=cc.RUNNER, use_v2_model_runner=False)
        assert current['resolved_cudagraph_mode'] == ('PIECEWISE' if lane == 'vanilla' else 'NONE')
        assert proof['row']['generated_index'] == 368
    e.restore(root)
    assert e.restoration_status(root)['state'] == 'RESTORED'
    assert {name: (package / name).read_bytes() for name in base} == base


def test_canonical_profile_does_not_depend_on_v2_selection(tmp_path, monkeypatch):
    config = config_at(tmp_path / 'adaptive')
    assert c.canonical_observation_profile(config) is True
    monkeypatch.setattr(s, 'v2_observation_profile', lambda _: False)
    assert s.profile_options(config) == {'production': True, 'canonical': True}


@pytest.mark.parametrize('change', [dict(observation_target=82), dict(target=82),
    dict(request_sha256='0' * 64), dict(observation_only=1),
    dict(case='w3_business_v2_007'), dict(diagnostic_control={'adaptive_cudagraph_mode': 'NONE'})])
def test_v2_predicate_validates_the_reviewed_spec_before_selection(tmp_path, change):
    config = config_at(tmp_path / 'adaptive')
    config.update(change)
    with pytest.raises(c.Invalid): c.v2_observation_profile(config)
    with pytest.raises(c.Invalid): s.profile_options(config)


@pytest.mark.parametrize('problem', ['legacy_run', 'legacy_observer', 'not_observation_only', 'wrong_target', 'missing_run'])
def test_v2_identity_alone_cannot_select_the_analyzer(tmp_path, monkeypatch, problem):
    directory = tmp_path / 'adaptive'
    config_at(directory)
    legacy_config = dict(observation_only=True, case=c.CASE, observation_target=368,
        request_sha256=c.REQUEST_HASH, production_attention_observation=True,
        production_patch_sha=s.PRODUCTION_COMMIT, diagnostic_control=c.observation_controls(c.CASE))
    run_path = directory.parent / 'run.json'
    if problem == 'legacy_run': run_path.write_bytes(c.encoded(legacy_config))
    elif problem == 'legacy_observer': (directory / 'observer.json').write_bytes(c.encoded(legacy_config))
    elif problem == 'missing_run': run_path.unlink()
    else:
        run = c.load(run_path)
        if problem == 'not_observation_only': run['observation_only'] = False
        else: run['observation_target'] = 82
        run_path.write_bytes(c.encoded(run))
    c.save(directory / 'capture/runtime_identity.json', dict(model_runner=cc.V2_RUNNER,
        use_v2_model_runner=True, execution_sha='d' * 40, lane='adaptive'))
    original = va.capture_v2
    monkeypatch.setattr(va, 'capture_v2', lambda *_: pytest.fail('V2 analyzer selected from identity alone'))
    with pytest.raises((c.Invalid, FileNotFoundError)):
        a.capture(directory, 'adaptive', 'd' * 40, 'api-1234abcd', [0] * 5, [1] * 90)
    # The public V2 analyzer entry also checks the run, independently of dispatch.
    with pytest.raises((c.Invalid, FileNotFoundError)):
        original(directory, 'adaptive', 'd' * 40, 'api-1234abcd', [0] * 5, [1] * 90)


@pytest.mark.parametrize('varlen', [False, True])
def test_canonical_v2_full_pipeline_restore_export_and_analysis(tmp_path, monkeypatch, varlen):
    root = focused.init_case(tmp_path / 'run', monkeypatch, observation_case=focused.CASE, observation_index=81)
    package = focused.prepare_runtime(root, monkeypatch)
    identity = dict(pid=171)
    monkeypatch.setattr(e, 'started_process', lambda _: (identity, {}))
    monkeypatch.setattr(e, 'process_identity', lambda _: identity)
    for lane in c.LANES:
        monkeypatch.delenv('M203_TARGETED_CONFIG', raising=False)
        argv, env = e.launch_args(root, lane)
        assert 'VLLM_USE_V2_MODEL_RUNNER' not in env
        assert '--compilation-config' not in argv and '--no-enable-prefix-caching' not in argv
        c.save(root / lane / 'launch.json', dict(execution_sha='d' * 40, argv=argv,
            environment=cc.environment_view(env), canonical_contract=cc.launch_evidence(lane, os.environ, env, str(c.ROOT))))
        layout = dict(padded=4, max_q=5, requests=4) if varlen and lane == 'adaptive' else {}
        data, ids = run_capture(root / lane, monkeypatch, q=1 if lane == 'vanilla' else 3, **layout)
        raw = dict(id='api', prompt_token_ids=data[-1], usage=dict(completion_tokens=len(ids)),
            choices=[dict(token_ids=ids, finish_reason='stop')])
        sent = []
        def http(port, method, path, body=None):
            if method == 'POST': sent.append(c.decode(body)); return 200, c.encoded(raw)
            return 200, c.encoded(dict(data=[dict(id='qwen2.5-7b-' + lane)]))
        monkeypatch.setattr(e, 'http_exchange', http)
        e.request(root, lane)
        assert len(sent) == 1 and sent[0]['seed'] == 42 and sent[0]['temperature'] == 0
        legacy.synthetic_shutdown(root / lane)
        assert a.validate_lane(root, lane)['lane_valid'] is True
    result = a.compare(root)
    assert result['state'] == 'NOT_REPRODUCED', result
    assert result['captures']['adaptive']['active_query_token_count'] == 3
    expected_geometry = dict(logical_runtime_q=3, captured_graph_q_shape=4 if varlen else 3,
                             max_seqlen_q=[5 if varlen else 3] * 28)
    assert result['v2_query_geometry']['adaptive'] == expected_geometry
    for field, value in expected_geometry.items():
        assert result['captures']['adaptive'][field] == value
    assert result['captures']['adaptive']['production_attention_dispatch']['mode'] == ('2D' if varlen else '3D')
    snapshot = c.checkpoint(root, 'complete')
    assert e.export_snapshot(snapshot).exists()
    assert a.compare(snapshot / 'data')['state'] == 'NOT_REPRODUCED'
    production_base = s.base_sources(**s.profile_options(c.load(root / 'run.json')))
    e.restore(root)
    assert e.restoration_status(root)['state'] == 'RESTORED'
    assert c.file_sha(package / s.ATTENTION_SOURCE) == s.PRODUCTION_ATTENTION_SHA256
    assert (package / vs.RUNNER).read_bytes() == vs.references()[vs.RUNNER]
    assert {name: (package / name).read_bytes() for name in production_base} == production_base
