"""CPU evidence/source contracts only; no fake claim about GPU arithmetic."""
import ast
import copy
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from experiments import m005_ab as ab, m005_ab_sources as sources, m005_ab_observer as observer
from experiments.protocol_b import EvidenceError, ENV_FIELDS, write_new
from src.server import t4_rowwise_attention as production
from src.proposer.controller import AdaptiveDraftController

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared_reservations(tmp_path, monkeypatch):
    # Preserve frozen input bytes; this test exercises lifecycle, not live capture.
    plan = ab.load_json(ab.REFERENCE / 'frozen_plan.json')
    monkeypatch.setattr(ab, 'load_plan', lambda: plan)
    directory = tmp_path / 'run'
    ab.prepare(directory, 'frozen-session', 'GPU-frozen', 'frozen-boot')
    return directory


@pytest.mark.parametrize('condition', ['A', 'B'])
def test_prepare_creates_exact_arm_reservation(prepared_reservations, condition):
    arm = prepared_reservations / condition
    assert (arm / 'observer').is_dir()
    assert (arm / 'observer_config.json').is_file()
    assert ab.load_json(arm / 'reservation.json') == dict(
        pair_id='m005-' + condition, run_id='m005-' + condition,
        lane='adaptive', session_id='frozen-session')


@pytest.mark.parametrize('condition', ['A', 'B'])
def test_capture_arm_consumes_prepared_reservation(prepared_reservations, monkeypatch, condition):
    from experiments import protocol_b_capture as capture
    calls = []
    monkeypatch.setattr(capture, 'capture', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(ab, 'authenticate', lambda *_: None)
    ab.capture_arm(prepared_reservations, condition)
    assert calls == [((prepared_reservations / condition, 'm005-' + condition,
                      'm005-' + condition, 'adaptive', 'frozen-session'),
                     dict(reserved=True, diagnostic_through=ab.SAMPLE))]


@pytest.mark.parametrize('condition', ['A', 'B'])
def test_precreated_arm_reaches_capture_claim_without_mkdir(prepared_reservations, monkeypatch, condition):
    from experiments import protocol_b_capture as capture
    reached = []
    def stop_before_runtime():
        reached.append(True)
        raise EvidenceError('CPU lifecycle probe: reservation accepted')
    monkeypatch.setattr(capture, 'load_plan', stop_before_runtime)
    with pytest.raises(EvidenceError, match='CPU lifecycle probe: reservation accepted'):
        ab.capture_arm(prepared_reservations, condition)
    assert reached == [True]
    assert (prepared_reservations / condition / 'capture.claim').is_file()


@pytest.mark.parametrize('field', ['pair_id', 'run_id', 'lane', 'session_id'])
def test_wrong_m005_reservation_fails_before_claim(prepared_reservations, monkeypatch, field):
    from experiments import protocol_b_capture as capture
    path = prepared_reservations / 'A' / 'reservation.json'
    reservation = ab.load_json(path)
    reservation[field] = 'wrong'
    path.write_text(json.dumps(reservation), encoding='utf-8')
    monkeypatch.setattr(capture, 'load_plan', lambda: pytest.fail('Reached capture with wrong reservation'))
    with pytest.raises(EvidenceError, match='Reservation mismatch'):
        ab.capture_arm(prepared_reservations, 'A')
    assert not (path.parent / 'capture.claim').exists()


@pytest.mark.parametrize('completed', [False, True])
def test_claimed_or_completed_m005_arm_cannot_be_reused(prepared_reservations, monkeypatch, completed):
    from experiments import protocol_b_capture as capture
    arm = prepared_reservations / 'A'
    write_new(arm / 'capture.claim', {'timestamp': 'original'})
    if completed:
        write_new(arm / 'run.json', {'complete': True})
    before = {p.name: p.read_bytes() for p in arm.iterdir() if p.is_file()}
    monkeypatch.setattr(capture, 'load_plan', lambda: pytest.fail('Retried already claimed arm'))
    with pytest.raises(FileExistsError): ab.capture_arm(prepared_reservations, 'A')
    assert before == {p.name: p.read_bytes() for p in arm.iterdir() if p.is_file()}


def test_generic_capture_still_rejects_arbitrary_existing_directory(tmp_path):
    from experiments import protocol_b_capture as capture
    arm = tmp_path / 'arbitrary'; arm.mkdir()
    with pytest.raises(FileExistsError):
        capture.capture(arm, 'pair1', 'run1', 'adaptive', 'session')
    assert list(arm.iterdir()) == []


@pytest.fixture(scope='module')
def originals():
    data = {}
    for path in (ROOT / 'tests/fixtures').glob('*v030.json.gz'):
        data.update({n: s.encode() for n, s in json.loads(gzip.decompress(path.read_bytes())).items()})
    data[production.HELPER] = (ROOT / 'tests/fixtures/m203_v030_triton_attention_helpers.py').read_bytes()
    data[sources.SCHEDULER] = subprocess.check_output(['git', '-c', 'safe.directory=' + ROOT.as_posix(),
                                                     'show', '5ae38fd:src/server/adaptive_scheduler.py'])
    with tarfile.open(ROOT / 'experiments/runs/m2_03/diagnostics/p177-attention-8c04f23-20261019T075533Z.tar.gz') as archive:
        name = next(n for n in archive.getnames() if n.endswith('/v1/core/sched/async_scheduler.py'))
        data['v1/core/sched/async_scheduler.py'] = archive.extractfile(name).read()
    return {n: data[n] for n in sources.HASHES}


class RemoveObserver(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'experiments.m005_ab_observer' else node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_m005_observe':
            return None
        return self.generic_visit(node)


def observer_calls(raw):
    return [ast.dump(n) for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == '_m005_observe']


@pytest.mark.parametrize('condition', ['A', 'B'])
def test_composition_restores_exact_base_ast_and_pins_tokenizer(originals, condition):
    for name, raw in originals.items():
        composed = sources.transform(name, raw, condition)
        restored = ast.dump(RemoveObserver().visit(ast.parse(composed)))
        assert restored == ast.dump(ast.parse(sources.base(name, raw, condition)))
    assert sources.transform(production.DRAFT, originals[production.DRAFT], condition) == production.transform(
        production.DRAFT, originals[production.DRAFT])
    assert sources.base(production.OPS, originals[production.OPS], 'B') == originals[production.OPS]
    assert sources.base(production.OPS, originals[production.OPS], 'A') == production.transform(production.OPS, originals[production.OPS])


def test_identical_observer_and_minimal_hook_surfaces(originals):
    for name, raw in originals.items():
        assert observer_calls(sources.transform(name, raw, 'A')) == observer_calls(sources.transform(name, raw, 'B'))
    assert 'v1/spec_decode/llm_base_proposer.py' not in sources.MODIFIED
    assert 'v1/core/sched/async_scheduler.py' not in sources.MODIFIED


def test_commit_hook_is_after_native_update(originals):
    tree = ast.parse(sources.transform('v1/core/sched/scheduler.py', originals['v1/core/sched/scheduler.py'], 'A'))
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == 'new_token_ids']
    selected = next(n for n in candidates if any(isinstance(v, ast.Expr) and isinstance(v.value, ast.Call)
        and isinstance(v.value.func, ast.Name) and v.value.func.id == '_m005_observe' for v in n.body))
    assert '_update_request_with_output' in ast.unparse(selected.body[-2])
    assert 'account' in ast.unparse(selected.body[-1])


@pytest.mark.parametrize('name', list(sources.HASHES))
def test_source_drift_rejected(originals, name):
    with pytest.raises(EvidenceError): sources.transform(name, originals[name] + b'\n', 'A')


def create_install_tree(tmp_path, originals):
    upstream, project = tmp_path / 'vllm', tmp_path / 'project'
    for name, raw in originals.items():
        path = sources.source_path(name, upstream, project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for name in ('experiments/m005_ab.py', 'experiments/m005_ab_sources.py', sources.OBSERVER,
                 'experiments/protocol_b_capture.py', 'experiments/protocol_b.py',
                 'experiments/lossless_verifier.py', 'experiments/workload_adapter.py',
                 'src/server/t4_rowwise_attention.py', 'src/server/budget_planner.py',
                 'src/proposer/controller.py', 'src/proposer/adapter.py',
                 'scripts/launch_lane1_vanilla.sh', 'scripts/launch_lane3_adaptive.sh', 'scripts/switch_lane.sh'):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    return upstream, project


@pytest.mark.parametrize('condition', ['A', 'B'])
def test_install_verify_restore(tmp_path, originals, condition):
    upstream, project = create_install_tree(tmp_path, originals)
    directory = tmp_path / 'sources'
    sources.install(upstream, project, directory, condition)
    sources.authenticate(directory)
    sources.authenticate(directory, restore=True)
    for name, raw in originals.items(): assert sources.source_path(name, upstream, project).read_bytes() == raw
    with pytest.raises(EvidenceError): sources.authenticate(directory)


@pytest.mark.parametrize('mutation', ['partial', 'unknown', 'backup', 'observer', 'manifest'])
def test_bad_install_restore_fails_before_any_write(tmp_path, originals, mutation):
    upstream, project = create_install_tree(tmp_path, originals)
    directory = tmp_path / 'sources'
    sources.install(upstream, project, directory, 'A')
    path = upstream / production.OPS
    if mutation == 'partial': path.write_bytes(originals[production.OPS])
    if mutation == 'unknown': path.write_bytes(path.read_bytes() + b'\n')
    if mutation == 'backup': (directory / 'originals' / production.OPS).write_bytes(b'bad')
    if mutation == 'observer': (project / sources.OBSERVER).write_bytes(b'bad')
    if mutation == 'manifest':
        p = directory / 'sources.json'; x = json.loads(p.read_text()); x['composed'][production.OPS] = '0'*64; p.write_text(json.dumps(x))
    before = {n: sources.source_path(n, upstream, project).read_bytes() for n in sources.HASHES}
    with pytest.raises(EvidenceError): sources.authenticate(directory, restore=True)
    assert before == {n: sources.source_path(n, upstream, project).read_bytes() for n in sources.HASHES}


@pytest.mark.parametrize('M', range(1, 6))
def test_actual_row_mapping(M):
    positions = list(range(observer.POSITION-M+1, observer.POSITION+1))
    rows = observer.map_rows(1522, positions, list(range(M)), M-1, list(range(M-1)), [M-1])
    assert rows[-1]['position'] == 1683 and rows[-1]['generated_index'] == 162


def test_bitwise_and_logical_cache_semantics():
    # Same logical data, different physical block assignment.
    key = torch.arange(3*16*4*128, dtype=torch.float32).reshape(3,16,4,128).to(torch.float16)
    a = observer.logical_kv(key, key, [0,1], 20)
    shuffled = key[[1,0,2]]
    b = observer.logical_kv(shuffled, shuffled, [1,0], 20)
    assert a == b
    shuffled[0,0,0,0] = -1
    assert observer.logical_kv(shuffled, shuffled, [1,0], 20) != a
    raw = observer.tensor_bytes(torch.tensor([0., -0.], dtype=torch.float16))
    assert bytes.fromhex(raw['hex'])[:2] != bytes.fromhex(raw['hex'])[2:]
    ab.validate_bits(raw, bytes_required=True)
    raw['hex'] = '00000000'
    with pytest.raises(EvidenceError): ab.validate_bits(raw, bytes_required=True)


def sample_attempt(tmp_path, *, stale=False, rejected=False):
    """Execute observer/accounting logic on CPU tensors, not model arithmetic."""
    config = dict(directory=str(tmp_path), profile=sources.PROFILES['A'])
    state = dict(event_seq=0, steps={}, producers={}, failed=False, closed=False)
    observer._state[str(tmp_path.resolve())] = state
    controller = observer.controller_state(AdaptiveDraftController())
    prefix = [10]*160
    request = NS(prompt_token_ids=[1]*1522, _output_token_ids=prefix[:], request_id='chatcmpl-one-suffix')
    config['prompt_token_ids'] = request.prompt_token_ids
    layer = NS(kv_cache=torch.zeros(110,4,16,256,dtype=torch.float16))
    impl = NS(); layer.impl = impl
    scheduled = NS(total_num_scheduled_tokens=4)
    entry = dict(schedule_seq=9, request_id=request.request_id, producer_schedule_seq=8, planned_k=4,
                 batch_k=4, future_k=4, controller=controller, invalid=0, committed_count=160,
                 past_budget_sha256='a'*64)
    state['steps'][id(scheduled)] = entry
    draft = torch.zeros((1,3),dtype=torch.int64)
    state['producers'][request.request_id] = dict(tensor=draft,row=0,schedule_seq=8,
        planned_k=3,batch_k=3,future_k=3,produced_count=3,controller=controller)
    table = torch.arange(110).reshape(1,-1)
    metadata = NS(query_start_loc=torch.tensor([0,4]), seq_lens=torch.tensor([1685]), block_table=table)
    runner = NS(input_batch=NS(req_ids=[request.request_id],num_reqs=1,
                    prev_req_id_to_index={request.request_id:0}), requests={request.request_id:request},
                uses_mrope=False, enable_prompt_embeds=False, query_start_loc=NS(np=[0,4]),
                _draft_token_ids=draft, vllm_config=NS(parallel_config=NS(),
                    compilation_config=NS(static_forward_context={observer.LAYER:layer})))
    positions = torch.arange(1681,1685)
    observer.prepare(config,state,runner,scheduled,torch.tensor([10]*4),positions,
                     {observer.LAYER:metadata},{observer.LAYER:positions},'PIECEWISE')
    q = torch.zeros(4,28,128,dtype=torch.float16)
    key,value = layer.kv_cache.transpose(1,2).split(128,dim=-1)
    output = torch.zeros_like(q)
    observer.attention_pre(config,state,impl,layer,q,key,value,metadata,output)
    observer.dispatch(config,state,dict(q=q,use_3d=True,rowwise_3d=True,TILE_SIZE_DECODE=16,
        TILE_SIZE_PREFILL=32,tile_size=16,grid=(4,4,16),num_par_softmax_segments=16,kv_quant_mode='NONE'))
    observer.attention_post(config,state,impl,layer,output)
    spec = NS(num_draft_tokens=[3],logits_indices=torch.arange(4),target_logits_indices=torch.arange(3),
              bonus_logits_indices=torch.tensor([3]),draft_token_ids=torch.tensor([10,10,5212]))
    logits = torch.zeros(4,6000); logits[:,5212] = 1
    observer.scores(config,state,runner,scheduled,logits,spec,None)
    observer.processed(config,state,'verify_scores',logits[:3],NS(all_greedy=True),spec)
    selected = [] if rejected == 'empty' else [10] if rejected else [10,10,5212,10]
    observer.sampled(config,state,runner,NS(sampled_token_ids=torch.tensor([selected])))
    observer.account_before(config,state,request,selected,scheduled,stale,selected)
    if not stale: request._output_token_ids += selected
    if selected: observer.account(config,state,request,selected,scheduled,stale,selected)
    return json.loads(next(tmp_path.glob('observer/*.json')).read_text())


@pytest.mark.parametrize('stale,rejected,committing', [(False,False,True),(True,False,False),(False,True,False),(False,'empty',False)])
def test_committing_rejected_stale_and_producer_consumer_lineage(tmp_path, stale, rejected, committing):
    (tmp_path/'observer').mkdir()
    event = sample_attempt(tmp_path,stale=stale,rejected=rejected)
    assert event['committing'] is committing
    assert event['planned_k'] == 4 and event['producer']['planned_k'] == event['draft_count'] == 3
    assert event['target_row']['generated_index'] == 162


def validated_pair():
    environment = {k:'frozen' for k in ENV_FIELDS}
    environment['gpu_name'] = 'Tesla T4'
    environment['config'] = {'environment': {'ADAPTIVE_TELEMETRY_PATH':'arm'}}
    records = [dict(token_ids=[1],finish_reason='stop',prompt_token_ids=[2]) for _ in range(4)]
    records.append(dict(token_ids=[10]*162+[5212]))
    producer = dict(planned_k=3,batch_k=3,future_k=3,produced_count=3,controller={'x':1})
    entry = dict(controller={'x':1},past_budget_sha256='same',planned_k=3,batch_k=3,future_k=3,
                 M=4,positions=[1681,1682,1683,1684],input_ids=[10]*4,draft_count=3,
                 draft_ids=[10]*3,effective_drafted=3,graph_mode='PIECEWISE',producer=producer,
                 dispatch={'rowwise':True},
                 pre={'Q':'same','KV':'same'},post={'sha256':'same'},raw={'sha256':'same'})
    a = dict(run={'environment':environment,'records':records},entry=entry,
             cache={'local':0,'external':0},source={'project_hashes':{'observer':'same'}})
    b = copy.deepcopy(a); b['run']['records'][4]['token_ids'][162] = 341
    b['entry']['dispatch']['rowwise'] = False
    return a,b


def test_classification_is_not_kernel_causality():
    a,b = validated_pair()
    b['entry']['post']['sha256'] = 'different'
    report = ab.compare_validated(a,b)
    assert report['classification'] == 'ROWWISE_POLICY_CAUSAL'
    assert report['localization'] == 'LAYER0_ATTENTION_DIFFERENCE_LOCALIZED'
    assert report['kernel_root_cause'] == 'UNKNOWN' and report['task_done'] is False
    b['run']['records'][4]['token_ids'][162] = 5212
    assert ab.compare_validated(a,b)['classification'] == 'ROWWISE_POLICY_NOT_SUFFICIENT'


@pytest.mark.parametrize('gate', ['controller','past_budget_sha256','planned_k','batch_k','future_k','M',
    'positions','input_ids','draft_count','draft_ids','effective_drafted','graph_mode','pre'])
def test_each_comparability_gate_fails_inconclusively(gate):
    a,b = validated_pair(); b['entry'][gate] = 'different'
    assert ab.compare_validated(a,b)['classification'] == 'INCONCLUSIVE'


def test_conditioning_prefix_environment_and_reproduction_gates():
    for mutation in ('conditioning','prefix','gpu','A_not_reproduced'):
        a,b = validated_pair()
        if mutation == 'conditioning': b['run']['records'][0]['finish_reason'] = 'length'
        if mutation == 'prefix': b['run']['records'][4]['token_ids'][12] = 20
        if mutation == 'gpu': b['run']['environment']['gpu_uuid'] = 'other'
        if mutation == 'A_not_reproduced': a['run']['records'][4]['token_ids'][162] = 341
        assert ab.compare_validated(a,b)['classification'] == 'INCONCLUSIVE'


def test_missing_evidence_invalid_not_inconclusive(tmp_path):
    assert ab.analyze(tmp_path)['classification'] == 'INVALID_EVIDENCE'


def test_no_outside_scope_tensor_capture(monkeypatch):
    observer._local.attention = None
    monkeypatch.setattr(observer,'tensor_bytes',lambda *a,**kw: pytest.fail('Unbounded tensor read'))
    observer.attention_pre({}, {}, object(), object(), None,None,None,None,None)
    observer.attention_post({}, {}, object(), object(), None)
    observer.dispatch({}, {}, {})


def test_closed_observer_is_noop_for_remaining_completion(tmp_path,monkeypatch):
    config={'directory':str(tmp_path)}
    path=tmp_path/'config.json'; path.write_text(json.dumps(config))
    observer._state[str(tmp_path.resolve())]={'failed':False,'closed':True}
    monkeypatch.setenv('M203_M005_CONFIG',str(path))
    monkeypatch.setattr(observer,'schedule',lambda *a: pytest.fail('Observed after committed boundary'))
    observer.observe('schedule',None,None)


def test_controller_snapshot_does_not_mutate():
    controller = AdaptiveDraftController()
    before = copy.deepcopy(controller.__dict__)
    observer.controller_state(controller)
    assert controller.__dict__ == before


def test_protected_profiles_byte_unchanged():
    for name in ('experiments/protocol_b_boundary.py','experiments/protocol_b_boundary_sources.py',
                 'experiments/protocol_b_boundary_observer.py','src/server/t4_rowwise_attention.py',
                 'patches/vllm-0.30.0-t4-rowwise-attention.patch'):
        historical = subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'show','5ae38fd:'+name])
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n') == historical


def sealed_arm(tmp_path, originals):
    """Synthetic HTTP/events with real pinned source compositions, never GPU evidence."""
    arm = tmp_path/'A'; arm.mkdir(); (arm/'observer').mkdir()
    upstream,project = create_install_tree(tmp_path/'runtime',originals)
    source = sources.install(upstream,project,arm/'sources','A')
    reference = ROOT/'experiments/runs/m2_03/protocol_b_20261018T085118Z_2aad4c00'
    plan = json.loads((reference/'frozen_plan.json').read_text(encoding='utf-8'))
    assert len(ab.conditioned_prompts(plan)) == 5
    run = json.loads((reference/'pair1-adaptive/run.json').read_text(encoding='utf-8'))
    run.update(pair_id='m005-A',run_id='m005-A',diagnostic_only=True,diagnostic_through=observer.SAMPLE)
    run['records'] = run['records'][:5]
    for record in run['records']: record.update(pair_id='m005-A',run_id='m005-A')
    record = run['records'][4]
    record.update(response_id='chatcmpl-one',token_ids=[10]*162+[5212,10],completion_tokens=164)
    config = dict(directory=str(arm),profile=sources.PROFILES['A'],condition='A',
        source_directory=str(arm/'sources'),sample_id=observer.SAMPLE,index=162,
        prompt_token_ids=record['prompt_token_ids'],request_sha256=ab.REQUEST,
        session_id=run['environment']['session_id'],boot_id=run['environment']['boot_id'],
        gpu_uuid=run['environment']['gpu_uuid'])
    sample_dir = tmp_path/'sample'; sample_dir.mkdir(); (sample_dir/'observer').mkdir()
    event = sample_attempt(sample_dir)
    # Rebind the synthetic event to the sealed run's real prompt/config identity.
    event.update(event_seq=3,config_sha256=ab.fingerprint(config),profile=config['profile'])
    write_new(arm/'observer/000001.json',dict(kind='bound',event_seq=1,config_sha256=ab.fingerprint(config),
        profile=config['profile'],pid=os.getpid(), process_chain=[dict(pid=run['live_server']['pid'],
        start_ticks=run['live_server']['process_start_ticks'])]))
    write_new(arm/'observer/000002.json',dict(kind='cache_lookup',event_seq=2,config_sha256=ab.fingerprint(config),
        profile=config['profile'],pid=os.getpid(),local=0,external=0))
    write_new(arm/'observer/000003.json',event)
    hashes = dict(source['project_hashes'])
    hashes['src/server/adaptive_scheduler.py'] = source['composed'][sources.SCHEDULER]
    for field in ('source_hashes','launcher_hashes'):
        run['environment'][field] = {n:hashes[n] for n in run['environment'][field]}
    journal = []
    for prompt,r in zip(plan['prompts'],run['records']):
        payload = dict(prompt['request'],model='qwen2.5-7b-adaptive',return_token_ids=True)
        body = dict(id=r['response_id'],model='qwen2.5-7b-adaptive',prompt_token_ids=r['prompt_token_ids'],
                    usage={'completion_tokens':len(r['token_ids'])},
                    choices=[dict(token_ids=r['token_ids'],finish_reason=r['finish_reason'])],
                    metrics={'speculative_decoding':{'num_draft_tokens':3,'num_spec_steps':1}})
        journal.extend([dict(event='attempt',request=payload),dict(event='result',record=r,
            response=body,raw_response_body=json.dumps(body),http_status=200)])
    (arm/'responses.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in journal))
    run['responses_sha256'] = ab.file_sha(arm/'responses.jsonl')
    write_new(arm/'run.json',run); write_new(arm/'observer_config.json',config)
    write_new(tmp_path/'experiment.json',dict(plan=plan))
    ab.seal_arm(tmp_path,'A')
    return arm


def test_sealed_raw_evidence_validates_and_tampering_rejected(tmp_path,originals):
    arm = sealed_arm(tmp_path,originals)
    assert ab.validate_arm(arm,'A')['entry']['committed'][162] == 5212
    (arm/'observer/000003.json').write_text('{}')
    with pytest.raises(EvidenceError): ab.validate_arm(arm,'A')


@pytest.mark.parametrize('mutation', ['missing','stale','wrong_row','wrong_producer','wrong_raw','observer_error','pre_bits'])
def test_resealed_semantic_failures_are_invalid(tmp_path,originals,mutation):
    arm = sealed_arm(tmp_path,originals)
    event_path=arm/'observer/000003.json'; event=json.loads(event_path.read_text())
    if mutation=='missing': event_path.unlink()
    else:
        if mutation=='stale': event['stale']=True
        if mutation=='wrong_row': event['target_row']['position']=1682
        if mutation=='wrong_producer': event['producer']['schedule_seq']=1
        if mutation=='wrong_raw':
            path=arm/'responses.jsonl'; rows=[json.loads(l) for l in path.read_text().splitlines()]
            rows[-1]['response']['choices'][0]['token_ids'][162]=341
            path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        if mutation=='observer_error': event['kind']='observer_error'
        if mutation=='pre_bits': event['pre']['Q']['hex']='00'
        event_path.write_text(json.dumps(event))
    # An internally consistent checksum inventory cannot authorize bad semantics.
    path=arm/'seal.json'; seal=json.loads(path.read_text())
    seal['files']={p.relative_to(arm).as_posix():ab.file_sha(p) for p in arm.rglob('*') if p.is_file() and p.name!='seal.json'}
    path.write_text(json.dumps(seal))
    with pytest.raises((EvidenceError,KeyError)): ab.validate_arm(arm,'A')
