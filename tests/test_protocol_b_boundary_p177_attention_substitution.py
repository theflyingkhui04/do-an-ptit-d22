"""CPU argument/byte/source/evidence contracts. No Triton/CUDA numerics are emulated."""
import ast
import copy
import functools
import hashlib
import io
import json
import os
import subprocess
import tarfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

import test_protocol_b_boundary as base_tests
import test_protocol_b_boundary_p177_k0_control as control_tests
from experiments import protocol_b_boundary as boundary
from experiments import protocol_b_boundary_observer as observer
from experiments import protocol_b_boundary_sources as sources
from experiments.protocol_b import EvidenceError, fingerprint, file_sha


def config():
    cfg = control_tests.config()
    cfg["p177_attention_substitution"] = dict(cfg.pop("p177_k0_control"),
        profile=observer.ATTENTION_PROFILE, control_sha256=boundary.ATTENTION_CONTROL_SHA256)
    return cfg


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("M203_P177_ATTENTION_SUBSTITUTION", "1")
    monkeypatch.delenv("M203_P177_PRODUCER_K0", raising=False)
    old = observer._local
    import threading
    observer._local = threading.local()
    observer._early.clear()
    yield
    observer._local = old
    observer._early.clear()


def tensors():
    impl = NS(num_heads=28, num_kv_heads=4, head_size=128, scale=128**-0.5,
        _kv_quant_mode=NS(name="NONE"), _is_per_token_head_quant=False, alibi_slopes=None,
        sinks=None, use_alibi_sqrt=False, use_td=False, chunk_lookback=-1,
        sliding_window=(-1,-1), logits_soft_cap=0)
    cache = torch.arange(12*4*16*256, dtype=torch.int32).to(torch.float16).reshape(12,4,16,256)
    layer = NS(impl=impl, kv_cache=cache, _q_scale=torch.tensor(1.), _k_scale=torch.tensor(1.), _v_scale=torch.tensor(1.))
    m = NS(num_actual_tokens=4, max_query_len=4, max_seq_len=181,
        query_start_loc=torch.tensor([0,4], dtype=torch.int32), seq_lens=torch.tensor([181], dtype=torch.int32),
        block_table=torch.arange(12, dtype=torch.int32).reshape(1,12), causal=True, use_cascade=False,
        mm_prefix_range_tensor=None, rswa_prefix_lens=None, rswa_window=None,
        seq_threshold_3D=32, num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(32,28,16,128), softmax_segm_max=torch.empty(32,28,16),
        softmax_segm_expsum=torch.empty(32,28,16))
    q = torch.arange(4*28*128, dtype=torch.float32).to(torch.float16).reshape(4,28,128)
    output = torch.ones_like(q)
    context = {observer.LAYER0: layer, observer.LAYER1: NS(kv_cache=cache.clone())}
    runner = NS(vllm_config=NS(compilation_config=NS(static_forward_context=context)))
    cfg = config()
    scheduled = NS(scheduled_spec_decode_tokens={"adaptive": [-1]*3})
    pending = dict(runner=runner, metadata={observer.LAYER0:m, observer.LAYER1:m}, config_key=fingerprint(cfg),
        inputs=dict(positions=torch.arange(177,181),input_ids=torch.tensor([7,8,9,10])),
        positions=[177,178,179,180], binding=dict(layers={observer.LAYER0:dict(cache=observer.tensor_identity(cache))}),
        slots={observer.LAYER0:torch.arange(177,181), observer.LAYER1:torch.arange(177,181)},
        attention_state=dict(shadow_count=0, substitution_count=0), request_id="adaptive", schedule_seq=23,
        scheduled=scheduled, forward_id="forward")
    pending['peers']={name:observer.tensor_identity(item.kv_cache) for name,item in context.items()}
    key,value=cache.transpose(1,2).split(128,dim=-1)
    return cfg,pending,impl,layer,q,key,value,m,output


def test_off_has_no_reads_and_wrong_sample_does_not_substitute(tmp_path, monkeypatch):
    args = [None]*10
    monkeypatch.delenv("M203_P177_ATTENTION_SUBSTITUTION")
    with patch.object(observer.Path,"read_text",side_effect=AssertionError("read")):
        observer.attention_substitute(*args)
    monkeypatch.setenv("M203_P177_ATTENTION_SUBSTITUTION","1")
    cfg=config();cfg["sample_id"]="other"
    path=tmp_path/"config.json";path.write_text(json.dumps(cfg))
    monkeypatch.setenv("PROTOCOL_B_BOUNDARY_CONFIG",str(path))
    with patch.object(observer,"attention_apply",side_effect=AssertionError("substitution")):
        observer.attention_substitute(*args)


def test_shadow_only_p177_live_views_private_metadata_and_scratch():
    cfg,p,impl,layer,q,k,v,m,out=tensors()
    args=observer.attention_shadow_arguments(impl,layer,q,k,v,m,out)
    assert args["q"].data_ptr()==q.data_ptr() and args["q"].shape==(1,28,128)
    assert args["k"] is k and args["v"] is v
    assert args["block_table"].data_ptr()==m.block_table.data_ptr()
    assert args["cu_seqlens_q"].tolist()==[0,1] and args["seqused_k"].tolist()==[178]
    assert args["max_seqlen_q"]==1 and args["max_seqlen_k"]==178 and args["causal"] is True
    assert args["softmax_scale"]==impl.scale and args["num_par_softmax_segments"]==16
    assert args["out"].data_ptr()!=out.data_ptr()
    for name in ("softmax_segm_output","softmax_segm_max","softmax_segm_expsum"):
        assert args[name].data_ptr()!=getattr(m,name).data_ptr() and args[name].shape[0]==1
    assert m.seq_lens.tolist()==[181] and m.query_start_loc.tolist()==[0,4]


def test_exact_bytes_one_compute_one_copy_no_cache_or_tail_mutation():
    cfg,p,impl,layer,q,k,v,m,out=tensors()
    cache=layer.kv_cache.clone(); tail=out[1:].clone();q_before=q.clone();events=[];calls=[]
    def compute(**args):
        calls.append(args)
        args["out"].fill_(-0.0)  # A sentinel, never a model of GPU attention.
    with patch.object(observer,"emit",side_effect=lambda c,kind,**kw:events.append(dict(kind=kind,**kw))), \
            patch.object(observer,"kv_authenticate",return_value="auth"):
        observer.attention_apply(cfg,p,impl,layer,q,k,v,m,out,compute)
        with pytest.raises(EvidenceError,match="Duplicate"):
            observer.attention_apply(cfg,p,impl,layer,q,k,v,m,out,compute)
    assert len(calls)==len(events)==1 and p["attention_state"]==dict(shadow_count=1,substitution_count=1)
    assert torch.equal(layer.kv_cache,cache) and torch.equal(out[1:],tail) and torch.equal(q,q_before)
    event=events[0]
    raw=out[:1].view(torch.uint8).numpy().tobytes()
    assert event["shadow_3d_attention_p177"]["hex"]==raw.hex()
    assert event["shadow_3d_attention_p177"]["sha256"]==hashlib.sha256(raw).hexdigest()
    assert raw[:2]==b"\x00\x80" and event["native_tail"]==event["tail_after"]


@pytest.mark.parametrize("field,value",[("positions",[177]),("config_key","wrong")])
def test_wrong_forward_binding_rejects(field,value):
    cfg,p,impl,layer,q,k,v,m,out=tensors();p[field]=value
    with pytest.raises(EvidenceError):observer.attention_apply(cfg,p,impl,layer,q,k,v,m,out,lambda **kw:None)


def test_wrong_layer_or_metadata_rejects():
    cfg,p,impl,layer,q,k,v,m,out=tensors()
    for wrong_layer,wrong_m in [(NS(impl=impl,kv_cache=layer.kv_cache),m),(layer,copy.copy(m))]:
        with pytest.raises(EvidenceError):observer.attention_apply(cfg,p,impl,wrong_layer,q,k,v,wrong_m,out,lambda **kw:None)


@pytest.mark.parametrize("mutation",["dtype","shape","device","causal","seq_len","segments"])
def test_unsupported_shadow_arguments_fail_closed(mutation):
    cfg,p,impl,layer,q,k,v,m,out=tensors()
    if mutation=="dtype":q=q.float()
    if mutation=="shape":q=q[:1]
    if mutation=="device":q=torch.empty(q.shape,device="meta",dtype=q.dtype)
    if mutation=="causal":m.causal=False
    if mutation=="seq_len":m.seq_lens[0]=182
    if mutation=="segments":m.num_par_softmax_segments=0
    with pytest.raises(EvidenceError):observer.attention_shadow_arguments(impl,layer,q,k,v,m,out)


def armed():
    cfg,p,*rest=tensors()
    producer=dict(schedule_seq=22,request_id="adaptive",positions=[176],active_query=[[176,7]],
                  planned_k=3,batch_k=3,future_k=3,verified=True)
    entry=dict(output=p["scheduled"],schedule_seq=23,request_id="adaptive",placeholder_origin_schedule_seq=22,
               positions=p["positions"],active_query=[[177,7],[178,8],[179,9],[180,10]])
    state=observer.early_state(cfg);state.update(request_id="adaptive",steps={id(p["scheduled"]):entry, 0:producer})
    p['runner']._draft_token_ids=torch.tensor([[8,9,10]])
    state['producers'][(id(p['runner']),'adaptive')]=dict(tensor=p['runner']._draft_token_ids,row=0,schedule_seq=22)
    return cfg,p,producer,entry


@pytest.mark.parametrize("mutation",["request","positions","budget","drafts","earlier_coverage"])
def test_wrong_arm_lineage_rejects(mutation):
    cfg,p,producer,entry=armed()
    if mutation=="request":entry["request_id"]="other"
    if mutation=="positions":p["positions"]=[177]
    if mutation=="budget":producer["planned_k"]=0
    if mutation=="drafts":p["scheduled"].scheduled_spec_decode_tokens={}
    if mutation=="earlier_coverage":producer["positions"]=[177]
    with pytest.raises(EvidenceError):observer.attention_arm(cfg,p)


def test_valid_arm_keeps_budget_and_drafts():
    cfg,p,producer,entry=armed();before=copy.deepcopy(p["scheduled"].__dict__)
    observer.attention_arm(cfg,p)
    assert observer._local.attention_pending is p and p["scheduled"].__dict__==before


@functools.lru_cache(maxsize=1)
def archived_metadata():
    """Read immutable normal-run metadata; all diagnostic outputs below are synthetic."""
    raw=subprocess.check_output(["git","-c","safe.directory="+boundary.ROOT.as_posix(),"show",
        "9c0f15e:experiments/runs/m2_03/diagnostics/"+boundary.CONTROL_BASELINE_NAME],cwd=boundary.ROOT)
    files=boundary.sealed_archive_files(raw)
    events={lane:[json.loads(v) for p,v in files.items() if p.startswith(lane+"/observer/") and p.endswith(".json")]
            for lane in ("vanilla","adaptive")}
    posts={lane:next(e for e in events[lane] if e["kind"]=="postwrite_position177") for lane in events}
    history=next(e for e in events["adaptive"] if e["kind"]=="kv_history")
    prefix={c:history["layers"][observer.LAYER1]["localization"][c]["position_sha256"][:177] for c in ("K","V")}
    return posts,prefix


def evidence(restored=True):
    cfg=config();posts,prefix=copy.deepcopy(archived_metadata());post=copy.deepcopy(posts["adaptive"])
    post.update(request_id="adaptive",pid=2,schedule_seq=23,scheduler_output_id=200,forward_id="forward",
        actual_positions=[177,178,179,180],active_query=[[177,7],[178,8],[179,9],[180,10]],input_token_id=7)
    if restored:
        for c in ("K","V"):post[c]=copy.deepcopy(posts["vanilla"][c])
    native=observer.attention_bytes(torch.ones(1,28,128,dtype=torch.float16))
    shadow=observer.attention_bytes(torch.zeros(1,28,128,dtype=torch.float16))
    tail=observer.attention_bytes(torch.ones(3,28,128,dtype=torch.float16))
    for x in (native,shadow,tail):x["device"]=post["cache_identity"]["device"]
    change=dict(kind="p177_attention_substitution",non_canonical=True,attention_profile=observer.ATTENTION_PROFILE,
        request_id="adaptive",pid=2,schedule_seq=23,scheduler_output_id=200,forward_id="forward",auth_id=post["auth_id"],
        actual_target_positions=[177,178,179,180],planned_k=3,scheduled_drafts=3,shadow_count=1,substitution_count=1,
        target_layer=observer.LAYER0,no_extra_kv_write=True,
        substitution_point="after_native_attention_before_return_to_output_projection",shadow_query_start_loc=[0,1],
        shadow_seq_lens=[178],shadow_max_query_len=1,shadow_max_seq_len=178,shadow_segments=16,graph_mode="PIECEWISE",
        native_attention_p177=native,shadow_3d_attention_p177=shadow,native_tail=tail,tail_after=copy.deepcopy(tail),
        q_bytes=copy.deepcopy(native),q_identity=dict(shape=[1,28,128],dtype='torch.float16',device=native['device'],data_ptr=1234),
        output_identity=dict(shape=[4,28,128],dtype="torch.float16",device=native["device"]),
        cache_identity=copy.deepcopy(post["runtime_binding"]["layers"][observer.LAYER0]["cache"]),
        prefix_0_176=dict(logical_positions=list(range(177)),**copy.deepcopy(prefix)))
    auth=dict(attention_substitution=observer.ATTENTION_PROFILE)
    auth_id=fingerprint(dict(auth,pid=2));post['auth_id']=change['auth_id']=auth_id
    for name,offset in (('key_identity',0),('value_identity',128)):
        cache=change['cache_identity']
        change[name]=dict(cache,shape=[cache['shape'][0],16,4,128],strides=[cache['strides'][i] for i in (0,2,1,3)],
            storage_offset=cache['storage_offset']+offset*cache['strides'][3],data_ptr=cache['data_ptr']+offset*cache['strides'][3]*2)
    events=[dict(kind="kv_sources",auth_id=post["auth_id"],pid=2,**auth),
        dict(kind="early_schedule",schedule_seq=22,request_id="adaptive",scheduler_output_id=100,
             planned_k=3,batch_k=3,num_spec_tokens_to_schedule=3,next_placeholders=[-1]*3),
        dict(kind="early_verify",schedule_seq=22,request_id="adaptive",scheduler_output_id=100,
             active_query=[[176,7]],verification_draft_slots=0,decoding_path="ordinary"),
        dict(kind="early_produced",schedule_seq=22,request_id="adaptive",scheduler_output_id=100,
             produced_count=3,producer_tensor_id=300,producer_row=0),
        dict(kind="early_schedule",schedule_seq=23,request_id="adaptive",scheduler_output_id=200,
             placeholder_origin_schedule_seq=22,scheduled_tokens=4,scheduled_spec_decode_tokens=[-1]*3),change,post,
        dict(kind="early_verify",schedule_seq=23,request_id="adaptive",scheduler_output_id=200,
             producer_schedule_seq=22,producer_tensor_id=300,previous_request_row=0,
             decoding_path="speculative",verification_draft_slots=3,effective_num_drafted=3,num_invalid=0,
             actual_draft_token_ids=[8,9,10],active_query=post["active_query"])]
    for i,e in enumerate(events):
        e.update(schema="m203.boundary.event.v1",lane="adaptive",execution_sha=cfg["execution_sha"],pid=2,
            config_fingerprint=fingerprint(cfg),timestamp=f"2026-10-19T00:00:{i:02d}+00:00")
    record=dict(sample_id=boundary.SAMPLE,ok=True,prompt_token_ids=cfg["prompt_token_ids"],token_ids=[7]*284,
        response_id="adaptive",request_sha256=cfg["p177_attention_substitution"]["request_sha256"])
    reference=dict(archive_sha256=boundary.ATTENTION_CONTROL_SHA256,
                   postwrite={c:copy.deepcopy(posts["vanilla"][c]) for c in ("K","V")})
    baseline=dict(identity=dict(archive_sha256="b"*64),attention_reference=reference,prefix=prefix,postwrites=posts,
        runs=dict(adaptive=dict(records=[{}]*45+[dict(prompt_token_ids=cfg["prompt_token_ids"],token_ids=record["token_ids"])])))
    return events,record,cfg,baseline,auth


@pytest.mark.parametrize("restored,classification",[(True,"LAYER0_ATTENTION_PATH_CAUSAL"),
                                                    (False,"ATTENTION_SUBSTITUTION_INSUFFICIENT")])
def test_scoped_classifications_never_complete_task(restored,classification):
    result=boundary.compare_attention(*evidence(restored))
    assert result["status"]=="INTERPRETABLE",result
    assert result["classification"]==classification and result["task_done"] is False
    assert result["same_kv_cache_state_proven"] is result["historical_kv_state_equivalent"] is False
    assert result["temporal_kv_onset"]=="UNKNOWN" and result["interpretation_status"]=="NOT_YET_APPROVED"


@pytest.mark.parametrize("mutation",["equal_outputs","duplicate","missing","two_shadows","zero_substitutions","budget","drafts", "request",
    "positions","layer","prefix","tail","bytes","graph","cache","error","override","canonical"])
def test_invalid_evidence(mutation):
    data=list(evidence());events,record,cfg,baseline,auth=data
    change=next(e for e in events if e["kind"]=="p177_attention_substitution")
    if mutation=="equal_outputs":change["shadow_3d_attention_p177"]=copy.deepcopy(change["native_attention_p177"])
    if mutation=="duplicate":events.append(copy.deepcopy(change))
    if mutation=="missing":events.remove(change)
    if mutation=="two_shadows":change["shadow_count"]=2
    if mutation=="zero_substitutions":change["substitution_count"]=0
    if mutation=="budget":events[1]["planned_k"]=0
    if mutation=="drafts":events[-1]["verification_draft_slots"]=0
    if mutation=="request":change["request_id"]="wrong"
    if mutation=="positions":change["actual_target_positions"]=[177]
    if mutation=="layer":change["target_layer"]=observer.LAYER1
    if mutation=="prefix":change["prefix_0_176"]["K"][0]="0"*64
    if mutation=="tail":change["tail_after"]=change["shadow_3d_attention_p177"]
    if mutation=="bytes":change["native_attention_p177"]["hex"]="0000"
    if mutation=="graph":change["graph_mode"]="FULL"
    if mutation=="cache":change["cache_identity"]["data_ptr"]+=2
    if mutation=="error":events.append(dict(change,kind="observer_error"))
    if mutation=="override":events.append(dict(change,kind="p177_control_override"))
    if mutation=="canonical":cfg["non_canonical"]=False
    result=boundary.compare_attention(*data)
    assert result["status"]==result["classification"]=="INVALID_EVIDENCE",result
    assert result["task_done"] is False


def test_exact_source_transform_order_and_unrelated_ast_rejection():
    raw=base_tests.pinned_sources()[sources.ATTENTION_SOURCE].encode()
    assert sources.transform(sources.ATTENTION_SOURCE,raw)==raw
    patched=sources.transform(sources.ATTENTION_SOURCE,raw,attention=True)
    assert ast.dump(sources.normalized_transform_ast(sources.ATTENTION_SOURCE,patched,attention=True))==ast.dump(ast.parse(raw))
    text=patched.decode()
    assert text.count(sources.ATTENTION_HOOK)==1
    assert text.index(sources.ATTENTION_HOOK)<text.index("        return output\n\n    def _pth_key_value_caches(")
    for bad in [patched.replace(b"softmax_scale=self.scale",b"softmax_scale=0.0"),patched.replace(sources.ATTENTION_HOOK.encode(),b""),
                patched.replace(sources.ATTENTION_HOOK.encode(),sources.ATTENTION_HOOK.encode()*2)]:
        with pytest.raises(EvidenceError):sources.validate_transform(sources.ATTENTION_SOURCE,raw,bad,attention=True)
    # The pinned compute-only operation has no cache-write argument or invocation.
    tree=ast.parse(__import__('inspect').getsource(observer.attention_apply))
    calls=[n.func.attr if isinstance(n.func,ast.Attribute) else getattr(n.func,'id','') for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any('cache_update' in name or 'reshape_and_cache' in name for name in calls)
    assert calls.count('compute')==calls.count('copy_')==1


def test_install_authenticate_restore_exact_pristine_bytes():
    case=base_tests.AtomicSourceWriteTests();case.setUp()
    try:
        manifest=dict(control_profile=observer.ATTENTION_PROFILE,non_canonical=True,
                      source_auth=dict(scheduler_sha256=file_sha(boundary.ROOT/'src/server/adaptive_scheduler.py')))
        (case.run/'manifest.json').write_text(json.dumps(manifest))
        path=case.package/'v1/core/sched/async_scheduler.py';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(control_tests.ASYNC_SCHEDULER_SOURCE.encode())
        sources.install(case.run)
        auth=sources.authenticate(case.run,installed=True)
        assert auth['attention_substitution']==observer.ATTENTION_PROFILE
        assert (case.package/sources.ATTENTION_SOURCE).read_bytes()!=case.originals[sources.ATTENTION_SOURCE]
        sources.restore(case.run)
        assert all((case.package/name).read_bytes()==raw for name,raw in case.originals.items())
    finally:case.doCleanups()


@pytest.fixture
def artifact():
    """Existing complete CPU artifact, upgraded to the separate attention profile."""
    case=control_tests.ControlArtifactTests()
    try:
        root,baseline=case.artifact()
        output=root/'adaptive'
        manifest=json.loads((root/'manifest.json').read_text())
        cfg=json.loads((output/'boundary_config.json').read_text())
        cfg['p177_attention_substitution']=dict(cfg.pop('p177_k0_control'),
            profile=observer.ATTENTION_PROFILE,control_sha256=boundary.ATTENTION_CONTROL_SHA256)
        events,record,_,new_baseline,_=evidence()
        baseline.update(postwrites=new_baseline['postwrites'],prefix=new_baseline['prefix'])
        reference=new_baseline['attention_reference']
        manifest.update(control_profile=observer.ATTENTION_PROFILE,attention_reference='sealed-control.tar.gz',
                        attention_reference_identity=reference,
                        source_auth=boundary.control_scheduler_auth(baseline,attention=True))
        (root/'manifest.json').write_text(json.dumps(manifest))
        state=json.loads((root/'instrumentation.json').read_text())
        raw=(root/'vllm-originals'/sources.ATTENTION_SOURCE).read_bytes()
        patched=sources.transform(sources.ATTENTION_SOURCE,raw,attention=True)
        (__import__('pathlib').Path(state['vllm_root'])/sources.ATTENTION_SOURCE).write_bytes(patched)
        state['patched_hashes'][sources.ATTENTION_SOURCE]=hashlib.sha256(patched).hexdigest()
        state['attention_substitution']=observer.ATTENTION_PROFILE
        state['scheduler_control']=manifest['source_auth']
        (root/'instrumentation.json').write_text(json.dumps(state))
        auth=sources.authenticate(root)
        live=json.loads((output/'run.json').read_text())['live_server']
        (output/'boundary_config.json').write_text(json.dumps(cfg))
        binding=json.loads((output/'observer_binding.json').read_text())
        binding.update(config_sha256=file_sha(output/'boundary_config.json'),config_fingerprint=fingerprint(cfg),
                       diagnostic_env_key='M203_P177_ATTENTION_SUBSTITUTION')
        (output/'observer_binding.json').write_text(json.dumps(binding))
        for path in (output/'observer').glob('*.json'):path.unlink()
        for i,event in enumerate(events):
            event.update(pid=live['pid'],execution_sha=cfg['execution_sha'],config_fingerprint=fingerprint(cfg))
            if event['kind']=='kv_sources':event.update(auth)
            if 'auth_id' in event:event['auth_id']=fingerprint(dict(auth,pid=live['pid']))
            (output/'observer'/f'{i}.json').write_text(json.dumps(event))
        yield root,baseline,reference
    finally:case.doCleanups()


def test_complete_attention_artifact_uses_real_provenance_and_source_gates(artifact):
    root,baseline,reference=artifact
    with patch.object(boundary,'control_baseline',return_value=baseline), \
            patch.object(boundary,'attention_control_reference',return_value=reference):
        result=boundary.analyze_control(root,attention=True)
    assert result['status']=='INTERPRETABLE',result
    assert result['classification']=='LAYER0_ATTENTION_PATH_CAUSAL' and result['task_done'] is False
    assert (root/'p177_layer0_attention_substitution.json').is_file()
    assert not (root/'p177_k0_control.json').exists()


@pytest.mark.parametrize('mutation',['packages','source','reference','environment','binding','journal','query'])
def test_complete_artifact_rejects_provenance_drift(artifact,mutation):
    root,baseline,reference=artifact
    if mutation=='packages':(root/'server-packages.txt').write_text('different==1\n')
    if mutation=='reference':reference=dict(reference,archive_sha256='0'*64)
    if mutation=='source':
        state=json.loads((root/'instrumentation.json').read_text());state['attention_substitution']='wrong'
        (root/'instrumentation.json').write_text(json.dumps(state))
    if mutation=='environment':
        path=root/'adaptive/run.json';run=json.loads(path.read_text());run['environment']['boot_id']='wrong'
        path.write_text(json.dumps(run))
    if mutation=='binding':
        path=root/'adaptive/observer_binding.json';binding=json.loads(path.read_text());binding['diagnostic_env_key']='M203_P177_PRODUCER_K0'
        path.write_text(json.dumps(binding))
    if mutation=='journal':
        with (root/'adaptive/responses.jsonl').open('a') as f:f.write('{}\n')
    if mutation=='query':
        path=root/'adaptive/observer/5.json';event=json.loads(path.read_text());event['actual_target_positions']=[177]
        path.write_text(json.dumps(event))
    with patch.object(boundary,'control_baseline',return_value=baseline), \
            patch.object(boundary,'attention_control_reference',return_value=reference):
        result=boundary.analyze_control(root,attention=True)
    assert result['status']==result['classification']=='INVALID_EVIDENCE',result


def test_pinned_control_archive_and_historical_bytes_unchanged(tmp_path):
    raw=subprocess.check_output(['git','-c','safe.directory='+boundary.ROOT.as_posix(),'show',
        '124db36:experiments/runs/m2_03/diagnostics/'+boundary.ATTENTION_CONTROL_NAME],cwd=boundary.ROOT)
    assert hashlib.sha256(raw).hexdigest()==boundary.ATTENTION_CONTROL_SHA256
    files=boundary.sealed_archive_files(raw)
    assert json.loads(files['p177_k0_control.json'])['task_done'] is False
    manifest=json.loads(files['manifest.json']); run=json.loads(files['adaptive/run.json'])
    posts,_=archived_metadata()
    baseline=dict(identity=manifest['baseline_identity'],postwrites=posts,runs=dict(vanilla=run))
    path=tmp_path/boundary.ATTENTION_CONTROL_NAME;path.write_bytes(raw)
    reference=boundary.attention_control_reference(path,baseline)
    assert reference['archive_sha256']==boundary.ATTENTION_CONTROL_SHA256
    path.write_bytes(raw+b'bad')
    with pytest.raises(EvidenceError):boundary.attention_control_reference(path,baseline)
