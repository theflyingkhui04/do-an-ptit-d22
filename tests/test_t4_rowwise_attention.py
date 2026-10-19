"""CPU execution of patched host dispatch and integer/index contracts.

No test here simulates Triton floating-point arithmetic or validates GPU output.
"""
import ast
import copy
import enum
import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from src.server import t4_rowwise_attention as patch
from experiments import protocol_b_boundary_sources as historical
from experiments.protocol_b import EvidenceError


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
ORIGINALS = json.loads(gzip.decompress((FIXTURES / "protocol_b_boundary_kv_v030.json.gz").read_bytes()))
ORIGINALS = {name: ORIGINALS[name].encode() for name in patch.HASHES}
ORIGINALS[patch.HELPER] = (FIXTURES / "m203_v030_triton_attention_helpers.py").read_bytes()


def tree(name=patch.OPS, pristine=False):
    raw = ORIGINALS[name] if pristine else patch.transform(name, ORIGINALS[name])
    return ast.parse(raw)


def function(name, module=None):
    return copy.deepcopy(next(n for n in (module or tree()).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))


def execute_functions(names, namespace, module=None):
    nodes = []
    for name in names:
        n = function(name, module)
        n.decorator_list = []
        n.returns = None
        for arg in n.args.args:
            arg.annotation = None
        nodes.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<pinned-source>", "exec"), namespace)
    return namespace


class Mode(enum.IntEnum):
    NONE = 0
    INT4_PER_TOKEN_HEAD = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    FP8_PER_TENSOR = 4


class Tensor:
    def __init__(self, shape, dtype=torch.float16, device=None, contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")
        self.contiguous = contiguous
        self.ndim = len(shape)

    def __rtruediv__(self, other):
        return self

    def __len__(self):
        return self.shape[0]

    def element_size(self):
        return 2

    def is_contiguous(self):
        return self.contiguous

    def stride(self, axis=None):
        strides = tuple(__import__("math").prod(self.shape[i+1:]) for i in range(len(self.shape)))
        return strides if axis is None else strides[axis]


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def call(**kwargs):
            self.calls.append((grid, kwargs))
        return call


def host(pristine=False):
    producer, reducer = Launch(), Launch()
    ns = dict(torch=torch, KVQuantMode=Mode, Any=object,
              triton=NS(next_power_of_2=lambda x: 1 << (x-1).bit_length()),
              current_platform=NS(is_cuda=lambda: True,
                                  get_device_capability=lambda: NS(to_int=lambda: 75),
                                  is_device_capability_family=lambda x: False,
                                  fp8_dtype=lambda: torch.float8_e4m3fn),
              is_batch_invariant=False, kernel_unified_attention=producer, reduce_segments=reducer)
    execute_functions(["_is_gemma3_attention", "_get_tile_size", "unified_attention"], ns,
                      tree(pristine=pristine))
    return ns, producer, reducer


def args(q=4, capacity=32):
    return dict(q=Tensor((q,28,128)), k=Tensor((64,16,4,128)), v=Tensor((64,16,4,128)),
                out=Tensor((q,28,128)), cu_seqlens_q=Tensor((2,),torch.int32),
                max_seqlen_q=q, seqused_k=Tensor((1,),torch.int32), max_seqlen_k=181,
                softmax_scale=128**-0.5, causal=True, window_size=(-1,-1),
                block_table=Tensor((1,256),torch.int32), softcap=0,
                q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=32, num_par_softmax_segments=16,
                softmax_segm_output=Tensor((capacity,28,16,128),torch.float32),
                softmax_segm_max=Tensor((capacity,28,16),torch.float32),
                softmax_segm_expsum=Tensor((capacity,28,16),torch.float32),
                rowwise_decode_max_q=5)


@pytest.mark.parametrize("q", [2,3,4,5,8])
def test_supported_host_dispatch_keeps_full_query_and_one_launch(q):
    ns, producer, reducer = host()
    a = args(q)
    a["rowwise_decode_max_q"] = q
    ns["unified_attention"](**a)
    assert len(producer.calls) == len(reducer.calls) == 1
    grid, kw = producer.calls[0]
    assert grid == (q,4,16)
    assert kw["query_ptr"] is a["q"] and kw["output_ptr"] is a["out"]
    assert kw["query_start_len_ptr"] is a["cu_seqlens_q"]
    assert kw["seq_lens_ptr"] is a["seqused_k"]
    assert kw["key_cache_ptr"] is a["k"] and kw["value_cache_ptr"] is a["v"]
    assert kw["block_tables_ptr"] is a["block_table"]
    assert kw["ROWWISE_DECODE"] is kw["IS_3D"] is True
    assert kw["BLOCK_M"] == 16 and kw["BLOCK_Q"] == 2 and kw["TILE_SIZE"] == 16
    assert reducer.calls[0][0] == (q,28)
    assert reducer.calls[0][1]["ROWWISE_DECODE"] is True


UNSUPPORTED = [
    {"rowwise_decode_max_q":0}, {"rowwise_decode_max_q":3},
    {"causal":False}, {"window_size":(15,0)}, {"softcap":1},
    {"use_td":True}, {"mm_prefix_clamp_sliding_window":True},
    {"sinks":Tensor((28,))}, {"alibi_slopes":Tensor((28,))},
    {"use_alibi_sqrt":True}, {"qq_bias":Tensor((4,4))},
    {"rswa_window":16}, {"rswa_prefix_lens":Tensor((1,),torch.int32)},
    {"mm_prefix_range":Tensor((1,1,2),torch.int32)},
    {"q_descale":Tensor((1,))}, {"output_scale":Tensor((1,))},
    {"kv_quant_mode":Mode.FP8_PER_TENSOR},
    {"num_par_softmax_segments":8}, {"seq_threshold_3D":None},
    {"softmax_segm_output":None}, {"softmax_segm_max":None}, {"softmax_segm_expsum":None},
    {"softmax_segm_output":Tensor((3,28,16,128),torch.float32)},
    {"softmax_segm_max":Tensor((3,28,16),torch.float32)},
    {"softmax_segm_expsum":Tensor((3,28,16),torch.float32)},
    {"softmax_segm_max":Tensor((32,28,16),torch.float16)},
    {"softmax_segm_output":Tensor((32,28,16,128),torch.float32,contiguous=False)},
    {"softmax_segm_expsum":Tensor((32,27,16),torch.float32)},
    {"q":Tensor((4,28,128),torch.bfloat16)},
    {"q":Tensor((4,28,128),device=torch.device("cpu"))},
    {"q":Tensor((4,32,128)), "out":Tensor((4,32,128))},
    {"seqused_k":Tensor((2,),torch.int32), "cu_seqlens_q":Tensor((3,),torch.int32)},
    {"max_seqlen_k":3},
]


@pytest.mark.parametrize("updates", UNSUPPORTED)
def test_unsupported_dispatch_matches_original_exactly(updates):
    a = args()
    a.update(updates)
    new, prod, red = host()
    old, old_prod, old_red = host(pristine=True)
    new["unified_attention"](**a)
    old["unified_attention"](**{k:v for k,v in a.items() if k != "rowwise_decode_max_q"})
    assert prod.calls[0][1].pop("ROWWISE_DECODE") is False
    for _, kw in red.calls:
        assert kw.pop("ROWWISE_DECODE") is False
    assert prod.calls == old_prod.calls and red.calls == old_red.calls


@pytest.mark.parametrize("cap", [None,70,80,89])
def test_other_gpu_or_unknown_capability_falls_back(cap):
    ns, prod, _ = host()
    ns["current_platform"].get_device_capability = lambda: None if cap is None else NS(to_int=lambda:cap)
    ns["unified_attention"](**args())
    assert prod.calls[0][1]["ROWWISE_DECODE"] is False


@pytest.mark.parametrize("setting", ["not_cuda", "batch_invariant"])
def test_platform_policy_falls_back(setting):
    ns, prod, _ = host()
    if setting == "not_cuda":
        ns["current_platform"].is_cuda = lambda:False
    else:
        ns["is_batch_invariant"] = True
    ns["unified_attention"](**args())
    assert prod.calls[0][1]["ROWWISE_DECODE"] is False


def test_exact_scratch_capacity_is_sufficient():
    ns, prod, _ = host()
    ns["unified_attention"](**args(capacity=4))
    assert prod.calls[0][1]["ROWWISE_DECODE"] is True


@pytest.mark.parametrize("q", [1,6])
def test_native_decode_and_over_capacity_queries_unchanged(q):
    a = args(q)
    ns, prod, red = host()
    old, op, ore = host(pristine=True)
    ns["unified_attention"](**a)
    old["unified_attention"](**{k:v for k,v in a.items() if k != "rowwise_decode_max_q"})
    assert prod.calls[0][1].pop("ROWWISE_DECODE") is False
    for _, kw in red.calls:
        assert kw.pop("ROWWISE_DECODE") is False
    assert prod.calls == op.calls and red.calls == ore.calls


class Ptr:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, n):
        return Ptr(self.data, self.offset+n)


def integer_env(q=4, final=181):
    tl = NS(load=lambda p:p.data[p.offset], int32=int, arange=torch.arange,
            int1=torch.bool, int64=torch.int64,
            where=lambda c,a,b:torch.where(c,torch.as_tensor(a),torch.as_tensor(b)))
    ns = dict(tl=tl)
    helper_tree = ast.parse(ORIGINALS[patch.HELPER])
    execute_functions(["find_seq_idx", "cdiv_fn"], ns, helper_tree)
    execute_functions(["_rowwise_decode_state"], ns)
    ns.update(query_start_len_ptr=Ptr([0,q]), seq_lens_ptr=Ptr([final]),
              num_seqs=1, BLOCK_Q=2, BLOCK_M=16, TILE_SIZE=16,
              NUM_SEGMENTS_PER_SEQ=16, num_queries_per_kv=7, num_query_heads=28,
              HEAD_SIZE=128, HEAD_SIZE_PADDED=128, query_stride_0=28*128,
              query_stride_1=128)
    return ns


def run_nodes(nodes, ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),
                 "<pinned-contract>", "exec"), ns)


def row_branch(name):
    return next(n for n in function(name).body if isinstance(n,ast.If)
                and isinstance(n.test,ast.Name) and n.test.id=="ROWWISE_DECODE")


@pytest.mark.parametrize("final", [4,15,16,17,178,181,255,256,257,258,511,512,513,4096])
def test_producer_reducer_use_identical_effective_length_and_geometry(final):
    producer = function("kernel_unified_attention")
    reduction = function("reduce_segments")
    segment_if = next(n for n in producer.body if isinstance(n,ast.If)
                      and isinstance(n.test,ast.Name) and n.test.id=="IS_3D")
    reduction_geometry = [n for n in reduction.body if isinstance(n,ast.Assign)
                          and isinstance(n.targets[0],ast.Name)
                          and n.targets[0].id in ("num_segments","tiles_per_segment","act_num_segments")]
    lengths = []
    for r in range(4):
        ns = integer_env(final=final)
        ns.update(q_block_global_idx=r, query_token_idx=r, segm_idx=0, seq_idx=0)
        run_nodes(row_branch("kernel_unified_attention").body, ns)
        lengths.append(ns["seq_len"])
        assert ns["cur_batch_in_all_start_index"] == r
        assert ns["cur_batch_query_len"] == 1 and ns["q_block_local_idx"] == 0
        run_nodes(segment_if.body[:1], ns)
        producer_len, producer_tiles = ns["seq_len"], ns["tiles_per_segment"]
        run_nodes(row_branch("reduce_segments").body + reduction_geometry, ns)
        assert ns["seq_len"] == producer_len == final-4+r+1
        assert ns["tiles_per_segment"] == producer_tiles
        # Exact boundaries and active segment count equal serial q_len=1.
        serial = integer_env(q=1, final=producer_len)
        serial.update(query_token_idx=0,seq_idx=0)
        run_nodes(row_branch("reduce_segments").orelse + reduction_geometry, serial)
        assert (serial["tiles_per_segment"],serial["act_num_segments"]) == (
            producer_tiles,ns["act_num_segments"])
        bounds = [(i*producer_tiles*16,min((i+1)*producer_tiles*16,producer_len))
                  for i in range(ns["act_num_segments"])]
        assert bounds[0][0]==0 and bounds[-1][1]==producer_len
        assert all(a[1]==b[0] for a,b in zip(bounds,bounds[1:]))
    if final == 181:
        assert lengths == [178,179,180,181]


def test_gqa7_lane_and_scratch_index_contract_matches_native_single_query():
    body = function("kernel_unified_attention").body
    first = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="offs_m")
    last = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="query_mask_1")
    scratch = next(n for n in ast.walk(function("kernel_unified_attention")) if isinstance(n,ast.Assign)
                   and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="segm_output_offset")
    all_addresses = set()
    for r in range(4):
        for kv_head in range(4):
            ns = integer_env()
            ns.update(q_block_global_idx=r,kv_head_idx=kv_head,segm_idx=0)
            run_nodes(row_branch("kernel_unified_attention").body + body[first:last+1],ns)
            valid = ns["query_mask_0"] & ns["query_mask_1"]
            assert valid.tolist() == [True]*7+[False]*9
            assert ns["query_offset_0"][valid].tolist() == [r]*7
            assert ns["query_offset_1"][valid].tolist() == list(range(kv_head*7,(kv_head+1)*7))
            # The pointer path uses precisely the original q_len=1 lane mapping.
            serial = integer_env(q=1, final=178+r)
            serial.update(cur_batch_query_len=1,q_block_local_idx=0,
                          cur_batch_in_all_start_index=0,kv_head_idx=kv_head)
            run_nodes(body[first:last+1],serial)
            assert torch.equal(valid,serial["query_mask_0"] & serial["query_mask_1"])
            for seg in range(16):
                ns["segm_idx"] = seg
                run_nodes([scratch],ns)
                addresses = set(ns["segm_output_offset"][valid].flatten().tolist())
                assert not (all_addresses & addresses)
                all_addresses |= addresses
    assert all_addresses == set(range(4*28*16*128))


def test_pinned_causal_mask_excludes_future_speculative_kv():
    ns=integer_env(final=257)
    helper_tree=ast.parse(ORIGINALS[patch.HELPER])
    execute_functions(["compute_kv_seq_mask"],ns,helper_tree)
    offsets=torch.arange(258)
    for r in range(4):
        length=ns["_rowwise_decode_state"](ns["query_start_len_ptr"],ns["seq_lens_ptr"],r,0)
        mask=ns["compute_kv_seq_mask"](
            torch.tensor([[length-1]]),offsets,0,length,None,
            SLIDING_WINDOW=0,USE_MM_PREFIX=False,MAX_MM_RANGES=0)
        assert mask[0,:length].all()
        assert not mask[0,length:].any()
        assert torch.equal(mask, offsets[None,:] <= 253+r)


def test_target_only_configured_capacity_and_piecewise_gate():
    backend = tree(patch.BACKEND)
    cls = next(n for n in backend.body if isinstance(n,ast.ClassDef) and n.name=="TritonAttentionMetadataBuilder")
    init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    assignment = next(n for n in init.body if isinstance(n,ast.Assign)
                      and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr=="rowwise_decode_max_q")
    for target,spec_on,dtype,mode,capacity,expected in [
        (True,True,torch.float16,"PIECEWISE",4,5),
        (True,True,torch.float16,"PIECEWISE",2,3),
        (True,True,torch.float16,"PIECEWISE",8,9),
        (False,True,torch.float16,"PIECEWISE",4,0),
        (True,False,torch.float16,"PIECEWISE",4,0),
        (True,True,torch.bfloat16,"PIECEWISE",4,0),
        (True,True,torch.float16,"FULL",4,0),
    ]:
        model=NS(dtype=dtype)
        spec=NS(num_speculative_tokens=capacity,target_model_config=model if target else NS()) if spec_on else None
        ns=dict(self=NS(),spec=spec,model_config=model,torch=torch,
                vllm_config=NS(compilation_config=NS(cudagraph_mode=mode)),
                CUDAGraphMode=NS(PIECEWISE="PIECEWISE"))
        run_nodes([assignment],ns)
        assert ns["self"].rowwise_decode_max_q==expected


def test_exact_source_auth_and_historical_profiles_unchanged():
    for name,raw in ORIGINALS.items():
        patched=patch.transform(name,raw)
        patch.validate_transform(name,raw,patched)
        with pytest.raises(ValueError):
            patch.validate_transform(name,raw,patched+b"\n")
        with pytest.raises(ValueError):
            patch.transform(name,raw+b"\n")
    for name in patch.HASHES:
        assert patch.HASHES[name]==historical.HASHES[name]
        raw=ORIGINALS[name]
        historical.validate_transform(name,raw,historical.transform(name,raw,attention=True),attention=True)
        with pytest.raises(EvidenceError,match="Unapproved inference AST|Missing/duplicate attention diagnostic hook"):
            historical.validate_transform(name,raw,patch.transform(name,raw),attention=True)


def test_review_diff_is_exact_generated_transform():
    artifact=ROOT/"patches/vllm-0.30.0-t4-rowwise-attention.patch"
    assert artifact.read_bytes()==patch.source_diff(ORIGINALS).encode()


def test_cache_update_and_model_forward_surfaces_unchanged():
    old,new=tree(patch.BACKEND,True),tree(patch.BACKEND)
    for name in ("do_kv_cache_update","do_rope_and_kv_cache_update"):
        a=next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
    # No new stores or model/cache calls are introduced in either kernel.
    for name in ("kernel_unified_attention","reduce_segments"):
        oldfn,newfn=function(name,tree(pristine=True)),function(name)
        stores=lambda fn:[ast.dump(n) for n in ast.walk(fn) if isinstance(n,ast.Call)
                          and isinstance(n.func,ast.Attribute) and n.func.attr=="store"]
        assert stores(oldfn)==stores(newfn)
        calls={ast.unparse(n.func) for n in ast.walk(newfn) if isinstance(n,ast.Call)}
        assert not any("forward" in n or "cache_update" in n for n in calls)


def cli(tmp_path,command,root,evidence=None,ok=True):
    cmd=[sys.executable,"-m","src.server.t4_rowwise_attention",command,"--source-root",str(root)]
    if evidence is not None:
        cmd += ["--evidence-dir",str(evidence)]
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    assert (result.returncode==0)==ok, result.stdout+result.stderr
    return result


def source_root(tmp_path):
    root=tmp_path/"vllm"
    for name,raw in ORIGINALS.items():
        dest=root/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
    return root


def test_install_verify_restore_and_dependency_auth(tmp_path):
    root=source_root(tmp_path)
    evidence=tmp_path/"evidence"
    diff=cli(tmp_path,"diff",root).stdout
    assert diff.count("--- a/vllm/")==2
    cli(tmp_path,"install",root,evidence)
    cli(tmp_path,"verify",root,evidence)
    cli(tmp_path,"install",root,evidence,ok=False)
    cli(tmp_path,"restore",root,evidence)
    assert all((root/n).read_bytes()==raw for n,raw in ORIGINALS.items())
    (root/patch.HELPER).write_bytes(b"drift")
    cli(tmp_path,"install",root,tmp_path/"invalid",ok=False)
    assert not (tmp_path/"invalid").exists()
    assert all((root/n).read_bytes()==ORIGINALS[n] for n in patch.HASHES)


def test_restore_recovers_authenticated_partial_install(tmp_path):
    root=source_root(tmp_path)
    evidence=tmp_path/"evidence"
    cli(tmp_path,"install",root,evidence)
    (root/patch.OPS).write_bytes(ORIGINALS[patch.OPS])
    cli(tmp_path,"verify",root,evidence,ok=False)
    cli(tmp_path,"restore",root,evidence)
    assert all((root/n).read_bytes()==raw for n,raw in ORIGINALS.items())


def test_restore_rejects_unknown_mutations_before_any_write(tmp_path):
    root=source_root(tmp_path)
    evidence=tmp_path/"evidence"
    cli(tmp_path,"install",root,evidence)
    (root/patch.OPS).write_bytes(b"unknown runtime change")
    cli(tmp_path,"restore",root,evidence,ok=False)
    assert (root/patch.BACKEND).read_bytes()==patch.transform(patch.BACKEND,ORIGINALS[patch.BACKEND])
