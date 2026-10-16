"""CPU source/dispatch/arithmetic checks. No Triton compilation or GPU claims."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from experiments.m2_03_targeted import sources as evidence_sources
from src.server import triton_spec_decode_patch as patch


def source(patched=True):
    raw = evidence_sources.fixture_sources()[patch.SOURCE].encode()
    return patch.transform(raw) if patched else raw


class Metadata:
    """Only host metadata may be accessed by the wrapper, never tensor contents."""
    def __init__(self, shape, dtype="half", device=None, contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.device = device or NS(type="cuda")
        self.ndim = len(shape)
        self.contiguous = contiguous

    def __len__(self): return self.shape[0]
    def stride(self, axis=None):
        strides = tuple(int(np.prod(self.shape[i + 1:])) for i in range(self.ndim))
        return strides if axis is None else strides[axis]
    def is_contiguous(self): return self.contiguous
    def element_size(self): return 2
    def __bool__(self): raise AssertionError("device tensor truthiness")
    def __getattr__(self, key):
        if key.startswith("__"): raise AttributeError(key)
        raise AssertionError("tensor read: " + key)


class Kernel:
    def __init__(self): self.calls = []
    def __getitem__(self, grid):
        def run(**kw): self.calls.append((grid, kw))
        return run


def wrapper(patched=True, invariant=False):
    nodes = [n for n in ast.parse(source(patched)).body if isinstance(n, ast.FunctionDef)
             and n.name in ("unified_attention", "_get_tile_size", "_is_gemma3_attention")]
    main, reduce = Kernel(), Kernel()
    ns = dict(torch=NS(Tensor=Metadata, float16="half", float32="float"), Any=object,
              KVQuantMode=NS(NONE=0, INT4_PER_TOKEN_HEAD=4,
                             INT8_PER_TOKEN_HEAD=2, FP8_PER_TOKEN_HEAD=3),
              current_platform=NS(fp8_dtype=lambda: "fp8",
                                  is_device_capability_family=lambda _: False),
              triton=NS(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
              is_batch_invariant=invariant, kernel_unified_attention=main,
              reduce_segments=reduce)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<wrapper>", "exec"), ns)
    return ns["unified_attention"], main, reduce


def arguments(q_len=3):
    device = NS(type="cuda")
    def tensor(shape, dtype="half", **kw): return Metadata(shape, dtype, device, **kw)
    return dict(q=tensor((q_len, 28, 128)), k=tensor((40, 16, 4, 128)),
                v=tensor((40, 16, 4, 128)), out=tensor((q_len, 28, 128)),
                cu_seqlens_q=tensor((2,)), max_seqlen_q=q_len,
                seqused_k=tensor((1,)), max_seqlen_k=556 + q_len,
                softmax_scale=128**-0.5, causal=True, window_size=(-1, -1),
                block_table=tensor((1, 40)), softcap=0., q_descale=None,
                k_descale=None, v_descale=None, seq_threshold_3D=32,
                num_par_softmax_segments=16,
                softmax_segm_output=tensor((32, 28, 16, 128), "float"),
                softmax_segm_max=tensor((32, 28, 16), "float"),
                softmax_segm_expsum=tensor((32, 28, 16), "float"))


@pytest.mark.parametrize("q_len", [3])
@pytest.mark.parametrize("kv_len", [1, 17, 33, 559, 4096])
def test_supported_query_launches_guarded_splitkv(q_len, kv_len):
    fn, main, reduce = wrapper()
    args = arguments(q_len)
    args["max_seqlen_k"] = kv_len
    fn(**args)
    grid, kw = main.calls[0]
    assert grid == (q_len // 2 + 1, 4, 16)
    assert kw["IS_3D"] is kw["_SHORT_QUERY_3D_GUARDS"] is True
    assert kw["TILE_SIZE"] == 16 and kw["BLOCK_M"] == 16 and kw["BLOCK_Q"] == 2
    assert reduce.calls[0][0] == (q_len, 28)
    assert reduce.calls[0][1]["_SHORT_QUERY_3D_GUARDS"] is True
    assert kw["query_ptr"] is args["q"]


UNSUPPORTED = [
    ("causal", False), ("causal", Metadata((1,))),
    ("window_size", (15, 0)), ("window_size", (15, 15)),
    ("mm_prefix_range", Metadata((1, 1, 2))),
    ("mm_prefix_clamp_sliding_window", True),
    ("rswa_prefix_lens", Metadata((1,))), ("rswa_window", 16),
    ("use_td", True), ("alibi_slopes", Metadata((28,))),
    ("use_alibi_sqrt", True), ("qq_bias", Metadata((3, 3))),
    ("sinks", Metadata((28,))), ("softcap", 10.), ("output_scale", 1.),
    ("kv_quant_mode", 1), ("seq_threshold_3D", 0), ("seq_threshold_3D", None),
    ("num_par_softmax_segments", None), ("num_par_softmax_segments", 0),
    ("num_par_softmax_segments", 3), ("max_seqlen_k", 0),
    ("softmax_segm_output", None), ("softmax_segm_max", None),
    ("softmax_segm_expsum", None), ("seqused_k", Metadata((2,))),
]


def without_guards(calls):
    return [(grid, {k: v for k, v in kw.items() if k != "_SHORT_QUERY_3D_GUARDS"})
            for grid, kw in calls]


@pytest.mark.parametrize("kv_len", [1, 17, 33, 559, 4096])
def test_q2_keeps_exact_stock_path(kv_len):
    args = arguments(2)
    args["max_seqlen_k"] = kv_len
    new, main, reduce = wrapper()
    old, old_main, old_reduce = wrapper(False)
    new(**args)
    old(**args)
    assert main.calls[0][1]["IS_3D"] is False
    assert main.calls[0][1]["_SHORT_QUERY_3D_GUARDS"] is False
    assert without_guards(main.calls) == old_main.calls
    assert reduce.calls == old_reduce.calls == []


def test_tuned_large_head_keeps_exact_stock_path():
    args = arguments()
    for name in ("q", "k", "v", "out", "softmax_segm_output"):
        buf = args[name]
        args[name] = Metadata((*buf.shape[:-1], 256), buf.dtype, buf.device)
    args["softmax_scale"] = 256**-0.5
    # The same inputs are otherwise eligible, so fallback tests the tuned gate.
    control, control_main, _ = wrapper()
    control(**args)
    assert control_main.calls[0][1]["IS_3D"] is True
    new, main, reduce = wrapper()
    old, old_main, old_reduce = wrapper(False)
    for fn in (new, old):
        fn.__globals__["current_platform"].is_device_capability_family = lambda n: n == 100
        fn(**args)
    kw = main.calls[0][1]
    # These settings independently prove the actual tuned_large_head branch ran.
    assert kw["TILE_SIZE"] == 128 and kw["BLOCK_M"] == 32 and kw["BLOCK_Q"] == 4
    assert kw["num_warps"] == 8 and kw["num_stages"] == 2
    assert kw["IS_3D"] is kw["_SHORT_QUERY_3D_GUARDS"] is False
    assert without_guards(main.calls) == old_main.calls
    assert reduce.calls == old_reduce.calls == []


@pytest.mark.parametrize("name,value", UNSUPPORTED)
def test_unsupported_launch_is_identical_to_stock(name, value):
    args = arguments()
    args[name] = value
    if name == "window_size": args["chunk_lookback"] = 1  # block-local too
    new, main, reduce = wrapper()
    old, old_main, old_reduce = wrapper(False)
    new(**args)
    old(**args)
    assert main.calls[0][1]["IS_3D"] is False
    assert main.calls[0][1]["_SHORT_QUERY_3D_GUARDS"] is False
    assert without_guards(main.calls) == old_main.calls
    assert without_guards(reduce.calls) == old_reduce.calls


@pytest.mark.parametrize("q_len", [0, 1, 4, 8, 16])
@pytest.mark.parametrize("invariant", [False, True])
def test_other_query_lengths_and_q1_keep_stock_launch(q_len, invariant):
    args = arguments(q_len)
    new, main, reduce = wrapper(invariant=invariant)
    old, old_main, old_reduce = wrapper(False, invariant)
    new(**args)
    old(**args)
    assert without_guards(main.calls) == old_main.calls
    assert without_guards(reduce.calls) == old_reduce.calls
    assert main.calls[0][1]["_SHORT_QUERY_3D_GUARDS"] is False


@pytest.mark.parametrize("name", ["softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"])
@pytest.mark.parametrize("problem", ["rows", "heads", "segments", "rank", "dtype", "device", "strides"])
def test_each_scratch_buffer_fails_closed(name, problem):
    args = arguments()
    buf = args[name]
    shape = list(buf.shape)
    if problem == "rows": shape[0] = 2
    if problem == "heads": shape[1] += 1
    if problem == "segments": shape[2] += 1
    if problem == "rank": shape.append(1)
    args[name] = Metadata(tuple(shape), "half" if problem == "dtype" else "float",
                          NS(type="cpu") if problem == "device" else buf.device,
                          contiguous=problem != "strides")
    fn, main, reduce = wrapper()
    fn(**args)
    assert not main.calls[0][1]["IS_3D"] and not reduce.calls


def test_wrong_scratch_head_width_and_padded_query_fall_back():
    for name, shape in [("softmax_segm_output", (32, 28, 16, 256)), ("q", (4, 28, 128))]:
        args = arguments()
        args[name] = Metadata(shape, "float" if name.startswith("softmax") else "half",
                              args["q"].device)
        fn, main, reduce = wrapper()
        fn(**args)
        assert not main.calls[0][1]["IS_3D"] and not reduce.calls


@pytest.mark.parametrize("name", ["q", "k", "v", "out"])
def test_non_fp16_inputs_keep_stock_path(name):
    args = arguments()
    buf = args[name]
    args[name] = Metadata(buf.shape, "bf16", buf.device)
    fn, main, reduce = wrapper()
    fn(**args)
    assert not main.calls[0][1]["IS_3D"] and not reduce.calls


def test_batch_invariant_short_query_keeps_stock_path():
    fn, main, reduce = wrapper(invariant=True)
    fn(**arguments())
    assert not main.calls[0][1]["IS_3D"] and not reduce.calls


@pytest.mark.parametrize("mode", [2, 3])
def test_per_token_kv_quantization_keeps_stock_launch(mode):
    args = arguments()
    args.update(kv_quant_mode=mode, k_scale_cache=Metadata((40, 16, 4)),
                v_scale_cache=Metadata((40, 16, 4)))
    new, main, reduce = wrapper()
    old, old_main, old_reduce = wrapper(False)
    new(**args)
    old(**args)
    assert not main.calls[0][1]["IS_3D"] and not reduce.calls
    assert without_guards(main.calls) == old_main.calls


def test_int4_delegation_unchanged(monkeypatch):
    import sys
    import types
    delegated = []
    module = types.ModuleType("vllm.v1.attention.ops.int4_per_token_head")
    module.unified_attention_int4 = lambda **kw: delegated.append(kw)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = arguments()
    args["kv_quant_mode"] = 4
    for patched in (False, True):
        fn, main, reduce = wrapper(patched)
        fn(**args)
        assert not main.calls and not reduce.calls
    assert len(delegated) == 2 and delegated[0] == delegated[1]


class Ptr:
    def __init__(self, data, offset=0): self.data, self.offset = data.flatten(), offset
    def __add__(self, offset): return Ptr(self.data, self.offset + offset)


class CPUtl:
    """Execute the actual reducer body on CPU, including pointer masks."""
    constexpr = object
    int64, int32, float32, int1 = torch.int64, torch.int32, torch.float32, torch.bool
    where, exp, sum, max, arange = torch.where, torch.exp, torch.sum, torch.max, torch.arange

    def __init__(self, row=0): self.row, self.loads = row, []
    def program_id(self, axis): return torch.tensor(self.row if axis == 0 else 0)
    def full(self, shape, value, dtype): return torch.full(shape, int(value), dtype=dtype)
    def load(self, ptr, mask=None, other=0.):
        self.loads.append(ptr.data)
        indices = torch.as_tensor(ptr.offset).long()
        if mask is None: return ptr.data[indices]
        indices, mask = torch.broadcast_tensors(indices, mask)
        return torch.where(mask, ptr.data[torch.where(mask, indices, 0)], other)
    def store(self, ptr, value, mask):
        indices = torch.as_tensor(ptr.offset).long()
        ptr.data[indices[mask]] = value[mask]


def reducer_run(maxima, mass, partial, *, guarded=True, row=0, query_end=3, seq_len=17):
    fn = next(n for n in ast.parse(source()).body
              if isinstance(n, ast.FunctionDef) and n.name == "reduce_segments")
    fn.decorator_list = []
    tl = CPUtl(row)
    ns = dict(tl=tl, float8_info=NS(min=-448, max=448),
              find_seq_idx=lambda *a: 0, cdiv_fn=lambda x, y: (x + y - 1) // y)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<reducer>", "exec"), ns)
    # 16 segments, 128 values per head. Only two segments are active at KV=17.
    def data(x): return torch.tensor(x, dtype=torch.float32)
    segmax, sums = torch.full((3, 16), float("nan")), torch.full((3, 16), float("nan"))
    values = torch.full((3, 16, 128), float("nan"))
    count = len(maxima)
    segmax[:, :count], sums[:, :count] = data(maxima), data(mass)
    values[:, :count] = data(partial)[:, None]
    out = torch.full((3, 128), -123.)
    ns["reduce_segments"](Ptr(out), Ptr(values), Ptr(segmax), Ptr(sums),
        Ptr(torch.tensor([seq_len])), 1, 1, 1., 128, 128, 40, 16, 128, 128,
        Ptr(torch.tensor([0, query_end])), 2, 16, False,
        _SHORT_QUERY_3D_GUARDS=guarded)
    return out, tl


def epilogue(maxima, mass, guarded=True):
    fn = next(n for n in ast.parse(source()).body
              if isinstance(n, ast.FunctionDef) and n.name == "kernel_unified_attention")
    guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "_SHORT_QUERY_3D_GUARDS")
    ns = dict(tl=NS(where=np.where), M=np.array(maxima), L=np.array(mass),
              _SHORT_QUERY_3D_GUARDS=guarded)
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "<epilogue>", "exec"), ns)
    return ns["M"]


def test_actual_guards_negative_logits_and_all_empty_reduction():
    maxima = epilogue([-1000., 0.], [1., 0.])
    out, _ = reducer_run(maxima, [1., 0.], [2., 0.])
    assert torch.all(out[0] == 2.)
    # Removing the epilogue loses the strongly negative segment.
    out, _ = reducer_run(epilogue([-1000., 0.], [1., 0.], False), [1., 0.], [2., 0.])
    assert torch.all(out[0] == 0.)
    empty = epilogue([0., 0.], [0., 0.])
    out, _ = reducer_run(empty, [0., 0.], [0., 0.])
    assert torch.all(out[0] == 0.)
    out, _ = reducer_run(empty, [0., 0.], [0., 0.], guarded=False)
    assert torch.isnan(out[0]).all()  # Reducer guard independently necessary.


@pytest.mark.parametrize("guarded", [True, False])
def test_nonempty_reducer_unchanged_and_unvisited_segments_ignored(guarded):
    out, _ = reducer_run([2., 1.], [1., 1.], [2., 4.], guarded=guarded)
    expected = (2 + 4 * np.exp(-1)) / (1 + np.exp(-1))
    assert torch.allclose(out[0], torch.full((128,), expected))
    assert np.array_equal(epilogue([2., 1.], [1., 1.], guarded), [2., 1.])


@pytest.mark.parametrize("row,query_end,seq_len", [(2, 2, 17), (0, 0, 0), (0, 3, 0)])
def test_reducer_padding_and_zero_kv_return_before_scratch(row, query_end, seq_len):
    out, tl = reducer_run([0., 0.], [0., 0.], [0., 0.],
                          row=row, query_end=query_end, seq_len=seq_len)
    assert torch.all(out == -123.)
    assert all(not value.is_floating_point() for value in tl.loads)


def test_transform_roundtrip_and_pinned_install_restore(tmp_path):
    raw, patched = source(False), source()
    assert patch.digest(raw) == patch.PRISTINE_SHA256
    assert patch.pristine(patched) == raw
    compile(patched, patch.SOURCE, "exec")
    assert b"observer" not in patched and b"M203_" not in patched
    path = tmp_path / patch.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    assert patch.apply(tmp_path, "status")["state"] == "PRISTINE"
    for _ in range(2): assert patch.apply(tmp_path, "install")["state"] == "PATCHED"
    assert path.read_bytes() == patched
    for _ in range(2): assert patch.apply(tmp_path, "restore")["state"] == "PRISTINE"
    assert path.read_bytes() == raw


def test_removing_only_admission_and_constexpr_guards_recovers_entire_stock_ast():
    class Strip(ast.NodeTransformer):
        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id in (
                    "short_query_3d", "_SHORT_QUERY_3D_GUARDS"):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if any(isinstance(n, ast.Name) and n.id == "short_query_3d"
                   for n in ast.walk(node)):
                return None
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            node = self.generic_visit(node)
            if node.args.args[-1].arg == "_SHORT_QUERY_3D_GUARDS":
                assert node.args.defaults[-1].value is False
                node.args.args.pop()
                node.args.defaults.pop()
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            node.keywords = [kw for kw in node.keywords if kw.arg != "_SHORT_QUERY_3D_GUARDS"]
            return node

    assert ast.dump(Strip().visit(ast.parse(source()))) == ast.dump(ast.parse(source(False)))


@pytest.mark.parametrize("raw", [b"foreign", source() + b"\n", source(False) + b"\n",
    evidence_sources.transform(patch.SOURCE, source(False))],
    ids=["foreign", "patched-drift", "pristine-drift", "diagnostic"])
def test_foreign_partial_diagnostic_sources_refused_without_writes(tmp_path, raw):
    path = tmp_path / patch.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    for action in ("install", "restore", "status"):
        with pytest.raises(ValueError): patch.apply(tmp_path, action)
        assert path.read_bytes() == raw


def test_failed_replace_leaves_pristine_and_no_pending_file(tmp_path, monkeypatch):
    path = tmp_path / patch.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(source(False))
    def fail(*args): raise OSError("replacement failed")
    monkeypatch.setattr(patch.os, "replace", fail)
    with pytest.raises(OSError): patch.apply(tmp_path, "install")
    assert path.read_bytes() == source(False)
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("bundle,state", [
    ("complete-3999b36f90114ad289daf22ff932ae01.tar.gz", "REPRODUCED"),
    ("complete-a4d65ac6c375455fad050e5d04e41b1d.tar.gz", "NOT_REPRODUCED"),
    ("complete-20dca630cc004e2cb577b6e66fdfdec5.tar.gz", "REPRODUCED"),
    ("complete-605f6d6615014428bb99d0a0c384f68a.tar.gz", "NOT_REPRODUCED"),
    ("complete-051f50ab905f4b36ad0054ec30017cf1.tar.gz", "NOT_REPRODUCED"),
])
def test_preserved_evidence_still_validates(tmp_path, bundle, state):
    import json
    import tarfile
    from experiments.m2_03_targeted import analysis
    archive = Path(__file__).resolve().parents[1] / "experiments/runs/m2_03/diagnostics" / bundle
    with tarfile.open(archive) as tar:
        manifest = json.load(tar.extractfile("manifest.json"))
        for member in tar.getmembers():
            if not member.isfile() or not member.name.startswith("data/"): continue
            relative = Path(member.name).relative_to("data")
            assert not relative.is_absolute() and ".." not in relative.parts
            raw = tar.extractfile(member).read()
            expected = manifest["files"][relative.as_posix()]
            assert len(raw) == expected["size"] and patch.digest(raw) == expected["sha256"]
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    report = analysis.compare(tmp_path)
    assert report["state"] == state, report
    if "051f50" in bundle:
        assert report["first_divergence"] is None and report["exact_output_token_ids_equal"]
        assert report["captures"]["adaptive"]["selected_token"] == 9765
        assert report["attention_dispatch_causal_test"]["all_28_calls_forced_3D"]
    if "20dca" in bundle:
        assert report["first_divergence"] == 368
        assert report["captures"]["adaptive"]["selected_token"] == 128281
