"""CPU indexing, source transforms and evidence contracts; never CUDA emulation."""
import ast
import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from experiments.protocol_b import EvidenceError
from experiments import protocol_b_boundary_observer as observer
from experiments.protocol_b_boundary import compare_kv_histories
from experiments import protocol_b_boundary_sources as sources


def cache_fixture(shift=0, noncontiguous=False):
    heads, dim, block = 2, 3, 16
    base = torch.full((64, heads, block, 4 * dim if noncontiguous else 2 * dim),
                      -19, dtype=torch.float16)
    cache = base[..., ::2] if noncontiguous else base
    blocks = [i + shift for i in range(29)]
    table = torch.tensor([blocks, list(reversed(blocks))])
    expected = torch.arange(454 * heads * dim, dtype=torch.float32).reshape(454, heads, dim)
    expected = (expected / 16).to(torch.float16)
    for p in range(454):
        cache[blocks[p // block], :, p % block, :dim] = expected[p]
        cache[blocks[p // block], :, p % block, dim:] = -expected[p]
    current = blocks[454 // block] * block + 454 % block
    return cache, table, expected, current


class HistoryExtractionTests(unittest.TestCase):
    def extract(self, shift=0, noncontiguous=False):
        cache, table, expected, current = cache_fixture(shift, noncontiguous)
        return observer.hash_history(cache, table, 16, 2, 3, current), expected

    def test_logical_to_kernel_block_and_offset(self):
        _, table, _, current = cache_fixture(3)
        slots, actual, blocks = observer.history_slots(table, 16)
        self.assertEqual(slots[:17], list(range(48, 65)))
        self.assertEqual((slots[-1], actual), (31 * 16 + 5, current))
        self.assertEqual(blocks, list(range(3, 32)))

    def test_batch_row_is_not_request_state_index(self):
        _, table, _, current = cache_fixture(2)
        slots, _, _ = observer.history_slots(table, 16, row=0)
        self.assertEqual(slots[0], 32)
        self.assertNotEqual(slots[0], observer.history_slots(table, 16, row=1)[0][0])
        self.assertGreater(current, slots[-1])
        with self.assertRaises(IndexError):
            observer.history_slots(table, 16, row=22)

    def test_canonical_k_and_v_order_preserves_bits(self):
        result, expected = self.extract()
        for key, value in (("K", expected), ("V", -expected)):
            digest = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            self.assertEqual(result["digests"][key], digest)
        self.assertEqual(result["elements_per_component"], 454 * 2 * 3)
        self.assertEqual(result["bytes_per_component"], 454 * 2 * 3 * 2)

    def test_noncontiguous_cache_uses_actual_strides(self):
        contiguous, _ = self.extract()
        strided, _ = self.extract(noncontiguous=True)
        self.assertNotEqual(contiguous["cache_identity"]["strides"], strided["cache_identity"]["strides"])
        self.assertEqual(contiguous["digests"], strided["digests"])

    def test_original_fp16_signed_zero_bits_are_not_normalized(self):
        cache, table, _, current = cache_fixture()
        before = observer.hash_history(cache, table, 16, 2, 3, current)
        cache[0, 0, 0, 0] = -0.0
        after = observer.hash_history(cache, table, 16, 2, 3, current)
        self.assertNotEqual(before["digests"]["K"], after["digests"]["K"])

    def test_current_unused_padding_and_future_values_excluded(self):
        cache, table, _, current = cache_fixture()
        before = observer.hash_history(cache, table, 16, 2, 3, current)
        cache[28, :, 6:, :] = 100  # position 454 and unused tail
        cache[29:] = 200  # unused blocks / other request / future
        after = observer.hash_history(cache, table, 16, 2, 3, current)
        self.assertEqual(before["digests"], after["digests"])

    def test_subset_or_reordered_coverage_rejected(self):
        cache, table, _, current = cache_fixture()
        for positions in (range(453), [0, 453], range(455), reversed(range(454))):
            with self.assertRaises(EvidenceError):
                observer.hash_history(cache, table, 16, 2, 3, current, positions=positions)

    def test_current_write_alias_or_wrong_slot_rejected(self):
        cache, table, _, current = cache_fixture()
        with self.assertRaises(EvidenceError):
            observer.hash_history(cache, table, 16, 2, 3, current + 1)
        table[0, -1] = table[0, 0]
        with self.assertRaises(EvidenceError):
            observer.hash_history(cache, table, 16, 2, 3, 6)

    def test_unsupported_dtype_or_out_of_bounds_mapping_rejected(self):
        cache, table, _, current = cache_fixture()
        with self.assertRaises(EvidenceError):
            observer.hash_history(cache.float(), table, 16, 2, 3, current)
        table[0, 0] = 1000
        with self.assertRaises(EvidenceError):
            observer.hash_history(cache, table, 16, 2, 3, current)


def synthetic_evidence():
    """Synthetic serialized records test the offline gate, not runtime GPU correctness."""
    snapshots, boundaries, captures, events = {}, {}, {}, {}
    authenticated = dict(original_hashes=sources.HASHES, installed_hashes={"synthetic": "a" * 64},
                         observer_sha256="b" * 64)
    for lane in ("vanilla", "adaptive"):
        extracted, _, _, _ = cache_fixture(0 if lane == "vanilla" else 3)
        table = torch.tensor([list(range(0 if lane == "vanilla" else 3,
                                        29 if lane == "vanilla" else 32))])
        current = int(table[0, 28]) * 16 + 6
        content = observer.hash_history(extracted, table, 16, 2, 3, current)
        content["cache_identity"]["device"] = "cuda:0"
        interpretation = dict(backend="TRITON_ATTN", quantization="NONE", causal=True,
                              window=[-1, -1], use_cascade=False, scale=0.125, soft_cap=0.0)
        scale_values = dict(_q_scale=[1.0], _k_scale=[1.0], _v_scale=[1.0])
        name = "model.layers.0.self_attn.attn"
        identity = observer.tensor_identity(torch.zeros(1))
        binding = dict(inputs=dict(input_ids=identity, positions=identity), expected_layers=[name],
                       model_layer_count=1, layers={name: dict(
                           group=0, cache=content["cache_identity"], metadata={k: identity for k in
                             ("query_start_loc", "seq_lens", "block_table", "slot_mapping",
                              "softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum")},
                           scales=dict(_q_scale=identity, _k_scale=identity, _v_scale=identity),
                           heads=2, head_dim=3, spec_block_size=16, interpretation=interpretation,
                           launch=dict(max_seq_len=4096, max_query_len=1))})
        mode = "FULL" if lane == "vanilla" else "PIECEWISE"
        snapshots[lane] = dict(
            history_schema="m203.boundary.kv.v1",
            snapshot_id=lane, auth_id=lane, pid=2, phase="pre_forward", timestamp="2",
            request_id=lane, request_state_index=22 if lane == "vanilla" else 0,
            batch_row=0, active_query=[[454, 128367]],
            position=454, generated_index=279, expected_layers=[name], model_layer_count=1,
            runtime_binding=binding, dispatched_graph_mode=mode,
            capture_id="capture" if mode == "FULL" else None,
            descriptor="FULL-1" if mode == "FULL" else None,
            layers={name: dict(content, group=0, spec_block_size=16,
                               interpretation=dict(interpretation, scales=scale_values))},
            visibility=dict(query_start_loc=[0, 1], seq_len=455, historical_positions=[0, 454]),
            runtime_probe_changes_timing=True, copies=dict(device_wide_synchronize=False))
        boundaries[lane] = dict(request_id=lane, pid=2, timestamp="3",
                                active_query=[[454, 128367]], draft_count=0,
                                target_row=dict(position=454))
        if lane == "vanilla":
            boundaries[lane].update(model_runner="v2", idx_mapping=[22])
        captures[lane] = [dict(capture_id="capture", descriptor="FULL-1", auth_id=lane, pid=2,
                               timestamp="1", bindings=copy.deepcopy(binding))] if mode == "FULL" else []
        events[lane] = [dict(auth_id=lane, pid=2, timestamp="1", **copy.deepcopy(authenticated))]
    return snapshots, captures, events, authenticated, boundaries


class HistoryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.args = synthetic_evidence()
        self.snapshots, self.captures, self.events, self.authenticated, self.boundaries = self.args
        self.name = self.snapshots["vanilla"]["expected_layers"][0]

    def compare(self):
        return compare_kv_histories(*self.args)

    def invalid(self):
        result = self.compare()
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertFalse(result["historical_kv_state_equivalent"])

    def test_complete_equality_with_different_physical_ids(self):
        self.assertNotEqual(self.snapshots["vanilla"]["layers"][self.name]["physical_blocks"],
                            self.snapshots["adaptive"]["layers"][self.name]["physical_blocks"])
        result = self.compare()
        self.assertEqual(result["status"], "STATE_EQUIVALENT", result)
        self.assertTrue(result["historical_kv_state_equivalent"])

    def test_one_digest_mismatch_is_state_different(self):
        for component in ("K", "V"):
            original = copy.deepcopy(self.snapshots["adaptive"]["layers"][self.name]["digests"])
            self.snapshots["adaptive"]["layers"][self.name]["digests"][component] = "f" * 64
            result = self.compare()
            self.assertEqual(result["status"], "STATE_DIFFERENT", result)
            self.assertEqual(result["differences"], [self.name + ":" + component])
            self.snapshots["adaptive"]["layers"][self.name]["digests"] = original

    def test_matching_subset_is_invalid(self):
        self.snapshots["adaptive"]["layers"][self.name]["coverage"] = [0, 16]
        self.invalid()

    def test_missing_layer_is_invalid(self):
        self.snapshots["adaptive"]["layers"].clear()
        self.invalid()

    def test_missing_visibility_or_component_is_invalid(self):
        self.snapshots["adaptive"]["visibility"] = {}
        self.invalid()
        self.snapshots["adaptive"]["visibility"] = copy.deepcopy(self.snapshots["vanilla"]["visibility"])
        del self.snapshots["adaptive"]["layers"][self.name]["digests"]["V"]
        self.invalid()

    def test_source_hash_mismatch_is_invalid(self):
        self.events["adaptive"][0]["original_hashes"]["forward_context.py"] = "0" * 64
        self.invalid()

    def test_full_pointer_mismatch_is_invalid(self):
        self.captures["vanilla"][0]["bindings"]["layers"][self.name]["cache"]["data_ptr"] += 8
        self.invalid()

    def test_capture_only_launch_scalars_need_not_equal_runtime_python_scalars(self):
        self.captures["vanilla"][0]["bindings"]["layers"][self.name]["launch"]["max_seq_len"] = 4096
        self.snapshots["vanilla"]["runtime_binding"]["layers"][self.name]["launch"]["max_seq_len"] = 455
        self.assertEqual(self.compare()["status"], "STATE_EQUIVALENT")

    def test_wrong_capture_descriptor_or_missing_binding_is_invalid(self):
        self.captures["vanilla"][0]["descriptor"] = "FULL-2"
        self.invalid()

    def test_snapshot_is_not_request_state_row(self):
        self.snapshots["vanilla"]["batch_row"] = 22
        self.invalid()

    def test_snapshot_state_index_must_match_v2_forward(self):
        self.snapshots["vanilla"]["request_state_index"] = 23
        self.invalid()

    def test_write_slot_alias_is_invalid(self):
        self.snapshots["adaptive"]["layers"][self.name]["physical_blocks"][-1] = 3
        self.invalid()

    def test_stale_or_wrong_query_is_invalid(self):
        self.snapshots["adaptive"]["timestamp"] = "4"
        self.invalid()
        self.snapshots["adaptive"]["timestamp"] = "2"
        self.snapshots["adaptive"]["active_query"] = [[453, 128367]]
        self.invalid()

    def test_dispatch_must_be_recorded_and_match_hook(self):
        self.assertEqual(self.snapshots["vanilla"]["dispatched_graph_mode"], "FULL")
        self.assertEqual(self.snapshots["adaptive"]["dispatched_graph_mode"], "PIECEWISE")
        self.snapshots["vanilla"]["dispatched_graph_mode"] = "PIECEWISE"
        self.invalid()

    def test_interpretation_difference_is_state_different(self):
        layer = self.snapshots["adaptive"]["layers"][self.name]
        layer["interpretation"]["scale"] = 0.25
        self.snapshots["adaptive"]["runtime_binding"]["layers"][self.name]["interpretation"]["scale"] = 0.25
        self.assertEqual(self.compare()["status"], "STATE_DIFFERENT")


class KVSourceTests(unittest.TestCase):
    def fixture(self):
        path = Path(__file__).parent / "fixtures/protocol_b_boundary_kv_v030.json.gz"
        return json.loads(gzip.decompress(path.read_bytes()))

    def test_whole_dependency_set_authenticated_and_observer_ast_strips(self):
        for name, text in self.fixture().items():
            with self.subTest(name=name):
                patched = sources.transform(name, text.encode())
                compile(patched, name, "exec")
                self.assertEqual(ast.dump(sources.normalized_transform_ast(name, patched)),
                                 ast.dump(ast.parse(text)))
                with self.assertRaises(EvidenceError):
                    sources.transform(name, text.encode() + b"\n")

    def test_hooks_are_at_pre_execution_and_final_capture_preparation(self):
        fixture = self.fixture()
        graph = sources.transform("v1/worker/gpu/cudagraph_utils.py",
                                  fixture["v1/worker/gpu/cudagraph_utils.py"].encode()).decode()
        start = graph.index("    def run_fullgraph(")
        self.assertLess(graph.index("sync_prev_onload()", start), graph.index('"kv_v2_full"', start))
        self.assertLess(graph.index('"kv_v2_full"', start), graph.index("self.graphs[desc].replay()", start))
        self.assertIn("kv_cache_config, warmup)", graph)
        self.assertLess(graph.index('("kv_capture"'), graph.index("            def forward_fn("))
        runner = sources.transform("v1/worker/gpu/model_runner.py",
                                   fixture["v1/worker/gpu/model_runner.py"].encode()).decode()
        self.assertLess(runner.index('"kv_v2_prepare"'), runner.index("# Run model."))
        self.assertLess(runner.index('"kv_v2_piecewise"'), runner.index("model_output = self.cudagraph_manager.run_pw_graph("))

    def test_warmup_capture_does_not_touch_tensor_values_or_sources(self):
        with patch.object(observer, "kv_authenticate") as auth, patch.object(observer, "kv_layer_bindings") as bind:
            observer.kv_capture({}, None, NS(cg_mode=NS(name="FULL")), None, None, None, None, True)
            auth.assert_not_called()
            bind.assert_not_called()

    def test_throwaway_profiling_graphs_are_not_capture_evidence(self):
        with patch.object(observer, "kv_authenticate") as auth, patch.object(observer, "kv_layer_bindings") as bind:
            observer.kv_capture({}, NS(_capture_mem_samples=[]), NS(cg_mode=NS(name="FULL")),
                                None, None, None, None, False)
            auth.assert_not_called()
            bind.assert_not_called()

    def test_preserved_bytes_and_patched_hashes_are_authenticated(self):
        # Exercise the actual installer/authenticator on a temporary package only.
        import importlib.util
        originals = self.fixture()
        for fixture in ("protocol_b_boundary_v030.json.gz", "protocol_b_boundary_v2_v030.json.gz"):
            originals.update(json.loads(gzip.decompress(
                (Path(__file__).parent / "fixtures" / fixture).read_bytes())))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, run = root / "vllm", root / "run"
            run.mkdir()
            for name, text in originals.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text.encode())
            with patch.object(importlib.util, "find_spec", return_value=NS(origin=str(package / "__init__.py"))):
                sources.install(run)
                auth = sources.authenticate(run, installed=True)
                self.assertEqual(set(auth["original_hashes"]), set(sources.HASHES))
                state = json.loads((run / "instrumentation.json").read_text())
                self.assertEqual(set(state["patched_hashes"]), set(sources.PATCHED_FILES))
                untouched = package / "forward_context.py"
                untouched.write_bytes(untouched.read_bytes() + b"\n")
                with self.assertRaises(EvidenceError):
                    sources.authenticate(run, installed=True)
                backup = run / "vllm-originals/forward_context.py"
                backup.write_bytes(backup.read_bytes() + b"\n")
                with self.assertRaises(EvidenceError):
                    sources.authenticate(run)

    def test_observer_failure_records_error_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "observer").mkdir()
            config = dict(directory=tmp, lane="vanilla", execution_sha="a" * 40,
                          prompt_token_ids=[8] * 176, kv_probe=True)
            path = root / "boundary_config.json"
            path.write_text(json.dumps(config))
            value = torch.arange(3)
            with patch.dict(observer.os.environ, PROTOCOL_B_BOUNDARY_CONFIG=str(path)), \
                 patch.object(observer, "kv_v2_prepare", side_effect=EvidenceError("injected")):
                observer.observe("kv_v2_prepare", value)
            self.assertTrue(torch.equal(value, torch.arange(3)))
            event, = [json.loads(p.read_text()) for p in (root / "observer").glob("*.json")]
            self.assertEqual(event["kind"], "observer_error")
            self.assertIsNone(observer._local.kv_pending)


def runtime_fixture():
    """CPU objects for Python binding/ownership checks, never model execution."""
    name = "model.layers.0.self_attn.attn"
    cache, table, _, _ = cache_fixture()
    slot = torch.tensor([454])
    metadata = NS(query_start_loc=torch.tensor([0, 1]), seq_lens=torch.tensor([455]),
                  block_table=table[:1], slot_mapping=slot, causal=True, use_cascade=False,
                  mm_prefix_range_tensor=None, rswa_prefix_lens=None, max_seq_len=455,
                  max_query_len=1, num_actual_tokens=1, seq_threshold_3D=64,
                  softmax_segm_output=torch.empty(1), softmax_segm_max=torch.empty(1),
                  softmax_segm_expsum=torch.empty(1))
    impl = NS(_kv_quant_mode=NS(name="NONE"), _is_per_token_head_quant=False,
              kv_cache_dtype="auto", sliding_window=(-1, -1), alibi_slopes=None,
              sinks=None, chunk_lookback=-1, use_alibi_sqrt=False, num_kv_heads=2,
              head_size=3, scale=0.125, logits_soft_cap=0.0)
    layer = NS(attn_backend=NS(get_name=lambda: "TRITON_ATTN"), impl=impl, kv_cache=cache,
               _q_scale=torch.tensor(1.0), _k_scale=torch.tensor(1.0), _v_scale=torch.tensor(1.0))
    spec = type("FullAttentionSpec", (NS,), {})(num_kv_heads=2, head_size=3, head_size_v=3,
                                               block_size=16, dtype=torch.float16)
    cache_config = NS(kv_cache_groups=[NS(layer_names=[name], kv_cache_spec=spec)])
    config = NS(parallel_config=NS(), model_config=NS(hf_text_config=NS(num_hidden_layers=1)),
                compilation_config=NS(static_forward_context={name: layer}))
    inputs = dict(input_ids=torch.tensor([128367]), positions=torch.tensor([454]))
    return config, cache_config, inputs, {name: metadata}, {name: slot}


class BindingContractTests(unittest.TestCase):
    def test_capture_binding_is_metadata_only_and_detects_replacement(self):
        args = runtime_fixture()
        with patch.object(observer, "ints", side_effect=AssertionError("capture copied values")):
            captured = observer.kv_layer_bindings(*args)
        runtime = observer.kv_layer_bindings(*args)
        observer.validate_full_binding(captured, runtime)
        args[2]["positions"] = args[2]["positions"].clone()
        with self.assertRaises(EvidenceError):
            observer.validate_full_binding(captured, observer.kv_layer_bindings(*args))

    def test_backend_quantization_parallelism_and_layer_omissions_fail_closed(self):
        mutations = [
            lambda a: setattr(next(iter(a[0].compilation_config.static_forward_context.values())).attn_backend,
                              "get_name", lambda: "OTHER"),
            lambda a: setattr(next(iter(a[0].compilation_config.static_forward_context.values())).impl,
                              "_kv_quant_mode", NS(name="FP8_PER_TENSOR")),
            lambda a: setattr(a[0].parallel_config, "tensor_parallel_size", 2),
            lambda a: a[3].clear(),
            lambda a: a[4].clear(),
            lambda a: setattr(a[0].model_config.hf_text_config, "num_hidden_layers", 2),
        ]
        for mutate in mutations:
            args = runtime_fixture()
            mutate(args)
            with self.assertRaises(EvidenceError):
                observer.kv_layer_bindings(*args)

    def test_v2_ownership_dispatch_and_pending_snapshot_are_step_scoped(self):
        config, caches, inputs, metadata, slots = runtime_fixture()
        rid = "request"
        prompt = [8] * 176
        states = NS(req_id_to_index={rid: 22}, prompt_len=NS(np=[0] * 22 + [176]),
                    all_token_ids=NS(gpu=torch.full((23, 176), 8, dtype=torch.long)))
        runner = NS(req_states=states, vllm_config=config, kv_cache_config=caches,
                    model_state=type("DefaultModelState", (NS,), {})(), pcp_manager=None,
                    cudagraph_manager=object())
        batch = NS(req_ids=[rid], num_reqs=1, num_tokens=1, positions=inputs["positions"],
                   input_ids=inputs["input_ids"], num_draft_tokens=0, is_prefilling_np=[False],
                   idx_mapping_np=[22], idx_mapping=torch.tensor([22]))
        observer_config = dict(lane="vanilla", prompt_token_ids=prompt)
        for mode in ("FULL", "PIECEWISE"):
            desc = NS(cg_mode=NS(name=mode), num_ubatches=1)
            observer.kv_v2_prepare(observer_config, runner, batch, desc, inputs, metadata, slots)
            binding = observer.kv_layer_bindings(config, caches, inputs, metadata, slots)
            observer._local.kv_captures = {(id(runner.cudagraph_manager), repr(desc)):
                                           dict(bindings=binding)}
            with patch.object(observer, "kv_snapshot") as snapshot:
                observer.kv_v2_observe(observer_config, runner.cudagraph_manager,
                                       desc if mode == "FULL" else None)
                self.assertEqual(snapshot.call_args.args[6:8], (22, mode))
                self.assertIsNone(observer._local.kv_pending)
                observer.kv_v2_observe(observer_config, runner.cudagraph_manager)
                self.assertEqual(snapshot.call_count, 1)
        batch.positions = torch.tensor([455])
        observer._local.kv_pending = ("stale",)
        observer.kv_v2_prepare(observer_config, runner, batch, desc, inputs, metadata, slots)
        self.assertIsNone(observer._local.kv_pending)


def adaptive_runtime_fixture(separate_draft_group=False):
    args = runtime_fixture()
    config, caches, _, metadata, slots = args
    target = next(iter(metadata))
    # Arbitrary name and unsupported draft object: ownership, not spelling,
    # determines exclusion. Draft metadata/layout must never enter target hashing.
    draft = "unrelated_attention_identity"
    context = config.compilation_config.static_forward_context
    draft_layer = NS(kv_cache=torch.tensor([999.0]))
    context[draft] = draft_layer
    if separate_draft_group:
        caches.kv_cache_groups.append(NS(layer_names=[draft], kv_cache_spec=object()))
    else:
        caches.kv_cache_groups[0].layer_names.append(draft)
    metadata[draft], slots[draft] = object(), object()
    proposer = type("DraftModelProposer", (NS,), {})(
        _draft_attn_layer_names={draft}, draft_attn_groups=[NS(layer_names=[draft])])
    runner = NS(drafter=proposer, speculative_config=NS(uses_draft_model=lambda: True),
                get_model=lambda: NS(modules=lambda: iter([context[target]])),
                get_draft_model=lambda: NS(modules=lambda: iter([draft_layer])))
    return args, runner, target, draft


class AdaptiveTargetSelectionTests(unittest.TestCase):
    def test_target_plus_draft_metadata_and_slots_select_only_target(self):
        for separate in (False, True):
            args, runner, target, draft = adaptive_runtime_fixture(separate)
            binding = observer.kv_layer_bindings(*args, v1_runner=runner)
            self.assertEqual(binding["expected_layers"], [target])
            self.assertEqual(len(binding["layers"]), binding["model_layer_count"])
            self.assertNotIn(draft, binding["layers"])

    def test_target_hidden_layer_count_is_still_required(self):
        args, runner, _, _ = adaptive_runtime_fixture()
        args[0].model_config.hf_text_config.num_hidden_layers = 2
        with self.assertRaises(EvidenceError):
            observer.kv_layer_bindings(*args, v1_runner=runner)

    def test_missing_target_metadata_or_slots_is_invalid(self):
        for index in (3, 4):
            args, runner, target, _ = adaptive_runtime_fixture()
            del args[index][target]
            with self.assertRaises(EvidenceError):
                observer.kv_layer_bindings(*args, v1_runner=runner)

    def test_missing_or_ambiguous_ownership_is_invalid(self):
        mutations = [
            lambda a, r, t, d: delattr(r.drafter, "_draft_attn_layer_names"),
            lambda a, r, t, d: setattr(r.drafter, "_draft_attn_layer_names", set()),
            lambda a, r, t, d: setattr(r.drafter, "_draft_attn_layer_names", {d, "unknown"}),
            lambda a, r, t, d: setattr(r.drafter, "_draft_attn_layer_names", {t}),
            lambda a, r, t, d: setattr(r.drafter, "draft_attn_groups", [NS(layer_names=[t])]),
            lambda a, r, t, d: setattr(r, "get_draft_model", r.get_model),
            lambda a, r, t, d: a[1].kv_cache_groups[0].layer_names.append(t),
        ]
        for mutate in mutations:
            args, runner, target, draft = adaptive_runtime_fixture()
            mutate(args, runner, target, draft)
            with self.assertRaises(EvidenceError):
                observer.kv_layer_bindings(*args, v1_runner=runner)

    def test_unsupported_target_layout_remains_invalid(self):
        args, runner, target, _ = adaptive_runtime_fixture()
        args[0].compilation_config.static_forward_context[target].kv_cache = torch.zeros(1)
        with self.assertRaises(EvidenceError):
            observer.kv_layer_bindings(*args, v1_runner=runner)

    def test_vanilla_remains_strict_about_draft_extras(self):
        args, _, _, _ = adaptive_runtime_fixture()
        with self.assertRaises(EvidenceError):
            observer.kv_layer_bindings(*args)
        baseline = runtime_fixture()
        self.assertEqual(observer.kv_layer_bindings(*baseline)["expected_layers"], list(baseline[3]))

    def test_adaptive_entry_point_passes_target_only_binding_to_snapshot(self):
        args, runner, target, draft = adaptive_runtime_fixture()
        config, caches, inputs, metadata, slots = args
        runner.vllm_config, runner.kv_cache_config = config, caches
        runner.input_batch = NS(req_ids=["r"], num_reqs=1)
        runner.requests = {"r": NS(prompt_token_ids=[8] * 176)}
        scheduled = NS(total_num_scheduled_tokens=1, scheduled_spec_decode_tokens={})
        with patch.object(observer, "kv_snapshot") as snapshot:
            observer.kv_v1(dict(lane="adaptive", prompt_token_ids=[8] * 176), runner, scheduled,
                           inputs["input_ids"], inputs["positions"], metadata, slots, NS(name="PIECEWISE"))
        binding = snapshot.call_args.args[8]
        self.assertEqual(set(binding["layers"]), {target})
        self.assertNotIn(draft, binding["expected_layers"])

    def test_hash_iteration_excludes_draft_cache(self):
        args, runner, target, draft = adaptive_runtime_fixture()
        config, caches, inputs, metadata, slots = args
        binding = observer.kv_layer_bindings(*args, v1_runner=runner)
        # Exercise the same selected-layer iteration used by kv_snapshot,
        # using real CPU tensors for the extraction contract only.
        target_cache = config.compilation_config.static_forward_context[target].kv_cache
        _, table, _, current = cache_fixture()
        def hashes():
            return {name: observer.hash_history(
                config.compilation_config.static_forward_context[name].kv_cache,
                table, 16, 2, 3, current)["digests"] for name in binding["expected_layers"]}
        before = hashes()
        config.compilation_config.static_forward_context[draft].kv_cache.fill_(-123)
        self.assertEqual(before, hashes())
        target_cache[0, 0, 0, 0] = 1
        self.assertNotEqual(before, hashes())


class KVAnalyzerIntegrationTests(unittest.TestCase):
    def artifact(self, other_boundary=False):
        import test_protocol_b_boundary as legacy
        from experiments.protocol_b import fingerprint
        case = legacy.BoundaryEvidenceTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.artifact(kv=True, other_boundary=other_boundary)
        root = case.output
        originals = KVSourceTests().fixture()
        for name in ("protocol_b_boundary_v030.json.gz", "protocol_b_boundary_v2_v030.json.gz"):
            originals.update(json.loads(gzip.decompress(
                (Path(__file__).parent / "fixtures" / name).read_bytes())))
        package = root / "installed-vllm"
        for name, text in originals.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())
        with patch.object(sources.importlib.util, "find_spec", return_value=NS(origin=str(package / "__init__.py"))):
            sources.install(root)
        auth = sources.authenticate(root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["diagnostic_source_hashes"] = {
            "protocol_b_boundary_observer.py": auth["observer_sha256"]}
        manifest_path.write_text(json.dumps(manifest))
        snapshots, captures, events, _, _ = synthetic_evidence()
        for lane in ("vanilla", "adaptive"):
            output = root / lane
            config = json.loads((output / "boundary_config.json").read_text())
            snapshots[lane]["request_id"] = "chatcmpl-45-1234abcd"
            snapshots[lane]["request_state_index"] = 0  # Legacy artifact fixture uses the V1 row contract.
            events[lane][0].update(auth)
            for i, event in enumerate([snapshots[lane], *captures[lane], *events[lane]]):
                kind = "kv_history" if i == 0 else "kv_capture" if event in captures[lane] else "kv_sources"
                event.update(kind=kind, schema="m203.boundary.event.v1", lane=lane,
                             execution_sha="a" * 40, config_fingerprint=fingerprint(config))
                (output / "observer" / f"kv-{i}.json").write_text(json.dumps(event))
        return root

    def test_real_source_bundle_and_complete_artifact_reach_equality(self):
        from experiments.protocol_b_boundary import analyze
        root = self.artifact()
        result = analyze(root)
        self.assertEqual(result["status"], "INTERPRETABLE", result)
        self.assertEqual(result["historical_kv"]["status"], "STATE_EQUIVALENT", result)
        self.assertTrue(result["historical_kv_state_equivalent"])
        self.assertFalse(result["same_kv_cache_state_proven"])

    def test_changed_boundary_is_invalid_kv_evidence(self):
        from experiments.protocol_b_boundary import analyze
        result = analyze(self.artifact(other_boundary=True))
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertEqual(result["boundary_status"], "NON_REPRODUCTION")
        self.assertFalse(result["historical_kv"]["historical_kv_state_equivalent"])

    def test_missing_snapshot_and_preserved_source_drift_fail_closed(self):
        from experiments.protocol_b_boundary import analyze
        for bad in ("missing_snapshot", "source_drift"):
            root = self.artifact()
            if bad == "missing_snapshot":
                (root / "adaptive/observer/kv-0.json").unlink()
            else:
                path = root / "vllm-originals/forward_context.py"
                path.write_bytes(path.read_bytes() + b"\n")
            result = analyze(root)
            self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
            self.assertFalse(result["historical_kv"]["historical_kv_state_equivalent"])


if __name__ == "__main__":
    unittest.main()
