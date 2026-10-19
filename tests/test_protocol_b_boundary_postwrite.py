"""CPU bytes, metadata and serialized evidence contracts; no GPU replay emulation."""
import ast
import copy
import hashlib
import inspect
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

import test_protocol_b_boundary as base_tests
import test_protocol_b_boundary_early as early_tests
import test_protocol_b_boundary_kv as kv_tests
import test_protocol_b_boundary_layer1 as layer1_tests
from experiments.protocol_b import EvidenceError, fingerprint, file_sha
from experiments import protocol_b_boundary as boundary
from experiments import protocol_b_boundary_observer as observer
from experiments import protocol_b_boundary_sources as sources


LANES = ("vanilla", "adaptive")
LAYER1 = observer.LAYER1


def stable_cache_addresses(snapshot):
    """Explicit disjoint synthetic addresses, independent of freed CPU allocations."""
    for i, name in enumerate(snapshot["expected_layers"], 1):
        identity = snapshot["layers"][name]["cache_identity"]
        identity.update(data_ptr=i * 1000000, storage_ptr=i * 1000000, storage_offset=0)
        snapshot["runtime_binding"]["layers"][name]["cache"] = copy.deepcopy(identity)


def post_event(lane, later, *, changes=(), query=None, schedule=3, scheduler_id=123):
    """Real CPU-derived row hashes with explicitly synthetic runtime metadata."""
    cache, table, _, _ = kv_tests.cache_fixture(3 if lane == "adaptive" else 0)
    block = int(table[0, 177 // 16])
    for component, value in changes:
        cache[block, 0, 177 % 16, 0 if component == "K" else 3] = value
    positions = [177] if lane == "vanilla" else [177, 178, 179, 180]
    binding = copy.deepcopy(later["runtime_binding"])
    n = len(positions)
    binding["inputs"]["positions"] = dict(binding["inputs"]["positions"], shape=[n])
    binding["inputs"]["input_ids"] = dict(binding["inputs"]["input_ids"], shape=[n])
    for layer in binding["layers"].values():
        layer["metadata"]["slot_mapping"] = dict(layer["metadata"]["slot_mapping"], shape=[n])
    slots = [int(table[0, p // 16]) * 16 + p % 16 for p in positions]
    query = query or [[p, 7] for p in positions]
    return dict(kind="postwrite_position177", postwrite_schema=observer.POSTWRITE_SCHEMA, timestamp="1.5",
        lane=lane, phase="post_target_forward", logical_position=177, target_layer=LAYER1,
        request_id=later["request_id"], pid=later["pid"], auth_id=later["auth_id"],
        request_state_index=later["request_state_index"], batch_row=0,
        actual_positions=positions, active_query=query, input_token_id=query[0][1],
        graph_mode="FULL" if lane == "vanilla" else "PIECEWISE",
        hook="postwrite_v2_full" if lane == "vanilla" else "postwrite_v1",
        forward_id=lane + "-forward", descriptor="position177-FULL" if lane == "vanilla" else None,
        capture_id="position177-capture" if lane == "vanilla" else None,
        schedule_seq=None if lane == "vanilla" else schedule,
        scheduler_output_id=None if lane == "vanilla" else scheduler_id,
        runtime_binding=binding, cache_identity=copy.deepcopy(binding["layers"][LAYER1]["cache"]),
        cache_peers={name: copy.deepcopy(layer["cache"]) for name, layer in binding["layers"].items()},
        block_table=table[0, :177 // 16 + 1].tolist(), write_slots=slots,
        execution_stream=dict(device="cuda:0", cuda_stream=0), runtime_probe_changes_timing=True,
        blocking_device_to_cpu=True, device_wide_synchronize=False, capture_warmup=False,
        **observer.position177_slot(table[0].tolist(), slots, positions, 16, cache.shape[0]),
        **observer.hash_position177(cache, block, 177 % 16))


def full_capture(event):
    return dict(kind="kv_capture", capture_id=event["capture_id"], descriptor=event["descriptor"],
                pid=event["pid"], auth_id=event["auth_id"], timestamp="1",
                bindings=copy.deepcopy(event["runtime_binding"]))


def evidence(changes=()):
    args = layer1_tests.evidence(changes=(("K", 177, -123), ("V", 177, -124)))
    snapshots, captures, _, _, boundaries = args
    for lane in LANES:
        stable_cache_addresses(snapshots[lane])
        if captures[lane]:
            captures[lane][0]["bindings"] = copy.deepcopy(snapshots[lane]["runtime_binding"])
    historical = boundary.compare_kv_histories(*args)
    historical["layer1_localization"] = boundary.compare_layer1_localization(snapshots, historical)
    events, records = {}, {}
    early = dict(status="INTERPRETABLE", first_coverage_path="speculative", first_coverage_step=3,
                 timeline=[dict(schedule_seq=3, decoding_path="speculative",
                    active_query=[[p, 7] for p in range(177, 181)],
                    accounting_before=dict(scheduler_output_id=123))])
    for lane in LANES:
        snapshots[lane]["kind"] = "kv_history"
        e = post_event(lane, snapshots[lane], changes=changes if lane == "adaptive" else ())
        events[lane] = [snapshots[lane], e]
        if lane == "vanilla":
            events[lane].append(full_capture(e))
        records[lane] = dict(response_id=lane, token_ids=[7] * 284)
    return events, records, boundaries, historical, early


class SlotAndBytesTests(unittest.TestCase):
    def test_nontrivial_blocks_and_selected_row(self):
        table = list(reversed(range(32)))
        slots = [table[p // 16] * 16 + p % 16 for p in range(177, 181)]
        for positions, selected in (([177], [slots[0]]), (list(range(177, 181)), slots)):
            result = observer.position177_slot(table, selected + [-1], positions, 16, 32)
            self.assertEqual((result["physical_block"], result["offset"], result["slot"], result["selected_row"]),
                             (20, 1, 321, 0))
        result = observer.position177_slot(table, [slots[1], slots[0]], [178, 177], 16, 32)
        self.assertEqual(result["selected_row"], 1)

    def test_wrong_missing_duplicate_positions_slots_blocks_and_aliases(self):
        table = list(range(32))
        cases = [
            (table, [178], [177], 16, 32),
            (table, [177], [178], 16, 32),
            (table, [177, 178], [177, 177], 16, 32),
            (table, [177, 177], [177, 178], 16, 32),
            (table, [177, 177], [177], 16, 32),
            (table[:11] + [32], [513], [177], 16, 32),
            (table[:11] + [-1], [177], [177], 16, 32),
            (table[:11], [177], [177], 16, 32),
            (table, [-1], [177], 16, 32),
            (table, [177, 512], [177], 16, 32),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(EvidenceError):
                observer.position177_slot(*args)

    def test_exact_k_v_bytes_and_only_two_bounded_cpu_transfers(self):
        cache, table, expected, _ = kv_tests.cache_fixture(3)
        calls, original = [], torch.Tensor.cpu
        def cpu(tensor, *args, **kwargs):
            calls.append((tensor.numel(), tensor.dtype))
            return original(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, "cpu", cpu):
            row = observer.hash_position177(cache, int(table[0, 11]), 1)
        self.assertEqual(calls, [(6, torch.float16), (6, torch.float16)])
        for component, value in (("K", expected[177]), ("V", -expected[177])):
            raw = value.contiguous().view(torch.uint8).numpy().tobytes()
            self.assertEqual(row[component]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual((row["shape"], row["byte_count"], row["total_value_bytes"]), ([2, 3], 12, 24))
        cache[int(table[0, 11]), :, 2:, :] = -123
        self.assertEqual(row, observer.hash_position177(cache, int(table[0, 11]), 1))

    def test_signed_zero_preserved_for_each_component(self):
        cache = torch.zeros(16, 2, 16, 6, dtype=torch.float16)
        for component, col in (("K", 0), ("V", 3)):
            before = observer.hash_position177(cache, 11, 1)
            cache[11, 0, 1, col] = -0.0
            after = observer.hash_position177(cache, 11, 1)
            self.assertNotEqual(before[component], after[component])

    def test_wrong_target_missing_peer_and_draft_overlap(self):
        e = evidence()[0]["adaptive"][1]
        observer.postwrite_ownership(e["runtime_binding"], e["cache_peers"])
        for change in ("missing", "wrong", "draft", "other_target"):
            peers = copy.deepcopy(e["cache_peers"])
            if change == "missing":
                peers.pop(LAYER1)
            elif change == "wrong":
                peers[LAYER1]["data_ptr"] += 2
            elif change == "draft":
                peers["draft.synthetic.attn"] = copy.deepcopy(peers[LAYER1])
            else:
                peers[next(n for n in peers if n != LAYER1)] = copy.deepcopy(peers[LAYER1])
            with self.subTest(change=change), self.assertRaises(EvidenceError):
                observer.postwrite_ownership(e["runtime_binding"], peers)


class StorageOwnershipTests(unittest.TestCase):
    def ownership(self, selected, peer):
        binding = dict(expected_layers=[LAYER1], layers={LAYER1: dict(cache=selected)})
        observer.postwrite_ownership(binding, {LAYER1: selected, "peer.attn": peer})

    def test_strided_and_contiguous_caches_with_separate_storage_are_accepted(self):
        peer = torch.zeros(64, 2, 16, 6, dtype=torch.float16)
        base = torch.zeros(64, 2, 16, 12, dtype=torch.float16)
        token_major = torch.zeros(64, 16, 2, 6, dtype=torch.float16).transpose(1, 2)
        for cache in (base[..., ::2], token_major, base[..., :6].contiguous()):
            with self.subTest(strides=cache.stride()):
                self.ownership(observer.tensor_identity(cache), observer.tensor_identity(peer))

    def test_identity_records_entire_backing_storage_for_offset_strided_view(self):
        base = torch.zeros(64, 2, 16, 12, dtype=torch.float16)
        view = base[1:, ..., ::2]
        identity = observer.tensor_identity(view)
        self.assertEqual(identity["storage_nbytes"], base.untyped_storage().nbytes())
        self.assertEqual(identity["storage_ptr"], base.untyped_storage().data_ptr())
        self.assertNotEqual(identity["data_ptr"], identity["storage_ptr"])
        self.assertGreater(identity["storage_nbytes"], view.numel() * view.element_size())

    def test_shared_storage_disjoint_contiguous_and_strided_view_intervals_are_accepted(self):
        base = torch.zeros(64, 2, 16, 12, dtype=torch.float16)
        for left, right in ((base[:32], base[32:]), (base[:32, ..., ::2], base[32:, ..., ::2])):
            selected, peer = observer.tensor_identity(left), observer.tensor_identity(right)
            self.assertEqual(selected["storage_ptr"], peer["storage_ptr"])
            with self.subTest(strides=left.stride()):
                self.ownership(selected, peer)

    def test_shared_storage_overlapping_views_rejected_despite_different_data_pointers(self):
        base = torch.zeros(64, 2, 16, 6, dtype=torch.float16)
        selected, peer = observer.tensor_identity(base[:32]), observer.tensor_identity(base[1:33])
        self.assertNotEqual(selected["data_ptr"], peer["data_ptr"])
        with self.assertRaises(EvidenceError):
            self.ownership(selected, peer)

    def test_interleaved_disjoint_elements_with_overlapping_bounds_fail_closed(self):
        base = torch.zeros(64, 2, 2, 16, 6, dtype=torch.float16)
        # Synthetic block-outermost layer views: distinct elements, overlapping bounds.
        selected, peer = observer.tensor_identity(base[:, 0]), observer.tensor_identity(base[:, 1])
        self.assertNotEqual(selected["data_ptr"], peer["data_ptr"])
        with self.assertRaises(EvidenceError):
            self.ownership(selected, peer)

    def test_partial_backing_overlap_rejected_and_adjacent_ranges_accepted(self):
        selected = observer.tensor_identity(torch.zeros(4, 2, 16, 6, dtype=torch.float16))
        selected.update(storage_ptr=100000, data_ptr=100000)
        peer = copy.deepcopy(selected)
        peer.update(storage_ptr=100000 + selected["storage_nbytes"] // 2, data_ptr=900000)
        with self.assertRaises(EvidenceError):
            self.ownership(selected, peer)
        peer["storage_ptr"] = 100000 + selected["storage_nbytes"]
        self.ownership(selected, peer)

    def test_invalid_storage_bounds_and_different_device_are_rejected(self):
        selected = observer.tensor_identity(torch.zeros(4, 2, 16, 6, dtype=torch.float16))
        peer = dict(selected, storage_ptr=selected["storage_ptr"] + 100000,
                    data_ptr=selected["data_ptr"] + 100000)
        self.ownership(selected, peer)
        for field, values in (("storage_nbytes", (0, -1, None, True, 1.5)),
                              ("storage_ptr", (0, -1, None, True, 1.5)),
                              ("storage_offset", (-1, None, True, 1.5)),
                              ("strides", ([-1, 96, 6, 1], [True, 96, 6, 1], [192, 96, 6])),
                              ("device", ("cuda:1",)),
                              ("dtype", ("torch.float32",)),
                              ("shape", ([0, 2, 16, 6], [4, 2, 16]))):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(EvidenceError):
                    self.ownership(selected, dict(peer, **{field: value}))

    def test_reachable_interval_must_fit_inside_backing_storage(self):
        cache = torch.zeros(4, 2, 16, 6, dtype=torch.float16)
        selected = observer.tensor_identity(cache)
        peer = dict(selected, storage_ptr=selected["storage_ptr"] + 100000,
                    data_ptr=selected["data_ptr"] + 100000)
        for change in (dict(storage_nbytes=selected["storage_nbytes"] - 1),
                       dict(storage_offset=1), dict(storage_offset=selected["storage_nbytes"] // 2),
                       dict(strides=[100000, 96, 6, 1])):
            with self.subTest(change=change), self.assertRaises(EvidenceError):
                self.ownership(selected, dict(peer, **change))

    def test_strided_position177_hashes_match_contiguous_exact_bytes(self):
        dense, table, expected, _ = kv_tests.cache_fixture(3)
        strided, strided_table, _, _ = kv_tests.cache_fixture(3, noncontiguous=True)
        self.assertFalse(strided.is_contiguous())
        self.assertEqual(table.tolist(), strided_table.tolist())
        block, offset = int(table[0, 177 // 16]), 177 % 16
        result = observer.hash_position177(strided, block, offset)
        self.assertEqual(result, observer.hash_position177(dense, block, offset))
        for component, values in (("K", expected[177]), ("V", -expected[177])):
            raw = values.contiguous().view(torch.uint8).numpy().tobytes()
            self.assertEqual(result[component]["sha256"], hashlib.sha256(raw).hexdigest())


class PostwriteEvidenceTests(unittest.TestCase):
    def report(self, data):
        return boundary.analyze_postwrite(*data)

    def invalid(self, data):
        result = self.report(data)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertEqual(result["classification"], "INVALID_EVIDENCE")
        self.assertFalse(result["task_done"])
        return result

    def test_k_only_v_only_and_both_immediate_differences(self):
        for changes, flags in (((("K", -123),), (False, True)),
                               ((("V", -124),), (True, False)),
                               ((("K", -123), ("V", -124)), (False, False))):
            result = self.report(evidence(changes))
            self.assertEqual(result["status"], "INTERPRETABLE", result)
            self.assertEqual(tuple(result["immediate"].values()), flags)
            self.assertEqual(result["classification"], "IMMEDIATE_STORED_KV_DIVERGENCE")

    def test_immediate_equality_links_later_change_to_lane_and_component(self):
        result = self.report(evidence())
        self.assertEqual(result["status"], "INTERPRETABLE", result)
        self.assertEqual(result["classification"], "LATER_STATE_CHANGE")
        self.assertEqual(result["later_linkage"], dict(
            vanilla=dict(K_same_as_later=True, V_same_as_later=True),
            adaptive=dict(K_same_as_later=False, V_same_as_later=False)))
        for field in ("task_done", "same_kv_cache_state_proven", "historical_kv_state_equivalent"):
            self.assertIs(result[field], False)
        self.assertEqual(result["temporal_kv_onset"], "UNKNOWN")
        self.assertEqual(result["interpretation_status"], "NOT_YET_APPROVED")

    def test_missing_and_duplicate_capture_for_each_lane(self):
        for lane in LANES:
            for duplicate in (False, True):
                data = evidence()
                if duplicate:
                    data[0][lane].append(copy.deepcopy(data[0][lane][1]))
                else:
                    data[0][lane].pop(1)
                self.invalid(data)

    def test_representation_phase_ownership_and_first_step_mismatches(self):
        mutations = [
            lambda e: e.update(byte_count=13), lambda e: e.update(dtype="torch.float32"),
            lambda e: e.update(shape=[3, 2]), lambda e: e.update(canonical_order=["head_dimension", "kv_head"]),
            lambda e: e.update(target_layer="draft.attn"), lambda e: e.update(request_id="wrong"),
            lambda e: e.update(pid=-1), lambda e: e.update(auth_id="wrong"),
            lambda e: e.update(phase="pre_forward"), lambda e: e.update(capture_warmup=True),
            lambda e: e.update(schedule_seq=4), lambda e: e.update(scheduler_output_id=-1),
            lambda e: e.update(slot=-1), lambda e: e.update(input_token_id=-1),
            lambda e: e["K"].update(sha256="not-a-digest"),
            lambda e: e["write_slots"].append(e["slot"]),
            lambda e: e["execution_stream"].update(device="cuda:1"),
            lambda e: e.update(runtime_probe_changes_timing=False),
            lambda e: e.update(blocking_device_to_cpu=False),
            lambda e: e.update(device_wide_synchronize=True),
            lambda e: e.update(batch_row=1), lambda e: e.update(request_state_index=-1),
            lambda e: e.update(timestamp="9"), lambda e: e.update(forward_id=None),
        ]
        for i, mutate in enumerate(mutations):
            data = evidence()
            mutate(data[0]["adaptive"][1])
            with self.subTest(mutation=i):
                self.invalid(data)

    def test_later_snapshot_digest_and_cache_linkage_mismatches(self):
        for mutate in (
            lambda s: s.update(snapshot_id="other"),
            lambda s: s["layers"][LAYER1]["localization"]["K"].update(prefix_sha256="a" * 64),
            lambda s: s["layers"][LAYER1]["localization"].update(bytes_per_position=24),
            lambda s: s["layers"][LAYER1]["cache_identity"].update(data_ptr=9),
            lambda s: s["layers"][LAYER1]["localization"]["V"]["position_sha256"].__setitem__(177, "bad"),
        ):
            data = evidence()
            mutate(data[0]["adaptive"][0])
            self.invalid(data)

    def test_vanilla_full_capture_binding_and_identity(self):
        for mutate in (lambda e: e.update(descriptor="wrong"), lambda e: e.update(capture_id="wrong"),
                       lambda e: e["runtime_binding"]["inputs"]["positions"].update(data_ptr=-1)):
            data = evidence()
            mutate(data[0]["vanilla"][1])
            self.invalid(data)

    def test_historical_localization_early_and_observer_gates(self):
        for mutate in (lambda d: d[3].update(status="STATE_EQUIVALENT"),
                       lambda d: d[3]["layer1_localization"].update(P_K=178),
                       lambda d: d[4].update(status="INVALID_EVIDENCE"),
                       lambda d: d[4].update(first_coverage_path="ordinary"),
                       lambda d: d[0]["vanilla"].append(dict(kind="observer_error"))):
            data = evidence()
            mutate(data)
            self.invalid(data)


class PendingContractTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(postwrite_position177=True, lane="vanilla", sample_id=boundary.SAMPLE,
                           prompt_token_ids=[8] * 176)
        self.addCleanup(self.clear_pending)
        self.clear_pending()
        self.stream = dict(device="cuda:0", cuda_stream=0)

    def clear_pending(self):
        for name in list(vars(observer._local)):
            if name.startswith("postwrite_"):
                delattr(observer._local, name)

    def arm(self, version="v2"):
        owner = NS(_graphs_captured=True, _capture_mem_samples=None)
        execution = object()
        pending = dict(config_key=fingerprint(self.config), stream=self.stream, manager=owner,
                       runner=owner, desc=execution, scheduled=execution, request_id="rid", schedule_seq=3)
        setattr(observer._local, "postwrite_" + version + "_pending", pending)
        setattr(observer._local, "postwrite_" + version + "_phase", "armed")
        return owner, execution, pending

    def test_stale_pending_cannot_be_reset_or_reused(self):
        self.arm()
        with self.assertRaisesRegex(EvidenceError, "Stale"):
            observer.postwrite_reset(self.config, "v2")
        owner, execution, pending = self.arm()
        pending["config_key"] = "stale"
        with patch.object(observer, "postwrite_stream", return_value=self.stream), \
             self.assertRaisesRegex(EvidenceError, "stale"):
            observer.postwrite_finish(self.config, "v2", owner, execution)
        self.assertIsNone(observer._local.postwrite_v2_pending)

    def test_wrong_v2_manager_descriptor_warmup_and_stream(self):
        for mode in ("manager", "descriptor", "warmup", "stream"):
            owner, execution, pending = self.arm()
            if mode == "warmup":
                owner._capture_mem_samples = []
            if mode == "stream":
                pending["stream"] = dict(self.stream, cuda_stream=9)
            with patch.object(observer, "postwrite_stream", return_value=self.stream), \
                 self.subTest(mode=mode), self.assertRaises(EvidenceError):
                observer.postwrite_finish(self.config, "v2", object() if mode == "manager" else owner,
                                          object() if mode == "descriptor" else execution)

    def test_v1_scheduler_runner_and_early_request_identity(self):
        self.config["lane"] = "adaptive"
        for mode in ("runner", "scheduler", "request", "step"):
            owner, execution, pending = self.arm("v1")
            entry = dict(request_id="wrong" if mode == "request" else "rid",
                         schedule_seq=4 if mode == "step" else 3)
            with patch.object(observer, "postwrite_stream", return_value=self.stream), \
                 patch.object(observer, "early_state", return_value={}), \
                 patch.object(observer, "early_step", return_value=entry), \
                 self.subTest(mode=mode), self.assertRaises(EvidenceError):
                observer.postwrite_finish(self.config, "v1", object() if mode == "runner" else owner,
                                          object() if mode == "scheduler" else execution)

    def test_missing_and_double_consumption_fail_closed(self):
        observer._local.postwrite_v2_phase = "armed"
        with self.assertRaisesRegex(EvidenceError, "Missing"):
            observer.postwrite_finish(self.config, "v2", None, None)
        observer._local.postwrite_v2_phase = "consumed"
        with self.assertRaisesRegex(EvidenceError, "Duplicate"):
            observer.postwrite_finish(self.config, "v2", None, None)

    def test_pending_consumed_once_without_touching_historical_state(self):
        """CPU orchestration only; bypass device binding, never claim CUDA execution."""
        owner, execution, pending = self.arm()
        cache, table, _, _ = kv_tests.cache_fixture()
        identity = observer.tensor_identity(cache)
        binding = dict(expected_layers=[LAYER1], layers={LAYER1: dict(cache=identity)})
        owner.vllm_config = NS(compilation_config=NS(static_forward_context={LAYER1: NS(kv_cache=cache)}))
        owner.kv_cache_config = NS(kv_cache_groups=[NS(layer_names=[LAYER1])])
        pending.update(binding=binding, peers={LAYER1: identity}, inputs={},
                       metadata={LAYER1: NS(block_table=table)}, slots={LAYER1: torch.tensor([177])},
                       positions=[177], write_count=1, capture=dict(capture_id="capture"),
                       active_query=[[177, 7]], request_state_index=22, forward_id="one")
        sentinel = object()
        observer._local.kv_pending = sentinel
        self.addCleanup(delattr, observer._local, "kv_pending")
        with patch.object(observer, "postwrite_stream", return_value=self.stream), \
             patch.object(observer, "kv_layer_bindings", return_value=binding), \
             patch.object(observer, "kv_authenticate", return_value="auth"), patch.object(observer, "emit") as emit:
            observer.postwrite_finish(self.config, "v2", owner, execution)
            self.assertEqual(emit.call_count, 1)
            self.assertEqual(emit.call_args.kwargs["total_value_bytes"], 24)
            self.assertIsNone(observer._local.postwrite_v2_pending)
            self.assertIs(observer._local.kv_pending, sentinel)
            with self.assertRaisesRegex(EvidenceError, "Duplicate"):
                observer.postwrite_finish(self.config, "v2", owner, execution)

    def test_prepare_rejects_warmup_and_first_query_before_device_reads(self):
        for positions, mode, captured, samples in (([177], "PIECEWISE", True, None),
                ([177, 178], "FULL", True, None), ([177], "FULL", False, None),
                ([177], "FULL", True, [])):
            manager = NS(_graphs_captured=captured, _capture_mem_samples=samples)
            with patch.object(observer, "kv_layer_bindings", side_effect=AssertionError("device binding reached")), \
                 self.assertRaises(EvidenceError):
                observer.postwrite_prepare(self.config, None, None, None, None, positions,
                    "rid", 22, NS(name=mode), 1, manager=manager, desc=NS(num_ubatches=1))

    def test_adaptive_rewrite_does_not_replace_first_capture_and_request_collision_fails(self):
        self.config["lane"] = "adaptive"
        observer._local.postwrite_claims = {fingerprint(self.config): "rid"}
        with patch.object(observer, "kv_layer_bindings", side_effect=AssertionError("rewrite read")):
            observer.postwrite_prepare(self.config, None, None, None, None, [177], "rid", 0, None, 1)
            with self.assertRaisesRegex(EvidenceError, "collision"):
                observer.postwrite_prepare(self.config, None, None, None, None, [177], "other", 0, None, 1)

    def test_capture_cannot_qualify_and_disabled_scope_has_no_cuda_calls(self):
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True), \
             patch.object(torch.cuda, "current_stream", side_effect=AssertionError("must not query stream")), \
             self.assertRaisesRegex(EvidenceError, "capture/warmup"):
            observer.postwrite_stream()
        with patch.object(observer, "postwrite_stream", side_effect=AssertionError("disabled read")):
            observer.postwrite_finish({}, "v2", None, None)

    def test_no_explicit_cuda_sync_and_no_values_copied_in_prepare(self):
        text = inspect.getsource(observer.postwrite_prepare)
        self.assertNotIn("hash_position177", text)
        self.assertNotIn(".cpu()", text)
        for function in (observer.postwrite_prepare, observer.postwrite_finish, observer.hash_position177):
            self.assertNotIn("synchronize(", inspect.getsource(function))


class SourcePlacementTests(unittest.TestCase):
    def test_exact_post_placements_old_pre_hooks_and_pristine_ast(self):
        raw_sources = base_tests.pinned_sources()
        for name, (anchor, insertion) in sources.POSTWRITE_INSERTIONS.items():
            raw = raw_sources[name].encode()
            patched = sources.transform(name, raw)
            sources.validate_transform(name, raw, patched)
            self.assertIn(anchor + insertion, patched.decode())
            self.assertEqual(ast.dump(sources.normalized_transform_ast(name, patched)), ast.dump(ast.parse(raw)))
            tree = ast.parse(patched)
            if name.endswith("gpu_model_runner.py"):
                enclosing = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                             and any(isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                                     and isinstance(stmt.value.func, ast.Name)
                                     and stmt.value.func.id == "_pb279_observe"
                                     and stmt.value.args[0].value == "postwrite_v1" for stmt in node.body)]
                self.assertEqual(len(enclosing), 1)
            bad = patched.decode().replace(anchor + insertion, insertion + anchor)
            with self.assertRaises(EvidenceError):
                sources.validate_transform(name, raw, bad)
            for bad in (patched.decode().replace(insertion, insertion * 2),
                        patched.decode().replace(insertion, "")):
                with self.assertRaises(EvidenceError):
                    sources.validate_transform(name, raw, bad)


class AnalyzerIntegrationTests(unittest.TestCase):
    def artifact(self):
        helper = early_tests.EarlyAnalyzerIntegrationTests()
        self.addCleanup(helper.doCleanups)
        root = helper.artifact()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["postwrite_position177"] = True
        manifest_path.write_text(json.dumps(manifest))
        # Replace the ordinary-only timeline with a valid speculative-first trace.
        trace = early_tests.fixture("multi")[0]
        token_fields = {"selected_token_ids", "output_token_ids", "committed_ids", "actual_draft_token_ids",
                        "scheduled_spec_decode_tokens", "produced_ids"}
        def tokens(values):
            return [7 if t in (101, 102, 11, 103, 104, 105) else t for t in values]
        for event in trace:
            for field in token_fields & event.keys():
                event[field] = tokens(event[field])
            if "active_query" in event:
                event["active_query"] = [[p, tokens([t])[0]] for p, t in event["active_query"]]
        for lane in LANES:
            output = root / lane
            config_path = output / "boundary_config.json"
            config = json.loads(config_path.read_text())
            config["postwrite_position177"] = True
            config_path.write_text(json.dumps(config))
            binding_path = output / "observer_binding.json"
            binding = json.loads(binding_path.read_text())
            binding.update(config_sha256=file_sha(config_path), config_fingerprint=fingerprint(config))
            binding_path.write_text(json.dumps(binding))
            paths = list((output / "observer").glob("*.json"))
            existing = [json.loads(p.read_text()) for p in paths]
            later = next(e for e in existing if e["kind"] == "kv_history")
            stable_cache_addresses(later)
            # Keep the historical FULL capture bound to its original forward.
            for e in existing:
                if e["kind"] == "kv_capture":
                    e["bindings"] = copy.deepcopy(later["runtime_binding"])
            if lane == "adaptive":
                for p, e in zip(paths, existing):
                    if e["kind"].startswith("early_"):
                        p.unlink()
                existing = [e for e in existing if not e["kind"].startswith("early_")]
                for event in copy.deepcopy(trace):
                    if not event["kind"].startswith("early_"):
                        continue
                    event.update(request_id=later["request_id"], pid=later["pid"])
                    if event["kind"] == "early_schedule" and event["planner_request_id"] is not None:
                        event["planner_request_id"] = later["request_id"]
                    existing.append(event)
                verify = next(e for e in existing if e["kind"] == "early_verify" and e["schedule_seq"] == 3)
                event = post_event(lane, later, query=verify["active_query"],
                    scheduler_id=verify["scheduler_output_id"])
            else:
                event = post_event(lane, later)
                existing.append(full_capture(event))
            existing.append(event)
            # Rewrite this temporary fixture only; immutable experiment evidence is never touched.
            for p in (output / "observer").glob("*.json"):
                p.unlink()
            for i, e in enumerate(existing):
                e.update(schema="m203.boundary.event.v1", lane=lane, execution_sha=manifest["execution_sha"],
                         config_fingerprint=fingerprint(config))
                if e["kind"] == "kv_capture" and "timestamp" not in e:
                    e["timestamp"] = "1"
                (output / "observer" / f"postwrite-test-{i}.json").write_text(json.dumps(e))
        return root

    def test_full_artifact_classifies_without_granting_completion(self):
        result = boundary.analyze(self.artifact())
        self.assertEqual(result["status"], "INTERPRETABLE", result)
        self.assertEqual(result["postwrite_position177"]["classification"], "LATER_STATE_CHANGE")
        self.assertEqual(result["early_timeline"]["first_coverage_path"], "speculative")
        for field in ("task_done", "same_kv_cache_state_proven", "historical_kv_state_equivalent"):
            self.assertIs(result[field], False)

    def test_boundary_source_kv_localization_and_early_gates_cannot_be_bypassed(self):
        for mode in ("boundary", "source", "kv", "localization", "early"):
            root = self.artifact()
            if mode == "source":
                path = root / "vllm-originals/forward_context.py"
                path.write_bytes(path.read_bytes() + b"\n# drift\n")
            elif mode == "kv":
                path = root / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["kv_probe"] = False
                path.write_text(json.dumps(manifest))
            else:
                kind = {"boundary": "forward", "localization": "kv_history", "early": "early_account_after"}[mode]
                path = next(p for p in (root / "adaptive/observer").glob("*.json")
                            if json.loads(p.read_text())["kind"] == kind)
                path.unlink()
            result = boundary.analyze(root)
            with self.subTest(mode=mode):
                self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
                self.assertEqual(result["postwrite_position177"]["status"], "INVALID_EVIDENCE")
                self.assertFalse(result["task_done"])


if __name__ == "__main__":
    unittest.main()
