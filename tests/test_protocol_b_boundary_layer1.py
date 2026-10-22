"""CPU byte/index and serialized-evidence contracts; no CUDA replay emulation."""
import copy
import ast
import hashlib
import inspect
import json
import unittest
from unittest.mock import patch

import torch

import test_protocol_b_boundary_kv as kv_tests
from experiments.protocol_b import EvidenceError, fingerprint
from experiments import protocol_b_boundary_observer as observer
from experiments.protocol_b_boundary import (first_position_differences, compare_layer1_localization,
                                             compare_kv_histories, analyze)


def extract(changes=(), shift=0, strided=False):
    cache, table, expected, current = kv_tests.cache_fixture(shift, strided)
    for component, p, value in changes:
        cache[int(table[0, p // 16]), 0, p % 16, 0 if component == "K" else 3] = value
    result = observer.hash_history(cache, table, 16, 2, 3, current, localize_layer1=True)
    return result, expected


def evidence(changes=(("K", 17, -123), ("V", 33, -124))):
    """Two-layer synthetic whole-model records, using real CPU-derived content digests."""
    args = kv_tests.synthetic_evidence()
    snapshots, captures = args[:2]
    for lane in ("vanilla", "adaptive"):
        s = snapshots[lane]
        layer0 = s["expected_layers"][0]
        content, _ = extract(changes if lane == "adaptive" else (), shift=3 if lane == "adaptive" else 0)
        content["cache_identity"]["device"] = "cuda:0"  # Serialization fixture, not a GPU result.
        s["layers"][observer.LAYER1] = dict(content, group=0, spec_block_size=16,
                                           interpretation=copy.deepcopy(s["layers"][layer0]["interpretation"]))
        b = s["runtime_binding"]
        b["layers"][observer.LAYER1] = copy.deepcopy(b["layers"][layer0])
        b["layers"][observer.LAYER1]["cache"] = content["cache_identity"]
        s["expected_layers"] = b["expected_layers"] = sorted(s["layers"])
        s["model_layer_count"] = b["model_layer_count"] = 2
        if captures[lane]:
            captures[lane][0]["bindings"] = copy.deepcopy(b)
    return args


class Layer1ExtractionTests(unittest.TestCase):
    def test_prefix_and_position_hashes_share_exact_original_bytes(self):
        content, expected = extract()
        loc = content["localization"]
        for component, values in (("K", expected), ("V", -expected)):
            raw = values.contiguous().view(torch.uint8).numpy().tobytes()
            spans = observer.position_spans(raw, loc["bytes_per_position"])
            self.assertEqual(len(spans), 454)
            self.assertEqual(b"".join(spans), raw)
            self.assertEqual(loc[component]["position_sha256"],
                             [hashlib.sha256(span).hexdigest() for span in spans])
            self.assertEqual(loc[component]["prefix_sha256"], content["digests"][component])
            self.assertEqual(content["digests"][component], hashlib.sha256(raw).hexdigest())
        cache, table, _, current = kv_tests.cache_fixture()
        self.assertEqual(content["digests"], observer.hash_history(cache, table, 16, 2, 3, current)["digests"])

    def test_byte_accounting_and_incomplete_raw_are_rejected(self):
        content, _ = extract()
        self.assertEqual(content["localization"]["bytes_per_position"], 2 * 3 * 2)
        for raw, width in ((b"x", 12), (b"x" * (454 * 12 - 1), 12),
                           (b"x" * (454 * 12 + 1), 12), (b"", 0)):
            with self.assertRaises(EvidenceError):
                observer.position_spans(raw, width)

    def test_strides_and_physical_allocations_do_not_change_logical_digests(self):
        baseline, _ = extract()
        for shift, strided in ((3, False), (0, True), (3, True)):
            other, _ = extract(shift=shift, strided=strided)
            self.assertEqual(baseline["localization"], other["localization"])

    def test_position_454_and_unused_tails_are_excluded(self):
        before, _ = extract()
        after, _ = extract((("K", 454, 99), ("V", 454, 101), ("K", 463, 77)))
        self.assertEqual(before["localization"], after["localization"])

    def test_signed_zero_original_bits_differ(self):
        before, _ = extract()
        after, _ = extract((("K", 0, -0.0),))
        result = first_position_differences(before["localization"], after["localization"])
        self.assertEqual((result["P_K"], result["P_V"], result["P"]), (0, None, 0))

    def test_existing_cpu_transfer_count_is_unchanged(self):
        cache, table, _, current = kv_tests.cache_fixture()
        cpu = torch.Tensor.cpu
        calls = []
        def copy(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return cpu(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, "cpu", copy):
            observer.hash_history(cache, table, 16, 2, 3, current)
        original = list(calls)
        calls.clear()
        with patch.object(torch.Tensor, "cpu", copy):
            observer.hash_history(cache, table, 16, 2, 3, current, localize_layer1=True)
        self.assertEqual(original, calls)  # Metadata row plus one transfer per component.
        self.assertEqual(calls.count((454, 2, 3)), 2)

    def test_default_extraction_has_no_per_position_content(self):
        cache, table, _, current = kv_tests.cache_fixture()
        self.assertNotIn("localization", observer.hash_history(cache, table, 16, 2, 3, current))

    def test_snapshot_enables_position_content_only_for_exact_target_layer1(self):
        # Evaluate the actual caller's scope guard without executing CUDA-only snapshot code.
        tree = ast.parse(inspect.getsource(observer.kv_snapshot))
        call, = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "hash_history"]
        guard, = [kw.value for kw in call.keywords if kw.arg == "localize_layer1"]
        code = compile(ast.Expression(guard), "<layer-1 scope guard>", "eval")
        for enabled in (False, True):
            for name in (observer.LAYER1, "model.layers.0.self_attn.attn",
                         "model.layers.10.self_attn.attn", "draft.model.layers.1.self_attn.attn"):
                self.assertEqual(eval(code, {"localize": enabled, "name": name, "LAYER1": observer.LAYER1}),
                                 enabled and name == observer.LAYER1)


class FirstPositionTests(unittest.TestCase):
    def test_k_only_v_only_both_and_distinct_first_positions(self):
        base, _ = extract()
        for changes, expected in (
            ((("K", 7, -99),), (7, None, 7, True, False)),
            ((("V", 9, -99),), (None, 9, 9, False, True)),
            ((("K", 8, -99), ("V", 8, -99)), (8, 8, 8, True, True)),
            ((("K", 20, -99), ("V", 3, -99)), (20, 3, 3, False, True)),
        ):
            altered, _ = extract(changes)
            result = first_position_differences(base["localization"], altered["localization"])
            self.assertEqual(tuple(result[k] for k in
                             ("P_K", "P_V", "P", "K_differs_at_P", "V_differs_at_P")), expected)

    def test_numeric_positions_zero_block_boundary_and_last(self):
        base, _ = extract()
        for p in (0, 16, 453):
            altered, _ = extract((("K", p, -123), ("V", p, -124)))
            self.assertEqual(first_position_differences(base["localization"], altered["localization"])["P"], p)

    def test_all_equal_produces_null_positions_without_equivalence_status(self):
        base, _ = extract()
        result = first_position_differences(base["localization"], base["localization"])
        self.assertIsNone(result["P_K"])
        self.assertIsNone(result["P_V"])
        self.assertIsNone(result["P"])
        self.assertNotIn("historical_kv_state_equivalent", result)
        self.assertNotIn("status", result)


class LocalizationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.args = evidence()
        self.snapshots = self.args[0]

    def compare(self):
        whole = compare_kv_histories(*self.args)
        return compare_layer1_localization(self.snapshots, whole)

    def loc(self):
        return self.snapshots["adaptive"]["layers"][observer.LAYER1]["localization"]

    def invalid(self):
        result = self.compare()
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertIsNone(result["P"])

    def test_complete_localization_links_to_whole_model_state_difference(self):
        whole = compare_kv_histories(*self.args)
        before = copy.deepcopy(whole)
        result = compare_layer1_localization(self.snapshots, whole)
        self.assertEqual(result["status"], "LOCALIZED", result)
        self.assertEqual((result["P_K"], result["P_V"], result["P"]), (17, 33, 17))
        self.assertTrue(result["K_differs_at_P"])
        self.assertFalse(result["V_differs_at_P"])
        self.assertEqual(whole, before)

    def test_missing_or_extra_localization_scope_is_invalid(self):
        layer = self.snapshots["adaptive"]["layers"][observer.LAYER1]
        loc = layer.pop("localization")
        self.invalid()
        layer["localization"] = loc
        self.snapshots["adaptive"]["layers"]["model.layers.0.self_attn.attn"]["localization"] = loc
        self.invalid()

    def test_wrong_layer_missing_component_malformed_or_wrong_count_is_invalid(self):
        original = copy.deepcopy(self.loc())
        mutations = [
            lambda l: l.update(target_layer="model.layers.0.self_attn.attn"),
            lambda l: l.pop("V"),
            lambda l: l["K"]["position_sha256"].pop(),
            lambda l: l["V"]["position_sha256"].append("a" * 64),
            lambda l: l["K"]["position_sha256"].__setitem__(4, "not-sha256"),
            lambda l: l["K"].update(prefix_sha256="a" * 64),
            lambda l: l.update(bytes_per_position=24),
            lambda l: l.update(heads=4),
            lambda l: l.update(dtype="torch.float32"),
        ]
        for mutate in mutations:
            self.snapshots["adaptive"]["layers"][observer.LAYER1]["localization"] = copy.deepcopy(original)
            mutate(self.loc())
            self.invalid()

    def test_reordered_missing_duplicate_or_wrong_coverage_is_invalid(self):
        original = copy.deepcopy(self.loc())
        for positions in (list(reversed(range(454))), list(range(453)), [0] * 454):
            self.loc()["logical_positions"] = positions
            self.invalid()
        self.snapshots["adaptive"]["layers"][observer.LAYER1]["localization"] = original
        self.loc()["coverage"] = [0, 455]
        self.invalid()

    def test_different_prefix_with_all_matching_position_digests_is_invalid(self):
        for component in ("K", "V"):
            self.loc()[component]["position_sha256"] = copy.deepcopy(
                self.snapshots["vanilla"]["layers"][observer.LAYER1]["localization"][component]["position_sha256"])
        self.invalid()

    def test_scoped_equality_never_grants_whole_model_equivalence(self):
        self.args = evidence(changes=())
        self.snapshots = self.args[0]
        whole = compare_kv_histories(*self.args)
        self.assertEqual(whole["status"], "STATE_EQUIVALENT")  # Existing complete-model gate only.
        self.assertEqual(compare_layer1_localization(self.snapshots, whole)["status"], "INVALID_EVIDENCE")

    def test_both_layer1_whole_prefix_differences_must_reproduce(self):
        self.args = evidence(changes=(("K", 17, -123),))
        self.snapshots = self.args[0]
        self.invalid()

    def test_source_full_binding_coverage_and_write_slot_gates_remain_fail_closed(self):
        mutations = [
            lambda a: a[2]["adaptive"][0]["original_hashes"].update({"forward_context.py": "0" * 64}),
            lambda a: a[1]["vanilla"][0]["bindings"]["inputs"]["positions"].update(data_ptr=-1),
            lambda a: a[0]["adaptive"]["layers"][observer.LAYER1].update(write_slot=-1),
            lambda a: a[0]["adaptive"]["layers"].pop(observer.LAYER1),
        ]
        for mutate in mutations:
            self.args = evidence()
            self.snapshots = self.args[0]
            mutate(self.args)
            self.invalid()

    def test_snapshot_linkage_cannot_be_reused_from_another_comparison(self):
        whole = compare_kv_histories(*self.args)
        self.snapshots["adaptive"]["snapshot_id"] = "another-snapshot"
        self.assertEqual(compare_layer1_localization(self.snapshots, whole)["status"], "INVALID_EVIDENCE")


class LocalizationAnalyzerIntegrationTests(unittest.TestCase):
    def artifact(self, *, other_boundary=False):
        fixture = kv_tests.KVAnalyzerIntegrationTests()
        self.addCleanup(fixture.doCleanups)
        root = fixture.artifact(other_boundary)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["layer1_localization"] = True
        manifest_path.write_text(json.dumps(manifest))
        snapshots, captures, _, _, _ = evidence()
        for lane in ("vanilla", "adaptive"):
            output = root / lane
            config_path = output / "boundary_config.json"
            config = json.loads(config_path.read_text())
            config["layer1_localization"] = True
            config_path.write_text(json.dumps(config))
            from experiments.protocol_b import file_sha
            binding = json.loads((output / "observer_binding.json").read_text())
            binding.update(config_sha256=file_sha(config_path), config_fingerprint=fingerprint(config))
            (output / "observer_binding.json").write_text(json.dumps(binding))
            for path in (output / "observer").glob("*.json"):
                event = json.loads(path.read_text())
                if event["kind"] == "kv_history":
                    snapshots[lane].update(request_id=event["request_id"], request_state_index=0)
                    event.update(snapshots[lane])
                elif event["kind"] == "kv_capture":
                    event.update(captures[lane][0])
                event["config_fingerprint"] = fingerprint(config)
                path.write_text(json.dumps(event))
        return root

    def test_valid_boundary_and_whole_model_gate_produce_localized_output(self):
        result = analyze(self.artifact())
        self.assertEqual(result["status"], "INTERPRETABLE", result)
        self.assertEqual(result["historical_kv"]["status"], "STATE_DIFFERENT")
        loc = result["historical_kv"]["layer1_localization"]
        self.assertEqual((loc["status"], loc["P_K"], loc["P_V"], loc["P"]), ("LOCALIZED", 17, 33, 17))
        self.assertFalse(result["historical_kv_state_equivalent"])

    def test_changed_boundary_never_accepts_localization(self):
        result = analyze(self.artifact(other_boundary=True))
        self.assertEqual(result["status"], "INVALID_EVIDENCE")
        self.assertNotIn("layer1_localization", result["historical_kv"])

    def test_duplicate_snapshot_or_missing_localization_is_invalid(self):
        for bad in ("duplicate", "missing"):
            root = self.artifact()
            path = root / "adaptive/observer/kv-0.json"
            event = json.loads(path.read_text())
            if bad == "duplicate":
                path.with_name("duplicate.json").write_text(json.dumps(event))
            else:
                del event["layers"][observer.LAYER1]["localization"]
                path.write_text(json.dumps(event))
            result = analyze(root)
            self.assertEqual(result["status"], "INVALID_EVIDENCE", result)
            self.assertFalse(result.get("historical_kv_state_equivalent", False))


if __name__ == "__main__":
    unittest.main()
