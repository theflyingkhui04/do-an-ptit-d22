"""CPU evidence/mapping checks; these do not simulate GPU nondeterminism."""
import ast
import copy
import gzip
import hashlib
import json
import tempfile
import unittest
import shutil
import os
import stat
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.protocol_b import EvidenceError, file_sha, write_new, fingerprint
from experiments.protocol_b_boundary import (SAMPLE, first_difference, is_fingerprint,
                                             same_request, validate_boundary, seal, analyze, SCHEMA)
from experiments.protocol_b_boundary_observer import map_rows, selected_tokens
from experiments.protocol_b_boundary_sources import HASHES, StripObserver, transform


def pinned_sources():
    sources = {}
    for name in ("protocol_b_boundary_v030.json.gz", "protocol_b_boundary_v2_v030.json.gz"):
        sources.update(json.loads(gzip.decompress((Path(__file__).parent / "fixtures" / name).read_bytes())))
    return sources


class ObserverImportPathTests(unittest.TestCase):
    def test_runbook_exports_repo_root_before_prepare_or_launch(self):
        root = Path(__file__).resolve().parents[1]
        runbook = (root / "docs/runbooks/m2-03-boundary-279.md").read_text(encoding="utf-8")
        block = runbook.split("```bash\n", 1)[1].split("```", 1)[0]
        commands = [
            'test "$PWD" = "$(git rev-parse --show-toplevel)"',
            'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"',
            "python -c 'import experiments.protocol_b_boundary_observer'",
            'python -m experiments.protocol_b_boundary prepare',
            'bash scripts/switch_lane.sh "$selector"',
        ]
        positions = [block.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))

    def test_observer_import_inherited_by_child_and_grandchild_outside_repo(self):
        root = Path(__file__).resolve().parents[1]
        probe = (
            "import experiments.protocol_b_boundary_observer as observer; "
            "from pathlib import Path; "
            f"assert Path(observer.__file__).resolve() == Path({str(root / 'experiments/protocol_b_boundary_observer.py')!r})"
        )
        child = (
            "import subprocess, sys; "
            f"exec({probe!r}); "
            f"subprocess.run([sys.executable, '-c', {probe!r}], check=True)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            for previous in (None, "", tmp):
                with self.subTest(previous=previous):
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(root) + (os.pathsep + previous if previous else "")
                    subprocess.run([sys.executable, "-c", child], cwd=tmp, env=env,
                                   check=True, capture_output=True, text=True, timeout=30)


class RowMappingTests(unittest.TestCase):
    def test_ordinary_boundary_uses_actual_position(self):
        rows = map_rows(176, [454], [0], 0, [], [0])
        self.assertEqual(rows[0]["generated_index"], 279)

    def test_speculative_boundary_may_be_bonus_or_verification(self):
        rows = map_rows(176, [452, 453, 454, 455, 456], list(range(5)), 4, list(range(4)), [4])
        self.assertEqual(rows[2]["generated_index"], 279)
        self.assertEqual(rows[2]["role"], "draft_verification")
        rows = map_rows(176, [452, 453, 454], [0, 1, 2], 2, [0, 1], [2])
        self.assertEqual(rows[-1]["generated_index"], 279)
        self.assertEqual(rows[-1]["role"], "bonus")

    def test_flat_indices_are_not_row_indices(self):
        rows = map_rows(176, list(range(450, 455)), [2, 3, 4], 2, [0, 1], [2])
        self.assertEqual((rows[2]["row"], rows[2]["flat_index"], rows[2]["generated_index"]), (2, 4, 279))

    def test_missing_duplicate_or_noncontiguous_mapping_fails(self):
        for positions, indices, n, targets, bonus in [
            ([454], [], 0, [], [0]), ([453, 454], [0, 0], 1, [0], [1]),
            ([452, 454], [0, 1], 1, [0], [1]), ([454], [1], 0, [], [0]),
            ([453, 454], [0, 1], 1, [1], [0]),
        ]:
            with self.subTest(indices=indices), self.assertRaises(EvidenceError):
                map_rows(176, positions, indices, n, targets, bonus)

    def test_padding_does_not_become_output(self):
        self.assertEqual(selected_tokens([1, 2, -1, -1]), [1, 2])
        for ids in ([1, -1, 2], [1, -2], [True]):
            with self.assertRaises(EvidenceError):
                selected_tokens(ids)

    def test_exact_request_join(self):
        self.assertTrue(same_request("chatcmpl-abc-1234abcd", "chatcmpl-abc"))
        self.assertFalse(same_request("chatcmpl-abc-other", "chatcmpl-abc"))
        self.assertFalse(same_request("chatcmpl-abcd", "chatcmpl-abc"))

    def test_fingerprint_rejects_other_boundary_length_and_finish(self):
        v = dict(sample_id=SAMPLE, token_ids=[1] * 284, finish_reason="stop")
        a = dict(sample_id=SAMPLE, token_ids=[1] * 282, finish_reason="stop")
        v["token_ids"][279], a["token_ids"][279] = 79126, 128273
        self.assertTrue(is_fingerprint(v, a))
        for change in ("earlier", "length", "finish", "token", "sample"):
            b = copy.deepcopy(a)
            if change == "earlier":
                b["token_ids"][200] = 9
            elif change == "length":
                b["token_ids"].append(1)
            elif change == "finish":
                b["finish_reason"] = "length"
            elif change == "token":
                b["token_ids"][279] = 79126
            else:
                b["sample_id"] = "other-sample"
            self.assertFalse(is_fingerprint(v, b))
        self.assertIsNone(first_difference([1], [1]))


class BoundaryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / "observer").mkdir()
        self.record = dict(response_id="chatcmpl-test", prompt_token_ids=[8] * 176,
                           token_ids=[7] * 279 + [79126] + [6] * 4)
        scores = {}
        for stage in ("raw", "processed"):
            vector = np.zeros(152064, dtype=np.float32)
            vector[79126] = 2
            path = self.output / "observer" / (stage + ".npy")
            np.save(path, vector, allow_pickle=False)
            scores[stage] = dict(file=path.name, sha256=file_sha(path), shape=[152064],
                                 argmax=79126, candidates=[dict(token_id=i, score=float(vector[i]))
                                                          for i in (79126, 128273)])
        rows = map_rows(176, [454], [0], 0, [], [0])
        self.forward = dict(kind="forward", request_id="chatcmpl-test-1234abcd",
                            prompt_token_ids=[8] * 176, active_query=[[454, 7]], query_bounds=[0, 1],
                            logits_indices=[0], rows=rows, target_row=rows[0], draft_count=0,
                            selected=[79126], accepted_drafts=0, rejected_drafts=0, **scores)
        self.events = [self.forward,
            dict(kind="account_after", request_id=self.forward["request_id"], stale=False,
                 output_count=280, committed_ids=self.record["token_ids"][:280]),
            dict(kind="cache_lookup", request_id=self.forward["request_id"],
                 cached_prefix_tokens=96, external_cached_tokens=0)]

    def test_complete_proof_links_model_row_sampler_commit_and_http(self):
        f, vectors = validate_boundary(self.record, self.events, self.output)
        self.assertEqual(f["target_row"]["position"], 454)
        self.assertEqual(vectors["raw"].dtype, np.float32)

    def test_missing_or_ambiguous_selected_forward_fails(self):
        for events in (self.events[1:], self.events + [self.forward]):
            with self.assertRaises(EvidenceError):
                validate_boundary(self.record, events, self.output)

    def test_incorrect_position_or_conditioning_input_fails(self):
        for position, token in ((453, 7), (454, 9)):
            events = copy.deepcopy(self.events)
            events[0]["active_query"] = [[position, token]]
            with self.assertRaises(EvidenceError):
                validate_boundary(self.record, events, self.output)

    def test_rejected_candidate_cannot_be_promoted(self):
        events = copy.deepcopy(self.events)
        events[0]["selected"] = []
        with self.assertRaises(EvidenceError):
            validate_boundary(self.record, events, self.output)

    def test_sampled_emitted_disagreement_is_retained_for_h3(self):
        events = copy.deepcopy(self.events)
        events[0]["selected"] = [128273]
        f, _ = validate_boundary(self.record, events, self.output)
        self.assertFalse(f["sampling_emission_agree"])
        self.assertEqual(f["sampled_token_id_at_target"], 128273)
        self.assertEqual(f["emitted_token_id_at_target"], 79126)

    def test_missing_commit_fails(self):
        with self.assertRaises(EvidenceError):
            validate_boundary(self.record, self.events[:1] + self.events[2:], self.output)

    def test_speculative_row_requires_accepted_prefix_and_keeps_draft_ids(self):
        events = copy.deepcopy(self.events)
        f = events[0]
        rows = map_rows(176, [452, 453, 454, 455, 456], list(range(5)), 4, list(range(4)), [4])
        f.update(active_query=[[452, 7], [453, 7], [454, 7], [455, 8], [456, 9]],
                 query_bounds=[0, 5], logits_indices=list(range(5)), rows=rows,
                 target_row=rows[2], draft_count=4, draft_token_ids=[7, 7, 8, 9],
                 selected=[7, 7, 79126], accepted_drafts=2, rejected_drafts=2)
        result, _ = validate_boundary(self.record, events, self.output)
        self.assertEqual(result["accepted_draft_token_ids"], [7, 7])
        self.assertEqual(result["rejected_draft_candidate_ids"], [8, 9])
        f["selected"][0] = 6
        with self.assertRaises(EvidenceError):
            validate_boundary(self.record, events, self.output)

    def test_boundary_commit_disagreement_is_retained(self):
        events = copy.deepcopy(self.events)
        events[1]["committed_ids"][279] = 128273
        result, _ = validate_boundary(self.record, events, self.output)
        self.assertEqual(result["committed_token_ids_at_target"], [128273])

    def test_missing_cache_or_observer_error_fails(self):
        for events in (self.events[:-1], self.events + [dict(kind="observer_error")]):
            with self.assertRaises(EvidenceError):
                validate_boundary(self.record, events, self.output)

    def test_changed_vector_or_missing_processed_scores_fails(self):
        path = self.output / "observer" / "raw.npy"
        with path.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaises(EvidenceError):
            validate_boundary(self.record, self.events, self.output)

    def test_schema_writes_and_sealing_never_overwrite(self):
        path = self.output / "manifest.json"
        write_new(path, {"schema": "m203.boundary.manifest.v1", "task_done": False})
        with self.assertRaises(FileExistsError):
            write_new(path, {})
        seal(self.output)
        with self.assertRaises(FileExistsError):
            seal(self.output)
        self.assertIn(file_sha(path), (self.output / "SHA256SUMS").read_text())

    def artifact(self, *, other_boundary=False):
        """Synthetic serialized artifacts, not a model/GPU simulator."""
        prompts = [dict(sample_id=f"fixture-{i}", request={"messages": [str(i)],
                        "temperature": 0, "seed": 42}) for i in range(54)]
        prompts[45]["sample_id"] = SAMPLE
        for p in prompts:
            p["request_sha256"] = fingerprint(p["request"])
        m = dict(plan={"prompts": prompts}, execution_sha="a" * 40, session_id="fixture-session",
                 gpu_uuid="GPU-fixture", boot_id="fixture-boot", target_revision="b" * 40,
                 draft_revision="c" * 40, reference_records={})
        env = dict(git_sha=m["execution_sha"], **{k: m[k] for k in
                   ("session_id", "gpu_uuid", "boot_id", "target_revision", "draft_revision")})
        for index, lane in enumerate(("vanilla", "adaptive")):
            output = self.output / lane
            output.mkdir()
            (output / "observer").mkdir()
            records, journal = [], []
            for i, p in enumerate(prompts[:46]):
                ids = [7] * 284 if lane == "vanilla" else [7] * 282
                ids[279] = 79126 if lane == "vanilla" else 128273
                if other_boundary and lane == "adaptive":
                    ids[200] = 9
                model = "fixture-" + lane
                record = dict(**p, ok=True, token_ids=ids, prompt_token_ids=[8] * 176,
                              completion_tokens=len(ids), finish_reason="stop", response_id=f"chatcmpl-{i}")
                payload = dict(p["request"], model=model, return_token_ids=True)
                body = dict(id=record["response_id"], model=model, prompt_token_ids=[8] * 176,
                            usage=dict(completion_tokens=len(ids)),
                            choices=[dict(token_ids=ids, finish_reason="stop")])
                journal.extend([dict(event="attempt", request=payload),
                    dict(event="result", request=payload, record=record,
                         http_status=200, raw_response_body=json.dumps(body))])
                records.append(record)
            m["reference_records"][lane] = records
            path = output / "responses.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in journal) + "\n")
            write_new(output / "run.json", dict(complete=True, diagnostic_only=True,
                diagnostic_through=SAMPLE, records=records, environment=env,
                responses_sha256=file_sha(path), started_at=str(index * 2), finished_at=str(index * 2 + 1)))
            config = dict(lane=lane, execution_sha=m["execution_sha"])
            write_new(output / "boundary_config.json", config)
            write_new(output / "observer_binding.json", dict(
                config_sha256=file_sha(output / "boundary_config.json"), config_fingerprint=fingerprint(config)))
            events = copy.deepcopy(self.events)
            for e in events:
                e.update(schema="m203.boundary.event.v1", lane=lane, execution_sha=m["execution_sha"],
                         config_fingerprint=fingerprint(config), request_id="chatcmpl-45-1234abcd")
            events[0]["selected"] = [records[-1]["token_ids"][279]]
            events[1]["committed_ids"] = records[-1]["token_ids"][:280]
            for i, e in enumerate(events):
                write_new(output / "observer" / f"{i}.json", e)
            for stage in ("raw", "processed"):
                shutil.copyfile(self.output / "observer" / f"{stage}.npy",
                                output / "observer" / f"{stage}.npy")
        write_new(self.output / "manifest.json", m)

    def test_output_schema_and_exact_fingerprint_gate_end_to_end(self):
        self.artifact()
        report = analyze(self.output)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "INTERPRETABLE", report)
        self.assertEqual(report["boundary"]["adaptive"]["selected"], [128273])
        self.assertFalse(report["task_done"])
        self.assertEqual(report["interpretation_status"], "NOT_YET_APPROVED")
        self.assertFalse(report["same_kv_cache_state_proven"])

    def test_index_200_is_non_reproduction_without_boundary_interpretation(self):
        self.artifact(other_boundary=True)
        report = analyze(self.output)
        self.assertEqual(report["status"], "NON_REPRODUCTION", report)
        self.assertNotIn("score_comparison", report)
        self.assertNotIn("boundary", report)

    def test_incomplete_run_retains_invalid_evidence_report(self):
        self.artifact()
        path = self.output / "adaptive" / "run.json"
        run = json.loads(path.read_text())
        run["complete"] = False
        path.write_text(json.dumps(run))
        report = analyze(self.output)
        self.assertEqual(report["status"], "INVALID_EVIDENCE")
        self.assertTrue((self.output / "boundary_comparison.json").exists())

    def test_valid_fingerprint_without_selected_row_is_invalid(self):
        self.artifact()
        path = self.output / "adaptive" / "observer" / "0.json"
        path.unlink()
        report = analyze(self.output)
        self.assertEqual(report["status"], "INVALID_EVIDENCE")


class SourceTransformTests(unittest.TestCase):
    def test_real_pinned_sources_restore_original_ast(self):
        sources = pinned_sources()
        self.assertEqual(set(sources), set(HASHES))
        for name, text in sources.items():
            with self.subTest(name=name):
                raw = text.encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), HASHES[name])
                patched = transform(name, raw)
                self.assertEqual(ast.dump(StripObserver().visit(ast.parse(patched))), ast.dump(ast.parse(raw)))
                with self.assertRaises(EvidenceError):
                    transform(name, raw + b"\n")
                with self.assertRaises(EvidenceError):
                    transform(name, patched)


class CapturePrefixTests(unittest.TestCase):
    def test_existing_collector_sends_exactly_46_requests(self):
        from experiments import protocol_b_capture as collector
        with tempfile.TemporaryDirectory() as tmp:
            prompts = [dict(workload="W1", sample_id=f"fixture-{i}", dataset_version="fixture",
                            prompt_version="fixture", request_sha256="a" * 64,
                            request={"messages": [str(i)]}) for i in range(54)]
            prompts[45]["sample_id"] = SAMPLE
            model = collector.LANES["vanilla"][1]
            sent = []

            def exchange(url, payload):
                sent.append(payload)
                return dict(http_status=200, raw_response_body=json.dumps(dict(
                    id=f"response-{len(sent)}", model=model, prompt_token_ids=[8],
                    choices=[dict(token_ids=[1], finish_reason="stop")], usage=dict(completion_tokens=1))))

            with patch.object(collector, "load_plan", return_value={"prompts": prompts}), \
                 patch.object(collector, "server_processes", return_value={}), \
                 patch.object(collector, "runtime", return_value={}), \
                 patch.object(collector, "validate_live_server"), \
                 patch.object(collector, "http_json", return_value={"data": [dict(id=model)]}), \
                 patch.object(collector, "request_raw", side_effect=exchange):
                run = collector.capture(Path(tmp) / "lane", "boundary279", "vanilla", "vanilla", "session",
                                        diagnostic_through=SAMPLE)
            self.assertEqual(len(sent), 46)
            self.assertEqual([r["sample_id"] for r in run["records"]],
                             [p["sample_id"] for p in prompts[:46]])
            self.assertEqual(sent[-1]["messages"], ["45"])
            self.assertTrue(run["diagnostic_only"])
            self.assertEqual(len(prompts), 54)

    def test_wrong_boundary_stops_before_contacting_server(self):
        from experiments import protocol_b_capture as collector
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(collector, "load_plan", return_value={"prompts": []}), \
             patch.object(collector, "server_processes") as server:
            with self.assertRaises(EvidenceError):
                collector.capture(Path(tmp) / "lane", "boundary279", "vanilla", "vanilla", "session",
                                  diagnostic_through="other-sample")
            server.assert_not_called()


class AtomicSourceWriteTests(unittest.TestCase):
    def setUp(self):
        from experiments import protocol_b_boundary_sources as sources
        self.sources = sources
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.package, self.run = root / "vllm", root / "run"
        self.run.mkdir()
        self.originals = {name: text.encode() for name, text in
                          pinned_sources().items()}
        for name, raw in self.originals.items():
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.spec = patch.object(sources.importlib.util, "find_spec",
                                 return_value=SimpleNamespace(origin=str(self.package / "__init__.py")))
        self.spec.start()
        self.addCleanup(self.spec.stop)

    def assert_complete_sources_and_backups(self):
        for name, raw in self.originals.items():
            self.assertIn((self.package / name).read_bytes(), (raw, transform(name, raw)))
            self.assertEqual((self.run / "vllm-originals" / name).read_bytes(), raw)
        self.assertFalse(list(self.package.rglob("pb279-source-*")))

    def test_atomic_install_and_restore_preserve_bytes_and_modes(self):
        modes = {name: stat.S_IMODE((self.package / name).stat().st_mode) for name in HASHES}
        self.sources.install(self.run)
        for name, raw in self.originals.items():
            self.assertEqual((self.package / name).read_bytes(), transform(name, raw))
            self.assertEqual(stat.S_IMODE((self.package / name).stat().st_mode), modes[name])
        self.sources.restore(self.run)
        for name, raw in self.originals.items():
            self.assertEqual((self.package / name).read_bytes(), raw)
            self.assertEqual(stat.S_IMODE((self.package / name).stat().st_mode), modes[name])

    def test_interrupted_install_staging_keeps_target_and_backups_intact(self):
        with patch.object(os, "fsync", side_effect=KeyboardInterrupt("interrupted staging")):
            with self.assertRaises(KeyboardInterrupt):
                self.sources.install(self.run)
        self.assert_complete_sources_and_backups()
        self.sources.restore(self.run)
        self.assertTrue(all((self.package / name).read_bytes() == raw
                            for name, raw in self.originals.items()))

    def test_failed_replace_during_install_leaves_recoverable_mixed_tree(self):
        real_replace, calls = os.replace, []

        def replace(source, target):
            self.assertEqual(Path(source).parent, Path(target).parent)
            calls.append(target)
            if len(calls) == 2:
                raise OSError("injected replace failure")
            return real_replace(source, target)

        with patch.object(os, "replace", side_effect=replace):
            with self.assertRaises(OSError):
                self.sources.install(self.run)
        self.assert_complete_sources_and_backups()
        self.sources.restore(self.run)
        self.assertTrue(all((self.package / name).read_bytes() == raw
                            for name, raw in self.originals.items()))

    def test_interrupted_restore_staging_leaves_sources_whole_and_can_resume(self):
        self.sources.install(self.run)
        with patch.object(os, "fsync", side_effect=KeyboardInterrupt("interrupted restore")):
            with self.assertRaises(KeyboardInterrupt):
                self.sources.restore(self.run)
        self.assert_complete_sources_and_backups()
        self.assertFalse((self.run / "restored.json").exists())
        self.sources.restore(self.run)
        self.assertTrue(all((self.package / name).read_bytes() == raw
                            for name, raw in self.originals.items()))

    def test_failed_replace_during_restore_preserves_authoritative_backups(self):
        self.sources.install(self.run)
        real_replace, calls = os.replace, []

        def replace(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("injected restore replace failure")
            return real_replace(source, target)

        with patch.object(os, "replace", side_effect=replace):
            with self.assertRaises(OSError):
                self.sources.restore(self.run)
        self.assert_complete_sources_and_backups()
        self.assertFalse((self.run / "restored.json").exists())
        self.sources.restore(self.run)
        self.assertTrue(all((self.package / name).read_bytes() == raw
                            for name, raw in self.originals.items()))


class ObserverConfigSafetyTests(unittest.TestCase):
    def test_malformed_or_unreadable_config_records_error_and_skips_capture(self):
        from experiments import protocol_b_boundary_observer as observer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "observer").mkdir()
            path = root / "boundary_config.json"
            for text in ("{broken", "[]", "{}", None):
                with self.subTest(config=text):
                    if text is None:
                        path.unlink()
                    else:
                        path.write_text(text)
                    before = set((root / "observer").glob("*.json"))
                    observer._local.context = ("stale", "capture")
                    with patch.dict(os.environ, PROTOCOL_B_BOUNDARY_CONFIG=str(path)), \
                         patch.object(observer, "begin") as begin:
                        observer.observe("begin")
                    begin.assert_not_called()
                    self.assertIsNone(observer._local.context)
                    added = set((root / "observer").glob("*.json")) - before
                    self.assertEqual(len(added), 1)
                    event = json.loads(added.pop().read_text())
                    self.assertEqual(event["kind"], "observer_error")
                    self.assertEqual(event["stage"], "begin")

    def test_config_error_with_unwritable_evidence_falls_back_to_server_log(self):
        from experiments import protocol_b_boundary_observer as observer
        with patch.dict(os.environ, PROTOCOL_B_BOUNDARY_CONFIG="missing-config.json"), \
             patch.object(Path, "read_text", side_effect=PermissionError("unreadable config")), \
             patch.object(observer, "write_new", side_effect=PermissionError("unwritable evidence")), \
             patch("sys.stderr") as log:
            observer._local.context = ("stale", "capture")
            observer.observe("begin")
        self.assertIsNone(observer._local.context)
        self.assertIn("observer_error", "".join(str(call) for call in log.write.call_args_list))


if __name__ == "__main__":
    unittest.main()
