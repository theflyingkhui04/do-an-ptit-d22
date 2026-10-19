"""CPU-only source/AST and temporary-package lifecycle checks; no tokenizer downloads."""
import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_protocol_b_boundary as boundary_tests
from experiments.protocol_b import EvidenceError
from experiments import protocol_b_boundary_sources as sources


def tokenizer_call(source, assignment):
    calls = [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id == assignment]
    if len(calls) != 1:
        raise AssertionError("ambiguous tokenizer assignment")
    return calls[0]


class DraftTokenizerRevisionTests(unittest.TestCase):
    def setUp(self):
        self.name = sources.DRAFT_SOURCE
        self.raw = boundary_tests.pinned_sources()[self.name].encode()
        self.patched = sources.transform(self.name, self.raw)

    def test_pristine_hash_required_and_reapplying_fix_rejected(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), sources.HASHES[self.name])
        for raw in (self.raw + b"\n", self.patched,
                    self.raw.replace(b"spec.draft_model_config.model", b"spec.draft_model_config.tokenizer")):
            with self.assertRaises(EvidenceError):
                sources.transform(self.name, raw)

    def test_exact_one_line_draft_change_and_no_observer_import(self):
        expected = self.raw.replace(
            b"                trust_remote_code=spec.draft_model_config.trust_remote_code,\n",
            sources.DRAFT_REVISION_LINE.encode() +
            b"                trust_remote_code=spec.draft_model_config.trust_remote_code,\n")
        self.assertEqual(self.patched, expected)
        self.assertNotIn(b"_pb279_observe", self.patched)
        call = tokenizer_call(self.patched, "draft_tokenizer")
        revisions = [kw for kw in call.keywords if kw.arg == "revision"]
        self.assertEqual(len(revisions), 1)
        self.assertEqual(ast.unparse(revisions[0].value), "spec.draft_model_config.tokenizer_revision")
        self.assertEqual(len(call.keywords), len(tokenizer_call(self.raw, "draft_tokenizer").keywords) + 1)

    def test_target_tokenizer_call_is_unchanged(self):
        self.assertEqual(ast.dump(tokenizer_call(self.raw, "target_tokenizer")),
                         ast.dump(tokenizer_call(self.patched, "target_tokenizer")))

    def test_draft_call_passes_tokenizer_revision_value_not_model_revision(self):
        revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        spec = NS(draft_model_config=NS(model="cached-draft", revision="different-model-revision",
                                       tokenizer_revision=revision, trust_remote_code=False))
        received = []
        def get_tokenizer(*args, **kwargs):
            received.append((args, kwargs))
        expr = ast.Expression(tokenizer_call(self.patched, "draft_tokenizer"))
        eval(compile(expr, "<draft tokenizer keyword>", "eval"), {"spec": spec, "get_tokenizer": get_tokenizer})
        self.assertEqual(received, [(("cached-draft",), dict(revision=revision, trust_remote_code=False))])

    def test_only_approved_normalization_reconstructs_pristine_ast(self):
        self.assertNotEqual(ast.dump(sources.StripObserver().visit(ast.parse(self.patched))),
                            ast.dump(ast.parse(self.raw)))
        self.assertEqual(ast.dump(sources.normalized_transform_ast(self.name, self.patched)),
                         ast.dump(ast.parse(self.raw)))
        sources.validate_transform(self.name, self.raw, self.patched)

    def test_wrong_missing_or_duplicate_revision_fails_closed(self):
        for expression in ("None", "'main'", "spec.draft_model_config.revision",
                           "spec.target_model_config.tokenizer_revision",
                           "getattr(spec.draft_model_config, 'tokenizer_revision', None)"):
            changed = self.patched.replace(b"revision=spec.draft_model_config.tokenizer_revision,",
                                          ("revision=" + expression + ",").encode())
            with self.subTest(expression=expression), self.assertRaises(EvidenceError):
                sources.validate_transform(self.name, self.raw, changed)
        for changed in (self.raw, self.patched.replace(sources.DRAFT_REVISION_LINE.encode(),
                                                      sources.DRAFT_REVISION_LINE.encode() * 2)):
            with self.assertRaises(EvidenceError):
                sources.validate_transform(self.name, self.raw, changed)

    def test_unrelated_ast_mutations_and_target_revision_are_rejected(self):
        mutations = [
            lambda s: s.replace("self._raise_if_vocab_size_mismatch()", "self._raise_if_draft_tp_mismatch()"),
            lambda s: s.replace("spec.draft_model_config.model,", "spec.draft_model_config.tokenizer,"),
            lambda s: s.replace("trust_remote_code=spec.target_model_config.trust_remote_code,",
                                "trust_remote_code=spec.target_model_config.trust_remote_code,\n"
                                "                revision=spec.target_model_config.tokenizer_revision,"),
            lambda s: s + "\n_pb279_observe('unexpected_runtime_hook')\n",
        ]
        for mutate in mutations:
            with self.assertRaises(EvidenceError):
                sources.validate_transform(self.name, self.raw, mutate(self.patched.decode()).encode())

    def test_observer_only_ast_gate_is_unchanged_for_every_observer_file(self):
        originals = boundary_tests.pinned_sources()
        for name in sources.INSERTIONS:
            raw = originals[name].encode()
            patched = sources.transform(name, raw)
            self.assertEqual(ast.dump(sources.StripObserver().visit(ast.parse(patched))),
                             ast.dump(ast.parse(raw)))
        name = "v1/sample/sampler.py"
        raw = originals[name].encode()
        patched = sources.transform(name, raw) + b"\nunapproved_runtime_change = True\n"
        with self.assertRaises(EvidenceError):
            sources.validate_transform(name, raw, patched)

    def test_runtime_normalization_cannot_be_applied_to_another_file(self):
        name = "v1/spec_decode/llm_base_proposer.py"
        raw = boundary_tests.pinned_sources()[name].encode()
        with self.assertRaises(EvidenceError):
            sources.validate_transform(name, raw, self.patched)


class TokenizerFixLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.package, self.run = root / "vllm", root / "run"
        self.run.mkdir()
        self.originals = {name: text.encode() for name, text in boundary_tests.pinned_sources().items()}
        for name, raw in self.originals.items():
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        mock = patch.object(sources.importlib.util, "find_spec",
                            return_value=NS(origin=str(self.package / "__init__.py")))
        mock.start()
        self.addCleanup(mock.stop)

    def test_install_authenticate_restore_and_explicit_runtime_declaration(self):
        sources.install(self.run)
        state = json.loads((self.run / "instrumentation.json").read_text())
        self.assertEqual(state["runtime_fixes"], sources.RUNTIME_FIXES)
        self.assertNotIn(sources.DRAFT_SOURCE, sources.INSERTIONS)
        self.assertIn(sources.DRAFT_SOURCE, state["patched_hashes"])
        expected = sources.transform(sources.DRAFT_SOURCE, self.originals[sources.DRAFT_SOURCE])
        self.assertEqual((self.package / sources.DRAFT_SOURCE).read_bytes(), expected)
        self.assertEqual((self.run / "vllm-originals" / sources.DRAFT_SOURCE).read_bytes(),
                         self.originals[sources.DRAFT_SOURCE])
        auth = sources.authenticate(self.run, installed=True)
        self.assertEqual(auth["runtime_fixes"], sources.RUNTIME_FIXES)
        sources.restore(self.run)
        for name, raw in self.originals.items():
            self.assertEqual((self.package / name).read_bytes(), raw)
        restored = json.loads((self.run / "restored.json").read_text())
        self.assertEqual(restored["runtime_fixes_removed"], sources.RUNTIME_FIXES)

    def test_source_drift_aborts_before_any_installed_write(self):
        path = self.package / sources.DRAFT_SOURCE
        path.write_bytes(path.read_bytes() + b"\n")
        with patch.object(sources, "_write_atomically") as write:
            with self.assertRaises(EvidenceError):
                sources.install(self.run)
        write.assert_not_called()

    def test_runtime_declaration_or_installed_fix_drift_is_rejected(self):
        sources.install(self.run)
        state_path = self.run / "instrumentation.json"
        state = json.loads(state_path.read_text())
        original = copy.deepcopy(state)
        state["runtime_fixes"] = {}
        state_path.write_text(json.dumps(state))
        with self.assertRaises(EvidenceError):
            sources.authenticate(self.run)
        state_path.write_text(json.dumps(original))
        path = self.package / sources.DRAFT_SOURCE
        path.write_bytes(path.read_bytes().replace(
            b"revision=spec.draft_model_config.tokenizer_revision,",
            b"revision=spec.draft_model_config.revision,"))
        with self.assertRaises(EvidenceError):
            sources.authenticate(self.run, installed=True)
        with self.assertRaises(EvidenceError):
            sources.restore(self.run)


if __name__ == "__main__":
    unittest.main()
