"""Archive dry-run/commit safeguards; temporary files only, zero GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'm2_03_archive_failed_prestarts.py'
SPEC = importlib.util.spec_from_file_location('m2archivetested', str(SCRIPT))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class TestFailedPrestartArchival(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.diag = self.root / 'experiments/runs/m2_03/diagnostics'
        self.diag.mkdir(parents=True)
        self.patch1 = patch.object(M, 'ROOT', self.root)
        self.patch2 = patch.object(M, 'DIAGNOSTICS', self.diag)
        self.patch1.start()
        self.patch2.start()
        for src, _ in M.candidates():
            src.mkdir()
            (src / 'run_state.json').write_text(json.dumps({
                'status': 'FAILED_OR_INTERRUPTED',
                'reason': 'RuntimeError: STOP: server start failed: exit=1'}))
            (src / 'startup_stdout.log').write_text('Usage: {lane1|lane2|lane3|stop|status}')
            (src / 'manifest.json').write_text('{"test": "no GPU"}')

    def tearDown(self):
        self.patch2.stop()
        self.patch1.stop()
        self.tmp.cleanup()

    def test_archives_without_modifying_evidence_bytes(self):
        manifest = M.plan()
        self.assertEqual(len(manifest), 4)
        M.archive()
        self.assertTrue((self.diag / M.INDEX_NAME).exists())
        index = json.loads((self.diag / M.INDEX_NAME).read_text())
        self.assertEqual(len(index['cells']), 4)
        for src, dst in M.candidates():
            self.assertFalse(src.exists())
            self.assertTrue(dst.exists())
        for rec in index['cells']:
            for path, sha in rec['files_sha256'].items():
                self.assertEqual(M.digest(self.root / rec['archive_path'] / path), sha)
        with self.assertRaisesRegex(RuntimeError, 'archive index already exists'):
            M.plan()

    def test_refuses_if_any_probes_exist_and_keeps_all_dirs(self):
        src = M.candidates()[1][0]
        (src / 'probes.jsonl').write_text('{"unexpected": 1}\n')
        with self.assertRaisesRegex(RuntimeError, 'non-prestart evidence exists'):
            M.plan()
        for original, archive in M.candidates():
            self.assertTrue(original.is_dir())
            self.assertFalse(archive.exists())

    def test_refuses_non_startup_failure(self):
        src = M.candidates()[0][0]
        (src / 'run_state.json').write_text(json.dumps({
            'status': 'FAILED_OR_INTERRUPTED', 'reason': 'bad GPU'}))
        with self.assertRaisesRegex(RuntimeError, 'unexpected failure reason'):
            M.plan()


if __name__ == '__main__':
    unittest.main()
