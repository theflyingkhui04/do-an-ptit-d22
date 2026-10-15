"""Exact upstream v0.30.0 source transform. No dependency on any previous hook.

Only import + standalone observer calls are added. Removing those AST nodes
must recover the exact original AST. Installed source hashes must match first.
"""
import ast
import gzip
from pathlib import Path
from .common import ROOT, decode, need, sha

MANIFEST = ROOT / "experiments/m2_03_targeted/upstream.json"
FIXTURE = ROOT / "tests/fixtures/m203_v030_sources.json.gz"
IMPORT = ("from experiments.m2_03_targeted.observer import observe as _m203_observe, "
          "control as _m203_control\n")

# Exact insertion anchors from official v0.30.0. Whole-file SHA pin is mandatory.
INSERTIONS = {
    "v1/worker/gpu_model_runner.py": [
        ("        # Update num_computed_tokens on GPU. In async spec decode,\n",
         '        _m203_observe("correction_before", self, scheduler_output)\n'),
        ("            model_output = self._model_forward(\n",
         '            _m203_observe("prepared", self, scheduler_output, input_ids, positions, slot_mappings_by_group)\n'),
        ("        # Apply structured output bitmasks if present.\n",
         '        _m203_observe("begin", self, scheduler_output, logits, spec_decode_metadata, grammar_output)\n'),
        ("        self._update_states_after_model_execute(\n",
         '        _m203_observe("end", self, sampler_output)\n'),
    ],
    "v1/sample/sampler.py": [
        ("        if sampling_metadata.all_random:\n",
         '        _m203_observe("sample_scores", logits, sampling_metadata)\n'),
    ],
    "v1/sample/rejection_sampler.py": [
        ("        output_token_ids = rejection_sample(\n",
         '        _m203_observe("verify_scores", target_logits, sampling_metadata, metadata)\n'),
    ],
    "v1/core/sched/scheduler.py": [
        ("        self.current_step += 1\n",
         '        _m203_control("before_schedule", self)\n'),
        ("        return scheduler_output\n",
         '        _m203_control("after_schedule", self, scheduler_output)\n'),
        ("            # Check for stop and update request status.\n",
         '            _m203_observe("account_before", request, new_token_ids, scheduler_output, output_is_stale)\n'),
        ("            elif request.pooling_params and pooler_output is not None:\n",
         '                _m203_observe("account_after", request, new_token_ids, scheduler_output, output_is_stale)\n'),
    ],
}


def manifest():
    return decode(MANIFEST.read_bytes())


def fixture_sources():
    return decode(gzip.decompress(FIXTURE.read_bytes()))


class StripObserver(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == "experiments.m2_03_targeted.observer" else node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("_m203_observe", "_m203_control")):
            return None
        return node


def transform(name, raw):
    need(sha(raw) == manifest()["sha256"][name], "unpatched vLLM source hash drift: " + name)
    if name not in INSERTIONS:
        return raw
    text = raw.decode("utf-8")
    for anchor, insertion in INSERTIONS[name]:
        need(text.count(anchor) == 1, "source anchor not unique: " + name)
        text = text.replace(anchor, insertion + anchor, 1)
    nodes = ast.parse(text).body
    boundary = max([n.end_lineno for n in nodes if isinstance(n, ast.ImportFrom)
                    and n.module == "__future__"] +
                   ([nodes[0].end_lineno] if isinstance(nodes[0], ast.Expr)
                    and isinstance(nodes[0].value, ast.Constant)
                    and isinstance(nodes[0].value.value, str) else [0]))
    lines = text.splitlines(keepends=True)
    lines.insert(boundary, IMPORT)
    patched = "".join(lines).encode("utf-8")
    compile(patched, name, "exec")
    original_tree = ast.dump(ast.parse(raw), include_attributes=False)
    stripped = StripObserver().visit(ast.parse(patched))
    need(ast.dump(stripped, include_attributes=False) == original_tree, "observer changed inference AST")
    return patched


def verify_installed(root, patched=False):
    hashes = {}
    for name, expected in manifest()["sha256"].items():
        raw = (Path(root) / name).read_bytes()
        reference = fixture_sources()[name].encode("utf-8")
        need(sha(reference) == expected, "vendored source fixture drift")
        wanted = transform(name, reference) if patched else reference
        need(raw == wanted, "installed source mismatch: " + name)
        hashes[name] = sha(raw)
    return hashes


def installed_state(root):
    """Read-only classification; a mixed/unknown tree is never inference-ready."""
    files, states = {}, set()
    for name, text in fixture_sources().items():
        raw = text.encode('utf-8')
        patched = transform(name, raw)
        try:
            actual = (Path(root) / name).read_bytes()
            actual_sha = sha(actual)
        except OSError:
            actual, actual_sha = None, None
        state = ('UNKNOWN' if actual not in (raw, patched) else
                 'UNCHANGED' if raw == patched else 'PRISTINE' if actual == raw else 'PATCHED')
        states.add(state)
        files[name] = {'state': state, 'sha256': actual_sha,
                       'pristine_sha256': sha(raw), 'patched_sha256': sha(patched)}
    mode = ('UNKNOWN' if 'UNKNOWN' in states else 'MIXED' if {'PRISTINE', 'PATCHED'} <= states
            else 'PATCHED' if 'PATCHED' in states else 'PRISTINE')
    return {'mode': mode, 'files': files}
