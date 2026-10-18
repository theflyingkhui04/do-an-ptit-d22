"""Minimal observation-only subset of archive a7b52e7's sources.py.

Exact source hashes and strip-to-original AST verification precede installation.
No attention patches, recompute controls, startup or graph-setting changes.
"""
import ast
import hashlib
import importlib.util
import os
import stat
import tempfile
from pathlib import Path

from experiments.protocol_b import ROOT, require, write_new

HASHES = {
    "v1/worker/gpu_model_runner.py": "87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51",
    "v1/sample/sampler.py": "f95805b2c6d06218c724826aa5f881a606e8d0a534a566aee7c85948520dd216",
    "v1/sample/rejection_sampler.py": "0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750",
    "v1/core/sched/scheduler.py": "c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884",
}
INSERTIONS = {
    "v1/worker/gpu_model_runner.py": [
        ("        # Apply structured output bitmasks if present.\n",
         '        _pb279_observe("begin", self, scheduler_output, logits, spec_decode_metadata, grammar_output)\n'),
        ("        self._update_states_after_model_execute(\n",
         '        _pb279_observe("end", self, sampler_output)\n'),
    ],
    "v1/sample/sampler.py": [
        ("        if sampling_metadata.all_random:\n",
         '        _pb279_observe("sample_scores", logits, sampling_metadata)\n'),
    ],
    "v1/sample/rejection_sampler.py": [
        ("        output_token_ids = rejection_sample(\n",
         '        _pb279_observe("verify_scores", target_logits, sampling_metadata, metadata)\n'),
    ],
    "v1/core/sched/scheduler.py": [
        ("                    # Total computed tokens (local + external).\n",
         '                    _pb279_observe("cache_lookup", request, num_new_local_computed_tokens, num_external_computed_tokens)\n'),
        ("            # Check for stop and update request status.\n",
         '            _pb279_observe("account_before", request, new_token_ids, scheduler_output, output_is_stale)\n'),
        ("            elif request.pooling_params and pooler_output is not None:\n",
         '                _pb279_observe("account_after", request, new_token_ids, scheduler_output, output_is_stale)\n'),
    ],
}


class StripObserver(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == "experiments.protocol_b_boundary_observer" else node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_pb279_observe"):
            return None
        return self.generic_visit(node)


def transform(name, raw):
    require(hashlib.sha256(raw).hexdigest() == HASHES[name], "Pristine vLLM source drift: " + name)
    text = raw.decode("utf-8")
    for anchor, insertion in INSERTIONS[name]:
        require(text.count(anchor) == 1, "Nonunique source anchor: " + name)
        text = text.replace(anchor, insertion + anchor, 1)
    nodes = ast.parse(text).body
    boundary = max([n.end_lineno for n in nodes if isinstance(n, ast.ImportFrom)
                    and n.module == "__future__"] +
                   ([nodes[0].end_lineno] if isinstance(nodes[0], ast.Expr)
                    and isinstance(nodes[0].value, ast.Constant)
                    and isinstance(nodes[0].value.value, str) else [0]))
    lines = text.splitlines(keepends=True)
    lines.insert(boundary, "from experiments.protocol_b_boundary_observer import observe as _pb279_observe\n")
    patched = "".join(lines).encode()
    compile(patched, name, "exec")
    require(ast.dump(StripObserver().visit(ast.parse(patched))) == ast.dump(ast.parse(raw)),
            "Observer changed inference AST")
    return patched


def _write_atomically(path, raw):
    """Narrow port of archive scripts/m2_03_install_a3_v8.py's writer."""
    temporary = None
    mode = stat.S_IMODE(path.stat().st_mode)
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix="pb279-source-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install(directory):
    """Explicit Kaggle-only preparation; caller must stop servers first."""
    spec = importlib.util.find_spec("vllm")
    require(spec is not None, "vLLM unavailable")
    root = Path(spec.origin).parent
    originals = {name: (root / name).read_bytes() for name in HASHES}
    patched = {name: transform(name, raw) for name, raw in originals.items()}
    backup = Path(directory) / "vllm-originals"
    backup.mkdir(exist_ok=False)
    for name, raw in originals.items():
        path = backup / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)
    write_new(Path(directory) / "instrumentation.json", {
        "vllm_root": str(root), "original_hashes": HASHES,
        "patched_hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in patched.items()},
        "observer_sha256": hashlib.sha256(
            (ROOT / "experiments/protocol_b_boundary_observer.py").read_bytes()).hexdigest()})
    # Every backup/provenance file exists before the first installed-source write.
    for name, raw in patched.items():
        _write_atomically(root / name, raw)


def restore(directory):
    import json
    directory = Path(directory)
    state = json.loads((directory / "instrumentation.json").read_text())
    root = Path(state["vllm_root"])
    originals = {name: (directory / "vllm-originals" / name).read_bytes() for name in HASHES}
    for name, raw in originals.items():
        require(hashlib.sha256(raw).hexdigest() == HASHES[name], "Backup drift")
        require((root / name).read_bytes() in (raw, transform(name, raw)), "Unknown installed source")
    for name, raw in originals.items():
        _write_atomically(root / name, raw)
    write_new(directory / "restored.json", {"original_hashes": HASHES})
