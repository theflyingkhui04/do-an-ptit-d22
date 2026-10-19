"""Pinned boundary observers plus one explicitly approved diagnostic-time runtime fix.

Exact source hashes and strip-to-original AST verification precede installation.
The draft-tokenizer revision fix has its own exact AST normalization; observer
hooks still strip to pristine AST without any runtime-change allowance.
"""
import ast
import hashlib
import importlib.util
import os
import stat
import tempfile
from pathlib import Path
import json

from experiments.protocol_b import ROOT, require, write_new

HASHES = {
    "v1/spec_decode/llm_base_proposer.py": "5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab",
    "v1/spec_decode/draft_model.py": "6b884f49a75f01d1aa6b2d27159b63fa27f7ba7f0518d2810d1a5b565e14f026",
    "v1/worker/gpu/model_runner.py": "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
    "v1/worker/gpu/cudagraph_utils.py": "6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff",
    "v1/worker/gpu/input_batch.py": "d5dd956eb319bd69dd9e762047833d81ad40b21fce619fcb2133759e73620694",
    "v1/worker/gpu/block_table.py": "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a",
    "v1/worker/gpu/model_states/default.py": "bb614f814780f052e869f9acf31dfbfa68718ee7dc6252eefa901b5730ee8a2a",
    "v1/worker/gpu/attn_utils.py": "b9b81e59dda2720b1f9c434471381f879228832443e52756e469b4ad4905888f",
    "v1/worker/utils.py": "0ca3ec6bf4d20076145b7fe64e564962adebedc8f75a856b517ba0b2f2d77acf",
    "v1/kv_cache_interface.py": "1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb",
    "forward_context.py": "cc1473477dcc7b762c3901b25cc7bf35c06a2c42f193136fed553abf6fe33f0e",
    "model_executor/layers/attention/attention.py": "bc897e452f3aea603e353ca42a365d3be8836e9485d78cbf8cc4677baca2fb42",
    "v1/attention/backends/triton_attn.py": "1057663d110e1d5229495a799027b27277c1eca325d5b6780fa2259afbf58a1a",
    "v1/attention/ops/triton_unified_attention.py": "65d0fbcf33d00789df8ca688c0a1c908e710cfaef53471178747ecae11f05829",
    "v1/worker/gpu/sample/sampler.py": "832c9945d201a1e730ff9b4b52076aa83bd4a5064e1e4cfea952cf84d4a8d1f0",
    "v1/worker/gpu_model_runner.py": "87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51",
    "v1/sample/sampler.py": "f95805b2c6d06218c724826aa5f881a606e8d0a534a566aee7c85948520dd216",
    "v1/sample/rejection_sampler.py": "0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750",
    "v1/core/sched/scheduler.py": "c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884",
}
INSERTIONS = {
    "v1/worker/gpu/model_runner.py": [
        ("        # Run model.\n",
         '        _pb279_observe("kv_v2_prepare", self, input_batch, batch_desc, model_inputs, attn_metadata, slot_mappings_by_layer)\n'),
        ("                    model_output = self.cudagraph_manager.run_pw_graph(\n",
         '                    _pb279_observe("kv_v2_piecewise", self.cudagraph_manager)\n'),
    ],
    "v1/worker/gpu/cudagraph_utils.py": [
        ("        self.graphs[desc].replay()\n",
         '        _pb279_observe("kv_v2_full", self, desc)\n'),
        ("            def forward_fn(cg_mode: CUDAGraphMode) -> None:\n",
         '            _pb279_observe("kv_capture", self, desc, model_inputs, attn_metadata, slot_mappings, kv_cache_config, warmup)\n\n'),
    ],
    "v1/worker/gpu/sample/sampler.py": [
        ("        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear\n",
         '        _pb279_observe("v2_begin", self, input_batch, logits)\n'),
        ("            sampled = gumbel_sample(\n",
         '            _pb279_observe("v2_processed", self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos)\n'),
        ("        return sampler_output\n",
         '        _pb279_observe("v2_end", sampler_output)\n'),
    ],
    "v1/worker/gpu_model_runner.py": [
        ("            model_output = self._model_forward(\n",
         '            _pb279_observe("kv_v1", self, scheduler_output, input_ids, positions, attn_metadata, slot_mappings, cudagraph_mode)\n'),
        ("        # Apply structured output bitmasks if present.\n",
         '        _pb279_observe("begin", self, scheduler_output, logits, spec_decode_metadata, grammar_output)\n'),
        ("        self._update_states_after_model_execute(\n",
         '        _pb279_observe("end", self, sampler_output)\n'),
        ("                self._copy_draft_token_ids_to_cpu(scheduler_output)\n",
         '                _pb279_observe("early_produced", self, scheduler_output)\n'),
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
         '            _pb279_observe("account_before", request, new_token_ids, scheduler_output, output_is_stale, generated_token_ids)\n'),
        ("            elif request.pooling_params and pooler_output is not None:\n",
         '                _pb279_observe("account_after", request, new_token_ids, scheduler_output, output_is_stale, generated_token_ids)\n'),
    ],
}

# These are AFTER-statement observations; historical INSERTIONS stay before.
POSTWRITE_INSERTIONS = {
    "v1/worker/gpu/cudagraph_utils.py": (
        "        self.graphs[desc].replay()\n",
        '        _pb279_observe("postwrite_v2_full", self, desc)\n'),
    "v1/worker/gpu_model_runner.py": (
        "            model_output = self._model_forward(\n"
        "                input_ids=input_ids,\n"
        "                positions=positions,\n"
        "                intermediate_tensors=intermediate_tensors,\n"
        "                inputs_embeds=inputs_embeds,\n"
        "                **model_kwargs,\n"
        "            )\n",
        '            _pb279_observe("postwrite_v1", self, scheduler_output)\n'),
}

# Keep runtime changes distinct from observation-only INSERTIONS in code and evidence.
DRAFT_SOURCE = "v1/spec_decode/draft_model.py"
RUNTIME_FIXES = {DRAFT_SOURCE: "draft-tokenizer-revision-v1"}
PATCHED_FILES = (*INSERTIONS, *RUNTIME_FIXES)
DRAFT_TOKENIZER_CALL = (
    "            draft_tokenizer = get_tokenizer(\n"
    "                spec.draft_model_config.model,\n"
    "                trust_remote_code=spec.draft_model_config.trust_remote_code,\n"
    "            )\n"
)
DRAFT_REVISION_LINE = "                revision=spec.draft_model_config.revision,\n"


class StripObserver(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == "experiments.protocol_b_boundary_observer" else node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_pb279_observe"):
            return None
        return self.generic_visit(node)


def validate_scheduler_observation(raw, patched):
    """Permit only the two scoped observations in the frozen scheduler glue."""
    tree = ast.parse(patched)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == "experiments.protocol_b_boundary_observer"]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_pb279_observe"]
    expected_import = ast.parse(
        "from experiments.protocol_b_boundary_observer import observe as _pb279_observe").body[0]
    expected_call = ast.parse('_pb279_observe("early_schedule", self, scheduler_output)').body[0].value
    require(len(imports) == 1 and ast.dump(imports[0]) == ast.dump(expected_import)
            and len(calls) == 2 and all(ast.dump(c) == ast.dump(expected_call) for c in calls),
            "Unapproved scheduler observation")
    require(ast.dump(StripObserver().visit(tree)) == ast.dump(ast.parse(raw)),
            "Frozen adaptive scheduler inference AST changed")
    compile(patched, "adaptive_scheduler.py", "exec")


def normalized_transform_ast(name, patched):
    """Strip hooks; only the pinned draft assignment may lose the exact approved keyword."""
    tree = ast.parse(patched)
    if name in INSERTIONS:
        tree = StripObserver().visit(tree)
    if name == DRAFT_SOURCE:
        calls = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "draft_tokenizer" and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name) and node.value.func.id == "get_tokenizer"]
        require(len(calls) == 1, "Missing/ambiguous draft tokenizer fix")
        call = calls[0]
        revisions = [kw for kw in call.keywords if kw.arg == "revision"]
        expected = ast.parse("spec.draft_model_config.revision", mode="eval").body
        require(len(revisions) == 1 and ast.dump(revisions[0].value) == ast.dump(expected),
                "Unapproved draft tokenizer revision expression")
        call.keywords.remove(revisions[0])
    return tree


def validate_transform(name, raw, patched):
    require(hashlib.sha256(raw).hexdigest() == HASHES[name], "Pristine vLLM source drift: " + name)
    if name in POSTWRITE_INSERTIONS:
        anchor, insertion = POSTWRITE_INSERTIONS[name]
        target = ast.dump(ast.parse(anchor.lstrip()).body[0])
        hook = ast.dump(ast.parse(insertion.lstrip()).body[0])
        tree = ast.parse(patched)
        locations = []
        for node in ast.walk(tree):
            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    locations.extend((value, i) for i, item in enumerate(value)
                                     if isinstance(item, ast.stmt) and ast.dump(item) == target)
        stage = ast.parse(insertion.lstrip()).body[0].value.args[0].value
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_pb279_observe"
                 and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == stage]
        require(len(locations) == len(calls) == 1, "Missing/duplicate post-write source hook")
        body, index = locations[0]
        require(index > 0 and index + 1 < len(body)
                and ast.dump(body[index + 1]) == hook, "Post-write hook is not immediately after execution")
        before = ('_pb279_observe("kv_v2_full", self, desc)' if stage == "postwrite_v2_full" else
                  '_pb279_observe("kv_v1", self, scheduler_output, input_ids, positions, attn_metadata, slot_mappings, cudagraph_mode)')
        require(ast.dump(body[index - 1]) == ast.dump(ast.parse(before).body[0]),
                "Historical pre-forward hook moved")
    require(ast.dump(normalized_transform_ast(name, patched)) == ast.dump(ast.parse(raw)),
            "Unapproved inference AST change: " + name)
    compile(patched, name, "exec")


def transform(name, raw):
    require(hashlib.sha256(raw).hexdigest() == HASHES[name], "Pristine vLLM source drift: " + name)
    if name not in PATCHED_FILES:
        return raw
    text = raw.decode("utf-8")
    for anchor, insertion in INSERTIONS.get(name, ()):
        require(text.count(anchor) == 1, "Nonunique source anchor: " + name)
        text = text.replace(anchor, insertion + anchor, 1)
    if name in POSTWRITE_INSERTIONS:
        anchor, insertion = POSTWRITE_INSERTIONS[name]
        require(text.count(anchor) == 1, "Nonunique post-write source anchor")
        text = text.replace(anchor, anchor + insertion, 1)
    if name == DRAFT_SOURCE:
        require(text.count(DRAFT_TOKENIZER_CALL) == 1, "Nonunique draft tokenizer anchor")
        text = text.replace(DRAFT_TOKENIZER_CALL,
                            DRAFT_TOKENIZER_CALL.replace("                trust_remote_code=",
                                                         DRAFT_REVISION_LINE + "                trust_remote_code="), 1)
    if name in INSERTIONS:
        nodes = ast.parse(text).body
        boundary = max([n.end_lineno for n in nodes if isinstance(n, ast.ImportFrom)
                        and n.module == "__future__"] +
                       ([nodes[0].end_lineno] if isinstance(nodes[0], ast.Expr)
                        and isinstance(nodes[0].value, ast.Constant)
                        and isinstance(nodes[0].value.value, str) else [0]))
        lines = text.splitlines(keepends=True)
        lines.insert(boundary, "from experiments.protocol_b_boundary_observer import observe as _pb279_observe\n")
        text = "".join(lines)
    patched = text.encode()
    validate_transform(name, raw, patched)
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
    # Authenticate the whole bounded dependency set before changing any source.
    supported = {name: transform(name, raw) for name, raw in originals.items()}
    patched = {name: supported[name] for name in PATCHED_FILES}
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
        "runtime_fixes": RUNTIME_FIXES,
        "observer_sha256": hashlib.sha256(
            (ROOT / "experiments/protocol_b_boundary_observer.py").read_bytes()).hexdigest()})
    # Every backup/provenance file exists before the first installed-source write.
    for name, raw in patched.items():
        _write_atomically(root / name, raw)


def authenticate(directory, installed=False):
    """Authenticate preserved bytes, transforms, and optionally live installed files."""
    directory = Path(directory)
    state = json.loads((directory / "instrumentation.json").read_text())
    require(state["original_hashes"] == HASHES, "KV source hash set differs")
    require(state.get("runtime_fixes") == RUNTIME_FIXES, "Runtime fix declaration differs")
    originals = {name: (directory / "vllm-originals" / name).read_bytes() for name in HASHES}
    supported = {name: transform(name, raw) for name, raw in originals.items()}
    patched_hashes = {name: hashlib.sha256(supported[name]).hexdigest() for name in PATCHED_FILES}
    require(state["patched_hashes"] == patched_hashes, "KV patched source hashes differ")
    expected = {name: hashlib.sha256(raw).hexdigest() for name, raw in supported.items()}
    if installed:
        spec = importlib.util.find_spec("vllm")
        require(spec is not None and Path(spec.origin).parent.resolve()
                == Path(state["vllm_root"]).resolve(), "KV installed package differs")
        for name, digest in expected.items():
            require(hashlib.sha256((Path(state["vllm_root"]) / name).read_bytes()).hexdigest()
                    == digest, "KV installed source drift: " + name)
        require(state["observer_sha256"] == hashlib.sha256(
            Path(__file__).with_name("protocol_b_boundary_observer.py").read_bytes()).hexdigest(),
            "KV observer source drift")
    return dict(original_hashes=HASHES, installed_hashes=expected,
                observer_sha256=state["observer_sha256"], runtime_fixes=RUNTIME_FIXES)


def restore(directory):
    import json
    directory = Path(directory)
    state = json.loads((directory / "instrumentation.json").read_text())
    root = Path(state["vllm_root"])
    originals = {name: (directory / "vllm-originals" / name).read_bytes() for name in HASHES}
    for name, raw in originals.items():
        require(hashlib.sha256(raw).hexdigest() == HASHES[name], "Backup drift")
        require((root / name).read_bytes() in (raw, transform(name, raw)), "Unknown installed source")
    for name in PATCHED_FILES:
        _write_atomically(root / name, originals[name])
    write_new(directory / "restored.json", {"original_hashes": HASHES, "runtime_fixes_removed": RUNTIME_FIXES})
