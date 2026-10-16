"""Exact upstream v0.30.0 source transform. No dependency on any previous hook.

Observer calls and R16-only constexpr correctness guards are added. Stripping
those additions must recover the exact original AST. Source hashes match first.
The separately pinned production profile adds only one observation call and
restores to its production baseline; it never installs diagnostic guards/control.
"""
import ast
import gzip
from pathlib import Path
from .common import ROOT, decode, need, sha

MANIFEST = ROOT / "experiments/m2_03_targeted/upstream.json"
FIXTURE = ROOT / "tests/fixtures/m203_v030_sources.json.gz"
ATTENTION_SOURCE = "v1/attention/ops/triton_unified_attention.py"
R15_ATTENTION_PATCHED_SHA256 = "2295fef50bb0c3400c384dc157032a36d6699402a850085f3e7419dd9c24fdb7"
PRODUCTION_COMMIT = "031bc43a9aee3d3c0c241ebd72cafef7a3ee5c38"
PRODUCTION_MODULE_SHA256 = "5969446e49218fa5067c11e17dbfbb7b6c22713670b5f4b64388dac9cc945971"
PRODUCTION_ATTENTION_SHA256 = "1dbfc2c1efb4012cc9bb801a89c2dbe355e4539a1c54599edc6e8a9a7db73087"
PRODUCTION_IMPORT = "from experiments.m2_03_targeted.observer import observe as _m203_observe\n"
PRODUCTION_OBSERVATION = (
    '    _m203_observe("production_attention_dispatch", max_seqlen_q, seq_threshold_3D,\n'
    '                  num_par_softmax_segments, num_seqs,\n'
    '                  softmax_segm_output is not None, softmax_segm_max is not None,\n'
    '                  softmax_segm_expsum is not None, is_batch_invariant, use_3d,\n'
    '                  grid, tile_size, BLOCK_M, BLOCK_Q, block_size, num_segments,\n'
    '                  short_query_3d, short_query_3d)\n')
IMPORT = ("from experiments.m2_03_targeted.observer import observe as _m203_observe, "
          "control as _m203_control\n")

# Exact insertion anchors from official v0.30.0. Whole-file SHA pin is mandatory.
# Guard arithmetic: upstream PR #59448 (non-DiffKV), also discussed in
# #59085/#56148. Only its empty-segment epilogue and reducer clamp are ported;
# the R16 full-attention gate excludes sliding-window/padding generalization.
INSERTIONS = {
    ATTENTION_SOURCE: [
        ("):\n    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.\n",
         "    _M203_Q3_GUARDS: tl.constexpr = False,\n"),
        ("):\n    query_token_idx = tl.program_id(0)\n",
         "    _M203_Q3_GUARDS: tl.constexpr = False,\n"),
        ("        store_segm_reduce_scalars(\n",
         '        if _M203_Q3_GUARDS:\n'
         '            M = tl.where(L > 0.0, M, float("-inf"))\n'),
        ("    # load and rescale segment exp sums\n",
         '    if _M203_Q3_GUARDS:\n'
         '        overall_max = tl.where(overall_max > float("-inf"), overall_max, 0.0)\n\n'),
        ("    # The kernel signature is the same for 2D and 3D — only the launch\n",
         '    _m203_q3_guards = False\n'
         '    if _m203_control("force_attention_3d", max_seqlen_q, num_seqs, use_3d):\n'
         '        assert q.shape[0] == max_seqlen_q == 3 and max_seqlen_k == 559\n'
         '        assert num_seqs == 1 and causal is True and window_size == (-1, -1)\n'
         '        assert not use_mm_prefix and not use_rswa and not use_td\n'
         '        assert not is_batch_invariant and kv_quant_mode == KVQuantMode.NONE\n'
         '        assert num_par_softmax_segments is not None and num_par_softmax_segments > 0\n'
         '        assert softmax_segm_output is not None and softmax_segm_output.shape[0] >= q.shape[0]\n'
         '        assert softmax_segm_max is not None and softmax_segm_max.shape[0] >= q.shape[0]\n'
         '        assert softmax_segm_expsum is not None and softmax_segm_expsum.shape[0] >= q.shape[0]\n'
         '        use_3d = True\n'
         '        _m203_q3_guards = True\n\n'),
        ("    kernel_unified_attention[grid](\n",
         '    _m203_observe("attention_dispatch", max_seqlen_q, seq_threshold_3D,\n'
         '                  num_par_softmax_segments, num_seqs,\n'
         '                  softmax_segm_output is not None, softmax_segm_max is not None,\n'
         '                  softmax_segm_expsum is not None, is_batch_invariant, use_3d,\n'
         '                  grid, tile_size, BLOCK_M, BLOCK_Q, block_size, num_segments,\n'
         '                  _m203_q3_guards)\n'),
        ("        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,\n",
         "        _M203_Q3_GUARDS=_m203_q3_guards,\n"),
        ("            USE_FP8=output_scale is not None,\n",
         "            _M203_Q3_GUARDS=_m203_q3_guards,\n"),
    ],
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


def production_attention_source():
    from src.server import triton_spec_decode_patch as production
    need(sha(Path(production.__file__).read_bytes()) == PRODUCTION_MODULE_SHA256,
         'production patch module differs from ' + PRODUCTION_COMMIT)
    raw = fixture_sources()[ATTENTION_SOURCE].encode()
    result = production.transform(raw)
    need(sha(result) == PRODUCTION_ATTENTION_SHA256, 'production attention source drift')
    return result


def profile_options(config):
    enabled = config.get('production_attention_observation', False)
    need(type(enabled) is bool, 'invalid production observation profile')
    if enabled:
        need(config.get('production_patch_sha') == PRODUCTION_COMMIT,
             'unreviewed production patch revision')
        need(not any(config.get(k) for k in (
            'r14_single_token_boundary', 'r15_attention_dispatch', 'r16_force_attention_3d')),
            'production observation cannot use diagnostic attention controls')
    else:
        need('production_patch_sha' not in config, 'production revision without profile')
    return {'production': True} if enabled else {}


def base_sources(production=False):
    result = {name: text.encode() for name, text in fixture_sources().items()}
    if production:
        result[ATTENTION_SOURCE] = production_attention_source()
    return result


def base_hashes(production=False):
    return {name: sha(raw) for name, raw in base_sources(production).items()}


def observe_production(raw):
    need(raw == production_attention_source(), 'exact production attention source required')
    anchor = '    kernel_unified_attention[grid](\n'
    text = raw.decode()
    need(text.count(anchor) == 1, 'production launch anchor drift')
    patched = (PRODUCTION_IMPORT + text.replace(anchor, PRODUCTION_OBSERVATION + anchor, 1)).encode()
    compile(patched, ATTENTION_SOURCE, 'exec')
    stripped = StripObserver().visit(ast.parse(patched))
    need(ast.dump(stripped) == ast.dump(ast.parse(raw)), 'production observer changed inference AST')
    return patched


class StripObserver(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == "experiments.m2_03_targeted.observer" else node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("_m203_observe", "_m203_control")):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == '_M203_Q3_GUARDS':
            return None
        if (isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                and node.test.func.id == "_m203_control"
                and node.test.args and isinstance(node.test.args[0], ast.Constant)
                and node.test.args[0].value == "force_attention_3d"):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == '_m203_q3_guards'):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        for index, argument in enumerate(node.args.args):
            if argument.arg == '_M203_Q3_GUARDS':
                start = len(node.args.args) - len(node.args.defaults)
                need(index >= start, 'diagnostic constexpr must have a default')
                node.args.args.pop(index)
                node.args.defaults.pop(index - start)
                break
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg != '_M203_Q3_GUARDS']
        return node


def transform(name, raw):
    if name == ATTENTION_SOURCE and sha(raw) == PRODUCTION_ATTENTION_SHA256:
        return observe_production(raw)
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


def verify_installed(root, patched=False, production=False):
    hashes = {}
    references = base_sources(production)
    for name, expected in manifest()["sha256"].items():
        raw = (Path(root) / name).read_bytes()
        need(sha(fixture_sources()[name].encode()) == expected, "vendored source fixture drift")
        reference = references[name]
        wanted = transform(name, reference) if patched else reference
        need(raw == wanted, "installed source mismatch: " + name)
        hashes[name] = sha(raw)
    return hashes


def installed_state(root, production=False):
    """Read-only classification; a mixed/unknown tree is never inference-ready."""
    files, states = {}, set()
    for name, raw in base_sources(production).items():
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
