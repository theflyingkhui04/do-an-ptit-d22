"""Small, exact-source V2 observation insertions. Never imported as vLLM."""
import ast
import gzip
from functools import lru_cache
from .common import ROOT, decode, need, sha

RUNNER = 'v1/worker/gpu/model_runner.py'
GRAPHS = 'v1/worker/gpu/cudagraph_utils.py'
SAMPLER = 'v1/worker/gpu/sample/sampler.py'
REJECTION = 'v1/worker/gpu/spec_decode/rejection_sampler.py'
IMPORT = 'from experiments.m2_03_targeted.v2_observer import observe as _m203_v2\n'
INSERTIONS = {
    RUNNER: [
        ('        self.model_config = vllm_config.model_config\n',
         '        _m203_v2("initialized", self)\n'),
        # execute_model's authoritative flag, also used by InputBatch.make_dummy.
        ('        input_ids = input_batch.input_ids\n',
         '        _m203_v2("prepared", self, scheduler_output, input_batch, batch_desc, block_tables, slot_mappings, attn_metadata, dummy_run)\n'),
        ('        self.kv_connector.finish_forward()\n',
         '        _m203_v2("forward_done", self, input_batch, dummy_run)\n'),
        ('        if grammar_output is not None:\n',
         '        _m203_v2("raw", self, input_batch, logits, grammar_output)\n'),
        ('        # Postprocess results and update request states.\n',
         '        _m203_v2("selected", self, input_batch, sampler_output)\n'),
    ],
    GRAPHS: [
        ('            return forward_fn\n',
         '            _m203_v2("graph_inputs", self, desc, model_inputs, attn_metadata, warmup)\n'),
        ('                        with torch.cuda.graph(\n',
         '                        _m203_v2("graph_begin", self, desc, graph)\n'),
        ('                        compilation_counter.num_cudagraph_captured += 1\n',
         '                        _m203_v2("graph_end", self, desc, graph)\n'),
        ('        self.graphs[desc].replay()\n',
         '        _m203_v2("graph_replay", self, desc, self.graphs[desc])\n'),
    ],
    SAMPLER: [
        ('            sampled = gumbel_sample(\n',
         '            _m203_v2("processed", processed_logits, expanded_idx_mapping, pos)\n'),
    ],
    REJECTION: [
        ('        sampled, num_sampled = rejection_sample(\n',
         '        _m203_v2("processed", processed_logits, expanded_idx_mapping, pos)\n'),
    ],
}


@lru_cache(maxsize=1)
def _references(fixture_raw, manifest_raw):
    data = decode(gzip.decompress(fixture_raw))
    manifest = decode(manifest_raw)
    need(set(data) == set(manifest['sha256']), 'V2 fixture inventory drift')
    need(all(sha(text.encode()) == manifest['sha256'][name] for name, text in data.items()),
         'V2 source fixture checksum drift')
    return {name: text.encode() for name, text in data.items()}


def references():
    # Cache by immutable bytes, not path/mtime: a changed fixture is revalidated.
    return dict(_references((ROOT / 'tests/fixtures/m203_v030_v2_sources.json.gz').read_bytes(),
                            (ROOT / 'experiments/m2_03_targeted/upstream_v2.json').read_bytes()))


class Strip(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'experiments.m2_03_targeted.v2_observer' else node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_m203_v2':
            return None
        return self.generic_visit(node)


def transform(name, raw):
    need(raw == references()[name], 'V2 installed source checksum drift: ' + name)
    if name not in INSERTIONS:
        return raw
    text = raw.decode()
    for anchor, hook in INSERTIONS[name]:
        need(text.count(anchor) == 1, 'V2 hook anchor missing/ambiguous: ' + name + ': ' + anchor.strip())
        # forward_done must follow the forward, all other hooks precede anchors.
        replacement = anchor + hook if '"forward_done"' in hook else hook + anchor
        text = text.replace(anchor, replacement, 1)
    result = (IMPORT + text).encode()
    compile(result, name, 'exec')
    need(ast.dump(Strip().visit(ast.parse(result))) == ast.dump(ast.parse(raw)),
         'V2 observer changed inference AST')
    return result
