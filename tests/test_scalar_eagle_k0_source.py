"""K0 semantics from the already hash-pinned EAGLE path; CPU/source only."""
import ast
import os
from pathlib import Path
from experiments.m4_pinned_contract import verify


def test_k0_pinned_eagle_forward_precedes_zero_width_return():
    root=Path(os.environ["M4_VLLM_SOURCE_ROOT"])
    assert verify(root)["verified_files"]==19
    path=root/"vllm/v1/spec_decode/llm_base_proposer.py"
    source=path.read_text(encoding="utf-8")
    tree=ast.parse(source)
    zero=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
              and "self.num_speculative_tokens == 0" == ast.unparse(n.test))
    returns=[n for n in ast.walk(zero) if isinstance(n,ast.Return)]
    assert returns and ast.unparse(returns[0].value.func)=="torch.empty"
    assert [ast.unparse(a) for a in returns[0].value.args]==["batch_size","0"]
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                  and any(x is zero for x in ast.walk(n)))
    assert any(isinstance(n,ast.Call) and ast.unparse(n.func).endswith("self.model")
               and n.lineno<zero.lineno for n in ast.walk(function))
    runner=(root/"vllm/v1/worker/gpu_model_runner.py").read_text(encoding="utf-8")
    assert 'self.drafter.propose' in runner
    assert 'speculative_config.use_eagle()' in runner
