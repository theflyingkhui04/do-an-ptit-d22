"""Replays a W1_V3 answer through xgrammar's builtin JSON grammar (vLLM json_object) and
applies vLLM 0.30 StructuredOutputGrammar.validate_tokens logic to drafts at the end.

Needs a separate environment with the Modal image's grammar library (no GPU):
    python -m venv xgv && xgv/Scripts/python -m pip install xgrammar==0.2.9 transformers tokenizers
    xgv/Scripts/python -I xg_term.py
Downloads the public Qwen/Qwen2.5-7B-Instruct tokenizer; reads raw.jsonl of flexsds-fast-a10-2611b.
Observed 2026-11-27: stop ids [151645]; 1072 tokens (+EOS = 1073); after the final '}' only
<|im_end|> is allowed; validate([<|im_end|>, x]) keeps 1 token; after EOS validate(...) -> [].
"""
import json, sys
import xgrammar as xgr
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=152064)
print("stop ids", info.stop_token_ids)
gram = xgr.GrammarCompiler(info).compile_builtin_json_grammar()
R = "D:/do-an-ptit-d22/experiments/runs/flexsds_bench/flexsds-fast-a10-2611b/attempts/flexsds-fast-a10-2611b-A10-adaptive-W1_V3-c32-a001/raw.jsonl"
rec = next(json.loads(l) for l in open(R, encoding="utf-8") if "req-05a67192" in l)
ids = tok.encode(rec["generated_text"], add_special_tokens=False)
print("re-tokenized", len(ids), "reported", rec["output_tokens"], "last tokens", [tok.decode([t]) for t in ids[-4:]])
EOS = tok.convert_tokens_to_ids("<|im_end|>")

def validate(m, tokens):  # vLLM 0.30 backend_xgrammar.validate_tokens
    if m.is_terminated():
        return []
    acc = []
    for t in tokens:
        if m.accept_token(t):
            acc.append(t)
            if m.is_terminated():
                break
        else:
            break
    if acc:
        m.rollback(len(acc))
    return acc

m = xgr.GrammarMatcher(gram, max_rollback_tokens=8)
for t in ids:
    assert m.accept_token(t)
print("after final '}': terminated =", m.is_terminated())
bm = xgr.allocate_token_bitmask(1, info.vocab_size)
m.fill_next_token_bitmask(bm)
allowed = [i for i in range(info.vocab_size) if (bm[0, i // 32] >> (i % 32)) & 1]
print("allowed next tokens after '}':", len(allowed), [tok.decode([a]) for a in allowed[:8]])
for drafts in ([EOS, EOS], [EOS, 198], [EOS, 151643]):
    v = validate(m, drafts)
    print("drafts", [tok.decode([d]) for d in drafts], "-> validated", len(v), "invalid", len(drafts) - len(v))
m.accept_token(EOS)
print("after EOS: terminated =", m.is_terminated(), "validate([x,y]) ->", validate(m, [198, 198]))
