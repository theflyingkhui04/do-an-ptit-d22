"""CPU-safe extended source gate. No runtime or dependency changes."""
import json
from pathlib import Path

from experiments.m4_cache_probe_hooks import verify_source_tree


def verify_internal_sources():
    import vllm
    manifest = json.loads(Path(__file__).with_name('m4_layer0_probe_sources.json').read_text())
    return verify_source_tree(Path(vllm.__file__).resolve().parent.parent,manifest)


def selected_implementations(model):
    """Fail closed on an unreviewed selection, without forcing another backend."""
    layer = model.layers[0]
    attention = layer.self_attn
    records = {}
    for name,projection in (('qkv',attention.qkv_proj),('o_proj',attention.o_proj),
                            ('gate_up',layer.mlp.gate_up_proj),('down',layer.mlp.down_proj)):
        method = projection.quant_method
        kernel = getattr(method,'kernel',None)
        if (type(method).__module__ != 'vllm.model_executor.layers.quantization.auto_awq'
                or type(method).__name__ != 'AutoAWQMarlinLinearMethod'
                or type(kernel).__module__ != 'vllm.model_executor.kernels.linear.mixed_precision.marlin'
                or type(kernel).__name__ != 'MarlinLinearKernel'):
            raise RuntimeError('Unreviewed layer-0 quantized implementation')
        records[name] = dict(method=type(method).__module__+'.'+type(method).__name__,
                            kernel=type(kernel).__module__+'.'+type(kernel).__name__)
    backend = attention.attn.attn_backend.get_name()
    if backend != 'FLASH_ATTN':
        raise RuntimeError('Unreviewed layer-0 attention backend')
    for obj,module,name in (
        (attention.attn.impl,'vllm.v1.attention.backends.flash_attn','FlashAttentionImpl'),
        (attention.rotary_emb,'vllm.model_executor.layers.rotary_embedding.base','RotaryEmbedding'),
        (layer.input_layernorm,'vllm.model_executor.layers.layernorm','RMSNorm'),
        (layer.post_attention_layernorm,'vllm.model_executor.layers.layernorm','RMSNorm'),
        (layer.mlp.act_fn,'vllm.model_executor.layers.activation','SiluAndMul')):
        if type(obj).__module__ != module or type(obj).__name__ != name:
            raise RuntimeError('Unreviewed layer-0 operator implementation')
    records['attention_backend'] = backend
    records['attention_impl'] = type(attention.attn.impl).__module__+'.'+type(attention.attn.impl).__name__
    records['rope'] = type(attention.rotary_emb).__module__+'.'+type(attention.rotary_emb).__name__
    return records
