# Speculative method identification and serving compatibility

`flexsds inspect <model-id-or-path>` identifies a model without starting a GPU.
The serving wizard and flag-mode preflight use the same detector. Reports keep
architecture family, method label, evidence and FlexSDS support separate. An
unsupported DFlash checkpoint is reported as DFlash, not as an unknown/non-EAGLE
model. Unrecognized or contradictory metadata remains unverified.

| Family | Methods recognized when metadata is present |
| --- | --- |
| Independent draft model | Causal LM, `draft_model` |
| Target-attached prediction heads | Medusa, MLP speculator, Hydra, MTP |
| Autoregressive feature drafting | EAGLE-1, EAGLE-2, EAGLE-3, EAGLE-3.1 |
| Parallel/block drafting | P-EAGLE, DFlash, DSpark, PARD, XPress |
| Self-speculative decoding | LayerSkip/early exit, self-speculation |
| Non-neural lookup/retrieval | N-gram, prompt lookup, suffix, retrieval |
| Iterative neural decoding | Jacobi |

These are descriptive families, not an exhaustive partition of all algorithms.
Jacobi uses neural forwards and is not classified as non-neural. A REST proposer
has an unknown implementation; a REST address does not establish neural usage.
An EAGLE-4 declaration is displayed as an unverified variant, without assuming
its architecture or enabling a backend.

Methods without their own checkpoint must be selected explicitly, for example:

```sh
flexsds inspect --method ngram
flexsds inspect --method jacobi
```

A normal causal LM can participate in more than one inference scheme. Config
metadata and checkpoint tensor structure are used instead of repository-name
substrings. Inference-only policies cannot always be recovered from a model ID.
Known `speculators_config`, nested transformer layer configs, architecture names,
and dedicated head/MTP fields are understood. Declarations alone do not verify
that weights can run.

## EAGLE support

EAGLE-1/2 checkpoints route to vLLM `eagle`; EAGLE-3/3.1 route to `eagle3`.
Display labels keep EAGLE uppercase. A 2H feature projection identifies the shared
EAGLE-1/2 format; a 3H projection identifies the supported EAGLE-3 layout. The
EAGLE-3.1 variant also requires `fc_norm`, `norm_output`,
`use_post_norm_feedback` and matching normalization tensors. Parallel EAGLE
metadata/weights are identified separately and cannot enter the sequential path.

EAGLE-1 and EAGLE-2 can use the same weights. Without explicit version metadata,
the wizard asks the user to choose **EAGLE-1** or **EAGLE-2**, then displays that
exact label and saves the selected version. Noninteractive `inspect` and `serve`
require `--eagle-version 1` or `--eagle-version 2` for these shared checkpoints.
Explicit selection cannot override contradictory checkpoint metadata. The current vLLM `eagle`
path drafts sequentially; recognizing an EAGLE-2 checkpoint does not implement
its original confidence-adaptive tree algorithm. The runtime policy is stated
explicitly. Loader support is checked separately against the pinned vLLM
architectures: for example, legacy Qwen EAGLE heads can be identified while
lacking a registered legacy Qwen loader. No architecture conversion is assumed.

Both backend method names reach static serving and the adaptive command path.
Adaptive operation still requires a measured profile for that exact pair/method,
and its source gate checks the pinned legacy EAGLE loader/proposer code. Model
inspection and CPU command tests do not constitute GPU inference validation.

## Pairing, vocabulary and memory

Pairing requires declared base-model/verifier metadata and matching feature
input dimensions. Legacy repositories missing trained-target metadata may be
identified correctly while remaining blocked by pairing validation. Heads without
a tokenizer must identify their base model; its actual tokenizer is compared with
the target. Token IDs, merges, added tokens and processing are compared, rather
than only `vocab_size`.

Reduced EAGLE-3 output vocabularies require a valid native `d2t` mapping. Checks
cover integer offsets, uniqueness, bounds and consistency with `t2d` when present.
The EAGLE loader consumes this mapping without automatically enabling a generic
heterogeneous-vocabulary flag. Arbitrary different tokenizers remain unsupported.
Mapping inspection currently requires safetensors; convert a reduced-vocabulary
PyTorch checkpoint before serving through this guard.

VRAM estimates use checkpoint storage bytes, BF16/FP16 KV capacity at configured
concurrency/context and an estimated reserve of 1 GiB plus 10% of weights. Storage
precision can differ from runtime allocation. Remote memory budgets are estimates;
a successful inference request is still required for readiness.

Hub metadata is refreshed after five minutes. Files are cached by immutable
commit, and serving receives the inspected revisions. Adaptive profiles with other
revisions are rejected. Inspection reads bounded safetensors headers or PyTorch
ZIP metadata. The latter uses a restricted decoder that constructs inert tensor
metadata, rejects arbitrary globals and never loads tensor storage or model code.
HTTP servers ignoring byte-range requests are rejected.

References:
- [vLLM pinned speculation config](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/speculative.py)
- [EAGLE model families and shared weights](https://github.com/SafeAILab/EAGLE)
- [LayerSkip self-speculation](https://arxiv.org/abs/2404.16710)
- [Parallel drafting](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-28-speculators-parallel-drafting.md)
