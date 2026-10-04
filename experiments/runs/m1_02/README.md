# M1-02 Canonical Benchmark Evidence

Benchmark source SHA:

f307e33f9aea4bbd45499f548480cb7d6f90a4b9

Canonical environment:

- GPU: Tesla T4
- visible CUDA devices: 1
- vLLM: 0.30.0
- PyTorch: 2.13.0+cu130
- CUDA: 13.0
- driver: 580.178.04
- aiohttp: 3.13.5
- nvidia-ml-py: 13.595.45

Protocol:

- lanes: Vanilla, Static
- workloads: W1, W2, W3
- concurrency: 1,2,4,8,16,32
- repeats: 5
- measured requests / repeat: 32
- sample policy: cycle-rotate

Only fully validated sweeps are committed here.
Partial sweeps must not be treated as canonical evidence.
