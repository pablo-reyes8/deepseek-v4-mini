# DeepSeek-V4 Mini Parallelism

This folder implements educational PyTorch parallelism for DeepSeek-V4 Mini.

It approximates the spirit of DeepSeek-V4 scaling with standard PyTorch primitives:

- DDP data parallelism.
- Layerwise model parallelism.
- Distributed-safe scalar metric aggregation.
- Rank-aware checkpoint save decisions.

It intentionally does not implement:

- custom CUDA kernels,
- FP4/FP8 kernels,
- true tensor parallelism,
- production all-to-all expert parallelism,
- heterogeneous on-disk KV cache,
- DualPipe pipeline scheduling,
- topology-aware communication overlap.

## Files

```text
parallel/
├── parallel_config.py   # shared config and validation
├── parallel_utils.py    # rank/world-size/device/reduction helpers
├── data_parallel.py     # DDP wrapping, samplers, train/eval adapters
└── model_parallel.py    # layerwise/blockwise educational model-parallel wrapper
```

## Recommended Order

1. Run normal single-device training.
2. Run DDP with `torchrun` and `backend=gloo` or `nccl`.
3. Run layerwise model parallel on one device to verify compatibility.
4. Run layerwise model parallel on two GPUs if available.
5. Treat hybrid DDP + model parallel as future work.

## CPU-Testable Surface

The current tests intentionally cover only CPU-safe behavior:

- config validation,
- rank/world-size helpers without initialized distributed state,
- scalar reductions in single process,
- distributed sampler construction,
- DDP wrapping under CPU gloo world-size 1,
- model-parallel wrapper on a single CPU device,
- block-device mapping and balance logic.

Real multi-process DDP and multi-GPU model parallelism should be validated manually with `torchrun` or GPU CI when available.
