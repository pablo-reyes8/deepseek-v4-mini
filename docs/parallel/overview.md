# Parallelism Guide

`parallel/` contains the repo's first PyTorch-native parallelism layer. It is intentionally explicit and educational: the code shows how data parallelism and coarse model placement fit around `DeepSeekV4LM` without depending on custom CUDA kernels or vendor-specific runtime features.

## What Is Implemented

### Data Parallelism

Data parallelism is implemented with standard `torch.distributed` and `DistributedDataParallel`.

Configurable knobs:

| Parameter | Role |
| --- | --- |
| `mode` | Use `ddp` to initialize distributed data parallel execution. |
| `backend` | `gloo` for CPU-safe tests and generic execution, `nccl` for CUDA multi-GPU runs. |
| `init_method` | Process-group rendezvous method. Use `env://` under `torchrun`, or `file://...` for one-process CPU smoke checks. |
| `find_unused_parameters` | Useful when optional branches are active and some parameters may not receive gradients in every step. |
| `gradient_as_bucket_view` | Lets DDP gradients share bucket memory when supported. |
| `broadcast_buffers` | Synchronizes buffers across ranks; usually not needed in this model. |
| `static_graph` | Enables DDP static graph optimizations when the forward graph is stable. |
| `save_rank0_only` | Keeps checkpoint writing on rank 0 by default. |

Main helpers:

- `setup_distributed(config)` initializes the process group and returns the local device.
- `wrap_ddp_model(model, config, device)` moves and wraps the model.
- `build_ddp_dataloader(...)` creates a dataloader with `DistributedSampler`.
- `ddp_train_one_epoch(...)` and `ddp_evaluate(...)` reuse the existing training/eval loops and aggregate scalar stats.

## Model Parallelism

Model parallelism is implemented as layerwise/blockwise placement. Whole Transformer blocks are assigned to devices, and activations are moved between block boundaries.

This is not tensor parallelism, expert all-to-all routing, pipeline scheduling, or the paper's production-grade parallel runtime. It is a transparent approximation that keeps the model structure intact.

Configurable knobs:

| Parameter | Role |
| --- | --- |
| `devices` | Ordered device list such as `cpu`, `cuda:0,cuda:1`, or any valid PyTorch device strings. |
| `balance` | Optional comma-separated layer counts per device, for example `2,2,4`. |
| `model_parallel_strategy` | Currently documents intent; the implemented path is layerwise/blockwise placement. |

Main helpers:

- `infer_auto_balance(n_layers, n_devices)` distributes layers as evenly as possible.
- `build_block_device_map(n_layers, devices, balance)` returns the per-layer placement plan.
- `ModelParallelDeepSeekV4LM(model, devices, balance)` wraps an existing `DeepSeekV4LM`.
- `wrap_model_parallel(...)` is the convenience constructor.

## CLI

Inspect a placement plan:

```bash
python -m scripts.parallel_cli plan --n-layers 6 --devices cpu,cpu --balance 2,4
```

Run a CPU-safe model-parallel smoke test:

```bash
python -m scripts.parallel_cli model-parallel-smoke --devices cpu --n-layers 2
```

Run a one-process CPU DDP smoke test:

```bash
python -m scripts.parallel_cli ddp-smoke --backend gloo --n-layers 1
```

Run the CPU-safe parallelism tests:

```bash
python -m scripts.parallel_cli tests --quiet
```

After editable install, the same commands are exposed as:

```bash
deepseekv4-parallel plan --n-layers 6 --devices cpu,cpu
deepseekv4-parallel model-parallel-smoke --devices cpu
deepseekv4-parallel ddp-smoke --backend gloo
```

## Testable Without CUDA

The repository tests only the parts that can be verified on CPU:

- Config validation.
- Single-process scalar aggregation.
- Distributed samplers in world size 1.
- One-process `gloo` DDP forward/backward.
- Model-parallel equivalence when all blocks are placed on CPU.
- mHC compatibility through the model-parallel wrapper.

Multi-GPU NCCL throughput, cross-device activation transfer costs, and true expert parallelism require CUDA hardware and are intentionally not claimed by these tests.
