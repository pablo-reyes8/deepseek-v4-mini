from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist

from parallel.parallel_config import ParallelConfig


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def setup_distributed(config: ParallelConfig) -> torch.device:
    config.validate()

    if config.mode == "hybrid":
        raise NotImplementedError("Hybrid DDP + model parallelism is planned but not implemented in v1.")

    if config.mode == "ddp":
        if not is_dist_available_and_initialized():
            init_kwargs = {}
            if config.init_method != "env://":
                init_kwargs["rank"] = int(os.environ.get("RANK", 0))
                init_kwargs["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
            dist.init_process_group(
                backend=config.backend,
                init_method=config.init_method,
                **init_kwargs,
            )

        local_rank = get_local_rank()
        if torch.cuda.is_available() and config.backend == "nccl":
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")

    if config.devices:
        return torch.device(config.devices[0])

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def set_distributed_seed(seed: int, rank: Optional[int] = None) -> None:
    rank = get_rank() if rank is None else int(rank)
    full_seed = int(seed) + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def move_batch_to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking=non_blocking)
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device, non_blocking=non_blocking) for x in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(x, device, non_blocking=non_blocking) for x in batch]
    return batch


def _as_float_tensor(value: float | int | torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().float()
        return tensor.to(device=device) if device is not None else tensor
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def all_reduce_mean(value: float | int | torch.Tensor, device: Optional[torch.device] = None) -> float:
    tensor = _as_float_tensor(value, device)
    if is_dist_available_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / get_world_size()
    return float(tensor.item())


def all_reduce_sum(value: float | int | torch.Tensor, device: Optional[torch.device] = None) -> float:
    tensor = _as_float_tensor(value, device)
    if is_dist_available_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def gather_scalar(value: float | int | torch.Tensor, device: Optional[torch.device] = None) -> list[float]:
    tensor = _as_float_tensor(value, device)
    if not is_dist_available_and_initialized():
        return [float(tensor.item())]

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return [float(x.item()) for x in gathered]


def aggregate_stats_mean(stats: dict[str, Any], device: Optional[torch.device] = None) -> dict[str, Any]:
    reduced = dict(stats)
    for key, value in stats.items():
        if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
            reduced[key] = all_reduce_mean(value, device=device)
    return reduced


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
