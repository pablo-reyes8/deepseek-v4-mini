from __future__ import annotations

from typing import Any, Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from parallel.parallel_config import ParallelConfig
from parallel.parallel_utils import (
    aggregate_stats_mean,
    get_rank,
    get_world_size,
    is_dist_available_and_initialized,
    setup_distributed,
    unwrap_model,
)
from training.eval_one_epoch import eval_one_epoch
from training.train_one_epoch import train_one_epoch


def wrap_ddp_model(
    model: torch.nn.Module,
    config: ParallelConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Move model to device and wrap it with DistributedDataParallel when initialized."""
    config.validate()
    model.to(device)

    if not is_dist_available_and_initialized():
        return model

    local_rank = int(torch.cuda.current_device()) if device.type == "cuda" else None
    return DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        find_unused_parameters=config.find_unused_parameters,
        gradient_as_bucket_view=config.gradient_as_bucket_view,
        broadcast_buffers=config.broadcast_buffers,
        static_graph=config.static_graph,
    )


def build_distributed_sampler(
    dataset,
    config: Optional[ParallelConfig] = None,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DistributedSampler:
    seed = config.seed if config is not None else 42
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def build_ddp_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    config: ParallelConfig,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    sampler = build_distributed_sampler(
        dataset,
        config=config,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def ddp_train_one_epoch(
    *,
    model: torch.nn.Module,
    dataloader,
    optimizer: Any,
    device: torch.device,
    precision: dict[str, Any],
    parallel_config: ParallelConfig,
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    **train_kwargs,
) -> tuple[dict[str, float], int]:
    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    stats, global_step = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        device=device,
        precision=precision,
        epoch=epoch,
        global_step=global_step,
        is_main_process=get_rank() == 0,
        **train_kwargs,
    )
    return aggregate_stats_mean(stats, device=device), global_step


def ddp_evaluate(
    *,
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    precision: dict[str, Any],
    **eval_kwargs,
) -> dict[str, float]:
    stats = eval_one_epoch(
        model=model,
        dataloader=dataloader,
        device=device,
        precision=precision,
        is_main_process=get_rank() == 0,
        **eval_kwargs,
    )
    return aggregate_stats_mean(stats, device=device)


def get_state_dict_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return unwrap_model(model).state_dict()


def should_save_checkpoint(config: ParallelConfig) -> bool:
    return (not config.save_rank0_only) or get_rank() == 0


def launch_ddp_training(*args, **kwargs):
    """Placeholder for torchrun-driven scripts.

    This project intentionally does not spawn subprocesses from inside Python yet.
    Use torchrun to launch a script that calls setup_distributed + wrap_ddp_model.
    """
    raise NotImplementedError("Use torchrun with setup_distributed/wrap_ddp_model for DDP v1.")


def setup_ddp_device(config: ParallelConfig) -> torch.device:
    return setup_distributed(config)
