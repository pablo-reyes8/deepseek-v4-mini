from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelConfig:
    """Shared config for educational PyTorch parallelism."""

    mode: str = "none"  # "none", "ddp", "model_parallel", "hybrid"
    backend: str = "nccl"
    init_method: str = "env://"
    seed: int = 42

    # DDP
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = False
    static_graph: bool = False

    # Model parallel
    model_parallel_strategy: str = "layerwise"  # "layerwise", "blockwise", "moe_expert"
    devices: Optional[list[str]] = None
    balance: Optional[list[int]] = None

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bf16"

    # Checkpointing
    save_rank0_only: bool = True
    distributed_checkpoint: bool = False

    # Debugging
    debug: bool = False

    def validate(self) -> None:
        allowed_modes = {"none", "ddp", "model_parallel", "hybrid"}
        if self.mode not in allowed_modes:
            raise ValueError(f"mode must be one of {allowed_modes}, got {self.mode!r}")

        allowed_backends = {"nccl", "gloo"}
        if self.backend not in allowed_backends:
            raise ValueError(f"backend must be one of {allowed_backends}, got {self.backend!r}")

        allowed_strategies = {"layerwise", "blockwise", "moe_expert"}
        if self.model_parallel_strategy not in allowed_strategies:
            raise ValueError(
                "model_parallel_strategy must be one of "
                f"{allowed_strategies}, got {self.model_parallel_strategy!r}"
            )

        allowed_amp = {"bf16", "fp16", "fp32"}
        if self.amp_dtype not in allowed_amp:
            raise ValueError(f"amp_dtype must be one of {allowed_amp}, got {self.amp_dtype!r}")

        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

        if self.balance is not None and any(x <= 0 for x in self.balance):
            raise ValueError(f"balance entries must be > 0, got {self.balance}")

    @property
    def is_distributed(self) -> bool:
        return self.mode in {"ddp", "hybrid"}

    @property
    def is_model_parallel(self) -> bool:
        return self.mode in {"model_parallel", "hybrid"}
