from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from inference.cache_utils import concat_optional, crop_last, move_optional, tensors_memory_bytes


@dataclass
class MHACache:
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    tokens_seen: int = 0

    def append(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> "MHACache":
        if k_new.dim() != 4 or v_new.dim() != 4:
            raise ValueError(
                f"MHA cache expects k/v [B,H,T,Dh], got {tuple(k_new.shape)} and {tuple(v_new.shape)}"
            )
        if k_new.shape != v_new.shape:
            raise ValueError(f"k_new and v_new shapes must match, got {k_new.shape} and {v_new.shape}")

        self.k = concat_optional(self.k, k_new, dim=2)
        self.v = concat_optional(self.v, v_new, dim=2)
        if position_ids is not None:
            self.positions = concat_optional(self.positions, position_ids, dim=-1)
        self.tokens_seen += int(k_new.shape[2])
        return self

    def get_kv(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.k, self.v

    def crop(self, max_length: Optional[int]) -> "MHACache":
        self.k = crop_last(self.k, max_length, dim=2)
        self.v = crop_last(self.v, max_length, dim=2)
        self.positions = crop_last(self.positions, max_length, dim=-1)
        return self

    def reset(self) -> None:
        self.k = None
        self.v = None
        self.positions = None
        self.tokens_seen = 0

    def num_tokens_seen(self) -> int:
        return int(self.tokens_seen)

    def memory_bytes(self) -> int:
        return tensors_memory_bytes(self.k, self.v, self.positions)

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MHACache":
        self.k = move_optional(self.k, device=device, dtype=dtype)
        self.v = move_optional(self.v, device=device, dtype=dtype)
        self.positions = move_optional(self.positions, device=device, dtype=None)
        return self
