from __future__ import annotations

from typing import Optional, Protocol

import torch


class LayerCacheProtocol(Protocol):
    def reset(self) -> None: ...

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "LayerCacheProtocol": ...

    def num_tokens_seen(self) -> int: ...

    def memory_bytes(self) -> int: ...
