from __future__ import annotations

from typing import Any, Optional

import torch


DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def resolve_cache_dtype(cache_dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(cache_dtype, torch.dtype):
        return cache_dtype
    if cache_dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported cache dtype {cache_dtype!r}")
    return DTYPE_MAP[cache_dtype]


def resolve_device(model: torch.nn.Module, device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_memory_bytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def tensors_memory_bytes(*tensors: Optional[torch.Tensor]) -> int:
    return sum(tensor_memory_bytes(tensor) for tensor in tensors)


def concat_optional(
    current: Optional[torch.Tensor],
    new: Optional[torch.Tensor],
    *,
    dim: int,
) -> Optional[torch.Tensor]:
    if new is None:
        return current
    if current is None:
        return new
    return torch.cat([current, new], dim=dim)


def crop_last(tensor: Optional[torch.Tensor], max_length: Optional[int], *, dim: int) -> Optional[torch.Tensor]:
    if tensor is None or max_length is None or tensor.shape[dim] <= max_length:
        return tensor
    index = [slice(None)] * tensor.dim()
    index[dim] = slice(tensor.shape[dim] - max_length, tensor.shape[dim])
    return tensor[tuple(index)]


def move_optional(
    tensor: Optional[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None and torch.is_floating_point(tensor):
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs)


def normalize_position_ids(
    position_ids: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    start_pos: int = 0,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is None:
        base = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.long)
        return base.unsqueeze(0).expand(batch_size, seq_len)

    position_ids = position_ids.to(device=device, dtype=torch.long)
    if position_ids.dim() == 1:
        if position_ids.numel() != seq_len:
            raise ValueError(
                f"position_ids length must be {seq_len}, got {position_ids.numel()}"
            )
        return position_ids.unsqueeze(0).expand(batch_size, seq_len)

    if position_ids.shape != (batch_size, seq_len):
        raise ValueError(
            f"position_ids must have shape {(batch_size, seq_len)} or [{seq_len}], "
            f"got {tuple(position_ids.shape)}"
        )
    return position_ids


def context_window_for_model(model: torch.nn.Module, max_cache_length: Optional[int] = None) -> int:
    model_max = int(getattr(getattr(model, "config", None), "max_seq_len", 0) or 0)
    if max_cache_length is None:
        return model_max if model_max > 0 else 2**30
    if model_max <= 0:
        return int(max_cache_length)
    return min(int(max_cache_length), model_max)


def encode_prompt(prompt: str | list[int] | torch.Tensor, tokenizer: Optional[Any] = None) -> torch.Tensor:
    if torch.is_tensor(prompt):
        ids = prompt.long()
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("A tokenizer is required when prompt is a string.")
        encoded = tokenizer.encode(prompt)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        return torch.tensor([list(encoded)], dtype=torch.long)

    return torch.tensor([list(prompt)], dtype=torch.long)


def decode_token_ids(token_ids: torch.Tensor, tokenizer: Optional[Any] = None) -> Optional[str | list[str]]:
    if tokenizer is None:
        return None
    ids = token_ids.detach().cpu().tolist()
    if token_ids.dim() == 2 and len(ids) == 1:
        ids = ids[0]
    elif token_ids.dim() == 2:
        return [decode_token_ids(row, tokenizer=tokenizer) or "" for row in token_ids]
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids)
    if hasattr(tokenizer, "idx_to_token"):
        return " ".join(tokenizer.idx_to_token.get(int(i), "<unk>") for i in ids)
    return None


def default_valid_mask(input_ids: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
    if pad_token_id is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids.ne(int(pad_token_id))


def token_hidden_state(
    model: torch.nn.Module,
    input_ids_t: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        batch, seq_len = input_ids_t.shape
        d_model = int(getattr(getattr(model, "config", None), "d_model", 1))
        return torch.zeros(batch, seq_len, d_model, device=device, dtype=dtype)
    hidden = embedding(input_ids_t.to(device=device, dtype=torch.long))
    if torch.is_floating_point(hidden):
        hidden = hidden.to(dtype=dtype)
    return hidden


def hidden_to_mha_kv(model: torch.nn.Module, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = getattr(model, "config", None)
    n_heads = int(getattr(cfg, "n_heads", 1) or 1)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = hidden.shape[-1] // n_heads
    head_dim = int(head_dim)
    needed = n_heads * head_dim
    if hidden.shape[-1] < needed:
        hidden = torch.nn.functional.pad(hidden, (0, needed - hidden.shape[-1]))
    elif hidden.shape[-1] > needed:
        hidden = hidden[..., :needed]
    kv = hidden.reshape(hidden.shape[0], hidden.shape[1], n_heads, head_dim).transpose(1, 2)
    return kv, kv


def hidden_to_index_state(hidden: torch.Tensor, indexer_dim: int) -> torch.Tensor:
    if hidden.shape[-1] < indexer_dim:
        return torch.nn.functional.pad(hidden, (0, indexer_dim - hidden.shape[-1]))
    return hidden[..., :indexer_dim]
