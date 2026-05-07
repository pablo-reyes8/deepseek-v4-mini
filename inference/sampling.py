from __future__ import annotations

import torch

from inference.inference_config import InferenceConfig


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float | None,
) -> torch.Tensor:
    if penalty is None or penalty == 1.0:
        return logits
    if penalty <= 0:
        raise ValueError(f"penalty must be > 0, got {penalty}")

    logits = logits.clone()
    for batch_idx in range(logits.shape[0]):
        seen = torch.unique(generated_ids[batch_idx].to(device=logits.device, dtype=torch.long))
        token_logits = logits[batch_idx, seen]
        logits[batch_idx, seen] = torch.where(token_logits < 0, token_logits * penalty, token_logits / penalty)
    return logits


def top_k_filtering(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(int(top_k), logits.shape[-1])
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def top_p_filtering(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    filtered_sorted = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.empty_like(logits)
    filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)
    return filtered


def _safe_argmax(logits: torch.Tensor) -> torch.Tensor:
    clean_logits = torch.nan_to_num(logits, nan=float("-inf"), posinf=1e30, neginf=-1e30)
    return torch.argmax(clean_logits, dim=-1, keepdim=True)


def sample_next_token(
    logits: torch.Tensor,
    config: InferenceConfig,
    generated_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    config.validate()
    if logits.dim() != 2:
        raise ValueError(f"sample_next_token expects logits [B,V], got {tuple(logits.shape)}")

    if not torch.isfinite(logits).all():
        return _safe_argmax(logits)

    if generated_ids is not None:
        logits = apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)

    if not config.do_sample:
        return _safe_argmax(logits)

    logits = logits / config.temperature
    logits = top_k_filtering(logits, config.top_k)
    logits = top_p_filtering(logits, config.top_p)

    probs = torch.softmax(logits, dim=-1)
    valid = torch.isfinite(probs).all(dim=-1) & (probs.sum(dim=-1) > 0)
    if not bool(valid.all().item()):
        return _safe_argmax(logits)

    return torch.multinomial(probs, num_samples=1)
