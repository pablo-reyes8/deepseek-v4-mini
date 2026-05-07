from __future__ import annotations

import time
from typing import Any, Optional

import torch

from inference.inference_config import InferenceConfig
from inference.prefill import prefill


def cache_memory_bytes(cache: Any) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "memory_bytes"):
        return int(cache.memory_bytes())
    return 0


def cache_summary(cache: Any) -> dict[str, Any]:
    if cache is None:
        return {}
    if hasattr(cache, "cache_summary"):
        return cache.cache_summary()
    return {"cache_memory_bytes": cache_memory_bytes(cache)}


def generation_speed_metrics(start_time: float, end_time: float, num_tokens: int) -> dict[str, float]:
    elapsed = max(float(end_time) - float(start_time), 0.0)
    return {
        "elapsed_seconds": elapsed,
        "num_generated_tokens": float(num_tokens),
        "tokens_per_second": float(num_tokens) / elapsed if elapsed > 0 else 0.0,
    }


@torch.no_grad()
def compare_full_vs_cached_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    inference_config: Optional[InferenceConfig] = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    cfg = inference_config or InferenceConfig(do_sample=False)
    cfg.validate()
    was_training = model.training
    model.eval()

    full = model(input_ids=input_ids)["logits"]
    cached_positions = []
    max_diff = 0.0
    cache = None
    try:
        for idx in range(input_ids.shape[1]):
            out = prefill(
                model,
                input_ids=input_ids[:, : idx + 1],
                inference_config=cfg,
                return_aux=False,
            )
            cache = out["cache"]
            cached = out["logits"][:, -1, :]
            target = full[:, idx, :]
            diff = torch.max(torch.abs(cached - target)).item()
            max_diff = max(max_diff, float(diff))
            cached_positions.append(bool(torch.allclose(cached, target, atol=atol, rtol=rtol)))
    finally:
        if was_training:
            model.train()

    return {
        "allclose": all(cached_positions),
        "per_position_allclose": cached_positions,
        "max_abs_diff": max_diff,
        "atol": atol,
        "rtol": rtol,
        "cache_summary": cache_summary(cache),
    }


def now() -> float:
    return time.perf_counter()
