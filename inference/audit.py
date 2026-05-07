from __future__ import annotations

from typing import Any, Optional

import torch

from inference.cache_utils import decode_token_ids, encode_prompt
from inference.generate import generate
from inference.inference_config import InferenceConfig
from inference.metrics import compare_full_vs_cached_logits


@torch.no_grad()
def audit_inference_pipeline(
    model: torch.nn.Module,
    prompt: str | list[int] | torch.Tensor,
    tokenizer: Optional[Any] = None,
    compare_logits: bool = True,
    **generation_kwargs,
) -> dict[str, Any]:
    config = InferenceConfig(return_cache_stats=True, **generation_kwargs)
    input_ids = encode_prompt(prompt, tokenizer=tokenizer)

    generation = generate(
        model=model,
        input_ids=input_ids,
        inference_config=config,
        return_dict=True,
    )

    comparison = None
    if compare_logits and input_ids.shape[1] > 0:
        comparison = compare_full_vs_cached_logits(
            model=model,
            input_ids=input_ids.to(generation["sequences"].device),
            inference_config=config,
        )

    return {
        "input_ids": input_ids,
        "sequences": generation["sequences"],
        "text": decode_token_ids(generation["sequences"], tokenizer=tokenizer),
        "generation": generation,
        "cache_stats": generation.get("cache_stats"),
        "full_vs_cached": comparison,
        "config": config,
    }
