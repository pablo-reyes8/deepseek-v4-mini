from __future__ import annotations

from typing import Any, Optional

import torch

from inference.inference_config import InferenceConfig


@torch.no_grad()
def mtp_draft_from_hidden(
    model: torch.nn.Module,
    hidden_states: Optional[torch.Tensor],
    inference_config: Optional[InferenceConfig] = None,
) -> dict[str, Any]:
    cfg = inference_config or InferenceConfig()
    cfg.validate()

    mtp_head = getattr(model, "mtp_head", None)
    if hidden_states is None or mtp_head is None or not getattr(model, "use_mtp", False):
        return {
            "enabled": False,
            "mtp_logits": None,
            "draft_token_ids": None,
        }

    outputs = mtp_head(hidden_states[:, -1:, :], mtp_labels=None, return_aux=False)
    logits = outputs["mtp_logits"]
    max_tokens = cfg.max_mtp_draft_tokens or logits.shape[1]
    max_tokens = min(max_tokens, logits.shape[1])
    draft_ids = torch.argmax(logits[:, :max_tokens, -1, :], dim=-1)

    return {
        "enabled": bool(cfg.use_mtp_draft),
        "accept_mode": cfg.mtp_accept_mode,
        "mtp_logits": logits,
        "draft_token_ids": draft_ids,
    }
