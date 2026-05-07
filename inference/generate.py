from __future__ import annotations

from typing import Any, Optional

import torch

from inference.cache_utils import decode_token_ids, encode_prompt, resolve_device
from inference.decode import decode_step
from inference.inference_config import InferenceConfig
from inference.metrics import cache_summary, generation_speed_metrics, now
from inference.mtp_decode import mtp_draft_from_hidden
from inference.prefill import prefill
from inference.sampling import sample_next_token


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    inference_config: InferenceConfig,
    attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> dict[str, Any] | torch.Tensor:
    inference_config.validate()
    was_training = model.training
    model.eval()

    device = resolve_device(model, inference_config.device)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    start = now()
    prompt_len = int(input_ids.shape[1])
    prefill_start = now()
    prefill_out = prefill(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        inference_config=inference_config,
        return_aux=inference_config.use_mtp_draft or inference_config.return_cache_stats,
    )
    prefill_end = now()

    cache = prefill_out["cache"]
    sequences = input_ids
    logits = prefill_out["logits"][:, -1, :]
    scores: list[torch.Tensor] = []
    mtp_drafts: list[dict[str, Any]] = []
    finished = torch.zeros(input_ids.shape[0], device=device, dtype=torch.bool)

    hidden_states = prefill_out.get("hidden_states")
    if inference_config.use_mtp_draft:
        mtp_drafts.append(mtp_draft_from_hidden(model, hidden_states, inference_config))

    decode_time = 0.0
    for _ in range(inference_config.max_new_tokens):
        if inference_config.return_cache_stats:
            scores.append(logits.detach().cpu())

        next_token = sample_next_token(logits, inference_config, sequences)
        if inference_config.eos_token_id is not None:
            eos = int(inference_config.eos_token_id)
            if inference_config.pad_token_id is not None:
                pad = torch.full_like(next_token, int(inference_config.pad_token_id))
                next_token = torch.where(finished.unsqueeze(1), pad, next_token)
            finished = finished | next_token.squeeze(1).eq(eos)

        sequences = torch.cat([sequences, next_token], dim=1)
        if bool(finished.all().item()):
            break

        step_start = now()
        decoded = decode_step(
            model,
            input_ids_t=next_token,
            cache=cache,
            inference_config=inference_config,
            return_aux=inference_config.use_mtp_draft or inference_config.return_cache_stats,
        )
        decode_time += now() - step_start
        cache = decoded["cache"]
        logits = decoded["logits"][:, -1, :]
        hidden_states = decoded.get("hidden_states")
        if inference_config.use_mtp_draft:
            mtp_drafts.append(mtp_draft_from_hidden(model, hidden_states, inference_config))

    end = now()
    num_generated = int(sequences.shape[1] - prompt_len)
    output = {
        "sequences": sequences,
        "scores": scores if inference_config.return_cache_stats else None,
        "cache": cache,
        "cache_stats": cache_summary(cache) if inference_config.return_cache_stats else None,
        "num_generated_tokens": num_generated,
        "prompt_length": prompt_len,
        "prefill_time": prefill_end - prefill_start,
        "decode_time": decode_time,
        "decode_time_per_token": decode_time / max(num_generated, 1),
        "speed": generation_speed_metrics(start, end, num_generated),
        "mtp_drafts": mtp_drafts if inference_config.use_mtp_draft else None,
    }

    if was_training:
        model.train()

    if return_dict:
        return output
    return sequences


@torch.no_grad()
def inference_autoregresive(
    model: torch.nn.Module,
    prompt: str | list[int] | torch.Tensor,
    tokenizer: Optional[Any] = None,
    **generation_kwargs,
) -> dict[str, Any]:
    config = InferenceConfig(**generation_kwargs)
    input_ids = encode_prompt(prompt, tokenizer=tokenizer)
    output = generate(
        model=model,
        input_ids=input_ids,
        inference_config=config,
        return_dict=True,
    )
    output["text"] = decode_token_ids(output["sequences"], tokenizer=tokenizer)
    return output


inference_autoregressive = inference_autoregresive
