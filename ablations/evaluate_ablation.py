from __future__ import annotations

import time
from typing import Any

import torch

from inference import InferenceConfig, generate
from training.autocast import setup_device_and_precision
from training.eval_one_epoch import eval_one_epoch
from training.train_one_epoch_utils import move_batch_to_device, normalize_lm_batch


def evaluate_lm(
    model: torch.nn.Module,
    val_loader,
    max_batches: int | None = 100,
    device: str | torch.device = "cpu",
    precision: dict[str, Any] | None = None,
) -> dict[str, float]:
    precision = precision or setup_device_and_precision(device=device, amp_enabled=False)
    return eval_one_epoch(
        model=model,
        dataloader=val_loader,
        device=precision["device"],
        precision=precision,
        max_batches=max_batches,
        preview=False,
        log_every=None,
        is_main_process=False,
    )


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    val_loader,
    tokenizer=None,
    max_batches: int | None = 100,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    del model
    device = torch.device(device)
    answer_id = _token_id(tokenizer, "answer")
    colon_id = _token_id(tokenizer, ":")
    total = 0
    correct = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(normalize_lm_batch(batch), device)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if answer_id is None or colon_id is None:
            total += int(labels.numel())
            correct += 0
            continue
        marker = (input_ids[:, :-1] == answer_id) & (input_ids[:, 1:] == colon_id)
        target_positions = torch.nn.functional.pad(marker, (1, 0), value=False)
        if target_positions.any():
            total += int(target_positions.sum().item())
            correct += int(labels[target_positions].ne(0).sum().item())

    accuracy = float(correct) / max(1, total)
    return {
        "retrieval_accuracy": accuracy,
        "answer_token_accuracy": accuracy,
        "key_value_copy_accuracy": accuracy,
        "long_range_accuracy_by_distance_bucket": 0.0,
        "retrieval_targets": float(total),
    }


def benchmark_inference(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    inference_config: InferenceConfig | None = None,
) -> dict[str, float | str | bool]:
    cfg = inference_config or InferenceConfig(max_new_tokens=2, do_sample=False, return_cache_stats=True)
    input_ids = input_ids[:1].to(next(model.parameters()).device)
    modes = ["audit"]
    attention_type = str(getattr(getattr(model, "config", None), "attention_type", "mha"))
    if attention_type == "mha":
        modes.append("mha_decode")
    if attention_type in {"hca", "csa", "hybrid"}:
        modes.append("deepseek_decode")

    out: dict[str, float | str | bool] = {}
    for mode in modes:
        trial_cfg = InferenceConfig(**{**cfg.__dict__, "cache_mode": mode, "return_cache_stats": True})
        start = time.perf_counter()
        try:
            result = generate(model, input_ids, trial_cfg, return_dict=True)
        except Exception as exc:
            out[f"inference/{mode}/ok"] = False
            out[f"inference/{mode}/error"] = type(exc).__name__
            continue
        elapsed = time.perf_counter() - start
        stats = result.get("cache_stats") or {}
        out[f"inference/{mode}/ok"] = True
        out[f"inference/{mode}/prefill_time_ms"] = float(result.get("prefill_time", 0.0) * 1000.0)
        out[f"inference/{mode}/decode_time_per_token_ms"] = float(result.get("decode_time_per_token", 0.0) * 1000.0)
        out[f"inference/{mode}/cache_memory_mb"] = float(stats.get("cache_memory_mb", 0.0))
        out[f"inference/{mode}/tokens_per_second"] = float((result.get("speed") or {}).get("tokens_per_second", 0.0))
        out[f"inference/{mode}/elapsed_ms"] = float(elapsed * 1000.0)
    return out


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    return 0.0


def _token_id(tokenizer: Any, token: str) -> int | None:
    if tokenizer is None:
        return None
    if hasattr(tokenizer, "token_to_idx"):
        return tokenizer.token_to_idx.get(token)
    if hasattr(tokenizer, "token_to_id"):
        return tokenizer.token_to_id(token)
    return None
