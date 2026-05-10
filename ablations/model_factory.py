from __future__ import annotations

from typing import Any

import torch

from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig
from src.transformer_modules.transformer import MiniCausalLM, MiniCausalLMConfig


def _deepseek_config(cfg: dict[str, Any]) -> DeepSeekV4LMConfig:
    clean = dict(cfg)
    clean.pop("model_class", None)
    if isinstance(clean.get("attention_pattern"), list):
        clean["attention_pattern"] = tuple(clean["attention_pattern"])
    if isinstance(clean.get("mtp_depth_loss_weights"), list):
        clean["mtp_depth_loss_weights"] = tuple(clean["mtp_depth_loss_weights"])
    return DeepSeekV4LMConfig(**clean)


def _mini_transformer_config(cfg: dict[str, Any]) -> MiniCausalLMConfig:
    return MiniCausalLMConfig(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        max_seq_len=int(cfg["max_seq_len"]),
        pad_token_id=int(cfg.get("pad_token_id", 0)),
        embedding_dropout=float(cfg.get("embedding_dropout", 0.0)),
        n_heads=int(cfg["n_heads"]),
        head_dim=int(cfg["head_dim"]),
        attention_dropout=float(cfg.get("attention_dropout", 0.0)),
        residual_dropout=float(cfg.get("residual_dropout", 0.0)),
        use_attention_bias=bool(cfg.get("use_attention_bias", False)),
        use_rope=bool(cfg.get("use_rope", True)),
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        rotary_dim=int(cfg.get("rotary_dim", cfg["head_dim"])),
        mlp_hidden_dim=int(cfg.get("mlp_hidden_dim") or cfg["d_model"] * 4),
        mlp_dropout=float(cfg.get("mlp_dropout", 0.0)),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        use_mlp_bias=bool(cfg.get("use_mlp_bias", False)),
    )


def build_model_from_ablation_config(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = dict(config["model_config"])
    model_class = str(model_cfg.get("model_class", "deepseek"))
    if model_class == "mini_transformer":
        return MiniCausalLM(_mini_transformer_config(model_cfg))
    if model_class == "deepseek":
        return DeepSeekV4LM(_deepseek_config(model_cfg))
    raise ValueError(f"Unsupported ablation model_class={model_class!r}")


def count_parameters(model: torch.nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "num_parameters_total": float(total),
        "num_parameters_trainable": float(trainable),
        "num_parameters_active_estimate": float(_estimate_active_parameters(model)),
    }


def _estimate_active_parameters(model: torch.nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    cfg = getattr(model, "config", None)
    if cfg is None or getattr(cfg, "ffn_type", "dense") != "moe":
        return total

    inactive = 0
    top_k = int(getattr(cfg, "top_k_experts", 1) or 1)
    num_experts = int(getattr(cfg, "num_experts", 1) or 1)
    inactive_fraction = max(0.0, 1.0 - min(top_k, num_experts) / max(1, num_experts))
    for name, param in model.named_parameters():
        if ".experts." in name:
            inactive += int(param.numel() * inactive_fraction)
    return max(0, total - inactive)
