from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


ABLATION_IDS = ("A1", "A2", "A3", "A4", "A5", "A6")


def _base_model(max_model: dict[str, Any] | None = None) -> dict[str, Any]:
    limits = {
        "vocab_size": 216,
        "d_model": 64,
        "n_layers": 4,
        "max_seq_len": 128,
        "n_heads": 4,
        "head_dim": 16,
        "rotary_dim": 16,
        "mlp_hidden_dim": 256,
        "expert_hidden_dim": 128,
        "shared_hidden_dim": 128,
        "num_experts": 4,
        "top_k_experts": 2,
        "n_hc": 4,
        "mtp_hidden_dim": 64,
    }
    if max_model:
        limits.update({k: v for k, v in max_model.items() if v is not None})

    d_model = int(limits["d_model"])
    n_heads = int(limits["n_heads"])
    head_dim = int(limits.get("head_dim") or max(1, d_model // max(1, n_heads)))
    rotary_dim = min(int(limits.get("rotary_dim") or head_dim), head_dim)
    return {
        "model_class": "deepseek",
        "vocab_size": int(limits["vocab_size"]),
        "d_model": d_model,
        "n_layers": int(limits["n_layers"]),
        "max_seq_len": int(limits["max_seq_len"]),
        "pad_token_id": 0,
        "embedding_dropout": 0.0,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
        "attention_type": "hybrid",
        "attention_pattern": ("csa", "hca"),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "compression_factor": 4,
        "hca_compression_factor": 4,
        "window_size": min(32, int(limits["max_seq_len"])),
        "top_k_blocks": 2,
        "indexer_dim": min(16, head_dim),
        "n_indexer_heads": 2,
        "query_compression_dim": min(16, d_model),
        "ffn_type": "dense",
        "mlp_hidden_dim": int(limits["mlp_hidden_dim"]),
        "num_experts": int(limits["num_experts"]),
        "top_k_experts": int(limits["top_k_experts"]),
        "expert_hidden_dim": int(limits["expert_hidden_dim"]),
        "shared_experts": 1,
        "shared_hidden_dim": int(limits["shared_hidden_dim"]),
        "balance_loss_weight": 0.0,
        "sequence_balance_loss_weight": 0.0,
        "router_jitter_noise": 0.0,
        "use_mhc": False,
        "n_hc": int(limits["n_hc"]),
        "mhc_sinkhorn_iters": 5,
        "use_mtp": False,
        "mtp_depth": 2,
        "mtp_hidden_dim": int(limits["mtp_hidden_dim"]),
        "mtp_loss_weight": 0.3,
    }


def _base_data(data_config: dict[str, Any] | None = None, quick: bool = False) -> dict[str, Any]:
    cfg = {
        "dataset": "synthetic_long_context",
        "block_size": 128 if quick else 256,
        "batch_size": 4 if quick else 8,
        "num_train_examples": 64 if quick else 20_000,
        "num_val_examples": 24 if quick else 2_000,
        "min_filler_tokens": 8 if quick else 64,
        "max_filler_tokens": 64 if quick else 420,
        "num_keys_per_example": 4 if quick else 8,
        "num_workers": 0,
        "seed": 42,
    }
    if data_config:
        cfg.update({k: v for k, v in data_config.items() if v is not None})
    return cfg


def _base_training(training_config: dict[str, Any] | None = None, quick: bool = False) -> dict[str, Any]:
    cfg = {
        "epochs": 1 if quick else 3,
        "max_batches_per_epoch": 2 if quick else 500,
        "eval_max_batches": 1 if quick else 100,
        "optimizer_type": "adamw" if quick else "muon_adamw",
        "learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "weight_decay": 0.1,
        "grad_clip_norm": 1.0,
        "grad_accum_steps": 1,
        "warmup_steps": 1,
        "device": "cpu" if quick else "auto",
        "amp_enabled": False if quick else True,
        "amp_dtype": "bf16",
        "log_every": 0,
        "module_metrics_every": None,
        "save_checkpoints": True,
        "run_inference_benchmark": True,
        "inference_max_new_tokens": 2 if quick else 16,
    }
    if training_config:
        cfg.update({k: v for k, v in training_config.items() if v is not None})
    return cfg


def _variant(name: str, overrides: dict[str, Any], note: str = "") -> dict[str, Any]:
    return {"variant_name": name, "model_overrides": overrides, "hypothesis_note": note}


def _variants_for(ablation_id: str) -> list[dict[str, Any]]:
    if ablation_id == "A1":
        return [
            _variant("dense_mha_baseline", {"attention_type": "mha", "ffn_type": "dense", "use_mhc": False, "use_mtp": False}),
            _variant("hca_only", {"attention_type": "hca", "ffn_type": "dense", "use_mhc": False, "use_mtp": False}),
            _variant("csa_only", {"attention_type": "csa", "ffn_type": "dense", "use_mhc": False, "use_mtp": False}),
            _variant("hybrid_csa_hca", {"attention_type": "hybrid", "attention_pattern": ("csa", "hca"), "ffn_type": "dense", "use_mhc": False, "use_mtp": False}),
            _variant("hybrid_hca_csa", {"attention_type": "hybrid", "attention_pattern": ("hca", "csa"), "ffn_type": "dense", "use_mhc": False, "use_mtp": False}),
        ]
    if ablation_id == "A2":
        return [
            _variant("hca_m2_w32", {"attention_type": "hca", "hca_compression_factor": 2, "window_size": 32}),
            _variant("hca_m4_w32", {"attention_type": "hca", "hca_compression_factor": 4, "window_size": 32}),
            _variant("hca_m8_w32", {"attention_type": "hca", "hca_compression_factor": 8, "window_size": 32}),
            _variant("hca_m4_w16", {"attention_type": "hca", "hca_compression_factor": 4, "window_size": 16}),
            _variant("hca_m4_w64", {"attention_type": "hca", "hca_compression_factor": 4, "window_size": 64}),
            _variant("csa_m2_w32_k1", {"attention_type": "csa", "compression_factor": 2, "window_size": 32, "top_k_blocks": 1}),
            _variant("csa_m4_w32_k1", {"attention_type": "csa", "compression_factor": 4, "window_size": 32, "top_k_blocks": 1}),
            _variant("csa_m8_w32_k1", {"attention_type": "csa", "compression_factor": 8, "window_size": 32, "top_k_blocks": 1}),
            _variant("csa_m4_w16_k1", {"attention_type": "csa", "compression_factor": 4, "window_size": 16, "top_k_blocks": 1}),
            _variant("csa_m4_w64_k1", {"attention_type": "csa", "compression_factor": 4, "window_size": 64, "top_k_blocks": 1}),
        ]
    if ablation_id == "A3":
        return [
            _variant("mha_no_mhc", {"attention_type": "mha", "use_mhc": False, "n_layers": 2, "max_seq_len": 128}),
            _variant("mha_mhc", {"attention_type": "mha", "use_mhc": True, "n_layers": 2, "max_seq_len": 128}),
            _variant("hybrid_no_mhc", {"attention_type": "hybrid", "attention_pattern": ("csa", "hca"), "use_mhc": False, "n_layers": 2, "max_seq_len": 128}),
            _variant("hybrid_mhc", {"attention_type": "hybrid", "attention_pattern": ("csa", "hca"), "use_mhc": True, "n_layers": 2, "max_seq_len": 128}),
            _variant("hybrid_no_mhc_deep", {"attention_type": "hybrid", "attention_pattern": ("csa", "hca"), "use_mhc": False, "n_layers": 6, "max_seq_len": 256}),
            _variant("hybrid_mhc_deep", {"attention_type": "hybrid", "attention_pattern": ("csa", "hca"), "use_mhc": True, "n_layers": 6, "max_seq_len": 256}),
        ]
    if ablation_id == "A4":
        return [
            _variant("dense_ffn_baseline", {"attention_type": "hybrid", "ffn_type": "dense"}),
            _variant("moe_no_shared", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 0, "balance_loss_weight": 0.01, "sequence_balance_loss_weight": 0.01}),
            _variant("moe_shared_1", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 1, "balance_loss_weight": 0.01, "sequence_balance_loss_weight": 0.01}),
            _variant("moe_shared_2", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 2, "balance_loss_weight": 0.01, "sequence_balance_loss_weight": 0.01}),
            _variant("moe_no_balance", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 1, "balance_loss_weight": 0.0, "sequence_balance_loss_weight": 0.0}),
            _variant("moe_hash_routing", {"attention_type": "hybrid", "ffn_type": "moe", "router_type": "hash", "shared_experts": 1}),
        ]
    if ablation_id == "A5":
        return [
            _variant("no_mtp", {"use_mtp": False}),
            _variant("mtp_depth_1_w0.1", {"use_mtp": True, "mtp_depth": 1, "mtp_loss_weight": 0.1}),
            _variant("mtp_depth_2_w0.3", {"use_mtp": True, "mtp_depth": 2, "mtp_loss_weight": 0.3}),
            _variant("mtp_depth_4_w0.3", {"use_mtp": True, "mtp_depth": 4, "mtp_loss_weight": 0.3}),
            _variant("mtp_depth_2_w1.0", {"use_mtp": True, "mtp_depth": 2, "mtp_loss_weight": 1.0}),
            _variant("mtp_depth_4_weighted", {"use_mtp": True, "mtp_depth": 4, "mtp_loss_weight": 0.3, "mtp_depth_loss_weights": (0.5, 0.25, 0.15, 0.10)}),
        ]
    if ablation_id == "A6":
        return [
            _variant("transformer_baseline", {"model_class": "mini_transformer", "attention_type": "mha", "ffn_type": "dense", "use_mhc": False, "use_mtp": False}),
            _variant("compressed_attention", {"attention_type": "hybrid", "attention_pattern": ("csa", "hca"), "ffn_type": "dense", "use_mhc": False, "use_mtp": False}),
            _variant("plus_moe", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 1, "balance_loss_weight": 0.01, "sequence_balance_loss_weight": 0.01, "use_mhc": False, "use_mtp": False}),
            _variant("plus_mhc", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 1, "use_mhc": True, "use_mtp": False}),
            _variant("plus_mtp", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 1, "use_mhc": True, "use_mtp": True}),
            _variant("full_minus_mhc", {"attention_type": "hybrid", "ffn_type": "moe", "shared_experts": 1, "use_mhc": False, "use_mtp": True}),
            _variant("full_minus_moe", {"attention_type": "hybrid", "ffn_type": "dense", "use_mhc": True, "use_mtp": True}),
        ]
    raise ValueError(f"Unknown ablation_id={ablation_id!r}. Expected one of {ABLATION_IDS} or 'ALL'.")


def build_ablation_suite(
    ablation_id: str,
    base_output_dir: str = "outputs/ablations",
    seeds: list[int] | None = None,
    quick: bool = False,
    data_config: dict[str, Any] | None = None,
    max_model: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    limit_variants: int | None = None,
) -> list[dict[str, Any]]:
    seeds = [1, 2, 3] if seeds is None else list(seeds)
    ablation_id = ablation_id.upper()
    if ablation_id == "ALL":
        suites: list[dict[str, Any]] = []
        for item in ABLATION_IDS:
            suites.extend(
                build_ablation_suite(
                    item,
                    base_output_dir=base_output_dir,
                    seeds=seeds,
                    quick=quick,
                    data_config=data_config,
                    max_model=max_model,
                    training_config=training_config,
                    limit_variants=limit_variants,
                )
            )
        return suites

    variants = _variants_for(ablation_id)
    if limit_variants is not None:
        variants = variants[: int(limit_variants)]

    runs: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            model_cfg = _base_model(max_model=max_model)
            model_cfg.update(deepcopy(variant["model_overrides"]))
            data_cfg = _base_data(data_config=data_config, quick=quick)
            train_cfg = _base_training(training_config=training_config, quick=quick)
            data_cfg["block_size"] = min(int(data_cfg["block_size"]), int(model_cfg["max_seq_len"]))
            data_cfg["seed"] = int(seed)
            if data_cfg.get("dataset", "synthetic_long_context") in {
                "synthetic",
                "synthetic_long_context",
                "synthetic_retrieval",
            }:
                model_cfg["vocab_size"] = _synthetic_vocab_size(data_cfg)

            variant_name = str(variant["variant_name"])
            out_dir = Path(base_output_dir) / ablation_id / variant_name / f"seed_{seed}"
            runs.append(
                {
                    "ablation_id": ablation_id,
                    "variant_name": variant_name,
                    "seed": int(seed),
                    "model_config": model_cfg,
                    "data_config": data_cfg,
                    "training_config": train_cfg,
                    "output_dir": str(out_dir),
                    "hypothesis_note": variant.get("hypothesis_note", ""),
                }
            )
    return runs


def _synthetic_vocab_size(data_cfg: dict[str, Any]) -> int:
    num_keys = int(data_cfg.get("num_key_types", 64))
    num_values = int(data_cfg.get("num_value_types", 64))
    filler = int(data_cfg.get("vocab_filler_size", 68))
    special = 4
    structural = 16
    return special + structural + num_keys + num_values + filler


def ablation_table() -> list[dict[str, str]]:
    return [
        {"id": "A1", "name": "Hybrid Attention Composition", "question": "MHA vs HCA vs CSA vs hybrid attention quality/memory trade-off."},
        {"id": "A2", "name": "Compression and Window Trade-off", "question": "How compression factor, local window, and sparse top-k affect retrieval and decode cost."},
        {"id": "A3", "name": "mHC Utility", "question": "Whether mHC improves stability in shallow and deeper mini regimes."},
        {"id": "A4", "name": "MoE Routing", "question": "Dense FFN vs routed experts, shared experts, and balance losses."},
        {"id": "A5", "name": "MTP Auxiliary Loss", "question": "Whether MTP helps convergence or distracts next-token learning."},
        {"id": "A6", "name": "System-Level Stack", "question": "Which DeepSeek-style components matter most when composed."},
    ]
