from __future__ import annotations

from typing import Any

from data.syntethic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)
from data.text_datasets import create_hf_text_dataloaders


def build_dataloaders_from_ablation_config(config: dict[str, Any]):
    data_cfg = dict(config["data_config"])
    model_cfg = dict(config["model_config"])
    dataset = str(data_cfg.get("dataset", "synthetic_long_context"))
    use_mtp = bool(model_cfg.get("use_mtp", False))
    mtp_depth = int(model_cfg.get("mtp_depth", 1) or 1)

    if dataset in {"synthetic", "synthetic_long_context", "synthetic_retrieval"}:
        cfg = SyntheticRetrievalConfig(
            num_train_examples=int(data_cfg.get("num_train_examples", 64)),
            num_val_examples=int(data_cfg.get("num_val_examples", 24)),
            block_size=int(data_cfg.get("block_size", model_cfg.get("max_seq_len", 128))),
            min_filler_tokens=int(data_cfg.get("min_filler_tokens", 8)),
            max_filler_tokens=int(data_cfg.get("max_filler_tokens", 64)),
            num_keys_per_example=int(data_cfg.get("num_keys_per_example", 4)),
            vocab_filler_size=int(data_cfg.get("vocab_filler_size", 68)),
            num_key_types=int(data_cfg.get("num_key_types", 64)),
            num_value_types=int(data_cfg.get("num_value_types", 64)),
            batch_size=int(data_cfg.get("batch_size", 4)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=int(data_cfg.get("seed", config.get("seed", 1))),
        )
        return create_synthetic_retrieval_dataloaders(
            cfg=cfg,
            use_mtp=use_mtp,
            mtp_depth=mtp_depth,
        )

    return create_hf_text_dataloaders(
        preset_name=dataset,
        block_size=int(data_cfg.get("block_size", model_cfg.get("max_seq_len", 128))),
        batch_size=int(data_cfg.get("batch_size", 4)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        tokenizer_path=data_cfg.get("tokenizer_path", None),
        vocab_size=int(data_cfg.get("vocab_size", 16_000)),
        min_frequency=int(data_cfg.get("min_frequency", 2)),
        max_tokenizer_documents=data_cfg.get("max_tokenizer_documents", 50_000),
        max_train_documents=data_cfg.get("max_train_documents", 20_000),
        max_validation_documents=data_cfg.get("max_validation_documents", 2_000),
    )
