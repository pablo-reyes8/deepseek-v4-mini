from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from data.inspection import inspect_lm_dataloader
from data.syntethic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)
from data.text_datasets import HF_TEXT_DATASETS, create_hf_text_dataloaders, resolve_hf_text_preset


def print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def make_synthetic_config(args: argparse.Namespace) -> SyntheticRetrievalConfig:
    return SyntheticRetrievalConfig(
        num_train_examples=args.num_train_examples,
        num_val_examples=args.num_val_examples,
        block_size=args.block_size,
        min_filler_tokens=args.min_filler_tokens,
        max_filler_tokens=args.max_filler_tokens,
        num_keys_per_example=args.num_keys,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def cmd_presets(_: argparse.Namespace) -> None:
    result = {
        name: {
            "dataset_name": preset.dataset_name,
            "subset": preset.subset,
            "text_field": preset.text_field,
            "train_split": preset.train_split,
            "validation_split": preset.validation_split,
            "recommended_block_size": preset.recommended_block_size,
            "notes": preset.notes,
        }
        for name, preset in HF_TEXT_DATASETS.items()
    }
    result["synthetic_retrieval"] = {
        "dataset_name": "local generator",
        "notes": "No download; useful for CLI, CSA/HCA, and training smoke tests.",
    }
    print_json(result)


def cmd_synthetic_inspect(args: argparse.Namespace) -> None:
    cfg = make_synthetic_config(args)
    train_loader, val_loader, tokenizer = create_synthetic_retrieval_dataloaders(
        cfg=cfg,
        use_mtp=args.use_mtp,
        mtp_depth=args.mtp_depth,
    )

    print_json(
        {
            "dataset": "synthetic_retrieval",
            "config": asdict(cfg),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "train": inspect_lm_dataloader(
                train_loader,
                tokenizer=tokenizer,
                num_batches=args.num_batches,
                max_preview_tokens=args.max_preview_tokens,
            ),
            "validation": inspect_lm_dataloader(
                val_loader,
                tokenizer=tokenizer,
                num_batches=1,
                max_preview_tokens=args.max_preview_tokens,
            ),
        }
    )


def cmd_hf_info(args: argparse.Namespace) -> None:
    print_json(asdict(resolve_hf_text_preset(args.preset)))


def cmd_hf_prepare(args: argparse.Namespace) -> None:
    train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
        args.preset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        max_tokenizer_documents=args.max_tokenizer_documents,
        max_train_documents=args.max_train_documents,
        max_validation_documents=args.max_validation_documents,
    )

    result = {
        "dataset": args.preset,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()) if args.tokenizer_path else None,
        "train": inspect_lm_dataloader(
            train_loader,
            tokenizer=tokenizer,
            num_batches=args.num_batches,
            max_preview_tokens=args.max_preview_tokens,
        ),
        "validation": None,
    }

    if val_loader is not None:
        result["validation"] = inspect_lm_dataloader(
            val_loader,
            tokenizer=tokenizer,
            num_batches=1,
            max_preview_tokens=args.max_preview_tokens,
        )

    print_json(result)


def add_synthetic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-train-examples", type=int, default=8)
    parser.add_argument("--num-val-examples", type=int, default=4)
    parser.add_argument("--min-filler-tokens", type=int, default=8)
    parser.add_argument("--max-filler-tokens", type=int, default=24)
    parser.add_argument("--num-keys", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--mtp-depth", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--max-preview-tokens", type=int, default=48)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Mini data CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    presets = subparsers.add_parser("presets", help="List available dataset presets")
    presets.set_defaults(func=cmd_presets)

    synthetic = subparsers.add_parser("synthetic-inspect", help="Inspect local synthetic data")
    add_synthetic_args(synthetic)
    synthetic.set_defaults(func=cmd_synthetic_inspect)

    hf_info = subparsers.add_parser("hf-info", help="Show one Hugging Face text preset")
    hf_info.add_argument("preset")
    hf_info.set_defaults(func=cmd_hf_info)

    hf_prepare = subparsers.add_parser("hf-prepare", help="Download/tokenize and inspect HF text")
    hf_prepare.add_argument("preset")
    hf_prepare.add_argument("--block-size", type=int, default=None)
    hf_prepare.add_argument("--batch-size", type=int, default=8)
    hf_prepare.add_argument("--num-workers", type=int, default=0)
    hf_prepare.add_argument("--tokenizer-path", default=None)
    hf_prepare.add_argument("--vocab-size", type=int, default=16_000)
    hf_prepare.add_argument("--max-tokenizer-documents", type=int, default=10_000)
    hf_prepare.add_argument("--max-train-documents", type=int, default=2_000)
    hf_prepare.add_argument("--max-validation-documents", type=int, default=500)
    hf_prepare.add_argument("--num-batches", type=int, default=1)
    hf_prepare.add_argument("--max-preview-tokens", type=int, default=48)
    hf_prepare.set_defaults(func=cmd_hf_prepare)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
