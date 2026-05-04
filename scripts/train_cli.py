from __future__ import annotations

import argparse
import json

import torch

from data.syntethic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig
from training.adam_optmizer import build_adamw_optimizer
from training.autocast import setup_device_and_precision
from training.scheduler import WarmupCosineLR
from training.train_one_epoch import train_one_epoch


def print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def make_model_config(args: argparse.Namespace, vocab_size: int, pad_token_id: int) -> DeepSeekV4LMConfig:
    return DeepSeekV4LMConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=args.block_size,
        pad_token_id=pad_token_id,
        attention_type=args.attention,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        rotary_dim=args.rotary_dim or args.head_dim,
        compression_factor=args.compression_factor,
        hca_compression_factor=args.compression_factor,
        top_k_blocks=args.top_k_blocks,
        window_size=args.window_size,
        indexer_dim=args.indexer_dim,
        n_indexer_heads=args.n_indexer_heads,
        query_compression_dim=args.indexer_dim,
        ffn_type=args.ffn,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_experts=args.num_experts,
        top_k_experts=args.top_k_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        shared_hidden_dim=args.expert_hidden_dim,
        shared_experts=args.shared_experts,
        balance_loss_weight=args.balance_loss_weight,
        use_mhc=args.use_mhc,
        n_hc=args.n_hc,
        mhc_sinkhorn_iters=args.mhc_sinkhorn_iters,
        use_mtp=args.use_mtp,
        mtp_depth=args.mtp_depth,
        mtp_hidden_dim=args.d_model,
    )


def cmd_smoke(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    data_cfg = SyntheticRetrievalConfig(
        num_train_examples=args.num_train_examples,
        num_val_examples=args.num_val_examples,
        block_size=args.block_size,
        min_filler_tokens=args.min_filler_tokens,
        max_filler_tokens=args.max_filler_tokens,
        num_keys_per_example=args.num_keys,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )

    train_loader, val_loader, tokenizer = create_synthetic_retrieval_dataloaders(
        cfg=data_cfg,
        use_mtp=args.use_mtp,
        mtp_depth=args.mtp_depth,
    )

    model = DeepSeekV4LM(
        make_model_config(args, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_id)
    )
    optimizer, opt_info = build_adamw_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    precision = setup_device_and_precision(
        device=args.device,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
    )

    scheduler = WarmupCosineLR(
        optimizer,
        total_steps=max(1, args.epochs * args.max_batches),
        warmup_steps=args.warmup_steps,
        min_lr=args.min_learning_rate,
    )

    global_step = 0
    stats = None
    for epoch in range(args.epochs):
        stats, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=precision["device"],
            precision=precision,
            epoch=epoch,
            global_step=global_step,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            max_batches=args.max_batches,
            log_every=args.log_every,
            module_metrics_every=None,
            print_module_diagnostics=False,
            is_main_process=not args.quiet,
        )

    print_json(
        {
            "status": "ok",
            "global_step": global_step,
            "train_stats": stats,
            "model": {
                "attention": args.attention,
                "ffn": args.ffn,
                "use_mhc": args.use_mhc,
                "use_mtp": args.use_mtp,
                "num_parameters": sum(p.numel() for p in model.parameters()),
            },
            "optimizer": {
                "num_decay_tensors": opt_info["num_decay_tensors"],
                "num_no_decay_tensors": opt_info["num_no_decay_tensors"],
            },
            "validation_batches_available": len(val_loader),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Mini training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run tiny synthetic training")
    smoke.add_argument("--attention", choices=["mha", "hca", "csa"], default="mha")
    smoke.add_argument("--ffn", choices=["dense", "moe"], default="dense")
    smoke.add_argument("--use-mhc", action="store_true")
    smoke.add_argument("--use-mtp", action="store_true")
    smoke.add_argument("--d-model", type=int, default=32)
    smoke.add_argument("--n-layers", type=int, default=1)
    smoke.add_argument("--n-heads", type=int, default=4)
    smoke.add_argument("--head-dim", type=int, default=8)
    smoke.add_argument("--rotary-dim", type=int, default=None)
    smoke.add_argument("--mlp-hidden-dim", type=int, default=64)
    smoke.add_argument("--num-experts", type=int, default=4)
    smoke.add_argument("--top-k-experts", type=int, default=2)
    smoke.add_argument("--expert-hidden-dim", type=int, default=64)
    smoke.add_argument("--shared-experts", type=int, default=1)
    smoke.add_argument("--balance-loss-weight", type=float, default=0.0)
    smoke.add_argument("--compression-factor", type=int, default=4)
    smoke.add_argument("--top-k-blocks", type=int, default=2)
    smoke.add_argument("--window-size", type=int, default=4)
    smoke.add_argument("--indexer-dim", type=int, default=8)
    smoke.add_argument("--n-indexer-heads", type=int, default=2)
    smoke.add_argument("--n-hc", type=int, default=2)
    smoke.add_argument("--mhc-sinkhorn-iters", type=int, default=5)
    smoke.add_argument("--mtp-depth", type=int, default=2)
    smoke.add_argument("--block-size", type=int, default=32)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--num-train-examples", type=int, default=8)
    smoke.add_argument("--num-val-examples", type=int, default=4)
    smoke.add_argument("--min-filler-tokens", type=int, default=4)
    smoke.add_argument("--max-filler-tokens", type=int, default=16)
    smoke.add_argument("--num-keys", type=int, default=4)
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--max-batches", type=int, default=2)
    smoke.add_argument("--grad-accum-steps", type=int, default=1)
    smoke.add_argument("--grad-clip", type=float, default=1.0)
    smoke.add_argument("--learning-rate", type=float, default=3e-4)
    smoke.add_argument("--min-learning-rate", type=float, default=3e-5)
    smoke.add_argument("--weight-decay", type=float, default=0.1)
    smoke.add_argument("--warmup-steps", type=int, default=1)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--amp", action="store_true")
    smoke.add_argument("--amp-dtype", default="bf16")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--log-every", type=int, default=0)
    smoke.add_argument("--quiet", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
