from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from ablations.ablation_configs import build_ablation_suite
from ablations.data_factory import build_dataloaders_from_ablation_config
from ablations.evaluate_ablation import (
    benchmark_inference,
    evaluate_lm,
    evaluate_retrieval,
    peak_memory_mb,
)
from ablations.model_factory import build_model_from_ablation_config, count_parameters
from ablations.report import append_summary_row, flatten_metrics, write_summary_markdown
from training.adam_optmizer import build_adamw_optimizer
from training.autocast import setup_device_and_precision
from training.muon_optimizer import build_muon_adamw_optimizer
from training.scheduler import WarmupCosineLR
from training.seed import set_seed
from training.train_one_epoch import train_one_epoch


def run_single_ablation_config(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    metrics_jsonl = output_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    set_seed(int(config.get("seed", 1)))
    train_loader, val_loader, tokenizer = build_dataloaders_from_ablation_config(config)
    model = build_model_from_ablation_config(config)
    train_cfg = dict(config["training_config"])
    precision = setup_device_and_precision(
        device=train_cfg.get("device", "auto"),
        amp_enabled=bool(train_cfg.get("amp_enabled", True)),
        amp_dtype=str(train_cfg.get("amp_dtype", "bf16")),
    )
    device = precision["device"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.to(device)
    optimizer, opt_info = _build_optimizer(model, train_cfg)
    scheduler = WarmupCosineLR(
        optimizer,
        total_steps=max(1, int(train_cfg["epochs"]) * int(train_cfg["max_batches_per_epoch"])),
        warmup_steps=int(train_cfg.get("warmup_steps", 1)),
        min_lr=float(train_cfg.get("min_learning_rate", 3e-5)),
    )

    global_step = 0
    train_stats = {}
    val_stats = {}
    start = time.perf_counter()
    for epoch in range(int(train_cfg["epochs"])):
        train_stats, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            precision=precision,
            epoch=epoch,
            global_step=global_step,
            grad_clip=float(train_cfg.get("grad_clip_norm", 1.0)),
            grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
            max_batches=int(train_cfg["max_batches_per_epoch"]),
            log_every=int(train_cfg.get("log_every", 0)),
            module_metrics_every=train_cfg.get("module_metrics_every", None),
            print_module_diagnostics=False,
            is_main_process=False,
        )
        val_stats = evaluate_lm(
            model=model,
            val_loader=val_loader,
            max_batches=int(train_cfg.get("eval_max_batches", 1)),
            device=device,
            precision=precision,
        )
        append_jsonl(
            metrics_jsonl,
            {
                "event": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_stats.get("loss"),
                "train_lm_loss": train_stats.get("lm_loss"),
                "train_mtp_loss": train_stats.get("mtp_loss"),
                "train_moe_aux_loss": train_stats.get("moe_aux_loss"),
                "val_loss": val_stats.get("loss"),
                "val_lm_loss": val_stats.get("lm_loss"),
                "val_perplexity": val_stats.get("perplexity"),
            },
        )
    elapsed = time.perf_counter() - start

    if train_cfg.get("save_checkpoints", True):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "global_step": global_step,
                "train_stats": train_stats,
            },
            output_dir / "last.pt",
        )

    if not val_stats:
        val_stats = evaluate_lm(
            model=model,
            val_loader=val_loader,
            max_batches=int(train_cfg.get("eval_max_batches", 1)),
            device=device,
            precision=precision,
        )
    retrieval_stats = evaluate_retrieval(
        model=model,
        val_loader=val_loader,
        tokenizer=tokenizer,
        max_batches=int(train_cfg.get("eval_max_batches", 1)),
        device=device,
    )

    inference_stats: dict[str, Any] = {}
    if train_cfg.get("run_inference_benchmark", True):
        sample = next(iter(val_loader))["input_ids"][:1]
        inference_stats = benchmark_inference(
            model=model,
            input_ids=sample,
            inference_config=None,
        )

    system_stats = {
        **count_parameters(model),
        "peak_memory_mb": peak_memory_mb(device),
        "train_elapsed_sec": float(elapsed),
        "tokens_per_second_train": _train_tokens_per_second(train_stats, config, elapsed),
        "global_step": float(global_step),
    }
    metrics = {
        "train": train_stats,
        "val": val_stats,
        "retrieval": retrieval_stats,
        "inference": inference_stats,
        "optimizer": opt_info,
        "system": system_stats,
    }
    final = {
        "ablation_id": config["ablation_id"],
        "variant_name": config["variant_name"],
        "seed": config["seed"],
        "output_dir": str(output_dir),
        "failed": False,
        "label_convention": "shifted_labels_from_dataloader",
        "model_config": config["model_config"],
        "data_config": config["data_config"],
        "training_config": config["training_config"],
        "metrics": metrics,
    }
    append_jsonl(
        metrics_jsonl,
        {
            "event": "final",
            "global_step": global_step,
            "train_loss": train_stats.get("loss"),
            "val_loss": val_stats.get("loss"),
            "val_perplexity": val_stats.get("perplexity"),
            "retrieval_accuracy": retrieval_stats.get("retrieval_accuracy"),
        },
    )
    (output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")

    summary_csv = _summary_csv_for_config(config)
    append_summary_row(summary_csv, flatten_metrics(final))
    write_summary_markdown(config["ablation_id"], summary_csv.parent)
    _cleanup_after_run(model, optimizer)
    return final


def run_ablation_suite(configs: list[dict[str, Any]], fail_fast: bool = False) -> list[dict[str, Any]]:
    results = []
    for config in configs:
        try:
            results.append(run_single_ablation_config(config))
        except Exception as exc:
            failure = _record_failed_run(config, exc)
            results.append(failure)
            if fail_fast:
                raise
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSeek-V4 Mini ablation suites")
    parser.add_argument("--ablation", default="A1", help="A1, A2, A3, A4, A5, A6, or ALL")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--base-output-dir", default="outputs/ablations")
    parser.add_argument("--limit-variants", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    training_overrides = {
        "device": args.device,
        "max_batches_per_epoch": args.max_batches_per_epoch,
        "eval_max_batches": args.eval_max_batches,
        "epochs": args.epochs,
    }
    configs = build_ablation_suite(
        args.ablation,
        base_output_dir=args.base_output_dir,
        seeds=args.seeds,
        quick=args.quick,
        training_config=training_overrides,
        limit_variants=args.limit_variants,
    )
    run_ablation_suite(configs, fail_fast=args.fail_fast)


def _build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]):
    opt_type = str(train_cfg.get("optimizer_type", "adamw"))
    kwargs = {
        "learning_rate": float(train_cfg.get("learning_rate", 3e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.1)),
    }
    if opt_type == "muon_adamw":
        try:
            return build_muon_adamw_optimizer(model, **kwargs)
        except RuntimeError:
            return build_adamw_optimizer(model, **kwargs)
    if opt_type == "adamw":
        return build_adamw_optimizer(model, **kwargs)
    raise ValueError(f"Unsupported optimizer_type={opt_type!r}")


def _train_tokens_per_second(train_stats: dict[str, Any], config: dict[str, Any], elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    seen_samples = float(train_stats.get("n_seen_samples", 0.0))
    block_size = float(config["data_config"].get("block_size", config["model_config"].get("max_seq_len", 1)))
    return seen_samples * block_size / elapsed


def _cleanup_after_run(model: torch.nn.Module, optimizer: Any) -> None:
    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _summary_csv_for_config(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output_dir"])
    return output_dir.parents[1] / "summary.csv"


def _record_failed_run(config: dict[str, Any], exc: Exception) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "ablation_id": config["ablation_id"],
        "variant_name": config["variant_name"],
        "seed": config["seed"],
        "output_dir": str(output_dir),
        "failed": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "label_convention": "shifted_labels_from_dataloader",
        "model_config": config["model_config"],
        "data_config": config["data_config"],
        "training_config": config["training_config"],
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(failure, indent=2, default=str),
        encoding="utf-8",
    )
    append_jsonl(
        output_dir / "metrics.jsonl",
        {
            "event": "failed",
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
        },
    )
    summary_csv = _summary_csv_for_config(config)
    append_summary_row(summary_csv, flatten_metrics(failure))
    write_summary_markdown(config["ablation_id"], summary_csv.parent)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return failure


if __name__ == "__main__":
    main()
