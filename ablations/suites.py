from __future__ import annotations

from typing import Any

from ablations.ablation_configs import build_ablation_suite
from ablations.run_ablation import run_ablation_suite


def run_named_ablation(
    ablation_id: str,
    *,
    data_config: dict[str, Any] | None = None,
    max_model: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    seeds: list[int] | None = None,
    quick: bool = False,
    base_output_dir: str = "outputs/ablations",
    limit_variants: int | None = None,
) -> list[dict[str, Any]]:
    configs = build_ablation_suite(
        ablation_id,
        base_output_dir=base_output_dir,
        seeds=seeds,
        quick=quick,
        data_config=data_config,
        max_model=max_model,
        training_config=training_config,
        limit_variants=limit_variants,
    )
    return run_ablation_suite(configs)


def ablation_1(**kwargs):
    return run_named_ablation("A1", **kwargs)


def ablation_2(**kwargs):
    return run_named_ablation("A2", **kwargs)


def ablation_3(**kwargs):
    return run_named_ablation("A3", **kwargs)


def ablation_4(**kwargs):
    return run_named_ablation("A4", **kwargs)


def ablation_5(**kwargs):
    return run_named_ablation("A5", **kwargs)


def ablation_6(**kwargs):
    return run_named_ablation("A6", **kwargs)
