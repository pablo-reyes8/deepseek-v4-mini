from __future__ import annotations

import torch

from ablations.ablation_configs import build_ablation_suite
from ablations.model_factory import build_model_from_ablation_config


def test_build_each_ablation_suite_quick():
    for ablation in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        runs = build_ablation_suite(ablation, seeds=[1], quick=True)
        assert runs
        assert all("model_config" in run for run in runs)
        assert all("data_config" in run for run in runs)
        assert all("training_config" in run for run in runs)
        assert all("output_dir" in run for run in runs)


def test_build_all_ablation_suite_quick():
    runs = build_ablation_suite("ALL", seeds=[1], quick=True, limit_variants=1)
    assert len(runs) == 6
    assert {run["ablation_id"] for run in runs} == {"A1", "A2", "A3", "A4", "A5", "A6"}


def test_model_factory_builds_all_quick_variants_for_core_suites():
    max_model = {
        "d_model": 16,
        "n_layers": 1,
        "max_seq_len": 16,
        "n_heads": 2,
        "head_dim": 8,
        "mlp_hidden_dim": 32,
        "expert_hidden_dim": 32,
        "shared_hidden_dim": 32,
        "n_hc": 2,
        "mtp_hidden_dim": 16,
    }
    for ablation in ["A1", "A6"]:
        for cfg in build_ablation_suite(ablation, seeds=[1], quick=True, max_model=max_model):
            model = build_model_from_ablation_config(cfg)
            assert model is not None
            batch = torch.randint(1, cfg["model_config"]["vocab_size"], (1, 8))
            with torch.no_grad():
                out = model(input_ids=batch)
            assert out["logits"].shape[:2] == (1, 8)
