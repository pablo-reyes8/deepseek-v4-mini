from __future__ import annotations

import csv
import json
from pathlib import Path

from ablations.ablation_configs import build_ablation_suite
from ablations.run_ablation import run_single_ablation_config


def test_ablation_runner_one_batch_smoke(tmp_path: Path):
    runs = build_ablation_suite(
        "A1",
        base_output_dir=str(tmp_path / "ablations"),
        seeds=[1],
        quick=True,
        limit_variants=1,
        max_model={
            "d_model": 16,
            "n_layers": 1,
            "max_seq_len": 16,
            "n_heads": 2,
            "head_dim": 8,
            "mlp_hidden_dim": 32,
        },
        data_config={
            "block_size": 16,
            "batch_size": 2,
            "num_train_examples": 4,
            "num_val_examples": 2,
            "min_filler_tokens": 2,
            "max_filler_tokens": 4,
            "num_keys_per_example": 2,
            "num_key_types": 8,
            "num_value_types": 8,
            "vocab_filler_size": 8,
        },
        training_config={
            "epochs": 1,
            "max_batches_per_epoch": 1,
            "eval_max_batches": 1,
            "run_inference_benchmark": True,
            "inference_max_new_tokens": 1,
            "device": "cpu",
            "amp_enabled": False,
        },
    )

    result = run_single_ablation_config(runs[0])
    output_dir = Path(result["output_dir"])
    summary_csv = tmp_path / "ablations" / "A1" / "summary.csv"

    assert (output_dir / "final_metrics.json").exists()
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "last.pt").exists()
    assert summary_csv.exists()
    assert (tmp_path / "ablations" / "A1" / "summary_by_variant.csv").exists()

    metrics = json.loads((output_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert metrics["ablation_id"] == "A1"
    assert metrics["label_convention"] == "shifted_labels_from_dataloader"
    assert "model_config" in metrics
    assert "data_config" in metrics
    assert "training_config" in metrics
    assert metrics["metrics"]["system"]["global_step"] >= 1

    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    assert len(rows) == 1
