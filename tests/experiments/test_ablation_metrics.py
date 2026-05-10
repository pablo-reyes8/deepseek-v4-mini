from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ablations.evaluate_ablation import evaluate_retrieval
from ablations.report import write_summary_markdown
from ablations.run_ablation import run_ablation_suite


class TinyTokenizer:
    token_to_idx = {"<pad>": 0, "answer": 1, ":": 2, "value_7": 7}


class RetrievalPredictionModel(torch.nn.Module):
    def __init__(self, predicted_id: int, vocab_size: int = 16):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.predicted_id = int(predicted_id)
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None, return_aux=False):
        del labels, attention_mask, return_aux
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[..., self.predicted_id] = 100.0 + self.anchor
        return {"logits": logits}


def test_evaluate_retrieval_uses_model_predictions():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[2, 7, 0]], dtype=torch.long),
    }
    loader = DataLoader([batch], batch_size=None)

    correct = evaluate_retrieval(
        RetrievalPredictionModel(predicted_id=7),
        loader,
        tokenizer=TinyTokenizer(),
        max_batches=1,
        device="cpu",
    )
    wrong = evaluate_retrieval(
        RetrievalPredictionModel(predicted_id=6),
        loader,
        tokenizer=TinyTokenizer(),
        max_batches=1,
        device="cpu",
    )

    assert correct["retrieval_accuracy"] == 1.0
    assert wrong["retrieval_accuracy"] == 0.0
    assert correct["retrieval_targets"] == 1.0


def test_report_writes_variant_aggregates(tmp_path: Path):
    ablation_dir = tmp_path / "A1"
    for seed, ppl in [(1, 10.0), (2, 14.0)]:
        run_dir = ablation_dir / "variant_a" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        payload = {
            "ablation_id": "A1",
            "variant_name": "variant_a",
            "seed": seed,
            "metrics": {
                "val": {"loss": 2.0, "perplexity": ppl},
                "retrieval": {"retrieval_accuracy": 0.5 + seed * 0.1},
                "system": {"tokens_per_second_train": 100.0 + seed, "peak_memory_mb": 0.0},
            },
        }
        (run_dir / "final_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    write_summary_markdown("A1", ablation_dir)

    assert (ablation_dir / "summary_by_variant.csv").exists()
    assert (ablation_dir / "summary.md").exists()
    text = (ablation_dir / "summary.md").read_text(encoding="utf-8")
    assert "Best Variant by Validation Perplexity" in text
    assert "Variant-Level Mean/Std" in text


def test_ablation_runner_records_failed_variant(tmp_path: Path):
    config = {
        "ablation_id": "A1",
        "variant_name": "broken",
        "seed": 1,
        "output_dir": str(tmp_path / "A1" / "broken" / "seed_1"),
        "model_config": {"model_class": "unknown"},
        "data_config": {},
        "training_config": {},
    }

    results = run_ablation_suite([config])
    final_path = Path(config["output_dir"]) / "final_metrics.json"
    summary_csv = tmp_path / "A1" / "summary.csv"

    assert results[0]["failed"] is True
    assert final_path.exists()
    assert summary_csv.exists()

    final = json.loads(final_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    assert final["failed"] is True
    assert final["error_type"]
    assert rows and rows[0]["failed"] == "True"
