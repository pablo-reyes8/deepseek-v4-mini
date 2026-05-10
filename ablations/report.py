from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
    return flat


def append_summary_row(summary_csv: str | Path, row: dict[str, Any]) -> None:
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_fields: list[str] = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_fields = next(reader, [])
    fields = sorted(set(existing_fields) | set(row))

    rows: list[dict[str, Any]] = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for old in rows:
            writer.writerow(old)
        writer.writerow({field: row.get(field, "") for field in fields})


def aggregate_ablation_results(ablation_dir: str | Path):
    ablation_dir = Path(ablation_dir)
    records = []
    for path in sorted(ablation_dir.glob("**/final_metrics.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    try:
        import pandas as pd
    except ImportError:
        return records
    return pd.DataFrame([flatten_metrics(record) for record in records])


def build_ablation_markdown_report(ablation_id: str, ablation_dir: str | Path) -> str:
    ablation_dir = Path(ablation_dir)
    records = []
    for path in sorted(ablation_dir.glob("**/final_metrics.json")):
        records.append(flatten_metrics(json.loads(path.read_text(encoding="utf-8"))))

    lines = [
        f"# DeepSeek-V4 Mini Ablation {ablation_id}",
        "",
        f"Runs found: {len(records)}",
        "",
    ]
    if not records:
        return "\n".join(lines)

    preferred = [
        "ablation_id",
        "variant_name",
        "seed",
        "metrics/val/perplexity",
        "metrics/val/loss",
        "metrics/retrieval/retrieval_accuracy",
        "metrics/system/num_parameters_total",
        "metrics/system/peak_memory_mb",
        "metrics/inference/inference/deepseek_decode/cache_memory_mb",
    ]
    columns = [col for col in preferred if any(col in record for record in records)]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in records:
        lines.append("| " + " | ".join(str(record.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_summary_markdown(ablation_id: str, ablation_dir: str | Path) -> Path:
    ablation_dir = Path(ablation_dir)
    text = build_ablation_markdown_report(ablation_id, ablation_dir)
    path = ablation_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path
