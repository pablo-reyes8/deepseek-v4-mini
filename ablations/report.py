from __future__ import annotations

import csv
import json
import math
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


def aggregate_by_variant(df):
    numeric_columns = [
        "metrics/val/loss",
        "metrics/val/perplexity",
        "metrics/retrieval/retrieval_accuracy",
        "metrics/system/tokens_per_second_train",
        "metrics/system/peak_memory_mb",
    ]
    if isinstance(df, list):
        try:
            import pandas as pd
        except ImportError:
            return []
        df = pd.DataFrame([flatten_metrics(record) for record in df])
    if len(df) == 0:
        return df
    available = [col for col in numeric_columns if col in df.columns]
    if not available:
        return df[["ablation_id", "variant_name"]].drop_duplicates().reset_index(drop=True)
    for col in available:
        df[col] = df[col].apply(_to_float_or_nan)
    grouped = df.groupby(["ablation_id", "variant_name"], dropna=False)[available]
    return grouped.agg(["mean", "std"]).reset_index()


def write_summary_by_variant(ablation_dir: str | Path) -> Path | None:
    ablation_dir = Path(ablation_dir)
    df = aggregate_ablation_results(ablation_dir)
    if isinstance(df, list):
        rows = _aggregate_records_by_variant(df)
        if not rows:
            return None
        path = ablation_dir / "summary_by_variant.csv"
        fields = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path
    summary = aggregate_by_variant(df)
    path = ablation_dir / "summary_by_variant.csv"
    summary.to_csv(path, index=False)
    return path


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

    lines.extend(["", "## Best Variant by Validation Perplexity", ""])
    lines.extend(_best_section(records, "metrics/val/perplexity", lower_is_better=True))
    lines.extend(["", "## Best Variant by Retrieval Accuracy", ""])
    lines.extend(_best_section(records, "metrics/retrieval/retrieval_accuracy", lower_is_better=False))
    lines.extend(["", "## Best Variant by Decode Memory", ""])
    lines.extend(_best_section(records, "metrics/inference/inference/deepseek_decode/cache_memory_mb", lower_is_better=True))
    lines.extend(["", "## Variant-Level Mean/Std", ""])
    lines.extend(_variant_mean_std_markdown(records))
    return "\n".join(lines) + "\n"


def write_summary_markdown(ablation_id: str, ablation_dir: str | Path) -> Path:
    ablation_dir = Path(ablation_dir)
    write_summary_by_variant(ablation_dir)
    text = build_ablation_markdown_report(ablation_id, ablation_dir)
    path = ablation_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def _to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _best_section(records: list[dict[str, Any]], metric: str, *, lower_is_better: bool) -> list[str]:
    candidates = [(record, _to_float_or_nan(record.get(metric))) for record in records]
    candidates = [(record, value) for record, value in candidates if math.isfinite(value)]
    if not candidates:
        return [f"No finite values for `{metric}` yet."]
    best_record, best_value = sorted(candidates, key=lambda item: item[1], reverse=not lower_is_better)[0]
    return [
        f"- Metric: `{metric}`",
        f"- Variant: `{best_record.get('variant_name', '')}`",
        f"- Seed: `{best_record.get('seed', '')}`",
        f"- Value: `{best_value:.6g}`",
    ]


def _variant_mean_std_markdown(records: list[dict[str, Any]]) -> list[str]:
    rows = _aggregate_records_by_variant(records)
    if not rows:
        return ["No numeric variant aggregates available yet."]
    columns = [
        "ablation_id",
        "variant_name",
        "metrics/val/perplexity_mean",
        "metrics/val/perplexity_std",
        "metrics/retrieval/retrieval_accuracy_mean",
        "metrics/retrieval/retrieval_accuracy_std",
        "metrics/system/tokens_per_second_train_mean",
        "metrics/system/tokens_per_second_train_std",
    ]
    columns = [col for col in columns if any(col in row for row in rows)]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col, "")) for col in columns) + " |")
    return lines


def _aggregate_records_by_variant(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "metrics/val/loss",
        "metrics/val/perplexity",
        "metrics/retrieval/retrieval_accuracy",
        "metrics/system/tokens_per_second_train",
        "metrics/system/peak_memory_mb",
    ]
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record.get("ablation_id"), record.get("variant_name")), []).append(record)
    rows = []
    for (ablation_id, variant_name), group_records in sorted(groups.items()):
        row = {"ablation_id": ablation_id, "variant_name": variant_name}
        for metric in metrics:
            values = [_to_float_or_nan(record.get(metric)) for record in group_records]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = math.sqrt(variance) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
