#!/usr/bin/env python3
"""Decompose rollout reward changes into task-mix and within-task residuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def first_int(row: dict[str, str], keys: tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        if row.get(key) not in ("", None):
            return as_int(row, key, default)
    return default


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def classify_task(rate: float) -> str:
    if rate == 0:
        return "zero"
    if rate <= 0.05:
        return "very_low"
    if rate <= 0.2:
        return "low"
    if rate <= 0.5:
        return "mid"
    if rate <= 0.8:
        return "high_mid"
    return "high"


def analyze(task_summary_path: Path, task_rollout_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_summary = read_csv(task_summary_path)
    task_rates = {
        row["task"]: as_float(row, "positive_rate")
        for row in task_summary
        if row.get("task")
    }

    by_rollout: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(task_rollout_path):
        by_rollout[as_int(row, "rollout")].append(row)

    rows: list[dict[str, Any]] = []
    for rollout in sorted(by_rollout):
        items = by_rollout[rollout]
        total_samples = sum(first_int(row, ("samples", "total_samples")) for row in items)
        total_pos = sum(first_int(row, ("positive_samples", "positives")) for row in items)
        if total_samples == 0:
            continue

        expected = 0.0
        bucket_samples: dict[str, int] = defaultdict(int)
        weighted_task_rate_rows = []
        zero_tasks = []
        high_tasks = []
        for row in items:
            task = row["task"]
            samples = first_int(row, ("samples", "total_samples"))
            positives = first_int(row, ("positive_samples", "positives"))
            rate = task_rates.get(task, 0.0)
            expected += rate * samples
            bucket = classify_task(rate)
            bucket_samples[bucket] += samples
            weighted_task_rate_rows.append((samples, task, positives / samples if samples else 0.0, rate))
            if rate == 0:
                zero_tasks.append(task)
            if rate > 0.8:
                high_tasks.append(task)

        actual = total_pos / total_samples
        expected_rate = expected / total_samples
        row_out: dict[str, Any] = {
            "rollout": rollout,
            "samples": total_samples,
            "positives": total_pos,
            "actual_positive_rate": actual,
            "expected_from_task_mix": expected_rate,
            "residual_actual_minus_mix": actual - expected_rate,
            "zero_sample_frac": bucket_samples["zero"] / total_samples,
            "very_low_sample_frac": bucket_samples["very_low"] / total_samples,
            "low_or_worse_sample_frac": (
                bucket_samples["zero"] + bucket_samples["very_low"] + bucket_samples["low"]
            )
            / total_samples,
            "high_sample_frac": bucket_samples["high"] / total_samples,
            "zero_tasks": ";".join(sorted(zero_tasks)),
            "high_tasks": ";".join(sorted(high_tasks)),
        }
        top_hard = [
            f"{task}:{samples}@{global_rate:.3f}"
            for samples, task, _actual_rate, global_rate in sorted(
                weighted_task_rate_rows,
                key=lambda item: (item[3], -item[0], item[1]),
            )[:5]
        ]
        row_out["top_low_global_rate_tasks"] = ";".join(top_hard)
        rows.append(row_out)

    actuals = [float(row["actual_positive_rate"]) for row in rows]
    expected = [float(row["expected_from_task_mix"]) for row in rows]
    zero_fracs = [float(row["zero_sample_frac"]) for row in rows]
    low_fracs = [float(row["low_or_worse_sample_frac"]) for row in rows]
    high_fracs = [float(row["high_sample_frac"]) for row in rows]
    residuals = [float(row["residual_actual_minus_mix"]) for row in rows]

    summary: dict[str, Any] = {
        "rollouts": len(rows),
        "actual_mean": sum(actuals) / len(actuals) if actuals else None,
        "expected_mix_mean": sum(expected) / len(expected) if expected else None,
        "actual_expected_pearson": pearson(actuals, expected),
        "actual_zero_frac_pearson": pearson(actuals, zero_fracs),
        "actual_low_or_worse_frac_pearson": pearson(actuals, low_fracs),
        "actual_high_frac_pearson": pearson(actuals, high_fracs),
        "residual_min": min(residuals) if residuals else None,
        "residual_max": max(residuals) if residuals else None,
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "rollout_task_mix.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = ["# Rollout task-mix decomposition", ""]
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Lowest Actual Reward Rollouts")
    lines.append("")
    lines.append(
        "| rollout | actual | expected mix | residual | zero frac | low-or-worse frac | high frac | top low-global-rate tasks |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: item["actual_positive_rate"])[:8]:
        lines.append(
            f"| {row['rollout']} | {row['actual_positive_rate']:.4f} | "
            f"{row['expected_from_task_mix']:.4f} | {row['residual_actual_minus_mix']:.4f} | "
            f"{row['zero_sample_frac']:.3f} | {row['low_or_worse_sample_frac']:.3f} | "
            f"{row['high_sample_frac']:.3f} | {row['top_low_global_rate_tasks']} |"
        )
    lines.append("")

    lines.append("## Largest Negative Residuals")
    lines.append("")
    lines.append("| rollout | actual | expected mix | residual | top low-global-rate tasks |")
    lines.append("|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: item["residual_actual_minus_mix"])[:8]:
        lines.append(
            f"| {row['rollout']} | {row['actual_positive_rate']:.4f} | "
            f"{row['expected_from_task_mix']:.4f} | {row['residual_actual_minus_mix']:.4f} | "
            f"{row['top_low_global_rate_tasks']} |"
        )
    lines.append("")

    lines.append("## Highest Actual Reward Rollouts")
    lines.append("")
    lines.append("| rollout | actual | expected mix | high frac | zero frac | high tasks |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for row in sorted(rows, key=lambda item: item["actual_positive_rate"], reverse=True)[:8]:
        lines.append(
            f"| {row['rollout']} | {row['actual_positive_rate']:.4f} | "
            f"{row['expected_from_task_mix']:.4f} | {row['high_sample_frac']:.3f} | "
            f"{row['zero_sample_frac']:.3f} | {row['high_tasks']} |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-summary", required=True)
    parser.add_argument("--task-rollout-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows, summary = analyze(Path(args.task_summary), Path(args.task_rollout_summary))
    write_outputs(rows, summary, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
