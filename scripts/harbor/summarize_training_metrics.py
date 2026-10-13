#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key) or ""
    if not value:
        return default
    return float(value)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-metrics-csv", type=Path, required=True)
    parser.add_argument("--task-trend-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.log_metrics_csv)
    perf = [row for row in rows if row.get("harbor/reward_mean")]
    train = [row for row in rows if row.get("train/entropy_loss")]
    task_rows = read_csv(args.task_trend_csv)

    group_attempts = sum(f(row, "harbor/num_group_attempts") for row in perf)
    accepted_groups = len(perf) * 16
    discarded_groups = sum(f(row, "harbor/num_discarded_completed_group_tasks") for row in perf)
    tail_cancelled = sum(f(row, "harbor/num_partial_tail_trials_cancelled") for row in perf)
    train_trajectories = sum(f(row, "harbor/num_trajectory_samples") for row in perf)
    intended_groups = 89 * 5

    appearances = [int(row["appearances"]) for row in task_rows]
    appearance_hist = Counter(appearances)

    lines: list[str] = []
    lines.append("# Training Metrics Summary")
    lines.append("")
    lines.append("## Fully-Async Accounting")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| rollout perf rows | {len(perf)} |")
    lines.append(f"| train steps | {len(train)} |")
    lines.append(f"| intended groups, 89 tasks x 5 epochs | {intended_groups} |")
    lines.append(f"| accepted train groups | {accepted_groups} |")
    lines.append(f"| completed/attempted groups consumed by fully-async | {int(group_attempts)} |")
    lines.append(f"| completed groups discarded after data-source consumption | {int(discarded_groups)} |")
    lines.append(f"| tail sample trials cancelled at partial cutoff | {int(tail_cancelled)} |")
    lines.append(f"| accepted train trajectories | {int(train_trajectories)} |")
    if group_attempts:
        lines.append(f"| discarded group fraction | {fmt(discarded_groups / group_attempts)} |")
    if train_trajectories:
        lines.append(f"| tail cancellations / accepted train trajectories | {fmt(tail_cancelled / train_trajectories)} |")
    lines.append("")

    lines.append("## Accepted Task Coverage")
    lines.append("")
    lines.append("| appearances in accepted rollout dumps | number of tasks |")
    lines.append("|---:|---:|")
    for count in sorted(appearance_hist):
        lines.append(f"| {count} | {appearance_hist[count]} |")
    lines.append("")
    lines.append(
        f"Accepted appearances: min={min(appearances)}, mean={fmt(statistics.mean(appearances))}, max={max(appearances)}."
    )
    lines.append("")
    for count in sorted(appearance_hist):
        if count <= 3 or count >= 7:
            tasks = [row["task"] for row in task_rows if int(row["appearances"]) == count]
            lines.append(f"- {count} appearances: {', '.join(tasks)}")
    lines.append("")

    lines.append("## Rollout Metrics")
    lines.append("")
    lines.append("| rollout | timestamp | reward_mean | group_attempts | discarded_groups | tail_cancelled | timeout_ratio |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in perf:
        lines.append(
            "| {rollout} | {ts} | {reward} | {attempts} | {discarded} | {tail} | {timeout} |".format(
                rollout=row.get("rollout", ""),
                ts=row.get("timestamp", ""),
                reward=fmt(f(row, "harbor/reward_mean")),
                attempts=int(f(row, "harbor/num_group_attempts")),
                discarded=int(f(row, "harbor/num_discarded_completed_group_tasks")),
                tail=int(f(row, "harbor/num_partial_tail_trials_cancelled")),
                timeout=fmt(f(row, "harbor/trial_timeout_ratio")),
            )
        )
    lines.append("")

    if train:
        entropies = [f(row, "train/entropy_loss") for row in train]
        clipfracs = [f(row, "train/pg_clipfrac") for row in train]
        rollout_diffs = [f(row, "train/train_rollout_logprob_abs_diff") for row in train]
        lines.append("## Train Step Health")
        lines.append("")
        lines.append(f"- entropy_loss range: {fmt(min(entropies))} .. {fmt(max(entropies))}")
        lines.append(f"- pg_clipfrac range: {fmt(min(clipfracs))} .. {fmt(max(clipfracs))}")
        lines.append(f"- train_rollout_logprob_abs_diff range: {fmt(min(rollout_diffs))} .. {fmt(max(rollout_diffs))}")
        lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
