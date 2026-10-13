#!/usr/bin/env python3
"""Summarize Harbor rollout dump .pt files.

The script intentionally avoids Harbor imports so it can run on remote nodes
with only PyTorch and repository source available. It reads the dict format
written by rollout_dumps/*.pt and optionally compares the rollout positives to
external Harbor eval run directories that contain verifier/reward.txt files.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


NUMERIC_DUMP_RE = re.compile(r"^(\d+)\.pt$")
EVAL_DUMP_RE = re.compile(r"^eval_(\d+)\.pt$")


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise SystemExit("PyTorch is required to read rollout dump .pt files") from exc
    return torch


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _reward_value(reward: Any) -> float | None:
    if reward is None:
        return None
    if isinstance(reward, dict):
        reward = reward.get("reward", reward.get("score"))
    if hasattr(reward, "item"):
        try:
            reward = reward.item()
        except Exception:
            pass
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def _status_name(status: Any) -> str:
    if status is None:
        return "missing"
    name = getattr(status, "name", None)
    if name:
        return str(name).lower()
    text = str(status)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _task_key(sample: Any) -> str:
    metadata = _sample_get(sample, "metadata", None)
    if isinstance(metadata, dict):
        for key in (
            "task_name",
            "task",
            "task_id",
            "prompt_id",
            "name",
            "slug",
        ):
            value = metadata.get(key)
            if value:
                return str(value)

    prompt = _sample_get(sample, "prompt", None)
    if isinstance(prompt, str) and prompt:
        return prompt
    if prompt is not None:
        return str(prompt)

    group_id = _sample_get(sample, "group_id", None)
    if group_id is not None:
        return str(group_id)
    return "<unknown>"


def _response_length(sample: Any) -> int | None:
    value = _sample_get(sample, "response_length", None)
    if value is None:
        response = _sample_get(sample, "response", None)
        if isinstance(response, str):
            return len(response)
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _loss_tokens(sample: Any) -> float | None:
    loss_mask = _sample_get(sample, "loss_mask", None)
    if loss_mask is None:
        return None
    if hasattr(loss_mask, "sum"):
        try:
            return float(loss_mask.sum().item())
        except Exception:
            pass
    try:
        return float(sum(float(x) for x in loss_mask))
    except Exception:
        return None


def _mean(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _median(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _fmt_int(value: float | int | None) -> str:
    if value is None:
        return "-"
    return str(int(value))


def _safe_torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _dump_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    match = NUMERIC_DUMP_RE.match(name)
    if match:
        return (0, int(match.group(1)), name)
    match = EVAL_DUMP_RE.match(name)
    if match:
        return (1, int(match.group(1)), name)
    return (2, 0, name)


def _iter_dump_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    if args.files:
        for pattern in args.files:
            matches = [Path(p) for p in glob.glob(pattern)]
            if not matches and Path(pattern).exists():
                matches = [Path(pattern)]
            files.extend(matches)
    else:
        dump_dir = Path(args.dump_dir)
        patterns = ["*.pt"] if args.include_eval else ["[0-9]*.pt"]
        for pattern in patterns:
            files.extend(dump_dir.glob(pattern))

    deduped: dict[Path, None] = {}
    for path in files:
        path = path.expanduser()
        if path.is_file():
            if not args.include_eval and EVAL_DUMP_RE.match(path.name):
                continue
            deduped[path] = None
    return sorted(deduped.keys(), key=_dump_sort_key)


def _read_eval_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(run_dir.glob("*__*")):
        if not task_dir.is_dir():
            continue
        slug = task_dir.name.split("__", 1)[0]
        reward_path = task_dir / "verifier" / "reward.txt"
        reward: float | None = None
        if reward_path.exists():
            try:
                reward = float(reward_path.read_text().strip())
            except Exception:
                reward = None
        episode_count = len(list((task_dir / "agent").glob("episode-*")))
        rows[slug] = {
            "reward": reward,
            "episodes": episode_count,
            "path": str(task_dir),
        }
    return rows


@dataclass
class TaskStats:
    task: str
    samples: int = 0
    positives: int = 0
    rewards: Counter[str] = field(default_factory=Counter)
    statuses: Counter[str] = field(default_factory=Counter)
    rollouts: set[str] = field(default_factory=set)
    positive_rollouts: set[str] = field(default_factory=set)
    group_ids: set[str] = field(default_factory=set)
    train_groups: dict[str, dict[str, float | None]] = field(default_factory=dict)
    trajectories: dict[str, float | None] = field(default_factory=dict)
    response_lengths: list[int] = field(default_factory=list)
    loss_tokens: list[float] = field(default_factory=list)

    def add(self, sample: Any, dump_label: str) -> None:
        self.samples += 1
        self.rollouts.add(dump_label)

        reward = _reward_value(_sample_get(sample, "reward", None))
        if reward is None:
            self.rewards["missing"] += 1
        else:
            self.rewards[f"{reward:g}"] += 1
            if reward > 0:
                self.positives += 1
                self.positive_rollouts.add(dump_label)

        self.statuses[_status_name(_sample_get(sample, "status", None))] += 1

        group_id = _sample_get(sample, "group_id", None)
        if group_id is not None:
            self.group_ids.add(str(group_id))

        group_index = _sample_get(sample, "group_index", None)
        if group_index is not None:
            group_key = f"{dump_label}:{group_index}"
            trajectory_key = _trajectory_key(sample, dump_label, group_index)
            trajectories = self.train_groups.setdefault(group_key, {})
            if trajectory_key in trajectories and trajectories[trajectory_key] != reward:
                # This should not happen for retokenized chunks of one
                # trajectory. Keep the max reward so a successful chunk cannot
                # be hidden by a duplicate zero, and leave detailed inspection
                # to the dump when needed.
                old_reward = trajectories[trajectory_key] or 0.0
                new_reward = reward or 0.0
                trajectories[trajectory_key] = max(old_reward, new_reward)
            else:
                trajectories[trajectory_key] = reward
            self.trajectories[trajectory_key] = trajectories[trajectory_key]

        response_length = _response_length(sample)
        if response_length is not None:
            self.response_lengths.append(response_length)

        loss_tokens = _loss_tokens(sample)
        if loss_tokens is not None:
            self.loss_tokens.append(loss_tokens)

    @property
    def positive_rate(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.positives / self.samples

    def to_row(self, eval_runs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        group_rewards = [list(values.values()) for values in self.train_groups.values()]
        group_sizes = [len(values) for values in group_rewards]
        trajectory_count = len(self.trajectories)
        positive_trajectories = sum(
            1 for value in self.trajectories.values() if value is not None and value > 0.0
        )
        groups_all_zero = sum(
            1 for values in group_rewards if values and all((value or 0.0) <= 0.0 for value in values)
        )
        groups_any_pos = sum(
            1 for values in group_rewards if any(value is not None and value > 0.0 for value in values)
        )
        groups_all_pos = sum(
            1 for values in group_rewards if values and all(value is not None and value > 0.0 for value in values)
        )
        groups_mixed = sum(
            1
            for values in group_rewards
            if any(value is not None and value > 0.0 for value in values)
            and any((value or 0.0) <= 0.0 for value in values)
        )
        row: dict[str, Any] = {
            "task": self.task,
            "samples": self.samples,
            "positives": self.positives,
            "positive_rate": self.positive_rate,
            "trajectories": trajectory_count,
            "positive_trajectories": positive_trajectories,
            "trajectory_positive_rate": (
                positive_trajectories / trajectory_count if trajectory_count else 0.0
            ),
            "rollouts": len(self.rollouts),
            "positive_rollouts": len(self.positive_rollouts),
            "groups": len(self.train_groups) if self.train_groups else len(self.group_ids),
            "raw_group_ids": len(self.group_ids),
            "groups_all_zero": groups_all_zero,
            "groups_any_pos": groups_any_pos,
            "groups_mixed": groups_mixed,
            "groups_all_pos": groups_all_pos,
            "group_size_mean": _mean(group_sizes),
            "status_completed": self.statuses.get("completed", 0),
            "status_failed": self.statuses.get("failed", 0),
            "status_truncated": self.statuses.get("truncated", 0),
            "response_len_mean": _mean(self.response_lengths),
            "response_len_median": _median(self.response_lengths),
            "response_len_max": max(self.response_lengths) if self.response_lengths else None,
            "loss_tokens_mean": _mean(self.loss_tokens),
            "loss_tokens_median": _median(self.loss_tokens),
            "loss_tokens_max": max(self.loss_tokens) if self.loss_tokens else None,
            "reward_counts": dict(sorted(self.rewards.items())),
            "status_counts": dict(sorted(self.statuses.items())),
            "rollout_ids": sorted(self.rollouts, key=_rollout_label_key),
            "positive_rollout_ids": sorted(self.positive_rollouts, key=_rollout_label_key),
        }
        for label, run in eval_runs.items():
            eval_row = run.get(self.task, {})
            row[f"eval_{label}_reward"] = eval_row.get("reward")
            row[f"eval_{label}_episodes"] = eval_row.get("episodes")
        return row


@dataclass
class DumpStats:
    label: str
    path: str
    samples: int = 0
    positives: int = 0
    tasks: set[str] = field(default_factory=set)
    positive_tasks: set[str] = field(default_factory=set)
    statuses: Counter[str] = field(default_factory=Counter)
    response_lengths: list[int] = field(default_factory=list)
    loss_tokens: list[float] = field(default_factory=list)

    @property
    def positive_rate(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.positives / self.samples

    def to_row(self) -> dict[str, Any]:
        return {
            "rollout": self.label,
            "path": self.path,
            "samples": self.samples,
            "positives": self.positives,
            "positive_rate": self.positive_rate,
            "tasks": len(self.tasks),
            "positive_tasks": len(self.positive_tasks),
            "status_completed": self.statuses.get("completed", 0),
            "status_failed": self.statuses.get("failed", 0),
            "status_truncated": self.statuses.get("truncated", 0),
            "response_len_mean": _mean(self.response_lengths),
            "response_len_median": _median(self.response_lengths),
            "response_len_max": max(self.response_lengths) if self.response_lengths else None,
            "loss_tokens_mean": _mean(self.loss_tokens),
            "loss_tokens_median": _median(self.loss_tokens),
            "loss_tokens_max": max(self.loss_tokens) if self.loss_tokens else None,
            "status_counts": dict(sorted(self.statuses.items())),
        }


def _rollout_label(path: Path) -> str:
    return path.stem


def _trajectory_key(sample: Any, dump_label: str, group_index: Any) -> str:
    metadata = _sample_get(sample, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    rollout_id = metadata.get("harbor_rollout_id", dump_label)
    sample_rank = metadata.get(
        "harbor_sample_rank",
        metadata.get("harbor_trial_sample_rank", _sample_get(sample, "index", None)),
    )
    return f"{rollout_id}:{group_index}:{sample_rank}"


def _rollout_label_key(label: str) -> tuple[int, int, str]:
    if label.isdigit():
        return (0, int(label), label)
    match = re.match(r"eval_(\d+)$", label)
    if match:
        return (1, int(match.group(1)), label)
    return (2, 0, label)


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"empty label in {value!r}")
        return label, Path(path)
    path = Path(value)
    return path.name.replace("-", "_"), path


def _parse_focus(values: list[str] | None) -> list[str]:
    if not values:
        return []
    tasks: list[str] = []
    for value in values:
        for part in re.split(r"[\s,]+", value.strip()):
            if part:
                tasks.append(part)
    return tasks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _print_rollout_table(rows: list[dict[str, Any]]) -> None:
    print("\nRollouts")
    print("rollout samples pos pos_rate tasks pos_tasks failed trunc resp_mean loss_mean")
    for row in rows:
        print(
            row["rollout"],
            row["samples"],
            row["positives"],
            _fmt_float(row["positive_rate"], 3),
            row["tasks"],
            row["positive_tasks"],
            row["status_failed"],
            row["status_truncated"],
            _fmt_int(row["response_len_mean"]),
            _fmt_int(row["loss_tokens_mean"]),
        )


def _print_task_table(title: str, rows: list[dict[str, Any]], limit: int | None) -> None:
    if limit is not None:
        rows = rows[:limit]
    print(f"\n{title}")
    print("task samples pos pos_rate rollouts pos_rollouts failed trunc resp_mean loss_mean")
    for row in rows:
        print(
            row["task"],
            row["samples"],
            row["positives"],
            _fmt_float(row["positive_rate"], 3),
            row["rollouts"],
            row["positive_rollouts"],
            row["status_failed"],
            row["status_truncated"],
            _fmt_int(row["response_len_mean"]),
            _fmt_int(row["loss_tokens_mean"]),
        )


def _print_focus(rows_by_task: dict[str, dict[str, Any]], focus_tasks: list[str]) -> None:
    if not focus_tasks:
        return
    print("\nFocus Tasks")
    print(
        "task samples pos pos_rate rollouts pos_rollouts failed trunc "
        "resp_mean loss_mean eval_columns"
    )
    for task in focus_tasks:
        row = rows_by_task.get(task)
        if row is None:
            print(task, "MISSING")
            continue
        eval_bits = []
        for key in sorted(row):
            if key.startswith("eval_"):
                eval_bits.append(f"{key}={row[key]}")
        print(
            row["task"],
            row["samples"],
            row["positives"],
            _fmt_float(row["positive_rate"], 3),
            row["rollouts"],
            row["positive_rollouts"],
            row["status_failed"],
            row["status_truncated"],
            _fmt_int(row["response_len_mean"]),
            _fmt_int(row["loss_tokens_mean"]),
            " ".join(eval_bits) if eval_bits else "-",
        )


def _summarize_eval_alignment(
    task_rows: list[dict[str, Any]], eval_runs: dict[str, dict[str, dict[str, Any]]]
) -> None:
    for label in eval_runs:
        reward_key = f"eval_{label}_reward"
        present = [row for row in task_rows if row.get(reward_key) is not None]
        if not present:
            continue
        pass_rows = [row for row in present if row.get(reward_key) == 1.0]
        fail_rows = [row for row in present if row.get(reward_key) == 0.0]
        trained_with_pos = [row for row in present if row["positives"] > 0]
        eval_fail_but_train_pos = [
            row for row in fail_rows if row["positives"] > 0
        ]
        eval_pass_but_train_zero = [
            row for row in pass_rows if row["positives"] == 0
        ]
        print(f"\nEval Alignment: {label}")
        print(
            "tasks",
            len(present),
            "eval_pass",
            len(pass_rows),
            "eval_fail",
            len(fail_rows),
            "train_has_pos",
            len(trained_with_pos),
            "eval_fail_but_train_pos",
            len(eval_fail_but_train_pos),
            "eval_pass_but_train_zero",
            len(eval_pass_but_train_zero),
        )
        if eval_fail_but_train_pos:
            names = [row["task"] for row in eval_fail_but_train_pos[:30]]
            print("eval_fail_but_train_pos_top", " ".join(names))
        if eval_pass_but_train_zero:
            names = [row["task"] for row in eval_pass_but_train_zero[:30]]
            print("eval_pass_but_train_zero_top", " ".join(names))


def analyze(args: argparse.Namespace) -> int:
    dump_files = _iter_dump_files(args)
    if not dump_files:
        raise SystemExit("No rollout dump files matched the requested inputs")

    torch = _import_torch()

    task_stats: dict[str, TaskStats] = {}
    dump_stats: list[DumpStats] = []
    global_statuses: Counter[str] = Counter()
    total_samples = 0
    total_positives = 0

    for index, path in enumerate(dump_files, start=1):
        label = _rollout_label(path)
        if args.progress:
            print(f"[{index}/{len(dump_files)}] loading {path}", flush=True)
        data = _safe_torch_load(torch, path)
        samples = data.get("samples", data) if isinstance(data, dict) else data
        if not isinstance(samples, Iterable):
            print(f"warning: {path} does not contain iterable samples; skipping")
            continue

        current = DumpStats(label=label, path=str(path))
        for sample in samples:
            task = _task_key(sample)
            stats = task_stats.setdefault(task, TaskStats(task=task))
            stats.add(sample, label)

            current.samples += 1
            current.tasks.add(task)
            total_samples += 1

            reward = _reward_value(_sample_get(sample, "reward", None))
            if reward is not None and reward > 0:
                current.positives += 1
                current.positive_tasks.add(task)
                total_positives += 1

            status = _status_name(_sample_get(sample, "status", None))
            current.statuses[status] += 1
            global_statuses[status] += 1

            response_length = _response_length(sample)
            if response_length is not None:
                current.response_lengths.append(response_length)
            loss_tokens = _loss_tokens(sample)
            if loss_tokens is not None:
                current.loss_tokens.append(loss_tokens)

        dump_stats.append(current)

    eval_runs: dict[str, dict[str, dict[str, Any]]] = {}
    for item in args.compare_run or []:
        label, path = _parse_labeled_path(item)
        eval_runs[label] = _read_eval_run(path)

    task_rows = [
        stats.to_row(eval_runs)
        for stats in sorted(task_stats.values(), key=lambda item: item.task)
    ]
    dump_rows = [stats.to_row() for stats in dump_stats]
    rows_by_task = {row["task"]: row for row in task_rows}

    task_rows_by_rate = sorted(
        task_rows,
        key=lambda row: (row["positive_rate"], row["positives"], row["samples"], row["task"]),
    )
    top_rows = list(reversed(task_rows_by_rate))

    print("Summary")
    print("files", len(dump_files))
    print("samples", total_samples)
    print("positives", total_positives)
    print("positive_rate", _fmt_float(total_positives / total_samples if total_samples else 0.0, 4))
    print("tasks_seen", len(task_rows))
    print("tasks_with_pos", sum(1 for row in task_rows if row["positives"] > 0))
    print("tasks_zero_pos", sum(1 for row in task_rows if row["positives"] == 0))
    print("status_counts", dict(sorted(global_statuses.items())))

    _print_rollout_table(dump_rows)
    _print_focus(rows_by_task, _parse_focus(args.focus))
    _print_task_table("Lowest Positive Rate Tasks", task_rows_by_rate, args.table_limit)
    _print_task_table("Highest Positive Rate Tasks", top_rows, args.table_limit)
    _summarize_eval_alignment(task_rows, eval_runs)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        _write_csv(output_dir / "rollout_summary.csv", dump_rows)
        _write_csv(output_dir / "task_summary.csv", task_rows)
        _write_json(
            output_dir / "summary.json",
            {
                "summary": {
                    "files": len(dump_files),
                    "samples": total_samples,
                    "positives": total_positives,
                    "positive_rate": total_positives / total_samples if total_samples else 0.0,
                    "tasks_seen": len(task_rows),
                    "tasks_with_pos": sum(1 for row in task_rows if row["positives"] > 0),
                    "tasks_zero_pos": sum(1 for row in task_rows if row["positives"] == 0),
                    "status_counts": dict(sorted(global_statuses.items())),
                },
                "rollouts": dump_rows,
                "tasks": task_rows,
            },
        )
        print(f"\nWrote {output_dir / 'rollout_summary.csv'}")
        print(f"Wrote {output_dir / 'task_summary.csv'}")
        print(f"Wrote {output_dir / 'summary.json'}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        default="rollout_dumps",
        help="Directory containing rollout dump .pt files. Default: rollout_dumps",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Explicit dump file paths or glob patterns. Overrides --dump-dir.",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Include eval_*.pt files. Default only includes numeric train dumps.",
    )
    parser.add_argument(
        "--compare-run",
        action="append",
        help=(
            "External Harbor eval run dir to compare, optionally label=path. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--focus",
        action="append",
        help="Comma or whitespace separated task names to print in a dedicated table.",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=30,
        help="Rows to print for top/bottom task tables. Use 0 to print none.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory for task_summary.csv, rollout_summary.csv, summary.json.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one line before loading each dump. Default: true.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
