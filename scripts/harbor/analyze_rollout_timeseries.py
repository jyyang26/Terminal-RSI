#!/usr/bin/env python3
"""Build per-task rollout time-series summaries for Harbor .pt dumps."""

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


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyTorch is required to read rollout dump .pt files") from exc
    return torch


def _sample_get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _metadata(sample: Any) -> dict[str, Any]:
    value = _sample_get(sample, "metadata", None)
    return value if isinstance(value, dict) else {}


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
    metadata = _metadata(sample)
    task_entry = metadata.get("harbor_task_entry")
    if isinstance(task_entry, dict):
        prompt = task_entry.get("prompt")
        if prompt:
            return str(prompt)
    for key in (
        "task_name",
        "task",
        "task_id",
        "prompt_id",
        "name",
        "slug",
        "harbor_task_name",
    ):
        value = metadata.get(key)
        if value:
            text = str(value)
            return text.rsplit("/", 1)[-1]

    prompt = _sample_get(sample, "prompt", None)
    if prompt is not None:
        return str(prompt)

    group_id = _sample_get(sample, "group_id", None)
    return str(group_id) if group_id is not None else "<unknown>"


def _response_length(sample: Any) -> int | None:
    value = _sample_get(sample, "response_length", None)
    if value is None:
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


def _safe_torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _iter_dump_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        files: list[Path] = []
        for pattern in args.files:
            matches = [Path(item) for item in glob.glob(pattern)]
            if not matches and Path(pattern).exists():
                matches = [Path(pattern)]
            files.extend(matches)
    else:
        files = list(Path(args.dump_dir).glob("[0-9]*.pt"))
    deduped: dict[Path, None] = {}
    for path in files:
        if path.is_file() and NUMERIC_DUMP_RE.match(path.name):
            deduped[path] = None
    return sorted(deduped, key=lambda path: int(path.stem))


def _trajectory_key(sample: Any, rollout_id: int, group_index: Any) -> str:
    metadata = _metadata(sample)
    sample_rank = metadata.get(
        "harbor_sample_rank",
        metadata.get("harbor_trial_sample_rank", _sample_get(sample, "index", None)),
    )
    return f"{rollout_id}:{group_index}:{sample_rank}"


@dataclass
class TrajectoryAgg:
    reward: float | None = None
    status_counts: Counter[str] = field(default_factory=Counter)
    chunk_count: int = 0
    response_lengths: list[int] = field(default_factory=list)
    loss_tokens: list[float] = field(default_factory=list)
    turns: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    length_finish_turns: int | None = None
    summarization_count: int | None = None
    conditioning_match: bool | None = None

    def add(self, sample: Any) -> None:
        reward = _reward_value(_sample_get(sample, "reward", None))
        if self.reward is None:
            self.reward = reward
        elif reward is not None:
            self.reward = max(self.reward, reward)

        self.status_counts[_status_name(_sample_get(sample, "status", None))] += 1
        self.chunk_count += 1

        response_length = _response_length(sample)
        if response_length is not None:
            self.response_lengths.append(response_length)
        loss_tokens = _loss_tokens(sample)
        if loss_tokens is not None:
            self.loss_tokens.append(loss_tokens)

        metadata = _metadata(sample)
        self.turns = _first_int(self.turns, metadata.get("harbor_num_turns"))
        self.output_tokens = _first_int(self.output_tokens, metadata.get("harbor_total_output_tokens"))
        self.finish_reason = self.finish_reason or _optional_str(metadata.get("harbor_finish_reason_type"))
        self.length_finish_turns = _first_int(
            self.length_finish_turns, metadata.get("harbor_length_finish_turns")
        )
        self.summarization_count = _first_int(
            self.summarization_count, metadata.get("harbor_summarization_count")
        )
        if self.conditioning_match is None and "harbor_train_inference_conditioning_match" in metadata:
            self.conditioning_match = bool(metadata.get("harbor_train_inference_conditioning_match"))

    @property
    def positive(self) -> bool:
        return self.reward is not None and self.reward > 0.0

    @property
    def status(self) -> str:
        if not self.status_counts:
            return "missing"
        return self.status_counts.most_common(1)[0][0]


def _first_int(current: int | None, value: Any) -> int | None:
    if current is not None:
        return current
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass
class TaskRolloutAgg:
    task: str
    rollout: int
    groups: dict[str, dict[str, TrajectoryAgg]] = field(default_factory=dict)
    trajectories: dict[str, TrajectoryAgg] = field(default_factory=dict)
    samples: int = 0
    positive_samples: int = 0
    statuses: Counter[str] = field(default_factory=Counter)

    def add(self, sample: Any) -> None:
        self.samples += 1
        reward = _reward_value(_sample_get(sample, "reward", None))
        if reward is not None and reward > 0.0:
            self.positive_samples += 1
        self.statuses[_status_name(_sample_get(sample, "status", None))] += 1

        group_index = _sample_get(sample, "group_index", None)
        group_key = f"{self.rollout}:{group_index}"
        trajectory_key = _trajectory_key(sample, self.rollout, group_index)
        trajectory = self.trajectories.setdefault(trajectory_key, TrajectoryAgg())
        trajectory.add(sample)
        self.groups.setdefault(group_key, {})[trajectory_key] = trajectory

    def to_row(self) -> dict[str, Any]:
        trajectories = list(self.trajectories.values())
        positive_trajectories = sum(1 for trajectory in trajectories if trajectory.positive)
        group_values = [list(group.values()) for group in self.groups.values()]
        all_zero_groups = sum(1 for group in group_values if group and all(not t.positive for t in group))
        mixed_groups = sum(
            1 for group in group_values if any(t.positive for t in group) and any(not t.positive for t in group)
        )
        all_positive_groups = sum(1 for group in group_values if group and all(t.positive for t in group))

        turns = [t.turns for t in trajectories if t.turns is not None]
        output_tokens = [t.output_tokens for t in trajectories if t.output_tokens is not None]
        chunks = [t.chunk_count for t in trajectories]
        response_lengths = [value for t in trajectories for value in t.response_lengths]
        loss_tokens = [value for t in trajectories for value in t.loss_tokens]
        finish_reasons = Counter(t.finish_reason or "missing" for t in trajectories)
        traj_statuses = Counter(t.status for t in trajectories)
        conditioning_matches = [t.conditioning_match for t in trajectories if t.conditioning_match is not None]

        return {
            "task": self.task,
            "rollout": self.rollout,
            "samples": self.samples,
            "positive_samples": self.positive_samples,
            "sample_positive_rate": self.positive_samples / self.samples if self.samples else 0.0,
            "trajectories": len(trajectories),
            "positive_trajectories": positive_trajectories,
            "trajectory_positive_rate": (
                positive_trajectories / len(trajectories) if trajectories else 0.0
            ),
            "groups": len(group_values),
            "all_zero_groups": all_zero_groups,
            "mixed_groups": mixed_groups,
            "all_positive_groups": all_positive_groups,
            "turns_mean": _mean(turns),
            "turns_median": _median(turns),
            "turns_max": max(turns) if turns else None,
            "output_tokens_mean": _mean(output_tokens),
            "output_tokens_median": _median(output_tokens),
            "output_tokens_max": max(output_tokens) if output_tokens else None,
            "chunks_per_trajectory_mean": _mean(chunks),
            "response_len_mean": _mean(response_lengths),
            "loss_tokens_mean": _mean(loss_tokens),
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "trajectory_status_counts": dict(sorted(traj_statuses.items())),
            "sample_status_counts": dict(sorted(self.statuses.items())),
            "conditioning_match_rate": (
                sum(1 for value in conditioning_matches if value) / len(conditioning_matches)
                if conditioning_matches
                else None
            ),
        }


def _weighted_rate(rows: list[dict[str, Any]], pos_key: str, denom_key: str) -> float | None:
    denom = sum(int(row[denom_key]) for row in rows)
    if denom <= 0:
        return None
    pos = sum(int(row[pos_key]) for row in rows)
    return pos / denom


def _trend_rows(task_rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rollout_rows:
        by_task[row["task"]].append(row)

    output: list[dict[str, Any]] = []
    for task, rows in by_task.items():
        rows = sorted(rows, key=lambda row: int(row["rollout"]))
        early = rows[: min(2, len(rows))]
        late = rows[-min(2, len(rows)) :]
        first = rows[0]
        last = rows[-1]
        early_rate = _weighted_rate(early, "positive_trajectories", "trajectories")
        late_rate = _weighted_rate(late, "positive_trajectories", "trajectories")
        output.append(
            {
                "task": task,
                "appearances": len(rows),
                "first_rollout": first["rollout"],
                "last_rollout": last["rollout"],
                "first_rate": first["trajectory_positive_rate"],
                "last_rate": last["trajectory_positive_rate"],
                "first_trajectories": first["trajectories"],
                "last_trajectories": last["trajectories"],
                "first_positive_trajectories": first["positive_trajectories"],
                "last_positive_trajectories": last["positive_trajectories"],
                "first_turns_mean": first["turns_mean"],
                "last_turns_mean": last["turns_mean"],
                "first_output_tokens_mean": first["output_tokens_mean"],
                "last_output_tokens_mean": last["output_tokens_mean"],
                "early2_rate": early_rate,
                "late2_rate": late_rate,
                "delta_first_last": (
                    float(last["trajectory_positive_rate"]) - float(first["trajectory_positive_rate"])
                ),
                "delta_early_late": (
                    None if early_rate is None or late_rate is None else late_rate - early_rate
                ),
            }
        )
    return sorted(output, key=lambda row: (row["delta_early_late"] is None, row["delta_early_late"]))


def _rollout_rows(task_rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rollout_rows:
        by_rollout[int(row["rollout"])].append(row)
    output = []
    for rollout, rows in sorted(by_rollout.items()):
        trajectories = sum(int(row["trajectories"]) for row in rows)
        positive = sum(int(row["positive_trajectories"]) for row in rows)
        output.append(
            {
                "rollout": rollout,
                "tasks": len(rows),
                "trajectories": trajectories,
                "positive_trajectories": positive,
                "trajectory_positive_rate": positive / trajectories if trajectories else 0.0,
                "samples": sum(int(row["samples"]) for row in rows),
                "positive_samples": sum(int(row["positive_samples"]) for row in rows),
                "all_zero_groups": sum(int(row["all_zero_groups"]) for row in rows),
                "mixed_groups": sum(int(row["mixed_groups"]) for row in rows),
                "all_positive_groups": sum(int(row["all_positive_groups"]) for row in rows),
                "turns_mean": _mean([row["turns_mean"] for row in rows if row["turns_mean"] is not None]),
                "output_tokens_mean": _mean(
                    [row["output_tokens_mean"] for row in rows if row["output_tokens_mean"] is not None]
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze(args: argparse.Namespace) -> int:
    files = _iter_dump_files(args)
    if not files:
        raise SystemExit("No numeric rollout dump files matched")
    torch = _import_torch()

    by_task_rollout: dict[tuple[str, int], TaskRolloutAgg] = {}
    for index, path in enumerate(files, start=1):
        rollout = int(path.stem)
        if args.progress:
            print(f"[{index}/{len(files)}] loading {path}", flush=True)
        data = _safe_torch_load(torch, path)
        samples = data.get("samples", data) if isinstance(data, dict) else data
        if not isinstance(samples, Iterable):
            continue
        for sample in samples:
            task = _task_key(sample)
            agg = by_task_rollout.setdefault((task, rollout), TaskRolloutAgg(task=task, rollout=rollout))
            agg.add(sample)

    task_rollout_rows = [agg.to_row() for agg in by_task_rollout.values()]
    task_rollout_rows.sort(key=lambda row: (row["task"], int(row["rollout"])))
    trends = _trend_rows(task_rollout_rows)
    rollout_summary = _rollout_rows(task_rollout_rows)

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "task_rollout_summary.csv", task_rollout_rows)
    _write_csv(output_dir / "task_trend_summary.csv", trends)
    _write_csv(output_dir / "rollout_timeseries.csv", rollout_summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_task_rollout_rows": len(task_rollout_rows),
                "num_tasks": len({row["task"] for row in task_rollout_rows}),
                "num_rollouts": len(rollout_summary),
                "total_trajectories": sum(row["trajectories"] for row in rollout_summary),
                "total_positive_trajectories": sum(row["positive_trajectories"] for row in rollout_summary),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("Wrote", output_dir / "task_rollout_summary.csv")
    print("Wrote", output_dir / "task_trend_summary.csv")
    print("Wrote", output_dir / "rollout_timeseries.csv")
    print("\nRollout trajectory positive rates")
    for row in rollout_summary:
        print(
            row["rollout"],
            "tasks",
            row["tasks"],
            "traj",
            row["trajectories"],
            "pos",
            row["positive_trajectories"],
            "rate",
            f"{row['trajectory_positive_rate']:.3f}",
            "zero_groups",
            row["all_zero_groups"],
        )

    print("\nLargest early-to-late drops")
    for row in trends[: min(args.print_limit, len(trends))]:
        print(
            row["task"],
            "appearances",
            row["appearances"],
            "first",
            f"{row['first_rate']:.3f}",
            "last",
            f"{row['last_rate']:.3f}",
            "early2",
            _fmt(row["early2_rate"]),
            "late2",
            _fmt(row["late2_rate"]),
            "delta",
            _fmt(row["delta_early_late"]),
        )

    print("\nLargest early-to-late gains")
    for row in reversed(trends[-min(args.print_limit, len(trends)) :]):
        print(
            row["task"],
            "appearances",
            row["appearances"],
            "first",
            f"{row['first_rate']:.3f}",
            "last",
            f"{row['last_rate']:.3f}",
            "early2",
            _fmt(row["early2_rate"]),
            "late2",
            _fmt(row["late2_rate"]),
            "delta",
            _fmt(row["delta_early_late"]),
        )
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.3f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", default="rollout_dumps")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-limit", type=int, default=20)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    return analyze(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
