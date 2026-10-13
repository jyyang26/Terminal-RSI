#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch


def _get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _metadata(sample: Any) -> dict[str, Any]:
    return _get(sample, "metadata", {}) or {}


def _trajectory_key(sample: Any) -> tuple[Any, Any, Any]:
    meta = _metadata(sample)
    return (
        meta.get("harbor_rollout_id"),
        meta.get("harbor_rollout_group_index", _get(sample, "group_index")),
        meta.get("harbor_sample_rank", meta.get("harbor_trial_sample_rank", _get(sample, "index"))),
    )


def _group_key(sample: Any) -> tuple[Any, Any, Any]:
    meta = _metadata(sample)
    return (
        meta.get("harbor_rollout_id"),
        meta.get("harbor_rollout_group_index", _get(sample, "group_index")),
        meta.get("harbor_task_name"),
    )


def _expand_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem))
        else:
            expanded.append(path)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    groups: dict[tuple[Any, Any, Any], dict[tuple[Any, Any, Any], dict[str, Any]]] = defaultdict(dict)
    turn_samples = 0
    for path in _expand_paths(args.paths):
        dump = torch.load(path, map_location="cpu", weights_only=False)
        samples = dump.get("samples", []) if isinstance(dump, dict) else dump
        turn_samples += len(samples)
        for sample in samples:
            traj_key = _trajectory_key(sample)
            group_key = _group_key(sample)
            meta = _metadata(sample)
            entry = groups[group_key].setdefault(
                traj_key,
                {
                    "reward": float(_get(sample, "reward", 0.0) or 0.0),
                    "status": _status_value(_get(sample, "status")),
                    "turns": 0,
                    "task": meta.get("harbor_task_name"),
                    "length_finish_turns": int(meta.get("harbor_length_finish_turns", 0) or 0),
                    "max_total_length": 0,
                },
            )
            entry["turns"] += 1
            entry["max_total_length"] = max(entry["max_total_length"], len(_get(sample, "tokens", []) or []))

    reward_pattern_counts = Counter()
    status_pattern_counts = Counter()
    task_group_counts = Counter()
    task_zero_groups = Counter()
    task_mixed_groups = Counter()
    group_sizes = []
    turn_counts = []
    max_lengths = []
    examples = []

    for group_key, trajectories in groups.items():
        rewards = [entry["reward"] for entry in trajectories.values()]
        statuses = Counter(entry["status"] for entry in trajectories.values())
        task = group_key[2]
        group_sizes.append(len(trajectories))
        turn_counts.extend(entry["turns"] for entry in trajectories.values())
        max_lengths.extend(entry["max_total_length"] for entry in trajectories.values())
        task_group_counts[task] += 1

        if all(reward == 0.0 for reward in rewards):
            reward_pattern = "all_zero"
            task_zero_groups[task] += 1
        elif all(reward == 1.0 for reward in rewards):
            reward_pattern = "all_one"
        else:
            reward_pattern = "mixed"
            task_mixed_groups[task] += 1
        reward_pattern_counts[reward_pattern] += 1

        status_pattern = ",".join(f"{key}:{value}" for key, value in sorted(statuses.items()))
        status_pattern_counts[status_pattern] += 1
        if len(examples) < args.top:
            examples.append((group_key, len(trajectories), Counter(rewards), statuses))

    print("turn_samples", turn_samples)
    print("groups", len(groups))
    print("reward_pattern_counts", dict(reward_pattern_counts))
    print("status_pattern_counts_top", status_pattern_counts.most_common(args.top))
    if group_sizes:
        print("group_size_min_mean_max", min(group_sizes), mean(group_sizes), max(group_sizes))
    if turn_counts:
        print("trajectory_turns_min_mean_max", min(turn_counts), mean(turn_counts), max(turn_counts))
    if max_lengths:
        print("trajectory_max_total_length_min_mean_max", min(max_lengths), mean(max_lengths), max(max_lengths))
    print("task_group_counts_top", task_group_counts.most_common(args.top))
    print("task_zero_group_counts_top", task_zero_groups.most_common(args.top))
    print("task_mixed_group_counts_top", task_mixed_groups.most_common(args.top))
    print("examples")
    for item in examples:
        print(item)


if __name__ == "__main__":
    main()
