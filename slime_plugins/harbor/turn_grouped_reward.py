from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from typing import Any

from slime.utils.types import Sample


def trajectory_grouped_turn_rewards(args, samples: list[Sample], **kwargs) -> tuple[list[float], list[float]]:
    """Normalize Harbor rewards by trajectory group, then broadcast to turns.

    ``harbor_sample_mode=turn_grouped`` and ``harbor_sample_mode=step`` expand
    one Terminus trajectory into multiple slime samples, one per LLM call. GRPO
    grouping must still be done over trajectories, not over the expanded step
    samples. This hook computes one normalized advantage per trajectory and
    assigns that value unchanged to every step belonging to the same trajectory.
    """

    raw_rewards = [_reward_value(args, sample) for sample in samples]
    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


post_process_rewards = trajectory_grouped_turn_rewards


def _reward_value(args, sample: Sample) -> float:
    reward = getattr(sample, "reward", 0.0)
    reward_key = getattr(args, "reward_key", None)
    if reward_key:
        reward = reward[reward_key]
    return float(reward)


def _should_normalize(args) -> bool:
    return (
        getattr(args, "advantage_estimator", "grpo") in {"grpo", "gspo", "reinforce_plus_plus_baseline"}
        and bool(getattr(args, "rewards_normalization", True))
    )


def _normalize_rewards(args, rewards: list[float]) -> list[float]:
    if not rewards:
        return []

    mean = sum(rewards) / len(rewards)
    advantages = [reward - mean for reward in rewards]
    if getattr(args, "advantage_estimator", "grpo") not in {"grpo", "gspo"}:
        return advantages
    if not bool(getattr(args, "grpo_std_normalization", True)):
        return advantages

    std = statistics.stdev(advantages) if len(advantages) > 1 else 0.0
    if not math.isfinite(std):
        std = 0.0
    return [advantage / (std + 1e-6) for advantage in advantages]


def _group_and_trajectory_key(sample: Sample) -> tuple[Any, Any]:
    metadata = getattr(sample, "metadata", None) or {}
    rollout_id = metadata.get("harbor_rollout_id")
    group_index = metadata.get("harbor_rollout_group_index", getattr(sample, "group_index", None))
    sample_rank = metadata.get(
        "harbor_sample_rank",
        metadata.get("harbor_trial_sample_rank", getattr(sample, "index", None)),
    )
    group_key = (rollout_id, group_index)
    trajectory_key = (rollout_id, group_index, sample_rank)
    return group_key, trajectory_key
