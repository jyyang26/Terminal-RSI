from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Any

from slime.utils.types import Sample
from slime_plugins.harbor.turn_grouped_reward import (
    _group_and_trajectory_key,
    _normalize_rewards,
    _reward_value,
    _should_normalize,
)


DEFAULT_LENGTH_REWARD_LAMBDA = 0.2


def trajectory_grouped_length_shaped_rewards(
    args, samples: list[Sample], **kwargs
) -> tuple[list[float], list[float]]:
    """Length-aware reward shaping, then normalize by Harbor trajectory group.

    Correctness stays dominant; output length only acts as a tie-breaker *within*
    the set of successful trajectories of a prompt group:

      * A trajectory is "successful" iff its terminal reward > 0.
      * Among the successful trajectories of one prompt group we rank by total
        output length (sum of ``effective_response_length`` over all chunks of
        the trajectory). The shortest gets ``terminal * (1 + lambda * 0.5)``,
        the longest ``terminal * (1 - lambda * 0.5)``, linearly interpolated:

            shaped = terminal * (1 + lambda * (0.5 - norm_len))
            norm_len = (len - min_len) / max(max_len - min_len, 1)   in [0, 1]

        So the length bonus is bounded to +/- lambda/2 of the terminal reward and
        can never flip a success below a failure (failures are exactly 0).
      * Failed trajectories (terminal reward <= 0) keep their reward unchanged.

    The shaped reward is then run through the same GRPO/GSPO group normalization
    used by ``turn_grouped_reward`` (mean-subtract, optional std-divide), one
    advantage per trajectory broadcast to every chunk/turn of that trajectory.

    ``lambda`` is read from ``args.harbor_length_reward_lambda`` or the
    ``HARBOR_LENGTH_REWARD_LAMBDA`` env var (default 0.2). Set lambda=0 to make
    this a pass-through equal to the plain binary reward (useful for A/B).

    Notes:
      * This hook REPLACES slime's default group normalization (which is skipped
        whenever ``--custom-reward-post-process-path`` is set), so it must do its
        own normalization -- mirroring ``turn_grouped_reward`` / ``partial_reward``.
      * Length is the trained, loss-masked output length (``effective_response_length``),
        summed across trajectory chunks so multi-chunk ``trajectory_retokenized``
        samples are measured by the whole trajectory's output, not per chunk.
    """

    lam = _length_reward_lambda(args)

    # Group samples: prompt-group -> trajectory -> [sample indices].
    group_to_trajectories: "OrderedDict[Any, OrderedDict[Any, list[int]]]" = OrderedDict()
    for index, sample in enumerate(samples):
        group_key, trajectory_key = _group_and_trajectory_key(sample)
        trajectories = group_to_trajectories.setdefault(group_key, OrderedDict())
        trajectories.setdefault(trajectory_key, []).append(index)

    # raw_rewards: the *shaped* per-sample reward BEFORE group normalization.
    # We report shaped reward as raw so wandb's raw_reward reflects what training
    # actually optimizes pre-advantage; the terminal binary value is preserved in
    # sample metadata for auditing.
    raw_rewards = [0.0] * len(samples)

    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())

        # One terminal reward + one total output length per trajectory.
        terminal_rewards: list[float] = []
        total_lengths: list[float] = []
        for _, indices in trajectory_items:
            terminal_rewards.append(_reward_value(args, samples[indices[0]]))
            total_lengths.append(float(_trajectory_output_length(samples, indices)))

        shaped_rewards = _shape_group_rewards(terminal_rewards, total_lengths, lam)

        for (_, indices), terminal, total_len, shaped in zip(
            trajectory_items, terminal_rewards, total_lengths, shaped_rewards, strict=True
        ):
            for index in indices:
                raw_rewards[index] = shaped
                _annotate_sample(samples[index], terminal, total_len, shaped, lam)

    if not _should_normalize(args):
        return raw_rewards, raw_rewards

    processed = [0.0] * len(samples)
    for trajectories in group_to_trajectories.values():
        trajectory_items = list(trajectories.items())
        trajectory_rewards = [raw_rewards[indices[0]] for _, indices in trajectory_items]
        trajectory_advantages = _normalize_rewards(args, trajectory_rewards)
        for (_, indices), advantage in zip(trajectory_items, trajectory_advantages, strict=True):
            for index in indices:
                processed[index] = advantage

    return raw_rewards, processed


# Public alias matching the convention of turn_grouped_reward / partial_reward.
post_process_rewards = trajectory_grouped_length_shaped_rewards


def _shape_group_rewards(
    terminal_rewards: list[float], total_lengths: list[float], lam: float
) -> list[float]:
    """Apply the length bonus to successful trajectories within one prompt group."""

    shaped = list(terminal_rewards)
    if lam <= 0.0:
        return shaped

    success_indices = [i for i, r in enumerate(terminal_rewards) if r > 0.0]
    if len(success_indices) < 2:
        # 0 or 1 success in the group -> no relative length signal to give.
        # (A lone success keeps its terminal reward; nothing to rank against.)
        return shaped

    success_lengths = [total_lengths[i] for i in success_indices]
    min_len = min(success_lengths)
    max_len = max(success_lengths)
    span = max_len - min_len
    if span <= 0:
        # All successful trajectories have identical length -> no tie-break.
        return shaped

    for i in success_indices:
        norm_len = (total_lengths[i] - min_len) / span  # 0 (shortest) .. 1 (longest)
        # shortest -> 1 + lam*0.5 ; longest -> 1 - lam*0.5
        shaped[i] = terminal_rewards[i] * (1.0 + lam * (0.5 - norm_len))

    return shaped


def _trajectory_output_length(samples: list[Sample], indices: list[int]) -> int:
    """Total trained output length of a trajectory = sum over its chunks/turns.

    Uses ``effective_response_length`` (loss-masked output tokens) so masked
    context-carry tokens in retokenized chunks are not counted as "answer length".
    """

    total = 0
    for index in indices:
        sample = samples[index]
        length = getattr(sample, "effective_response_length", None)
        if length is None:
            length = getattr(sample, "response_length", 0) or 0
        total += int(length)
    return total


def _annotate_sample(
    sample: Sample, terminal_reward: float, total_length: float, shaped_reward: float, lam: float
) -> None:
    shaping_metadata = {
        "harbor_terminal_reward": terminal_reward,
        "harbor_length_reward_shaped": shaped_reward,
        "harbor_length_reward_total_output_len": total_length,
        "harbor_length_reward_lambda": lam,
    }
    metadata = dict(getattr(sample, "metadata", None) or {})
    metadata.update(shaping_metadata)
    sample.metadata = metadata

    train_metadata = dict(getattr(sample, "train_metadata", None) or {})
    train_metadata.update(shaping_metadata)
    sample.train_metadata = train_metadata


def _length_reward_lambda(args) -> float:
    value = getattr(args, "harbor_length_reward_lambda", None)
    if value is None:
        value = os.environ.get("HARBOR_LENGTH_REWARD_LAMBDA", DEFAULT_LENGTH_REWARD_LAMBDA)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_LENGTH_REWARD_LAMBDA
    if not math.isfinite(value) or value < 0.0:
        value = DEFAULT_LENGTH_REWARD_LAMBDA
    return value
