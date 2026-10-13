from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from slime.utils.types import Sample
from slime_plugins.harbor.length_shaped_reward import trajectory_grouped_length_shaped_rewards


NUM_GPUS = 0


def _chunk_sample(
    index: int,
    group_index: int,
    sample_rank: int,
    chunk_index: int,
    reward: float,
    length: int,
) -> Sample:
    """One trajectory chunk; chunks of one trajectory share (group_index, sample_rank)."""
    return Sample(
        index=index,
        group_index=group_index,
        tokens=[0] * length,
        response_length=length,
        loss_mask=[1] * length,
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata={
            "harbor_rollout_id": 7,
            "harbor_rollout_group_index": group_index,
            "harbor_sample_rank": sample_rank,
            "harbor_trajectory_chunk_index": chunk_index,
            "harbor_sample_mode": "trajectory_retokenized",
        },
    )


def _args(rewards_normalization=False, grpo_std_normalization=False, lam=0.2):
    return SimpleNamespace(
        reward_key=None,
        advantage_estimator="gspo",
        rewards_normalization=rewards_normalization,
        grpo_std_normalization=grpo_std_normalization,
        harbor_length_reward_lambda=lam,
    )


def test_shorter_success_gets_higher_reward_failure_stays_zero():
    # One prompt group, 3 single-chunk trajectories: short success, long success, failure.
    samples = [
        _chunk_sample(0, group_index=0, sample_rank=0, chunk_index=0, reward=1.0, length=10),
        _chunk_sample(1, group_index=0, sample_rank=1, chunk_index=0, reward=1.0, length=30),
        _chunk_sample(2, group_index=0, sample_rank=2, chunk_index=0, reward=0.0, length=50),
    ]

    raw, processed = trajectory_grouped_length_shaped_rewards(_args(lam=0.2), samples)

    # Normalization OFF -> processed == raw == shaped reward.
    # shortest success: 1.0 * (1 + 0.2*(0.5-0)) = 1.1 ; longest: 1.0*(1 + 0.2*(0.5-1)) = 0.9
    assert raw == pytest.approx([1.1, 0.9, 0.0])
    assert processed == pytest.approx([1.1, 0.9, 0.0])
    assert raw[0] > raw[1] > raw[2]
    assert raw[2] == 0.0  # failure untouched


def test_multi_chunk_trajectory_aggregates_total_output_length():
    # Trajectory A = 2 chunks of len 5 (total 10); trajectory B = 1 chunk of len 30.
    # A is shorter overall -> A's chunks should both get the higher shaped reward.
    samples = [
        _chunk_sample(0, group_index=0, sample_rank=0, chunk_index=0, reward=1.0, length=5),
        _chunk_sample(1, group_index=0, sample_rank=0, chunk_index=1, reward=1.0, length=5),
        _chunk_sample(2, group_index=0, sample_rank=1, chunk_index=0, reward=1.0, length=30),
    ]

    raw, _ = trajectory_grouped_length_shaped_rewards(_args(lam=0.2), samples)

    # A's two chunks share the trajectory-level shaped reward (1.1); B gets 0.9.
    assert raw == pytest.approx([1.1, 1.1, 0.9])


def test_group_normalization_makes_failure_negative_and_short_success_largest():
    samples = [
        _chunk_sample(0, group_index=0, sample_rank=0, chunk_index=0, reward=1.0, length=10),
        _chunk_sample(1, group_index=0, sample_rank=1, chunk_index=0, reward=1.0, length=30),
        _chunk_sample(2, group_index=0, sample_rank=2, chunk_index=0, reward=0.0, length=50),
    ]

    raw, processed = trajectory_grouped_length_shaped_rewards(
        _args(rewards_normalization=True, grpo_std_normalization=False, lam=0.2), samples
    )

    # shaped = [1.1, 0.9, 0.0], mean = 2.0/3 -> mean-subtracted advantages.
    mean = (1.1 + 0.9 + 0.0) / 3.0
    assert processed == pytest.approx([1.1 - mean, 0.9 - mean, 0.0 - mean])
    assert processed[0] > processed[1] > processed[2]
    assert processed[2] < 0.0  # failure gets negative advantage


def test_each_prompt_group_shaped_independently():
    # Two independent prompt groups; lengths in one must not affect the other.
    samples = [
        _chunk_sample(0, group_index=0, sample_rank=0, chunk_index=0, reward=1.0, length=10),
        _chunk_sample(1, group_index=0, sample_rank=1, chunk_index=0, reward=1.0, length=30),
        _chunk_sample(2, group_index=1, sample_rank=0, chunk_index=0, reward=1.0, length=100),
        _chunk_sample(3, group_index=1, sample_rank=1, chunk_index=0, reward=1.0, length=200),
    ]

    raw, _ = trajectory_grouped_length_shaped_rewards(_args(lam=0.2), samples)

    # Each group ranks its own two successes -> shortest 1.1, longest 0.9 in BOTH groups.
    assert raw == pytest.approx([1.1, 0.9, 1.1, 0.9])


def test_lambda_zero_is_passthrough():
    samples = [
        _chunk_sample(0, group_index=0, sample_rank=0, chunk_index=0, reward=1.0, length=10),
        _chunk_sample(1, group_index=0, sample_rank=1, chunk_index=0, reward=1.0, length=30),
        _chunk_sample(2, group_index=0, sample_rank=2, chunk_index=0, reward=0.0, length=50),
    ]

    raw, _ = trajectory_grouped_length_shaped_rewards(_args(lam=0.0), samples)

    assert raw == pytest.approx([1.0, 1.0, 0.0])


def test_lone_success_in_group_keeps_terminal_reward():
    # Only one success in the group -> nothing to rank against -> terminal unchanged.
    samples = [
        _chunk_sample(0, group_index=0, sample_rank=0, chunk_index=0, reward=1.0, length=10),
        _chunk_sample(1, group_index=0, sample_rank=1, chunk_index=0, reward=0.0, length=50),
    ]

    raw, _ = trajectory_grouped_length_shaped_rewards(_args(lam=0.2), samples)

    assert raw == pytest.approx([1.0, 0.0])


def test_length_bonus_never_flips_success_below_failure():
    # Even the longest success (max length) must stay strictly above any failure.
    samples = [
        _chunk_sample(0, group_index=0, sample_rank=0, chunk_index=0, reward=1.0, length=1),
        _chunk_sample(1, group_index=0, sample_rank=1, chunk_index=0, reward=1.0, length=100000),
        _chunk_sample(2, group_index=0, sample_rank=2, chunk_index=0, reward=0.0, length=2),
    ]

    raw, _ = trajectory_grouped_length_shaped_rewards(_args(lam=0.2), samples)

    # longest success = 1.0 * (1 - 0.2*0.5) = 0.9 > 0.0 failure
    assert raw[1] == pytest.approx(0.9)
    assert raw[1] > raw[2]
    assert raw[2] == 0.0
