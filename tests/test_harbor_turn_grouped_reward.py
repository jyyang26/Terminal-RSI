from __future__ import annotations

import sys
import types
from types import SimpleNamespace

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from slime.utils.types import Sample
from slime_plugins.harbor.turn_grouped_reward import trajectory_grouped_turn_rewards


NUM_GPUS = 0


def _turn_sample(index: int, group_index: int, sample_rank: int, turn_index: int, reward: float) -> Sample:
    return Sample(
        index=index,
        group_index=group_index,
        tokens=[1, 2],
        response_length=1,
        loss_mask=[1],
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata={
            "harbor_rollout_id": 3,
            "harbor_rollout_group_index": group_index,
            "harbor_sample_rank": sample_rank,
            "harbor_turn_index": turn_index,
            "harbor_sample_mode": "turn_grouped",
        },
    )


def test_trajectory_grouped_turn_rewards_broadcasts_trajectory_advantage_to_all_turns():
    samples = [
        _turn_sample(0, group_index=0, sample_rank=0, turn_index=0, reward=0.0),
        _turn_sample(1, group_index=0, sample_rank=0, turn_index=1, reward=0.0),
        _turn_sample(2, group_index=0, sample_rank=1, turn_index=0, reward=1.0),
    ]
    args = SimpleNamespace(
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
    )

    raw_rewards, rewards = trajectory_grouped_turn_rewards(args, samples)

    assert raw_rewards == [0.0, 0.0, 1.0]
    assert rewards == [-0.5, -0.5, 0.5]


def test_trajectory_grouped_turn_rewards_normalizes_each_prompt_group_independently():
    samples = [
        _turn_sample(0, group_index=0, sample_rank=0, turn_index=0, reward=0.0),
        _turn_sample(1, group_index=0, sample_rank=1, turn_index=0, reward=1.0),
        _turn_sample(2, group_index=1, sample_rank=0, turn_index=0, reward=10.0),
        _turn_sample(3, group_index=1, sample_rank=1, turn_index=0, reward=14.0),
    ]
    args = SimpleNamespace(
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
    )

    _, rewards = trajectory_grouped_turn_rewards(args, samples)

    assert rewards == [-0.5, 0.5, -2.0, 2.0]
