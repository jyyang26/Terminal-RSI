from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from slime.utils.types import Sample
from slime_plugins.harbor.partial_reward import trajectory_grouped_progress_rewards


NUM_GPUS = 0


def _args(**overrides):
    values = {
        "reward_key": None,
        "advantage_estimator": "grpo",
        "rewards_normalization": False,
        "grpo_std_normalization": False,
        "harbor_partial_reward_max": 0.4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_ctrf(trial_dir: Path, *, passed: int, failed: int = 0, skipped: int = 0, other: int = 0) -> None:
    ctrf_path = trial_dir / "verifier" / "ctrf.json"
    ctrf_path.parent.mkdir(parents=True)
    total = passed + failed + skipped + other
    ctrf_path.write_text(
        json.dumps(
            {
                "results": {
                    "summary": {
                        "tests": total,
                        "passed": passed,
                        "failed": failed,
                        "skipped": skipped,
                        "other": other,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _turn_sample(
    index: int,
    *,
    group_index: int,
    sample_rank: int,
    turn_index: int,
    reward: float,
    trial_dir: Path | None,
) -> Sample:
    metadata = {
        "harbor_rollout_id": 1,
        "harbor_rollout_group_index": group_index,
        "harbor_sample_rank": sample_rank,
        "harbor_turn_index": turn_index,
        "harbor_sample_mode": "turn_grouped",
    }
    if trial_dir is not None:
        metadata["harbor_trial_uri"] = f"file://{trial_dir}"
    return Sample(
        index=index,
        group_index=group_index,
        tokens=[1, 2],
        response_length=1,
        loss_mask=[1],
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata=metadata,
    )


def test_ctrf_progress_adds_capped_partial_reward_to_unsolved_trial(tmp_path):
    trial = tmp_path / "trial-a"
    _write_ctrf(trial, passed=9, failed=2)
    samples = [
        _turn_sample(0, group_index=0, sample_rank=0, turn_index=0, reward=0.0, trial_dir=trial),
        _turn_sample(1, group_index=0, sample_rank=0, turn_index=1, reward=0.0, trial_dir=trial),
    ]

    raw_rewards, rewards = trajectory_grouped_progress_rewards(_args(), samples)

    expected = 0.4 * 9 / 11
    assert raw_rewards == pytest.approx([expected, expected])
    assert rewards == pytest.approx([expected, expected])
    assert samples[0].metadata["harbor_progress_score"] == pytest.approx(9 / 11)
    assert samples[0].metadata["harbor_progress_source"] == "ctrf"
    assert samples[0].metadata["harbor_progress_passed"] == 9
    assert samples[0].metadata["harbor_progress_total"] == 11
    assert samples[0].train_metadata["harbor_progress_score"] == pytest.approx(9 / 11)


def test_terminal_success_keeps_full_reward_even_with_partial_cap(tmp_path):
    trial = tmp_path / "trial-success"
    _write_ctrf(trial, passed=1, failed=0)
    sample = _turn_sample(0, group_index=0, sample_rank=0, turn_index=0, reward=1.0, trial_dir=trial)

    raw_rewards, rewards = trajectory_grouped_progress_rewards(_args(), [sample])

    assert raw_rewards == [1.0]
    assert rewards == [1.0]
    assert sample.metadata["harbor_progress_score"] == 1.0
    assert sample.metadata["harbor_progress_reward"] == 1.0


def test_missing_ctrf_has_zero_progress_without_timeout_reward(tmp_path):
    trial = tmp_path / "trial-missing"
    sample = _turn_sample(0, group_index=0, sample_rank=0, turn_index=0, reward=0.0, trial_dir=trial)

    raw_rewards, rewards = trajectory_grouped_progress_rewards(_args(), [sample])

    assert raw_rewards == [0.0]
    assert rewards == [0.0]
    assert sample.metadata["harbor_progress_score"] == 0.0
    assert sample.metadata["harbor_progress_source"] == "missing_ctrf"


def test_progress_rewards_normalize_by_trajectory_and_broadcast_to_turns(tmp_path):
    trial_a = tmp_path / "trial-a"
    trial_b = tmp_path / "trial-b"
    _write_ctrf(trial_a, passed=9, failed=1)
    _write_ctrf(trial_b, passed=0, failed=10)
    samples = [
        _turn_sample(0, group_index=0, sample_rank=0, turn_index=0, reward=0.0, trial_dir=trial_a),
        _turn_sample(1, group_index=0, sample_rank=0, turn_index=1, reward=0.0, trial_dir=trial_a),
        _turn_sample(2, group_index=0, sample_rank=1, turn_index=0, reward=0.0, trial_dir=trial_b),
    ]

    raw_rewards, rewards = trajectory_grouped_progress_rewards(
        _args(rewards_normalization=True),
        samples,
    )

    assert raw_rewards == pytest.approx([0.36, 0.36, 0.0])
    assert rewards == pytest.approx([0.18, 0.18, -0.18])
