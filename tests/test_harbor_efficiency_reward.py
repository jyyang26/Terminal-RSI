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
# Alias on import: the module-level name starts with 'test_' (it is a CLI
# --custom-reward-post-process-path target), which pytest would otherwise try
# to collect as a test case.
from slime_plugins.harbor.partial_reward import (
    test_count_efficiency_rewards as efficiency_rewards,
)


SCALE = 20.0
BETA = 0.05


def _args(**overrides):
    # PPO path: advantage_estimator="ppo" -> _should_normalize is False, so the
    # hook returns raw shaped rewards unchanged (base + efficiency bonus).
    values = {
        "reward_key": None,
        "advantage_estimator": "ppo",
        "rewards_normalization": False,
        "grpo_std_normalization": False,
        "harbor_testcount_scale": SCALE,
        "harbor_efficiency_bonus_beta": BETA,
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


def _traj_sample(
    index: int,
    *,
    group_index: int,
    reward: float,
    num_steps: int,
    trial_dir: Path,
) -> Sample:
    # trajectory_retokenized mode: one Sample per trajectory here (sample_rank
    # distinguishes trajectories within the rollout); harbor_num_steps is shared
    # across a trajectory's samples.
    metadata = {
        "harbor_rollout_id": 1,
        "harbor_rollout_group_index": group_index,
        "harbor_sample_rank": group_index,
        "harbor_sample_mode": "trajectory_retokenized",
        "harbor_num_steps": num_steps,
        "harbor_num_turns": num_steps,
        "harbor_trial_uri": f"file://{trial_dir}",
    }
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


def test_solved_bucket_shortest_gets_full_bonus_longest_gets_zero(tmp_path):
    # Two fully-solved trajectories with the SAME test count (10): the shorter
    # (20 steps) gets the full bonus, the longer (60 steps) gets zero.
    trial_short = tmp_path / "short"
    trial_long = tmp_path / "long"
    _write_ctrf(trial_short, passed=10)  # solved: passed==total==10
    _write_ctrf(trial_long, passed=10)
    samples = [
        _traj_sample(0, group_index=0, reward=1.0, num_steps=20, trial_dir=trial_short),
        _traj_sample(1, group_index=1, reward=1.0, num_steps=60, trial_dir=trial_long),
    ]

    raw, processed = efficiency_rewards(_args(), samples)

    base = 10 / SCALE  # 0.5
    assert raw == pytest.approx([base + BETA, base + 0.0])
    assert processed == pytest.approx([base + BETA, base + 0.0])
    assert samples[0].metadata["harbor_efficiency_norm"] == pytest.approx(1.0)
    assert samples[0].metadata["harbor_efficiency_bonus"] == pytest.approx(BETA)
    assert samples[1].metadata["harbor_efficiency_norm"] == pytest.approx(0.0)
    assert samples[1].metadata["harbor_efficiency_bonus"] == pytest.approx(0.0)


def test_unsolved_trajectory_never_gets_bonus(tmp_path):
    # Unsolved (terminal < 1.0) trajectory: base test-count reward only, no bonus,
    # even if it is the shortest.
    trial = tmp_path / "unsolved"
    _write_ctrf(trial, passed=3, failed=7)  # solved==False (passed != total)
    sample = _traj_sample(0, group_index=0, reward=0.0, num_steps=5, trial_dir=trial)

    raw, processed = efficiency_rewards(_args(), [sample])

    assert raw == pytest.approx([3 / SCALE])
    assert processed == pytest.approx([3 / SCALE])
    # Unsolved trajectories are never annotated with an efficiency bonus.
    assert "harbor_efficiency_bonus" not in sample.metadata


def test_singleton_bucket_gets_full_bonus(tmp_path):
    # A solved trajectory with no same-test-count peer is the "unique best":
    # norm=1.0, full bonus (human-decided).
    trial = tmp_path / "solo"
    _write_ctrf(trial, passed=7)
    sample = _traj_sample(0, group_index=0, reward=1.0, num_steps=99, trial_dir=trial)

    raw, processed = efficiency_rewards(_args(), [sample])

    base = 7 / SCALE
    assert raw == pytest.approx([base + BETA])
    assert sample.metadata["harbor_efficiency_norm"] == pytest.approx(1.0)
    assert sample.metadata["harbor_efficiency_bonus"] == pytest.approx(BETA)


def test_different_test_counts_are_separate_buckets(tmp_path):
    # Two solved trajectories with DIFFERENT test counts do NOT compete: each is
    # a singleton bucket -> both get the full bonus (they are not compared on
    # step length across difficulty).
    trial_small = tmp_path / "small"
    trial_big = tmp_path / "big"
    _write_ctrf(trial_small, passed=4)  # bucket total=4
    _write_ctrf(trial_big, passed=30)  # bucket total=30
    samples = [
        _traj_sample(0, group_index=0, reward=1.0, num_steps=10, trial_dir=trial_small),
        _traj_sample(1, group_index=1, reward=1.0, num_steps=80, trial_dir=trial_big),
    ]

    raw, _ = efficiency_rewards(_args(), samples)

    assert raw == pytest.approx([4 / SCALE + BETA, 30 / SCALE + BETA])


def test_linear_interpolation_for_middle_trajectory(tmp_path):
    # Three solved same-count trajectories: steps 10 / 30 / 50 ->
    # norm = 1.0 / 0.5 / 0.0.
    trials = []
    for i, steps in enumerate((10, 30, 50)):
        t = tmp_path / f"t{i}"
        _write_ctrf(t, passed=10)
        trials.append((t, steps))
    samples = [
        _traj_sample(i, group_index=i, reward=1.0, num_steps=steps, trial_dir=t)
        for i, (t, steps) in enumerate(trials)
    ]

    raw, _ = efficiency_rewards(_args(), samples)

    base = 10 / SCALE
    assert raw == pytest.approx([base + BETA * 1.0, base + BETA * 0.5, base + BETA * 0.0])


def test_base_reward_unchanged_matches_test_count_shaped(tmp_path):
    # Sanity: base component equals passed/SCALE (the plain test-count reward)
    # for a solved singleton before the bonus is layered on.
    trial = tmp_path / "base"
    _write_ctrf(trial, passed=12)
    sample = _traj_sample(0, group_index=0, reward=1.0, num_steps=40, trial_dir=trial)

    raw, _ = efficiency_rewards(_args(), [sample])

    # base = 12/20 = 0.6, plus full bonus (singleton) 0.05.
    assert raw == pytest.approx([12 / SCALE + BETA])
    assert sample.metadata["harbor_progress_passed"] == 12
    assert sample.metadata["harbor_progress_total"] == 12
