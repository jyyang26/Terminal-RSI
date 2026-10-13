from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from pathlib import Path

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
    test_count_length_penalty_rewards as length_penalty_rewards,
)


SCALE = 20.0
LAMBDA = 0.03
BUDGET = 40000.0


def _args(**overrides):
    # PPO path: advantage_estimator="ppo" -> _should_normalize is False, so the
    # hook returns raw shaped rewards unchanged (base - length penalty).
    values = {
        "reward_key": None,
        "advantage_estimator": "ppo",
        "rewards_normalization": False,
        "grpo_std_normalization": False,
        "harbor_testcount_scale": SCALE,
        "harbor_length_penalty_lambda": LAMBDA,
        "harbor_length_token_budget": BUDGET,
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
    output_tokens: int | None,
    trial_dir: Path,
) -> Sample:
    metadata = {
        "harbor_rollout_id": 1,
        "harbor_rollout_group_index": group_index,
        "harbor_sample_rank": group_index,
        "harbor_sample_mode": "trajectory_retokenized",
        "harbor_trial_uri": f"file://{trial_dir}",
    }
    if output_tokens is not None:
        metadata["harbor_total_output_tokens"] = output_tokens
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


def test_penalty_grows_with_output_tokens(tmp_path):
    # Same correctness (5 solved tests), different output lengths: the longer
    # trajectory is penalized more, so its reward is strictly lower.
    trial_short = tmp_path / "short"
    trial_long = tmp_path / "long"
    _write_ctrf(trial_short, passed=5)
    _write_ctrf(trial_long, passed=5)
    samples = [
        _traj_sample(0, group_index=0, reward=1.0, output_tokens=4000, trial_dir=trial_short),
        _traj_sample(1, group_index=1, reward=1.0, output_tokens=20000, trial_dir=trial_long),
    ]

    raw, processed = length_penalty_rewards(_args(), samples)

    base = 5 / SCALE  # 0.25
    exp_short = base - LAMBDA * (4000 / BUDGET)   # 0.25 - 0.003 = 0.247
    exp_long = base - LAMBDA * (20000 / BUDGET)   # 0.25 - 0.015 = 0.235
    assert raw == pytest.approx([exp_short, exp_long])
    assert processed == pytest.approx([exp_short, exp_long])
    assert raw[0] > raw[1]  # shorter wins at equal correctness
    assert samples[1].metadata["harbor_length_penalty"] == pytest.approx(LAMBDA * 0.5)
    assert samples[0].metadata["harbor_length_output_tokens"] == 4000


def test_correctness_strictly_dominates_length(tmp_path):
    # Lexicographic guarantee: 5 tests in the LONGEST trajectory must still beat
    # 4 tests in the SHORTEST — passing one more test (+0.05) always outweighs
    # any length saving (<= lambda = 0.03).
    trial_more = tmp_path / "more_tests_long"
    trial_fewer = tmp_path / "fewer_tests_short"
    _write_ctrf(trial_more, passed=5)
    _write_ctrf(trial_fewer, passed=4)
    samples = [
        _traj_sample(0, group_index=0, reward=1.0, output_tokens=BUDGET * 2, trial_dir=trial_more),  # max penalty
        _traj_sample(1, group_index=1, reward=0.0, output_tokens=1, trial_dir=trial_fewer),          # ~no penalty
    ]

    raw, _ = length_penalty_rewards(_args(), samples)

    more_tests = 5 / SCALE - LAMBDA * 1.0   # 0.25 - 0.03 = 0.22 (clamped at budget)
    fewer_tests = 4 / SCALE - LAMBDA * (1 / BUDGET)  # ~0.20
    assert raw[0] == pytest.approx(more_tests)
    assert raw[0] > raw[1]  # more correct beats shorter-but-less-correct


def test_penalty_clamped_at_budget(tmp_path):
    # Tokens beyond BUDGET do not increase the penalty past lambda (bounded).
    trial = tmp_path / "over_budget"
    _write_ctrf(trial, passed=2)
    sample = _traj_sample(0, group_index=0, reward=0.0, output_tokens=int(BUDGET * 5), trial_dir=trial)

    raw, _ = length_penalty_rewards(_args(), [sample])

    assert raw == pytest.approx([2 / SCALE - LAMBDA * 1.0])
    assert sample.metadata["harbor_length_norm"] == pytest.approx(1.0)


def test_zero_pass_long_trajectory_goes_negative(tmp_path):
    # The disease run#6 targets: a zero-pass trajectory that rambled to ~max output
    # now scores NEGATIVE (base 0 minus a real penalty), unlike run#5 where it was 0.
    trial = tmp_path / "zero_rambling"
    _write_ctrf(trial, passed=0, failed=5)
    sample = _traj_sample(0, group_index=0, reward=0.0, output_tokens=38000, trial_dir=trial)

    raw, _ = length_penalty_rewards(_args(), [sample])

    exp = 0.0 - LAMBDA * (38000 / BUDGET)  # -0.0285
    assert raw == pytest.approx([exp])
    assert raw[0] < 0.0


def test_missing_output_tokens_applies_zero_penalty(tmp_path):
    # If the token field is absent, fall back to zero penalty (base only) rather
    # than crash.
    trial = tmp_path / "no_tokens"
    _write_ctrf(trial, passed=6)
    sample = _traj_sample(0, group_index=0, reward=1.0, output_tokens=None, trial_dir=trial)

    raw, _ = length_penalty_rewards(_args(), [sample])

    assert raw == pytest.approx([6 / SCALE])
    assert sample.metadata["harbor_length_norm"] == pytest.approx(0.0)
    assert sample.metadata["harbor_length_penalty"] == pytest.approx(0.0)


def test_base_reward_matches_test_count_shaped(tmp_path):
    # Sanity: base component equals passed/SCALE before the penalty is applied.
    trial = tmp_path / "base"
    _write_ctrf(trial, passed=12)
    sample = _traj_sample(0, group_index=0, reward=1.0, output_tokens=0, trial_dir=trial)

    raw, _ = length_penalty_rewards(_args(), [sample])

    # tokens=0 -> zero penalty -> raw == base.
    assert raw == pytest.approx([12 / SCALE])
    assert sample.metadata["harbor_progress_passed"] == 12
    assert sample.metadata["harbor_progress_total"] == 12
