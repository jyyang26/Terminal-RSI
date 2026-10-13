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
# Alias on import: the module-level name starts with 'test_' (CLI hook target),
# which pytest would otherwise collect as a test case.
from slime_plugins.harbor.partial_reward import (
    test_count_length_discount_rewards as length_discount_rewards,
)


SCALE = 20.0
L_FREE = 25.0
L_FULL = 50.0
ALPHA = 0.5
BETA = 0.1


def _args(**overrides):
    values = {
        "reward_key": None,
        "advantage_estimator": "ppo",  # -> _should_normalize False -> raw shaped returned
        "rewards_normalization": False,
        "grpo_std_normalization": False,
        "harbor_testcount_scale": SCALE,
        "harbor_length_discount_free_turns": L_FREE,
        "harbor_length_discount_full_turns": L_FULL,
        "harbor_length_discount_alpha": ALPHA,
        "harbor_length_discount_zero_beta": BETA,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_ctrf(trial_dir: Path, *, passed: int, failed: int = 0, skipped: int = 0, other: int = 0) -> None:
    ctrf_path = trial_dir / "verifier" / "ctrf.json"
    ctrf_path.parent.mkdir(parents=True)
    total = passed + failed + skipped + other
    ctrf_path.write_text(
        json.dumps({"results": {"summary": {
            "tests": total, "passed": passed, "failed": failed, "skipped": skipped, "other": other,
        }}}),
        encoding="utf-8",
    )


def _traj(index, *, group_index, reward, turns, trial_dir):
    metadata = {
        "harbor_rollout_id": 1,
        "harbor_rollout_group_index": group_index,
        "harbor_sample_rank": group_index,
        "harbor_sample_mode": "trajectory_retokenized",
        "harbor_num_turns": turns,
        "harbor_num_steps": turns,
        "harbor_trial_uri": f"file://{trial_dir}",
    }
    return Sample(
        index=index, group_index=group_index, tokens=[1, 2], response_length=1,
        loss_mask=[1], reward=reward, status=Sample.Status.COMPLETED, metadata=metadata,
    )


def test_free_zone_no_discount(tmp_path):
    # turns <= L_FREE: full correctness, no discount.
    t = tmp_path / "healthy"
    _write_ctrf(t, passed=5)
    s = _traj(0, group_index=0, reward=1.0, turns=20, trial_dir=t)  # 20 < 25
    raw, _ = length_discount_rewards(_args(), [s])
    assert raw == pytest.approx([5 / SCALE])  # 0.25, undiscounted
    assert s.metadata["harbor_length_turns_norm"] == pytest.approx(0.0)


def test_farming_is_unprofitable(tmp_path):
    # THE core property: a 5-test solve farmed to L_FULL scores BELOW a 4-test
    # healthy solve. Brushing an extra test by rambling must lose.
    t_farm = tmp_path / "farm5"
    t_healthy = tmp_path / "healthy4"
    _write_ctrf(t_farm, passed=5)
    _write_ctrf(t_healthy, passed=4)
    samples = [
        _traj(0, group_index=0, reward=1.0, turns=50, trial_dir=t_farm),     # 5-test, farmed
        _traj(1, group_index=1, reward=1.0, turns=20, trial_dir=t_healthy),  # 4-test, healthy
    ]
    raw, _ = length_discount_rewards(_args(), samples)
    farmed = (5 / SCALE) * (1 - ALPHA * 1.0)   # 0.25*0.5 = 0.125
    healthy = 4 / SCALE                          # 0.20
    assert raw == pytest.approx([farmed, healthy])
    assert raw[0] < raw[1]  # farming loses


def test_efficient_solve_beats_all(tmp_path):
    # 5-test solved at healthy length is the global best (full 0.25, no discount).
    t = tmp_path / "eff5"
    _write_ctrf(t, passed=5)
    s = _traj(0, group_index=0, reward=1.0, turns=25, trial_dir=t)  # exactly L_FREE
    raw, _ = length_discount_rewards(_args(), [s])
    assert raw == pytest.approx([5 / SCALE])  # 0.25


def test_linear_discount_midpoint(tmp_path):
    # turns at midpoint (37.5) -> norm 0.5 -> discount factor (1 - 0.5*0.5)=0.75.
    t = tmp_path / "mid"
    _write_ctrf(t, passed=6)
    s = _traj(0, group_index=0, reward=1.0, turns=int((L_FREE + L_FULL) / 2), trial_dir=t)  # 37
    raw, _ = length_discount_rewards(_args(), [s])
    norm = (37 - L_FREE) / (L_FULL - L_FREE)  # 0.48
    assert raw == pytest.approx([(6 / SCALE) * (1 - ALPHA * norm)])


def test_zero_pass_rambling_goes_negative(tmp_path):
    # passed==0 at full length: multiplication alone would give 0; BETA makes it negative.
    t = tmp_path / "zero_ramble"
    _write_ctrf(t, passed=0, failed=5)
    s = _traj(0, group_index=0, reward=0.0, turns=50, trial_dir=t)
    raw, _ = length_discount_rewards(_args(), [s])
    assert raw == pytest.approx([-BETA * 1.0])  # -0.1
    assert raw[0] < 0.0


def test_zero_pass_short_not_penalized(tmp_path):
    # passed==0 but short (free zone): no penalty (norm 0 -> 0).
    t = tmp_path / "zero_short"
    _write_ctrf(t, passed=0, failed=3)
    s = _traj(0, group_index=0, reward=0.0, turns=20, trial_dir=t)
    raw, _ = length_discount_rewards(_args(), [s])
    assert raw == pytest.approx([0.0])


def test_discount_clamped_beyond_full(tmp_path):
    # turns beyond L_FULL: discount clamped at ALPHA (norm 1), not more.
    t = tmp_path / "way_over"
    _write_ctrf(t, passed=8)
    s = _traj(0, group_index=0, reward=1.0, turns=120, trial_dir=t)
    raw, _ = length_discount_rewards(_args(), [s])
    assert raw == pytest.approx([(8 / SCALE) * (1 - ALPHA)])  # norm clamped to 1
    assert s.metadata["harbor_length_turns_norm"] == pytest.approx(1.0)


def test_missing_turns_no_discount(tmp_path):
    # No turns field -> treat as free zone (no discount), don't crash.
    t = tmp_path / "no_turns"
    _write_ctrf(t, passed=6)
    s = _traj(0, group_index=0, reward=1.0, turns=None, trial_dir=t)
    # strip the turns keys
    s.metadata.pop("harbor_num_turns", None)
    s.metadata.pop("harbor_num_steps", None)
    raw, _ = length_discount_rewards(_args(), [s])
    assert raw == pytest.approx([6 / SCALE])
