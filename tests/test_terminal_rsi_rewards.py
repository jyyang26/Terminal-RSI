"""Terminal-RSI reward functions: FDR (RPR-1) and SPR (RPR-2).

  FDR  R = C * P / D_f + (1 - C) * S      test_count_progress_rewards
  SPR  R = S + C * P / (P + D'_f)         success_smoothed_progress_rewards

P = passed CTRF checks, S = strict terminal success, C = valid CTRF report, D_f = the
task family's divisor (resolved from tests/private/manifest.json), falling back to the
global harbor_testcount_scale.
"""

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
from slime_plugins.harbor import partial_reward

# Imported via the module: a bare `test_count_progress_rewards` name would be collected by pytest.
fdr_rewards = partial_reward.test_count_progress_rewards
spr_rewards = partial_reward.success_smoothed_progress_rewards


NUM_GPUS = 0


def _args(**overrides):
    values = {
        "reward_key": None,
        "advantage_estimator": "ppo",
        "rewards_normalization": False,
        "grpo_std_normalization": False,
        "harbor_testcount_scale": 82.0,
        "harbor_family_testcount_scales": {"blender_scene_graph": 60.0, "ajv_schema": 3.5},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _task(tmp_path: Path, name: str, family_id: str | None) -> Path:
    task_dir = tmp_path / "tasks" / name
    manifest = task_dir / "tests" / "private" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"family_id": family_id} if family_id else {}), encoding="utf-8")
    return task_dir


def _trial(tmp_path: Path, name: str, *, passed: int | None, failed: int = 0) -> Path:
    trial_dir = tmp_path / "trials" / name
    trial_dir.mkdir(parents=True)
    if passed is not None:
        ctrf = trial_dir / "verifier" / "ctrf.json"
        ctrf.parent.mkdir(parents=True)
        summary = {"tests": passed + failed, "passed": passed, "failed": failed}
        ctrf.write_text(json.dumps({"results": {"summary": summary}}), encoding="utf-8")
    return trial_dir


def _sample(index: int, *, reward: float, trial_dir: Path, task_dir: Path) -> Sample:
    metadata = {
        "harbor_rollout_id": 1,
        "harbor_rollout_group_index": index,
        "harbor_sample_rank": 0,
        "harbor_turn_index": 0,
        "harbor_sample_mode": "trajectory_retokenized",
        "harbor_trial_uri": f"file://{trial_dir}",
        "harbor_task_entry": {"task": {"path": str(task_dir)}},
    }
    return Sample(
        index=index,
        group_index=index,
        tokens=[1, 2],
        response_length=1,
        loss_mask=[1],
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata=metadata,
    )


def test_fdr_uses_family_divisor_and_global_fallback(tmp_path):
    samples = [
        _sample(0, reward=0.0, trial_dir=_trial(tmp_path, "a", passed=30, failed=5),
                task_dir=_task(tmp_path, "blender-1", "blender_scene_graph")),
        _sample(1, reward=0.0, trial_dir=_trial(tmp_path, "b", passed=41, failed=1),
                task_dir=_task(tmp_path, "other-1", "family_not_in_table")),
    ]

    raw, processed = fdr_rewards(_args(), samples)

    assert raw == pytest.approx([30 / 60.0, 41 / 82.0])
    assert processed == raw  # PPO: no group normalisation
    assert samples[0].metadata["harbor_progress_passed"] == 30


def test_fdr_is_unbounded_and_can_rank_failure_above_success(tmp_path):
    # The pathology SPR fixes: many passed checks on a failed trajectory outscore a
    # genuine success with few checks.
    failed = _sample(0, reward=0.0, trial_dir=_trial(tmp_path, "f", passed=120, failed=3),
                     task_dir=_task(tmp_path, "blender-f", "blender_scene_graph"))
    solved = _sample(1, reward=1.0, trial_dir=_trial(tmp_path, "s", passed=20),
                     task_dir=_task(tmp_path, "blender-s", "blender_scene_graph"))

    raw, _ = fdr_rewards(_args(), [failed, solved])

    assert raw[0] == pytest.approx(2.0)
    assert raw[1] == pytest.approx(20 / 60.0)
    assert raw[0] > raw[1]


def test_spr_success_offset_dominates_partial_progress(tmp_path):
    failed = _sample(0, reward=0.0, trial_dir=_trial(tmp_path, "f", passed=120, failed=3),
                     task_dir=_task(tmp_path, "blender-f", "blender_scene_graph"))
    solved = _sample(1, reward=1.0, trial_dir=_trial(tmp_path, "s", passed=20),
                     task_dir=_task(tmp_path, "blender-s", "blender_scene_graph"))
    ajv = _sample(2, reward=0.0, trial_dir=_trial(tmp_path, "j", passed=7, failed=7),
                  task_dir=_task(tmp_path, "ajv-1", "ajv_schema"))

    raw, processed = spr_rewards(_args(), [failed, solved, ajv])

    assert raw == pytest.approx([120 / 180.0, 1.0 + 20 / 80.0, 7 / 10.5])
    assert processed == raw
    assert raw[0] < 1.0 <= raw[1]


def test_spr_without_ctrf_falls_back_to_strict_success(tmp_path):
    task = _task(tmp_path, "nofam", None)
    solved = _sample(0, reward=1.0, trial_dir=_trial(tmp_path, "s", passed=None), task_dir=task)
    failed = _sample(1, reward=0.0, trial_dir=_trial(tmp_path, "f", passed=None), task_dir=task)

    raw, _ = spr_rewards(_args(), [solved, failed])

    assert raw == pytest.approx([1.0, 0.0])
