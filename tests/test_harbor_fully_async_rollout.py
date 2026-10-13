from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

if "ray" not in sys.modules:
    ray_stub = types.ModuleType("ray")
    ray_stub._private = SimpleNamespace(services=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    sys.modules["ray"] = ray_stub

from slime.utils.types import Sample
from slime_plugins.harbor import fully_async_rollout


def _train_sample(index: int) -> Sample:
    return Sample(
        index=index,
        tokens=[index, 100],
        response_length=1,
        loss_mask=[1],
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )


class _DataSource:
    def __init__(self):
        self.next_index = 0
        self.requests = []

    def get_samples(self, num_samples):
        self.requests.append(num_samples)
        groups = []
        for _ in range(num_samples):
            sample = Sample(index=self.next_index, group_index=self.next_index)
            self.next_index += 1
            groups.append([sample])
        return groups

    def add_samples(self, samples):
        raise AssertionError("Harbor fully-async rollout should not requeue incomplete trial groups")


def _args(**overrides):
    values = dict(
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        n_samples_per_prompt=1,
        harbor_require_full_rollout_groups=True,
        harbor_rollout_group_attempt_multiplier=4,
        harbor_num_concurrent_trials=2,
        harbor_retry_missing_samples=0,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        use_rollout_logprobs=False,
        harbor_sample_mode="last_turn",
        harbor_group_unit=None,
        start_rollout_id=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_harbor_fully_async_rollout_drops_incomplete_groups_and_backfills(monkeypatch):
    async def fake_run_one_group(
        args,
        rollout_id,
        trial_group_index,
        base_group,
        semaphore,
        *,
        retry_missing_samples,
    ):
        if trial_group_index == 0:
            return [], {"converted": 0, "trajectories": 0, "failed": 0, "missing": 1}
        sample = _train_sample(trial_group_index)
        sample.metadata = {
            "harbor_rollout_id": rollout_id,
            "harbor_rollout_group_index": trial_group_index,
            "harbor_sample_rank": 0,
        }
        return [sample], {"converted": 1, "trajectories": 1, "failed": 0, "missing": 0}

    monkeypatch.setattr(fully_async_rollout, "_run_one_group", fake_run_one_group)
    monkeypatch.setattr(fully_async_rollout, "_global_worker", None)

    data_source = _DataSource()
    try:
        output_groups, metrics = asyncio.run(
            asyncio.wait_for(
                fully_async_rollout._generate_rollout_async(_args(), 5, data_source),
                timeout=5.0,
            )
        )
    finally:
        fully_async_rollout._stop_global_worker()

    samples = [sample for group in output_groups for sample in group]
    assert [sample.index for sample in samples] == [1, 2]
    assert [sample.metadata["harbor_rollout_id"] for sample in samples] == [5, 5]
    assert metrics["harbor/fully_async_enabled"] == 1
    assert metrics["harbor/num_incomplete_groups_dropped"] == 1
    assert metrics["harbor/num_accepted_groups"] == 2
    assert data_source.requests[:2] == [1, 1]


def test_harbor_fully_async_rollout_accepts_partial_groups(monkeypatch):
    async def fake_run_one_group(
        args,
        rollout_id,
        trial_group_index,
        base_group,
        semaphore,
        *,
        retry_missing_samples,
    ):
        group = []
        for offset in range(3):
            sample = _train_sample(trial_group_index * 10 + offset)
            sample.metadata = {
                "harbor_rollout_id": rollout_id,
                "harbor_rollout_group_index": trial_group_index,
                "harbor_sample_rank": offset,
            }
            group.append(sample)
        return group, {
            "converted": 3,
            "trajectories": 3,
            "failed": 0,
            "missing": 0,
            "partial_cutoff": 1,
            "partial_min_units": 3,
            "cancelled_tail_trials": 1,
        }

    monkeypatch.setattr(fully_async_rollout, "_run_one_group", fake_run_one_group)
    monkeypatch.setattr(fully_async_rollout, "_global_worker", None)

    try:
        output_groups, metrics = asyncio.run(
            asyncio.wait_for(
                fully_async_rollout._generate_rollout_async(
                    _args(
                        rollout_batch_size=1,
                        over_sampling_batch_size=1,
                        harbor_fully_async_max_concurrent_groups=1,
                        n_samples_per_prompt=4,
                        harbor_partial_group_min_units=3,
                    ),
                    5,
                    _DataSource(),
                ),
                timeout=5.0,
            )
        )
    finally:
        fully_async_rollout._stop_global_worker()

    samples = [sample for group in output_groups for sample in group]
    assert [sample.index for sample in samples] == [0, 1, 2]
    assert metrics["harbor/num_partial_groups_accepted"] == 1
    assert metrics["harbor/num_partial_tail_trials_cancelled"] == 1
    assert metrics["harbor/partial_group_min_units"] == 3


def test_harbor_fully_async_worker_stop_cancels_active_groups(monkeypatch):
    started = threading.Event()
    cancelled = threading.Event()

    async def hanging_run_one_group(
        args,
        rollout_id,
        trial_group_index,
        base_group,
        semaphore,
        *,
        retry_missing_samples,
    ):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(fully_async_rollout, "_run_one_group", hanging_run_one_group)

    worker = fully_async_rollout.HarborFullyAsyncRolloutWorker(
        _args(
            rollout_batch_size=1,
            over_sampling_batch_size=1,
            harbor_fully_async_max_concurrent_groups=1,
            harbor_fully_async_shutdown_drain_timeout_sec=0.0,
            harbor_fully_async_shutdown_join_timeout_sec=5.0,
        ),
        _DataSource(),
    )
    worker.start()
    try:
        assert started.wait(timeout=2.0)
        worker.stop()
        assert cancelled.wait(timeout=2.0)
        assert worker.worker_thread is not None
        assert not worker.worker_thread.is_alive()
    finally:
        worker.stop()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
