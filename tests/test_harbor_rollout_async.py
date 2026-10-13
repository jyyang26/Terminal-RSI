from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import types
from types import SimpleNamespace

import pytest

NUM_GPUS = 0

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
from slime_plugins.harbor import rollout


def _train_sample(index: int) -> Sample:
    return Sample(
        index=index,
        tokens=[index, 100],
        response_length=1,
        loss_mask=[1],
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )


def test_run_one_group_consumes_samples_as_they_complete(monkeypatch):
    async def fake_run_one_with_semaphore(args, rollout_id, group_index, sample_rank, base_sample, semaphore):
        if base_sample.index == 0:
            await asyncio.sleep(0.05)
        if base_sample.index == 1 and sample_rank == 1:
            return []
        return [_train_sample(base_sample.index)]

    monkeypatch.setattr(rollout, "_run_one_with_semaphore", fake_run_one_with_semaphore)

    base_group = [Sample(index=0), Sample(index=1)]
    samples, stats = asyncio.run(
        rollout._run_one_group(
            SimpleNamespace(harbor_fail_on_trial_error=False),
            rollout_id=0,
            trial_group_index=0,
            base_group=base_group,
            semaphore=asyncio.Semaphore(2),
            retry_missing_samples=1,
        )
    )

    assert [sample.index for sample in samples] == [1, 0]
    assert stats == {"converted": 2, "trajectories": 2, "failed": 0, "missing": 1}


def test_run_one_group_partial_cutoff_cancels_tail_trials(monkeypatch):
    cancelled = []

    async def fake_run_one_with_semaphore(args, rollout_id, group_index, sample_rank, base_sample, semaphore):
        if base_sample.index < 2:
            return [_train_sample(base_sample.index)]
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(base_sample.index)
            raise

    monkeypatch.setattr(rollout, "_run_one_with_semaphore", fake_run_one_with_semaphore)

    base_group = [Sample(index=i) for i in range(4)]
    samples, stats = asyncio.run(
        rollout._run_one_group(
            SimpleNamespace(
                harbor_fail_on_trial_error=False,
                harbor_partial_group_min_units=2,
                harbor_sample_mode="last_turn",
                harbor_group_unit=None,
            ),
            rollout_id=0,
            trial_group_index=0,
            base_group=base_group,
            semaphore=asyncio.Semaphore(4),
            retry_missing_samples=0,
        )
    )

    assert sorted(sample.index for sample in samples) == [0, 1]
    assert sorted(cancelled) == [2, 3]
    assert stats["partial_cutoff"] == 1
    assert stats["partial_min_units"] == 2
    assert stats["cancelled_tail_trials"] == 2


def test_generate_rollout_backfills_incomplete_groups_before_slow_groups_finish(monkeypatch):
    class DataSource:
        def __init__(self):
            self.next_index = 0
            self.requests = []

        def get_samples(self, num_samples):
            self.requests.append(num_samples)
            groups = []
            for _ in range(num_samples):
                groups.append([Sample(index=self.next_index)])
                self.next_index += 1
            return groups

    async def scenario():
        group_two_started = asyncio.Event()

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
                await group_two_started.wait()
                await asyncio.sleep(0.01)
            if trial_group_index == 1:
                return [], {"converted": 0, "failed": 0, "missing": 1}
            if trial_group_index == 2:
                group_two_started.set()
            return [_train_sample(trial_group_index)], {"converted": 1, "failed": 0, "missing": 0}

        monkeypatch.setattr(rollout, "_run_one_group", fake_run_one_group)
        args = SimpleNamespace(
            rollout_batch_size=2,
            n_samples_per_prompt=1,
            harbor_require_full_rollout_groups=True,
            harbor_rollout_group_attempt_multiplier=3,
            harbor_num_concurrent_trials=4,
            harbor_retry_missing_samples=0,
            rollout_sample_filter_path=None,
            rollout_all_samples_process_path=None,
        )

        data_source = DataSource()
        output = await asyncio.wait_for(
            rollout._generate_rollout_async(args, 0, data_source),
            timeout=1.0,
        )
        return output, data_source

    output, data_source = asyncio.run(scenario())
    samples = [sample for group in output["samples"] for sample in group]

    assert [sample.index for sample in samples] == [2, 0]
    assert data_source.requests == [2, 1]
    assert output["metrics"]["harbor/num_incomplete_groups_dropped"] == 1


def test_generate_rollout_turn_grouped_counts_trajectories_without_trimming_turns(monkeypatch):
    class DataSource:
        def get_samples(self, num_samples):
            assert num_samples == 1
            return [[Sample(index=0, group_index=0), Sample(index=1, group_index=0)]]

    def turn_sample(index, sample_rank, turn_index):
        sample = _train_sample(index)
        sample.group_index = 0
        sample.metadata = {
            "harbor_rollout_id": 7,
            "harbor_rollout_group_index": 0,
            "harbor_sample_rank": sample_rank,
            "harbor_turn_index": turn_index,
            "harbor_sample_mode": "turn_grouped",
        }
        return sample

    async def fake_run_one_group(
        args,
        rollout_id,
        trial_group_index,
        base_group,
        semaphore,
        *,
        retry_missing_samples,
    ):
        return (
            [
                turn_sample(10, sample_rank=0, turn_index=0),
                turn_sample(11, sample_rank=0, turn_index=1),
                turn_sample(12, sample_rank=1, turn_index=0),
            ],
            {"converted": 3, "trajectories": 2, "failed": 0, "missing": 0},
        )

    monkeypatch.setattr(rollout, "_run_one_group", fake_run_one_group)
    args = SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        harbor_sample_mode="turn_grouped",
        harbor_require_full_rollout_groups=True,
        harbor_rollout_group_attempt_multiplier=1,
        harbor_num_concurrent_trials=2,
        harbor_retry_missing_samples=0,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )

    output = asyncio.run(rollout._generate_rollout_async(args, 7, DataSource()))
    samples = [sample for group in output["samples"] for sample in group]

    assert [sample.index for sample in samples] == [10, 11, 12]
    assert output["metrics"]["harbor/num_trajectory_samples"] == 2
    assert output["metrics"]["harbor/num_accepted_group_units"] == 2
    assert output["metrics"]["harbor/num_overflow_samples_trimmed"] == 0


def test_generate_rollout_accepts_partial_group_when_threshold_is_met(monkeypatch):
    class DataSource:
        def get_samples(self, num_samples):
            assert num_samples == 1
            return [[Sample(index=i) for i in range(4)]]

    async def fake_run_one_group(
        args,
        rollout_id,
        trial_group_index,
        base_group,
        semaphore,
        *,
        retry_missing_samples,
    ):
        return (
            [_train_sample(0), _train_sample(1), _train_sample(2)],
            {"converted": 3, "trajectories": 3, "failed": 0, "missing": 0, "cancelled_tail_trials": 1},
        )

    monkeypatch.setattr(rollout, "_run_one_group", fake_run_one_group)
    args = SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=4,
        harbor_sample_mode="last_turn",
        harbor_group_unit=None,
        harbor_partial_group_min_units=3,
        harbor_require_full_rollout_groups=True,
        harbor_rollout_group_attempt_multiplier=1,
        harbor_num_concurrent_trials=4,
        harbor_retry_missing_samples=0,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )

    output = asyncio.run(rollout._generate_rollout_async(args, 0, DataSource()))
    samples = [sample for group in output["samples"] for sample in group]

    assert [sample.index for sample in samples] == [0, 1, 2]
    assert output["metrics"]["harbor/num_partial_groups_accepted"] == 1
    assert output["metrics"]["harbor/num_partial_tail_trials_cancelled"] == 1
    assert output["metrics"]["harbor/partial_group_min_units"] == 3


def test_step_mode_defaults_to_trajectory_group_unit():
    args = SimpleNamespace(
        harbor_sample_mode="step",
        harbor_group_unit=None,
        harbor_require_full_rollout_groups=None,
        n_samples_per_prompt=2,
    )

    assert rollout._group_unit(args) == "trajectory"
    assert rollout._require_full_rollout_groups(args) is True


def test_daytona_environment_does_not_receive_singularity_defaults():
    args = SimpleNamespace(
        harbor_environment_type="daytona",
        harbor_singularity_image_cache_dir="/tmp/sif-cache",
        harbor_singularity_no_mount="home,tmp,bind-paths,proc",
    )
    config = {
        "environment": {
            "type": "daytona",
            "kwargs": {"connection_pool_maxsize": 16},
        }
    }

    rollout._set_default_environment_config(args, config)

    assert config["environment"]["type"] == "daytona"
    assert config["environment"]["kwargs"] == {"connection_pool_maxsize": 16}


def test_singularity_environment_keeps_singularity_defaults():
    args = SimpleNamespace(
        harbor_environment_type="daytona",
        harbor_singularity_image_cache_dir="/tmp/sif-cache",
        harbor_singularity_no_mount="home,tmp,bind-paths,proc",
    )
    config = {
        "environment": {
            "import_path": "slime_plugins.harbor.singularity_env:SlimeSingularityEnvironment",
            "kwargs": {},
        }
    }

    rollout._set_default_environment_config(args, config)

    assert config["environment"]["kwargs"]["singularity_image_cache_dir"] == "/tmp/sif-cache"
    assert config["environment"]["kwargs"]["singularity_no_mount"] == "home,tmp,bind-paths,proc"


def test_max_rollout_group_attempts_scales_with_oversampling_window():
    args = SimpleNamespace(harbor_rollout_group_attempt_multiplier=2)

    assert rollout._max_rollout_group_attempts(args, target_groups=2, over_sampling_groups=8) == 16


def test_validate_rollout_logprobs_requires_complete_logprobs_when_enabled():
    good = _train_sample(1)
    good.rollout_log_probs = [-0.1]
    bad = _train_sample(2)

    rollout._validate_rollout_logprobs(SimpleNamespace(use_rollout_logprobs=True), [good])

    try:
        rollout._validate_rollout_logprobs(SimpleNamespace(use_rollout_logprobs=True), [bad])
    except RuntimeError as exc:
        assert "rollout_log_probs" in str(exc)
    else:
        raise AssertionError("Expected missing rollout_log_probs to fail when enabled")


def test_run_one_with_semaphore_fails_without_workers_when_spread_requires_remote(monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("in-process Harbor trial should not run when fallback is disabled")

    monkeypatch.setattr(rollout, "_run_one_trial", fail_if_called)
    args = SimpleNamespace(
        _harbor_trial_workers=[],
        harbor_trial_node_spread=True,
        harbor_fallback_to_inprocess_trials=False,
    )

    with pytest.raises(RuntimeError, match="no TrialWorkers are available"):
        asyncio.run(rollout._run_one_with_semaphore(args, 0, 0, 0, Sample(index=9), asyncio.Semaphore(1)))


def test_sample_health_metrics_separate_timeout_length_trim_and_logprob():
    sample = _train_sample(1)
    sample.status = Sample.Status.TRUNCATED
    sample.rollout_log_probs = [-0.1]
    sample.metadata = {
        "harbor_num_turns": 3,
        "harbor_length_finish_turns": 1,
        "harbor_finish_reason_type": "length",
        "harbor_original_total_length": 40000,
        "harbor_trimmed_total_length": 32768,
    }

    metrics = {}
    rollout._add_sample_health_metrics(metrics, [sample])

    assert metrics["harbor/trial_timeout_ratio"] == 1.0
    assert metrics["harbor/step_length_finish_turn_ratio"] == 1 / 3
    assert metrics["harbor/step_length_finish_sample_ratio"] == 1.0
    assert metrics["harbor/train_context_trimmed_sample_ratio"] == 1.0
    assert metrics["harbor/rollout_logprob_sample_ratio"] == 1.0


def test_trajectory_shape_metrics_dedup_chunks_and_report_avg_turns_and_summary():
    """avg turns / output-tokens-per-turn / summary-rate are per-TRAJECTORY.

    In trajectory_retokenized mode one trajectory is emitted as several chunk
    samples carrying identical trajectory-level metadata. _add_trajectory_shape_metrics
    must dedup by (rollout_id, group_index, sample_rank) so a multi-chunk trajectory
    is not over-weighted. Here trajectory A spans 2 chunks (10 turns, 200 output
    tokens, summarized once) and trajectory B is a single sample (5 turns, 50 output
    tokens, never summarized).
    """

    def chunk(rank, turns, out_tokens, summ):
        sample = _train_sample(rank)
        sample.metadata = {
            "harbor_rollout_id": 3,
            "harbor_rollout_group_index": 0,
            "harbor_sample_rank": rank,
            "harbor_num_turns": turns,
            "harbor_total_output_tokens": out_tokens,
            "harbor_summarization_count": summ,
        }
        return sample

    samples = [
        chunk(rank=0, turns=10, out_tokens=200, summ=1),  # trajectory A, chunk 0
        chunk(rank=0, turns=10, out_tokens=200, summ=1),  # trajectory A, chunk 1 (dup)
        chunk(rank=1, turns=5, out_tokens=50, summ=0),    # trajectory B
    ]

    metrics = {}
    rollout._add_sample_health_metrics(metrics, samples)

    # 2 unique trajectories, not 3 chunk-samples.
    assert metrics["harbor/num_unique_trajectories"] == 2
    assert metrics["harbor/avg_turns_per_trajectory"] == (10 + 5) / 2
    # token-weighted: (200 + 50) total output / (10 + 5) total turns.
    assert metrics["harbor/avg_output_tokens_per_turn"] == (200 + 50) / (10 + 5)
    assert metrics["harbor/avg_output_tokens_per_trajectory"] == (200 + 50) / 2
    # 1 of 2 trajectories triggered summarization.
    assert metrics["harbor/summarization_trajectories_reported"] == 2
    assert metrics["harbor/summarization_triggered_trajectory_ratio"] == 0.5
    assert metrics["harbor/avg_summarization_count_per_trajectory"] == 0.5


def test_trajectory_shape_metrics_omit_summary_when_agent_did_not_report():
    sample = _train_sample(0)
    sample.metadata = {
        "harbor_rollout_id": 1,
        "harbor_rollout_group_index": 0,
        "harbor_sample_rank": 0,
        "harbor_num_turns": 4,
        "harbor_total_output_tokens": 80,
    }

    metrics = {}
    rollout._add_sample_health_metrics(metrics, [sample])

    assert metrics["harbor/num_unique_trajectories"] == 1
    assert metrics["harbor/avg_turns_per_trajectory"] == 4
    assert metrics["harbor/avg_output_tokens_per_turn"] == 20
    assert "harbor/summarization_triggered_trajectory_ratio" not in metrics


def test_generate_rollout_uses_oversampling_window_and_cancels_tail(monkeypatch):
    cancelled = []

    async def fake_run_one_group(
        args,
        rollout_id,
        trial_group_index,
        base_group,
        semaphore,
        *,
        retry_missing_samples,
    ):
        if trial_group_index in {0, 1}:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(trial_group_index)
                raise
        return [_train_sample(trial_group_index)], {"converted": 1, "failed": 0, "missing": 0}

    class DataSource:
        def __init__(self):
            self.next_index = 0
            self.requests = []

        def get_samples(self, num_samples):
            self.requests.append(num_samples)
            groups = []
            for _ in range(num_samples):
                groups.append([Sample(index=self.next_index)])
                self.next_index += 1
            return groups

    monkeypatch.setattr(rollout, "_run_one_group", fake_run_one_group)
    args = SimpleNamespace(
        rollout_batch_size=2,
        over_sampling_batch_size=4,
        n_samples_per_prompt=1,
        harbor_require_full_rollout_groups=True,
        harbor_rollout_group_attempt_multiplier=2,
        harbor_num_concurrent_trials=4,
        harbor_retry_missing_samples=0,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )

    data_source = DataSource()
    output = asyncio.run(rollout._generate_rollout_async(args, 0, data_source))
    samples = [sample for group in output["samples"] for sample in group]

    assert data_source.requests == [4]
    assert sorted(sample.index for sample in samples) == [2, 3]
    assert output["metrics"]["harbor/over_sampling_batch_size"] == 4
    assert output["metrics"]["harbor/num_submitted_groups"] == 4
    assert output["metrics"]["harbor/num_accepted_groups"] == 2
    assert output["metrics"]["harbor/num_cancelled_group_tasks"] == 2
    assert output["metrics"]["harbor/num_cancelled_tail_group_tasks"] == 2
    assert sorted(cancelled) == [0, 1]


def test_run_one_with_semaphore_cancels_ray_ref_on_task_cancel(monkeypatch):
    async def scenario():
        remote_called = asyncio.Event()
        cancelled_refs = []
        concurrent_future = concurrent.futures.Future()

        class Ref:
            def future(self):
                return concurrent_future

        ref = Ref()

        class RemoteMethod:
            def remote(self, *args, **kwargs):
                remote_called.set()
                return ref

        ray_module = types.ModuleType("ray")
        ray_module.cancel = lambda cancel_ref, force=False: cancelled_refs.append((cancel_ref, force))
        ray_exceptions = types.ModuleType("ray.exceptions")
        ray_exceptions.RayActorError = type("RayActorError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "ray", ray_module)
        monkeypatch.setitem(sys.modules, "ray.exceptions", ray_exceptions)
        monkeypatch.setattr(rollout, "_worker_cycle", None)
        monkeypatch.setattr(rollout, "_worker_cycle_len", 0)
        monkeypatch.setattr(rollout, "_worker_cycle_fingerprint", None)

        args = SimpleNamespace(
            _harbor_trial_workers=[SimpleNamespace(run_trial=RemoteMethod())],
            harbor_trial_remote_timeout_sec=0.0,
        )
        task = asyncio.create_task(
            rollout._run_one_with_semaphore(args, 0, 0, 0, Sample(index=9), asyncio.Semaphore(1))
        )
        await remote_called.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Expected cancelled Harbor trial task to propagate CancelledError")

        # Keep cancellation cooperative so Harbor Trial can run its shielded
        # cleanup path and delete remote environments such as Daytona sandboxes.
        assert cancelled_refs == [(ref, False)]

    asyncio.run(scenario())


def _eval_sample(index: int, reward: float) -> Sample:
    return Sample(
        index=index,
        tokens=[index, 100],
        response_length=1,
        loss_mask=[1],
        reward=reward,
        status=Sample.Status.COMPLETED,
    )


def test_generate_eval_async_total_timeout_scores_finished_and_cancels_pending(monkeypatch):
    """A deterministically-hanging eval group must not stall the whole eval.

    Group 2 never returns; with harbor_eval_total_timeout_sec set, the coordinator
    must cancel it at the deadline and still score on groups 0/1 that finished,
    reporting it under harbor/eval_num_timed_out_groups. This is the fix for the
    eval_9 (tb2of_raw_20260619_115727) >105min stall where a hung trial was
    re-dispatched across workers, each retry costing another remote_timeout.
    """
    cancelled_groups = []

    async def fake_run_one_group(args, rollout_id, trial_group_index, base_group, semaphore, *, retry_missing_samples):
        if trial_group_index == 2:
            try:
                await asyncio.Event().wait()  # hang forever until cancelled
            except asyncio.CancelledError:
                cancelled_groups.append(trial_group_index)
                raise
        reward = 1.0 if trial_group_index == 0 else 0.0
        return [_eval_sample(trial_group_index, reward)], {"converted": 1, "failed": 0, "missing": 0}

    monkeypatch.setattr(rollout, "_run_one_group", fake_run_one_group)

    data_source = SimpleNamespace(get_samples=lambda n, evaluation=False: [[Sample(index=i)] for i in range(3)])
    args = SimpleNamespace(harbor_num_concurrent_trials=3, harbor_eval_total_timeout_sec=0.2)

    out = asyncio.run(rollout._generate_eval_async(args, rollout_id=9, data_source=data_source))

    m = out["metrics"]
    assert m["harbor/eval_num_tasks"] == 3
    assert m["harbor/eval_num_timed_out_groups"] == 1
    assert m["harbor/eval_num_groups"] == 2  # only the 2 finished groups scored
    assert cancelled_groups == [2]  # the hung group was cancelled (so its trial/sandbox is reaped)
    assert sorted(out["eval"]["rewards"]) == [0.0, 1.0]


def test_generate_eval_async_disabled_timeout_waits_for_all(monkeypatch):
    """harbor_eval_total_timeout_sec<=0 keeps the legacy unbounded gather."""

    async def fake_run_one_group(args, rollout_id, trial_group_index, base_group, semaphore, *, retry_missing_samples):
        if trial_group_index == 1:
            await asyncio.sleep(0.05)
        return [_eval_sample(trial_group_index, 1.0)], {"converted": 1, "failed": 0, "missing": 0}

    monkeypatch.setattr(rollout, "_run_one_group", fake_run_one_group)

    data_source = SimpleNamespace(get_samples=lambda n, evaluation=False: [[Sample(index=i)] for i in range(2)])
    args = SimpleNamespace(harbor_num_concurrent_trials=2, harbor_eval_total_timeout_sec=0.0)

    out = asyncio.run(rollout._generate_eval_async(args, rollout_id=9, data_source=data_source))
    m = out["metrics"]
    assert m["harbor/eval_num_tasks"] == 2
    assert m["harbor/eval_num_groups"] == 2  # waited for the slow one
    assert m["harbor/eval_num_timed_out_groups"] == 0


def test_async_token_bucket_shared_across_two_event_loops():
    """The create-rate bucket is a module-global reused by two DISTINCT loops:
    the fully-async training loop (its own ``asyncio.run`` on a worker thread)
    and the shared eval loop (``async_utils.run``'s persistent AsyncLoopThread).

    Regression for run 20260703_001109: a single ``asyncio.Lock()`` bound to
    the first loop that awaited it, so the *other* loop's ``acquire()`` raised
    "is bound to a different event loop" and every trial on that loop was
    dropped before dispatch (82/89 eval tasks lost). The bucket must serve both
    loops without error, minting one lock per loop while sharing the token
    budget. See PPO_INTEGRATION_NOTES.md §10.
    """
    import threading

    bucket = rollout._AsyncTokenBucket(rate=1000.0, capacity=5.0)  # fast: keeps the test sub-second

    # Loop A stays ALIVE (mimics the persistent fully-async training loop),
    # binding a lock to itself, while loop B (eval) runs concurrently.
    bound = threading.Event()
    stop = threading.Event()
    errors: dict[str, str] = {}

    def loop_a():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def bind_then_idle():
            await bucket.acquire()  # binds a loop-A lock
            bound.set()
            while not stop.is_set():
                await asyncio.sleep(0.005)

        try:
            loop.run_until_complete(bind_then_idle())
        except Exception as exc:  # noqa: BLE001
            errors["A"] = repr(exc)
        finally:
            loop.close()

    thread_a = threading.Thread(target=loop_a)
    thread_a.start()
    assert bound.wait(timeout=5.0), "loop A never acquired"

    def loop_b():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def go():
            for _ in range(3):
                await bucket.acquire()

        try:
            loop.run_until_complete(go())
        except Exception as exc:  # noqa: BLE001
            errors["B"] = repr(exc)
        finally:
            loop.close()

    thread_b = threading.Thread(target=loop_b)
    thread_b.start()
    thread_b.join(timeout=5.0)
    stop.set()
    thread_a.join(timeout=5.0)

    assert not errors, f"cross-loop acquire raised: {errors}"
    assert len(bucket._locks) == 2, f"expected one lock per loop, got {len(bucket._locks)}"


def test_async_token_bucket_rate_zero_is_noop():
    """rate<=0 short-circuits before touching any loop-bound lock."""
    bucket = rollout._AsyncTokenBucket(rate=0.0, capacity=0.0)
    asyncio.run(bucket.acquire())
    assert bucket._locks == {}  # never even reached the lock path


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_generate_eval_async_requires_all_pass_at_k_attempts_and_uses_eval_sampling(monkeypatch):
    """Eval must not inherit training's partial-group cutoff.

    Regression for the first pass@3 run: harbor_partial_group_min_units=1 made
    each three-attempt eval task return after its first attempt, so 180/267
    attempts were cancelled and the reported metric was effectively pass@1.
    """
    seen = []

    async def fake_run_one_group(args, rollout_id, trial_group_index, base_group, semaphore, *, retry_missing_samples):
        seen.append(
            {
                "partial": getattr(args, "harbor_partial_group_min_units", None),
                "temperature": getattr(args, "rollout_temperature", None),
                "top_p": getattr(args, "rollout_top_p", None),
                "top_k": getattr(args, "rollout_top_k", None),
                "group_size": len(base_group),
                "trial_name_template": getattr(args, "harbor_trial_name_template", None),
            }
        )
        rewards = [0.0, 1.0, 0.0] if trial_group_index == 0 else [0.0, 0.0, 0.0]
        samples = []
        for rank, reward in enumerate(rewards):
            sample = _eval_sample(trial_group_index * 10 + rank, reward)
            sample.group_index = trial_group_index
            sample.metadata = {
                "harbor_rollout_id": rollout_id,
                "harbor_rollout_group_index": trial_group_index,
                "harbor_sample_rank": rank,
            }
            samples.append(sample)
        return samples, {"converted": 3, "failed": 0, "missing": 0}

    monkeypatch.setattr(rollout, "_run_one_group", fake_run_one_group)
    data_source = SimpleNamespace(
        get_samples=lambda n, evaluation=False: [
            [Sample(index=0), Sample(index=1), Sample(index=2)],
            [Sample(index=3), Sample(index=4), Sample(index=5)],
        ]
    )
    args = SimpleNamespace(
        harbor_num_concurrent_trials=6,
        harbor_eval_total_timeout_sec=0,
        harbor_partial_group_min_units=1,
        n_samples_per_eval_prompt=3,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        eval_temperature=0.6,
        eval_top_p=0.95,
        eval_top_k=20,
    )

    out = asyncio.run(rollout._generate_eval_async(args, rollout_id=0, data_source=data_source))
    assert len(out["eval"]["rewards"]) == 6
    assert out["metrics"]["harbor/eval_num_missing_attempts"] == 0
    # eval_resolved_rate_full is the avg@3 alias (1 resolved attempt / (2*3) = 1/6);
    # pass@3 still resolves task 0 via its rank-1 success. Guards that all three
    # attempts ran (not the partial-cutoff-truncated effectively-pass@1 case).
    assert out["metrics"]["harbor/eval_resolved_rate_full"] == 1 / 6
    assert out["metrics"]["harbor/eval_resolved_rate_full_pass@3"] == 0.5
    assert out["metrics"]["harbor/eval_resolved_rate_full_pass@1"] == 0.0
    assert all(
        item == {
            "partial": 0,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "group_size": 3,
            "trial_name_template": "slime-eval-r{rollout_id}-g{group_index}-s{sample_rank}-i{sample_index}",
        }
        for item in seen
    )
    # Shared training args remain untouched.
    assert args.harbor_partial_group_min_units == 1
    assert args.rollout_temperature == 1.0


def test_run_one_group_propagates_fatal_trial_worker_loss(monkeypatch):
    async def no_workers(*args, **kwargs):
        raise rollout.HarborTrialWorkersUnavailableError("all workers gone")

    monkeypatch.setattr(rollout, "_run_one_with_semaphore", no_workers)
    with pytest.raises(rollout.HarborTrialWorkersUnavailableError):
        asyncio.run(
            rollout._run_one_group(
                SimpleNamespace(harbor_fail_on_trial_error=False),
                rollout_id=0,
                trial_group_index=0,
                base_group=[Sample(index=0)],
                semaphore=asyncio.Semaphore(1),
                retry_missing_samples=0,
            )
        )
