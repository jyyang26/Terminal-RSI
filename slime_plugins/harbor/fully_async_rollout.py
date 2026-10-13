from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.async_utils import run
from slime.utils.types import Sample

from slime_plugins.harbor.rollout import (
    _add_latency_metrics,
    _add_sample_health_metrics,
    _apply_rollout_sample_hooks,
    _count_unique_trajectories,
    _float_arg,
    _group_unit,
    _group_unit_count,
    _max_rollout_group_attempts,
    _over_sampling_group_window,
    _accepts_partial_group,
    _partial_group_min_units,
    _require_full_rollout_groups,
    _run_one_group,
    _validate_rollout_logprobs,
)

logger = logging.getLogger(__name__)


@dataclass
class _CompletedGroup:
    gid: int
    group: list[Sample]
    stats: dict[str, int]
    latency_seconds: float
    error: BaseException | None = None


class HarborFullyAsyncRolloutWorker:
    def __init__(self, args, data_source):
        self.args = args
        self.data_source = data_source
        self.running = True
        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.rollout_id_hint = int(getattr(args, "start_rollout_id", 0) or 0)
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    def set_rollout_id_hint(self, rollout_id: int) -> None:
        self.rollout_id_hint = int(rollout_id)

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._thread_main,
                name="harbor-fully-async-rollout",
                daemon=True,
            )
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        loop = self._event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._request_stop)
        if self.worker_thread and self.worker_thread.is_alive():
            drain_timeout = _float_arg(self.args, "harbor_fully_async_shutdown_drain_timeout_sec", 30.0)
            join_timeout = _float_arg(
                self.args,
                "harbor_fully_async_shutdown_join_timeout_sec",
                max(drain_timeout + 5.0, 5.0),
            )
            self.worker_thread.join(timeout=max(join_timeout, 0.0))
            if self.worker_thread.is_alive():
                logger.warning(
                    "Harbor fully-async rollout worker did not stop within %.1fs; "
                    "remaining in-flight trials may require launcher-level cleanup.",
                    join_timeout,
                )

    def _request_stop(self) -> None:
        self.running = False
        if self._stop_event is not None:
            self._stop_event.set()

    def get_completed_groups(self) -> list[_CompletedGroup]:
        completed: list[_CompletedGroup] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def max_concurrent_groups(self) -> int:
        explicit = getattr(self.args, "harbor_fully_async_max_concurrent_groups", None)
        if explicit is not None:
            return max(1, int(explicit))
        return max(1, _over_sampling_group_window(self.args, int(self.args.rollout_batch_size)))

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        active_tasks: set[asyncio.Task] = set()
        trial_concurrency = int(getattr(self.args, "harbor_num_concurrent_trials", 1) or 1)
        trial_semaphore = asyncio.Semaphore(trial_concurrency)
        gid_counter = 0

        try:
            while self.running:
                try:
                    self._harvest_done_tasks(active_tasks)

                    while len(active_tasks) < self.max_concurrent_groups() and self.running:
                        groups = self.data_source.get_samples(1)
                        if not groups:
                            break
                        base_group = groups[0]
                        gid = gid_counter
                        gid_counter += 1
                        started_at = time.monotonic()
                        rollout_id = self.rollout_id_hint
                        group_index = _base_group_index(base_group, gid)
                        task = asyncio.create_task(
                            self._run_group(
                                gid,
                                rollout_id,
                                group_index,
                                base_group,
                                trial_semaphore,
                                started_at,
                            )
                        )
                        active_tasks.add(task)

                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Harbor fully-async loop iteration error: %s", exc)
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
        finally:
            await self._shutdown_active_tasks(active_tasks)
            self._stop_event = None
            self._event_loop = None

    def _harvest_done_tasks(self, active_tasks: set[asyncio.Task]) -> None:
        done = {task for task in active_tasks if task.done()}
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                logger.debug("Harbor fully-async group task cancelled")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Harbor fully-async group task crashed: %r", exc)
        active_tasks -= done

    async def _shutdown_active_tasks(self, active_tasks: set[asyncio.Task]) -> None:
        self._harvest_done_tasks(active_tasks)
        if not active_tasks:
            return

        drain_timeout = _float_arg(self.args, "harbor_fully_async_shutdown_drain_timeout_sec", 30.0)
        if drain_timeout > 0:
            logger.info("Harbor fully-async: draining %d in-flight groups for %.1fs", len(active_tasks), drain_timeout)
            done, pending = await asyncio.wait(active_tasks, timeout=drain_timeout)
            self._harvest_done_tasks(done)
            active_tasks.clear()
            active_tasks.update(pending)

        if not active_tasks:
            return

        logger.warning("Harbor fully-async: cancelling %d in-flight groups during shutdown", len(active_tasks))
        for task in active_tasks:
            task.cancel()
        await asyncio.gather(*active_tasks, return_exceptions=True)
        active_tasks.clear()

    async def _run_group(
        self,
        gid: int,
        rollout_id: int,
        group_index: int,
        base_group: list[Sample],
        trial_semaphore: asyncio.Semaphore,
        started_at: float,
    ) -> None:
        try:
            group, stats = await _run_one_group(
                self.args,
                rollout_id,
                group_index,
                base_group,
                trial_semaphore,
                retry_missing_samples=int(getattr(self.args, "harbor_retry_missing_samples", 0) or 0),
            )
            completed = _CompletedGroup(
                gid=gid,
                group=group,
                stats=stats,
                latency_seconds=time.monotonic() - started_at,
            )
        except Exception as exc:  # noqa: BLE001
            completed = _CompletedGroup(
                gid=gid,
                group=[],
                stats={"converted": 0, "trajectories": 0, "failed": 1, "missing": 0},
                latency_seconds=time.monotonic() - started_at,
                error=exc,
            )
        try:
            self.output_queue.put_nowait(completed)
        except queue.Full:
            logger.warning("Harbor fully-async output queue is full; dropping completed group %s", gid)


_global_worker: HarborFullyAsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_source, rollout_id: int) -> HarborFullyAsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or _global_worker.worker_thread is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting Harbor fully-async rollout worker")
            _global_worker = HarborFullyAsyncRolloutWorker(args, data_source)
            _global_worker.set_rollout_id_hint(rollout_id)
            _global_worker.start()
        else:
            _global_worker.set_rollout_id_hint(rollout_id)
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


def cleanup_rollout(args=None) -> None:
    _stop_global_worker()


atexit.register(_stop_global_worker)


async def _generate_rollout_async(args, rollout_id: int, data_source) -> tuple[list[list[Sample]], dict[str, Any]]:
    worker = _get_global_worker(args, data_source, rollout_id)

    target_groups = int(args.rollout_batch_size)
    target_samples_per_group = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    require_full_groups = _require_full_rollout_groups(args)
    max_group_attempts = _max_rollout_group_attempts(
        args,
        target_groups,
        over_sampling_groups=worker.max_concurrent_groups(),
    )

    output_groups: list[list[Sample]] = []
    converted_count = 0
    failed_count = 0
    missing_count = 0
    trajectory_count = 0
    completed_group_tasks = 0
    incomplete_groups = 0
    partial_groups = 0
    partial_tail_trials_cancelled = 0
    overflow_samples = 0
    overflow_group_units = 0
    accepted_group_units = 0
    discarded_completed_groups = 0
    completed_group_latencies: list[float] = []

    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    logger.info(
        "Harbor fully-async rollout %d: target=%d queue_warm=%d max_concurrent_groups=%d",
        rollout_id,
        target_groups,
        worker.queue_size(),
        worker.max_concurrent_groups(),
    )

    while len(output_groups) < target_groups:
        drained = worker.get_completed_groups()
        if not drained:
            await asyncio.sleep(0.05)
        for completed in drained:
            completed_group_tasks += 1
            completed_group_latencies.append(completed.latency_seconds)

            if completed.error is not None and getattr(args, "harbor_fail_on_trial_error", False):
                raise RuntimeError("Harbor fully-async group task failed") from completed.error

            stats = completed.stats
            converted_count += int(stats.get("converted", 0) or 0)
            trajectory_count += int(stats.get("trajectories", 0) or 0)
            failed_count += int(stats.get("failed", 0) or 0)
            missing_count += int(stats.get("missing", 0) or 0)
            partial_tail_trials_cancelled += int(stats.get("cancelled_tail_trials", 0) or 0)

            group = completed.group
            group_unit_count = _group_unit_count(args, group, stats)
            if len(output_groups) >= target_groups:
                discarded_completed_groups += 1
                continue

            if require_full_groups and group_unit_count < target_samples_per_group:
                if _accepts_partial_group(args, group_unit_count, target_samples_per_group):
                    partial_groups += 1
                    logger.info(
                        "Accepting partial Harbor fully-async group with %s/%s %s units.",
                        group_unit_count,
                        target_samples_per_group,
                        _group_unit(args),
                    )
                else:
                    incomplete_groups += 1
                    logger.warning(
                        "Dropping incomplete Harbor fully-async group with %s/%s %s units.",
                        group_unit_count,
                        target_samples_per_group,
                        _group_unit(args),
                    )
                    continue

            if require_full_groups and _group_unit(args) == "sample" and len(group) > target_samples_per_group:
                overflow_samples += len(group) - target_samples_per_group
                group = group[:target_samples_per_group]
                group_unit_count = len(group)
            elif require_full_groups and group_unit_count > target_samples_per_group:
                overflow_group_units += group_unit_count - target_samples_per_group

            if group:
                output_groups.append(group)
                accepted_group_units += group_unit_count

        if completed_group_tasks >= max_group_attempts and len(output_groups) < target_groups:
            raise RuntimeError(
                "Harbor fully-async rollout could not produce enough complete trainable groups: "
                f"{len(output_groups)}/{target_groups} groups after {completed_group_tasks} completed group tasks. "
                "Increase `harbor_rollout_group_attempt_multiplier`, reduce failing tasks, or relax "
                "`harbor_require_full_rollout_groups`."
            )

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "Harbor fully-async rollout %d: collected %d/%d, queue=%d, completed=%d, elapsed=%.1fs",
                rollout_id,
                len(output_groups),
                target_groups,
                worker.queue_size(),
                completed_group_tasks,
                now - started,
            )
            last_log = now

    _apply_rollout_sample_hooks(args, output_groups, data_source)

    final_samples = [sample for group in output_groups for sample in group]
    _validate_rollout_logprobs(args, final_samples)
    metrics: dict[str, Any] = {
        "harbor/fully_async_enabled": 1,
        "harbor/fully_async_queue_left": worker.queue_size(),
        "harbor/fully_async_max_concurrent_groups": worker.max_concurrent_groups(),
        "harbor/num_groups": len(output_groups),
        "harbor/num_samples": len(final_samples),
        "harbor/num_converted_samples": converted_count,
        "harbor/num_failed_trials": failed_count,
        "harbor/num_missing_rollout_details": missing_count,
        "harbor/num_trajectory_samples": _count_unique_trajectories(final_samples),
        "harbor/num_converted_trajectories": trajectory_count,
        "harbor/num_accepted_group_units": accepted_group_units,
        "harbor/num_group_attempts": completed_group_tasks,
        "harbor/num_completed_group_tasks": completed_group_tasks,
        "harbor/num_consumed_group_tasks": completed_group_tasks,
        "harbor/num_accepted_groups": len(output_groups),
        "harbor/num_partial_groups_accepted": partial_groups,
        "harbor/num_partial_tail_trials_cancelled": partial_tail_trials_cancelled,
        "harbor/partial_group_min_units": _partial_group_min_units(args, target_samples_per_group),
        "harbor/num_incomplete_groups_dropped": incomplete_groups,
        "harbor/num_overflow_samples_trimmed": overflow_samples,
        "harbor/num_overflow_group_units": overflow_group_units,
        "harbor/num_discarded_completed_group_tasks": discarded_completed_groups,
        "harbor/trial_worker_count": int(getattr(args, "_harbor_trial_worker_count", 0) or 0),
        "harbor/trial_worker_healthy_count": int(getattr(args, "_harbor_trial_worker_healthy_count", 0) or 0),
        "harbor/trial_worker_healthcheck_failed": int(
            getattr(args, "_harbor_trial_worker_healthcheck_failed", 0) or 0
        ),
        "harbor/trial_worker_healthcheck_killed": int(
            getattr(args, "_harbor_trial_worker_healthcheck_killed", 0) or 0
        ),
    }
    _add_latency_metrics(metrics, "harbor/group_latency", completed_group_latencies)
    if final_samples:
        metrics["harbor/reward_mean"] = sum(float(sample.reward) for sample in final_samples) / len(final_samples)
    _add_sample_health_metrics(metrics, final_samples)

    logger.info(
        "Harbor fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return output_groups, metrics


def generate_rollout_fully_async(args, rollout_id, data_source, evaluation: bool = False):
    if evaluation:
        raise ValueError("Harbor fully-async rollout does not support evaluation mode")
    output_groups, metrics = run(_generate_rollout_async(args, rollout_id, data_source))
    return RolloutFnTrainOutput(samples=output_groups, metrics=metrics)


def _base_group_index(base_group: list[Sample], fallback: int) -> int:
    if base_group:
        group_index = getattr(base_group[0], "group_index", None)
        if group_index is not None:
            return int(group_index)
    return int(fallback)
