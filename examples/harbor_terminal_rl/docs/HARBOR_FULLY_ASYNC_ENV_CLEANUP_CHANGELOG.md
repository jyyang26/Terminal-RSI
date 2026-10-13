# Harbor Fully Async Env Cleanup Changelog

Date: 2026-06-06

This document records the cleanup-related code changes made for Harbor fully
async rollout. The goal is to make normal Python/Ray shutdown paths clean up
in-flight Apptainer/Singularity trial environments more reliably.

## Motivation

Harbor fully async rollout keeps a background thread alive inside the
RolloutManager process. Before this change, `stop()` only set `running = False`
and joined the thread for 5 seconds. The worker loop then passively waited for
in-flight group tasks for up to 30 seconds, but did not explicitly cancel the
remaining tasks.

That meant normal training exit, training exceptions, or cancellation of an
outstanding rollout could leave active Harbor trial groups running longer than
expected. The lower Harbor stack already had useful cleanup support:

- `_run_one_group()` cancels remaining sample tasks in its `finally`.
- `_run_one_with_semaphore()` calls `ray.cancel(ref, force=True)` for remote
  TrialWorker calls when its await is cancelled.
- Harbor `Trial.run()` calls `environment.stop(...)` in `finally`.
- Singularity env `stop()` terminates the server process and child processes.

The missing piece was making the fully async layer actually trigger that
cancellation path.

## Changed Files

### `slime_plugins/harbor/fully_async_rollout.py`

Added explicit shutdown lifecycle to `HarborFullyAsyncRolloutWorker`:

- Stores the worker event loop in `self._event_loop`.
- Adds `self._stop_event` so `stop()` can wake the worker thread immediately.
- `start()` resets `running = True` before starting the thread.
- `stop()` now:
  - sets `running = False`;
  - calls `_request_stop()` on the worker event loop using
    `loop.call_soon_threadsafe(...)`;
  - joins with configurable timeout.
- `_loop()` now exits through a `finally` block that calls
  `_shutdown_active_tasks(...)`.
- `_shutdown_active_tasks(...)` now:
  - harvests already completed tasks;
  - soft-drains in-flight group tasks for
    `harbor_fully_async_shutdown_drain_timeout_sec`;
  - explicitly cancels remaining tasks;
  - awaits cancellation with `return_exceptions=True`.
- Completed groups now use `output_queue.put_nowait(...)` so a full queue cannot
  block the worker event loop during shutdown.
- Added module-level `cleanup_rollout(args=None)` as the RolloutManager cleanup
  hook.

Default timeout behavior:

```yaml
harbor_fully_async_shutdown_drain_timeout_sec: 30
harbor_fully_async_shutdown_join_timeout_sec: 35  # default is drain + 5
```

The full89 9B config overrides join timeout to 45 seconds.

### `slime/ray/rollout.py`

Added a generic rollout cleanup hook:

- New optional argument consumed by RolloutManager:
  `rollout_cleanup_function_path`.
- RolloutManager loads the cleanup function in `__init__`.
- `dispose()` now calls the cleanup function before killing rollout auxiliary
  workers.
- Added `_call_rollout_cleanup(reason)` helper.
- `generate()` now catches `BaseException`, calls rollout cleanup with reason
  `generate_exception`, then re-raises.

The ordering is intentional:

1. Stop/cancel the fully async rollout worker.
2. Let cancellation propagate into group/sample tasks and remote TrialWorker
   refs.
3. Then kill auxiliary TrialWorker actors.

This gives Harbor trials a chance to run their normal cleanup path before the
actors are killed.

### `slime/utils/arguments.py`

Registered:

```bash
--rollout-cleanup-function-path
```

Expected function signature:

```python
def cleanup_rollout(args) -> None:
    ...
```

This is intentionally generic and not Harbor-specific.

### `train_async.py`

Wrapped the async training loop in `try/finally`:

- Tracks the outstanding `rollout_manager.generate.remote(...)` future.
- On exit, cancels that future with `ray.cancel(..., force=False)`.
- Attempts `rollout_manager.dispose.remote()` on both normal and exceptional
  exits.
- Calls `finish_tracking(args)` from the `finally` path if tracking was started.

`force=False` is used so the RolloutManager actor can receive cancellation and
run `generate()` cleanup instead of being hard-killed immediately.

### Full89 9B Config

Updated:

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_trajectory_normalized.yaml
```

Added:

```yaml
rollout_cleanup_function_path: slime_plugins.harbor.fully_async_rollout.cleanup_rollout
harbor_fully_async_shutdown_drain_timeout_sec: 30
harbor_fully_async_shutdown_join_timeout_sec: 45
```

### `tests/test_harbor_fully_async_rollout.py`

Added/updated CPU-only tests:

- Backfill behavior still works when incomplete groups are dropped.
- Worker `stop()` cancels an active hanging group task.
- Added `NUM_GPUS = 0`.
- Added direct `python tests/test_harbor_fully_async_rollout.py` compatibility.

## Verification

Commands run:

```bash
PYTHONPYCACHEPREFIX=/tmp/harbor_v030_pycache python -m py_compile \
  slime_plugins/harbor/fully_async_rollout.py \
  slime_plugins/harbor/rollout.py \
  slime/ray/rollout.py \
  slime/utils/arguments.py \
  train_async.py \
  tests/test_harbor_fully_async_rollout.py
```

Result: passed. There is an existing `SyntaxWarning` in `slime/utils/arguments.py`
around an unrelated regex help string.

```bash
python -m pytest \
  tests/test_harbor_fully_async_rollout.py \
  tests/test_harbor_rollout_async.py \
  tests/test_rollout_aux_workers.py
```

Result: `13 passed`.

```bash
python tests/test_harbor_fully_async_rollout.py
```

Result: `2 passed`.

## Expected Runtime Behavior

For normal Python/Ray shutdown paths:

1. `train_async.py` enters `finally`.
2. The outstanding next-rollout future is cancelled.
3. RolloutManager `generate()` cleanup or `dispose()` cleanup calls
   `slime_plugins.harbor.fully_async_rollout.cleanup_rollout`.
4. The fully async worker stops topping up new groups.
5. The worker drains active groups briefly.
6. Remaining active groups are cancelled.
7. Cancellation propagates through `_run_one_group()` and remote TrialWorker
   awaits.
8. TrialWorker remote refs are cancelled.
9. Harbor `Trial.run()` should execute its normal `finally` cleanup and stop
   the Singularity environment.
10. RolloutManager then kills rollout auxiliary workers.

This is materially better than the old passive drain, but it is still bounded
best-effort cleanup.

## Remaining Risks

This patch does not make hard failure modes fully clean:

- platform sends SIGKILL;
- node is evicted;
- Ray head dies;
- RolloutManager process is killed without Python cleanup;
- Apptainer/Singularity runtime is stuck in kernel I/O;
- trial process ignores cancellation until after a long blocking syscall.

For those cases, launcher-level cleanup is still required:

```bash
ray stop --force
pkill -TERM -f '[a]pptainer'
pkill -TERM -f '[s]ingularity'
```

Use stronger `pkill -KILL` only as a last resort.

## Still Pending

This change is shutdown cleanup only. It does not yet implement a true
update-weight quiesce.

The important remaining training-correctness issue is:

- fully async can keep a warm queue or active hidden groups after one rollout
  call has returned;
- if SGLang weights are updated while hidden old-weight groups are still running
  or queued, stale samples can cross the weight boundary.

Recommended next work:

1. Add `quiesce_rollout()` on RolloutManager.
2. Add Harbor fully async `quiesce()` that stops top-up, cancels active groups,
   and drops stale queue entries.
3. Add generation epoch tagging to `_CompletedGroup`.
4. Call quiesce before `actor_model.update_weights()`.
5. Add smoke validation that tail logs no longer show old in-flight groups after
   checkpoint/update.
