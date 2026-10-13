# Harbor Fully Async Shutdown / Drain Notes

Status: basic shutdown cleanup implemented; update-weight quiesce still pending.
Date: 2026-06-06.

## Background

We added a Harbor-specific fully async rollout entrypoint:

- `slime_plugins.harbor.fully_async_rollout.generate_rollout_fully_async`
- Launcher switch: `HARBOR_ROLLOUT_MODE=fully_async`
- Smoke output root:
  `/path/to/workspace/slime_harbor_v030_fully_async_smoke_gbs4`

The smoke run succeeded end to end:

- Ray job: `raysubmit_qsr37cD1fpFRpUXE`
- Rollout entrypoint was correctly set to Harbor fully async.
- Rollout 1 collected 2 groups / 4 samples.
- `harbor/num_converted_samples = 4`
- `harbor/num_failed_trials = 0`
- `harbor/num_missing_rollout_details = 0`
- `harbor/num_incomplete_groups_dropped = 0`
- `harbor/rollout_logprob_sample_ratio = 1.0`
- Actor train completed 1 microbatch.
- Iteration 1 checkpoint was saved successfully.
- Final Ray status: `Job 'raysubmit_qsr37cD1fpFRpUXE' succeeded`

The smoke also exposed a shutdown/drain issue:

```text
Trial slime-r1-g1-s1-i3 failed: Server disconnected without sending a response.
Harbor fully-async: waiting for 2 in-flight groups to drain
```

This happened after rollout 1 had already been consumed, actor train had completed,
and checkpoint iteration 1 had been saved. It did not fail the job, but it shows
that the fully async background worker can still have in-flight Harbor trials when
the training loop starts weight update / final cleanup.

## Current Code Path

The current Harbor fully async worker lives in:

```text
slime_plugins/harbor/fully_async_rollout.py
```

Important current behavior after the 2026-06-06 cleanup patch:

- The worker is global to the RolloutManager process.
- `generate_rollout_fully_async()` returns as soon as enough completed groups are available.
- The background thread keeps topping up new groups while `self.running` is true.
- `stop()` sets `self.running = False`, wakes the worker event loop through a
  thread-safe stop event, and joins with a configurable timeout.
- `_loop()` soft-drains active group tasks for
  `harbor_fully_async_shutdown_drain_timeout_sec` seconds, then explicitly
  cancels remaining active groups.
- Cancelling a group task reaches `_run_one_group()`, which cancels sample tasks;
  for remote TrialWorker calls, `_run_one_with_semaphore()` calls
  `ray.cancel(ref, force=True)`.
- Completed groups are enqueued with `put_nowait()` so a full queue cannot block
  the worker event loop during shutdown.
- `cleanup_rollout(args)` is exposed as the module-level cleanup hook.
- Normal `RolloutManager.dispose()` can call that hook through
  `rollout_cleanup_function_path`.
- `atexit.register(_stop_global_worker)` remains as a final fallback.

The relevant config for Harbor fully async is:

```yaml
rollout_cleanup_function_path: slime_plugins.harbor.fully_async_rollout.cleanup_rollout
harbor_fully_async_shutdown_drain_timeout_sec: 30
harbor_fully_async_shutdown_join_timeout_sec: 45
```

The basic deterministic shutdown path is now:

1. Driver enters `train_async.py` `finally`.
2. Driver cancels any outstanding `rollout_manager.generate.remote(...)` future.
3. Driver calls `rollout_manager.dispose.remote()`.
4. RolloutManager calls configured `rollout_cleanup_function_path`.
5. Harbor fully async worker stops top-up, drains active groups briefly, cancels
   leftovers, then joins the background thread.
6. RolloutManager kills rollout auxiliary workers after the rollout cleanup hook.

## Why This Matters

There are two separate issues.

### 1. Clean process / container exit

If the process exits while Harbor trials are still running, those trials may be
interrupted while they are talking to one of:

- SGLang `/generate`
- SGLang `/continue_generation`
- the Harbor Singularity FastAPI server
- tmux / terminal commands inside the container

That produces noisy but understandable errors such as:

```text
Server disconnected without sending a response.
```

The current code now explicitly cancels active group tasks on shutdown, so the
lower-layer cancellation path is exercised during normal Python-level cleanup.
This still cannot cover hard cluster failure modes such as process SIGKILL,
node eviction, Ray head death, or a container runtime stuck in kernel I/O.

### 2. Train / inference weight boundary

The more important correctness concern is the update-weight boundary.

`train_async.py` does this:

1. start next rollout early
2. train on current rollout
3. before `actor_model.update_weights()`, sync the explicit
   `rollout_data_next_future`
4. call `actor_model.update_weights()`

For normal rollout, syncing the explicit future is enough. For fully async Harbor
rollout, the explicit future returning does not mean the hidden background worker
has stopped producing. The worker may still have extra trials in flight or queued.

If we update SGLang weights while hidden old-weight Harbor trials are still
generating, we can get both:

- operational noise: disconnected generation requests during pause/flush/update
- semantic risk: old-weight samples sitting in the completed queue and being
  consumed after the weight update

The smoke log suggests this can happen: after iteration 1 checkpoint, SGLang
weight update / pause / flush started while the fully async worker still had
in-flight groups.

## Existing Lower-Level Cancellation Support

The lower Harbor rollout path already has useful cancellation handling:

```text
slime_plugins/harbor/rollout.py
```

`_run_one_group()` has a `finally` block that cancels remaining sample tasks and
awaits them with `return_exceptions=True`.

`_run_one_with_semaphore()` handles `asyncio.CancelledError` while waiting for a
Ray TrialWorker remote ref:

```python
except asyncio.CancelledError:
    try:
        ray.cancel(ref, force=True)
    except Exception:
        logger.exception("Failed to cancel over-sampled TrialWorker task ...")
    raise
```

Therefore the fully async layer does not need to invent the whole cancellation
mechanism. It needs to actually cancel its active group tasks during quiesce /
shutdown, so the existing lower-level cleanup path can run.

## Remaining Design Work

The implemented patch covers shutdown / dispose. It does not yet solve the
update-weight quiesce problem.

### Quiesce before weight update

We still need an explicit method similar to:

```python
def quiesce(self, reason: str, timeout: float) -> dict:
    ...
```

Semantics:

1. Stop topping up new groups.
2. Increment `generation_epoch`.
3. Drop queued completed groups from old epochs.
4. Cancel all active group tasks.
5. Await cancellation with a bounded timeout.
6. Return metrics:
   - active task count before cancel
   - cancelled count
   - completed during drain count
   - queue entries dropped
   - timeout flag

This method is for update-weight boundaries. It should prefer correctness over
keeping the warm queue.

## Integration Points

### RolloutManager

Implemented:

- `slime/ray/rollout.py` loads optional `rollout_cleanup_function_path`.
- `RolloutManager.dispose()` calls the cleanup hook before killing rollout aux
  workers.

Still pending:

- a separate `quiesce_rollout()` remote method for update-weight boundaries.

### train_async.py

Before `actor_model.update_weights()`, call:

```python
ray.get(rollout_manager.quiesce_rollout.remote(reason="before_update_weights"))
actor_model.update_weights()
```

This is the most important hook. It prevents hidden old-weight Harbor rollout
from crossing the SGLang weight-update boundary.

Implemented:

- `train_async.py` wraps the main train loop in `try/finally`.
- It cancels the outstanding next-rollout future before calling dispose.
- It attempts `rollout_manager.dispose()` on both normal and exceptional exits.

Still pending:

- before `actor_model.update_weights()`, call a future quiesce method to stop
  hidden old-weight Harbor rollout from crossing the weight boundary.

### Launcher cleanup

Keep the launcher-level cleanup trap as a final backstop:

- write worker stop marker
- `ray stop --force`
- kill lingering SGLang / train processes
- clean Apptainer / Singularity helper processes if needed

This is still useful because Python-level cleanup cannot handle all cluster
failure modes, for example Ray head death, container runtime hangs, or processes
stuck in kernel I/O wait.

## Generation Epoch Guard

Add an epoch guard to prevent stale completed samples from entering a later
training step.

Suggested behavior:

- Each group task captures `task_epoch = self.generation_epoch` at submit time.
- `_CompletedGroup` records `epoch`.
- `get_completed_groups()` only returns groups matching the current epoch.
- On quiesce/update, increment `generation_epoch` and drop old epoch queue
  entries.

This protects against a race where a group finishes just after quiesce starts.

Policy options for stale groups:

1. Drop stale completed samples.
2. Requeue their base task group into `HarborTaskDataSource`.

For the first implementation, dropping is simpler and safer. The data source is
cyclic for Harbor tasks, so losing a small number of speculative prefetches is
acceptable for correctness. If later we need exact task accounting, store the
base group in `_CompletedGroup` or in a separate active-task registry and requeue
on cancellation/stale completion.

## Metrics To Add

Add rollout/shutdown metrics and log lines:

- `harbor/fully_async_quiesce_count`
- `harbor/fully_async_shutdown_count`
- `harbor/fully_async_active_before_quiesce`
- `harbor/fully_async_cancelled_on_quiesce`
- `harbor/fully_async_cancel_timeout`
- `harbor/fully_async_stale_completed_dropped`
- `harbor/fully_async_queue_dropped`
- `harbor/fully_async_active_after_shutdown`
- `harbor/fully_async_shutdown_timeout`

These should appear in logs even if they are not attached to a rollout metrics
dict, because quiesce happens outside a normal rollout return.

## Verification Plan

### Unit tests

Add tests under `tests/test_harbor_fully_async_rollout.py`:

1. shutdown cleanup cancels active tasks.
2. `quiesce()` stops top-up and clears queue.
3. stale completed groups are not returned after epoch increment.
4. cancelling an active group task triggers `_run_one_group()` cancellation.
5. `shutdown()` is idempotent.
6. `quiesce()` followed by another rollout restarts top-up cleanly.

The existing async rollout tests already have patterns for cancellation and
over-sampling tail behavior.

### Smoke tests

Run the same conservative smoke:

```bash
HARBOR_ROLLOUT_MODE=fully_async
ROLLOUT_BATCH_SIZE=2
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=4
OVER_SAMPLING_BATCH_SIZE=2
NUM_ROLLOUT=2
```

Expected result after the current patch:

- Job succeeds.
- Rollout produces 2 groups / 4 samples.
- Actor train and checkpoint still complete.
- Logs show explicit fully async drain/cancel on dispose if in-flight groups exist.
- Tail `Server disconnected without sending a response` should be reduced on
  normal shutdown, but may still appear when a running TrialWorker remote call is
  forcibly cancelled.
- Final shutdown should not leave active Python group tasks in the RolloutManager
  process.

### Node cleanup check

After job exit, check each node for leftovers:

```bash
ps -ef | grep -E '[s]glang|[a]pptainer|[s]ingularity|[f]use-overlayfs|[r]aylet|train_async.py'
```

Expected:

- No active `train_async.py`.
- No active SGLang engines.
- No unexpected Apptainer/Singularity trial containers.
- Ray workers are stopped after launcher writes the stop marker.

## Open Questions

1. Does `harbor.trial.trial.Trial` expose an explicit `close()`, `cleanup()`, or
   cancellation API? The local Codex environment did not have the Harbor package
   importable, so this should be inspected inside the training container.
2. Should cancelled speculative groups be dropped or requeued?
3. Should quiesce happen before every `update_weights()` only, or also before
   eval? Harbor fully async currently rejects evaluation mode, so update-weight
   is the main boundary.
4. Should we keep a small warm queue across multiple rollout IDs when
   `update_weights_interval > 1`? That is safe only if no weight update occurs
   between those rollout IDs.
5. Should shutdown kill TrialWorker actors, or only cancel their active remote
   refs? For normal operation, cancel refs first. Kill actors only as a hard
   fallback on timeout.

## Recommended Next Step

Implement in this order:

1. Re-run the fully async smoke and compare tail logs before/after the shutdown
   patch.
2. Add a true quiesce method before `actor_model.update_weights()`.
3. Add generation epoch / stale queue guards.
4. Add launcher-level node cleanup checks as a final hard-failure backstop.

The highest-priority fix is quiescing before `update_weights()`. That is both
a cleanliness fix and a training-correctness fix, because it prevents hidden
old-weight Harbor rollout from crossing the SGLang weight-update boundary.
