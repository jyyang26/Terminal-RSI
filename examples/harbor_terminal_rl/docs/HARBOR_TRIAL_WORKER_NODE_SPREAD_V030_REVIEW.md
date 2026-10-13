# Harbor TrialWorker Node-Spread v0.3 Review

Status: implemented locally, needs real cluster smoke.
Date: 2026-06-06.

## Why This Change Exists

The previous 32GPU full89 / 9B / `trajectory_normalized` fully async run was
reclaimed for low GPU utilization before the first train batch completed.

The confirmed state before reclaim was:

- Ray placement group had 4 active nodes and 32 GPUs.
- SGLang engines were up.
- Harbor rollout had started real terminal-bench tasks.
- 16 `trajectory.json` files were observed on node0.
- No `rollout_dumps/*.pt` had been produced yet.

That run used:

```yaml
harbor_trial_node_spread: false
harbor_num_concurrent_trials: 16
```

With `rollout_batch_size=16`, `n_samples_per_prompt=16`, and
`harbor_group_unit=trajectory`, the first train batch needs 256 completed
terminal trajectories. Running only 16 in-flight terminal trials means the
actor side can wait a long time before seeing any train data, while the rollout
side is also mostly CPU / IO / container bound between LLM calls.

The immediate goal of this change is to restore multi-node Harbor trial
execution so terminal containers are not all launched from the RolloutManager
node and the rollout side can keep more requests in flight.

## Design Decision

We did not directly copy the old Harbor-specific `TrialWorker` class back into
core `slime/ray/rollout.py`.

Instead, the v0.3 design is:

- slime core provides a thin rollout auxiliary worker bridge.
- Harbor provides the actual worker implementation in the plugin.
- Core pins plugin workers to rollout placement-group bundles.
- Harbor rollout consumes the resulting worker handles through the existing
  `_harbor_trial_workers` path.

This keeps Harbor / Singularity / terminal-bench logic inside
`slime_plugins.harbor`, while allowing core to expose the one thing only core
knows reliably: which placement-group bundles belong to rollout nodes.

## High-Level Flow

```text
create_placement_groups()
  -> returns rollout placement tuple with bundle ids, gpu ids, node ips

RolloutManager.__init__()
  -> start rollout servers
  -> create rollout auxiliary workers if configured
  -> one worker per unique rollout node
  -> each worker pinned to a representative rollout bundle

RolloutManager.generate()
  -> health-check workers
  -> attach handles to args._harbor_trial_workers
  -> call Harbor rollout function

Harbor rollout
  -> _run_one_with_semaphore()
  -> round-robin worker.run_trial.remote(...)
  -> HarborTrialWorker runs _run_one_trial() on its own node
  -> Singularity / Apptainer container starts on that node
```

## Files Changed

### `slime/ray/placement_group.py`

Purpose: preserve node placement metadata.

Changes:

- `_create_placement_group()` now returns a 4-tuple:

```python
(pg, reordered_bundle_indices, reordered_gpu_ids, reordered_node_ips)
```

- `reordered_node_ips` is parallel to the reordered bundle and GPU lists.
- Added `unique_node_bundles(reordered_bundle_indices, reordered_node_ips)`.
  It picks the first bundle for each distinct node in reordered order.
- `create_placement_groups()` now carries node IP lists into both actor and
  rollout placement tuples.

Review focus:

- Existing consumers must tolerate 4-tuples.
- The rollout tuple should contain only the rollout slice after
  `rollout_offset`, not actor bundles in non-colocate runs.

### `slime/ray/actor_group.py`

Purpose: keep train actor placement compatible with the new 4-tuple.

Change:

```python
pg, reordered_bundle_indices, _reordered_gpu_ids, *_ = pg
```

Review focus:

- Actor placement still uses the same bundle index list.
- Node IP metadata is ignored by train actors.

### `slime/ray/rollout.py`

Purpose: add the generic-ish rollout auxiliary worker bridge.

Key additions:

- `RolloutManager._rollout_aux_worker_cls_path()`
- `RolloutManager._maybe_spawn_rollout_aux_workers()`
- `RolloutManager._attach_rollout_aux_workers_to_args()`
- `RolloutManager._kill_rollout_aux_workers()`

Behavior:

- If `args.rollout_aux_worker_cls_path` is set, core uses that class.
- If `args.harbor_trial_node_spread` is true, core defaults to:

```text
slime_plugins.harbor.trial_worker.HarborTrialWorker
```

- Workers are created with:

```python
PlacementGroupSchedulingStrategy(
    placement_group=pg_obj,
    placement_group_capture_child_tasks=False,
    placement_group_bundle_index=bundle_idx,
)
```

- Per-node concurrency is read from:

```text
rollout_aux_worker_max_concurrency
or harbor_trials_per_node
or 4
```

- CPU allocation is read from:

```text
rollout_aux_worker_num_cpus
or harbor_trial_worker_num_cpus
or 0.05
```

- Startup and health-check timeouts use the generic names when present, falling
  back to the Harbor names.
- Worker handles are attached to both:

```text
args._rollout_aux_workers
args._harbor_trial_workers
```

The Harbor-specific `_harbor_trial_workers` assignment is intentionally a
backward-compatible bridge for the existing plugin dispatcher.

Review focus:

- Core still does not import Harbor unless the Harbor config asks for it through
  `load_function()`.
- `dispose()` now kills auxiliary workers.
- `eval()` and train rollout both refresh worker health before calling the
  rollout function.
- Health check can keep busy workers when
  `harbor_kill_unhealthy_trial_workers=false`.

### `slime_plugins/harbor/trial_worker.py`

Purpose: keep Harbor trial execution in the plugin.

New class:

```python
class HarborTrialWorker:
    async def run_trial(self, rollout_id, group_index, sample_rank, base_sample):
        from slime_plugins.harbor.rollout import _run_one_trial
        return await _run_one_trial(...)

    def get_node_ip(self):
        return ray.util.get_node_ip_address()
```

Review focus:

- `_run_one_trial` is lazy-imported so core does not hard-depend on Harbor.
- The worker logs its node IP and rollout/group/sample identifiers.
- The Singularity / Apptainer child process should be spawned from this worker
  process, therefore on the worker's node.

### `slime_plugins/harbor/rollout.py`

Purpose: make missing worker behavior safe.

Existing code already supported:

```python
args._harbor_trial_workers
```

and dispatched remote trial calls via:

```python
worker.run_trial.remote(...)
```

New behavior:

- If no workers exist and `harbor_fallback_to_inprocess_trials` allows fallback,
  behavior remains old/in-process.
- If no workers exist and fallback is disabled, it raises immediately:

```text
Harbor trial node-spread is enabled but no TrialWorkers are available...
```

This avoids accidentally moving high-concurrency container rollout back onto the
RolloutManager node.

Review focus:

- Failure is intentional when node-spread is required but workers failed to
  start.
- Existing single-node or fallback-enabled runs still work.

### `examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_trajectory_normalized.yaml`

Purpose: switch the full89 trajectory-normalized config away from the previous
low-concurrency in-process trial execution.

Current important values:

```yaml
harbor_sample_mode: trajectory_normalized
harbor_group_unit: trajectory
harbor_num_concurrent_trials: 64
harbor_trial_node_spread: true
harbor_trials_per_node: 32
harbor_trial_worker_num_cpus: 0.05
harbor_trial_worker_startup_timeout_sec: 60
harbor_trial_worker_healthcheck_timeout_sec: 30
harbor_kill_unhealthy_trial_workers: false
harbor_fallback_to_inprocess_trials: false
```

For the 32GPU actor2/rollout2 layout, this expects 2 rollout nodes, so the
intended terminal trial capacity is:

```text
2 rollout nodes * 32 trials per node = 64 in-flight trials
```

Review focus:

- 64 is intentionally moderate for the first restored run.
- `fallback=false` is intentional so a missing TrialWorker setup fails loudly.
- If the next smoke shows worker pressure or Ray actor health-check problems,
  lower `harbor_trials_per_node` to 16 or 24 first.

## Tests Added / Updated

### `tests/test_harbor_rollout_async.py`

Added:

- `NUM_GPUS = 0`
- `__main__` pytest entrypoint
- `test_run_one_with_semaphore_fails_without_workers_when_spread_requires_remote`

The new test verifies that when node-spread is required and fallback is disabled,
Harbor does not silently call `_run_one_trial()` in the RolloutManager process.

### `tests/test_rollout_aux_workers.py`

New CPU-only test file.

It stubs Ray imports and verifies:

```python
unique_node_bundles(
    [8, 9, 0, 1, 16],
    ["10.0.0.2", "10.0.0.2", "10.0.0.1", "10.0.0.1", "10.0.0.3"],
)
==
[(8, "10.0.0.2"), (0, "10.0.0.1"), (16, "10.0.0.3")]
```

This protects the representative-bundle selection used for per-node worker
pinning.

## Local Validation Already Run

Compilation:

```bash
PYTHONPYCACHEPREFIX=/tmp/harbor_v030_pycache python -m py_compile \
  slime/ray/placement_group.py \
  slime/ray/actor_group.py \
  slime/ray/rollout.py \
  slime_plugins/harbor/rollout.py \
  slime_plugins/harbor/fully_async_rollout.py \
  slime_plugins/harbor/trial_worker.py \
  tests/test_harbor_rollout_async.py \
  tests/test_rollout_aux_workers.py
```

Result: passed.

Tests:

```bash
python -m pytest \
  tests/test_harbor_rollout_async.py \
  tests/test_rollout_aux_workers.py \
  tests/test_harbor_fully_async_rollout.py
```

Result:

```text
12 passed
```

## Required Cluster Smoke Before Full Retry

Before restarting full89, run a smaller node-spread smoke and check logs for:

```text
Spawned rollout auxiliary worker slime_plugins.harbor.trial_worker.HarborTrialWorker ...
Total rollout auxiliary workers spawned: 2
[HarborTrialWorker <ip>] running r1 g0 s0
[HarborTrialWorker <different-ip>] running r1 g0 s1
```

Also check process distribution:

```bash
ps -ef | grep -E '[a]pptainer|[s]ingularity' | wc -l
```

Expected:

- Apptainer / Singularity processes appear on rollout nodes, not only node0.
- Each rollout node stays roughly at or below `harbor_trials_per_node`.
- `harbor/trial_worker_count` metric should be nonzero.
- If `harbor_fallback_to_inprocess_trials=false`, missing workers should fail
  early instead of running in-process.

## Main Risks / Review Questions

1. Placement tuple shape changed from 3 to 4 elements.
   Known consumers in `ServerGroup` and `RayTrainGroup` were changed to use
   `*_`. Review should look for any hidden unpacking outside the grep results.

2. Auxiliary workers are created after rollout servers in `RolloutManager`.
   This is intentional because it uses the same rollout placement tuple and only
   tiny CPU. If scheduling pressure appears, worker creation order can be
   revisited.

3. Health checks use the same async actor concurrency pool as trials.
   For high `harbor_trials_per_node`, a fully occupied worker may not answer
   health check immediately. The full89 config sets
   `harbor_kill_unhealthy_trial_workers=false` to avoid killing busy workers.

4. This is not a fully generic public plugin API yet.
   The bridge has generic names, but Harbor compatibility is explicitly wired
   by setting `_harbor_trial_workers` when `harbor_trial_node_spread=true`.
   That is deliberate for this migration.

5. This change improves rollout throughput, but it does not by itself fix the
   fully async drain / weight-boundary issue documented separately in
   `HARBOR_FULLY_ASYNC_SHUTDOWN_DRAIN_NOTES.md`.

## Suggested Next Steps

1. Run node-spread smoke with a small batch.
2. Confirm worker spawn count and multi-node container distribution.
3. If healthy, rerun full89 with the updated 64-trial config.
4. If utilization is still low, reduce first-batch size or make fully async
   return smaller batches before raising trial concurrency further.
