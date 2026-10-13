# Harbor v0.3 32GPU Smoke Environment Fixes

Date: 2026-06-06

## Context

Target job:

- Taiji instance: `self_improving_terminal_agent_RL_32GPU_slime_v4 / 8B1D80239E921BC7019E98D627820C74`
- Smoke config: `examples/harbor_terminal_rl/harbor_terminal_bench2_v030_fully_async_cleanup_smoke.yaml`
- Mode: fully async rollout + TrialWorker node-spread + `trajectory_normalized`
- Model: Qwen3.5-9B

The old Harbor v0.2 path used the same basic Singularity command shape:

- `--writable-tmpfs`
- `--fakeroot`
- `--containall`
- `--pid`
- `--no-mount home,tmp,bind-paths,proc`
- `-B /proc:/proc`

On the new Taiji nodes, that exact path failed before trial execution.

## Findings

### 1. `--fakeroot` enters the wrong runtime path on this pod

With `--fakeroot`, Singularity tried to use FUSE/rootless support. Before installing node-local helpers it failed with missing `fusermount`; after adding `fuse3/uidmap/subuid` on one node it could start, but `--writable-tmpfs` was disabled under underlay and the rootfs became read-only.

Canary result:

- Old args with `--fakeroot`: container starts only after extra node packages, but `/tmp` or `/etc` can become read-only.
- Same args without `--fakeroot`: container starts as root and `/tmp` plus `/etc` are writable.

Fix:

- Added `singularity_fakeroot` to `SlimeSingularityEnvironment`.
- Current smoke/full89 configs set `singularity_fakeroot: false`.

### 2. Dropping all capabilities is too aggressive

Dropping all capabilities avoids Singularity's capability-set error, but breaks commands inside the container:

```text
su: cannot set groups: Operation not permitted
```

That caused tmux/verifier setup failures and zero-reward failed samples.

Canary result:

- `--drop-caps ALL`: container writable, but `su` fails.
- `--drop-caps CAP_PERFMON,CAP_BPF,CAP_CHECKPOINT_RESTORE`: container writable and `su -s /bin/sh root -c id` works.

Fix:

- Added `singularity_drop_caps` to `SlimeSingularityEnvironment`.
- Current smoke/full89 configs set:

```yaml
singularity_drop_caps: CAP_PERFMON,CAP_BPF,CAP_CHECKPOINT_RESTORE
```

### 3. Worker IP auto-selection picked the wrong NIC

`hostname -I | awk '{print $1}'` selected `29.73.*` / `29.97.*` addresses on workers. Ray initially joined but later lost heartbeats.

Correct worker IPs for this job:

- node1: `29.79.8.37`
- node2: `29.79.9.235`
- node3: `29.81.228.89`

Fix:

- `pick_ray_node_ip` now supports comma-separated prefixes.
- If no prefix matches, it prefers an IP that appears more than once in `hostname -I`, which matches the stable Taiji service IP pattern on this job.
- For the verified smoke, `WORKER_ADDR` was still passed explicitly.

### 4. Stale stop marker can stop workers immediately

Reusing `BASE_FOLDER` can leave `.stop_workers_<head>_<port>` from a prior run. If workers start before the launcher removes it, they can immediately exit.

Fix:

- Worker keep-alive now ignores and removes a stop marker older than the worker process start time.

### 5. Cleanup should not force-cancel Ray actor tasks

During successful shutdown, over-sampled TrialWorker actor tasks can be cancelled. Ray does not support `force=True` for actor tasks, which produced a traceback even when the job succeeded.

Fix:

- Changed TrialWorker task cancellation in `slime_plugins/harbor/rollout.py` to `ray.cancel(ref, force=False)`.

## Verified Smoke Result

Clean run used:

- `BASE_FOLDER=/path/to/workspace/slime_harbor_v030_fully_async_cleanup_smoke/run_mincaps`
- `SAVE_SUBDIR=Qwen3.5-9B_v030_cleanup_smoke_mincaps`
- `SAVE_DEBUG_TRAIN_DATA=/path/to/workspace/slime_harbor_v030_fully_async_cleanup_smoke/train_data_mincaps/{rollout_id}_{rank}.pt`

Observed result:

- Ray cluster reached 4 nodes / 32 GPUs.
- Fully async rollout function loaded.
- Cleanup function loaded.
- TrialWorker node-spread spawned 3 workers.
- Singularity servers started with the new no-fakeroot/minimal-drop-caps settings.
- No `su: cannot set groups` errors in the clean run.
- No `No reward file` trial failures in the clean run.
- Rollout 1 completed:
  - `harbor/num_samples: 4`
  - `harbor/num_converted_samples: 4`
  - `harbor/num_trajectory_samples: 4`
  - `harbor/num_converted_trajectories: 4`
  - `harbor/status_completed_samples: 4`
  - `harbor/num_failed_trials: 0`
- Actor train completed one microbatch.
- Checkpoint iteration 1 saved successfully.
- Ray job succeeded.

Remaining caveat:

- The smoke was intentionally tiny (`harbor_tasks_limit: 1`, `max_turns: 2`, `num_rollout: 2`), so it validates the v0.3 fully-async path and environment startup, not full89 convergence or long-time stability.
