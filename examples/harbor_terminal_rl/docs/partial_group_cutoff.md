# Harbor Partial Group Cutoff

## Background

Daytona currently caps Harbor at about 60 live 5GiB sandboxes. The 32 GPU
Daytona run uses two rollout groups per step and 32 trajectories per group,
so a full step can submit 64 trajectory trials while only 60 can run at once.
If the rollout waits for all 64 trajectories, a few terminal long-tail trials
can hold the train side idle and keep average GPU utilization low.

Slime's built-in oversampling operates at the prompt-group level: it can launch
extra full groups and cancel unused groups after enough complete groups finish.
It does not accept the first N trajectories inside one group. Harbor terminal
rollout needs a finer-grained cutoff: accept a group after enough trajectory
units finish, then cancel the remaining tail trial tasks.

## Implemented Behavior

New Harbor config:

```yaml
harbor_partial_group_min_units: 26
```

`harbor_partial_group_min_trajectories` is also accepted as an alias. A value
of 0 or an unset value preserves the old full-group behavior.

## Enable And Disable

Enable the feature by setting a positive partial threshold:

```yaml
harbor_partial_group_min_units: 26
```

For the current Daytona 60x run shape, this means each 32-trajectory prompt
group can return after 26 trajectory units finish. With two prompt groups per
rollout step, training can start from about 54 trajectories instead of waiting
for all 64.

Disable the feature by omitting the option or setting it to 0:

```yaml
harbor_partial_group_min_units: 0
```

When disabled, Harbor keeps the previous behavior:

- `_run_one_group` waits for every trajectory task in the group to finish or
  fail.
- `harbor_require_full_rollout_groups: true` drops incomplete groups instead
  of accepting them.
- short global-batch steps are not enabled by this feature.

Partial training steps require `USE_DYNAMIC_GLOBAL_BATCH_SIZE=1`; however,
`USE_DYNAMIC_GLOBAL_BATCH_SIZE=1` alone does not enable this feature. A positive
`harbor_partial_group_min_units` or explicit `allow_partial_global_batch: true`
is required before the DP scheduler keeps a short final step.

When enabled, `_run_one_group` keeps submitting the normal
`n_samples_per_prompt` trial tasks, but returns early once the converted group
has at least the configured number of units. For `trajectory_retokenized`,
`turn_grouped`, `step`, and `step_grouped`, the unit is trajectory; otherwise it
is sample. Remaining trial tasks are cancelled cooperatively so Harbor Trial can
run its cleanup/finalization path.

The fully async collector now accepts a partial group when:

- `harbor_require_full_rollout_groups: true`
- group units are below `n_samples_per_prompt`
- group units are at least `harbor_partial_group_min_units`

Groups below the partial threshold are still dropped/backfilled.

## Training Schedule

Partial groups can produce fewer training groups than the configured
`GLOBAL_BATCH_SIZE`. The DP scheduler now supports a final short step when
`--use-dynamic-global-batch-size` is enabled. The short step records its real
group count in `global_batch_sizes`, so loss scaling and LR scheduler progress
use the actual number of consumed groups.

This still requires dynamic token batching for practical Daytona runs. Static
microbatching can still fail alignment when the partial step shape is uneven.

## Metrics

Rollout metrics now include:

- `harbor/num_partial_groups_accepted`
- `harbor/num_partial_tail_trials_cancelled`
- `harbor/partial_group_min_units`

Existing group-level oversampling metrics are unchanged.

## Daytona Cleanup Notes

Tail cancellation uses normal asyncio cancellation and keeps Ray cancellation
cooperative (`force=False`). Harbor `Trial.run()` catches `CancelledError` and
always calls `_cleanup_and_finalize()`, which shields
`environment.stop(delete=True)`. Daytona environment creation is also shielded
so cancellation during create can still capture and delete the sandbox.

Smoke validation must still check the external Daytona state because a Ray task
cancel only proves cancellation was requested, not that the remote sandbox was
deleted. The smoke should verify:

- `harbor/num_partial_tail_trials_cancelled > 0`
- `harbor/num_partial_groups_accepted > 0`
- rollout returns before slow tail trials finish
- Daytona sandbox count returns to the expected level after rollout cleanup
- no stale `slime-r*-g*-s*` sandboxes remain after the job exits

## Current Daytona Smoke Config

The 32 GPU Daytona 60x smoke config enables:

```yaml
harbor_num_concurrent_trials: 60
harbor_fully_async_max_concurrent_groups: 2
harbor_partial_group_min_units: 26
harbor_require_full_rollout_groups: true
```

With `ROLLOUT_BATCH_SIZE=2` and `N_SAMPLES_PER_PROMPT=32`, the trainer can start
from roughly 54 completed trajectories instead of waiting for all 64.
