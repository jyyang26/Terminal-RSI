# Harbor Terminal RL migration to slime v0.3.0

Date: 2026-06-04

## Current anchors

- Old runnable worktree: `/path/to/workspace/code/slime`
- Old runnable commit: `07a25cc7 Add Harbor terminal RL integration`
- Safety branch: `harbor/pre-v030-runnable`
- Safety tag: `harbor-terminal-rl-runnable-20260604`
- New migration worktree: `/path/to/workspace/code/slime-v0.3-harbor`
- New base: `v0.3.0`, commit `bf14dc21 [release] bump to v0.3.0 (#1975)`
- Migration branch: `harbor/v030-port`

The old worktree stays runnable. All migration work happens in the v0.3.0 worktree.

## Migration goals

1. Keep Harbor / Terminus2 integration mostly plugin-owned.
2. Avoid blindly cherry-picking old slime core patches into v0.3.0.
3. Prefer v0.3.0 public hooks and contracts:
   - `--rollout-function-path`
   - `--custom-reward-post-process-path`
   - `--rollout-sample-filter-path`
   - `--rollout-all-samples-process-path`
   - `--rollout-data-postprocess-path`
   - `--loss-type custom_loss`
   - `Sample.group_id`
   - `slime.agent.trajectory`
4. Preserve the old runnable path as fallback until v0.3.0 passes smoke training.

## Key v0.3.0 interface changes and opportunities

### Rollout function

v0.3.0 expects:

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False):
    ...
```

Training should return `RolloutFnTrainOutput(samples=list[list[Sample]], metrics=dict)`.
Evaluation should return `RolloutFnEvalOutput(data=dict, metrics=dict)`.

The old Harbor rollout already follows this shape, so the module can be migrated first. The main changes are around hook contracts and using v0.3.0 group semantics.

### Grouped / fan-out samples

v0.3.0 has `Sample.group_id`. This is directly relevant to Harbor `turn_grouped` mode:

- One Terminus trajectory may fan out into many turn samples.
- All sibling turn samples should share the same `group_id`.
- slime v0.3.0 uses `group_id` in DP splitting and loss aggregation, so trajectory-level accounting no longer needs as much custom core logic.

Old code used metadata plus reward post-process to broadcast trajectory advantages. We should keep that hook initially for behavioral parity, but migrate ATIF conversion to set `group_id` explicitly.

### Agent trajectory helpers

v0.3.0 includes `slime.agent.trajectory.merge_turns`.

This matters because our manual data review found many `prefix_mismatch_turns`. Old trajectory mode rejected non-prefix prompt history. v0.3.0 merge logic handles prompt drift by keeping the common prefix and masking drifted / environment tokens. We should use this helper for future whole-trajectory mode instead of maintaining custom fragile prefix logic.

Initial migration can keep old `turn`, `last_turn`, and `turn_grouped` paths. Whole-trajectory mode should be revisited after the first smoke run.

### All-samples hook

v0.3.0 plugin contract expects:

```python
def process(args, all_groups: list[list[Sample]], data_source) -> None:
    ...
```

Old Harbor sample postprocess helpers are already compatible if they iterate nested groups. Verify tests against this contract.

### Reward shaping

v0.3.0 still supports:

```bash
--custom-reward-post-process-path slime_plugins.harbor.partial_reward.post_process_rewards
```

The CTRF progress reward can be migrated mostly unchanged. It should annotate both `metadata` and `train_metadata` so dumped train data remains inspectable.

### Custom loss

v0.3.0 has official `--loss-type custom_loss` and `--custom-loss-function-path`.

Old Harbor custom loss only skips entropy graph construction when `entropy_coef=0`. We should initially keep it as an optional path for memory/speed, but it is not required for functional correctness. It must be rechecked against v0.3.0 `slime.backends.megatron_utils.loss.policy_loss_function` before enabling in launch scripts.

## Old core patches: migration decision

### `slime/ray/rollout.py` TrialWorker node-spread patch

Old patch added Harbor-specific `TrialWorker` actors inside core rollout manager.

Migration decision: do not cherry-pick directly.

Reason:

- v0.3.0 has more explicit plugin contracts and agent examples.
- A Harbor-specific Ray actor inside `slime/ray/rollout.py` violates the plugin boundary.
- We should first run in-process / existing worker mode, then design a plugin-owned remote trial dispatcher if needed.

Near-term action:

- Keep the old implementation as reference.
- Do not modify v0.3.0 core for TrialWorker in phase 1.
- If rollout env remains bottleneck, add a plugin module that creates its own Ray actors from `generate_rollout`, or propose a small generic upstream hook for colocated auxiliary rollout workers.

### `slime/ray/placement_group.py` node IP return patch

Old patch extended placement group return tuple with node IPs for TrialWorker scheduling.

Migration decision: do not cherry-pick in phase 1.

Reason:

- It only served Harbor TrialWorker placement.
- Direct tuple shape changes risk breaking v0.3.0 core assumptions.

### `slime/ray/train_actor.py` `SLIME_SKIP_NUMA_AFFINITY`

Old patch avoided NUMA-local OOM on large shmem footprints.

Migration decision: inspect v0.3.0 first. If not present and 27B/64GPU still hits NUMA-local OOM, re-add as a tiny generic core patch with tests/docs. This patch is generic enough to keep, but not needed for first plugin migration.

### `slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py`

Old patch normalized `module.module.language_model.*` names and skipped VLM vision params.

Migration decision: compare against v0.3.0 qwen3.5 bridge. If missing, migrate this patch because it affects model conversion/weight sync correctness and is not Harbor-specific.

### `slime/ray/actor_group.py`

Old patch tolerated longer placement-group tuples.

Migration decision: do not migrate unless we also migrate placement tuple changes.

## Phase plan

### Phase 0: Safety and baseline

- Completed: old runnable commit saved as branch/tag.
- Completed: v0.3.0 worktree created on `harbor/v030-port`.
- Keep old train dumps and local files out of migration commit.

### Phase 1: Plugin-only import

Bring over:

- `slime_plugins/harbor/`
- `tests/test_harbor_*.py`
- `examples/harbor_terminal_rl/` docs/config/scripts, excluding data dumps:
  - exclude `1.pt`
  - exclude `trajectories_*.jsonl`
  - exclude `inspect_rollout_dump.ipynb`
  - exclude `*.local`
- top-level notes if still useful

Immediate code adjustments:

- Set `Sample.group_id` for all samples derived from the same Harbor trajectory.
- Keep `rollout_all_samples_process_path` functions compatible with `list[list[Sample]]`.
- Check `train_loss.py` against v0.3.0 loss function before wiring it in launch scripts.
- Keep TrialWorker/core-node-spread disabled in phase 1.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/harbor_v030_pycache python -m py_compile slime_plugins/harbor/*.py
python -m pytest tests/test_harbor_atif_adapter.py tests/test_harbor_partial_reward.py tests/test_harbor_sample_postprocess.py tests/test_harbor_turn_grouped_reward.py tests/test_harbor_rollout_async.py
```

`test_harbor_data_source.py` may require `ray`; run it only in an environment with v0.3.0 runtime deps.

### Phase 2: v0.3.0 native trajectory mode

- Use `slime.agent.trajectory.TurnRecord` and `merge_turns` for whole-trajectory ATIF conversion.
- Preserve response/environment masking.
- Record prefix drift metrics instead of treating drift as fatal.
- Add tests based on the real reviewed samples where old trajectory conversion failed.

### Phase 3: Launch config refresh

For 9B / 32GPU:

- context window aligned to 64k or 128k explicitly across rollout and train.
- enable progress reward.
- use `group_id` and `turn_grouped` mode.
- keep rollout concurrency conservative first.
- enable `--use-rollout-logprobs` only after Harbor samples validate complete `rollout_log_probs`.

For 27B / 64GPU:

- start with v0.3.0 default async mechanisms.
- avoid old Harbor TrialWorker core patch.
- re-evaluate NUMA skip and qwen3.5 bridge patch.

### Phase 4: Performance work

Only after phase 1/2 smoke runs:

- Evaluate v0.3.0 fully async rollout buffer.
- Evaluate `over_sampling_batch_size`, dynamic filter, and partial rollout applicability to terminal-agent trials.
- Reintroduce remote env workers as plugin-owned infrastructure if GPU utilization is still rollout-bound.

## First implementation checklist

- [x] Copy Harbor plugin/tests/examples into v0.3.0 worktree.
- [x] Add `group_id` assignment in ATIF conversion.
- [x] Adapt tests to v0.3.0 contracts.
- [x] Run py_compile.
- [x] Run Harbor unit tests that do not require cluster/runtime dependencies.
- [ ] Write first migration commit on `harbor/v030-port`.

## Phase 1 progress log

2026-06-04:

- Copied `slime_plugins/harbor/`, `tests/test_harbor_*.py`, and `examples/harbor_terminal_rl/` into the v0.3.0 worktree.
- Excluded local / data artifacts from the migration:
  - `examples/harbor_terminal_rl/1.pt`
  - `examples/harbor_terminal_rl/trajectories_*.jsonl`
  - `examples/harbor_terminal_rl/inspect_rollout_dump.ipynb`
  - `*.local`
- Added `Sample.group_id` assignment for:
  - `harbor_sample_mode=turn_grouped`, where all turns from one Terminus trajectory share one `group_id`;
  - `harbor_sample_mode=trajectory`, where the single trajectory sample also receives a stable `group_id`.
- Kept ordinary `harbor_sample_mode=turn` unchanged: each turn remains an independent training sample unless another hook groups it.
- Updated `tests/test_harbor_atif_adapter.py` to assert v0.3.0 grouping behavior.
- Ran syntax validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/harbor_v030_pycache python -m py_compile slime_plugins/harbor/*.py examples/harbor_terminal_rl/*.py
```

Result: passed.

- Ran Harbor unit tests that do not require cluster/runtime dependencies:

```bash
python -m pytest \
  tests/test_harbor_atif_adapter.py \
  tests/test_harbor_import_utils.py \
  tests/test_harbor_llm.py \
  tests/test_harbor_partial_reward.py \
  tests/test_harbor_rollout_async.py \
  tests/test_harbor_sample_postprocess.py \
  tests/test_harbor_turn_grouped_reward.py
```

Result: `29 passed`.

- Secret scan did not find any real W&B token. Matches were limited to placeholders, environment-variable based SSH password handling, and task names containing the word `secret`.
