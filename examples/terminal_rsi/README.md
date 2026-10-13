# Terminal-RSI recipe: Recursive Policy-Refine (RPR) on Qwen3.8-27B

Self-contained launchers for the reinforcement-learning phases of Terminal-RSI. Every
path and credential comes from environment variables; nothing here is tied to a cluster.

| File | Purpose |
| --- | --- |
| `run_rpr1_fdr.sh` | **RPR-1**: actor RSR-1 + Warm-up Critic, **Family Dense Reward**, actor lr `5e-7` |
| `run_rpr2_spr.sh` | **RPR-2**: actor RSR-2 + retained RPR-1 critic, **Success-Prioritized Reward**, actor lr `1e-7` |
| `run_eval.sh` | Rollout-only evaluation of an HF checkpoint on a Harbor task pool |
| `prepare_load_root.sh` | Builds the actor / `_critic` load-root pair from two torch_dist iteration dirs |
| `common.sh` | Shared argument builder + `ray job submit` (sourced, not run directly) |
| `configs/harbor_rpr1_fdr.yaml` | Harbor agent/env/rollout config + FDR per-family divisors (`D_f`) |
| `configs/harbor_rpr2_spr.yaml` | Same, with the SPR-retuned divisors (`D'_f`) |
| `configs/megatron_roles_rpr{1,2}.yaml` | Per-role actor/critic optimizer overrides |

## Rewards

Both rewards live in [`slime_plugins/harbor/partial_reward.py`](../../slime_plugins/harbor/partial_reward.py)
and are selected with `--custom-reward-post-process-path`. `P` is the number of passed
semantic CTRF checks (hygiene checks are excluded), `S` is strict terminal success, `C`
marks a valid CTRF report and `f` is the task family (read from
`tests/private/manifest.json`).

| Phase | Reward | Formula | Hook |
| --- | --- | --- | --- |
| RPR-1 | FDR | `R = C * P / D_f + (1 - C) * S` | `test_count_progress_rewards` |
| RPR-2 | SPR | `R = S + C * P / (P + D'_f)` | `success_smoothed_progress_rewards` |

The divisor tables are data, not code: `harbor_family_testcount_scales` in the YAML (global
fallback `harbor_testcount_scale: 82`). SPR doubles the divisor of high-scoring-failure
families, quarters it for long-horizon low-yield families and fills one missing family.

## Running

```bash
# 0. A Ray cluster with 64 training GPUs + 56 rollout GPUs must already be up.
export HF_CKPT=/path/to/Qwen3.8-27B             # tokenizer + config (and weights for eval)
export TASKS_DIR=/path/to/swr_tasks              # */task.toml training pool
export EVAL_TASKS_DIR=/path/to/terminal-bench-2  # */task.toml held-out pool
export HARBOR_SRC=/path/to/harbor/src
export DAYTONA_API_KEY=...                       # sandbox provider
export WANDB_API_KEY=...                         # optional; USE_WANDB=0 to disable

# 1. RPR-1 (FDR): RSR-1 actor + Warm-up Critic
bash examples/terminal_rsi/prepare_load_root.sh /ckpt/rsr1/iter_XXXXXXX /ckpt/warmup_critic/iter_XXXXXXX /ckpt/loadroot_rpr1
LOAD_ROOT=/ckpt/loadroot_rpr1 bash examples/terminal_rsi/run_rpr1_fdr.sh

# 2. RPR-2 (SPR): RSR-2 actor + critic retained from RPR-1
bash examples/terminal_rsi/prepare_load_root.sh /ckpt/rsr2/iter_XXXXXXX /ckpt/rpr1/checkpoints_critic/iter_XXXXXXX /ckpt/loadroot_rpr2
LOAD_ROOT=/ckpt/loadroot_rpr2 bash examples/terminal_rsi/run_rpr2_spr.sh

# 3. Evaluate a converted HF checkpoint
HF_CKPT=/ckpt/terminal_rsi_27b_hf bash examples/terminal_rsi/run_eval.sh
```

The Warm-up Critic is prepared the same way from a cold value head: build an actor-only
root (`prepare_load_root.sh <actor> none <root>`) and run with `WARM_CRITIC=0` and
`NUM_CRITIC_ONLY_STEPS=NUM_ROLLOUT` so that only the critic is updated.

Useful switches: `DRY_RUN=1` prints the full command without launching, `DIRECT_RUN=1`
runs the driver in the current shell instead of `ray job submit`, `RAY_DASHBOARD` points
at the job API, `SAVE_ROLLOUT_DUMPS=1` writes per-step rollout dumps (large) and
`EXTRA_ARGS="..."` appends raw flags (e.g. `--enable-tb-gae --tb-gae-gamma-turn 0.99
--tb-gae-lambda-turn 0.95` for turn-boundary GAE).

Outputs go to `${OUTPUT_ROOT:-./outputs}/terminal_rsi_27b_<phase>_<RUN_TAG>/`: actor
checkpoints in `checkpoints/`, critic in `checkpoints_critic/`, Harbor trials in
`harbor_trials/` and the materialised config in `harbor_config_resolved.yaml`.
