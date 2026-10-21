# Terminal-RSI training recipes

Self-contained launchers for the three Terminal-RSI training setups. Every path and
credential comes from environment variables; nothing here is tied to a cluster.

| Launcher | Model | Reward | Context |
| --- | --- | --- | --- |
| `run_rpr1_fdr.sh` | Qwen3.8-27B (dense) | FDR | 256k |
| `run_rpr2_spr.sh` | Qwen3.8-27B (dense) | SPR | 256k |
| `run_122b_terminal.sh` | Qwen3.5-122B-A10B (MoE) | CTRF progress (`P / 20`) | 84k |

| File | Purpose |
| --- | --- |
| `run_rpr1_fdr.sh` | **RPR-1**: actor RSR-1 + Warm-up Critic, **Family Dense Reward**, actor lr `5e-7` |
| `run_rpr2_spr.sh` | **RPR-2**: actor RSR-2 + retained RPR-1 critic, **Success-Prioritized Reward**, actor lr `1e-7` |
| `run_122b_terminal.sh` | **Qwen3.5-122B-A10B Terminal RL**: PPO actor-critic, CTRF progress reward, actor lr `1e-6` |
| `run_eval.sh` | Rollout-only evaluation of an HF checkpoint on a Harbor task pool |
| `prepare_load_root.sh` | Builds the actor / `_critic` load-root pair from two torch_dist iteration dirs |
| `common.sh` | Shared argument builder + `ray job submit` (sourced, not run directly) |
| `configs/harbor_rpr1_fdr.yaml` | Harbor agent/env/rollout config + FDR per-family divisors (`D_f`) |
| `configs/harbor_rpr2_spr.yaml` | Same, with the SPR-retuned divisors (`D'_f`) |
| `configs/harbor_122b_terminal.yaml` | Harbor config for the 122B run (84k agent window, batch 512, 560 concurrent trials) |
| `configs/megatron_roles_{rpr1,rpr2,122b}.yaml` | Per-role actor/critic optimizer overrides |

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

Configure resources for your environment before launching. No experiment-specific
resource defaults are supplied:

| Mode | Required resource variables |
| --- | --- |
| Training | `ACTOR_NUM_NODES`, `ACTOR_NUM_GPUS_PER_NODE`, `TENSOR_MODEL_PARALLEL_SIZE`, `PIPELINE_MODEL_PARALLEL_SIZE`, `CONTEXT_PARALLEL_SIZE` |
| Training and evaluation | `ROLLOUT_NUM_GPUS`, `ROLLOUT_NUM_GPUS_PER_ENGINE` |
| MoE training | Additionally `EXPERT_MODEL_PARALLEL_SIZE`, `EXPERT_TENSOR_PARALLEL_SIZE` |

All values must be positive integers and satisfy the model/backend parallelism and
cluster-capacity constraints. Training resources must accommodate the chosen
parallel layout; rollout resources must divide evenly into engines. Rollout-only
evaluation does not require training resource variables. `DRY_RUN=1` prints your
configured command and writes the resolved Harbor configuration without starting a job.

```bash
# 0. Start a Ray cluster and export your resource configuration.
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

# 4. Qwen3.5-122B-A10B Terminal RL (fresh start, cold value head)
bash examples/terminal_rsi/prepare_load_root.sh /ckpt/Qwen3.5-122B-A10B_torch_dist/iter_XXXXXXX none /ckpt/loadroot_122b
MODEL_HF=/path/to/Qwen3.5-122B-A10B
WARM_CRITIC=0 HF_CKPT=$MODEL_HF LOAD_ROOT=/ckpt/loadroot_122b \
TASKS_DIR=/path/to/strategy15k/tasks bash examples/terminal_rsi/run_122b_terminal.sh

# ... or resume it: LOAD_ROOT = the previous run's save dir, REF_LOAD = the original root
HF_CKPT=$MODEL_HF LOAD_ROOT=/outputs/terminal_rsi_122b_terminal_<tag>/checkpoints \
REF_LOAD=/ckpt/loadroot_122b START_ROLLOUT_ID=40 TASKS_DIR=/path/to/strategy15k/tasks \
  bash examples/terminal_rsi/run_122b_terminal.sh
```

The 122B config trains on the tail of the 15k strategy pool (`harbor_tasks_offset: 11900`);
set it to `0` to use the whole `TASKS_DIR`. It uses rollout routing replay
(`--use-rollout-routing-replay`) so MoE training replays the rollout's expert choices, and
saves weights only (`--no-save-optim`).

The Warm-up Critic is prepared the same way from a cold value head: build an actor-only
root (`prepare_load_root.sh <actor> none <root>`) and run with `WARM_CRITIC=0` and
`NUM_CRITIC_ONLY_STEPS=NUM_ROLLOUT` so that only the critic is updated.

RPR-1 enables TB-GAE by default through `configs/harbor_rpr1_fdr.yaml`, with
`tb_gae_gamma_turn: 1.0` and `tb_gae_lambda_turn: 1.0`. Explicit CLI flags override
these YAML settings; use `EXTRA_ARGS="--no-enable-tb-gae"` to disable it.
TB-GAE requires a Harbor Terminus-2 implementation supporting termination metadata
version 2 and truncated-trajectory bootstrap state.

Useful switches: `DRY_RUN=1` prints the full command without launching, `DIRECT_RUN=1`
runs the driver in the current shell instead of `ray job submit`, `RAY_DASHBOARD` points
at the job API, `MODEL=qwen3.5-122b-a10b` switches `run_eval.sh` to the 122B profile,
`SAVE_ROLLOUT_DUMPS=1` writes per-step rollout dumps (large) and
`EXTRA_ARGS="..."` appends raw flags (e.g. `--enable-tb-gae --tb-gae-gamma-turn 0.99
--tb-gae-lambda-turn 0.95` for turn-boundary GAE).

Outputs go to `${OUTPUT_ROOT:-./outputs}/terminal_rsi_<27b|122b>_<phase>_<RUN_TAG>/`: actor
checkpoints in `checkpoints/`, critic in `checkpoints_critic/`, Harbor trials in
`harbor_trials/` and the materialised config in `harbor_config_resolved.yaml`.
