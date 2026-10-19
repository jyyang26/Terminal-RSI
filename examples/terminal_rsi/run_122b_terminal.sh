#!/usr/bin/env bash
# Terminal-RSI | Qwen3.5-122B-A10B Terminal RL (PPO actor-critic, CTRF progress reward).
#
#   model  : Qwen3.5-122B-A10B MoE, 84k context
#   layout : 64 train GPUs (TP2 x PP4 x CP4, EP16) + 64 rollout GPUs (8 SGLang engines x 8)
#   reward : R = C * P / D + (1 - C) * S
#            slime_plugins.harbor.partial_reward.test_count_progress_rewards
#   batch 512 (oversample 560), 100 rollouts, actor lr 1e-6 / critic lr 1.5e-5 (per-role),
#   PPO clip 0.2, 2 critic-only warm-up steps, rollout routing replay, weights-only saves
#
# Fresh start from a torch_dist checkpoint (cold value head):
#   bash examples/terminal_rsi/prepare_load_root.sh /ckpt/Qwen3.5-122B-A10B_torch_dist/iter_XXXXXXX none /ckpt/loadroot_122b
#   WARM_CRITIC=0 LOAD_ROOT=/ckpt/loadroot_122b HF_CKPT=/path/to/Qwen3.5-122B-A10B \
#   TASKS_DIR=/path/to/strategy15k/tasks EVAL_TASKS_DIR=/path/to/terminal-bench-2 \
#   HARBOR_SRC=/path/to/harbor/src DAYTONA_API_KEY=... \
#     bash examples/terminal_rsi/run_122b_terminal.sh
#
# Resume a previous run (LOAD_ROOT = its save dir, which holds both actor and _critic):
#   LOAD_ROOT=<prev run>/checkpoints REF_LOAD=/ckpt/loadroot_122b START_ROLLOUT_ID=40 ... \
#     bash examples/terminal_rsi/run_122b_terminal.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL=qwen3.5-122b-a10b
PHASE=terminal
HARBOR_CONFIG="${RSI_DIR}/configs/harbor_122b_terminal.yaml"
MEGATRON_ROLE_CONFIG="${RSI_DIR}/configs/megatron_roles_122b.yaml"
REWARD_FN=slime_plugins.harbor.partial_reward.test_count_progress_rewards
ACTOR_LR=5e-7

rpr_launch
