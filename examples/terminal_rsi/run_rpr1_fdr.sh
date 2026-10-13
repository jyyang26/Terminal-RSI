#!/usr/bin/env bash
# Terminal-RSI | RPR-1: first Recursive Policy-Refine phase with Family Dense Reward.
#
#   actor  : RSR-1 (output of the first Recursive Self-Rewrite stage)
#   critic : Warm-up Critic (one critic-only epoch on the SWR pool, actor frozen)
#   reward : FDR  R = C * P / D_f + (1 - C) * S
#            slime_plugins.harbor.partial_reward.test_count_progress_rewards
#   actor lr 5e-7, critic lr 1.5e-5, 4 critic-only calibration steps, 36 rollouts
#
# The paper retains the RPR-1 checkpoint at local step 20.
#
# Usage:
#   bash examples/terminal_rsi/prepare_load_root.sh <RSR-1 actor iter dir> <warm-up critic iter dir> /path/to/loadroot_rpr1
#   HF_CKPT=/path/to/Qwen3.8-27B LOAD_ROOT=/path/to/loadroot_rpr1 \
#   TASKS_DIR=/path/to/swr_tasks EVAL_TASKS_DIR=/path/to/terminal-bench-2 \
#   HARBOR_SRC=/path/to/harbor/src DAYTONA_API_KEY=... \
#     bash examples/terminal_rsi/run_rpr1_fdr.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PHASE=rpr1
HARBOR_CONFIG="${RSI_DIR}/configs/harbor_rpr1_fdr.yaml"
MEGATRON_ROLE_CONFIG="${RSI_DIR}/configs/megatron_roles_rpr1.yaml"
REWARD_FN=slime_plugins.harbor.partial_reward.test_count_progress_rewards
ACTOR_LR=5e-7

rpr_launch
