#!/usr/bin/env bash
# Terminal-RSI | RPR-2: second Recursive Policy-Refine phase with Success-Prioritized Reward.
#
#   actor  : RSR-2 (second Recursive Self-Rewrite stage, initialised from retained RPR-1)
#   critic : the retained RPR-1 critic, carried unchanged through actor-only RSR-2
#   reward : SPR  R = S + C * P / (P + D'_f)
#            slime_plugins.harbor.partial_reward.success_smoothed_progress_rewards
#   actor lr 1e-7 (1/5 of RPR-1), critic lr 1.5e-5, 4 critic-only calibration steps
#
# The paper's final model, Terminal-RSI-27B, is the RPR-2 checkpoint at local step 18.
#
# Usage:
#   bash examples/terminal_rsi/prepare_load_root.sh <RSR-2 actor iter dir> <RPR-1 critic iter dir> /path/to/loadroot_rpr2
#   HF_CKPT=/path/to/Qwen3.8-27B LOAD_ROOT=/path/to/loadroot_rpr2 \
#   TASKS_DIR=/path/to/swr_tasks EVAL_TASKS_DIR=/path/to/terminal-bench-2 \
#   HARBOR_SRC=/path/to/harbor/src DAYTONA_API_KEY=... \
#     bash examples/terminal_rsi/run_rpr2_spr.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PHASE=rpr2
HARBOR_CONFIG="${RSI_DIR}/configs/harbor_rpr2_spr.yaml"
MEGATRON_ROLE_CONFIG="${RSI_DIR}/configs/megatron_roles_rpr2.yaml"
REWARD_FN=slime_plugins.harbor.partial_reward.success_smoothed_progress_rewards
ACTOR_LR=1e-7

rpr_launch
