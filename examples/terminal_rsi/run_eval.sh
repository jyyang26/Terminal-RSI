#!/usr/bin/env bash
# Terminal-RSI | rollout-only evaluation of a HuggingFace checkpoint on a Harbor task pool.
#
# Reuses the RPR model / SGLang / Harbor-agent stack, but starts no trainer or critic:
# SGLang serves HF_CKPT directly, the evaluation pool in EVAL_TASKS_DIR is run once
# (N_SAMPLES_PER_EVAL_PROMPT samples per task, temperature 1.0 / top-p 0.95 / top-k 20,
# matching in-training evaluation) and the job exits. Per-task Harbor trials are written
# under OUTPUT_ROOT; aggregate metrics go to the log (and W&B if enabled).
#
# Usage:
#   HF_CKPT=/path/to/Terminal-RSI-27B EVAL_TASKS_DIR=/path/to/terminal-bench-2 \
#   HARBOR_SRC=/path/to/harbor/src DAYTONA_API_KEY=... \
#     bash examples/terminal_rsi/run_eval.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EVAL_ONLY=1
PHASE=eval
HARBOR_CONFIG="${RSI_DIR}/configs/harbor_rpr2_spr.yaml"
MEGATRON_ROLE_CONFIG="${RSI_DIR}/configs/megatron_roles_rpr2.yaml"
REWARD_FN=slime_plugins.harbor.partial_reward.success_smoothed_progress_rewards
ACTOR_LR=0

rpr_launch
