#!/bin/bash
# Production: strongest PPO recipe with quality-first TP8 rollout engines.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tmax15k_56gpu_35b_a3b_daytona_ppo_criticlr30x_strongest_tp8.yaml}"
# 24 rollout GPUs => 3 independent TP8 engines. No DP-attention or EP path.
export ROLLOUT_NUM_GPUS_PER_ENGINE=8
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-mamba-scheduler-strategy extra_buffer --sglang-disable-custom-all-reduce}"
# Required production identity and schedule.
export HARBOR_ROLLOUT_MODE=standard
export ROLLOUT_BATCH_SIZE=256
export GLOBAL_BATCH_SIZE=256
export N_SAMPLES_PER_PROMPT=1
# Terminal-Bench leaderboard evaluation: task-level pass@3. Training rollout remains n=1.
export N_SAMPLES_PER_EVAL_PROMPT=3
export EVAL_TEMPERATURE=0.6
export EVAL_TOP_P=0.95
export EVAL_TOP_K=20
export OVER_SAMPLING_BATCH_SIZE=260
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"
export WANDB_GROUP="${WANDB_GROUP:-q35-35b-a3b-tmax15k-56g-ppo-strongest-standard-tp8-os260-bs256-eval-pass3}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_tmax15k_56gpu_ppo_strongest_standard_tp8}"
exec "${SCRIPT_DIR}/run_tmax15k_35b_a3b_56gpu_ppo_criticlr30x_strongest.sh"
