#!/bin/bash
# Two-rollout integration smoke for Qwen3.5-35B-A3B after SGLang PR27140 backport.
# Uses production topology but tiny batch; rollout 1 necessarily follows actor train,
# online weight update and release/resume of GDN recurrent state.
set -euo pipefail
export NUM_ROLLOUT=2
export NUM_CRITIC_ONLY_STEPS=0
export ROLLOUT_BATCH_SIZE=16
export GLOBAL_BATCH_SIZE=16
export OVER_SAMPLING_BATCH_SIZE=16
export N_SAMPLES_PER_PROMPT=1
export EVAL_INTERVAL=999
export SKIP_EVAL_BEFORE_TRAIN=1
export SAVE_INTERVAL=99
export WANDB_GROUP="${WANDB_GROUP:-q35-35b-a3b-pr27140-two-rollout-smoke}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_pr27140_two_rollout_smoke}"
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_tmax15k_35b_a3b_56gpu_ppo_criticlr30x_strongest.sh"
