#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export CONFIG="${CONFIG:-${SLIME_DIR}/examples/harbor_terminal_rl/harbor_terminal_bench2_v030_trajectory_normalized_case_study.yaml}"

CASE_ROOT="${CASE_ROOT:-/path/to/workspace/slime_harbor_v030_case_study}"
export BASE_FOLDER="${BASE_FOLDER:-${CASE_ROOT}/run/no_summary_normalized}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_v030_no_summary_normalized_smoke}"
if [ -z "${SAVE_DEBUG_TRAIN_DATA:-}" ]; then
  export SAVE_DEBUG_TRAIN_DATA="${CASE_ROOT}/train_data/no_summary_normalized/{rollout_id}_{rank}.pt"
fi

export EXPECTED_NODES="${EXPECTED_NODES:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"
export RAY_NODE_IP_PREFIX="${RAY_NODE_IP_PREFIX:-29.73}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"

export USE_WANDB="${USE_WANDB:-0}"
export CLEAN_START="${CLEAN_START:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

exec bash "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
