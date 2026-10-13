#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROLE="${1:-launcher}"

if [ -n "${DAYTONA_ENV_FILE:-}" ]; then
  # shellcheck disable=SC1090
  set -a
  source "${DAYTONA_ENV_FILE}"
  set +a
fi

if [ "${ROLE}" = "launcher" ]; then
  if [ -z "${DAYTONA_API_KEY:-}" ] && { [ -z "${DAYTONA_JWT_TOKEN:-}" ] || [ -z "${DAYTONA_ORGANIZATION_ID:-}" ]; }; then
    echo "ERROR: set DAYTONA_API_KEY, or both DAYTONA_JWT_TOKEN and DAYTONA_ORGANIZATION_ID, before launching Daytona smoke." >&2
    exit 2
  fi
fi

RUN_TAG="${RUN_TAG:-daytona_2train2rollout_60x_$(date +%Y%m%d_%H%M%S)}"

export RUN_TAG
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rdaytona60}"
export RESTART_TMUX="${RESTART_TMUX:-1}"
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_32gpu_9b_daytona_retokenized_2train2rollout_60x_smoke.yaml}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_daytona_2train2rollout_60x_retokenized_smoke_${RUN_TAG}}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_full89_32gpu_daytona_2train2rollout_60x_retokenized_smoke_${RUN_TAG}}"
export RUN_TRIALS_DIR="${RUN_TRIALS_DIR:-/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_9b_daytona_2train2rollout_60x_retokenized_${RUN_TAG}}"

export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-16}"
export EXPECTED_NODES="${EXPECTED_NODES:-4}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"

export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-32768}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"
export INSTALL_HARBOR_RUNTIME_DEPS="${INSTALL_HARBOR_RUNTIME_DEPS:-1}"
export INSTALL_DAYTONA="${INSTALL_DAYTONA:-1}"

exec "${SCRIPT_DIR}/start_full89_32gpu_9b_actor1_rollout3_batch3_n32_trajectory_normalized_fully_async_tmux.sh" "$@"
