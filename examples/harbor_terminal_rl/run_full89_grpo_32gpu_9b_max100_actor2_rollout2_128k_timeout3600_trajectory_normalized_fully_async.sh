#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

# v0.3 Harbor port, not the old /code/slime tree.
export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_trajectory_normalized.yaml}"

export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor2_rollout2_fully_async_trajectory_normalized_timeout3600}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_full89_32gpu_actor2_rollout2_fully_async_trajectory_normalized_timeout3600}"

export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-9B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-9B_torch_dist}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"

export EXPECTED_NODES="${EXPECTED_NODES:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-16}"

export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# 89 tasks * 5 epochs = 445 task groups. With start_rollout_id=1 and
# rollout_batch_size=16, NUM_ROLLOUT=29 trains rollout ids 1..28, covering
# 448 groups.
export NUM_ROLLOUT="${NUM_ROLLOUT:-29}"

export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-131072}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-131072}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SEQ_LENGTH="${SEQ_LENGTH:-131072}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-131072}"

export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"

export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

export USE_WANDB="${USE_WANDB:-0}"
export WANDB_GROUP="${WANDB_GROUP:-q35-9b-tb2-full89-32g-fullyasync-trajnorm-timeout3600}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray9b32-full89-trajnorm-fa}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-900}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
