#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Approximate 2 epochs without dropping the 89-task tail:
# 89 tasks * 2 = 178 task groups. With rollout_batch_size=16, slime's
# --num-epoch path would floor to 10 rollouts. We use 12 rollouts instead,
# covering 192 task groups, i.e. two full passes plus 14 rollover tasks.
export CONFIG=${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_gy2_runtime_sif_trajectory.yaml}
export BASE_FOLDER=${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_grpo_n16_2epoch_ctx128k_trajectory}
export SAVE_SUBDIR=${SAVE_SUBDIR:-Qwen3.5-9B_terminal_bench2_full89_grpo_n16_2epoch_ctx128k_trajectory}

export HF_CKPT=${HF_CKPT:-/path/to/models/Qwen3.5-9B}
export PRETRAINED_CKPT=${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-9B_torch_dist}
export STAGE_HF_CKPT_TO_TMP=${STAGE_HF_CKPT_TO_TMP:-1}

# Default to a 64-GPU layout with 32 GPUs for actor training and 32 GPUs for rollout.
export ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4}
export ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-32}
export EXPECTED_NODES=${EXPECTED_NODES:-8}

export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
export NUM_ROLLOUT=${NUM_ROLLOUT:-12}

export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8096}
export ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-131072}
export SEQ_LENGTH=${SEQ_LENGTH:-131072}

export SAVE_INTERVAL=${SAVE_INTERVAL:-4}
export DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.50}
export SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-131072}
export SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-131072}
export SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-512}
export HARBOR_MAX_TRAIN_TOTAL_TOKENS=${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-131072}
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH=${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_samples}
export ROLLOUT_DATA_POSTPROCESS_PATH=${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_rollout_data}
export USE_CUSTOM_NO_ENTROPY_LOSS=${USE_CUSTOM_NO_ENTROPY_LOSS:-1}
export KEEP_OLD_ACTOR=${KEEP_OLD_ACTOR:-1}
export CALCULATE_PER_TOKEN_LOSS=${CALCULATE_PER_TOKEN_LOSS:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export USE_WANDB=${USE_WANDB:-1}
export WANDB_PROJECT=${WANDB_PROJECT:-slime-harbor-terminal-rl}
export WANDB_GROUP=${WANDB_GROUP:-qwen35-9b-tb2-full89-grpo-n16-2epoch-ctx128k-trajectory}
export WANDB_MODE=${WANDB_MODE:-online}
# Match the existing Qwen3.5-9B_torch_dist checkpoint layout. Running TP=2/PP=1
# forces Megatron to reshard the PP=8 checkpoint at startup and can stall before
# rollout begins.
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
export PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-8}
export SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-0}
export CP_SIZE=${CP_SIZE:-4}
export QKV_FORMAT=${QKV_FORMAT:-thd}

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
