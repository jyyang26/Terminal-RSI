#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# 89 tasks * 5 epochs = 445 task groups. With rollout_batch_size=4 and fresh
# torch_dist start_rollout_id=1, NUM_ROLLOUT=113 trains rollout ids 1..112,
# covering 448 groups.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_32gpu_max100_5epoch_last_turn.yaml}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_5epoch_resp2048_ctx128k}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-9B_terminal_bench2_full89_32gpu_max100_n16_5epoch_resp2048_ctx128k}"

export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-9B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-9B_torch_dist}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"

export EXPECTED_NODES="${EXPECTED_NODES:-4}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-113}"

export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
# Rollout context must cover prompt + max_new_tokens. max_step=100 terminal
# histories have exceeded 63k prompt tokens, so use 128k for generation while
# keeping actor train capped below.
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-131072}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"

export DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-131072}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-0}"
export CP_SIZE="${CP_SIZE:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-512}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-32768}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_rollout_data}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-harbor-terminal-rl}"
export WANDB_GROUP="${WANDB_GROUP:-qwen35-9b-tb2-full89-32gpu-max100-n16-5epoch-resp2048-ctx128k}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray9bf89m100}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
