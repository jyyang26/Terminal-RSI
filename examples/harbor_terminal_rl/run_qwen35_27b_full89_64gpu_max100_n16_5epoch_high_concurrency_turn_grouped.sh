#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# 89 tasks * 5 epochs = 445 task groups. With rollout_batch_size=16 and fresh
# torch_dist start_rollout_id=1, NUM_ROLLOUT=29 trains rollout ids 1..28,
# covering 448 groups.
#
# This wrapper uses turn_grouped samples: every Terminus LLM turn is trained as a
# slime sample, while group completeness and GRPO normalization stay at the
# trajectory level.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_turn_grouped.yaml}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_64gpu_max100_n16_5epoch_resp4096_ctx128k_actor4_tp2pp8cp2_thd_rb16_os32_conc96_pn24_traincap128k_lpc32_logp_turngrouped}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_terminal_bench2_full89_64gpu_max100_n16_5epoch_resp4096_ctx128k_actor4_tp2pp8cp2_thd_rb16_os32_conc96_pn24_traincap128k_lpc32_logp_turngrouped}"

export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-32}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-29}"

export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-131072}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-131072}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SEQ_LENGTH="${SEQ_LENGTH:-131072}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-131072}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-64g-turngrouped-logp-ctx128k-r4096-a4-tp2pp8cp2-rb16os32-c96p24}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray27b4turn}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_27b_full89_64gpu_max100_n16_5epoch.sh"
