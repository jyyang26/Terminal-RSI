#!/bin/bash
#
# Run-wrapper for the Daytona oversampling + long-tail-kill SMOKE on 96-H20
# Qwen3.5-27B, training the WGDS-v3 Round9 valid359 dataset.
#
# This is a COPY of the proven 27B fully-async wrapper
#   run_full89_grpo_96gpu_27b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh
# with ONLY the experiment deltas changed (each marked >>EXP). Everything else
# — model, TP2*PP8*CP2 parallelism, checkpoints, 64k context, GRPO/KL/entropy
# recipe, trajectory_normalized postprocess — is IDENTICAL to the proven run so
# the only variables under test are: Daytona host, oversampling+cutoff, shorter
# per-trial budget, and the new dataset.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
# >>EXP: point at the new Daytona oversample smoke config.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_valid359_96gpu_27b_daytona_n20_partial16_max100_timeout2400_oversample_fully_async.yaml}"

# >>EXP: distinct run/save folders so this smoke never collides with the prod run.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_valid359_96gpu_daytona_n20_partial16_max100_timeout2400_oversample_smoke}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_valid359_96gpu_daytona_n20_partial16_max100_timeout2400_oversample_smoke}"

# 27B checkpoints — IDENTICAL to proven run. PP8 torch_dist matches PP=8.
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-27B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60}"

# 12 nodes total: 4 actor (32 GPU) + 8 rollout (64 GPU) — IDENTICAL.
export EXPECTED_NODES="${EXPECTED_NODES:-12}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"

# 27B actor parallelism TP2*PP8*CP2=32 — IDENTICAL (only layout that clears the
# 27B logprob/cross-entropy memory peak).
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

# >>EXP: oversampling + long-tail cutoff batch math.
#   n_samples=20 trials launched per group; partial cutoff (yaml=16) accepts the
#   first 16 trajectories and cancels the tail 4. rollout_batch=2 groups/step.
#   GLOBAL_BATCH_SIZE follows the PROVEN 9B-Daytona convention: set to the NOMINAL
#   full count rollout_batch*n_samples = 2*20 = 40, and allow_partial_global_batch
#   (auto-enabled by harbor_partial_group_min_units>0) absorbs the shortfall when
#   only 2*16=32 land. USE_DYNAMIC_GLOBAL_BATCH_SIZE stays ON (token-level packing),
#   matching the proven run.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-40}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# Smoke: short run, save early. 359 tasks, rollout_batch=2 => ~180 groups/epoch.
export NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

# Unified 64k context — IDENTICAL to proven run.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

# Actor train cap 64k — IDENTICAL.
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"

# 32 x TP2 SGLang engines across 64 rollout GPUs — IDENTICAL.
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# 27B memory peak control + GRPO recipe — IDENTICAL.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"

# Retrain v2 regularization — IDENTICAL.
export ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
export USE_KL_LOSS="${USE_KL_LOSS:-1}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# trajectory_normalized postprocess + CTRF progress reward — IDENTICAL.
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"

export USE_WANDB="${USE_WANDB:-1}"
# >>EXP: distinct W&B group encoding the experiment knobs.
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-valid359-96g-daytona-n20-partial16-max100-t2400-oversample-smoke}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray27b96daytona}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
