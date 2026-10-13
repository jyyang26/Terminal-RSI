#!/bin/bash
#
# TINY validation run-wrapper: TerminalWorld self-contained-Dockerfile tasks on
# Daytona, 96-GPU 27B. Copy of the valid359 wrapper with ONLY the config + batch +
# folder names changed for an 8-task validation smoke. Goal: prove Daytona builds
# from per-task Dockerfile AND reward != 0 (unlike valid359, whose runtime-setup
# injection Daytona never ran).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminalworld_tiny8_96gpu_27b_daytona_smoke_fully_async.yaml}"

export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_terminalworld_tiny8_daytona_smoke}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_terminalworld_tiny8_daytona_smoke}"

# 27B checkpoints — IDENTICAL to proven run.
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-27B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60}"

# 12 nodes: 4 actor (32 GPU) + 8 rollout (64 GPU) — IDENTICAL.
export EXPECTED_NODES="${EXPECTED_NODES:-12}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"

# 27B parallelism TP2*PP8*CP2=32 — IDENTICAL.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

# >>TINY: n=8 samples/group, accept first 6 (yaml partial_group_min_units=6),
# 2 groups/step => global = 2*8 = 16 nominal; allow_partial absorbs the cutoff.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# >>TINY: very short — this is a validation run, not training.
export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

# Unified 64k context — IDENTICAL.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"

export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# GRPO recipe — IDENTICAL.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
export USE_KL_LOSS="${USE_KL_LOSS:-1}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-terminalworld-tiny8-daytona-smoke}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtw8}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
