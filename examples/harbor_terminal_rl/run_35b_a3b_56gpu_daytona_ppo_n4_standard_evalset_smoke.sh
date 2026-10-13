#!/bin/bash

set -euo pipefail

# ============================================================================
# Qwen3.5-35B-A3B (MoE) Harbor Terminal-RL — PPO, STANDARD (batch-wise) rollout,
# DAYTONA backend, on the CURRENT 56-GPU Ray cluster (7 nodes x 8 H20).
#
# SMOKE: train pool == eval pool == terminal-bench-2 (89 tasks). This is a bring-up
# run to validate the full pipeline (ckpt load, critic reinit, Daytona sandboxes,
# SGLang rollout, wandb) before wiring in the real 2820-task training set.
#
# GPU split (user request "32 train, rest inference"):
#   * 4 actor nodes = 32 GPU  (TP2 x PP1 x CP1 x EP8, DP16). PPO critic shares these.
#   * 3 rollout nodes = 24 GPU (3 SGLang engines, dp-attention + EP8).
#
# All paths are HARDCODED for this environment. Launch on node0 (see companion
# start_*_tmux.sh), or run this wrapper directly under a ray job.
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
export CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
# Harbor Terminus2 agent runtime (Python) — the harbor-code-synthesis src tree.
export HARBOR_SRC="${HARBOR_SRC:-/path/to/workspace/harbor-code-synthesis/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh}"

# >>STANDARD (batch-wise) rollout mode — NOT fully_async.
export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-standard}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_56gpu_35b_a3b_daytona_ppo_n4_standard_evalset_smoke.yaml}"

# Save results under slime-terminal-RL/checkpoints (user request).
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime-terminal-RL/checkpoints}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_56gpu_ppo_n4_standard_evalset_smoke}"

# actor=critic-backbone=ref all load from this one torch_dist (converted from Instruct HF).
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/$(basename "${PRETRAINED_CKPT}")}"

# 7 nodes: 2 actor (16 GPU) + 5 rollout (40 GPU). Critic shares the actor PG.
export EXPECTED_NODES="${EXPECTED_NODES:-7}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-40}"

# 35B-A3B MoE actor parallelism. CP MUST be 1 (GatedDeltaNet recurrent scan not CP-safe).
# 16 actor GPU: TP2 x PP1 x CP1 = 2 => DP = 16/2 = 8; EP8 over 256 experts (32/GPU).
# TP*CP*DP = 16 divisible by EP*ETP = 8. ✅
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

# >>PPO n=4 BATCH: 64 prompts x 4 = 256 traj/step. over_sampling = rollout_batch_size
# (group-level oversample window; within-group cutoff OFF via yaml min_units=4=n).
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# Short smoke run; save every 2 steps.
export NUM_ROLLOUT="${NUM_ROLLOUT:-8}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

# Unified 64k context.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
# PPO adds a 2nd 35B-A3B (critic) on the same 32 GPUs. 16384 conservative; halve to 8192 on OOM.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"

# MoE rollout: 1 SGLang engine per 8-GPU node => 3 engines over 24 GPU.
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
export ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-8}"
export ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-enable-dp-attention --sglang-dp-size ${ROLLOUT_DP_SIZE} --sglang-ep-size ${ROLLOUT_EP_SIZE} --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 --sglang-mamba-scheduler-strategy extra_buffer}"
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type flex --moe-enable-deepep}"

# >>PPO OFFLOAD-CRITICAL: use_critic forces offload_train=True; torch_memory_saver
# rejects expandable_segments:True.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# >>PPO actor-critic + GAE.
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ppo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-1}"
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_35b_a3b_criticlr10x.yaml}"

# ENTROPY OFF + KL OFF (PPO relies on the critic baseline).
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# token-capping postprocess KEPT; reward = raw terminal binary checker (no shaping).
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"

# Daytona create-rate throttle.
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

# >>Held-out eval on the 89 tasks EVERY rollout step (eval_interval=1), PLUS a
# baseline eval BEFORE training starts (skip_eval_before_train=0).
export EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"

# >>WANDB — project name per user request.
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-terminal-rl}"
export WANDB_GROUP="${WANDB_GROUP:-q35-35b-a3b-tb2-56g-standard-ppo-n4-evalset-smoke}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray35ba3b56ppostd}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
