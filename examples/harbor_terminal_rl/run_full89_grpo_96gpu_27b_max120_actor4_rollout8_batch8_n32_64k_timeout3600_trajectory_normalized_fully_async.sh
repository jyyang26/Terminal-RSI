#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_96gpu_27b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.yaml}"

export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_96gpu_actor4_rollout8_batch8_n32_fully_async_trajnorm_64k_timeout3600}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_full89_96gpu_actor4_rollout8_batch8_n32_fa_trajnorm_64k_timeout3600}"

# 27B checkpoints. PP8 torch_dist is the layout matching PIPELINE_MODEL_PARALLEL_SIZE=8.
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-27B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60}"
# Stage the (large) 27B checkpoints to node-local /tmp to avoid the slow shared-FS
# "Multi-thread loading shards" stall seen on 27B runs.
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60}"

# 12 nodes total: 4 actor (32 GPU) + 8 rollout (64 GPU).
export EXPECTED_NODES="${EXPECTED_NODES:-12}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"

# 27B actor parallelism: TP2 * PP8 * CP2 = 32 = actor GPUs. This is the only
# parallel layout that has cleared the 27B logprob/cross-entropy memory peak;
# TP1/PP8 OOM'd in fused_vocab_parallel_cross_entropy.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

# 8 task groups per update, 32 trajectories each. over_sampling == batch so the
# fully-async window is exactly the 8 active groups.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# 89 tasks, rollout_batch_size=8 => ~12 task groups/epoch. NUM_ROLLOUT=24 trains
# ~2 epochs (covers ~192 task groups). SAVE_INTERVAL=2 saves early+often so a
# rollout/host-RAM failure can't waste many steps (run96retok died at step 3 with
# 0 checkpoints under the old interval of 6).
export NUM_ROLLOUT="${NUM_ROLLOUT:-24}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

# Unified 64k context: rollout + SGLang + train cap all 64k. The agent's context
# budget (ROLLOUT_MAX_CONTEXT_LEN) now matches the train cap, so the trajectory
# rarely needs train-side trimming, and SGLang won't 400 on prompt+max_new_tokens
# overrunning a 128k window. Matches the tested 9B unified-64k recipe.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

# Actor train capped at 64k via Megatron seq-length + Harbor train-token cap, and
# token-level dynamic batching. With CP=2, max-tokens-per-gpu is multiplied by CP,
# so 32768 * 2 ~= 64k tokens/sample equivalent. Context is unified at 64k across
# rollout/SGLang/train, so the 27B step stays well off the 128k OOM cliff.
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"

# 32 x TP2 SGLang engines across the 64 rollout GPUs.
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# 27B memory peak control + GRPO alignment recipe (the proven 27B settings).
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"

# Retrain v2 regularization (see examples/harbor_terminal_rl/PERF_DEBUG.md Case 4).
# Counter the length/entropy collapse + repetition observed in run96retok2:
#   - entropy regularization to keep exploration / stop the repeat-until-4096 shortcut
#   - small KL-loss anchor to the reference model (REQUIRES --use-kl-loss, which the
#     base script adds when USE_KL_LOSS=1; that flag also triggers ref-model load)
#   - lower LR so the policy does not collapse to a shortcut so fast
# kl-coef MUST stay 0 (arguments.py asserts kl_coef and kl_loss_coef are not both set).
export ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
export USE_KL_LOSS="${USE_KL_LOSS:-1}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# trajectory_normalized postprocess: hard cap that may trim oldest response /
# observation suffix (keeps tokens/response_length/loss_mask/rollout_log_probs in
# sync) + trajectory-grouped progress (CTRF) reward.
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-96g-fa-trajretok-ctx64k-train64k-a4r8-b8n32-c256p32-t3600-v2-summ-ent01-kl001-lr5e7}"
# Ray AF_UNIX socket path kept short.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray27b96fa}"
# 27B stages large checkpoints before ray start; wait long enough for all 12 nodes.
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
