#!/bin/bash

set -euo pipefail

# Qwen3.5-35B-A3B (MoE, 256 experts top-8, hybrid GatedDeltaNet linear attention)
# Harbor Terminus2 RL on terminal-bench-2, 96 H20 GPU, fully-async,
# trajectory_retokenized. Adapted from the proven dense-27B v2r5 wrapper
# (run_full89_grpo_96gpu_27b_..._trajectory_normalized_fully_async.sh).
#
# DELTAS vs the 27B v2r5 recipe (everything else identical on purpose):
#   * model           : qwen3.5-35B-A3B.sh (MoE) instead of qwen3.5-27B.sh (dense)
#   * actor parallel  : TP2 x PP1 x CP1 x EP8 (DP16) = 32, vs 27B's TP2 x PP8 x CP2.
#                       CP MUST be 1 here: the GatedDeltaNet linear-attention scan
#                       (slime_plugins/models/qwen3_5.py:Qwen3_5GatedDeltaNet) is a
#                       per-rank recurrent kernel and is NOT context-parallel-safe.
#                       EP8 spreads the 256 experts so each GPU holds 32 experts,
#                       matching the validated SFT/coding-agent MoE layout.
#   * rollout engine  : MoE served with dp-attention + expert parallel
#                       (8 GPU / engine, dp-size 8, ep-size 8) + mamba scheduler,
#                       via EXTRA_SGLANG_ARGS. Dense 27B used plain TP2 engines.
#   * train MoE flags : --moe-token-dispatcher-type flex --moe-enable-deepep via
#                       EXTRA_MISC_ARGS (DeepEP all-to-all for 256 experts).
#   * use_rollout_logprobs : OFF (USE_ROLLOUT_LOGPROBS=0) per request. old_log_probs
#                       are recomputed train-side (train_loss.py handles both paths).
#   * entropy_coef    : 0.005 (was 0.01 on 27B v2r5; PERF_DEBUG.md Case 6 found 0.01
#                       slightly too strong / over-exploring late). entropy term is
#                       still ON (coef != 0 => with_entropy=True in train_loss.py).
#   * max-tokens/gpu  : 16384 (down from 27B's 32768). With CP=1 there is no
#                       sequence halving, so the vocab=248320 cross-entropy peak is
#                       higher per rank; start conservative. This is the FIRST knob
#                       to raise if rollout throughput is GPU-train-bound, or lower
#                       if the actor step OOMs.
#
# Kept identical: 89 tasks, rollout_batch_size 8, n_samples 32, NUM_ROLLOUT 24
# (~2 epochs), unified 64k context, KL-loss anchor (coef 1e-3), constant LR 5e-7,
# trajectory_retokenized + CTRF partial reward (max 0.4), summary ON.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_full89_96gpu_35b_a3b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.yaml}"

export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_35b_a3b_tb2_full89_96gpu_actor4_rollout8_batch8_n32_fully_async_trajnorm_64k_timeout3600}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_full89_96gpu_actor4_rollout8_batch8_n32_fa_trajnorm_64k_timeout3600}"

# 35B-A3B checkpoints. The torch_dist was produced by scripts/convert-qwen3.5-35B-A3B.sh
# (PP8 at conversion time, but torch_dist is resharding-capable: it loads fine into
# the PP1/EP8 train layout below).
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-35B-A3B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-35B-A3B_torch_dist}"
# Stage the (large) checkpoints to node-local /tmp to avoid the slow shared-FS
# "Multi-thread loading shards" stall.
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B_torch_dist}"

# 12 nodes total: 4 actor (32 GPU) + 8 rollout (64 GPU).
export EXPECTED_NODES="${EXPECTED_NODES:-12}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"

# 35B-A3B MoE actor parallelism: TP2 * PP1 * CP1 = 2; DP = 32/2 = 16; EP8 over the
# 256 experts (32 experts/GPU). (TP*CP*DP = 32) is divisible by (EP*ETP = 8). CP=1
# is mandatory for the GatedDeltaNet recurrent scan (not CP-safe).
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
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
# ~2 epochs. SAVE_INTERVAL=2 saves early+often.
export NUM_ROLLOUT="${NUM_ROLLOUT:-24}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

# Unified 64k context: rollout + SGLang + train cap all 64k.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

# Actor train capped at 64k. CP=1 => no per-rank sequence halving, so start with a
# smaller per-GPU token budget than the 27B/CP2 recipe (which effectively allowed
# 32768*2). 16384 is the conservative MoE starting point.
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"

# MoE rollout: one SGLang engine spans 8 rollout GPUs with dp-attention + EP8.
# 64 rollout GPUs => 8 engines. dp-size/ep-size 8 match the engine's 8 GPUs.
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
export ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-8}"
export ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
# MoE + hybrid-mamba serving flags (verified present in installed SGLang 0.5.10).
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-enable-dp-attention --sglang-dp-size ${ROLLOUT_DP_SIZE} --sglang-ep-size ${ROLLOUT_EP_SIZE} --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 --sglang-mamba-scheduler-strategy extra_buffer}"
# Train-side MoE dispatch: DeepEP all-to-all for 256 experts (matches coding-agent).
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type flex --moe-enable-deepep}"

# 27B memory peak control + GRPO alignment recipe.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
# use_rollout_logprobs OFF per request: old_log_probs recomputed train-side.
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"

# Regularization (PERF_DEBUG.md Case 4/6): entropy 0.005 (softer than 27B's 0.01),
# small KL-loss anchor to the reference model, low constant LR.
# kl-coef MUST stay 0 (arguments.py asserts kl_coef and kl_loss_coef not both set).
export ENTROPY_COEF="${ENTROPY_COEF:-0.005}"
export USE_KL_LOSS="${USE_KL_LOSS:-1}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# trajectory_normalized postprocess + trajectory-grouped progress (CTRF) reward.
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards}"
export HARBOR_PARTIAL_REWARD_MAX="${HARBOR_PARTIAL_REWARD_MAX:-0.4}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-35b-a3b-tb2-96g-fa-trajretok-ctx64k-train64k-a4r8-b8n32-c256p32-t3600-norollogprob-ent005-kl001-lr5e7}"
# Ray AF_UNIX socket path kept short.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray35ba3b96fa}"
# Stages large checkpoints before ray start; wait long enough for all 12 nodes.
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
