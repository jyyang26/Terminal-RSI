#!/bin/bash

set -euo pipefail

# Qwen3.5-35B-A3B (MoE, 256 experts top-8, hybrid GatedDeltaNet linear attention)
# Harbor Terminus2 RL on the CURRENT Ray cluster: 7 nodes x 8 H20 = 56 GPU,
# fully-async, trajectory_normalized, DAYTONA sandbox backend.
#
# Adapted from run_full89_grpo_96gpu_35b_a3b_..._fully_async.sh (96 GPU). Only the
# CLUSTER SIZING and the on-disk PATHS change; the MoE parallelism recipe (TP2 x PP1
# x CP1 x EP8, dp-attention rollout, DeepEP dispatch) and the RL hyper-parameters are
# kept identical to the proven 96-GPU wrapper. Everything then exec's the unchanged
# base script run_qwen35_9b_terminal_bench2_64h20.sh.
#
# DELTAS vs the 96-GPU 35B-A3B wrapper:
#   * nodes           : 7 (was 12). 4 actor (32 GPU) + 3 rollout (24 GPU).
#   * ROLLOUT_NUM_GPUS: 24 (was 64) => 3 SGLang engines of 8 GPU each.
#   * paths           : point at this checkout + the gy4 HF/torch_dist model, not
#                       the original <user> shared-storage/gy2 shares.
#   * env backend     : Daytona (needs DAYTONA_API_KEY) instead of local Singularity.
#
# CP MUST stay 1: the GatedDeltaNet linear-attention scan
# (slime_plugins/models/qwen3_5.py:Qwen3_5GatedDeltaNet) is a per-rank recurrent
# kernel and is NOT context-parallel-safe. EP8 spreads the 256 experts (32/GPU).
#
# BEFORE LAUNCH you must export (or set via the tmux launcher):
#   DAYTONA_API_KEY   - your Daytona key (rollout sandboxes).
#   HARBOR_SRC        - path to the harbor runtime source (harbor/src) on PYTHONPATH.
#   HARBOR_TASKS_DIR  - a terminal-bench task pool (dir of "*/task.toml").
#   HARBOR_TRIALS_DIR - scratch dir for trial outputs.
# The `harbor` / `terminus2` / `daytona_sdk` python packages must also be installed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# This checkout is both the slime source and the harbor plugin host.
export SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
export CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
# Harbor agent runtime (harbor/src) — NOT bundled in this checkout; set it yourself.
export HARBOR_SRC="${HARBOR_SRC:?set HARBOR_SRC=/path/to/harbor/src before launch}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_56gpu_35b_a3b_daytona_max120_actor4_rollout3_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.yaml}"

export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_35b_a3b_tb2_56gpu_actor4_rollout3_batch8_n32_fa_trajnorm_64k_timeout3600}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_56gpu_actor4_rollout3_batch8_n32_fa_trajnorm_64k_timeout3600}"

# 35B-A3B checkpoints on the gy4 share.
#
# PRETRAINED_CKPT is a Megatron torch_dist (DCP) checkpoint converted from the
# INSTRUCT HF weights (Qwen3.5-35B-A3B) with tools/convert_hf_to_torch_dist.py via
#   bash scripts/convert-qwen3.5-35B-A3B-gy4.sh          # -> Qwen3.5-35B-A3B_torch_dist
# DCP is resharding-agnostic on load (stores GLOBAL tensor shapes, re-split to the
# TARGET TP2/EP8 layout at --load / --ref-load time), so this dir works regardless
# of the PP8 used at convert time.
# NOTE: do NOT use Qwen3.5-35B-A3B_torch_dist_tp4_ep4 — that was converted from the
# -Base weights, not this Instruct model.
export HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist}"
# Stage the (large) checkpoints to node-local /tmp to avoid the slow shared-FS stall.
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/$(basename "${PRETRAINED_CKPT}")}"

# 7 nodes total: 4 actor (32 GPU) + 3 rollout (24 GPU).
export EXPECTED_NODES="${EXPECTED_NODES:-7}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"

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

export NUM_ROLLOUT="${NUM_ROLLOUT:-24}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

# Unified 64k context: rollout + SGLang + train cap all 64k.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

# Actor train capped at 64k. CP=1 => no per-rank sequence halving; 16384 is the
# conservative MoE starting point (first knob to raise if train-throughput-bound).
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"

# MoE rollout: one SGLang engine spans 8 rollout GPUs with dp-attention + EP8.
# 24 rollout GPUs => 3 engines. dp-size/ep-size 8 match the engine's 8 GPUs.
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
export ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-8}"
export ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-enable-dp-attention --sglang-dp-size ${ROLLOUT_DP_SIZE} --sglang-ep-size ${ROLLOUT_EP_SIZE} --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 --sglang-mamba-scheduler-strategy extra_buffer}"
# Train-side MoE dispatch: DeepEP all-to-all for 256 experts.
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type flex --moe-enable-deepep}"

# Memory-peak control + GRPO alignment recipe.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"

# Regularization: entropy 0.005, small KL-loss anchor to the reference model, low
# constant LR. kl-coef MUST stay 0 (arguments.py forbids kl_coef & kl_loss_coef both set).
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
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

# Daytona create-rate throttle (pace sandbox creates to stay under account limits).
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

export USE_WANDB="${USE_WANDB:-0}"
export WANDB_GROUP="${WANDB_GROUP:-q35-35b-a3b-tb2-56g-fa-trajnorm-ctx64k-train64k-a4r3-b8n32-daytona-ent005-kl001-lr5e7}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray35ba3b56}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
