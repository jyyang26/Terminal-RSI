#!/bin/bash

set -euo pipefail

# Qwen3.5-35B-A3B (MoE, 256 experts top-8, hybrid GatedDeltaNet linear attention)
# Harbor Terminus2 RL on the CURRENT Ray cluster: 7 nodes x 8 H20 = 56 GPU,
# fully-async, trajectory_normalized, DAYTONA sandbox backend. PPO (actor-critic + GAE).
#
# This is the PPO sibling of
#   run_35b_a3b_56gpu_daytona_actor4_rollout3_batch8_n32_..._fully_async.sh   (GRPO).
# The MoE parallelism recipe (TP2 x PP1 x CP1 x EP8, dp-attention rollout, DeepEP
# dispatch) and the cluster sizing (4 actor / 32 GPU + 3 rollout / 24 GPU) are IDENTICAL
# to the GRPO wrapper. Only the RL ALGORITHM block changes, grafting the proven 27B-PPO
# knobs (run_tb2_full89_96gpu_27b_daytona_ppo_n4_fully_async.sh) onto the 35B-A3B recipe.
#
# PPO vs GRPO deltas (all algorithm; sizing/paths unchanged):
#   * ADVANTAGE_ESTIMATOR = ppo  => arguments.py derives use_critic=True and allocates
#     an independent critic train-group that SHARES the actor's placement group
#     (slime/ray/placement_group.py: result["critic"] = result["actor"], NO extra GPUs).
#     The critic LM backbone loads from the same torch_dist as the actor; its value
#     head (output_size=1) is auto-reinitialized on shape-mismatch.
#   * n=4 (was 32). PPO uses the critic's PER-TRAJECTORY GAE baseline, so group-relative
#     multi-sampling is redundant. The batch is rebuilt from more distinct PROMPTS
#     (ROLLOUT_BATCH_SIZE 64) instead => 64 prompts x 4 = 256 traj/step (same 256).
#   * EPS_CLIP 0.2 (standard PPO per-token clip); NUM_CRITIC_ONLY_STEPS=1 (step-0 critic
#     warmup before the actor starts training).
#   * KL OFF + ENTROPY OFF (the critic baseline handles sparse binary reward natively;
#     GRPO's entropy/KL regularizers are dropped for PPO).
#   * PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 -- PPO (use_critic) forces
#     offload_train=True, and torch_memory_saver rejects expandable_segments:True.
#   * per-role critic LR 10x the actor (MEGATRON_CONFIG_PATH -> megatron_roles_35b_a3b_criticlr10x.yaml).
#   * reward = raw terminal binary checker (CUSTOM_REWARD_POST_PROCESS_PATH cleared);
#     the token-capping postprocess (NOT a reward) is KEPT.
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
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminal_bench2_56gpu_35b_a3b_daytona_max120_ppo_n4_64k_timeout3600_trajectory_normalized_fully_async.yaml}"

export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_35b_a3b_tb2_56gpu_ppo_n4_fa_trajnorm_64k_timeout3600}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_56gpu_ppo_n4_fa_trajnorm_64k_timeout3600}"

# 35B-A3B checkpoints on the gy4 share.
#
# PRETRAINED_CKPT is a Megatron torch_dist (DCP) checkpoint converted from the
# INSTRUCT HF weights (Qwen3.5-35B-A3B) with tools/convert_hf_to_torch_dist.py via
#   bash scripts/convert-qwen3.5-35B-A3B-gy4.sh          # -> Qwen3.5-35B-A3B_torch_dist
# DCP is resharding-agnostic on load (stores GLOBAL tensor shapes, re-split to the
# TARGET TP2/EP8 layout at --load / --ref-load time), so this dir works regardless
# of the PP8 used at convert time. Both the actor and the PPO critic load their LM
# backbone from here.
# NOTE: do NOT use Qwen3.5-35B-A3B_torch_dist_tp4_ep4 — that was converted from the
# -Base weights, not this Instruct model.
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist}"
# Stage the (large) checkpoints to node-local /tmp to avoid the slow shared-FS stall.
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/$(basename "${PRETRAINED_CKPT}")}"

# 7 nodes total: 4 actor (32 GPU) + 3 rollout (24 GPU). The PPO critic shares the
# actor PG, so it uses no additional GPUs.
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

# >>PPO n=4 BATCH: 64 prompts x 4 = 256 traj/step (same effective batch as the GRPO
# sibling's 8x32). over_sampling_batch_size = ROLLOUT_BATCH_SIZE (the group-level
# oversample window); the within-group cutoff is OFF (yaml partial_group_min_units=4=n).
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
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

# Actor train capped at 64k. CP=1 => no per-rank sequence halving.
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
# PPO adds a SECOND 35B-A3B model (the critic) on the same 32 GPUs, so the memory
# peak is higher than GRPO. 16384 is the conservative MoE start; FIRST knob to halve
# to 8192 on OOM.
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

# >>PPO OFFLOAD-CRITICAL: use_critic forces offload_train=True; torch_memory_saver
# rejects expandable_segments:True, so use a non-expandable alloc conf.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# Memory-peak control. NO ROLLOUT LOGPROBS.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# >>PPO actor-critic + GAE. eps-clip 0.2 (standard per-token clip). critic warmup 1 step.
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ppo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-1}"
# Per-role critic LR 10x the actor's (5e-6 vs 5e-7).
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_35b_a3b_criticlr10x.yaml}"

# >>ENTROPY OFF + entropy display (PPO relies on the critic baseline, not entropy reg).
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"

# >>KL OFF (PPO handles sparse binary reward via the critic; no reference anchor).
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# trajectory_normalized postprocess (token capping — NOT a reward — KEPT).
# >>LENGTH/PARTIAL REWARD OFF: reward = raw terminal binary checker. The base script
# skips --custom-reward-post-process-path when CUSTOM_REWARD_POST_PROCESS_PATH is empty.
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"

# Daytona create-rate throttle (pace sandbox creates to stay under account limits).
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

export USE_WANDB="${USE_WANDB:-0}"
export WANDB_GROUP="${WANDB_GROUP:-q35-35b-a3b-tb2-56g-fa-trajnorm-ctx64k-train64k-ppo-n4-bs64-criticwarmup1-noKL-noEnt-lr5e7-criticlr5e6-daytona}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray35ba3b56ppo}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
