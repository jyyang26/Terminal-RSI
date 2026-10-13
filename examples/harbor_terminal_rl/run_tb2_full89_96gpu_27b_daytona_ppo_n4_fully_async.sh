#!/bin/bash
#
# n=4 FORMAL PPO run-wrapper: Terminal-Bench 2 FULL 89 tasks on Daytona, 96-GPU 27B.
# Successor to the n=20 smoke (run_tb2_full89_96gpu_27b_daytona_ppo_lenreward_overfit_*).
# The smoke proved PPO trains end-to-end on 27B (critic learns: value_loss ~5.4;
# actor forward+train run; fd + TorchMemorySaver fixes hold). This is the proper
# PPO config that actually exploits the critic baseline.
#
# THREE deliberate changes vs the n=20 smoke (user-directed):
#
#   >>n=4 (was 20). PPO's advantage uses the critic's GAE value baseline computed
#       PER-TRAJECTORY (get_advantages_and_returns_batch), NOT a group-relative mean
#       like GRPO/GSPO. So multiple samples per prompt are redundant for advantage
#       estimation -- the critic IS the baseline. n=4 is kept (not n=1) only so the
#       data pipeline + group plumbing stay well-formed; the batch is rebuilt from
#       more distinct PROMPTS instead (see below). ~80% fewer sandboxes per group.
#
#   >>LENGTH REWARD OFF. CUSTOM_REWARD_POST_PROCESS_PATH is cleared -> reward is the
#       raw terminal binary checker (0/1), no length-shaped tie-break. PPO handles
#       sparse binary reward natively via the critic; the length bonus (which needed
#       multiple successful siblings per group to rank) is no longer meaningful at
#       n=4 and is dropped. The token-capping postprocess (ROLLOUT_ALL_SAMPLES_* /
#       ROLLOUT_DATA_POSTPROCESS_*) is NOT a reward and is KEPT.
#
#   >>OVERSAMPLING = GROUP-LEVEL ONLY. The yaml sets partial_group_min_units=4 (=n,
#       so NO within-group tail-cancel -- all 4 trials of a prompt are collected) and
#       max_concurrent_groups=24 > rollout_batch_size=64-window so slow groups are
#       overlapped by fast ones (the Daytona long-tail backstop). This keeps PPO's
#       effective batch intact while still hedging against slow sandboxes.
#
# BATCH (rebuilt for n=4): 64 prompts x n=4 = 256 trajectories/step (vs the smoke's
#   16 x 20 = 320). PPO doesn't care that samples come from 64 distinct prompts
#   rather than 16 -- more prompt diversity is strictly better for a value function.
#
# EPOCHS: 89 tasks / 64 prompts-per-step ~= 1.4 steps/epoch. NUM_ROLLOUT=7 ~= 5
#   epochs (matches the GSPO overfit run's 5-epoch budget for a clean comparison).
#   NOTE: with the 4x larger batch, 7 steps == ~5 epochs (NOT the smoke's 8 steps).
#   Going to 28 here would be ~20 epochs (overfit blow-up), hence 7.
#
# KEPT from the smoke: raw mode (bridge can't train 27B GDN), PPO/eps-clip 0.2,
# num-critic-only-steps=1 (step0 critic warmup), KL OFF, no rollout-logprobs,
# entropy OFF + display, override_timeout 3600s, periodic TB2 eval, NO step-0
# baseline eval, the nofile ulimit fix (base script), and the offload-required
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 (NOT expandable_segments).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tb2_full89_96gpu_27b_daytona_ppo_n4.yaml}"

# Separate save dir for the n=4 formal PPO run.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_ppo_n4}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tb2_full89_overfit_ppo_n4}"

# raw mode (bridge crashes 27B packed-seq GDN). actor+critic load LM backbone from
# the torch_dist ckpt; critic value head auto-reinitialized.
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/models/Qwen3.5-27B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60}"

# 12 nodes: 4 actor (32 GPU) + 8 rollout (64 GPU). critic shares the actor PG.
export EXPECTED_NODES="${EXPECTED_NODES:-12}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"

# 27B parallelism TP2*PP8*CP2=32.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

# >>n=4 BATCH: 64 prompts x 4 = 256 traj/step. over_sampling_batch_size=64 (= prompt
# count; the group-level oversample window). The within-group cutoff is OFF (yaml
# partial_group_min_units=4=n).
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# >>EPOCHS: 89/64 ~= 1.4 steps/epoch; 7 steps ~= 5 epochs (NOT 28 -> that'd be ~20
# epochs of overfit). Save every 20 -> only the final force_sync saves here.
export NUM_ROLLOUT="${NUM_ROLLOUT:-7}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

# Unified 64k context.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
# First knob to drop on OOM (second 27B critic). 32768 to start; halve to 16384.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"

export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# >>OFFLOAD-CRITICAL (PPO use_critic forces offload_train=True): torch_memory_saver
# rejects expandable_segments:True, so use a non-expandable alloc conf. (Same fix as
# the smoke; see that wrapper's comment for the full rationale.)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# 27B memory peak control. NO ROLLOUT LOGPROBS.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# >>PPO actor-critic + GAE. eps-clip 0.2 (standard per-token clip). critic warmup 1 step.
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ppo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-1}"

# ENTROPY LOSS OFF + entropy display.
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"

# KL OFF.
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# >>LENGTH REWARD OFF: clear the reward-shaping hook -> reward = raw terminal binary
# checker. The base script skips --custom-reward-post-process-path when this is empty.
# The token-capping postprocess (NOT a reward) is KEPT.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-full89-overfit-ppo-n4-noLenReward-noKL-nologprob-to3600-criticwarmup1-bs64x4-5ep}"

# >>EVAL: periodic TB2 (89 tasks) every 10 steps. With NUM_ROLLOUT=7 eval will NOT
# fire mid-run (first eval would be at step 10); set EVAL_INTERVAL lower to exercise
# it, or rely on the final state. NO step-0 baseline eval.
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2ppon4}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
