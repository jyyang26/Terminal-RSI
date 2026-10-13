#!/bin/bash
#
# PPO run-wrapper on the TerminalWorld by_length CURRICULUM (medium-hard 830 slice),
# 96-GPU 27B. This is the n=4 FORMAL PPO recipe that trained end-to-end on the TB2-89
# overfit (run_tb2_full89_96gpu_27b_daytona_ppo_n4_fully_async.sh), repointed from the
# overfit pool to the REAL by_length training curriculum.
#
# User directive: "run PPO, no length reward." So this keeps the overfit recipe's PPO
# (actor-critic + GAE) + LENGTH-REWARD-OFF exactly, and changes ONLY:
#   - DATA: --config -> the by_length mid-830 yaml (offset 700, limit 1000 => the 830
#       medium-hard tasks of the 1530-task ascending-difficulty curriculum). Eval pool
#       is genuinely-held-out TB2-89.
#   - NUM_ROLLOUT: a real multi-epoch budget (not the overfit's 7). 830/64 ~= 13
#       steps/epoch; 60 steps ~= 4.6 epochs. (Generous so the run keeps making
#       progress overnight; stop it manually or lower this for a fixed budget.)
#   - EVAL: step-0 held-out TB2 baseline ON (SKIP_EVAL_BEFORE_TRAIN=0) so there's a
#       clean starting anchor, then every 10 steps. (Overfit had it off since eval==train.)
#   - separate BASE_FOLDER / SAVE_SUBDIR / WANDB_GROUP / RAY_TEMP_DIR.
#
# EVERYTHING ELSE IS THE PROVEN PPO RECIPE, UNCHANGED: raw mode (bridge can't train
# 27B GDN), PPO/eps-clip 0.2, num-critic-only-steps=1 (step0 critic warmup), KL OFF,
# no rollout-logprobs, entropy OFF + display, the nofile ulimit fix (base script), and
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 (PPO offload_train forces
# TorchMemorySaver, which rejects expandable_segments:True). n=4 + group-level
# oversampling (96 concurrent sandboxes) per the yaml. See PPO_INTEGRATION_NOTES.md.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminalworld_mid830_96gpu_27b_daytona_ppo_n4_nolenreward.yaml}"

# Separate save dir for the by_length PPO run.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_terminalworld_mid830_ppo_n4_nolenreward}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_terminalworld_mid830_ppo_n4_nolenreward}"

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

# n=4 BATCH: 64 prompts x 4 = 256 traj/step. over_sampling_batch_size=64 (= prompt
# count; the group-level oversample window). within-group cutoff OFF (yaml min_units=4=n).
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# EPOCHS: 830 tasks / 64 prompts ~= 13 steps/epoch; 60 steps ~= 4.6 epochs. Generous
# so the run keeps progressing overnight. Save every 10 steps (force_sync at the end).
# SAVE_INTERVAL=10 (was 20): last run died at step3 < 20 and saved nothing; 10 banks
# the first actor checkpoint earlier. The critic-isolation fix
# (placement_group._isolate_critic_checkpoint_dirs) keeps the actor's iter dir from
# being overwritten by the critic's value head -> the saved ckpt is a real policy.
export NUM_ROLLOUT="${NUM_ROLLOUT:-60}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# Unified 64k context.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"

export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
# First knob to drop on real OOM (second 27B critic). 32768 to start; halve to 16384.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"

export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"

# OFFLOAD-CRITICAL (PPO use_critic forces offload_train=True): torch_memory_saver
# rejects expandable_segments:True, so use a non-expandable alloc conf.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# 27B memory peak control. NO ROLLOUT LOGPROBS.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# PPO actor-critic + GAE. eps-clip 0.2 (standard per-token clip). critic warmup 1 step.
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

# LENGTH REWARD OFF: clear the reward-shaping hook -> reward = raw terminal binary
# checker. The base script skips --custom-reward-post-process-path when this is empty.
# The token-capping postprocess (NOT a reward) is KEPT.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tw-mid830-ppo-n4-noLenReward-noKL-nologprob-to3600-criticwarmup1-bs64x4}"

# EVAL: held-out TB2 (89 tasks) every 10 steps + a step-0 baseline (TB2 is genuinely
# held out here, so the baseline is a meaningful starting anchor).
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtwppon4}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
