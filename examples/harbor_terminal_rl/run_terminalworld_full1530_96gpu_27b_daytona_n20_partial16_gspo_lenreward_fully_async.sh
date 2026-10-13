#!/bin/bash
#
# FULL TRAINING run-wrapper: TerminalWorld 1530 tasks on Daytona, 96-GPU 27B.
# Variant of run_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_fully_async.sh
# with TWO algorithm changes on top of the proven binary-reward recipe:
#
#   >>CHG A: GSPO instead of GRPO.
#       --advantage-estimator gspo  (sequence-level, length-normalized importance
#       sampling + sequence-level clipping). Better suited to long terminal-agent
#       trajectories + fully-async off-policy gap + train/rollout precision drift.
#       REQUIRES a much tighter clip: GSPO's ratio is a length-normalized geometric
#       mean (~1, low variance), so EPS_CLIP=4e-4 (repo GSPO configs use 4e-4;
#       tests/test_gspo.sh uses 3.5e-4). The old GRPO 0.2 would never clip.
#       The custom no-entropy loss (train_loss.py) already implements the GSPO
#       branch with cross-CP all_gather of full log-probs (correct under CP2).
#
#   >>CHG B: length-aware reward shaping (success-only tie-breaker).
#       CUSTOM_REWARD_POST_PROCESS_PATH -> length_shaped_reward. Among the
#       SUCCESSFUL trajectories of a prompt group, shorter total output length ->
#       higher reward (bounded to +/- lambda/2 of the terminal reward); failures
#       stay exactly 0; correctness always dominates length. lambda=0.2.
#       This hook does its own GRPO/GSPO group normalization (it replaces slime's
#       default normalization, same as turn_grouped_reward / partial_reward).
#
# Everything else (parallelism, batch=256 traj/update, KL anchor, entropy off,
# Daytona infra + leak fixes) is IDENTICAL to the proven binary-reward run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
# Curriculum-learning config: data dir is .../unpacked/by_length (1530 symlinks
# ordered short->long by solve.sh line count), consumed in sorted order with no
# shuffle. Differs from the proven yaml only in harbor_tasks_dir + trials_dir.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_gspo_lenreward_curriculum.yaml}"

# Separate save dir so GSPO+length-reward checkpoints don't collide with the
# proven binary-reward run's ladder.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_terminalworld_full1530_daytona_n20_partial16_gspo_lenreward}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_terminalworld_full1530_daytona_n20_partial16_gspo_lenreward}"

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

# 16 tasks/step x 16 accepted = 256 traj/update — IDENTICAL.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-320}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# Real training run.
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

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

# 27B memory peak control — IDENTICAL.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# >>CHG A: GSPO + retuned clip. The launcher passes ADVANTAGE_ESTIMATOR/EPS_CLIP
# through to --advantage-estimator / --eps-clip (and --eps-clip-high iff set).
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-gspo}"
export EPS_CLIP="${EPS_CLIP:-4e-4}"
# EPS_CLIP_HIGH unset -> arguments.py defaults eps_clip_high = eps_clip (symmetric),
# matching the repo's GSPO configs. Set EPS_CLIP_HIGH to enable asymmetric clip-higher.

# ENTROPY LOSS OFF — IDENTICAL to proven run.
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"

# KL anchor kept — IDENTICAL.
export USE_KL_LOSS="${USE_KL_LOSS:-1}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# >>CHG B: length-aware reward shaping. The hook reads HARBOR_LENGTH_REWARD_LAMBDA
# (propagated to the Ray job env below if exported here) and does its own group
# normalization. lambda=0.2 keeps the length bonus a small tie-breaker (+/-0.1 of a
# unit terminal reward) so correctness always dominates. Set lambda=0 for a clean
# A/B pass-through equal to the binary reward. The two cap_total_tokens_* paths
# are TOKEN-CAPPING (not reward) and are KEPT.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.length_shaped_reward.trajectory_grouped_length_shaped_rewards}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.2}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-terminalworld-full1530-daytona-n20-p16-bs16-256traj-gspo-lenreward-l0.2-noentropy}"

# >>EVAL: periodic held-out eval on Terminal-Bench 2 (89 tasks). Fires every
# EVAL_INTERVAL rollout steps; SKIP_EVAL_BEFORE_TRAIN=0 also runs a step-0
# baseline. The eval task pool is set in the YAML (harbor_eval_tasks_dir/glob);
# eval uses the STANDARD rollout fn (full set, no oversample, no tail-cancel),
# independent of the fully_async TRAINING path. Each pass = 89 sandboxes
# (n_samples_per_eval_prompt=1), ~one train-step rollout of cost. Set
# EVAL_INTERVAL= (empty) to disable eval entirely.
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtw1530gspo}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
