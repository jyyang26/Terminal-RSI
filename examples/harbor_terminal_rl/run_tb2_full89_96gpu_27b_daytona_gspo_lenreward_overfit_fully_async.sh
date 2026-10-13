#!/bin/bash
#
# OVERFIT run-wrapper: Terminal-Bench 2 FULL 89 tasks on Daytona, 96-GPU 27B.
# Goes back to overfitting the 89 TB2 tasks (train pool == eval pool). Keeps the
# proven mid830 recipe (raw mode, GSPO eps4e-4, length-reward lambda 0.4, KL OFF,
# no rollout-logprobs, entropy OFF + display, periodic TB2 eval) and only changes:
#
#   >>CHECKPOINT: raw (torch_dist) load via --ref-load. bridge mode was tried
#       2026-06-20 (run tb2of_bridge_20260620_030923): load + rollout +
#       distributed weight-sync all worked, but the first TRAINING step crashed
#       with `NotImplementedError: GDN does not support packed sequence for now.`
#       (megatron-core stock GatedDeltaNet rejects packed_seq_params/thd; the raw
#       --spec get_qwen3_5_spec path wires slime's own varlen Qwen3_5GatedDeltaNet
#       which handles it). See QWEN35_BRIDGE_DISTRIBUTED_FIX_REPORT.md. Until the
#       bridge provider is taught to use the varlen GDN, raw mode is the only path
#       that actually trains Qwen3.5.
#
#   >>EVAL: NO step-0 baseline eval (SKIP_EVAL_BEFORE_TRAIN=1); periodic TB2 eval
#       every EVAL_INTERVAL=10 rollout steps only.
#
#   >>DATA: train + eval both = the full 89-task Terminal-Bench 2 set (no
#       offset/limit). This is an OVERFIT run: eval pass-rate measures how well
#       the policy solves the very tasks it trains on.
#
#   >>EPOCHS: 89 tasks x 5 epochs / batch 16 = 27.8 ~= 28 rollout steps.
#       (NUM_ROLLOUT set manually; Harbor's data source doesn't satisfy the
#       rollout_global_dataset assert that --num-epoch auto-calc requires.)
#
#   >>TIMEOUT: override_timeout_sec 1800 -> 3600 (set in the yaml). The mid830
#       eval diagnosis found ~40-46% of eval trajectories were TRUNCATED =
#       Harbor agent hit the 30-min wall-clock timeout (NOT max_turns, NOT the
#       64k context wall); truncated trajectories passed ~0% vs ~60% for
#       completed ones. Doubling the per-trial budget to 3600s lets long tasks
#       finish. Applies to training AND eval (eval reuses the trial config).
#
#   >>EVAL BASELINE: DISABLED for this run (SKIP_EVAL_BEFORE_TRAIN=1). The step-0
#       baseline eval (train_async.py's `start_rollout_id==0 and not
#       skip_eval_before_train` branch) is skipped; eval runs only every
#       EVAL_INTERVAL=10 rollout steps. Saves the (slow, ~1h) pre-train eval.
#
# KEPT from the mid830 run: raw (torch_dist) checkpoint load, GSPO, length-reward
# shaping, entropy OFF, all parallelism / batch / Daytona infra + leak fixes.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tb2_full89_96gpu_27b_daytona_gspo_lenreward_overfit.yaml}"

# Separate save dir so this overfit run's checkpoints don't collide with prior runs.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_overfit_gspo_lenreward}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tb2_full89_overfit_gspo_lenreward}"

# raw (torch_dist) checkpoint load -- the PROVEN training path. bridge mode was
# tried 2026-06-20 (tb2of_bridge_20260620_030923): it loads + rolls out + syncs
# weights fine, but the first training forward crashes with
# `NotImplementedError: GDN does not support packed sequence for now.` because the
# bridge provider wires megatron-core's stock GatedDeltaNet (rejects packed/thd),
# whereas the raw `--spec get_qwen3_5_spec` path wires slime's varlen
# Qwen3_5GatedDeltaNet which handles it. See QWEN35_BRIDGE_DISTRIBUTED_FIX_REPORT.md.
# Stay on raw: actor loads the pre-converted torch_dist ckpt via --ref-load, and
# the raw converter matches its param naming.
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
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

# 16 tasks/step x 16 accepted = 256 traj/update — IDENTICAL to mid830.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-320}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# >>EPOCHS: 89 tasks x 5 epochs / batch 16 = 27.8 ~= 28 steps.
export NUM_ROLLOUT="${NUM_ROLLOUT:-28}"
# Save every 20 steps (kept from mid830). With 28 total steps this saves at
# step 20 and at the final step (force_sync on last).
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

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

# 27B memory peak control. NO ROLLOUT LOGPROBS.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
# calculate_per_token_loss=0: per-sample-mean loss normalization, as PROVEN by
# mid830. (It only needs to be 1 under bridge mode with CP>1, where the bridge
# provider asserts it; raw mode has no such constraint, so keep mid830 semantics.)
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# GSPO + retuned clip — KEPT from the mid830 run.
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-gspo}"
export EPS_CLIP="${EPS_CLIP:-4e-4}"

# ENTROPY LOSS OFF + entropy display — KEPT.
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"

# KL OFF — per the post-mortem (KL=0.001 was too weak to matter).
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# length-aware reward shaping — lambda RAISED 0.2 -> 0.4 (stronger short-output
# bias). The bonus is bounded +/- lambda/2 (=+/-0.2) of the terminal reward and
# applies only among the SUCCESSFUL trajectories of a group, so it stays a
# tie-breaker (lambda<2 => longest success terminal*(1-0.2)=0.8>0 can never flip
# below a failure=0). Pushes the policy toward shorter/more-efficient solutions,
# which also reduces the chance of hitting the 3600s wall-clock timeout.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.length_shaped_reward.trajectory_grouped_length_shaped_rewards}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.4}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tb2-full89-overfit-gspo-lenreward-l0.4-noKL-nologprob-to3600-noBaselineEval}"

# >>EVAL: periodic held-out eval on the same Terminal-Bench 2 (89 tasks) — KEPT.
# Every EVAL_INTERVAL=10 rollout steps. NO step-0 baseline eval
# (SKIP_EVAL_BEFORE_TRAIN=1) — skip the slow ~1h pre-train eval.
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtb2of}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
