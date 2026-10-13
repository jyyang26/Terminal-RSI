#!/bin/bash
#
# RESTART run-wrapper: TerminalWorld MID-1000 (tasks 500-1499 of by_length) on
# Daytona, 96-GPU 27B. Based on the gspo_lenreward wrapper, with the 5 setting
# changes from the gspolen_20260616 post-mortem (DIAGNOSIS_gspolen_20260616_run.md):
#
#   >>1 DATA: skip the easiest 500 tasks; train on the MIDDLE 1000 (offset 500,
#       limit 1000) of the by_length curriculum. The previous run wasted ~72% of
#       early-step gradient on all-pass/all-fail (zero-signal) easy groups and
#       overfit a sloppy "2-turn task_complete" shortcut. Starting at medium
#       difficulty keeps more groups in the learnable (mixed-reward) band. NO
#       shuffle -- still an ascending-difficulty curriculum, just a harder start.
#
#   >>2 BRIDGE: load actor weights directly from the HF checkpoint via mbridge
#       (--megatron-to-hf-mode bridge). No local torch_dist --ref-load, no
#       torch_dist staging. (Base script run_qwen35_9b_terminal_bench2_64h20.sh
#       is bridge-aware: bridge => omit --ref-load, --load only when resuming.)
#
#   >>3 NO ROLLOUT LOGPROBS: USE_ROLLOUT_LOGPROBS=0 -- GSPO recomputes logprobs on
#       the train side instead of reusing rollout-side ones (drops the
#       rollout/train logprob mismatch noise source).
#
#   >>4 SAVE every 20 steps.
#
#   >>5 EPOCHS: 1000 tasks x 3 epochs / batch 16 ~= 186 rollout steps (NUM_ROLLOUT
#       set manually; Harbor's data source doesn't satisfy the rollout_global_dataset
#       assert that --num-epoch auto-calc requires).
#
#   KL is OFF (USE_KL_LOSS=0): bridge mode can't load a Megatron ref model for the
#   KL anchor, and the previous KL=0.001 was too weak to matter anyway. Drift is
#   instead controlled by the harder-start curriculum + multi-epoch + GSPO.
#
# KEPT from the gspo run: GSPO (eps_clip 4e-4), length-reward shaping (lambda 0.2),
# entropy OFF + entropy display, periodic TB2 eval (89 tasks, step-0 baseline),
# all parallelism / batch / Daytona infra + leak fixes.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CODE_ROOT="${CODE_ROOT:-/path/to/workspace/code}"

export CODE_ROOT
export SLIME_DIR="${SLIME_DIR:-${CODE_ROOT}/slime-v0.3-harbor}"
export HARBOR_SRC="${HARBOR_SRC:-${CODE_ROOT}/harbor/src}"
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"

export HARBOR_ROLLOUT_MODE="${HARBOR_ROLLOUT_MODE:-fully_async}"
# >>1 DATA: middle-1000 slice of the by_length curriculum. Slice (offset/limit) is
# set in the yaml below; data dir is unchanged (.../unpacked/by_length), still
# consumed in sorted (difficulty) order with no shuffle.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_terminalworld_mid1000_96gpu_27b_daytona_bridge_gspo_lenreward.yaml}"

# Separate save dir so this restart's checkpoints don't collide with prior runs.
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_qwen35_27b_terminalworld_mid1000_bridge_gspo_lenreward}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_terminalworld_mid1000_bridge_gspo_lenreward}"

# >>2 CKPT LOADING: bridge mode does NOT work for Qwen3.5 hybrid (SSM) weight
# sync in this repo — megatron.bridge names the SSM params self_attention.dt_bias
# (flat) while slime's megatron->HF converter (megatron_to_hf/qwen3_5.py) only
# knows self_attention.linear_attn.dt_bias (nested), so update_weights() to SGLang
# raises "Unknown parameter name". Reverted to the PROVEN raw (torch_dist) path
# used by the gspo run. Set MEGATRON_TO_HF_MODE=bridge only once that converter
# learns the bridge SSM naming. (KL is still OFF, so --ref-load here only seeds
# the actor's initial weights; no separate ref model is loaded.)
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

# 16 tasks/step x 16 accepted = 256 traj/update — IDENTICAL.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-320}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"

# >>5 EPOCHS: offset 700 -> 830 tasks / batch 16 = ~52 steps/epoch; x3 epochs ~= 156 steps.
export NUM_ROLLOUT="${NUM_ROLLOUT:-156}"
# >>4 SAVE every 20 steps.
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

# 27B memory peak control. >>3 NO ROLLOUT LOGPROBS.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
# Matches the proven gspo run (raw mode, per-sample-mean reduction). bridge mode
# would have REQUIRED this =1 (megatron.bridge's Qwen3VLModel asserts
# calculate_per_token_loss when CP_SIZE>1), but we reverted to raw, which has no
# such constraint — so keep the validated 0 to stay closest to the proven recipe.
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# GSPO + retuned clip — KEPT from the gspo run.
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-gspo}"
export EPS_CLIP="${EPS_CLIP:-4e-4}"

# ENTROPY LOSS OFF + entropy display — KEPT.
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"

# KL OFF — required by bridge mode (no Megatron ref model) and per the post-mortem
# (KL=0.001 was too weak to matter). KL_COEF kept 0 too.
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export LR="${LR:-5e-7}"

# length-aware reward shaping — KEPT (lambda 0.2).
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.length_shaped_reward.trajectory_grouped_length_shaped_rewards}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.2}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tw-mid1000-bridge-gspo-lenreward-l0.2-noKL-nologprob}"

# >>EVAL: periodic held-out eval on Terminal-Bench 2 (89 tasks) — KEPT. Every
# EVAL_INTERVAL rollout steps + a step-0 baseline (SKIP_EVAL_BEFORE_TRAIN=0).
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtwmid1000}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
