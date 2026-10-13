#!/bin/bash
#
# tmax-15k + Qwen3.5-27B + strongest PPO recipe, on the CURRENT 56-GPU cluster
# (7 nodes x 8 H20), STANDARD (skim-oversample) rollout, DAYTONA backend.
#
# This is the 56-GPU port of run_tmax15k_qwen35_120gpu_ppo_criticlr30x_strongest.sh.
# The RL RECIPE is kept IDENTICAL to that "strongest run":
#   - PPO (advantage-estimator ppo), eps-clip 0.2, KL OFF, entropy OFF, no rollout-logprobs
#   - cold-start critic (raw torch_dist load, value head reinit), NUM_CRITIC_ONLY_STEPS=2
#   - critic lr 30x (1.5e-5) / actor lr 2x (1.0e-6) via megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml
#   - standard mode, OVER_SAMPLING_BATCH_SIZE=260 skim window
#   - NUM_ROLLOUT=100, SAVE_INTERVAL=5, EVAL_INTERVAL=10
#   - 64k context on all 5 walls, raw ckpt load, offload alloc-conf, FT on
#
# DELTAS vs the 120-GPU strongest run (per user):
#   - ROLLOUT_BATCH_SIZE 512 -> 256 (n_samples_per_prompt=1 unchanged)
#   - TOPOLOGY reset to 56 GPU: 4 actor nodes (32 GPU, TP2 x PP8 x CP2) + 3 rollout
#     nodes (24 GPU). (source was 15 nodes / 120 GPU, PP16 x CP2 x TP2.)
#   - DATA: shared-fs converted tmax-15k (14601 tasks), NO node-local /tmp tar staging.
#   - PATHS: gy4 model + this checkout.
#
# 27B is DENSE (64 layers). 32 actor GPU => TP2 x PP8 x CP2 = 32 (DP1), 8 layers/stage.
#
# BEFORE LAUNCH export: DAYTONA_API_KEY, HARBOR_SRC, WANDB (logged in).
# NOTE: PRETRAINED_CKPT (Qwen3.5-27B_torch_dist) does NOT exist yet — convert it first
#       (HF -> torch_dist) or the raw-mode --ref-load/--load check will fail.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
export CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
export HARBOR_SRC="${HARBOR_SRC:-/path/to/workspace/harbor-code-synthesis/src}"

# --- ROLLOUT MODE: force standard (starter/base default to fully_async; must override) -
export HARBOR_ROLLOUT_MODE="standard"

# --- MODEL: Qwen3.5-27B (raw torch_dist load; cold-start critic from base, no _critic) -
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-27B.sh}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-27B}"
# torch_dist to be produced (HF -> torch_dist) before launch; no _critic sibling keeps
# the critic COLD START (value head reinit).
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-27B_torch_dist}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-1}"
export STAGED_HF_CKPT="${STAGED_HF_CKPT:-/tmp/slime-harbor-model-cache/Qwen3.5-27B}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-1}"
export STAGED_TORCH_DIST_CKPT="${STAGED_TORCH_DIST_CKPT:-/tmp/slime-harbor-model-cache/$(basename "${PRETRAINED_CKPT}")}"

export OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER:-1}"

# --- per-role lr: critic 30x (1.5e-5) / actor 2x (1.0e-6) — reuse the source YAML ------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml}"
export LR="${LR:-5e-7}"
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

# --- TOPOLOGY: 56 GPU = 4 actor nodes (32 GPU) + 3 rollout nodes (24 GPU) --------------
# 27B dense: TP2 x PP8 x CP2 = 32 (DP1). (source 120-GPU used PP16 x CP2 x TP2.)
export EXPECTED_NODES="${EXPECTED_NODES:-7}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-2}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"
# 3 rollout nodes / 24 GPU: 27B dense served TP2 per engine => 12 engines.
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"

# --- 5 context walls + train-side length, all 64k (proven-safe window) ----------------
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-65536}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
# 32 actor GPU (vs source 64): halve the per-GPU token budget from 8192 to a safe 8192
# stays fine at PP8xCP2 (per-rank seq halved by CP2). Drop to 4096 on OOM.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

# --- batch / n / oversample / schedule (per user: batch 256, n=1) ---------------------
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-260}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"

# --- PPO knobs (identical to source recipe) -------------------------------------------
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ppo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
# length reward OFF (n=1): clear the reward-shaping hook -> raw terminal binary reward.
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

# --- PPO offload alloc-conf (torch_memory_saver rejects expandable_segments) ----------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# --- fault tolerance + timeouts (source values) ---------------------------------------
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"

# --- data/timeout config yaml (56-GPU tmax-15k, shared-fs data) ------------------------
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tmax15k_56gpu_27b_daytona_ppo_criticlr30x_strongest.yaml}"

# --- run identity ---------------------------------------------------------------------
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime-terminal-RL/checkpoints}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-27B_tmax15k_56gpu_ppo_criticlr30x_actorlr2x_strongest}"
export WANDB_GROUP="${WANDB_GROUP:-q35-27b-tmax15k-56g-ppo-criticlr30x-actorlr2x-standard-os260-64k-n1-bs256}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-terminal-rl}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k27b56}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

# Daytona build-context S3 proxy + create rate limit (source values).
export HARBOR_DAYTONA_S3_PROXY_URL="${HARBOR_DAYTONA_S3_PROXY_URL:-http://star-proxy.oa.com:3128}"
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
