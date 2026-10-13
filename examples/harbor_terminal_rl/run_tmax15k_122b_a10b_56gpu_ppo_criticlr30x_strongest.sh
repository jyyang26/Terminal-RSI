#!/bin/bash
#
# tmax-15k + Qwen3.5-122B-A10B (MoE) + strongest PPO recipe, on the CURRENT 56-GPU
# cluster (7 nodes x 8 H20), STANDARD (skim-oversample) rollout, DAYTONA backend.
#
# Cloned from run_tmax15k_35b_a3b_56gpu_ppo_criticlr30x_strongest.sh. SAME RL RECIPE:
#   - PPO (advantage-estimator ppo), eps-clip 0.2, KL OFF, entropy OFF, no rollout-logprobs
#   - cold-start critic (value head reinit), NUM_CRITIC_ONLY_STEPS=2
#   - critic lr 30x / actor lr 2x via megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml
#   - standard mode, OVER_SAMPLING_BATCH_SIZE=260 skim window
#   - NUM_ROLLOUT=100, SAVE_INTERVAL=5, EVAL_INTERVAL=10, n=1 train / pass@3 eval
#   - 64k context on all walls, FT on
#
# DELTAS vs the 35B-A3B run (this is a 122B / ~10B-active model, 3.5x bigger):
#   - MODEL: Qwen3.5-122B-A10B (hidden 3072, 48 layers, moe/shared inter 1024).
#   - LOAD: BRIDGE mode. There is NO pre-converted 122B torch_dist, so the actor loads
#     its weights directly from the HF checkpoint via mbridge at startup, and the PPO
#     critic cold-starts (value head reinit) off the same HF backbone. Bridge requires
#     KL OFF (it is), and needs no --ref-load / torch_dist. HF weights read from the
#     shared fs (245GB; NOT staged to node-local /tmp).
#   - TOPOLOGY (sized so actor+critic fit in 95GB/GPU): 4 actor nodes (32 GPU) =
#     TP2 x PP2 x CP1 x EP16 x ETP1 => DP8. PP2 is REQUIRED: with fp32 grad accumulation
#     (--accumulate-allreduce-grads-in-fp32) each param costs 6B x 2 models; pure EP
#     leaves the ~6B non-routed params sharded only by TP2 (~28GB/GPU for both models
#     before any expert) -> OOM. PP2 halves the dense/non-routed block. World check:
#     TP2*PP2*CP1*DP8=32; TP*CP*DP=16 divisible by EP*ETP=16. 48 layers / PP2 = 24/stage.
#   - MAX_TOKENS_PER_GPU 16384 -> 8192 (35B OOM'd at 16384 in a tighter setup; 122B
#     hidden is 1.5x so activations/token are larger).
#   - ROLLOUT: 3 rollout nodes (24 GPU) = 3 independent TP8 SGLang engines (the eval-
#     proven config), NOT dp-attention+EP8 (which had single-request batching problems).
#
# CP MUST stay 1: GatedDeltaNet's recurrent scan is NOT context-parallel-safe.
#
# BEFORE LAUNCH export: DAYTONA_API_KEY, HARBOR_SRC, WANDB (logged in).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
export CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
export HARBOR_SRC="${HARBOR_SRC:-/path/to/workspace/harbor-code-synthesis/src}"

# --- ROLLOUT MODE: force standard --------------------------------------------------
export HARBOR_ROLLOUT_MODE="standard"

# --- MODEL: Qwen3.5-122B-A10B (RAW torch_dist load; cold-start critic) -------------
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-122B-A10B.sh}"
# RAW mode: load the pre-converted Megatron torch_dist (built by
# convert_qwen35_122b_a10b_torch_dist.sh). The RUNTIME bridge path (megatron.bridge)
# fails for this 122B model at PP>1 (AttributeError: NoneType has no megatron_module,
# unmapped output_layer), so we do NOT use bridge; instead the offline convert (which
# uses slime_plugins.mbridge with the output_layer->lm_head mapping) produces the
# torch_dist, and here we --ref-load / --load it. The torch_dist has no _critic
# sibling -> PPO critic cold-starts (value head reinit) off the same backbone.
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B_torch_dist}"
# 245GB HF + ~230GB torch_dist: read both from shared fs, do NOT rsync to node-local /tmp.
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"

export OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER:-1}"

# --- per-role lr: critic 30x / actor 2x — reuse the source YAML --------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml}"
export LR="${LR:-5e-7}"
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

# --- TOPOLOGY: 56 GPU = 4 actor nodes (32 GPU) + 3 rollout nodes (24 GPU) -----------
# 122B-A10B MoE: TP2 x PP2 x CP1 x EP16 x ETP1 => DP8. CP MUST be 1.
export EXPECTED_NODES="${EXPECTED_NODES:-7}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-2}"
export CP_SIZE="${CP_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-16}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"
# Rollout: 3 independent TP8 engines over 24 GPU (eval-proven; no dp-attention/EP).
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-mamba-scheduler-strategy extra_buffer --sglang-disable-custom-all-reduce}"
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type flex --moe-enable-deepep}"

# --- context walls: EVAL serves 128k (paper-style), TRAINING token budget stays 64k -
# SGLang serving context = 131072 so the eval agent can use a 128k window (proven OK on
# 122B TP8 in the standalone 131072 eval). Training stays memory-safe: SEQ_LENGTH and
# HARBOR_MAX_TRAIN_TOTAL_TOKENS stay 64k, and cap_total_tokens_allow_response_trim hard-
# trims any training sample >64k, so raising the served context does NOT blow up the
# actor+critic activation memory on the 32 GPU. Agent max_context_tokens (128000) is set
# in the yaml (shared train/eval agent config).
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-131072}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-131072}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
# 8k output limit (paper). Training samples still hard-capped to 64k total by postprocess.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-0}"
# actor + 122B critic colocated on 32 GPU. With USE_DYNAMIC_BATCH_SIZE=0 (static,
# micro-batch-size 1) this cap is IGNORED by the scheduler; kept for reference.
# Why dynamic is OFF: trajectory_retokenized emits a data-dependent chunk count per
# step (e.g. 268 for a 256-group step). At DP8 the dynamic packer made 268 all-
# singleton microbatches and could not split up to the required 272 (=ceil(268/8)*8),
# an UNSATISFIABLE assert in build_dp_schedule. The static path instead uses the
# designed alignment repair (drop zero-reward groups until sample_count % dp_size == 0)
# and, with mbs=1, has peak memory <= the dynamic path -> memory-safe here.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

# --- batch / n / oversample / schedule (per user: batch 256, n=1) ------------------
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-260}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
# Terminal-Bench leaderboard evaluation: task-level pass@3. Training rollout stays n=1.
# Paper (Qwen3.6 / Terminal-Bench 2.0) eval sampling: temp 1.0, top_p 0.95, top_k 20.
# n=3 per user (paper uses avg-over-5; kept at 3 here). avg@3 is the headline number.
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-3}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"

# --- PPO knobs (identical to source recipe) ----------------------------------------
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
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

# --- PPO offload alloc-conf --------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# --- fault tolerance + timeouts ----------------------------------------------------
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"

# --- data/timeout config yaml — 122B tmax-15k config with eval wall 10800 ----------
# (cloned from the 35B tmax config but harbor_eval_total_timeout_sec 3600 -> 10800 so
# the baseline pass@3 eval matches the standalone eval that got avg@3~0.43; the 3600s
# wall was cancelling ~14 slow tasks -> coverage 0.84, avg@3 dragged to 0.371.)
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tmax15k_56gpu_122b_a10b_daytona_ppo_criticlr30x_strongest.yaml}"

# --- run identity ------------------------------------------------------------------
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime-terminal-RL/checkpoints}"
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-122B-A10B_tmax15k_56gpu_ppo_criticlr30x_actorlr2x_strongest}"
export WANDB_GROUP="${WANDB_GROUP:-q35-122b-a10b-tmax15k-56g-ppo-criticlr30x-actorlr2x-standard-tp8-os260-64k-n1-bs256}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-terminal-rl}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k122b56}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

export HARBOR_DAYTONA_S3_PROXY_URL="${HARBOR_DAYTONA_S3_PROXY_URL:-http://star-proxy.oa.com:3128}"
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
