#!/bin/bash
#
# tmax-15k + Qwen3.5-35B-A3B (MoE) + strongest PPO recipe, DP1 topology, with
# train/inference log-prob MISMATCH METRICS logged to wandb (NO loss change).
#
# Thin wrapper over run_tmax15k_35b_a3b_56gpu_ppo_criticlr30x_strongest.sh. Inherits
# the entire RL recipe (PPO eps-clip 0.2, KL OFF, entropy OFF, cold-start critic,
# critic lr 30x / actor lr 2x, NUM_CRITIC_ONLY_STEPS=2, standard os260, 64k walls,
# tmax-15k Daytona config). ONLY the deltas below are overridden.
#
# WHY THIS RUN:
#   The 122B DP1 run OOM'd at the first actor-train step. This 35B-A3B run is the
#   smaller, memory-safe follow-up, and additionally logs how far the training-engine
#   log-probs drift from the SGLang inference log-probs — WITHOUT perturbing the loss.
#
# METRICS (surface as wandb train/* once actor training starts, i.e. step >= 2):
#   train/tis                         = mean exp(train_logp - rollout_logp)   (token IS ratio)
#   train/tis_abs                     = mean |exp(train_logp - rollout_logp) - 1|  (mismatch magnitude)
#   train/ois                         = mean exp(-ppo_kl)  (slime adds this; = current-vs-old-policy ratio)
#   train/train_rollout_logprob_abs_diff = mean |train_logp - rollout_logp|
# These are computed by loss.py when --get-mismatch-metrics is set. That flag REQUIRES
# a --custom-tis-function-path; we point it at a RECORD-ONLY tis fn that returns pg_loss
# UNCHANGED (no truncation), so the PPO recipe is bit-identical to a metrics-off run.
# We do NOT enable --use-rollout-logprobs: Harbor sets sample.rollout_log_probs
# unconditionally (trajectory_retokenized), so the metrics have data while the PPO ratio
# still uses the training-side recomputed old_log_probs.
#
# TOPOLOGY: DP1 (per user). 35B-A3B has 40 layers, hidden 2048, 256 experts, 2 KV heads.
#   DP=1 => TP*PP*CP = actor_gpus; CP MUST be 1 (GatedDeltaNet not CP-safe); TP<=2 (2 KV
#   heads); EP*ETP must divide TP*CP*DP=2 => EP<=2. 40 layers is NOT divisible by 16 (the
#   32-GPU PP16 option would need a 5x-imbalanced uneven split), so we use 2 ACTOR NODES:
#     16-GPU actor = TP2 x PP8 x CP1 x EP2 x ETP1 => DP1, 40/8 = 5 layers/stage (EVEN).
#   Theoretical actor-join peak ~66 GB / 96 GB (>=30 GB margin) -> memory-safe.
#   Remaining 40 GPU (5 nodes) => rollout: 5 independent TP8 SGLang engines.
#
# BEFORE LAUNCH export: DAYTONA_API_KEY, HARBOR_SRC, WANDB (logged in).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# --- MISMATCH METRICS (record-only; loss unchanged) --------------------------------
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"          # PPO ratio stays on train old_log_probs
export GET_MISMATCH_METRICS="${GET_MISMATCH_METRICS:-1}"         # trigger tis/tis_abs/ois logging
export CUSTOM_TIS_FUNCTION_PATH="${CUSTOM_TIS_FUNCTION_PATH:-slime_plugins.harbor.mismatch_metrics.record_only_tis_function}"

# --- DP1 TOPOLOGY: 16-GPU actor (2 nodes) + 40-GPU rollout (5 nodes) ----------------
export EXPECTED_NODES="${EXPECTED_NODES:-7}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-40}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-2}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
# DP1 (align_to=1) bypasses the dp_schedule alignment crash; keep the static packer.
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-0}"
# Rollout: 5 independent TP8 engines over 40 GPU (eval-proven; no dp-attention/EP).
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-mamba-scheduler-strategy extra_buffer --sglang-disable-custom-all-reduce}"
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type flex --moe-enable-deepep}"

# --- batch: rollout/global 256 (per user) ------------------------------------------
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

# --- eval: paper (Qwen3.6 / TB2) sampling, avg@3 -----------------------------------
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-3}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"

# --- eval TIMEOUT: dedicated config with harbor_eval_total_timeout_sec 10800 (gate OFF) --------
# The default strongest.yaml uses a 3600s TOTAL eval wall, which cancelled ~13 slow-but-nearly-
# done TB2 tasks at n=3 (coverage ~0.85, 13 timed_out_groups) and dragged avg@3 down to 0.288 vs
# the previous strongest run's 0.348. This config clones the *_tp8.yaml (which raised the wall to
# 10800s / 3h) but DISABLES the baseline eval gate — this run's purpose is the mismatch metrics,
# so a borderline baseline (pass@3 near the 0.35 gate edge) must not hard-RuntimeError the run.
export CONFIG="${CONFIG:-${SCRIPT_DIR}/harbor_tmax15k_56gpu_35b_a3b_daytona_ppo_dp1_mismatch_metrics.yaml}"

# --- run identity ------------------------------------------------------------------
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-35B-A3B_tmax15k_56gpu_ppo_dp1_mismatch_metrics}"
export WANDB_GROUP="${WANDB_GROUP:-q35-35b-a3b-tmax15k-56g-ppo-dp1-mismatch-metrics-n1-bs256}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k35bdp1}"

exec "${SCRIPT_DIR}/run_tmax15k_35b_a3b_56gpu_ppo_criticlr30x_strongest.sh"
