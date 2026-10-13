#!/bin/bash
# One-step SMOKE run to instrument loss.py shapes and explain logprob_diff=163.
# Loads the real rollout dump (rollout_dumps/2.pt) via --load-debug-rollout-data,
# skips SGLang, runs ONE training step with the shape-printing monkeypatch, then exits.
#
# Uses --debug-train-only + load-debug-rollout-data. num_rollout=1, no eval, no save.
# PYTHONPATH prepends /path/to/shared_storage/.../slime_debug_inject so every worker auto-imports
# usercustomize.py -> installs the loss instrumentation.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export SLIME_DIR="/path/to/workspace/slime-terminal-RL"
export HARBOR_SRC="/path/to/workspace/harbor-code-synthesis/src"

RUN=/path/to/workspace/slime_harbor_qwen35_35b_a3b_tb2_56gpu_ppo_dp1_mismatch_35bdp1mm_20260731_160647

# --- inject the shape-printing monkeypatch into every worker via usercustomize ---
export PYTHONPATH="/path/to/workspace/slime_debug_inject:${PYTHONPATH:-}"

# --- feed the real step-2 dump, skip SGLang, 1 step only ---
# NOTE: point directly at 2.pt (no {rollout_id} template) so EVERY step loads the same
# real actor-step dump; loop starts at rollout_id=0 and with NUM_CRITIC_ONLY_STEPS=0 the
# actor trains immediately -> pg_loss/entropy/tis/tis_abs/logprob_diff all computed.
# --load-debug-rollout-data alone triggers skip_sglang (arguments.py:1507); no engines.
export LOAD_DEBUG_ROLLOUT_DATA="${RUN}/rollout_dumps/2.pt"
export LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE="0.12"   # ~32 of 273 samples, fast, still reproduces
export NUM_ROLLOUT=1          # ONE step then exit
export SKIP_EVAL_BEFORE_TRAIN=1
export EVAL_INTERVAL=100000
export SAVE_INTERVAL=100000
export NUM_CRITIC_ONLY_STEPS=0   # actor trains at rollout_id=0 so pg_loss/entropy computed

# --- topology: SINGLE node 8-GPU actor (no worker -> no PYTHONPATH issues) ---
# 35B-A3B DP1 on 8 GPU: TP2 x PP4 x CP1 x EP2 -> DP1, 40/PP4=10 layers/stage (even).
export EXPECTED_NODES=1
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=8
export ROLLOUT_NUM_GPUS=8      # ignored under load-debug-rollout-data (skip sglang)
export TENSOR_MODEL_PARALLEL_SIZE=2
export PIPELINE_MODEL_PARALLEL_SIZE=4
export CP_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2
export EXPERT_TENSOR_PARALLEL_SIZE=1
export USE_DYNAMIC_BATCH_SIZE=0

# --- mismatch metrics ON (same as prod) so tis/tis_abs + logprob_diff all print ---
export USE_ROLLOUT_LOGPROBS=0
export GET_MISMATCH_METRICS=1
export CUSTOM_TIS_FUNCTION_PATH=slime_plugins.harbor.mismatch_metrics.record_only_tis_function
# deep loss instrumentation installs via mismatch_metrics.py import in the ACTOR process
export SLIME_DEBUG_LOGPROB_SHAPES=1
export SLIME_DEBUG_DIR="${SLIME_DEBUG_DIR:-/path/to/workspace/slime_dbg_out}"

# --- identity: isolated smoke dir, wandb OFF ---
export USE_WANDB=0
export BASE_FOLDER="/path/to/workspace/slime_smoke_logprob_dbg"
export SAVE_SUBDIR="smoke"
export RAY_TEMP_DIR="/tmp/rsmoke35b"
# avoid the ray dashboard MetricsHead startup timeout; run train directly (no job submit).
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-0}"
export DIRECT_RUN="${DIRECT_RUN:-1}"

# reuse the mismatch wrapper (it sets model/config/ckpt); it execs the base runner.
exec "${SCRIPT_DIR}/run_tmax15k_35b_a3b_56gpu_ppo_mismatch_metrics.sh"
